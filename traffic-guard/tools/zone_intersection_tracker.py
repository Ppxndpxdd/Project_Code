import os
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import cv2
import numpy as np
import torch
from .detection_entry import DetectionEntry
from .mqtt_publisher import MqttPublisher
from ultralytics import YOLO

class ZoneIntersectionTracker:
    """Tracks objects in a video and detects intersections with defined zones."""

    def __init__(self, config: Dict[str, Any], show_result: bool = True):
        # Configuration
        self.config = config
        model_path = config['model_path']
        mask_json_path = config['mask_json_path']
        tracker_config = config.get('tracker_config', 'bytetrack.yaml')
        self.show_result = show_result

        # Handle model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        # Handle mask_json_path
        if not os.path.exists(mask_json_path):
            raise FileNotFoundError(f"Mask positions file '{mask_json_path}' not found.")
        with open(mask_json_path, 'r') as f:
            self.mask_positions = json.load(f)
        self.zones = {}
        self.detection_log = []
        self.tracked_objects = {}
        self.tracker_config = tracker_config
        self.object_zone_timers = defaultdict(lambda: defaultdict(float))
        self.fps = 30  # Default FPS, update this in track_intersections method

        # Set output directory and file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, config.get('output_dir', 'output'))
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, config.get('output_file', 'detection_log.json'))

        # Set device
        if torch.cuda.is_available():
            self.device = 'cuda'
            logging.info("Using GPU for inference.")
        else:
            self.device = 'cpu'
            logging.info("Using CPU for inference.")

        # Load the model with the specified device
        self.model = YOLO(model_path).to(self.device)

        # Load total duration and compute time thresholds
        self.total_duration = config.get('total_duration', 5)  # Default to 5 seconds if not specified
        # Calculate time thresholds for three colors
        self.time_thresholds = [
            2 * self.total_duration / 3,   # Threshold for green to orange
            self.total_duration            # Threshold for orange to red
        ]

        # Initialize MQTT publisher
        self.mqtt_publisher = MqttPublisher(config)

        # Load event-specific configurations
        self.no_entry_zones = config.get('no_entry_zones', [])
        self.no_parking_duration = config.get('no_parking_duration', 60)  # in seconds

    def load_zones_for_frame(self, frame_id: int):
        """Loads zones for the specified frame."""
        self.zones.clear()
        frame_data = [entry for entry in self.mask_positions if entry['frame'] == frame_id]
        for entry in frame_data:
            self.zones[entry['zone_id']] = np.array(entry['points'])

    def calculate_iou(self, bbox: np.ndarray, polygon: np.ndarray) -> float:
        """Calculates the Intersection over Union (IoU) between a bounding box and a polygon."""
        x1, y1, x2, y2 = bbox
        bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # Calculate intersection
        intersection = cv2.intersectConvexConvex(bbox_poly, polygon.astype(np.float32))
        if intersection[1] is None:
            return 0.0

        intersection_area = cv2.contourArea(intersection[1])

        # Calculate union
        bbox_area = (x2 - x1) * (y2 - y1)
        polygon_area = cv2.contourArea(polygon)
        union_area = bbox_area + polygon_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    def intersects(self, bbox: np.ndarray, polygon: np.ndarray) -> Tuple[bool, float]:
        """Determines if a bounding box intersects with a polygon."""
        iou = self.calculate_iou(bbox, polygon)
        return iou > self.config.get('iou_threshold', 0.1), iou

    def draw_zones(self, frame: np.ndarray):
        """Draws zones on the frame."""
        for zone_id, polygon in self.zones.items():
            color = (0, 255, 0)  # Green color
            thickness = 2
            cv2.polylines(frame, [polygon.astype(np.int32)], isClosed=True, color=color, thickness=thickness)

            # Calculate centroid of the polygon for text placement
            centroid = np.mean(polygon, axis=0).astype(int)
            cv2.putText(frame, f"Zone {zone_id}", tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

            # Draw points of the polygon
            for point in polygon:
                cv2.circle(frame, tuple(point.astype(int)), 3, color, -1)

    def track_intersections(self, video_path: str, frame_to_edit: int):
        """Tracks objects in the video and detects zone intersections."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video source '{video_path}'.")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / self.fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_edit)
        ret, frame = cap.read()
        if not ret:
            logging.error("Could not read the frame.")
            return

        self.load_zones_for_frame(frame_to_edit)

        frame_id = frame_to_edit

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            results = self.model.track(frame, persist=True, stream=True, tracker=self.tracker_config, device=self.device)

            self.draw_zones(frame)

            for result in results:
                boxes = result.boxes

                # Check if tracking IDs are available
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()
                else:
                    continue

                for i, (bbox, conf, class_id, track_id) in enumerate(
                        zip(boxes.xyxy, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), track_ids)):
                    bbox_np = bbox.cpu().numpy()
                    track_id = int(track_id)

                    max_iou = 0
                    intersecting_zone = None

                    # Initialize tracked_objects entry if it doesn't exist
                    if track_id not in self.tracked_objects:
                        self.tracked_objects[track_id] = {
                            'class_id': int(class_id),
                            'zone_entries': []
                        }

                    for zone_id, polygon in self.zones.items():
                        intersects, iou = self.intersects(bbox_np, polygon)

                        if iou > max_iou:
                            max_iou = iou
                            intersecting_zone = zone_id if intersects else None

                        # Handle zone entry and exit logic
                        zone_entry_exists = any(entry['zone_id'] == zone_id for entry in
                                                self.tracked_objects[track_id]['zone_entries'])
                        if intersects and not zone_entry_exists:
                            # Object just entered the zone
                            entry = {
                                'zone_id': int(zone_id),
                                'first_seen': float(timestamp),
                                'last_seen': None
                            }
                            self.tracked_objects[track_id]['zone_entries'].append(entry)

                            # Prepare entry message
                            detection_entry = DetectionEntry(
                                frame_id=frame_id,
                                object_id=track_id,
                                class_id=int(class_id),
                                confidence=float(conf),
                                zone_id=int(zone_id),
                                first_seen=float(timestamp),
                                event='enter',
                                bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))  # Convert bbox to float
                            )
                            # Publish the detection_entry to EMQX
                            self.detection_log.append(detection_entry)
                            self.save_detection_log()
                            self.mqtt_publisher.send_incident(detection_entry)
                            
                        elif not intersects and zone_entry_exists:
                            # Object just exited the zone
                            for entry in self.tracked_objects[track_id]['zone_entries']:
                                if entry['zone_id'] == zone_id and entry['last_seen'] is None:
                                    entry['last_seen'] = float(timestamp)
                                    # Log the zone entry only when the object leaves
                                    detection_entry = DetectionEntry(
                                        frame_id=frame_id,
                                        object_id=track_id,
                                        class_id=int(class_id),
                                        confidence=float(conf),
                                        zone_id=int(zone_id),
                                        first_seen=float(entry['first_seen']),
                                        last_seen=float(timestamp),
                                        duration=float(timestamp - entry['first_seen']),
                                        event='exit',
                                        bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))  # Convert bbox to float
                                    )
                                    self.detection_log.append(detection_entry)
                                    self.save_detection_log()  # Save when the object leaves the zone

                                    # Publish the detection_entry to EMQX
                                    self.mqtt_publisher.send_incident(detection_entry)

                    time_in_zone = 0
                    current_zone = None
                    for zone_entry in self.tracked_objects[track_id]['zone_entries']:
                        if zone_entry['last_seen'] is None:
                            current_time_in_zone = timestamp - zone_entry['first_seen']
                            if current_time_in_zone > time_in_zone:
                                time_in_zone = current_time_in_zone
                                current_zone = zone_entry['zone_id']
                            # Check if total duration exceeded and not yet logged
                            if 'threshold_logged' not in zone_entry and time_in_zone > self.no_parking_duration:
                                # Log the event
                                detection_entry = DetectionEntry(
                                    frame_id=frame_id,
                                    object_id=track_id,
                                    class_id=int(class_id),
                                    confidence=float(conf),
                                    zone_id=int(current_zone),
                                    first_seen=float(zone_entry['first_seen']),
                                    duration=float(time_in_zone),
                                    event='no_parking',
                                    bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))  # Convert bbox to float
                                )
                                self.detection_log.append(detection_entry)
                                self.save_detection_log()
                                self.mqtt_publisher.send_incident(detection_entry)
                                zone_entry['threshold_logged'] = True

                    # Check for no_entry event
                    if current_zone in self.no_entry_zones:
                        logging.info(f"Object {track_id} is in a no_entry zone: {current_zone}")
                        detection_entry = DetectionEntry(
                            frame_id=frame_id,
                            object_id=track_id,
                            class_id=int(class_id),
                            confidence=float(conf),
                            zone_id=int(current_zone),
                            first_seen=float(timestamp),
                            event='no_entry',
                            bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))  # Convert bbox to float
                        )
                        self.detection_log.append(detection_entry)
                        self.save_detection_log()
                        self.mqtt_publisher.send_incident(detection_entry)

                    # Determine color based on time in zone
                    if time_in_zone > self.time_thresholds[1]:
                        bbox_color = (0, 0, 255)  # Red
                    elif time_in_zone > self.time_thresholds[0]:
                        bbox_color = (0, 165, 255)  # Orange
                    else:
                        bbox_color = (0, 255, 0)  # Green

                    # Draw bounding box
                    x1, y1, x2, y2 = bbox_np.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

                    # Draw object ID, IoU, time in zone, and zone info
                    label1 = f"ID: {track_id} IoU: {max_iou:.2f}"
                    label2 = f"Time: {time_in_zone:.1f}s Zone: {current_zone}"
                    cv2.putText(frame, label1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
                    cv2.putText(frame, label2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

                    # Print debugging info
                    logging.debug(
                        f"Object {track_id}: bbox={bbox_np}, IoU={max_iou:.2f}, time_in_zone={time_in_zone:.1f}, zone={intersecting_zone}")

            if self.show_result:
                cv2.imshow('Zone Intersections', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

        cap.release()
        if self.show_result:
            cv2.destroyAllWindows()

        # Disconnect MQTT client
        self.mqtt_publisher.mqtt_handler.disconnect()
        logging.info("Disconnected from EMQX.")

    def save_detection_log(self):
        """Saves the detection log in JSON format."""
        detection_log_list = []
        for entry in self.detection_log:
            detection_log_list.append(entry.__dict__)

        # Convert numpy float32 to native Python float
        for entry in detection_log_list:
            if 'bbox' in entry:
                entry['bbox'] = tuple(float(coord) for coord in entry['bbox'])

        # Save the list of dictionaries to a JSON file
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(detection_log_list, f, indent=4)

        logging.info(f"Detection log saved to {self.output_file}")