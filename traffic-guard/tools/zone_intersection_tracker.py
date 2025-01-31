import os
import json
import logging
import threading
from collections import defaultdict
from typing import Dict, Any, Tuple, List
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from .detection_entry import DetectionEntry
from .mqtt_publisher import MqttPublisher
from .mqtt_subscriber import MqttSubscriber
from ultralytics import YOLO


class ZoneIntersectionTracker:
    """Tracks objects in a video and detects intersections with defined zones."""

    def __init__(self, config: Dict[str, Any], show_result: bool = True):
        self.config = config
        model_path = config['model_path']
        mask_json_path = config['mask_json_path']
        
        edge_id = config.get('edge_id', 'default_id')
        logging.info(f"ZoneIntersectionTracker initialized with edge_id: {edge_id}")
        
        self.tracker_config = config.get('tracker_config', 'bytetrack.yaml')
        self.show_result = show_result

        # Handle model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        # Handle mask_json_path
        if not os.path.exists(mask_json_path):
            raise FileNotFoundError(f"Mask positions file '{mask_json_path}' not found.")
        self.mask_json_path = config['mask_json_path']
        self.load_mask_positions()

        self.zones = {}
        self.arrows = {}
        self.detection_log = []
        self.tracked_objects = {}
        self.object_zone_timers = defaultdict(lambda: defaultdict(float))
        self.lock = threading.Lock()

        # Initialize marker dictionaries
        self.zones = {}
        self.arrows = {}
        
        # Load markers
        self.load_zones()
        self.load_arrows()

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

        # Define vehicle class IDs (adjust based on your YOLO model)
        self.vehicle_class_ids = [2, 3, 5, 7]  # Example: car, motorcycle, bus, truck

        # Load total duration and compute time thresholds
        self.total_duration = config.get('total_duration', 5)  # Default to 5 seconds if not specified
        # Calculate time thresholds for three colors
        self.time_thresholds = [
            2 * self.total_duration / 3,   # Threshold for green to orange
            self.total_duration            # Threshold for orange to red
        ]

        # Initialize MQTT publisher
        mqtt_config = {
            'mqtt_broker': config['mqtt_broker'],
            'mqtt_port': config['mqtt_port'],
            'mqtt_username': config['mqtt_username'],
            'mqtt_password': config['mqtt_password'],
            'ca_cert_path': config['ca_cert_path']
        }
        publisher_config = {
            'edge_id': edge_id,
            'heartbeat_interval': config.get('heartbeat_interval', 60),
            'heartbeat_topic': f'{edge_id}/heartbeat',
            'incident_info_topic': f'{edge_id}/incident'
        }
        self.mqtt_publisher = MqttPublisher({**mqtt_config, **publisher_config})

        # Initialize MQTT subscriber and pass the MqttPublisher instance
        self.mqtt_subscriber = MqttSubscriber(config, self.mqtt_publisher)
        self.mqtt_subscriber.mqtt_client.message_callback_add(f'{edge_id}/marker/create', self.on_create_marker)
        self.mqtt_subscriber.mqtt_client.message_callback_add(f'{edge_id}/marker/update', self.on_update_marker)
        self.mqtt_subscriber.mqtt_client.message_callback_add(f'{edge_id}/marker/delete', self.on_delete_marker)

        # Load event-specific configurations
        self.no_entry_zones = config.get('no_entry_zones', [])
        self.no_parking_duration = config.get('no_parking_duration', 60)  # in seconds

    def load_mask_positions(self):
        """Loads mask positions from the JSON file."""
        if not os.path.exists(self.mask_json_path):
            logging.error(f"Mask positions file '{self.mask_json_path}' not found.")
            return

        try:
            with open(self.mask_json_path, 'r') as f:
                mask_positions = json.load(f)
            logging.info("Mask positions loaded successfully.")
            
            # Remove duplicate marker_ids, keeping the first occurrence
            unique_mask_positions = []
            seen_marker_ids = set()
            for entry in mask_positions:
                marker_id = entry.get('marker_id')
                if marker_id not in seen_marker_ids:
                    unique_mask_positions.append(entry)
                    seen_marker_ids.add(marker_id)
                else:
                    logging.warning(f"Duplicate marker_id {marker_id} found. Skipping duplicate entry.")
            self.mask_positions = unique_mask_positions
        except json.JSONDecodeError as e:
            logging.error(f"Error loading mask positions: {e}")
            self.mask_positions = []
        except Exception as e:
            logging.error(f"Unexpected error loading mask positions: {e}")
            self.mask_positions = []

    def load_zones(self):
        """Loads zones from mask positions."""
        self.zones.clear()
        for entry in self.mask_positions:
            if entry.get('type') == 'zone':
                marker_id = entry.get('marker_id')
                points = entry.get('points', [])
                if marker_id and points:
                    self.zones[marker_id] = np.array(points)
                else:
                    logging.warning(f"Zone entry missing 'marker_id' or 'points': {entry}")

    def load_arrows(self):
        """Loads arrows from mask positions."""
        self.arrows.clear()
        for entry in self.mask_positions:
            if entry.get('type') == 'movement':
                marker_id = entry.get('marker_id')
                polygon_points = entry.get('polygon_points', [])
                line_points = entry.get('line_points', [])
                if marker_id and polygon_points and line_points:
                    self.arrows[marker_id] = {
                        'polygon_points': np.array(polygon_points),
                        'line_points': np.array(line_points)
                    }
                else:
                    logging.warning(f"Movement entry missing 'marker_id', 'polygon_points', or 'line_points': {entry}")

    def on_create_marker(self, client, userdata, msg):
        """Handles the creation of a new marker from an MQTT message."""
        with self.lock:
            try:
                data = json.loads(msg.payload.decode())

                is_valid, message = self._validate_payload(data)
                if not is_valid:
                    logging.error(f"Invalid payload received: {message}")
                    self.mqtt_publisher.send_incident({"error": message})
                    return

                # Check for duplicate marker_id
                if any(entry['marker_id'] == data['marker_id'] for entry in self.mask_positions):
                    error_msg = f"Duplicate marker_id {data['marker_id']} detected. Creation aborted."
                    logging.error(error_msg)
                    self.mqtt_publisher.send_incident({"error": error_msg})
                    return

                self.mask_positions.append(data)
                self.save_mask_positions()
                self.load_zones()
                self.load_arrows()
                logging.info(f"Created marker position: {data}")
                self.mqtt_publisher.send_incident({"message": "create complete", "marker_id": data['marker_id']})

            except json.JSONDecodeError:
                error_msg = "Invalid JSON payload. Marker creation aborted."
                logging.error(error_msg)
                self.mqtt_publisher.send_incident({"error": error_msg})
            except Exception as e:
                error_msg = f"Error creating marker position: {e}"
                logging.error(error_msg)
                self.mqtt_publisher.send_incident({"error": error_msg})

    def _validate_payload(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates the payload for creating a marker."""
        required_fields = ['marker_id', 'type', 'status']
        
        # Check for required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f"Payload missing required fields: {missing_fields}."
            logging.error(error_msg)
            return False, error_msg

        # Validate 'type' field
        if data['type'] not in ['zone', 'movement']:
            error_msg = f"Invalid type: {data['type']}. Must be 'zone' or 'movement'."
            logging.error(error_msg)
            return False, error_msg

        # Validate 'marker_id' uniqueness
        existing_ids = {entry['marker_id'] for entry in self.mask_positions}
        if data['marker_id'] in existing_ids:
            error_msg = f"Marker with marker_id={data['marker_id']} already exists."
            logging.error(error_msg)
            return False, error_msg

        # Validate 'points' format for 'zone' type
        if data['type'] == 'zone':
            if 'points' not in data:
                error_msg = "Missing 'points' for zone type."
                logging.error(error_msg)
                return False, error_msg
            if (not isinstance(data['points'], list) or 
                not all(isinstance(point, list) and len(point) == 2 for point in data['points'])):
                error_msg = "Invalid 'points' format for zone. Must be a list of [x, y] pairs."
                logging.error(error_msg)
                return False, error_msg

        # Validate 'polygon_points' and 'line_points' for 'movement' type
        if data['type'] == 'movement':
            if 'polygon_points' not in data or 'line_points' not in data:
                error_msg = "Missing 'polygon_points' or 'line_points' for movement type."
                logging.error(error_msg)
                return False, error_msg
            if (not isinstance(data['polygon_points'], list) or 
                not all(isinstance(point, list) and len(point) == 2 for point in data['polygon_points'])):
                error_msg = "Invalid 'polygon_points' format for movement. Must be a list of [x, y] pairs."
                logging.error(error_msg)
                return False, error_msg
            if (not isinstance(data['line_points'], list) or 
                not all(isinstance(point, list) and len(point) == 2 for point in data['line_points'])):
                error_msg = "Invalid 'line_points' format for movement. Must be a list of [x, y] pairs."
                logging.error(error_msg)
                return False, error_msg

        # Validate 'status' field
        if data['status'] not in ['activate', 'deactivate']:
            error_msg = f"Invalid status: {data['status']}. Must be 'activate' or 'deactivate'."
            logging.error(error_msg)
            return False, error_msg

        return True, "Payload is valid."

    def on_update_marker(self, client, userdata, msg):
        """Handles the update of an existing marker from an MQTT message."""
        try:
            data = json.loads(msg.payload.decode())
            is_valid, message = self._validate_payload_update(data)
            if not is_valid:
                logging.error(f"Invalid payload received: {message}")
                self.mqtt_publisher.send_incident({"error": message})
                return

            updated = False
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == data['marker_id']:
                    self.mask_positions[i].update(data)
                    updated = True
                    break
            if not updated:
                warning_msg = "No matching marker found to update."
                logging.warning(warning_msg)
                self.mqtt_publisher.send_incident({"warning": warning_msg})
                return
            self.save_mask_positions()
            self.load_zones()
            self.load_arrows()
            logging.info(f"Updated marker position: {data}")
            self.mqtt_publisher.send_incident({"message": "update complete", "marker_id": data['marker_id']})
        except json.JSONDecodeError:
            error_msg = "Invalid JSON payload. Marker update aborted."
            logging.error(error_msg)
            self.mqtt_publisher.send_incident({"error": error_msg})
        except Exception as e:
            error_msg = f"Error updating marker position: {e}"
            logging.error(error_msg)
            self.mqtt_publisher.send_incident({"error": error_msg})

    def _validate_payload_update(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates the payload for updating a marker."""
        required_fields = ['marker_id']
        
        # Check for required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f"Payload missing required fields: {missing_fields}."
            logging.error(error_msg)
            return False, error_msg

        # Validate 'type' field if present
        if 'type' in data and data['type'] not in ['zone', 'movement']:
            error_msg = f"Invalid type: {data['type']}. Must be 'zone' or 'movement'."
            logging.error(error_msg)
            return False, error_msg

        # Validate 'points' format if present for 'zone'
        if data.get('type') == 'zone':
            if 'points' not in data:
                error_msg = "Missing 'points' for zone type."
                logging.error(error_msg)
                return False, error_msg
            if (not isinstance(data['points'], list) or 
                not all(isinstance(point, list) and len(point) == 2 for point in data['points'])):
                error_msg = "Invalid 'points' format for zone. Must be a list of [x, y] pairs."
                logging.error(error_msg)
                return False, error_msg

        # Validate 'polygon_points' and 'line_points' format if present for 'movement'
        if data.get('type') == 'movement':
            if 'polygon_points' not in data or 'line_points' not in data:
                error_msg = "Missing 'polygon_points' or 'line_points' for movement type."
                logging.error(error_msg)
                return False, error_msg
            if (not isinstance(data['polygon_points'], list) or 
                not all(isinstance(point, list) and len(point) == 2 for point in data['polygon_points'])):
                error_msg = "Invalid 'polygon_points' format for movement. Must be a list of [x, y] pairs."
                logging.error(error_msg)
                return False, error_msg
            if (not isinstance(data['line_points'], list) or 
                not all(isinstance(point, list) and len(point) == 2 for point in data['line_points'])):
                error_msg = "Invalid 'line_points' format for movement. Must be a list of [x, y] pairs."
                logging.error(error_msg)
                return False, error_msg

        # Validate 'status' field if present
        if 'status' in data and data['status'] not in ['activate', 'deactivate']:
            error_msg = f"Invalid status: {data['status']}. Must be 'activate' or 'deactivate'."
            logging.error(error_msg)
            return False, error_msg

        return True, "Payload is valid."

    def on_delete_marker(self, client, userdata, msg):
        """Handles the deletion of a marker from an MQTT message."""
        try:
            data = json.loads(msg.payload.decode())
            markers_deleted = 0

            if 'marker_id' in data:
                marker_id = data['marker_id']
                original_count = len(self.mask_positions)
                self.mask_positions = [
                    position for position in self.mask_positions
                    if position.get('marker_id') != marker_id
                ]
                markers_deleted = original_count - len(self.mask_positions)
                logging.debug(f"Markers deleted with marker_id={marker_id}: {markers_deleted}")

                # Remove marker entries from tracked_objects
                for track_id, obj in self.tracked_objects.items():
                    original_entries = len(obj['marker_entries'])
                    obj['marker_entries'] = [
                        entry for entry in obj['marker_entries']
                        if entry['marker_id'] != marker_id
                    ]
                    if len(obj['marker_entries']) < original_entries:
                        logging.debug(f"Reset marker entries for object {track_id} due to marker deletion.")

            else:
                error_msg = "Delete payload must contain 'marker_id'."
                logging.error(error_msg)
                self.mqtt_publisher.send_incident({"error": error_msg})
                return

            if markers_deleted == 0:
                warning_msg = "No matching marker found to delete."
                logging.warning(warning_msg)
                self.mqtt_publisher.send_incident({"warning": warning_msg})
            else:
                logging.info(f"Deleted marker position: {data}")
                self.mqtt_publisher.send_incident({"message": "delete complete", "marker_id": marker_id})

            self.save_mask_positions()
            self.load_zones()
            self.load_arrows()

        except json.JSONDecodeError:
            error_msg = "Invalid JSON payload. Marker deletion aborted."
            logging.error(error_msg)
            self.mqtt_publisher.send_incident({"error": error_msg})
        except Exception as e:
            error_msg = f"Error deleting marker position: {e}"
            logging.error(error_msg)
            self.mqtt_publisher.send_incident({"error": error_msg})

    def save_mask_positions(self):
        """Saves the mask positions to the JSON file."""
        try:
            with open(self.mask_json_path, 'w') as f:
                json.dump(self.mask_positions, f, indent=4)
            logging.info("Mask positions saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save mask positions: {e}")

    def calculate_iou(self, bbox: np.ndarray, polygon: np.ndarray) -> float:
        """Calculates the Intersection over Union (IoU) between a bounding box and a polygon."""
        x1, y1, x2, y2 = bbox
        bbox_polygon = Polygon([
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ])
        polygon_shape = Polygon(polygon)

        intersection = bbox_polygon.intersection(polygon_shape)
        union = bbox_polygon.union(polygon_shape)
        if union.area == 0:
            return 0.0
        return intersection.area / union.area

    def intersects(self, bbox: np.ndarray, polygon: np.ndarray) -> Tuple[bool, float]:
        """Determines if a bounding box intersects with a polygon."""
        iou = self.calculate_iou(bbox, polygon)
        intersects = iou > self.config.get('iou_threshold', 0.1)
        return intersects, iou

    def draw_zones(self, frame: np.ndarray):
        """Draws zones on the frame."""
        for marker_id, polygon in self.zones.items():
            color = (0, 255, 0)  # Green color for zones
            thickness = 2
            if polygon.shape[0] >= 3:
                cv2.polylines(frame, [polygon.astype(np.int32)], isClosed=True, color=color, thickness=thickness)
                centroid = np.mean(polygon, axis=0).astype(int)
                cv2.putText(frame, f"Zone {marker_id}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            else:
                logging.warning(f"Zone {marker_id} has insufficient points to draw.")

    def draw_arrows(self, frame: np.ndarray):
        """Draws arrows on the frame."""
        for marker_id, arrow_data in self.arrows.items():
            polygon_points = arrow_data['polygon_points']
            line_points = arrow_data['line_points']
            color_polygon = (0, 255, 255)  # Yellow color for arrow zones
            color_line = (255, 0, 0)        # Blue color for arrows
            thickness = 2

            if polygon_points.shape[0] >= 3:
                cv2.polylines(frame, [polygon_points.astype(np.int32)], isClosed=True, color=color_polygon, thickness=thickness)
                centroid = np.mean(polygon_points, axis=0).astype(int)
                cv2.putText(frame, f"Arrow Zone {marker_id}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_polygon, thickness)
            else:
                logging.warning(f"Arrow Zone {marker_id} has insufficient points to draw.")

            if line_points.shape[0] >= 2:
                for i in range(len(line_points) - 1):
                    start_point = tuple(map(int, line_points[i]))
                    end_point = tuple(map(int, line_points[i + 1]))
                    cv2.arrowedLine(frame, start_point, end_point, color_line, thickness=2, tipLength=0.05)
                cv2.putText(frame, f"Movement {marker_id}", tuple(line_points[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line, thickness)
            else:
                logging.warning(f"Movement {marker_id} has insufficient points to draw.")

    def track_intersections(self, video_path: str, frame_to_edit: int):
        """Tracks objects in the video and detects zone intersections."""
        # Check if the input is a YouTube URL
        if video_path.startswith(('http://', 'https://')):
            try:
                import yt_dlp
            except ImportError:
                raise ImportError("Please install yt-dlp to handle YouTube URLs: pip install yt-dlp")

            # Extract the best video stream URL using yt-dlp
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_path, download=False)
                    if 'url' in info:
                        video_url = info['url']
                    else:
                        # Fallback to the first available format
                        formats = info.get('formats', [])
                        if not formats:
                            raise ValueError("No playable formats found for the YouTube URL.")
                        video_url = formats[0]['url']
                    video_path = video_url
            except Exception as e:
                logging.error(f"Failed to process YouTube URL: {e}")
                return

        # Proceed with OpenCV VideoCapture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video source '{video_path}'.")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / self.fps

        # Attempt to seek only if it's not a live stream
        if frame_to_edit > 0 and not video_path.startswith(('rtsp://', 'rtmp://')):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_edit)
        
        ret, frame = cap.read()
        if not ret:
            logging.error("Could not read the frame.")
            return

        # Rest of the original method remains unchanged
        self.load_zones()
        self.load_arrows()

        frame_id = frame_to_edit

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            results = self.model.track(frame, verbose=False, persist=True, stream=True, tracker=self.tracker_config, device=self.device)

            # Draw zones and arrows on the current frame
            self.draw_zones(frame)
            self.draw_arrows(frame)

            for result in results:
                boxes = result.boxes

                # Check if tracking IDs are available
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()
                else:
                    continue

                for i, (bbox, conf, class_id, track_id) in enumerate(
                        zip(boxes.xyxy, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), track_ids)):
                    # Filter out non-vehicle objects
                    if int(class_id) not in self.vehicle_class_ids:
                        continue

                    bbox_np = bbox.cpu().numpy()
                    track_id = int(track_id)

                    max_iou = 0
                    intersecting_marker_id = None

                    # Initialize tracked_objects entry if it doesn't exist
                    if track_id not in self.tracked_objects:
                        self.tracked_objects[track_id] = {
                            'class_id': int(class_id),
                            'marker_entries': []
                        }

                    for marker_id, polygon in self.zones.items():
                        intersects, iou = self.intersects(bbox_np, polygon)

                        if iou > max_iou:
                            max_iou = iou
                            intersecting_marker_id = marker_id if intersects else None

                        # Handle marker entry and exit logic
                        marker_entry_exists = any(entry['marker_id'] == marker_id for entry in
                                                 self.tracked_objects[track_id]['marker_entries'])
                        if intersects and not marker_entry_exists:
                            # Object just entered the marker zone
                            entry = {
                                'marker_id': int(marker_id),
                                'first_seen': float(timestamp),
                                'last_seen': None
                            }
                            self.tracked_objects[track_id]['marker_entries'].append(entry)

                            # Prepare entry message
                            detection_entry = DetectionEntry(
                                object_id=track_id,
                                class_id=int(class_id),
                                confidence=float(conf),
                                marker_id=int(marker_id),
                                first_seen=float(timestamp),
                                last_seen=None,
                                duration=None,
                                event='enter',
                                bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))  # Convert bbox to float
                            )
                            # Publish the detection_entry to EMQX
                            self.detection_log.append(detection_entry)
                            self.save_detection_log()
                            self.mqtt_publisher.send_incident(detection_entry)
                            
                        elif not intersects and marker_entry_exists:
                            # Object just exited the marker zone
                            for entry in self.tracked_objects[track_id]['marker_entries']:
                                if entry['marker_id'] == marker_id and entry['last_seen'] is None:
                                    entry['last_seen'] = float(timestamp)
                                    # Log the marker entry only when the object leaves
                                    detection_entry = DetectionEntry(
                                        object_id=track_id,
                                        class_id=int(class_id),
                                        confidence=float(conf),
                                        marker_id=int(marker_id),
                                        first_seen=float(entry['first_seen']),
                                        last_seen=float(timestamp),
                                        duration=float(timestamp - entry['first_seen']),
                                        event='exit',
                                        bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))  # Convert bbox to float
                                    )
                                    self.detection_log.append(detection_entry)
                                    self.save_detection_log()  # Save when the object leaves the marker zone

                                    # Publish the detection_entry to EMQX
                                    self.mqtt_publisher.send_incident(detection_entry)

                    # Handle movement markers
                    for marker_id, arrow_data in self.arrows.items():
                        polygon_points = arrow_data['polygon_points']
                        line_points = arrow_data['line_points']
                        intersects_movement, iou_movement = self.intersects(bbox_np, polygon_points)
                        if iou_movement > max_iou:
                            max_iou = iou_movement
                            intersecting_marker_id = marker_id if intersects_movement else None

                        # Handle movement entry and exit logic
                        marker_entry_exists = any(entry['marker_id'] == marker_id for entry in
                                                 self.tracked_objects[track_id]['marker_entries'])
                        if intersects_movement and not marker_entry_exists:
                            # Object just entered the movement zone
                            entry = {
                                    'marker_id': int(marker_id),
                                    'type': 'movement',  # Add type
                                    'first_seen': float(timestamp),
                                    'last_seen': None,
                                    'trajectory': []  # Initialize trajectory
                                }
                            self.tracked_objects[track_id]['marker_entries'].append(entry)

                            # Prepare entry message
                            detection_entry = DetectionEntry(
                                object_id=track_id,
                                class_id=int(class_id),
                                confidence=float(conf),
                                marker_id=int(marker_id),
                                first_seen=float(timestamp),
                                last_seen=None,
                                duration=None,
                                event='enter_movement',
                                bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))
                            )
                            # Publish the detection_entry to EMQX
                            self.detection_log.append(detection_entry)
                            self.save_detection_log()
                            self.mqtt_publisher.send_incident(detection_entry)
                        
                        elif intersects_movement:
                            # Update trajectory for existing entry
                            for entry in self.tracked_objects[track_id]['marker_entries']:
                                if entry['marker_id'] == marker_id and entry['last_seen'] is None:
                                    # Append current position (center of bbox)
                                    x_center = (bbox_np[0] + bbox_np[2]) / 2
                                    y_center = (bbox_np[1] + bbox_np[3]) / 2
                                    entry['trajectory'].append((x_center, y_center))
                                    # Keep only the last 10 points to avoid memory issues
                                    if len(entry['trajectory']) > 10:
                                        entry['trajectory'] = entry['trajectory'][-10:]
                            
                        elif not intersects_movement and marker_entry_exists:
                            # Object just exited the movement zone
                            for entry in self.tracked_objects[track_id]['marker_entries']:
                                if entry['marker_id'] == marker_id and entry['last_seen'] is None:
                                    entry['last_seen'] = float(timestamp)
                                    # Log the marker entry only when the object leaves
                                    detection_entry = DetectionEntry(
                                        object_id=track_id,
                                        class_id=int(class_id),
                                        confidence=float(conf),
                                        marker_id=int(marker_id),
                                        first_seen=float(entry['first_seen']),
                                        last_seen=float(timestamp),
                                        duration=float(timestamp - entry['first_seen']),
                                        event='exit_movement',
                                        bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))
                                    )
                                    self.detection_log.append(detection_entry)
                                    self.save_detection_log()  # Save when the object leaves the movement zone

                                    # Publish the detection_entry to EMQX
                                    self.mqtt_publisher.send_incident(detection_entry)

                    # Check for no_parking and no_entry events
                    self._check_no_parking(track_id, class_id, conf, bbox_np, timestamp)
                    self._check_no_entry(track_id, class_id, conf, bbox_np, timestamp)
                    self._check_wrong_way(track_id, class_id, conf, bbox_np, timestamp)  # Add this line

                    # Determine color based on time in zone
                    time_in_zone = self._get_time_in_zone(track_id, timestamp)
                    bbox_color = self._get_bbox_color(time_in_zone)

                    # Draw bounding box
                    x1, y1, x2, y2 = bbox_np.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

                    # Draw object ID, IoU, time in zone, and marker info
                    label1 = f"ID: {track_id} IoU: {max_iou:.2f}"
                    marker_info = intersecting_marker_id if intersecting_marker_id else 'N/A'
                    if isinstance(intersecting_marker_id, list):
                        marker_info = ', '.join(map(str, intersecting_marker_id))
                    label2 = f"Time: {time_in_zone:.1f}s Marker: {marker_info}"
                    cv2.putText(frame, label1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
                    cv2.putText(frame, label2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

                    # Print debugging info
                    logging.debug(
                        f"Object {track_id}: bbox={bbox_np}, IoU={max_iou:.2f}, time_in_zone={time_in_zone:.1f}, marker={intersecting_marker_id}")

            if self.show_result:
                cv2.imshow('Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

        cap.release()
        if self.show_result:
            cv2.destroyAllWindows()

        # Disconnect MQTT client
        self.mqtt_publisher.mqtt_handler.disconnect()
        logging.info("Disconnected from EMQX.")

    def _check_no_parking(self, track_id: int, class_id: int, conf: float, bbox_np: np.ndarray, timestamp: float):
        """Checks if an object has exceeded the no_parking duration in a zone."""
        for marker_entry in self.tracked_objects[track_id]['marker_entries']:
            if marker_entry['last_seen'] is None:
                time_in_zone = timestamp - marker_entry['first_seen']
                if time_in_zone > self.no_parking_duration and 'threshold_logged' not in marker_entry:
                    logging.info(f"Object {track_id} exceeded no_parking duration in marker {marker_entry['marker_id']}")
                    detection_entry = DetectionEntry(
                        object_id=track_id,
                        class_id=int(class_id),
                        confidence=float(conf),
                        marker_id=int(marker_entry['marker_id']),
                        first_seen=float(marker_entry['first_seen']),
                        last_seen=None,
                        duration=float(time_in_zone),
                        event='no_parking',
                        bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))
                    )
                    self.detection_log.append(detection_entry)
                    self.save_detection_log()
                    self.mqtt_publisher.send_incident(detection_entry)
                    marker_entry['threshold_logged'] = True

    def _check_no_entry(self, track_id: int, class_id: int, conf: float, bbox_np: np.ndarray, timestamp: float):
        """Checks if an object is in a no_entry zone."""
        for marker_entry in self.tracked_objects[track_id]['marker_entries']:
            if marker_entry['marker_id'] in self.no_entry_zones and marker_entry['last_seen'] is None:
                logging.info(f"Object {track_id} is in a no_entry zone: {marker_entry['marker_id']}")
                detection_entry = DetectionEntry(
                    object_id=track_id,
                    class_id=int(class_id),
                    confidence=float(conf),
                    marker_id=int(marker_entry['marker_id']),
                    first_seen=float(timestamp),
                    last_seen=None,
                    duration=None,
                    event='no_entry',
                    bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))
                )
                self.detection_log.append(detection_entry)
                self.save_detection_log()
                self.mqtt_publisher.send_incident(detection_entry)
                
                
    def calculate_direction(self, start_point, end_point):
        """Calculate the unit vector for the direction."""
        direction = np.array(end_point) - np.array(start_point)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return direction  # Avoid division by zero
        return direction / norm

    def is_wrong_way(self, car_trajectory, polyline):
        """
        Determine if a car is going the wrong way by comparing its trajectory to the polyline direction.
        """
        if len(polyline) < 2 or len(car_trajectory) < 2:
            return False  # Not enough points
        polyline_direction = self.calculate_direction(polyline[0], polyline[-1])
        car_direction = self.calculate_direction(car_trajectory[0], car_trajectory[-1])
        dot_product = np.dot(car_direction, polyline_direction)
        return dot_product < 0
    
    def _check_wrong_way(self, track_id: int, class_id: int, conf: float, bbox_np: np.ndarray, timestamp: float):
        """Checks if a vehicle is moving in the wrong direction within a movement zone."""
        for marker_entry in self.tracked_objects[track_id]['marker_entries']:
            # Check only active movement markers
            if marker_entry.get('type') == 'movement' and marker_entry['last_seen'] is None:
                marker_id = marker_entry['marker_id']
                arrow_data = self.arrows.get(marker_id)
                if not arrow_data or 'line_points' not in arrow_data:
                    continue
                line_points = arrow_data['line_points']
                if len(line_points) < 2:
                    continue  # Invalid arrow
                # Get the vehicle's trajectory
                trajectory = marker_entry.get('trajectory', [])
                if len(trajectory) < 2:
                    continue  # Not enough trajectory points
                # Check direction
                if self.is_wrong_way(trajectory, line_points):
                    logging.info(f"Vehicle {track_id} is going the wrong way in marker {marker_id}")
                    detection_entry = DetectionEntry(
                        object_id=track_id,
                        class_id=int(class_id),
                        confidence=float(conf),
                        marker_id=int(marker_id),
                        first_seen=float(timestamp),
                        last_seen=None,
                        duration=None,
                        event='wrong_way',
                        bbox=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))
                    )
                    self.detection_log.append(detection_entry)
                    self.save_detection_log()
                    self.mqtt_publisher.send_incident(detection_entry)
                    # Prevent duplicate logging
                    marker_entry['wrong_way_logged'] = True

    def _get_time_in_zone(self, track_id: int, timestamp: float) -> float:
        """Calculates the time an object has spent in any zone."""
        time_in_zone = 0
        for marker_entry in self.tracked_objects[track_id]['marker_entries']:
            if marker_entry['last_seen'] is None:
                current_time_in_zone = timestamp - marker_entry['first_seen']
                if current_time_in_zone > time_in_zone:
                    time_in_zone = current_time_in_zone
        return time_in_zone

    def _get_bbox_color(self, time_in_zone: float) -> Tuple[int, int, int]:
        """Determines the bounding box color based on the time spent in a zone."""
        if time_in_zone > self.time_thresholds[1]:
            return (0, 0, 255)  # Red
        elif time_in_zone > self.time_thresholds[0]:
            return (0, 165, 255)  # Orange
        else:
            return (0, 255, 0)  # Green

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