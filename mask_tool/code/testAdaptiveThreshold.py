import cv2
import numpy as np
import json
import os
import torch
import uuid
import logging
import ssl
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from ultralytics import YOLO
import paho.mqtt.client as mqtt_client
import scipy.stats as stats  # For multivariate normal distribution calculations


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

@dataclass
class DetectionEntry:
    """Data class for storing detection entries."""
    frame_id: int
    object_id: int
    class_id: int
    confidence: float
    zone_id: int
    first_seen: float
    last_seen: float = None
    duration: float = None
    event: str = ''
    timestamp: float = None  # Added timestamp field for events like 'threshold_exceeded'

class MaskTool:
    """Interactive tool for defining polygonal zones on a video frame."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        video_source = config['video_source']
        frame_to_edit = config['frame_to_edit']
        mask_json_path = config.get('mask_json_path', None)
        self.cap = cv2.VideoCapture(video_source)
        self.mask_positions = []
        self.drawing = False
        self.editing = False
        self.current_polygon: List[Tuple[int, int]] = []
        self.zone_id = 1
        self.frame_id = frame_to_edit
        self.dragging_point = False
        self.target_frame = None
        self.zones: Dict[int, List[Tuple[int, int]]] = {}
        self.undo_stack = []
        self.redo_stack = []
        self.highlight_radius = config.get('highlight_radius', 10)
        self.point_radius = config.get('point_radius', 5)
        self.line_threshold = config.get('line_threshold', 10)
        self.selection_threshold = config.get('selection_threshold', 10)
        self.selected_point_index = -1

        # Handle mask_json_path
        if mask_json_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mask_json_path = os.path.join(script_dir, 'config', 'mask_positions.json')
        self.mask_json_path = mask_json_path

        # Load existing mask positions
        if os.path.exists(self.mask_json_path):
            try:
                with open(self.mask_json_path, 'r') as f:
                    self.mask_positions = json.load(f)
            except Exception as e:
                logging.error(f"Error loading mask positions from {self.mask_json_path}: {e}")

        # Set the video capture to the selected frame
        if not self.cap.isOpened():
            logging.error(f"Cannot open video source '{video_source}'.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        ret, self.target_frame = self.cap.read()
        if not ret:
            logging.error("Could not read the frame.")
            return

        # Load existing polygons for the current frame (if any)
        self.load_polygons_for_frame(self.frame_id)

    def draw_polygon(self, img: np.ndarray, polygon: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0)):
        """Draws a polygon on the image."""
        if len(polygon) > 0:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
            for p in polygon:
                p = self.ensure_point_format(p)
                cv2.circle(img, p, self.point_radius, color, -1)

    def ensure_point_format(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Ensures the point is in integer format."""
        return tuple(map(int, point))

    def draw_zone_ids(self, img: np.ndarray):
        """Draws zone IDs on the image."""
        for zone_id, polygon in self.zones.items():
            if len(polygon) > 0:
                self.draw_polygon(img, polygon, color=(0, 255, 0))
                pts = np.array(polygon, np.float32)
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, f'Zone {zone_id}', (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    def draw_instructions(self, img: np.ndarray):
        """Displays instructions on the image."""
        instructions = [
            "Press 'e' to toggle Editing Mode",
            "Press 's' to Save Mask",
            "Press 'u' to Undo",
            "Press 'r' to Redo",
            "Press 'q' to Quit"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_axes(self, img: np.ndarray):
        """Draws X and Y axes on the image."""
        height, width = img.shape[:2]
        # Draw x-axis (blue line)
        cv2.line(img, (0, height - 1), (width, height - 1), (255, 0, 0), 2)
        cv2.putText(img, 'X', (width - 20, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Draw y-axis (green line)
        cv2.line(img, (0, 0), (0, height), (0, 255, 0), 2)
        cv2.putText(img, 'Y', (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def is_within_bounds(self, point: Tuple[int, int], img: np.ndarray) -> bool:
        """Checks if a point is within the image boundaries."""
        px, py = point
        height, width = img.shape[:2]
        return 0 <= px < width and 0 <= py < height

    def add_point_to_polygon(self, point: Tuple[int, int], img: np.ndarray):
        """Adds a point to the current polygon."""
        if not self.is_within_bounds(point, img):
            return

        if len(self.current_polygon) == 0:
            self.current_polygon.append(point)
        elif self.is_point_near_polygon_point(point, self.current_polygon):
            if len(self.current_polygon) > 2 and self.current_polygon[0] != point:
                self.current_polygon.append(self.current_polygon[0])
                self.draw_polygon(img, self.current_polygon)
                self.save_polygon()
                self.current_polygon = []
                self.drawing = False
                self.zone_id = self.get_next_available_zone_id()
            else:
                logging.info("Polygon too small to close or point too close to start")
        else:
            self.current_polygon.append(point)

    def draw_mask(self, event, x, y, flags, param):
        """Handles mouse events for drawing and editing polygons."""
        img = self.target_frame.copy()

        def is_point_near_line_segment(point, line_start, line_end, threshold):
            """Checks if a point is near a line segment."""
            if np.array_equal(line_start, line_end):
                return False, None

            line_vec = np.array(line_end) - np.array(line_start)
            point_vec = np.array(point) - np.array(line_start)
            line_len = np.linalg.norm(line_vec)

            proj = np.dot(point_vec, line_vec) / line_len ** 2

            if proj < 0:
                closest_point = line_start
            elif proj > 1:
                closest_point = line_end
            else:
                closest_point = line_start + proj * line_vec

            dist = np.linalg.norm(np.array(point) - np.array(closest_point))

            return dist < self.line_threshold, closest_point

        def insert_point_between_segments(point):
            """Inserts a point into the active polygon if near a segment."""
            inserted = False
            if self.zone_id is not None and self.zone_id in self.zones:
                polygon = self.zones[self.zone_id]
                for i in range(len(polygon) - 1):
                    near, closest_point = is_point_near_line_segment(
                        point, polygon[i], polygon[i + 1], self.line_threshold
                    )
                    if near:
                        self.current_polygon = self.zones[self.zone_id]
                        self.current_polygon.insert(i + 1, tuple(closest_point))
                        self.zones[self.zone_id] = self.current_polygon
                        inserted = True
                        break
                if len(polygon) > 2:
                    near, closest_point = is_point_near_line_segment(
                        point, polygon[-1], polygon[0], self.line_threshold
                    )
                    if near:
                        self.current_polygon = self.zones[self.zone_id]
                        self.current_polygon.insert(0, tuple(closest_point))
                        self.zones[self.zone_id] = self.current_polygon
                        inserted = True

                if inserted:
                    self.draw_polygon(img, self.current_polygon)

        def select_nearest_point(point):
            """Selects the nearest point in the current polygon."""
            min_dist = float('inf')
            selected_index = -1
            for i, p in enumerate(self.current_polygon):
                dist = np.linalg.norm(np.array(p) - np.array(point))
                if dist < min_dist:
                    min_dist = dist
                    selected_index = i
            return selected_index if min_dist < self.selection_threshold else -1

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.editing:
                if not self.dragging_point:
                    self.zone_id = None
                    clicked_existing_point = False

                    # Iterate through zones to find if a point or line is clicked
                    for zone_id, polygon in self.zones.items():
                        # Check for click near existing point
                        if self.is_point_near_polygon_point((x, y), polygon):
                            # Existing point clicked - start dragging
                            self.selected_point_index = self.find_closest_point_index(polygon, (x, y))
                            if self.selected_point_index != -1:
                                self.current_polygon = list(polygon)
                                self.dragging_point = True
                                self.zone_id = zone_id
                                self.drawing = False
                                clicked_existing_point = True
                                break

                        # Check for click near line segment (excluding vertices)
                        for i in range(len(polygon) - 1):
                            near_line, _ = is_point_near_line_segment(
                                (x, y), polygon[i], polygon[i + 1], self.line_threshold
                            )
                            if near_line:
                                self.zone_id = zone_id
                                self.current_polygon = list(self.zones[self.zone_id])
                                break
                        if self.zone_id is not None:
                            break

                        # Check for click near closing segment (excluding vertices)
                        near_line, _ = is_point_near_line_segment(
                            (x, y), polygon[-1], polygon[0], self.line_threshold
                        )
                        if near_line:
                            self.zone_id = zone_id
                            self.current_polygon = list(self.zones[self.zone_id])
                            break

                    if not clicked_existing_point:
                        # If near a line, attempt to insert a point
                        insert_point_between_segments((x, y))
                        if self.zone_id is not None:
                            self.selected_point_index = self.find_closest_point_index(self.current_polygon, (x, y))
                            self.dragging_point = True
                            self.drawing = False

                    # If clicked near a vertex (and not already dragging), duplicate the vertex
                    if self.zone_id is not None and not self.dragging_point:
                        polygon = self.zones[self.zone_id]
                        for i, point in enumerate(polygon):
                            if np.linalg.norm(np.array(point) - np.array((x, y))) < self.selection_threshold:
                                self.current_polygon = list(polygon)
                                self.current_polygon.insert(i, point)  # Insert duplicate at the same index
                                self.zones[self.zone_id] = self.current_polygon
                                self.selected_point_index = i
                                self.dragging_point = True
                                self.drawing = False
                                break

            else:  # Not in editing mode
                if not self.drawing:
                    self.current_polygon = []  # Clear the polygon when starting a new one

                if self.drawing:
                    self.add_point_to_polygon((x, y), img)
                else:
                    insert_point_between_segments((x, y))
                    self.current_polygon.append((x, y))
                    self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            self.draw_zone_ids(img)
            self.draw_instructions(img)
            self.draw_polygon(img, self.current_polygon)
            self.draw_axes(img)

            if self.dragging_point and self.selected_point_index != -1 and self.zone_id is not None:
                self.current_polygon[self.selected_point_index] = (x, y)
                self.zones[self.zone_id] = self.current_polygon  # Update the correct zone
                self.draw_polygon(img, self.current_polygon)
            elif self.drawing:
                temp_polygon = self.current_polygon + [(x, y)]
                self.draw_polygon(img, temp_polygon)
            else:
                highlight_color = (0, 0, 255)
                highlight_point = select_nearest_point((x, y))
                if highlight_point != -1:
                    cv2.circle(img, self.ensure_point_format(self.current_polygon[highlight_point]), 7, highlight_color, -1)

            self.draw_polygon(img, self.current_polygon)
            cv2.imshow('Target Frame', img)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_point:
                self.dragging_point = False
                self.save_polygon()
                self.current_polygon = []

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self.editing:
                if self.drawing:
                    self.add_point_to_polygon((x, y), img)
                else:
                    if len(self.current_polygon) > 1:
                        self.save_polygon()
                    self.current_polygon = []
                    self.drawing = False

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.editing = not self.editing
            if self.editing:
                self.current_polygon = []
                self.drawing = False
                self.dragging_point = False
                self.selected_point_index = -1
                logging.info("Switched to Editing mode")
            else:
                self.current_polygon = []
                self.drawing = False
                self.zone_id = self.get_next_available_zone_id()
                logging.info(f"Switched to Drawing mode. New zone_id: {self.zone_id}")

        self.draw_polygon(img, self.current_polygon)
        cv2.imshow('Target Frame', img)

    def is_point_near_polygon_point(self, point: Tuple[int, int], polygon: List[Tuple[int, int]], threshold: float = None) -> bool:
        """Checks if a point is near any point of a polygon."""
        if threshold is None:
            threshold = self.selection_threshold
        for p in polygon:
            if np.linalg.norm(np.array(p) - np.array(point)) < threshold:
                return True
        return False

    def get_next_available_zone_id(self) -> int:
        """Generates the next available zone ID."""
        if len(self.zones) == 0:
            return 1
        return max(self.zones.keys()) + 1

    def find_closest_point_index(self, polygon: List[Tuple[int, int]], point: Tuple[int, int]) -> int:
        """Finds the index of the closest point in a polygon to a given point."""
        min_distance = float('inf')
        closest_index = -1
        for i, p in enumerate(polygon):
            distance = np.linalg.norm(np.array(p) - np.array(point))
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index

    def save_polygon(self):
        """Saves the current polygon to the mask positions."""
        if len(self.current_polygon) > 2:
            if self.current_polygon[0] == self.current_polygon[-1]:
                self.current_polygon.pop()

            # Update zones dictionary
            self.zones[self.zone_id] = self.current_polygon

            # Update the mask_positions list
            self.mask_positions = [entry for entry in self.mask_positions if entry['frame'] != self.frame_id]
            for zone_id, polygon in self.zones.items():
                new_entry = {'frame': self.frame_id, 'zone_id': zone_id, 'points': polygon}
                self.mask_positions.append(new_entry)
            self.undo_stack.append((self.zones.copy(), self.mask_positions.copy()))
            self.redo_stack.clear()

    def run(self) -> List[Dict[str, Any]]:
        """Runs the mask tool interface."""
        cv2.imshow('Target Frame', self.target_frame)
        cv2.setMouseCallback('Target Frame', self.draw_mask)

        logging.info("Edit the mask and use the following commands:")
        logging.info(" - 's' to save the mask")
        logging.info(" - 'u' to undo")
        logging.info(" - 'r' to redo")
        logging.info(" - 'e' to toggle editing mode")
        logging.info(" - 'q' to start detection")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Ensure the directory exists
                mask_dir = os.path.dirname(self.mask_json_path)
                os.makedirs(mask_dir, exist_ok=True)
                with open(self.mask_json_path, 'w') as f:
                    json.dump(self.mask_positions, f)
                logging.info(f"Masks and positions have been saved to {self.mask_json_path}.")
            elif key == ord('u'):
                if self.undo_stack:
                    self.zones, self.mask_positions = self.undo_stack.pop()
                    self.redo_stack.append((self.zones.copy(), self.mask_positions.copy()))
                    logging.info("Undo last action.")
            elif key == ord('r'):
                if self.redo_stack:
                    self.zones, self.mask_positions = self.redo_stack.pop()
                    self.undo_stack.append((self.zones.copy(), self.mask_positions.copy()))
                    logging.info("Redo last undone action.")
            elif key == ord('e'):
                self.editing = not self.editing
                if self.editing:
                    self.current_polygon = []
                    self.drawing = False
                    self.dragging_point = False
                    self.selected_point_index = -1
                    logging.info("Switched to Editing mode")
                else:
                    self.current_polygon = []
                    self.drawing = False
                    self.zone_id = self.get_next_available_zone_id()
                    logging.info(f"Switched to Drawing mode. New zone_id: {self.zone_id}")
            elif key == 27 or key == ord('q'):
                break

            # Redraw after each action to reflect changes
            img = self.target_frame.copy()
            self.draw_zone_ids(img)
            self.draw_instructions(img)
            self.draw_axes(img)
            cv2.imshow('Target Frame', img)
        cv2.destroyAllWindows()
        self.cap.release()
        return self.mask_positions

    def load_polygons_for_frame(self, frame_id: int):
        """Loads polygons from the JSON data for the specified frame."""
        self.zones.clear()
        frame_data = [entry for entry in self.mask_positions if entry['frame'] == frame_id]
        for entry in frame_data:
            self.zones[entry['zone_id']] = entry['points']

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

        # MQTT configuration
        self.emqx_host = config.get('emqx_host', 'localhost')
        self.emqx_port = config.get('emqx_port', 8883)
        self.emqx_username = config.get('emqx_username', '')
        self.emqx_password = config.get('emqx_password', '')
        self.emqx_topic = config.get('emqx_topic', 'detection_log')

        # Generate a unique client ID
        self.client_id = 'client-' + str(uuid.uuid4())

        # Initialize MQTT client with the unique client ID
        self.mqtt_client = mqtt_client.Client(
            client_id=self.client_id,
            protocol=mqtt_client.MQTTv311
        )

        # Set username and password
        if self.emqx_username and self.emqx_password:
            self.mqtt_client.username_pw_set(self.emqx_username, self.emqx_password)

        # Configure TLS/SSL
        try:
            # Path to the CA certificate
            ca_cert_path = config.get('ca_cert_path', 'emqxsl-ca.crt')

            self.mqtt_client.tls_set(
                ca_certs=ca_cert_path,
                certfile=None,
                keyfile=None,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLSv1_2,
                ciphers=None
            )
            self.mqtt_client.tls_insecure_set(False)  # Require valid certificate

            # Assign callbacks
            self.mqtt_client.on_connect = self.on_connect
            self.mqtt_client.on_publish = self.on_publish
            self.mqtt_client.on_log = self.on_log

            # Connect to the broker
            self.mqtt_client.connect(self.emqx_host, self.emqx_port)
            self.mqtt_client.loop_start()
            logging.info(f"Connecting to EMQX at {self.emqx_host}:{self.emqx_port} over TLS/SSL...")
        except Exception as e:
            logging.error(f"Could not connect to EMQX: {e}")
            return

    # Callback methods
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("MQTT client connected successfully.")
        else:
            logging.error(f"MQTT client failed to connect. Return code: {rc}")

    def on_publish(self, client, userdata, mid):
        logging.debug(f"Message {mid} published.")

    def on_log(self, client, userdata, level, buf):
        logging.debug(f"MQTT Log: {buf}")

    def load_zones_for_frame(self, frame_id: int):
        """Loads zones for the specified frame."""
        self.zones.clear()
        frame_data = [entry for entry in self.mask_positions if entry['frame'] == frame_id]
        for entry in frame_data:
            self.zones[entry['zone_id']] = np.array(entry['points'])

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

    def publish_detection(self, detection_entry: DetectionEntry):
        """Publishes detection events to the MQTT broker."""
        try:
            # Convert the dataclass to a dictionary and ensure JSON serialization
            serializable_entry = detection_entry.__dict__
            for key, value in serializable_entry.items():
                if isinstance(value, (np.integer, np.int32, np.int64)):
                    serializable_entry[key] = int(value)
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_entry[key] = float(value)
                elif value is None:
                    serializable_entry[key] = None

            # Publish the message and check the result
            result = self.mqtt_client.publish(self.emqx_topic, json.dumps(serializable_entry))
            status = result[0]
            if status == mqtt_client.MQTT_ERR_SUCCESS:
                logging.info(f"Sent `{serializable_entry}` to topic `{self.emqx_topic}`")
            else:
                logging.error(f"Failed to send message to topic {self.emqx_topic}")
        except Exception as e:
            logging.error(f"Error publishing to EMQX: {e}")

    def point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """Returns True if the point is inside the polygon using point-in-polygon test."""
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def is_contained_in_zone(self, bbox: np.ndarray, polygon: np.ndarray) -> bool:
        """Determine if the bounding box is contained in the zone polygon."""
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Check if at least one corner or the centroid is inside the polygon
        if any(self.point_in_polygon(corner, polygon) for corner in corners) or self.point_in_polygon(centroid, polygon):
            return True

        # Check for overlap between bounding box and zone polygon by constructing bounding box as a polygon
        bbox_poly = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.int32)
        intersection = cv2.intersectConvexConvex(bbox_poly, polygon.astype(np.float32))
        

        if intersection[1] is not None:
            intersection_area = cv2.contourArea(intersection[1])
            bbox_area = (x2 - x1) * (y2 - y1)

            # Consider it inside the zone if intersection area is significant and centroid is inside the polygon
            overlap_ratio = intersection_area / bbox_area
            centroid_in_polygon = cv2.pointPolygonTest(polygon, centroid, False) >= 0
            
            # Adjust overlap ratio and centroid check requirements as needed
            return overlap_ratio > 0.5 and centroid_in_polygon  # 50% overlap and centroid inside

        return False

    def track_intersections(self, video_path: str, frame_to_edit: int):
        """Tracks objects in the video and detects zone intersections using the new containment method."""
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

                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()
                else:
                    continue

                for i, (bbox, conf, class_id, track_id) in enumerate(
                        zip(boxes.xyxy, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), track_ids)):
                    bbox_np = bbox.cpu().numpy()
                    track_id = int(track_id)
                    x1, y1, x2, y2 = bbox_np.astype(int)

                    intersecting_zones = []

                    # Initialize tracked_objects entry if it doesn't exist
                    if track_id not in self.tracked_objects:
                        self.tracked_objects[track_id] = {
                            'class_id': int(class_id),
                            'zone_entries': []
                        }

                    for zone_id, polygon in self.zones.items():
                        contained = self.is_contained_in_zone(bbox_np, polygon)

                        zone_entry_exists = any(entry['zone_id'] == zone_id for entry in
                                                self.tracked_objects[track_id]['zone_entries'])
                        if contained and not zone_entry_exists:
                            entry = {
                                'zone_id': int(zone_id),
                                'first_seen': float(timestamp),
                                'last_seen': None
                            }
                            self.tracked_objects[track_id]['zone_entries'].append(entry)

                            detection_entry = DetectionEntry(
                                frame_id=frame_id,
                                object_id=track_id,
                                class_id=int(class_id),
                                confidence=float(conf),
                                zone_id=int(zone_id),
                                first_seen=float(timestamp),
                                event='enter'
                            )
                            self.detection_log.append(detection_entry)
                            self.save_detection_log()
                            self.publish_detection(detection_entry)

                        elif not contained and zone_entry_exists:
                            for entry in self.tracked_objects[track_id]['zone_entries']:
                                if entry['zone_id'] == zone_id and entry['last_seen'] is None:
                                    entry['last_seen'] = float(timestamp)
                                    detection_entry = DetectionEntry(
                                        frame_id=frame_id,
                                        object_id=track_id,
                                        class_id=int(class_id),
                                        confidence=float(conf),
                                        zone_id=int(zone_id),
                                        first_seen=float(entry['first_seen']),
                                        last_seen=float(timestamp),
                                        duration=float(timestamp - entry['first_seen']),
                                        event='exit'
                                    )
                                    self.detection_log.append(detection_entry)
                                    self.save_detection_log()
                                    self.publish_detection(detection_entry)

                        if contained:
                            intersecting_zones.append(zone_id)

                    # Time exceed event check
                    time_in_zone = 0
                    current_zone = None
                    for zone_entry in self.tracked_objects[track_id]['zone_entries']:
                        if zone_entry['last_seen'] is None:
                            current_time_in_zone = timestamp - zone_entry['first_seen']
                            if current_time_in_zone > time_in_zone:
                                time_in_zone = current_time_in_zone
                                current_zone = zone_entry['zone_id']
                            # Check if total duration exceeded and not yet logged
                            if 'threshold_logged' not in zone_entry and time_in_zone > self.total_duration:
                                detection_entry = DetectionEntry(
                                    frame_id=frame_id,
                                    object_id=track_id,
                                    class_id=int(class_id),
                                    confidence=float(conf),
                                    zone_id=int(current_zone),
                                    first_seen=float(zone_entry['first_seen']),
                                    duration=float(time_in_zone),
                                    event='time_exceeded'
                                )
                                self.detection_log.append(detection_entry)
                                self.save_detection_log()
                                self.publish_detection(detection_entry)
                                zone_entry['threshold_logged'] = True

                    # Visualization of bounding box and zone information
                    if intersecting_zones:
                        current_zone = intersecting_zones[0]
                        zone_entry = next(
                            (entry for entry in self.tracked_objects[track_id]['zone_entries']
                            if entry['zone_id'] == current_zone and entry['last_seen'] is None),
                            None
                        )

                        if zone_entry is not None:
                            time_in_zone = timestamp - zone_entry['first_seen']

                            # Determine color based on time in zone
                            if time_in_zone > self.time_thresholds[1]:
                                bbox_color = (0, 0, 255)  # Red
                            elif time_in_zone > self.time_thresholds[0]:
                                bbox_color = (0, 165, 255)  # Orange
                            else:
                                bbox_color = (0, 255, 0)  # Green

                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

                            # Draw object ID, time in zone, and zone info
                            label1 = f"ID: {track_id}"
                            label2 = f"Time: {time_in_zone:.1f}s Zone: {current_zone}"
                            cv2.putText(frame, label1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
                            cv2.putText(frame, label2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
                        else:
                            # Object is outside zones
                            bbox_color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                            label = f"ID: {track_id}"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

            if self.show_result:
                cv2.imshow('Zone Intersections', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

        cap.release()
        if self.show_result:
            cv2.destroyAllWindows()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logging.info("Disconnected from EMQX.")



    def save_detection_log(self):
        """Saves the detection log in JSON format."""
        detection_log_list = []
        for entry in self.detection_log:
            detection_log_list.append(entry.__dict__)

        # Save the list of dictionaries to a JSON file
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(detection_log_list, f, indent=4)

        logging.info(f"Detection log saved to {self.output_file}")

# Usage
if __name__ == "__main__":
    import os

    # Load configuration from config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # 1. Run MaskTool to define zones on the selected frame
    mask_tool = MaskTool(config)
    mask_positions = mask_tool.run()

    # 2. Ask the user whether to display images
    show_result_input = input("Do you want to display the result images? (y/n): ").lower()
    show_result = show_result_input == 'y'

    # 3. Initialize ZoneIntersectionTracker with the configuration and show_result parameter
    tracker = ZoneIntersectionTracker(config, show_result=show_result)
    tracker.track_intersections(config['video_source'], config['frame_to_edit'])