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

    def __init__(self, config: Dict[str, Any], show_result: bool = True, extract_image: bool = True):
        """Initializes the ZoneIntersectionTracker with configuration."""
        self.config = config
        self.detection_topic = f"{config.get('detection_topic', 'detection')}/{config.get('unique_id', 'default_id')}"
        self.heartbeat_topic = f"{config.get('heartbeat_topic', 'keep-alive')}/{config.get('unique_id', 'default_id')}"
        self.log_topic = config.get('log_topic', 'log')
        self.show_result = show_result
        self.extract_image = extract_image

        # Load rule configuration from rule.json.
        self.rule_config = self.load_rule_config()

        # Get rule-based settings with fallback to original config.
        # In __init__
        self.no_parking_duration = 0  # default value
        for rule in self.rule_config.get("rules", []):
            if rule.get("name", "").lower() == "no parking":
                duration = None
                if rule.get("rule_applied"):
                    try:
                        applied_record = rule["rule_applied"][0]
                        params = json.loads(applied_record.get("json_params", "{}"))
                        duration = params.get("duration")
                    except Exception as e:
                        logging.error(f"Error parsing no parking ruleApplied jsonParams: {e}")
                if duration is None:
                    try:
                        params = json.loads(rule.get("json_params", "{}"))
                        duration = params.get("duration")
                    except Exception as e:
                        logging.error(f"Error parsing no parking jsonParams: {e}")
                if duration is not None:
                    self.no_parking_duration = float(duration)
                break
        self.wrong_way_config = self.rule_config.get("wrong_way", config.get("wrong_way", {}))

        # Initialize time thresholds for bounding box color changes
        self.time_thresholds = [
            2 * self.no_parking_duration / 3,  # First threshold (e.g., orange color)
            self.no_parking_duration            # Second threshold (e.g., red color)
        ]

        self.model_path = config['model_path']
        self.mask_json_path = config['mask_json_path']
        self.unique_id = config.get('unique_id', 'default_id')
        self.edge_device_id = config.get('edge_device_id', 'default_edge_id')
        logging.info(f"ZoneIntersectionTracker initialized with edge_device_id: {self.unique_id}")

        self.tracker_config = config.get('tracker_config', 'bytetrack.yaml')
        if not self.tracker_config:
            logging.warning("tracker_config not found in config. Using default: 'bytetrack.yaml'")
            self.tracker_config = 'bytetrack.yaml'

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found.")
        if not os.path.exists(self.mask_json_path):
            raise FileNotFoundError(f"Mask positions file '{self.mask_json_path}' not found.")

        self.load_mask_positions()
        self.zones = {}
        self.arrows = {}
        self.detection_log = []
        self.tracked_objects = {}
        self.object_zone_timers = defaultdict(lambda: defaultdict(float))
        self.lock = threading.Lock()
        self.load_zones()
        self.load_arrows()

        # Output directory and file setup
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.script_dir, config.get('output_dir', 'output'))
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, config.get('output_file', 'detection_log.json'))

        # Device setup (GPU or CPU)
        if torch.cuda.is_available():
            self.device = 'cuda'
            logging.info("Using GPU for inference.")
        else:
            self.device = 'cpu'
            logging.info("Using CPU for inference.")

        # Load YOLO model
        self.model = YOLO(self.model_path).to(self.device)
        self.vehicle_class_ids = [2,5,7]  # Vehicle classes in COCO dataset, car only

        # MQTT configuration
        mqtt_config = {
            "mqtt_broker": config['mqtt_broker'],
            "mqtt_port": config['mqtt_port'],
            "mqtt_username": config['mqtt_username'],
            "mqtt_password": config['mqtt_password'],
            "ca_cert_path": config['ca_cert_path']
        }
        publisher_config = {
            "unique_id": self.unique_id,
            "edge_device_id": self.edge_device_id,
            "heartbeat_interval": config.get('heartbeat_interval', 60),
            "heartbeat_topic": self.heartbeat_topic,
            "detection_topic": self.detection_topic,
            "log_topic": self.log_topic
        }
        logging.info(f"ZoneIntersectionTracker initialized with edge_device_id: {self.edge_device_id}")
        self.mqtt_publisher = MqttPublisher({**mqtt_config, **publisher_config})
        self.mqtt_subscriber = MqttSubscriber(config, self.mqtt_publisher)

        # MQTT callbacks for marker management
        self.mqtt_subscriber.mqtt_client.message_callback_add(
            f'{self.unique_id}/marker/create', self.on_create_marker
        )
        self.mqtt_subscriber.mqtt_client.message_callback_add(
            f'{self.unique_id}/marker/update', self.on_update_marker
        )
        self.mqtt_subscriber.mqtt_client.message_callback_add(
            f'{self.unique_id}/marker/delete', self.on_delete_marker
        )
        
        self.mqtt_subscriber.mqtt_client.message_callback_add(
            f'{self.unique_id}/rule/create', self.on_create_rule_applied
        )
        self.mqtt_subscriber.mqtt_client.message_callback_add(
            f'{self.unique_id}/rule/update', self.on_update_rule_applied
        )
        self.mqtt_subscriber.mqtt_client.message_callback_add(
            f'{self.unique_id}/rule/delete', self.on_delete_rule_applied
        )
        
    def load_rule_config(self) -> Dict[str, Any]:
        """Loads rule configuration from the rule.json file."""
        rule_config_path = self.config.get('rule_config_path', 'traffic-guard/config/rule.json')
        if not os.path.exists(rule_config_path):
            logging.error(f"Rule config file not found: {rule_config_path}")
            return {}

        try:
            with open(rule_config_path, 'r') as f:
                rule_config = json.load(f)
                # logging.info("Rule configuration loaded successfully.")
                return rule_config
        except Exception as e:
            logging.error(f"Failed to load rule config: {e}")
            return {}        
        
    def get_rules_for_marker(self, marker_id: int, event: str) -> List[Dict[str, Any]]:
        """
        Retrieves all rules from rule.json applicable to a marker for a specific event.
        It searches for a rule whose name (in snake case) matches the event, and then
        returns all applied entries for that marker.
        """
        event_str = event.lower().replace('_', ' ')
        matched_rules = []
        for rule in self.rule_config.get('rules', []):
            rule_name = rule.get('name', '').lower().replace('_', ' ')
            if rule_name == event_str:
                for applied in rule.get('rule_applied', []):
                    if applied.get('marker_id') == marker_id:
                        rule_copy = rule.copy()
                        rule_copy['applied_id'] = applied  # attach applied rule details
                        matched_rules.append(rule_copy)
        return matched_rules

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
                    self.mqtt_publisher.publish_log(json.dumps({"error": message}))
                    return
    
                # Check for duplicate marker_id
                if any(entry['marker_id'] == data['marker_id'] for entry in self.mask_positions):
                    error_msg = f"Duplicate marker_id {data['marker_id']} detected. Creation aborted."
                    logging.error(error_msg)
                    self.mqtt_publisher.publish_log(json.dumps({"error": error_msg}))
                    return
    
                self.mask_positions.append(data)
                self.save_mask_positions()
                self.load_zones()
                self.load_arrows()
                logging.info(f"Created marker position: {data}")
                self.mqtt_publisher.publish_log(json.dumps({"message": "create complete", "marker_id": data['marker_id']}))
    
            except json.JSONDecodeError:
                error_msg = "Invalid JSON payload. Marker creation aborted."
                logging.error(error_msg)
                self.mqtt_publisher.publish_log(json.dumps({"error": error_msg}))
            except Exception as e:
                error_msg = f"Error creating marker position: {e}"
                logging.error(error_msg)
                self.mqtt_publisher.publish_log(json.dumps({"error": error_msg}))

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
        with self.lock:
            """Handles the update of an existing marker from an MQTT message."""
            try:
                data = json.loads(msg.payload.decode())
                is_valid, message = self._validate_payload_update(data)
                if not is_valid:
                    logging.error(f"Invalid payload received: {message}")
                    self.mqtt_publisher.publish_log(json.dumps({"error": message}))
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
                    self.mqtt_publisher.publish_log(json.dumps({"warning": warning_msg}))
                    return
                self.save_mask_positions()
                self.load_zones()
                self.load_arrows()
                logging.info(f"Updated marker position: {data}")
                self.mqtt_publisher.publish_log(json.dumps({"message": "update complete", "marker_id": data['marker_id']}))
            except json.JSONDecodeError:
                error_msg = "Invalid JSON payload. Marker update aborted."
                logging.error(error_msg)
                self.mqtt_publisher.publish_log(json.dumps({"error": error_msg}))
            except Exception as e:
                error_msg = f"Error updating marker position: {e}"
                logging.error(error_msg)
                self.mqtt_publisher.publish_log(json.dumps({"error": error_msg}))

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
        with self.lock:
            try:
                data = json.loads(msg.payload.decode())
                marker_id = data.get('marker_id')
    
                # Find the marker to delete
                marker_to_delete = next((marker for marker in self.mask_positions if marker.get('marker_id') == marker_id), None)
    
                if marker_to_delete:
                    self.mask_positions.remove(marker_to_delete)
                    self.save_mask_positions()
                    self.load_zones()
                    self.load_arrows()
                    logging.info(f"Deleted marker position: {data}")
                    self.mqtt_publisher.publish_log(json.dumps({"message": "delete complete", "marker_id": marker_id}))
                else:
                    warning_msg = "No matching marker found to delete."
                    logging.warning(warning_msg)
                    self.mqtt_publisher.publish_log(json.dumps({"warning": warning_msg}))
    
            except json.JSONDecodeError:
                error_msg = "Invalid JSON payload. Marker deletion aborted."
                logging.error(error_msg)
                self.mqtt_publisher.publish_log(json.dumps({"error": error_msg}))
            except Exception as e:
                error_msg = f"Error deleting marker position: {e}"
                logging.error(error_msg)
                self.mqtt_publisher.publish_log(json.dumps({"error": error_msg}))

    def on_create_rule_applied(self, client, userdata, msg):
        with self.lock:
            """Handles the creation of a new rule applied from an MQTT message."""
            self.rule_config = self.load_rule_config()
            # Optionally trigger other things like reloading zones/arrows if needed
            self.load_zones()
            self.load_arrows()
            logging.info("Rule config reloaded after create rule applied.")

    def on_update_rule_applied(self, client, userdata, msg):
        with self.lock:        
            """Handles the update of a rule applied from an MQTT message."""
            self.rule_config = self.load_rule_config()
            self.load_zones()
            self.load_arrows()
            logging.info("Rule config reloaded after update rule applied.")

    def on_delete_rule_applied(self, client, userdata, msg):
        with self.lock:
            """Handles the deletion of a rule applied from an MQTT message."""
            self.rule_config = self.load_rule_config()
            self.load_zones()
            self.load_arrows()
            logging.info("Rule config reloaded after delete rule applied.")

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
        """Draws zones on the frame and displays assigned rule text, including id_rule_applied."""
        for marker_id, polygon in self.zones.items():
            # Retrieve rule info for this zone from rule_config
            rule_info = None
            for rule in self.rule_config.get('rules', []):
                applied_list = rule.get("rule_applied", [])
                # Check if any applied rule relates to the current marker_id
                matching = [applied for applied in applied_list if applied.get("marker_id") == marker_id]
                if matching:
                    # Create a list of dictionaries with the rule name and applied id
                    if rule_info is None:
                        rule_info = []
                    for applied in matching:
                        rule_info.append({"applied_id": applied.get("applied_id"), "name": rule.get("name")})
            # Draw the zone polygon
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            # Handle rule_info display if available
            if rule_info:
                # If rule_info is a list, join all rule names
                if isinstance(rule_info, list):
                    text = f"ID: {marker_id} " + ", ".join([info.get("name", "N/A") for info in rule_info if info.get("applied_id")])
                else:
                    text = f"ID: {marker_id} " + rule_info.get("name", "N/A")
                pos = tuple(polygon[0]) if polygon.size > 0 and len(polygon[0]) >= 2 else (10, 10)
            else:
                text = f"ID: {marker_id}"
            centroid = np.mean(polygon, axis=0).astype(int)
            cv2.putText(frame, text, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def draw_arrows(self, frame: np.ndarray):
        """Draws arrows on the frame."""
        self.rule_config = self.load_rule_config()
        for marker_id, arrow_data in list(self.arrows.items()):
            polygon_points = arrow_data['polygon_points']
            line_points = arrow_data['line_points']
            color_polygon = (0, 255, 255)  # Yellow color for arrow zones
            color_line = (255, 0, 0)        # Blue color for arrows
            thickness = 2

            # Retrieve rule info for this arrow from rule_config
            rule_info = None
            for rule in self.rule_config.get('rules', []):
                applied_list = rule.get("rule_applied", [])
                # Check if any applied rule relates to the current marker_id
                matching = [applied for applied in applied_list if applied.get("marker_id") == marker_id]
                if matching:
                    # Create a list of dictionaries with the rule name and applied id
                    if rule_info is None:
                        rule_info = []
                    for applied in matching:
                        rule_info.append({"applied_id": applied.get("applied_id"), "name": rule.get("name")})

            if polygon_points.shape[0] >= 3:
                cv2.polylines(frame, [polygon_points.astype(np.int32)], isClosed=True, color=color_polygon, thickness=thickness)
                # Handle rule_info display if available
                if rule_info:
                    # If rule_info is a list, join all rule names
                    if isinstance(rule_info, list):
                        text = f"ID: {marker_id} " + ", ".join([info.get("name", "N/A") for info in rule_info if info.get("applied_id")])
                    else:
                        text = f"ID: {marker_id} " + rule_info.get("name", "N/A")
                else:
                    text = f"Arrow Zone {marker_id}"
                centroid = np.mean(polygon_points, axis=0).astype(int)
                cv2.putText(frame, text, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_polygon, thickness)
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

    def _get_object_event(self, track_id: int) -> List[str]:
        """
        Returns a list of unique events for an object based on detection_log.
        It first checks if the tracked object has a severe event (no_entry, no_parking, wrong_way)
        and then collects all associated events from the detection_log.
        """
        events = []
        obj = self.tracked_objects.get(track_id)
        if not obj:
            return ["unknown"]

        # Include the currently assigned severe event if present.
        current_event = obj.get('current_event') if isinstance(obj, dict) else getattr(obj, 'current_event', None)
        if current_event in ['no_entry', 'no_parking', 'wrong_way']:
            events.append(current_event)

        # Collect unique events from detection_log for this object.
        for entry in self.detection_log:
            # Retrieve object_id from entry either by dict.get or attribute.
            if isinstance(entry, dict):
                object_id = entry.get("object_id", None)
                ev = entry.get("event", "unknown")
            else:
                object_id = getattr(entry, "object_id", None)
                ev = getattr(entry, "event", "unknown")
            if object_id == track_id and ev not in events:
                events.append(ev)

        return events if events else None

    def xyxy_to_xywh(self, x1, y1, x2, y2, iw, ih):
        x_temp = ((x1 + x2) / 2)/iw
        y_temp = ((y1 + y2) / 2)/ih
        w_temp = (abs(x1 - x2))/iw
        h_temp = (abs(y1 - y2))/ih
        return [float(x_temp), float(y_temp), float(w_temp), float(h_temp)]

    def track_intersections(self, video_path: str, frame_to_edit: int):

        # Check if the input is a YouTube URL
        if video_path.startswith(('http://', 'https://')):
            try:
                import yt_dlp
            except ImportError:
                raise ImportError("Please install yt-dlp to handle YouTube URLs: pip install yt-dlp")

            # Extract the best video stream URL using yt-dlp
            ydl_opts = {
                'format': 'bestvideo[ext=ts]/best',
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

        # Resize frame to fit screen
        frame_height, frame_width = frame.shape[:2]

        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            results = self.model.track(frame, verbose=False, persist=True, stream=True, tracker=self.tracker_config, device=self.device, classes=self.vehicle_class_ids)

            self.draw_zones(frame)
            self.draw_arrows(frame)

            for result in results:
                boxes = result.boxes

                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()
                else:
                    continue

                for i, (bbox, conf, class_id, track_id) in enumerate(
                        zip(boxes.xyxy, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), track_ids)):
                    if int(class_id) not in self.vehicle_class_ids:
                        continue

                    bbox_np = bbox.cpu().numpy()
                    track_id = int(track_id)

                    # Convert xyxy to xywh immediately
                    bbox_xywh = self.xyxy_to_xywh(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]), frame_width, frame_height)

                    max_iou = 0
                    intersecting_marker_id = None

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

                        marker_entry_exists = any(entry['marker_id'] == marker_id for entry in
                                                self.tracked_objects[track_id]['marker_entries'])
                        if intersects and not marker_entry_exists:
                            entry = {
                                'marker_id': int(marker_id),
                                'first_seen': float(timestamp),
                                'last_seen': None
                            }
                            self.tracked_objects[track_id]['marker_entries'].append(entry)

                            detection_entry = DetectionEntry(
                                object_id=track_id,
                                class_id=int(class_id),
                                confidence=float(conf),
                                marker_id=int(marker_id),
                                first_seen=float(entry['first_seen']),
                                last_seen=float(timestamp),
                                duration=None,
                                event='enter',
                                bbox=bbox_xywh  # Use xywh format
                            )
                            self.detection_log.append(detection_entry)
                            self.save_detection_log()
                            # self.mqtt_publisher.send_incident(detection_entry)

                        elif not intersects and marker_entry_exists:
                            for entry in self.tracked_objects[track_id]['marker_entries']:
                                if entry['marker_id'] == marker_id and entry['last_seen'] is None:
                                    entry['last_seen'] = float(timestamp)
                                    detection_entry = DetectionEntry(
                                        object_id=track_id,
                                        class_id=int(class_id),
                                        confidence=float(conf),
                                        marker_id=int(marker_id),
                                        first_seen=float(entry['first_seen']),
                                        last_seen=float(timestamp),
                                        duration=float(timestamp - entry['first_seen']),
                                        event='exit',
                                        bbox=bbox_xywh  # Use xywh format
                                    )
                                    self.detection_log.append(detection_entry)
                                    self.save_detection_log()
                                    # self.mqtt_publisher.send_incident(detection_entry)

                    # Handle movement markers
                    for marker_id, arrow_data in self.arrows.items():
                        polygon_points = arrow_data['polygon_points']
                        line_points = arrow_data['line_points']
                        intersects_movement, iou_movement = self.intersects(bbox_np, polygon_points)
                        if iou_movement > max_iou:
                            max_iou = iou_movement
                            intersecting_marker_id = marker_id if intersects_movement else None

                        marker_entry_exists = any(entry['marker_id'] == marker_id for entry in
                                                self.tracked_objects[track_id]['marker_entries'])
                        if intersects_movement and not marker_entry_exists:
                            entry = {
                                    'marker_id': int(marker_id),
                                    'type': 'movement',  # Add type
                                    'first_seen': float(timestamp),
                                    'last_seen': None,
                                    'trajectory': []  # Initialize trajectory
                                }
                            self.tracked_objects[track_id]['marker_entries'].append(entry)

                            detection_entry = DetectionEntry(
                                object_id=track_id,
                                class_id=int(class_id),
                                confidence=float(conf),
                                marker_id=int(marker_id),
                                first_seen=float(entry['first_seen']),
                                last_seen=float(timestamp),
                                duration=None,
                                event='enter_movement',
                                bbox=bbox_xywh  # Use xywh format
                            )
                            self.detection_log.append(detection_entry)
                            self.save_detection_log()
                            # self.mqtt_publisher.send_incident(detection_entry)

                        elif intersects_movement:
                            for entry in self.tracked_objects[track_id]['marker_entries']:
                                if entry['marker_id'] == marker_id and entry['last_seen'] is None:
                                    x_center = (bbox_np[0] + bbox_np[2]) / 2
                                    y_center = (bbox_np[1] + bbox_np[3]) / 2
                                    entry['trajectory'].append((x_center, y_center))
                                    if len(entry['trajectory']) > 10:
                                        entry['trajectory'] = entry['trajectory'][-10:]

                        elif not intersects_movement and marker_entry_exists:
                            for entry in self.tracked_objects[track_id]['marker_entries']:
                                if entry['marker_id'] == marker_id and entry['last_seen'] is None:
                                    entry['last_seen'] = float(timestamp)
                                    detection_entry = DetectionEntry(
                                        object_id=track_id,
                                        class_id=int(class_id),
                                        confidence=float(conf),
                                        marker_id=int(marker_id),
                                        first_seen=float(entry['first_seen']),
                                        last_seen=float(timestamp),
                                        duration=float(timestamp - entry['first_seen']),
                                        event='exit_movement',
                                        bbox=bbox_xywh  # Use xywh format
                                    )
                                    self.detection_log.append(detection_entry)
                                    self.save_detection_log()
                                    # self.mqtt_publisher.send_incident(detection_entry)

                    # Check for no_parking and no_entry events
                    self._check_no_parking(track_id, class_id, conf, bbox_np, timestamp, frame_width, frame_height)
                    self._check_no_entry(track_id, class_id, conf, bbox_np, timestamp, frame_width, frame_height)
                    self._check_wrong_way(track_id, class_id, conf, bbox_np, timestamp, frame_width, frame_height)

                    # Determine color based on time in zone
                    time_in_zone = self._get_time_in_zone(track_id, timestamp)
                    bbox_color = self._get_bbox_color(time_in_zone)
                    text_color = (0, 0, 255)
                    # Draw bounding box
                    x1, y1, x2, y2 = bbox_np.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

                    # Draw object ID, IoU, time in zone, and marker info
                    event = self._get_object_event(track_id)
                    label1 = f"ID: {track_id} IoU: {max_iou:.2f} Event: {event}"
                    marker_info = intersecting_marker_id if intersecting_marker_id else 'N/A'
                    if isinstance(intersecting_marker_id, list):
                        marker_info = ', '.join(map(str, intersecting_marker_id))
                    label2 = f"Time: {time_in_zone:.1f}s Marker: {marker_info}"
                    cv2.putText(frame, label1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    cv2.putText(frame, label2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

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

        self.mqtt_publisher.mqtt_handler.disconnect()
        logging.info("Disconnected from EMQX.")
        
    def _check_no_parking(self, track_id: int, class_id: int, conf: float, bbox_np: np.ndarray, timestamp: float, frame_width, frame_height):
        for marker_entry in self.tracked_objects[track_id]['marker_entries']:
            marker_id = marker_entry.get('marker_id')
            if marker_id is None:
                continue
    
            # Retrieve rule info for "no parking" based on marker_id
            rules = self.get_rules_for_marker(marker_id, 'no parking')
            if not rules:
                logging.debug(f"No parking rules found for marker {marker_id}.")
                continue
        
            for rule in rules:
                # Check if an applied rule is present and use its json_params; otherwise, use rule-level params.
                applied = rule.get("applied_id")
                if applied and applied.get("json_params"):
                    try:
                        params = json.loads(applied.get("json_params"))
                        parking_duration_threshold = float(params.get("duration", self.no_parking_duration))
                    except Exception as e:
                        logging.error(f"Error parsing applied json_params: {e}")
                        parking_duration_threshold = self.no_parking_duration
                else:
                    try:
                        params = json.loads(rule.get("json_params", "{}"))
                        parking_duration_threshold = float(params.get("duration", self.no_parking_duration))
                    except Exception as e:
                        logging.error(f"Error parsing rule json_params: {e}")
                        continue
    
                # Optionally remove the last_seen check if you want to trigger the event after exit as well
                if marker_entry['last_seen'] is None:
                    time_in_zone = self._get_time_in_zone(track_id, timestamp)
                    logging.debug(f"Track {track_id} in marker {marker_id}: time_in_zone={time_in_zone}, threshold={parking_duration_threshold}")
                    if time_in_zone > parking_duration_threshold and 'threshold_logged' not in marker_entry:
                        logging.info(f"Object {track_id} exceeded no_parking duration in marker {marker_id}")
                        bbox_xywh = self.xyxy_to_xywh(float(bbox_np[0]),
                                                       float(bbox_np[1]),
                                                       float(bbox_np[2]),
                                                       float(bbox_np[3]),
                                                       frame_width,
                                                       frame_height)
                        detection_entry = DetectionEntry(
                            object_id=track_id,
                            class_id=int(class_id),
                            confidence=float(conf),
                            marker_id=int(marker_id),
                            id_rule_applied=applied["applied_id"] if applied else None,
                            first_seen=float(marker_entry['first_seen']),
                            last_seen=float(timestamp),
                            duration=float(time_in_zone),
                            event='no_parking',
                            bbox=bbox_xywh
                        )
                        self.detection_log.append(detection_entry)
                        self.save_detection_log()
                        self.mqtt_publisher.send_incident(detection_entry)
                        marker_entry['threshold_logged'] = True
                    
    def _check_no_entry(self, track_id: int, class_id: int, conf: float, bbox_np: np.ndarray,
                            timestamp: float, frame_width, frame_height):
            """Checks if an object is in a no_entry zone using marker IDs from rule.json and prevents duplicate MQTT messages.
            This updated version supports multiple applied rules per marker.
            """
            # Build set of marker IDs registered as no_entry zones from rule.json
            no_entry_zone_ids = set()
            for rule in self.rule_config.get('rules', []):
                if rule.get("name", "").lower() == "no entry":
                    for applied in rule.get("rule_applied", []):
                        marker_id = applied.get("marker_id")
                        if marker_id is not None:
                            no_entry_zone_ids.add(marker_id)
    
            # Iterate over tracked marker entries for the object
            for marker_entry in self.tracked_objects[track_id].get('marker_entries', []):
                marker_id = marker_entry.get('marker_id')
                if marker_id is None:
                    continue
                # Process only if marker_id is in no_entry zones,
                # event hasn't been logged yet and no MQTT message has been sent.
                if marker_id in no_entry_zone_ids and marker_entry.get('last_seen') is None \
                and not marker_entry.get('mqtt_sent', False):
                    logging.info(f"Object {track_id} is in a no_entry zone: {marker_id}")
                    bbox_xywh = self.xyxy_to_xywh(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]),
                                                float(bbox_np[3]), frame_width, frame_height)
                    # Get all matching no entry rules for the marker
                    rules = self.get_rules_for_marker(marker_id, 'no entry')
                    if not rules:
                        continue
                    for rule in rules:
                        for applied in rule.get("rule_applied", []):
                            if applied.get("marker_id") == marker_id:
                                detection_entry = DetectionEntry(
                                    object_id=track_id,
                                    class_id=int(class_id),
                                    confidence=float(conf),
                                    marker_id=int(marker_id),
                                    id_rule_applied=applied.get("applied_id"),
                                    first_seen=float(marker_entry['first_seen']),  # Use the initial entry time
                                    last_seen=float(timestamp),
                                    duration=None,
                                    event='no_entry',
                                    bbox=bbox_xywh
                                )
                                self.detection_log.append(detection_entry)
                                self.save_detection_log()
                                self.mqtt_publisher.send_incident(detection_entry)
                                # Mark this entry to prevent duplicate MQTT messages for this marker entry
                                marker_entry['mqtt_sent'] = True
                
    def calculate_direction(self, start_point, end_point):
        """Calculate the direction vector from start to end point."""
        direction = np.array(end_point) - np.array(start_point)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return direction  # Avoid division by zero
        return direction / norm

    def is_wrong_way(self, car_trajectory, polyline):
        """
        Determine if a car is going the wrong way by comparing its trajectory to the polyline direction.
        Returns 'left', 'right', 'opposite', or None if no violation.
        """
        if len(polyline) < 2 or len(car_trajectory) < 2:
            return None  # Not enough points

        # Calculate arrow direction from polyline points
        arrow_start = np.array(polyline[0])
        arrow_end = np.array(polyline[-1])
        arrow_vector = arrow_end - arrow_start
        arrow_norm = np.linalg.norm(arrow_vector)
        if arrow_norm == 0:
            return None  # Invalid arrow direction
        arrow_dir = arrow_vector / arrow_norm

        # Calculate vehicle direction from trajectory
        vehicle_start = np.array(car_trajectory[0])
        vehicle_end = np.array(car_trajectory[-1])
        vehicle_vector = vehicle_end - vehicle_start
        vehicle_norm = np.linalg.norm(vehicle_vector)
        if vehicle_norm == 0:
            return None  # Vehicle not moving
        vehicle_dir = vehicle_vector / vehicle_norm

        # Calculate dot and cross products
        dot = np.dot(arrow_dir, vehicle_dir)
        # cross = np.cross(arrow_dir, vehicle_dir)

        # Check direction violations
        if dot < -0.7:  # ~135 degrees threshold for opposite direction
            return 'opposite'
        # elif dot < 0.7:  # ~45 degrees threshold for side directions
        #     if cross > 0:
        #         return 'left'
        #     else:
        #         return 'right'
        return None

    def _check_wrong_way(self, track_id: int, class_id: int, conf: float, bbox_np: np.ndarray, timestamp: float, frame_width, frame_height):
        """Checks if a vehicle is moving in the wrong direction within a movement zone."""
        if self.tracked_objects[track_id].get('wrong_way_logged', False):
            return

        for marker_entry in self.tracked_objects[track_id]['marker_entries']:
            if marker_entry.get('type') == 'movement' and marker_entry['last_seen'] is None:
                marker_id = marker_entry['marker_id']
                if marker_id is None:
                    continue

                # Retrieve rule info for "wrong way" based on marker_id
                rules = self.get_rules_for_marker(marker_id, 'wrong way')
                if not rules:
                    continue  # Skip if no rule applies to this marker_id

                arrow_data = self.arrows.get(marker_id)
                if not arrow_data or 'line_points' not in arrow_data:
                    continue

                line_points = arrow_data['line_points']
                if len(line_points) < 2:
                    continue

                trajectory = marker_entry.get('trajectory', [])
                if len(trajectory) < 2:
                    continue

                violation = self.is_wrong_way(trajectory, line_points)
                if violation:
                    event_type = 'wrong_way'
                    logging.info(f"Vehicle {track_id} violated wrong way rule in marker {marker_id}: {violation}")
                    bbox_xywh = self.xyxy_to_xywh(float(bbox_np[0]), float(bbox_np[1]),
                                                float(bbox_np[2]), float(bbox_np[3]),
                                                frame_width, frame_height)
                    for rule in rules:
                        detection_entry = DetectionEntry(
                            object_id=track_id,
                            class_id=int(class_id),
                            confidence=float(conf),
                            marker_id=int(marker_id),
                            id_rule_applied = rule.get("applied_id")["applied_id"] if isinstance(rule.get("applied_id"), dict) else rule.get("applied_id"),
                            first_seen=float(marker_entry['first_seen']),
                            last_seen=float(timestamp),
                            duration=None,
                            event=event_type,
                            bbox=bbox_xywh,
                        )
                        self.detection_log.append(detection_entry)
                        self.save_detection_log()
                        self.mqtt_publisher.send_incident(detection_entry)
                    # Mark the track_id so we don't log wrong_way repeatedly
                    self.tracked_objects[track_id]['wrong_way_logged'] = True

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