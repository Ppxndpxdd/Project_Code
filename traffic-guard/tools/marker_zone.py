import cv2
import json
import logging
import numpy as np
import os
from typing import Any, Dict, List, Tuple

class MarkerZone:
    """Interactive tool for defining both polygonal zones and arrows on a video frame."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.video_source = config['video_source']
        self.frame_to_edit = config['frame_to_edit']
        self.mask_json_path = config.get('mask_json_path', None)
        self.cap = cv2.VideoCapture(self.video_source)
        self.drawing = False
        self.current_tool = 'zone'  # Default tool
        self.arrow_points = []
        self.current_polygon: List[Tuple[int, int]] = []
        self.markers: Dict[int, Dict[str, Any]] = {}
        self.marker_id = 1

        # Combined mask positions loaded from file
        self.mask_positions = []

        # Ensure mask_json_path is defined
        if not self.mask_json_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.mask_json_path = os.path.join(script_dir, '..', 'config', 'marker_positions.json')

        # Load if exists
        if os.path.exists(self.mask_json_path):
            try:
                with open(self.mask_json_path, 'r') as f:
                    self.mask_positions = json.load(f)
            except Exception as e:
                logging.error(f"Error loading mask positions: {e}")

        if not self.cap.isOpened():
            logging.error(f"Cannot open video source '{self.video_source}'.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_to_edit)
        ret, self.target_frame = self.cap.read()
        if not ret:
            logging.error("Could not read the frame.")
            return

        # Initialize dictionaries
        self.markers = {}
        self.zones = {}
        self.arrow_zones = {}
        self.arrows = {}
        self.load_markers()

        # Visual properties
        self.highlight_radius = config.get('highlight_radius', 10)
        self.point_radius = config.get('point_radius', 5)
        self.line_threshold = config.get('line_threshold', 10)
        self.selection_threshold = config.get('selection_threshold', 10)

        self.undo_stack = []
        self.redo_stack = []

    def load_markers(self):
        """Loads markers from mask_positions.json and populates zones and arrows."""
        self.markers.clear()
        self.zones.clear()
        self.arrow_zones.clear()
        self.arrows.clear()
        for entry in self.mask_positions:
            marker_id = entry.get('marker_id')
            marker_type = entry.get('type')
            if marker_id and marker_type:
                self.markers[marker_id] = entry
                if marker_type == 'zone':
                    self.zones[marker_id] = entry['points']
                elif marker_type == 'movement':
                    self.arrow_zones[marker_id] = entry['polygon_points']
                    self.arrows[marker_id] = entry['line_points']
                else:
                    logging.warning(f"Unknown marker type '{marker_type}' for marker_id={marker_id}")

    def get_next_marker_id(self) -> int:
        if len(self.markers) == 0:
            return 1
        return max(self.markers.keys()) + 1

    def run(self) -> List[Dict[str, Any]]: 
        cv2.imshow('Target Frame', self.target_frame)
        cv2.setMouseCallback('Target Frame', self.mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('z'):
                self.current_tool = 'zone'
                logging.info("Switched to Zone tool")
            elif key == ord('a'):
                self.current_tool = 'arrow_zone'
                logging.info("Switched to Arrow Zone tool")
            elif key == ord('m'):
                self.current_tool = 'arrow'
                logging.info("Switched to Arrow tool")
            elif key == ord('s'):
                self.save_mask_positions()
            elif key == ord('u'):
                # Handle undo
                self.handle_undo()
            elif key == ord('r'):
                # Handle redo
                self.handle_redo()
            elif key == 27 or key == ord('q'):
                break

            img = self.target_frame.copy()
            self.draw_marker_ids(img)
            self.draw_instructions(img)
            self.draw_axes(img)
            cv2.imshow('Target Frame', img)

        cv2.destroyAllWindows()
        self.cap.release()
        return self.mask_positions

    def mouse_callback(self, event, x, y, flags, param):
        img = self.target_frame.copy()

        if self.current_tool == 'zone':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.drawing:
                    self.current_polygon = []
                    self.drawing = True
                self.current_polygon.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_polygon = self.current_polygon + [(x, y)]
                self.draw_polygon(img, temp_polygon, (0, 255, 0))
                cv2.imshow('Target Frame', img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                if len(self.current_polygon) > 2:
                    # Close polygon
                    self.current_polygon.append(self.current_polygon[0])
                    marker_id = self.get_next_marker_id()
                    self.zones[marker_id] = self.current_polygon.copy()
                    self.markers[marker_id] = {
                        "marker_id": marker_id,
                        "type": "zone",
                        "status": "activate",
                        "points": self.current_polygon
                    }
                    self.mask_positions.append({
                        "marker_id": marker_id,
                        "type": "zone",
                        "status": "activate",
                        "points": self.current_polygon
                    })
                    logging.info(f"Added Zone marker with marker_id={marker_id}")
                self.current_polygon = []
                self.save_mask_positions()

        elif self.current_tool == 'arrow_zone':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.drawing:
                    self.current_polygon = []
                    self.drawing = True
                self.current_polygon.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_polygon = self.current_polygon + [(x, y)]
                self.draw_polygon(img, temp_polygon, (0, 255, 255))
                cv2.imshow('Target Frame', img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                if len(self.current_polygon) > 2:
                    # Close polygon
                    self.current_polygon.append(self.current_polygon[0])
                    marker_id = self.get_next_marker_id()
                    self.arrow_zones[marker_id] = self.current_polygon.copy()
                    self.markers[marker_id] = {
                        "marker_id": marker_id,
                        "type": "movement",
                        "status": "activate",
                        "polygon_points": self.current_polygon,
                        "line_points": []
                    }
                    self.mask_positions.append({
                        "marker_id": marker_id,
                        "type": "movement",
                        "status": "activate",
                        "polygon_points": self.current_polygon,
                        "line_points": []
                    })
                    logging.info(f"Added Arrow Zone marker with marker_id={marker_id}")
                self.current_polygon = []
                self.save_mask_positions()

        elif self.current_tool == 'arrow':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.drawing:
                    self.arrow_points = []
                    self.drawing = True
                self.arrow_points.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_arrow = self.arrow_points + [(x, y)]
                self.draw_multipoint_arrow(img, temp_arrow, (255, 0, 0))
                cv2.imshow("Target Frame", img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                if len(self.arrow_points) > 1:
                    # Ensure an arrow zone is defined before defining an arrow
                    if not self.arrow_zones:
                        logging.error("Define an arrow zone before defining an arrow.")
                        return
                    marker_id = self.get_next_marker_id()
                    # Use the last defined arrow zone for the arrow
                    last_marker_id = max(self.arrow_zones.keys())
                    last_arrow_zone = self.arrow_zones[last_marker_id]
                    self.arrows[marker_id] = self.arrow_points.copy()
                    self.markers[last_marker_id]['line_points'] = self.arrow_points
                    self.mask_positions = [m for m in self.mask_positions if m['marker_id'] != last_marker_id]
                    self.mask_positions.append(self.markers[last_marker_id])
                    logging.info(f"Added Movement marker with marker_id={last_marker_id}")
                self.arrow_points = []
                self.save_mask_positions()

        # Draw everything
        self.draw_marker_ids(img)
        self.draw_instructions(img)
        self.draw_axes(img)
        cv2.imshow('Target Frame', img)

    def draw_marker_ids(self, img: np.ndarray):
        """Draws zones and arrows with their respective marker IDs on the image."""
        # Draw Zones
        for marker_id, poly in self.zones.items():
            self.draw_polygon(img, poly, (0, 255, 0))
            if len(poly) > 2:
                pts = np.array(poly, np.float32)
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, f"Marker {marker_id} (Zone)", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Arrow Zones and Arrows
        for marker_id, marker_data in self.markers.items():
            if marker_data['type'] == 'movement':
                polygon_points = marker_data['polygon_points']
                line_points = marker_data['line_points']
                self.draw_polygon(img, polygon_points, (0, 255, 255))
                self.draw_multipoint_arrow(img, line_points, (255, 0, 0))
                if len(line_points) > 1:
                    # Display the marker_id near the start point
                    start_point = line_points[0]
                    cv2.putText(img, f"Marker {marker_id} (Movement)", start_point,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def draw_instructions(self, img: np.ndarray):
        lines = [
            "Keys:",
            " Z = Zone tool",
            " A = Arrow Zone tool",
            " M = Arrow tool",
            " S = Save",
            " U = Undo",
            " R = Redo",
            " Q or ESC = Quit"
        ]
        for i, txt in enumerate(lines):
            cv2.putText(img, txt, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_axes(self, img: np.ndarray):
        h, w = img.shape[:2]
        cv2.line(img, (0, h - 1), (w, h - 1), (255, 0, 0), 2)
        cv2.putText(img, "X", (w - 20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.line(img, (0, 0), (0, h), (0, 255, 0), 2)
        cv2.putText(img, "Y", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_polygon(self, img: np.ndarray, polygon: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0)):
        if len(polygon) > 0:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
            for p in polygon:
                cv2.circle(img, (int(p[0]), int(p[1])), self.point_radius, color, -1)

    def draw_multipoint_arrow(self, img: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, int, int] = (255, 0, 0)):
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            cv2.arrowedLine(img, points[i], points[i + 1], color, thickness=2, tipLength=0.05)

    def handle_undo(self):
        if not self.undo_stack:
            logging.info("Nothing to undo.")
            return
        last_action = self.undo_stack.pop()
        self.redo_stack.append(last_action)
        marker_id = last_action['marker_id']
        marker_type = last_action['type']
        if marker_type == 'zone':
            self.zones.pop(marker_id, None)
        elif marker_type == 'movement':
            self.arrow_zones.pop(marker_id, None)
            self.arrows.pop(marker_id, None)
        self.markers.pop(marker_id, None)
        self.mask_positions = [m for m in self.mask_positions if m['marker_id'] != marker_id]
        self.save_mask_positions()
        logging.info(f"Undid action: Removed marker_id={marker_id}")

    def handle_redo(self):
        if not self.redo_stack:
            logging.info("Nothing to redo.")
            return
        action = self.redo_stack.pop()
        marker_id = action['marker_id']
        marker_type = action['type']
        if marker_type == 'zone':
            self.zones[marker_id] = action['points']
        elif marker_type == 'movement':
            self.arrow_zones[marker_id] = action['polygon_points']
            self.arrows[marker_id] = action['line_points']
        self.markers[marker_id] = action
        self.mask_positions.append(action)
        self.undo_stack.append(action)
        self.save_mask_positions()
        logging.info(f"Redid action: Added marker_id={marker_id}")

    def save_mask_positions(self):
        with open(self.mask_json_path, 'w') as f:
            json.dump(self.mask_positions, f, indent=4)
        logging.info("Mask positions saved.")