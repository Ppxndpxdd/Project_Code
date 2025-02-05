from typing import Any, Dict, List, Tuple
import cv2
import os
import json
import logging
import numpy as np
import screeninfo
import yt_dlp
import copy

class MarkerZone:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.video_source = config['video_source']
        self.frame_to_edit = config['frame_to_edit']
        self.mask_json_path = config.get('mask_json_path', None)

        # Get screen dimensions
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height

        # Handle YouTube URLs
        if self.video_source.startswith(('http://', 'https://')):
            ydl_opts = {'format': 'bestvideo[ext=mp4]/best', 'quiet': True}
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.video_source, download=False)
                    self.video_source = info['url']
            except Exception as e:
                logging.error(f"Failed to process YouTube URL: {e}")
                self.video_source = None

        self.cap = cv2.VideoCapture(self.video_source) if self.video_source else None
        self.drawing = False
        self.current_tool = 'zone'  # zone, arrow_zone, arrow
        self.arrow_points = []
        self.current_polygon: List[Tuple[int, int]] = []
        self.markers: Dict[int, Dict[str, Any]] = {}
        self.marker_id = 1
        self.target_frame = None

        # For rule assignment and marker selection
        self.selected_marker_id = None

        # Mode: "draw" uses drawing callbacks, "select" uses selection callbacks
        self.mode = "draw"

        # Load mask positions
        if not self.mask_json_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.mask_json_path = os.path.join(script_dir, '..', 'config', 'marker_positions.json')
        self.mask_positions = []
        if os.path.exists(self.mask_json_path):
            try:
                with open(self.mask_json_path, 'r') as f:
                    self.mask_positions = json.load(f)
            except Exception as e:
                logging.error(f"Error loading mask positions: {e}")

        # Validate video capture and frame
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_to_edit)
            ret, self.target_frame = self.cap.read()
            if not ret:
                logging.error("Could not read the frame.")
            else:
                self.target_frame = self.resize_frame_to_screen(self.target_frame)
        else:
            logging.error(f"Failed to open video source: {self.video_source}")

        # Initialize marker data structures
        self.zones = {}
        self.arrow_zones = {}
        self.arrows = {}
        self.load_markers()

        # Visual properties
        self.highlight_radius = config.get('highlight_radius', 10)
        self.point_radius = config.get('point_radius', 5)
        self.undo_stack = []
        self.redo_stack = []

    def resize_frame_to_screen(self, frame: np.ndarray) -> np.ndarray:
        """Resizes the frame to fit the screen while maintaining aspect ratio."""
        frame_height, frame_width = frame.shape[:2]
        scale_width = self.screen_width / frame_width
        scale_height = self.screen_height / frame_height
        scale = min(scale_width, scale_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        return cv2.resize(frame, (new_width, new_height))

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
                    self.zones[marker_id] = entry.get('points', [])
                elif marker_type == 'movement':
                    self.arrow_zones[marker_id] = entry.get('polygon_points', [])
                    self.arrows[marker_id] = entry.get('line_points', [])
                else:
                    logging.warning(f"Unknown marker type: {marker_type}")

    def get_next_marker_id(self) -> int:
        if not self.markers:
            return 1
        return max(self.markers.keys()) + 1

    def handle_selection(self, x: int, y: int):
        """Selects a marker if the clicked point is near any marker."""
        tolerance = 10
        # Check zone markers
        for marker_id, marker_data in self.markers.items():
            if marker_data.get('type') == 'zone':
                pts = self.zones.get(marker_id, [])
                if pts:
                    contour = np.array(pts, dtype=np.float32)
                    dist = cv2.pointPolygonTest(contour, (x, y), True)
                    if dist >= -tolerance:
                        self.selected_marker_id = marker_id
                        logging.info(f"Selected zone marker {marker_id}")
                        return
                    pts_array = np.array(pts, dtype=np.float32)
                    M = cv2.moments(pts_array)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        if np.hypot(cx - x, cy - y) < tolerance:
                            self.selected_marker_id = marker_id
                            logging.info(f"Selected zone marker {marker_id}")
                            return
        # Check movement markers
        for marker_id, marker_data in self.markers.items():
            if marker_data.get('type') == 'movement':
                pts = marker_data.get('polygon_points', [])
                if pts:
                    contour = np.array(pts, dtype=np.float32)
                    dist = cv2.pointPolygonTest(contour, (x, y), True)
                    if dist >= -tolerance:
                        self.selected_marker_id = marker_id
                        logging.info(f"Selected movement marker {marker_id}")
                        return
                pts = marker_data.get('line_points', [])
                if pts:
                    avg_x = sum(p[0] for p in pts) / len(pts)
                    avg_y = sum(p[1] for p in pts) / len(pts)
                    if np.hypot(avg_x - x, avg_y - y) < tolerance:
                        self.selected_marker_id = marker_id
                        logging.info(f"Selected movement marker {marker_id}")
                        return
        self.selected_marker_id = None
        logging.info("No marker selected.")

    def draw_selected_marker_info(self, img: np.ndarray):
        """Draws the selected marker's information on the top right of the screen."""
        if self.selected_marker_id is not None and self.selected_marker_id in self.markers:
            marker_data = self.markers[self.selected_marker_id]
            info_lines = [
                f"Selected Marker: {self.selected_marker_id}",
                f"Type: {marker_data.get('type', 'N/A')}"
            ]
            if 'rule' in marker_data:
                info_lines.append(f"Rule: {marker_data['rule']}")
            else:
                info_lines.append("Rule: None")
            x = img.shape[1] - 250
            y = 20
            overlay = img.copy()
            cv2.rectangle(overlay, (x-10, y-15), (img.shape[1]-10, y + 15 * len(info_lines)), (0, 0, 0), -1)
            alpha = 0.5
            img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            for line in info_lines:
                cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 30

    def draw_mode_indicator(self, img: np.ndarray):
        """Displays the current mode on the bottom right of the screen."""
        mode_text = f"Mode: {self.mode.upper()}"
        (text_width, text_height), baseline = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = img.shape[1] - text_width - 20
        y = img.shape[0] - 20
        cv2.putText(img, mode_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
            # Toggle modes using T key
            elif key == ord('t'):
                self.mode = "select" if self.mode == "draw" else "draw"
                logging.info(f"Switched to {self.mode.upper()} mode")
            # In select mode, press X to delete the selected marker (zone or movement)
            elif key == ord('x'):
                if self.mode == "select" and self.selected_marker_id is not None:
                    marker = self.markers.get(self.selected_marker_id, {})
                    marker_type = marker.get('type')
                    # Record deletion action with a deep copy and clear redo stack
                    self.undo_stack.append({'action': 'delete', 'marker': copy.deepcopy(marker)})
                    self.redo_stack.clear()
                    if marker_type == 'zone':
                        self.zones.pop(self.selected_marker_id, None)
                    elif marker_type == 'movement':
                        self.arrow_zones.pop(self.selected_marker_id, None)
                        self.arrows.pop(self.selected_marker_id, None)
                    self.markers.pop(self.selected_marker_id, None)
                    self.mask_positions = [m for m in self.mask_positions if m.get('marker_id') != self.selected_marker_id]
                    logging.info(f"Deleted {marker_type} marker {self.selected_marker_id}")
                    self.selected_marker_id = None
                    self.save_mask_positions()
            elif key == ord('s'):
                self.save_mask_positions()
            elif key == ord('u'):
                self.handle_undo()
            elif key == ord('r'):
                self.handle_redo()
            elif key == 27 or key == ord('q'):
                break

            img = self.target_frame.copy()
            self.draw_marker_ids(img)
            self.draw_instructions(img)
            self.draw_axes(img)
            self.draw_selected_marker_info(img)
            self.draw_mode_indicator(img)
            cv2.imshow('Target Frame', img)
        cv2.destroyAllWindows()
        self.cap.release()
        return self.mask_positions

    def is_inside_arrow_zone(self, x: int, y: int) -> bool:
        """Checks if a given point (x, y) is inside any defined arrow zone."""
        for zone in self.arrow_zones.values():
            if zone:
                contour = np.array(zone, dtype=np.float32)
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    return True
        return False

    def mouse_callback(self, event, x, y, flags, param):
        img = self.target_frame.copy()
        # In select mode: left-click selects, right-click assigns rule
        if self.mode == "select":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.handle_selection(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.selected_marker_id is not None:
                    rule = input(f"Enter rule for marker {self.selected_marker_id}: ")
                    self.markers[self.selected_marker_id]['rule'] = rule
                    for i, m in enumerate(self.mask_positions):
                        if m.get('marker_id') == self.selected_marker_id:
                            self.mask_positions[i]['rule'] = rule
                    self.save_mask_positions()
                    logging.info(f"Assigned rule '{rule}' to marker {self.selected_marker_id}")
            self.draw_marker_ids(img)
            self.draw_instructions(img)
            self.draw_axes(img)
            self.draw_selected_marker_info(img)
            self.draw_mode_indicator(img)
            cv2.imshow('Target Frame', img)
            return

        # Draw mode
        if self.current_tool == 'zone':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.drawing:
                    self.current_polygon = []
                    self.drawing = True
                self.current_polygon.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_polygon = self.current_polygon + [(x, y)]
                self.draw_polygon(img, temp_polygon, (0, 255, 0))
                self.draw_mode_indicator(img)
                cv2.imshow('Target Frame', img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                if len(self.current_polygon) > 2:
                    self.current_polygon.append(self.current_polygon[0])
                    marker_id = self.get_next_marker_id()
                    self.zones[marker_id] = self.current_polygon.copy()
                    marker = {
                        "marker_id": marker_id,
                        "type": "zone",
                        "status": "activate",
                        "points": self.current_polygon.copy()
                    }
                    self.markers[marker_id] = marker
                    self.mask_positions.append(marker)
                    # Record creation action with deep copy and clear redo stack
                    self.undo_stack.append({'action': 'create', 'marker': copy.deepcopy(marker)})
                    self.redo_stack.clear()
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
                self.draw_mode_indicator(img)
                cv2.imshow('Target Frame', img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                if len(self.current_polygon) > 2:
                    self.current_polygon.append(self.current_polygon[0])
                    marker_id = self.get_next_marker_id()
                    self.arrow_zones[marker_id] = self.current_polygon.copy()
                    marker = {
                        "marker_id": marker_id,
                        "type": "movement",
                        "status": "activate",
                        "polygon_points": self.current_polygon.copy(),
                        "line_points": []
                    }
                    self.markers[marker_id] = marker
                    self.mask_positions.append(marker)
                    # Record creation action and clear redo stack
                    self.undo_stack.append({'action': 'create', 'marker': copy.deepcopy(marker)})
                    self.redo_stack.clear()
                    logging.info(f"Added Arrow Zone marker with marker_id={marker_id}")
                self.current_polygon = []
                self.save_mask_positions()

        elif self.current_tool == 'arrow':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.drawing:
                    if not self.is_inside_arrow_zone(x, y):
                        logging.error("Arrow drawing must start within an arrow zone.")
                        return
                    self.arrow_points = []
                    self.drawing = True
                # Only add arrow point if it is within an arrow zone.
                if self.is_inside_arrow_zone(x, y):
                    self.arrow_points.append((x, y))
                else:
                    logging.error("Arrow point must be inside an arrow zone.")
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_arrow = self.arrow_points + [(x, y)]
                self.draw_multipoint_arrow(img, temp_arrow, (255, 0, 0))
                self.draw_mode_indicator(img)
                cv2.imshow('Target Frame', img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                if len(self.arrow_points) > 1:
                    # Use the arrow zone corresponding to the first arrow point.
                    valid_zone = None
                    for zone in self.arrow_zones.values():
                        if zone:
                            contour = np.array(zone, dtype=np.float32)
                            if cv2.pointPolygonTest(contour, self.arrow_points[0], False) >= 0:
                                valid_zone = contour
                                break
                    if valid_zone is None:
                        logging.error("Arrow must start inside a valid arrow zone.")
                        return
                    # Ensure all arrow points are inside the same arrow zone.
                    for pt in self.arrow_points:
                        if cv2.pointPolygonTest(valid_zone, pt, False) < 0:
                            logging.error("All arrow points must lie within the arrow zone.")
                            return
                    # Attach arrow to the arrow zone. Here we use the last defined arrow zone.
                    last_marker_id = max(self.arrow_zones.keys())
                    self.markers[last_marker_id]['line_points'] = self.arrow_points.copy()
                    # Update mask_positions
                    self.mask_positions = [m for m in self.mask_positions if m.get('marker_id') != last_marker_id]
                    self.mask_positions.append(self.markers[last_marker_id])
                    # Record update action for undo and clear redo stack
                    self.undo_stack.append({'action': 'create', 'marker': copy.deepcopy(self.markers[last_marker_id])})
                    self.redo_stack.clear()
                    logging.info(f"Added Movement marker with marker_id={last_marker_id}")
                self.arrow_points = []
                self.save_mask_positions()

        # Redraw updated items
        self.draw_marker_ids(img)
        self.draw_instructions(img)
        self.draw_axes(img)
        self.draw_mode_indicator(img)
        if self.selected_marker_id is not None and self.selected_marker_id in self.markers:
            marker_data = self.markers[self.selected_marker_id]
            pts = []
            if marker_data.get('type') == 'zone':
                pts = self.zones.get(self.selected_marker_id, [])
            else:
                pts = marker_data.get('polygon_points', []) or marker_data.get('line_points', [])
            if pts:
                cx = int(sum(p[0] for p in pts) / len(pts))
                cy = int(sum(p[1] for p in pts) / len(pts))
                cv2.circle(img, (cx, cy), 12, (0, 0, 255), 2)
        cv2.imshow('Target Frame', img)

    def draw_marker_ids(self, img: np.ndarray):
        """Draws zones and arrows with their marker IDs and rules if assigned."""
        for marker_id, poly in self.zones.items():
            self.draw_polygon(img, poly, (0, 255, 0))
            if len(poly) > 2:
                pts = np.array(poly, np.float32)
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    label = f"Marker {marker_id}"
                    marker_data = self.markers.get(marker_id, {})
                    if 'rule' in marker_data:
                        label += f" ({marker_data['rule']})"
                    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for marker_id, marker_data in self.markers.items():
            if marker_data.get('type') == 'movement':
                polygon = marker_data.get('polygon_points', [])
                if polygon:
                    self.draw_polygon(img, polygon, (0, 255, 255))
                line_points = marker_data.get('line_points', [])
                if line_points and len(line_points) > 1:
                    self.draw_multipoint_arrow(img, line_points, (255, 0, 0))
                    label = f"Marker {marker_id}"
                    if 'rule' in marker_data:
                        label += f" ({marker_data['rule']})"
                    cv2.putText(img, label, line_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def draw_instructions(self, img: np.ndarray):
        lines = [
            "Keys:",
            " Z = Zone tool",
            " A = Arrow Zone tool",
            " M = Arrow tool",
            " T = Toggle Mode (Draw/Select)",
            " X = Delete selected marker (in select mode)",
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

    def draw_polygon(self, img: np.ndarray, polygon: List[Tuple[int, int]], color: Tuple[int, int, int]=(0, 255, 0)):
        if polygon:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
            for p in polygon:
                cv2.circle(img, p, 3, color, -1)

    def draw_multipoint_arrow(self, img: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, int, int]=(255, 0, 0)):
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            cv2.arrowedLine(img, points[i], points[i+1], color, thickness=2, tipLength=0.05)

    def handle_undo(self):
        if not self.undo_stack:
            logging.info("Nothing to undo.")
            return
        last_action = self.undo_stack.pop()
        action_type = last_action['action']
        marker = last_action['marker']
        marker_id = marker['marker_id']
        if action_type == 'create':
            # Undo creation: remove marker
            if marker['type'] == 'zone':
                self.zones.pop(marker_id, None)
            elif marker['type'] == 'movement':
                self.arrow_zones.pop(marker_id, None)
                self.arrows.pop(marker_id, None)
            self.markers.pop(marker_id, None)
            self.mask_positions = [m for m in self.mask_positions if m.get('marker_id') != marker_id]
            # Push opposite action for redo (i.e., re-create)
            self.redo_stack.append({'action': 'create', 'marker': copy.deepcopy(marker)})
            logging.info(f"Undid creation: Removed marker_id={marker_id}")
        elif action_type == 'delete':
            # Undo deletion: restore marker
            if marker['type'] == 'zone':
                self.zones[marker_id] = marker.get('points', [])
            elif marker['type'] == 'movement':
                self.arrow_zones[marker_id] = marker.get('polygon_points', [])
                self.arrows[marker_id] = marker.get('line_points', [])
            self.markers[marker_id] = marker
            self.mask_positions.append(marker)
            # Push opposite action for redo (i.e., re-delete)
            self.redo_stack.append({'action': 'delete', 'marker': copy.deepcopy(marker)})
            logging.info(f"Undid deletion: Restored marker_id={marker_id}")
        self.save_mask_positions()

    def handle_redo(self):
        if not self.redo_stack:
            logging.info("Nothing to redo.")
            return
        action = self.redo_stack.pop()
        action_type = action['action']
        marker = action['marker']
        marker_id = marker['marker_id']
        if action_type == 'create':
            # Redo creation: re-add marker
            if marker['type'] == 'zone':
                self.zones[marker_id] = marker.get('points', [])
            elif marker['type'] == 'movement':
                self.arrow_zones[marker_id] = marker.get('polygon_points', [])
                self.arrows[marker_id] = marker.get('line_points', [])
            self.markers[marker_id] = marker
            self.mask_positions.append(marker)
            self.undo_stack.append({'action': 'create', 'marker': copy.deepcopy(marker)})
            logging.info(f"Redid creation: Restored marker_id={marker_id}")
        elif action_type == 'delete':
            # Redo deletion: remove marker
            if marker['type'] == 'zone':
                self.zones.pop(marker_id, None)
            elif marker['type'] == 'movement':
                self.arrow_zones.pop(marker_id, None)
                self.arrows.pop(marker_id, None)
            self.markers.pop(marker_id, None)
            self.mask_positions = [m for m in self.mask_positions if m.get('marker_id') != marker_id]
            self.undo_stack.append({'action': 'delete', 'marker': copy.deepcopy(marker)})
            logging.info(f"Redid deletion: Removed marker_id={marker_id}")
        self.save_mask_positions()

    def save_mask_positions(self):
        with open(self.mask_json_path, 'w') as f:
            json.dump(self.mask_positions, f, indent=4)
        logging.info("Mask positions saved.")