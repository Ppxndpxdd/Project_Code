import cv2
import json
import logging
import screeninfo
import numpy as np
import os
import yt_dlp
from typing import Any, Dict, List, Tuple

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
        self.current_tool = 'zone'
        self.arrow_points = []
        self.current_polygon: List[Tuple[int, int]] = []
        self.markers: Dict[int, Dict[str, Any]] = {}
        self.marker_id = 1
        self.target_frame = None

        # Selected marker for rule assignment
        self.selected_marker_id = None

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
                # Resize frame to fit screen
                self.target_frame = self.resize_frame_to_screen(self.target_frame)
        else:
            logging.error(f"Failed to open video source: {self.video_source}")

        # Initialize data structures
        self.zones = {}
        self.arrow_zones = {}
        self.arrows = {}
        self.load_markers()

        # Visual properties remain unchanged
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
        if len(self.markers) == 0:
            return 1
        return max(self.markers.keys()) + 1

    def draw_selected_marker_info(self, img: np.ndarray):
        """Draws selected marker information on the top right of the screen."""
        if self.selected_marker_id is not None and self.selected_marker_id in self.markers:
            marker_data = self.markers[self.selected_marker_id]
            info_lines = [
                f"Selected Marker: {self.selected_marker_id}",
                f"Type: {marker_data.get('type', 'N/A')}",
            ]
            if 'rule' in marker_data:
                info_lines.append(f"Rule: {marker_data['rule']}")
            else:
                info_lines.append("Rule: None")
            
            # Define top right position for info block
            x = img.shape[1] - 250  # adjust width as needed
            y = 20
            # Draw semi-transparent background
            overlay = img.copy()
            cv2.rectangle(overlay, (x-10, y-15), (img.shape[1]-10, y + 15 * len(info_lines)), (0, 0, 0), -1)
            alpha = 0.5  # transparency factor
            img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Draw each info line on top right corner
            for line in info_lines:
                cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 30

    def run(self) -> List[Dict[str, Any]]:
        cv2.imshow('Target Frame', self.target_frame)
        cv2.setMouseCallback('Target Frame', self.mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('z'):
                self.current_tool = 'zone'
            elif key == ord('a'):
                self.current_tool = 'arrow_zone'
            elif key == ord('m'):
                self.current_tool = 'arrow'
            elif key == ord('s'):
                self.save_mask_positions()
            elif key == ord('u'):
                self.handle_undo()
            elif key == ord('r'):
                self.handle_redo()
            elif key == ord('c'):
                if self.selected_marker_id is not None:
                    rule = input(f"Enter rule for marker {self.selected_marker_id}: ")
                    self.markers[self.selected_marker_id]['rule'] = rule
                    for m in self.mask_positions:
                        if m.get('marker_id') == self.selected_marker_id:
                            m['rule'] = rule
                            break
                    self.save_mask_positions()
                    logging.info(f"Assigned rule '{rule}' to marker {self.selected_marker_id}")
                else:
                    logging.info("No marker selected. Click near a marker to select it first.")
            elif key == 27 or key == ord('q'):
                break

            img = self.target_frame.copy()
            self.draw_marker_ids(img)
            self.draw_instructions(img)
            self.draw_axes(img)
            self.draw_selected_marker_info(img)
            cv2.imshow('Target Frame', img)

        cv2.destroyAllWindows()
        self.cap.release()
        return self.mask_positions

    def mouse_callback(self, event, x, y, flags, param):
        img = self.target_frame.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            selected = False

            # First, try selecting zone markers by polygon test and center test
            for marker_id, marker_data in self.markers.items():
                if marker_data.get('type') == 'zone':
                    points = self.zones.get(marker_id, [])
                    if points:
                        contour = np.array(points, dtype=np.float32)
                        # Use pointPolygonTest with tolerance
                        dist = cv2.pointPolygonTest(contour, (x, y), True)
                        if dist >= -10:  # within or near the polygon
                            self.selected_marker_id = marker_id
                            logging.info(f"Selected zone marker {marker_id} via polygon test")
                            selected = True
                            break

            # If no zone marker selected, try center test for zones if available
            if not selected:
                for marker_id, marker_data in self.markers.items():
                    if marker_data.get('type') == 'zone':
                        points = self.zones.get(marker_id, [])
                        if points:
                            cx = int(sum(p[0] for p in points) / len(points))
                            cy = int(sum(p[1] for p in points) / len(points))
                            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                            if distance < 30:  # increased threshold for easier selection
                                self.selected_marker_id = marker_id
                                logging.info(f"Selected zone marker {marker_id} via center test")
                                selected = True
                                break

            # If still not selected, check movement markers (which may have polygon and/or line points)
            if not selected:
                for marker_id, marker_data in self.markers.items():
                    if marker_data.get('type') == 'movement':
                        # Try polygon test if polygon_points exist
                        polygon = marker_data.get('polygon_points')
                        if polygon and len(polygon) >= 3:
                            contour = np.array(polygon, dtype=np.float32)
                            dist = cv2.pointPolygonTest(contour, (x, y), True)
                            if dist >= -10:
                                self.selected_marker_id = marker_id
                                logging.info(f"Selected movement marker {marker_id} via polygon test")
                                selected = True
                                break
                        # Fallback: try center test on line_points if exist
                        line_points = marker_data.get('line_points')
                        if line_points and len(line_points) > 0:
                            cx = int(sum(pt[0] for pt in line_points) / len(line_points))
                            cy = int(sum(pt[1] for pt in line_points) / len(line_points))
                            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                            if distance < 30:
                                self.selected_marker_id = marker_id
                                logging.info(f"Selected movement marker {marker_id} via center test")
                                selected = True
                                break

            if not selected:
                self.selected_marker_id = None
                logging.info("No marker selected.")
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

        # Redraw and highlight the selected marker if any
        self.draw_marker_ids(img)
        self.draw_instructions(img)
        self.draw_axes(img)
        if self.selected_marker_id is not None and self.selected_marker_id in self.markers:
            # Highlight center of the marker polygon for zone markers or average point for movement markers
            marker_data = self.markers[self.selected_marker_id]
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
                    label = f"Marker {marker_id}"
                    # If marker has an assigned rule, display it.
                    marker_data = self.markers.get(marker_id, {})
                    if 'rule' in marker_data:
                        label += f" | Rule: {marker_data['rule']}"
                    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Arrow Zones and Arrows for movement markers
        for marker_id, marker_data in self.markers.items():
            if marker_data.get('type') == 'movement':
                # Draw polygon for arrow zone
                polygon = marker_data.get('polygon_points', [])
                if polygon:
                    self.draw_polygon(img, polygon, (0, 255, 255))
                # Draw arrow line
                line_points = marker_data.get('line_points', [])
                if line_points and len(line_points) > 1:
                    self.draw_multipoint_arrow(img, line_points, (255, 0, 0))
                    label = f"Marker {marker_id}"
                    if 'rule' in marker_data:
                        label += f" | Rule: {marker_data['rule']}"
                    cv2.putText(img, label, line_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def draw_instructions(self, img: np.ndarray):
        lines = [
            "Keys:",
            " Z = Zone tool",
            " A = Arrow Zone tool",
            " M = Arrow tool",
            " S = Save",
            " U = Undo",
            " R = Redo",
            " C = Assign rule to selected marker",
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
                cv2.circle(img, p, 3, color, -1)

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