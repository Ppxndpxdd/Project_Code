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
        self.editing = False
        self.dragging_point = False
        self.selected_point_index = -1

        # zone (polygon) data
        self.current_polygon: List[Tuple[int, int]] = []
        self.zones: Dict[int, List[Tuple[int, int]]] = {}
        self.zone_id = 1

        # arrow data
        self.arrow_points: List[Tuple[int,int]] = []
        self.arrows: Dict[int, List[Tuple[int,int]]] = {}
        self.arrow_id = 1

        # combined mask positions loaded from file
        self.mask_positions = []

        # ensure mask_json_path is defined
        if not self.mask_json_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.mask_json_path = os.path.join(script_dir, 'config', 'marker_positions.json')

        # load if exists
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

        # load polygons & arrows
        self.load_polygons()
        self.load_arrows()

        # user can switch tools: "zone" or "arrow"
        self.current_tool = 'zone'

        # visual properties
        self.highlight_radius = config.get('highlight_radius', 10)
        self.point_radius = config.get('point_radius', 5)
        self.line_threshold = config.get('line_threshold', 10)
        self.selection_threshold = config.get('selection_threshold', 10)

        self.undo_stack = []
        self.redo_stack = []

    def load_polygons(self):
        self.zones.clear()
        zone_entries = [m for m in self.mask_positions if m.get('type') == 'zone']
        for e in zone_entries:
            self.zones[e['zone_id']] = e['points']

    def load_arrows(self):
        self.arrows.clear()
        arrow_entries = [m for m in self.mask_positions if m.get('type') == 'movement']
        for e in arrow_entries:
            self.arrows[e['movement_id']] = e['points']

    def save_mask_positions(self):
        with open(self.mask_json_path, 'w') as f:
            json.dump(self.mask_positions, f, indent=4)
        logging.info("Mask positions saved.")

    def get_next_zone_id(self) -> int:
        if len(self.zones) == 0:
            return 1
        return max(self.zones.keys()) + 1

    def get_next_arrow_id(self) -> int:
        if len(self.arrows) == 0:
            return 1
        return max(self.arrows.keys()) + 1

    def is_within_bounds(self, point: Tuple[int,int], img: np.ndarray) -> bool:
        px, py = point
        h, w = img.shape[:2]
        return (0 <= px < w) and (0 <= py < h)

    def draw_polygon(self, img: np.ndarray, polygon: List[Tuple[int, int]], color: Tuple[int,int,int] = (0, 255, 0)):
        if len(polygon) > 0:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
            for p in polygon:
                cv2.circle(img, (int(p[0]), int(p[1])), self.point_radius, color, -1)

    def draw_multipoint_arrow(self, img: np.ndarray, points: List[Tuple[int,int]], color=(0,255,0)):
        if len(points) < 2:
            return
        pts = np.array(points, np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2)

    def draw_zone_ids(self, img: np.ndarray):
        # draw zones
        for zone_id, poly in self.zones.items():
            self.draw_polygon(img, poly, (0,255,0))
            if len(poly) > 2:
                pts = np.array(poly, np.float32)
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, f"Zone {zone_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # draw arrows
        for arr_id, arr_points in self.arrows.items():
            self.draw_multipoint_arrow(img, arr_points, (255,0,0))
            if len(arr_points) > 1:
                cv2.putText(img, f"Arrow {arr_id}", arr_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    def draw_instructions(self, img: np.ndarray):
        lines = [
            "Keys:",
            " Z = Zone tool",
            " A = Arrow tool",
            " E = Toggle Editing",
            " S = Save",
            " U = Undo",
            " R = Redo",
            " Q or ESC = Quit"
        ]
        for i, txt in enumerate(lines):
            cv2.putText(img, txt, (10, 30 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    def draw_axes(self, img: np.ndarray):
        h, w = img.shape[:2]
        cv2.line(img, (0,h-1), (w,h-1), (255,0,0), 2)
        cv2.putText(img, "X", (w-20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.line(img, (0,0), (0,h), (0,255,0), 2)
        cv2.putText(img, "Y", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    def mouse_callback(self, event, x, y, flags, param):
        img = self.target_frame.copy()

        # ZONE drawing
        if self.current_tool == 'zone':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.drawing:
                    self.current_polygon = []
                    self.drawing = True
                self.current_polygon.append((x,y))
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_polygon = self.current_polygon + [(x,y)]
                self.draw_polygon(img, temp_polygon)
                cv2.imshow('Target Frame', img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                # close polygon
                if len(self.current_polygon) > 2:
                    self.current_polygon.append(self.current_polygon[0])
                    zone_id = self.get_next_zone_id()
                    self.zones[zone_id] = self.current_polygon.copy()
                    self.mask_positions.append({
                        "zone_id": zone_id,
                        "type": "zone",
                        "status": "activate",
                        "points": self.current_polygon
                    })
                self.current_polygon = []
                self.save_mask_positions()

        # ARROW drawing
        else:  # self.current_tool == 'arrow'
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.drawing:
                    self.arrow_points = []
                    self.drawing = True
                self.arrow_points.append((x,y))
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp_arrow = self.arrow_points + [(x,y)]
                self.draw_multipoint_arrow(img, temp_arrow, (255,0,0))
                cv2.imshow("Target Frame", img)
            elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
                self.drawing = False
                if len(self.arrow_points) > 1:
                    arr_id = self.get_next_arrow_id()
                    self.arrows[arr_id] = self.arrow_points.copy()
                    self.mask_positions.append({
                        "movement_id": arr_id,
                        "type": "movement",
                        "status": "activate",
                        "points": self.arrow_points
                    })
                self.arrow_points = []
                self.save_mask_positions()

        # draw everything
        self.draw_zone_ids(img)
        self.draw_instructions(img)
        self.draw_axes(img)
        cv2.imshow('Target Frame', img)

    def run(self) -> List[Dict[str, Any]]:
        cv2.imshow('Target Frame', self.target_frame)
        cv2.setMouseCallback('Target Frame', self.mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('z'):
                self.current_tool = 'zone'
                logging.info("Switched to Zone tool")
            elif key == ord('a'):
                self.current_tool = 'arrow'
                logging.info("Switched to Arrow tool")
            elif key == ord('s'):
                self.save_mask_positions()
            elif key == ord('u'):
                pass  # optional: handle undo
            elif key == ord('r'):
                pass  # optional: handle redo
            elif key == 27 or key == ord('q'):
                break

            img = self.target_frame.copy()
            self.draw_zone_ids(img)
            self.draw_instructions(img)
            self.draw_axes(img)
            cv2.imshow('Target Frame', img)

        cv2.destroyAllWindows()
        self.cap.release()
        return self.mask_positions