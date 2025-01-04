import cv2
import numpy as np
import json
import os
from .utils import (
    draw_polygon,
    draw_zone_ids,
    draw_instructions,
    draw_axes,
    is_within_bounds,
    is_point_near_polygon_point,
    get_next_available_zone_id,
    find_closest_point_index,
    ensure_point_format
)

class MaskTool:
    def __init__(self, config):
        self.config = config
        video_source = config['video_source']
        frame_to_edit = config['frame_to_edit']
        mask_json_path = config.get('mask_json_path', None)
        self.cap = cv2.VideoCapture(video_source)
        self.mask_positions = []
        self.drawing = False
        self.editing = False
        self.current_polygon = []
        self.zone_id = 1
        self.frame_id = frame_to_edit
        self.dragging_point = False
        self.target_frame = None
        self.zones = {}
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
            mask_json_path = os.path.join(script_dir, '..', 'config', 'mask_positions.json')
        self.mask_json_path = mask_json_path

        # Load existing mask positions
        if os.path.exists(self.mask_json_path):
            try:
                with open(self.mask_json_path, 'r') as f:
                    self.mask_positions = json.load(f)
            except Exception as e:
                print(f"Error loading mask positions from {self.mask_json_path}: {e}")

        # Set the video capture to the selected frame
        if not self.cap.isOpened():
            print(f"Error: Cannot open video source '{video_source}'.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        ret, self.target_frame = self.cap.read()
        if not ret:
            print("Error: Could not read the frame.")
            return

        # Load existing polygons for the current frame (if any)
        self.load_polygons_for_frame(self.frame_id)

    def add_point_to_polygon(self, point, img):
        if not is_within_bounds(point, img, self.target_frame.shape):
            return

        if len(self.current_polygon) == 0:
            self.current_polygon.append(point)
        elif is_point_near_polygon_point(point, self.current_polygon, self.selection_threshold):
            if len(self.current_polygon) > 2 and self.current_polygon[0] != point:
                self.current_polygon.append(self.current_polygon[0])
                draw_polygon(img, self.current_polygon, self.point_radius)
                self.save_polygon()
                self.current_polygon = []
                self.drawing = False
                self.zone_id = get_next_available_zone_id(self.zones)
            else:
                print("Polygon too small to close or point too close to start")
        else:
            self.current_polygon.append(point)

    def draw_mask(self, event, x, y, flags, param):
        img = self.target_frame.copy()

        def is_point_near_line_segment(point, line_start, line_end, threshold):
            """Checks if a point is near a line segment, including the closing segment of a polygon."""

            # Avoid division by zero
            if np.array_equal(line_start, line_end):
                return False, None

            line_vec = np.array(line_end) - np.array(line_start)
            point_vec = np.array(point) - np.array(line_start)
            line_len = np.linalg.norm(line_vec)

            # Project point onto the line
            proj = np.dot(point_vec, line_vec) / line_len ** 2

            # Calculate closest point on the line segment
            if proj < 0:
                closest_point = line_start
            elif proj > 1:
                closest_point = line_end
            else:
                closest_point = line_start + proj * line_vec

            # Check distance from point to closest point
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
                    draw_polygon(img, self.current_polygon, self.point_radius)

        def select_nearest_point(point):
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
                        if is_point_near_polygon_point((x, y), polygon, self.selection_threshold):
                            # Existing point clicked - start dragging
                            self.selected_point_index = find_closest_point_index(polygon, (x, y))
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
                            self.selected_point_index = find_closest_point_index(self.current_polygon, (x, y))
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
            draw_zone_ids(img, self.zones, self.point_radius)
            draw_instructions(img)
            draw_polygon(img, self.current_polygon, self.point_radius)
            draw_axes(img)

            if self.dragging_point and self.selected_point_index != -1 and self.zone_id is not None:
                self.current_polygon[self.selected_point_index] = (x, y)
                self.zones[self.zone_id] = self.current_polygon  # Update the correct zone
                draw_polygon(img, self.current_polygon, self.point_radius)
            elif self.drawing:
                temp_polygon = self.current_polygon + [(x, y)]
                draw_polygon(img, temp_polygon, self.point_radius)
            else:
                highlight_color = (0, 0, 255)
                highlight_point = select_nearest_point((x, y))
                if highlight_point != -1:
                    cv2.circle(img, ensure_point_format(self.current_polygon[highlight_point]), 7, highlight_color, -1)

            draw_polygon(img, self.current_polygon, self.point_radius)
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
                print("Switched to Editing mode")
            else:
                self.current_polygon = []
                self.drawing = False
                self.zone_id = get_next_available_zone_id(self.zones)
                print(f"Switched to Drawing mode. New zone_id: {self.zone_id}")

        draw_polygon(img, self.current_polygon, self.point_radius)
        cv2.imshow('Target Frame', img)

    def save_polygon(self):
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

    def run(self):
        cv2.imshow('Target Frame', self.target_frame)
        cv2.setMouseCallback('Target Frame', self.draw_mask)

        print("Edit the mask and use the following commands:")
        print(" - 's' to save the mask")
        print(" - 'u' to undo")
        print(" - 'r' to redo")
        print(" - 'e' to toggle editing mode")
        print(" - 'q' to start detection")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Ensure the directory exists
                mask_dir = os.path.dirname(self.mask_json_path)
                os.makedirs(mask_dir, exist_ok=True)
                with open(self.mask_json_path, 'w') as f:
                    json.dump(self.mask_positions, f)
                print(f"Masks and positions have been saved to {self.mask_json_path}.")
            elif key == ord('u'):
                if self.undo_stack:
                    self.zones, self.mask_positions = self.undo_stack.pop()
                    self.redo_stack.append((self.zones.copy(), self.mask_positions.copy()))
                    print("Undo last action.")
            elif key == ord('r'):
                if self.redo_stack:
                    self.zones, self.mask_positions = self.redo_stack.pop()
                    self.undo_stack.append((self.zones.copy(), self.mask_positions.copy()))
                    print("Redo last undone action.")
            elif key == ord('e'):
                self.editing = not self.editing
                if self.editing:
                    self.current_polygon = []
                    self.drawing = False
                    self.dragging_point = False
                    self.selected_point_index = -1
                    print("Switched to Editing mode")
                else:
                    self.current_polygon = []
                    self.drawing = False
                    self.zone_id = get_next_available_zone_id(self.zones)
                    print(f"Switched to Drawing mode. New zone_id: {self.zone_id}")
            elif key == 27 or key == ord('q'):
                break

            # Redraw after each action to reflect changes
            img = self.target_frame.copy()
            draw_zone_ids(img, self.zones, self.point_radius)
            draw_instructions(img)
            draw_axes(img)
            cv2.imshow('Target Frame', img)
        cv2.destroyAllWindows()
        self.cap.release()
        return self.mask_positions

    def load_polygons_for_frame(self, frame_id):
        """Loads polygons from the JSON data for the specified frame."""
        self.zones.clear()
        frame_data = [entry for entry in self.mask_positions if entry['frame'] == frame_id]
        for entry in frame_data:
            self.zones[entry['zone_id']] = entry['points']