import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import atexit

class MaskTool:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        self.mask_positions = pd.DataFrame(columns=['frame', 'zone_id', 'points'])
        self.drawing = False
        self.editing = False
        self.current_polygon = []
        self.zone_id = 1
        self.frame_id = 0
        self.dragging_point = False
        self.target_frame = None
        self.zones = {}  # Store polygons for the current frame
        self.undo_stack = []
        self.redo_stack = []
        self.highlight_radius = 10
        self.point_radius = 5
        self.line_threshold = 10
        self.selection_threshold = 10

    def draw_polygon(self, img, polygon, color=(0, 255, 0)):
        if len(polygon) > 0:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
            for p in polygon:
                p = self.ensure_point_format(p)
                cv2.circle(img, p, self.point_radius, color, -1)

    def ensure_point_format(self, point):
        return tuple(map(int, point))

    def draw_zone_ids(self, img):
        for zone_id, polygon in self.zones.items():
            if len(polygon) > 0:
                self.draw_polygon(img, polygon, color=(0, 255, 0))
                pts = np.array(polygon, np.float32)
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, f'Zone {zone_id}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    def draw_instructions(self, img):
        instructions = [
            "Press 'e' to toggle Editing Mode",
            "Press 's' to Save Mask",
            "Press 'u' to Undo",
            "Press 'r' to Redo",
            "Press 'q' to Quit"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_axes(self, img):
        height, width = img.shape[:2]

        # Draw x-axis (start at bottom-left)
        cv2.line(img, (0, height - 1), (width, height - 1), (255, 0, 0), 2)  # Blue line
        cv2.putText(img, 'X', (width - 20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw y-axis (start at bottom-left)
        cv2.line(img, (0, 0), (0, height), (0, 255, 0), 2)  # Green line
        cv2.putText(img, 'Y', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def is_within_bounds(self, point, img):
        """Checks if a point is within the image boundaries."""
        px, py = point
        height, width = img.shape[:2]
        return 0 <= px < width and 0 <= py < height

    def add_point_to_polygon(self, point, img):
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

            return dist < threshold, closest_point

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
            else:
                self.drawing = True
                self.add_point_to_polygon((x, y), img)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point and self.selected_point_index != -1:
                self.current_polygon[self.selected_point_index] = (x, y)
                self.zones[self.zone_id] = self.current_polygon
                self.draw_polygon(img, self.current_polygon)
            elif self.drawing:
                temp_polygon = self.current_polygon.copy()
                temp_polygon.append((x, y))
                self.draw_polygon(img, temp_polygon)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_point:
                self.dragging_point = False
                self.drawing = False
                self.save_polygon()
            elif self.editing:
                if self.zone_id is not None and not self.dragging_point:
                    if select_nearest_point((x, y)) != -1:
                        self.drawing = False
                        self.dragging_point = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.editing and self.zone_id is not None:
                self.remove_point_from_polygon(x, y)
                self.save_polygon()

        self.target_frame = img

    def is_point_near_polygon_point(self, point, polygon):
        """Checks if a point is near a point in a polygon."""
        px, py = point
        for (vx, vy) in polygon:
            if abs(px - vx) < self.highlight_radius and abs(py - vy) < self.highlight_radius:
                return True
        return False

    def find_closest_point_index(self, polygon, point):
        """Find the closest point in the polygon to the given point."""
        min_dist = float('inf')
        closest_index = -1
        for i, (px, py) in enumerate(polygon):
            dist = np.linalg.norm(np.array((px, py)) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        return closest_index

    def remove_point_from_polygon(self, x, y):
        """Removes a point from the current polygon."""
        closest_index = self.find_closest_point_index(self.current_polygon, (x, y))
        if closest_index != -1:
            self.current_polygon.pop(closest_index)
            self.zones[self.zone_id] = self.current_polygon

    def save_polygon(self):
        """Save the current polygon to the zones."""
        if len(self.current_polygon) > 2:
            self.zones[self.zone_id] = self.current_polygon.copy()
            self.current_polygon = []
            self.drawing = False

    def undo_last_action(self):
        if len(self.undo_stack) > 0:
            last_action = self.undo_stack.pop()
            self.redo_stack.append(self.zones.copy())
            self.zones = last_action

    def redo_last_action(self):
        if len(self.redo_stack) > 0:
            last_action = self.redo_stack.pop()
            self.undo_stack.append(self.zones.copy())
            self.zones = last_action

    def get_next_available_zone_id(self):
        """Get the next available zone ID."""
        existing_ids = list(self.zones.keys())
        return max(existing_ids) + 1 if existing_ids else 1

    def save_mask_positions(self, path='mask_positions.csv'):
        """Save mask positions to a CSV file."""
        self.mask_positions.to_csv(path, index=False)

    def display(self):
        """Main display loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.target_frame = frame.copy()
            self.draw_zone_ids(frame)
            self.draw_instructions(frame)
            self.draw_axes(frame)
            cv2.imshow('Video Frame', frame)
            cv2.setMouseCallback('Video Frame', self.draw_mask)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('e'):
                self.editing = not self.editing
            elif key == ord('s'):
                self.save_mask_positions()
            elif key == ord('u'):
                self.undo_last_action()
            elif key == ord('r'):
                self.redo_last_action()
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

class ZoneIntersectionTracker:
    def __init__(self, zones, video_source, model_path):
        self.zones = zones
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO(model_path)
        self.frame_id = 0

    def detect_objects(self, frame):
        results = self.model(frame)
        if results[0].boxes is not None:
            detections = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes as xyxy format
        else:
            detections = np.array([])  # No detections
        return detections


    def draw_detections(self, img, detections):
        for (xmin, ymin, xmax, ymax, conf, cls) in detections:
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

    def check_intersections(self, frame):
        detections = self.detect_objects(frame)
        self.draw_detections(frame, detections)
        for (xmin, ymin, xmax, ymax, _, _) in detections:
            bbox_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            for zone_id, polygon in self.zones.items():
                if cv2.pointPolygonTest(np.array(polygon, np.int32), bbox_center, False) >= 0:
                    cv2.putText(frame, f'Intersection with Zone {zone_id}', (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def track(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.check_intersections(frame)
            cv2.imshow('Zone Tracker', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Initialize the MaskTool and ZoneIntersectionTracker with appropriate parameters
video_source = 'mask_tool\\test_source\\traffic.mp4'
model_path = 'mask_tool\\YoLo\\model\\yolov8n-seg.onnx'
mask_tool = MaskTool(video_source)
atexit.register(mask_tool.save_mask_positions)
mask_tool.display()
zone_intersection_tracker = ZoneIntersectionTracker(mask_tool.zones, video_source, model_path)
zone_intersection_tracker.track()
