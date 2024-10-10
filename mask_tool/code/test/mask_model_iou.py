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
        self.zones = {}  # Store polygons for current frame
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
            proj = np.dot(point_vec, line_vec) / line_len**2 

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

                    # If clicked near a vertex (and not already dragging), duplicate the vertex
                    if self.zone_id is not None and not self.dragging_point:
                        polygon = self.zones[self.zone_id]
                        for i, point in enumerate(polygon):
                            if np.linalg.norm(np.array(point) - np.array((x, y))) < self.selection_threshold:
                                self.current_polygon = list(polygon)
                                self.current_polygon.insert(i, point) # Insert duplicate at the same index
                                self.zones[self.zone_id] = self.current_polygon
                                self.selected_point_index = i
                                self.dragging_point = True
                                self.drawing = False
                                break

            else:  # Not in editing mode
                if not self.drawing:
                    self.current_polygon = []  # Clear the polygon when starting a new one

                if self.drawing:
                    self.add_point_to_polygon((x, y), img)  # Call using self.
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
                print("Switched to Editing mode")
            else:
                self.current_polygon = []
                self.drawing = False
                self.zone_id = self.get_next_available_zone_id()
                print(f"Switched to Drawing mode. New zone_id: {self.zone_id}")
                
        self.draw_polygon(img, self.current_polygon)
        cv2.imshow('Target Frame', img) 
        
    def is_point_near_polygon_point(self, point, polygon, threshold=10):
        for p in polygon:
            if np.linalg.norm(np.array(p) - np.array(point)) < threshold:
                return True
        return False

    def get_next_available_zone_id(self):
        if len(self.zones) == 0:
            return 1
        return max(self.zones.keys()) + 1

    def find_closest_point_index(self, polygon, point):
        min_distance = float('inf')
        closest_index = -1
        for i, p in enumerate(polygon):
            distance = np.linalg.norm(np.array(p) - np.array(point))
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index

    def save_polygon(self):
        if len(self.current_polygon) > 2:
            if self.current_polygon[0] == self.current_polygon[-1]:
                self.current_polygon.pop()
            
            # Update zones dictionary instead of directly modifying the DataFrame
            self.zones[self.zone_id] = self.current_polygon
            
            # Update the DataFrame with the current zones
            self.mask_positions = self.mask_positions[self.mask_positions['frame'] != self.frame_id]
            for zone_id, polygon in self.zones.items():
                new_entry = pd.DataFrame([{'frame': self.frame_id, 'zone_id': zone_id, 'points': polygon}])
                self.mask_positions = pd.concat([self.mask_positions, new_entry], ignore_index=True)
            self.undo_stack.append((self.zones.copy(), self.mask_positions.copy()))
            self.redo_stack.clear()

    def run(self):
        self.frame_id = frame_to_edit
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_edit)
        ret, self.target_frame = self.cap.read()
        if not ret:
            print("Error: Could not read the frame.")
            return

        # Load existing polygons for the current frame
        self.load_polygons_for_frame(self.frame_id)

        cv2.imshow('Target Frame', self.target_frame)
        cv2.setMouseCallback('Target Frame', self.draw_mask)

        print("Edit the mask and use the following commands:")
        print(" - 's' to save the mask")
        print(" - 'u' to undo")
        print(" - 'r' to redo")
        print(" - 'e' to toggle editing mode")
        print(" - 'q' to quit")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.mask_positions.to_csv('mask_tool\\result\\mask_positions.csv', index=True)
                print("Masks and positions have been saved.")
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
                    self.zone_id = self.get_next_available_zone_id()
                    print(f"Switched to Drawing mode. New zone_id: {self.zone_id}")
            elif key == 27 or key == ord('q'):
                break

            # Redraw after each action to reflect changes
            img = self.target_frame.copy()
            self.draw_zone_ids(img)
            self.draw_instructions(img)
            self.draw_axes(self.target_frame)
            cv2.imshow('Target Frame', img)
        cv2.destroyAllWindows()
        self.cap.release()

    def load_polygons_for_frame(self, frame_id):
        """Loads polygons from the DataFrame for the specified frame."""
        self.zones.clear()  # Clear existing zones
        frame_data = self.mask_positions[self.mask_positions['frame'] == frame_id]
        for _, row in frame_data.iterrows():
            self.zones[row['zone_id']] = row['points']

class ZoneIntersectionTracker:
    def __init__(self, model_path, mask_csv_path):
        self.model = YOLO(model_path)
        self.mask_positions = pd.read_csv(mask_csv_path)
        self.zones = {}
        self.next_object_id = 1
        self.object_ids = {}  # Track object IDs across frames
        self.detection_log = []  # Store detection log
        atexit.register(self.save_detection_log)  # Register save function for exit
        self.object_ids = {}  # Track object IDs across frames
        self.tracked_objects = {}  # Store information about tracked objects
        self.iou_threshold = 0.55 # Adjust as needed
        self.max_lost_frames = 10 # Adjust as needed

    def load_zones_for_frame(self, frame_id):
        self.zones.clear()
        frame_data = self.mask_positions[self.mask_positions['frame'] == frame_id]
        for _, row in frame_data.iterrows():
            self.zones[row['zone_id']] = np.array(eval(row['points']))

    def intersects(self, bbox_polygon, polygon):
        bbox_polygon = bbox_polygon.reshape((-1, 1, 2)).astype(np.int32)
        polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
        return cv2.intersectConvexConvex(bbox_polygon, polygon)[0]

    def track_intersections(self, video_path, frame_to_edit):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Set frame position for editing
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_edit)

        # Get the frame for mask definition
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame.")
            return

        # Use MaskTool to define the mask
        mask_tool = MaskTool(video_path)
        mask_tool.frame_id = frame_to_edit
        mask_tool.target_frame = frame
        mask_tool.run()

        # Load zones after saving
        self.load_zones_for_frame(frame_to_edit)

        frame_id = frame_to_edit
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            
            # Draw all user-defined masks
            for zone_id, polygon in self.zones.items():
                cv2.polylines(frame, [polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"Zone {zone_id}", tuple(polygon[0].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            current_object_ids = set()  # Keep track of IDs seen in the current frame
            
            for result in results:
                masks = result.masks.xy  # Get segmentation masks

                for mask, conf, class_id in zip(masks, result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    # Assign unique object ID
                    object_id = self.get_object_id(mask)
                    current_object_ids.add(object_id)  # Add ID to the set of seen IDs
                    
                    for zone_id, polygon in self.zones.items():
                        # Use masks directly for intersection check
                        if self.intersects(mask, polygon):
                            print(f"Frame {frame_id}: Object {class_id} (ID: {object_id}) with confidence {conf:.2f} intersects with Zone {zone_id}")
                            self.detection_log.append({
                                'frame_id': frame_id,
                                'object_id': object_id,
                                'class_id': class_id,
                                'confidence': conf,
                                'zone_id': zone_id
                            })
                            
                            # Update tracked object information
                            if object_id in self.tracked_objects:
                                self.tracked_objects[object_id]['last_seen'] = frame_id
                            else:
                                self.tracked_objects[object_id] = {
                                    'first_seen': frame_id,
                                    'last_seen': frame_id,
                                    'class_id': class_id,
                                    'zone_id': zone_id
                                }
                                
                            # Draw segmentation mask and label
                            cv2.polylines(frame, [mask.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
                            label = f"ID: {object_id} Class: {class_id} Conf: {conf:.2f}"
                            cv2.putText(frame, label, tuple(mask[0].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Remove objects that haven't been seen in a while (optional)
            self.remove_lost_objects(frame_id, current_object_ids)
            
            cv2.imshow('Zone Intersections', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
    
    def get_object_id(self, mask):
        for existing_id, existing_mask in self.object_ids.items():
            # Convert masks to binary images for accurate IOU calculation
            mask1_binary = self.mask_to_binary_image(mask)
            mask2_binary = self.mask_to_binary_image(existing_mask)

            # Resize masks to have the same shape before calculating IOU
            mask1_resized = cv2.resize(mask1_binary, (mask2_binary.shape[1], mask2_binary.shape[0]))

            iou = self.calculate_iou(mask1_resized, mask2_binary)
            if iou > self.iou_threshold:
                return existing_id

        new_id = self.next_object_id
        self.object_ids[new_id] = mask
        self.next_object_id += 1
        return new_id

    def mask_to_binary_image(self, mask):
        # Create a blank binary image
        binary_image = np.zeros((np.max(mask[:, 1]).astype(int) + 1, np.max(mask[:, 0]).astype(int) + 1), dtype=np.uint8)

        # Fill the polygon defined by the mask
        cv2.fillPoly(binary_image, [mask.astype(np.int32)], 255)

        return binary_image

    def calculate_iou(self, mask1, mask2):
        # Calculate Intersection over Union (IOU) between two binary masks
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union

    def remove_lost_objects(self, current_frame_id, current_object_ids):
        # Remove objects that haven't been seen for a certain number of frames
        for object_id in list(self.tracked_objects.keys()):
            if object_id not in current_object_ids and current_frame_id - self.tracked_objects[object_id]['last_seen'] > self.max_lost_frames: # Use self.max_lost_frames
                del self.tracked_objects[object_id]
                
    def save_detection_log(self):
        # Create a list of dictionaries for the DataFrame
        detection_log = []
        for object_id, data in self.tracked_objects.items():
            detection_log.append({
                'object_id': object_id,
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen'],
                'class_id': data['class_id'],
                'zone_id': data['zone_id']
            })

        df = pd.DataFrame(detection_log)
        df.to_csv('detection_log.csv', index=False)
        print("Detection log saved to detection_log.csv")

# Usage
if __name__ == "__main__":
    video_source = 'mask_tool\\test_source\\traffic2.mp4'
    model_path = 'mask_tool\\YoLo\\model\\yolov8n-seg.onnx'
    mask_csv_path = 'mask_tool\\result\\mask_positions.csv'
    frame_to_edit = int(input("Enter the frame number to edit: "))

    tracker = ZoneIntersectionTracker(model_path, mask_csv_path)
    tracker.track_intersections(video_source, frame_to_edit)