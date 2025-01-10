import cv2
import numpy as np
import json
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
from sklearn.cluster import DBSCAN
import os

# PLICP Tracker for reference line matching
class PLICPTracker:
    def __init__(self, reference_polylines, max_distance=50):
        self.reference_polylines = reference_polylines
        self.max_distance = max_distance
        self.line_segments = self._create_line_segments()

    def _create_line_segments(self):
        segments = []
        for polyline in self.reference_polylines:
            line_segments = [(polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1)]
            segments.extend(line_segments)
        return segments

    def _point_to_line_distance(self, point, line_segment):
        p1, p2 = line_segment
        line_vec = p2 - p1
        point_vec = point - p1
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_proj_len = np.dot(point_vec, line_unitvec)

        if point_proj_len < 0:
            return np.linalg.norm(point - p1), p1
        elif point_proj_len > line_len:
            return np.linalg.norm(point - p2), p2
        closest_point = p1 + line_unitvec * point_proj_len
        return np.linalg.norm(point - closest_point), closest_point

    def match_trajectory(self, detected_points):
        matched_points = []
        total_error = 0
        for point in detected_points:
            min_distance = float('inf')
            closest_match = None
            for segment in self.line_segments:
                distance, match_point = self._point_to_line_distance(point, segment)
                if distance < min_distance:
                    min_distance = distance
                    closest_match = match_point
            matched_points.append(closest_match)
            total_error += min_distance

        avg_error = total_error / len(detected_points)
        is_matching = avg_error < 10.0  # Tolerance threshold
        return np.array(matched_points), is_matching


class MarkerMovement:
    def __init__(self, video_source, frame_to_edit, mask_json_path='arrow_positions.json'):
        self.video_source = video_source
        self.frame_to_edit = frame_to_edit
        self.mask_json_path = mask_json_path

        self.cap = cv2.VideoCapture(video_source)
        self.arrow_points = []
        self.drawing = False
        self.mask_positions = []
        self.target_frame = None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_edit)
        ret, self.target_frame = self.cap.read()
        if not ret:
            raise ValueError("Unable to load the specified frame.")

    def draw_multipoint_arrow(self, img, arrow_points, color=(0, 255, 0)):
        if len(arrow_points) < 2:
            return
        pts = np.array(arrow_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2)

    def mouse_callback(self, event, x, y, flags, param):
        img = self.target_frame.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                self.arrow_points = []
                self.drawing = True
            self.arrow_points.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            temp_arrow = self.arrow_points + [(x, y)]
            self.draw_multipoint_arrow(img, temp_arrow)
            cv2.imshow('Draw Arrow', img)

        elif event == cv2.EVENT_RBUTTONDOWN and self.drawing:
            self.drawing = False
            self.draw_multipoint_arrow(img, self.arrow_points)
            self.mask_positions.append({"arrow": self.arrow_points})
            with open(self.mask_json_path, 'w') as f:
                json.dump(self.mask_positions, f, indent=4)
            print("Arrow saved!")

    def run(self):
        cv2.imshow('Draw Arrow', self.target_frame)
        cv2.setMouseCallback('Draw Arrow', self.mouse_callback)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_source = "mask_tool/test_source/stream2.mp4"
    frame_to_edit = 100
    mask_json_path = "arrow_positions.json"
    yolo_model_path = "mask_tool/code/YoLo/model/yolo11m.pt"
    object_trajectory_log = {}  # To store object trajectories
    non_matching_objects_log = []  # To store non-matching object logs

    # Load reference arrows (Polylines)
    with open(mask_json_path, 'r') as f:
        reference_polylines = [np.array(arrow['arrow']) for arrow in json.load(f)]

    # Initialize YOLO model and PLICP tracker
    model = YOLO(yolo_model_path)
    tracker = PLICPTracker(reference_polylines)

    cap = cv2.VideoCapture(video_source)
    trajectory_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection and tracking using YOLO
        results = model.track(frame)  # Using YOLOv8's track method
        for obj in results:
            obj_id = obj.get('track_id')  # Use the appropriate ID field for YOLOv8
            if obj_id is None:
                continue

            obj_center = obj.get('center', (0, 0))
            timestamp = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Capture timestamp/frame number

            if obj_id not in object_trajectory_log:
                object_trajectory_log[obj_id] = []

            object_trajectory_log[obj_id].append({"timestamp": timestamp, "coordinates": obj_center})

        # Check for disappeared objects and match with reference lines
        detected_ids = {obj.get('track_id') for obj in results}  # Collect IDs of objects detected in this frame
        missing_ids = set(object_trajectory_log.keys()) - detected_ids

        for missing_id in missing_ids:
            trajectory = object_trajectory_log[missing_id]
            last_position = trajectory[-1]['coordinates']

            # Try to match trajectory with reference arrows using PLICP
            matched_points, is_matching = tracker.match_trajectory([p['coordinates'] for p in trajectory])

            if not is_matching:
                # If trajectory doesn't match, log the object details
                non_matching_objects_log.append({
                    "id": missing_id,
                    "timestamp": trajectory[-1]['timestamp'],
                    "trajectory": trajectory
                })

            # Optionally remove the object from the log if you no longer want to track it after disappearing
            # del object_trajectory_log[missing_id]

        # Display the frame (with trajectory lines or other visualizations)
        for obj_id, trajectory in object_trajectory_log.items():
            points = [p['coordinates'] for p in trajectory]
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        # Show the frame with visualized tracking
        cv2.imshow("Tracked Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Optionally print non-matching objects for debugging
    print("Non-matching objects log:")
    for log_entry in non_matching_objects_log:
        print(log_entry)
