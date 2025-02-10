import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, frame_numbers, points):
    """Extracts specified frames from a video file and draws a polygon on them.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save the extracted frames.
        frame_numbers (list): List of frame numbers to extract.
        points (list): List of points to draw a polygon in the format [(x1, y1), (x2, y2), ...].
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in frame_numbers:
            cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Extracted {saved_count} frames to {output_folder}")

# Example usage
video_file = "traffic-guard\src\output.mp4"
output_directory = "traffic-guard\src"
selected_frames = [1]  # User-selected frame numbers
polygon_points = [(631, 603), (474, 662), (720, 751), (910, 622)]
extract_frames(video_file, output_directory, selected_frames, polygon_points)
