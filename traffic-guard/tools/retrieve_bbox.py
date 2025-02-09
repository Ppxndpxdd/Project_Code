import cv2
import json
import os

def extract_frame_at_timestamp(video_path, output_dir, json_data):
    """
    Extracts an image from a video at the "last_seen" timestamp and crops it using the provided bounding box.
    :param video_path: Path to the video file
    :param output_dir: Directory to save the extracted image
    :param json_data: JSON data containing last_seen and bbox
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Extract timestamp and bbox from JSON
    last_seen = json_data.get("last_seen", 0)
    x, y, w, h = json_data.get("bbox", [0, 0, 0, 0])
    
    # Set the frame position to the last_seen timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, last_seen * 1000)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to extract frame at {last_seen:.2f} seconds.")
        cap.release()
        return
    
    frame_height, frame_width, _ = frame.shape
    
    # Ensure bbox values are within the frame dimensions
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    
    x2, y2 = min(x + w, frame_width), min(y + h, frame_height)
    cropped_frame = frame[y:y2, x:x2]
    
    if cropped_frame.size == 0:
        print("Error: Cropped frame is empty. Check bbox values.")
        cap.release()
        return
    
    # Save the extracted frame
    output_filename = os.path.join(output_dir, f"object_{json_data['object_id']}_frame.jpg")
    cv2.imwrite(output_filename, cropped_frame)
    print(f"Saved cropped frame to {output_filename}")
    
    cap.release()

if __name__ == "__main__":
    video_path = "traffic-guard\src\output.mp4"  # Change this to your actual video file
    output_dir = "extracted_images"
    
    json_data = {
        "object_id": 2,
        "class_id": 2,
        "confidence": 0.8887564539909363,
        "marker_id": 1,
        "first_seen": 0.06666666666666667,
        "last_seen": 0.06666666666666667,
        "duration": None,
        "event": "enter_movement",
        "bbox": [
            685.3544921875,  # x
            395.4864501953125,  # y
            1032.69091796875,  # width
            601.85791015625  # height
        ],
        "id_rule_applied": None
    }
    
    extract_frame_at_timestamp(video_path, output_dir, json_data)
