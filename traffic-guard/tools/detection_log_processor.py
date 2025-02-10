import json
import logging
import os
import cv2
import numpy as np
from typing import Tuple, List, Optional

def get_frame_from_timestamp(video_path: str, timestamp: float) -> np.ndarray:
    """
    Retrieves a frame from the video at a specific timestamp.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video source '{video_path}'.")
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    if not ret:
        logging.warning(f"Could not retrieve frame at timestamp {timestamp} from '{video_path}'.")
        cap.release()
        return None

    cap.release()
    return frame

def crop_object_from_frame(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Crops the object from the frame using the bounding box coordinates.
    Assumes bbox format: (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    try:
        x_center_norm, y_center_norm, width_norm, height_norm = bbox
        frame_height, frame_width = frame.shape[:2]

        # Scale normalized coordinates to pixel values.
        x_center = x_center_norm * frame_width
        y_center = y_center_norm * frame_height
        width = width_norm * frame_width
        height = height_norm * frame_height

        # Calculate top-left and bottom-right corners.
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Ensure coordinates are within frame boundaries.
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)

        cropped_object = frame[y1:y2, x1:x2]
        return cropped_object
    except Exception as e:
        logging.error(f"Error cropping object from frame: {e}")
        return None

def save_cropped_image(cropped_object: np.ndarray, output_path: str) -> None:
    """
    Saves the cropped image to the specified output path.
    """
    try:
        if cropped_object is not None:
            cv2.imwrite(output_path, cropped_object)
            logging.info(f"Cropped image saved to {output_path}")
        else:
            logging.warning("No cropped object to save.")
    except Exception as e:
        logging.error(f"Error saving cropped image: {e}")

def process_detection_log(*, detection_log_path: str, video_path: str, output_dir: str,
                          get_frame_from_timestamp_func, crop_object_from_frame_func, save_cropped_image_func,
                          events_to_process: Optional[List[str]] = None) -> None:
    """
    Processes selected events (enter, exit, enter_movement, exit_movement,
    no_entry, no_parking, wrong_way, etc.) from detection_log.json.

    Parameters:
        detection_log_path: Path to the detection log JSON file.
        video_path: Path to the video file.
        output_dir: Directory to save the cropped images.
        get_frame_from_timestamp_func: Helper function to retrieve a frame.
        crop_object_from_frame_func: Helper function to crop an object.
        save_cropped_image_func: Helper function to save the cropped image.
        events_to_process: Optional list of event names to process. If None, process all events.
    """
    try:
        with open(detection_log_path, 'r') as f:
            detection_log_list = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load detection log from {detection_log_path}: {e}")
        return

    if not detection_log_list:
        logging.info("Detection log is empty. No images to extract.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory ensured at: {output_dir}")

    # Sort entries by last_seen timestamp (ascending).
    detection_log_list.sort(key=lambda entry: float(entry.get('last_seen', 0)))

    # Cache last known good normalized bbox for each object id.
    last_bbox_by_object = {}

    for entry in detection_log_list:
        try:
            event = entry.get('event', 'unknown')
            # Only process if event is in events_to_process (if provided).
            if events_to_process and event not in events_to_process:
                continue

            last_seen = float(entry.get('last_seen', 0))
            object_id = entry.get('object_id', 'unknown')
            logging.info(f"Processing event '{event}' for object {object_id} at timestamp {last_seen}")
            logging.debug(f"Raw entry: {entry}")

            frame = get_frame_from_timestamp_func(video_path, last_seen)
            if frame is None:
                logging.error(f"Could not retrieve a frame for timestamp {last_seen} for event {event}")
                continue
            else:
                logging.debug(f"Frame retrieved (shape: {frame.shape}) for event {event}")

            bbox_norm = entry.get('bbox')
            if not bbox_norm or len(bbox_norm) != 4:
                logging.error(f"Invalid bounding box found for entry: {entry}")
                continue

            logging.debug(f"Raw normalized bbox from log: {bbox_norm}")
            frame_height, frame_width = frame.shape[:2]
            # Compute bbox width and height in pixels.
            width_pixels = bbox_norm[2] * frame_width
            height_pixels = bbox_norm[3] * frame_height
            area = width_pixels * height_pixels
            area_threshold = 10  # minimum acceptable area in pixels

            # For events with potentially unreliable bbox, use cached bbox if available.
            if event in ["wrong_way", "no_entry", "no_parking"] and area < area_threshold:
                if object_id in last_bbox_by_object:
                    logging.info(f"Using cached bbox for event '{event}' for object {object_id}")
                    bbox_norm = last_bbox_by_object[object_id]
                else:
                    logging.warning(f"No cached bbox available for object {object_id} on event '{event}'; using original bbox.")

            # Cache the bbox for events that are not in the problematic group OR if area is acceptable.
            if event not in ["wrong_way", "no_entry", "no_parking"] or area >= area_threshold:
                last_bbox_by_object[object_id] = bbox_norm

            cropped_img = crop_object_from_frame_func(frame, tuple(bbox_norm))
            if cropped_img is None:
                logging.error("Failed to crop the object from the frame.")
                continue

            text = f"id:{object_id}-{event}"
            # Annotate the cropped image with the event type.
            cv2.putText(cropped_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2, cv2.LINE_AA)

            filename = f"{event}_{object_id}_{int(last_seen)}.jpg"
            output_path = os.path.join(output_dir, filename)

            save_cropped_image_func(cropped_img, output_path)
            logging.info(f"Saved event image to {output_path}")
        except Exception as ex:
            logging.error(f"Error processing detection log entry: {ex}")