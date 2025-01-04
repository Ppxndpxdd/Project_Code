import cv2
import numpy as np

def draw_polygon(img, polygon, point_radius, color=(0, 255, 0)):
    if len(polygon) > 0:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)
        for p in polygon:
            p = ensure_point_format(p)
            cv2.circle(img, p, point_radius, color, -1)

def ensure_point_format(point):
    return tuple(map(int, point))

def draw_zone_ids(img, zones, point_radius):
    for zone_id, polygon in zones.items():
        if len(polygon) > 0:
            draw_polygon(img, polygon, point_radius, color=(0, 255, 0))
            pts = np.array(polygon, np.float32)
            M = cv2.moments(pts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(img, f'Zone {zone_id}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

def draw_instructions(img):
    instructions = [
        "Press 'e' to toggle Editing Mode",
        "Press 's' to Save Mask",
        "Press 'u' to Undo",
        "Press 'r' to Redo",
        "Press 'q' to Quit"
    ]
    for i, instruction in enumerate(instructions):
        cv2.putText(img, instruction, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def draw_axes(img):
    height, width = img.shape[:2]

    # Draw x-axis (start at bottom-left)
    cv2.line(img, (0, height - 1), (width, height - 1), (255, 0, 0), 2)  # Blue line
    cv2.putText(img, 'X', (width - 20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw y-axis (start at bottom-left)
    cv2.line(img, (0, 0), (0, height), (0, 255, 0), 2)  # Green line
    cv2.putText(img, 'Y', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def is_within_bounds(point, img, frame_shape):
    """Checks if a point is within the image boundaries."""
    px, py = point
    height, width = frame_shape[:2]
    return 0 <= px < width and 0 <= py < height

def is_point_near_polygon_point(point, polygon, threshold):
    for p in polygon:
        if np.linalg.norm(np.array(p) - np.array(point)) < threshold:
            return True
    return False

def get_next_available_zone_id(zones):
    if len(zones) == 0:
        return 1
    return max(zones.keys()) + 1

def find_closest_point_index(polygon, point):
    min_distance = float('inf')
    closest_index = -1
    for i, p in enumerate(polygon):
        distance = np.linalg.norm(np.array(p) - np.array(point))
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    return closest_index