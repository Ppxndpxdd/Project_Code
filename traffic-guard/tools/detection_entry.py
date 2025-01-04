from dataclasses import dataclass

@dataclass
class DetectionEntry:
    frame_id: int
    object_id: int
    class_id: int
    confidence: float
    zone_id: int
    first_seen: float
    last_seen: float = None
    duration: float = None
    event: str = None
    bbox: tuple = None  # Add bounding box data