from dataclasses import dataclass
from typing import Tuple

@dataclass
class DetectionEntry:
    def __init__(self, object_id: int, class_id: int, confidence: float, zone_id: int = None, movement_id: int = None, first_seen: float = None, last_seen: float = None, duration: float = None, event: str = None, bbox: Tuple[float, float, float, float] = None):
        self.object_id = object_id
        self.class_id = class_id
        self.confidence = confidence
        self.zone_id = zone_id
        self.movement_id = movement_id
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.duration = duration
        self.event = event
        self.bbox = bbox