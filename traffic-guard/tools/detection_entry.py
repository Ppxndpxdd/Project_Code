from dataclasses import dataclass
from typing import Tuple

@dataclass
class DetectionEntry:
    def __init__(self, object_id, class_id, confidence, marker_id, first_seen, last_seen, duration, event, bbox):
        self.object_id = object_id
        self.class_id = class_id
        self.confidence = confidence
        self.marker_id = marker_id
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.duration = duration
        self.event = event
        self.bbox = bbox