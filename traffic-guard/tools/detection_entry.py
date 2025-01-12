from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class DetectionEntry:
    object_id: int
    class_id: int
    confidence: float
    marker_id: int
    first_seen: float
    last_seen: Optional[float]
    duration: Optional[float]
    event: str
    bbox: Tuple[float, float, float, float]