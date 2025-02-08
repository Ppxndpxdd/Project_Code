# Reorder fields so non-default arguments come before default arguments:

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class DetectionEntry:
    object_id: int
    class_id: int
    confidence: float
    marker_id: int
    first_seen: str
    last_seen: Optional[str]
    duration: Optional[float]
    event: str
    bbox: Tuple[float, float, float, float]
    id_rule_applied: Optional[int] = None