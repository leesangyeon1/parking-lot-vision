import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Slot:
    index: int
    polygon: np.ndarray
    centroid: tuple[int, int]


def load_slots(path) -> list[Slot]:
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Slots must be a nonempty list of polygons")
    slots = []
    for index, entry in enumerate(entries):
        points = np.asarray(entry.get("points", []) if isinstance(entry, dict) else [], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"Slot {index} must contain at least three [x, y] points")
        slots.append(Slot(index, points.reshape(-1, 1, 2), tuple(map(int, points.mean(axis=0)))))
    return slots
