import sys
from collections import deque
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass

import cv2
import numpy as np


class Detector:
    def __init__(self, model, classes, conf, device, imgsz=None):
        with redirect_stdout(sys.stderr):
            from ultralytics import YOLO
            self.model = YOLO(model)
        if self.model.task not in {"detect", "obb"}:
            raise ValueError("Use a detection or oriented-box model")
        vehicles = {"car", "motorcycle", "bus", "truck", "small vehicle", "large vehicle"}
        self.classes = classes if classes is not None else [
            i for i, name in self.model.names.items() if name.replace("-", " ") in vehicles]
        if not self.classes or any(i not in self.model.names for i in self.classes):
            raise ValueError("Specify valid vehicle class IDs with --classes for this model")
        self.conf, self.device, self.imgsz = conf, device, imgsz

    def boxes(self, frame) -> np.ndarray:
        size = self.imgsz or (1920 if frame.shape[0] > frame.shape[1] else 1280)
        with redirect_stdout(sys.stderr):
            result = self.model.predict(frame, classes=self.classes, conf=self.conf,
                                        device=self.device, imgsz=size, verbose=False)[0]
        detections = result.obb if result.obb is not None else result.boxes
        return detections.xyxy.cpu().numpy().astype(float).reshape(-1, 4)


def assign(boxes, slots) -> list[bool]:
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    flags = []
    for slot in slots:
        occupied = False
        for center in centers:
            if cv2.pointPolygonTest(slot.polygon, tuple(map(float, center)), False) >= 0:
                occupied = True
                break
        flags.append(occupied)
    return flags


class VoteBuffer:
    def __init__(self, n):
        if n < 1:
            raise ValueError("Vote window must be positive")
        self.history = deque(maxlen=n)

    def push(self, flags) -> list[bool]:
        """Vote over available frames; ties are empty (no strict majority)."""
        if self.history and len(flags) != len(self.history[0]):
            raise ValueError("Slot count changed within the vote window")
        self.history.append(list(flags))
        return (np.sum(self.history, axis=0) > len(self.history) / 2).tolist()


@dataclass
class LotState:
    frame: int
    total: int
    empty_count: int
    occupied_count: int
    empty: list[int]
    occupied: list[int]
    empty_locations: dict[str, list[int]]
    slots: list[dict]

    def __post_init__(self):
        if sorted(self.empty + self.occupied) != list(range(self.total)):
            raise ValueError("Empty and occupied indices must partition all slots")

    @classmethod
    def from_flags(cls, slots, flags, frame_idx):
        if len(slots) != len(flags):
            raise ValueError("Expected one occupancy flag per slot")
        empty = [s.index for s, flag in zip(slots, flags) if not flag]
        occupied = [s.index for s, flag in zip(slots, flags) if flag]
        locations = {str(s.index): list(s.centroid) for s, flag in zip(slots, flags) if not flag}
        details = [dict(index=s.index, occupied=bool(flag), centroid=list(s.centroid))
                   for s, flag in zip(slots, flags)]
        return cls(frame_idx, len(slots), len(empty), len(occupied), empty, occupied, locations, details)

    def to_dict(self):
        return asdict(self)
