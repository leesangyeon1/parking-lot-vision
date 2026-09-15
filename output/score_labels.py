"""Score the detector against manual-labels.json and regenerate images/*-output.png.

Run from the repository root:  .venv/bin/python output/score_labels.py [--model yolo11s-obb.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parking_vision.annotate import draw
from parking_vision.occupancy import Detector, LotState, assign
from parking_vision.slots import load_slots

OUT = Path(__file__).resolve().parent
PHOTOS = ROOT / "Example photos" / "example photos"
# manual-labels.json names the original screenshots; these were renamed after labeling.
RENAMED = {3: "nigh1.png", 5: "nigh2.png", 6: "park1.png", 7: "park2.png"}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolo11s-obb.pt")
parser.add_argument("--imgsz", type=int, default=1280)
parser.add_argument("--conf", type=float, default=0.25)
args = parser.parse_args()

detector = Detector(args.model, None, args.conf, "cpu", args.imgsz)
wrong = total = 0
for record in json.load(open(OUT / "manual-labels.json")):
    k, expected = record["image"], record["expected_occupied"]
    path = PHOTOS / RENAMED.get(k, Path(record["source"]).name)
    if not path.exists():
        print(f"image {k}: source missing, skipped ({path.name})")
        continue
    frame = cv2.imread(str(path))
    frame = cv2.resize(frame, (1280, round(frame.shape[0] * 1280 / frame.shape[1])))
    slots = load_slots(OUT / record["slot_file"])
    boxes = detector.boxes(frame)
    flags = assign(boxes, slots)
    state = LotState.from_flags(slots, flags, 0)
    errors = [i for i, (e, f) in enumerate(zip(expected, flags)) if e != f]
    wrong += len(errors)
    total += len(expected)
    cv2.imwrite(str(OUT / "images" / f"{k:02d}-output.png"), draw(frame, slots, state, boxes))
    print(f"image {k}: vehicles={len(boxes)} occupied={state.occupied} empty={state.empty} wrong_slots={errors}")
print(f"accuracy {total - wrong}/{total} = {(total - wrong) / total:.3f}  (model={args.model}, imgsz={args.imgsz}, conf={args.conf})")
