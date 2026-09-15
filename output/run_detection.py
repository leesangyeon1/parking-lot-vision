"""Manual image evaluation; run from the repository root with .venv/bin/python."""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parking_vision.occupancy import Detector

OUT = Path(__file__).resolve().parent
model = Detector(str(ROOT / 'yolo11s.pt'), [2, 3, 5, 7], 0.25, 'cpu')
records = []
for number, path in enumerate(sorted((ROOT / 'Example photos').rglob('*.png')), 1):
    frame = cv2.imread(str(path))
    original_shape = list(frame.shape)
    height = max(1, round(frame.shape[0] * 1280 / frame.shape[1]))
    frame = cv2.resize(frame, (1280, height))
    boxes = model.boxes(frame)
    preview = frame.copy()
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.circle(preview, (int((x1+x2)/2), int((y1+y2)/2)), 3, (255, 0, 255), -1)
        cv2.putText(preview, str(index), (int(x1), max(20, int(y1))), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
    cv2.rectangle(preview, (0, 0), (680, 35), (25,25,25), -1)
    cv2.putText(preview, f'Image {number} | Vehicle detections: {len(boxes)} | conf 0.25', (10,24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
    assert cv2.imwrite(str(OUT / f'{number:02d}-detections.jpg'), preview)
    record = dict(image=number, source=str(path.relative_to(ROOT)), original_shape=original_shape,
                  resized_shape=list(frame.shape), vehicle_detections=len(boxes), boxes=boxes.tolist())
    records.append(record)
    print(f'Image {number}: {len(boxes)} vehicles', flush=True)
(OUT / 'detections.json').write_text(json.dumps(dict(model='yolo11s.pt', classes=[2,3,5,7], conf=.25, width=1280, images=records), indent=2))
