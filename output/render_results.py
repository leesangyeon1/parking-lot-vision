"""Render cached model results through the actual occupancy and overlay modules."""
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from parking_vision.annotate import draw
from parking_vision.occupancy import LotState, VoteBuffer, assign
from parking_vision.slots import load_slots

records = json.loads((OUT / 'detections.json').read_text())['images']
records = sorted(records, key=lambda r: (r['image'] == 1, r['image']))
labels = {r['image']: r for r in json.loads((OUT / 'manual-labels.json').read_text())}
summary, states, cards = [], [], []
for record in records:
    i = record['image']
    frame = cv2.imread(str(ROOT / record['source']))
    frame = cv2.resize(frame, tuple(record['resized_shape'][1::-1]))
    boxes = np.asarray(record['boxes'], dtype=float).reshape(-1, 4)
    if i in labels:
        label = labels[i]
        slots = load_slots(OUT / label['slot_file'])
        flags = VoteBuffer(1).push(assign(boxes, slots))
        state = LotState.from_flags(slots, flags, 0)
        assert sorted(state.empty + state.occupied) == list(range(state.total))
        expected = label['expected_occupied']
        false_empty = [s.index for s, truth, predicted in zip(slots, expected, flags) if truth and not predicted]
        false_occupied = [s.index for s, truth, predicted in zip(slots, expected, flags) if not truth and predicted]
        draw(frame, slots, state, boxes)
        states.append(dict(image=i, source=record['source'], state=state.to_dict()))
        (OUT / f'{i:02d}-state.json').write_text(json.dumps(state.to_dict(), indent=2))
        description = f'Selected bays: {state.total}. Visually occupied: {sum(expected)}. Predicted occupied: {state.occupied_count}. False-empty bays: {false_empty}.'
        result = dict(image=i, source=record['source'], detected_vehicles=len(boxes), sampled_slots=state.total,
                      visually_occupied=sum(expected), predicted_occupied=state.occupied_count,
                      false_empty=false_empty, false_occupied=false_occupied)
    else:
        description = 'Calendar screenshot: negative control. No parking bays annotated; 0 vehicles detected.'
        cv2.rectangle(frame, (0,0), (1280,36), (25,25,25), -1)
        cv2.putText(frame, 'Negative control | Calendar | Vehicle detections: 0', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 1)
        result = dict(image=i, source=record['source'], detected_vehicles=len(boxes), sampled_slots=None)
    output = f'images/{i:02d}-output.png'
    assert cv2.imwrite(str(OUT / output), frame)
    result['output_image'] = output
    summary.append(result)
    cards.append(f'<section><h2>Image {i}</h2><p>{html.escape(Path(record["source"]).name)}</p><p>{html.escape(description)}</p><a href="{output}"><img src="{output}" alt="Image {i} actual occupancy output"></a></section>')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'states.jsonl').write_text(''.join(json.dumps(s['state'])+'\n' for s in states))
(OUT / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Example photo test outputs</title>
<style>body{font:16px system-ui;background:#151719;color:#f1f1f1;margin:32px;max-width:1400px}h1{font-size:32px}p{line-height:1.5;color:#d5d5d5}section{border-top:1px solid #555;margin-top:36px;padding-top:16px}img{width:100%;max-width:1280px;height:auto}a{color:#83d9ff}.notice{background:#4a2d20;padding:20px;border-radius:8px}</style>
<h1>Example photo test outputs</h1><p>YOLO11s · confidence 0.25 · vehicle classes 2, 3, 5, 7 · width 1280 · CPU · vote 1</p>
<div class="notice"><strong>Result: vehicle detection failed on these overhead parking photos.</strong><p>All six parking images returned zero vehicle detections. Green polygons are the program's predictions and include false empty results. Polygons cover manually selected visible bays, not every bay in the lot. The calendar image is a negative control.</p></div>
''' + ''.join(cards) + '</html>')
print(json.dumps(summary, indent=2))
