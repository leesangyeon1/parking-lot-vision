"""Evaluate the real pipeline against fixed view maps and independent visual labels."""
import hashlib
import html
import json
import sys
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output'
sys.path.insert(0,str(ROOT))
from parking_vision.annotate import draw
from parking_vision.occupancy import Detector,LotState,VoteBuffer,assign
from parking_vision.slots import load_slots

manifest=json.loads((ROOT/'config/examples/manifest.json').read_text())
old=Detector('yolo11s.pt',None,.25,'cpu',640)
new=Detector('yolo11s-obb.pt',None,.25,'cpu')
summary=[]
cards=[]
for item in manifest:
    path=ROOT/item['source']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError(f'Image changed: {path}. Review its slot map before evaluation.')
    frame=cv2.imread(str(path))
    frame=cv2.resize(frame,tuple(item['shape'][::-1]))
    slots=load_slots(ROOT/item['slots'])
    baseline=old.boxes(frame)
    boxes=new.boxes(frame)
    flags=VoteBuffer(1).push(assign(boxes,slots))
    state=LotState.from_flags(slots,flags,0)
    truth=item['expected_occupied']
    false_empty=[i for i,(p,t) in enumerate(zip(flags,truth)) if t is True and not p]
    false_occupied=[i for i,(p,t) in enumerate(zip(flags,truth)) if t is False and p]
    unknown=[i for i,t in enumerate(truth) if t is None]
    baseline_flags=assign(baseline,slots)
    row=dict(key=item['key'],source=item['source'],total=state.total,occupied=state.occupied_count,empty=state.empty_count,
             inference_size=1920 if frame.shape[0]>frame.shape[1] else 1280,
             baseline_vehicle_detections=len(baseline),vehicle_detections=len(boxes),
             visual_occupied=sum(t is True for t in truth),visual_unknown=unknown,
             false_empty=false_empty,false_occupied=false_occupied,
             baseline_false_empty=[i for i,(p,t) in enumerate(zip(baseline_flags,truth)) if t is True and not p],
             boxes=boxes.tolist())
    summary.append(row)
    draw(frame,slots,state,boxes)
    image_path=f'images/{item["key"]}-output.png'
    assert cv2.imwrite(str(OUT/image_path),frame)
    (OUT/f'{item["key"]}-state.json').write_text(json.dumps(state.to_dict(),indent=2)+'\n')
    cards.append(f'<section><h2>{html.escape(item["key"])}: Total {state.total} | Empty {state.empty_count} | Occupied {state.occupied_count}</h2><p>Detected vehicles: {len(baseline)} before → {len(boxes)} after. Known false-empty slots: {false_empty}; known false-occupied slots: {false_occupied}. Uncertain visual labels: {unknown}.</p><p>{html.escape(item["scope"])}</p><a href="{image_path}"><img src="{image_path}" alt="{html.escape(item["key"])} occupancy output"></a></section>')
    print(item['key'], 'total',state.total,'empty',state.empty_count,'occupied',state.occupied_count,'false_empty',false_empty,'false_occupied',false_occupied,flush=True)
(OUT/'fix-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parking occupancy — corrected image evaluation</title><style>body{font:16px system-ui;background:#151719;color:#f1f1f1;margin:32px;max-width:1400px}h1{font-size:32px}p{line-height:1.5;color:#d5d5d5}section{border-top:1px solid #555;margin-top:36px;padding-top:16px}img{width:100%;max-width:1280px;height:auto}a{color:#83d9ff}.notice{background:#163d35;padding:20px;border-radius:8px}</style><h1>Parking occupancy: corrected image evaluation</h1><p>YOLO11s-OBB · confidence 0.25 · vehicle classes inferred from model · inference auto (1280 landscape / 1920 portrait) · frame width 1280 · CPU</p><div class="notice"><strong>Fixed: aerial vehicle model, correct class mapping, and full visible-bay maps.</strong><p>Totals come from manually traced view-specific parking polygons. Aisles, hatch zones and unresolvable tree cover are excluded. Red/green are predictions; per-image errors below are measured against independent visual labels, with uncertain labels excluded.</p></div>'''+''.join(cards)+'</html>')
