# parking-lot-vision

Per-slot parking occupancy from a fixed camera. Every slot gets an index `0..N-1`.
Each frame reports: total slots, empty count, occupied count, which indices are empty,
which are occupied, and where the empty ones are.

```
Total: 12   Empty: 5   Occupied: 7
Empty    -> [1, 4, 6, 9, 11]
Occupied -> [0, 2, 3, 5, 7, 8, 10]
Empty locations -> {1: (412, 233), 4: (620, 233), ...}   # slot centroid, px
```

## 1. How it works

```
video/camera ──> YOLO detect (vehicle classes) ──> box centers
                                                      │
slots.json (polygons, index = list order) ────────────┼──> pointPolygonTest per slot
                                                      │
                                              per-slot vote buffer (N frames)
                                                      │
                                              LotState (metrics JSON)
                                                      │
                                       ┌──────────────┴──────────────┐
                                  overlay window / mp4          stdout / --json-out
```

Occupancy rule (same as Ultralytics `ParkingManagement`): slot is **occupied** if any
detected vehicle's bounding-box center lies inside the slot polygon. Otherwise **empty**.
A per-slot majority vote over the last `--vote` frames removes flicker.
During startup, voting uses the frames available so far; ties count as empty.

## 2. Reference research → what we take

| Reference | What it does | What we reuse |
|---|---|---|
| [Ultralytics parking-management](https://docs.ultralytics.com/guides/parking-management) | `ParkingPtsSelection` tkinter picker → `bounding_boxes.json`; `ParkingManagement` counts Occupancy/Available via box-center `cv2.pointPolygonTest` | Picker tool as-is. JSON schema as-is. Occupancy rule. **Not** the class itself: it exposes only aggregate counts, no per-slot index. |
| [freedomwebtech/yolo11-parkinglot](https://github.com/freedomwebtech/yolo11-parkinglot) | Vendored copy of the Ultralytics class, `yolo11s.pt`, `classes=[2]`, frames resized to 1080×600, `img.py` dumps a frame for annotation | Frame-extract step. Lesson: annotate on the **same resolution** you infer on. |
| [verjin-dev/Car_Parking_Space_Detection](https://github.com/verjin-dev/Car_Parking_Space_Detection) | Classical OpenCV: adaptive threshold + pixel count per bay (threshold 900 px), click picker → pickle, `Available: n/total` overlay | Overlay text pattern. Kept as optional no-GPU fallback idea, not implemented in v1. |
| [PKLot on Roboflow](https://public.roboflow.com/object-detection/pklot) | 12,416 surveillance images, classes `space-empty` / `space-occupied`, CC BY 4.0 | Phase 2 fine-tune data (direct slot-state classifier instead of car detector). |
| [Labellerr fine-tune notebook](https://github.com/Labellerr/Hands-On-Learning-in-Computer-Vision/blob/main/fine-tune%20YOLO%20for%20various%20use%20cases/Fine-Tune-YOLO-for-Parking-Space-Monitoring.ipynb) | Fine-tunes YOLOv8 (`epochs=200 imgsz=640 batch=30`), occupancy via IoU of vehicle mask vs slot polygon | Training recipe for Phase 2. IoU rule as alternative to box-center for tight/angled slots. |

## 3. Pipeline — precise steps

### Step 0. Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install python-tk        # macOS, needed by the picker (apt: python3-tk)
```

### Step 1. Grab a reference frame
```bash
python -m parking_vision.tools.extract_frame data/lot.mp4 data/ref.jpg --width 1280
```
Use the exact `--width` you will pass to `run`. Polygon coords are in pixels of this frame.

### Step 2. Annotate slots (index = click order)
```bash
python -m parking_vision.tools.pick_slots          # opens Ultralytics ParkingPtsSelection
```
Upload `data/ref.jpg`, click 4 corners per slot, **Save** → `bounding_boxes.json` in CWD.
Move it to `config/slots.json`. Slot `i` = i-th polygon in the file. Re-order the list if
you want a specific numbering.
The included `config/slots.json` contains example coordinates; replace them with your
own annotations before monitoring a real lot. Polygons may contain three or more points.

`config/slots.json`
```json
[
  {"points": [[100, 200], [180, 200], [180, 260], [100, 260]]},
  {"points": [[190, 200], [270, 200], [270, 260], [190, 260]]}
]
```

### Step 3. Run
```bash
python -m parking_vision.main data/lot.mp4 --slots config/slots.json \
    --model yolo11s.pt --width 1280 --vote 5 --show --json-out out/state.jsonl --save out/lot.mp4
```
Source may be a file path, RTSP URL, or webcam index (`0`).
Frame indices start at zero; `--every 3` reports frames 0, 3, 6, and so on.
Output directories are created automatically. JSONL files are appended to; MP4 files
are overwritten. Saved video uses the source FPS (25 if unavailable), so frame
skipping produces a shorter video. Model weights download on first use if absent;
use a local weights path for offline inference.

### Step 4. Consume metrics
One JSON object per processed frame on stdout (and `--json-out`):
```json
{
  "frame": 120,
  "total": 12,
  "empty_count": 5,
  "occupied_count": 7,
  "empty": [1, 4, 6, 9, 11],
  "occupied": [0, 2, 3, 5, 7, 8, 10],
  "empty_locations": {"1": [412, 233], "4": [620, 233]},
  "slots": [{"index": 0, "occupied": true, "centroid": [140, 230]}, ...]
}
```

### Step 5 (optional, Phase 2). Fine-tune on PKLot
This section describes future work. `--mode`, `--rule`, and `--iou-thresh` are not
implemented in v1.
Only if the COCO detector misses cars from a high/oblique camera.
```bash
# download PKLot (YOLOv8 format) from Roboflow into data/pklot
yolo detect train data=data/pklot/data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=32
```
Then run with `--model runs/detect/train/weights/best.pt --mode slotcls`: the model
predicts `space-empty`/`space-occupied` boxes directly; a slot's state = class of the
prediction whose center is inside it.

## 4. Program structure

```
parking-lot-vision/
├── README.md
├── PROMPT.md                  # implementation prompt for a coding agent
├── requirements.txt
├── config/
│   └── slots.json             # polygons, index = list position
├── data/                      # videos, ref frames (gitignored)
├── out/                       # json / mp4 outputs (gitignored)
├── parking_vision/
│   ├── __init__.py
│   ├── slots.py               # load_slots(path) -> list[Slot]; Slot(index, polygon, centroid)
│   ├── occupancy.py           # Detector (YOLO wrapper) + assign(boxes, slots) -> raw bool[]
│   │                          #   + VoteBuffer(n) + LotState.from_flags()
│   ├── annotate.py            # draw(frame, slots, state, boxes) -> frame
│   ├── main.py                # CLI: source loop, wiring, stdout/json/mp4 outputs
│   └── tools/
│       ├── extract_frame.py   # video -> ref.jpg at target width
│       └── pick_slots.py      # thin wrapper: solutions.ParkingPtsSelection()
└── tests/
    └── test_occupancy.py      # assign() + VoteBuffer + LotState on synthetic boxes
```

### Module contracts

| Module | Function | In | Out |
|---|---|---|---|
| `slots.py` | `load_slots(path)` | JSON path | `list[Slot]`, index by position, centroid = mean of points |
| `occupancy.py` | `Detector(model, classes, conf, device).boxes(frame)` | BGR frame | `ndarray (K,4)` xyxy |
| `occupancy.py` | `assign(boxes, slots)` | boxes, slots | `list[bool]` length N (box center in polygon) |
| `occupancy.py` | `VoteBuffer(n).push(flags)` | raw flags | smoothed flags (majority of last n) |
| `occupancy.py` | `LotState.from_flags(slots, flags, frame_idx)` | flags | dataclass with all metrics + `to_dict()` |
| `annotate.py` | `draw(frame, slots, state, boxes)` | | frame with green/red polygons, index labels, header text |
| `main.py` | `main(argv)` | CLI args | exit code |

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `source` | required | file / RTSP / webcam index |
| `--slots` | `config/slots.json` | polygon file |
| `--model` | `yolo11s.pt` | any Ultralytics detect weights |
| `--classes` | `2 3 5 7` | COCO car, motorcycle, bus, truck |
| `--conf` | `0.25` | detection threshold |
| `--width` | `1280` | resize width before inference; must match annotation frame |
| `--vote` | `5` | frames in majority-vote buffer (1 = off) |
| `--every` | `1` | process every k-th frame |
| `--show` | off | cv2 window, `q` quits |
| `--save` | none | output mp4 path |
| `--json-out` | none | JSONL path |
| `--device` | auto | `cpu`, `0`, `mps` |

## 5. Tuning knobs (real cameras drift)

- **Camera moved** → re-run Step 1–2. Slots are pixel-fixed.
- **Slot flickers** → raise `--vote`.
- **Truck spans two slots** → box-center rule marks only one. Switch to IoU rule
  (`--rule iou --iou-thresh 0.3`) in Phase 2.
- **Missed cars from steep angle** → lower `--conf` first, then Phase 2 fine-tune.
- **Slow on CPU** → `--model yolo11n.pt --every 3 --width 960`.

## 6. Testing

```bash
pytest tests/
```
`test_occupancy.py` builds 3 square slots, feeds synthetic boxes, asserts:
`assign` marks exactly the hit slot; `VoteBuffer(3)` flips only after 2/3 agreeing
frames; `LotState` counts and index lists are consistent (`sorted(empty + occupied) == list(range(N))`).
Tests also check polygon ordering, overlay colors, frame extraction, frame skipping,
JSONL append, MP4 decoding, and cleanup. They use synthetic video and a fake detector;
no model is instantiated or downloaded, and network connections are blocked.

## 7. Out of scope (v1)

Multi-camera, database persistence, web dashboard, license plates, automatic slot
discovery. Add when needed.
