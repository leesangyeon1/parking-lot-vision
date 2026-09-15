# Implementation prompt — parking-lot-vision

Paste this into a coding agent (or follow it yourself). README.md is the spec; this prompt
is the build order and acceptance criteria.

---

## Goal

Build a Python CLI that watches a parking-lot video/camera and reports, per frame:
total slots, empty count, occupied count, list of empty slot indices, list of occupied
slot indices, and the pixel location (centroid) of every empty slot. Slots are indexed
`0..N-1` by their order in `config/slots.json`. Render an overlay that draws each slot
polygon green (empty) or red (occupied) with its index, plus a header line
`Total N | Empty E | Occupied O`.

## Constraints

- Python 3.10+. Dependencies: `ultralytics`, `opencv-python`, `numpy`, `pytest`. Nothing else.
- Package layout exactly as in README §4. No extra abstraction layers, no config classes,
  no plugin systems. One implementation per concept.
- Occupancy rule: a slot is occupied if the center `((x1+x2)/2, (y1+y2)/2)` of any
  detected vehicle box satisfies `cv2.pointPolygonTest(poly, center, False) >= 0`.
  Break on first hit per slot.
- Detection: `ultralytics.YOLO(model).predict(frame, classes=classes, conf=conf,
  device=device, verbose=False)`. No tracking needed.
- Temporal smoothing: `VoteBuffer(n)` keeps last `n` boolean vectors; output is
  per-slot majority. `n=1` means pass-through.
- Slots JSON schema: `[{"points": [[x, y], ...]}, ...]` (Ultralytics
  `ParkingPtsSelection` output). Accept 4+ points per polygon.
- Frames are resized to `--width` (keep aspect) **before** inference and drawing. Slot
  coordinates are already in that resolution. Do not rescale polygons.
- Metrics JSON schema (README §3 Step 4) is a contract; do not rename keys.
- Every `LotState` must satisfy `sorted(empty + occupied) == list(range(total))`.

## Build order

1. `parking_vision/slots.py`
   - `@dataclass Slot: index:int, polygon:np.ndarray (P,1,2) int32, centroid:tuple[int,int]`
   - `load_slots(path) -> list[Slot]`. Raise `ValueError` on empty file or polygon with <3 points.

2. `parking_vision/occupancy.py`
   - `class Detector: __init__(model, classes, conf, device); boxes(frame) -> np.ndarray (K,4) float`
   - `assign(boxes, slots) -> list[bool]`
   - `class VoteBuffer: __init__(n); push(flags) -> list[bool]` (use `collections.deque(maxlen=n)`)
   - `@dataclass LotState` with fields `frame, total, empty_count, occupied_count, empty,
     occupied, empty_locations, slots`; `from_flags(slots, flags, frame_idx)`; `to_dict()`.

3. `tests/test_occupancy.py` — write before step 4. Cover:
   - `assign`: box centered in slot 1 of 3 → `[False, True, False]`; box outside all → all False;
     box on polygon edge → True (`>= 0`).
   - `VoteBuffer(3)`: pushes `[T],[F],[T]` → `[T]`; `[T],[F],[F]` → `[F]`.
   - `LotState.from_flags`: counts, index lists, `empty_locations` keys are strings of
     empty indices, invariant `sorted(empty+occupied)==range(total)`.
   - `load_slots`: rejects `[]` and a 2-point polygon.
   No model download in tests. Do not instantiate `Detector` in tests.

4. `parking_vision/annotate.py`
   - `draw(frame, slots, state, boxes) -> frame` (mutates in place, returns it).
   - Polygon `cv2.polylines` thickness 2, green `(0,255,0)` empty, red `(0,0,255)` occupied.
   - Index label `str(slot.index)` at centroid, `cv2.putText` with dark filled background rect.
   - Small magenta dot at each box center.
   - Header at top-left: `Total {t} | Empty {e} | Occupied {o}` and second line
     `Empty: {empty list}`.

5. `parking_vision/main.py`
   - `argparse` with flags from README §4 table. `source` parsed as int if it is all digits.
   - Loop: read → skip unless `frame_idx % every == 0` → resize to width → detect → assign →
     vote → `LotState` → print `json.dumps(state.to_dict())` → optional JSONL append →
     `draw` → optional `cv2.imshow` (quit on `q`) → optional `VideoWriter`.
   - Open `VideoWriter` lazily on first processed frame using resized shape and source fps
     (fallback 25).
   - Release everything in `finally`.

6. `parking_vision/tools/extract_frame.py`
   - `python -m parking_vision.tools.extract_frame <video> <out.jpg> [--width W] [--frame K]`
   - Reads frame K (default 0), resizes to W (default 1280), writes jpg, prints shape.

7. `parking_vision/tools/pick_slots.py`
   - Body: `from ultralytics import solutions; solutions.ParkingPtsSelection()`.
   - Print a one-line reminder afterwards: move `bounding_boxes.json` to `config/slots.json`.

8. `requirements.txt`, `.gitignore` (`data/`, `out/`, `*.pt`, `.venv/`, `runs/`).

## Acceptance

- `pytest tests/` passes with no network.
- `python -m parking_vision.main data/lot.mp4 --slots config/slots.json --show` runs on a
  sample video, prints one JSON line per frame, overlay shows numbered green/red slots
  and the header counts.
- `--json-out out/state.jsonl` produces valid JSONL; `--save out/lot.mp4` produces a
  playable file.
- Changing the order of entries in `slots.json` changes slot numbering and nothing else.
- Code total (excluding tests) under ~250 lines. If you exceed it, you over-built.

## Do not

- Do not add a database, web server, YAML config, logging framework, or class hierarchy.
- Do not use `ultralytics.solutions.ParkingManagement` for inference; it hides per-slot
  state. The picker (`ParkingPtsSelection`) is fine.
- Do not implement Phase 2 (PKLot fine-tune, IoU rule) unless asked.
