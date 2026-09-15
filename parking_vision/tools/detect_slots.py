"""Auto-generate slots.json from painted stall lines in a top-down frame.

    python -m parking_vision.tools.detect_slots data/ref.jpg config/slots.json [--width 1280]
        [--model yolo11s-obb.pt] [--preview out/slots.png]

Painted lines give each row's orientation, pitch and extent; vehicle detections extend
rows whose end stalls are occupied (lines hidden under cars). Output is a starting point:
review the preview, then delete or reorder entries in the JSON by hand.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parking_vision.occupancy import Detector  # noqa: E402


def line_mask(gray):
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    return cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)[1]


def segments(mask, min_len):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, 30, minLineLength=min_len, maxLineGap=6)
    return segs.reshape(-1, 4).astype(float) if segs is not None else np.zeros((0, 4))


def dominant_angle(segs):
    """Angle (deg) of the most common segment direction, folded to [-45, 45)."""
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 90
    hist, edges = np.histogram(ang, bins=90, range=(0, 90))
    a = edges[np.argmax(hist)] + 0.5
    return a - 90 if a >= 45 else a


def rotate(points, angle, center):
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.transform(np.asarray(points, float).reshape(-1, 1, 2), m).reshape(-1, 2)


def cluster_1d(values, tol):
    """Group sorted scalars whose gaps are <= tol; return group means."""
    values = np.sort(np.asarray(values, float))
    if not len(values):
        return []
    groups, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def rows_from_segments(horiz, spines, min_len):
    """Split segments at vertical spines, then union segments whose x-intervals overlap."""
    parts = []
    for x1, y, x2 in horiz:
        # a back-to-back divider crosses its spine near the middle; edge-aligned clutter (car sides) does not
        cuts = [x1] + [s for s in spines if x1 + 0.3 * (x2 - x1) < s < x2 - 0.3 * (x2 - x1)] + [x2]
        parts += [(a, y, b) for a, b in zip(cuts, cuts[1:]) if b - a >= min_len]
    parts.sort(key=lambda p: (p[0] + p[2]) / 2)
    rows = []
    for x1, y, x2 in parts:
        for row in rows:
            lo, hi = max(x1, row["x1"]), min(x2, row["x2"])
            if hi - lo >= 0.5 * min(x2 - x1, row["x2"] - row["x1"]):
                row["ys"].append(y)
                row["x1s"].append(x1), row["x2s"].append(x2)
                row["x1"], row["x2"] = np.percentile(row["x1s"], 10), np.percentile(row["x2s"], 90)
                break
        else:
            rows.append({"x1": x1, "x2": x2, "x1s": [x1], "x2s": [x2], "ys": [y]})
    return rows


def estimate_pitch(diffs, lo=12, hi=160):
    """Period that explains the gaps best, allowing hidden lines (gaps of 1-3 periods)."""
    diffs = np.asarray([d for d in diffs if d >= lo], float)
    if not len(diffs):
        return None
    best = (0, None)
    for p in np.arange(lo, hi, 0.5):
        k = np.round(diffs / p)
        fit = np.clip(1 - np.abs(diffs / p - k) / 0.15, 0, None) * (k >= 1) * (k <= 3)
        score = np.sum(fit / np.maximum(k, 1))  # harmonics p/2 score half as much
        if score >= best[0]:
            best = (score, float(p))
    return best[1]


def fill_row(ys, pitch, extra_points=()):
    """Snap to detected lines, interpolate missing ones, extend to cover extra_points."""
    lo, hi = min(ys), max(ys)
    for p in extra_points:
        lo, hi = min(lo, p - pitch / 2), max(hi, p + pitch / 2)
    n = int(round((hi - lo) / pitch))
    grid = lo + pitch * np.arange(n + 1)
    ys = np.asarray(ys)
    snapped = [ys[np.argmin(np.abs(ys - g))] if np.min(np.abs(ys - g)) < 0.35 * pitch else g for g in grid]
    return [float(v) for v in snapped]


def _slots_for_angle(frame, angle, min_len, car_pts, segs):
    """Slots whose divider lines lie at `angle` (deg); returns polygons in frame coords."""
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    # rotate so stall divider lines are horizontal; work in that frame, rotate polygons back
    pts = rotate(segs.reshape(-1, 2), angle, center).reshape(-1, 4)
    dx, dy = pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 1]
    horizontal = np.abs(dy) <= 0.05 * np.abs(dx)
    vertical = np.abs(dx) <= 0.05 * np.abs(dy)
    horiz = [(min(a, c), (b + d) / 2, max(a, c)) for a, b, c, d in pts[horizontal]]
    # spine = collinear vertical pieces (dashed or broken center line) whose lengths sum to >= 6 stall lines
    vx, vlen = (pts[vertical][:, 0] + pts[vertical][:, 2]) / 2, np.abs(dy[vertical])
    spines = [x for x in cluster_1d(vx, 6) if vlen[np.abs(vx - x) <= 6].sum() >= 6 * min_len]
    cars = rotate(car_pts, angle, center) if len(car_pts) else []
    rows = rows_from_segments(horiz, spines, min_len)
    gaps = []
    for row in rows:
        row["ys"] = cluster_1d(row["ys"], 4)
        row["cars"] = sorted(p[1] for p in cars if row["x1"] <= p[0] <= row["x2"])
        row["gaps"] = list(np.diff(row["ys"])) + list(np.diff(row["cars"]))
        gaps += row["gaps"]
    global_pitch = estimate_pitch(gaps)
    if global_pitch is None:
        return []
    slots = []
    for row in rows:
        ys, x1, x2 = row["ys"], row["x1"], row["x2"]
        if len(ys) < 3:
            continue
        pitch = estimate_pitch(row["gaps"], 0.7 * global_pitch, 1.4 * global_pitch) or global_pitch
        ratio = (x2 - x1) / pitch  # stall depth / width: ~2 for a single row, ~4 for back-to-back rows
        if not 1.4 <= ratio <= 6.0:  # rejects hatching, crosswalks, aisles
            continue
        halves = [(x1, (x1 + x2) / 2), ((x1 + x2) / 2, x2)] if ratio > 3.6 else [(x1, x2)]  # no spine found
        ys = fill_row(ys, pitch, row["cars"])
        for xa, xb in halves:
            for a, b in zip(ys, ys[1:]):
                quad = rotate([(xa, a), (xb, a), (xb, b), (xa, b)], -angle, center)
                slots.append({"points": [[int(round(x)), int(round(y))] for x, y in quad]})
    return slots


def detect_slots(frame, min_len=25, model=None, angle=None):
    """Return (slots, divider_angle). Tries the dominant line angle and its perpendicular,
    keeps whichever yields more slots. Pass `angle` to force one (mixed-orientation lots:
    run once per angle and concatenate the JSON files)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segs = segments(line_mask(gray), min_len)
    if not len(segs):
        return [], 0.0
    car_pts = []
    if model:
        boxes = Detector(model, None, 0.25, None).boxes(frame)
        if len(boxes):
            car_pts = (boxes[:, :2] + boxes[:, 2:]) / 2
            # car edges look like short painted lines: drop segments whose midpoint is inside a vehicle
            mid = (segs[:, :2] + segs[:, 2:]) / 2
            shrink = 0.15 * (boxes[:, 2:] - boxes[:, :2])
            lo, hi = boxes[:, :2] + shrink, boxes[:, 2:] - shrink
            inside = ((mid[:, None, :] >= lo[None]) & (mid[:, None, :] <= hi[None])).all(axis=2).any(axis=1)
            segs = segs[~inside]
    candidates = [angle] if angle is not None else [dominant_angle(segs), dominant_angle(segs) + 90]
    best = max(((_slots_for_angle(frame, a, min_len, car_pts, segs), a) for a in candidates), key=lambda t: len(t[0]))
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("out", help="slots.json path")
    p.add_argument("--width", type=int, default=1280, help="resize width; must match main.py --width")
    p.add_argument("--min-len", type=int, default=25, help="min stall line length in px")
    p.add_argument("--model", default="yolo11s-obb.pt", help="detector used to extend occupied row ends; '' to skip")
    p.add_argument("--angle", type=float, help="force divider-line angle in degrees (default: auto, best of two)")
    p.add_argument("--preview", help="write annotated png here")
    args = p.parse_args(argv)
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Cannot read {args.image}")
    frame = cv2.resize(frame, (args.width, round(frame.shape[0] * args.width / frame.shape[1])))
    slots, angle = detect_slots(frame, args.min_len, args.model or None, args.angle)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(slots, indent=2))
    print(f"{len(slots)} slots, stall lines at {angle:.1f} deg -> {args.out}")
    if args.preview:
        for i, s in enumerate(slots):
            poly = np.array(s["points"], np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [poly], True, (0, 255, 255), 1)
            cx, cy = poly.reshape(-1, 2).mean(axis=0).astype(int)
            cv2.putText(frame, str(i), (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imwrite(args.preview, frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
