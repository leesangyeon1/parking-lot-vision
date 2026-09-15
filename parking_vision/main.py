import argparse
import json
import math
from pathlib import Path

import cv2

from .annotate import draw
from .occupancy import Detector, LotState, VoteBuffer, assign
from .slots import load_slots


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report per-slot parking occupancy as JSONL.")
    parser.add_argument("source", help="Video path, RTSP URL, or webcam index")
    parser.add_argument("--slots", default="config/slots.json")
    parser.add_argument("--model", default="yolo11s-obb.pt",
                        help="yolo11s-obb.pt (DOTA aerial, top-down cameras) or yolo11s.pt (COCO, ground cameras)")
    parser.add_argument("--classes", nargs="+", type=int, default=None,
                        help="Vehicle class IDs; default inferred from model names")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference resolution")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--vote", type=int, default=5)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", help="output .mp4 (video) or .png/.jpg (single image source)")
    parser.add_argument("--json-out")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if min(args.width, args.imgsz, args.vote, args.every) < 1 or not 0 <= args.conf <= 1:
        parser.error("width, imgsz, vote, every must be positive; conf must be between 0 and 1")
    slots, vote = load_slots(args.slots), VoteBuffer(args.vote)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    writer = json_file = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {args.source}")
        detector = Detector(args.model, args.classes, args.conf, args.device, args.imgsz)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25
        for path in (args.json_out, args.save):
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        if args.json_out:
            json_file = open(args.json_out, "a", encoding="utf-8")
        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % args.every:
                continue
            height = max(1, round(frame.shape[0] * args.width / frame.shape[1]))
            frame = cv2.resize(frame, (args.width, height))
            boxes = detector.boxes(frame)
            state = LotState.from_flags(slots, vote.push(assign(boxes, slots)), frame_idx)
            line = json.dumps(state.to_dict())
            print(line, flush=True)
            if json_file:
                json_file.write(line + "\n")
                json_file.flush()
            draw(frame, slots, state, boxes)
            if args.show:
                cv2.imshow("Parking occupancy", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.save and Path(args.save).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if not cv2.imwrite(args.save, frame):  # image source -> annotated image
                    raise ValueError(f"Cannot write image: {args.save}")
            elif args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise ValueError(f"Cannot open video output: {args.save}")
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if json_file is not None:
            json_file.close()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
