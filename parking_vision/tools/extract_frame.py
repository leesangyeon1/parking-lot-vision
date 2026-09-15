import argparse
from pathlib import Path

import cv2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract a reference frame for slot annotation.")
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args(argv)
    if args.width < 1 or args.frame < 0:
        parser.error("width must be positive and frame must be nonnegative")
    cap = cv2.VideoCapture(args.video)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read frame {args.frame} from {args.video}")
        height = max(1, round(frame.shape[0] * args.width / frame.shape[1]))
        frame = cv2.resize(frame, (args.width, height))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(args.output, frame):
            raise ValueError(f"Cannot write image: {args.output}")
        print(frame.shape)
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
