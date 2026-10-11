"""In-repo clip eval for motion detectors (Tier 2).

Drives the mog2 (OpenCV MOG2) and stock improved motion detectors against
local video clips and reports trigger % and coverage % per clip, using only
cv2 + numpy. This is the in-Frigate substitute for the benchmark harness
comparison (see FRIGATE_OCV_MOG2.md section 10).

Metrics:
    trigger %   = frames with >= 1 motion box / total frames * 100
    coverage %  = mean over frames of (ROI union area / frame area) * 100

Usage (from the repo root, inside the Frigate environment):
    python3 -m frigate.test.eval_mog2_motion --detector mog2 --video <path> [<path> ...]
    python3 frigate/test/eval_mog2_motion.py --detector mog2 --persistence 0 --contrast off --video <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# allow direct script invocation (python3 frigate/test/eval_mog2_motion.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from frigate.config.camera.motion import MotionConfig
from frigate.motion.cv2_mog2_motion import Cv2Mog2MotionDetector
from frigate.motion.improved_motion import ImprovedMotionDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip eval for the mog2 / improved motion detectors"
    )
    parser.add_argument(
        "--video", nargs="+", required=True, help="clip path(s) to evaluate"
    )
    parser.add_argument(
        "--detector",
        choices=["mog2", "improved"],
        default="mog2",
        help="detector to evaluate (default: mog2)",
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=0,
        help="mog2 persistence_frames; 0 disables the gate (ignored for improved)",
    )
    parser.add_argument(
        "--contrast",
        choices=["on", "off"],
        default="on",
        help="contrast normalization: motion.improve_contrast, shared by both detectors",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="motion.frame_height; 0 = config default (100)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        help="override motion.contour_area, shared by both detectors (0 = config default)",
    )
    parser.add_argument(
        "--morph",
        choices=["on", "off"],
        default="on",
        help="morphology on/off (mog2 only)",
    )
    parser.add_argument(
        "--morph-iterations",
        type=int,
        default=1,
        help="morphology iterations (3x3 kernel, default 3)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="limit frames per clip (0 = all)",
    )
    return parser.parse_args()


def eval_clip(path: str, args: argparse.Namespace) -> dict:
    """Run one clip through the detector and collect the metrics."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 5)
    frame_shape = (height, width)

    config = MotionConfig()
    contrast_on = args.contrast == "on"
    object.__setattr__(
        config,
        "rasterized_mask",
        np.full(frame_shape, 255, dtype=np.uint8),
    )
    # shared motion settings, both detectors read them
    config.improve_contrast = contrast_on
    if args.height:
        config.frame_height = args.height
    if args.min_area:
        config.contour_area = args.min_area

    if args.detector == "mog2":
        config.mog2.persistence_frames = args.persistence
        if args.morph == "off":
            config.mog2.morphology.enabled = False
        else:
            config.mog2.morphology.iterations = args.morph_iterations
        detector = Cv2Mog2MotionDetector(frame_shape, config, fps, name="eval")
    else:
        detector = ImprovedMotionDetector(frame_shape, config, fps, name="eval")

    total = 0
    triggered = 0
    coverage_sum = 0.0
    total_boxes = 0
    canvas = np.zeros((height, width), np.uint8)
    start = time.perf_counter()

    while True:
        ret, bgr = cap.read()
        if not ret or (args.max_frames and total >= args.max_frames):
            break
        total += 1

        # feed both detectors the production frame format (YUV420p)
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        boxes = detector.detect(frame)
        if boxes:
            triggered += 1
            canvas[:] = 0
            for (x1, y1, x2, y2) in boxes:
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                if x2 > x1 and y2 > y1:
                    canvas[y1:y2, x1:x2] = 255
            coverage_sum += cv2.countNonZero(canvas) / (height * width)
        total_boxes += len(boxes)

    elapsed = time.perf_counter() - start
    cap.release()

    return {
        "clip": Path(path).name,
        "frames": total,
        "trigger_pct": 100.0 * triggered / max(total, 1),
        "coverage_pct": 100.0 * coverage_sum / max(total, 1),
        "avg_boxes": total_boxes / max(total, 1),
        "ms_frame": elapsed * 1000.0 / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    print(
        f"detector={args.detector} persistence={args.persistence} "
        f"contrast={args.contrast} height={args.height or 'default'} "
        f"min_area={args.min_area or 'default'} "
        f"morph={'off' if args.morph == 'off' else f'3x3x{args.morph_iterations}'}"
    )
    header = (
        f"{'clip':<24} {'frames':>7} {'trigger%':>9} {'coverage%':>10} "
        f"{'avg boxes':>10} {'ms/frame':>9}"
    )
    print(header)
    print("-" * len(header))
    for video in args.video:
        stats = eval_clip(video, args)
        print(
            f"{stats['clip']:<24} {stats['frames']:>7} "
            f"{stats['trigger_pct']:>9.2f} {stats['coverage_pct']:>10.2f} "
            f"{stats['avg_boxes']:>10.2f} {stats['ms_frame']:>9.1f}"
        )


if __name__ == "__main__":
    main()
