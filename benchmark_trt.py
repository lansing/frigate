#!/usr/bin/env python3
"""TensorRT detector benchmark. Mimics DetectorRunner.run() from object_detection/base.py."""

import argparse
import time

import numpy as np
import yaml

from frigate.config import FrigateConfig
from frigate.object_detection.base import LocalObjectDetector


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a TensorRT (or any) Frigate detector."
    )
    parser.add_argument(
        "--config", required=True, help="Path to frigate config.yml"
    )
    parser.add_argument(
        "--detector",
        default=None,
        help="Detector name from config (default: first detector)",
    )
    parser.add_argument(
        "--warmup", type=int, default=100, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Number of timed iterations"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    config = FrigateConfig(**raw)

    detectors = config.detectors
    if args.detector:
        detector_config = detectors[args.detector]
    else:
        detector_config = next(iter(detectors.values()))

    model = detector_config.model
    print(f"Detector: {detector_config.type}, model: {model.path}")
    print(f"Input shape: (1, {model.height}, {model.width}, 3)")

    detector = LocalObjectDetector(detector_config=detector_config)

    dummy = np.zeros((1, model.height, model.width, 3), dtype=np.uint8)

    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        detector.detect_raw(dummy)

    print(f"Benchmarking ({args.iters} iterations)...")
    latencies = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        detector.detect_raw(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = sum(latencies) / len(latencies)
    p50 = float(np.percentile(latencies, 50))
    p99 = float(np.percentile(latencies, 99))
    fps = 1000.0 / avg

    print("\n--- Benchmark Results ---")
    print(f"Iterations:      {args.iters}")
    print(f"Avg latency:     {avg:.3f} ms")
    print(f"P50 latency:     {p50:.3f} ms")
    print(f"P99 latency:     {p99:.3f} ms")
    print(f"Throughput:      {fps:.2f} FPS")
    print("-------------------------")


if __name__ == "__main__":
    main()
