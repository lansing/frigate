"""Parse two line_profiler output files and print a comparison table."""

import re
import sys


def parse_total_time(path):
    with open(path) as f:
        for line in f:
            m = re.search(r"Total time:\s+([0-9.]+)\s+s", line)
            if m:
                return float(m.group(1))
    return None


def sum_line_times(path, *patterns):
    total = 0.0
    found = False
    with open(path) as f:
        for line in f:
            for pat in patterns:
                if re.search(pat, line):
                    # line_profiler columns: linenum  hits  time  per_hit  pct  content
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            total += float(parts[2])
                            found = True
                        except ValueError:
                            pass
                    break
    return total if found else None


OPTIMIZATIONS = [
    (
        "percentile (opt 1)",
        # original: two np.percentile calls
        [r"np\.percentile"],
        # optimized: single dispatch to the histogram method
        [r"percentiles\("],
    ),
    (
        "LUT contrast (opt 2)",
        # original: np.clip over the full frame + floating-point arithmetic + astype
        [r"resized_frame = np\.clip", r"avg_min\) / \(avg_max - avg_min\)"],
        # optimized: lut construction (np.arange path + clip + astype) + cv2.LUT application
        [r"np\.arange\(256\)", r"cv2\.LUT\("],
    ),
    (
        "masking     (opt 3)",
        [r"self\.mask\]\s*="],
        [r"cv2\.bitwise_and\("],
    ),
    (
        "gaussian    (opt 4)",
        [r"gaussian_filter\("],
        [r"gaussian\("],
    ),
]


def main(orig_path, opt_path):
    orig_total = parse_total_time(orig_path)
    opt_total = parse_total_time(opt_path)

    if orig_total is None or opt_total is None:
        print("ERROR: could not parse Total time from profile output", file=sys.stderr)
        sys.exit(1)

    # Infer frame count from hits on a line that runs every frame
    frames = None
    with open(orig_path) as f:
        for line in f:
            if "resized_frame = cv2.resize" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # hits column may be doubled (two detectors); take half if even
                        hits = int(parts[1])
                        frames = hits // 2 if hits % 2 == 0 else hits
                    except ValueError:
                        pass
                break
    if frames is None:
        frames = round(orig_total / 0.002)  # rough fallback

    COL = 24
    SEP = "─" * 60

    print()
    print(f"  Motion detector benchmark  ·  {frames} frames")
    print(f"  ghcr.io/blakeblackshear/frigate:0.17.1")
    print(f"  {SEP}")
    print(f"  {'':24}  {'Before':>10}  {'After':>10}  {'Speedup':>8}")
    print(f"  {SEP}")

    for label, orig_pats, opt_pats in OPTIMIZATIONS:
        o = sum_line_times(orig_path, *orig_pats)
        n = sum_line_times(opt_path, *opt_pats)
        if o and n:
            print(f"  {label:{COL}}  {o/1e6:>8.1f}ms  {n/1e6:>8.1f}ms  {o/n:>7.1f}x")
        else:
            print(f"  {label:{COL}}  (pattern not matched — check profile output)")

    print(f"  {SEP}")
    print(
        f"  {'Total detect()':24}  {orig_total:>8.3f}s   {opt_total:>8.3f}s"
        f"   {orig_total / opt_total:>6.1f}x"
    )
    o_ms = orig_total / frames * 1000
    n_ms = opt_total / frames * 1000
    print(f"  {'Per frame':24}  {o_ms:>8.2f}ms  {n_ms:>8.2f}ms")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} profile_original.txt profile_optimized.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
