#!/usr/bin/env bash
# Usage (from repo root):
#   docker run --rm --entrypoint /bin/bash \
#     -v "$(pwd)":/opt/frigate-dev \
#     ghcr.io/blakeblackshear/frigate:0.17.1 \
#     /opt/frigate-dev/benchmark/compare.sh
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BENCH_DIR/.." && pwd)"
ORIG_OUT="$BENCH_DIR/profile_original.txt"
OPT_OUT="$BENCH_DIR/profile_optimized.txt"

echo "==> Installing line_profiler..."
pip install --break-system-packages line_profiler -q 2>/dev/null

echo 'VERSION = "0.17.1"' > "$REPO_DIR/frigate/version.py"

mkdir -p /tmp/bench
cp "$BENCH_DIR/benchmark_motion.py" /tmp/bench/

echo "Profiling original v0.17.1"
PYTHONPATH=/opt/frigate python3 /tmp/bench/benchmark_motion.py > "$ORIG_OUT" 2>/dev/null

echo "Profiling optimized"
PYTHONPATH="$REPO_DIR" python3 /tmp/bench/benchmark_motion.py > "$OPT_OUT" 2>/dev/null

echo "Results"
python3 "$BENCH_DIR/compare_results.py" "$ORIG_OUT" "$OPT_OUT"
