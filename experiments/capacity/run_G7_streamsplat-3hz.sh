#!/usr/bin/env bash
# G7 (LOG.md): at three keyframes per user per second, close to the paper's keyframe rate, does MPS serve more
# StreamSplat users than time-slicing? Exploratory lane. Plan `misty-painting-summit`, approved 2026-11-13.
# Runs only after G4 passes; it is the first entry cut if time runs short.
#
# Setup: as G6, but period 1/3 s and N in 1 2 3 4 6. The period is passed as 0.3333333333333333, for which the
# expected count ceil(60 / period) and the updates the user loop releases are both 180 (bench/capacity.py:128 is
# only off for periods such as 0.3).
#
# Registered before the run:
#   SLO p95 latency <= 1/3 s, no OOM, 180/180 updates. Capacity and stop rule as G6.
#   Prediction: per-user GPU share about 0.09 / (1/3) = 27%, the low-utilisation regime where G2's refuted
#   "MPS serves at least one more user" gets its real test: ts 3, mps 3-4. Refuted if mps equals ts in both repeats.
#
# Usage: bash experiments/capacity/run_G7_streamsplat-3hz.sh [metrics]
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G7_OUT:-outputs/capacity/G7}
. bench/sweep_lib.sh
if [ "${1:-}" = metrics ]; then metrics ts mps; exit; fi
check_no_mps
PERIOD=0.3333333333333333
sweep_arms "1 2 3 4 6" --model streamsplat --compile --period "$PERIOD" --slo "$PERIOD" --duration 60 --warmup 3
metrics ts mps
echo "G7 done"
