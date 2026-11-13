#!/usr/bin/env bash
# G6 (LOG.md): how many StreamSplat users, one container each, does the RTX 6000 Ada serve at one keyframe per user
# per second, under time-slicing and under MPS? The StreamSplat analogue of G2. Exploratory lane.
# Plan `misty-painting-summit`, approved 2026-11-13. Runs only after G4 passes.
#
# Setup: as G5 with torch.compile on (the authors' default), period 1 s, 60 s per point, 3 warm-ups, evenly staggered
# phases; arms ts and mps (user-started MPS daemon; containers share it through the bind-mounted pipe directory,
# --ipc=host and the same UID) interleaved per N; 2 repeats; N in 1 2 4 6 8 10 12 14.
# ts is the matched control for mps; N = 1 is the slowdown reference.
#
# Registered before the run:
#   SLO p95 latency <= 1.0 s, no OOM, 60/60 updates. Capacity = largest N where both repeats pass (the largest N where
#   any repeat passes is also reported). An arm stops after two consecutive N where neither repeat passes.
#   Prediction: compute bound about 1 / 0.09 = 11, memory bound about 49,140 / 4,200 = 11, so ts 9 (scheduling
#   overhead) and mps 10-11: the limit moves from compute (G2) to memory. Refuted if ts >= 11 (no room for MPS to
#   gain) or if either capacity is <= 5. A capacity of 14 is a bound at the grid's edge, not a capacity.
#
# Usage: bash experiments/capacity/run_G6_streamsplat-1hz.sh [metrics]
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G6_OUT:-outputs/capacity/G6}
. bench/sweep_lib.sh
if [ "${1:-}" = metrics ]; then metrics ts mps; exit; fi
check_no_mps
sweep_arms "1 2 4 6 8 10 12 14" --model streamsplat --compile --period 1 --slo 1 --duration 60 --warmup 3
metrics ts mps
echo "G6 done"
