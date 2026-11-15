#!/usr/bin/env bash
# G9 (LOG.md): does one StreamSplat server that batches pending updates across users serve more users at one
# keyframe per user per second than the same server run serially? Exploratory lane. Plan `misty-painting-summit`,
# approved 2026-11-15 (design drafted by a Fable 5.1 Plan agent).
#
# Setup: bench/capacity.py sweep --model streamsplat --batch B: one container, one copy of the weights, N users in
# one process on G6's release schedules (period 1 s, evenly staggered phases, 60 s, latency = end - release).
# Each step takes the pending updates (at most one per user, since update j+1 needs j's keyframe), oldest release
# first, up to B, and runs them as one batch with every depth min-max per sample (greedy, no batching window).
# Warm-up: each user's first keyframe pair, then 3 rounds of every batch size the run can form.
# Arms: serial (B = 1) and batched (B = 8, the authors' batch size), interleaved per N, 2 repeats; serial is the
# matched control (only B differs). torch.compile is off in both (batch sizes vary; compile saved 4% in G5), so the
# batch-1 reference is G5 nocompile (0.198 s, 5.02 updates/s). G6's per-process ts (4 users) differs in two
# factors and is a bound only. Scope: periodic, evenly staggered users on one clip; no bursts, one resolution.
#
# Registered before any batch > 1 timing:
#   Step 0 (gate): back to back (--period 0, 20 updates, 3 warm-ups), N = B = b for b in 1 2 4 8, one run each;
#     throughput = updates / (last end - first start). In order: (1) b = 1 must be within 5% of 5.02 updates/s,
#     else the server adds overhead: a harness bug, stop (exit 3); (2) if b = 8 OOMs, stop (exit 2) and log it;
#     (3) if b = 8 gives < 1.2x the b = 1 throughput, batching cannot move capacity: stop (exit 2), log a negative.
#   Sweep: SLO p95 <= 1.0 s, no OOM, 60/60 updates; capacity = largest N where both repeats pass (any-repeat also
#     reported); an arm stops after two consecutive N where neither repeat passes; N in 1 2 4 5 6 8 10 12 14 16
#     (a capacity of 16 is a bound at the grid's edge, not a capacity).
#   P1 step 0: b = 8 throughput 2.5x b = 1 (10-17 updates/s), rising with b. Refuted if < 2.0x.
#   P2 serial capacity 4. Refuted if != 4 in both repeats.
#   P3 serial saturated throughput at N >= 6 is 4.9-5.1 updates/s (no multi-process loss). Refuted if <= 4.75.
#   P4 batched capacity 8 (6-11). Refuted if <= 5.
#   P5 batched saturated throughput at N >= 10 is >= 10 updates/s, with batches of 8. Refuted if < 8.
#   P6 serial p95 at N = 4 is 0.20-0.22 s. Refuted if > 0.25 s.
#   P7 at N <= 4 the batched arm's batch p50 is 1 and its p95 within 0.05 s of serial's; at its own edge its p95
#      is 0.6-1.0 s. Refuted if batched p95 exceeds serial p95 by > 0.05 s at any N <= 4.
#   P8 the server's GPU memory does not grow with N: gpu_mem_mib_max 6,000-10,000 MiB at every N (G6: N x 4,094).
#      Refuted if > 12,000 MiB at any N, or rising with N past 8.
#   Per-user VRAM in metrics.json is the one server's VRAM repeated N times; gpu_mem_mib_max is the number to read.
#
# Usage: bash experiments/capacity/run_G9_streamsplat-batched.sh           (resumable)
#        bash experiments/capacity/run_G9_streamsplat-batched.sh metrics   (re-derive metrics.json)
# Start it only after any long co-tenant job family has finished, not between its stages.
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G9_OUT:-outputs/capacity/G9}
. bench/sweep_lib.sh
if [ "${1:-}" = metrics ]; then metrics serial batched; exit; fi
check_no_mps
for batch in 1 2 4 8; do
  measure "$OUT/step0/b$batch" contaminated --model streamsplat --no-compile --n "$batch" --batch "$batch" \
    --period 0 --max-updates 20 --warmup 3 || exit 1
done
metrics serial batched
python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
throughput = json.loads((out / "metrics.json").read_text())["step0_updates_per_s"]
users = {batch: json.loads((out / "step0" / f"b{batch}" / "summary.json").read_text())["users"] for batch in (1, 2, 4, 8)}
print("step 0 updates/s:", {name: round(value, 2) if value else value for name, value in throughput.items()})
if any(user["rc"] not in (0, 3) for runs in users.values() for user in runs):
    sys.exit("step 0: a server crashed; read its server.log")
if abs(throughput["b1"] / 5.02 - 1) > 0.05:
    print("gate (1): b = 1 is not within 5% of G5 nocompile: the server adds overhead, a harness bug; stop")
    sys.exit(3)
if any(user["oom"] for user in users[8]):
    print("gate (2): b = 8 ran out of memory; stop (registered)")
    sys.exit(2)
if throughput["b8"] < 1.2 * throughput["b1"]:
    print("gate (3): b = 8 gives < 1.2x the b = 1 throughput; stop (registered)")
    sys.exit(2)
EOF
gate=$?
[ "$gate" = 0 ] || exit "$gate"
ARMS="serial batched" sweep_arms "1 2 4 5 6 8 10 12 14 16" --model streamsplat --no-compile --period 1 --slo 1 \
  --duration 60 --warmup 3
metrics serial batched
echo "G9 done"
