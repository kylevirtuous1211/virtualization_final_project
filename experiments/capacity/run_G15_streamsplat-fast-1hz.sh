#!/usr/bin/env bash
# G15 (LOG.md): the last step of plan `misty-painting-summit` (approved 2026-11-18). At one keyframe per user per
# second, how many StreamSplat users does the RTX 6000 Ada serve with the components that passed G11-G14, against
# the serial server, and at what VRAM per user? Exploratory lane.
#
# Setup. G6's protocol: period 1 s, phases evenly staggered, 60 s per point, 3 warm-ups, compile off; SLO p95
# latency <= 1.0 s, 60/60 updates, no OOM; capacity = the largest N where both repeats pass (the largest N where any
# repeat passes is also reported). Two arms on the same server code, both batch 1, interleaved per N, 2 repeats:
#   serial = G9's server (--batch 1: TF32, no cache, checkpointing on, eager), the matched control;
#   fast   = --batch 1 plus every component that passed: --cache (G11), --bf16 (G12, quality gate passed), --nosync
#            (G13, kept as C4's prerequisite), --no-checkpointing (G14's prerequisite) and --graphs (G14, 1.05x at
#            batch 1). Batching across users is not a passed component (G9), so fast stays at batch 1.
# Grid N in 1 2 4 5 6 7 8 10 12 (5 and 7 added to G6's grid); an arm stops after two consecutive N where neither
# repeat passes. A capacity at the grid's edge (12) is a bound, not a capacity.
# Step 0, before the sweep: each arm back to back at N = 1 (--period 0 --max-updates 20 --warmup 3), which for a
# batch-1 server is its saturated throughput. Gate: fast >= 6.0 updates/s (G9's 4.92 x 1.22), else the capacity
# could move from 4 to at most 5: no sweep, a negative result.
#
# Registered in plan mode (plan approved 2026-11-18, before G10-G14 ran) and kept unchanged, although G11-G14's
# numbers (fast at about 11.5 updates/s back to back) were known when this header was written (estimates):
#   serial capacity 4 (5 is possible; G6 never ran 5). Refuted if neither 4 nor 5.
#   fast capacity 6 (5-8). Refuted if fast's capacity is <= serial's.
#   Saturated throughput (step 0): serial 4.7-4.9, fast 7-9 updates/s.
#   p95 latency at N = 4 (mean of the repeats' worst user): serial 0.19-0.21 s, fast 0.11-0.16 s. Refuted if fast is
#     not below 0.9x serial.
#   GPU utilisation at saturation (the largest N the arm ran): fast >= 85%. Refuted if < 78%.
#   VRAM per served user (gpu_mem_mib_max / N), reported at every N both arms ran. Refuted if fast's exceeds 2x
#     serial's at any such N (the trade would not pay).
#
# Usage: bash experiments/capacity/run_G15_streamsplat-fast-1hz.sh           (resumable)
#        bash experiments/capacity/run_G15_streamsplat-fast-1hz.sh metrics   (re-derive metrics.json)
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G15_OUT:-outputs/capacity/G15}
. bench/sweep_lib.sh
FAST_FLAGS="--cache --bf16 --nosync --no-checkpointing --graphs"
COMMON=(--model streamsplat --no-compile)
report() {
  metrics serial fast
  "$PY" - "$OUT/metrics.json" <<'EOF'
import json, sys
metrics = json.load(open(sys.argv[1]))
points, step0 = metrics["points"], metrics.get("step0_updates_per_s", {})
def mean(arm, n, key):
    runs = [run[key] for run in points[arm].get(str(n), {}).values() if run[key] is not None]
    return sum(runs) / len(runs) if runs else None
print("capacity:", metrics["capacity"])
print("step 0, back to back at N = 1:", {arm: round(value, 2) for arm, value in step0.items()})
p95 = {arm: mean(arm, 4, "lat_p95_max") for arm in ("serial", "fast")}
print("p95 latency at N = 4 (s):", {arm: value and round(value, 3) for arm, value in p95.items()},
      "fast/serial = %.3f" % (p95["fast"] / p95["serial"]) if all(p95.values()) else "")
for arm in ("serial", "fast"):
    largest = max(map(int, points[arm]))
    print(f"{arm}: largest N run {largest}, GPU utilisation there {mean(arm, largest, 'gpu_util_mean'):.0f}%")
shared = sorted(set(map(int, points["serial"])) & set(map(int, points["fast"])))
print("VRAM per user (MiB), serial / fast:",
      {n: (round(mean("serial", n, "vram_per_user_mib")), round(mean("fast", n, "vram_per_user_mib"))) for n in shared})
EOF
}
if [ "${1:-}" = metrics ]; then report; exit; fi
check_no_mps
for arm in serial fast; do
  flag=(--batch 1)
  [ "$arm" = fast ] && flag=(--batch 1 $FAST_FLAGS)
  measure "$OUT/step0/$arm" contaminated "${COMMON[@]}" --n 1 --period 0 --max-updates 20 --warmup 3 "${flag[@]}" || exit 1
done
metrics serial fast > /dev/null
gate=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["step0_updates_per_s"]["fast"] >= 6.0)' "$OUT/metrics.json")
if [ "$gate" != True ]; then
  echo "gate: fast runs below 6.0 updates/s back to back; no sweep, as registered"
  report
  exit 0
fi
ARMS="serial fast" sweep_arms "1 2 4 5 6 7 8 10 12" "${COMMON[@]}" --period 1 --slo 1 --duration 60 --warmup 3
report
echo "G15 done"
