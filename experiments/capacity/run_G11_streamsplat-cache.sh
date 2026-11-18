#!/usr/bin/env bash
# G11 (LOG.md): C1 of plan `misty-painting-summit` (approved 2026-11-18), built because G10 found the previous
# keyframe's one-frame work to be 21.5% of the model's time (rule: >= 12%). Does keeping each user's previous
# keyframe's one-frame features in the batching server (the paper's Algorithm 2) make an update cheaper, and what
# VRAM does it cost? Exploratory lane.
#
# Setup: bench/capacity.py sweep --model streamsplat --no-compile, one server, N = batch = b users back to back
# (--period 0 --max-updates 20 --warmup 3, G9's step 0), b in 1 4; arm `cache` adds --cache, arm `nocache` is the
# matched control (the same server, only the cache differs); arms interleaved, 2 repeats each. The cached update
# encodes only the new keyframe (bench/streamsplat_stages.py encode_frames) and normalises its depth per frame,
# then runs the pair stage with the user's cached features. TF32, as served so far.
# First probes/streamsplat_cache_equivalence.py must pass (G9's criterion at full FP32, plus every clip frame's
# disparity spanning exactly 0..255); if it fails the cache is wrong: a bug, stop (exit 3).
#
# Registered in plan mode before any cache timing (estimates):
#   P1 at b = 1 the cache raises back-to-back throughput from 4.92 to 5.6-6.2 updates/s. Refuted if < 5.3 (gain
#      < 8%); the cache then stays out of the final configuration.
#   P2 the cache's gain ratio at b = 4 is within 5 percentage points of b = 1's. Refuted if they differ by > 10.
#   P3 gpu_mem_mib_max at b = 4 grows by <= 250 MiB with the cache (4 users x about 37 MiB plus allocator slack).
#      Refuted if > 500 MiB.
#   metrics.json's "capacity" fields mean nothing at --period 0 (as in G5) and are not results.
#
# Usage: bash experiments/capacity/run_G11_streamsplat-cache.sh           (resumable)
#        bash experiments/capacity/run_G11_streamsplat-cache.sh metrics   (re-derive metrics.json)
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G11_OUT:-outputs/capacity/G11}
. bench/sweep_lib.sh
report() {
  metrics nocache cache
  "$PY" - "$OUT/metrics.json" <<'EOF'
import json, sys
points = json.load(open(sys.argv[1]))["points"]
def mean(arm, n, key):
    runs = points[arm][str(n)].values()
    return sum(run[key] for run in runs) / len(runs)
for n in (1, 4):
    plain, cached = mean("nocache", n, "updates_per_s"), mean("cache", n, "updates_per_s")
    memory = mean("cache", n, "gpu_mem_mib_max") - mean("nocache", n, "gpu_mem_mib_max")
    print(f"b = {n}: nocache {plain:.2f}, cache {cached:.2f} updates/s (x{cached / plain:.3f}); "
          f"gpu_mem_mib_max {memory:+.0f} MiB with the cache")
EOF
}
if [ "${1:-}" = metrics ]; then report; exit; fi
check_no_mps
if [ ! -f "$OUT/probe.ok" ]; then
  VIDEO=$("$PY" -c 'import sys; sys.path.insert(0, "bench"); import capacity; print(capacity.VIDEO)')
  STREAMSPLAT=$(realpath third_party/StreamSplat)
  wait_free
  docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/torch_home \
    -e USER="$USER" -v "$PWD:$PWD:ro" -v "$STREAMSPLAT:$STREAMSPLAT:ro" -v "$HOME/.cache/torch:/torch_home:ro" \
    -v "$(dirname "$VIDEO"):$(dirname "$VIDEO"):ro" streamsplat:5d686e9 \
    python "$PWD/probes/streamsplat_cache_equivalence.py" "$VIDEO" > "$OUT/probe.log" 2>&1
  if ! grep -q CACHE_EQUIVALENCE_OK "$OUT/probe.log"; then
    echo "the cache equivalence probe failed: a bug, not a result; read $OUT/probe.log"
    exit 3
  fi
  touch "$OUT/probe.ok"
fi
for n in 1 4; do
  for rep in 1 2; do
    for arm in nocache cache; do
      flag=()
      [ "$arm" = cache ] && flag=(--cache)
      measure "$OUT/$arm/n$n/r$rep" contaminated --model streamsplat --no-compile --n "$n" --batch "$n" \
        --period 0 --max-updates 20 --warmup 3 "${flag[@]}" || exit 1
      echo "$arm b=$n r$rep $(verdict "$OUT/$arm/n$n/r$rep")"
    done
  done
done
report
echo "G11 done"
