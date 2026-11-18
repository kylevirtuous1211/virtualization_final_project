#!/usr/bin/env bash
# G14 (LOG.md): C4 of plan `misty-painting-summit` (approved 2026-11-18), built because G10 found the GPU busy only
# 86% (TF32) and 78% (bf16) of a batch-1 forward (rule: < 90%) and G13 brought the host waits in the model calls to
# zero (rule). With the cache (G11), bf16 (G12) and no host waits (G13), does replaying each update's depth and
# model stages as CUDA graphs make it cheaper, and what VRAM does that hold? Exploratory lane.
#
# Setup. --graphs (bench/capacity.py, bench/streamsplat_stages.py Graphed): per batch size, the depth stage (Depth
# Anything plus min-max) and the model stage (encode_frames plus pair_stage) are each recorded as a CUDA graph on
# first use, after 3 warm-up calls, all in one memory pool, and replayed after that; the frames are uploaded and
# the cached features gathered outside the graphs. Two deviations from the plan, both in place before any graph
# timing: (1) graphs are recorded for every batch size the server forms, not for a fixed list with padding, which
# would spend GPU work on empty slots; (2) graph capture failed on StreamSplat's training-time block checkpointing
# (torch.utils.checkpoint reads the CUDA RNG state even under no_grad), so both arms run --no-checkpointing, a
# prerequisite like G13's -O; its effect alone is not measured by this entry. Autocast keeps no cast cache under
# --graphs, as capture requires.
# First probes/streamsplat_graph_equivalence.py must pass (python -O, 4 users, 3 updates: checkpointing off bitwise
# equal to G13's server, and the graphs within G9's criterion of the eager control); else a bug: stop (exit 3).
# Timing as G11-G13: one server, N = batch = b users back to back (--period 0 --max-updates 20 --warmup 3), compile
# off, --cache --bf16 --nosync --no-checkpointing in both arms, b in 1 4; arm `graphs` adds --graphs, arm `eager`
# is the matched control; arms interleaved, 2 repeats each.
# The probe and a dry run (capacity.py --dry-run, batch 2, outputs/scratch/g14-dry) ran after the plan's approval
# and before this runner; the dry run's numbers are not results.
#
# Registered in plan mode (plan approved 2026-11-18) and kept unchanged (estimates):
#   P1 back-to-back throughput rises about by G10's idle share, 1.08-1.20x the matched control. Refuted if < 1.05x.
#   P2 the graphs hold about 2 GB more VRAM in bf16 (4 GB in TF32). If > 8 GB, only batch 1 would be recorded.
#   metrics.json's "capacity" fields mean nothing at --period 0.
#
# Usage: bash experiments/capacity/run_G14_streamsplat-graphs.sh           (resumable)
#        bash experiments/capacity/run_G14_streamsplat-graphs.sh metrics   (re-derive metrics.json)
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G14_OUT:-outputs/capacity/G14}
. bench/sweep_lib.sh
report() {
  metrics eager graphs
  "$PY" - "$OUT/metrics.json" <<'EOF'
import json, sys
points = json.load(open(sys.argv[1]))["points"]
def mean(arm, n, key):
    runs = points[arm][str(n)].values()
    return sum(run[key] for run in runs) / len(runs)
for n in (1, 4):
    control, graphs = mean("eager", n, "updates_per_s"), mean("graphs", n, "updates_per_s")
    print(f"b = {n}: eager {control:.2f}, graphs {graphs:.2f} updates/s (x{graphs / control:.3f}); service "
          f"{mean('eager', n, 'svc_p50_mean'):.4f} -> {mean('graphs', n, 'svc_p50_mean'):.4f} s, depth "
          f"{mean('eager', n, 'svc_depth_p50_mean'):.4f} -> {mean('graphs', n, 'svc_depth_p50_mean'):.4f} s; "
          f"gpu_mem_mib_max {mean('graphs', n, 'gpu_mem_mib_max') - mean('eager', n, 'gpu_mem_mib_max'):+.0f} MiB")
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
    python -O "$PWD/probes/streamsplat_graph_equivalence.py" "$VIDEO" > "$OUT/probe.log" 2>&1
  if ! grep -q GRAPH_EQUIVALENCE_OK "$OUT/probe.log"; then
    echo "the graph equivalence probe failed: a bug, not a result; read $OUT/probe.log"
    exit 3
  fi
  touch "$OUT/probe.ok"
fi
for n in 1 4; do
  for rep in 1 2; do
    for arm in eager graphs; do
      flag=()
      [ "$arm" = graphs ] && flag=(--graphs)
      measure "$OUT/$arm/n$n/r$rep" contaminated --model streamsplat --no-compile --n "$n" --batch "$n" \
        --period 0 --max-updates 20 --warmup 3 --cache --bf16 --nosync --no-checkpointing "${flag[@]}" || exit 1
      echo "$arm b=$n r$rep $(verdict "$OUT/$arm/n$n/r$rep")"
    done
  done
done
report
echo "G14 done"
