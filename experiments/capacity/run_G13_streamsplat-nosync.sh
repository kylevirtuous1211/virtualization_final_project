#!/usr/bin/env bash
# G13 (LOG.md): C3 of plan `misty-painting-summit` (approved 2026-11-18), built because G10 found the GPU busy only
# 86% (TF32) and 78% (bf16) of a batch-1 forward (rule: < 90%), with 14 host waits per forward. With the cache (G11)
# and bf16 (G12), does removing the host waits in the model calls make an update cheaper? It is also C4's
# prerequisite: a CUDA graph cannot be captured across a host wait. Exploratory lane.
#
# Setup. --nosync (bench/capacity.py): the server runs under `python -O`, which drops the static head's isnan
# asserts (each a device-to-host read), and the model calls run with CUDA as the default device, so the tensors the
# model builds with factory functions start on the GPU instead of being copied there from pageable memory. The plan
# says to wrap the step; only the model calls are wrapped, since the depth stage builds nothing on the CPU.
# First bench/streamsplat_sync_count.py counts the CUDA runtime calls in one cached bf16 update at batch 1 with
# torch.profiler, once per arm (plain python, and python -O with --nosync), and checks that the default-device
# context leaves the output bitwise unchanged; if it does not, or the -O arm did not run optimised: a bug, stop
# (exit 3).
# Timing as G11 and G12: one server, N = batch = b users back to back (--period 0 --max-updates 20 --warmup 3),
# compile off, --cache --bf16 in both arms (G11, G12), b in 1 4; arm `nosync` adds --nosync, arm `sync` is the
# matched control; arms interleaved, 2 repeats each.
# A smoke run of the counter (outputs/scratch/g13-smoke) and two dry runs (outputs/scratch/g13-dry, g13-dry-sync)
# checked the code after the plan's approval and before this runner; their numbers are not results.
#
# Registered in plan mode (plan approved 2026-11-18) and kept unchanged (estimates):
#   P1 the host waits (cudaStreamSynchronize) in an update's model calls drop to 0 (G10 counted 14 in a full
#      forward). Refuted if > 2 remain.
#   P2 back-to-back throughput rises 1-5% over the matched control. Below 1% C3 stays anyway, as C4's
#      prerequisite, and the entry says so.
#   metrics.json's "capacity" fields mean nothing at --period 0.
#
# Usage: bash experiments/capacity/run_G13_streamsplat-nosync.sh           (resumable)
#        bash experiments/capacity/run_G13_streamsplat-nosync.sh metrics   (re-derive metrics.json)
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G13_OUT:-outputs/capacity/G13}
. bench/sweep_lib.sh
VIDEO=$("$PY" -c 'import sys; sys.path.insert(0, "bench"); import capacity; print(capacity.VIDEO)')
STREAMSPLAT=$(realpath third_party/StreamSplat)
mkdir -p "$OUT/syncs"
SYNCS=$(realpath "$OUT/syncs")

count_syncs() {  # count_syncs <arm>
  local optimise=() flag=()
  [ "$1" = nosync ] && optimise=(-O) && flag=(--nosync)
  docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/torch_home \
    -e USER="$USER" -v "$PWD:$PWD:ro" -v "$STREAMSPLAT:$STREAMSPLAT:ro" -v "$HOME/.cache/torch:/torch_home:ro" \
    -v "$(dirname "$VIDEO"):$(dirname "$VIDEO"):ro" -v "$SYNCS:$SYNCS" streamsplat:5d686e9 \
    python "${optimise[@]}" "$PWD/bench/streamsplat_sync_count.py" "$VIDEO" "$SYNCS/$1.json" "${flag[@]}" \
    > "$SYNCS/$1.log" 2>&1
}

report() {
  metrics sync nosync
  "$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
metrics = json.loads((out / "metrics.json").read_text())
points = metrics["points"]
def mean(arm, n, key):
    runs = points[arm][str(n)].values()
    return sum(run[key] for run in runs) / len(runs)
def optimised(arm, n):  # the server's own record of python -O, from every kept run's ready lines
    return [json.loads(line)["python_optimize"] for run in sorted((out / arm / f"n{n}").glob("r*"))
            for line in (run / "user0.jsonl").read_text().splitlines() if '"ready"' in line]
metrics["syncs"] = {arm: json.loads((out / "syncs" / f"{arm}.json").read_text()) for arm in ("sync", "nosync")}
metrics["server_python_optimize"] = {arm: {n: optimised(arm, n) for n in (1, 4)} for arm in ("sync", "nosync")}
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
for arm, counted in metrics["syncs"].items():
    print(f"{arm}: {counted['runtime_calls']} in one update; output unchanged by the device context: "
          f"{counted['device_context_bitwise_equal']}")
print("server under python -O:", metrics["server_python_optimize"])
for n in (1, 4):
    control, nosync = mean("sync", n, "updates_per_s"), mean("nosync", n, "updates_per_s")
    print(f"b = {n}: sync {control:.2f}, nosync {nosync:.2f} updates/s (x{nosync / control:.3f}); service "
          f"{mean('sync', n, 'svc_p50_mean'):.4f} -> {mean('nosync', n, 'svc_p50_mean'):.4f} s, depth "
          f"{mean('sync', n, 'svc_depth_p50_mean'):.4f} -> {mean('nosync', n, 'svc_depth_p50_mean'):.4f} s")
EOF
}

if [ "${1:-}" = metrics ]; then report; exit; fi
check_no_mps
if [ ! -f "$SYNCS/counted.ok" ]; then
  wait_free
  count_syncs sync && count_syncs nosync || { echo "the sync counter failed; read $SYNCS/*.log"; exit 1; }
  "$PY" - "$SYNCS" <<'EOF' || exit 3
import json, sys
from pathlib import Path
counted = {arm: json.loads((Path(sys.argv[1]) / f"{arm}.json").read_text()) for arm in ("sync", "nosync")}
if not all(entry["device_context_bitwise_equal"] for entry in counted.values()):
    sys.exit("the default-device context changed the output: a bug, not a result")
if not counted["nosync"]["python_optimize"] or counted["sync"]["python_optimize"]:
    sys.exit("the arms did not run with the intended python -O setting: a bug")
EOF
  touch "$SYNCS/counted.ok"
fi
for n in 1 4; do
  for rep in 1 2; do
    for arm in sync nosync; do
      flag=()
      [ "$arm" = nosync ] && flag=(--nosync)
      measure "$OUT/$arm/n$n/r$rep" contaminated --model streamsplat --no-compile --n "$n" --batch "$n" \
        --period 0 --max-updates 20 --warmup 3 --cache --bf16 "${flag[@]}" || exit 1
      echo "$arm b=$n r$rep $(verdict "$OUT/$arm/n$n/r$rep")"
    done
  done
done
report
echo "G13 done"
