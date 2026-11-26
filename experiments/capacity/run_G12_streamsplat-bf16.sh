#!/usr/bin/env bash
# G12 (LOG.md): C2 of plan `misty-painting-summit` (approved 2026-11-18), built because G10 found bf16 autocast to run
# the model at 0.42x TF32 (rule: <= 0.87x). With each user's previous keyframe cached (G11), what does running the
# StreamSplat model under bf16 autocast do to an update's time and memory, and what does it cost in quality?
# Exploratory lane.
#
# Setup. Quality first: experiments/streamsplat-repro/davis_eval.py eval --bf16 on the 30 DAVIS 2017 val sequences,
# every 6-frame window (1,849), G8's depth maps reused read-only, the authors' metrics psnr_given (the two
# keyframes) and psnr_novel (the middle 4). The control is G8's run: the same code, windows and depth without
# autocast; inference is deterministic, so it is not rerun, and the comparison is paired per window. The heads the
# authors pin to fp32 stay fp32, and model/splat_model.py keeps the renderer and the losses out of autocast.
# Timing as G11: bench/capacity.py sweep --model streamsplat --no-compile, one server, N = batch = b users back to
# back (--period 0 --max-updates 20 --warmup 3), --cache in both arms (G11's decision), b in 1 4; arm `bf16` adds
# --bf16, arm `tf32` is the matched control (the same server, only autocast differs); arms interleaved, 2 repeats
# each. The depth model is never under autocast.
# A dry run (capacity.py --dry-run, batch 2, --cache --bf16, outputs/scratch/g12-dry) checked the code after the
# plan's approval and before this runner; its numbers are not results.
#
# Registered in plan mode before G10 ran (plan approved 2026-11-18) and kept unchanged, although G10's 0.42x on
# forward_gaussians was known when this header was written (estimates):
#   P1 with the cache, bf16 brings the Gaussian part of an update (service minus depth) at b = 1 to 0.55-0.75x
#      the matched control (no separate refutation registered).
#   P2 at b = 1, bf16 with the cache runs 7.0-8.5 updates/s back to back. Refuted if < 1.15x the matched control.
#   P3 val psnr_given and psnr_novel each drop by 0-0.15 dB against G8 (37.76 and 23.76 dB).
#   Quality gate: if either drops by > 0.5 dB, bf16 stays out of the final configuration unless the user, having
#     seen the numbers, accepts it.
#   Memory is reported, not predicted. metrics.json's "capacity" fields mean nothing at --period 0.
#
# Usage: bash experiments/capacity/run_G12_streamsplat-bf16.sh           (resumable)
#        bash experiments/capacity/run_G12_streamsplat-bf16.sh metrics   (re-derive both metrics.json files)
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G12_OUT:-outputs/capacity/G12}
. bench/sweep_lib.sh
STREAMSPLAT=$(realpath third_party/StreamSplat)
DAVIS=$(realpath third_party/data/davis/DAVIS)
FRAMES=$DAVIS/JPEGImages/480p
VAL=$(cat "$DAVIS/ImageSets/2017/val.txt")
G8=$(realpath outputs/streamsplat-repro/G8)
mkdir -p "$OUT/davis"
QUALITY=$(realpath "$OUT/davis")
HELPERS=$PWD/experiments/streamsplat-repro

in_image() {  # gpu flag first ("--gpus all" or ""), then the command; as G8's runner, plus G8's outputs read-only
  local gpus=$1; shift
  docker run --rm $gpus --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/torch_home \
    -e USER="$USER" -e PYTHONPATH="$STREAMSPLAT" -w "$STREAMSPLAT" -v "$PWD:$PWD:ro" \
    -v "$STREAMSPLAT:$STREAMSPLAT:ro" -v "$HOME/.cache/torch:/torch_home:ro" -v "$DAVIS:$DAVIS:ro" \
    -v "$G8:$G8:ro" -v "$QUALITY:$QUALITY" streamsplat:5d686e9 "$@"
}

quality_metrics() {  # davis_eval's metrics, then the paired per-window change against G8, into the same file
  in_image "" python "$HELPERS/davis_eval.py" metrics "$QUALITY/windows.jsonl" "$QUALITY/metrics.json" || exit 1
  "$PY" - "$QUALITY/metrics.json" "$QUALITY/windows.jsonl" "$G8/windows.jsonl" <<'EOF'
import json, sys
path, ours, theirs = sys.argv[1:]
def rows(jsonl):
    return {(row["seq"], row["start"]): row for row in map(json.loads, open(jsonl)) if row["split"] == "val"}
bf16, fp32 = rows(ours), rows(theirs)
assert bf16.keys() == fp32.keys(), "the bf16 run and G8 cover different windows"
metrics = json.load(open(path))
metrics["against_G8"] = {"windows": len(bf16)}
for key in ("psnr_given", "psnr_novel"):
    change = sorted(bf16[window][key] - fp32[window][key] for window in bf16)
    entry = {"G8_mean": sum(fp32[window][key] for window in fp32) / len(fp32),
             "bf16_mean": sum(bf16[window][key] for window in bf16) / len(bf16),
             "change_mean": sum(change) / len(change), "change_min": change[0], "change_max": change[-1],
             "share_dropping_over_0.5_dB": sum(value < -0.5 for value in change) / len(change)}
    metrics["against_G8"][key] = entry
    print(f"val {key}: G8 {entry['G8_mean']:.3f}, bf16 {entry['bf16_mean']:.3f} dB, change {entry['change_mean']:+.3f} "
          f"(per window {entry['change_min']:+.3f} to {entry['change_max']:+.3f}; "
          f"{entry['share_dropping_over_0.5_dB']:.1%} of windows drop > 0.5 dB)")
json.dump(metrics, open(path, "w"), indent=1)
EOF
}

report() {
  metrics tf32 bf16
  "$PY" - "$OUT/metrics.json" <<'EOF'
import json, sys
points = json.load(open(sys.argv[1]))["points"]
def mean(arm, n, key):
    runs = points[arm][str(n)].values()
    return sum(run[key] for run in runs) / len(runs)
def gaussian_part(arm, n):
    return mean(arm, n, "svc_p50_mean") - mean(arm, n, "svc_depth_p50_mean")
for n in (1, 4):
    control, bf16 = mean("tf32", n, "updates_per_s"), mean("bf16", n, "updates_per_s")
    print(f"b = {n}: tf32 {control:.2f}, bf16 {bf16:.2f} updates/s (x{bf16 / control:.3f}); Gaussian part "
          f"{gaussian_part('tf32', n):.4f} -> {gaussian_part('bf16', n):.4f} s (x{gaussian_part('bf16', n) / gaussian_part('tf32', n):.3f}); "
          f"gpu_mem_mib_max {mean('bf16', n, 'gpu_mem_mib_max') - mean('tf32', n, 'gpu_mem_mib_max'):+.0f} MiB, "
          f"peak allocated {(mean('bf16', n, 'peak_alloc_max') - mean('tf32', n, 'peak_alloc_max')) / 2**20:+.0f} MiB with bf16")
EOF
}

if [ "${1:-}" = metrics ]; then quality_metrics; report; exit; fi
check_no_mps
if [ ! -f "$QUALITY/eval.done" ]; then
  wait_free
  rm -f "$QUALITY/windows.jsonl"  # a partial evaluation is redone whole
  in_image "--gpus all" python "$HELPERS/davis_eval.py" eval --bf16 "$FRAMES" "$G8/depth" "$QUALITY/windows.jsonl" val $VAL \
    > "$QUALITY/eval.log" 2>&1 || { echo "the bf16 evaluation failed; read $QUALITY/eval.log"; exit 1; }
  touch "$QUALITY/eval.done"
fi
quality_metrics
for n in 1 4; do
  for rep in 1 2; do
    for arm in tf32 bf16; do
      flag=()
      [ "$arm" = bf16 ] && flag=(--bf16)
      measure "$OUT/$arm/n$n/r$rep" contaminated --model streamsplat --no-compile --n "$n" --batch "$n" \
        --period 0 --max-updates 20 --warmup 3 --cache "${flag[@]}" || exit 1
      echo "$arm b=$n r$rep $(verdict "$OUT/$arm/n$n/r$rep")"
    done
  done
done
report
echo "G12 done"
