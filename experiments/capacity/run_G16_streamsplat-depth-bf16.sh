#!/usr/bin/env bash
# G16 (LOG.md): plan `misty-painting-summit`, G16 version (approved 2026-11-19). G15's fast server spends 47.6 ms of
# an 87 ms update in the depth model (Depth Anything V2 ViT-L, TF32, untouched by G11-G14). Does running the depth
# model under bf16 autocast let the RTX 6000 Ada serve more users at one keyframe per user per second, and what
# does it cost in quality? Exploratory lane.
#
# Setup. --depth-bf16 (bench/capacity.py, 1c39de2): disparity_of runs the depth model alone under bf16 autocast
# with no cast cache (so the depth graph can hold it), then casts to fp32 for the interpolation and min-max; the
# priming and every update use it. Two arms on the same server code, both batch 1: fast = G15's (--batch 1 --cache
# --bf16 --nosync --no-checkpointing --graphs), the matched control; depthbf16 = fast plus --depth-bf16.
# Bug checks (exit 3): probes/streamsplat_graph_equivalence.py --depth-bf16 under python -O; then
# bench/streamsplat_depth_check.py on all 64 clip frames (the depth model's output is bf16 with the flag and the
# disparity differs: a positive control; every frame still spans 0..255; it also gives P3's server-side numbers).
# Every timed run must record depth_bf16 in its server's ready lines as its arm says, else a bug (exit 3).
# Step 0: N = batch = 1 back to back (--period 0 --max-updates 20 --warmup 3), compile off, arms interleaved, 2
# repeats, runs step0/<arm>_r<rep>. Gate: mean depthbf16 / mean fast < 1.10 (capacity could rise by at most 1): no
# quality check, no sweep, a negative result. A missing step-0 run is a bug (exit 1), not a negative result.
# Step 1, quality (davis_eval.py on the 30 DAVIS 2017 val sequences, the authors' psnr_given and psnr_novel):
#   reproducibility first: G12's command (G8's depth maps, eval --bf16) rerun on the 2 shortest val sequences,
#   compared window by window with G12's rows (if they differ, the rerun's spread is the noise floor, and per-window
#   extremes and tail shares are not read below it); and fp32 depth maps for the shortest sequence from the changed
#   davis_bear.py, which must equal G8's PNGs byte for byte (else exit 3).
#   Then bf16 depth maps for all val frames (count must equal frames), compared with G8's (100% equal is a bug);
#   eval --bf16 on all 1,849 val windows with them; paired per window against G12 (the same model setting, only the
#   depth maps differ) and against G8's val windows (full precision; G8 also scored train's bear, left out).
#   Differences from the server, all making the check stricter or merely different: G8's DAVIS depth ran in full
#   fp32 (davis_eval.py depth sets no TF32), the server's in TF32; DAVIS frames give 924x518 depth inputs (2,442
#   tokens) against the clip's 686x518 (1,813); the evaluation runs SplatModel.forward, not the cached server.
#   Quality gate: the cumulative mean drop against G8 > 0.5 dB in psnr_given or psnr_novel: depth bf16 stays out of
#   the final configuration unless the user, having seen the numbers, accepts it; the runner stops before the sweep.
#   G12's model bf16 already took 0.18 dB of psnr_given, so this step has 0.32 dB of it left.
# Step 2: G6's and G15's 1 Hz protocol (period 1 s, phases evenly staggered, 60 s, 3 warm-ups, compile off; SLO p95
# <= 1.0 s, 60/60 updates, no OOM; capacity = the largest N where both repeats pass), arms fast and depthbf16
# interleaved per N, 2 repeats, grid N in 10..18 20 22 24 (from 10: G15's fast passed every N from 1 to 10 at p95 <=
# 0.096 s); an arm stops after two consecutive N where neither repeat passes. Capacity is reported as a range where
# the grid has a gap, as a lower bound (>= 24) at the top, and as "< 10" if no N passes. Selection is by that rule,
# not by quality.
# Known when this header was written: G15's fast at 11.53 updates/s back to back, capacity 10, depth 47.6 ms at
# step 0; G14's depth stage 47.8 ms; G10's kernel rates; the reviewers' FLOP count (1.81 TFLOP per depth call). After
# the plan's approval and before this runner: a dry run (outputs/scratch/g16-dry: --n 2 with every flag), a smoke run
# of the depth check (outputs/scratch/g16-depth-smoke: mean 0.29 levels, max 8, 71% equal pixels, every check
# passed) and of compare-depth on G8's maps against themselves (CPU); their numbers are not results.
# Frozen while this runs: bench/capacity.py, bench/streamsplat_stages.py, bench/streamsplat_depth_check.py,
# bench/sweep_lib.sh, experiments/streamsplat-repro/davis_eval.py and davis_bear.py, the graph probe; any code change
# after this header's commit is a deviation, written into the entry's Setup.
#
# Registered in plan mode (approved 2026-11-19) (estimates):
#   P1 depth stage (svc_depth_p50, upload included), step 0, mean of 2: depthbf16 0.43-0.60x fast. Refuted if > 0.80x.
#   P2 step-0 throughput: depthbf16 1.30-1.45x fast (about 15-16.7 updates/s). Refuted if < 1.10x (the gate).
#   P3 depth differences, reported only: the server's 64 frames mean abs <= 1 level; DAVIS maps against G8 <= 3.
#   P4 quality against G12: psnr_given drops 0-0.10 dB, psnr_novel 0-0.05 dB, <= 2% of windows drop > 0.5 dB.
#      Refuted if either mean drops > 0.20 dB or > 5% of windows drop > 0.5 dB. Cumulative against G8: psnr_given
#      drops 0.18-0.28 dB (gate above).
#   P5 capacity: fast 10 or 11 (G15 reproduced); depthbf16 15 (14-16). Refuted if depthbf16's <= fast's.
#   P6 capacity tracks throughput: each arm's capacity is the largest grid N below its step-0 throughput. Refuted if
#      either arm's is more than one grid step lower.
#   P7 fast reproduces G15: step 0 within +-5% of 11.53. If not, said so; the interleaved comparison still holds.
#   P8 p95 of the worst user at N = 10: depthbf16 below 0.85x fast. Refuted if not below 0.9x.
#   P9 whole-GPU memory at equal N changes within +-0.5 GB. Refuted if it rises > 1.5 GB.
#   P10 GPU utilisation at the largest N each arm ran >= 85%. Refuted if < 78%.
#
# Usage: bash experiments/capacity/run_G16_streamsplat-depth-bf16.sh           (resumable)
#        bash experiments/capacity/run_G16_streamsplat-depth-bf16.sh metrics   (re-derive the metrics files)
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G16_OUT:-outputs/capacity/G16}
. bench/sweep_lib.sh
FAST_FLAGS="--cache --bf16 --nosync --no-checkpointing --graphs"
COMMON=(--model streamsplat --no-compile)
GRID="10 11 12 13 14 15 16 17 18 20 22 24"
VIDEO=$("$PY" -c 'import sys; sys.path.insert(0, "bench"); import capacity; print(capacity.VIDEO)')
STREAMSPLAT=$(realpath third_party/StreamSplat)
DAVIS=$(realpath third_party/data/davis/DAVIS)
FRAMES=$DAVIS/JPEGImages/480p
VAL=$(cat "$DAVIS/ImageSets/2017/val.txt")
SHORT=$(for sequence in $VAL; do echo "$(ls "$FRAMES/$sequence" | wc -l) $sequence"; done | sort -n | head -2 | cut -d' ' -f2)
G8=$(realpath outputs/streamsplat-repro/G8)
G12=$(realpath outputs/capacity/G12/davis)
mkdir -p "$OUT/checks" "$OUT/davis"
CHECKS=$(realpath "$OUT/checks")
QUALITY=$(realpath "$OUT/davis")
HELPERS=$PWD/experiments/streamsplat-repro

in_image() {  # gpu flag first ("--gpus all" or ""), then the command; G12's mounts, G16's outputs writable
  local gpus=$1; shift
  docker run --rm $gpus --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/torch_home \
    -e USER="$USER" -e PYTHONPATH="$STREAMSPLAT" -w "$STREAMSPLAT" -v "$PWD:$PWD:ro" \
    -v "$STREAMSPLAT:$STREAMSPLAT:ro" -v "$HOME/.cache/torch:/torch_home:ro" -v "$DAVIS:$DAVIS:ro" \
    -v "$G8:$G8:ro" -v "$(dirname "$VIDEO"):$(dirname "$VIDEO"):ro" -v "$CHECKS:$CHECKS" -v "$QUALITY:$QUALITY" \
    streamsplat:5d686e9 "$@"
}

check_runs() {  # check_runs <arm> <run-dir>...: each server's ready lines record the arm's depth_bf16
  "$PY" - "$@" <<'EOF' || exit 3
import json, sys
from pathlib import Path
arm, runs = sys.argv[1], sys.argv[2:]
for run in map(Path, runs):
    seen = {json.loads(line)["depth_bf16"] for log in run.glob("user*.jsonl")
            for line in log.read_text().splitlines() if '"ready"' in line}
    if seen != {arm == "depthbf16"}:
        sys.exit(f"{run}: the server recorded depth_bf16 {seen} in arm {arm}: a bug")
EOF
}

quality_metrics() {  # davis_eval's metrics, the depth-map comparison, then the paired changes, into one file
  in_image "" python "$HELPERS/davis_eval.py" metrics "$QUALITY/windows.jsonl" "$QUALITY/metrics.json" || exit 1
  "$PY" - "$QUALITY" "$G12/windows.jsonl" "$G8/windows.jsonl" <<'EOF'
import json, sys
from pathlib import Path
quality, g12, g8 = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def rows(jsonl):
    return {(row["seq"], row["start"]): row for row in map(json.loads, open(jsonl)) if row["split"] == "val"}
ours, controls = rows(quality / "windows.jsonl"), {"G12": rows(g12), "G8": rows(g8)}
metrics = json.loads((quality / "metrics.json").read_text())
metrics["reproducibility"] = json.loads((quality / "repro.json").read_text())
compared = json.loads((quality / "depth_against_G8.json").read_text())
metrics["depth_maps_against_G8"] = {key: value for key, value in compared.items() if key != "per_map"}
for name, control in controls.items():
    assert ours.keys() == control.keys(), f"the windows differ from {name}'s: a bug"
    entry = {"windows": len(ours)}
    for key in ("psnr_given", "psnr_novel"):
        change = sorted(ours[window][key] - control[window][key] for window in ours)
        entry[key] = {f"{name}_mean": sum(row[key] for row in control.values()) / len(control),
                      "ours_mean": sum(row[key] for row in ours.values()) / len(ours),
                      "change_mean": sum(change) / len(change), "change_min": change[0], "change_max": change[-1],
                      "share_dropping_over_0.5_dB": sum(value < -0.5 for value in change) / len(change)}
        print(f"against {name}, val {key}: {entry[key][f'{name}_mean']:.3f} -> {entry[key]['ours_mean']:.3f} dB, "
              f"change {entry[key]['change_mean']:+.3f} (per window {change[0]:+.3f} to {change[-1]:+.3f}; "
              f"{entry[key]['share_dropping_over_0.5_dB']:.1%} drop > 0.5 dB)")
    metrics[f"against_{name}"] = entry
metrics["quality_gate_passed"] = all(metrics["against_G8"][key]["change_mean"] >= -0.5 for key in ("psnr_given", "psnr_novel"))
print("depth maps against G8:", metrics["depth_maps_against_G8"])
print("reproducibility:", metrics["reproducibility"])
print("quality gate (cumulative drop against G8 <= 0.5 dB) passed:", metrics["quality_gate_passed"])
(quality / "metrics.json").write_text(json.dumps(metrics, indent=1))
EOF
}

report() {
  metrics fast depthbf16 > /dev/null
  "$PY" - "$OUT" "$GRID" <<'EOF'
import json, sys
from pathlib import Path
out, grid = Path(sys.argv[1]), [int(value) for value in sys.argv[2].split()]
metrics = json.loads((out / "metrics.json").read_text())
step0, points, arms = metrics["step0_updates_per_s"], metrics["points"], ("fast", "depthbf16")
def depth_p50(arm, rep):
    return json.loads((out / "step0" / f"{arm}_r{rep}" / "summary.json").read_text())["users"][0]["svc_depth_p50"]
derived = {"step0": {arm: {"updates_per_s": [step0[f"{arm}_r{rep}"] for rep in (1, 2)],
                           "svc_depth_p50": [depth_p50(arm, rep) for rep in (1, 2)]} for arm in arms}}
mean = lambda values: sum(values) / len(values)
for arm in arms:
    derived["step0"][arm]["mean_updates_per_s"] = mean(derived["step0"][arm]["updates_per_s"])
    derived["step0"][arm]["mean_svc_depth_p50"] = mean(derived["step0"][arm]["svc_depth_p50"])
derived["throughput_ratio"] = derived["step0"]["depthbf16"]["mean_updates_per_s"] / derived["step0"]["fast"]["mean_updates_per_s"]
derived["depth_ratio"] = derived["step0"]["depthbf16"]["mean_svc_depth_p50"] / derived["step0"]["fast"]["mean_svc_depth_p50"]
derived["fast_against_G15_step0"] = derived["step0"]["fast"]["mean_updates_per_s"] / 11.53
print("step 0:", json.dumps(derived["step0"]))
print(f"throughput depthbf16/fast x{derived['throughput_ratio']:.3f}; depth stage x{derived['depth_ratio']:.3f}; "
      f"fast against G15's 11.53: x{derived['fast_against_G15_step0']:.3f}")
if points.get("fast") or points.get("depthbf16"):
    derived["capacity"] = {}
    for arm in arms:
        both = metrics["capacity"][arm]["both_repeats"]
        higher = [n for n in grid if n > both]
        text = ("< %d" % grid[0] if not both else ">= %d" % both if not higher else
                str(both) if higher[0] == both + 1 else f"{both} to {higher[0] - 1}")
        expected = max([n for n in grid if n < derived["step0"][arm]["mean_updates_per_s"]], default=0)
        ran = sorted(map(int, points[arm]))
        utilisation = mean([run["gpu_util_mean"] for run in points[arm][str(ran[-1])].values()])
        derived["capacity"][arm] = {"both_repeats": both, "reported": text, "expected_from_step0": expected,
                                    "largest_n_run": ran[-1], "gpu_util_at_largest_n": utilisation}
        print(f"{arm}: capacity {text} (from step 0 expected {expected}); largest N run {ran[-1]}, "
              f"GPU utilisation there {utilisation:.0f}%")
    shared = sorted(set(map(int, points["fast"])) & set(map(int, points["depthbf16"])))
    at = lambda arm, n, key: mean([run[key] for run in points[arm][str(n)].values()])
    if 10 in shared:
        derived["p95_at_10"] = {arm: at(arm, 10, "lat_p95_max") for arm in arms}
        derived["p95_at_10"]["ratio"] = derived["p95_at_10"]["depthbf16"] / derived["p95_at_10"]["fast"]
        print("p95 of the worst user at N = 10 (s):", derived["p95_at_10"])
    derived["gpu_mem_change_mib"] = {n: at("depthbf16", n, "gpu_mem_mib_max") - at("fast", n, "gpu_mem_mib_max") for n in shared}
    print("whole-GPU memory, depthbf16 minus fast (MiB):", derived["gpu_mem_change_mib"])
metrics["derived"] = derived
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
EOF
}

if [ "${1:-}" = metrics ]; then
  [ -f "$QUALITY/eval.done" ] && quality_metrics
  report
  exit
fi
check_no_mps

# bug checks
if [ ! -f "$CHECKS/checks.ok" ]; then
  wait_free
  in_image "--gpus all" python -O "$PWD/probes/streamsplat_graph_equivalence.py" "$VIDEO" --depth-bf16 \
    > "$CHECKS/probe.log" 2>&1
  grep -q GRAPH_EQUIVALENCE_OK "$CHECKS/probe.log" || { echo "the graph probe failed: a bug; read $CHECKS/probe.log"; exit 3; }
  in_image "--gpus all" python "$PWD/bench/streamsplat_depth_check.py" "$VIDEO" "$CHECKS/depth_check.json" \
    > "$CHECKS/depth_check.log" 2>&1
  grep -q DEPTH_CHECK_OK "$CHECKS/depth_check.log" || { echo "the depth check failed: a bug; read $CHECKS/depth_check.log"; exit 3; }
  touch "$CHECKS/checks.ok"
fi
echo "bug checks passed: $(head -c 300 "$CHECKS/depth_check.json" | tr -d '\n')"

# step 0
for rep in 1 2; do
  for arm in fast depthbf16; do
    flag=(--batch 1 $FAST_FLAGS)
    [ "$arm" = depthbf16 ] && flag+=(--depth-bf16)
    measure "$OUT/step0/${arm}_r$rep" contaminated "${COMMON[@]}" --n 1 --period 0 --max-updates 20 --warmup 3 \
      "${flag[@]}" || exit 1
    check_runs "$arm" "$OUT/step0/${arm}_r$rep"
  done
done
report
gate=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["derived"]["throughput_ratio"] >= 1.10)' \
  "$OUT/metrics.json") || { echo "step 0 is incomplete: a bug"; exit 1; }
if [ "$gate" != True ]; then
  echo "gate: depthbf16 runs below 1.10x fast back to back; no quality check and no sweep, as registered"
  exit 0
fi

# step 1, quality
if [ ! -f "$QUALITY/repro.ok" ]; then
  wait_free
  rm -rf "$QUALITY/repro_windows.jsonl" "$QUALITY/fp32_depth"
  in_image "--gpus all" python "$HELPERS/davis_eval.py" eval --bf16 "$FRAMES" "$G8/depth" "$QUALITY/repro_windows.jsonl" \
    val $SHORT > "$QUALITY/repro_eval.log" 2>&1 || { echo "the rerun of G12 failed; read $QUALITY/repro_eval.log"; exit 1; }
  shortest=$(echo "$SHORT" | head -1)
  in_image "--gpus all" python "$HELPERS/davis_eval.py" depth "$FRAMES" "$QUALITY/fp32_depth" "$shortest" \
    > "$QUALITY/fp32_depth.log" 2>&1 || { echo "the fp32 depth rerun failed; read $QUALITY/fp32_depth.log"; exit 1; }
  for map in "$G8/depth/$shortest"/*_pred.png; do
    cmp -s "$map" "$QUALITY/fp32_depth/$shortest/$(basename "$map")" || { echo "fp32 depth differs from G8's at $map: a bug"; exit 3; }
  done
  "$PY" - "$QUALITY/repro_windows.jsonl" "$G12/windows.jsonl" "$QUALITY/repro.json" "$shortest" <<'EOF' || exit 1
import json, sys
rerun, g12, out, shortest = sys.argv[1:]
ours = {(row["seq"], row["start"]): row for row in map(json.loads, open(rerun))}
theirs = {(row["seq"], row["start"]): row for row in map(json.loads, open(g12))}
gaps = [abs(ours[window][key] - theirs[window][key]) for window in ours for key in ("psnr_given", "psnr_novel")]
result = {"sequences": sorted({sequence for sequence, _ in ours}), "windows": len(ours),
          "bitwise_equal_windows": sum(all(ours[window][key] == theirs[window][key] for key in ("psnr_given", "psnr_novel"))
                                       for window in ours),
          "max_abs_change_dB": max(gaps), "fp32_depth_maps_equal_to_G8": shortest}
json.dump(result, open(out, "w"), indent=1)
print("reproducibility:", result)
EOF
  touch "$QUALITY/repro.ok"
fi
if [ ! -f "$QUALITY/depth.done" ]; then
  wait_free
  in_image "--gpus all" python "$HELPERS/davis_eval.py" depth --bf16 "$FRAMES" "$QUALITY/depth" $VAL \
    > "$QUALITY/depth.log" 2>&1 || { echo "the bf16 depth maps failed; read $QUALITY/depth.log"; exit 1; }
  for sequence in $VAL; do
    [ "$(ls "$FRAMES/$sequence" | wc -l)" = "$(ls "$QUALITY/depth/$sequence"/*_pred.png | wc -l)" ] \
      || { echo "depth maps missing for $sequence"; exit 1; }
  done
  in_image "" python "$HELPERS/davis_eval.py" compare-depth "$QUALITY/depth" "$G8/depth" "$QUALITY/depth_against_G8.json" \
    $VAL > "$QUALITY/compare_depth.log" 2>&1 || { echo "the depth comparison failed"; exit 1; }
  "$PY" -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["equal_share_mean"] == 1)' \
    "$QUALITY/depth_against_G8.json" || { echo "bf16 depth maps equal G8's: the flag did nothing, a bug"; exit 3; }
  touch "$QUALITY/depth.done"
fi
if [ ! -f "$QUALITY/eval.done" ]; then
  wait_free
  rm -f "$QUALITY/windows.jsonl"  # a partial evaluation is redone whole
  in_image "--gpus all" python "$HELPERS/davis_eval.py" eval --bf16 "$FRAMES" "$QUALITY/depth" "$QUALITY/windows.jsonl" \
    val $VAL > "$QUALITY/eval.log" 2>&1 || { echo "the evaluation failed; read $QUALITY/eval.log"; exit 1; }
  touch "$QUALITY/eval.done"
fi
quality_metrics
passed=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["quality_gate_passed"])' "$QUALITY/metrics.json")
if [ "$passed" != True ]; then
  echo "quality gate: the cumulative drop against G8 exceeds 0.5 dB; no sweep, as registered (the user decides)"
  exit 0
fi

# step 2, the 1 Hz sweep
ARMS="fast depthbf16" sweep_arms "$GRID" "${COMMON[@]}" --period 1 --slo 1 --duration 60 --warmup 3
for arm in fast depthbf16; do
  check_runs "$arm" $(find "$OUT/$arm" -mindepth 2 -maxdepth 2 -type d -name 'r*')
done
report
echo "G16 done"
