#!/usr/bin/env bash
# G10 (LOG.md): where does one StreamSplat update spend its GPU time on the RTX 6000 Ada, and which of the planned
# components is worth building: C1 a per-user cache of the previous keyframe's one-frame work (the paper's
# Algorithm 2), C2 bf16 autocast, C3 no host syncs, C4 CUDA graphs, C5 stage-level batching across users, C6 two
# CUDA streams with priority? Step 0 of plan `misty-painting-summit`, approved 2026-11-18 (drafted by two Fable
# 5.1 Plan agents). Exploratory lane. Measurement only: serving is unchanged.
#
# Setup: bench/streamsplat_profile.py in the StreamSplat image (streamsplat:5d686e9, compile off), one container,
# pairs from the clip as the batching server builds them. Sections: stages (forward_gaussians end to end; the same
# pairs through bench/streamsplat_stages.py split_forward, per stage with CUDA events; the one-frame stage on the
# B new frames; the pair stage; the depth stage; peak memory per stage; TF32 and bf16 autocast at batch 1, 2, 4, 8,
# plus FP32 "highest" at batch 1), profile (torch.profiler: kernels, attention kernels, host syncs, busy share =
# kernel time over unprofiled wall time), ceiling (matmul and scaled_dot_product_attention with each backend
# forced), overlap (depth stage and pair stage on two streams from two threads). 3 warm-ups, 20 timed calls.
# A co-tenant on the GPU (any compute process other than ours, sampled every 0.5 s) voids the run, which is redone.
# A smoke run (G10_SMOKE=1, 2 calls per point, outputs/scratch/G10-smoke) checked the code after the plan's
# approval; its numbers are not results.
#
# Registered in plan mode before any timing (all estimates):
#   Gate: split_forward's output is bitwise identical to forward_gaussians at the same shapes, and the per-stage
#     times sum to within 10% of forward_gaussians, at every batch and precision; else the split is wrong: a
#     harness bug, stop (exit 3).
#   P1 at batch 1, TF32, the one-frame stage of the frame a cache would keep takes 20-30% of forward_gaussians.
#      Refuted if < 12% or > 40%.
#   P2 bf16 autocast brings forward_gaussians at batch 1 to 0.55-0.75x TF32. Refuted if > 0.87x.
#   P3 GPU busy share at batch 1 is 75-90%; one forward makes >= 14 host waits (synchronise calls plus pageable
#      host-to-device copies: 8 isnan asserts and 6 tensors built on the CPU). Refuted if the busy share > 95%.
#   P4 in FP32, scaled_dot_product_attention runs the memory-efficient kernel, in bf16 the flash kernel; attention
#      takes 40-55% of kernel time in TF32. Refuted if < 30%.
#   P5 matmul ceiling at 8192^3: TF32 70-95, bf16 140-190 TFLOP/s; batch 1 achieves 28-38% of the TF32 ceiling
#      (4.11 TFLOP per pair, analytical). Refuted if the TF32 ceiling > 120 TFLOP/s.
#   P6 at batch 4, per-sample time is <= 0.6x batch 1 for DINO and the two heads, and 0.9-1.15x for the encoder,
#      decoder, both upsamplers and the depth stage; stage batching by the C5 rule saves 5-15 ms per update.
#   P7 two-stream overlap gain 5-15%.
#   P8 activation peak grows 650-750 MiB per extra sample, peaking in an upsampler or a head (stage names, the two
#      upsampler levels are not separated); bf16 330-400 MiB.
#   Decisions: C1 if P1 >= 12%; C2 if bf16 <= 0.87x TF32; C3 and C4 if the busy share < 90%; C5 if the C5 rule (per
#     stage, the largest b in 1 2 4 8 with per-sample time <= 0.9x batch 1) saves >= 10 ms per update at batch 4;
#     C6 if the overlap gain with the depth stream at high priority >= 5%. Stop the whole line (a negative result)
#     if P1 < 12% and bf16 > 0.87x and the busy share > 95%.
#
# Usage: bash experiments/capacity/run_G10_streamsplat-stage-profile.sh           (redoes a contaminated run)
#        bash experiments/capacity/run_G10_streamsplat-stage-profile.sh metrics   (re-derive metrics.json)
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G10_OUT:-outputs/capacity/G10}
. bench/sweep_lib.sh
RUN=$OUT/run
if [ "${1:-}" = metrics ]; then "$PY" bench/streamsplat_profile.py metrics "$RUN"; exit; fi
check_no_mps
VIDEO=${G10_VIDEO:-$("$PY" -c 'import sys; sys.path.insert(0, "bench"); import capacity; print(capacity.VIDEO)')}
STREAMSPLAT=$(realpath third_party/StreamSplat)
for attempt in 1 2 3 4 5; do
  [ -f "$OUT/clean" ] && break
  [ -d "$RUN" ] && mv "$RUN" "$OUT/_aborted/run_$(date +%s)"
  wait_free
  mkdir -p "$RUN"
  run_abs=$(realpath "$RUN")
  while true; do echo "$(date +%s.%N) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"; sleep 0.5; done > "$RUN/smi.log" &
  sampler=$!
  docker run --rm --name "capacity-g10-$$" --gpus all --network none --pid=host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e TORCH_HOME=/torch_home -e USER="$USER" -v "$PWD:$PWD:ro" -v "$STREAMSPLAT:$STREAMSPLAT:ro" \
    -v "$HOME/.cache/torch:/torch_home:ro" -v "$(dirname "$VIDEO"):$(dirname "$VIDEO"):ro" -v "$run_abs:$run_abs" \
    streamsplat:5d686e9 python "$PWD/bench/streamsplat_profile.py" run "$VIDEO" "$run_abs" > "$RUN/run.log" 2>&1
  status=$?
  kill "$sampler"
  if [ "$status" != 0 ]; then echo "attempt $attempt: the profiler failed (exit $status); read $RUN/run.log"; exit 1; fi
  ours=$(cat "$RUN/pid")
  foreign=$(awk '{for (i = 2; i <= NF; i++) print $i}' "$RUN/smi.log" | tr -d ',' | sort -u | grep -v -x "$ours")
  if [ -n "$foreign" ]; then
    echo "attempt $attempt: co-tenant GPU processes $(echo $foreign); redoing"
    mv "$RUN" "$OUT/_redo/run_contaminated_$(date +%s)"
    continue
  fi
  touch "$OUT/clean"
done
[ -f "$OUT/clean" ] || { echo "gave up after 5 attempts"; exit 1; }
"$PY" bench/streamsplat_profile.py metrics "$RUN"
"$PY" - "$RUN/metrics.json" <<'EOF'
import json, sys
metrics = json.load(open(sys.argv[1]))
if not metrics["gate"]["passed"]:
    print("gate: the stage split does not reproduce forward_gaussians: a harness bug; stop")
    sys.exit(3)
print("decisions:", metrics["decisions"])
EOF
