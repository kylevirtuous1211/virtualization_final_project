#!/usr/bin/env bash
# G5 (LOG.md): what does one StreamSplat user cost on the RTX 6000 Ada, with torch.compile on and off?
# The StreamSplat analogue of G1. Exploratory lane. Plan `misty-painting-summit`, approved 2026-11-13. Runs only after G4 passes.
#
# Setup: bench/capacity.py --model streamsplat, one user in its own container (streamsplat:5d686e9), back to back
# (--period 0), 3 warm-up updates, then 20 measured updates, for each of compile on and off (the authors' inference
# default is on). One update = one new keyframe of examples/video/kitti360.mp4 (the clip G1-G3 use) at the
# checkpoint's 512x288: Depth Anything V2 ViT-L depth for it, then Gaussians for the (previous, new) keyframe pair.
# Service time is upload plus GPU work; service_depth is the depth part. No rendering, as in G1.
#
# Registered before the run:
#   service p50 0.09 s (range 0.06-0.15); depth 30-40% of service; weights 2.39 GB (1.34 depth + 0.35 DINOv2 + 0.70
#   StreamSplat; the CPU key check measured 1032 MiB for the model without depth); process VRAM about 4,200 MiB
#   (3,500-5,000). "Lighter than AnySplat at k = 8 (G1: 0.395 s, 8,640 MiB)" is refuted if service p50 > 0.3 s or
#   process VRAM > 8,000 MiB.
#
# Usage: bash experiments/capacity/run_G5_streamsplat-cost.sh [metrics]
set -u
cd "$(dirname "$0")/../.."
PY=python3
OUT=${G5_OUT:-outputs/capacity/G5}
. bench/sweep_lib.sh
if [ "${1:-}" = metrics ]; then metrics compile nocompile; exit; fi
check_no_mps
for arm in compile nocompile; do
  measure "$OUT/$arm/n1/r1" "contaminated crashed" --model streamsplat --n 1 --period 0 --max-updates 20 --warmup 3 \
    "--$( [ "$arm" = compile ] && echo compile || echo no-compile )" || exit 1
done
metrics compile nocompile
echo "G5 done"
