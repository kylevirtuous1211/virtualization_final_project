#!/usr/bin/env bash
# G4 (LOG.md): does the released StreamSplat reproduce on the kind of data it was trained for, before G5-G7 time it?
# The first run of a borrowed method is a reproduction attempt. Plan `misty-painting-summit`, approved 2026-11-13.
#
# Setup: code DSL-Lab/StreamSplat @ 5d686e9 (third_party/StreamSplat), weights as in its checkpoints/SHA256SUMS,
# image streamsplat:5d686e9 (docker/streamsplat/Dockerfile). Data: DAVIS 2017 trainval 480p, sequence `bear`
# (82 frames). Depth: the authors' preprocess_depth_davis.py procedure (davis_bear.py depth). Inference: the
# authors' splat_inference.py with every default (frame gap 3, batch 8, torch.compile on; with compile on it drops
# the last partial batch, as it prints). Number: mean PSNR of each rendered keyframe against its input keyframe,
# both read from the two videos the script writes, so both sides carry the same video compression.
#
# Registered before the run: pass if the mean keyframe PSNR is >= 30 dB (the paper's Table 1 reports 37.83 dB on
# DAVIS keyframes; video re-encoding costs a few dB). Below 30 dB the weights or the depth pipeline are wrong:
# stop, and G5-G7 are not run. Before inference, the number of depth maps must equal the number of frames,
# because splat_inference.py silently falls back to zero depth otherwise.
#
# Usage: bash experiments/streamsplat-repro/run_G4_davis-bear.sh           (resumable)
#        bash experiments/streamsplat-repro/run_G4_davis-bear.sh metrics   (re-derive metrics.json)
set -u
cd "$(dirname "$0")/../.."
OUT=${G4_OUT:-outputs/streamsplat-repro/G4}
PY=python3
. bench/sweep_lib.sh
STREAMSPLAT=$(realpath third_party/StreamSplat)
FRAMES=$(realpath third_party/data/davis/DAVIS/JPEGImages/480p/bear)
GAP=3  # splat_inference.py's default --frame_gap
mkdir -p "$OUT"
OUT=$(realpath "$OUT")

in_image() {
  docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/torch_home \
    -e USER="$USER" -e PYTHONPATH="$STREAMSPLAT" -w "$STREAMSPLAT" -v "$PWD:$PWD:ro" -v "$STREAMSPLAT:$STREAMSPLAT:ro" \
    -v "$HOME/.cache/torch:/torch_home:ro" -v "$FRAMES:$FRAMES:ro" -v "$OUT:$OUT" streamsplat:5d686e9 "$@"
}
psnr() { in_image python "$PWD/experiments/streamsplat-repro/davis_bear.py" psnr "$OUT/inference" "$GAP" "$OUT/metrics.json"; }

if [ "${1:-}" = metrics ]; then psnr; exit; fi
if [ ! -f "$OUT/depth/.done" ]; then
  wait_free
  in_image python "$PWD/experiments/streamsplat-repro/davis_bear.py" depth "$FRAMES" "$OUT/depth" > "$OUT/depth.log" 2>&1 || exit 1
  touch "$OUT/depth/.done"
fi
frames=$(find "$FRAMES" -name '*.jpg' | wc -l)
depths=$(find "$OUT/depth" -name '*_pred.png' | wc -l)
if [ "$frames" != "$depths" ]; then
  echo "depth maps $depths != frames $frames: splat_inference.py would fall back to zero depth"
  exit 1
fi
if [ ! -f "$OUT/inference/render_video_0.mp4" ]; then
  wait_free
  mkdir -p "$OUT/inference"
  in_image python splat_inference.py --resume checkpoints/streamsplat.safetensors --input_frames_path "$FRAMES" \
    --input_depths_path "$OUT/depth" --output_dir "$OUT/inference" > "$OUT/inference.log" 2>&1 || exit 1
fi
psnr
