#!/usr/bin/env bash
# G8 (LOG.md): by the authors' own metric, does the released StreamSplat come near its paper's Table 1 keyframe
# PSNR (37.83 dB) on DAVIS 2017 val? A reproduction attempt; plan `misty-painting-summit`, approved 2026-11-14.
# It replaces G4's 30 dB gate for G5-G7: G4 (27.51 dB) used a metric unlike the paper's (mp4 at 512x288 against
# 256x256 tensors), so G4's gate stays failed as registered and this entry decides G5-G7 afresh.
#
# Setup: code, weights and image as G4. Model: the authors' training SplatModel (model/splat_model.py) with the
# inference options (configs/options_inference.py, whose heads match the checkpoint), output_frames 6, compile off,
# batch 1; the strict=False load is checked (davis_eval.py keycheck). Data: DAVIS 2017 trainval 480p, the 30 val
# sequences (1,999 frames) and, as a control, `bear` (train, 82 frames). Every window of 6 consecutive frames
# i..i+5, sampled and loaded as datasets/provider_davis.py does unshuffled (output_frames 6 gives interval 1):
# 1,849 val windows and 77 bear windows. Depth: davis_bear.py depth (the authors' preprocess procedure).
# Metric: SplatModel.forward's own metrics['psnr_given'] (the two keyframes) and ['psnr_novel'] (the middle 4),
# both computed by the authors' compute_losses at 256x256 bilinear; the mean over windows per split.
# Known differences from the authors: 480p JPEGs, not Full-Resolution (both are resized to 288x512); no
# Annotations_unsupervised (masks only enter the losses, not PSNR); compile off; Table 1's exact windows are not
# published. Deterministic inference, so one run.
#
# Registered before the run:
#   Gate: val psnr_given mean >= 35.0 dB, then G5-G7 run with their registrations unchanged. Below 35.0 dB the
#     StreamSplat line stops and G5-G7 are not run ("did not reproduce").
#   R1 val psnr_given in 36-38 dB. Refuted outside.
#   R2 val psnr_novel in 22-25 dB (paper 23.66); reported, not a gate. Refuted outside.
#   R3 bear by this protocol >= 33 dB, i.e. most of G4's gap is the metric. Refuted if bear < 31 dB.
#   Per-sequence mean, min and max are reported; sequences differ in length, so they weigh differently.
#
# Usage: bash experiments/streamsplat-repro/run_G8_davis-val-authors-metric.sh           (resumable)
#        bash experiments/streamsplat-repro/run_G8_davis-val-authors-metric.sh keycheck  (CPU, no GPU needed)
#        bash experiments/streamsplat-repro/run_G8_davis-val-authors-metric.sh metrics   (re-derive metrics.json)
set -u
cd "$(dirname "$0")/../.."
OUT=${G8_OUT:-outputs/streamsplat-repro/G8}
PY=python3
. bench/sweep_lib.sh
STREAMSPLAT=$(realpath third_party/StreamSplat)
DAVIS=$(realpath third_party/data/davis/DAVIS)
FRAMES=$DAVIS/JPEGImages/480p
VAL=$(cat "$DAVIS/ImageSets/2017/val.txt")
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
HELPERS=$PWD/experiments/streamsplat-repro

in_image() {  # gpu flag first ("--gpus all" or ""), then the command
  local gpus=$1; shift
  docker run --rm $gpus --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/torch_home \
    -e USER="$USER" -e PYTHONPATH="$STREAMSPLAT" -w "$STREAMSPLAT" -v "$PWD:$PWD:ro" \
    -v "$STREAMSPLAT:$STREAMSPLAT:ro" -v "$HOME/.cache/torch:/torch_home:ro" -v "$DAVIS:$DAVIS:ro" -v "$OUT:$OUT" \
    streamsplat:5d686e9 "$@"
}
metrics() { in_image "" python "$HELPERS/davis_eval.py" metrics "$OUT/windows.jsonl" "$OUT/metrics.json"; }

case "${1:-}" in
  keycheck) in_image "" python "$HELPERS/davis_eval.py" keycheck; exit ;;
  metrics) metrics; exit ;;
esac
if [ ! -f "$OUT/depth.done" ]; then
  wait_free
  in_image "--gpus all" python "$HELPERS/davis_eval.py" depth "$FRAMES" "$OUT/depth" $VAL bear >> "$OUT/depth.log" 2>&1 || exit 1
  for seq in $VAL bear; do
    frames=$(find "$FRAMES/$seq" -name '*.jpg' | wc -l); depths=$(find "$OUT/depth/$seq" -name '*_pred.png' | wc -l)
    [ "$frames" = "$depths" ] || { echo "$seq: depth maps $depths != frames $frames"; exit 1; }
  done
  touch "$OUT/depth.done"
fi
if [ ! -f "$OUT/eval.done" ]; then
  wait_free
  rm -f "$OUT/windows.jsonl"  # a partial evaluation is redone whole
  in_image "--gpus all" python "$HELPERS/davis_eval.py" eval "$FRAMES" "$OUT/depth" "$OUT/windows.jsonl" val $VAL > "$OUT/eval.log" 2>&1 || exit 1
  in_image "--gpus all" python "$HELPERS/davis_eval.py" eval "$FRAMES" "$OUT/depth" "$OUT/windows.jsonl" bear bear >> "$OUT/eval.log" 2>&1 || exit 1
  touch "$OUT/eval.done"
fi
metrics
