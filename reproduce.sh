#!/usr/bin/env bash
# Reproduce the progress report's StreamSplat numbers on one NVIDIA GPU (README.md, "Reproduce").
#
#   ./reproduce.sh setup        fetch StreamSplat @ 5d686e9, the weights, the KITTI-360 clip and DAVIS 2017; verify sha256
#   ./reproduce.sh build        build the StreamSplat and TeX images
#   ./reproduce.sh smoke        ~10 GPU minutes: does this machine give the published numbers? (PASS/FAIL)
#   ./reproduce.sh experiments  rerun every StreamSplat experiment, G4-G16 (~6 GPU hours, needs an idle GPU)
#   ./reproduce.sh collect      copy the new numbers into results/; `git diff results/` shows what moved
#   ./reproduce.sh report       redraw the figures from results/ and rebuild the PDF (byte-identical when unchanged)
#   ./reproduce.sh all          all of the above, in order
#
# Environment: STREAMSPLAT_IMAGE, TEXLIVE_IMAGE (image tags), DOCKER_BUILD_ARGS (extra docker build flags).
# The experiment runners use the default StreamSplat tag, so `experiments` needs STREAMSPLAT_IMAGE unset.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
export STREAMSPLAT_IMAGE=${STREAMSPLAT_IMAGE:-streamsplat:5d686e9}
TEXLIVE_IMAGE=${TEXLIVE_IMAGE:-texlive-report:medium-usenix}
DOCKER_BUILD_ARGS=${DOCKER_BUILD_ARGS:-}
STREAMSPLAT=$ROOT/third_party/StreamSplat
STREAMSPLAT_COMMIT=5d686e9f5869207055802721eb736f977d1539ef
DATA=$ROOT/third_party/data
TORCH_HUB=$HOME/.cache/torch/hub/checkpoints  # the runners mount ~/.cache/torch into the containers
REPORT=$ROOT/docs/reports/progress_2026-11
export SOURCE_DATE_EPOCH=1795564800  # 2026-11-25: fixed timestamps make the figures and the PDF byte-reproducible

# name | destination directory | sha256 | url
DOWNLOADS="
streamsplat.safetensors|$STREAMSPLAT/checkpoints|8a1506acbf65a07333c096cd0e2131f90eeff5be12d6e9fb716b3a840fc1b6dd|https://drive.usercontent.google.com/download?id=1xMsK8_EfFFdXX9oeP-696xWRDv4qhDnL&export=download&confirm=t
depth_anything_v2_vitl.pth|$STREAMSPLAT/checkpoints|a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345|https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/depth_anything_v2_vitl.pth
dinov2_vitb14_reg4_pretrain.pth|$TORCH_HUB|73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71|https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
vgg.pth|$TORCH_HUB|a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868|https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/vgg.pth
vgg16-397923af.pth|$TORCH_HUB|397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0|https://download.pytorch.org/models/vgg16-397923af.pth
kitti360.mp4|$DATA/kitti360|8c6b13929b2c2aae8b95921d8626f5be06f6afffe05ea4e47940ffeb9906f9fc|https://raw.githubusercontent.com/InternRobotics/AnySplat/5f5e208a7dd57d52e43ea0d553a95eab526e8775/examples/video/kitti360.mp4
DAVIS-2017-trainval-480p.zip|$DATA/davis|e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492|https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
"

fetch() {  # fetch <name> <dir> <sha256> <url>: download unless a verified copy exists
  local name=$1 dir=$2 sha=$3 url=$4
  mkdir -p "$dir"
  if [ -f "$dir/$name" ] && echo "$sha  $dir/$name" | sha256sum -c --status; then
    echo "ok       $name"
    return
  fi
  echo "fetching $name"
  curl -fL --retry 3 -o "$dir/$name.part" "$url"
  echo "$sha  $dir/$name.part" | sha256sum -c --status || { echo "sha256 mismatch for $name from $url"; exit 1; }
  mv "$dir/$name.part" "$dir/$name"
  echo "ok       $name"
}

setup() {
  if [ ! -d "$STREAMSPLAT/.git" ]; then
    git clone https://github.com/DSL-Lab/StreamSplat "$STREAMSPLAT"
    git -C "$STREAMSPLAT" checkout -q "$STREAMSPLAT_COMMIT"
  fi
  [ "$(git -C "$STREAMSPLAT" rev-parse HEAD)" = "$STREAMSPLAT_COMMIT" ] || { echo "$STREAMSPLAT is not at $STREAMSPLAT_COMMIT"; exit 1; }
  git -C "$STREAMSPLAT" submodule update --init --recursive
  echo "ok       StreamSplat @ ${STREAMSPLAT_COMMIT:0:7}"
  local name dir sha url
  while IFS='|' read -r name dir sha url; do
    [ -n "$name" ] && fetch "$name" "$dir" "$sha" "$url"
  done <<< "$DOWNLOADS"
  while IFS='|' read -r name dir sha url; do  # bench/capacity.py stamps this file into every run's meta.json
    [ -n "$name" ] && echo "$sha  $name"
  done <<< "$(echo "$DOWNLOADS" | grep -e '\.pth|' -e '\.safetensors|')" > "$STREAMSPLAT/checkpoints/SHA256SUMS"
  cd "$DATA/davis"
  set -f  # the patterns are for unzip, not for the shell
  unzip -q -n DAVIS-2017-trainval-480p.zip 'DAVIS/ImageSets/2017/*' 'DAVIS/JPEGImages/480p/bear/*'
  unzip -q -n DAVIS-2017-trainval-480p.zip $(sed 's|.*|DAVIS/JPEGImages/480p/&/*|' DAVIS/ImageSets/2017/val.txt)
  set +f
  cd "$ROOT"
  echo "ok       DAVIS 2017 val ($(wc -l < "$DATA/davis/DAVIS/ImageSets/2017/val.txt") sequences) and bear"
}

build() {
  local arch
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  docker build $DOCKER_BUILD_ARGS --build-arg TORCH_CUDA_ARCH_LIST="$arch" -t "$STREAMSPLAT_IMAGE" \
    -f docker/streamsplat/Dockerfile "$STREAMSPLAT"
  docker build $DOCKER_BUILD_ARGS -t "$TEXLIVE_IMAGE" docker/texlive-report
}

in_image() {  # in_image <gpu flag or ""> <writable dir> <command...>: the StreamSplat image, offline, as the runners use it
  local gpus=$1 writable=$2 davis
  shift 2
  davis=$(realpath "$DATA/davis/DAVIS")
  docker run --rm $gpus --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_HOME=/torch_home \
    -e USER="$USER" -e PYTHONPATH="$STREAMSPLAT" -w "$STREAMSPLAT" -v "$ROOT:$ROOT:ro" -v "$STREAMSPLAT:$STREAMSPLAT:ro" \
    -v "$HOME/.cache/torch:/torch_home:ro" -v "$davis:$davis:ro" -v "$writable:$writable" "$STREAMSPLAT_IMAGE" "$@"
}

smoke() {  # G16's back-to-back step 0 for both arms, and G8's full-precision quality on the 2 shortest val sequences
  local OUT=$ROOT/outputs/reproduce/smoke PY=python3 frames short rep arm
  mkdir -p "$OUT"
  . bench/sweep_lib.sh
  check_no_mps
  for rep in 1 2; do
    for arm in fast depthbf16; do
      local flags=(--batch 1 --cache --bf16 --nosync --no-checkpointing --graphs)
      [ "$arm" = depthbf16 ] && flags+=(--depth-bf16)
      measure "$OUT/step0/${arm}_r$rep" contaminated --model streamsplat --no-compile --n 1 --period 0 \
        --max-updates 20 --warmup 3 "${flags[@]}" || { echo "step 0 run failed: $OUT/step0/${arm}_r$rep"; exit 1; }
    done
  done
  frames=$(realpath "$DATA/davis/DAVIS/JPEGImages/480p")
  short=$(for sequence in $(cat "$DATA/davis/DAVIS/ImageSets/2017/val.txt"); do
            echo "$(ls "$frames/$sequence" | wc -l) $sequence"; done | sort -n | head -2 | cut -d' ' -f2)
  if [ ! -f "$OUT/quality.done" ]; then
    rm -rf "$OUT/depth" "$OUT/windows.jsonl"
    wait_free
    in_image "--gpus all" "$OUT" python "$ROOT/experiments/streamsplat-repro/davis_eval.py" depth "$frames" "$OUT/depth" $short \
      > "$OUT/depth.log" 2>&1 || { echo "depth maps failed; read $OUT/depth.log"; exit 1; }
    in_image "--gpus all" "$OUT" python "$ROOT/experiments/streamsplat-repro/davis_eval.py" eval "$frames" "$OUT/depth" \
      "$OUT/windows.jsonl" val $short > "$OUT/eval.log" 2>&1 || { echo "evaluation failed; read $OUT/eval.log"; exit 1; }
    touch "$OUT/quality.done"
  fi
  python3 - "$OUT" "$ROOT/results" <<'EOF'
import json, statistics, sys
from pathlib import Path
out, results = Path(sys.argv[1]), Path(sys.argv[2])

def updates_per_s(run):  # bench/sweep_lib.sh's definition
    rows = [json.loads(line) for log in sorted(run.glob("user*.jsonl")) for line in log.read_text().splitlines()]
    rows = [row for row in rows if row["event"] == "update"]
    return len(rows) / (max(row["end"] for row in rows) - min(row["start"] for row in rows))

published = json.loads((results / "capacity/G16/metrics.json").read_text())["derived"]
ours = {arm: statistics.mean(updates_per_s(out / f"step0/{arm}_r{rep}") for rep in (1, 2)) for arm in ("fast", "depthbf16")}
ratio, published_ratio = ours["depthbf16"] / ours["fast"], published["throughput_ratio"]
throughput = {arm: {"ours": ours[arm], "published": published["step0"][arm]["mean_updates_per_s"],
                    "change": ours[arm] / published["step0"][arm]["mean_updates_per_s"] - 1} for arm in ours}
windows = {(row["seq"], row["start"]): row for row in map(json.loads, (out / "windows.jsonl").read_text().splitlines())}
g8 = {(row["seq"], row["start"]): row for row in map(json.loads, (results / "streamsplat-repro/G8/windows.jsonl").read_text().splitlines())
      if row["split"] == "val" and (row["seq"], row["start"]) in windows}
assert windows.keys() == g8.keys(), "the smoke windows are not a subset of G8's: a bug"
gaps = [abs(windows[key][metric] - g8[key][metric]) for key in windows for metric in ("psnr_given", "psnr_novel")]
quality = {"sequences": sorted({sequence for sequence, _ in windows}), "windows": len(windows),
           "bitwise_equal_windows": sum(all(windows[key][m] == g8[key][m] for m in ("psnr_given", "psnr_novel")) for key in windows),
           "max_abs_change_dB": max(gaps)}
checks = {"throughput within 5% of G16": all(abs(entry["change"]) <= 0.05 for entry in throughput.values()),
          "ratio within 5% of G16": abs(ratio / published_ratio - 1) <= 0.05,
          "quality within 0.01 dB of G8": quality["max_abs_change_dB"] <= 0.01}
metrics = {"throughput": throughput, "ratio": {"ours": ratio, "published": published_ratio}, "quality": quality, "checks": checks}
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
for arm, entry in throughput.items():
    print(f"{arm:10s} {entry['ours']:6.2f} updates/s, published {entry['published']:6.2f} ({entry['change']:+.1%})")
print(f"ratio      {ratio:6.3f}, published {published_ratio:.3f}")
print(f"quality    {quality['bitwise_equal_windows']}/{quality['windows']} windows bitwise equal to G8's, "
      f"max change {quality['max_abs_change_dB']:.4f} dB ({', '.join(quality['sequences'])})")
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", name)
sys.exit(0 if all(checks.values()) else 1)
EOF
}

experiments() {  # dependency order: G12 reads G8's outputs, G16 reads G8's and G12's
  local runner code summary=""
  for runner in experiments/streamsplat-repro/run_G4_davis-bear.sh experiments/capacity/run_G5_streamsplat-cost.sh \
      experiments/capacity/run_G6_streamsplat-1hz.sh experiments/capacity/run_G7_streamsplat-3hz.sh \
      experiments/streamsplat-repro/run_G8_davis-val-authors-metric.sh experiments/capacity/run_G9_streamsplat-batched.sh \
      experiments/capacity/run_G10_streamsplat-stage-profile.sh experiments/capacity/run_G11_streamsplat-cache.sh \
      experiments/capacity/run_G12_streamsplat-bf16.sh experiments/capacity/run_G13_streamsplat-nosync.sh \
      experiments/capacity/run_G14_streamsplat-graphs.sh experiments/capacity/run_G15_streamsplat-fast-1hz.sh \
      experiments/capacity/run_G16_streamsplat-depth-bf16.sh; do
    echo "=== $runner"
    code=0
    bash "$runner" || code=$?
    summary+="$(basename "$runner") exit $code"$'\n'
  done
  echo "=== exit codes (a registered gate that stops a runner, such as G9's exit 3, is a result; LOG.md says which):"
  printf '%s' "$summary"
}

collect() { python3 results/collect.py; }

report() {
  docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e SOURCE_DATE_EPOCH \
    -v "$ROOT:$ROOT" -w "$ROOT" "$STREAMSPLAT_IMAGE" python "$REPORT/make_figures.py"
  docker run --rm --network none -u "$(id -u):$(id -g)" -e SOURCE_DATE_EPOCH -e FORCE_SOURCE_DATE=1 \
    -v "$REPORT:/w" -w /w "$TEXLIVE_IMAGE" latexmk -g -pdf -interaction=nonstopmode -quiet main.tex
  echo "report: $REPORT/main.pdf"
}

[ $# -gt 0 ] || { sed -n '2,13p' "$0"; exit 2; }
for step in "$@"; do
  case $step in
    setup|build|smoke|experiments|collect|report) "$step" ;;
    all) setup; build; smoke; experiments; collect; report ;;
    *) echo "unknown step: $step"; sed -n '2,13p' "$0"; exit 2 ;;
  esac
done
