#!/usr/bin/env bash
# G17 (LOG.md). Does a fresh clone of the public repository, set up only by reproduce.sh, give the published
# numbers? Quotable lane: README.md cites it.
#
# Setup. git clone https://github.com/kylevirtuous1211/virtualization_final_project at commit $1 (default: this
# checkout's HEAD, which must be pushed) into outputs/reproduce/G17/clone; in it,
#   STREAMSPLAT_IMAGE=streamsplat:g17 TEXLIVE_IMAGE=texlive-report:g17 DOCKER_BUILD_ARGS="--no-cache --pull" \
#   ./reproduce.sh setup build smoke report
# with image tags of its own, so the local streamsplat:5d686e9 that the runners use stays as it is. The clone fetches
# StreamSplat, both checkpoints, the clip and DAVIS afresh. The three torch hub weights already sit in ~/.cache/torch,
# where setup only verifies them, so this runner also downloads them into outputs/reproduce/G17/hub_check.
# Controls: G16's step 0 (Opt 11.64, Opt+DepthBF16 15.71 updates/s, ratio 1.350); G8's per-window PSNR and its depth
# maps (outputs/streamsplat-repro/G8/depth, on this machine only); the committed docs/reports/progress_2026-11/main.pdf.
# Known when this header was written: reproduce.sh setup on this machine (every file verified, only the clip was new);
# a smoke run on this machine with the existing image, started before this header was written and not read before
# its commit; its numbers are not results.
#
# Predictions, registered in plan mode before the run:
#   P1 every download matches its pinned sha256, StreamSplat is at 5d686e9, and none needs a login. Refuted by any miss.
#   P2 the --no-cache --pull build succeeds, and /opt/freeze.txt equals the G16 image's except perhaps jaxtyping.
#      Refuted if torch, torchvision, xformers or numpy differ.
#   P3 step 0: each arm's mean of 2 runs within 5% of G16's, and the ratio within 5% of 1.350. Refuted if outside.
#   P4 the shortest val sequence's fp32 depth maps equal G8's byte for byte, and the 2 shortest sequences' windows
#      equal G8's bitwise. Refuted if any window moves by more than 0.01 dB; 0 to 0.01 dB is recorded as not bitwise.
#   P5 the PDF rebuilt from results/ has the text of the committed main.pdf. Refuted if any line differs.
#
# Usage: bash experiments/reproduce/run_G17_fresh-clone-smoke.sh [commit]   (resumable: finished steps are kept)
set -u
cd "$(dirname "$0")/../.."
REPO_URL=https://github.com/kylevirtuous1211/virtualization_final_project
COMMIT=${1:-$(git rev-parse HEAD)}
OUT=$(realpath -m outputs/reproduce/G17)
CLONE=$OUT/clone
G8_DEPTH=$(realpath outputs/streamsplat-repro/G8/depth)
export STREAMSPLAT_IMAGE=streamsplat:g17 TEXLIVE_IMAGE=texlive-report:g17 DOCKER_BUILD_ARGS="--no-cache --pull"
mkdir -p "$OUT"

[ -d "$CLONE/.git" ] || git clone -q "$REPO_URL" "$CLONE" || { echo "clone failed"; exit 1; }
git -C "$CLONE" fetch -q origin && git -C "$CLONE" checkout -q "$COMMIT" || { echo "$COMMIT is not on GitHub"; exit 1; }
echo "clone at $(git -C "$CLONE" rev-parse HEAD)"

for step in setup build smoke report; do
  [ -f "$OUT/$step.ok" ] && continue
  start=$(date +%s)
  (cd "$CLONE" && ./reproduce.sh "$step") > "$OUT/$step.log" 2>&1
  code=$?
  echo "$step exit $code, $(( $(date +%s) - start )) s" | tee -a "$OUT/steps.txt"
  [ "$code" = 0 ] && touch "$OUT/$step.ok"
  [ "$code" = 0 ] || [ "$step" = smoke ] || { echo "stopping: $step failed; read $OUT/$step.log"; exit 1; }
done

mkdir -p "$OUT/hub_check"
grep -F '|$TORCH_HUB|' "$CLONE/reproduce.sh" | while IFS='|' read -r name _ sha url; do
  [ -f "$OUT/hub_check/$name" ] || curl -sfL -o "$OUT/hub_check/$name" "$url"
  echo "$sha  $OUT/hub_check/$name" | sha256sum -c --status && echo "$name ok" || echo "$name MISMATCH"
done > "$OUT/hub_check.txt"

for image in streamsplat:5d686e9 "$STREAMSPLAT_IMAGE"; do
  docker run --rm --network none --entrypoint cat "$image" /opt/freeze.txt > "$OUT/freeze_${image//[:\/]/_}.txt"
done
git -C "$CLONE" show "$COMMIT:docs/reports/progress_2026-11/main.pdf" > "$OUT/committed.pdf"
cp "$CLONE/docs/reports/progress_2026-11/main.pdf" "$OUT/rebuilt.pdf"
docker run --rm --network none -u "$(id -u):$(id -g)" -v "$OUT:/o" -w /o "$TEXLIVE_IMAGE" sh -c \
  'gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=committed.txt committed.pdf &&
   gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=rebuilt.txt rebuilt.pdf'

python3 - "$OUT" "$CLONE" "$G8_DEPTH" "$COMMIT" <<'EOF'
import filecmp, json, sys
from pathlib import Path
out, clone, g8_depth, commit = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
steps = dict(line.split(" exit ", 1) for line in (out / "steps.txt").read_text().splitlines())
smoke = json.loads((clone / "outputs/reproduce/smoke/metrics.json").read_text())
hub = (out / "hub_check.txt").read_text().split("\n")
streamsplat_head = (clone / "third_party/StreamSplat/.git/HEAD").read_text().strip()

def freeze(name):
    return dict(line.split("==", 1) for line in (out / name).read_text().splitlines() if "==" in line)
old, new = freeze("freeze_streamsplat_5d686e9.txt"), freeze("freeze_streamsplat_g17.txt")
differ = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))

sequences = smoke["quality"]["sequences"]
depth = {}
for sequence in sequences:
    ours = sorted((clone / "outputs/reproduce/smoke/depth" / sequence).glob("*_pred.png"))
    depth[sequence] = {"maps": len(ours), "equal_to_G8": sum(filecmp.cmp(p, g8_depth / sequence / p.name, shallow=False) for p in ours)}
frames = {s: len(list((clone / "third_party/data/davis/DAVIS/JPEGImages/480p" / s).glob("*.jpg"))) for s in sequences}
shortest = min(sequences, key=lambda s: frames[s])
committed_text, rebuilt_text = (out / "committed.txt").read_text().splitlines(), (out / "rebuilt.txt").read_text().splitlines()

predictions = {
    "P1": steps["setup"].startswith("0") and all(line.endswith(" ok") for line in hub if line)
          and streamsplat_head.startswith("5d686e9"),
    "P2": steps["build"].startswith("0") and not {"torch", "torchvision", "xformers", "numpy"} & set(differ),
    "P3": smoke["checks"]["throughput within 5% of G16"] and smoke["checks"]["ratio within 5% of G16"],
    "P4": depth[shortest]["equal_to_G8"] == depth[shortest]["maps"] > 0 and smoke["quality"]["max_abs_change_dB"] <= 0.01,
    "P5": committed_text == rebuilt_text,
}
metrics = {"commit": commit, "steps": steps, "hub_check": [line for line in hub if line],
           "streamsplat_head": streamsplat_head, "freeze_differs": {name: [old.get(name), new.get(name)] for name in differ},
           "smoke": smoke, "depth_maps": depth, "shortest": shortest,
           "pdf": {"text_equal": committed_text == rebuilt_text, "bytes_equal": filecmp.cmp(out / "committed.pdf", out / "rebuilt.pdf", shallow=False),
                   "differing_lines": sum(a != b for a, b in zip(committed_text, rebuilt_text)) + abs(len(committed_text) - len(rebuilt_text))},
           "predictions_hold": predictions}
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps({key: metrics[key] for key in ("steps", "freeze_differs", "depth_maps", "pdf", "predictions_hold")}, indent=1))
print(json.dumps({"throughput": smoke["throughput"], "ratio": smoke["ratio"], "quality": smoke["quality"]}, indent=1))
EOF
echo "G17 done: $OUT/metrics.json"
