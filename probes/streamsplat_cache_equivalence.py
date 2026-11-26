"""One-off correctness check for C1 of the G10 plan (approved 2026-11-18), no log entry, and it prints no timing:
does capacity.py's batching server with --cache (each user keeps its previous keyframe's one-frame features, the
paper's Algorithm 2) give every user the same Gaussians as the server without it? Two servers, 4 users at their
own places in the clip, served together for 3 updates; every update's pred_gs["xyz"] and ["rgb"] must agree by
G9's criterion (probes/streamsplat_batch_equivalence.py): a mean absolute difference at least 100x below a
mismatched user's, and under 1% of the values off by more than 1e-3, at full FP32 by default. The cached path
also changes the encoder's batch (B new frames instead of 2B), which can change kernel choice and last bits.
First it checks the premise that makes the cache's per-frame depth normalisation equal to the server's pair
min-max: every clip frame's disparity spans exactly 0..255.

docker run --rm --gpus all --network none --user $(id -u):$(id -g) -e HOME=/tmp -e TORCH_HOME=/torch_home \
  -e USER=$USER -v $PWD:$PWD:ro -v $(realpath third_party/StreamSplat):$(realpath third_party/StreamSplat):ro \
  -v ~/.cache/torch:/torch_home:ro -v $(dirname <clip>):$(dirname <clip>):ro \
  streamsplat:5d686e9 python $PWD/probes/streamsplat_cache_equivalence.py <clip> [precision]
The clip is capacity.VIDEO (python3 -c 'import sys; sys.path.insert(0, "bench"); import capacity; print(capacity.VIDEO)').
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import capacity

video = sys.argv[1]
inputs = capacity.load_streamsplat_inputs(argparse.Namespace(compile=False, video=video))
frames = len(inputs[4])
with torch.no_grad():
    for first in range(0, frames, 8):
        _, disparity = capacity.streamsplat_keyframes(inputs, list(range(first, min(first + 8, frames))))
        assert (disparity.amin(dim=(1, 2, 3)) == 0).all() and (disparity.amax(dim=(1, 2, 3)) == 255).all(), first
print(f"premise holds: all {frames} clip frames span exactly 0..255")
del inputs, disparity
torch.cuda.empty_cache()

users = list(range(4))
plain_step, _ = capacity.setup_streamsplat_server(argparse.Namespace(n=len(users), compile=False, video=video, cache=False))
cached_step, _ = capacity.setup_streamsplat_server(argparse.Namespace(n=len(users), compile=False, video=video, cache=True))
precision = sys.argv[2] if len(sys.argv) > 2 else "highest"
torch.set_float32_matmul_precision(precision)
torch.backends.cudnn.allow_tf32 = precision != "highest"
print(f"matmul precision {precision}, cudnn TF32 {torch.backends.cudnn.allow_tf32}")
for update in range(3):
    _, plain = plain_step(users)
    _, cached = cached_step(users)
    for key in plain:
        gaps = (cached[key] - plain[key]).abs()
        line = f"update {update} {key}: mean abs diff {gaps.mean().item():.2e}, max {gaps.max().item():.2e}"
        if key not in ("xyz", "rgb"):
            print(line + " (not judged)")
            continue
        mismatched = (cached[key][0] - plain[key][1]).abs().mean().item()
        share = (gaps > 1e-3).float().mean().item()
        print(f"{line}; mismatched user {mismatched:.2e}; share of values off by > 1e-3: {share:.2e}")
        assert gaps.mean().item() * 100 <= mismatched and share < 0.01, (update, key, gaps.mean().item(), mismatched, share)
print("CACHE_EQUIVALENCE_OK")
