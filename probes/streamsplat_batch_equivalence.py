"""One-off correctness check for G9, no log entry, and it prints no timing: does capacity.py's batched StreamSplat
server give every user the same Gaussians as serving that user alone? Two servers for 4 users, each at its own
place in the clip (so their depth ranges differ, and a batch-wide min-max would leak between users): one runs the
4 users as one batch, the other one at a time; after the same first update, the second update's pred_gs["xyz"]
and ["rgb"] must agree within rounding: a mean absolute difference at least 100x below a mismatched user's, and
under 1% of the values off by more than 1e-3. A leak through a shared normalisation would move every Gaussian;
rounding moves a few (2026-11-15: the first criterion, max abs diff <= 1e-2, was beaten by a handful of outliers,
4.1e-2 in xyz under TF32 and 3.6e-3 without it, while the mean was 3.5e-5; the user approved this criterion).
It runs at full FP32 by default: a leak is a logic error and does not depend on precision, while under TF32
(about 5e-4 relative) 3.1% of rgb values sit more than 1e-3 apart from rounding alone (user's choice, same day).
G9's timed runs keep TF32, the authors' setting.

docker run --rm --gpus all --network none --user $(id -u):$(id -g) -e HOME=/tmp -e TORCH_HOME=/torch_home \
  -e USER=$USER -v $PWD:$PWD:ro -v $(realpath third_party/StreamSplat):$(realpath third_party/StreamSplat):ro \
  -v ~/.cache/torch:/torch_home:ro -v $HOME/Desktop/event_camera_2DGS/third_party/clones/AnySplat/examples/video:$HOME/Desktop/event_camera_2DGS/third_party/clones/AnySplat/examples/video:ro \
  streamsplat:5d686e9 python $PWD/probes/streamsplat_batch_equivalence.py \
  $HOME/Desktop/event_camera_2DGS/third_party/clones/AnySplat/examples/video/kitti360.mp4
The clip path is an argument because HOME is /tmp inside the container, so capacity.VIDEO does not resolve there.
An optional second argument sets the matmul precision after loading (default "highest", TF32 off; "high" is
what load_streamsplat sets and the timed runs use): rounding shrinks without TF32, a leak does not.
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import capacity

users = list(range(4))
args = argparse.Namespace(n=len(users), compile=False, video=sys.argv[1])
together_step, _ = capacity.setup_streamsplat_server(args)
alone_step, _ = capacity.setup_streamsplat_server(args)
precision = sys.argv[2] if len(sys.argv) > 2 else "highest"
torch.set_float32_matmul_precision(precision)
torch.backends.cudnn.allow_tf32 = precision != "highest"
print(f"matmul precision {precision}, cudnn TF32 {torch.backends.cudnn.allow_tf32}")
together_step(users)  # each user's first keyframe pair
for rank in users:
    alone_step([rank])
_, together = together_step(users)
alone = [alone_step([rank])[1] for rank in users]
for key in ("xyz", "rgb"):
    per_user = [(together[key][slot] - alone[slot][key][0]).abs().max().item() for slot in users]
    difference = max(per_user)
    mismatched = (together[key][0] - alone[1][key][0]).abs().max().item()
    gaps = torch.stack([(together[key][slot] - alone[slot][key][0]).abs() for slot in users])
    print(f"{key}: batched vs one at a time, max abs diff {difference:.2e} (per user {', '.join(f'{value:.1e}' for value in per_user)}; "
          f"|{key}| max {together[key].abs().max().item():.2e}); against a mismatched user {mismatched:.2e}")
    # a leak through a shared normalisation moves every Gaussian; rounding flips a few
    print(f"{key}: mean abs diff {gaps.mean().item():.2e} (mismatched {(together[key][0] - alone[1][key][0]).abs().mean().item():.2e}), "
          f"share of values off by > 1e-3: {(gaps > 1e-3).float().mean().item():.2e}")
    mean_gap, mean_mismatched = gaps.mean().item(), (together[key][0] - alone[1][key][0]).abs().mean().item()
    assert mean_gap * 100 <= mean_mismatched and (gaps > 1e-3).float().mean().item() < 0.01, (key, mean_gap, mean_mismatched)
print("BATCH_EQUIVALENCE_OK")
