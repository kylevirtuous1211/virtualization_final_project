"""One-off correctness check for G9, no log entry, and it prints no timing: does capacity.py's batched StreamSplat
server give every user the same Gaussians as serving that user alone? Two servers for 4 users, each at its own
place in the clip (so their depth ranges differ, and a batch-wide min-max would leak between users): one runs the
4 users as one batch, the other one at a time; after the same first update, the second update's pred_gs["xyz"]
and ["rgb"] must agree within TF32 noise. A mismatched sample must not, or the check could not see an error.

docker run --rm --gpus all --network none --user $(id -u):$(id -g) -e HOME=/tmp -e TORCH_HOME=/torch_home \
  -e USER=$USER -v $PWD:$PWD:ro -v $(realpath third_party/StreamSplat):$(realpath third_party/StreamSplat):ro \
  -v ~/.cache/torch:/torch_home:ro -v $HOME/Desktop/event_camera_2DGS/third_party/clones/AnySplat/examples/video:$HOME/Desktop/event_camera_2DGS/third_party/clones/AnySplat/examples/video:ro \
  streamsplat:5d686e9 python $PWD/probes/streamsplat_batch_equivalence.py
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import capacity

users = list(range(4))
args = argparse.Namespace(n=len(users), compile=False, video=str(capacity.VIDEO.resolve()))
together_step, _ = capacity.setup_streamsplat_server(args)
alone_step, _ = capacity.setup_streamsplat_server(args)
together_step(users)  # each user's first keyframe pair
for rank in users:
    alone_step([rank])
_, together = together_step(users)
alone = [alone_step([rank])[1] for rank in users]
for key in ("xyz", "rgb"):
    difference = max((together[key][slot] - alone[slot][key][0]).abs().max().item() for slot in users)
    mismatched = (together[key][0] - alone[1][key][0]).abs().max().item()
    print(f"{key}: batched vs one at a time, max abs diff {difference:.2e}; against a mismatched user {mismatched:.2e}")
    assert difference <= 1e-2 and mismatched >= 0.1, (key, difference, mismatched)
print("BATCH_EQUIVALENCE_OK")
