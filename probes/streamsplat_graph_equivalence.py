"""One-off correctness check for C4 of the G10 plan (approved 2026-11-18), no log entry, and it prints no timing:
does capacity.py's batching server with --graphs (each batch size's depth and model stages replayed as CUDA graphs)
give every user the same Gaussians as the same server without graphs? Graphs also need --no-checkpointing (the
training-time block checkpointing reads the CUDA RNG state, which capture forbids), so three servers run, all with
--cache --bf16 --nosync and TF32 matmuls as in G13's `nosync` arm: G13's (checkpointing on), the eager control
(checkpointing off) and the graphs. 4 users at their own places in the clip, served together for 3 updates; for
every update, checkpointing off must equal G13's bitwise, and the graphs' pred_gs["xyz"] and ["rgb"] must agree
with the control by G9's criterion (probes/streamsplat_batch_equivalence.py): a mean absolute difference at least
100x below a mismatched user's, and under 1% of the values off by more than 1e-3; whether they are bitwise equal is
reported. Run it under python -O, as the --nosync server runs: the static head's isnan asserts wait on the host,
which graph capture also forbids. With --depth-bf16 (G16) all three servers run the depth model under bf16
autocast.

docker run --rm --gpus all --network none --user $(id -u):$(id -g) -e HOME=/tmp -e TORCH_HOME=/torch_home \
  -e USER=$USER -v $PWD:$PWD:ro -v $(realpath third_party/StreamSplat):$(realpath third_party/StreamSplat):ro \
  -v ~/.cache/torch:/torch_home:ro -v $(dirname <clip>):$(dirname <clip>):ro \
  streamsplat:5d686e9 python -O $PWD/probes/streamsplat_graph_equivalence.py <clip> [--depth-bf16]
The clip is capacity.VIDEO (python3 -c 'import sys; sys.path.insert(0, "bench"); import capacity; print(capacity.VIDEO)').
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import capacity

if __debug__:
    sys.exit("run under python -O")
video = sys.argv[1]
users = list(range(4))
settings = dict(n=len(users), compile=False, video=video, cache=True, bf16=True, nosync=True,
                depth_bf16="--depth-bf16" in sys.argv[2:])
print("depth_bf16:", settings["depth_bf16"])
g13_step, _ = capacity.setup_streamsplat_server(argparse.Namespace(**settings, checkpointing=True, graphs=False))
eager_step, _ = capacity.setup_streamsplat_server(argparse.Namespace(**settings, checkpointing=False, graphs=False))
graph_step, _ = capacity.setup_streamsplat_server(argparse.Namespace(**settings, checkpointing=False, graphs=True))
for update in range(3):
    _, g13 = g13_step(users)
    g13 = {key: value.clone() for key, value in g13.items()}
    _, eager = eager_step(users)
    eager = {key: value.clone() for key, value in eager.items()}
    unequal = [key for key in eager if not torch.equal(g13[key], eager[key])]
    print(f"update {update}: checkpointing off bitwise equal to G13's server: {not unequal}")
    assert not unequal, unequal
    _, graphed = graph_step(users)
    for key in eager:
        gaps = (graphed[key].float() - eager[key].float()).abs()
        line = (f"update {update} {key}: bitwise equal {torch.equal(graphed[key], eager[key])}, "
                f"mean abs diff {gaps.mean().item():.2e}, max {gaps.max().item():.2e}")
        if key not in ("xyz", "rgb"):
            print(line + " (not judged)")
            continue
        mismatched = (graphed[key][0].float() - eager[key][1].float()).abs().mean().item()
        share = (gaps > 1e-3).float().mean().item()
        print(f"{line}; mismatched user {mismatched:.2e}; share of values off by > 1e-3: {share:.2e}")
        assert gaps.mean().item() * 100 <= mismatched and share < 0.01, (update, key, gaps.mean().item(), mismatched, share)
print("GRAPH_EQUIVALENCE_OK")
