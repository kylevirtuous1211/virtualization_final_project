"""One-off bug check for G5: is the harness's Gaussian part of a StreamSplat update (service minus service_depth,
about 0.14 s) the model's own cost, or overhead my harness adds? Times model.forward_gaussians alone, one pair
(batch 1, as the harness runs it) at 288x512 with inputs already on the GPU, compile off, 3 warm-ups then 20 timed
calls, each synchronised. Inputs are random: the work depends only on their shape.
Batch sizes are optional arguments (default 1); G9 added them to check whether its step 0 (Gaussian part of a batch-8
step 1.37 s, 9.0x batch 1) is the model's own cost or the server's.

docker run --rm --gpus all --network none --user $(id -u):$(id -g) -e HOME=/tmp -e TORCH_HOME=/torch_home \
  -e USER=$USER -v $PWD:$PWD:ro -v $(realpath third_party/StreamSplat):$(realpath third_party/StreamSplat):ro \
  -v ~/.cache/torch:/torch_home:ro streamsplat:5d686e9 python $PWD/probes/streamsplat_forward_timing.py [batch ...]
"""
import statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import capacity

sys.path.insert(0, str(capacity.STREAMSPLAT))
model, opt = capacity.load_streamsplat(False)
for batch in map(int, sys.argv[1:] or ["1"]):
    frames = torch.rand(batch, 2, 3, 288, 512, device="cuda")
    depths = torch.rand(batch, 2, 1, 288, 512, device="cuda")
    times = torch.tensor([[0.0, 1.0]], device="cuda").expand(batch, 2)
    seconds = []
    with torch.no_grad():
        for call in range(23):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.forward_gaussians(frames, depths, times)
            torch.cuda.synchronize()
            if call >= 3:
                seconds.append(time.perf_counter() - start)
    print(f"forward_gaussians batch {batch}: p50 {statistics.median(seconds):.4f} s, min {min(seconds):.4f}, max {max(seconds):.4f}, n {len(seconds)}")
