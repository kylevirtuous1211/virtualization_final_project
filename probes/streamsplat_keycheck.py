"""One-off, no log entry: before any GPU run, check on the CPU inside the StreamSplat image that the released
checkpoint loads into the inference model with no unexpected key and missing keys only in the DINO condition
encoder (the upstream loader uses strict=False and would hide a wrong file), for compile on and off.
SplatModel creates one tensor directly on "cuda"; that single call is redirected to the CPU here.

docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp -e TORCH_HOME=/torch_home -e USER=$USER \
  -v $PWD:$PWD:ro -v $(realpath third_party/StreamSplat):$(realpath third_party/StreamSplat):ro \
  -v ~/.cache/torch:/torch_home:ro streamsplat:5d686e9 python $PWD/probes/streamsplat_keycheck.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import capacity

sys.path.insert(0, str(capacity.STREAMSPLAT))
real_tensor = torch.tensor
torch.tensor = lambda *args, device=None, **kwargs: real_tensor(*args, **kwargs)  # SplatModel's background colour
for compile_model in (False, True):
    model, opt = capacity.load_streamsplat(compile_model, device="cpu")
    print(f"compile={compile_model}: keys OK, {capacity.tensor_bytes(model) / 2**20:.0f} MiB of weights and buffers")
print("KEYCHECK_OK")
