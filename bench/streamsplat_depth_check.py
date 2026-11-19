"""What --depth-bf16 (G16) changes in the depth the server feeds StreamSplat, run in the StreamSplat image.
For every clip frame at batch 1, capacity.disparity_of (the server's own function, TF32 as the server sets it)
with bf16 off and on:
  - positive control: the depth model's output must be bf16 with it on, and the disparity must differ;
  - the difference in 0..255 levels at the model's 288x512: per frame mean and max, and the share of equal pixels;
  - every frame must still span exactly 0..255, which per_frame_depth (--cache) relies on.
Exits 3 if a check fails.

  python bench/streamsplat_depth_check.py <clip> <out.json>
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity

video, out = sys.argv[1], Path(sys.argv[2])
inputs = capacity.load_streamsplat_inputs(argparse.Namespace(compile=False, video=video))
dtypes = []
inputs[1].register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
frames = []
with torch.no_grad():
    for index in range(len(inputs[4])):
        _, pixels = capacity.upload_keyframes(inputs, [index])
        dtypes.clear()
        fp32 = capacity.disparity_of(inputs, pixels)
        bf16 = capacity.disparity_of(inputs, pixels, bf16=True)
        gap = (bf16 - fp32).abs()
        frames.append({"output_dtypes": [str(dtype) for dtype in dtypes],
                       "mean_abs_levels": gap.mean().item(), "max_abs_levels": gap.max().item(),
                       "equal_share": (gap == 0).float().mean().item(),
                       "spans_0_255": all(value.amin().item() == 0 and value.amax().item() == 255 for value in (fp32, bf16))})

failures = []
if any(frame["output_dtypes"] != ["torch.float32", "torch.bfloat16"] for frame in frames):
    failures.append("the depth model's output dtypes are not fp32 then bf16")
if all(frame["equal_share"] == 1 for frame in frames):
    failures.append("bf16 depth equals fp32 depth on every frame: the flag changed nothing")
if not all(frame["spans_0_255"] for frame in frames):
    failures.append("a frame does not span exactly 0..255")
result = {"frames": len(frames), "failures": failures,
          "mean_abs_levels": sum(frame["mean_abs_levels"] for frame in frames) / len(frames),
          "max_abs_levels": max(frame["max_abs_levels"] for frame in frames),
          "equal_share_mean": sum(frame["equal_share"] for frame in frames) / len(frames),
          "per_frame": frames}
out.write_text(json.dumps(result, indent=1))
print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}))
if failures:
    sys.exit(3)
print("DEPTH_CHECK_OK")
