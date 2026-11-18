"""Host waits in one cached StreamSplat update's model calls (C3 of the G10 plan, G13), in the StreamSplat image.
The update is what capacity.py's batching server runs with --cache --bf16 at batch 1: encode_frames on the new
keyframe, then pair_stage with the previous keyframe's features, inside capacity.model_context. Run it under
`python -O` with --nosync for that arm, since the server drops the static head's isnan asserts that way.

  python [-O] bench/streamsplat_sync_count.py <clip> <out.json> [--nosync]

Writes the CUDA runtime calls torch.profiler records in one update (3 warm-ups first), and checks that the
default-device context leaves the output unchanged: the update with nosync off, twice (the repeat noise), against
the update with it on."""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity, streamsplat_stages

video, out = sys.argv[1], Path(sys.argv[2])
nosync = "--nosync" in sys.argv[3:]
inputs = capacity.load_streamsplat_inputs(argparse.Namespace(compile=False, video=video))
predictor = streamsplat_stages.predictor_of(inputs[0])


def settings(device_context):
    return argparse.Namespace(bf16=True, nosync=device_context)


with torch.no_grad():
    rgb, disparity = capacity.streamsplat_keyframes(inputs, [0, 1])
    depth = capacity.per_frame_depth(disparity)
    with capacity.model_context(settings(nosync)):
        previous = streamsplat_stages.encode_frames(predictor, rgb[:1], depth[:1])

    def update(device_context):
        with capacity.model_context(settings(device_context)):
            features = streamsplat_stages.encode_frames(predictor, rgb[1:], depth[1:])
            return streamsplat_stages.pair_stage(predictor, previous, features)

    for _ in range(3):
        update(nosync)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        update(nosync)
        torch.cuda.synchronize()  # ours, so the profile holds the whole update; counted as one device synchronise
    names = [event.name for event in profile.events()]
    calls = {name: names.count(name) for name in sorted(set(names)) if (name.startswith("cuda") and "Synchron" in name)
             or name.startswith("cudaMemcpy")}
    plain, repeat, device = update(False), update(False), update(True)
    torch.cuda.synchronize()


def max_gap(first, second):
    return max((first[key].float() - second[key].float()).abs().max().item() for key in first)


result = {"python_optimize": not __debug__, "nosync": nosync, "runtime_calls": calls,
          "repeat_bitwise_equal": all(torch.equal(plain[key], repeat[key]) for key in plain),
          "device_context_bitwise_equal": all(torch.equal(plain[key], device[key]) for key in plain),
          "repeat_max_abs_diff": max_gap(plain, repeat), "device_context_max_abs_diff": max_gap(plain, device)}
out.write_text(json.dumps(result, indent=1))
print(json.dumps(result))
