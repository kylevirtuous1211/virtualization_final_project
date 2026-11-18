"""StreamSplat's SplatPredictor.forward (third_party/StreamSplat model/model_utils.py:623-673) split into a
one-frame stage and a pair stage, so the stages can be timed apart and a server can keep each user's previous
keyframe (the paper's Algorithm 2). This is our own code: it calls the predictor's submodules in forward's order
and copies none of its source. docs/reading_notes.md, "StreamSplat released implementation", maps the stages.

Feature dicts hold, per frame (rows = frames): "encoded" [F, 1, 2304, 768], "condition" [F, 1, 576, 768],
"gaussians" (the static head's dict) and "projected" [F, 1, 36864, 192].
Every function takes an optional `timer`, any callable that maps a stage name to a context manager."""
from contextlib import contextmanager, nullcontext

import torch


def predictor_of(model):
    """The SplatPredictor inside the inference SplatModel, compiled or not."""
    module = model.model
    return getattr(module, "_orig_mod", module)


def stage(timer, name):
    return timer(name) if timer else nullcontext()


def frame_features(predictor, encoded, condition, timer=None):
    """The one-frame work after the encoder and DINO: static upsampler, static head and encoder_proj."""
    with stage(timer, "static_upsampler"):
        upsampled = predictor.gs_predictor.upsampling(encoded)
    with stage(timer, "static_head"):
        gaussians, _ = predictor.gs_predictor.predictor(upsampled)
    with stage(timer, "encoder_proj"):
        projected = predictor.encoder_proj(upsampled)
    return {"encoded": encoded, "condition": condition, "gaussians": gaussians, "projected": projected}


def encode_frames(predictor, frames, depths, timer=None):
    """frames [F, 3, H, W] in [0, 1], depths [F, 1, H, W] normalised to [0, 1] per frame: one-frame features."""
    frames, depths = frames[:, None], depths[:, None]
    with stage(timer, "encoder"):
        encoded = predictor.forward_encoder(frames, depths)
    with stage(timer, "dino"):
        condition = predictor.forward_condition(frames, depths)
    return frame_features(predictor, encoded, condition, timer)


def pair_stage(predictor, first, second, timer=None):
    """The pair work on two frames' features (first = previous keyframe): decoder, dynamic upsampler, dynamic head,
    then each frame's Gaussians combined and concatenated, as forward returns them in "pred_gs"."""
    with stage(timer, "decoder"):
        decoded = predictor.decoder(torch.cat([first["encoded"], second["encoded"]], dim=2),
                                    torch.cat([second["condition"], first["condition"]], dim=2))
    tokens = predictor.decoder.token_len
    halves = []
    for features, part in ((first, decoded[:, :, :tokens]), (second, decoded[:, :, tokens:])):
        with stage(timer, "dynamic_upsampler"):
            upsampled = predictor.upsampling(part)
        with stage(timer, "dynamic_head"):
            dynamic, _ = predictor.gs_dynamic_predictor(torch.cat([features["projected"], upsampled], dim=-1))
            gaussians = dict(features["gaussians"])  # combine pops keys, and a cached frame is used twice
            for key in ("xyz_dynamic", "opacity_dynamic"):
                gaussians[key] = dynamic[key]
            halves.append(predictor.combine(gaussians))
    return {key: torch.cat([halves[0][key], halves[1][key]], dim=1) for key in halves[0]}


def split_forward(predictor, frames, depths, timer=None):
    """predictor.forward(frames [B, 2, 3, H, W], depths [B, 2, 1, H, W])["pred_gs"] built from the stages above,
    with the same tensor shapes at every call as forward, so its output should be bitwise identical."""
    pairs = frames.shape[0]
    rows_frames = frames.reshape(pairs * 2, 1, *frames.shape[2:])
    rows_depths = depths.reshape(pairs * 2, 1, *depths.shape[2:])
    with stage(timer, "encoder"):
        encoded = predictor.forward_encoder(rows_frames, rows_depths)
    encoded = encoded.reshape(pairs, 2, *encoded.shape[2:])
    with stage(timer, "dino"):
        condition = predictor.forward_condition(frames, depths)
    first, second = (frame_features(predictor, encoded[:, slot:slot + 1], condition[:, slot:slot + 1], timer)
                     for slot in (0, 1))
    return pair_stage(predictor, first, second, timer)


class StageTimer:
    """CUDA-event seconds per stage name, summed over the stages of that name since the last seconds() call."""

    def __init__(self):
        self.events = []

    @contextmanager
    def __call__(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.events.append((name, start, end))

    def seconds(self):
        torch.cuda.synchronize()
        totals = {}
        for name, start, end in self.events:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end) / 1000
        self.events.clear()
        return totals


class PeakTracker:
    """Peak bytes allocated above the stage's starting allocation, per stage name (largest over calls)."""

    def __init__(self):
        self.peaks = {}

    @contextmanager
    def __call__(self, name):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        yield
        torch.cuda.synchronize()
        self.peaks[name] = max(self.peaks.get(name, 0), torch.cuda.max_memory_allocated() - base)
