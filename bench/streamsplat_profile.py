"""G10 (LOG.md): where one StreamSplat update spends its GPU time on this GPU. Run by
experiments/capacity/run_G10_streamsplat-stage-profile.sh.

  python bench/streamsplat_profile.py run <video> <out-dir>   (in the StreamSplat image) writes stages.json,
                                                              profile.json, ceiling.json and overlap.json,
                                                              skipping any that exist
  python3 bench/streamsplat_profile.py metrics <out-dir>      (host, no torch) metrics.json: the registered
                                                              predictions and decisions, from those files

Sections:
  stages   for TF32 and bf16 autocast at batch 1, 2, 4, 8: model.forward_gaussians end to end; the same pairs
           through streamsplat_stages.split_forward, timed per stage with CUDA events and checked bitwise against
           forward; the one-frame stage alone on the B new frames (what a per-user cache would still compute);
           the pair stage alone; the depth stage (Depth Anything V2 plus min-max) on the B new frames; peak memory
           per stage; plus forward at batch 1 in full FP32 ("highest") as a reference.
  profile  torch.profiler over one forward at batch 1 and 8, TF32 and bf16: kernel table, attention kernels,
           host synchronisations and copies, GPU busy share (kernel time over unprofiled wall time).
  ceiling  matmul TFLOP/s at 8192^3 and at the model's own shapes, and scaled_dot_product_attention at the
           encoder, decoder and upsampler shapes with each backend forced.
  overlap  two threads, each on its own CUDA stream, one repeating the depth stage and one the pair stage, alone
           and together, with and without a high priority for the depth stream.
Pairs come from the clip (user b's pair is frames 7b and 7b + 1, as the server walks it); work depends only on
shapes. Timings: 3 warm-ups, then 20 timed calls, each synchronised; p50, min, max, n."""
import json, os, statistics, sys, threading, time
from contextlib import contextmanager, nullcontext
from pathlib import Path

BATCHES = (1, 2, 4, 8)
REPS, WARMUP = 20, 3
if os.environ.get("G10_SMOKE"):  # a code check with too few calls to time anything; never a result
    REPS, WARMUP = 2, 1
PAIR_TFLOP = 4.11  # analytical FLOPs of one pair (docs/reading_notes.md, StreamSplat released implementation)


def summary(values):
    return {"p50": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def timed(function, reps=REPS, warmup=WARMUP):
    import torch
    for _ in range(warmup):
        function()
    seconds = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    return summary(seconds)


@contextmanager
def precision(name):
    """tf32: fp32 weights, TF32 matmuls (the served setting); bf16: that plus bf16 autocast; fp32: no TF32."""
    import torch
    matmul = "highest" if name == "fp32" else "high"
    torch.set_float32_matmul_precision(matmul)
    torch.backends.cudnn.allow_tf32 = matmul != "highest"
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=name == "bf16"):
            yield
    finally:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True


class Setup:
    def __init__(self, video):
        import argparse, torch
        import capacity, streamsplat_stages
        self.torch, self.capacity, self.stages = torch, capacity, streamsplat_stages
        self.inputs = capacity.load_streamsplat_inputs(argparse.Namespace(compile=False, video=video))
        self.model = self.inputs[0]
        self.predictor = streamsplat_stages.predictor_of(self.model)
        self.times = torch.tensor([[0.0, 1.0]], device="cuda")

    def keyframes(self, batch, offset):
        frames = len(self.inputs[4])
        return self.capacity.streamsplat_keyframes(self.inputs, [(rank * 7 + offset) % frames for rank in range(batch)])

    def pairs(self, batch):
        """The server's inputs for batch users' first pair, plus the new frames normalised per frame."""
        torch = self.torch
        with torch.no_grad():
            previous_rgb, previous_disparity = self.keyframes(batch, 0)
            rgb, disparity = self.keyframes(batch, 1)
        frames = torch.stack([previous_rgb, rgb], dim=1)
        depths = torch.stack([previous_disparity, disparity], dim=1)
        low, high = depths.amin(dim=(1, 2, 3, 4), keepdim=True), depths.amax(dim=(1, 2, 3, 4), keepdim=True)
        depths = (depths - low) / (high - low + 1e-8)  # bench/capacity.py setup_streamsplat_server's pair min-max
        low, high = disparity.amin(dim=(1, 2, 3), keepdim=True), disparity.amax(dim=(1, 2, 3), keepdim=True)
        return frames, depths, rgb, (disparity - low) / (high - low + 1e-8)

    def forward(self, frames, depths):
        return self.model.forward_gaussians(frames, depths, self.times.expand(frames.shape[0], 2))["pred_gs"]


def per_stage(function, timer):
    """Run function REPS times after WARMUP, timer reset each call: (p50 per stage, total wall summary)."""
    import torch
    for _ in range(WARMUP):
        function(timer)
        timer.seconds()
    runs, walls = [], []
    for _ in range(REPS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function(timer)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - start)
        runs.append(timer.seconds())
    return {name: statistics.median(run[name] for run in runs) for name in runs[0]}, summary(walls)


def stages(setup):
    torch, S = setup.torch, setup.stages
    result = {}
    for name in ("tf32", "bf16"):
        result[name] = {}
        for batch in BATCHES:
            frames, depths, rgb, depth = setup.pairs(batch)
            with precision(name):
                entry = {"forward": timed(lambda: setup.forward(frames, depths))}
                reference, repeat = setup.forward(frames, depths), setup.forward(frames, depths)
                split = S.split_forward(setup.predictor, frames, depths)
                entry["split_identical"] = all(torch.equal(split[key], reference[key]) for key in reference)
                entry["forward_self_identical"] = all(torch.equal(repeat[key], reference[key]) for key in reference)
                entry["split_stages"], entry["split_wall"] = per_stage(
                    lambda timer: S.split_forward(setup.predictor, frames, depths, timer), S.StageTimer())
                entry["frame_stages"], entry["frame_wall"] = per_stage(
                    lambda timer: S.encode_frames(setup.predictor, rgb, depth, timer), S.StageTimer())
                previous = S.encode_frames(setup.predictor, frames[:, 0], depths[:, 0])
                current = S.encode_frames(setup.predictor, rgb, depth)
                entry["pair_stages"], entry["pair_wall"] = per_stage(
                    lambda timer: S.pair_stage(setup.predictor, previous, current, timer), S.StageTimer())
                del previous, current, reference, repeat, split
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                setup.forward(frames, depths)
                torch.cuda.synchronize()
                entry["forward_peak_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
                tracker = S.PeakTracker()
                S.split_forward(setup.predictor, frames, depths, tracker)
                entry["stage_peak_mib"] = {key: value / 2**20 for key, value in tracker.peaks.items()}
            with precision("tf32"):
                entry["depth"] = timed(lambda: setup.keyframes(batch, 1))
            result[name][str(batch)] = entry
            print(f"stages {name} b{batch}: forward p50 {entry['forward']['p50']:.4f} s, "
                  f"split identical {entry['split_identical']}", flush=True)
    frames, depths, _, _ = setup.pairs(1)
    with precision("fp32"):
        result["fp32_highest_b1_forward"] = timed(lambda: setup.forward(frames, depths))
    return result


def profile(setup, out):
    torch = setup.torch
    from torch.profiler import ProfilerActivity, profile as torch_profile
    from torch.autograd import DeviceType
    result = {}
    for name in ("tf32", "bf16"):
        for batch in (1, 8):
            frames, depths, _, _ = setup.pairs(batch)
            with precision(name):
                wall = timed(lambda: setup.forward(frames, depths))
                torch.cuda.synchronize()
                with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    setup.forward(frames, depths)
            if batch == 1:
                prof.export_chrome_trace(str(Path(out, f"trace_{name}_b{batch}.json")))
            rows = prof.key_averages()

            def device_us(row):
                value = getattr(row, "self_device_time_total", None)
                return value if value is not None else row.self_cuda_time_total

            device_rows = [row for row in rows if row.device_type == DeviceType.CUDA]
            kernels = [row for row in device_rows if not row.key.startswith(("Memcpy", "Memset"))]
            kernel_us = sum(device_us(row) for row in device_rows)
            attention = [row for row in kernels if any(word in row.key.lower()
                         for word in ("fmha", "flash", "attention", "attn", "efficient"))]
            runtime = {row.key: row.count for row in rows if row.key.startswith(("cuda", "Memcpy", "Memset"))}
            result[f"{name}_b{batch}"] = {
                "wall": wall,
                "kernel_seconds": kernel_us / 1e6,
                "busy_share": kernel_us / 1e6 / wall["p50"],
                "kernel_launches": sum(row.count for row in kernels),
                "attention_seconds": sum(device_us(row) for row in attention) / 1e6,
                "attention_share": sum(device_us(row) for row in attention) / max(kernel_us, 1),
                "attention_kernels": sorted({row.key[:120] for row in attention}),
                "runtime_calls": runtime,
                "host_syncs": sum(count for key, count in runtime.items() if "Synchronize" in key),
                "htod_copies": sum(count for key, count in runtime.items() if key.startswith("Memcpy HtoD")),
                "top_kernels": [{"name": row.key[:120], "count": row.count, "ms": device_us(row) / 1e3}
                                for row in sorted(kernels, key=device_us, reverse=True)[:40]],
            }
            entry = result[f"{name}_b{batch}"]
            print(f"profile {name} b{batch}: busy {entry['busy_share']:.2f}, syncs {entry['host_syncs']}, "
                  f"attention {entry['attention_share']:.2f}", flush=True)
    return result


def ceiling(setup):
    torch = setup.torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    result = {"matmul": [], "sdpa": []}
    shapes = [("square", 8192, 8192, 8192), ("decoder_qkv", 4608, 768, 2304), ("decoder_mlp_up", 4608, 768, 3072),
              ("decoder_mlp_down", 4608, 3072, 768), ("upsampler_stage1_qkv", 36864, 192, 576),
              ("upsampler_stage1_mlp_up", 36864, 192, 768)]
    for label, rows, inner, columns in shapes:
        for name in ("fp32", "tf32", "bf16"):
            if name == "fp32" and label != "square":
                continue
            dtype = torch.bfloat16 if name == "bf16" else torch.float32
            left = torch.randn(rows, inner, device="cuda", dtype=dtype)
            right = torch.randn(inner, columns, device="cuda", dtype=dtype)
            with precision("fp32" if name == "fp32" else "tf32"):
                seconds = timed(lambda: left @ right, reps=50, warmup=10)
            result["matmul"].append({"shape": label, "m": rows, "k": inner, "n": columns, "precision": name,
                                     "p50": seconds["p50"], "tflops": 2 * rows * inner * columns / seconds["p50"] / 1e12})
            del left, right
    attention_shapes = [("encoder", 2, 12, 2304), ("decoder", 1, 12, 4608), ("upsampler_stage1", 16, 3, 2304)]
    backends = {"default": None, "flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH}
    for label, batch, heads, length in attention_shapes:
        for name in ("tf32", "bf16"):
            dtype = torch.bfloat16 if name == "bf16" else torch.float32
            query, key, value = (torch.randn(batch, heads, length, 64, device="cuda", dtype=dtype) for _ in range(3))
            for backend, choice in backends.items():
                entry = {"shape": label, "batch": batch, "heads": heads, "length": length, "precision": name,
                         "backend": backend}
                try:
                    with precision("tf32"), (sdpa_kernel(choice) if choice else nullcontext()):
                        seconds = timed(lambda: torch.nn.functional.scaled_dot_product_attention(query, key, value),
                                        reps=20, warmup=3)
                    entry.update(p50=seconds["p50"], tflops=4 * batch * heads * length * length * 64 / seconds["p50"] / 1e12)
                except RuntimeError as error:
                    entry["error"] = str(error).splitlines()[0][:200]
                result["sdpa"].append(entry)
            del query, key, value
    return result


def overlap(setup):
    torch, S = setup.torch, setup.stages
    frames, depths, rgb, depth = setup.pairs(1)
    with precision("tf32"):
        previous = S.encode_frames(setup.predictor, frames[:, 0], depths[:, 0])
        current = S.encode_frames(setup.predictor, rgb, depth)
        depth_call = timed(lambda: setup.keyframes(1, 1))["p50"]
        pair_call = timed(lambda: S.pair_stage(setup.predictor, previous, current))["p50"]
    torch.cuda.synchronize()
    pair_reps = REPS
    depth_reps = max(1, round(REPS * pair_call / depth_call))  # about equal work on each stream

    def worker(job, reps, stream, latencies, barrier):
        torch.set_float32_matmul_precision("high")
        with torch.no_grad(), torch.cuda.stream(stream):  # grad mode is per thread
            barrier.wait()
            for _ in range(reps):
                start = time.perf_counter()
                job()
                stream.synchronize()
                latencies.append(time.perf_counter() - start)

    def run_threads(jobs):
        barrier = threading.Barrier(len(jobs) + 1)
        threads = [threading.Thread(target=worker, args=(*job, barrier)) for job in jobs]
        for thread in threads:
            thread.start()
        barrier.wait()
        start = time.perf_counter()
        for thread in threads:
            thread.join()
        return time.perf_counter() - start

    result = {"depth_call_p50": depth_call, "pair_call_p50": pair_call, "depth_reps": depth_reps, "pair_reps": pair_reps}
    for label, priority in (("no_priority", 0), ("depth_high_priority", -1)):
        depth_stream, pair_stream = torch.cuda.Stream(priority=priority), torch.cuda.Stream()
        depth_job = lambda: setup.keyframes(1, 1)
        pair_job = lambda: S.pair_stage(setup.predictor, previous, current)
        run_threads([(depth_job, 2, depth_stream, []), (pair_job, 2, pair_stream, [])])  # warm-up of both streams
        depth_alone, depth_loaded = [], []
        alone = (run_threads([(depth_job, depth_reps, depth_stream, depth_alone)])
                 + run_threads([(pair_job, pair_reps, pair_stream, [])]))
        together = run_threads([(depth_job, depth_reps, depth_stream, depth_loaded), (pair_job, pair_reps, pair_stream, [])])
        result[label] = {"alone_seconds": alone, "together_seconds": together, "gain": alone / together - 1,
                         "depth_latency_alone": summary(depth_alone), "depth_latency_loaded": summary(depth_loaded)}
        print(f"overlap {label}: gain {result[label]['gain']:.3f}", flush=True)
    return result


def run(video, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Path(out, "pid").write_text(str(os.getpid()))  # the runner tells our GPU process from a co-tenant's by this
    setup = Setup(video)
    sections = {"stages": lambda: stages(setup), "profile": lambda: profile(setup, out),
                "ceiling": lambda: ceiling(setup), "overlap": lambda: overlap(setup)}
    for name, section in sections.items():
        path = out / f"{name}.json"
        if path.exists():
            continue
        path.write_text(json.dumps(section(), indent=1))


def metrics(out):
    """The registered quantities and decisions of G10 (runner header), from the section files."""
    out = Path(out)
    stage_data, prof, ceil, over = (json.loads((out / f"{name}.json").read_text())
                                    for name in ("stages", "profile", "ceiling", "overlap"))
    tf32, bf16 = stage_data["tf32"], stage_data["bf16"]
    gate = {}
    for name in ("tf32", "bf16"):
        for batch, entry in stage_data[name].items():
            summed = sum(entry["split_stages"].values())
            gate[f"{name}_b{batch}"] = {"split_identical": entry["split_identical"],
                                        "forward_self_identical": entry["forward_self_identical"],
                                        "stage_sum_over_forward": summed / entry["forward"]["p50"]}
    gate_passed = all(value["split_identical"] and abs(value["stage_sum_over_forward"] - 1) <= 0.10
                      for value in gate.values())
    forward_b1 = tf32["1"]["forward"]["p50"]
    cacheable_share = tf32["1"]["frame_wall"]["p50"] / forward_b1
    bf16_ratio = bf16["1"]["forward"]["p50"] / forward_b1
    busy = prof["tf32_b1"]["busy_share"]
    square = {entry["precision"]: entry["tflops"] for entry in ceil["matmul"] if entry["shape"] == "square"}
    stage_names = ("encoder", "dino", "static_upsampler", "static_head", "encoder_proj",
                   "decoder", "dynamic_upsampler", "dynamic_head")
    curves = {}  # seconds per call of each stage of the cached server's update, by batch
    for stage in stage_names:
        source = "frame_stages" if stage in stage_names[:5] else "pair_stages"
        curves[stage] = {int(batch): tf32[batch][source][stage] for batch in tf32}
    curves["depth"] = {int(batch): tf32[batch]["depth"]["p50"] for batch in tf32}
    per_sample_b4 = {stage: curve[4] / 4 / curve[1] for stage, curve in curves.items()}
    stage_batch, saving_b4 = {}, 0.0
    for stage, curve in curves.items():
        chosen = max([batch for batch in BATCHES if curve[batch] / batch <= 0.9 * curve[1]], default=1)
        stage_batch[stage] = chosen
        used = min(chosen, 4)
        saving_b4 += curve[1] - curve[used] / used
    peak = {name: (stage_data[name]["8"]["forward_peak_mib"] - stage_data[name]["1"]["forward_peak_mib"]) / 7
            for name in ("tf32", "bf16")}
    peak_stage = {name: max(stage_data[name]["8"]["stage_peak_mib"].items(), key=lambda item: item[1])[0]
                  for name in ("tf32", "bf16")}
    gain = over["depth_high_priority"]["gain"]
    result = {
        "gate": {"passed": gate_passed, "per_run": gate},
        "P1_cacheable_share_b1_tf32": cacheable_share,
        "P2_bf16_over_tf32_forward_b1": bf16_ratio,
        "P3_busy_share_b1_tf32": busy, "P3_host_syncs_b1_tf32": prof["tf32_b1"]["host_syncs"],
        "P3_pageable_htod_copies_b1_tf32": prof["tf32_b1"]["htod_copies"],
        "P4_attention_kernels": {key: prof[key]["attention_kernels"] for key in ("tf32_b1", "bf16_b1")},
        "P4_attention_share_b1_tf32": prof["tf32_b1"]["attention_share"],
        "P5_matmul_ceiling_tflops": square,
        "P5_achieved_share_of_tf32_ceiling_b1": PAIR_TFLOP / forward_b1 / square["tf32"],
        "P6_per_sample_time_b4_over_b1": per_sample_b4,
        "P6_stage_batch_by_rule": stage_batch, "P6_saving_per_update_b4_seconds": saving_b4,
        "P7_overlap_gain": {key: over[key]["gain"] for key in ("no_priority", "depth_high_priority")},
        "P8_peak_mib_per_extra_sample": peak, "P8_peak_stage_b8": peak_stage,
        "forward_p50": {name: {batch: stage_data[name][batch]["forward"]["p50"] for batch in stage_data[name]}
                        for name in ("tf32", "bf16")},
        "fp32_highest_b1_forward_p50": stage_data["fp32_highest_b1_forward"]["p50"],
        "cached_update_estimate_b1_tf32": tf32["1"]["depth"]["p50"] + tf32["1"]["frame_wall"]["p50"] + tf32["1"]["pair_wall"]["p50"],
        "decisions": {
            "C1_cache": cacheable_share >= 0.12,
            "C2_bf16": bf16_ratio <= 0.87,
            "C3_nosync": busy < 0.90,
            "C4_graphs": busy < 0.90,
            "C5_stage_batching": saving_b4 >= 0.010,
            "C6_two_streams": gain >= 0.05,
            "stop_all": cacheable_share < 0.12 and bf16_ratio > 0.87 and busy > 0.95,
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({key: value for key, value in result.items() if key != "gate"} | {"gate_passed": gate_passed}, indent=1))


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "run":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        run(*arguments)
    else:
        metrics(*arguments)
