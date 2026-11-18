#!/usr/bin/env python
"""Capacity probe (LOG.md G1-G9): N users share one GPU, each a separate process with its own model copy,
or (--batch, streamsplat only) all N users served by one process that batches their pending updates.

  user   : load the model, then every --period s run one update; one JSONL line per update.
           anysplat: reconstruct the latest --k frames.
           streamsplat: estimate depth for the next keyframe, then predict Gaussians for the (previous, next) pair.
  server : streamsplat with one model for --n users on the same release schedules as user; each step takes the
           pending updates (at most one per user), oldest release first, up to --batch, and runs them as one batch.
  sweep  : refuse to run unless the GPU is free, optionally start a user-level MPS daemon, spawn --n users (or one
           server), sample nvidia-smi every 0.5 s, then write summary.json into a fresh --run-dir.

The sweep itself only needs the stdlib. anysplat users run on the host with ENV_PY; each streamsplat user is
its own container from STREAMSPLAT_IMAGE (--pid=host, so nvidia-smi on the host sees the user's own PID).
"""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parent.parent
ANYSPLAT = HOME / "Desktop/event_camera_2DGS/third_party/clones/AnySplat"  # @ 5f5e208, read only
ENV_PY = HOME / "Desktop/event_camera_2DGS/envs/anysplat/bin/python"      # torch 2.2.0+cu121
HF_REPO, HF_REV = "lhjiang/anysplat", "d2e8c343672646041ad4ea518184968f94362f01"
VIDEO = ANYSPLAT / "examples/video/kitti360.mp4"                           # 64 real frames, 448x336
SIZE = 448
STREAMSPLAT = (REPO / "third_party/StreamSplat").resolve()  # @ 5d686e9, untracked: the repository has no license
STREAMSPLAT_IMAGE = "streamsplat:5d686e9"                   # docker/streamsplat/Dockerfile
TORCH_HUB = HOME / ".cache/torch"                            # holds dinov2_vitb14_reg4 and vgg16, so users run offline


def load_frames(path):
    """All frames as uint8 [F, 3, 448, 448], resized and centre-cropped like AnySplat's process_image."""
    import imageio.v2 as iio, numpy as np, torch
    from PIL import Image
    out = []
    for array in iio.get_reader(path):
        image = Image.fromarray(array).convert("RGB")
        scale = SIZE / min(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
        left, top = (image.width - SIZE) // 2, (image.height - SIZE) // 2
        image = image.crop((left, top, left + SIZE, top + SIZE))
        out.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    return torch.stack(out)


def tensor_bytes(*modules):
    return sum(tensor.numel() * tensor.element_size()
               for module in modules for tensor in [*module.parameters(), *module.buffers()])


def setup_anysplat(args):
    """Returns (step, weight_bytes); step(update) runs one update and returns the fields it adds to the log."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(ANYSPLAT))
    import torch
    from src.model.model.anysplat import AnySplat
    frames = load_frames(args.video)
    model = AnySplat.from_pretrained(HF_REPO, revision=HF_REV).cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    windows = len(frames) - args.k + 1
    assert windows > 0, f"clip has {len(frames)} frames, fewer than k={args.k}"

    def step(update):
        first = (args.rank * 7 + update) % windows  # each user at its own place in the clip
        batch = frames[first:first + args.k].cuda().float().div_(255).unsqueeze(0)  # [1, k, 3, 448, 448] in [0, 1]
        gaussians, _ = model.inference(batch)
        torch.cuda.synchronize()
        return {"gaussians": gaussians.means.shape[1]}

    return step, tensor_bytes(model)


def load_streamsplat(compile_model, device="cuda"):
    """StreamSplat's inference model with its released weights, loaded as splat_inference.py does, plus the check
    that its strict=False load hides: no unexpected key, and missing keys only in the DINO condition encoder."""
    import torch
    from safetensors.torch import load_file
    from configs.options_inference import Options
    from model.splat_model_inference import SplatModel
    torch.set_float32_matmul_precision("high")
    opt = Options()
    opt.input_frames, opt.output_frames, opt.epoch, opt.compile = 1, 2, 0, compile_model
    assert (opt.image_height, opt.image_width) == (288, 512), (opt.image_height, opt.image_width)
    model = SplatModel(opt).to(device)
    remapped = {}
    for key, value in load_file(str(STREAMSPLAT / "checkpoints/streamsplat.safetensors"), device=device).items():
        if "_orig_mod." in key and not compile_model:
            key = key.replace("_orig_mod.", "")
        if "_orig_mod." not in key and compile_model:
            key = key.replace("model.", "model._orig_mod.", 1)
        remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert not unexpected and not [key for key in missing if "condition_encoder" not in key], (missing, unexpected)
    return model.eval(), opt


def load_streamsplat_inputs(args):
    """StreamSplat, its depth model, and every clip frame resized on the CPU for both."""
    sys.path.insert(0, str(STREAMSPLAT))
    import imageio.v2 as iio, numpy as np, torch
    from PIL import Image
    from model.depth_wrapper import DepthAnythingWrapper
    model, opt = load_streamsplat(args.compile)
    depth_model = DepthAnythingWrapper(opt.depth_model_name).to("cuda")
    height, width = opt.image_height, opt.image_width
    model_inputs, depth_inputs = [], []
    for array in iio.get_reader(args.video):
        image = Image.fromarray(array).convert("RGB")
        model_inputs.append(torch.from_numpy(np.asarray(image.resize((width, height), Image.BILINEAR)).copy()).permute(2, 0, 1))
        scale = 518 / min(image.size)
        shorter = image.resize((int(image.width * scale), int(image.height * scale)), Image.BILINEAR)
        depth_size = (round(shorter.width / 14) * 14, round(shorter.height / 14) * 14)
        depth_inputs.append(torch.from_numpy(np.asarray(shorter.resize(depth_size, Image.BILINEAR)).copy()).permute(2, 0, 1))
    return model, depth_model, height, width, model_inputs, depth_inputs


def setup_streamsplat(args):
    """Returns (step, weight_bytes). One update = one new keyframe: Depth Anything V2 depth for it, prepared as
    preprocess_depth_davis.py does (un-normalised RGB at shorter side 518, min-max to uint8), then Gaussians for
    the (previous, new) keyframe pair as splat_inference.py prepares it. Frames are resized on the CPU at load time,
    like AnySplat's; service time is upload plus GPU work. Each user walks the clip from its own offset."""
    import torch, torch.nn.functional as functional
    model, depth_model, height, width, model_inputs, depth_inputs = load_streamsplat_inputs(args)
    previous, consumed = None, 0
    times = torch.tensor([[0.0, 1.0]], device="cuda")

    def keyframe(index):
        rgb = model_inputs[index].cuda().float().div_(255)  # [3, H, W]
        disparity = depth_model(depth_inputs[index].cuda().float().div_(255).unsqueeze(0))  # [1, h, w]
        disparity = functional.interpolate(disparity[:, None], size=(height, width), mode="bilinear", align_corners=True)[0]
        disparity = ((disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8) * 255).floor()
        return rgb, disparity  # [3, H, W], [1, H, W] in 0..255

    def step(update):
        nonlocal previous, consumed
        with torch.no_grad():
            if previous is None:
                previous = keyframe((args.rank * 7 + consumed) % len(model_inputs))
                consumed += 1
            depth_start = time.time()
            current = keyframe((args.rank * 7 + consumed) % len(model_inputs))
            torch.cuda.synchronize()
            depth_seconds = time.time() - depth_start
            consumed += 1
            frames = torch.stack([previous[0], current[0]]).unsqueeze(0)  # [1, 2, 3, H, W]
            depths = torch.stack([previous[1], current[1]]).unsqueeze(0)  # [1, 2, 1, H, W]
            depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
            output = model.forward_gaussians(frames, depths, times)
            torch.cuda.synchronize()
        previous = current
        return {"gaussians": output["pred_gs"]["xyz"].shape[1], "service_depth": depth_seconds}

    return step, tensor_bytes(model, depth_model)


def streamsplat_keyframes(inputs, indices):
    """inputs as load_streamsplat_inputs returns them; RGB in [0, 1] and Depth Anything disparity min-maxed to
    0..255 per frame, for the listed clip frames, as one batch: [B, 3, H, W], [B, 1, H, W]."""
    import torch, torch.nn.functional as functional
    _, depth_model, height, width, model_inputs, depth_inputs = inputs
    rgb = torch.stack([model_inputs[index] for index in indices]).cuda().float().div_(255)
    disparity = depth_model(torch.stack([depth_inputs[index] for index in indices]).cuda().float().div_(255))  # [B, h, w]
    disparity = functional.interpolate(disparity[:, None], size=(height, width), mode="bilinear", align_corners=True)
    low, high = disparity.amin(dim=(1, 2, 3), keepdim=True), disparity.amax(dim=(1, 2, 3), keepdim=True)
    return rgb, ((disparity - low) / (high - low + 1e-8) * 255).floor()


def per_frame_depth(disparity):
    """[B, 1, H, W] disparity in 0..255 to [0, 1], each frame by its own min-max. It equals the server's pair
    min-max whenever every frame spans exactly 0..255, which probes/streamsplat_cache_equivalence.py checks."""
    low, high = disparity.amin(dim=(1, 2, 3), keepdim=True), disparity.amax(dim=(1, 2, 3), keepdim=True)
    return (disparity - low) / (high - low + 1e-8)


def batch_of(rows):
    """Per-user feature dicts (leading dim 1, nested dicts allowed) to one batch, in order."""
    import torch
    if len(rows) == 1:
        return rows[0]
    return {key: batch_of([row[key] for row in rows]) if isinstance(value, dict) else torch.cat([row[key] for row in rows])
            for key, value in rows[0].items()}


def row_of(features, slot, copy):
    """One user's slice of a batch of feature dicts; copied when other users' rows would otherwise stay alive."""
    import torch
    return {key: row_of(value, slot, copy) if isinstance(value, dict) else
            (value[slot:slot + 1].clone() if copy else value[slot:slot + 1]) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()}


def setup_streamsplat_server(args):
    """Returns (step, weight_bytes); step(ranks) runs one update for each listed user as one batch and returns
    (fields, pred_gs). The same update as setup_streamsplat's, with every min-max taken per sample, since a
    batch-wide one would let users change each other's inputs. With --cache (C1 of G10's plan, the paper's
    Algorithm 2) each user keeps its previous keyframe's one-frame features, so an update encodes only the new
    keyframe and then runs the pair stage (bench/streamsplat_stages.py)."""
    import torch
    inputs = load_streamsplat_inputs(args)
    model, depth_model, _, _, model_inputs, _ = inputs
    previous, consumed = [None] * args.n, [0] * args.n
    times = torch.tensor([[0.0, 1.0]], device="cuda")
    if getattr(args, "cache", False):
        import streamsplat_stages
        predictor = streamsplat_stages.predictor_of(model)

    def keyframes(indices):
        return streamsplat_keyframes(inputs, indices)

    def clip_index(rank):
        return (rank * 7 + consumed[rank]) % len(model_inputs)

    def cached_step(ranks):
        with torch.no_grad():
            unprimed = [rank for rank in ranks if previous[rank] is None]
            if unprimed:
                rgb, disparity = keyframes([clip_index(rank) for rank in unprimed])
                features = streamsplat_stages.encode_frames(predictor, rgb, per_frame_depth(disparity))
                for slot, rank in enumerate(unprimed):
                    previous[rank] = row_of(features, slot, len(unprimed) > 1)
                    consumed[rank] += 1
            depth_start = time.time()
            rgb, disparity = keyframes([clip_index(rank) for rank in ranks])
            torch.cuda.synchronize()
            depth_seconds = time.time() - depth_start
            features = streamsplat_stages.encode_frames(predictor, rgb, per_frame_depth(disparity))
            pred_gs = streamsplat_stages.pair_stage(predictor, batch_of([previous[rank] for rank in ranks]), features)
            torch.cuda.synchronize()
        for slot, rank in enumerate(ranks):
            previous[rank] = row_of(features, slot, len(ranks) > 1)
            consumed[rank] += 1
        return {"gaussians": pred_gs["xyz"].shape[1], "service_depth": depth_seconds}, pred_gs

    def step(ranks):
        with torch.no_grad():
            unprimed = [rank for rank in ranks if previous[rank] is None]
            if unprimed:
                rgb, disparity = keyframes([clip_index(rank) for rank in unprimed])
                for slot, rank in enumerate(unprimed):
                    previous[rank] = (rgb[slot], disparity[slot])
                    consumed[rank] += 1
            depth_start = time.time()
            rgb, disparity = keyframes([clip_index(rank) for rank in ranks])
            torch.cuda.synchronize()
            depth_seconds = time.time() - depth_start
            frames = torch.stack([torch.stack([previous[rank][0], rgb[slot]]) for slot, rank in enumerate(ranks)])  # [B, 2, 3, H, W]
            depths = torch.stack([torch.stack([previous[rank][1], disparity[slot]]) for slot, rank in enumerate(ranks)])  # [B, 2, 1, H, W]
            low, high = depths.amin(dim=(1, 2, 3, 4), keepdim=True), depths.amax(dim=(1, 2, 3, 4), keepdim=True)
            depths = (depths - low) / (high - low + 1e-8)
            output = model.forward_gaussians(frames, depths, times.expand(len(ranks), 2))  # cond_times is unused upstream
            torch.cuda.synchronize()
        for slot, rank in enumerate(ranks):
            previous[rank] = (rgb[slot], disparity[slot])
            consumed[rank] += 1
        return {"gaussians": output["pred_gs"]["xyz"].shape[1], "service_depth": depth_seconds}, output["pred_gs"]

    return cached_step if getattr(args, "cache", False) else step, tensor_bytes(model, depth_model)


def serve(args):
    """One process, --n users: the release schedule, termination and log lines of user, for every user at once."""
    import torch
    outs = [open(Path(args.out, f"user{rank}.jsonl"), "w") for rank in range(args.n)]

    def emit(rank, **fields):
        outs[rank].write(json.dumps({"rank": rank, "pid": os.getpid(), "t": time.time(), **fields}) + "\n")
        outs[rank].flush()

    try:
        step, weight_bytes = setup_streamsplat_server(args)
        for first in range(0, args.n, args.batch):  # every user's first keyframe
            step(list(range(first, min(first + args.batch, args.n))))
        for _ in range(args.warmup):  # every batch size the run can form
            for size in range(1, min(args.batch, args.n) + 1):
                step(list(range(size)))
        torch.cuda.reset_peak_memory_stats()
        for rank in range(args.n):
            emit(rank, event="ready", weight_bytes=weight_bytes, reserved=torch.cuda.memory_reserved())
        Path(args.barrier, "ready.0").touch()
        go = Path(args.barrier, "go")
        while not go.exists():
            time.sleep(0.01)
        origin = [float(go.read_text()) + rank / args.n * args.period for rank in range(args.n)]  # as user's t0
        done = [0] * args.n
        active = set(range(args.n))

        def release(rank):
            return origin[rank] + done[rank] * args.period

        while active:
            now = time.time()
            active = {rank for rank in active if not (args.max_updates and done[rank] >= args.max_updates)
                      and max(release(rank), now) - origin[rank] < args.duration}
            pending = sorted((rank for rank in active if release(rank) <= now), key=release)[:args.batch]
            if not pending:
                if active:
                    time.sleep(max(0.0, min(release(rank) for rank in active) - time.time()))
                continue
            start = time.time()
            fields, _ = step(pending)
            end = time.time()
            for rank in pending:
                emit(rank, event="update", j=done[rank], release=release(rank), start=start, end=end,
                     latency=end - release(rank), service=end - start, batch=len(pending),
                     peak_alloc=torch.cuda.max_memory_allocated(), reserved=torch.cuda.memory_reserved(), **fields)
                done[rank] += 1
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if "out of memory" not in str(error).lower():
            raise
        for rank in range(args.n):
            emit(rank, event="oom", msg=str(error)[:300])
        sys.exit(3)


def user(args):
    import torch
    out = open(args.out, "w")

    def emit(**fields):
        out.write(json.dumps({"rank": args.rank, "pid": os.getpid(), "t": time.time(), **fields}) + "\n")
        out.flush()

    try:
        step, weight_bytes = (setup_streamsplat if args.model == "streamsplat" else setup_anysplat)(args)
        for update in range(args.warmup):
            step(update)
        torch.cuda.reset_peak_memory_stats()
        emit(event="ready", weight_bytes=weight_bytes, reserved=torch.cuda.memory_reserved())
        Path(args.barrier, f"ready.{args.rank}").touch()
        go = Path(args.barrier, "go")
        while not go.exists():
            time.sleep(0.01)
        t0 = float(go.read_text()) + args.rank / args.n * args.period  # evenly staggered phases
        update = 0
        while not (args.max_updates and update >= args.max_updates):
            release = t0 + update * args.period
            if max(release, time.time()) - t0 >= args.duration:
                break
            time.sleep(max(0.0, release - time.time()))
            start = time.time()
            fields = step(update)
            end = time.time()
            # latency includes waiting behind this user's own late updates; service is compute only
            emit(event="update", j=update, release=release, start=start, end=end, latency=end - release,
                 service=end - start, peak_alloc=torch.cuda.max_memory_allocated(),
                 reserved=torch.cuda.memory_reserved(), **fields)
            update += 1
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if "out of memory" not in str(error).lower():
            raise
        emit(event="oom", msg=str(error)[:300])
        sys.exit(3)


def smi(*query):
    result = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return result.stdout.strip()


def sample(path, stop):
    with open(path, "w") as out:
        while not stop.is_set():
            now = time.time()
            out.write(json.dumps({"t": now, "apps": smi("--query-compute-apps=pid,process_name,used_memory").splitlines(),
                                  "gpu": smi("--query-gpu=utilization.gpu,memory.used")}) + "\n")
            out.flush()
            stop.wait(max(0.0, 0.5 - (time.time() - now)))


def provenance(args):
    def git(where, *command):
        return subprocess.run(["git", "-C", str(where), *command], capture_output=True, text=True).stdout.strip()
    common = {"argv": sys.argv, "args": vars(args), "commit": git(REPO, "rev-parse", "HEAD"),
              "dirty": bool(git(REPO, "status", "--porcelain")), "video": str(args.video),
              "gpu": smi("--query-gpu=name,driver_version,memory.total"),
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.model == "streamsplat":
        image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", STREAMSPLAT_IMAGE],
                               capture_output=True, text=True).stdout.strip()
        return {**common, "streamsplat_commit": git(STREAMSPLAT, "rev-parse", "HEAD"), "image": f"{STREAMSPLAT_IMAGE} {image}",
                "weights_sha256": (STREAMSPLAT / "checkpoints/SHA256SUMS").read_text()}
    return {**common, "anysplat_commit": git(ANYSPLAT, "rev-parse", "HEAD"), "hf": f"{HF_REPO}@{HF_REV}",
            "env_freeze_sha256": hashlib.sha256((ENV_PY.parent.parent / "freeze.txt").read_bytes()).hexdigest()}


def percentile(values, q):
    """Nearest-rank percentile; None for no values."""
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def summarise(run, args, procs, t_go):
    expected = args.max_updates or math.ceil(args.duration / args.period)
    users = []
    for rank in range(args.n):
        proc = procs[rank % len(procs)]
        log = run / f"user{rank}.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        updates = [row for row in rows if row["event"] == "update"]
        ready = next((row for row in rows if row["event"] == "ready"), {})
        latency, service = [row["latency"] for row in updates], [row["service"] for row in updates]
        oom = any(row["event"] == "oom" for row in rows)
        # the PID the user process reports itself: under docker --pid=host that is its host PID, not the client's
        users.append({"rank": rank, "pid": rows[0]["pid"] if rows else proc.pid, "rc": proc.returncode, "oom": oom,
                      "updates": len(updates), "expected": expected,
                      "lat_p50": percentile(latency, .5), "lat_p95": percentile(latency, .95),
                      "svc_p50": percentile(service, .5), "svc_p95": percentile(service, .95),
                      "svc_depth_p50": percentile([row["service_depth"] for row in updates if "service_depth" in row], .5),
                      "gaussians_p50": percentile([row["gaussians"] for row in updates], .5),
                      "weight_bytes": ready.get("weight_bytes"),
                      "peak_alloc": max((row["peak_alloc"] for row in updates), default=None),
                      "meets_slo": proc.returncode == 0 and not oom and len(updates) >= expected
                                   and percentile(latency, .95) <= args.slo,
                      **({"batch_p50": percentile([row["batch"] for row in updates], .5),
                          "batch_max": max((row["batch"] for row in updates), default=None)} if args.batch else {})})
    samples = [json.loads(line) for line in (run / "smi.jsonl").read_text().splitlines()]
    proc_mib, foreign = {}, set()
    ours = {entry["pid"] for entry in users}
    for row in samples:
        for line in row["apps"]:
            pid, rest = line.split(", ", 1)
            name, mib = rest.rsplit(", ", 1)
            if int(pid) not in ours and not name.endswith("nvidia-cuda-mps-server"):
                foreign.add(line)  # someone else used the GPU during the run: the run is contaminated
            if mib.isdigit():  # MPS clients can report [N/A]
                proc_mib[int(pid)] = max(proc_mib.get(int(pid), 0), int(mib))
    util = [int(row["gpu"].split(", ")[0]) for row in samples if t_go and row["t"] >= t_go]
    return {"n": args.n, "k": args.k, "period": args.period, "mps": args.mps,
            **({"batch": args.batch, "cache": args.cache} if args.batch else {}),
            "slo": args.slo, "expected": expected,
            "all_meet_slo": all(entry["meets_slo"] for entry in users), "users": users,
            "proc_vram_mib_max": proc_mib, "foreign_procs": sorted(foreign), "gpu_util_mean": sum(util) / len(util) if util else None,
            "gpu_mem_mib_max": max((int(row["gpu"].split(", ")[1]) for row in samples), default=None)}


def sweep(args):
    if args.dry_run:
        args.n, args.max_updates, args.warmup, args.period = 2, 3, 1, 1.0
    args.video = str(Path(args.video).resolve())  # also the mount path inside a container
    run = Path(args.run_dir).resolve()
    if run.exists():
        sys.exit(f"{run} exists; every run gets a fresh directory")
    busy = smi("--query-compute-apps=pid,process_name,used_memory")
    if busy:
        sys.exit(f"GPU not free, refusing to measure:\n{busy}")
    run.mkdir(parents=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    # MPS sockets must fit sun_path: the daemon exits 1 once pipe/control_privileged reaches 107 bytes, which
    # run dirs under outputs/capacity/ do, so the pipe dir is a short one in /tmp (the log stays with the run)
    pipe = Path(tempfile.mkdtemp(prefix="mps-", dir="/tmp")) if args.mps else None
    if args.mps:
        env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(run / "mps/log"))
        (run / "mps/log").mkdir(parents=True)
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
    (run / "meta.json").write_text(json.dumps(provenance(args), indent=1, default=str))
    forwarded = ["--model", args.model, "--n", args.n, "--k", args.k, "--period", args.period, "--duration", args.duration,
                 "--max-updates", args.max_updates, "--warmup", args.warmup, "--video", args.video, "--barrier", run,
                 "--batch", args.batch, "--compile" if args.compile else "--no-compile",
                 "--cache" if args.cache else "--no-cache"]
    workers = 1 if args.batch else args.n
    containers = [f"capacity-{os.getpid()}-{rank}" for rank in range(workers)] if args.model == "streamsplat" else []

    def launcher(rank):
        if args.model == "anysplat":
            return [str(ENV_PY)]
        # one container per user; the same absolute paths inside, so nothing is translated
        mounts = [f"{REPO}:{REPO}:ro", f"{STREAMSPLAT}:{STREAMSPLAT}:ro", f"{TORCH_HUB}:/torch_home:ro",
                  f"{Path(args.video).resolve()}:{Path(args.video).resolve()}:ro", f"{run}:{run}",
                  *([f"{pipe}:{pipe}"] if pipe else [])]
        return ["docker", "run", "--rm", "--name", containers[rank], "--gpus", "all", "--network", "none",
                "--pid=host", "--ipc=host", "--user", f"{os.getuid()}:{os.getgid()}",
                "-e", "HOME=/tmp", "-e", "TORCH_HOME=/torch_home", "-e", f"USER={os.environ.get('USER', 'user')}",
                "-e", "CUDA_MPS_PIPE_DIRECTORY", "-e", "CUDA_MPS_LOG_DIRECTORY",
                *[part for mount in mounts for part in ("-v", mount)], STREAMSPLAT_IMAGE, "python"]

    if args.batch:  # one server for all users; it writes every user{rank}.jsonl and ready.0
        procs = [subprocess.Popen([*launcher(0), __file__, "server", "--out", str(run), *map(str, forwarded)], env=env,
                                  stdout=open(run / "server.log", "w"), stderr=subprocess.STDOUT)]
    else:
        procs = [subprocess.Popen([*launcher(rank), __file__, "user", "--rank", str(rank), "--out", str(run / f"user{rank}.jsonl"),
                                   *map(str, forwarded)], env=env, stdout=open(run / f"user{rank}.log", "w"),
                                  stderr=subprocess.STDOUT) for rank in range(args.n)]
    stop, t_go = threading.Event(), None
    threading.Thread(target=sample, args=(run / "smi.jsonl", stop), daemon=True).start()
    ready_seconds = 1800 if args.model == "streamsplat" else 900  # every streamsplat user compiles its own model
    try:
        deadline = time.time() + ready_seconds
        while not all(Path(run, f"ready.{rank}").exists() or proc.poll() is not None for rank, proc in enumerate(procs)):
            if time.time() > deadline:
                raise SystemExit(f"users never got ready within {ready_seconds} s")
            time.sleep(0.2)
        t_go = time.time() + 1.0
        Path(run, "go").write_text(repr(t_go))
        for proc in procs:
            proc.wait()
    finally:
        for name in containers:  # killing a docker client leaves its container running
            subprocess.run(["docker", "kill", name], capture_output=True)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        stop.set()
        time.sleep(0.6)
        if args.mps:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env)
            shutil.rmtree(pipe, ignore_errors=True)
    summary = summarise(run, args, procs, t_go)
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    for entry in summary["users"]:
        print(f"user{entry['rank']} rc={entry['rc']} oom={entry['oom']} upd={entry['updates']}/{entry['expected']} "
              f"lat p50/p95={entry['lat_p50']}/{entry['lat_p95']} svc p50/p95={entry['svc_p50']}/{entry['svc_p95']} "
              f"weights={entry['weight_bytes']} peak_alloc={entry['peak_alloc']} slo={entry['meets_slo']}")
    print(f"n={args.n} k={args.k} mps={args.mps} batch={args.batch} cache={args.cache} all_meet_slo={summary['all_meet_slo']} "
          f"util={summary['gpu_util_mean']} proc_vram_mib={summary['proc_vram_mib_max']}")
    if summary["foreign_procs"]:
        print(f"CONTAMINATED, other processes used the GPU: {summary['foreign_procs']}")
    if args.dry_run:  # the smallest check that fails if the harness breaks
        assert all(entry["rc"] == 0 and entry["updates"] == 3 and entry["lat_p95"] is not None
                   for entry in summary["users"]), summary["users"]
        assert summary["proc_vram_mib_max"] and summary["gpu_util_mean"] is not None and not summary["foreign_procs"], summary
        print("dry run OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["user", "server", "sweep"])
    parser.add_argument("--model", choices=["anysplat", "streamsplat"], default="anysplat")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="streamsplat only: torch.compile the predictor (its inference default)")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=8, help="frames per update")
    parser.add_argument("--period", type=float, default=1.0, help="seconds between a user's updates; 0 = back to back")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--slo", type=float, default=1.0, help="p95 latency bound in seconds")
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--batch", type=int, default=0,
                        help="streamsplat only: one server for all --n users, batching up to this many updates; 0 = a process per user")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False,
                        help="with --batch: keep each user's previous keyframe's one-frame features (G10 plan, C1)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--barrier")
    args = parser.parse_args()
    if args.period == 0 and not args.max_updates:
        parser.error("--period 0 needs --max-updates")
    if args.batch and args.model != "streamsplat":
        parser.error("--batch needs --model streamsplat")
    if args.cache and not args.batch:
        parser.error("--cache needs --batch")
    {"user": user, "server": serve, "sweep": sweep}[args.mode](args)


if __name__ == "__main__":
    main()
