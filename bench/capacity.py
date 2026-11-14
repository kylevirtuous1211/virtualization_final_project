#!/usr/bin/env python
"""Capacity probe (LOG.md G1, G2): N users share one GPU, each a separate process with its own AnySplat.

  user  : load the model, then every --period s reconstruct the latest --k frames; one JSONL line per update.
  sweep : refuse to run unless the GPU is free, optionally start a user-level MPS daemon, spawn --n users,
          sample nvidia-smi every 0.5 s, then write summary.json into a fresh --run-dir.

Run both modes with the AnySplat env's python (ENV_PY below); the sweep itself only needs the stdlib.
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


def user(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(ANYSPLAT))
    import torch
    out = open(args.out, "w")

    def emit(**fields):
        out.write(json.dumps({"rank": args.rank, "pid": os.getpid(), "t": time.time(), **fields}) + "\n")
        out.flush()

    try:
        from src.model.model.anysplat import AnySplat
        frames = load_frames(args.video)
        model = AnySplat.from_pretrained(HF_REPO, revision=HF_REV).cuda().eval()
        for param in model.parameters():
            param.requires_grad = False
        weight_bytes = sum(tensor.numel() * tensor.element_size()
                           for tensor in [*model.parameters(), *model.buffers()])
        windows = len(frames) - args.k + 1
        assert windows > 0, f"clip has {len(frames)} frames, fewer than k={args.k}"

        def step(update):
            first = (args.rank * 7 + update) % windows  # each user at its own place in the clip
            batch = frames[first:first + args.k].cuda().float().div_(255).unsqueeze(0)  # [1, k, 3, 448, 448] in [0, 1]
            gaussians, _ = model.inference(batch)
            torch.cuda.synchronize()
            return gaussians.means.shape[1]

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
            count = step(update)
            end = time.time()
            # latency includes waiting behind this user's own late updates; service is compute only
            emit(event="update", j=update, release=release, start=start, end=end, latency=end - release,
                 service=end - start, gaussians=count, peak_alloc=torch.cuda.max_memory_allocated(),
                 reserved=torch.cuda.memory_reserved())
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
    return {"argv": sys.argv, "args": vars(args), "commit": git(REPO, "rev-parse", "HEAD"),
            "dirty": bool(git(REPO, "status", "--porcelain")), "anysplat_commit": git(ANYSPLAT, "rev-parse", "HEAD"),
            "hf": f"{HF_REPO}@{HF_REV}", "video": str(args.video),
            "env_freeze_sha256": hashlib.sha256((ENV_PY.parent.parent / "freeze.txt").read_bytes()).hexdigest(),
            "gpu": smi("--query-gpu=name,driver_version,memory.total"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def percentile(values, q):
    """Nearest-rank percentile; None for no values."""
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def summarise(run, args, procs, t_go):
    expected = args.max_updates or math.ceil(args.duration / args.period)
    users = []
    for rank, proc in enumerate(procs):
        log = run / f"user{rank}.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        updates = [row for row in rows if row["event"] == "update"]
        ready = next((row for row in rows if row["event"] == "ready"), {})
        latency, service = [row["latency"] for row in updates], [row["service"] for row in updates]
        oom = any(row["event"] == "oom" for row in rows)
        users.append({"rank": rank, "pid": proc.pid, "rc": proc.returncode, "oom": oom,
                      "updates": len(updates), "expected": expected,
                      "lat_p50": percentile(latency, .5), "lat_p95": percentile(latency, .95),
                      "svc_p50": percentile(service, .5), "svc_p95": percentile(service, .95),
                      "gaussians_p50": percentile([row["gaussians"] for row in updates], .5),
                      "weight_bytes": ready.get("weight_bytes"),
                      "peak_alloc": max((row["peak_alloc"] for row in updates), default=None),
                      "meets_slo": proc.returncode == 0 and not oom and len(updates) >= expected
                                   and percentile(latency, .95) <= args.slo})
    samples = [json.loads(line) for line in (run / "smi.jsonl").read_text().splitlines()]
    proc_mib, foreign = {}, set()
    ours = {proc.pid for proc in procs}
    for row in samples:
        for line in row["apps"]:
            pid, rest = line.split(", ", 1)
            name, mib = rest.rsplit(", ", 1)
            if int(pid) not in ours and not name.endswith("nvidia-cuda-mps-server"):
                foreign.add(line)  # someone else used the GPU during the run: the run is contaminated
            if mib.isdigit():  # MPS clients can report [N/A]
                proc_mib[int(pid)] = max(proc_mib.get(int(pid), 0), int(mib))
    util = [int(row["gpu"].split(", ")[0]) for row in samples if t_go and row["t"] >= t_go]
    return {"n": args.n, "k": args.k, "period": args.period, "mps": args.mps, "slo": args.slo, "expected": expected,
            "all_meet_slo": all(entry["meets_slo"] for entry in users), "users": users,
            "proc_vram_mib_max": proc_mib, "foreign_procs": sorted(foreign), "gpu_util_mean": sum(util) / len(util) if util else None,
            "gpu_mem_mib_max": max((int(row["gpu"].split(", ")[1]) for row in samples), default=None)}


def sweep(args):
    if args.dry_run:
        args.n, args.max_updates, args.warmup, args.period = 2, 3, 1, 1.0
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
    forwarded = ["--n", args.n, "--k", args.k, "--period", args.period, "--duration", args.duration,
                 "--max-updates", args.max_updates, "--warmup", args.warmup, "--video", args.video, "--barrier", run]
    procs = [subprocess.Popen([str(ENV_PY), __file__, "user", "--rank", str(rank), "--out", str(run / f"user{rank}.jsonl"),
                               *map(str, forwarded)], env=env, stdout=open(run / f"user{rank}.log", "w"),
                              stderr=subprocess.STDOUT) for rank in range(args.n)]
    stop, t_go = threading.Event(), None
    threading.Thread(target=sample, args=(run / "smi.jsonl", stop), daemon=True).start()
    try:
        deadline = time.time() + 900
        while not all(Path(run, f"ready.{rank}").exists() or proc.poll() is not None for rank, proc in enumerate(procs)):
            if time.time() > deadline:
                raise SystemExit("users never got ready within 900 s")
            time.sleep(0.2)
        t_go = time.time() + 1.0
        Path(run, "go").write_text(repr(t_go))
        for proc in procs:
            proc.wait()
    finally:
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
    print(f"n={args.n} k={args.k} mps={args.mps} all_meet_slo={summary['all_meet_slo']} "
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
    parser.add_argument("mode", choices=["user", "sweep"])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=8, help="frames per update")
    parser.add_argument("--period", type=float, default=1.0, help="seconds between a user's updates; 0 = back to back")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--slo", type=float, default=1.0, help="p95 latency bound in seconds")
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--barrier")
    args = parser.parse_args()
    if args.period == 0 and not args.max_updates:
        parser.error("--period 0 needs --max-updates")
    user(args) if args.mode == "user" else sweep(args)


if __name__ == "__main__":
    main()
