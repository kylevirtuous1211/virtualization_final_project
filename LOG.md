# Experiment log

Append-only.
A later entry corrects an earlier one; entries are never edited.
Entries carry the prefix `G` (GPU sharing: many users, one GPU) and a running integer that is never reused.
The prefix `V` used in `docs/implementation_plan.md` belongs to the earlier streaming plan and is not available here.
The index of refuted claims is at the foot of this file; new entries go above it.

Lane: G1 and G2 are exploratory numbers (learning lane), approved in plan mode on 2026-11-06 (plan `misty-painting-summit`).
They are not report results.

## G1 - One AnySplat user costs 0.12-0.86 s of GPU per update and 4.6-9.4 GiB, depending on frames per update

**Question.**
What does one user cost on the RTX 6000 Ada (48 GB), as a function of k, the number of frames reconstructed per update?

**Setup.**
- AnySplat (arXiv 2505.23716) at commit 5f5e208, weights `lhjiang/anysplat@d2e8c34`, env `event_camera_2DGS/envs/anysplat` (torch 2.2.0+cu121).
- Input: `examples/video/kitti360.mp4` from the AnySplat repo (64 frames, resized and centre-cropped to 448x448).
- One process, back to back (`--period 0`), 3 warm-up updates, then 20 measured updates per k.
- Harness commit 11f8bc7, clean tree; runs started 2026-11-06T11:43-11:45Z; no other GPU process during any run.
- Service time is reconstruction only: frames to GPU, `model.inference`, `torch.cuda.synchronize`; no rendering.

**Command.**
```
for k in 2 4 8 16; do ~/Desktop/event_camera_2DGS/envs/anysplat/bin/python -B bench/capacity.py sweep --n 1 --k $k --period 0 --max-updates 20 --warmup 3 --run-dir runs/G1/k$k; done
```
The numbers below are in `runs/G1/k*/summary.json` (`svc_p50`, `svc_p95`, `peak_alloc`, `proc_vram_mib_max`, `gpu_util_mean`, `gaussians_p50`).

**Result** (n = 20 updates per k, one run each).

| k | service p50 / p95 (s) | peak allocated (GiB) | process VRAM, nvidia-smi (MiB) | GPU util mean (%) | Gaussians p50 |
|---|---|---|---|---|---|
| 2 | 0.122 / 0.123 | 3.72 | 4726 | 56 | 271,223 |
| 4 | 0.192 / 0.195 | 4.69 | 5964 | 67 | 499,157 |
| 8 | 0.395 / 0.398 | 6.63 | 8640 | 88 | 895,842 |
| 16 | 0.859 / 0.864 | 8.12 | 9576 | 97 | 1,635,866 |

- Weights and buffers held by the process: 2,944,721,308 B (2.74 GiB, 2808 MiB), the same at every k.
- The summaries mark these runs `meets_slo=False`.
  That is an artifact of `--period 0`: every update is released at t0, so `latency` is queueing by design here.
  G1 is read from service time only.
- Spread: p95 is within 2% of p50 at every k.

**Null.**
None; this is the first measurement on this line.

**Predictions for G2 (k = 8, period 1 s), written before G2 ran.**
- Memory bound: 49,140 MiB / 8,640 MiB = 5.7, so at most 5 users fit; the dry run (`runs/dryrun_1`) saw one of two processes at 10,722 MiB, so possibly 4.
- Compute bound under time-slicing: 1 s / 0.395 s = 2.5 updates per second, so capacity 2 (registered prediction 3).
- One user already keeps the GPU 88% busy at k = 8, so MPS has about 12% to recover.
  I expect MPS capacity to also be 2, which would refute registered prediction 1 (MPS serves at least one more user) at k = 8.
  At k = 2 (56% util) MPS has more room; that is not part of G2.
- Registered prediction 2 (weights dominate a process's VRAM) is already answered at N = 1:
  at k = 8 the weights are 2,808 of 8,640 MiB (33%), so the rest is 67%, above the 50% bound.
  At k = 2 the weights are 59% of the process.
  Prediction 2 is refuted at k = 8, the G2 setting, and holds at k = 2.

**What it overturns.**
Nothing; first measurement.

---

## Refuted claims

| Claim | Refuted by | Note |
|---|---|---|
| Per-user VRAM is dominated by the user's own copy of the weights (plan prediction 2) | G1 | Holds at k = 2 (59% weights), fails at k = 8 (33%) and k = 16 (29%). |
