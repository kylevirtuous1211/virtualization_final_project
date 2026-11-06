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

## G2 - The RTX 6000 Ada serves 2 AnySplat users at k = 8 and 1 update/s; MPS does not raise that, and the GPU's compute, not its memory, is the limit

**Question.**
How many users, each its own process with its own AnySplat, can share the RTX 6000 Ada while every user keeps p95 update latency within 1 s, under time-slicing and under MPS?

**Setup.**
- Same model, weights, env and input as G1; k = 8 frames per update; each user releases one update per second for 60 s after 3 warm-up updates; users' phases are evenly staggered (user r starts r/N s late).
- `latency` = finish time minus release time, so it includes waiting behind the user's own late updates; an overloaded user's queue is unbounded (no frame dropping).
- Arms: `ts`, the driver's default time-slicing, and `mps`, a user-started MPS daemon with no limits; arms interleaved per N; 2 repeats per point.
- Registered before the run: SLO p95 ≤ 1.0 s, no OOM, all 60 updates done; capacity = the largest N where every user meets it; an arm stops after two consecutive N where neither repeat passes.
- `bench/capacity.py` is byte-identical across all runs (`git diff 9e64527 b3f2bb0 -- bench/capacity.py` is empty); N=1 for both arms and N=2 `ts` ran at 9e64527, the rest at b3f2bb0, all with clean trees; no other GPU process during any kept run (`foreign_procs` empty).
- Runs 2026-11-06 11:46-14:06Z.

**Command.**
```
bench/g2.sh
```
It reuses finished runs and redoes contaminated ones; the table is printed from `runs/G2/{ts,mps}/n*/r*/summary.json` and `meta.json`.

**Result** (one row per run; per-user ranges over the N users).

| arm | N | rep | p50 latency (s) | p95 latency (s) | mean service p50 (s) | min updates / 60 | total updates/s | GPU memory max (MiB) | util (%) | SLO |
|---|---|---|---|---|---|---|---|---|---|---|
| ts | 1 | 1 | 0.393 | 0.398 | 0.392 | 60 | 1.00 | 8,680 | 36 | pass |
| ts | 1 | 2 | 0.394 | 0.398 | 0.394 | 60 | 1.00 | 9,190 | 48 | pass |
| mps | 1 | 1 | 0.400 | 0.403 | 0.399 | 60 | 1.00 | 9,219 | 44 | pass |
| mps | 1 | 2 | 0.399 | 0.402 | 0.398 | 60 | 1.00 | 9,219 | 49 | pass |
| ts | 2 | 1 | 0.396-0.400 | 0.411 | 0.397 | 60 | 2.00 | 14,985 | 96 | pass |
| ts | 2 | 2 | 0.403 | 0.409-0.411 | 0.402 | 60 | 2.00 | 12,605 | 96 | pass |
| mps | 2 | 1 | 0.400-0.402 | 0.407 | 0.400 | 60 | 2.00 | 15,010 | 98 | pass |
| mps | 2 | 2 | 0.397-0.401 | 0.405-0.406 | 0.398 | 60 | 2.00 | 15,500 | 95 | pass |
| ts | 3 | 1 | 5.39-5.48 | 10.58-10.74 | 1.207 | 50 | 2.52 | 22,494 | 98 | fail |
| ts | 3 | 2 | 5.34-5.69 | 10.54-10.74 | 1.208 | 50 | 2.52 | 21,270 | 99 | fail |
| mps | 3 | 1 | 5.64-6.36 | 11.70-12.41 | 1.248 | 49 | 2.47 | 25,991 | 99 | fail |
| mps | 3 | 2 | 5.90-6.40 | 11.63-12.32 | 1.253 | 49 | 2.45 | 25,991 | 98 | fail |
| ts | 4 | 1 | 11.65-12.18 | 23.10-23.58 | 1.614 | 38 | 2.53 | 29,895 | 98 | fail |
| ts | 4 | 2 | 11.58-12.31 | 23.31-23.72 | 1.621 | 37 | 2.52 | 29,935 | 98 | fail |
| mps | 4 | 1 | 12.65-12.89 | 24.71-25.00 | 1.680 | 36 | 2.45 | 32,272 | 98 | fail |
| mps | 4 | 2 | 12.59-12.82 | 24.61-24.73 | 1.678 | 36 | 2.45 | 29,952 | 98 | fail |

- **Capacity: 2 under `ts` and 2 under `mps`, in both repeats.**
  The sweep stopped at N = 4 by the registered rule; N = 6 and above were not run.
- The GPU delivers about 2.5 updates/s however it is shared: `ts` 2.52-2.53, `mps` 2.45-2.47, against 1 / 0.395 s = 2.53 from G1.
  Past N = 2, each user's service time stretches to about N x 0.4 s and the backlog grows every second.
- The p95 values at N ≥ 3 are not stable quantities: with an unbounded queue they grow with the run length (60 s here).
  What is stable is that demand (N updates/s) exceeds about 2.5 updates/s, so latency diverges.
- MPS against time-slicing, the matched control: about 4 ms lower p95 at N = 2 (0.405-0.407 against 0.409-0.411 s) and about 2-3% less throughput at N ≥ 3 (2.45-2.47 against 2.52-2.53 updates/s), with p95 1.0-1.7 s higher.
  n = 2 runs per point, so treat these small gaps as direction only.
- Memory was never the limit: 4 users peaked at 29.9-32.3 GB of 48 GB, with no OOM.

**Null.**
Time-slicing is the matched control for MPS (one factor differs); N = 1 is the reference for slowdown.

**Against the predictions registered in plan `misty-painting-summit` and in G1.**
- Prediction 1, MPS serves at least one more user: **refuted** at k = 8; capacity is 2 in both arms and both repeats.
- Prediction 3, time-slicing capacity ≈ floor(1 s / 0.395 s) = 2: **holds** (2).
- G1's memory bound of 4-5 users is above the compute bound of 2, as predicted, so memory does not decide capacity at this setting.

**Failed attempts, kept for the record.**
- The first sweep (an inline shell loop, 2026-11-06 ~11:46-12:05Z) counted runs that the harness refused for a busy GPU as SLO failures and so stopped at N = 4 with no N ≥ 3 data; `bench/g2.sh` (b3f2bb0) replaced it.
- Another project's job (`digital_health_blood`, on the same GPU) started during `mps` N = 2 repeat 1 twice (11:57Z and 13:27Z); the harness flagged both runs, they are kept in `runs/G2/_contaminated/` and were redone.
  Its p95 under contamination was 2.04 s, which shows how large the effect of an unnoticed co-tenant is.

**What it overturns.**
Plan prediction 1, for k = 8.
It does not test small k, where one user leaves the GPU idle (56% at k = 2 in G1) and MPS has more room; that would be a new entry.

---

## Refuted claims

| Claim | Refuted by | Note |
|---|---|---|
| Per-user VRAM is dominated by the user's own copy of the weights (plan prediction 2) | G1 | Holds at k = 2 (59% weights), fails at k = 8 (33%) and k = 16 (29%). |
| MPS serves at least one more user than time-slicing (plan prediction 1) | G2 | At k = 8 and 1 update/s both serve 2; small k untested. |
