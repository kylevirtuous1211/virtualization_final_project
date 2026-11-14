# Experiment log

Append-only.
A later entry corrects an earlier one; entries are never edited.
Entries carry the prefix `G` (GPU sharing: many users, one GPU) and a running integer that is never reused.
The prefix `V` used in `docs/implementation_plan.md` belongs to the earlier streaming plan and is not available here.
The index of refuted claims is at the foot of this file; new entries go above it.

Lane: G1 and G2 are exploratory numbers (learning lane), approved in plan mode on 2026-11-06 (plan `misty-painting-summit`).
G3 is exploratory too, approved in plan mode on 2026-11-13 (the same plan file, reused).
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

## G3 - At k = 2 and 2 updates/s the RTX 6000 Ada serves 5 AnySplat users under time-slicing and 5 under MPS; time-slicing alone already fills most of one user's idle GPU time

**Question.**
At k = 2 frames per update, where one AnySplat user leaves part of the GPU idle (G1), does MPS serve more users than time-slicing?

**Setup.**
- Lane: exploratory, like G1 and G2; plan `misty-painting-summit`, approved in plan mode on 2026-11-13.
- Runner `experiments/capacity/run_G3_k2-mps.sh` (a4261cd); its header holds the registration and was committed before any run.
- Same model, weights, env and input as G1 and G2; k = 2; each user releases one update every 0.5 s for 60 s (120 updates) after 3 warm-up updates; phases evenly staggered.
- Arms `ts` and `mps` as in G2, interleaved per N, 2 repeats per point; registered grid N in 1 2 3 4 5 6 7 8 10 12.
- Registered: SLO p95 ≤ 1.0 s, no OOM, 120/120 updates; a secondary SLO p95 ≤ 0.5 s is reported only; capacity = the largest N where both repeats pass; an arm stops after two consecutive N where neither repeat passes.
- Step 0, registered as a gate: 4 users back to back, 40 updates each, per arm; throughput = 160 / (last end - first start); a `ts` throughput ≥ 14 updates/s would have stopped the run.
- Code: step 0 `ts` ran at a4261cd (2026-11-13 23:56-23:59Z); every other kept run at 75884fd (2026-11-14 00:16-01:38Z), all with clean trees (`meta.json`).
  75884fd changes only where the MPS daemon keeps its sockets (see Failed attempts); the `ts` path is unchanged, so the header's "capacity.py unchanged since G2" holds for `ts` and not for `mps`.
- No kept run saw another GPU process (`foreign_procs` empty in all 30), so none was redone; no OOM and no crash in any run.

**Command.**
```
bash experiments/capacity/run_G3_k2-mps.sh           # resumable
bash experiments/capacity/run_G3_k2-mps.sh metrics   # re-derives outputs/capacity/G3/metrics.json
```
Every number below is in `outputs/capacity/G3/metrics.json` (`step0_updates_per_s`, `capacity`, `points.<arm>.<N>.<rep>`).

**Step 0.**
`ts` 10.06 updates/s and `mps` 10.92 updates/s, below the 14 updates/s gate, so the sweep ran at 0.5 s as registered.

**Result** (one row per run; maxima and ranges over the N users).

| arm | N | rep | p95 latency, max over users (s) | mean service p50 (s) | min updates / 120 | total updates/s | per-user VRAM min-max (MiB) | GPU memory max (MiB) | SLO 1 s | SLO 0.5 s |
|---|---|---|---|---|---|---|---|---|---|---|
| ts | 1 | 1 | 0.128 | 0.126 | 120 | 2.01 | 3,954-3,954 | 3,996 | pass | pass |
| ts | 1 | 2 | 0.134 | 0.129 | 120 | 2.01 | 3,954-3,954 | 3,994 | pass | pass |
| mps | 1 | 1 | 0.135 | 0.128 | 120 | 2.01 | 4,726-4,726 | 4,023 | pass | pass |
| mps | 1 | 2 | 0.133 | 0.130 | 120 | 2.01 | 4,726-4,726 | 4,023 | pass | pass |
| ts | 2 | 1 | 0.136 | 0.126 | 120 | 4.01 | 3,954-4,804 | 8,831 | pass | pass |
| ts | 2 | 2 | 0.142 | 0.130 | 120 | 4.01 | 3,954-3,954 | 7,953 | pass | pass |
| mps | 2 | 1 | 0.128 | 0.126 | 120 | 4.01 | 4,528-4,804 | 7,978 | pass | pass |
| mps | 2 | 2 | 0.131 | 0.127 | 120 | 4.01 | 4,008-4,804 | 9,026 | pass | pass |
| ts | 3 | 1 | 0.141 | 0.128 | 120 | 6.00 | 3,954-4,400 | 11,912 | pass | pass |
| ts | 3 | 2 | 0.138 | 0.129 | 120 | 6.00 | 3,954-4,726 | 12,684 | pass | pass |
| mps | 3 | 1 | 0.134 | 0.129 | 120 | 6.00 | 3,954-3,954 | 12,115 | pass | pass |
| mps | 3 | 2 | 0.137 | 0.129 | 120 | 6.00 | 3,954-5,002 | 11,933 | pass | pass |
| ts | 4 | 1 | 0.150 | 0.132 | 120 | 8.00 | 3,954-4,804 | 16,721 | pass | pass |
| ts | 4 | 2 | 0.147 | 0.138 | 120 | 8.00 | 3,954-3,954 | 16,643 | pass | pass |
| mps | 4 | 1 | 0.136 | 0.128 | 120 | 8.00 | 3,954-3,954 | 15,888 | pass | pass |
| mps | 4 | 2 | 0.152 | 0.131 | 120 | 8.00 | 3,954-3,958 | 16,660 | pass | pass |
| ts | 5 | 1 | 0.318 | 0.243 | 120 | 9.98 | 3,954-5,002 | 21,375 | pass | pass |
| ts | 5 | 2 | 0.330 | 0.238 | 120 | 9.98 | 3,954-4,726 | 21,453 | pass | pass |
| mps | 5 | 1 | 0.177 | 0.170 | 120 | 9.99 | 3,954-4,726 | 19,974 | pass | pass |
| mps | 5 | 2 | 0.180 | 0.171 | 120 | 9.99 | 3,954-4,726 | 21,386 | pass | pass |
| ts | 6 | 1 | 9.542 | 0.595 | 102 | 10.07 | 4,726-4,832 | 25,412 | fail | fail |
| ts | 6 | 2 | 9.581 | 0.595 | 102 | 10.07 | 4,726-4,804 | 24,640 | fail | fail |
| mps | 6 | 1 | 8.490 | 0.586 | 104 | 10.28 | 4,726-4,726 | 26,113 | fail | fail |
| mps | 6 | 2 | 8.464 | 0.584 | 104 | 10.29 | 4,726-4,726 | 25,915 | fail | fail |
| ts | 7 | 1 | 16.611 | 0.693 | 87 | 10.09 | 4,726-4,804 | 28,799 | fail | fail |
| ts | 7 | 2 | 16.603 | 0.693 | 87 | 10.09 | 4,726-4,726 | 29,293 | fail | fail |
| mps | 7 | 1 | 16.106 | 0.689 | 88 | 10.18 | 4,726-4,804 | 31,612 | fail | fail |
| mps | 7 | 2 | 16.006 | 0.687 | 88 | 10.20 | 4,726-4,804 | 31,794 | fail | fail |

- **Capacity: 5 under `ts` and 5 under `mps`, in both repeats** (the any-repeat capacity is also 5 for both).
  Both arms fail at N = 6 by latency, with no OOM; the stop rule ended the sweep after N = 7, so N = 8, 10 and 12 were not run.
- Saturated throughput is about 10.1 updates/s under both arms: `ts` 10.07-10.09 and `mps` 10.18-10.29 at N = 6-7, against 12-14 demanded.
  MPS gains 1-2% in the sweep and 8.5% in step 0, far short of the 2 updates/s a sixth user needs.
- Up to N = 4 both arms keep p95 at 0.128-0.152 s, close to the N = 1 service time (0.126-0.130 s).
- At N = 5, where demand (10 updates/s) meets the ceiling, MPS cuts p95 from 0.318-0.330 s to 0.177-0.180 s and the mean service p50 from 0.238-0.243 s to 0.170-0.171 s, in both repeats (n = 2).
  This is the only point in G1-G3 where MPS measurably helps a user, and it is a latency gain at a fixed N, not a capacity gain.
- Per-user VRAM clusters at a few values between 3,954 and 5,002 MiB under both arms, so a per-user memory difference at one N is an allocator step, not an arm effect.
  GPU memory at N = 7 was 28,799-31,794 MiB of 49,140 MiB.
- `gpu_util_mean` is not usable here: at N = 1 it reads 0.7% and 67.5% for two otherwise identical `ts` repeats and -1 (no sample) for one `mps` repeat, so it is left out of the table.

**Null.**
`ts` is the matched control for `mps` (one factor differs); N = 1 is the reference for slowdown.

**Against the predictions registered in the runner header.**
- P1, capacity(mps) ≥ capacity(ts) + 1: **refuted**.
  Both are 5 and both fail at 6 by latency, which is the registered refutation case.
- P2, `ts` ceiling 8.2 ± 0.4 updates/s: **refuted** (10.06, above the 9.0 bound).
  8.2 was 1 / 0.122 s, G1's k = 2 service time, which assumed time-slicing cannot overlap users.
  A likely reading is that one user's CPU-side gaps (G1's 44% idle) are filled by other processes' kernels without MPS; that explains the number but is not a tested mechanism.
- P3, `mps` ceiling ≥ 9.5 updates/s: holds as stated (10.92), but its premise does not.
  The threshold was set against a `ts` ceiling of 8.2; `ts` alone reaches 10.06, so MPS's margin is 0.86 updates/s.
- P4, MPS does not lower per-user memory (median within -300..+600 MiB of `ts` at N = 2 and 4): **not refuted** by the registered criterion, which needed `mps` ≥ 300 MiB lower at both N.
  Means of the two repeats' medians: N = 2 `mps` 4,536 against `ts` 4,167 MiB (+370); N = 4 `mps` 3,954 against `ts` 4,360 MiB (-406, outside the band).
  With the discrete levels above, this reads as no arm effect.
- P5, the memory wall lies between N = 8 and 10: **not tested**; the stop rule ended the sweep at N = 7.
- P6, `mps` N = 1 service p50 ≤ 0.128 s: **refuted in one of two repeats** (0.128 and 0.130 s).
  Against the matched `ts` N = 1 (0.126 and 0.129 s), MPS's single-user overhead is about 1%, inside the 5% the prediction was about; the absolute threshold came from G1 (0.122 s x 1.05), and this session's `ts` is itself 3-5% slower than G1.

**Failed attempts, kept for the record.**
- The first chain run (2026-11-13 23:56Z) gave up on step 0 `mps` after 5 attempts, because `nvidia-cuda-mps-control -d` exited 1 each time.
  Bisected on this machine, the daemon starts with a pipe directory of up to 87 characters and exits 1 at 88, that is once `<pipe dir>/control_privileged` reaches 107 bytes, consistent with the 108-byte `sun_path` of a Unix socket.
  G2's `runs/G2/...` paths fit; G3's `outputs/capacity/G3/step0/mps/mps/pipe` (96 characters) does not.
  75884fd moves the pipe directory to `/tmp/mps-*`, removed after the run, and keeps the MPS log in the run dir; a dry run under MPS from a 124-byte socket path then passed.
  The 5 attempts are in `outputs/capacity/G3/_aborted/`.

**Cost.**
30 kept runs (2 step 0, 28 sweep) and 5 aborted daemon starts; sweep wall clock 82 min plus about 4 min per step-0 arm, on the local GPU only.

**What it overturns.**
- Plan P1 now at k = 2 as well as at k = 8 (G2): in neither setting does MPS serve another user.
- Plan P2, the reasoning that chose k = 2 for G3: once 4 or more users share the GPU, time-slicing alone fills most of one user's idle time.
- Scope: evenly staggered phases, which favour time-slicing; a 0.5 s period; one RTX 6000 Ada.

---

## Refuted claims

| Claim | Refuted by | Note |
|---|---|---|
| Per-user VRAM is dominated by the user's own copy of the weights (plan prediction 2) | G1 | Holds at k = 2 (59% weights), fails at k = 8 (33%) and k = 16 (29%). |
| MPS serves at least one more user than time-slicing (plan prediction 1) | G2, G3 | Both serve 2 at k = 8 and 1 update/s (G2) and 5 at k = 2 and 2 updates/s (G3). |
| At k = 2, time-slicing tops out near one user's rate, 1 / 0.122 s = 8.2 updates/s (G3 P2) | G3 | Step 0 reached 10.06 updates/s with 4 users back to back. |
