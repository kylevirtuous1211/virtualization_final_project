# Experiment log

Append-only.
A later entry corrects an earlier one; entries are never edited.
Entries carry the prefix `G` (GPU sharing: many users, one GPU) and a running integer that is never reused.
The prefix `V` used in `docs/implementation_plan.md` belongs to the earlier streaming plan and is not available here.
The index of refuted claims is at the foot of this file; new entries go above it.

Lane: G1 and G2 are exploratory numbers (learning lane), approved in plan mode on 2026-11-06 (plan `misty-painting-summit`).
G3 is exploratory too, approved in plan mode on 2026-11-13 (the same plan file, reused); G4 is the reproduction attempt that gated the StreamSplat entries, and G8 (approved 2026-11-14) replaced its gate.
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

## G4 - The released StreamSplat reconstructs DAVIS `bear` keyframes at 27.51 dB by my registered metric, below its 30 dB gate, so G5-G7 were not run

**Question.**
Does the released StreamSplat reproduce on the kind of data it was trained for, before G5-G7 time it?
The first run of a borrowed method is a reproduction attempt.

**Setup.**
- Lane: reproduction attempt that gates the exploratory G5-G7; plan `misty-painting-summit`, approved in plan mode on 2026-11-13; registration in the runner header, committed before the run.
- Method: StreamSplat, ICLR 2026, arXiv 2506.08862.
  Claims borrowed: feed-forward dynamic 3D Gaussians from an uncalibrated video stream at 0.049 s per frame (Section 4.2); Table 1 reports 37.83 dB keyframe PSNR on DAVIS.
- Code DSL-Lab/StreamSplat at 5d686e9 in `third_party/StreamSplat` (untracked: the repository has no LICENSE); weight hashes in `third_party/StreamSplat/checkpoints/SHA256SUMS`; image `streamsplat:5d686e9` (`docker/streamsplat/Dockerfile`, id sha256:1d92e87c0ede...).
- Data: DAVIS 2017 trainval 480p, sequence `bear`, 82 frames.
  `bear` is in the 2017 train split, and the paper trains on train and evaluates on val, so the model may have seen this sequence.
- Depth: Depth Anything V2 ViT-L as the authors' `preprocess_depth_davis.py` makes it; 82 depth maps for 82 frames, checked before inference because `splat_inference.py` silently uses zero depth for a missing map.
- Inference: the authors' `splat_inference.py` with every default (frame gap 3, batch 8, torch.compile on), so 28 keyframes form 27 pairs, compile drops the last partial batch, and 24 pairs give 25 keyframes and 73 rendered frames.
- Registered metric: mean PSNR of each rendered keyframe (render frame 3j) against input keyframe j, both read from the two mp4 files `splat_inference.py` writes (libx264, imageio quality 8), at 512x288; pass if ≥ 30 dB, otherwise stop and do not run G5-G7.
- Run 2026-11-14 01:38-01:41Z from the worktree at 88caf08, clean tree.

**Command.**
```
bash experiments/streamsplat-repro/run_G4_davis-bear.sh           # resumable
bash experiments/streamsplat-repro/run_G4_davis-bear.sh metrics   # re-derives outputs/streamsplat-repro/G4/metrics.json
bash experiments/streamsplat-repro/run_G4_davis-bear.sh checks    # the bug checks below
```

**Result.**
- Mean keyframe PSNR **27.51 dB** over 25 keyframes (min 26.59, max 28.18); one sequence, one run.
- **Below the registered 30 dB gate, so G5-G7 were not run**, as registered.
- The per-keyframe values move smoothly, from 26.97 dB at keyframe 0 up to 28.18 at keyframe 16 and down to 26.59 at keyframe 24; no keyframe collapses.

**Bug checks, added after the gate failed** (`checks` above, 81a1c5d).
- Alignment: render offset 0 gives 27.51 dB, against 21.84 at -1 and 21.68 at +1, so keyframe j is render frame 3j as the metric assumes.
- Input-side codec loss: the input video against the JPEG keyframes, resized as `splat_inference.py` resizes them, is 40.99 dB, so encoding the input costs little.
  The render side's codec loss is not measured, because the renders are not saved uncompressed.
- Neither check finds a bug in my metric.

**Where my metric differs from the paper's** (found after the run, by reading the paper and code).
- The paper evaluates "at a resolution of 256×256" (Section 4.1), and the authors' metric (`compute_losses` in `model/splat_model.py`) bilinearly resizes both 288x512 frames to 256x256 before PSNR; mine is at 512x288.
- The authors compute PSNR on tensors; mine reads two mp4 files.
- Table 1's DAVIS numbers are on the val split; `bear` is in train.
- The registered margin from 37.83 to 30 dB allowed only for video re-encoding and missed the resolution difference.
  That is a design error in my registration, not a finding about StreamSplat.
- So the result is that StreamSplat did not reach the gate by my registered metric; whether it reproduces by the authors' protocol is untested and would be a new entry, registered before its number is computed.

**Null.**
The paper's 37.83 dB belongs to its own protocol and is not a null for this metric; the registered gate was the criterion.

**Cost.**
One run, about 3 min of GPU (82 depth maps, then inference with compile), on the local GPU only.

**What it overturns.**
The registered expectation that the released StreamSplat clears 30 dB by this metric.
Under the plan, it blocks G5-G7 until a new registration decides otherwise.

---

## G8 - By the authors' own metric the released StreamSplat reproduces its paper on DAVIS 2017 val: 37.76 dB keyframe PSNR against 37.83, 23.76 dB on the middle frames against 23.66

**Question.**
By the authors' own metric, does the released StreamSplat come near its paper's Table 1 on DAVIS 2017 val?
It replaces G4's gate for G5-G7: G4 used a metric unlike the paper's, so G4 stays failed as registered and this entry decides G5-G7 afresh.

**Setup.**
- Lane: reproduction attempt that gates the exploratory G5-G7; plan `misty-painting-summit`, revision approved in plan mode on 2026-11-14; registration in the runner header, committed before the run (7f7ea74).
- Code, weights and image as G4.
- Model: the authors' training `SplatModel` (`model/splat_model.py`) with the inference options (`configs/options_inference.py`, whose heads match the checkpoint), 6 output frames, compile off, batch 1.
  The `strict=False` load is checked on the CPU first (`keycheck`): no unexpected key, and the missing keys are only in `condition_encoder` and `lpips_loss`, which the checkpoint never holds.
- Data: DAVIS 2017 trainval 480p, the 30 val sequences (1,999 frames), and `bear` (train, 82 frames) as a control.
- Windows: every run of 6 consecutive frames i..i+5, sampled and loaded as `datasets/provider_davis.py` does when not shuffled (6 output frames give interval 1); 1,849 val windows and 77 `bear` windows.
- Depth: `davis_bear.py depth`, as in G4; depth maps equal frames for all 31 sequences, checked before evaluation.
- Metric: `SplatModel.forward`'s own `metrics['psnr_given']` (the two keyframes of a window) and `metrics['psnr_novel']` (its middle 4), both computed by the authors' `compute_losses` at 256x256 bilinear; the registered number is the mean over windows per split.
- Known differences from the authors: 480p JPEGs instead of Full-Resolution (both are resized to 288x512); no `Annotations_unsupervised` (masks enter only the losses, not PSNR); compile off; the paper does not say which windows Table 1 uses.
- Registered: gate val `psnr_given` ≥ 35.0 dB; R1 val `psnr_given` in 36-38 dB; R2 val `psnr_novel` in 22-25 dB (reported, not a gate); R3 `bear` ≥ 33 dB by this protocol, refuted below 31.
- Run 2026-11-14 06:42-07:03Z at 7f7ea74, clean tree; deterministic inference, so one run.
- G5-G7 keep the numbers their runners registered on 2026-11-13, so their entries come after this one.

**Command.**
```
bash experiments/streamsplat-repro/run_G8_davis-val-authors-metric.sh            # resumable
bash experiments/streamsplat-repro/run_G8_davis-val-authors-metric.sh keycheck   # the CPU load check
bash experiments/streamsplat-repro/run_G8_davis-val-authors-metric.sh metrics    # re-derives outputs/streamsplat-repro/G8/metrics.json
```
Per-window values are in `outputs/streamsplat-repro/G8/windows.jsonl`, per-sequence mean, min and max in `metrics.json`.

**Result.**

| split | sequences | windows | `psnr_given` (dB) | `psnr_novel` (dB) | paper Table 1 (dB) |
|---|---|---|---|---|---|
| val | 30 | 1,849 | **37.76** | **23.76** | 37.83 / 23.66 |
| `bear` (train) | 1 | 77 | 35.94 | 32.30 | - |

- **The gate passes (37.76 ≥ 35.0), so G5-G7 run with their registrations unchanged.**
- Across val sequences the per-sequence mean `psnr_given` runs from 33.96 (`cows`) to 42.40 dB (`dogs-jump`), and single windows from 33.78 to 42.82 dB.
  `psnr_novel` per sequence runs from 17.09 (`drift-straight`) to 34.61 dB (`drift-chicane`).
- The mean over windows weights long sequences more; the mean of the 30 per-sequence means is 37.98 dB (`psnr_given`) and 23.62 dB (`psnr_novel`), so the weighting moves neither number by more than 0.22 dB.

**Null.**
The paper's Table 1 numbers are the reference for its own protocol, which this entry imitates; they are not a null for my capacity data.

**Against the registered predictions.**
- Gate: **passes**.
- R1, val `psnr_given` in 36-38 dB: **holds** (37.76).
- R2, val `psnr_novel` in 22-25 dB: **holds** (23.76).
- R3, `bear` ≥ 33 dB by this protocol: **holds** (35.94).
  The same sequence gives 27.51 dB in G4, so most of G4's gap to the paper is G4's protocol, not the model.
  G4 and G8 differ in more than the metric, though (resolution, codec, keyframe spacing 3 against 5, the inference against the training model class, sequence-wide against per-frame depth normalisation), so this does not say which of those carries the 8.4 dB.
- `bear`'s middle frames (32.30 dB) are far above val's (23.76); `bear` is a training sequence with slow motion, and this entry does not separate those two.

**Cost.**
One run, 21 min wall clock (depth for 2,081 frames about 6 min, then 1,926 windows about 14.5 min), on the local GPU only; coding and the CPU key check took about 1 hour.

**What it overturns.**
Nothing measured: G4's failed gate stands as registered, and the claim it made (30 dB by its own metric) is still refuted.
It removes G4's block on G5-G7, as the revised plan registered.

---

## G5 - One StreamSplat user costs 0.19 s of GPU per keyframe and 4.0 GiB on the RTX 6000 Ada, twice the predicted time, and a single back-to-back user already runs the GPU at 5.2 updates/s

**Question.**
What does one StreamSplat user cost on the RTX 6000 Ada, with torch.compile on and off?
The StreamSplat analogue of G1.

**Setup.**
- Lane: exploratory; plan `misty-painting-summit`, approved 2026-11-13; registration in the runner header (e760d10), run after G8 cleared the gate.
- `bench/capacity.py --model streamsplat`: one user in its own container (`streamsplat:5d686e9`), back to back (`--period 0`), 3 warm-up updates, then 20 measured updates, once with compile on (the authors' default) and once off.
- One update = one new keyframe of `examples/video/kitti360.mp4` (the clip G1-G3 use) at the checkpoint's 512x288: Depth Anything V2 ViT-L depth for it, then Gaussians for the (previous, new) keyframe pair; service time is upload plus GPU work, `service_depth` the depth part; no rendering, as in G1.
- KITTI-360 is out of distribution for this model; the cost depends only on the input shape, and quality is not measured here.
- Run 2026-11-14 07:06-07:08Z at 8090b64, clean tree, `foreign_procs` empty.

**Command.**
```
bash experiments/capacity/run_G5_streamsplat-cost.sh           # resumable
bash experiments/capacity/run_G5_streamsplat-cost.sh metrics   # re-derives outputs/capacity/G5/metrics.json
```

**Result** (one run per arm, 20 updates each).

| arm | service p50 / p95 (s) | depth p50 (s) | depth share | updates/s back to back | weights (B) | peak allocated (MiB) | process VRAM, nvidia-smi (MiB) | Gaussians |
|---|---|---|---|---|---|---|---|---|
| compile | 0.190 / 0.199 | 0.049 | 26% | 5.20 | 2,423,496,008 | 2,870 | 4,094 | 73,728 |
| no compile | 0.198 / 0.204 | 0.049 | 25% | 5.02 | 2,423,496,008 | 2,834 | 3,846 | 73,728 |

- The Gaussian part is about 0.14-0.15 s and depth about 0.05 s; compile saves 4% of service time and costs 248 MiB.
- `metrics.json` also lists a "capacity" of 0 for both arms: with `--period 0` every update queues behind the previous one, so that SLO field means nothing here and is not a result.
- Bug check, a one-off probe: `forward_gaussians` alone, one pair at batch 1 with inputs already on the GPU, compile off, takes p50 0.152 s (min 0.149, max 0.163, n = 20; `probes/streamsplat_forward_timing.py`, command in its docstring).
  That matches the harness's Gaussian part, so the harness adds no measurable overhead and 0.19 s is the model's cost on this GPU.

**Null.**
Nothing to subtract; this is the first StreamSplat cost measurement here. G1's k = 8 AnySplat user (0.395 s, 8,640 MiB) is the bound the "lighter" claim is checked against.

**Against the predictions registered in the runner header.**
- Service p50 0.09 s (range 0.06-0.15): **refuted** (0.190 and 0.198 s).
  The paper's 0.049 s per frame was measured on an unnamed GPU and excludes depth; whether it is per pair at batch 1 or amortised over a batch is not tested here.
- Depth 30-40% of service: **refuted, narrowly** (26% and 25%).
- Weights 2.39 GB: **holds** (2.42 GB).
- Process VRAM about 4,200 MiB (3,500-5,000): **holds** (4,094 and 3,846 MiB).
- "Lighter than AnySplat at k = 8", refuted if service p50 > 0.3 s or VRAM > 8,000 MiB: **holds** (0.19 s against 0.395 s, 4,094 against 8,640 MiB).

**Cost.**
2 runs, 3 min wall clock, local GPU only; the probe took about 1 min.

**What it overturns.**
Nothing measured before; it doubles the per-update cost that G6 and G7's predictions assumed (0.09 s).

---

## G6 - At one keyframe per user per second the RTX 6000 Ada serves 4 StreamSplat users under time-slicing and 4 under MPS; compute, not memory, is the limit

**Question.**
How many StreamSplat users, one container each, does the RTX 6000 Ada serve at one keyframe per user per second, under time-slicing and under MPS?
The StreamSplat analogue of G2.

**Setup.**
- Lane: exploratory; registration in the runner header (e760d10), run after G8 cleared the gate.
- As G5 with compile on; each user releases one update per second for 60 s after 3 warm-ups; phases evenly staggered; arms `ts` and `mps` interleaved per N, 2 repeats; registered grid N in 1 2 4 6 8 10 12 14.
- MPS: one user-started daemon, shared by the containers through the bind-mounted pipe directory, `--ipc=host` and the same UID; each container runs with `--pid=host`, so the host's nvidia-smi sees every user's own PID.
- Registered: SLO p95 ≤ 1.0 s, no OOM, 60/60 updates; capacity and stop rule as G2.
- Run 2026-11-14 07:08-08:14Z at 8090b64, clean tree; `foreign_procs` empty in all 20 runs, none redone; no OOM, no crash.

**Command.**
```
bash experiments/capacity/run_G6_streamsplat-1hz.sh           # resumable
bash experiments/capacity/run_G6_streamsplat-1hz.sh metrics   # re-derives outputs/capacity/G6/metrics.json
```

**Result** (one row per run; maxima and ranges over the N users).

| arm | N | rep | p95 latency, max over users (s) | mean service p50 (s) | mean depth p50 (s) | min updates / 60 | total updates/s | per-user VRAM min-max (MiB) | GPU memory max (MiB) | SLO |
|---|---|---|---|---|---|---|---|---|---|---|
| ts | 1 | 1 | 0.193 | 0.180 | 0.041 | 60 | 1.01 | 4,094-4,094 | 4,134 | pass |
| ts | 1 | 2 | 0.193 | 0.182 | 0.041 | 60 | 1.01 | 4,094-4,094 | 4,134 | pass |
| mps | 1 | 1 | 0.191 | 0.179 | 0.041 | 60 | 1.01 | 4,094-4,094 | 4,163 | pass |
| mps | 1 | 2 | 0.189 | 0.181 | 0.041 | 60 | 1.01 | 4,094-4,094 | 4,163 | pass |
| ts | 2 | 1 | 0.193 | 0.182 | 0.041 | 60 | 2.01 | 4,094-4,094 | 8,233 | pass |
| ts | 2 | 2 | 0.193 | 0.182 | 0.041 | 60 | 2.01 | 4,094-4,094 | 8,233 | pass |
| mps | 2 | 1 | 0.195 | 0.182 | 0.041 | 60 | 2.01 | 4,094-4,094 | 8,258 | pass |
| mps | 2 | 2 | 0.194 | 0.183 | 0.041 | 60 | 2.01 | 4,094-4,094 | 8,258 | pass |
| ts | 4 | 1 | 0.198 | 0.184 | 0.042 | 60 | 4.00 | 4,094-4,094 | 16,432 | pass |
| ts | 4 | 2 | 0.196 | 0.184 | 0.042 | 60 | 4.00 | 4,094-4,094 | 16,432 | pass |
| mps | 4 | 1 | 0.197 | 0.182 | 0.042 | 60 | 4.00 | 4,094-4,094 | 16,448 | pass |
| mps | 4 | 2 | 0.196 | 0.182 | 0.042 | 60 | 4.00 | 4,094-4,094 | 16,448 | pass |
| ts | 6 | 1 | 13.048 | 1.271 | 0.311 | 48 | 4.73 | 4,094-4,094 | 24,630 | fail |
| ts | 6 | 2 | 12.951 | 1.269 | 0.317 | 48 | 4.73 | 4,094-4,094 | 24,630 | fail |
| mps | 6 | 1 | 13.928 | 1.296 | 0.491 | 47 | 4.65 | 4,094-4,094 | 24,637 | fail |
| mps | 6 | 2 | 13.992 | 1.293 | 0.499 | 47 | 4.65 | 4,094-4,094 | 24,637 | fail |
| ts | 8 | 1 | 25.173 | 1.699 | 0.425 | 36 | 4.72 | 4,094-4,094 | 32,831 | fail |
| ts | 8 | 2 | 25.280 | 1.699 | 0.427 | 36 | 4.71 | 4,094-4,094 | 32,831 | fail |
| mps | 8 | 1 | 25.868 | 1.732 | 0.708 | 35 | 4.62 | 4,094-4,094 | 32,828 | fail |
| mps | 8 | 2 | 25.997 | 1.733 | 0.678 | 35 | 4.63 | 4,094-4,094 | 32,828 | fail |

- **Capacity: 4 under `ts` and 4 under `mps`, in both repeats.**
  Both fail at N = 6 by latency with no OOM; the stop rule ended the sweep after N = 8, so 10, 12 and 14 were not run.
  N = 5 is not in the grid, so the edge lies between 4 and 6; the saturated throughput below makes 5 users at 1 update/s unlikely to pass, but that is an inference, not a run.
- Saturated throughput: `ts` 4.71-4.73 and `mps` 4.62-4.65 updates/s at N = 6-8, below the 5.20 one back-to-back user reaches (G5), so sharing costs 9% (`ts`) to 11% (`mps`) of the GPU's throughput.
  MPS is 1.7-1.9% below time-slicing here (n = 2).
- Below the edge, p95 stays at 0.189-0.198 s for N = 1-4 under both arms, close to the service time.
- Every user's VRAM is 4,094 MiB under both arms, and 8 users peak at 32,831 MiB of 49,140 MiB, so the memory bound is about 12 users, three times the compute bound.

**Null.**
`ts` is the matched control for `mps`; N = 1 is the reference for slowdown.

**Against the prediction registered in the runner header.**
- ts 9 and mps 10-11 users, with the limit moving from compute (G2) to memory: **refuted** by the registered criterion (either capacity ≤ 5): both are 4, and the limit is compute.
  The prediction used 0.09 s per update; G5's 0.19 s alone puts the compute bound near 5.

**Cost.**
20 runs, 66 min wall clock, local GPU only.

**What it overturns.**
The registered expectation that StreamSplat moves the capacity limit from compute to memory; on this GPU it stays compute.

---

## G7 - At three keyframes per user per second the RTX 6000 Ada serves 1 StreamSplat user under time-slicing and 1 under MPS

**Question.**
At three keyframes per user per second, close to the paper's keyframe rate, does MPS serve more StreamSplat users than time-slicing?

**Setup.**
- Lane: exploratory; registration in the runner header (e760d10), run after G8 cleared the gate.
- As G6, but a period of 1/3 s (passed as 0.3333333333333333, so the expected and released counts are both 180) and the registered grid N in 1 2 3 4 6.
- Registered: SLO p95 ≤ 1/3 s, no OOM, 180/180 updates; capacity and stop rule as G6.
- Run 2026-11-14 08:14-08:50Z at 8090b64, clean tree; `foreign_procs` empty in all 12 runs, none redone; no OOM, no crash.

**Command.**
```
bash experiments/capacity/run_G7_streamsplat-3hz.sh           # resumable
bash experiments/capacity/run_G7_streamsplat-3hz.sh metrics   # re-derives outputs/capacity/G7/metrics.json
```

**Result** (one row per run; maxima and ranges over the N users).

| arm | N | rep | p95 latency, max over users (s) | mean service p50 (s) | mean depth p50 (s) | min updates / 180 | total updates/s | per-user VRAM min-max (MiB) | GPU memory max (MiB) | SLO |
|---|---|---|---|---|---|---|---|---|---|---|
| ts | 1 | 1 | 0.194 | 0.184 | 0.041 | 180 | 3.01 | 4,094-4,094 | 4,134 | pass |
| ts | 1 | 2 | 0.194 | 0.182 | 0.041 | 180 | 3.01 | 4,094-4,094 | 4,134 | pass |
| mps | 1 | 1 | 0.191 | 0.181 | 0.041 | 180 | 3.01 | 4,094-4,094 | 4,163 | pass |
| mps | 1 | 2 | 0.195 | 0.182 | 0.041 | 180 | 3.01 | 4,094-4,094 | 4,163 | pass |
| ts | 2 | 1 | 11.448 | 0.415 | 0.098 | 146 | 4.83 | 4,094-4,094 | 8,233 | fail |
| ts | 2 | 2 | 11.650 | 0.416 | 0.110 | 145 | 4.81 | 4,094-4,094 | 8,233 | fail |
| mps | 2 | 1 | 10.312 | 0.405 | 0.128 | 149 | 4.94 | 4,094-4,094 | 8,258 | fail |
| mps | 2 | 2 | 10.241 | 0.406 | 0.128 | 149 | 4.95 | 4,094-4,094 | 8,258 | fail |
| ts | 3 | 1 | 27.553 | 0.630 | 0.167 | 96 | 4.77 | 4,094-4,094 | 12,332 | fail |
| ts | 3 | 2 | 27.596 | 0.630 | 0.165 | 96 | 4.78 | 4,094-4,094 | 12,332 | fail |
| mps | 3 | 1 | 27.316 | 0.624 | 0.216 | 97 | 4.81 | 4,094-4,094 | 12,353 | fail |
| mps | 3 | 2 | 27.212 | 0.623 | 0.203 | 97 | 4.83 | 4,094-4,094 | 12,353 | fail |

- **Capacity: 1 under `ts` and 1 under `mps`, in both repeats.**
  Both fail at N = 2 by latency with no OOM; the stop rule ended the sweep after N = 3, so 4 and 6 were not run.
- One user at 3 updates/s needs 3 x 0.18 = 0.55 of the GPU, so two need 6 updates/s against a saturated 4.77-4.95; the queue grows and p95 reaches 10-12 s within the 60 s run.
- MPS delivers 2-3% more throughput than time-slicing at N = 2 (4.94-4.95 against 4.81-4.83 updates/s) and 0.6-1.3% more at N = 3, the opposite sign to G6 (n = 2 per point); either way the difference is a few percent.

**Null.**
`ts` is the matched control for `mps`; N = 1 is the reference for slowdown.

**Against the prediction registered in the runner header.**
- ts 3 and mps 3-4 users: **refuted** by the registered criterion (mps equals ts in both repeats); both are 1.
  As in G6, the prediction used 0.09 s per update; at 0.19 s the 27% per-user GPU share it assumed is 55%, so the low-utilisation regime it meant to test does not exist at this rate.

**Cost.**
12 runs, 36 min wall clock, local GPU only.

**What it overturns.**
Plan prediction 1 (MPS serves at least one more user) is refuted again, now for StreamSplat at 3 updates/s; G2, G3, G6 and G7 all find equal capacity under the two arms.

---

## G9 - Batching StreamSplat updates across users in one server does not raise throughput: batch 8 gives 4.52 updates/s against 4.92 at batch 1, so the registered gate stopped G9 before its sweep

**Question.**
Does one StreamSplat server that runs the pending updates of several users as one batch serve more users at one keyframe per user per second than the same server run one update at a time?
Application-level multiplexing, against the GPU-level sharing of G6 and G7.

**Setup.**
- Lane: exploratory; plan `misty-painting-summit`, approved 2026-11-15 (design drafted by a Fable 5.1 Plan agent); registration in the runner header (ec728bd), committed before any batch > 1 timing.
- `bench/capacity.py sweep --model streamsplat --batch B` (ffb40ee): one container, one copy of the weights, N users in one process.
  Each step takes the pending updates (at most one per user), oldest release first, up to B, and runs them as one batch: depth for the B new keyframes in one call, then `forward_gaussians` on B (previous, new) pairs, every depth min-max per sample.
  Service time is the whole batch's time; every user in it gets the same service and its own latency.
- Correctness, before any timing (`probes/streamsplat_batch_equivalence.py`, command in its docstring, 2026-11-16): 4 users at different places in the clip, served as one batch and one at a time; the second update's Gaussians agree at full FP32 (mean absolute difference xyz 1.2e-6, rgb 4.7e-6, against 2.3e-2 and 0.37 for a mismatched user).
- Step 0, the registered gate: back to back (`--period 0`, 20 steps after warm-up), N = B = b for b in 1 2 4 8, one run each, compile off; throughput = updates / (last end - first start).
- Run 2026-11-16 20:03-20:08Z at 7de58bf, clean tree, load average 0.52 at the start; `foreign_procs` empty in all 4 runs; no OOM, no crash; every step was a full batch (batch p50 = max = b).
- Deviations from the plan file, all fixed in the runner header or the probe before the timed runs:
  - The probe's criterion changed twice, both with the user's approval on 2026-11-15/16: from max absolute difference ≤ 1e-2 to a mean 100x below a mismatched user's with < 1% of values off by > 1e-3 (a few TF32 outliers beat the first), then to full FP32 (under TF32 3.1% of rgb values sit > 1e-3 apart from rounding alone).
  - The runner checks OOM at b = 8 before the 1.2x test, the reverse of the plan's order; nothing ran out of memory, so the order did not matter.
  - Warm-up is each user's first keyframe pair, then 3 rounds of every batch size the run can form, so no batch size is first seen while timed.
- Two earlier b = 1 runs were set aside, both reported here:
  - 16:30Z: `foreign_procs` caught another project's process on the GPU (digital_health_blood M21), and the harness redid the run, as it does for every such run.
  - 16:47Z: 4.32 updates/s (service p50 0.228 s, GPU utilisation 42%) with `foreign_procs` empty; that project's next stage had started 12 s earlier at 100% of a CPU core.
    I set this run aside **after seeing its number**, on that overlap.
    Kept, it would have failed gate (1) (4.32 is 14% below 5.02); the runner then exits 3 for a harness bug, and the quiet rerun below is the check that would have followed.
    That CPU contention caused the slowdown is not tested.
  - The run above started only after that project's chain had exited and the GPU had been empty for 10 min.

**Command.**
```
bash experiments/capacity/run_G9_streamsplat-batched.sh           # resumable; stops at the step-0 gate (exit 2)
bash experiments/capacity/run_G9_streamsplat-batched.sh metrics   # re-derives outputs/capacity/G9/metrics.json
```

**Result** (step 0, one run per batch size).

| b (users = batch) | updates/s | x b = 1 | service per batch p50 (s) | depth per batch p50 (s) | Gaussian part per batch (s) | GPU utilisation mean (%) | GPU memory max (MiB) |
|---|---|---|---|---|---|---|---|
| 1 | 4.92 | 1.00 | 0.202 | 0.049 | 0.153 | 72 | 3,890 |
| 2 | 4.83 | 0.98 | 0.411 | 0.093 | 0.318 | 78 | 5,090 |
| 4 | 4.70 | 0.96 | 0.846 | 0.192 | 0.654 | 93 | 7,162 |
| 8 | 4.52 | 0.92 | 1.767 | 0.394 | 1.373 | 96 | 11,220 |

- Gate (1) passes: b = 1 gives 4.92 updates/s, 2.0% below G5's 5.02, so the server adds no overhead that matters.
- Gate (2) passes: no OOM at b = 8.
- **Gate (3) fails: b = 8 gives 0.92x the b = 1 throughput, against the 1.2x required.** As registered, G9 stopped here; the serial-vs-batched sweep was not run, and the capacities of 0 in `metrics.json` are empty fields, not results.
- A batch of b takes b times as long as one update, or longer: depth 8.0x and the Gaussian part 9.0x at b = 8, so throughput falls slightly as b grows.
- GPU utilisation rises from 72% to 96% while throughput falls; the 28% of sample time with no kernel running at b = 1 was not spare capacity that batching could use.
- Bug check, a one-off probe: `forward_gaussians` alone, random inputs already on the GPU, compile off, takes p50 0.155, 0.324, 0.676 and 1.403 s at batch 1, 2, 4 and 8 (n = 20 each; `probes/streamsplat_forward_timing.py 1 2 4 8`, command in its docstring, 2026-11-16 20:09Z).
  That is 9.1x at batch 8, matching the server's Gaussian part (1.373 s), so the scaling is the model's own on this GPU, not the server's.
  Why the per-sample cost rises with batch is not tested.
- Memory grows by about 1,000-1,200 MiB per sample in the batch (3,890 MiB at b = 1, 11,220 MiB at b = 8).

**Null.**
b = 1 in the same server is the matched control for every b > 1 (only the batch size differs); G5's 5.02 updates/s is the reference that gate (1) checks the server against.

**Against the predictions registered in the runner header.**
- P1, b = 8 at 2.5x b = 1 (10-17 updates/s), rising with b; refuted if < 2.0x: **refuted** (0.92x, 4.52 updates/s, falling with b).
  Its reasoning, that G5's 64-73% utilisation at batch 1 left compute idle, does not hold: utilisation reached 96% and throughput did not rise.
- P2-P7: **not tested**; the gate stopped G9 before the sweep.
- P8, server memory 6,000-10,000 MiB at every N of the sweep: **not tested** as registered (no sweep).
  Its assumed 520 MiB of activations per sample is about half of what step 0 shows, and b = 8 alone reaches 11,220 MiB, above that range.

**Cost.**
4 runs, 5 min wall clock for step 0, plus the 2 set-aside b = 1 runs (16:29-16:47Z, about 18 min including waits), a 2 min dry run (2026-11-15), the correctness probe (about 2 min per attempt, 3 attempts) and the 2 min timing probe; local GPU only.
Waiting for other projects to leave the GPU took longer than the measurements (2026-11-15 13:01-13:29Z, 2026-11-16 10:42-16:29Z and 17:01-20:02Z).

**What it overturns.**
The plan's premise that cross-user batching could raise StreamSplat throughput on this GPU: at 288x512 on the RTX 6000 Ada the model gains nothing from batches of 2-8.
With G6 and G7, neither GPU-level sharing (time-slicing, MPS) nor application-level batching raises how many StreamSplat users this GPU serves; all are bounded near 5 updates/s.
That one server holding one copy of the weights for N users would save memory is not tested here; memory was not the limit in G6.

---

## G10 - A StreamSplat update spends 21% of its model time re-encoding the previous keyframe, and bf16 runs the model 2.4x faster; stage batching and a second CUDA stream gain nothing

**Question.**
Where does one StreamSplat update spend its GPU time on the RTX 6000 Ada, and which of the planned components is worth building: C1 a per-user cache of the previous keyframe's one-frame work (the paper's Algorithm 2), C2 bf16 autocast, C3 no host syncs, C4 CUDA graphs, C5 stage-level batching across users, C6 two CUDA streams with priority?
Step 0 of the plan that follows G9's negative result.

**Setup.**
- Lane: exploratory; plan `misty-painting-summit`, approved 2026-11-18 (drafted by two Fable 5.1 Plan agents, one for model efficiency and one for a parallel staged server); registration in the runner header (0bdbd3c).
- The code reading behind the stages is in `docs/reading_notes.md`, "StreamSplat released implementation" (15ebbfd).
- `bench/streamsplat_stages.py` splits `SplatPredictor.forward` into a one-frame stage (static encoder, DINO condition, static upsampler, static head, `encoder_proj`) and a pair stage (decoder, dynamic upsampler, dynamic head) by calling the predictor's submodules in forward's order; `split_forward` recombines them with forward's tensor shapes.
- `bench/streamsplat_profile.py`, one container, compile off, pairs from the clip as the batching server builds them (user b: frames 7b and 7b + 1):
  - stages: `forward_gaussians` end to end; `split_forward` per stage (CUDA events); the one-frame stage on the B new frames alone (what a cache would still compute); the pair stage alone; the depth stage (Depth Anything V2 plus min-max, never under autocast); activation peaks; TF32 and bf16 autocast at batch 1, 2, 4, 8; FP32 "highest" at batch 1.
  - profile: `torch.profiler` over one forward; busy share = kernel time over the unprofiled wall time.
  - ceiling: matmul and `scaled_dot_product_attention` with each backend forced.
  - overlap: the depth stage and the pair stage at batch 1 from two threads on two streams, work sized so both take about 1.7 s alone, with and without high priority for the depth stream.
  - 3 warm-ups, then 20 timed calls per point (50 for matmuls), each synchronised.
- Run 2026-11-18 16:40-16:47Z at 0bdbd3c, clean tree; the GPU sampler saw only the profiler's own process in all 722 samples; one attempt, nothing redone.
- A smoke run (2 calls per point, `outputs/scratch/G10-smoke`) checked the code after the plan's approval and before the runner was committed; its numbers are not used.

**Command.**
```
bash experiments/capacity/run_G10_streamsplat-stage-profile.sh           # one container, about 7 min
bash experiments/capacity/run_G10_streamsplat-stage-profile.sh metrics   # re-derives outputs/capacity/G10/run/metrics.json
```

**Result.**

`forward_gaussians` p50 (s; n = 20, min-max within 3% of p50 at every point):

| precision | b = 1 | b = 2 | b = 4 | b = 8 | b = 8 over b = 1 |
|---|---|---|---|---|---|
| TF32 (as served so far) | 0.158 | 0.326 | 0.683 | 1.428 | 9.03 |
| bf16 autocast | 0.066 | 0.119 | 0.240 | 0.500 | 7.58 |
| FP32 "highest" | 0.248 | | | | |

Stages of a cached update at batch 1 (ms, p50), and TF32 per-sample time at batch 4 over batch 1:

| stage | TF32 | bf16 | TF32 per sample, b = 4 over b = 1 |
|---|---|---|---|
| static encoder | 14.2 | 5.3 | 1.11 |
| DINO condition | 3.0 | 5.9 | 0.93 |
| static upsampler | 12.0 | 4.0 | 1.46 |
| static head | 4.4 | 9.4 | 0.44 |
| `encoder_proj` | 0.1 | 0.1 | 2.03 |
| one-frame stage, wall | 34.0 | 25.0 | |
| decoder | 53.4 | 14.8 | 1.09 |
| dynamic upsampler (both frames) | 28.5 | 9.4 | 1.25 |
| dynamic head (both frames) | 1.8 | 1.7 | 1.16 |
| pair stage, wall | 84.1 | 26.0 | |
| depth stage (no autocast in either) | 55.1 | 57.0 | 1.06 |

- **The split reproduces forward**: `split_forward` is bitwise identical to `forward_gaussians` at all 8 batch and precision points, and its stage times sum to 0.90-1.01 of forward's p50.
- **The cacheable work is 21%**: the one-frame stage of the frame a cache would keep takes 34.0 ms of the 158 ms forward at batch 1 (21.5%).
  Summing the stages, a cached update would cost about 55 + 34 + 84 = 173 ms against 213 ms today in TF32, and 57 + 25 + 26 = 108 ms with bf16 on the model; these are sums of stage timings, not server runs, and C1 and C2 measure them.
- **bf16 is 2.4x on the model** (0.066 against 0.158 s at batch 1).
  Attention explains most of it: in FP32 `scaled_dot_product_attention` has no flash kernel and runs the memory-efficient CUTLASS kernel at 13-19 TFLOP/s at the encoder, decoder and upsampler shapes, against 142-173 TFLOP/s for the flash kernel in bf16; attention takes 67% of kernel time at batch 1 in TF32 and 25% in bf16.
  In bf16 the static head and DINO got slower than in TF32 (9.4 against 4.4 ms, 5.9 against 3.0 ms); both hold host synchronisations, and why they slow down was not tested.
- **The GPU is busy 86% of a batch-1 forward in TF32 and 78% in bf16** (100% and 97% at batch 8).
  One forward makes 14 stream synchronisations: 8 device-to-host reads from the static head's `isnan` asserts and 6 pageable host-to-device copies of tensors built on the CPU; plus one device synchronise; 1,315 kernel launches in TF32, 1,599 in bf16.
- **Matmul ceilings on this card**, 8192^3: FP32 28.8, TF32 65.8, bf16 103.3 TFLOP/s; at the model's own GEMM shapes TF32 reaches 38-64 and bf16 86-101.
  Against the analytical 4.11 TFLOP per pair, batch 1 achieves 26 TFLOP/s, 39.5% of the TF32 ceiling.
- **Where the superlinear batch cost sits**: per sample, the two upsamplers cost 1.46x and 1.25x more at batch 4 than at batch 1, the encoder 1.11x, the decoder 1.09x; G9's 9.1x at batch 8 comes mostly from the upsamplers.
  The mechanism (the upsampler's windowed attention and `torch.roll` copies spilling the 96 MB L2 at larger batch was the hypothesis) was not tested.
- **Stage batching saves 2.9 ms per update at batch 4** by the C5 rule (only DINO at 2 and the static head at 8 qualify).
- **Two streams overlap nothing**: running the depth stage and the pair stage together took as long as running them one after the other (gain -0.1% without priority, -0.6% with the depth stream at high priority).
  Under load the depth stage's latency rose from 0.051 s to 0.062 s (p50) without priority and to 0.140 s with it; n = 32 calls, one run, not explained.
- **Activation memory** grows 381 MiB per extra sample in TF32 and 331 MiB in bf16 (peak allocated above the inputs, b = 8 against b = 1), peaking in the upsamplers; G9's 1,000-1,200 MiB per sample from nvidia-smi is mostly allocator reserve, not live tensors.

**Null.**
`forward_gaussians` at the same batch and precision is the reference every stage is checked and timed against; there is no arm to subtract, since this entry changes nothing in serving.

**Against the predictions registered in the runner header.**
- Gate (bitwise split, stage sum within 10%): **holds**.
- P1 cacheable share 20-30%: **holds** (21.5%).
- P2 bf16 at 0.55-0.75x TF32, refuted if > 0.87x: **not refuted, outside the predicted range on the fast side** (0.42x).
- P3 busy share 75-90% and >= 14 host waits, refuted if busy > 95%: **holds** (86%; 14 stream synchronisations, as predicted, plus one device synchronise).
- P4 memory-efficient kernel in FP32 and flash in bf16, attention 40-55% of TF32 kernel time, refuted if < 30%: **kernels as predicted; share above the range** (67%).
- P5 matmul ceilings TF32 70-95 and bf16 140-190 TFLOP/s, batch 1 at 28-38% of the TF32 ceiling, refuted if TF32 > 120: **not refuted; both ceilings below their ranges** (65.8 and 103.3), achieved share just above (39.5%).
- P6 DINO and both heads <= 0.6x per sample at batch 4, encoder, decoder, upsamplers and depth 0.9-1.15x, stage batching saving 5-15 ms: **refuted** (DINO 0.93x, dynamic head 1.16x, upsamplers 1.46x and 1.25x; only the static head at 0.44x; saving 2.9 ms).
- P7 two-stream overlap 5-15%: **refuted** (-0.1% and -0.6%).
- P8 650-750 MiB per extra sample, peaking in an upsampler or a head; bf16 330-400: **refuted for TF32** (381 MiB), holds for bf16 (331) and for the peak stage (the upsamplers).
- Decisions by the registered rules: C1 cache (21.5% >= 12%), C2 bf16 (0.42x <= 0.87x), C3 no syncs and C4 CUDA graphs (busy 86% < 90%) go ahead; C5 stage batching (2.9 ms < 10 ms) and C6 two streams (-0.6% < 5%) are dropped; the stop-all condition does not hold.

**Cost.**
One container, 7 min wall clock (16:40-16:47Z), local GPU only; the smoke run took about 5 min.

**What it overturns.**
G9 left open why per-sample cost rises with batch; G10 locates it mainly in the two upsamplers (mechanism untested).
The parallel-pipeline half of the plan loses its two scheduling components: on this GPU neither stage batching nor a second stream adds throughput, so what remains of it is the cache.
G9's reading that 1.0-1.2 GB of memory comes with each extra sample is corrected to 381 MiB of live activations.

---

## G11 - Keeping each user's previous keyframe features (the paper's Algorithm 2) raises StreamSplat throughput 1.27x at batch 1 and 1.21x at batch 4, for about 37 MiB of live memory per user

**Question.**
Does keeping each user's previous keyframe's one-frame features in the batching server, so an update encodes only the new keyframe, make an update cheaper, and what VRAM does it cost?
C1 of plan `misty-painting-summit`, built because G10 found that work to be 21.5% of the model's time (rule: >= 12%).

**Setup.**
- Lane: exploratory; registration in the runner header (00d7dce), written in plan mode before any cache timing.
- `bench/capacity.py` `--cache` (9443b83): the server keeps, per user, the previous keyframe's `encode_frames` output (`bench/streamsplat_stages.py`: encoder, DINO condition, static upsampler, static head, `encoder_proj`); an update runs depth and `encode_frames` on the B new keyframes only, then `pair_stage` on the cached and new features.
  The new keyframe's depth is normalised per frame, which equals the server's pair min-max only when every frame's disparity spans 0..255; the probe checks that premise.
- Equivalence probe first (`probes/streamsplat_cache_equivalence.py`, FP32 "highest"): 4 users, 3 updates, the server with and without `--cache`, G9's criterion on `pred_gs` xyz and rgb.
- Timing as G9's step 0: one server, N = batch = b users back to back (`--period 0 --max-updates 20 --warmup 3`), compile off, TF32, b in 1 4; arm `cache` adds `--cache`, arm `nocache` is the matched control (same server, only the cache differs); arms interleaved, 2 repeats each (n = 2 per cell).
- Run 2026-11-18: probe 16:50-16:52Z, the 8 timed runs 22:45-22:54Z, at 00d7dce; the tree was dirty only by an uncommitted edit to `experiments/streamsplat-repro/davis_eval.py`, which the server does not import, and by untracked docs.
  `foreign_procs` is empty in all 8 runs; nothing was redone.

**Command.**
```
bash experiments/capacity/run_G11_streamsplat-cache.sh           # probe, then 8 runs; resumable
bash experiments/capacity/run_G11_streamsplat-cache.sh metrics   # re-derives outputs/capacity/G11/metrics.json and the ratios
python3 -c 'import json,glob; [print(f, *(round(u[k],4) for k in ("svc_p50","svc_depth_p50")), u["peak_alloc"]>>20) for f in sorted(glob.glob("outputs/capacity/G11/*/n*/r*/summary.json")) for u in json.load(open(f))["users"][:1]]'   # service, depth, peak MiB per run
```

**Result.**

- **The cache gives the same Gaussians**: `CACHE_EQUIVALENCE_OK`; all 64 clip frames span exactly 0..255; at every update the mean absolute difference is <= 4.0e-8 in xyz (a mismatched user: 2.0-2.6e-2) and <= 4.2e-8 in rgb (0.37-0.38), and no value differs by more than 1e-3.

| b (users = batch) | arm | updates/s (r1, r2) | x nocache | service per batch p50 (s) | depth per batch p50 (s) | Gaussian part per batch (s) | peak allocated (MiB) | GPU memory max (MiB) |
|---|---|---|---|---|---|---|---|---|
| 1 | nocache | 4.65, 4.71 | 1.00 | 0.214 | 0.056 | 0.158 | 2,840 | 3,890 |
| 1 | cache | 5.94, 5.92 | 1.27 | 0.169 | 0.055 | 0.114 | 2,877 | 3,890 |
| 4 | nocache | 4.54, 4.49 | 1.00 | 0.883 | 0.223 | 0.661 | 5,021 | 7,162 |
| 4 | cache | 5.48, 5.42 | 1.21 | 0.734 | 0.226 | 0.508 | 5,164 | 7,174 |

(service, depth and Gaussian part are means of the two repeats' p50; Gaussian part = service minus depth.)

- **At batch 1 the cache cuts the Gaussian part to 0.72x** (0.114 against 0.158 s) and the update by 45 ms; throughput rises from 4.68 to 5.93 updates/s (means, x1.268).
  G10's stage sum predicted 173 ms per cached update against 213 ms; measured 169 against 214.
  The saving is 10 ms larger than G10's 34.0 ms one-frame stage; where the rest comes from was not tested.
- **At batch 4 the gain is smaller**, 1.207x (4.52 to 5.45 updates/s); the Gaussian part falls to 0.77x, 38 ms per update against 44 ms at batch 1.
  Why was not tested.
- **Memory**: the registered metric, whole-GPU memory from nvidia-smi, grows by 0 MiB at batch 1 and 12 MiB at batch 4, because the allocator's existing reserve absorbs the cache.
  Live tensors (`peak_alloc`) grow by 37 MiB at batch 1 and 143 MiB at batch 4, about 36 MiB per user, as estimated.
- Depth is unchanged by the cache (0.055 against 0.056 s at batch 1), as it should be.
- GPU utilisation means are 57-77% at batch 1 and 86-92% at batch 4 in both arms; with about 4 s of timed work per run the spread within an arm is larger than between arms, so they say nothing here.
- **The control drifted**: the same `step` code as G9 gives 4.68 updates/s at batch 1 against G9's 4.92 (2026-11-16), and 4.52 against 4.70 at batch 4; service 0.214 against 0.202 s at batch 1, depth 0.056 against 0.049 and the Gaussian part 0.158 against 0.153.
  G10's `forward_gaussians` on the same day (0.158 s) agrees with today's control, so the GPU ran this work about 5% slower on 2026-11-18; the cause was not tested.
  The ratios above are against the interleaved control and do not depend on it; absolute numbers compared across days do.

**Null.**
The matched control `nocache`: the same server, run interleaved with `cache`, differing only in `--cache`.

**Against the predictions registered in the runner header.**
- P1 at b = 1 from 4.92 to 5.6-6.2 updates/s, refuted if < 5.3: **holds** (5.93; x1.268 against the matched control, x1.205 against G9's 4.92).
- P2 batch 4's gain within 5 percentage points of batch 1's, refuted if they differ by > 10: **not refuted, outside the predicted band** (26.8% and 20.7%, 6.1 points apart).
- P3 `gpu_mem_mib_max` at b = 4 grows <= 250 MiB, refuted if > 500: **holds** (+12 MiB; +143 MiB of live tensors).
- Decision by the plan's rule: the cache stays in the final configuration; C2 (bf16) runs with `--cache` in both arms.

**Cost.**
About 2 min of GPU for the probe and 9 min for the runs; 6 h 5 min wall clock (16:49-22:54Z), almost all of it waiting for other projects to leave the GPU.

**What it overturns.**
Nothing is refuted.
It replaces G10's stage-sum estimate of a cached update (173 ms) with a server measurement (169 ms).
It shows that G9's 4.92 updates/s is not a stable reference across days: the same code gives 4.68 two days later, so the plan's final gate, written as an absolute `fast` b = 1 >= 6.0 updates/s (G9's 4.92 x 1.22), is now harder to pass by about 5%; changing that gate needs a new approval.

---

## G12 - With the cache, bf16 autocast on the model runs StreamSplat at 10.8 updates/s at batch 1 (1.77x TF32) and 11.0 at batch 4 (1.95x), for 0.18 dB of keyframe PSNR on DAVIS val

**Question.**
With each user's previous keyframe cached (G11), what does running the StreamSplat model under bf16 autocast do to an update's time and memory, and what does it cost in quality?
C2 of plan `misty-painting-summit`, built because G10 found bf16 to run the model at 0.42x TF32 (rule: <= 0.87x).

**Setup.**
- Lane: exploratory; registration in the runner header (b51a196), with the predictions approved in the plan before G10 ran and kept unchanged; G10's 0.42x and a dry run (`outputs/scratch/g12-dry`, batch 2) were seen before the header was written, and the header says so.
- `bench/capacity.py` `--bf16` (e664b77) wraps the StreamSplat model calls (`encode_frames` and `pair_stage` with the cache, `forward_gaussians` without) in `torch.autocast("cuda", dtype=torch.bfloat16)`; the depth model never runs under it, and the heads the authors pin to fp32 stay fp32.
- Quality: `experiments/streamsplat-repro/davis_eval.py eval --bf16` (1d559f2) on the 30 DAVIS 2017 val sequences, all 1,849 six-frame windows, G8's depth maps reused read-only, the authors' `psnr_given` (the two keyframes) and `psnr_novel` (the middle 4); `model/splat_model.py` keeps the renderer and the losses out of autocast.
  The control is G8's run (the same code, windows and depth without autocast; inference is deterministic, so it was not rerun), compared per window.
  This evaluates `SplatModel.forward` in bf16, not the cached server path; the cache's equivalence was checked in FP32 only (G11).
- Timing as G11: one server, N = batch = b users back to back (`--period 0 --max-updates 20 --warmup 3`), compile off, TF32 matmuls, `--cache` in both arms, b in 1 4; arm `bf16` adds `--bf16`, arm `tf32` is the matched control; arms interleaved, 2 repeats each (n = 2 per cell).
- Run 2026-11-18 at b51a196, the tree dirty only by untracked files: the evaluation 22:59-23:15Z, the timed runs 23:15-23:27Z.
  A co-tenant (`digital_health_blood`) appeared during the first `tf32` run at b = 1 and at b = 4; both were moved to `_redo` and redone; `foreign_procs` is empty in all 8 kept runs.

**Command.**
```
bash experiments/capacity/run_G12_streamsplat-bf16.sh           # the bf16 DAVIS evaluation, then 8 runs; resumable
bash experiments/capacity/run_G12_streamsplat-bf16.sh metrics   # re-derives outputs/capacity/G12/davis/metrics.json (with the per-window change against G8) and outputs/capacity/G12/metrics.json, and prints the ratios
python3 -c 'import json,glob; [print(f, *(round(u[k],4) for k in ("svc_p50","svc_depth_p50")), u["peak_alloc"]>>20) for f in sorted(glob.glob("outputs/capacity/G12/*/n*/r*/summary.json")) for u in json.load(open(f))["users"][:1]]'   # service, depth, peak MiB per run
```

**Result.**

Quality on DAVIS 2017 val (1,849 windows, mean dB):

| metric | G8 (no autocast) | bf16 | change | per-window change, min to max | windows dropping > 0.5 dB |
|---|---|---|---|---|---|
| `psnr_given` (keyframes) | 37.757 | 37.581 | -0.176 | -0.741 to +0.198 | 4.3% |
| `psnr_novel` (middle frames) | 23.761 | 23.750 | -0.011 | -0.520 to +0.446 | 0.1% |

Timing, both arms with the cache:

| b (users = batch) | arm | updates/s (r1, r2) | x tf32 | service per batch p50 (s) | depth per batch p50 (s) | Gaussian part per batch (s) | peak allocated (MiB) | GPU memory max (MiB) |
|---|---|---|---|---|---|---|---|---|
| 1 | tf32 | 6.10, 6.15 | 1.00 | 0.162 | 0.049 | 0.113 | 2,877 | 3,890 |
| 1 | bf16 | 10.73, 10.93 | 1.77 | 0.090 | 0.045 | 0.046 | 2,929 | 3,906 |
| 4 | tf32 | 5.66, 5.65 | 1.00 | 0.705 | 0.192 | 0.514 | 5,164 | 7,174 |
| 4 | bf16 | 11.03, 10.98 | 1.95 | 0.358 | 0.189 | 0.169 | 5,109 | 7,190 |

(service, depth and Gaussian part are means of the two repeats' p50; Gaussian part = service minus depth.)

- **bf16 cuts the Gaussian part to 0.40x at batch 1** (46 against 113 ms) and to 0.33x at batch 4; throughput rises 1.77x and 1.95x.
  G10's stage sum for a cached bf16 update was 108 ms; measured 90 ms.
- **With bf16, batch 4 no longer costs more per user than batch 1**: 42 ms of Gaussian part per user at batch 4 against 46 ms at batch 1 (0.93x), where TF32 pays 1.13x (128 against 113 ms); throughput at batch 4 is 1.02x batch 1.
  Why the superlinear batch cost of TF32 (G9, G10) disappears in bf16 was not tested.
- **Depth is now half of an update**: 45 of 90 ms at batch 1 and 47 of 89 ms per user at batch 4, against 30% and 27% in the `tf32` arm.
  Depth also measured 9% faster in the `bf16` arm at batch 1 (0.045 against 0.049 s), though it runs identically in both; not explained.
- **Memory hardly moves**: whole-GPU memory +16 MiB at both batch sizes; live tensors +52 MiB at batch 1 and -55 MiB at batch 4.
- **Quality**: the keyframe PSNR drops 0.18 dB on average and by more than 0.5 dB in 4.3% of windows (worst -0.74); the novel-view PSNR is unchanged on average (-0.01).
- **G11's drift reading does not hold as a day-level change**: the `tf32` arm here is G11's `cache` arm, the same code and flags, and 30 minutes later it ran 3.4% faster (6.13 against 5.93 updates/s at batch 1, 5.65 against 5.45 at batch 4), with depth at 0.049 s as in G9 rather than G11's 0.055.
  This GPU's speed for the same work varies by about 5% within one evening; the cause (co-tenant CPU load, clocks after a co-tenant's GPU work) was not tested, and cross-run absolute numbers carry it.

**Null.**
Timing: the matched control `tf32`, the same server with the cache, interleaved, differing only in `--bf16`.
Quality: G8's per-window PSNR from the same code, windows and depth without autocast.

**Against the predictions registered in the runner header.**
- P1 the Gaussian part at b = 1 at 0.55-0.75x the control: **refuted, faster than predicted** (0.40x).
- P2 7.0-8.5 updates/s at b = 1, refuted if < 1.15x the control: **not refuted, outside the predicted range on the fast side** (10.83, 1.77x).
- P3 `psnr_given` and `psnr_novel` each drop by 0-0.15 dB: **refuted for `psnr_given`** (-0.18 dB), **holds for `psnr_novel`** (-0.01 dB).
- Quality gate, either mean drop > 0.5 dB keeps bf16 out: **passes** (0.18 and 0.01 dB); bf16 stays in the final configuration, so C3 and later run with `--cache --bf16`.

**Cost.**
28 min wall clock (22:59-23:27Z): 15 min for the DAVIS evaluation, 12 min for 10 timed runs (2 redone after a co-tenant appeared), the rest waiting for the GPU.

**What it overturns.**
G9's and G10's superlinear per-user cost of batching holds for TF32 only; under bf16 a batch of 4 costs no more per user than a batch of 1 (mechanism untested).
G11's statement that the GPU ran about 5% slower on 2026-11-18 than on 2026-11-16 is corrected: the same configuration ran 3.4% faster later the same evening, so the variation is within a day, cause untested.
After bf16, the depth model, not StreamSplat, is the largest part of an update.

---

## G13 - Removing the host waits from StreamSplat's model calls (8 per cached update to 0) does not make an update measurably cheaper: 0.98x at batch 1, 1.02x at batch 4

**Question.**
With the cache (G11) and bf16 (G12), does removing the host waits in the model calls make an update cheaper?
C3 of plan `misty-painting-summit`, built because G10 found the GPU busy only 86% (TF32) and 78% (bf16) of a batch-1 forward (rule: < 90%); it is also C4's prerequisite, since a CUDA graph cannot be captured across a host wait.

**Setup.**
- Lane: exploratory; registration in the runner header (c2ddf3f), predictions as approved in the plan.
- `bench/capacity.py` `--nosync` (2cbd514): the server runs under `python -O`, which drops the static head's `isnan` asserts (each a device-to-host read), and the model calls run in `torch.device("cuda")`, so the tensors the model builds with factory functions start on the GPU instead of being copied there from pageable memory.
  The plan said to wrap the step; only the model calls are wrapped, since the depth stage builds nothing on the CPU.
  The server records `python_optimize` in its ready lines, and our two load checks became `raise`, which `-O` keeps.
- `bench/streamsplat_sync_count.py` (8459ca1) counts the CUDA runtime calls `torch.profiler` records in one cached bf16 update at batch 1 (`encode_frames` on the new keyframe, then `pair_stage`), once per arm, and checks that the default-device context leaves the output bitwise unchanged.
- Timing as G11 and G12: one server, N = batch = b users back to back (`--period 0 --max-updates 20 --warmup 3`), compile off, `--cache --bf16` in both arms, b in 1 4; arm `nosync` adds `--nosync`, arm `sync` is the matched control; arms interleaved, 2 repeats each (n = 2 per cell).
- A smoke run of the counter and two dry runs (`outputs/scratch/g13-smoke`, `g13-dry`, `g13-dry-sync`) checked the code before the runner was committed; the smoke run showed the same counts as below.
- Run 2026-11-18 23:33-23:42Z at c2ddf3f, the tree dirty only by untracked files; `foreign_procs` is empty in all 8 runs, nothing was redone.
  The runner's report crashed on a glob that also matched the `r*.out` files; fixed (2149f33, report code only) and re-derived with the `metrics` command below.

**Command.**
```
bash experiments/capacity/run_G13_streamsplat-nosync.sh           # sync counts, then 8 runs; resumable
bash experiments/capacity/run_G13_streamsplat-nosync.sh metrics   # re-derives outputs/capacity/G13/metrics.json (with the sync counts and the servers' -O record) and prints the ratios
```

**Result.**

| | `sync` (control) | `nosync` |
|---|---|---|
| `cudaStreamSynchronize` in one cached update | 8 | 0 |
| `cudaMemcpyAsync` in one cached update | 25 | 17 |
| `cudaDeviceSynchronize` (one is the counter's own) | 2 | 2 |
| output bitwise equal to the update without the device context | yes | yes |
| server ran under `python -O` (ready lines of all kept runs) | no | yes |

| b (users = batch) | arm | updates/s (r1, r2) | x sync | service per batch p50 (s) | depth per batch p50 (s) |
|---|---|---|---|---|---|
| 1 | sync | 11.11, 10.89 | 1.00 | 0.089 | 0.044 |
| 1 | nosync | 10.62, 10.84 | 0.98 | 0.092 | 0.045 |
| 4 | sync | 11.17, 10.91 | 1.00 | 0.356 | 0.188 |
| 4 | nosync | 11.19, 11.22 | 1.02 | 0.353 | 0.188 |

- **The waits are gone**: all 8 stream synchronisations of a cached update disappear, and the output is bitwise unchanged.
- **Throughput does not rise**: -2.4% at batch 1, where both `nosync` runs are below both control runs (whose own spread is 2.0%), and +1.5% at batch 4, within the control's spread (2.4%); n = 2 per cell.
  So in this server the GPU's idle time is not spent waiting on these 8 reads and copies; where it is spent was not measured here.
  Why batch 1 got slightly slower was not tested; one candidate is that the `torch.device` context intercepts every torch call in Python, which costs most where the GPU waits on the CPU.
- Memory is unchanged (peak allocated 2,927-2,929 MiB at batch 1 and 5,109 MiB at batch 4 in both arms).
- The `sync` arm here is G12's `bf16` arm, the same code and flags; it ran at 11.00 and 11.04 updates/s against G12's 10.83 and 11.00.

**Null.**
The matched control `sync`: the same server with `--cache --bf16`, interleaved, differing only in `--nosync`.

**Against the predictions registered in the runner header.**
- P1 the host waits in an update's model calls drop to 0, refuted if > 2 remain: **holds** (8 to 0).
- P2 back-to-back throughput rises 1-5%: **refuted** (-2.4% at batch 1; +1.5% at batch 4, within the control's own spread).
- Decision by the plan's rule: C3 stays in the configuration as C4's prerequisite, though it gains nothing by itself; C4's condition (C3 brings the waits to zero) holds, so C4 runs next with `--cache --bf16 --nosync` in both arms.

**Cost.**
9 min wall clock (23:33-23:42Z): 1 min for the counts, 7 min for the 8 runs.

**What it overturns.**
G10's reading that a batch-1 update leaves the GPU idle partly because of its host waits: removing all of them in the cached update gains nothing measurable.

---

## G14 - Replaying each update's stages as CUDA graphs speeds StreamSplat up 1.05x at batch 1 (11.54 updates/s) and not at all at batch 4, holding 0.3 GB more VRAM at batch 1 and 4.2 GB at batch 4

**Question.**
With the cache (G11), bf16 (G12) and no host waits (G13), does replaying each update's depth and model stages as CUDA graphs make it cheaper, and what VRAM does that hold?
C4 of plan `misty-painting-summit`, built because G10 found the GPU idle 14-22% of a batch-1 forward (rule: busy < 90%) and G13 brought the model's host waits to zero (rule).

**Setup.**
- Lane: exploratory; registration in the runner header (9a07deb), predictions as approved in the plan.
- `--graphs` (0f8d295; `bench/streamsplat_stages.py` `Graphed`): per batch size, the depth stage (Depth Anything plus min-max) and the model stage (`encode_frames` plus `pair_stage`) are each recorded as a CUDA graph on first use, after 3 warm-up calls, in one shared memory pool, and replayed after that; frames are uploaded and cached features gathered outside the graphs.
- Two deviations from the plan, both made before any graph timing and stated in the header:
  - graphs are recorded for every batch size the server forms, not for a fixed list with padding;
  - capture failed on StreamSplat's training-time block checkpointing (`torch.utils.checkpoint` reads the CUDA RNG state even under `no_grad`), so both arms run with the new `--no-checkpointing`, a prerequisite like G13's `-O`.
  Autocast keeps no cast cache under `--graphs`, as capture requires.
- `probes/streamsplat_graph_equivalence.py` (03ce095), run under `python -O` by the runner before timing: 4 users, 3 updates; `GRAPH_EQUIVALENCE_OK`, with checkpointing off bitwise equal to G13's server and the graphs bitwise equal to the eager control for every output key at every update.
- Timing as G11-G13: one server, N = batch = b users back to back (`--period 0 --max-updates 20 --warmup 3`), compile off, `--cache --bf16 --nosync --no-checkpointing` in both arms, b in 1 4; arm `graphs` adds `--graphs`, arm `eager` is the matched control; arms interleaved, 2 repeats each (n = 2 per cell).
- Run 2026-11-18 23:49-23:59Z, started at 9a07deb.
  A commit that touched only `bench/sweep_lib.sh` (1badee0) landed while the runs were going, so their `meta.json` names 1badee0; the runner had sourced `sweep_lib.sh` at its start, and `bench/capacity.py` and `bench/streamsplat_stages.py` are identical at both commits.
  The tree was otherwise dirty only by untracked files; `foreign_procs` is empty in all 8 runs, nothing was redone.

**Command.**
```
bash experiments/capacity/run_G14_streamsplat-graphs.sh           # the probe, then 8 runs; resumable
bash experiments/capacity/run_G14_streamsplat-graphs.sh metrics   # re-derives outputs/capacity/G14/metrics.json and prints the ratios
python3 -c 'import json,glob; [print(f, *(round(u[k],4) for k in ("svc_p50","svc_depth_p50")), u["peak_alloc"]>>20, s["gpu_mem_mib_max"]) for f in sorted(glob.glob("outputs/capacity/G14/*/n*/r*/summary.json")) for s in [json.load(open(f))] for u in s["users"][:1]]'
```

**Result.**

| b (users = batch) | arm | updates/s (r1, r2) | x eager | service per batch p50 (s) | depth per batch p50 (s) | Gaussian part per batch (s) | GPU memory max (MiB, r1, r2) | peak allocated (MiB) |
|---|---|---|---|---|---|---|---|---|
| 1 | eager | 10.96, 10.99 | 1.00 | 0.0885 | 0.0461 | 0.0424 | 3,906, 3,906 | 2,931 |
| 1 | graphs | 11.52, 11.57 | 1.05 | 0.0851 | 0.0478 | 0.0373 | 4,068, 4,366 | 2,461 |
| 4 | eager | 11.18, 11.11 | 1.00 | 0.352 | 0.188 | 0.165 | 7,200, 7,200 | 5,113 |
| 4 | graphs | 11.18, 10.95 | 0.99 | 0.356 | 0.190 | 0.166 | 11,352, 11,352 | 3,202 |

- **Batch 1 gains 5.2%**: both `graphs` runs above both control runs; the Gaussian part falls to 0.88x (37.3 against 42.4 ms), while the measured depth stage, upload included, rises from 46.1 to 47.8 ms (not explained).
- **Batch 4 gains nothing** (0.99x): at 4 frames per call the kernels are large enough that launching them was not what kept the GPU waiting; this reading is not tested.
- **Memory**: the graphs' pool holds 162-460 MiB more at batch 1 (whole-GPU maximum) and 4,152 MiB more at batch 4, where the server records graphs for batch sizes 1 to 4 during its warm-up; live tensors outside the pool fall (`peak_alloc` 2,461 against 2,931 MiB and 3,202 against 5,113), since the graphs' intermediates live in the pool.
- Throughput against G9's serial server (4.92 updates/s at batch 1, 2026-11-16): the components so far, cache, bf16, no host waits, no checkpointing and graphs, reach 11.54, 2.35x; this compares runs on different days (G12 found a few percent of variation within one evening).
- Checkpointing off alone is not measured here; the `eager` arm against G13's `nosync` arm (the same server with checkpointing on, run 15 minutes earlier) is an unmatched comparison, 10.97 against 10.73 updates/s at batch 1 and 11.14 against 11.20 at batch 4, inside the few percent that repeats of one configuration vary across runs.

**Null.**
The matched control `eager`: the same server with `--cache --bf16 --nosync --no-checkpointing`, interleaved, differing only in `--graphs`.

**Against the predictions registered in the runner header.**
- P1 1.08-1.20x the matched control, refuted if < 1.05x: **not refuted at batch 1, just above the bound and below the predicted range** (1.052x); **refuted at batch 4** (0.993x).
- P2 about 2 GB more VRAM in bf16, only batch 1 recorded if > 8 GB: **refuted as a size** (0.2-0.5 GB at batch 1, 4.2 GB at batch 4 with graphs for four batch sizes, against the plan's three); the 8 GB fallback is not triggered.
- Decision: the final 1 Hz sweep's `fast` arm runs the batch-1 server (`--batch 1`, the same as `serial`), where the graphs gain 5% for about 0.3 GB, so `--graphs` is in it: `fast` = `--batch 1 --cache --bf16 --nosync --no-checkpointing --graphs`.

**Cost.**
9 min wall clock (23:49-23:59Z): about 1 min for the probe, 8 min for the 8 runs.

**What it overturns.**
G10's 14-22% idle share at batch 1 predicted the graphs' gain as 8-20%; they recover 5%, and nothing at batch 4.
After G13 and G14, neither host waits nor kernel launches explain much of the remaining time: at batch 1 an update is 48 ms of depth and 37 ms of model.

---

## G15 - At one keyframe per user per second the RTX 6000 Ada serves 10 StreamSplat users with the fast server against 4 with the serial one, at 0.09 s p95 latency and about 1.0 GB more VRAM

**Question.**
At one keyframe per user per second, how many StreamSplat users does the RTX 6000 Ada serve with the components that passed G11-G14, against the serial server, and at what VRAM per user?
The last step of plan `misty-painting-summit`.

**Setup.**
- Lane: exploratory; registration in the runner header (5f7c77c), predictions as approved in the plan before G10-G14 ran and kept unchanged, although G11-G14's numbers were known when the header was written (the header says so).
- G6's protocol: period 1 s, phases evenly staggered, 60 s per point, 3 warm-ups, compile off; SLO p95 latency <= 1.0 s, 60/60 updates, no OOM; capacity is the largest N where both repeats pass.
- Two arms on the same server code, both batch 1, interleaved per N, 2 repeats (n = 2 per point):
  - `serial` = G9's server (`--batch 1`: TF32, no cache, checkpointing on, eager), the matched control;
  - `fast` = `--batch 1 --cache --bf16 --nosync --no-checkpointing --graphs`, every component that passed G11-G14.
  The arms differ in five settings at once, so this entry measures their sum; G11-G14 measured each one.
- Grid N in 1 2 4 5 6 7 8 10 12; an arm stops after two consecutive N where neither repeat passes.
- Step 0 before the sweep: each arm back to back at N = 1 (`--period 0 --max-updates 20 --warmup 3`); gate `fast` >= 6.0 updates/s.
- Selection criterion declared before the sweep: the capacity rule above; quality is not what `fast` is selected on (G12: bf16 costs 0.18 dB of keyframe PSNR on DAVIS val).
- Run 2026-11-19 00:00:48-00:55:08Z at 5f7c77c (every `meta.json`), tree dirty only by untracked files; `foreign_procs` is empty in all 30 runs, nothing was redone or aborted.

**Command.**
```
bash experiments/capacity/run_G15_streamsplat-fast-1hz.sh           # step 0, the gate, then the sweep; resumable
bash experiments/capacity/run_G15_streamsplat-fast-1hz.sh metrics   # re-derives outputs/capacity/G15/metrics.json and prints the checks below
python3 -c 'import json; m=json.load(open("outputs/capacity/G15/metrics.json")); [print(a, n, r, p["peak_alloc_max"]>>20, round(p["svc_p50_mean"],4), round(p["svc_depth_p50_mean"],4), round(p["gpu_util_mean"] or 0), p["updates_min"]) for a in m["points"] for n in sorted(m["points"][a], key=int) for r, p in sorted(m["points"][a][n].items())]'
```

**Result.**
Step 0, back to back at N = 1: `serial` 4.95, `fast` 11.53 updates/s (2.33x); the gate passed.

| N | `serial` pass | `serial` p95 worst user (s, r1, r2) | `fast` pass | `fast` p95 worst user (s, r1, r2) | GPU memory max (MiB), `serial` / `fast` | per user (MiB), `serial` / `fast` |
|---|---|---|---|---|---|---|
| 1 | 2/2 | 0.210, 0.209 | 2/2 | 0.090, 0.090 | 3,890 / 4,806 and 4,068 | 3,890 / 4,437 |
| 2 | 2/2 | 0.207, 0.210 | 2/2 | 0.087, 0.089 | 3,890 / 4,908 | 1,945 / 2,454 |
| 4 | 2/2 | 0.205, 0.209 | 2/2 | 0.090, 0.089 | 3,892 / 4,914 | 973 / 1,228 |
| 5 | 0/2 (58/60 updates) | 2.298, 2.778 | 2/2 | 0.089, 0.089 | 3,894 / 4,916 | 779 / 983 |
| 6 | 0/2 (48/60) | 11.847, 12.013 | 2/2 | 0.092, 0.089 | 3,894 / 4,918 | 649 / 820 |
| 7 | not run | | 2/2 | 0.090, 0.089 | - / 4,920 | - / 703 |
| 8 | not run | | 2/2 | 0.090, 0.089 | - / 4,922 | - / 615 |
| 10 | not run | | 2/2 | 0.096, 0.094 | - / 4,928 | - / 493 |
| 12 | not run | | 0/2 (55/60) | 4.963, 5.226 | - / 4,932 | - / 411 |

- **Capacity: `serial` 4, `fast` 10** (both repeats and any repeat alike); `serial` stopped after failing at 5 and 6, as registered; `fast` ran the whole grid and failed at 12, so 10 is not at the grid's edge, but 11 was not run: the capacity is 10 or 11.
- **The capacity follows step 0's throughput**: every N below the arm's back-to-back rate passed and every N above it failed (`serial` 4 of 4.95 passes, 5 fails; `fast` 10 of 11.53 passes, 12 fails); N between 10 and 11.53 was not tested.
- **Latency**: `fast`'s median service is 0.077-0.088 s at every passing N, against `serial`'s 0.191-0.196 s; its worst user's p95 stays at 0.087-0.096 s up to N = 10.
  Past the rate both arms queue without bound: p50 latency 0.89-1.34 s for `serial` at N = 5, 2.4 s for `fast` at N = 12.
- **VRAM**, the registered metric (`gpu_mem_mib_max / N`): `fast` holds about 1.0 GB more for the whole GPU at every shared N (4,914 against 3,892 MiB at N = 4), 1.14-1.26x `serial` per user.
  At each arm's own capacity, `fast` serves 10 users in 4,928 MiB (493 per user) and `serial` 4 users in 3,892 MiB (973 per user); both use 10% or less of the 49,140 MiB card.
  Live tensors (`peak_alloc`) grow by 24.6 MiB per user under `fast` (2,462 MiB at N = 1 to 2,732 at N = 12), the cached bf16 features, and by 2.6 MiB per user under `serial`.
  `fast`'s whole-GPU memory at N = 1 differs between repeats (4,806 against 4,068 MiB) with the same allocator reserve (3,474 MiB), and the reserve steps to 4,314 MiB from N = 2 on; neither was explained.
- **GPU utilisation** (nvidia-smi means) at the largest N each arm ran: `serial` 93% at N = 6, `fast` 94% at N = 12.
  Below saturation the means scatter between repeats of one point (`fast` N = 4: 9% and 68%; `serial` N = 2: 56% and 92%), so they say nothing there.
- Depth stays at 41-51 ms of every update in both arms; in `fast` it is now about half of the service time.

**Null.**
The matched control `serial`, G9's batch-1 server on the same code and protocol, interleaved per N; its capacity reproduces G6's 4 users (separate containers, time-slicing and MPS, 2026-11-14).

**Against the predictions registered in the runner header.**
- `serial` capacity 4 (5 possible), refuted if neither: **holds** (4).
- `fast` capacity 6 (5-8), refuted if <= `serial`: **not refuted; the estimate is missed high** (10).
- Saturated throughput `serial` 4.7-4.9, `fast` 7-9 updates/s: **missed high** for both (4.95; 11.53).
- p95 at N = 4 `serial` 0.19-0.21 s, `fast` 0.11-0.16 s, refuted if `fast` is not below 0.9x `serial`: **not refuted** (0.207 and 0.090 s, 0.43x); `fast` is below its range.
- `fast` GPU utilisation at saturation >= 85%, refuted if < 78%: **holds** (94% at N = 12).
- VRAM per served user, refuted if `fast` exceeds 2x `serial` at any shared N: **not refuted** (at most 1.26x).
- None of the plan's stop conditions was triggered.

**Cost.**
54 min wall clock, 30 runs (2 at step 0, 28 in the sweep: 10 `serial`, 18 `fast`), none redone; the lab's own GPU, no charge.

**What it overturns.**
No earlier entry is refuted outright; G15 changes how G6 reads.
G6 found 4 users at 1 Hz with the limit in compute, from the released inference code's 0.19 s per update; that bound belongs to the code, not to the card: with the paper's own feature cache (G11), the precision the model was trained in (G12) and CUDA graphs (G14), the same card serves 10, for 1.0 GB of VRAM and 0.18 dB of keyframe PSNR.
G2-G9 found that sharing mechanisms (time-slicing, MPS, batching across users) serve no more users; G15 does not change that: the gain comes from doing less work per update, not from sharing the GPU differently.
The capacity is still compute-bound: at 10 users the card uses about 5 GB, and depth (Depth Anything V2 ViT-L in TF32, untouched by every component) is about half of each update.

---

## G16 - Running Depth Anything V2 under bf16 autocast lets the RTX 6000 Ada serve 14 StreamSplat users at one keyframe per second against 10, for 0.013 dB of keyframe PSNR against G12

**Question.**
G15's fast server spends 47.6 ms of an 87 ms update in the depth model (Depth Anything V2 ViT-L in TF32, untouched by G11-G14).
Does running the depth model under bf16 autocast let the RTX 6000 Ada serve more users at one keyframe per user per second, and what does it cost in quality?
Plan `misty-painting-summit`, G16 version, approved 2026-11-19.

**Setup.**
- Lane: exploratory; registration in the runner header (ded164d), as approved in plan mode; the header lists what was known when it was written (G15's 11.53 updates/s, capacity 10, depth 47.6 ms; the reviewers' FLOP count) and the dry run and smoke runs that came before it.
- `--depth-bf16` (bench/capacity.py, 1c39de2): `disparity_of` runs the depth model alone under bf16 autocast with no cast cache, then casts to fp32 for the interpolation and the per-frame min-max; the priming frame and every update use it, and `--graphs` records it.
- Two arms on the same server code, both batch 1, interleaved at every step, 2 repeats (n = 2 per point):
  - `fast` = G15's `--batch 1 --cache --bf16 --nosync --no-checkpointing --graphs`, the matched control;
  - `depthbf16` = `fast` plus `--depth-bf16`, one factor apart.
- Bug checks first, exit 3 on failure: the graph equivalence probe with `--depth-bf16` under `python -O`; `bench/streamsplat_depth_check.py` on all 64 clip frames (positive control: the depth output is bf16 and the disparity differs; every frame still spans 0..255); every timed run's server ready line must carry `depth_bf16` as its arm says.
- Step 0: N = batch = 1 back to back (`--period 0 --max-updates 20 --warmup 3`), compile off; gate `depthbf16` / `fast` >= 1.10.
- Step 1, quality on the 30 DAVIS 2017 val sequences with the authors' `psnr_given` and `psnr_novel` (G8's protocol):
  - reproducibility first: G12's command rerun on the 2 shortest val sequences (car-shadow, judo), and judo's fp32 depth maps from the changed `davis_bear.py` compared byte for byte with G8's;
  - bf16 depth maps for all 1,999 val frames, then `eval --bf16` on all 1,849 val windows, paired per window against G12 (the same model precision, only the depth maps differ) and against G8's val windows (full precision);
  - quality gate: a cumulative mean drop against G8 > 0.5 dB in either metric stops the run before the sweep.
  - The check differs from the server in three ways, written in the header: G8's DAVIS depth ran in full fp32, the server's in TF32; DAVIS frames give 924x518 depth inputs (2,442 tokens) against the clip's 686x518 (1,813); the evaluation runs `SplatModel.forward`, not the cached server.
- Step 2: G6's and G15's 1 Hz protocol (period 1 s, phases evenly staggered, 60 s per point, 3 warm-ups, compile off; SLO p95 latency <= 1.0 s, 60/60 updates, no OOM; capacity is the largest N where both repeats pass), grid N in 10 11 12 13 14 15 16 17 18 20 22 24; an arm stops after two consecutive N where neither repeat passes.
- Selection criterion declared before the sweep: the capacity rule above; quality is not what the arms are selected on (Step 1 reports it).
- Run 2026-11-19 13:36:00-14:43:00Z at ded164d (every `meta.json`), tree dirty only by untracked files; no code changed after the header's commit; `foreign_procs` is empty in all 26 runs, nothing was redone or aborted.

**Command.**
```
bash experiments/capacity/run_G16_streamsplat-depth-bf16.sh           # bug checks, step 0, quality, sweep; resumable
bash experiments/capacity/run_G16_streamsplat-depth-bf16.sh metrics   # re-derives outputs/capacity/G16/metrics.json and davis/metrics.json, prints the step-0, quality and sweep numbers below
python3 -c 'import json; d=json.load(open("outputs/capacity/G16/checks/depth_check.json")); print({k: d[k] for k in d if k != "per_frame"})'
python3 -c 'import json; m=json.load(open("outputs/capacity/G16/metrics.json")); [print(a, n, r, p["peak_alloc_max"]>>20, round(p["svc_p50_mean"],4), round(p["svc_depth_p50_mean"],4), round(p["gpu_util_mean"] or 0), p["updates_min"], p["gpu_mem_mib_max"]) for a in m["points"] for n in sorted(m["points"][a], key=int) for r, p in sorted(m["points"][a][n].items())]'
for r in fast_r1 fast_r2 depthbf16_r1 depthbf16_r2; do python3 -c 'import json,sys; u=json.load(open(sys.argv[1]))["users"][0]; print(sys.argv[2], round(u["svc_p50"],4), round(u["svc_depth_p50"],4))' outputs/capacity/G16/step0/$r/summary.json $r; done
```

**Result.**
The bug checks passed: GRAPH_EQUIVALENCE_OK; the depth model's output was fp32 then bf16 on all 64 frames, the disparity differed, and every frame spans 0..255; every run's ready line carried its arm's `depth_bf16`.

Step 0, back to back at N = 1:

| | `fast` (r1, r2) | `depthbf16` (r1, r2) | `depthbf16` / `fast` |
|---|---|---|---|
| updates/s | 11.62, 11.66 | 15.84, 15.59 | 1.350x |
| service p50 (ms) | 83.9, 84.5 | 60.2, 61.9 | 0.73x |
| depth stage p50 (ms) | 46.7, 47.5 | 26.5, 28.5 | 0.584x |

The gate passed; `fast` runs 1.0% above G15's 11.53 updates/s.
The service falls 23.1 ms and the depth stage 19.6 ms; the 3.5 ms between them is a difference of medians and is not attributed.

Depth differences, in levels of 0..255:
- The server's input on the clip's 64 frames, TF32 against bf16: mean 0.29, max 8, 71% of pixels equal.
- The 1,999 DAVIS val maps, G8's full fp32 against bf16: mean 0.36, max 36, 67% of pixels equal.

Reproducibility: G12's rerun on car-shadow and judo gives 64 of 64 windows bitwise equal to G12's rows (max change 0.0 dB), so the per-window numbers below have no run-to-run noise floor; judo's fp32 depth maps from the changed `davis_bear.py` equal G8's PNGs byte for byte.

Quality on the 1,849 val windows in 30 sequences:

| metric | G8 (full precision) | G12 (model bf16) | G16 (model and depth bf16) | against G12: mean; per window min to max; windows dropping > 0.5 dB | against G8, cumulative: same columns |
|---|---|---|---|---|---|
| `psnr_given` | 37.757 | 37.581 | 37.568 | -0.013; -0.262 to +0.143; 0.0% | -0.189; -0.844 to +0.255; 5.1% (94 windows; G12 alone 4.3%) |
| `psnr_novel` | 23.761 | 23.750 | 23.749 | -0.001; -0.147 to +0.219; 0.0% | -0.012; -0.528 to +0.439; 0.05% (1 window) |

The quality gate passed (cumulative drops 0.19 and 0.01 dB against 0.5).

The sweep:

| N | `fast` pass | `fast` p95 worst user (s, r1, r2) | `depthbf16` pass | `depthbf16` p95 worst user (s, r1, r2) | GPU memory max (MiB), `fast` / `depthbf16` | GPU utilisation (%, r1, r2), `fast` / `depthbf16` |
|---|---|---|---|---|---|---|
| 10 | 2/2 | 0.094, 0.100 | 2/2 | 0.074, 0.077 | 4,928 / 4,766 | 79, 82 / 55, 55 |
| 11 | 0/2 (57/60 updates) | 3.618, 3.587 | 2/2 | 0.076, 0.076 | 4,930 / 4,768 | 85, 85 / 61, 60 |
| 12 | 0/2 (52/60) | 8.126, 8.363 | 2/2 | 0.075, 0.075 | 4,932 / 4,880 | 84, 84 / 64, 68 |
| 13 | not run | | 2/2 | 0.078, 0.074 | - / 4,882 | - / 72, 73 |
| 14 | not run | | 2/2 | 0.074, 0.087 | - / 4,884 | - / 85, 84 |
| 15 | not run | | 1/2 (r2 59/60) | 0.376, 1.130 | - / 4,888 | - / 91, 90 |
| 16 | not run | | 0/2 (57 and 56/60) | 3.617, 4.492 | - / 5,000 | - / 91, 88 |
| 17 | not run | | 0/2 (52/60) | 8.072, 8.530 | - / 5,002 | - / 88, 88 |

N = 18 to 24 ran for neither arm; each stopped after two consecutive failing N, as registered.

- **Capacity: `fast` 10, `depthbf16` 14** with both repeats; with any repeat, 10 and 15.
  Neither is at the grid's edge and the grid has no gap next to either, so both are exact on this grid; G15's "10 or 11" for `fast` is 10.
- **Where the boundary sits**: in both arms N passes up to about 0.89 of the back-to-back rate and fails from about 0.95 (`fast` 10 / 11.64 = 0.86 passes, 11 / 11.64 = 0.945 fails; `depthbf16` 14 / 15.71 = 0.89 passes, 15 / 15.71 = 0.955 passes once in 2).
  N below the rate is not enough, as G15 read it; why the last 5-10% of the rate cannot be used was not tested.
- **Latency**: at N = 10 the worst user's p95 is 0.076 s against 0.097 (0.78x); at its capacity of 14, `depthbf16`'s worst user's p95 is 0.074 and 0.087 s.
  Past capacity both arms queue without bound (p95 3.6-8.5 s).
- **Service under load**: `depthbf16`'s median service is 64.5-72.0 ms at every N, its depth stage 29.6-38.2 ms.
  `fast`'s is 85.6-89.2 ms at N = 10 but 96.3-96.7 ms at N = 11 and 12 (depth 57.8-58.0 ms), longer than at its step 0 (84.2 and 47.1 ms) and than G15's `fast` at N = 12 (90.0 ms, depth 50.8 ms), which also ran the GPU at 93-95% against 84% here and delivered 55/60 updates against 52/60; the cause was not tested.
- **VRAM**: whole-GPU memory at equal N is 52-162 MiB lower with `depthbf16`; live tensors (`peak_alloc`) are equal at equal N (2,682-2,683 MiB at N = 10, 2,732 at N = 12).
  At capacity, `depthbf16` serves 14 users in 4,884 MiB (349 per user), `fast` 10 users in 4,928 MiB (493 per user); both use 10% of the 49,140 MiB card.
- **GPU utilisation** (nvidia-smi means) at the largest N each arm ran: `fast` 84% (N = 12), `depthbf16` 88% (N = 17); at capacity, 79-82% and 84-85%.
- Depth is 44-46% of a `depthbf16` update at step 0 (26.5-28.5 of 60.2-61.9 ms), against 55-56% for `fast`.

**Null.**
The matched control `fast`, G15's server on the same code, differing only in `--depth-bf16` and interleaved at every step; it reproduces G15 (11.64 against 11.53 updates/s back to back, capacity 10).
For quality, G12's windows (the same model precision, only the depth maps differ) and G8's (full precision).

**Against the predictions registered in the runner header.**
- P1 depth stage 0.43-0.60x `fast`, refuted if > 0.80x: **holds** (0.584x, 47.1 to 27.5 ms).
- P2 step-0 throughput 1.30-1.45x, refuted if < 1.10x: **holds** (1.350x, 11.64 to 15.71 updates/s).
- P3 depth differences, reported only, the server's frames mean <= 1 level, DAVIS maps <= 3: **holds** (0.29; 0.36).
- P4 against G12 `psnr_given` 0 to -0.10 dB, `psnr_novel` 0 to -0.05 dB, <= 2% of windows dropping > 0.5 dB, refuted if a mean drops > 0.20 dB or > 5% of windows drop > 0.5 dB: **holds** (-0.013 and -0.001 dB, 0.0%); cumulative against G8 `psnr_given` 0.18-0.28 dB: **holds** (0.189).
- P5 `fast` 10 or 11, `depthbf16` 15 (14-16), refuted if `depthbf16`'s <= `fast`'s: **holds** (10; 14, the low end of the range).
- P6 each arm's capacity is the largest grid N below its step-0 throughput (11 and 15), refuted if either is more than one grid step lower: **not refuted, missed in both arms** (10 and 14, one step lower each).
- P7 `fast` step 0 within ±5% of 11.53: **holds** (+1.0%).
- P8 p95 at N = 10 below 0.85x `fast`, refuted if not below 0.9x: **holds** (0.78x).
- P9 whole-GPU memory at equal N within ±0.5 GB, refuted if it rises > 1.5 GB: **holds** (-0.05 to -0.16 GB).
- P10 GPU utilisation >= 85% at the largest N each arm ran, refuted if < 78%: **holds for `depthbf16`** (88%), **missed by 1 point for `fast`** (84%), not refuted.
- Both gates passed; none of the plan's stop conditions was triggered.

**Cost.**
67 min wall clock: bug checks 1.5 min, step 0 3.6 min, quality 19 min (reproducibility 1.3, depth maps 4.6, evaluation 13), sweep 43 min; 26 timed runs (4 at step 0, 22 in the sweep: 6 `fast`, 16 `depthbf16`), none redone; the lab's own GPU, no charge.

**What it overturns.**
- G15's reading that every N below the back-to-back rate passes at 1 Hz (G15 Result): `fast` fails at N = 11, below its 11.64 updates/s, which G15 had not run; in both arms N passes up to about 0.89 of the rate.
- G15's open capacity "10 or 11" is 10.
- G15's "depth is about half of each update" holds for `fast` (55-56% here); with `--depth-bf16` it is 44-46%, and the card serves 14 users at 1 Hz in 4.9 GB, under 10% of its memory.
No other entry is refuted; no earlier entry changed the depth model's precision.
The configuration carried forward is `fast` plus `--depth-bf16`.

---

## G17 - A fresh public clone set up by reproduce.sh fetched and verified every file and rebuilt the StreamSplat image with identical packages, but the TeX image failed on a CTAN mirror, so smoke and report never ran

**Question.**
Does a fresh clone of the public repository, set up only by `reproduce.sh`, give the published numbers?
Plan `misty-painting-summit`, public-release version, approved in plan mode on 2026-11-25.

**Setup.**
- Lane: quotable verification (the README cites it); registration in the runner header, committed before the run (0784830).
- The runner clones github.com/kylevirtuous1211/virtualization_final_project at e344823 into `outputs/reproduce/G17/clone` and runs, inside it, `STREAMSPLAT_IMAGE=streamsplat:g17 TEXLIVE_IMAGE=texlive-report:g17 DOCKER_BUILD_ARGS="--no-cache --pull" ./reproduce.sh setup build smoke report`.
- Controls: G16's step 0 (11.64 and 15.71 updates/s, ratio 1.350); G8's per-window PSNR and depth maps; the committed `main.pdf`.
- Known before the run: `reproduce.sh setup` on this machine (every file verified) and a smoke run on this machine with the existing image (PASS; a development test, not a result).
- Run 2026-11-25 14:30:30-14:50:00Z; setup 718 s, build 452 s.

**Command.**
```
bash experiments/reproduce/run_G17_fresh-clone-smoke.sh e344823a12c673b88b13864fdca0b0cc013b10de
cat outputs/reproduce/G17/steps.txt; grep -E '^(ok|fetching)' outputs/reproduce/G17/setup.log; grep -n 'cannot download' outputs/reproduce/G17/build.log
for i in 5d686e9 g17; do docker run --rm --network none --entrypoint cat streamsplat:$i /opt/freeze.txt > /tmp/freeze_$i.txt; done; diff /tmp/freeze_5d686e9.txt /tmp/freeze_g17.txt   # by hand, after the run
```

**Result.**
- `setup` exited 0. It cloned StreamSplat at 5d686e9 and fetched `streamsplat.safetensors` (Google Drive), `depth_anything_v2_vitl.pth` (Hugging Face), `kitti360.mp4` and the DAVIS zip afresh; every sha256 matched. The three torch hub weights were already in `~/.cache/torch` and were verified there.
- `build` exited 1. The StreamSplat image built first with `--no-cache --pull` (`streamsplat:g17`, sha256:5974afed6a09), and its `/opt/freeze.txt` equals the G16 image's in all 154 packages (checked by hand after the run).
- The TeX image then failed: `tlmgr install sttools breakurl` could not download `texlive.tlpdb.sha512` from the CTAN mirror `ctan.mirror.twds.com.tw`.
- Two `--no-cache` rebuilds of the same Dockerfile, minutes later, succeeded, so the mirror failure was transient. A reproduction step that needs a live CTAN mirror is still fragile, and it will break for good once TeX Live 2026 is frozen.
- `smoke`, `report`, the torch hub URL check and `metrics.json` did not run, because the runner stops on a failed build.

**Against the predictions registered in the runner header.**
- P1 every download matches its sha256, StreamSplat at 5d686e9, no login: **holds** for everything this run fetched; the torch hub URLs were not tested by the run.
- P2 the `--no-cache --pull` build succeeds and the freeze differs at most in jaxtyping: **refuted** by the TeX image; the StreamSplat image's freeze is identical.
- P3-P5: not tested.

**Cost.**
20 min wall clock, no GPU time, about 2.9 GB downloaded; the lab's own machine, no charge.

**Fix.**
208f59f ships `flushend.sty` and `breakurl.sty` (LPPL) beside the report, and the TeX Dockerfile is the pinned base image alone, so the build downloads nothing from CTAN.
With that image the report rebuilds byte-identical to the committed PDF.
G18 reruns this registration unchanged on the fixed commit.

**What it overturns.**
- No measured number.
- Correction found while building `results/tables.md`: G13's "1.02x at batch 4" (its title and table) is 1.01x. The ratio of the means is 11.205 / 11.04 = 1.0147, as G13's own "+1.5%" says. The report's Table 3 gives 1.01.

---

## G18 - From a fresh clone of the public repository, reproduce.sh gives G16's throughput within 3.0%, G8's quality windows bitwise, and the report byte for byte

**Question.**
G17's question, rerun after 208f59f removed the TeX build's CTAN download: does a fresh clone of the public repository, set up only by `reproduce.sh`, give the published numbers?

**Setup.**
- Lane: quotable verification; G17's registration unchanged, in the runner header (236120d), committed and pushed before the run.
- The runner clones github.com/kylevirtuous1211/virtualization_final_project at 236120d into `outputs/reproduce/G18/clone` and runs `STREAMSPLAT_IMAGE=streamsplat:g18 TEXLIVE_IMAGE=texlive-report:g18 DOCKER_BUILD_ARGS="--no-cache --pull" ./reproduce.sh setup build smoke report` inside it.
- `smoke` is G16's step 0 for `fast` and `depthbf16` (2 runs each, 20 updates after 3 warm-ups), then G8's full-precision depth maps and evaluation on the 2 shortest val sequences, car-shadow and judo.
- Known before the run: G17, and a PASS of `smoke` on this machine with the existing image (a development test).
- Run 2026-11-25 14:53:49-15:18:46Z on the RTX 6000 Ada, driver 595.91.07; images streamsplat:g18 sha256:e669696b5faf and texlive-report:g18 sha256:f534be1e15b6.

**Command.**
```
bash experiments/reproduce/run_G18_fresh-clone-smoke-rerun.sh 236120d7b06ec11f0d6251682fe34ed1a2a9ce33
python3 -c 'import json; m=json.load(open("outputs/reproduce/G18/metrics.json")); print(m["steps"], m["predictions_hold"]); print(m["smoke"]["throughput"], m["smoke"]["ratio"], m["smoke"]["quality"]); print(m["depth_maps"], m["pdf"], m["freeze_differs"])'
```

**Result.**
- Every step exited 0: setup 698 s (4 files fetched afresh, every sha256 matched), build 470 s, smoke 324 s, report 5 s.
- The three torch hub URLs, downloaded again into `hub_check`, match their sha256.
- The new StreamSplat image's `/opt/freeze.txt` equals the G16 image's in every package.

| | G18 (fresh clone) | Published (G16) | Change |
|---|---|---|---|
| `fast` back to back, updates/s | 11.29 | 11.64 | -3.0% |
| `depthbf16` back to back, updates/s | 15.25 | 15.71 | -2.9% |
| ratio | 1.351 | 1.350 | +0.1% |

- Quality: all 64 windows of car-shadow and judo equal G8's bitwise (largest change 0.0 dB), and all 74 fp32 depth maps of the two sequences equal G8's byte for byte (40 and 34).
- The report rebuilt from `results/` equals the committed `main.pdf` byte for byte, so its text is equal too.

**Against the predictions registered in the runner header.**
- P1 downloads match, StreamSplat at 5d686e9, no login: **holds**.
- P2 the `--no-cache --pull` build succeeds, freeze differs at most in jaxtyping: **holds** (no difference at all).
- P3 each arm within 5% of G16 and the ratio within 5% of 1.350: **holds** (-3.0%, -2.9%, +0.1%).
- P4 the shortest sequence's depth maps byte-equal to G8's and the windows bitwise equal: **holds**, for both sequences.
- P5 the rebuilt PDF has the committed PDF's text: **holds**, byte for byte.

**Cost.**
25 min wall clock, about 5 GPU minutes, about 2.9 GB downloaded; the lab's own machine, no charge.

**What it overturns.**
- G17's refuted P2 (the TeX image failed on a CTAN mirror): with 208f59f both images build from a fresh clone.
- No measured number is overturned. Both arms ran about 3% below G16, inside the drift between experiments that the report already states (about 7% for `fast` at saturation between G15 and G16); the ratio, the comparison the report rests on, matches to 0.1%.

---

## Refuted claims

| Claim | Refuted by | Note |
|---|---|---|
| Per-user VRAM is dominated by the user's own copy of the weights (plan prediction 2) | G1 | Holds at k = 2 (59% weights), fails at k = 8 (33%) and k = 16 (29%). |
| MPS serves at least one more user than time-slicing (plan prediction 1) | G2, G3, G6, G7 | Equal capacity in every setting: AnySplat 2 at k = 8 and 1 update/s (G2), 5 at k = 2 and 2 updates/s (G3); StreamSplat 4 at 1 update/s (G6), 1 at 3 updates/s (G7). |
| At k = 2, time-slicing tops out near one user's rate, 1 / 0.122 s = 8.2 updates/s (G3 P2) | G3 | Step 0 reached 10.06 updates/s with 4 users back to back. |
| The released StreamSplat clears 30 dB keyframe PSNR on DAVIS `bear`, from its mp4 outputs at 512x288 (G4 gate) | G4 | 27.51 dB. By the paper's protocol (256x256, tensors, val split), G8 gives 37.76 dB on val and 35.94 dB on `bear`. |
| One StreamSplat update (depth plus a keyframe pair) costs about 0.09 s on the RTX 6000 Ada (G5) | G5 | 0.190 s with compile, 0.198 s without; `forward_gaussians` alone is 0.152 s. |
| With StreamSplat at 1 update/s, the capacity limit moves from compute to memory, about 9-11 users (G6) | G6 | 4 users under both arms; compute-bound at about 4.7 updates/s, memory would allow about 12. |
| At 3 updates/s, a StreamSplat user leaves the GPU mostly idle, so MPS serves more users (G7) | G7 | 1 user under both arms; one user takes 55% of the GPU. |
| Batching the updates of 8 StreamSplat users into one call raises throughput about 2.5x, because batch 1 leaves the GPU 27-36% idle (G9 P1) | G9 | 0.92x (4.52 against 4.92 updates/s); a batch of 8 takes 8.7x as long, `forward_gaussians` alone 9.1x. |
| Batching one stage across users helps the small stages: DINO and both heads cost <= 0.6x per sample at batch 4, saving 5-15 ms per update (G10 P6) | G10 | DINO 0.93x, dynamic head 1.16x, upsamplers 1.25-1.46x; only the static head reaches 0.44x; 2.9 ms saved. |
| Running the depth stage and the pair stage on two CUDA streams overlaps 5-15% of their work (G10 P7) | G10 | -0.1% without priority, -0.6% with a high-priority depth stream. |
| Each extra StreamSplat sample in a batch adds 650-750 MiB of live activations in TF32 (G10 P8) | G10 | 381 MiB; G9's 1.0-1.2 GB per sample from nvidia-smi is mostly allocator reserve. |
| With the cache, bf16 autocast brings the Gaussian part of a batch-1 update to 0.55-0.75x TF32 (G12 P1) | G12 | 0.40x, faster than predicted; 0.33x at batch 4. |
| bf16 autocast costs StreamSplat at most 0.15 dB of keyframe PSNR on DAVIS val (G12 P3) | G12 | -0.18 dB (37.58 against 37.76); 4.3% of windows drop > 0.5 dB; novel-view PSNR -0.01 dB. |
| The RTX 6000 Ada ran StreamSplat about 5% slower on 2026-11-18 than on 2026-11-16 (G11, its control at 4.68 against G9's 4.92 updates/s) | G12 | The same configuration ran 3.4% faster 30 minutes after G11, with depth back at G9's 0.049 s; the speed varies within a day, cause untested. |
| Removing the host waits from the model calls raises back-to-back StreamSplat throughput 1-5% (G13 P2) | G13 | -2.4% at batch 1 (both runs below both control runs), +1.5% at batch 4 (within the control's spread); the 8 waits per cached update did go to 0. |
| Replaying the cached bf16 update as CUDA graphs raises back-to-back StreamSplat throughput 1.08-1.20x, about G10's idle share (G14 P1) | G14 | 1.052x at batch 1 (not refuted, below the range), 0.993x at batch 4 (refuted). |
| The CUDA graphs hold about 2 GB more VRAM in bf16 (G14 P2) | G14 | 0.2-0.5 GB at batch 1; 4.2 GB at batch 4, with graphs for batch sizes 1 to 4. |
| The fast StreamSplat server runs 7-9 updates/s back to back and serves 6 (5-8) users at 1 Hz, at 0.11-0.16 s p95 with 4 users (G15 estimates) | G15 | 11.53 updates/s, 10 users (11 not run), 0.090 s; all beyond the estimates in the favourable direction, no registered refutation criterion met. |
| The 1 Hz capacity is the largest N below the server's back-to-back throughput (G15's reading; G16 P6) | G16 | `fast` fails at 11 below its 11.64 updates/s (57/60 updates, p95 3.6 s) and `depthbf16` passes 15 below 15.71 in 1 of 2 repeats: capacity 10 and 14, one grid step below the rule in both arms, inside P6's tolerance; N passed up to about 0.89 of the rate. |
| `reproduce.sh` builds both images from a fresh public clone (G17 P2) | G17 | The TeX image's `tlmgr install` failed on a CTAN mirror; 208f59f ships the two `.sty` files instead; rerun as G18. |
| Removing the host waits gives 1.02x at batch 4 (G13 title and table) | G17 | A rounding error: the ratio of the means is 1.0147, so 1.01x, as G13's "+1.5%" says. |
