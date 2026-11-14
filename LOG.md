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
