# Direction review

How the project direction in `PLAN.md` was chosen, written 2026-10-27.
Three proposals were written independently from different angles, each grounded in `docs/reading_notes.md`.
A judge scored them and wrote one recommended direction, re-run after the student decided that the first milestones stream 3DGS only, with no feed-forward model until the streaming path is understood.
`PLAN.md` is the readable Chinese version; this file keeps the full English specification it was derived from.

## Scores

Scale 1-5.

| Proposal | Novelty | Feasibility | Course fit | Evaluation rigor | Learnability |
| --- | --- | --- | --- | --- | --- |
| A. Capacity-coupled admission | 3 | 3 | 4 | 5 | 3 |
| B. Tenants per GPU | 2 | 2 | 5 | 4 | 2 |
| C. Measure first | 3 | 3 | 4 | 4 | 4 |

### A. Capacity-coupled admission: is it worth feeding the client's measured run-time integration capacity into the sender's admission decisions? Expected near-null for one client, plus a two-client noisy-neighbour test

Best evaluation design of the three, and the most honest novelty statement.
It narrows the claim to the one coupling that no paper tests, declares a premise test (H0) with a pivot rule, and isolates the effect with a single matched null (arm 5 vs arm 6).
It has two built-in negative controls (the network-bound regime, and an aggressor-capacity-to-link ratio of 0.5), tunes every arm on the same budget, and checks the rAF (requestAnimationFrame) latency proxy against a camera.
Problems:
(1) It skips the student's binding milestone 1 (stream a static .ply).
Its milestone 1 instead builds T-real by online gsplat training on a video.
That is not 'built from existing scenes', and it adds a training pipeline in weeks 3-5.
(2) It specifies 8 arms up front, against decision (2).
(3) Its matched null is not the strongest baseline.
A client that acknowledges data only after integrating it (plain receive-window flow control) also throttles the sender to what the client can consume.
Without that arm, the coupling may just be flow control rediscovered (baseline-hacking risk).
(4) H2's predicted effect follows almost mechanically from TCP fair sharing: a greedy browser reader lets the aggressor hold about C/2 of the link while using only 0.2C.
So H2 measures a magnitude; it is not a surprise.
(5) H2 needs a second client machine.
The novelty claim also rests on GaussAnything (2609.13859) and 2606.17212, which only the sweep read.
Neither had a reader/verifier pass.

### B. Tenants per GPU: consolidation and isolation of live feed-forward 3DGS streams on one server GPU, comparing time-slicing, MPS (Multi-Process Service), MPS with SM caps, and in-process multiplexing

Strongest course fit.
Each arm maps one-to-one onto a virtualization concept: time-slicing is VM scheduling, the SM cap is resource reservation, the shared copy of the weights is page sharing, and fault isolation is the fault domain.
Its rigor is also good: an A/A null (same arm run twice), and a capacity prediction written down before the measurement.
Problems:
(1) It contradicts binding decision (1).
A feed-forward model is the core workload from M2 (weeks 3-6): StreamSplat, with Depth Anything inline and an orthographic WebGPU port.
Static-scene streaming never appears.
(2) The factor under study needs exclusive time on a GPU that was 87-100% busy at both checks, so M0 alone may stall the project.
(3) Novelty is the weakest of the three.
MPS vs time-slicing for DNN inference is well studied in systems research; the proposal admits Clockwork, REEF and Orion were not read.
What remains is an application measurement.
(4) Learnability is low.
The student would have to learn CUDA contexts, MPS and a research model's internals at the same time, before streaming a single Gaussian.
Worth keeping only as an optional late branch, if the instructor requires device-level virtualization.

### C. Measure first: a WebGPU paging client fed by AnySplat, with a declared decision rule that picks the research question, and a one-factor ladder of arms

Closest match to 'understand every step'.
Step 0 is a measurement in week 1, every milestone produces one number, and decision rules declared up front choose the question committed in M5.
The ladder makes adjacent arms matched contrasts.
Its 'version f or later' latency definition is fair to coalescing.
The queueing-theory rung is honestly labelled a sanity check.
Problems:
(1) M2 runs AnySplat offline on four videos in weeks 3-6, which violates binding decision (1).
(2) The 25% threshold in H-main is not derived from anything.
(3) A ladder fixes the order in which factors are added, which can hide interactions, and there is only one 2x2 check.
(4) M6 is the full grid (7 arms x 3 traces x 5 cells, about 35 h) arriving at week 11.
(5) Like A, it lacks a flow-control arm.
(6) Its cache study (LRU vs view-distance vs Belady) is classic OS material, with Spark SplatPager and MoQSplat's watermark swap as prior art.
It is good teaching, not novelty.

## Judge rationale

Totals tie between A and C (18 each); B trails (15) and breaks binding decision (1).
A has the strongest science.
It narrows the favoured joint scheduler to the one piece the verified and swept papers do not cover, declares a premise test, and isolates the effect with one matched null plus negative controls.
C is the best fit for a student whose top priority is to understand every step.
Step 0 is a measurement, each milestone yields one number, and a declared rule picks the question only after the premise has been measured.
Both A and C conflict with the binding decisions.
A's T-real requires online training on video, and C runs AnySplat in weeks 3-6.
Both also leave out the arm that most threatens their novelty: plain acknowledge-on-integrate flow control, which already throttles a sender to a slow client.
So the recommendation takes A's question, hypotheses and controls, restaged on C's measure-first skeleton: static .ply first, then synthetic updates from existing scenes, then the model last.
It adds flow control as the strongest baseline and keeps arms to one or two until M5.
B's virtualization content is the most literal.
Its research question is the least novel, though, and it depends on exclusive time on a GPU that is rarely free, so it is kept only as an optional branch the instructor can call in.
The honest expected outcome:
- The premise (H0) binds only under induced contention.
- For one client, coupling saves bytes, not latency.
- The isolation gain in H2 is large, but largely arithmetic.
- Whether coupling beats plain flow control is the open question, and a negative answer there is still a reportable course result.
- Every number on this path can be produced with the hardware at hand, without the shared GPU and without sudo until week 9.

## What the recommendation took from each proposal

From A (the base question and controls):
- the H0 premise test with its declared pivot thresholds, and the 32 B wire record;
- the single-client near-null H1, and the two-guest noisy-neighbour H2 with the 0.5 negative control;
- the network-bound regime as an internal negative control;
- beacon IDs, and camera validation of the rAF proxy;
- TCP_NOTSENT_LOWAT and sender-side coalescing via a dict keyed by ID;
- equal-budget tuning (at most 9 configurations on a dev trace, criterion declared up front);
- the foreign-GPU exclusion rule and randomised paired blocks;
- the honest per-paper novelty table;
- reading LTS, LMG and Vega and contacting the Hsu group in weeks 1-2.

From C (the skeleton and pace):
- step 0 as a week-1 static frame-time null on a real GPU display;
- one number per milestone, with decision rules declared before measuring;
- the 'version f or later' latency definition, which stays fair to coalescing;
- per-stage latency decomposition with a stage-sum check;
- the synthetic trace that sets rho exactly, and the FIFO knee against rho;
- queueing-theory results labelled as a sanity rung, not a finding;
- measuring head-of-line blocking under loss instead of assuming it, with the MSS clamp checked by tcpdump;
- the Belady-bounded cache curves, as an optional virtualization extension;
- a trace-predicted vs live latency gap check at the end;
- an explicit drop order for the schedule.

From B:
- the A/A null (same arm run twice) as the noise floor for every comparison;
- the one-week log of foreign GPU processes;
- writing a model-predicted capacity into the log before measuring it, with a mismatch treated as a bug first;
- server-GPU tenancy (time-slicing vs MPS), kept only as an optional late branch for a course that requires device-level virtualization.

Added by the judge:
- plain acknowledge-on-integrate flow control (A2) as the strongest no-new-idea baseline; the novelty claim must beat it;
- a coupling-activity-fraction check, so a degenerate matched control is caught;
- an emulated aggressor guest, so H2 runs without a second machine;
- a stated warning that H2 follows from TCP fair-share arithmetic, so a null there means a bug;
- re-staging so milestone 1 streams a static .ply and milestone 3 replays synthetic updates built from existing scenes, with no model before M8, as the binding decisions require.

## Recommended direction (full specification)

### Title

Measure First: Streaming Gaussian Scenes to a WebGPU Browser, and Testing Whether the Client's Measured Capacity Belongs in the Sender's Scheduler

### Summary

The student builds a Gaussian-scene streaming pipeline one measured layer at a time.
Each layer answers one question with one number before the next layer is built.

The pipeline:
- A Python server sends Gaussians over a WebSocket.
- A browser page, forked from the MIT-licensed WebSplatter WebGPU renderer, keeps them in a fixed pool of GPU slots, plays the role a page table plays in an OS, and draws at the display's own rate from any viewpoint.

The stages, in order:
1. Measure what a static scene costs to draw.
   This is the noise floor and the null for every later stall claim.
2. Stream an existing trained .ply in chunks.
3. Measure how many Gaussian updates per second the browser can absorb without dropping frames.
   Under the student's material this capacity is the 'GPU budget'.
4. Replay synthetic update sequences derived from existing scenes, with a beacon that makes each update's moment of visibility measurable.
5. Add tc netem and find where first-in-first-out delivery breaks.

Only then is the main question committed, by a rule declared now.
Prior work already publishes each half of the favoured 'joint scheduler':
- deadline-aware network scheduling (arXiv 2606.17212);
- budgeted, coalescing client publication (GaussAnything, arXiv 2609.13859).

What is left is the coupling.
Does feeding the client's measured integration capacity back into the sender's admission decisions help?
The comparisons are:
- against running the two halves independently;
- against plain receive-window flow control, which is the textbook way a slow receiver throttles a sender.

The prediction, declared in advance: for one client the coupling mostly saves wasted bytes.
Its real value is performance isolation when several clients with unequal GPUs share one bottleneck link.
That gives the virtualization story: the guest reports what it can actually use, and the host gives the rest to other guests.

If the browser's capacity never binds at realistic link rates, a declared pivot turns the project into a network-only study, and the report says so.

A feed-forward model (StreamSplat or AnySplat) enters only in the last milestone.
It replaces the trace producer, and its result is labelled a demo.

### Research question

Setting: a server streams a Gaussian scene with persistent IDs to a WebGPU browser over WebSocket.
The scene is static first, then continuously updated.
The browser renders at its own frame rate.

Q0, premise.
How many Gaussian updates per second can the browser integrate without missing render frames?
Integrating means decode, write into GPU slots, and get them drawn.
Is that rate ever below realistic link rates, in any declared client condition?
This is measured in M2.
It decides whether a 'GPU budget' exists for the scheduler to respect.

Q-main.
Committed in M5, only if Q0 says the budget binds.
Should the sender's admission and deadline-feasibility check use the client's measured integration capacity on top of the network delivery rate?
'Better' means one of:
- lower p95 event-to-photon latency;
- fewer render stalls;
- fewer wasted bytes;
- less latency inflation for a victim client sharing a bottleneck with a weak client.

Compared against:
- (i) the same sender and client running independently;
- (ii) the same pair joined only by plain flow control (an acknowledge-on-integrate byte window).

Q-alt.
Committed instead if Q0 says the budget never binds.
At equal bandwidth, which of these lowers p95 event-to-photon latency relative to FIFO?
- sender-side latest-value coalescing;
- earliest-deadline-first (EDF) ordering;
- MoQSplat's Eq. 1 view priority.
And does per-client fair queueing isolate clients on a shared link?

### Hypotheses and refutation thresholds

All thresholds below are fixed now, before any run.

Wire record: 32 B per Gaussian, a u32 ID plus 14 fp16 values (position 3, scale 3, quaternion 4, opacity 1, RGB 3; spherical harmonics degree 0).
Rates are quoted in Gaussians/s and in Mbit/s-equivalent: 50 Mbit/s-eq = 195k Gaussians/s, about 3.3k per frame at 60 Hz.

A stall is a presented-frame interval longer than 2 vsync periods (33.3 ms at 60 Hz).

H0, premise (M2).
It decides whether the GPU half of the scheduler exists.
- Definition: R_max is the highest sustained update rate with p99 frame interval at most 25 ms and no growth in integration lag over 30 s.
- Three client conditions are declared:
  - unthrottled;
  - Chrome CPU throttling at 4x, set through the Chrome DevTools Protocol (CDP) call Emulation.setCPUThrottlingRate;
  - a co-tenant WebGPU burner tab tuned to at most double idle frame time.
- PASS if R_max is at most 50 Mbit/s-eq in at least one condition.
- REFUTED if R_max is above 200 Mbit/s-eq in every condition.
  The coupling question is then dropped, Q-alt is committed, and the write-up states that the browser GPU budget never bound on this hardware.
- Between 50 and 200: the main experiment runs only in the conditions where it binds.
- Before measuring, the student writes a predicted R_max into the log, from the step-0 per-stage costs and the M2 cost model.
  A measured value more than 2x off the prediction means hunting for an instrument bug first.
- Expectation: unthrottled RTX 6000 Ada far above 200, because WebSplatter re-sorts every Gaussian every frame anyway.
  Binding is plausible only under throttling or co-tenant load.
  That is legitimate under the virtualization framing, but it must be reported as induced.

H-sanity (M4).
An instrument check, not a finding, because queueing theory predicts it.
- At offered load / bottleneck rate (rho) of 1.2 or more, FIFO p95 latency grows with elapsed run time: the slope is positive and its CI excludes 0.
- Sender-side latest-value coalescing keeps p95 at rho = 2.0 within 2x of its value at rho = 0.5.
- Failure means the harness is wrong, and nothing downstream runs.

Arms for H1 and H2 (Path A only):
- A1, independent: a coalescing EDF sender paced to the delivery rate, plus a GaussAnything-style budgeted client.
- A2, flow control: A1 plus an acknowledge-on-integrate window W.
- A3, coupled: A1 with sender admission at min(network rate, measured client integration rate) and the deadline check run on that rate.

H1, single client (M6).
Declared as an expected near-null.
- Client-bound regime: A3 cuts wasted bytes by at least 50% versus A1, and the paired p95 event-to-photon difference stays within +/-10%.
- Network-bound regime, the negative control, where A3 equals A1 by construction: every difference within +/-5%.
- Refutation (a): a p95 win of 10% or more in the network-bound regime is treated as an implementation confound.
  Hunt for the bug before reporting anything.
- Refutation (b): a p95 win of 10% or more in the client-bound regime refutes the near-null.
  Before any claim, test the mechanism: measure decode CPU time and main-thread contention.

H2, main hypothesis (M6).
The virtualization result.
- Two guests share one shaped bottleneck, C = 50 Mbit/s unless M4 moves it.
- The victim is the real browser, with trace demand 0.6C.
- The aggressor has the same demand and an integration capacity of 0.2C.
  It is either an emulated guest process that integrates at a declared rate, labelled as such, or a second real browser if one is available.
- Victim inflation = victim p95 with the aggressor streaming / victim p95 with the aggressor idle, minus 1.
- Prediction: A3 reduces victim inflation by at least 30% versus A1, and the aggressor's own p95 rises by no more than one RTT plus one frame.
- REFUTED if the paired median reduction is below 15%, or its bootstrap 95% CI (n = 10 paired runs) includes 0.
- Negative control: aggressor capacity 0.5C, where its fair share equals what it can use.
  A win of 10% or more there withdraws the mechanism claim.
- This effect is largely predicted by TCP fair-share arithmetic.
  A1's greedy reader lets the aggressor hold about 25 of its usable 10 Mbit/s, leaving the victim at rho of about 1.2.
  So the value of H2 is the measured magnitude in a real browser stack, not the surprise.
  A null here is a bug until shown otherwise.

H-coupling, the novelty claim (M6).
The coupling is claimed to add something beyond flow control only if A3 beats A2 by 10% or more, with the paired CI excluding 0, on either victim inflation or the aggressor's own p95 staleness.
Otherwise the report states that the coupling reduces to plain flow control, which is a reportable negative result.
- Expectation: A2 captures most of the isolation gain, because both arms cap the aggressor at its consumption rate.
  A3 should differ mainly in staleness when W exceeds one RTT x client rate.

### Novelty against the closest work

Short version: no mechanism here is new.
The contribution is a measured answer: does coupling the sender to the client's measured run-time capacity matter, in which regime, by how much, and beyond plain flow control?
- Measured in a browser WebGPU client, as event-to-photon latency under tc netem, plus a multi-client isolation measurement.
- That is a solid course contribution, workshop-level if H2 and H-coupling both hold, and still reportable if they do not.

Verification status:
- Read and checked by a reader/verifier pair: 2604.02851, 2506.08862, 2603.14232, 2512.08478, 2609.18624, 2602.03207, 2511.06046, and LMG (metadata, symbol table and code only; full text unread).
- Read only by the sweep: GaussAnything 2609.13859, 2606.17212, AirGS, L3GS, CAGS, MD2G-Cast, TIGAS.
- Not read at all: LTS, Vega, SGSS.
- The novelty claim depends most on the sweep-only and unread ones, so reading them is week-1 work.

Streaming Real-Time Rendered Scenes as 3D Gaussians (arXiv 2604.02851, preprint 2026):
- Already does the live server-optimised snapshot plus per-attribute delta loop with persistent IDs, to a native Unity client.
  We do not claim that loop.
- Its schedule is fixed per attribute, plus an unspecified residual gate.
  It does no deadline-, bandwidth-, view- or GPU-budget-driven selection, and measures no latency, stalls or client cost (its Sec. 5 calls these future work).
- What we add: a browser client, adaptive admission, and latency and stall measurement.

GaussAnything (arXiv 2609.13859, 2026, no venue; sweep-read):
- Already has stable-ID insert/remove/update, a preallocated slot buffer, latest-value coalescing, a slack-adaptive per-frame publication allowance and an age escape.
- Per the sweep it has no network emulation, no FIFO comparison, and no single end-to-end latency.
  Its client capacity is static configuration, not a measured signal fed back to the sender.
- Our arm A1's client half reimplements it and is not claimed.

Renderable Partial Representations (arXiv 2606.17212, 2026, no venue; sweep-read):
- Already has deadline-probability x utility-per-byte network scheduling against layer-order and DASH-style arms, on a shared congestion controller with paired seeds.
- Its receiver is an emulator with no model of client GPU or CPU, its content is offline, and it has no browser.
- We copy its evaluation design and claim nothing about its scheduler.

MoQSplat (IEEE MMSP 2026, arXiv 2609.18624):
- Static scenes.
  Eq. 1 distance-plus-angle priority.
  A VRAM watermark that ranks by the same score but never feeds back into what is sent.
- No latency or baseline evaluation.
  Its public code is a TCP placeholder.
- Eq. 1 becomes our view-aware arm in Q-alt.

WebSplatter (ACM MM 2026 Main Track, arXiv 2602.03207) and Visionary (arXiv 2512.08478, preprint):
- Browser renderers.
  WebSplatter is static.
  Visionary runs inference and render in lockstep with a forced GPU readback each inference.
- We claim nothing about rendering.
  WebSplatter is our base.

AirGS (INFOCOM 2026, arXiv 2512.20943; sweep-read): ID-based Gaussian diffs with bandwidth-driven pruning on offline 4DGS content.
No viewport and no client budget.

L3GS (MobiCom 2025, arXiv 2504.05517; sweep-read): knapsack view-utility scheduling for static scenes, SSIM only.
We borrow its pre-load ceiling and oracle-predictor patterns.

MD2G-Cast (ACM MM 2026 per arXiv comments, arXiv 2608.10020; sweep-read):
- Multi-user admission at MoQ relays using receiver capability, but capability taken from static device specifications.
- Our difference: capacity measured at run time under contention, Gaussian content, and an isolation metric.

LMG (ACM MMSys 2026, DOI 10.1145/3793853.3795764, NTHU Hsu group): tile urgency scheduling for static scenes.
Its symbol table has no client-budget symbol, but the full text is unread.
Read it before claiming LMG lacks a client budget.

LTS (ACM MMSys 2025 Best Paper, NTHU) and Vega (MobiCom 2025): unread.
LTS has a live mode with freeze metrics; Vega does deadline-driven client GPU work.
Either could overlap.
Both must be read through the NTHU library in weeks 1-2.

Spark SplatPager (local .d.ts only): an LRU GPU page pool for static splats.
The 'virtual scene cache' itself is therefore prior art and is not claimed.

StreamSplat (ICLR 2026 poster, arXiv 2506.08862): a producer only.
Used in M8.
Not to be confused with arXiv 2608.01659.

### Virtualization angle

The primary framing is performance isolation of guests sharing a host resource.
It is backed by a measurement (H2), not only an analogy.

The guest (each browser client):
- Its WebGPU slot pool is a capacity-limited physical memory: slot = page frame, ID-to-slot map = page table, insert/evict = page-in/page-out, opacity-0 tombstone = invalidated page.
- The student builds this in M1-M3 and should explain it in these terms.
  It is descriptive, not claimed as new; SplatPager and MoQSplat already do it for static scenes.
- The guest's per-frame integration allowance is its usable compute share.
  It shrinks when a co-tenant (another tab or process) contends for the same GPU or CPU, which is the noisy-neighbour effect on the client side.

The host (the streaming server's admission loop):
- It divides one shared uplink among guests.
- Coupling is the analogue of a balloon driver or a cgroup demand cap: the guest reports what it can actually consume, and the host hands the remainder to others in a work-conserving way.
- Plain flow control (arm A2) is the analogue of the host learning the same thing only through back-pressure.
  The A3-vs-A2 contrast asks whether the explicit report is worth having.

Measured outcomes and their course names:
- victim p95 inflation and Jain's fairness index: noisy-neighbour isolation;
- the 0.5C negative control: when fair sharing alone is already enough;
- the aggressor's own p95 cost: the price of work conservation.

Optional extensions (M7), chosen by the instructor's answer:
- (a) Capacity pressure.
  Client capacity set below the working set, giving a working-set curve and miss-ratio curves for LRU vs view-distance eviction, with Belady's optimum as a bound (from C).
- (b) Literal device virtualization.
  After the live producer exists, run several producer tenants on one server GPU and compare default time-slicing with MPS (Multi-Process Service), as in B.
  MPS runs as a normal user; MIG is not available (Mode N/A on this card), nor is vGPU, Docker or root.

Ask the instructor in week 1 which of these, if any, the course requires.

### System design

The design is one server process, one browser page and one runner script.
Arm differences are config flags, not separate code paths.
WebSplatter is the only new external dependency, and adding it needs the user's approval.

1. Trace producer (offline Python, no model).
   It builds update sequences from existing trained scenes, as a list of ticks of (id u32, op insert|update|delete, 14 fp16 attributes), with tick timestamps:
   - T-static: one existing ScanNet++ 3DGS .ply as a stream of inserts.
     Used in M1.
   - T-reveal: voxel pages of a static scene appear in camera-discovery order along a pose path.
     Mimics a growing reconstruction.
   - T-refine: a fraction f of resident IDs gets attribute changes every 100 ms tick.
     The knob is the probability that an ID is updated again within 200 ms, set to 0.1 or 0.5; that probability is the coalescing opportunity.
   - T-churn: a region's Gaussians are deleted and replaced by new IDs each tick.
     Mimics a feed-forward model with no Gaussian identity.
   - Beacon block: a fixed block of Gaussians whose colours encode the tick number in binary, so a captured frame shows which tick is on screen.
   - Rate is set exactly in Gaussians/s, so rho can be set exactly.

2. Server: one asyncio process using the websockets package already in the puffin-world env.
   - Each client has a pending dict mapping ID to its latest record.
     That dict is sender-side latest-value coalescing for free; FIFO uses a plain queue instead.
   - An admission loop runs every 5 ms with a byte budget = estimated rate x dt.
     - The rate comes from client acknowledgements.
     - A2 adds a byte window: unacknowledged-and-unintegrated bytes at most W.
     - A3 uses min(network rate, client integration rate).
   - Ordering is EDF with deadline t_event + D, D = 250 ms declared.
     In Q-alt, MoQSplat Eq. 1 is the view variant.
   - Intermediate values predicted to miss D are left to be superseded rather than sent.
     The final value is always sent.
   - Records are packed into 64 KB binary messages with a header of (tick, t_event, count).
   - TCP_NOTSENT_LOWAT and a small write limit stop stale bytes piling up in the kernel, where they can no longer be recalled.

3. Browser client: a fork of WebSplatter (MIT, HEAD 16bb06e).
   - Slot buffers are preallocated to a fixed capacity, e.g. 1M slots, because WebSplatter sizes every buffer at construction.
   - A JS Map from ID to slot, plus a free list.
     Delete writes opacity 0, since WebSplatter culls opacity at or below 1/255.
   - The reorder path (maybeReorderAfterSubmit) is disabled, because it permutes indices.
   - A Worker owns the WebSocket and parses messages.
   - On each requestAnimationFrame (rAF) the main thread takes up to 'allowance' records, issues queue.writeBuffer per contiguous slot run, and renders.
   - The allowance is fixed, or adapted to display slack (the GaussAnything mechanism, used in A1-A3).
   - Every 100 ms the client sends upstream: bytes received, bytes integrated, measured integration rate, and pose.
   - A ring buffer logs per-update stage timestamps: received, decoded, written, render submitted, onSubmittedWorkDone, next rAF.

4. Network: on the server egress, tc htb sets the rate and netem sets delay and loss.
   A one-time sudo grant is needed.
   Each session is verified with iperf3.
   If loopback is used, the MSS (maximum segment size) is clamped and checked with tcpdump.
   One shared qdisc gives the shared bottleneck for H2.

5. Runner: one Python script per log entry, built on the stdlib CDP harness pattern of fmri3d-viewer/scripts/wz2_browser_probe.py.
   - It starts the server with an arm config, drives a real GPU-backed Chrome (never headless, which falls back to SwiftShader), replays a scripted 6-DoF pose path, runs 120 s and drops the first 20 s.
   - It writes one JSONL per run, stamped with git hash, trace sha256, arm, seed, netem settings, chrome://gpu summary, and nvidia-smi before and after.

6. M8 only: a model process replaces the trace producer.
   StreamSplat slot identity is (keyframe mod 2, grid cell); AnySplat is the alternative.
   Scheduler and client are unchanged.

### Evaluation

Independent variables, introduced in this order so that early milestones have one or two arms:
- Step 0-M3: scene size N, per-frame allowance k, client condition, trace type.
  FIFO only.
- M4: arm (FIFO vs sender coalescing), rho in {0.5, 0.8, 1.0, 1.2, 2.0}, loss {0, 1}%.
- M6 Path A: arm (A1, A2, A3) x regime (network-bound, client-bound, step trace 50->10->50 Mbit/s with 30 s phases) x, for H2, aggressor capacity ratio {0.2, 0.5}.
  Bounds: FIFO and oracle.
- M6 Path B: FIFO, coalescing, coalescing+EDF, coalescing+Eq.1 view order, per-client fair queueing.
- The full rate x RTT x loss grid (rate {10, 25, 50} Mbit/s, RTT {20, 80} ms, loss {0, 1}%) is run only late, and only in the cells M4 shows to be informative.

Matched vs bounding controls:
- Matched (one factor): A3 vs A1 (any coupling), and A3 vs A2 (explicit capacity report vs plain flow control).
  In H2, each arm with vs without the aggressor.
- Bounding: FIFO push with immediate integration (the source material's baseline), and an oracle (unshaped loopback, idle GPU, immediate integration).
  Both are named as bounds, never as controls.
- Negative controls: the network-bound regime (A3 equals A1 by construction), and aggressor ratio 0.5.
- Degeneracy check: every run logs the coupling-activity fraction, i.e. the share of admission ticks where the client rate is below the network estimate.
  Below 0.3 in the client-bound regime, the matched control is dead and the regime is redefined before any test run.

Equal budget:
- Every arm with knobs gets at most 9 configurations, tuned on a dev trace built from a different scene: pacing gain x D for A1-A3, the same allowance limits for A1-A3, and W for A2.
- One criterion, declared before tuning: lowest p95 event-to-photon latency subject to stalls no worse than FIFO.
- The metric not selected on (wasted bytes) is reported anyway.
- A winner on a grid edge means the grid is extended.
- Parameters are frozen and committed before test traces are opened.

Metric, direction and null, one line each:
- p95 event-to-photon latency, lower is better; nulls are the matched arm and the A/A spread of the same arm run twice.
- Stalls per minute (interval > 33.3 ms), lower is better; null is the step-0 static rate at the same N on the same client.
- Wasted bytes (values superseded before integration / total bytes), lower is better; null is A1.
- Time-averaged staleness of the displayed state, lower is better; null is A1.
- Deadline-miss rate (visible after t_event + D), lower is better; null is A1.
- Victim inflation and Jain's fairness index (H2); null is A1, plus the 0.5 negative control.
- Guardrail, reported and never selected on: PSNR of sampled displayed frames against a gsplat render of the latest server state at the same pose.
- Reported numbers from other papers (e.g. 92.5 Mbit/s in 2604.02851) are motivation only, never nulls.

Event-to-photon latency, measured as follows:
- For an update u of ID i released at t_event, latency is the time to the first presented frame whose GPU state holds i at u's version or a later one.
  'Or later' keeps the metric fair to coalescing.
- Presented = the first rAF timestamp after onSubmittedWorkDone of the render that used it.
- Clocks: same host means a shared wall clock.
  If the client is win-desktop, the offset comes from WebSocket ping-pong (min RTT / 2), and the residual bound is reported.
- Validation (M3): a 240 fps phone films the monitor showing the client and a minimal reference page that draws the tick code at t_event through no pipeline, so display latency cancels.
  Over at least 50 beacon events, report the mean and sd of (proxy - camera).
- If the sd is above 8 ms, the main results use the camera.
- True glass-to-glass (physical event in front of a real camera -> visible in browser) is measured only in M8, with at least 30 filmed events per arm.
- Latency is also decomposed by stage, and the stage sum must be within 5% of the total.

Workloads: T-static, T-reveal, T-refine (coalescing opportunity 0.1 and 0.5) and T-churn, built from local ScanNet++ 3DGS scenes (527k-794k Gaussians).
A different scene is the dev trace.
Pose paths are replayed identically across arms: an orbit and a recorded walk, or EyeNavGS / Rekenar NTHU traces once their licence is checked.

Network: iperf3 within 5% of the setpoint before each session.
Optionally one public cellular throughput trace replayed as 1 s rate steps (late only).

Hardware: in trace mode the server needs only the Xeon 6542Y workstation, with no server GPU.
The client is the machine that passes step 0: win-desktop (dedicated RTX 6000 Ada) or the workstation on display :20 (shared GPU, logged).
The H2 aggressor is an emulated guest process or a second browser; if it is a second browser, first check that it does not raise the victim's frame time.

Runs and spread:
- n = 5 for exploratory milestones, n = 10 paired runs for M6.
- Pairing is by netem seed and trace start offset, with arm order randomised within blocks.
- The unit is the per-run p95.
- Report n, median and IQR, and a paired bootstrap 95% CI (10k resamples) for each matched contrast.
- Exclusion rule, declared now: drop runs where foreign GPU utilisation on the client exceeds 10%, and report the count.
- netem loss cannot be seeded, so the n runs absorb that randomness.
- M6 cost: about 10 h unattended (Path A: 5 arm configs x 3 regimes x 10 x 2.3 min, plus H2 about 3.5 h).

### Milestones

#### Weeks 1: 0. Static render null and noise floor (step 0 is a measurement).

   Run WebSplatter's eval=2 (wall-clock per frame) and eval=3 (per-stage GPU timestamps), each with 100 warm-up and 500 measured frames.
   Add a roughly 10-line rAF interval logger.
   Run this in real GPU-backed Chrome on both candidate client machines (workstation display :20, and win-desktop).
   Scenes: subsets of a local ScanNet++ 3DGS scene at N = 0.25, 0.5, 1 and 2 M Gaussians.
   Repeat one configuration as an A/A pair.
   Start a 1-week log of nvidia-smi --query-compute-apps every 30 s on the workstation.
   In parallel, start reading GaussAnything, 2606.17212, LTS, LMG and Vega in full (NTHU library for the ACM ones), and email the Hsu group.

Number produced:

For each N and machine:
- frame interval p50 and p99, stalls per minute, and per-stage ms (preprocess, sort, raster), over 5 repeats;
- A/A spread of p99;
- whether WebGPU works on Linux Chrome here (yes or no);
- hours per week with no foreign GPU process on the workstation.

What the student learns:

- What a frame budget, vsync and a stall are.
- That sort cost grows with N on every frame, whether or not anything changes.
- What a noise floor (A/A spread) is, and why a shared GPU makes it wide.
- Why headless Chrome numbers are meaningless: it falls back to SwiftShader.

Exit criterion:

The measurement client is the machine whose A/A p99 spread is within 10%.
If neither machine qualifies, stop and log a failure entry.
This step sets the stall null for every later stall claim.
If WebGPU fails on both machines, switch the client to Spark 2.1.0 (WebGL2), relinking its dangling scene.ply first, and log the failure as a failed approach.

#### Weeks 2-3: 1. Stream an existing static .ply in chunks (binding decision 1).

- The server reads a local ScanNet++ 3DGS .ply, converts it to 32 B records (spherical harmonics degree 0), and sends them as 64 KB binary WebSocket messages.
- The client Worker parses them.
  The main thread writes them into a preallocated slot buffer through an ID-to-slot Map and a free list, at most k records per frame.
- Two arms: k unlimited (apply everything at once) and k = 16k.

Number produced:

For each of the two k settings, over 5 runs:
- time to first frame;
- time to complete;
- stalls per minute during the load, against the step-0 null.

Correctness: the image difference between the streamed render and a local-file render of the same SH0 scene, at 10 poses.

Cost of SH0: PSNR of the SH0 render against the full-SH render at the same 10 poses.

What the student learns:

- Binary WebSocket messages, ArrayBuffer/DataView and fp16 packing.
- Worker versus main thread.
- queue.writeBuffer.
- Why a slot buffer with an ID map is a page table.
- Why applying a whole chunk in one frame can stall rendering.

Exit criterion:

The streamed render must be pixel-identical to the local load.
Otherwise it is a bug, and it gets fixed first.
This decides the chunk size.
It also decides whether the SH0 wire format is acceptable.
If SH0 costs more than 2 dB, that becomes an open question for the student: bytes versus quality.

#### Weeks 4-5: 2. Client integration capacity: H0, the premise test.

- An in-page injector, with no network, updates resident IDs at increasing rates.
- Sweep the per-frame allowance k over {1k, 4k, 16k, 64k, 256k}.
- Run three client conditions: unthrottled; CDP CPU throttling 4x; a WebGPU burner tab tuned to at most double idle frame time.
- Write the predicted R_max into the log before measuring.

Number produced:

- Cost model a + b*k for writeBuffer ms and for frame interval, with residuals.
- R_max in Gaussians/s and in Mbit/s-equivalent for each condition, n = 5, with spread.
- Prediction versus measurement.

What the student learns:

- What a GPU budget actually means for this client: update throughput, or resident size.
- What queue.writeBuffer costs compared with the sort that runs every frame anyway.
- How contention from a co-tenant shrinks a guest's usable share.

Exit criterion:

Apply the H0 rule exactly as declared:
- Some condition at or below 50 Mbit/s-eq: Path A, the coupling study.
- Every condition above 200: Path B, the network-only study (Q-alt); the coupling claim is dropped in writing.
- In between: restrict the study to the conditions that bind.
If the prediction and the measurement differ by more than 2x, hunt for an instrument bug before applying the rule.

#### Weeks 6-8: 3. Replay synthetic updates and build the latency instrument.

- Traces: T-reveal, T-refine at two coalescing levels, and T-churn, built from existing scenes.
  Insert, update and delete are all keyed by ID.
- A beacon block encodes the tick number.
- FIFO only, on unshaped loopback.
- Camera validation of the latency proxy.

Number produced:

Per trace, over 5 runs:
- FIFO event-to-photon latency, p50 and p95;
- the stage decomposition: released, sent, received, decoded, written, submitted, done, presented;
- the clock-offset bound;
- proxy minus 240 fps camera, mean and sd, over at least 50 beacon events.

Also, per trace: the volume in Gaussians/s and Mbit/s-eq, placed against R_max from milestone 2.

What the student learns:

- What Gaussian identity is, and why deltas need it.
- Why a feed-forward model breaks identity (T-churn) while an optimiser keeps it.
- Tombstones.
- Latency as a sum of stages plus the queues between them.
- Clock domains, vsync and compositor delay.
- Why network arrival is not the same as visibility.

Exit criterion:

- The stage sum must be within 5% of the total.
- If the camera sd is 8 ms or less, the proxy is used for main results.
  Otherwise main results use the camera, and the run count is replanned.
- If the coalescing opportunity in a trace is below 0.05, record that coalescing has little to act on there.

This milestone decides the latency instrument.

#### Weeks 9-10: 4. Network emulation and the FIFO knee.

- Network shaping with tc htb and netem, verified by iperf3.
  Needs the user's sudo.
- Two arms only: FIFO, and sender-side latest-value coalescing with TCP_NOTSENT_LOWAT.
- Sweep rho in {0.5, 0.8, 1.0, 1.2, 2.0} using T-refine at 25 Mbit/s and RTT 20 ms.
- One extra cell: 1% loss at rho = 0.8.

Number produced:

Over 5 runs per cell:
- p95 latency against rho and against elapsed time;
- server backlog growth;
- the knee rho*;
- latency inflation under 1% loss, which measures head-of-line blocking directly instead of assuming it.

Output: the H-sanity verdict.

What the student learns:

- How latency explodes as utilisation approaches 1.
- What tc htb and netem actually do.
- How TCP send buffers behave.
- Why coalescing must happen before bytes enter the socket.
- What MoQSplat's head-of-line-blocking argument costs over TCP.

Exit criterion:

H-sanity must hold.
If it fails, the harness is wrong: debug it before anything else.
This milestone fixes the bottleneck rate C used in M6, chosen at or just above rho* for realistic traces.

#### Weeks 11-12: 5. Commit the question, pre-register, and tune.

   This is the exploratory lane and is labelled as such.
- Apply the M2 rule, and log the reading notes on whether LTS, LMG, Vega or GaussAnything already do capacity-coupled admission.
- Write a one-page pre-registration before any test trace is opened.
- Implement arms A1, A2 and A3 (Path A), or the Q-alt arms (Path B), as config flags.
- Tune every arm on the dev trace.

Number produced:

For each arm:
- the tuning surface (at most 9 configurations, n = 3 per point);
- the winning configuration and the spread of the runners-up;
- an edge-of-grid flag.

Also: the coupling-activity fraction in the client-bound regime.

What the student learns:

- How to give baselines an equal budget, and what baseline nerfing and harness hacking look like in practice.
- Why the textbook flow-control arm (A2) must be in the comparison.
- How to write a pre-registration.

Exit criterion:

- Parameters are frozen and committed.
- The pre-registration is in the log.
- Coupling-activity fraction is at least 0.3; otherwise redefine the regime.
- The novelty statement is updated after the reading.
  If a prior paper already answers the question, the project narrows to a browser replication and says so.

#### Weeks 13-16: 6. Main pre-registered matched experiment (tentative; its form is fixed by M2 and M5).


Path A:
- H1 in the client-bound regime, with the network-bound regime and the step trace as controls.
- H2 with two guests on one shaped bottleneck, at aggressor ratios 0.2 and 0.5.
- Arms A1, A2 and A3, plus the FIFO and oracle bounds.

Path B: the Q-alt arms, plus per-client fair queueing.

Number produced:

Paired differences with bootstrap 95% CIs, n = 10, for:
- A3 vs A1 and A3 vs A2: p95 event-to-photon latency, stalls per minute, deadline-miss rate, wasted bytes, staleness;
- victim inflation and Jain's index.

Also: latency CDFs and a forest plot of the paired differences.

Outputs: the H1, H2 and H-coupling verdicts.

What the student learns:

- The difference between a matched control and a bounding one.
- Performance isolation and work conservation, in numbers.
- Whether an explicit capacity report beats plain back-pressure.
- How metrics trade against each other: latency, bytes, staleness, stalls.

Exit criterion:

- Each verdict is logged by its declared rule, whichever way it falls.
- A surprising win triggers a bug hunt and the decode-CPU mechanism test before anything is written.
- If the 0.5 negative control shows an effect of 10% or more, the mechanism claim is withdrawn.

#### Weeks 16-18: 7. Virtualization extension (tentative; choose one according to the instructor's answer).

- (a) Capacity-pressure cache: set client slot capacity to {0.25, 0.5, 1, 2} x the measured 10 s working set.
  Compare LRU and view-distance eviction, with Belady's optimum as a bound.
- (b) Device virtualization, only if required: several producer tenants on one server GPU, comparing default time-slicing and MPS.
  This needs a live producer, so it comes after M8, or uses a synthetic GPU load as the tenant.

Number produced:

Option (a):
- the working-set curve W(tau);
- miss ratio, refetch bytes and p95 latency against capacity, per policy, n = 10.

Option (b): for each sharing mechanism,
- each tenant's producer p95;
- the browser's event-to-photon p95.

What the student learns:

Option (a): working sets, page replacement, Belady's optimum as a bound, and miss-ratio curves, all measured on the student's own system.

Option (b): CUDA contexts, the difference between temporal and spatial sharing, and what an SM cap does and does not isolate.

Exit criterion:

The curves or the table are logged with CIs.
This is the first milestone to drop if time runs short.

#### Weeks 18-21: 8. Feed-forward producer swap-in, a labelled demo, and the write-up.

- StreamSplat (ICLR 2026, arXiv 2506.08862, with Depth Anything V2 inline) or AnySplat replaces the trace producer, on a CVLab GPU or the workstation.
- Slot identity comes from (keyframe mod 2, grid cell).
- Scheduler and client are unchanged.

Number produced:

- Model inference time per keyframe on the GPU actually used.
- True glass-to-glass latency, p50 and p95, over at least 30 physical events filmed at 240 fps.
  Comparison: A1 against the best arm, at one network condition.
- The gap between the latency trace replay predicts and the live value.

What the student learns:

- How the inference rate, the network update rate and the render rate interact when a real model is the source.
- How to build slot identity for a model that has no native IDs.
- Why this run is a demo and not a matched experiment.

Exit criterion:

- The gap between trace-predicted and live latency is reported.
- If that gap exceeds 2 sd of the camera validation plus one frame, the trace-mode conclusions are scoped down in the write-up.
- Figures are made with the nature-figure skill.
- Every quoted number names its log entry and the command that produced it.

### Reuse

Local, already on disk (no installs):
- /home/intern_2603056/Desktop/GS_playground/envs/puffin-world-c6d9ae5-py310-cu126: torch 2.7, gsplat 1.5.3, websockets, aiohttp, cv2 and plyfile.
  Covers the server, the trace producer and reference renders.
- /home/intern_2603056/Desktop/GS_playground/src/gs_playground/gs/ply.py and render.py: PLY load/save with subsets, and a gsplat render wrapper.
  Used for traces, the SH0 cost check and the PSNR guardrail.
- /home/intern_2603056/Desktop/fmri3d-viewer/scripts/wz2_browser_probe.py: a stdlib Chrome DevTools Protocol client.
  It is the base for the runner and for CPU throttling.
- /home/intern_2603056/Desktop/PairGS_ours/scripts/serve_demo.py: a stdlib server skeleton.
  It has no WebSocket, so that layer has to be added.
- /mnt/nchc-2603056/datasets/ScanNet++_3dgs/*/point_cloud.ply: 6 static scenes, 527k-794k Gaussians each.
  Copy them to local disk first, because the CIFS mount is slow.
- Spark 2.1.0, vendored in /home/intern_2603056/Desktop/PairGS_ours/viewer/vendor/: the WebGL2 fallback.
  Its scene.ply symlinks are dangling and must be relinked.
- Prior art for the cache, in /home/intern_2603056/Desktop/archive/EviPartGS/web_viewer/node_modules: the SplatPager .d.ts files.
- For M8: the AnySplat env (/home/intern_2603056/Desktop/event_camera_2DGS/envs/anysplat) with its local weights.

External (each needs the user's approval as a dependency):
- WebSplatter, MIT licence, HEAD 16bb06e (Zenodo 21798977).
  - Upstream: https://github.com/websplatter/WebSplatter (the audit read commit 16bb06e).
  - Its eval=2 and eval=3 modes give step 0 directly.
  - Opacity at or below 1/255 is culled, so opacity 0 works as a delete tombstone.
- StreamSplat code and checkpoint, for M8 only.
  There is no licence file, so use it only for private coursework.
  Depth Anything V2 weights are CC BY-NC.

Patterns reimplemented, not copied:
- MoQSplat Eq. 1 (its repo declares MIT, but there is no LICENSE file);
- GaussAnything's slack-adaptive publication, from the paper text;
- the paired-trial evaluation design of 2606.17212;
- the pre-load ceiling and oracle patterns of L3GS;
- the fixed per-attribute schedule of 2604.02851, as an optional extra baseline.

Pose traces: EyeNavGS (arXiv 2506.02380, ACM MM 2025), or the Rekenar/GaussianAdaptiveStreamer TestMovements/NTHU traces.
Check the licence first.

### Risks

1. The premise may fail.
   On an RTX 6000 Ada, integration may never bind.
   M2 tests this with a declared pivot to Path B, so the project still produces a result either way.
   Contention that is induced deliberately must be reported as induced.
2. Novelty.
   GaussAnything and 2606.17212 have only been read by the sweep.
   LTS, LMG and Vega are unread, and the NTHU Hsu group may already have this.
   The reading starts in week 1, and M5 can narrow the claim.
3. The coupling may reduce to plain flow control (A3 roughly equal to A2).
   This is a declared, reportable negative result.
   It is the most likely outcome for the novelty claim.
4. H2 is close to mechanical.
   A null result there is a bug until shown otherwise.
   The 0.5 negative control guards the mechanism claim.
5. Implementation confound.
   The coupled arm adds code paths.
   The network-bound regime and the degeneracy check (coupling-activity fraction) catch this.
6. Measurement.
   - rAF is not photon time, so it is checked against the camera in M3.
   - Headless Chrome is invalid (it falls back to SwiftShader).
   - Visionary reports a Linux Chrome WebGPU fp16 bug.
     Step 0 detects it, with win-desktop and Spark as fallbacks.
   - A cross-machine clock adds offset error.
     Its bound is reported.
7. Shared hardware.
   The workstation GPU was 87-100% busy.
   Mitigations: the step-0 machine choice, randomised arm order, and the foreign-utilisation exclusion rule.
8. Infrastructure.
   - tc and netem need the user's sudo.
     Docker and user namespaces are unavailable, and CVLab machines cannot shape traffic.
     The fallback is a userspace shaping relay, labelled as such.
   - On loopback, the 64 KB MTU distorts loss unless the MSS clamp works.
     That is checked with tcpdump.
9. Scope.
   - Transport is TCP and WebSocket only.
     QUIC and WebTransport are named as out of scope.
   - SH0 records undercount bytes relative to full-SH systems.
   - H2 applies only to shared bottlenecks.
   - An emulated aggressor guest is labelled as emulated.
10. Live model.
    Feed-forward models have no native IDs.
    StreamSplat's 49 ms was measured on an A100 and excludes depth estimation, and it adds at least one keyframe interval of latency.
    So M8 is a demo.
11. Pace.
    M7, then M8 and then H2 are dropped in that order.
    Milestones 0-6 with H0, H-sanity and H1 still answer the main question for a single client.

### Compute

Milestones 0-7, trace mode:
- Server: the Xeon 6542Y workstation CPU only.
  No server GPU is needed.
- Client: one GPU-backed display, whichever passes step 0.
  Either win-desktop (RTX 6000 Ada, dedicated, Chrome on Windows) or the workstation on display :20, used only when idle and logged on every run.
- The H2 aggressor: an emulated guest process (no extra hardware) or a second browser machine such as a laptop.
- Unattended wall clock: M4 about 3 h, M5 tuning about 3-4 h, M6 about 10 h.
  All of it fits in overnight blocks.

Optional (b) in M7, and M8:
- One GPU with at least 16-24 GB for the producer.
  A CVLab RTX 4090 is preferred, after a fresh status.sh snapshot and the user's approval.
- Do not use the 5090s for AnySplat: its torch 2.2 cu121 build lacks sm_120.
- StreamSplat's memory footprint on 48 GB is unverified.
- If the model and the browser share one box, that contention must be logged.

Other:
- A one-time sudo grant for tc on the workstation, needed by M4.
- iperf3.
- A phone that records 240 fps slow motion, for M3 and M8.
- A few GB of traces and JSONL logs on local disk.

## Open questions for the student or instructor

1. Instructor: does the course accept a virtualization angle built on resource isolation and paging analogies?
   That means guest/host, a balloon-style capacity report, a page-table slot pool, and noisy-neighbour isolation measured in H2.
   Or does it require literal hypervisor, VM or device virtualization?
   If the latter, M7 option (b) (MPS vs time-slicing on the server GPU) or VM-hosted clients must be added.
   This machine has no vGPU, MIG, Docker or root.
2. Approve adding WebSplatter (MIT, HEAD 16bb06e) as the client base, or start on Spark 2.1.0 WebGL2.
   This is a dependency decision.
3. Which machine is the measurement client?
   Win-desktop has a dedicated GPU but a cross-machine clock.
   The workstation display :20 shares a clock with the server but its GPU is shared and often busy.
   Step 0 gives the numbers; the choice is yours.
4. Will you grant a narrow sudoers rule for tc/netem on the workstation, or run a helper script yourself each session?
   M4 cannot start without one of these.
5. Wire format: spherical harmonics degree 0 only (32 B per Gaussian, fewer bytes, some view-dependent colour lost), or keep higher-order SH?
   M1 measures the PSNR cost; you decide what is acceptable.
6. Application deadline D (declared as 250 ms) and the target use case (telepresence vs monitoring).
   These change which latency counts as a miss.
7. For H2: is an emulated aggressor guest process, labelled as such, acceptable for the course?
   Or must both guests be real browsers, which needs a second machine?
8. Should you contact the NTHU Cheng-Hsin Hsu group (LTS, LMG, EyeNavGS, TIGAS) before building, to rule out overlap and to ask about LTS code and the NTHU pose traces' licence?
9. Is a live feed-forward demo (M8) required for the grade, or is the trace-mode study enough?
   Which model: StreamSplat has no licence file (private coursework only); AnySplat is local but slow, at about 0.3 Hz.
10. Is a CVLab GPU acceptable for the producer in M8, subject to a status.sh snapshot and your approval at that time?
11. What form must the course deliverable take (report, demo, poster, paper)?
    This sets how much of M7 and M8 is worth the time.

## The three original proposals

### Systems contribution. The core is the joint network-GPU, deadline-aware, GPU-budget-aware update scheduler. After checking it against the verified papers, it is narrowed to the one piece none of them has: Capacity-Coupled Admission for Live Gaussian Scene Streaming to the Browser: When Does the Client's GPU Budget Belong in the Sender's Scheduler?

Most of the favoured scheduler already exists in pieces, so the honest starting point is this.
Deadline-aware network scheduling that ranks by render utility per byte, compared against layer-order and DASH-style arms, is in arXiv 2606.17212 (offline dynamic Gaussian content, emulated transport).
GPU-budget-aware client publication is in GaussAnything (arXiv 2609.13859, a native OpenXR headset): it has stable-ID deltas, latest-value coalescing, a per-frame work allowance that adapts to display slack, and an age escape.
View-aware priority is in MoQSplat (MMSP 2026, Eq. 1), L3GS (MobiCom 2025) and LMG (MMSys 2026).
Multi-user admission that accounts for receiver capability exists for point-cloud video in MD2G-Cast (arXiv 2608.10020), but it takes capability from static device specifications.
None of them closes the loop.
None feeds the client's measured, run-time integration capacity back into the sender's admission and deadline-feasibility decisions.
None tests whether that coupling helps compared with simply running a network-paced sender and a client-side publication scheduler side by side.

This project builds a trace-driven pipeline for streaming a live Gaussian scene: a Python WebSocket server and a WebGPU browser client forked from WebSplatter.
It reimplements the prior mechanisms as baseline arms.
It isolates the coupling with one matched control that differs only in whether the sender sees the client's capacity.

My prediction, stated so that it can be wrong: for a single client the coupling buys little latency and mostly saves wasted bytes.
Its real value is performance isolation when several clients with unequal GPUs share one bottleneck link.
That is also the virtualization story: the guest reports what it can actually use, and the host gives the slack to others.

Step 0 measures whether the browser's integration budget ever binds at realistic link rates.
If it does not, the GPU half is dropped and the report says so.

Research question:

Setting: a server streams a continuously updated Gaussian scene with persistent IDs to WebGPU browser clients over WebSocket.

Question: should the sender's admission and deadline-feasibility check use each client's measured integration capacity (Gaussians per second it can decode, upload and publish without missing render frames) on top of the network delivery rate?

What "better" means: lower event-to-photon latency, fewer render stalls, fewer wasted bytes, or better isolation between clients.

What it is compared against: the same two mechanisms running independently.
One is a network-paced sender with sender-side coalescing.
The other is a GaussAnything-style client publication scheduler.

Regimes to test:
- network-bound;
- client-bound;
- a link whose rate steps up and down;
- two clients with unequal client capacity sharing one bottleneck link.

### Virtualization-course angle: Tenants per GPU: Consolidation and Performance Isolation of Live Feed-Forward 3DGS Streams on a Shared Server GPU

The student's system ships each live reconstruction to a browser.
In any real deployment, many such streams would share a few server GPUs, so the resource that is actually being virtualised is the server GPU.
This project treats each stream (a replayed camera, its StreamSplat model state and its viewer) as a tenant.

It asks how the GPU sharing mechanisms available on a commodity GPU without MIG trade consolidation density against isolation.
Two quantities are measured.
Density is the number of tenants admitted while each still meets a declared per-tenant service-level objective (SLO).
Isolation is the latency inflation and deadline misses that live tenants suffer next to a noisy neighbour or a heavy best-effort AnySplat job.
The mechanisms compared are time-slicing between CUDA contexts, MPS, MPS with per-client SM caps (a soft stand-in for MIG), and a single process that multiplexes all tenants over one shared copy of the weights.

Every number is end to end.
The headline metric is glass-to-glass latency (from the moment a keyframe is released by the replayed camera to the first browser frame that draws it), measured on a probe tenant whose browser runs on a separate GPU machine.
Alongside it come server deadline-miss rate, update stalls, aggregate throughput and device memory per tenant.
Each comparison is a matched pair that differs in exactly one factor, repeated n=5 times against our own A/A (same arm run twice) spread.

The survey found no 3DGS work that compares MPS, MIG or time-slicing, or that measures GPU isolation (sweep note 7).
The closest prior systems either test a single client (Siekkinen and Kamarainen, arXiv 2604.02851) or schedule only the network side (arXiv 2606.17212) or the client side (GaussAnything, arXiv 2609.13859).
The browser is kept deliberately simple, because the factor under study is on the server.

Research question:

Take one commodity server GPU without MIG (RTX 6000 Ada 48 GB; `nvidia-smi -q` reports "MIG Mode: N/A", checked 2026-10-27), shared by N live feed-forward 3DGS streams.
Each stream is a StreamSplat online loop, ICLR 2026, arXiv 2506.08862, with keyframes every 5 frames at 30 fps, so the keyframe interval is Delta = 166.7 ms.

The four sharing mechanisms are:
(A1) default time-slicing between per-tenant processes;
(A2) MPS spatial sharing;
(A3) MPS with a static per-client SM cap of 100/N percent;
(A4) one process that multiplexes all tenants over a single copy of the weights, either round-robin (A4a) or batched (A4b).

RQ1, density.
How many tenants does each mechanism admit at the per-tenant SLO?

RQ2, isolation.
When the live tenants share the GPU with a best-effort heavy reconstruction job (AnySplat over 32-view windows), how much do the live tenants' p95 latency and deadline-miss rate inflate under each mechanism?
What does each MPS cap level on the best-effort job cost that job in throughput?

RQ3, fault isolation.
When one tenant hits a fatal GPU error, how many of the other tenants survive under each mechanism?

### Feasibility and learning first. The smallest end-to-end path is built from code that already runs on this machine or is MIT/Apache licensed: Measure first: paging a live feed-forward 3DGS stream into a WebGPU browser, and testing whether coupling the network and GPU schedulers pays off

The student builds a pipeline one measured layer at a time.
A replayed camera video (live in the last month) feeds AnySplat on a server GPU.
The server cuts each output into versioned voxel pages and sends them over a WebSocket.
A fork of the WebSplatter WebGPU renderer writes the pages into a fixed pool of GPU slots and renders at display rate from any viewpoint.
Each milestone answers one question with one number:
- how long a static frame takes and how often it stalls (the null for every stall claim);
- what writing N Gaussians into GPU memory costs;
- how big and how "new" each feed-forward update is.
  Consecutive AnySplat outputs share no Gaussian identity, so the delta unit is chosen from measured page churn;
- where the time goes end to end;
- at what offered load FIFO delivery breaks under tc netem.

A decision rule declared now turns those numbers into the research question.
The default question is the one the literature leaves open.
GaussAnything (arXiv 2609.13859) already budgets client GPU uploads with coalescing. arXiv 2606.17212 already schedules network sends by deadline.
Nobody has tested whether coupling the two, so that the sender sees the client's upload backlog, lowers latency compared with running both independently.

The main evaluation is a ladder of arms, each adding exactly one factor to FIFO.
Every cell gets n = 10 interleaved runs under netem, and a ceiling arm serves as a bound.
A final live run is checked with a 240 fps camera.
The virtualization content is the scene cache treated as demand-paged memory: page table, dirty pages, write coalescing, eviction and a measured miss-ratio curve.
If the course needs it, a per-client isolation milestone is added.

Research question:

Default question (committed in M5 unless the rule below switches it): a feed-forward 3DGS producer emits consecutive outputs with no shared Gaussian identity.
These are delivered as versioned voxel pages to a WebGPU browser with a fixed GPU slot budget.
Does coupling the sender to the client's reported upload backlog and view (the "joint" scheduler) reduce 95th-percentile trace-to-glass latency compared with the same coalescing sender and the same budgeted uploader running independently?
The comparison is at equal bandwidth, and the joint scheduler must not add render stalls.

Decision rule, declared now, before any measurement:
(a) If M1 shows that applying a realistic update never stalls a frame (p99 frame interval during the write is at most 1.2x the static p99 from M0), the GPU factor is dropped.
The question becomes network-only: do page-level latest-value coalescing and view priority cut p95 latency relative to FIFO?
(b) If M3 shows model inference is at least 80% of FIFO trace-to-glass latency even on unshaped loopback, delivery is not the bottleneck at AnySplat's rate.
The question then moves to the synthetic offered-load trace, plus StreamSplat (ICLR 2026, arXiv 2506.08862) as a faster producer.
(c) If M2 finds at least 70% of Gaussians matchable between consecutive outputs, per-Gaussian ID deltas replace pages.
(d) If the LTS full text (MMSys 2025), read in M5, already tests coupling, the question narrows to a browser replication of that test, and the write-up says so.
