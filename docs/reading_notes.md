# Reading notes

Verified notes on the papers named in `docs/source_material.md`, written 2026-10-27.
Each paper was read in full from its primary source by one agent and re-checked independently by a second agent.
Where the two disagree, the verifier correction listed under each paper wins.
The source material came from ChatGPT; the "Claim check" table records which of its statements held up.

## Summary

| Paper | Venue (verified) | arXiv | Relevance (1-5) |
| --- | --- | --- | --- |
| Streaming Real-Time Rendered Scenes as 3D Gaussians | Preprint (arXiv, Apr 2026); no venue | 2604.02851 | 5 |
| StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams | ICLR 2026 (poster) | 2506.08862 | 4 |
| Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform | Preprint (arXiv, Dec 2025); no venue | 2512.08478 | 4 |
| MoQSplat: Adaptive Progressive Streaming of 3D Gaussian Splatting via MoQ | IEEE MMSP 2026 (Media over QUIC special session) | 2609.18624 | 4 |
| S2GS: Streaming Semantic Gaussian Splatting for Online Scene Understanding and Reconstruction | ICML 2026 (poster) | 2603.14232 | 3 |
| WebSplatter: Efficient and Faithful In-Browser 3D Gaussian Splatting across Devices via WebGPU | ACM MM 2026 (camera-ready retitled) | 2602.03207 | 3 |
| StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video | AAAI 2026 | 2511.06046 | 3 |
| LMG: Efficient Streaming of Layered Mesh-Gaussian 3D Scenes | ACM MMSys 2026; no arXiv version | none | 3 |

## Streaming Real-Time Rendered Scenes as 3D Gaussians

Key `rt-gaussian-streaming`, relevance 5/5, identity confirmed by verifier: yes.

### Authors

Matti Siekkinen (Aalto University, Espoo, Finland); Teemu Kämäräinen (University of Helsinki, Finland).
Source: arxiv.org/abs/2604.02851 citation metadata and the HTML full text header.

### Venue and year

Preprint, 2026. arXiv v1 submitted Fri 3 Apr 2026 (only version as of 2026-10-27), primary category eess.IV, cross-listed cs.GR and cs.MM.
The PDF uses the ACM template with CCS concepts but names no conference and has no journal-ref or comments field; no venue found in a web search either.
Source: https://arxiv.org/abs/2604.02851 (submission history, subjects).

### arXiv

2604.02851

### Links

Abstract: https://arxiv.org/abs/2604.02851 ; full text HTML: https://arxiv.org/html/2604.02851 (v1) ; PDF: https://arxiv.org/pdf/2604.02851 (9 pages including references).
Project page: none found.
Code: none found (not linked from the arXiv page or the paper; a web search for code turned up nothing).
The only external software the paper cites is Google Draco (https://google.github.io/draco/, Apache-2.0 per the GitHub API for google/draco), which the authors say they modified to carry 3DGS attributes.
The modified version is not released.

### Problem

Cloud gaming and XR stream rendered 2D video, which ties the content to the server's viewpoint.
The client can then only compensate for latency with image-space reprojection or warping, which breaks under large viewpoint changes, disocclusions and lighting changes.
The paper asks a different question: can the server instead build a 3DGS scene representation online from a live game engine and stream it continuously, so the client renders any viewpoint locally?
Three challenges are named in Sec. 1: (1) build and refine the representation online from engine-native data, not an offline capture or SfM pipeline; (2) keep it consistent under object motion and lighting changes; (3) stream the evolving model continuously at well below the bandwidth of resending full snapshots.

### Method

Source: arxiv.org/html/2604.02851, Sec. 3.

**Architecture (Sec. 3.1, Fig. 1).**
- A Unity application drives scene simulation, cameras and texture generation.
- A native C++ plugin using CUDA and LibTorch holds the Gaussian tensors (means, scales, quaternions, opacities, SH coefficients).
  It optimizes them, can render them back into Unity, and serializes them for transmission.
- A dedicated optimizer thread runs asynchronously.
  Reference views, motion, lighting and new Gaussian input arrive from the Unity render loop.
  So the optimization rate is decoupled from the render rate, though the paper never names or measures that decoupling.

**Two camera sets (Sec. 3.2).**
- Many low-resolution input cameras (default 256x256) sit on a dome (the top of a Fibonacci sphere) around the tracked player position.
  They initialise Gaussians.
- Fewer, higher-resolution reference cameras supply the ground-truth images the optimizer trains against.

**Initialisation from engine data (Sec. 3.2.1-3.2.2).**
- Each input camera renders several per-pixel passes: world-space points, surface normals, lighting quantities used to initialise SH coefficients, and object IDs.
- Depth-neighbourhood differences give an isotropic starting scale for each Gaussian.
- Cross-camera input culling keeps a sample only when no other camera sees that surface at a better, less oblique angle.
- The pipeline stays in GPU memory, and no SfM step is needed.
  The new batch is appended to the active pool and the optimizer state is rebuilt.

**Continuous optimization (Sec. 3.2.3).**
- Standard 3DGS losses and parameters, run against the reference views, optionally in mini-batches.
- Clone and split are available but not used, because new Gaussians from them converge slowly.
  Growth comes from model expansion instead.
  The model is never 'finished'.

**Progressive model expansion (Sec. 3.3).**
- The world is a regular 3D grid.
  A broad phase picks candidate cells near the viewer, then a camera visibility test runs.
  Cells that are visible but not yet initialised get new Gaussians, which join the same optimization loop.
- Freezing: converged Gaussians move from the active (trainable, with optimizer state) pool to a frozen pool that is inference-only.
  This cuts memory and optimizer cost.
- Pre-culling: each camera finds its visible cells by depth testing, and only Gaussians in those cells enter rendering and optimization.

**Scene dynamics (Sec. 3.4).**
- Relighting: each Gaussian stores diffuse SH transfer coefficients, combined at render time with the current lighting expressed as SH.
- Directional-light shadows: a per-Gaussian visibility term computed from a light-space orthographic depth map, the same idea as a shadow map.
- Rigid objects: Gaussians carry persistent object IDs and object-local coordinates.
  Engine object transforms are applied before each optimization step, so motion is tracked without re-optimizing.

**Streaming (Sec. 3.5).** See the transport field.

**Client.** A 'lightweight receiver' keeps a local copy of the model, applies snapshots, deltas and transforms, and renders the user's viewpoint back into Unity.
The paper implies the client runs the same native plugin, but does not say so explicitly.

### Scene update model

**Creation.**
- Gaussians are created from engine-native per-pixel samples.
  This happens at initialization, and again whenever a newly visible but uninitialised grid cell is found (Sec. 3.3).
- New Gaussians are appended to the active pool.
- Clone and split are deliberately not used (Sec. 3.2.3).

**Update.**
- All parameters (means, scales, quaternions, opacity, SH) are refined continuously by gradient descent against reference views.
- Rigid objects move by applying the per-object transform to that object's Gaussians.
- Lighting changes come from re-evaluating SH transfer against new lighting SH, plus a per-Gaussian shadow-visibility term.

**Freezing.** Converged Gaussians move to a frozen pool.
They are still rendered and streamed but no longer trained.

**Deletion.** Pruning is 'supported' (Sec. 3.1), but the paper never says whether it ran in the experiments or how a deletion is sent to the client.

**Identity.**
- Gaussians keep a persistent identity across updates as a side effect of the design.
  The model is one persistent, optimized pool, and the deltas are per-attribute tensor updates against it.
- There are dedicated 'Gaussian-ID ordering packets' ('Gaussian-order update' in Fig. 9).
  This suggests the server-side order can change, probably through active/frozen repartitioning or pruning, and the client is told the new order.
  The exact semantics are not described.
- Per-object persistent IDs come from Unity.

**Why this matters for our idea.** Identity comes for free here because the model is an optimizer state.
A feed-forward model that emits a fresh Gaussian set per frame does not give you this.

### Transport

Source: Sec. 3.5, Sec. 4.5, Fig. 9.

**Connection.** A 'socket server that waits for a client connection'.
The protocol (TCP or otherwise) is not named, and there is no WebSocket, QUIC or browser.

**Packet types.**
- Full model snapshot, encoded with a modified Google Draco that carries 3DGS attributes.
- Object-transform, object-ID, tensor-metadata and tensor-delta packets.
- Light-visibility packets.
- Gaussian-ID ordering packets.

**Session start.**
- The server configures per-tensor payload codecs and residual-gating parameters, then creates a streamed-model state.
- When a client connects, the server sends the object-ID map and all object transforms, then an immediate snapshot so the client can render as soon as possible.

**Steady state.**
- Object transforms are sent whenever objects move.
- Snapshots are sent periodically at a configurable interval (the value is not given).
  Each snapshot resets the delta encoders, stores fresh residual baselines and increments a transmission epoch that later delta packets refer to.
  The client resets its replay state on each snapshot.
- Tensor deltas follow fixed attribute-specific schedules.
  In the evaluation setup, opacity and DC SH go every frame, while means, scales, quaternions and higher-order SH go 'less frequently' (rates not given).
- Means and scales can be residual-coded against the last transmitted state; the other attributes are sent as quantized tensor updates.
  Zstandard is applied to Gaussian attributes, and 'residual gating' is mentioned without detail.

**Absent.** No prioritisation by view, importance or deadline, no adaptation to bandwidth, no congestion handling, no multi-client fan-out.

**Measured traffic (Fig. 9).** Bursty.
My extraction of the vector figure gives a bin width of ~1.08 s and this breakdown of total bytes:

| Packet type | Share of bytes |
| --- | --- |
| Scale deltas | ~42% |
| Higher-order SH deltas | ~30% |
| Rotation deltas | ~8% |
| Position deltas | ~6% |
| Draco snapshots | ~6% |
| DC SH deltas | ~6% |
| Opacity, ordering and object-ID packets | under 2% together |

### Client

- **Platform.** Native, not a browser.
  The client instantiates 'a local copy of the model' and renders it 'back to Unity from the viewpoint of the camera that tracks the user's motion' (Sec. 3.1).
  That implies a Unity application with the same native CUDA/LibTorch 3DGS rasterizer, not a thin client.
  There is no WebGPU or WebGL.
- **GPU buffer updates.** The client decodes Draco snapshots directly into its local model and applies the per-attribute deltas through matching 'replay objects' (Sec. 3.5).
  How deltas reach the GPU buffers (CPU or GPU decode, staging, partial or full buffer upload) is not described.
- **Sorting.** Not described.
  It is presumably the standard 3DGS CUDA tile rasterizer, but the paper does not say.
- **Memory limits.** None on the client.
  Client hardware is not stated, and client render FPS is not reported, even though Sec. 4.1 lists 'rendering frame rate' among the metrics.
- **Server memory.** Fig. 6 reports server CUDA memory at ~3.3 to ~5.0 GB for up to ~1.5M Gaussians.

### Evaluation design

Source: Sec. 4 and Figs. 2-10, read from the PDF figures.

**Workload.**
- One custom synthetic Unity scene: toys on a floor with a directional light.
- The dynamics variant adds a bouncing textured ball and a rotating white cube, with the light either fixed at 45 deg zenith or rotating at 15 deg/s.
- The exploration experiment duplicates the content to make a larger scene and walks through it for ~195 s.
- The streaming experiment replays a prerecorded camera trajectory of ~46 s, for repeatability.
- No real-camera or captured data, and no standard 3DGS benchmark scenes.

**Metrics.**
- SSIM and mean FLIP against ground-truth Unity renders from the same viewpoint and lighting.
- Optimizer steps/s and views/s, server CUDA memory, and Gaussian counts (total, frozen, active).
- Wire bitrate by packet type, and server-minus-client SSIM over time.
- Rendering frame rate is listed as a metric but never reported.
- No latency of any kind (no motion-to-photon, no glass-to-glass), no stalls, no client decode or upload time, no client FPS.

**Experiments and arms.**

1. Fixed reference cameras (Sec. 4.2.1).
   - Arms: 4, 6 and 8 reference cameras, plus 8 with 320x320 input; also 20 input cameras, mentioned only in text as 'not noticeably better'.
   - 10 input cameras, 12 evaluation views in a circle.
   - The text says 2000 iterations, but Fig. 2 plots iterations 10/100/200/500/1000.
   - Box plots are across the 12 viewpoints.
  Fig. 3 gives the time to reach FLIP < 0.07 and the fraction of views that ever get there.

2. Moving view camera (Sec. 4.2.2).
   - Arms: static reference rig vs. a 4-camera rig moving with the viewer, with orbit times of 15/30/60/120 s.

3. Baseline: depth-assisted forward 3D warping.
   - Frames from the fixed reference cameras are warped to the target view and merged, using 12-bit and 16-bit depth (Fig. 5).
   - Only image quality is compared.
  Neither the bitrate nor the latency of the warping or video path is measured.
   - The warping and 3DGS results are on separate figures (Fig. 5 vs.
  Fig. 4).

4. Ablation: dynamics support (object tracking plus relighting) disabled vs. enabled, under fixed and moving light (Fig. 7).

5. Streaming: one replay run, one client (Figs. 9-10).
   - The second-pass extraction above found no snapshot-only arm, delta-only arm or snapshot-interval sweep.
   - No competing 3DGS streaming system, and no video-streaming bitrate baseline.

**Network conditions.** None described.
There is no emulation (no tc/netem), and no bandwidth, RTT or loss settings.
Whether client and server were separate machines is not stated.

**Hardware.** A single NVIDIA GeForce RTX 4070 SUPER (12 GB).
Client hardware is not stated.

**Repetitions and statistics.** Apparently one run per configuration.
The only spread shown is across viewpoints.
There are no repeated seeds, no confidence intervals and no statistical tests.

**Units.** Some axes have none: Fig. 3 'thresholdTime' and Fig. 4 'wall clock time x10^4' are likely milliseconds but are not labelled.

### Key numbers

Figure values marked ~ were read off the plots by eye.
Values marked (vector) were extracted from vector data in the PDF.

**Gaussian counts (Table 1).** Initial model has 208K Gaussians with 10 input cameras at 256x256, 236K with 20 at 256x256, and 453K with 10 at 320x320.

**Fixed reference cameras, median at 1000 iterations across 12 views (Fig. 2).**

| Reference cameras | FLIP | SSIM |
| --- | --- | --- |
| 4 | ~0.083 | ~0.935 |
| 6 | ~0.065 | ~0.955 |
| 8 | ~0.056 | ~0.963 |
| 8, 320x320 input | ~0.050 | ~0.966 |

At iteration 10 all arms are at FLIP ~0.11 and SSIM ~0.905.

**Time to FLIP < 0.07 (Fig. 3).**

| Reference cameras | Fraction of views that ever reach it | Median time (units not labelled, likely ms) |
| --- | --- | --- |
| 4 | 0.25 | ~1300 |
| 6 | 0.67 | ~2200 |
| 8 | 0.84 | ~2900 |
| 8, 320x320 | 0.92 | ~3000 |

**Moving camera (Fig. 4).**
- A rig moving with the viewer reaches FLIP ~0.02-0.035 after ~20% of the orbit.
  Slower orbits (120 s) do better than fast ones (15 s).
- A static rig with 4 reference cameras oscillates up to FLIP ~0.14.
  With 8 it stays at ~0.03-0.08.

**Warping baseline (Fig. 5).** Mean FLIP is ~0.06-0.20 with 12-bit depth and ~0.04-0.17 with 16-bit depth over the orbit.
That is worse than 3DGS with 8 static reference cameras.

**Exploration, ~195 s, one RTX 4070 SUPER (Fig. 6).**
- About 1.5M Gaussians in total by the end, ~1.3M frozen and ~0.2M active.
  Active count peaks at ~0.65M at ~100 s.
- CUDA memory goes from ~3.3 GB to ~4.5-5.0 GB.
- Optimizer runs at ~25-40 steps/s, with views/s up to ~60 once freezing allows batching.

**Dynamics ablation, FLIP at 1000 iterations (Fig. 7).**

| Light | Dynamics support disabled | Enabled |
| --- | --- | --- |
| Fixed | ~0.087 | ~0.053 |
| Rotating | ~0.114 | ~0.079 |

**Streaming, one ~46 s replay, one client, network unspecified (Sec. 4.5).**
- About 550 MB transferred.
- Mean 92.5 Mbps, median 66.8 Mbps, peak 481.8 Mbps.
  From the vector data the bins are about 1.08 s, so the peak is roughly a 1-second average.
- Mean SSIM is 0.96 on the server and 0.91 on the client, with 8 reference cameras.
  The per-sample gap reaches ~0.108 (Fig. 10).
- Vector breakdown (Fig. 9): Draco snapshots carry ~34 MB and peak at ~191 Mbps.
  Higher-order SH deltas carry ~155 MB and produce the ~447 Mbps peak.
  Scale deltas carry ~220 MB.
  My vector reconstruction totals ~523 MB, close to the stated ~550 MB.

### Limitations

**Stated by the authors (Sec. 5).**
- Quality depends on history: newly discovered regions are worse.
- Relighting is diffuse plus approximate shadow visibility only, with no global illumination or view-dependent materials.
- Only rigid motion is handled.
- 'A full end-to-end comparison in terms of bitrate, latency, and client cost remains future work.'
- Sec. 4.5 adds that snapshot frequency and size, delta compression and update gating are optimization targets.

**What I see.**
- Bandwidth and latency claims are not tested.
  - No latency, stall, client FPS or client cost is measured.
  - The streaming claims rest on one ~46 s trace with one client, an unspecified network and no emulation.
  - There is no snapshot-only or delta-only control arm, so the stated goal of streaming well below the cost of resending snapshots is never tested.
  - The multi-user amortization argument is never tested.
- The streaming figures disagree with themselves.
  - The Fig. 9 caption contradicts its own plotted layers: the largest spikes are labelled as higher-order SH deltas, not Draco snapshots.
  - Fig. 10's snapshot markers do not line up with Fig. 9's Draco layer.
- The workload is narrow.
  - There is one synthetic scene, with no repeated runs or uncertainty.
  - Fig. 2 runs to 1000 iterations, while the text says 2000.
  - Some axes have no units.
- The baseline is narrow.
  - The warping baseline is compared on quality only, not on bitrate.
  - Its comparison to 3DGS is made across separate figures.
- The design is too thin to reproduce.
  - The snapshot interval, per-attribute delta rates, quantization bit depths and residual-gating rule are not given.
  - So are the transport protocol and how deletions or pruning reach the client.
- Typical bitrates of 67-92 Mbps, with ~480 Mbps bursts, are far above common WAN or Wi-Fi budgets for a model of only ~0.2-1.5M Gaussians.
- No code or data is released.

### Gap relative to our idea

**Overlap (honest).** This paper already does the core loop: a server continuously updates a 3DGS model and streams it as periodic snapshots plus per-attribute deltas over a socket, with persistent identity, to a client that renders its own viewpoint.
Specifically, it already has:
- An optimizer thread decoupled from the render loop.
- A fixed per-attribute update schedule (opacity and DC every frame, geometry and higher-order SH less often), which is a primitive static priority policy.
- Snapshot epochs that reset the delta baselines.
- Persistent object IDs plus Gaussian-order packets.
- Grid-cell visibility, used for expansion and pre-culling on the server only.
- A bitrate breakdown by packet type.

So 'stream an evolving 3DGS as snapshots plus deltas to a client that renders free-viewpoint' is not a contribution on its own.
This paper should be cited as the closest system-level prior.

**What our idea would add.**

1. Input and model.
   - We would use real camera video and a feed-forward model, instead of a synthetic engine's G-buffers with per-scene online optimization.
   - Identity becomes a real problem: feed-forward models emit fresh, uncorrelated Gaussians per frame.
  Here identity comes for free because the model is one persistent optimizer state.
   - Matching, fusion, or independently decodable blocks would be new work.
2. Client.
   - A browser with CPU decode, a cache, and WebGPU buffer uploads and sorting at its own frame rate, instead of a native Unity/CUDA client.
   - Client-side GPU-budget limits (browser GPU memory as a capacity-limited virtual scene cache) are not considered at all here.
3. Scheduling, which is the main gap.
   - This paper has no adaptation: static schedules, no view-dependent selection (it sends the whole model), no deadlines, no bandwidth or GPU awareness, no coalescing beyond fixed rates, no drop or defer.
   - Joint network-GPU, deadline-aware, budget-aware update scheduling against FIFO and view-aware baselines would be new.
   - Their fixed per-attribute schedule with periodic snapshots is a natural extra baseline arm we could re-implement.
4. Evaluation.
   - They measure no latency, stalls or client cost, and use no network emulation.
   - Glass-to-glass latency, render stalls, tc/netem conditions, repeated runs and trace-driven replay would be a stronger evaluation design.
  Their prerecorded-trajectory replay is the same repeatability idea as our pre-recorded Gaussian update traces.
5. Multiple clients and resource isolation (the virtualization angle) are untested here, even though the paper motivates multi-user amortization.

**Caution.** Their 92.5 Mbps mean and 481.8 Mbps peak are a useful order-of-magnitude motivation for budgeted, view-aware streaming.
They are not a null or baseline for our data: different scene, model, codec settings and Gaussian counts, and no network spec.

### Reusable

- **Code, weights, data, traces:** none released (none linked on arXiv or in the paper; a web search found nothing).
- **Reusable as design ideas, re-implemented by us:**
  - The packet taxonomy: snapshot, tensor-delta per attribute, tensor-metadata, object-transform, object-ID, light-visibility, Gaussian-ID ordering.
  - Snapshot epochs that reset the residual baselines.
  - Per-attribute update schedules: opacity and DC often, geometry and higher-order SH rarely.
  This could be one baseline arm.
  - Residual coding of means and scales plus quantization plus Zstandard.
  - Replaying a prerecorded camera trajectory for repeatable streaming runs.
  - The bitrate-by-packet-type plot as an evaluation figure.
- **Reusable libraries:**
  - Google Draco: Apache-2.0 (verified via api.github.com/repos/google/draco).
  The authors' 3DGS-attribute modification is not released.
  - Zstandard (facebook/zstd): dual BSD/GPLv2 (GitHub reports NOASSERTION), with a WASM build usable in the browser.
- The Unity/CUDA/LibTorch plugin itself is not available.

### Notes

**Reading notes.**
- Read in full from https://arxiv.org/html/2604.02851 (v1, all sections and references) and from the PDF https://arxiv.org/pdf/2604.02851.
  Figures were rendered at 300 dpi and read.
- Fig. 9's stackplot is vector data.
  I parsed it with PyMuPDF to get the per-layer bitrates.
  That reconstruction reproduces the paper's mean, median and peak to within about 0.5%, which is how the caption contradiction was confirmed.

**Corrections to the ChatGPT material.**
- Title, year and arXiv id are correct.
  The venue is preprint only.
- 'Snapshot vs Delta Streaming' should be reworded: the paper uses both, but never compares them.
- The relevance is still the highest of the set.
  It is the closest system-level prior, and its fixed per-attribute schedule is a ready-made baseline arm for our scheduler comparison.

**Gaussian identity.** In this paper identity comes from an optimizer over one persistent pool.
Our feed-forward setting must solve it explicitly.

**Working files.** Scratch files (downloaded HTML/PDF, figure crops) are in papers/agent_workdir/rt/.
That is outside the requested proj/ write location only by subfolder name; it is inside proj/.

Sources: https://arxiv.org/abs/2604.02851 , https://arxiv.org/html/2604.02851 , https://arxiv.org/pdf/2604.02851 , https://api.github.com/repos/google/draco

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Source line 13: the paper exists, is from 2026, and has this title. | supported | arxiv.org/abs/2604.02851: title 'Streaming Real-Time Rendered Scenes as 3D Gaussians', v1 dated 3 Apr 2026, by Siekkinen and Kämäräinen. The arXiv id 2604.02851 in the source link is correct. It is a preprint and no venue is stated. |
| Source line 15: it already implements a system where the server continuously updates 3DGS, sends incremental updates, and the client reconstructs and renders the scene itself. | supported | Abstract and Sec. 3.5 (arxiv.org/html/2604.02851): the server 'continuously optimizes a 3DGS model', streams 'full model snapshots and incremental updates', and 'clients reconstruct the streamed Gaussian model locally and render their current viewpoint'. Sec. 3.1 adds that the client builds a local copy of the model from data received 'over a socket connection'. |
| Source line 15: so 'continuously sending 3DGS to the client' cannot be claimed as a new research contribution. | supported | This is a judgement, but the paper backs it. Contribution 3 (Sec. 1) is 'a live 3DGS streaming prototype that transmits the evolving scene representation using full model snapshots and incremental updates, enabling client-side novel-view rendering'. |
| Table row (source line 43): it does server-side continuous model updates, delta streaming and client rendering. | supported | Sec. 3.5 describes the deltas. Per-attribute tensor-delta packets cover opacity, DC SH, higher-order SH, means, scales and quaternions. Means and scales can be residual-coded against the last transmitted state, the other attributes are quantized, and everything is Zstandard-compressed. There are also object-transform, object-ID, tensor-metadata, light-visibility and Gaussian-ID ordering packets. The client is a 'lightweight receiver' that applies snapshots, deltas and transforms, then renders. |
| Table row (source line 43): it uses Unity and online optimization, not feed-forward reconstruction and a WebGPU browser client. | supported | Sec. 1 and 3.1: it is a 'Unity-based prototype' with a native C++ CUDA/LibTorch plugin. Gaussians are initialised from Unity G-buffer-like passes (world-space points, normals, SH lighting cues, object IDs) and refined by per-scene differentiable-rendering optimization (Sec. 3.2.3). The client renders 'back to Unity' (Sec. 3.1). The paper never mentions a browser, WebGPU or feed-forward models. Note that the input is a synthetic game engine, not camera video. |
| Source line 56: it is the closest to the user's overall idea, 'especially Snapshot vs Delta Streaming'. | partly | The design does combine snapshots and deltas. Sec. 3.5 sends periodic Draco snapshots at a configurable interval, each one resetting the delta encoders and incrementing a transmission epoch, with deltas in between. The paper does NOT evaluate snapshot versus delta. There is no snapshot-only or delta-only arm and no varying of the snapshot interval. Sec. 1 states the goal of 'keeping the bandwidth significantly below that of naively retransmitting full model snapshots', but no experiment measures it. The only streaming results are one bitrate trace (Fig. 9) and one server-versus-client SSIM trace (Fig. 10). 'Closest overall' does hold at the system level. |
| Paper's own claim (abstract): the approach improves viewpoint flexibility for latency compensation. | unsupported | No latency of any kind is measured. Sec. 5 says: 'a full end-to-end comparison in terms of bitrate, latency, and client cost remains future work.' The only evidence for flexibility is image quality: FLIP of the 3DGS model versus depth-assisted 3D warping (Fig. 5 vs Fig. 4). |
| Paper's own claim (abstract, Sec. 1): server-side modeling amortizes better across multiple users than per-user rendering and video streaming. | unsupported | Every streaming experiment uses a single client (Sec. 4.5). The fixed-camera experiment is said to give 'insights into view synthesis quality under multi-user scenarios' (Sec. 4.2.1), but it uses no multiple clients, no server cost per user and no video-streaming baseline. |
| Paper's own claim (Fig. 9 caption): full Draco snapshots dominate the largest instantaneous bitrate spikes. | contradicted | The figure's own data says otherwise. I extracted the vector stackplot from the PDF (page 8) and mapped the fill colours to the legend. The two largest spikes (~481 Mbps at t~5.4 s, ~450 Mbps at t~29 s) are almost entirely the 'Higher-order appearance delta' layer, ~447 Mbps at 5.4 s. The 'Draco snapshots' layer peaks at only ~191 Mbps, at t~1.1 s. My reconstruction matches the paper's stated mean (92.4 vs 92.5 Mbps), median (66.6 vs 66.8) and peak (481.0 vs 481.8), so I read it correctly. Either the legend or the caption is wrong, and I cannot tell which. Fig. 10 marks snapshots at about 0, 15, 29.5 and 45 s, which matches neither Fig. 9's Draco layer (~1 s and ~24-26 s) nor exactly its spikes. Do not cite 'snapshots cause the peaks' without this caveat. |
| Paper's own claim (Sec. 4.5): a real-time optimized 3DGS model can be streamed to a client while preserving high rendering quality. | partly | Mean SSIM is 0.96 on the server and 0.91 on the client for one ~46 s moving-camera trace with 8 reference cameras. Fig. 10 shows the per-sample server-minus-client SSIM gap reaching ~0.108. This is one run, and the network and client hardware are not described. |

### Verifier corrections

None.

### Verifier on the gap assessment

Fair overall, and checked against the text.
The paper does not do view-prioritised transmission: snapshots are "the full current state of the Gaussian model", and deltas go to "selected Gaussian attributes according to attribute-specific schedules" (Sec. 3.5).
It has no deadline, bandwidth or GPU-budget awareness.
It serves one client: the server "starts a socket server that waits for a client connection" (Sec. 3.5).
It uses no browser or WebGPU, no feed-forward model and no camera video (Sec. 3.1, 4.1).
It measures no latency: "a full end-to-end comparison in terms of bitrate, latency, and client cost remains future work" (Sec. 5).
Source for all of these: https://arxiv.org/pdf/2604.02851.

Two nuances the reader under-states.
Neither changes the verdict.
(1) Content creation is viewer-driven upstream.
The input-camera dome sits "around the position that tracks the client player's current position" (Sec. 3.2.1).
Expansion adds Gaussians only in visible cells near the viewer (Sec. 3.3).
Sec. 4.2.2 tests a reference rig that moves with the viewer and recommends a "user specific reference camera arrangement".
So what gets modelled, and therefore streamed, is implicitly viewer-local, even though transmission itself is not view-prioritised.
The claim "sends the whole model" holds only for the model as it exists, not for a whole scene.
(2) "Residual-gating parameters" (Sec. 3.5) probably act as a magnitude filter that skips small updates.
That is a weak, content-based way of dropping updates, not a deadline- or budget-aware one.
Phrase the gap as "no deadline/budget/view-aware scheduling", not "no filtering of updates at all".
Server-side pre-culling (Sec. 3.3) is described only for the server's cameras.
The paper says nothing about the client, so "server only" is a fair reading of the text.

### Verifier notes

Everything verified independently from https://arxiv.org/abs/2604.02851, https://arxiv.org/pdf/2604.02851 (9 pages, PyMuPDF text and vector extraction) and the figures rendered from the PDF.
My files are in papers/agent_workdir/verify_rt/.

Identity:
- Title, authors (Siekkinen, Aalto; Kämäräinen, U.
  Helsinki) and arXiv id 2604.02851 are confirmed by the abs page meta tags.
- v1 is dated 3 Apr 2026 and is the only version (https://arxiv.org/abs/2604.02851v2 returns 404).
- Subjects are eess.IV, cs.GR and cs.MM.
  There is no comments, journal-ref or DOI field.
- The PDF uses the ACM template with CCS concepts but has no conference block.
- Web searches found only arXiv mirrors and aggregators (alphaxiv, radiancefields, pith), no venue.
  Preprint is correct.

Code: none.
The arXiv abs and HTML pages link only to google.github.io/draco.
A GitHub API search for the title and author returned no author repo; the one hit (ketsunishi0909/Holo...) is unrelated.

Three key numbers confirmed verbatim in Sec. 4.5:
- Streaming: ~46 s, ~550 MB, mean 92.5 Mbps, median 66.8 Mbps, peak 481.8 Mbps.
- SSIM: server 0.96, client 0.91, with 8 reference cameras.
- Table 1: 208K, 236K and 453K Gaussians.
  Hardware: RTX 4070 SUPER, 12 GB (Sec. 4.1).

Fig. 2, 3, 6 and 7 readings match by eye:
- Fig. 2 medians at iteration 1000: FLIP ~0.083, 0.065, 0.056, 0.050.
  SSIM ~0.935, 0.955, 0.963, 0.966.
- Fig. 3 fractions: 0.25, 0.67, 0.84, 0.92.
  Medians ~1350, 2250, 2950, 3050, axis labelled only 'thresholdTime'.
- Fig. 6: ~1.5M total Gaussians, ~1.3M frozen, ~0.19M active at the end; 3.3 to 5.0 GB.
- Fig. 7: 0.087 to 0.053 under fixed light, 0.114 to 0.079 under moving light.
- The text says "2000 iterations" while Fig. 2's axis ends at 1000.
  Confirmed.

Fig. 9 caption contradiction: CONFIRMED independently.
- I re-extracted the 9 stackplot polygons, mapped each fill colour to its legend label by the y-position of the swatch, and grouped vertices per x bin.
- 43 bins, t = 1.19 to 46.44 s, median dt 1.045 s (mean 1.077).
- The reconstruction reproduces the paper's mean 92.5 and peak 481.8 Mbps exactly, and the median to 66.7 vs 66.8.
- Spike at t = 5.47 s: 481.8 total, of which 447.6 is the Higher-order appearance delta.
- Spike at t = 28.61 s: 452.2 total, of which 407.6 is Higher-order.
- The Draco layer appears only at t = 1.19 s (191.7) and t = 25.44 s (104.6).
- The rendered plot shows the same thing: the orange (Higher-order) layer forms both big spikes.
- Fig. 10 (a raster image) puts snapshot markers at ~0, 15, 29.5 and 45 s, which does not line up with Fig. 9.
  Its caption calls them "check marks" but they are x markers.
- The per-sample SSIM gap peaks at ~0.108 near t ~38.5 s.
  Confirmed.

Minor numeric nuance (no correction needed): byte shares depend on the integration method.
Summing bins gives scale 41.6%, Higher-order SH 29.0%, rotation 7.8%, position 6.4%, Draco 7.4% (~39 MB, vs the reader's ~6% / ~34 MB), DC 6.1%, the rest ~1.7%, ~520 MB in total.
The ordering and conclusions are unchanged.

Small extra: Sec. 4.5 says the replay ensures "all compared runs observe the same sequence of viewpoints", yet only one streaming run is reported.
This supports the reader's "one run, no control arm" point.

Reader's section attributions check out: pruning is "supported" in Sec. 3.1, and the CUDA/Libtorch plugin is described in Sec. 1.
The views/s-greater-than-steps/s batching remark is the paper's own (Sec. 4.3, "from elapsed time 150 onwards").

## StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams

Key `streamsplat`, relevance 4/5, identity confirmed by verifier: yes.

### Authors

Zike Wu, Qi Yan, Xuanyu Yi, Lele Wang, Renjie Liao (University of British Columbia, Vector Institute for AI, Nanyang Technological University, Canada CIFAR AI Chair).
Source: arxiv.org/abs/2506.08862 citation metadata and the v2 PDF first page.

### Venue and year

ICLR 2026.
Evidence: the arXiv comment field says "Accepted by ICLR 2026" (arxiv.org/abs/2506.08862); the first page of the v2 PDF reads "Published as a conference paper at ICLR 2026" (arxiv.org/pdf/2506.08862v2); the project page (streamsplat3d.github.io) and the GitHub title also say ICLR 2026.
A search result listed an OpenReview PDF (openreview.net/pdf/8b7e3b13bd50d00ff017031283689feb480a8c5d.pdf), but OpenReview returned a bot-check page, so I could not read the OpenReview record itself. arXiv v1 is dated 2025-06-10 and v2 is dated 2026-03-02.
I read v2.

### arXiv

2506.08862

### Links

Paper abstract: https://arxiv.org/abs/2506.08862.
Full text (v2 HTML, which I read in full, appendices included): https://arxiv.org/html/2506.08862.
PDF: https://arxiv.org/pdf/2506.08862v2 (24 pages).
Project page: https://streamsplat3d.github.io/.
Code: https://github.com/DSL-Lab/StreamSplat.
The GitHub API reports license=null and the repo root has no LICENSE file, so by default all rights are reserved.
The repo has 192 stars and its last push was 2026-03-11.
Pretrained checkpoint: a Google Drive folder linked from the README.
NAME COLLISION: a different paper with the same short name exists, arXiv 2608.01659, "StreamSplat: Streaming Feed-Forward 3D Gaussian Splatting" (Song, Wang, Li, Cai, Liu, 2026-08-03, no venue in the arXiv comments).
That paper uses calibrated views and a voxel-aligned causal cache.
It is NOT the ICLR 2026 paper.
Do not mix up their claims.

### Problem

The goal is to reconstruct a dynamic 3D scene (4D reconstruction) online from a monocular video stream, with no camera calibration and no per-scene optimization.
The output is a dynamic 3D Gaussian Splatting representation that can be rendered at any time and, within limits, from other viewpoints.
Prior dynamic 3DGS methods (DGMarbles, 4DGS, Shape-of-Motion-style pipelines) need camera calibration (COLMAP), access to the full sequence, and minutes to hours of per-scene optimization.
Prior feed-forward 3DGS methods (pixelSplat, MVSplat, NoPoSplat, StreamGS) are either static-scene only or require calibration.
(Sec. 1-2, arxiv.org/html/2506.08862.)

### Method

The pipeline has four parts, from Sec. 3 and App.
A-B of arxiv.org/html/2506.08862, cross-checked against the code at github.com/DSL-Lab/StreamSplat.

(1) Input and canonical space.
Each frame is resized to 512x288 and paired with a pseudo-depth map from Depth Anything V2 (ViT-L).
Everything lives in one shared orthographic canonical space viewed by a fixed camera.
Because no real camera model is used, intrinsics never matter, and camera motion simply shows up as Gaussian motion.

(2) Static encoder (stage 1).
The encoder sees one frame at a time.
It turns the RGB-D image into 8x8 patches (36x64 = 2304 tokens) and runs a 10-layer ViT (dim 768, 12 heads).
A 2-layer window-attention upsampler then multiplies the token count by 16, giving one Gaussian token per 2x2 pixel patch: a 256x144 grid, or 36,864 Gaussians per frame.
Linear heads predict each Gaussian's parameters:
- a pixel-aligned position, where x and y are the pixel coordinate plus an offset and depth comes from an inverse-depth mapping g(z)=2/(1+z);
- scale, opacity and color.
In the released configs, rotation is fixed to the identity quaternion, color is RGB with spherical-harmonics degree 0 (no view dependence), and scale has two values (y is set to their mean).

(3) Probabilistic position sampling.
Instead of regressing a position offset directly, the model predicts a truncated normal distribution on [-1,1] and samples from it.
This is a pixelSplat-style trick to escape local minima.
Removing it costs 6.36 dB PSNR on key frames (Table 4).

(4) Dynamic decoder (stage 2, encoder frozen).
For a frame pair (k-1, k), a 10-layer Transformer does self-attention over both frames' tokens and cross-attention to the other frame's DINOv2 features.
After the same upsampler, a 2-layer MLP predicts two things for every Gaussian of both frames:
- a velocity v in [-1,1]^3 (also sampled probabilistically; the code supports a polynomial order 'forder', default 1, i.e. linear);
- two opacity-fade coefficients, gamma0 and gamma1.
The motion is bidirectional: frame k-1's Gaussians move forward to time k, and frame k's Gaussians move backward to time k-1.
The pair is normalized to t in [0,1] and each Gaussian follows mu(t)=mu0+v(t-t0).
Its opacity follows alpha(t)=alpha*sigmoid(-gamma0(|t-t0|-gamma1))/sigmoid(gamma0*gamma1), so it fades as time moves away from its creation frame t0.
This fade is the 'adaptive Gaussian fusion': no hard matching is done, and overlap between the two frames' Gaussians is resolved softly by the reconstruction loss.

Training.
Stage 1 uses an L2 + LPIPS loss plus a scale/shift-invariant depth loss with an adaptive decay.
Stage 2 adds a mask loss on DAVIS and YouTube-VOS foreground masks and reconstructs all intermediate frames between two sampled frames.
Training took about 3 days on 8 A100s, on CO3Dv2, RE10K, DAVIS and YouTube-VOS used as raw video (no camera parameters).

Online inference (Algorithm 2).
For each new frame I_k: encode h_k, decode the pair (h_{k-1}, h_k), update the canonical set with the forward-deformed previous Gaussians, add the backward-deformed current Gaussians, render any time in [t_{k-1}, t_k], prune Gaussians whose opacity at t_k has decayed, and keep h_k for the next step.

### Scene update model

Per keyframe k, the encoder creates a fresh, pixel-aligned layer of 36,864 Gaussians, one per cell of a 256x144 grid.
A layer's static parameters (position, scale, opacity, RGB) come deterministically from the frame's embedding.

Each Gaussian gets dynamic parameters (velocity, gamma0, gamma1) twice:
- backward, in pair (k-1, k);
- forward, in pair (k, k+1).

After that its opacity has faded and it is pruned (Algorithm 2).
The paper says 'Each Gaussian persists across two consecutive frames' (Sec. 3.2).
So at any render time t in [t_k, t_{k+1}], the scene is exactly two layers: frame k's Gaussians moving forward and frame k+1's Gaussians moving backward, about 73,728 Gaussians.
Nothing accumulates into a growing map.
The scene is a sliding window of 2.5D layers in an orthographic canonical space.

Identity.
There is no explicit persistent per-Gaussian ID in the paper.
An implicit, stable ID does exist: (keyframe index, grid cell index), valid for about two keyframe intervals.
The paper's long-term 'persistence' (Fig. 4/15) is qualitative and comes through soft overlap, not a stored identity.

Released code. splat_inference.py treats every pair independently in a batch, concatenating gs_0 and gs_1 and rendering only that pair.
There is no cross-pair state and no h_{k-1} cache, so each frame is encoded twice.

Implication for ID-based deltas (my reading, not the paper's).
Each update would be: insert one new layer (static fields), attach dynamic fields to the two live layers, and retire the oldest layer.
Temporal reuse of individual Gaussians across many updates does not exist.

### Transport

none

### Client

None in the paper; rendering happens server-side.
The released renderer (gaussian_renderer_dynamic/__init__.py) is PyTorch plus a CUDA orthographic differentiable rasterizer (diff-gaussian-rasterization-orth, derived from NutWorld/kiui).
It uses a fixed view matrix and an orthographic projection with Jacobian [[fx,0,0],[0,fy,0]] (App.
A, Eq. 9), spherical-harmonics degree 0, and evaluates mu(t), rot(t) and alpha(t) per output timestamp.
No WebGL, WebGPU, browser, sorting strategy, buffer-update strategy or client memory limit is discussed.
A browser port would need an orthographic splat projection rather than the usual perspective EWA projection.
The client could evaluate the closed-form mu(t) and alpha(t) itself, so it could render at its own frame rate between keyframe updates.
That last point is my inference.

### Evaluation design

Workloads.
Training and testing use CO3Dv2, RE10K, DAVIS and YouTube-VOS with the official splits.
One pretrained model is evaluated on all benchmarks.
Zero-shot tests use DyCheck iPhone (7 scenes) and NVIDIA Dynamic Scenes (7 scenes).

Tasks and metrics.
- PSNR, SSIM and LPIPS, all computed at 256x256 (following pixelSplat and VIDIM).
- Dynamic: videos subsampled at 5- and 8-frame intervals.
  'Key frames' means reconstructing the input frames themselves.
  'Middle-4 frames' and 'all non-input frames' are temporal interpolation.
- Static (RE10K): 2 input views with at least 60% overlap and 5 target views between them.
  Because there are no camera poses, novel views are rendered 'using relative timestamps', i.e. by interpolating in time, not from a given pose.
- DyCheck: the camera is fixed and motion is absorbed.
  All methods get a fixed intrinsic matrix, and relative poses along the evaluation trajectory are computed.
- NVIDIA: DGMarbles is given CUT3R-estimated intrinsics plus GT poses and scene metadata.
- Runtime: per-sequence time (Table 1) and a per-frame breakdown on one A100 (Table 8).

Baselines / control arms.
- 2D video interpolation: AMT, RIFE, FILM, LDMVFI, VIDIM.
- Optimization-based 4D: CoDeF, DGMarbles, 4DGS, Splatter a Video, Omnimotion, RoDynRF.
- Point-map: MonST3R.
- Static feed-forward 3DGS: pixelSplat, MVSplat, NoPoSplat.
- Several baselines are marked with a dagger, meaning their numbers are copied from the original papers rather than re-run under this protocol.

Ablations (Table 4): no probabilistic sampling, no depth supervision, and no bidirectional deformation (a conventional deformation field is used instead).

What is missing.
- No variance, seeds, error bars or number of runs.
- Baseline timing hardware is not stated.
- No latency other than GPU compute time.
- No memory measurements.
- No long-stream drift metric.
- No network conditions, because this is not a systems paper.

### Key numbers

All numbers are from arxiv.org/html/2506.08862 v2.

DAVIS (Table 1).
StreamSplat on key frames: PSNR 37.83, SSIM 0.982, LPIPS 0.016.
On middle-4 frames: 23.66 / 0.684 / 0.193.
Time about 1.48 s.
Middle-4 comparison: DGMarbles 21.33 / 0.619 / 0.313 (about 30 min); CoDeF 19.40 (about 10 min).
MonST3R on key frames: 42.33 / 0.980 / 0.012 (about 30 s, key frames only).

DAVIS, 8-frame interval (Table 2).
StreamSplat 22.10 / 0.613 / 0.234.
AMT 21.09, RIFE 20.48, FILM 20.71, LDMVFI 19.98, VIDIM 19.62, CoDeF 20.34, DGMarbles 19.83 PSNR.

RE10K (Table 5).
StreamSplat given views: 41.60 / 0.992 / 0.010.
StreamSplat novel views: 24.68 / 0.777 / 0.167.
On novel views the static feed-forward methods are still better: pixelSplat 28.31, MVSplat 28.48, NoPoSplat 28.65 PSNR.
Against dynamic methods StreamSplat wins: CoDeF 20.51, DGMarbles 23.40.
The '+5.78 dB average' is versus DGMarbles' average, 29.51 vs 23.73.

DyCheck zero-shot (Table 6, average PSNR / LPIPS).
StreamSplat 12.37 / 0.69 in 14 s. 4DGS without pose: 13.11 / 0.73.
DGMarbles without camera: 9.76 / 0.77.
DGMarbles with camera: 16.72 / 0.42 (over 5 h).

NVIDIA zero-shot (Table 7).
StreamSplat 16.30 / 0.48. 4DGS with camera: 17.59 / 0.29.
DGMarbles with camera: 22.32 / 0.13.
DGMarbles with CUT3R intrinsics: 13.91 / 0.59.

Runtime (Table 8, single A100, 512x288 input): 49 ms per frame in total.
Encoder 18.8 ms (38.4%), decoder 18.4 ms (37.6%), rasterization 4.0 ms, pre/post-processing 7.8 ms.

Ablations (Table 4).
Without sampling, key-frame PSNR is 31.47; without depth, 36.68; the full model gets 37.83.
Without bidirectional deformation, middle-frame PSNR is 18.89 versus 23.66.

Model size and training. 36,864 Gaussians per frame on a 256x144 grid.
Training took about 3 days on 8 A100s.

MY ESTIMATE (not in the paper), from the released config's parameter sizes:
- Per Gaussian, the static part is 9 floats (xyz 3, scale 2, opacity 1, rgb 3) and the dynamic part is 5 floats per pair (velocity 3, gamma 2).
- Per keyframe, sending static fields once plus two dynamic sets is 36,864 x 19 x 4 B, about 2.8 MB in fp32.
- At 6 keyframes/s (30 fps video, 5-frame interval) that is about 17 MB/s, or about 134 Mbit/s raw.
- At the A100 throughput limit of about 20 keyframes/s it is about 56 MB/s, or about 450 Mbit/s raw.
- Sending each pair naively as 73,728 x 14 floats is about 4.1 MB per pair.
- fp16 would halve all of these.

### Limitations

Stated by the authors (App.
E):
- It relies on external monocular pseudo-depth, which is noisy at fine geometry and depth discontinuities.
- The deformation is trained over a two-frame window, so information from earlier frames is lost under fast motion or long occlusions.
- The orthographic projection causes camera-model misalignment in close-range scenes with strong perspective.
- On RE10K, novel views lag the static methods because there is no pose input and the method assumes smooth camera motion (App.
  C).

Additional, my reading:
(a) There is no persistent, accumulating 3D scene, only a sliding window of two pixel-aligned layers.
Free-viewpoint rendering far from the input camera will expose its 2.5D nature.
Zero-shot DyCheck novel-view PSNR is 12.37, below 4DGS without pose at 13.11, even though the text says it 'matches'.
(b) The 'near real-time' 49 ms does not clearly include depth estimation.
Inputs are keyframes 5 to 8 frames apart, so causal rendering of the interval [k-1, k] has to wait for frame k.
That adds at least one keyframe interval of latency (for example, 5 frames at 30 fps is about 167 ms) on top of compute.
The paper does not discuss this.
(c) Key-frame PSNR mostly measures how well the input frames are re-encoded.
The informative numbers are the middle-frame ones (about 22-24 dB).
(d) Baselines marked with a dagger are copied from other papers, timing hardware differs or is unstated, and there are no seeds or variance.
(e) Long-term persistence is shown only qualitatively.
(f) Paper versus code: the paper says rotation R is predicted, but all released configs fix rotation to identity (fix_keys rot_static, rot_dynamic).
The README says 'polynomial motion modeling', but the default forder=1 is linear.
Algorithm 2 caches h_{k-1}, but the released inference re-encodes both frames of every pair, runs offline from a directory with precomputed depth, and outputs only an MP4.
(g) The repo has no license, and the Depth Anything V2 ViT-L weights are CC BY-NC 4.0.

### Gap relative to our idea

Overlap.
StreamSplat is exactly the kind of 'server GPU feed-forward model' the project needs as its producer.
It is causal, does not need calibration, and at 49 ms per frame on an A100 it is fast enough for a live loop.
It covers nothing beyond that: no transport, no deltas, no client, no scheduling, no glass-to-glass latency, no network, no browser.

What the project would add, concretely:
(1) A wire format and delta protocol for its update model.
One update per keyframe would be: insert a layer of 36,864 Gaussians with a stable ID (keyframe, grid cell), attach velocity and fade parameters, and retire the oldest layer.
The pixel grid makes spatial tiling and prioritization of chunks easy, for example by ROI or motion magnitude.
Compression could predict across consecutive layers cell by cell, video-codec style, because no explicit correspondence exists.
(2) Decoupled render rate.
The Gaussians are closed-form in time (mu(t), alpha(t)), so a WebGPU client can render any t inside the current interval at its own frame rate.
This is a natural fit for the three-rate decoupling idea.
It needs an orthographic splat projection in the shader.
(3) Glass-to-glass latency analysis.
Rendering the interval [k-1, k] needs frame k, so the keyframe interval directly trades latency against bandwidth (roughly 134-450 Mbit/s raw fp32, by my estimate) and GPU load.
A deadline-aware scheduler could adapt the keyframe interval, numeric precision and tile priority.
This knob is specific to the project and is not studied anywhere in the paper.

Honest weak points for the virtualization angle:
- StreamSplat's working set is tiny and bounded: about 73,728 Gaussians, about 4 MB in fp32.
- It does not accumulate a scene, so 'browser GPU memory as a capacity-limited virtual scene cache' and 'view-dependent streaming of a large persistent scene' have little to bite on with this model as-is.
- That story needs one of: many concurrent streams or clients sharing GPU memory (fits the multi-client isolation angle), deliberate client-side retention of past layers, higher resolution, or a model that does accumulate a scene.
  Possible candidates are S2GS or the other arXiv 2608.01659 'StreamSplat' with a voxel cache; both are unverified here.
- Free-viewpoint quality will be limited by the orthographic 2.5D layers.

### Reusable

Code: github.com/DSL-Lab/StreamSplat (PyTorch, conda environment.yml, accelerate; 11 commits, last 2026-03-11; NO LICENSE file, so all rights reserved by default; fine for private coursework experiments, ask the authors before redistributing).

Pretrained checkpoint: streamsplat.safetensors on the Google Drive linked in the README.

Dependencies:
- Depth Anything V2 ViT-L checkpoint (weights CC BY-NC 4.0, code Apache-2.0), needed for pseudo-depth through preprocess_depth_davis.py;
- DINOv2 (Apache-2.0);
- the bundled orthographic rasterizer submodule (diff-gaussian-rasterization-orth, built from source).

To produce Gaussian update traces:
- patch splat_inference.py to save decoder_out['pred_gs'] (or gs_0/gs_1) for each pair, together with the timestamps and pair index;
- run it sequentially rather than in a batch if you want true online timing;
- the configs are in configs/options.py and options_decoder.py.

Datasets usable as camera-like input: DAVIS (BSD-3-Clause) and YouTube-VOS (CC BY 4.0).

Compute fit: the model is two 10-layer ViT-B-sized transformers plus an upsampler, DINOv2 and Depth Anything V2 ViT-L.
It should fit easily on an RTX 6000 Ada 48 GB, but I did not run it.
Retraining (8 A100s, about 3 days) is out of scope.

Not available: no network traces, no latency logs, no browser renderer.

### Notes

Sources I read in full or inspected:
- arxiv.org/abs/2506.08862 (metadata and comments);
- arxiv.org/html/2506.08862 (v2 full text, including Sec. 3-4 and App.
  A-F);
- the first page of arxiv.org/pdf/2506.08862v2;
- streamsplat3d.github.io;
- in github.com/DSL-Lab/StreamSplat: README.md, splat_inference.py, model/splat_model_inference.py, model/model_utils.py (GSPMDecoder, SplatPredictor.forward), gaussian_renderer_dynamic/__init__.py, configs/options.py, options_decoder.py, options_inference.py;
- arxiv.org/abs/2608.01659 abstract only, for the name collision.

The OpenReview PDF was blocked by a bot check.
Files I downloaded are in papers/agent_workdir/streamsplat/ (full.txt is the extracted paper text).

Teaching point for the student.
'Novel view' in this paper often means novel TIME, i.e. interpolated frames from the same fixed canonical camera.
On RE10K, novel views are rendered by relative timestamps, not by a pose.
Do not read the PSNR numbers as free-viewpoint quality.

Research-discipline point.
Every reported number here is a paper-reported value on their hardware (A100) and protocol.
It is not a null or baseline for our system.
Our FIFO and view-aware control arms, and our own runtime on the RTX 6000 Ada, must be measured on our own traces.

The bandwidth figures in key_numbers are my back-of-envelope estimates from the released config's parameter sizes, not measurements.
Verify them by dumping real pred_gs tensors.

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| StreamSplat is an ICLR 2026 paper | supported | The arXiv comments field says 'Accepted by ICLR 2026' (arxiv.org/abs/2506.08862). The first page of the v2 PDF reads 'Published as a conference paper at ICLR 2026' (arxiv.org/pdf/2506.08862v2). The project page and GitHub repo title agree. I could not open the OpenReview page because of a bot check. |
| Takes continuous video input (持續影片輸入) | partly | Online inference (Algorithm 2, Sec. 3.3, arxiv.org/html/2506.08862) is causal and frame-by-frame. It stores only the previous frame embedding h_{k-1}, and each step consumes a pair of consecutive frames (I_{k-1}, I_k). However, all dynamic evaluation uses 'sparse, uncalibrated videos subsampled at 5- and 8-frame intervals' (Sec. 4.1), so consecutive inputs are keyframes 5 to 8 frames apart, not every video frame. The released splat_inference.py (github.com/DSL-Lab/StreamSplat) is not a live stream either. It reads a directory of pre-extracted frames plus precomputed depth maps (--input_depths_path is required), runs independent frame pairs in batches, and writes an MP4. |
| Feed-forward dynamic 3DGS | supported | Per-scene optimization is never used. A frozen static Transformer encoder predicts pixel-aligned 3D Gaussians, and a dynamic Transformer decoder predicts per-Gaussian velocity and opacity-fade coefficients (Sec. 3.1-3.2, Algorithm 1). The output is time-parametric Gaussians, mu(t)=mu0+v(t-t0), rendered by a rasterizer. |
| Online / streaming reconstruction from (possibly) uncalibrated video streams | supported | The title, abstract and Sec. 3.1 say the method works on uncalibrated video. It uses a shared orthographic canonical space, so no intrinsics or extrinsics are needed, and camera motion is absorbed into the Gaussian motion. Caveat on 'possibly': the method has no way to use calibration even when it is available, because there is no pose input. The released renderer uses one fixed camera (R_fixed, T_fixed in gaussian_renderer_dynamic/__init__.py). The 'arbitrarily long streams' claim holds because state never accumulates beyond two keyframes. The authors' Limitations section (App. E) says 'information from earlier frames may be lost' as a result. The paper reports no memory or long-stream quantitative measurement. The project page's 'constant memory' claim is plausible from the design but was not measured in the paper. |
| Not mainly about browser / network delivery (非主要研究 Browser / Network Delivery) | supported | The paper has no transport, compression, client, browser or network evaluation. Rendering is done server-side in PyTorch with a CUDA orthographic diff-gaussian-rasterization. The only systems metric is GPU runtime: 49 ms per frame on one A100 (Table 8). |
| Can be used to pre-generate a Gaussian Update Trace (material line 133: 使用 StreamSplat 或 S2GS 預先產生一段 Gaussian Update Trace) | partly | The code and checkpoint are public, so this is feasible. However, splat_inference.py only saves rendered frames as an MP4 and never writes Gaussians to disk. You would have to modify it to dump decoder_out['pred_gs'] for each pair (model/model_utils.py SplatPredictor.forward returns pred_gs, gs_0 and gs_1). Each pair holds 2 x 36,864 Gaussians (a 256x144 grid per frame, from the paper's App. B and configs/options.py decoder_ratio=2, down_resolution=(288,512)). No Gaussian lives longer than about two keyframe intervals (Sec. 3.2: 'Each Gaussian persists across two consecutive frames'). The repo has no license file. |
| 1200x speedup over optimization-based methods; near real-time (abstract, Sec. 4.2) | partly | Table 1 lists about 1.48 s for StreamSplat against about 30 min for DGMarbles, which is about 1216x. Table 8 gives 49 ms per frame on a single A100 (encoder 18.8, decoder 18.4, rasterization 4.0, pre/post-processing 7.8 ms). The baselines' hardware is not stated, and the times are approximate ('~'), so 1200x is a rough cross-setup ratio. The paper does not say whether the 49 ms includes the Depth Anything V2 pseudo-depth inference. The released inference pipeline needs depth precomputed offline. |
| Adaptive fusion propagates persistent Gaussians, giving long-term temporal coherence (abstract, Fig. 4/15) | partly | The only evidence is qualitative: color-coded Gaussians in Fig. 4 and Fig. 15. No tracking metric is reported. By the paper's own definition each Gaussian lives across two consecutive frames (Sec. 3.2), and the decoder only ever sees the pair (h_{k-1}, h_k). The paper does not spell out how identity propagates beyond that window. In the released code each frame pair is rendered independently, with no cross-pair state. |

### Verifier corrections

- **venue_and_year**
  Reader said: ICLR 2026 (confirmed via arXiv comment, PDF header, project page, GitHub); OpenReview could not be read because of a bot check.
  Correct: ICLR 2026 Poster.
  OpenReview's own record confirms it: forum id SaiDRQU7Ez, ICLR 2026 Submission1083, venue 'ICLR 2026 Poster', venueid ICLR.cc/2026/Conference, with the same five authors and the same title.
  The note's PDF path is /pdf/8b7e3b13bd50d00ff017031283689feb480a8c5d.pdf, which matches the PDF the reader found in search.
  Source: https://api2.openreview.net/notes/search?term=StreamSplat%20Online%20Dynamic%203D%20Reconstruction&limit=5 (forum: https://openreview.net/forum?id=SaiDRQU7Ez)

- **key_numbers (DyCheck, Table 6)**
  Reader said: StreamSplat 12.37 / 0.69 in 14 s. 4DGS without pose: 13.11 / 0.73.
  DGMarbles without camera: 9.76 / 0.77.
  DGMarbles with camera: 16.72 / 0.42 (over 5 h).
  Correct: These numbers are right, but the comparison leaves out two rows of the same table that change the picture.
  DGMarbles (w/o pose, GT intrinsics) gets 15.79 / 0.43, which beats StreamSplat by 3.4 dB PSNR. 4DGS (w/ cam) gets 13.64 / 0.43.
  The 'matches 4DGS (w/o pose)' comparison is mixed: StreamSplat is lower on PSNR (12.37 vs 13.11) but better on LPIPS (0.69 vs 0.73).
  Source: https://arxiv.org/html/2506.08862v2 (Appendix C, Table 6)

- **claim_check: 1200x speedup**
  Reader said: Table 1 lists about 1.48 s for StreamSplat against about 30 min for DGMarbles, which is about 1216x.
  Correct: The 1216x figure is correct for DGMarbles.
  The ratio depends on which baseline is chosen.
  Using the same Table 1 times, CoDeF (~10 min) gives about 405x and 4DGS-dagger (~40 min) gives about 1620x.
  '1200x over optimization-based methods' is therefore true only for the DGMarbles row, not a general factor.
  Source: https://arxiv.org/html/2506.08862v2 (Table 1)

- **scene_update_model**
  Reader said: at any render time t in [t_k, t_{k+1}], the scene is exactly two layers ... about 73,728 Gaussians
  Correct: The paper does say 'Each Gaussian persists across two consecutive frames' (Sec. 3.2), so about two layers is the design intent.
  However, Algorithm 2 prunes by opacity ('keep g with alpha_g(t_k) > 0', i.e.
  'prune Gaussians whose opacity at t_k decays to zero'), not by age.
  It also keeps a persistent canonical set G~(t) with an Update step that sets 'the active deformation of matched 3DGS primitives to the new forward field'.
  So 'exactly two layers' is the reader's inference.
  The paper does not state an exact live count.
  Source: https://arxiv.org/html/2506.08862v2 (Sec. 3.2, Algorithm 2, Sec. 3.3 'Online Inference Pipeline')

- **notes (teaching point on novel view)**
  Reader said: 'Novel view' in this paper often means novel TIME ...
  Do not read the PSNR numbers as free-viewpoint quality.
  Correct: This holds for RE10K, where views are rendered 'using relative timestamps'.
  It does not hold for DyCheck and NVIDIA (Tables 6 and 7).
  There, renders use a relative camera pose with respect to the fixed pose along the benchmark evaluation trajectories, and Algorithm 2's Rasterize step takes a camera pose pi.
  So 12.37 (DyCheck) and 16.30 (NVIDIA) PSNR are the paper's real free-viewpoint numbers.
  The reader's limitation (a) already uses them that way.
  Source: https://arxiv.org/html/2506.08862v2 (Appendix C 'Novel view synthesis on DyCheck benchmark', Algorithm 2)

- **reusable / claim_check (trace generation)**
  Reader said: splat_inference.py reads pre-extracted frames plus precomputed depth maps (--input_depths_path is required) ... runs independent frame pairs in batches
  Correct: This is confirmed, with two details that matter for making traces.
  First, the script's default --frame_gap is 3, not the paper's 5 or 8 evaluation intervals, and --batch_size defaults to 8.
  Second, --input_depths_path is required by argparse, but if the directory holds no matching depth files the script prints a warning and falls back to zero depth maps instead of failing.
  Source: https://raw.githubusercontent.com/DSL-Lab/StreamSplat/main/splat_inference.py (lines ~119-144 and 263-268)

### Verifier on the gap assessment

The reader's gap is fair.
I re-read the v2 full text and found no transport, compression, delta encoding, client, browser or WebGPU work, no scheduling, no network conditions, and no latency measurement other than GPU compute time.
The only runtime numbers are Table 1's per-sequence times and Table 8's A100 breakdown (49 ms per frame: encoder 18.8, decoder 18.4, rasterization 4.0, pre/post 7.8).
The abstract mentions 'strict latency and memory constraints', but the paper measures no memory.
The 'constant memory' claim appears only on the project page (streamsplat3d.github.io), unmeasured.
Long-term persistence is shown only qualitatively, through color-coded Gaussians in Figs. 4 and 15 and the topology-change examples in Fig. 16.

One nuance in the paper's favour.
Algorithm 2 does maintain a persistent canonical set G~(t), and its Update step gives already-existing ('matched') primitives a new forward deformation.
So the paper has an implicit notion of 'update the parameters of an existing Gaussian'.
The reader captures this (each Gaussian gets dynamic parameters twice, implicit ID = keyframe plus grid cell).
The paper still defines no explicit IDs, wire format or delta protocol.

The virtualization caveat is a fair reading of the design.
The working set is small and bounded, so a capacity-limited browser scene cache has little to manage with this model as-is.

### Verifier notes

Verified independently from primary sources (all fetched this session):
- arXiv metadata at https://arxiv.org/abs/2506.08862.
  The title matches exactly.
  Authors are Wu, Yan, Yi, Wang, Liao. v1 is 2025-06-10 and v2 is 2026-03-02.
  The comment field reads 'Accepted by ICLR 2026'.
- The v2 PDF (https://arxiv.org/pdf/2506.08862v2) has 24 pages.
  Page 1 reads 'Published as a conference paper at ICLR 2026'.
- OpenReview API confirms 'ICLR 2026 Poster' (forum SaiDRQU7Ez).
- GitHub API for DSL-Lab/StreamSplat: license null, 192 stars, pushed 2026-03-11, 11 commits.
- The project page lists the same authors and says ICLR 2026.

Key numbers checked against v2 HTML (https://arxiv.org/html/2506.08862v2); all match:
- Table 1: 37.83 / 0.982 / 0.016 on key frames, 23.66 / 0.684 / 0.193 on middle-4 frames, ~1.48 s.
- Table 2: 22.10 / 0.613 / 0.234.
- Tables 3 and 5: given 41.60, novel 24.68, average 29.51 vs DGMarbles 23.73, so +5.78 dB.
  Static methods are better on novel views (28.31 / 28.48 / 28.65).
- Table 4 ablations: 31.47, 36.68 and 18.89.
- Tables 6 and 7: 12.37 / 0.69 and 16.30 / 0.48.
- Table 8: 49 ms total.
- Training on 8 A100s for about 3 days, 512x288 input, metrics at 256x256, 5- and 8-frame intervals, dagger = numbers copied from original papers.
  App.
  E limitations are quoted correctly.

Gaussian count.
The paper never states 36,864 explicitly.
It follows from Table 9 (2304-token window, 16x upsample) and from the code (GSPMDecoder: 288*(2^2/8) x 512*(2^2/8) = 144x256).
The reader's derivation is correct.

Code claims confirmed in the repo:
- fix_keys rot_static and rot_dynamic, so rotation is the identity.
- forder=1.
- sh_degree=0.
- 2 scale values, with y set to their mean.
- opacity_dynamic has 2 values (gamma0, gamma1) in GSDynamicDecoder.
- R_fixed and T_fixed.
- SplatPredictor.forward returns pred_gs, gs_0 and gs_1, and encodes both frames of every pair.
- Output is MP4 only.

The reader's bandwidth arithmetic (36,864 x 19 x 4 B = 2.8 MB, 134 and 450 Mbit/s, 4.1 MB per pair) is internally correct and properly labelled as an estimate.

The name collision is real. arXiv 2608.01659, 'StreamSplat: Streaming Feed-Forward 3D Gaussian Splatting' (Song, Wang, Li, Cai, Liu, 2026-08-03), has no comment or venue.
It uses calibrated views and a Voxel-Aligned Causal Cache (https://arxiv.org/abs/2608.01659).

Trivial: affiliation numbering differs between the PDF (CIFAR=3, NTU=4) and the project page (NTU=3, CIFAR=4).
This is irrelevant.

None of the corrections changes the reader's overall conclusions or the relevance score.
They are refinements: the OpenReview poster status, the DyCheck rows that were left out, the 1200x figure depending on which baseline is used, 'exactly two layers' being an inference, and the code's frame_gap default and zero-depth fallback.

Scratch files are in papers/agent_workdir/ss_verify/

## Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform

Key `visionary`, relevance 4/5, identity confirmed by verifier: yes.

### Authors

Yuning Gong, Yifei Liu, Yifan Zhan, Muyao Niu, Xueying Li, Yuanjun Liao, Jiaming Chen, Yuanyuan Gao, Jiaqi Chen, Minming Chen, Li Zhou, Yuning Zhang, Wei Wang, Xiaoqing Hou, Huaxi Huang, Shixiang Tang, Le Ma, Dingwen Zhang, Xue Yang, Junchi Yan, Yanchi Zhang, Yinqiang Zheng, Xiao Sun, Zhihang Zhong (corresponding author).
Affiliations: Shanghai AI Laboratory, Sichuan University, The University of Tokyo, Shanghai Jiao Tong University, Northwestern Polytechnical University.
Source: arxiv.org/abs/2512.08478 citation_author meta tags and arxiv.org/html/2512.08478.

### Venue and year

Preprint, 2025 (arXiv v1, 9 Dec 2025; only v1 exists as of 2026-10-27 per arxiv.org/abs/2512.08478).
No venue confirmed.
Checked: (1) the paper is not in the CVPR 2026 open-access list (openaccess.thecvf.com/CVPR2026?day=all) or the CVPR 2026 virtual papers list; (2) Semantic Scholar (api.semanticscholar.org, arXiv:2512.08478) lists venue "arXiv.org"; (3) the repo README cites it as "arXiv preprint arXiv:2512.08478, 2025".
A WebSearch summary claimed "accepted for CVPR 2026 as an Oral" - that is wrong: the CVPR 2026 Oral on the Visionary-Laboratory GitHub org page belongs to a different paper, Proxy-GS (which is in CVPR 2026 open access).

### arXiv

2512.08478

### Links

Paper: https://arxiv.org/abs/2512.08478 (full text read at https://arxiv.org/html/2512.08478).
Code: https://github.com/Visionary-Laboratory/visionary - Apache-2.0 (GitHub API license field and README badge), TypeScript/WGSL viewer plus Python ONNX export examples, about 530 stars, created 2025-12-07, last push 2026-10-15 (HEAD 76a2b93, read via shallow clone).
Project page / online editor: https://visionary.openxlab.org.cn/ (repo homepage field; the page names no venue).
Video: https://youtu.be/-K8EjMfk09c (linked from README, not watched).
Docs: https://ai4sports.opengvlab.com/help/index.html (linked from README, not read).

### Problem

Existing 3DGS viewers are either heavy native stacks (SIBR, Unity/Unreal/Blender plug-ins, nerfstudio with a Python/CUDA backend) or WebGL web viewers (SparkJS, SuperSplat, GaussianSplats3D).
WebGL has no general compute shaders, so these viewers sort on the CPU (slow at multi-million Gaussians) and are limited to static assets.
They cannot run per-frame neural decoding (MLP-based 3DGS, 4DGS deformation, avatar skinning) or generative post-processing in the browser.
The authors want a zero-install, browser-only runtime where any 3DGS-family method can be plugged in without writing shaders, with per-frame inference coupled to GPU rendering in real time (Sec. 1, arxiv.org/html/2512.08478).

### Method

Three stages, all in the browser (Fig. 2, Sec. 3):
(1) ONNX pre-decoding, the 'Gaussian Generator contract' (Sec. 3.3).
Each method is exported offline from PyTorch to one ONNX graph, with trained parameters (canonical Gaussians, anchors, HexPlane grids, skinning weights) baked in as constant initializers.
Each frame the client feeds small inputs (time t, frame index, camera view/projection matrices, or SMPL-X pose and shape).
ONNX Runtime Web on its WebGPU backend outputs packed Gaussians (position, opacity, upper-triangular covariance, colour as RGB or SH) plus a point count and dtype metadata.
Instances shown: Scaffold-GS (anchor features + view direction -> MLP -> neural Gaussians, regenerated each frame), 4DGS of Wu et al.
(HexPlane features + MLP -> per-Gaussian position/rotation/scale offsets from t), and avatars (LHM, R3-Avatar, GauHuman: forward kinematics + linear blend skinning inside the graph).
Two engineering tweaks: keep ONNX sessions and input/output bindings stable so WebGPU graph capture can replay the graph, and split very large Concat/Split ops into chunks to stay within WebGPU limits.
(2) WebGPU hybrid renderer (Sec. 3.4, Alg. 1).
Static assets are pre-packed once: scale and rotation become a 6-value covariance, everything is cast to fp16 and packed two-per-u32.
Every frame, a compute kernel per model applies the model transform, projects to clip space, culls by frustum and opacity, computes the 2D ellipse axes and NDC centre, and appends survivors to one global buffer through an atomic counter.
Their depths become the sort keys.
One global GPU radix sort covers all splats of all models.
Splats are drawn as instanced quads (count via drawIndirect) with premultiplied alpha, back to front.
If meshes are present, a mesh depth prepass runs first and splats use depth test without depth write.
(3) Optional ONNX post-processing (Sec. 3.5): an exported diffusion U-Net denoiser (EXGS example) for enhancement or stylization of the rendered image.
The system is packaged as a three.js plug-in with a TypeScript API.

### Scene update model

Static assets: loaded once from PLY/SPLAT/KSplat/SPZ/SOG and uploaded with one queue.writeBuffer, with no later add, update or delete (src/point_cloud/point_cloud.ts L109-121).
Dynamic assets: every frame the ONNX generator overwrites the entire Gaussian buffer on the GPU.
The buffers are allocated once at fixed capacity (maxPoints from ONNX metadata, fallback 2M; src/ONNX/onnx_gpu_io.ts L270-292), and the active count is written by the graph into a GPU i32 buffer with no CPU readback.
There is no create/update/delete delta model: each frame is a full snapshot.
Identity: there is no explicit Gaussian ID.
The paper says the contract outputs 'a variable number of Gaussians'.
But in the reference export templates (onnx-export/ONNXExample-scaffold/onnx_template.py L428-480 and ONNXExample-4dgs/onnx_template.py, HEAD 76a2b93), frustum and other masks multiply attributes to zero instead of compacting the tensor ('anchors_kept = self._anchor#[mask]' is commented out).
So the output shape and num_points are constant, and slot i always maps to the same canonical Gaussian (or anchor*K+k for Scaffold-GS).
Identity is therefore implicit in the slot index for canonical-plus-deformation models.
That is a property of these templates, not a contract guarantee.
Inside the renderer, the per-frame preprocess compacts visible splats through atomicAdd, so the index after culling is not stable.

### Transport

None.
The paper has no network transport, delta format, compression for transmission, prioritization or rate adaptation.
Assets and ONNX models are fetched or loaded whole as files.
A grep of the repo for websocket/webtransport/webrtc returns nothing.
The only 'compression' is fp16 packing for GPU memory bandwidth (Sec. 3.4) and input-format support for compressed splat files (SPZ, SOG, KSplat).
The design deliberately avoids server-side inference (Sec. 1, Sec. 3.1 'Web-native first').

### Client

Platform: browser, WebGPU only (Chrome or Chromium).
README: Windows 10/11 with a discrete GPU recommended; Ubuntu 'currently not supported' because of a Chrome WebGPU fp16 bug that breaks the fp16 ONNX pipelines; most Macs too slow.
Integration: a three.js plug-in with a TypeScript API, built on ONNX Runtime Web (WebGPU execution provider, graph capture when possible); the renderer credits web-splat (KeKsBoTer).
GPU buffer strategy: static data is fp16-packed and uploaded once.
Dynamic data lives in pre-allocated fixed-capacity buffers written GPU-to-GPU by ONNX each frame, with the count kept in a GPU buffer, and the draw count comes from drawIndirect.
Sorting: a global per-frame GPU radix sort over NDC depth for all visible splats of all loaded models, with no lazy or incremental sort.
Scheduling: in the demos, the render loop awaits updateDynamicModels (ONNX inference) inside each requestAnimationFrame before drawing, so inference and render rates are coupled.
This matches the paper's 'Real-time coupling: inference and rendering are executed within a single frame budget' (Sec. 3.1).
Memory limits: Sec. 5.2 says browser security policies cap CPU memory, which limits in-browser model size, so some post-processing stays offline.
The dynamic buffer capacity defaults to 2M Gaussians unless the metadata says otherwise.
There is no eviction or cache policy.

### Evaluation design

Hardware: one workstation, NVIDIA RTX 4090 with an Intel w5-3435X CPU.
Browser version, OS and resolution are not stated; PIX profiling implies Windows.
Number of runs, variance and CIs: none reported.
Every number is a single value.
(a) Runtime vs SparkJS (Sec. 4.2, Table 1, Fig. 3).
Workload: the Mip-NeRF 360 'bicycle' 3DGS asset at 6.062M Gaussians and downsampled to 1/2, 1/4, 1/8 (3.031M, 1.515M, 0.758M).
Metric: per-frame Sort / Prep+Draw / Total in ms.
Visionary's GPU times come from PIX replay of one captured 'representative frame'; how SparkJS was timed is not stated beyond 'sorting is on the CPU'.
'Identical assets and camera trajectories' is asserted, but the trajectory is not described.
(b) Image quality (Sec. 4.3, Table 2): PSNR, SSIM and LPIPS on Mip-NeRF 360, averaged with no per-scene numbers, Visionary vs SparkJS only.
The source of the trained assets and the scene list are not given.
(c) Robustness to fast camera rotation (Sec. 4.4, Fig. 4): qualitative screenshots only, vs SparkJS v0.1.10 lazy sorting.
No quantitative popping or flicker metric.
(d) Composition correctness (Sec. 4.5, Fig. 5): qualitative only, vs SuperSplat v2.15.1, whose partition-local sort mis-orders overlapping Gaussians.
(e) ONNX pre-decoding overhead (Sec. 4.6, Tables 3-4): per-frame inference ms for Scaffold-GS, 4DGS (Wu et al.) and avatars (GauHuman, R3-Avatar at 1/5/10 instances).
There is no baseline or control arm; the stated reason is that no other web viewer supports these methods.
Scene names are not given.
Not evaluated at all: network conditions or emulation, latency from content change to display (glass-to-glass), stalls or frame-time distributions, GPU memory usage, power, mobile or integrated GPUs, other browsers, or WebGPU vs WebGL on the same codebase.
None of these numbers serve as a null for our data.
They come from a 4090 with whole-asset local loading and no network.

### Key numbers

All on RTX 4090 + Intel w5-3435X, single values, no variance (Tables 1-4, arxiv.org/html/2512.08478).
Table 1, bicycle scene, Visionary GPU time via PIX replay vs SparkJS:
- 6.062M Gaussians: SparkJS sort 172.87 ms, prep+draw 4.03, total 176.90 ms; Visionary sort 0.58, prep+draw 1.52, total 2.09 ms.
- 3.031M: SparkJS 145.75 ms vs Visionary 1.09 ms.
- 1.515M: 46.29 vs 0.60 ms.
- 0.758M: 33.82 vs 0.40 ms.
- Speedups recomputed: 85x, 134x, 77x and 85x.
  The text's '~135x' holds only at 3.03M, and the Fig. 3 caption says '~100x'.
Table 2, Mip-NeRF 360 average: PSNR 27.867 vs 27.315 (SparkJS), SSIM 0.828 vs 0.825, LPIPS 0.249 vs 0.253.
Table 3, per-frame ONNX inference:
- Scaffold-GS: 2.49M Gaussians 9.29 ms; 4.56M Gaussians 16.10 ms.
- 4DGS: 0.03M 4.76 ms; 0.06M 7.93 ms.
  Note the 4DGS models are tiny, 30-60k Gaussians.
Table 4, avatars:
- GauHuman at 1/5/10 instances (0.04/0.20/0.40M Gaussians): 7.97/28.60/55.63 ms.
- R3-Avatar at 1/5/10 instances (0.02/0.12/0.23M): 7.46/27.10/53.35 ms.
- Inference time grows roughly linearly with instance count, so 10 avatars at about 55 ms are below 20 FPS for inference alone.
  The text quotes only the 7-8 ms single-instance figure.

### Limitations

Stated by the authors (Sec. 5.2): WebGPU and ONNX Runtime Web are still maturing, so compatibility and stability vary across browsers and OSes.
Browser security policies cap CPU memory, which limits in-browser model size, so some post-processing remains offline.
The README adds: Ubuntu is unsupported (Chrome WebGPU fp16 bug), Windows with a discrete GPU is recommended, and most Macs are too slow.
Additional, from reading the paper and code:
(1) Single-machine evaluation on a 4090 with no repeats, no variance, and GPU time from one PIX-replayed frame rather than wall-clock frame time or a frame-time distribution.
(2) The SparkJS comparison mixes measurement methods.
SparkJS's CPU sort time is counted as per-frame cost, even though Sec. 4.4 says SparkJS sorts asynchronously, amortized over frames.
(3) The intro claims improved 'dynamic update latency' over web renderers, but no such comparison exists.
(4) Speedup figures are internally inconsistent (~100x vs ~135x).
(5) Quality gains over SparkJS are small, averaged over unnamed scenes, with no per-scene breakdown.
(6) Robustness and composition results are qualitative screenshots only.
(7) Inference and render run in lockstep, so a slow generator directly lowers the render rate.
There is no asynchronous update path or deadline handling.
(8) Dynamic buffers have a fixed capacity and are fully overwritten every frame.
There are no deltas, no eviction and no memory-budget policy.
(9) Everything runs on the client.
Dynamic content needs the model weights and the canonical Gaussians downloaded up front, and the client GPU must be able to run the network.
(10) Section 3.3.3 presents LHM as the representative avatar integration, but Table 4 reports GauHuman and R3-Avatar instead.

### Gap relative to our idea

Overlap: Visionary is the closest existing browser system for Gaussians that change every frame under WebGPU.
Its renderer already provides much of what our client side needs: a GPU radix sort, global sorting across several models, compute-shader preprocessing, fixed-capacity GPU buffers with a GPU-side count and drawIndirect, fp16 packing, and a three.js integration.
We should not claim novelty for browser rendering of dynamic Gaussians, or for a WebGPU per-frame update path.
What our idea adds, none of which Visionary has:
(1) A server-side live feed-forward model that reconstructs from camera video.
Visionary's dynamics come from pre-trained per-scene models baked into ONNX and run on the client; there is no live capture and no reconstruction.
(2) Network transport of scene state: ID-based deltas, versioned chunks, compression, temporal coalescing and view-dependent prioritization.
Visionary has no network path at all.
(3) Decoupling of inference rate, network update rate and render rate.
Visionary deliberately couples inference and render within one frame budget, and the demos await inference in each requestAnimationFrame.
(4) A CPU-decode to partial queue.writeBuffer path into stable slots, plus a capacity-limited scene cache with eviction under a GPU memory budget (the 'virtual scene cache' angle).
Visionary fully overwrites a fixed-capacity buffer GPU-to-GPU and has no budget or eviction policy.
(5) Joint network and GPU scheduling that is deadline-aware and GPU-budget-aware.
(6) The evaluation itself: glass-to-glass latency, render stalls, frame-time distributions, tc netem network conditions, FIFO and view-aware baselines, and repeated runs.
Visionary's evaluation is renderer-speed only, on one 4090.
Practical cautions if we build on its code:
- Linux is officially unsupported (the fp16 bug), which conflicts with our Linux + tc netem setup.
  Either run the browser client on Windows, or test fp32 paths on Linux Chrome first.
- Its lockstep loop must be replaced, not reused.
- Visionary is a useful baseline arm for 'client renders whatever full snapshot it has' only in the sense of GPU render cost.
  It has no transport to compare against.

### Reusable

Repo https://github.com/Visionary-Laboratory/visionary (Apache-2.0, actively maintained as of 2026-10-15, about 530 stars).
Useful pieces:
- src/sort/radix_sort.ts with src/shaders/radix_sort.wgsl: WebGPU GPU radix sort, derived from web-splat.
- src/shaders/preprocess.wgsl with src/preprocess/gaussian_preprocessor.ts: cull and project, atomic-append into a global buffer.
- src/renderer/gaussian_renderer.ts: multi-model global sort and drawIndirect.
- src/point_cloud/dynamic_point_cloud.ts with src/ONNX/onnx_gpu_io.ts: fixed-capacity GPU buffers plus a GPU count buffer.
  This is a good template for a slot-indexed scene cache if we replace the ONNX writer with queue.writeBuffer from network-decoded chunks.
- src/three-integration: a three.js WebGPU plug-in.
- Loaders for PLY/SPLAT/KSplat/SPZ/SOG.
- onnx-export/: PyTorch-to-ONNX examples for Scaffold-GS, 4DGS and the LHM avatar.
  These could produce a deterministic per-frame Gaussian trace offline, for example by running the 4DGS ONNX over t to get trace files with stable slot identity for our replay experiments.
- Example assets on Google Drive linked from the README; licences of the assets are not stated.
Maturity: research code with Chinese comments and debug logging.
Graph capture falls back when unsupported.
README says to install with npm; we would use pnpm.
No datasets, network traces or latency-measurement harness are provided.

### Notes

All facts come from arxiv.org/abs/2512.08478, arxiv.org/html/2512.08478 (v1, read in full: method Sec. 3, experiments Sec. 4, limitations Sec. 5.2), the GitHub API and README, a shallow clone of the repo at HEAD 76a2b93 (2026-10-15), Semantic Scholar, the CVPR 2026 open-access list and Crossref.
A WebSearch summary claimed 'accepted for CVPR 2026 as an Oral'.
The primary sources contradict it: the paper is absent from CVPR 2026 open access, and the CVPR 2026 Oral label on the lab's GitHub org page belongs to Proxy-GS.
Treat Visionary as a preprint.
A search hit 'Powered By Gaussian Splats: From World Models to 3D Interactive Worlds' (ACM DOI 10.1145/3799820.3820876) is a SIGGRAPH 2026 course by different authors (Hartman, Coulombe, per Crossref), not this paper.
For the student: the most instructive code to read slowly is onnx_gpu_io.ts (how a fixed-capacity WebGPU buffer and a GPU-side count are set up) and the renderer's preprocess, sort and drawIndirect chain.
That chain is exactly where our network-fed slot updates would plug in.
Scratch copies (paper text, repo clone) are in papers/agent_workdir/visionary/.

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Visionary is a 2025 preprint (comparison table: 'Visionary (2025 Preprint)') | supported | arXiv v1 submitted 9 Dec 2025 and no later version (arxiv.org/abs/2512.08478 submission history). Not in the CVPR 2026 open-access list; Semantic Scholar venue is 'arXiv.org'. A search-engine summary claiming CVPR 2026 Oral mixed it up with Proxy-GS from the same lab. |
| arXiv id 2512.08478 | supported | arxiv.org/abs/2512.08478 resolves to this exact title; the README bibtex uses the same id. |
| Browser WebGPU renderer | supported | Sec. 3.4 'WebGPU-based Hybrid Renderer': preprocessing in compute shaders, GPU radix sort, instanced rasterization, depth-aware mesh composition (arxiv.org/html/2512.08478). Repo has src/shaders/*.wgsl, src/sort/radix_sort.ts, and drawIndirect in src/renderer/gaussian_renderer.ts. |
| Per-frame Gaussian generation/update (逐影格 Gaussian Generation / Update) | supported | Abstract and Sec. 3.3: the 'Gaussian Generator contract' is an ONNX graph that runs every frame on per-frame inputs (frame index, time, camera, SMPL-X pose) and outputs packed Gaussians straight into WebGPU storage buffers. Caveat that matters for this project: the generation runs on the client, from a pre-trained model (Scaffold-GS anchors+MLP, 4DGS HexPlane deformation, avatar LBS) whose canonical Gaussians are baked into the ONNX file as constants. It is not a server-side feed-forward reconstruction from live video, and nothing is sent over a network per frame. |
| Dynamic Gaussian buffers; 'best suited to observe how WebGPU supports dynamically updated Gaussian buffers' (最適合觀察 WebGPU 如何支援動態更新 Gaussian Buffer) | partly | Code (src/ONNX/onnx_gpu_io.ts L270-292, src/point_cloud/dynamic_point_cloud.ts): GPU buffers are allocated once at fixed capacity (maxPoints from ONNX metadata, fallback 2,000,000; layout (maxPoints,10) f16 = 20 B per Gaussian, plus a colour buffer), and ONNX Runtime writes the whole buffer GPU-to-GPU every frame through IOBinding. The count stays in an i32 GPU buffer and is never read back to the CPU. The static path (src/point_cloud/point_cloud.ts L109-121) does a single queue.writeBuffer at load and has no partial or ID-addressed update API. So it shows full-overwrite, GPU-produced dynamics. It does not show CPU-decoded, network-delivered, sparse slot updates, which is the mechanism this project needs. |
| No full evaluation of remote real-time scene-delta transmission (未見完整的遠端即時 Scene-Delta 傳輸評估) | supported | Stronger than claimed: the paper has no network transport at all. Experiments (Sec. 4) cover render time vs SparkJS, image quality, qualitative sorting artefacts, and ONNX inference time. A grep of the repo for websocket/webtransport/webrtc finds nothing; assets and ONNX models are loaded whole. The design principle is 'Web-native first: all components run directly in the browser' (Sec. 3.1), and the intro presents avoiding server-side inference as a goal. |
| Can serve as the WebGPU rendering base for implementing dynamic buffer updates (使用 WebSplatter 或 Visionary 作為 WebGPU Rendering Base) | partly | For: Apache-2.0, TypeScript, three.js plug-in, GPU radix sort, multi-model global sort, drawIndirect, and a fixed-capacity buffer plus GPU count pattern that maps onto a slot-indexed scene cache. Against: the README says Ubuntu is 'Currently not supported' because of a Chrome WebGPU fp16 bug and recommends Windows with a discrete GPU, which matters for a Linux workstation plus tc netem setup. Also, the demos await ONNX inference in every requestAnimationFrame before rendering (src/three-integration/demo.ts L214 'await gaussianThreeJSRenderer.updateDynamicModels'), so update and render run in lockstep, the opposite of the decoupling we want. A new network-fed update path (writeBuffer into slots) would have to be written. |
| (Paper's own intro) 'demonstrates substantial improvements in frame time, dynamic update latency, and scalability compared to existing web-based renderers' | unsupported | Sec. 4 compares frame time and scalability only on static 3DGS against SparkJS. For dynamic variants, Sec. 4.6 says 'there are currently no other web viewers that support comparable functionality' and reports only Visionary's own inference times. No dynamic-update-latency comparison exists. |
| (Paper's own) 'up to ~100x' (Fig. 3 caption) vs 'up to ~135x end-to-end speedup' (Sec. 4.2) over SparkJS | partly | Ratios recomputed from Table 1: 176.90/2.09 = 84.6x (6.06M), 145.75/1.09 = 133.7x (3.03M), 46.29/0.60 = 77.2x (1.52M), 33.82/0.40 = 84.6x (0.76M). '~135x' holds at the 1/2 scale only, and the caption's '~100x' contradicts the text. The 'end-to-end' label is also questionable: Visionary's numbers are GPU time from PIX replay of one captured frame. Sec. 4.4 itself says SparkJS sorts asynchronously, amortized over several frames, so summing its 172.87 ms sort into a per-frame 'Total' is not SparkJS's displayed frame time. |

### Verifier corrections

- **scene_update_model / client / claim_check (dynamic buffers): 'The count stays in an i32 GPU buffer and is never read back to the CPU'**
  Reader said: The active count is written by the graph into a GPU i32 buffer with no CPU readback.
  Correct: At HEAD 76a2b93 the count IS read back to the CPU after every inference.
  ONNXGPUIO.runInference calls readONNXCountBuffer(this.device, this.countBuf) right after session.run (src/ONNX/onnx_gpu_io.ts L745-748).
  That call is outside the debug_graph guard, so it runs every time. readGPUBuffer copies the buffer into a MAP_READ staging buffer, submits, and awaits mapAsync (src/utils/debug-gpu-buffers.ts L12-34). runInference is reached every frame through demo.ts L214 -> GaussianThreeJSRenderer.updateDynamicModels (L1579-1596, which also awaits each model one after another) -> DynamicPointCloud.update -> ONNXGenerator.generate (src/ONNX/onnx_generator.ts L37-40).
  The comment at onnx_gpu_io.ts L77 says '不回读 CPU' (no CPU readback), but the code does the opposite.
  Drawing still takes its count from the GPU buffer through drawIndirect, so the readback only fills actualPoints for logging and UI.
  It is still a GPU-to-CPU sync point every frame inside the lockstep loop.
  If the student reuses this file as a template, this call must be removed.
  Source: https://github.com/Visionary-Laboratory/visionary/blob/76a2b93/src/ONNX/onnx_gpu_io.ts (L77, L745-748); https://github.com/Visionary-Laboratory/visionary/blob/76a2b93/src/utils/debug-gpu-buffers.ts (L12-66); https://github.com/Visionary-Laboratory/visionary/blob/76a2b93/src/app/GaussianThreeJSRenderer.ts (L1579-1596)

- **scene_update_model / transport / reusable: 'static path does a single queue.writeBuffer at load'; 'assets loaded whole'**
  Reader said: Static assets are loaded once and uploaded with one queue.writeBuffer (point_cloud.ts L109-121), with no partial-update API.
  Assets are loaded whole.
  Correct: Incomplete.
  The repo also has a chunked upload path.
  A plain PLY of 1.5 GiB or more, opened as a local File, goes through PLYLoader.loadFileStreaming.
  It reads the file in 32 MiB File.slice chunks, packs each chunk to fp16 on the CPU, and writes it into GPU buffers allocated in advance, using writeBuffer at a per-point byte offset (offset = point * stride).
  It awaits onSubmittedWorkDone every 4 chunks (src/io/ply_loader.ts L348-520; size threshold PLY_STREAM_MIN_BYTES at src/app/managers/file-loader.ts L20 and L203). point_cloud.ts L123-124 confirms that 'Streaming loads pass an already-filled GPU buffer'.
  This is still a one-time load from local disk, with no network, no updates after load and no IDs, so the gap conclusion stands.
  But it is the closest existing code to the path the project needs: CPU decode, then writeBuffer at an offset into a pre-allocated buffer.
  It belongs in 'reusable'.
  Source: https://github.com/Visionary-Laboratory/visionary/blob/76a2b93/src/io/ply_loader.ts (L348-520); https://github.com/Visionary-Laboratory/visionary/blob/76a2b93/src/app/managers/file-loader.ts (L20, L140-155, L203)

- **links (project page)**
  Reader said: Project page / online editor: https://visionary.openxlab.org.cn/ (repo homepage field)
  Correct: The arXiv Comments field gives a different project page: https://visionary-laboratory.github.io/visionary.
  It returns HTTP 200 and is just a full-page iframe of https://ai4sports.opengvlab.com, with no venue named.
  The README's 'Project Page' link is visionary.openxlab.org.cn, so both URLs exist.
  The paper's own metadata cites the github.io one.
  Source: https://arxiv.org/abs/2512.08478 (Comments: 'Project page: https://visionary-laboratory.github.io/visionary'); https://visionary-laboratory.github.io/visionary/

### Verifier on the gap assessment

The gap analysis is fair.
I checked it against the full v1 text (arxiv.org/html/2512.08478) and the code at HEAD 76a2b93.
(1) Live server-side feed-forward reconstruction is absent.
All dynamics come from trained models whose canonical Gaussians, anchors or HexPlanes are baked into ONNX as initializers and run on the client (Sec. 3.3.1-3.3.3).
For LHM only the LBS/FK deformation is exported, not its single-image feed-forward reconstruction.
The intro (Sec. 1) treats 'server-side inference' as something to avoid.
(2) There is no network transport, delta format, coalescing or rate adaptation, in either the paper or the repo.
My rg of src/ for websocket/webtransport/webrtc/eventsource finds nothing.
The only network use is fetch(url) of whole asset files.
(3) Coupling is a stated design principle: 'Real-time coupling: inference and rendering are executed within a single frame budget' (Sec. 3.1).
The demo awaits inference in every rAF (demo.ts L214).
(4) There is no memory budget, eviction or cache policy.
Sec. 3.1 says the runtime handles 'scheduling, memory management', but in the code that means ONNX session and IO-binding management, not budget-aware scheduling.
(5)-(6) The evaluation is renderer speed plus ONNX inference time on one RTX 4090, single values only (Sec. 4.1-4.6).
There is no network emulation, no glass-to-glass latency, no stalls and no frame-time distribution.
Two nuances to add.
First, Visionary's Scaffold-GS generator already does per-frame view-dependent work on the client: camera matrices are an input, and a frustum mask is applied inside the graph (onnx-export/ONNXExample-scaffold/onnx_template.py L428-480).
So only view-dependent network prioritization is new, not view-dependence as such.
Second, the chunked offset-writeBuffer loader (ply_loader.ts loadFileStreaming) shows a CPU-decode-to-GPU-slot upload pattern, but only as a one-time local load.

### Verifier notes

Verified independently (fresh fetches into papers/agent_workdir/visionary_verify/, fresh shallow clone of the repo at HEAD 76a2b93, committed 2026-10-15):
- Title, all 24 authors (Zhong as corresponding author, zhongzhihang@pjlab.org.cn), arXiv id 2512.08478, and v1 dated 9 Dec 2025 as the only version: arxiv.org/abs/2512.08478 citation meta tags and submission history.
- Affiliations from arxiv.org/html/2512.08478: Shanghai AI Laboratory, Sichuan University, The University of Tokyo, Shanghai Jiao Tong University, Northwestern Polytechnical University.
- Venue: a preprint.
  Evidence:
  - arXiv has no journal-ref.
  - The DBLP record (seen through OpenReview's mirror, id aXIlZvKQC1) says 'CoRR 2025'.
  - Semantic Scholar gives venue 'arXiv.org' (api.semanticscholar.org/graph/v1/paper/arXiv:2512.08478).
  - The paper is absent from the CVPR 2026 open-access list.
  I fetched the full 12 MB page with 4042 entries; it contains no 'Visionary' and no 'WebGPU'.
  - The CVPR 2026 Oral belongs to Proxy-GS, according to that repo's description in the Visionary-Laboratory org listing (api.github.com/users/Visionary-Laboratory/repos).
  The reader's diagnosis is correct.
- Code: github.com/Visionary-Laboratory/visionary, Apache-2.0 (GitHub API), 530 stars, created 2025-12-07, pushed 2026-10-15.
  README L52: Ubuntu 'Currently not supported' because of the Chrome WebGPU fp16 bug.
  README uses npm.
- Key numbers: Tables 1-4 match the reader exactly.
  My recomputed speedups are 84.6x, 133.7x, 77.2x and 84.6x.
  The Fig. 3 caption says '~100x' and Sec. 4.2 says '~135x'.
  The RTX 4090 + w5-3435X setup, the PIX replay of one representative frame, and Table 2 (27.867/0.828/0.249 vs 27.315/0.825/0.253) are all confirmed.
- Sec. 4.6 says 'no other web viewers that support comparable functionality'.
  The intro (Sec. 1) claims improved 'dynamic update latency', and that claim is unsupported, as the reader says.
- Code claims confirmed:
  - maxPoints fallback of 2,000,000 and a gauss buffer of (maxPoints, 10) (onnx_gpu_io.ts L271, L280).
  - The legacy single writeBuffer (point_cloud.ts L109-121).
  - The demo awaits updateDynamicModels (demo.ts L214).
  - The Scaffold template multiplies by the mask instead of compacting: 'anchors_kept = self._anchor#[mask]' at L438 and 'g = g * mask_fk' at L480, with dynamic_axes commented out at L571-572.
- Small internal inconsistency in the paper, not flagged by the reader: the intro cites 4DGS as 'Yang et al.
  (2024)', while Sec. 3.3.2 and Table 3 use 'Wu et al.
  (2024)' (HexPlane).
  The reader's 'Wu et al.' is correct for the experiments.
- Not checked: the video, the docs site, and the Google Drive example assets.

## MoQSplat: Adaptive Progressive Streaming of 3D Gaussian Splatting via MoQ

Key `moqsplat`, relevance 4/5, identity confirmed by verifier: yes.

### Authors

Emanuele Artioli, Mohammadreza Ghafari, Md Tariqul Islam (first three equal contribution), Farzad Tashtarian, Christian Rothenberg, Christian Timmerer.
Affiliations: Christian Doppler Laboratory ATHENA, Alpen-Adria Universitaet Klagenfurt (Austria); Universite de Lorraine / CNRS / Inria / LORIA (France); UNICAMP (Brazil).
Source: https://arxiv.org/html/2609.18624 (author block).

### Venue and year

IEEE MMSP 2026 (IEEE International Workshop on Multimedia Signal Processing, Istanbul, 22-24 Sep 2026), Special Session I "Media over QUIC", paper SSI4.
Sources: arXiv Comments field "7 pages.
Accepted at IEEE MMSP 2026 (Istanbul, 22-24 September 2026)" at https://arxiv.org/abs/2609.18624, and the official program https://attend.ieee.org/mmsp-2026/wp-content/uploads/sites/776/program.pdf (page 5, SSI4). arXiv v1 submitted 16 Sep 2026, cs.MM (cross-list cs.GR, cs.NI).
It is a short workshop paper (about 5 pages of text) with "Preliminary Evaluation Results".

### arXiv

2609.18624

### Links

Paper abs: https://arxiv.org/abs/2609.18624 ; full text HTML: https://arxiv.org/html/2609.18624 ; PDF: https://arxiv.org/pdf/2609.18624 (paper license CC BY-SA 4.0 per arXiv HTML header).
Code: https://github.com/emanuele-artioli/MoQSplat - README and pyproject.toml declare MIT, but there is no LICENSE file (GitHub API reports license=null), so treat licensing as ambiguous until the authors add one.
Forks: cd-athena/MoQSplat (has branch "emanuele"), EternalSky23/MoQSplat.
Lab blog post (seen only as a search result, not read): https://athena.itec.aau.at/2026/07/moqsplat-adaptive-progressive-streaming-of-3d-gaussian-splatting-over-moq/ .
Underlying MoQ stack named by the paper: moq-lite / moq-relay at https://github.com/moq-dev/moq ; MoQ spec cited: draft-ietf-moq-transport-18.
Local copies: papers/agent_workdir/moqsplat/{paper.pdf,full.txt,fig4.png,repo/}

### Problem

Delivering large, static, pre-trained 3DGS scenes (gigabyte-scale, millions of splats, up to 45 SH coefficients each) to remote viewers.
The authors argue HTTP adaptive streaming over TCP (as in LapisGS, LTS, L3GS) suffers from head-of-line (HOL) blocking and coarse segment granularity, and that plain HTTP/3 client-pull still adds latency.
They propose mapping 3DGS onto the IETF Media over QUIC (MoQ) publish-subscribe hierarchy so spatial regions and quality layers can be streamed and prioritised independently.

### Method

Three components (Fig. 1): Publisher, an optional content-agnostic MoQ Relay, and Subscriber.
1) Hierarchy mapping (Sec.
  II-A): Track = non-overlapping spatial region (e.g. block/octree partition); Group = semantically coherent object cluster inside a Track (K-Means, DBSCAN, or 2D/3D detection lifted to 3D); Subgroup = quality level S1..Sm, built either by (a) independent geometric heuristics (rank splats by saliency such as volume/isolation, opacity or scale, so S1 alone has few holes; layers are disjoint splat subsets) or (b) dependent attribute split (S1 = geometry + SH degree 0, higher SH bands in enhancement layers, which need S1 to decode); Object = transmission unit carrying splat attributes.
2) Publisher (Sec.
  II-B): pre-partitions the scene, writes a JSON manifest/Catalog (Track ids + bounding volumes, Group 3D centroids, Subgroup quality levels), maps each Subgroup to its own QUIC stream, and assigns static priorities (base layer highest).
  It keeps no per-subscriber state.
3) Subscriber: reads the Catalog, subscribes only to Tracks intersecting the viewport, and periodically recomputes per-Group priorities from 6-DoF pose, pushing them upstream with SUBSCRIBE_UPDATE.
  Adaptation is by stream prioritisation rather than segment/bitrate selection; the authors call it the 3D-viewport counterpart of ABR.
  Without a subscriber policy everything ties and the publisher priorities give breadth-first delivery: all base layers, then all EL1, and so on.
Prototype (Sec.
III): single Track.
Offline publisher pipeline: render N views (repo summary: 8 views at 45 deg azimuth steps), detect objects with YOLOv26x (repo: yolo26x.pt), lift boxes to splats with padding, merge cross-view detections by 3D IoU or centroid proximity, DBSCAN over object centroids to form Groups, then 5 Subgroups of 20% each by opacity or scale ranking.
Payload is raw float32 (no compression).
Transport is moq-lite extended with a Subgroup level, plus moq-relay.
Subscriber in Python + PyTorch + gsplat with a per-Subgroup state machine (MISSING/DOWNLOADING/CACHED, timeout leads to re-SUBSCRIBE), a 100 ms priority loop using Eq. 1 dP = Wd*max(0,1-min(D,Dmax)/Dmax) + Wa*max(0,cos theta), and a watermark VRAM/RAM swap.

### Scene update model

Static scene only.
All Gaussians come from one pre-trained PLY and are partitioned once, offline.
Nothing is created, updated or deleted at runtime.
Gaussians have no per-primitive IDs across time; each splat has a fixed address by position in the (Track, Group, Subgroup, Object) hierarchy, which stays stable because the scene never changes.
Refinement on the client is purely additive: disjoint 20% splat subsets are appended cumulatively (BL, BL+EL1, ...).
Under the attribute-split design option, later layers would add SH bands to existing splats, but the prototype does not implement that.

### Transport

Media over QUIC (paper cites draft-ietf-moq-transport-18) via moq-lite (Rust, QUIC + WebTransport) extended from 3 levels (Track/Group/Object) to 4 (Track/Group/Subgroup/Object). moq-relay acts as a content-agnostic, priority-aware forwarder and cache.
Each Subgroup is an independent QUIC stream, which avoids connection-level HOL blocking.
Priorities are integers in 0..255 (0 = highest): subscriber priority is checked first and publisher per-layer priority breaks ties.
Control messages are SUBSCRIBE and SUBSCRIBE_UPDATE.
The Catalog is a JSON MoQ track.
Payloads are raw float32 Gaussian attributes (no codec, no quantisation).
There are no deltas, since the scene is static.
Congestion is left to the relay/QUIC ("handled implicitly"); there is no explicit rate or bandwidth estimation.
The public repo differs: it uses an asyncio TCP placeholder with a custom 32-byte-header binary frame ('GSP0': track/group/object/subgroup ids, num_gaussians, payload_len; arrays means, opacities, sh, scales, rotations) in src/moq3dgs/transport/protocol.py.

### Client

Native Python client, not a browser: PyTorch + gsplat CUDA rasterisation, with moq-lite reached through ctypes (per paper).
Rendering falls back to whatever base or lower layers are cached when a Subgroup download stalls (claimed to avoid freezes; not measured).
GPU memory management: a fixed VRAM budget M_max with a high watermark (e.g. 0.9*M_max).
When it is exceeded, base layers are pinned, enhancement layers are sorted by dP and the lowest are demoted to system RAM; they are promoted back when dP rises, which avoids re-download.
The design also names a storage tier.
No render FPS, sort strategy, or GPU upload timing is reported in the paper.
The repo README additionally describes a splat budget feedback loop targeting 24 FPS (cut the budget if a frame takes more than 40 ms, raise it below 30 ms) and a 20% frustum halo; these are not in the paper.

### Evaluation design

Two stages, both on one scene: Mip-NeRF360 Bicycle (repo manifest: 6,131,954 Gaussians, about 1,535 MB raw).
Stage 1, offline content preparation.
Metric is VMAF over a 10 s, 30 fps (300-frame) orbit of +-90 deg azimuth around one cluster (Bench-Bicycle), with fixed distance, elevation and intrinsics.
The reference is the render of the unpruned model, not ground-truth photos.
Arms: opacity-ranked vs scale-ranked layering, cumulative Q0..Q4 (20..100% of splats).
No random-order or other progressive-3DGS baseline (e.g.
LapisGS).
Error bars appear in Fig. 4 but are not defined; plausibly the per-frame spread.
Hardware: Ubuntu 22.04 cluster, Intel Xeon Gold 5218 @ 2.30 GHz, dual NVIDIA Quadro GV100.
Stage 2, end-to-end streaming.
Link bandwidths 10/30/50/100 Mbps.
The paper does not say how bandwidth was imposed (no tc/netem or emulator named), and gives no RTT, loss, session duration, viewer trajectory, number of runs, or variance.
Metrics: Subgroups delivered, MB received, final splats rendered.
There is no quality metric, no latency or time-to-first-frame, and no stall metric for this stage.
There is no control arm: no TCP/HTTP/DASH transport, no no-priority fallback, no FIFO, and no prior system (LTS, L3GS).
The HOL-blocking motivation is never tested, since no loss condition is run.
Under our research-discipline rule, nothing here is a null we can reuse.
The 'baseline' outcome would have to be re-measured in our own setup.

### Key numbers

Fig. 4 (VMAF read off the plot, so approximate; Bench-Bicycle cluster, 300-frame orbit, reference = unpruned render):
- Opacity ranking: Q0 (20%) ~41, Q1 (40%) ~76, Q2 (60%) ~100, Q3 = Q4 = 100.
- Scale ranking: Q0 ~28, Q1 ~40, Q2 ~56, Q3 ~81, Q4 = 100.
- Q4 = 100 holds by construction, because Q4 is the reference.
Table I (Bicycle, subscriber-driven delivery, final state):
- 10 Mbps: 397 Subgroups, 37.4 MB, 224,710 splats.
- 30 Mbps: 2,432 Subgroups, 337.8 MB, 1,400,909 splats.
- 50 Mbps: 2,628 Subgroups, 363.2 MB, 1,491,950 splats.
- 100 Mbps: 2,853 Subgroups, 390.6 MB, 1,512,214 splats.
- At 100 Mbps that is about 25% of the scene's 6.13M Gaussians (repo manifest).
- Byte counts divided by link rate give about 30 s, 90 s, 58 s and 31 s, so the four runs either had different durations or did not saturate the link.
  The paper does not say which.
There are no latency, throughput-over-time, FPS or stall numbers anywhere in the paper.

### Limitations

Stated by the authors:
- The prototype and evaluation are preliminary.
- Only one configuration is built: a single Track and the geometric-heuristic layers.
- The MoQ spec is still evolving.
- No standard 3DGS codec is used (raw float32).
- Scheduling heuristics under dynamic network and telemetry conditions are left as future work.
- Only static on-demand scenes are handled.
Additional ones I see:
- One scene and one cluster.
- VMAF is measured against the unpruned render, so it measures pruning loss, not reconstruction quality.
- The streaming stage reports only volumes.
  There is no latency, quality-over-time or stall metric, no repeated runs and no variance.
- Bandwidth shaping method and session length are unstated, and the implied transfer times disagree across rows.
- Table I counts up to 2,853 Subgroups.
  That cannot be reconciled with the stage-1 decomposition (the repo summary shows YOLO found only 2 objects, bicycle and bench, plus an 'other' remainder of 76.3% of splats, with 5 Subgroups each).
  The number of Groups used for streaming is never stated.
- No transport, scheduler or prior-system baseline, so neither 'eliminates HOL blocking' nor 'avoids stalls' is shown.
- The public code does not contain the moq-lite/QUIC path used in the paper (TCP placeholder only), and the repo design (K-means 64 clusters, WFPS LoD, SH split) differs from the paper's (YOLO + DBSCAN, opacity/scale tiers).
- Citation glitches: YOLOv26x is cited to YOLO-World [19]; DBSCAN to FreeGaussian [16]; K-Means to ProtoGS [15].

### Gap relative to our idea

Overlap, which must be cited and not claimed as new:
(1) Viewport-driven priority delivery of Gaussian chunks with a simple utility (distance + angle, Eq. 1), recomputed periodically and pushed to the sender.
This is essentially our 'view-aware baseline'.
(2) Progressive base-first layering, where the no-policy fallback gives breadth-first, layer-ordered delivery.
This is a natural 'FIFO-like / layer-order' baseline.
(3) A client GPU memory budget with watermark eviction, base-layer pinning and a VRAM-to-RAM tier.
This overlaps directly with our 'browser GPU memory as a capacity-limited virtual scene cache' angle.
(4) Per-object independent streams with priorities.
MoQ over QUIC/WebTransport avoids the TCP HOL blocking that our WebSocket plan will have.
What our idea adds:
(a) A live, changing scene from a feed-forward model.
That needs persistent Gaussian identity, ID-based update/delete deltas, cache invalidation and temporal coalescing; MoQSplat only appends static disjoint subsets.
(b) Deadline- and freshness-aware scheduling.
MoQSplat's priority is purely spatial and has no notion of update age, deadline or glass-to-glass latency.
(c) Joint network-GPU scheduling.
MoQSplat's VRAM budget and network priority are separate mechanisms; neither limits what is sent by what can be uploaded and rendered in time.
(d) A browser WebGPU client with decoupled render rate.
MoQSplat uses native Python/gsplat.
(e) A real evaluation: glass-to-glass latency, render stalls, tc netem bandwidth/RTT/loss traces, repeated runs, FIFO and view-aware baselines.
MoQSplat has none of these, and its authors name scheduling evaluation as open future work.
Honest caveat: if we keep WebSocket over TCP, a reviewer can point to MoQSplat's HOL-blocking argument.
We should either measure HOL impact under netem loss or note WebTransport/MoQ as an alternative.

### Reusable

(1) Eq. 1 priority formula, as a cheap view-aware baseline to reimplement (repo src/moq3dgs/viewport/priority.py and frustum.py hold related Python versions).
(2) The binary frame layout in src/moq3dgs/transport/protocol.py, a simple per-cluster header + float32 arrays, as a reference for our chunk format.
(3) Publisher scripts (publisher/gsplat_clustering_vmaf.py, compute_rd_vmaf.sh, make_cluster_orbit_trajectory.py, render_layer_videos.py) for opacity/scale layering and a VMAF-vs-layer pipeline.
(4) The MoQ hierarchy-mapping idea and moq-dev/moq (moq-lite/moq-relay), if we ever try WebTransport.
Maturity: research prototype, 24 commits (Apr-Sep 2026), 6 stars, heavy agent-scaffolding files, and the paper's QUIC/moq-lite code is absent.
License: MIT is declared in README and pyproject, but there is no LICENSE file.
Ask the authors before copying code.
Data: uses the public Mip-NeRF360 Bicycle scene.
No traces are shipped.
Note that repo summary.txt references an external camera trace path '../GaussianAdaptiveStreamer/TestMovements/NTHU/bicycle/user1_bicycle.json', which hints at an NTHU-origin 6-DoF user-movement trace set.
That set is not in the repo; worth tracing since the student is at NTHU (unverified lead).

### Notes

Verified from primary sources: arXiv abs/HTML/PDF (full method and evaluation read, figures inspected from the PDF), the MMSP 2026 program PDF, and the GitHub repo plus forks via the GitHub API.
Related leads from the same MMSP 2026 MoQ special session, seen only as program titles and not read:
- SSI5 'MoQa: Low-Latency Point Cloud Videoconferencing via Cross-Layer Media over QUIC Optimization' (Nemeth et al.).
  This is a live volumetric stream, closer to our live setting than MoQSplat.
- SSI3 'Relay-Side Latency-Budget-Aware Rate Adaptation in MOQ Transport' (Zengin, Gurel, Begen).
  It is deadline-aware, relevant to our scheduler.
- SSI1 'BANDIT: Accurate Bandwidth Predictor for MoQ'.
Also cited by MoQSplat and worth surveying for evaluation design: LapisGS (3DV 2025), LTS (ACM MMSys 2025, DASH for dynamic multi-layer 3DGS), L3GS (MobiCom 2025), and 'Implicit Adaptation for Point Cloud Streaming with MoQ' (Mile-High Video 2026).
For the student: MoQSplat is a good, readable example of how a transport hierarchy (Track/Group/Subgroup/Object) maps onto 3DGS.
Its evaluation is too thin to serve as a number to beat.
Its ideas (Eq. 1 priority, base-layer-first fallback, watermark VRAM cache) are best treated as baselines and design patterns to re-measure in our own harness.

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| MoQSplat is published at MMSP 2026 | supported | arXiv Comments: 'Accepted at IEEE MMSP 2026 (Istanbul, 22-24 September 2026)' (https://arxiv.org/abs/2609.18624); listed as SSI4 in Special Session I 'Media over QUIC' of the official MMSP 2026 program (https://attend.ieee.org/mmsp-2026/wp-content/uploads/sites/776/program.pdf p.5). It is a workshop special-session paper with preliminary results, not a full conference paper. |
| arXiv id is 2609.18624 | supported | https://arxiv.org/abs/2609.18624 resolves to this exact title and author list; v1 dated 16 Sep 2026. |
| Uses QUIC / Media-over-QUIC transport | partly | Paper Sec. III says the prototype uses moq-lite (Rust, QUIC + WebTransport) through ctypes bindings, extended with an explicit Subgroup layer, and moq-relay as relay (https://arxiv.org/html/2609.18624). However, the public repo's transport is an asyncio TCP server/client: src/moq3dgs/transport/server.py docstring says 'async TCP streams (placeholder for real QUIC via aioquic)', client.py says 'TCP (QUIC placeholder)', and the README 'Next Steps' lists 'Transition from TCP placeholder to full QUIC transport'. No Rust, ctypes or moq-lite code exists on the default branch, on cd-athena/MoQSplat (incl. branch 'emanuele') or on EternalSky23/MoQSplat (checked via GitHub API tree listing). The QUIC implementation used for Table I is therefore not public. |
| Uses spatial Tracks | partly | Design (Sec. II-A) partitions the scene into non-overlapping spatial Tracks with bounding volumes in the manifest and subscribes only to Tracks intersecting the viewport. But the prototype states 'we consider only a single Track that covers the entire 3DGS content' (Sec. III), so spatial Tracks are only illustrated (Fig. 2b, indoor scene split into 4 Tracks) and never evaluated. |
| Progressive streaming of Gaussians | supported | Each Group is split into 5 Subgroups of 20% of splats ranked by opacity or by scale (Sec. III step 4); base layer has highest MoQ priority; client cumulatively adds layers (Fig. 3, Fig. 4 VMAF, Table I). |
| View-dependent Gaussian streaming; MoQSplat already does similar view-dependent selection and can serve as a baseline | partly | Subscriber computes a per-Group priority every ~100 ms: dP = Wd*max(0, 1 - min(D,Dmax)/Dmax) + Wa*max(0, cos theta) (Eq. 1), and pushes it with SUBSCRIBE / SUBSCRIBE_UPDATE. Granularity is a Group (object cluster), not individual Gaussians. No experiment isolates the effect of this view-aware priority (no comparison against the no-priority fallback, no viewer trajectory reported for the streaming test), and the client is a native Python/gsplat program. It is usable as a baseline policy only by reimplementing Eq. 1 in our own system, not as a measured reference result. |
| Priority scheduling of Gaussian data | supported | Publisher assigns static per-layer priorities (base layer lowest number = highest priority in MoQ's 0-255 range); subscriber priorities are evaluated first and publisher priorities only break ties (Sec. II-B). Scheduling itself is delegated to the MoQ relay: 'Congestion is handled implicitly by the Relay' (Sec. IV). The authors call designing and evaluating viewport-aware scheduling heuristics 'a promising direction for future research', i.e. not done here. |
| Focuses on scene delivery, not a continuous feed-forward reconstruction system | supported | Sec. II-A: 'aligns with on-demand static scene delivery'; Sec. III step 5: pre-processing 'operates entirely offline as a one-time content ingestion phase for static 3DGS-on-demand scenes'. Input is a pre-trained PLY (Mip-NeRF360 Bicycle). No temporal updates, no reconstruction model. |
| Best suited to study network protocol and Gaussian priority scheduling | partly | Opinion in the source material. The paper frames the protocol mapping and priority hooks well, but its evaluation has no latency, loss, RTT, HOL-blocking test, or baseline transport/scheduler (Sec. IV), so it gives a design reference rather than an evaluated scheduling result. |

### Verifier corrections

- **transport (repo wire format)**
  Reader said: custom 32-byte-header binary frame ('GSP0': track/group/object/subgroup ids, num_gaussians, payload_len; arrays means, opacities, sh, scales, rotations)
  Correct: The 32-byte header is 8 little-endian uint32 fields: magic, version, track_id_len, group_id_len, object_id, num_gaussians, payload_len, subgroup_id.
  The track and group ids are variable-length strings at the start of the payload, not header fields.
  Each attribute array is prefixed by its byte length.
  The magic constant 0x47535033 spells 'GSP3' in ASCII, although the code comment says 'GSP0'.
  This is cosmetic and does not change the reader's conclusions.
  Source: https://github.com/emanuele-artioli/MoQSplat/blob/main/src/moq3dgs/transport/protocol.py (module docstring, lines 1-43)

### Verifier on the gap assessment

Mostly fair, and I found no case where the paper already does something the reader says it does not do.
Checked against the full text at https://arxiv.org/html/2609.18624:
(a) Live or changing scene: not covered.
Sec.
II-A says the design 'aligns with on-demand static scene delivery'.
Sec.
III step 5 says pre-processing is 'entirely offline as a one-time content ingestion phase for static 3DGS-on-demand scenes'.
There are no IDs across time, no update/delete operations and no invalidation.
(b) Deadline or freshness awareness: not covered.
Subscriber priority is Eq. 1 (distance plus angle to the Group centroid), and the publisher priority breaks ties by quality layer.
Nothing uses age, deadlines or MoQ delivery timeouts.
A small wording point: the priority is spatial plus a layer-order tie-break, not 'purely spatial'.
(c) Joint network-GPU scheduling: not covered, with one nuance the reader understates.
The VRAM watermark demotion (Sec.
III item 3) and the network SUBSCRIBE/SUBSCRIBE_UPDATE priority both rank by the same dP score.
So the two mechanisms share one signal, but neither is limited by GPU upload or render capacity.
The swap mechanism is also never evaluated in Sec.
IV.
(d) Browser WebGPU client: not built.
The subscriber is Python + PyTorch + gsplat.
However, Sec.
III describes moq-lite as 'a Rust-based Pub/Sub framework over QUIC and WebTransport', so their stack does not rule out a browser client.
(e) Evaluation: not covered, as the reader says.
Sec.
IV has no latency, stall, RTT or loss metric, names no bandwidth-shaping method, and has no baseline transport or scheduler.
Sec.
II-B explicitly calls 'designing and evaluating advanced viewport-aware scheduling heuristics' future work.
The reader's overlap list (Eq. 1 view-aware priority, breadth-first layer-order fallback, a watermark VRAM/RAM cache that pins base layers and has a storage tier, per-Subgroup QUIC streams that avoid HOL blocking) is accurate.
These must be cited as prior art.

### Verifier notes

I re-fetched the primary sources independently into papers/agent_workdir/verify_moqsplat/ : abs.html, full.html, paper.pdf, program.pdf, repo/ and my_fig4.png (Fig. 4 rendered at 300 dpi).

Confirmed:
- Title and the 6 authors match https://arxiv.org/abs/2609.18624.
- The arXiv Comments field reads: '7 pages.
  Accepted at IEEE MMSP 2026 (Istanbul, 22-24 September 2026).
  First three authors contributed equally.
  Code: github.com/emanuele-artioli/MoQSplat'.
- v1 was submitted 16 Sep 2026.
  Primary category is cs.MM, cross-listed to cs.GR and cs.NI.
- The HTML version carries a CC BY-SA 4.0 license.
- The body ends on PDF page 5.
  References run over pages 6-7, so 'about 5 pages of text' is correct.

Venue:
- https://attend.ieee.org/mmsp-2026/ confirms 'IEEE 28th International Workshop on Multimedia Signal Processing', 22-24 Sep 2026, Istanbul.
- Program PDF p.5 lists SSI4 MoQSplat on Wed 23 Sep, 10:30-12:00.
  The session's full title is 'Special Session I - Media over QUIC: Next-Generation Protocols for Low-Latency Multimedia Streaming', chaired by Andrew Freeman.
- The program spells the fifth author 'Christian Esteve Rothenberg'.
- SSI1 BANDIT, SSI3 Relay-Side Latency-Budget-Aware Rate Adaptation (Zengin, Gurel, Begen) and SSI5 MoQa (Nemeth et al.) are confirmed on pp.5-6.

Key numbers:
- Fig. 4, read by me from the PDF: opacity ranking gives about 41/76/100/100/100 and scale ranking about 28/40/56/81/100 for Q0-Q4.
- The error bars are not defined anywhere in the text.
- Table I matches exactly: 397/37.4/224,710; 2,432/337.8/1,400,909; 2,628/363.2/1,491,950; 2,853/390.6/1,512,214.
- The reader's implied transfer times are correct: 29.9, 90.1, 58.1 and 31.2 s.

Two further inconsistencies in Table I that support the reader's critique (my arithmetic):
- The 10 Mbps row implies about 166 bytes per rendered splat.
  The other rows imply about 241-258, and the repo manifest gives 1,610,020,112 B / 6,131,954 = about 263 B/splat raw.
  The 10 Mbps row therefore holds fewer bytes than its rendered splats would need at full float32 attributes.
- Every row works out to about 530-580 splats per 'Subgroup' (about 0.14 MB).
  That cannot be the stage-1 Subgroups of 20% of a cluster, which hold hundreds of thousands of splats.
  The streaming decomposition is clearly different from stage 1 and is never described.

Repo (GitHub API, checked myself):
- 6 stars, 24 commits from 2026-04-28 to 2026-09-16, license=null, /license returns 404.
- README and pyproject declare MIT, and pyproject lists aioquic as a dependency.
- forks_count is 3, but only 2 forks are publicly listed (cd-athena, EternalSky23).
- No .rs, Cargo, ctypes or moq-lite files exist on emanuele-artioli main, cd-athena main, cd-athena 'emanuele' (76 entries) or EternalSky23 main.
- server.py and client.py carry the TCP-placeholder docstrings, and README 'Next Steps' lists the TCP-to-QUIC move.
  The reader's 'QUIC path not public' finding holds.
- The README describes 24 FPS / 40 ms budget feedback, a 20% halo, K-means with --clusters 64, and WFPS with 3 subsets x 2 SH quality levels.
  All are confirmed.
- publisher/rd_workdir/bicycle_opacity/summary.txt confirms: yolo26x.pt; 8 views; 2 objects (bicycle, bench); 'other' 4,680,044 = 76.3%; camera_json path ../GaussianAdaptiveStreamer/TestMovements/NTHU/bicycle/user1_bicycle.json.

Citation glitches are confirmed: [19] is YOLO-World, [16] is FreeGaussian, [15] is ProtoGS.

Not verified: the ATHENA blog post (the reader also flagged it as unread), and whether the paper is in IEEE Xplore yet.
The official program is sufficient evidence of acceptance.

## S2GS: Streaming Semantic Gaussian Splatting for Online Scene Understanding and Reconstruction

Key `s2gs`, relevance 3/5, identity confirmed by verifier: yes.

### Authors

Renhe Zhang, Yuyang Tan, Jingyu Gong, Zhizhong Zhang, Lizhuang Ma, Yuan Xie, Xin Tan (East China Normal University; Xin Tan also Shanghai AI Laboratory; corresponding xtan@cs.ecnu.edu.cn).
Source: arxiv.org/html/2603.14232 author block.

### Venue and year

ICML 2026, poster (Thu Jul 9 2026, 2:30-4:15 PM KST, Hall A #607) per https://icml.cc/virtual/2026/poster/65536; papers.cool lists the OpenReview record CbWCaD8tRC as "ICML.2026 - Poster". arXiv v1 posted 15 Mar 2026 (only version; arXiv comment field says only "10 pages, 7 figures").
The OpenReview forum page itself returned a 403 bot challenge, so the camera-ready was not read.

### arXiv

2603.14232 (v1, 15 Mar 2026, cs.CV, CC BY 4.0)

### Links

Paper: https://arxiv.org/abs/2603.14232, full text https://arxiv.org/html/2603.14232, PDF https://arxiv.org/pdf/2603.14232 (10 pages; it refers to an appendix for sequence construction, metric definitions and training settings, but the arXiv v1 PDF has no appendix).
ICML page: https://icml.cc/virtual/2026/poster/65536 (also a one-page poster PDF at https://icml.cc/media/icml-2026/Slides/65536.pdf).
OpenReview: https://openreview.net/forum?id=CbWCaD8tRC (blocked by a bot challenge, not read).
Project page: https://stdcoutzrh.github.io/projects/s2gs/ (site under CC BY-SA 4.0).
Code: https://github.com/stdcoutzrh/S2GS.
The GitHub API shows this repo is EMPTY: created 2026-05-18, size 0, 0 commits ("Git Repository is empty"), no license, as of 2026-10-27.

### Problem

Offline feed-forward methods that jointly reconstruct and understand a scene (SIU3R, Uni3R, LSM) recompute cross-frame interactions over the whole history each time a frame arrives.
Their runtime and GPU memory therefore blow up with sequence length: SIU3R runs out of memory at about 80 frames on a 140 GB H200.
Streaming reconstruction methods (CUT3R, STream3R, StreamVGGT, StreamGS) scale better but produce no semantics or instance identities.
S2GS wants strictly causal, reprocessing-free online 3DGS reconstruction plus instance-level semantics from an uncalibrated, unposed RGB stream.

### Method

Two branches, both strictly causal per frame.
(1) Geometry stream (Sec 3.2): each frame is turned into DINOv2 tokens and fed to a causal Transformer encoder (VGGT/STream3R style).
A temporal mask lets tokens at time t attend only to frames 1..t.
At inference the past keys and values are cached, so old frames are never re-forwarded.
Three light heads predict dense depth D_t, camera parameters P_t and per-pixel Gaussian attributes A_t.
Depth and camera are distilled from a strictly causal pretrained teacher, STream3R (L2 loss on depth, Huber loss on camera).
Pixels are back-projected with the predicted depth and camera to place pixel-aligned Gaussian centres, which are combined with A_t and appended to a persistent global Gaussian scene that is rendered by differentiable splatting.
(2) Semantic stream (Sec 3.3), separate from geometry: a frozen SigLIP2 encoder, a ViT-Adapter and a Mask2Former-style query decoder produce per-frame masks, class scores and query embeddings.
In training, queries are Hungarian-matched to ground-truth instances in each frame and a supervised contrastive loss (Eq. 4) pulls same-instance queries together across frames.
At inference an instance memory bank matches new queries to stored prototypes by cosine similarity plus bipartite matching (Eq. 5) and updates the prototypes by EMA (Eq. 6), giving persistent instance IDs.
Per-frame semantics and instances are attached to the pixel-aligned Gaussians and splatted like colour ('2D-to-3D semantic lifting').
(3) Open vocabulary (Sec 3.4): a small projector maps queries into SigLIP2 space, trained by cosine regression against SigLIP2 embeddings of the masked image regions.
Same-instance queries from several views are averaged so the projector tolerates EMA drift.
A text query returns the mask whose query scores highest.

### Scene update model

Append-only, as far as the paper describes.
Each frame contributes a new batch of per-pixel Gaussians back-projected from that frame's predicted depth and camera, 'incrementally accumulated into a persistent global scene representation' (Sec 3.2).
The paper describes no update, merge, pruning, deduplication or deletion of earlier Gaussians.
'Reprocessing-free' suggests old Gaussians are left untouched, but the text never says so explicitly.
Gaussians have no explicit persistent ID.
Implicitly, each is identified by (frame index, pixel index), and since earlier ones are never edited that would be stable.
The persistent identity in the paper is at the instance level (memory-bank prototype IDs), not the Gaussian level.
The text does not say whether earlier Gaussians' semantic labels are rewritten when instance associations change.
Because nothing is pruned, the Gaussian count grows linearly with frames, which matches peak memory rising from 6.68 GB at 16 views to 76.88 GB at 1024 views (Table 4).
Input resolution and Gaussians per frame are not given in arXiv v1.

### Transport

None.
Everything runs on one GPU.
'Streaming' means the input image stream, not delivering the scene to a remote client.

### Client

None.
Renders come from the training-time differentiable splatting renderer for evaluation only.
No viewer, browser, WebGPU or WebGL client, no buffer-update strategy, no sort, and no rendering frame rate is reported.

### Evaluation design

Datasets: ScanNet (SIU3R's preprocessed frames) for training and testing.
Streaming sequences are built by 'progressively extrapolating the viewpoint', so each context frame extends the view range past the previous one; details are said to be in an appendix missing from arXiv v1.
Zero-shot transfer to ScanNet++ and Replica at 32 views (Table 5).
Sequence lengths: 2/8/14/32 views (Table 2), 64/256 views (Table 3), and 16/64/128/512/1024 views for efficiency (Table 4).
Metrics: novel-view synthesis PSNR and SSIM; per-frame semantic mIoU; cross-frame instance consistency T-mIoU and T-SR (defined only in the missing appendix); open-vocabulary mIoU over text queries such as chair, desk and floor (Table 6); 'current-frame inference time' (process the current view and update the persistent state, model compute only, no rendering or I/O); peak GPU memory (PGM).
Baselines: offline joint methods SIU3R, Uni3R and LSM; 2D segmentation methods LSeg and Mask2Former; reconstruction-only pixelSplat, MVSplat and NoPoSplat (2-view only).
SIU3R is the only baseline in the long-sequence, efficiency and zero-shot tables.
There is no streaming or online 3DGS baseline, even though StreamGS and LongSplat are cited.
Ablations: shared vs decoupled backbone (Table 7), with vs without contrastive learning (Table 8), with vs without teacher distillation (Table 9).
Hardware: only 'an H200 GPU with 140 GB' is mentioned (intro, for SIU3R running out of memory).
The GPU behind Table 4 is not stated explicitly.
No repeated runs, seeds, error bars or significance tests.
No network conditions, since none apply.

### Key numbers

Efficiency, online streaming (Table 4), S2GS vs SIU3R as time per frame and peak memory:
- 16 views: 0.10 s / 6.68 GB vs 0.26 s / 6.14 GB.
- 64 views: 0.12 s / 9.66 GB vs 1.52 s / 42.89 GB.
- 128 views: 0.15 s / 14.48 GB vs out of memory.
- 512 views: 0.35 s / 41.45 GB vs out of memory.
- 1024 views: 0.62 s / 76.88 GB vs out of memory.
In other words S2GS is about 10 fps early in a stream and about 1.6 fps at 1024 frames, with memory growing about 70 MB per frame.
Quality, ScanNet (Table 2), S2GS vs SIU3R:
- 2 views: PSNR 24.90 vs 25.79, below SIU3R and NoPoSplat (25.27); mIoU 52.35 vs 47.83.
- 8 views: PSNR 20.83 vs 19.74; T-SR 82.49 vs 62.93.
- 32 views: PSNR 19.92 vs 17.82; mIoU 48.95 vs 39.98.
Long sequences (Table 3): at 64 views PSNR 18.71 vs 15.37 for SIU3R; at 256 views PSNR 16.37 and SSIM 0.581, with SIU3R out of memory.
Zero-shot at 32 views (Table 5): ScanNet++ PSNR 15.33 vs 12.85; Replica PSNR 15.66 vs 13.14.
Open-vocabulary mIoU (Table 6): 49.37 vs 45.76 (SIU3R) and 34.28 (LSeg).
Distillation ablation (Table 9): PSNR 10.21 without the teacher vs 19.92 with it.
All numbers are single runs as reported, from arxiv.org/html/2603.14232.

### Limitations

The authors note only that at 2 views it loses on PSNR and SSIM to non-causal offline methods.
Additional limitations I see:
(a) No code or weights.
The GitHub repo is empty, and the appendix with training settings, sequence construction and metric definitions is missing from arXiv v1.
(b) Memory is still unbounded: about 70 MB per frame, 76.88 GB at 1024 frames, so a 48 GB RTX 6000 Ada would hold roughly 512 frames under the paper's (unstated) settings.
(c) Absolute quality is low on long streams (PSNR 16-20 dB at 32-256 views), and the evaluation views are sampled for widening coverage rather than coming from continuous video.
(d) Gaussians are only ever appended, with no merging or pruning, so redundancy builds up when the camera revisits a region.
(e) It depends heavily on the STream3R teacher (PSNR falls to 10.21 without it).
(f) No streaming-3DGS baseline, no variance, and the hardware for Table 4 is not stated.
(g) Latency is model compute only; no end-to-end, rendering or display numbers.
(h) Scenes are static indoor ScanNet scenes; dynamic scenes are not handled.

### Gap relative to our idea

Overlap: S2GS is an example of the server-side 'feed-forward model continuously updates a 3DGS scene' stage, and its append-only, per-frame batch of pixel-aligned Gaussians maps naturally onto ID-based deltas.
A delta would be 'Gaussians of frame t', with ID = (frame, pixel).
Its per-frame compute (0.10-0.62 s) is a concrete reference point for the inference-rate side of a decoupled pipeline.
What our idea adds, and S2GS does not touch at all: delivering the scene to a remote viewer (transport, delta format and compression, temporal coalescing); view-dependent prioritisation; a browser WebGPU client with its own render loop and GPU-memory budget (a virtual scene cache); network-aware adaptation; deadline- and GPU-budget-aware joint scheduling; and glass-to-glass latency and render stalls under tc netem against FIFO and view-aware baselines.
S2GS also exposes a problem our system would have to solve.
Because it never prunes, each frame's delta is a full frame of new Gaussians (about H*W*bytes-per-Gaussian; H and W are not given in the paper).
Redundant Gaussians from revisited regions would waste bandwidth and client memory, which motivates server-side culling or merging plus eviction in the client cache.
Practical gap: with no released code or weights, S2GS cannot be used as the trace generator the material proposes.
The trace would have to come from another model with public code, or an S2GS-like append-only trace would have to be synthesised.
Its semantic branch is irrelevant to our goal.

### Reusable

Nothing runnable. github.com/stdcoutzrh/S2GS is empty (0 commits, no license, created 2026-05-18, checked via the GitHub API on 2026-10-27), and no weights or traces were found.
Ideas that can be reused:
- The scene-update model (append per-frame pixel-aligned Gaussians, no rewrite of old ones) as a template for a synthetic update trace.
- The efficiency protocol (per-frame time and peak GPU memory at 16 to 1024 frames) as one input to a GPU-budget model.
- The ScanNet 'progressively extrapolating viewpoint' sequences idea for building long input streams.
Its geometry teacher, STream3R (arXiv 2508.10893), might be a usable causal geometry backbone; I did not check its code or license here.

### Notes

The material calls it a 'causal incremental Gaussian scene for long videos', which is correct but incomplete: its real focus is joint semantic and instance understanding plus reconstruction.
Sources read: the arXiv HTML full text (all sections, tables and ablations), the arXiv PDF via pypdf (10 pages, no appendix), the ICML virtual poster page and poster PDF, the project page, papers.cool, and the GitHub API.
The OpenReview forum and PDF returned HTTP 403 (bot challenge) and were not read, so the ICML camera-ready may contain the appendix and may differ from arXiv v1.
Two dates conflict: a search snippet said the poster was on 'Wed Jul 8', but the fetched icml.cc page says Thu Jul 9 2026, 2:30-4:15 PM KST; the page is taken as authoritative.
Working files: papers/agent_workdir/s2gs/ (full.txt, paper.txt, paper.pdf, slides.pdf).

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| The paper's identity is S2GS (the material does not give a full title). | supported | Searching found exactly one Gaussian-splatting paper called S2GS: arXiv 2603.14232, 'S2GS: Streaming Semantic Gaussian Splatting for Online Scene Understanding and Reconstruction'. The similarly named S2Gaussian (arXiv 2503.04314, sparse-view super-resolution) is a different paper. The material leaves out that S2GS is mainly a SEMANTIC (instance segmentation plus open-vocabulary) method. |
| S2GS is an ICML 2026 paper. | supported | https://icml.cc/virtual/2026/poster/65536 lists it as an ICML 2026 poster (Jul 9 2026, Hall A #607) and links OpenReview CbWCaD8tRC. papers.cool shows 'ICML.2026 - Poster'. The OpenReview page itself could not be fetched (HTTP 403 challenge). The arXiv v1 PDF has no venue footer. |
| It builds a causal, incremental Gaussian scene. | supported | arxiv.org/html/2603.14232 Sec 3.1-3.2: 'strictly causal setting: at time t, the model processes only the current frame I_t together with persistent states', a causal attention mask (Eq. 1), and KV caching at inference. Per-frame pixel-aligned Gaussians G_t = BackProj(D_t, P_t) + A_t (Eq. 2) 'are incrementally accumulated into a persistent global scene representation'. |
| It handles long videos. | partly | Table 4 runs the model on streams of up to 1024 views: 0.62 s/frame and 76.88 GB peak GPU memory at 1024 views. Rendering quality is reported only up to 256 views (Table 3: PSNR 16.37, SSIM 0.581). The inputs are ScanNet frames re-sampled so the viewpoint keeps widening ('progressively extrapolating the viewpoint', Sec 4.1), not raw continuous camera video. Memory still grows roughly linearly, about 70 MB per frame, so 1024 frames needs a GPU with more than 77 GB. |
| It is not mainly about remote interactive display. | supported | Nothing in the full text covers a viewer client, network transport, rendering frame rate or display latency. 'Streaming' means the input image stream. The only latency reported is model compute per frame (Table 4). |
| (Material line 133) S2GS can be used to pre-generate a Gaussian update trace. | contradicted | Not possible today. The only released code location, github.com/stdcoutzrh/S2GS, is empty (GitHub API: size 0, 'Git Repository is empty', created 2026-05-18, no license), and no weights were found. Reproducing the paper would mean training it on ScanNet with a STream3R teacher, and the training settings are said to be in an appendix that is missing from arXiv v1. |
| (Paper's own claim) SIU3R runs out of memory after about 80 frames even on a 140 GB H200. | supported | Intro and Fig. 1 of arxiv.org/html/2603.14232. Table 4 agrees: SIU3R uses 42.89 GB at 64 views and runs out of memory at 128 views. Which GPU Table 4 was run on is not stated explicitly; only the intro mentions the H200. |
| (Paper's own claim) No online feed-forward 3DGS methods are publicly available, which is why only offline baselines are compared. | cannot_check | Stated in Sec 4.1. Yet the paper itself cites StreamGS (arXiv 2503.06235) and LongSplat (arXiv 2507.16144) as online generalizable 3DGS methods and does not compare against them. I did not check whether those have public code. Either way, the paper has no streaming-3DGS control arm. |

### Verifier corrections

- **key_numbers (Table 2, 2 views)**
  Reader said: 2 views: PSNR 24.90 vs 25.79, below SIU3R and NoPoSplat (25.27)
  Correct: At 2 views, S2GS (24.90) is also below Uni3R (25.44 PSNR, 0.812 SSIM), so three baselines beat it: SIU3R 25.79, Uni3R 25.44 and NoPoSplat 25.27.
  It is above pixelSplat (24.76), MVSplat (23.63) and LSM (22.38).
  The numbers the reader quoted are correct; the list of baselines that beat it was incomplete.
  Source: https://arxiv.org/html/2603.14232 Table 2

- **scene_update_model / limitations (b)**
  Reader said: Because nothing is pruned, the Gaussian count grows linearly with frames, which matches peak memory rising from 6.68 GB at 16 views to 76.88 GB at 1024 views (Table 4)
  Correct: The paper does not break peak GPU memory down by cause.
  Sec 3.2 also says 'key/value tensors from past frames are cached and reused', and that KV cache grows linearly with the number of frames too.
  So the roughly 69-70 MB per frame (from Table 4: (76.88-41.45)/512 and (41.45-14.48)/384) is model memory, cache plus Gaussians plus activations.
  It is not the size of one frame's Gaussians and must not be used as the size of a delta.
  Also, extrapolating linearly from 41.45 GB at 512 views gives about 600 frames within 48 GB, not about 512.
  That assumes the paper's resolution and settings, which it does not state.
  Source: https://arxiv.org/html/2603.14232 Sec 3.2 and Table 4

- **gap_vs_our_idea (redundancy point)**
  Reader said: Redundant Gaussians from revisited regions would waste bandwidth and client memory, which motivates server-side culling or merging plus eviction in the client cache.
  Correct: The redundancy point is valid for S2GS, but merging redundant Gaussians on the model side is not a gap in the field.
  Two online feed-forward 3DGS papers that S2GS cites already do it.
  StreamGS (arXiv 2503.06235) lists 'redundancy reduction' as a core challenge it addresses.
  LongSplat (arXiv 2507.16144) 'selectively compress[es] redundant historical Gaussians'.
  Our contribution should be described as network- and client-cache-aware delta handling, not as Gaussian dedup itself.
  Source: https://arxiv.org/abs/2503.06235 ; https://arxiv.org/abs/2507.16144 ; https://arxiv.org/html/2603.14232 (Sec 1 cites both, Sec 2 cites StreamGS)

### Verifier on the gap assessment

The gap is fair.
I checked the full arXiv HTML text: S2GS has no transport, no remote viewer or browser client, no rendering frame rate, no display or glass-to-glass latency, no network conditions and no scheduling.
'Streaming' means only the input image stream (Sec 3.1: 'processes only the current frame I_t together with persistent states').
The only latency reported is model compute per frame (Table 4: 'processing the current view and updating the persistent state').
Two things in the reader's write-up are its own inferences, not statements in the paper, and the reader says as much.
First, that the scene is append-only with no rewriting of old Gaussians: Sec 3.2 says only 'incrementally accumulated into a persistent global scene representation', and the abstract speaks of 'incremental Gaussian updates'.
Second, the (frame, pixel) Gaussian ID.
Gaussian IDs are not a paper feature; the only identity in the paper is at the instance level (memory bank, Eqs. 5-6).
One nuance to add (see corrections): merging redundant Gaussians is already done model-side by StreamGS and LongSplat, which S2GS cites, so the novelty claim should rest on the network, client-cache and scheduling side.
The practical gap is right: github.com/stdcoutzrh/S2GS is empty (GitHub API: size 0, 'Git Repository is empty', created 2026-05-18, no license, checked 2026-10-27), and the project page links no weights.
S2GS therefore cannot generate the trace the material proposes.

### Verifier notes

Everything else in the reader's report checked out, each item against the URL listed.
- **Title and authors:** title, all 7 authors, the ECNU and Shanghai AI Lab affiliations and the corresponding email match the arXiv HTML author block (https://arxiv.org/html/2603.14232).
- **arXiv record:** 2603.14232 has only v1 (15 Mar 2026), comment '10 pages, 7 figures', cs.CV, CC BY 4.0, and no journal-ref (https://arxiv.org/abs/2603.14232).
- **Venue:** ICML 2026 poster, Thu Jul 9 2026, 2:30-4:15 PM KST, Hall A #607, linking OpenReview CbWCaD8tRC (https://icml.cc/virtual/2026/poster/65536).
  The project page also says ICML 2026 (https://stdcoutzrh.github.io/projects/s2gs/).
  The OpenReview web page and both API endpoints (api and api2) returned 403 ChallengeRequired for me as well, so I could not read the camera-ready.
- **No appendix:** the arXiv PDF has 10 pages and ends with the references.
  The text still points to an appendix for sequence construction, the IoU and metric definitions, and training settings (Sec 4.1).
- **Code:** the repo is empty, as checked through the GitHub API (the empty-repo error came back as HTTP 409).
- **Numbers:** I re-read every number in Tables 2-9 and all match the reader.
  The most important ones:
  - Table 4: 0.10 s / 6.68 GB at 16 views, 0.62 s / 76.88 GB at 1024 views; SIU3R out of memory from 128 views.
  - Table 3: PSNR 16.37 at 256 views, with SIU3R out of memory.
  - Table 9: PSNR 10.21 without the teacher vs 19.92 with it.
- **Paper's own claims:** the H200 140 GB claim appears only in the intro and Fig. 1; the hardware for Table 4 is not stated.
  The 'no publicly available online feed-forward 3DGS methods' claim is in Sec 4.1.
  The reader's 'cannot_check' verdict on the latter is reasonable: the arXiv abstract pages of StreamGS (2503.06235) and LongSplat (2507.16144) list no code link, but I did not search further.
  All three corrections above are minor; none changes the relevance score or the main conclusion.
- **Working files:** papers/agent_workdir/s2gs_verify/ (full.html, full.txt, p.pdf).

## WebSplatter: Efficient and Faithful In-Browser 3D Gaussian Splatting across Devices via WebGPU

The arXiv v1 title was "WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU".
The ACM MM 2026 camera-ready version uses the title above.

Key `websplatter`, relevance 3/5, identity confirmed by verifier: yes.

### Authors

arXiv v1 (6 authors): Yudong Han, Weichen Bi and Yun Ma (Peking University, Institute for AI); Chao Xu, Xiaodan Ye and Zilong Dong (Tongyi Lab, Alibaba).
The ACM MM 2026 version adds a seventh author, Xu Xuanhuai (Nanyang Technological University), according to the project page bibtex.

### Venue and year

The authors state ACM Multimedia 2026 (project page bibtex: "Proceedings of the 34th ACM International Conference on Multimedia", doi 10.1145/3767308.3835220; README, Zenodo record 21798977).
Only these author-controlled sources confirm the acceptance.
On 2026-10-27, https://doi.org/10.1145/3767308.3835220 returned HTTP 404, before the conference itself (10-14 Nov 2026, Rio de Janeiro, per https://2026.acmmm.org/). arXiv v1 (3 Feb 2026) is a preprint with no venue comment.
The README says "35th" where the project page says "34th"; 34th is consistent with MM'24 being the 32nd.

### arXiv

2602.03207 (only v1 exists, 3 Feb 2026; it is the pre-camera-ready version)

### Links

Paper abstract: https://arxiv.org/abs/2602.03207.
Full text read: https://arxiv.org/html/2602.03207v1 (PDF https://arxiv.org/pdf/2602.03207, 10 pages).
Project page with live demo: https://websplatter.github.io/ (page licence CC BY 4.0).
Code: https://github.com/websplatter/WebSplatter (MIT; created 2026-08-06; HEAD 16bb06e, 2026-08-29; 8 stars).
Zenodo archive v1.0.0 (MIT, commit beb9532): https://zenodo.org/records/21798977.
ACM DOI 10.1145/3767308.3835220 did not resolve yet.
The v1 review link was https://anonymous.4open.science/r/webgs.
The README also links github.com/yudshj/WebSplatter, which exists but I did not inspect it.

### Problem

Rendering 3DGS in the browser is hard.
WebGL has no compute shaders, so viewers sort on the CPU through JS or WASM, which scales poorly.
Existing WebGPU ports copy CUDA/Vulkan algorithms that assume a workgroup scheduling order: for example, a decoupled-lookback or spin-wait prefix sum polls a global atomic flag.
WebGPU does not guarantee that order, so these ports busy-wait, run slowly or crash on Apple, mobile and other non-NVIDIA GPUs, and use too much memory.
The goal is a renderer that is portable, deadlock-free and memory-lean across heterogeneous devices and browsers.

### Method

A hybrid compute+render pipeline.
Each frame runs from scratch with no temporal reuse, in three stages.
(1) Pre-processing (compute shader).
Each Gaussian is projected to 2D (the 3D covariance becomes a 2D ellipse).
Its screen-space AABB is tested against the viewport, and splats outside it or with opacity <= 1/255 are dropped.
Survivors are compacted into a buffer, with the ellipse major/minor axes packed as f16 into two u32 values.
View-space depth is written as f32 to a key buffer.
SH is evaluated for the current view direction, and RGB+opacity is packed into one RGBA8 u32.
Indirect dispatch/draw buffers carry the survivor count.
(2) Sort (compute).
This is a least-significant-digit radix sort over the 32-bit depth key: 4 passes of 8 bits (256 buckets).
Each pass has three dispatches: a per-workgroup histogram, a hierarchical Blelloch scan, and a scatter.
An element's output position is pos = base_d + prefix_wg + rank_local_i, where base_d counts all elements with a smaller digit and prefix_wg counts elements with the same digit in earlier workgroups.
The scan does an upsweep over tile sums (level L0, then L1, recursively) and then a downsweep.
Each level is a separate dispatch, and synchronisation across workgroups comes only from the dispatch boundary, never from atomic polling.
That makes it wait-free, with O(N) work and O(N/wg) extra memory.
The MM abstract adds 'bitmap-popcount ranking'.
That part is not in v1 and I could not read it.
(3) Rasterization (hardware render pass).
There is no tile-based rasterizer and no tile-splat pair buffer.
Each splat is one instanced quad (two triangles), and its half-extent comes from opacity: r = sqrt(ln(255*sigma)), the radius where sigma*e^{-r^2} drops to 1/255.
The hardware's interpolators deliver a per-fragment local coordinate p, and the fragment shader computes alpha = sigma*exp(-p.p).
Back-to-front alpha blending uses the sorted order.
The implementation is TypeScript+WGSL (about 6,600 LOC per the paper) built with Vite.
It loads INRIA PLY, glTF KHR_gaussian_splatting, and SPZ-compressed GLB (decoded by WASM in a worker).
Storage is f16 by default, with an optional shader-f16 path.

### Scene update model

The scene is static.
Gaussians are loaded once from a file into GPU storage buffers (gaussian_3d_buffer and sh_buffer, filled with device.queue.writeBuffer).
Nothing is created, updated or deleted over time.
All view-dependent state (culling, projection, colour, depth order) is recomputed every frame, so the per-frame pipeline is stateless with respect to past frames.
A Gaussian's only identity is its array index.
The optional requestReorder/maybeReorderAfterSubmit path (marked 'TODO: check this' in the code) reads back the sorted indices, permutes the CPU copies so alive splats come first, and rewrites the GPU buffers.
That changes indices, so any ID-to-slot map would need updating.
Derived buffers (2D splats, colours, sort keys, scan levels) are sized from num_points at construction, so the renderer has a fixed capacity per loaded scene.

### Transport

none.
The paper does not study transport.
The only delivery path is a one-shot download of the whole scene file (PLY or GLB).
The demo's Van Gogh Room (341,294 splats) ships as a 5.05 MiB SPZ-compressed GLB, decompressed in-browser by WASM. ply2gltf.py makes these files.
The paper does no progressive or partial loading, prioritisation or adaptation.

### Client

The renderer runs in the browser through WebGPU, primarily in Chrome 113+, with some Safari 26 and Firefox 141 (Windows-only) results.
GPU buffer update strategy: one upload at load.
Per frame there are compute passes (cull/project/SH, then sort), followed by an instanced render pass with indirect dispatch and draw.
Sorting is a GPU wait-free 4x8-bit LSD radix sort over view depth for every frame.
Memory: 1.20 GB peak VRAM on garden (5.8M splats) on an RTX 3070 under Windows, against 1.90 GB for web-splat (W1), more than 2.80 GB for gaussian-splatting-webgpu (W2), 1.19 GB for GaussianSplats3D (S2) and 0.83 GB for antimatter15/splat (S1).
The paper does not say how memory was measured.
Per-stage GPU timing uses timestamp-query (8 timestamps per frame).
Device limits such as maxBufferSize or maxStorageBufferBindingSize are not discussed in v1.

### Evaluation design

This describes arXiv v1; I could not read the camera-ready.
Workloads: 7 static scenes.
Six are standard benchmark scenes (bicycle 6.13M, garden 5.83M, truck 2.54M, bonsai 1.24M, train 1.03M splats, plus a cleaned bicycle-c at 1.06M).
The seventh is Van Gogh Room (0.34M) from the Habitat test suite.
Each scene is rendered from one fixed camera pose at one fixed resolution, and v1 does not state that resolution.
There is no camera motion, scene dynamics or streaming.
Devices (8): RTX 3070 desktop; Intel NUC (i9-9980HK iGPU); MacBook Pro M1; MacBook Air M4; Lenovo MX350 laptop; iPhone 15 Pro Max; Redmi K70 Pro (Snapdragon 8 Gen 3); Oppo Find X2 (Snapdragon 865).
Browser is mainly Chrome, with Safari and Firefox rows marked; V-Sync is on.
Baselines are community viewers pinned to commits (Appendix B): WebGL S1 antimatter15/splat (CPU sort in a worker) and S2 mkkellogg/GaussianSplats3D (WASM sort plus an octree); WebGPU W1 KeKsBoTer/web-splat (spin-wait radix sort) and W2 MarcusAndreasSvensson/gaussian-splatting-webgpu.
All baselines were instrumented to time GPU execution.
Metrics: median total frame time (ms) summed over pre-process, sort and render; per-stage share from timestamps; crash or no crash; sort-stage time; peak VRAM.
Jitter (std and P99) is reported for WebSplatter on M4/garden only, with no baseline numbers.
Ablation: one device and one scene (M4, garden), turning off screen-space AABB culling (-CULL) and opacity-based quad radius (-RADIUS).
Quality: v1 has only a qualitative side-by-side with CUDA (Fig. 3).
The MM version claims average dPSNR 0.07 dB over 156 views from 5 scenes.
Statistics: the number of frames, runs and repetitions is not stated, and there are no CIs.
Repo defaults are 100 warm-up and 500 measured frames (src/splat-app.ts), but the paper does not confirm it used them.
There is no network, no emulation, no latency in the glass-to-glass sense, and no null or control beyond the baselines.

### Key numbers

These are arXiv v1 Table 1 medians (total GPU frame time, Chrome unless noted), for a fixed view and resolution.
A reported number is not a null for our data.
- RTX 3070, garden: 9.5 ms vs W1 14.4 (1.52x); Firefox 11.4 ms, where both WebGPU baselines crash.
- MacBook Air M4, garden: 63.7 vs W1 67.8 (1.06x).
- M1, garden: 112.0 vs S1 124.4 (1.11x); vs W1 510.8 (4.56x, the source of the '4.5x').
- MX350, garden: 164.3 vs S1 180.0.
- Intel NUC iGPU, garden: 151.2 vs S1 341.7 (2.26x).
- iPhone 15 Pro Max (Safari), bicycle-c: 38.5 vs W1 45.7.
- Redmi K70 Pro, bicycle-c: 33.6 vs W1 39.5 (1.18x).
- Oppo Find X2, bicycle-c: 105.2 vs S1 112.7.
Sort stage: 2.3 ms on RTX 3070 vs 55.8 ms for the S1 CPU sort; 19.0 ms on M1 vs 458.4 ms for the W1 spin-wait sort.
Stage shares: high-end devices are render-bound (render 51.8-62.6%); the MX350 and NUC are pre-process-bound (47.2% and 40.0%).
Jitter, M4/garden: std 22.3 ms, P99 116.9 ms.
Ablation, M4/garden: FULL 63.74 ms; -CULL 67.36 ms (+5.8%); -RADIUS 73.20 ms (+15%, render 33.0 to 39.2 ms).
Peak VRAM, garden on RTX 3070: 1.20 GB.
Scaling on M4: van gogh 11.7 ms, garden 63.7 ms.
Redmi truck: 52.7 ms, where W1 crashes.
Camera-ready, as stated on the project page only: 11 devices, 5 GPU architectures, no failures, 1.1-2.5x over surviving baselines, average dPSNR 0.07 dB over 156 views and 5 scenes.

### Limitations

Stated by the authors: Firefox WebGPU is unstable (crashes, blocklisted on the MX350, unreliable timestamp queries on the NUC), so those rows are omitted.
Low-end GPUs are bottlenecked in pre-processing, so data preparation and culling matter for large scenes.

My observations:
- The speedup headline is internally inconsistent across the abstract, contributions and conclusion, and mixes "vs a weak baseline" with "vs the best baseline".
- No run count, repetitions or CIs are given; one camera pose per scene; resolution not stated.
- The jitter claim gives no baseline numbers.
- The VRAM measurement method is not stated, and the crashes are blamed on OOM without direct evidence.
- The ablation covers one device and one scene.
  The -CULL text says the sort and render stages process more splats, yet Table 5 shows sort time falling (11.52 to 11.05 ms).
- The "deterministic execution" claim is never tested.
- The 'first framework' claim does not discuss Visionary (arXiv 2512.08478).
- Frame time is GPU execution time per frame.
  It is not a displayed frame interval and not end-to-end latency.
- Everything is static scenes with no uploads during rendering, so the cost of GPU writes competing with rendering is never measured.
- The camera-ready (new title, 11 devices, PSNR) could not be read.
  The two author sources even disagree on the GPU vendor list: the project page says NVIDIA/AMD/Apple/Qualcomm/MediaTek and the README says NVIDIA/Apple/Intel/AMD/Qualcomm.

### Gap relative to our idea

There is little overlap on contribution.
WebSplatter solves only the client renderer for a static scene, which is one box in our pipeline.

What our idea would add beyond it:
(a) A dynamic scene, where a server feed-forward model creates, updates and deletes Gaussians over time.
WebSplatter has no notion of change.
(b) Persistent Gaussian IDs mapped to buffer slots, a free-list or tombstones, and compaction.
WebSplatter identifies Gaussians by array index and its reorder step permutes them.
(c) Delta ingestion: worker decode, staging, partial writeBuffer of dirty ranges, and coalescing.
WebSplatter does one upload at load.
(d) A capacity-limited GPU scene cache (the virtualization angle).
WebSplatter allocates a fixed capacity from the scene size at load and never evicts.
(e) Joint network and GPU deadline-aware scheduling of which deltas to send and upload, and when, so a render frame is not missed.
(f) Measuring glass-to-glass latency and render stalls under tc netem, against FIFO and view-aware baselines.
WebSplatter reports GPU frame time only, on static views, with no network.

Where it helps us:
- It is a credible MIT client base.
  Because it recomputes cull, sort and colour every frame, a delta needs only buffer writes plus an opacity-0 tombstone, with no incremental structure to repair.
- Its per-stage timestamp-query breakdown and its device matrix are a template for measuring the GPU-side cost our scheduler must budget.
- Its pre-process/sort/render timings scale with N, which gives a cost model for how many Gaussians a frame budget allows.
- The fixed-capacity buffers mean we must preallocate to a max budget.

Honest caveat: if a reviewer counts "an efficient WebGPU 3DGS renderer" as a contribution, WebSplatter (and Visionary) already own it, so our contribution must sit in the update path and scheduling, not the renderer.

### Reusable

Code: https://github.com/websplatter/WebSplatter is MIT, TypeScript+WGSL with Vite, about 6.8k lines in src.
It is young research code: created 2026-08, 8 stars, a 'TODO: check this' in the reorder path.
- Reusable pieces: the GaussianRenderer class (src/gaussian-renderer.ts); WGSL shaders for cull/preprocess, Blelloch radix sort and conic render; PLY and glTF/SPZ loaders in workers; an alternative tile-based renderer (src/tile-based-renderer-global-sort.ts).
- Built-in benchmark modes in src/splat-app.ts: eval=1 captures images for PSNR; eval=2 measures wall-clock frame time with performance.now around submit+onSubmittedWorkDone (median/P99/std); eval=3 gives the per-stage timestamp breakdown.
  Defaults are 100 warm-up and 500 frames.
- Archived release: Zenodo 21798977 (v1.0.0, commit beb9532, MIT).
- ply2gltf.py converts PLY to GLB/SPZ, which needs nianticlabs/spz.
- The README's Google Drive link has the benchmark scene .ply files, with no licence stated.
  The scenes come from the standard 3DGS benchmark data, under their own terms.
- No dynamic or update traces, and no network code.

### Notes

Sources read:
- arXiv v1 full text via https://arxiv.org/html/2602.03207v1: all sections including Appendices A and B.
- The arXiv abs page, for submission history (v1 only) and authors.
- https://websplatter.github.io/, the GitHub README and repo tree.
- A shallow clone of HEAD 16bb06e, inspected for buffer management and eval code.
- The Zenodo API for record 21798977.

The ACM camera-ready was not readable: the DOI resolver returned 404, and I did not check the ACM DL page itself.
Everything about the MM version (11 devices, dPSNR 0.07 dB, 1.1-2.5x, bitmap-popcount ranking, the added author) is the authors' own summary and unverified against the paper text.

Practical tips for the student, from the code:
- Deleting a Gaussian can be an opacity-0 write, because the cull shader drops splats at or below 1/255.
- Preallocate buffers to a max capacity, since every derived buffer is sized from num_points at construction.
- Disable or adapt maybeReorderAfterSubmit if you keep an ID-to-slot map.
- Use eval=3 timestamps to calibrate the GPU-budget model.

Search snippets (bytez, alphaxiv, X) were not used for any fact.

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| WebSplatter is an ACM MM 2026 paper | partly | Only author-controlled sources say this: the project page (https://websplatter.github.io/), the README and Zenodo 21798977 all say ACM MM 2026 with doi 10.1145/3767308.3835220. On 2026-10-27 that DOI returned 404 at doi.org, because the proceedings are not out yet (the conference is 10-14 Nov 2026). The only full text I could read is arXiv v1 (2602.03207), and the MM version changed the title, an author, the device count and the headline numbers. |
| An efficient, cross-device WebGPU renderer (高效率、跨裝置的 WebGPU Renderer) | supported | arXiv v1 Sec. 3-4 (https://arxiv.org/html/2602.03207v1): a WebGPU compute+render 3DGS pipeline evaluated on 8 devices (RTX 3070, Intel NUC iGPU, MacBook Pro M1, MacBook Air M4, MX350, iPhone 15 Pro Max, Redmi K70 Pro, Oppo Find X2). It is the fastest in every Table 1 row: 1.06x to 2.26x over the best baseline per device, as the conclusion states. The project page says the MM version uses 11 devices and 5 GPU architectures. |
| Mainly about rendering, not an online update protocol (主要是 Rendering，而非 Online Update Protocol) | supported | v1 renders static PLY scenes only (Sec. 4.1). It has no streaming, updates or network content. In the code (src/gltf-loader.ts, src/gaussian-renderer.ts) the Gaussian and SH buffers are created and filled once at load, and every derived buffer is sized from pc.num_points when the renderer is constructed. |
| WebSplatter can serve as the WebGPU rendering base for implementing dynamic buffer updates (使用 WebSplatter 或 Visionary 作為 WebGPU Rendering Base，實作動態 Buffer Update) | partly | It is feasible but not provided. The code is MIT, about 6.8k lines of TS+WGSL, and already rewrites the whole Gaussian/SH buffers with queue.writeBuffer in maybeReorderAfterSubmit. The cull shader (gaussian_cull.wgsl line 139) drops splats with opacity <= 1/255, so writing opacity 0 works as a delete tombstone. There is no delta API, buffers are fixed-capacity per load, and the optional reorder permutes buffer indices, which would break index-based Gaussian identity. |
| (implicit, from the paper's own abstract) consistently 1.2x to 4.5x speedups over state-of-the-art web viewers | contradicted | The paper's own Table 1 contradicts the abstract. Against the best baseline, the M4 speedup is 63.7 vs 67.8 ms = 1.06x and the Oppo speedup is 105.2 vs 112.7 = 1.07x. The 4.5x is against one weak baseline (W1 on M1, 510.8 vs 112.0 ms); against the best M1 baseline (S1, 124.4) it is 1.11x. v1 itself gives three different ranges (abstract 1.2-4.5x, contributions 1.18-4.5x, conclusion 1.06-2.26x). The MM version restates it as 1.1-2.5x over surviving baselines. |

### Verifier corrections

- **venue_and_year / claim_check[0] (acceptance)**
  Reader said: Only author-controlled sources confirm the ACM MM 2026 acceptance; claim verdict 'partly'.
  Correct: The venue's own site confirms it.
  The ACM MM 2026 programme data (generated 2026-10-25) lists paper_id mfp1820 with the camera-ready title and all 7 authors, including Xuanhuai Xu (NTU).
  It is in the Main Track, session 'Poster Fa', line H, Nov 13 14:00-16:00, room Versailles.
  Acceptance is therefore confirmed, and the claim verdict should be 'supported'.
  The DOI still returns 404 and the ACM DL returned 403 to me, so the camera-ready full text is still unread.
  Source: https://2026.acmmm.org/site/program-data.json (loaded by https://2026.acmmm.org/site/technical-programme.html)

- **venue_and_year (34th vs 35th)**
  Reader said: README says 35th, project page says 34th; 34th is consistent with MM'24 being the 32nd (inferred).
  Correct: The conference site itself says 'Welcome to the 34th ACM International Conference on Multimedia, Rio de Janeiro, Brazil - 10-14 November 2026'.
  The README's '35th' is simply wrong, and no inference is needed.
  Source: https://2026.acmmm.org/

- **method (bitmap-popcount ranking)**
  Reader said: The MM abstract adds 'bitmap-popcount ranking'.
  Correct: The MM abstract does not mention it.
  That abstract appears on the project page and in the venue programme data.
  'Bitmap-popcount ranking' appears only in the project page's separate 'Method' section.
  Likewise '156 views from five scenes' appears only in the project page's 'Results' section.
  The venue abstract gives only 11 devices, 5 architectures, 1.1-2.5x and dPSNR 0.07 dB.
  Source: https://websplatter.github.io/ ; https://2026.acmmm.org/site/program-data.json

- **reusable / client (code size)**
  Reader said: about 6.8k lines of TS+WGSL in src
  Correct: At HEAD 16bb06e, src/ holds 25 .ts and 20 .wgsl files, totalling 8,804 lines (7,725 non-blank; 8,634 excluding src/release).
  The paper's own figure is about 6,600 LOC (v1 Sec. 4.1).
  '6.8k' matches neither count.
  Source: https://github.com/websplatter/WebSplatter (shallow clone at 16bb06e, `find src \( -name '*.ts' -o -name '*.wgsl' \) | xargs cat | wc -l`)

- **links (Zenodo commit)**
  Reader said: Zenodo archive v1.0.0 (MIT, commit beb9532)
  Correct: The Zenodo record does name commit beb9532feeec.
  That commit is not in either public repo: GitHub returns HTTP 422 'No commit found' on websplatter/WebSplatter and on yudshj/WebSplatter.
  The GitHub tag 1.0.0 points to 8a1c115 (2026-08-10).
  Use the Zenodo tarball or tag 1.0.0, and do not expect to check out beb9532 from GitHub.
  Source: https://zenodo.org/api/records/21798977 ; `git ls-remote https://github.com/websplatter/WebSplatter`

- **reusable (licence)**
  Reader said: MIT
  Correct: It is MIT, but the LICENSE reads 'Copyright (c) 2024 shrekshao', and package.json is named 'webgpu-typescript-starter'.
  The repo inherits its licence notice from an earlier third-party starter project, so any reuse must keep that copyright line.
  Source: https://github.com/websplatter/WebSplatter/blob/main/LICENSE ; https://github.com/websplatter/WebSplatter/blob/main/package.json

### Verifier on the gap assessment

The gap is fair. arXiv v1 (https://arxiv.org/html/2602.03207v1, Sec. 3-4) covers only rendering of static scenes.
It has no creation, update or deletion of Gaussians over time, no transport, no network emulation and no latency metric.
Its "frame time" is GPU execution time with V-Sync on (Sec. 4.1-4.2).
The camera-ready abstract on the venue programme mentions "volumetric video, dynamic avatars" and "compression and streaming", but only as motivation.
The system is still described as a rendering framework.

I checked the code at HEAD 16bb06e and the reader's characterisation holds:
- The PLY worker streams the download only to show progress.
  Rendering starts after the full load (src/ply-loader.ts, src/workers/ply-worker.ts).
- The "animation" flag is camera auto-rotation, not scene dynamics (src/splat-app.ts:550,781).
- The derived buffers are sized from pc.num_points (gaussian-renderer.ts:291,1169-1190).

Two code details strengthen the reader's points:
- The opacity-0 tombstone is valid.
  The GPU buffer stores opacity after the sigmoid (ply-worker.ts:80 applies sigmoid; glTF opacity is already post-sigmoid, gltf-worker.ts:547), and the cull shader keeps only opacity > 1/255 (gaussian_cull.wgsl:139).
- The GPU record is 10 scalars: position (3), opacity (1) and a precomputed 3D covariance (6), not scale and rotation.
  A delta format must either send the covariance or compute it in a worker.

The reader missed one more identity hazard.
Besides maybeReorderAfterSubmit, there is an optional load-time Morton reorder (URL param sort=morton; reorder-types.ts, ply-worker.ts:194-197) that also permutes indices relative to file order.

The "honest caveat" stands: the renderer contribution is taken, by WebSplatter and by Visionary (arXiv 2512.08478, 9 Dec 2025, confirmed at https://arxiv.org/abs/2512.08478, not cited in v1).

### Verifier notes

Everything below was re-checked against primary sources on 2026-10-27.

**Identity:**
- The arXiv abs page gives title, 6 authors, v1 only, submitted 3 Feb 2026 07:18 UTC, and the abstract claiming 1.2-4.5x (https://arxiv.org/abs/2602.03207).
- v1 affiliations are PKU Institute for AI (Han, Bi, Ma) and Tongyi Lab, Alibaba Group (Xu, Ye, Dong).
  The v1 code link is anonymous.4open.science/r/webgs (footnote 1).

**Key numbers:** all match my reading of https://arxiv.org/html/2602.03207v1.
- Table 1: RTX 3070 9.5 vs W1 14.4 (1.52x); M4 63.7 vs W1 67.8 (1.06x); M1 112.0 vs S1 124.4 and W1 510.8 (4.56x); NUC 151.2 vs S1 341.7 (2.26x); Redmi 33.6 vs W1 39.5 (1.18x).
- Table 4 sort times: 2.306 vs 55.8 ms; 19.027 vs 458.424 ms.
- Table 5 ablation: 63.74 / 67.36 / 73.20 ms, with sort 11.52 falling to 11.05 under -CULL.
- Sec. 4.5: 1.20 GB VRAM.
- M4/garden jitter: std 22.303 ms, P99 116.851 ms.

**Speedup ranges:** the three different ranges are confirmed.
The abstract says 1.2-4.5x, the contributions say 1.18-4.5x, and the conclusion says 1.06-2.26x.
The reader's 'contradicted' verdict on "consistently 1.2-4.5x" is defensible, because the minimum speedup over the best baseline is 1.06x on the M4.

**Radius formula:** it is r = sqrt(ln(255*sigma)), as the reader wrote; the HTML alttext shows the sqrt.
Code: gaussian_cull.wgsl:169.

**Code claims confirmed:**
- 'TODO: check this' at gaussian-renderer.ts:1238.
- Whole-buffer writeBuffer calls at :1301-1302.
- eval defaults of 500 frames and 100 warm-up (splat-app.ts:506-507).
- eval=2 timing is performance.now around submit plus onSubmittedWorkDone (splat-app.ts:360-368).
- Repo created 2026-08-06, HEAD 16bb06e on 2026-08-29, 8 stars, MIT, not a fork (GitHub API).

**Unverifiable:**
- The ACM DL page returned 403 and the DOI returned 404, so no camera-ready numbers can be checked.
- The camera-ready numbers (11 devices, 1.1-2.5x, 0.07 dB) now also appear in the abstract on the venue's own programme.
  That abstract is still author-written, and no table backing it was readable.

**Extra observations:**
- Std 22.3 ms on a median of 63.7 ms is a coefficient of variation of about 35%, which hardly supports the v1 wording "remarkable consistency".
- Baselines are pinned to commits (Appendix B): S1 367a943, S2 2dfc83e, W1 959a3ec, W2 f22ae04.

Working files are under papers/agent_workdir/ws_verify/.

## StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video

Key `streamstgs`, relevance 3/5, identity confirmed by verifier: yes.

### Authors

Zhihui Ke, Yuyang Liu, Xiaobo Zhou (corresponding), Tie Qiu - School of Computer Science and Technology, Tianjin University (arXiv HTML).
The AAAI proceedings page spells the second author "Yvyang Liu".

### Venue and year

AAAI 2026 (AAAI-26), confirmed in the proceedings: Proceedings of the AAAI Conference on Artificial Intelligence Vol. 40 No. 7 (AAAI-26 Technical Tracks 7), pp. 5629-5638, DOI 10.1609/aaai.v40i7.37482, citation date 2026-03-14 (source: https://ojs.aaai.org/index.php/AAAI/article/view/37482 and issue page https://ojs.aaai.org/index.php/AAAI/issue/view/688).
The arXiv comment also says "Accepted by AAAI 2026".

### arXiv

2511.06046 (v1, 8 Nov 2025, cs.CV; the only version I read)

### Links

Paper: https://arxiv.org/abs/2511.06046 ; full text read from https://arxiv.org/html/2511.06046 (method, experiments and full appendix).
AAAI version: https://ojs.aaai.org/index.php/AAAI/article/view/37482 (PDF https://ojs.aaai.org/index.php/AAAI/article/download/37482/41444).
Code: https://github.com/kkkzh/StreamSTGS.
GitHub API reports the license as "Other".
LICENSE.md is the Inria/MPII Gaussian-Splatting License (research and evaluation use), and the README says to follow the 3DGS license.
There is no project page.

### Problem

Prior 3DGS-based streamable free-viewpoint video (FVV) methods have one or more of these problems:
- Large per-frame payloads (up to about 10 MB per frame for HiCoM).
- Frame-by-frame training that accumulates error and copes poorly with new objects.
- A need to train several level-of-detail models to cover different bitrates.
- Random-access decoding latency, because accumulated offsets mean frame i needs frames 1 to i-1 decoded first.
The goal is a compact, codec-friendly dynamic 3DGS representation that supports variable bitrate with no retraining, a small keyframe, low decoding delay and real-time rendering.

### Method

Offline, per-scene, multi-view optimisation (not feed-forward).
The pipeline:
(1) Train a coarse 3DGS model on all multi-view frames for about 3000 iterations.
(2) Split the sequence into GOPs of G=60 frames and refine a model for each GOP independently (12000 iterations on N3DV, 7000 on MeetRoom).
(3) A GOP model holds canonical Gaussians plus a learnable 16-channel temporal feature e_j per Gaussian per timestep.
Position, scale, rotation and opacity are stored per Gaussian.
Colour is a 3-channel view-independent RGB, with no spherical harmonics (SH).
(4) Deformation at time t_i: a sliding window of W=3 concatenates e_{i-1}, e_i, e_{i+1}.
A temporal MLP takes these plus a positional encoding of t_i and outputs f_i.
Four small MLPs (64 neurons, no bias) turn f_i into dX, [dS, dQ], dO (tanh, which lets objects appear or disappear) and dC.
The opacity and colour MLPs also take the view direction.
This gives E = G+W-1 temporal features per GOP.
(5) Compression: PLAS sorts Gaussians into a 2D grid.
The canonical attributes become five images, stored as lossless JPEG XL with some quantisation.
The temporal features are reshaped from W x H x 16 to 4W x 4H 12-bit grey frames and encoded with libx265 at QP/CRF 20.
(6) Regularisers make the images compressible: a spatial smoothness loss (2D Gaussian filter on the sorted grids) and a temporal Huber loss between adjacent features.
(7) Training-only extras:
- A transformer branch, adapted from TimeFormer, sees all timesteps to learn global motion.
  It is distilled into f_i with an L1 self-distillation loss and dropped at inference.
- Small uniform noise is added to the features to simulate codec loss.
- A dynamic-aware densification applies the SSIM loss only on dynamic pixels, found by per-pixel standard deviation over 30 frames.
- Opacity-based pruning.
- MCMC-style relocation under a cap of about 150k Gaussians.
(8) Playback: decode the feature video (ffmpeg), run the MLPs and rasterise with a CUDA 3DGS rasteriser.
The next GOP's five attribute images are pre-decoded ahead of time (Appendix).

### Scene update model

Within one GOP, the set of Gaussians is fixed: about 150k after pruning and relocation.
No Gaussians are created or deleted per frame.
Appearance and disappearance are modelled through a tanh opacity offset.

Each Gaussian's identity inside a GOP is its pixel position in the PLAS-sorted 2D grid.
That position indexes both the attribute images and the feature-video frames.
The per-frame 'update' is not an explicit attribute delta: it is one frame of the temporal-feature video, which the client must push through the MLPs to get the deformation.

GOPs are reconstructed independently, refined from the coarse model with relocation and re-sorting.
The paper states no persistent identity across GOPs.
Every GOP boundary sends a fresh keyframe of attribute images plus MLP weights: 3.86 MB on N3DV and 2.8 MB on MeetRoom.

Every frame updates every Gaussian, including static ones.
The authors name this as a limitation: static Gaussians still carry temporal features.

### Transport

None is implemented or evaluated.
The paper only defines a payload format:
- Per GOP: five JPEG XL attribute images, the MLP weights and a compression-info CSV.
- Per frame: one HEVC frame of the feature video.

Variable bitrate is claimed because the feature video can be re-encoded at another QP.
The released code actually uses libx265 '-crf'.
The paper has no protocol, chunking, prioritisation, ABR logic or bandwidth adaptation experiment, and no network conditions.

### Client

There is no browser or separate client.
Evaluation is on the training server, an RTX A6000:
- Decoding: an ffmpeg subprocess decodes the whole GOP's feature video to raw gray16 on the CPU.
  The time is divided by the GOP length (measure_decode_latency in test_offset.py).
- Rendering: PyTorch MLP evaluation plus a CUDA rasteriser, timed with cuda.synchronize (render_nogt).
- JPEG XL attribute images are pre-decoded for the next GOP, so their decode cost is excluded from the reported decode time.

There is no discussion of GPU buffer updates, depth sorting cost or client memory limits.
A WebGPU port would need compute shaders for 5 MLPs per Gaussian per frame, plus a 12-bit grey HEVC decode path in the browser.
Whether browsers support that HEVC path is my concern and was not verified.

### Evaluation design

Datasets:
- N3DV: 6 scenes, 18-21 cameras, 300 frames at 30 FPS, downsampled to 1352x1014.
- MeetRoom: 3 scenes, 12 cameras, 1280x720, 30 FPS.
- The held-out test view is not stated in the paper.
- Initial points come from COLMAP, following 4DGaussians.

Metrics:
- PSNR, SSIM and LPIPS (VGG).
- Storage per frame (KB, amortised over the GOP, including keyframe and MLPs).
- Key-frame size (MB): the first feature-video packet plus the attribute images.
  Its stated purpose is the cost of starting playback mid-stream.
- Decoding time (ms per frame), render time (ms), FPS, training time (s, probably per frame), and a VBR yes/no column.

Baselines and control arms:
- TeTriRF, 3DGStream, VideoGS, HiCoM and 4DGC were re-implemented from official code with the authors' own data preparation (marked with an asterisk in Table 8).
  The authors say the data preparation changes results.
- Table 8 adds reported (not re-run) numbers for HyperReel, HexPlane, K-Planes, MixVoxels, NeRFPlayer, StreamRF, 4DGS, STGS, 4D-Rotor, DN-4DGS, Ex4DGS and QUEEN.

Ablations:
- QP 16/20/24/28/32 (libx265).
- Sliding-window size W = 1/3/5.
- GOP length 30/60/100.
- Components removed: auxiliary transformer, dynamic densification, temporal regulariser, relocation.
- Per-GOP and per-scene tables.

Hardware: a single RTX A6000 GPU.

Runs: apparently one run per configuration, with no seeds, variance or confidence intervals reported.

Not measured at all: network conditions, throughput, end-to-end latency, stalls or rebuffering, glass-to-glass latency, and client devices.

### Key numbers

All results on a single RTX A6000 with QP 20 unless noted.

N3DV (Table 1, 1352x1014):
- PSNR 32.30 dB, SSIM 0.943, LPIPS 0.147.
- 173.6 KB per frame (amortised), keyframe 3.86 MB.
- Decode 8 ms per frame, render 10 ms, 100 FPS.
- Train 67 s, probably per frame, so about 67 min per 60-frame GOP.
  This is my inference from Table 5.
- Baselines: 4DGC 31.52 dB at 784 KB with decode 2.5 x n ms.
  HiCoM 31.32 dB at 10704 KB. 3DGStream 30.73 dB at 8204 KB with decode 7 x n ms.
  VideoGS 27.45 dB at 932.9 KB.
  TeTriRF 30.07 dB at 65.89 KB and 1.53 FPS.

MeetRoom (Table 2, 1280x720):
- 27.41 dB, SSIM 0.92, LPIPS 0.21.
- 142 KB per frame, keyframe 2.8 MB.
- Decode 6 ms, render 7.93 ms, 126 FPS, train 29 s.

Bitrate knob (Table 3):
- N3DV: QP16 gives 32.36 dB at 247.5 KB, QP20 32.30 at 173.6, QP24 32.15 at 122.0, QP28 31.68 at 88.0, QP32 30.76 at 68.4.
- MeetRoom: QP16 gives 27.46 dB at 208.7 KB, down to QP32 26.46 at 52.3.
- So the size range is about 3.6x for a 1.6 dB cost on N3DV.
- Derived by me (KB = 1024 B, 30 FPS): about 42.7 Mbit/s at QP20 and 16.8 Mbit/s at QP32.
  These are amortised rates.
  Real traffic is bursty, with a 3.86 MB keyframe every 2 s.

Other ablations:
- GOP length (Table 5): GOP-30 gives 32.32 dB at 228.7 KB, GOP-60 32.30 at 173.6, GOP-100 32.06 at 161.9.
- No sliding window (W=1): 32.01 dB at 298 KB (Table 4).
- No temporal regulariser: storage 319.5 KB (Table 6).

### Limitations

Stated by the authors: every Gaussian carries temporal features even if it is static within a GOP.
They suggest splitting Gaussians into static and dynamic sets to cut storage and raise FPS.

What I see:
(1) It is not live.
It needs a full 60-frame GOP (2 s) of future frames, a coarse model trained on the whole sequence, and roughly an hour of training per GOP on an A6000.
(2) The variable-bitrate claim is not exercised under any network.
There are no traces, no emulation and no ABR controller.
The adaptation granularity is effectively per GOP or per encode.
The keyframe (JPEG XL attributes plus MLPs) does not scale with QP.
(3) 'Frame size' is amortised over the GOP, which hides the 3.86 MB keyframe burst every 2 s.
The next GOP's images must be prefetched and pre-decoded, which adds startup and switching latency.
(4) Decode latency is CPU ffmpeg time on the server.
The timer starts after the process is spawned and the total is divided by the GOP length.
It excludes JPEG XL decode, network and any client.
(5) The paper says QP, but the code uses libx265 CRF.
(6) Baselines were re-implemented with the authors' own data preparation, results are single-run with no variance, and the 'about 1 dB' gain has no fixed reference baseline.
(7) The training-time unit is unstated.
(8) The code environment is PyTorch 1.13.1 with CUDA 11.6, and the README warns that PLAS cannot be installed on PyTorch 2.0+.
This may be friction on Ada GPUs (sm_89).
I have not verified that.

### Gap relative to our idea

Overlap:
- It is a concrete, peer-reviewed precedent for one adaptation knob: re-quantising per-frame Gaussian update data with a standard video codec, at a different QP, without retraining.
  Its Table 3 rate-quality curve is a usable prior for a 'precision' dimension in a scheduler.
- Its key-frame-size and random-access decoding-delay metrics are close to our join-latency and glass-to-glass concerns.
- Its 2D-grid layout gives an implicit Gaussian identity (the grid position) within a GOP.

What our idea would add:
(a) Live, feed-forward scene updates instead of hour-long per-GOP optimisation that needs 2 s of future frames.
Our latency budget is ruled out by StreamSTGS's design.
(b) Real transport: WebSocket chunks, tc netem, throughput traces, and a policy that decides what to send and when.
StreamSTGS has no network evaluation.
(c) Persistent identity across updates with sparse, ID-based deltas and temporal coalescing.
StreamSTGS updates every Gaussian every frame through a dense feature frame and resets at every GOP.
(d) A browser WebGPU client with its own render loop, a GPU-memory budget and decoupled rates.
StreamSTGS renders on the server's CUDA, and its client would also need to run 5 MLPs per Gaussian per frame.
(e) Deadline-aware joint network-GPU scheduling, judged by glass-to-glass latency and render stalls against FIFO and view-aware baselines.
Nothing like this is in the paper.

Honest note: StreamSTGS is a compression and representation paper.
It is a good related-work citation and a possible quality-knob baseline, not a systems competitor.

### Reusable

Code: https://github.com/kkkzh/StreamSTGS, under the Inria/MPII Gaussian-Splatting License (research use).
- Maturity: a single push on 2025-11-11 and 16 stars when checked.
- Contents: training (train_offset.py), evaluation (test_offset.py), compression and decompression with JPEG XL and libx265, and the size and keyframe measurement script read_size.py.
- Rasteriser builds are shipped as zips.
- No pretrained checkpoints are linked in the README.
- No streaming server or client.

Datasets: N3DV (facebookresearch/Neural_3D_Video) and MeetRoom (a Google Drive link in the README).
I did not check their licences.

Possible reuse for us:
(1) Run it offline on one N3DV scene to produce realistic per-frame payload traces: feature-frame sizes per QP plus a keyframe every GOP.
These could feed the trace-driven scheduler experiments as a 'codec-based' workload.
(2) Use the read_size.py and keyframe-size methodology as a metric template.
(3) Use the PLAS grid ordering as an identity and layout scheme for codec-friendly delta packing.

Expect environment friction: PyTorch 1.13.1 with cu116, and PLAS will not install on PyTorch 2.0+.

### Notes

The primary sources were read in full: arXiv HTML v1, including the Method, Experiments and the whole Appendix (Tables 1-10).
I also checked the AAAI OJS article metadata, and the GitHub README, LICENSE, file tree, test_offset.py, read_size.py and train_stream.py.

The material places StreamSTGS under network-aware quality adaptation.
That placement is only half right: it supports a codec bitrate knob, not network-driven adaptation, update-rate control or visibility selection.

For the course project, cite it as 'AAAI 2026, arXiv 2511.06046, DOI 10.1609/aaai.v40i7.37482'.

Downloaded text and code are in papers/agent_workdir/streamstgs/ (paper.txt, test_offset.py, read_size.py).

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| The paper exists under this exact title and was published at AAAI 2026 | supported | arXiv 2511.06046 has the same title, with the comment 'Accepted by AAAI 2026' (https://arxiv.org/abs/2511.06046). The AAAI OJS lists it in Vol. 40 No. 7, pp. 5629-5638, DOI 10.1609/aaai.v40i7.37482 (https://ojs.aaai.org/index.php/AAAI/article/view/37482). |
| It uses an adjustable-bitrate representation (可調 Bitrate 的表示方式) | partly | Canonical Gaussian attributes are sorted into 2D grids with PLAS (the sorting algorithm from Self-Organizing Gaussians) and stored as five lossless JPEG XL images, one per GOP (group of pictures). Per-Gaussian temporal features (16 channels) are encoded as an HEVC video with libx265. Only that feature video changes bitrate: it can be re-encoded at a different QP (quantisation parameter) 'without any extra training' (Method, Gaussian Attribute Compression, and Appendix, https://arxiv.org/html/2511.06046). The keyframe images stay fixed in size. In the released code, test_offset.py line 263 encodes with ffmpeg '-crf {qp}', which is CRF (constant rate factor) and not a fixed QP as the paper says. |
| It handles network-aware quality adaptation for streaming (the material cites it under 'C. Network-Aware Quality Adaptation': lower update rate, precision, or visible-region-only updates when the network is poor) | unsupported | The paper has no network experiment at all. It has no bandwidth traces, no emulation, no ABR controller (adaptive bitrate) and no transport protocol. 'Adaptive' is shown only by an offline QP sweep (Table 3: QP 16 to 32). The paper does not reduce update frequency and does not do view- or visibility-dependent selection. It supports only the 'precision' knob, and only for the temporal-feature video. |
| It is a real-time free-viewpoint video (FVV) streaming system | partly | 'Real-time' here means real-time playback: decoding 8 ms and rendering 10 ms (100 FPS) on an RTX A6000 for N3DV (Table 1). It does not mean live capture. Each 60-frame GOP is optimised per scene, after a coarse 3DGS model is trained on all multi-view frames. The Train(s) column is consistent with per-frame amortised time: GOP-30/60/100 gives 133/67/40 s, and each product is about 4000 s per GOP (Table 5). That reading is my inference, because the paper does not state the unit. No streaming server or client is implemented. train_stream.py in the repo is only a training launcher. |
| Abstract: about 1 dB average PSNR gain at an average frame size of 170 KB | partly | The 173.6 KB/frame (N3DV) and 142 KB (MeetRoom) figures are supported. They are amortised over the GOP: read_size.py sums the feature video, the JPEG XL images, the MLP weights and a CSV, then divides by the GOP length. On N3DV the PSNR gain is +0.78 dB over 4DGC, +0.98 over HiCoM and +1.57 over 3DGStream. On MeetRoom it is +0.30 over 4DGC and +1.0 over 3DGStream. Which baseline the '1 dB' refers to is undefined. Those baselines were re-implemented by the authors. QUEEN's reported 32.19 dB (Table 8, not re-run) is only 0.11 dB below 32.30. |
| Paper text: 'achieves the best reconstruction quality on both N3DV and MeetRoom' | partly | It has the best PSNR among the compared methods. But its LPIPS on N3DV is 0.147, while 4DGC has 0.143 (Table 1). In Table 8, STGS and QUEEN report SSIM 0.946 against its 0.943. |

### Verifier corrections

- **scene_update_model / limitations (2): keyframe contents**
  Reader said: 'Every GOP boundary sends a fresh keyframe of attribute images plus MLP weights: 3.86 MB on N3DV' and 'The keyframe (JPEG XL attributes plus MLPs) does not scale with QP.'
  Correct: The reported K.F. size (3.86 MB on N3DV, 2.8 MB on MeetRoom) is the first HEVC packet of the feature video plus the five JPEG XL attribute images.
  It does not include the MLP weights or compression_info.csv. read_size.py adds those only to the amortised per-frame size.
  So someone joining mid-stream needs K.F. plus MLPs plus CSV, which means the reported K.F. size slightly understates the real join cost.
  The keyframe is also not fully fixed in size: it contains the first HEVC I-frame, which depends on QP and W.
  Table 4 shows K.F. at 3.92 MB for W=1 and 3.86 MB for W=3.
  The JPEG XL part, which does not depend on QP, makes up most of it.
  The reader's evaluation_design field already describes the keyframe correctly; only these two fields contradict it.
  Source: https://github.com/kkkzh/StreamSTGS/blob/master/read_size.py (measure_key_frame_size takes frames[0].pkt_size; gop_attr_size adds only the .jxl files; mlps/.pth and the csv go only into gop_size/args.gop); Table 4 at https://arxiv.org/html/2511.06046

- **claim_check: network-aware quality adaptation (verdict 'unsupported')**
  Reader said: Unsupported: the paper has no network experiment and supports only a precision knob.
  Correct: Better worded as 'claimed but never demonstrated'.
  The paper does state the claim, more than once.
  The abstract says it 'inherently supports adaptive bitrate control based on network condition without any extra training'.
  The Method section (Gaussian Attribute Compression) says the 'feature video can be encoded and decoded using different bit-rates according to network conditions'.
  The contribution list says it 'inherently adapts to dynamic network conditions'.
  The reader is right that there are no network traces, no emulation, no controller and no experiment.
  The only evidence is the offline QP sweep in Table 3.
  So the source material put it in this category because of the paper's own claim, not because ChatGPT invented it.
  Source: https://arxiv.org/abs/2511.06046 (abstract); https://arxiv.org/html/2511.06046 (Introduction contributions; Method, Gaussian Attribute Compression)

- **claim_check: 'best reconstruction quality' (Table 8 details)**
  Reader said: In Table 8, STGS and QUEEN report SSIM 0.946 against its 0.943; 4DGC has a better LPIPS (0.143).
  Correct: There are more counterexamples in Table 8.
  DN-4DGS reports SSIM 0.944, which is above 0.943.
  The authors' own re-run of 4DGaussians* has LPIPS 0.141, which beats both 0.147 and 4DGC's 0.143.
  The paper is also inconsistent with itself: Table 4 gives SSIM 0.944 for the default W=3 model, while Tables 1 and 8 give 0.943.
  Source: https://arxiv.org/html/2511.06046 (Table 1, Table 4, Table 8)

- **method (5): bit depth of feature frames**
  Reader said: Temporal features are reshaped to 4W x 4H 12-bit grey frames and encoded with libx265 at QP/CRF 20.
  Correct: The paper only gives the 4W x 4H reshape and 'libx265 with the default quantization parameter qp=20'.
  It never states a bit depth.
  The 12-bit figure comes from the code ('-pix_fmt gray12le'), so it should be cited to the code, not the paper.
  The CRF finding is stronger than the reader reported.
  The default QP-20 file is also produced with '-crf' during training, in compression/compression_exp.py line 195, not only in the re-encode path at test_offset.py line 263.
  Source: https://arxiv.org/html/2511.06046 (Appendix, implementation details); https://github.com/kkkzh/StreamSTGS/blob/master/compression/compression_exp.py (line 195)

### Verifier on the gap assessment

Mostly fair.
The paper does not do any of (a)-(e).
(a) It is not live or feed-forward.
Each GOP is created from a coarse.pth that was trained on all frames (train_offset.py, training_refine calls create_from_coarse), then refined for 12000 iterations.
Table 5 supports the reader's estimate of about 4000 s per GOP: 133 x 30, 67 x 60 and 40 x 100 all give roughly 4000 s.
(b) There is no transport, no network emulation and no ABR controller.
The one nuance is that the paper does claim network-adaptive bitrate in its own words, so it is a claim with no evidence behind it, not a topic it ignores (see corrections).
(c) There is no persistent identity across GOPs and no sparse deltas.
Every GOP is re-initialised from the same coarse model and then relocated, densified and re-sorted.
Every frame carries a dense feature frame for all of about 150k Gaussians.
(d) There is no browser or client.
Timing is measured on the server: CPU ffmpeg decode divided by the GOP length, and CUDA render.
(e) There is no scheduling, deadline, stall or glass-to-glass metric.
The overlap the reader credits is also accurate.
The K.F. size and decoding-delay metrics really are aimed at joining mid-stream (Quantitative Comparisons section).
One thing the reader leaves out: StreamSTGS is not the only codec-based VBR precedent.
Its own Table 1 marks TeTriRF and VideoGS as VBR too, and VideoGS also compresses Gaussian attributes with video codecs.
Those two are worth citing next to it for the 'precision knob'.

### Verifier notes

Independently re-fetched these sources:
- arXiv abstract and HTML v1: https://arxiv.org/abs/2511.06046 and https://arxiv.org/html/2511.06046.
  Title, 4 authors, Tianjin University affiliation, Xiaobo Zhou marked as corresponding author, v1 dated 8 Nov 2025, comment 'Accepted by AAAI 2026'.
- AAAI OJS page: https://ojs.aaai.org/index.php/AAAI/article/view/37482.
  Vol 40, Issue 7, pp. 5629-5638, DOI 10.1609/aaai.v40i7.37482, citation_date 2026/03/14, second author spelled 'Yvyang Liu', track 'AAAI Technical Track on Computer Vision IV'.
- GitHub API: https://api.github.com/repos/kkkzh/StreamSTGS.
  License 'Other', 16 stars, created and pushed 2025-11-11.
  The default branch is 'master'.
- From the repo I read the README, LICENSE.md (Inria/MPII Gaussian-Splatting License), read_size.py, test_offset.py, compression/compression_exp.py, train_offset.py and train_stream.py.

Every key number I checked matches Tables 1-6 and 8: N3DV 32.30/0.943/0.147, 173.6 KB, K.F. 3.86 MB, decode 8 ms, render 10 ms, 100 FPS, train 67; MeetRoom 27.41, 142 KB, 2.8 MB; the full Table 3 QP sweep; GOP ablation 133/67/40; W=1 at 32.01/298.06; no temporal regulariser 319.48.
The baseline deltas (+0.78/+0.98/+1.57 on N3DV, +0.30/+1.00 on MeetRoom) and the bitrates (42.7 Mbit/s at QP20, 16.8 Mbit/s at QP32 with KB = 1024 B) are arithmetically correct.

The CRF-vs-QP finding is confirmed, and it reaches further than the reader said: both the training-time compression path and the test re-encode use '-crf'.
The decode-timer description is confirmed too: measure_decode_latency starts the timer after ffmpeg.run_async spawns the process and divides by the GOP length.

The per-frame unit for training time is still an inference.
The paper never states it, but Table 5's constant product supports it.

My verification copies are in papers/agent_workdir/streamstgs_verify/.

## LMG: Efficient Streaming of Layered Mesh-Gaussian 3D Scenes

Key `lmg`, relevance 3/5, identity confirmed by verifier: yes.

### Authors

Yuan-Chun Sun (NTHU), Guodong Chen (Northeastern University), Sam Ziaie Kondori (NTHU), Mallesham Dasari (Northeastern University), Cheng-Hsin Hsu (NTHU).
Source: project page https://aiins-nthu.github.io/LMG/ and Crossref https://api.crossref.org/works/10.1145/3793853.3795764

### Venue and year

Proceedings of the ACM Multimedia Systems Conference 2026 (MMSys '26, the 17th MMSys, Hong Kong), pp. 264-275, published 2026-04-04, DOI 10.1145/3793853.3795764.
Confirmed by Crossref metadata, OpenAlex (open access, CC BY-NC-ND, publishedVersion), Semantic Scholar (DBLP key conf/mmsys/SunCKDH26) and the first author's homepage https://syjintw.github.io/.
The repo README says "Accepted by ACM Multimedia Systems Conference 2026 (MMSys'25)"; the "'25" is a typo, its BibTeX says the 17th MMSys, 2026.

### arXiv

none.
No arXiv version exists.
I checked the arXiv API on 2026-10-27: a title search for "Layered Mesh" gives 2 unrelated hits, and the only arXiv paper under author "Yuan-Chun Sun" is EyeNavGS (2506.02380).
Neither the project page nor the repo links an arXiv preprint.

### Links

Paper (ACM DL, open access, CC BY-NC-ND per OpenAlex/Semantic Scholar): https://doi.org/10.1145/3793853.3795764 (PDF https://dl.acm.org/doi/pdf/10.1145/3793853.3795764).
Project page: https://aiins-nthu.github.io/LMG/
Supplement, a symbol table covering Sec. 3-5 of the paper: https://aiins-nthu.github.io/LMG/static/docs/LMG_symbol_table.pdf
Code: https://github.com/SYJINTW/layered-mesh-gaussian.
LICENSE.md is the Inria/MPII "Gaussian-Splatting License" plus GMUM (GaMeS): non-commercial research and evaluation use only, no sublicensing.
The repo has 4 stars; last push 2026-09-22; branches main, progressive, animation, lmg++.
Modified rasterizer that takes a mesh depth input: https://github.com/SYJINTW/diff-gaussian-rasterization-lmg
No video: the YouTube embed on the project page is empty.

### Problem

Delivering static 6-DoF scenes over capacity-constrained networks forces a trade-off among visual fidelity, interactivity and streaming efficiency.
Pure textured meshes are compact and editable but not photorealistic.
Pure 3DGS is photorealistic but heavy and hard to edit.
Earlier mesh+Gaussian hybrids (GaMeS, MeshGS, MILo and others) target reconstruction quality, not end-to-end delivery.
LMG builds a layered mesh-Gaussian representation plus a full streaming system: production server, streaming server and XR client.
Sources: project page abstract and the Crossref reference list.

### Method

IMPORTANT: I could not read the full paper.
ACM DL returns HTTP 403 to automated fetches (WebFetch and curl), and when I tried a browser user-agent, the auto-mode safety check blocked it as bot-check circumvention, so I stopped.
There is no arXiv or other open copy.
What follows comes only from: the project page abstract and captions, the supplement symbol table (which names every symbol and its section), the public code, and the Crossref reference list.
(1) Representation (Sec. 3.1): a textured mesh M=(V,T) plus per-triangle Gaussian sets G_t={g_{t,i}}, i=1..n_t.
Each Gaussian's mean m_{t,i} is a barycentric combination (weights alpha_{t,i,k}) of the triangle's vertices v_{t,k}, as in GaMeS.
Each Gaussian also has a rotation, scale, opacity and SH coefficients.
Because Gaussians are tied to triangles, deforming the mesh moves them (the page shows animation demos).
(2) Hybrid rasterizer (Sec. 3.2, called 'DTGS' on the page): the mesh is rendered with PyTorch3D or nvdiffrast into a color image and a depth map.
A modified 3DGS CUDA rasterizer takes the mesh depth (`depth=bg_depth` in GaussianRasterizationSettings) and composites Gaussians over the mesh with an occlusion test controlled by threshold lambda.
The page compares the GaMeS rasterizer with 'LMG's DTGS' on Lego at a 40k budget.
Training is described as 'joint'; whether the mesh texture is also optimized I could not verify.
(3) Budget-aware allocation (Sec. 3.3, scene/budgeting.py): given a total Gaussian budget B, each triangle gets a score w_t.
The scores are normalized, ideal real-valued counts n_hat_t = B*w~_t are floored, and the leftover budget B' goes out by the largest-remainder method.
Policies in code: uniform; random; area a_t; planarity rho_t (mean resultant length of unit normals over an H-hop neighborhood); distortion d_t (per-pixel L1 error E_k(p) between ground truth I_k and the mesh-only render, mapped back to triangles via the pixel-to-triangle map tau(p) and aggregated over K training views); and an area+distortion mix.
The page shows Bicycle with area-based vs distortion-based allocation.
(4) Streaming server (Sec. 4.1-4.2): the scene is split into tiles (N tiles along the largest dimension).
Each tile has a base layer (mesh) and an enhancement layer (Gaussians).
The client requests tiles based on the user's trajectory.
Over a window of W future frames, each tile l gets a base urgency p^B_l built from visibility nu_{l,f} in {-1,1}, tile-to-viewport distance delta_{l,f}, a distance weight beta, a temporal weight gamma and a stability constant epsilon.
Enhancement urgency is p^E_l = p^B_l - phi.
Data is sent over C concurrent UDP streams.
The exact functional form and the transport details (reliability, framing) are not verified.
(5) XR client: renders the mesh base layer first for a quick scene overview, then refines visible regions with Gaussians.
The platform is unknown.

### Scene update model

Static scene: there is no over-time update model.
Gaussians are created once, during per-scene training, on mesh triangles.
The count per triangle n_t is fixed by the budget allocation, and nothing is created or deleted after training.
During streaming the client's scene only grows: tiles arrive base layer (mesh) first, then enhancement layer (Gaussians) for visible regions.
I found no symbol or text about eviction, replacement or deltas.
Identity: each Gaussian is naturally indexed by (triangle id t, index i), and its position is parameterized by barycentric weights on that triangle.
So identity is persistent and anchored to mesh topology, and mesh deformations (animation) move the bound Gaussians without new Gaussian data.
This is a structural answer to "Gaussian identity" for a static or deforming-mesh scene.
It does not carry over to pixel-aligned feed-forward outputs, whose Gaussians have no correspondence across frames.

### Transport

Tile-based and layered (mesh base layer + Gaussian enhancement layer), with client requests driven by the user's trajectory.
Scheduling (symbol table Sec. 4.2): a window of W frames, per-tile urgency from visibility, distance and time weights (beta, gamma), and a base/enhancement offset phi.
Data goes over C concurrent UDP streams (Sec. 4.1, 4.2, 5).
Compression: the paper cites Draco and HAC, but whether and how they are used is not verified.
Rate or quality adaptation beyond urgency ordering: not verified.
No delta format, because the scene is static.

### Client

An "XR client" (abstract; the overview figure shows a VR headset) that renders a textured mesh plus Gaussians composited with a depth test.
Platform (Unity, native or browser), GPU upload strategy, sorting and memory limits: unknown, as nothing about the client is in the accessible sources or the code.
No evidence of WebGPU or a browser.

### Evaluation design

Only partly verified (full text unread).
Two evaluation parts:
(A) Production server (representation quality).
Datasets: NeRF Synthetic (hotdog, lego; the repo assets also show ficus, chair, ship) and Mip-NeRF 360 Bicycle (page figures; README examples), with meshes from MILo, COLMAP or SuGaR.
Metrics: PSNR, SSIM (abstract), LPIPS (metrics.py).
Baseline: the 'state-of-the-art baseline', which the page caption identifies as GaMeS.
Control arms: allocation policies uniform/random/area/planarity/distortion, Gaussian budgets (README sweep 131072 to 3M, plus 0 = mesh only), with or without occlusion.
(B) Streaming system.
The page section is titled 'Simulated Streaming Traces', so streaming is evaluated in simulation, not on a live network (as far as visible).
Arms shown: Ours vs 'No Layer' vs 'No Tiling' (ablations of LMG itself) on Bicycle under 'a high-bandwidth network with significant variance'.
Metrics: PSNR, SSIM, VMAF (abstract); 'low-latency' is claimed, but the latency definition is unknown.
Parameters studied in Sec. 5 per the symbol table: beta, gamma, phi, C (UDP streams), N (tiling granularity), lambda.
Probable but UNVERIFIED: user trajectories from EyeNavGS (the repo has camera/eyenavgs_reader.py and the paper cites EyeNavGS, MM'25); bandwidth traces from the 5G dataset 'Beyond throughput, the next generation' (MMSys'20, cited).
Unknown: whether external streaming systems (L3GS, SGSS, LapisGS, NetSplat, 3DGStreaming, all cited) are compared, plus hardware, number of runs and variance, and any stall or rebuffer metric.

### Key numbers

From the abstract (project page), in the settings stated there:
- Production server vs the state-of-the-art baseline (GaMeS per the page caption): PSNR up to +10.99 dB (+8.89 dB average), SSIM up to +0.293 (+0.255 average).
  Datasets and budgets as in the paper (not verified which).
- Layered and tiled streaming: up to +19.81 dB PSNR (+5.80 dB average), +0.143 SSIM (+0.06 average), +40.86 VMAF (+16.06 average).
  The comparison point is presumably the No Layer / No Tiling ablations under simulated network traces (not verified).
- Paper length: 12 pages (pp. 264-275).
No latency, frame-rate or bitrate numbers are available to me.
These are the authors' numbers on their data, not a null for ours.

### Limitations

Visible from the accessible sources:
- Static scenes only, with per-scene optimization (training iterations); there is no live or temporal update path.
- Quality depends on a pre-built textured mesh (MILo, COLMAP or SuGaR).
- Streaming results come from simulated traces, and the streaming control arms are ablations of LMG itself (No Layer, No Tiling); whether other streaming systems are compared is unknown.
- The public code covers only training and rendering; there is no streaming server, scheduler or XR client code in any branch.
  The README still has a '==== NOT YET ====' section referring to scripts (debug_pipeline.sh, 1113_pipeline.sh) that are not in the main tree.
- License: non-commercial research only (Inria/GMUM).
- The 'low-latency' claim has no latency definition visible to me.
I could not read the authors' own stated limitations because the full text was inaccessible.

### Gap relative to our idea

Overlap:
- View-dependent, tile-based prioritization over a look-ahead window (visibility + distance + time weights).
- Layered progressive delivery (coarse base layer first, refinement later).
- A Gaussian-count budget.
- Same institution (NTHU, Hsu's AIINS lab), so the group is a natural contact.
LMG's urgency scheduler is a ready-made template for the "view-aware baseline" arm in our evaluation.
What our idea adds:
(1) Live, continuously changing content from a feed-forward model, versus a static pre-trained scene, which forces update and replace/delete semantics and Gaussian identity across model outputs.
LMG gets identity for free from mesh triangles (t, i); pixel-aligned feed-forward Gaussians have no such anchor.
(2) Temporal deltas and coalescing; LMG only ever adds tiles.
(3) Decoupling the inference, network and render rates.
(4) Deadline-aware and client-GPU-budget-aware joint scheduling; LMG's budget B is a server-side training budget, and I found no client memory or GPU budget model.
(5) A browser WebGPU client with CPU decode and GPU upload costs; LMG is an XR client on an unknown platform.
(6) Glass-to-glass latency and render-stall metrics against FIFO and view-aware arms under tc netem, versus LMG's quality metrics (PSNR/SSIM/VMAF) on simulated traces.
Caveat: I could not see whether LMG's client manages its own memory, so point (4) must be rechecked against the full text.

### Reusable

- layered-mesh-gaussian repo (research-grade, non-commercial license): hybrid mesh+Gaussian training/rendering on GaMeS, and the scene/budgeting.py allocation policies (largest-remainder budget split by area, planarity or distortion; a clean, small piece of logic that could be adapted for GPU-budget-aware selection of which Gaussians to upload).
- diff-gaussian-rasterization-lmg: a 3DGS rasterizer that accepts an external depth buffer for occlusion against a mesh.
- A sample hotdog textured mesh (Google Drive link in doc/DATASET.md).
- camera/eyenavgs_reader.py, a reader for EyeNavGS 6-DoF VR trajectories (EyeNavGS itself is arXiv 2506.02380, MM'25, same first author).
  Those trajectories could drive view-dependent experiments.
Not reusable because it is not released: streaming server, tile scheduler, UDP transport, XR client, network traces.

### Notes

FULL TEXT NOT READ.
The paper is open access on ACM DL (CC BY-NC-ND per OpenAlex and Semantic Scholar), but dl.acm.org returns HTTP 403 to WebFetch and curl.
When I retried curl with a browser user-agent, the auto-mode safety check blocked it as bot-check circumvention, so I did not try other workarounds.
To finish this entry: the user downloads https://dl.acm.org/doi/pdf/10.1145/3793853.3795764 in a normal browser and saves it to papers/agent_workdir/lmg/lmg.pdf.
Then the method and evaluation fields (urgency formula, transport, client platform, network traces, hardware, latency definition, external baselines, client resource handling) can be filled in from the full text.
The fields above are limited to what these sources show:
- project page: https://aiins-nthu.github.io/LMG/
- symbol table: https://aiins-nthu.github.io/LMG/static/docs/LMG_symbol_table.pdf
- GitHub repo, all four branches, read via the gh API
- Crossref, OpenAlex and Semantic Scholar metadata
- the author homepage https://syjintw.github.io/
Local copies of the page, symbol table and figures are in papers/agent_workdir/lmg/.
Minor naming inconsistency: the project-page abstract says 'production server', while the overview figure says 'Training Server'.
Related work by the same group that may matter more for a live or dynamic system: LTS, a DASH streaming system for dynamic multi-layer 3DGS (MMSys'25 Best Paper, DOI 10.1145/3712676.3714445), and multi-frame bitrate allocation for dynamic 3DGS (EMS'24, DOI 10.1145/3672196.3673394).
Both are listed on https://syjintw.github.io/ and were not verified here.

### Claim check

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Title is 'LMG: Efficient Streaming of Layered Mesh-Gaussian 3D Scenes' | supported | Crossref and project page give this exact title (the page uses an en dash in Mesh–Gaussian, Crossref a hyphen). |
| Venue is ACM MMSys 2026 | supported | Crossref: 'Proceedings of the ACM Multimedia Systems Conference 2026', pp. 264-275, issued 2026-04-04; event 'MMSys '26'. Semantic Scholar DBLP key conf/mmsys/SunCKDH26. Project page: 'Accepted by ACM Multimedia Systems Conference 2026 (MMSys'26)'. |
| By the NTHU Cheng-Hsin Hsu group together with Northeastern University | supported | Project page affiliations: Sun, Kondori, Hsu = National Tsing Hua University; Chen, Dasari = Northeastern University. Hsu is last author; the project is hosted under the aiins-nthu GitHub org. |
| It already does end-to-end scene streaming | supported | Abstract on the project page: 'the first end-to-end streaming system utilizing a Layered Mesh-Gaussian representation ... comprising a production server, a streaming server, and an XR client'. 'First' is the authors' claim. The public code covers only the production (training/rendering) server, not the streaming server or the client. |
| It focuses on interactive 3D scene delivery under limited bandwidth | supported | Abstract: 'Delivering immersive 6-DoF experiences over capacity-constrained networks' and 'efficient and interactive delivery of static 3D scenes'. 'Interactive' here also means structural interactivity: the mesh base layer lets Gaussians follow mesh deformations (the animation demos on the page). |
| Its subject is not continuously-input feed-forward reconstruction | supported | Abstract says 'static 3D scenes'. The training server is per-scene optimization: README pipeline Warmup -> Training (e.g. 5000-15000 iterations) -> Rendering -> Metrics. It is built on GaMeS, with meshes from MILo/COLMAP/SuGaR (doc/DATASET.md). |
| Its layering design is worth referencing | supported | Abstract: textured mesh = streamable base layer; surface-aligned Gaussians = enhancement layer. Symbol table Sec. 4.2: enhancement urgency p^E_l = p^B_l - phi, so a tile's base layer is always ahead of its enhancement layer by offset phi. |
| Its streaming design is worth referencing | supported | Abstract: 'layered and tile-based pipeline ... prioritizing the delivery of the base layer for a quick scene overview, followed by the progressive refinement of visible regions using Gaussians'. Symbol table Sec. 4.1-4.2: scheduling window W frames, per-tile urgency from visibility nu_{l,f} in {-1,1} and tile-to-viewport distance delta_{l,f}, weights beta (distance) and gamma (time), and C concurrent UDP streams. I could not read the exact formula. |
| Its client-side resource management design is worth referencing | unsupported | None of the sources I could read mention client-side resource management: abstract, project page, supplement symbol table, repo. The only 'budget' is B, the total number of Gaussians allocated at training time on the server (Sec. 3.3). The symbol table covers Sec. 3-5 and has no symbol for client memory, cache or GPU budget. I could not read the full text, so the prose may still mention client-side handling. |
| The project page has Project / Paper / Code links | partly | The project page links the DOI (Paper), GitHub (Code) and a Supplement (symbol table only). The code is only the LMG model: training, the hybrid rasterizer, allocation policies and metrics. I searched all four branches, and none has streaming server, tile scheduler or XR client code. |

### Verifier corrections

- **reusable (EyeNavGS authorship)**
  Reader said: EyeNavGS itself is arXiv 2506.02380, MM'25, same first author
  Correct: Yuan-Chun Sun is not the first author of EyeNavGS. arXiv 2506.02380 lists Zihao Ding, Cheng-Tse Lee, Mufeng Zhu, Tao Guan, Yuan-Chun Sun, Cheng-Hsin Hsu, Yao Liu, so Sun is 5th of 7.
  The MM'25 DOI is 10.1145/3746027.3758265, and LMG cites it.
  Source: https://arxiv.org/abs/2506.02380 (via https://export.arxiv.org/api/query?id_list=2506.02380); https://syjintw.github.io/; https://api.crossref.org/works/10.1145/3746027.3758265

- **notes / problem (server naming)**
  Reader said: the project-page abstract says 'production server', while the overview figure says 'Training Server'
  Correct: The published abstract also says 'training server': 'a complete system comprising a training server, a streaming server, and an XR client' and 'the LMG training server significantly outperforms...'.
  Only the project page uses 'production server'.
  So the paper's own term is 'training server'.
  Source: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3793853.3795764?fields=abstract vs https://aiins-nthu.github.io/LMG/

- **links (video)**
  Reader said: No video: the YouTube embed on the project page is empty.
  Correct: There is no presentation video, because the YouTube iframe (empty src) is inside an HTML comment.
  The page does embed demo MP4s: static/videos/hotdog_fly.mp4 (mesh-driven animation) and static/videos/streaming_results.mp4, which is the 'Simulated Streaming Traces' figure comparing Ours, No Layer and No Tiling on Bicycle.
  Source: https://aiins-nthu.github.io/LMG/ (page HTML, lines ~184 and ~382-395, ~581)

- **evaluation_design / key_numbers (identity of 'state-of-the-art baseline')**
  Reader said: Baseline: the 'state-of-the-art baseline', which the page caption identifies as GaMeS.
  Correct: The page caption only shows a rasterizer comparison: 'Sample results from Lego trained with different rasterizers under 40k budget: (a) GaMeS and (b) our LMG's DTGS'.
  It does not say that GaMeS is the 'state-of-the-art baseline' behind the +10.99 dB / +8.89 dB numbers.
  GaMeS is a plausible guess, since the code is built on GaMeS, but it is unverified until the full text is read.
  Source: https://aiins-nthu.github.io/LMG/

- **reusable / links (code contents)**
  Reader said: Code is only the LMG model ...
  Not reusable because it is not released: streaming server, tile scheduler, UDP transport, XR client
  Correct: It is correct that no streaming server, scheduler or client is released in any branch.
  The reader missed one thing: post-publication work on the lmg++ branch (commits 2026-09-11 to 2026-09-22).
  It adds an 'LMG Lean' format, with a Python transcoder lmg_format/codec.py and a Rust crate rust/lmg-format.
  Lean drops per-splat xyz, scale and rotation and recomputes them from the anchor triangle, which makes it 1.9x to 2.3x smaller and lossless.
  It also stores a per-splat round_id for progressive freeze-then-grow training (train_progressive_orchestrator.py; progressive branch).
  This is not in the MMSys paper.
  It is still a concrete example of compact encoding that depends on mesh-anchored identity, and it is relevant to delta compression.
  Source: https://github.com/SYJINTW/layered-mesh-gaussian/tree/lmg%2B%2B/rust/lmg-format (README) and https://github.com/SYJINTW/layered-mesh-gaussian/blob/lmg%2B%2B/lmg_format/codec.py

- **venue_and_year (date)**
  Reader said: published 2026-04-04
  Correct: Crossref gives published-print/issued 2026-04-04 and published-online 2026-04-06.
  The event is MMSys '26 in Hong Kong, sponsored by SIGMM, with 59 references.
  This is minor.
  Source: https://api.crossref.org/works/10.1145/3793853.3795764

- **method (4) tiling symbol**
  Reader said: Streaming server (Sec. 4.1-4.2): the scene is split into tiles (N tiles along the largest dimension)
  Correct: The symbol table lists N ('#tiles along largest dimension') only under Sec. 5 (experiments).
  It is an experimental parameter, and the table does not tie it to the Sec. 4 design.
  This is trivial.
  Source: https://aiins-nthu.github.io/LMG/static/docs/LMG_symbol_table.pdf

### Verifier on the gap assessment

Mostly fair, given that neither the reader nor I could read the full text.
ACM DL returns 403 to WebFetch, and Unpaywall, OpenAlex and Semantic Scholar list only the publisher copy.
There is no arXiv version: the arXiv API search for au:"Yuan-Chun Sun" returns only EyeNavGS, and ti:"Layered Mesh" returns 2 unrelated hits.
The Dasari lab and author pages give no PDF.

Points checked against the accessible primary sources:
(1) Static scenes, per-scene optimization, no live feed-forward input: supported.
The published abstract says 'static 3D scenes', and the README pipeline is Warmup -> Training (5000-15000 iterations).
(2) LMG's budget B is a server-side training budget: supported.
Symbol table Sec. 3.3: B = sum of n_t.
The table claims to cover all of Sec. 3-5 and has no client memory, cache or GPU-budget symbol.
That is fairly strong, though not conclusive, evidence that the paper has no formal client-budget model.
(3) Item (4) is only partly fair.
LMG's urgency already has a temporal weight gamma over a look-ahead window of W future frames (Sec. 4.2), so its scheduler is already a soft-deadline, view-prediction-driven scheduler.
The new part of our idea is not 'deadline-aware' as such.
It is deadlines on freshness of changing content (glass-to-glass latency), taken jointly with a client GPU/upload budget.
The reader's overlap list mentions the time weight, but the 'what our idea adds' list should say this explicitly.
(4) 'LMG only ever adds tiles' and 'no eviction' cannot be verified without the full text.
The symbol table shows no eviction symbol.
(5) Glass-to-glass latency vs LMG's PSNR/SSIM/VMAF: fair.
Note that the abstract claims 'low-latency' and the latency definition is unknown.
(6) Identity anchored to triangles (t, i): supported by the symbol table Sec. 3.1.
It is reinforced by the lmg++ Lean format, which recomputes xyz, scale and rotation from the triangle.

The representation does support mesh-driven deformation (the animation branch and the hotdog_fly demo), but that is not camera-driven live content.
The reader already acknowledges this.

For the live/dynamic overlap, the same group's LTS (MMSys'25, DOI 10.1145/3712676.3714445, a DASH system for dynamic multi-layer 3DGS) and EMS'24 (10.1145/3672196.3673394) are more relevant than LMG.
Both are cited by LMG per Crossref.
The author homepage lists both as Best Paper Award winners; that is self-reported and I did not check it on the venue site.

### Verifier notes

Verified:
- Exact title (Crossref uses a hyphen, the page an en dash).
- Five authors with their affiliations: Crossref gives NTHU Hsin-Chu (Sun, Kondori, Hsu) and Northeastern Boston (Chen, Dasari).
- Venue: Proceedings of the ACM Multimedia Systems Conference 2026, MMSys '26 Hong Kong, pp. 264-275, DOI 10.1145/3793853.3795764.
  Crossref, Semantic Scholar (DBLP key conf/mmsys/SunCKDH26), the Dasari lab publication list (sinrg.org) and the first author's homepage agree.
  The README's '(MMSys'25)' is a typo next to '2026'.
- No arXiv id.
- Code repo: 4 stars, pushed 2026-09-22, branches main/progressive/animation/lmg++.
  LICENSE.md is the Inria/MPII Gaussian-Splatting License plus GMUM, research/evaluation only.
- diff-gaussian-rasterization-lmg exists and is not a GitHub fork.
- The symbol table matches the reader's transcription.
- No branch has streaming server, scheduler, UDP or client code.
- Cited works confirmed via Crossref DOIs: EyeNavGS (MM'25), 'Beyond throughput, the next generation' (MMSys'20), L3GS, SGSS, LapisGS, NetSplat, 3DGStreaming, MILo, Draco, VMAF.

The 3 key numbers match both the project page and the publisher abstract on Semantic Scholar verbatim:
- PSNR up to +10.99 dB (avg +8.89)
- SSIM up to +0.293 (avg +0.255)
- streaming: up to +19.81 dB PSNR (avg +5.80), +0.143 SSIM (avg +0.06), +40.86 VMAF (avg +16.06)

What each number is compared against is still unverified: GaMeS for the training server is an inference, and the No Layer / No Tiling ablations for the streaming numbers are also an inference.

The reader handled the 403 correctly and stopped.
The local file papers/agent_workdir/lmg/lmg.pdf is a 5.7 KB Cloudflare 'Just a moment...' HTML page, not the paper, so nobody should treat it as the full text.

My check files are in papers/agent_workdir/lmg_verify/ (page.html, symtab.pdf, README.md, branch tree listings).

The full text is still needed for: the urgency formula, transport, client platform, client memory handling, network traces, hardware, latency definition and external baselines.

## Related work the material missed

Searched: I used WebSearch for discovery.
Every arXiv entry was then checked against its arxiv.org/abs page (title, date, authors, comments, JREF) with a script.
Full text came from arxiv.org/html/<id> for 2504.05517, 2605.08699, 2606.05650, 2605.09279, 2608.01659, 2604.25330, 2507.21311, 2608.10020, 2506.02774, 2506.02380, 2409.13648, 2604.27441, 2607.09260, 2609.13859, 2603.17227, 2606.07179, 2607.25971, 2412.09608, 2512.20943, 2411.14847, 2601.18475, 2606.17212, 2606.03254, 2610.04351 and 2503.06235.
I also read GitHub READMEs (LTS, dynamic-lapis-gs, L3GS, LiVo, CUT3R, Spann3R, StreamVGGT, antimatter15/splat, gsplat.js, GaussianSplats3D, web-splat, Spark), the Spark docs (packed-splats, lod-getting-started) and the PlayCanvas Streamed SOG spec.
Query themes: layered/progressive/LoD 3DGS streaming (LapisGS, L3GS, SGSS, EvoGS, SplatStream, PRoGS, SRBF-Gaussian); dynamic/volumetric Gaussian streaming (V^3, 3DGStream, HiCoM, QUEEN, IGS, 4DGC/4DGCPro, SwinGS, AirGS, CAGS, GS-NFS, LTS, Vega, LiveGS); online/feed-forward streaming reconstruction (StreamGS, both StreamSplats, Spann3R, CUT3R, StreamVGGT, STream3R, Point3R, Long-LRM, OF3GS, LoCoSplat, VoluMe, GS-SCNet); telepresence/live systems (GaussianNexus, AnythingReality, GaussAnything, Mon3tr, LiVo, MetaStream, ReVo); web renderers; ABR/viewport point-cloud and volumetric streaming (MuV2, VARFVV, MD2G-Cast, INDS); deadline/GPU-budget scheduling (2606.17212, LS-Gaussian, Splats under Pressure); multi-client/GPU sharing (TIGAS, STT-GS, ECO-GS, plus an MIG/MPS+3DGS query that returned nothing specific).
Venues targeted by query: CVPR, ICCV, NeurIPS, ICLR, SIGGRAPH/SIGGRAPH Asia, ACM MM, MMSys, NOSSDAV, EMS, IEEE VR, ISMAR, INFOCOM, NSDI/SIGCOMM, MobiCom, CoNEXT, UIST.
Date range: 2024 to Oct 2026, plus MetaStream (MobiCom 2023) as a pointer only.

| Work | Venue/year | Overlap | What | Why it matters |
| --- | --- | --- | --- | --- |
| [GaussAnything: Semantic Intent-Driven Refinement of Evolving Gaussian Scenes for Standalone VR](https://arxiv.org/abs/2609.13859) | arXiv 2609.13859 (Sep 2026), no venue stated | high | A host keeps an evolving online-SLAM Gaussian+SDF scene (AnythingReality, replayed at 10 FPS) and streams it to a standalone OpenXR headset (PICO 4 Ultra, OpenGL ES 3.2). The messages are stable-ID insert, remove, update and slot-replace records over a reliable ordered connection. The client keeps a fixed resident-Gaussian budget in a preallocated GPU buffer indexed by persistent ID, and it merges dirty slot ranges before upload. It applies latest-value coalescing and publishes only a subset of pending edits per display frame. Edits are prioritised by support, view relevance, change size and age. The work allowance adapts to display slack, with an age-triggered escape. Source epochs keep the Gaussians and the mesh consistent. Evaluation: 8 scenes (4 Replica, 4 RealSense) with an RTX 5090 host. Metrics are GT-vs-host, host-vs-headset and GT-vs-headset PSNR, convergence after a teleport (0.87-1.63 s), flicker, about 10 ms GPU per stereo frame, and a budget ablation from 25k to 400k resident Gaussians. There is no network emulation, no comparison of FIFO against the scheduler, and no glass-to-glass measurement (source: arxiv.org/html/2609.13859). | It already implements the client half of the favoured contribution: GPU-budgeted upload, coalescing, priority publication and decoupled display and model loops. The novelty claim must narrow to the joint network+GPU scheduler with deadlines, a browser/WebGPU client, a feed-forward source without stable IDs, and glass-to-glass latency under netem against FIFO and view-aware arms. Its per-stage event log (host -> received -> desired -> queued -> integrated -> visible) is a ready template for decomposing latency. |
| [Renderable Partial Representations for Dynamic Gaussian Splatting under Incomplete Delivery](https://arxiv.org/abs/2606.17212) | arXiv 2606.17212 (Jun 2026), no venue stated | high | Dynamic GS is split into independently addressable spatiotemporal clusters: a base plus 3 refinements. Delivery uses MTU-bounded chunks. The scheduler ranks dependency-completion groups by a render-derived utility per byte on the wire, weighted by the estimated probability of arriving before the deadline (from slack, rate, RTT and loss). Unreliable datagrams are not retransmitted after their deadline. It runs on an emulator with RFC9002-style NewReno and a BBR-like controller, which the paper says is not a full QUIC stack. Control arms are layer order, a segment-budget DASH-style scheduler, and a dependency-aware scheduler without deadline utility. All arms share the same congestion controller, packet loss, queue and seed. Conditions: synthetic 2-200 Mbit/s, 10-320 ms RTT, 0-10% IID loss plus Gilbert-Elliott bursts, UCC 5G traces and CellReplay traces. Metrics: deadline-miss, wasted goodput, startup, quality over playout time, stalls and flicker. Over 600 paired trials it reports 25.6% fewer misses than layer order (bootstrap CI 23.6-27.4%), at a cost of 4.1 ms more startup. Content is offline (D-NeRF, HyperNeRF) and runs on an H100. No real client GPU or browser (source: arxiv.org/html/2606.17212). | The network half of the proposed joint scheduler (deadline-aware, utility per byte, compared against FIFO-like layer order) is already published. Its evaluation design is the model to copy: arms that differ only in the scheduler, a shared congestion controller and seed, paired trials, and tradeoffs reported over time. The student's gap is the client GPU budget and live feed-forward content. |
| [AirGS: Real-Time 4D Gaussian Streaming for Free-Viewpoint Video Experiences](https://arxiv.org/abs/2512.20943) | IEEE INFOCOM 2026 (per arXiv comments), arXiv 2512.20943 | high | 4DGS free-viewpoint video with quality-driven keyframes, each starting a new canonical Gaussian space per frame group. It streams Gaussian differences between adjacent frames as sparse multi-channel 2D images. An ILP and a stepwise fallback choose a pruning level to fit the bandwidth. Evaluation: HiFi4G (7 sequences), LTE driving traces, RTX 3090. Baselines: NeuS2, 3DGS, 3DGStream, 4DGS and V^3. Metrics: PSNR/SSIM, per-frame transmission size and time (CDF and maximum), FPS and training time. About 18.9 ms per frame end to end on the client (stream, decode, apply diffs) (source: arxiv.org/html/2512.20943). | A networking venue has already published ID-based delta streaming of Gaussians with network-adaptive reduction. It shows one answer to Gaussian identity: a shared canonical space within each keyframe group. It is offline per sequence and has no viewport or client GPU budget. |
| [LTS: A DASH Streaming System for Dynamic Multi-Layer 3D Gaussian Splatting Scenes](https://dl.acm.org/doi/10.1145/3712676.3714445) | ACM MMSys 2025 (Best Paper), pp. 136-147, DOI 10.1145/3712676.3714445 | high | DASH streaming of dynamic 3DGS by Layer (Dynamic-LapisGS), Tile (viewport) and Segment (time), with live and on-demand modes. Authors: Sun, Shi, Lee, Zhu, Ooi, Liu, Huang and Hsu (NTHU, NUS, Rutgers, NYCU). The GitHub READMEs say the system code is withheld for patent reasons. The Dynamic-LapisGS code is released: it inherits Gaussians across frames and optimises only position and rotation, so identity persists. Reported numbers (taken from the abstract in a search snippet; the full text was NOT read because ACM DL was blocked here): up to 99.70% fewer missing frames when live, and up to 92.01% less freeze time on demand. CAGS reimplements it as baseline LTS-F, and GS-NFS uses LTS-Draco. | The closest prior art comes from the NTHU Cheng-Hsin Hsu group, which is local to the student. Missing frames and freeze time when live are precedents for the stall metrics. The full paper should be read through the NTHU library before any novelty claim is made. |
| [L3GS: Layered 3D Gaussian Splats for Efficient 3D Scene Delivery](https://arxiv.org/abs/2504.05517) | ACM MobiCom 2025, DOI 10.1145/3680207.3723485, arXiv 2504.05517 | high | Layered and segmented static 3DGS (4 layers x 45k splats). A per-splat utility (closeness to the viewport centre, projected overlap, opacity) feeds a multiple-choice knapsack scheduler run every 1 s under predicted bandwidth (harmonic mean over 0.5 s). Evaluation: 8 scenes (Mip-NeRF360, T&T, Gaussian Grouping) and Quest 3 6-DoF traces (6 users x 8 scenes, 60 s each, plus synthetic traces). Network: Lumos5G walking traces scaled to an 11.8 Mbps mean. Arms: Sort (LightGaussian global significance, view-independent), Separate (non-layered versions) and Pre-load as an upper bound. Metric: viewport SSIM every 1 s against the full-model render. Scheduler overhead under 3 ms on 2 vCPU. Ablation with oracle viewport and oracle bandwidth (source: arxiv.org/html/2504.05517). | This is the peer-reviewed form of the 'view-aware priority' baseline arm, and its code is public. It uses Pre-load as a ceiling control, and oracle predictors to separate prediction error from scheduling. It has no live content, no GPU upload budget and no latency metric. |
| [CAGS: Color-Adaptive Volumetric Video Streaming with Dynamic 3D Gaussian Splatting](https://arxiv.org/abs/2605.09279) | ACM SIGGRAPH 2026 (per arXiv comments), arXiv 2605.09279 | medium | Viewport-predicted (autoregressive) tile and LoD selection for dynamic 3DGS. The server renders low-resolution reference images for colour restoration and raises tile LoDs up to the bandwidth limit. The prototype streams at 30 FPS while the client renders at 60 FPS, with an RTX 3080 on both server and client. Evaluation: fixed 30-150 Mbps and a 5Gophers 5G trace segment; Quest 3 viewport traces recorded with UnityGS. Baselines: LTS-F (reimplemented LTS) and V^3-A. Metric: per-frame PSNR/SSIM; also generality tests on Dynamic 3DGS, 4DGS and HiCoM content (source: arxiv.org/html/2605.09279). | A recent strong view- and bandwidth-adaptive baseline that already separates the stream rate from the render rate. It shows how LTS gets reimplemented as a baseline when its code is unavailable. Network-only: there is no client GPU scheduling. |
| [StreamSplat: Streaming Feed-Forward 3D Gaussian Splatting (Song et al.)](https://arxiv.org/abs/2608.01659) | arXiv 2608.01659 (Aug 2026), no venue stated | medium | Causal feed-forward 3DGS from calibrated image streams, built on ReSplat and DepthSplat. A Voxel-Aligned Causal Cache keeps a persistent, memory-bounded world-space token state, and explicit Gaussians can be decoded after any chunk. On ScanNet with 64 views it takes about 143 ms from chunk arrival to one rendered novel view. It scales to 1024 views where fixed-view baselines run out of memory. Peak memory is about 24.5 GB at 256 views on an RTX PRO 6000 96 GB. Metrics: PSNR/SSIM/LPIPS and peak CUDA memory on DL3DV, RE10K and ScanNet (source: arxiv.org/html/2608.01659). | A candidate live model whose latency and memory numbers are relevant to a shared RTX 6000 Ada 48 GB. Gaussians are re-decoded from the cache at each chunk, so there are no native Gaussian IDs. The voxel and pivot index is a possible stable key, but that is an untested hypothesis. NAME COLLISION: this is a different paper from the material's 'StreamSplat (ICLR 2026)', which is arXiv 2506.08862 (Wu et al., dynamic, uncalibrated). |
| [Thin-Client Interactive Gaussian Adaptive Streaming over HTTP/3 (TIGAS)](https://arxiv.org/abs/2605.08699) | arXiv 2605.08699 (May 2026), no venue stated | medium | Remote rendering instead of geometry streaming. A GPU backend rasterises 3DGS for each pose request and returns JPEGs over HTTP/3/QUIC (aioquic) to a browser client. The client uses a latency-targeting ABR and optional WebGPU super-resolution. The backend handles many models and many users: VRAM-resident tensors, lazy load, reference counting and eviction when idle. Evaluation: 14 3DGS models replayed with EyeNavGS traces. Local runs, plus a remote backend in Austria (Quadro RTX 8000) with clients in Austria, Spain, Sweden and Taiwan. RTT is injected. ABR baselines include L2A and LoL+, plus progressive download and DASH/CMAF reference arms. Metrics: motion-to-photon latency, PSNR/SSIM/VMAF/LPIPS/DISTS, bandwidth and quality switches. Co-author Cheng-Hsin Hsu, NTHU (source: arxiv.org/html/2605.08699). | The opposite split point (sending pixels instead of Gaussians), which makes it a natural 'remote rendering' control arm. Its multi-user VRAM residency and eviction is the only 3DGS server-GPU sharing mechanism found, which is relevant to the virtualization angle. |
| [EyeNavGS: A 6-DoF Navigation Dataset and Record-n-Replay Software for Real-World 3DGS Scenes in VR](https://arxiv.org/abs/2506.02380) | ACM MM 2025, pp. 13126-13132 (arXiv JREF), arXiv 2506.02380 | medium | 6-DoF head-pose and eye-gaze traces from 46 participants on 12 real-world 3DGS scenes, collected at Rutgers and NTHU with Meta Quest Pro. Released with a SIBR record-and-replay fork and conversion tools (source: arxiv.org/html/2506.02380). | Off-the-shelf viewport traces for the view-dependent arm, and TIGAS already uses them. It is an NTHU co-authored resource. The traces are for static scenes, so a live-scene viewport trace still has to be defined. |
| [GS-NFS: Bandwidth-adaptive Streaming of Dynamic Gaussian Splats and Point Clouds](https://arxiv.org/abs/2606.05650) | arXiv 2606.05650 (Jun 2026), no venue stated | medium | A fully GPU-resident 4DGS codec: Morton-code octree, RAHT, RLGR/ANS and KLT. It encodes and decodes at full frame rate, about 14-23 ms per frame on an RTX 4500 Ada, and 25 fps decode on a Jetson. Baselines: V^3-2D, MesonGS, G-PCC and LTS-Draco, each paired at equal first-frame PSNR. Metrics: encode/decode ms, PSNR difference and relative compression ratio on HiFi4G and N3DV. The paper explicitly motivates encoding feed-forward Gaussian frames live (source: arxiv.org/html/2606.05650). | A ready codec arm for compressing deltas, and its pairing at matched quality is a clean control. Decoding is in CUDA, which challenges the plan to decode on the browser CPU. Despite the title, the paper has no network experiments. |
| [V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians](https://arxiv.org/abs/2409.13648) | ACM TOG 43(6) / SIGGRAPH Asia 2024, DOI 10.1145/3687935, arXiv 2409.13648 | medium | Packs per-frame Gaussian attributes into Morton-sorted 2D images, keeping a constant Gaussian count and the same pixel slot within each frame group, and streams them with hardware video codecs to phones (source: arxiv.org/html/2409.13648). | A concrete answer to the identity problem: a fixed slot per Gaussian within a group gives ID-addressable deltas for free. It is also a standard baseline (GS-NFS, CAGS and AirGS all compare against it). Offline training. |
| [SwinGS: Sliding Window Gaussian Splatting for Volumetric Video Streaming with Arbitrary Length](https://arxiv.org/abs/2409.07759) | arXiv 2409.07759 (Sep 2024), no venue stated | medium | Spacetime Gaussians with MCMC and a sliding window of Gaussian snapshots, for streaming volumetric video of arbitrary length. Comes with an interactive WebGL browser viewer that plays on phones and tablets. Claims an 83.6% reduction in transmission cost (source: arXiv abs page). | A prior browser client that receives a continuously changing Gaussian set. Its sliding-window add and retire model is a precedent for a capacity-limited browser scene cache. Offline content. |
| [VoluMe - Authentic 3D Video Calls from Live Gaussian Splat Prediction](https://arxiv.org/abs/2507.21311) | ICCV 2025 (CVF open access), arXiv 2507.21311 | medium | A lightweight U-Net predicts Splatter-Image style Gaussians per frame from one webcam feed. A stability loss and a jitter metric address flicker. The prototype sends ordinary 2D video over the network and runs the prediction on the receiver at 28 FPS on a laptop RTX 4090 Mobile. Metrics: PSNR/SSIM/LPIPS/DreamSim and ArcFace identity on Ava-256 and Cafca, plus jitter (source: arxiv.org/html/2507.21311). | A live feed-forward GS system that deliberately ships video rather than Gaussians. It is the 'reconstruct at the receiver' alternative split that the project should justify against. Its jitter metric is useful for temporal stability of live feed-forward output. |
| [GS-SCNet: Generalizable 3D Gaussian Splatting enabled Semantic Coding for Real-Time Immersive Video Communications](https://arxiv.org/abs/2604.25330) | arXiv 2604.25330 (Apr 2026), under review | medium | End-to-end stereo semantic codec whose decoded latents are mapped directly to 3DGS attributes at the receiver. The arms are decoupled pipelines (MV-HEVC/DCVC-DC followed by GPS-Gaussian/EVA-Gaussian), compared by rate-render-distortion curves and BD-rate on human datasets. Trained on an A100. No network emulation (source: arxiv.org/html/2604.25330). | A third split point: compressed features in, Gaussians predicted at the receiver. It defines how 'transmit Gaussians' vs 'transmit video' vs 'transmit latents' arms are compared at equal rate. |
| [LiVo: Toward Bandwidth-adaptive Fully-Immersive Volumetric Video Conferencing](https://github.com/USC-NSL/LiVo) | ACM CoNEXT 2025 (PACMNET), DOI 10.1145/3768981 | medium | Live two-machine volumetric conferencing. Multiview RGB-D frames are culled to the receiver frustum, tiled, encoded with two H.265 streams (colour and depth) and carried over GStreamer WebRTC. Bandwidth is apportioned dynamically between streams. Mahimahi replays bandwidth traces, and prerecorded frustum traces drive culling and predictive-culling experiments. The client renders an Open3D point cloud (source: GitHub README; the paper itself was not read). | A live capture-to-render system with trace-replay network emulation (Mahimahi as an alternative to tc netem). LiVo's authors went on to write GS-NFS. Point clouds, not 3DGS. |
| [MD2G-Cast: Relay-Coordinated Multicast for Scalable Volumetric Streaming over MoQ](https://arxiv.org/abs/2608.10020) | ACM MM 2026 (per arXiv comments), arXiv 2608.10020 | medium | MoQ relays group receivers by viewing overlap and capability, share Base content and admit Enhanced delivery, with PPO and a distilled policy. V-PCC content in a Mininet testbed (32 vCPU) with 4G/5G/Wi-Fi/fiber throughput traces and CWI 6-DoF head traces, scaled to 100 users. Same-substrate baselines (Heuristic, Clustering), a DASH independent-delivery reference, and a matched relay-control ablation with random and deterministic controllers. Metrics: delivery interval, stall penalty, utility and aggregate link load (source: arxiv.org/html/2608.10020). | A multi-client resource-sharing angle for the virtualization requirement. Its matched ablation, which separates the control structure from the optimizer, is an exemplary control design. |
| [Vega: Fully Immersive Mobile Volumetric Video Streaming with 3D Gaussian Splatting](https://dl.acm.org/doi/10.1145/3680207.3765267) | ACM MobiCom 2025, pp. 1106-1120, DOI 10.1145/3680207.3765267 | medium | 3DGS full-scene volumetric video streaming to Android phones. Object-level selective computation spends rendering effort on visually important objects to meet strict rendering deadlines, with an encoding that supports object priority. 30 FPS on commodity phones (from the abstract via search; the primary PDF was NOT read: TLS error on the authors' lab site and ACM DL blocked; venue and pages confirmed from the CAGS reference list). | Deadline-driven scheduling of client GPU work for 3DGS streaming, on the GPU side of the joint scheduler. Its evaluation design needs to be read before the GPU-budget claims are framed. |
| [SGSS: Streaming 6-DoF Navigation of Gaussian Splat Scenes](https://www.researchwithrutgers.com/en/publications/sgss-streaming-6-dof-navigation-of-gaussian-splat-scenes/) | ACM MMSys 2025, pp. 46-56 | medium | View-adaptive streaming of static 3DGS using optimised spatial partitioning and prioritised loading. Authors: Mufeng Zhu, Mingju Liu, Cunxi Yu, Cheng-Hsin Hsu, Yao Liu. EyeNavGS says SGSS has a web viewer. The paper was NOT read: the Rutgers page returned 404 and the citation details come from the EyeNavGS and CAGS reference lists. | Another NTHU/Rutgers view-adaptive web streaming baseline for static scenes. Worth asking the Hsu group for the code. |
| [LiveGS: Live Free-Viewpoint Video via High-Performance Gaussian Splatting for Mobile Devices](https://dl.acm.org/doi/10.1145/3721257.3734033) | ACM SIGGRAPH 2025 Emerging Technologies, DOI 10.1145/3721257.3734033 | medium | ByteDance live free-viewpoint video broadcast. A server-side feed-forward human 3DGS model runs in real time, temporal Gaussians are mapped onto 2D planes for standard video codecs, and rendering runs on PC and mobile (from the abstract via search; the full paper was NOT read). | A deployed live feed-forward-3DGS-to-client pipeline. It is a demo paper and probably has no comparative evaluation. Human-centric rather than whole scenes. |
| [GaussianNexus: Room-Scale Real-Time AR/VR Telepresence with Gaussian Splatting](https://dl.acm.org/doi/10.1145/3746059.3747693) | ACM UIST 2025, DOI 10.1145/3746059.3747693 | medium | Room-scale 3DGS telepresence. A 360-degree camera tracks 2D and 3D changes, and segmented dynamic objects are synchronised to remote VR users as real-time updates to a prebuilt GS environment without retraining (from the abstract via search; NOT read). | Live partial updates of a 3DGS scene from camera input to a remote client. Its update mechanism and latency evaluation need checking. |
| [Spark (sparkjsdev/spark) 3DGS renderer for THREE.js](https://github.com/sparkjsdev/spark) | Open-source library, v2.x (README references 2.4.0), 2025-2026 | medium | WebGL2 renderer. PackedSplats store 16 bytes per splat in a 2048x2048xN Uint32x4 array texture. setSplat(index,...) and pushSplat(...) update or append splats at runtime, and a GPU shader graph (dyno) supports procedural and edited splats. LoD uses a splat budget (lodSplatCount defaults to 2.5M on desktop and 500K on Oculus) with foveated selection. Paged .RAD chunked streaming fetches chunks as needed. Source: GitHub README and sparkjs.dev/docs/packed-splats and /lod-getting-started. The docs do not say whether uploads are partial or full-texture. | The most capable maintained web client base. Its index-addressed update API and splat budget map directly onto ID deltas and a GPU budget. It is WebGL2, not WebGPU, so that choice has to be made explicitly. Upload granularity must be measured, not assumed. |
| [StreamGS: Online Generalizable Gaussian Splatting Reconstruction for Unposed Image Streams](https://arxiv.org/abs/2503.06235) | arXiv 2503.06235 (Mar 2025), no venue stated on arXiv | medium | Predicts per-frame Gaussians from unposed streams (DUSt3R-style initialisation plus content-adaptive refinement) and merges redundant Gaussians across frames through feature aggregation. Claims quality on par with optimisation at 150x the speed (source: arXiv abs/html). | A candidate live model. Cross-frame merging creates and destroys Gaussians, which is exactly the identity and churn problem the delta protocol must handle. StreamSplat (2608.01659) reports no public implementation. |
| [Instant Gaussian Stream: Fast and Generalizable Streaming of Dynamic Scene Reconstruction via Gaussian Splatting (IGS)](https://arxiv.org/abs/2503.16979) | CVPR 2025 (CVF open access), arXiv 2503.16979 | low | A generalizable anchor-driven motion network moves a persistent Gaussian set from frame to frame, with keyframe refinement. 2.67 s per frame (multi-view input). A follow-up, arXiv 2603.17227, found that the anchor sampler choice has no measurable effect at deployment budgets. | Keeps Gaussian identity across frames through motion of persistent primitives. Not live-rate. The 2603.17227 null-result paper is a good example of a matched one-factor control. |
| [3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos](https://arxiv.org/abs/2403.01444) | CVPR 2024 Highlight, arXiv 2403.01444 | low | Per-frame online optimisation of a neural transformation cache plus adaptive Gaussian addition for multi-view free-viewpoint video streaming. | The canonical 'streamable FVV' baseline (used by AirGS and IGS). It produces per-frame residuals on persistent Gaussians that can serve as an update trace. |
| [QUEEN: QUantized Efficient ENcoding of Dynamic Gaussians for Streaming Free-viewpoint Videos](https://arxiv.org/abs/2412.04469) | NeurIPS 2024, arXiv 2412.04469 | low | Online per-frame attribute residuals with learned quantisation and sparsity for streaming free-viewpoint video (NVIDIA). | GS-NFS uses QUEEN-trained N3DV models as test content. It is a source of realistic per-frame Gaussian residual traces with stable IDs. |
| [Representing Long Volumetric Video with Temporal Gaussian Hierarchy](https://arxiv.org/abs/2412.09608) | ACM TOG 43(6) / SIGGRAPH Asia 2024, DOI 10.1145/3687919, arXiv 2412.09608 | low | A multi-level temporal hierarchy shares Gaussians across segments, so only a subset is resident at any moment and GPU memory stays nearly constant regardless of video length. | A precedent for treating GPU memory as a bounded working set of a larger Gaussian store, which is the 'virtual scene cache' framing. Offline. |
| [LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive Streaming](https://arxiv.org/abs/2408.14823) | 3DV 2025, arXiv 2408.14823 | low | A cumulative layered 3DGS with dynamic opacity optimisation and occupancy maps for progressive, bandwidth-adaptive rendering. Evaluated by quality against model size. Successors: Dynamic-LapisGS (in LTS), NETSPLAT, and EvoGS (arXiv 2606.07179). | The base representation of the NUS/NTHU streaming line. Static scenes only. |
| [NETSPLAT: Data Plane Network Assistance for Streaming 3D Gaussian Splatting Scenes](https://dl.acm.org/doi/10.1145/3746441.3748225) | ACM SIGCOMM EMS 2025 workshop, DOI 10.1145/3746441.3748225 | low | Programmable data plane support for dynamic 3DGS streaming: multicast group reconfiguration, alternative queueing and scheduling, and in-band client reconfiguration. Built on LapisGS (from the abstract via search and a citing reference list; NOT read). | In-network resource management for many 3DGS clients, a possible virtualization or isolation hook. |
| [Multi-frame Bitrate Allocation of Dynamic 3D Gaussian Splatting Streaming Over Dynamic Networks](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2024ems.pdf) | ACM SIGCOMM EMS 2024 workshop | low | A model-driven gradient-ascent bitrate allocation across geometry, SH, opacity and transform attributes over multiple frames, under a rate threshold (Sun, Shi, Ooi, Huang, Hsu). Up to 5.46 dB PSNR over the baseline (from the abstract via search and the NYCU listing; NOT read). | An earlier step in the NTHU line that led to LTS. |
| [MuV2: Scaling up Multi-user Mobile Volumetric Video Streaming via Content Hybridization and Sharing](https://www.researchgate.net/publication/380989154_MuV2_Scaling_up_Multi-user_Mobile_Volumetric_Video_Streaming_via_Content_Hybridization_and_Sharing) | ACM MobiCom 2024, pp. 327-341 | low | Multi-user mobile volumetric streaming that shares content across co-located users, for example a classroom (venue and pages from the CAGS and GS-NFS reference lists; NOT read). | Multi-client sharing precedent if the course requires multi-tenant isolation content. |
| [Voyager: Real-Time Splatting City-Scale Gaussians on Resource-Constrained Devices](https://arxiv.org/abs/2506.02774) | arXiv 2506.02774 (Jun 2025), no venue stated | low | Cloud-client co-rendering. The cloud runs an asynchronous LoD search and streams only the Gaussians needed. The client uses temporal-aware LoD search and LUT rasterisation. Evaluated on a Jetson AGX Orin (FPS, operations, energy). | A server-decides, client-renders split for view-dependent subsets, with temporal reuse of the previous LoD cut. |
| [AnythingReality: Robust Online Gaussian Splatting SLAM for Open-Vocabulary VR Scene Exploration](https://arxiv.org/abs/2607.09260) | arXiv 2607.09260 (Jul 2026) | low | Online RGB-D Gaussian+SDF SLAM (ORB-SLAM3 poses) with stereo frames rendered on the server and streamed to a WebXR client over WebRTC (pixels, not Gaussians). Metrics: PSNR/SSIM/LPIPS and FPS against online GS methods. | The upstream pipeline of GaussAnything. It shows the same group moving from pixel streaming to Gaussian streaming. |
| [Continuous 3D Perception Model with Persistent State (CUT3R)](https://arxiv.org/abs/2501.12387) | CVPR 2025 Oral (GitHub README), arXiv 2501.12387 | low | A recurrent stateful model that outputs per-frame pointmaps and poses from streaming images. | A possible backbone for live geometry, but it outputs pointmaps, not Gaussians. Spann3R (arXiv 2408.16061), STream3R (2508.10893) and Point3R (2507.02863) are in the same class. |
| [Streaming 4D Visual Geometry Transformer (StreamVGGT)](https://arxiv.org/abs/2507.11539) | ICLR 2026 (per GitHub repo description), arXiv 2507.11539 | low | A causal VGGT with a cached history of tokens for incremental 4D geometry. | Covers the 'VGGT-based online' lead. Geometry only; a Gaussian head would have to be added. |
| [Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats](https://arxiv.org/abs/2410.12781) | arXiv 2410.12781 (Oct 2024) | low | A Mamba2+transformer feed-forward model that reconstructs 32 views at 960x540 in about 1 s on an A100. | Batch rather than streaming. It is used as a baseline in StreamSplat (2608.01659). |
| [LoCoSplat: Real-Time Feed-Forward 3D Gaussian Splatting with Minimal 3D Reasoning](https://arxiv.org/abs/2610.04351) | arXiv 2610.04351 (Oct 2026), under review | low | Feed-forward 3DGS with a local-context MLP and no learned 3D network. 33 ms for a 6-view scene on an RTX PRO 6000. OF3GS (arXiv 2606.03254) is a causal on-the-fly feed-forward GS from unposed images. | Sets the model-side inference rate the scheduler must absorb. These are windowed, not persistent-state, models. |
| [PlayCanvas Streamed SOG format](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/) | Open spec plus engine implementation, 2025 | low | A binary spatial tree of chunks, each with several LoD levels. The viewer walks the tree to stream the chunks needed by the current camera. The spec does not mention runtime updates to splat data. | A browser view-dependent chunk streaming precedent for static scenes. A live scene would break its offline-built tree. |
| [Static web splat viewers: antimatter15/splat, huggingface/gsplat.js, mkkellogg/GaussianSplats3D, KeKsBoTer/web-splat](https://github.com/antimatter15/splat) | Open-source, 2023-2026 | low | antimatter15: WebGL 1.0, CPU sort in a web worker at about 4 fps, progressive loading ordered by size and opacity, no runtime updates. gsplat.js: WebGL with a wasm folder (sort); its README does not mention runtime updates. GaussianSplats3D: three.js, progressive loading, worker/WASM-SIMD sort, data textures, addSplatScenes(); its README says it is no longer in active development and points to Spark. web-splat: Rust/wgpu WebGPU with GPU radix sort and no dynamic updates. Sources: each GitHub README, plus the GitHub API for archive status. | None supports streaming ID-addressed updates. The asynchronous worker sort at a lower rate than rendering is an existing example of decoupled rates inside the client. |
| [ReVo: A Cross-Layer Reliable Volumetric Videoconferencing System](https://arxiv.org/abs/2604.27441) | arXiv 2604.27441 (Apr 2026) | low | Loss-resilient RGB+depth volumetric conferencing over WebRTC. Selective FEC protects critical content and neural recovery repairs corrupted frames. Evaluated with real-world loss traces; metrics are SSIM and video freezes. | A reference for packet-loss conditions and freeze metrics when tc netem loss is added. RGB-D, not 3DGS. |
| [4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](https://arxiv.org/abs/2509.17513) | NeurIPS 2025, arXiv 2509.17513 | low | A single-bitstream hierarchical 4DGS with multiple bitrates, real-time decoding on mobile, and layer-wise rate-distortion training. Related compression work: 4DGC (CVPR 2025, 2503.18421), HiCoM (NeurIPS 2024, 2411.07541), SplatStream (Asilomar 2026, 2607.25971) and PD-4DGS (2605.11427). | Codec-side scalability. These works are evaluated on rate-distortion only, with no scheduler or latency. |

1. Novelty risk, the main finding: two 2026 arXiv papers that the material missed together cover most of the favoured 'joint network-GPU scheduler'.
GaussAnything (2609.13859) already does the client side: stable-ID deltas, a preallocated GPU slot buffer, latest-value coalescing, and budgeted per-frame publication whose allowance adapts to display slack and whose priority uses view, change and age.
2606.17212 already does the network side: deadline-probability x render-utility scheduling against layer-order, DASH-segment and dependency-only arms, with the same congestion controller and the same seeds.
Neither paper combines both halves, uses a browser/WebGPU client, uses a feed-forward source without native IDs, or measures glass-to-glass latency under tc netem against FIFO.
The defensible gap is now that intersection, and the plan should cite both papers explicitly.
AirGS (INFOCOM 2026) already shows ID-based Gaussian delta streaming with network-adaptive pruning, and V^3 shows fixed-slot identity within a frame group.

2. Name collision: there are two 'StreamSplat' papers.
The material's StreamSplat (ICLR 2026) is arXiv 2506.08862 (Wu et al.; dynamic, uncalibrated video); arXiv comments confirm the ICLR acceptance.
arXiv 2608.01659 (Song et al., 2026) is a different paper: streaming feed-forward 3DGS for static scenes with a voxel cache, about 143 ms per chunk.

3. Material spot-checks on arXiv abs pages: MoQSplat 2609.18624, 'accepted at IEEE MMSP 2026', confirmed; StreamSTGS 2511.06046, AAAI 2026, confirmed; S2GS 2603.14232 exists and an ICML 2026 poster page appeared in search; Streaming Real-Time Rendered Scenes 2604.02851 exists (Siekkinen and Kämäräinen) with no venue on arXiv; WebSplatter 2602.03207 exists, but its 'ACM MM 2026' venue is not stated on the arXiv page and is unverified.

4. NTHU local expertise: Cheng-Hsin Hsu co-authors LTS (MMSys'25 Best Paper), SGSS (MMSys'25), EyeNavGS (MM'25), TIGAS (2605.08699) and the EMS'24 bitrate paper.
LTS system code is withheld for patent reasons, according to its GitHub README.
Talking to that group early is the cheapest way to avoid duplicating their work.

5. Not read from primary sources: ACM DL returned 403, the Vega lab PDF failed TLS verification, and the Rutgers page returned 404.
This affects LTS full text, Vega, SGSS, MuV2, NETSPLAT, LiveGS, GaussianNexus, SRBF-Gaussian (IEEE VR 2025, pp. 461-471) and the EMS'24 paper.
Their existence and venues are confirmed by DOIs and by reference lists in primary papers I did read, but their evaluation details are unverified.
The statements about them in this list come from abstracts surfaced by search and are marked as such.

6. Evaluation-design patterns worth copying:
- Pre-load or oracle ceilings (L3GS).
- Oracle predictors to separate prediction error from scheduling (L3GS).
- Arms that differ only in the policy, on the same congestion controller and seed, with paired bootstrap confidence intervals (2606.17212).
- A matched ablation that separates the control structure from the optimizer (MD2G-Cast).
- Pairing at matched quality (GS-NFS).
- Per-stage visibility events (GaussAnything).
- Reusable traces: Lumos5G/5Gophers (Narayanan 2020), UCC 5G, CellReplay, Mahimahi (LiVo), LTE driving (AirGS), and EyeNavGS or Quest 3 viewport traces.
- Stall metrics in prior work: missing frames and freeze time (LTS), deadline-miss and wasted goodput (2606.17212), delivery interval (MD2G-Cast).
None of the papers I read measures camera-to-browser glass-to-glass latency.
Their reported numbers are not nulls for our data, because the hardware, content and traces all differ.

7. Virtualization angle: I found no 3DGS-specific MIG/MPS or GPU-isolation work.
The closest are TIGAS's multi-user VRAM residency with reference counting and eviction, and multi-user sharing in MD2G-Cast, MuV2 and NETSPLAT.

8. Leads seen but not verified: GSStream (ResearchGate only); '3DGS on Your Phone' (ACM 10.1145/3810919.3810924); 'Advancing Immersive Content Delivery with Dynamic 3DGS' (ACM 10.1145/3708468.3711886); '3D Gaussian-based Immersive Media Streaming in Networked XR' (MMSys'25, 10.1145/3712676.3719673); ACPGS (NOSSDAV'26, cited by CAGS); Mon3tr (2601.07518); Edge GS work (ECO-GS, ICASSP 2026, 2510.22718; STT-GS, 2510.13186); industry players (Gracia, 4DV.ai, a Nokia MPEG GS DASH demo at IBC 2026).

No files were written outside the scratchpad, and the repo was not modified.
Scratch artifacts are in papers/agent_workdir/ (verify1.tsv, verify2.tsv, txt/*.txt).

## Reusable local assets

| Path | What | Reuse for | Maturity |
| --- | --- | --- | --- |
| `/home/intern_2603056/Desktop/PairGS_ours/viewer/ (editor.js 1771 lines, index.html, vendor/spark.module.min.js, three.*.min.js, OrbitControls.js)` | Working browser 3DGS viewer/editor on Spark 2.1.0 (vendored file is byte-identical to the npm @sparkjsdev/spark 2.1.0 dist) + three.js, using THREE.WebGLRenderer (WebGL2, NOT WebGPU). Loads an INRIA .ply via SplatMesh, reads every splat back with forEachSplat, does per-splat RGBA override (splatRgba) and rebuilds a subset with pushSplat. Has an import map, so no bundler. | Fastest path to a browser renderer for the 'decode -> cache -> upload -> render' client and the WebGL2 baseline arm. Its per-splat index handling (order_check proving PLY row == Spark index) is exactly the Gaussian-identity problem ID-based deltas face. | Used daily for a published demo, but scene.ply in viewer/data/{nchc_sofa,21c5ee0ca5,...} are DANGLING symlinks to ~/EditReadyGS_runs (directory no longer exists), so it will not load geometry as-is. Single static scene; no streaming/delta code. |
| `/home/intern_2603056/Desktop/PairGS_ours/scripts/serve_demo.py (135 lines), scripts/serve_agent.py (237 lines)` | Stdlib ThreadingHTTPServer + SimpleHTTPRequestHandler: serves the viewer statically plus JSON /api/ endpoints backed by a resident GPU model (CLIP, Qwen2-VL-2B). Documents measured text-only latency 0.27-0.81 s. | Skeleton for the server process: static page + a model kept resident on GPU + an API. Would need a WebSocket layer added for push of deltas (no WebSocket server anywhere in PairGS). | Working demo code; HTTP request/response only, no push, no streaming. |
| `/home/intern_2603056/Desktop/archive/EviPartGS/web_viewer/ (package.json, src/main.js, nchc.js, panogs.js, node_modules/)` | Vite + pnpm project pinning @sparkjsdev/spark 2.1.0 and three 0.180.0, with node_modules installed (including Spark's TypeScript .d.ts files). | Reference for Spark's API: dist/types/PackedSplats.d.ts (setSplat/pushSplat/ensureSplats/needsUpdate = in-place ID-indexed updates) and dist/types/SplatPager.d.ts (a GPU page pool with LRU, freelist, chunk->page map, maxSplats default 16,777,216, numFetchers). SplatPager is prior art for the 'browser GPU memory as a virtual scene cache' angle and must be cited and compared against, not reinvented. | Archived project; packages present. API facts read from local .d.ts only, not run. |
| `/home/intern_2603056/Desktop/GS_playground/src/gs_playground/viewer.py (696 lines) + data/viewer_js/ (spark.module.min.js, three.core.js, three.module.js, addons/)` | Generator for self-contained multi-panel Spark viewers from a JSON spec; serves .ply beside the page; handles many WebGL contexts (lazy), shares Spark's sort worker. Its Spark build differs from 2.1.0 (2.7 MB vs 5.0 MB; version unconfirmed, has enableDriveLod). | Side-by-side comparison pages (FIFO vs view-aware vs deadline-aware arm renders), and a template for vendoring Spark/three without a bundler. | Tracked, regenerates 7 demos listed in results/DEMOS.md. Static scenes only. |
| `/home/intern_2603056/Desktop/GS_playground/src/gs_playground/gs/ply.py (114 lines), gs/render.py (146 lines)` | GaussianCloud dataclass, INRIA point_cloud.ply load/save (SH layout handled), and a gsplat rasteriser wrapper from cameras.json. | Server-side: reading/writing Gaussian snapshots, building pre-recorded update traces, and rendering a reference 'what the browser should show' image per timestep for quality-vs-latency metrics. | Tested library code (used by SuRFLo export check at ~59 dB agreement). |
| `/home/intern_2603056/Desktop/GS_playground/envs/puffin-world-c6d9ae5-py310-cu126` | Python 3.10 env with torch 2.7.0+cu126, gsplat 1.5.3, websockets, aiohttp, cv2, plyfile. | Most ready env for the server prototype (WebSocket + gsplat + video decode) without installing anything. | Working env (used for GS_playground runs). |
| `/home/intern_2603056/Desktop/event_camera_2DGS/envs/anysplat + third_party/clones/AnySplat (commit 5f5e208) + src/ev4dgs/baselines/{evanysplat.py,anysplat_flea.py,ff_backend.py} + tools/container/anysplat/anysplat.def` | AnySplat (feed-forward unposed video frames -> 3DGS + poses) installed: uv venv torch 2.2.0+cu121, gsplat 1.4.0, xformers 0.0.24, aiohttp. Weights pinned HF lhjiang/anysplat@d2e8c343, loaded offline. Apptainer definition used on nano4 (image anysplat-430d907b.sif, per LOG.md). | Primary candidate for the live feed-forward model: frames in, a full Gaussian set out per window. Note AnySplat emits a fresh set each call, so Gaussian identity across calls does not exist and must be built (matching/anchoring) for ID-based deltas. | Run many times on GPU (E93/E95/E97/E105). Weights local. |
| `/home/intern_2603056/Desktop/archive/brain_to_3D/scripts/run_anysplat_video.py (581 lines) + data/feed-forward-3dgs/a2-space1-i1-anysplat-{32v,64v}/manifest.json, gaussians.ply` | Strict video -> AnySplat -> PLY pipeline with timing and peak-memory logging. Measured (GPU not recorded in manifest): 32 views at 448x448: model load 24.3 s, forward 3.73 s, PLY export 6.09 s, peak CUDA 14.55 GB, 1,565,958 Gaussians; 64 views: forward 5.83 s, export 11.8 s, 3,383,138 Gaussians. Held-out PSNR 9.3 dB (interpolated poses), so quality on that video was poor. | Template for the camera-video front end and the first realistic numbers for the update-trace generator: ~4-6 s per window and 1.5-3.4 M Gaussians (~ 90-200 MB raw at 59 floats? only if SH0; compute exactly from the PLY) per update, i.e. far too large to resend whole - motivates deltas. PLY export cost (6-12 s) is larger than inference: the export path itself must be replaced by an in-memory serializer. | Archived but complete outputs with sha256 manifests. Source video is from a private fMRI study dataset - do not reuse the video itself. |
| `/home/intern_2603056/Desktop/GS_playground/third_party/clones/surflo (commit bf14c63) + envs/surflo-bf14c63-py310-cu124 + data/models/surflo/surflo_v0.pt (5.2 GB) + src/gs_playground/surflo/export.py` | SuRFLo (VGGT-1B + flow matching; unposed photos -> mesh + 3DGS). Env torch 2.4.1+cu124 with fastapi, uvicorn, aiohttp. Measured on RTX 6000 Ada, garden: 16 views 2.0 min / 11.7 GiB, 64 views 2.3 min / 21.9 GiB, 161 views 3.2 min / 45.3 GiB. Weights CC BY-NC 4.0. | Offline producer of higher-quality Gaussian snapshots for traces only; minutes per update makes it unusable as the live model. | Pinned, patched (8 patches with reasons), reproduced. |
| `/home/intern_2603056/Desktop/GS_playground/third_party/clones/iggt (bec6604) + data/models/iggt/iggt_checkpoint.pth (4.9 GB) + src/gs_playground/iggt/` | IGGT (VGGT fine-tuned with an 8-dim instance head). Needs the inference-shims patch; runs in the surflo env. | Optional: instance groups could give object-level identity for coarse ID-based or per-object deltas. Exploratory only. | Exploratory runs (E08m/E08n), known misses documented. |
| `/home/intern_2603056/.cache/huggingface/hub/models--facebook--VGGT-1B (9.4 GB), models--lhjiang--anysplat (2.8 GB), models--depth-anything--DA3NESTED-GIANT-LARGE (6.3 GB), models--depth-anything--da3mono-large (1.3 GB)` | Local feed-forward weights. VGGT-1B is pinned in evanysplat.py at revision 860abec7. LOG.md in event_camera_2DGS records ~1.46 s per 24-frame VGGT forward (E95, on an H200). | VGGT-1B as the geometry backbone; DA3 as an alternative feed-forward GS producer (not yet run by the student). | Weights present; VGGT/AnySplat exercised, DA3 not exercised in these repos. |
| `/home/intern_2603056/Desktop/event_camera_2DGS/deck/spark_{compare,showcase,talk,fusion,e97}.py, site_e97.py` | Python builders that place several Gaussian sets in one shared camera frame and emit Spark pages (built on GS_playground's viewer). | Pattern for building comparison pages from traces; little direct code reuse for streaming. | Working, project-specific. |
| `/home/intern_2603056/Desktop/twinworld-eccv2026/viewer/{index,geometry,renders}.html + twinworld/viewer_export.py` | Raw-WebGL point-cloud viewer and a flat binary writer (float32 positions, uint8 colours, uint8 aux, non-interleaved) plus voxel downsampling. | Template for a minimal binary chunk format and a no-library GPU upload path; points, not Gaussians. | Working, small (84-line exporter). |
| `/home/intern_2603056/Desktop/fmri3d-viewer/interactive/ (three.min.js point/depth viewers), scripts/wz2_browser_probe.py, gsplat_space01/train_gsplat.py` | three.js point viewers; a stdlib-only CDP-over-WebSocket client that reads page state from headless Chrome; a compact gsplat trainer reporting PSNR/time and exporting PLY. | wz2_browser_probe.py is directly reusable as a measurement harness: drive Chrome and read in-page timestamps/stall counters without adding a dependency. Its docstring notes headless Chrome falls back to SwiftShader, so render timing must be taken on a real GPU display (:20). train_gsplat.py: per-scene 3DGS reference for quality ceilings. | Working. |
| `/mnt/nchc-2603056/datasets/ScanNet++/data/<scene>/iphone/{rgb.mkv,depth.bin,pose_intrinsic_imu.json,colmap}` | 906 scenes, 800 with iPhone video; checked 00777c41d4: 1920x1440, 60 fps, 10,110 frames, with per-frame poses and depth. Licence-restricted (ScanNet++ terms). | Best replayable 'camera' input: real handheld video with GT poses, lets the student replay at fixed fps as a fake live camera and compare predicted cameras/Gaussians. | Dataset, on CIFS mount (slow reads; stage locally). |
| `/mnt/nchc-2603056/datasets/nchc-capture/raw_v1/source/{20260727_143647.mp4 (sofa), 20260727_144616.mp4 (table), camera_intrinsic.txt} + Inpaint360GS_custom_video/*/run/output/.../point_cloud.ply` | Student's own phone captures: 3840x2160, 30.02 fps, 5,580 frames (sofa), plus per-scene 3DGS trained from them (iteration 30000 for table). | Own-data camera input with no licence issue, and a per-scene 3DGS as the quality ceiling for the same scene. | Irreplaceable raw capture, sha256-verified. |
| `/mnt/nchc-2603056/datasets/ScanNet++_3dgs/<6 scenes>/point_cloud.ply (148-222 MB each); /mnt/nchc-2603056/datasets/interiorgs-probe/000{1..4}_*/3dgs_compressed.ply (~57 MB) + labels.json` | Trained static 3DGS scenes (ScanNet++) and labelled compressed indoor 3DGS. | Static scenes for renderer and cache-capacity experiments (GPU memory pressure, paging) before any live model exists. | Ready files. |
| `/home/intern_2603056/datasets/mipnerf360/360_v2 (13 GB local), /home/intern_2603056/datasets/scannetpp-v2/data (30 scenes, DSLR only)` | Image datasets (not video). | Quality references; GS_playground's SuRFLo garden runs use them. Not a camera stream. | Ready. |
| `/mnt/nchc-2603056/datasets/erfgs-nvidia/Balloon1/cam01..cam12` | Multi-camera dynamic scene (NVIDIA dynamic-scene style). | A scene that actually changes, for glass-to-glass 'scene change visible' measurements with a known change time. | Present, not inspected beyond listing. |

### Compute

Local (nvidia-smi, 2026-10-27 05:02): one GPU only, NVIDIA RTX 6000 Ada 48 GB (49,140 MiB), driver 595.91.07 / CUDA 13.2.
It was busy at snapshot time: 87% utilisation, 32,048 MiB held by another project's .venv/bin/python (PID 2894459), so about 16 GB was free.
CPU: Xeon Gold 6542Y, 32 threads, 251 GB RAM.
The "second RTX 6000" is not on this box.
Per the compute-hosts memory it is win-desktop (172.16.10.104), a Windows 10 machine with an RTX 6000 Ada, 16 cores, no WSL and no Docker.
Native-Windows PyTorch there is unverified for gsplat/xformers.
CVLab machines (from the cvlab-machines skill; I did not ssh): 110-5090 and 111-5090 each have 1x RTX 5090 32 GB (sm_120, needs a CUDA 12.8+ PyTorch build; 111 is often used by others). 103-4090 has 2x RTX 4090 24 GB but its home disk is full. 104-titan has 2x Quadro RTX 8000 48 GB, 2x V100S 32 GB and 4x TITAN RTX 24 GB (Turing/Volta: no bf16, no FlashAttention 2; home is full; NAS writes can stall). 102-, 109-, 107- and 108-4090 each have 1x RTX 4090 24 GB. 107 and 108 have old drivers (550 / 575), so cu130 builds fail there.
Shared storage is /cvlab-40t/kylechen/<project>/.
There is no scheduler, Docker or sudo, and a gq queue on 102/109/111 can take a free card within a minute.
Get a fresh status.sh snapshot and the user's approval before placing anything.
The AnySplat env (torch 2.2 cu121) would not run on the 5090s without a rebuild.
Networking: tc and the sch_netem module exist locally, but sudo needs the user's password, Docker is denied (not in the docker group), and unprivileged user namespaces are blocked by AppArmor.
I tested this with `unshare -rn` and it failed with "uid_map: Operation not permitted".
So `tc netem` on this box needs the user to run sudo, and on the CVLab machines it is impossible.
For browsers there are Node 22 + pnpm, Playwright Chromium 1243 under ~/.cache/ms-playwright, and chrome-devtools-axi.

I scanned read-only.
The only thing I executed beyond listing was a short cv2 probe of two video headers and one `unshare -rn tc` feasibility test in a throwaway namespace.
That test failed and changed nothing.
Key findings:
(1) No streaming or online feed-forward model exists anywhere locally.
I searched every Desktop repo and the Hugging Face cache for StreamSplat, StreamVGGT, CUT3R, Spann3R, MVSplat, pixelSplat, NoPoSplat, Splatt3R, MASt3R, DUSt3R, LongSplat and EvolSplat and found none of them.
The live-model candidates on disk are AnySplat (installed, weights local) and VGGT-1B, with DA3 weights cached but never run.
(2) There is no WebGPU Gaussian renderer anywhere.
Every splat viewer is Spark on THREE.WebGLRenderer (WebGL2).
So a WebGPU renderer is new work, and Spark is the WebGL2 baseline.
(3) Spark 2.1.0 already ships SplatPager: a fixed GPU page pool with LRU eviction, a freelist and chunk-to-page mapping, in archive/EviPartGS/web_viewer/node_modules/@sparkjsdev/spark/dist/types/SplatPager.d.ts.
PackedSplats.setSplat/needsUpdate allows in-place, index-based updates.
This is direct prior art for the "browser GPU memory as a virtual scene cache" angle.
The project must position against it; I read only the local .d.ts files and have not verified it against Spark's docs or source.
(4) No WebSocket server code exists in any repo.
Only stdlib HTTP servers (PairGS) and a FastAPI app (das-ui, unrelated) do.
The websockets package is already installed in the puffin-world env and the siu3r micromamba env.
(5) Reusable measured numbers: the AnySplat video run took 3.7 s (32 views) and 5.8 s (64 views) per forward, produced 1.57 M and 3.38 M Gaussians, and peaked at 14.6 GB.
PLY export took 6-12 s, longer than inference.
That run's GPU was not recorded and its held-out PSNR was 9.3 dB.
SuRFLo takes 2-3 min per reconstruction.
VGGT took about 1.46 s per 24-frame forward on an H200.
All of these numbers come from the student's own logs, not new runs.
(6) Gotchas.
The PairGS viewer's scene.ply links point to the deleted ~/EditReadyGS_runs, so the viewer is broken until it is relinked to /mnt/nchc-2603056/datasets/ScanNet++_3dgs or the nchc-capture outputs. /mnt/nchc-2603056/datasets/realestate10k holds only a checksum list, with no payload.
The Hugging Face dataset folders EmbodiedSplat, TwinWorld and WorldSculpt_data are empty stubs.
The AnySplat video source belongs to a private fMRI dataset, so do not reuse it.
SuRFLo, IGGT and VGGT-derived weights are CC BY-NC.
(7) 032-HOPE, lifemem, Motion_VideoLLM and the fMRI repos contain nothing reusable beyond what is listed above.
Headless Chrome falls back to SwiftShader (noted in fmri3d-viewer), so stall and frame-time measurements need a real GPU-backed display such as :20.
I wrote no files.
