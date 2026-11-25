# Streaming feed-forward 3D reconstruction models whose per-stream state grows with stream length, and their systems characteristics (2024 to November 2026)

Scope note: written 2026-11-18.
Sources were read from primary documents: arXiv metadata (export.arxiv.org API), arXiv HTML full text, GitHub READMEs and LICENSE files (GitHub API), and Hugging Face model metadata (HF API).
Papers already covered in `docs/reading_notes.md` (StreamSplat ICLR 2026 arXiv 2506.08862, S2GS arXiv 2603.14232) are only cited here, not re-surveyed.
"Venue (self-reported)" means the venue comes from the authors' README or repo description, not from a proceedings page I opened.
Every number derived by me (not printed by the authors) is in an Inferences list and shows its arithmetic.

## Q1. Which streaming reconstruction models keep growing state, with venue and year?

### Takeaway
The growing-state family is now large: one causal-KV-cache line (StreamVGGT, STream3R, and at least seven 2025-2026 cache-bounding add-ons that reuse the StreamVGGT checkpoint), one explicit-memory line (Spann3R, MUSt3R, Point3R, LONG3R), and a new 2026 line of online feed-forward 3DGS models with KV caches (ReCoSplat, StreamSplat arXiv 2608.01659).
CUT3R, TTT3R and MeMix keep a constant-size recurrent state and are the natural contrast arm.

### Cited Findings

Peer-reviewed (venue source given per row):

| Model | Title | Venue (source) | arXiv | Growing state | Code / weights | License (code; weights) |
| --- | --- | --- | --- | --- | --- | --- |
| Spann3R | 3D Reconstruction with Spatial Memory | 3DV 2025, award candidate (repo description) | 2408.16061 | Memory bank: dense working memory (last 5 frames) + sparse long-term memory | [github.com/HengyiWang/spann3r](https://github.com/HengyiWang/spann3r); checkpoint on Google Drive | CC BY-NC-SA 4.0 (LICENSE file) |
| CUT3R (contrast) | Continuous 3D Perception Model with Persistent State | CVPR 2025 Oral (README) | 2501.12387 | Constant: 768 state tokens of dim 768 | [github.com/CUT3R/CUT3R](https://github.com/CUT3R/CUT3R); Google Drive checkpoints | CC BY-NC-SA 4.0 |
| MUSt3R | MUSt3R: Multi-view Network for Stereo 3D Reconstruction | CVPR 2025 (arXiv comment) | 2503.01661 | Multi-layer memory of past-frame tokens, grows linearly with images; keyframe selection | [github.com/naver/must3r](https://github.com/naver/must3r); NAVER download server | MUSt3R Non-Commercial License (NAVER), plus training-dataset terms |
| SLAM3R | SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos | CVPR 2025 Highlight (arXiv comment, repo) | 2412.09401 | Sliding window + keyframe buffer (windowed, not surveyed in depth) | [github.com/PKU-VCL-3DV/SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R) | CC BY-NC-SA 4.0 |
| Point3R | Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory | NeurIPS 2025 (repo description) | 2507.02863 | Explicit 3D pointer memory that grows with explored space; fusion merges pointers | [github.com/YkiWu/Point3R](https://github.com/YkiWu/Point3R); Google Drive checkpoint | CC BY-NC-SA 4.0 |
| LONG3R | LONG3R: Long Sequence Streaming 3D Reconstruction | ICCV 2025 (arXiv comment) | 2507.18255 | Spatio-temporal 3D memory bank with gating and voxel pruning | README says "Code Coming Soon" ([repo](https://github.com/zgchen33/LONG3R)) | No license file, no code |
| StreamVGGT | Streaming 4D Visual Geometry Transformer | ICLR 2026 (repo description) | 2507.11539 | KV cache of all past frames' tokens in every global (temporal) attention layer | [github.com/wzzheng/StreamVGGT](https://github.com/wzzheng/StreamVGGT); [HF lch01/StreamVGGT](https://huggingface.co/lch01/StreamVGGT) | Code CC BY-NC-SA 4.0; weights tagged cc-by-nc-4.0 on HF |
| STream3R | STream3R: Scalable Sequential 3D Reconstruction with Causal Transformer | ICLR 2026 (README "Accepted to ICLR 2026", Jan 26 2026) | 2508.10893 | KV cache (causal mode) or sliding window of 5 (window mode) | [github.com/NIRVANALAN/STream3R](https://github.com/NIRVANALAN/STream3R); [HF yslan/STream3R](https://huggingface.co/yslan/STream3R) | S-Lab License 1.0 (non-commercial); HF tag "other" |
| TTT3R (contrast) | TTT3R: 3D Reconstruction as Test-Time Training | ICLR 2026 (arXiv journal_ref) | 2509.26645 | Constant (CUT3R state with a training-free test-time-training update rule) | [github.com/Inception3D/TTT3R](https://github.com/Inception3D/TTT3R); uses CUT3R checkpoint | CC BY-NC-SA 4.0 (LICENSE file is CUT3R's text) |
| VGGT-Long | VGGT-Long: Chunk it, Loop it, Align it | ICRA 2026 (arXiv comment, README) | 2507.16443 | Chunked offline VGGT plus alignment; GPU memory per chunk is constant, intermediate results spill to disk | [github.com/DengKaiCQ/VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) | VGGT License (code); VGGT-1B weights cc-by-nc-4.0, VGGT-1B-Commercial gated |
| OVGGT | OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer | ECCV 2026 (README "[2026.07] accepted to ECCV 2026"; an ECCV 2026 spotlight page appeared in search, not opened) | 2603.05959 | Bounded StreamVGGT KV cache (fixed token budget) | [github.com/VAISR/OVGGT](https://github.com/VAISR/OVGGT); reuses StreamVGGT checkpoint | CC BY-NC-SA 4.0 (StreamVGGT text) |
| XStreamVGGT | XStreamVGGT: Extremely Memory-Efficient Streaming VGGT with KV Cache Compression | SID Display Week 2026 conference paper (README), also listed in J. Soc. Information Display ([Wiley](https://sid.onlinelibrary.wiley.com/doi/10.1002/jsid.70086)) | 2601.01204 | Bounded StreamVGGT KV cache (pruning + quantization) | [github.com/ywh187/XStreamVGGT](https://github.com/ywh187/XStreamVGGT); reuses StreamVGGT checkpoint | CC BY-NC-SA 4.0 |
| LongSplat (online generalizable 3DGS) | LongSplat: Online Generalizable 3D Gaussian Splatting from Long Sequence Images | Cited as "Huang et al. 2026, Proceedings of the AAAI Conference" by arXiv 2608.01659; not verified on AAAI site | 2507.16144 | Accumulated Gaussian set, compressed via Gaussian-Image Representation | None found; arXiv 2608.01659 states it has no public implementation | Unknown |

Not to be confused with LongSplat (NVIDIA, ICCV 2025, optimization-based unposed 3DGS); that is a different paper and not feed-forward.

arXiv-only preprints (no venue found as of 2026-11-18):

| Model | arXiv (first version) | Growing state / bound | Code / weights | License |
| --- | --- | --- | --- | --- |
| StreamGS (online generalizable 3DGS, unposed) | 2503.06235 (Mar 2025) | Per-frame Gaussians, adjacent-frame merging | None; arXiv 2606.03254 says "official implementations are unavailable" | Unknown |
| WinT3R | 2509.05296 (Sep 2025) | Window of 4, stride 2, plus a camera-token pool that grows by one compact token per frame | [github.com/LiZizun/WinT3R](https://github.com/LiZizun/WinT3R); [HF lizizun/WinT3R](https://huggingface.co/lizizun/WinT3R) | Repo LICENSE "WinT3R for non-commercial purposes"; HF weights tagged apache-2.0 (conflict) |
| Evict3R | 2509.17650 (Sep 2025) | StreamVGGT KV cache with per-layer budget and attention-score eviction | [github.com/soroush-mim/evict3r](https://github.com/soroush-mim/evict3r); reuses StreamVGGT checkpoint | CC BY-NC-SA 4.0 |
| InfiniteVGGT | 2601.02281 (Jan 2026) | Rolling memory: first frame's KV kept as anchor, rest pruned by key cosine diversity | [github.com/AutoLab-SAI-SJTU/InfiniteVGGT](https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT); reuses StreamVGGT checkpoint; Long3D dataset on HF | Code Apache-2.0; weights are StreamVGGT's (CC BY-NC) |
| FrameVGGT | 2603.07690 (Mar 2026) | Frame-level KV segments + sparse anchors under a fixed budget | [github.com/ZhisongXu/FrameVGGT](https://github.com/ZhisongXu/FrameVGGT) | Apache-2.0 (code); StreamVGGT weights |
| ReCoSplat (online feed-forward 3DGS) | 2603.09968 (Mar 2026, v2 Sep 2026) | KV cache over past chunks, compressed (early-layer truncation + chunk-level retention) | [github.com/peelstnac/ReCoSplat](https://github.com/peelstnac/ReCoSplat); [HF chfrn/ReCoSplat](https://huggingface.co/chfrn/ReCoSplat) (224x224 checkpoint) | GitHub repo has no license file; HF weights tagged MIT |
| MeMix (contrast) | 2603.15330 (Mar 2026) | Constant: recurrent state split into memory patches, O(1) memory | Project page only in abstract | Not checked |
| StreamCacheVGGT | 2604.15237 (Apr 2026) | StreamVGGT KV cache, keep / merge / evict three tiers, budgets 200K or 400K tokens | No code link found in the arXiv HTML | Unknown |
| OF3GS (online feed-forward 3DGS, unposed) | 2606.03254 (Jun 2026) | Causal feature extractor with KV caching (per search summary; I did not confirm the cache design in the full text) | No code link found in the arXiv HTML | Unknown |
| StreamSplat (Song et al.; different paper from the DSL-Lab StreamSplat) | 2608.01659 (Aug 2026) | Voxel-Aligned Causal Cache (VACC): memory grows with explored geometry, not stream length | "Code will be made publicly available upon acceptance" | Unknown |
| SceneBound | 2611.00756 (Oct 31 2026) | Training-free write rule for recurrent state, TTT fast weights or KV cache; writes a frame only if its pose is far from stored keyframes | Not checked | Not checked |

Sources for the tables: arXiv metadata for all ids via the [arXiv API](https://export.arxiv.org/api/query?id_list=2408.16061,2501.12387,2503.01661,2507.02863,2507.11539,2508.10893,2507.16443,2507.16144,2503.06235,2509.17650,2601.02281,2601.01204,2603.05959,2604.15237,2603.07690,2509.26645,2507.18255,2509.05296,2412.09401); arXiv 2611.00756, 2603.15330, 2603.09968, 2608.01659, 2606.03254 via the [arXiv API](https://export.arxiv.org/api/query?id_list=2611.00756,2603.15330,2603.09968,2608.01659,2606.03254); repo licenses and descriptions via the GitHub API on each linked repo; HF licenses via `huggingface.co/api/models/<id>`.

Claims borrowed (one per model, exact wording where quoted):
- StreamVGGT: "our method exhibits a moderate increase in inference time and memory usage when processing longer sequences. This is mainly because the number of cached memory tokens grows with the sequence length." - [arXiv 2507.11539 HTML, Sec. 4.6](https://arxiv.org/html/2507.11539)
- STream3R: the README reports per-frame peak VRAM and runtime for causal (KV cache) and window (size 5) modes up to 200 frames on one H200 - [STream3R README](https://github.com/NIRVANALAN/STream3R)
- MUSt3R: "As the memory grows linearly with the number of images, this can become an computational issue for large image collections. To mitigate this, we resort to a heuristic selection of memory tokens." - [arXiv 2503.01661 HTML, Sec. 3.4](https://arxiv.org/html/2503.01661)
- Point3R: "since the spatial pointer memory expands as the scene exploration progresses, we design a simple yet effective fusion mechanism" - [arXiv 2507.02863 HTML](https://arxiv.org/html/2507.02863)
- Spann3R: "long-term memory features accumulate over time, which can increase GPU memory usage and slow down speed ... Once the long-term memory reaches a predefined threshold, we perform memory sparsification by retaining only the top-k tokens." - [arXiv 2408.16061 HTML, Sec. 3.2](https://arxiv.org/html/2408.16061)
- CUT3R: "The state consists of 768 tokens, each with a dimensionality of 768." - [arXiv 2501.12387 HTML](https://arxiv.org/html/2501.12387)
- StreamSplat (2608.01659): VACC "stores historical 3D tokens in a memory-bounded voxel structure so that memory grows with explored scene geometry rather than stream length" - [arXiv 2608.01659 abstract](https://arxiv.org/abs/2608.01659)
- SceneBound: "the count grows with the places visited and stays flat when the camera returns" - [arXiv 2611.00756 abstract](https://arxiv.org/abs/2611.00756)

### Inferences
- For a "memory grows with stream length" tenant, the cleanest choice is StreamVGGT with its full cache: the state is a plain KV cache, its growth is linear in frames, the weights are on HF, and five bounding methods (Evict3R, InfiniteVGGT, XStreamVGGT, OVGGT, FrameVGGT) reuse the same checkpoint, so cache policy can be varied with everything else held fixed.
- Point3R, LONG3R, StreamSplat (2608.01659) and SceneBound grow with explored space rather than time, so a manager that predicts memory from stream length would mispredict them when a user revisits the same room.
- ReCoSplat is the only Gaussian-producing model with a growing KV cache that I found with released code and weights; it fits the project's 3DGS framing better than the pointmap models.

### Gaps
- Venues for StreamVGGT, Point3R, Spann3R and OVGGT rest on author READMEs or repo descriptions; I did not open OpenReview, CVF or ECVA proceedings pages.
- LongSplat's AAAI 2026 venue comes only from a citing paper's bibliography.
- I found no venue for InfiniteVGGT, Evict3R, StreamCacheVGGT, FrameVGGT, WinT3R, ReCoSplat, OF3GS, StreamSplat (2608.01659), StreamGS or SceneBound; some may have been accepted after their last arXiv update.
- OnlineX (Xia et al. 2026, cited by OF3GS as an online 3DGS method without public code) was not surveyed.

## Q2. What is the state, how many bytes per frame, how does per-frame compute grow, and are memory or latency curves versus frames reported?

### Takeaway
For the StreamVGGT/STream3R family the authors publish memory and latency versus frames: StreamVGGT adds about 0.12 GB and 3.4-3.9 ms per cached frame on an A800 at 518x392, STream3R's causal mode adds about 0.20 GB and 5.3 ms per frame on an H200 at 518x384, and both hit 25-45 GB by 200 frames.
Explicit-memory models (Spann3R, Point3R, MUSt3R) grow far more slowly in practice because they sparsify, fuse or select keyframes.

### Cited Findings

Systems numbers (GPU and resolution as stated by the source):

| Model | Memory vs frames | Per-frame latency vs frames | GPU, resolution | Source |
| --- | --- | --- | --- | --- |
| StreamVGGT (full cache) | 2.1 / 2.7 / 3.2 / 4.3 / 5.5 / 6.6 GB at 1 / 5 / 10 / 20 / 30 / 40 frames; 25.3 GB peak on a 200-frame sequence | 63 / 88 / 120 / 152 / 184 / 216 ms for the last frame at 1-40 frames; 733 ms at frame 200 | A800, FlashAttention-2, 518x392 | [arXiv 2507.11539 HTML, Tables 7-8](https://arxiv.org/html/2507.11539) |
| Spann3R | 5.3 GB (1 frame) to 5.7 GB (40); 6.7 GB at 200 frames | 82-500 ms over 1-40 frames; 221 ms at frame 200 | A800, set to 512x384 by the StreamVGGT authors | [arXiv 2507.11539 HTML, Tables 7-8](https://arxiv.org/html/2507.11539) |
| Spann3R (own paper) | 11 GB in the default online setting | about 65 fps (abstract says "over 50 fps") | RTX 4090 | [arXiv 2408.16061 HTML, Sec. 4](https://arxiv.org/html/2408.16061) |
| CUT3R (constant state) | 3.4 GB (1) to 4.2 GB (40); 7.6 GB at 200 frames | about 100 ms flat (99-102 ms) over 1-40 frames; 102.5 ms at frame 200 | A800, 512x384 | [arXiv 2507.11539 HTML, Tables 7-8](https://arxiv.org/html/2507.11539) |
| CUT3R (own paper) | n/a | 16.58 FPS on KITTI | A100, 512x144 | [arXiv 2501.12387 HTML, Table 2](https://arxiv.org/html/2501.12387) |
| STream3R, causal (KV cache) | 5.49 / 9.02 / 12.92 / 21.00 / 25.03 / 29.10 / 33.21 / 41.31 / 45.41 GB at 1 / 20 / 40 / 80 / 100 / 120 / 140 / 180 / 200 frames | 0.116 / 0.203 / 0.306 / 0.499 / 0.595 / 0.695 / 0.792 / 0.991 / 1.170 s at the same frame counts | H200, 518x384, `StreamSession` | [STream3R README, "GPU Memory Usage and Runtime"](https://github.com/NIRVANALAN/STream3R) |
| STream3R, window 5 | 5.49 GB at 1 frame, then flat 6.53 GB from 20 to 200 frames | 0.117 s at 1 frame, then 0.144-0.153 s | H200, 518x384 | [STream3R README](https://github.com/NIRVANALAN/STream3R) |
| MUSt3R-224 / MUSt3R-512 | 4.1 GB / 8.1 GB | 40.41 FPS / 12.10 FPS (Spann3R in the same table: 27.38 FPS, 5.0 GB; DUSt3R-224 global alignment: 0.74 FPS, 38.1 GB) | A100; rendering batch capped at 10 (224) and 5 (512) | [arXiv 2503.01661 HTML, Table 6](https://arxiv.org/html/2503.01661) |
| MUSt3R (visual odometry) | n/a | "around 10 FPS" | A100 | [arXiv 2503.01661 HTML, Sec. 5.1](https://arxiv.org/html/2503.01661) |
| Point3R | 5.46 GB peak (MASt3R-SLAM: 7.18 GB) | 0.20 s per frame (MASt3R-SLAM: 0.11 s) | 7-Scenes seq-01; GPU not stated in that table | [arXiv 2507.02863 HTML, Table 8](https://arxiv.org/html/2507.02863) |
| Point3R vs others, KITTI video depth | n/a | Point3R 3.6 FPS; StreamVGGT 13.7; CUT3R 12.9; Spann3R 10.4 | As measured by WinT3R authors | [arXiv 2509.05296 HTML, Table 4](https://arxiv.org/html/2509.05296) |
| TTT3R (constant) | "just 6 GB of GPU memory to process thousands of images" | "20 FPS" | Evaluated on "a single 48GB NVIDIA GPU", 50-1000 views | [arXiv 2509.26645 HTML](https://arxiv.org/html/2509.26645) |
| VGGT-Long | Constant per chunk; disk grows: about 50 GiB for the 4,500-frame KITTI 00, about 5 GiB for 300 frames | n/a | RTX 4090 24 GiB | [VGGT-Long README](https://github.com/DengKaiCQ/VGGT-Long) |
| ReCoSplat | KV cache cut by 93% at N=256, n=8 (uncompressed curve only in Fig. 6, not read) | 45.1 FPS average input throughput, 41.1 FPS end-of-stream, 256-view streams | RTX 6000 Ada, 224x224 checkpoint | [arXiv 2603.09968 HTML](https://arxiv.org/html/2603.09968) |
| StreamSplat (2608.01659) | 24.5 GB peak at 256 views with VACC; 41.3 GB with an append-all cache | about 143 ms from chunk to one rendered novel view (6.98 FPS) | RTX PRO 6000 96 GB | [arXiv 2608.01659 HTML, ablation](https://arxiv.org/html/2608.01659) |

Per-token structure:
- OVGGT states that StreamVGGT's cache "grows by M entries per layer per frame" and that "At 518x392 resolution (M=1,041), merely 100 frames consume ~10 GB of VRAM" - [arXiv 2603.05959 HTML, Sec. 3](https://arxiv.org/html/2603.05959)
- OVGGT also writes that per-step attention cost O(M·|C_t|) "increases with t, degrading throughput over time" - [arXiv 2603.05959 HTML](https://arxiv.org/html/2603.05959)
- Evict3R states the total KV footprint scales with sequence length × tokens per frame × global layers × heads × head dimension, "This also results in quadratic computat[ion]" over a stream - [arXiv 2509.17650 HTML, Sec. III](https://arxiv.org/html/2509.17650)
- The VGGT aggregator that StreamVGGT inherits defaults to `patch_size=14`, `embed_dim=1024`, `depth=24`, `num_heads=16`, `num_register_tokens=4`, `aa_order=["frame","global"]` - [vggt/models/aggregator.py](https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py)
- The CUT3R README warns that the released inference "process[es] inputs in parallel within the encoder, which results in linear memory consumption as the number of frames increases" - [CUT3R README](https://github.com/CUT3R/CUT3R)
- TTT3R reports that full-attention methods (VGGT, StreamVGGT) "hit OOM at >150 frames" in its video-depth test on the 48 GB GPU - [arXiv 2509.26645 HTML, Fig. 8 caption](https://arxiv.org/html/2509.26645)
- ReCoSplat's backbone has 36 frame/global alternating layers (18 global) and "Naively accumulating a KV cache across all global attention layers causes VRAM usage to grow rapidly with sequence length" - [arXiv 2603.09968 HTML, Sec. 3.4](https://arxiv.org/html/2603.09968)

### Inferences
- Bytes per frame, derived for StreamVGGT: 1,041 tokens × 24 global layers × 2 (K and V) × 1,024 dims × 2 bytes (bf16) = 102,334,464 B, about 0.10 GB (97.6 MiB) per frame. This matches OVGGT's "~10 GB per 100 frames" and the measured slope (25.3 - 2.1) / 199 = 0.117 GB per frame from StreamVGGT Table 8 (the extra ~15% is plausibly activations; not tested).
- STream3R's README slope is (45.41 - 5.49) / 199 = 0.201 GB per frame. At 518x384 there are 37 × 27 + 5 = 1,004 tokens; with the same 24 × 1,024 layout an fp32 cache would be 1,004 × 24 × 2 × 1,024 × 4 = 197 MB per frame, which matches. I did not check STream3R's backbone dimensions or cache dtype, so "fp32 cache" is a hypothesis.
- Latency growth, derived: StreamVGGT (733 - 63) / 199 = 3.4 ms per cached frame (A800); STream3R causal (1.170 - 0.116) / 199 = 5.3 ms per cached frame (H200). Total compute for a T-frame stream therefore grows as O(T²), and a single user's update rate falls over the session.
- CUT3R's state is 768 × 768 values, about 2.4 MB in fp32, yet StreamVGGT's measurement shows CUT3R's memory rising 3.4 to 7.6 GB over 200 frames; together with the CUT3R README warning, this growth is an implementation artifact (outputs and parallel encoding), not model state. A "constant-state" control arm must be checked for this before it is used as the null.
- Extrapolated single-tenant ceiling on the 48 GB RTX 6000 Ada (linear extrapolation, not measured): StreamVGGT full cache at (48 - 2.1) / 0.117 ≈ 390 frames; STream3R causal at (48 - 5.49) / 0.201 ≈ 210 frames. At 1 frame per second of input, either fills the GPU within about 4-7 minutes; at 30 fps, within seconds, which is why every paper subsamples frames (stride 2 to 100).
- Compared to the project's own measurements, StreamVGGT at 40 frames (6.6 GB, 216 ms) already exceeds the measured StreamSplat (DSL-Lab) constant 4,094 MiB, and at 200 frames (25.3 GB, 733 ms) it is about 2.7× AnySplat k=16 (9.4 GiB, 0.86 s); the two have opposite growth shapes (AnySplat re-computes k frames, StreamVGGT caches all frames).

### Gaps
- I did not read the numbers inside figures (StreamVGGT Fig. 2, Point3R Fig. 4 pointer count vs frames, OVGGT Fig. 6 FPS and VRAM vs length, ReCoSplat Fig. 6 measured VRAM), only text and tables.
- Point3R's per-frame memory growth before and after fusion is shown only as a figure; I have no bytes-per-frame number for it.
- LONG3R reports FPS on an RTX 3090 at 224x224, but its FPS column was in math markup I could not extract; no memory number found.
- WinT3R's own FPS row in its Table 4 was truncated in my extraction.
- No paper reports memory per user under concurrent multi-stream load; every number above is one stream on one GPU.

## Q3. How is the state bounded, and what does it cost in accuracy?

### Takeaway
Bounding methods fall into windows/chunks, attention- or activation-scored token eviction, diversity-based rolling memory, quantization, merging, and spatial (voxel or pose) gating.
On StreamVGGT's own 200-frame test a 50-frame window cut memory from 25.3 to 8.2 GB and latency from 733 to 219 ms with no accuracy loss, and later eviction papers report roughly 2× memory savings at near-zero accuracy cost, but the accuracy cost depends strongly on the budget.

### Cited Findings
- StreamVGGT Table 8, 200-frame sequence, Acc mean (lower is better) / last-frame latency / peak memory: no pruning 0.058 / 733 ms / 25.3 GB; window 50 frames 0.054 / 219 ms / 8.2 GB; window 100 frames 0.062 / 393 ms / 14.0 GB; K-nearest-frames K=50 0.157 / 220 ms / 8.4 GB; K=100 0.054 / 393 ms / 14.0 GB - [arXiv 2507.11539 HTML, Table 8](https://arxiv.org/html/2507.11539)
- STream3R window mode (size 5) holds 6.53 GB and about 0.15 s per frame up to 200 frames versus 45.41 GB and 1.17 s in causal mode - [STream3R README](https://github.com/NIRVANALAN/STream3R)
- Evict3R "reduces peak memory from 18.63 GB to 9.39 GB while accuracy and completeness drop by only 0.003" on 7-Scenes long sequences, and enables "2× - 10×" longer sequences on a single GPU - [Evict3R README](https://github.com/soroush-mim/evict3r); [arXiv 2509.17650 HTML](https://arxiv.org/html/2509.17650)
- Evict3R per-frame latency on 7-Scenes (Table VI, 1× / 8× length): StreamVGGT 0.107 / 0.142 s; StreamVGGT∗ 0.141 / 0.475 s; Evict3R B=0.1 0.131 / 0.187 s; B=0.7 0.158 / 0.491 s - [arXiv 2509.17650 HTML, Table VI](https://arxiv.org/html/2509.17650)
- OVGGT notes that Evict3R's allocated VRAM is only ~1 GB above OVGGT's "yet its reserved memory is considerably higher because materializing attention maps for eviction precludes FlashAttention" - [arXiv 2603.05959 HTML, Sec. 5](https://arxiv.org/html/2603.05959)
- OVGGT default budget is B = 200K tokens, "occupying approximately 10 GB", "fitting within a 12 GB VRAM envelope"; all experiments on one 32 GB RTX 5090, where StreamVGGT OOMs beyond ~200 frames - [arXiv 2603.05959 HTML](https://arxiv.org/html/2603.05959)
- XStreamVGGT, with a 2K cache length plus KV quantization, reports memory reduced 4.42× and inference 5.48× faster with "mostly negligible performance degradation", measured on one 80 GB A100 - [arXiv 2601.01204 HTML](https://arxiv.org/html/2601.01204)
- InfiniteVGGT keeps the first frame's complete KV cache as immutable anchors, prunes the rest by key cosine similarity (FlashAttention compatible), allocates budget per layer by measured diversity, and ablates per-head per-layer budgets of 10,000 / 25,000 / 50,000 tokens; it introduces Long3D with sequences of about 10,000 frames - [arXiv 2601.02281 HTML](https://arxiv.org/html/2601.02281)
- StreamCacheVGGT replaces pure eviction with three tiers (retain, merge, evict; merge ratio 0.15) under 200K or 400K-token budgets - [arXiv 2604.15237 HTML](https://arxiv.org/html/2604.15237)
- FrameVGGT argues token-level retention "can fragment geometric context under fixed budgets" and keeps whole-frame KV segments plus sparse anchors; experiments on one RTX A6000 - [arXiv 2603.07690 HTML](https://arxiv.org/html/2603.07690)
- Spann3R: working memory of the last 5 frames (insert only if max similarity < 0.95), long-term memory sparsified to top-k by accumulated attention once a threshold is reached - [arXiv 2408.16061 HTML, Sec. 3.2](https://arxiv.org/html/2408.16061)
- LONG3R: memory gating removes memory irrelevant to the current frame ("a 27% reduction on 7Scenes") and 3D spatial pruning keeps only the highest-attention token per adaptive-size voxel - [arXiv 2507.18255 HTML](https://arxiv.org/html/2507.18255)
- MUSt3R's README advises up to 300 memory images, "the more images there are, the slower/more memory hungry it'll be" - [MUSt3R README](https://github.com/naver/must3r)
- ReCoSplat drops the KV cache in the first 10 of 18 global layers and keeps, in the other 8, every view of the first chunk plus only the last view of each later chunk (8 + ⌈(N-8)/n⌉ views), "reducing the overall KV cache size by 93%" at N=256, n=8 - [arXiv 2603.09968 HTML, Sec. 3.4](https://arxiv.org/html/2603.09968)
- StreamSplat (2608.01659): replacing VACC with append-all "slightly improves the 256-view score but greatly increases peak memory from 24.5 GB to 41.3 GB"; VACC "costs 0.29 dB at 256 views but enables the full model to scale to 512" views and beyond (K=4 pivots per voxel, voxel size 0.04) - [arXiv 2608.01659 HTML](https://arxiv.org/html/2608.01659)
- SceneBound: "spending the update budget of a gate at uniformly spaced frames, with no signal, matches nine published gates"; its pose-radius rule gives the lowest mean trajectory error among the training-free rules it evaluated on CUT3R's state and StreamVGGT's KV cache, with keyframes growing 3% to 6% over nine passes of a scene - [arXiv 2611.00756 abstract](https://arxiv.org/abs/2611.00756)
- Constant-state models trade accuracy over long streams: TTT3R says recurrent memory methods "suffer from the forgetting problem from earlier frames, leading to significant performance degradation as the number of input views increases" - [arXiv 2509.26645 HTML](https://arxiv.org/html/2509.26645)

### Inferences
- On StreamVGGT's own 200-frame test, a 50-frame window was no worse in accuracy than the full cache, so "full cache" is not obviously the quality upper bound at that length; bounding policies should be compared against the window baseline, not only against full cache.
- SceneBound's uniform-spacing result implies that the amount written, not which frames, drives most of the mean accuracy, which makes "budget in tokens" a reasonable single knob for a memory manager.
- Eviction that needs attention maps (Evict3R) loses FlashAttention and so costs reserved memory and latency; activation- or key-similarity scoring (OVGGT, InfiniteVGGT) does not. A systems study should report reserved memory (what the allocator holds), not only allocated memory.

### Gaps
- No paper reports the accuracy cost of resizing a running stream's budget mid-stream (shrinking a cache that has already grown), which is what an elastic memory manager would do.
- I did not compare these methods on one benchmark; OVGGT Table 1 and StreamCacheVGGT Table 1 contain head-to-head numbers that I did not extract.

## Q4. Is there systems work on serving such models to many users, or on KV-cache management for vision transformers that transfers?

### Takeaway
I found no paper that serves streaming 3D reconstruction models to multiple concurrent users, measures per-user memory under multi-tenancy, or schedules their caches across users.
All the cache work above is single-stream, single-GPU; the transferable systems ideas come from LLM serving.

### Cited Findings
- A search for multi-user or multi-tenant serving of streaming 3D reconstruction returned only single-stream cache papers (StreamCacheVGGT, XStreamVGGT, MeMix, SceneBound) and LLM serving papers - [web search results, 2026-11-18: arXiv 2611.00756, 2604.15237, 2603.15330, 2601.01204](https://arxiv.org/abs/2611.00756)
- Several of the 3D cache papers explicitly borrow LLM methods: Evict3R follows "layer-wise budget allocation driven by attention sparsity" and cites H2O and Scissorhands - [arXiv 2509.17650 HTML](https://arxiv.org/html/2509.17650); StreamCacheVGGT cites H2O, AdaKV and SnapKV - [arXiv 2604.15237 HTML](https://arxiv.org/html/2604.15237)
- Transfer source, paging: PagedAttention / vLLM, "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023, arXiv 2309.06180 - [arXiv](https://arxiv.org/abs/2309.06180)
- Transfer source, sink plus recent window: StreamingLLM, "Efficient Streaming Language Models with Attention Sinks", ICLR 2024, arXiv 2309.17453 - [arXiv](https://arxiv.org/abs/2309.17453); the first-frame anchors in InfiniteVGGT and XStreamVGGT ("tokens from the first and current frames are always preserved") play the same role - [arXiv 2601.01204](https://arxiv.org/abs/2601.01204)
- Transfer source, attention-score eviction: H2O, arXiv 2306.14048 (NeurIPS 2023 per Evict3R's bibliography) - [arXiv](https://arxiv.org/abs/2306.14048)
- Transfer source, KV offload and streaming between GPU and host: DéjàVu, arXiv 2403.01876 (venue not checked) - [arXiv](https://arxiv.org/abs/2403.01876)

### Inferences
- The direct transfer for the project's "grow a container's GPU budget with its cache" idea is a vLLM-style paged KV pool inside one serving process shared by all users, rather than per-container limits: this repo's nano4 probes (jobs 704421/704422, recorded in project memory, not a primary source here) found that a running MPS client could not grow after its server-side memory limit was raised, so growing a per-process cap would need a restart or migration.
- The 3D models differ from LLMs in two ways that matter for a manager: a frame adds about 1,000 tokens at once (about 0.1 GB for StreamVGGT), so allocation granularity is coarse; and space-indexed models (Point3R, StreamSplat 2608.01659, SceneBound) stop growing when a user revisits, so growth is not a function of time alone.

### Gaps
- I did not search ACM MM, MobiSys, SenSys or EuroSys systematically for multi-user streaming reconstruction serving; the null result covers web and arXiv searches only.
- Video-LLM streaming KV-cache work (for example ReKV, StreamMem, InfiniPot-V) was not checked and may be a closer transfer source than text LLMs.

## Q5. Code, weights and license: which can legally be run for a course project, and which run on a 48 GB RTX 6000 Ada?

### Takeaway
Almost all released models are non-commercial (CC BY-NC-SA 4.0, S-Lab 1.0, NAVER non-commercial, or StreamVGGT weights CC BY-NC 4.0), which permits non-commercial academic use such as a course project but forbids commercial reuse and, for share-alike, requires modified code to carry the same license if redistributed.
StreamVGGT and its bounded variants, STream3R, CUT3R/TTT3R, Spann3R, Point3R, MUSt3R, WinT3R and ReCoSplat have released code and weights; LONG3R, LongSplat, StreamGS, OF3GS, StreamCacheVGGT and StreamSplat (2608.01659) do not.

### Cited Findings
- StreamVGGT, Evict3R, OVGGT and XStreamVGGT all ship the StreamVGGT CC BY-NC-SA 4.0 text as their LICENSE; StreamVGGT weights on HF are tagged cc-by-nc-4.0 - [StreamVGGT LICENSE.txt](https://github.com/wzzheng/StreamVGGT/blob/main/LICENSE.txt); [HF lch01/StreamVGGT](https://huggingface.co/lch01/StreamVGGT)
- InfiniteVGGT and FrameVGGT code is Apache-2.0, but both load the StreamVGGT checkpoint, so the weights stay CC BY-NC - [InfiniteVGGT README](https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT); [FrameVGGT README](https://github.com/ZhisongXu/FrameVGGT)
- STream3R: "S-Lab License 1.0 ... Redistribution and use for non-commercial purpose" - [STream3R LICENSE](https://github.com/NIRVANALAN/STream3R/blob/main/LICENSE)
- MUSt3R: NAVER Non-Commercial License, and the README warns that "The mapfree dataset in particular, which was used to train all models, has a very restrictive license" - [MUSt3R README](https://github.com/naver/must3r)
- VGGT code was relicensed on July 29 2025 to permit commercial use, but "only the newly released checkpoint VGGT-1B-Commercial is licensed for commercial usage - the original checkpoint remains non-commercial"; the commercial checkpoint is gated by an application form - [VGGT README](https://github.com/facebookresearch/vggt)
- WinT3R: repo LICENSE says "WinT3R for non-commercial purposes", while the HF weights are tagged apache-2.0 - [WinT3R LICENSE](https://github.com/LiZizun/WinT3R/blob/main/LICENSE); [HF lizizun/WinT3R](https://huggingface.co/lizizun/WinT3R)
- ReCoSplat: the GitHub repo has no license file (GitHub API returns none), and the HF model is tagged MIT; released checkpoint is 224x224 - [ReCoSplat repo](https://github.com/peelstnac/ReCoSplat); [HF chfrn/ReCoSplat](https://huggingface.co/chfrn/ReCoSplat)
- LONG3R has no license and no code ("Code Coming Soon") - [LONG3R repo](https://github.com/zgchen33/LONG3R)
- GPUs the authors themselves used for inference: StreamVGGT A800; STream3R H200 (README table) and A100 (paper FPS); Evict3R GPU not stated in the text I read; OVGGT RTX 5090 32 GB; XStreamVGGT A100 80 GB; FrameVGGT RTX A6000; TTT3R "a single 48GB NVIDIA GPU"; LONG3R RTX 3090 24 GB; VGGT-Long RTX 4090 24 GiB; Spann3R RTX 4090; MUSt3R A100; ReCoSplat RTX 6000 Ada; StreamSplat (2608.01659) RTX PRO 6000 96 GB - sources as in the Q2 and Q3 tables above.

### Inferences
- On a 48 GB RTX 6000 Ada, every model with released weights runs a single stream; the full-cache causal modes are the only ones that will OOM within a session (roughly 390 frames for StreamVGGT and 210 for STream3R causal, by the extrapolation in Q2), which is exactly the behaviour a growing-budget study needs.
- ReCoSplat was benchmarked by its authors on the same GPU model as the project's local card and outputs Gaussians, so it is the lowest-friction growing-state 3DGS tenant; its code license is unclear and should be confirmed with the authors before redistribution.
- A course project that only runs these models and publishes numbers fits non-commercial terms; publishing modified StreamVGGT-derived code would need CC BY-NC-SA on the modifications. This is my reading of the license texts, not legal advice.

### Gaps
- I did not test-run any of these models; "runs on 48 GB" is inferred from authors' GPUs and the memory curves.
- I did not read the full NAVER, S-Lab or VGGT license texts beyond their opening clauses.
- I did not check the CUT3R or Spann3R Google Drive checkpoints for separate license terms.

### Candidate project questions (suggestions only, not pre-registrations)

These are my suggestions for discussion; each would need its own plan-mode pre-registration before any run.

1. Suggestion: How many StreamVGGT users can one 48 GB RTX 6000 Ada serve at a per-frame latency target under three memory policies: (a) static reservation of each user's worst-case cache, (b) on-demand growth inside one serving process with a shared paged KV pool, (c) a fixed per-user cache budget via an existing bounded variant (OVGGT or InfiniteVGGT, same checkpoint)?
   Step 0 is cheap: measure one user's VRAM and latency versus frames on the local card and check it against the published 0.117 GB and 3.4 ms per frame (StreamVGGT Table 8).
   Null: policy (a). Trade-off metric to report alongside users served: camera-pose error or depth AbsRel on 7-Scenes, since (c) trades accuracy.
2. Suggestion: Is per-user memory better predicted by frames seen or by area covered?
   Run the same scene twice (a revisit trajectory) through StreamVGGT (grows with frames) and a space-indexed policy (SceneBound-style keyframe gating, or Point3R) and measure whether the manager's growth forecast based on frame count over-reserves.
   This decides whether a "grow with stream length" controller is the right abstraction at all.
3. Suggestion: When the GPU is full, is it cheaper to shrink a tenant's cache budget, offload its cache to host memory, or reject the newcomer?
   Measure accuracy loss for shrinking (budget sweep on one bounded variant), resume latency for offload (cache size about 0.1 GB per frame, so the transfer time must be measured on this machine), and admission rate for rejection, all on the same trace.
