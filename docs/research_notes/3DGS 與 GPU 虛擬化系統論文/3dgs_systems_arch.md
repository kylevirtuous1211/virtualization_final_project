# 3DGS, NeRF and neural reconstruction in systems and computer-architecture venues (2022 to November 2026)

Written 2026-11-18 for the virtualization course project (many users doing live feed-forward 3DGS reconstruction on one shared GPU).
Network streaming of 3DGS, generic GPU-sharing systems, cloud gaming and KV-cache streaming reconstruction are out of scope; the streaming papers already verified in `docs/reading_notes.md` (L3GS and Vega at MobiCom 2025, LTS at MMSys 2025, TIGAS, AirGS and others) are cited from there, not re-surveyed.
Method: WebSearch for discovery; venue claims checked against the arXiv abstract page (title, comments, journal-ref) by script, and author PDFs or arXiv HTML read for the papers whose numbers matter most.
Verification marks used below:
- [P] I read the number or claim in the primary source (paper PDF, arXiv HTML, or the arXiv abstract itself).
- [A] The venue comes only from the authors' own arXiv comment or journal-ref field; the paper body was not read beyond the abstract.
- [S] Secondary only (search snippet, library catalogue, lab news page, another paper's reference list); not verified from a primary source.

## Q1. Which 3DGS / NeRF papers appeared at systems and architecture venues, with venue and year?

### Takeaway
The architecture community (ISCA, MICRO, HPCA, ASPLOS, DAC, ICCAD, DATE) has published at least 35 neural-rendering papers since 2022, almost all of them single-device accelerators or GPU micro-architecture changes for rendering, plus a 2025-2026 wave of training systems (CLM, GS-Scale, ARC, GSArch, Cambricon-GS).
The classic OS, networking and mobile systems venues (OSDI, SOSP, EuroSys, NSDI, ATC, MobiSys, SenSys) have no 3DGS paper I could find outside network streaming; MLSys 2026 has one (SwiftGS) and HPDC 2026 one poster (Odin).
Only one systems-venue paper targets feed-forward (generalizable) 3DGS reconstruction: SCARF (MICRO 2026), an edge encoder accelerator.

### Cited Findings

#### A. Peer-reviewed at systems / architecture venues: 3DGS rendering

| Paper | Venue, year | arXiv / DOI | Exact claim borrowed | Evaluation design | Code | Check |
| --- | --- | --- | --- | --- | --- | --- |
| [GSCore: Efficient Radiance Field Rendering via Architectural Support for 3D Gaussian Splatting](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf) | ASPLOS 2024 | DOI [10.1145/3620666.3651385](https://doi.org/10.1145/3620666.3651385); no arXiv | Sorting + rasterization are about 80% of rendering time on Jetson Xavier NX and RTX 3090; preprocessing and sorting are memory-bandwidth-bound, rasterization is compute-bound (roofline, Train/30K) | Profiling on Xavier NX and RTX 3090; RTL synthesised at 28 nm (3.95 mm², 272 KB SRAM, 51.2 GB/s, 64 VR cores); scenes from Tanks&Temples, DeepBlending, Synthetic-NSVF; 15.86x average speedup over Xavier NX (91.2 vs 6.4 FPS on the Fig. 1 scene, simulated) | No release statement found in the PDF | [P] |
| [Neo: Real-Time On-Device 3DGS with Reuse-and-Update Sorting Acceleration](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf) | ASPLOS 2026 | [arXiv 2511.12930](https://arxiv.org/abs/2511.12930); DOI 10.1145/3779212.3790192 | Sorting is the major bottleneck at high resolution because of memory bandwidth; GSCore drops from 66.7 FPS (HD) to 31.1 (FHD) and 15.8 (QHD); Gaussian depth order is temporally stable across frames | ASIC vs Orin AGX GPU and GSCore at HD/FHD/QHD; ASPLOS PDF reports up to 12.4x and 5.5x throughput and 97.7 FPS average at QHD; the arXiv v1 abstract says 10.0x and 5.6x and 94.5% / 81.3% less DRAM traffic (the two versions disagree) | [casys-kaist/Neo](https://github.com/casys-kaist/Neo) | [P] |
| [Lumina: Real-Time Mobile Neural Rendering by Exploiting Computational Redundancy](https://chenc10.github.io/assets/pdf/2025_isca_lumina.pdf) | ISCA 2025 | [arXiv 2506.05682](https://arxiv.org/abs/2506.05682); DOI 10.1145/3695053.3731003 | On a mobile Volta GPU (Xavier) 3DGS runs at 5-21 FPS; sorting + rasterization exceed 90% of execution; threads stay masked over 69% of the time (warp divergence); only about 10.3% of Gaussians are significant per pixel | Algorithm (sort skipping, radiance cache) plus LuminCore accelerator; 4.5x speedup and 5.3x energy reduction vs mobile Volta GPU, under 0.2 dB PSNR loss | Not found | [P] |
| [Efficient 3DGS with Axis-Shared Rasterization and Order-independent Transmittance](https://arxiv.org/abs/2506.07069) | ISCA 2026 | arXiv 2506.07069 | Axis-shared rasterization cuts MACs by up to 38%; a small MLP approximates transmittance so explicit sorting can be removed | 1.33-1.88x over prior 3DGS accelerators | [WangZhican/ISCA26_3DGS_Acc](https://github.com/WangZhican/ISCA26_3DGS_Acc) | [A] |
| [GCC: A 3DGS Inference Architecture with Gaussian-Wise and Cross-Stage Conditional Processing](https://arxiv.org/abs/2507.15300) | MICRO 2025 | arXiv 2507.15300 | Many preprocessed Gaussians are never used in rendering, and the same Gaussian is reloaded for every tile it touches | 28 nm accelerator vs GSCore (performance, energy) | Not stated | [A] |
| [DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization](https://arxiv.org/abs/2608.02099) | MICRO 2026 | arXiv 2608.02099 | Adding PEs to existing 3DGS accelerators gives marginal gains because the coupled checking-while-blending dataflow leaves PEs idle | 28 nm; 2.36-7.25x throughput, 1.82-6.02x end-to-end, 1.59-4.42x energy efficiency over GSCore, GBU and GCC at 720p to 8K; over 80% PE utilisation from 16 to 1024 PEs at high resolution | Not stated | [A] |
| [VR-Pipe: Streamlining Hardware Graphics Pipeline for Volume Rendering](https://arxiv.org/abs/2502.17078) | HPCA 2025 | arXiv 2502.17078 | 3DGS on the fixed-function graphics pipeline lacks early termination; repurposed hardware plus multi-granular tile binning with quad merging fixes it | 3DGS implemented with graphics APIs on real GPUs, then modelled changes; up to 2.78x over the conventional graphics pipeline | Not stated | [A] |
| [Gaussian Blending Unit (GBU): An Edge GPU Plug-in for Real-Time Gaussian-Based Rendering in AR/VR](https://arxiv.org/html/2503.23625) | HPCA 2025 | arXiv 2503.23625 | On Orin NX 16 GB, 3DGS reaches only 7-17 FPS; blending is 70-78% of time for static scenes (sorting 14-24%); row imbalance leaves 18.9% thread-lane utilisation | Static (Mip-NeRF 360), dynamic (4D-GS) and avatar scenes; RTL at 28 nm (0.90 mm², 0.22 W, replaces one SM); GPGPU-Sim emulator validated within 10% of Jetson; 92 / 80 / 102 FPS and 10.8x / 4.4x / 2.5x energy efficiency | Not stated | [P] |
| [Uni-Render: A Unified Accelerator for Real-Time Rendering Across Diverse Neural Renderers](https://arxiv.org/abs/2503.23644) | HPCA 2025 | arXiv 2503.23644 | Different neural renderers share common operators, so one reconfigurable accelerator can serve several pipelines | Synthetic and real scenes, edge setting | Not stated | [A] |
| [GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering](https://arxiv.org/abs/2601.20429) | HPCA 2026 | arXiv 2601.20429 | Anisotropic Gaussians can be traced as unit spheres after a ray-space transform, shrinking the BVH; traversal checkpointing in RT units avoids restarts | Software plus RT-unit hardware change vs baseline Gaussian ray tracing | Not stated | [A] |
| [GSPIM: A Temporal-Locality-Aware 4DGS Accelerator with Polymorphic PIM](https://arxiv.org/abs/2611.04804) | HPCA 2027 (accepted Nov 2026) | arXiv 2611.04804 | Frame-isolated 4DGS rendering reloads largely overlapping active sets across adjacent frames | GPU-PIM model calibrated to Jetson AGX Orin 64 GB; 3 4DGS methods, 6 scenes; 4.15x geomean speedup, 4.99x energy efficiency | [MAdrid1011/GSPIM](https://github.com/MAdrid1011/GSPIM) | [A] |
| [GauRast: Enhancing GPU Triangle Rasterizers to Accelerate 3DGS](https://arxiv.org/abs/2503.16681) | DAC 2025 | arXiv 2503.16681 | The existing GPU triangle rasterizer can be extended to rasterize Gaussians at 0.2% SoC area | 23x faster and 24x less energy for the rasterization step; 6x end-to-end (24 FPS) for original 3DGS and 4x (46 FPS) for an efficient variant ([NVIDIA page](https://research.nvidia.com/publication/2025-06_gaurast-enhancing-gpu-triangle-rasterizers-accelerate-3d-gaussian-splatting)) | Not found | [A] |
| [GS-TG: 3DGS Accelerator with Tile Grouping](https://arxiv.org/abs/2509.00911) | DAC 2025 | arXiv 2509.00911 | Large tiles cut redundant sorting but add rasterization work; grouping small tiles for sorting gets both | 1.54x average over prior 3DGS accelerators | Not stated | [A] |
| [GSAcc: Accelerate 3DGS via Depth Speculation and Gaussian-centric Rasterization](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-a179t-4cf6d8d12f9dac0fa7f22b6de5d7f312721c9278bcb3a40c5cb1ad5bcbdc5bea3?lng=en) | DAC 2025 | none found | Up to 2.3x PPA and 2.9x energy vs GSCore | Not read | Unknown | [S] |
| [AdaGScale: Viewpoint-Adaptive Gaussian Scaling to Reduce Gaussian-Tile Pairs](https://arxiv.org/abs/2604.18980) | DAC 2026 | arXiv 2604.18980 | Peripheral tiles of a Gaussian contribute negligibly, so shrinking Gaussians for the tile test only cuts Gaussian-tile pairs | 13.8x geomean over original 3DGS on a GPU, about 0.5 dB loss on city-scale scenes | [askmgk/AdaGScale](https://github.com/askmgk/AdaGScale) | [A] |
| [GEMM-GS: Accelerating 3DGS on Tensor Cores with GEMM-Compatible Blending](https://arxiv.org/abs/2604.02120) | DAC 2026 | arXiv 2604.02120 | Blending can be reformulated exactly as GEMM to use idle Tensor Cores | 1.42x over vanilla 3DGS, a further 1.47x when stacked on other accelerations | [shieldforever/GEMM-GS](https://github.com/shieldforever/GEMM-GS) | [A] |
| [LS-Gaussian: No Redundancy, No Stall](https://arxiv.org/abs/2507.21572) | ICCAD 2025 | arXiv 2507.21572 | Consecutive frames from a moving camera allow sparse re-rendering; tile workload imbalance causes stalls that can be predicted | Algorithm plus accelerator | Not stated | [A] |
| [FLICKER: A Fine-Grained Contribution-Aware Accelerator for Real-Time 3DGS](https://arxiv.org/abs/2603.01158) | DATE 2026 | arXiv 2603.01158 | Contribution testing per pixel group skips non-contributing Gaussians | 1.5x speedup, 2.6x energy efficiency, 14% less area than a prior accelerator; 19.8x and 26.7x over an edge GPU | Not stated | [A] |
| [SwiftGS (also listed as Flash3DGS): Algorithm and System Co-Optimization for Fast 3DGS on GPUs](https://proceedings.mlsys.org/paper_files/paper/2026/file/5f96a21345c138da929e99871fda138e-Paper-Conference.pdf) | MLSys 2026 ([poster](https://mlsys.org/virtual/2026/poster/3550), [oral](https://mlsys.org/virtual/2026/oral/3773)) | no arXiv | On RTX 3090, rasterization is 55.47% and sorting 27.31% of rendering time on average; the Gaussian-tile intersection list can be up to 7.61x longer than the number of Gaussians | RTX 3090 and L40S; seven Mip-NeRF 360 scenes at 7k and 30k checkpoints; vs gsplat; 1.11-1.41x speedup; artifact-evaluation badges | [LingjunGao/SwiftGS](https://github.com/LingjunGao/SwiftGS) | [P] |

#### B. Peer-reviewed at systems / architecture venues: 3DGS training, SLAM and feed-forward reconstruction

| Paper | Venue, year | arXiv / DOI | Exact claim borrowed | Evaluation design | Code | Check |
| --- | --- | --- | --- | --- | --- | --- |
| [CLM: Removing the GPU Memory Barrier for 3DGS](https://arxiv.org/html/2511.04951v1) | ASPLOS 2026 | arXiv 2511.04951 | A Gaussian has 59 parameters stored 4 times (value, gradient, two Adam states), so model state is N x 59 x 4 x 4 bytes; a 24 GB RTX 4090 holds at most about 26 M Gaussians; each view touches under 1% of a large scene (0.39% average on MatrixCity BigCity) | RTX 4090 (PCIe 4.0) and RTX 2080 Ti (PCIe 3.0); Bicycle, Rubble, Alameda, Ithaca365, BigCity; baselines Grendel single-GPU, an enhanced baseline with frustum culling, naive ZeRO-Offload-style offloading; 102.2 M Gaussians on one 4090; 55-97% of GPU-only throughput | [nyu-systems/CLM-GS](https://github.com/nyu-systems/CLM-GS) | [P] |
| [GS-Scale: Unlocking Large-Scale 3DGS Training via Host Offloading](https://99dhl.github.io/assets/pdf/asplos26_gsscale.pdf) | ASPLOS 2026 | [arXiv 2509.15645](https://arxiv.org/abs/2509.15645); DOI 10.1145/3779212.3790167 | Keeping all Gaussians in host memory and moving only the visible subset cuts GPU memory 3.3-5.6x at near GPU-only speed | Large-scale datasets; 4 M to 18 M Gaussians on an RTX 4070 Mobile, 23-35% LPIPS improvement | [SNU-ARC/GS-Scale](https://github.com/SNU-ARC/GS-Scale) | [P] (abstract and first page) |
| [ARC: Warp-level Adaptive Atomic Reduction in GPUs to Accelerate Differentiable Rendering](https://people.mpi-sws.org/~cgiannoula/assets/publications/ARC_asplos25_full.pdf) | ASPLOS 2025 | DOI 10.1145/3669940.3707238; predecessor [DISTWAR, arXiv 2401.05345](https://arxiv.org/abs/2401.05345) | Gradient computation in 3DGS training issues so many atomics that the L2 atomic units saturate and stall | ARC-HW and ARC-SW; 2.6x average, up to 5.7x on differentiable rendering workloads ([author page](https://pawks.github.io/publication/asplos-2025)) | Not checked | [S] (abstract only; PDF did not parse) |
| [GSArch: Breaking Memory Barriers in 3DGS Training via Architectural Support](https://doi.org/10.1109/HPCA61900.2025.00037) | HPCA 2025 | DOI 10.1109/HPCA61900.2025.00037 | Memory traffic in 3DGS training needs architectural support (title-level only) | Not read | Unknown | [S] (DOI seen only in another paper's reference list) |
| [Cambricon-GS: An Accelerator for 3DGS Training with Gaussian-Pixel Hybrid Parallelism](https://2026.cgo.org/profile/jinli) | HPCA 2026 | none found | Title only | Not read | Unknown | [S] |
| [RTGS: Real-Time 3DGS SLAM via Multi-Level Redundancy Reduction](https://arxiv.org/abs/2510.06644) | MICRO 2025 | arXiv 2510.06644 | Atomics in rendering backpropagation and tile workload imbalance dominate 3DGS-SLAM on edge GPUs | Integrated into an edge GPU; at least 30 FPS on 4 datasets and 3 algorithms; up to 82.5x energy efficiency | [UMN-ZhaoLab/RTGS](https://github.com/UMN-ZhaoLab/RTGS) | [A] |
| [REACT3D: Real-time Edge Accelerator for Incremental Training in 3DGS-based SLAM](https://fact-lab.hkust.edu.hk/publications/conference-paper/2025/wang-2025-react-3-d/) | MICRO 2025 | not found | Title only | Not read | Unknown | [S] |
| [GauSPU: 3DGS Processor for Real-Time SLAM Systems](https://computer.org/csdl/proceedings-article/micro/2024/505700b562/22nirMP5znq) | MICRO 2024 | DOI 10.1109/MICRO61859.2024.00114 | Sparse tile sampling for pose tracking; GauSPU plus GPU reaches 33.6 FPS tracking, 63.9x energy efficiency | Not read | Unknown | [S] |
| [SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder Accelerator](https://arxiv.org/html/2611.02365) | MICRO 2026 (all three artifact badges) | arXiv 2611.02365 | For cost-volume feed-forward 3DGS encoders (MVSplat, DepthSplat, TranSplat) on Jetson Orin NX: depth search is over 65% of memory traffic with a 31.2% average L2 hit rate; depth prediction plus Gaussian generation is 54% of time; one encoder pass costs about 50x one rasterized view; over 52% of generated Gaussians are redundant | Re10K, ACID, DL3DV; baseline Orin NX with PyTorch, FP16, CUDA Graphs; 28 nm, 27.8 mm², 2.1 W; 2.94x geomean end-to-end speedup, 34.5x energy efficiency; the paper says no prior accelerator targets G-3DGS encoders | [MAdrid1011/SCARF](https://github.com/MAdrid1011/SCARF) | [P] |
| [BalanceGS: Algorithm-System Co-design for Efficient 3DGS Training on GPU](https://arxiv.org/abs/2510.14564) | ASP-DAC 2026 | arXiv 2510.14564 | Training suffers from skewed densification, imbalanced projection work and fragmented memory access in splatting | 1.44x training speedup on an A100 | Not stated | [A] |
| [Odin: Primitive-Level Synchronization for Distributed Point-Based Neural Rendering](https://arxiv.org/abs/2607.19893) | HPDC 2026 (poster) | arXiv 2607.19893 | In distributed 3DGS training, global barriers rather than rendering sit on the critical path | 4 pipelines, 13 scenes, 8 GPUs: 1.22x average throughput, 82% of critical-path wait hidden; up to 1.89x over Grendel at 64 GPUs | Not stated | [A] |

#### C. Peer-reviewed at systems / architecture venues: NeRF-era accelerators (cited, not surveyed in depth)

| Paper | Venue, year | arXiv / DOI | Exact claim borrowed | Check |
| --- | --- | --- | --- | --- |
| [Hardware Acceleration of Neural Graphics (NGPC)](https://passat.crhc.illinois.edu/isca23_a.pdf) | ISCA 2023 | [arXiv 2303.05735](https://arxiv.org/abs/2303.05735); DOI 10.1145/3579371.3589085 | On an RTX 3090, 4K at 60 FPS is 1.51-55.50x out of reach; input encoding plus MLP take 72.37%, 60.0% and 59.96% of time for hash-grid, dense-grid and low-res dense-grid encodings | [P] |
| [NeuRex: A Case for Neural Rendering Acceleration](https://junseo013.github.io/assets/pubs/isca23-neurex.pdf) | ISCA 2023 | (author PDF) | On commodity GPUs multi-resolution hash encoding takes more time than the MLP, and both are under-utilised; NeuRex-Edge and NeuRex-Server at 28 nm | [S] |
| [Instant-3D: Instant NeRF Training Towards On-Device AR/VR 3D Reconstruction](https://arxiv.org/abs/2304.12467) | ISCA 2023 | arXiv 2304.12467 | Grid interpolation and its backward dominate NeRF training; 41-248x training-time reduction, 1.6 s per scene at 1.9 W | [A] |
| [Gen-NeRF: Efficient and Generalizable NeRF via Algorithm-Hardware Co-Design](https://arxiv.org/abs/2304.11842) | ISCA 2023 | arXiv 2304.11842 | First accelerator for generalizable (feed-forward) NeRF; reuses data across rays via epipolar geometry | [A] |
| [Cicero: Radiance Warping and Memory Optimizations](https://arxiv.org/abs/2404.11852) | ISCA 2024 ([author page](https://cs.sjtu.edu.cn/~leng-jw/pubs/feng24isca-cicero.html)) | arXiv 2404.11852 | NeRF runs at 0.8 FPS on a mobile Volta GPU; warping skips up to 88% of radiance work; 8.0x speedup in software | [P] (abstract) |
| [FlexNeRFer](https://arxiv.org/abs/2505.06504) | ISCA 2025 | arXiv 2505.06504 | 28 nm layout; 8.2-243.3x speedup over RTX 2080 Ti across NeRF variants | [A] |
| [Fusion-3D: Integrated Acceleration for Instant 3D Reconstruction and Real-Time Rendering](https://par.nsf.gov/servlets/purl/10574254) | MICRO 2024, best paper ([Georgia Tech news](https://www.cc.gatech.edu/news/framework-sets-new-benchmark-creating-3d-scenes-2d-images)) | none found | NeRF training and rendering on one multi-chip accelerator | [S] |
| [MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning and Foveated Rendering](https://arxiv.org/abs/2407.00435) | ASPLOS 2025 | arXiv 2407.00435; DOI 10.1145/3669940.3707227 | Point-based rendering made real time on mobile via pruning, foveation and an accelerator for the resulting load imbalance; user study for quality | [A]; code via [project page](https://horizon-lab.org/metasapiens/) |
| [ASDR: Adaptive Sampling and Data Reuse for CIM-based Instant Neural Rendering](https://arxiv.org/abs/2508.02304) | ASPLOS 2025 (presented 2026) | arXiv 2508.02304 | ReRAM compute-in-memory NeRF; up to 9.55x over NeRF accelerators, 69.75x over Xavier NX | [A] |
| [Instant-NeRF (near-memory processing)](https://arxiv.org/abs/2305.05766) | DAC 2023 | arXiv 2305.05766 | Profiling shows NeRF training is memory-bound | [A] |
| [RT-NeRF](https://arxiv.org/abs/2212.01120) | ICCAD 2022 | arXiv 2212.01120 | Uniform point sampling and dense embedding access cause low NeRF throughput on AR/VR devices | [A] |
| [SpNeRF](https://arxiv.org/abs/2505.08191) | DATE 2025 | arXiv 2505.08191 | Voxel-grid NeRF rendering on edge is memory-bound; 21.07x memory reduction | [A] |

#### D. Systems-flavoured work at ML, vision or graphics venues (not systems venues)

- Grendel ("On Scaling Up 3D Gaussian Splatting Training") is ICLR 2025 Oral, not a systems venue; it distributes Gaussians across GPUs and pixels across GPUs with sparse all-to-all and dynamic load balancing, on Perlmutter (4 x A100 40 GB per node, up to 32 GPUs); an A100 40 GB holds up to 11.2 M Gaussians for 4K Rubble, 16 GPUs train 40.4 M Gaussians to 27.28 dB vs 26.28 dB at 11.2 M on one GPU; code [nyu-systems/Grendel-GS](https://github.com/nyu-systems/Grendel-GS) - [arXiv 2406.18533](https://arxiv.org/abs/2406.18533), [HTML](https://arxiv.org/html/2406.18533v2) [P]
- gsplat is a JMLR MLOSS library paper; since v1.5.x its rasterizer renders a batch of scenes, not only a batch of cameras, in one call - [arXiv 2409.06765](https://arxiv.org/abs/2409.06765) [A]; [release note](https://radiancefields.com/gsplat-1-5-3-released-by-nerfstudio) [S]
- "A LoD of Gaussians" trains and renders ultra-large scenes out-of-core on one consumer GPU (journal-ref: Proceedings of SIGGRAPH 2026) - [arXiv 2507.01110](https://arxiv.org/abs/2507.01110) [A]
- TideGS (ICML 2026) trains over one billion Gaussians on a single 24 GB GPU by spreading parameters across SSD, CPU and GPU memory - [ICML page](https://icml.cc/virtual/2026/poster/64763) [S]
- Mobile-GS (ICLR 2026) removes depth sorting with order-independent rendering for phones - [arXiv 2603.11531](https://arxiv.org/abs/2603.11531), [mlanthology](https://mlanthology.org/iclr/2026/du2026iclr-mobilegs/) [S for venue]
- Faster-GS (CVPR 2026) consolidates training optimisations to up to 5x faster training - [arXiv 2602.09999](https://arxiv.org/abs/2602.09999), [CVPR page](https://cvpr.thecvf.com/virtual/2026/poster/36550) [S for venue]
- A 129 FPS full-HD 3DGS accelerator (TSMC 28 nm, 0.66 mm², 0.219 W) appeared in IEEE TVCG 2026 - [arXiv 2604.10223](https://arxiv.org/abs/2604.10223) [A]

#### E. arXiv-only (no venue stated as of 2026-11-18) with systems content

- "From Splats to Silicon" (Pei et al., cs.AR, Sep 2026), a workload-centric survey plus controlled profiling: gsplat cuts the median Gaussian-tile association count by about 42% relative to the original (Graphdeco) rasterizer, yet complete forward time falls only 4% and DRAM traffic rises 19%; generic indexing kernels move a median 0.47 GB per view, about 27% of gsplat's DRAM traffic - [arXiv 2609.06157](https://arxiv.org/html/2609.06157) [P]
- RoofGS: a stage-wise roofline on RTX 4090 at 4K finds the front end bound by global memory traffic and rasterization bound by instruction throughput; 10.1x end-to-end (61 to 616 FPS) - [arXiv 2608.15785](https://arxiv.org/abs/2608.15785) [P abstract]
- TensorGS: rasterization dominates end-to-end time and is compute-bound, FP16 is sufficient, and Tensor Cores sit idle; 1.65x end-to-end - [arXiv 2605.17855](https://arxiv.org/abs/2605.17855) [P abstract]
- HiGS (NVIDIA authors): partitioning prefers large tiles and rasterization small ones, and a few dense tiles dominate frame time; up to 15.8x over original 3DGS - [arXiv 2606.00352](https://arxiv.org/abs/2606.00352) [P abstract]
- "Splats under Pressure": emulates weaker GPUs on an RTX 4090 with `nvidia-smi` power caps and clock locks (no MPS, MIG or SM limits); single workload per run; real-time (60 FPS) holds below about 600 k splats on the 4090, 4070 Ti and 3070 tiers - [arXiv 2604.07177](https://arxiv.org/html/2604.07177) [P]
- STREAMINGGS: 3DGS reaches only 2-9 FPS on mobile Ampere; memory-centric rendering gives up to 45.7x speedup - [arXiv 2506.09070](https://arxiv.org/abs/2506.09070) [P abstract]; a search snippet says DAC 2025 but the arXiv page states no venue [S]
- Texture3dgs: on mobile GPUs sorting dominates 3DGS and the 2D texture cache is the key resource - [arXiv 2511.16298](https://arxiv.org/abs/2511.16298) [P abstract]
- Vorion: a RISC-V GPGPU prototype with Gaussian rasterizers, TSMC 16 nm, 19 FPS rendering; 38.6 training iterations/s with 16 rasterizers - [arXiv 2511.16831](https://arxiv.org/abs/2511.16831) [P abstract]
- Splatonic (3DGS-SLAM, up to 274.9x over mobile GPUs) - [arXiv 2511.18755](https://arxiv.org/abs/2511.18755); AGS (codec-assisted SLAM) - [arXiv 2509.00433](https://arxiv.org/abs/2509.00433) [P abstract]
- Distributed training: ByteSplat (fused forward/backward rasterization and sparse gradient exchange, up to 6.1x on 8 GPUs) - [arXiv 2610.02851](https://arxiv.org/abs/2610.02851); Splaxel (pixel-level exchange, up to 7.6x over the state of the art) - [arXiv 2606.18588](https://arxiv.org/abs/2606.18588); Gaian (general PBDR training system, up to 128 GPUs, 91% less communication) - [arXiv 2512.20017](https://arxiv.org/abs/2512.20017); a PyTorch multi-GPU device over unified memory and NVLink, over 1 billion splats - [arXiv 2606.11390](https://arxiv.org/abs/2606.11390) [P abstracts]
- On-device training and rendering: PocketGS trains 3DGS on a smartphone in about 255.2 s post-capture under 3 GB peak memory - [arXiv 2601.17354](https://arxiv.org/abs/2601.17354); LiteGS up to 13.4x faster training - [arXiv 2503.01199](https://arxiv.org/abs/2503.01199) [P abstracts]
- Cloud-client: Voyager (temporal-aware LoD search, up to 6.6x on mobile) - [arXiv 2506.02774](https://arxiv.org/abs/2506.02774); Nebula (cloud LoD search plus client stereo rasterization, 2.7x motion-to-photon) - [arXiv 2512.20495](https://arxiv.org/abs/2512.20495); NEPHELE (multi-user cloud NeRF, below) - [arXiv 2303.04086](https://arxiv.org/abs/2303.04086) [P abstracts]

#### F. Venues searched with no non-streaming 3DGS/NeRF hit

- I searched arXiv comment fields for MLSys, EuroSys, OSDI, SOSP, NSDI, ATC, MobiSys, MobiCom, SenSys, ISSCC, IISWC and ISPASS combined with Gaussian splatting / radiance field / NeRF, and got no hits (arXiv API, 2026-11-18); targeted web searches for each venue found only MLSys 2026 SwiftGS and, at MobiSys 2025, a two-page RF-propagation abstract that uses 3DGS as a keyword - [ACM DL 10.1145/3711875.3736669](https://dl.acm.org/doi/10.1145/3711875.3736669) [S]
- MobiCom's 3DGS papers (L3GS 2025, Vega 2025) are network streaming and already in `docs/reading_notes.md`.

### Inferences
- The architecture community treats 3DGS as a single-frame, single-device rendering kernel to be made cheaper; the shared baseline is GSCore, and the shared platform is a Jetson-class edge GPU.
- Training work since 2025 is about memory capacity (CLM, GS-Scale, TideGS, LoD of Gaussians) and multi-GPU communication (Grendel, Odin, ByteSplat, Splaxel, Gaian), not about sharing one GPU among jobs.
- Feed-forward reconstruction, which is the project's workload, has one accelerator paper (SCARF, MICRO 2026) and one NeRF predecessor (Gen-NeRF, ISCA 2023); neither runs on a datacenter GPU or with more than one stream.

### Gaps
- I could not read GSArch, Cambricon-GS, REACT3D, GauSPU, GSAcc, Fusion-3D or NeuRex beyond titles, abstracts or news pages; their numbers above are unverified.
- The ISCA 2026 and MICRO 2026 lists were not checked against the official programmes, so a 3DGS paper without an arXiv comment naming the venue could be missing.
- The arXiv comment field is author-written and incomplete (SwiftGS has no arXiv entry at all), so the venue search in F can miss papers.

## Q2. What do characterisation studies say about where time and memory go in 3DGS rendering, training and feed-forward reconstruction?

### Takeaway
For rendering, every characterisation agrees that sorting plus rasterization/blending take 80-90% of frame time, with preprocessing and sorting memory-bandwidth-bound and rasterization compute- or instruction-bound, and with low lane utilisation from divergence and tile imbalance.
For training, the binding resources are memory capacity (about 944 bytes of model state per Gaussian), atomic gradient traffic, and in multi-GPU runs, communication and synchronisation.
For feed-forward reconstruction, the only systems characterisation (SCARF, edge GPU) finds the encoder costs about 50x a rendered view and is dominated by depth search with poor cache reuse, so the rendering-centric literature mostly optimises the cheap part of a live reconstruction pipeline.

### Cited Findings
- Rendering, edge and desktop GPU: sorting and rasterization are about 80% of rendering time on both Jetson Xavier NX and RTX 3090; the edge GPU renders at a few FPS while the desktop GPU exceeds 100 FPS - [GSCore, ASPLOS 2024](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf) [P]
- Rendering roofline: preprocessing is memory-bandwidth-bound (loads many parameters per Gaussian with no reuse), rasterization is compute-bound (Gaussians reused within a tile), and sorting is memory-bandwidth-bound (multiple passes over more than 4 M indices and depths) - [GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf) [P]
- Wasted work: 18.50-54.59% of Gaussians assigned to a tile are never used because of early termination (scene averages), up to 90-98% in some tiles - [GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf) [P]
- On RTX 3090 rasterization is 55.47% and sorting 27.31% of rendering time on average - [SwiftGS, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/file/5f96a21345c138da929e99871fda138e-Paper-Conference.pdf) [P]
- On a mobile Volta GPU sorting plus rasterization exceed 90% of execution, threads are masked over 69% of the time, and only about 10.3% of Gaussians are significant - [Lumina, ISCA 2025](https://chenc10.github.io/assets/pdf/2025_isca_lumina.pdf) [P]
- On Orin NX blending is 70-78% (static), 62-65% (dynamic) and 48-51% (avatar) of rendering time, and row imbalance leaves 18.9% lane utilisation - [GBU, HPCA 2025](https://arxiv.org/html/2503.23625) [P]
- At high resolution sorting becomes the bottleneck because of memory bandwidth - [Neo, ASPLOS 2026](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf) [P]
- On RTX 4090 at 4K the front end is bound by global memory traffic and rasterization by instruction throughput - [RoofGS, arXiv 2608.15785](https://arxiv.org/abs/2608.15785) [P abstract]
- Reducing work counts does not translate one-to-one into time: a 42% cut in associations gave a 4% cut in forward time with 19% more DRAM traffic - [From Splats to Silicon, arXiv 2609.06157](https://arxiv.org/html/2609.06157) [P]
- Accelerator scaling: adding PEs to existing 3DGS accelerators gives marginal gains because PEs idle under irregular coverage and asynchronous per-pixel termination - [DeGS, MICRO 2026](https://arxiv.org/abs/2608.02099) [A]
- Training memory: model state is N x 59 x 4 x 4 bytes, so a 24 GB GPU holds at most about 26 M Gaussians, and one view touches under 1% of a large scene - [CLM, ASPLOS 2026](https://arxiv.org/html/2511.04951v1) [P]
- Training memory on A100 40 GB: up to 11.2 M Gaussians for 4K Rubble; 90% of Gaussians have a radius under 2% of image width, which bounds communication - [Grendel, ICLR 2025](https://arxiv.org/html/2406.18533v2) [P]
- Training compute: the gradient phase saturates the L2 atomic units - [ARC, ASPLOS 2025](https://pawks.github.io/publication/asplos-2025) [S]; [DISTWAR, arXiv 2401.05345](https://arxiv.org/abs/2401.05345) [P abstract]
- Distributed training: off-chip traffic in forward and backward rasterization and inter-GPU gradient exchange dominate - [ByteSplat, arXiv 2610.02851](https://arxiv.org/abs/2610.02851) [P abstract]; global barriers sit on the critical path - [Odin, HPDC 2026 poster](https://arxiv.org/abs/2607.19893) [A]
- Feed-forward encoders on Jetson Orin NX: depth search is over 65% of memory traffic with a 31.2% average L2 hit rate, depth prediction plus Gaussian generation is 54% of time, an encoder pass costs about 50x one rasterized view, and over 52% of generated Gaussians are redundant - [SCARF, MICRO 2026](https://arxiv.org/html/2611.02365) [P]
- Feed-forward backbones (non-systems venues): in VGGT, global attention increasingly dominates as frame count grows - [FastVGGT, arXiv 2509.02560](https://arxiv.org/abs/2509.02560) [S]
- NeRF era, for contrast: encoding plus MLP take 60-72% of time on RTX 3090 - [NGPC, ISCA 2023](https://passat.crhc.illinois.edu/isca23_a.pdf) [P]; NeRF training is memory-bound - [Instant-NeRF, DAC 2023](https://arxiv.org/abs/2305.05766) [A]

### Inferences
- The project's measured numbers (StreamSplat 0.19 s and 4 GiB per update, about 5 updates/s per RTX 6000 Ada; LOG.md G5, G6) are of the same shape as SCARF's: the per-user cost is the DNN encoder, and rendering is one to two orders of magnitude cheaper.
- That makes the bulk of the 3DGS systems literature (sorting, blending, tile binning) relevant to the browser client's rendering, not to the server's capacity.
- SCARF's edge-GPU finding (memory-traffic-heavy depth search, poor L2 reuse) does not transfer automatically to a 48 GB GDDR6 datacenter GPU where the project measured compute saturation; that transfer is untested.
- Low lane utilisation inside 3DGS kernels (18.9% in GBU, over 69% masked in Lumina) suggests that a single 3DGS rendering job may leave SM issue slots idle even when the GPU reports 100% "utilisation"; this is a hypothesis, not a measured fact for this project.

### Gaps
- No primary source characterises a feed-forward 3DGS model (AnySplat, StreamSplat, MVSplat, DepthSplat) on a datacenter GPU with kernel-level roofline data.
- No paper reports SM-count scaling (throughput vs number of SMs) for any 3DGS stage on a GPU; DeGS reports PE scaling only for accelerators.
- IISWC and ISPASS characterisation papers on 3DGS were not found; I cannot say none exist.

## Q3. Does any of this work consider several users or scenes sharing one GPU, or serving 3DGS to many clients?

### Takeaway
No paper in a systems or architecture venue evaluates several users, scenes or tenants sharing one GPU for 3DGS or NeRF.
The nearest things are a 2023 arXiv-only multi-user cloud NeRF platform (NEPHELE), library-level batching of scenes in gsplat, view batching inside one training job (Grendel), and passing remarks.

### Cited Findings
- NEPHELE builds a cloud NeRF platform with render farms, a task assigner, a frame composer and a ray-level task scheduler so that several people can view the same scene; it claims an order-of-magnitude efficiency gain over i-NGP "especially for the multi-user multi-viewpoint setting"; arXiv only (cs.GR, cs.DC), no venue found, and no scheduler numbers on the abstract page - [arXiv 2303.04086](https://arxiv.org/abs/2303.04086) [P abstract]
- gsplat's rasterizer accepts a batch of scenes in one pass as well as a batch of cameras - [gsplat 1.5.3 release note](https://radiancefields.com/gsplat-1-5-3-released-by-nerfstudio) [S]
- Grendel batches views from one scene across GPUs and gives learning-rate and momentum scaling rules for batched training; it does not mix scenes or tenants - [Grendel](https://arxiv.org/html/2406.18533v2) [P]
- GBU measures each application type alone and only remarks that multi-avatar settings could be limited by the GPU-side preprocessing step, without measuring it - [GBU](https://arxiv.org/html/2503.23625) [P]
- SCARF's "single hardware instance" claim refers to supporting several encoder models, not several concurrent users or streams; batching is not discussed - [SCARF](https://arxiv.org/html/2611.02365) [P]
- "Splats under Pressure" varies power caps and clocks for one workload at a time and never shares the GPU - [arXiv 2604.07177](https://arxiv.org/html/2604.07177) [P]
- From Splats to Silicon mentions that asynchronous overlap in reuse schemes "may introduce contention" but does not study multi-tenancy - [arXiv 2609.06157](https://arxiv.org/html/2609.06157) [P]
- Already in `docs/reading_notes.md`: TIGAS keeps multi-user VRAM residency with reference counting; Streaming Real-Time Rendered Scenes (arXiv 2604.02851) argues 3DGS amortises server work across users; MD2G-Cast, MuV2 and NETSPLAT share network delivery across users.

### Inferences
- Multi-tenant GPU sharing for 3DGS or feed-forward reconstruction is open in the systems literature; the project would not be re-doing a published systems study.
- The absence is partly structural: architecture papers target one AR/VR headset rendering one scene, so tenants never arise in their problem statement.

### Gaps
- A vendor blog claims an L40S serves "30+ concurrent client streams" of 3DGS at 1080p, with no method; I found no measurement to confirm it - [Spheron blog](https://www.spheron.network/blog/deploy-3d-gaussian-splatting-gpu-cloud/) [S, unverified]
- NEPHELE's full text (scheduler evaluation, users per GPU) was not read.

## Q4. How are these papers evaluated?

### Takeaway
Architecture papers report speedup and energy efficiency of RTL-synthesised designs (mostly 28 nm) or cycle-level simulators against a Jetson-class edge GPU and against GSCore, on the standard novel-view-synthesis scenes, with PSNR as the quality guard.
GPU-software and training papers report FPS, images/s or iterations/s and maximum Gaussian count against the original 3DGS rasterizer or gsplat on RTX 3090/4090, L40S or A100.
None reports per-user latency distributions, tail latency, users served per GPU, or interference between co-located jobs.

### Cited Findings
- Accelerator methodology: RTL with a commercial 28 nm library, area and power from synthesis, DRAM energy from a DRAM model, compared with Jetson Xavier NX - [GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf) [P]; 28 nm RTL plus a GPGPU-Sim emulator validated within 10% of Jetson Orin NX - [GBU](https://arxiv.org/html/2503.23625) [P]; 28 nm Chisel RTL plus simulator, baseline Orin NX with PyTorch FP16 and CUDA Graphs - [SCARF](https://arxiv.org/html/2611.02365) [P]; GPU-PIM model calibrated to Jetson AGX Orin - [GSPIM](https://arxiv.org/abs/2611.04804) [A]
- Accelerator baselines: GSCore is the common comparison point (GCC, Neo, FLICKER, DeGS, GSAcc all compare with it) - [GCC](https://arxiv.org/abs/2507.15300), [Neo](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf), [FLICKER](https://arxiv.org/abs/2603.01158), [DeGS](https://arxiv.org/abs/2608.02099) [A/P]
- Scenes: Tanks&Temples, DeepBlending, Synthetic-NSVF - [GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf) [P]; Mip-NeRF 360 static, 4D-GS dynamic and avatar scenes - [GBU](https://arxiv.org/html/2503.23625) [P]; Re10K, ACID, DL3DV for feed-forward - [SCARF](https://arxiv.org/html/2611.02365) [P]; Rubble, MatrixCity, Ithaca365, Alameda for large-scale training - [CLM](https://arxiv.org/html/2511.04951v1) [P]
- Resolution sweeps from HD to QHD (Neo) and 720p to 8K (DeGS) are used to show where prior designs stop scaling - [Neo](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf) [P]; [DeGS](https://arxiv.org/abs/2608.02099) [A]
- GPU software: RTX 3090 and L40S, seven Mip-NeRF 360 scenes, 7k and 30k checkpoints, gsplat baseline, PSNR change reported per GPU - [SwiftGS](https://proceedings.mlsys.org/paper_files/paper/2026/file/5f96a21345c138da929e99871fda138e-Paper-Conference.pdf) [P]
- Training systems: maximum trainable Gaussians and throughput relative to GPU-only, on RTX 4090 and RTX 2080 Ti, against an "enhanced" GPU-only baseline and naive offloading - [CLM](https://arxiv.org/html/2511.04951v1) [P]; images/s from 4 to 32 A100s and PSNR at matched images - [Grendel](https://arxiv.org/html/2406.18533v2) [P]
- Constrained-GPU studies emulate weaker GPUs with power caps and clock locks, 2-minute runs, mean ± SD FPS, on one Garden scene - [Splats under Pressure](https://arxiv.org/html/2604.07177) [P]
- Artifact evaluation is starting to appear: SCARF (MICRO 2026) earned all three badges, the SwiftGS (MLSys 2026) PDF embeds artifact-badge images (available, functional, reproduced) - [SCARF](https://arxiv.org/abs/2611.02365) [A]; [SwiftGS PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/5f96a21345c138da929e99871fda138e-Paper-Conference.pdf) [P]

### Inferences
- Quality is held fixed by a PSNR tolerance (typically under 0.2-1.0 dB), and the reported number is speed or energy; this matches the student's preference for systems metrics over reconstruction quality.
- Speedups are against an edge GPU or a prior accelerator, and their reported numbers are not nulls for a datacenter GPU serving many users; a project must measure its own baselines.
- The missing evaluation dimension across all of them is concurrency: no p50/p99 latency per request, no users-per-GPU at an SLO, and no co-location interference.

### Gaps
- I did not extract variance or seed counts; most accelerator papers report deterministic simulated numbers, so "n" is the scene count, but I did not verify this per paper.

## Q5. What is open that a course project on GPU sharing / virtualization could contribute?

### Takeaway
The open space is the intersection the literature skips: several independent users' feed-forward reconstruction (and possibly rendering) jobs on one datacenter GPU, judged by users served at a per-user latency SLO.
The papers supply mechanisms to borrow (stage-level roofline analysis, temporal reuse across frames, SM/PE scaling limits, low lane utilisation) but no multi-tenant measurement to compare against.
The three questions below are my suggestions, not findings; each is built so that step 0 is a cheap measurement that can kill it.

### Cited Findings
- No systems or architecture paper evaluates multi-tenant sharing of a GPU for 3DGS or feed-forward reconstruction (see Q3) - [GBU](https://arxiv.org/html/2503.23625), [SCARF](https://arxiv.org/html/2611.02365), [Splats under Pressure](https://arxiv.org/html/2604.07177) [P]
- The only feed-forward characterisation is on an edge GPU and finds memory-traffic-heavy depth search with a 31.2% L2 hit rate - [SCARF](https://arxiv.org/html/2611.02365) [P]
- Prior accelerators stop scaling with more PEs because of idle PEs - [DeGS](https://arxiv.org/abs/2608.02099) [A]; GSCore's FPS falls from 66.7 to 15.8 between HD and QHD under a fixed 51.2 GB/s - [Neo](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf) [P]
- Inside 3DGS kernels lanes are poorly used: 18.9% lane utilisation (GBU), threads masked over 69% of the time (Lumina) - [GBU](https://arxiv.org/html/2503.23625), [Lumina](https://chenc10.github.io/assets/pdf/2025_isca_lumina.pdf) [P]
- 3DGS rendering stages split between memory-bound (preprocess, sort) and compute-bound (rasterize) - [GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf) [P]; [RoofGS](https://arxiv.org/abs/2608.15785) [P abstract]
- Consecutive frames or updates are highly redundant, which several papers exploit: sort reuse (Neo), radiance caching (Lumina), multi-frame reuse (GSPIM), over 52% redundant generated Gaussians (SCARF) - [Neo](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf), [Lumina](https://chenc10.github.io/assets/pdf/2025_isca_lumina.pdf), [GSPIM](https://arxiv.org/abs/2611.04804), [SCARF](https://arxiv.org/html/2611.02365)
- MPS exposes a per-client SM limit (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) that can be used to measure throughput as a function of SM share - [NVIDIA MPS documentation](https://docs.nvidia.com/deploy/mps/index.html) [S, not re-read this session]

### Inferences

#### Why the current direction looks flat, read through this literature
- The project found StreamSplat compute-bound, MPS no better than time-slicing (G6, G7), and cross-user batching no faster (G9: 4.52 updates/s at batch 8 vs 4.92 at batch 1).
- If one update already saturates the SMs, neither MPS (which only overlaps work) nor batching (which only raises parallelism) can add throughput, so these nulls are the expected outcome for a homogeneous, compute-saturating tenant mix.
- The literature suggests two ways the picture could change: tenants with different bottlenecks (memory-bound sorting vs compute-bound rasterization or encoders), and per-job sublinear scaling with SM count (DeGS, GBU, Lumina point to idle lanes); both are untested for this workload.

#### Suggestion 1 (characterisation, smallest): Why did batching and MPS not help? A kernel-level roofline of one feed-forward update on the RTX 6000 Ada and an H200
- Question: what fraction of a StreamSplat (and AnySplat) update's GPU time is in kernels that are compute-bound, memory-bound, or latency/launch-bound, and does that profile predict the measured capacity (G6) and the batching null (G9)?
- Step 0: one Nsight Compute / Nsight Systems pass over a single update at batch 1 and batch 8, recording achieved SM throughput, DRAM bandwidth and occupancy per kernel.
- What would refute "compute-bound": the dominant kernels sit well below the compute roof and near the DRAM roof, as SCARF found on Orin NX; that would also suggest that MIG slices (which partition memory bandwidth) behave differently from MPS.
- Borrowed method: stage-wise roofline (GSCore, RoofGS) and per-stage breakdown (SCARF); this is the first datacenter-GPU characterisation of a feed-forward 3DGS model in a systems setting that I could find.
- Feasibility: days, local GPU only; it explains numbers already in LOG.md rather than adding a new system.

#### Suggestion 2 (resource allocation): Does a feed-forward update scale linearly with its SM share, and does spatial partitioning (MPS SM caps, H200 MIG) serve more users at a p99 latency SLO than time-slicing?
- Question: measure updates/s as a function of SM share (MPS active-thread percentage 10% to 100% on the RTX 6000 Ada; MIG 1g to 7g on the H200), then compare users served at a fixed per-user p99 update latency under time-slicing, MPS with caps, and MIG.
- Step 0: the SM-scaling curve for one user; if throughput at 50% of SMs is close to half of full, spatial partitioning cannot beat time-slicing on throughput and the question reduces to latency isolation only.
- What would make it interesting: a concave curve (for example 70% of full throughput at 50% SMs), which the idle-lane findings in GBU and Lumina and the PE-scaling finding in DeGS make plausible but do not establish for this workload.
- Metrics fit the student's interest: users served at SLO, per-user p50/p99 latency, and the throughput-vs-isolation trade-off between MPS (shared memory bandwidth and L2) and MIG (partitioned).
- Feasibility: about two weeks; MIG needs the H200 cluster; it reuses the existing capacity harness.

#### Suggestion 3 (co-location): Do heterogeneous 3DGS tenants pack better than homogeneous ones?
- Question: co-locate a compute-bound tenant (feed-forward update) with a memory-bound one (3DGS rendering at high resolution, whose sorting and preprocessing are bandwidth-bound per GSCore and Neo, or per-user gsplat refinement, which is atomic- and memory-heavy per ARC and CLM) under time-slicing, MPS and MIG, and ask whether total useful work rises above the sum predicted from each tenant alone.
- Step 0: check the premise with Suggestion 1's profile plus one gsplat render benchmark at 1080p and 4K; if both tenants turn out compute-bound on this GPU, the complementary-packing argument fails and the study stops.
- Null: the same pair under time-slicing; a control that differs in exactly one factor is the homogeneous pair (two feed-forward tenants) at the same offered load.
- Why it matters for the system: a real live-reconstruction service would run both kinds of work (server-side updates and either server-side rendering for thin clients or test-time refinement), and no paper measures their interference.
- Feasibility: two to three weeks; gsplat is already a standard dependency of these pipelines (ask before adding it).

#### Weaker alternative, noted for completeness
- Temporal-redundancy-aware admission: skip or down-sample a user's update when its input changed little, in the spirit of Neo, Lumina and SCARF's reuse; it can raise users served, but it trades update freshness and quality, which the student said is not the focus.

### Gaps
- Whether StreamSplat or AnySplat is compute- or memory-bound at the kernel level on the RTX 6000 Ada is not known from any source; the project's "compute-bound" label rests on capacity measurements (LOG.md G2, G6), not on counters.
- I found no published SM-share scaling curve for any 3DGS or feed-forward reconstruction workload, so Suggestion 2's step 0 has no prior to compare against.
- Whether MIG on the H200 cluster is actually enabled and schedulable for the student was not checked here.
