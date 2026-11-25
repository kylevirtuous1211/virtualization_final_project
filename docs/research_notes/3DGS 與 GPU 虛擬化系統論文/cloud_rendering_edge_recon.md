# Cloud gaming, cloud VR / remote rendering on shared GPUs, and multi-user edge SLAM / reconstruction as analogues for "many users doing live 3D reconstruction on one GPU" (2012 to November 2026)

Scope note, written 2026-11-18.
None of the papers below appear in `docs/reading_notes.md`; that file covers 3DGS streaming (MD2G-Cast, MuV2, TIGAS, GaussAnything and others), which are cited here only by name.
Generic DNN/LLM serving, 3DGS rendering acceleration, 3DGS network streaming and KV-cache reconstruction models are out of scope and were not surveyed.
Verification method: full text read (via `pypdf` extraction of the primary PDF) is marked **[full text]**.
Abstract-only reads (Semantic Scholar / publisher metadata) are marked **[abstract]**.
Claims taken only from search-engine snippets of the paper or a secondary page are marked **[unverified, snippet]**.
Venue, year and DOI were checked against Crossref or Semantic Scholar metadata unless marked otherwise.
ACM DL and IEEE Xplore returned HTTP 403 to every fetch, so several ACM/IEEE papers could only be read at abstract level.

## Q1. Which papers on GPU virtualization / sharing for cloud gaming or cloud rendering exist, with venue and year?

### Takeaway
There is a continuous line of systems work from 2012 to 2026 on sharing one GPU among many cloud-game or cloud-render sessions: mediated pass-through vGPUs (gVirt, gScale), frame-rate-SLA schedulers built by intercepting the graphics API (VGRIS, vGASA), quality-knob density optimizers (dJay, Stimpack), co-location of games with best-effort GPU work (PilotFish, Combo, CoCG, VUDA), and measurement/benchmark frameworks (Pictor, NetGames 2014 consolidation study).
Most of it is from SJTU (Haibing Guan, Minyi Guo, Quan Chen, Chao Li groups), Intel, Microsoft Research, Georgia Tech and NTHU (Cheng-Hsin Hsu), and almost none of it is at OSDI/SOSP; the venues are ATC, HPDC, SoCC, MICRO, TPDS, IPDPS, MM, NetGames.

### Cited Findings

#### Peer-reviewed venue papers (GPU sharing for cloud gaming / rendering)

- **gVirt: A Full GPU Virtualization Solution with Mediated Pass-Through** (Tian, Dong, Cowperthwaite, Intel), USENIX ATC 2014, no DOI/arXiv. **[full text]** - [USENIX](https://www.usenix.org/conference/atc14/technical-sessions/presentation/tian)
  - Claim borrowed: performance-critical resources are passed through and privileged operations trapped; a GPU scheduler gives each vGPU a time quantum of 16 ms, "because it is less human perceptible to image change", and the large quantum is needed because a GPU context switch costs over 1000x a CPU context switch.
  - Because most GPUs are non-preemptive, gVirt waits for the guest ring buffer to idle before switching, so it only achieves "a coarse-grain QoS policy".
  - Evaluation: Intel 4th-gen integrated graphics on Xen; up to 95% of native performance for GPU-intensive workloads and "scale well up to 7 VMs".
  - Code: open source (later upstreamed as Intel GVT-g), per the paper's statement that "all the gVirt source code is now available".
- **gScale: Scaling up GPU Virtualization with Dynamic Sharing of Graphics Memory Space** (Xue, Tian, Dong, Ma, Wang, Qi, He, Guan), USENIX ATC 2016. **[full text]** - [USENIX](https://www.usenix.org/conference/atc16/technical-sessions/presentation/xue)
  - Claim borrowed: the binding limit on vGPU density was graphics memory address space, not compute; gVirt's static partition of global graphics memory allowed only 3 guest vGPUs in a realistic configuration.
  - gScale gives each vGPU a private shadow GTT and shares memory dynamically, reaching 15 Linux or 12 Windows guest vGPUs (5x and 4x gVirt) at up to 96% of gVirt's performance.
  - Code: [github.com/inkpool/XenGT-Preview-kernel (branch gScale)](https://github.com/inkpool/XenGT-Preview-kernel/tree/gScale), per the paper.
- **VGRIS: Virtualized GPU Resource Isolation and Scheduling in Cloud Gaming** (Yu, Zhang, Qi, Yao, Wang, Guan), ACM HPDC 2013, pp. 203-214; journal extension in ACM TACO 11(2) 2014, DOI 10.1145/2632216. **[unverified, snippet]** for content; full text was not reachable (ACM 403, author copy 404). - [bibtex record](https://bibtex.github.io/HPDC-2013-YuZQYWG.html); [AMiner abstract](https://www.aminer.org/pub/5390baa120f70186a0f37d0c/vgris-virtualized-gpu-resource-isolation-and-scheduling-in-cloud-gaming); [TACO PDF link](https://dl.acm.org/doi/pdf/10.1145/2632216)
  - The HPDC DOI differs between sources: 10.1145/2462902.2462914 (bibtex.github.io) versus 10.1145/2493123.2462914 (Semantic Scholar); not resolved.
  - Claim borrowed: VGRIS sits in the host via library API interception (guest OS and games unmodified) and implements three policies: SLA-aware (guarantee each VM its FPS target), proportional share, and a hybrid that "guarantees minimum resources for SLA while proportionally sharing surplus resources among all VMs".
  - Reported: SLA-aware scheduling raises the average FPS of workloads by 65%; pure proportional sharing can leave the GPU under-used and the hybrid prevents the starvation that proportional sharing may cause.
- **vGASA: Adaptive Scheduling Algorithm of Virtualized GPU Resource in Cloud Gaming** (Zhang, Qi, Yao, Yu, Guan), IEEE TPDS 25(11) 2014, DOI 10.1109/TPDS.2013.288. **[unverified, snippet]** - [DOI](https://doi.org/10.1109/TPDS.2013.288)
  - Claim borrowed: three adaptive scheduling algorithms based on feedback control compute a desired runtime FPS for every game on the machine while fully using the GPU, keeping each game near its FPS target with 5-12% overhead.
  - A companion paper, **Energy-Efficient SLA Guarantees for Virtualized GPU in Cloud Gaming**, IEEE TPDS 2015, DOI 10.1109/TPDS.2014.2350499, adds a second feedback loop that lowers GPU frequency (DVFS) once the FPS loop is satisfied. **[unverified, snippet]** - [DOI](https://doi.org/10.1109/TPDS.2014.2350499)
  - A third SJTU paper, **A Cloud Gaming System Based on User-Level Virtualization and Its Resource Scheduling**, IEEE TPDS 2016, DOI 10.1109/TPDS.2015.2433916, exists; its content was not read. - [DOI](https://doi.org/10.1109/TPDS.2015.2433916)
- **On GPU pass-through performance for cloud gaming: Experiments and analysis** (Shea, Liu), NetGames 2013, DOI 10.1109/NETGAMES.2013.6820614. Read only through the NetGames 2014 paper's summary of it. - [DOI](https://doi.org/10.1109/NETGAMES.2013.6820614)
  - Claim (as reported by Hong et al. 2014): Doom 3 ran below 40 FPS even with one-to-one pass-through on Xen and KVM, partly due to excessive context switches, leading them to conclude that sharing one GPU among cloud-gaming VMs gives unacceptably low frame rates.
- **GPU Consolidation for Cloud Games: Are We There Yet?** (Hong, Fan-Chiang, Lee, K.-T. Chen, C.-Y. Huang, C.-H. Hsu; NTHU / Academia Sinica / NTOU), NetGames 2014, DOI 10.1109/NetGames.2014.7008969. **[full text]** - [author PDF](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2014netgames1.pdf)
  - Claim borrowed: on an NVIDIA GRID K2 (two physical GPU instances, each configurable as pass-through or vGPU with 2, 4 or 8 VMs, up to 16 VMs per card) running GamingAnywhere servers, modern mediated vGPUs had low consolidation overhead, K2 beat Quadro 6000 by up to 3.46x in FPS, virtualized GPUs could outperform pass-through, "more context switches not necessarily lead to lower FPS", and the hypervisor was not the bottleneck.
  - Evaluation: Xen, 8 client machines running the GamingAnywhere client, FPS and context-switch counts as metrics.
- **Placing Virtual Machines to Optimize Cloud Gaming Experience** (Hong, D.-Y. Chen, C.-Y. Huang, K.-T. Chen, C.-H. Hsu), IEEE TCC 2015, DOI 10.1109/TCC.2014.2338295; preliminary 2-page version at NetGames 2013. **[full text]** - [author PDF](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2014tcc.pdf)
  - Claim borrowed: VM placement for cloud gaming is formulated as maximizing provider profit subject to "just-good-enough" QoE, using QoE and performance models measured on GamingAnywhere; the problem is NP-complete, and heuristics reach near-optimal results.
  - Evaluation: trace-driven simulation scaling to 20,000 servers and 40,000 gamers; up to 3.5x the net profit of a state-of-the-art placement heuristic; a second, gamer-centric formulation maximizes total QoE for closed deployments (hotels, internet cafes).
- **dJay: Enabling High-density Multi-tenancy for Cloud Gaming Servers with Dynamic Cost-Benefit GPU Load Balancing** (Grizan, Chu, Wolman), ACM SoCC 2015, DOI 10.1145/2806777.2806942. **[unverified, snippet]**; the full text was not reachable. - [DOI](https://www.doi.org/10.1145/2806777.2806942); [MSR page](https://www.microsoft.com/en-us/research/?p=242138)
  - Claim borrowed: static per-client GPU time slices waste capacity because clients join and leave and scene cost varies; dJay instead tunes each client's rendering workload online with a regression cost model and solves a multiple-choice knapsack to maximize total utility while keeping every client at satisfactory quality.
  - Reported: up to 4x multi-tenant density on one server versus a static configuration, with under 3 ms per frame of overhead; demonstrated on Doom 3 and Fable 3.
- **ShareRender: Bypassing GPU Virtualization to Enable Fine-grained Resource Sharing for Cloud Gaming**, ACM MM 2017, DOI 10.1145/3123266.3123306. Content not read (ACM 403); a search snippet says it shares rendering at frame granularity across GPUs instead of using vGPUs. - [DOI](https://dl.acm.org/doi/pdf/10.1145/3123266.3123306)
- **A Benchmarking Framework for Interactive 3D Applications in the Cloud (Pictor)** (Tianyi Liu et al.), IEEE/ACM MICRO 2020, DOI 10.1109/MICRO50266.2020.00076, arXiv 2006.13378. **[full text]** - [arXiv](https://arxiv.org/abs/2006.13378)
  - Claim borrowed: an "intelligent client" that generates human-like inputs, plus hooks that tag each input and its response frame, gives per-input round-trip time (RTT) with 1.6% mean error versus human players (DeskBench 11.6%, other methods 28-30%).
  - On a GTX 1080 Ti server, co-running two different 3D apps "can provide acceptable QoS" and cuts energy by at least 37% versus two servers, but "resource contention and slowed IPC can severely degrade" performance; Dota 2's performance depended strongly on its co-runner, and L2-cache contention appeared when frames from different instances rendered simultaneously.
  - Docker containers added little average overhead (GPU render time +2.9% mean, +8% max).
  - Two optimizations to the TurboVNC/VirtualGL-style pipeline raised frame rate by 57.7% on average.
  - Code release not found in the paper text.
- **PilotFish: Harvesting Free Cycles of Cloud Gaming with Deep Learning Training** (Wei Zhang, Binghao Chen, Zhenhua Han, Quan Chen, et al.; SJTU / MSRA), USENIX ATC 2022. **[full text]** - [USENIX PDF](https://www.usenix.org/system/files/atc22-zhang-wei.pdf)
  - Claim borrowed: cloud games leave GPUs under-used (5 of 8 popular games used under 50% of an RTX 2060 at 1080p/60 FPS), but co-locating two games interferes badly even below saturation: RDR2 fell from 60 to 20 FPS while GPU utilization rose by at most 24%, because contention on CPU, PCIe and draw-call submission blocks the GPU.
  - Mechanism: hook the graphics library to detect each frame's end, compute the free slice as (frame-time QoS minus last frame's render time), launch only DL-training kernels predicted to finish inside it, and kill stragglers with a high-priority "asserting" kernel; CPU and I/O isolation by thread priority and I/O priority.
  - Evaluation: Steam Remote Play on one RTX 2060, five games at 60 FPS targets, four DL training jobs; metric is 99th-percentile FPS normalized to 60 and "harvest ratio" = (co-located utilization minus game-alone utilization) / (100% minus game-alone utilization); up to 85.1% harvest with no QoS loss, versus straw-man baselines that lose over 30% of 99th-percentile FPS at the same harvest ratio.
  - Code: no repository link in the paper.
- **Maximizing the Utilization of GPUs Used by Cloud Gaming through Adaptive Co-location with Combo** (Chen et al.), ACM SoCC 2023, DOI 10.1145/3620678.3624660. **[abstract]** - [ACM](https://dl.acm.org/doi/10.1145/3620678.3624660)
  - Claim borrowed: prior work co-locates games with best-effort jobs only temporally; Combo adds spatial sharing (intra-SM and inter-SM), compiles RT-core kernels for fine-grained control, and picks per runtime rendering load between "exact integrated SM sharing" and "relaxed intra-SM sharing", reaching up to 38.2% (14.0% average) more best-effort throughput than temporal-only sharing while keeping frame QoS.
- **CoCG: Fine-grained Cloud Game Co-location on Heterogeneous Platform** (Taolei Wang, Chao Li, et al., SJTU), IEEE IPDPS 2024 per the authors' file name; DOI not checked. **[full text]** - [author PDF](https://www.cs.sjtu.edu.cn/~lichao/publications/CoCG_Fine-grained_IPDPS-2024-Wang.pdf)
  - Claim borrowed: split each game into stages by clustering frames on CPU/GPU usage, predict stage resource use with ML, co-locate games so their peaks do not overlap, and "steal time from the loading stage to avoid oversubscribing"; throughput of game deployments +23.7% over prior work.
  - Evaluation: 4-core i7-7700, 2x RTX 2080 ("GTX 2080" in the text), Linux, games including Dota 2 and CS:GO, FPS as QoS.

#### Peer-reviewed venue papers (cloud / edge VR and multi-user rendering)

- **Firefly: Untethered Multi-user VR for Commodity Mobile Devices** (Xing Liu, Vlachou, Feng Qian, Chendong Wang, Kyu-Han Kim), USENIX ATC 2020. **[full text]** - [USENIX](https://www.usenix.org/conference/atc20/presentation/liu-xing)
  - Claim borrowed: per-user online rendering plus encoding on one server does not scale (an H.264 encoding test on a GTX 1080 "clearly cannot support 10+ users" at 60 FPS), so Firefly moves rendering offline, serves pre-encoded tiles, and runs Adaptive Quality Control (AQC) that picks each user's quality from total and per-user bandwidth with a fairness constraint (quality roughly proportional to each user's channel capacity).
  - Evaluation: 15 users at 60 FPS on one commodity server and one 802.11ac AP, motion-to-photon at most 15 ms for 99% of frames, about 1 s of stall per minute, real users' motion traces.
- **Coterie: Exploiting Frame Similarity to Enable High-Quality Multiplayer VR on Commodity Mobile Devices** (Meng, Paul, Hu), ACM ASPLOS 2020, DOI 10.1145/3373376.3378516. **[abstract]** - [DOI](https://doi.org/10.1145/3373376.3378516)
  - Claim borrowed: replicating a 2-layer remote-rendering architecture per player cannot support more than one player because network demand grows linearly; caching and reusing similar background frames across positions (with split rendering to beat the "near-object" effect) cut per-player network need by 10.6-25.7x and supported 4 players at 60 FPS and under 16 ms on Pixel 2 over 802.11ac.
- **MUVR: Supporting Multi-User Mobile Virtual Reality with Resource Constrained Edge Cloud**, IEEE/ACM SEC 2018, DOI 10.1109/SEC.2018.00008 (Semantic Scholar mislabels the venue). **[abstract]** - [DOI](https://doi.org/10.1109/SEC.2018.00008)
  - Claim borrowed: the edge cloud memoizes rendered VR frames and reuses redundant ones across users, cutting edge computation by more than 90% and transmitted frame data by more than 95%.
- **Edge-assisted Real-time Dynamic 3D Point Cloud Rendering for Multi-party Mobile Virtual Reality (PoClVR)**, ACM MM 2024, DOI 10.1145/3664647.3681650; extended in IEEE TMC 2026, DOI 10.1109/TMC.2026.3669201. **[abstract]** - [ACM](https://dl.acm.org/doi/10.1145/3664647.3681650)
  - Claim borrowed: splitting a dynamic point-cloud scene into per-object rendering tasks across edge servers removes repeated scene loading; up to 15.1% lower GPU utilization and up to 34.6% higher frame rate than baselines.
- **Heimdall: Mobile GPU Coordination Platform for Augmented Reality Applications**, ACM MobiCom 2020, DOI 10.1145/3372224.3419192. **[abstract]** - [DOI](https://doi.org/10.1145/3372224.3419192)
  - Claim borrowed (co-location of rendering with other GPU work, on a mobile GPU): DNN contention drops rendering from 30 to about 12 FPS and multi-DNN contention raises inference latency from 59.93 to 1181 ms; "pseudo-preemption" (splitting DNNs into small units and scheduling by priority) restores about 30 FPS and cuts worst-case DNN latency by up to about 15x.
- **Deadline-Based Scheduling for GPU with Preemption Support** (Capodieci et al.), IEEE RTSS 2018, DOI 10.1109/RTSS.2018.00021. **[abstract]** - [author PDF](https://iris.unimore.it/bitstream/11380/1182256/4/RTSS18.pdf)
  - Claim borrowed: a preemptive EDF scheduler with Constant Bandwidth Server (CBS) bandwidth isolation for GPU tasks, running under the NVIDIA hypervisor and using pixel-level and thread-level preemption on an automotive SoC.
- **Efficient Performance-Aware GPU Sharing with Compatibility and Isolation through Kernel Space Interception (KRYPTON)**, USENIX ATC 2025. Generic GPU sharing, not rendering; listed only because VUDA uses it as a time-sharing comparison point. **[unverified, snippet]** - [USENIX](https://www.usenix.org/conference/atc25/presentation/zhang-shulai)

#### arXiv-only preprints (no peer-reviewed venue found)

- **Stimpack: An Adaptive Rendering Optimization System for Scalable Cloud Gaming** (Jin Heo, Vic Wang, Ketan Bhardwaj, Ada Gavrilovska), arXiv 2412.19446 (v1 Dec 2024, v2 19 Apr 2026), no venue listed. **[full text]** - [arXiv](https://arxiv.org/abs/2412.19446)
  - Claim borrowed: render quality (RQ) beyond what survives lossy video compression is wasted GPU time; a round-based optimizer scores each user's resource efficiency (rendering time versus estimated user-side VMAF from RQ and QP), demotes the least efficient user when anyone drops below the FPS threshold and promotes the most efficient otherwise, with exponential backoff against oscillation.
  - Evaluation: one RTX 4090, Unreal Engine games, up to six users in two groups of three; baselines highest-only (each user gets the highest RQ and users are queued when GPUs are full, "inspired by an existing cloud gaming service"), lowest-only and dJay; user study; up to 24% higher service quality and "twice as many users with the same resources".
  - Code: [github.com/gt-stimpack/Stimpack](https://github.com/gt-stimpack/Stimpack).
- **VUDA: Enabling Controlled Spatial Sharing of Graphics and Compute on NVIDIA GPUs** (Bin Xu, Pengfei Hu, Jinyu Gu, Haibo Chen, SJTU), arXiv 2605.01352 (v2 Sep 2026), no venue. **[full text, via arXiv HTML]** - [arXiv](https://arxiv.org/abs/2605.01352)
  - Claim borrowed: Vulkan and CUDA contexts land in separate time-slice groups, so they never run concurrently by default; VUDA moves CUDA channels into the Vulkan TSG and uses undocumented front-end commands to set a per-GPC TPC quota, with a tuner that checks per-frame and per-kernel times against a frame-time SLO and a compute latency or throughput SLO.
  - Evaluation: RTX 4090, A800, Jetson AGX Orin; arms are baseline time sharing, redirect-only, and VUDA with a quota (MPS and priority streams not measured as arms); VR case (90 Hz stereo rendering plus 60 Hz perception on RTX 4090, five 12-second runs): under heavy load the non-partitioned arms missed 52.1% and 78.1% of rendering deadlines, while VUDA had zero rendering misses and a 2.78% median perception miss rate.
  - Self-reported limitations: shared L2 and memory bandwidth remain, the 5% perception-miss limit was chosen after looking at the data, and the interface is undocumented.

### Inferences
- The field split early into two allocation styles that map onto the student's arms: static slicing (gVirt's 16 ms quantum, NVIDIA vGPU profiles, MIG) and SLA-driven pacing (VGRIS, vGASA), which caps a session at its target rate so the surplus goes to others.
- The papers that report striking density gains (dJay 4x, Stimpack 2x, Firefly 15 users) all change the per-user cost of the work through a knob or precomputation; pure scheduling papers (VGRIS, PilotFish, Combo, CoCG) report 14-65% improvements or harvest of idle time, not multiples.
- gScale's lesson that address space, not compute, capped vGPU count echoes the project's 4 GB per StreamSplat update: on a 48 GB card memory caps resident instances at about 12 even though compute caps throughput at about 5 updates/s (project numbers from the brief, not from these papers).

### Gaps
- VGRIS, vGASA, dJay and ShareRender full texts were not reachable (ACM/IEEE 403, author mirrors 404); their mechanisms and numbers above rest on abstracts and search snippets, and per-VM counts and GPU models for VGRIS and dJay are unknown.
- I found no OSDI, SOSP, EuroSys, ASPLOS (except Coterie), ISCA or HPCA paper specifically on GPU sharing for cloud gaming; the searches were not exhaustive over MMSys/NOSSDAV proceedings.
- I found no paper on MIG or MPS specifically for interactive rendering or reconstruction sessions with per-user latency SLOs.
  NVIDIA states that one RTX PRO 6000 Blackwell Server Edition can serve up to 48 concurrent users with MIG plus vGPU for game development (GDC 2026); this is a vendor claim seen only in a search snippet and was not opened.
- Kahawai (MobiSys 2015, DOI 10.1145/2742647.2742657) and Server Allocation for Multiplayer Cloud Gaming (ACM MM 2016, DOI 10.1145/2964284.2964301) exist per Semantic Scholar but were not read.

## Q2. How do they allocate GPU share per user, how many users per GPU do they reach, and what made the results interesting when the GPU was saturated?

### Takeaway
Allocation falls into five families: fixed time quanta (gVirt), SLA pacing with a frame-rate target plus proportional sharing of the surplus (VGRIS hybrid), feedback control on FPS (vGASA), cost-benefit quality tuning to fit more users (dJay, Stimpack, Firefly AQC), and best-effort harvesting under a frame deadline (PilotFish, Combo, VUDA).
Reported densities are small, from 4 to 16 sessions per GPU, and the interesting saturated-regime results come from showing that naive sharing collapses everyone's frame rate (Shea and Liu, PilotFish), while a controller either protects the admitted users or degrades everyone gracefully.

### Cited Findings
- Fixed quantum: gVirt switches vGPUs every 16 ms, and non-preemptive GPUs make the slice only approximately enforced ("coarse-grain QoS"); 7 VMs. - [gVirt, ATC 2014](https://www.usenix.org/conference/atc14/technical-sessions/presentation/tian)
- Density limited by memory: 3 vGPUs under gVirt's static partition became 15 (Linux) or 12 (Windows) with gScale's dynamic sharing. - [gScale, ATC 2016](https://www.usenix.org/conference/atc16/technical-sessions/presentation/xue)
- Hardware vGPU: GRID K2 supports up to 8 VMs per physical GPU instance, 16 per card, and the study found low consolidation overhead. - [Hong et al., NetGames 2014](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2014netgames1.pdf)
- SLA pacing: VGRIS guarantees each VM a minimum for its FPS SLA and shares the surplus proportionally (hybrid); SLA-aware scheduling raised average FPS by 65%. **[unverified, snippet]** - [VGRIS, HPDC 2013](https://bibtex.github.io/HPDC-2013-YuZQYWG.html)
- Feedback control: vGASA's three feedback-control algorithms hold each game near a target FPS while using the whole GPU, with 5-12% overhead. **[unverified, snippet]** - [vGASA, TPDS 2014](https://doi.org/10.1109/TPDS.2013.288)
- Cost-benefit tuning: dJay solves a multiple-choice knapsack over per-client rendering settings with a regression cost model and reports up to 4x density over static slices. **[unverified, snippet]** - [dJay, SoCC 2015](https://www.doi.org/10.1145/2806777.2806942)
- Round-based greedy tuning: Stimpack keeps every user's FPS between a threshold and an upper bound, adjusts one user's quality per round, and serves 6 users on an RTX 4090 where the highest-quality baseline must queue users (2x users). - [Stimpack, arXiv 2412.19446](https://arxiv.org/abs/2412.19446)
- Bandwidth-side allocation with fairness: Firefly's AQC chooses per-user quality "largely proportional to the users' wireless channel capacities" and reaches 15 users at 60 FPS. - [Firefly, ATC 2020](https://www.usenix.org/conference/atc20/presentation/liu-xing)
- Deadline harvesting: PilotFish computes the free slice after each frame and admits only best-effort kernels predicted to fit, with preemption of stragglers; the game is strictly higher priority. - [PilotFish, ATC 2022](https://www.usenix.org/system/files/atc22-zhang-wei.pdf)
- SLO tuner on spatial partitions: VUDA re-partitions TPCs between graphics and compute against a frame-time SLO and a compute SLO. - [VUDA, arXiv 2605.01352](https://arxiv.org/abs/2605.01352)
- Saturation result 1, collapse without control: co-locating two games on an RTX 2060 cut RDR2 from 60 to 20 FPS while utilization rose only up to 24%, so interference appears before the GPU is busy. - [PilotFish, ATC 2022](https://www.usenix.org/system/files/atc22-zhang-wei.pdf)
- Saturation result 2, collapse even with pass-through: Doom 3 ran below 40 FPS on a pass-through GPU, which was read at the time as "sharing is virtually impossible"; the NetGames 2014 study overturned this on newer vGPU hardware. - [Hong et al., NetGames 2014](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2014netgames1.pdf)
- Saturation result 3, deadline misses under time sharing: in VUDA's VR case, time-shared arms missed 52.1% and 78.1% of 90 Hz rendering deadlines under heavy load, against zero with partitioning. - [VUDA](https://arxiv.org/abs/2605.01352)
- Saturation result 4, co-runner identity matters: Dota 2's performance varied strongly with which app shared the server (SuperTuxKart worst, 0AD least). - [Pictor, MICRO 2020](https://arxiv.org/abs/2006.13378)
- Admission in production: Stimpack notes that commercial cloud-gaming services "limit session playtime and queue users when nearby servers are fully occupied". - [Stimpack](https://arxiv.org/abs/2412.19446)

### Inferences
- The project's observation (p95 of 13 s at 6 users, latency unbounded past about 5 updates/s) is the open-loop, no-admission regime that this literature treats as the failure to be fixed; none of these systems leaves arrival rate unregulated once capacity is exceeded.
- In the cloud-gaming papers the per-session work is elastic (FPS and quality knobs), which is why pacing and knapsack-style allocation help; a StreamSplat update has a fixed 0.19 s cost, so the knobs that remain are update rate per user, input resolution and admission.
  Whether input resolution changes the cost of a StreamSplat forward is unmeasured and should be step 0 before any knapsack-style design.
- PilotFish and Combo get their gains from idle cycles in games that use under 50% of the GPU; the project's GPU is compute-bound at capacity, so harvesting can only help below the knee (1 to 4 users), which matches the finding that batching and MPS did not add a user.

### Gaps
- No paper here reports the full curve of per-user tail latency against the number of users up to and past saturation for a fixed-cost GPU job; most report FPS at a few fixed user counts.
- Jain's fairness index or another fairness metric is mentioned only qualitatively in the papers read in full (Firefly); I did not find a cloud-gaming paper that reports a numeric fairness index.

## Q3. Which papers offload SLAM, AR or 3D reconstruction for multiple users to an edge server, and how do they share the server GPU among users?

### Takeaway
Multi-user edge SLAM is a small, mostly networking-venue literature (NSDI, CoNEXT, MobiSys, INFOCOM) built on ORB-SLAM; the server runs one process or thread per client plus a shared global map, and GPU sharing among users is either absent (CPU-bound SLAM) or left to the default driver with no per-user GPU allocation.
The scheduling that exists is request-level prioritization (SwarmMap's SLAM-specific task scheduler), not GPU partitioning, and no paper found measures how users per edge GPU scale under a per-user latency SLO for dense or neural reconstruction.

### Cited Findings
- **SwarmMap: Scaling Up Real-time Collaborative Visual SLAM at the Edge** (Jingao Xu, Hao Cao, Zheng Yang, Longfei Shangguan, et al.), USENIX NSDI 2022; extended as "Scaling Up Edge-Assisted Real-Time Collaborative Visual SLAM Applications", IEEE/ACM ToN 2024, DOI 10.1109/TNET.2023.3330763. **[full text]** - [USENIX](https://www.usenix.org/conference/nsdi22/presentation/xu)
  - Claim borrowed: as agents grow, concurrent optimization requests "block at the edge node's processing pipeline, resulting in excessive queuing delays", and CCM-SLAM, Multi-UAV and CarMap fail the localization requirement beyond 5 agents indoors and 10 outdoors.
  - Mechanism: change-log-based map synchronization, a lean global-map representation, and STS, a multi-level priority queue that serves agents in an "emergency group" first and then agents whose data gives higher information gain to the global map.
  - Evaluation: TUM, KITTI, EuRoC and a 22,927 sq ft mall dataset, testbed of Jetson boards, phones, RoboMasters and drones, Jetson AGX Xavier as the edge node, three-month oil-field case study; over 20 agents (2x the baselines) at 38 cm average trajectory error, link throughput under 17 MB/s and RAM under 26 GB.
  - Code: [github.com/MobiSense/SwarmMap](https://github.com/MobiSense/SwarmMap).
- **SLAM-Share: Visual Simultaneous Localization and Mapping for Real-time Multi-user Augmented Reality** (Dhakal, Ran, Wang, Jiasi Chen, K. K. Ramakrishnan), ACM CoNEXT 2022, DOI 10.1145/3555050.3569142. **[full text]** - [author PDF](https://www.cs.ucr.edu/~jiasi/pub/SLAMShare-conext22.pdf)
  - Claim borrowed: each client gets its own server process that runs tracking with a CUDA kernel for ORB feature extraction, and all processes merge into a global map in shared memory guarded by a shareable reader-writer mutex; tracking latency drops by up to 50% versus CPU-only, and map merging is at least 30x faster than a baseline without shared memory.
  - GPU sharing among the per-client processes is not controlled; the group's earlier GSLICE (spatial GPU sharing for inference) is cited but not applied as a per-user allocation in what I read.
  - Evaluation: EuRoC, KITTI, TUM, timelines with 3 clients, ATE and latency against a multi-user extension of Edge-SLAM; scalability is argued from network bandwidth, not measured on the GPU.
  - Code: [github.com/network-lab2/slam-share](https://github.com/network-lab2/slam-share).
- **Edge-SLAM: Edge-Assisted Visual Simultaneous Localization and Mapping** (Ben Ali, Hashemifar, Dantu), ACM MobiSys 2020, DOI 10.1145/3386901.3389033 per Crossref; Crossref also lists a MobiCom 2020 item (10.1145/3372224.3417326), probably a demo; journal version DOI 10.1145/3561972. **[abstract]** - [DOI](https://doi.org/10.1145/3561972)
  - Claim borrowed: single-user split of ORB-SLAM2 with tracking on the device and local mapping plus loop closing on the edge keeps device compute and memory constant over long runs; it is the single-user baseline SLAM-Share extends.
- **Edge Assisted Mobile Semantic Visual SLAM** (Hao Cao, Jingao Xu, Danyang Li, et al.), IEEE INFOCOM 2020, with a TMC extension that adds a multi-user support mechanism. Venue taken from the author's PDF file name, not DOI-checked; not read. - [author PDF](https://www.cs.cmu.edu/~jingaox/assets/pdf/papers/infocom20_edgeSLAM.pdf); [TMC PDF](https://tns.thss.tsinghua.edu.cn/~danyang/assets/pdf/edgeSLAM-TMC.pdf)
- **AdaptSLAM: Edge-Assisted Adaptive SLAM with Resource Constraints via Uncertainty Minimization** (Ying Chen, Inaltekin, Gorlatova), IEEE INFOCOM 2023, DOI 10.1109/INFOCOM53939.2023.10229009, arXiv 2301.04620. **[full text]** - [arXiv](https://arxiv.org/abs/2301.04620)
  - Claim borrowed: choose the subset of keyframes for the local and global maps that minimizes map uncertainty under explicit computation and communication budgets, with a greedy near-submodular solver; this is the clearest "resource budget as an input to the algorithm" design in the SLAM line.
  - It is single-device with a budget; it does not share a server among users.
  - Code: [github.com/i3tyc/AdaptSLAM](https://github.com/i3tyc/AdaptSLAM).
- **Orchestrating Joint Offloading and Scheduling for Low-Latency Edge SLAM**, IEEE TMC 2025, DOI 10.1109/TMC.2025.3547256, arXiv 2502.16495. **[abstract]** - [arXiv](https://arxiv.org/abs/2502.16495)
  - Claim borrowed: DRL-based configuration adaptation (compression plus offloading) and a learned scheduler with constraint satisfaction across edge servers; up to 47% less communication than a popular edge SLAM system.
- **ARISE: High-Capacity AR Offloading Inference Serving via Proactive Scheduling**, ACM MobiSys 2024, DOI 10.1145/3643832.3661894. **[abstract]** - [ACM](https://dl.acm.org/doi/10.1145/3643832.3661894)
  - Claim borrowed (pointer only; it is DNN serving): the objective is the number of AR clients each GPU server can serve concurrently while meeting per-client accuracy SLAs, achieved by coordinating clients' offloading schedules with server batching; the evaluation uses many emulated clients and a 10-phone testbed.
- **Characterizing Distributed Mobile Augmented Reality Applications at the Edge (scAtteR)**, CoNEXT 2023 Companion, DOI 10.1145/3624354.3630584. Not read; listed because a search snippet says it characterizes GPU contention among distributed AR services. - [PDF](https://www.nitindermohan.com/documents/2023/pubs/marOakestraCoNext2023.pdf)
- **Edge Collaborative Gaussian Splatting with Integrated Rendering and Communication (ECO-GS)**, arXiv 2510.22718; Semantic Scholar says ICASSP 2025, which conflicts with an October 2025 arXiv date, so the venue is unverified. **[abstract]** - [arXiv](https://arxiv.org/abs/2510.22718)
  - Claim: per user, choose a local small GS model or a remote large GS model jointly with edge power allocation; it optimizes radio resources, not GPU sharing.

### Inferences
- SwarmMap is the closest published analogue to the project's failure mode: unbounded queueing at a shared edge node once agents exceed a threshold, fixed by reducing per-agent work and prioritizing requests by agent state rather than by FIFO order.
- The multi-user SLAM papers measure accuracy (ATE) against the number of agents, not per-user GPU time or tail latency against users per GPU, which leaves the GPU-allocation question open for reconstruction workloads.

### Gaps
- I found no MobiSys, MobiCom, SenSys or NSDI paper (to November 2026) that serves several users' dense or neural (NeRF / 3DGS / feed-forward) reconstruction from one edge GPU with per-user GPU allocation; this negative result covers WebSearch queries over those venues and arXiv and is not exhaustive.
- Whether SLAM-Share's per-client processes ran under MPS or default time-slicing is not stated in the text I extracted.

## Q4. How are these systems evaluated (users per server, FPS, tail latency, fairness, traces)?

### Takeaway
The standard design is a single server with a small number of concurrent sessions (2 to 20), a per-user frame-rate or latency target, and a metric of how many users meet that target, reported against a naive baseline (static slices, highest-quality-plus-queue, FIFO, or time sharing).
Tail metrics are normalized to the target (99th-percentile FPS / 60), and the better papers add input-to-frame RTT, harvest ratio, and accuracy-versus-users curves.

### Cited Findings
- Users served at target: Stimpack counts users kept at playable FPS on one RTX 4090 against highest-only (with queueing), lowest-only and dJay, plus a user study. - [Stimpack](https://arxiv.org/abs/2412.19446)
- Tail FPS normalized to target and a harvest ratio defined from nvidia-smi-style utilization: PilotFish. - [PilotFish](https://www.usenix.org/system/files/atc22-zhang-wei.pdf)
- Per-input round-trip time with tagged input-frame pairs and a bot that reproduces human input timing within 1.6%: Pictor. - [Pictor](https://arxiv.org/abs/2006.13378)
- 99th-percentile motion-to-photon (at most 15 ms), stall seconds per minute, 60 FPS, fairness across 15 users, real users' motion traces: Firefly. - [Firefly](https://www.usenix.org/conference/atc20/presentation/liu-xing)
- Deadline-miss rate against a frame-time budget, with repeated short runs (five 12-second runs) and medians with min-max: VUDA. - [VUDA](https://arxiv.org/abs/2605.01352)
- Accuracy (ATE) and resource use versus number of agents, compared with CCM-SLAM and Multi-UAV, plus a long field deployment: SwarmMap. - [SwarmMap](https://www.usenix.org/conference/nsdi22/presentation/xu)
- Trace-driven simulation at datacenter scale (20,000 servers, 40,000 gamers) with QoE models measured on a real platform: Hong et al. - [TCC 2015](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2014tcc.pdf)
- Throughput of best-effort work at unchanged frame QoS: Combo (+38.2% max, +14.0% average) and CoCG (+23.7% deployment throughput). - [Combo](https://dl.acm.org/doi/10.1145/3620678.3624660); [CoCG](https://www.cs.sjtu.edu.cn/~lichao/publications/CoCG_Fine-grained_IPDPS-2024-Wang.pdf)

### Inferences
- A design the project could copy directly: Stimpack's "highest-only plus queue" baseline is exactly an admission-control null, and PilotFish's normalized tail metric (p95 latency divided by the SLO) makes arms comparable across user counts.
- VUDA's own admission that its 5% miss limit was chosen after seeing the data is a concrete example of the post-hoc threshold the project's research discipline forbids; the project should fix its SLO before the run.

### Gaps
- Few papers report run-to-run spread; VUDA is the exception among those read in full, and most others report single runs or averages without n.

## Q5. What is open that a project on resource allocation and per-user latency could measure on one GPU, and which project questions do these papers suggest?

### Takeaway
What is open is the allocation layer for a fixed-cost, compute-bound per-user GPU job with a latency SLO: admission, per-user pacing and isolation, which cloud gaming solved for elastic rendering work but which nobody has measured for live reconstruction.
Three candidate questions follow (suggestions, not decisions); each needs only the existing StreamSplat server, one GPU, and a scheduler in front of it.

### Cited Findings
- Cloud gaming solves overload by pacing each session to its SLA and sharing only the surplus (VGRIS hybrid), not by running everyone as fast as possible. **[unverified, snippet]** - [VGRIS](https://bibtex.github.io/HPDC-2013-YuZQYWG.html)
- Production services queue users when servers are full; Stimpack uses this as its baseline. - [Stimpack](https://arxiv.org/abs/2412.19446)
- Density gains in the literature come from changing per-user cost: dJay up to 4x **[unverified, snippet]**, Stimpack 2x, Firefly's offline precompute. - [dJay](https://www.doi.org/10.1145/2806777.2806942); [Stimpack](https://arxiv.org/abs/2412.19446); [Firefly](https://www.usenix.org/conference/atc20/presentation/liu-xing)
- Request prioritization by client state (SwarmMap STS) is the only per-user scheduler in the multi-user SLAM literature, and it doubled agents served. - [SwarmMap](https://www.usenix.org/conference/nsdi22/presentation/xu)
- Interference appears before full utilization, so isolation is a separate question from capacity. - [PilotFish](https://www.usenix.org/system/files/atc22-zhang-wei.pdf); [Pictor](https://arxiv.org/abs/2006.13378)

### Inferences
- **Suggestion 1 (admission and pacing, after VGRIS):** "When offered load exceeds the GPU's capacity of about 5 updates/s, how many users meet a pre-declared p95 update-latency SLO under (a) the current unregulated FIFO, (b) admission control with per-user rate caps (SLA pacing), (c) proportional share without admission, and (d) the hybrid of a guaranteed minimum plus proportional surplus?"
  Sweep 1 to 10 users; report users meeting the SLO, per-user p95 normalized to the SLO, GPU utilization, and Jain's index.
  The matched null is the same server with admission and pacing switched off, so the one factor is the controller.
  Expected near-mechanical result for (a) versus (b): 0 users meet the SLO past the knee under (a) and about floor(5/r) under (b) at per-user rate r; the measurable part is how close (b) and (d) get to the bound and what (c) does to tail latency.
- **Suggestion 2 (rate as the knob, after dJay and Stimpack):** "If each user's update rate (or input resolution, if it changes cost) is allocated by an optimizer instead of fixed, how many more users fit under the same SLO, and at what staleness cost?"
  Baselines copied from Stimpack: highest-only with a queue, lowest-only, and a fixed equal split.
  Step 0 is measuring whether any StreamSplat knob changes the 0.19 s cost; if none does, the optimizer reduces to arithmetic and this question should be dropped and logged as a negative result.
  Report staleness (age of each user's latest reconstruction) instead of reconstruction quality, since the student does not care about quality.
- **Suggestion 3 (isolation under a noisy neighbour, after VGRIS's isolation goal and VUDA's SLO tuner):** "With one user submitting at 3x its share, how much do the other users' p95 latencies degrade under default time-slicing, MPS, MPS with active-thread-percentage caps, per-user software fair queuing, and MIG on the H200?"
  The victim's p95 with the aggressor absent is the matched control, and the RTX 6000 Ada (no MIG) and H200 (MIG) arms must be reported separately because the hardware differs.
  An optional arm is SwarmMap-style priority (serve the most stale or fastest-moving user first) against FIFO and round-robin, measured as per-user staleness.
- PilotFish-style harvesting (running best-effort work in idle gaps) only makes sense below the knee, since the project's GPU is compute-bound at capacity; if it is tried, the harvest ratio at 1 to 4 users is the metric.
- Local expertise: Cheng-Hsin Hsu's NTHU group wrote the GamingAnywhere consolidation and VM-placement papers above, as well as the 3DGS streaming papers already in `docs/reading_notes.md`.

### Gaps
- None of these suggestions has a published null on reconstruction workloads; the expected numbers above are reasoning from the project's own measured capacity, not from any paper.
- I could not verify that MPS active-thread-percentage caps or MIG behave on the H200 for this model as they do for the inference workloads in the generic literature; that literature was out of scope.
