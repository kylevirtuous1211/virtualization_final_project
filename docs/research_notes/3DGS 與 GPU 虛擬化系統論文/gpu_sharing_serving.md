# GPU sharing, GPU virtualization and multi-tenant DNN inference serving (2018 to November 2026)

Compiled 2026-11-18 as input to the direction review of virtualization_final_project.
Four parallel readers covered sharing mechanisms, SLO-aware serving with Kubernetes GPU sharing, per-tenant memory growth, and video/vision serving.
I read the NVIDIA MPS and MIG documentation (both Release 620) myself.
I spot-checked the load-bearing quotes against the downloaded full texts: Tally, Gilman & Walls, GPREEMPT, Nexus, ARISE, Clockwork and AlpaServe all matched.
`docs/reading_notes.md` covers only 3DGS streaming papers, so none of the papers below had been surveyed before.
"Full text" means the PDF or arXiv HTML was read; "abstract" means only the abstract or proceedings page was read.
[unverified] marks anything that no primary source confirmed.
The project's own numbers are cited to its `LOG.md` entries (prefix G) on branch `feat/capacity-probe`.

## 1. Which mechanisms exist for spatial and temporal GPU sharing, and where was each published?

### Takeaway
There are four families.
The first is vendor mechanisms: time-slicing, MPS (with optional SM caps, memory caps, static SM partitions and, in the 2026 MPS v3 interface, weighted time slices (driver r620) and cgroup memory limits (CUDA 13.4)) and MIG.
The second is kernel-level software schedulers (REEF, Orion, Paella, Tally, LithOS, XSched, GPREEMPT, SGDRC).
The third is cluster and serving-level multiplexing (Gandiva, AntMan, TGS, Clockwork, Shepherd, AlpaServe, MuxServe).
The fourth is OS or container-layer virtualization (TGS, HAMi, KubeShare, gShare, Ghost).
Peer-reviewed systems work since 2022 has moved from "pack more onto the GPU" to "preempt and prioritise inside a shared GPU", and NVIDIA's own driver absorbed some of that in 2026 (MPS v3).

### Cited Findings

#### Vendor mechanisms (NVIDIA documentation, read in full)
- NVIDIA states that MPS "is useful when each application process does not generate enough work to saturate the GPU", identified by "a small number of blocks-per-grid" or low occupancy - [MPS doc, Release 620, "When to Use MPS"](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- A uniform `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` limit "is configured for a client process when it starts and cannot be changed for the client process afterwards".
  A per-context ("non-uniform") limit, enabled with `CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING`, "can be changed throughout the client process" - [MPS doc](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- NVIDIA's own provisioning advice: 100%/n per client "will allocate close to the minimum amount of execution resources, but it could restrict performance"; 100%/(0.5n) gives the load balancer more freedom; a non-uniform split by workload ratio is called "near optimal" - [MPS doc](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- `set_active_thread_percentage <PID>` and `set_device_pinned_mem_limit <PID> <dev>` both say "All clients created with that server afterwards will observe the new limit. Existing clients are not affected." - [MPS doc, nvidia-cuda-mps-control](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
  This documents the project's 2026-11-04 nano4 probe (a running client could not grow after the server limit was raised).
- A client exceeding its pinned device memory limit gets an out-of-memory error from the allocation call; the limit also counts CUDA's internal allocations - [MPS doc, "MPS Device Memory Limit"](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- MPS client priority has only two levels, 0 (normal) and 1 (below normal), takes effect at the next client connection, and "CUDA priority levels are not guarantees of execution order - they are only a performance hint" - [MPS doc, "Client Priority Level Control"](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- Client limits: "Each MPS server supports up to 60 client CUDA contexts per-device" with the default `CUDA_DEVICE_MAX_CONNECTIONS=2` - [MPS doc](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf); contradicted by the MIG guide's comparison table, which lists 48 as the MPS maximum - [MIG User Guide, Release 620, Table 3](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf)
- Fault isolation under MPS (Volta and later): a fatal fault is contained to the GPUs shared with the faulting client and reported to every client on them - [MPS doc](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
  The MIG guide puts it more bluntly: "MPS currently does not offer error isolation between clients" and memory bandwidth, caches and capacity "are all shared" - [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf)
- Static SM partitioning (`nvidia-cuda-mps-control -d -S`) gives exclusive SM partitions on Ampere and newer GPUs.
  Partitions are rejected if the device already has active clients and cannot be removed while in use.
  When it is on, active thread percentages are ignored.
  From driver r610 it adds "partial error isolation" between partitions - [MPS doc, "Static SM Partitioning"](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
  The local driver (595.91.07) lists the `-S, --static-partitioning` flag in `nvidia-cuda-mps-control -h` (checked 2026-11-18; flag presence only, not run).
- New in 2026, "MPS v3" (opt-in control interface), three features:
  - SM partitions can lend idle SMs (`--shareable`) and borrow up to `--max-chunks`; "Partitions never share SMs at the same time, so error isolation is preserved" - [MPS doc, "SM Partition Load Balancing"](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
  - "Starting with driver version r620, MPS v3 supports weighted timeslicing": a base slice of short 10 ms, medium 20 ms or long 50 ms, scaled per server by `timeslice-pct`; "it does not guarantee a percentage of GPU utilization", and updates are rejected while clients are attached - [MPS doc, "Weighted Timeslicing"](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
  - Memory partitioning per cgroup: needs cgroup v2, "CUDA 13.4 or newer, and a non-MIG device".
    It has a soft limit (guaranteed zone, `dmem.min`) and a hard limit (`dmem.max`, allocations above it fail).
    Between the two is a "pressure zone" where a client is "borrowing" from others' share.
    Eviction is "reactive, not proactive": when a client still under its soft limit would OOM, MPS force-terminates the oldest over-soft cgroup, repeating within a 10 s budget.
    Limits are set with `sudo nvidia-smi memory-limits` or privileged NVML calls; the `dmem` backend needs Linux 6.14 or newer, otherwise a driver-internal "misc" backend is used - [MPS doc, "MPS v3 Memory Partitioning"](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- The local RTX 6000 Ada runs driver 595.91.07 with CUDA 13.2 and reports MIG mode N/A (`nvidia-smi`, checked 2026-11-18), so the r620 and CUDA 13.4 features above are not available on it without a driver upgrade.
- `nvidia-smi compute-policy --set-timeslice` (0 default, 1 short, 2 medium, 3 long) exists in the local driver, and the current setting is "Default" (`nvidia-smi compute-policy -l`, checked 2026-11-18).
  Whether it needs root, and what it does on Ada, was not tested.
- The MIG guide's comparison: Streams / MPS / MIG have max partitions unlimited / 48 / 7; SM performance isolation No / "Yes (by percentage, not partitioning)" / Yes; memory bandwidth QoS No / No / Yes; error isolation No / No / Yes; reconfiguration Dynamic / Process Launch / When Idle - [MIG User Guide, Table 3](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf)
- H200 (SXM5 and NVL, 141 GB) supports up to 7 MIG instances (CUDA 12, driver R550 >= 550.90.07); its profiles include `1g.18gb` (7 instances), `1g.35gb` (4) and `2g.35gb` - [MIG User Guide, Sec. 9.10](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf)
- From Hopper on, "enabling MIG mode no longer requires a GPU reset", and MIG mode does not persist across driver reloads - [MIG User Guide, Sec. 10.2.1](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf)
- "CUDA MPS is supported on top of MIG", with the 48-client maximum "lowered proportionally to the Compute Instance size" - [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf)
- The RTX 6000 Ada is not in the MIG supported-GPU list; the RTX PRO 6000 Blackwell (96 GB, up to 4 instances) is - [MIG User Guide, Supported GPUs](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf)
- NVIDIA k8s-device-plugin time-slicing: "nothing special is done to isolate workloads that are granted replicas from the same underlying GPU"; "if one workload crashes, they all do"; asking for more replicas does not guarantee proportionally more compute - [k8s-device-plugin README](https://github.com/NVIDIA/k8s-device-plugin)
- NVIDIA k8s-device-plugin MPS sharing is "experimental" (since v0.15.0).
  With `replicas: N`, each client's memory is "limited to an equal fraction of the total device memory" and compute is also capped.
  It is not supported with MIG, and time-slicing and MPS cannot be mixed - [k8s-device-plugin README](https://github.com/NVIDIA/k8s-device-plugin)
  This plugin-MPS arm (equal 1/N caps) differs from the default MPS with no per-client limits that the project measured in G2, G3, G6 and G7.
- HAMi (CNCF Incubating): `libvgpu.so` is loaded by `LD_PRELOAD` and "hijacks the API calls between the CUDA runtime (libcudart.so) and the CUDA driver (libcuda.so)".
  `CUDA_DEVICE_MEMORY_LIMIT` caps memory, and `CUDA_DEVICE_SM_LIMIT` is enforced by "a self-implemented time-sharing scheme", a software throttle rather than a hardware partition - [HAMi README](https://github.com/Project-HAMi/HAMi)
- KubeShare 2.0 pods carry `gpu_request` (guaranteed share), `gpu_limit` (burst ceiling), `gpu_mem` and a priority (0 = opportunistic); isolation is delegated to the Gemini library; tested only on Kubernetes v1.18.10 - [KubeShare README](https://github.com/NTHU-LSALAB/KubeShare)
- Kubernetes in-place pod resize is "Stable since Kubernetes v1.35", and a container's resources "are mutable for CPU and memory"; GPU extended resources are not listed as resizable - [Kubernetes docs](https://kubernetes.io/docs/tasks/configure-pod-container/resize-container-resources/)

#### Index of papers in this note
Status is "peer" for a peer-reviewed venue, "arXiv" for preprint-only.
Detailed records are under the question each paper bears on most (Q1 mechanisms, Q3 serving, Q4 memory, Q5 video).

| Paper | Venue, year | Status | ID | Code | Section |
| --- | --- | --- | --- | --- | --- |
| Gandiva | OSDI 2018 | peer | usenix osdi18/xiao | not found | Q1 |
| FGPU (Fractional GPUs) | RTAS 2019 | peer | DOI 10.1109/RTAS.2019.00011 | not found | Q1 |
| Nexus | SOSP 2019 | peer | DOI 10.1145/3341301.3359658 | github.com/uwsampl/nexus (not opened) | Q5 |
| Salus | MLSys 2020 [venue unverified] | peer? | arXiv 1902.04610 | not checked | Q1 |
| AntMan | OSDI 2020 | peer | usenix osdi20/xiao | not checked | Q1 |
| PipeSwitch | OSDI 2020 | peer | usenix osdi20/bai | not checked | Q1 |
| Clockwork | OSDI 2020 | peer | arXiv 2006.02464 | [unverified URL] | Q3 |
| GSLICE | SoCC 2020 | peer | NSF PAR 10299299 | not found | Q1 |
| InferLine | SoCC 2020 | peer | DOI 10.1145/3419111.3421285 | not checked | Q3 |
| KubeShare | HPDC 2020 | peer | DOI 10.1145/3369583.3392679 | github.com/NTHU-LSALAB/KubeShare | Q1 |
| Zico | ATC 2021 | peer | usenix atc21/lim | not checked | Q1 |
| INFaaS | ATC 2021 | peer | usenix atc21/romero | [unverified] | Q3 |
| Abacus | SC 2021 | peer | SC21 pap133 | github.com/Raphael-Hao/Abacus (search result) | Q1 |
| Gilman & Walls | IFIP Performance 2021 | peer | arXiv 2110.00459 | not found | Q1 |
| DeepRT | SEC 2021 | peer | arXiv 2105.01803 | not found | Q5 |
| Llama (video pipelines) | SoCC 2021 | peer | arXiv 2102.01887 | not found | Q5 |
| Cocktail | NSDI 2022 | peer | arXiv 2106.05345 | not checked | Q3 |
| Ekya | NSDI 2022 | peer | usenix nsdi22/bhardwaj | not found | Q5 |
| REEF | OSDI 2022 | peer | usenix osdi22/han | artifact badge, URL not confirmed | Q1 |
| Orca | OSDI 2022 | peer | usenix osdi22/yu | not found | Q4 |
| gpulet | ATC 2022 | peer | usenix atc22/choi-seungbeom | artifact badge, no URL | Q1 |
| MIG characterization (Li et al.) | IPDPSW 2022 | peer (workshop) | DOI 10.1109/IPDPSW55747.2022.00124 | not found | Q1 |
| PARIS and ELSA | DAC 2022 | peer | arXiv 2202.13481 | not checked | Q3 |
| MISO | SoCC 2022 [venue from search] | peer | arXiv 2207.11428 | github.com/boringlee24/socc22-miso | Q3 |
| INFless | ASPLOS 2022 | peer | DOI 10.1145/3503222.3507709 | not checked | Q3 |
| Jellyfish | RTSS 2022 | peer | IEEE 9984750 | not found | Q5 |
| MArk (journal version) | IEEE TCC 2022 | peer | DOI 10.1109/TCC.2020.3006751 | not checked | Q3 |
| KRISP | HPCA 2023 | peer | IEEE 10071121 | not found | Q1 |
| TGS | NSDI 2023 | peer | usenix nsdi23/wu | none on page | Q1 |
| Shepherd | NSDI 2023 | peer | usenix nsdi23/zhang-hong | not found | Q3 |
| RECL | NSDI 2023 | peer | usenix nsdi23/khani | not found | Q5 |
| Gemel | NSDI 2023 | peer | usenix nsdi23/padmanabhan | github.com/artpad6/gemel_nsdi23 | Q5 |
| AlpaServe | OSDI 2023 | peer | arXiv 2302.11665 | github.com/alpa-projects/mms | Q3 |
| iGniter | IEEE TPDS 2023 | peer | DOI 10.1109/TPDS.2022.3232715 | not checked | Q3 |
| FaST-GShare | ICPP 2023 | peer | arXiv 2309.00558 | not checked | Q3 |
| vLLM (PagedAttention) | SOSP 2023 | peer | arXiv 2309.06180 | github.com/vllm-project/vllm | Q4 |
| Paella | SOSP 2023 | peer | DOI 10.1145/3600006.3613163 | github.com/eniac/paella (search result) | Q1 |
| Miriam | SenSys 2023 (dblp) | peer | arXiv 2307.04339 | not found | Q5 |
| RAVAS | SEC 2023 | peer | DOI 10.1145/3583740.3628443 | not found | Q5 |
| Orion | EuroSys 2024 | peer | DOI 10.1145/3627703.3629578 | github.com/eth-easl/orion | Q1 |
| GMLake | ASPLOS 2024 | peer | arXiv 2401.08156 | github.com/intelligent-machine-learning/glake | Q4 |
| Proteus | ASPLOS 2024 | peer | DOI 10.1145/3617232.3624849 | not checked | Q3 |
| S-LoRA | MLSys 2024 | peer | arXiv 2311.03285 | github.com/S-LoRA/S-LoRA | Q4 |
| MuxServe | ICML 2024 | peer | arXiv 2404.02015 | github.com/hao-ai-lab/MuxServe | Q3, Q4 |
| Usher | OSDI 2024 | peer | usenix osdi24/shubha | not found | Q1 |
| Llumnix | OSDI 2024 | peer | arXiv 2406.03243 | github.com/AlibabaPAI/llumnix | Q4 |
| VTC (fairness in LLM serving) | OSDI 2024 | peer | arXiv 2401.00588 | github.com/Ying1123/VTC-artifact | Q4 |
| dLoRA | OSDI 2024 | peer | usenix osdi24/wu-bingyang | not found | Q4 |
| InfiniGen | OSDI 2024 | peer | usenix osdi24/lee | not found | Q4 |
| ServerlessLLM | OSDI 2024 | peer | usenix osdi24/fu | not found | Q4 |
| DistServe | OSDI 2024 | peer | arXiv 2401.09670 | not checked | Q4 |
| CachedAttention | ATC 2024 | peer | usenix atc24/gao-bin-cost | not found | Q4 |
| ARISE | MobiSys 2024 | peer | author PDF | not found | Q5 |
| ParvaGPU | SC 2024 | peer | arXiv 2409.14447 | not checked | Q3 |
| QLM | SoCC 2024 [venue from DOI prefix only] | peer? | arXiv 2407.00047 | github.com/QLM-project/QLM | Q4 |
| Tally | ASPLOS 2025 | peer | arXiv 2410.07381 | none linked | Q1 |
| vAttention | ASPLOS 2025 | peer | arXiv 2405.04437 | [microsoft/vattention unverified] | Q4 |
| SGDRC | PPoPP 2025 | peer | arXiv 2407.13996 | not found | Q1 |
| XSched | OSDI 2025 | peer | usenix osdi25/shen-weihang | gitee.com/openeuler/xsched (search result) | Q1 |
| GPREEMPT | ATC 2025 | peer | usenix atc25/fan | not found | Q1 |
| Torpor | ATC 2025 | peer | usenix atc25/yu | github.com/FCSLab/torpor | Q3 |
| SIRIUS | ATC 2025 | peer | usenix atc25 wang-jiali | not checked | Q3 |
| LithOS | SOSP 2025 | peer | arXiv 2504.15465 | none linked | Q1 |
| Aegaeon | SOSP 2025 | peer | author PDF (not opened) | not checked | Q4 |
| PARD | EuroSys 2026 | peer | arXiv 2602.08747 | not checked | Q3 |
| gShare | ASPLOS 2026 | peer | DOI 10.1145/3779212.3790168 | not checked | Q1 |
| FlexLLM | NSDI 2026 | peer | usenix nsdi26/oliaro | flexllm.github.io | Q3 |
| Prism (kvcached) | OSDI 2026 | peer | arXiv 2505.04021 | github.com/ovg-project/kvcached | Q4 |
| Nixie | OSDI 2026 | peer | arXiv 2601.11743 | not checked | Q1 |
| eLLM | DAC 2026 | peer (design venue) | arXiv 2506.15155 | not found | Q4 |
| KunServe | EuroSys 2026 [unverified] | arXiv? | arXiv 2412.18169 | not checked | Q4 |
| MIG-serving | - | arXiv | 2109.11067 | not found | Q3 |
| MIGPerf | - | arXiv | 2301.00407 | github.com/MLSysOps/MIGProfiler | Q1 |
| MuxFlow | - | arXiv | 2303.13803 | not checked | Q1 |
| FastServe | - | arXiv | 2305.05920 | not found | Q4 |
| Andes | - | arXiv | 2404.16283 | not checked | Q4 |
| Sarathi-Serve | OSDI 2024 [unverified] | arXiv? | 2403.02310 | not checked | Q4 |
| DARIS | DAC 2025 per one reader, arXiv-only per another | conflict | arXiv 2504.08795 | not checked | Q5 |
| MSched | - | arXiv | 2512.24637 | not checked | Q4 |
| Hummingbird | - | arXiv | 2601.04071 | not checked | Q1 |
| EMA | - | arXiv | 2609.27040 | not checked | Q4 |
| ASTRA | - | arXiv | 2609.07020 | not checked | Q5 |
| Mosaic | - | arXiv | 2610.07504 | not found | Q1 |
| Ghost | venue unknown | unverified | UCLA PDF (403) | not checked | Q1 |

#### Mechanism paper records

Characterization studies (closest to what a course project can do):
- **Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads (Gilman & Walls)** | IFIP Performance 2021 (peer) | arXiv 2110.00459 | code not found | read: full text - [arXiv](https://arxiv.org/abs/2110.00459)
  - Mechanisms compared: priority streams in one process, default time-slicing between processes, and MPS without per-client limits, on one RTX 3090 (no MIG).
  - Two tenants: a training task batched to just below OOM plus an inference task; MLPerf (BERT, ResNet-34 inference, RNNT training) and five PyTorch examples; Poisson (500 requests) or back-to-back (5000 requests); metrics are mean and variance of inference turnaround and training time; no SLO.
  - Claims: the "lack of fine-grained preemption mechanisms, robust task prioritization options, and contention-aware thread block placement policies limits the effectiveness" of all three.
    Time-slicing is the most predictable but never overlaps tasks; the slice length "is fixed to approximately 2ms" and they found no way to change it.
    MPS raises utilisation but intra-SM contention slows both tasks, and most of the slowdown lands on inference.
    MPS "is perhaps best suited to cases where the kernels utilize less than the total available resources of the GPU".
- **Characterizing Multi-Instance GPU for Machine Learning Workloads (B. Li, Gadepally, Samsi, Tiwari)** | IPDPSW 2022, pp. 724-731 (peer, workshop) | DOI 10.1109/IPDPSW55747.2022.00124 | read: listing only - [IEEE CSDL](https://computer.org/csdl/proceedings-article/ipdpsw/2022/974700a724/1Fu9oBpaenm)
  - Content [unverified]: not read.
  - The arXiv id 2209.09356 given in the brief belongs to an unrelated paper.
- **MIGPerf** | arXiv-only 2023 ("still in progress") | arXiv 2301.00407 | [code](https://github.com/MLSysOps/MIGProfiler) | read: part of full text - [arXiv](https://arxiv.org/abs/2301.00407)
  - A100 benchmark of MIG against MPS (1/2/4 instances against 1/2/4 MPS processes, ResNet18/50, batch 1-32); states that "MPS can not provide real isolation, the interference can not be avoided especially for large workloads".
  - No headline number extracted.
- **An Analysis of Collocation on GPUs for Deep Learning Training (Robroek et al.)** | venue [unverified] | arXiv 2209.06018 - [arXiv](https://arxiv.org/abs/2209.06018)
  - From a search excerpt only: "MIG can significantly improve the utilization of the GPU when the workload is too small to utilize the whole GPU in isolation", with smaller gains for larger workloads.

Preemption and priority (one latency-critical tenant plus best-effort tenants):
- **REEF: Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences** | OSDI 2022 (peer) | read: full text - [USENIX](https://www.usenix.org/conference/osdi22/presentation/han)
  - Reset-based preemption (kills and re-runs best-effort kernels, relying on inference kernels being "mostly idempotent") plus dynamic kernel padding; needs kernel code instrumentation and a modified AMD ROCm runtime; NVIDIA got only a restricted port.
  - Claim: real-time task latency overhead under 2% while throughput rises "by up to 7.7x, compared to dedicating the GPU to real-time tasks", on one AMD MI50 with the DISB benchmark and a real trace.
- **GPREEMPT: GPU Preemptive Scheduling Made General and Efficient** | ATC 2025 (peer) | read: partial full text - [USENIX](https://www.usenix.org/conference/atc25/presentation/fan)
  - Context-switch preemption via a timeslice-based yield plus hint-based pre-preemption; does not need idempotent kernels; preemption latency within 40 us.
  - Claim, A100-40GB: average latency increase of latency-critical tasks over running alone is 58.4% (multi-stream, no preemption), 96.3% (sequential), 15.3% (block-level preemption) and 2.4% (GPREEMPT).
- **XSched: Preemptive Scheduling for Diverse XPUs** | OSDI 2025 (peer) | read: partial full text - [USENIX](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)
  - A preemptible command-queue abstraction (XQueue) ported to 10 XPUs; fixed-priority and bandwidth-partition policies.
  - Claims: high-priority P99 "up to 2.10x" lower than the native hardware scheduler; with two containers it harvests "2.74x more GPU resources than TGS while maintaining production container performance"; in Triton multi-model serving P99 drops 30.0%, comparable to Paella.
- **Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads** | ASPLOS 2025 (peer) | arXiv 2410.07381 | DOI 10.1145/3669940.3707282 | read: full text - [arXiv](https://arxiv.org/abs/2410.07381)
  - `LD_PRELOAD` interception and thread-block-level kernel slicing and preemption; no application changes.
  - Claim: average p99 overhead of high-priority inference "of only 7.2%", against 252.3% (time-slicing), 345.0% (MPS), 195.5% (MPS with priority) and 188.9% (TGS), while "achieving over 80% of TGS's system throughput".
  - Evaluation: one A100 40 GB; high-priority inference with best-effort training (ResNet50, PointNet, BERT, GPT2-Large, PEGASUS, Whisper-v3); Azure Functions Trace 2021 arrivals.
- **LithOS: An Operating System for Efficient Machine Learning on GPUs** | SOSP 2025 (peer) | arXiv 2504.15465 | read: abstract and setup - [arXiv](https://arxiv.org/abs/2504.15465)
  - TPC-level spatial scheduler with TPC stealing, kernel atomization against head-of-line blocking, hardware right-sizing and power management; transparent to applications.
  - Claims: inference stacking tail latency 13x lower than MPS (3x lower and 1.6x throughput against the best prior system); inference plus training tail 4.7x lower than MPS; right-sizing saves a quarter of GPU capacity at under 4% cost; one A100 40 GB; baselines include time slicing, MPS, priority streams, MIG, TGS, REEF and Orion.
- **Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale** | arXiv-only, Jan 2026 | arXiv 2601.04071 | read: abstract - [arXiv](https://arxiv.org/abs/2601.04071)
  - Claims: high-priority SLO attainment 9.7x and 3.5x better than the best spatial- and temporal-sharing systems, under 1% drop against running alone, low-priority throughput 2.4x the best temporal-sharing system.

Spatial sharing and interference:
- **Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications** | EuroSys 2024 (peer) | DOI 10.1145/3627703.3629578 | [code](https://github.com/eth-easl/orion) | read: README; numbers from an unofficial PDF mirror [unverified against the ACM version]
  - Intercepts CUDA, cuDNN and cuBLAS calls and dispatches each operator by its profiled SM needs and priority (one high-priority client, the rest best-effort); needs offline per-operator profiles.
  - Claims: with best-effort inference, aggregate throughput up to 7.3x against a dedicated GPU with high-priority p99 within 15% on average; with training, p99 overhead about 14%.
  - The README says it was tested on H100 and RTX 3090 with CUDA 12.6.
- **Mosaic: GPU Sharing with Latency Guarantees through Kernel-Level Interference Prediction** | arXiv-only, Oct 2026 | arXiv 2610.07504 | read: abstract - [arXiv](https://arxiv.org/abs/2610.07504)
  - A kernel-level interference predictor plus online kernel admission control that chooses between full-GPU colocation and SM partitioning; "keeps the p99 latency below or very close to the target SLO" on four GPU architectures.
- **gpulet: Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing** | ATC 2022 (peer) | read: abstract - [USENIX](https://www.usenix.org/conference/atc22/presentation/choi-seungbeom)
  - Virtual GPU partitions plus temporal sharing and batch-size selection, with an interference model; premise is that "with SLO constraints, GPUs ... are not fully utilized".
  - Claim: throughput up "61.7% on average compared to the prior temporal scheduler, while satisfying SLOs" on RTX 2080 Ti servers.
- **GSLICE: Controlled Spatial Sharing of GPUs for a Scalable Inference Platform** | SoCC 2020 (peer) | read: search excerpt only - [NSF PAR](https://par.nsf.gov/biblio/10299299)
  - Controlled MPS-based spatial sharing with dynamic reallocation and adaptive batching; 60-800% better utilisation efficiency than default MPS and TensorRT [unverified].
- **Usher: Holistic Interference Avoidance for Resource Optimized ML Inference** | OSDI 2024 (peer) | read: introduction - [USENIX](https://www.usenix.org/conference/osdi24/presentation/shubha)
  - Kernel-based resource estimation, interference-aware placement and operator graph merging; up to 2.6x goodput and 3.5x cost-efficiency against Shepherd, gpulet and AlpaServe.
- **SGDRC: Software-Defined Dynamic Resource Control for Concurrent DNN Inference on NVIDIA GPUs** | PPoPP 2025 (peer) | arXiv 2407.13996 (earlier listed under the name "Missile", not confirmed) | DOI 10.1145/3710848.3710863 | read: abstract - [arXiv](https://arxiv.org/abs/2407.13996)
  - Software cache colouring against VRAM channel conflicts and "tidal SM masking"; SLO attainment 99.0% on average, overall throughput up to 1.47x and best-effort throughput up to 2.36x against the best sharing systems.
- **KRISP** (HPCA 2023, peer), **Abacus** (SC 2021, peer) and **FGPU** (RTAS 2019, peer): kernel-wise right-sizing of spatial partitions (KRISP, which ECLIP says needs microarchitectural extensions), deterministic operator overlap with latency prediction (Abacus: 51.3% fewer QoS violations, 29.8% more throughput [search excerpt only]), and software compute plus memory-bandwidth reservation by page colouring (FGPU) - [KRISP](https://ieeexplore.ieee.org/document/10071121); [Abacus](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap133.html); [FGPU](https://doi.org/10.1109/RTAS.2019.00011)

Temporal sharing, memory sharing and cluster systems:
- **Gandiva** | OSDI 2018 (peer) | read: abstract - [USENIX](https://www.usenix.org/conference/osdi18/presentation/xiao)
  - Application-level suspend-resume time-slicing at mini-batch boundaries plus migration; speeds up hyperparameter search "by up to an order of magnitude"; the gain is early feedback for every job, not added throughput.
- **Salus** | MLSys 2020 [venue unverified] | arXiv 1902.04610 | read: abstract - [arXiv](https://arxiv.org/abs/1902.04610)
  - Iteration-level scheduling and memory sharing; inference GPU utilisation 42x higher than no sharing and 7x higher than MPS, from packing applications that each underuse the GPU.
- **AntMan** | OSDI 2020 (peer) | read: abstract - [USENIX](https://www.usenix.org/conference/osdi20/presentation/xiao)
  - In-framework dynamic scaling of memory and compute; opportunistic jobs use spare resources; memory utilisation +42% and compute +34% in Alibaba's cluster "without compromising fairness".
- **PipeSwitch** | OSDI 2020 (peer) | read: abstract - [USENIX](https://www.usenix.org/conference/osdi20/presentation/bai)
  - Pipelined model transfer for millisecond context switching between tasks owning the whole GPU memory; startup overhead 3.6-6.6 ms, "10-50x better than NVIDIA MPS".
- **Zico** | ATC 2021 (peer) | read: abstract - [USENIX](https://www.usenix.org/conference/atc21/presentation/lim)
  - Memory reclamation and sharing between concurrent training jobs; no numbers in the abstract.
- **TGS: Transparent GPU Sharing in Container Clouds for Deep Learning Workloads** | NSDI 2023 (peer) | read: full text - [USENIX](https://www.usenix.org/conference/nsdi23/presentation/wu)
  - OS-layer sharing below containers: adaptive rate control throttles opportunistic jobs' kernel launches, CUDA unified memory allows oversubscription; integrated with Docker and Kubernetes.
  - Claims: little effect on production jobs; opportunistic jobs get throughput close to AntMan and "up to 15x" that under MPS; MPS "requires application knowledge to set resource limits" and fails when total memory demand exceeds the GPU.
- **MuxFlow** | arXiv-only 2023 | arXiv 2303.13803 | read: abstract - [arXiv](https://arxiv.org/abs/2303.13803)
  - Production MPS sharing with memory and compute protection and dynamic SM allocation; over 20,000 GPUs, utilisation 26% to 76%, SM activity 16% to 33%.
- **Paella: Low-latency Model Serving with Software-defined GPU Scheduling** | SOSP 2023 (peer) | DOI 10.1145/3600006.3613163 | read: full text - [PDF](https://homes.cs.washington.edu/~vincent/files/ng23paella.pdf)
  - Compiler (modified TVM), client and scheduler co-design that takes kernel ordering away from the hardware scheduler; models must be recompiled.
  - Claims: the scheduling policy alone sustains "1.4x more load and 1.35x lower latency than the baseline"; the whole system sustains up to 58x more load with 11x lower latency than Triton on one T4 with lognormal (bursty) arrivals.
- **Nixie: Efficient, Transparent Temporal Multiplexing for Consumer GPUs** | OSDI 2026 (peer; listed in the OSDI '26 program) | arXiv 2601.11743 | read: abstract - [arXiv](https://arxiv.org/abs/2601.11743); [OSDI '26 program](https://www.usenix.org/conference/osdi26/technical-sessions)
  - A system service that coordinates memory allocation and kernel launches on a consumer GPU, with an MLFQ-style scheduler favouring interactive jobs; compared with NVIDIA Unified Virtual Memory.
  - Claims: up to 3.8x lower latency for interactive code completion and up to 66.8% less CPU pinned memory at the same latency.
- **gShare: Efficient GPU Sharing with Aggressive Scheduling in Multi-tenant FaaS platform** | ASPLOS 2026 (peer) | DOI 10.1145/3779212.3790168 | read: abstract and Sec. 3 - [ACM PDF](https://cis.temple.edu/~jiewu/research/publications/Publication_files/3779212.3790168.pdf)
  - VM-based multi-tenant FaaS: user-transparent vGPU remapping, round-robin kernel time slices proportional to each function's memory share (20 ms period), vGPU hot-plugging and model swapping to oversell idle vGPU memory.
  - Claim: 43%-63% lower GPU usage than the baseline.
- **Breaking the Tradeoff: Elastic and Isolated GPU Sharing with Ghost** | venue unknown | read: search snippet only (PDF returned 403) - [UCLA PDF](https://web.cs.ucla.edu/~harryxu/papers/GVM.pdf)
  - [unverified] An OS-level GPU virtualization layer in the open-source NVIDIA driver with a "GPU container abstraction with cgroup-like APIs for compute and memory control" and per-container memory accounting with proactive swapping, adapting allocations "without requiring GPU resets or interrupting running applications".

### Inferences
- NVIDIA's MPS v3 (documented in the Release 620 guide) now ships in the driver what several papers built in software: weighted time slices per server, SM partitions that lend and borrow idle SMs, and per-cgroup memory with a guaranteed soft limit, a hard limit, and eviction of borrowers.
  For a course project, this means "MPS cannot change a running client's limits" (the project's probe, confirmed by the MPS doc) is a property of legacy MPS v2 and of the installed driver, not a permanent fact.
- Two things in MPS v3 still do not change limits under running clients: weighted time slices reject updates while clients are attached, and static partitions cannot be removed while in use.
  Only the memory soft/hard model behaves elastically at runtime, and even there the elasticity is "borrow, then get killed", not "grow on demand".
- The kernel-level schedulers (REEF, Orion, Tally, LithOS, XSched, GPREEMPT) all assume asymmetric tenants: one latency-critical tenant and best-effort tenants.
  None of the papers read here targets N equal-priority interactive tenants running the same heavy model, which is the project's setting.

### Gaps
- The MIG characterization paper (Li et al., IPDPSW 2022) and GSLICE, KRISP, Abacus and FGPU were not read beyond listings or search excerpts, so their numbers are not quoted.
- I could not confirm Ghost's venue or read its text; it is the work closest to "cgroup-like GPU memory control that adapts at runtime" and is worth reading first if the PDF becomes reachable.
- Whether `nvidia-smi compute-policy --set-timeslice` needs root, and whether it changes the roughly 2 ms slice Gilman & Walls measured on an RTX 3090, was not tested on the RTX 6000 Ada.
- The OSDI '26 program was read in full for titles; SOSP 2026's accepted-papers page still read "TBD" when searched, and only one SOSP 2026 poster title ("Beyond Utilization: Energy-Conscious GPU Sharing for Inference Serving") was found - [SOSP 2026 posters](https://sigops.org/s/conferences/sosp/2026/posters.html).
  NSDI 2026, EuroSys 2026 and ASPLOS 2026 were covered by keyword search only, so the 2026 coverage is incomplete.
- A BLESS paper ("bubble-less spatial-temporal GPU sharing") could not be found.

## 2. On compute-bound workloads, which papers still got gains, and from what?

### Takeaway
No paper read here claims that MPS or time-slicing adds throughput once one tenant saturates the GPU, and two papers say outright that it does not.
Every verified throughput gain is measured against a dedicated or underused GPU.
Under saturation, papers report gains of four other kinds: the latency of a high-priority tenant (preemption and priority), goodput under overload (rejecting or dropping doomed requests), capacity bought by lowering each tenant's work (rate, resolution, model variant), and early progress or fairness (time-slicing as in Gandiva, VTC).

### Cited Findings
- The project's own runs: time-slicing and default MPS served the same number of users in every capacity test (G2, G3, G6, G7), StreamSplat is compute-bound near 4.7-5 updates/s in total (G5, G6), and batching users gave 4.52 updates/s at batch 8 against 4.92 at batch 1 (G9) - [project LOG.md](../../../LOG.md)
- Clockwork measured that running inferences concurrently "gains up to 25% throughput ... at a cost of substantially increased latency variability" (ResNet50, V100), and that "one-at-a-time execution of DNN inferences on GPUs has closely comparable throughput to concurrent execution" - [Clockwork, OSDI 2020](https://arxiv.org/abs/2006.02464)
- AlpaServe: "when the arrival rate approaches the peak serving rate of the cluster, the benefit of model-parallel placement starts to diminish. Eventually, it starts to perform worse than replication", "because when all models are equally saturated" - [AlpaServe, OSDI 2023](https://arxiv.org/abs/2302.11665)
- Gilman & Walls: time-slicing never overlaps two tasks, and MPS "is perhaps best suited to cases where the kernels utilize less than the total available resources of the GPU" - [arXiv 2110.00459](https://arxiv.org/abs/2110.00459); NVIDIA says the same of MPS in its own documentation - [MPS doc](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- Priority and preemption, with a best-effort tenant saturating the GPU:
  - Tally holds high-priority p99 overhead at 7.2%, against 345.0% under MPS and 252.3% under time-slicing, at the cost of reaching ">80% of TGS's system throughput" - [Tally, ASPLOS 2025](https://arxiv.org/abs/2410.07381)
  - GPREEMPT: 2.4% average latency increase for latency-critical tasks, against 58.4% with multi-stream sharing - [GPREEMPT, ATC 2025](https://www.usenix.org/conference/atc25/presentation/fan)
  - LithOS: tail latency 13x lower than MPS when stacking inference, 4.7x lower with training - [LithOS, SOSP 2025](https://arxiv.org/abs/2504.15465)
  - REEF: under 2% overhead for the real-time task - [REEF, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/han)
  - XSched: high-priority P99 up to 2.10x lower than the native scheduler - [XSched, OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)
  - TGS: production jobs barely affected while opportunistic jobs take what is left - [TGS, NSDI 2023](https://www.usenix.org/conference/nsdi23/presentation/wu)
- Overload control:
  - Clockwork "achieves close to ideal goodput even under overload" by rejecting requests that would miss even if run immediately - [Clockwork](https://arxiv.org/abs/2006.02464)
  - Shepherd proves that "no online algorithm can achieve performance competitive with the optimal offline solution without using preemption" and gets up to 5.2x goodput over Clockwork-style heuristics, mostly from choosing which requests to drop or preempt in bursts - [Shepherd, NSDI 2023](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong)
  - PARD drops requests proactively before they waste upstream compute: 16%-176% higher goodput and 1.5x-62x less wasted computation - [PARD, EuroSys 2026](https://arxiv.org/abs/2602.08747)
  - Nexus: early drop gave up to 25% more throughput than lazy drop (simulation, 100 ms SLO) - [Nexus, SOSP 2019](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf)
- Lowering per-tenant work:
  - ARISE supports "1.7x-6.9x more clients" by computing a near-minimal offloading frequency per AR client and aligning request timing so requests batch - [ARISE, MobiSys 2024](https://qiangxu1996.github.io/assets/pdf/Kong2024_ARISEOffloadingAR.pdf)
  - Chameleon reaches the same accuracy "with only 30-50% of the resources" by adapting frame rate, resolution and model - [Chameleon, SIGCOMM 2018](https://www.microsoft.com/en-us/research/?p=486414)
  - Ekya and RAVAS degrade per-stream configuration as streams are added - [Ekya, NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/bhardwaj); [RAVAS, SEC 2023](https://lass.cs.umass.edu/papers/pdf/sec2023-ravas.pdf)
- Fairness: VTC proves a 2x bound on the service difference between two backlogged clients under a work-conserving scheduler, scheduling compute share by tokens processed - [VTC, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/sheng)
- Gandiva's time-slicing adds no throughput; its gain is that every job gets early feedback, which shortened hyperparameter search "by up to an order of magnitude" - [Gandiva, OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/xiao)
- Gains that do not transfer to a saturated GPU: REEF's 7.7x and Orion's 7.3x throughput are measured against "dedicating the GPU" to an underusing real-time task - [REEF](https://www.usenix.org/conference/osdi22/presentation/han); [Orion](https://github.com/eth-easl/orion); gpulet's premise is that GPUs under SLOs "are not fully utilized" - [gpulet](https://www.usenix.org/conference/atc22/presentation/choi-seungbeom); Miriam fills idle SMs left by small kernels - [Miriam](https://arxiv.org/abs/2307.04339)
- Batching gains in video serving were reported on small models: in Nexus's game case ResNet-50 takes about 6 ms at batch 1, and its largest single factor was overlapping CPU pre-processing with GPU work, without which "roughly half the cycles of the GPU" sat idle - [Nexus](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf)

### Inferences
- The project's G-series results are what this literature predicts, so the result "MPS and time-slicing do not serve more users of a saturating model" would not surprise a systems reader.
  What would be new is stating it for N equal-priority interactive tenants and then measuring what does change: per-user latency distribution, fairness, and overload behaviour.
- Preemption systems (Tally, REEF, GPREEMPT, LithOS) need either kernel instrumentation, `LD_PRELOAD` interception with offline profiles, or AMD runtimes.
  Of the systems read, only Orion (open code, tested on RTX 3090 and H100 with CUDA 12.6) looks plausible to run on the RTX 6000 Ada within weeks, and the first run would have to be a reproduction.
- A 0.19 s StreamSplat update is about 100x longer than Gilman & Walls' measured 2 ms time slice, so under time-slicing each update is cut into many slices, while under a FIFO in one process it runs whole.
  That predicts different per-user latency distributions with the same total throughput, which is measurable.

### Gaps
- No paper found measures N identical heavy tenants with equal priority and reports per-tenant fairness or tail latency across vendor sharing modes; this is a gap in what was found, not a proof that none exists.
- The trade-off of priority systems against total throughput is reported only by Tally (">80% of TGS"); the other systems' throughput costs were not extracted.

## 3. How do these papers evaluate: traces, SLO definitions, metrics, baselines, hardware, tenants?

### Takeaway
There is a standard recipe that a course project can copy.
Arrival rates come from the Microsoft Azure Functions traces (MAF 2019, MAF 2021), with Poisson, MMPP or Gamma inter-arrivals at tunable burstiness.
The SLO is either an absolute per-request deadline or a multiple of solo latency.
Goodput or SLO attainment is swept against offered load past saturation, and compared with FCFS, reactive dropping and early rejection.
Kernel-level papers use one GPU and two tenants; serving papers use 1 to 64 GPUs (plus simulators for hundreds) and tens to thousands of model instances.

### Cited Findings

#### Serving and partitioning paper records
- **Clockwork: Serving DNNs like Clockwork: Performance Predictability from the Bottom Up** | OSDI 2020 (peer) | arXiv 2006.02464 | code URL [unverified] | read: full text - [USENIX](https://www.usenix.org/conference/osdi20/presentation/gujarati); [arXiv](https://arxiv.org/abs/2006.02464)
  - Workers run one inference at a time for predictability; a controller sends actions with [earliest, latest] start windows and "cancels the request before performing any fruitless work".
  - Claim: "can support thousands of models while simultaneously meeting 100ms latency targets for 99.9999% of requests".
  - Evaluation: 6 hours of MAF 2019 replayed in real time; 61 TVM-compiled models duplicated 66 times (4,026 instances); 12 workers with 2x V100 32 GB each; SLO is an absolute deadline swept from 2.9 to 500 ms.
    Goodput is "the number of successful requests that completed within the target SLO; it excludes timed out requests and requests that responded after the SLO"; workload satisfaction is goodput divided by offered load.
    Baselines Clipper and INFaaS; Clipper "only treats SLOs as an average latency target", and both baselines' goodput "collapses below a 100ms SLO".
    Sec. 6.4 runs 6 latency-sensitive clients at 200 r/s each with 12 or 48 batch clients without SLOs; latency-sensitive satisfaction was unaffected.
- **SHEPHERD: Serving DNNs in the Wild** | NSDI 2023 (peer) | code not found | read: full text - [USENIX](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong)
  - A periodic ILP planner (HERD) plus an online preemptive scheduler per group (FLEX) with a constant-factor goodput guarantee.
  - Claim: "up to 18.1x higher goodput and 1.8x higher utilization" than Nexus-style per-stream planning and "up to 5.2x higher goodput" than Clockwork-style heuristics.
  - Evaluation: 46,000 MAF streams mapped onto 13 models; within-minute arrivals Poisson ("MAF-stable") or MMPP ("MAF-bursty"); per-stream SLO default 250 ms, also swept; goodput is "the number of requests that meet their SLO deadlines per unit time", averaged over 5 runs; 12 AWS p3.2xlarge (1x V100 16 GB each) plus emulation up to hundreds of workers; baselines reimplemented in Shepherd's runtime.
- **AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving** | OSDI 2023 (peer) | arXiv 2302.11665 | [code](https://github.com/alpa-projects/mms) | read: full text - [arXiv](https://arxiv.org/abs/2302.11665)
  - Model parallelism even for models that fit one GPU, so bursts can use several GPUs.
  - Claim: up to 10x higher request rate or 6x more burstiness at over 99% SLO attainment.
  - Evaluation: MAF1 and MAF2 sliced into windows, fitted with Gamma processes (rate, CV) and rescaled; SLO is an "SLO scale" multiple of single-device latency; requests "that will exceed the deadline even if we schedule it immediately" are dropped; metric is SLO attainment; 64 V100 16 GB, mostly simulated (simulator validated); baselines Selective Replication and Clockwork++.
- **MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving** | ICML 2024 (peer) | arXiv 2404.02015 | [code](https://github.com/hao-ai-lab/MuxServe) | read: full text - [PMLR](https://proceedings.mlr.press/v235/duan24a.html)
  - Co-locates LLMs by popularity and splits SMs between prefill and decode jobs "using CUDA MPS", with a unified KV and memory manager on vLLM.
  - Claim: "up to 1.8x higher throughput", or 2.9x more requests, within 99% SLO attainment; 19 LLMs on 32 A100 80 GB; power-law per-model rates with Poisson arrivals; SLO scale relative to single-device latency.
- **MIG-serving: Serving DNN Models with Multi-Instance GPUs: A Case of the Reconfigurable Machine Scheduling Problem** | arXiv-only 2021 | arXiv 2109.11067 | read: full text - [arXiv](https://arxiv.org/abs/2109.11067)
  - Formulates reconfigurable machine scheduling (NP-hard), solved with greedy, genetic and MCTS algorithms on Kubernetes with live repartitioning.
  - Claim: "can save up to 40% GPUs while providing the same throughput" against A100s with MIG disabled, on 24 A100s; transitions finish within half an hour; the gain is packing models that cannot fill an A100.
- **ParvaGPU** | SC 2024 (peer) | arXiv 2409.14447 | read: abstract and introduction - [arXiv](https://arxiv.org/abs/2409.14447)
  - MIG plus MPS "segments" sized per workload; "no SLO violations and a significant reduction in GPU usage" against GSLICE, gpulet, iGniter, PARIS-ELSA and MIG-serving on 8x A100 80 GB with 11 workloads.
- **MISO** | SoCC 2022 [venue from search results] | arXiv 2207.11428 | [code](https://github.com/boringlee24/socc22-miso) - [arXiv](https://arxiv.org/abs/2207.11428)
  - Predicts the best MIG partition from MPS co-location profiles to avoid reconfiguration during exploration; "49% and 16% lower average job completion time than the unpartitioned and optimal static GPU partition schemes"; the metric is job completion time, not an inference SLO.
- **PARIS and ELSA** | DAC 2022 (peer) | arXiv 2202.13481 - [arXiv](https://arxiv.org/abs/2202.13481)
  - A heterogeneous mix of MIG partition sizes plus an elastic scheduler; numbers not extracted.
- **SIRIUS: Colocating ML Inference and Training with Fast GPU Memory Handover** | ATC 2025 (peer) | read: partial full text - [USENIX PDF](https://www.usenix.org/system/files/atc25-wang-jiali.pdf)
  - Hands GPU memory between inference and training at millisecond scale during bursts; SLO "set as four times of inference model standalone execution time"; "an average of 57.0% (up to 97.0%) improvement in inference SLO compliance and 2.2x (up to 13.7x) speedup in training throughput"; MAF (capped at 150 req/s) and BurstGPT traces; V100 and A100.
- **Torpor** | ATC 2025 (peer) | [code](https://github.com/FCSLab/torpor) | read: partial full text - [USENIX](https://www.usenix.org/conference/atc25/presentation/yu)
  - Serverless inference that swaps models from host memory onto GPUs on request arrival; motivated by an Alibaba trace where 85% of functions are invoked at most once per minute; headline numbers not extracted.
- **PARD: Enhancing Goodput for Inference Pipeline via Proactive Request Dropping** | EuroSys 2026 (peer, per arXiv comment and paper header) | arXiv 2602.08747 | read: abstract and partial full text - [arXiv](https://arxiv.org/abs/2602.08747)
  - Wikipedia, Twitter and Azure Functions traces; 64 GPUs; baselines Nexus and Clipper (reactive dropping) and no-drop; goodput is requests completed within the latency objective.
- **FlexLLM: Token-Level Co-Serving of LLM Inference and Finetuning with SLO Guarantees** | NSDI 2026 (peer) | read: search summary - [USENIX](https://www.usenix.org/conference/nsdi26/presentation/oliaro)
  - Throttles finetuning tokens inside the same kernel when inference load spikes; keeps inference SLO compliance up to 20 req/s [numbers from search summary].
- **INFaaS** | ATC 2021 (peer, Best Paper) | read: abstract - [USENIX](https://www.usenix.org/conference/atc21/presentation/romero)
  - Picks model variant and hardware per query from declared latency and accuracy needs; 1.3x throughput, 1.6x fewer latency violations, up to 21.6x cost savings on AWS EC2.
- **InferLine** | SoCC 2020 (peer) | arXiv 1812.01776 | read: abstract - [arXiv](https://arxiv.org/abs/1812.01776)
  - Offline planner plus network-calculus online tuner for pipelines; "up to 7.6x in cost while achieving up to 34.5x lower latency SLO miss rate".
- **Cocktail** (NSDI 2022, arXiv 2106.05345), **Proteus** (ASPLOS 2024, DOI 10.1145/3617232.3624849), **INFless** (ASPLOS 2022, DOI 10.1145/3503222.3507709), **MArk** (journal version IEEE TCC 2022, DOI 10.1109/TCC.2020.3006751), **iGniter** (IEEE TPDS 2023, DOI 10.1109/TPDS.2022.3232715) and **FaST-GShare** (ICPP 2023, arXiv 2309.00558): venues verified, contents not read beyond Cocktail's abstract ("reduce deployment cost by 1.45x, while providing 2x reduction in latency") - [Cocktail](https://arxiv.org/abs/2106.05345); [FaST-GShare](https://arxiv.org/abs/2309.00558)
- Other 2024-2026 hits seen only in search snippets [all unverified]: Mudi (EuroSys 2025, inference-training multiplexing under SLOs), MIGRator (arXiv 2407.13126, dynamic MIG reconfiguration), "Predictable LLM Serving on GPU Clusters" (arXiv 2508.20274), HAS-GPU (Euro-Par 2025), Roomie (arXiv 2607.16784), DetShare (arXiv 2603.15042), DuetServe (arXiv 2511.04791), Drift (arXiv 2504.14489).

#### Evaluation recipe across papers
- Traces: MAF 2019 is used by Clockwork, Shepherd, AlpaServe and SIRIUS; MAF 2021 by AlpaServe and Tally; PARD adds Wikipedia and Twitter; Prism uses Hyperbolic and Arena-Chat - [Clockwork](https://arxiv.org/abs/2006.02464); [Shepherd](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong); [AlpaServe](https://arxiv.org/abs/2302.11665); [Tally](https://arxiv.org/abs/2410.07381); [PARD](https://arxiv.org/abs/2602.08747); [Prism](https://arxiv.org/abs/2505.04021)
- SLO forms: absolute per-request deadlines (Clockwork 2.9-500 ms; Shepherd 250 ms; Nexus 50 and 400 ms; RAVAS 80-1000 ms) or multiples of solo latency (AlpaServe and MuxServe "SLO scale"; SIRIUS 4x; Prism per-model P95 TTFT and TPOT on a dedicated GPU times a scale factor) - same sources plus [Nexus](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf); [RAVAS](https://lass.cs.umass.edu/papers/pdf/sec2023-ravas.pdf); [SIRIUS](https://www.usenix.org/system/files/atc25-wang-jiali.pdf)
- Metrics: goodput with late and rejected requests counted as failures (Clockwork, Shepherd, PARD); SLO attainment (AlpaServe, MuxServe, Prism, SGDRC); capacity at a fixed attainment, such as the maximum rate with 99% of requests within SLO (Nexus, AlpaServe); p99 of the high-priority tenant (Tally, LithOS, XSched); for LLMs, TTFT, TPOT and goodput as "the maximum rate that can be served within both TTFT and TPOT constraints on each GPU" - [DistServe](https://arxiv.org/abs/2401.09670)
- Hardware and tenant counts: kernel-level papers use one GPU and two tenants (Gilman & Walls on an RTX 3090; Tally and LithOS on one A100; REEF on one AMD MI50; Paella on one T4); serving papers use 12-64 GPUs plus simulation, with 13 to 4,026 model instances - sources as above.

### Inferences
- The project already measures throughput and capacity at fixed per-user rates; the missing pieces relative to this recipe are an arrival process with burstiness (Poisson or Gamma with CV), an SLO stated before the run, goodput with late updates counted as failures, and a sweep of offered load past the roughly 5 updates/s saturation point.
- Clockwork's core premise, that one-at-a-time execution is predictable enough for deadline-aware rejection, can be checked cheaply for StreamSplat by measuring the CV and p99/median of its 0.19 s service time solo and under each sharing mode.

### Gaps
- INFaaS, Proteus, INFless, MArk, iGniter, FaST-GShare and KubeShare evaluations were not read, so their SLO definitions are not recorded here.
- Search coverage of 2024-2026 serving papers relied on a few keyword searches rather than a program-by-program pass.

## 4. Which papers treat per-tenant memory that grows over time, and what mechanisms do they use?

### Takeaway
LLM serving solved growing per-request memory (the KV cache) inside one cooperating process: allocate pages on demand (vLLM, vAttention), and when memory runs out, preempt with swap or recompute, migrate, offload, or take memory from co-located models.
Prism (OSDI 2026) with its kvcached library is the closest prior art to "grow a tenant's budget as its cache grows": a global scheduler moves per-model memory limits on one GPU using CUDA virtual memory, at millisecond-level overhead.
Prism's limits are enforced by the engines' own allocator, not by a container runtime.
Raising an externally enforced limit on a running, non-cooperating process is still mostly open, but NVIDIA's MPS v3 cgroup memory partitioning (Release 620 guide, CUDA 13.4) now provides a soft/hard "borrow then be evicted" model in the driver.

### Cited Findings

#### Memory paper records
- **vLLM: Efficient Memory Management for LLM Serving with PagedAttention** | SOSP 2023 (peer) | arXiv 2309.06180 | [code](https://github.com/vllm-project/vllm) | read: full text - [arXiv](https://arxiv.org/abs/2309.06180)
  - KV cache in fixed blocks (default 16 tokens); when blocks run out, whole sequences are preempted ("all-or-nothing eviction policy") and either swapped to CPU memory or dropped and recomputed in one prefill pass; no new requests are admitted while any are preempted.
  - Claims: existing systems use "only 20.4% - 38.2% of the KV cache memory ... to store the actual token states" (vLLM 96.3%; the "Orca" arms are the authors' emulations of reservation policies); "1.7x-2.7x higher request rates compared to Orca (Oracle) and 2.7x-8x compared to Orca (Max)" on ShareGPT.
  - Swap versus recompute: swapping has "excessive overhead with small block sizes"; recomputation is "never higher than 20% of swapping's latency".
  - Evaluation: OPT-13B/66B/175B and LLaMA-13B on A100 (1, 4, 8 GPUs); ShareGPT and Alpaca; normalized latency versus request rate.
- **Orca** | OSDI 2022 (peer) | read: abstract - [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
  - Iteration-level scheduling and selective batching; "36.9x throughput improvement at the same level of latency" against FasterTransformer on GPT-3 175B; its memory reservation policy is described only through vLLM's emulation [unverified from Orca's own text].
- **Llumnix** | OSDI 2024 (peer) | arXiv 2406.03243 | [code](https://github.com/AlibabaPAI/llumnix) | read: abstract - [arXiv](https://arxiv.org/abs/2406.03243)
  - Live migration of running requests with their KV state across instances for load balancing, de-fragmentation and priority isolation; tail latency improved by an order of magnitude, high-priority requests up to 1.5x faster, cost down up to 36% at similar tail latency.
- **vAttention** | ASPLOS 2025 (peer) | arXiv 2405.04437 | read: abstract - [arXiv](https://arxiv.org/abs/2405.04437)
  - Keeps the KV cache contiguous in virtual memory and maps physical pages on demand with CUDA VMM APIs; up to 1.23x throughput over PagedAttention kernels of FlashAttention and FlashInfer.
- **VTC: Fairness in Serving Large Language Models** | OSDI 2024 (peer) | arXiv 2401.00588 | [code](https://github.com/Ying1123/VTC-artifact) | read: abstract - [USENIX](https://www.usenix.org/conference/osdi24/presentation/sheng)
  - A virtual token counter charges each client for tokens processed; proven 2x bound on service difference between backlogged clients; schedules compute share, not memory.
- **dLoRA** | OSDI 2024 (peer) - [USENIX](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang); **S-LoRA** | MLSys 2024 (peer) - [arXiv](https://arxiv.org/abs/2311.03285)
  - S-LoRA's "Unified Paging" puts adapter weights and KV tensors in one paged pool (up to 4x throughput over HuggingFace PEFT and naive vLLM LoRA); dLoRA merges and unmerges adapters and co-migrates requests (up to 57.9x throughput over vLLM, "up to" maximums).
- **MuxServe** | ICML 2024 (peer) - [PMLR](https://proceedings.mlr.press/v235/duan24a.html)
  - Unified resource manager for co-located LLMs; details of the shared KV cache [unverified, abstract only from the memory reader; the serving reader confirmed a "unified KV/memory manager" in the full text].
- **CachedAttention (AttentionStore)** | ATC 2024 (peer) - [USENIX](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost)
  - Keeps per-conversation KV caches in a host memory and storage hierarchy across turns with layer-wise preloading; up to 87% lower TTFT and up to 70% lower cost for multi-turn conversations.
- **InfiniGen** | OSDI 2024 (peer) - [USENIX](https://www.usenix.org/conference/osdi24/presentation/lee)
  - Offloading-based inference that prefetches only important KV entries; up to 3.00x over prior KV management in an offloading system.
- **ServerlessLLM** | OSDI 2024 (peer) - [USENIX](https://www.usenix.org/conference/osdi24/presentation/fu)
  - Frees memory by unloading whole models with fast multi-tier checkpoint loading and live migration; "10 - 200X" lower latency than serverless baselines.
- **Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning** | OSDI 2026 (peer; confirmed in the OSDI '26 program) | arXiv 2505.04021 (v3 10 Jun 2026) | [kvcached code, Apache-2.0](https://github.com/ovg-project/kvcached) | read: full text (first 100k of 131k characters) and README - [arXiv](https://arxiv.org/abs/2505.04021); [OSDI '26 program](https://www.usenix.org/conference/osdi26/technical-sessions)
  - Mechanism: each engine reserves a large virtual range; "Physical pages are created only on demand and mapped to virtual addresses lazily", "at a 2 MB granularity"; weights and KV share one pool; a global scheduler moves per-model limits ("when an evicted model is reactivated, it shrinks the limits of other models on the same GPU"); idle models are evicted to CPU DRAM; long decodes are preempted "only when memory is severely constrained"; placement uses a KV Pressure Ratio.
  - Claims: on 32 H100s with 58 LLMs, "up to 3.3x higher [TTFT] SLO attainment and 2x higher [TPOT] SLO attainment"; eight models on two GPUs at 99% attainment, "up to 2.3x and 3.5x more requests than MuxServe++ and static partitioning"; elasticity overhead "3 ms (4 %) TTFT and 4 ms (13 %) TPOT" for two Llama-3.2-3B models on an A100-40G; model activation 0.7-1.5 s.
  - Evaluation: Hyperbolic (24 models, 4 months) and Arena-Chat (84 models, 11 days) traces; per-model P95 TTFT and TPOT SLOs from dedicated-GPU runs times a scale; baselines static partition, MuxServe++, QLM, ServerlessLLM; models 1B-70B.
  - The README says kvcached lets LLM inference "coexist with other GPU workloads such as training jobs, fine-tuning, or vision models", but only the LLM's KV and weights are elastic; it does not say whether a limit can change under a running engine.
- **Aegaeon** | SOSP 2025 (peer) | author PDF not opened - [PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
  - Per-token model switching on pooled GPUs; up to 7 models per GPU and GPUs cut from 1,192 to 213 in production [search snippets and press only, unverified].
- **QLM** | SoCC 2024 [from DOI prefix only] | arXiv 2407.00047 | [code](https://www.github.com/QLM-project/QLM) - [arXiv](https://arxiv.org/abs/2407.00047)
  - One global queue with request pulling, eviction, load balancing and model swapping; 40-90% better SLO attainment.
- **GMLake** | ASPLOS 2024 (peer) | arXiv 2401.08156 - [arXiv](https://arxiv.org/abs/2401.08156)
  - VMM "virtual memory stitching" for training; on eight LLMs on A100 80 GB, 9.2 GB less memory on average (up to 25 GB) and 15% less fragmentation.
- **eLLM** | DAC 2026 (peer, design venue) | arXiv 2506.15155 - [arXiv](https://arxiv.org/abs/2506.15155)
  - Ballooning-style inflate and deflate of a unified activation and KV pool over VMM, with CPU memory as buffer; 2.32x decoding throughput.
- **KunServe** | EuroSys 2026 [unverified] | arXiv 2412.18169 - [arXiv](https://arxiv.org/abs/2412.18169)
  - Drops parameter replicas under memory overload to free KV memory instantly; up to 72.2x lower tail TTFT (v2 said 27.3x, so the headline changed across versions).
- **FastServe** (arXiv 2305.05920) and **Andes** (arXiv 2404.16283), both arXiv-only: token-level preemption with proactive KV swap (up to 31.4x throughput over vLLM at the same average latency) and QoE-driven preemption (up to 4.7x average QoE) - [FastServe](https://arxiv.org/abs/2305.05920); [Andes](https://arxiv.org/abs/2404.16283)
- **MSched** (arXiv 2512.24637): proactive working-set migration at context switch for oversubscribed GPU multitasking; claimed 11.05x over demand paging at 150% oversubscription on RTX 5080 and 3080 [search summary, verify] - [arXiv](https://arxiv.org/abs/2512.24637)
- **EMA** (arXiv 2609.27040, Sep 2026): GPUs in one server lend and reclaim memory; up to 52% higher per-user throughput - [arXiv](https://arxiv.org/abs/2609.27040)
- A position paper, "Towards Efficient and Practical GPU Multitasking in the Era of LLM" (arXiv 2508.08448), argues for an OS-like GPU resource layer, with no numbers - [arXiv](https://arxiv.org/abs/2508.08448)

#### Runtime memory limits below the application
- NVIDIA MPS (legacy) memory limits apply only to clients created after the change: "Existing clients are not affected" - [MPS doc](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- NVIDIA MPS v3 memory partitioning (requires CUDA 13.4; the guide does not name a minimum driver for this feature) fractionalizes memory "across cgroups and containers, so that a single tenant cannot monopolize the GPU and schedulers such as Kubernetes can pack workloads by memory", with soft and hard limits and reactive eviction of the oldest over-soft cgroup - [MPS doc, Sec. 1.1.10](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)
- TGS uses CUDA unified memory to let opportunistic containers oversubscribe GPU memory - [TGS](https://www.usenix.org/conference/nsdi23/presentation/wu); gShare hot-plugs vGPUs and swaps models to oversell idle vGPU memory - [gShare](https://cis.temple.edu/~jiewu/research/publications/Publication_files/3779212.3790168.pdf)
- HAMi caps memory per container through an environment variable read by an `LD_PRELOAD` library; no live-resize mechanism was confirmed - [HAMi](https://github.com/Project-HAMi/HAMi)
- Kubernetes in-place resize covers CPU and memory, not GPU extended resources - [Kubernetes docs](https://kubernetes.io/docs/tasks/configure-pod-container/resize-container-resources/)

#### The project's own memory numbers
- StreamSplat weights are 2.42 GB and one process uses about 4,094 MiB (G5); in one batching server, memory grows "about 1,000-1,200 MiB per sample in the batch (3,890 MiB at b = 1, 11,220 MiB at b = 8)" (G9); AnySplat used 4.6-9.4 GiB per user depending on frames per update (G1) - [project LOG.md](../../../LOG.md)

### Inferences
- The analogy to a KV cache holds only if a user's persistent state grows with stream length.
  For StreamSplat, the logged memory is weights plus per-update activations, which are transient; persistent per-user state has not been measured, so the premise is unchecked.
  AnySplat's memory grows with frames per update (G1), but whether it grows with accumulated stream length was not measured.
- Memory does not bind in the measured regime.
  By memory, process-per-user StreamSplat fits about 48 GB / 4 GB = 11-12 users, while compute gives about 4.7-5 updates/s in total (G5-G7), so at one update per user per second compute binds at 4-5 users.
  By this arithmetic, memory would bind first only below roughly 0.4 updates per user per second (4.7 / 11), for example slow scenes or idle users; this is an inference, not a measurement.
- In a single shared server, the weights are held once, so per-user cost falls to about 1.1 GB of transient activations per concurrent update (G9), which pushes the memory limit further out.
  Elastic memory therefore matters mainly for process-per-user isolation (containers), which is exactly the arrangement where the installed driver cannot grow a running client's limit.

### Gaps
- No paper found grows an externally enforced GPU memory limit on a running, unmodified container in response to its growth; Ghost claims something close but was not readable.
- Whether kvcached's allocator or limit API can back non-LLM tensors (a vision model's per-user state) was not checked.
- The behaviour of MPS v3 memory partitioning was read from documentation only; it could not be tried, because the local driver is 595.91.07 with CUDA 13.2 and setting limits is privileged.

## 5. Which papers address real-time video / vision inference serving with per-stream SLOs?

### Takeaway
Video analytics systems gain capacity on a saturated GPU mainly by lowering per-stream work (frame rate, resolution, model variant, offloading frequency), by dropping or admitting requests, and by batching small models.
ARISE (MobiSys 2024) is the closest analogue to live reconstruction: each AR client keeps using the last server result between offloads, so the per-client update rate is the knob, and capacity is measured as clients served within a per-client accuracy SLA.

### Cited Findings

#### Video and vision paper records
- **Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis** | SOSP 2019 (peer) | DOI 10.1145/3341301.3359658 | read: full text - [PDF](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf)
  - "Squishy bin packing" of model sessions with latency-profiled batch sizes, SLO splitting across pipeline stages, prefix batching, early drop, and CPU-GPU overlap.
  - Claim: "when required to stay within latency constraints at least 99% of the time, Nexus can process requests at rates 1.8-12.7x higher than state of the art systems can"; on 100 GPUs it "violates latency SLOs on 0.27% of requests".
  - Evaluation: 20 Twitch game streams (50 ms SLO, LeNet plus ResNet-50) and 20 traffic camera streams (400 ms SLO); 16 GTX 1080 Ti; baselines Clipper and TF Serving; throughput is the maximum request rate with 99% of requests within SLO.
- **Chameleon: Scalable Adaptation of Video Analytics** | SIGCOMM 2018 (peer) | DOI 10.1145/3230543.3230574 | read: abstract - [MSR](https://www.microsoft.com/en-us/research/?p=486414)
  - Periodically re-picks resolution, frame rate and model per camera, sharing profiling cost across time and correlated cameras; "20-50% higher accuracy with the same amount of resources" on five traffic cameras; constraint is a resource budget, not a latency SLO.
- **Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers** | NSDI 2022 (peer) | read: full text - [USENIX](https://www.usenix.org/conference/nsdi22/presentation/bhardwaj)
  - A "thief" scheduler splits a GPU between inference and retraining per stream and picks inference configurations; "29% higher" accuracy than a baseline scheduler, which "requires 4x more GPU resources to achieve the same accuracy" (10 streams, simulator); V100; the real-time criterion is to "always keep up with the video frame rate"; capacity is "the number of concurrently processed video streams".
- **RECL** | NSDI 2023 (peer; not 2024) | read: full text - [USENIX](https://www.usenix.org/conference/nsdi23/presentation/khani)
  - Shared model zoo plus retraining; up to 9.0% mAP and 7.4% accuracy gains; motivating fact: one "V100 GPU can only support two video streams running ... YOLOv5-L ... at 30 FPS".
- **Jellyfish: Timely Inference Serving for Dynamic Edge Networks** | RTSS 2022 (peer) | read: abstract and slides - [VU record](https://research.vu.nl/en/publications/jellyfish-timely-inference-serving-for-dynamic-edge-networks/)
  - Jointly adapts client input size and server DNN variant and batch size across clients; "can meet latency SLOs at around the 99th percentile while maintaining high accuracy" over real WiFi and LTE traces; the SLO is end-to-end including the network; GPU and client counts [unverified].
- **Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU** | SenSys 2023 (peer, per dblp) | arXiv 2307.04339 | read: full text - [arXiv](https://arxiv.org/abs/2307.04339)
  - Source-rewritten "elastic" kernels let best-effort work fill gaps around critical tasks; "increase system throughput by 92% while only incurring less than 10% latency overhead for critical tasks" (abstract), but the body reports 21-28% overhead for closed-loop critical tasks; RTX 2060 and Jetson AGX Xavier; no MPS or MIG comparison.
- **RAVAS: Interference-Aware Model Selection and Resource Allocation for Live Edge Video Analytics** | SEC 2023 (peer) | read: full text - [PDF](https://lass.cs.umass.edu/papers/pdf/sec2023-ravas.pdf)
  - Per-stream Q-learning model selection plus a global allocator that downgrades high-accuracy streams until the total GPU share fits, then sets per-model frame rate and spatial GPU share.
  - Claim: "up to 99% and 40% reductions in latency and overall GPU usage"; at a 200 ms deadline, baselines miss 82.8%-100% of frames against 0.3% for RAVAS; one edge GPU (K80, P4, T4 or V100) with 1-6 streams at 10 fps on VIRAT.
- **ARISE: High-Capacity AR Offloading Inference Serving via Proactive Scheduling** | MobiSys 2024 (peer) | read: full text - [PDF](https://qiangxu1996.github.io/assets/pdf/Kong2024_ARISEOffloadingAR.pdf)
  - Per-client near-minimal offloading frequency from an accuracy estimator, then aligned request timing across clients for larger batches.
  - Claim: "ARISE supports 1.7x-6.9x more clients compared to various baselines while keeping the per-client accuracy within the client-specified accuracy SLAs"; depth estimation up to 5.2x and 6.9x over Clipper-like and Chameleon-like baselines, object detection 1.9x and 2.0x; NVIDIA A40; emulated clients plus a 10-phone testbed.
- **DeepRT: A Soft Real Time Scheduler for Computer Vision Applications on the Edge** | SEC 2021 (peer) | arXiv 2105.01803 | read: abstract - [arXiv](https://arxiv.org/abs/2105.01803)
  - Batching under latency guarantees plus an Admission Control Module for new video-inference requests and an Adaptation Module for overruns; no numbers in the abstract.
- **DART** | RTSS 2019 (peer) | DOI 10.1109/RTSS46320.2019.00042 - [PDF](https://intra.ece.ucr.edu/~hyoseung/pdf/rtss19-dart.pdf)
  - Pipelined data-parallel CPU/GPU scheduling for real-time and best-effort DNN tasks; evaluation [unverified].
- **Llama** (SoCC 2021, arXiv 2102.01887): per-operation latency targets and serverless hardware selection; on average 7.8x lower latency and 16x lower cost - [arXiv](https://arxiv.org/abs/2102.01887)
- **Gemel** (NSDI 2023): merges layers across streams' models to cut GPU memory by up to 60.7% [search snippet] - [USENIX](https://usenix.org/conference/nsdi23/presentation/padmanabhan)
- **DARIS** (arXiv 2504.08795; one reader found it listed as DAC 2025, another as arXiv-only): oversubscribed MPS-plus-streams scheduler, +15% throughput over batching, all high-priority deadlines met - [arXiv](https://arxiv.org/abs/2504.08795)
- **ASTRA** (arXiv 2609.07020, Sep 2026): adapts camera resolution and frame rate for up to 8 emulated cameras, reliability above 90% - [arXiv](https://arxiv.org/abs/2609.07020)
- Cite-only, thinly verified: VideoStorm (NSDI 2017), VideoEdge (SEC 2018), Distream (SenSys 2020), OneAdapt (SoCC 2023), Scrooge (SoCC 2021, content unverified).

### Inferences
- The three ways this literature defines "streams served" map onto the project directly: maximum request rate at 99% within SLO (Nexus), number of streams that keep up with their frame rate (Ekya), and number of clients within a per-client SLA (ARISE).
- The ARISE design suggests the reconstruction analogue of an accuracy SLA: a staleness SLA, the age of the newest reconstruction a user sees, which depends on the per-user update rate and queueing delay.

### Gaps
- Jellyfish's hardware and client counts, DART's and DeepRT's evaluations, and Scrooge's content were not verified.
- No paper found serves a heavy feed-forward 3D reconstruction per stream; this was out of the search scope by design.

## 6. What is still open or under-studied that a small course project could measure on one GPU?

### Takeaway
The open space is not "which sharing mechanism serves more users" (the literature and G2-G9 both say none does on a saturated GPU).
It is what happens to each user when demand exceeds capacity: per-user tail latency and fairness among equal tenants, overload control for updates that supersede each other, and the regime where memory rather than compute binds.
All three can be measured on the RTX 6000 Ada (and the kind cluster) in a few weeks with the existing StreamSplat harness.

### Cited Findings
- Nearly all kernel-level sharing papers study one latency-critical tenant plus best-effort tenants (Tally, REEF, Orion, LithOS, XSched, GPREEMPT, TGS) - [Tally](https://arxiv.org/abs/2410.07381); [REEF](https://www.usenix.org/conference/osdi22/presentation/han); [LithOS](https://arxiv.org/abs/2504.15465)
- Every serving paper read treats requests as independent; dropping is judged per request (Clockwork, Shepherd, PARD, Nexus) - [Clockwork](https://arxiv.org/abs/2006.02464); [Shepherd](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong); [PARD](https://arxiv.org/abs/2602.08747)
- Admission control appears at kernel level (Mosaic), request level (Clockwork) and edge video (DeepRT), but no paper read measures admission in units of users at a known per-user rate - [Mosaic](https://arxiv.org/abs/2610.07504); [DeepRT](https://arxiv.org/abs/2105.01803)
- The device plugin's MPS mode enforces equal 1/N memory and compute caps, the documentation promises different isolation for time-slicing, MPS replicas and HAMi, and none of these documents publishes a measurement of that isolation - [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin); [HAMi](https://github.com/Project-HAMi/HAMi)
- Gilman & Walls list what is missing from the stock mechanisms: "fine-grained preemption mechanisms, robust task prioritization options, and contention-aware thread block placement policies" - [arXiv 2110.00459](https://arxiv.org/abs/2110.00459)

### Candidate project questions (suggestions only, not decisions)
These are suggestions drawn from the papers above; each would need its own pre-registration in plan mode before any run.

**Suggestion A: overload control for live-update tenants (admission and supersede-dropping).**
- Question: when N StreamSplat users together ask for more than the GPU's roughly 4.7-5 updates/s, which policy keeps the most users within a staleness or latency SLO?
- Arms: FCFS with no dropping (null); Clockwork-style rejection of updates that would miss their deadline even if started now; "latest only" per user (a new frame from a user replaces that user's queued, not-yet-started update); user-level admission control that refuses user N+1 once the declared per-user rate times N exceeds measured capacity.
- Metrics: goodput (updates delivered within SLO per second, late and dropped counted as failures), per-user p99 update latency and p99 staleness, SLO attainment, Jain fairness; offered load swept from below to well above saturation, with Poisson or Gamma arrivals at two CVs.
- Literature anchors: Clockwork, Shepherd, PARD, Nexus early drop, AlpaServe drop rule, ARISE.
- Why it might be interesting: superseding updates are not modelled by any serving paper read here, and under overload FCFS is expected to collapse while the others should not, so the result has a clear refutable shape.
- Virtualization angle: admission control and overcommit policy for a shared device.

**Suggestion B: isolation and fairness of commodity sharing modes for interactive tenants on a non-MIG GPU.**
- Question: with the same total throughput, how do the vendor and Kubernetes sharing modes distribute latency across tenants, and can any of them protect one interactive user from background reconstruction load?
- Arms: default time-slicing; default MPS (already measured for capacity); MPS with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` = 100/N; MPS static SM partitions (`-S`, flag present in driver 595); MPS client priority 0 versus 1; in kind, device-plugin time-slicing versus device-plugin MPS `replicas: N`, and optionally HAMi `gpucores`.
- Scenarios: N equal users at a fixed rate; and one interactive user plus background "bulk" users running back to back (the Tally and REEF setting).
- Metrics: per-user p50 and p99 update latency, Jain fairness, interactive user's p99 inflation over solo, total updates/s (expected flat, which is the control).
- Literature anchors: Gilman & Walls (time-slicing predictable, MPS contention, priority weak), Tally (+345% p99 under MPS), NVIDIA's own statement that client priority is only a hint, and the 2 ms time slice against a 190 ms update.
- Why it might be interesting: the literature predicts throughput will not move, so the finding is about distribution and isolation, which is what the student says matters (per-user latency and resource allocation).

**Suggestion C: when does memory, not compute, bind, and what does reclaiming idle users' memory buy?**
- Step 0 (a measurement that decides whether to continue): log persistent versus transient GPU memory per user over long streams for StreamSplat and AnySplat (`torch.cuda.memory_allocated` and `max_memory_allocated` per update, n >= 3 streams each), and the split between weights, per-user state and activations.
- Question, only if step 0 shows memory can bind: at low per-user update rates (below roughly 0.4 updates/s for process-per-user StreamSplat, by the arithmetic in Q4), how many more users does offloading idle users' state or process to host memory admit, and at what wake-up latency?
- Arms: static per-process limits (MPS pinned memory caps, which cannot grow a running client); one shared server holding weights once; idle-user eviction to host memory in the style of Prism and gShare; optionally kvcached-style on-demand mapping if it can back non-LLM tensors.
- Metrics: users admitted at a fixed SLO, wake-up latency p99, total updates/s.
- Literature anchors: Prism/kvcached, vLLM swap versus recompute, PipeSwitch and Torpor model swapping, gShare vGPU hot-plugging, NVIDIA MPS v3 soft/hard memory limits.
- Caveat: the MPS v3 cgroup memory partitioning that implements "borrow then be evicted" natively needs CUDA 13.4 (the local driver ships CUDA 13.2) and root to set limits, none of which the local machine has, and nano4 has no cgroup delegation (project memory note, 2026-11-05), so this suggestion is the weakest on feasibility.

### Inferences
- Suggestions A and B reuse the existing harness (`bench/capacity.py`) and change only the scheduling policy or sharing mode, so their first number could arrive within days; C depends on a step-0 measurement that may end it.
- A and B produce results whose direction the literature lets one predict in advance, which makes them easy to pre-register with a refutation criterion; the novelty lies in the workload (one heavy model, equal interactive tenants, superseding updates), not in a new mechanism.

### Gaps
- No search was made for course-scale or workshop papers on N-identical-tenant fairness under MPS; such a paper may exist outside the top venues searched.
- The feasibility of running Orion, Tally or LithOS on the RTX 6000 Ada was not tested; only Orion's README claims support for comparable GPUs and CUDA 12.6.
- Whether the course accepts scheduling and admission control as "virtualization" was not assessed here.
