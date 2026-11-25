# Multi-user live feed-forward 3DGS on one GPU

Many users stream live video to one server.
One GPU runs the same feed-forward 3D Gaussian Splatting (3DGS) model, [StreamSplat](https://github.com/DSL-Lab/StreamSplat), to reconstruct each user's scene.
To our knowledge, no published system serves many users of a feed-forward 3DGS model from one GPU.
This project builds a framework for that setting and asks which methods serve more users, lower each user's latency and keep the reconstruction quality.
The progress report is [docs/reports/progress_2026-11/main.pdf](docs/reports/progress_2026-11/main.pdf) (5 pages).

## Findings so far

One NVIDIA RTX 6000 Ada; every user sends one keyframe per second, and every user's p95 latency must stay within 1 s.

| Goal | Result |
|---|---|
| More users | Time-slicing and MPS added no users to the released code. Less work per update raised capacity from 4 to 10 to 14 users. |
| Lower latency | At capacity, the worst user's p95 latency fell from 0.21 s to 0.07-0.09 s. |
| Keep quality | The two bf16 changes cost 0.19 dB of keyframe PSNR on DAVIS 2017 val. |

Memory is not the limit: 14 users use 4,884 MiB, about 10% of the card.

## Read first

1. The report PDF: the question, the setup and the four results in 5 pages.
2. `results/tables.md`: every table of the report, computed from the files in `results/`.
3. `bench/capacity.py`: the harness that runs N users and checks the latency SLO.

## Repository map

| Path | What it holds |
|---|---|
| `bench/` | The measurement harness. `capacity.py` runs N users; `sweep_lib.sh` holds the functions the runners share. |
| `experiments/<study>/run_G<n>_<slug>.sh` | One runner per experiment. Its header holds the predictions, written before the run. |
| `probes/` | Equivalence checks that the runners call before they time anything. |
| `results/` | The numbers behind the report, copied from the runs by `collect.py`. |
| `LOG.md` | The experiment log: question, command, number and what each entry overturns. Read an entry when you need the evidence for a number. |
| `docker/` | The StreamSplat image and the TeX image. |
| `docs/reports/progress_2026-11/` | The report's LaTeX source and figures. |
| `reproduce.sh` | Fetch, build, check and rerun everything below. |

Each number in the report source carries a `% log: G<n>` tag that names its log entry.
Earlier notes and plans are in the git history at commit `0784830`.

## Reproduce

You need Linux, an NVIDIA GPU, Docker with the NVIDIA Container Toolkit, git, curl, unzip, Python 3 and about 35 GB of disk.

```bash
./reproduce.sh setup        # fetch StreamSplat @ 5d686e9, the weights, the clip and DAVIS 2017; verify every sha256
./reproduce.sh build        # build the StreamSplat image (about 20 minutes) and the TeX image
./reproduce.sh smoke        # about 10 GPU minutes: PASS/FAIL against the published throughput and quality
./reproduce.sh experiments  # rerun every StreamSplat experiment, G4 to G16 (about 6 GPU hours)
./reproduce.sh collect      # copy the new numbers into results/; `git diff results/` shows what moved
./reproduce.sh report       # redraw the figures and rebuild the PDF from results/
```

- Verified from a fresh clone of this repository on 2026-11-25 (`LOG.md` G18): every download matched, both images built, `smoke` passed (throughput 3.0% and 2.9% below G16, quality windows bitwise equal to G8), and `report` rebuilt the PDF byte for byte.
- `report` uses the images from `build` but no GPU, and it rebuilds the committed PDF byte for byte.
- `experiments` needs a GPU that nothing else uses: the runners redo any run that another process touched.
- The numbers come from an RTX 6000 Ada. On another GPU, a gate registered for that GPU (G9, G15, G16) may stop a runner; `LOG.md` gives each gate.

The AnySplat runs (G1 to G3, the first two rows of the sharing table) are not scripted.
They need AnySplat at commit `5f5e208` in `third_party/AnySplat`, a Python 3.10 environment with torch 2.2.0+cu121 at `third_party/envs/anysplat` (with its `uv pip freeze` in `freeze.txt`), and the cached Hugging Face weights `lhjiang/anysplat@d2e8c34` and `facebook/VGGT-1B@860abec`.

## Third-party code, models and data

`./reproduce.sh setup` downloads these; none are in this repository.

| Item | License |
|---|---|
| [StreamSplat](https://github.com/DSL-Lab/StreamSplat) code and checkpoint | none published, so we do not redistribute it |
| [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) weights | CC BY-NC 4.0 |
| [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 weights | Apache 2.0 |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [torchvision](https://github.com/pytorch/vision) VGG16 weights | BSD-2-Clause, BSD-3-Clause |
| Test clip from [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/), via AnySplat's examples | CC BY-NC-SA 3.0 |
| [DAVIS 2017](https://davischallenge.org/davis2017/code.html) trainval 480p | CC BY-NC 4.0 |

## License

The code is under the MIT License ([LICENSE](LICENSE)).
