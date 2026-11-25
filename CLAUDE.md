# virtualization_final_project

課程期末專題：多使用者 feed-forward 3DGS 串流系統。
多位使用者各自上傳即時影片；一張效能有限的 GPU 用同一個 feed-forward 3DGS 模型（StreamSplat）替每位使用者重建場景，再串流回各自的瀏覽器。
就我們所查（2023-2026 年的系統會議與 arXiv，查到 2026-11），還沒有發表過的系統做這件事，這是本專題的 novelty。
本專題提出一個框架，研究在這個條件下有哪些方法可以：
1. 服務更多使用者（capacity：每人每秒 1 個 keyframe，最差使用者 p95 ≤ 1 s）；
2. 降低每位使用者的延遲（p95）；
3. 保持 3DGS 品質（DAVIS 2017 val 上作者的 PSNR）。

目前（RTX 6000 Ada，LOG G15、G16）：GPU 共用機制（time-slicing、MPS、batching）沒有多服務一人；減少每次 update 的計算量讓 capacity 從 4 人到 10 人再到 14 人，keyframe PSNR 代價 0.19 dB。
瀏覽器端串流是下一步。

## 東西在哪

- 報告 `docs/reports/progress_2026-11/`；實驗紀錄 `LOG.md`；報告引用的數字 `results/`。
- 量測框架 `bench/`；每個實驗的 runner `experiments/<study>/run_G<n>_<slug>.sh`；重現 `reproduce.sh`。

## 規則

- 交付成果是量測出來的數字；這個 repo 在本機連結了 `.claude/rules/research-discipline.md`。
- 事先登記一律先進入 plan mode，核准後才開始任何 run；修改已核准的登記也一樣。
- 每個實驗一個 runner、一個 LOG entry；LOG.md 只往後加，新 entry 放在 refuted-claims index 上方。
- 使用者最重視的是理解每一步：一次只加一個新元件，每一步說明它做什麼、量出什麼。
