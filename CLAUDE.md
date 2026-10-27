# virtualization_final_project

課程期末專題：把持續更新的 3D Gaussian 場景串流到 WebGPU 瀏覽器。
計畫在 `PLAN.md`，核實過的論文筆記在 `docs/reading_notes.md`。
這個 repo 連結了 `.claude/rules/research-discipline.md`，交付成果是量測出來的數字。

## 規則

- 事先登記（pre-registration）一律先進入 plan mode 呈現給使用者，核准後才開始任何 run；修改已核准的登記也一樣。
- 使用者最重視的是理解每一步：一次只加一個新元件，每一步說明它做什麼、量出什麼。
- 前兩個里程碑只做 3DGS 串流，不跑 feed-forward 模型；模型在串流管線建好、理解之後才接上（2026-10-27 決定）。
