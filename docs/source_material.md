# Source material (verbatim, as provided on 2026-10-27)

This is the user's input material, kept unedited for provenance.
It was produced by a ChatGPT research session (links carry `utm_source=chatgpt.com`), so titles, venues and arXiv IDs must be checked against primary sources before they are cited.
The verified version lives in `docs/reading_notes.md` and `PLAN.md`.

---

對！✅ 你現在描述的就是一個 Live Feed-forward 3DGS Streaming System，而且你提到的 Network Optimization、CPU–GPU Collaboration、WebGPU Rendering 都可以放進同一個研究架構。

不過我查完 2025–2026 年的相關研究後，發現一篇和你想法非常接近、必須優先閱讀的 2026 年論文：

Streaming Real-Time Rendered Scenes as 3D Gaussians（2026）。

它已經實作 Server 持續更新 3DGS、傳送增量更新、Client 自己重建並渲染場景的系統。所以不能直接把「持續傳 3DGS 給 Client」當成全新的研究貢獻。

但你的 Feed-forward Reconstruction + WebGPU Browser + Network/GPU Joint Scheduling 仍然有值得探索的空間。

### 1. 你想做的完整系統架構

你的研究系統：從持續輸入的影片建構 3D 場景，將場景變更串流至瀏覽器，使用 Client GPU 本地渲染。

理想狀況下：

* Camera 持續輸出 RGB Video。
* Server GPU 上的模型持續更新 3DGS 場景。
* Server 不需要每次重新傳送完整的幾百萬個 Gaussians，而是傳送 Scene Delta。
* Browser 收到更新後，透過 CPU 解碼、維護 Cache，並上傳到 GPU。
* WebGPU 持續渲染，使用者可以自由改變相機視角，不必完全跟著原始影片的相機走。

這裡最重要的設計是：把 Model Inference Rate、Network Update Rate 和 Browser Render Rate 解耦。

例如模型每秒產生 10 次更新，但 Browser 仍能以 60 FPS 渲染目前最新、已經完成套用的場景。

另外，這裡的 Delta 必須有明確的 Gaussian Identity 或對應關係。若模型每次都吐出一套全新且無法對應的 Gaussians，就不能直接使用 ID-based Delta；需要額外的 Matching、Fusion，或改成傳送可獨立解碼的區塊。

### 2. 有沒有人做過？有，而且已經很接近

這是我認為最重要的六篇，按照和你構想的相似程度排序。

| 論文 | 做到了什麼 | 還沒有完全覆蓋的部分 |
| --- | --- | --- |
| Streaming Real-Time Rendered Scenes as 3D Gaussians (2026) | Server 連續更新模型、Delta Streaming、Client Rendering | 使用 Unity / Online Optimization，而非你規劃的 Feed-forward + WebGPU Browser |
| StreamSplat (ICLR 2026) | 持續影片輸入、Feed-forward Dynamic 3DGS | 非主要研究 Browser / Network Delivery |
| S2GS (ICML 2026) | Causal Incremental Gaussian Scene，處理長影片 | 非主要研究遠端互動式顯示 |
| Visionary (2025 Preprint) | Browser WebGPU、逐影格 Gaussian Generation / Update | 未見完整的遠端即時 Scene-Delta 傳輸評估 |
| MoQSplat (MMSP 2026) | QUIC、Spatial Tracks、Progressive Streaming | 著重場景配送，不是完整的持續 Feed-forward 重建系統 |
| WebSplatter (ACM MM 2026) | 高效率、跨裝置的 WebGPU Renderer | 主要是 Rendering，而非 Online Update Protocol |

這個判斷根據公開論文與專案描述；「未覆蓋」指該部分不是明確的主要研究貢獻，不代表作者的程式絕對沒有相關功能。

尤其建議先讀下面三篇原文：

[Streaming Real-Time Rendered Scenes as 3D Gaussians](https://arxiv.org/abs/2604.02851?utm_source=chatgpt.com)

最接近你整體想法，尤其是 Snapshot vs Delta Streaming。

[MoQSplat](https://arxiv.org/abs/2609.18624?utm_source=chatgpt.com)

最適合研究 Network Protocol 和 Gaussian Priority Scheduling。

[Visionary](https://arxiv.org/abs/2512.08478?utm_source=chatgpt.com)

最適合觀察 WebGPU 如何支援動態更新 Gaussian Buffer。

### 3. Network Optimization 完全可以加進去，而且可能是主要貢獻

我覺得你真正值得研究的，不是「怎麼讓 3DGS 能傳到 Browser」，而是：

Network 和 GPU 共同決定哪些 Gaussian Update 應該先被傳送、先進入 GPU，哪些更新可以延後甚至丟棄。

因為即使網路能及時傳送，Browser GPU 也不一定能夠及時處理所有更新。

我會考慮四種方法：

A. View-Dependent Streaming

使用者正在看房間左側，就優先傳左側的 Gaussian，右側資料延後傳。

MoQSplat 已經做了相似的視角相關選擇，所以可以作為 Baseline。

B. Delta Compression + Temporal Coalescing

假設同一顆 Gaussian 在短時間內更新了三次：

Update 1 → Skip，Update 2 → Skip，Update 3 → Transmit

當中間更新沒有被 Client 消費，而且更新語意允許覆蓋時，只傳最新狀態。

但如果是非交換性操作，例如先建立再刪除 Gaussian，就需要保留順序、版本或對操作進行合併。

C. Network-Aware Quality Adaptation

網路好的時候傳高品質 Gaussian，網路差時降低更新頻率、精度，或只更新可見區域。

可以參考 StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video（AAAI 2026），這篇使用可調 Bitrate 的表示方式來處理 Real-Time Free-Viewpoint Video。

D. Joint Network–GPU Scheduling

這是我最看好的創新切入點。

現有 Network Scheduler 往往主要考慮可見度、資料大小、頻寬及傳輸優先順序。

但對 WebGPU Client 來說，還要考慮：

* Gaussian 解碼與上傳 GPU 需要多少時間？
* GPU Buffer 還有多少空間？
* 更新是否會導致下一幀超出 Render Deadline？
* 某個 Chunk 即使已經傳到 Client，會不會根本沒時間顯示？

因此可以設計 Deadline-Aware、GPU-Budget-Aware 的 Update Scheduler。

### 4. 我建議的研究題目

End-to-End Resource-Aware Streaming for Online Feed-Forward 3D Gaussian Splatting in Web Browsers

核心 Hypothesis：

> Jointly scheduling network transmission and GPU updates can reduce scene-update latency and rendering stalls compared with independently optimizing network delivery and GPU rendering.

你可以做三組 Baseline：

| System | Network 策略 | GPU 策略 |
| --- | --- | --- |
| Baseline 1 | FIFO 全量更新 | 立即 Upload |
| Baseline 2 | View-Aware Priority | 立即 Upload |
| Proposed | View + Deadline + GPU Budget | Batching + Budgeted Upload |

而且一定要測 Glass-to-Glass Latency：從一個場景變化進入 Camera，到 Browser 真正顯示該變化的時間。單獨測 FPS 並不足夠。

期末專題可以先這樣落地：

* 使用 StreamSplat 或 S2GS 預先產生一段 Gaussian Update Trace，暫時不用即時執行模型。
* 使用 WebSplatter 或 Visionary 作為 WebGPU Rendering Base，實作動態 Buffer Update。
* 先透過 WebSocket 傳送版本化 Gaussian Chunks，並以 Linux `tc netem` 模擬網路延遲和封包遺失。
* 實作自己的 Joint Scheduler，測量不同網路、GPU Memory Budget 下的 End-to-End Performance。
* 最後再接上真正的 Feed-forward Model，完成 Live Demo。

這也讓你的專題有一個明確的虛擬化切入點：把 Browser GPU Memory 當成容量受限的虛擬場景快取，並在不同更新工作之間分配資源。

不過這仍比較接近 Resource Management；如果老師要求嚴格的 Hypervisor、VM 或 Device Virtualization，則需要另外加入多 Client 或多 Workload 的資源隔離問題。

### 5. 還有一篇值得你特別注意的清華研究

LMG: Efficient Streaming of Layered Mesh–Gaussian 3D Scenes — ACM MMSys 2026

這篇由清華 Cheng-Hsin Hsu 團隊與 Northeastern University 合作，已經做了 End-to-End Scene Streaming，著重有限頻寬下的互動式 3D 場景配送。

[LMG Project / Paper / Code](https://aiins-nthu.github.io/LMG/?utm_source=chatgpt.com)

雖然它的研究對象不是持續輸入的 Feed-forward Reconstruction，但其中 Layering、Streaming 和 Client-Side Resource Management 的設計都值得參考。

### 最後，我會把你的想法收斂成這樣

你的核心目標不是發明另一個 3DGS Reconstruction Model，而是建立 Feed-forward 3D Reconstruction 的即時 Web Delivery Runtime。

這個 Runtime 同時負責三件事：

* Scene Consistency：確保 Browser 正確地套用持續變動的 Gaussian Scene。
* Network Efficiency：只傳送真正有價值的更新。
* GPU Efficiency：確保 Client 能在有限 GPU Memory、Upload Bandwidth 和 Render Budget 下流暢顯示。

其中我最推薦選 Joint Network–GPU Scheduling 當你期末專題真正要研究、設計、實作與評估的主題。

這樣即使 StreamSplat、Visionary、WebSplatter、MoQSplat 都已經存在，你仍然有一個清楚的、需要實驗驗證的系統研究問題。
