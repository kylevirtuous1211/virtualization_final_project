# 文獻早已預言零結果，下一步該量隔離與准入

在 OSDI、SOSP、EuroSys、NSDI、ATC、ASPLOS、MLSys、MobiSys、MobiCom、SIGCOMM、MM 與 ISCA/MICRO/HPCA 裡，這次整理沒有找到任何一篇論文讓多位使用者共用一張 GPU 做 3DGS 或 feed-forward 重建、並量每位使用者的延遲。
最接近的是三類工作：單機、單場景的 3DGS 加速器（其中只有 MICRO 2026 的 SCARF 針對 feed-forward 重建，而且只服務一路串流）、以頻寬與硬體編碼器為共享資源的 3DGS 與體積影片串流（MuV2、L3GS、Vega），以及雲端遊戲、多人 VR、edge SLAM 這些類比。
專題 G2 到 G9 的核心結果是 time-slicing、預設 MPS 與跨使用者 batching 都沒有多服務一人，RTX 6000 Ada 在約 5 個 StreamSplat updates/s 飽和，而這正是 Clockwork、AlpaServe、Gilman & Walls 與 NVIDIA 自己的 MPS 文件寫好的答案。
所以這批結果是新工作負載上的正確複製，不是失敗，但單靠它不會讓系統領域的讀者驚訝。
真正開放的有兩塊：負載超過容量時每位使用者遭遇什麼（尾延遲、隔離、公平、准入、場景新鮮度），以及記憶體先於運算綁住的情境，也就是狀態隨串流長度成長的重建模型。
依「幾週內做出課程可發表的有趣結果」排序，第一名是時間切片對空間切分的隔離研究（含 Kubernetes device plugin 的共享模式），第二名是使用者層准入加「只留最新」的過載控制，第三名是成長狀態模型的記憶體超額配置，它的天花板最高、風險也最大。
每個方向都有一個一天內做得完、結果可能直接殺掉它的第 0 步，而任何 MIG 對照臂都要先確認拿得到，因為 RTX 6000 Ada 不支援 MIG，專題自己的 nano4 探測也記錄 MIG 未啟用。

## 交集是空的：3DGS 的系統論文只有單機加速器與網路串流

架構會議把 3DGS 當成單機、單場景的算繪核心來加速。
GSCore（ASPLOS 2024）量到排序加光柵化在 Jetson Xavier NX 與 RTX 3090 上都佔算繪時間約 80%，之後 GCC、Neo、DeGS 等加速器都以它為比較基準，Lumina（ISCA 2025）與 GBU（HPCA 2025）則針對行動與 edge GPU ([GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf); [Neo](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf); [Lumina](https://chenc10.github.io/assets/pdf/2025_isca_lumina.pdf); [GBU](https://arxiv.org/html/2503.23625))。
2025 到 2026 年的訓練系統處理的是單一工作的記憶體容量，例如 CLM 指出 24 GB 的 RTX 4090 最多放約 2,600 萬顆 Gaussian，GS-Scale 則把 Gaussian 放在主記憶體、只搬可見的子集 ([CLM, ASPLOS 2026](https://arxiv.org/html/2511.04951v1); [GS-Scale, ASPLOS 2026](https://99dhl.github.io/assets/pdf/asplos26_gsscale.pdf))。
**唯一針對 feed-forward 3DGS 重建的系統論文是 SCARF（MICRO 2026）**，它在 Jetson Orin NX 上量到一次 encoder 約等於 50 次光柵化視角的成本，depth search 佔超過 65% 的記憶體流量、L2 命中率平均只有 31.2% ([SCARF](https://arxiv.org/html/2611.02365))。
SCARF 所說的「single hardware instance」指的是支援多個 encoder 模型，不是多位使用者，它也沒有討論 batching ([SCARF](https://arxiv.org/html/2611.02365))。
MLSys 2026 的 SwiftGS 是單 GPU 算繪最佳化 ([SwiftGS](https://proceedings.mlsys.org/paper_files/paper/2026/file/5f96a21345c138da929e99871fda138e-Paper-Conference.pdf))，而 OSDI、SOSP、EuroSys、NSDI、ATC、MobiSys 在網路串流之外找不到 3DGS 論文。
這個「找不到」來自 arXiv comment 欄位與逐會議的關鍵字搜尋，不是逐篇翻議程，所以只是弱證據。
唯一明說多使用者的神經算繪平台 NEPHELE 只在 arXiv，摘要頁沒有排程器的數字 ([NEPHELE](https://arxiv.org/abs/2303.04086))。

網路與多媒體會議有一條連續的 3DGS 與體積影片串流工作，但它們共享的是頻寬與硬體編碼器，不是做重建的 GPU。
MobiCom 有 MuV2（2024）、L3GS 與 Vega（2025），MobiSys 有 NeRFHub（2024）與 NeRFlow（2025），NSDI 有 YuZu（2022），MMSys 有 LTS（2025 最佳論文），ACM MM 有 MD2G-Cast（2026），INFOCOM 有 AirGS（2026）([L3GS](https://arxiv.org/abs/2504.05517); [NeRFHub](https://par.nsf.gov/servlets/purl/10540540); [YuZu](https://www.usenix.org/conference/nsdi22/presentation/zhang-anlan); [MD2G-Cast](https://arxiv.org/abs/2608.10020))。
**MuV2 是形狀最接近的範本。**
它用一台 2080Ti edge server 加一個 WiFi AP，伺服器替每人算繪加編碼的設計在 8 人時 GPU 編碼器用到 95%，直接串 3D 內容的設計 3 人就被頻寬卡住，跨使用者共享 edge 算繪的視角之後撐到 30 人以上 ([MuV2, MobiCom 2024](https://par.nsf.gov/servlets/purl/10633047))。
它的教訓直接對應專題：單一資源飽和後換共享機制不會多服務人，要讓需求不再隨人數成長才行。
主張「伺服器建一次 Gaussian 就能攤提給多位使用者」的 arXiv 2604.02851，所有串流實驗都只有一個客戶端 ([arXiv 2604.02851](https://arxiv.org/abs/2604.02851))。

最接近「多人各自在一張 GPU 上做即時 3D 工作」的，是三條相鄰的文獻。
雲端遊戲的 GPU 虛擬化從 gVirt（ATC 2014，16 ms 固定時間片，7 台 VM）、gScale（ATC 2016，綁住 vGPU 密度的是圖形記憶體位址空間，動態共享後從 3 個擴到 15 個）一路發展到 PilotFish（ATC 2022）([gVirt](https://www.usenix.org/conference/atc14/technical-sessions/presentation/tian); [gScale](https://www.usenix.org/conference/atc16/technical-sessions/presentation/xue))。
PilotFish 量到兩款遊戲共置時 RDR2 從 60 掉到 20 FPS，GPU 使用率卻最多只多 24%，可見干擾在 GPU 忙滿之前就出現 ([PilotFish](https://www.usenix.org/system/files/atc22-zhang-wei.pdf))。
多人 VR 的 Firefly（ATC 2020）與 Coterie（ASPLOS 2020）靠離線預算或跨位置重用畫面撐到 15 人與 4 人，靠的不是排程 ([Firefly](https://www.usenix.org/conference/atc20/presentation/liu-xing); [Coterie](https://doi.org/10.1145/3373376.3378516))。
多人 edge SLAM 的 SwarmMap（NSDI 2022）描述了和專題一樣的失效模式，代理人變多後請求「在 edge 節點的處理管線排隊，造成過度的佇列延遲」，它靠依代理人狀態排優先序把可服務的代理人數加倍 ([SwarmMap](https://www.usenix.org/conference/nsdi22/presentation/xu))。
SLAM-Share（CoNEXT 2022）每個客戶端一個 GPU 行程，完全沒有控制它們怎麼共享 GPU ([SLAM-Share](https://www.cs.ucr.edu/~jiasi/pub/SLAMShare-conext22.pdf))。
通用的 GPU 共享與 DNN serving 文獻（Tally、REEF、Orion、LithOS、XSched、GPREEMPT、TGS、Clockwork、Shepherd）提供了機制與評估方法，但核心層級的排程器幾乎都假設一個延遲敏感租戶加上 best-effort 租戶 ([Tally](https://arxiv.org/abs/2410.07381); [LithOS](https://arxiv.org/abs/2504.15465))。

| 論文 | 會議與年份 | 共享的資源 | 和本專題的距離 |
| --- | --- | --- | --- |
| SCARF | MICRO 2026 | 無，單一串流 | 唯一的 feed-forward 3DGS 系統論文，平台是 edge GPU |
| GSCore、Neo、Lumina、GBU | ASPLOS 2024、ASPLOS 2026、ISCA 2025、HPCA 2025 | 無 | 加速的是算繪，不是重建的 encoder |
| CLM、GS-Scale | ASPLOS 2026 | 無 | 單一訓練工作的記憶體容量 |
| SwiftGS | MLSys 2026 | 無 | 單 GPU 算繪最佳化 |
| MuV2 | MobiCom 2024 | WiFi AP、edge GPU 編碼器 | 多人體積串流，靠共享視角撐到 30 人以上 |
| L3GS、Vega | MobiCom 2025 | 頻寬 | 單人 3DGS 串流 |
| NeRFHub、NeRFlow | MobiSys 2024、MobiSys 2025 | 頻寬 | NeRF 服務與 ABR |
| Firefly、Coterie | ATC 2020、ASPLOS 2020 | 伺服器、AP | 多人 VR，靠預算與重用 |
| gScale、PilotFish | ATC 2016、ATC 2022 | vGPU 記憶體、GPU 閒置週期 | 雲端遊戲共置 |
| SwarmMap | NSDI 2022 | edge 節點 | 多人 SLAM 的佇列延遲 |
| ARISE | MobiSys 2024 | GPU 伺服器 | AR 卸載，容量以每客戶端 SLA 計 |
| NEPHELE | arXiv 2303.04086 | 算繪農場 | 多人看同一個 NeRF，未經同儕審查 |

## G2 到 G9 的零結果，飽和 GPU 的文獻早就寫好了

專題量到，AnySplat 在 k = 8、每人每秒一次更新時，time-slicing 與 MPS 都服務 2 人（G2），k = 2、每 0.5 秒一次時都服務 5 人（G3），StreamSplat 每人每秒一個 keyframe 時都服務 4 人（G6），每秒三個時都只服務 1 人（G7）([LOG.md G2, G3, G6, G7](../../LOG.md))。
把 8 位使用者的更新合成一個 batch，吞吐量反而從 batch 1 的 4.92 降到 4.52 updates/s（G9）([LOG.md G9](../../LOG.md))。
這些結果的方向，文獻都預言過。
NVIDIA 自己說 MPS「在每個應用行程產生的工作不足以填滿 GPU 時」才有用 ([MPS 文件](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf))。
Gilman & Walls 在 RTX 3090 上比較 priority streams、預設 time-slicing 與不設限的 MPS，結論是 MPS 最適合「kernel 用不到 GPU 全部資源」的情況，而 time-slicing 從不讓兩個任務重疊 ([Gilman & Walls, IFIP Performance 2021](https://arxiv.org/abs/2110.00459))。
Clockwork（OSDI 2020）量到同時執行多個推論最多只多 25% 吞吐量，代價是延遲變異大增，並寫明一次跑一個推論的吞吐量「和並行執行相當接近」([Clockwork](https://arxiv.org/abs/2006.02464))。
AlpaServe（OSDI 2023）也寫到，當到達率逼近叢集的峰值服務率、所有模型同樣飽和時，聰明的放置就不再有好處 ([AlpaServe](https://arxiv.org/abs/2302.11665))。
video serving 的 batching 收益是在小模型上量到的，Nexus 遊戲案例裡的 ResNet-50 在 batch 1 只要約 6 ms ([Nexus, SOSP 2019](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf))，而一次 StreamSplat 更新是 0.19 s。

「綁住的是運算而不是記憶體」也在預料之中。
一位 StreamSplat 使用者佔 4,094 MiB，記憶體上限約 12 人，運算上限卻只有 4 到 5 人（G6）([LOG.md G6](../../LOG.md))。
SCARF 在 edge GPU 上同樣發現 feed-forward 3DGS 的成本集中在 DNN encoder，算繪便宜一到兩個數量級 ([SCARF](https://arxiv.org/html/2611.02365))。
換句話說，3DGS 架構文獻大多在最佳化即時重建管線裡便宜的那一段，對伺服器能服務幾人幫助不大，它真正相關的是瀏覽器端的算繪。

有三個結果是文獻沒有直接預言的，而它們剛好指向下一步。
第一，G3 的 time-slicing 在 4 位使用者背靠背時達到 10.06 updates/s，高於假設「time-slicing 無法重疊使用者」算出的 8.2 ([LOG.md G3](../../LOG.md))。
比較可能的解讀是其他行程的 kernel 填進了單一使用者在 CPU 端的空檔，這和 Nexus 發現 CPU 前處理與 GPU 重疊是最大單一因素屬於同一類，但專題沒有測過這個機制 ([Nexus](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf))。
第二，G9 裡 8 個樣本的 batch 花了單一樣本 8.7 倍的時間，`forward_gaussians` 單獨跑更是 9.1 倍，也就是單樣本成本隨 batch 上升，`nvidia-smi` 的使用率則從 72% 升到 96%，原因沒有測，也沒有任何一手來源在資料中心 GPU 上剖析過 feed-forward 3DGS 的 kernel ([LOG.md G9](../../LOG.md))。
第三，**模式之間的差異出現在延遲分佈，而不在容量**。
G3 在 N = 5 的膝點，MPS 把 p95 從 0.318-0.330 s 壓到 0.177-0.180 s，G6 在 N = 6 的過載下，MPS 卻讓 depth 階段的 p50 從 time-slicing 的 0.31 s 變成 0.49-0.50 s ([LOG.md G3, G6](../../LOG.md))。
每個點都只有 n = 2，只能當方向看，但它說明「容量相同」不等於「每位使用者的遭遇相同」。

最後，專題的過載區間是開迴路、沒有准入、佇列無上限的 FIFO，所以 N = 6 時 p95 到 13 s、N = 8 時到 25 s，而且會隨執行時間繼續長 ([LOG.md G2, G6](../../LOG.md))。
這正是 serving 與雲端遊戲文獻當作「要修的失效」的區間。
Clockwork 拒絕即使馬上開始也會遲到的請求，因而在過載下維持接近理想的 goodput ([Clockwork](https://arxiv.org/abs/2006.02464))。
Shepherd 證明不做 preemption 的線上演算法無法和最佳離線解競爭 ([Shepherd, NSDI 2023](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong))。
Stimpack 則指出商業雲端遊戲在附近伺服器滿載時直接讓使用者排隊 ([Stimpack](https://arxiv.org/abs/2412.19446))。

## 開放的是每位使用者的遭遇，以及記憶體先綁住的情境

**沒有論文研究 N 個同等優先、跑同一個重模型的互動租戶。**
核心層級的排程器（REEF、Orion、Tally、LithOS、XSched、GPREEMPT）都假設一個延遲敏感租戶加上 best-effort 租戶，並以前者的 p99 為指標 ([REEF, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/han); [Tally](https://arxiv.org/abs/2410.07381))。
在那個設定下，模式之間的差距很大。
Tally 量到高優先推論的平均 p99 額外開銷在 time-slicing 下是 252.3%、在 MPS 下是 345.0%，Tally 自己只有 7.2% ([Tally, ASPLOS 2025](https://arxiv.org/abs/2410.07381))。
VUDA 的 VR 案例裡，不切分的時間共享在重負載下錯過 52.1% 與 78.1% 的 90 Hz 算繪期限，切分後一個都沒錯過，不過它只在 arXiv ([VUDA](https://arxiv.org/abs/2605.01352))。
一次 0.19 s 的 StreamSplat 更新約是 Gilman & Walls 量到的 2 ms 時間片的 95 倍，所以 time-slicing 會把每次更新切成許多片，而單一行程內的 FIFO 會整段跑完，這預言了吞吐量相同、每位使用者延遲分佈不同 ([Gilman & Walls](https://arxiv.org/abs/2110.00459))。
Kubernetes 層同樣沒人量過。
NVIDIA device plugin 的 time-slicing 明說不做任何隔離，「一個工作負載當掉，全部都會當掉」，MPS 模式給每個 replica 相等的 1/N 記憶體與運算上限但仍標為實驗性，HAMi 則用 `LD_PRELOAD` 攔截 CUDA 呼叫、以軟體時間共享實作 SM 上限，這些文件都沒有發表隔離的量測 ([k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin); [HAMi](https://github.com/Project-HAMi/HAMi))。

**更新會互相取代，這件事沒有 serving 論文建模。**
Clockwork、Shepherd、PARD、Nexus 都把請求當成彼此獨立，逐個請求判斷要不要丟 ([Shepherd](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong); [PARD, EuroSys 2026](https://arxiv.org/abs/2602.08747))。
但一位使用者的新 keyframe 一到，佇列裡還沒開始的舊更新就沒有價值了。
串流文獻裡的公平性也都是品質公平，MuV2 的 Jain fairness index 作用在失真上，沒有看到以每位使用者的場景新鮮度或更新延遲為對象的分配 ([MuV2](https://par.nsf.gov/servlets/purl/10633047))。
ARISE（MobiSys 2024）最接近：每個 AR 客戶端在兩次卸載之間沿用上一次的伺服器結果，所以每客戶端的更新頻率就是旋鈕，容量定義成「在每客戶端 SLA 內的客戶端數」，它因此多服務 1.7 到 6.9 倍客戶端 ([ARISE](https://qiangxu1996.github.io/assets/pdf/Kong2024_ARISEOffloadingAR.pdf))。
把它的準確度 SLA 換成「使用者看到的最新重建有多舊」的新鮮度 SLA，就是本專題的版本。

工作負載本身的特性也還沒人量過。
沒有一手來源在資料中心 GPU 上做過 feed-forward 3DGS 模型的 kernel 層 roofline，也沒有任何 3DGS 階段的「吞吐量對 SM 數」曲線，DeGS 只對加速器報了 PE 擴展 ([DeGS, MICRO 2026](https://arxiv.org/abs/2608.02099))。
3DGS 核心內部的 lane 利用率很低，GBU 量到 18.9%，Lumina 量到執行緒超過 69% 的時間被遮罩，這讓「單一工作在使用率 100% 時仍留下閒置發射槽」成為可以檢驗的假設 ([GBU](https://arxiv.org/html/2503.23625); [Lumina](https://chenc10.github.io/assets/pdf/2025_isca_lumina.pdf))。
不過 G6 與 G9 的結果讓它對 StreamSplat 不太可能成立。

**記憶體先綁住的情境，才是虛擬化的記憶體機制能改變可服務人數的地方。**
LLM serving 用分頁（vLLM，SOSP 2023）與 ballooning（Prism，OSDI 2026）處理每個請求不斷長大的 KV cache ([vLLM](https://arxiv.org/abs/2309.06180); [Prism](https://arxiv.org/abs/2505.04021))。
vLLM 因此比預留式的 Orca 模擬多撐 1.7 到 2.7 倍請求率，Prism 在 8 個模型共用 2 張 GPU 時比靜態切分多服務最多 3.5 倍請求 ([vLLM](https://arxiv.org/abs/2309.06180); [Prism](https://arxiv.org/abs/2505.04021))。
重建這邊的對應物已經存在。
StreamVGGT 每快取一幀約多 0.117 GB，在 A800 上 200 幀時佔 25.3 GB，最後一幀的延遲從 63 ms 長到 733 ms ([StreamVGGT](https://arxiv.org/html/2507.11539))。
STream3R 的 causal 模式在 H200 上 200 幀佔 45.41 GB ([STream3R](https://github.com/NIRVANALAN/STream3R))。
但**沒有任何論文把這類串流重建模型同時服務給多位使用者**，OVGGT、InfiniteVGGT 等快取研究全是單串流、單 GPU ([OVGGT](https://arxiv.org/html/2603.05959); [InfiniteVGGT](https://arxiv.org/html/2601.02281))。
要注意的是，StreamSplat 的 4,094 MiB 是權重加上每次更新的暫時 activation，專題沒有量過它的持久狀態是否隨串流長度成長，所以「KV cache 類比」對目前的模型還只是假設 ([LOG.md G5, G9](../../LOG.md))。

用 StreamVGGT 在 A800 上公開的斜率粗算（這是推論，不是量測），記憶體會在每人幀率低的時候先綁住。
一個 200 幀的工作階段結束時約 25.3 GB，按最壞情況替每人預留，48 GB 只放得下 1 人。
若依當下用量准入、讓使用者處在不同階段，平均每人約 2.1 + 0.117 × 100 ≈ 13.8 GB，可放約 3 人，代價是大家同時長大時會 OOM。
改用 50 幀視窗，峰值 8.2 GB、每幀 219 ms，可放約 5 人 ([StreamVGGT](https://arxiv.org/html/2507.11539))。
運算端在每人每 2 秒 1 幀時，全快取的平均每幀約 0.4 s，每人只用掉約 0.2 的 GPU，可放約 5 人，所以在這個區間裡三種記憶體策略給出 1、3、5 人，差距完全由記憶體管理決定。
StreamVGGT 自己的 Table 8 顯示 50 幀視窗在 200 幀測試上沒有損失準確度（Acc 0.054，全快取 0.058，越低越好），所以研究紀律要求一併報告的「被交換的指標」也量得到 ([StreamVGGT](https://arxiv.org/html/2507.11539))。

虛擬化層要怎麼實作「隨快取長大而放大預算」，在這台機器上有明確的限制。
NVIDIA 的 MPS 文件寫明，伺服器端調高記憶體上限之後「既有的客戶端不受影響」，這和專題 2026-11-04 在 nano4 的探測一致，執行中的 MPS 客戶端在上限調高後長不大 ([MPS 文件](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf))。
r620 驅動與 CUDA 13.4 的 MPS v3 加入了以 cgroup 為單位的軟上限、硬上限與「借用之後被逐出」，但要 root 權限與非 MIG 裝置，而本機是 595.91.07 驅動與 CUDA 13.2 ([MPS 文件](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf))。
所以彈性記憶體在這裡只能做在單一服務行程內（Prism 的 kvcached 那種按需映射），或靠重啟與遷移。

## 七個候選方向，依幾週內做出有趣結果的機會排序

五份研究筆記一共提出 17 個候選題，合併重複之後剩下 7 個方向。
排序標準是幾週內做出「課程可發表、讀者會覺得有趣」的結果的機會。
「有趣」在這裡有兩個條件：結果不能只靠算術就事先寫出來，而且要落在學生在乎的指標上，也就是可服務人數、資源分配與每使用者延遲。
每個方向的第 0 步都是一天內、在本機 GPU 上做得完的量測，殺掉條件是我建議在 plan mode 事先登記的門檻，不是已經跑過的結果。

| 排名 | 方向 | 文獻依據 | 第 0 步（一天內） | 殺掉條件 | 工期 |
| --- | --- | --- | --- | --- | --- |
| 1 | 隔離：時間切片對空間切分，含 Kubernetes 共享模式 | Tally、Gilman & Walls、VUDA、MIG 文件 | 一位使用者在 MPS active thread 25/50/75/100% 下的 updates/s，加上一位 1 Hz 受害者配一位背靠背干擾者，time-slicing 與 MPS 各跑 60 s | 兩種預設模式下受害者 p95 都不超過單獨執行的 1.2 倍 | 約 2 週 |
| 2 | 過載控制：使用者層准入、只留最新、新鮮度 SLO | Clockwork、Shepherd、PARD、ARISE、SwarmMap | 用 G6 的服務時間跑小型離散事件模型，再用 G9 的伺服器實跑一次 N = 8 的只留最新 | 實測每位使用者的 p95 新鮮度與模型差不到 10% | 1 到 2 週 |
| 3 | 成長狀態的使用者：記憶體超額配置 | vLLM、Prism、StreamVGGT、OVGGT | StreamVGGT 單人 200 幀的 VRAM 與每幀延遲，加上 StreamSplat 長串流的持久記憶體 | 在每個仍符合延遲 SLO 的每人幀率下，運算上限都不高於最壞預留的記憶體上限 | 3 到 4 週 |
| 4 | kernel 層剖析：為什麼共享與 batching 都沒用 | GSCore、RoofGS、SCARF | Nsight Compute 剖析 batch 1 與 batch 8 各一次更新 | 主要 kernel 貼在運算屋頂，且 batch 8 的變慢有直接解釋，則收成附錄 | 2 到 3 天 |
| 5 | 降低每使用者工作量：依需要分配更新頻率或解析度 | ARISE、Stimpack、Chameleon | 降解析度後的服務時間，加上幾段 DAVIS 序列的重建過時速度 | 解析度不改成本，且各序列能容忍的過時程度差不到 2 倍 | 2 到 3 週 |
| 6 | 一個重建、多人觀看：送 Gaussian 或送像素 | MuV2、arXiv 2604.02851、NeRFlow | 實測每次更新的位元組數，加上伺服器替一位觀看者算繪並 NVENC 編碼的成本 | 交叉觀看人數落在任何合理的觀眾數之外 | 3 到 4 週 |
| 7 | 異質共置：運算密集的重建加記憶體密集的算繪 | GSCore、Neo、PilotFish | 第 4 名的剖析，加上一次 gsplat 在 1080p 與 4K 的算繪基準 | 兩種租戶在這張卡上都是運算密集 | 2 到 3 週 |

### 第 1 名：隔離研究最可能得到非零的結果

這一項合併了三個建議：商用共享模式的隔離與公平、吵鬧鄰居下的隔離，以及 SM 佔比擴展與空間切分。
問題是在總吞吐量相同時，time-slicing、預設 MPS、MPS 的 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 上限、MPS 靜態 SM 切分（`-S`）、MPS 客戶端優先序，以及 kind 叢集上 device plugin 的 time-slicing 與 MPS `replicas: N`，各自如何把延遲分給每位使用者，又能不能保護一位互動使用者不被背景重建拖累。
它排第一有四個理由：文獻預言差距很大（Tally 的 252.3% 與 345.0%），專題的 G3 與 G6 已經看到模式在分佈上的差異，所有對照臂都只是 `bench/capacity.py` 的設定變化，而且每一臂都對應一個虛擬化概念。
時間片對應 VM 的 CPU 排程，SM 上限對應資源保留，靜態切分與 MIG 對應硬體分割，device plugin 的共享模式則是同一件事在編排層的版本。
第 0 步有兩件事，一是 SM 佔比擴展曲線，這是任何 3DGS 或 feed-forward 重建工作負載都沒有人發表過的數字，二是一位每秒一次的受害者加一位背靠背的干擾者，在 time-slicing 與 MPS 下各跑 60 秒。
若受害者 p95 在兩種模式下都不超過單獨執行（約 0.19 s）的 1.2 倍，預設機制已經足夠隔離，這個方向就沒有東西可修。
擴展曲線則決定題目的形狀。
若 50% 的 SM 給出接近 50% 的吞吐量（G6 與 G9 讓我預期如此），空間切分在吞吐量上贏不了時間切片，還會把膝點以下每次更新的延遲拉長成佔比的倒數倍，題目就收斂成「膝點以下的延遲」對「膝點以上的隔離」的取捨。
若曲線明顯下凹，空間切分還可能提高總吞吐量，那會是意外而且值得另外登記的結果。
MIG 是唯一提供記憶體頻寬 QoS 與錯誤隔離的模式，H200 能切成 7 個 `1g.18gb`，放得下一位 4 GiB 的 StreamSplat 使用者 ([MIG User Guide](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf))。
但 RTX 6000 Ada 不在 MIG 支援清單上，nano4 的探測又記錄 MIG 未啟用，所以 MIG 臂在確認之前只能當選配。

### 第 2 名：過載控制的元素是新的，但結果可能只是算術

這一項合併了過載控制、仿 VGRIS 的准入與配速，以及「把更新頻率當 ABR 階梯」中關於公平與准入門檻的部分，SwarmMap 式「最舊的使用者先服務」則是選配臂。
對照臂是不丟棄的 FIFO（現況，也是 null）、Clockwork 式拒絕即使馬上開始也會遲到的更新、「只留最新」（使用者的新 keyframe 取代他佇列中尚未開始的舊更新），以及在宣告的每人頻率乘上 N 超過量到的容量時拒絕第 N+1 人的使用者層准入。
指標照 serving 文獻的配方，把每秒供給的更新數掃過約 5 updates/s 的飽和點，算 goodput（遲到與被丟的都算失敗）、每位使用者的 p99 延遲與 p99 新鮮度、Jain fairness，到達過程用 Poisson 或兩種 CV 的 Gamma ([AlpaServe](https://arxiv.org/abs/2302.11665); [Clockwork](https://arxiv.org/abs/2006.02464))。
它排第二而不是第一，是因為結果很可能可以事先算出來。
StreamSplat 單獨執行的 p95 只比 p50 高 5%（G5），服務時間這麼穩，FIFO 必然發散，只留最新則必然讓每位使用者大約每 N / 4.7 秒拿到一次更新 ([LOG.md G5](../../LOG.md))。
所以第 0 步就是檢查它是不是只是算術：先用 G6 的服務時間寫一個幾十行的離散事件模型，再在 G9 的單行程伺服器上真的跑一次 N = 8 的只留最新。
若實測每位使用者的 p95 新鮮度和模型差不到 10%，這題就降成第 1 名的一頁附錄。
若差很多，例如行程之間的 time-slicing 讓某些使用者系統性地比較舊，那個落差本身就是發現。
它的虛擬化對應是共享裝置的准入控制與超額配置政策，而「更新會互相取代」與「新鮮度 SLO」都是文獻裡沒有的元素。

### 第 3 名：成長狀態的使用者是唯一能讓人數移動的方向

這一項合併了四個建議：記憶體何時綁住與回收閒置使用者能換到什麼、三種記憶體策略下的可服務人數、記憶體該用幀數還是覆蓋面積來預測，以及滿載時該縮快取、卸載還是拒絕新人，也就是學生 2026-11-04 提出的「讓容器的 GPU 預算隨 KV cache 長大」。
它的天花板最高，因為它是七個方向裡唯一一個虛擬化機制有機會真的改變可服務人數的，前面的粗算顯示三種策略可能差到 1、3、5 人，而且沒有論文把串流重建模型同時服務給多位使用者。
它的地板也最低。
它要接一個新模型，而 StreamVGGT 輸出的是 pointmap，不是 Gaussian。
唯一有程式碼與權重、會長 KV cache 的 3DGS 模型 ReCoSplat 又把快取壓掉了 93%，作者正好就在 RTX 6000 Ada 上測 ([ReCoSplat](https://arxiv.org/html/2603.09968))。
每幀延遲也隨快取長大，運算可能和記憶體一起綁住。
若改用依空間成長的模型（Point3R、SceneBound），使用者回到同一個房間時記憶體就不再長，以串流長度預測記憶體的管理器會高估 ([SceneBound](https://arxiv.org/abs/2611.00756))。
第 0 步是在本機跑一位 StreamVGGT 使用者 200 幀，記錄 VRAM 與每幀延遲對幀數的曲線，和公開的每幀 0.117 GB、3.4 ms 比較，同時用 `torch.cuda.memory_allocated` 量 StreamSplat 長串流每次更新後的持久記憶體，確認目前的模型是不是根本不會長。
若在每個仍符合延遲 SLO 的每人幀率下，運算上限都不高於最壞預留的記憶體上限，記憶體政策就永遠不決定容量，這個方向關閉。
StreamVGGT 的程式碼是 CC BY-NC-SA 4.0、權重是 CC BY-NC 4.0，課程使用沒有問題，但修改後的程式碼若要散布必須沿用 share-alike ([StreamVGGT LICENSE](https://github.com/wzzheng/StreamVGGT/blob/main/LICENSE.txt))。

### 第 4 名：kernel 層剖析一定有結果，但單獨太薄

這一項用 GSCore 與 RoofGS 的分階段 roofline、SCARF 的逐階段拆解，回答「為什麼 MPS 與 batching 都沒用」([GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf); [RoofGS](https://arxiv.org/abs/2608.15785))。
它幾乎一定會得到結果，而且會是資料中心 GPU 上第一份 feed-forward 3DGS 的 kernel 層特性描述，但單獨當期末專題太薄，所以排第四。
我建議把它當成第 1 名與第 3 名的第 0 步一起做。
若主要 kernel 貼近 DRAM 屋頂而不是運算屋頂（SCARF 在 Orin NX 上看到的就是這種情形），連記憶體頻寬一起切的 MIG 就會和 MPS 表現不同，第 7 名的異質共置也才有前提 ([SCARF](https://arxiv.org/html/2611.02365))。
若主要 kernel 貼在運算屋頂，而且 batch 8 花 batch 1 的 9.1 倍時間（多於 8 倍）有直接的解釋，它就收成一頁附錄，同時殺掉第 7 名。

### 第 5 名：降低每人工作量是文獻裡真正加人的路，卻繞回品質

這一項合併了仿 dJay 與 Stimpack 的「把頻率當旋鈕」，以及「輸入變化小就跳過更新」，也借用了把更新頻率當 ABR 階梯的配置想法。
文獻裡在飽和伺服器上真正多服務人的，幾乎都是降低每位使用者的工作量。
ARISE 多服務 1.7 到 6.9 倍客戶端，Stimpack 在同樣資源下服務兩倍使用者，Chameleon 只用 30% 到 50% 的資源就達到同樣的準確度 ([ARISE](https://qiangxu1996.github.io/assets/pdf/Kong2024_ARISEOffloadingAR.pdf); [Stimpack](https://arxiv.org/abs/2412.19446); [Chameleon, SIGCOMM 2018](https://www.microsoft.com/en-us/research/?p=486414))。
它排第五，因為「每位使用者需要多少更新」必須用某種品質或過時程度的訊號來定義，而那正是學生說不在乎的東西。
StreamSplat 的權重又是在固定解析度訓練的，可調的旋鈕可能只剩頻率。
第 0 步是量降解析度後 `forward_gaussians` 的時間，以及在幾段運動快慢不同的 DAVIS 序列上，沿用舊重建時算繪品質隨時間掉得多快。
若解析度不改變成本，且各序列能容忍的過時程度差不到 2 倍，最佳化器就退化成「每人同一頻率」的算術，這題記成負面結果關閉。

### 第 6 名：多人觀看能讓需求和人數脫鉤，但換掉了題目

這一項把設定從「每人重建自己的場景」換成「一個重建、K 人觀看」，比較送 Gaussian 由客戶端算繪，和伺服器替每位觀看者算繪、再用 NVENC 編碼送像素。
它直接檢驗 arXiv 2604.02851 沒有測的攤提主張，也是 MuV2 在點雲上做過、沒有人在 3DGS 重建上做過的取捨，而且它是唯一能讓伺服器需求和人數脫鉤的設定 ([MuV2](https://par.nsf.gov/servlets/purl/10633047); [arXiv 2604.02851](https://arxiv.org/abs/2604.02851))。
它排第六，因為 repo 裡還沒有客戶端算繪路徑與 NVENC 路徑，要三到四週的工程，而且它更像串流設計而不是虛擬化。
第 0 步是實測每次更新的位元組數（目前只有約 2.8 MB fp32 的估計，從未量過），以及伺服器替一位觀看者算繪並編碼一幀的成本，由此算出交叉的觀看人數。
若交叉點落在任何合理的觀眾數之外，結論在跑之前就知道了。
同一個量測也回答「網路何時才是瓶頸」：依估計，GPU 飽和時總出口只有約 117 到 172 Mbps（fp32），但單一使用者每秒 1 到 3 個 keyframe 就要約 22 到 99 Mbps，已經逼近 NeRFlow 所用頻寬資料集的平均 36.87 Mbps，所以若網路會綁住，綁住的是個別使用者的最後一哩 ([NeRFlow, MobiSys 2025](https://par.nsf.gov/servlets/purl/10635564))。

### 第 7 名：異質共置的前提已經被專題自己的數字削弱

這一項把運算密集的 feed-forward 更新，和記憶體頻寬密集的高解析度 3DGS 算繪放在同一張卡上，看總有用工作量是否高於兩者各自單跑的預測，依據是 GSCore 與 Neo 指出前處理與排序受記憶體頻寬限制 ([GSCore](https://yong2355.github.io/assets/pdf/asplos24-gscore.pdf); [Neo](https://jongse-park.github.io/files/paper/2026-asplos-neo.pdf))。
它也包含 PilotFish 式的收割，也就是在空檔塞 best-effort 工作。
它排最後，因為專題已經有兩個不利的證據：G9 裡 72% 的 `nvidia-smi` 使用率並不是可用的閒置算力，G6 裡共享本身就吃掉 9% 到 11% 的吞吐量 ([LOG.md G6, G9](../../LOG.md))。
PilotFish 的收割也只在 GPU 未飽和時有效，在這裡只適用於 1 到 4 人的膝點以下 ([PilotFish](https://www.usenix.org/system/files/atc22-zhang-wei.pdf))。
它還要加入 gsplat 這個新依賴，加之前要先問。
第 0 步是第 4 名的剖析加上一次 gsplat 在 1080p 與 4K 的算繪基準，若兩種租戶在這張卡上都是運算密集，互補打包的前提就不成立，直接關閉。

## 重新評估專題：G 系列當第一章，換問題、換指標、先確認硬體

G2 到 G9 應該重新定位成專題的第一章，而不是一個走不通的方向。
它在一個沒有系統論文量過的新工作負載上，用事先登記的預測與配對對照，複製了「飽和 GPU 上的共享機制只重新分配、不增加容量」這條已知規律，並留下第一批 feed-forward 3DGS 的多租戶數字：每次更新 0.19 s、每人 4,094 MiB、飽和約 4.7 updates/s、共享代價 9% 到 11% ([LOG.md G5, G6](../../LOG.md))。
這在課程報告裡站得住，但要明說它是預期內的結果，並引用 Clockwork、Gilman & Walls 與 NVIDIA 文件，否則讀者會誤以為它是新發現，或誤以為它是失敗。
問題本身要換掉，從「哪個機制能多服務人」（答案已知：沒有）換成「GPU 飽和時，虛擬化層能為每位使用者做什麼、不能做什麼」。
前者回答容量，後者回答隔離、准入與記憶體，正好就是學生在乎的資源分配與每使用者延遲。

指標也要從「固定每人頻率下 p95 不超過 1 s 的人數」換成 serving 文獻的標準配方。
G2 自己已經指出，開迴路下 N ≥ 3 的 p95 會隨執行長度一起長，不是穩定的量 ([LOG.md G2](../../LOG.md))。
可以照抄的做法是事先寫下 SLO、把供給負載掃過飽和點、到達過程用 Poisson 或指定 CV 的 Gamma、goodput 把遲到與被丟的更新都算失敗，並把容量定義成固定達成率下的最大使用者數。
這個容量定義在文獻裡有三種現成的寫法：Nexus 用「99% 在 SLO 內的最大請求率」，Ekya 用「跟得上幀率的串流數」，ARISE 用「SLA 內的客戶端數」([Nexus](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf); [Ekya, NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/bhardwaj); [ARISE](https://qiangxu1996.github.io/assets/pdf/Kong2024_ARISEOffloadingAR.pdf))。
再加上三個學生在乎、文獻少見的量：每位使用者的 p99 新鮮度、Jain fairness，以及受害者相對單獨執行的延遲膨脹倍數。
PilotFish 把 p99 FPS 除以目標值的正規化做法，也能讓不同 N 之間的數字可以比較 ([PilotFish](https://www.usenix.org/system/files/atc22-zhang-wei.pdf))。

硬體前提要在登記前確認，否則排名會被可行性推翻。
RTX 6000 Ada 不在 MIG 支援清單上，H200 最多支援 7 個 MIG 實例 ([MIG User Guide](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_MIG_User_Guide.pdf))。
專題 2026-11-02 的 nano4 探測（job 682344，記錄在專案記憶筆記，不是 repo 檔案）看到 MIG 未啟用、也沒有 MIG GRES，這和「可以用 H200 的 MIG」的前提相衝突，要先向管理者確認。
本機驅動 595.91.07 列出了 MPS 靜態切分的 `-S` 旗標，但還沒跑過，MPS v3 的加權時間片要 r620，cgroup 記憶體切分要 CUDA 13.4 與 root，本機都沒有 ([MPS 文件](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf))。
`nvidia-smi compute-policy --set-timeslice` 存在且目前是 Default，但它要不要 root、在 Ada 上是否真的改變約 2 ms 的時間片都還沒測，若可用就是第 1 名一個便宜的對照臂。
kind 叢集上 device plugin 的 MPS 模式不支援 MIG，也不能和 time-slicing 混用，所以 Kubernetes 層的比較要分開跑 ([k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin))。

新穎性要誠實地放在工作負載上，而不是機制上。
第 1、2 名的結果方向都能從文獻事先預測，所以很容易登記反駁條件。
它們的新意是「N 個同等優先、跑同一個重模型、更新會互相取代的互動租戶」這個設定，以及 Kubernetes 共享模式的隔離第一次被量化。
只有第 3 名有機會在可服務人數這個學生最在乎的指標上出現非零的差異，所以若它的第 0 步顯示記憶體會綁住，它值得當延伸。

## 結論

這次整理改變的是問題的位置。
在運算飽和的 GPU 上，虛擬化的工作從「打包」變成「保護與拒絕」，能讓可服務人數移動的槓桿不在共享機制，而在每位使用者的工作量與狀態；文獻裡 ARISE、vLLM 與 MuV2 的倍數全都來自這兩處，沒有一個來自 time-slicing 或 MPS。
專題的零結果因此不是死路，而是一條量出來的界線：它證明了槓桿不在哪裡，剩下要做的只是換一個能被反駁的問題。
一個值得記下的反直覺點是，專題目前最有價值的未解之謎是 G9 裡 batch 8 的 `forward_gaussians` 花了 batch 1 的 9.1 倍時間，也就是一個 feed-forward 3DGS 模型在資料中心 GPU 上每個樣本越批次越貴，這比 serving 文獻對飽和 GPU 的預期（batching 不再加速）還要糟，也是第 4 名的剖析雖然單獨太薄、仍值得最先做的原因。

最划算的下一步是本週在本機 GPU 上把第 1、2、4 名的第 0 步跑完，它們各不超過一天，而且共用 `bench/capacity.py` 與 G9 的伺服器；第 3 名則等它自己的第 0 步決定要不要做。
依專題規則，每個第 0 步與後續 sweep 都要先在 plan mode 事先登記指標、null 與殺掉條件，本報告裡的門檻只是建議值。
