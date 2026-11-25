# 3DGS、體積影片與神經場景表示的網路串流（2022 至 2026-11）

本筆記寫於 2026-11-18，範圍是把 3DGS、體積影片（point cloud / mesh）、NeRF 送到使用者端的網路與多媒體系統研究。
已在 `docs/reading_notes.md`（2026-10-27 雙人核實）深讀過的論文只引用，不重做；本次新讀的全文是 MuV2、NeRFHub、NeRFlow、MagicStream、DeformStream。
「已核實」指本次或 reading_notes 讀過一手來源（論文 PDF 或 arXiv abs 頁）；「未核實」指只看到搜尋摘要、二手網站或他人參考文獻。
不涵蓋：3DGS 硬體加速與 GPU 端算繪最佳化、通用 GPU 共享或 serving 系統、雲端遊戲、帶 KV cache 的串流重建模型。
所有搜尋都用 WebSearch，沒有逐一翻閱 SIGCOMM、NSDI、NOSSDAV、ISMAR 的議程目錄，所以「沒找到」只是弱證據。

## 問題 1：SIGCOMM、NSDI、MobiCom、MobiSys、ACM MM、MMSys、NOSSDAV、INFOCOM、IEEE VR、ISMAR 上有哪些 3DGS / 體積影片 / NeRF 串流論文？

### Takeaway
2022 年以來這條線集中在 MobiCom（Vues'22、MetaStream'23、FarfetchFusion'23、MuV2'24、L3GS'25、Vega'25）、MobiSys（NeRFHub'24、NeRFlow'25）、NSDI（YuZu'22）、MMSys（LTS'25 最佳論文、SGSS'25、LMG'26）、ACM MM（EyeNavGS'25、MD2G-Cast'26、WebSplatter'26）、INFOCOM（AirGS'26）、CoNEXT（LiVo'25）與 SIGCOMM 的 EMS workshop（EMS'24、NETSPLAT'25）。
2023 到 2026 年的 SIGCOMM / NSDI 主會、NOSSDAV、ISMAR，我用搜尋都沒找到 3DGS 串流論文；IEEE VR 只有 SRBF-Gaussian（2025，未核實）。
大量 3DGS 串流工作仍是 arXiv 預印本或圖學 / 視覺會議（SIGGRAPH、CVPR、AAAI）。

### Cited Findings

#### A. 同儕審查，本次新讀全文（已核實）

- **MuV2: Scaling up Multi-user Mobile Volumetric Video Streaming via Content Hybridization and Sharing**，ACM MobiCom 2024，pp. 327-341（頁碼取自 reading_notes 引用的參考文獻），作者 Yu Liu、Puqi Zhou、Zejun Zhang、Anlan Zhang、Bo Han、Zhenhua Li、Feng Qian - [PDF (NSF PAR)](https://par.nsf.gov/servlets/purl/10633047)
  - 借用的主張：用一台 commodity edge server 加一個 WiFi AP 服務「30 台以上同時連線的無線行動裝置」，做法是混合直接串 3D 內容與遠端算繪、跨使用者共享 edge 轉碼的視角、把多個轉碼 session 多工到少數硬體編碼器 - [PDF](https://par.nsf.gov/servlets/purl/10633047)
  - 評估設計：edge server 用 NVIDIA 2080Ti，AP 平均 450 Mbps、ping 小於 10 ms；四段自錄體積影片壓縮後約 150 Mbps；33 位使用者的觀看軌跡重播；對照組 DirectM、TransM、HybridM、ShareM 每個只差一個設計；指標含失真、Jain fairness index、MOS、每使用者頻寬、端到端延遲分解 - [PDF](https://par.nsf.gov/servlets/purl/10633047)
  - 可擴充性的測試混用真實裝置與模擬客戶端，論文明說「真實與模擬的比例保持一致」 - [PDF](https://par.nsf.gov/servlets/purl/10633047)
  - 程式碼：論文內沒看到釋出連結（只引用 Draco）。
  - 日期衝突：PDF 頁首寫 MobiCom '24「September 30-October 4, 2024」，搜尋到的投影片與列表寫 2024-11-18 至 22，以後者為會議實際日期較可能 - [投影片](https://zhan6841.github.io/assets/pdf/slides/mobicom24_muv2.pdf)
- **NeRFHub: A Context-Aware NeRF Serving Framework for Mobile Immersive Applications**，ACM MobiSys 2024，DOI 10.1145/3643832.3661879，pp. 85-98，作者 Bo Chen、Zhisheng Yan、Bo Han、Klara Nahrstedt - [PDF (NSF PAR)](https://par.nsf.gov/servlets/purl/10540540)；[researchr](https://researchr.org/publication/0025YHN24)
  - 借用的主張：「NeRF serving」是伺服器把資料送到行動端、由行動端算繪；把它寫成在品質與流暢度限制下最小化資料表示大小的問題；相對 MobileNeRF 基線，合成物件延遲降 56%-66%、真實場景降 26%-55% - [PDF](https://par.nsf.gov/servlets/purl/10540540)
  - 延遲的定義是「從使用者請求場景到第一幀算繪出來」，不是 motion-to-photon - [PDF](https://par.nsf.gov/servlets/purl/10540540)
  - 程式碼：本次沒找到釋出連結（未核實是否存在）。
- **NeRFlow: Towards Adaptive Streaming for NeRF Videos**，ACM MobiSys 2025，DOI 10.1145/3711875.3729160，作者 Rui-Xiao Zhang、Tianchi Huang、Bo Chen、Klara Nahrstedt - [PDF (NSF PAR)](https://par.nsf.gov/servlets/purl/10635564)
  - 借用的主張：NeRF 影片的 ABR，包含依算繪重要性做 voxel 剪枝、用預編碼的 master / sub-frame 補視角缺口、用遺傳演算法產生考慮頻寬與 ABR 的 bitrate ladder；QoE 提升 31.3%-41.2% - [PDF](https://par.nsf.gov/servlets/purl/10635564)
  - 動機量測：一個產品級網頁播放器的公開頻寬資料集平均約 36.87 Mbps，超過 94% 的 trace 低於 100 Mbps；30 FPS 下 INGP 與 DVGO 需要超過 2x10^4 Mbps，ReRF 約 200 Mbps，INGP-T 與 TiNeuVox 約 200-300 Mbps - [PDF](https://par.nsf.gov/servlets/purl/10635564)
  - 評估用 5G trace 測 rebuffering，超過 20% 的 trace 出現 rebuffering（原文段落細節未逐字核對）- [PDF](https://par.nsf.gov/servlets/purl/10635564)
  - 程式碼：本次沒找到釋出連結。
- **MagicStream: Bandwidth-conserving Immersive Telepresence via Semantic Communication**，ACM SenSys 2024（不在指定會議清單，但屬遠距臨場網路系統），作者 Ruizhi Cheng、Nan Wu、Vu Le、Eugene Chai、Matteo Varvello、Bo Han - [PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)
  - 借用的主張：只傳語意（膚色 / 衣服顏色與身體關鍵點），在接收端重建人體並做神經算繪；與 MetaStream 相近畫質下頻寬降 1195 倍，只用 0.2 Mbps；RTT 40 ms 下單向端到端延遲約 95 ms - [PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)
  - 評估設計：重新實作 MetaStream 作基線，五段各五分鐘的 session 平均 72.3 Mbps（SD 8.65）；在 HoloLens 2 上量 SSIM（0.92 對約 0.8）；兩次使用者研究；網路是約 150 Mbps 對稱頻寬 - [PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)
  - DOI 本次未取得；程式碼未見釋出連結。
- **DeformStream: Deformation-based Adaptive Volumetric Video Streaming**，arXiv 2409.16615，3 頁短文，作者 Boyan Li、Yongting Chen、Dayou Zhang、Fangxin Wang（CUHK-Shenzhen） - [arXiv](https://arxiv.org/abs/2409.16615)
  - 借用的主張：mesh 體積影片用嵌入式變形從幀間運動重建後續幀，並把「客戶端變形延遲」放進 QoE 模型，用動態規劃選 bitrate - [arXiv](https://arxiv.org/abs/2409.16615)
  - 版式像 IEEE VR / ISMAR 海報，但 PDF 內沒有寫出會議名稱，venue 未核實。

#### B. 同儕審查，只看到摘要或列表（未核實，數字來自摘要）

- **YuZu: Neural-Enhanced Volumetric Video Streaming**，USENIX NSDI 2022，pp. 137-154，作者 Anlan Zhang、Chendong Wang、Bo Han、Feng Qian - [USENIX](https://www.usenix.org/conference/nsdi22/presentation/zhang-anlan)
  - 借用的主張：點雲 3D super-resolution 以 30+ FPS 線速執行，配合 QoE 驅動的網路與運算資源調適；QoE 比不做 SR 高 37%-178%，比視埠自適應方案高 101%-175%；QoE 模型由 1,446 名受試者驗證，中位估計誤差 12.49% - [USENIX](https://www.usenix.org/conference/nsdi22/presentation/zhang-anlan)
- **Vues: Practical Mobile Volumetric Video Streaming through Multiview Transcoding**，ACM MobiCom 2022，作者 Yu Liu、Bo Han、Feng Qian、Arvind Narayanan、Zhi-Li Zhang - [PDF](https://www-users.cse.umn.edu/~fengqian/paper/vues_mobicom22.pdf)
  - 借用的主張：edge 依預測視埠把體積內容轉成多個 2D 視角；QoE 比單視角轉碼平均高 35%（最多 85%），頻寬比直接串體積影片少 95%；目標是 VoD - [PDF](https://www-users.cse.umn.edu/~fengqian/paper/vues_mobicom22.pdf)
- **MetaStream: Live Volumetric Content Capture, Creation, Delivery, and Rendering in Real Time**，ACM MobiCom 2023，DOI 10.1145/3570361.3592530，作者 Yongjie Guan、Xueyu Hou、Nan Wu、Bo Han、Tao Han - [NSF PAR](https://par.nsf.gov/biblio/10467140)
  - 借用的主張：直播體積內容端到端系統，含 edge 輔助物件分割、跨相機冗餘點移除、foveated 算繪；WiFi 上接近 30 FPS，端到端延遲最多降 31.7% - [NSF PAR](https://par.nsf.gov/biblio/10467140)
  - MagicStream 的重新實作量到它每幀約 200K 點、平均 72.3 Mbps - [MagicStream PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)
- **FarfetchFusion: Towards Fully Mobile Live 3D Telepresence Platform**，ACM MobiCom 2023，DOI 10.1145/3570361.3592525，作者 Kyungjin Lee、Juheon Yi、Youngki Lee（首爾大學） - [PDF](https://juheonyi.github.io/files/FarFetch.pdf)
  - 借用的主張：行動 GPU 上算繪需 100-180 ms，所以全行動 3D 臨場需要端到端協同設計 - [PDF](https://juheonyi.github.io/files/FarFetch.pdf)
  - 一個中文彙整網站把作者寫成 Kämäräinen 與 Siekkinen，與論文本身矛盾，以論文為準。

#### C. 同儕審查，已在 `docs/reading_notes.md` 核實，這裡只引用

- **LTS**，ACM MMSys 2025 最佳論文，DOI 10.1145/3712676.3714445：動態多層 3DGS 的 DASH 串流（Layer、Tile、Segment），系統程式碼因專利保留，Dynamic-LapisGS 有釋出 - [ACM DL](https://dl.acm.org/doi/10.1145/3712676.3714445)
- **L3GS**，ACM MobiCom 2025，DOI 10.1145/3680207.3723485，arXiv 2504.05517：分層靜態 3DGS，每秒跑一次 multiple-choice knapsack 排程 - [arXiv](https://arxiv.org/abs/2504.05517)
- **Vega**，ACM MobiCom 2025，DOI 10.1145/3680207.3765267：手機上 30 FPS 的 3DGS 體積影片，物件級選擇性運算（reading_notes 標註主文未讀） - [ACM DL](https://dl.acm.org/doi/10.1145/3680207.3765267)
- **SGSS**，ACM MMSys 2025，pp. 46-56：靜態 3DGS 的視角自適應串流（reading_notes 標註未讀全文） - [Rutgers](https://www.researchwithrutgers.com/en/publications/sgss-streaming-6-dof-navigation-of-gaussian-splat-scenes/)
- **LMG**，ACM MMSys 2026：分層 mesh-Gaussian 場景串流，NTHU Cheng-Hsin Hsu 團隊 - [專案頁](https://aiins-nthu.github.io/LMG/)
- **EyeNavGS**，ACM MM 2025，arXiv 2506.02380：46 名受試者在 12 個 3DGS 場景的 6-DoF 頭部與眼動 trace，含重播工具 - [arXiv](https://arxiv.org/abs/2506.02380)
- **MD2G-Cast**，ACM MM 2026（依 arXiv comments），arXiv 2608.10020：MoQ relay 依視角重疊把接收者分組做 multicast，擴到 100 名使用者 - [arXiv](https://arxiv.org/abs/2608.10020)
- **WebSplatter**，ACM MM 2026（reading_notes 摘要表，相關段落另註 arXiv 頁未寫 venue），arXiv 2602.03207：WebGPU 跨裝置 3DGS 算繪 - [arXiv](https://arxiv.org/abs/2602.03207)
- **AirGS**，IEEE INFOCOM 2026（依 arXiv comments），arXiv 2512.20943：串流相鄰幀間的 Gaussian 差異，用 ILP 依頻寬選剪枝等級 - [arXiv](https://arxiv.org/abs/2512.20943)
- **LiVo**，ACM CoNEXT 2025（PACMNET），DOI 10.1145/3768981：雙機體積視訊會議，WebRTC 上依接收端視錐裁切並動態分配兩條 H.265 串流的頻寬 - [GitHub](https://github.com/USC-NSL/LiVo)
- **Multi-frame Bitrate Allocation of Dynamic 3DGS Streaming**，ACM SIGCOMM EMS 2024 workshop（未讀全文） - [PDF](https://people.cs.nycu.edu.tw/~chuang/pubs/pdf/2024ems.pdf)
- **NETSPLAT**，ACM SIGCOMM EMS 2025 workshop，DOI 10.1145/3746441.3748225：可程式化資料平面輔助 3DGS 串流（multicast 群組重組、佇列排程；未讀全文） - [ACM DL](https://dl.acm.org/doi/10.1145/3746441.3748225)
- **SRBF-Gaussian**，IEEE VR 2025，pp. 461-471：reading_notes 只確認存在，未讀 - 見 `docs/reading_notes.md` 第 5 點。
- **MoQSplat**，IEEE MMSP 2026，arXiv 2609.18624；**StreamSTGS**，AAAI 2026，arXiv 2511.06046；**CAGS**，SIGGRAPH 2026，arXiv 2605.09279；**V^3**，SIGGRAPH Asia 2024 / ACM TOG，arXiv 2409.13648：都不在指定的網路會議，但都是串流設計 - [MoQSplat](https://arxiv.org/abs/2609.18624)；[StreamSTGS](https://arxiv.org/abs/2511.06046)；[CAGS](https://arxiv.org/abs/2605.09279)；[V^3](https://arxiv.org/abs/2409.13648)
- **GaussianNexus**（ACM UIST 2025）與 **LiveGS**（SIGGRAPH 2025 Emerging Technologies）只讀過摘要 - [GaussianNexus](https://dl.acm.org/doi/10.1145/3746059.3747693)；[LiveGS](https://dl.acm.org/doi/10.1145/3721257.3734033)

#### D. 期刊（不在指定會議清單）

- **Immersive Volumetric Video Playback: Near-RT Resource Allocation and O-RAN-based Implementation**，IEEE Transactions on Mobile Computing 2026，DOI 10.1109/TMC.2026.3674975，arXiv 2601.20625，作者 Yao Wen、Luping Xiang、Kun Yang - [arXiv](https://arxiv.org/abs/2601.20625)
  - 借用的主張：在 O-RAN 的 Near-RT 控制迴圈裡，把「算繪像素比例」當連續控制變數，與 O-Cloud 運算、gNB 發射功率、頻寬一起最佳化；用 Soft Actor-Critic 求解；中位 MTP 延遲降 11% 以上，並改善平均 QoE 與公平性 - [arXiv](https://arxiv.org/abs/2601.20625)
  - 評估：5G O-RAN 測試床加系統模擬；使用者人數在摘要頁沒寫 - [arXiv](https://arxiv.org/abs/2601.20625)

#### E. arXiv 預印本（沒有同儕審查 venue，除非另註）

- **Streaming Real-Time Rendered Scenes as 3D Gaussians**，arXiv 2604.02851：伺服器持續更新 3DGS 並送 delta，單一客戶端平均 92.5 Mbps、中位 66.8 Mbps、峰值 481.8 Mbps；「多使用者攤提」的主張沒有被測（reading_notes 已核實） - [arXiv](https://arxiv.org/abs/2604.02851)
- **GaussAnything**（2609.13859）、**Renderable Partial Representations for Dynamic GS under Incomplete Delivery**（2606.17212）、**TIGAS**（2605.08699）、**GS-NFS**（2606.05650）、**SwinGS**（2409.07759）、**Voyager**（2506.02774）、**ReVo**（2604.27441）、**VoluMe**（ICCV 2025, 2507.21311）：都已在 reading_notes 深讀 - [GaussAnything](https://arxiv.org/abs/2609.13859)；[2606.17212](https://arxiv.org/abs/2606.17212)；[TIGAS](https://arxiv.org/abs/2605.08699)；[GS-NFS](https://arxiv.org/abs/2606.05650)
- **ResStreamGS: Residual-Sensitive Dual-Stream Gaussian Splatting for Mobile Volumetric Video Streaming**，arXiv 2610.21121（2026-10-15），無 venue - [arXiv](https://arxiv.org/abs/2610.21121)
  - 借用的主張：V^3 類管線把每顆 Gaussian 當成每幀都在變；ResStreamGS 把穩定的 Gaussian 每個 GOP 只送一次，殘差敏感的 Gaussian 走可硬體解碼的逐幀影像串流；對照 V^3 式基線 post-codec PSNR 高 7.51-8.35 dB，攤提酬載最多少 56.5%，Snapdragon 778G 上 30.883 FPS - [arXiv](https://arxiv.org/abs/2610.21121)
  - 程式碼：摘要頁沒寫。
- **Nebula: Enable City-Scale 3D Gaussian Splatting in Virtual Reality via Collaborative Rendering and Accelerated Stereo Rasterization**，arXiv 2512.20495（cs.AR），無 venue - [arXiv](https://arxiv.org/abs/2512.20495)
  - 借用的主張：雲端做 LoD 搜尋、客戶端做延遲敏感的算繪；摘要報告 motion-to-photon 加速 2.7 倍；頻寬降幅在摘要頁被抽成「1925%」，應是「19-25%」的排版錯誤，未核對 PDF - [arXiv](https://arxiv.org/abs/2512.20495)
  - 只有單一客戶端；它的硬體加速部分不在本筆記範圍。
- **Mon3tr: Monocular 3D Telepresence with Pre-built Gaussian Avatars as Amortization**，arXiv 2601.07518，無 venue - [arXiv](https://arxiv.org/abs/2601.07518)
  - 借用的主張：離線建好個人 Gaussian avatar，線上只用 WebRTC data channel 傳動作與外觀特徵，低於 0.2 Mbps，比點雲串流少 1000 倍以上；Quest 3 上約 60 FPS，端到端約 80 ms - [arXiv](https://arxiv.org/abs/2601.07518)
- **Adaptive 3D Gaussian Splatting Video Streaming: Visual Saliency-Aware Tiling and Meta-Learning-Based Bitrate Adaptation**，arXiv 2507.14454，作者 Han Gong、Qiyue Li、Jie Li、Zhi Liu，無 venue - [arXiv](https://arxiv.org/abs/2507.14454)
  - 借用的主張：3DGS 影片的顯著性分塊、每塊多個品質版本、meta-learning ABR；摘要沒有數字，程式碼說接受後釋出 - [arXiv](https://arxiv.org/abs/2507.14454)
  - 同名的 arXiv 2507.14432 只看到搜尋摘要（未核實） - [arXiv](https://arxiv.org/abs/2507.14432)
- **ECO-GS (Edge Collaborative Gaussian Splatting with Integrated Rendering and Communication)**，arXiv 2510.22718，reading_notes 記為 ICASSP 2026 - [arXiv](https://arxiv.org/abs/2510.22718)
  - 借用的主張（只看到搜尋摘要，未核實）：每個使用者在本地小模型與 edge 大模型之間切換，聯合最佳化協作狀態與 edge 發射功率；所有人都協作時 10 mW 總功率下延遲超過 80 ms，作者的演算法壓在 60 ms 預算內 - [arXiv HTML](https://arxiv.org/html/2510.22718v2)
- **STT-GS (Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control)**，arXiv 2510.13186 - [arXiv](https://arxiv.org/abs/2510.13186)
  - 借用的主張（只看到二手評論網站，未核實）：多個客戶端把樣本上傳 edge 做 3DGS 重建，聯合選客戶端與功率控制；rubble-pixsfm 上 PSNR 比 MaxRate 高 4.50%、比 Fairness 高 7.81% - [二手評論](https://www.themoonlight.io/review/stt-gs-sample-then-transmit-edge-gaussian-splatting-with-joint-client-selection-and-power-control)
- **GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System**，只在 ResearchGate（2026-03），有深度強化學習元件，venue 不明（未核實） - [ResearchGate](https://www.researchgate.net/publication/401772993_GSStream_3D_Gaussian_Splatting_based_Volumetric_Scene_Streaming_System)
- **Evaluation of strategies for efficient rate-distortion NeRF streaming**，arXiv 2410.19459：比較傳編碼後的影像與傳 NeRF 參數，結論是一對多時傳參數較省（只看到搜尋摘要，未核實） - [arXiv](https://arxiv.org/abs/2410.19459)

### Inferences
- 網路系統會議裡做 3DGS / 體積串流的作者群很集中：Bo Han（GMU）與 Feng Qian 一線（YuZu、Vues、MuV2、MetaStream、MagicStream、NeRFHub），以及 NTHU Cheng-Hsin Hsu 一線（LTS、SGSS、EyeNavGS、LMG、TIGAS）。
- 3DGS 本身進入 MobiCom / MMSys / INFOCOM 是 2025 年以後的事；2022-2024 的網路會議論文主要是點雲、mesh 與 NeRF。

### Gaps
- ACM DL 擋下載，所以 YuZu、Vues、MetaStream、FarfetchFusion、Vega、SGSS、NETSPLAT、EMS'24 的評估細節只來自摘要。
- NOSSDAV 2025 / 2026、ISMAR 2024-2026、SIGCOMM / NSDI 2025-2026 的議程沒有逐篇翻；reading_notes 提到的 ACPGS（NOSSDAV'26，CAGS 引用）仍未核實。
- MuV2、NeRFHub、NeRFlow、MagicStream 是否釋出程式碼：論文內沒找到連結，也沒有另外查 GitHub。

## 問題 2：系統設計（伺服器端或客戶端算繪、每次更新送什麼、每幀多少位元組、調適邏輯）與瓶頸在哪？

### Takeaway
設計可分四種切點：送 3D 內容由客戶端算繪、伺服器算繪後送像素、兩者混合、只送語意或參數讓接收端生成。
前者的瓶頸是頻寬與客戶端解碼 / 算繪，伺服器算繪的瓶頸是伺服器 GPU 的算繪與硬體編碼（隨使用者數線性成長），語意傳輸把瓶頸推到接收端運算。
已發表的頻寬數字從每使用者 0.2 Mbps（語意）到 70-150 Mbps（點雲）到上萬 Mbps（未壓縮 NeRF）都有。

### Cited Findings

#### 切點 1：送 3D 內容，客戶端算繪
- 直接串點雲體積影片時，ViVo 無損約 90 Mbps，500 Mbps 的 AP 理想上只能服務 5 人；MuV2 實測 DirectM 只撐到 3 人，受頻寬限制 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MuV2 的四段測試影片壓縮後約 150 Mbps - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MetaStream 式點雲臨場（每幀約 200K 點）平均 72.3 Mbps；每幀 1M 點、30 FPS 可能要 450 Mbps；HoloLens 2 解 1M 點不到 7 FPS - [MagicStream PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)
- 持續更新的 3DGS 用 delta 串流（2604.02851），單一客戶端平均 92.5 Mbps、峰值 481.8 Mbps，送的是每個屬性的量化 tensor delta 加週期性 Draco 快照，沒有任何頻寬調適 - [arXiv 2604.02851](https://arxiv.org/abs/2604.02851)；細節見 `docs/reading_notes.md`
- NeRF 影片在 30 FPS 下 INGP / DVGO 超過 2x10^4 Mbps，ReRF 約 200 Mbps；NeRFlow 用 voxel 剪枝與多 bitrate 階梯做 ABR - [NeRFlow PDF](https://par.nsf.gov/servlets/purl/10635564)
- NeRFHub 把 NeRF 訓練成可設定運算量與 bitrate 的表示，依裝置與場景挑設定來最小化首幀延遲 - [NeRFHub PDF](https://par.nsf.gov/servlets/purl/10540540)
- 3DGS 的調適邏輯：L3GS 依視埠效用跑 knapsack；LTS 走 DASH 層 / 塊 / 段；AirGS 用 ILP 選剪枝等級；2606.17212 依「準時到達機率 x 算繪效用 / 位元組」排程；GaussAnything 在客戶端用固定常駐 Gaussian 預算與合併 dirty slot - [L3GS](https://arxiv.org/abs/2504.05517)；[AirGS](https://arxiv.org/abs/2512.20943)；[2606.17212](https://arxiv.org/abs/2606.17212)；[GaussAnything](https://arxiv.org/abs/2609.13859)
- ResStreamGS 把穩定 Gaussian 每 GOP 送一次、殘差敏感的走影像編碼器，攤提酬載最多少 56.5% - [arXiv 2610.21121](https://arxiv.org/abs/2610.21121)
- 手機端的瓶頸是算繪：Vega 用物件級選擇性運算趕算繪期限；FarfetchFusion 報告行動 GPU 算繪 100-180 ms - [Vega](https://dl.acm.org/doi/10.1145/3680207.3765267)；[FarfetchFusion](https://juheonyi.github.io/files/FarFetch.pdf)

#### 切點 2：伺服器算繪，送像素
- MuV2 的 TransM 每位使用者都要在 edge 算繪並編碼一張 RGB 與一張深度圖；2080Ti 在 1080p 編碼 480 FPS，所以 30 FPS 下最多約 8 人；實測 8 人時 GPU 編碼器用量到 95% - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MuV2 的註腳說多數支援並行 session 的 NVIDIA GPU 編碼 session 上限是 3 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- Vues 由 edge 依預測視埠轉成多個 2D 視角，頻寬比直接串體積內容少 95%（摘要） - [Vues PDF](https://www-users.cse.umn.edu/~fengqian/paper/vues_mobicom22.pdf)
- TIGAS 讓 GPU 後端針對每個 pose 請求算繪 3DGS 並經 HTTP/3 回傳 JPEG，客戶端用延遲導向 ABR - [TIGAS](https://arxiv.org/abs/2605.08699)
- O-RAN 系統把「算繪像素比例」當控制變數，同時分配 O-Cloud 運算與無線資源，目標是 MTP 延遲 - [arXiv 2601.20625](https://arxiv.org/abs/2601.20625)

#### 切點 3：混合或拆分
- MuV2 動態決定哪些使用者收 3D 幀、哪些收轉碼視角，並讓相近使用者共用同一個 edge 算繪的視角，靠影像扭曲生成各自的新視角 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MuV2 端到端處理延遲：轉碼視角約 50 ms、體積幀約 26 ms（投影片） - [MuV2 投影片](https://zhan6841.github.io/assets/pdf/slides/mobicom24_muv2.pdf)
- Nebula 與 Voyager 讓雲端做 LoD 搜尋、客戶端算繪 - [Nebula](https://arxiv.org/abs/2512.20495)；[Voyager](https://arxiv.org/abs/2506.02774)

#### 切點 4：只送語意或參數
- MagicStream 0.2 Mbps、單向端到端約 95 ms（RTT 40 ms）；伺服器端的 edge 承擔重建與神經算繪 - [MagicStream PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)
- Mon3tr 低於 0.2 Mbps，約 80 ms 端到端，算繪在 Quest 3 上 - [arXiv 2601.07518](https://arxiv.org/abs/2601.07518)
- VoluMe 網路上送普通 2D 影片，由接收端每幀預測 Gaussian - [arXiv 2507.21311](https://arxiv.org/abs/2507.21311)

#### 本專題自己的數字（repo 內）
- StreamSplat 每次更新約 73,728 顆 Gaussian；reading_notes 估計每個 keyframe fp32 約 2.8 MB（靜態欄位送一次加兩組動態欄位），天真送法約 4.1 MB；fp16 減半；這是估計，從未量測 - `docs/reading_notes.md` 第 720-726 行
- 一個 StreamSplat 使用者每個 keyframe 花 0.19 s GPU、4.0 GiB，單一使用者背靠背跑時 GPU 約 5.2 updates/s（G5）；每人每秒一個 keyframe 時服務 4 人（G6）；跨使用者 batching 沒有提高吞吐（G9：batch 8 為 4.52 updates/s，batch 1 為 4.92） - `LOG.md` G5、G6、G9

### Inferences
- 用上面的估計算：每個 StreamSplat 更新約 22-33 Mbit（fp32），所以整張 RTX 6000 Ada 在 5.2 updates/s 飽和時，總出口流量約 117-172 Mbps（fp32），fp16 約 59-86 Mbps。
  這比 MuV2 用的單一 450 Mbps WiFi AP 還小，所以在「每個使用者重建自己的場景、自己看」的設定下，伺服器 GPU 會比共享網路先滿。
- 但對單一使用者，每秒 1 個 keyframe 的 fp32 串流約 22-33 Mbps，已接近 NeRFlow 資料集的平均頻寬 36.87 Mbps；每秒 3 個 keyframe（G7 的設定）約 66-99 Mbps，會超過大多數該資料集的 trace。
  所以網路若成為瓶頸，會是個別使用者的最後一哩，不是伺服器出口。
- 伺服器算繪送像素的設計，其 GPU 成本隨觀看人數線性成長（MuV2 的 TransM 8 人上限）；送 Gaussian 的設計，伺服器成本只隨「更新數」成長，不隨觀看人數成長。
  在本專題「一人一場景」的設定下，這個差異用不上；在「一個重建、多人觀看」的設定下，它是主要的取捨。

### Gaps
- 沒有任何論文量測 feed-forward 3DGS 模型（StreamSplat、AnySplat）輸出的每次更新位元組數或在真實網路上的延遲；上面的數字是 reading_notes 的估計。
- MuV2 的每使用者頻寬（Figure 13）與延遲分解（Figure 14）只有圖，文字裡沒給數字，本次沒有從圖中讀值。

## 問題 3：有沒有論文處理許多並行使用者共享伺服器資源（GPU 或頻寬）？如何分配（QoE 公平性、准入控制）？

### Takeaway
有，但很少，而且它們共享的都是「送達或算繪 / 轉碼」資源：WiFi AP、GPU 硬體編碼器、relay 頻寬、無線與 edge 運算。
我沒找到任何論文讓一張伺服器 GPU 同時替許多使用者各自做即時 3D 重建，並量每使用者延遲。
MuV2 是最接近的模板：它先找出每種設計的單一瓶頸，再用跨使用者共享把運算需求與人數脫鉤，並用 Jain fairness index 評估。

### Cited Findings
- MuV2 的設計原則：同時用滿網路與運算、讓運算需求與使用者人數脫鉤、把剩下的工作多工到有限的硬體編碼器 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MuV2 的可擴充性以「最差使用者」為下界：DirectM 3 人（頻寬）、TransM 8 人（GPU 編碼器 95%）、HybridM 10 人（兩者皆受限）、ShareM 與 MuV2 30 人以上（固定數量的共享視角讓編碼負載不再隨人數成長） - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MuV2 的最佳化目標含失真、品質切換與 Jain fairness index；選視角用 forward-greedy，與窮舉相比目標值差不到 0.5%，耗時從 100 ms 降到 10 ms 以下 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MuV2 引用的 M5 是唯一更早的多使用者體積串流系統，靠 mmWave multicast，最多 5 人 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- MD2G-Cast 用 MoQ relay 依觀看重疊與能力把使用者分組，共享 Base 內容並對 Enhanced 內容做准入（PPO 與蒸餾策略），Mininet 中擴到 100 人 - [arXiv 2608.10020](https://arxiv.org/abs/2608.10020)
- TIGAS 的後端服務多個模型與多個使用者：tensor 常駐 VRAM、延遲載入、reference counting、閒置時逐出 - [arXiv 2605.08699](https://arxiv.org/abs/2605.08699)
- O-RAN 的系統以 SAC 在 Near-RT 迴圈中分配算繪像素比例、運算、功率、頻寬，報告 MTP 中位數降 11% 以上與公平性改善 - [arXiv 2601.20625](https://arxiv.org/abs/2601.20625)
- ECO-GS 在 edge 功率與協作決策上做多使用者分配，以 60 ms 延遲預算為限制（未核實） - [arXiv 2510.22718](https://arxiv.org/html/2510.22718v2)
- STT-GS 是少數「多個客戶端上傳、edge 做 3DGS 重建」的設定，但分配的是無線功率與客戶端選擇，不是 GPU，結果以 PSNR 而非延遲表示（未核實） - [arXiv 2510.13186](https://arxiv.org/abs/2510.13186)
- NETSPLAT 在資料平面做 multicast 群組重組與佇列排程以支援多個 3DGS 客戶端（未讀全文） - [ACM DL](https://dl.acm.org/doi/10.1145/3746441.3748225)
- 2604.02851 主張「伺服器端建模比逐使用者算繪加影片串流更能攤提到多個使用者」，但所有串流實驗都只有一個客戶端，reading_notes 判為 unsupported - [arXiv 2604.02851](https://arxiv.org/abs/2604.02851)；`docs/reading_notes.md` 第 405 行

### Inferences
- MuV2 的教訓直接對應本專題的 G2-G9：當單一資源（它的 NVENC，我們的 SM 運算）飽和時，換共享機制（time-slicing、MPS、batching）不會多服務一個人；要多服務人，必須讓需求不隨人數成長，也就是跨使用者共享工作。
- 本專題「每人一個獨立場景」的設定裡沒有可共享的工作，所以 MuV2 式的脫鉤無從發生；要有可共享的工作，必須改設定成多人看同一個重建，或多人的輸入有重疊。
- 文獻裡的公平性一律是品質公平（Jain index 作用在失真或 QoE 上）；沒有看到以「每使用者場景新鮮度」或「每使用者更新延遲」為公平對象的分配。

### Gaps
- 沒有找到任何以 GPU 時間為分配對象、以每使用者重建延遲為指標的多使用者 3DGS 或體積重建系統。
- MuV2 的准入邏輯只到「選哪些人收 3D 幀」，沒有拒絕使用者的准入控制；本次讀的論文都沒有明確的 admission control。
- MD2G-Cast 與 TIGAS 的多使用者細節以 reading_notes 為準，本次沒有重讀。

## 問題 4：怎麼評估（trace、QoE 指標、頻寬、motion-to-photon 延遲、基線）？

### Takeaway
網路會議的典型評估是：重播真實頻寬 trace 與真實使用者動作 trace，搭配只差一個設計的系統對照組，報告 QoE（品質、切換、卡頓）加頻寬加延遲，常附使用者研究。
延遲的定義各篇不同（首幀延遲、單向端到端、MTP、處理延遲），本專題要的「每使用者更新延遲」或 glass-to-glass 在文獻中幾乎沒人量。

### Cited Findings
- **頻寬 trace**：NeRFlow 用產品級播放器的公開頻寬資料集（平均 36.87 Mbps）與 5G trace；L3GS 用 Lumos5G 步行 trace；AirGS 用 LTE 行車 trace；LiVo 用 Mahimahi 重播；MD2G-Cast 用 4G / 5G / Wi-Fi / 光纖 trace - [NeRFlow PDF](https://par.nsf.gov/servlets/purl/10635564)；[L3GS](https://arxiv.org/abs/2504.05517)；[AirGS](https://arxiv.org/abs/2512.20943)；[LiVo](https://github.com/USC-NSL/LiVo)；[MD2G-Cast](https://arxiv.org/abs/2608.10020)
- **使用者動作 trace**：MuV2 收 33 名使用者的觀看軌跡；EyeNavGS 46 名受試者、12 個 3DGS 場景；MD2G-Cast 用 CWI 6-DoF 頭部 trace - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)；[EyeNavGS](https://arxiv.org/abs/2506.02380)
- **QoE 與公平性**：MuV2 用失真、品質切換、Jain fairness index、MOS；YuZu 建了經 1,446 名受試者驗證的 QoE 模型；DeformStream 的 QoE 把客戶端變形延遲列為一項 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)；[YuZu](https://www.usenix.org/conference/nsdi22/presentation/zhang-anlan)；[DeformStream](https://arxiv.org/abs/2409.16615)
- **延遲**：NeRFHub 的延遲是請求到首幀；MagicStream 是單向端到端（約 95 ms）；Mon3tr 約 80 ms 端到端；O-RAN 系統用 MTP 中位數；TIGAS 量 MTP；MuV2 給端到端延遲分解 - [NeRFHub PDF](https://par.nsf.gov/servlets/purl/10540540)；[MagicStream PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)；[Mon3tr](https://arxiv.org/abs/2601.07518)；[arXiv 2601.20625](https://arxiv.org/abs/2601.20625)
- **可擴充性**：MuV2 以最差使用者的幀率判定能撐幾人，並混用真實與模擬客戶端 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- **基線**：MuV2 的 DirectM / TransM / HybridM / ShareM 是一次只加一個設計的階梯；NeRFHub 以 MobileNeRF 為基線；MagicStream 重新實作 MetaStream 並說明效能與原論文相近 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)；[NeRFHub PDF](https://par.nsf.gov/servlets/purl/10540540)；[MagicStream PDF](https://vule20.github.io/papers/magicstream-sensys24.pdf)
- reading_notes 第 6 點已整理值得照抄的評估設計：預載或 oracle 上限（L3GS）、同一壅塞控制與種子下只差策略的對照與 paired bootstrap CI（2606.17212）、配對在相同品質下比較（GS-NFS）；並指出沒有一篇量相機到瀏覽器的 glass-to-glass 延遲 - `docs/reading_notes.md`「Related work the material missed」第 6 點

### Inferences
- MuV2 的評估形狀（每個對照組只差一個設計、以最差使用者判定可服務人數、同時報告頻寬與 GPU 編碼器用量）正好就是本專題 G 系列的形狀，可以直接沿用並引用。
- 文獻報的頻寬與延遲數字都不能當本專題的 null，因為硬體、內容、trace 都不同；要用的話只能當數量級參考。

### Gaps
- 沒有找到以「每使用者場景更新的年齡（staleness）」為指標的論文。
- 多數論文的使用者研究與 trace 沒有公開下載位置；EyeNavGS 與 LiVo 例外。

## 問題 5：對一個關心資源分配與每使用者延遲的專題，還有什麼是開放、可以量的？

### Takeaway
最清楚的空白有兩個：一是沒人量過 feed-forward 3DGS 串流的每次更新位元組數與它在真實網路上何時變成瓶頸；二是「伺服器建一次 Gaussian、多人各自算繪」比「伺服器替每人算繪送像素」更能攤提的主張，連提出它的 2604.02851 都沒測。
本專題已量到 GPU 運算在約 5 updates/s 飽和、共享機制不加人，這反而讓「一個重建、多人觀看」的扇出問題變成最可能出有趣結果的方向。

### Cited Findings
- 2604.02851 的多使用者攤提主張未經測試 - [arXiv 2604.02851](https://arxiv.org/abs/2604.02851)；`docs/reading_notes.md` 第 405 行
- 伺服器算繪的 GPU 編碼成本隨人數線性成長，2080Ti 約 8 人就到頂 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- 共享視角讓 MuV2 的編碼負載不再隨人數成長，撐到 30 人以上 - [MuV2 PDF](https://par.nsf.gov/servlets/purl/10633047)
- 本專題：GPU 約 5.2 updates/s 飽和（G5），每人每秒一更新時 4 人（G6），time-slicing、MPS、batching 都沒有多服務一人（G6、G7、G9） - `LOG.md`
- 沒有論文量相機到瀏覽器的 glass-to-glass 延遲 - `docs/reading_notes.md`

### Inferences

以下三題是**建議**，由上面的論文推出，不是文獻結論；每題都可在幾週內用 repo 現有的 `bench/capacity.py`、StreamSplat 環境與 `tc netem` 做。

#### 建議題 A（step 0，約 1 週）：網路什麼時候才會是瓶頸？
- 問題：StreamSplat 與 AnySplat 每次更新實際送多少位元組（fp32、fp16、16 B 打包、相對前一次的 delta），以及在真實的每使用者頻寬 trace 下，限制可服務人數的是伺服器 GPU 還是使用者的最後一哩？
- 為什麼值得做：任務限制裡說網路成本從未量過；本筆記的估計顯示伺服器出口不會是瓶頸，但單一使用者的連線在每秒 1-3 個更新時可能是。
- 量什麼：每次更新位元組數（實測）；重播 Lumos5G 或 NeRFlow 用的頻寬 trace 時每使用者更新延遲的 p50 / p99；在給定延遲 SLO 下的可服務人數。
- 事先可寫的反駁條件：若在所有 trace 下只用 GPU 容量預測的可服務人數與實測差不到 1 人，網路方向就關閉。
- 借用：MuV2 的瓶頸分析方法（DirectM 受頻寬限、TransM 受 GPU 限）。

#### 建議題 B（最可能有有趣結果，約 3-4 週）：一個重建、K 人觀看時，送 Gaussian 與送像素哪個每張 GPU 服務更多人？
- 問題：在遠距臨場或教室式的扇出設定（一個 StreamSplat 重建、K 名觀看者各自視角），比較「送 Gaussian、客戶端算繪」與「伺服器每人算繪加 NVENC 編碼、送像素」，在每使用者延遲 SLO 下，每張 GPU 與每條上行能服務幾人，交叉點在哪個 K？
- 為什麼值得做：它直接測 2604.02851 沒測的攤提主張，也是 MuV2 在點雲上做過、但沒人在 3DGS 重建上做過的取捨；本專題已證明「每人一場景」時 GPU 無法多服務人，扇出是唯一能讓需求與人數脫鉤的設定。
- 量什麼：伺服器 GPU 時間與 NVENC 用量對 K 的曲線；總出口頻寬對 K；每位觀看者的更新延遲 p50 / p99 與 Jain fairness；只差「送什麼」一個因素的對照。
- 風險：送 Gaussian 那臂需要瀏覽器或本地 gsplat 客戶端代表算繪端；客戶端 GPU 成本要另外報，不然是把成本藏到客戶端。

#### 建議題 C（約 2-3 週）：GPU 飽和時，把「更新頻率」當作 ABR 的品質階梯來分配
- 問題：N 個使用者各自重建、GPU 已飽和時，依延遲或新鮮度目標動態調每人的更新頻率（類比 ABR 的 bitrate ladder），能否在不加 GPU 的情況下改善最差使用者的更新延遲與公平性，並給出一個准入門檻？
- 量什麼：每使用者場景年齡（staleness）p50 / p99、Jain fairness、准入門檻；對照 FIFO 或 round-robin 與 deadline / 公平導向排程，只差排程策略。
- 誠實的限制：飽和時排程只能重新分配延遲，不能增加容量，所以結果可能是可預期的；它的價值在尾延遲、公平性與准入，而不是多服務人。
- 借用：MuV2 的 Jain fairness 與品質切換項、O-RAN 系統的「把連續的品質變數與資源一起分配」。

### Gaps
- 上面三題的可行性判斷沒有實際試跑；建議題 B 需要的客戶端算繪路徑（瀏覽器或 gsplat）與 NVENC 編碼路徑在 repo 中尚未存在，工時是估計。
- 我沒有查證 NTHU Cheng-Hsin Hsu 團隊或 Bo Han 團隊是否已有未發表的「多人觀看 3DGS 重建」工作；reading_notes 建議及早與 Hsu 團隊談。
