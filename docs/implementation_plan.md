# 實作計畫：M0 到 M3 的每一步怎麼做、產出哪個數字

課程：虛擬化（Virtualization）期末專題。
撰寫日期：2026-10-27，branch `docs/plan`，放在 `docs/implementation_plan.md`，和 `PLAN.md` 並列。

## 0. 這份文件怎麼讀

`PLAN.md` 說「為什麼量、量什麼、判定規則是什麼」；這份文件只說「怎麼做」。
研究理由一律連回 `PLAN.md` 的章節，這裡不重述。
每一個步驟都寫出：執行的指令、寫出哪個檔案、產出哪個數字、事先宣告的成功條件。
依照 `.claude/rules/research-discipline.md`，第 0 步就是一個量測，每個 runner 都把數字寫進 `metrics.json`，LOG 條目在 run 結束後才寫、寫了就不改。
目錄配置照 `~/.claude/skills/directory-layout/SKILL.md`，第 1 節是它在這個 repo 的具體長相。

術語第一次出現時會簡短解釋，`PLAN.md` 第 13 節已經解釋過的詞只補實作上需要的細節。
這裡先補四個第 1 節就會用到的瀏覽器安全名詞。
COOP（Cross-Origin-Opener-Policy）和 COEP（Cross-Origin-Embedder-Policy）是兩個 HTTP 回應標頭，同時設定時頁面進入「跨來源隔離」狀態，瀏覽器才允許 `SharedArrayBuffer`（多個執行緒共用同一塊記憶體）。
COEP 設成 `require-corp` 之後，頁面向別的來源抓的每個資源都必須帶 CORP（Cross-Origin-Resource-Policy）標頭明確同意被嵌入，否則瀏覽器拒絕載入。
ACAO（Access-Control-Allow-Origin）是允許跨來源 `fetch` 讀取回應內容的標頭；我們的靜態檔案伺服器和 Vite 開發伺服器在不同 port，所以 PLY 和紀錄檔的回應兩個標頭都要帶。

### 0.1 機器盤點後和 PLAN.md 不同的地方

這些是 2026-10-27 實際檢查這台工作站（t-3dgs-02-linux）的結果，PLAN.md 寫作時沒有這些資訊。

1. 這台機器沒有 display :20，也沒有 Chrome Remote Desktop；:0 是 root 的 LightDM 登入畫面，這個使用者拿不到授權。
   所以 M0 在本機只能用 `--headless=new` 搭配 NVIDIA Vulkan 驅動跑 WebGPU（第 2.3 節），它的 `requestAnimationFrame` 可能是計時器驅動而不是真的 vsync，所有數字都要標 `vsync_mode=headless_timer`。
   「幀間隔在 GPU 負載下會不會變長」在這種模式下是一個未驗證的假設，所以 V4 是一個正控制（positive control）：先確認這個儀器在已知會卡的負載下真的量得到卡，才能拿它去量別的東西。
   真正 vsync 的數字要等有人在主機登入桌面（:0 變成 xfce session）或用 Windows 那台。
2. 系統沒有 Chrome；唯一能用 GPU 的瀏覽器是 Playwright 裝的 Chrome for Testing 153.0.8010.12，路徑 `~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome`。
   它屬於另一個專案（`demo-assessment-system/.venv`）的 Playwright，不能刪那個 venv。
   Ubuntu 24.04 的 AppArmor 擋住非特權 user namespace，所以必須加 `--no-sandbox`。
3. Docker 29.7.1 加 nvidia-container-toolkit 可以用（使用者在 docker 群組，但要用 `sg docker -c '...'`）。
   PLAN.md 第 4.3 節和 11.8 節說「沒有 Docker」是錯的，這給 M4 一條不需要 sudo 的 `tc` 路線（第 9.4 節）。
4. `sudo` 要密碼、沒有 NOPASSWD；`iperf3` 沒裝；`tcpdump` 沒有抓包權限。
5. ScanNet++ 的 6 個場景最多 793,734 顆 Gaussian，做不出 N = 1 M 和 2 M 的子集。
   N 曲線的四個點改用本機已有的 Mip-NeRF 360 garden（gsplat 訓練，6,606,688 顆，標準 INRIA 格式）取子集，ScanNet++ `0d2ee665be` 另成一條兩點的曲線，第 3 節說明。
6. WebSplatter 本機 clone 的 HEAD 是 `16bb06e`，來自 GitHub；`reading_notes.md:2095` 說 GitHub 上找不到的是 Zenodo 記錄的 `beb9532`，不是 `16bb06e`。
   PINS.tsv 釘 `16bb06e`，另外在 repo 裡存一份 git bundle 當備援（第 1.4 節）。
7. Windows 那台（172.16.10.104）不回 ping，但 ARP 表看得到它在線上；M0 在那台的量測是可選項，能連上才做。
8. 工作站 GPU 現在被別人的程序用著（72% 使用率）；PLAN 第 8.4 節的剔除規則從第一個 run 就生效，而且要在 run 進行中持續取樣，不能只在啟動前看一眼（第 6.4 節）。

### 0.2 這份文件對 PLAN.md 做的具體化和偏離（每一條都需要你知道）

9. 線上紀錄最後三個 fp16 放的是 SH 0 階係數 `f_dc`，不是 PLAN 第 5 節寫的 RGB（第 4.2 節）。
   理由：渲染器儲存和計算用的就是 `f_dc`（`preprocess_f16.wgsl:82,119`），上游讀到 RGB 也是先轉成 `f_dc`（`ply-worker.ts:92-94`）。
   後果：若堅持 RGB 上線，client 要對每筆算 `(c - 0.5) / 0.28209`，而且 >1 的顏色會先被夾掉，V12 的畫質數字會把這個損失一起量進去。
   這條需要你確認。
10. M1 的「直接讀檔版」具體化成「同一個 `.bin` 紀錄檔用同一個 Worker 解碼、一次全部套用（`src=file`）」，不是上游的 PLY 載入路徑（第 8 節開頭）。
    理由：GPU 存的是 covariance 不是 scale 和 quat，client 必須從 fp16 的 scale/quat 算 cov，和上游用 f64 算的結果不會逐位元相同。
    fp16 線上格式相對上游 f32 PLY 載入的代價另外以 V12 回報。
    這條需要你確認。
11. H0 的「30 秒內吸收落後不增加」在這裡的操作型定義是：每階最後 25 秒的 `lag`（已產生減已寫入的筆數）增量 ≤ `rate / 60`，也就是允許一幀份量的離散誤差（第 9.1 節 V16）。
12. PLAN M0 說「兩台都不合格就停下來討論」；這台 Linux 工作站是目前唯一能自動量的候選，所以規則寫成：V5 的 A/A 差異 > 10% 時 M1 不開始，直到你選定量測 client。
13. M0 的幀間隔不是用上游的 eval=2 量（它沒有 rAF，見第 7 節 V2），而是 runner 注入到每個頁面的同一段 10 行 rAF 紀錄器（第 6.4 節），上游頁面和我們的 client 都用它，所以 M0 的 null 和之後的 run 是同一個儀器。
14. M2 的「旁邊開一個佔用 WebGPU 的分頁」照 PLAN 做成同一個 Chrome 裡的第二個分頁（CDP `Target.createTarget`），和主頁共用同一個 GPU 程序；不用第二個 Chrome 程序，因為那會多一個 GPU 程序，排程路徑不同（第 9.1 節）。
15. M2 的 k 網格照 PLAN 用 {1k, 4k, 16k, 64k, 256k}，不省略 1k。

## 1. Repo 配置

### 1.1 樹狀圖

```
virtualization_final_project/
├── LOG.md                      # V 編號、只能往後加、最底下是「已推翻的說法」索引（第 11 節）
├── README.md                   # 一句話的發現、適用範圍、怎麼重現一個數字
├── PLAN.md                     # 研究計畫（已有）
├── pyproject.toml              # 讓 src/gsstream 可以 import，uv 管理
├── uv.lock
├── .gitignore
├── docs/                       # 已有：direction_review.md, reading_notes.md, source_material.md, implementation_plan.md
├── src/gsstream/               # 安裝進 venv 的 Python 套件，所有可重用程式碼
│   ├── paths.py                # ROOT、DATA、OUTPUTS、CLONE，全部可用環境變數 GSSTREAM_* 覆寫
│   ├── wire.py                 # 32 B 紀錄的 numpy dtype、32 B 訊息標頭的 pack/unpack、pack 子命令
│   ├── ply.py                  # 從 GS_playground 複製的 load_ply/save_ply（numpy 版，不用 torch）、subset 子命令
│   ├── traces/                 # 序列產生器：tstatic.py reveal.py refine.py churn.py beacon.py cameras.py format.py
│   ├── server/                 # __main__.py（asyncio + websockets）、session.py、replay.py
│   ├── runner/                 # chrome.py（CDP）、run.py、httpd.py（靜態檔案伺服器）、gpucheck.py、nvsmi.py、raf_logger.js
│   └── report/                 # quantiles.py（唯一的 p50/p99 定義）、stalls.py、compare_png.py、sh_psnr.py、gpu_idle.py、costmodel.py、predict_rmax.py、h0_verdict.py、m3_verdict.py、film.py
├── client/                     # WebSplatter 的延伸套件（第 1.3 節），tracked
│   ├── package.json  pnpm-lock.yaml  vite.config.ts  tsconfig.json
│   ├── index.html              # 一個 canvas、一個隱藏的 #log、一個 #status
│   ├── burner.html             # M2 的「鄰居分頁」WebGPU 負載頁
│   ├── beacon-ref.html         # M3 攝影機驗證用的參考頁（只畫 DOM 方塊）
│   └── src/
│       ├── types.d.ts          # 照抄 clone 的 src/types.d.ts：讓 tsc 認得 .wgsl 的 import
│       ├── main.ts             # requestAdapter/requestDevice，照 clone 的 main.ts 但把 adapter.info 存到 window.__gpu_info
│       ├── app.ts              # 讀 URL 參數、建 pool、起 Worker、rAF 迴圈、eval 模式分派
│       ├── eval.ts             # eval=1 截圖（從 splat-app.ts:140-275 複製並改 alpha）和 eval=2/3 的量測迴圈
│       ├── stream/slot-pool.ts # 固定容量 slot 池、id→slot Map、free list
│       ├── stream/apply.ts     # 每幀最多 k 筆、合併連續 slot、queue.writeBuffer
│       ├── stream/stage-log.ts # 每則訊息的時間戳列、beacon 的逐筆時間戳
│       ├── stream/injector.ts  # M2 的頁內注入器
│       ├── stream/selftest.ts  # ?selftest=1 跑的自我檢查，結果由 runner 寫進 stamp.json
│       └── workers/stream-worker.ts  # 擁有 WebSocket、解碼 32 B 紀錄、算 cov
├── experiments/
│   ├── README.md               # 索引：條目 → study → 結果
│   ├── m0_static_null/         # README.md, grid.json, run_V1_gpucheck.sh, run_V2_eval2_n_sweep.sh, ...
│   ├── m1_static_stream/       # README.md, grid.json, run_V7_*.sh ...
│   ├── m2_absorb_capacity/
│   ├── m3_trace_replay/
│   └── m4_netem/               # 第 9.4 節只先寫 README 和 docker/
├── results/                    # TRACKED
│   ├── collect.py              # 讀 outputs/*/V*/metrics.json，產生 tables/，--check 模式
│   ├── tables/*.csv            # 機器產生，不手改
│   └── figures/                # make_m0.py ... 和產出的 png（nature-figure skill）
├── tools/
│   ├── setup/                  # env.sh, env.local.example.sh, python_env.sh, client_env.sh, clone_websplatter.sh, export_patches.sh, stamp.sh, nvsmi_logger.sh
│   └── fetch/                  # fetch_scannetpp.sh, link_garden.sh, check_cameras.sh
├── third_party/
│   ├── README.md               # 怎麼重建 clone、clone 裡允許哪些未追蹤狀態、永遠不能呼叫 requestReorder
│   ├── PINS.tsv                # repo  commit  url  note
│   ├── websplatter-16bb06e.bundle   # git bundle 備援，約數 MB
│   ├── patches/                # 目前是空的（README 說明為什麼），export_patches.sh 會寫到這裡
│   └── clones/websplatter/     # 不進 git；clone_websplatter.sh 重建
├── tests/                      # test_wire.py, test_traces.py
├── probes/                     # 一次性的探測腳本，不進管線
├── data/                       # 不進 git，只有 README.md 進
│   ├── README.md               # 來源、sha256、授權（ScanNet++ 條款）
│   ├── scannetpp_3dgs/<scene>/point_cloud.ply + archive_receipt.json
│   ├── subsets/<scene>_n=<N>_seed=<s>.ply
│   ├── records/<scene>_n=<N>_seed=<s>_sh0.bin + .json
│   ├── cameras/<scene>_<mode>_k=10.json + <scene>_<mode>_k=10.coverage.json
│   └── traces/<trace_name>/{trace.json, ticks.bin, records.bin}
└── outputs/                    # 不進 git，只有 README.md 進
    ├── README.md
    ├── gpu_log/                # nvidia-smi 一週紀錄的原始 csv（沒有 metrics.json，數字在 V6 的 run 資料夾）
    └── <study>/{V<n>_<slug>[_params]_rep=<r>, smoke_<slug>}/{stamp.json, metrics.json, ...}
```

### 1.2 命名規則

- `src/gsstream/` 是套件名；所有 Python 程式用 `python -m gsstream.<module>` 執行，不用 `sys.path`。
- 一個 study 一個資料夾（`experiments/m0_static_null/`），一個 LOG 條目一個 runner（`run_V2_eval2_n_sweep.sh`）。
  資料夾名是 slug 永不改；`V` 編號只在 runner 檔名、run 資料夾名和 LOG 裡。
- 每個 study 有一個 `grid.json`：跑之前就寫好的網格（arm × 參數 × 重複次數）、每個條目的成功條件和預測；`collect.py` 用它補空格，LOG 條目引用它。
- **每個 V 條目的每一個數字都住在恰好一個 `outputs/<study>/V<n>_<slug>[_params]/metrics.json`，深度固定兩層。**
  比較型的條目（兩個 run 的 PNG 相減）、預測型的條目（只算不跑瀏覽器）、判定型的條目（套規則）都一樣有自己的 runner 和 run 資料夾，`collect.py` 才看得到它們。
- 煙霧測試（shake down runner 用的）寫到 `outputs/<study>/smoke_<slug>/`，`collect.py` 只 glob `V*`，所以它們永遠不會被引用。
- 衍生資料把參數寫進檔名：`garden_n=1000000_seed=0.ply`，不是 `garden_v2.ply`。
- run 資料夾名帶語意，不用 uuid 或純時間戳；一個 arm 一種拼法（`k=inf` 和 `k=16000`，不另外叫 `unlimited`）。
  本計畫會出現的拼法全列在這裡，`grid.json` 用同一套：

| 條目 | run 資料夾名 |
| --- | --- |
| V1 | `V1_gpucheck` |
| V2 | `V2_eval2_scene={garden,0d2ee665be}_n=<N>_rep=<r>`；備援 arm `V2_eval2_scene=garden_n=500000_f16feat=0_rep=<r>`、`V2_eval2_scene=garden_n=500000_prec=f32_rep=<r>` |
| V3 | `V3_eval3_scene=<s>_n=<N>_rep=<r>` |
| V4 | `V4_rafpos_scene=garden_n=6606688_win=3840x2160_rep=<r>`；對照 `V4_rafpos_scene=garden_n=6606688_win=3840x2160_eval2` |
| V5 | `V5_raf_scene=<s>_n=<N>_rep=<r>` |
| V6 | `V6_gpu_idle` |
| V7 | `V7_poolnull_scene=0d2ee665be_n=527437_cap=1048576_rep=<r>` |
| V8 | `V8_pixel_aa_rep={0,1}`、`V8_pixel_aa_compare` |
| V9 | `V9_pixel_identity_src={file,ws}`、`V9_pixel_identity_compare` |
| V10 | `V10_karms_k={inf,16000}_rep=<r>` |
| V11 | `V11_sh0_psnr` |
| V12 | `V12_fp16_cost_src={file,ply}`、`V12_fp16_cost_compare` |
| V13 | `V13_cost_k=<k>_loc={contig,random}_rep=<r>` |
| V14 | `V14_predict` |
| V15 | `V15_burner_iters=<I>_rep=<r>` |
| V16-V18 | `V16_rmax_state=unthrottled_k=<k>_rep=<r>`、`V17_rmax_state=cpu4x_k=<k>_rep=<r>`、`V18_rmax_state=burner_k=<k>_rep=<r>` |
| V19 | `V19_h0_verdict` |
| V20 | `V20_traces` |
| V21 | `V21_static_stages` |
| V22 | `V22_clock` |
| V23 | `V23_fifo_trace=<name>_k={inf,<k>}_rep=<r>` |
| V24 | `V24_camera_trace=refine_rep=<r>` |
| V25 | `V25_m3_verdict` |

### 1.3 WebSplatter 怎麼接：延伸套件，零 patch

directory-layout skill 的規則是：上游 clone 釘住 commit 且不進 git，本地修改以 patch 存放，新增的功能放在自己的套件裡 import 上游，fork 只帶「改到上游既有行」的 diff。
這個專案的決定是：**`client/` 是一個獨立的 Vite 專案（tracked），用路徑別名 `@ws` 直接 import clone 裡的 `src/*.ts`；上游一行不改。**

理由如下。

1. 學生寫的程式（slot 池、apply、Worker、stage log）是這個專案的核心，必須是正常的 tracked 原始碼，能 diff、能 review、能一週一週累積。
2. 上游的 `src/splat-app.ts` 是一個 805 行的 `init()` 函數，把 Tweakpane 面板、檔案選取、快速連結、除錯互動和渲染迴圈混在一起。
   我們不需要那些 UI，自己寫一個約 150 行的 `app.ts` 比在它裡面插程式乾淨。
3. 渲染器本體（`gaussian-renderer.ts`、`camera.ts`、`ply-loader.ts`、`render-settings.ts`、所有 `.wgsl`）直接 import，這正是 skill 說的「extend upstream by importing it and overriding」。
4. 上游唯一值得重用的量測碼是 `splat-app.ts:140-261` 的 `evalRenderOneView`（渲染一個視角、把 rgba16float 讀回來）和 `:264-275` 的 `pixelsToPngDataUrl`。
   它們不是 `export`，所以要嘛 patch 上游加兩個 `export`，要嘛複製約 130 行到 `client/src/eval.ts`。
   選複製，原因有三：import `splat-app.ts` 會把 tweakpane、`@spz-loader/core`、gltf-loader、另外兩個渲染器都拉進 bundle；上游在 `:218-221` 把累積的 alpha 原樣寫進 u8，再經 2D canvas 匯出時會被預乘又反預乘，alpha < 255 的像素會被捨入，V12 的 PSNR 需要一份把 alpha 強制設 255 的版本；零 patch 讓每個 run 的 `upstream.patch` 永遠是空的，`drift` 檢查最乾淨。
   `eval.ts` 的檔頭註明來源行號和兩處改動（alpha = 255；另外把原始 u8 陣列掛在 `window.__eval_progress.raw`）。
5. `maybeReorderAfterSubmit`（`gaussian-renderer.ts:1237-1313`）會把 GPU buffer 整個重排，之後 slot 編號就對不上 id；在 `16bb06e` 的渲染迴圈沒有人呼叫它，唯一的呼叫者是 `splat-app.ts:727` 的 Tweakpane 按鈕，而我們不用 `splat-app.ts`。
   刪它要動約 95 行上游、而且 `tile-based-renderer-global-sort.ts:798,850` 還有一份一樣的，沒有數字支持這個 patch。
   改成在 `third_party/README.md` 寫一行：「永遠不要呼叫 `requestReorder`，它會重排 GPU buffer，slot 和 id 的對應會壞掉」。
6. run 的輸出永遠寫在 `outputs/`；clone 裡只允許 `pnpm install` 的建置狀態（第 1.4 節）。

`vite.config.ts` 的關鍵設定：

```ts
import { fileURLToPath } from 'node:url';
import rawPlugin from 'vite-raw-plugin';
const WS = fileURLToPath(new URL('../third_party/clones/websplatter/src', import.meta.url));
export default {
  base: './',
  resolve: { alias: { '@ws': WS }, dedupe: ['wgpu-matrix'] },   // dedupe：clone 和 client 各有一份 wgpu-matrix，只打包一份
  server: { fs: { allow: ['..'] },
            headers: { 'Cross-Origin-Opener-Policy': 'same-origin',        // COOP + COEP：跨來源隔離，上游 PLY worker 才能用 SharedArrayBuffer
                       'Cross-Origin-Embedder-Policy': 'require-corp' } },
  plugins: [rawPlugin({ fileRegex: /\.wgsl$/ })],
  build: { outDir: 'dist' },
};
```

`tsconfig.json` 整份照抄 clone 的（`strict: false`、`lib` 含 `ESNext.Float16`、`moduleResolution: "bundler"`、`allowImportingTsExtensions: true`、`types: ["@webgpu/types"]`），再加 `paths: { "@ws/*": ["../third_party/clones/websplatter/src/*"] }`，`include` 加 `src/**/*.ts`（含 `types.d.ts`）。
`moduleResolution: bundler` 和 `allowImportingTsExtensions` 是必要的，因為上游的 import 寫 `./ply-loader.ts` 這種帶副檔名的路徑；`types.d.ts` 讓 tsc 認得 `import x from './a.wgsl'`。
`strict: false` 是因為 import 進來的上游檔案會一起被 tsc 檢查，開 strict 會先卡在上游的程式。

上游的 `@ws/*.ts` 檔案 import 的裸模組名（`wgpu-matrix`、`tweakpane`）由 Vite 從 clone 的目錄往上找 `node_modules`，不是從 `client/node_modules`，所以 clone 裡必須 `pnpm install` 過。

`index.html` 必須有 `<div id="log" hidden>`，因為 `@ws/utils/simple-console.ts:1` 在模組載入時就抓 `#log`，`log()` 會對它 `appendChild`；`GaussianRenderer` 的建構子本身就呼叫 `log()`（`gaussian-renderer.ts:265-266`），所以連不載 PLY 也需要它。
也要有 `<canvas id="webgpu-canvas">`，因為 `gaussian-renderer.ts:1160` 用 `document.querySelector('canvas')` 讀寬高。

### 1.4 third_party

`PINS.tsv`（tab 分隔）：

```
repo	commit	url	note
websplatter	16bb06e86b8fb12193d4bb4cd2b0b3d4f3ab62be	https://github.com/websplatter/WebSplatter	MIT (c) 2024 shrekshao; ACM MM 2026; Zenodo 21798977 names beb9532 which GitHub lacks; bundle beside this file is the fallback
```

`tools/setup/clone_websplatter.sh`（可重複執行）：

```bash
#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
read -r _ COMMIT URL _ < <(sed -n 2p "$ROOT/third_party/PINS.tsv")
DST="$ROOT/third_party/clones/websplatter"
if [ ! -d "$DST/.git" ]; then
  git clone "$URL" "$DST" || git clone "$ROOT/third_party/websplatter-16bb06e.bundle" "$DST"
fi
git -C "$DST" cat-file -e "$COMMIT" || git -C "$DST" fetch "$ROOT/third_party/websplatter-16bb06e.bundle"   # GitHub 歷史被改寫時從 bundle 補
if [ -n "$(git -C "$DST" status --porcelain --untracked-files=no)" ]; then
  echo "clone has tracked modifications; run tools/setup/export_patches.sh first" >&2; exit 1
fi
git -C "$DST" checkout --detach "$COMMIT" && git -C "$DST" reset --hard
for p in "$ROOT"/third_party/patches/*.patch; do
  git -C "$DST" apply "$p" || { echo "patch failed: $p" >&2; exit 1; }
done
echo pnpm-lock.yaml >> "$DST/.git/info/exclude"
git -C "$DST" status --short --untracked-files=no
```

patch 用 `git apply` 不用 `git am`，所以 HEAD 永遠是 `16bb06e`，`git diff` 就是「patch 加上任何未整理的改動」，第 6.6 節的出處戳記會把它存下來。
`tools/setup/export_patches.sh` 把 clone 的 `git diff` 依檔案切回 `patches/`，學生改了上游就跑它；目前 `patches/` 是空的。
bundle 的製作（一次）：`git -C papers/agent_workdir/ws/repo bundle create third_party/websplatter-16bb06e.bundle 16bb06e`，不碰原 clone。

`third_party/README.md` 要寫的四件事：怎麼重建；clone 裡允許的未追蹤狀態是 `node_modules/`（含 Vite 的依賴快取 `node_modules/.vite`）和 `pnpm-lock.yaml`，它們是 `@ws` 裸 import 能解析的前提，`git diff` 不受影響；`stamp.sh` 會把 `git status --porcelain --untracked-files=no` 存進 run 資料夾，所以任何追蹤檔的改動都會被記錄；永遠不要呼叫 `requestReorder`。

### 1.5 .gitignore 和本機設定

```
# 根目錄的雜物，但 results/ 刻意 tracked
/*.png
/*.csv
!results/
!results/**
# 整棵樹用前綴排除
outputs/*
!outputs/README.md
data/*
!data/README.md
third_party/clones/
papers/
client/node_modules/
client/dist/
.venv/
tools/setup/env.local.sh
debug*
scratch*
```

`.claude/rules/research-discipline.md` 這個 symlink 已經在 `.git/info/exclude`，不進 `.gitignore`。
`tools/setup/env.local.sh` 是唯一含機器專屬路徑的檔案，不進 git；第 2 節說明。

### 1.6 LOG 前綴

LOG 前綴是 `V`，編號規則和條目模板在第 11 節；`LOG.md` 現在還不存在，V1 跑完才建立。

## 2. 環境設定

所有進 git 的腳本只用 `$ROOT`、`~` 或環境變數，不出現任何人的家目錄、帳號或掛載點。
`tools/setup/env.sh`（tracked）不含任何機器專屬的值，它只做兩件事：載入 `env.local.sh`，然後對每個必要變數做「沒設就大聲失敗」：

```bash
# tools/setup/env.sh  (source me)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export ROOT
[ -f "$ROOT/tools/setup/env.local.sh" ] || { echo "copy tools/setup/env.local.example.sh to env.local.sh and edit it" >&2; return 1 2>/dev/null || exit 1; }
source "$ROOT/tools/setup/env.local.sh"
: "${GSSTREAM_CHROME:?set in env.local.sh}"            # Chrome 執行檔
: "${GSSTREAM_SCANNETPP_SRC:?set in env.local.sh}"     # ScanNet++ 3DGS 的來源目錄
: "${GSSTREAM_GARDEN_PLY:?set in env.local.sh}"        # garden 的 point_cloud_29999.ply
: "${GSSTREAM_PUFFIN_PY:?set in env.local.sh}"         # 有 torch + gsplat 的 python（只給 V11、check_cameras 用）
: "${GS_PLAYGROUND_SRC:?set in env.local.sh}"          # gs_playground 套件的 src
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:?set in env.local.sh}"
```

`tools/setup/env.local.example.sh`（tracked）放這台機器現在的值當範例，學生複製成 `env.local.sh` 再改；裡面的 `$HOME/...`、`/mnt/...`、`/usr/share/...` 都只存在於不進 git 的檔案。
每週結束時 `git grep -n -E '/home/|/mnt/|/usr/share|ms-playwright|\.nvm|intern_' -- ':!docs/' ':!tools/setup/env.local.example.sh'` 必須是空的（文件和範例檔可以提到機器，程式和設定不行）。

### 2.1 Python（uv）

`uv` 是 Python 的套件和虛擬環境管理工具，這裡用它取代 pip。
`hatchling` 是 uv 安裝 `src/gsstream` 時用的建置後端（把原始碼目錄變成可 import 的套件的工具）。
`pyproject.toml`：

```toml
[project]
name = "gsstream"
version = "0.0.1"
requires-python = ">=3.12"
dependencies = ["numpy>=2.2", "plyfile>=1.1", "websockets>=16"]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gsstream"]

[tool.uv]
package = true
```

指令：

```bash
cd "$ROOT" && uv sync          # 建 .venv、裝三個依賴、pytest 和 gsstream 本身（editable）；第一次需要網路，hatchling 不在 uv 快取裡
uv run python -c "import gsstream, websockets, numpy; print(websockets.__version__)"
uv run pytest tests/
```

gsplat 參考渲染（V11 的 PSNR 和相機檢查）不裝進這個 venv，而是用 `$GSSTREAM_PUFFIN_PY` 跑，`PYTHONPATH=$GS_PLAYGROUND_SRC:$ROOT/src`；那個環境已經有 torch 2.7、gsplat 1.5.3 和 `gs_playground` 套件。

### 2.2 Client（pnpm）

Node 22.23.1 和 pnpm 12.8.1 已經在這台機器上，`env.local.sh` 負責把它們加進 PATH。
第一次安裝需要網路，而且這是加入外部依賴，要先得到你的同意（PLAN 第 12 節第 2 項）。

```bash
source tools/setup/env.sh && tools/setup/clone_websplatter.sh
cd third_party/clones/websplatter && pnpm import && pnpm install --frozen-lockfile   # 從 package-lock.json 產生 pnpm-lock.yaml 再裝；這是 clone 裡唯一允許的寫入
cd "$ROOT/client" && pnpm install                                                    # 我們自己的 vite/typescript/vite-raw-plugin/@webgpu/types/wgpu-matrix
pnpm build                                                                           # 產生 client/dist/
```

`client/package.json` 把版本釘在 clone 的 lockfile 用的版本：vite 4.5.14、typescript 5.8.3、vite-raw-plugin 1.0.2、@webgpu/types 0.1.64、wgpu-matrix 3.4.0。
開發時 `pnpm dev`（port 5173）；量測時用 `pnpm build` 的 `dist/`，由 `gsstream.runner.httpd` 提供（第 6.4 節），這樣每個 run 的 client 是一份固定的 build。

### 2.3 Chrome 和 WebGPU

Chrome 來源只有一個：Playwright 的 Chrome for Testing 153（`$GSSTREAM_CHROME`）。
啟動旗標（runner 的 `chrome.py` 固定用這組）：

```bash
"$GSSTREAM_CHROME" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --enable-webgpu-developer-features \
  --enable-features=Vulkan --use-angle=vulkan --ignore-gpu-blocklist \
  --disable-vulkan-surface --disable-dawn-features=timestamp_quantization \
  --remote-debugging-port=0 --user-data-dir="$RUN/profile" --window-size=1920,1080
```

每個旗標的意思：`--headless=new` 是新式無頭模式，有 GPU 程序和 rAF；`--no-sandbox` 因為 AppArmor；`--enable-unsafe-webgpu` 在 Linux 開 WebGPU；`--enable-webgpu-developer-features` 讓 `GPUAdapterInfo.description` 有內容（預設是空字串，只有 `vendor` 和 `architecture` 有值）；`--enable-features=Vulkan --use-angle=vulkan` 讓 GPU 程序走 Vulkan；`--ignore-gpu-blocklist` 跳過 Linux 的黑名單；`--disable-vulkan-surface` 無頭模式不需要視窗表面；`--disable-dawn-features=timestamp_quantization` 關掉 WebGPU 時間戳的 100 µs 量化（eval=3 要）；`--remote-debugging-port=0` 讓 Chrome 自己挑一個 CDP port 並印在 stderr。
`VK_ICD_FILENAMES` 指向 NVIDIA 的 ICD，意圖是避免 Chrome 挑到自己附的 SwiftShader（軟體渲染）。

這一組裡有四件事在 V1 跑之前都是猜測，V1 和 V3、V4 會逐一驗證：`--enable-features=Vulkan --use-angle=vulkan` 對 Dawn 是必要的還是無害的（V1：拿掉它們當備援路線）；Chrome 自帶的 Vulkan loader 會不會理 `VK_ICD_FILENAMES`（V1：adapter 的 vendor）；`timestamp_quantization` 旗標有沒有生效（V3 的 `quantized_fraction`）；無頭模式的 rAF 在 GPU 負載下會不會變慢（V4 正控制）。

驗證 WebGPU 真的用到 NVIDIA（這就是 M0 的第 1 個量測，V1）：

1. `gpucheck.py` 在背景執行緒起一個標準函式庫的 `http.server`，服務一個只有 `<title>` 的 `index.html`，runner 導向 `http://127.0.0.1:<port>/`，不用 `about:blank`，確保是安全情境（secure context，WebGPU 只在 https 或 localhost 提供）。
2. `Runtime.evaluate`（`awaitPromise`、`returnByValue`）下面這段；`GPUAdapterInfo` 的欄位是原型上的存取器，`JSON.stringify(a.info)` 會得到 `{}`，所以要逐欄位列出：

```js
navigator.gpu.requestAdapter({powerPreference:'high-performance'}).then(a => JSON.stringify(a ? {
  vendor: a.info.vendor, architecture: a.info.architecture, device: a.info.device, description: a.info.description,
  isFallbackAdapter: a.info.isFallbackAdapter ?? a.isFallbackAdapter,
  features: [...a.features],
  limits: {maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize}} : null))
```

3. 對瀏覽器層級的 CDP 連線（Chrome 在 stderr 印的 `DevTools listening on ws://.../devtools/browser/<id>`）呼叫 `SystemInfo.getInfo`，它回傳 `gpu.devices[]`（`vendorString`、`deviceString`、`driverVersion`）和 `gpu.featureStatus`（`chrome://gpu` 那張表的 JSON 版）；整個回傳原樣存進 `stamp.json.gpu_info`。
   `chrome://gpu` 頁面是 shadow DOM，`innerText` 通常是空的，所以不刮那個頁面。
4. 定義 `is_hardware = (not isFallbackAdapter) and ('swiftshader' not in description.lower()) and (vendor not in ('', 'google'))`。
5. 成功條件（V1）：`vendor == "nvidia"` 且 `gpu.featureStatus` 裡 WebGPU 那一項是硬體加速（Chrome 對這個鍵的確切拼法以 V1 實際回傳為準，V1 的 LOG 條目會把它抄下來）。

有真正螢幕的路線（之後需要時）：有人在主機登入 xfce 後，去掉 `--headless=new` 和 `--disable-vulkan-surface`，加 `DISPLAY=:0`，stamp 記 `vsync_mode=display`。
Windows 那台：在它上面手動啟動 `chrome.exe --remote-debugging-port=9222 --remote-debugging-address=0.0.0.0 --enable-unsafe-webgpu --enable-webgpu-developer-features`，runner 用 `--cdp-url http://172.16.10.104:9222` 接上去而不自己啟動；資料和 Server 仍在工作站。

### 2.4 一次性檢查

```bash
uv run python -m gsstream.runner.gpucheck --out outputs/m0_static_null/V1_gpucheck   # 寫 stamp.json 和 metrics.json
uv run pytest tests/                                                                # wire 和 trace 的單元測試
```

## 3. 資料

### 3.1 來源和複製

來源：`$GSSTREAM_SCANNETPP_SRC/<scene>/point_cloud.ply`，6 個場景，每個旁邊有 `archive_receipt.json`（含 `sha256` 和顆數）。
這些是用 opensplat3d 從 ScanNet++ v2 訓練出來的，ScanNet++ 的使用條款（非商業研究、不得再散布）適用於它們和它們的渲染圖；`data/README.md` 要寫明，repo 不得含任何 `.ply` 或渲染圖。

`tools/fetch/fetch_scannetpp.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$ROOT/tools/setup/env.sh"
for s in 0d2ee665be 13c3e046d7 21d970d8de 578511c8a9 a24f64f7fb d755b3d9d8; do
  d="$ROOT/data/scannetpp_3dgs/$s"; mkdir -p "$d"
  cp --update=none "$GSSTREAM_SCANNETPP_SRC/$s/archive_receipt.json" "$d/"
  if [ ! -f "$d/point_cloud.ply" ]; then
    cp "$GSSTREAM_SCANNETPP_SRC/$s/point_cloud.ply" "$d/point_cloud.ply.part" && mv "$d/point_cloud.ply.part" "$d/point_cloud.ply"
  fi
  want=$(python3 -c "import json;print(json.load(open('$d/archive_receipt.json'))['sha256'])")
  got=$(sha256sum "$d/point_cloud.ply" | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "$s ok $got" || { echo "$s SHA MISMATCH"; exit 1; }
done | tee "$ROOT/data/scannetpp_3dgs/SHA256SUMS.txt"
```

6 個檔共 1.12 GB，本機硬碟有 1.3 TB，全部複製。
CIFS 掛載是 soft mount，複製中斷只會留下 `.part`，重跑就會重新複製，不會把半個檔當完成品。
`tools/fetch/link_garden.sh` 用相對路徑把 `$GSSTREAM_GARDEN_PLY` symlink 到 `data/garden/point_cloud.ply`，同樣記 sha256 到 `data/garden/SHA256SUMS.txt`。

### 3.2 場景分工

| 用途 | 場景 | 顆數 | 理由 |
| --- | --- | --- | --- |
| M0 的 N 曲線（全部四點） | garden | 6.6 M → 取 0.25/0.5/1/2 M | 唯一一個夠大的本機場景 |
| M0 的正控制（V4） | garden 全量 | 6,606,688 | 已知會超過幀預算的負載 |
| M0 的第二條曲線、M1、M2、M3 主場景 | ScanNet++ `0d2ee665be` | 527,437 | 最小的 ScanNet++ 場景，載入快 |
| M3 和 M5 的開發序列（dev trace） | ScanNet++ `21d970d8de` | 666,925 | 另一個場景，避免在測試場景上調參（PLAN 8.4） |

子集用均勻隨機抽樣（固定 seed），不是空間裁切，這樣每個視角看到的場景範圍不變，只有密度變。
`0d2ee665be` 的 0.25 M 子集和全量 527 k 就是 ScanNet++ 那條曲線的兩個點。

### 3.3 子集、32 B 紀錄檔和相機

```bash
uv run python -m gsstream.ply subset --ply data/garden/point_cloud.ply --n 250000 --seed 0 \
    --out data/subsets/garden_n=250000_seed=0.ply          # 標準 INRIA 格式，含 SH 3 階，不含 feat_*
uv run python -m gsstream.wire pack --ply data/subsets/garden_n=250000_seed=0.ply \
    --out data/records/garden_n=250000_seed=0_sh0.bin      # 32 B 紀錄，id 從 16 開始（0-15 保留給 beacon）
```

`pack` 做的事（`src/gsstream/wire.py`，學生手寫）：

1. 讀 PLY（`gsstream.ply.load_ply`，從 GS_playground 的 `ply.py:57-77` 複製成 numpy 版，約 40 行；未知屬性如 `feat_0..7` 自然被略過）。
2. 啟動函數：`opacity = sigmoid(opacity_logit)`、`scale = exp(log_scale)`、`quat = quat / |quat|`（wxyz 順序不變）、顏色用 `f_dc` 原值。
3. 丟掉 `opacity <= 1/255` 的顆（渲染器本來就不畫它們，而且 opacity 0 在線上是刪除記號）；丟掉的數量寫進旁邊的 `<name>.json`。
4. 轉 fp16（numpy `astype(np.float16)`，四捨五入到最近偶數）。
   numpy 轉 fp16 時 |x| > 65504 會悄悄變成 inf，|x| < 6.1e-5 會變成次正常數，所以轉完要 `assert np.isfinite(packed).all()`，並把 `n_overflow`（轉換前 |x| > 65504 的值數）和 `n_subnormal_scale`（scale < 6.1e-5 的值數）寫進 sidecar。
   另外渲染器存的是 cov = scale² 的 fp16，scale 小於約 2.4e-4 的顆在 GPU 上 cov 會是 0；這是上游本來就有的極限，不是我們引入的，但要知道。
5. 輸出檔旁邊寫 `<name>.json`：顆數、sha256、來源 PLY 的 sha256、丟掉的顆數、fp16 轉換的最大相對誤差、`n_overflow`、`n_subnormal_scale`。

相機：

```bash
uv run python -m gsstream.traces.cameras --ply data/subsets/0d2ee665be_n=527437_seed=0.ply --mode lookaround --k 10 \
    --width 1280 --height 720 --fov-deg 60 --out data/cameras/0d2ee665be_lookaround_k=10.json
tools/fetch/check_cameras.sh data/cameras/0d2ee665be_lookaround_k=10.json   # 用 gsplat 渲染 preset 0，看圖，寫 coverage.json
```

`cameras.json` 用 INRIA 3DGS 的格式（`id, img_name, width, height, position, rotation(3x3), fx, fy`），WebSplatter 的 `camera.ts:73-94` 和 GS_playground 的 `render.py:30-54` 都讀這個格式，所以同一個檔同時給瀏覽器和 gsplat 用。

座標慣例必須寫死，因為兩個渲染器會互相「同意」一個看錯方向的相機，V8-V12 抓不到這種錯：
`rotation` 是 camera-to-world 的 3x3 矩陣（c2w），OpenCV 慣例，第 0 欄是相機的右方向、第 1 欄是相機的下方向（也就是 -up）、第 2 欄是視線方向（forward）；`position` 是相機中心的世界座標。
`camera.ts` 把 `rotation` 轉置後乘上 `T(-position)` 得到 view 矩陣，投影時翻轉 Y（`camera.ts:49`），GS_playground 的 `load_cameras` 用同一個假設。
產生的方法：給定 `forward` 和 `up`，`right = normalize(cross(forward, up))`，`down = cross(forward, right)`，`c2w = [right, down, forward]` 按欄擺。
`lookaround`：相機放在場景 5-95 百分位 bbox 的中心，`up` 是 bbox 最短軸的正方向，`forward` 在與 `up` 垂直的平面上每 36 度一個，共 10 個視角；室內場景用這個。
`orbit`：繞 bbox 中心、半徑 0.8 倍 bbox 半徑的圓，`forward` 指向 bbox 中心，10 個視角；garden 用這個。
`fx = fy = width / (2 tan(fov/2))`。
beacon 的中心 = `position + 1.0 × c2w[:, 2]`（preset 0 正前方 1 m，第 5.2 節）。

`check_cameras.sh` 用 `$GSSTREAM_PUFFIN_PY` 跑 `gsstream.report.sh_psnr --preview-only`：渲染 preset 0（gsplat，DC only）存 `data/cameras/<name>_preset0.png`，並寫 `<name>.coverage.json`，裡面是 `coverage` = 非背景像素（alpha > 0.5）的比例。
這是資料製作的 sidecar，不是 LOG 條目；要求 `coverage ≥ 0.5`，而且學生要親眼看那張 PNG 確認不是倒的、不是看牆。

## 4. 傳輸格式（wire protocol）

一切都是 little-endian（x86 和 ARM 的原生位元組序，JS 的 `DataView` 要明確傳 `true`）。
fp16 是 IEEE 754 binary16：1 個符號位、5 個指數位、10 個尾數位，最大 65504，正常數最小 6.1e-5，次正常數到 6e-8。
在 10 m 處一個 fp16 的間距是 7.8 mm；WebSplatter 本來就以 fp16 儲存位置（`ply_precision=f16` 預設），所以這不是我們額外引入的誤差，V12 會量它。
Python 端用 numpy 的 `float16`（四捨五入到最近偶數），JS 端用 `Float16Array`（Chrome ≥ 135）。

### 4.1 訊息標頭（32 bytes）

一個 WebSocket 二進位訊息 = 1 個標頭 + `count` 筆紀錄，最大 65,536 bytes，所以 `count ≤ 2047`。

| 位移 | 型別 | 欄位 | 說明 |
| --- | --- | --- | --- |
| 0 | u8 | version | 目前 = 1；client 不等於就斷線並記錄 |
| 1 | u8 | type | 1 = snapshot chunk（初始常駐集合，tick 0），2 = delta chunk（tick ≥ 1 的新增、修改、刪除），3 = end（沒有紀錄，串流結束） |
| 2 | u16 | count | 這則訊息的紀錄數，type 3 時為 0 |
| 4 | u32 | stream_id | Server 這次串流的編號（啟動時的 Unix 秒）；一個 run 裡它是常數，client 看到它改變就記錯誤並停止，runner 把這個 run 標成 failed |
| 8 | u32 | seq | 這個連線上的訊息流水號，從 0 起 |
| 12 | u32 | tick | 這些紀錄所屬的 tick |
| 16 | f64 | t_event | 這個 tick 的預定釋出時間，Unix 毫秒（`time.time()*1000`） |
| 24 | f64 | t_sent | Server 呼叫 `ws.send` 前一刻的時間，Unix 毫秒 |

snapshot 和 delta 在 client 走同一條程式路徑，區分只為了 log。
beacon 不是獨立的訊息型別：它是 id 0-15 的 16 筆修改紀錄，放在每個 tick 的第一則 delta chunk 開頭，這樣它的時間戳和其他紀錄走完全一樣的路。
`stream_id` 改變時「清空 slot 池」沒有實作，因為 M0-M3 一個 run 只有一次連線；真的需要重連才加（第 12 節）。

### 4.2 紀錄（32 bytes）

| 位移 | 型別 | 欄位 | 說明 |
| --- | --- | --- | --- |
| 0 | u32 | id | Gaussian 身分；0-15 保留給 beacon，場景從 16 起 |
| 4 | f16 x3 | x, y, z | 世界座標位置 |
| 10 | f16 x3 | sx, sy, sz | 縮放（已 exp，線性） |
| 16 | f16 x4 | qw, qx, qy, qz | 單位四元數，順序同 PLY 的 rot_0..3 |
| 24 | f16 | opacity | 已 sigmoid，範圍 (1/255, 1]；**正好 0x0000 代表刪除** |
| 26 | f16 x3 | fdc0, fdc1, fdc2 | SH 0 階係數 f_dc；顯示顏色 = 0.5 + 0.28209 × f_dc |

最後三欄和 PLAN 第 5 節的 RGB 不同，理由和後果在第 0.2 節第 9 條。

沒有 op 欄位。
規則是 **upsert 加 tombstone**：id 不在對照表就是新增，在就是修改；opacity 為 0 就是刪除（client 把 slot 的 opacity 寫 0 並釋放）。
這和 GPU 端的刪除記號是同一個語意（`gaussian_cull.wgsl:139` 丟掉 opacity ≤ 1/255），學生只要理解一件事。
活的紀錄 opacity 一定 > 1/255（第 3.3 節第 3 點保證），所以 0 不會撞到真的值。
刪除紀錄仍是 32 B，M3 的 T-churn 會因此多送一半 bytes；若 M4 顯示刪除的 bytes 超過總量 10%，再加一種 4 B 的刪除訊息（第 12 節）。

numpy 的 dtype（`wire.py`）：

```python
REC = np.dtype([('id', '<u4'), ('pos', '<f2', 3), ('scale', '<f2', 3), ('quat', '<f2', 4),
                ('opacity', '<f2'), ('fdc', '<f2', 3)])          # itemsize == 32
HDR = struct.Struct('<BBHIIIdd')                                # size == 32：version type count stream_id seq tick t_event t_sent
assert REC.itemsize == 32 and HDR.size == 32
```

### 4.3 文字訊息（JSON）

兩種在 M3 就會用到：`{"type":"ping","t_c":...}`（client 每秒送一次，Server 回 `{"type":"pong","t_c":...,"t_s":...}`，V22 用它量時鐘差）和 beacon-ref 頁的 `{"type":"ref_raf","tick":...,"t_raf":...}`（V24）。
第三種是 client 的 10 Hz 回報（`bytes_received`、`records_integrated` 當 ack、`integ_rate`、`pending_records`、`pose`），它的第一個消費者是 M4 的 Server 准入迴圈，所以 M0-M3 不建它、不在這裡規格化；M4 開始時連同第一個用到它的條目一起寫（第 12 節）。
用 JSON 是因為 10 Hz、幾百 bytes，不值得二進位。

### 4.4 單元測試

`tests/test_wire.py`：標頭和紀錄大小是 32；pack 後 unpack 回來 id 相同、浮點值在 fp16 誤差內；opacity 0 的紀錄 `is_delete` 為真；一則 2047 筆的訊息正好 65,536 bytes；fp16 的捨入是 RNE（0.1 → 0x2E66）；含 inf 的輸入會被 `pack` 拒絕。

## 5. 序列檔格式（trace）

一個序列一個資料夾 `data/traces/<name>/`，名字帶參數，例如 `refine_scene=0d2ee665be_rate=78125_tick=100_pre=0.5_seed=0/`。

```
trace.json    # 中繼資料（下）
ticks.bin     # dtype [('tick','<u4'), ('t_rel_ms','<f8'), ('offset','<u8'), ('count','<u4')]，一列一個 tick
records.bin   # 所有紀錄依 tick 順序連續擺放，每筆 32 B，格式同第 4.2 節
```

`trace.json`：

```json
{"schema": 1, "kind": "refine", "scene": "0d2ee665be", "source_records": "data/records/0d2ee665be_n=527437_seed=0_sh0.bin",
 "source_sha256": "...", "seed": 0, "tick_ms": 100, "rate_gps": 78125, "duration_s": 120,
 "params": {"p_recoalesce": 0.5}, "n_ticks": 1201, "n_records": 9375000, "bytes": 300000000,
 "realised": {"rate_gps": 78120.3, "mbps_eq": 20.0, "coalescing_opportunity": 0.498, "deletes": 0, "inserts": 527437, "updates": 9375000, "capacity_needed": 527453},
 "beacon_ids": [0, 15],
 "built_by": "python -m gsstream.traces.refine ...", "repo_sha": "..."}
```

Server 照 `t_rel_ms` 重播：tick i 在 `t0 + t_rel_ms[i]` 釋出 `records[offset:offset+count]`，切成最多 2047 筆的訊息，標頭 `t_event = t0 + t_rel_ms[i]`。
tick 0 是 snapshot（type 1），`t_rel_ms = 0`；之後的 tick 是 delta（type 2）。

### 5.1 ρ 怎麼精確設定

ρ 是提供的位元率除以瓶頸速率（PLAN 第 13 節）。
每筆 32 B = 256 bit，所以在瓶頸 C（bit/s）下要 ρ 的話，速率是 `rate_gps = ρ × C / 256` 顆每秒，每個 tick 要 `n_tick = rate_gps × tick_ms / 1000` 筆。
例子：C = 25 Mbit/s、ρ = 0.8 → 78,125 顆/s → 100 ms 一個 tick 7,812.5 筆。
非整數用累加誤差處理：第 i 個 tick 放 `floor((i+1) × n_tick) - floor(i × n_tick)` 筆，長期平均正好是 n_tick。
`trace.json` 的 `realised.rate_gps` 回報實際平均，ρ 用它算，記到小數四位；`mbps_eq = rate_gps × 256 / 1e6`。
beacon 的 16 筆每 tick 都算在 rate 裡（它們佔 0.2%）。

### 5.2 四種序列

- **T-static**（M1）：來源紀錄檔的全部顆依 id 順序當新增，每 2047 筆一個 tick，所有 `t_rel_ms = 0`（Server 盡快連續送）。
  `python -m gsstream.traces.tstatic --records <bin> --out data/traces/static_scene=0d2ee665be_n=527437/`。
- **T-reveal**：把場景切成邊長 0.5 m 的體素格；沿 `cameras.json` 的 10 個視角之間線性內插成 10 Hz 的相機路徑（`--path-s 60`）；每個格子在它的中心第一次落入視錐（fov 60 度、距離 8 m 內）的那個路徑取樣點被「發現」，它的顆在該 tick 當新增送出；每 tick 上限 `rate_gps × tick` 筆，超過的順延到下一個 tick；路徑走完還沒被發現的格子在最後依距離補上。
  tick 0 只有 beacon。
- **T-refine**：tick 0 把整個場景當 snapshot；之後每 100 ms 挑 `n_tick` 個常駐 id 修改。
  挑法：以機率 `p_recoalesce` 從「前兩個 tick（200 ms 內）改過的 id」裡挑，否則從其餘 id 均勻挑；這個機率就是 PLAN 第 5 節說的可合併比例，取 0.1 和 0.5 兩個版本。
  修改內容：位置加上 σ = 1 cm 的高斯雜訊、f_dc 加上 σ = 0.05 的雜訊（讓畫面看得出在變，但不破壞場景）。
  `realised.coalescing_opportunity` = 200 ms 內同一 id 再被改的紀錄比例，必須在目標 ±0.02 內（`tests/test_traces.py` 檢查）。
- **T-churn**：tick 0 snapshot；之後每 tick 輪流挑一個體素格（和 T-reveal 同樣的格子），先送格內所有 id 的刪除紀錄，再把同一批 Gaussian 用新 id（從目前最大 id + 1 起連續編）當新增送出。
  一個 tick 的筆數 = 2 × 格內顆數；格子依顆數排序後輪流，`rate_gps` 由格子大小決定，`trace.json` 回報實際值。
  刪除在前、新增在後，所以 client 的 free list 在同一 tick 內就能把 slot 回收再用，容量不會成長。
- **Beacon**（所有序列）：16 顆 id 0-15，放在 `cameras.json` 第 0 個視角前方 1 m（第 3.3 節的公式），4×4 格、間距 6 cm、縮放 1 cm、opacity 0.99；每個 tick 的第一則訊息前 16 筆是它們的修改紀錄，第 b 顆的顏色是 tick 的第 b 個位元（1 = 白 f_dc = +1.772，0 = 黑 f_dc = -1.772）。
  16 個位元能編 65,536 個 tick，100 ms 一個 tick 夠 109 分鐘。

`python -m gsstream.traces.<kind> ... --out <dir>` 都寫上面三個檔，並把實際統計寫進 `trace.json.realised`；V20 的 runner 把每個序列的 `realised` 區塊收進 `outputs/m3_trace_replay/V20_traces/metrics.json`，數字才有 `collect.py` 看得到的家。

## 6. 元件設計

### 6.1 (a) 序列產生器 `src/gsstream/traces/`

職責：離線讀紀錄檔和 `cameras.json`，產生第 5 節的三個檔，只用 numpy。

主迴圈（以 T-refine 為例）：

```python
def build_refine(records, rate_gps, tick_ms, p_re, duration_s, seed, beacon):
    rng = np.random.default_rng(seed)
    ids = records['id']; n_tick = rate_gps * tick_ms / 1000
    ticks, out = [], []
    emit(tick=0, t_rel=0, recs=np.concatenate([beacon.records(0), records]))      # snapshot
    recent = deque(maxlen=2)                                                        # 前兩個 tick 改過的 id
    for i in range(1, int(duration_s * 1000 / tick_ms) + 1):
        n = int(np.floor((i + 1) * n_tick) - np.floor(i * n_tick))
        pool = np.concatenate(recent) if recent else ids
        from_recent = rng.random(n) < p_re if len(pool) else np.zeros(n, bool)
        chosen = np.where(from_recent, rng.choice(pool, n), rng.choice(ids, n))
        recs = records[np.searchsorted(ids, chosen)].copy()                        # ids 已排序
        recs['pos'] += rng.normal(0, 0.01, (n, 3)).astype(np.float16)
        recs['fdc'] += rng.normal(0, 0.05, (n, 3)).astype(np.float16)
        emit(tick=i, t_rel=i * tick_ms, recs=np.concatenate([beacon.records(i), recs]))
        recent.append(chosen)
    write(ticks, out, meta)
```

資料結構：`records` 是第 4.2 節的 structured array；`ticks` 是列表的 (tick, t_rel_ms, offset, count)。
寫出的數字：`trace.json.realised`（實際速率、可合併比例、新增/修改/刪除數、所需容量）。

### 6.2 (b) Server `src/gsstream/server/`

職責：接受一個或多個 WebSocket 連線；對每個連線重播一個序列（或 M1 的紀錄檔）；收 client 的文字訊息並寫 JSONL；回 pong。
M0-M3 只有 FIFO，沒有合併、沒有准入迴圈；`session.py` 的待送佇列用 `deque`，M4 才把它換成 dict（合併）加准入迴圈。

```python
# python -m gsstream.server --trace <dir> [--records <bin>] --port 8765 --log <run>/server.jsonl [--linger 5]
async def handle(ws):
    sess = Session(ws, stream_id=int(time.time()), log=jsonl)
    recv_task = asyncio.create_task(sess.recv_loop())        # 文字訊息：ping → pong 並寫 pings.jsonl；ref_raf → ref_raf.jsonl
    t0 = now_ms()
    for tick, t_rel, off, cnt in ticks:
        await sleep_until(t0 + t_rel)                           # asyncio.sleep(max(0, target - now)/1000)
        t_event = t0 + t_rel
        for chunk in np.array_split(records[off:off+cnt], max(1, ceil(cnt / 2047))):
            hdr = HDR.pack(1, 1 if tick == 0 else 2, len(chunk), sess.stream_id, sess.seq, tick, t_event, now_ms())
            await ws.send(hdr + chunk.tobytes())               # websockets 的寫入緩衝有高水位，這裡會被 TCP 反壓擋住
            jsonl.write({'seq': sess.seq, 'tick': tick, 'count': len(chunk), 't_event': t_event, 't_sent': now_ms(), 'lag_ms': now_ms() - t_event})
            sess.seq += 1
    await ws.send(HDR.pack(1, 3, 0, sess.stream_id, sess.seq, tick, t_event, now_ms()))
    await asyncio.sleep(linger); recv_task.cancel()
```

M1 的 `--records <bin>` 模式等於 T-static：不用序列檔，直接把紀錄檔切成 2047 筆一則連續送。
Socket 選項：`sock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)`；`TCP_NOTSENT_LOWAT`（`socket.TCP_NOTSENT_LOWAT`，Linux 上是 25）和 `TCP_MAXSEG` 到 M4 才加。
`websockets.serve(..., max_size=None, compression=None)`：關掉 permessage-deflate，否則 fp16 資料會被壓縮而扭曲 bytes 計數。

寫出的 log：`server.jsonl`（每則送出的訊息一行：seq、tick、count、t_event、t_sent、lag_ms）、`pings.jsonl`、`ref_raf.jsonl`。
結束時寫 `server_metrics.json`：送出 bytes、訊息數、釋出延遲 `lag_ms` 的 p50/p99（Server 自己晚了多少才送）；runner 把它併進 run 的 `metrics.json.server`。

### 6.3 (c) Client fork `client/`

職責：把 32 B 紀錄變成 GPU 裡的 Gaussian，每幀渲染，記錄每個階段的時間戳。
所有新程式在 `client/src/`，上游零修改。

#### URL 參數（全部由 runner 組）

`src=ply|file|ws|inject`、`model_url`（src=ply）、`records_url`（src=file 和 inject 的預填）、`ws_url`（src=ws）、`camera_url`、`capacity`（預設 1,048,576）、`k`（每幀上限，`inf` 或整數）、`max_runs`（預設 4096，第「套用更新」段）、`eval`（0 互動、1 截圖、2 原始幀時間、3 階段分解）、`eval_frames`、`eval_warmup`、`inject_rate`、`inject_locality=contig|random`、`inject_cap`、`clip_sh_degree`、`ply_precision`、`enable_f16_feature`、`selftest=1`。

#### 載入路徑和 WebSplatter 的接點

| 接點 | 上游位置（`third_party/clones/websplatter/src/`） | 我們的程式 | 改什麼 |
| --- | --- | --- | --- |
| 建 device | `main.ts:15-43` | `client/src/main.ts` | 照抄 requestAdapter/requestDevice，多存 adapter 資訊（逐欄位）到 `window.__gpu_info`；eval=3 才要 `timestamp-query` |
| 直接讀 PLY | `ply-loader.ts:46 loadFromURL` | `app.ts` 的 `src=ply` 分支 | 直接呼叫，得到 `PLYObject` |
| 渲染器 | `gaussian-renderer.ts:252 constructor(pc, device, format, camera_buffer, features, ...)` | `app.ts` | 不改；給它 `PLYObject` 形狀的 pool 就好（參數型別 `PlyLoader` 是 `PLYObject` 的別名，`ply-loader.ts:32`） |
| 相機 | `camera.ts:73 load_camera_presets`、`:192 set_preset`、`:203 set_eval_preset` | `app.ts` | 不改 |
| 渲染設定 | `render-settings.ts:59 setCurSHDegree`、`:71 setKernelSize` | `app.ts` | 不改；pool 模式固定 cur_sh_deg=0、kernel 0.3、mip 0 |
| 截圖 | `splat-app.ts:140 evalRenderOneView`、`:264 pixelsToPngDataUrl` | `eval.ts` | 複製 130 行，alpha 強制 255，另掛原始 u8 |
| 階段時間 | `gaussian-renderer.ts:900 readBreakdown` | `eval.ts` | 每 200 幀呼叫一次並串接（環形緩衝只有 200 幀，`:205`） |
| cov 計算 | `workers/ply-worker.ts:19 build_cov`、`:81-85` | `workers/stream-worker.ts` | 複製這 8 行（wgpu-matrix 已是依賴） |
| 補 4 bytes | `utils/util.ts PadTo4Bytes` | `slot-pool.ts` | `import { PadTo4Bytes } from '@ws/utils/util'`；它把 byte 數向上湊到 4 的倍數，因為 GPU buffer 的大小必須是 4 的倍數 |
| 渲染迴圈 | `splat-app.ts:774-802 frame()` | `app.ts` 的 `frame(now)` | 自己寫；用 rAF 的 `now` 參數，不用 `performance.now()` |
| 重排路徑 | `gaussian-renderer.ts:1237 maybeReorderAfterSubmit` | 無 | 不呼叫；README 警語 |

#### slot 池 `stream/slot-pool.ts`

```ts
import { PadTo4Bytes } from '@ws/utils/util';
import type { PLYObject } from '@ws/ply-loader';
export function createPool(device: GPUDevice, capacity: number) {
  const gaussian_cpu = new Float16Array(capacity * 10);            // x y z opacity cov6，同 ply-worker.ts:17
  const sh_cpu = new Float16Array(PadTo4Bytes(capacity * 6) / 2);  // f_dc x3，sh_stride = 3
  const mk = (bytes: number, label: string) => device.createBuffer({ label, size: PadTo4Bytes(bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
  const obj: PLYObject = { sh_degree: 0, num_points: capacity, ply_precision: 'f16', gs_stride: 10, sh_stride: 3,
    gaussian_3d_buffer: mk(capacity * 20, 'pool gaussians'), sh_buffer: mk(capacity * 6, 'pool sh'), gaussian_cpu, sh_cpu };
  // 不呼叫 writeBuffer：WebGPU 新 buffer 全為 0，opacity 0 = 每個 slot 都是死的
  const slotOf = new Map<number, number>();                        // ponytail: Map；id 稠密且 < capacity 時可換成 Uint32Array
  const free = new Uint32Array(capacity); for (let i = 0; i < capacity; i++) free[i] = capacity - 1 - i; let top = capacity;
  return { obj, slotOf, live: () => slotOf.size,
    alloc(id: number): number { let s = slotOf.get(id); if (s !== undefined) return s; if (top === 0) return -1; s = free[--top]; slotOf.set(id, s); return s; },
    get(id: number) { return slotOf.get(id); },
    release(id: number) { const s = slotOf.get(id); if (s === undefined) return false; slotOf.delete(id); free[top++] = s; return true; } };
}
```

`alloc` 池滿時回傳 -1 而不是丟例外：例外會從 rAF 回呼逃出去，迴圈不再註冊下一幀，`__gs.done` 永遠不會發生，runner 只會等到逾時。
free list 是從 capacity-1 倒著填的堆疊，所以第一次配置拿到 0、1、2…，T-static 的 slot 和紀錄順序一致，寫入自然連續。
用 `PLYObject` 的形狀是為了讓 `GaussianRenderer` 的建構子原封不動接受它；它所有依 `num_points` 配置的 buffer（`gaussian-renderer.ts:291, 1167-1192`）會以 `capacity` 配置，每 slot 約 82 B，1 M slot 約 82 MB。
代價是 cull pass 每幀對 `capacity` 個 slot 跑一次（`gaussian-renderer.ts:768-769`），不管活的有幾顆；這就是為什麼 pool 模式需要自己的 null（V7），不能借用上游 `src=ply` 的數字。

#### 套用更新 `stream/apply.ts`

```ts
export function applyUpdates(device: GPUDevice, pool, pending: Batch[], k: number, maxRuns: number, log: StageLog, ctr: Counters) {
  const t0 = performance.now(); let n = 0; const dirty: number[] = [];
  while (n < k && pending.length) {
    const b = pending[0];
    while (b.cursor < b.count && n < k) {
      const i = b.cursor++, id = b.ids[i], o = i * 13;
      if (b.out[o + 3] === 0) {                                     // 刪除：tombstone
        const s = pool.get(id); if (s === undefined) { ctr.delete_unknown++; continue; }
        pool.obj.gaussian_cpu[s * 10 + 3] = 0; pool.release(id); dirty.push(s);
      } else {                                                      // 新增或修改：同一條路
        const s = pool.alloc(id); if (s < 0) { ctr.pool_full++; continue; }
        pool.obj.gaussian_cpu.set(b.out.subarray(o, o + 10), s * 10);
        pool.obj.sh_cpu.set(b.out.subarray(o + 10, o + 13), s * 3);
        dirty.push(s);
      }
      if (id < 16) log.beacon(id, b, epochNow());                   // beacon 逐筆記
      n++;
    }
    if (b.cursor === b.count) { pending.shift(); log.written(b, epochNow()); }
  }
  dirty.sort((a, c) => a - c);
  const runs: [number, number][] = [];
  for (let i = 0; i < dirty.length;) { let j = i; while (j + 1 < dirty.length && dirty[j + 1] <= dirty[j] + 1) j++; runs.push([dirty[i], dirty[j] + 1]); i = j + 1; }
  const span = runs.length > maxRuns;                                                   // ponytail: 超過 max_runs 段就整段重寫，mirror 是正確來源；上限是 max(slot)-min(slot) 的一次大寫入
  const spans = span ? [[runs[0][0], runs[runs.length - 1][1]]] : runs;
  for (const [s0, s1] of spans) {
    device.queue.writeBuffer(pool.obj.gaussian_3d_buffer, s0 * 20, pool.obj.gaussian_cpu.buffer, s0 * 20, (s1 - s0) * 20);
    const lo = (s0 * 6) & ~3, hi = Math.min(pool.obj.sh_buffer.size, (s1 * 6 + 3) & ~3);     // writeBuffer 的位移和大小要是 4 的倍數
    device.queue.writeBuffer(pool.obj.sh_buffer, lo, pool.obj.sh_cpu.buffer, lo, hi - lo);
  }
  ctr.records += n; ctr.runs += runs.length; ctr.span_writes += span ? 1 : 0;
  ctr.bytes_written += spans.reduce((a, [s0, s1]) => a + (s1 - s0) * 26, 0);
  ctr.frames_with_writes += dirty.length ? 1 : 0;
  ctr.write_ms += performance.now() - t0;
}
```

重點：SH 的 stride 是 6 B，奇數 slot 的位移不是 4 的倍數，所以寫入範圍向外擴到 4 的倍數；CPU mirror 存著正確的鄰居 bytes，擴寫是精確的。
`queue.writeBuffer` 的寫入在同一 queue 之後的 `submit` 之前完成，所以這一幀的渲染一定看到這些更新。
`max_runs` 來自 URL 參數（預設 4096）並記進 `stamp.json`；`runs_per_frame`、`bytes_written_per_frame`、`span_writes` 都進 `metrics.json`，V13 要用它們把「每筆成本」和「每次呼叫成本」分開。

#### Worker `workers/stream-worker.ts`

```ts
const HDR = 32;
self.onmessage = (e) => {
  if (e.data.type === 'connect') { ws = new WebSocket(e.data.url); ws.binaryType = 'arraybuffer'; ws.onmessage = onFrame;
                                   ws.onopen = () => post({ type: 'open', t: epochNow() }); ws.onclose = () => post({ type: 'closed' }); }
  if (e.data.type === 'fetch') fetch(e.data.url).then(r => r.arrayBuffer()).then(buf => { for (const chunk of chunks(buf, 2047)) decode(chunk, ...) });   // src=file
  if (e.data.type === 'send') ws.send(JSON.stringify(e.data.msg));                    // ping
};
function onFrame(ev: MessageEvent) {
  const t_recv = epochNow(); const buf = ev.data as ArrayBuffer; const dv = new DataView(buf);
  const version = dv.getUint8(0), type = dv.getUint8(1), count = dv.getUint16(2, true), stream_id = dv.getUint32(4, true),
        seq = dv.getUint32(8, true), tick = dv.getUint32(12, true), t_event = dv.getFloat64(16, true), t_sent = dv.getFloat64(24, true);
  if (version !== 1) { post({ type: 'error', msg: 'wire version' }); ws.close(); return; }
  if (seenStream !== undefined && stream_id !== seenStream) { post({ type: 'error', msg: 'stream_id changed' }); ws.close(); return; }
  seenStream = stream_id;
  if (type === 3) { post({ type: 'end', seq, tick, t_recv }); return; }
  const ids = new Uint32Array(count), out = new Float16Array(count * 13);
  const u32 = new Uint32Array(buf, HDR, count * 8), h = new Float16Array(buf, HDR, count * 16);   // 一筆 = 8 u32 = 16 halves
  for (let i = 0; i < count; i++) {
    const b = i * 16, o = i * 13; ids[i] = u32[i * 8];
    out[o] = h[b + 2]; out[o + 1] = h[b + 3]; out[o + 2] = h[b + 4]; out[o + 3] = h[b + 12];
    if (out[o + 3] === 0) continue;                                                   // 刪除不用算 cov
    const q = quat.create(h[b + 9], h[b + 10], h[b + 11], h[b + 8]); quat.normalize(q, q);   // 線上 wxyz → wgpu-matrix xyzw
    out.set(build_cov(q, vec3.create(h[b + 5], h[b + 6], h[b + 7])), o + 4);                 // ply-worker.ts:19-28 的 8 行
    out[o + 10] = h[b + 13]; out[o + 11] = h[b + 14]; out[o + 12] = h[b + 15];
  }
  post({ type: 'batch', seq, tick, mtype: type, count, bytes: buf.byteLength, t_event, t_sent, t_recv, t_decoded: epochNow(), ids, out }, [ids.buffer, out.buffer]);
}
```

`epochNow() = performance.timeOrigin + performance.now()`：Worker 和主執行緒各有自己的 `timeOrigin`，兩邊都換成 Unix 毫秒才能相減。
啟動時主執行緒和 Worker 來回一次 `{type:'clock'}`，把兩邊 `epochNow()` 的差記進 `metrics.json.clock.worker_main_offset_ms`，理論上接近 0。

#### 渲染迴圈和時間戳 `app.ts`、`stream/stage-log.ts`

```ts
function frame(now: DOMHighResTimeStamp) {            // rAF 給的時間戳，不是 performance.now()
  stage.onRaf(epochAt(now));                           // 所有 t_done 已填、t_raf 未填的列：t_raf = 這一幀
  if (injector) injector.produce(now, pending);
  applyUpdates(device, pool, pending, k, maxRuns, stage, ctr);   // 填 t_written
  const enc = device.createCommandEncoder();
  renderer.frame(enc, context.getCurrentTexture().createView());
  device.queue.submit([enc.finish()]);
  const fid = stage.onSubmit(epochNow());              // 這一幀寫過的列：t_submitted
  device.queue.onSubmittedWorkDone().then(() => stage.onDone(fid, epochNow()));   // 同一批列：t_done
  requestAnimationFrame(frame);
}
```

幀間隔和卡頓不在 app.ts 裡算：runner 對每個頁面（上游的和我們的）注入同一段 `raf_logger.js`（第 6.4 節），所以 M0 的 null 和之後的 run 是同一個儀器。

`stage-log.ts` 的列（每則訊息一列，beacon id 另外逐筆一列）：`[seq, tick, mtype, count, bytes, t_event, t_sent, t_recv, t_decoded, t_written, t_submitted, t_done, t_raf]`，存在預先配置的 `Float64Array` 環（65,536 列 × 13）。
`t_submitted`、`t_done`、`t_raf` 是「寫入該訊息最後一筆的那一幀」的三個時間戳。
階段延遲是相鄰欄位的差：release lag = t_sent − t_event、network = t_recv − t_sent、decode = t_decoded − t_recv、queue+write = t_written − t_decoded、encode = t_submitted − t_written、gpu = t_done − t_submitted、present = t_raf − t_done；總延遲 = t_raf − t_event。
`window.__gs = { drain(): {rows: 只有用到的 n_rows 列, counters, clock}, done, t_connect, t_first_frame, t_complete }` 給 runner 用 CDP 讀；`drain()` 只序列化用過的列，不是整個環，一列 13 個 double 的 JSON 約 300 B，65,536 列約 20 MB，V21 會實際拉一次 6 萬列確認 CDP 拉得動。

`t_first_frame` = 寫入第一筆紀錄那一幀的 `t_raf`；`t_complete` = 寫入最後一筆那一幀的 `t_raf`（Worker 送來 `end` 且佇列清空後確定）；兩者都減 `t_connect`（WebSocket `open` 事件的時間）。

#### 注入器 `stream/injector.ts`（M2）

```ts
start(rate, locality) {                                                   // 每個速率階開始時呼叫一次
  nPerFrame = Math.ceil(rate / 60); ring = [];
  const live = pool.liveIdsSnapshot();
  for (let r = 0; r < 16; r++) {                                          // 預先產生 16 批，之後每幀輪流重用，產生成本不進幀間隔
    const base = Math.floor(Math.random() * (live.length - nPerFrame));
    const ids = new Uint32Array(nPerFrame), out = new Float16Array(nPerFrame * 13);
    for (let i = 0; i < nPerFrame; i++) { const id = locality === 'contig' ? live[base + i] : live[Math.floor(Math.random() * live.length)];
      const s = pool.get(id); ids[i] = id; out.set(pool.obj.gaussian_cpu.subarray(s * 10, s * 10 + 10), i * 13); out.set(pool.obj.sh_cpu.subarray(s * 3, s * 3 + 3), i * 13 + 10); }
    ring.push({ ids, out, count: nPerFrame, bytes: nPerFrame * 32 });
  }
}
produce(now, pending) {
  const t0 = performance.now();
  const dt = (now - last) / 1000; last = now; acc += rate * dt; const n = Math.floor(acc); acc -= n; if (!n) return;
  if (produced - ctr.records > cap) { paused_frames++; return; }         // 佇列封頂，避免吃光記憶體
  const b = ring[(seq++) % ring.length];
  pending.push({ ...b, count: Math.min(n, b.count), cursor: 0, tick: seq, t_event: epochNow(), t_sent: NaN, t_recv: NaN, t_decoded: epochNow() });
  produced += Math.min(n, b.count); inject_ms += performance.now() - t0;   // lag = produced - ctr.records
}
```

它直接推進和網路一樣的 `pending`，所以 `applyUpdates` 和所有 log 走的是同一條程式路徑；payload 陣列在批之間共用，`applyUpdates` 只改 `cursor`，所以共用是安全的。
`inject_ms` 另外回報，`write_ms` 和幀間隔才不會被產生紀錄的成本混進去。
`contig` 讓 dirty slot 連續（最少的 writeBuffer 呼叫），`random` 讓它們散開（最多），兩個極端把成本模型的「每筆成本 b」和「每次呼叫成本」分開。
待處理筆數超過 `inject_cap` 時暫停產生並累計 `paused_frames`；R_max 的判定只看 lag 有沒有成長，暫停過的階一律算失敗。

#### 自我檢查 `stream/selftest.ts`

`?selftest=1` 時跑：alloc 三個 id 得到 slot 0、1、2；release 中間那個再 alloc 新 id 拿回 slot 1；池滿時 alloc 回傳 -1；dirty = [0, 1, 2, 7, 8] 合併成兩段；奇數 slot 的 SH 範圍擴寫後位移和大小都是 4 的倍數；把結果放在 `window.__selftest = {pass, fail, detail}`。
runner 在每個 pool 模式的 run 開始時讀它並寫進 `stamp.json.selftest`，`fail > 0` 就不跑、run 標 failed。

### 6.4 (d) Runner `src/gsstream/runner/`

職責：對一個 LOG 條目的一個 run，啟動該啟動的東西、驅動 Chrome、把數字和出處寫到 run 資料夾。

`chrome.py`：從 `fmri3d-viewer/scripts/wz2_browser_probe.py:21-73` 複製 `WS` 類別（手刻的 WebSocket client，只做文字訊框）和 `:107-143` 的啟動、接上流程，檔頭註明來源。
改動：binary 換成 `$GSSTREAM_CHROME`、旗標換成第 2.3 節那組、對瀏覽器層級和頁面層級各開一條 WS（前者給 `SystemInfo.getInfo` 和 `Target.createTarget`，後者給 `Runtime`/`Page`/`Emulation`）、加 `Runtime.consoleAPICalled` 和 `Runtime.exceptionThrown` 事件的收集（WebGPU 的驗證錯誤只會出現在 console，`gaussian-renderer.ts:281-284`）、加 `Page.javascriptDialogOpening` 的自動關閉（上游 eval 模式在上傳成功時會 `alert()`，`splat-app.ts:398`）、加 `cdp.evaluate(expr)` 和 `cdp.poll(expr, timeout)` 兩個便利函數、加 `Emulation.setCPUThrottlingRate`、加 `--cdp-url` 接既有 Chrome。
`WS.recv` 已處理 127 長度形式，幾十 MB 的 `drain()` 回傳夠用；`WS.send` 只做 126 形式，CDP 的請求都很短。

`raf_logger.js`（約 10 行，經 `Page.addScriptToEvaluateOnNewDocument` 注入每個頁面）：

```js
window.__raf = (() => { const iv = new Float32Array(1 << 20); let n = 0, prev = 0, stalls = 0;
  (function f(t) { if (prev) { const d = t - prev; iv[n++ % iv.length] = d; if (d > 2 * 16.667) stalls++; } prev = t; requestAnimationFrame(f); })(performance.now());
  return { drain: () => ({ intervals: Array.from(iv.subarray(0, Math.min(n, iv.length))), n, stalls }) }; })();
```

同一幀裡所有 rAF 回呼拿到同一個 `t`，所以這個獨立的迴圈量到的間隔就是頁面自己的幀間隔，上游頁面和我們的 client 用同一支。

`httpd.py`：`http.server` 的 `ThreadingHTTPServer`（`serve_demo.py:21,125` 的寫法，多執行緒才不會讓 472 MB 的 PLY 下載擋住 `cameras.json`）加 `SimpleHTTPRequestHandler`，多送 `Access-Control-Allow-Origin: *` 和 `Cross-Origin-Resource-Policy: cross-origin` 兩個標頭（第 0 節解釋過的 ACAO 和 CORP），同時服務 `client/dist/` 和 `data/`（約 25 行）。

`run.py` 的主流程：

```python
def run(args):
    run_dir = OUTPUTS / args.study / args.slug; run_dir.mkdir(parents=True)
    stamp = provenance(run_dir)                                   # 第 6.6 節
    gpu_watch = nvsmi.Watcher(interval_s=10).start()             # 背景執行緒：nvidia-smi pmon -s u -c 1，記每個 pid 的 sm%
    srv = httpd.start(port=args.http_port)
    ws_srv = start_ws_server(args) if args.src == 'ws' else None
    chrome = Chrome.launch(run_dir, cpu_throttle=args.cpu_throttle, cdp_url=args.cdp_url)
    chrome.inject(RAF_LOGGER_JS)
    stamp['adapter'] = chrome.evaluate(ADAPTER_INFO_JS); stamp['gpu_info'] = chrome.system_info()
    burner = chrome.new_tab(f'{base}/burner.html?iters={args.burner_iters}') if args.burner_iters else None   # 同一個 Chrome 的第二個分頁
    chrome.navigate(build_url(args))
    if args.selftest: stamp['selftest'] = chrome.evaluate('JSON.stringify(window.__selftest)')
    if args.eval in (1, 2, 3):
        res = chrome.poll('window.__eval_progress && window.__eval_progress.done && JSON.stringify(window.__eval_progress)', timeout=args.timeout)
    else:
        if args.until_done: chrome.poll('window.__gs && window.__gs.done', timeout=args.timeout); time.sleep(args.tail_s)
        else: time.sleep(args.duration_s)
        res = chrome.evaluate('JSON.stringify({gs: window.__gs && window.__gs.drain(), raf: window.__raf.drain()})')
    write_raw(run_dir, res)                                       # intervals.json / stages.jsonl / eval_results.json / png/
    gpu_watch.stop(); stamp['gpu_watch'] = gpu_watch.summary(own_pids=chrome.pids() + (ws_srv.pids() if ws_srv else []))
    stamp['console'] = chrome.console_lines
    chrome.kill(); ws_srv and ws_srv.stop(); srv.stop()
    metrics = compute_metrics(run_dir, args, stamp)               # 用 gsstream.report.*，寫 metrics.json
    json.dump(stamp, open(run_dir / 'stamp.json', 'w'), indent=1)
```

剔除規則（PLAN 8.4）在 run 全程生效：`Watcher` 每 10 秒取樣一次每個 pid 的 SM 使用率，`foreign_sm_max` = 不屬於我們的 Chrome 程序樹和 Server 的 pid 的最大值，`excluded = foreign_sm_max > 10`；取樣的 pid 清單一起進 `stamp.json`。
`--until-done --tail-s 10` 的意思是「等到 `__gs.done`，再多跑固定 10 秒」；它和 `--duration-s` 互斥，runner 兩個都給就報錯。

`metrics.json` 的共同欄位：`entry`、`study`、`arm`、`params`（dict，含 `capacity`、`k`、`max_runs`、`clip_sh_degree`）、`repeat`、`vsync_mode`、`excluded`、`foreign_sm_max`、`n_frames`、`frame_interval_ms: {p50, p99, max, n}`、`stalls`、`stalls_per_min`、`duration_s`。
各里程碑再加：M0 的 `eval2: {median, p99}`、`stages_ms: {cull, preprocess, sort, render: {p50, p99}}`；M1 的 `t_first_frame_ms`、`t_complete_ms`、`load_duration_s`、`stalls_during_load`、`pixel_diff: {max_abs, n_diff_px}`；M2 的 `write_ms_per_frame`、`inject_ms_per_frame`、`runs_per_frame`、`bytes_written_per_frame`、`span_writes`、`lag_records`、`r_max_gps`；M3 的 `latency_ms: {p50, p95}`、`stages: {...}`、`stage_sum_err_pct`、`pool_full`、`camera_validated`。
分位數只有一個定義：`gsstream.report.quantiles.q(xs, p) = sorted(xs)[floor(p * n)]`（和 `splat-app.ts:374-375` 一樣的 nearest-rank），所有里程碑都用它。

`nvsmi.py`：`Watcher`（上面）、`snapshot()`（`--query-gpu=timestamp,utilization.gpu,memory.used`）和週紀錄器用的 `--query-compute-apps=pid,process_name,used_memory`。

### 6.5 (e) `results/collect.py`

```python
# python results/collect.py [--study m0_static_null] [--check]
rows = [json.load(open(p)) | {'run': p.parent.name} for p in OUTPUTS.glob('*/V*/metrics.json')]   # smoke_* 永遠看不到
for study, group in group_by(rows, 'study'):                                   # 標準函式庫：itertools.groupby + dict；不加 pandas
    grid = json.load(open(ROOT / 'experiments' / study / 'grid.json'))
    table = aggregate(group, by=['entry', 'arm', 'params'], stats={'frame_interval_ms.p99': ['median', 'iqr'], ...}, n='repeat')
    table = fill_blank_cells(table, grid)                                      # 沒跑的格子留空，不略過那一列
    write_csv(RESULTS / 'tables' / f'{study}.csv', table)                      # csv 模組
if args.check: sys.exit(0 if all_tables_match_fresh_build() and no_drift(rows) else 1)
```

`grid.json` 是每個 runner 跑之前就寫好的網格，所以表格有洞一眼看得出，不會被「報成完整」。
`results/figures/make_m0.py` 等讀 `tables/*.csv` 畫圖（nature-figure skill），每張圖的檔名寫在第 7-9 節對應的條目旁。
剔除的 run（`excluded=true`）進表格但在 `excluded_n` 欄計數，不進中位數。

### 6.6 出處戳記（每個 run）

`tools/setup/stamp.sh "$RUN"` 在 Chrome 啟動前寫：

```bash
git -C "$ROOT" rev-parse HEAD                                     > "$RUN/repo.sha"
git -C "$ROOT" diff                                               > "$RUN/repo.patch"
git -C "$ROOT/third_party/clones/websplatter" rev-parse HEAD      > "$RUN/upstream.sha"
git -C "$ROOT/third_party/clones/websplatter" diff                > "$RUN/upstream.patch"
git -C "$ROOT/third_party/clones/websplatter" status --porcelain --untracked-files=no > "$RUN/upstream.status"
"$GSSTREAM_CHROME" --version                                      > "$RUN/chrome.version"
sha256sum "$ROOT"/client/dist/assets/*.js | sort                  > "$RUN/client_dist.sha256"
```

`run.py` 再把 adapter 資訊、`SystemInfo.getInfo`、Chrome 旗標、`VK_ICD_FILENAMES`、nvidia-smi 前後和全程的 `Watcher` 摘要、selftest、序列或紀錄檔的 sha256、所有參數、主機名、時間，一起寫進 `stamp.json`。
`upstream.patch` 若和 `third_party/patches/` 串起來不一致（目前 patches 是空的，所以它必須是空的），`collect.py --check` 會把這個 run 標成 `drift=true`。

## 7. M0 一步一步

Study：`experiments/m0_static_null/`。
對應 PLAN 第 6 節 M0；判定規則在 PLAN 6 M0「完成條件」。
這裡所有 run 都在本機的 headless Chrome（`vsync_mode=headless_timer`），Windows 那台只在連得上時加跑同一組 runner（`--cdp-url`）。

每個 runner 的檔頭是這個條目的事先宣告，在跑之前 commit，git 歷史就是「先宣告後量」的證據；LOG 條目在跑完後引用它，不另外抄一份佔位的條目。
V1 的檔頭長這樣，其他條目同一格式：

```bash
#!/usr/bin/env bash
# V1 - Linux Chrome 153 的 WebGPU 跑在 NVIDIA 上嗎 [explore]
# 問題: 這台工作站的 Playwright Chrome 153.0.8010.12 以 --headless=new 加 Vulkan 啟動時，WebGPU adapter 是 NVIDIA RTX 6000 Ada 還是 SwiftShader？
# 成功條件: adapter.vendor == "nvidia"，且 SystemInfo.getInfo 的 featureStatus 說 WebGPU 是硬體加速。
# 預測: 會是 nvidia；若 vendor 是 "google" 或 isFallbackAdapter 為真，表示 VK_ICD_FILENAMES 沒生效或 Vulkan 被擋。
# Null: 無，第一次量測。
# 失敗時: 先拿掉 --enable-features=Vulkan --use-angle=vulkan 再跑一次（slug 加 _noangle）；還不行就走 Docker 加 Xvfb；都不行記一筆失敗並轉向 PLAN 的 Spark 備案。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$ROOT/tools/setup/env.sh"
uv run python -m gsstream.runner.gpucheck --out "$ROOT/outputs/m0_static_null/V1_gpucheck"
```

### 步驟 0：開始一週的 GPU 紀錄（和 V1 同一天）

```bash
tools/setup/nvsmi_logger.sh start     # nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 >> outputs/gpu_log/gpu_log.csv
                                       # 同時 --query-compute-apps=timestamp,pid,process_name,used_memory -l 60 >> outputs/gpu_log/compute_apps.csv
```

這一步沒有條目，它只是在累積 V6 的原始資料；`outputs/gpu_log/` 裡沒有 `metrics.json`。

### V1：這台 Linux Chrome 的 WebGPU 跑在 NVIDIA 上嗎？

- 指令：`experiments/m0_static_null/run_V1_gpucheck.sh`（上面）。
- 做什麼：起一個只有 `<title>` 的本機 http 頁面，啟動 Chrome（第 2.3 節旗標），做第 2.3 節的 adapter 檢查和 `SystemInfo.getInfo`，不載入任何場景。
- 寫出：`outputs/m0_static_null/V1_gpucheck/{stamp.json, metrics.json}`。
- 數字：`adapter.vendor`、`adapter.description`、`is_hardware`（bool，第 2.3 節定義）、`features` 含不含 `shader-f16` 和 `timestamp-query`、`maxStorageBufferBindingSize`、`gpu_info.devices[0].deviceString`。
- 成功條件（事先宣告）：`vendor == "nvidia"` 且 WebGPU 的 featureStatus 是硬體加速。
- 這一步只有三個 Python 檔（`chrome.py`、`gpucheck.py`、`nvsmi.py` 的 snapshot），不需要 client 程式，所以是第一天就能做的量測。
  這份文件 merge 之後的第一個動作就是跑它；這份文件寫作時被限制不能啟動 Chrome，所以它還沒有數字。

### V2：靜態場景一幀要多久（eval=2），N 掃描加備援檢查

- 前置：第 3 節的 garden 子集四個和 `0d2ee665be` 兩個（0.25 M、全量）；clone 裝好；`cameras.json` 兩個並通過 `check_cameras.sh`。
- 煙霧測試（第 1 週）：同一個 runner 只跑 `scene=garden n=250000 rep=0`，`--slug smoke_eval2_garden_n=250000`，目的是把 `run.py`、`httpd.py`、`stamp.sh`、`quantiles.py` 在最小規模下接通；它的輸出永遠不被引用。
- 指令：`run_V2_eval2_n_sweep.sh`，對每個 (scene, N) 跑 5 次（每次獨立啟動 Chrome）：

```bash
uv run python -m gsstream.runner.run --entry V2 --study m0_static_null --mode upstream --eval 2 \
  --ply "data/subsets/garden_n=${N}_seed=0.ply" --cameras data/cameras/garden_orbit_k=10.json \
  --clip-sh-degree 3 --warmup 100 --frames 500 --bench-server http://127.0.0.1:1 \
  --slug "V2_eval2_scene=garden_n=${N}_rep=${REP}"
```

  `--mode upstream` 表示用 clone 自己的 app：runner 在 clone 裡啟動 `pnpm dev --port 5173`（Vite 從記憶體服務，建置輸出不落地），`model_url` 和 `camera_url` 給 `httpd.py` 的絕對網址，這是對上游計時模式的「重現」，不是我們的程式。
  `clip_sh_degree=3` 是上游預設，V2 和 V3 都用它，因為它們重現的是上游的數字；V5 之後的 null 用 0。
  `--bench-server http://127.0.0.1:1` 讓上游的結果上傳立刻失敗（`done` 在上傳之前就設了，`splat-app.ts:383`），不會卡住 runner。
  另外在 N = 500 k 加兩個 arm：`--extra enable_f16_feature=0` 和 `--extra ply_precision=f32`，各 5 次，這是 PLAN 11.6 說的 Linux fp16 bug 的備援檢查。
- 寫出：每個 run 的 `eval_results.json`（500 個 frame_times）和 `metrics.json`（`eval2.median`、`eval2.p99` 用我們的分位數定義重算）。
- 數字：每個 N 的 eval=2 中位數和 p99（ms），5 次的中位數和四分位距；`aa_spread` = 5 次 p99 的 (max − min) / median，n = 5，對每個格子都算。
- 成功條件：所有格子都跑完且 `excluded=false`；兩個備援 arm 的中位數和主 arm 差 < 2 倍（否則 fp16 路徑有問題，記下來）。
  eval=2 是「編碼到 GPU 完成」的同步迴圈（`splat-app.ts:359-368`），沒有 rAF，它的 `aa_spread` 只是回報，M0 的 A/A 門檻在 V5。
- 圖：`results/figures/fig_m0_eval2_vs_n.png`（兩條曲線：garden、ScanNet++）。

### V3：各階段 GPU 時間（eval=3）

- 指令同 V2，`--eval 3 --warmup 100 --frames 200`，每個 (scene, N) 5 次。
  `eval_frames=200` 是因為上游的時間戳環只有 200 幀（`gaussian-renderer.ts:205`），超過會被覆蓋；上游模式不能 patch，所以用 5 次 200 幀代替 1 次 1000 幀。
- 寫出：`eval_results.json`（`raw.cull_ms[]` 等四個陣列）、`metrics.json.stages_ms`。
- 數字：cull、preprocess、sort、render 各自的 p50 和 p99（ms）對 N；另外 `quantized_fraction` = 四個陣列裡值是 0.1 ms 整數倍的比例，用來確認 `--disable-dawn-features=timestamp_quantization` 有效（期望接近 0；接近 1 表示旗標沒生效，要記錄）。
- 成功條件：四個階段的 p50 加總 ≤ V2 的 eval=2 中位數的 1.2 倍（不然時間戳有問題）；sort 的 p50 隨 N 單調增加。
- 圖：`results/figures/fig_m0_stages_vs_n.png`（堆疊長條）。

### V4：無頭模式的幀間隔在已知會卡的負載下會變長嗎（正控制，門檻）

- 為什麼先做這個：第 0.1 節第 1 條。
  上游的 `frame()` 從不等 GPU 完成（`splat-app.ts:774-802`），無頭模式的 rAF 若是純計時器，GPU 再慢間隔也可能一直是 16.67 ms；那樣的儀器量不到任何卡頓，之後所有卡頓比較都是空的。
- 指令：`run_V4_raf_positive.sh`，3 次：

```bash
uv run python -m gsstream.runner.run --entry V4 --study m0_static_null --mode upstream --eval 0 \
  --ply data/garden/point_cloud.ply --cameras data/cameras/garden_orbit_k=10.json --clip-sh-degree 3 \
  --window 3840x2160 --duration-s 60 --drop-s 10 --slug "V4_rafpos_scene=garden_n=6606688_win=3840x2160_rep=${REP}"
```

  同一個設定再用 `--eval 2 --frames 200` 跑 1 次（slug 加 `_eval2`），得到這個負載的 GPU 幀時間當對照。
- 寫出：`intervals.json`、`metrics.json.frame_interval_ms.{p50,p99,max}`、`stalls_per_min`、`eval2.median`。
- 數字：6.6 M 顆、4K 視窗、SH 3 階下的幀間隔 p50、p99、每分鐘卡頓；同設定的 eval=2 中位數。
- 成功條件（事先宣告）：rAF 的 p99 > 25 ms，且 p50 和 eval=2 中位數的比值在 [0.8, 1.5]（rAF 跟著 GPU 走）。
  失敗的意思是「eval=2 說一幀超過預算，但 rAF 說沒有」：幀間隔在這台的無頭模式下不是可用的指標，M2 和 M3 的卡頓與延遲量測改走有螢幕的路線（登入主機或 Windows），在那之前不跑 V13 以後的條目；V5 仍然跑，但只當描述性數字。
- 這是一個真正的門檻，和 V1 一樣是「儀器能不能用」的問題，所以排在 null 之前。

### V5：幀間隔和卡頓（rAF），M0 的卡頓基準和 A/A 門檻

- 指令：`run_V5_raf_null.sh`，每個 (scene, N) 5 次：

```bash
uv run python -m gsstream.runner.run --entry V5 --study m0_static_null --mode upstream --eval 0 \
  --ply "data/subsets/garden_n=${N}_seed=0.ply" --cameras data/cameras/garden_orbit_k=10.json --clip-sh-degree 0 \
  --duration-s 120 --drop-s 20 --slug "V5_raf_scene=garden_n=${N}_rep=${REP}"
```

  相機固定在 preset 0，不旋轉（`animation=0`）；120 秒，丟掉前 20 秒；`clip_sh_degree=0` 和之後的 pool 一致。
- 寫出：`intervals.json`（約 6,000 個幀間隔）、`metrics.json.frame_interval_ms.{p50,p99,max}`、`stalls_per_min`（間隔 > 33.3 ms 的次數 ÷ 分鐘數）。
- 數字：每個 N 的幀間隔 p50、p99、每分鐘卡頓數，5 次的中位數和四分位距；`aa_spread` = 5 次 p99 的 (max − min) / median，n = 5；A/A 門檻看 garden N = 500 k 那一格。
- 成功條件：garden 500 k 的 `aa_spread ≤ 10%`，這決定這台機器是不是量測 client（PLAN M0 完成條件）。
  超過 10% 時依第 0.2 節第 12 條：M1 不開始，直到你選定 client。
  這裡的 `stalls_per_min` 是上游載入路徑的靜態卡頓率；pool 模式（M1-M3）的 null 是 V7，不是這裡。
- 圖：`results/figures/fig_m0_frame_interval_vs_n.png`。

### V6：工作站 GPU 每週有幾小時是空的

- 指令：`run_V6_gpu_idle.sh` → `uv run python -m gsstream.report.gpu_idle outputs/gpu_log/ --threshold 10 --own-process-name chrome-linux64 --out outputs/m0_static_null/V6_gpu_idle`。
  `--own-process-name` 讓它把我們自己的 Chrome 從「別人」裡排除（`compute_apps.csv` 有 `process_name`）。
- 寫出：`outputs/m0_static_null/V6_gpu_idle/metrics.json`：`idle_hours_per_week`（別人的 GPU 使用率 ≤ 10% 的小時數）、每小時時段的空閒機率表、取樣涵蓋的天數。
- 成功條件：無（這是描述性數字）；低於 20 小時就表示無人值守的 M4-M6 要排在特定時段，記在 LOG。
- 這個條目在紀錄滿 7 天後才寫（第 2 週末）。

## 8. M1 一步一步

Study：`experiments/m1_static_stream/`。
對應 PLAN 第 6 節 M1；M1 用 ScanNet++ `0d2ee665be` 全量（527,437 顆），capacity = 1,048,576。
「直接讀檔版」的定義在第 0.2 節第 10 條：同一個 `.bin` 經 `src=file`，k = inf。
逐像素檢查測的是「WebSocket 傳輸、分塊、slot 配置」有沒有改變任何 bytes，正是 M1 要驗的東西。

### V7：pool 模式的靜態 null（M1-M3 所有卡頓和 p99 比較的基準）

- 前置：第 6.3 節的 `main.ts`、`app.ts`、`slot-pool.ts`、`stream-worker.ts` 的解碼、`apply.ts`、`selftest.ts`；`tests/test_wire.py` 通過；紀錄檔 `data/records/0d2ee665be_n=527437_seed=0_sh0.bin`。
- 指令：`run_V7_pool_null.sh`，5 次：

```bash
uv run python -m gsstream.runner.run --entry V7 --study m1_static_stream --mode client --src file --eval 0 \
  --records data/records/0d2ee665be_n=527437_seed=0_sh0.bin --cameras data/cameras/0d2ee665be_lookaround_k=10.json \
  --capacity 1048576 --k inf --selftest --until-done --tail-s 140 --drop-s 20 \
  --slug "V7_poolnull_scene=0d2ee665be_n=527437_cap=1048576_rep=${REP}"
```

  等整個紀錄檔套用完（`__gs.done`），再量 140 秒並丟掉前 20 秒，和 V5 一樣是 120 秒的靜態畫面，只是場景住在 1 M 容量的 pool 裡、零更新。
- 寫出：`metrics.json.frame_interval_ms.{p50,p99,max}`、`stalls_per_min`、`stamp.json.selftest`。
- 數字：5 次的 p50、p99、每分鐘卡頓的中位數和四分位距；`aa_spread`；和 V5 的 ScanNet++ 527 k arm（同 N、同 SH 0 階，差別只在容量 1 M 對 527 k 和 cov 由 fp16 scale/quat 算）的 p50 差。
- 成功條件：`selftest.fail == 0`；所有 5 次 `excluded=false`；`aa_spread ≤ 10%`；p99 ≤ 25 ms。
  p99 > 25 ms 表示 1 M 容量的靜態成本已經超過幀預算，容量要降（例如 600 k），這是一個要給你的決策項，而且 V8 以後都用降過的容量重跑。
- 這個 null 和後面的 arm（V10、V13-V18、V23）只差一個因素（有沒有更新），所以它是「相減」的對照，不只是上界。

### V8：同一個 arm 跑兩次，畫面會逐像素相同嗎（A/A）

- 指令：`run_V8_pixel_aa.sh`：`--src file --eval 1 --k inf --cameras data/cameras/0d2ee665be_lookaround_k=10.json`，跑 2 次（`V8_pixel_aa_rep=0`、`V8_pixel_aa_rep=1`）；app 在 `__gs.done` 之後才進 eval=1，所以 10 張 PNG 都是全場景。
  接著 `uv run python -m gsstream.report.compare_png outputs/m1_static_stream/V8_pixel_aa_rep=0/png outputs/m1_static_stream/V8_pixel_aa_rep=1/png --out outputs/m1_static_stream/V8_pixel_aa_compare`。
- 寫出：每次 `png/pose_00.png` … `pose_09.png` 和 `raw/pose_xx.u8`（`eval.ts` 掛出來的原始 u8，alpha 已設 255）；`V8_pixel_aa_compare/metrics.json` 寫每個視角的 `max_abs_diff` 和 `n_diff_px`（比 u8 陣列，不比 PNG）。
- 數字：10 個視角的最大絕對像素差和不同像素數。
- 成功條件：全部為 0。
  不為 0 的話（原因會是 cull 的 atomicAdd 讓同深度的 splat 順序不固定），把這裡的最大值宣告為 V9 的容忍值，並在 LOG 寫明。

### V9：串流版和讀檔版逐像素相同嗎

- 指令：`run_V9_pixel_identity.sh`：先 `--src file`（`V9_pixel_identity_src=file`），再 `--src ws`（Server 用 `--records <bin>`，`V9_pixel_identity_src=ws`），都 `--eval 1 --k inf`；`compare_png` 寫 `V9_pixel_identity_compare/metrics.json`。
- 數字：同 V8。
- 成功條件：≤ V8 的容忍值（理想為 0）。
  不成立就是 bug，先修（PLAN M1 完成條件）；常見嫌疑依序是 SH 擴寫範圍、刪除記號、Worker 的 quat 順序。

### V10：k = inf 對 k = 16000，第一張畫面、完成時間、載入期間的卡頓

- 指令：`run_V10_k_arms.sh`，兩個 arm × 5 次：`--src ws --eval 0 --k {inf,16000} --until-done --tail-s 10`（等到載入完成再多跑 10 秒，不丟前段，因為要量的就是載入期間）。
- 寫出：`metrics.json.{t_first_frame_ms, t_complete_ms, load_duration_s, stalls_during_load, frame_interval_ms.max}`、`stages.jsonl`、`server.jsonl`。
- 數字：兩個 arm 各 5 次的第一張畫面時間、完成時間、載入期間的卡頓次數和最長幀間隔；`expected_null_stalls` = V7 的 `stalls_per_min` × `load_duration_s` / 60 + 1（加 1 是離散化的寬容，null 在 headless 下很可能是 0 次）。
- 成功條件：k = 16000 的 `stalls_during_load ≤ expected_null_stalls`；k = inf 的 `t_complete_ms` 不晚於 k = 16000 加 1 秒（否則分塊邏輯有額外開銷）。
  k 的選擇標準事先宣告：卡頓不超過 `expected_null_stalls` 的前提下完成時間最短；兩個 arm 都超過就把 k 降到 4000 再跑一輪（新條目）。
- 圖：`results/figures/fig_m1_load_timeline.png`（兩個 arm 的累計吸收筆數對時間，疊上卡頓事件）。

### V11：0 階球諧的代價（gsplat，10 個視角的 PSNR）

- 指令：`run_V11_sh0_psnr.sh`：

```bash
PYTHONPATH="$GS_PLAYGROUND_SRC:$ROOT/src" "$GSSTREAM_PUFFIN_PY" -m gsstream.report.sh_psnr \
  --ply data/subsets/0d2ee665be_n=527437_seed=0.ply --cameras data/cameras/0d2ee665be_lookaround_k=10.json \
  --out outputs/m1_static_stream/V11_sh0_psnr
```

  `sh_psnr.py` 用 `gs_playground.gs.ply.load_ply`、`gs_playground.gs.render.load_cameras`、`render(cloud, cam)`（DC only，等於線上格式）和 `render(cloud, cam, sh_degree=3)`、`psnr`，兩張圖都存 PNG。
- 數字：10 個視角的 PSNR（dB）、平均、最小。
- 成功條件（PLAN M1）：平均掉幅 ≤ 2 dB 就接受 SH0；超過就開一個決策項給你（bytes 對畫質）。
  這是 gsplat 對 gsplat，和 WebSplatter 的渲染無關，所以相機座標慣例的差異不會混進來。

### V12：fp16 線上格式相對 f32 直接載入的代價（回報，不門檻）

- 指令：`run_V12_fp16_cost.sh`：`--mode client --src file --eval 1`（`V12_fp16_cost_src=file`）對 `--mode client --src ply --ply <subset.ply> --ply-precision f32 --clip-sh-degree 0 --eval 1`（`V12_fp16_cost_src=ply`，上游載入路徑、f32 儲存、SH 0 階，但截圖走我們的 `eval.ts`，所以兩邊 alpha 都是 255），兩邊同一個 `cameras.json`；`compare_png --psnr` 寫 `V12_fp16_cost_compare/metrics.json`。
- 數字：10 個視角的 PSNR 和最大像素差。
- 成功條件：無門檻；只回報。
  它同時是對「cov 由 fp16 scale/quat 算」和「線上放 f_dc」這兩個設計的量化，M8 之前不改格式。

## 9. M2 和 M3

### 9.1 M2：瀏覽器的吸收上限（Study `experiments/m2_absorb_capacity/`）

對應 PLAN M2 和第 7 節 H0；前提是 V4 通過（幀間隔在這台是可用的指標）。
全部用 `--src inject`：頁面先從 `records_url` 預填 `0d2ee665be` 全量，再由注入器修改既有 id，不經網路。
三種 client 狀態：`unthrottled`、`cpu4x`（runner 對頁面下 `Emulation.setCPUThrottlingRate {rate: 4}`）、`burner`（同一個 Chrome 的第二個分頁跑 `burner.html?iters=I`，第 0.2 節第 14 條）。

建的東西：`injector.ts`、`burner.html`（一個每幀 dispatch `iters` 次 1 M 執行緒的 busy compute shader，並把自己的幀數掛在 `window.__burner.frames` 讓 runner 確認它在跑）、runner 的 `--cpu-throttle` 和 `--burner-iters`、`report/costmodel.py`、`report/predict_rmax.py`、`report/h0_verdict.py`。

順序和條目：

1. **V13 成本模型**（一個 sweep 條目）：`unthrottled`，`k ∈ {1k, 4k, 16k, 64k, 256k}` × `locality ∈ {contig, random}`，`inject_rate = k × 120`（保證每幀都有 k 筆可寫，`inject_cap = 4k` 讓佇列只保留兩幀的量），每格 60 秒、5 次。
   數字：每幀 `write_ms` 對 k 的線性擬合 `a + b·k`（每個 locality 一條）和殘差 RMS；`inject_ms` 另列；幀間隔 p99 對 k；`runs_per_frame`、`bytes_written_per_frame`、`span_writes`；random 在哪個 k 開始 `runs_per_frame > max_runs`（4096）。
   成功條件：contig 的殘差 RMS ≤ 整個網格上擬合 `write_ms` 平均值的 20%；`span_writes > 0` 若且唯若 `runs_per_frame > max_runs`（這是對 apply.ts 邏輯的檢查，不是物理結果）。
   random 跨過 `max_runs` 之後成本模型不再是 `a + b·k` 而是一次 `(max slot − min slot) × 26 B` 的大寫入，擬合只用跨過之前的點，跨過的 k 當數字回報。
2. **V14 預測**（有 runner 的條目，跑 V16 前）：`run_V14_predict.sh` → `uv run python -m gsstream.report.predict_rmax --from outputs/m0_static_null outputs/m1_static_stream outputs/m2_absorb_capacity --out outputs/m2_absorb_capacity/V14_predict`。
   用 V7 的幀間隔 p99（`p99_static`）、V3 的階段時間和 V13 的 a、b，算 `k_max = (25 − p99_static − a) / b`、`R_max_pred = k_max × 60` 顆/s、換算 Mbit/s-eq（× 256 / 1e6），三種狀態各寫一個預測進 `metrics.json`（cpu4x 假設 a、b 乘 4；burner 假設 p99_static 乘 2）。
   V16-V18 的「差在 2 倍內」比的是這個檔裡的數字，不是 LOG 裡手打的數字。
3. **V15 burner 調整**（V18 前）：`iters ∈ {1, 2, 4, 8, 16}`，每個 3 次，量主頁在沒有注入時（`inject_rate=0`）的幀間隔 p50 相對 V7 的比值，選比值 ≤ 2.0 的最大 iters；全部超過就把 shader 縮小再掃（新條目）。
4. **V16 R_max unthrottled**、**V17 R_max cpu4x**、**V18 R_max burner**（各一個 sweep 條目）：對每個 k ∈ {1k, 4k, 16k, 64k, 256k}，注入速率階梯 {25k, 50k, 100k, 200k, 400k, 800k, 1.6M, 3.2M} 顆/s，每階 30 秒，`locality = contig`，超過 `k × 60` 的階直接略過並記錄；每階取後 25 秒算幀間隔 p99 和 `lag_growth`（`produced − integrated` 在階末減階初）。
   `R_max(k)` = 滿足 p99 ≤ 25 ms 且 `lag_growth ≤ rate / 60`（第 0.2 節第 11 條）且 `paused_frames == 0` 的最高階；該狀態的 `R_max` = 各 k 的最大值，5 次。
   V18 每個 run 同時記 `burner_ratio_observed` = 該 run 第一階（25k）的 p50 相對 V7 p50 的比值，只有落在 [1.5, 2.0] 的 run 算有效，其他標 `excluded` 並寫原因（別人的 GPU 負載會改變 burner 的效果，所以不能借 V15 的數字）。
   數字：`R_max` 顆/s 和 Mbit/s-eq、n = 5 的中位數和四分位距、達成它的 k、每個 k 的 R_max(k)。
   成功條件：和 V14 的預測差在 2 倍內；否則先找量測 bug（PLAN M2）。
   `R_max` 落在階梯最高階就是「邊界上的勝者」，要延長階梯（新條目），不能叫上限。
5. **V19 H0 判定**：`run_V19_h0_verdict.sh` → `uv run python -m gsstream.report.h0_verdict --from outputs/m2_absorb_capacity --out outputs/m2_absorb_capacity/V19_h0_verdict`，照 PLAN 第 7 節 H0 的規則（至少一種狀態 ≤ 50 Mbit/s-eq 走 Path A；全部 > 200 走 Path B；介於之間只在會成為瓶頸的狀態做主實驗）寫 `metrics.json.path` 和三種狀態的 R_max 中位數；LOG 條目引用它。

圖：`results/figures/fig_m2_rmax.png`（三種狀態的 R_max，畫 50 和 200 Mbit/s-eq 兩條線）、`fig_m2_cost_model.png`。

### 9.2 M3：重播序列、建立延遲量測（Study `experiments/m3_trace_replay/`）

對應 PLAN M3 和第 8.2 節。
Server 在本機回環（`ws://127.0.0.1:8765`），FIFO，不限速；client `k = inf` 以外另加 V10 選出的 k；相機固定 preset 0，beacon 在視野內。

建的東西：`traces/{reveal,refine,churn,beacon}.py`、Server 的序列重播（第 6.2 節）、`stage-log.ts` 的 beacon 逐筆列、ping/pong、`beacon-ref.html`、`report/film.py`、`report/m3_verdict.py`。

順序和條目：

1. **V20 序列統計**：`run_V20_traces.sh` 建 4 個序列（T-reveal 60 秒、T-refine p = 0.1 和 0.5 各 120 秒、T-churn 120 秒，速率用 C = 25 Mbit/s、ρ = 0.8 的 78,125 顆/s）加 dev 場景的 T-refine，然後把每個 `trace.json.realised` 收進 `outputs/m3_trace_replay/V20_traces/metrics.json`。
   數字：每個序列的實際速率（顆/s 和 Mbit/s-eq）、可合併比例、新增/修改/刪除數、所需容量；全部對照 V16-V18 的 R_max。
   成功條件：T-refine 的可合併比例在目標 ±0.02；T-churn 的所需容量 ≤ 1,048,576；可合併比例 < 0.05 的序列要記「合併沒有東西可做」。
2. **V21 T-static 經 Server 重播的階段完整性**：`--trace static_... --eval 0 --until-done --tail-s 5`，1 次。
   數字：`stages.jsonl` 裡每則訊息的 7 個階段差；`stage_sum_err_pct` = |各階段加總 − (t_raf − t_event)| / 總延遲 的最大值；負的階段差的數量；`drain_ms` = CDP 拉回全部列（這裡約 260 則訊息）的時間，另外以 `?selftest=1` 填滿 65,536 列再 `drain()` 一次，記 `drain_full_ms` 和 bytes，確認 CDP 拉得動最大的 log。
   成功條件：負值為 0 且 `stage_sum_err_pct` ≤ 5%；`drain_full_ms` < 10 秒。
   因為階段是相鄰時間戳的差，加總在數學上等於總延遲；這個檢查抓的是漏填、順序錯、Worker 和主執行緒時間原點換算錯這三種 bug，不是物理上的驗證，物理驗證是 V24。
3. **V22 時鐘差上限**：client 每秒 ping 一次，60 秒。
   數字：`offset = t_s − (t_c + rtt/2)` 的中位數、`rtt/2` 的最大值（這就是上限）、Worker 對主執行緒的 `clock.worker_main_offset_ms`。
   成功條件：同一主機下上限 ≤ 1 ms；之後所有延遲數字附上這個上限。
4. **V23 FIFO 重播**（一個 sweep 條目）：4 個序列 × k ∈ {inf, V10 選出的 k} × 5 次，每個序列跑到它結束（T-reveal 60 秒，其他 120 秒）；T-refine 和 T-churn 丟前 20 秒，T-reveal 不丟，因為它的前 20 秒就是量測對象。
   數字：每個序列的事件到顯示延遲 p50、p95（以訊息為單位，`t_raf − t_event`；beacon 逐筆另算一組）、7 個階段的 p50 堆疊、每分鐘卡頓（對照 V7）、Server 的釋出延遲 p99、`pool_full` 次數。
   M3 是 FIFO 且沒有合併，所以 PLAN 8.2 的「版本等於或晚於」規則退化成「寫入它的那一幀」；訊息層級用該訊息最後一筆的那一幀，對同一則訊息裡較早的紀錄是上界，beacon 的逐筆列才是精確值。
   成功條件：所有格子跑完；T-churn 的 `pool_full` 為 0；卡頓不高於 V7 null 的 2 倍。
   這裡的延遲數字來自 `vsync_mode=headless_timer`，`metrics.json.camera_validated = false`，V24 之後只有 display 模式的 run 會變成 true。
5. **V24 攝影機驗證**：需要有實體螢幕的機器（Windows 那台或一台筆電）和 240 fps 慢動作手機。
   程序：同一螢幕並排開 client 頁（T-refine p = 0.5，`k = inf`）和 `beacon-ref.html`（同一個 Server，收到 delta 就把 16 個 DOM 方塊設成 tick 的位元，並把它自己的 rAF 時間戳以 `ref_raf` 送回 Server）；拍 60 秒；`uv run python -m gsstream.report.film --video x.mp4 --roi-main x,y,w,h --roi-ref x,y,w,h` 對每個 tick 找兩個視窗第一次顯示該 tick 的影格 `c_main(t)`、`c_ref(t)`，以及程式端 `t_raf_main(t)`、`t_raf_ref(t)`。
   同一台機器、同一個 runner（`--cdp-url`、`vsync_mode=display`）在拍攝的同時就是一個 V23 的 T-refine arm 的重跑（`V24_camera_trace=refine_rep=<r>`，3 次），所以至少有一組被引用的延遲數字是在被驗證的儀器上量的。
   數字：`D_cam = c_main − c_ref`（攝影機量到的管線附加延遲，螢幕延遲互相抵消）、`D_proxy = t_raf_main − t_raf_ref`，`(D_proxy − D_cam)` 在 ≥ 50 個 tick 上的平均和標準差（攝影機的解析度是 4.2 ms）；display 模式下 T-refine 的延遲 p50、p95，`camera_validated = true`。
   成功條件（PLAN M3）：標準差 ≤ 8 ms 就以程式量測為主；否則主結果改用攝影機並重排次數。
   這個驗證只涵蓋 display 模式；V23 的 headless 數字仍標 `camera_validated=false`，報告裡要分開寫。
6. **V25 M3 判定**：`run_V25_m3_verdict.sh` → `uv run python -m gsstream.report.m3_verdict --from outputs/m3_trace_replay outputs/m2_absorb_capacity --out outputs/m3_trace_replay/V25_m3_verdict`，寫延遲量測工具的最終定義（程式或攝影機）、每個序列的流量相對 R_max 的位置、`camera_validated` 的涵蓋範圍。

圖：`fig_m3_stage_stack.png`（堆疊長條，4 個序列）、`fig_m3_proxy_vs_camera.png`（散點）。

### 9.3 時鐘和時間戳的約定（M3 用）

- Server：`time.time() * 1000`。
- 主執行緒和 Worker：`performance.timeOrigin + performance.now()`，各自換算；啟動時做一次來回校準。
- rAF 的 `now` 是相對 `timeOrigin` 的，`epochAt(now) = performance.timeOrigin + now`。
- 同一主機不做時鐘同步；跨主機（Windows client）用 V22 的 ping/pong 偏移，並回報上限。

### 9.4 M4 需要先準備的東西（只列，不做）

對應 PLAN M4。
在這台機器上 `tc` 有兩條路，第 9 週前要你選一條：

1. **Docker 路線（不需要 sudo）**：`experiments/m4_netem/docker/Dockerfile`（`python:3.12-slim` 加 `iproute2`、`tcpdump`、`websockets`），`sg docker -c 'docker run --rm --cap-add NET_ADMIN --cap-add NET_RAW -p 8765:8765 -v "$ROOT":/w gsstream-server'`。
   容器裡在 `eth0` 出口：`tc qdisc add dev eth0 root handle 1: htb default 10`、`tc class add dev eth0 parent 1: classid 1:10 htb rate 25mbit`、`tc qdisc add dev eth0 parent 1:10 handle 10: netem delay 20ms loss 1%`。
   延遲只加在 Server 出口（單向 20 ms），回程 ack 不加，RTT 20 ms 是不對稱的，要在 stamp 標明。
   容器內 `tcpdump -i eth0 -w` 確認 MSS。
2. **sudoers 路線**：一條只允許 `/usr/sbin/tc` 的 NOPASSWD 規則，在 `lo` 上做同樣設定；`lo` 的 MTU 是 65536，所以 MSS 用 Server socket 的 `TCP_MAXSEG` 限制到 1460，不用改路由。

兩條路都還需要：`iperf3` 的替代品（`tools/setup/netcheck.py`，30 行的 Python 吞吐量檢查，或經你同意 `micromamba create -n nettools -c conda-forge iperf`）；Server 加 `TCP_NOTSENT_LOWAT`（`socket.TCP_NOTSENT_LOWAT`）和合併用的 dict；client 的 10 Hz 回報（第 4.3 節）；runner 的 `--netem` 參數和 stamp 欄位。
`sch_htb`、`sch_netem` 模組在 `/lib/modules/<kernel>/kernel/net/sched/` 都有，沒載入，第一次用會自動載入。

## 10. 八週時程

每週列出學生親手寫的部分、跑的東西、寫下的 LOG 條目；LOG 條目一律在 run 結束後才寫。
「親手寫」的意思是學生自己打字並能解釋每一行；runner 和 CDP 的樣板可以從 `wz2_browser_probe.py` 複製。
M0 用兩週（PLAN 估一週），因為 pnpm 安裝要等你同意、V4 門檻要先過；M3 壓在第 7-8 週，攝影機那一步若沒有有螢幕的機器會溢到第 9 週。

| 週 | 學生親手建 | 跑 | 寫下的 LOG 條目 |
| --- | --- | --- | --- |
| 1 | repo 骨架（第 1 節）、`env.sh` 和 `env.local.example.sh`、`clone_websplatter.sh`、`chrome.py`（複製 WS 類別，改旗標，兩條 CDP 連線）、`gpucheck.py`、`nvsmi.py`、`raf_logger.js`、最小的 `httpd.py`、`run.py`（只做 eval 輪詢和 drain）、`stamp.sh`、`quantiles.py`、`fetch_scannetpp.sh`、`ply.py` 的 subset、`cameras.py` 和 `check_cameras.sh`、nvsmi logger；`grid.json` 的 V1-V6 部分 | V1；nvsmi logger 開始；garden 250 k 子集和兩個 cameras.json；V2 的煙霧格 | V1 |
| 2 | pnpm 安裝（經你同意）、其餘子集和紀錄檔、`run.py` 的 `--mode upstream`、`Watcher`、`gpu_idle.py`、`collect.py` 第一版、`fig_m0_*` | V2、V3、V4、V5、V6（週末，紀錄滿 7 天） | V2、V3、V4、V5、V6；M0 判定寫在 V4 和 V5 裡 |
| 3 | `client/` 骨架（`index.html`、`main.ts`、`app.ts`、`types.d.ts`、`eval.ts` 的複製）、`wire.py` 和 `test_wire.py`、`slot-pool.ts`、`stream-worker.ts`、`apply.ts`、`selftest.ts`、Server 的 `--records` 模式、`compare_png.py`；`grid.json` 的 V7-V12 部分 | selftest、V7、V8、V9 | V7、V8、V9 |
| 4 | `stage-log.ts`、`sh_psnr.py`、`fig_m1_*`、`injector.ts` 開工 | V10、V11、V12 | V10、V11、V12；M1 判定寫在 V9、V10、V11 裡 |
| 5 | `injector.ts`、`burner.html`、runner 的 `--cpu-throttle`、`--burner-iters`、`costmodel.py`、`predict_rmax.py`；`grid.json` 的 V13-V19 部分 | V13、V14、V15 | V13、V14、V15 |
| 6 | 速率階梯邏輯（runner 端）、`h0_verdict.py`、`fig_m2_*` | V16、V17、V18、V19 | V16-V19（H0 判定） |
| 7 | `traces/format.py`、`tstatic.py`、`reveal.py`、`refine.py`、`churn.py`、`beacon.py`、`test_traces.py`、Server 的序列重播、ping/pong、stage-log 的 beacon 列；`grid.json` 的 V20-V25 部分 | V20、V21、V22 | V20、V21、V22 |
| 8 | `beacon-ref.html`、`film.py`、`m3_verdict.py`、`fig_m3_*`；`collect.py --check` 補完；M4 的 Dockerfile 和 `netcheck.py`（只建不量） | V23；V24（有螢幕的機器可用時）；V25 | V23、V24、V25（M3 判定）；M4 的決策項提給你 |

每週結束時：`python results/collect.py --check` 必須是 0，第 2 節的 `git grep` 必須是空的。
平行進行：第 1-2 週讀 GaussAnything、2606.17212、LTS、LMG、Vega（PLAN M0「同時」）。

## 11. LOG.md 模板和 V1 的事先宣告

`LOG.md` 在 V1 跑完時建立，內容是下面的檔頭區塊、第一個條目、分隔線和索引。
條目模板住在檔頭區塊裡（編號規則之後），不放在條目之間，這樣新條目永遠直接加在索引上面，讀的人不用跳過模板。

```markdown
# LOG

這條工作線的前綴是 `V`（virtualization final project），編號 `V1`、`V2`、… 只往上加。
編號是地址不是標籤：用過不重用、不重新編號、撤回也不回收。
撞號時後者加字母（例如 `V2b`），並在這一段註明撞號的原因。
每個條目：問題、確切指令、數字、比較基準（null）、推翻了什麼；只能往後加，後來的條目修正先前的，條目本身永不編輯。
條目在 run 結束後才寫；事先宣告的成功條件、預測和 null 住在 runner 的檔頭和 study 的 grid.json，跑之前 commit，條目引用它們。
失敗的 run 也得到一個條目（成功條件寫「未達」和原因），之後的條目可以取代它，但不改它。
新條目加在最底下「已推翻的說法」索引的**上面**。
探索用的條目（M0-M5）和要發表的條目（M6 事先登記）在標題後面標 `[explore]` 或 `[publish]`。

條目模板：

## V<n> - <日期> - <一句話的問題> [explore|publish]
- Study: experiments/<study>  Runner: run_V<n>_<slug>.sh（檔頭有事先宣告，commit <sha>）
- 問題:
- 成功條件（引用 runner 檔頭）:
- 預測（引用 runner 檔頭或 V14 這類預測條目的 metrics.json）:
- 指令:
- 網格（sweep 才有）: 實際跑的格子、事先宣告的挑選標準、勝出點、其他點的分布、沒跑成的格子
- 數字: n、中位數、四分位距；沒被拿來挑選的指標也列
- Null:
- 推翻: 無 | V<m> 的「...」
- 標記: vsync_mode、camera_validated、excluded 的 run 數、drift
- 成本: run 數、實際耗時
- 輸出: outputs/<study>/V<n>_*/metrics.json；圖 results/figures/<name>.png
- 結果: 成功條件達成 | 未達（原因）

---

## 已推翻的說法索引
| 說法 | 在哪裡說的 | 被哪個條目推翻 |
| --- | --- | --- |
```

V1 的事先宣告已經寫在第 7 節的 runner 檔頭裡，那就是跑之前能寫的全部。
跑完之後 V1 的條目會長這樣（這是形狀的示範，不貼進 LOG.md，數字欄位從 `metrics.json` 抄）：

```markdown
## V1 - <跑的日期> - Linux Chrome 153 的 WebGPU 跑在 NVIDIA 上嗎 [explore]
- Study: experiments/m0_static_null  Runner: run_V1_gpucheck.sh（檔頭有事先宣告，commit <sha>）
- 問題: 這台工作站（t-3dgs-02-linux）的 Playwright Chrome 153.0.8010.12 以 --headless=new 加 Vulkan 啟動時，WebGPU adapter 是 NVIDIA RTX 6000 Ada 還是 SwiftShader？
- 成功條件（引用 runner 檔頭）: adapter.vendor == "nvidia" 且 SystemInfo.getInfo 的 featureStatus 說 WebGPU 是硬體加速。
- 預測（引用 runner 檔頭）: 會是 nvidia。
- 指令: source tools/setup/env.sh && experiments/m0_static_null/run_V1_gpucheck.sh
- 數字: vendor、description、is_hardware、shader-f16、timestamp-query、maxStorageBufferBindingSize、featureStatus 的 WebGPU 鍵和值（全部從 metrics.json 抄）
- Null: 無，第一次量測。
- 推翻: 無。
- 標記: vsync_mode = headless_timer（這台沒有可用的顯示器，見 docs/implementation_plan.md 0.1）。
- 成本: 1 run，實際秒數。
- 輸出: outputs/m0_static_null/V1_gpucheck/{stamp.json, metrics.json}
- 結果: 成功條件達成 | 未達（原因）
```

## 12. 刻意還沒建的東西（YAGNI）和刪減順序

還沒建，以及什麼時候才建：

1. Server 端的合併 dict、准入迴圈、EDF、`TCP_NOTSENT_LOWAT`、A1/A2/A3 開關、client 的 10 Hz 回報：M4 開始才加（PLAN M4、M5），和第一個用到它們的條目同一週。
2. 4 B 的刪除訊息：M4 顯示刪除 bytes > 總量 10% 才加。
3. `stream_id` 改變時清空 slot 池：一個 run 只連線一次，改變就是錯誤；需要重連的情境出現才加。
4. 相機路徑重播（orbit 或錄製的 walk）：M0-M3 全部固定 preset 0；M4 的視角相關 arm 出現才加。
5. `Uint32Array` 取代 id→slot 的 `Map`：V13 的 profile 顯示 Map 在熱路徑上才換。
6. 每則訊息的壓縮（zstd 之類）：不在範圍內，線上 bytes 必須等於 32 × 筆數才能精確設 ρ。
7. Docker 加 Xvfb 的有頭 Chrome：headless 的 vsync 問題用「有人登入主機」或 Windows 解，Docker 只給 M4 的 `tc`；V4 失敗時才重新考慮。
8. 第二個 Chrome 程序當 burner：PLAN 說的是分頁，先做分頁；要比較兩種鄰居才加。
9. GaussAnything 式依剩餘幀時間調整的 k：M5 的 A1 才需要；M1-M3 的 k 是固定值。
10. 第二個真實瀏覽器當 aggressor：H2 用模擬的 guest 程序（PLAN 12 第 8 項），M6 才做。
11. 多 client 的 Server 公平佇列：Path B 才需要。
12. 自動化的 win-desktop 量測：只在 TCP 連得到 9222 時手動做一次 M0，不寫自動流程。
13. 錄影的自動 ROI 偵測：`film.py` 的 ROI 用參數手動給。
14. tile_based 渲染器、morton 排序、SH 1-3 階的串流：固定 `renderer=gaussian`、`sort=none`、`clip_sh_degree=0`。
15. 對上游的任何 patch：目前零個；`export_patches.sh` 和 `patches/` 留著是為了哪天真的需要時流程已經在。

時間不夠時的刪減順序（和 PLAN 11.11 一致，前面補細節）：

1. V12（fp16 代價）和 V15 的細掃，改成單點。
2. V24 攝影機驗證改成只做一次、30 個事件（不到 50），並在 LOG 標明 n 不足。
3. T-reveal 序列（M3 只留 T-refine 兩個版本和 T-churn）。
4. M7，然後 M8，然後 H2（PLAN 的順序）。

M0 到 M3 這四個里程碑和它們的 25 個條目不刪，因為後面每一個數字的 null 都在這裡；V1、V4、V7 三個「儀器能不能用」的條目尤其不能跳。
