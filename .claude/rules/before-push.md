# Push 前的檢查

這個 repo 推出去的東西要讓組員很快看懂，所以每次 push（包括 PR 的 branch）之前都先做這個檢查。

1. 列出 push 之後 tip 上的每個檔案：`git ls-tree -r --name-only HEAD`。
2. 逐一問「組員需要它嗎」。以下四類不推：
   - 不是最終結果的實驗：被後來的實驗取代的、失敗的、中間的 ablation；
   - 報告沒有用到的圖；
   - 現有程式能重新產生的結果（`.json`、`.jsonl`、log、表格）：README 寫哪個指令會產生它；
   - 沒有任何 runner、`reproduce.sh` 或報告會呼叫的腳本。
3. 拿掉檔案之前，先在當下的 commit 打 tag 並推送，README 寫 tag 名稱，之後才找得回來。
4. 拿掉之後，用 `rg` 確認留下的檔案沒有指向被拿掉的檔案。
5. push 之前告訴使用者：tip 上有幾個檔案、拿掉了哪些、為什麼。

我們假設每位協作者都有 Claude Code 或其他 agent，能照 README 的指令重新產生結果。
在這個 repo，這條規則取代 research-discipline 的「`results/` 要追蹤」：報告的數字記在 LOG.md，原始輸出由 runner 寫到 `outputs/`，不追蹤。
