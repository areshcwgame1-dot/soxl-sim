# SOXL 對角單蒙地卡羅模擬（Rust + GitHub Actions）

啟用步驟：
1. Settings → Pages → Deploy from a branch → `main` / `/docs`
2. Settings → Actions → General → Workflow permissions 選 Read and write
3. Actions 頁籤手動執行 `Daily SOXL MC sim`

日常：IV 有更新時，直接在網頁編輯 config.txt 新增 `iv = 日期 各腿IV`。
收盤價由排程自動更新。到期日／停損參數過期時自行更新。
改了出場規則或分類邏輯後，手動執行 `Kani verification`。
