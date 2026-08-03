# GUI merged outlines 雙模式顯示設計

## 目標

讓舊資料即使已清除 `*_nuc_seg.npy` 與 `*_cyto_seg.npy`，只要仍保留
`*_merged_cp_outlines.txt`，GUI 就能切換：

- 僅顯示核質配對。
- 顯示全部 nucleus 與 cytoplasm segmentation 物件。

這項功能只影響 GUI 預覽，不改寫任何既有檔案，也不改變分析與自動輸出。

## 現況與可用資料

Pipeline 會將 merged outlines 以兩行一組保存：第一行是 nucleus，第二行是
cytoplasm。缺少其中一側時以 `-1,-1` 佔位。因此同一個檔案已同時保存：

- nucleus 與 cytoplasm 都存在的配對物件。
- 只有 nucleus 的未配對物件。
- 只有 cytoplasm 的未配對物件。

目前 GUI 在缺少 NPY 時只呼叫 `load_paired_merged_outlines()`，會刻意略過含
`-1,-1` 的記錄，並停用「僅顯示核質配對」開關。未配對輪廓仍存在於檔案中，
不需要重新執行 segmentation，也不需要新增 sidecar 檔。

## 採用方案

直接從同一份 merged outlines 建立兩組唯讀 polygon 集合：

- `paired polygons`：只包含核與質兩側都存在的記錄。
- `all polygons`：包含每筆記錄中實際存在的 nucleus 或 cytoplasm polygon。

不從 polygon 重建完整 label mask，避免額外記憶體、填滿規則差異與相鄰物件
互相覆蓋。也不新增壓縮快取，確保現有舊資料可以立即使用。

## 元件與責任

### Outline loader

沿用 `load_paired_merged_outlines()` 取得配對 polygons，並強化
`load_merged_outlines()` 的安全解析後用它取得全部 polygons。兩者必須：

- 將 merged 檔視為固定的 nucleus／cytoplasm 兩行記錄。
- 忽略 `-1,-1`，但保留另一側有效 polygon。
- 對無法解析、點數不足或奇數座標的行採安全降級，不讓 GUI 當機。
- 不修改來源檔案。

### MainWindow 顯示快取

保留現有 `_current_overlay_polygons` 作為配對 polygons 與 cell highlight 的索引來源，
另外保存每張影像的全部 polygons。分開保存可避免未配對物件改變 `Cell_ID` 與
highlight 的既有索引對應。

### Renderer 選擇

顯示來源優先順序固定如下：

1. Pipeline 記憶體中的 `PairedOverlayData`。
2. 從 NPY 建立的 `PairedOverlayData`。
3. merged outlines 的 paired／all polygons。
4. 無可用 segmentation 資料時只顯示原圖。

使用 NPY 或記憶體 regions 時維持目前 renderer。只有落到 merged outlines 時，
GUI 才依 `_paired_only_preference` 選擇 paired polygons 或 all polygons，再交給既有
outline renderer。顏色、透明度、nucleus／cytoplasm 顯示開關維持一致。

## 開關狀態

- 完整 NPY／記憶體 regions 可用：啟用開關。
- NPY 不可用，但 merged outlines 是可解析的兩行記錄：啟用開關。
- merged 檔只有配對記錄、沒有未配對物件：仍啟用；兩種模式畫面可以相同。
- NPY 與有效 merged outlines 都不可用：勾選並停用，顯示非阻斷狀態訊息。
- 載入新資料集、Run 或 Reset：仍恢復 Paired Only 預設。
- Pipeline failed／stop：依目前影像的 regions、完整 masks 或有效 merged outlines
  恢復開關 availability。

## 錯誤處理

- merged 檔不存在：沿用目前缺少 outlines 的提示。
- merged 檔為空、記錄不完整或沒有任何有效 polygon：視為無完整顯示資料，停用開關。
- 部分行損壞：略過損壞 polygon；其餘有效記錄仍可顯示。若沒有任何有效 polygon，
  才降級為無資料。
- 讀檔錯誤不得中止 GUI，也不得觸發 pipeline。

## 輸出隔離

以下路徑不得讀取 GUI 的 `_paired_only_preference`：

- CSV、cleaned CSV 與 XLSX。
- NPY、面積圖與分析結果。
- `save_pipeline_fill_overlays()` 產生的 overlay PNG。

自動 overlay PNG 繼續使用 `render_paired_overlay_bgr()`；若走 outline fallback，
也只能使用 paired polygons。新增的 all-polygons 快取只供 GUI 預覽。

## 測試策略

1. 建立不含 NPY、但 merged 檔同時含配對與兩種未配對記錄的舊資料 fixture。
2. 驗證開關啟用，Paired Only 不顯示未配對像素，取消勾選後會顯示。
3. 驗證切換不啟動 pipeline、不修改 merged 檔，且 cell highlight 狀態不變。
4. 驗證空檔、損壞行與缺檔會安全降級，不讓 GUI 當機。
5. 驗證 NPY 路徑仍優先，自動 overlay PNG 與既有輸出維持 Paired Only。

## 驗收條件

- 現有含有效 `*_merged_cp_outlines.txt` 的舊資料，即使沒有任何 `*_seg.npy`，
  仍可切換 Paired Only／全部 segmentation。
- 未配對 nucleus 與 cytoplasm 只在取消勾選時出現在 GUI。
- 開關文字、預設值與新資料集／Run／Reset 行為不變。
- 所有輸出檔案內容與命名不受 GUI 顯示偏好影響。
- 既有與新增測試全部通過，且不提交使用者目前的其他工作樹修改。

## 非目標

- 不從只有原圖或 overlay PNG 的資料重建 segmentation。
- 不改變核質配對規則、Cellpose、面積過濾或 `cell_status`。
- 不修改 merged outlines 格式，也不批次轉換舊資料。
