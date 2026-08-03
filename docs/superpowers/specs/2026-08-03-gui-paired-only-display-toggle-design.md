# GUI 核質配對顯示開關設計

日期：2026-08-03

## 背景

目前 GUI 的 segmentation overlay 固定採用 Paired Only 顯示：以細胞核中心點
所在的細胞質 label 建立核質配對，只呈現成功配對的 nucleus 與 cytoplasm。
這項行為適合檢視有效細胞，但使用者無法在同一畫面比較未配對的 segmentation
物件。

現有 pipeline 會在 segmentation 完成後、清理暫存檔案前建立
`PairedOverlayData`，因此即使預設啟用「清理暫存檔案」，GUI 仍可顯示配對結果。
新的開關必須沿用這個時機快取全部顯示資料，且不得改變 segmentation、分析或
輸出內容。

## 目標

1. 在 GUI overlay 控制列加入「僅顯示核質配對」checkbox。
2. 預設維持目前的 Paired Only 顯示。
3. 關閉 checkbox 時顯示全部 segmentation nucleus 與 cytoplasm 物件。
4. 切換只重新繪製 GUI，不重新執行 pipeline，也不改變任何輸出檔案。
5. 即使 pipeline 清除 segmentation 暫存檔，當次分析完成後仍可切換顯示模式。

## 非目標

- 不調整 Cellpose 或配對感知小面積過濾。
- 不修改核質配對規則、`cell_status` 或 cleaned CSV 篩選規則。
- 不讓 GUI 顯示模式影響 CSV、XLSX、NPY 或自動輸出的 overlay PNG。
- 不新增顯示模式的永久設定；每次啟動 GUI 仍以 Paired Only 為預設。

## 採用方案

採用「影像控制列 checkbox + 精簡區域快取」。在 pipeline 尚持有完整 label
masks 時，將每個 label 轉成裁切後的布林遮罩與全圖輪廓。這比為每張影像保留
兩張完整 label masks 更節省記憶體，也能在暫存 NPY 被刪除後繼續切換顯示。

不採用以下方案：

- 完整 masks 快取：實作直接，但大型資料集的記憶體占用較高。
- 切換時從 NPY 讀取：預設清理暫存檔案後無法使用，不符合需求。

## GUI 行為

### 控制項

在 `ki67dtc/gui/main_window.py` 的影像標題控制列加入 checkbox：

- 顯示文字：`僅顯示核質配對`
- `objectName`：`pairedOnlyCheck`
- 位置：接在「顯示核輪廓」與「顯示質輪廓」之後、Ki67 顯示控制之前
- 預設：勾選

### 切換規則

- 勾選：只繪製成功配對的 nucleus／cytoplasm。
- 取消勾選：繪製完整 segmentation 中的所有 nucleus／cytoplasm label。
- 切換時只呼叫目前影像的重繪流程。
- 原有核顯示、質顯示、Ki67、透明度、縮放與 Cell 高亮狀態保持有效。
- 載入新資料集或重設 GUI 時，恢復預設的 Paired Only。

## 顯示資料結構

延伸 `ki67dtc/paired_overlay.py` 的 `PairedOverlayData`，在既有 `pairs` 與數量
統計之外，加入每個唯一 label 的精簡區域：

- 全部 cytoplasm `LabelRegion`
- 全部 nucleus `LabelRegion`

`LabelRegion` 已包含裁切後布林 mask、全圖座標偏移與 contours，可直接沿用。
建立資料時，每個 label 只抽取一次；配對資料與全部資料引用相同的 region，避免
重複建立同一物件。

## 資料流程

```text
Cellpose segmentation masks
  -> build_paired_overlay_data()
       -> pairs（Paired Only）
       -> all nucleus regions
       -> all cytoplasm regions
  -> PipelineResult.paired_overlays（記憶體快取）
  -> GUI 依 pairedOnlyCheck 選擇 renderer
```

`run_pipeline()` 維持在 segmentation 完成後立即收集 overlay 資料。後續
`clean_temp=True` 即使刪除 NPY，也不會影響當次 GUI 的兩種顯示模式。

若使用者直接開啟既有資料夾，且兩張 segmentation NPY 仍存在，GUI 可從 masks
即時建立相同顯示資料；不需要重新執行分析。

## 繪製規則

Paired Only renderer 保留目前行為與預設參數。另提供全部物件的 renderer：

- cytoplasm 依穩定的 label 順序循環使用既有色盤。
- nucleus 沿用既有藍色填色與輪廓。
- `show_nucleus`、`show_cytoplasm` 與 `alpha` 在兩種模式都生效。
- renderer 不得修改輸入影像、label masks 或快取資料。

GUI 根據 checkbox 選擇 renderer；輸出 overlay 的函式不接收 GUI 顯示模式，固定
呼叫 Paired Only renderer。

## 資料不足與錯誤處理

GUI 需要同時具備 nucleus 與 cytoplasm 的完整 segmentation 顯示資料，才能取消
Paired Only。若 NPY 已清除、損壞或缺少其中一張：

1. 當前影像維持 Paired Only。
2. checkbox 顯示為勾選並停用。
3. 不顯示阻斷操作的 modal dialog。
4. 狀態訊息顯示「缺少完整 segmentation 資料，僅能顯示核質配對」。

GUI 另保存使用者最近一次可用的顯示偏好。切換到具有完整資料的影像時，
checkbox 重新啟用並恢復該偏好。更新 checkbox 的可用狀態時需阻擋 Qt signal，
避免程式性勾選觸發額外重繪或覆蓋偏好。

只有 outline fallback、沒有完整 masks 或全部區域快取時，不嘗試把 paired outline
誤當成全部 segmentation。

## 輸出隔離

下列內容不得讀取 `pairedOnlyCheck` 或 GUI 顯示偏好：

- segmentation NPY
- 每張影像的 final CSV
- cleaned CSV 與 XLSX
- 面積分析圖
- `save_pipeline_fill_overlays()` 產生的 overlay PNG

`save_pipeline_fill_overlays()` 繼續固定輸出 Paired Only，確保相同輸入與分析設定
產生的輸出不會因使用者切換 GUI 顯示而改變。

## 測試

### `tests/test_paired_overlay.py`

- 建立顯示資料時包含已配對與未配對的唯一 label regions。
- Paired Only renderer 不繪製未配對物件。
- 全部物件 renderer 會繪製未配對 nucleus／cytoplasm。
- 兩種 renderer 都遵守核、質與透明度設定，且不修改輸入資料。
- `save_pipeline_fill_overlays()` 在 GUI 顯示偏好為全部物件時仍輸出 Paired Only。

### `tests/test_main_window_layout_contract.py`

- `pairedOnlyCheck` 位於影像標題控制列，且預設勾選。
- 切換 checkbox 只改變畫面 renderer，不啟動 pipeline。
- 有完整資料時可切換；缺少資料時停用並顯示 Paired Only。
- 從缺少資料的影像切回完整影像時，恢復最近一次可用的顯示偏好。
- 既有核、質、Ki67、透明度與高亮控制沒有回歸。

### 驗證命令

```powershell
python -m unittest tests.test_paired_overlay -v
python -m unittest tests.test_main_window_layout_contract -v
python -m unittest tests.test_pipeline_device_option -v
```

## 驗收條件

1. GUI 啟動後 checkbox 預設勾選，畫面與目前版本一致。
2. 當完整顯示資料可用時，取消勾選會立即顯示未配對物件。
3. 再次勾選會立即恢復 Paired Only。
4. 清理暫存檔案後，當次 pipeline 的顯示切換仍可使用。
5. 缺少完整資料時不會誤顯示或中斷 GUI。
6. 切換前後所有輸出檔案內容與路徑保持一致。
7. 相關測試全部通過。
