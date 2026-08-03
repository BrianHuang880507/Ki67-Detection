# GUI 預設值、運算裝置選項與 cleaned 保留範圍

日期：2026-07-31

## 目標

三項互相獨立的調整：

1. GUI 分析選項的預設值改為「核來源 = PC」「分析方法 = Python」，CLI 預設同步。
2. GUI 與 CLI 新增運算裝置（GPU / CPU）選擇，讓無 GPU 或需要排除 GPU 變因的機器可以明確指定 CPU。
3. `*_cleaned.csv` 與 `*_cleaned.xlsx` 保留 `cyto_only`、`nuc_cut`、`both_cut` 三種狀態。

三項可獨立實作與獨立驗證，沒有先後相依。

## 變更一：預設值

| 位置 | 現值 | 改為 |
|---|---|---|
| `ki67dtc/gui/main_window.py:331` `_nuc_source` | `"dapi"` | `"pc"` |
| `ki67dtc/gui/main_window.py:333` `_feature_backend` | `"pyimagej"` | `"python"` |
| `main.py:35` `--nuc_source` default | `"dapi"` | `"pc"` |
| `main.py:51` `--feature_backend` default | `"pyimagej"` | `"python"` |
| `scripts/run_all_data_input.py:50` `--nuc_source` default | `"dapi"` | `"pc"` |
| `scripts/run_all_data_input.py:66` `--feature_backend` default | `"pyimagej"` | `"python"` |

`run_all_data_input.py` 必須一起改：`build_command()`（`scripts/run_all_data_input.py:169-182`）永遠會把 `--nuc_source` 與 `--feature_backend` 明確加進命令列，只改 `main.py` 的話批次執行仍會被覆寫回舊值。

`ki67dtc/app_pipeline.py:88,92` 的 `run_pipeline` 預設值維持不變 —— GUI 一律明確傳值，這兩個預設只在直接呼叫函式時生效，改動沒有效益且會擴大影響面。

對應的 `help` 字串（`main.py:37,53`）要一併更新，否則說明文字會與實際預設不符。

## 變更二：運算裝置選項

### 型別分界

UI 與 CLI 層使用字串 `"gpu"` / `"cpu"`，與既有的 `nuc_source`、`ki67_backend`、`feature_backend` 一致；`img_prep` 層使用 `use_gpu: bool`，與 Cellpose 的 `CellposeModel(gpu=...)` 一致。字串轉 bool 只發生在兩個入口：`run_pipeline()` 與 `run_preprocessing()`。

這樣 `img_prep` 不需要認得 UI 的字彙，UI 也不需要處理布林值與下拉選單的對應。

### 傳遞鏈

```
GUI: _device ("gpu"/"cpu")
  → PipelineThread(device=)
  → run_pipeline(device=)          ← 字串轉 bool
  → segment_all(use_gpu=)
  → segment(use_gpu=)
  → CellposeModel(gpu=use_gpu)     ← ki67dtc/img_prep.py:397

CLI: args.device ("gpu"/"cpu")
  → run_preprocessing(device=)     ← 字串轉 bool
  → segment_all(use_gpu=)
  → （同上）
```

### 逐檔變更

**`ki67dtc/img_prep.py`**
- `segment()` 簽章加 `use_gpu: bool = True`，`models.CellposeModel(gpu=True, ...)`（`img_prep.py:397`）改為 `gpu=use_gpu`。
- `segment_all()` 簽章加 `use_gpu: bool = True`，透傳給**六處** `segment()` 呼叫（`img_prep.py:687, 701, 724, 747, 810, 826`）。漏掉任何一處，該分支就會靜默地回到 GPU，因此實作後需逐一核對。

**`ki67dtc/gui/main_window.py`**
- 新增狀態 `self._device: str = "gpu"`（放在 `main_window.py:331` 的 `_nuc_source` 之前，順序與對話框一致）。
- `_build_analysis_options_dialog()`（`main_window.py:515`）在 form 的**第一列**插入「運算裝置」下拉，`objectName = "deviceCombo"`，選項 `[("GPU", "gpu"), ("CPU", "cpu")]`。
- `_apply_analysis_options_dialog()`（`main_window.py:601`）讀取 `deviceCombo` 寫回 `self._device`。
- `PipelineThread.__init__()`（`main_window.py:206`）加 `device: str` 參數並存為 `self._device`；`run()`（`main_window.py:251`）傳入 `run_pipeline(device=self._device)`。
- 建立執行緒的呼叫點 `PipelineThread(...)`（`main_window.py:935`）補上 `device=self._device`。

**`ki67dtc/app_pipeline.py`**
- `run_pipeline()` 簽章加 `device: str = "gpu"`，docstring 補說明。
- `segment_all(data_folder, nuc_source=nuc_source)`（`app_pipeline.py:130`）改為一併傳 `use_gpu=(device != "cpu")`。

**`main.py`**
- 新增 `--device`，`choices=["gpu", "cpu"]`，`default="gpu"`。
- `run_preprocessing()` 簽章加 `device: str`，內部 `segment_all(data_folder, nuc_source=nuc_source, use_gpu=(device != "cpu"))`。
- 呼叫點 `run_preprocessing(data_folder, args.nuc_source)`（`main.py:117`）補傳 `args.device`。
- 資訊區塊（`main.py:106-115`）加一行印出運算裝置。

**`scripts/run_all_data_input.py`**
- 新增 `--device`，`choices=["gpu", "cpu"]`，`default="gpu"`。
- `build_command()`（`scripts/run_all_data_input.py:169`）把 `--device` 加進轉發清單。

### 為什麼是二選一而不是三選一

`CellposeModel(gpu=True)` 在沒有 CUDA 時會自行退回 CPU，因此「自動」與「GPU」在無 GPU 機器上行為完全相同。多一個「自動」選項只會讓使用者誤以為兩者有差別。二選一直接對應 `gpu=True/False`，沒有語意重複的狀態。

## 變更三：cleaned 保留範圍

`ki67dtc/utils/io.py:317-319` 的保留清單由三種擴為六種：

```python
keep_mask = merged_df["cell_status"].isin(
    ["full_cell", "nuc_only", "cyto_cut", "cyto_only", "nuc_cut", "both_cut"]
)
```

`empty` 與 `unknown` 仍然移除 —— 這兩者代表輪廓分類失敗或缺 `Cell_ID` 對應，不是有效的細胞判定結果。

`*_cleaned.xlsx` 不需另外修改：`ki67dtc/workbook_export.py` 讀的就是同一份 cleaned CSV，且只把 `cell_status` 當 meta 欄位寫進 `Cell_type`（`workbook_export.py:19,714`），沒有自己的狀態篩選。

### 下游影響（已查證）

| 位置 | 影響 |
|---|---|
| `ki67dtc/cell_anal_plot.py:33` | 以 `isfinite & > 0` 過濾面積，新增列的 NaN 自動排除，散點圖與長條圖不受影響 |
| `immunity/build_dataset.py:458` | 自行再篩 `full_cell`，不受影響 |
| `analysis/ki67_pred_utils.py:34` | 以 `*_cleaned.csv` 為訓練資料來源，會混入缺 `_nuc` 或缺 `_cyto` 特徵的列 |

第三項本次**不處理**（使用者決定）。現況 cleaned.csv 已有 19.5% 為 `nuc_only`（110 個 `_cyto` 欄全為 NaN），本次變更會再擴大這個比例。若日後預測結果異常，這裡是第一個要檢查的地方。

## 測試

**新增**
- `merge_all_final_csvs()` 目前沒有任何測試。補測試涵蓋：六種狀態全數保留、`empty` 與 `unknown` 被移除、缺 `cell_status` 欄位時走舊的 `dropna` 分支。
- `segment_all(use_gpu=False)` 會讓 `CellposeModel` 收到 `gpu=False`。`tests/test_img_prep_resize.py:24` 的 fake model 已記錄 `gpu` 屬性，可直接沿用。

**修改 `tests/test_main_window_layout_contract.py`**
- `test_analysis_options_dialog_exposes_expected_controls`（第 63 行）：清單加入 `deviceCombo`。
- `test_analysis_options_dialog_exposes_backend_values`（第 80 行）：`option_groups` 加入 `"deviceCombo": [("GPU", "gpu"), ("CPU", "cpu")]`。
- `test_analysis_options_dialog_applies_pipeline_values`（第 99 行）：目前把 `nucSourceCombo` 設為 `"pc"`、`featureBackendCombo` 設為 `"python"` 後斷言（第 102-104、129-131 行）。改預設值之後這兩個值會與預設相同，測試就不再能證明「套用」真的有作用。**必須把這兩個改設為非預設值**（`"dapi"`、`"pyimagej"`）並同步調整斷言，同時把 `deviceCombo` 設為 `"cpu"` 加入驗證。
- 新增一個測試斷言對話框預設值：`_device == "gpu"`、`_nuc_source == "pc"`、`_feature_backend == "python"`，並確認「運算裝置」是 form 的第一列。

**回歸**
- `python -m pytest tests/ -q` 全綠。

### 變更三的資料層驗證

改完之後，每個資料集的 `sum(*_final.csv 筆數)` 應等於該資料集 `*_cleaned.csv` 的筆數。

已對 73 個有 `*_final.csv` 的現有資料集實測（把真實 final CSV 複製到暫存目錄後跑改動後的 `merge_all_final_csvs()`，不覆寫既有輸出），確認等式**精確成立**：

| 項目 | 筆數 |
|---|---|
| `sum(*_final.csv)` | 261,127 |
| 改動後 `*_cleaned.csv` | 261,127 |
| 差異 | **0** |
| 改動前 `*_cleaned.csv` | 201,389（少 59,738 筆） |

現有資料裡沒有任何一列是 `empty` 或 `unknown`，所以擴為 6 種等同於完全不過濾。若日後這個等式不成立，差額就是 `empty` / `unknown` 的筆數 —— 那代表輪廓分類出了問題，值得追查而不是放寬清單。

## 不在範圍內

- `analysis/ki67_pred_utils.py` 的訓練資料篩選（使用者明確決定先不管）。
- `_filter_small_labels` 與配對過濾邏輯的任何調整。
- 同仁機器雜訊問題的後續診斷（等待對方環境資料，另案處理）。
- 選項的持久化：目前所有分析選項都只存在記憶體，重開 GUI 即還原預設。本次維持相同行為。
