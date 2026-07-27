# Immunity Analysis

> 本資料夾用於存放 MSC morphology 與 IDO 反應相關的分析、模型、測試及實驗文件。

## 目前目標

使用現有 `B4 p6` 的 phase、DAPI 與 IDO 影像，建立 morphology-only 模型，預測背景校正後的 `IDO_score`。其中 phase 用於取得細胞外框／細胞質範圍，DAPI 用於分割細胞核，IDO 只用於建立預測目標。

本階段的結果定位為：

> 細胞形態對 IDO-related immunosuppressive proxy 的探索性預測。

在尚未使用 PBMC／T-cell functional assay 驗證前，不將模型輸出稱為真正的免疫抑制能力。

## 資料

- 輸入資料：`data/input/B4  p6/`
- 影像通道：`phase`（細胞外框／細胞質分割）、`DAPI`（細胞核分割）、`IDO`（target measurement）
- 刺激條件：IFN-γ 0／25／50／100，以及部分 TNF-α 0／25／50 組合
- 目前生物批次：B4 p6，一個 lot／passage

## 功能

- 依檔名解析 IFN-γ／TNF-α 劑量並嚴格配對 phase、DAPI、IDO triplets。
- 使用 phase 的 Cellpose 輪廓計算 whole-cell morphology，使用 DAPI 輪廓計算 nucleus morphology。
- 以 `whole-cell mask − nucleus mask` 量測細胞質 IDO，建立背景校正 `IDO_score`。
- 將每張影像內的 morphology 彙整為 median 與 IQR。
- 比較 Dummy、dose-only、morphology-only、ElasticNet 及 dose＋morphology models。
- 執行 condition-stratified repeated CV 與 leave-one-condition-out validation。
- 計算三條描述性 IDO dose-response 曲線與 AUC，輸出圖表及中文報告。

## 執行

### 環境

- Python 3.10–3.12
- Cellpose 3.1.1.1
- scikit-image、scikit-learn、pandas、SciPy、matplotlib
- 建議使用本機既有的 `ki67dtc` Conda 環境與 GPU。

### 1. 完整影像分析

在專案根目錄執行：

```powershell
conda run --no-capture-output -n ki67dtc python main.py --data_folder "data/input/B4  p6" --nuc_source dapi --fluor_analy --feature_backend python --clean_temp --xlsx-version both
```

這一步會執行 Cellpose、完整 morphology／texture／intensity 特徵與 IDO 量測，產生 `data/output/results/B4  p6/B4  p6_cleaned.csv`，可供 immunity 與其他應用重用。

### 2. Immunity 模型與 AUC

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.run_experiment --config immunity/configs/b4_p6.yaml
```

這一步只讀取 cleaned CSV、挑選 morphology predictors、執行 repeated CV／LOCO，並由 out-of-fold IDO predictions 重建 dose-response curve 與 predicted AUC，不會再次執行 Cellpose。

`--no-capture-output` 用來保留 Conda 所需 DLL 環境，同時避免 Windows CP950 無法輸出部分中文字元。

### 測試

```powershell
conda run --no-capture-output -n ki67dtc python -m pytest -q
```

## 資料夾說明

```text
immunity/
├── README.md
├── EXPERIMENT_PLAN.md
├── configs/
│   └── b4_p6.yaml           # B4 p6 資料、分割、QC 與模型參數
├── outputs/                 # 執行後的模型、表格、圖及報告
├── __init__.py
├── build_dataset.py         # 影像配對、morphology、IDO target 與 AUC
├── train_ido_proxy.py       # 模型、cross-validation 與圖表
└── run_experiment.py        # 完整實驗入口與報告輸出
```

測試放在專案共用的 `tests/test_immunity_experiment.py`。

## 主要輸出

執行結果預設存於 `immunity/outputs/b4_p6/`：

- `data_manifest.csv`：80 組 triplets 與刺激條件。
- `cell_level_features.csv`：每顆通過 QC 細胞的 morphology 與 IDO_score。
- `image_level_dataset.csv`：模型實際使用的一圖一列資料。
- `dose_response_auc.csv`：三個描述性 IDO dose-response AUC。
- `predicted_dose_response_auc_summary.csv`：由 OOF predictions 推導的 predicted AUC 與 observed AUC 誤差。
- `cv_metrics_summary.csv`：模型 repeated CV 與 LOCO 指標。
- `oof_predictions.csv`：每張影像的 out-of-fold prediction。
- `feature_coefficients.csv`：Ridge／ElasticNet coefficients。
- `models/`：以全部 B4 p6 影像 fit 的探索性模型。
- `figures/`：observed-vs-predicted 與 dose-response 圖。
- `REPORT.md`：白話結果、證據邊界與限制。
- `run.log`：完整執行紀錄。

## 實驗計畫

完整方法、資料切分、評估指標與結果解讀限制，請見 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。
