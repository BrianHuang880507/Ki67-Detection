# Morphology 預測 IDO 反應實驗計畫

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-16
- Verification Status: CODE VERIFIED；DATA RUN PENDING
- Version Label: implementation_v1

## 實驗概要

- **實驗名稱**：B4 p6 morphology-based IDO response prediction
- **目的**：使用 phase 輔助取得細胞外框／細胞質範圍、DAPI 分割細胞核，從兩者建立細胞形態特徵，預測同一視野中背景校正後的 IDO 螢光反應。
- **主要研究問題**：在 B4 p6 這個批次內，morphology 是否能預測 image-level `IDO_score`？
- **次要研究問題**：控制 IFN-γ／TNF-α 劑量後，morphology 是否仍提供額外預測資訊？
- **假設**：刺激造成的形態變化與 IDO 螢光反應存在可量化關聯，但其關聯不等於已證實的 T-cell suppression。
- **類型**：analysis／training

## 結論邊界

本實驗允許回答：

> 在 B4 p6 技術影像中，細胞 morphology 對 IDO 螢光 proxy 是否具有預測能力。

本實驗不能回答：

- MSC 是否真的抑制 T-cell 活化或增生。
- 模型是否能泛化到其他 donor、lot、passage 或製程。
- IDO proxy 與臨床免疫抑制效果是否相同。

對外報告使用 `Predicted IDO response` 或 `IDO-related immunosuppressive proxy`，不使用 `overall immunosuppressive capacity` 作為目前模型的結果名稱。

## 實驗環境

- **語言**：Python 3.10–3.12
- **主要套件**：Cellpose、NumPy、pandas、SciPy、scikit-image、scikit-learn、matplotlib
- **工作目錄**：`D:/Project/Ki67-Detection`
- **執行入口**：`conda run --no-capture-output -n ki67dtc python -m immunity.run_experiment --config immunity/configs/b4_p6.yaml`
- **狀態**：入口、資料整理、模型、圖表、報告與測試已實作；B4 p6 完整實驗尚待執行

## 輸入資料

### 原始影像

| 輸入 | 路徑 | 說明 |
|---|---|---|
| B4 p6 images | `data/input/B4  p6/` | 240 張 JPG，組成 80 組 phase／DAPI／IDO triplets |

各通道角色固定如下：

- `phase`：協助 Cellpose 取得細胞外框與細胞質範圍。
- `DAPI`：協助 Cellpose 分割細胞核。
- `IDO`：只用於量測 ground truth proxy，不參與 morphology segmentation 或 predictors 計算。

由 phase 得到的細胞外框形成 whole-cell mask；真正的 cytoplasm mask 定義為 `whole-cell mask − nucleus mask`。

### 現有條件

| IFN-γ | TNF-α | Triplets |
|---:|---:|---:|
| 0 | 0 | 9 |
| 0 | 25 | 10 |
| 0 | 50 | 10 |
| 25 | 0 | 9 |
| 25 | 25 | 12 |
| 25 | 50 | 12 |
| 50 | 0 | 9 |
| 100 | 0 | 9 |

目前 IFN×TNF 不是完整 factorial design，因此不對缺少的組合做插值或推測。

## 分析單位

### 主要分析

- 一組 phase／DAPI／IDO triplet 為一個 image-level 樣本。
- 預計樣本數為 80 張配對視野。
- 同一張影像內的細胞先彙整成 morphology summary 與一個 `IDO_score`。

### 次要分析

- 可建立 cell-level sensitivity analysis。
- cell-level cross-validation 必須以 `image_key` 分組；同一張圖的細胞不得同時出現在 train 與 validation。

影像視野屬於同一個 B4 p6 生物批次，因此 80 個視野不是 80 個獨立生物樣本。所有 confidence interval 只代表目前影像的技術不確定性。

## 變數定義

### 預測目標 Y

每顆細胞：

```text
cell_IDO_score
= cytoplasm IDO mean intensity
− non-cell background median intensity
```

每張影像的主要 target：

```text
image_IDO_score = median(cell_IDO_score)
```

保留以下次要 target：

- `IDO_score_IQR`：每張影像內 cell_IDO_score 的 IQR。
- `IDO_positive_fraction`：以 0/0 control 預先定義 threshold 後的陽性細胞比例；僅作 sensitivity analysis。

### Morphology predictors X

主要模型只使用 phase-derived whole-cell／cytoplasm mask 與 DAPI-derived nucleus mask 產生的幾何特徵，不使用 IDO channel 的任何 intensity、texture 或 threshold-derived feature。

第一版特徵集：

- Area
- Perimeter
- Eccentricity
- Extent
- FormFactor，使用 `Sphericity = 4πA/P²`
- MajorAxisLength／MinorAxisLength
- Maximum／Minimum Feret Diameter
- MaximumRadius／MeanRadius／MedianRadius
- AspectRatio
- Perimeter／Area Ratio
- Solidity
- Karyoplasmic Ratio
- Compactness（CellProfiler 相容定義）

以上特徵分別在 nucleus 與 whole-cell mask 計算；每張影像彙整 `median` 與 `IQR`。`Area_cyto` 在目前專案中代表 whole-cell area，不能誤當成排除細胞核後的 cytoplasm area。

以下欄位不放入第一版主要模型：

- IDO intensity 或 IDO texture features
- IFN-γ／TNF-α 劑量
- 由檔名直接推導的 condition label
- Ki67 label 或 Ki67 prediction
- 無法在 train fold 內獨立建立的全資料統計量

劑量只用於分層切分、dose-response 圖及 sensitivity analysis。

## 資料處理流程

### 1. 影像配對與 manifest

1. 解析檔名中的 IFN-γ、TNF-α、channel 與 image index。
2. 將 phase、DAPI、IDO 配成唯一 triplet。
3. 檢查重複、缺少 channel、無法解析及尺寸不一致。
4. 輸出 `data_manifest.csv`，每列代表一個 image triplet。

### 2. Cellpose segmentation

1. 透過 `main.py` 使用 phase 與現有 Cellpose 流程取得細胞外框，建立 whole-cell mask。
2. 使用 DAPI 分割 nucleus mask。
3. 建立 nucleus-to-cell 配對。
4. 以 `whole-cell mask − nucleus mask` 建立 cytoplasm mask。
5. 排除接觸影像邊界、面積異常、沒有 nucleus 或配對失敗的物件。
6. 輸出 segmentation QC 圖與每張圖的保留／排除細胞數。

### 3. 特徵與 IDO_score

1. 對 DAPI-derived nucleus mask 與 phase-derived whole-cell mask 計算 morphology。
2. 對 IDO channel 計算 non-cell background median。
3. 在 cytoplasm mask 內計算每顆細胞的背景校正 IDO_score。
4. 產生 cell-level table。
5. 以影像為單位計算 morphology median／IQR 與 image_IDO_score。

### 4. 描述性 dose-response

繪製三條現有曲線：

- TNF=0：IFN=0／25／50／100。
- IFN=0：TNF=0／25／50。
- IFN=25：TNF=0／25／50。

計算三個 normalized IDO response AUC，但只作 B4 p6 描述性結果：

- `IFN_response_AUC`
- `TNF_response_AUC_IFN0`
- `TNF_response_AUC_IFN25`

因目前只有一個生物批次，這三個 AUC 不作為 machine-learning target。

## 模型設計

### Baseline 0：Dummy model

- `DummyRegressor(strategy="median")`
- 目的：確認模型是否優於只猜整體中位數。

### Baseline 1：Dose-only model

- Predictors：IFN dose、TNF dose、IFN×TNF interaction。
- Model：Ridge regression。
- 目的：量化單靠已知刺激劑量可以預測多少 IDO 變化。

### Primary：Morphology-only model

- Predictors：image-level morphology median／IQR。
- Model：Ridge regression。
- 目的：提供生醫同仁要求的 morphology direct prediction result。

### Secondary：ElasticNet

- 在相同 predictors 與 folds 下執行。
- 目的：檢查結果是否依賴單一 regularization 方法，並產生較稀疏的特徵清單。

### Sensitivity：Dose＋Morphology model

- Predictors：dose variables＋morphology。
- 目的：確認 morphology 在已知劑量後是否還有額外資訊。

第一版不以 Random Forest、XGBoost 或 neural network 作為主要結果，避免在 80 個 image-level 樣本與高維特徵下過度擬合。

## Preprocessing 與 leakage 防護

所有 preprocessing 必須放在 scikit-learn `Pipeline` 中，並只在 train fold fit：

1. 移除全空與常數欄位。
2. Median imputation。
3. StandardScaler。
4. Ridge／ElasticNet hyperparameter selection。

禁止事項：

- 先用全資料做標準化再 cross-validation。
- 先用全資料挑選與 IDO 最相關的特徵。
- 將同一 image 的 cells 隨機拆到 train 與 validation。
- 把 IFN／TNF 檔名資訊混進 morphology-only predictors。
- 把 IDO intensity 或 IDO-derived feature 混進 predictors。

## Cross-validation

### Primary validation

- 5-fold cross-validation。
- 以八種 condition label 做 fold 分層，使每個 fold 盡量包含各條件。
- 重複多個固定 random seeds，保存所有 out-of-fold predictions。

### Robustness validation

- Leave-one-condition-out，共八個 folds。
- 目的：檢查 morphology-only 模型面對未參與訓練的刺激組合是否仍有預測能力。

### 未來多批次

累積多個 donor／lot／passage 後，改成 Leave-one-donor 或 Leave-one-lot-out；屆時才建立 morphology → IDO response AUC 模型。

## 評估指標

### Primary metric

- Out-of-fold MAE。

### Secondary metrics

- R²。
- Spearman correlation。
- RMSE。
- 各 IFN／TNF condition 的 MAE 與 residual distribution。

### Exploratory success criterion

同時符合以下條件才稱為「目前資料存在 morphology predictive signal」：

1. Morphology-only model 的 repeated-CV MAE 穩定低於 Dummy model。
2. Out-of-fold Spearman correlation 方向穩定為正。
3. 結果不是由極少數影像或單一 condition 主導。
4. segmentation QC 與 IDO background QC 沒有系統性失敗。

模型完成 out-of-fold image-level IDO_score 預測後，會再依三條現有劑量曲線計算 predicted IDO dose-response AUC，作為可交付的下游摘要；AUC 不會反過來當成本批模型的訓練 target。

另外比較 morphology-only 與 dose-only model。若 morphology-only 只能在隨機切分下表現良好，但 leave-one-condition-out 失效，結論應寫成「morphology 主要反映既有刺激條件」，而不是可泛化的 IDO predictor。

## 預期輸出

| 輸出 | 預計路徑 | 格式 | 完成條件 |
|---|---|---|---|
| Data manifest | `immunity/outputs/b4_p6/data_manifest.csv` | CSV | 80 組 triplets 唯一且 channel 完整 |
| Cell-level data | `immunity/outputs/b4_p6/cell_level_features.csv` | CSV | 每列為一個通過 QC 的細胞 |
| Image-level data | `immunity/outputs/b4_p6/image_level_dataset.csv` | CSV | 每列為一張影像，包含 X、Y、condition |
| Dose summary | `immunity/outputs/b4_p6/dose_response_summary.csv` | CSV | 三組曲線及 normalized AUC 可追溯 |
| CV metrics | `immunity/outputs/b4_p6/cv_metrics.csv` | CSV | 所有模型與 folds 指標完整 |
| OOF predictions | `immunity/outputs/b4_p6/oof_predictions.csv` | CSV | 每張影像恰有對應預測 |
| Feature coefficients | `immunity/outputs/b4_p6/feature_coefficients.csv` | CSV | Ridge／ElasticNet coefficients 可追溯 |
| Observed vs predicted | `immunity/outputs/b4_p6/figures/observed_vs_predicted.png` | PNG | 顯示 OOF predictions，不顯示 training fit |
| Dose-response curves | `immunity/outputs/b4_p6/figures/dose_response.png` | PNG | 顯示三條現有曲線與變異 |
| Predicted AUC | `immunity/outputs/b4_p6/predicted_dose_response_auc_summary.csv` | CSV | 由 OOF predictions 重建曲線後計算，並列 observed AUC 誤差 |
| Result report | `immunity/outputs/b4_p6/REPORT.md` | Markdown | 清楚區分證據、限制與未驗證推論 |

## 生醫同仁交付結果

第一版交付不只提供一個模型分數，而是四個結果：

1. Morphology-only out-of-fold MAE、R² 與 Spearman correlation。
2. Dummy、dose-only、morphology-only、dose＋morphology 的公平比較。
3. Observed-vs-predicted 圖及最穩定 morphology coefficients。
4. 三組 IDO dose-response curve 與描述性 AUC。

如果需要單一顯示分數，可以將 out-of-fold predicted IDO 值轉成 B4 p6 內部的 0–100 `Relative IDO Response Index`。該指數只表示此批影像內的相對位置，不是絕對免疫力，也不能跨批次直接比較。

## 主要限制與風險

| 風險 | 影響 | 處理方式 |
|---|---|---|
| 只有 B4 p6 一個生物批次 | 無法證明跨 donor／lot 泛化 | 結果標示 exploratory／within-lot |
| 80 個視野為技術樣本 | 容易誤當成生物重複 | 主要分析以 image 為單位，明確揭露 grain |
| 缺少 well ID | 無法正確估計 well-level variation | 保留檔案 index；不假設它代表 biological replicate |
| JPG 8-bit 且缺少曝光 metadata | IDO 強度可能受 acquisition 影響 | 做 background QC；結果限於同批影像 |
| IFN×TNF 組合不完整 | 無法估計完整 interaction surface | 只分析現有三條一維曲線 |
| Dose 同時改變 morphology 與 IDO | 可能產生共同原因相關 | 比較 dose-only、morphology-only、combined 與 LOCO |
| 細胞分割錯誤 | 同時污染 X 與 Y | 排除邊界／配對失敗物件並輸出 QC 圖 |
| 特徵數相對樣本數過多 | 過度擬合 | 限定 paper-inspired geometry、regularization、nested preprocessing |

## Monitoring Configuration

- **預計 timeout**：4 hours
- **監控檔案**：`immunity/outputs/b4_p6/run.log`、`cv_metrics.csv`
- **實驗類型**：analysis／training
- **主要 metric key**：`oof_mae`
- **失敗條件**：triplet 不完整、有效影像少於預期、target 全空、fold 無法涵蓋條件、模型產生非有限 predictions

## 執行順序

```text
建立 manifest 與配對 QC
        ↓
Cellpose segmentation 與 segmentation QC
        ↓
計算 paper-inspired morphology
        ↓
計算背景校正 cell／image IDO_score
        ↓
建立 image-level dataset
        ↓
計算描述性 dose-response 與三個 AUC
        ↓
執行 Dummy／Dose／Morphology／Combined models
        ↓
產生 out-of-fold metrics、圖與 coefficients
        ↓
撰寫 B4 p6 exploratory result report
```

## 未來升級條件

當取得多個獨立 donor／lot／passage，且每批皆有相同影像條件後：

1. 每個獨立批次計算自己的三個 IDO response AUC。
2. 將一個 donor／lot／passage 視為一筆 machine-learning sample。
3. 使用 Leave-one-donor／lot-out validation。
4. 建立 morphology → IDO response AUC 模型。
5. 最後以 PBMC／T-cell assay 驗證，才評估能否稱為免疫抑制能力預測。
