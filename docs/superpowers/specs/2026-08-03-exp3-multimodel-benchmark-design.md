# Exp3 Phase-only Morphology 多模型 Benchmark 設計

## 文件狀態

- 日期：2026-08-03
- 狀態：對話設計已確認，待使用者審閱書面規格
- 範圍：B4／B7／B8、P5／P6／P7 的 phase-only morphology → IDO proxy benchmark
- 不包含：本文件不執行 segmentation、模型訓練或正式 biological validation

## 背景

Exp2 只使用 B4 P6 單一資料組。Repeated CV 顯示 morphology 具有探索性 IDO proxy 訊號，但
Leave-one-condition-out 表現失效，且 Exp2 的 nucleus features 來自 DAPI，不符合未來無染色推論需求。

Exp3 新增九個 `B-ID × Passage` 組合：

| Passage | B4 | B7 | B8 |
|---|---|---|---|
| P5 | 80 PC＋80 IDO | 80 PC＋79 IDO | 80 PC＋80 IDO |
| P6 | 80 PC＋80 IDO | 80 PC＋80 IDO | 80 PC＋80 IDO |
| P7 | 80 PC＋80 IDO | 80 PC＋80 IDO | 80 PC＋80 IDO |

目前最多有 719 組 PC／IDO 配對 FOV。`B4`、`B7`、`B8` 的 donor／lot 關係未知，文件中一律稱為
`B-ID`，不得宣稱跨 donor 或跨製造 lot 泛化。

## 目標

1. 建立只依賴 PC 影像的 whole-cell、nucleus 與 cytoplasm morphology predictors。
2. 以公平且無 leakage 的 grouped nested validation 比較七種 regression model families。
3. 以多種 grouped validations 的平均排名選擇 phase-only 候選模型。
4. 使用 dose-only 與 dose＋morphology models 診斷 IFN/TNF confounding，但不允許其成為部署模型。
5. 將 ΔMorphology 用於描述、穩定性分析與候選 feature 排序，不以九套 signatures 訓練正式模型。
6. 將 Exp3 新特徵與 benchmark 完整隔離於 `immunity/exp3/`，不擴張既有 main/core 分析流程。

## 非目標

- 不將 IDO intensity、IFN/TNF dose、檔名或路徑衍生資訊放入 phase-only predictors。
- 不把 719 張 FOV 宣稱為 719 個獨立 biological samples。
- 不以 9 套 ΔMorphology signatures 訓練 651-feature regression model。
- 不使用 image-level random split 作為主要模型證據。
- 不將 IDO proxy 稱為整體免疫抑制能力；除非未來另有 PBMC／T-cell functional assay 驗證。
- 不修改 `main.py` 的 CLI、既有 cleaned CSV／XLSX 欄位、GUI、一般 morphology 分析輸出或
  `ki67dtc` core feature schema 來承載 Exp3 新特徵。

## 1. Data manifest 與 QC

### 1.1 統一欄位

每一列代表一組可配對 FOV：

```text
b_id
passage
condition_index
condition
ifn_dose
tnf_dose
fov
group_id = b_id × passage
pc_path
ido_path
```

### 1.2 Mandatory condition mapping

正式執行前必須提供明確的 condition mapping，將編號 1–8 對應到八個 IFN/TNF conditions。
程式不得從既有 Exp2 排序猜測 mapping；mapping 缺失時應停止執行並指出缺少欄位。

### 1.3 Pairing 規則

- 容許 `B8-P7` 檔名中的多餘句點，例如 `10X-3.-01` 與 `IDO.-01`。
- `B7-P5` group 4、FOV 10 缺少 IDO，該 PC 影像排除並寫入 `pairing_qc.csv`。
- 所有重複 key、缺少 PC、缺少 IDO、無法解析 condition/FOV 的檔案都必須記錄。
- 不得靜默略過無法配對的影像。

### 1.4 Biological grouping

- Primary grouping metadata 使用 `b_id`、`passage` 與 `group_id`。
- 在 B-ID 身分未確認前，只能報告「跨 B-ID」結果。
- 若日後確認 B-ID 是 donor 或 lot，只更新 metadata 與結果用語，不改寫原始資料。

## 2. Phase-only feature pipeline

### 2.1 Segmentation

```text
PC image
→ PC whole-cell segmentation
→ PC nucleus segmentation
→ cytoplasm mask = whole-cell mask − nucleus mask
```

新資料不得以 DAPI 產生 predictors。舊 B4 P6 的 DAPI 可作為 development-only reference，評估
PC nucleus segmentation 的 Dice／IoU、area agreement、Feret agreement 與 matched-cell coverage。

### 2.2 Primary feature set

每顆細胞計算：

- 16 個 whole-cell basic morphology features
- 16 個 nucleus basic morphology features
- 1 個 nucleus/cytoplasm ratio

每張 FOV 對各 feature 取 median，得到 33 個 Primary predictors。IDO channel 只用來建立
background-corrected image-level `IDO_score` target。

### 2.3 Secondary feature sets

| Feature set | Predictors | 定位 |
|---|---:|---|
| Basic median | 33 | Primary benchmark |
| Basic median＋IQR | 66 | Population heterogeneity ablation |
| Paper-style median | 最多 93 | Zernike／paper feature ablation |

Round 2 ablation 只使用 Round 1 的冠軍與第二名，結果標記為 exploratory，不改寫 Round 1 冠軍。
若要讓 secondary feature set 成為正式候選，必須在新的獨立資料中預先指定並驗證。

### 2.4 Exp3 feature isolation boundary

Exp3 使用獨立 package 與 entry point：

```text
immunity/exp3/
├─ manifest.py
├─ phase_features.py
├─ feature_sets.py
├─ benchmark.py
├─ reporting.py
└─ run_benchmark.py
```

執行入口限定為 Exp3 專用命令，例如：

```powershell
python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml
```

邊界規則：

- Exp3 可將現有 segmentation function 當成 library dependency 使用，但不得呼叫或擴張 `main.py` 的 CLI／export
  workflow。
- Exp3 直接讀取 raw PC／IDO paths 或自己的 intermediate cache；不得要求 main cleaned CSV 增加欄位。
- 新增的 paper-style、Zernike、IQR 或 ΔMorphology features 只能寫入 `immunity/outputs/exp3/`。
- Primary run 預設只計算 33 個 basic medians；66／93-feature sets 必須由 Exp3 config 明確啟用，採 lazy
  computation，不因 import 或一般分析自動產生。
- 不將 Exp3 feature names 加入全域 `main.py`、GUI、XLSX 或 legacy report 的 predictor lists。
- 若需要共用純計算邏輯，先以不改變既有 public behavior 的 adapter 包裝；main/core regression tests 必須證明
  原輸出 schema 與數值不變。

## 3. Model registry

### 3.1 Phase-only candidate models

| Model | Feature set | Preprocessing | 調參重點 |
|---|---|---|---|
| Paper 3-feature Linear Regression | Whole-cell perimeter、nucleus/cytoplasm ratio、whole-cell maximum Feret | Median imputer、scaler | 無 |
| Ridge | 33 medians | Median imputer、scaler | alpha |
| ElasticNet | 33 medians | Median imputer、scaler | alpha、l1 ratio |
| RBF-SVR | 33 medians | Median imputer、scaler | C、gamma、epsilon |
| Random Forest | 33 medians | Median imputer | depth、leaf size、max features |
| Extra Trees | 33 medians | Median imputer | depth、leaf size、max features |
| HistGradientBoosting | 33 medians | Median imputer | learning rate、leaf nodes、L2 |

### 3.2 Diagnostic models

以下模型使用相同 outer folds，但不參與 phase-only 冠軍排名：

- `DummyRegressor(strategy="median")`
- Dose-only Ridge：IFN dose、TNF dose、IFN×TNF
- Dose＋Morphology Ridge：dose features＋33 morphology medians

### 3.3 Target scaling

所有非 Dummy models 都在 training fold 內標準化 `IDO_score`，預測後 inverse transform，再以原始 IDO
單位計算 MAE／RMSE。Target transformer 必須位於 estimator 內，不能在切分資料前 fit。

### 3.4 Hyperparameter spaces

```text
Ridge alpha:
0.0001、0.001、0.01、0.1、1、10、100、1000、10000

ElasticNet alpha:
0.0001、0.001、0.01、0.1、1、10
l1_ratio:
0.1、0.5、0.9、1.0

RBF-SVR C:
0.1、1、10、100
gamma:
scale、0.01、0.1、1
epsilon（標準化 target 單位）:
0.05、0.1、0.2
```

Tree ensembles 至少使用 400 trees，搜尋 `max_depth`、`min_samples_leaf` 與 `max_features`。
HistGradientBoosting 搜尋：

```text
learning_rate: 0.03、0.1
max_leaf_nodes: 7、15、31
min_samples_leaf: 10、20、40
l2_regularization: 0、1、10
```

每個模型、每個 outer fold 最多評估 24 組 hyperparameters。小型空間使用完整 grid；大型空間使用
固定 seed 的 deterministic parameter sampling。Inner scoring 統一為 negative MAE。

## 4. Grouped nested validation

### 4.1 Outer validations

| Validation | Folds | 問題 |
|---|---:|---|
| Leave-one-B-out | 3 | 能否泛化到未看過的 B-ID？ |
| Leave-one-passage-out | 3 | 能否泛化到未看過的 passage？ |
| Leave-one-B×Passage-out | 9 | 能否泛化到未看過的實驗組合？ |
| Leave-one-condition-out | 8 | 能否泛化到未看過的刺激條件？ |

所有模型共用完全相同的 outer split manifest。任何 outer test row 都不得出現在 training、imputation、
scaling、target scaling、hyperparameter tuning 或 supervised feature selection。

### 4.2 Inner validation

- Inner folds 以 `group_id = B-ID × Passage` 分組。
- Outer training data 中 group 少於兩組時，該 validation 應失敗，不得改用 image-level random split。
- Hyperparameter selection 只看 inner-fold MAE。
- ElasticNet sparsity是 model-embedded selection；不得先用完整資料做 univariate screening。

### 4.3 Metrics

每個 outer fold 計算：

- MAE：Primary metric
- RMSE
- R²
- Spearman correlation
- Prediction variance 與 residual summaries

Constant target 或 constant prediction 時，無法定義的 R²／Spearman 記為 `NaN`；MAE 與 RMSE 仍保留。

## 5. Ranking 與合格門檻

### 5.1 Average-rank rule

每種 validation 先依 median fold MAE 對七個 phase-only models 排名。四種 validation 各占 25%，
Overall rank 為四個 ranks 的平均。Fold 數較多的 validation 不因此取得更高權重。

### 5.2 Eligibility gate

冠軍必須符合：

1. 至少 3／4 種 validation 的 MAE 優於 Dummy。
2. 至少 2／4 種 validation 的 median fold R² 大於 0。
3. 所有 outer folds 成功完成。
4. Phase-only predictors 未含 IDO、dose 或 test-derived 資訊。
5. Prediction 未退化為近乎常數的 training mean；若某 validation 的 OOF prediction standard
   deviation 小於 observed target standard deviation 的 5%，即判定該 validation 退化。

若沒有模型通過，正式結論為「目前沒有可泛化的 phase-only model」，不得從不合格模型中強迫選冠軍。

### 5.3 Tie-breaker

Average rank 差距小於 0.25 時，依序比較：

1. 最差 validation rank
2. Leave-one-B-out MAE
3. Spearman correlation
4. 模型複雜度；表現接近時選較簡單模型

## 6. Condition-adjusted sensitivity analysis

此分析回答 morphology 是否含有 condition 之外的 IDO 訊號，不參與冠軍排名。

對 Leave-one-B-out、Leave-one-passage-out 與 Leave-one-B×Passage-out：

1. 只以 outer training fold 計算各 condition 的平均 IDO。
2. Training 與 test target 分別減去 training-derived condition mean。
3. 使用相同 phase-only model pipeline 預測 residual IDO。

Leave-one-condition-out 因 test condition 在 training 中不存在，不執行 condition-mean residualization，僅保留 raw IDO
結果。如果模型只在 raw IDO 表現良好，報告必須標記「模型可能透過 morphology 間接辨認刺激條件」。

## 7. ΔMorphology

每個 `B-ID × Passage` 計算七組預先指定 contrasts，總計九套 group-level ΔMorphology signatures。

用途限定為：

- heatmap 與 PCA
- B-ID／passage trend
- 各 feature 的方向一致性
- paper features replication check
- 排出最多 3–5 個候選 ΔFeatures，供未來資料預先驗證

九套 signatures 不用於 651-feature regression。正式 batch-level ΔMorphology model 的最低目標為 30 個真正
獨立 biological signatures，理想為 50 個，且 Primary model 先限制為 1–3 個預先指定 features。

## 8. 輸出

```text
immunity/outputs/exp3/
├─ data_manifest.csv
├─ pairing_qc.csv
├─ segmentation_qc.csv
├─ feature_cache/
├─ feature_sets.json
├─ outer_splits.csv
├─ oof_predictions.csv
├─ fold_metrics.csv
├─ hyperparameters.csv
├─ model_ranking.csv
├─ feature_importance.csv
├─ condition_adjusted_metrics.csv
├─ EXPERIMENT_RECORD.md
└─ figures/
```

必要圖表：

- Model × validation MAE rank heatmap
- Fold-level MAE distributions
- 冠軍模型 observed-vs-predicted
- Phase-only、Dummy、dose-only 與 dose＋morphology 比較
- Coefficient／permutation importance stability
- B-ID、passage、condition residual plots

Linear models 輸出 standardized coefficients。Nonlinear models 的 feature importance 必須由 outer test fold 的
permutation importance 計算，並以 fold-level distribution 呈現；不得使用 training-only importance 冒充泛化解釋，
也不得根據 outer-test permutation importance 回頭改模型、feature set 或排名。

## 9. 錯誤處理

- Pairing、condition mapping 或必要 metadata 缺失時，停止下游分析並輸出 QC 原因。
- Exp3 output path 指向 main/core result directory 時直接阻擋，避免覆寫 legacy analysis artifacts。
- 任一 model/fold 失敗時記錄 model、validation、fold、exception 與 hyperparameters。
- 模型 fold 失敗後不得以不同 split 重試並取較佳結果。
- Outer split manifest 必須可重現並保存，所有模型強制共用。
- IDO、dose 或路徑衍生欄位進入 phase-only feature set 時直接阻擋執行。
- Segmentation 通過率與每張 FOV 細胞數低於設定門檻時，影像排除並保留原因。

## 10. 測試與驗證

### Unit tests

1. Parser 支援 P5／P6／P7 命名與 `B8-P7` 多餘句點。
2. 缺少 IDO pair 會出現在 QC，且不會進入 dataset。
3. Condition mapping 缺失時明確失敗。
4. Feature whitelist 阻擋 IDO／dose leakage。
5. Metrics 正確處理 constant vectors 與 `NaN`。
6. Exp3 secondary features 不會出現在 main cleaned CSV、XLSX、GUI 或 legacy predictor schema。

### Split／leakage tests

- 每個 outer fold 的 train/test keys 完全不重疊。
- 對應 validation 的 held-out B-ID、passage、group 或 condition 不出現在 training。
- Imputer、scaler、target scaler 與 tuning 只 fit outer training fold。
- 所有 models 使用相同 outer split IDs。

### Integration tests

- 以小型 synthetic grouped dataset 執行七個 candidates 與三個 diagnostics。
- 驗證固定 seeds 可重現 splits、hyperparameters 與 predictions。
- 先執行 5–15 分鐘 smoke benchmark，再執行完整 benchmark。

## 11. 執行與監控

預估執行時間：

```text
Smoke test：5–15 分鐘
完整 phase segmentation：依 GPU 約 1–4 小時
完整 nested benchmark：依 CPU 約 4–12 小時
```

執行紀錄必須保存：

- 完整 config
- Git commit／dirty-worktree metadata
- Python 與套件版本
- Random seeds
- Input manifest 與檔案數
- 每個 phase 的開始、完成、錯誤與耗時

## 12. 結果解讀限制

- B-ID 生物學關係未知，因此結果只稱為跨 B-ID exploratory validation。
- P5／P6／P7 可能來自相同 B-ID 的連續 passages，不視為完全獨立 donor samples。
- IDO fluorescence 是 immunosuppressive proxy，不等於 PBMC／T-cell functional suppression。
- Round 2 feature ablation 屬探索性，不得當成獨立確認性證據。
- 通過 eligibility gate 代表目前資料中的候選模型，不代表已完成臨床或製造流程 validation。

## 13. 驗收條件

1. 719 組預期 PC／IDO pairs 經 QC 後可追溯至原始檔案。
2. Primary predictors 完全由 PC morphology 產生。
3. 七個 candidate models 與三個 diagnostic models 使用相同 outer folds。
4. 所有 preprocessing、tuning 與 selection 位於 outer training fold 內。
5. 四種 grouped validations、完整 OOF predictions 與排名輸出成功。
6. Ranking、eligibility gate 與 no-winner 結果可由輸出檔案重算。
7. ΔMorphology 僅作描述與候選排序，不進入本次正式 regression。
8. `EXPERIMENT_RECORD.md` 清楚區分證據、推論、限制與下一批資料需求。
9. 一般 `main.py` 分析在實作前後維持相同 CLI 與輸出欄位；Exp3 secondary features 只有明確執行
   `immunity.exp3.run_benchmark` 時才會計算。
