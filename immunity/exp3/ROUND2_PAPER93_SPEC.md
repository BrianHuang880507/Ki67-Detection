# Exp3 Round 2：Phase-only Paper-style 93 特徵實驗規格

- 狀態：四節口頭設計已確認，等待書面規格複核
- 日期：2026-08-11
- 實驗角色：Exp3 的隔離式探索性 Round 2
- 參考研究：[Morphological features of IFN-γ–stimulated mesenchymal stromal cells predict overall immunosuppressive capacity](https://www.pnas.org/doi/10.1073/pnas.1617933114)

## 1. 實驗目的

本次實驗只回答一個問題：把現有 33 個 phase morphology predictors 擴充為 93 個
paper-style predictors 後，是否能更準確地預測每張 FOV 的 IDO 螢光亮度 proxy。

本次模型預測的是 `IDO_score`，不是「免疫力」或功能性免疫抑制能力。IDO 亮度與免疫力之間的
生醫關係，留給生醫同仁另外制定，不納入本程式的 label、模型或結論。

下列項目不在本次範圍內：

- 建立資料庫。
- 功能性免疫抑制 assay 或免疫力 label。
- donor label、donor 推論或 donor-specific model。
- 使用 IFN／TNF dose、condition、影像路徑或批次代碼作 predictor。
- 使用 IDO channel、DAPI 或其他染色 channel 作 predictor。
- 重新執行 33-feature × 7-model 的 Round 1。
- 修改 `main.py`、GUI、既有 CSV／XLSX schema 或既有主流程。

## 2. 輸入、目標與分析單位

### 2.1 Predictor 輸入

模型推論時只需要 phase-contrast 影像。Whole-cell、nucleus 與 cytoplasm 都由現有
phase segmentation 產生；這裡的 nucleus 不是依賴 DAPI 建立。

IDO 影像只在訓練／驗證資料中用來建立 target，絕不進入 predictor matrix。

### 2.2 Target 定義

沿用 Round 1 的 target，不重新定義：

1. 對每顆通過配對與 segmentation QC 的細胞，計算 cytoplasm 內 IDO 平均亮度。
2. 扣除 whole-cell 以外背景區域的 IDO 中位數，得到 cell-level `IDO_score`。
3. 同一張 FOV 內所有有效細胞的 `IDO_score` 取中位數，得到 FOV-level target。

因此 MAE／RMSE 的單位是 background-corrected IDO intensity proxy，而不是百分比或免疫力單位。

### 2.3 分析單位

- 一列模型資料代表一張 FOV。
- 每張 FOV 至少需要 3 顆有效且 nucleus–cell 已配對的細胞。
- Cell-level morphology 以中位數彙整成 FOV-level predictors。
- 同一批細胞在不同 condition 的 FOV 仍不是額外的 donor；本實驗不建立 donor 欄位。

## 3. 固定資料快照

Round 2 必須使用與已完成 Round 1 相同的資料母體：

- Raw PC images：720。
- Raw IDO images：719。
- 完整 PC–IDO pairs：719。
- 已確認的 segmentation／feature QC exclusions：26 張。
- 正式模型資料：693 張 FOV。

Round 1 的 `data_manifest.csv`、`outer_splits.csv`、設定、seed 與 exclusion list 是唯讀基準。
FOV-level predictor 與 target 的唯一權威來源是
`immunity/outputs/exp3/feature_cache/image_level_basic.csv`；目前該檔有 693 個唯一 image keys，
SHA-256 為 `DAD479258D08846D05A43F551B0001A52D96FA5E2E8EA6A98645723931AA0D14`。
Round 2 必須先驗證此檔 hash，再以 `image_key` 一對一對齊；重新擷取出的 33 個 basic values 與
`IDO_score` 使用 `rtol=0`、`atol=1e-12`、`equal_nan=False` 比對。Image keys、數值、split
membership 或 provenance 任一不一致就停止。

目前 `B4`、`B7`、`B8` 只視為既有 batch identifiers。它們可作 grouped validation 的
split metadata，但不可改名或解讀成 donor，也不可進入 predictors。

Condition 與 IFN／TNF dose 只可用於重建既有 split 或報表分層，不可進入 predictors，未來推論
也不要求提供 dose。

## 4. Paper-style 93 predictors

`paper_style_median` 固定為 93 個 phase-derived FOV predictors。所有 cell-level 數值都在每張
FOV 內取中位數，FOV 欄名使用 `__median` 後綴。

### 4.1 現有 33 個基本特徵

Whole-cell 與 nucleus 各計算下列 16 個 morphology features，共 32 個：

`area`、`compactness`、`eccentricity`、`extent`、`sphericity`、
`major_axis_length`、`feret_length`、`minor_axis_length`、`feret_width`、
`maximum_radius`、`mean_radius`、`median_radius`、`aspect_ratio`、
`perimeter_area_ratio`、`perimeter`、`solidity`。

再加入 1 個 `nucleus_cytoplasm_area_ratio`，合計 33 個。

### 4.2 新增 60 個 paper-style 近似特徵

- 50 個 Zernike descriptors：whole-cell 25 個與 nucleus 25 個，index 固定為 `00`–`24`。
- 6 個區域比例／差值：
  - `nucleus_cytoplasm_mean_ratio`
  - `nucleus_cytoplasm_intden_ratio`
  - `nucleus_cytoplasm_raw_intden_ratio`
  - `nucleus_cell_intden_ratio`
  - `nucleus_cytoplasm_entropy_difference`
  - `nucleus_cytoplasm_cv_difference`
- 1 個位置特徵：`nucleus_centroid_offset`，定義為 nucleus 與 whole-cell centroid 距離，除以
  whole-cell equivalent radius。
- 3 個 phase intensity dispersion features：whole-cell、nucleus、cytoplasm 各自的 intensity
  coefficient of variation（CV）。

### 4.3 與 paper 的關係

「93」模仿 paper 的 feature-count 與以形態描述細胞狀態的思路，但不是逐欄位完全複製。
本實驗使用 phase-derived segmentation 與 phase intensity descriptors，並排除需要染色 channel
或目前資料無法可靠重建的 paper features。因此正式名稱必須保留
`phase-only paper-style approximation`，不可寫成 exact paper replication。

`ΔMorphology` 不進入本次模型。既有 9 組 Batch × Passage signatures 仍只用於描述性圖表，
不會擴張成 651 predictors，也不會當成 693 個獨立樣本。

## 5. 實驗架構與資料流

Round 2 使用獨立設定與執行入口，不能以啟用 secondary feature set 的方式讓既有 Exp3 runner
重跑完整 Round 1。預定入口與設定為：

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_round2_paper93 --config immunity/configs/exp3_round2_paper93.yaml
```

新 runner 可以重用 `immunity.exp3` 內已驗證的純函式與 model registry，但不可呼叫會執行完整
七模型 Round 1 的 orchestration。資料流固定如下：

1. 驗證 Round 1 基準 artifacts、693 個 image keys、26 個 exclusions、target 與 outer splits。
2. 只重用 provenance 完全相符的既有 mask cache；cache 缺少或 provenance 不符時停止，不在
   Round 2 內自動重新 segmentation。若未來另行重切，必須建立新的資料快照與實驗版本。
3. 對相同 693 張 FOV 擷取並彙整 93 predictors。
4. 執行 feature/leakage/data-lock preflight。
5. 只訓練 `extra_trees` 與 `random_forest` 的 93-feature 版本。
6. 從 Round 1 唯讀載入相同兩個模型的 33-feature metrics／OOF predictions，以及
   `dummy_median` 的 eligibility evidence。
7. 產生 33 vs 93 比較、eligibility、解釋性結果與繁體中文實驗記錄。

若 Round 1 baseline 的 provenance 不符，本流程必須停止並列出差異；不得自動重跑七個模型，
也不得自行決定新的 baseline。

## 6. 模型與驗證設計

### 6.1 模型

新訓練僅包含：

- `extra_trees`
- `random_forest`

兩者沿用 Round 1 相同的 hyperparameter search space、候選上限、seed 與 inner-CV 規則，避免把
搜尋資源差異誤認成 feature set 效果。所有 preprocessing、median imputation、target scaling
與 hyperparameter tuning 都只能在各 outer training fold 內 fit。因正式 preflight 要求所有
predictors 為 finite，median imputer 是為了維持 Round 1 pipeline parity 的防禦性步驟；正式資料
預期不會實際發生插補。

### 6.2 Outer validation families

沿用 Round 1 已凍結的四組 split membership：

- `leave_one_b_out`
- `leave_one_passage_out`
- `leave_one_group_out`（Batch × Passage）
- `leave_one_condition_out`

每張 FOV 在每個 validation family 必須恰好出現在一次 test fold，train/test 不可重疊。

### 6.3 評估指標

- Primary：MAE，越低越好。
- Secondary：RMSE 越低越好；R² 與 Spearman 越高越好。
- 報表同時呈現每個 outer fold、每個 validation family 的中位數與完整 OOF 結果。

比較單位是四個 model × feature-set configurations：

- `extra_trees + 33`
- `extra_trees + 93`
- `random_forest + 33`
- `random_forest + 93`

33-feature configurations 來自已驗證 Round 1 artifacts；93-feature configurations 才是本次新
訓練結果。`dummy_median` 的既有 fold metrics 與 OOF predictions 也從 Round 1 唯讀載入，只作
eligibility reference，不算新的候選模型或本次 model fit。

### 6.4 Eligibility 與選擇規則

沿用 Round 1 的 fail-closed gates：

- 至少 3 個 validation families 的 MAE 優於 `dummy_median`。
- 至少 2 個 validation families 的 median R² 大於 0。
- 所有預期 outer folds 完整成功。
- Predictor identity 通過 phase-only whitelist。
- OOF prediction standard deviation 通過既有 non-degeneration gate。

唯一決策演算法如下：

1. 先對四個 configurations 分別套用 eligibility gates，不合格者不可成為建議版本。
2. 在每個 validation family 依 OOF MAE 對四個 configurations 排名；完全相同值使用最小名次，
   再取四個 family ranks 的平均。
3. 令最佳合格 configuration 的 average rank 為 `R*`；只有
   `average_rank - R* < 0.25` 才屬同一 tie band，等於 `0.25` 不算。
4. 若同一 tie band 同時包含同一 algorithm 的 33-feature 與 93-feature 版本，先淘汰該
   algorithm 的 93-feature 版本。
5. 剩餘者依序比較 worst validation rank、leave-one-B-out MAE、整體 OOF Spearman、
   feature count、既有 model simplicity rank 與 model name，得到唯一建議版本。

「93-feature 有明確優勢」專指它通過 gates、未被第 4 步的簡潔性規則淘汰，並由第 5 步選為
唯一建議版本；不另作主觀判斷。

這個選擇只形成 Round 2 建議，不改寫 Round 1 的既有 winner 或正式結果。

## 7. 隔離與輸出

正式輸出根目錄固定為：

`immunity/outputs/exp3/round2_paper93/`

Smoke artifacts 必須放在其下的 `smoke/`，不得混入正式比較。Round 2 不得覆寫、搬動或刪除
`immunity/outputs/exp3/` 的 Round 1 artifacts。

至少需要下列可稽核輸出：

- 93 predictor registry 與 feature-count／leakage QC。
- 本次使用的 693-row data snapshot 與 frozen split identity。
- 33 vs 93 comparison table。
- 兩個模型的 fold metrics 與完整 OOF predictions。
- Hyperparameters 與 feature importance。
- Eligibility gate 結果與選擇理由。
- Config、seed、Git 狀態、runtime、套件版本與 artifact hashes。
- 繁體中文 `EXPERIMENT_RECORD.md`。
- `run.log` 與失敗時的具名影像路徑／image key。

讀入的 Round 1 rows 與本次新產生的 Round 2 rows 必須有清楚的 `source_round` 與 `feature_set`
身分，避免把既有結果誤認成新訓練結果。

## 8. Preflight 與錯誤處理

完整 Round 2 開始模型 fitting 前，必須全部通過：

1. `paper_style_median` 恰好有 93 個、名稱唯一且順序固定的 predictors。
2. 其中恰好包含 canonical 33-feature baseline，再加上恰好 60 個 extras。
3. Predictor 名稱與 matrix 不含 IDO、IFN、TNF、dose、condition、path、filename、donor、
   batch identifier 或 delta feature。
4. 33 個 basic predictors 與 `IDO_score` 必須保持 Round 1 的值；60 個 extras 必須從相同的
   Round 1 valid-cell roster 計算。每個 extra 在每張 FOV 至少需要 3 個 finite cell-level
   observations，並輸出 per-FOV／per-feature valid count；不得讓不同規則靜默改變 cell roster。
5. 93 個 FOV-level predictor values 全部為有限值。只有一個唯一有限值的 exact constant
   feature 要列入 redundancy QC，但本身不 hard-fail；以 placeholder 常數取代未成功計算的
   feature 則必須 hard-fail。
6. 正式資料恰好是相同 693 個 image keys，target 與 Round 1 一致。
7. 四個 outer split families 的 fold IDs 與 row memberships 都與 Round 1 一致。
8. Median imputer 與任何 preprocessing 只在 outer training fold 內 fit。
9. Round 1 baseline artifacts 的 config、manifest、split、dummy evidence 與 model identity 可驗證。

任何新 segmentation／extraction failure、少於 3 顆有效細胞或 provenance mismatch 都必須：

- 以非零 exit code 停止。
- 保留 QC 與 `run.log`。
- 列出 image key 與可用的完整影像路徑。
- 不自動新增 exclusion，不降低 `min_cells_per_image`，也不強制 nucleus–cell 配對。

## 9. 驗證順序

實作後固定依序執行：

1. Feature definition、count、aggregation 與 leakage unit tests。
2. Baseline provenance、data-lock、split-lock 與 fold-local preprocessing tests。
3. 小型 synthetic／fixture integration tests。
4. 隔離的 smoke run；只驗證 schema、93-feature count、leakage、mask provenance、valid-count
   規則與輸出完整性，不要求 smoke subset 滿足 693-row 或 frozen split-membership gates，也不
   產生 33 vs 93 科學結論。
5. 完整 693-FOV Round 2。
6. Artifact completeness、row count、OOF coverage、hash 與 eligibility validation。
7. 產生並人工複核繁體中文 `EXPERIMENT_RECORD.md`。

## 10. 完成標準

只有同時符合下列條件，才能宣告 Round 2 完成：

- 沒有修改或覆寫 Round 1 結果與主流程。
- 93-feature matrix 通過 feature-count、finite-value 與 leakage gates。
- 正式分析維持同一組 693 FOV、target、seed 與 outer splits。
- 只有 `extra_trees` 與 `random_forest` 產生新的 93-feature model fits。
- 四組 validation family 都有完整、可追溯的 fold metrics 與 OOF predictions。
- 33 vs 93 結論依預先寫定的 gates 與 tie rule 產生。
- `EXPERIMENT_RECORD.md` 清楚區分 IDO brightness prediction、paper-style approximation 與
  不在範圍內的免疫力解讀。
