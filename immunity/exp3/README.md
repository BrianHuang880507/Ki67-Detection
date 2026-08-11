# Exp3 Phase-only Morphology Benchmark

> 目前下一步：先向實驗人員取得 condition index 1–8 對應的 IFN/TNF 劑量，填入
> `immunity/configs/exp3.yaml`。在 mapping 完整前，程式只產生 pairing QC，不會訓練模型。

此流程使用 PC 影像的 whole-cell、nucleus 與 cytoplasm morphology，預測 image-level
background-corrected `IDO proxy`。B4、B7、B8 的生物學身分尚未確認，因此結果只代表跨
B-ID 的 exploratory validation，不可解讀為跨 donor 或跨製造 lot 泛化。

## 正式執行

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml
```

正式結果寫入 `immunity/outputs/exp3/`。執行成功的必要條件是
`EXPERIMENT_RECORD.md` 已完成發布；失敗時 CLI 會回傳非零 exit code，並保留已產生的 QC
與 `run.log`。

## 72-image smoke run

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml --smoke-fovs-per-condition 1
```

Smoke run 會在每個 `group_id × condition_index` 依 FOV 排序取第一張影像，共 9 groups ×
8 conditions = 72 張。所有 smoke artifacts 都隔離在
`immunity/outputs/exp3/smoke/`，不得作為 full-run ranking 或 winner。

## Round 2：Paper-style 93 特徵

Round 2 只從 phase 影像建立 93 個 paper-style approximation predictors，並只新訓練
`extra_trees` 與 `random_forest`。Round 1 的 33-feature 結果與 Dummy evidence 僅供唯讀引用，
不會重跑七個模型。

### 正式執行

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_round2_paper93 --config immunity/configs/exp3_round2_paper93.yaml
```

### 流程 smoke

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_round2_paper93 --config immunity/configs/exp3_round2_paper93.yaml --smoke-fovs-per-condition 1
```

正式結果只會寫入 `immunity/outputs/exp3/round2_paper93/`；smoke 只會寫入其
`smoke/` 子目錄。Smoke 僅驗證流程，不可用於 33 vs 93 的科學比較或 recommendation。

每個成功發布的 Round 2 bundle 固定包含 15 個 CSV、三個 caller JSON
（`feature_sets.json`、`baseline_provenance.json`、`run_metadata.json`）、
`artifact_hashes.json`、`EXPERIMENT_RECORD.md` 與 `run.log`；不產生 PNG。

IDO 只作亮度 target；結果不是免疫力 label，也不建立 donor label。Mask provenance 不一致時，
流程會停止，並在 failed generation 保留 `mask_provenance_qc.csv`。Frozen Round 1 evidence 載入
不一致時，流程同樣停止，原因寫入 quarantined `run.log`，但此階段不保證存在 QC CSV。兩種情況
都不會自動重新 segmentation、重寫 cache 或增加 exclusion。

## 特徵與模型範圍

- Primary `basic_median` 固定為 33 個 PC-only morphology medians。
- `basic_median_iqr` 的 66 predictors 與 `paper_style_median` 的 93 predictors 必須在
  config 明確 opt-in；兩者只用於 exploratory Round 2。
- `paper_linear_3f` 永遠只使用三個 paper predictors，不會被擴張成 66／93-feature
  linear model。
- IDO channel 只建立 target，不會進入 phase-only predictors。
- IFN／TNF dose models 只用於 confounding diagnostics，不參與 Primary ranking 或 winner。

## Validation 與結果判讀

Round 1 的七個 phase-only candidates 與三個 diagnostics 共用相同 outer splits：
Leave-one-B-out、Leave-one-passage-out、Leave-one-B×Passage-out 與
Leave-one-condition-out。Eligibility evidence 不完整時流程會 fail closed；沒有模型通過全部
gate 時不會強迫發布 winner。

Round 2 先取距最佳 average rank 小於固定 0.25 的 tie band，再依 worst validation rank、
LOBO MAE、Spearman、模型簡潔度與名稱決定前兩個 phase candidates；不足兩個才從 band 外
依相同完整證據補足。此門檻不可由 benchmark config 覆寫。Round 2 只使用明確啟用的
secondary sets，結果標記為
`exploratory_round_2`，不會回流 Primary ranking、eligibility 或 winner。

## ΔMorphology

ΔMorphology 只描述 9 個 `B-ID × passage` biological groups 的 group-level patterns，不進入
本次 regression，也不能把最多 719 張 FOV 當成 719 個獨立 biological samples。

## Round 1 主要輸出 contract

Round 1 每次成功執行會產生固定 13 個結果 CSV、`feature_sets.json`、`run_metadata.json`、
8 張 PNG 與 `EXPERIMENT_RECORD.md`。若 eligibility gate 選出 winner，另在 `models/` 保存 model
bundle 及相符的 `final_model.json`。Metadata 包含 UTC timestamps、runtime、Git 狀態、config／
manifest SHA-256、Python／套件版本、seeds、feature sets、input counts、cache hits 與 failure count。

## Round 1 執行世代與 cache

新 run 會先把固定名稱的上一世代 tables、JSON、record、figures 與 models 搬到同一 run root
下的 `_generations/previous-*`，不會刪除 `feature_cache/` 或移動正在寫入的 `run.log`。若本世代
中途失敗，已產生的成功 artifacts 會進入 `_generations/failed-*`，固定 root 只保留本次可用的
pairing、manifest、segmentation 或 model failure evidence。

`feature_cache/masks/` 的 NPZ 只有在 PC bytes SHA-256、segmentation config 與
segmenter/model signature 全部吻合時才會重用。Injected Segmenter 若未提供明確且非空的
`cache_signature`，即使 class 相同也一律重新 segmentation。缺少 provenance 的舊 cache 或
任一證據不符時，流程會重新 segmentation，並在 `segmentation_qc.csv` 記錄原因與
provenance hash。
