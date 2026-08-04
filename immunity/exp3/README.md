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

Round 2 只比較 Round 1 overall rank 可用的前兩個 phase candidates 與明確啟用的 secondary
sets。Round 2 結果標記為 `exploratory_round_2`，不會回流 Primary ranking、eligibility 或
winner。

## ΔMorphology

ΔMorphology 只描述 9 個 `B-ID × passage` biological groups 的 group-level patterns，不進入
本次 regression，也不能把最多 719 張 FOV 當成 719 個獨立 biological samples。

## 主要輸出

每次成功執行會產生固定 13 個結果 CSV、`feature_sets.json`、`run_metadata.json`、8 張 PNG
與 `EXPERIMENT_RECORD.md`。若 eligibility gate 選出 winner，另在 `models/` 保存 model bundle
及相符的 `final_model.json`。Metadata 包含 UTC timestamps、runtime、Git 狀態、config／
manifest SHA-256、Python／套件版本、seeds、feature sets、input counts、cache hits 與 failure
count。
