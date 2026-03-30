# Ki67 單顆 Cell Ordinal 三分類 Benchmark

本目錄提供一套**不修改主流程**（`main.py`、`ki67dtc/` 不動）即可獨立執行的實驗框架，符合以下原則：

- 任務定義：單顆 cell `0/1/2` ordinal classification
- split 規則：同一張 `Image` 的 cells 不可跨 split
- split 程序：每個 dataset 先做 image-grouped train/val/test，再合併全域 split
- 比較模型：LogisticRegression / RandomForest / ExtraTrees / HistGradientBoosting
- 可選模型：XGBoost / CatBoost（環境可用時自動納入）
- 雙軌比較：`without passage feature` vs `with passage feature`
- 評估輸出：accuracy / macro_f1 / balanced_accuracy / quadratic_weighted_kappa / confusion matrix / per-passage

---

## 1) 執行方式

### A. 正式三分類（建議）
有外部標註檔（欄位：`dataset,Image,Cell_ID,label`）時：

```bash
python analysis/ki67_cell_ordinal_benchmark/run_benchmark.py --labels-csv <PATH_TO_LABELS_CSV> --random-state 42
```

### A-2. 正式三分類調參版（Baseline + Tuned + Ensemble）

```bash
python analysis/ki67_cell_ordinal_benchmark/optimize_benchmark.py --labels-csv <PATH_TO_LABELS_CSV> --random-state 42
```

輸出會在 `analysis/ki67_cell_ordinal_benchmark/run_tuned_<timestamp>/`：
- `tuning_candidates_val.csv`：每個候選參數在 val 的分數
- `best_configs_by_family.csv`：每個模型家族選中的最佳參數
- `tuned_test_metrics.csv`：refit 後 test 結果（含 baseline_refit / tuned_refit / soft_voting_refit）
- `tuned_model_ranking_test.csv`：test 排名

### B. 先匯出待標註 keys 模板

```bash
python analysis/ki67_cell_ordinal_benchmark/run_benchmark.py --label-column ki67_positive --allow-missing-classes --export-cell-template analysis/ki67_cell_ordinal_benchmark/cell_keys_template.csv
```

說明：
- `--allow-missing-classes` 只建議用於 smoke test。正式三分類應該拿掉此參數，並提供 `label` 0/1/2。

---

## 2) 可重現 split 策略

實作位置：`run_benchmark.py` 內 `build_grouped_split()`。

- 以 `(dataset, Image)` 為不可分割 group。
- 每個 dataset 內先抽出 unique `Image`，使用 `random_state + dataset hash` 做可重現打亂。
- 依 `train/val/test` 比例分配 image 數量（預設 0.8/0.1/0.1）。
- 小資料集保護：
  - 1 張 image：全放 train
  - 2 張 image：train/test 各 1（val 0）
  - >=3 張 image：盡量確保 train/val/test 都有 image
- merge 回 cell-level 後，會檢查每個 `(dataset, Image)` 只屬於單一 split（防 leakage）。

---

## 3) 主要輸出檔案

每次執行會在 `analysis/ki67_cell_ordinal_benchmark/run_<timestamp>/` 產生：

- `run_config.json`
- `source_manifest.csv`
- `image_split_map.csv`
- `cell_split_detail.csv`
- `split_summary_by_dataset.csv`
- `split_summary_by_label.csv`
- `split_summary_by_passage.csv`
- `without_passage_feature_columns.txt`
- `with_passage_feature_columns.txt`
- `metrics_summary.csv`
- `model_ranking_test.csv`（若 test 有成功結果）
- `confusion_matrices/*.csv`
- `per_passage/*_per_passage_metrics.csv`
- `predictions/*_predictions.csv`
- `feature_importance/*.csv`（模型支援時）

---

## 4) 結果解讀框架（建議）

先看 `metrics_summary.csv` 的 test split：

1. `quadratic_weighted_kappa`：優先指標（有序分類一致性）
2. `macro_f1`：三類均衡表現
3. `balanced_accuracy`：對不平衡資料更穩定
4. `accuracy`：整體正確率輔助參考

再看 `per_passage`：

- 比較同模型在不同 `P` 的指標波動，確認是否只在少數 passage 高分。
- 比較 `with_passage` vs `without_passage`：
  - 若 `with_passage` 在大多 passage 都更穩，表示 passage 確實提供泛化訊息。
  - 若只有少數 passage 改善，需警惕模型過度依賴 passage。

最後看 `confusion_matrix`：

- 重點檢查是否大量 `low<->medium`、`medium<->high` 相鄰誤分（可接受度較高）
- 若常出現 `low<->high` 直接跨兩級誤分，代表 ordinal 結構學得不足。

---

## 5) 風險提醒

- 若標註覆蓋率不足（labels merge 後丟失太多 rows），結果可信度會下降。
- 某些 dataset image 太少時，val/test 可能樣本不足，請搭配 `n_samples` 一起判讀。
- XGBoost/CatBoost 若環境未安裝，會自動略過並記錄於 `run_config.json`。
