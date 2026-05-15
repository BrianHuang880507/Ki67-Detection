"""Ki67 正式訓練流程。

此版本包含三個重點:
1. 正式驗證改成 leave-source-folder-out。
2. stage1 加入 probability calibration。
3. 自動比較 `class_weight=True/False` 與不同 stage2 模型。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

from ki67_pred_utils import (
    DEFAULT_BDL_RATIO_REFERENCE,
    DEFAULT_DATA_PATTERN,
    DEFAULT_EXCLUDED_SOURCE_FOLDERS,
    DEFAULT_HOLDOUT_SOURCE_FOLDERS,
    DEFAULT_LABEL_CANDIDATES,
    DEFAULT_PASSAGE_CANDIDATES,
    aggregate_to_image_features,
    apply_probability_calibration,
    build_image_design_matrices,
    compare_folder_ratio_to_reference,
    detect_numeric_feature_columns,
    evaluate_ratio_predictions,
    find_cleaned_csv_files,
    filter_similar_features,
    fit_preprocessor,
    fit_probability_calibrator,
    fit_ratio_models,
    fit_stage1_cell_model_with_oof,
    load_ki67_dataset,
    predict_stage1_probability,
    save_calibration_curve_plot,
    save_config_comparison_plot,
    save_feature_correlation_plot,
    save_json,
    select_positive_correlation_features,
    split_dev_and_holdout,
    summarize_folder_ratio,
    transform_features,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "output" / "results"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "train" / "ki67_pred"
MODEL_DIR = OUTPUT_ROOT / "model"
REPORT_DIR = OUTPUT_ROOT / "report"

DATA_PATTERN = DEFAULT_DATA_PATTERN
EXCLUDED_SOURCE_FOLDERS = sorted(DEFAULT_EXCLUDED_SOURCE_FOLDERS)
HOLDOUT_SOURCE_FOLDERS = list(DEFAULT_HOLDOUT_SOURCE_FOLDERS)
LABEL_CANDIDATES = list(DEFAULT_LABEL_CANDIDATES)
PASSAGE_CANDIDATES = list(DEFAULT_PASSAGE_CANDIDATES)
BDL_RATIO_REFERENCE = dict(DEFAULT_BDL_RATIO_REFERENCE)

RANDOM_STATE = 42
SIMILARITY_THRESHOLD = 0.90
POSITIVE_CORR_THRESHOLD = 0.00
STAGE1_CALIBRATION_METHOD = "isotonic"
CLASS_WEIGHT_OPTIONS = [False, True]
FORMAL_SELECTED_CONFIG_NAME = "cw_false__LightGBMRegressor"


def ensure_clean_output_dirs() -> None:
    """清空並重建正式訓練輸出目錄。"""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_pipeline(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, Any]:
    """只用訓練資料建立前處理與特徵選擇。

    Args:
        train_df: 訓練資料。
        feature_cols: 原始數值特徵欄位。

    Returns:
        前處理與最終特徵相關資訊。
    """
    prep = fit_preprocessor(train_df, feature_cols)
    x_train_all = transform_features(train_df, feature_cols, prep)

    similarity_result = filter_similar_features(
        x_train=x_train_all,
        y_train=train_df["ki67_label"].astype(int),
        threshold=SIMILARITY_THRESHOLD,
    )
    kept_after_similarity = similarity_result["kept_features"]
    x_train_sim = x_train_all[kept_after_similarity].copy()

    corr_result = select_positive_correlation_features(
        x_train=x_train_sim,
        y_train=train_df["ki67_label"].astype(int),
        min_corr=POSITIVE_CORR_THRESHOLD,
    )
    selected_features = corr_result["selected_features"]

    return {
        "prep": prep,
        "correlation_report": corr_result["correlation_report"],
        "selected_features": selected_features,
        "x_train_final": x_train_sim[selected_features].copy(),
    }


def transform_selected_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    prep: Dict[str, Any],
    selected_features: Sequence[str],
) -> pd.DataFrame:
    """將資料轉成最終特徵空間。

    Args:
        df: 待轉換資料。
        feature_cols: 原始特徵欄位。
        prep: 前處理器。
        selected_features: 最終選用特徵。

    Returns:
        最終特徵矩陣。
    """
    x_all = transform_features(df, feature_cols, prep)
    return x_all.loc[:, list(selected_features)].copy()


def fit_stage1_with_calibration(
    train_df: pd.DataFrame,
    x_train_final: pd.DataFrame,
    use_class_weight: bool,
) -> Dict[str, Any]:
    """訓練 stage1 並用 OOF probability 擬合 calibrator。

    Args:
        train_df: 訓練資料。
        x_train_final: stage1 最終特徵。
        use_class_weight: 是否使用 `balanced`。

    Returns:
        stage1 模型、校正器與校正前後 OOF probability。
    """
    stage1_result = fit_stage1_cell_model_with_oof(
        x_train=x_train_final,
        y_train=train_df["ki67_label"].astype(int),
        image_keys=train_df["image_key"].astype(str),
        random_state=RANDOM_STATE,
        use_class_weight=use_class_weight,
    )
    raw_oof_prob = np.asarray(stage1_result["train_oof_prob"], dtype=np.float64)
    calibrator = fit_probability_calibrator(
        raw_prob=raw_oof_prob,
        y_true=train_df["ki67_label"].to_numpy(dtype=np.int64),
        method=STAGE1_CALIBRATION_METHOD,
    )
    calibrated_oof_prob = apply_probability_calibration(calibrator, raw_oof_prob)
    return {
        "stage1_model": stage1_result["model"],
        "calibrator": calibrator,
        "raw_oof_prob": raw_oof_prob,
        "calibrated_oof_prob": calibrated_oof_prob,
        "n_splits": int(stage1_result["n_splits"]),
    }


def predict_calibrated_stage1(
    stage1_model: Any,
    calibrator: Any,
    x_df: pd.DataFrame,
) -> np.ndarray:
    """輸出校正後的 stage1 機率。

    Args:
        stage1_model: stage1 模型。
        calibrator: 機率校正器。
        x_df: 特徵矩陣。

    Returns:
        校正後的陽性機率。
    """
    raw_prob = predict_stage1_probability(stage1_model, x_df)
    return apply_probability_calibration(calibrator, raw_prob)


def evaluate_single_config_on_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    use_class_weight: bool,
    stage2_model_name: str,
) -> Dict[str, Any]:
    """在單一 leave-source-folder-out fold 上評估一組設定。

    Args:
        train_df: 訓練資料。
        val_df: 驗證資料，應為單一 source_folder。
        feature_cols: 原始特徵欄位。
        use_class_weight: stage1 是否使用 `balanced`。
        stage2_model_name: stage2 模型名稱。

    Returns:
        單 fold 評估結果。
    """
    feature_pipeline = build_feature_pipeline(train_df, feature_cols)
    x_train_final = feature_pipeline["x_train_final"]
    x_val_final = transform_selected_features(
        df=val_df,
        feature_cols=feature_cols,
        prep=feature_pipeline["prep"],
        selected_features=feature_pipeline["selected_features"],
    )

    stage1_artifacts = fit_stage1_with_calibration(
        train_df=train_df,
        x_train_final=x_train_final,
        use_class_weight=use_class_weight,
    )

    train_scored_df = train_df.copy()
    train_scored_df["cell_prob"] = stage1_artifacts["calibrated_oof_prob"]

    val_scored_df = val_df.copy()
    val_scored_df["cell_prob"] = predict_calibrated_stage1(
        stage1_model=stage1_artifacts["stage1_model"],
        calibrator=stage1_artifacts["calibrator"],
        x_df=x_val_final,
    )

    train_image_df = aggregate_to_image_features(train_scored_df, prob_col="cell_prob")
    val_image_df = aggregate_to_image_features(val_scored_df, prob_col="cell_prob")
    design_result = build_image_design_matrices({"Train": train_image_df, "Val": val_image_df})

    ratio_model = fit_ratio_models(RANDOM_STATE)[stage2_model_name]
    ratio_model.fit(
        design_result["matrices"]["Train"],
        train_image_df["true_ratio"].to_numpy(dtype=np.float64),
    )
    val_pred = np.clip(ratio_model.predict(design_result["matrices"]["Val"]), 0.0, 1.0)

    image_metrics = evaluate_ratio_predictions(
        y_true=val_image_df["true_ratio"].to_numpy(dtype=np.float64),
        y_pred=val_pred,
    )
    val_image_eval_df = val_image_df.copy()
    val_image_eval_df["pred_ratio"] = val_pred
    folder_df = summarize_folder_ratio(val_image_eval_df)

    return {
        "fold_source_folder": str(val_df["source_folder"].iloc[0]),
        "stage1_use_class_weight": bool(use_class_weight),
        "stage2_model": stage2_model_name,
        "image_mae": float(image_metrics["mae"]),
        "image_rmse": float(image_metrics["rmse"]),
        "folder_mae": float(folder_df["abs_error"].mean()),
        "folder_within_0p05": float((folder_df["abs_error"] <= 0.05).mean()),
        "folder_within_0p10": float((folder_df["abs_error"] <= 0.10).mean()),
    }


def run_leave_source_folder_out_cv(
    dev_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """執行 leave-source-folder-out 正式驗證。

    Args:
        dev_df: 開發資料。
        feature_cols: 原始特徵欄位。

    Returns:
        每個 fold 的詳細評估表。
    """
    rows: List[Dict[str, Any]] = []
    source_folders = sorted(dev_df["source_folder"].astype(str).unique().tolist())
    stage2_model_names = list(fit_ratio_models(RANDOM_STATE).keys())

    for fold_source_folder in source_folders:
        train_df = dev_df[dev_df["source_folder"] != fold_source_folder].copy().reset_index(drop=True)
        val_df = dev_df[dev_df["source_folder"] == fold_source_folder].copy().reset_index(drop=True)

        for use_class_weight in CLASS_WEIGHT_OPTIONS:
            for stage2_model_name in stage2_model_names:
                rows.append(
                    evaluate_single_config_on_fold(
                        train_df=train_df,
                        val_df=val_df,
                        feature_cols=feature_cols,
                        use_class_weight=use_class_weight,
                        stage2_model_name=stage2_model_name,
                    )
                )

    return pd.DataFrame(rows)


def train_final_pipeline(
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: Sequence[str],
    use_class_weight: bool,
    stage2_model_name: str,
) -> Dict[str, Any]:
    """用完整 dev 資料訓練正式模型，並對 holdout 做推論。

    Args:
        dev_df: 開發資料。
        holdout_df: 正式預測資料。
        feature_cols: 原始特徵欄位。
        use_class_weight: stage1 是否使用 `balanced`。
        stage2_model_name: stage2 模型名稱。

    Returns:
        正式模型、報表與 holdout 比對結果。
    """
    feature_pipeline = build_feature_pipeline(dev_df, feature_cols)
    x_dev_final = feature_pipeline["x_train_final"]

    stage1_artifacts = fit_stage1_with_calibration(
        train_df=dev_df,
        x_train_final=x_dev_final,
        use_class_weight=use_class_weight,
    )

    dev_scored_df = dev_df.copy()
    dev_scored_df["cell_prob"] = stage1_artifacts["calibrated_oof_prob"]

    holdout_pred_df = holdout_df.drop(columns=["ki67_label"], errors="ignore").copy()
    if len(holdout_pred_df) > 0:
        x_holdout_final = transform_selected_features(
            df=holdout_pred_df,
            feature_cols=feature_cols,
            prep=feature_pipeline["prep"],
            selected_features=feature_pipeline["selected_features"],
        )
        holdout_pred_df["cell_prob"] = predict_calibrated_stage1(
            stage1_model=stage1_artifacts["stage1_model"],
            calibrator=stage1_artifacts["calibrator"],
            x_df=x_holdout_final,
        )

    dev_image_df = aggregate_to_image_features(dev_scored_df, prob_col="cell_prob")
    holdout_image_df = (
        aggregate_to_image_features(holdout_pred_df, prob_col="cell_prob")
        if len(holdout_pred_df) > 0
        else pd.DataFrame()
    )

    design_result = build_image_design_matrices(
        image_tables={"Train": dev_image_df, "Holdout": holdout_image_df}
    )
    x_train_img = design_result["matrices"]["Train"]
    x_holdout_img = design_result["matrices"]["Holdout"]

    stage2_model = fit_ratio_models(RANDOM_STATE)[stage2_model_name]
    stage2_model.fit(x_train_img, dev_image_df["true_ratio"].to_numpy(dtype=np.float64))

    holdout_compare_df = pd.DataFrame()
    if len(holdout_image_df) > 0:
        holdout_image_df = holdout_image_df.copy()
        holdout_image_df["pred_ratio"] = np.clip(stage2_model.predict(x_holdout_img), 0.0, 1.0)
        holdout_folder_df = summarize_folder_ratio(holdout_image_df)
        holdout_compare_df = compare_folder_ratio_to_reference(
            folder_ratio_df=holdout_folder_df,
            reference_map=BDL_RATIO_REFERENCE,
            reference_col_name="bdl_true_ratio",
        )

    return {
        "feature_pipeline": feature_pipeline,
        "stage1_artifacts": stage1_artifacts,
        "stage2_model": stage2_model,
        "design_feature_names": design_result["feature_names"],
        "holdout_compare_df": holdout_compare_df,
        "dev_image_df": dev_image_df,
    }


def build_source_folder_usage(dev_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    """建立正式訓練資料夾使用摘要。

    Args:
        dev_df: 開發資料。
        holdout_df: holdout 資料。

    Returns:
        source_folder 使用摘要。
    """
    rows: List[Dict[str, Any]] = []
    for dataset_role, part_df in [("DevTrain", dev_df), ("HoldoutPredict", holdout_df)]:
        if len(part_df) == 0:
            continue
        for source_folder, folder_df in part_df.groupby("source_folder", sort=True):
            row = {
                "dataset_role": dataset_role,
                "source_folder": source_folder,
                "image_count": int(folder_df["image_key"].nunique()),
                "cell_count": int(len(folder_df)),
                "positive_ratio": np.nan if dataset_role == "HoldoutPredict" else float(folder_df["ki67_label"].mean()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """執行正式訓練。"""
    ensure_clean_output_dirs()

    csv_files = find_cleaned_csv_files(
        results_dir=RESULTS_DIR,
        pattern=DATA_PATTERN,
        excluded_source_folders=EXCLUDED_SOURCE_FOLDERS,
    )
    dataset = load_ki67_dataset(
        csv_files=csv_files,
        label_candidates=LABEL_CANDIDATES,
        passage_candidates=PASSAGE_CANDIDATES,
        require_label=True,
    )
    dataset["image_key"] = dataset["source_folder"].astype(str) + "::" + dataset["Image"].astype(str)
    feature_cols = detect_numeric_feature_columns(dataset)

    split_result = split_dev_and_holdout(dataset, HOLDOUT_SOURCE_FOLDERS)
    dev_df = split_result["dev_df"].copy().reset_index(drop=True)
    holdout_df = split_result["holdout_df"].copy().reset_index(drop=True)

    lfso_detail_df = run_leave_source_folder_out_cv(dev_df=dev_df, feature_cols=feature_cols)
    config_summary_df = (
        lfso_detail_df.groupby(["stage1_use_class_weight", "stage2_model"], as_index=False)
        .agg(
            lfso_image_mae=("image_mae", "mean"),
            lfso_image_rmse=("image_rmse", "mean"),
            lfso_folder_mae=("folder_mae", "mean"),
            lfso_folder_within_0p05=("folder_within_0p05", "mean"),
            lfso_folder_within_0p10=("folder_within_0p10", "mean"),
        )
        .sort_values(["lfso_folder_mae", "lfso_image_mae", "stage2_model"])
        .reset_index(drop=True)
    )
    config_summary_df["config_name"] = config_summary_df.apply(
        lambda row: f"cw_{str(bool(row['stage1_use_class_weight'])).lower()}__{row['stage2_model']}",
        axis=1,
    )

    final_results: Dict[str, Dict[str, Any]] = {}
    holdout_bdl_mae_list: List[float] = []
    for _, row in config_summary_df.iterrows():
        config_name = row["config_name"]
        final_result = train_final_pipeline(
            dev_df=dev_df,
            holdout_df=holdout_df,
            feature_cols=feature_cols,
            use_class_weight=bool(row["stage1_use_class_weight"]),
            stage2_model_name=str(row["stage2_model"]),
        )
        final_results[config_name] = final_result
        compare_df = final_result["holdout_compare_df"]
        holdout_bdl_mae = float(compare_df["abs_error"].mean()) if len(compare_df) > 0 else np.nan
        holdout_bdl_mae_list.append(holdout_bdl_mae)

    config_summary_df["holdout_bdl_mae"] = holdout_bdl_mae_list

    auto_best_row = config_summary_df.sort_values(
        ["lfso_folder_mae", "lfso_image_mae", "stage2_model"]
    ).iloc[0]
    auto_best_config_name = str(auto_best_row["config_name"])
    formal_match_df = config_summary_df[config_summary_df["config_name"] == FORMAL_SELECTED_CONFIG_NAME].copy()
    if len(formal_match_df) != 1:
        raise ValueError(f"找不到正式指定組態: {FORMAL_SELECTED_CONFIG_NAME}")

    best_row = formal_match_df.iloc[0]
    best_config_name = str(best_row["config_name"])
    best_final_result = final_results[best_config_name]
    best_holdout_bdl_config_name = str(
        config_summary_df.sort_values(["holdout_bdl_mae", "lfso_folder_mae"]).iloc[0]["config_name"]
    )

    joblib.dump(best_final_result["stage1_artifacts"]["stage1_model"], MODEL_DIR / "stage1_cell_model.joblib")
    joblib.dump(best_final_result["stage1_artifacts"]["calibrator"], MODEL_DIR / "stage1_calibrator.joblib")
    joblib.dump(best_final_result["stage2_model"], MODEL_DIR / "stage2_ratio_model.joblib")
    joblib.dump(
        {
            "imputer": best_final_result["feature_pipeline"]["prep"]["imputer"],
            "variance_selector": best_final_result["feature_pipeline"]["prep"]["variance_selector"],
            "scaler": best_final_result["feature_pipeline"]["prep"]["scaler"],
            "kept_features": best_final_result["feature_pipeline"]["prep"]["kept_features"],
            "raw_feature_columns": feature_cols,
            "selected_features": best_final_result["feature_pipeline"]["selected_features"],
        },
        MODEL_DIR / "preprocess_bundle.joblib",
    )
    save_json(
        MODEL_DIR / "ratio_feature_columns.json",
        {"feature_names": best_final_result["design_feature_names"]},
    )
    save_json(
        MODEL_DIR / "training_meta.json",
        {
            "selected_config_name": best_config_name,
            "selected_by": "manual_override",
            "formal_selected_config_name": FORMAL_SELECTED_CONFIG_NAME,
            "auto_best_config_name": auto_best_config_name,
            "closest_bdl_config_name": best_holdout_bdl_config_name,
            "stage1_model_name": "SGDClassifier(log_loss)",
            "stage1_calibration_method": STAGE1_CALIBRATION_METHOD,
            "stage1_use_class_weight": bool(best_row["stage1_use_class_weight"]),
            "stage2_model_name": str(best_row["stage2_model"]),
            "random_state": RANDOM_STATE,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "positive_corr_threshold": POSITIVE_CORR_THRESHOLD,
            "excluded_source_folders": EXCLUDED_SOURCE_FOLDERS,
            "holdout_source_folders": HOLDOUT_SOURCE_FOLDERS,
            "selected_features": best_final_result["feature_pipeline"]["selected_features"],
            "raw_feature_columns": feature_cols,
        },
    )

    save_json(
        REPORT_DIR / "training_summary.json",
        {
            "csv_file_count": len(csv_files),
            "total_rows": int(len(dataset)),
            "dev_rows": int(len(dev_df)),
            "holdout_rows": int(len(holdout_df)),
            "raw_feature_count": len(feature_cols),
            "selected_feature_count": len(best_final_result["feature_pipeline"]["selected_features"]),
            "selected_config_name": best_config_name,
            "auto_best_config_name": auto_best_config_name,
            "closest_bdl_config_name": best_holdout_bdl_config_name,
            "selected_config_lfso_folder_mae": float(best_row["lfso_folder_mae"]),
            "selected_config_holdout_bdl_mae": float(best_row["holdout_bdl_mae"]),
            "stage1_group_cv_splits": int(best_final_result["stage1_artifacts"]["n_splits"]),
        },
    )

    print("===== Ki67 正式訓練完成 =====")
    print(f"輸出目錄: {OUTPUT_ROOT}")
    print(f"正式選用組態: {best_config_name}")
    print(f"自動最佳組態: {auto_best_config_name}")
    print(f"最接近 BDL 的組態: {best_holdout_bdl_config_name}")
    print("===== 正式選用組態指標 =====")
    print(pd.DataFrame([best_row]).to_string(index=False))


if __name__ == "__main__":
    main()
