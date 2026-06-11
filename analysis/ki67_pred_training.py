"""Ki67 新版 Stage 1 / Stage 2 訓練流程。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ki67_pred_utils import (
    DEFAULT_EXCLUDED_SOURCE_FOLDERS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SUMMARY_HTML,
    DEFAULT_TRAIN_OUTPUT_DIR,
    ExperimentConfig,
    aggregate_to_image_features,
    attach_image_embeddings,
    build_feature_groups,
    default_experiment_configs,
    detect_numeric_feature_columns,
    evaluate_cell_predictions,
    evaluate_image_predictions,
    extract_image_embeddings,
    filter_extreme_image_ratios,
    find_cleaned_csv_files,
    fit_feature_group_models,
    fit_selected_parameter_transformer,
    fit_stage1_classifier_with_oof,
    fit_stage2_ratio_model,
    fit_table_transformer,
    format_percent,
    format_pp,
    load_ki67_dataset,
    predict_feature_group_scores,
    predict_positive_probability,
    predict_stage2_ratio,
    render_summary_html,
    reset_directory,
    save_json,
    save_model_bundle,
    split_images_within_source_folder,
    summarize_split,
    transform_table_features,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)

RANDOM_STATE = 42
BASE_FEATURE_GROUPS = ("Texture", "Intensity distribution", "Halo / rounding")
FORMAL_CONFIG_KEY = "feature_cnn_s1"


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="Train the new Ki67 Stage 1 / Stage 2 pipeline.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument("--summary-html", type=Path, default=DEFAULT_SUMMARY_HTML)
    parser.add_argument("--selected-top-k", type=int, default=40)
    parser.add_argument("--formal-config", default=FORMAL_CONFIG_KEY)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--cnn-pretrained", action="store_true")
    parser.add_argument("--keep-extreme-image-ratios", action="store_true")
    parser.add_argument("--save-prediction-table", action="store_true")
    return parser.parse_args()


def build_stage1_matrix(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    evidence_scores: pd.DataFrame,
    image_embeddings: pd.DataFrame,
    selected_transformer: Any,
    all_parameter_transformer: Any,
) -> pd.DataFrame:
    """依實驗設定組合 Stage 1 input matrix。"""
    parts: list[pd.DataFrame] = []
    if config.include_feature_scores:
        parts.append(evidence_scores.reset_index(drop=True))
    if config.include_cnn_embedding:
        parts.append(attach_image_embeddings(frame, image_embeddings).reset_index(drop=True))
    if config.parameter_mode == "selected":
        parts.append(transform_table_features(frame, selected_transformer).reset_index(drop=True))
    elif config.parameter_mode == "all":
        parts.append(transform_table_features(frame, all_parameter_transformer).reset_index(drop=True))

    if not parts:
        raise ValueError(f"實驗 {config.key} 沒有任何 Stage 1 輸入欄位。")
    matrix = pd.concat(parts, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()].copy()
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def attach_scores_and_probability(
    frame: pd.DataFrame,
    probability: np.ndarray,
    evidence_scores: pd.DataFrame,
) -> pd.DataFrame:
    """建立含 evidence score 與 cell probability 的 cell-level table。"""
    scored = frame.reset_index(drop=True).copy()
    for column in evidence_scores.columns:
        scored[column] = evidence_scores.reset_index(drop=True)[column].to_numpy(dtype=np.float64)
    scored["cell_prob"] = np.asarray(probability, dtype=np.float64)
    return scored


def evaluate_config(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_evidence: pd.DataFrame,
    valid_evidence: pd.DataFrame,
    test_evidence: pd.DataFrame,
    image_embeddings: pd.DataFrame,
    selected_transformer: Any,
    all_parameter_transformer: Any,
    cv_splits: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """訓練並評估單一實驗設定。

    Args:
        config: 實驗設定。
        train_df: Train split cell-level table。
        valid_df: Valid split cell-level table。
        test_df: Test split cell-level table。
        train_evidence: Train OOF feature-group evidence scores。
        valid_evidence: Valid feature-group evidence scores。
        test_evidence: Test feature-group evidence scores。
        image_embeddings: image_key indexed CNN embedding table。
        selected_transformer: Selected parameters 前處理器。
        all_parameter_transformer: All parameters 前處理器。
        cv_splits: Stage 1 GroupKFold 切分數。

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: metric row 與可供部署的模型 artifacts。
    """
    x_train = build_stage1_matrix(
        train_df,
        config,
        train_evidence,
        image_embeddings,
        selected_transformer,
        all_parameter_transformer,
    )
    x_valid = build_stage1_matrix(
        valid_df,
        config,
        valid_evidence,
        image_embeddings,
        selected_transformer,
        all_parameter_transformer,
    ).reindex(columns=x_train.columns, fill_value=0.0)
    x_test = build_stage1_matrix(
        test_df,
        config,
        test_evidence,
        image_embeddings,
        selected_transformer,
        all_parameter_transformer,
    ).reindex(columns=x_train.columns, fill_value=0.0)

    stage1_model, train_prob = fit_stage1_classifier_with_oof(
        x_train,
        train_df["ki67_label"],
        train_df["image_key"].astype(str),
        random_state=RANDOM_STATE,
        cv_splits=cv_splits,
    )
    valid_prob = predict_positive_probability(stage1_model, x_valid)
    test_prob = predict_positive_probability(stage1_model, x_test)

    train_cell = attach_scores_and_probability(train_df, train_prob, train_evidence)
    valid_cell = attach_scores_and_probability(valid_df, valid_prob, valid_evidence)
    test_cell = attach_scores_and_probability(test_df, test_prob, test_evidence)
    evidence_columns = list(train_evidence.columns)

    train_image = aggregate_to_image_features(train_cell, "cell_prob", evidence_columns)
    valid_image = aggregate_to_image_features(valid_cell, "cell_prob", evidence_columns)
    test_image = aggregate_to_image_features(test_cell, "cell_prob", evidence_columns)

    stage2_model, stage2_columns = fit_stage2_ratio_model(train_image, config.stage2_mode)
    train_image["pred_ratio"] = predict_stage2_ratio(train_image, stage2_model, config.stage2_mode, stage2_columns)
    valid_image["pred_ratio"] = predict_stage2_ratio(valid_image, stage2_model, config.stage2_mode, stage2_columns)
    test_image["pred_ratio"] = predict_stage2_ratio(test_image, stage2_model, config.stage2_mode, stage2_columns)

    train_cell_metrics = evaluate_cell_predictions(train_df["ki67_label"], train_prob)
    valid_cell_metrics = evaluate_cell_predictions(valid_df["ki67_label"], valid_prob)
    test_cell_metrics = evaluate_cell_predictions(test_df["ki67_label"], test_prob)
    train_image_metrics = evaluate_image_predictions(train_image, train_image["pred_ratio"])
    valid_image_metrics = evaluate_image_predictions(valid_image, valid_image["pred_ratio"])
    test_image_metrics = evaluate_image_predictions(test_image, test_image["pred_ratio"])

    row = {
        "config_key": config.key,
        "display_name": config.display_name,
        "stage1_input": config.stage1_input_name,
        "stage2_input": config.stage2_name,
        "train_cell_accuracy": train_cell_metrics["accuracy"],
        "valid_cell_accuracy": valid_cell_metrics["accuracy"],
        "test_cell_accuracy": test_cell_metrics["accuracy"],
        "train_image_mae": train_image_metrics["image_mae"],
        "valid_image_mae": valid_image_metrics["image_mae"],
        "test_image_mae": test_image_metrics["image_mae"],
        "train_image_bias": train_image_metrics["image_bias"],
        "valid_image_bias": valid_image_metrics["image_bias"],
        "test_image_bias": test_image_metrics["image_bias"],
        "test_tn": test_cell_metrics["tn"],
        "test_fp": test_cell_metrics["fp"],
        "test_fn": test_cell_metrics["fn"],
        "test_tp": test_cell_metrics["tp"],
    }
    artifacts = {
        "config": asdict(config),
        "stage1_model": stage1_model,
        "stage1_feature_columns": list(x_train.columns),
        "stage2_model": stage2_model,
        "stage2_feature_columns": list(stage2_columns),
        "train_cell_predictions": train_cell,
        "valid_cell_predictions": valid_cell,
        "test_cell_predictions": test_cell,
        "train_image_predictions": train_image,
        "valid_image_predictions": valid_image,
        "test_image_predictions": test_image,
    }
    return row, artifacts


def prepare_training_data(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """讀取與切分新版訓練資料。"""
    csv_files = find_cleaned_csv_files(
        results_dir=args.results_dir,
        excluded_source_folders=DEFAULT_EXCLUDED_SOURCE_FOLDERS,
    )
    dataset = load_ki67_dataset(csv_files, require_label=True)
    if not args.keep_extreme_image_ratios:
        dataset = filter_extreme_image_ratios(dataset)
    splits = split_images_within_source_folder(dataset, random_state=RANDOM_STATE)
    feature_columns = detect_numeric_feature_columns(dataset)
    return splits, feature_columns


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """執行完整新版訓練流程。"""
    output_dir = args.output_dir
    model_dir = output_dir / "model"
    reset_directory(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits, feature_columns = prepare_training_data(args)
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]
    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError("Train / valid / test 任一切分為空，請檢查 source folder 設定。")

    feature_groups = build_feature_groups(feature_columns)
    feature_group_models, train_evidence = fit_feature_group_models(
        train_df,
        feature_groups,
        BASE_FEATURE_GROUPS,
        cv_splits=args.cv_splits,
        random_state=RANDOM_STATE,
        group_column="image_key",
    )
    valid_evidence = predict_feature_group_scores(valid_df, feature_group_models)
    test_evidence = predict_feature_group_scores(test_df, feature_group_models)

    selected_transformer = fit_selected_parameter_transformer(train_df, feature_columns, top_k=args.selected_top_k)
    all_parameter_transformer = fit_table_transformer(train_df, feature_columns)

    all_for_embedding = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    if args.skip_cnn:
        image_embeddings = pd.DataFrame(index=[])
        cnn_meta = {"status": "skipped"}
    else:
        image_embeddings, cnn_meta = extract_image_embeddings(
            all_for_embedding,
            pretrained=bool(args.cnn_pretrained),
            random_state=RANDOM_STATE,
        )
    if image_embeddings.empty:
        print("[警告] CNN embedding 為空；包含 CNN 的實驗會使用空矩陣並可能失敗。")

    configs = default_experiment_configs()
    metrics: list[dict[str, Any]] = []
    artifact_by_key: dict[str, dict[str, Any]] = {}
    for config in configs:
        if config.include_cnn_embedding and image_embeddings.empty:
            print(f"[略過] {config.display_name}: 沒有 CNN embedding。")
            continue
        row, artifacts = evaluate_config(
            config,
            train_df,
            valid_df,
            test_df,
            train_evidence,
            valid_evidence,
            test_evidence,
            image_embeddings,
            selected_transformer,
            all_parameter_transformer,
            cv_splits=args.cv_splits,
        )
        metrics.append(row)
        artifact_by_key[config.key] = artifacts

    if not metrics:
        raise RuntimeError("沒有任何實驗成功完成。")
    metrics_df = pd.DataFrame(metrics).sort_values("test_image_mae").reset_index(drop=True)
    formal_key = args.formal_config if args.formal_config in artifact_by_key else str(metrics_df.iloc[0]["config_key"])
    best_artifacts = artifact_by_key[formal_key]
    split_df = pd.DataFrame(
        [
            summarize_split(train_df, "train"),
            summarize_split(valid_df, "valid"),
            summarize_split(test_df, "test"),
        ]
    )

    metrics_out = metrics_df.copy()
    metrics_out.insert(0, "item", "experiment_metric")
    split_out = split_df.copy()
    split_out.insert(0, "item", "split_summary")
    training_summary = pd.concat([metrics_out, split_out], ignore_index=True, sort=False)
    training_summary.to_csv(output_dir / "ki67_training_summary.csv", index=False, encoding="utf-8-sig")

    if args.save_prediction_table:
        prediction_tables: list[pd.DataFrame] = []
        for split_name in ("train", "valid", "test"):
            cell_table = best_artifacts[f"{split_name}_cell_predictions"].copy()
            cell_table.insert(0, "level", "cell")
            cell_table.insert(0, "split", split_name)
            image_table = best_artifacts[f"{split_name}_image_predictions"].copy()
            image_table.insert(0, "level", "image")
            image_table.insert(0, "split", split_name)
            prediction_tables.extend([cell_table, image_table])
        pd.concat(prediction_tables, ignore_index=True, sort=False).to_csv(
            output_dir / "ki67_training_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    model_bundle = {
        "version": "feature_group_evidence_cnn_s1",
        "formal_config_key": formal_key,
        "random_state": RANDOM_STATE,
        "feature_columns": list(feature_columns),
        "feature_group_names": list(BASE_FEATURE_GROUPS),
        "feature_groups": {name: list(cols) for name, cols in feature_groups.items()},
        "feature_group_models": feature_group_models,
        "selected_parameter_transformer": selected_transformer,
        "all_parameter_transformer": all_parameter_transformer,
        "stage1_model": best_artifacts["stage1_model"],
        "stage1_feature_columns": best_artifacts["stage1_feature_columns"],
        "stage2_model": best_artifacts["stage2_model"],
        "stage2_feature_columns": best_artifacts["stage2_feature_columns"],
        "config": best_artifacts["config"],
        "cnn_meta": cnn_meta,
        "cnn_pretrained": bool(args.cnn_pretrained),
        "cell_threshold": 0.5,
    }
    save_model_bundle(model_dir / "ki67_model_bundle.joblib", model_bundle)
    save_json(
        model_dir / "training_meta.json",
        {
            "formal_config_key": formal_key,
            "cnn_meta": cnn_meta,
            "selected_top_k": int(args.selected_top_k),
            "cv_splits": int(args.cv_splits),
            "summary_html": str(args.summary_html),
        },
    )

    render_summary_html(args.summary_html, metrics_df, split_df, formal_key)

    best_row = metrics_df[metrics_df["config_key"] == formal_key].iloc[0]
    print("===== Ki67 新版訓練完成 =====")
    print(f"模型 bundle: {model_dir / 'ki67_model_bundle.joblib'}")
    print(f"summary.html: {args.summary_html}")
    print(
        f"正式組合: {best_row['display_name']} | "
        f"test image MAE={format_pp(best_row['test_image_mae'])} | "
        f"test cell acc={format_percent(best_row['test_cell_accuracy'])}"
    )
    return {
        "metrics": metrics_df,
        "split_summary": split_df,
        "model_bundle": model_bundle,
        "output_dir": output_dir,
    }


def main() -> None:
    """命令列入口。"""
    run_training(parse_args())


if __name__ == "__main__":
    main()
