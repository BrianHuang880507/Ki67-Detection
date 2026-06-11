"""Ki67 新版預測流程。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ki67_pred_utils import (
    DEFAULT_PREDICT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TRAIN_OUTPUT_DIR,
    aggregate_to_image_features,
    attach_image_embeddings,
    evaluate_cell_predictions,
    evaluate_image_predictions,
    extract_image_embeddings,
    find_cleaned_csv_files,
    list_input_pc_image_stems,
    load_ki67_dataset,
    load_model_bundle,
    predict_feature_group_scores,
    predict_positive_probability,
    predict_stage2_ratio,
    reset_directory,
    transform_table_features,
)

MODEL_BUNDLE_PATH = DEFAULT_TRAIN_OUTPUT_DIR / "model" / "ki67_model_bundle.joblib"


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="Predict Ki67 positive ratio with the new pipeline.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-bundle", type=Path, default=MODEL_BUNDLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREDICT_OUTPUT_DIR)
    parser.add_argument("--source-folder", action="append", default=None)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--cnn-pretrained", action="store_true", default=None)
    parser.add_argument("--save-detail-table", action="store_true")
    return parser.parse_args()


def build_stage1_matrix_for_prediction(
    frame: pd.DataFrame,
    config: dict[str, Any],
    evidence_scores: pd.DataFrame,
    image_embeddings: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """依訓練 config 建立預測用 Stage 1 matrix。"""
    parts: list[pd.DataFrame] = []
    if config.get("include_feature_scores", False):
        parts.append(evidence_scores.reset_index(drop=True))
    if config.get("include_cnn_embedding", False):
        parts.append(attach_image_embeddings(frame, image_embeddings).reset_index(drop=True))
    if config.get("parameter_mode") == "selected":
        parts.append(transform_table_features(frame, bundle["selected_parameter_transformer"]).reset_index(drop=True))
    elif config.get("parameter_mode") == "all":
        parts.append(transform_table_features(frame, bundle["all_parameter_transformer"]).reset_index(drop=True))

    if not parts:
        raise ValueError("模型 bundle 的 Stage 1 設定沒有任何輸入欄位。")
    matrix = pd.concat(parts, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()].copy()
    return matrix.reindex(columns=bundle["stage1_feature_columns"], fill_value=0.0).fillna(0.0)


def add_folder_image_diagnostics(folder_df: pd.DataFrame, image_df: pd.DataFrame) -> pd.DataFrame:
    """加入 data/input PC 影像數量比對資訊。"""
    rows = []
    for _, row in folder_df.iterrows():
        source_folder = str(row["source_folder"])
        expected = set(list_input_pc_image_stems(source_folder))
        used = set(
            image_df.loc[image_df["source_folder"].astype(str) == source_folder, "image_name"].astype(str)
        )
        result = row.to_dict()
        result["input_image_count"] = len(expected)
        result["used_image_count"] = len(used)
        result["missing_image_count"] = len(expected - used)
        result["extra_image_count"] = len(used - expected)
        result["missing_images"] = ";".join(sorted(expected - used))
        result["extra_images"] = ";".join(sorted(used - expected))
        rows.append(result)
    return pd.DataFrame(rows)


def summarize_folder_predictions(image_df: pd.DataFrame) -> pd.DataFrame:
    """將 image-level prediction 彙整到 source folder。"""
    rows: list[dict[str, Any]] = []
    for source_folder, part in image_df.groupby("source_folder", sort=True):
        weights = part["cell_count"].to_numpy(dtype=np.float64)
        pred = np.average(part["pred_ratio"].to_numpy(dtype=np.float64), weights=weights)
        row = {
            "source_folder": source_folder,
            "image_count": int(part["image_key"].nunique()),
            "cell_count": int(part["cell_count"].sum()),
            "pred_ratio": float(pred),
        }
        if "true_ratio" in part.columns and part["true_ratio"].notna().any():
            true_ratio = np.average(part["true_ratio"].to_numpy(dtype=np.float64), weights=weights)
            row["true_ratio"] = float(true_ratio)
            row["abs_error"] = abs(float(pred) - float(true_ratio))
        rows.append(row)
    return pd.DataFrame(rows)


def run_prediction(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """執行新版 Ki67 預測。

    Args:
        args: 命令列參數，需包含 results-dir、model-bundle 與 output-dir。

    Returns:
        dict[str, pd.DataFrame]: cell、image 與 folder 三層預測結果。
    """
    if not args.model_bundle.exists():
        raise FileNotFoundError(f"找不到模型 bundle: {args.model_bundle}")
    bundle = load_model_bundle(args.model_bundle)

    csv_files = find_cleaned_csv_files(
        results_dir=args.results_dir,
        include_source_folders=args.source_folder,
    )
    dataset = load_ki67_dataset(csv_files, require_label=False)
    if len(dataset) == 0:
        raise ValueError("沒有可預測的 cleaned CSV。")

    output_dir = args.output_dir
    if args.keep_output:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        reset_directory(output_dir)

    evidence_scores = predict_feature_group_scores(dataset, bundle["feature_group_models"])
    config = dict(bundle["config"])
    if config.get("include_cnn_embedding", False):
        pretrained = bool(bundle.get("cnn_pretrained", False))
        if args.cnn_pretrained is not None:
            pretrained = bool(args.cnn_pretrained)
        image_embeddings, cnn_meta = extract_image_embeddings(
            dataset,
            pretrained=pretrained,
            random_state=int(bundle.get("random_state", 42)),
        )
        if image_embeddings.empty:
            raise RuntimeError(f"CNN embedding 為空，無法執行此模型。狀態: {cnn_meta}")
    else:
        image_embeddings = pd.DataFrame(index=[])

    x_pred = build_stage1_matrix_for_prediction(dataset, config, evidence_scores, image_embeddings, bundle)
    cell_prob = predict_positive_probability(bundle["stage1_model"], x_pred)
    cell_df = dataset.reset_index(drop=True).copy()
    for column in evidence_scores.columns:
        cell_df[column] = evidence_scores.reset_index(drop=True)[column].to_numpy(dtype=np.float64)
    cell_df["cell_prob"] = cell_prob
    cell_df["predicted_ki67_positive"] = (cell_df["cell_prob"] >= float(bundle.get("cell_threshold", 0.5))).astype(int)

    if "ki67_label" in cell_df.columns:
        cell_df["prediction_correct"] = cell_df["predicted_ki67_positive"] == cell_df["ki67_label"].astype(int)

    image_df = aggregate_to_image_features(cell_df, "cell_prob", list(evidence_scores.columns))
    image_df["pred_ratio"] = predict_stage2_ratio(
        image_df,
        bundle["stage2_model"],
        config["stage2_mode"],
        bundle["stage2_feature_columns"],
    )
    folder_df = summarize_folder_predictions(image_df)
    folder_df = add_folder_image_diagnostics(folder_df, image_df)

    cell_columns = [
        col
        for col in [
            "source_folder",
            "Image",
            "Cell_ID",
            "_source_row_id",
            "ki67_label",
            *list(evidence_scores.columns),
            "cell_prob",
            "predicted_ki67_positive",
            "prediction_correct",
        ]
        if col in cell_df.columns
    ]
    image_columns = [
        col
        for col in [
            "source_folder",
            "image_name",
            "image_key",
            "cell_count",
            "true_ratio",
            "pred_ratio",
            "mean_cell_prob",
            "median_cell_prob",
            "p90_cell_prob",
            "frac_prob_gt_0_3",
            "frac_prob_gt_0_5",
            "frac_prob_gt_0_7",
        ]
        if col in image_df.columns
    ]
    folder_df.to_csv(output_dir / "ki67_predictions.csv", index=False, encoding="utf-8-sig")

    if args.save_detail_table:
        folder_table = folder_df.copy()
        folder_table.insert(0, "level", "folder")
        image_table = image_df.loc[:, image_columns].copy()
        image_table.insert(0, "level", "image")
        cell_table = cell_df.loc[:, cell_columns].copy()
        cell_table.insert(0, "level", "cell")
        pd.concat([folder_table, image_table, cell_table], ignore_index=True, sort=False).to_csv(
            output_dir / "ki67_prediction_details.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if "ki67_label" in cell_df.columns and "true_ratio" in image_df.columns:
        cell_metrics = evaluate_cell_predictions(cell_df["ki67_label"], cell_df["cell_prob"])
        image_metrics = evaluate_image_predictions(image_df.dropna(subset=["true_ratio"]), image_df.dropna(subset=["true_ratio"])["pred_ratio"])
        print(
            "評估結果 | "
            f"cell accuracy={100 * cell_metrics['accuracy']:.2f}% | "
            f"image MAE={100 * image_metrics['image_mae']:.2f} pp"
        )

    print(f"新版預測輸出: {output_dir}")
    return {"cell": cell_df, "image": image_df, "folder": folder_df}


def main() -> None:
    """命令列入口。"""
    run_prediction(parse_args())


if __name__ == "__main__":
    main()
