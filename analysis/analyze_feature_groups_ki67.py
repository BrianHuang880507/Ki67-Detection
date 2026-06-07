"""Analyze associations between PC-derived feature groups and Ki67 labels.

The analysis deliberately separates two grains:
1. Cell-local parameters are evaluated against the cell-level Ki67 label.
2. FOV/context parameters are evaluated against image-level Ki67 positive ratio.

All predictive estimates use source-folder grouped cross-validation so cells from
the same experimental folder never appear in both train and validation folds.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "output" / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "ki67_feature_relation"
LABEL_COLUMN = "ki67_positive"
RANDOM_STATE = 42
MAX_VALID_KI67_MASK_FRACTION = 0.10

GEOMETRY_BASES = [
    "Area",
    "Perimeter",
    "Convex Perimeter",
    "Circular Diameter",
    "Feret Length",
    "Feret Width",
    "Aspect Ratio",
    "Eccentricity",
    "Roundness",
    "Circularity",
    "Sphericity",
    "Roughness",
]
INTENSITY_BASES = [
    "Mean",
    "StdDev",
    "Min",
    "Max",
    "IntDen",
    "RawIntDen",
    "CV",
    "Range",
    "P10",
    "P25",
    "P75",
    "P90",
    "IQR80",
    "Entropy",
    "Skewness",
    "Kurtosis",
]
DERIVED_INTENSITY_COLUMNS = [
    "Nuc Cyto Mean Ratio",
    "Nuc Cyto IntDen Ratio",
    "Nuc Cyto RawIntDen Ratio",
    "Nuc Cell IntDen Ratio",
    "Nuc Cyto Entropy Difference",
    "Nuc Cyto CV Difference",
]
HALO_COLUMNS = [
    "Halo Outer Mean",
    "Halo Outer StdDev",
    "Halo Outer CV",
    "Halo Inner Mean",
    "Halo Inner StdDev",
    "Halo Inner Outer Diff",
    "Halo Angular Variance",
    "Halo Radial Gradient",
    "Halo Width",
    "Edge Sharpness",
]
SHAPE_COLUMNS = [
    "Protrusion Count",
    "Mean Convex Defect Depth",
    "Mean Protrusion Length Norm",
    "Max Convex Defect Depth",
    "Fractal Dimension",
    "Boundary Inflection Count",
]
LOCAL_CROWDING_COLUMNS = [
    "Nearest Neighbor Distance",
    "Nearest Neighbor Distance Norm",
    "Local Neighbor Count",
    "Local Density",
    "Neighbour Area Ratio",
]
COLONY_CONTEXT_COLUMNS = [
    "Image Confluency",
    "Population Area CV",
    "Population Circularity CV",
    "Cluster Size",
    "Cluster Size Norm",
    "Largest Cluster Ratio",
]
MITOSIS_COLUMNS = [
    "Mitotic Score",
    "Daughter Pair Flag",
    "Protrusion Retraction Score",
    "Mitotic Index",
]
DEBRIS_COLUMNS = [
    "Debris Count",
    "Debris Area Fraction",
    "Nearest Debris Distance",
    "Debris Mean Area",
    "Debris Density",
]
NUCLEOLUS_COLUMNS = [
    "Nucleolus Count",
    "Mean Nucleolus Area",
    "Max Nucleolus Area",
]
FOV_CONTEXT_GROUPS = {
    "Local crowding": LOCAL_CROWDING_COLUMNS,
    "Colony / FOV context": COLONY_CONTEXT_COLUMNS,
    "Debris / culture health": DEBRIS_COLUMNS,
    "Mitosis context": ["Mitotic Index"],
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "blue_light": "#A3BEFA",
    "orange": "#CC6F47",
    "orange_light": "#F0986E",
    "gold": "#B8A037",
    "neutral": "#C5CAD3",
}


@dataclass
class DatasetBundle:
    all_labeled: pd.DataFrame
    primary: pd.DataFrame
    inventory: pd.DataFrame


def ki67_mask_coverage_by_pc_image(
    source_folder: str,
) -> dict[str, float]:
    """Map PC image stems to Ki67 binary-mask foreground fractions."""
    input_dir = PROJECT_ROOT / "data" / "input" / source_folder
    binary_dir = PROJECT_ROOT / "data" / "output" / "binary" / source_folder
    mapping_path = input_dir / "image_mapping.csv"
    if not mapping_path.exists() or not binary_dir.exists():
        return {}

    mapping = pd.read_csv(mapping_path)
    if "PC_Name" not in mapping or "KI67_Name" not in mapping:
        return {}

    coverage: dict[str, float] = {}
    for _, row in mapping.iterrows():
        pc_name = str(row.get("PC_Name", "")).strip()
        ki67_name = str(row.get("KI67_Name", "")).strip()
        if not pc_name or not ki67_name or pc_name == "nan" or ki67_name == "nan":
            continue
        mask_path = binary_dir / f"{Path(ki67_name).stem}_binary.png"
        if not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"))
        coverage[Path(pc_name).stem] = float(np.count_nonzero(mask) / mask.size)
    return coverage


def outline_pair_quality(source_folder: str) -> dict[str, float | int]:
    """Summarize whether nucleus and whole-cell outlines are distinguishable."""
    outline_dir = (
        PROJECT_ROOT / "data" / "output" / "outline" / source_folder
    )
    total_pairs = 0
    paired_outlines = 0
    identical_pairs = 0
    for outline_path in outline_dir.glob("*_merged_cp_outlines.txt"):
        lines = [
            line.strip()
            for line in outline_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            if line.strip()
        ]
        for index in range(0, len(lines) - 1, 2):
            nucleus = lines[index]
            cell = lines[index + 1]
            total_pairs += 1
            if nucleus == "-1,-1" or cell == "-1,-1":
                continue
            paired_outlines += 1
            identical_pairs += int(nucleus == cell)
    return {
        "outline_pairs": total_pairs,
        "paired_outlines": paired_outlines,
        "identical_outline_pairs": identical_pairs,
        "identical_outline_pair_fraction": (
            float(identical_pairs / paired_outlines)
            if paired_outlines
            else np.nan
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PC feature-group relationships with Ki67 labels."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Folder containing per-dataset *_cleaned.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Analysis output folder.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Source-folder grouped CV splits.",
    )
    return parser.parse_args()


def existing(columns: Iterable[str], candidates: Sequence[str]) -> list[str]:
    column_set = set(columns)
    return [column for column in candidates if column in column_set]


def texture_columns(columns: Iterable[str]) -> list[str]:
    tokens = ("GLCM ", "LBP ", "Tamura Coarseness", "Zernike Moment")
    return [
        column
        for column in columns
        if any(token in column for token in tokens)
        and (
            column.endswith("_nuc")
            or column.endswith("_cyto")
            or column.startswith("Whole Cell ")
        )
    ]


def build_feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    columns = list(columns)
    nucleus_geometry = [f"{base}_nuc" for base in GEOMETRY_BASES]
    cyto_geometry = [f"{base}_cyto" for base in GEOMETRY_BASES]
    intensity = [
        *[f"{base}_nuc" for base in INTENSITY_BASES],
        *[f"{base}_cyto" for base in INTENSITY_BASES],
        *[f"Whole Cell {base}" for base in INTENSITY_BASES],
        *DERIVED_INTENSITY_COLUMNS,
    ]
    groups = {
        "Attachment / spreading": existing(
            columns,
            [
                *cyto_geometry,
                *SHAPE_COLUMNS,
                "Nucleus Centroid Offset",
            ],
        ),
        "Nuclear morphology": existing(
            columns,
            [
                *nucleus_geometry,
                "Karyoplasmic Ratio",
                "Nucleus Centroid Offset",
                *NUCLEOLUS_COLUMNS,
            ],
        ),
        "Intensity distribution": existing(columns, intensity),
        "Texture": texture_columns(columns),
        "Halo / rounding": existing(columns, HALO_COLUMNS),
        "Local crowding": existing(columns, LOCAL_CROWDING_COLUMNS),
        "Colony / FOV context": existing(columns, COLONY_CONTEXT_COLUMNS),
        "Mitosis likelihood": existing(columns, MITOSIS_COLUMNS),
        "Debris / culture health": existing(columns, DEBRIS_COLUMNS),
    }
    groups["Texture + Intensity + Halo"] = sorted(
        set(
            groups["Texture"]
            + groups["Intensity distribution"]
            + groups["Halo / rounding"]
        )
    )
    all_pc = sorted({column for values in groups.values() for column in values})
    groups["All PC parameters"] = all_pc
    return groups


def load_dataset(results_dir: Path) -> DatasetBundle:
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []

    for csv_path in sorted(results_dir.glob("*/*_cleaned.csv")):
        source_folder = csv_path.parent.name
        df = pd.read_csv(csv_path)
        has_label = LABEL_COLUMN in df.columns
        labels = (
            pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
            if has_label
            else pd.Series(dtype=np.float64)
        )
        label_non_null = labels.dropna()
        positive_count = int((label_non_null > 0).sum()) if has_label else 0
        negative_count = int((label_non_null <= 0).sum()) if has_label else 0
        mask_coverage = ki67_mask_coverage_by_pc_image(source_folder)
        invalid_mask_images = {
            image
            for image, fraction in mask_coverage.items()
            if fraction > MAX_VALID_KI67_MASK_FRACTION
        }
        outline_quality = outline_pair_quality(source_folder)
        all_null_columns = df.columns[df.isna().all()].tolist()
        inventory_rows.append(
            {
                "source_folder": source_folder,
                "path": str(csv_path),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "images": int(df["Image"].nunique()) if "Image" in df else 0,
                "has_label": bool(has_label),
                "positive_cells": positive_count,
                "negative_cells": negative_count,
                "positive_rate": (
                    float(positive_count / len(label_non_null))
                    if len(label_non_null)
                    else np.nan
                ),
                "label_variation": bool(
                    positive_count > 0 and negative_count > 0
                ),
                "blank_cell_fraction": float(df.isna().mean().mean()),
                "fully_blank_columns": int(len(all_null_columns)),
                "fully_blank_cyto_columns": int(
                    sum(column.endswith("_cyto") for column in all_null_columns)
                ),
                "mask_images": int(len(mask_coverage)),
                "invalid_mask_images": int(len(invalid_mask_images)),
                "median_mask_foreground_fraction": (
                    float(np.median(list(mask_coverage.values())))
                    if mask_coverage
                    else np.nan
                ),
                "max_mask_foreground_fraction": (
                    float(max(mask_coverage.values()))
                    if mask_coverage
                    else np.nan
                ),
                **outline_quality,
                "has_expanded_schema": bool(
                    "Tamura Coarseness_nuc" in df.columns
                    and "Zernike Moment 24_nuc" in df.columns
                    and "Halo Angular Variance" in df.columns
                ),
                "modified_time": pd.Timestamp(
                    csv_path.stat().st_mtime, unit="s"
                ).isoformat(),
            }
        )
        if not has_label:
            continue
        local = df.copy()
        local[LABEL_COLUMN] = (
            pd.to_numeric(local[LABEL_COLUMN], errors="coerce") > 0
        ).astype("Int64")
        local = local.dropna(subset=[LABEL_COLUMN]).copy()
        local[LABEL_COLUMN] = local[LABEL_COLUMN].astype(int)
        local["source_folder"] = source_folder
        if "Image" not in local.columns:
            local["Image"] = local["Cell_ID"].astype(str).str.rsplit(
                "_", n=1
            ).str[0]
        local["image_key"] = (
            local["source_folder"].astype(str)
            + "::"
            + local["Image"].astype(str)
        )
        local["ki67_mask_foreground_fraction"] = local["Image"].map(
            mask_coverage
        )
        local["label_quality_valid"] = (
            local["ki67_mask_foreground_fraction"].isna()
            | (
                local["ki67_mask_foreground_fraction"]
                <= MAX_VALID_KI67_MASK_FRACTION
            )
        )
        local["outline_quality_valid"] = bool(
            pd.isna(outline_quality["identical_outline_pair_fraction"])
            or outline_quality["identical_outline_pair_fraction"] < 0.95
        )
        frames.append(local)

    if not frames:
        raise ValueError(f"No labeled cleaned CSV files found in {results_dir}")
    all_labeled = pd.concat(frames, ignore_index=True, sort=False)
    all_labeled = all_labeled.replace([np.inf, -np.inf], np.nan)
    inventory = pd.DataFrame(inventory_rows)
    valid_labeled = all_labeled[
        all_labeled["label_quality_valid"]
        & all_labeled["outline_quality_valid"]
    ].copy()
    mixed_folders = set(
        valid_labeled.groupby("source_folder")[LABEL_COLUMN]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    primary = valid_labeled[
        valid_labeled["source_folder"].isin(mixed_folders)
    ].copy()
    inventory["analysis_included"] = inventory["source_folder"].isin(
        mixed_folders
    )
    return DatasetBundle(
        all_labeled=all_labeled.reset_index(drop=True),
        primary=primary.reset_index(drop=True),
        inventory=inventory,
    )


def clean_group_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    kept: list[str] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        finite_count = int(values.notna().sum())
        if finite_count < max(20, int(0.05 * len(df))):
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        kept.append(column)
    return kept


def balanced_image_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("image_key")["image_key"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=np.float64)
    return weights / float(np.mean(weights))


def choose_training_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probability,
        sample_weight=sample_weight,
    )
    finite = (
        np.isfinite(thresholds)
        & (thresholds >= 0.0)
        & (thresholds <= 1.0)
    )
    if not finite.any():
        return 0.5
    youden_j = tpr - fpr
    best = float(np.nanmax(youden_j[finite]))
    candidates = np.flatnonzero(
        finite & np.isclose(youden_j, best, rtol=0.0, atol=1e-12)
    )
    return float(
        thresholds[candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]]
    )


def image_auc_table(
    frame: pd.DataFrame, score_column: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_key, image_df in frame.groupby("image_key", sort=False):
        y_true = image_df[LABEL_COLUMN].to_numpy(dtype=np.int64)
        score = pd.to_numeric(
            image_df[score_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(score)
        y_true = y_true[valid]
        score = score[valid]
        if len(np.unique(y_true)) < 2 or len(y_true) < 6:
            continue
        rows.append(
            {
                "image_key": image_key,
                "source_folder": str(image_df["source_folder"].iloc[0]),
                "image": str(image_df["Image"].iloc[0]),
                "cells": int(len(y_true)),
                "positive_rate": float(np.mean(y_true)),
                "auc": float(roc_auc_score(y_true, score)),
            }
        )
    return pd.DataFrame(rows)


def folder_auc_table(
    frame: pd.DataFrame, score_column: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_folder, folder_df in frame.groupby("source_folder", sort=False):
        y_true = folder_df[LABEL_COLUMN].to_numpy(dtype=np.int64)
        score = pd.to_numeric(
            folder_df[score_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(score)
        y_true = y_true[valid]
        score = score[valid]
        if len(np.unique(y_true)) < 2:
            continue
        rows.append(
            {
                "source_folder": source_folder,
                "cells": int(len(y_true)),
                "positive_rate": float(np.mean(y_true)),
                "auc": float(roc_auc_score(y_true, score)),
                "average_precision": float(
                    average_precision_score(y_true, score)
                ),
            }
        )
    return pd.DataFrame(rows)


def image_positive_ratio_table(
    frame: pd.DataFrame,
    score_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    """Aggregate out-of-fold cell predictions into image positive ratios."""
    rows: list[dict[str, object]] = []
    for image_key, image_df in frame.groupby("image_key", sort=False):
        probability = pd.to_numeric(
            image_df[score_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        prediction = pd.to_numeric(
            image_df[prediction_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        label = image_df[LABEL_COLUMN].to_numpy(dtype=np.float64)
        valid = np.isfinite(probability) & np.isfinite(prediction)
        if not valid.any():
            continue
        rows.append(
            {
                "image_key": image_key,
                "source_folder": str(image_df["source_folder"].iloc[0]),
                "image": str(image_df["Image"].iloc[0]),
                "cells": int(valid.sum()),
                "actual_positive_ratio": float(np.mean(label[valid])),
                "predicted_probability_ratio": float(
                    np.mean(probability[valid])
                ),
                "predicted_label_ratio": float(np.mean(prediction[valid])),
            }
        )
    return pd.DataFrame(rows)


def cross_validated_group_scores(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_names = df["source_folder"].astype(str)
    unique_groups = int(group_names.nunique())
    splitter = GroupKFold(n_splits=min(n_splits, unique_groups))
    y = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    score_frame = df[
        ["Cell_ID", "Image", "image_key", "source_folder", LABEL_COLUMN]
    ].copy()
    summary_rows: list[dict[str, object]] = []
    folder_metric_frames: list[pd.DataFrame] = []

    for group_name, candidate_columns in groups.items():
        columns = clean_group_columns(df, candidate_columns)
        if not columns:
            continue
        x_raw = df[columns].apply(pd.to_numeric, errors="coerce")
        oof_probability = np.full(len(df), np.nan, dtype=np.float64)
        oof_prediction = np.full(len(df), -1, dtype=np.int8)
        fold_thresholds: list[float] = []

        for train_index, validation_index in splitter.split(
            x_raw, y, groups=group_names
        ):
            x_train = x_raw.iloc[train_index]
            x_validation = x_raw.iloc[validation_index]
            y_train = y[train_index]
            imputer = SimpleImputer(
                strategy="median", keep_empty_features=True
            )
            scaler = StandardScaler()
            x_train_transformed = scaler.fit_transform(
                imputer.fit_transform(x_train)
            )
            x_validation_transformed = scaler.transform(
                imputer.transform(x_validation)
            )
            model = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=1e-4,
                l1_ratio=0.15,
                max_iter=2500,
                tol=1e-4,
                random_state=RANDOM_STATE,
            )
            train_weights = balanced_image_weights(df.iloc[train_index])
            model.fit(
                x_train_transformed,
                y_train,
                sample_weight=train_weights,
            )
            train_probability = model.predict_proba(
                x_train_transformed
            )[:, 1]
            threshold = choose_training_threshold(
                y_train,
                train_probability,
                train_weights,
            )
            validation_probability = model.predict_proba(
                x_validation_transformed
            )[:, 1]
            oof_probability[validation_index] = validation_probability
            oof_prediction[validation_index] = (
                validation_probability >= threshold
            ).astype(np.int8)
            fold_thresholds.append(threshold)

        safe_name = (
            group_name.lower()
            .replace(" / ", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        score_column = f"{safe_name}_ki67_evidence"
        prediction_column = f"{safe_name}_ki67_prediction"
        score_frame[score_column] = oof_probability
        score_frame[prediction_column] = oof_prediction
        scored = df[
            ["Cell_ID", "Image", "image_key", "source_folder", LABEL_COLUMN]
        ].copy()
        scored[score_column] = oof_probability
        scored[prediction_column] = oof_prediction
        valid = np.isfinite(oof_probability) & (oof_prediction >= 0)
        prevalence = float(np.mean(y[valid]))
        pooled_auc = float(roc_auc_score(y[valid], oof_probability[valid]))
        pooled_ap = float(
            average_precision_score(y[valid], oof_probability[valid])
        )
        pooled_accuracy = float(
            accuracy_score(y[valid], oof_prediction[valid])
        )
        pooled_balanced_accuracy = float(
            balanced_accuracy_score(y[valid], oof_prediction[valid])
        )
        pooled_sensitivity = float(
            recall_score(y[valid], oof_prediction[valid], pos_label=1)
        )
        pooled_specificity = float(
            recall_score(y[valid], oof_prediction[valid], pos_label=0)
        )
        image_metrics = image_auc_table(scored, score_column)
        image_ratios = image_positive_ratio_table(
            scored,
            score_column,
            prediction_column,
        )
        folder_metrics = folder_auc_table(scored, score_column)
        folder_metrics.insert(0, "feature_group", group_name)
        folder_metric_frames.append(folder_metrics)
        actual_ratio = image_ratios[
            "actual_positive_ratio"
        ].to_numpy(dtype=np.float64)
        predicted_ratio = image_ratios[
            "predicted_probability_ratio"
        ].to_numpy(dtype=np.float64)
        predicted_label_ratio = image_ratios[
            "predicted_label_ratio"
        ].to_numpy(dtype=np.float64)
        ratio_error = predicted_ratio - actual_ratio
        summary_rows.append(
            {
                "feature_group": group_name,
                "parameters_used": int(len(columns)),
                "cells": int(valid.sum()),
                "source_folders": int(
                    scored.loc[valid, "source_folder"].nunique()
                ),
                "images_within_image_auc": int(len(image_metrics)),
                "pooled_oof_auc": pooled_auc,
                "pooled_oof_average_precision": pooled_ap,
                "pooled_oof_accuracy": pooled_accuracy,
                "pooled_oof_balanced_accuracy": pooled_balanced_accuracy,
                "pooled_oof_sensitivity": pooled_sensitivity,
                "pooled_oof_specificity": pooled_specificity,
                "image_ratio_images": int(len(image_ratios)),
                "image_ratio_mae": float(np.mean(np.abs(ratio_error))),
                "image_ratio_rmse": float(
                    np.sqrt(np.mean(np.square(ratio_error)))
                ),
                "image_ratio_bias": float(np.mean(ratio_error)),
                "image_ratio_binary_mae": float(
                    np.mean(np.abs(predicted_label_ratio - actual_ratio))
                ),
                "image_ratio_spearman": float(
                    spearmanr(actual_ratio, predicted_ratio).statistic
                ),
                "image_ratio_pearson": float(
                    np.corrcoef(actual_ratio, predicted_ratio)[0, 1]
                ),
                "median_training_threshold": float(
                    np.median(fold_thresholds)
                ),
                "average_precision_lift": (
                    float(pooled_ap / prevalence)
                    if prevalence > 0
                    else np.nan
                ),
                "median_folder_auc": (
                    float(folder_metrics["auc"].median())
                    if len(folder_metrics)
                    else np.nan
                ),
                "mean_folder_auc": (
                    float(folder_metrics["auc"].mean())
                    if len(folder_metrics)
                    else np.nan
                ),
                "median_within_image_auc": (
                    float(image_metrics["auc"].median())
                    if len(image_metrics)
                    else np.nan
                ),
                "mean_within_image_auc": (
                    float(image_metrics["auc"].mean())
                    if len(image_metrics)
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["image_ratio_mae", "image_ratio_rmse"],
        ascending=True,
        na_position="last",
    )
    folder_metrics = (
        pd.concat(folder_metric_frames, ignore_index=True)
        if folder_metric_frames
        else pd.DataFrame()
    )
    return summary, folder_metrics, score_frame


def univariate_image_stratified_associations(
    df: pd.DataFrame, groups: dict[str, list[str]]
) -> pd.DataFrame:
    parameter_to_groups: dict[str, list[str]] = {}
    for group_name, columns in groups.items():
        if group_name in {
            "All PC parameters",
            "Texture + Intensity + Halo",
        }:
            continue
        for column in columns:
            parameter_to_groups.setdefault(column, []).append(group_name)

    rows: list[dict[str, object]] = []
    for parameter, parameter_groups in parameter_to_groups.items():
        auc_values: list[float] = []
        effect_values: list[float] = []
        for _, image_df in df.groupby("image_key", sort=False):
            values = pd.to_numeric(
                image_df[parameter], errors="coerce"
            ).to_numpy(dtype=np.float64)
            labels = image_df[LABEL_COLUMN].to_numpy(dtype=np.int64)
            valid = np.isfinite(values)
            values = values[valid]
            labels = labels[valid]
            positive = values[labels == 1]
            negative = values[labels == 0]
            if (
                len(positive) < 3
                or len(negative) < 3
                or np.unique(values).size < 2
            ):
                continue
            auc_values.append(float(roc_auc_score(labels, values)))
            pooled = np.concatenate([positive, negative])
            q25, q75 = np.percentile(pooled, [25, 75])
            robust_scale = float(q75 - q25)
            if robust_scale > 1e-12:
                effect_values.append(
                    float(
                        (np.median(positive) - np.median(negative))
                        / robust_scale
                    )
                )
        if not auc_values:
            continue
        median_auc = float(np.median(auc_values))
        direction_positive = median_auc >= 0.5
        sign_consistency = float(
            np.mean(
                np.asarray(auc_values) >= 0.5
                if direction_positive
                else np.asarray(auc_values) < 0.5
            )
        )
        rows.append(
            {
                "parameter": parameter,
                "feature_groups": " | ".join(parameter_groups),
                "images_evaluated": int(len(auc_values)),
                "median_image_auc": median_auc,
                "association_effect": float(2.0 * (median_auc - 0.5)),
                "association_strength": float(
                    2.0 * abs(median_auc - 0.5)
                ),
                "direction": (
                    "higher in Ki67+"
                    if direction_positive
                    else "lower in Ki67+"
                ),
                "sign_consistency": sign_consistency,
                "median_robust_effect": (
                    float(np.median(effect_values))
                    if effect_values
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["association_strength", "sign_consistency"],
        ascending=False,
    )


def image_context_associations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows: list[dict[str, object]] = []
    context_columns = sorted(
        {
            column
            for columns in FOV_CONTEXT_GROUPS.values()
            for column in columns
            if column in df.columns
        }
    )
    for image_key, image_df in df.groupby("image_key", sort=False):
        row: dict[str, object] = {
            "image_key": image_key,
            "source_folder": str(image_df["source_folder"].iloc[0]),
            "image": str(image_df["Image"].iloc[0]),
            "cell_count": int(len(image_df)),
            "ki67_positive_ratio": float(image_df[LABEL_COLUMN].mean()),
        }
        for column in context_columns:
            values = pd.to_numeric(image_df[column], errors="coerce")
            row[column] = float(values.median()) if values.notna().any() else np.nan
        image_rows.append(row)
    image_df = pd.DataFrame(image_rows)

    parameter_group = {
        column: group_name
        for group_name, columns in FOV_CONTEXT_GROUPS.items()
        for column in columns
    }
    rows: list[dict[str, object]] = []
    for parameter in context_columns:
        overall = image_df[[parameter, "ki67_positive_ratio"]].dropna()
        overall_rho = (
            float(spearmanr(overall[parameter], overall["ki67_positive_ratio"]).statistic)
            if len(overall) >= 5
            and overall[parameter].nunique() > 1
            and overall["ki67_positive_ratio"].nunique() > 1
            else np.nan
        )
        folder_rhos: list[float] = []
        for _, folder_df in image_df.groupby("source_folder"):
            valid = folder_df[[parameter, "ki67_positive_ratio"]].dropna()
            if (
                len(valid) < 4
                or valid[parameter].nunique() <= 1
                or valid["ki67_positive_ratio"].nunique() <= 1
            ):
                continue
            rho = float(
                spearmanr(
                    valid[parameter], valid["ki67_positive_ratio"]
                ).statistic
            )
            if np.isfinite(rho):
                folder_rhos.append(rho)
        median_folder_rho = (
            float(np.median(folder_rhos)) if folder_rhos else np.nan
        )
        direction_positive = (
            median_folder_rho >= 0
            if np.isfinite(median_folder_rho)
            else overall_rho >= 0
        )
        rows.append(
            {
                "parameter": parameter,
                "feature_group": parameter_group.get(parameter, "Context"),
                "images": int(len(overall)),
                "overall_spearman_rho": overall_rho,
                "folders_evaluated": int(len(folder_rhos)),
                "median_within_folder_spearman_rho": median_folder_rho,
                "sign_consistency": (
                    float(
                        np.mean(
                            np.asarray(folder_rhos) >= 0
                            if direction_positive
                            else np.asarray(folder_rhos) < 0
                        )
                    )
                    if folder_rhos
                    else np.nan
                ),
            }
        )
    associations = pd.DataFrame(rows)
    associations["sort_strength"] = associations[
        "median_within_folder_spearman_rho"
    ].abs()
    associations = associations.sort_values(
        "sort_strength", ascending=False, na_position="last"
    ).drop(columns="sort_strength")
    return associations, image_df


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "text.color": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Aptos",
                "Segoe UI",
                "DejaVu Sans",
                "Arial",
            ],
        },
    )


def add_chart_header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        textwrap.fill(title, 74),
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.91,
        textwrap.fill(subtitle, 112),
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def save_group_performance_chart(
    summary: pd.DataFrame, output_path: Path
) -> None:
    plot_df = summary[
        summary["feature_group"] != "All PC parameters"
    ].copy()
    plot_df = plot_df.sort_values("image_ratio_mae", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.barh(
        plot_df["feature_group"],
        plot_df["image_ratio_mae"] * 100.0,
        color=TOKENS["blue_light"],
        edgecolor=TOKENS["blue"],
    )
    ax.set_xlabel("Image positive-ratio MAE (percentage points; lower is better)")
    ax.set_ylabel("")
    for index, value in enumerate(plot_df["image_ratio_mae"] * 100.0):
        ax.text(
            value + 0.15,
            index,
            f"{value:.1f} pp",
            va="center",
            fontsize=9,
            color=TOKENS["ink"],
        )
    add_chart_header(
        fig,
        ax,
        "Image-level Ki67 positive-ratio prediction error by feature group",
        "Cell-level out-of-fold probabilities are averaged within each image, then compared with the observed positive-cell fraction.",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_image_ratio_scatter(
    score_frame: pd.DataFrame,
    output_path: Path,
    feature_group: str = "Texture + Intensity + Halo",
) -> None:
    safe_name = (
        feature_group.lower()
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    ratio_df = image_positive_ratio_table(
        score_frame,
        f"{safe_name}_ki67_evidence",
        f"{safe_name}_ki67_prediction",
    )
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.scatter(
        ratio_df["actual_positive_ratio"] * 100.0,
        ratio_df["predicted_probability_ratio"] * 100.0,
        s=np.clip(ratio_df["cells"], 12, 90),
        alpha=0.55,
        color=TOKENS["blue"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.plot([0, 100], [0, 100], linestyle="--", color=TOKENS["ink"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observed Ki67-positive ratio (%)")
    ax.set_ylabel("Predicted Ki67-positive ratio (%)")
    add_chart_header(
        fig,
        ax,
        f"Observed vs predicted image Ki67 ratio: {feature_group}",
        "Each point is one image. Point size reflects the number of analyzed cells; the dashed line is perfect agreement.",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top_parameter_chart(
    associations: pd.DataFrame, output_path: Path, top_n: int = 20
) -> None:
    plot_df = associations[
        (associations["images_evaluated"] >= 20)
        & (associations["sign_consistency"] >= 0.55)
    ].head(top_n).copy()
    plot_df = plot_df.sort_values("association_effect")
    colors = [
        TOKENS["blue"] if value >= 0 else TOKENS["orange"]
        for value in plot_df["association_effect"]
    ]
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.barh(
        plot_df["parameter"],
        plot_df["association_effect"],
        color=colors,
        edgecolor=TOKENS["ink"],
        linewidth=0.5,
    )
    ax.axvline(0, color=TOKENS["ink"], linewidth=1)
    ax.set_xlabel("Image-stratified association effect: 2 × (median AUC - 0.5)")
    ax.set_ylabel("")
    max_abs = max(0.05, float(plot_df["association_effect"].abs().max()))
    ax.set_xlim(-max_abs * 1.22, max_abs * 1.22)
    for index, value in enumerate(plot_df["association_effect"]):
        ax.text(
            value + (0.006 if value >= 0 else -0.006),
            index,
            f"{value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
        )
    add_chart_header(
        fig,
        ax,
        "Strongest individual PC parameter associations",
        "Positive values mean higher parameter values in Ki67-positive cells; negative values mean lower values. Rankings use median within-image AUC.",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_context_correlation_chart(
    associations: pd.DataFrame, output_path: Path
) -> None:
    plot_df = associations.dropna(
        subset=["median_within_folder_spearman_rho"]
    ).copy()
    plot_df = plot_df.sort_values("median_within_folder_spearman_rho")
    colors = [
        TOKENS["blue"] if value >= 0 else TOKENS["orange"]
        for value in plot_df["median_within_folder_spearman_rho"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.barh(
        plot_df["parameter"],
        plot_df["median_within_folder_spearman_rho"],
        color=colors,
        edgecolor=TOKENS["ink"],
        linewidth=0.5,
    )
    ax.axvline(0, color=TOKENS["ink"], linewidth=1)
    ax.set_xlabel("Median within-folder Spearman correlation with image Ki67 ratio")
    ax.set_ylabel("")
    ax.set_xlim(-1, 1)
    add_chart_header(
        fig,
        ax,
        "FOV and neighbourhood context versus image Ki67 ratio",
        "Each point of evidence is an image; correlations are calculated within source folders first and then summarized by the median.",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_folder_rate_chart(
    inventory: pd.DataFrame, output_path: Path
) -> None:
    plot_df = inventory[inventory["has_label"]].copy()
    plot_df = plot_df.sort_values("positive_rate")
    colors = [
        TOKENS["orange_light"] if not varied else TOKENS["blue_light"]
        for varied in plot_df["label_variation"]
    ]
    fig_height = max(7.5, 0.31 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.barh(
        plot_df["source_folder"],
        plot_df["positive_rate"],
        color=colors,
        edgecolor=[
            TOKENS["orange"] if not varied else TOKENS["blue"]
            for varied in plot_df["label_variation"]
        ],
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Ki67-positive fraction")
    ax.set_ylabel("")
    for index, value in enumerate(plot_df["positive_rate"]):
        ax.text(
            value + 0.01,
            index,
            f"{value:.1%}",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
        )
    add_chart_header(
        fig,
        ax,
        "Ki67 prevalence varies substantially by source folder",
        "Orange folders contain only one label class and are retained in inventory but excluded from the primary association models.",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_html(
    df: pd.DataFrame,
    columns: Sequence[str],
    rename: dict[str, str],
    formats: dict[str, str] | None = None,
    max_rows: int | None = None,
) -> str:
    display = df.loc[:, [column for column in columns if column in df]].copy()
    if max_rows is not None:
        display = display.head(max_rows)
    formats = formats or {}
    for column, format_string in formats.items():
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: (
                    format_string.format(value)
                    if pd.notna(value)
                    else ""
                )
            )
    display = display.rename(columns=rename)
    return display.to_html(index=False, escape=True, classes="data-table")


def write_missingness_audit(results_dir: Path, output_dir: Path) -> None:
    """Write refreshed dataset- and column-level cleaned CSV missingness audits."""
    summary_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    for csv_path in sorted(results_dir.glob("*/*_cleaned.csv")):
        frame = pd.read_csv(csv_path)
        missing_fraction = frame.isna().mean()
        status_counts = (
            frame["cell_status"].value_counts(dropna=False).to_dict()
            if "cell_status" in frame
            else {}
        )
        summary_rows.append(
            {
                "source_folder": csv_path.parent.name,
                "path": str(csv_path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "blank_cell_fraction": float(frame.isna().mean().mean()),
                "rows_with_any_blank_fraction": float(
                    frame.isna().any(axis=1).mean()
                ),
                "fully_blank_columns": int((missing_fraction == 1.0).sum()),
                "columns_at_least_half_blank": int(
                    (missing_fraction >= 0.5).sum()
                ),
                "cell_status_counts": json.dumps(
                    status_counts, ensure_ascii=False, sort_keys=True
                ),
            }
        )
        for column, fraction in missing_fraction.items():
            if column.endswith("_nuc"):
                feature_scope = "nucleus"
            elif column.endswith("_cyto"):
                feature_scope = "cytoplasm"
            else:
                feature_scope = "cell_or_metadata"
            column_rows.append(
                {
                    "source_folder": csv_path.parent.name,
                    "column": column,
                    "feature_scope": feature_scope,
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_fraction": float(fraction),
                    "fully_blank": bool(fraction == 1.0),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cleaned_csv_missingness_summary.csv",
        index=False,
    )
    pd.DataFrame(column_rows).to_csv(
        output_dir / "cleaned_csv_missingness_columns.csv",
        index=False,
    )


def generate_report(
    output_dir: Path,
    bundle: DatasetBundle,
    group_summary: pd.DataFrame,
    parameter_assoc: pd.DataFrame,
    context_assoc: pd.DataFrame,
    chart_paths: dict[str, Path],
) -> None:
    primary = bundle.primary
    inventory = bundle.inventory
    total_cells = int(len(primary))
    positive_cells = int(primary[LABEL_COLUMN].sum())
    positive_rate = float(primary[LABEL_COLUMN].mean())
    mixed_folders = int(primary["source_folder"].nunique())
    images = int(primary["image_key"].nunique())

    biological = group_summary[
        group_summary["feature_group"] != "All PC parameters"
    ].copy()
    best_group = biological.sort_values(
        "image_ratio_mae", ascending=True
    ).iloc[0]
    all_pc_baseline = group_summary[
        group_summary["feature_group"] == "All PC parameters"
    ].iloc[0]
    three_group_combo = group_summary[
        group_summary["feature_group"] == "Texture + Intensity + Halo"
    ]
    three_group_combo = (
        three_group_combo.iloc[0] if len(three_group_combo) else None
    )
    three_group_gap = (
        float(
            three_group_combo["image_ratio_mae"]
            - all_pc_baseline["image_ratio_mae"]
        )
        if three_group_combo is not None
        else np.nan
    )
    best_parameter = parameter_assoc[
        parameter_assoc["images_evaluated"] >= 20
    ].iloc[0]
    context_valid = context_assoc.dropna(
        subset=["median_within_folder_spearman_rho"]
    )
    strongest_context = (
        context_valid.iloc[
            context_valid[
                "median_within_folder_spearman_rho"
            ].abs().argmax()
        ]
        if len(context_valid)
        else None
    )
    context_images = (
        int(context_valid["images"].max()) if len(context_valid) else 0
    )
    all_positive_folders = inventory.loc[
        inventory["has_label"] & ~inventory["label_variation"],
        "source_folder",
    ].tolist()
    invalid_mask_folders = inventory.loc[
        inventory["invalid_mask_images"] > 0,
        "source_folder",
    ].tolist()
    invalid_cyto_folders = inventory.loc[
        inventory["identical_outline_pair_fraction"] >= 0.95,
        "source_folder",
    ].tolist()

    group_table = dataframe_to_html(
        group_summary.sort_values(
            "image_ratio_mae", ascending=True
        ),
        [
            "feature_group",
            "parameters_used",
            "image_ratio_mae",
            "image_ratio_rmse",
            "image_ratio_bias",
            "image_ratio_spearman",
            "median_within_image_auc",
        ],
        {
            "feature_group": "Feature group",
            "parameters_used": "Parameters",
            "image_ratio_mae": "Image ratio MAE",
            "image_ratio_rmse": "Image ratio RMSE",
            "image_ratio_bias": "Mean bias",
            "image_ratio_spearman": "Ratio Spearman rho",
            "median_within_image_auc": "Cell-level AUROC",
        },
        {
            "image_ratio_mae": "{:.1%}",
            "image_ratio_rmse": "{:.1%}",
            "image_ratio_bias": "{:+.1%}",
            "image_ratio_spearman": "{:+.3f}",
            "median_within_image_auc": "{:.3f}",
        },
    )
    parameter_table = dataframe_to_html(
        parameter_assoc[
            (parameter_assoc["images_evaluated"] >= 20)
            & (parameter_assoc["sign_consistency"] >= 0.55)
        ],
        [
            "parameter",
            "feature_groups",
            "direction",
            "association_effect",
            "sign_consistency",
            "images_evaluated",
        ],
        {
            "parameter": "Parameter",
            "feature_groups": "Feature group",
            "direction": "Direction",
            "association_effect": "Effect",
            "sign_consistency": "Sign consistency",
            "images_evaluated": "Images",
        },
        {
            "association_effect": "{:+.3f}",
            "sign_consistency": "{:.1%}",
        },
        max_rows=15,
    )
    context_table = dataframe_to_html(
        context_assoc,
        [
            "parameter",
            "feature_group",
            "median_within_folder_spearman_rho",
            "sign_consistency",
            "folders_evaluated",
            "images",
        ],
        {
            "parameter": "Context parameter",
            "feature_group": "Feature group",
            "median_within_folder_spearman_rho": "Median within-folder rho",
            "sign_consistency": "Sign consistency",
            "folders_evaluated": "Folders",
            "images": "Images",
        },
        {
            "median_within_folder_spearman_rho": "{:+.3f}",
            "sign_consistency": "{:.1%}",
        },
        max_rows=12,
    )

    strongest_context_text = (
        f"<strong>{html.escape(str(strongest_context['parameter']))}</strong> "
        f"是影像層級關聯最強的 context 參數 "
        f"(<code>rho={strongest_context['median_within_folder_spearman_rho']:+.3f}</code>)."
        if strongest_context is not None
        else "沒有 context 參數具備足夠的資料夾內變異以估計穩定關聯。"
    )
    report_html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase-contrast 特徵與 Ki67 關聯分析</title>
  <style>
    :root {{
      --surface: #FCFCFD; --panel: #FFFFFF; --ink: #1F2430;
      --muted: #6F768A; --grid: #E6E8F0; --blue: #5477C4;
      --blue-light: #EAF1FE; --orange-light: #FFEDDE;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: "Segoe UI", "Noto Sans TC", sans-serif; margin: 0; background: var(--surface); color: var(--ink); }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 42px 24px 72px; }}
    header, section {{ margin-bottom: 38px; }}
    h1 {{ font-size: 34px; margin: 0 0 10px; line-height: 1.2; }}
    h2 {{ font-size: 23px; margin: 0 0 14px; line-height: 1.25; }}
    h3 {{ font-size: 18px; margin: 24px 0 10px; }}
    p, li {{ line-height: 1.72; }}
    .meta {{ color: var(--muted); margin: 0; }}
    .summary {{ background: var(--blue-light); border-left: 5px solid var(--blue); padding: 18px 22px; border-radius: 10px; }}
    .warning {{ background: var(--orange-light); padding: 16px 20px; border-radius: 10px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .metric {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 23px; margin-bottom: 3px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    figure {{ margin: 22px 0 28px; background: var(--panel); border: 1px solid var(--grid); border-radius: 12px; padding: 14px; }}
    figure img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ color: var(--muted); font-size: 14px; line-height: 1.55; padding: 8px 4px 2px; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: var(--panel); }}
    .data-table th, .data-table td {{ padding: 9px 10px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top; }}
    .data-table th {{ background: #F4F5F7; position: sticky; top: 0; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--grid); border-radius: 10px; }}
    code {{ background: #F4F5F7; padding: 2px 5px; border-radius: 4px; }}
    pre {{ overflow-x: auto; background: #F4F5F7; padding: 14px; border-radius: 10px; line-height: 1.5; }}
    @media (max-width: 760px) {{
      .metrics {{ grid-template-columns: repeat(2, 1fr); }}
      main {{ padding: 28px 15px 52px; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
<main data-report-audience="technical">
  <header data-contract-section="title">
    <h1>Phase-contrast 特徵與 Ki67 關聯分析</h1>
    <p class="meta">分析日期：2026-06-07｜主要目標：預測每張影像的 Ki67 陽性細胞比例</p>
  </header>

  <section data-contract-section="technical-summary">
    <h2>技術摘要</h2>
    <div class="summary">
      <p><strong>這份報告現在以「每張影像的 Ki67 陽性比例」為主要終點，不再把 cell-level Accuracy 當成主要答案。</strong></p>
      <p>影像陽性率誤差最低的特徵組合是
      <strong>{html.escape(str(best_group['feature_group']))}</strong>，out-of-fold MAE 為
      <code>{best_group['image_ratio_mae']:.1%}</code>（平均相差 {best_group['image_ratio_mae'] * 100:.1f} 個百分點），
      RMSE 為 <code>{best_group['image_ratio_rmse']:.1%}</code>。最強單一參數關聯是
      <strong>{html.escape(str(best_parameter['parameter']))}</strong>，方向為
      <strong>{html.escape(str(best_parameter['direction']))}</strong>。
      全部 {int(all_pc_baseline['parameters_used'])} 個 PC 參數模型的影像陽性率 MAE 為
      <code>{all_pc_baseline['image_ratio_mae']:.1%}</code>。</p>
      {(
        f"<p><strong>Texture + Intensity + Halo</strong> 合計使用 "
        f"{int(three_group_combo['parameters_used'])} 個參數，影像陽性率 MAE 為 "
        f"<code>{three_group_combo['image_ratio_mae']:.1%}</code>，"
        f"RMSE 為 <code>{three_group_combo['image_ratio_rmse']:.1%}</code>，"
        f"與全部參數模型的 MAE 差為 "
        f"<code>{three_group_gap * 100:+.1f}</code> 個百分點。</p>"
        if three_group_combo is not None
        else ""
      )}
      <p>{strongest_context_text}</p>
    </div>
    <div class="metrics">
      <div class="metric"><strong>{total_cells:,}</strong><span>主要分析細胞</span></div>
      <div class="metric"><strong>{images:,}</strong><span>影像 FOV（context 有效 {context_images:,}）</span></div>
      <div class="metric"><strong>{mixed_folders}</strong><span>含正負標籤來源資料夾</span></div>
      <div class="metric"><strong>{positive_rate:.1%}</strong><span>主要資料 Ki67 陽性率</span></div>
    </div>
  </section>

  <section data-contract-section="key-findings">
    <h2>哪些特徵群組較能估計影像 Ki67 陽性比例</h2>
    <p>模型先對未看過的來源資料夾中的每顆細胞產生 out-of-fold 陽性機率，再將同一張影像內的機率平均，得到預測陽性率。MAE 代表預測值與實際陽性率平均相差多少；例如 MAE 12% 就是平均相差 12 個百分點，越低越好。</p>
    <figure>
      <img src="{chart_paths['group'].relative_to(output_dir).as_posix()}" alt="Feature group performance">
      <figcaption>這裡比較的是影像陽性率誤差，不是單顆細胞分類 Accuracy。</figcaption>
    </figure>
    <div class="table-wrap">{group_table}</div>
    <figure>
      <img src="{chart_paths['image_ratio'].relative_to(output_dir).as_posix()}" alt="Observed and predicted Ki67 positive ratios">
      <figcaption>每個點是一張影像；越接近對角線，代表預測陽性比例越接近實際比例。</figcaption>
    </figure>

    <h3>最具方向一致性的單一特徵參數</h3>
    <p>單參數分析先在每張影像內計算 AUC，再取跨影像中位數。這能降低細胞數很多的影像支配結果，也能避免把照明、倍率或實驗批次差異誤認為 Ki67 生物訊號。</p>
    <figure>
      <img src="{chart_paths['parameter'].relative_to(output_dir).as_posix()}" alt="Top parameter associations">
      <figcaption>Effect = 2 × (median image AUC − 0.5)，範圍約為 −1 至 +1；正值表示 Ki67+ 較高，負值表示 Ki67+ 較低。</figcaption>
    </figure>
    <div class="table-wrap">{parameter_table}</div>

    <h3>密集度、群聚與 debris 應在影像層級解讀</h3>
    <p>Confluency、population CV、cluster 與多數 debris 參數在同一張影像中的所有細胞會重複，因此不能被當成獨立的單細胞證據。這些參數改以每張影像的 Ki67 陽性比例作為結果，並在每個來源資料夾內先計算相關性。</p>
    <figure>
      <img src="{chart_paths['context'].relative_to(output_dir).as_posix()}" alt="Context correlations">
      <figcaption>正相關表示該 context 值較高的 FOV 通常有較高 Ki67 ratio；負相關相反。不同細胞株、passage 或密度條件可能改變方向。</figcaption>
    </figure>
    <div class="table-wrap">{context_table}</div>
  </section>

  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>資料範圍與指標定義</h2>
    <p>來源為 <code>data/output/results/*/*_cleaned.csv</code>。共有
    {int(inventory['has_label'].sum())} 個有 Ki67 標籤的資料夾，其中 {mixed_folders} 個同時包含陽性與陰性細胞並進入主要關聯分析。
    主要資料包含 {positive_cells:,} 顆 Ki67 陽性細胞與 {total_cells-positive_cells:,} 顆陰性細胞。</p>
    <ul>
      <li><strong>Ki67 label：</strong><code>ki67_positive</code>，由 nucleus ROI 與 Ki67 mask 的重疊規則產生。</li>
      <li><strong>Feature-group score：</strong>只使用該群組 PC 參數產生的 out-of-fold Ki67 陽性機率傾向。</li>
      <li><strong>Within-image AUROC：</strong>在同一影像內，參數或 score 將陽性細胞排在陰性細胞前面的能力。</li>
      <li><strong>Image Ki67 ratio：</strong>單張影像中 <code>ki67_positive=1</code> 的細胞比例。</li>
      <li><strong>Image ratio MAE：</strong>預測陽性率與實際陽性率的平均絕對差，報告中以百分點解讀。</li>
      <li><strong>Cell-level Accuracy：</strong>仍輸出在 CSV 供診斷，但不是這個研究問題的主要評估指標。</li>
    </ul>
    <div class="warning">
      <strong>資料品質提醒：</strong>
      mask 白色面積超過 10% 而排除的資料夾：
      {html.escape(', '.join(invalid_mask_folders)) if invalid_mask_folders else '無'}。
      只有單一標籤的資料夾：
      {html.escape(', '.join(all_positive_folders)) if all_positive_folders else '無'}。
      核與細胞外框幾乎完全相同、因此 cyto／halo／核質比不可用的資料夾：
      {html.escape(', '.join(invalid_cyto_folders)) if invalid_cyto_folders else '無'}。
    </div>
    <figure>
      <img src="{chart_paths['folder'].relative_to(output_dir).as_posix()}" alt="Folder positive rates">
      <figcaption>不同來源資料夾的陽性率差異很大，因此隨機切細胞會高估模型表現；本分析使用 source-folder grouped CV。</figcaption>
    </figure>
  </section>

  <section data-contract-section="methodology">
    <h2>分析方法</h2>
    <ol>
      <li>只使用目前 Python backend 產生的 phase-contrast 特徵參數；排除標籤、ID、影像名稱與螢光 ring intensity。</li>
      <li>將參數分成 attachment/spreading、nuclear morphology、intensity、texture、halo、crowding、colony context、mitosis 與 debris。</li>
      <li>特徵群組模型使用 median imputation、standardization 與 elastic-net logistic SGD classifier。</li>
      <li>採 5-fold source-folder grouped CV，並以每張影像反比細胞數作 sample weighting，避免大 FOV 支配模型。</li>
      <li>主要影像預測值是同一張影像內所有 OOF cell probability 的平均；驗證影像不參與模型訓練。</li>
      <li>單參數方向由每張混合標籤影像的 AUC 中位數決定；FOV context 則用 image Ki67 ratio 的 within-folder Spearman correlation。</li>
    </ol>
    <pre>association_effect = 2 × (median_within_image_AUC − 0.5)
positive effect  → parameter tends to be higher in Ki67-positive cells
negative effect  → parameter tends to be lower in Ki67-positive cells</pre>
  </section>

  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>限制、不確定性與穩健性</h2>
    <ul>
      <li><strong>相關不是因果：</strong>PC 形態與 Ki67 可能共同受到 passage、細胞株、密度、焦距與培養條件影響。</li>
      <li><strong>Ki67 標籤不是人工 ground truth：</strong>標籤來自二值化與重疊閾值，分割或配對錯誤會傳遞到分析結果。</li>
      <li><strong>Attachment 尚無直接標註：</strong>Attachment/spreading 群組對 Ki67 的預測力不能直接解讀為貼附 good/bad。</li>
      <li><strong>多重比較：</strong>單參數排名用於探索與候選縮減，不能只憑排名宣稱生物機制。</li>
      <li><strong>核仁與 debris：</strong>兩者受影像品質與閾值影響較大，應搭配人工抽查。</li>
    </ul>
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>建議下一步</h2>
    <ol>
      <li>以表現穩定的群組建立 out-of-fold <code>Ki67 Evidence Score</code>，再和全參數 baseline 比較。</li>
      <li>人工標註 attachment good / poor / mitotic-or-rounding，另訓練真正的 Attachment Probability，避免與 Ki67 evidence 混淆。</li>
      <li>把最終模型驗證固定為 leave-source-folder-out，並保留完全未參與選參數的獨立實驗批次。</li>
      <li>針對前 10–20 個參數做 segmentation、焦距與倍率敏感度檢查後，再決定是否縮減特徵。</li>
    </ol>
  </section>

  <section data-contract-section="further-questions">
    <h2>後續需要回答的問題</h2>
    <ul>
      <li>關聯方向在 B4、B8、不同 passage 與不同日期是否一致？</li>
      <li>排除 mitotic cells 後，attachment/spreading 與 Ki67 是否仍有關聯？</li>
      <li>使用人工核對的 Ki67 label 後，texture 與 nucleolus 排名是否改變？</li>
    </ul>
  </section>
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")


def write_source_notes(
    output_dir: Path,
    feature_groups: dict[str, list[str]],
    chart_paths: dict[str, Path],
    bundle: DatasetBundle,
) -> None:
    notes = {
        "analysis_question": "Which PC-derived feature groups and parameters are associated with Ki67-positive cells?",
        "source_pattern": str(DEFAULT_RESULTS_DIR / "*" / "*_cleaned.csv"),
        "primary_population": {
            "rows": int(len(bundle.primary)),
            "source_folders": int(bundle.primary["source_folder"].nunique()),
            "images": int(bundle.primary["image_key"].nunique()),
            "rule": (
                "Labeled mixed-class source folders after excluding images "
                "with Ki67 mask coverage above 10% and folders whose paired "
                "nucleus/cell outlines are at least 95% identical."
            ),
        },
        "excluded_from_primary": bundle.inventory.loc[
            bundle.inventory["has_label"]
            & ~bundle.inventory["analysis_included"],
            "source_folder",
        ].tolist(),
        "feature_groups": feature_groups,
        "chart_map": [
            {
                "section": "Feature-group discrimination",
                "question": "Which biological feature groups best estimate image Ki67 positive ratio?",
                "family": "Comparison & Ranking",
                "type": "Ranked horizontal bars",
                "metric": "Image positive-ratio out-of-fold MAE",
                "artifact": str(chart_paths["group"]),
            },
            {
                "section": "Image positive-ratio validation",
                "question": "How closely do predicted image ratios match observed ratios?",
                "family": "Correlation",
                "type": "Scatter plot with identity line",
                "metric": "Observed and predicted image Ki67-positive ratio",
                "artifact": str(chart_paths["image_ratio"]),
            },
            {
                "section": "Individual parameter associations",
                "question": "Which individual PC parameters have the strongest directionally consistent association?",
                "family": "Comparison & Ranking",
                "type": "Diverging horizontal bars",
                "metric": "2 × (median image AUC - 0.5)",
                "artifact": str(chart_paths["parameter"]),
            },
            {
                "section": "FOV context",
                "question": "How do image-level context parameters relate to image Ki67 ratio?",
                "family": "Comparison & Ranking",
                "type": "Diverging horizontal bars",
                "metric": "Median within-folder Spearman rho",
                "artifact": str(chart_paths["context"]),
            },
            {
                "section": "Data quality",
                "question": "How much does Ki67 prevalence vary by source folder?",
                "family": "Comparison & Ranking",
                "type": "Ranked horizontal bars",
                "metric": "Ki67-positive fraction",
                "artifact": str(chart_paths["folder"]),
            },
        ],
        "caveats": [
            "Association is not causation.",
            "Ki67 labels are derived from segmentation and mask overlap.",
            "Images with Ki67 mask foreground coverage above 10% are excluded.",
            "Folders with at least 95% identical nucleus/cell outlines are excluded.",
            "Attachment/spreading parameters are not attachment ground truth.",
            "FOV-level parameters are analyzed at image grain.",
            "Fluorescence-derived ring intensity columns are excluded.",
        ],
    }
    (output_dir / "source_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_three_group_comparison(
    output_dir: Path,
    group_summary: pd.DataFrame,
    folder_metrics: pd.DataFrame,
    score_frame: pd.DataFrame,
) -> None:
    combo_name = "Texture + Intensity + Halo"
    baseline_name = "All PC parameters"
    selected_names = [
        baseline_name,
        combo_name,
        "Texture",
        "Intensity distribution",
        "Halo / rounding",
    ]
    comparison = group_summary[
        group_summary["feature_group"].isin(selected_names)
    ].copy()
    baseline = comparison[
        comparison["feature_group"] == baseline_name
    ].iloc[0]
    for metric in [
        "median_within_image_auc",
        "median_folder_auc",
        "pooled_oof_auc",
        "pooled_oof_accuracy",
        "pooled_oof_balanced_accuracy",
        "image_ratio_mae",
        "image_ratio_rmse",
        "image_ratio_spearman",
    ]:
        comparison[f"{metric}_gap_vs_all_pc"] = (
            comparison[metric] - float(baseline[metric])
        )
    comparison.to_csv(
        output_dir / "three_group_vs_all_pc_comparison.csv",
        index=False,
    )

    all_image = image_auc_table(
        score_frame,
        "all_pc_parameters_ki67_evidence",
    ).rename(columns={"auc": "all_pc_auc"})
    combo_image = image_auc_table(
        score_frame,
        "texture_+_intensity_+_halo_ki67_evidence",
    ).rename(columns={"auc": "combo_auc"})
    image_comparison = all_image[
        ["image_key", "source_folder", "image", "all_pc_auc"]
    ].merge(
        combo_image[["image_key", "combo_auc"]],
        on="image_key",
        how="inner",
    )
    image_comparison["difference"] = (
        image_comparison["combo_auc"] - image_comparison["all_pc_auc"]
    )
    image_comparison.to_csv(
        output_dir / "three_group_vs_all_pc_image_metrics.csv",
        index=False,
    )
    combo_ratio = image_positive_ratio_table(
        score_frame,
        "texture_+_intensity_+_halo_ki67_evidence",
        "texture_+_intensity_+_halo_ki67_prediction",
    )
    combo_ratio.to_csv(
        output_dir / "texture_intensity_halo_image_ratio_predictions.csv",
        index=False,
    )

    all_folder = folder_metrics[
        folder_metrics["feature_group"] == baseline_name
    ][["source_folder", "auc"]].rename(columns={"auc": "all_pc_auc"})
    combo_folder = folder_metrics[
        folder_metrics["feature_group"] == combo_name
    ][["source_folder", "auc"]].rename(columns={"auc": "combo_auc"})
    folder_comparison = all_folder.merge(
        combo_folder,
        on="source_folder",
        how="inner",
    )
    folder_comparison["difference"] = (
        folder_comparison["combo_auc"] - folder_comparison["all_pc_auc"]
    )
    folder_comparison.to_csv(
        output_dir / "three_group_vs_all_pc_folder_metrics.csv",
        index=False,
    )


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(results_dir)
    feature_groups = build_feature_groups(bundle.primary.columns)
    group_summary, folder_metrics, score_frame = cross_validated_group_scores(
        bundle.primary,
        feature_groups,
        n_splits=max(2, int(args.cv_splits)),
    )
    parameter_assoc = univariate_image_stratified_associations(
        bundle.primary, feature_groups
    )
    context_assoc, image_context = image_context_associations(bundle.primary)

    bundle.inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    write_missingness_audit(results_dir, output_dir)
    group_summary.to_csv(
        output_dir / "feature_group_summary.csv", index=False
    )
    folder_metrics.to_csv(
        output_dir / "feature_group_folder_metrics.csv", index=False
    )
    score_frame.to_csv(
        output_dir / "out_of_fold_feature_group_scores.csv", index=False
    )
    parameter_assoc.to_csv(
        output_dir / "feature_parameter_associations.csv", index=False
    )
    context_assoc.to_csv(
        output_dir / "image_context_associations.csv", index=False
    )
    image_context.to_csv(
        output_dir / "image_context_dataset.csv", index=False
    )
    write_three_group_comparison(
        output_dir,
        group_summary,
        folder_metrics,
        score_frame,
    )

    use_chart_theme()
    chart_paths = {
        "group": charts_dir / "feature_group_cv_performance.png",
        "image_ratio": charts_dir / "image_positive_ratio_prediction.png",
        "parameter": charts_dir / "top_parameter_associations.png",
        "context": charts_dir / "image_context_correlations.png",
        "folder": charts_dir / "folder_positive_rates.png",
    }
    save_group_performance_chart(group_summary, chart_paths["group"])
    save_image_ratio_scatter(score_frame, chart_paths["image_ratio"])
    save_top_parameter_chart(parameter_assoc, chart_paths["parameter"])
    save_context_correlation_chart(context_assoc, chart_paths["context"])
    save_folder_rate_chart(bundle.inventory, chart_paths["folder"])
    generate_report(
        output_dir,
        bundle,
        group_summary,
        parameter_assoc,
        context_assoc,
        chart_paths,
    )
    write_source_notes(
        output_dir, feature_groups, chart_paths, bundle
    )

    print(f"[INFO] Primary cells: {len(bundle.primary):,}")
    print(
        f"[INFO] Primary source folders: "
        f"{bundle.primary['source_folder'].nunique()}"
    )
    print(f"[INFO] Report: {output_dir / 'report.html'}")
    print(group_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
