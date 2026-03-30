#!/usr/bin/env python
"""Ki67 single-cell ordinal classification benchmark.

Key properties:
- Reads per-cell tabular features from data/output/results/*/*_cleaned.csv
- Splits by image groups (never cell-level random split)
- Splits each dataset separately, then merges into global train/val/test
- Compares multiple tabular baselines
- Runs both:
  1) without passage feature
  2) with passage feature
- Supports external label merge on (dataset, Image, Cell_ID)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PASSAGE_PATTERN = re.compile(r"[Pp](\d+)")
KEY_COLUMNS = ["dataset", "Image", "Cell_ID"]
REQUIRED_LABEL_COLUMNS = ["dataset", "Image", "Cell_ID", "label"]
ORDINAL_LABELS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ki67 single-cell ordinal benchmark")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/output/results"),
        help="Root directory containing dataset folders with *_cleaned.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: analysis/ki67_cell_ordinal_benchmark/run_<timestamp>",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional external labels CSV with columns: dataset,Image,Cell_ID,label",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column in cleaned CSV to use as label when --labels-csv is not provided",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Allow label set to be a subset of {0,1,2}. By default all three classes are required.",
    )
    parser.add_argument(
        "--dataset-include",
        nargs="*",
        default=None,
        help="Optional whitelist of dataset folder names",
    )
    parser.add_argument(
        "--dataset-exclude",
        nargs="*",
        default=None,
        help="Optional blacklist of dataset folder names",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help="Optional explicit feature column list. If omitted, all numeric features are used.",
    )
    parser.add_argument(
        "--export-cell-template",
        type=Path,
        default=None,
        help="Optional path to export unique (dataset,Image,Cell_ID,label) template for manual labeling.",
    )
    return parser.parse_args()


def canonicalize_id_series(series: pd.Series) -> pd.Series:
    def normalize(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text.endswith(".0"):
            try:
                num = float(text)
                if num.is_integer():
                    return str(int(num))
            except ValueError:
                return text
        return text

    return series.map(normalize)


def infer_passage_from_dataset(dataset_name: str) -> float:
    matches = PASSAGE_PATTERN.findall(dataset_name)
    if not matches:
        return math.nan
    return float(matches[-1])


def stable_seed(base_seed: int, dataset_name: str) -> int:
    digest = hashlib.md5(dataset_name.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return (base_seed + offset) % (2**32)


def ensure_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def list_cleaned_csv_files(results_root: Path) -> list[Path]:
    files = sorted(results_root.glob("*/*_cleaned.csv"))
    if not files:
        raise FileNotFoundError(f"No *_cleaned.csv found under: {results_root}")
    return files


def load_cleaned_table(path: Path) -> pd.DataFrame:
    dataset = path.parent.name
    df = pd.read_csv(path)
    ensure_columns(df, ["Image", "Cell_ID"], context=str(path))

    df = df.copy()
    df["dataset"] = dataset
    df["Image"] = canonicalize_id_series(df["Image"])
    df["Cell_ID"] = canonicalize_id_series(df["Cell_ID"])

    passage_fallback = infer_passage_from_dataset(dataset)
    if "P" in df.columns:
        p_numeric = pd.to_numeric(df["P"], errors="coerce")
        df["P"] = p_numeric.fillna(passage_fallback)
    else:
        df["P"] = passage_fallback
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    return df


def load_all_features(
    results_root: Path,
    dataset_include: set[str] | None = None,
    dataset_exclude: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = list_cleaned_csv_files(results_root)

    records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for path in files:
        dataset = path.parent.name
        if dataset_include is not None and dataset not in dataset_include:
            continue
        if dataset_exclude is not None and dataset in dataset_exclude:
            continue
        frame = load_cleaned_table(path)
        frames.append(frame)
        records.append(
            {
                "dataset": dataset,
                "path": str(path),
                "n_cells": int(len(frame)),
                "n_images": int(frame["Image"].nunique()),
            }
        )

    if not frames:
        raise ValueError("No dataset selected after include/exclude filtering.")

    full_df = pd.concat(frames, ignore_index=True)
    full_df["dataset"] = canonicalize_id_series(full_df["dataset"])
    manifest_df = pd.DataFrame(records).sort_values("dataset").reset_index(drop=True)
    return full_df, manifest_df


def export_cell_template(df: pd.DataFrame, output_path: Path) -> None:
    template = df[KEY_COLUMNS].drop_duplicates().sort_values(KEY_COLUMNS).reset_index(drop=True)
    template["label"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)


def merge_labels(feature_df: pd.DataFrame, labels_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_csv(labels_csv)
    ensure_columns(labels, REQUIRED_LABEL_COLUMNS, context=f"labels_csv={labels_csv}")

    labels = labels.copy()
    for col in KEY_COLUMNS:
        labels[col] = canonicalize_id_series(labels[col])

    duplicate_mask = labels.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        dup_preview = labels.loc[duplicate_mask, KEY_COLUMNS].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate keys in labels CSV (showing up to 10): {dup_preview}")

    merged = feature_df.merge(
        labels[REQUIRED_LABEL_COLUMNS],
        how="left",
        on=KEY_COLUMNS,
        validate="m:1",
    )
    matched = int(merged["label"].notna().sum())
    total = int(len(merged))
    coverage = matched / total if total else 0.0

    info = {
        "label_source": str(labels_csv),
        "matched_rows": matched,
        "total_rows": total,
        "coverage": coverage,
    }
    return merged, info


def attach_labels(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.labels_csv is not None:
        labeled_df, info = merge_labels(df, args.labels_csv)
    else:
        if args.label_column not in df.columns:
            raise ValueError(
                f"Label column '{args.label_column}' not found. "
                "Provide --labels-csv or a valid --label-column."
            )
        labeled_df = df.copy()
        labeled_df["label"] = labeled_df[args.label_column]
        info = {"label_source": f"column:{args.label_column}"}

    before_drop = len(labeled_df)
    labeled_df["label"] = pd.to_numeric(labeled_df["label"], errors="coerce")
    labeled_df = labeled_df[labeled_df["label"].notna()].copy()
    labeled_df["label"] = labeled_df["label"].astype(int)
    dropped = before_drop - len(labeled_df)
    info["dropped_missing_label"] = int(dropped)
    info["rows_after_label_drop"] = int(len(labeled_df))

    unique_labels = sorted(labeled_df["label"].unique().tolist())
    invalid = [x for x in unique_labels if x not in ORDINAL_LABELS]
    if invalid:
        raise ValueError(f"Labels must be in {ORDINAL_LABELS}, found invalid labels: {invalid}")

    if not args.allow_missing_classes:
        missing = [x for x in ORDINAL_LABELS if x not in unique_labels]
        if missing:
            raise ValueError(
                f"Expected all ordinal classes {ORDINAL_LABELS}, missing: {missing}. "
                "Use --allow-missing-classes to bypass this check."
            )

    info["unique_labels"] = unique_labels
    return labeled_df, info


def compute_split_counts(n_images: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if n_images <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if n_images == 1:
        return {"train": 1, "val": 0, "test": 0}
    if n_images == 2:
        return {"train": 1, "val": 0, "test": 1}

    raw = {
        "train": n_images * train_ratio,
        "val": n_images * val_ratio,
        "test": n_images * test_ratio,
    }
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = n_images - sum(counts.values())
    order = sorted(raw.keys(), key=lambda k: (raw[k] - counts[k]), reverse=True)
    for i in range(remaining):
        counts[order[i % len(order)]] += 1

    for split in ("val", "test"):
        if counts[split] == 0:
            donor = max((k for k in counts if counts[k] > 1), key=lambda k: counts[k], default=None)
            if donor is not None:
                counts[donor] -= 1
                counts[split] += 1

    if counts["train"] == 0:
        donor = "val" if counts["val"] > counts["test"] else "test"
        if counts[donor] > 1:
            counts[donor] -= 1
            counts["train"] += 1

    if sum(counts.values()) != n_images:
        raise RuntimeError("Split count computation failed to preserve total image count.")
    return counts


def build_grouped_split(
    labeled_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments: list[dict[str, Any]] = []

    for dataset, sub in labeled_df.groupby("dataset", sort=True):
        images = sub["Image"].dropna().unique().tolist()
        images = sorted(images)
        rng = np.random.default_rng(stable_seed(random_state, dataset))
        rng.shuffle(images)

        counts = compute_split_counts(len(images), train_ratio, val_ratio, test_ratio)
        train_end = counts["train"]
        val_end = train_end + counts["val"]

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for image_id in train_images:
            assignments.append({"dataset": dataset, "Image": image_id, "split": "train"})
        for image_id in val_images:
            assignments.append({"dataset": dataset, "Image": image_id, "split": "val"})
        for image_id in test_images:
            assignments.append({"dataset": dataset, "Image": image_id, "split": "test"})

    split_map = pd.DataFrame(assignments)
    if split_map.empty:
        raise ValueError("No image split assignments were produced.")

    duplicated = split_map.duplicated(subset=["dataset", "Image"], keep=False)
    if duplicated.any():
        dup_preview = split_map.loc[duplicated].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate dataset-image assignments detected: {dup_preview}")

    merged = labeled_df.merge(split_map, on=["dataset", "Image"], how="left", validate="m:1")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), ["dataset", "Image"]].drop_duplicates().head(10)
        raise ValueError(f"Some rows did not receive split assignment: {missing.to_dict(orient='records')}")

    leakage_check = merged.groupby(["dataset", "Image"])["split"].nunique()
    if (leakage_check > 1).any():
        raise ValueError("Leakage detected: a dataset-image pair appears in multiple splits.")

    return merged, split_map


def summarize_split(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_counts = (
        split_df.groupby(["dataset", "split"], as_index=False)["Image"]
        .nunique()
        .rename(columns={"Image": "n_images"})
    )
    cell_counts = split_df.groupby(["dataset", "split"], as_index=False).size().rename(columns={"size": "n_cells"})
    dataset_split_summary = image_counts.merge(cell_counts, on=["dataset", "split"], how="outer").fillna(0)

    split_label_summary = (
        split_df.groupby(["split", "label"], as_index=False).size().rename(columns={"size": "n_cells"})
    )
    split_passage_summary = (
        split_df.groupby(["split", "P"], as_index=False)
        .agg(n_cells=("Cell_ID", "size"), n_images=("Image", "nunique"))
        .sort_values(["split", "P"])
        .reset_index(drop=True)
    )
    return dataset_split_summary, split_label_summary, split_passage_summary


def resolve_feature_columns(
    df: pd.DataFrame,
    explicit_features: list[str] | None,
    include_passage: bool,
) -> list[str]:
    if explicit_features:
        missing = [col for col in explicit_features if col not in df.columns]
        if missing:
            raise ValueError(f"Explicit feature columns not found: {missing}")
        features = list(explicit_features)
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        excluded = {"label", "P", "ki67_positive"}
        features = [c for c in numeric_cols if c not in excluded]

    if include_passage:
        if "P" not in df.columns:
            raise ValueError("Column 'P' does not exist but include_passage=True.")
        if "P" not in features:
            features = features + ["P"]
    else:
        features = [f for f in features if f != "P"]

    if not features:
        raise ValueError("No feature columns selected for modeling.")
    return features


def get_classes_from_pipeline(model: Pipeline) -> np.ndarray | None:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return np.asarray(getattr(model.named_steps["model"], "classes_", None))
    return None


def predict_proba_aligned(model: Pipeline, x: pd.DataFrame, labels: list[int]) -> pd.DataFrame:
    prob_cols = [f"prob_{label}" for label in labels]
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame(index=x.index, columns=prob_cols)

    raw_proba = model.predict_proba(x)
    model_classes = get_classes_from_pipeline(model)
    out = pd.DataFrame(index=x.index, columns=prob_cols, dtype=float)

    if model_classes is None:
        for idx, label in enumerate(labels):
            if idx < raw_proba.shape[1]:
                out[f"prob_{label}"] = raw_proba[:, idx]
        return out

    for idx, cls in enumerate(model_classes):
        col = f"prob_{int(cls)}"
        if col in out.columns:
            out[col] = raw_proba[:, idx]
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
    }


def to_confusion_df(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def per_passage_metrics(pred_df: pd.DataFrame, labels: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for passage, sub in pred_df.groupby("P", dropna=False):
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        metrics = compute_metrics(y_true, y_pred, labels=labels)
        row = {
            "P": np.nan if pd.isna(passage) else float(passage),
            "n_cells": int(len(sub)),
            "n_images": int(sub["Image"].nunique()),
            **metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("P", na_position="last").reset_index(drop=True)


def get_feature_importance(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame | None:
    if not hasattr(model, "named_steps") or "model" not in model.named_steps:
        return None
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
        return (
            pd.DataFrame({"feature": feature_cols, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 2:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
        return (
            pd.DataFrame({"feature": feature_cols, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return None


def build_model_registry(random_state: int, num_classes: int) -> tuple[dict[str, Pipeline], dict[str, str]]:
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=600,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    skipped_optional: dict[str, str] = {}

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=num_classes,
                        eval_metric="mlogloss",
                        n_estimators=600,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        skipped_optional["xgboost"] = str(exc)

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        loss_function="MultiClass",
                        iterations=700,
                        depth=6,
                        learning_rate=0.05,
                        random_seed=random_state,
                        verbose=False,
                    ),
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        skipped_optional["catboost"] = str(exc)

    return models, skipped_optional


def evaluate_model_on_split(
    model: Pipeline,
    model_name: str,
    variant_name: str,
    split_name: str,
    split_frame: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    labels = ORDINAL_LABELS
    x = split_frame[feature_cols]
    y_true = split_frame["label"].to_numpy()
    y_pred = model.predict(x)
    metrics = compute_metrics(y_true, y_pred, labels=labels)

    prediction = split_frame[KEY_COLUMNS + ["P"]].copy()
    prediction["y_true"] = y_true
    prediction["y_pred"] = y_pred
    proba_df = predict_proba_aligned(model, x, labels=labels)
    prediction = pd.concat([prediction.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)

    predictions_dir = output_dir / "predictions"
    confusion_dir = output_dir / "confusion_matrices"
    passage_dir = output_dir / "per_passage"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    passage_dir.mkdir(parents=True, exist_ok=True)

    filename_prefix = f"{variant_name}__{model_name}__{split_name}"
    prediction.to_csv(predictions_dir / f"{filename_prefix}_predictions.csv", index=False)

    cm_df = to_confusion_df(y_true, y_pred, labels=labels)
    cm_df.to_csv(confusion_dir / f"{filename_prefix}_confusion_matrix.csv", index=True)

    per_p_df = per_passage_metrics(prediction, labels=labels)
    per_p_df.to_csv(passage_dir / f"{filename_prefix}_per_passage_metrics.csv", index=False)

    return {
        "variant": variant_name,
        "model": model_name,
        "split": split_name,
        "status": "ok",
        "reason": "",
        "n_samples": int(len(split_frame)),
        **metrics,
    }


def run_benchmark(args: argparse.Namespace) -> Path:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("analysis/ki67_cell_ordinal_benchmark") / f"run_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    include = set(args.dataset_include) if args.dataset_include else None
    exclude = set(args.dataset_exclude) if args.dataset_exclude else None

    raw_df, source_manifest = load_all_features(
        results_root=args.results_root,
        dataset_include=include,
        dataset_exclude=exclude,
    )
    source_manifest.to_csv(output_dir / "source_manifest.csv", index=False)

    if args.export_cell_template is not None:
        export_cell_template(raw_df, args.export_cell_template)

    labeled_df, label_info = attach_labels(raw_df, args)

    split_df, split_map = build_grouped_split(
        labeled_df=labeled_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    split_map.sort_values(["dataset", "split", "Image"]).to_csv(output_dir / "image_split_map.csv", index=False)
    split_df[KEY_COLUMNS + ["P", "label", "split"]].to_csv(
        output_dir / "cell_split_detail.csv", index=False
    )

    dataset_split_summary, split_label_summary, split_passage_summary = summarize_split(split_df)
    dataset_split_summary.to_csv(output_dir / "split_summary_by_dataset.csv", index=False)
    split_label_summary.to_csv(output_dir / "split_summary_by_label.csv", index=False)
    split_passage_summary.to_csv(output_dir / "split_summary_by_passage.csv", index=False)

    models, skipped_optional = build_model_registry(
        random_state=args.random_state,
        num_classes=len(ORDINAL_LABELS),
    )

    config = {
        "results_root": str(args.results_root),
        "output_dir": str(output_dir),
        "random_state": args.random_state,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "dataset_include": sorted(include) if include else None,
        "dataset_exclude": sorted(exclude) if exclude else None,
        "label_info": label_info,
        "models_requested": [
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "xgboost",
            "catboost",
        ],
        "models_run": sorted(models.keys()),
        "models_skipped_optional": skipped_optional,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    split_frames = {"val": val_df, "test": test_df}
    summary_rows: list[dict[str, Any]] = []

    variants = [
        ("without_passage", False),
        ("with_passage", True),
    ]

    for variant_name, include_passage in variants:
        feature_cols = resolve_feature_columns(
            df=split_df,
            explicit_features=args.feature_columns,
            include_passage=include_passage,
        )
        (output_dir / f"{variant_name}_feature_columns.txt").write_text(
            "\n".join(feature_cols) + "\n",
            encoding="utf-8",
        )

        x_train = train_df[feature_cols]
        y_train = train_df["label"].to_numpy()

        if len(train_df) == 0:
            raise RuntimeError("Train split is empty; unable to fit models.")

        for model_name, model in models.items():
            try:
                model.fit(x_train, y_train)
            except Exception as exc:
                reason = f"fit_failed: {exc}"
                for split_name, split_frame in split_frames.items():
                    summary_rows.append(
                        {
                            "variant": variant_name,
                            "model": model_name,
                            "split": split_name,
                            "status": "fail",
                            "reason": reason,
                            "n_samples": int(len(split_frame)),
                            "accuracy": math.nan,
                            "macro_f1": math.nan,
                            "balanced_accuracy": math.nan,
                            "quadratic_weighted_kappa": math.nan,
                        }
                    )
                continue

            importance_df = get_feature_importance(model, feature_cols)
            if importance_df is not None:
                fi_dir = output_dir / "feature_importance"
                fi_dir.mkdir(parents=True, exist_ok=True)
                importance_df.to_csv(fi_dir / f"{variant_name}__{model_name}_importance.csv", index=False)

            for split_name, split_frame in split_frames.items():
                if split_frame.empty:
                    summary_rows.append(
                        {
                            "variant": variant_name,
                            "model": model_name,
                            "split": split_name,
                            "status": "skip",
                            "reason": "empty_split",
                            "n_samples": 0,
                            "accuracy": math.nan,
                            "macro_f1": math.nan,
                            "balanced_accuracy": math.nan,
                            "quadratic_weighted_kappa": math.nan,
                        }
                    )
                    continue

                try:
                    row = evaluate_model_on_split(
                        model=model,
                        model_name=model_name,
                        variant_name=variant_name,
                        split_name=split_name,
                        split_frame=split_frame,
                        feature_cols=feature_cols,
                        output_dir=output_dir,
                    )
                except Exception as exc:
                    row = {
                        "variant": variant_name,
                        "model": model_name,
                        "split": split_name,
                        "status": "fail",
                        "reason": f"eval_failed: {exc}",
                        "n_samples": int(len(split_frame)),
                        "accuracy": math.nan,
                        "macro_f1": math.nan,
                        "balanced_accuracy": math.nan,
                        "quadratic_weighted_kappa": math.nan,
                    }
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    test_ok = summary_df[(summary_df["split"] == "test") & (summary_df["status"] == "ok")].copy()
    if not test_ok.empty:
        rank_df = test_ok.sort_values(
            ["quadratic_weighted_kappa", "macro_f1", "balanced_accuracy", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)
        rank_df.to_csv(output_dir / "model_ranking_test.csv", index=False)

    return output_dir


def main() -> None:
    args = parse_args()
    try:
        out_dir = run_benchmark(args)
    except Exception:
        print("[ERROR] Benchmark failed.")
        print(traceback.format_exc())
        raise
    print(f"[OK] Benchmark complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
