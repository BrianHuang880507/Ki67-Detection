"""Ki67 特徵群、feature combo 與正式候選模型實驗。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, mean_absolute_error, roc_auc_score

from ki67_pred_utils import (
    DEFAULT_EXCLUDED_SOURCE_FOLDERS,
    DEFAULT_INPUT_DIR,
    DEFAULT_RESULTS_DIR,
    aggregate_to_image_features,
    attach_image_embeddings,
    build_feature_groups,
    detect_numeric_feature_columns,
    evaluate_cell_predictions,
    evaluate_image_predictions,
    extract_image_embeddings,
    filter_extreme_image_ratios,
    find_cleaned_csv_files,
    fit_feature_group_models,
    fit_stage1_classifier_with_oof,
    fit_stage2_ratio_model,
    load_ki67_dataset,
    predict_feature_group_scores,
    predict_positive_probability,
    predict_stage2_ratio,
    safe_score_name,
    split_images_within_source_folder,
    summarize_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output"
DEFAULT_EMBEDDING_CACHE = DEFAULT_OUTPUT_DIR / "cache" / "image_embeddings_resnet18_imagenet.joblib"
LEGACY_EMBEDDING_CACHE = (
    PROJECT_ROOT
    / "analysis"
    / "feature_embedding_ablation"
    / "cache"
    / "image_embeddings_resnet18_imagenet.joblib"
)
RANDOM_STATE = 42

SINGLE_GROUPS = [
    "Texture",
    "Intensity distribution",
    "Attachment / spreading",
    "Halo / rounding",
    "Local crowding",
    "Mitosis likelihood",
    "Nuclear morphology",
    "Debris / culture health",
    "Colony / FOV context",
]
MAIN_GROUPS = [
    "Texture",
    "Intensity distribution",
    "Attachment / spreading",
    "Halo / rounding",
]
COMBO_GROUPS = [
    ("Texture",),
    ("Intensity distribution",),
    ("Attachment / spreading",),
    ("Halo / rounding",),
    ("Local crowding",),
    ("Mitosis likelihood",),
    ("Nuclear morphology",),
    ("Debris / culture health",),
    ("Colony / FOV context",),
    ("Texture", "Intensity distribution"),
    ("Texture", "Attachment / spreading"),
    ("Texture", "Halo / rounding"),
    ("Intensity distribution", "Attachment / spreading"),
    ("Intensity distribution", "Halo / rounding"),
    ("Attachment / spreading", "Halo / rounding"),
    ("Texture", "Intensity distribution", "Attachment / spreading"),
    ("Texture", "Intensity distribution", "Halo / rounding"),
    ("Texture", "Attachment / spreading", "Halo / rounding"),
    ("Intensity distribution", "Attachment / spreading", "Halo / rounding"),
    ("Texture", "Intensity distribution", "Attachment / spreading", "Halo / rounding"),
    tuple(SINGLE_GROUPS),
]
FORMAL_GROUPS = [
    ("Texture",),
    ("Intensity distribution",),
    ("Texture", "Intensity distribution"),
    ("Texture", "Attachment / spreading"),
    ("Texture", "Halo / rounding"),
    ("Intensity distribution", "Attachment / spreading"),
    ("Intensity distribution", "Halo / rounding"),
    ("Texture", "Intensity distribution", "Halo / rounding"),
    ("Intensity distribution", "Attachment / spreading", "Halo / rounding"),
    ("Texture", "Intensity distribution", "Attachment / spreading", "Halo / rounding"),
]

ZH_GROUP_LABELS = {
    "Texture": "紋理 Texture",
    "Intensity distribution": "灰階分布 Intensity",
    "Attachment / spreading": "貼附 / 展延形態",
    "Halo / rounding": "Halo / 圓化",
    "Local crowding": "局部擁擠度",
    "Mitosis likelihood": "有絲分裂傾向",
    "Nuclear morphology": "細胞核形態",
    "Debris / culture health": "碎屑 / 培養狀態",
    "Colony / FOV context": "群落 / 視野背景",
}


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    Returns:
        argparse.Namespace: 實驗資料來源、輸出位置與執行選項。
    """
    parser = argparse.ArgumentParser(description="Run Ki67 experiment summaries.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--skip-image-embedding", action="store_true")
    parser.add_argument("--keep-extreme-image-ratios", action="store_true")
    return parser.parse_args()


def setup_font() -> fm.FontProperties | None:
    """設定 Matplotlib 繁體中文字型。

    Returns:
        fm.FontProperties | None: 可用的中文字型；若系統無該字型則回傳 None。
    """
    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    if not font_path.exists():
        return None
    font_prop = fm.FontProperties(fname=str(font_path))
    fm.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return font_prop


def combo_label(groups: Sequence[str]) -> str:
    """將 feature group 組合轉成簡潔名稱。"""
    return " + ".join(
        group.replace("Intensity distribution", "Intensity")
        .replace("Attachment / spreading", "Attachment")
        .replace("Halo / rounding", "Halo")
        .replace("Local crowding", "Crowding")
        .replace("Mitosis likelihood", "Mitosis")
        .replace("Nuclear morphology", "Nuclear")
        .replace("Debris / culture health", "Debris")
        .replace("Colony / FOV context", "FOV")
        for group in groups
    )


def safe_roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """計算 ROC AUC，若資料只有單一類別則回傳 NaN。"""
    y = np.asarray(y_true, dtype=int)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def prepare_data(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """讀取 cleaned CSV、移除異常影像並建立新版切分。

    Args:
        args: 命令列參數。

    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, list[str]]]: train/valid/test
        cell table 與 feature group 欄位清單。
    """
    csv_files = find_cleaned_csv_files(
        args.results_dir,
        excluded_source_folders=DEFAULT_EXCLUDED_SOURCE_FOLDERS,
    )
    dataset = load_ki67_dataset(csv_files, require_label=True)
    if not args.keep_extreme_image_ratios:
        dataset = filter_extreme_image_ratios(dataset)
    splits = split_images_within_source_folder(dataset, random_state=RANDOM_STATE)
    numeric_columns = set(detect_numeric_feature_columns(dataset))
    raw_groups = build_feature_groups(dataset.columns)
    feature_groups = {
        name: [column for column in columns if column in numeric_columns]
        for name, columns in raw_groups.items()
    }
    return {name: frame.reset_index(drop=True) for name, frame in splits.items()}, {
        name: columns for name, columns in feature_groups.items() if columns
    }


def load_or_extract_embeddings(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """載入或抽取 ResNet18 image embedding。

    Args:
        frame: 需要 embedding 的 cell table。
        args: 命令列參數。

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: image embedding table 與 metadata。
    """
    if args.skip_image_embedding:
        return pd.DataFrame(index=[]), {"status": "skipped"}
    cache_path = args.embedding_cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return joblib.load(cache_path), {"status": "loaded_cache", "path": str(cache_path)}
    if LEGACY_EMBEDDING_CACHE.exists():
        embedding_df = joblib.load(LEGACY_EMBEDDING_CACHE)
        joblib.dump(embedding_df, cache_path)
        return embedding_df, {"status": "loaded_legacy_cache", "path": str(cache_path)}
    embedding_df, meta = extract_image_embeddings(frame, pretrained=True, random_state=RANDOM_STATE)
    if not embedding_df.empty:
        joblib.dump(embedding_df, cache_path)
    meta["path"] = str(cache_path)
    return embedding_df, meta


def score_columns_for_combo(groups: Sequence[str]) -> list[str]:
    """取得一組 feature group 對應的 evidence score 欄位名。"""
    return [safe_score_name(group) for group in groups]


def build_stage1_matrix(
    frame: pd.DataFrame,
    evidence_scores: pd.DataFrame,
    score_columns: Sequence[str],
    image_embeddings: pd.DataFrame,
    with_embedding: bool,
) -> pd.DataFrame:
    """建立 Stage 1 模型輸入矩陣。

    Args:
        frame: cell-level 資料。
        evidence_scores: feature-group evidence score table。
        score_columns: 本組實驗採用的 evidence score 欄位。
        image_embeddings: image-level embedding table。
        with_embedding: 是否加入 image embedding。

    Returns:
        pd.DataFrame: Stage 1 design matrix。
    """
    parts = [evidence_scores.reindex(columns=score_columns).reset_index(drop=True)]
    if with_embedding:
        parts.append(attach_image_embeddings(frame, image_embeddings).reset_index(drop=True))
    matrix = pd.concat(parts, axis=1)
    return matrix.loc[:, ~matrix.columns.duplicated()].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def attach_predictions(
    frame: pd.DataFrame,
    probability: np.ndarray,
    evidence_scores: pd.DataFrame,
    score_columns: Sequence[str],
) -> pd.DataFrame:
    """把 Stage 1 probability 與 evidence score 回填到 cell table。"""
    out = frame.reset_index(drop=True).copy()
    for column in score_columns:
        out[column] = evidence_scores.reset_index(drop=True)[column].to_numpy(dtype=float)
    out["cell_prob"] = np.asarray(probability, dtype=float)
    out["cell_pred"] = (out["cell_prob"] >= 0.5).astype(int)
    return out


def evaluate_stage12_combo(
    item: str,
    groups: Sequence[str],
    with_embedding: bool,
    splits: dict[str, pd.DataFrame],
    evidence_by_split: dict[str, pd.DataFrame],
    image_embeddings: pd.DataFrame,
    cv_splits: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """評估一組 Stage 1 / Stage 2 feature combo。

    Args:
        item: 結果類型，例如 ``feature_combo`` 或 ``formal``。
        groups: 使用的 feature group 組合。
        with_embedding: 是否加入 image embedding。
        splits: train/valid/test 資料。
        evidence_by_split: 各 split 的 evidence score。
        image_embeddings: image embedding table。
        cv_splits: Stage 1 OOF fold 數。

    Returns:
        tuple[dict[str, Any], dict[str, pd.DataFrame]]: 指標列與必要 artifacts。
    """
    score_columns = score_columns_for_combo(groups)
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]

    x_train = build_stage1_matrix(train_df, evidence_by_split["train"], score_columns, image_embeddings, with_embedding)
    x_valid = build_stage1_matrix(valid_df, evidence_by_split["valid"], score_columns, image_embeddings, with_embedding)
    x_test = build_stage1_matrix(test_df, evidence_by_split["test"], score_columns, image_embeddings, with_embedding)
    x_valid = x_valid.reindex(columns=x_train.columns, fill_value=0.0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)

    stage1_model, train_prob = fit_stage1_classifier_with_oof(
        x_train,
        train_df["ki67_label"],
        train_df["image_key"].astype(str),
        random_state=RANDOM_STATE,
        cv_splits=cv_splits,
    )
    valid_prob = predict_positive_probability(stage1_model, x_valid)
    test_prob = predict_positive_probability(stage1_model, x_test)

    cell_tables = {
        "train": attach_predictions(train_df, train_prob, evidence_by_split["train"], score_columns),
        "valid": attach_predictions(valid_df, valid_prob, evidence_by_split["valid"], score_columns),
        "test": attach_predictions(test_df, test_prob, evidence_by_split["test"], score_columns),
    }
    image_tables = {
        split: aggregate_to_image_features(cell_table, "cell_prob", score_columns)
        for split, cell_table in cell_tables.items()
    }
    stage2_model, stage2_columns = fit_stage2_ratio_model(image_tables["train"], "S1")
    for image_df in image_tables.values():
        image_df["pred_ratio"] = predict_stage2_ratio(image_df, stage2_model, "S1", stage2_columns)

    train_cell = evaluate_cell_predictions(train_df["ki67_label"], train_prob)
    valid_cell = evaluate_cell_predictions(valid_df["ki67_label"], valid_prob)
    test_cell = evaluate_cell_predictions(test_df["ki67_label"], test_prob)
    train_image = evaluate_image_predictions(image_tables["train"], image_tables["train"]["pred_ratio"])
    valid_image = evaluate_image_predictions(image_tables["valid"], image_tables["valid"]["pred_ratio"])
    test_image = evaluate_image_predictions(image_tables["test"], image_tables["test"]["pred_ratio"])
    row = {
        "item": item,
        "config_key": "_".join(safe_score_name(group).replace("_ki67_evidence", "") for group in groups)
        + ("_with_embedding" if with_embedding else "_no_embedding"),
        "feature_combo": combo_label(groups),
        "feature_group_names": " + ".join(groups),
        "with_image_embedding": bool(with_embedding),
        "stage1_feature_count": int(x_train.shape[1]),
        "train_cell_accuracy": train_cell["accuracy"],
        "valid_cell_accuracy": valid_cell["accuracy"],
        "test_cell_accuracy": test_cell["accuracy"],
        "train_cell_roc_auc": safe_roc_auc(train_df["ki67_label"], train_prob),
        "valid_cell_roc_auc": safe_roc_auc(valid_df["ki67_label"], valid_prob),
        "test_cell_roc_auc": safe_roc_auc(test_df["ki67_label"], test_prob),
        "train_image_mae": train_image["image_mae"],
        "valid_image_mae": valid_image["image_mae"],
        "test_image_mae": test_image["image_mae"],
        "test_tn": test_cell["tn"],
        "test_fp": test_cell["fp"],
        "test_fn": test_cell["fn"],
        "test_tp": test_cell["tp"],
    }
    return row, {"test_cell": cell_tables["test"]}


def relation_rows(
    splits: dict[str, pd.DataFrame],
    evidence_by_split: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """計算單一 feature group 與 Ki67 的關聯性列。"""
    rows: list[dict[str, Any]] = []
    for group in SINGLE_GROUPS:
        score_column = safe_score_name(group)
        if group not in feature_groups or score_column not in evidence_by_split["test"].columns:
            continue
        row: dict[str, Any] = {
            "item": "feature_group_relation",
            "feature_group": group,
            "feature_combo": combo_label((group,)),
            "parameters_used": int(len(feature_groups[group])),
        }
        for split in ("train", "valid", "test"):
            frame = splits[split]
            score = evidence_by_split[split][score_column].to_numpy(dtype=float)
            scored = frame.loc[:, ["image_key", "source_folder", "Image", "ki67_label"]].copy()
            scored[score_column] = score
            image_df = (
                scored.groupby("image_key", as_index=False)
                .agg(
                    source_folder=("source_folder", "first"),
                    true_ratio=("ki67_label", "mean"),
                    pred_ratio=(score_column, "mean"),
                    cell_count=(score_column, "size"),
                )
            )
            row[f"{split}_cell_accuracy"] = float(accuracy_score(frame["ki67_label"], score >= 0.5))
            row[f"{split}_cell_roc_auc"] = safe_roc_auc(frame["ki67_label"], score)
            row[f"{split}_image_mae"] = float(mean_absolute_error(image_df["true_ratio"], image_df["pred_ratio"]))
        row["relation_score"] = row["test_cell_roc_auc"]
        rows.append(row)
    return rows


def split_rows(splits: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """把切分摘要轉成總實驗 CSV 的列。"""
    rows = []
    for split in ("train", "valid", "test"):
        row = summarize_split(splits[split], split)
        row["item"] = "split_summary"
        row["ratio_by_images"] = row["images"] / sum(splits[name]["image_key"].nunique() for name in splits)
        rows.append(row)
    return rows


def save_relation_chart(summary: pd.DataFrame, output_path: Path, font_prop: fm.FontProperties | None) -> None:
    """輸出單一 feature group 關聯性圖。"""
    plot_df = summary[summary["item"] == "feature_group_relation"].copy()
    plot_df = plot_df.sort_values("relation_score", ascending=False).reset_index(drop=True)
    plot_df["label"] = plot_df["feature_group"].map(ZH_GROUP_LABELS).fillna(plot_df["feature_group"])
    colors = []
    for value in plot_df["relation_score"]:
        if value >= 0.80:
            colors.append("#2ca25f")
        elif value >= 0.68:
            colors.append("#3182bd")
        elif value >= 0.60:
            colors.append("#756bb1")
        else:
            colors.append("#9e9ac8")
    fig, ax = plt.subplots(figsize=(10.6, 7.2), dpi=220)
    y = np.arange(len(plot_df))
    bars = ax.barh(y, plot_df["relation_score"], color=colors, edgecolor="#334155", linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontproperties=font_prop, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0.50, color="#64748b", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.set_title("單一 Feature Group 與 Ki67 關聯性比較", fontproperties=font_prop, fontsize=18)
    ax.set_xlabel("與 Ki67 的關聯性", fontproperties=font_prop, fontsize=13)
    ax.set_ylabel("Feature group", fontproperties=font_prop, fontsize=13)
    ax.set_xlim(0.45, 0.85)
    ax.grid(axis="x", color="#dbe3ec", linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, plot_df["relation_score"]):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_embedding_mae_chart(summary: pd.DataFrame, output_path: Path, item: str, title: str) -> None:
    """輸出加入/不加入 image embedding 的 MAE 比較圖。"""
    plot_df = summary[summary["item"] == item].copy()
    pivot = plot_df.pivot_table(index="feature_combo", columns="with_image_embedding", values="test_image_mae").dropna()
    pivot = pivot.sort_values(False).head(16)
    x = np.arange(len(pivot))
    width = 0.36
    fig_width = max(11.0, 0.78 * len(pivot) + 3.8)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0), dpi=220)
    bars_no = ax.bar(x - width / 2, pivot[False] * 100.0, width, color="#3182bd", edgecolor="#334155", label="不加入 image embedding")
    bars_yes = ax.bar(x + width / 2, pivot[True] * 100.0, width, color="#f59e0b", edgecolor="#334155", label="加入 image embedding")
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Test image Ki67 positive rate MAE (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=28, ha="right", fontsize=8.5)
    ax.grid(axis="y", color="#dbe3ec", linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)
    for bars in (bars_no, bars_yes):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_formal_mae_chart(summary: pd.DataFrame, output_path: Path) -> None:
    """輸出正式候選模型的 image-level MAE 排序圖。"""
    plot_df = summary[summary["item"] == "formal"].copy()
    plot_df["mae_pp"] = plot_df["test_image_mae"] * 100.0
    plot_df["label"] = plot_df["feature_combo"] + "\n" + np.where(plot_df["with_image_embedding"], "With embedding", "No embedding")
    plot_df = plot_df.sort_values("mae_pp").reset_index(drop=True)
    colors = np.where(plot_df["with_image_embedding"], "#f59e0b", "#2ca25f")
    fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=220)
    bars = ax.bar(np.arange(len(plot_df)), plot_df["mae_pp"], color=colors, edgecolor="#334155", linewidth=0.6)
    ax.set_title("Test image Ki67 positive rate MAE by formal combo", fontsize=16)
    ax.set_ylabel("MAE (pp)")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", color="#dbe3ec", linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, plot_df["mae_pp"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices(
    formal_summary: pd.DataFrame,
    artifacts: dict[str, dict[str, pd.DataFrame]],
    output_path: Path,
) -> None:
    """輸出正式候選模型的 test confusion matrix。"""
    plot_df = formal_summary.sort_values("test_image_mae").head(12).reset_index(drop=True)
    n_cols = 4
    n_rows = int(np.ceil(len(plot_df) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.5 * n_rows), dpi=220)
    axes = np.asarray(axes).reshape(-1)
    for ax, row in zip(axes, plot_df.itertuples()):
        test_cell = artifacts[row.config_key]["test_cell"]
        cm = confusion_matrix(test_cell["ki67_label"], test_cell["cell_pred"], labels=[0, 1])
        display = ConfusionMatrixDisplay(cm, display_labels=["Ki67-", "Ki67+"])
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{row.feature_combo}\n{'With' if row.with_image_embedding else 'No'} embedding | Acc {row.test_cell_accuracy:.1%}", fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
    for ax in axes[len(plot_df) :]:
        ax.axis("off")
    fig.suptitle("Formal Stage 1 Test Confusion Matrices", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_version_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """輸出舊版與目前最佳正式模型比較圖。"""
    best = summary[summary["item"] == "formal"].sort_values("test_image_mae").iloc[0]
    old_mae = 17.34
    old_acc = 71.02
    new_mae = float(best["test_image_mae"]) * 100.0
    new_acc = float(best["test_cell_accuracy"]) * 100.0
    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=220)
    x = np.arange(2)
    width = 0.32
    bars_old = ax.bar(x - width / 2, [old_mae, old_acc], width, color="#7b8798", edgecolor="#334155", label="舊版\nFeature parameters only / S1")
    bars_new = ax.bar(x + width / 2, [new_mae, new_acc], width, color="#2f80ed", edgecolor="#334155", label=f"目前版\n{best['feature_combo']} / S1")
    ax.set_title("舊版與目前版 Ki67 預測效果比較", fontsize=16)
    ax.set_ylabel("數值")
    ax.set_xticks(x)
    ax.set_xticklabels(["Test image Ki67 MAE\n(pp; lower is better)", "Test single-cell accuracy\n(%; higher is better)"])
    ax.set_ylim(0, 82)
    ax.grid(axis="y", color="#dbe3ec", linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)
    for bars, suffixes in ((bars_old, [" pp", "%"]), (bars_new, [" pp", "%"])):
        for bar, suffix in zip(bars, suffixes):
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.2f}{suffix}", ha="center", va="bottom", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_experiments(args: argparse.Namespace) -> pd.DataFrame:
    """執行所有新版 Ki67 實驗。

    Args:
        args: 命令列參數。

    Returns:
        pd.DataFrame: 所有實驗與 split summary 的整合結果表。
    """
    output_dir = args.output_dir.resolve()
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    splits, feature_groups = prepare_data(args)
    group_names = [name for name in SINGLE_GROUPS if name in feature_groups]
    feature_models, train_evidence = fit_feature_group_models(
        splits["train"],
        feature_groups,
        group_names,
        cv_splits=args.cv_splits,
        random_state=RANDOM_STATE,
        group_column="image_key",
    )
    evidence_by_split = {
        "train": train_evidence,
        "valid": predict_feature_group_scores(splits["valid"], feature_models),
        "test": predict_feature_group_scores(splits["test"], feature_models),
    }

    all_frames = pd.concat([splits["train"], splits["valid"], splits["test"]], ignore_index=True)
    image_embeddings, embedding_meta = load_or_extract_embeddings(all_frames, args)
    rows: list[dict[str, Any]] = []
    rows.extend(split_rows(splits))
    rows.extend(relation_rows(splits, evidence_by_split, feature_groups))

    formal_artifacts: dict[str, dict[str, pd.DataFrame]] = {}
    for item, combos in (("feature_combo", COMBO_GROUPS), ("formal", FORMAL_GROUPS)):
        for groups in combos:
            if any(group not in feature_groups for group in groups):
                continue
            for with_embedding in (False, True):
                if with_embedding and image_embeddings.empty:
                    continue
                row, artifacts = evaluate_stage12_combo(
                    item,
                    groups,
                    with_embedding,
                    splits,
                    evidence_by_split,
                    image_embeddings,
                    cv_splits=args.cv_splits,
                )
                rows.append(row)
                if item == "formal":
                    formal_artifacts[row["config_key"]] = artifacts
                print(
                    f"[完成] {item} | {row['feature_combo']} | embedding={with_embedding} | "
                    f"test MAE={row['test_image_mae'] * 100:.2f} pp | "
                    f"acc={row['test_cell_accuracy']:.2%}",
                    flush=True,
                )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "ki67_expt_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with (output_dir / "ki67_expt_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "random_state": RANDOM_STATE,
                "cv_splits": int(args.cv_splits),
                "embedding_meta": embedding_meta,
                "feature_groups_used": {name: feature_groups[name] for name in group_names},
                "output_summary": str(summary_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    setup_font()
    save_relation_chart(summary, charts_dir / "feature_group_relation.png", None)
    save_embedding_mae_chart(summary, charts_dir / "feature_combo_embedding_mae.png", "feature_combo", "Image embedding 對 Feature combo Test MAE 的影響")
    save_embedding_mae_chart(summary, charts_dir / "formal_embedding_mae.png", "formal", "Image embedding 對正式流程 Test MAE 的影響")
    save_formal_mae_chart(summary, charts_dir / "formal_combo_image_mae.png")
    formal_summary = summary[summary["item"] == "formal"].copy()
    if formal_artifacts:
        save_confusion_matrices(formal_summary, formal_artifacts, charts_dir / "formal_confusion_matrices.png")
    save_version_comparison(summary, charts_dir / "version_effect_comparison.png")
    print(f"[輸出] {summary_path}")
    return summary


def main() -> int:
    """命令列入口。"""
    run_experiments(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
