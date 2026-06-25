"""Ki67 新版訓練與預測共用函式。"""

from __future__ import annotations

import html
import json
import math
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "output" / "results"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output"
DEFAULT_TRAIN_OUTPUT_DIR = DEFAULT_ANALYSIS_OUTPUT_DIR / "training"
DEFAULT_PREDICT_OUTPUT_DIR = DEFAULT_ANALYSIS_OUTPUT_DIR / "prediction"
DEFAULT_SUMMARY_HTML = DEFAULT_TRAIN_OUTPUT_DIR / "summary.html"

DEFAULT_DATA_PATTERN = "*_cleaned.csv"
DEFAULT_LABEL_CANDIDATES = ("ki67_positive", "ki67_label", "label", "target")
DEFAULT_PASSAGE_CANDIDATES = ("passage", "Passage", "passage_id")
DEFAULT_EXCLUDED_SOURCE_FOLDERS = (
    "0819",
    "ki67trainset-20250806",
    "ki67trainset-20250807",
)
DEFAULT_VALID_SOURCE_FOLDERS = (
    "2025-06-19-B4-P6-P10-P14-Ki67-P10-1",
    "2025-06-19-B4-P6-P10-P14-Ki67-P14-1",
    "2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P6",
    "P7-P10-P9",
)
DEFAULT_TEST_SOURCE_FOLDERS = ("P6-1", "P6-2", "P6-3", "P12-1", "P12-2", "P12-3")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

INTENSITY_BASES = (
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
)
DERIVED_INTENSITY_COLUMNS = (
    "Nuc Cyto Mean Ratio",
    "Nuc Cyto IntDen Ratio",
    "Nuc Cyto RawIntDen Ratio",
    "Nuc Cell IntDen Ratio",
    "Nuc Cyto Entropy Difference",
    "Nuc Cyto CV Difference",
)
HALO_COLUMNS = (
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
)
GEOMETRY_BASES = (
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
)
SHAPE_COLUMNS = (
    "Protrusion Count",
    "Mean Convex Defect Depth",
    "Mean Protrusion Length Norm",
    "Max Convex Defect Depth",
    "Fractal Dimension",
    "Boundary Inflection Count",
)
LOCAL_CROWDING_COLUMNS = (
    "Nearest Neighbor Distance",
    "Nearest Neighbor Distance Norm",
    "Local Neighbor Count",
    "Local Density",
    "Neighbour Area Ratio",
)
COLONY_CONTEXT_COLUMNS = (
    "Image Confluency",
    "Population Area CV",
    "Population Circularity CV",
    "Cluster Size",
    "Cluster Size Norm",
    "Largest Cluster Ratio",
)
MITOSIS_COLUMNS = (
    "Mitotic Score",
    "Daughter Pair Flag",
    "Protrusion Retraction Score",
    "Mitotic Index",
)
DEBRIS_COLUMNS = (
    "Debris Count",
    "Debris Area Fraction",
    "Nearest Debris Distance",
    "Debris Mean Area",
    "Debris Density",
)
NUCLEOLUS_COLUMNS = (
    "Nucleolus Count",
    "Mean Nucleolus Area",
    "Max Nucleolus Area",
)
TEXTURE_TOKENS = ("GLCM ", "LBP ", "Tamura Coarseness", "Zernike Moment")
META_COLUMNS = {
    "Cell_ID",
    "Image",
    "source_file",
    "source_folder",
    "image_key",
    "passage",
    "_source_row_id",
    "split",
    "ki67_label",
    "ki67_positive",
    "cell_status",
}
PROBABILITY_SUMMARY_COLUMNS = (
    "mean_cell_prob",
    "median_cell_prob",
    "p90_cell_prob",
    "frac_prob_gt_0_3",
    "frac_prob_gt_0_5",
    "frac_prob_gt_0_7",
    "cell_count",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """描述一組 Stage 1 / Stage 2 實驗設定。

    Args:
        key: 程式內使用的穩定識別名稱。
        display_name: 報告表格顯示名稱。
        stage1_input_name: Stage 1 輸入摘要。
        stage2_name: Stage 2 輸入或模型摘要。
        include_feature_scores: 是否使用 feature-group Ki67 evidence scores。
        include_cnn_embedding: 是否使用 ResNet18 image embedding。
        parameter_mode: 參數輸入模式，支援 ``none``、``selected``、``all``。
        stage2_mode: Stage 2 模式，支援 ``S0``、``S1``、``S3``、``S4``。
        feature_score_groups: 使用哪些 feature-group evidence scores；``None`` 表示使用全部。
    """

    key: str
    display_name: str
    stage1_input_name: str
    stage2_name: str
    include_feature_scores: bool
    include_cnn_embedding: bool
    parameter_mode: str
    stage2_mode: str
    feature_score_groups: tuple[str, ...] | None = None


@dataclass
class TableFeatureTransformer:
    """表格特徵前處理器。

    Args:
        feature_columns: 原始欄位名稱。
        imputer: 缺失值補值器。
        scaler: 標準化器。
        selected_columns: 轉換後保留的欄位名稱。
        selector: 選欄位器，沒有使用時為 ``None``。
    """

    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    selected_columns: list[str]
    selector: Any | None = None


@dataclass
class FeatureGroupModel:
    """單一 feature group 的 Ki67 evidence model。

    Args:
        group_name: Feature group 名稱。
        feature_columns: 該 group 使用的 feature parameters。
        imputer: 訓練資料 fit 出來的缺失值補值器。
        scaler: 訓練資料 fit 出來的標準化器。
        model: Elastic-net logistic classifier。
        score_column: 輸出的 evidence score 欄位名稱。
    """

    group_name: str
    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    model: SGDClassifier
    score_column: str


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """儲存 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """讀取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def reset_directory(path: Path) -> None:
    """清空並重建資料夾。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def source_folder_key(value: str) -> str:
    """將 source folder 名稱轉為比對用 key。"""
    return str(value).strip().lower()


def infer_passage_from_text(text: str) -> str:
    """從文字推估 passage。"""
    matches = re.findall(r"(?i)\bP(?:assage)?[-_ ]?(\d{1,2})\b", str(text))
    if matches:
        return f"P{int(matches[-1])}"
    return "P_UNKNOWN"


def pick_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    """從候選欄位中挑出第一個存在者。"""
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = lookup.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


def find_cleaned_csv_files(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    pattern: str = DEFAULT_DATA_PATTERN,
    excluded_source_folders: Sequence[str] = DEFAULT_EXCLUDED_SOURCE_FOLDERS,
    include_source_folders: Sequence[str] | None = None,
) -> list[Path]:
    """尋找可用的 cleaned CSV。"""
    excluded = {source_folder_key(item) for item in excluded_source_folders}
    included = None
    if include_source_folders:
        included = {source_folder_key(item) for item in include_source_folders}

    paths: list[Path] = []
    for csv_path in sorted(results_dir.rglob(pattern)):
        source_folder = csv_path.parent.name
        key = source_folder_key(source_folder)
        if key in excluded:
            continue
        if included is not None and key not in included:
            continue
        paths.append(csv_path)
    return paths


def build_image_key(frame: pd.DataFrame) -> pd.Series:
    """建立 source_folder::Image 的影像鍵。"""
    image = frame["Image"].astype(str)
    return frame["source_folder"].astype(str) + "::" + image


def load_ki67_dataset(
    csv_files: Sequence[Path],
    require_label: bool = True,
    label_candidates: Sequence[str] = DEFAULT_LABEL_CANDIDATES,
    passage_candidates: Sequence[str] = DEFAULT_PASSAGE_CANDIDATES,
) -> pd.DataFrame:
    """讀取 Ki67 cleaned CSV 並整理成 cell-level table。

    Args:
        csv_files: ``*_cleaned.csv`` 檔案清單。
        require_label: 是否要求資料必須含 Ki67 ground truth。
        label_candidates: Ki67 label 候選欄位。
        passage_candidates: Passage 候選欄位。

    Returns:
        pd.DataFrame: 含 ``source_folder``、``Image``、``image_key`` 與
        ``ki67_label`` 的 cell-level table。
    """
    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        local = pd.read_csv(csv_path)
        local = local.loc[:, ~local.columns.duplicated()].copy()

        label_col = pick_existing_column(local.columns, label_candidates)
        if require_label and label_col is None:
            continue
        if label_col is not None:
            local["ki67_label"] = pd.to_numeric(local[label_col], errors="coerce")
            local = local.dropna(subset=["ki67_label"]).copy()
            local["ki67_label"] = (local["ki67_label"] > 0).astype(int)

        passage_col = pick_existing_column(local.columns, passage_candidates)
        if passage_col is not None:
            local["passage"] = local[passage_col].astype(str).map(infer_passage_from_text)
        else:
            local["passage"] = infer_passage_from_text(csv_path.parent.name)

        if "Image" not in local.columns:
            stem = csv_path.name.replace("_cleaned.csv", "")
            local["Image"] = stem

        local["source_file"] = str(csv_path)
        local["source_folder"] = csv_path.parent.name
        local["_source_row_id"] = np.arange(len(local), dtype=np.int64)
        local["image_key"] = build_image_key(local)
        frames.append(local)

    if not frames:
        raise ValueError("找不到可用的 Ki67 cleaned CSV。")
    dataset = pd.concat(frames, ignore_index=True)
    return dataset.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def filter_extreme_image_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """移除 Ki67 陽性比例為 0 或 100% 的影像。"""
    if "ki67_label" not in frame.columns:
        return frame.reset_index(drop=True)
    ratio = frame.groupby("image_key")["ki67_label"].transform("mean")
    return frame[(ratio > 0.0) & (ratio < 1.0)].copy().reset_index(drop=True)


def split_by_source_folder(
    dataset: pd.DataFrame,
    valid_source_folders: Sequence[str] = DEFAULT_VALID_SOURCE_FOLDERS,
    test_source_folders: Sequence[str] = DEFAULT_TEST_SOURCE_FOLDERS,
) -> dict[str, pd.DataFrame]:
    """依 source folder 切成 train / valid / test sets。"""
    valid = {source_folder_key(item) for item in valid_source_folders}
    test = {source_folder_key(item) for item in test_source_folders}
    source_key = dataset["source_folder"].astype(str).map(source_folder_key)

    train_df = dataset[~source_key.isin(valid | test)].copy().reset_index(drop=True)
    valid_df = dataset[source_key.isin(valid)].copy().reset_index(drop=True)
    test_df = dataset[source_key.isin(test)].copy().reset_index(drop=True)
    return {"train": train_df, "valid": valid_df, "test": test_df}


def _stable_folder_seed(folder_name: str, random_state: int) -> int:
    """Return a deterministic seed for one source folder."""
    folder_crc = zlib.crc32(str(folder_name).encode("utf-8")) & 0xFFFFFFFF
    return int((int(random_state) + folder_crc) % (2**32 - 1))


def _split_counts(n_items: int, train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """Allocate image counts while keeping all splits non-empty when possible."""
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    if n_items == 2:
        return 1, 0, 1

    ratios = np.asarray([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0) or not np.isfinite(ratios).all() or ratios.sum() <= 0:
        raise ValueError("Split ratios must be finite non-negative values with a positive sum.")
    ratios = ratios / ratios.sum()

    raw = ratios * n_items
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(raw - counts)[::-1][: int(n_items - counts.sum())]:
        counts[idx] += 1

    for idx in range(3):
        if counts[idx] == 0:
            donor_order = np.argsort(counts)[::-1]
            donor = next((int(item) for item in donor_order if counts[item] > 1), None)
            if donor is not None:
                counts[donor] -= 1
                counts[idx] += 1

    while counts.sum() > n_items:
        counts[int(np.argmax(counts))] -= 1
    while counts.sum() < n_items:
        counts[int(np.argmax(ratios))] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _ratio_strata(values: pd.Series, max_bins: int = 5) -> pd.Series | None:
    """Build robust ratio strata for image-level stratified splitting."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.nunique(dropna=True) < 2:
        return None
    bins = min(int(max_bins), int(numeric.nunique(dropna=True)), len(numeric))
    if bins < 2:
        return None
    try:
        strata = pd.qcut(numeric.rank(method="first"), q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    strata = pd.Series(strata, index=values.index)
    if strata.nunique(dropna=True) < 2 or strata.value_counts().min() < 2:
        return None
    return strata


def _safe_image_train_test_split(
    image_df: pd.DataFrame,
    test_size: int,
    random_state: int,
    stratify_col: str = "true_ratio",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an image table with stratification when feasible."""
    if test_size <= 0:
        return image_df.copy(), image_df.iloc[0:0].copy()
    if test_size >= len(image_df):
        return image_df.iloc[0:0].copy(), image_df.copy()

    strata = _ratio_strata(image_df[stratify_col])
    try:
        train_part, test_part = train_test_split(
            image_df,
            test_size=int(test_size),
            random_state=int(random_state),
            shuffle=True,
            stratify=strata,
        )
    except ValueError:
        train_part, test_part = train_test_split(
            image_df,
            test_size=int(test_size),
            random_state=int(random_state),
            shuffle=True,
            stratify=None,
        )
    return train_part.copy(), test_part.copy()


def split_images_within_source_folder(
    dataset: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split images within every source_folder into train / valid / test sets."""
    if "image_key" not in dataset.columns:
        dataset = dataset.copy()
        dataset["image_key"] = build_image_key(dataset)
    if "source_folder" not in dataset.columns:
        raise ValueError("dataset must contain source_folder.")

    if "ki67_label" in dataset.columns:
        image_rows = (
            dataset.groupby(["source_folder", "image_key"], as_index=False)["ki67_label"]
            .mean()
            .rename(columns={"ki67_label": "true_ratio"})
        )
    else:
        image_rows = (
            dataset.groupby(["source_folder", "image_key"], as_index=False)
            .size()
            .rename(columns={"size": "true_ratio"})
        )
    image_rows = image_rows.sort_values(["source_folder", "image_key"]).reset_index(drop=True)

    assignment_frames: list[pd.DataFrame] = []
    for source_folder, folder_images in image_rows.groupby("source_folder", sort=True):
        folder_images = folder_images.reset_index(drop=True)
        n_train, n_valid, n_test = _split_counts(
            len(folder_images),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )
        seed = _stable_folder_seed(str(source_folder), random_state)
        train_images, temp_images = _safe_image_train_test_split(
            folder_images,
            test_size=n_valid + n_test,
            random_state=seed,
        )
        if len(temp_images) and n_valid > 0 and n_test > 0:
            valid_images, test_images = _safe_image_train_test_split(
                temp_images,
                test_size=n_test,
                random_state=seed + 17,
            )
        elif n_test > 0:
            valid_images = temp_images.iloc[0:0].copy()
            test_images = temp_images.copy()
        else:
            valid_images = temp_images.copy()
            test_images = temp_images.iloc[0:0].copy()

        for part, split_name in (
            (train_images, "train"),
            (valid_images, "valid"),
            (test_images, "test"),
        ):
            if part.empty:
                continue
            local = part[["image_key"]].copy()
            local["split"] = split_name
            assignment_frames.append(local)

    if not assignment_frames:
        raise ValueError("No images available for train / valid / test split.")
    assignments = pd.concat(assignment_frames, ignore_index=True)
    split_lookup = assignments.set_index("image_key")["split"]
    split_values = dataset["image_key"].map(split_lookup)
    if split_values.isna().any():
        missing = int(split_values.isna().sum())
        raise ValueError(f"{missing} rows were not assigned to a split.")

    return {
        split_name: dataset[split_values == split_name].copy().reset_index(drop=True)
        for split_name in ("train", "valid", "test")
    }


def summarize_split(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    """彙整切分資料量。"""
    return {
        "split": split_name,
        "source_folders": int(frame["source_folder"].nunique()),
        "images": int(frame["image_key"].nunique()),
        "cells": int(len(frame)),
        "positive_ratio": (
            float(frame["ki67_label"].mean()) if "ki67_label" in frame.columns and len(frame) else np.nan
        ),
        "source_folder_list": ", ".join(sorted(frame["source_folder"].astype(str).unique())),
    }


def detect_numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    """偵測可作為模型輸入的數值 feature parameters。"""
    columns = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in META_COLUMNS:
            continue
        columns.append(str(column))
    if not columns:
        raise ValueError("找不到數值 feature parameters。")
    return columns


def existing_columns(columns: Iterable[str], names: Sequence[str]) -> list[str]:
    """依原始順序保留存在的欄位。"""
    available = set(columns)
    return [name for name in names if name in available]


def texture_columns(columns: Iterable[str]) -> list[str]:
    """選出 Texture feature parameters。"""
    return [col for col in columns if any(token in str(col) for token in TEXTURE_TOKENS)]


def intensity_columns(columns: Iterable[str]) -> list[str]:
    """選出 Intensity distribution feature parameters。"""
    result: list[str] = []
    available = set(columns)
    for base in INTENSITY_BASES:
        candidates = [f"{base}_nuc", f"{base}_cyto", f"Whole Cell {base}"]
        result.extend([col for col in candidates if col in available])
    result.extend(existing_columns(columns, DERIVED_INTENSITY_COLUMNS))
    return list(dict.fromkeys(result))


def build_feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """建立 Ki67 實驗使用的 feature group 欄位清單。

    Args:
        columns: cleaned CSV 中可用的欄位名稱。

    Returns:
        dict[str, list[str]]: feature group 名稱與對應 feature parameter
        欄位清單。
    """
    all_columns = list(columns)
    nucleus_geometry = [f"{base}_nuc" for base in GEOMETRY_BASES]
    cyto_geometry = [f"{base}_cyto" for base in GEOMETRY_BASES]
    texture = texture_columns(all_columns)
    intensity = intensity_columns(all_columns)
    halo = existing_columns(all_columns, HALO_COLUMNS)
    attachment = existing_columns(
        all_columns,
        [*cyto_geometry, *SHAPE_COLUMNS, "Nucleus Centroid Offset"],
    )
    nuclear = existing_columns(
        all_columns,
        [*nucleus_geometry, "Karyoplasmic Ratio", "Nucleus Centroid Offset", *NUCLEOLUS_COLUMNS],
    )
    groups = {
        "Texture": texture,
        "Intensity distribution": intensity,
        "Attachment / spreading": attachment,
        "Halo / rounding": halo,
        "Local crowding": existing_columns(all_columns, LOCAL_CROWDING_COLUMNS),
        "Colony / FOV context": existing_columns(all_columns, COLONY_CONTEXT_COLUMNS),
        "Mitosis likelihood": existing_columns(all_columns, MITOSIS_COLUMNS),
        "Nuclear morphology": nuclear,
        "Debris / culture health": existing_columns(all_columns, DEBRIS_COLUMNS),
    }
    groups = {name: values for name, values in groups.items() if values}
    groups["Texture + Intensity"] = sorted(set(texture + intensity))
    groups["Texture + Attachment"] = sorted(set(texture + attachment))
    groups["Texture + Halo"] = sorted(set(texture + halo))
    groups["Intensity + Attachment"] = sorted(set(intensity + attachment))
    groups["Intensity + Halo"] = sorted(set(intensity + halo))
    groups["Attachment + Halo"] = sorted(set(attachment + halo))
    groups["Texture + Intensity + Attachment"] = sorted(set(texture + intensity + attachment))
    groups["Texture + Intensity + Halo"] = sorted(set(texture + intensity + halo))
    groups["Texture + Attachment + Halo"] = sorted(set(texture + attachment + halo))
    groups["Intensity + Attachment + Halo"] = sorted(set(intensity + attachment + halo))
    groups["Texture + Intensity + Attachment + Halo"] = sorted(
        set(texture + intensity + attachment + halo)
    )
    groups["All PC parameters"] = sorted(
        {column for values in groups.values() for column in values}
    )
    return {name: values for name, values in groups.items() if values}


def safe_score_name(group_name: str) -> str:
    """將 feature group 名稱轉為 evidence score 欄名。"""
    safe = (
        group_name.lower()
        .replace(" / ", "_")
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"{safe}_ki67_evidence"


def balanced_image_weights(frame: pd.DataFrame) -> np.ndarray:
    """讓每張影像在 cell-level model 中權重接近一致。"""
    counts = frame.groupby("image_key")["image_key"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=np.float64)
    return weights / float(np.mean(weights))


def fit_table_transformer(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    selected_columns: Sequence[str] | None = None,
    selector: Any | None = None,
) -> TableFeatureTransformer:
    """Fit 數值表格前處理器。"""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_raw = train_df.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    x_imp = imputer.fit_transform(x_raw)
    if selector is not None:
        x_imp = selector.fit_transform(x_imp, train_df["ki67_label"].to_numpy(dtype=np.int64))
        selected = list(selected_columns or [f"selected_param_{i:03d}" for i in range(x_imp.shape[1])])
    else:
        selected = list(selected_columns or feature_columns)
    scaler.fit(x_imp)
    return TableFeatureTransformer(
        feature_columns=list(feature_columns),
        imputer=imputer,
        scaler=scaler,
        selected_columns=selected,
        selector=selector,
    )


def transform_table_features(frame: pd.DataFrame, transformer: TableFeatureTransformer) -> pd.DataFrame:
    """使用已 fit 的表格前處理器轉換資料。"""
    x_raw = frame.reindex(columns=transformer.feature_columns).apply(pd.to_numeric, errors="coerce")
    x_imp = transformer.imputer.transform(x_raw)
    if transformer.selector is not None:
        x_imp = transformer.selector.transform(x_imp)
    x_scaled = transformer.scaler.transform(x_imp)
    return pd.DataFrame(x_scaled, columns=transformer.selected_columns, index=frame.index)


def fit_selected_parameter_transformer(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    top_k: int = 40,
) -> TableFeatureTransformer:
    """Fit ANOVA F-score top-k selected feature parameters。"""
    top_k = min(int(top_k), len(feature_columns))
    selector = SelectKBest(score_func=f_classif, k=top_k)
    names = [f"selected_param_{i:02d}" for i in range(top_k)]
    return fit_table_transformer(train_df, feature_columns, names, selector=selector)


def fit_elastic_net_evidence_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None,
    random_state: int,
) -> SGDClassifier:
    """訓練 elastic-net logistic classifier。"""
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.15,
        max_iter=2500,
        tol=1e-4,
        random_state=random_state,
    )
    model.fit(x_train, y_train.astype(int), sample_weight=sample_weight)
    return model


def predict_positive_probability(model: Any, x_frame: pd.DataFrame) -> np.ndarray:
    """輸出 positive class probability。"""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_frame)[:, 1], dtype=np.float64)
    decision = np.asarray(model.decision_function(x_frame), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-decision))


def fit_feature_group_models(
    train_df: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    group_names: Sequence[str],
    cv_splits: int = 5,
    random_state: int = 42,
    group_column: str = "source_folder",
) -> tuple[dict[str, FeatureGroupModel], pd.DataFrame]:
    """訓練 feature-group evidence models 並產生 train OOF scores。

    Args:
        train_df: 訓練資料。
        groups: Feature group 與其欄位清單。
        group_names: 要訓練的 feature group 名稱。
        cv_splits: GroupKFold 切分數。
        random_state: 隨機種子。
        group_column: GroupKFold 使用的欄位，image-level split 時建議使用
            ``image_key``。

    Returns:
        tuple[dict[str, FeatureGroupModel], pd.DataFrame]: 最終模型與 train
        out-of-fold evidence score table。
    """
    evidence = pd.DataFrame(index=train_df.index)
    models: dict[str, FeatureGroupModel] = {}
    y = train_df["ki67_label"].astype(int)
    if group_column not in train_df.columns:
        raise ValueError(f"group_column not found: {group_column}")
    source_groups = train_df[group_column].astype(str)
    unique_groups = int(source_groups.nunique())
    n_splits = min(max(2, int(cv_splits)), unique_groups)

    for group_name in group_names:
        columns = [col for col in groups.get(group_name, []) if col in train_df.columns]
        if not columns:
            continue
        score_column = safe_score_name(group_name)
        oof = np.full(len(train_df), np.nan, dtype=np.float64)
        splitter = GroupKFold(n_splits=n_splits)

        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(train_df, y, groups=source_groups)):
            fold_train = train_df.iloc[fit_idx]
            fold_valid = train_df.iloc[val_idx]
            transformer = fit_table_transformer(fold_train, columns)
            x_fit = transform_table_features(fold_train, transformer)
            x_val = transform_table_features(fold_valid, transformer)
            model = fit_elastic_net_evidence_model(
                x_fit,
                fold_train["ki67_label"],
                balanced_image_weights(fold_train),
                random_state + fold_idx,
            )
            oof[val_idx] = predict_positive_probability(model, x_val)

        final_transformer = fit_table_transformer(train_df, columns)
        x_all = transform_table_features(train_df, final_transformer)
        final_model = fit_elastic_net_evidence_model(
            x_all,
            y,
            balanced_image_weights(train_df),
            random_state,
        )
        models[group_name] = FeatureGroupModel(
            group_name=group_name,
            feature_columns=columns,
            imputer=final_transformer.imputer,
            scaler=final_transformer.scaler,
            model=final_model,
            score_column=score_column,
        )
        evidence[score_column] = np.nan_to_num(oof, nan=np.nanmean(oof))

    return models, evidence


def predict_feature_group_scores(frame: pd.DataFrame, models: Mapping[str, FeatureGroupModel]) -> pd.DataFrame:
    """用已訓練 feature-group models 輸出 evidence scores。"""
    scores = pd.DataFrame(index=frame.index)
    for bundle in models.values():
        transformer = TableFeatureTransformer(
            feature_columns=bundle.feature_columns,
            imputer=bundle.imputer,
            scaler=bundle.scaler,
            selected_columns=bundle.feature_columns,
        )
        x_frame = transform_table_features(frame, transformer)
        scores[bundle.score_column] = predict_positive_probability(bundle.model, x_frame)
    return scores


def resolve_pc_image_path(source_folder: str, image_name: str, input_dir: Path = DEFAULT_INPUT_DIR) -> Path | None:
    """依 source_folder 與 Image 欄位尋找 PC 原圖。"""
    pc_dir = input_dir / str(source_folder) / "PC"
    if not pc_dir.exists():
        return None
    image_stem = Path(str(image_name)).stem
    candidates = [pc_dir / f"{image_stem}{ext}" for ext in IMAGE_EXTENSIONS]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    files = [p for p in pc_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    for path in files:
        if path.stem.endswith(image_stem) or image_stem.endswith(path.stem):
            return path
    for path in files:
        if image_stem in path.stem or path.stem in image_stem:
            return path
    return None


def list_input_pc_image_stems(source_folder: str, input_dir: Path = DEFAULT_INPUT_DIR) -> list[str]:
    """列出 data/input/<source_folder>/PC 內的原始 PC 影像 stem。"""
    pc_dir = input_dir / str(source_folder) / "PC"
    if not pc_dir.exists():
        return []
    return sorted(p.stem for p in pc_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def build_resnet18_extractor(pretrained: bool, random_state: int = 42) -> tuple[Any, Any, str]:
    """建立 ResNet18 image embedding extractor。

    Args:
        pretrained: 是否嘗試使用 torchvision 預訓練權重。
        random_state: 未使用預訓練權重時的初始化種子。

    Returns:
        tuple[Any, Any, str]: ``(model, preprocess, status)``。

    Raises:
        ImportError: 目前環境沒有 torch / torchvision / PIL。
    """
    import torch
    from PIL import Image  # noqa: F401
    from torchvision import models, transforms

    torch.manual_seed(int(random_state))
    status = "resnet18_random"
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            status = "resnet18_pretrained"
        except Exception:
            weights = None
            status = "resnet18_random"

    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess, status


def extract_image_embeddings(
    frame: pd.DataFrame,
    input_dir: Path = DEFAULT_INPUT_DIR,
    pretrained: bool = False,
    random_state: int = 42,
    batch_size: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """從 PC 原圖抽 ResNet18 image embeddings。

    Args:
        frame: 需包含 ``source_folder``、``Image``、``image_key`` 的資料。
        input_dir: ``data/input`` 路徑。
        pretrained: 是否使用 torchvision 預訓練權重。
        random_state: 隨機種子。
        batch_size: CNN 推論 batch size。

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: 以 ``image_key`` 為 index 的
        embedding table 與執行狀態。
    """
    try:
        import torch
        from PIL import Image
    except Exception as error:  # pylint: disable=broad-except
        return pd.DataFrame(index=[]), {"status": "missing_torch_or_pillow", "error": str(error)}

    try:
        model, preprocess, status = build_resnet18_extractor(pretrained, random_state)
    except Exception as error:  # pylint: disable=broad-except
        return pd.DataFrame(index=[]), {"status": "cnn_init_failed", "error": str(error)}

    unique_images = (
        frame[["image_key", "source_folder", "Image"]]
        .drop_duplicates("image_key")
        .sort_values("image_key")
        .reset_index(drop=True)
    )
    vectors: list[np.ndarray] = []
    keys: list[str] = []
    missing: list[str] = []
    tensors: list[Any] = []
    tensor_keys: list[str] = []

    def flush_batch() -> None:
        """抽取目前累積批次的 CNN embeddings。"""
        if not tensors:
            return
        with torch.no_grad():
            batch = torch.stack(tensors, dim=0)
            features = model(batch).detach().cpu().numpy().astype(np.float32)
        for key, vector in zip(tensor_keys, features):
            keys.append(key)
            vectors.append(vector)
        tensors.clear()
        tensor_keys.clear()

    for _, row in unique_images.iterrows():
        image_path = resolve_pc_image_path(str(row["source_folder"]), str(row["Image"]), input_dir)
        if image_path is None:
            missing.append(str(row["image_key"]))
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            tensors.append(preprocess(image))
            tensor_keys.append(str(row["image_key"]))
            if len(tensors) >= batch_size:
                flush_batch()
        except Exception:  # pylint: disable=broad-except
            missing.append(str(row["image_key"]))
    flush_batch()

    if vectors:
        matrix = np.vstack(vectors)
        columns = [f"cnn_emb_{i:04d}" for i in range(matrix.shape[1])]
        embedding_df = pd.DataFrame(matrix, index=keys, columns=columns)
    else:
        embedding_df = pd.DataFrame(index=[])
    meta = {
        "status": status,
        "images_total": int(len(unique_images)),
        "images_embedded": int(len(embedding_df)),
        "images_missing": int(len(missing)),
        "missing_image_keys": missing[:100],
    }
    return embedding_df, meta


def attach_image_embeddings(frame: pd.DataFrame, embedding_df: pd.DataFrame) -> pd.DataFrame:
    """將 image-level embedding 回填到 cell-level table。"""
    if embedding_df.empty:
        return pd.DataFrame(index=frame.index)
    return frame[["image_key"]].join(embedding_df, on="image_key").drop(columns=["image_key"]).fillna(0.0)


def fit_stage1_classifier_with_oof(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    random_state: int = 42,
    cv_splits: int = 5,
) -> tuple[LogisticRegression, np.ndarray]:
    """訓練 Stage 1 classifier 並產生 train OOF probability。

    Args:
        x_train: Stage 1 訓練矩陣。
        y_train: Cell-level Ki67 ground truth。
        groups: GroupKFold 使用的 group，通常為 source folder。
        random_state: 隨機種子。
        cv_splits: GroupKFold 切分數。

    Returns:
        tuple[LogisticRegression, np.ndarray]: 最終 fit 全訓練資料的模型與
        train out-of-fold positive probability。
    """
    y = y_train.astype(int).to_numpy()
    group_count = int(pd.Series(groups).nunique())
    n_splits = min(max(2, int(cv_splits)), group_count)
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(x_train), np.nan, dtype=np.float64)
    x_np = x_train.to_numpy(dtype=np.float32)

    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(x_np, y, groups=groups)):
        model = LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state + fold_idx,
        )
        model.fit(x_np[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(x_np[val_idx])[:, 1]

    final_model = LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    final_model.fit(x_np, y)
    return final_model, np.nan_to_num(oof, nan=np.nanmean(oof))


def aggregate_to_image_features(
    cell_df: pd.DataFrame,
    probability_col: str = "cell_prob",
    feature_score_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """將 cell-level probability 彙整成 image-level features。"""
    score_columns = list(feature_score_columns or [])
    work = cell_df.copy()
    grouped = work.groupby("image_key", as_index=False)
    image_df = grouped.agg(
        source_folder=("source_folder", "first"),
        image_name=("Image", "first"),
        passage=("passage", "first"),
        cell_count=(probability_col, "size"),
        mean_cell_prob=(probability_col, "mean"),
        median_cell_prob=(probability_col, "median"),
        p90_cell_prob=(probability_col, lambda x: float(np.quantile(x, 0.90))),
        frac_prob_gt_0_3=(probability_col, lambda x: float((np.asarray(x) > 0.30).mean())),
        frac_prob_gt_0_5=(probability_col, lambda x: float((np.asarray(x) > 0.50).mean())),
        frac_prob_gt_0_7=(probability_col, lambda x: float((np.asarray(x) > 0.70).mean())),
    )
    if "ki67_label" in work.columns:
        true_ratio = grouped["ki67_label"].mean().rename(columns={"ki67_label": "true_ratio"})
        image_df = image_df.merge(true_ratio, on="image_key", how="left")

    for score_col in score_columns:
        if score_col not in work.columns:
            continue
        score_stats = work.groupby("image_key")[score_col].agg(["mean", "median"]).reset_index()
        score_stats = score_stats.rename(
            columns={"mean": f"{score_col}_mean", "median": f"{score_col}_median"}
        )
        image_df = image_df.merge(score_stats, on="image_key", how="left")
    return image_df


def build_stage2_matrix(
    image_df: pd.DataFrame,
    mode: str,
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """依 Stage 2 模式建立 image-level design matrix。"""
    mode = mode.upper()
    if mode == "S0":
        return pd.DataFrame(index=image_df.index), []
    columns = list(PROBABILITY_SUMMARY_COLUMNS)
    if mode in {"S3", "S4"}:
        passage_dummies = pd.get_dummies(image_df["passage"].astype(str), prefix="passage", dtype=float)
    else:
        passage_dummies = pd.DataFrame(index=image_df.index)
    feature_summary = pd.DataFrame(index=image_df.index)
    if mode == "S4":
        feature_summary = image_df.filter(regex=r"_ki67_evidence_(mean|median)$").copy()
    base = image_df.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = pd.concat([base, passage_dummies, feature_summary], axis=1)
    if feature_columns is not None:
        for column in feature_columns:
            if column not in matrix.columns:
                matrix[column] = 0.0
        matrix = matrix.reindex(columns=list(feature_columns), fill_value=0.0)
    return matrix, list(matrix.columns)


def fit_stage2_ratio_model(train_image_df: pd.DataFrame, mode: str) -> tuple[Any, list[str]]:
    """訓練 Stage 2 image Ki67 ratio model。"""
    if mode.upper() == "S0":
        return None, []
    x_train, columns = build_stage2_matrix(train_image_df, mode)
    model = Ridge(alpha=1.0)
    model.fit(x_train, train_image_df["true_ratio"].to_numpy(dtype=np.float64))
    return model, columns


def predict_stage2_ratio(image_df: pd.DataFrame, model: Any, mode: str, columns: Sequence[str]) -> np.ndarray:
    """預測 image-level Ki67 positive rate。"""
    if mode.upper() == "S0":
        return np.clip(image_df["mean_cell_prob"].to_numpy(dtype=np.float64), 0.0, 1.0)
    x_df, _ = build_stage2_matrix(image_df, mode, columns)
    return np.clip(np.asarray(model.predict(x_df), dtype=np.float64), 0.0, 1.0)


def evaluate_cell_predictions(
    y_true: Sequence[int],
    probability: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """計算 cell-level accuracy 與 confusion matrix。"""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=np.float64)
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def evaluate_image_predictions(image_df: pd.DataFrame, pred_ratio: Sequence[float]) -> dict[str, float]:
    """計算 image-level Ki67 ratio 指標。"""
    true = image_df["true_ratio"].to_numpy(dtype=np.float64)
    pred = np.asarray(pred_ratio, dtype=np.float64)
    error = pred - true
    return {
        "image_mae": float(mean_absolute_error(true, pred)),
        "image_bias": float(np.mean(error)),
        "image_rmse": float(math.sqrt(np.mean(np.square(error)))),
        "within_10pp": float((np.abs(error) <= 0.10).mean()),
    }


def format_percent(value: float) -> str:
    """格式化百分比。"""
    if pd.isna(value):
        return "-"
    return f"{100.0 * float(value):.2f}%"


def format_pp(value: float) -> str:
    """格式化 percentage points。"""
    if pd.isna(value):
        return "-"
    return f"{100.0 * float(value):.2f} pp"


def save_model_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    """儲存新版 Ki67 model bundle。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), path)


def load_model_bundle(path: Path) -> dict[str, Any]:
    """載入新版 Ki67 model bundle。"""
    return joblib.load(path)


def render_summary_html(
    output_path: Path,
    metrics_df: pd.DataFrame,
    split_df: pd.DataFrame,
    best_key: str,
) -> None:
    """輸出新版訓練 summary.html。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = metrics_df[metrics_df["config_key"] == best_key].iloc[0]
    rows = []
    for _, row in metrics_df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['display_name']))}</td>"
            f"<td>{html.escape(str(row['stage1_input']))}</td>"
            f"<td>{html.escape(str(row['stage2_input']))}</td>"
            f"<td>{format_percent(row['train_cell_accuracy'])}</td>"
            f"<td>{format_percent(row['valid_cell_accuracy'])}</td>"
            f"<td>{format_percent(row['test_cell_accuracy'])}</td>"
            f"<td>{format_pp(row['train_image_mae'])}</td>"
            f"<td>{format_pp(row['valid_image_mae'])}</td>"
            f"<td>{format_pp(row['test_image_mae'])}</td>"
            "</tr>"
        )
    split_rows = []
    for _, row in split_df.iterrows():
        split_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['split']))}</td>"
            f"<td>{int(row['source_folders'])}</td>"
            f"<td>{int(row['images'])}</td>"
            f"<td>{int(row['cells'])}</td>"
            f"<td>{format_percent(row['positive_ratio'])}</td>"
            f"<td>{html.escape(str(row['source_folder_list']))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>Ki67 新版 Stage 1 / Stage 2 訓練摘要</title>
<style>
body {{ font-family: "Segoe UI", "Noto Sans TC", sans-serif; margin: 28px; color: #1f2937; }}
h1 {{ margin-bottom: 8px; }}
h2 {{ margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 8px; vertical-align: top; }}
th {{ background: #f3f4f6; }}
.note {{ color: #4b5563; line-height: 1.6; }}
.metric {{ display:inline-block; margin-right:20px; padding:10px 14px; background:#eff6ff; border-radius:8px; }}
</style>
</head>
<body>
<h1>Ki67 新版 Stage 1 / Stage 2 訓練摘要</h1>
<p class="note">最佳組合：<strong>{html.escape(str(best['display_name']))}</strong>；
test image Ki67 MAE = <strong>{format_pp(best['test_image_mae'])}</strong>；
test cell accuracy = <strong>{format_percent(best['test_cell_accuracy'])}</strong>。</p>
<div class="metric">Train images: {int(split_df.loc[split_df['split']=='train','images'].iloc[0])}</div>
<div class="metric">Valid images: {int(split_df.loc[split_df['split']=='valid','images'].iloc[0])}</div>
<div class="metric">Test images: {int(split_df.loc[split_df['split']=='test','images'].iloc[0])}</div>
<h2>Train / Valid / Test 資料數量</h2>
<table><thead><tr><th>切分</th><th>來源資料夾數</th><th>影像數</th><th>細胞數</th><th>Ki67+ 比例</th><th>來源資料夾</th></tr></thead>
<tbody>{''.join(split_rows)}</tbody></table>
<h2>模型比較</h2>
<table><thead><tr><th>實驗組合</th><th>Stage 1 輸入</th><th>Stage 2 輸入</th><th>Train cell acc.</th><th>Valid cell acc.</th><th>Test cell acc.</th><th>Train image MAE</th><th>Valid image MAE</th><th>Test image MAE</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>說明</h2>
<p class="note">Feature 代表指定 feature-group Ki67 evidence scores；各 feature group 會先以 elastic-net logistic classifier 轉為 cell-level Ki67 evidence score。若該組合啟用 CNN embedding，則由每張 PC 原圖經 ResNet18 轉為一維 image embedding，再回填到同影像內所有 cell。Stage 2 S1 使用 probability summary 預測每張影像 Ki67 positive rate。</p>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def default_experiment_configs() -> list[ExperimentConfig]:
    """回傳新版預設實驗組合。"""
    standard_feature_groups = ("Texture", "Intensity distribution", "Halo / rounding")
    report_feature_groups = ("Texture", "Attachment / spreading")
    return [
        ExperimentConfig(
            key="texture_attachment_s1",
            display_name="Texture + Attachment / S1 no embedding",
            stage1_input_name="Texture + Attachment feature-group evidence scores",
            stage2_name="S1: probability summary",
            include_feature_scores=True,
            include_cnn_embedding=False,
            parameter_mode="none",
            stage2_mode="S1",
            feature_score_groups=report_feature_groups,
        ),
        ExperimentConfig(
            key="feature_cnn_s0",
            display_name="Feature + CNN embedding / S0 direct mean",
            stage1_input_name="Feature-group evidence scores + ResNet18 CNN embedding",
            stage2_name="S0: direct mean cell probability",
            include_feature_scores=True,
            include_cnn_embedding=True,
            parameter_mode="none",
            stage2_mode="S0",
            feature_score_groups=standard_feature_groups,
        ),
        ExperimentConfig(
            key="feature_cnn_s1",
            display_name="Feature + CNN embedding / S1 probability summary",
            stage1_input_name="Feature-group evidence scores + ResNet18 CNN embedding",
            stage2_name="S1: probability summary",
            include_feature_scores=True,
            include_cnn_embedding=True,
            parameter_mode="none",
            stage2_mode="S1",
            feature_score_groups=standard_feature_groups,
        ),
        ExperimentConfig(
            key="feature_cnn_s3",
            display_name="Feature + CNN embedding / S3 probability + passage",
            stage1_input_name="Feature-group evidence scores + ResNet18 CNN embedding",
            stage2_name="S3: probability summary + passage",
            include_feature_scores=True,
            include_cnn_embedding=True,
            parameter_mode="none",
            stage2_mode="S3",
            feature_score_groups=standard_feature_groups,
        ),
        ExperimentConfig(
            key="feature_cnn_selected_s1",
            display_name="Feature + CNN embedding + selected parameters / S1",
            stage1_input_name="Feature-group evidence scores + ResNet18 CNN embedding + selected parameters",
            stage2_name="S1: probability summary",
            include_feature_scores=True,
            include_cnn_embedding=True,
            parameter_mode="selected",
            stage2_mode="S1",
            feature_score_groups=standard_feature_groups,
        ),
        ExperimentConfig(
            key="feature_cnn_selected_s4",
            display_name="Feature + CNN embedding + selected parameters / S4",
            stage1_input_name="Feature-group evidence scores + ResNet18 CNN embedding + selected parameters",
            stage2_name="S4: probability summary + Feature summary + passage",
            include_feature_scores=True,
            include_cnn_embedding=True,
            parameter_mode="selected",
            stage2_mode="S4",
            feature_score_groups=standard_feature_groups,
        ),
        ExperimentConfig(
            key="feature_only_s1",
            display_name="Feature only / S1 baseline",
            stage1_input_name="Feature-group evidence scores only",
            stage2_name="S1: probability summary",
            include_feature_scores=True,
            include_cnn_embedding=False,
            parameter_mode="none",
            stage2_mode="S1",
            feature_score_groups=standard_feature_groups,
        ),
        ExperimentConfig(
            key="cnn_only_s1",
            display_name="CNN embedding only / S1 baseline",
            stage1_input_name="ResNet18 CNN embedding only",
            stage2_name="S1: probability summary",
            include_feature_scores=False,
            include_cnn_embedding=True,
            parameter_mode="none",
            stage2_mode="S1",
        ),
        ExperimentConfig(
            key="parameters_only_s1",
            display_name="Feature parameters only / S1 baseline",
            stage1_input_name="All feature parameters only",
            stage2_name="S1: probability summary",
            include_feature_scores=False,
            include_cnn_embedding=False,
            parameter_mode="all",
            stage2_mode="S1",
        ),
    ]
