"""定義 Exp3 phase-only 特徵集合與 FOV 彙整規則。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


BASIC_GEOMETRY = (
    "area",
    "compactness",
    "eccentricity",
    "extent",
    "sphericity",
    "major_axis_length",
    "feret_length",
    "minor_axis_length",
    "feret_width",
    "maximum_radius",
    "mean_radius",
    "median_radius",
    "aspect_ratio",
    "perimeter_area_ratio",
    "perimeter",
    "solidity",
)
PRIMARY_CELL_FEATURES = (
    *(f"cell__{name}" for name in BASIC_GEOMETRY),
    *(f"nucleus__{name}" for name in BASIC_GEOMETRY),
    "nucleus_cytoplasm_area_ratio",
)
PRIMARY_FOV_FEATURES = tuple(
    f"{name}__median" for name in PRIMARY_CELL_FEATURES
)
FEATURE_SET_REGISTRIES = {
    "basic_median": PRIMARY_FOV_FEATURES,
}

FOV_METADATA_COLUMNS = (
    "image_key",
    "b_id",
    "passage",
    "group_id",
    "condition_index",
    "condition",
    "ifn_dose",
    "tnf_dose",
    "fov",
)
_FORBIDDEN_PREDICTOR_TOKENS = (
    "ido",
    "ifn",
    "tnf",
    "dose",
    "condition",
    "path",
    "filename",
    "delta",
)


def validate_phase_predictors(columns: Sequence[str]) -> None:
    """驗證 predictor 僅包含已註冊的 phase morphology 欄位。

    Args:
        columns: 待驗證的 image-level predictor 欄位名稱。

    Raises:
        ValueError: 欄位含 target、實驗條件、路徑、delta 或未註冊特徵時拋出。
    """
    registered = {
        column
        for registry in FEATURE_SET_REGISTRIES.values()
        for column in registry
    }
    invalid = []
    for column in columns:
        name = str(column)
        lowered = name.lower()
        if (
            any(token in lowered for token in _FORBIDDEN_PREDICTOR_TOKENS)
            or name not in registered
        ):
            invalid.append(name)
    if invalid:
        raise ValueError(
            "predictor 必須是已註冊的 phase-only morphology 欄位："
            f"{invalid}"
        )


def aggregate_fov_features(
    cells: pd.DataFrame,
    manifest: pd.DataFrame,
    feature_set_names: Sequence[str] = ("basic_median",),
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """以中位數將 cell-level morphology 彙整成 FOV-level 特徵。

    Args:
        cells: 含 Primary morphology 與 ``IDO_score`` 的 cell-level 資料。
        manifest: 一圖一列且含實驗 metadata 的 Exp3 manifest。
        feature_set_names: 要啟用的 feature-set 名稱；Primary 預設為
            ``("basic_median",)``。

    Returns:
        FOV-level 資料表，以及各 feature set 對應的 predictor 欄位清單。

    Raises:
        ValueError: feature set 未註冊、欄位缺漏、image key 重複、細胞數不足，
            或 target 非有限值或無變異時拋出。
    """
    selected_names = tuple(str(name) for name in feature_set_names)
    unknown = [
        name for name in selected_names if name not in FEATURE_SET_REGISTRIES
    ]
    if unknown:
        raise ValueError(f"未註冊的 phase-only feature set：{unknown}")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("feature_set_names 不可重複")

    registries = {
        name: list(FEATURE_SET_REGISTRIES[name]) for name in selected_names
    }
    predictor_columns = [
        column for name in selected_names for column in registries[name]
    ]
    validate_phase_predictors(predictor_columns)
    base_features = [
        column.removesuffix("__median") for column in predictor_columns
    ]
    required_cell_columns = {"image_key", "IDO_score", *base_features}
    missing_cell_columns = sorted(required_cell_columns - set(cells.columns))
    if missing_cell_columns:
        raise ValueError(
            f"cell-level 資料缺少 Primary 欄位：{missing_cell_columns}"
        )
    missing_manifest_columns = sorted(
        set(FOV_METADATA_COLUMNS) - set(manifest.columns)
    )
    if missing_manifest_columns:
        raise ValueError(f"manifest 缺少欄位：{missing_manifest_columns}")
    if manifest["image_key"].duplicated().any():
        raise ValueError("manifest 出現重複 image_key")
    expected_keys = manifest["image_key"].astype(str).tolist()
    unexpected_keys = sorted(
        set(cells["image_key"].astype(str)) - set(expected_keys)
    )
    if unexpected_keys:
        raise ValueError(f"cell-level 資料含 manifest 以外的 image_key：{unexpected_keys}")

    min_cells = int(
        cells.attrs.get(
            "min_cells_per_image",
            manifest.attrs.get("min_cells_per_image", 3),
        )
    )
    if min_cells < 1:
        raise ValueError("min_cells_per_image 必須至少為 1")
    cache_dir_value = cells.attrs.get("feature_cache_dir")
    grouped = {
        str(image_key): group
        for image_key, group in cells.groupby("image_key", sort=False)
    }
    rows = []
    insufficient = []
    for metadata in manifest.loc[:, FOV_METADATA_COLUMNS].to_dict(orient="records"):
        image_key = str(metadata["image_key"])
        group = grouped.get(image_key)
        cell_count = 0 if group is None else len(group)
        if cell_count < min_cells:
            insufficient.append(image_key)
            continue
        assert group is not None
        row = {
            **metadata,
            "image_key": image_key,
            "cell_count": int(cell_count),
            "IDO_score": float(group["IDO_score"].median()),
        }
        for feature, output_column in zip(base_features, predictor_columns):
            row[output_column] = float(group[feature].median())
        rows.append(row)
    if insufficient:
        raise ValueError(
            f"每張 FOV 至少 {min_cells} 顆有效細胞；不足者：{insufficient}"
        )

    output_columns = [
        *FOV_METADATA_COLUMNS,
        "cell_count",
        "IDO_score",
        *predictor_columns,
    ]
    images = pd.DataFrame(rows, columns=output_columns)
    if not np.isfinite(images["IDO_score"].to_numpy(dtype=float)).all():
        raise ValueError("FOV-level IDO_score 必須全部為有限值")
    if images["IDO_score"].nunique(dropna=True) < 2:
        raise ValueError("FOV-level IDO_score 至少需要兩個不同值")
    if cache_dir_value:
        cache_dir = Path(str(cache_dir_value))
        cache_dir.mkdir(parents=True, exist_ok=True)
        images.to_csv(cache_dir / "image_level_basic.csv", index=False)
    return images, registries


__all__ = [
    "BASIC_GEOMETRY",
    "FEATURE_SET_REGISTRIES",
    "PRIMARY_CELL_FEATURES",
    "PRIMARY_FOV_FEATURES",
    "aggregate_fov_features",
    "validate_phase_predictors",
]
