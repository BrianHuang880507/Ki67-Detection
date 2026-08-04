"""定義 Exp3 phase-only 特徵集合與 FOV 彙整規則。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from immunity.exp3.run_benchmark import resolve_exp3_output_dir


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
PAPER_STYLE_EXTRA_FEATURES = (
    *(f"cell__zernike_{index:02d}" for index in range(25)),
    *(f"nucleus__zernike_{index:02d}" for index in range(25)),
    "nucleus_cytoplasm_mean_ratio",
    "nucleus_cytoplasm_intden_ratio",
    "nucleus_cytoplasm_raw_intden_ratio",
    "nucleus_cell_intden_ratio",
    "nucleus_cytoplasm_entropy_difference",
    "nucleus_cytoplasm_cv_difference",
    "nucleus_centroid_offset",
    "cell__phase_intensity_cv",
    "nucleus__phase_intensity_cv",
    "cytoplasm__phase_intensity_cv",
)
PAPER_STYLE_CELL_FEATURES = (
    *PRIMARY_CELL_FEATURES,
    *PAPER_STYLE_EXTRA_FEATURES,
)
BASIC_MEDIAN_IQR_FOV_FEATURES = (
    *PRIMARY_FOV_FEATURES,
    *(f"{name}__iqr" for name in PRIMARY_CELL_FEATURES),
)
PAPER_STYLE_FOV_FEATURES = tuple(
    f"{name}__median" for name in PAPER_STYLE_CELL_FEATURES
)


@dataclass(frozen=True)
class FeatureSetSpec:
    """描述單一 Exp3 phase-only feature set。

    Attributes:
        name: feature set 的穩定識別名稱。
        cell_features: 彙整前的 cell-level morphology 欄位。
        predictor_columns: 供模型使用的 FOV-level predictor 欄位。
        aggregation: cell-to-FOV 的彙整規則。
        replication_scope: 此集合相對於參考研究的重現範圍。
    """

    name: str
    cell_features: tuple[str, ...]
    predictor_columns: tuple[str, ...]
    aggregation: str
    replication_scope: str

    @property
    def predictor_count(self) -> int:
        """回傳此 feature set 的 FOV predictor 數量。"""
        return len(self.predictor_columns)


FEATURE_SET_REGISTRY: dict[str, FeatureSetSpec] = {
    "basic_median": FeatureSetSpec(
        name="basic_median",
        cell_features=PRIMARY_CELL_FEATURES,
        predictor_columns=PRIMARY_FOV_FEATURES,
        aggregation="median",
        replication_scope="primary_phase_only",
    ),
    "basic_median_iqr": FeatureSetSpec(
        name="basic_median_iqr",
        cell_features=PRIMARY_CELL_FEATURES,
        predictor_columns=BASIC_MEDIAN_IQR_FOV_FEATURES,
        aggregation="median_iqr",
        replication_scope="secondary_phase_only",
    ),
    "paper_style_median": FeatureSetSpec(
        name="paper_style_median",
        cell_features=PAPER_STYLE_CELL_FEATURES,
        predictor_columns=PAPER_STYLE_FOV_FEATURES,
        aggregation="median",
        replication_scope="partial_label_free_approximation",
    ),
}
FEATURE_SET_REGISTRIES = {
    name: spec.predictor_columns for name, spec in FEATURE_SET_REGISTRY.items()
}

CONTRASTS = (
    ("IFN25_vs_0_at_TNF0", (0.0, 0.0), (25.0, 0.0)),
    ("IFN50_vs_0_at_TNF0", (0.0, 0.0), (50.0, 0.0)),
    ("IFN100_vs_0_at_TNF0", (0.0, 0.0), (100.0, 0.0)),
    ("TNF25_vs_0_at_IFN0", (0.0, 0.0), (0.0, 25.0)),
    ("TNF50_vs_0_at_IFN0", (0.0, 0.0), (0.0, 50.0)),
    ("TNF25_vs_0_at_IFN25", (25.0, 0.0), (25.0, 25.0)),
    ("TNF50_vs_0_at_IFN25", (25.0, 0.0), (25.0, 50.0)),
)

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
        for spec in FEATURE_SET_REGISTRY.values()
        for column in spec.predictor_columns
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


def calculate_delta_signatures(
    images: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """計算每個實驗群組的七組描述性、非配對形態差異。

    Args:
        images: 含 ``group_id``、IFN/TNF dose 與 FOV-level features 的資料。
        feature_columns: 要計算 contrast 的已註冊 phase-only predictors。

    Returns:
        每個群組、contrast 與 feature 一列的長格式描述表。

    Raises:
        ValueError: 輸入為空、欄位或 ``group_id`` 不合法、任一群組缺少預定條件，
            或 delta/global-IQR scaling 無法得到有限值時拋出。
    """
    features = tuple(str(column) for column in feature_columns)
    if not features:
        raise ValueError("feature_columns 不可為空")
    if len(set(features)) != len(features):
        raise ValueError("feature_columns 不可重複")
    validate_phase_predictors(features)
    required = {"group_id", "ifn_dose", "tnf_dose", *features}
    missing = sorted(required - set(images.columns))
    if missing:
        raise ValueError(f"ΔMorphology 輸入缺少欄位：{missing}")
    if images.empty:
        raise ValueError("ΔMorphology images 不可為空")
    group_ids = images["group_id"].astype("string")
    if (group_ids.isna() | group_ids.fillna("").str.strip().eq("")).any():
        raise ValueError("ΔMorphology group_id 不可為 null 或空白")

    global_iqrs = {
        feature: _finite_iqr(images[feature].to_numpy(dtype=float))
        for feature in features
    }
    for feature, global_iqr in global_iqrs.items():
        if not np.isfinite(global_iqr):
            raise ValueError(
                f"ΔMorphology global IQR 必須是有限值；feature={feature!r}"
            )
    rows: list[dict[str, object]] = []
    for group_id, group in images.groupby("group_id", sort=False):
        condition_rows = {
            (ifn, tnf): group.loc[
                np.isclose(group["ifn_dose"].to_numpy(dtype=float), ifn)
                & np.isclose(group["tnf_dose"].to_numpy(dtype=float), tnf)
            ]
            for _, control, treated in CONTRASTS
            for ifn, tnf in (control, treated)
        }
        missing_conditions = [
            condition
            for condition, subset in condition_rows.items()
            if subset.empty
        ]
        if missing_conditions:
            raise ValueError(
                f"group_id {group_id!r} 缺少 ΔMorphology 條件："
                f"{missing_conditions}"
            )
        for contrast_id, control, treated in CONTRASTS:
            for feature in features:
                control_median = float(condition_rows[control][feature].median())
                treated_median = float(condition_rows[treated][feature].median())
                if not np.isfinite(control_median) or not np.isfinite(treated_median):
                    raise ValueError(
                        f"group_id {group_id!r} 的 {feature!r} contrast median "
                        "必須是有限值"
                    )
                delta_raw = treated_median - control_median
                if not np.isfinite(delta_raw):
                    raise ValueError(
                        f"group_id {group_id!r} 的 {feature!r} delta 必須是有限值"
                    )
                global_iqr = global_iqrs[feature]
                if global_iqr == 0.0:
                    if delta_raw != 0.0:
                        raise ValueError(
                            "ΔMorphology global IQR 為 0，"
                            f"feature {feature!r} 的非零 delta 無法縮放"
                        )
                    scaled_delta = 0.0
                else:
                    scaled_delta = delta_raw / global_iqr
                    if not np.isfinite(scaled_delta):
                        raise ValueError(
                            f"group_id {group_id!r} 的 {feature!r} "
                            "scaled delta 必須是有限值"
                        )
                rows.append(
                    {
                        "group_id": str(group_id),
                        "contrast_id": contrast_id,
                        "feature": feature,
                        "control_median": control_median,
                        "treated_median": treated_median,
                        "delta_raw": delta_raw,
                        "global_iqr": global_iqr,
                        "delta_scaled_by_global_iqr": scaled_delta,
                        "comparison_type": "group_level_unpaired",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=(
            "group_id",
            "contrast_id",
            "feature",
            "control_median",
            "treated_median",
            "delta_raw",
            "global_iqr",
            "delta_scaled_by_global_iqr",
            "comparison_type",
        ),
    )


def _finite_iqr(values: np.ndarray) -> float:
    """計算有限值的 75th minus 25th percentile。"""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    lower, upper = np.percentile(finite, [25.0, 75.0])
    return float(upper - lower)


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
    selected_names = _validate_feature_set_specs(feature_set_names)
    selected_specs = [FEATURE_SET_REGISTRY[name] for name in selected_names]
    registries = {
        spec.name: list(spec.predictor_columns) for spec in selected_specs
    }
    predictor_columns = list(
        dict.fromkeys(
            column
            for spec in selected_specs
            for column in spec.predictor_columns
        )
    )
    validate_phase_predictors(predictor_columns)
    required_cell_columns = {
        "image_key",
        "IDO_score",
        *(feature for spec in selected_specs for feature in spec.cell_features),
    }
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
        for spec in selected_specs:
            for feature in spec.cell_features:
                row[f"{feature}__median"] = float(group[feature].median())
                if spec.aggregation == "median_iqr":
                    row[f"{feature}__iqr"] = _finite_iqr(
                        group[feature].to_numpy(dtype=float)
                    )
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
    nonfinite_predictors = [
        column
        for column in predictor_columns
        if not np.isfinite(images[column].to_numpy(dtype=float)).all()
    ]
    if nonfinite_predictors:
        raise ValueError(
            "FOV-level phase-only predictors 必須全部為有限值："
            f"{nonfinite_predictors}"
        )
    cache_dir = _feature_cache_dir_from_output(
        cells.attrs.get("exp3_output_dir")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_names = {
        "basic_median": "image_level_basic.csv",
        "basic_median_iqr": "image_level_basic_median_iqr.csv",
        "paper_style_median": "image_level_paper_style_median.csv",
    }
    leading_columns = [*FOV_METADATA_COLUMNS, "cell_count", "IDO_score"]
    for spec in selected_specs:
        cache_path = _safe_output_file(cache_dir, cache_names[spec.name])
        _write_csv_atomically(
            images.loc[:, [*leading_columns, *spec.predictor_columns]],
            cache_path,
        )
    metadata = {
        spec.name: {
            "predictor_columns": list(spec.predictor_columns),
            "predictor_count": spec.predictor_count,
            "aggregation": spec.aggregation,
            "replication_scope": spec.replication_scope,
        }
        for spec in selected_specs
    }
    metadata_path = _safe_output_file(cache_dir.parent, "feature_sets.json")
    _write_json_atomically(metadata, metadata_path)
    return images, registries


def _validate_feature_set_specs(feature_set_names: Sequence[str]) -> tuple[str, ...]:
    """驗證啟用集合的名稱、固定數量與欄位唯一性。"""
    selected_names = tuple(str(name) for name in feature_set_names)
    if not selected_names:
        raise ValueError("feature_set_names 不可為空")
    unknown = [name for name in selected_names if name not in FEATURE_SET_REGISTRY]
    if unknown:
        raise ValueError(f"未註冊的 phase-only feature set：{unknown}")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("feature_set_names 不可重複")

    expected_counts = {
        "basic_median": 33,
        "basic_median_iqr": 66,
        "paper_style_median": 93,
    }
    for name in selected_names:
        spec = FEATURE_SET_REGISTRY[name]
        if spec.name != name:
            raise ValueError(f"feature set {name!r} 的 spec name 不一致")
        if spec.predictor_count != expected_counts[name]:
            raise ValueError(
                f"feature set {name!r} predictor count 必須是 "
                f"{expected_counts[name]}，實際為 {spec.predictor_count}"
            )
        if len(set(spec.predictor_columns)) != spec.predictor_count:
            raise ValueError(f"feature set {name!r} 含重複 predictor 名稱")
        if len(set(spec.cell_features)) != len(spec.cell_features):
            raise ValueError(f"feature set {name!r} 含重複 cell feature 名稱")
    return selected_names


def _feature_cache_dir_from_output(output_dir_value: object) -> Path:
    """從已驗證的 Exp3 output dir 衍生唯一 feature cache 位置。"""
    if not isinstance(output_dir_value, (str, Path)):
        raise ValueError("_output_dir 必須是非空路徑")
    if isinstance(output_dir_value, str) and not output_dir_value.strip():
        raise ValueError("_output_dir 必須是非空路徑")
    output_dir = resolve_exp3_output_dir(output_dir_value)
    candidate = output_dir / "feature_cache"
    resolved = candidate.resolve(strict=False)
    if resolved != candidate:
        raise ValueError("feature_cache 不可透過 symlink 或 junction 離開預定位置")
    return resolved


def _safe_output_file(parent: Path, name: str) -> Path:
    """驗證單一輸出檔未經 symlink 或 junction 改寫目的地。"""
    parent_resolved = parent.resolve(strict=False)
    candidate = parent_resolved / name
    resolved = candidate.resolve(strict=False)
    if resolved != candidate:
        raise ValueError(f"輸出檔 {name!r} 不可透過 symlink 離開預定位置")
    return resolved


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    """在已驗證的目錄內原子寫入 CSV。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomically(payload: object, path: Path) -> None:
    """在已驗證的 Exp3 run 目錄內原子寫入 JSON。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


__all__ = [
    "BASIC_GEOMETRY",
    "BASIC_MEDIAN_IQR_FOV_FEATURES",
    "CONTRASTS",
    "FEATURE_SET_REGISTRY",
    "FEATURE_SET_REGISTRIES",
    "FeatureSetSpec",
    "PAPER_STYLE_CELL_FEATURES",
    "PAPER_STYLE_EXTRA_FEATURES",
    "PAPER_STYLE_FOV_FEATURES",
    "PRIMARY_CELL_FEATURES",
    "PRIMARY_FOV_FEATURES",
    "aggregate_fov_features",
    "calculate_delta_signatures",
    "validate_phase_predictors",
]
