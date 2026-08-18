"""Exp4 Rui-style whole-cell 49 特徵擷取。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import platform
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping

import mahotas
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import regionprops
from skimage.restoration import rolling_ball

from immunity.exp4.border_exclusion import find_border_labels
from immunity.exp4.cell_dedup import assert_full_rui49_feature_consistency
from immunity.exp4.feature_health import (
    FeatureHealthResult,
    evaluate_feature_health,
)
from ki67dtc.cell_anal import (
    _geometry_from_measurements,
    _measure_roi_with_python,
    _quantize_to_levels,
)


BACKGROUND_RADIUS = 50

MORPHOLOGY_FEATURE_NAMES = (
    "Area",
    "BoundingBoxArea",
    "BoundingBoxMinimum_X",
    "BoundingBoxMinimum_Y",
    "BoundingBoxMaximum_X",
    "BoundingBoxMaximum_Y",
    "Center_X",
    "Center_Y",
    "Compactness",
    "ConvexArea",
    "Eccentricity",
    "EquivalentDiameter",
    "Extent",
    "FormFactor",
    "MajorAxisLength",
    "MaxFeretDiameter",
    "MaximumRadius",
    "MeanRadius",
    "MedianRadius",
    "MinFeretDiameter",
    "MinorAxisLength",
    "Orientation",
    "Perimeter",
    "Solidity",
)

INTENSITY_FEATURE_NAMES = (
    "IntegratedIntensity",
    "MeanIntensity",
    "StdIntensity",
    "MinIntensity",
    "MaxIntensity",
    "IntegratedIntensityEdge",
    "MeanIntensityEdge",
    "StdIntensityEdge",
    "MinIntensityEdge",
    "MaxIntensityEdge",
    "MassDisplacement",
    "MADIntensity",
)

HARALICK_FEATURE_NAMES = (
    "AngularSecondMoment",
    "Contrast",
    "Correlation",
    "Variance",
    "InverseDifferenceMoment",
    "SumAverage",
    "SumVariance",
    "SumEntropy",
    "Entropy",
    "DifferenceVariance",
    "DifferenceEntropy",
    "InfoMeas1",
    "InfoMeas2",
)

RUI49_FEATURE_COLUMNS = tuple(
    f"cell__{name}"
    for name in (
        *MORPHOLOGY_FEATURE_NAMES,
        *INTENSITY_FEATURE_NAMES,
        *HARALICK_FEATURE_NAMES,
    )
)

FALLBACK_SCOPE_PRE_BORDER = "unique_whole_cell_labels_before_border_exclusion"
SMOKE_HEALTH_SCOPE_POST_BORDER = (
    "smoke_master_labels_after_border_exclusion"
)
TEXTURE_SCOPE_ALL_MASK_LABELS = "all_positive_mask_labels"
HEALTH_SCOPE_MASTER_POST_BORDER = "master_labels_after_border_exclusion"


class SmokeFeatureHealthError(ValueError):
    """表示 provisional smoke feature-health gate 已完整落盤後阻擋。"""


@dataclass(frozen=True)
class RuiFeatureExtractionResult:
    """保存 Rui 49 特徵、legacy texture evidence 與 fallback diagnostics。

    Attributes:
        features: 正式 v2 特徵；texture 的 0 僅代表 mask 外 padding。
        legacy_texture_features: 可選的 v1 texture evidence，不得進入正式 X。
        fallback_flags: 每個 ``cell_label × Rui feature`` 的 fallback 布林旗標。
        cell_count: 本次擷取的 positive whole-cell label 數。
        fallback_scope: fallback denominator 所涵蓋的 cell 範圍。
    """

    features: pd.DataFrame
    legacy_texture_features: pd.DataFrame | None
    fallback_flags: pd.DataFrame
    cell_count: int
    fallback_scope: str = FALLBACK_SCOPE_PRE_BORDER

    @property
    def fallback_counts(self) -> dict[str, int]:
        """由 per-label flags 派生逐欄 fallback 次數。"""
        return {
            column: int(self.fallback_flags[column].sum())
            for column in RUI49_FEATURE_COLUMNS
        }

    @property
    def fallback_rates(self) -> dict[str, float]:
        """回傳以本次 eligible cells 為分母的逐欄 fallback rate。"""
        if self.cell_count == 0:
            return {column: 0.0 for column in RUI49_FEATURE_COLUMNS}
        return {
            column: count / self.cell_count
            for column, count in self.fallback_counts.items()
        }


def preprocess_phase_image(rgb_image: np.ndarray) -> np.ndarray:
    """將 RGB phase 影像轉為 luminance 並執行 rolling-ball 背景扣除。

    Args:
        rgb_image: 高、寬、三色版的 RGB 影像；整數與浮點影像皆可。

    Returns:
        非負、有限的二維 ``float64`` 背景扣除影像。

    Raises:
        ValueError: 影像不是 RGB、含非有限值，或背景扣除結果不合法時拋出。
    """
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("phase image 必須是 H×W×3 RGB 陣列")
    luminance = np.asarray(rgb2gray(image), dtype=np.float64)
    if not np.isfinite(luminance).all():
        raise ValueError("phase image 含非有限像素")
    background = rolling_ball(luminance, radius=BACKGROUND_RADIUS)
    corrected = np.clip(luminance - background, 0.0, None).astype(
        np.float64, copy=False
    )
    if not np.isfinite(corrected).all():
        raise ValueError("rolling-ball 背景扣除產生非有限像素")
    return corrected


def quantize_masked_texture(
    signal: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """量化 masked texture，將 0 專門保留給 mask 外 padding。

    Args:
        signal: 二維、有限的背景扣除 phase 訊號。
        mask: 與訊號同尺寸的二維 whole-cell 布林遮罩。

    Returns:
        ``uint8`` 量化影像；mask 外固定為 0，mask 內單調映射至 1..255。

    Raises:
        ValueError: 輸入維度、尺寸、有限性或非空 mask 不符合要求時拋出。
    """
    values = np.asarray(signal, dtype=np.float64)
    cell_mask = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or cell_mask.ndim != 2:
        raise ValueError("texture signal 與 mask 都必須是二維陣列")
    if values.shape != cell_mask.shape:
        raise ValueError("texture signal 與 mask 必須具有相同尺寸")
    if not np.isfinite(values).all():
        raise ValueError("texture signal 含非有限像素")
    if not np.any(cell_mask):
        raise ValueError("texture mask 不可為空")

    inside = _quantize_to_levels(values[cell_mask], 255).astype(np.uint16) + 1
    quantized = np.zeros(values.shape, dtype=np.uint8)
    quantized[cell_mask] = inside.astype(np.uint8)
    return quantized


def extract_rui49_features(
    phase_signal: np.ndarray,
    cell_labels: np.ndarray,
) -> pd.DataFrame:
    """擷取每個 positive whole-cell label 的 49 個 Rui-style 特徵。

    Args:
        phase_signal: 已完成背景扣除的二維、非負 phase luminance。
        cell_labels: 與 phase 同尺寸的二維 whole-cell label mask。

    Returns:
        每個 positive label 恰一列的資料表；首欄為 ``cell_label``，後續
        49 欄依 ``RUI49_FEATURE_COLUMNS`` 固定排序。

    Raises:
        ValueError: 輸入維度、尺寸或像素值不符合 whole-cell contract 時拋出。
    """
    return extract_rui49_features_with_diagnostics(
        phase_signal,
        cell_labels,
    ).features


def extract_rui49_features_with_diagnostics(
    phase_signal: np.ndarray,
    cell_labels: np.ndarray,
    *,
    include_legacy_texture: bool = False,
) -> RuiFeatureExtractionResult:
    """擷取 Rui 49 特徵並提供 fallback 與可選 legacy texture evidence。

    Args:
        phase_signal: 已完成背景扣除的二維、非負 phase luminance。
        cell_labels: 與 phase 同尺寸的二維 whole-cell label mask。
        include_legacy_texture: 是否在同一次 cell iteration 計算 v1 texture evidence。

    Returns:
        正式 features、逐欄 fallback diagnostics 與可選 legacy texture。

    Raises:
        ValueError: 輸入或輸出違反 Rui whole-cell feature contract 時拋出。
    """
    signal = np.asarray(phase_signal, dtype=np.float64)
    labels = np.asarray(cell_labels)
    if signal.ndim != 2 or labels.ndim != 2:
        raise ValueError("phase_signal 與 cell_labels 都必須是二維陣列")
    if signal.shape != labels.shape:
        raise ValueError("phase_signal 與 cell_labels 必須具有相同尺寸")
    if not np.isfinite(signal).all():
        raise ValueError("phase_signal 含非有限像素")
    if np.any(signal < 0.0):
        raise ValueError("phase_signal 必須是背景扣除後的非負影像")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("cell_labels 必須是整數 label mask")
    if np.any(labels < 0):
        raise ValueError("cell_labels 不可含負數 label")

    rows: list[dict[str, float | int]] = []
    legacy_rows: list[dict[str, float | int]] = []
    fallback_rows: list[dict[str, bool | int]] = []
    for region in regionprops(labels):
        label = int(region.label)
        cell_mask = np.asarray(region.image, dtype=bool)
        signal_crop = signal[region.slice]
        cell_fallback_counts = {column: 0 for column in RUI49_FEATURE_COLUMNS}
        morphology = _morphology_features(
            signal_crop,
            cell_mask,
            region,
            cell_fallback_counts,
        )
        intensity = _intensity_features(signal_crop, cell_mask)
        haralick = _haralick_features(
            signal_crop,
            cell_mask,
            reserve_zero=True,
        )
        values = {
            **morphology,
            **intensity,
            **haralick,
        }
        rows.append(
            {
                "cell_label": label,
                **{
                    f"cell__{name}": float(values[name])
                    for name in (
                        *MORPHOLOGY_FEATURE_NAMES,
                        *INTENSITY_FEATURE_NAMES,
                        *HARALICK_FEATURE_NAMES,
                    )
                },
            }
        )
        fallback_rows.append(
            {
                "cell_label": label,
                **{
                    column: bool(cell_fallback_counts[column])
                    for column in RUI49_FEATURE_COLUMNS
                },
            }
        )
        if include_legacy_texture:
            legacy = _haralick_features(
                signal_crop,
                cell_mask,
                reserve_zero=False,
            )
            legacy_rows.append(
                {
                    "cell_label": label,
                    **{
                        f"cell__{name}": float(legacy[name])
                        for name in HARALICK_FEATURE_NAMES
                    },
                }
            )

    result = pd.DataFrame(rows, columns=("cell_label", *RUI49_FEATURE_COLUMNS))
    if not result.empty and not np.isfinite(
        result.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Rui 49 特徵擷取產生非有限值")
    texture_columns = tuple(f"cell__{name}" for name in HARALICK_FEATURE_NAMES)
    legacy_result = (
        pd.DataFrame(legacy_rows, columns=("cell_label", *texture_columns))
        if include_legacy_texture
        else None
    )
    fallback_flags = pd.DataFrame(
        fallback_rows,
        columns=("cell_label", *RUI49_FEATURE_COLUMNS),
    )
    return RuiFeatureExtractionResult(
        features=result,
        legacy_texture_features=legacy_result,
        fallback_flags=fallback_flags,
        cell_count=len(result),
    )


def _morphology_features(
    signal: np.ndarray,
    cell_mask: np.ndarray,
    region: Any,
    fallback_counts: dict[str, int],
) -> dict[str, float]:
    measured = _measure_roi_with_python(signal, cell_mask)
    geometry = _geometry_from_measurements(measured)
    min_row, min_col, max_row, max_col = region.bbox
    center_y, center_x = region.centroid
    major_fallback = float(region.axis_major_length)
    minor_fallback = float(region.axis_minor_length)
    min_feret, max_feret = _feret_diameters(cell_mask)
    if not 0.0 <= min_feret <= max_feret:
        raise ValueError(
            "Feret diameter invariant 不符："
            f"cell_label={int(region.label)}, "
            f"MinFeretDiameter={min_feret}, "
            f"MaxFeretDiameter={max_feret}"
        )
    return {
        "Area": float(geometry["area"]),
        "BoundingBoxArea": float((max_row - min_row) * (max_col - min_col)),
        "BoundingBoxMinimum_X": float(min_col),
        "BoundingBoxMinimum_Y": float(min_row),
        "BoundingBoxMaximum_X": float(max_col),
        "BoundingBoxMaximum_Y": float(max_row),
        "Center_X": float(center_x),
        "Center_Y": float(center_y),
        "Compactness": _finite_or(
            geometry["compactness"],
            0.0,
            feature="cell__Compactness",
            counts=fallback_counts,
        ),
        "ConvexArea": float(region.area_convex),
        "Eccentricity": _finite_or(
            region.eccentricity,
            0.0,
            feature="cell__Eccentricity",
            counts=fallback_counts,
        ),
        "EquivalentDiameter": float(region.equivalent_diameter_area),
        "Extent": _finite_or(
            geometry["extent"],
            region.extent,
            feature="cell__Extent",
            counts=fallback_counts,
        ),
        "FormFactor": _finite_or(
            geometry["sphericity"],
            0.0,
            feature="cell__FormFactor",
            counts=fallback_counts,
        ),
        "MajorAxisLength": _finite_or(
            geometry["major_axis_length"],
            major_fallback,
            feature="cell__MajorAxisLength",
            counts=fallback_counts,
        ),
        "MaxFeretDiameter": max_feret,
        "MaximumRadius": _finite_or(
            geometry["maximum_radius"],
            0.0,
            feature="cell__MaximumRadius",
            counts=fallback_counts,
        ),
        "MeanRadius": _finite_or(
            geometry["mean_radius"],
            0.0,
            feature="cell__MeanRadius",
            counts=fallback_counts,
        ),
        "MedianRadius": _finite_or(
            geometry["median_radius"],
            0.0,
            feature="cell__MedianRadius",
            counts=fallback_counts,
        ),
        "MinFeretDiameter": min_feret,
        "MinorAxisLength": _finite_or(
            geometry["minor_axis_length"],
            minor_fallback,
            feature="cell__MinorAxisLength",
            counts=fallback_counts,
        ),
        "Orientation": _finite_or(
            region.orientation,
            0.0,
            feature="cell__Orientation",
            counts=fallback_counts,
        ),
        "Perimeter": _finite_or(
            geometry["perimeter"],
            region.perimeter,
            feature="cell__Perimeter",
            counts=fallback_counts,
        ),
        "Solidity": _finite_or(
            geometry["solidity"],
            region.solidity,
            feature="cell__Solidity",
            counts=fallback_counts,
        ),
    }


def _feret_diameters(cell_mask: np.ndarray) -> tuple[float, float]:
    """計算同一 whole-cell convex hull 的最小與最大 Feret diameter。

    CellProfiler 先取物件像素中心的 convex hull，再沿每條 hull edge 的
    法向量量測兩條平行支撐線距離。Rotating-calipers theorem 保證最小值
    必定出現在其中一個 edge orientation；最大值則是 hull vertices 的
    最大 pairwise distance。點的兩個 diameter 都是 0，線的 minimum 為 0。

    Args:
        cell_mask: 單一 whole-cell label 的二維布林裁切遮罩。

    Returns:
        ``(minimum, maximum)``，兩者皆為非負有限的 pixel 距離。

    Raises:
        ValueError: mask 維度不符、為空，或 diameter contract 不成立。
    """
    mask = np.asarray(cell_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("MinFeretDiameter mask 必須是二維陣列")
    if not np.any(mask):
        raise ValueError("MinFeretDiameter mask 不可為空")

    boundary = np.logical_and(
        mask,
        np.logical_not(
            binary_erosion(
                mask,
                structure=np.ones((3, 3), dtype=bool),
                border_value=0,
            )
        ),
    )
    points = np.argwhere(boundary).astype(np.float64, copy=False)
    if len(points) == 1:
        return 0.0, 0.0
    if len(points) == 2 or np.linalg.matrix_rank(points - points[0]) < 2:
        maximum = float(np.max(pdist(points)))
        return 0.0, maximum

    hull_points = points[ConvexHull(points).vertices]
    maximum = float(np.max(pdist(hull_points)))
    edges = np.roll(hull_points, -1, axis=0) - hull_points
    edge_lengths = np.linalg.norm(edges, axis=1)
    valid_edges = edge_lengths > 0.0
    if not np.any(valid_edges):
        return 0.0, maximum
    edges = edges[valid_edges]
    edge_lengths = edge_lengths[valid_edges]
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= edge_lengths[:, np.newaxis]
    projections = hull_points @ normals.T
    widths = np.ptp(projections, axis=0)
    minimum = float(np.min(widths))
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or minimum < 0.0
        or maximum < 0.0
        or minimum > maximum
    ):
        raise ValueError(
            "Feret diameter 計算違反 contract："
            f"minimum={minimum}, maximum={maximum}"
        )
    return minimum, maximum


def _finite_or(
    value: float,
    fallback: float,
    *,
    feature: str,
    counts: dict[str, int],
) -> float:
    numeric = float(value)
    if np.isfinite(numeric):
        return numeric
    fallback_numeric = float(fallback)
    if not np.isfinite(fallback_numeric):
        raise ValueError(f"{feature} 的 primary 與 fallback 皆為非有限值")
    counts[feature] += 1
    return fallback_numeric


def _intensity_features(
    signal: np.ndarray,
    cell_mask: np.ndarray,
) -> dict[str, float]:
    values = signal[cell_mask]
    eroded = binary_erosion(
        cell_mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    edge_values = signal[np.logical_and(cell_mask, np.logical_not(eroded))]

    y_coords, x_coords = np.nonzero(cell_mask)
    binary_y = float(np.mean(y_coords))
    binary_x = float(np.mean(x_coords))
    total_intensity = float(np.sum(values))
    if total_intensity > 0.0:
        weighted_y = float(np.dot(values, y_coords) / total_intensity)
        weighted_x = float(np.dot(values, x_coords) / total_intensity)
        mass_displacement = float(
            np.hypot(weighted_y - binary_y, weighted_x - binary_x)
        )
    else:
        #全零背景扣除細胞沒有可辨識的強度重心，採二值重心，位移定義為 0。
        mass_displacement = 0.0

    median = float(np.median(values))
    return {
        "IntegratedIntensity": total_intensity,
        "MeanIntensity": float(np.mean(values)),
        "StdIntensity": float(np.std(values)),
        "MinIntensity": float(np.min(values)),
        "MaxIntensity": float(np.max(values)),
        "IntegratedIntensityEdge": float(np.sum(edge_values)),
        "MeanIntensityEdge": float(np.mean(edge_values)),
        "StdIntensityEdge": float(np.std(edge_values)),
        "MinIntensityEdge": float(np.min(edge_values)),
        "MaxIntensityEdge": float(np.max(edge_values)),
        "MassDisplacement": mass_displacement,
        "MADIntensity": float(np.median(np.abs(values - median))),
    }


def _haralick_features(
    signal: np.ndarray,
    cell_mask: np.ndarray,
    *,
    reserve_zero: bool,
) -> dict[str, float]:
    y_coords, x_coords = np.nonzero(cell_mask)
    min_row, max_row = int(y_coords.min()), int(y_coords.max()) + 1
    min_col, max_col = int(x_coords.min()), int(x_coords.max()) + 1
    crop_mask = cell_mask[min_row:max_row, min_col:max_col]
    crop_signal = signal[min_row:max_row, min_col:max_col]
    if reserve_zero:
        quantized = quantize_masked_texture(crop_signal, crop_mask)
    else:
        masked_crop = np.where(crop_mask, crop_signal, 0.0)
        quantized = _quantize_to_levels(masked_crop, 256)

    if not _has_four_direction_pairs(quantized, distance=3):
        #mahotas 對沒有四方向有效 non-zero pair 的小物件會拋錯；此退化情境
        #沒有可估計的共生矩陣，明確以中性的全零 texture vector 表示。
        mean_features = np.zeros(len(HARALICK_FEATURE_NAMES), dtype=np.float64)
    else:
        directional = np.asarray(
            mahotas.features.haralick(
                quantized,
                distance=3,
                ignore_zeros=True,
            ),
            dtype=np.float64,
        )
        expected_shape = (4, len(HARALICK_FEATURE_NAMES))
        if directional.shape != expected_shape:
            raise ValueError(
                "mahotas Haralick 輸出尺寸不符："
                f"expected {expected_shape}, actual {directional.shape}"
            )
        mean_features = np.mean(directional, axis=0)
        if not np.isfinite(mean_features).all():
            raise ValueError("mahotas Haralick 產生非有限值")

    return {
        name: float(value)
        for name, value in zip(HARALICK_FEATURE_NAMES, mean_features)
    }


def _has_four_direction_pairs(image: np.ndarray, distance: int) -> bool:
    valid = np.asarray(image) != 0
    height, width = valid.shape
    for row_offset, col_offset in (
        (0, distance),
        (distance, distance),
        (distance, 0),
        (distance, -distance),
    ):
        row_start = max(0, -row_offset)
        row_stop = min(height, height - row_offset)
        col_start = max(0, -col_offset)
        col_stop = min(width, width - col_offset)
        if row_start >= row_stop or col_start >= col_stop:
            return False
        first = valid[row_start:row_stop, col_start:col_stop]
        second = valid[
            row_start + row_offset : row_stop + row_offset,
            col_start + col_offset : col_stop + col_offset,
        ]
        if not np.any(np.logical_and(first, second)):
            return False
    return True


def run_rui49_smoke(
    *,
    manifest_path: str | Path,
    masks_dir: str | Path,
    report_path: str | Path,
    metadata_path: str | Path,
    image_limit: int = 5,
    health_report_path: str | Path | None = None,
    cell_level_path: str | Path | None = None,
) -> dict[str, Any]:
    """對 manifest 排序後的前幾張影像執行 Rui 49 特徵 smoke test。

    Args:
        manifest_path: 含 ``image_key`` 與 ``pc_path`` 的 Exp3 manifest。
        masks_dir: Exp3 mask cache 根目錄；只索引 `.npz` 檔名 stem。
        report_path: machine-readable smoke JSON 輸出位置。
        metadata_path: 既有 ``run_metadata.json``；以 merge 方式補上實際方法。
        image_limit: 依 ``image_key`` 排序後選取的影像數，預設為 5。
        health_report_path: 可選的 provisional smoke health CSV 路徑；提供時
            會排除觸及 mask 四邊的 labels 後執行 fail-closed health gate。
        cell_level_path: 可選的 locked raw master cell-level CSV；提供時執行
            selected-FOV canonical Rui49 consistency，啟用 health 時亦為必填。

    Returns:
        與 ``report_path`` 內容相同的 smoke report。

    Raises:
        OSError: 輸入或輸出檔案無法讀寫時拋出。
        ValueError: manifest、mask、影像或 feature contract 不符時拋出。
        SmokeFeatureHealthError: provisional health gate blocked；此時 JSON、
            CSV 與 metadata 已完整寫出。
    """
    if isinstance(image_limit, bool) or not isinstance(image_limit, int):
        raise ValueError("image_limit 必須是正整數")
    if image_limit <= 0:
        raise ValueError("image_limit 必須是正整數")
    if health_report_path is not None and cell_level_path is None:
        raise ValueError("啟用 smoke feature health 時必須提供 cell_level_path")

    metadata_file = Path(metadata_path).expanduser().resolve(strict=False)
    environment_packages = _load_environment_packages(metadata_file)
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = pd.read_csv(manifest_file)
    required_columns = {"image_key", "pc_path"}
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"manifest 缺少欄位：{missing_columns}")
    keys = manifest["image_key"].astype("string").str.strip()
    if keys.isna().any() or (keys == "").any():
        raise ValueError("manifest image_key 不可為空")
    if keys.duplicated().any():
        duplicates = sorted(keys[keys.duplicated(keep=False)].astype(str).unique())
        raise ValueError(f"manifest image_key 不可重複：{duplicates}")
    manifest = manifest.assign(image_key=keys.astype(str)).sort_values(
        "image_key", kind="stable"
    )
    if len(manifest) < image_limit:
        raise ValueError(
            f"manifest 只有 {len(manifest)} 張，少於 smoke 要求 {image_limit} 張"
        )

    mask_index = _index_npz_masks(Path(masks_dir))
    selected = manifest.head(image_limit)
    selected_keys = selected["image_key"].tolist()
    master_rows: pd.DataFrame | None = None
    master_labels_by_key: dict[str, set[int]] | None = None
    if cell_level_path is not None:
        master_rows, master_labels_by_key = _load_smoke_master_rows(
            Path(cell_level_path),
            selected_keys,
        )
    missing_masks = [key for key in selected_keys if key not in mask_index]
    if missing_masks:
        raise ValueError(f"smoke manifest keys 缺少 mask：{missing_masks}")

    image_reports: list[dict[str, Any]] = []
    all_features: list[pd.DataFrame] = []
    legacy_texture_frames: list[pd.DataFrame] = []
    fallback_counts = {column: 0 for column in RUI49_FEATURE_COLUMNS}
    eligible_cell_count = 0
    health_features: list[pd.DataFrame] = []
    health_fallback_flags: list[pd.DataFrame] = []
    consistency_features: list[pd.DataFrame] = []
    health_master_pre_border_label_count = 0
    health_master_post_border_label_count = 0
    health_master_border_excluded_label_count = 0
    health_mask_only_label_count = 0
    for row in selected.itertuples(index=False):
        image_started = perf_counter()
        pc_path = Path(str(row.pc_path)).expanduser()
        if not pc_path.is_absolute():
            pc_path = manifest_file.parent / pc_path
        pc_path = pc_path.resolve(strict=True)
        mask_path = mask_index[str(row.image_key)]

        load_started = perf_counter()
        rgb_image = np.asarray(imread(pc_path))
        cell_labels = _load_whole_cell_labels(mask_path)
        load_seconds = perf_counter() - load_started

        preprocess_started = perf_counter()
        phase_signal = preprocess_phase_image(rgb_image)
        preprocess_seconds = perf_counter() - preprocess_started

        extraction_started = perf_counter()
        extraction = extract_rui49_features_with_diagnostics(
            phase_signal,
            cell_labels,
            include_legacy_texture=True,
        )
        features = extraction.features
        extraction_seconds = perf_counter() - extraction_started
        if features.empty:
            raise ValueError(f"{row.image_key} 的 mask 沒有 positive whole-cell label")
        if extraction.legacy_texture_features is None:
            raise ValueError("smoke 缺少 legacy texture evidence")
        all_features.append(features.loc[:, RUI49_FEATURE_COLUMNS])
        if master_rows is not None:
            keyed_features = features.loc[
                :, ("cell_label", *RUI49_FEATURE_COLUMNS)
            ].copy()
            keyed_features.insert(0, "image_key", str(row.image_key))
            consistency_features.append(keyed_features)
        legacy_texture_frames.append(extraction.legacy_texture_features)
        eligible_cell_count += extraction.cell_count
        for column, count in extraction.fallback_counts.items():
            fallback_counts[column] += count
        per_image_health_counts: dict[str, int] | None = None
        if health_report_path is not None:
            if master_labels_by_key is None:
                raise ValueError("smoke health 缺少 master label index")
            fallback_flags = extraction.fallback_flags
            feature_labels = features["cell_label"].to_numpy(dtype=np.int64)
            fallback_labels = fallback_flags["cell_label"].to_numpy(
                dtype=np.int64
            )
            if not np.array_equal(feature_labels, fallback_labels):
                raise ValueError(
                    f"{row.image_key} feature/fallback cell_label 順序不一致"
                )
            master_labels = master_labels_by_key[str(row.image_key)]
            mask_labels = set(feature_labels.tolist())
            master_only = sorted(master_labels - mask_labels)
            if master_only:
                raise ValueError(
                    f"{row.image_key} master label 不在 mask：{master_only}"
                )
            border_labels = find_border_labels(cell_labels)
            health_inclusion = np.logical_and(
                np.isin(
                    feature_labels,
                    np.fromiter(master_labels, dtype=np.int64),
                ),
                ~np.isin(
                    feature_labels,
                    np.fromiter(border_labels, dtype=np.int64),
                ),
            )
            before_count = len(master_labels)
            after_count = int(np.count_nonzero(health_inclusion))
            excluded_count = before_count - after_count
            mask_only_count = len(mask_labels - master_labels)
            health_master_pre_border_label_count += before_count
            health_master_post_border_label_count += after_count
            health_master_border_excluded_label_count += excluded_count
            health_mask_only_label_count += mask_only_count
            health_features.append(
                features.loc[health_inclusion, RUI49_FEATURE_COLUMNS].copy()
            )
            health_fallback_flags.append(
                fallback_flags.loc[
                    health_inclusion, RUI49_FEATURE_COLUMNS
                ].copy()
            )
            per_image_health_counts = {
                "mask_positive_label_count": len(mask_labels),
                "mask_only_label_count": mask_only_count,
                "master_pre_border_label_count": before_count,
                "master_post_border_label_count": after_count,
                "master_border_excluded_label_count": excluded_count,
            }
        per_image_fallback = _fallback_diagnostics(
            counts=extraction.fallback_counts,
            eligible_cell_count=extraction.cell_count,
            scope=extraction.fallback_scope,
        )
        image_reports.append(
            {
                "image_key": str(row.image_key),
                "pc_path": str(pc_path),
                "mask_path": str(mask_path),
                "cell_count": int(len(features)),
                "timing_seconds": {
                    "load": float(load_seconds),
                    "preprocess": float(preprocess_seconds),
                    "extraction": float(extraction_seconds),
                    "total": float(perf_counter() - image_started),
                },
                "feature_ranges": _feature_ranges(features),
                "fallback_diagnostics": per_image_fallback,
                **(
                    {"feature_health_label_counts": per_image_health_counts}
                    if per_image_health_counts is not None
                    else {}
                ),
            }
        )

    consistency_summary: dict[str, Any] | None = None
    if master_rows is not None:
        consistency_summary = _smoke_feature_consistency_summary(
            master_rows,
            consistency_features,
            image_count=len(selected_keys),
        )

    combined = pd.concat(all_features, ignore_index=True)
    combined_legacy_texture = pd.concat(legacy_texture_frames, ignore_index=True)
    texture_columns = tuple(
        f"cell__{name}" for name in HARALICK_FEATURE_NAMES
    )
    before_ranges = _ranges_for_columns(
        combined_legacy_texture,
        texture_columns,
    )
    after_ranges = _ranges_for_columns(combined, texture_columns)
    range_deltas = _texture_range_deltas(before_ranges, after_ranges)
    health_result: FeatureHealthResult | None = None
    health_summary: dict[str, Any] | None = None
    if health_report_path is not None:
        if (
            health_master_pre_border_label_count + health_mask_only_label_count
            != len(combined)
        ):
            raise ValueError("smoke master/mask-only label count 與 features 不一致")
        if health_master_post_border_label_count <= 0:
            raise ValueError("smoke border exclusion 後沒有 eligible whole-cell label")
        combined_health_features = pd.concat(
            health_features,
            ignore_index=True,
        )
        combined_health_flags = pd.concat(
            health_fallback_flags,
            ignore_index=True,
        )
        if len(combined_health_features) != health_master_post_border_label_count:
            raise ValueError("smoke health post-border feature count 不一致")
        if len(combined_health_flags) != health_master_post_border_label_count:
            raise ValueError("smoke health post-border fallback flag count 不一致")
        post_border_fallback_counts = {
            column: int(combined_health_flags[column].sum())
            for column in RUI49_FEATURE_COLUMNS
        }
        health_result = evaluate_feature_health(
            combined_health_features,
            fallback_counts=post_border_fallback_counts,
            eligible_cell_count=health_master_post_border_label_count,
            feature_columns=RUI49_FEATURE_COLUMNS,
            fallback_scope=SMOKE_HEALTH_SCOPE_POST_BORDER,
        )
        health_file = (
            Path(health_report_path).expanduser().resolve(strict=False)
        )
        health_result.write_report(health_file)
        health_summary = _smoke_health_summary(
            health_result,
            report_path=health_file,
            mask_positive_label_count=len(combined),
            mask_only_label_count=health_mask_only_label_count,
            master_pre_border_label_count=health_master_pre_border_label_count,
            master_post_border_label_count=health_master_post_border_label_count,
            master_border_excluded_label_count=(
                health_master_border_excluded_label_count
            ),
        )
    report: dict[str, Any] = {
        "status": (
            "blocked_feature_health"
            if health_result is not None and health_result.blocked
            else "passed"
        ),
        "selection": "first_image_keys_sorted_ascending",
        "image_limit": image_limit,
        "image_keys": selected_keys,
        "texture_scope": TEXTURE_SCOPE_ALL_MASK_LABELS,
        "full_693_feature_consistency_status": "not_run",
        **(
            {"whole_cell_feature_consistency": consistency_summary}
            if consistency_summary is not None
            else {}
        ),
        "background_subtraction_method": "rolling_ball_radius_50",
        "images": image_reports,
        "overall": {
            "image_count": len(image_reports),
            "cell_count": int(len(combined)),
            "feature_ranges": _feature_ranges(combined),
            "texture_feature_ranges_before_zero_reservation": before_ranges,
            "texture_feature_ranges_after_zero_reservation": after_ranges,
            "texture_feature_range_deltas": range_deltas,
            "fallback_diagnostics": _fallback_diagnostics(
                counts=fallback_counts,
                eligible_cell_count=eligible_cell_count,
                scope=FALLBACK_SCOPE_PRE_BORDER,
            ),
            **(
                {"feature_health": health_summary}
                if health_summary is not None
                else {}
            ),
        },
    }
    report_file = Path(report_path).expanduser().resolve(strict=False)
    _write_json(report_file, report)
    _merge_run_metadata(
        metadata_file,
        smoke_report=report,
        report_path=report_file,
        environment_packages=environment_packages,
        health_summary=health_summary,
        consistency_summary=consistency_summary,
    )
    if health_result is not None and health_result.blocked:
        raise SmokeFeatureHealthError(
            "smoke provisional feature-health blocked after reports were written: "
            + "; ".join(health_result.reasons)
        )
    return report


def _load_smoke_master_rows(
    cell_level_path: Path,
    selected_keys: list[str],
) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    """載入 selected FOV 的 raw whole-cell/nucleus pair rows。

    Args:
        cell_level_path: locked raw master CSV。
        selected_keys: smoke 選取且排序完成的 image keys。

    Returns:
        保留每個 nucleus pair row 的 frame，以及逐 FOV distinct cell labels。

    Raises:
        ValueError: schema、FOV coverage、label 或 raw pair uniqueness 不合法。
    """
    master_path = cell_level_path.expanduser().resolve(strict=True)
    frame = pd.read_csv(master_path)
    required_columns = {"image_key", "cell_label", "nucleus_label"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"cell-level master 缺少欄位：{missing_columns}")
    normalized_keys = frame["image_key"].astype("string").str.strip()
    selected = frame.assign(image_key=normalized_keys).loc[
        normalized_keys.isin(selected_keys),
        ["image_key", "cell_label", "nucleus_label"],
    ].copy()
    missing_fovs = sorted(set(selected_keys) - set(selected["image_key"]))
    if missing_fovs:
        raise ValueError(f"cell-level master 缺少 smoke FOV：{missing_fovs}")
    labels = pd.to_numeric(selected["cell_label"], errors="coerce")
    numeric = labels.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or np.any(numeric <= 0)
    ):
        raise ValueError("cell-level master cell_label 必須是 positive integers")
    selected["cell_label"] = numeric.astype(np.int64)
    nucleus_labels = pd.to_numeric(selected["nucleus_label"], errors="coerce")
    nucleus_numeric = nucleus_labels.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(nucleus_numeric).all()
        or not np.equal(nucleus_numeric, np.floor(nucleus_numeric)).all()
        or np.any(nucleus_numeric <= 0)
    ):
        raise ValueError("cell-level master nucleus_label 必須是 positive integers")
    selected["nucleus_label"] = nucleus_numeric.astype(np.int64)
    raw_pair_columns = ["image_key", "cell_label", "nucleus_label"]
    if selected.duplicated(raw_pair_columns).any():
        duplicates = selected.loc[
            selected.duplicated(raw_pair_columns, keep=False),
            raw_pair_columns,
        ].drop_duplicates()
        raise ValueError(
            "cell-level master raw pair key 不可重複："
            + repr(list(duplicates.itertuples(index=False, name=None))[:5])
        )
    labels_by_key = {
        key: set(
            selected.loc[selected["image_key"].eq(key), "cell_label"].tolist()
        )
        for key in selected_keys
    }
    return selected.reset_index(drop=True), labels_by_key


def _smoke_feature_consistency_summary(
    master_rows: pd.DataFrame,
    feature_frames: list[pd.DataFrame],
    *,
    image_count: int,
) -> dict[str, Any]:
    """將 unique label features join 回 raw pairs 並執行 canonical 49 gate。

    Args:
        master_rows: selected FOV 的 raw ``image×cell×nucleus`` rows。
        feature_frames: 每張影像每個 positive cell label 恰一列的 Rui49 features。
        image_count: 本次 smoke 真正選取的 FOV 數。

    Returns:
        可直接寫入 report/metadata 的 smoke-scope consistency 證據。

    Raises:
        ValueError: feature key 重複、master label 無 feature 或 join coverage 不符。
        CellDedupError: canonical 49 exact consistency assertion 失敗時由下層拋出。
    """
    if not feature_frames:
        raise ValueError("smoke consistency 缺少 extracted Rui49 features")
    features = pd.concat(feature_frames, ignore_index=True)
    key_columns = ["image_key", "cell_label"]
    duplicate_mask = features.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = features.loc[duplicate_mask, key_columns].drop_duplicates()
        raise ValueError(
            "smoke Rui49 feature key 不可重複："
            + repr(list(duplicates.itertuples(index=False, name=None))[:5])
        )
    master_keys = pd.MultiIndex.from_frame(master_rows.loc[:, key_columns])
    feature_keys = pd.MultiIndex.from_frame(features.loc[:, key_columns])
    missing_feature_keys = master_keys.unique().difference(feature_keys)
    if len(missing_feature_keys):
        raise ValueError(
            "smoke master label 缺少 Rui49 feature："
            + repr(list(missing_feature_keys[:5]))
        )
    joined = master_rows.loc[
        :, ("image_key", "cell_label", "nucleus_label")
    ].merge(
        features,
        on=key_columns,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(joined) != len(master_rows):
        raise ValueError("smoke Rui49 many-to-one join 未保留全部 raw pair rows")
    if joined.loc[:, RUI49_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("smoke Rui49 many-to-one join 產生缺失 feature")

    scope = f"smoke_selected_{image_count}_fovs"
    evidence = assert_full_rui49_feature_consistency(joined, scope=scope)
    return {
        "status": "passed",
        "scope": evidence.scope,
        "feature_count": evidence.feature_count,
        "raw_pair_row_count": evidence.checked_row_count,
        "distinct_label_count": evidence.checked_cell_count,
        "multirow_key_count": evidence.checked_duplicate_key_count,
    }


def _index_npz_masks(masks_dir: Path) -> dict[str, Path]:
    root = masks_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"masks_dir 不是目錄：{root}")
    index: dict[str, Path] = {}
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == ".npz"
        ),
        key=lambda candidate: candidate.as_posix().lower(),
    ):
        if path.stem in index:
            raise ValueError(
                f"mask stem 重複：{path.stem}: {index[path.stem]}, {path}"
            )
        index[path.stem] = path.resolve(strict=True)
    return index


def _load_whole_cell_labels(mask_path: Path) -> np.ndarray:
    with np.load(mask_path, allow_pickle=False) as cached:
        if "cell_mask" not in cached:
            raise ValueError(f"mask cache 缺少 cell_mask：{mask_path}")
        return np.asarray(cached["cell_mask"])


def _feature_ranges(features: pd.DataFrame) -> dict[str, dict[str, float | bool]]:
    return _ranges_for_columns(features, RUI49_FEATURE_COLUMNS)


def _ranges_for_columns(
    features: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, dict[str, float | bool]]:
    ranges: dict[str, dict[str, float | bool]] = {}
    for column in columns:
        values = features[column].to_numpy(dtype=np.float64)
        finite = bool(np.isfinite(values).all())
        if values.size == 0 or not finite:
            raise ValueError(f"smoke feature range 無法計算：{column}")
        ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "finite": finite,
        }
    return ranges


def _texture_range_deltas(
    before: Mapping[str, Mapping[str, float | bool]],
    after: Mapping[str, Mapping[str, float | bool]],
) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for column in before:
        before_min = float(before[column]["min"])
        before_max = float(before[column]["max"])
        after_min = float(after[column]["min"])
        after_max = float(after[column]["max"])
        deltas[column] = {
            "min_delta": after_min - before_min,
            "max_delta": after_max - before_max,
            "min_change_percent": _change_percent(before_min, after_min),
            "max_change_percent": _change_percent(before_max, after_max),
        }
    return deltas


def _change_percent(before: float, after: float) -> float | None:
    if before == 0.0:
        return 0.0 if after == 0.0 else None
    return (after - before) / abs(before) * 100.0


def _fallback_diagnostics(
    *,
    counts: Mapping[str, int],
    eligible_cell_count: int,
    scope: str,
) -> dict[str, Any]:
    rates = {
        column: (
            count / eligible_cell_count if eligible_cell_count > 0 else 0.0
        )
        for column, count in counts.items()
    }
    return {
        "scope": scope,
        "eligible_cell_count": eligible_cell_count,
        "counts": dict(counts),
        "rates": rates,
    }


def _smoke_health_summary(
    result: FeatureHealthResult,
    *,
    report_path: Path,
    mask_positive_label_count: int,
    mask_only_label_count: int,
    master_pre_border_label_count: int,
    master_post_border_label_count: int,
    master_border_excluded_label_count: int,
) -> dict[str, Any]:
    payload = result.to_metadata_payload()
    payload.pop("fallback_scope", None)
    return {
        "status": (
            "smoke_provisional_blocked"
            if result.blocked
            else "smoke_provisional_passed"
        ),
        "scope": result.fallback_scope,
        "health_scope": HEALTH_SCOPE_MASTER_POST_BORDER,
        "mask_positive_label_count": mask_positive_label_count,
        "mask_only_label_count": mask_only_label_count,
        "master_pre_border_label_count": master_pre_border_label_count,
        "master_post_border_label_count": master_post_border_label_count,
        "master_border_excluded_label_count": (
            master_border_excluded_label_count
        ),
        "report_path": str(report_path),
        **payload,
    }


def _merge_run_metadata(
    metadata_path: Path,
    *,
    smoke_report: dict[str, Any],
    report_path: Path,
    environment_packages: tuple[dict[str, str], Path] | None,
    health_summary: dict[str, Any] | None,
    consistency_summary: dict[str, Any] | None,
) -> None:
    if metadata_path.exists():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("run_metadata.json root 必須是 object")
        metadata = loaded
    else:
        metadata = {}
    snapshot_status = (
        "validated"
        if metadata.get("snapshot_validation_status") == "validated"
        or metadata.get("status") == "snapshot_validated"
        else "unknown"
    )
    smoke_status = str(smoke_report["status"])
    feature_smoke_status = (
        "validated" if smoke_status == "passed" else smoke_status
    )
    if environment_packages is not None:
        package_versions, package_path = environment_packages
        metadata["packages"] = package_versions
        metadata["python"] = package_versions["python"]
        metadata["environment_package_count"] = len(package_versions)
        metadata["environment_packages_path"] = str(package_path)
        metadata["runtime_executable"] = str(Path(sys.executable).resolve())
    metadata.update(
        {
            "background_subtraction_method": "rolling_ball_radius_50",
            "background_subtraction_status": "smoke_validated",
            "haralick_implementation": "mahotas.features.haralick",
            "haralick_distance": 3,
            "haralick_directions_averaged": 4,
            "haralick_quantization_levels": 256,
            "haralick_padding_level": 0,
            "haralick_in_mask_levels": "1..255",
            "haralick_zero_reserved_for_padding": True,
            "whole_cell_only": True,
            "feature_extraction_status": "smoke_validated",
            "feature_smoke_status": feature_smoke_status,
            "texture_scope": TEXTURE_SCOPE_ALL_MASK_LABELS,
            "full_693_feature_consistency_status": "not_run",
            "validation_status": {
                "snapshot": snapshot_status,
                "feature_smoke": feature_smoke_status,
            },
            "smoke_test": {
                "status": smoke_status,
                "image_keys": smoke_report["image_keys"],
                "image_count": smoke_report["overall"]["image_count"],
                "cell_count": smoke_report["overall"]["cell_count"],
                "report_path": str(report_path),
                "texture_scope": TEXTURE_SCOPE_ALL_MASK_LABELS,
                "texture_feature_ranges_before_zero_reservation": smoke_report[
                    "overall"
                ]["texture_feature_ranges_before_zero_reservation"],
                "texture_feature_ranges_after_zero_reservation": smoke_report[
                    "overall"
                ]["texture_feature_ranges_after_zero_reservation"],
                "texture_feature_range_deltas": smoke_report["overall"][
                    "texture_feature_range_deltas"
                ],
                "fallback_diagnostics": smoke_report["overall"][
                    "fallback_diagnostics"
                ],
            },
        }
    )
    if health_summary is not None:
        provisional_status = str(health_summary["status"])
        metadata["feature_health_status"] = provisional_status
        metadata["feature_health_smoke"] = health_summary
        metadata["validation_status"]["feature_health"] = provisional_status
        metadata["smoke_test"]["feature_health"] = health_summary
    if consistency_summary is not None:
        consistency_status = str(consistency_summary["status"])
        metadata["whole_cell_feature_consistency_smoke"] = consistency_summary
        metadata["validation_status"][
            "whole_cell_feature_consistency_smoke"
        ] = consistency_status
        metadata["smoke_test"][
            "whole_cell_feature_consistency"
        ] = consistency_summary
    _write_json(metadata_path, metadata)


def _load_environment_packages(
    metadata_path: Path,
) -> tuple[dict[str, str], Path] | None:
    package_path = metadata_path.with_name("environment_packages.json")
    if not package_path.exists():
        return None
    loaded = json.loads(package_path.read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, list):
        raise ValueError("environment_packages.json root 必須是 array")
    versions: dict[str, str] = {}
    for index, record in enumerate(loaded):
        if not isinstance(record, Mapping):
            raise ValueError(f"environment package #{index} 必須是 object")
        name = str(record.get("name", "")).strip()
        version = str(record.get("version", "")).strip()
        if not name or not version:
            raise ValueError(f"environment package #{index} 缺少 name/version")
        if name in versions and versions[name] != version:
            raise ValueError(
                f"environment package 版本衝突：{name}="
                f"{versions[name]} / {version}"
            )
        versions[name] = version
    recorded_python = versions.get("python")
    runtime_python = platform.python_version()
    if recorded_python is None:
        raise ValueError("environment_packages.json 缺少 python 套件版本")
    if recorded_python != runtime_python:
        raise ValueError(
            "runtime Python 與 environment snapshot 不符："
            f"runtime={runtime_python}, expected={recorded_python}"
        )
    return dict(sorted(versions.items())), package_path.resolve(strict=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
