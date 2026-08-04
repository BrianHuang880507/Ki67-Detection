"""提供 Exp3 的 PC-only segmentation、mask cache 與開發期 DAPI 驗證。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import ntpath
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

from immunity.exp3.feature_sets import (
    BASIC_GEOMETRY,
    PAPER_STYLE_EXTRA_FEATURES,
    PRIMARY_CELL_FEATURES,
    _feature_cache_dir_from_output,
    _safe_output_file,
    _validate_feature_set_specs,
    _write_csv_atomically,
)
from ki67dtc.cell_anal import (
    _geometry_from_measurements,
    _measure_roi_with_python,
    _safe_divide,
    _zernike_feature_values_python,
)
from ki67dtc.paired_overlay import find_paired_labels


SEGMENTATION_QC_COLUMNS = [
    "image_key",
    "group_id",
    "mask_path",
    "cache_status",
    "cache_reason",
    "pc_sha256",
    "cache_provenance_hash",
    "status",
    "error",
    "whole_cell_labels",
    "nucleus_labels",
    "paired_cells",
]

NUCLEUS_COMPARISON_COLUMNS = [
    "pc_label",
    "dapi_label",
    "dice",
    "iou",
    "area_ratio_pc_to_dapi",
    "pc_feret_length",
    "dapi_feret_length",
    "matched_cell_coverage",
]

DEVELOPMENT_VALIDATION_COLUMNS = [
    "image_key",
    "pc_path",
    "dapi_path",
    "mask_path",
    *NUCLEUS_COMPARISON_COLUMNS,
    "role",
]

FEATURE_EXTRACTION_QC_COLUMNS = [
    "image_key",
    "status",
    "error",
    "paired_cells",
    "kept_full_cells",
    "excluded_nucleus_outside",
    "excluded_empty_cytoplasm",
    "background_median",
]
CELL_LEVEL_COLUMNS = [
    "image_key",
    "cell_label",
    "nucleus_label",
    *PRIMARY_CELL_FEATURES,
    "IDO_score",
]


def paper_style_extras(
    phase: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> dict[str, float]:
    """計算 opt-in paper-style label-free 近似特徵。

    Args:
        phase: 二維 phase-contrast 影像。
        cell_mask: 單顆 whole-cell 布林遮罩。
        nucleus_mask: 與 cell 配對的單顆 nucleus 布林遮罩。

    Returns:
        60 個 paper-style 額外 cell-level 特徵。

    Raises:
        ValueError: 影像與遮罩尺寸不一致，或遮罩沒有有效 cell/cytoplasm 時拋出。
    """
    phase_array = np.asarray(phase, dtype=np.float64)
    cell_array = np.asarray(cell_mask, dtype=bool)
    nucleus_array = np.asarray(nucleus_mask, dtype=bool)
    if any(array.ndim != 2 for array in (phase_array, cell_array, nucleus_array)):
        raise ValueError("phase、cell_mask 與 nucleus_mask 都必須是二維陣列")
    if len({array.shape for array in (phase_array, cell_array, nucleus_array)}) != 1:
        raise ValueError("phase、cell_mask 與 nucleus_mask 必須具有相同尺寸")
    cytoplasm_array = cell_array & ~nucleus_array
    if not np.any(cell_array) or not np.any(cytoplasm_array):
        raise ValueError("paper-style features 需要非空 cell 與 cytoplasm")

    cell_measurements = _measure_roi_with_python(phase_array, cell_array)
    nucleus_measurements = _measure_roi_with_python(phase_array, nucleus_array)
    cytoplasm_measurements = _measure_roi_with_python(
        phase_array, cytoplasm_array
    )
    cell_zernike = _zernike_feature_values_python(phase_array, cell_array)
    nucleus_zernike = _zernike_feature_values_python(phase_array, nucleus_array)
    if len(cell_zernike) != 25 or len(nucleus_zernike) != 25:
        raise ValueError("paper-style Zernike helper 必須各回傳 25 個 moments")

    cell_area = float(cell_measurements["area"])
    centroid_distance = math.hypot(
        float(nucleus_measurements["x_centroid"])
        - float(cell_measurements["x_centroid"]),
        float(nucleus_measurements["y_centroid"])
        - float(cell_measurements["y_centroid"]),
    )
    equivalent_cell_radius = (
        math.sqrt(cell_area / math.pi)
        if np.isfinite(cell_area) and cell_area > 0
        else np.nan
    )
    values = {
        **{
            f"cell__zernike_{index:02d}": float(value)
            for index, value in enumerate(cell_zernike)
        },
        **{
            f"nucleus__zernike_{index:02d}": float(value)
            for index, value in enumerate(nucleus_zernike)
        },
        "nucleus_cytoplasm_mean_ratio": _safe_divide(
            float(nucleus_measurements["mean"]),
            float(cytoplasm_measurements["mean"]),
        ),
        "nucleus_cytoplasm_intden_ratio": _safe_divide(
            float(nucleus_measurements["intden"]),
            float(cytoplasm_measurements["intden"]),
        ),
        "nucleus_cytoplasm_raw_intden_ratio": _safe_divide(
            float(nucleus_measurements["raw_intden"]),
            float(cytoplasm_measurements["raw_intden"]),
        ),
        "nucleus_cell_intden_ratio": _safe_divide(
            float(nucleus_measurements["intden"]),
            float(cell_measurements["intden"]),
        ),
        "nucleus_cytoplasm_entropy_difference": _finite_difference(
            float(nucleus_measurements["entropy"]),
            float(cytoplasm_measurements["entropy"]),
        ),
        "nucleus_cytoplasm_cv_difference": _finite_difference(
            float(nucleus_measurements["cv"]),
            float(cytoplasm_measurements["cv"]),
        ),
        "nucleus_centroid_offset": _safe_divide(
            centroid_distance, equivalent_cell_radius
        ),
        "cell__phase_intensity_cv": float(cell_measurements["cv"]),
        "nucleus__phase_intensity_cv": float(nucleus_measurements["cv"]),
        "cytoplasm__phase_intensity_cv": float(cytoplasm_measurements["cv"]),
    }
    return {name: float(values[name]) for name in PAPER_STYLE_EXTRA_FEATURES}


def _finite_difference(left: float, right: float) -> float:
    """只在兩側皆為有限值時回傳差值。"""
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return float(left - right)


class Segmenter(Protocol):
    """定義 phase mask cache 所需的最小 segmentation 介面。"""

    def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """將一張 PC 影像轉成 whole-cell 與 nucleus labels。"""


class PhaseSegmenter:
    """只從單一 PC 影像產生 whole-cell 與 nucleus masks。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """載入並重用兩個 PC segmentation models。

        Args:
            config: segmentation 設定；``device="cpu"`` 時停用 GPU。
        """
        from cellpose import models
        from ki67dtc.img_prep import CYTO_MODEL_PATH, PC_NUC_MODEL_PATH

        use_gpu = str(config.get("device", "gpu")).lower() != "cpu"
        self.cell_model = models.CellposeModel(
            gpu=use_gpu,
            pretrained_model=CYTO_MODEL_PATH,
        )
        self.nucleus_model = models.CellposeModel(
            gpu=use_gpu,
            pretrained_model=PC_NUC_MODEL_PATH,
        )
        self.config = dict(config)
        self._model_paths = {
            "whole_cell": Path(CYTO_MODEL_PATH),
            "nucleus": Path(PC_NUC_MODEL_PATH),
        }

    def cache_signature(self) -> Mapping[str, Any]:
        """建立 production model 版本與 pretrained weights 內容簽章。

        Returns:
            Cellpose 版本、模型類別與兩個模型檔案的 SHA-256。

        Raises:
            OSError: 任一 pretrained model 檔案無法讀取時拋出。
        """
        try:
            cellpose_version = importlib.metadata.version("cellpose")
        except importlib.metadata.PackageNotFoundError:
            cellpose_version = "unavailable"
        return {
            "cellpose_version": cellpose_version,
            "model_class": (
                f"{type(self.cell_model).__module__}."
                f"{type(self.cell_model).__qualname__}"
            ),
            "pretrained_models": {
                name: {
                    "path": str(path.resolve(strict=True)),
                    "sha256": _sha256_file(path),
                }
                for name, path in sorted(self._model_paths.items())
            },
        }

    def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """讀取一張 PC 影像並回傳原始尺寸的配對 labels。

        Args:
            path: PC 影像路徑。

        Returns:
            過濾後的 whole-cell 與 PC nucleus label masks。

        Raises:
            ValueError: 影像或模型輸出維度不符合預期時拋出。
        """
        from cellpose import io
        from ki67dtc.img_prep import (
            CYTO_MODEL_INPUT_SIZE,
            NUC_MODEL_INPUT_SIZE,
            _filter_small_unpaired_labels,
        )

        image = np.asarray(io.imread(path))
        if image.ndim < 2:
            raise ValueError("PC 影像至少必須具有高度與寬度兩個維度")
        threshold = float(self.config.get("cellprob_threshold", 0.0))
        cell = _eval_label_mask(
            self.cell_model,
            image,
            CYTO_MODEL_INPUT_SIZE,
            threshold,
        )
        nucleus = _eval_label_mask(
            self.nucleus_model,
            image,
            NUC_MODEL_INPUT_SIZE,
            threshold,
        )
        filtered = _filter_small_unpaired_labels(
            cell,
            nucleus,
            min_area_ratio=float(self.config.get("min_area_ratio", 0.15)),
            min_area_floor=int(self.config.get("min_area_floor", 30)),
        )
        return (
            np.asarray(filtered.cytoplasm_mask, dtype=np.int32),
            np.asarray(filtered.nucleus_mask, dtype=np.int32),
        )


def _eval_label_mask(
    model: Any,
    image: np.ndarray,
    model_input_size: tuple[int, int],
    threshold: float,
    *,
    channels: Sequence[int] = (0, 0),
) -> np.ndarray:
    """以既有輸入尺寸執行 Cellpose，並還原原圖大小的整數 labels。

    Args:
        model: 具有 ``eval`` 方法的 Cellpose model。
        image: 原始影像陣列。
        model_input_size: OpenCV 使用的 ``(width, height)`` 目標尺寸。
        threshold: Cellpose cell probability threshold。
        channels: Cellpose channel mapping。

    Returns:
        與原圖高度、寬度完全相同的 ``np.int32`` 二維 label mask。

    Raises:
        ValueError: 原圖或模型輸出不是預期維度時拋出。
    """
    image_array = np.asarray(image)
    if image_array.ndim < 2:
        raise ValueError("輸入影像至少必須具有高度與寬度兩個維度")
    original_height, original_width = image_array.shape[:2]
    resized_image = cv2.resize(
        image_array,
        tuple(int(value) for value in model_input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    result = model.eval(
        resized_image,
        diameter=None,
        channels=list(channels),
        cellprob_threshold=float(threshold),
        flow_threshold=0.4,
        invert=False,
    )
    labels = result[0] if isinstance(result, (tuple, list)) else result
    label_array = np.asarray(labels)
    if label_array.ndim != 2:
        raise ValueError("Cellpose label mask 必須是二維陣列")
    restored = cv2.resize(
        label_array.astype(np.int32, copy=False),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.asarray(restored, dtype=np.int32)


def cache_phase_masks(
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    config: Mapping[str, Any],
    segmenter: Segmenter | None = None,
) -> pd.DataFrame:
    """只使用 manifest 的 PC 路徑建立或重用每張影像的 mask cache。

    Args:
        manifest: 至少包含 ``image_key``、``group_id`` 與 ``pc_path``。
        cache_dir: feature cache 根目錄。
        config: segmentation 設定；也接受含 ``segmentation`` 子區段的設定。
        segmenter: 可選的 segmentation adapter，測試可注入 synthetic 實作。

    Returns:
        每張 FOV 一列的 segmentation QC；單張失敗不會中止後續 FOV。

    Raises:
        ValueError: manifest 缺少 Primary path 必要欄位時拋出。
    """
    required_columns = {"image_key", "group_id", "pc_path"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"manifest 缺少欄位：{sorted(missing_columns)}")

    segmentation_config = _segmentation_config(config)
    force = bool(segmentation_config.get("force", False))
    cache_dir_path = _validate_exp3_cache_root(Path(cache_dir))
    cache_root = cache_dir_path / "masks"
    rows: list[dict[str, Any]] = []

    #完整 preflight 必須早於 segmenter 初始化與任何 cache exists/read/write。
    primary_manifest = manifest.loc[:, ["image_key", "group_id", "pc_path"]]
    cache_entries: list[tuple[str, str, Path, Path]] = []
    for values in primary_manifest.itertuples(index=False, name=None):
        image_key = _required_cache_component(values[0], "image_key")
        group_id = _required_cache_component(values[1], "group_id")
        pc_path = Path(values[2])
        proposed = cache_root / group_id / f"{image_key}.npz"
        mask_path = _validate_exp3_mask_cache_path(
            proposed,
            cache_dir_path,
            group_id,
            image_key,
        )
        cache_entries.append((image_key, group_id, pc_path, mask_path))

    active_segmenter = segmenter or PhaseSegmenter(segmentation_config)
    run_provenance = _mask_run_provenance(segmentation_config, active_segmenter)

    for image_key, group_id, pc_path, mask_path in cache_entries:
        whole_cell_labels = 0
        nucleus_labels = 0
        paired_cells = 0
        cache_status = "failed"
        cache_reason = "processing_failed"
        pc_sha256 = ""
        provenance_hash = ""

        try:
            pc_sha256 = _sha256_file(pc_path)
            provenance = {**run_provenance, "pc_sha256": pc_sha256}
            provenance_json = _canonical_json(provenance)
            provenance_hash = _sha256_text(provenance_json)
            existed = mask_path.exists()
            masks = None
            if existed and not force:
                try:
                    masks, cached_provenance = _load_mask_cache_entry(mask_path)
                    cache_reason = _cache_mismatch_reason(
                        cached_provenance,
                        provenance,
                    )
                    if cache_reason == "provenance_match":
                        cache_status = "reused"
                    else:
                        masks = None
                        cache_status = "replaced"
                except KeyError as error:
                    masks = None
                    cache_status = "replaced"
                    cache_reason = (
                        "legacy_cache"
                        if "provenance" in str(error)
                        else "invalid_cache"
                    )
                except (
                    OSError,
                    ValueError,
                    EOFError,
                    zipfile.BadZipFile,
                ):
                    #損毀或不可驗證的 cache 一律由相同 PC 重新產生。
                    masks = None
                    cache_status = "replaced"
                    cache_reason = "invalid_cache"

            if masks is None:
                masks = active_segmenter.segment(pc_path)
                cache_status = "replaced" if existed else "created"
                if force:
                    cache_reason = "force"
                elif not existed:
                    cache_reason = "cache_missing"

            cell_mask, nucleus_mask = _validate_mask_pair(*masks)
            whole_cell_labels = _label_count(cell_mask)
            nucleus_labels = _label_count(nucleus_mask)

            from ki67dtc.paired_overlay import find_paired_labels

            paired_cells = len(find_paired_labels(cell_mask, nucleus_mask))
            if paired_cells == 0:
                raise ValueError("配對細胞數為 0")

            if cache_status != "reused":
                _write_mask_cache(
                    mask_path,
                    cell_mask,
                    nucleus_mask,
                    provenance_json,
                    provenance_hash,
                )
            status = "passed"
            error_text = ""
        except Exception as error:  #單張失敗必須保留 QC 並繼續處理下一張。
            status = "failed"
            error_text = str(error)
            cache_status = "failed"
            cache_reason = "processing_failed"

        rows.append(
            {
                "image_key": image_key,
                "group_id": group_id,
                "mask_path": str(mask_path),
                "cache_status": cache_status,
                "cache_reason": cache_reason,
                "pc_sha256": pc_sha256,
                "cache_provenance_hash": provenance_hash,
                "status": status,
                "error": error_text,
                "whole_cell_labels": whole_cell_labels,
                "nucleus_labels": nucleus_labels,
                "paired_cells": paired_cells,
            }
        )

    return pd.DataFrame(rows, columns=SEGMENTATION_QC_COLUMNS)


def load_cached_masks(mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """載入單張 ``.npz`` cache 中的 whole-cell 與 nucleus masks。

    Args:
        mask_path: 由 ``cache_phase_masks`` 建立的 cache 路徑。

    Returns:
        ``(cell_mask, nucleus_mask)``。

    Raises:
        KeyError: cache 缺少必要陣列時拋出。
        OSError: cache 無法讀取時拋出。
        EOFError: cache 內容提前結束時拋出。
        zipfile.BadZipFile: ZIP 結構或 member CRC 無效時拋出。
    """
    with np.load(Path(mask_path), allow_pickle=False) as cached:
        if "cell_mask" not in cached or "nucleus_mask" not in cached:
            raise KeyError("mask cache 缺少 cell_mask 或 nucleus_mask")
        return (
            np.asarray(cached["cell_mask"], dtype=np.int32),
            np.asarray(cached["nucleus_mask"], dtype=np.int32),
        )


def compare_nucleus_masks(
    pc_mask: np.ndarray,
    dapi_mask: np.ndarray,
) -> pd.DataFrame:
    """以 Hungarian assignment 比較 PC 與 DAPI nucleus labels。

    Args:
        pc_mask: PC nucleus 二維 label mask。
        dapi_mask: DAPI nucleus 二維 label mask。

    Returns:
        每個 IoU 大於零之配對的 Dice、IoU、面積比、Feret 長度，及相同的
        image-level matched-cell coverage。無 labels 或無 overlap 時仍回傳一列
        coverage 為零的 sentinel summary，避免該 FOV 從開發驗證消失。

    Raises:
        ValueError: masks 不是二維或尺寸不一致時拋出。
    """
    pc_array, dapi_array = _validate_mask_pair(pc_mask, dapi_mask)
    pc_labels = _nonzero_labels(pc_array)
    dapi_labels = _nonzero_labels(dapi_array)
    if not pc_labels or not dapi_labels:
        return _zero_match_summary()

    pc_areas = {label: int(np.count_nonzero(pc_array == label)) for label in pc_labels}
    dapi_areas = {
        label: int(np.count_nonzero(dapi_array == label)) for label in dapi_labels
    }
    intersections = np.zeros((len(pc_labels), len(dapi_labels)), dtype=np.int64)
    ious = np.zeros_like(intersections, dtype=float)
    for pc_index, pc_label in enumerate(pc_labels):
        pc_region = pc_array == pc_label
        overlapping_dapi = dapi_array[pc_region]
        for dapi_index, dapi_label in enumerate(dapi_labels):
            intersection = int(np.count_nonzero(overlapping_dapi == dapi_label))
            intersections[pc_index, dapi_index] = intersection
            union = pc_areas[pc_label] + dapi_areas[dapi_label] - intersection
            ious[pc_index, dapi_index] = intersection / union if union else 0.0

    pc_indices, dapi_indices = linear_sum_assignment(-ious)
    matches = [
        (int(pc_index), int(dapi_index))
        for pc_index, dapi_index in zip(pc_indices, dapi_indices)
        if ious[pc_index, dapi_index] > 0.0
    ]
    if not matches:
        return _zero_match_summary()
    coverage = len(matches) / max(len(pc_labels), len(dapi_labels))
    rows: list[dict[str, Any]] = []
    for pc_index, dapi_index in matches:
        pc_label = pc_labels[pc_index]
        dapi_label = dapi_labels[dapi_index]
        intersection = int(intersections[pc_index, dapi_index])
        area_sum = pc_areas[pc_label] + dapi_areas[dapi_label]
        rows.append(
            {
                "pc_label": pc_label,
                "dapi_label": dapi_label,
                "dice": 2.0 * intersection / area_sum,
                "iou": float(ious[pc_index, dapi_index]),
                "area_ratio_pc_to_dapi": (
                    pc_areas[pc_label] / dapi_areas[dapi_label]
                ),
                "pc_feret_length": _feret_length(pc_array == pc_label),
                "dapi_feret_length": _feret_length(dapi_array == dapi_label),
                "matched_cell_coverage": coverage,
            }
        )
    return pd.DataFrame(rows, columns=NUCLEUS_COMPARISON_COLUMNS)


def run_development_nucleus_validation(
    config: Mapping[str, Any],
    cache_dir: str | Path,
) -> pd.DataFrame:
    """在明確 opt-in 後執行 B4 p6 PC/DAPI nucleus 開發期比較。

    Args:
        config: 含 ``development_validation`` 與可選 ``segmentation`` 的設定。
        cache_dir: feature cache 根目錄。

    Returns:
        僅供開發診斷使用、帶有固定 role 的逐配對比較表；未啟用時為空表。

    Raises:
        ValueError: 啟用時缺少必要設定或 sample size 不合法時拋出。
    """
    development_config = config.get("development_validation", {})
    if not isinstance(development_config, Mapping):
        raise ValueError("development_validation 必須是 mapping")
    if not bool(development_config.get("enabled", False)):
        return pd.DataFrame(columns=DEVELOPMENT_VALIDATION_COLUMNS)

    input_dir = development_config.get("input_dir")
    if not input_dir:
        raise ValueError("啟用 development_validation 時必須設定 input_dir")
    sample_size = int(development_config.get("sample_size", 10))
    if sample_size < 0:
        raise ValueError("development_validation.sample_size 不可小於 0")
    if sample_size == 0:
        return pd.DataFrame(columns=DEVELOPMENT_VALIDATION_COLUMNS)

    #DAPI 相關 import、路徑與模型只存在於這個 enabled 分支。
    from cellpose import io, models
    from immunity.build_dataset import build_manifest
    from ki67dtc.img_prep import (
        DAPI_NUC_MODEL_PATH,
        NUC_MODEL_INPUT_SIZE,
        PC_NUC_MODEL_PATH,
    )

    segmentation_config = _segmentation_config(config)
    use_gpu = str(segmentation_config.get("device", "gpu")).lower() != "cpu"
    threshold = float(segmentation_config.get("cellprob_threshold", 0.0))
    pc_model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=PC_NUC_MODEL_PATH,
    )
    dapi_model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=DAPI_NUC_MODEL_PATH,
    )
    manifest = build_manifest(input_dir).sort_values("image_key").head(sample_size)
    output_dir = Path(cache_dir) / "development_validation"
    rows: list[dict[str, Any]] = []

    for item in manifest.loc[:, ["image_key", "phase_path", "dapi_path"]].itertuples(
        index=False,
        name=None,
    ):
        image_key, pc_path_value, dapi_path_value = item
        pc_path = Path(pc_path_value)
        dapi_path = Path(dapi_path_value)
        pc_mask = _eval_label_mask(
            pc_model,
            np.asarray(io.imread(pc_path)),
            NUC_MODEL_INPUT_SIZE,
            threshold,
            channels=(0, 0),
        )
        dapi_mask = _eval_label_mask(
            dapi_model,
            np.asarray(io.imread(dapi_path)),
            NUC_MODEL_INPUT_SIZE,
            threshold,
            channels=(3, 3),
        )
        pc_mask, dapi_mask = _validate_mask_pair(pc_mask, dapi_mask)
        mask_path = output_dir / f"{image_key}.npz"
        _write_development_cache(mask_path, pc_mask, dapi_mask)
        comparison = compare_nucleus_masks(pc_mask, dapi_mask)
        for metrics in comparison.to_dict(orient="records"):
            rows.append(
                {
                    "image_key": str(image_key),
                    "pc_path": str(pc_path),
                    "dapi_path": str(dapi_path),
                    "mask_path": str(mask_path),
                    **metrics,
                    "role": "development_only_dapi_reference",
                }
            )
    return pd.DataFrame(rows, columns=DEVELOPMENT_VALIDATION_COLUMNS)


def extract_features_from_arrays(
    image_key: str,
    phase: np.ndarray,
    ido: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    max_nucleus_outside_fraction: float,
    enabled_feature_sets: Sequence[str] = ("basic_median",),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """從單張 synthetic 或實際影像陣列擷取 Primary features 與 IDO target。

    Args:
        image_key: 影像唯一識別碼。
        phase: 二維 phase-contrast 影像。
        ido: 與 phase 對齊的二維 IDO 影像。
        cell_mask: whole-cell label mask。
        nucleus_mask: PC nucleus label mask。
        max_nucleus_outside_fraction: nucleus 可落在配對 whole-cell 外的最大比例。
        enabled_feature_sets: 已啟用的 feature-set 名稱。

    Returns:
        通過 QC 的 cell rows，以及記錄配對、排除與背景值的 QC 字典。

    Raises:
        ValueError: 陣列不是相同尺寸的二維資料、閾值不合法、feature set 未註冊，
            或影像沒有可用的 whole-cell 外背景時拋出。
    """
    enabled = _validate_feature_set_specs(enabled_feature_sets)
    if not 0.0 <= float(max_nucleus_outside_fraction) <= 1.0:
        raise ValueError("max_nucleus_outside_fraction 必須介於 0 與 1")

    phase_array = np.asarray(phase)
    ido_array = np.asarray(ido, dtype=np.float64)
    cell_array, nucleus_array = _validate_mask_pair(cell_mask, nucleus_mask)
    arrays = (phase_array, ido_array, cell_array, nucleus_array)
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("phase、IDO 與 masks 都必須是二維陣列")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("phase、IDO 與 masks 必須具有相同尺寸")

    background_values = ido_array[(cell_array == 0) & np.isfinite(ido_array)]
    if background_values.size == 0:
        raise ValueError("IDO 影像沒有可計算 median 的 whole-cell 外背景")
    background_median = float(np.median(background_values))
    pairs = find_paired_labels(cell_array, nucleus_array)
    rows: list[dict[str, Any]] = []
    excluded_outside = 0
    excluded_empty = 0

    for cell_label, nucleus_label in pairs:
        cell_region = cell_array == cell_label
        nucleus_region = nucleus_array == nucleus_label
        nucleus_area = int(np.count_nonzero(nucleus_region))
        outside_area = int(np.count_nonzero(nucleus_region & ~cell_region))
        outside_fraction = _safe_divide(float(outside_area), float(nucleus_area))
        if not np.isfinite(outside_fraction) or (
            outside_fraction > float(max_nucleus_outside_fraction)
        ):
            excluded_outside += 1
            continue

        true_cytoplasm = cell_region & ~nucleus_region
        cytoplasm_area = int(np.count_nonzero(true_cytoplasm))
        if cytoplasm_area == 0:
            excluded_empty += 1
            continue
        ido_score = float(np.mean(ido_array[true_cytoplasm]) - background_median)
        row = {
            "image_key": str(image_key),
            "cell_label": int(cell_label),
            "nucleus_label": int(nucleus_label),
            **_geometry_values(phase_array, cell_region, "cell"),
            **_geometry_values(phase_array, nucleus_region, "nucleus"),
            "nucleus_cytoplasm_area_ratio": _safe_divide(
                float(nucleus_area), float(cytoplasm_area)
            ),
            "IDO_score": ido_score,
        }
        if "paper_style_median" in enabled:
            row.update(
                paper_style_extras(
                    phase_array,
                    cell_region,
                    nucleus_region,
                )
            )
        rows.append(row)

    qc = {
        "image_key": str(image_key),
        "paired_cells": len(pairs),
        "kept_full_cells": len(rows),
        "excluded_nucleus_outside": excluded_outside,
        "excluded_empty_cytoplasm": excluded_empty,
        "background_median": background_median,
    }
    return rows, qc


def extract_basic_cell_features(
    manifest: pd.DataFrame,
    segmentation_qc: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """從通過 segmentation QC 的影像建立 Exp3 Primary cell-level cache。

    Args:
        manifest: 含 ``image_key``、``pc_path`` 與 ``ido_path`` 的完整 manifest。
        segmentation_qc: ``cache_phase_masks`` 產生的逐影像 QC 與 mask path。
        config: 經 ``load_config`` 驗證的 Exp3 設定；使用 ``_output_dir``、
            ``segmentation`` 與 ``feature_sets.enabled``。

    Returns:
        cell-level feature 資料表，以及逐影像 feature extraction QC 表。

    Raises:
        ValueError: 輸入欄位缺漏、image key 重複或 feature 設定不合法時拋出。
    """
    required_manifest = {"image_key", "group_id", "pc_path", "ido_path"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"manifest 缺少欄位：{missing_manifest}")
    required_qc = {"image_key", "mask_path", "status"}
    missing_qc = sorted(required_qc - set(segmentation_qc.columns))
    if missing_qc:
        raise ValueError(f"segmentation_qc 缺少欄位：{missing_qc}")
    if manifest["image_key"].duplicated().any():
        raise ValueError("manifest 出現重複 image_key")
    if segmentation_qc["image_key"].duplicated().any():
        raise ValueError("segmentation_qc 出現重複 image_key")

    cache_dir = _feature_cache_dir_from_output(config.get("_output_dir"))

    segmentation_config = _segmentation_config(config)
    max_outside = float(
        segmentation_config.get("max_nucleus_outside_fraction", 0.05)
    )
    min_cells = int(segmentation_config.get("min_cells_per_image", 3))
    feature_config = config.get("feature_sets", {})
    enabled = (
        feature_config.get("enabled", ("basic_median",))
        if isinstance(feature_config, Mapping)
        else ("basic_median",)
    )
    enabled_names = _validate_feature_set_specs(enabled)

    qc_by_key = segmentation_qc.set_index(
        segmentation_qc["image_key"].astype(str), drop=False
    )
    cell_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for manifest_row in manifest.to_dict(orient="records"):
        image_key = _required_cache_component(
            manifest_row["image_key"], "image_key"
        )
        group_id = _required_cache_component(
            manifest_row["group_id"], "group_id"
        )
        empty_qc = {
            "image_key": image_key,
            "paired_cells": 0,
            "kept_full_cells": 0,
            "excluded_nucleus_outside": 0,
            "excluded_empty_cytoplasm": 0,
            "background_median": np.nan,
        }
        if image_key not in qc_by_key.index:
            qc_rows.append(
                {**empty_qc, "status": "failed", "error": "缺少 segmentation QC"}
            )
            continue
        mask_qc = qc_by_key.loc[image_key]
        if str(mask_qc["status"]) != "passed":
            qc_rows.append(
                {
                    **empty_qc,
                    "status": "failed",
                    "error": str(mask_qc.get("error", "segmentation 未通過")),
                }
            )
            continue
        mask_path = _validate_exp3_mask_cache_path(
            mask_qc["mask_path"],
            cache_dir,
            group_id,
            image_key,
        )
        try:
            phase = _read_grayscale_image(manifest_row["pc_path"])
            ido = _read_grayscale_image(manifest_row["ido_path"])
            cell_mask, nucleus_mask = load_cached_masks(mask_path)
            rows, metrics = extract_features_from_arrays(
                image_key,
                phase,
                ido,
                cell_mask,
                nucleus_mask,
                max_outside,
                enabled_names,
            )
            cell_rows.extend(rows)
            qc_rows.append({**metrics, "status": "passed", "error": ""})
        except Exception as error:  #逐圖記錄失敗，讓後續 minimum-cell gate 統一阻擋。
            qc_rows.append(
                {**empty_qc, "status": "failed", "error": str(error)}
            )

    cell_level_columns = list(CELL_LEVEL_COLUMNS)
    if "paper_style_median" in enabled_names:
        cell_level_columns[-1:-1] = list(PAPER_STYLE_EXTRA_FEATURES)
    cells = pd.DataFrame(cell_rows, columns=cell_level_columns)
    extraction_qc = pd.DataFrame(qc_rows, columns=FEATURE_EXTRACTION_QC_COLUMNS)
    cells.attrs["min_cells_per_image"] = min_cells
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomically(
        cells.loc[:, CELL_LEVEL_COLUMNS],
        _safe_output_file(cache_dir, "cell_level_basic.csv"),
    )
    cells.attrs["exp3_output_dir"] = str(cache_dir.parent)
    return cells, extraction_qc


def _geometry_values(
    signal: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """將既有 geometry helper 限縮成 Primary 指定的 16 欄。"""
    measured = _geometry_from_measurements(
        _measure_roi_with_python(signal, mask)
    )
    return {
        f"{prefix}__{name}": float(measured[name]) for name in BASIC_GEOMETRY
    }


def _read_grayscale_image(path: str | Path) -> np.ndarray:
    """讀取單張影像為二維浮點灰階陣列。"""
    image_path = Path(path)
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"無法讀取影像：{image_path}")
    return np.asarray(image, dtype=np.float64)


def _required_cache_component(value: object, name: str) -> str:
    """驗證組成 Exp3 mask cache 路徑的必要文字值。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必須是非空字串")
    if value != value.strip():
        raise ValueError(f"{name} 不可包含前後空白")
    if value in {".", ".."}:
        raise ValueError(f"{name} 不可使用相對路徑 component")
    drive, _ = ntpath.splitdrive(value)
    if (
        drive
        or ":" in value
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"{name} 不可包含路徑分隔符或 drive")
    return value


def _validate_exp3_cache_root(cache_dir: Path) -> Path:
    """驗證 feature cache root 未被 symlink／junction 重新導向。"""
    lexical = Path(os.path.abspath(cache_dir))
    if lexical.resolve(strict=False) != lexical:
        raise ValueError("mask_path cache root 不可透過 symlink 或 junction 改寫")
    return lexical


def _validate_exp3_mask_cache_path(
    value: object,
    cache_dir: Path,
    group_id: str,
    image_key: str,
) -> Path:
    """驗證 mask path 精確位於目前 Exp3 run 的 cache 結構。"""
    if not isinstance(value, (str, Path)):
        raise ValueError("mask_path 必須是非空路徑")
    if isinstance(value, str) and not value.strip():
        raise ValueError("mask_path 必須是非空路徑")
    try:
        group_id = _required_cache_component(group_id, "group_id")
        image_key = _required_cache_component(image_key, "image_key")
    except ValueError as error:
        raise ValueError(f"mask_path component 不合法：{error}") from error
    candidate = Path(value).resolve(strict=False)
    resolved_cache_dir = _validate_exp3_cache_root(cache_dir)
    masks_root = (resolved_cache_dir / "masks").resolve(strict=False)
    if masks_root.parent != resolved_cache_dir:
        raise ValueError("mask_path masks root 必須是 Exp3 feature cache 的直接子目錄")
    expected = (
        masks_root / group_id / f"{image_key}.npz"
    ).resolve(strict=False)
    if expected.parent.parent != masks_root:
        raise ValueError("mask_path resolved parent 必須位於 Exp3 masks root 下")
    if candidate != expected:
        raise ValueError(
            "mask_path 必須符合 Exp3 feature_cache/masks/<group_id>/"
            "<image_key>.npz 結構"
        )
    return candidate


def _canonical_json(value: Any) -> str:
    """將 cache provenance 序列化為禁止 NaN 的 canonical JSON。"""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("mask cache provenance 必須是 strict JSON") from error


def _sha256_text(value: str) -> str:
    """計算 UTF-8 文字的 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """以串流方式計算檔案內容 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_run_provenance(
    segmentation_config: Mapping[str, Any],
    segmenter: Segmenter,
) -> dict[str, Any]:
    """計算每次 cache 呼叫只需建立一次的 config/model provenance。"""
    relevant_config = {
        str(key): value
        for key, value in segmentation_config.items()
        if str(key) != "force"
    }
    config_json = _canonical_json(relevant_config)
    canonical_config = json.loads(config_json)
    class_identity = f"{type(segmenter).__module__}.{type(segmenter).__qualname__}"
    explicit_signature = getattr(segmenter, "cache_signature", None)
    if callable(explicit_signature):
        explicit_signature = explicit_signature()
    signature = (
        explicit_signature if explicit_signature is not None else class_identity
    )
    segmenter_evidence = {
        "class_identity": class_identity,
        "signature": signature,
    }
    signature_json = _canonical_json(segmenter_evidence)
    return {
        "schema_version": 1,
        "segmentation_config": canonical_config,
        "segmentation_config_hash": _sha256_text(config_json),
        "segmenter": json.loads(signature_json),
        "segmenter_signature_hash": _sha256_text(signature_json),
    }


def _cache_mismatch_reason(
    cached: Mapping[str, Any],
    current: Mapping[str, Any],
) -> str:
    """依可稽核欄位指出 cache 未重用的第一個原因。"""
    if cached.get("pc_sha256") != current.get("pc_sha256"):
        return "pc_content_changed"
    if cached.get("segmentation_config_hash") != current.get(
        "segmentation_config_hash"
    ):
        return "segmentation_config_changed"
    if cached.get("segmenter_signature_hash") != current.get(
        "segmenter_signature_hash"
    ):
        return "segmenter_signature_changed"
    if dict(cached) != dict(current):
        return "provenance_mismatch"
    return "provenance_match"


def _npz_scalar_text(cached: Any, name: str) -> str:
    """從 allow_pickle=False 的 NPZ 讀取單一文字 scalar。"""
    if name not in cached:
        raise KeyError(f"mask cache 缺少 provenance 欄位：{name}")
    value = np.asarray(cached[name])
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"mask cache {name} 必須是文字 scalar")
    scalar = value.item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def _load_mask_cache_entry(
    mask_path: Path,
) -> tuple[tuple[np.ndarray, np.ndarray], Mapping[str, Any]]:
    """載入 masks 並嚴格驗證 canonical provenance 與內容 hash。"""
    with np.load(mask_path, allow_pickle=False) as cached:
        if "cell_mask" not in cached or "nucleus_mask" not in cached:
            raise KeyError("mask cache 缺少 cell_mask 或 nucleus_mask")
        provenance_json = _npz_scalar_text(cached, "provenance_json")
        provenance_hash = _npz_scalar_text(cached, "provenance_hash")
        if provenance_hash != _sha256_text(provenance_json):
            raise ValueError("mask cache provenance hash 不符合內容")
        try:
            provenance = json.loads(
                provenance_json,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"不允許的 JSON constant：{value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("mask cache provenance 不是 strict JSON") from error
        if not isinstance(provenance, dict):
            raise ValueError("mask cache provenance 必須是 JSON object")
        if _canonical_json(provenance) != provenance_json:
            raise ValueError("mask cache provenance 必須使用 canonical JSON")
        masks = (
            np.asarray(cached["cell_mask"], dtype=np.int32),
            np.asarray(cached["nucleus_mask"], dtype=np.int32),
        )
    return masks, provenance


def _segmentation_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """取得直接或巢狀的 segmentation 設定。"""
    nested = config.get("segmentation")
    return nested if isinstance(nested, Mapping) else config


def _validate_mask_pair(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """驗證兩個 label masks 都是相同尺寸的二維陣列。"""
    first = np.asarray(first_mask)
    second = np.asarray(second_mask)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("兩個 mask 都必須是二維陣列")
    if first.shape != second.shape:
        raise ValueError("兩個 mask 必須具有相同尺寸")
    return (
        np.asarray(first, dtype=np.int32),
        np.asarray(second, dtype=np.int32),
    )


def _nonzero_labels(mask: np.ndarray) -> list[int]:
    """依數值順序回傳非背景 labels。"""
    return [int(label) for label in np.unique(mask) if int(label) != 0]


def _label_count(mask: np.ndarray) -> int:
    """計算不含背景的 label 數。"""
    return len(_nonzero_labels(mask))


def _feret_length(region: np.ndarray) -> float:
    """計算二值區域的最大 Feret diameter。"""
    properties = regionprops(np.asarray(region, dtype=np.uint8))
    return float(properties[0].feret_diameter_max) if properties else 0.0


def _zero_match_summary() -> pd.DataFrame:
    """建立 coverage 為零且不偽造配對 metrics 的 sentinel row。"""
    row = {column: pd.NA for column in NUCLEUS_COMPARISON_COLUMNS}
    row["matched_cell_coverage"] = 0.0
    return pd.DataFrame([row], columns=NUCLEUS_COMPARISON_COLUMNS)


def _write_mask_cache(
    mask_path: Path,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    provenance_json: str,
    provenance_hash: str,
) -> None:
    """以原子替換方式寫入單張 Primary mask cache。"""
    _write_npz_atomically(
        mask_path,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        provenance_json=np.asarray(provenance_json),
        provenance_hash=np.asarray(provenance_hash),
    )


def _write_development_cache(
    mask_path: Path,
    pc_mask: np.ndarray,
    dapi_mask: np.ndarray,
) -> None:
    """寫入 development-only PC 與 DAPI nucleus masks。"""
    _write_npz_atomically(
        mask_path,
        pc_nucleus_mask=pc_mask,
        dapi_nucleus_mask=dapi_mask,
    )


def _write_npz_atomically(mask_path: Path, **arrays: np.ndarray) -> None:
    """先完成暫存 ``.npz``，再原子替換目標 cache。"""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".npz",
            dir=mask_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
        os.replace(temporary_path, mask_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


__all__ = [
    "CELL_LEVEL_COLUMNS",
    "DEVELOPMENT_VALIDATION_COLUMNS",
    "FEATURE_EXTRACTION_QC_COLUMNS",
    "NUCLEUS_COMPARISON_COLUMNS",
    "SEGMENTATION_QC_COLUMNS",
    "PhaseSegmenter",
    "cache_phase_masks",
    "compare_nucleus_masks",
    "extract_basic_cell_features",
    "extract_features_from_arrays",
    "load_cached_masks",
    "paper_style_extras",
    "run_development_nucleus_validation",
]
