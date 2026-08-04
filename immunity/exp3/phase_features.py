"""提供 Exp3 的 PC-only segmentation、mask cache 與開發期 DAPI 驗證。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops


SEGMENTATION_QC_COLUMNS = [
    "image_key",
    "group_id",
    "mask_path",
    "cache_status",
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
    active_segmenter = segmenter or PhaseSegmenter(segmentation_config)
    force = bool(segmentation_config.get("force", False))
    cache_root = Path(cache_dir) / "masks"
    rows: list[dict[str, Any]] = []

    #明確選取三個 Primary 欄位，避免流程讀取或建構任何 DAPI 路徑。
    primary_manifest = manifest.loc[:, ["image_key", "group_id", "pc_path"]]
    for image_key_value, group_id_value, pc_path_value in primary_manifest.itertuples(
        index=False,
        name=None,
    ):
        image_key = str(image_key_value)
        group_id = str(group_id_value)
        pc_path = Path(pc_path_value)
        mask_path = cache_root / group_id / f"{image_key}.npz"
        whole_cell_labels = 0
        nucleus_labels = 0
        paired_cells = 0
        cache_status = "failed"

        try:
            existed = mask_path.exists()
            masks = None
            if existed and not force:
                try:
                    masks = load_cached_masks(mask_path)
                    cache_status = "reused"
                except (OSError, ValueError, KeyError):
                    #不可讀取或缺少欄位的 cache 不可重用，改以 PC 重新推論。
                    masks = None

            if masks is None:
                masks = active_segmenter.segment(pc_path)
                cache_status = "replaced" if existed else "created"

            cell_mask, nucleus_mask = _validate_mask_pair(*masks)
            whole_cell_labels = _label_count(cell_mask)
            nucleus_labels = _label_count(nucleus_mask)

            from ki67dtc.paired_overlay import find_paired_labels

            paired_cells = len(find_paired_labels(cell_mask, nucleus_mask))
            if paired_cells == 0:
                raise ValueError("配對細胞數為 0")

            if cache_status != "reused":
                _write_mask_cache(mask_path, cell_mask, nucleus_mask)
            status = "passed"
            error_text = ""
        except Exception as error:  #單張失敗必須保留 QC 並繼續處理下一張。
            status = "failed"
            error_text = str(error)
            cache_status = "failed"

        rows.append(
            {
                "image_key": image_key,
                "group_id": group_id,
                "mask_path": str(mask_path),
                "cache_status": cache_status,
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
        image-level matched-cell coverage。

    Raises:
        ValueError: masks 不是二維或尺寸不一致時拋出。
    """
    pc_array, dapi_array = _validate_mask_pair(pc_mask, dapi_mask)
    pc_labels = _nonzero_labels(pc_array)
    dapi_labels = _nonzero_labels(dapi_array)
    if not pc_labels or not dapi_labels:
        return pd.DataFrame(columns=NUCLEUS_COMPARISON_COLUMNS)

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


def _write_mask_cache(
    mask_path: Path,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> None:
    """以原子替換方式寫入單張 Primary mask cache。"""
    _write_npz_atomically(
        mask_path,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
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
    "DEVELOPMENT_VALIDATION_COLUMNS",
    "NUCLEUS_COMPARISON_COLUMNS",
    "SEGMENTATION_QC_COLUMNS",
    "PhaseSegmenter",
    "cache_phase_masks",
    "compare_nucleus_masks",
    "load_cached_masks",
    "run_development_nucleus_validation",
]
