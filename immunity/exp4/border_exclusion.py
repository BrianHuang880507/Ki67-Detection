"""Exp4 觸邊細胞排除、target 組裝與安全 CSV 輸出。

此模組是 Exp4 的資料組裝 seam：它讀取既有的 whole-cell mask，建立一個
與 ``cell_level_basic`` 原始列一一對齊的布林 inclusion mask，再由同一個 mask
產生所有後續模型會使用的細胞與 target。它不會修改 mask cache，也不會讀取
nucleus mask。
"""

from __future__ import annotations

import os
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data_snapshot import DEFAULT_GROUP_KEYS
from .cell_dedup import (
    DEDUP_REPORT_COLUMNS,
    CellDedupResult,
    NUCLEUS_FEATURE_COLUMNS,
    assert_full_rui49_feature_consistency,
    deduplicate_master_cells,
)
from .targets import (
    GROUP_TARGET_COLUMNS as HIGH_GROUP_TARGET_COLUMNS,
    FOV_IDO_SCORE_COLUMNS as HIGH_FOV_IDO_SCORE_COLUMNS,
    SENSITIVITY_REPORT_COLUMNS as HIGH_SENSITIVITY_REPORT_COLUMNS,
    SparseFovSensitivityResult,
    TargetAggregationResult,
    build_exp4_targets,
    evaluate_sparse_fov_sensitivity,
)


BORDER_REPORT_COLUMNS = (
    "image_key",
    "b_id",
    "passage",
    "group_id",
    "cells_before",
    "cells_excluded_border",
    "cells_after",
    "whole_cell_labels_before",
    "whole_cell_labels_excluded_border",
    "whole_cell_labels_after",
    "border_exclusion_fraction",
    "border_exclusion_row_fraction",
    "minimum_cell_gate_status",
    # Raw master-row counts are observations only. All ``cells_*`` fields above
    # are distinct whole-cell counts after F0 deduplication.
    "raw_pair_cells_before",
    "raw_pair_cells_excluded_border",
    "raw_pair_cells_after",
    "raw_pair_exclusion_fraction",
)

# The high-owned modules are the only authority for target/sensitivity schemas.
# Keep these aliases for existing callers without manufacturing columns here.
GROUP_TARGET_COLUMNS = HIGH_GROUP_TARGET_COLUMNS
GROUP_TARGET_SENSITIVITY_COLUMNS = HIGH_SENSITIVITY_REPORT_COLUMNS


class BorderExclusionError(ValueError):
    """觸邊排除的輸入或 fail-closed gate 不符合規格。"""


@dataclass(frozen=True)
class BorderExclusionResult:
    """保存排除前後資料與同一個 inclusion mask 的量測結果。

    Attributes:
        manifest: 經過正規化且驗證過的 manifest，保留輸入列順序。
        raw_cells: raw ``cell_level_basic.csv`` rows (the 23976-row snapshot).
        pre_cells: F0-deduplicated distinct whole-cell frame.
        eligible_cells: 以 ``inclusion_mask`` 篩出的 distinct whole-cell rows。
        excluded_cells: 觸邊而被排除的 distinct whole-cell rows。
        inclusion_mask: 與 ``pre_cells`` 列順序完全相同的布林陣列。
        excluded_indices: ``pre_cells`` 中被排除列的原始 index labels。
        excluded_composite_keys: 被排除的 ``(image_key, cell_label)`` keys。
        border_report: 每個 manifest FOV 一列的觸邊統計。
        expected_group_keys: 此結果允許的 exact group 順序。
    """

    manifest: pd.DataFrame
    raw_cells: pd.DataFrame
    pre_cells: pd.DataFrame
    eligible_cells: pd.DataFrame
    excluded_cells: pd.DataFrame
    inclusion_mask: np.ndarray
    excluded_indices: tuple[object, ...]
    excluded_composite_keys: tuple[tuple[str, int], ...]
    border_report: pd.DataFrame
    expected_group_keys: tuple[str, ...]
    minimum_cell_gate_status: str
    expected_sparse_fov_keys: tuple[str, ...] | None = None
    dedup_report: pd.DataFrame | None = None
    raw_inclusion_mask: np.ndarray | None = None
    duplicate_key_count: int = 0
    duplicate_excess_row_count: int = 0
    mask_only_label_total: int = 0
    mask_only_label_max_per_fov: int = 0


def find_border_labels(cell_mask: np.ndarray) -> set[int]:
    """找出出現在 mask 四條影像邊界上的 positive cell labels。

    Args:
        cell_mask: 二維整數 whole-cell label mask；0 代表背景。

    Returns:
        出現在第 0／最後列或第 0／最後行的 positive labels。

    Raises:
        BorderExclusionError: mask 不是非空二維整數陣列。
    """
    labels = np.asarray(cell_mask)
    if labels.ndim != 2 or labels.size == 0:
        raise BorderExclusionError("cell_mask 必須是非空二維陣列")
    if not np.issubdtype(labels.dtype, np.integer):
        raise BorderExclusionError("cell_mask 必須是整數 label mask")
    if np.any(labels < 0):
        raise BorderExclusionError("cell_mask 不可含負數 label")
    edge_values = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    return {int(value) for value in np.unique(edge_values) if int(value) > 0}


def apply_border_exclusion(
    manifest: pd.DataFrame,
    cells: pd.DataFrame,
    masks_dir: str | Path,
    *,
    expected_manifest_keys: int = 693,
    expected_cell_rows: int = 23976,
    expected_cell_keys: int = 693,
    expected_dedup_cells: int | None = None,
    expected_duplicate_keys: int | None = None,
    expected_sparse_fov_keys: Sequence[str] | None = None,
    expected_group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cells_per_fov: int | None = None,
    deduplicated_cells: pd.DataFrame | None = None,
    dedup_report: pd.DataFrame | None = None,
) -> BorderExclusionResult:
    """驗證輸入並建立全 pipeline 共用的觸邊 inclusion mask。

    Args:
        manifest: Exp3 manifest，至少含 ``image_key``、``b_id``、``passage``、
            ``group_id``。
        cells: Exp3 ``cell_level_basic`` raw rows，至少含 ``image_key``、
            ``cell_label``、``nucleus_label``、``IDO_score``。此 frame 用於
            23976-row snapshot gate；border seam 僅使用 F0 去重後 frame。
        masks_dir: 既有 mask cache 根目錄；只讀取 manifest keys 對應的 `.npz`。
        expected_manifest_keys: 預期 manifest unique image keys。
        expected_cell_rows: 預期排除前 cell row 數。
        expected_cell_keys: 預期 cell distinct image keys。
        expected_dedup_cells: optional F0 distinct whole-cell lock。
        expected_duplicate_keys: optional duplicate whole-cell-key lock。
        expected_sparse_fov_keys: 正式 v4 的 exact ``cells_after == 2`` FOV roster。
        expected_group_keys: 允許的 exact group ids；正式執行為 9 組。
        min_cells_per_fov: 僅保留舊 API 相容性；v4 不阻擋稀疏 FOV。
        deduplicated_cells: optional high ``cell_dedup`` output。

    Returns:
        同時帶有 raw snapshot、distinct whole-cell rows、eligible rows、
        excluded keys、report 與 inclusion mask。輸入 snapshot/label gate
        失敗會 raise；所有 post-border FOV（含 2-cell FOV）都保留。

    Raises:
        BorderExclusionError: 任一 snapshot、label、finite target 或 post-count
            gate 不符合規格。
    """
    manifest_frame = _normalise_manifest(manifest)
    raw_cells = _normalise_cells(cells)
    expected_groups = tuple(str(key) for key in expected_group_keys)
    sparse_expected = _normalise_sparse_fov_keys(expected_sparse_fov_keys)
    if min_cells_per_fov is not None and min_cells_per_fov < 1:
        raise BorderExclusionError("min_cells_per_fov 必須是正整數")

    _validate_snapshot_counts(
        manifest_frame,
        raw_cells,
        expected_manifest_keys=expected_manifest_keys,
        expected_cell_rows=expected_cell_rows,
        expected_cell_keys=expected_cell_keys,
        expected_group_keys=expected_groups,
    )
    cells_frame, resolved_dedup_report = _deduplicate_for_border(
        raw_cells,
        deduplicated_cells=deduplicated_cells,
        dedup_report=dedup_report,
        expected_raw_rows=expected_cell_rows,
        expected_distinct_cells=expected_dedup_cells,
        expected_multirow_keys=expected_duplicate_keys,
    )
    actual_dedup_cells = int(len(cells_frame))
    # v4 locks duplicate *keys* (938), while the raw-to-dedup row difference
    # (964) is a separate observation because 26 keys have three nuclei rows.
    actual_duplicate_keys = int(len(resolved_dedup_report))
    duplicate_excess_rows = int(len(raw_cells) - actual_dedup_cells)
    if expected_dedup_cells is not None and actual_dedup_cells != int(expected_dedup_cells):
        raise BorderExclusionError(
            f"dedup distinct whole-cell count expected {expected_dedup_cells}, "
            f"actual {actual_dedup_cells}"
        )
    if expected_duplicate_keys is not None and actual_duplicate_keys != int(expected_duplicate_keys):
        raise BorderExclusionError(
            f"duplicate whole-cell key count expected {expected_duplicate_keys}, "
            f"actual {actual_duplicate_keys}"
        )
    cells_frame = _normalise_cells(cells_frame, allow_duplicate_master_keys=False)
    _validate_distinct_cells(cells_frame)
    manifest_key_set = set(manifest_frame["image_key"])
    dedup_key_set = set(cells_frame["image_key"])
    if dedup_key_set != manifest_key_set:
        raise BorderExclusionError(
            "dedup distinct cells 與 manifest image_key 集合不一致: "
            f"manifest_only={sorted(manifest_key_set - dedup_key_set)}, "
            f"dedup_only={sorted(dedup_key_set - manifest_key_set)}"
        )
    mask_paths = _mask_paths_by_stem(Path(masks_dir))
    manifest_keys = manifest_frame["image_key"].tolist()
    missing_masks = sorted(set(manifest_keys) - set(mask_paths))
    if missing_masks:
        raise BorderExclusionError(
            "manifest key 找不到 mask: " + ", ".join(missing_masks)
        )
    ambiguous_masks = sorted(
        key for key, paths in mask_paths.items() if key in set(manifest_keys) and len(paths) != 1
    )
    if ambiguous_masks:
        raise BorderExclusionError(
            "manifest key 對應多個 mask `.npz`: " + ", ".join(ambiguous_masks)
        )

    inclusion = np.ones(len(cells_frame), dtype=bool)
    report_rows: list[dict[str, object]] = []
    mask_only_label_counts: list[int] = []
    for manifest_row in manifest_frame.itertuples(index=False):
        image_key = str(manifest_row.image_key)
        mask = _load_cell_mask(mask_paths[image_key][0])
        positive_labels = {
            int(value) for value in np.unique(mask) if int(value) > 0
        }
        image_rows = cells_frame["image_key"].eq(image_key)
        raw_image_rows = raw_cells["image_key"].eq(image_key)
        csv_labels = set(cells_frame.loc[image_rows, "cell_label"].astype(int))
        csv_only_labels = csv_labels - positive_labels
        mask_only_labels = positive_labels - csv_labels
        if csv_only_labels:
            raise BorderExclusionError(
                f"{image_key} CSV cell labels 不在 mask positive labels: "
                f"csv_only={sorted(csv_only_labels)}"
            )
        mask_only_label_counts.append(len(mask_only_labels))

        # Mask-only objects are QC-excluded and must not enter Exp4 counts.
        border_labels = find_border_labels(mask) & csv_labels
        border_rows = image_rows & cells_frame["cell_label"].isin(border_labels)
        inclusion[border_rows.to_numpy()] = False
        cells_before = int(image_rows.sum())
        cells_excluded = int(border_rows.sum())
        cells_after = cells_before - cells_excluded
        whole_cell_labels_before = int(cells_frame.loc[image_rows, "cell_label"].nunique())
        whole_cell_labels_excluded = int(
            cells_frame.loc[border_rows, "cell_label"].nunique()
        )
        whole_cell_labels_after = whole_cell_labels_before - whole_cell_labels_excluded
        raw_cells_before = int(raw_image_rows.sum())
        raw_cells_excluded = int(
            (raw_image_rows & raw_cells["cell_label"].isin(border_labels)).sum()
        )
        raw_cells_after = raw_cells_before - raw_cells_excluded
        report_rows.append(
            {
                "image_key": image_key,
                "b_id": str(manifest_row.b_id),
                "passage": int(manifest_row.passage),
                "group_id": str(manifest_row.group_id),
                "cells_before": cells_before,
                "cells_excluded_border": cells_excluded,
                "cells_after": cells_after,
                "whole_cell_labels_before": whole_cell_labels_before,
                "whole_cell_labels_excluded_border": whole_cell_labels_excluded,
                "whole_cell_labels_after": whole_cell_labels_after,
                "border_exclusion_fraction": (
                    float(whole_cell_labels_excluded / whole_cell_labels_before)
                    if whole_cell_labels_before
                    else np.nan
                ),
                "border_exclusion_row_fraction": (
                    float(cells_excluded / cells_before) if cells_before else np.nan
                ),
                "raw_pair_cells_before": raw_cells_before,
                "raw_pair_cells_excluded_border": raw_cells_excluded,
                "raw_pair_cells_after": raw_cells_after,
                "raw_pair_exclusion_fraction": (
                    float(raw_cells_excluded / raw_cells_before)
                    if raw_cells_before
                    else np.nan
                ),
            }
        )

    border_report = pd.DataFrame(report_rows, columns=BORDER_REPORT_COLUMNS)
    if border_report.empty or len(border_report) != len(manifest_frame):
        raise BorderExclusionError("border report 必須覆蓋每個 manifest FOV")
    if not np.isfinite(
        border_report["border_exclusion_fraction"].to_numpy(dtype=float)
    ).all():
        raise BorderExclusionError("border exclusion fraction 必須是 finite")
    # v4 explicitly retains all manifest FOVs, including FOVs with only two
    # distinct whole cells after border exclusion. Sparse-FOV sensitivity is a
    # target-side diagnostic, not a border hard gate.
    minimum_cell_gate_status = "passed"
    border_report["minimum_cell_gate_status"] = minimum_cell_gate_status
    inferred_sparse = tuple(
        border_report.loc[
            border_report["cells_after"].astype(int).eq(2), "image_key"
        ].astype(str)
    )
    if sparse_expected is not None and inferred_sparse != sparse_expected:
        raise BorderExclusionError(
            "sparse FOV roster 必須等於 inferred cells_after==2 roster: "
            f"expected={list(sparse_expected)}, actual={list(inferred_sparse)}"
        )

    excluded_indices = tuple(cells_frame.index[~inclusion].tolist())
    excluded_keys = tuple(
        dict.fromkeys(
            (str(row.image_key), int(row.cell_label))
            for row in cells_frame.loc[
                ~inclusion, ["image_key", "cell_label"]
            ].itertuples(index=False)
        )
    )
    return BorderExclusionResult(
        manifest=manifest_frame,
        raw_cells=raw_cells,
        pre_cells=cells_frame,
        eligible_cells=cells_frame.loc[inclusion].copy(),
        excluded_cells=cells_frame.loc[~inclusion].copy(),
        inclusion_mask=inclusion,
        excluded_indices=excluded_indices,
        excluded_composite_keys=excluded_keys,
        border_report=border_report,
        expected_group_keys=expected_groups,
        minimum_cell_gate_status=minimum_cell_gate_status,
        expected_sparse_fov_keys=sparse_expected,
        dedup_report=resolved_dedup_report,
        raw_inclusion_mask=_raw_inclusion_mask(raw_cells, cells_frame, inclusion),
        duplicate_key_count=actual_duplicate_keys,
        duplicate_excess_row_count=duplicate_excess_rows,
        mask_only_label_total=int(sum(mask_only_label_counts)),
        mask_only_label_max_per_fov=int(max(mask_only_label_counts, default=0)),
    )


def apply_inclusion_mask(frame: pd.DataFrame, inclusion_mask: Sequence[bool]) -> pd.DataFrame:
    """以既有 inclusion mask 篩選另一個與 cell rows 對齊的 frame。

    Args:
        frame: 與 ``BorderExclusionResult.pre_cells`` 相同列順序的 X 或 y frame。
        inclusion_mask: 排除結果提供的布林 mask。

    Returns:
        套用同一 mask 的新資料表。

    Raises:
        BorderExclusionError: mask 長度與 frame 列數不一致。
    """
    mask = np.asarray(inclusion_mask, dtype=bool)
    if len(frame) != len(mask):
        raise BorderExclusionError(
            f"inclusion mask 長度 {len(mask)} 與 frame rows {len(frame)} 不一致"
        )
    return frame.loc[mask].copy()


def build_target_aggregation(
    result: BorderExclusionResult,
    *,
    expected_pre_border_cells: int | None = None,
    expected_post_border_cells: int | None = None,
    expected_fov_count: int | None = None,
) -> TargetAggregationResult:
    """直接呼叫 high target API，並只驗證其 canonical result schema。

    Args:
        result: 已通過 snapshot、dedup 與 border gates 的結果。
        expected_pre_border_cells: optional fail-closed F0 distinct-cell lock；
            正式 safe run 應直接傳入 config lock，不由實際值回填。
        expected_post_border_cells: optional fail-closed post-border distinct-cell
            lock；正式 safe run 應直接傳入 config lock。
        expected_fov_count: optional fail-closed manifest FOV lock。

    Returns:
        high-owned TargetAggregationResult；其中 FOV_IDO 僅供 QC。

    Raises:
        BorderExclusionError: high result 不符合 exact schema 或計數不一致。
    """
    _require_columns(result.pre_cells, ("image_key", "IDO_score"), "pre_cells")
    _require_columns(
        result.eligible_cells,
        ("image_key", "IDO_score"),
        "eligible_cells",
    )
    pre_expected = (
        len(result.pre_cells)
        if expected_pre_border_cells is None
        else int(expected_pre_border_cells)
    )
    post_expected = (
        len(result.eligible_cells)
        if expected_post_border_cells is None
        else int(expected_post_border_cells)
    )
    fov_expected = (
        len(result.manifest)
        if expected_fov_count is None
        else int(expected_fov_count)
    )
    try:
        target_result = build_exp4_targets(
            result.pre_cells,
            result.manifest,
            post_border_cells=result.eligible_cells,
            expected_pre_border_cells=pre_expected,
            expected_post_border_cells=post_expected,
            expected_fov_count=fov_expected,
            expected_group_keys=result.expected_group_keys,
        )
    except Exception as error:
        raise BorderExclusionError(f"build_exp4_targets failed: {error}") from error
    if not isinstance(target_result, TargetAggregationResult):
        raise BorderExclusionError(
            "build_exp4_targets 必須回傳 TargetAggregationResult"
        )
    _validate_high_target_schema(target_result, result.expected_group_keys)
    return target_result


def build_group_targets(result: BorderExclusionResult) -> pd.DataFrame:
    """回傳 high target result 的 canonical group table。"""
    return build_target_aggregation(result).group_targets.copy()


def build_group_target_sensitivity_result(
    result: BorderExclusionResult,
    target_result: TargetAggregationResult | None = None,
) -> SparseFovSensitivityResult:
    """直接呼叫 high sparse-FOV API，保留可延後執行的 blocked gate。

    Args:
        result: 已驗證的 border result。
        target_result: 可重用的 high target result；省略時重新建立。

    Returns:
        high-owned SparseFovSensitivityResult，其 report 可先寫出。

    Raises:
        BorderExclusionError: sparse roster 或 high result schema 不符合規格。
    """
    aggregation = target_result or build_target_aggregation(result)
    sparse_keys = _resolved_sparse_fov_keys(result)
    try:
        sensitivity_result = evaluate_sparse_fov_sensitivity(
            aggregation,
            sparse_keys,
            expected_sparse_fov_count=len(sparse_keys),
            expected_sparse_cell_count=2,
            general_threshold_pct=5.0,
            low_confidence_group="B8_P7",
            low_confidence_threshold_pct=15.0,
        )
    except Exception as error:
        raise BorderExclusionError(
            f"evaluate_sparse_fov_sensitivity failed: {error}"
        ) from error
    if not isinstance(sensitivity_result, SparseFovSensitivityResult):
        raise BorderExclusionError(
            "evaluate_sparse_fov_sensitivity 必須回傳 SparseFovSensitivityResult"
        )
    _validate_high_sensitivity_schema(
        sensitivity_result.report,
        result.expected_group_keys,
    )
    return sensitivity_result


def build_group_target_sensitivity(result: BorderExclusionResult) -> pd.DataFrame:
    """回傳 high sensitivity result 的 canonical report；不在此執行 gate。"""
    return build_group_target_sensitivity_result(result).report.copy()


def _validate_high_target_schema(
    target_result: TargetAggregationResult,
    expected_group_keys: Sequence[str],
) -> None:
    """拒絕任何由 adapter 合成或改名的 target 欄位。"""
    frame = target_result.group_targets
    expected = tuple(str(key) for key in expected_group_keys)
    if tuple(frame.columns) != tuple(HIGH_GROUP_TARGET_COLUMNS):
        raise BorderExclusionError(
            "high group target schema 不符: "
            f"expected={list(HIGH_GROUP_TARGET_COLUMNS)}, actual={list(frame.columns)}"
        )
    if tuple(frame["group_id"].astype(str)) != expected:
        raise BorderExclusionError("high group target 順序/集合不符 expected groups")


def _validate_high_sensitivity_schema(
    report: pd.DataFrame,
    expected_group_keys: Sequence[str],
) -> None:
    """拒絕任何由 adapter 合成、改名或補值的 sensitivity 欄位。"""
    expected = tuple(str(key) for key in expected_group_keys)
    if tuple(report.columns) != tuple(HIGH_SENSITIVITY_REPORT_COLUMNS):
        raise BorderExclusionError(
            "high sensitivity schema 不符: "
            f"expected={list(HIGH_SENSITIVITY_REPORT_COLUMNS)}, actual={list(report.columns)}"
        )
    if tuple(report["group_id"].astype(str)) != expected:
        raise BorderExclusionError("high sensitivity 順序/集合不符 expected groups")


def _resolved_sparse_fov_keys(result: BorderExclusionResult) -> tuple[str, ...]:
    """回傳並驗證 inferred roster 與 production exact roster。"""
    inferred = tuple(
        result.border_report.loc[
            result.border_report["cells_after"].astype(int).eq(2),
            "image_key",
        ].astype(str)
    )
    expected = result.expected_sparse_fov_keys
    if expected is not None and inferred != expected:
        raise BorderExclusionError(
            "sparse FOV roster 必須等於 inferred cells_after==2 roster"
        )
    return expected if expected is not None else inferred


def assemble_xy(
    result: BorderExclusionResult,
    *,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """由 eligible cells 組出共享同一 border mask 的 X 與 group-level y。

    Args:
        result: 已驗證且完成觸邊排除的結果。
        feature_columns: 要放入 X 的欄位；IDO target 不應列入正式 predictors。

    Returns:
        ``(X, y)``，兩者列索引與 eligible cells 一致，border rows 不存在於任一者。

    Raises:
        BorderExclusionError: feature 欄位缺少或 group target 對不上。
    """
    columns = tuple(feature_columns)
    _require_columns(result.eligible_cells, columns, "eligible_cells")
    targets = build_group_targets(result)
    return _assemble_xy_frame(result, result.eligible_cells, targets, columns)


def _assemble_xy_frame(
    result: BorderExclusionResult,
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    image_to_group = result.manifest.set_index("image_key")["group_id"]
    group_to_target = targets.set_index("group_id")["group_IDO_score"]
    rows = frame.copy()
    rows["group_id"] = rows["image_key"].map(image_to_group)
    y = rows["group_id"].map(group_to_target)
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise BorderExclusionError("eligible cells 無法映射 finite group_IDO_score")
    return rows.loc[:, columns].copy(), pd.Series(
        y.to_numpy(dtype=float), index=rows.index, name="group_IDO_score"
    )


def join_label_features(
    result: BorderExclusionResult,
    label_features: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """將每 label 特徵以 many-to-one join 回 master cell rows。

    Args:
        result: 已驗證的 border 結果；其 ``pre_cells`` 可能含同一
            ``(image_key, cell_label)`` 的多個 nucleus rows。
        label_features: 每個 ``(image_key, cell_label)`` 恰一列的 feature table。
        feature_columns: 要檢查不可缺失的 feature 欄位；省略時檢查所有非 key 欄位。

    Returns:
        與排除前 master rows 等長、保留原列順序的 joined frame。

    Raises:
        BorderExclusionError: feature keys 重複、缺少 master label，或欄位不完整。
    """
    keys = ("image_key", "cell_label")
    _require_columns(result.pre_cells, keys, "pre_cells")
    _require_columns(label_features, keys, "label_features")
    features = label_features.copy()
    feature_image_keys = features["image_key"].astype(str).str.strip()
    try:
        feature_labels = pd.to_numeric(features["cell_label"], errors="raise")
    except (TypeError, ValueError) as error:
        raise BorderExclusionError("label_features cell_label 必須是整數") from error
    if (
        feature_image_keys.eq("").any()
        or not np.isfinite(feature_labels.to_numpy(dtype=float)).all()
        or not np.equal(feature_labels, np.floor(feature_labels)).all()
    ):
        raise BorderExclusionError(
            "label_features 必須對每個 (image_key, cell_label) 提供唯一 finite key"
        )
    features["image_key"] = feature_image_keys
    features["cell_label"] = feature_labels.astype(int)
    feature_key = pd.MultiIndex.from_arrays(
        [features["image_key"], features["cell_label"]],
        names=keys,
    )
    if feature_key.duplicated().any():
        raise BorderExclusionError(
            "label_features 必須對每個 (image_key, cell_label) 提供唯一 finite key"
        )
    master_key_set = {
        (str(row.image_key), int(row.cell_label))
        for row in result.pre_cells.loc[:, list(keys)].drop_duplicates().itertuples(
            index=False
        )
    }
    feature_key_set = {
        (str(row.image_key), int(row.cell_label))
        for row in features.loc[:, list(keys)].itertuples(index=False)
    }
    if feature_key_set != master_key_set:
        raise BorderExclusionError(
            "label_features key 集合必須與 master distinct keys 完全相同: "
            f"extra={sorted(feature_key_set - master_key_set)}, "
            f"missing={sorted(master_key_set - feature_key_set)}"
        )
    if feature_columns is None:
        value_columns = tuple(column for column in features.columns if column not in keys)
    else:
        value_columns = tuple(feature_columns)
    _require_columns(features, value_columns, "label_features")
    merged = result.pre_cells.merge(
        features,
        on=list(keys),
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_feature"),
    )
    missing = merged.loc[merged["_merge"].eq("left_only"), list(keys)]
    if not missing.empty:
        raise BorderExclusionError(
            "label_features 缺少 master labels: "
            + ", ".join(
                f"({row.image_key}, {int(row.cell_label)})"
                for row in missing.itertuples(index=False)
            )
        )
    if len(merged) != len(result.pre_cells):
        raise BorderExclusionError(
            f"many-to-one join row count changed: expected {len(result.pre_cells)}, "
            f"actual {len(merged)}"
        )
    original_master_key = pd.MultiIndex.from_frame(
        result.pre_cells.loc[:, ["image_key", "cell_label", "nucleus_label"]]
    )
    joined_master_key = pd.MultiIndex.from_frame(
        merged.loc[:, ["image_key", "cell_label", "nucleus_label"]]
    )
    if not joined_master_key.equals(original_master_key):
        raise BorderExclusionError(
            "many-to-one join 改變 master 三鍵順序或 identity"
        )
    if value_columns and merged.loc[:, list(value_columns)].isna().any().any():
        missing_columns = merged.loc[:, list(value_columns)].columns[
            merged.loc[:, list(value_columns)].isna().any()
        ].tolist()
        raise BorderExclusionError(
            f"joined feature values 不可缺失: {missing_columns}"
        )
    return merged.drop(columns="_merge")


def assemble_feature_xy(
    result: BorderExclusionResult,
    label_features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """以 many-to-one label feature join 加上同一 border mask 組出 X/y。

    Args:
        result: 已驗證的排除結果。
        label_features: 一列一 whole-cell label 的 feature table。
        feature_columns: 放入 X 的欄位。

    Returns:
        只含 eligible master rows 的 ``(X, y)``。
    """
    joined = join_label_features(
        result,
        label_features,
        feature_columns=feature_columns,
    )
    _assert_rui49_feature_join_if_requested(result, label_features, feature_columns)
    eligible = apply_inclusion_mask(joined, result.inclusion_mask)
    targets = build_group_targets(result)
    columns = tuple(feature_columns)
    _require_columns(eligible, columns, "joined eligible cells")
    return _assemble_xy_frame(result, eligible, targets, columns)


def _assert_rui49_feature_join_if_requested(
    result: BorderExclusionResult,
    label_features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    """在真正 Rui49 join seam 檢查 raw multirow whole-cell consistency。"""
    from .rui_features import RUI49_FEATURE_COLUMNS

    if tuple(feature_columns) != tuple(RUI49_FEATURE_COLUMNS):
        return
    raw = result.raw_cells.loc[:, ["image_key", "cell_label"]].copy()
    features = label_features.loc[
        :, ["image_key", "cell_label", *RUI49_FEATURE_COLUMNS]
    ]
    joined = raw.merge(
        features,
        on=["image_key", "cell_label"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    try:
        assert_full_rui49_feature_consistency(
            joined,
            scope="full_join_693_fovs",
        )
    except Exception as error:
        raise BorderExclusionError(
            f"full Rui49 feature consistency gate failed: {error}"
        ) from error


def write_border_exclusion_report(
    result: BorderExclusionResult, path: str | Path
) -> None:
    """在完整驗證後以 atomic replace 寫出 border report CSV。"""
    if len(result.border_report) != len(result.manifest):
        raise BorderExclusionError("border report 不完整，拒絕寫出")
    _write_csv_atomic(result.border_report, BORDER_REPORT_COLUMNS, Path(path))


def write_group_targets(
    targets: pd.DataFrame,
    path: str | Path,
    *,
    expected_group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> None:
    """驗證 exact groups 後以 atomic replace 寫出 group targets CSV。

    Args:
        targets: ``build_group_targets`` 產生的九組 target table。
        path: 輸出 CSV 路徑。
        expected_group_keys: 本次 snapshot 允許的 exact groups；unit fixture 可
            傳入較小的明確集合，正式 Exp4 預設為九組。

    Raises:
        BorderExclusionError: schema、row count、group keys 或 finite values 不符。
    """
    if list(targets.columns) != list(GROUP_TARGET_COLUMNS):
        raise BorderExclusionError(
            f"group targets schema 不符: expected {list(GROUP_TARGET_COLUMNS)}, "
            f"actual {list(targets.columns)}"
        )
    expected_groups = tuple(sorted(str(key) for key in expected_group_keys))
    actual_groups = tuple(sorted(targets["group_id"].astype(str))) if "group_id" in targets else ()
    if len(targets) != len(expected_groups) or actual_groups != expected_groups:
        raise BorderExclusionError(
            f"group targets row/group keys 不符: expected={list(expected_groups)}, "
            f"actual={list(actual_groups)}"
        )
    if targets.empty or not np.isfinite(
        targets[["group_IDO_score", "group_IDO_score_all_cells", "delta"]]
        .to_numpy(dtype=float)
    ).all():
        raise BorderExclusionError("group targets 不可為空或含非 finite 值")
    _write_csv_atomic(targets, GROUP_TARGET_COLUMNS, Path(path))


def write_fov_ido_scores(
    target_result: TargetAggregationResult,
    path: str | Path,
    *,
    expected_fov_count: int,
) -> None:
    """以 atomic replace 寫出 high target result 的 descriptive FOV QC。"""
    scores = target_result.fov_ido_scores
    if tuple(scores.columns) != tuple(HIGH_FOV_IDO_SCORE_COLUMNS):
        raise BorderExclusionError(
            "FOV_IDO_score schema 不符: "
            f"expected={list(HIGH_FOV_IDO_SCORE_COLUMNS)}, actual={list(scores.columns)}"
        )
    if len(scores) != int(expected_fov_count):
        raise BorderExclusionError(
            f"FOV_IDO_score rows expected {expected_fov_count}, actual {len(scores)}"
        )
    if scores.isna().any().any() or not np.isfinite(
        pd.to_numeric(scores["FOV_IDO_score"], errors="coerce")
        .to_numpy(dtype=float)
    ).all():
        raise BorderExclusionError("FOV_IDO_score 不可含非 finite 值")
    _write_csv_atomic(scores, HIGH_FOV_IDO_SCORE_COLUMNS, Path(path))


def write_group_target_sensitivity(
    sensitivity: pd.DataFrame,
    path: str | Path,
    *,
    expected_group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> None:
    """驗證 9 組 sparse-FOV evidence 後 atomic 寫出 sensitivity CSV。"""
    if list(sensitivity.columns) != list(GROUP_TARGET_SENSITIVITY_COLUMNS):
        raise BorderExclusionError(
            "group sensitivity schema 不符: "
            f"expected={list(GROUP_TARGET_SENSITIVITY_COLUMNS)}, "
            f"actual={list(sensitivity.columns)}"
        )
    expected = tuple(sorted(str(key) for key in expected_group_keys))
    actual = tuple(sorted(sensitivity["group_id"].astype(str)))
    if len(sensitivity) != len(expected) or actual != expected:
        raise BorderExclusionError(
            f"group sensitivity row/group keys 不符: expected={list(expected)}, "
            f"actual={list(actual)}"
        )
    numeric = sensitivity[
        [
            "target_with_sparse_fovs",
            "target_without_sparse_fovs",
            "delta_without_minus_with",
            "absolute_delta",
            "baseline_target_range",
            "absolute_delta_pct_of_range",
            "threshold_pct",
            "threshold_absolute",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise BorderExclusionError("group sensitivity numeric evidence 不可含非 finite")
    _write_csv_atomic(
        sensitivity,
        GROUP_TARGET_SENSITIVITY_COLUMNS,
        Path(path),
    )


def write_cell_dedup_report(
    report: pd.DataFrame,
    path: str | Path,
    *,
    expected_duplicate_keys: int | None = None,
) -> None:
    """以 atomic replace 寫出 high ``cell_dedup`` 的 938-key report。"""
    if not isinstance(report, pd.DataFrame):
        raise BorderExclusionError("cell dedup report 必須是 pandas DataFrame")
    if tuple(report.columns) != tuple(DEDUP_REPORT_COLUMNS):
        raise BorderExclusionError(
            "cell dedup report schema 不符: "
            f"expected={list(DEDUP_REPORT_COLUMNS)}, actual={list(report.columns)}"
        )
    if expected_duplicate_keys is not None and len(report) != int(expected_duplicate_keys):
        raise BorderExclusionError(
            f"cell dedup report rows expected {expected_duplicate_keys}, actual {len(report)}"
        )
    if report.empty and expected_duplicate_keys not in (None, 0):
        raise BorderExclusionError("cell dedup report 不可為空")
    _write_csv_atomic(report, DEDUP_REPORT_COLUMNS, Path(path))


def update_border_exclusion_metadata(
    result: BorderExclusionResult,
    metadata_path: str | Path,
    report_path: str | Path,
    group_targets_path: str | Path | None,
    *,
    expected_manifest_keys: int = 693,
    expected_cell_rows: int = 23976,
    expected_cell_keys: int = 693,
    expected_dedup_cells: int | None = None,
    expected_duplicate_keys: int | None = None,
    expected_sparse_fov_keys: Sequence[str] | None = None,
    expected_post_border_cells: int | None = None,
    expected_group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> dict[str, object]:
    """以 atomic JSON 更新觸邊結果，並鎖定第一次量測的 master row count。

    Args:
        result: ``apply_border_exclusion`` 的結果。
        metadata_path: 既有或待建立的 ``run_metadata.json``。
        report_path: 已完整寫出的 border report 路徑。
        group_targets_path: 已完整寫出的 group target 路徑；blocked 結果應傳
            ``None``，不得宣稱 downstream targets 已成功。
        expected_manifest_keys: 本次 snapshot manifest key count。
        expected_cell_rows: 本次 snapshot 排除前 master row count。
        expected_cell_keys: 本次 snapshot cell distinct image key count。
        expected_group_keys: 本次 snapshot exact group ids。

    Returns:
        寫入後的 metadata mapping；既有套件、seed 等欄位會保留。

    Raises:
        BorderExclusionError: input gate、既有 lock、輸出路徑或 metadata 格式不符。
    """
    expected_groups = tuple(str(key) for key in expected_group_keys)
    _validate_snapshot_counts(
        result.manifest,
        result.raw_cells,
        expected_manifest_keys=expected_manifest_keys,
        expected_cell_rows=expected_cell_rows,
        expected_cell_keys=expected_cell_keys,
        expected_group_keys=expected_groups,
    )
    if expected_dedup_cells is not None and len(result.pre_cells) != int(expected_dedup_cells):
        raise BorderExclusionError(
            f"dedup distinct whole-cell count expected {expected_dedup_cells}, "
            f"actual {len(result.pre_cells)}"
        )
    if expected_duplicate_keys is not None and result.duplicate_key_count != int(expected_duplicate_keys):
        raise BorderExclusionError(
            f"duplicate whole-cell key count expected {expected_duplicate_keys}, "
            f"actual {result.duplicate_key_count}"
        )
    if expected_sparse_fov_keys is not None:
        expected_sparse = _normalise_sparse_fov_keys(expected_sparse_fov_keys)
        if result.expected_sparse_fov_keys != expected_sparse:
            raise BorderExclusionError("sparse FOV roster lock mismatch")
    report = result.border_report
    if len(report) != len(result.manifest):
        raise BorderExclusionError("border report rows 不完整，拒絕更新 metadata")
    report_file = Path(report_path).expanduser()
    if not report_file.is_file():
        raise BorderExclusionError("border report 尚未完整寫出")
    metadata_file = Path(metadata_path).expanduser()
    existing = _read_metadata_json(metadata_file)
    proposed = dict(existing)

    raw_mask = result.raw_inclusion_mask
    if raw_mask is None or len(raw_mask) != len(result.raw_cells):
        raise BorderExclusionError("raw inclusion mask 尚未與 raw master rows 對齊")
    row_before = int(len(result.raw_cells))
    row_excluded = int((~raw_mask).sum())
    row_after = int(raw_mask.sum())
    distinct_before = int(len(result.pre_cells))
    distinct_excluded = int(len(result.excluded_cells))
    distinct_after = int(len(result.eligible_cells))
    whole_before = int(report["whole_cell_labels_before"].sum())
    whole_excluded = int(report["whole_cell_labels_excluded_border"].sum())
    whole_after = int(report["whole_cell_labels_after"].sum())
    minimum_whole_after = int(report["whole_cell_labels_after"].min())
    passed = result.minimum_cell_gate_status == "passed"

    if passed:
        if group_targets_path is None or not Path(group_targets_path).expanduser().is_file():
            raise BorderExclusionError("group targets 尚未完整寫出")
        expected_lock = distinct_after
        if expected_post_border_cells is not None and expected_lock != int(expected_post_border_cells):
            raise BorderExclusionError(
                f"post-border distinct cell count expected {expected_post_border_cells}, "
                f"actual {expected_lock}"
            )
        if "cells_after_border_exclusion" in existing:
            try:
                actual_lock = int(existing["cells_after_border_exclusion"])
            except (TypeError, ValueError) as error:
                raise BorderExclusionError(
                    "既有 cells_after_border_exclusion lock 無效"
                ) from error
            if actual_lock != expected_lock:
                raise BorderExclusionError(
                    "cells_after_border_exclusion lock mismatch: "
                    f"existing={actual_lock}, measured={expected_lock}"
                )
        else:
            proposed["cells_after_border_exclusion"] = expected_lock
        proposed["border_exclusion_status"] = "validated"
        proposed["target_status"] = "not_run"
        # A border/target pass does not validate a feature extraction run. Any
        # previous smoke evidence is retained only as historical evidence.
        _mark_feature_smoke_historical(proposed)
        validation_status = proposed.get("validation_status")
        validation = dict(validation_status) if isinstance(validation_status, Mapping) else {}
        validation["target"] = "validated"
        proposed["validation_status"] = validation
    else:
        proposed["border_exclusion_status"] = "blocked"
        _mark_blocked_metadata_freshness(proposed)

    proposed.update(
        {
            "border_exclusion_gate_status": result.minimum_cell_gate_status,
            "minimum_cell_gate_status": result.minimum_cell_gate_status,
            "cells_before_border_exclusion": row_before,
            "cells_excluded_border": row_excluded,
            "cells_after_border_exclusion_raw_pairs": row_after,
            "distinct_cells_before_border_exclusion": distinct_before,
            "distinct_cells_excluded_border": distinct_excluded,
            "distinct_cells_after_border_exclusion": distinct_after,
            "duplicate_whole_cell_key_count": int(result.duplicate_key_count),
            "duplicate_excess_row_count": int(result.duplicate_excess_row_count),
            "post_border_cell_count": distinct_after,
            "dedup_cell_count": distinct_before,
            "whole_cell_labels_before_border_exclusion": whole_before,
            "whole_cell_labels_excluded_border": whole_excluded,
            "whole_cell_labels_after_border_exclusion": whole_after,
            "fov_count": int(len(result.manifest)),
            "minimum_whole_cell_labels_after_border_exclusion": minimum_whole_after,
            "mask_only_label_total": int(result.mask_only_label_total),
            "mask_only_label_max_per_fov": int(result.mask_only_label_max_per_fov),
            "sparse_fov_keys": list(
                result.expected_sparse_fov_keys
                or tuple(
                    result.border_report.loc[
                        result.border_report["cells_after"].astype(int).eq(2),
                        "image_key",
                    ].astype(str)
                )
            ),
            "border_exclusion_report_path": str(
                report_file.resolve(strict=False)
            ),
        }
    )
    if group_targets_path is not None and passed:
        proposed["group_targets_path"] = str(
            Path(group_targets_path).expanduser().resolve(strict=False)
        )
        proposed["target_status"] = "validated"
    _write_json_atomic(proposed, metadata_file)
    return proposed


def _mark_blocked_metadata_freshness(metadata: dict[str, object]) -> None:
    """將既有 smoke/target evidence 明確降級為非 current 狀態。"""
    blocked_status = "not_run_due_to_border_gate"
    metadata["status"] = "blocked"
    metadata["target_status"] = "blocked"
    metadata["feature_smoke_status"] = blocked_status
    metadata["feature_extraction_status"] = blocked_status

    validation = metadata.get("validation_status")
    validation_status = dict(validation) if isinstance(validation, Mapping) else {}
    validation_status["feature_smoke"] = blocked_status
    metadata["validation_status"] = validation_status

    smoke_test = metadata.get("smoke_test")
    if isinstance(smoke_test, Mapping):
        smoke_status = dict(smoke_test)
        smoke_status["status"] = blocked_status
        smoke_status["evidence_status"] = "historical_stale"
        smoke_status["current_source_validated"] = False
        metadata["smoke_test"] = smoke_status

    if "feature_smoke_report" in metadata:
        smoke_report = metadata["feature_smoke_report"]
        if isinstance(smoke_report, Mapping):
            report_status = dict(smoke_report)
            report_status["status"] = "historical_stale"
            report_status["current_source_validated"] = False
            metadata["feature_smoke_report"] = report_status
        else:
            metadata["feature_smoke_report_status"] = "historical_stale"
            metadata["feature_smoke_report_current"] = False

    if "group_targets_path" in metadata:
        metadata["stale_group_targets_path"] = metadata.pop("group_targets_path")
        metadata["group_targets_status"] = "historical_stale"
    elif metadata.get("group_targets_status") == "validated":
        metadata["group_targets_status"] = "historical_stale"


def _mark_feature_smoke_historical(metadata: dict[str, object]) -> None:
    """將舊 feature smoke evidence 降級，並與 target status 分離。"""
    previous = str(metadata.get("feature_smoke_status", "not_run"))
    status = "historical_stale" if previous in {
        "validated",
        "smoke_validated",
        "passed",
    } else "not_run"
    metadata["feature_smoke_status"] = status
    metadata["feature_extraction_status"] = status
    validation = metadata.get("validation_status")
    validation_status = dict(validation) if isinstance(validation, Mapping) else {}
    validation_status["feature_smoke"] = status
    metadata["validation_status"] = validation_status
    smoke_test = metadata.get("smoke_test")
    if isinstance(smoke_test, Mapping):
        smoke = dict(smoke_test)
        if status == "historical_stale":
            smoke["status"] = "historical_stale"
            smoke["evidence_status"] = "historical_stale"
            smoke["current_source_validated"] = False
        metadata["smoke_test"] = smoke
    smoke_report = metadata.get("feature_smoke_report")
    if isinstance(smoke_report, Mapping):
        report = dict(smoke_report)
        report["status"] = "historical_stale"
        report["current_source_validated"] = False
        metadata["feature_smoke_report"] = report
    elif smoke_report is not None and status == "historical_stale":
        metadata["feature_smoke_report_status"] = "historical_stale"
        metadata["feature_smoke_report_current"] = False



def _deduplicate_for_border(
    raw_cells: pd.DataFrame,
    *,
    deduplicated_cells: pd.DataFrame | None,
    dedup_report: pd.DataFrame | None,
    expected_raw_rows: int,
    expected_distinct_cells: int,
    expected_multirow_keys: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """取得 high F0 dedup seam 的 distinct whole-cell frame。

    正式與測試資料都必須直接經過 high ``deduplicate_master_cells``；caller
    必須明確傳入 raw、distinct whole-cell 與 multirow-key 三個 expected locks，
    避免以輸入列數推測分析單位或繞過 high 的 F0 聚合契約。

    Args:
        raw_cells: raw cell master rows。
        deduplicated_cells: 已由 safe pipeline 產生的 high distinct frame。
        dedup_report: 已由 high dedup 產生的 multirow report。
        expected_raw_rows: raw row lock。
        expected_distinct_cells: distinct whole-cell lock。
        expected_multirow_keys: multirow whole-cell key lock。

    Returns:
        去重後 whole-cell frame 與 multirow report。

    Raises:
        BorderExclusionError: high API 或其 result schema 不符合規格。
    """
    for name, value in (
        ("expected_raw_rows", expected_raw_rows),
        ("expected_distinct_cells", expected_distinct_cells),
        ("expected_multirow_keys", expected_multirow_keys),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise BorderExclusionError(f"{name} 必須是非負整數")
        if int(value) < 0:
            raise BorderExclusionError(f"{name} 必須是非負整數")

    if (deduplicated_cells is None) != (dedup_report is None):
        raise BorderExclusionError(
            "deduplicated_cells 與 dedup_report 必須同時提供或同時省略"
        )
    if deduplicated_cells is None:
        try:
            high_result = deduplicate_master_cells(
                raw_cells,
                expected_raw_rows=int(expected_raw_rows),
                expected_distinct_cells=int(expected_distinct_cells),
                expected_multirow_key_count=int(expected_multirow_keys),
            )
        except Exception as error:
            raise BorderExclusionError(
                f"deduplicate_master_cells failed: {error}"
            ) from error
        if not isinstance(high_result, CellDedupResult):
            raise BorderExclusionError(
                "deduplicate_master_cells 必須回傳 CellDedupResult"
            )
        distinct = high_result.deduplicated_cells
        report = high_result.dedup_report
    else:
        distinct = deduplicated_cells.copy()
        report = dedup_report.copy()

    if not isinstance(distinct, pd.DataFrame):
        raise BorderExclusionError(
            "CellDedupResult.deduplicated_cells 必須是 DataFrame"
        )
    if not isinstance(report, pd.DataFrame):
        raise BorderExclusionError("CellDedupResult.dedup_report 必須是 DataFrame")
    if tuple(report.columns) != tuple(DEDUP_REPORT_COLUMNS):
        raise BorderExclusionError(
            "dedup_report schema 不符: "
            f"expected={list(DEDUP_REPORT_COLUMNS)}, actual={list(report.columns)}"
        )
    return distinct, report


def _validate_distinct_cells(cells: pd.DataFrame) -> None:
    """確認 border seam 接收的是恰一列一 whole-cell 的 frame。"""
    keys = pd.MultiIndex.from_frame(cells[["image_key", "cell_label"]])
    if keys.duplicated().any():
        raise BorderExclusionError(
            "border 分析單位必須是 distinct (image_key, cell_label) frame"
        )


def _normalise_sparse_fov_keys(
    values: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """正規化 optional sparse-FOV lock，拒絕空值與重複 image key。"""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise BorderExclusionError("expected_sparse_fov_keys 必須是 image_key 序列")
    keys = tuple(str(value).strip() for value in values)
    if any(not value for value in keys):
        raise BorderExclusionError("expected_sparse_fov_keys 不可含空值")
    if len(keys) != len(set(keys)):
        raise BorderExclusionError("expected_sparse_fov_keys 不可重複")
    return keys


def _raw_inclusion_mask(
    raw_cells: pd.DataFrame,
    distinct_cells: pd.DataFrame,
    distinct_inclusion: np.ndarray,
) -> np.ndarray:
    """把 distinct whole-cell border mask 對齊回 raw master rows。"""
    included_keys = set(
        (str(row.image_key), int(row.cell_label))
        for row in distinct_cells.loc[distinct_inclusion, ["image_key", "cell_label"]]
        .itertuples(index=False)
    )
    return np.asarray(
        [
            (str(row.image_key), int(row.cell_label)) in included_keys
            for row in raw_cells[["image_key", "cell_label"]].itertuples(index=False)
        ],
        dtype=bool,
    )


def _normalise_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = ("image_key", "b_id", "passage", "group_id")
    _require_columns(manifest, required, "manifest")
    frame = manifest.copy()
    if frame["image_key"].isna().any() or frame["group_id"].isna().any():
        raise BorderExclusionError("manifest image_key/group_id 不可為空")
    frame["image_key"] = frame["image_key"].astype(str).str.strip()
    frame["b_id"] = frame["b_id"].astype(str).str.strip().str.upper()
    frame["group_id"] = frame["group_id"].astype(str).str.strip()
    try:
        frame["passage"] = pd.to_numeric(frame["passage"], errors="raise").astype(int)
    except (TypeError, ValueError) as error:
        raise BorderExclusionError("manifest passage 必須是整數") from error
    if (frame["image_key"] == "").any() or (frame["group_id"] == "").any():
        raise BorderExclusionError("manifest image_key/group_id 不可為空")
    expected_group_from_parts = frame["b_id"] + "_P" + frame["passage"].astype(str)
    if not frame["group_id"].eq(expected_group_from_parts).all():
        raise BorderExclusionError("manifest group_id 與 b_id/passage 不一致")
    if frame["image_key"].duplicated().any():
        raise BorderExclusionError("manifest image_key 必須 unique")
    return frame


def _normalise_cells(
    cells: pd.DataFrame,
    *,
    allow_duplicate_master_keys: bool = True,
) -> pd.DataFrame:
    required = ("image_key", "cell_label", "nucleus_label", "IDO_score")
    _require_columns(cells, required, "cells")
    frame = cells.copy()
    if frame["image_key"].isna().any():
        raise BorderExclusionError("cells image_key 不可為空")
    frame["image_key"] = frame["image_key"].astype(str).str.strip()
    if (frame["image_key"] == "").any():
        raise BorderExclusionError("cells image_key 不可為空")
    labels: list[int] = []
    for value in frame["cell_label"].tolist():
        if pd.isna(value):
            raise BorderExclusionError("cells cell_label 不可為空")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise BorderExclusionError("cells cell_label 必須是 positive 整數") from error
        if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise BorderExclusionError("cells cell_label 必須是 positive 整數")
        labels.append(int(numeric))
    frame["cell_label"] = labels
    nucleus_labels: list[int] = []
    for value in frame["nucleus_label"].tolist():
        if pd.isna(value):
            raise BorderExclusionError("cells nucleus_label 不可為空")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise BorderExclusionError(
                "cells nucleus_label 必須是 positive 整數"
            ) from error
        if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise BorderExclusionError("cells nucleus_label 必須是 positive 整數")
        nucleus_labels.append(int(numeric))
    frame["nucleus_label"] = nucleus_labels
    frame["IDO_score"] = pd.to_numeric(frame["IDO_score"], errors="coerce")
    if not np.isfinite(frame["IDO_score"].to_numpy(dtype=float)).all():
        raise BorderExclusionError("cells IDO_score 必須全為 finite")
    master_key = pd.MultiIndex.from_arrays(
        [frame["image_key"], frame["cell_label"], frame["nucleus_label"]],
        names=("image_key", "cell_label", "nucleus_label"),
    )
    if not allow_duplicate_master_keys and master_key.duplicated().any():
        raise BorderExclusionError(
            "cells master key (image_key, cell_label, nucleus_label) 必須 unique"
        )
    return frame


def _validate_snapshot_counts(
    manifest: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    expected_manifest_keys: int,
    expected_cell_rows: int,
    expected_cell_keys: int,
    expected_group_keys: Sequence[str],
) -> None:
    actual_manifest_keys = set(manifest["image_key"])
    actual_cell_keys = set(cells["image_key"])
    if len(actual_manifest_keys) != int(expected_manifest_keys):
        raise BorderExclusionError(
            f"manifest unique image_key count expected {expected_manifest_keys}, "
            f"actual {len(actual_manifest_keys)}"
        )
    if len(cells) != int(expected_cell_rows):
        raise BorderExclusionError(
            f"cell rows expected {expected_cell_rows}, actual {len(cells)}"
        )
    if len(actual_cell_keys) != int(expected_cell_keys):
        raise BorderExclusionError(
            f"cell distinct image_key count expected {expected_cell_keys}, "
            f"actual {len(actual_cell_keys)}"
        )
    if actual_manifest_keys != actual_cell_keys:
        raise BorderExclusionError(
            "manifest/cell image_key sets 不一致: "
            f"manifest_only={sorted(actual_manifest_keys - actual_cell_keys)}, "
            f"cell_only={sorted(actual_cell_keys - actual_manifest_keys)}"
        )
    actual_groups = tuple(sorted(manifest["group_id"].unique()))
    expected_groups = tuple(sorted(str(key) for key in expected_group_keys))
    if actual_groups != expected_groups:
        raise BorderExclusionError(
            f"group keys expected {list(expected_groups)}, actual {list(actual_groups)}"
        )


def _mask_paths_by_stem(path: Path) -> dict[str, list[Path]]:
    if not path.is_dir():
        return {}
    result: dict[str, list[Path]] = {}
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() == ".npz":
            result.setdefault(candidate.stem, []).append(candidate)
    for paths in result.values():
        paths.sort(key=lambda candidate: candidate.as_posix().lower())
    return result


def _load_cell_mask(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "cell_mask" not in archive.files:
                raise BorderExclusionError(f"mask 缺少 cell_mask: {path.name}")
            mask = np.asarray(archive["cell_mask"])
    except BorderExclusionError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise BorderExclusionError(f"無法讀取 cell_mask: {path}") from error
    return mask


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise BorderExclusionError(f"{name} 必須是 pandas DataFrame")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise BorderExclusionError(f"{name} 缺少欄位: {missing}")


def _write_csv_atomic(frame: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    _require_columns(frame, columns, "CSV frame")
    parent = path.expanduser().resolve(strict=False).parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.loc[:, list(columns)].to_csv(temporary, index=False)
        os.replace(temporary, path.expanduser())
    except (OSError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BorderExclusionError(f"CSV atomic write failed: {path}") from error


def _read_metadata_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BorderExclusionError(f"metadata JSON 無法讀取: {path}") from error
    if not isinstance(value, dict):
        raise BorderExclusionError("metadata JSON root 必須是 object")
    return dict(value)


def _write_json_atomic(value: Mapping[str, object], path: Path) -> None:
    parent = path.resolve(strict=False).parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path.expanduser())
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BorderExclusionError(f"metadata atomic write failed: {path}") from error
