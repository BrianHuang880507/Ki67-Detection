"""Exp4 whole-cell 去重與多核特徵聚合介面。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from immunity.exp3.feature_sets import BASIC_GEOMETRY


CELL_KEY_COLUMNS = ("image_key", "cell_label")
NUCLEUS_GEOMETRY_COLUMNS = tuple(
    f"nucleus__{name}" for name in BASIC_GEOMETRY
)
NUCLEUS_FEATURE_COLUMNS = (
    *NUCLEUS_GEOMETRY_COLUMNS,
    "nucleus_cytoplasm_area_ratio",
)
DEDUP_REPORT_COLUMNS = (
    *CELL_KEY_COLUMNS,
    "original_row_count",
    "nucleus_labels_before",
    "IDO_score_values_before",
    "IDO_score_after",
    *NUCLEUS_FEATURE_COLUMNS,
)
RUI49_FEATURE_COUNT = 49


class CellDedupError(ValueError):
    """表示 raw master cell 資料不符合 Exp4 F0 contract。"""


@dataclass(frozen=True)
class WholeCellFeatureConsistencyResult:
    """記錄 whole-cell feature consistency assertion 的實際檢查範圍。

    Attributes:
        scope: 呼叫端明確提供的 full/smoke scope，不會由函式自行宣稱 full。
        checked_row_count: 實際檢查的 joined rows 數。
        checked_cell_count: 實際檢查的 distinct whole-cell key 數。
        checked_duplicate_key_count: 真正有跨列比較的 multirow key 數。
        feature_count: 本次逐位元比較的 feature 欄數。
    """

    scope: str
    checked_row_count: int
    checked_cell_count: int
    checked_duplicate_key_count: int
    feature_count: int

    def to_dict(self) -> dict[str, str | int]:
        """回傳可寫入 metadata/CLI summary 的明確 scope 證據。"""
        return {
            "scope": self.scope,
            "checked_row_count": self.checked_row_count,
            "checked_cell_count": self.checked_cell_count,
            "checked_duplicate_key_count": self.checked_duplicate_key_count,
            "feature_count": self.feature_count,
        }


@dataclass(frozen=True)
class CellDedupResult:
    """保存 whole-cell 去重後資料與可追溯報告。

    Attributes:
        deduplicated_cells: 每個 ``image_key × cell_label`` 恰一列的完整資料。
        dedup_report: 僅含原始 multirow keys 的固定 schema 報告。
        raw_row_count: 輸入 raw master 列數。
        distinct_cell_count: 去重後相異 whole-cell 數。
        multirow_key_count: 原始列數大於一的 whole-cell key 數。
        duplicate_excess_row_count: 相對相異 whole-cell 多出的 raw rows 數。
        whole_cell_consistency: raw master 中 ``cell__*`` 欄的 exact check 證據。
    """

    deduplicated_cells: pd.DataFrame
    dedup_report: pd.DataFrame
    raw_row_count: int
    distinct_cell_count: int
    multirow_key_count: int
    duplicate_excess_row_count: int
    whole_cell_consistency: WholeCellFeatureConsistencyResult

    def to_dict(self) -> dict[str, object]:
        """回傳可供 CLI summary 使用的計數。"""
        return {
            "raw_row_count": self.raw_row_count,
            "distinct_cell_count": self.distinct_cell_count,
            "multirow_key_count": self.multirow_key_count,
            "duplicate_excess_row_count": self.duplicate_excess_row_count,
            "whole_cell_consistency": self.whole_cell_consistency.to_dict(),
        }


def assert_full_rui49_feature_consistency(
    joined_cells: pd.DataFrame,
    *,
    scope: str = "full_join_693_fovs",
) -> WholeCellFeatureConsistencyResult:
    """以 canonical 49 欄執行 full F0 whole-cell consistency gate。

    延遲讀取 ``rui_features.RUI49_FEATURE_COLUMNS``，避免 module import 階段
    形成循環依賴；canonical tuple 不接受 caller override。smoke subset 應改呼叫
    ``assert_whole_cell_feature_consistency`` 並明示 smoke scope。

    Args:
        joined_cells: full Rui49 join 後仍保留 raw multirow keys 的資料表。
        scope: metadata 中記錄的 full-run scope。

    Returns:
        exact comparison 的 rows、keys、49 欄數與 full scope 證據。

    Raises:
        CellDedupError: 權威 tuple 不是恰好 49 個唯一欄名，或一致性 gate 失敗。
    """
    from .rui_features import RUI49_FEATURE_COLUMNS

    columns = tuple(RUI49_FEATURE_COLUMNS)
    if len(columns) != RUI49_FEATURE_COUNT or len(set(columns)) != RUI49_FEATURE_COUNT:
        raise CellDedupError("full Rui49 gate 必須使用恰好 49 個唯一 canonical 欄位")
    return assert_whole_cell_feature_consistency(
        joined_cells,
        columns,
        scope=scope,
    )


def assert_whole_cell_feature_consistency(
    joined_cells: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    scope: str,
) -> WholeCellFeatureConsistencyResult:
    """斷言同一 whole-cell key 的指定特徵完全一致。

    比較語意等同於有限 ``float64`` 陣列上的 ``array_equal``：不使用任何
    tolerance（``rtol=0, atol=0``），NaN、Infinity 與無法轉成數值的值一律
    fail-closed。caller 必須提供真實 scope，避免 smoke 結果被誤標為 full。

    Args:
        joined_cells: feature join 後仍保留 raw multirow keys 的資料表。
        feature_columns: 本次要逐列完全比較的 whole-cell feature 欄位。
        scope: 本次檢查範圍，例如 ``smoke_join_5_fovs`` 或
            ``full_join_693_fovs``。

    Returns:
        實際 rows、cells、duplicate keys、feature count 與 scope 的證據。

    Raises:
        CellDedupError: scope、欄位、key、finite gate 或 exact equality 失敗。
    """
    checked_scope = str(scope).strip()
    if not checked_scope:
        raise CellDedupError("whole-cell consistency scope 不可為空")
    columns = tuple(str(column) for column in feature_columns)
    if not columns:
        raise CellDedupError("whole-cell feature columns 不可為空")
    if len(columns) != len(set(columns)):
        raise CellDedupError("whole-cell feature columns 不可重複")
    _require_columns(joined_cells, (*CELL_KEY_COLUMNS, *columns))
    working = joined_cells.reset_index(drop=True)
    _validate_keys(working)
    if any(list(joined_cells.columns).count(column) != 1 for column in columns):
        raise CellDedupError("joined cells 的 whole-cell feature 欄名必須唯一")

    numeric = working.loc[:, columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise CellDedupError("whole-cell feature consistency 輸入含非 finite 值")

    duplicate_key_count = 0
    grouped = working.groupby(
        list(CELL_KEY_COLUMNS),
        sort=False,
        dropna=False,
    )
    for (image_key, cell_label), group in grouped:
        if len(group) <= 1:
            continue
        duplicate_key_count += 1
        group_values = numeric.loc[group.index, columns].to_numpy(
            dtype=np.float64
        )
        reference = np.broadcast_to(group_values[0], group_values.shape)
        if not np.array_equal(group_values, reference, equal_nan=False):
            unequal = np.flatnonzero(np.any(group_values != reference, axis=0))
            failed_columns = ", ".join(columns[index] for index in unequal)
            raise CellDedupError(
                "同一 whole-cell key 的 feature 不完全相同："
                f"key=({image_key!r}, {cell_label!r}), "
                f"columns={failed_columns}"
            )

    distinct_count = int(
        working.loc[:, CELL_KEY_COLUMNS].drop_duplicates().shape[0]
    )
    return WholeCellFeatureConsistencyResult(
        scope=checked_scope,
        checked_row_count=len(joined_cells),
        checked_cell_count=distinct_count,
        checked_duplicate_key_count=duplicate_key_count,
        feature_count=len(columns),
    )


def deduplicate_master_cells(
    cells: pd.DataFrame,
    *,
    expected_raw_rows: int = 23976,
    expected_distinct_cells: int = 23012,
    expected_multirow_key_count: int = 938,
) -> CellDedupResult:
    """將 (whole-cell, nucleus) raw rows 去重成 whole-cell 分析單位。

    ``IDO_score`` 取同一 key 的 median；``nucleus__area`` 加總，其餘
    nucleus geometry 取 median，核質面積比則以加總核面積重新計算。函式只
    回傳資料，不寫檔，讓 CLI adapter 自行決定輸出位置。

    Args:
        cells: 原始 cell master frame，必須含 cell key、nucleus label、IDO、
            ``cell__area`` 與完整 17 個 nucleus arm 欄位。
        expected_raw_rows: fail-closed 的預期 raw row count。
        expected_distinct_cells: fail-closed 的預期 distinct whole-cell count。
        expected_multirow_key_count: fail-closed 的預期 multirow whole-cell key 數。

    Returns:
        完整去重 frame、僅 multirow keys 的 report 與各階段計數。

    Raises:
        CellDedupError: 欄位、key、數值、snapshot count 或核質面積不合法。
    """
    _validate_expected_count(expected_raw_rows, "expected_raw_rows")
    _validate_expected_count(
        expected_distinct_cells,
        "expected_distinct_cells",
    )
    _validate_expected_count(
        expected_multirow_key_count,
        "expected_multirow_key_count",
    )
    required = (
        *CELL_KEY_COLUMNS,
        "nucleus_label",
        "IDO_score",
        "cell__area",
        *NUCLEUS_FEATURE_COLUMNS,
    )
    _require_columns(cells, required)
    frame = cells.copy()
    if len(frame) != expected_raw_rows:
        raise CellDedupError(
            "raw master row count 不符："
            f"expected {expected_raw_rows}, actual {len(frame)}"
        )
    _validate_keys(frame)
    whole_cell_columns = tuple(
        str(column)
        for column in frame.columns
        if str(column).startswith("cell__")
    )
    consistency = assert_whole_cell_feature_consistency(
        frame,
        whole_cell_columns,
        scope="raw_master_before_dedup",
    )
    numeric_columns = (
        "IDO_score",
        "cell__area",
        *NUCLEUS_FEATURE_COLUMNS,
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric_values = frame.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise CellDedupError("IDO、cell area 與 nucleus 特徵必須全為 finite")

    distinct_count = int(
        frame.loc[:, CELL_KEY_COLUMNS].drop_duplicates().shape[0]
    )
    if distinct_count != expected_distinct_cells:
        raise CellDedupError(
            "distinct whole-cell count 不符："
            f"expected {expected_distinct_cells}, actual {distinct_count}"
        )
    key_sizes = frame.groupby(
        list(CELL_KEY_COLUMNS),
        sort=False,
        dropna=False,
    ).size()
    multirow_key_count = int((key_sizes > 1).sum())
    if multirow_key_count != expected_multirow_key_count:
        raise CellDedupError(
            "multirow whole-cell key count 不符："
            f"expected {expected_multirow_key_count}, actual {multirow_key_count}"
        )

    deduplicated_rows: list[pd.Series] = []
    report_rows: list[dict[str, object]] = []
    grouped = frame.groupby(list(CELL_KEY_COLUMNS), sort=False, dropna=False)
    median_columns = tuple(
        column
        for column in NUCLEUS_GEOMETRY_COLUMNS
        if column != "nucleus__area"
    )
    for (image_key, cell_label), group in grouped:
        aggregated = group.iloc[0].copy()
        ido_after = float(np.median(group["IDO_score"].to_numpy(dtype=float)))
        nucleus_area = float(
            np.sum(group["nucleus__area"].to_numpy(dtype=float))
        )
        cell_area = float(group.iloc[0]["cell__area"])
        denominator = cell_area - nucleus_area
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise CellDedupError(
                "nucleus/cytoplasm denominator 必須為 finite 且大於 0："
                f"key=({image_key!r}, {cell_label!r}), "
                f"cell_area={cell_area}, nucleus_area={nucleus_area}"
            )
        aggregated["IDO_score"] = ido_after
        aggregated["nucleus__area"] = nucleus_area
        for column in median_columns:
            aggregated[column] = float(
                np.median(group[column].to_numpy(dtype=float))
            )
        ratio = nucleus_area / denominator
        if not np.isfinite(ratio):
            raise CellDedupError(
                "nucleus_cytoplasm_area_ratio 聚合後不是 finite："
                f"key=({image_key!r}, {cell_label!r})"
            )
        aggregated["nucleus_cytoplasm_area_ratio"] = ratio
        deduplicated_rows.append(aggregated)

        if len(group) > 1:
            stable_group = group.sort_values("nucleus_label", kind="stable")
            report_rows.append(
                {
                    "image_key": image_key,
                    "cell_label": cell_label,
                    "original_row_count": int(len(group)),
                    "nucleus_labels_before": _stable_json_values(
                        stable_group["nucleus_label"]
                    ),
                    "IDO_score_values_before": _stable_json_values(
                        stable_group["IDO_score"],
                        force_float=True,
                    ),
                    "IDO_score_after": ido_after,
                    **{
                        column: aggregated[column]
                        for column in NUCLEUS_FEATURE_COLUMNS
                    },
                }
            )

    deduplicated = pd.DataFrame(deduplicated_rows, columns=frame.columns)
    if deduplicated.duplicated(list(CELL_KEY_COLUMNS)).any():
        raise CellDedupError("去重結果仍含重複 whole-cell key")
    report = pd.DataFrame(report_rows, columns=DEDUP_REPORT_COLUMNS)
    return CellDedupResult(
        deduplicated_cells=deduplicated.reset_index(drop=True),
        dedup_report=report.reset_index(drop=True),
        raw_row_count=len(frame),
        distinct_cell_count=distinct_count,
        multirow_key_count=multirow_key_count,
        duplicate_excess_row_count=len(frame) - distinct_count,
        whole_cell_consistency=consistency,
    )


def _validate_expected_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CellDedupError(f"{name} 必須是非負整數")
    if int(value) < 0:
        raise CellDedupError(f"{name} 必須是非負整數")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CellDedupError("raw master 缺少欄位: " + ", ".join(missing))


def _validate_keys(frame: pd.DataFrame) -> None:
    if frame["image_key"].isna().any() or frame["cell_label"].isna().any():
        raise CellDedupError("whole-cell key 不可為空")
    image_keys = frame["image_key"].astype(str)
    if image_keys.str.strip().eq("").any():
        raise CellDedupError("image_key 不可為空字串")


def _stable_json_values(
    values: pd.Series,
    *,
    force_float: bool = False,
) -> str:
    serializable: list[object] = []
    for value in values.tolist():
        if force_float:
            serializable.append(float(value))
        elif isinstance(value, np.integer):
            serializable.append(int(value))
        elif isinstance(value, np.floating):
            serializable.append(float(value))
        else:
            serializable.append(value)
    return json.dumps(
        serializable,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
