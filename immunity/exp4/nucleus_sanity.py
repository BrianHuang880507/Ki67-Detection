"""Exp4 v5 phase-derived nucleus proxy 的內部一致性描述。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .cell_dedup import NUCLEUS_FEATURE_COLUMNS


EXPECTED_RETAINED_CELL_COUNT = 19_648
EXPECTED_FOV_COUNT = 693
NUCLEUS_SANITY_COLUMNS = (
    "section",
    "metric",
    "feature",
    "statistic",
    "value",
    "n_cells",
    "notes",
)


class NucleusSanityError(ValueError):
    """表示 E3 sanity report 的正式母體或數值 contract 不成立。"""


def build_nucleus_sanity_report(retained_cells: pd.DataFrame) -> pd.DataFrame:
    """建立 19,648 cells／693 FOV 的 nucleus proxy 描述表。

    輸入必須已套用 v4 whole-cell 去重、多核聚合與觸邊排除。此檢查只描述
    phase-derived nucleus proxy 的內部一致性，不設定品質通過門檻，也不改變
    arm 4 是否執行。

    Args:
        retained_cells: 每個 ``image_key × cell_label`` 恰一列，含 cell area 與
            v4 聚合後完整 17 個 nucleus 欄位的正式建模母體。

    Returns:
        長格式 machine-readable table，涵蓋 area fraction、相關、CV，以及依
        cell area quintile 的中位數與 Q5/Q1 fold-change。

    Raises:
        TypeError: 輸入不是 DataFrame 時拋出。
        NucleusSanityError: 母體鎖定值、identity、欄位或有限數值不合法時拋出。
    """
    cells = _validated_retained_cells(retained_cells)
    cell_area = cells["cell__Area"].to_numpy(dtype=np.float64)
    nucleus_area = cells["nucleus__area"].to_numpy(dtype=np.float64)
    nucleus_ratio = cells["nucleus_cytoplasm_area_ratio"].to_numpy(
        dtype=np.float64
    )
    area_fraction = nucleus_area / cell_area
    n_cells = len(cells)
    rows: list[dict[str, object]] = []

    _append_row(
        rows,
        section="area_fraction",
        metric="nucleus_area_fraction_gt_one_count",
        feature="nucleus__area/cell__Area",
        statistic="count",
        value=float(np.count_nonzero(area_fraction > 1.0)),
        n_cells=n_cells,
        notes="descriptive_only; value > 1 is geometrically impossible",
    )
    for statistic, quantile in (("p25", 0.25), ("p50", 0.50), ("p75", 0.75)):
        _append_row(
            rows,
            section="area_fraction",
            metric="nucleus_area_fraction_quantile",
            feature="nucleus__area/cell__Area",
            statistic=statistic,
            value=float(np.quantile(area_fraction, quantile)),
            n_cells=n_cells,
            notes="v4 aggregated nucleus area divided by whole-cell area",
        )

    spearman = spearmanr(cell_area, nucleus_area).statistic
    _append_row(
        rows,
        section="association",
        metric="spearman_cell_area_nucleus_area",
        feature="cell__Area~nucleus__area",
        statistic="spearman_r",
        value=float(spearman),
        n_cells=n_cells,
        notes="two-sided p-value intentionally omitted from descriptive contract",
    )
    _append_row(
        rows,
        section="shape",
        metric="nucleus_sphericity_median",
        feature="nucleus__sphericity",
        statistic="p50",
        value=float(np.median(cells["nucleus__sphericity"].to_numpy(dtype=np.float64))),
        n_cells=n_cells,
        notes="v4 aggregated phase-derived nucleus proxy; descriptive only",
    )

    for feature in ("cell__Area", *NUCLEUS_FEATURE_COLUMNS):
        values = cells[feature].to_numpy(dtype=np.float64)
        _append_row(
            rows,
            section="dispersion",
            metric="coefficient_of_variation",
            feature=feature,
            statistic="std_ddof0_over_abs_mean",
            value=_coefficient_of_variation(values),
            n_cells=n_cells,
            notes="descriptive_only; not an automatic feature-removal gate",
        )

    pearson = pearsonr(cell_area, nucleus_ratio).statistic
    _append_row(
        rows,
        section="association",
        metric="pearson_nucleus_ratio_cell_area",
        feature="nucleus_cytoplasm_area_ratio~cell__Area",
        statistic="pearson_r",
        value=float(pearson),
        n_cells=n_cells,
        notes="ratio uses summed nucleus area after v4 multinuclear aggregation",
    )

    quintile = pd.qcut(cell_area, q=5, labels=("q1", "q2", "q3", "q4", "q5"))
    quintile_frame = pd.DataFrame(
        {
            "quintile": quintile,
            "cell__Area": cell_area,
            "nucleus__area": nucleus_area,
            "nucleus_cytoplasm_area_ratio": nucleus_ratio,
        }
    )
    quintile_medians: dict[str, dict[str, float]] = {}
    for feature in (
        "cell__Area",
        "nucleus__area",
        "nucleus_cytoplasm_area_ratio",
    ):
        medians: dict[str, float] = {}
        for label in ("q1", "q2", "q3", "q4", "q5"):
            selected = quintile_frame.loc[
                quintile_frame["quintile"].eq(label), feature
            ]
            value = float(selected.median())
            medians[label] = value
            _append_row(
                rows,
                section="cell_area_quintile",
                metric="cell_area_quintile_median",
                feature=feature,
                statistic=label,
                value=value,
                n_cells=int(len(selected)),
                notes="quintiles are assigned from cell__Area only",
            )
        quintile_medians[feature] = medians
        denominator = medians["q1"]
        fold_change = (
            float(medians["q5"] / denominator)
            if denominator != 0.0
            else float("inf")
        )
        _append_row(
            rows,
            section="cell_area_quintile",
            metric="cell_area_quintile_fold_change",
            feature=feature,
            statistic="q5_over_q1_median",
            value=fold_change,
            n_cells=n_cells,
            notes="Q5 median divided by Q1 median",
        )

    return pd.DataFrame(rows, columns=NUCLEUS_SANITY_COLUMNS)


def _validated_retained_cells(retained_cells: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(retained_cells, pd.DataFrame):
        raise TypeError("retained_cells 必須是 pandas DataFrame")
    required = {
        "image_key",
        "cell_label",
        "cell__Area",
        *NUCLEUS_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(retained_cells.columns))
    if missing:
        raise NucleusSanityError(f"retained_cells 缺少欄位：{missing}")
    if len(retained_cells) != EXPECTED_RETAINED_CELL_COUNT:
        raise NucleusSanityError(
            "retained cell count expected "
            f"{EXPECTED_RETAINED_CELL_COUNT}, actual {len(retained_cells)}"
        )
    identity = retained_cells.loc[:, ["image_key", "cell_label"]]
    if identity.isna().any().any() or identity.duplicated().any():
        raise NucleusSanityError(
            "retained cells 必須是唯一且非空的 (image_key, cell_label)"
        )
    image_keys = identity["image_key"].astype(str).str.strip()
    if image_keys.eq("").any():
        raise NucleusSanityError("image_key 不可為空")
    actual_fovs = int(image_keys.nunique())
    if actual_fovs != EXPECTED_FOV_COUNT:
        raise NucleusSanityError(
            f"retained FOV count expected {EXPECTED_FOV_COUNT}, actual {actual_fovs}"
        )
    numeric_columns = ("cell__Area", *NUCLEUS_FEATURE_COLUMNS)
    try:
        numeric = retained_cells.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise NucleusSanityError("cell/nucleus sanity 欄位必須為數值") from error
    if not np.isfinite(numeric).all():
        raise NucleusSanityError("cell/nucleus sanity 欄位必須全為 finite")
    if np.any(numeric[:, 0] <= 0.0):
        raise NucleusSanityError("cell__Area 必須大於 0")
    nucleus_area_index = 1 + NUCLEUS_FEATURE_COLUMNS.index("nucleus__area")
    if np.any(numeric[:, nucleus_area_index] < 0.0):
        raise NucleusSanityError("nucleus__area 不可小於 0")
    return retained_cells.copy()


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if mean == 0.0:
        return 0.0 if std == 0.0 else float("inf")
    return float(std / abs(mean))


def _append_row(
    rows: list[dict[str, object]],
    *,
    section: str,
    metric: str,
    feature: str,
    statistic: str,
    value: float,
    n_cells: int,
    notes: str,
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "feature": feature,
            "statistic": statistic,
            "value": value,
            "n_cells": n_cells,
            "notes": notes,
        }
    )


__all__ = [
    "EXPECTED_FOV_COUNT",
    "EXPECTED_RETAINED_CELL_COUNT",
    "NUCLEUS_SANITY_COLUMNS",
    "NucleusSanityError",
    "build_nucleus_sanity_report",
]
