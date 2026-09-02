"""Exp8：每個 donor 在相同刺激濃度下的形狀變化量。

每個 donor 的基線形狀本來就不同（未刺激的離心率 B4=0.924、B7=0.932、
B8=0.950），所以比較的一定是**變化量**，不是絕對值：每個 donor 先減掉
自己同一個 passage 的未刺激對照。

變化量再除以對照組的 IQR 才能跨特徵比較——面積差 300 像素和離心率差 0.03
沒有共同單位，除以各自的離散程度之後才有。

重要限制：passage 不是生物重複，是同一個 lot 的連續傳代，而傳代本身就會
改變形狀。三個 passage 的散布是「傳代間變異」，不是生物信賴區間，因此圖上
一律畫個別點，不畫誤差棒。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..exp6.dataset import feature_label
from .brightness import CONTROL_CONDITION
from .shape import ShapeError, shape_columns


BLOCK_KEYS = ["b_id", "passage"]


def control_iqr(fov: pd.DataFrame, features: list[str]) -> pd.Series:
    """每個形狀特徵在全部未刺激影像上的 IQR，作為標準化的分母。"""
    control = fov[fov["condition"] == CONTROL_CONDITION]
    if control.empty:
        raise ShapeError(f"找不到對照條件 {CONTROL_CONDITION}")
    spread = control[features].quantile(0.75) - control[features].quantile(0.25)
    spread = spread.replace(0.0, np.nan)
    if spread.isna().all():
        raise ShapeError("對照組所有形狀特徵的 IQR 都是 0，無法標準化")
    return spread


def block_shape_summary(fov: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """每個 donor×passage×條件一列，值為該格所有影像的中位數。"""
    keys = BLOCK_KEYS + ["condition", "ifn_dose", "tnf_dose"]
    summary = fov.groupby(keys, as_index=False)[features].median()
    counts = fov.groupby(keys, as_index=False).size().rename(columns={"size": "n_images"})
    return summary.merge(counts, on=keys, how="left")


def shape_delta(fov: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """每個 donor×passage×條件相對於自己未刺激對照的形狀變化量。

    Returns:
        長表，每列一個 (donor, passage, 條件, 特徵)，含原始差值與標準化差值。
    """
    features = features or shape_columns(fov)
    summary = block_shape_summary(fov, features)
    spread = control_iqr(fov, features)

    controls = summary[summary["condition"] == CONTROL_CONDITION]
    missing = set(map(tuple, summary[BLOCK_KEYS].drop_duplicates().to_numpy())) - set(
        map(tuple, controls[BLOCK_KEYS].to_numpy())
    )
    if missing:
        raise ShapeError(f"下列 donor×passage 缺少未刺激對照，無法計算變化量：{sorted(missing)}")

    baseline = controls.set_index(BLOCK_KEYS)[features]
    rows: list[dict[str, object]] = []
    for record in summary.itertuples(index=False):
        block = (record.b_id, record.passage)
        if record.condition == CONTROL_CONDITION:
            continue
        for name in features:
            value = float(getattr(record, name))
            base = float(baseline.loc[block, name])
            delta = value - base
            rows.append(
                {
                    "b_id": record.b_id,
                    "passage": int(record.passage),
                    "condition": record.condition,
                    "ifn_dose": float(record.ifn_dose),
                    "tnf_dose": float(record.tnf_dose),
                    "feature": name,
                    "feature_label": feature_label(name),
                    "value": value,
                    "control_value": base,
                    "delta": delta,
                    "delta_standardized": delta / float(spread[name])
                    if np.isfinite(spread[name])
                    else float("nan"),
                    "n_images": int(record.n_images),
                }
            )
    return pd.DataFrame(rows)


def delta_matrix(delta: pd.DataFrame, *, axis: str = "ifn") -> pd.DataFrame:
    """把變化量整理成 heatmap 用的矩陣：列是特徵，欄是 donor × 濃度。

    Args:
        delta: `shape_delta` 的輸出。
        axis: `ifn` 只取 TNF=0 的條件，`tnf` 只取 IFN=0 的條件。
    """
    if axis == "ifn":
        subset = delta[delta["tnf_dose"] == 0]
        dose_column = "ifn_dose"
    elif axis == "tnf":
        subset = delta[delta["ifn_dose"] == 0]
        dose_column = "tnf_dose"
    else:
        raise ShapeError(f"未知的劑量軸：{axis}")
    if subset.empty:
        raise ShapeError(f"{axis} 軸沒有可用的變化量資料")

    # 先對 passage 取中位數，避免單一 passage 主導。
    collapsed = subset.groupby(["feature_label", "b_id", dose_column], as_index=False)[
        "delta_standardized"
    ].median()
    matrix = collapsed.pivot_table(
        index="feature_label", columns=["b_id", dose_column], values="delta_standardized"
    )
    order = matrix.abs().max(axis=1).sort_values(ascending=False).index
    return matrix.loc[order]


def overall_magnitude(delta: pd.DataFrame, *, condition: str) -> pd.DataFrame:
    """每個 donor×passage 在指定條件下的整體形狀變化幅度。

    定義為 17 個形狀特徵標準化變化量的絕對值平均。這是一個摘要數字，方便
    回答「哪個 donor 反應大」，但它把方向不同的特徵混在一起，只能當概觀。
    """
    subset = delta[delta["condition"] == condition]
    if subset.empty:
        raise ShapeError(f"找不到條件 {condition}")
    per_block = (
        subset.groupby(BLOCK_KEYS)["delta_standardized"]
        .apply(lambda values: float(np.mean(np.abs(values.dropna()))))
        .rename("magnitude")
        .reset_index()
    )
    per_donor = (
        per_block.groupby("b_id")["magnitude"]
        .agg(["median", "min", "max", "size"])
        .rename(columns={"size": "n_passages"})
        .reset_index()
    )
    per_block["condition"] = condition
    per_donor["condition"] = condition
    return per_block.merge(per_donor, on=["b_id", "condition"], how="left")


def donor_spread(delta: pd.DataFrame, *, axis: str = "ifn") -> pd.DataFrame:
    """每個特徵×濃度下，三個 donor 的變化量差距有多大。

    `donor_range` 是三個 donor 中位數的極差；把它和變化量本身的大小相比，
    就知道 donor 之間的差異是否值得討論。
    """
    matrix = delta_matrix(delta, axis=axis)
    dose_column = "ifn_dose" if axis == "ifn" else "tnf_dose"
    rows: list[dict[str, object]] = []
    for dose in sorted({level for _, level in matrix.columns}):
        block = matrix.xs(dose, axis=1, level=dose_column)
        for feature_name, values in block.iterrows():
            finite = values.dropna()
            if finite.empty:
                continue
            rows.append(
                {
                    "dose_ng_ml": float(dose),
                    "feature_label": feature_name,
                    "delta_median": float(finite.median()),
                    "donor_range": float(finite.max() - finite.min()),
                    "donor_min": finite.idxmin(),
                    "donor_max": finite.idxmax(),
                    "n_donors": int(len(finite)),
                }
            )
    table = pd.DataFrame(rows)
    table["range_vs_effect"] = table["donor_range"] / table["delta_median"].abs().replace(0, np.nan)
    return table.sort_values(["dose_ng_ml", "donor_range"], ascending=[True, False])
