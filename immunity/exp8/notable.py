"""Exp8：挑出形狀「特別」的細胞。

fig06 原本每個濃度都取最接近中位數的細胞，但濃度之間的中位數位移很小
（離心率 0.931 → 0.960），肉眼看不出差別。改成用形狀特徵的門檻挑細胞：
達標的那群和典型細胞擺在一起，差異才看得出來。

門檻取自未刺激對照組，和 `brightness.py` 定義 IDO 亮暗的邏輯一致：對照組
本身就是「沒有被刺激改變形狀」的分布，它的百分位就是有生物意義的那條線。

**不使用「全資料最極端的 K 顆」**。實測前 40 名的離心率落在 0.9975–0.9989、
長軸長落在 416–541 像素，多數是分割把相鄰細胞合併成一個物件的結果，而且
沒有劑量梯度（IFN0 佔 12–25%，和無關聯時的期望值差不多）。挑那一群等於在
展示分割失敗，不是展示生物差異。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .brightness import CONTROL_CONDITION
from .shape import ShapeError


#: 門檻百分位。正相關特徵取高分位，負相關取對稱的低分位。
DEFAULT_PERCENTILE = 90.0

#: 排除最極端的這個比例，避開分割把多顆細胞併在一起造成的假極端值。
ARTIFACT_TRIM = 0.01


@dataclass(frozen=True)
class FeatureThreshold:
    """單一形狀特徵的「特別」門檻。"""

    feature: str
    feature_label: str
    direction: int
    threshold: float
    artifact_cutoff: float
    percentile: float
    control_median: float
    control_cells: int

    @property
    def comparison(self) -> str:
        """門檻的比較方向文字。"""
        return "高於" if self.direction > 0 else "低於"

    def qualifies(self, values: pd.Series) -> pd.Series:
        """回傳布林序列：達標且未落入極端尾端。"""
        if self.direction > 0:
            return (values > self.threshold) & (values <= self.artifact_cutoff)
        return (values < self.threshold) & (values >= self.artifact_cutoff)

    def describe(self) -> str:
        """一行文字說明，寫進報告與紀錄。"""
        return (
            f"{self.feature_label}：{self.comparison} {self.threshold:.4g}"
            f"（對照組 P{self.percentile:.0f}，中位數 {self.control_median:.4g}）"
        )


def combined_ranking(
    ifn_table: pd.DataFrame, tnf_table: pd.DataFrame, *, top_n: int
) -> pd.DataFrame:
    """把兩條劑量軸的排名合併，取兩軸都排在前面的形狀特徵。

    只保留兩軸同號的特徵：如果某個特徵對 IFN 是正相關、對 TNF 是負相關，
    就沒有一致的「特別」方向可言，不適合當篩選條件。
    """
    left = ifn_table[["feature", "feature_label", "rank", "spearman_rho"]].rename(
        columns={"rank": "rank_ifn", "spearman_rho": "rho_ifn"}
    )
    right = tnf_table[["feature", "rank", "spearman_rho"]].rename(
        columns={"rank": "rank_tnf", "spearman_rho": "rho_tnf"}
    )
    merged = left.merge(right, on="feature", how="inner")
    merged = merged[np.sign(merged["rho_ifn"]) == np.sign(merged["rho_tnf"])]
    if merged.empty:
        raise ShapeError("沒有任何形狀特徵在兩條劑量軸上同號")
    merged["mean_rank"] = (merged["rank_ifn"] + merged["rank_tnf"]) / 2.0
    merged["direction"] = np.sign(merged["rho_ifn"]).astype(int)
    merged = merged.sort_values(["mean_rank", "rank_ifn"], kind="mergesort")
    merged["combined_rank"] = np.arange(1, len(merged) + 1)
    return merged.head(top_n).reset_index(drop=True)


def build_thresholds(
    cells: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    percentile: float = DEFAULT_PERCENTILE,
    artifact_trim: float = ARTIFACT_TRIM,
) -> list[FeatureThreshold]:
    """以未刺激對照組的百分位，為每個特徵建立門檻。"""
    control = cells[cells["condition"] == CONTROL_CONDITION]
    if len(control) < 100:
        raise ShapeError(f"對照組 {CONTROL_CONDITION} 細胞數不足，無法定義門檻")

    thresholds: list[FeatureThreshold] = []
    for row in ranking.itertuples():
        values = control[row.feature].dropna()
        pooled = cells[row.feature].dropna()
        if row.direction > 0:
            cut = float(np.percentile(values, percentile))
            artifact = float(np.percentile(pooled, 100.0 * (1.0 - artifact_trim)))
        else:
            cut = float(np.percentile(values, 100.0 - percentile))
            artifact = float(np.percentile(pooled, 100.0 * artifact_trim))
        thresholds.append(
            FeatureThreshold(
                feature=row.feature,
                feature_label=row.feature_label,
                direction=int(row.direction),
                threshold=cut,
                artifact_cutoff=artifact,
                percentile=percentile,
                control_median=float(np.median(values)),
                control_cells=int(len(values)),
            )
        )
    return thresholds


def notable_fraction(
    cells: pd.DataFrame, thresholds: list[FeatureThreshold]
) -> pd.DataFrame:
    """每個條件下，形狀達標的細胞比例。

    這張表是 fig06 的量化對照：影像庫只能顯示少數幾顆細胞，容易被質疑是
    挑出來的；比例才說得出「高濃度真的比較多這種細胞」。
    """
    rows: list[dict[str, object]] = []
    for order, threshold in enumerate(thresholds, start=1):
        for (condition, ifn, tnf), block in cells.groupby(
            ["condition", "ifn_dose", "tnf_dose"], sort=False
        ):
            qualifies = threshold.qualifies(block[threshold.feature])
            per_block = block.groupby(["b_id", "passage"])[threshold.feature].apply(
                lambda values, t=threshold: float(t.qualifies(values).mean())
            )
            rows.append(
                {
                    "feature_order": order,
                    "feature": threshold.feature,
                    "feature_label": threshold.feature_label,
                    "condition": condition,
                    "ifn_dose": float(ifn),
                    "tnf_dose": float(tnf),
                    "n_cells": int(len(block)),
                    "n_notable": int(qualifies.sum()),
                    "notable_fraction": float(qualifies.mean()),
                    "notable_fraction_min": float(per_block.min()),
                    "notable_fraction_max": float(per_block.max()),
                    "threshold": threshold.threshold,
                    "direction": threshold.direction,
                }
            )
    table = pd.DataFrame(rows)
    return table.sort_values(["feature_order", "ifn_dose", "tnf_dose"]).reset_index(drop=True)


def _spread_over_quantiles(frame: pd.DataFrame, column: str, count: int) -> pd.DataFrame:
    """在某一欄的值域上等分位取樣，回傳依該欄排序的 `count` 列。

    全部挑靠近中位數的細胞，一整列的值會幾乎相同（例如離心率全是 0.99），
    看起來像挑過的。改成等分位取樣，一列就從「剛過門檻」漸變到「明顯特別」。
    """
    if frame.empty:
        raise ShapeError("沒有可取樣的細胞")
    ordered = frame.sort_values(column, kind="mergesort").reset_index()
    if len(ordered) <= count:
        return ordered.set_index("index")
    positions = np.unique(np.linspace(0, len(ordered) - 1, count).round().astype(int))
    return ordered.iloc[positions].set_index("index")


def select_notable_cells(
    cells: pd.DataFrame,
    threshold: FeatureThreshold,
    *,
    count: int,
) -> pd.DataFrame:
    """挑出達標的細胞，在達標區間內等分位取樣。

    不取最極端的那一端：門檻已經確保它們「特別」，再往尾端挑只會挑到分割
    失敗的物件（相鄰細胞被併成一個）。
    """
    qualifying = cells[threshold.qualifies(cells[threshold.feature])]
    if qualifying.empty:
        raise ShapeError(f"沒有任何細胞達到 {threshold.feature_label} 的門檻")

    chosen = _spread_over_quantiles(qualifying, threshold.feature, count)
    if threshold.direction < 0:
        chosen = chosen.iloc[::-1]
    return chosen.assign(
        notable_feature=threshold.feature,
        notable_label=threshold.feature_label,
        notable_value=lambda frame: frame[threshold.feature],
    )


def select_typical_cells(
    cells: pd.DataFrame, thresholds: list[FeatureThreshold], *, count: int
) -> pd.DataFrame:
    """挑出未刺激且所有特徵都沒達標的「典型細胞」，作為對照列。

    同樣在主要特徵的值域上等分位取樣，但只取中間 80%，避免對照列出現
    緊貼門檻或極小的細胞。
    """
    control = cells[cells["condition"] == CONTROL_CONDITION]
    if control.empty:
        raise ShapeError(f"找不到對照條件 {CONTROL_CONDITION}")
    mask = pd.Series(True, index=control.index)
    for threshold in thresholds:
        mask &= ~threshold.qualifies(control[threshold.feature])
    typical = control[mask]
    if typical.empty:
        raise ShapeError("對照組沒有任何全部特徵都未達標的細胞")

    primary = thresholds[0]
    low, high = np.percentile(typical[primary.feature], [10, 90])
    middle = typical[typical[primary.feature].between(low, high)]
    chosen = _spread_over_quantiles(middle if len(middle) >= count else typical, primary.feature, count)
    return chosen.assign(
        notable_feature="typical",
        notable_label="典型細胞",
        notable_value=lambda frame: frame[primary.feature],
    )


def cell_caption(row: pd.Series) -> tuple[str, str]:
    """回傳兩行標題：完整細胞編號，以及刺激條件＋該特徵的值。"""
    value = float(row["notable_value"])
    text = f"{value:.3g}" if abs(value) < 1000 else f"{value:.0f}"
    identifier = f"{row['image_key']} #{int(row['cell_label'])}"
    return identifier, f"{row['condition']}　{text}"
