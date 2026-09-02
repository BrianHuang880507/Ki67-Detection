"""Exp8：用未刺激對照組定義「亮」與「暗」，並在同一張影像內比較形狀。

不用百分位硬切。IFN0_TNF0 的 4,066 顆細胞本身就是「完全沒亮」的分布，
它的 P95 與 P99 直接給出生物意義明確的兩條線：

- 暗：`IDO_score_ff <= P95`，和完全沒刺激的細胞分不出來。
- 亮：`IDO_score_ff > P99`，在對照組裡只有 1% 會到這個程度。
- 中間灰帶不納入比較。

比較一律在**同一張影像內**進行。同一張影像的細胞共享 donor、passage、
劑量、拍攝時間、照明與焦平面，把這些全部配對掉之後，剩下的差異才是
細胞本身的差異。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..exp6.dataset import feature_label
from ..exp6.dose_association import benjamini_hochberg


CONTROL_CONDITION = "IFN0_TNF0"

#: 亮暗比較的主力條件。IFN25_TNF0 的影像內暗細胞比例約 47%，兩群人數最平衡；
#: 濃度更高的條件幾乎全亮，暗的那群人數太少。
PRIMARY_CONDITION = "IFN25_TNF0"

DARK_LABEL = "暗"
BRIGHT_LABEL = "亮"
MIDDLE_LABEL = "灰帶"


class BrightnessError(RuntimeError):
    """表示亮暗分組的輸入不符預期。"""


@dataclass(frozen=True)
class Thresholds:
    """由未刺激對照組導出的兩條門檻。"""

    dark_max: float
    bright_min: float
    control_cells: int
    control_median: float

    def describe(self) -> str:
        """一行文字說明，寫進報告與紀錄。"""
        return (
            f"對照組 {CONTROL_CONDITION} 共 {self.control_cells:,} 顆細胞，"
            f"中位數 {self.control_median:.2f}；"
            f"暗 <= {self.dark_max:.2f}（P95）、亮 > {self.bright_min:.2f}（P99）灰階"
        )


def control_thresholds(cells: pd.DataFrame, *, target: str = "IDO_score_ff") -> Thresholds:
    """從未刺激對照組算出暗／亮兩條門檻。"""
    control = cells.loc[cells["condition"] == CONTROL_CONDITION, target].dropna()
    if len(control) < 100:
        raise BrightnessError(
            f"對照組 {CONTROL_CONDITION} 只有 {len(control)} 顆細胞，不足以定義門檻"
        )
    return Thresholds(
        dark_max=float(np.percentile(control, 95)),
        bright_min=float(np.percentile(control, 99)),
        control_cells=int(len(control)),
        control_median=float(np.median(control)),
    )


def label_brightness(
    cells: pd.DataFrame, thresholds: Thresholds, *, target: str = "IDO_score_ff"
) -> pd.DataFrame:
    """把每顆細胞標成暗、灰帶或亮。"""
    frame = cells.copy()
    frame["ido_class"] = MIDDLE_LABEL
    frame.loc[frame[target] <= thresholds.dark_max, "ido_class"] = DARK_LABEL
    frame.loc[frame[target] > thresholds.bright_min, "ido_class"] = BRIGHT_LABEL
    return frame


def positive_fraction(labelled: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    """每個條件的 IDO 陽性細胞比例，並附上各 donor×passage 的個別值。

    陽性定義為超過對照組 P95，也就是「亮到在未刺激細胞裡看不到」。
    """
    rows: list[dict[str, object]] = []
    for (condition, ifn, tnf), block in labelled.groupby(
        ["condition", "ifn_dose", "tnf_dose"], sort=False
    ):
        per_group = block.groupby(["b_id", "passage"])["IDO_score_ff"].apply(
            lambda values: float((values > thresholds.dark_max).mean())
        )
        rows.append(
            {
                "condition": condition,
                "ifn_dose": float(ifn),
                "tnf_dose": float(tnf),
                "n_cells": int(len(block)),
                "positive_fraction": float((block["IDO_score_ff"] > thresholds.dark_max).mean()),
                "positive_fraction_min": float(per_group.min()),
                "positive_fraction_max": float(per_group.max()),
                "n_blocks": int(len(per_group)),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["ifn_dose", "tnf_dose"]).reset_index(drop=True)


def _rank_biserial(bright: np.ndarray, dim: np.ndarray) -> float:
    """由 Mann–Whitney U 換算的 rank-biserial 效果量，正值代表亮組較大。"""
    statistic = stats.mannwhitneyu(bright, dim, alternative="two-sided").statistic
    return 2.0 * float(statistic) / (bright.size * dim.size) - 1.0


def paired_shape_contrast(
    labelled: pd.DataFrame,
    features: list[str],
    *,
    condition: str,
    min_per_group: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在同一張影像內比較亮細胞與暗細胞的形狀。

    每張同時含有足夠亮細胞與暗細胞的影像各算一次 rank-biserial，再對這些
    影像層級效果量做 Wilcoxon signed-rank 檢定。這樣 donor、passage、劑量、
    照明與焦平面都被影像本身配對掉。

    Returns:
        `(每個特徵的彙總表, 每張影像每個特徵的明細表)`。
    """
    subset = labelled[labelled["condition"] == condition]
    if subset.empty:
        raise BrightnessError(f"找不到條件 {condition}")

    detail_rows: list[dict[str, object]] = []
    for image_key, block in subset.groupby("image_key", sort=True):
        bright_block = block[block["ido_class"] == BRIGHT_LABEL]
        dim_block = block[block["ido_class"] == DARK_LABEL]
        if len(bright_block) < min_per_group or len(dim_block) < min_per_group:
            continue
        for name in features:
            bright_values = bright_block[name].dropna().to_numpy()
            dim_values = dim_block[name].dropna().to_numpy()
            if bright_values.size < min_per_group or dim_values.size < min_per_group:
                continue
            detail_rows.append(
                {
                    "image_key": image_key,
                    "b_id": block["b_id"].iloc[0],
                    "passage": int(block["passage"].iloc[0]),
                    "feature": name,
                    "rank_biserial": _rank_biserial(bright_values, dim_values),
                    "n_bright": int(bright_values.size),
                    "n_dim": int(dim_values.size),
                }
            )
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise BrightnessError(
            f"{condition} 沒有任何影像同時具備至少 {min_per_group} 顆亮細胞與暗細胞"
        )

    summary_rows: list[dict[str, object]] = []
    for name, block in detail.groupby("feature", sort=False):
        effects = block["rank_biserial"].to_numpy()
        if np.allclose(effects, 0.0):
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(effects, zero_method="zsplit").pvalue)
        summary_rows.append(
            {
                "feature": name,
                "feature_label": feature_label(name),
                "condition": condition,
                "effect_median": float(np.median(effects)),
                "abs_effect": float(abs(np.median(effects))),
                "effect_q1": float(np.percentile(effects, 25)),
                "effect_q3": float(np.percentile(effects, 75)),
                "n_images": int(len(effects)),
                "images_same_sign": int((np.sign(effects) == np.sign(np.median(effects))).sum()),
                "p_value": p_value,
                "n_bright": int(block["n_bright"].sum()),
                "n_dim": int(block["n_dim"].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["q_value_bh"] = benjamini_hochberg(summary["p_value"].to_numpy())
    summary = summary.sort_values("abs_effect", ascending=False, kind="mergesort")
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary, detail


def images_with_both_groups(
    labelled: pd.DataFrame, *, condition: str, min_per_group: int
) -> pd.DataFrame:
    """列出同時含有足夠亮細胞與暗細胞的影像，供配對影像庫挑選。"""
    subset = labelled[labelled["condition"] == condition]
    counts = (
        subset.groupby(["image_key", "ido_class"]).size().unstack(fill_value=0).reset_index()
    )
    for column in (BRIGHT_LABEL, DARK_LABEL):
        if column not in counts.columns:
            counts[column] = 0
    keep = counts[
        (counts[BRIGHT_LABEL] >= min_per_group) & (counts[DARK_LABEL] >= min_per_group)
    ].copy()
    keep["n_paired"] = keep[[BRIGHT_LABEL, DARK_LABEL]].min(axis=1)
    donors = subset.drop_duplicates("image_key").set_index("image_key")[["b_id", "passage"]]
    keep = keep.join(donors, on="image_key")
    # 依 donor 輪流取，避免整張圖都來自同一個 donor。
    keep = keep.sort_values("n_paired", ascending=False)
    keep["donor_order"] = keep.groupby("b_id").cumcount()
    return keep.sort_values(["donor_order", "n_paired"], ascending=[True, False])
