"""Exp6 第一部分：IDO 與外觀特徵對 IFN-γ／TNF-α 劑量的關聯性。

分析單位是「一張影像」，不是「一顆細胞」：劑量施加在整個孔上，同一張影像
內的細胞共享同一個劑量，把 23,012 顆細胞當成獨立觀測會把 p 值灌到無意義。

劑量軸的定義見 `dataset.dose_slices`，本批次不是完整 factorial design。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from .dataset import feature_columns, feature_label


@dataclass(frozen=True)
class DoseAxis:
    """一條可解釋的劑量軸。"""

    name: str
    dose_column: str
    strata: tuple[str, ...]
    title: str
    description: str


IFN_AXIS = DoseAxis(
    name="IFN",
    dose_column="ifn_dose",
    strata=(),
    title="IFN-γ 劑量（TNF-α = 0）",
    description="TNF-α 固定為 0，IFN-γ 0／25／50／100 ng/mL",
)

TNF_AXIS = DoseAxis(
    name="TNF",
    dose_column="tnf_dose",
    strata=("ifn_dose",),
    title="TNF-α 劑量（固定 IFN-γ）",
    description="IFN-γ 固定為 0 或 25 ng/mL，TNF-α 0／25／50 ng/mL；相關係數在 IFN 區塊內計算",
)

DOSE_AXES = (IFN_AXIS, TNF_AXIS)

GROUP_STRATA = ("b_id", "passage")


def _pearson_on_centered_ranks(
    x_ranks: np.ndarray, y_ranks: np.ndarray, n_strata: int
) -> tuple[float, float]:
    """對區塊內置中的 rank 求 Pearson r，並以 t 檢定給近似 p 值。

    Args:
        x_ranks: 已在區塊內置中的 x rank。
        y_ranks: 已在區塊內置中的 y rank。
        n_strata: 區塊數，用來扣自由度。

    Returns:
        `(r, p)`；資料不足或任一邊變異為 0 時回傳 `(nan, nan)`。
    """
    n = x_ranks.size
    degrees = n - n_strata - 1
    if degrees <= 0:
        return float("nan"), float("nan")
    sx = float(np.dot(x_ranks, x_ranks))
    sy = float(np.dot(y_ranks, y_ranks))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan"), float("nan")
    r = float(np.dot(x_ranks, y_ranks) / sqrt(sx * sy))
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * sqrt(degrees / (1.0 - r * r))
    p = float(2.0 * stats.t.sf(abs(t_stat), degrees))
    return r, p


def stratified_spearman(
    frame: pd.DataFrame, x: str, y: str, strata: tuple[str, ...] = ()
) -> tuple[float, float, int]:
    """區塊內 Spearman 相關係數。

    `strata` 為空時退化成一般 Spearman。有 strata 時先在每個區塊內把 x、y
    各自轉成 rank 並置中，再合併求 Pearson r——等價於控制區塊變數後的
    partial Spearman，可避免用 IFN 區塊差異冒充 TNF 效果。

    Returns:
        `(rho, p, n)`。
    """
    columns = [x, y, *strata]
    data = frame[columns].dropna()
    if len(data) < 3:
        return float("nan"), float("nan"), len(data)
    # 常數欄位（例如影像中位數全為 0 的 MinIntensityEdge）沒有定義的 rho。
    if data[x].nunique() < 2 or data[y].nunique() < 2:
        return float("nan"), float("nan"), len(data)

    if not strata:
        rho, p = stats.spearmanr(data[x], data[y])
        return float(rho), float(p), len(data)

    key = list(strata)
    grouped = data.groupby(key, sort=False)
    ranked = data.assign(_x_rank=grouped[x].rank(), _y_rank=grouped[y].rank())
    regrouped = ranked.groupby(key, sort=False)
    x_centered = (ranked["_x_rank"] - regrouped["_x_rank"].transform("mean")).to_numpy()
    y_centered = (ranked["_y_rank"] - regrouped["_y_rank"].transform("mean")).to_numpy()
    n_strata = int(grouped.ngroups)
    rho, p = _pearson_on_centered_ranks(x_centered, y_centered, n_strata)
    return rho, p, len(data)


def per_group_spearman(
    frame: pd.DataFrame, x: str, y: str, strata: tuple[str, ...]
) -> pd.DataFrame:
    """在每個 donor×passage（必要時再乘 IFN 區塊）內各算一次 Spearman。

    用來檢查合併後的相關係數是不是由某一個 donor 或 passage 帶動的。
    """
    keys = list(GROUP_STRATA) + [name for name in strata]
    rows: list[dict[str, object]] = []
    for values, block in frame.groupby(keys, sort=True):
        block = block[[x, y]].dropna()
        if len(block) < 5 or block[x].nunique() < 2 or block[y].nunique() < 2:
            continue
        rho, p = stats.spearmanr(block[x], block[y])
        record: dict[str, object] = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        record.update({"rho": float(rho), "p_value": float(p), "n_images": int(len(block))})
        rows.append(record)
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR 校正，回傳 q 值。"""
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not finite.any():
        return q
    subset = p[finite]
    order = np.argsort(subset)
    ranked = subset[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[finite] = restored
    return q


def _dose_magnitude(frame: pd.DataFrame, dose_column: str, target: str) -> dict[str, float]:
    """回傳最低與最高劑量的 target 中位數，以及兩者相差幾個灰階。

    只有 rho 會誤導：TNF 在 IFN=0 時 rho 顯著為正，但灰階差不到 0.1，
    這一欄就是為了讓「顯著」與「有多大」分開看。
    """
    doses = sorted(frame[dose_column].unique())
    if len(doses) < 2:
        return {"dose_min": float("nan"), "dose_max": float("nan"), "median_at_min": float("nan"),
                "median_at_max": float("nan"), "delta_grey_levels": float("nan")}
    low = float(np.median(frame.loc[frame[dose_column] == doses[0], target]))
    high = float(np.median(frame.loc[frame[dose_column] == doses[-1], target]))
    return {
        "dose_min": float(doses[0]),
        "dose_max": float(doses[-1]),
        "median_at_min": low,
        "median_at_max": high,
        "delta_grey_levels": high - low,
    }


def correlate_ido_with_dose(slices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """IDO 對兩條劑量軸的關聯性，含 flat-field 前後對照與逐 IFN 區塊明細。

    TNF 軸額外拆成 IFN=0 與 IFN=25 兩列：本批資料裡 TNF 單獨作用幾乎沒有
    效果，只有在 IFN 存在時才把 IDO 推上去，合併成一個數字會蓋掉這件事。
    """
    rows: list[dict[str, object]] = []
    for axis in DOSE_AXES:
        frame = slices[axis.name.lower()]
        for target, target_label in (
            ("IDO_score_ff", "IDO_score_ff（flat-field 校正，主要）"),
            ("IDO_score_scalar", "IDO_score_scalar（僅扣純量背景，對照）"),
        ):
            if target not in frame.columns:
                continue
            subsets: list[tuple[str, pd.DataFrame, tuple[str, ...]]] = [("全部", frame, axis.strata)]
            if axis.strata == ("ifn_dose",):
                subsets.extend(
                    (f"IFN-γ = {int(level)}", frame[frame["ifn_dose"] == level], ())
                    for level in sorted(frame["ifn_dose"].unique())
                )
            for block_label, subset, strata in subsets:
                rho, p, n = stratified_spearman(subset, axis.dose_column, target, strata)
                per_group = per_group_spearman(subset, axis.dose_column, target, strata)
                rows.append(
                    {
                        "dose_axis": axis.name,
                        "dose_axis_title": axis.title,
                        "block": block_label,
                        "target": target,
                        "target_label": target_label,
                        "spearman_rho": rho,
                        "p_value": p,
                        "n_images": n,
                        "n_blocks": len(per_group),
                        "rho_median_within_block": float(per_group["rho"].median())
                        if not per_group.empty
                        else float("nan"),
                        "blocks_same_sign": int((np.sign(per_group["rho"]) == np.sign(rho)).sum())
                        if not per_group.empty
                        else 0,
                        **_dose_magnitude(subset, axis.dose_column, target),
                    }
                )
    return pd.DataFrame(rows)


def ido_dose_response(fov: pd.DataFrame) -> pd.DataFrame:
    """三條描述性劑量曲線的影像層級摘要。"""
    curves = {
        "IFN-γ（TNF-α = 0）": (fov["tnf_dose"] == 0, "ifn_dose"),
        "TNF-α（IFN-γ = 0）": (fov["ifn_dose"] == 0, "tnf_dose"),
        "TNF-α（IFN-γ = 25）": (fov["ifn_dose"] == 25, "tnf_dose"),
    }
    rows: list[dict[str, object]] = []
    for curve_name, (mask, dose_column) in curves.items():
        subset = fov[mask]
        for dose, block in subset.groupby(dose_column):
            values = block["IDO_score_ff"].to_numpy()
            rows.append(
                {
                    "curve": curve_name,
                    "dose_ng_ml": float(dose),
                    "n_images": int(len(block)),
                    "n_cells": int(block["n_cells"].sum()),
                    "ido_median": float(np.median(values)),
                    "ido_q1": float(np.percentile(values, 25)),
                    "ido_q3": float(np.percentile(values, 75)),
                    "ido_mean": float(np.mean(values)),
                }
            )
    return pd.DataFrame(rows)


def correlate_features_with_dose(slices: dict[str, pd.DataFrame], features: list[str]) -> pd.DataFrame:
    """每個外觀特徵對兩條劑量軸的關聯性排名表。"""
    rows: list[dict[str, object]] = []
    for axis in DOSE_AXES:
        frame = slices[axis.name.lower()]
        axis_rows: list[dict[str, object]] = []
        for name in features:
            rho, p, n = stratified_spearman(frame, axis.dose_column, name, axis.strata)
            per_group = per_group_spearman(frame, axis.dose_column, name, axis.strata)
            same_sign = (
                int((np.sign(per_group["rho"]) == np.sign(rho)).sum()) if not per_group.empty else 0
            )
            axis_rows.append(
                {
                    "dose_axis": axis.name,
                    "dose_axis_title": axis.title,
                    "feature": name,
                    "feature_label": feature_label(name),
                    "spearman_rho": rho,
                    "abs_rho": abs(rho) if np.isfinite(rho) else float("nan"),
                    "p_value": p,
                    "n_images": n,
                    "n_blocks": len(per_group),
                    "blocks_same_sign": same_sign,
                    "rho_median_within_block": float(per_group["rho"].median())
                    if not per_group.empty
                    else float("nan"),
                }
            )
        axis_frame = pd.DataFrame(axis_rows)
        axis_frame["q_value_bh"] = benjamini_hochberg(axis_frame["p_value"].to_numpy())
        axis_frame = axis_frame.sort_values("abs_rho", ascending=False, kind="mergesort")
        axis_frame["rank"] = np.arange(1, len(axis_frame) + 1)
        rows.extend(axis_frame.to_dict("records"))
    return pd.DataFrame(rows)


def top_features(table: pd.DataFrame, axis_name: str, top_n: int = 10) -> pd.DataFrame:
    """取某條劑量軸相關性最強的前 N 個特徵。"""
    subset = table[table["dose_axis"] == axis_name]
    return subset.nsmallest(top_n, "rank").sort_values("rank")


def correlate_features_with_ido(
    cells: pd.DataFrame, fov: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """外觀特徵對 IDO 本身的關聯性（影像層級，並在 condition 內重算）。

    `rho_within_condition` 控制刺激條件後才算，可分辨「特徵真的跟 IDO 有關」
    與「特徵只是跟著劑量一起變」。
    """
    rows: list[dict[str, object]] = []
    for name in features:
        rho, p, n = stratified_spearman(fov, name, "IDO_score_ff", ())
        rho_within, p_within, _ = stratified_spearman(
            fov, name, "IDO_score_ff", ("condition_index",)
        )
        cell_rho, _, _ = stratified_spearman(cells, name, "IDO_score_ff", ("image_key",))
        rows.append(
            {
                "feature": name,
                "feature_label": feature_label(name),
                "rho_image_level": rho,
                "abs_rho": abs(rho) if np.isfinite(rho) else float("nan"),
                "p_value": p,
                "n_images": n,
                "rho_within_condition": rho_within,
                "p_within_condition": p_within,
                "rho_cell_level_within_image": cell_rho,
            }
        )
    table = pd.DataFrame(rows)
    table["q_value_bh"] = benjamini_hochberg(table["p_value"].to_numpy())
    table = table.sort_values("abs_rho", ascending=False, kind="mergesort")
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def build_all_tables(cells: pd.DataFrame, fov: pd.DataFrame, slices: dict[str, pd.DataFrame]):
    """一次產生第一部分的四張表。"""
    features = feature_columns(cells)
    return {
        "ido_dose_correlations": correlate_ido_with_dose(slices),
        "ido_dose_response": ido_dose_response(fov),
        "feature_dose_correlations": correlate_features_with_dose(slices, features),
        "feature_ido_correlations": correlate_features_with_ido(cells, fov, features),
    }
