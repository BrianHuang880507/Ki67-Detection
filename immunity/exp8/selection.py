"""Exp8：挑選代表性細胞並算繪去背影像。

代表性細胞一律取「最接近該組中位數」，不是隨機也不是挑最好看的。挑選依據
是前幾個形狀特徵標準化後的距離，選中的細胞清單全部寫進 CSV 可追溯。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..exp6.gallery import (
    GalleryConfig,
    GalleryError,
    collect_crops,
    incell_ido_vmax,
    render_background_removed,
)
from .brightness import BRIGHT_LABEL, DARK_LABEL


def representative_cells(
    cells: pd.DataFrame, features: list[str], *, count: int, spread_key: str | None = None
) -> pd.DataFrame:
    """取最接近中位數的 `count` 顆細胞。

    每個特徵先用該組的中位數與四分位距標準化，再取各特徵標準化距離的
    平方和最小者。`spread_key` 給定時，先在每個分層各取一顆再補齊，避免
    代表性細胞全部來自同一個 donor。
    """
    if cells.empty:
        raise GalleryError("沒有可挑選的細胞")
    usable = cells.dropna(subset=features).copy()
    if usable.empty:
        raise GalleryError("挑選用的形狀特徵全為空值")

    distance = np.zeros(len(usable), dtype=float)
    for name in features:
        values = usable[name].to_numpy(dtype=float)
        centre = float(np.median(values))
        spread = float(np.percentile(values, 75) - np.percentile(values, 25))
        if spread <= 0:
            continue
        distance += ((values - centre) / spread) ** 2
    usable = usable.assign(_distance=np.sqrt(distance))

    if spread_key and spread_key in usable.columns:
        picks: list[int] = []
        for _, block in usable.groupby(spread_key, sort=True):
            picks.append(int(block["_distance"].idxmin()))
        remaining = usable.drop(index=picks).nsmallest(max(0, count - len(picks)), "_distance")
        chosen = pd.concat([usable.loc[picks], remaining])
    else:
        chosen = usable.nsmallest(count, "_distance")

    return chosen.nsmallest(count, "_distance").drop(columns="_distance")


def fixed_window_span(
    selection: pd.DataFrame, *, quantile: float = 0.95, margin: float = 1.2, minimum: int = 96
) -> int:
    """算出所有細胞共用的裁切視窗邊長。

    影像庫若讓每顆細胞各自貼合外框再縮放到同尺寸，大細胞和小細胞看起來會
    一樣大，「濃度越高細胞越長」「亮細胞比較小」這些結論在圖上就看不出來。
    改成固定視窗之後，圖上的相對大小才是真的。
    """
    column = "cell__MaxFeretDiameter"
    if column not in selection.columns:
        raise GalleryError(f"選集缺少 {column}，無法決定固定視窗大小")
    span = float(np.percentile(selection[column].dropna(), quantile * 100))
    return int(max(minimum, round(span * margin)))


def render_selection(
    data_root: Path, selection: pd.DataFrame, config: GalleryConfig
) -> tuple[dict[tuple[str, int], np.ndarray], float]:
    """把選中的細胞算繪成去背 RGB，所有細胞共用同一個顯示範圍。"""
    if selection.empty:
        raise GalleryError("沒有選中的細胞可算繪")
    crops = collect_crops(data_root, selection, config)
    vmax = incell_ido_vmax(crops)
    rendered = {
        key: render_background_removed(crop["ido"], crop["mask"], crop["nucleus"], vmax=vmax)
        for key, crop in crops.items()
    }
    return rendered, vmax


def cells_by_dose(
    labelled: pd.DataFrame,
    features: list[str],
    *,
    per_dose: int,
    dose_column: str = "ifn_dose",
    fixed_column: str = "tnf_dose",
) -> pd.DataFrame:
    """每個濃度各挑一組代表性細胞。"""
    subset = labelled[labelled[fixed_column] == 0]
    blocks: list[pd.DataFrame] = []
    for dose, block in subset.groupby(dose_column, sort=True):
        chosen = representative_cells(block, features, count=per_dose, spread_key="b_id")
        blocks.append(chosen.assign(row_dose=float(dose)))
    if not blocks:
        raise GalleryError("沒有任何濃度可挑選代表性細胞")
    return pd.concat(blocks, ignore_index=True)


def paired_cells_within_image(
    labelled: pd.DataFrame,
    features: list[str],
    image_keys: list[str],
) -> pd.DataFrame:
    """每張影像各挑一顆亮細胞與一顆暗細胞，兩者都取該組的中位數代表。"""
    blocks: list[pd.DataFrame] = []
    for image_key in image_keys:
        block = labelled[labelled["image_key"] == image_key]
        for group in (BRIGHT_LABEL, DARK_LABEL):
            candidates = block[block["ido_class"] == group]
            if candidates.empty:
                break
            chosen = representative_cells(candidates, features, count=1)
            blocks.append(chosen.assign(pair_group=group, pair_image=image_key))
    if not blocks:
        raise GalleryError("沒有任何影像可組成亮暗配對")
    return pd.concat(blocks, ignore_index=True)
