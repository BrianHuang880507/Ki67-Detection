"""Exp6：整張視野的去背算繪。

把整張影像的背景換成白色，只留下通過 QC 的細胞：綠色是 IDO 強度、藍色是
細胞核、紅線是分割輪廓。細胞維持在原本的座標，所以看得到密度與排列，
而不只是裁切後的單顆細胞。

只算繪出現在 cell-level 表裡的 `cell_label`，也就是有配到細胞核、沒有貼邊、
通過面積 QC 的細胞。畫面上看到的細胞，就是統計用到的細胞。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _new_figure, _save
from .gallery import GalleryError, load_image_bundle, render_background_removed


#: 預設對照的兩個極端條件。
LOW_CONDITION = "IFN0_TNF0"
HIGH_CONDITION = "IFN25_TNF50"


def pick_reference_fovs(
    fov: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    condition: str,
    count: int,
    ascending: bool,
    min_cells: int = 10,
    max_cells: int = 40,
) -> pd.DataFrame:
    """挑出細胞數適中、IDO 位於該條件極端的視野。

    細胞太少的視野看起來是空白，太多則互相重疊看不清楚，因此先用細胞數過濾，
    再取 IDO 的頭尾。
    """
    subset = fov[fov["condition"] == condition]
    subset = subset[subset["n_cells"].between(min_cells, max_cells)]
    if subset.empty:
        raise GalleryError(f"{condition} 沒有細胞數介於 {min_cells}–{max_cells} 的視野")
    ordered = subset.sort_values("IDO_score_ff", ascending=ascending)
    chosen = ordered.head(count).copy()
    paths = cells.drop_duplicates("image_key").set_index("image_key")[["pc_path", "ido_path"]]
    return chosen.join(paths, on="image_key")


def render_fov(
    data_root: Path,
    row: pd.Series,
    keep_labels: set[int],
    *,
    vmax: float,
) -> np.ndarray:
    """算繪單一視野的去背影像，只保留 `keep_labels` 內的細胞。"""
    bundle = load_image_bundle(data_root, row)
    keep = np.isin(bundle.cell_mask, list(keep_labels)) if keep_labels else np.zeros_like(
        bundle.cell_mask, dtype=bool
    )
    return render_background_removed(
        bundle.ido, keep, (bundle.nucleus_mask > 0) & keep, vmax=vmax
    )


def _incell_vmax(
    data_root: Path, rows: pd.DataFrame, labels_by_image: dict[str, set[int]], percentile: float
) -> float:
    """所有面板共用的綠色飽和值，只由通過 QC 的細胞內像素決定。"""
    pooled: list[np.ndarray] = []
    for row in rows.itertuples():
        bundle = load_image_bundle(data_root, pd.Series(row._asdict()))
        keep = np.isin(bundle.cell_mask, list(labels_by_image.get(str(row.image_key), set())))
        if keep.any():
            pooled.append(bundle.ido[keep])
    if not pooled:
        raise GalleryError("沒有任何細胞內像素可決定顯示範圍")
    return float(max(np.percentile(np.concatenate(pooled), percentile), 1.0))


def build_fov_views(
    cells: pd.DataFrame,
    fov: pd.DataFrame,
    data_root: Path,
    out_root: Path,
    *,
    per_condition: int = 3,
    low_condition: str = LOW_CONDITION,
    high_condition: str = HIGH_CONDITION,
    write_full_size: bool = True,
) -> dict[str, object]:
    """產生「未刺激 vs. 強刺激」的整張視野去背對照圖。"""
    selections = {
        low_condition: pick_reference_fovs(
            fov, cells, condition=low_condition, count=per_condition, ascending=True
        ),
        high_condition: pick_reference_fovs(
            fov, cells, condition=high_condition, count=per_condition, ascending=False
        ),
    }
    chosen = pd.concat(selections.values())
    labels_by_image = {
        str(key): set(int(value) for value in block["cell_label"])
        for key, block in cells[cells["image_key"].isin(chosen["image_key"])].groupby("image_key")
    }
    vmax = _incell_vmax(data_root, chosen, labels_by_image, percentile=99.0)

    rendered: dict[str, np.ndarray] = {}
    for row in chosen.itertuples():
        rendered[str(row.image_key)] = render_fov(
            data_root,
            pd.Series(row._asdict()),
            labels_by_image.get(str(row.image_key), set()),
            vmax=vmax,
        )

    figures_dir = out_root / "figures"
    fig = _new_figure(per_condition * 4.6, 8.4)
    axes = fig.subplots(2, per_condition, squeeze=False)
    for row_index, (condition, block) in enumerate(selections.items()):
        for column, entry in enumerate(block.itertuples()):
            ax = axes[row_index][column]
            ax.imshow(rendered[str(entry.image_key)], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d4d3cf")
            ax.set_title(
                f"{entry.image_key}　{int(entry.n_cells)} 顆細胞　"
                f"影像 IDO {entry.IDO_score_ff:.2f}",
                fontsize=8.5,
                color=TEXT_SECONDARY,
                pad=4,
            )
            if column == 0:
                ax.set_ylabel(condition, fontsize=12, color=TEXT_PRIMARY, fontweight="bold", labelpad=8)

    fig.suptitle(
        "整張視野去背：未刺激 vs. 強刺激",
        fontsize=15,
        color=TEXT_PRIMARY,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=1.045,
    )
    fig.text(
        0.02,
        1.008,
        "白底＝背景已移除　|　綠色＝IDO 強度（六格共用同一顯示範圍）　|　"
        "藍色＝細胞核　|　紅線＝分割輪廓　|　只畫通過 QC、有配到細胞核的細胞",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        ha="left",
    )
    fig.tight_layout(h_pad=2.4, w_pad=1.2)
    comparison_path = _save(fig, figures_dir / "fig07_fov_background_removed.png")

    full_size: list[str] = []
    if write_full_size:
        full_dir = out_root / "fov_background_removed"
        full_dir.mkdir(parents=True, exist_ok=True)
        for row in chosen.itertuples():
            image = rendered[str(row.image_key)]
            height, width = image.shape[:2]
            single = plt.figure(figsize=(width / 200, height / 200), dpi=200, facecolor=SURFACE)
            ax = single.add_axes((0.0, 0.0, 1.0, 1.0))
            ax.imshow(image, interpolation="nearest")
            ax.set_axis_off()
            name = f"{row.condition}_{row.image_key}.png"
            single.savefig(full_dir / name, facecolor="white", dpi=200)
            plt.close(single)
            full_size.append(name)

    return {
        "comparison_figure": comparison_path,
        "full_size_files": full_size,
        "vmax": vmax,
        "selected": chosen[
            ["image_key", "condition", "b_id", "passage", "n_cells", "IDO_score_ff"]
        ].copy(),
    }
