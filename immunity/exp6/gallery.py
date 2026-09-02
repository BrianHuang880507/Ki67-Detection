"""Exp6 第二部分：切出 IDO 亮／暗細胞的影像。

兩種切法都會產生：

- `global`：不分刺激條件，取全部細胞 IDO_score_ff 的上／下 10%。
  這是最直觀的「亮 vs 不亮」，但亮的一群幾乎必然來自 IFN-γ 刺激組。
- `matched`：在同一 donor×passage×刺激條件內取上／下 10%。
  劑量被固定住，剩下的差異才是「同樣刺激下，為什麼有些細胞亮」。

影像本身不重算：mask 直接讀 Exp3 的 cache，IDO 灰階轉換與 pinned pipeline
（`cv2.IMREAD_GRAYSCALE`）一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from .dataset import feature_columns, feature_label
from .dose_association import benjamini_hochberg
from .figures import GRID, NEUTRAL, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _new_figure, _save


IDO_CMAP = LinearSegmentedColormap.from_list("ido_green", ["#000000", "#0f3d1e", "#1baf7a", "#d8f7e6"])
BRIGHT_COLOR = "#1baf7a"
DIM_COLOR = "#4a3aa7"

#: 去背算繪的三個顏色，沿用螢光影像的慣例：白底、綠色訊號、藍色細胞核、紅色輪廓。
WHITE_BG = (1.0, 1.0, 1.0)
NUCLEUS_BLUE = (0.09, 0.09, 0.78)
OUTLINE_RED = (0.85, 0.06, 0.06)

MATCH_STRATA = ["group_id", "condition_index"]


class GalleryError(RuntimeError):
    """表示影像或 mask 無法取得。"""


@dataclass(frozen=True)
class GalleryConfig:
    """影像庫參數。"""

    decile_pct: float = 10.0
    tiles_per_group: int = 30
    crops_per_group: int = 40
    tile_pixels: int = 180
    padding: float = 1.35
    seed: int = 0
    #: 固定裁切視窗邊長（原始像素）。設定後每顆細胞都用同樣大小的視窗，
    #: 縮放到 tile 之後相對大小才會保留；`None` 則依各自外框加邊界。
    fixed_span: int | None = None
    #: 算繪前先扣掉該張影像的細胞外背景中位數。刺激孔的背景本身就偏亮，
    #: 不扣背景時「暗細胞」在畫面上看起來和亮細胞一樣綠，與 IDO 分數不符。
    subtract_background: bool = False


def assign_brightness_groups(
    cells: pd.DataFrame, *, decile_pct: float, strata: list[str] | None
) -> pd.DataFrame:
    """標記每顆細胞屬於 IDO 亮組、暗組或中間。

    Args:
        cells: cell-level 表，需含 `IDO_score_ff`。
        decile_pct: 上／下多少百分比視為亮／暗。
        strata: 分層欄位；`None` 代表不分層（global 切法）。

    Returns:
        原表加上 `ido_percentile` 與 `brightness_group`（bright／dim／mid）。
    """
    frame = cells.copy()
    if strata:
        frame["ido_percentile"] = frame.groupby(strata)["IDO_score_ff"].rank(pct=True) * 100.0
    else:
        frame["ido_percentile"] = frame["IDO_score_ff"].rank(pct=True) * 100.0
    frame["brightness_group"] = "mid"
    frame.loc[frame["ido_percentile"] >= 100.0 - decile_pct, "brightness_group"] = "bright"
    frame.loc[frame["ido_percentile"] <= decile_pct, "brightness_group"] = "dim"
    return frame


def contrast_features(labelled: pd.DataFrame, features: list[str], *, decile_pct: float) -> pd.DataFrame:
    """比較亮組與暗組的外觀特徵，回傳 rank-biserial 效果量排名表。

    使用 Mann–Whitney U 與由 U 換算的 rank-biserial correlation：
    `r = 2U/(n1·n2) − 1`，正值代表亮細胞該特徵較大。這是非參數效果量，
    不假設常態，也不受特徵單位影響。
    """
    bright = labelled[labelled["brightness_group"] == "bright"]
    dim = labelled[labelled["brightness_group"] == "dim"]
    if bright.empty or dim.empty:
        raise GalleryError("亮組或暗組沒有細胞，請調整 decile_pct")

    rows: list[dict[str, object]] = []
    for name in features:
        left = bright[name].dropna().to_numpy()
        right = dim[name].dropna().to_numpy()
        if left.size < 5 or right.size < 5:
            continue
        u_statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
        rank_biserial = 2.0 * float(u_statistic) / (left.size * right.size) - 1.0
        rows.append(
            {
                "feature": name,
                "feature_label": feature_label(name),
                "rank_biserial": rank_biserial,
                "abs_effect": abs(rank_biserial),
                "p_value": float(p_value),
                "median_bright": float(np.median(left)),
                "median_dim": float(np.median(right)),
                "n_bright": int(left.size),
                "n_dim": int(right.size),
                "decile_pct": decile_pct,
            }
        )
    table = pd.DataFrame(rows)
    table["q_value_bh"] = benjamini_hochberg(table["p_value"].to_numpy())
    table = table.sort_values("abs_effect", ascending=False, kind="mergesort")
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def sample_cells(
    labelled: pd.DataFrame, group: str, count: int, *, seed: int, strata: list[str] | None
) -> pd.DataFrame:
    """從亮組或暗組抽樣，盡量把 donor／passage／條件攤平。"""
    subset = labelled[labelled["brightness_group"] == group]
    if subset.empty:
        raise GalleryError(f"{group} 組沒有細胞")
    if len(subset) <= count:
        return subset.copy()
    spread_keys = strata or ["group_id", "condition_index"]
    rng = np.random.default_rng(seed)
    # 每個分層先抽一顆再打散，避免名額不夠時只留下排序最前面的 donor。
    picks: list[int] = []
    for _, block in subset.groupby(spread_keys, sort=True):
        picks.append(int(rng.choice(block.index.to_numpy())))
    rng.shuffle(picks)
    picks = picks[:count]
    if len(picks) < count:
        remaining = subset.index.difference(pd.Index(picks))
        extra = rng.choice(remaining.to_numpy(), size=count - len(picks), replace=False)
        picks.extend(int(value) for value in extra)
    ordered = subset.loc[picks].sort_values("IDO_score_ff", ascending=(group == "dim"))
    return ordered


def _read_grayscale(path: Path) -> np.ndarray:
    """以 pinned pipeline 相同的方式讀成二維灰階陣列。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise GalleryError(f"無法讀取影像：{path}")
    return np.asarray(image, dtype=np.float32)


def _mask_path(data_root: Path, group_id: str, image_key: str) -> Path:
    """組出 Exp3 mask cache 路徑。"""
    return data_root / "immunity/outputs/exp3/feature_cache/masks" / group_id / f"{image_key}.npz"


@dataclass
class ImageBundle:
    """一張影像所需的四個陣列。"""

    phase: np.ndarray
    ido: np.ndarray
    cell_mask: np.ndarray
    nucleus_mask: np.ndarray


def load_image_bundle(data_root: Path, row: pd.Series) -> ImageBundle:
    """載入單張影像的 phase、IDO 與 whole-cell／nucleus mask。"""
    mask_file = _mask_path(data_root, str(row["group_id"]), str(row["image_key"]))
    if not mask_file.exists():
        raise GalleryError(f"找不到 mask cache：{mask_file}")
    with np.load(mask_file) as archive:
        cell_mask = np.asarray(archive["cell_mask"])
        nucleus_mask = np.asarray(archive["nucleus_mask"])
    phase = _read_grayscale(Path(str(row["pc_path"])))
    ido = _read_grayscale(Path(str(row["ido_path"])))
    shapes = {phase.shape, ido.shape, cell_mask.shape, nucleus_mask.shape}
    if len(shapes) != 1:
        raise GalleryError(f"{row['image_key']} 的影像與 mask 尺寸不符")
    return ImageBundle(
        phase=phase, ido=ido, cell_mask=cell_mask, nucleus_mask=nucleus_mask
    )


def render_background_removed(
    ido: np.ndarray,
    cell: np.ndarray,
    nucleus: np.ndarray,
    *,
    vmax: float,
    ring_width: int = 2,
    draw_outline: bool = True,
) -> np.ndarray:
    """把 IDO 訊號畫成白底去背影像。

    細胞外的所有像素換成白色，細胞內以「黑→綠」呈現 IDO 強度，細胞核填藍色
    並加紅色環。輸出是 float RGB，可直接餵給 `imshow`。

    Args:
        ido: 二維 IDO 灰階影像。
        cell: 與 `ido` 同尺寸的細胞布林遮罩。
        nucleus: 細胞核布林遮罩，已限制在 `cell` 之內。
        vmax: 綠色飽和對應的灰階值；所有面板共用同一個值才能互相比較。
        ring_width: 細胞核紅環的粗細（像素）。
        draw_outline: 是否額外畫出細胞外輪廓。
    """
    if ido.shape != cell.shape or ido.shape != nucleus.shape:
        raise GalleryError("去背算繪的影像與遮罩尺寸不符")
    cell_bool = cell.astype(bool)
    nucleus_bool = nucleus.astype(bool) & cell_bool

    rgb = np.empty((*ido.shape, 3), dtype=np.float32)
    rgb[...] = WHITE_BG
    green = np.clip(ido / max(vmax, 1e-6), 0.0, 1.0)
    rgb[cell_bool] = 0.0
    rgb[..., 1] = np.where(cell_bool, green, np.float32(WHITE_BG[1]))

    if draw_outline:
        eroded = cv2.erode(cell_bool.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        rgb[cell_bool & (eroded == 0)] = OUTLINE_RED

    if nucleus_bool.any():
        rgb[nucleus_bool] = NUCLEUS_BLUE
        if ring_width > 0:
            kernel = np.ones((3, 3), np.uint8)
            grown = cv2.dilate(nucleus_bool.astype(np.uint8), kernel, iterations=ring_width)
            ring = (grown > 0) & ~nucleus_bool & cell_bool
            rgb[ring] = OUTLINE_RED
    return rgb


def crop_cell(
    bundle: ImageBundle,
    cell_label: int,
    *,
    padding: float,
    tile_pixels: int | None,
    fixed_span: int | None = None,
) -> dict[str, np.ndarray]:
    """裁出單顆細胞的 phase／IDO 影像與遮罩。

    `tile_pixels` 為 `None` 時保留原始像素尺寸（供單顆細胞出圖用），否則統一
    重採樣成 `tile_pixels` 見方（供影像庫排版用）：

    - `phase`、`ido`：帶周圍環境的原始裁切。
    - `phase_segmented`、`ido_segmented`：只留下該細胞 mask 內的像素（背景為 0）。
    - `mask`：該細胞的布林遮罩，供畫輪廓用。
    - `nucleus`：該細胞內的細胞核遮罩，供去背算繪填藍色用。
    """
    ys, xs = np.nonzero(bundle.cell_mask == cell_label)
    if xs.size == 0:
        raise GalleryError(f"mask 內找不到 cell_label={cell_label}")
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    if fixed_span is not None:
        # 所有細胞共用同一個視窗大小，縮放後相對大小才不會被抹掉。
        half = float(fixed_span) / 2.0
    else:
        span = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        half = max(32.0, span * padding / 2.0)
    height, width = bundle.cell_mask.shape
    x0 = int(round(center_x - half))
    x1 = int(round(center_x + half))
    y0 = int(round(center_y - half))
    y1 = int(round(center_y + half))

    def _window(array: np.ndarray, fill: float) -> np.ndarray:
        """取出視窗；超出影像邊界的部分補 `fill`，維持視窗尺寸不變。"""
        out = np.full((y1 - y0, x1 - x0), fill, dtype=np.float32)
        src_y0, src_y1 = max(0, y0), min(height, y1)
        src_x0, src_x1 = max(0, x0), min(width, x1)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[src_y0 - y0 : src_y1 - y0, src_x0 - x0 : src_x1 - x0] = array[
                src_y0:src_y1, src_x0:src_x1
            ]
        return out

    mask = (_window(bundle.cell_mask.astype(np.float32), 0.0) == float(cell_label)).astype(
        np.float32
    )
    nucleus = ((_window(bundle.nucleus_mask.astype(np.float32), 0.0) > 0) & (mask > 0)).astype(
        np.float32
    )
    phase = _window(bundle.phase, float(np.median(bundle.phase)))
    ido = _window(bundle.ido, float(np.median(bundle.ido)))

    def _resize(array: np.ndarray, nearest: bool = False) -> np.ndarray:
        if tile_pixels is None:
            return array.copy()
        mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(array, (tile_pixels, tile_pixels), interpolation=mode)

    mask_resized = _resize(mask, nearest=True)
    return {
        "phase": _resize(phase),
        "ido": _resize(ido),
        "phase_segmented": _resize(phase) * mask_resized,
        "ido_segmented": _resize(ido) * mask_resized,
        "mask": mask_resized,
        "nucleus": _resize(nucleus, nearest=True),
    }


def collect_crops(
    data_root: Path,
    selection: pd.DataFrame,
    config: GalleryConfig,
    *,
    include_native: bool = False,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """把選中的細胞逐張影像讀進來，避免同一張影像重複解碼。

    `include_native` 為真時，每個 crop 額外掛一份未經重採樣的 `native` 子字典，
    供單顆細胞出圖使用；影像庫排版仍用統一尺寸的版本。
    """
    crops: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for image_key, block in selection.groupby("image_key", sort=False):
        bundle = load_image_bundle(data_root, block.iloc[0])
        if config.subtract_background:
            background = bundle.ido[bundle.cell_mask == 0]
            if background.size:
                bundle = ImageBundle(
                    phase=bundle.phase,
                    ido=np.clip(bundle.ido - float(np.median(background)), 0.0, None),
                    cell_mask=bundle.cell_mask,
                    nucleus_mask=bundle.nucleus_mask,
                )
        for row in block.itertuples():
            crop = crop_cell(
                bundle,
                int(row.cell_label),
                padding=config.padding,
                tile_pixels=config.tile_pixels,
                fixed_span=config.fixed_span,
            )
            if include_native:
                crop["native"] = crop_cell(
                    bundle, int(row.cell_label), padding=config.padding, tile_pixels=None
                )
            crops[(str(image_key), int(row.cell_label))] = crop
    return crops


def write_nobg_cells(
    selection: pd.DataFrame,
    crops: dict[tuple[str, int], dict[str, np.ndarray]],
    out_dir: Path,
) -> list[dict[str, object]]:
    """每顆細胞輸出一張純去背 PNG：一張圖一顆細胞，沒有邊框、標題或座標軸。

    使用未經重採樣的原始像素，所以解析度由細胞本身大小決定，貼進簡報不會糊。
    顯示範圍由所有細胞的細胞內像素共同決定，亮暗才可以互相比較。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    natives = {key: crop["native"] for key, crop in crops.items() if "native" in crop}
    if not natives:
        raise GalleryError("collect_crops 未產生 native 版本，無法輸出單顆細胞圖")
    vmax = incell_ido_vmax(natives)

    records: list[dict[str, object]] = []
    for row in selection.itertuples():
        native = natives[(str(row.image_key), int(row.cell_label))]
        rgb = render_background_removed(
            native["ido"], native["mask"], native["nucleus"], vmax=vmax
        )
        filename = f"{row.brightness_group}_{row.image_key}_cell{int(row.cell_label):04d}.png"
        plt.imsave(out_dir / filename, np.clip(rgb, 0.0, 1.0))
        records.append(
            {
                "image_key": row.image_key,
                "cell_label": int(row.cell_label),
                "nobg_file": filename,
                "nobg_pixels": int(rgb.shape[0]),
            }
        )
    return records


def _ido_display_range(crops: dict[tuple[str, int], dict[str, np.ndarray]]) -> tuple[float, float]:
    """所有 tile 共用一組 IDO 顯示範圍，否則亮暗對比會被自動縮放抹掉。"""
    values = np.concatenate([crop["ido"].ravel() for crop in crops.values()])
    return 0.0, float(np.percentile(values, 99.5))


def incell_ido_vmax(
    crops: dict[tuple[str, int], dict[str, np.ndarray]], *, percentile: float = 99.0
) -> float:
    """去背算繪的綠色飽和值，只由細胞內像素決定。

    去背之後畫面上超過九成是白色背景，若沿用整張影像的百分位，亮度會被背景
    拉低，所有細胞都會偏暗。
    """
    values = [crop["ido"][crop["mask"] > 0.5] for crop in crops.values()]
    pooled = np.concatenate([block for block in values if block.size])
    if pooled.size == 0:
        raise GalleryError("沒有任何細胞內像素可決定顯示範圍")
    return float(max(np.percentile(pooled, percentile), 1.0))


def _draw_tile(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    cmap,
    vmin: float,
    vmax: float,
    outline: np.ndarray | None,
    outline_color: str,
) -> None:
    """畫一格 tile，必要時疊上細胞輪廓。RGB 影像直接顯示，不套 colormap。"""
    if image.ndim == 3:
        ax.imshow(image, interpolation="nearest")
    else:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if outline is not None:
        ax.contour(outline, levels=[0.5], colors=[outline_color], linewidths=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_gallery(
    selection: pd.DataFrame,
    crops: dict[tuple[str, int], dict[str, np.ndarray]],
    out_path: Path,
    *,
    channel: str,
    title: str,
    subtitle: str,
    columns: int = 10,
) -> Path:
    """畫「亮組在上、暗組在下」的影像庫。

    `channel` 為 `"ido_nobg"` 時改走去背算繪：白底、綠色 IDO、藍色細胞核、
    紅色輪廓，綠色飽和值只由細胞內像素決定。
    """
    no_background = channel == "ido_nobg"
    vmin, vmax = _ido_display_range(crops)
    is_ido = channel.startswith("ido")
    cmap = IDO_CMAP if is_ido else "gray"
    if no_background:
        vmax = incell_ido_vmax(crops)
    elif not is_ido:
        phase_values = np.concatenate([crop["phase"].ravel() for crop in crops.values()])
        vmin = float(np.percentile(phase_values, 1))
        vmax = float(np.percentile(phase_values, 99))

    blocks = [
        ("bright", "IDO 亮（上 10%）", BRIGHT_COLOR),
        ("dim", "IDO 暗（下 10%）", DIM_COLOR),
    ]
    rows_per_block = max(
        1,
        int(np.ceil(max(len(selection[selection["brightness_group"] == name]) for name, _, _ in blocks) / columns)),
    )
    total_rows = rows_per_block * len(blocks)
    fig = _new_figure(columns * 1.32, total_rows * 1.48 + 1.15)
    grid = fig.add_gridspec(
        total_rows,
        columns,
        hspace=0.32,
        wspace=0.06,
        top=0.865,
        bottom=0.035,
        left=0.035,
        right=0.985,
    )

    for block_index, (name, block_title, color) in enumerate(blocks):
        subset = selection[selection["brightness_group"] == name]
        for position, row in enumerate(subset.itertuples()):
            if position >= rows_per_block * columns:
                break
            grid_row = block_index * rows_per_block + position // columns
            grid_col = position % columns
            ax = fig.add_subplot(grid[grid_row, grid_col])
            crop = crops[(str(row.image_key), int(row.cell_label))]
            if no_background:
                tile = render_background_removed(
                    crop["ido"], crop["mask"], crop["nucleus"], vmax=vmax
                )
                tile_outline = None
            else:
                tile = crop[channel]
                tile_outline = crop["mask"] if not channel.endswith("segmented") else None
            _draw_tile(
                ax,
                tile,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                outline=tile_outline,
                outline_color=color,
            )
            ax.set_title(
                f"{row.condition}\nIDO {row.IDO_score_ff:.1f}",
                fontsize=6.2,
                color=TEXT_SECONDARY,
                pad=2.0,
                linespacing=1.15,
            )
            if grid_col == 0:
                ax.set_ylabel(
                    block_title if position == 0 else "",
                    fontsize=9.5,
                    color=color,
                    fontweight="bold",
                    rotation=90,
                    labelpad=6,
                )

    fig.suptitle(title, fontsize=14, color=TEXT_PRIMARY, fontweight="bold", x=0.035, ha="left", y=0.985)
    fig.text(0.035, 0.925, subtitle, fontsize=9.5, color=TEXT_SECONDARY, ha="left")
    return _save(fig, out_path)


def write_cell_crops(
    selection: pd.DataFrame,
    crops: dict[tuple[str, int], dict[str, np.ndarray]],
    out_dir: Path,
) -> list[dict[str, object]]:
    """每顆細胞輸出一張五格 PNG，最後一格是白底去背版本。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    vmin_ido, vmax_ido = _ido_display_range(crops)
    vmax_nobg = incell_ido_vmax(crops)
    phase_values = np.concatenate([crop["phase"].ravel() for crop in crops.values()])
    vmin_pc = float(np.percentile(phase_values, 1))
    vmax_pc = float(np.percentile(phase_values, 99))

    records: list[dict[str, object]] = []
    for row in selection.itertuples():
        crop = crops[(str(row.image_key), int(row.cell_label))]
        color = BRIGHT_COLOR if row.brightness_group == "bright" else DIM_COLOR
        nobg = render_background_removed(
            crop["ido"], crop["mask"], crop["nucleus"], vmax=vmax_nobg
        )
        fig = _new_figure(8.2, 2.15)
        axes = fig.subplots(1, 5)
        panels = (
            (crop["phase"], "phase 原圖", "gray", vmin_pc, vmax_pc, crop["mask"]),
            (crop["phase_segmented"], "phase 分割", "gray", vmin_pc, vmax_pc, None),
            (crop["ido"], "IDO 原圖", IDO_CMAP, vmin_ido, vmax_ido, crop["mask"]),
            (crop["ido_segmented"], "IDO 分割（黑底）", IDO_CMAP, vmin_ido, vmax_ido, None),
            (nobg, "IDO 去背（白底）", IDO_CMAP, 0.0, 1.0, None),
        )
        for ax, (image, label, cmap, low, high, outline) in zip(axes, panels):
            _draw_tile(ax, image, cmap=cmap, vmin=low, vmax=high, outline=outline, outline_color=color)
            ax.set_title(label, fontsize=8, color=TEXT_SECONDARY, pad=3)
        group_zh = "亮" if row.brightness_group == "bright" else "暗"
        fig.suptitle(
            f"{group_zh}　{row.image_key}　cell {int(row.cell_label)}　"
            f"{row.condition}　IDO_score_ff = {row.IDO_score_ff:.2f}",
            fontsize=9.5,
            color=color,
            y=1.06,
        )
        filename = f"{row.brightness_group}_{row.image_key}_cell{int(row.cell_label):04d}.png"
        _save(fig, out_dir / filename)
        records.append({"image_key": row.image_key, "cell_label": int(row.cell_label), "crop_file": filename})
    return records


def build_split(
    cells: pd.DataFrame,
    data_root: Path,
    out_root: Path,
    config: GalleryConfig,
    *,
    split_name: str,
    strata: list[str] | None,
    title: str,
    subtitle: str,
    write_crops: bool,
) -> dict[str, object]:
    """執行一種切法：標記亮暗、比較特徵、抽樣、裁圖、出圖。"""
    labelled = assign_brightness_groups(cells, decile_pct=config.decile_pct, strata=strata)
    features = feature_columns(labelled)
    contrast = contrast_features(labelled, features, decile_pct=config.decile_pct)

    selected = pd.concat(
        [
            sample_cells(labelled, "bright", config.tiles_per_group, seed=config.seed, strata=strata),
            sample_cells(labelled, "dim", config.tiles_per_group, seed=config.seed + 1, strata=strata),
        ]
    )
    missing_paths = {"pc_path", "ido_path"} - set(selected.columns)
    if missing_paths or selected["pc_path"].isna().any():
        raise GalleryError("選中的細胞缺少影像路徑，請確認 manifest 已併入 cell 表")

    crops = collect_crops(data_root, selected, config, include_native=write_crops)
    figures_dir = out_root / "figures"
    paths = {
        "ido": plot_gallery(
            selected,
            crops,
            figures_dir / f"fig_gallery_{split_name}_ido.png",
            channel="ido",
            title=title,
            subtitle=f"{subtitle}　|　IDO 通道，所有格子共用同一顯示範圍",
        ),
        "phase": plot_gallery(
            selected,
            crops,
            figures_dir / f"fig_gallery_{split_name}_phase.png",
            channel="phase",
            title=f"{title}　— 相位差通道",
            subtitle=f"{subtitle}　|　相位差通道，綠／紫線為細胞分割輪廓",
        ),
        "ido_segmented": plot_gallery(
            selected,
            crops,
            figures_dir / f"fig_gallery_{split_name}_ido_segmented.png",
            channel="ido_segmented",
            title=title,
            subtitle=f"{subtitle}　|　只保留分割輪廓內的 IDO 像素",
        ),
        "ido_nobg": plot_gallery(
            selected,
            crops,
            figures_dir / f"fig_gallery_{split_name}_ido_nobg.png",
            channel="ido_nobg",
            title=f"{title}　— 去背",
            subtitle=(
                f"{subtitle}　|　背景移除：白底、綠色為 IDO 強度、"
                "藍色為細胞核、紅線為分割輪廓"
            ),
        ),
    }

    crop_records: list[dict[str, object]] = []
    nobg_records: list[dict[str, object]] = []
    if write_crops:
        crop_records = write_cell_crops(selected, crops, out_root / f"cell_crops_{split_name}")
        nobg_records = write_nobg_cells(selected, crops, out_root / f"cell_nobg_{split_name}")

    export_columns = [
        "image_key",
        "cell_label",
        "b_id",
        "passage",
        "group_id",
        "condition",
        "condition_index",
        "ifn_dose",
        "tnf_dose",
        "IDO_score_ff",
        "ido_percentile",
        "brightness_group",
        "pc_path",
        "ido_path",
    ]
    export = selected[export_columns].copy()
    for records in (crop_records, nobg_records):
        if records:
            export = export.merge(
                pd.DataFrame(records), on=["image_key", "cell_label"], how="left"
            )

    counts = labelled["brightness_group"].value_counts()
    return {
        "labelled_counts": {
            "bright": int(counts.get("bright", 0)),
            "dim": int(counts.get("dim", 0)),
            "mid": int(counts.get("mid", 0)),
        },
        "contrast": contrast,
        "selection": export,
        "figures": paths,
        "n_crop_files": len(crop_records),
        "n_nobg_files": len(nobg_records),
    }
