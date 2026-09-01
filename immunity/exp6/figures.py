"""Exp6 圖表：劑量關聯長條圖與 IDO 劑量曲線。

配色沿用 data-viz 參考調色盤：帶正負號的 Spearman rho 屬於 polarity，使用
藍（正）／紅（負）發散配對加中性灰零線；三條劑量曲線屬於 identity，使用
categorical 前三個色位。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"
NEUTRAL = "#b9b8b2"

POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
ACCENT = "#4a3aa7"

CJK_CANDIDATES = ("Microsoft JhengHei", "Microsoft YaHei", "MingLiU", "DFKai-SB")


def configure_fonts() -> str:
    """挑一個系統上實際存在的中日韓字型，避免圖上出現方框。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in CJK_CANDIDATES if name in available), "DejaVu Sans")
    plt.rcParams["font.family"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def _style_axes(ax: plt.Axes, *, xgrid: bool = True, ygrid: bool = False) -> None:
    """統一的低調座標軸樣式：留下資料，收掉裝飾。"""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3, width=1.0)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)


def _new_figure(width: float, height: float):
    """建立統一底色的 figure。"""
    fig = plt.figure(figsize=(width, height), dpi=200, facecolor=SURFACE)
    return fig


def _save(fig: plt.Figure, path: Path) -> Path:
    """存檔並關閉 figure。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _significance_mark(q_value: float) -> str:
    """把 BH 校正後的 q 值轉成星號標記。"""
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "n.s."


def _draw_rho_bars(
    ax: plt.Axes,
    labels: list[str],
    rhos: list[float],
    marks: list[str],
    *,
    title: str,
    subtitle: str,
) -> None:
    """畫一張帶正負號的水平 rho 長條圖，最強的排在最上面。"""
    y = np.arange(len(labels))[::-1]
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in rhos]
    ax.barh(y, rhos, height=0.62, color=colors, zorder=3)
    ax.axvline(0.0, color=NEUTRAL, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
    limit = max(0.1, max(abs(value) for value in rhos)) * 1.32
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Spearman ρ（影像層級）", fontsize=9.5)
    for position, value, mark in zip(y, rhos, marks):
        offset = 0.018 * limit
        align = "left" if value >= 0 else "right"
        text = f"{value:+.2f} {mark}".strip()
        ax.text(
            value + (offset if value >= 0 else -offset),
            position,
            text,
            va="center",
            ha=align,
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    ax.set_title(title, fontsize=13, color=TEXT_PRIMARY, loc="left", pad=27, fontweight="bold")
    ax.text(
        0.0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        fontsize=9.5,
        color=TEXT_SECONDARY,
        va="bottom",
    )
    _style_axes(ax)


def plot_top_features(
    table: pd.DataFrame,
    axis_name: str,
    axis_title: str,
    out_path: Path,
    *,
    top_n: int = 10,
) -> Path:
    """單一劑量軸的前 N 名外觀特徵長條圖。"""
    subset = table[table["dose_axis"] == axis_name].nsmallest(top_n, "rank").sort_values("rank")
    labels = subset["feature_label"].tolist()
    rhos = subset["spearman_rho"].tolist()
    marks = [_significance_mark(value) for value in subset["q_value_bh"]]
    n_images = int(subset["n_images"].iloc[0])

    fig = _new_figure(9.0, 5.6)
    ax = fig.add_subplot(111)
    _draw_rho_bars(
        ax,
        labels,
        rhos,
        marks,
        title=f"外觀特徵與 {axis_title} 的關聯性　前 {top_n} 名",
        subtitle=(
            f"n = {n_images} 張影像　|　藍＝正相關、紅＝負相關　|　"
            "* q<0.05、** q<0.01、*** q<0.001（BH FDR）"
        ),
    )
    return _save(fig, out_path)


def plot_top_features_combined(
    table: pd.DataFrame, out_path: Path, *, top_n: int = 10
) -> Path:
    """IFN 與 TNF 並排的前 N 名長條圖，給簡報用。"""
    fig = _new_figure(15.0, 5.8)
    axes = fig.subplots(1, 2)
    for ax, (axis_name, axis_title) in zip(
        axes, (("IFN", "IFN-γ 劑量（TNF-α = 0）"), ("TNF", "TNF-α 劑量（固定 IFN-γ）"))
    ):
        subset = table[table["dose_axis"] == axis_name].nsmallest(top_n, "rank").sort_values("rank")
        _draw_rho_bars(
            ax,
            subset["feature_label"].tolist(),
            subset["spearman_rho"].tolist(),
            [_significance_mark(value) for value in subset["q_value_bh"]],
            title=f"vs. {axis_title}",
            subtitle=f"n = {int(subset['n_images'].iloc[0])} 張影像",
        )
    fig.suptitle(
        "細胞外觀特徵與細胞激素劑量的關聯性　前 10 名",
        fontsize=15,
        color=TEXT_PRIMARY,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=1.08,
    )
    fig.tight_layout(w_pad=6.0)
    return _save(fig, out_path)


def plot_features_vs_ido(table: pd.DataFrame, out_path: Path, *, top_n: int = 10) -> Path:
    """外觀特徵與 IDO 的關聯性：全部影像 vs. 同一刺激條件內。

    兩條序列刻意並列：若某特徵只在左邊高、右邊掉到 0，代表它跟的是劑量，
    不是 IDO 本身。
    """
    subset = table.nsmallest(top_n, "rank").sort_values("rank")
    labels = subset["feature_label"].tolist()
    overall = subset["rho_image_level"].to_numpy()
    within = subset["rho_within_condition"].to_numpy()

    y = np.arange(len(labels))[::-1]
    height = 0.34
    fig = _new_figure(9.6, 6.0)
    ax = fig.add_subplot(111)
    ax.barh(y + height / 2, overall, height=height, color=CATEGORICAL[0], zorder=3, label="全部影像")
    ax.barh(
        y - height / 2,
        within,
        height=height,
        color=CATEGORICAL[1],
        zorder=3,
        label="同一刺激條件內",
    )
    ax.axvline(0.0, color=NEUTRAL, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
    limit = max(0.1, float(np.nanmax(np.abs(np.concatenate([overall, within]))))) * 1.35
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Spearman ρ（影像層級，對 IDO_score_ff）", fontsize=9.5)
    for position, value in zip(y + height / 2, overall):
        offset = 0.018 * limit if value >= 0 else -0.018 * limit
        ax.text(
            value + offset,
            position,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )
    ax.set_title(
        "外觀特徵與 IDO 的關聯性　前 10 名",
        fontsize=13,
        color=TEXT_PRIMARY,
        loc="left",
        pad=27,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.012,
        f"n = {int(subset['n_images'].iloc[0])} 張影像　|　"
        "右側掉到接近 0 表示該特徵跟的是劑量，不是 IDO 本身",
        transform=ax.transAxes,
        fontsize=9.5,
        color=TEXT_SECONDARY,
        va="bottom",
    )
    _style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    return _save(fig, out_path)


def plot_ido_dose_response(
    fov: pd.DataFrame, ido_correlations: pd.DataFrame, out_path: Path
) -> Path:
    """三條 IDO 劑量曲線的小倍數圖，每格標註該軸的 Spearman rho。"""
    panels = [
        ("IFN-γ（TNF-α = 0）", fov["tnf_dose"] == 0, "ifn_dose", "IFN-γ (ng/mL)", ("IFN", "全部")),
        ("TNF-α（IFN-γ = 0）", fov["ifn_dose"] == 0, "tnf_dose", "TNF-α (ng/mL)", ("TNF", "IFN-γ = 0")),
        ("TNF-α（IFN-γ = 25）", fov["ifn_dose"] == 25, "tnf_dose", "TNF-α (ng/mL)", ("TNF", "IFN-γ = 25")),
    ]
    fig = _new_figure(13.0, 4.6)
    axes = fig.subplots(1, 3, sharey=True)

    ff_rows = ido_correlations[ido_correlations["target"] == "IDO_score_ff"]
    rho_by_axis = {
        (row.dose_axis, row.block): row.spearman_rho for row in ff_rows.itertuples()
    }

    upper = float(np.percentile(fov["IDO_score_ff"], 99)) * 1.18
    lower = float(np.percentile(fov["IDO_score_ff"], 1)) - 0.5

    for index, (title, mask, dose_column, xlabel, axis_key) in enumerate(panels):
        ax = axes[index]
        subset = fov[mask]
        doses = sorted(subset[dose_column].unique())
        groups = [subset.loc[subset[dose_column] == dose, "IDO_score_ff"].to_numpy() for dose in doses]
        color = CATEGORICAL[index]
        positions = np.arange(len(doses))
        parts = ax.boxplot(
            groups,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": TEXT_PRIMARY, "linewidth": 1.8},
            whiskerprops={"color": NEUTRAL, "linewidth": 1.2},
            capprops={"color": NEUTRAL, "linewidth": 1.2},
        )
        for patch in parts["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.6)
        for offset, values in zip(positions, groups):
            jitter = np.random.default_rng(0).normal(0.0, 0.055, size=values.size)
            ax.scatter(
                offset + jitter,
                values,
                s=7,
                color=color,
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )
        medians = [float(np.median(values)) for values in groups]
        ax.plot(positions, medians, color=color, linewidth=2.0, zorder=4)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{int(dose)}" for dose in doses])
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylim(lower, upper)
        ax.set_title(title, fontsize=11.5, color=TEXT_PRIMARY, loc="left", pad=8)
        counts = "　".join(f"n={int((subset[dose_column] == dose).sum())}" for dose in doses)
        ax.text(
            0.0,
            -0.235,
            counts,
            transform=ax.transAxes,
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )
        rho = rho_by_axis.get(axis_key, float("nan"))
        delta = float(np.median(groups[-1]) - np.median(groups[0]))
        ax.text(
            0.97,
            0.955,
            f"ρ = {rho:+.2f}\nΔ中位數 = {delta:+.2f}",
            transform=ax.transAxes,
            fontsize=11,
            color=color,
            ha="right",
            va="top",
            fontweight="bold",
        )
        _style_axes(ax, xgrid=False, ygrid=True)
        if index == 0:
            ax.set_ylabel("影像 IDO_score_ff（灰階）", fontsize=9.5)

    fig.suptitle(
        "IDO 螢光對 IFN-γ／TNF-α 劑量的反應（flat-field 校正後）",
        fontsize=14,
        color=TEXT_PRIMARY,
        fontweight="bold",
        x=0.045,
        ha="left",
        y=1.11,
    )
    fig.text(
        0.045,
        1.048,
        "每點 = 一張影像的細胞中位數　|　ρ 為該軸的 Spearman 相關係數，Δ中位數為最高與最低劑量的灰階差",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        ha="left",
    )
    fig.tight_layout(w_pad=2.0)
    return _save(fig, out_path)


def plot_bright_dim_contrast(table: pd.DataFrame, out_path: Path, *, top_n: int = 10) -> Path:
    """IDO 亮／暗細胞的外觀特徵差異（rank-biserial 效果量）長條圖。"""
    subset = table.nsmallest(top_n, "rank").sort_values("rank")
    labels = subset["feature_label"].tolist()
    values = subset["rank_biserial"].tolist()
    marks = [_significance_mark(value) for value in subset["q_value_bh"]]

    fig = _new_figure(9.4, 5.8)
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))[::-1]
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in values]
    ax.barh(y, values, height=0.62, color=colors, zorder=3)
    ax.axvline(0.0, color=NEUTRAL, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
    limit = max(0.1, max(abs(value) for value in values)) * 1.34
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("rank-biserial 效果量（正＝亮細胞較大）", fontsize=9.5)
    for position, value, mark in zip(y, values, marks):
        offset = 0.018 * limit if value >= 0 else -0.018 * limit
        ax.text(
            value + offset,
            position,
            f"{value:+.2f} {mark}".strip(),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    n_bright = int(subset["n_bright"].iloc[0])
    n_dim = int(subset["n_dim"].iloc[0])
    ax.set_title(
        "IDO 亮細胞 vs. 暗細胞的外觀差異　前 10 名",
        fontsize=13,
        color=TEXT_PRIMARY,
        loc="left",
        pad=27,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.012,
        f"同一刺激條件內取上／下 {int(subset['decile_pct'].iloc[0])}%　|　"
        f"亮 n = {n_bright}、暗 n = {n_dim} 顆細胞",
        transform=ax.transAxes,
        fontsize=9.5,
        color=TEXT_SECONDARY,
        va="bottom",
    )
    _style_axes(ax)
    return _save(fig, out_path)
