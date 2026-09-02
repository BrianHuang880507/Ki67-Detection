"""Exp8 圖表：大標簡短、沒有副標題、不使用自訂縮寫。

與 Exp6 的差別：

- 標題只留一句話，說明性文字全部移到 REPORT.md 與 CSV。
- 不放顯著性星號。17 個形狀特徵幾乎全部通過 FDR，星號只是雜訊。
- 不寫 `ρ`、`Δ`、`FOV` 這類符號，座標軸一律用完整中文。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from ..exp6.figures import (
    GRID,
    NEUTRAL,
    NEGATIVE,
    POSITIVE,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _new_figure,
    _save,
    _style_axes,
)
from ..exp6.gallery import BRIGHT_COLOR, DIM_COLOR, render_background_removed
from .brightness import BRIGHT_LABEL, DARK_LABEL


#: 三個 donor 用 categorical 前三個色位，三色在全配對下仍可分辨。
DONOR_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
CURVE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

#: 帶正負號的變化量用發散配色：藍↔紅，中性灰為零。
DIVERGING = LinearSegmentedColormap.from_list(
    "delta", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8f1d1c"]
)

TITLE_SIZE = 15
PANEL_TITLE_SIZE = 11.5


def _title(fig_or_ax, text: str, *, axes: bool = False) -> None:
    """統一的大標樣式：左對齊、粗體、只有一行。"""
    if axes:
        fig_or_ax.set_title(
            text, fontsize=13, color=TEXT_PRIMARY, loc="left", pad=12, fontweight="bold"
        )
    else:
        fig_or_ax.suptitle(
            text,
            fontsize=TITLE_SIZE,
            color=TEXT_PRIMARY,
            fontweight="bold",
            x=0.03,
            ha="left",
            y=1.02,
        )


def _donor_color(donor: str, donors: list[str]) -> str:
    """依 donor 在排序後清單中的位置取固定顏色，篩選不會換色。"""
    return DONOR_COLORS[donors.index(donor) % len(DONOR_COLORS)]


def plot_ido_dose_response(fov: pd.DataFrame, out_path: Path) -> Path:
    """fig01：IDO 對刺激濃度的反應。"""
    panels = [
        ("IFN-γ（TNF-α = 0）", fov["tnf_dose"] == 0, "ifn_dose", "IFN-γ 濃度 (ng/mL)"),
        ("TNF-α（IFN-γ = 0）", fov["ifn_dose"] == 0, "tnf_dose", "TNF-α 濃度 (ng/mL)"),
        ("TNF-α（IFN-γ = 25）", fov["ifn_dose"] == 25, "tnf_dose", "TNF-α 濃度 (ng/mL)"),
    ]
    fig = _new_figure(13.0, 4.6)
    axes = fig.subplots(1, 3, sharey=True)
    upper = float(np.percentile(fov["IDO_score_ff"], 99)) * 1.18
    lower = float(np.percentile(fov["IDO_score_ff"], 1)) - 0.5

    for index, (panel_title, mask, dose_column, xlabel) in enumerate(panels):
        ax = axes[index]
        subset = fov[mask]
        doses = sorted(subset[dose_column].unique())
        groups = [
            subset.loc[subset[dose_column] == dose, "IDO_score_ff"].to_numpy() for dose in doses
        ]
        color = CURVE_COLORS[index]
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
        rng = np.random.default_rng(0)
        for offset, values in zip(positions, groups):
            ax.scatter(
                offset + rng.normal(0.0, 0.055, size=values.size),
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
        ax.set_title(panel_title, fontsize=PANEL_TITLE_SIZE, color=TEXT_PRIMARY, loc="left", pad=8)
        ax.text(
            0.97,
            0.95,
            f"中位數變化 {medians[-1] - medians[0]:+.2f}",
            transform=ax.transAxes,
            fontsize=10.5,
            color=color,
            ha="right",
            va="top",
            fontweight="bold",
        )
        _style_axes(ax, xgrid=False, ygrid=True)
        if index == 0:
            ax.set_ylabel("影像 IDO 強度（灰階）", fontsize=9.5)

    _title(fig, "IDO 對刺激濃度的反應")
    fig.tight_layout(w_pad=2.0)
    return _save(fig, out_path)


def plot_positive_fraction(table: pd.DataFrame, out_path: Path) -> Path:
    """fig02：IDO 陽性細胞比例。"""
    ordered = table.sort_values("positive_fraction")
    labels = ordered["condition"].tolist()
    values = (ordered["positive_fraction"] * 100).to_numpy()
    low = (ordered["positive_fraction_min"] * 100).to_numpy()
    high = (ordered["positive_fraction_max"] * 100).to_numpy()

    fig = _new_figure(9.2, 5.2)
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=POSITIVE, zorder=3)
    ax.hlines(y, low, high, color=TEXT_SECONDARY, linewidth=1.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("超過未刺激對照上限的細胞比例 (%)", fontsize=9.5)
    # 數字放在誤差線右側，避免被線壓過去。
    for position, value, end in zip(y, values, high):
        ax.text(
            max(value, end) + 2.0,
            position,
            f"{value:.0f}%",
            va="center",
            fontsize=9.5,
            color=TEXT_SECONDARY,
        )
    _title(ax, "IDO 陽性細胞比例", axes=True)
    _style_axes(ax)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_shape_dose_correlation(table: pd.DataFrame, out_path: Path, title: str) -> Path:
    """fig03／fig04：形狀特徵與濃度的關聯性，含細胞密度校正後的對照。"""
    ordered = table.sort_values("rank")
    labels = ordered["feature_label"].tolist()
    raw = ordered["spearman_rho"].to_numpy()
    adjusted = ordered["rho_density_adjusted"].to_numpy()

    y = np.arange(len(labels))[::-1]
    height = 0.36
    fig = _new_figure(9.8, 7.6)
    ax = fig.add_subplot(111)
    # 兩根長條是兩種算法，不是正負號，因此用 categorical 兩色而不是發散配色。
    ax.barh(y + height / 2, raw, height=height, color=CURVE_COLORS[0], zorder=3, label="原始")
    ax.barh(
        y - height / 2,
        adjusted,
        height=height,
        color=CURVE_COLORS[1],
        zorder=3,
        label="扣除細胞密度後",
    )
    ax.axvline(0.0, color=NEUTRAL, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
    limit = float(np.nanmax(np.abs(np.concatenate([raw, adjusted])))) * 1.28
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Spearman 相關係數", fontsize=9.5)
    for position, value in zip(y + height / 2, raw):
        if not np.isfinite(value):
            continue
        offset = 0.015 * limit if value >= 0 else -0.015 * limit
        ax.text(
            value + offset,
            position,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )
    _title(ax, title, axes=True)
    _style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_shape_dose_response(
    fov: pd.DataFrame, features: list[str], labels: list[str], out_path: Path
) -> Path:
    """fig05：細胞形狀隨刺激濃度的變化。"""
    curves = [
        ("IFN-γ（TNF-α = 0）", fov["tnf_dose"] == 0, "ifn_dose", "IFN-γ 濃度 (ng/mL)"),
        ("TNF-α（IFN-γ = 0）", fov["ifn_dose"] == 0, "tnf_dose", "TNF-α 濃度 (ng/mL)"),
        ("TNF-α（IFN-γ = 25）", fov["ifn_dose"] == 25, "tnf_dose", "TNF-α 濃度 (ng/mL)"),
    ]
    rows = len(features)
    fig = _new_figure(13.0, 3.5 * rows)
    axes = fig.subplots(rows, len(curves), squeeze=False)

    for row_index, (feature, label) in enumerate(zip(features, labels)):
        row_values = fov[feature].dropna()
        lower = float(np.percentile(row_values, 1))
        upper = float(np.percentile(row_values, 99))
        margin = (upper - lower) * 0.12
        for column_index, (panel_title, mask, dose_column, xlabel) in enumerate(curves):
            ax = axes[row_index][column_index]
            subset = fov[mask]
            doses = sorted(subset[dose_column].unique())
            groups = [
                subset.loc[subset[dose_column] == dose, feature].dropna().to_numpy()
                for dose in doses
            ]
            color = CURVE_COLORS[column_index]
            positions = np.arange(len(doses))
            parts = ax.boxplot(
                groups,
                positions=positions,
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": TEXT_PRIMARY, "linewidth": 1.6},
                whiskerprops={"color": NEUTRAL, "linewidth": 1.1},
                capprops={"color": NEUTRAL, "linewidth": 1.1},
            )
            for patch in parts["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.26)
                patch.set_edgecolor(color)
                patch.set_linewidth(1.5)
            medians = [float(np.median(values)) for values in groups]
            ax.plot(positions, medians, color=color, linewidth=2.0, zorder=4)
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{int(dose)}" for dose in doses])
            ax.set_ylim(lower - margin, upper + margin)
            ax.text(
                0.97,
                0.95,
                f"中位數變化 {medians[-1] - medians[0]:+.3g}",
                transform=ax.transAxes,
                fontsize=9.5,
                color=color,
                ha="right",
                va="top",
                fontweight="bold",
            )
            if row_index == 0:
                ax.set_title(
                    panel_title, fontsize=PANEL_TITLE_SIZE, color=TEXT_PRIMARY, loc="left", pad=8
                )
            if row_index == rows - 1:
                ax.set_xlabel(xlabel, fontsize=9.5)
            if column_index == 0:
                ax.set_ylabel(f"{label}（像素）" if "率" not in label and "度" not in label else label,
                              fontsize=9.5)
            _style_axes(ax, xgrid=False, ygrid=True)

    _title(fig, "細胞形狀隨刺激濃度的變化")
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    return _save(fig, out_path)


def _tile_axes(ax: plt.Axes, image: np.ndarray) -> None:
    """畫一格去背細胞，不留座標軸。"""
    ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_cells_by_dose(
    rows: list[tuple[str, list[np.ndarray]]], out_path: Path, title: str, row_color: str | None = None
) -> Path:
    """fig06：每一列一個濃度，列出該濃度的代表性去背細胞。"""
    columns = max(len(images) for _, images in rows)
    fig = _new_figure(columns * 1.5, len(rows) * 1.62 + 0.7)
    grid = fig.add_gridspec(
        len(rows), columns, hspace=0.12, wspace=0.06, top=0.9, bottom=0.02, left=0.06, right=0.99
    )
    for row_index, (row_label, images) in enumerate(rows):
        for column_index in range(columns):
            ax = fig.add_subplot(grid[row_index, column_index])
            if column_index < len(images):
                _tile_axes(ax, images[column_index])
            else:
                ax.set_axis_off()
            if column_index == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=11,
                    color=row_color or TEXT_PRIMARY,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=14,
                )
    _title(fig, title)
    return _save(fig, out_path)


def plot_bright_dim_shape(table: pd.DataFrame, out_path: Path) -> Path:
    """fig07：IDO 亮細胞與暗細胞的形狀差異。"""
    ordered = table.sort_values("rank")
    labels = ordered["feature_label"].tolist()
    values = ordered["effect_median"].to_numpy()
    low = ordered["effect_q1"].to_numpy()
    high = ordered["effect_q3"].to_numpy()

    fig = _new_figure(9.6, 7.4)
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))[::-1]
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in values]
    ax.barh(y, values, height=0.6, color=colors, zorder=3)
    ax.hlines(y, low, high, color=TEXT_SECONDARY, linewidth=1.3, zorder=4)
    ax.axvline(0.0, color=NEUTRAL, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
    limit = float(np.nanmax(np.abs(np.concatenate([low, high])))) * 1.35
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("亮細胞與暗細胞的效果量（正值代表亮細胞較大）", fontsize=9.5)
    # 數字放在四分位範圍的外側，避免被範圍線壓到。
    for position, value, low_end, high_end in zip(y, values, low, high):
        outer = high_end if value >= 0 else low_end
        offset = 0.02 * limit if value >= 0 else -0.02 * limit
        ax.text(
            outer + offset,
            position,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )
    _title(ax, "IDO 亮細胞與暗細胞的形狀差異", axes=True)
    _style_axes(ax)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_bright_dim_distribution(
    labelled: pd.DataFrame, features: list[str], labels: list[str], out_path: Path
) -> Path:
    """fig08：IDO 亮細胞與暗細胞的形狀分布。"""
    columns = 2
    rows = int(np.ceil(len(features) / columns))
    fig = _new_figure(11.0, 3.3 * rows)
    axes = fig.subplots(rows, columns, squeeze=False)

    for index, (feature, label) in enumerate(zip(features, labels)):
        ax = axes[index // columns][index % columns]
        groups = [
            (BRIGHT_LABEL, labelled.loc[labelled["ido_class"] == BRIGHT_LABEL, feature].dropna(), BRIGHT_COLOR),
            (DARK_LABEL, labelled.loc[labelled["ido_class"] == DARK_LABEL, feature].dropna(), DIM_COLOR),
        ]
        pooled = np.concatenate([values.to_numpy() for _, values, _ in groups])
        lower, upper = np.percentile(pooled, [1, 99])
        edges = np.linspace(lower, upper, 44)
        for name, values, color in groups:
            ax.hist(
                values,
                bins=edges,
                density=True,
                histtype="stepfilled",
                color=color,
                alpha=0.42,
                zorder=3,
            )
            ax.hist(values, bins=edges, density=True, histtype="step", color=color, linewidth=1.8, zorder=4)
            ax.axvline(float(np.median(values)), color=color, linewidth=1.6, linestyle="--", zorder=5)
        ax.set_xlim(lower, upper)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=9.5)
        _style_axes(ax, xgrid=False)
        if index == 0:
            handles = [
                plt.Line2D([], [], color=BRIGHT_COLOR, linewidth=6, label=f"{BRIGHT_LABEL}細胞"),
                plt.Line2D([], [], color=DIM_COLOR, linewidth=6, label=f"{DARK_LABEL}細胞"),
            ]
            legend = ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="upper right")
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)

    for index in range(len(features), rows * columns):
        axes[index // columns][index % columns].set_axis_off()

    _title(fig, "IDO 亮細胞與暗細胞的形狀分布")
    fig.tight_layout(h_pad=2.0, w_pad=2.4)
    return _save(fig, out_path)


def plot_paired_cells(
    pairs: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path, title: str
) -> Path:
    """fig09：同一張影像中的亮細胞與暗細胞，上下配對。"""
    columns = len(pairs)
    fig = _new_figure(columns * 1.55, 4.1)
    grid = fig.add_gridspec(
        2, columns, hspace=0.1, wspace=0.06, top=0.84, bottom=0.03, left=0.07, right=0.99
    )
    for column_index, (image_key, bright_image, dim_image) in enumerate(pairs):
        for row_index, (image, color) in enumerate(
            ((bright_image, BRIGHT_COLOR), (dim_image, DIM_COLOR))
        ):
            ax = fig.add_subplot(grid[row_index, column_index])
            _tile_axes(ax, image)
            if column_index == 0:
                ax.set_ylabel(
                    f"{BRIGHT_LABEL}細胞" if row_index == 0 else f"{DARK_LABEL}細胞",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=16,
                )
            if row_index == 0:
                ax.set_title(image_key, fontsize=7, color=TEXT_SECONDARY, pad=3)
    _title(fig, title)
    return _save(fig, out_path)


def plot_donor_delta_heatmap(matrix: pd.DataFrame, out_path: Path) -> Path:
    """fig10：各 donor 的形狀變化量。"""
    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values)))
    limit = max(limit, 0.1)
    fig = _new_figure(1.0 + matrix.shape[1] * 0.86, 1.6 + matrix.shape[0] * 0.36)
    ax = fig.add_subplot(111)
    mesh = ax.imshow(
        values,
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9, color=TEXT_PRIMARY)

    donors = [str(donor) for donor, _ in matrix.columns]
    doses = [f"{int(dose)}" for _, dose in matrix.columns]
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(doses, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("IFN-γ 濃度 (ng/mL)", fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    boundaries = [index for index in range(1, len(donors)) if donors[index] != donors[index - 1]]
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color=SURFACE, linewidth=3.0)
    seen: dict[str, list[int]] = {}
    for index, donor in enumerate(donors):
        seen.setdefault(donor, []).append(index)
    for donor, indices in seen.items():
        ax.text(
            float(np.mean(indices)),
            -0.9,
            donor,
            ha="center",
            va="bottom",
            fontsize=11,
            color=TEXT_PRIMARY,
            fontweight="bold",
        )

    bar = fig.colorbar(mesh, ax=ax, fraction=0.026, pad=0.02)
    bar.set_label("形狀變化量（除以對照組四分位距）", fontsize=9, color=TEXT_SECONDARY)
    bar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    bar.outline.set_visible(False)

    _title(fig, "各 donor 的形狀變化量")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_donor_dose_slopes(
    delta: pd.DataFrame, features: list[str], labels: list[str], out_path: Path
) -> Path:
    """fig11：各 donor 的形狀變化量與濃度的關係。"""
    subset = delta[delta["tnf_dose"] == 0]
    donors = sorted(subset["b_id"].unique())
    columns = 2
    rows = int(np.ceil(len(features) / columns))
    fig = _new_figure(11.0, 3.4 * rows)
    axes = fig.subplots(rows, columns, squeeze=False)

    for index, (feature, label) in enumerate(zip(features, labels)):
        ax = axes[index // columns][index % columns]
        block = subset[subset["feature"] == feature]
        for donor in donors:
            donor_block = block[block["b_id"] == donor]
            color = _donor_color(donor, donors)
            medians = donor_block.groupby("ifn_dose")["delta_standardized"].median()
            doses = [0.0, *medians.index.tolist()]
            values = [0.0, *medians.to_numpy()]
            ax.plot(doses, values, color=color, linewidth=2.0, marker="o", markersize=6, zorder=4, label=donor)
            ax.scatter(
                donor_block["ifn_dose"],
                donor_block["delta_standardized"],
                s=22,
                color=color,
                alpha=0.5,
                linewidths=0,
                zorder=3,
            )
        ax.axhline(0.0, color=NEUTRAL, linewidth=1.2, zorder=2)
        ax.set_xticks([0, 25, 50, 100])
        ax.set_xlabel("IFN-γ 濃度 (ng/mL)", fontsize=9.5)
        ax.set_ylabel("形狀變化量", fontsize=9.5)
        ax.set_title(label, fontsize=PANEL_TITLE_SIZE, color=TEXT_PRIMARY, loc="left", pad=8)
        _style_axes(ax, xgrid=False, ygrid=True)
        if index == 0:
            legend = ax.legend(frameon=False, fontsize=9.5, loc="best")
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)

    for index in range(len(features), rows * columns):
        axes[index // columns][index % columns].set_axis_off()

    _title(fig, "各 donor 的形狀變化量與濃度的關係")
    fig.tight_layout(h_pad=2.4, w_pad=2.4)
    return _save(fig, out_path)


def plot_donor_magnitude(table: pd.DataFrame, out_path: Path) -> Path:
    """fig12：各 donor 的整體形狀變化幅度。"""
    donors = sorted(table["b_id"].unique())
    fig = _new_figure(7.6, 5.0)
    ax = fig.add_subplot(111)
    positions = np.arange(len(donors))
    heights = [float(table.loc[table["b_id"] == donor, "median"].iloc[0]) for donor in donors]
    for position, donor, height in zip(positions, donors, heights):
        color = _donor_color(donor, donors)
        ax.bar(position, height, width=0.55, color=color, alpha=0.42, zorder=3)
        block = table[table["b_id"] == donor]
        ax.scatter(
            np.full(len(block), position),
            block["magnitude"],
            s=48,
            color=color,
            zorder=5,
            edgecolors=SURFACE,
            linewidths=1.2,
        )
        # 標出每個 passage，否則看不出離群點是哪一次傳代。
        for row in block.itertuples():
            ax.text(
                position + 0.16,
                float(row.magnitude),
                f"P{int(row.passage)}",
                ha="left",
                va="center",
                fontsize=9,
                color=TEXT_SECONDARY,
            )
        ax.text(
            position,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=TEXT_SECONDARY,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(donors, fontsize=11, color=TEXT_PRIMARY)
    ax.set_ylabel("形狀變化幅度（除以對照組四分位距）", fontsize=9.5)
    # 上限要蓋住所有 passage 的點；把離群點切掉會讓 donor 差異看起來比實際明確。
    ax.set_ylim(0, float(table["magnitude"].max()) * 1.12)
    _title(ax, "各 donor 的整體形狀變化幅度", axes=True)
    _style_axes(ax, xgrid=False, ygrid=True)
    fig.tight_layout()
    return _save(fig, out_path)
