"""細胞面積分析圖表產生工具。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _valid_area_values(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """取出可用的細胞核與細胞面積數值。

    Args:
        df: 包含 `Area_nuc` 與 `Area_cyto` 欄位的分析結果。

    Returns:
        `(nucleus_area, cell_area)`，兩者皆已過濾非有限值與非正值。

    Raises:
        KeyError: 缺少必要面積欄位時拋出。
    """
    required = {"Area_nuc", "Area_cyto"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing area columns: {', '.join(sorted(missing))}")

    nuc = pd.to_numeric(df["Area_nuc"], errors="coerce").to_numpy(dtype=float)
    cell = pd.to_numeric(df["Area_cyto"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(nuc) & np.isfinite(cell) & (nuc > 0) & (cell > 0)
    return nuc[valid], cell[valid]


def plot_global_area_analysis(
    all_final_csv: str | Path,
    outdir: Path,
    thres: float = 6.0,
    width_um_per_px: float = 1.5896,
    height_um_per_px: float = 1.5876,
) -> tuple[Path, Path]:
    """輸出全資料夾的細胞核/細胞面積散點圖與細胞面積長條圖。

    Args:
        all_final_csv: 合併後的分析 CSV，需包含 `Area_nuc` 與 `Area_cyto`。
        outdir: 圖檔輸出資料夾。
        thres: 長條圖的 log3 細胞面積閾值。
        width_um_per_px: X 方向 pixel scale。
        height_um_per_px: Y 方向 pixel scale。

    Returns:
        `(scatter_path, histogram_path)`。
    """
    all_final_csv = Path(all_final_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(all_final_csv)
    nuc_area, cell_area = _valid_area_values(df)
    pixel_area = float(width_um_per_px) * float(height_um_per_px)
    nuc_area = nuc_area * pixel_area
    cell_area = cell_area * pixel_area

    scatter_path = outdir / "all_cell_nucleus_area.png"
    hist_path = outdir / "all_log_cell_area_distribution.png"

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(cell_area, nuc_area, c="green", s=5, alpha=0.5)
    ax.set_xlabel("Cell Area")
    ax.set_ylabel("Nucleus Area")
    ax.set_title("Cell vs Nucleus Area")
    fig.savefig(
        scatter_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)

    log_cell_area = np.log(cell_area) / np.log(3)
    log_cell_area = log_cell_area[np.isfinite(log_cell_area)]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("white")
    if log_cell_area.size > 0:
        bin_width = 0.2
        if np.isclose(log_cell_area.min(), log_cell_area.max()):
            bins = np.array(
                [
                    log_cell_area.min() - bin_width,
                    log_cell_area.max() + bin_width,
                ]
            )
        else:
            bins = np.arange(
                log_cell_area.min(),
                log_cell_area.max() + bin_width,
                bin_width,
            )
        ax.hist(
            log_cell_area,
            bins=bins,
            color="blue",
            edgecolor="black",
            weights=np.ones_like(log_cell_area) / len(log_cell_area) * 100,
        )
        ax.axvline(thres, color="red", linestyle="dashed", linewidth=2)
        percent_above_threshold = float(
            np.sum(log_cell_area > thres) / len(log_cell_area) * 100
        )
        ax.text(
            thres + 0.1,
            ax.get_ylim()[1] * 0.9,
            f"{percent_above_threshold:.2f}%",
            color="red",
            fontsize=12,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No valid area data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.set_xlabel("Cell Area (log3)")
    ax.set_ylabel("Cell Percentage (%)")
    ax.set_title("Cell Area Distribution (log3)")
    fig.savefig(
        hist_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)

    return scatter_path, hist_path
