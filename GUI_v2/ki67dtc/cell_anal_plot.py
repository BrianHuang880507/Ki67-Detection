import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd
from pathlib import Path


def plot_area_analysis(final_csv, outdir: Path, prefix: str, thres: float = 6.0,
                       width_um_per_px: float = 1.5896, height_um_per_px: float = 1.5876):
    """
    畫：
    1. 細胞質 vs 細胞核面積散布圖
    2. 細胞質面積對數分布圖（含閾值紅線）

    Parameters
    ----------
    width_um_per_px  : WIDTH  µm/pixel 倍率（預設 1.5896）
    height_um_per_px : HEIGHT µm/pixel 倍率（預設 1.5876）
    """
    # ===============================
    # 影像物理尺寸換算（由呼叫端傳入，不再硬編碼）
    # ===============================
    PIXEL_AREA = width_um_per_px * height_um_per_px  # µm²/pixel²

    df = pd.read_csv(final_csv)
    nuc_area_list = df["Area_nuc"].values * PIXEL_AREA   # 換算成 mm²
    cyto_area_list = df["Area_cyto"].values * PIXEL_AREA   # 換算成 mm²

    # --- 1. 散布圖 ---
    plt.figure(figsize=(12,5))
    plt.scatter(cyto_area_list, nuc_area_list, c="green", s=5)
    plt.xlabel("Cytoplasm Area")
    plt.ylabel("Nucleus Area")
    plt.title("Cytoplasm vs Nucleus Area")
    scatter_path = outdir / f"{prefix}_cell_nucleus_area.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    #print(f"[INFO] 已輸出散布圖 → {scatter_path}")

    # --- 2. 細胞質面積對數分布 ---
    #print(f"cyto_area_list:{cyto_area_list}")
    
    log_cell_area = np.log(cyto_area_list) / np.log(3)  # log3
    #print(f"log_cell_areasss:{log_cell_area}")
    log_cell_area = log_cell_area[~np.isnan(log_cell_area)]  # ← 加這行過濾 nan
    threshold = thres
    bin_width = 0.2
    bins = np.arange(min(log_cell_area), max(log_cell_area) + bin_width, bin_width)

    plt.figure(figsize=(12,5))
    counts, edges, patches = plt.hist(
        log_cell_area,
        bins=bins,
        color="blue",
        edgecolor="black",
        weights=np.ones_like(log_cell_area) / len(log_cell_area) * 100,
    )

    # 閾值紅線
    plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2)

    # 算大於 threshold 的百分比
    percent_above_threshold = np.sum(log_cell_area > threshold) / len(log_cell_area) * 100
    plt.text(
        threshold + 0.1,
        plt.ylim()[1] * 0.9,
        f"{percent_above_threshold:.2f}%",
        color="red",
        fontsize=12,
    )

    plt.xlabel("Cell Area (log3)")
    plt.ylabel("Cell Percentage (%)")
    plt.title("Cell Area Distribution (log3)")
    hist_path = outdir / f"{prefix}_log_cell_area_distribution.png"
    plt.savefig(hist_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
   #print(f"[INFO] 已輸出面積分布圖 → {hist_path}")

def plot_global_area_analysis(all_final_csv, outdir: Path, thres: float = 6.0,
                              width_um_per_px: float = 1.5896, height_um_per_px: float = 1.5876):
    """
    畫全資料夾的統計圖：
    1. 細胞質 vs 細胞核面積散布圖
    2. 細胞質面積對數分布圖（含閾值紅線）

    Parameters
    ----------
    width_um_per_px  : WIDTH  µm/pixel 倍率（預設 1.5896）
    height_um_per_px : HEIGHT µm/pixel 倍率（預設 1.5876）
    """
    # ===============================
    # 影像物理尺寸換算（由呼叫端傳入，不再硬編碼）
    # ===============================
    PIXEL_AREA = width_um_per_px * height_um_per_px  # µm²/pixel²


    df = pd.read_csv(all_final_csv)
    nuc_area_list = df["Area_nuc"].values * PIXEL_AREA   # 換算成 mm²
    cyto_area_list = df["Area_cyto"].values * PIXEL_AREA   # 換算成 mm²

    # --- 1. 散布圖 ---
    plt.figure(figsize=(12,5))
    plt.scatter(cyto_area_list, nuc_area_list, c="green", s=5, alpha=0.5)
    plt.xlabel("Cytoplasm Area")
    plt.ylabel("Nucleus Area")
    plt.title("Cytoplasm vs Nucleus Area (All Images)")
    scatter_path = outdir / "all_cell_nucleus_area.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    #print(f"[INFO] 已輸出總體散布圖 → {scatter_path}")

    # --- 2. 細胞質面積對數分布 ---
    log_cell_area = np.log(cyto_area_list) / np.log(3)  # log3
    log_cell_area = log_cell_area[~np.isnan(log_cell_area)]  # ← 加這行過濾 nan
    threshold = thres
    
    bin_width = 0.2
    bins = np.arange(min(log_cell_area), max(log_cell_area) + bin_width, bin_width)

    plt.figure(figsize=(12,5))
    plt.hist(
        log_cell_area,
        bins=bins,
        color="blue",
        edgecolor="black",
        weights=np.ones_like(log_cell_area) / len(log_cell_area) * 100,
    )
    #print(f"log_cell_area:{log_cell_area}")
    # 閾值紅線
    plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2)

    # 算大於 threshold 的百分比
    percent_above_threshold = np.sum(log_cell_area > threshold) / len(log_cell_area) * 100
    plt.text(
        threshold + 0.1,
        plt.ylim()[1] * 0.9,
        f"{percent_above_threshold:.2f}%",
        color="red",
        fontsize=12,
    )

    plt.xlabel("Cell Area (log3)")
    plt.ylabel("Cell Percentage (%)")
    plt.title("Cell Area Distribution (log3, All Images)")
    hist_path = outdir / "all_log_cell_area_distribution.png"
    plt.savefig(hist_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    #print(f"[INFO] 已輸出總體面積分布圖 → {hist_path}")