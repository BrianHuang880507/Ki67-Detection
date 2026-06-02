from pathlib import Path
from natsort import natsorted
from typing import Union
import pandas as pd
import os


CELL_LEVEL_PARAMETER_COLUMNS = [
    "Karyoplasmic Ratio",
    "Nuc Cyto Mean Ratio",
    "Nuc Cyto IntDen Ratio",
    "Nuc Cyto RawIntDen Ratio",
    "Nuc Cyto Entropy Difference",
    "Nuc Cyto CV Difference",
    "Nucleus Centroid Offset",
    "Halo Outer Mean",
    "Halo Outer StdDev",
    "Halo Inner Mean",
    "Halo Inner StdDev",
    "Halo Inner Outer Diff",
    "Halo Width",
    "Edge Sharpness",
    "Image Confluency",
    "Population Area CV",
    "Population Circularity CV",
    "Nearest Neighbor Distance",
    "Nearest Neighbor Distance Norm",
    "Local Neighbor Count",
    "Local Density",
    "Cluster Size",
    "Cluster Size Norm",
    "Largest Cluster Ratio",
    "Protrusion Count",
    "Mean Convex Defect Depth",
    "Max Convex Defect Depth",
    "Fractal Dimension",
    "Boundary Inflection Count",
    "Debris Count",
    "Debris Area Fraction",
    "Nearest Debris Distance",
    "Debris Mean Area",
    "Debris Density",
    "Mitotic Score",
    "Daughter Pair Flag",
    "Protrusion Retraction Score",
]


def extract_id(label):
    """從螢光 Label 欄位解析主流程 Cell_ID。

    Args:
        label: 形如 `<image>:NewCell-<id>-and<roi>` 的 Label 字串。

    Returns:
        str: 主流程使用的 `<image>_<cell_id>` 格式。
    """
    parts = label.split(":")[1].split("-")  # NewCell-1-and2
    img = label.split(":")[0]
    return f"{img}_{parts[1]}"


def extract_index(cell_id):
    """取得 Cell_ID 最後一段數字索引。

    Args:
        cell_id: 形如 `<image>_<cell_id>` 的細胞識別字串。

    Returns:
        int: 細胞數字索引。
    """
    return int(cell_id.split("_")[-1])


def list_files(folder: Path, exts: Union[str, tuple, list] = None) -> list[Path]:
    """
    掃描資料夾，依自然排序回傳符合副檔名的檔案路徑。
    """
    if exts is None:
        return natsorted([f for f in folder.iterdir() if f.is_file()])

    if isinstance(exts, str):
        exts = (exts,)
    else:
        exts = tuple(exts)

    return natsorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts]
    )


def output_dir(input_dir: Path, subfolder: str) -> Path:
    """
    根據 input_dir 自動建立對應的輸出資料夾
    """
    output_root = Path(f"./data/output/{subfolder}")
    output_dir = output_root / input_dir.name
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_outlines(filepath):
    """讀取 outlines txt 檔案內容，回傳字串列表"""
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def remove_temp_files(folder: Path, keywords: list[str] = None, exts: list[str] = None):
    """
    刪除含有特定關鍵字與副檔名的暫存檔案。
    """
    if keywords is None:
        keywords = [
            "params",
            "params_merged",
            "fluorescence",
            "fluor_flat",
            "ido_measurements",
            "_cyto_seg_cp_outlines",
            "_nuc_seg_cp_outlines",
            "_cyto_seg.npy",
            "_nuc_seg.npy",
            "_seg.npy",
        ]

    deleted_files = []
    for f in folder.glob("*"):
        if not f.is_file():
            continue
        if exts and f.suffix.lower() not in exts:
            continue
        if any(k in f.name for k in keywords):
            try:
                f.unlink()
                deleted_files.append(f.name)
            except Exception as e:
                print(f"[WARN] 無法刪除 {f.name} → {e}")

    print(f"[INFO] 刪除暫存檔案共 {len(deleted_files)} 個")


# ===============================
# Excel / CSV 資料處理工具
# ===============================
def merged_excel(input: Union[str, Path], output: Union[str, Path]):
    """
    合併兩個 CSV 檔案：
    - 將核與質兩部分以 Cell_ID 合併
    - ROI-specific 參數保留 _nuc/_cyto
    - cell-level 參數只保留一份，不再加 _nuc/_cyto
    """
    df = pd.read_csv(input)

    df["ROI_Type"] = df["Cell_ID"].apply(lambda x: x.split("_")[-1])
    df["Cell_ID"] = df["Cell_ID"].apply(lambda x: "_".join(x.split("_")[:-1]))

    nucleus_df = df[df["ROI_Type"].str.lower() == "nuc"].copy()
    cyto_df = df[df["ROI_Type"].str.lower() == "cyto"].copy()

    nucleus_df = nucleus_df.drop(columns=["ROI_Type"]).set_index("Cell_ID")
    cyto_df = cyto_df.drop(columns=["ROI_Type"]).set_index("Cell_ID")

    cell_level_cols = [
        col for col in CELL_LEVEL_PARAMETER_COLUMNS if col in cyto_df.columns
    ]
    cell_level_df = cyto_df[cell_level_cols].copy() if cell_level_cols else None

    nucleus_df = nucleus_df.drop(
        columns=CELL_LEVEL_PARAMETER_COLUMNS, errors="ignore"
    )
    cyto_df = cyto_df.drop(columns=cell_level_cols, errors="ignore")

    nucleus_df = nucleus_df.add_suffix("_nuc")
    cyto_df = cyto_df.add_suffix("_cyto")

    parts = [nucleus_df, cyto_df]
    if cell_level_df is not None:
        parts.append(cell_level_df)
    merged_df = pd.concat(parts, axis=1).reset_index()
    merged_df.to_csv(output, index=False)


def flatten_fluor_table(flour_csv: Union[str, Path], output: Union[str, Path]):
    """
    攤平成一列一個 Label 的螢光分析表：
    - 從 Label 欄位提取 Cell_ID 與 ROI_ID
    - 每列代表一個 Cell，每欄對應一個 ROI 的 IntDen / RawIntDen 值
    - 輸出為寬表（wide format）至新 CSV
    """
    df = pd.read_csv(flour_csv)
    df = df.drop(columns=[" "], errors="ignore")

    df["Cell_ID"] = df["Label"].apply(extract_id)
    df["ROI_ID"] = df["Label"].str.extract(r"-and(\d+)")[0]
    df = df[df["ROI_ID"].notna()].copy()
    df["ROI_ID"] = df["ROI_ID"].astype(int)

    if df.empty:
        print("[警告] 沒有 ROI_ID 可轉換，請檢查 Label 格式")
        return pd.DataFrame()

    df_wide = df.pivot(
        index="Cell_ID", columns="ROI_ID", values=["IntDen", "RawIntDen"]
    )
    df_wide.columns = [f"{col[0]}-{col[1]}" for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    df_wide["SortKey"] = df_wide["Cell_ID"].apply(extract_index)
    df_wide = df_wide.sort_values("SortKey").drop(columns="SortKey")

    df_wide.to_csv(output, index=False)


def merge_with_flour(
    df1_csv: Union[str, Path], df2_csv: Union[str, Path], output_csv: Union[str, Path]
):
    """
    依照 Cell_ID 合併幾何與螢光分析結果：
    - 以 index 對應方式合併兩份資料
    - 輸出合併後的 CSV
    """
    df1 = pd.read_csv(df1_csv)
    df2 = pd.read_csv(df2_csv)

    df1["Cell_Index"] = df1["Cell_ID"].apply(extract_index)
    df2["Cell_Index"] = df2["Cell_ID"].apply(extract_index)

    merged = pd.merge(df1, df2.drop(columns=["Cell_ID"]), on="Cell_Index", how="left")
    merged = merged.drop(columns=["Cell_Index"])
    merged.to_csv(output_csv, index=False)


def merge_all_final_csvs(input_dir: Union[str, Path]):
    """
    合併某資料夾內所有 *_final.csv 為一個 CSV 檔。
    - 會在每筆資料中加入 'Image' 欄位標記來源影像
    - 會清除任一 `_nuc` 或 `_cyto` 欄位缺值的列，避免殘缺紀錄
    - 輸出為 {input_dir.name}_final.csv
    """
    input_dir = Path(input_dir)
    result_dir = output_dir(input_dir, "results")
    final_files = sorted(result_dir.glob("*_final.csv"))

    if not final_files:
        print(f"[WARN] 找不到 *_final.csv 於 {input_dir}")
        return

    all_dfs = []
    for file in final_files:
        df = pd.read_csv(file)
        df["Image"] = file.stem.replace("_final", "")
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    nuc_cols = [c for c in merged_df.columns if c.endswith("_nuc")]
    cyto_cols = [c for c in merged_df.columns if c.endswith("_cyto")]
    cols_to_check = []
    if "cell_status" in merged_df.columns:
        before = len(merged_df)
        keep_mask = merged_df["cell_status"].isin(
            ["full_cell", "nuc_only", "cyto_cut"]
        )
        merged_df = merged_df[keep_mask].reset_index(drop=True)
        removed = before - len(merged_df)
        if removed > 0:
            print(f"[INFO] 依 cell_status 移除 {removed} 筆不保留的細胞資料")

        status = merged_df.pop("cell_status")
        merged_df["cell_status"] = status
    else:
        cols_to_check = nuc_cols + cyto_cols

    if "cell_status" not in merged_df.columns and cols_to_check:
        before = len(merged_df)
        merged_df = merged_df.dropna(subset=cols_to_check, how="any").reset_index(
            drop=True
        )
        removed = before - len(merged_df)
        if removed > 0:
            print(f"[INFO] 已移除 {removed} 筆缺少核/質特徵的資料列")

    output_path = result_dir / f"{input_dir.name}_cleaned.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"[INFO] 已合併 {len(final_files)} 個檔案 → {output_path}")


def generate_image_mapping(
    pc_dir, df_dir, ki67_dir, output_csv="image_mapping.csv"
) -> Path:
    """依 PC/DF/KI67 資料夾建立影像對照表。

    Args:
        pc_dir: PC 影像資料夾。
        df_dir: DF 影像資料夾。
        ki67_dir: Ki67 影像資料夾。
        output_csv: 輸出的對照表檔名。

    Returns:
        Path: 產生的 `image_mapping.csv` 路徑。
    """
    pc_dir = Path(pc_dir)
    df_dir = Path(df_dir)
    ki67_dir = Path(ki67_dir)

    def get_images(folder):
        """列出資料夾內支援的影像檔名。"""
        return natsorted(
            [
                f.name
                for f in folder.glob("*")
                if f.suffix.lower() in [".jpg", ".png", ".tif", ".tiff"]
            ]
        )

    pc_imgs = get_images(pc_dir)
    df_imgs = get_images(df_dir)
    ki67_imgs = get_images(ki67_dir)

    max_len = max(len(pc_imgs), len(df_imgs), len(ki67_imgs))

    def safe_get(lst, idx):
        """安全取得清單元素，索引超出時回傳空字串。"""
        return lst[idx] if idx < len(lst) else ""

    rows = []
    for i in range(max_len):
        rows.append(
            {
                "PC_Name": safe_get(pc_imgs, i),
                "DF_Name": safe_get(df_imgs, i),
                "KI67_Name": safe_get(ki67_imgs, i),
            }
        )

    df = pd.DataFrame(rows)
    out_path = pc_dir.parent / output_csv
    df.to_csv(str(out_path), index=False)
    print(f"[INFO] 產生 image_mapping.csv → {pc_dir.parent / output_csv}")
    return out_path
