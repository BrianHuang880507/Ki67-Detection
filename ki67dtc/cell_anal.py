import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange
from natsort import natsorted
from typing import Union
from skimage.draw import polygon, polygon2mask
from skimage.io import imread
from skimage.measure import regionprops
from shapely.affinity import scale
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from skimage import io as skio
from shutil import copyfile
import cv2


from ki67dtc.utils.io import (
    list_files,
    output_dir,
    remove_temp_files,
    merged_excel,
    flatten_fluor_table,
    merge_with_flour,
    merge_all_final_csvs,
    generate_image_mapping,
)


# ===============================
# Geometry Utilities
# ===============================
def extract_polygons(geom):
    """將輸入幾何物件統一轉成 Polygon list"""
    if isinstance(geom, Polygon):
        return [geom]
    elif isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    else:
        return []


# ===============================
# 幾何參數分析
# ===============================
def param_anal(img_path, outlines_txt, output_csv):
    """計算 outlines 的各項幾何參數並輸出 CSV"""
    img = imread(img_path, as_gray=True)
    h, w = img.shape
    rows = []
    nucleus_areas = {}

    with open(outlines_txt, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(lines):
        coords = list(map(int, line.split(",")))
        X, Y = coords[::2], coords[1::2]
        if len(X) < 3:
            continue

        rr, cc = polygon(Y, X, (h, w))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[rr, cc] = 1
        props = regionprops(mask.astype(int), intensity_image=img)[0]

        roi_type = "nuc" if idx % 2 == 0 else "cyto"
        roi_id = idx // 2 + 1
        roi_name = f"{img_path.stem}_{roi_id}_{roi_type}"

        area = props.area
        perimeter = props.perimeter
        major_axis = props.major_axis_length
        minor_axis = props.minor_axis_length
        feret_max = props.feret_diameter_max
        feret_min = minor_axis

        poly = Polygon(zip(X, Y))
        convex_perimeter = poly.convex_hull.length if poly.is_valid else np.nan

        circular_diameter = 2 * np.sqrt(area / np.pi)
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else np.nan
        roundness = (4 * area) / (np.pi * major_axis**2) if major_axis > 0 else np.nan
        circularity = (
            (2 * np.sqrt(np.pi * area)) / perimeter if perimeter > 0 else np.nan
        )
        sphericity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else np.nan
        roughness = (
            1 - (convex_perimeter / perimeter)
            if perimeter > 0 and convex_perimeter > 0
            else np.nan
        )

        if roi_type == "nuc":
            nucleus_areas[roi_id] = area
            karyoplasmic_ratio = np.nan
        else:
            nuc_area = nucleus_areas.get(roi_id, np.nan)
            karyoplasmic_ratio = area / nuc_area if nuc_area > 0 else np.nan

        rows.append(
            [
                roi_name,
                area,
                perimeter,
                convex_perimeter,
                circular_diameter,
                feret_max,
                feret_min,
                aspect_ratio,
                roundness,
                circularity,
                sphericity,
                roughness,
                karyoplasmic_ratio,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Cell_ID",
            "Area",
            "Perimeter",
            "Convex Perimeter",
            "Circular Diameter",
            "Feret Length",
            "Feret Width",
            "Aspect Ratio",
            "Roundness",
            "Circularity",
            "Sphericity",
            "Roughness",
            "Karyoplasmic Ratio",
        ],
    )
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved full measurement table → {output_csv}")


# ===============================
# 螢光分析
# ===============================
def flour_anal(
    img_path, outlines_txt, output_csv, max_expand_steps=20, expand_factor=0.5
):
    """
    螢光分析流程：
    - 逐步擴張 nucleus
    - 每次與前一次做 XOR
    - XOR 結果再與 cytoplasm 做 AND
    - 計算 IntDen 與 RawIntDen
    """
    img = imread(img_path, as_gray=True)
    h, w = img.shape

    with open(outlines_txt, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    num_pairs = len(lines) // 2
    rows = []

    for i in range(num_pairs):
        nuc_coords = list(map(int, lines[2 * i].split(",")))
        cyto_coords = list(map(int, lines[2 * i + 1].split(",")))

        if len(nuc_coords) < 6 or len(cyto_coords) < 6:
            continue

        nuc_poly = Polygon(np.array(nuc_coords).reshape(-1, 2))
        cyto_poly = Polygon(np.array(cyto_coords).reshape(-1, 2))

        nuc_poly = nuc_poly.buffer(0)
        cyto_poly = cyto_poly.buffer(0)

        if not nuc_poly.is_valid or not cyto_poly.is_valid:
            continue

        prev_poly = nuc_poly

        for j in range(max_expand_steps):
            factor = 1 + expand_factor * (j + 1)
            scaled_nuc = scale(nuc_poly, xfact=factor, yfact=factor, origin="center")

            xor_poly = scaled_nuc.symmetric_difference(prev_poly)
            if xor_poly.is_empty:
                prev_poly = scaled_nuc
                continue

            and_poly = xor_poly.intersection(cyto_poly)
            if and_poly.is_empty:
                prev_poly = scaled_nuc
                continue

            polys = extract_polygons(and_poly)
            mask = np.zeros((h, w), dtype=bool)
            for poly in polys:
                coords = np.array(poly.exterior.coords).round().astype(int)
                rr, cc = polygon(coords[:, 1], coords[:, 0], (h, w))
                mask[rr, cc] = True

            if np.any(mask):
                props = regionprops(mask.astype(int), intensity_image=img)[0]
                area = props.area
                mean_gray_value = props.mean_intensity
                int_den = area * mean_gray_value
                raw_int_den = img[mask].sum()
                label = f"{Path(img_path).name}:NewCell-{i+1}-and{j}"
                rows.append([label, int_den, raw_int_den])

            prev_poly = scaled_nuc

    df = pd.DataFrame(rows, columns=["Label", "IntDen", "RawIntDen"])
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved fluorescence analysis → {output_csv}")


# ===============================
# Ki67 陽性判斷與合併
# ===============================
def ki67_binarize(img_path: Union[str, Path]) -> Path:
    """
    對單張 Ki67 影像進行 Otsu 二值化，輸出為 PNG 格式，回傳輸出路徑。
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"[ERROR] 找不到圖片: {img_path}")

    # Otsu 二值化
    _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 輸出路徑
    binary_dir = output_dir(img_path.parent.parent, "binary")  # 從 PC 資料夾回到 root
    out_path = binary_dir / f"{img_path.stem}_binary.png"

    cv2.imwrite(str(out_path), binary_otsu)
    print(f"[INFO] 已輸出 Ki67 二值圖: {out_path}")

    return out_path


def detect_ki67_positive(
    roi_dir: Path, ki67_dir: Path, output_dir: Path, threshold: float = 0.10
):
    """Ki67 陽性 ROI 判斷"""
    """判斷單一 ROI txt 是否為 Ki67 陽性，並輸出 label.txt"""
    if not ki67_dir.exists():
        print(f"[WARN] 找不到 Ki67 mask: {ki67_dir}")
        return

    ki67_mask = skio.imread(str(ki67_dir)) > 0
    shape = ki67_mask.shape

    with open(roi_dir) as f:
        lines = f.readlines()

    positive_labels = []
    for idx, line in enumerate(lines):
        if idx % 2:
            continue
        coords = list(map(int, line.strip().split(",")))
        xy = np.array(coords).reshape(-1, 2)
        mask = polygon2mask(shape, xy[:, [1, 0]])
        roi_area = mask.sum()
        if roi_area == 0:
            continue
        overlap_area = np.logical_and(mask, ki67_mask).sum()
        if overlap_area / roi_area >= threshold:
            positive_labels.append(idx + 1)

    np.savetxt(output_dir, positive_labels, fmt="%d")
    print(f"[INFO] 已輸出 Ki67 陽性 label: {output_dir}")


def merge_ki67_labels(param_csv: Path, label_file: Path, output_csv: Path):
    """將單一檔案合併 Ki67 陽性標記"""
    if not label_file.exists():
        print(f"[WARN] 缺少陽性 label: {label_file.name}")
        return

    df = pd.read_csv(param_csv)
    positive_labels = np.loadtxt(label_file, dtype=int)

    if positive_labels.ndim == 0:
        positive_labels = [int(positive_labels)]
    positive_groups = set((label - 1) // 2 for label in positive_labels)

    df["ki67_positive"] = [1 if idx in positive_groups else 0 for idx in df.index]
    df.to_csv(output_csv, index=False)
    print(f"[INFO] 已合併 Ki67 標記 → {output_csv}")


# ===============================
# 主流程
# ===============================
def run_all(
    data_name: str,
    fluor_analy: bool = False,
    ki67: bool = False,
    clean_temp: bool = True,
) -> None:
    data_path = Path(data_name)
    pc_dir = data_path / "PC"
    df_dir = data_path / "DF"
    ki67_dir = data_path / "KI67"
    mapping = pd.read_csv(generate_image_mapping(pc_dir, df_dir, ki67_dir))

    analy_dir = output_dir(data_path, "results")
    outline_dir = output_dir(data_path, "outline")
    # all_img_files = list_files(pc_dir, [".jpg", ".png", ".tif", ".tiff"])

    for i in trange(len(mapping), desc="Processing images(analysis)"):
        row = mapping.iloc[i]
        pc_name = str(row["PC_Name"]) if pd.notna(row["PC_Name"]) else ""
        df_name = str(row["DF_Name"]) if pd.notna(row["DF_Name"]) else ""
        ki67_name = str(row["KI67_Name"]) if pd.notna(row["KI67_Name"]) else ""

        pc_img = pc_dir / pc_name
        df_img = df_dir / df_name
        ki67_img = ki67_dir / ki67_name

        outlines_txt = outline_dir / f"{pc_img.stem}_merged_cp_outlines.txt"
        if not outlines_txt.exists():
            print(f"[WARN] 缺少 outlines 檔案: {outlines_txt}")
            continue

        param_csv = analy_dir / f"{pc_img.stem}_params.csv"

        merged_param_csv = analy_dir / f"{pc_img.stem}_params_merged.csv"
        fluor_csv = analy_dir / f"{pc_img.stem}_fluorescence.csv"
        flat_fluor_csv = analy_dir / f"{pc_img.stem}_fluor_flat.csv"
        final_csv = analy_dir / f"{pc_img.stem}_final.csv"

        # 幾何分析 + 合併
        param_anal(pc_img, outlines_txt, param_csv)
        merged_excel(param_csv, merged_param_csv)

        if fluor_analy:
            # 螢光分析 + 攤平 + 合併
            flour_anal(df_img, outlines_txt, fluor_csv)
            flatten_fluor_table(fluor_csv, flat_fluor_csv)
            merge_with_flour(merged_param_csv, flat_fluor_csv, final_csv)
            print(f"[INFO] 螢光分析完成 → {final_csv}")
        else:
            copyfile(merged_param_csv, final_csv)

        if ki67:
            # Ki67 陽性 ROI 判斷與合併
            ki67_mask = ki67_binarize(ki67_img)
            binary_dir = output_dir(data_path, "binary")
            ki67_label = binary_dir / f"{pc_img.stem}_label.txt"

            detect_ki67_positive(outlines_txt, ki67_mask, ki67_label)
            merge_ki67_labels(final_csv, ki67_label, final_csv)
            print(f"[INFO] Ki67 判斷完成 → {final_csv}")

        if not fluor_analy and not ki67:
            # 什麼都沒選就直接 copy
            merged_param_csv.replace(final_csv)
            print(f"[INFO] Copied merged params to final → {final_csv}")

    merge_all_final_csvs(data_path)

    if clean_temp:
        remove_temp_files(analy_dir)
        remove_temp_files(output_dir(data_path, "outline"))
