import shutil
from pathlib import Path
import numpy as np
from tqdm import trange
from cellpose import models, io
from shapely.geometry import Point, Polygon
from skimage.draw import polygon
from PIL import Image
from imageio.v2 import imread
import cv2
from ki67dtc.utils.io import list_files, output_dir, load_outlines, remove_temp_files

# ===============================
# 分割流程
# ===============================
NUC_MODEL_INPUT_SIZE  = (1280, 1024)  # (width, height)，細胞核模型訓練尺寸
CYTO_MODEL_INPUT_SIZE = (1280, 1024)  # (width, height)，細胞質模型訓練尺寸

def segment(model_path: str, img_files: list[Path], suffix: str, progress_callback=None, total=None, offset=0, status_callback=None):
    """
    使用指定模型進行細胞質 (cyto) 或細胞核 (nuc) 分割
    輸出 segmentation 結果到指定資料夾
    cyto / nuc 模型皆會先將影像 resize 至各自的訓練尺寸再推論，
    取得 masks 後再 resize 回原始尺寸，確保後續 outline/csv 座標正確。
    """
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    if suffix == "nuc":
        target_w, target_h = NUC_MODEL_INPUT_SIZE
    elif suffix == "cyto":
        target_w, target_h = CYTO_MODEL_INPUT_SIZE
    else:
        target_w, target_h = None, None

    for i in trange(len(img_files), desc=f"Segmenting ({suffix})"):
        f = img_files[i]
        if status_callback:
            status_callback(f"Segmenting ({suffix}) {i+1}/{len(img_files)}: {f.name}")
        img = io.imread(f)
        f_new = f.with_name(f"{f.stem}_{suffix}{f.suffix}")

        if target_w is not None:
            orig_h, orig_w = img.shape[:2]
            # Resize 到模型訓練尺寸
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            masks_resized, flows_resized, styles = model.eval(img_resized, diameter=None, channels=[0, 0])
            # Resize masks 回原始尺寸（使用最近鄰插值以保留 label ID）
            masks = cv2.resize(
                masks_resized.astype(np.int32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )
            # flows[0] 是 RGB 顯示圖，也需 resize 回原尺寸
            flows_rgb = cv2.resize(flows_resized[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            flows_resized[0] = flows_rgb
            io.masks_flows_to_seg(img, masks, flows_resized, f_new, channels=[0, 0], diams=None)
        else:
            masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])
            io.masks_flows_to_seg(img, masks, flows, f_new, channels=[0, 0], diams=None)

        # Progress update
        if progress_callback and total:
            progress_callback(offset + i + 1)


def segment_all(input_dir: str, out_dir: str, CYTO_MODEL_PATH: str, NUC_MODEL_PATH: str, progress_callback=None, status_callback=None):
    """
    遍歷資料夾，僅處理相位差圖片 (檔名不包含 Ki67 或 DF)
    """
    input_dir = Path(input_dir)
    img_files = [
        f
        for f in list_files(input_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
        if "ki67" not in f.stem.lower() and "df" not in f.stem.lower()
    ]
    if not img_files:
        print(f"[WARN] 找不到相位差圖片於 {input_dir}")
        return
    total = len(img_files)
    # Pass progress_callback and total to segment
    segment(CYTO_MODEL_PATH, img_files, "cyto", progress_callback, total, offset=0, status_callback=status_callback)
    segment(NUC_MODEL_PATH, img_files, "nuc", progress_callback, total, offset=total, status_callback=status_callback)


# ===============================
# Mask 輸出與 outlines 轉換
# ===============================
def mask2txt(npy_files: list[Path], output_dir: Path, input_dir: Path, progress_callback=None, offset=0):
    """將所有 segmentation npy 檔轉換為 outlines (txt)"""
    for i in trange(len(npy_files), desc=f"Saving {len(npy_files)} masks"):
        f = npy_files[i]

        data = np.load(f, allow_pickle=True).item()
        masks, flows = data["masks"], data["flows"]
        #-----------------
        path = Path(npy_files[i])

        if "_cyto_seg" in f.stem:
            path_stem = path.stem.replace("_cyto_seg", "")
        elif "_nuc_seg" in f.stem:
            path_stem = path.stem.replace("_nuc_seg", "")
        img = imread(f"{input_dir}/{path_stem}.jpg")
        #-----------------
        io.save_masks(
            img,
            masks,
            flows,
            f.stem,
            png=False,
            channels=[0, 0],
            save_txt=True,
            savedir=output_dir,
        )
        if progress_callback:
            progress_callback(offset + i + 1)


def mask2txt_all(input_dir: str, out_dir: str, progress_callback=None, base_offset=0):
    """
    將 cytoplasm 和 nucleus segmentation 的 npy 結果
    輸出為 mask/txt 格式
    """
    input_dir = Path(input_dir)
    mask2txt_out = output_dir(out_dir, "outline")
    # Clear output folder
    if mask2txt_out.exists():
        for f in mask2txt_out.iterdir():
            if f.is_file():
                f.unlink()
    cyto_files = [f for f in list_files(input_dir, ".npy") if "_cyto_seg" in f.stem]
    nuc_files = [f for f in list_files(input_dir, ".npy") if "_nuc_seg" in f.stem]
    mask2txt(cyto_files, mask2txt_out, input_dir, progress_callback=progress_callback, offset=base_offset)
    mask2txt(nuc_files, mask2txt_out, input_dir, progress_callback=progress_callback, offset=base_offset + len(cyto_files))


# ===============================
# Outlines 合併
# ===============================
def create_mask(outlines, shape):
    """將 outlines 轉換為 mask 陣列"""
    mask = np.zeros(shape, dtype=np.int32)
    for i, line in enumerate(outlines, start=1):
        coords = list(map(int, line.split(",")))
        xs, ys = coords[::2], coords[1::2]
        rr, cc = polygon(ys, xs, shape)
        mask[rr, cc] = i
    return mask


from pathlib import Path
from typing import Optional, Tuple

def find_image_and_nuc_file(input_dir: Path, cyto_file: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """根據 cytoplasm outlines 找對應的 nucleus outlines 與原圖"""
    # 建立 nucleus 檔案路徑
    nuc_file = cyto_file.with_name(
        cyto_file.name.replace("_cyto_seg_cp_outlines.txt", "_nuc_seg_cp_outlines.txt")
    )
    if not nuc_file.exists():
        return None, None

    # 從 cyto_file 取得 dataset_name
    dataset_name = cyto_file.parent.name  # e.g., 2025-07-10-B8-P6...
    index_key = cyto_file.stem.replace("_cyto_seg_cp_outlines", "")

    # 拼接圖片路徑
    for ext in [".jpg", ".png", ".tif", ".tiff"]:
        img_candidate = input_dir / f"{index_key}{ext}"
        if img_candidate.exists():
            return nuc_file, img_candidate

    return None, None


def match_and_write(cyto_lines, nuc_lines, cyto_mask, nuc_mask, out_path: Path):
    """配對 nucleus 與 cytoplasm 並輸出合併的 outlines"""
    cyto_ids = np.unique(cyto_mask)[1:]
    nuc_ids = np.unique(nuc_mask)[1:]
    cyto_polygons = {}
    for i, line in enumerate(cyto_lines):
        coords = list(map(int, line.strip().split(",")))
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        poly = Polygon(points)
        if poly.is_valid and not poly.is_empty:
            cyto_polygons[i] = poly
    pairings, used_cyto = {}, set()
    for ni, line in enumerate(nuc_lines):
        coords = list(map(int, line.strip().split(",")))
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        if len(points) < 3:
            continue
        center = np.mean(points, axis=0)
        point = Point(center)
        matched = None
        for ci, poly in cyto_polygons.items():
            if ci in used_cyto:
                continue
            if poly.contains(point):
                matched = ci
                break
        if matched is not None:
            pairings[ni] = matched
            used_cyto.add(matched)
    with open(out_path, "w") as f:
        written_nuc, written_cyto = set(), set()
        for ni, ci in sorted(pairings.items()):
            f.write(nuc_lines[ni].strip() + "\n")
            f.write(cyto_lines[ci].strip() + "\n")
            written_nuc.add(ni)
            written_cyto.add(ci)
        for ni in sorted(set(range(len(nuc_lines))) - written_nuc):
            f.write(nuc_lines[ni].strip() + "\n")
            f.write("-1,-1\n")
        for ci in sorted(set(range(len(cyto_lines))) - written_cyto):
            f.write("-1,-1\n")
            f.write(cyto_lines[ci].strip() + "\n")
    num_matched = len(pairings)
    return float(num_matched)
    


def combined(input_dir: str, out_dir: str, progress_callback=None, offset=0):
    """合併 cytoplasm 與 nucleus outlines"""
    input_dir = Path(input_dir)
    outline_dir = output_dir(out_dir, "outline")
    txt_files = [f for f in list_files(outline_dir, ".txt") if "_cyto_seg" in f.stem]
    if not txt_files:
        print(f"[WARN] 找不到 cytoplasm outlines 檔案於 {outline_dir}")
        return
    for cyto_file in trange(len(txt_files), desc="Merging outlines"):
        if progress_callback:
            progress_callback(offset + cyto_file + 1)
        cyto_path = txt_files[cyto_file]
        nuc_path, img_path = find_image_and_nuc_file(input_dir, cyto_path)
        print(nuc_path, img_path)
        if not nuc_path or not img_path:
            print(f"[WARN] 缺少 nucleus 或原圖: {cyto_path.name}")
            continue
        img = Image.open(img_path)
        shape = img.size[::-1]
        cyto_lines = load_outlines(cyto_path)
        nuc_lines = load_outlines(nuc_path)
        cyto_mask = create_mask(cyto_lines, shape)
        nuc_mask = create_mask(nuc_lines, shape)
        out_path = cyto_path.with_name(
            cyto_path.stem.replace("_cyto_seg_cp_outlines", "_merged_cp_outlines.txt")
        )
        num_matched = match_and_write(cyto_lines, nuc_lines, cyto_mask, nuc_mask, out_path)
        print(f"num_matched:{num_matched}")
