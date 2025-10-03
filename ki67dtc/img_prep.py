import shutil
from pathlib import Path
import numpy as np
from tqdm import trange
from cellpose import models, io
from shapely.geometry import Point, Polygon
from skimage.draw import polygon
from PIL import Image


from ki67dtc.utils.io import list_files, output_dir, load_outlines, remove_temp_files

# 模型路徑固定
CYTO_MODEL_PATH = "model/model_BDL6_label_new"
NUC_MODEL_PATH = "model/model_BDL3_label_dapi"


# ===============================
# 分割流程
# ===============================
def segment(model_path: str, img_files: list[Path], output_dir: Path, suffix: str):
    """
    使用指定模型進行細胞質 (cyto) 或細胞核 (nuc) 分割
    輸出 segmentation 結果到指定資料夾
    """
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    for i in trange(len(img_files), desc=f"Segmenting ({suffix})"):
        f = img_files[i]
        img = io.imread(f)
        masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])
        io.masks_flows_to_seg(img, masks, flows, f, channels=[0, 0], diams=None)
        seg_file = f.with_name(f"{f.stem}_seg.npy")
        target_file = output_dir / f"{f.stem}_{suffix}_seg.npy"
        shutil.move(seg_file, target_file)


def segment_all(input_dir: str):
    """
    遍歷資料夾，僅處理相位差圖片 (檔名不包含 Ki67 或 DF)
    """
    input_dir = Path(input_dir) / "PC"
    seg_dir = output_dir(input_dir.parent, "segment")
    img_files = [
        f
        for f in list_files(input_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
        if "ki67" not in f.stem.lower() and "df" not in f.stem.lower()
    ]
    if not img_files:
        print(f"[WARN] 找不到相位差圖片於 {input_dir}")
        return
    segment(CYTO_MODEL_PATH, img_files, seg_dir, "cyto")
    segment(NUC_MODEL_PATH, img_files, seg_dir, "nuc")


# ===============================
# Mask 輸出與 outlines 轉換
# ===============================
def mask2txt(npy_files: list[Path], output_dir: Path):
    """將所有 segmentation npy 檔轉換為 outlines (txt)"""
    for i in trange(len(npy_files), desc=f"Saving {len(npy_files)} masks"):
        f = npy_files[i]
        data = np.load(f, allow_pickle=True).item()
        masks, flows = data["masks"], data["flows"]
        io.save_masks(
            None,
            masks,
            flows,
            f.stem,
            png=False,
            channels=[0, 0],
            save_txt=True,
            savedir=output_dir,
        )


def mask2txt_all(input_dir: str):
    """
    將 cytoplasm 和 nucleus segmentation 的 npy 結果
    輸出為 mask/txt 格式
    """
    input_dir = Path(input_dir)
    seg_dir = output_dir(input_dir, "segment")
    mask2txt_out = output_dir(input_dir, "outline")
    cyto_files = [f for f in list_files(seg_dir, ".npy") if "_cyto_seg" in f.stem]
    nuc_files = [f for f in list_files(seg_dir, ".npy") if "_nuc_seg" in f.stem]
    mask2txt(cyto_files, mask2txt_out)
    mask2txt(nuc_files, mask2txt_out)


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

def find_image_and_nuc_file(cyto_file: Path) -> Tuple[Optional[Path], Optional[Path]]:
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
        img_candidate = Path("data/input") / dataset_name / "PC" / f"{index_key}{ext}"
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


def combined(input_dir: str):
    """合併 cytoplasm 與 nucleus outlines"""
    input_dir = Path(input_dir)
    outline_dir = output_dir(input_dir, "outline")
    txt_files = [f for f in list_files(outline_dir, ".txt") if "_cyto_seg" in f.stem]
    if not txt_files:
        print(f"[WARN] 找不到 cytoplasm outlines 檔案於 {outline_dir}")
        return
    for cyto_file in trange(len(txt_files), desc="Merging outlines"):
        cyto_path = txt_files[cyto_file]
        nuc_path, img_path = find_image_and_nuc_file(cyto_path)
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
        match_and_write(cyto_lines, nuc_lines, cyto_mask, nuc_mask, out_path)
    remove_temp_files(outline_dir)
