import shutil
import re
from pathlib import Path
import numpy as np
from tqdm import trange
from cellpose import models, io
from shapely.geometry import Point, Polygon
from skimage.draw import polygon
from PIL import Image


from ki67dtc.utils.io import list_files, output_dir, load_outlines, remove_temp_files

# 璅∪?頝臬??箏?
CYTO_MODEL_PATH = "model/model_BDL6_label_new"
NUC_MODEL_PATH = "cyto3"


# ===============================
# ?瘚?
# ===============================
def segment(
    model_path: str,
    img_files: list[Path],
    output_dir: Path,
    suffix: str,
    output_stems: list[str] | None = None,
    channels: list[int] | tuple[int, int] = (0, 0),
    diameter: float | None = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    invert: bool = False,
):
    """
    雿輻??璅∪??脰?蝝啗?鞈?(cyto) ?敦? (nuc) ?
    頛詨 segmentation 蝯??唳?摰??冗
    """
    if output_stems is not None and len(output_stems) != len(img_files):
        raise ValueError("output_stems length must match img_files length.")

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    for i in trange(len(img_files), desc=f"Segmenting ({suffix})"):
        f = img_files[i]
        img = io.imread(f)
        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            channels=list(channels),
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            invert=invert,
        )
        io.masks_flows_to_seg(
            img, masks, flows, f, channels=list(channels), diams=diameter
        )
        seg_file = f.with_name(f"{f.stem}_seg.npy")
        target_stem = output_stems[i] if output_stems is not None else f.stem
        target_file = output_dir / f"{target_stem}_{suffix}_seg.npy"
        shutil.move(seg_file, target_file)


def _last_numeric_token(stem: str) -> int | None:
    match = re.search(r"(\d+)(?!.*\d)", stem)
    return int(match.group(1)) if match else None


def segment_all(input_dir: str, nuc_source: str = "dapi", dapi_dir_name: str = "DAPI"):
    """
    執行細胞質與細胞核分割。
    - cytoplasm 一律使用 PC 影像
    - nucleus 可選擇 PC 或 DAPI；若選 DAPI，channel 使用 [3, 3]
    """
    root_dir = Path(input_dir)
    pc_dir = root_dir / "PC"
    seg_dir = output_dir(root_dir, "segment")

    img_files = [
        f
        for f in list_files(pc_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
        if "ki67" not in f.stem.lower() and "df" not in f.stem.lower()
    ]
    if not img_files:
        print(f"[WARN] 找不到相位差圖片於 {pc_dir}")
        return

    nuc_source = nuc_source.strip().lower()
    if nuc_source not in {"pc", "dapi"}:
        raise ValueError(f"Unsupported nuc_source: {nuc_source}")

    # Cytoplasm segmentation always uses PC channel [0, 0].
    segment(CYTO_MODEL_PATH, img_files, seg_dir, "cyto", channels=(0, 0))

    if nuc_source == "pc":
        segment(NUC_MODEL_PATH, img_files, seg_dir, "nuc", channels=(0, 0))
        return

    dapi_dir = root_dir / dapi_dir_name
    if not dapi_dir.exists() or not dapi_dir.is_dir():
        print(f"[WARN] 找不到 DAPI 資料夾 {dapi_dir}，nucleus 改回使用 PC。")
        segment(NUC_MODEL_PATH, img_files, seg_dir, "nuc", channels=(0, 0))
        return

    dapi_files = list_files(dapi_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    if not dapi_files:
        print(f"[WARN] DAPI 資料夾沒有可用影像 {dapi_dir}，nucleus 改回使用 PC。")
        segment(NUC_MODEL_PATH, img_files, seg_dir, "nuc", channels=(0, 0))
        return

    dapi_by_stem: dict[str, list[Path]] = {}
    dapi_by_idx: dict[int, list[Path]] = {}
    for dapi in dapi_files:
        dapi_by_stem.setdefault(dapi.stem.lower(), []).append(dapi)
        idx = _last_numeric_token(dapi.stem)
        if idx is not None:
            dapi_by_idx.setdefault(idx, []).append(dapi)

    used_dapi: set[Path] = set()
    dapi_nuc_img_files: list[Path] = []
    dapi_output_stems: list[str] = []
    fallback_pc_nuc_img_files: list[Path] = []
    fallback_pc_output_stems: list[str] = []
    fallback_pc_count = 0

    for pc in img_files:
        candidates: list[Path] = []
        candidates.extend(dapi_by_stem.get(pc.stem.lower(), []))
        idx = _last_numeric_token(pc.stem)
        if idx is not None:
            candidates.extend(dapi_by_idx.get(idx, []))

        matched: Path | None = None
        for c in candidates:
            if c not in used_dapi:
                matched = c
                used_dapi.add(c)
                break

        if matched is None:
            fallback_pc_count += 1
            fallback_pc_nuc_img_files.append(pc)
            fallback_pc_output_stems.append(pc.stem)
        else:
            dapi_nuc_img_files.append(matched)
            dapi_output_stems.append(pc.stem)

    if fallback_pc_count > 0:
        print(f"[WARN] 有 {fallback_pc_count} 張 PC 找不到對應 DAPI，該些影像的 nucleus 仍使用 PC。")

    if dapi_nuc_img_files:
        segment(
            NUC_MODEL_PATH,
            dapi_nuc_img_files,
            seg_dir,
            "nuc",
            output_stems=dapi_output_stems,
            channels=(3, 3),
        )

    if fallback_pc_nuc_img_files:
        segment(
            NUC_MODEL_PATH,
            fallback_pc_nuc_img_files,
            seg_dir,
            "nuc",
            output_stems=fallback_pc_output_stems,
            channels=(0, 0),
        )
# ===============================
# Mask 頛詨??outlines 頧?
# ===============================
def mask2txt(npy_files: list[Path], output_dir: Path):
    """撠???segmentation npy 瑼?? outlines (txt)"""
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
    撠?cytoplasm ??nucleus segmentation ??npy 蝯?
    頛詨??mask/txt ?澆?
    """
    input_dir = Path(input_dir)
    seg_dir = output_dir(input_dir, "segment")
    mask2txt_out = output_dir(input_dir, "outline")
    cyto_files = [f for f in list_files(seg_dir, ".npy") if "_cyto_seg" in f.stem]
    nuc_files = [f for f in list_files(seg_dir, ".npy") if "_nuc_seg" in f.stem]
    mask2txt(cyto_files, mask2txt_out)
    mask2txt(nuc_files, mask2txt_out)


# ===============================
# Outlines ?蔥
# ===============================
def create_mask(outlines, shape):
    """撠?outlines 頧???mask ???"""
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
    """Match cytoplasm outlines to nucleus outlines and locate the source image."""
    # 撱箇? nucleus 瑼?頝臬?
    nuc_file = cyto_file.with_name(
        cyto_file.name.replace("_cyto_seg_cp_outlines.txt", "_nuc_seg_cp_outlines.txt")
    )
    if not nuc_file.exists():
        return None, None

    # 敺?cyto_file ?? dataset_name
    dataset_name = cyto_file.parent.name  # e.g., 2025-07-10-B8-P6...
    index_key = cyto_file.stem.replace("_cyto_seg_cp_outlines", "")

    # ?潭??頝臬?
    for ext in [".jpg", ".png", ".tif", ".tiff"]:
        img_candidate = Path("data/input") / dataset_name / "PC" / f"{index_key}{ext}"
        if img_candidate.exists():
            return nuc_file, img_candidate

    return None, None


def match_and_write(cyto_lines, nuc_lines, cyto_mask, nuc_mask, out_path: Path):
    """?? nucleus ??cytoplasm 銝西撓?箏?雿萇? outlines"""
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
    """Merge cytoplasm and nucleus outlines into paired merged outline files."""
    input_dir = Path(input_dir)
    outline_dir = output_dir(input_dir, "outline")
    txt_files = [f for f in list_files(outline_dir, ".txt") if "_cyto_seg" in f.stem]
    if not txt_files:
        print(f"[WARN] 找不到 cytoplasm outlines 於 {outline_dir}")
        return

    for cyto_file in trange(len(txt_files), desc="Merging outlines"):
        cyto_path = txt_files[cyto_file]
        nuc_path, img_path = find_image_and_nuc_file(cyto_path)
        if not nuc_path or not img_path:
            print(f"[WARN] 缺少 nucleus 或對應影像: {cyto_path.name}")
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

