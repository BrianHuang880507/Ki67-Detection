from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .img_prep import segment_all, mask2txt_all, combined
from .cell_anal import run_all


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class PipelineResult:
    data_folder: Path
    image_files: Sequence[Path]


@dataclass
class OverlayPolygons:
    nuc_polygons: list[np.ndarray]
    cyto_polygons: list[np.ndarray]


def _resolve_data_folder(raw_data_folder: Path) -> Path:
    candidates = []
    if raw_data_folder.is_absolute():
        candidates.append(raw_data_folder)
    else:
        base_dir = Path("data/input")
        candidates.append(base_dir / raw_data_folder)
        candidates.append(raw_data_folder)

    search_targets = []
    seen = set()
    for candidate in candidates:
        absolute = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        key = str(absolute.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            search_targets.append(absolute)

    for candidate in search_targets:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "找不到資料夾，請確認路徑是否存在: " + ", ".join(str(c) for c in search_targets)
    )


def run_pipeline(
    data_folder: Path,
    fluor_analy: bool = True,
    ki67: bool = True,
    clean_temp: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """高階 pipeline 入口，給 GUI 或其他程式呼叫使用。

    依序執行：
    1. segmentation (cyto & nuc)
    2. segmentation npy -> outlines txt
    3. 合併 nucleus 與 cytoplasm outlines
    4. 幾何參數與螢光/陽性分析
    """

    data_folder = _resolve_data_folder(Path(data_folder))

    total_steps = 4  # 現在啟用 4 個步驟
    current_step = 0

    # Step 1: segmentation
    if progress_callback:
        progress_callback(current_step, total_steps, "執行 segmentation (cyto & nuc)")
    segment_all(data_folder)
    current_step += 1

    # Step 2: mask -> outlines
    if progress_callback:
        progress_callback(
            current_step, total_steps, "將 segmentation npy 轉成 outlines txt"
        )
    mask2txt_all(data_folder)
    current_step += 1

    # Step 3: combine outlines
    if progress_callback:
        progress_callback(
            current_step, total_steps, "合併 nucleus 與 cytoplasm outlines"
        )
    combined(data_folder)
    current_step += 1

    # Step 4: geometry & intensity analysis
    if progress_callback:
        progress_callback(current_step, total_steps, "幾何參數與螢光/陽性分析")
    run_all(data_folder, fluor_analy=fluor_analy, ki67=ki67, clean_temp=clean_temp)
    current_step += 1

    if progress_callback:
        progress_callback(current_step, total_steps, "Pipeline 完成")

    # 依照實際資料結構決定顯示的原始影像：
    # 優先使用 data_folder/PC/ 底下的圖；若無 PC 資料夾，再退回 data_folder 本身。
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files: list[Path] = []

    pc_dir = data_folder / "PC"
    search_dir = pc_dir if pc_dir.exists() and pc_dir.is_dir() else data_folder

    for p in sorted(search_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            image_files.append(p)

    return PipelineResult(data_folder=data_folder, image_files=image_files)


def find_merged_outline_for_image(image_path: Path) -> Path | None:
    """根據原始影像路徑，推導對應的 merged outlines 檔案路徑。

    目前約定：
    - 影像：data/input/<dataset_name>/PC/<index>.<ext>
    - merged：data/output/outline/<dataset_name>/<index>_merged_cp_outlines.txt
    """
    image_path = image_path.resolve()
    try:
        dataset_dir = image_path.parents[1]  # .../<dataset_name>/PC/<file>
        dataset_name = dataset_dir.name
    except IndexError:
        return None

    index_key = image_path.stem
    merged_path = (
        Path("data")
        / "output"
        / "outline"
        / dataset_name
        / f"{index_key}_merged_cp_outlines.txt"
    )
    return merged_path if merged_path.exists() else None


def load_merged_outlines(merged_path: Path) -> OverlayPolygons:
    """讀取 merged_cp_outlines.txt 並拆成 nucleus / cytoplasm polygons。

    檔案格式：兩行為一組，偶數行 (0-based) 為 nucleus，奇數行為 cytoplasm。
    缺少一側時會以 "-1,-1" 佔位，這裡會直接略過該行。
    """
    nuc_polys: list[np.ndarray] = []
    cyto_polys: list[np.ndarray] = []

    with merged_path.open("r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    def _line_to_poly(line: str) -> Optional[np.ndarray]:
        if line == "-1,-1":
            return None
        coords = list(map(int, line.split(",")))
        if len(coords) < 6:  # 至少 3 個點
            return None
        arr = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        return arr

    for idx, line in enumerate(lines):
        poly = _line_to_poly(line)
        if poly is None:
            continue
        if idx % 2 == 0:
            nuc_polys.append(poly)
        else:
            cyto_polys.append(poly)

    return OverlayPolygons(nuc_polygons=nuc_polys, cyto_polygons=cyto_polys)
