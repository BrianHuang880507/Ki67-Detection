from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .img_prep import segment_all, mask2txt_all, combined
from .cell_anal import run_all
from .cell_anal_plot import plot_global_area_analysis
from .paired_overlay import PairedOverlayData, collect_paired_overlay_data
from .utils.io import output_dir


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class PipelineResult:
    """GUI 執行 pipeline 後回傳的資料集與影像清單。"""

    data_folder: Path
    image_files: Sequence[Path]
    results_dir: Path | None = None
    area_scatter_plot: Path | None = None
    area_histogram_plot: Path | None = None
    paired_overlays: dict[str, PairedOverlayData] = field(default_factory=dict)


@dataclass
class OverlayPolygons:
    """GUI 疊圖所需的 nucleus 與 cytoplasm polygon 集合。"""

    nuc_polygons: list[np.ndarray]
    cyto_polygons: list[np.ndarray]


def _resolve_data_folder(raw_data_folder: Path) -> Path:
    """解析 GUI 輸入的資料夾名稱或路徑。

    Args:
        raw_data_folder (Path): 使用者輸入的資料夾名稱、相對路徑或絕對路徑。

    Returns:
        Path: 實際存在的資料集資料夾。

    Raises:
        FileNotFoundError: 找不到任何可用資料夾時拋出。
    """
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


def _list_display_image_files(data_folder: Path) -> list[Path]:
    """列出 GUI 應顯示的 PC 或資料集根目錄影像。"""
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    pc_dir = data_folder / "PC"
    search_dir = pc_dir if pc_dir.exists() and pc_dir.is_dir() else data_folder
    return [
        path
        for path in sorted(search_dir.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    ]


def run_pipeline(
    data_folder: Path,
    nuc_source: str = "dapi",
    fluor_analy: bool = True,
    ki67: bool = True,
    ki67_backend: str = "pyimagej",
    feature_backend: str = "pyimagej",
    clean_temp: bool = True,
    width_um_per_px: float = 1.5896,
    height_um_per_px: float = 1.5876,
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
    image_files = _list_display_image_files(data_folder)

    total_steps = 4  # 現在啟用 4 個步驟
    current_step = 0

    # Step 1: segmentation
    if progress_callback:
        progress_callback(current_step, total_steps, "執行 segmentation (cyto & nuc)")
    segment_all(data_folder, nuc_source=nuc_source)
    paired_overlays = collect_paired_overlay_data(
        image_files,
        output_dir(data_folder, "segment"),
    )
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
    run_all(
        data_folder,
        fluor_analy=fluor_analy,
        ki67=ki67,
        ki67_backend=ki67_backend,
        feature_backend=feature_backend,
        clean_temp=clean_temp,
    )
    current_step += 1

    results_dir = output_dir(data_folder, "results")
    cleaned_csv = results_dir / f"{data_folder.name}_cleaned.csv"
    area_scatter_plot: Path | None = None
    area_histogram_plot: Path | None = None
    if cleaned_csv.exists():
        area_scatter_plot, area_histogram_plot = plot_global_area_analysis(
            cleaned_csv,
            results_dir,
            thres=6.0,
            width_um_per_px=width_um_per_px,
            height_um_per_px=height_um_per_px,
        )

    if progress_callback:
        progress_callback(current_step, total_steps, "Pipeline 完成")

    return PipelineResult(
        data_folder=data_folder,
        image_files=image_files,
        results_dir=results_dir,
        area_scatter_plot=area_scatter_plot,
        area_histogram_plot=area_histogram_plot,
        paired_overlays=paired_overlays,
    )


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
        """將 merged outline 單行座標轉為 polygon 陣列。"""
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


def load_paired_merged_outlines(merged_path: Path) -> OverlayPolygons:
    """讀取 merged outlines，只保留核與質兩側都存在的配對。

    這是原始 segmentation masks 不存在時的顯示 fallback。正常 pipeline
    會優先使用記憶體中的 SegUI 中心點配對結果。

    Args:
        merged_path: ``*_merged_cp_outlines.txt`` 路徑。

    Returns:
        核與質索引彼此對齊的 polygon 集合。
    """
    with merged_path.open("r", encoding="utf-8", errors="ignore") as file:
        lines = [line.strip() for line in file if line.strip()]

    def parse_polygon(line: str) -> np.ndarray | None:
        """解析單行 outline；缺值或格式錯誤時回傳 ``None``。"""
        if line == "-1,-1":
            return None
        try:
            coords = list(map(int, line.split(",")))
        except ValueError:
            return None
        if len(coords) < 6 or len(coords) % 2:
            return None
        return np.asarray(coords, dtype=np.int32).reshape(-1, 2)

    nucleus_polygons: list[np.ndarray] = []
    cytoplasm_polygons: list[np.ndarray] = []
    for index in range(0, len(lines) - 1, 2):
        nucleus = parse_polygon(lines[index])
        cytoplasm = parse_polygon(lines[index + 1])
        if nucleus is None or cytoplasm is None:
            continue
        nucleus_polygons.append(nucleus)
        cytoplasm_polygons.append(cytoplasm)

    return OverlayPolygons(
        nuc_polygons=nucleus_polygons,
        cyto_polygons=cytoplasm_polygons,
    )
