from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .cell_anal_plot import plot_global_area_analysis
from .paired_overlay import PairedOverlayData, collect_paired_overlay_data
from .utils.io import output_dir
from .workbook_export import WorkbookProfile, workbook_output_path

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class PipelineResult:
    """GUI 執行 pipeline 後回傳的資料集與影像清單。"""

    data_folder: Path
    image_files: Sequence[Path]
    results_dir: Path | None = None
    area_scatter_plot: Path | None = None
    area_histogram_plot: Path | None = None
    workbook_path: Path | None = None
    paired_overlays: dict[str, PairedOverlayData] = field(default_factory=dict)


@dataclass
class OverlayPolygons:
    """GUI 疊圖所需的 nucleus 與 cytoplasm polygon 集合。

    Attributes:
        nuc_polygons: 依載入模式保留的 nucleus polygons。
        cyto_polygons: 依載入模式保留的 cytoplasm polygons。
        pair_record_indices: 每組完整配對在原始檔案中的 0-based record index；
            非配對載入模式維持空串列。
    """

    nuc_polygons: list[np.ndarray]
    cyto_polygons: list[np.ndarray]
    pair_record_indices: list[int] = field(default_factory=list)


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
    device: str = "gpu",
    nuc_source: str = "dapi",
    fluor_analy: bool = True,
    ki67: bool = True,
    ki67_backend: str = "pyimagej",
    feature_backend: str = "pyimagej",
    clean_temp: bool = True,
    width_um_per_px: float = 1.5896,
    height_um_per_px: float = 1.5876,
    xlsx_profile: WorkbookProfile = "biomedical",
    progress_callback: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """執行 GUI 使用的完整影像分析流程。

    Args:
        data_folder: 輸入資料集資料夾。
        device: 分割使用的運算裝置，``gpu`` 或 ``cpu``。只有 ``cpu`` 會強制關閉
            GPU；其餘值一律視為 ``gpu``，實際有無 GPU 由 Cellpose 自行判斷。
        nuc_source: 細胞核分割來源，可為 ``dapi`` 或 ``pc``。
        fluor_analy: 是否執行螢光分析。
        ki67: 是否執行 Ki67 判定。
        ki67_backend: Ki67 二值化後端。
        feature_backend: 特徵提取後端。
        clean_temp: 是否清理中間暫存檔。
        width_um_per_px: 影像寬度方向的像素比例。
        height_um_per_px: 影像高度方向的像素比例。
        xlsx_profile: cleaned XLSX 版本；GUI 預設為 ``biomedical``。
        progress_callback: 接收完成步驟、總步驟與訊息的回呼函式。

    Returns:
        Pipeline 輸出路徑與影像清單。
    """

    from .cell_anal import run_all
    from .img_prep import combined, mask2txt_all, segment_all

    data_folder = _resolve_data_folder(Path(data_folder))
    image_files = _list_display_image_files(data_folder)

    total_steps = 4  # 現在啟用 4 個步驟
    current_step = 0

    # Step 1: segmentation
    if progress_callback:
        progress_callback(current_step, total_steps, "執行 segmentation (cyto & nuc)")
    segment_all(data_folder, nuc_source=nuc_source, use_gpu=device != "cpu")
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
        xlsx_profile=xlsx_profile,
    )
    current_step += 1

    results_dir = output_dir(data_folder, "results")
    cleaned_csv = results_dir / f"{data_folder.name}_cleaned.csv"
    area_scatter_plot: Path | None = None
    area_histogram_plot: Path | None = None
    workbook_path: Path | None = None
    if cleaned_csv.exists():
        candidate_workbook = workbook_output_path(cleaned_csv, xlsx_profile)
        if candidate_workbook.exists():
            workbook_path = candidate_workbook
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
        workbook_path=workbook_path,
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


def _load_merged_outline_lines(merged_path: Path) -> list[str]:
    """安全讀取 merged outlines 的非空白資料列。

    Args:
        merged_path: ``*_merged_cp_outlines.txt`` 檔案路徑。

    Returns:
        已移除前後空白的非空白資料列；檔案無法讀取時回傳空串列。
    """
    try:
        with merged_path.open("r", encoding="utf-8", errors="ignore") as file:
            return [line.strip() for line in file if line.strip()]
    except OSError:
        return []


def _parse_merged_outline_polygon(line: str) -> np.ndarray | None:
    """將單筆 merged outline 解析為 polygon。

    Args:
        line: 以逗號分隔的 x、y 座標，或缺失標記 ``-1,-1``。

    Returns:
        合法座標的 ``int32`` polygon；缺失或格式無效時回傳 ``None``。
    """
    if line == "-1,-1":
        return None
    try:
        coords = list(map(int, line.split(",")))
    except ValueError:
        return None
    if len(coords) < 6 or len(coords) % 2:
        return None
    int32_info = np.iinfo(np.int32)
    if any(coord < int32_info.min or coord > int32_info.max for coord in coords):
        return None
    return np.asarray(coords, dtype=np.int32).reshape(-1, 2)


def load_merged_outlines(merged_path: Path) -> OverlayPolygons:
    """讀取 merged outlines 中所有有效的 nucleus 與 cytoplasm polygons。

    Args:
        merged_path: ``*_merged_cp_outlines.txt`` 檔案路徑。

    Returns:
        依原始列位置分類的有效 polygon；不完整配對中的有效 polygon 也會保留。
    """
    lines = _load_merged_outline_lines(merged_path)
    nucleus_polygons: list[np.ndarray] = []
    cytoplasm_polygons: list[np.ndarray] = []
    for index in range(0, len(lines) - 1, 2):
        nucleus = _parse_merged_outline_polygon(lines[index])
        cytoplasm = _parse_merged_outline_polygon(lines[index + 1])
        if nucleus is not None:
            nucleus_polygons.append(nucleus)
        if cytoplasm is not None:
            cytoplasm_polygons.append(cytoplasm)
    return OverlayPolygons(nucleus_polygons, cytoplasm_polygons)


def load_paired_merged_outlines(merged_path: Path) -> OverlayPolygons:
    """讀取 merged outlines 中完整配對的 nucleus 與 cytoplasm polygons。

    僅保留同一配對中兩者皆有效的 polygon，供 segmentation masks 不存在時的
    舊版 outline fallback 使用。

    Args:
        merged_path: ``*_merged_cp_outlines.txt`` 檔案路徑。

    Returns:
        完整配對的 nucleus 與 cytoplasm polygon。
    """
    lines = _load_merged_outline_lines(merged_path)
    nucleus_polygons: list[np.ndarray] = []
    cytoplasm_polygons: list[np.ndarray] = []
    pair_record_indices: list[int] = []
    for index in range(0, len(lines) - 1, 2):
        nucleus = _parse_merged_outline_polygon(lines[index])
        cytoplasm = _parse_merged_outline_polygon(lines[index + 1])
        if nucleus is None or cytoplasm is None:
            continue
        nucleus_polygons.append(nucleus)
        cytoplasm_polygons.append(cytoplasm)
        pair_record_indices.append(index // 2)
    return OverlayPolygons(
        nucleus_polygons,
        cytoplasm_polygons,
        pair_record_indices=pair_record_indices,
    )
