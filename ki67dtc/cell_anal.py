import os
import sys
import tempfile
from contextlib import contextmanager
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange
from typing import Any, Union
from skimage.draw import polygon, polygon2mask
from skimage.io import imread
from shapely.affinity import scale
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from skimage import io as skio
from shutil import copyfile
import cv2
import tifffile


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

_PYIMAGEJ = None


@contextmanager
def _suppress_console_output():
    """Temporarily silence Python/native stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_fd_backup = None
        stderr_fd_backup = None
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                stdout_fd_backup = os.dup(1)
                os.dup2(devnull.fileno(), 1)
            except OSError:
                stdout_fd_backup = None
            try:
                stderr_fd_backup = os.dup(2)
                os.dup2(devnull.fileno(), 2)
            except OSError:
                stderr_fd_backup = None
            yield
        finally:
            if stdout_fd_backup is not None:
                os.dup2(stdout_fd_backup, 1)
                os.close(stdout_fd_backup)
            if stderr_fd_backup is not None:
                os.dup2(stderr_fd_backup, 2)
                os.close(stderr_fd_backup)
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _run_macro_quiet(ij: Any, macro: str, args: dict[str, Any]) -> Any:
    """Run an IJ1 macro while suppressing noisy terminal output."""
    with _suppress_console_output():
        return ij.py.run_macro(macro, args=args)

_KI67_IMAGEJ_MACRO = r"""
#@ String input_path
#@ String binary_path
#@ String particle_options

run("Close All");
run("Clear Results");
setBatchMode(true);

open(input_path);
orig_title = getTitle();

run("8-bit");
setAutoThreshold("Default dark no-reset");
setAutoThreshold("Otsu dark no-reset");
setThreshold(13, 255);
setOption("BlackBackground", true);
run("Convert to Mask");
run("Watershed");
run("Duplicate...", "title=Ki67BinaryFallback");
run("Analyze Particles...", particle_options);

mask_title = "Mask of " + orig_title;
if (isOpen(mask_title)) {
    selectWindow(mask_title);
    saveAs("PNG", binary_path);
    close();
} else if (isOpen("Mask")) {
    selectWindow("Mask");
    saveAs("PNG", binary_path);
    close();
} else if (isOpen("Ki67BinaryFallback")) {
    selectWindow("Ki67BinaryFallback");
    saveAs("PNG", binary_path);
}

if (isOpen("Ki67BinaryFallback")) {
    selectWindow("Ki67BinaryFallback");
    close();
}
if (isOpen(orig_title)) {
    selectWindow(orig_title);
    close();
}
run("Clear Results");
setBatchMode(false);
close("*");
"""

_PREPROCESS_MACRO = r"""
#@ String input_path
#@ String output_path
#@ double rolling_ball_radius

open(input_path);
run("16-bit");
run("Subtract Background...", "rolling=" + rolling_ball_radius);
saveAs("Tiff", output_path);
close();
"""

_MEASURE_ROI_MACRO = r"""
#@ String signal_path
#@ String mask_path
#@ String output_path

run("Close All");
run("Clear Results");
setBatchMode(true);

open(signal_path);
rename("signal");
open(mask_path);
rename("mask");

selectWindow("mask");
run("8-bit");
setThreshold(1, 255);
run("Convert to Mask");
run("Create Selection");

stype = selectionType();
if (stype < 0) {
    File.saveString("valid=0\n", output_path);
    setBatchMode(false);
    close("*");
    exit();
}
getSelectionCoordinates(xpoints, ypoints);

selectWindow("signal");
makeSelection("polygon", xpoints, ypoints);
run("Set Measurements...", "area mean standard min integrated perimeter shape feret's fit decimal=6");
run("Measure");

area = getResult("Area", 0);
mean = getResult("Mean", 0);
std = getResult("StdDev", 0);
minv = getResult("Min", 0);
maxv = getResult("Max", 0);
intden = getResult("IntDen", 0);
rawintden = getResult("RawIntDen", 0);
perim = getResult("Perim.", 0);
feret = getResult("Feret", 0);
minferet = getResult("MinFeret", 0);
major = getResult("Major", 0);
minor = getResult("Minor", 0);
ar = getResult("AR", 0);
roundv = getResult("Round", 0);
circ = getResult("Circ.", 0);

selectWindow("signal");
makeSelection("polygon", xpoints, ypoints);
run("Convex Hull");
run("Clear Results");
run("Set Measurements...", "perimeter decimal=6");
run("Measure");
conv_perim = getResult("Perim.", 0);

result_txt = "";
result_txt = result_txt + "valid=1\n";
result_txt = result_txt + "area=" + area + "\n";
result_txt = result_txt + "mean=" + mean + "\n";
result_txt = result_txt + "std=" + std + "\n";
result_txt = result_txt + "min=" + minv + "\n";
result_txt = result_txt + "max=" + maxv + "\n";
result_txt = result_txt + "intden=" + intden + "\n";
result_txt = result_txt + "raw_intden=" + rawintden + "\n";
result_txt = result_txt + "perimeter=" + perim + "\n";
result_txt = result_txt + "feret=" + feret + "\n";
result_txt = result_txt + "minferet=" + minferet + "\n";
result_txt = result_txt + "major=" + major + "\n";
result_txt = result_txt + "minor=" + minor + "\n";
result_txt = result_txt + "ar=" + ar + "\n";
result_txt = result_txt + "round=" + roundv + "\n";
result_txt = result_txt + "circ=" + circ + "\n";
result_txt = result_txt + "convex_perimeter=" + conv_perim + "\n";
File.saveString(result_txt, output_path);

run("Clear Results");
setBatchMode(false);
close("*");
"""


def _get_pyimagej():
    """初始化並回傳 PyImageJ 物件（含快取）。

    Args:
        無。

    Returns:
        Any: 可執行 IJ1 macro 的 PyImageJ 物件。

    Raises:
        RuntimeError: 未安裝 PyImageJ 相關套件，或 legacy 模式未啟用時拋出。
    """
    global _PYIMAGEJ
    if _PYIMAGEJ is not None:
        return _PYIMAGEJ

    try:
        import imagej
        import scyjava
    except ImportError as exc:
        raise RuntimeError(
            "PyImageJ is unavailable. Install pyimagej/scyjava/jpype1 first."
        ) from exc

    scyjava.config.add_option("-Dscijava.log.level=error")
    fiji_app_path = os.environ.get("FIJI_APP_PATH", "").strip()
    if fiji_app_path:
        ij = imagej.init(fiji_app_path, mode="headless", add_legacy=True)
    else:
        ij = imagej.init("sc.fiji:fiji", mode="headless", add_legacy=True)

    if not (ij.legacy and ij.legacy.isActive()):
        raise RuntimeError("ImageJ legacy mode is inactive, cannot run IJ1 macro.")

    _PYIMAGEJ = ij
    return _PYIMAGEJ


def _preprocess_signal_with_imagej(
    ij: Any, signal_2d: np.ndarray, rolling_ball_radius: float = 50.0
) -> np.ndarray:
    """使用 PyImageJ 對單通道影像做背景扣除前處理。

    Args:
        ij (Any): PyImageJ 物件。
        signal_2d (np.ndarray): 單通道 2D 影像。
        rolling_ball_radius (float, optional): Rolling-ball 半徑。預設為 `50.0`。

    Returns:
        np.ndarray: 背景扣除後的 `float32` 2D 影像。

    Raises:
        ValueError: 輸入影像不是 2D 時拋出。
        RuntimeError: ImageJ 前處理失敗時拋出。
    """
    if signal_2d.ndim != 2:
        raise ValueError(f"signal_2d 必須為 2D，實際為 {signal_2d.shape}")

    with tempfile.TemporaryDirectory(prefix="pyimagej_bgsub_") as tmp:
        tmp_dir = Path(tmp)
        in_path = tmp_dir / "signal_input.tif"
        out_path = tmp_dir / "signal_bgsub.tif"
        tifffile.imwrite(in_path, signal_2d.astype(np.float32))
        _run_macro_quiet(
            ij,
            _PREPROCESS_MACRO,
            args={
                "input_path": str(in_path),
                "output_path": str(out_path),
                "rolling_ball_radius": float(rolling_ball_radius),
            },
        )
        if not out_path.exists():
            raise RuntimeError(f"ImageJ 前處理失敗，找不到輸出檔：{out_path}")
        out = tifffile.imread(out_path).astype(np.float32)
        out = np.squeeze(out)
        if out.ndim != 2:
            raise ValueError(f"前處理結果維度錯誤：{out.shape}")
        return out


def _empty_imagej_measurements() -> dict[str, float]:
    """建立 ImageJ 量測結果的預設字典。

    Args:
        無。

    Returns:
        dict[str, float]: 各欄位以 `np.nan` 初始化的量測字典。
    """
    return {
        "area": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "min": np.nan,
        "max": np.nan,
        "intden": np.nan,
        "raw_intden": np.nan,
        "perimeter": np.nan,
        "feret": np.nan,
        "minferet": np.nan,
        "major": np.nan,
        "minor": np.nan,
        "ar": np.nan,
        "round": np.nan,
        "circ": np.nan,
        "convex_perimeter": np.nan,
    }


def _read_kv_measurements(path: Path) -> dict[str, float]:
    """讀取 ImageJ macro 輸出的 key-value 量測檔。

    Args:
        path (Path): 量測輸出文字檔路徑。

    Returns:
        dict[str, float]: 量測結果字典。
    """
    out = _empty_imagej_measurements()
    if not path.exists():
        return out

    parsed: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()

    if parsed.get("valid") != "1":
        return out

    for key in out:
        if key not in parsed:
            continue
        try:
            out[key] = float(parsed[key])
        except Exception:
            out[key] = np.nan

    if (
        np.isnan(out["raw_intden"])
        and not np.isnan(out["area"])
        and not np.isnan(out["mean"])
    ):
        out["raw_intden"] = float(out["area"] * out["mean"])
    return out


def _measure_roi_with_imagej(
    ij: Any,
    signal_path: Path,
    roi_mask: np.ndarray | None,
    temp_dir: Path,
    slot: str,
) -> dict[str, float]:
    """使用 PyImageJ 量測單一 ROI 遮罩。

    Args:
        ij (Any): PyImageJ 物件。
        signal_path (Path): 被量測影像路徑。
        roi_mask (np.ndarray | None): ROI 二值遮罩。
        temp_dir (Path): 暫存目錄。
        slot (str): 暫存檔命名識別字。

    Returns:
        dict[str, float]: ROI 量測結果字典。
    """
    if roi_mask is None or not np.any(roi_mask):
        return _empty_imagej_measurements()

    mask_path = temp_dir / f"{slot}_mask.tif"
    out_path = temp_dir / f"{slot}_measure.txt"
    tifffile.imwrite(mask_path, (roi_mask.astype(np.uint8) * 255))
    _run_macro_quiet(
        ij,
        _MEASURE_ROI_MACRO,
        args={
            "signal_path": str(signal_path),
            "mask_path": str(mask_path),
            "output_path": str(out_path),
        },
    )
    return _read_kv_measurements(out_path)


def _geometry_from_imagej_measurements(m: dict[str, float]) -> dict[str, float]:
    """將 ImageJ 量測結果轉為主流程幾何欄位。

    Args:
        m (dict[str, float]): ImageJ 量測結果字典。

    Returns:
        dict[str, float]: 主流程幾何欄位對應值。
    """
    area = m["area"]
    perimeter = m["perimeter"]
    convex_perimeter = m["convex_perimeter"]
    major = m["major"]
    minor = m["minor"]

    circular_diameter = (
        2 * np.sqrt(area / np.pi) if not np.isnan(area) and area > 0 else np.nan
    )
    sphericity = m["circ"]
    if np.isnan(sphericity):
        sphericity = (
            (4 * np.pi * area) / (perimeter**2)
            if not np.isnan(area) and not np.isnan(perimeter) and perimeter > 0
            else np.nan
        )

    roughness = (
        (convex_perimeter / perimeter)
        if not np.isnan(convex_perimeter) and not np.isnan(perimeter) and perimeter > 0
        else np.nan
    )

    aspect_ratio = m["ar"]
    roundness = m["round"]
    circularity = np.nan

    if (
        np.isnan(aspect_ratio)
        and not np.isnan(major)
        and not np.isnan(minor)
        and minor > 0
    ):
        aspect_ratio = major / minor

    if np.isnan(roundness) and not np.isnan(area) and not np.isnan(major) and major > 0:
        roundness = (4 * area) / (np.pi * major**2)

    if not np.isnan(area) and not np.isnan(perimeter) and perimeter > 0:
        circularity = (2 * np.sqrt(np.pi * area)) / (perimeter**2)

    return {
        "area": area,
        "perimeter": perimeter,
        "convex_perimeter": convex_perimeter,
        "circular_diameter": circular_diameter,
        "feret_length": m["feret"],
        "feret_width": m["minferet"],
        "aspect_ratio": aspect_ratio,
        "roundness": roundness,
        "circularity": circularity,
        "sphericity": sphericity,
        "roughness": roughness,
    }


def _parse_outline_pairs(
    outlines_txt: Union[str, Path],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """解析 merged outline 檔為成對的核/質座標。

    Args:
        outlines_txt (Union[str, Path]): merged outline 文字檔路徑。

    Returns:
        list[tuple[int, np.ndarray, np.ndarray]]:
        每個元素為 `(roi_id, nucleus_xy, cytoplasm_xy)`。
    """
    with open(outlines_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    num_pairs = len(lines) // 2
    for i in range(num_pairs):
        try:
            nuc = np.array(
                list(map(int, lines[2 * i].split(","))), dtype=np.int32
            ).reshape(-1, 2)
            cyto = np.array(
                list(map(int, lines[2 * i + 1].split(","))), dtype=np.int32
            ).reshape(-1, 2)
        except Exception:
            continue
        if nuc.shape[0] < 3 or cyto.shape[0] < 3:
            continue
        pairs.append((i + 1, nuc, cyto))
    return pairs


def _polygon_to_mask(points_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """將 polygon 座標轉為二值遮罩。

    Args:
        points_xy (np.ndarray): `(N, 2)` 座標，欄位順序為 `(x, y)`。
        shape (tuple[int, int]): 遮罩尺寸 `(height, width)`。

    Returns:
        np.ndarray: `bool` 型態遮罩。
    """
    rr, cc = polygon(points_xy[:, 1], points_xy[:, 0], shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


# ===============================
# Geometry Utilities
# ===============================


def extract_polygons(geom):
    """將 Shapely 幾何物件展平成 Polygon 清單。

    Args:
        geom: `Polygon`、`MultiPolygon` 或 `GeometryCollection`。

    Returns:
        list[Polygon]: 可直接迭代處理的 Polygon 清單。
    """
    if isinstance(geom, Polygon):
        return [geom]
    elif isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    else:
        return []


def param_anal(img_path, outlines_txt, output_csv):
    """以 PyImageJ 量測核與細胞質幾何參數，並輸出主流程相容 CSV。

    Args:
        img_path (Union[str, Path]): PC 影像路徑。
        outlines_txt (Union[str, Path]): `_merged_cp_outlines.txt` 路徑。
        output_csv (Union[str, Path]): 參數輸出 CSV 路徑。

    Returns:
        None: 此函式僅負責輸出檔案。
    """
    img_path = Path(img_path)
    output_csv = Path(output_csv)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到影像：{img_path}")

    signal = imread(img_path, as_gray=True).astype(np.float32)
    shape = signal.shape
    ij = _get_pyimagej()
    pairs = _parse_outline_pairs(outlines_txt)
    rows: list[list[float | str]] = []

    with tempfile.TemporaryDirectory(prefix="pyimagej_param_") as tmp:
        tmp_dir = Path(tmp)
        signal_path = tmp_dir / f"{img_path.stem}_pc_signal.tif"
        tifffile.imwrite(signal_path, signal)

        for roi_id, nuc_xy, cyto_xy in pairs:
            nuc_mask = _polygon_to_mask(nuc_xy, shape)
            cyto_raw_mask = _polygon_to_mask(cyto_xy, shape)
            cyto_mask = np.logical_and(cyto_raw_mask, np.logical_not(nuc_mask))

            nuc_m = _measure_roi_with_imagej(
                ij, signal_path, nuc_mask, tmp_dir, f"roi_{roi_id}_nuc"
            )
            cyto_m = _measure_roi_with_imagej(
                ij, signal_path, cyto_mask, tmp_dir, f"roi_{roi_id}_cyto"
            )

            nuc_g = _geometry_from_imagej_measurements(nuc_m)
            cyto_g = _geometry_from_imagej_measurements(cyto_m)

            karyoplasmic_ratio = np.nan
            if (
                not np.isnan(cyto_g["area"])
                and not np.isnan(nuc_g["area"])
                and cyto_g["area"] > 0
            ):
                # Use nucleus area / cytoplasm area so the value matches the intended
                # biological interpretation of a nucleocytoplasmic ratio.
                karyoplasmic_ratio = float(nuc_g["area"] / cyto_g["area"])

            rows.append(
                [
                    f"{img_path.stem}_{roi_id}_nuc",
                    nuc_g["area"],
                    nuc_g["perimeter"],
                    nuc_g["convex_perimeter"],
                    nuc_g["circular_diameter"],
                    nuc_g["feret_length"],
                    nuc_g["feret_width"],
                    nuc_g["aspect_ratio"],
                    nuc_g["roundness"],
                    nuc_g["circularity"],
                    nuc_g["sphericity"],
                    nuc_g["roughness"],
                    np.nan,
                ]
            )
            rows.append(
                [
                    f"{img_path.stem}_{roi_id}_cyto",
                    cyto_g["area"],
                    cyto_g["perimeter"],
                    cyto_g["convex_perimeter"],
                    cyto_g["circular_diameter"],
                    cyto_g["feret_length"],
                    cyto_g["feret_width"],
                    cyto_g["aspect_ratio"],
                    cyto_g["roundness"],
                    cyto_g["circularity"],
                    cyto_g["sphericity"],
                    cyto_g["roughness"],
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
    print(f"[INFO] 已輸出參數分析結果：{output_csv}")


def flour_anal(
    img_path, outlines_txt, output_csv, max_expand_steps=20, expand_factor=0.5
):
    """以 PyImageJ 量測環狀 ROI 的螢光強度。

    Args:
        img_path (Union[str, Path]): 螢光影像路徑（如 DF/LT）。
        outlines_txt (Union[str, Path]): `_merged_cp_outlines.txt` 路徑。
        output_csv (Union[str, Path]): 螢光輸出 CSV 路徑。
        max_expand_steps (int, optional): 向外擴張圈層數。預設為 `20`。
        expand_factor (float, optional): 每一步核輪廓擴張比例。預設為 `0.5`。

    Returns:
        None: 此函式僅負責輸出檔案。
    """
    img_path = Path(img_path)
    output_csv = Path(output_csv)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到影像：{img_path}")

    signal = imread(img_path, as_gray=True).astype(np.float32)
    h, w = signal.shape
    ij = _get_pyimagej()
    signal_bgsub = _preprocess_signal_with_imagej(ij, signal, rolling_ball_radius=50.0)
    pairs = _parse_outline_pairs(outlines_txt)
    rows: list[list[float | str]] = []

    with tempfile.TemporaryDirectory(prefix="pyimagej_fluor_") as tmp:
        tmp_dir = Path(tmp)
        signal_path = tmp_dir / f"{img_path.stem}_fluor_signal.tif"
        tifffile.imwrite(signal_path, signal_bgsub.astype(np.float32))

        for roi_id, nuc_xy, cyto_xy in pairs:
            nuc_poly = Polygon(nuc_xy).buffer(0)
            cyto_poly = Polygon(cyto_xy).buffer(0)
            if nuc_poly.is_empty or cyto_poly.is_empty:
                continue
            if not nuc_poly.is_valid or not cyto_poly.is_valid:
                continue

            prev_poly = nuc_poly
            for ring_id in range(max_expand_steps):
                factor = 1 + expand_factor * (ring_id + 1)
                scaled_nuc = scale(
                    nuc_poly, xfact=factor, yfact=factor, origin="center"
                )
                ring_poly = scaled_nuc.symmetric_difference(prev_poly)
                prev_poly = scaled_nuc
                if ring_poly.is_empty:
                    continue

                target = ring_poly.intersection(cyto_poly)
                if target.is_empty:
                    continue

                ring_mask = np.zeros((h, w), dtype=bool)
                for poly in extract_polygons(target):
                    coords = np.array(poly.exterior.coords, dtype=np.float32)[:, :2]
                    if coords.shape[0] < 3:
                        continue
                    ring_mask |= _polygon_to_mask(coords.astype(np.int32), (h, w))

                if not np.any(ring_mask):
                    continue

                measured = _measure_roi_with_imagej(
                    ij,
                    signal_path,
                    ring_mask,
                    tmp_dir,
                    f"roi_{roi_id}_ring_{ring_id}",
                )
                rows.append(
                    [
                        f"{img_path.name}:NewCell-{roi_id}-and{ring_id}",
                        measured["intden"],
                        measured["raw_intden"],
                    ]
                )

    df = pd.DataFrame(rows, columns=["Label", "IntDen", "RawIntDen"])
    df.to_csv(output_csv, index=False)
    print(f"[INFO] 已輸出螢光分析結果：{output_csv}")


def ido_anal(img_path, outlines_txt, output_csv):
    """依使用者定義輸出 IDO 細胞層級量測欄位。"""
    img_path = Path(img_path)
    output_csv = Path(output_csv)
    if not img_path.exists():
        raise FileNotFoundError(f"?曆??啣蔣??{img_path}")

    signal = imread(img_path, as_gray=True).astype(np.float32)
    shape = signal.shape
    pairs = _parse_outline_pairs(outlines_txt)

    cell_mask_union = np.zeros(shape, dtype=bool)
    roi_masks: list[tuple[int, np.ndarray]] = []
    for roi_id, nuc_xy, cyto_xy in pairs:
        nuc_mask = _polygon_to_mask(nuc_xy, shape)
        cyto_raw_mask = _polygon_to_mask(cyto_xy, shape)
        cyto_mask = np.logical_and(cyto_raw_mask, np.logical_not(nuc_mask))

        cell_mask_union |= cyto_raw_mask
        roi_masks.append((roi_id, cyto_mask))

    background_pixels = signal[np.logical_not(cell_mask_union)]
    # The requested field name keeps "Mean", but the value is the
    # non-cell pixel median defined by the user.
    ido_background_mean = (
        float(np.median(background_pixels)) if background_pixels.size > 0 else np.nan
    )

    rows: list[list[float | str]] = []
    for roi_id, cyto_mask in roi_masks:
        area_cyto = float(np.count_nonzero(cyto_mask))
        ido_mean_intensity = (
            float(np.mean(signal[cyto_mask])) if area_cyto > 0 else np.nan
        )

        if np.isnan(ido_mean_intensity) or np.isnan(ido_background_mean):
            ido_mean_intensity_bgsub = np.nan
            ido_intden = np.nan
            ido_intden_bgsub = np.nan
        else:
            ido_mean_intensity_bgsub = float(
                ido_mean_intensity - ido_background_mean
            )
            ido_intden = ido_mean_intensity_bgsub
            ido_intden_bgsub = float(area_cyto * ido_mean_intensity_bgsub)

        rows.append(
            [
                f"{img_path.stem}_{roi_id}",
                ido_background_mean,
                ido_mean_intensity,
                ido_mean_intensity_bgsub,
                ido_intden,
                ido_intden_bgsub,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Cell_ID",
            "IDO_BackgroundMean",
            "IDO_MeanIntensity",
            "IDO_MeanIntensity_BgSub",
            "IDO_IntDen",
            "IDO_IntDen_BgSub",
        ],
    )
    df.to_csv(output_csv, index=False)
    print(f"[INFO] IDO 量測完成：{output_csv}")


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """移除小於門檻面積的連通元件。

    Args:
        mask (np.ndarray): 二值影像。
        min_area (int): 保留元件的最小像素面積。

    Returns:
        np.ndarray: 僅保留面積大於等於 `min_area` 的二值影像。
    """
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def ki67_binarize(
    img_path: Union[str, Path],
    channel: str = "auto",  # "auto"|"r"|"g"|"b"
    gauss_sigma: float = 1.5,
    clahe_clip: float = 2.0,
    otsu_scale: float = 0.85,
    min_obj_area: int = 50,
    open_kernel: int = 3,
    backend: str = "pyimagej",  # "pyimagej"|"opencv"
) -> Path:
    """執行 Ki67 二值化。

    Args:
        img_path (Union[str, Path]): Ki67 影像路徑。
        channel (str, optional): OpenCV 模式下使用的色彩通道。
        gauss_sigma (float, optional): 高斯平滑 sigma。
        clahe_clip (float, optional): CLAHE clip limit?
        otsu_scale (float, optional): Otsu 門檻縮放比例。
        min_obj_area (int, optional): 最小物件面積。
        open_kernel (int, optional): 開運算 kernel 尺寸。
        backend (str, optional): `pyimagej` 或 `opencv`。

    Returns:
        Path: 二值化輸出影像路徑。

    Raises:
        FileNotFoundError: 找不到輸入影像時拋出。
        ValueError: backend 參數不支援時拋出。
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"[ERROR] 找不到圖片: {img_path}")

    binary_dir = output_dir(img_path.parent.parent, "binary")
    out_path = binary_dir / f"{img_path.stem}_binary.png"

    backend = backend.lower().strip()
    if backend not in {"pyimagej", "opencv"}:
        raise ValueError(f"Unsupported ki67 backend: {backend}")

    if backend == "pyimagej":
        try:
            _ki67_binarize_pyimagej(
                img_path=img_path,
                out_path=out_path,
                min_obj_area=min_obj_area,
            )
            print(f"[INFO] 已輸出 PyImageJ Ki67 二值圖: {out_path}")
            return out_path
        except Exception as exc:
            print(f"[WARN] PyImageJ Ki67 二值化失敗，改用 OpenCV。原因: {exc}")

    _ki67_binarize_opencv(
        img_path=img_path,
        out_path=out_path,
        channel=channel,
        gauss_sigma=gauss_sigma,
        clahe_clip=clahe_clip,
        otsu_scale=otsu_scale,
        min_obj_area=min_obj_area,
        open_kernel=open_kernel,
    )
    print(f"[INFO] 已輸出 OpenCV Ki67 二值圖: {out_path}")
    return out_path


def _ki67_binarize_pyimagej(
    img_path: Path,
    out_path: Path,
    min_obj_area: int = 50,
) -> None:
    """使用 PyImageJ macro 執行 Ki67 二值化。

    Args:
        img_path (Path): Ki67 影像路徑。
        out_path (Path): 二值化輸出路徑。
        min_obj_area (int, optional): Analyze Particles 最小面積門檻。

    Returns:
        None: 此函式僅負責輸出檔案。
    """

    ij = _get_pyimagej()
    if out_path.exists():
        out_path.unlink()

    particle_options = f"size={int(max(0, min_obj_area))}-Infinity show=Masks clear"
    args = {
        "input_path": str(img_path),
        "binary_path": str(out_path),
        "particle_options": particle_options,
    }
    _run_macro_quiet(ij, _KI67_IMAGEJ_MACRO, args=args)

    binary = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise RuntimeError(f"ImageJ macro did not produce a readable file: {out_path}")

    _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(out_path), binary)


def _ki67_binarize_opencv(
    img_path: Path,
    out_path: Path,
    channel: str = "auto",
    gauss_sigma: float = 1.5,
    clahe_clip: float = 2.0,
    otsu_scale: float = 0.85,
    min_obj_area: int = 50,
    open_kernel: int = 3,
) -> None:
    """使用 OpenCV 執行 Ki67 二值化。

    Args:
        img_path (Path): Ki67 影像路徑。
        out_path (Path): 二值化輸出路徑。
        channel (str, optional): 使用的色彩通道。
        gauss_sigma (float, optional): 高斯平滑 sigma。
        clahe_clip (float, optional): CLAHE clip limit?
        otsu_scale (float, optional): Otsu 門檻縮放比例。
        min_obj_area (int, optional): 最小物件面積。
        open_kernel (int, optional): 開運算 kernel 尺寸。

    Returns:
        None: 此函式僅負責輸出檔案。
    """

    img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"[ERROR] 找不到圖片: {img_path}")

    if img_color.ndim == 3:
        chans = {
            "b": img_color[:, :, 0],
            "g": img_color[:, :, 1],
            "r": img_color[:, :, 2],
        }
        if channel == "auto":
            channel = max(chans, key=lambda k: np.mean(chans[k]))
        chan_img = chans.get(channel.lower(), chans["g"])
    else:
        chan_img = img_color

    gray = chan_img.astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=gauss_sigma)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if clahe_clip and clahe_clip > 0:
        gray = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(gray)

    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = max(1, int(ret * otsu_scale))
    _, binary = cv2.threshold(gray, low_thresh, 255, cv2.THRESH_BINARY)

    if open_kernel and open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    binary = _remove_small_components(binary, min_obj_area)
    cv2.imwrite(str(out_path), binary)


def detect_ki67_positive(
    roi_dir: Path, ki67_dir: Path, output_dir: Path, threshold: float = 0.10
):
    """依 ROI 與 Ki67 mask 重疊比例判定陽性。

    Args:
        roi_dir (Path): outlines 文字檔路徑。
        ki67_dir (Path): Ki67 二值化影像路徑。
        output_dir (Path): 陽性標記輸出路徑。
        threshold (float, optional): 陽性判定的重疊比例門檻。

    Returns:
        None: 此函式僅負責輸出檔案。
    """
    if not ki67_dir.exists():
        print(f"[WARN] 找不到 Ki67 mask: {ki67_dir}")
        return

    ki67_mask = skio.imread(str(ki67_dir)) > 0
    positive_labels = _ki67_positive_labels_from_mask(roi_dir, ki67_mask, threshold)
    np.savetxt(output_dir, positive_labels, fmt="%d")
    print(f"[INFO] 已輸出 Ki67 陽性 label: {output_dir}")


def _ki67_positive_labels_from_mask(
    roi_dir: Path, ki67_mask: np.ndarray, threshold: float = 0.10
) -> list[int]:
    """Return merged-outline nucleus line labels that overlap a Ki67 mask."""
    shape = ki67_mask.shape
    with open(roi_dir) as f:
        lines = [line.strip() for line in f if line.strip()]

    positive_labels = []
    for idx, line in enumerate(lines):
        if idx % 2 or line == "-1,-1":
            continue
        coords = list(map(int, line.split(",")))
        if len(coords) < 6:
            continue
        xy = np.array(coords).reshape(-1, 2)
        mask = polygon2mask(shape, xy[:, [1, 0]])
        roi_area = mask.sum()
        if roi_area == 0:
            continue
        overlap_area = np.logical_and(mask, ki67_mask).sum()
        if overlap_area / roi_area >= threshold:
            positive_labels.append(idx + 1)
    return positive_labels


def _score_ki67_mask_for_outline(
    roi_dir: Path, ki67_mask_path: Path, threshold: float = 0.10
) -> tuple[int, int, float]:
    """Score a Ki67 mask against one PC merged-outline file."""
    if not roi_dir.exists() or not ki67_mask_path.exists():
        return 0, 0, 0.0

    ki67_mask = skio.imread(str(ki67_mask_path)) > 0
    shape = ki67_mask.shape
    with open(roi_dir) as f:
        lines = [line.strip() for line in f if line.strip()]

    positive = 0
    total = 0
    fractions: list[float] = []
    for idx, line in enumerate(lines):
        if idx % 2 or line == "-1,-1":
            continue
        coords = list(map(int, line.split(",")))
        if len(coords) < 6:
            continue
        xy = np.array(coords).reshape(-1, 2)
        mask = polygon2mask(shape, xy[:, [1, 0]])
        roi_area = mask.sum()
        if roi_area == 0:
            continue
        total += 1
        overlap_fraction = float(np.logical_and(mask, ki67_mask).sum() / roi_area)
        fractions.append(overlap_fraction)
        if overlap_fraction >= threshold:
            positive += 1

    return positive, total, float(np.mean(fractions)) if fractions else 0.0


def _linear_assignment(score_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Maximize row/column assignment scores with scipy when available."""
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-score_matrix)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for row, col in sorted(
            np.ndindex(score_matrix.shape),
            key=lambda rc: float(score_matrix[rc[0], rc[1]]),
            reverse=True,
        ):
            if row in used_rows or col in used_cols:
                continue
            pairs.append((int(row), int(col)))
            used_rows.add(int(row))
            used_cols.add(int(col))
            if len(used_rows) == score_matrix.shape[0] or len(used_cols) == score_matrix.shape[1]:
                break
        return pairs


def _remap_ki67_masks_to_outlines(
    outline_paths_by_pc_stem: dict[str, Path],
    mask_paths_by_pc_stem: dict[str, Path],
    threshold: float = 0.10,
) -> dict[str, Path]:
    """Detect and fix whole-batch Ki67/PC filename shifts."""
    pc_stems = [
        stem
        for stem, path in outline_paths_by_pc_stem.items()
        if path.exists() and stem in mask_paths_by_pc_stem
    ]
    source_stems = [
        stem for stem in mask_paths_by_pc_stem if mask_paths_by_pc_stem[stem].exists()
    ]
    if len(pc_stems) < 2 or len(source_stems) < 2:
        return dict(mask_paths_by_pc_stem)

    score_matrix = np.zeros((len(pc_stems), len(source_stems)), dtype=np.float32)
    positive_counts = np.zeros_like(score_matrix, dtype=np.int32)
    total_counts = np.zeros_like(score_matrix, dtype=np.int32)
    for row, pc_stem in enumerate(pc_stems):
        outline_path = outline_paths_by_pc_stem[pc_stem]
        for col, source_stem in enumerate(source_stems):
            positive, total, mean_overlap = _score_ki67_mask_for_outline(
                outline_path, mask_paths_by_pc_stem[source_stem], threshold
            )
            positive_counts[row, col] = positive
            total_counts[row, col] = total
            score_matrix[row, col] = float(positive * 1000.0 + mean_overlap * 100.0)

    assignment = _linear_assignment(score_matrix)
    source_col_by_stem = {stem: col for col, stem in enumerate(source_stems)}
    identity_pairs = [
        (row, source_col_by_stem[stem])
        for row, stem in enumerate(pc_stems)
        if stem in source_col_by_stem
    ]

    changed = any(pc_stems[row] != source_stems[col] for row, col in assignment)
    if not changed:
        print("[INFO] Ki67 masks match PC filenames; no remap needed.")
        return dict(mask_paths_by_pc_stem)

    assigned_positive = int(sum(positive_counts[row, col] for row, col in assignment))
    identity_positive = int(sum(positive_counts[row, col] for row, col in identity_pairs))
    required_gain = max(20, len(pc_stems) * 2)
    if assigned_positive < identity_positive * 3 or assigned_positive - identity_positive < required_gain:
        print(
            "[INFO] Ki67 remap skipped; best assignment did not improve enough "
            f"({assigned_positive} vs {identity_positive} positives)."
        )
        return dict(mask_paths_by_pc_stem)

    remapped = dict(mask_paths_by_pc_stem)
    for row, col in assignment:
        target_stem = pc_stems[row]
        source_stem = source_stems[col]
        remapped[target_stem] = mask_paths_by_pc_stem[source_stem]

        positive = int(positive_counts[row, col])
        total = int(total_counts[row, col])
        if total > 0 and positive / total < 0.05:
            print(
                f"[WARN] Low-confidence Ki67 remap: {source_stem} -> {target_stem} "
                f"({positive}/{total} positive nuclei)."
            )

    print(
        f"[INFO] Remapped Ki67 masks for {len(assignment)} PC images "
        f"({assigned_positive} positives vs {identity_positive} by filename)."
    )
    return remapped


def merge_ki67_labels(param_csv: Path, label_file: Path, output_csv: Path):
    """將 Ki67 陽性標記合併回參數表。

    Args:
        param_csv (Path): 參數表路徑。
        label_file (Path): 陽性標記文字檔路徑。
        output_csv (Path): 合併後輸出路徑。

    Returns:
        None: 此函式僅負責輸出檔案。
    """
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
    ki67_backend: str = "pyimagej",
    clean_temp: bool = True,
) -> None:
    """執行主流程的幾何、螢光與 Ki67 分析。

    Args:
        data_name (str): 資料集路徑或名稱。
        fluor_analy (bool, optional): 是否啟用螢光分析。
        ki67 (bool, optional): 是否啟用 Ki67 判定。
        ki67_backend (str, optional): Ki67 二值化後端。
        clean_temp (bool, optional): 是否清理暫存檔。

    Returns:
        None: 此函式僅負責輸出檔案。
    """

    data_path = Path(data_name)
    pc_dir = data_path / "PC"
    ki67_dir = data_path / "KI67"

    fluor_subdirs = ["DF", "LT", "IDO"]
    existing_fluor_dirs = [
        (sub, data_path / sub)
        for sub in fluor_subdirs
        if (data_path / sub).exists() and (data_path / sub).is_dir()
    ]

    if not pc_dir.exists() or not pc_dir.is_dir():
        print(f"[錯誤] 找不到 PC 資料夾：{pc_dir}")
        return

    if existing_fluor_dirs:
        folders = ", ".join(label for label, _ in existing_fluor_dirs)
        print(f"[資訊] 已偵測到螢光資料夾：{folders}")
    elif fluor_analy:
        print("[警告] 已啟用螢光分析，但未找到 DF 或 LT 資料夾；僅輸出幾何參數。")

    def get_image_names(folder: Path) -> list[str]:
        if not folder.exists() or not folder.is_dir():
            return []
        return [f.name for f in list_files(folder, [".jpg", ".png", ".tif", ".tiff"])]

    pc_images = get_image_names(pc_dir)
    if not pc_images:
        print(f"[錯誤] PC 資料夾中找不到影像：{pc_dir}")
        return

    ki67_images = get_image_names(ki67_dir)
    fluor_images = {label: get_image_names(path) for label, path in existing_fluor_dirs}

    counts = [
        len(pc_images),
        len(ki67_images),
        *(len(v) for v in fluor_images.values()),
    ]
    max_len = max(counts) if counts else 0

    def safe_get(items: list[str], idx: int) -> str:
        return items[idx] if idx < len(items) else ""

    rows = []
    for idx in range(max_len):
        row = {"PC_Name": safe_get(pc_images, idx)}
        for label, _ in existing_fluor_dirs:
            row[f"{label}_Name"] = safe_get(fluor_images.get(label, []), idx)
        row["KI67_Name"] = safe_get(ki67_images, idx)
        rows.append(row)

    mapping = pd.DataFrame(rows)
    mapping_path = data_path / "image_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    print(f"[資訊] 已更新影像對照表：{mapping_path}")

    analy_dir = output_dir(data_path, "results")
    outline_dir = output_dir(data_path, "outline")
    ki67_masks_by_pc_stem: dict[str, Path] = {}
    if ki67:
        binary_dir = output_dir(data_path, "binary")
        raw_ki67_masks_by_pc_stem: dict[str, Path] = {}
        outline_paths_by_pc_stem: dict[str, Path] = {}
        for _, map_row in mapping.iterrows():
            pc_name = str(map_row.get("PC_Name", "")).strip()
            ki67_name = str(map_row.get("KI67_Name", "")).strip()
            if not pc_name:
                continue

            pc_stem = Path(pc_name).stem
            outline_paths_by_pc_stem[pc_stem] = (
                outline_dir / f"{pc_stem}_merged_cp_outlines.txt"
            )
            if not ki67_name:
                continue

            ki67_img = ki67_dir / ki67_name
            if not ki67_img.exists():
                print(f"[WARN] Missing Ki67 image: {ki67_img}")
                continue
            raw_ki67_masks_by_pc_stem[pc_stem] = ki67_binarize(
                ki67_img, backend=ki67_backend
            )

        ki67_masks_by_pc_stem = _remap_ki67_masks_to_outlines(
            outline_paths_by_pc_stem, raw_ki67_masks_by_pc_stem
        )

    for i in trange(len(mapping), desc="Processing images(analysis)"):
        row = mapping.iloc[i]
        pc_name = str(row.get("PC_Name", "")).strip()
        if not pc_name:
            print(f"[警告] 第 {i + 1} 筆缺少 PC 影像名稱，已略過。")
            continue

        pc_img = pc_dir / pc_name
        if not pc_img.exists():
            print(f"[警告] 找不到 PC 影像：{pc_img}")
            continue

        outlines_txt = outline_dir / f"{pc_img.stem}_merged_cp_outlines.txt"
        if not outlines_txt.exists():
            print(f"[警告] 缺少輪廓檔案：{outlines_txt}")
            continue

        param_csv = analy_dir / f"{pc_img.stem}_params.csv"
        merged_param_csv = analy_dir / f"{pc_img.stem}_params_merged.csv"
        final_csv = analy_dir / f"{pc_img.stem}_final.csv"

        fluor_outputs: dict[str, dict[str, Path]] = {}
        for label, _ in existing_fluor_dirs:
            suffix = label.lower()
            if label.upper() == "IDO":
                fluor_outputs[label] = {
                    "measure_csv": analy_dir / f"{pc_img.stem}_{suffix}_measurements.csv",
                    "flat_csv": analy_dir / f"{pc_img.stem}_{suffix}_measurements.csv",
                    "merged_csv": analy_dir / f"{pc_img.stem}_final_{suffix}.csv",
                }
            else:
                fluor_outputs[label] = {
                    "measure_csv": analy_dir
                    / f"{pc_img.stem}_{suffix}_fluorescence.csv",
                    "flat_csv": analy_dir / f"{pc_img.stem}_{suffix}_fluor_flat.csv",
                    "merged_csv": analy_dir / f"{pc_img.stem}_final_{suffix}.csv",
                }

        param_anal(pc_img, outlines_txt, param_csv)
        merged_excel(param_csv, merged_param_csv)

        current_final_path: Path = merged_param_csv
        processed_fluor = False

        if fluor_analy and existing_fluor_dirs:
            for label, fluor_dir in existing_fluor_dirs:
                col_name = f"{label}_Name"
                df_name = str(row.get(col_name, "")).strip()
                if not df_name:
                    print(f"[警告] {pc_img.stem} 缺少 {label} 影像，已略過。")
                    continue

                df_img = fluor_dir / df_name
                if not df_img.exists():
                    print(f"[警告] 找不到 {label} 影像：{df_img}")
                    continue

                outputs = fluor_outputs[label]
                merge_input_path: Path
                if label.upper() == "IDO":
                    ido_anal(df_img, outlines_txt, outputs["measure_csv"])
                    merge_input_path = outputs["measure_csv"]
                else:
                    flour_anal(df_img, outlines_txt, outputs["measure_csv"])
                    flatten_fluor_table(outputs["measure_csv"], outputs["flat_csv"])
                    merge_input_path = outputs["flat_csv"]

                if (
                    not merge_input_path.exists()
                    or merge_input_path.stat().st_size == 0
                ):
                    print(
                        f"[警告] {label} 螢光分析結果為空，略過合併：{outputs['flat_csv']}"
                    )
                    continue

                merge_with_flour(
                    current_final_path, merge_input_path, outputs["merged_csv"]
                )
                current_final_path = outputs["merged_csv"]
                processed_fluor = True

        if processed_fluor:
            if current_final_path != final_csv:
                copyfile(current_final_path, final_csv)
        else:
            if merged_param_csv != final_csv:
                copyfile(merged_param_csv, final_csv)
        current_final_path = final_csv

        for outputs in fluor_outputs.values():
            merged_path = outputs["merged_csv"]
            if merged_path.exists() and merged_path != final_csv:
                merged_path.unlink()

        if ki67:
            ki67_mask = ki67_masks_by_pc_stem.get(pc_img.stem)
            if ki67_mask is None:
                print(f"[WARN] Missing Ki67 mask for {pc_img.stem}")
            else:
                binary_dir = output_dir(data_path, "binary")
                ki67_label = binary_dir / f"{pc_img.stem}_label.txt"

                detect_ki67_positive(outlines_txt, ki67_mask, ki67_label)
                merge_ki67_labels(final_csv, ki67_label, final_csv)
                print(f"[INFO] Ki67 merged into: {final_csv}")

    merge_all_final_csvs(data_path)

    if clean_temp:
        remove_temp_files(analy_dir)
        remove_temp_files(output_dir(data_path, "outline"))
