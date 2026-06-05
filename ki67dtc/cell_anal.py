import os
import sys
import tempfile
import math
import json
import csv
from contextlib import contextmanager
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange
from typing import Any, Optional, Union
from skimage.draw import polygon, polygon2mask
from shapely.affinity import scale
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
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


def _read_image_array(path: Union[str, Path]) -> np.ndarray:
    """Read an image with OpenCV first, then fall back to tifffile."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if image is None:
        image = tifffile.imread(path)
    return np.squeeze(image)


def _read_gray_float(path: Union[str, Path]) -> np.ndarray:
    """Read an image as grayscale float, keeping skimage-like [0, 1] scaling."""
    image = _read_image_array(path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        return image.astype(np.float32) / float(max_value)
    return image.astype(np.float32)


def _read_binary_mask(path: Union[str, Path]) -> np.ndarray:
    """Read any mask image as a boolean array."""
    image = _read_image_array(path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image > 0


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

_FIND_EDGES_MACRO = r"""
#@ String input_path
#@ String output_path

run("Close All");
setBatchMode(true);
open(input_path);
run("Find Edges");
saveAs("Tiff", output_path);
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

_DEBRIS_IMAGEJ_MACRO = r"""
#@ String signal_path
#@ String background_mask_path
#@ String debris_mask_path
#@ String particle_table_path
#@ String threshold_method
#@ double min_particle_area
#@ double max_particle_area
#@ double rolling_ball_radius
#@ boolean subtract_background
#@ boolean watershed

run("Close All");
run("Clear Results");
setBatchMode(true);

open(signal_path);
rename("signal");
selectWindow("signal");
run("8-bit");
if (subtract_background) {
    run("Subtract Background...", "rolling=" + rolling_ball_radius);
}

open(background_mask_path);
rename("background_mask");
selectWindow("background_mask");
run("8-bit");
setThreshold(1, 255);
run("Convert to Mask");
run("Create Selection");

if (selectionType() < 0) {
    File.saveString("Area\tMean\tMin\tMax\tX\tY\n", particle_table_path);
    selectWindow("signal");
    width = getWidth();
    height = getHeight();
    newImage("DebrisMaskBlank", "8-bit black", width, height, 1);
    saveAs("PNG", debris_mask_path);
    run("Clear Results");
    setBatchMode(false);
    close("*");
    exit();
}

selectWindow("signal");
run("Restore Selection");
setBackgroundColor(0, 0, 0);
run("Clear Outside");
setAutoThreshold(threshold_method + " dark no-reset");
setOption("BlackBackground", true);
run("Convert to Mask");
rename("threshold_mask");

selectWindow("background_mask");
run("Create Selection");
selectWindow("threshold_mask");
run("Restore Selection");
setBackgroundColor(0, 0, 0);
run("Clear Outside");

if (watershed) {
    run("Watershed");
}

width = getWidth();
height = getHeight();
run("Set Measurements...", "area mean min max centroid perimeter shape feret's display redirect=None decimal=6");
particle_options = "size=" + min_particle_area + "-" + max_particle_area + " show=Masks display clear";
run("Analyze Particles...", particle_options);
particle_count = nResults;

if (particle_count > 0) {
    saveAs("Results", particle_table_path);
} else {
    File.saveString("Area\tMean\tMin\tMax\tX\tY\tPerim.\tCirc.\tFeret\n", particle_table_path);
}

if (particle_count > 0) {
    mask_title = "Mask of threshold_mask";
    if (isOpen(mask_title)) {
        selectWindow(mask_title);
    } else if (isOpen("Mask")) {
        selectWindow("Mask");
    } else {
        newImage("DebrisMaskBlank", "8-bit black", width, height, 1);
    }
    run("Invert");
} else {
    newImage("DebrisMaskBlank", "8-bit black", width, height, 1);
}
run("8-bit");
saveAs("PNG", debris_mask_path);

run("Clear Results");
setBatchMode(false);
close("*");
"""

_LBP_ROI_MACRO = r"""
#@ String signal_path
#@ String mask_path
#@ String output_path

function bitAt(code, bit) {
    return floor(code / pow(2, bit)) % 2;
}

function isUniform(code) {
    previous = bitAt(code, 7);
    transitions = 0;
    for (bit = 0; bit < 8; bit++) {
        current = bitAt(code, bit);
        if (current != previous) {
            transitions++;
        }
        previous = current;
    }
    if (transitions <= 2) {
        return 1;
    }
    return 0;
}

run("Close All");
setBatchMode(true);
open(signal_path);
rename("signal");
selectWindow("signal");
run("8-bit");
open(mask_path);
rename("mask");
selectWindow("mask");
run("8-bit");
setThreshold(1, 255);
run("Convert to Mask");

selectWindow("signal");
w = getWidth();
h = getHeight();
hist = newArray(256);
total = 0;
sum_code = 0;
sum_sq_code = 0;
uniform_count = 0;

for (y = 1; y < h - 1; y++) {
    for (x = 1; x < w - 1; x++) {
        selectWindow("mask");
        if (getPixel(x, y) <= 0) {
            continue;
        }

        selectWindow("signal");
        center = getPixel(x, y);
        code = 0;
        if (getPixel(x - 1, y - 1) >= center) code += 1;
        if (getPixel(x, y - 1) >= center) code += 2;
        if (getPixel(x + 1, y - 1) >= center) code += 4;
        if (getPixel(x + 1, y) >= center) code += 8;
        if (getPixel(x + 1, y + 1) >= center) code += 16;
        if (getPixel(x, y + 1) >= center) code += 32;
        if (getPixel(x - 1, y + 1) >= center) code += 64;
        if (getPixel(x - 1, y) >= center) code += 128;

        hist[code]++;
        total++;
        sum_code += code;
        sum_sq_code += code * code;
        if (isUniform(code) == 1) {
            uniform_count++;
        }
    }
}

if (total <= 0) {
    File.saveString("valid=0\n", output_path);
    setBatchMode(false);
    close("*");
    exit();
}

mean_code = sum_code / total;
variance_code = sum_sq_code / total - mean_code * mean_code;
if (variance_code < 0) variance_code = 0;
std_code = sqrt(variance_code);
entropy = 0;
for (i = 0; i < 256; i++) {
    if (hist[i] > 0) {
        p = hist[i] / total;
        entropy -= p * log(p) / log(2);
    }
}
uniform_ratio = uniform_count / total;

result_txt = "";
result_txt = result_txt + "valid=1\n";
result_txt = result_txt + "mean=" + mean_code + "\n";
result_txt = result_txt + "std=" + std_code + "\n";
result_txt = result_txt + "entropy=" + entropy + "\n";
result_txt = result_txt + "uniform_ratio=" + uniform_ratio + "\n";
for (bin = 0; bin < 16; bin++) {
    bin_count = 0;
    for (code = bin * 16; code < bin * 16 + 16; code++) {
        bin_count += hist[code];
    }
    result_txt = result_txt + "hist_bin_" + bin + "=" + (bin_count / total) + "\n";
}

File.saveString(result_txt, output_path);
setBatchMode(false);
close("*");
"""

_MEASURE_ROI_MACRO = r"""
#@ String signal_path
#@ String mask_path
#@ String output_path

function percentileFromHistogram(values, counts, q) {
    total = 0;
    for (i = 0; i < counts.length; i++) {
        total = total + counts[i];
    }
    if (total <= 0) {
        return 0/0;
    }
    target = q * total;
    cumulative = 0;
    for (i = 0; i < counts.length; i++) {
        cumulative = cumulative + counts[i];
        if (cumulative >= target) {
            return values[i];
        }
    }
    return values[values.length - 1];
}

function entropyFromHistogram(counts) {
    total = 0;
    for (i = 0; i < counts.length; i++) {
        total = total + counts[i];
    }
    if (total <= 0) {
        return 0/0;
    }
    entropy = 0;
    for (i = 0; i < counts.length; i++) {
        if (counts[i] > 0) {
            p = counts[i] / total;
            entropy = entropy - p * log(p) / log(2);
        }
    }
    return entropy;
}

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
run("Set Measurements...", "area mean standard min integrated centroid perimeter shape feret's fit median skewness kurtosis decimal=6");
run("Measure");

area = getResult("Area", 0);
mean = getResult("Mean", 0);
std = getResult("StdDev", 0);
minv = getResult("Min", 0);
maxv = getResult("Max", 0);
intden = getResult("IntDen", 0);
rawintden = getResult("RawIntDen", 0);
medianv = getResult("Median", 0);
skewv = getResult("Skew", 0);
kurtv = getResult("Kurt", 0);
xcentroid = getResult("X", 0);
ycentroid = getResult("Y", 0);
perim = getResult("Perim.", 0);
feret = getResult("Feret", 0);
minferet = getResult("MinFeret", 0);
major = getResult("Major", 0);
minor = getResult("Minor", 0);
ar = getResult("AR", 0);
roundv = getResult("Round", 0);
circ = getResult("Circ.", 0);
cv = 0/0;
if (mean != 0) {
    cv = std / mean;
}
rangev = maxv - minv;
getHistogram(hist_values, hist_counts, 256);
p10 = percentileFromHistogram(hist_values, hist_counts, 0.10);
p25 = percentileFromHistogram(hist_values, hist_counts, 0.25);
p75 = percentileFromHistogram(hist_values, hist_counts, 0.75);
p90 = percentileFromHistogram(hist_values, hist_counts, 0.90);
iqr80 = p90 - p10;
entropyv = entropyFromHistogram(hist_counts);

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
result_txt = result_txt + "cv=" + cv + "\n";
result_txt = result_txt + "range=" + rangev + "\n";
result_txt = result_txt + "median=" + medianv + "\n";
result_txt = result_txt + "p10=" + p10 + "\n";
result_txt = result_txt + "p25=" + p25 + "\n";
result_txt = result_txt + "p75=" + p75 + "\n";
result_txt = result_txt + "p90=" + p90 + "\n";
result_txt = result_txt + "iqr80=" + iqr80 + "\n";
result_txt = result_txt + "entropy=" + entropyv + "\n";
result_txt = result_txt + "skewness=" + skewv + "\n";
result_txt = result_txt + "kurtosis=" + kurtv + "\n";
result_txt = result_txt + "x_centroid=" + xcentroid + "\n";
result_txt = result_txt + "y_centroid=" + ycentroid + "\n";
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

INTENSITY_FEATURE_PARAMETER_COLUMNS = [
    "Mean",
    "StdDev",
    "Min",
    "Max",
    "IntDen",
    "RawIntDen",
    "CV",
    "Range",
    "P10",
    "P25",
    "P75",
    "P90",
    "IQR80",
    "Entropy",
    "Skewness",
    "Kurtosis",
]

TEXTURE_FEATURE_PARAMETER_COLUMNS = [
    "GLCM ASM",
    "GLCM Contrast",
    "GLCM Correlation",
    "GLCM Difference Variance",
    "GLCM Entropy",
    "GLCM Homogeneity",
    "LBP Mean",
    "LBP StdDev",
    "LBP Entropy",
    "LBP Uniform Ratio",
    *[f"LBP Hist Bin {idx:02d}" for idx in range(16)],
]

DERIVED_FEATURE_PARAMETER_COLUMNS = [
    "Nuc Cyto Mean Ratio",
    "Nuc Cyto IntDen Ratio",
    "Nuc Cyto RawIntDen Ratio",
    "Nuc Cyto Entropy Difference",
    "Nuc Cyto CV Difference",
    "Nucleus Centroid Offset",
]

HALO_FEATURE_PARAMETER_COLUMNS = [
    "Halo Outer Mean",
    "Halo Outer StdDev",
    "Halo Inner Mean",
    "Halo Inner StdDev",
    "Halo Inner Outer Diff",
    "Halo Width",
    "Edge Sharpness",
]

POPULATION_FEATURE_PARAMETER_COLUMNS = [
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
]

SHAPE_COMPLEXITY_FEATURE_PARAMETER_COLUMNS = [
    "Protrusion Count",
    "Mean Convex Defect Depth",
    "Max Convex Defect Depth",
    "Fractal Dimension",
    "Boundary Inflection Count",
]

MITOSIS_FEATURE_PARAMETER_COLUMNS = [
    "Mitotic Score",
    "Daughter Pair Flag",
    "Protrusion Retraction Score",
]

DEBRIS_FEATURE_PARAMETER_COLUMNS = [
    "Debris Count",
    "Debris Area Fraction",
    "Nearest Debris Distance",
    "Debris Mean Area",
    "Debris Density",
]

EXTRA_FEATURE_PARAMETER_COLUMNS = (
    DERIVED_FEATURE_PARAMETER_COLUMNS
    + HALO_FEATURE_PARAMETER_COLUMNS
    + POPULATION_FEATURE_PARAMETER_COLUMNS
    + SHAPE_COMPLEXITY_FEATURE_PARAMETER_COLUMNS
    + DEBRIS_FEATURE_PARAMETER_COLUMNS
    + MITOSIS_FEATURE_PARAMETER_COLUMNS
)
CELL_LEVEL_FEATURE_PARAMETER_COLUMNS = ["Karyoplasmic Ratio"] + EXTRA_FEATURE_PARAMETER_COLUMNS

DEBRIS_CONFIG_PATH = Path(__file__).with_name("debris_feature_config.json")
DEFAULT_DEBRIS_CONFIG: dict[str, Any] = {
    "threshold_method": "Triangle",
    "min_size_px": 5,
    "max_size_fraction_of_median_nucleus_area": 0.08,
    "cell_dilate_px": 3,
    "subtract_background": False,
    "rolling_ball_radius": 50.0,
    "watershed": False,
}

TEXTURE_GLCM_GRAY_LEVELS = 64
TEXTURE_GLCM_DISTANCE = 2
_GLCM_ORIENTATIONS = None


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
        "cv": np.nan,
        "range": np.nan,
        "median": np.nan,
        "p10": np.nan,
        "p25": np.nan,
        "p75": np.nan,
        "p90": np.nan,
        "iqr80": np.nan,
        "entropy": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "x_centroid": np.nan,
        "y_centroid": np.nan,
        "perimeter": np.nan,
        "feret": np.nan,
        "minferet": np.nan,
        "major": np.nan,
        "minor": np.nan,
        "eccentricity": np.nan,
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


def _intensity_feature_parameter_values(measurement: dict[str, float]) -> list[float]:
    """Return intensity feature parameters in INTENSITY_FEATURE_PARAMETER_COLUMNS order."""
    return [
        measurement["mean"],
        measurement["std"],
        measurement["min"],
        measurement["max"],
        measurement["intden"],
        measurement["raw_intden"],
        measurement["cv"],
        measurement["range"],
        measurement["p10"],
        measurement["p25"],
        measurement["p75"],
        measurement["p90"],
        measurement["iqr80"],
        measurement["entropy"],
        measurement["skewness"],
        measurement["kurtosis"],
    ]


def _empty_texture_feature_values() -> list[float]:
    """回傳 texture 參數的空值清單。"""
    return [np.nan] * len(TEXTURE_FEATURE_PARAMETER_COLUMNS)


def _empty_glcm_feature_values() -> list[float]:
    """回傳 GLCM 參數的空值清單。"""
    return [np.nan] * 6


def _empty_lbp_feature_values() -> list[float]:
    """回傳 LBP 參數的空值清單。"""
    return [np.nan] * 20


def _get_glcm_orientations() -> list[Any]:
    """取得 PyImageJ Haralick GLCM 的四個量測方向。"""
    global _GLCM_ORIENTATIONS
    if _GLCM_ORIENTATIONS is not None:
        return _GLCM_ORIENTATIONS
    import imagej as imagej_module

    matrix_orientation = imagej_module.sj.jimport(
        "net.imagej.ops.image.cooccurrenceMatrix.MatrixOrientation2D"
    )
    _GLCM_ORIENTATIONS = [
        matrix_orientation.ANTIDIAGONAL,
        matrix_orientation.DIAGONAL,
        matrix_orientation.HORIZONTAL,
        matrix_orientation.VERTICAL,
    ]
    return _GLCM_ORIENTATIONS


def _java_numeric_value(ij: Any, value: Any) -> float:
    """將 Java numeric 物件轉為 Python float。"""
    converted = ij.py.from_java(value)
    if hasattr(converted, "value"):
        return float(converted.value)
    return float(converted)


def _prepare_texture_crop(
    signal: np.ndarray,
    roi_mask: np.ndarray,
    erode_px: int,
    distance: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """裁切 ROI texture 量測區域，並以 ROI 平均值填補背景。

    Args:
        signal (np.ndarray): 2D 影像強度。
        roi_mask (np.ndarray): ROI 布林遮罩。
        erode_px (int): 量測前 mask erosion 次數。
        distance (int): 紋理演算法使用的鄰近距離。

    Returns:
        tuple[np.ndarray, np.ndarray] | None: 裁切後影像與 mask；ROI 太小時回傳 `None`。
    """
    if roi_mask is None or not np.any(roi_mask):
        return None

    mask = roi_mask.astype(bool)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=erode_px).astype(bool)
        if np.count_nonzero(eroded) >= max(16, (distance + 1) ** 2):
            mask = eroded

    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    pad = max(1, int(distance))
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad + 1, signal.shape[0])
    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, signal.shape[1])
    if (y2 - y1) <= distance or (x2 - x1) <= distance:
        return None

    crop = signal[y1:y2, x1:x2].astype(np.float32, copy=True)
    crop_mask = mask[y1:y2, x1:x2]
    roi_values = crop[crop_mask]
    if roi_values.size < max(16, (distance + 1) ** 2):
        return None

    fill_value = float(np.mean(roi_values))
    crop[~crop_mask] = fill_value
    return crop, crop_mask


def _texture_feature_parameter_values(
    ij: Any,
    signal: np.ndarray,
    roi_mask: np.ndarray,
    erode_px: int,
    gray_levels: int = TEXTURE_GLCM_GRAY_LEVELS,
    distance: int = TEXTURE_GLCM_DISTANCE,
) -> list[float]:
    """使用 PyImageJ Haralick ops 提取 GLCM texture 參數。

    Args:
        ij (Any): 已初始化的 PyImageJ 物件。
        signal (np.ndarray): 2D 影像強度。
        roi_mask (np.ndarray): ROI 布林遮罩。
        erode_px (int): 量測前 mask erosion 次數。
        gray_levels (int): GLCM 灰階層數。
        distance (int): GLCM pixel distance。

    Returns:
        list[float]: GLCM ASM、Contrast、Correlation、Difference Variance、Entropy、Homogeneity。
    """
    prepared = _prepare_texture_crop(signal, roi_mask, erode_px, distance)
    if prepared is None:
        return _empty_glcm_feature_values()
    crop, _ = prepared

    try:
        dataset = ij.py.to_dataset(crop)
        orientations = _get_glcm_orientations()
        angle_values: list[list[float]] = []
        for angle in orientations:
            angle_values.append(
                [
                    _java_numeric_value(
                        ij,
                        ij.op().haralick().asm(
                            dataset, int(gray_levels), int(distance), angle
                        ),
                    ),
                    _java_numeric_value(
                        ij,
                        ij.op().haralick().contrast(
                            dataset, int(gray_levels), int(distance), angle
                        ),
                    ),
                    _java_numeric_value(
                        ij,
                        ij.op().haralick().correlation(
                            dataset, int(gray_levels), int(distance), angle
                        ),
                    ),
                    _java_numeric_value(
                        ij,
                        ij.op().haralick().differenceVariance(
                            dataset, int(gray_levels), int(distance), angle
                        ),
                    ),
                    _java_numeric_value(
                        ij,
                        ij.op().haralick().entropy(
                            dataset, int(gray_levels), int(distance), angle
                        ),
                    ),
                    _java_numeric_value(
                        ij,
                        ij.op().haralick().textureHomogeneity(
                            dataset, int(gray_levels), int(distance), angle
                        ),
                    ),
                ]
            )
    except Exception as exc:
        print(f"[WARN] GLCM texture extraction failed: {exc}")
        return _empty_glcm_feature_values()

    values = np.asarray(angle_values, dtype=np.float64)
    if values.size == 0:
        return _empty_glcm_feature_values()
    with np.errstate(all="ignore"):
        means = np.nanmean(values, axis=0)
    return [float(value) if np.isfinite(value) else np.nan for value in means]


def _read_lbp_measurements(path: Path) -> list[float]:
    """讀取 ImageJ LBP macro 輸出的 key-value 結果。"""
    if not path.exists():
        return _empty_lbp_feature_values()
    parsed: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
    if parsed.get("valid") != "1":
        return _empty_lbp_feature_values()
    keys = ["mean", "std", "entropy", "uniform_ratio"] + [
        f"hist_bin_{idx}" for idx in range(16)
    ]
    values: list[float] = []
    for key in keys:
        try:
            values.append(float(parsed.get(key, "nan")))
        except Exception:
            values.append(np.nan)
    return values


def _lbp_feature_parameter_values(
    ij: Any,
    signal: np.ndarray,
    roi_mask: np.ndarray,
    erode_px: int,
    temp_dir: Path,
    slot: str,
) -> list[float]:
    """使用 ImageJ macro 提取 LBP texture 參數。

    Args:
        ij (Any): 已初始化的 PyImageJ 物件。
        signal (np.ndarray): 2D 影像強度。
        roi_mask (np.ndarray): ROI 布林遮罩。
        erode_px (int): 量測前 mask erosion 次數。
        temp_dir (Path): 暫存影像與文字結果的資料夾。
        slot (str): 暫存檔名識別字。

    Returns:
        list[float]: LBP summary 與 16 個 histogram bin。
    """
    prepared = _prepare_texture_crop(signal, roi_mask, erode_px, distance=1)
    if prepared is None:
        return _empty_lbp_feature_values()
    crop, crop_mask = prepared
    signal_path = temp_dir / f"{slot}_lbp_signal.tif"
    mask_path = temp_dir / f"{slot}_lbp_mask.tif"
    output_path = temp_dir / f"{slot}_lbp.txt"
    tifffile.imwrite(signal_path, crop.astype(np.float32))
    tifffile.imwrite(mask_path, (crop_mask.astype(np.uint8) * 255))
    try:
        _run_macro_quiet(
            ij,
            _LBP_ROI_MACRO,
            args={
                "signal_path": str(signal_path),
                "mask_path": str(mask_path),
                "output_path": str(output_path),
            },
        )
    except Exception as exc:
        print(f"[WARN] LBP texture extraction failed: {exc}")
        return _empty_lbp_feature_values()
    return _read_lbp_measurements(output_path)


def _safe_divide(numerator: float, denominator: float) -> float:
    """安全除法，遇到 NaN 或近零分母時回傳 NaN。"""
    if np.isnan(numerator) or np.isnan(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def _safe_difference(left: float, right: float) -> float:
    """安全相減，任一輸入為 NaN 時回傳 NaN。"""
    if np.isnan(left) or np.isnan(right):
        return np.nan
    return float(left - right)


def _safe_abs_difference(left: float, right: float) -> float:
    """安全絕對差，任一輸入為 NaN 時回傳 NaN。"""
    if np.isnan(left) or np.isnan(right):
        return np.nan
    return float(abs(left - right))


def _safe_cv(values: np.ndarray) -> float:
    """計算變異係數，資料不足或平均近零時回傳 NaN。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    mean_value = float(values.mean())
    if abs(mean_value) < 1e-12:
        return np.nan
    return float(values.std() / mean_value)


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


def _make_edge_signal_with_imagej(ij: Any, signal_path: Path, temp_dir: Path) -> Path:
    """Create an ImageJ Find Edges image and return its temporary path."""
    edge_path = temp_dir / "signal_find_edges.tif"
    _run_macro_quiet(
        ij,
        _FIND_EDGES_MACRO,
        args={"input_path": str(signal_path), "output_path": str(edge_path)},
    )
    if not edge_path.exists():
        raise RuntimeError(f"ImageJ Find Edges failed: {edge_path}")
    return edge_path


def _load_debris_config() -> dict[str, Any]:
    """讀取 debris feature 的門檻設定。

    Returns:
        dict[str, Any]: 合併預設值與 JSON 設定檔後的 debris 參數。
    """
    config = dict(DEFAULT_DEBRIS_CONFIG)
    config_path = Path(os.environ.get("KI67_DEBRIS_CONFIG_PATH", DEBRIS_CONFIG_PATH))
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception as exc:
            print(f"[WARN] Failed to read debris config {config_path}: {exc}")
    return config


def _find_ki67_image_for_pc(pc_path: Path) -> Path | None:
    """尋找 PC 影像對應的 Ki67 影像。

    Args:
        pc_path (Path): PC 影像路徑。

    Returns:
        Path | None: 對應 Ki67 影像路徑；找不到時回傳 `None`。
    """
    dataset_dir = pc_path.parent.parent if pc_path.parent.name.upper() == "PC" else pc_path.parent
    ki67_dir = dataset_dir / "KI67"
    if not ki67_dir.exists():
        return None

    mapping_path = dataset_dir / "image_mapping.csv"
    if mapping_path.exists():
        try:
            mapping = pd.read_csv(mapping_path)
            if "PC_Name" in mapping.columns and "KI67_Name" in mapping.columns:
                matched = mapping[mapping["PC_Name"].astype(str) == pc_path.name]
                if not matched.empty:
                    ki67_name = str(matched.iloc[0]["KI67_Name"]).strip()
                    ki67_path = ki67_dir / ki67_name
                    if ki67_name and ki67_path.exists():
                        return ki67_path
        except Exception as exc:
            print(f"[WARN] Failed to read image mapping for debris: {exc}")

    pc_images = list_files(pc_path.parent, [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    ki67_images = list_files(ki67_dir, [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    stems = [path.stem for path in pc_images]
    if pc_path.stem in stems:
        idx = stems.index(pc_path.stem)
        if idx < len(ki67_images):
            return ki67_images[idx]
    return None


def _debris_background_mask(cell_union_mask: np.ndarray, cell_dilate_px: int) -> np.ndarray:
    """由細胞 union mask 建立 debris 分析用 background mask。

    Args:
        cell_union_mask (np.ndarray): 所有細胞 ROI 的 union mask。
        cell_dilate_px (int): 排除細胞邊緣時的 dilation 半徑。

    Returns:
        np.ndarray: background 區域布林遮罩。
    """
    cell_mask = cell_union_mask.astype(np.uint8)
    if cell_dilate_px > 0 and np.any(cell_mask):
        kernel_size = int(cell_dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cell_mask = cv2.dilate(cell_mask, kernel, iterations=1)
    return cell_mask == 0


def _parse_debris_particle_table(table_path: Path) -> dict[str, Any]:
    """解析 ImageJ Analyze Particles 輸出的 debris 表格。

    Args:
        table_path (Path): ImageJ 輸出的 TSV 結果路徑。

    Returns:
        dict[str, Any]: debris 面積陣列與 centroid 座標陣列。
    """
    areas: list[float] = []
    centroids: list[list[float]] = []
    if not table_path.exists():
        return {"areas": np.asarray([], dtype=np.float64), "centroids": np.empty((0, 2), dtype=np.float64)}

    with table_path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            try:
                area = float(row.get("Area", "nan"))
                x = float(row.get("X", "nan"))
                y = float(row.get("Y", "nan"))
            except Exception:
                continue
            if not np.isfinite(area) or area <= 0:
                continue
            areas.append(area)
            centroids.append([x, y])

    return {
        "areas": np.asarray(areas, dtype=np.float64),
        "centroids": np.asarray(centroids, dtype=np.float64).reshape(-1, 2),
    }


def _nearest_distances_to_debris(
    cell_centroids: np.ndarray, debris_centroids: np.ndarray
) -> np.ndarray:
    """計算每顆細胞到最近 debris centroid 的距離。

    Args:
        cell_centroids (np.ndarray): 細胞 centroid 座標，shape 為 `(N, 2)`。
        debris_centroids (np.ndarray): debris centroid 座標，shape 為 `(M, 2)`。

    Returns:
        np.ndarray: 每顆細胞的最近 debris 距離。
    """
    nearest = np.full(len(cell_centroids), np.nan, dtype=np.float64)
    if len(cell_centroids) == 0 or len(debris_centroids) == 0:
        return nearest
    valid_cells = np.isfinite(cell_centroids).all(axis=1)
    valid_debris = np.isfinite(debris_centroids).all(axis=1)
    if not np.any(valid_cells) or not np.any(valid_debris):
        return nearest
    debris_points = debris_centroids[valid_debris]
    for idx in np.where(valid_cells)[0]:
        diffs = debris_points - cell_centroids[idx]
        nearest[idx] = float(np.sqrt(np.sum(diffs * diffs, axis=1)).min())
    return nearest


def _empty_debris_feature_values(n_cells: int) -> list[list[float]]:
    """回傳指定細胞數的 debris 空值矩陣。"""
    return [[np.nan] * len(DEBRIS_FEATURE_PARAMETER_COLUMNS) for _ in range(n_cells)]


def _debris_feature_values_for_cells(
    ij: Any,
    pc_path: Path,
    cell_union_mask: np.ndarray,
    cell_centroids: np.ndarray,
    median_nucleus_area: float,
    temp_dir: Path,
) -> list[list[float]]:
    """使用 ImageJ threshold 與 Analyze Particles 提取 debris 特徵。

    Args:
        ij (Any): 已初始化的 PyImageJ 物件。
        pc_path (Path): 目前 PC 影像路徑，用於定位對應 Ki67/debris 影像。
        cell_union_mask (np.ndarray): 所有細胞 ROI 的 union mask。
        cell_centroids (np.ndarray): 每顆細胞 centroid 座標。
        median_nucleus_area (float): 同張影像 nucleus 面積中位數。
        temp_dir (Path): 暫存檔資料夾。

    Returns:
        list[list[float]]: 每顆細胞的 debris feature parameter 清單。
    """
    n_cells = len(cell_centroids)
    if n_cells == 0:
        return []

    config = _load_debris_config()
    ki67_path = _find_ki67_image_for_pc(pc_path)
    if ki67_path is None:
        return _empty_debris_feature_values(n_cells)

    min_size = float(config.get("min_size_px", DEFAULT_DEBRIS_CONFIG["min_size_px"]))
    max_fraction = float(
        config.get(
            "max_size_fraction_of_median_nucleus_area",
            DEFAULT_DEBRIS_CONFIG["max_size_fraction_of_median_nucleus_area"],
        )
    )
    if not np.isfinite(median_nucleus_area) or median_nucleus_area <= 0:
        return _empty_debris_feature_values(n_cells)
    max_size = float(median_nucleus_area * max_fraction)
    if max_size < min_size:
        return _empty_debris_feature_values(n_cells)

    cell_dilate_px = int(config.get("cell_dilate_px", DEFAULT_DEBRIS_CONFIG["cell_dilate_px"]))
    background_mask = _debris_background_mask(cell_union_mask, cell_dilate_px)
    background_area = int(np.count_nonzero(background_mask))
    if background_area <= 0:
        return _empty_debris_feature_values(n_cells)

    signal_path = temp_dir / f"debris_signal{ki67_path.suffix.lower()}"
    background_mask_path = temp_dir / "debris_background_mask.tif"
    debris_mask_path = temp_dir / "debris_mask.png"
    particle_table_path = temp_dir / "debris_particles.tsv"
    try:
        copyfile(ki67_path, signal_path)
        tifffile.imwrite(background_mask_path, (background_mask.astype(np.uint8) * 255))

        _run_macro_quiet(
            ij,
            _DEBRIS_IMAGEJ_MACRO,
            args={
                "signal_path": str(signal_path),
                "background_mask_path": str(background_mask_path),
                "debris_mask_path": str(debris_mask_path),
                "particle_table_path": str(particle_table_path),
                "threshold_method": str(config.get("threshold_method", "Triangle")),
                "min_particle_area": float(min_size),
                "max_particle_area": float(max_size),
                "rolling_ball_radius": float(config.get("rolling_ball_radius", 50.0)),
                "subtract_background": bool(config.get("subtract_background", False)),
                "watershed": bool(config.get("watershed", False)),
            },
        )
    except Exception as exc:
        print(f"[WARN] Debris feature extraction failed for {pc_path.name}: {exc}")
        return _empty_debris_feature_values(n_cells)

    particles = _parse_debris_particle_table(particle_table_path)
    areas = particles["areas"]
    debris_centroids = particles["centroids"]
    debris_count = float(len(areas))
    debris_area_sum = float(np.sum(areas)) if len(areas) else 0.0
    debris_area_fraction = float(debris_area_sum / background_area)
    debris_mean_area = float(np.mean(areas)) if len(areas) else np.nan
    debris_density = float(debris_count / background_area)
    nearest_distances = _nearest_distances_to_debris(cell_centroids, debris_centroids)

    return [
        [
            debris_count,
            debris_area_fraction,
            float(nearest_distances[idx]),
            debris_mean_area,
            debris_density,
        ]
        for idx in range(n_cells)
    ]


def _derived_feature_parameter_values(
    nuc_m: dict[str, float],
    cyto_m: dict[str, float],
    nuc_g: dict[str, float],
    cell_g: dict[str, float],
) -> list[float]:
    """Build cross-ROI parameters from ImageJ-produced measurements."""
    dx = _safe_difference(nuc_g["x_centroid"], cell_g["x_centroid"])
    dy = _safe_difference(nuc_g["y_centroid"], cell_g["y_centroid"])
    whole_cell_radius = (
        math.sqrt(cell_g["area"] / math.pi)
        if not np.isnan(cell_g["area"]) and cell_g["area"] > 0
        else np.nan
    )
    centroid_offset = (
        _safe_divide(math.hypot(dx, dy), whole_cell_radius)
        if not np.isnan(dx) and not np.isnan(dy)
        else np.nan
    )
    return [
        _safe_divide(nuc_m["mean"], cyto_m["mean"]),
        _safe_divide(nuc_m["intden"], cyto_m["intden"]),
        _safe_divide(nuc_m["raw_intden"], cyto_m["raw_intden"]),
        _safe_difference(nuc_m["entropy"], cyto_m["entropy"]),
        _safe_difference(nuc_m["cv"], cyto_m["cv"]),
        centroid_offset,
    ]


def _halo_feature_parameter_values(
    ij: Any,
    signal_path: Path,
    edge_signal_path: Path,
    cell_mask: np.ndarray,
    background_m: dict[str, float],
    temp_dir: Path,
    roi_id: int,
) -> list[float]:
    """Measure halo and edge feature parameters using ImageJ measurements."""
    cell_u8 = cell_mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    inner_eroded = cv2.erode(cell_u8, kernel, iterations=2).astype(bool)
    inner_ring = np.logical_and(cell_mask, np.logical_not(inner_eroded))
    outer_dilated = cv2.dilate(cell_u8, kernel, iterations=2).astype(bool)
    outer_ring = np.logical_and(outer_dilated, np.logical_not(cell_mask))
    boundary_dilated = cv2.dilate(cell_u8, kernel, iterations=1).astype(bool)
    boundary_eroded = cv2.erode(cell_u8, kernel, iterations=1).astype(bool)
    boundary_ring = np.logical_and(boundary_dilated, np.logical_not(boundary_eroded))

    outer_m = _measure_roi_with_imagej(
        ij, signal_path, outer_ring, temp_dir, f"roi_{roi_id}_halo_outer"
    )
    inner_m = _measure_roi_with_imagej(
        ij, signal_path, inner_ring, temp_dir, f"roi_{roi_id}_halo_inner"
    )
    edge_m = _measure_roi_with_imagej(
        ij, edge_signal_path, boundary_ring, temp_dir, f"roi_{roi_id}_edge"
    )

    threshold = (
        background_m["mean"] + 1.5 * background_m["std"]
        if not np.isnan(background_m["mean"]) and not np.isnan(background_m["std"])
        else np.nan
    )
    halo_width = np.nan
    previous = cell_mask.copy()
    for width in [3, 5, 7, 9, 11, 13, 15]:
        dilated = cv2.dilate(cell_u8, kernel, iterations=width).astype(bool)
        ring = np.logical_and(dilated, np.logical_not(previous))
        previous = dilated
        if not np.any(ring):
            continue
        ring_m = _measure_roi_with_imagej(
            ij, signal_path, ring, temp_dir, f"roi_{roi_id}_halo_width_{width}"
        )
        if not np.isnan(threshold) and not np.isnan(ring_m["mean"]) and ring_m["mean"] <= threshold:
            halo_width = float(width)
            break

    return [
        outer_m["mean"],
        outer_m["std"],
        inner_m["mean"],
        inner_m["std"],
        _safe_abs_difference(outer_m["mean"], inner_m["mean"]),
        halo_width,
        edge_m["mean"],
    ]


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
    eccentricity = m.get("eccentricity", np.nan)

    if (
        np.isnan(aspect_ratio)
        and not np.isnan(major)
        and not np.isnan(minor)
        and minor > 0
    ):
        aspect_ratio = major / minor

    if np.isnan(roundness) and not np.isnan(area) and not np.isnan(major) and major > 0:
        roundness = (4 * area) / (np.pi * major**2)

    if (
        np.isnan(eccentricity)
        and not np.isnan(major)
        and not np.isnan(minor)
        and major > 0
    ):
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))

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
        "eccentricity": eccentricity,
        "roundness": roundness,
        "circularity": circularity,
        "sphericity": sphericity,
        "roughness": roughness,
        "x_centroid": m["x_centroid"],
        "y_centroid": m["y_centroid"],
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


def _parse_outline_line(line: str) -> Optional[np.ndarray]:
    """Parse one outline line; return None for missing or malformed outlines."""
    line = line.strip()
    if not line or line == "-1,-1":
        return None

    try:
        coords = list(map(int, line.split(",")))
    except Exception:
        return None

    if len(coords) < 6 or len(coords) % 2:
        return None

    points = np.array(coords, dtype=np.int32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    return points


def _parse_outline_pairs(
    outlines_txt: Union[str, Path],
    include_unpaired: bool = False,
) -> list[tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Parse merged outline pairs, optionally keeping nuc-only/cyto-only rows."""
    with open(outlines_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    pairs: list[tuple[int, Optional[np.ndarray], Optional[np.ndarray]]] = []
    num_pairs = len(lines) // 2
    for i in range(num_pairs):
        nuc = _parse_outline_line(lines[2 * i])
        cyto = _parse_outline_line(lines[2 * i + 1])
        if nuc is None and cyto is None:
            continue
        if not include_unpaired and (nuc is None or cyto is None):
            continue
        pairs.append((i + 1, nuc, cyto))
    return pairs


def _empty_geometry() -> dict[str, float]:
    """回傳 geometry 欄位的空值字典。"""
    return {
        "area": np.nan,
        "perimeter": np.nan,
        "convex_perimeter": np.nan,
        "circular_diameter": np.nan,
        "feret_length": np.nan,
        "feret_width": np.nan,
        "aspect_ratio": np.nan,
        "eccentricity": np.nan,
        "roundness": np.nan,
        "circularity": np.nan,
        "sphericity": np.nan,
        "roughness": np.nan,
        "x_centroid": np.nan,
        "y_centroid": np.nan,
    }


def _outline_touches_image_edge(
    points_xy: Optional[np.ndarray],
    shape: tuple[int, int],
    margin: int = 1,
    min_edge_points: int = 2,
) -> bool:
    """判斷 outline 是否觸碰影像邊界。

    Args:
        points_xy (Optional[np.ndarray]): outline 座標，shape 為 `(N, 2)`。
        shape (tuple[int, int]): 影像 `(height, width)`。
        margin (int): 距離邊界多少 pixel 內視為切邊。
        min_edge_points (int): 同一側至少幾個點才判定為切邊。

    Returns:
        bool: 是否觸碰影像邊界。
    """
    if points_xy is None or points_xy.shape[0] < 3:
        return False

    height, width = shape
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    side_counts = (
        np.count_nonzero(x <= margin),
        np.count_nonzero(y <= margin),
        np.count_nonzero(x >= width - 1 - margin),
        np.count_nonzero(y >= height - 1 - margin),
    )
    return any(count >= min_edge_points for count in side_counts)


def _classify_outline_status(
    nuc_xy: Optional[np.ndarray],
    cyto_xy: Optional[np.ndarray],
    shape: tuple[int, int],
) -> str:
    """依 nucleus/cytoplasm outline 完整性分類 cell_status。

    Args:
        nuc_xy (Optional[np.ndarray]): nucleus outline 座標。
        cyto_xy (Optional[np.ndarray]): cytoplasm outline 座標。
        shape (tuple[int, int]): 影像 `(height, width)`。

    Returns:
        str: `full_cell`、`nuc_only`、`cyto_only`、`nuc_cut`、`cyto_cut` 或 `both_cut`。
    """
    has_nuc = nuc_xy is not None
    has_cyto = cyto_xy is not None
    nuc_cut = _outline_touches_image_edge(nuc_xy, shape)
    cyto_cut = _outline_touches_image_edge(cyto_xy, shape)

    if has_nuc and not has_cyto:
        return "nuc_cut" if nuc_cut else "nuc_only"
    if has_cyto and not has_nuc:
        return "cyto_only"
    if nuc_cut and cyto_cut:
        return "both_cut"
    if nuc_cut:
        return "nuc_cut"
    if cyto_cut:
        return "cyto_cut"
    if has_nuc and has_cyto:
        return "full_cell"
    return "empty"


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


def add_outline_status_column(
    final_csv: Union[str, Path],
    outlines_txt: Union[str, Path],
    img_path: Union[str, Path],
) -> None:
    """Append per-cell outline completeness/cut status to a final CSV."""
    final_csv = Path(final_csv)
    img_path = Path(img_path)
    signal = _read_gray_float(img_path)
    shape = signal.shape
    status_by_cell_id = {
        f"{img_path.stem}_{roi_id}": _classify_outline_status(nuc_xy, cyto_xy, shape)
        for roi_id, nuc_xy, cyto_xy in _parse_outline_pairs(
            outlines_txt, include_unpaired=True
        )
    }

    df = pd.read_csv(final_csv)
    df["cell_status"] = df["Cell_ID"].map(status_by_cell_id).fillna("unknown")
    status = df.pop("cell_status")
    df["cell_status"] = status
    df.to_csv(final_csv, index=False)


def _empty_extra_feature_values() -> list[float]:
    """回傳 cell-level 額外 feature 參數的空值清單。"""
    return [np.nan] * len(EXTRA_FEATURE_PARAMETER_COLUMNS)


def merged_excel(input: Union[str, Path], output: Union[str, Path]):
    """Merge nuc/cyto long-form rows into wide-form feature parameters.

    ROI-specific parameters keep _nuc/_cyto suffixes. Cell-level parameters are
    stored once without ROI suffixes to avoid empty duplicate columns.
    """
    df = pd.read_csv(input)
    df["ROI_Type"] = df["Cell_ID"].apply(lambda x: x.split("_")[-1])
    df["Cell_ID"] = df["Cell_ID"].apply(lambda x: "_".join(x.split("_")[:-1]))

    nucleus_df = df[df["ROI_Type"].str.lower() == "nuc"].copy()
    cyto_df = df[df["ROI_Type"].str.lower() == "cyto"].copy()

    nucleus_df = nucleus_df.drop(columns=["ROI_Type"]).set_index("Cell_ID")
    cyto_df = cyto_df.drop(columns=["ROI_Type"]).set_index("Cell_ID")

    cell_level_cols = [
        col
        for col in CELL_LEVEL_FEATURE_PARAMETER_COLUMNS
        if col in cyto_df.columns
    ]
    cell_level_df = cyto_df[cell_level_cols].copy() if cell_level_cols else None

    nucleus_df = nucleus_df.drop(columns=CELL_LEVEL_FEATURE_PARAMETER_COLUMNS, errors="ignore")
    cyto_df = cyto_df.drop(columns=cell_level_cols, errors="ignore")

    nucleus_df = nucleus_df.add_suffix("_nuc")
    cyto_df = cyto_df.add_suffix("_cyto")

    parts = [nucleus_df, cyto_df]
    if cell_level_df is not None:
        parts.append(cell_level_df)
    merged_df = pd.concat(parts, axis=1).reset_index()
    merged_df.to_csv(output, index=False)


def _shape_complexity_feature_values(cell_mask: np.ndarray) -> list[float]:
    """計算 whole-cell 邊界形狀複雜度參數。

    Args:
        cell_mask (np.ndarray): whole-cell 布林遮罩。

    Returns:
        list[float]: protrusion、convex defect、fractal dimension 與 inflection 參數。
    """
    cell_u8 = (cell_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(cell_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [np.nan] * len(SHAPE_COMPLEXITY_FEATURE_PARAMETER_COLUMNS)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return [np.nan] * len(SHAPE_COMPLEXITY_FEATURE_PARAMETER_COLUMNS)

    protrusion_count = np.nan
    mean_depth = np.nan
    max_depth = np.nan
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) >= 3 and len(contour) >= 4:
            defects = cv2.convexityDefects(contour, hull_indices)
            protrusion_count = 0
            if defects is not None and len(defects) > 0:
                depths = defects[:, 0, 3].astype(np.float64) / 256.0
                depth_threshold = max(3.0, 0.03 * math.sqrt(area / math.pi) * 2.0)
                valid_depths = depths[depths > depth_threshold]
                protrusion_count = int(len(valid_depths))
                if len(valid_depths) > 0:
                    mean_depth = float(valid_depths.mean())
                    max_depth = float(valid_depths.max())
    except cv2.error as exc:
        print(f"[WARN] Convex defect features skipped for self-intersecting contour: {exc}")

    boundary = cv2.morphologyEx(cell_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    box_sizes = np.asarray([2, 4, 8, 16, 32], dtype=int)
    counts = []
    height, width = boundary.shape
    for box_size in box_sizes:
        pad_h = (box_size - height % box_size) % box_size
        pad_w = (box_size - width % box_size) % box_size
        padded = np.pad(boundary, ((0, pad_h), (0, pad_w)), mode="constant")
        blocks = padded.reshape(
            padded.shape[0] // box_size,
            box_size,
            padded.shape[1] // box_size,
            box_size,
        )
        counts.append(int(np.any(blocks, axis=(1, 3)).sum()))
    valid = np.asarray(counts) > 0
    fractal_dimension = (
        float(np.polyfit(np.log(1.0 / box_sizes[valid]), np.log(np.asarray(counts)[valid]), 1)[0])
        if valid.sum() >= 2
        else np.nan
    )

    points = contour[:, 0, :].astype(np.float64)
    if len(points) >= 8:
        prev_points = np.roll(points, 1, axis=0)
        next_points = np.roll(points, -1, axis=0)
        v1 = points - prev_points
        v2 = next_points - points
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        signs = np.sign(cross)
        nonzero = signs != 0
        signs = signs[nonzero]
        boundary_inflection_count = int(np.sum(signs != np.roll(signs, 1))) if len(signs) else 0
    else:
        boundary_inflection_count = np.nan

    return [
        float(protrusion_count) if not np.isnan(protrusion_count) else np.nan,
        mean_depth,
        max_depth,
        fractal_dimension,
        float(boundary_inflection_count) if not np.isnan(boundary_inflection_count) else np.nan,
    ]


def _distance_matrix(points: np.ndarray) -> np.ndarray:
    """計算點集合的兩兩歐氏距離矩陣。"""
    diffs = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def _cluster_sizes_from_distances(distances: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    """用距離門檻建立連通群集並回傳每點 cluster size。

    Args:
        distances (np.ndarray): 點與點之間的距離矩陣。
        eps (float): 視為同群的距離門檻。

    Returns:
        tuple[np.ndarray, float]: 每個點所屬 cluster 大小與最大 cluster 比例。
    """
    n = distances.shape[0]
    if n == 0:
        return np.asarray([], dtype=int), np.nan
    visited = np.zeros(n, dtype=bool)
    cluster_sizes = np.ones(n, dtype=int)
    largest = 1
    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        component = []
        visited[start] = True
        while queue:
            idx = queue.pop()
            component.append(idx)
            neighbors = np.where(distances[idx] <= eps)[0]
            for neighbor in neighbors:
                if neighbor == idx or visited[neighbor]:
                    continue
                visited[neighbor] = True
                queue.append(int(neighbor))
        size = len(component)
        largest = max(largest, size)
        for idx in component:
            cluster_sizes[idx] = size
    return cluster_sizes, float(largest / n)


def _scale_0_1(value: float, low: float, high: float) -> float:
    """將數值依指定上下界縮放到 0 到 1。"""
    if np.isnan(value) or high <= low:
        return np.nan
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _mitosis_feature_values(
    cell_g: dict[str, float],
    cyto_g: dict[str, float],
    nuc_m: dict[str, float],
    shape_values: list[float],
    median_area: float,
    nearest_neighbor: float,
    nearest_neighbor_index: int,
    cell_areas: np.ndarray,
) -> list[float]:
    """依形狀、紋理與鄰近關係計算 mitosis proxy 參數。

    Args:
        cell_g (dict[str, float]): whole-cell 幾何量測結果。
        cyto_g (dict[str, float]): cytoplasm 幾何量測結果。
        nuc_m (dict[str, float]): nucleus ImageJ 強度量測結果。
        shape_values (list[float]): shape complexity 參數。
        median_area (float): 同張影像 whole-cell 面積中位數。
        nearest_neighbor (float): 最近鄰細胞距離。
        nearest_neighbor_index (int): 最近鄰細胞索引。
        cell_areas (np.ndarray): 全影像 cell area 陣列。

    Returns:
        list[float]: `Mitotic Score`、`Daughter Pair Flag`、`Protrusion Retraction Score`。
    """
    area = cell_g["area"]
    circularity = cyto_g["sphericity"]
    area_norm = _safe_divide(area, median_area)
    small_area_score = 1.0 - _scale_0_1(area_norm, 0.5, 1.2) if not np.isnan(area_norm) else np.nan
    round_score = _scale_0_1(circularity, 0.65, 0.95)
    uniform_texture_score = 1.0 - _scale_0_1(nuc_m["entropy"], 4.0, 7.0)
    valid_score_parts = [
        value
        for value in [round_score, small_area_score, uniform_texture_score]
        if not np.isnan(value)
    ]
    mitotic_score = float(np.mean(valid_score_parts)) if valid_score_parts else np.nan

    daughter_pair_flag = 0.0
    if (
        nearest_neighbor_index >= 0
        and not np.isnan(nearest_neighbor)
        and not np.isnan(median_area)
        and median_area > 0
    ):
        mean_diameter = 2.0 * math.sqrt(median_area / math.pi)
        neighbor_area = cell_areas[nearest_neighbor_index]
        area_delta = abs(area - neighbor_area) / max((area + neighbor_area) / 2.0, 1e-12)
        if nearest_neighbor < 0.5 * mean_diameter and area_delta < 0.25:
            daughter_pair_flag = 1.0

    protrusion_count = shape_values[0]
    protrusion_score = 1.0 - _scale_0_1(protrusion_count, 0.0, 6.0)
    retraction_parts = [
        value
        for value in [round_score, small_area_score, protrusion_score]
        if not np.isnan(value)
    ]
    protrusion_retraction_score = float(np.mean(retraction_parts)) if retraction_parts else np.nan

    return [mitotic_score, daughter_pair_flag, protrusion_retraction_score]


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

    signal = _read_gray_float(img_path)
    shape = signal.shape
    ij = _get_pyimagej()
    pairs = _parse_outline_pairs(outlines_txt, include_unpaired=True)
    rows: list[list[float | str]] = []

    with tempfile.TemporaryDirectory(prefix="pyimagej_param_") as tmp:
        tmp_dir = Path(tmp)
        signal_path = tmp_dir / f"{img_path.stem}_pc_signal.tif"
        tifffile.imwrite(signal_path, signal)
        edge_signal_path = _make_edge_signal_with_imagej(ij, signal_path, tmp_dir)

        mask_records: list[dict[str, Any]] = []
        cell_union_mask = np.zeros(shape, dtype=bool)
        for roi_id, nuc_xy, cyto_xy in pairs:
            nuc_mask = np.zeros(shape, dtype=bool)
            cell_mask = np.zeros(shape, dtype=bool)
            cyto_mask = np.zeros(shape, dtype=bool)

            if nuc_xy is not None:
                nuc_mask = _polygon_to_mask(nuc_xy, shape)
            if cyto_xy is not None:
                cell_mask = _polygon_to_mask(cyto_xy, shape)

            has_nuc = bool(np.any(nuc_mask))
            has_cyto = bool(np.any(cell_mask))
            if has_cyto:
                cyto_mask = (
                    np.logical_and(cell_mask, np.logical_not(nuc_mask))
                    if has_nuc
                    else cell_mask.copy()
                )

            if not has_nuc and not has_cyto:
                continue

            reference_mask = cell_mask if has_cyto else nuc_mask
            cell_union_mask |= reference_mask
            mask_records.append(
                {
                    "roi_id": roi_id,
                    "nuc_mask": nuc_mask,
                    "cyto_mask": cyto_mask,
                    "cell_mask": cell_mask,
                    "reference_mask": reference_mask,
                    "has_nuc": has_nuc,
                    "has_cyto": has_cyto,
                }
            )

        background_mask = np.logical_not(cell_union_mask)
        background_m = _measure_roi_with_imagej(
            ij, signal_path, background_mask, tmp_dir, "image_background"
        )

        cell_records: list[dict[str, Any]] = []
        for record in mask_records:
            roi_id = int(record["roi_id"])
            nuc_mask = record["nuc_mask"]
            cyto_mask = record["cyto_mask"]
            cell_mask = record["cell_mask"]
            reference_mask = record["reference_mask"]
            has_nuc = bool(record["has_nuc"])
            has_cyto = bool(record["has_cyto"])
            has_cyto_roi = has_cyto and bool(np.any(cyto_mask))

            nuc_m = (
                _measure_roi_with_imagej(
                    ij, signal_path, nuc_mask, tmp_dir, f"roi_{roi_id}_nuc"
                )
                if has_nuc
                else _empty_imagej_measurements()
            )
            cyto_m = (
                _measure_roi_with_imagej(
                    ij, signal_path, cyto_mask, tmp_dir, f"roi_{roi_id}_cyto"
                )
                if has_cyto_roi
                else _empty_imagej_measurements()
            )
            cell_m = (
                _measure_roi_with_imagej(
                    ij, signal_path, reference_mask, tmp_dir, f"roi_{roi_id}_cell"
                )
                if bool(np.any(reference_mask))
                else _empty_imagej_measurements()
            )

            nuc_g = _geometry_from_imagej_measurements(nuc_m)
            cyto_g = _geometry_from_imagej_measurements(cyto_m)
            cell_g = _geometry_from_imagej_measurements(cell_m)
            nuc_intensity_features = (
                _intensity_feature_parameter_values(nuc_m)
                if has_nuc
                else _intensity_feature_parameter_values(_empty_imagej_measurements())
            )
            cyto_intensity_features = (
                _intensity_feature_parameter_values(cyto_m)
                if has_cyto_roi
                else _intensity_feature_parameter_values(_empty_imagej_measurements())
            )
            nuc_texture_features = (
                _texture_feature_parameter_values(ij, signal, nuc_mask, erode_px=1)
                + _lbp_feature_parameter_values(
                    ij,
                    signal,
                    nuc_mask,
                    erode_px=1,
                    temp_dir=tmp_dir,
                    slot=f"roi_{roi_id}_nuc",
                )
                if has_nuc
                else _empty_texture_feature_values()
            )
            cyto_texture_features = (
                _texture_feature_parameter_values(ij, signal, cyto_mask, erode_px=2)
                + _lbp_feature_parameter_values(
                    ij,
                    signal,
                    cyto_mask,
                    erode_px=2,
                    temp_dir=tmp_dir,
                    slot=f"roi_{roi_id}_cyto",
                )
                if has_cyto_roi
                else _empty_texture_feature_values()
            )
            derived_values = (
                _derived_feature_parameter_values(nuc_m, cyto_m, nuc_g, cell_g)
                if has_nuc and has_cyto_roi
                else [np.nan] * len(DERIVED_FEATURE_PARAMETER_COLUMNS)
            )
            halo_values = (
                _halo_feature_parameter_values(
                    ij,
                    signal_path,
                    edge_signal_path,
                    cell_mask,
                    background_m,
                    tmp_dir,
                    roi_id,
                )
                if has_cyto
                else [np.nan] * len(HALO_FEATURE_PARAMETER_COLUMNS)
            )
            shape_values = (
                _shape_complexity_feature_values(cell_mask)
                if has_cyto
                else [np.nan] * len(SHAPE_COMPLEXITY_FEATURE_PARAMETER_COLUMNS)
            )

            karyoplasmic_ratio = np.nan
            if (
                not np.isnan(cyto_g["area"])
                and not np.isnan(nuc_g["area"])
                and cyto_g["area"] > 0
            ):
                # Use nucleus area / cytoplasm area so the value matches the intended
                # biological interpretation of a nucleocytoplasmic ratio.
                karyoplasmic_ratio = float(nuc_g["area"] / cyto_g["area"])

            cell_records.append(
                {
                    "roi_id": roi_id,
                    "nuc_g": nuc_g,
                    "cyto_g": cyto_g,
                    "cell_g": cell_g,
                    "nuc_m": nuc_m,
                    "nuc_intensity_features": nuc_intensity_features,
                    "cyto_intensity_features": cyto_intensity_features,
                    "nuc_texture_features": nuc_texture_features,
                    "cyto_texture_features": cyto_texture_features,
                    "derived_values": derived_values,
                    "halo_values": halo_values,
                    "shape_values": shape_values,
                    "karyoplasmic_ratio": karyoplasmic_ratio,
                    "has_nuc": has_nuc,
                    "has_cyto": has_cyto,
                }
            )

        cell_areas = np.asarray([r["cell_g"]["area"] for r in cell_records], dtype=np.float64)
        cell_circularities = np.asarray([r["cyto_g"]["sphericity"] for r in cell_records], dtype=np.float64)
        centroids = np.asarray(
            [
                [r["cell_g"]["x_centroid"], r["cell_g"]["y_centroid"]]
                for r in cell_records
            ],
            dtype=np.float64,
        )
        valid_centroids = np.isfinite(centroids).all(axis=1) if len(centroids) else np.asarray([], dtype=bool)
        median_area = float(np.nanmedian(cell_areas)) if len(cell_areas) else np.nan
        nucleus_areas = np.asarray([r["nuc_g"]["area"] for r in cell_records], dtype=np.float64)
        median_nucleus_area = (
            float(np.nanmedian(nucleus_areas)) if len(nucleus_areas) else np.nan
        )
        median_diameter = (
            2.0 * math.sqrt(median_area / math.pi)
            if not np.isnan(median_area) and median_area > 0
            else np.nan
        )
        image_confluency = (
            float(np.nansum(cell_areas) / (shape[0] * shape[1]))
            if shape[0] > 0 and shape[1] > 0
            else np.nan
        )
        population_area_cv = _safe_cv(cell_areas)
        population_circularity_cv = _safe_cv(cell_circularities)

        nearest_distances = np.full(len(cell_records), np.nan, dtype=np.float64)
        nearest_indices = np.full(len(cell_records), -1, dtype=int)
        local_counts = np.zeros(len(cell_records), dtype=int)
        cluster_sizes = np.ones(len(cell_records), dtype=int)
        largest_cluster_ratio = np.nan
        if len(cell_records) >= 2 and valid_centroids.sum() >= 2:
            valid_indices = np.where(valid_centroids)[0]
            valid_points = centroids[valid_indices]
            distances = _distance_matrix(valid_points)
            np.fill_diagonal(distances, np.inf)
            valid_nearest_local = np.argmin(distances, axis=1)
            nearest_distances[valid_indices] = distances[np.arange(len(valid_indices)), valid_nearest_local]
            nearest_indices[valid_indices] = valid_indices[valid_nearest_local]
            if not np.isnan(median_diameter):
                radius = 2.0 * median_diameter
                local_counts[valid_indices] = (distances <= radius).sum(axis=1)
                valid_cluster_sizes, largest_cluster_ratio = _cluster_sizes_from_distances(
                    distances,
                    eps=2.5 * median_diameter,
                )
                cluster_sizes[valid_indices] = valid_cluster_sizes

        debris_values_by_cell = _debris_feature_values_for_cells(
            ij=ij,
            pc_path=img_path,
            cell_union_mask=cell_union_mask,
            cell_centroids=centroids,
            median_nucleus_area=median_nucleus_area,
            temp_dir=tmp_dir,
        )

        for idx, record in enumerate(cell_records):
            nuc_g = record["nuc_g"]
            cyto_g = record["cyto_g"]
            cell_g = record["cell_g"]
            nn_distance = float(nearest_distances[idx])
            nn_distance_norm = _safe_divide(nn_distance, median_diameter)
            local_density = (
                _safe_divide(float(local_counts[idx]), math.pi * (2.0 * median_diameter) ** 2)
                if not np.isnan(median_diameter)
                else np.nan
            )
            cluster_size = float(cluster_sizes[idx])
            cluster_size_norm = _safe_divide(cluster_size, float(len(cell_records)))
            population_values = [
                image_confluency,
                population_area_cv,
                population_circularity_cv,
                nn_distance,
                nn_distance_norm,
                float(local_counts[idx]),
                local_density,
                cluster_size,
                cluster_size_norm,
                largest_cluster_ratio,
            ]
            mitosis_values = _mitosis_feature_values(
                cell_g=cell_g,
                cyto_g=cyto_g,
                nuc_m=record["nuc_m"],
                shape_values=record["shape_values"],
                median_area=median_area,
                nearest_neighbor=nn_distance,
                nearest_neighbor_index=int(nearest_indices[idx]),
                cell_areas=cell_areas,
            )
            extra_values = (
                record["derived_values"]
                + record["halo_values"]
                + population_values
                + record["shape_values"]
                + debris_values_by_cell[idx]
                + mitosis_values
            )

            roi_id = int(record["roi_id"])
            if record["has_nuc"]:
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
                        nuc_g["eccentricity"],
                        nuc_g["roundness"],
                        nuc_g["circularity"],
                        nuc_g["sphericity"],
                        nuc_g["roughness"],
                        np.nan,
                        *record["nuc_intensity_features"],
                        *record["nuc_texture_features"],
                        *_empty_extra_feature_values(),
                    ]
                )
            if record["has_cyto"]:
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
                        cyto_g["eccentricity"],
                        cyto_g["roundness"],
                        cyto_g["circularity"],
                        cyto_g["sphericity"],
                        cyto_g["roughness"],
                        record["karyoplasmic_ratio"],
                        *record["cyto_intensity_features"],
                        *record["cyto_texture_features"],
                        *extra_values,
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
            "Eccentricity",
            "Roundness",
            "Circularity",
            "Sphericity",
            "Roughness",
            "Karyoplasmic Ratio",
            *INTENSITY_FEATURE_PARAMETER_COLUMNS,
            *TEXTURE_FEATURE_PARAMETER_COLUMNS,
            *EXTRA_FEATURE_PARAMETER_COLUMNS,
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

    signal = _read_gray_float(img_path)
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

    signal = _read_gray_float(img_path)
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
    img_path = Path(img_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    particle_options = f"size={int(max(0, min_obj_area))}-Infinity show=Masks clear"
    args = {
        "input_path": img_path.as_posix(),
        "binary_path": out_path.as_posix(),
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

    ki67_mask = _read_binary_mask(ki67_dir)
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

    ki67_mask = _read_binary_mask(ki67_mask_path)
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

    def cell_outline_group(cell_id: object) -> Optional[int]:
        """由 Cell_ID 解析 merged outline 的 0-based group index。"""
        try:
            return int(str(cell_id).rsplit("_", 1)[1]) - 1
        except Exception:
            return None

    df["ki67_positive"] = [
        1 if cell_outline_group(cell_id) in positive_groups else 0
        for cell_id in df["Cell_ID"]
    ]
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
        """列出資料夾內支援的影像檔名。"""
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
        """安全取得清單元素，索引超出時回傳空字串。"""
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

        add_outline_status_column(final_csv, outlines_txt, pc_img)

    merge_all_final_csvs(data_path)

    if clean_temp:
        remove_temp_files(analy_dir)
        remove_temp_files(output_dir(data_path, "outline"))
        segment_dir = Path("data/output/segment") / data_path.name
        if segment_dir.exists():
            remove_temp_files(segment_dir, keywords=["_seg.npy"])
        for raw_image_dir in (data_path / "PC", data_path / "DAPI"):
            if raw_image_dir.exists():
                remove_temp_files(raw_image_dir, keywords=["_seg.npy"])
