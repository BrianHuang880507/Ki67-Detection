#!/usr/bin/env python3
"""Quantify nucleus/cytoplasm fluorescence from pre-existing contour coordinates.

This script does NOT perform segmentation. It builds ROI masks from input contours,
preprocesses the selected signal channel with ImageJ-style background subtraction,
and measures whole-cell / nucleus / whole-cytoplasm intensity metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tifffile

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from ki67dtc.utils.io import output_dir as main_output_dir
except Exception:  # pragma: no cover
    main_output_dir = None

try:
    from shapely.geometry import MultiPolygon, Polygon
except ImportError:  # pragma: no cover
    Polygon = None
    MultiPolygon = None

LOGGER = logging.getLogger("cytoplasm_from_contours")

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

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_QUICK_SIGNAL_PRIORITY = ("PC", "DF", "KI67", "IDO", "LT", "DAPI")


@dataclass
class ContourRecord:
    image_name: str
    cell_id: str
    contour_type: str
    points: np.ndarray  # shape (N, 2), columns: x, y


@dataclass
class CellRoiResult:
    image_name: str
    cell_id: str
    nucleus_mask: np.ndarray | None
    cytoplasm_mask: np.ndarray | None
    whole_cell_mask: np.ndarray | None
    nucleus_points: np.ndarray | None
    cytoplasm_points: np.ndarray | None
    whole_cell_points: np.ndarray | None
    flags: list[str]


# ------------------------------
# Argument parsing
# ------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify fluorescence in whole cytoplasm from contour coordinates"
    )

    parser.add_argument(
        "data_folder",
        nargs="?",
        type=Path,
        help=(
            "Main-pipeline style dataset root (or dataset name under data/input). "
            "When provided, script can auto-infer other paths."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        type=Path,
        help="Image folder (required only when positional data_folder is not provided).",
    )
    parser.add_argument(
        "--contours-file",
        default=None,
        type=Path,
        help=(
            "Contour source path. For CSV/JSON: file path. "
            "For TXT: either one merged outlines txt or a folder. "
            "Required only when positional data_folder is not provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Output folder for CSV and QC (optional with --use-main-output-layout)",
    )
    parser.add_argument(
        "--use-main-output-layout",
        action="store_true",
        help=(
            "Use main pipeline style output path: "
            "data/output/<subfolder>/<dataset_name>"
        ),
    )

    parser.add_argument(
        "--contour-format",
        default="auto",
        choices=["auto", "csv", "json", "txt"],
        help="Contour file format",
    )
    parser.add_argument(
        "--txt-glob",
        default="*_merged_cp_outlines.txt",
        help="Glob pattern when --contours-file is a folder in txt mode",
    )
    parser.add_argument(
        "--txt-second-contour-type",
        default=None,
        choices=["cytoplasm", "cell"],
        help=(
            "For merged txt pair format, first line is nucleus and second line type "
            "is set by this option."
        ),
    )
    parser.add_argument(
        "--contour-mode",
        default=None,
        choices=["nucleus_cell", "nucleus_cytoplasm"],
        help="nucleus+whole-cell or nucleus+cytoplasm contour mode",
    )
    parser.add_argument(
        "--cytoplasm-contour-interpretation",
        default=None,
        choices=["direct", "outer_cell"],
        help=(
            "Only used with contour-mode=nucleus_cytoplasm. "
            "direct: contour is cytoplasm ROI; outer_cell: contour is outer cell boundary"
        ),
    )
    parser.add_argument(
        "--signal-dir-name",
        default="auto",
        help=(
            "Quick mode only (when using positional data_folder). "
            "Set signal subfolder name (e.g. DF/KI67/IDO/DAPI/PC), or 'auto'."
        ),
    )
    parser.add_argument(
        "--image-name-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help=(
            "Optional contour name matching rewrite rule. "
            "Applied to input image filename before contour lookup. "
            "Example: --image-name-replace IDO:PHASE (can be repeated)."
        ),
    )

    parser.add_argument(
        "--signal-channel",
        type=int,
        default=0,
        help="0-based channel index for multi-channel image",
    )
    parser.add_argument(
        "--channel-axis",
        default="auto",
        choices=["auto", "first", "last"],
        help="Channel axis handling for multi-channel images",
    )
    parser.add_argument(
        "--rolling-ball-radius",
        type=float,
        default=50.0,
        help="ImageJ Subtract Background rolling radius",
    )
    parser.add_argument(
        "--fiji-app-path",
        default="",
        help="Optional local Fiji.app path for pyimagej init",
    )
    parser.add_argument(
        "--intden-mode",
        default="integrated_bgsub",
        choices=["integrated_bgsub", "mean_bgsub"],
        help=(
            "How to populate IntDen output column. "
            "integrated_bgsub: ID - (Area * mean_background); "
            "mean_bgsub: (ID - Area * mean_background) / Area."
        ),
    )
    parser.add_argument(
        "--intensity-scale",
        default="raw",
        choices=[
            "raw",
            "zero_to_one_auto",
            "zero_to_one_255",
            "zero_to_one_65535",
        ],
        help=(
            "Optional scaling for IntDen output values. "
            "raw: no scaling. "
            "zero_to_one_auto: divide by inferred denominator (1/255/65535) from image range. "
            "zero_to_one_255: divide by 255. "
            "zero_to_one_65535: divide by 65535."
        ),
    )

    parser.add_argument(
        "--nucleus-labels",
        default="nucleus,nuc",
        help="Comma-separated contour_type aliases for nucleus",
    )
    parser.add_argument(
        "--cell-labels",
        default="cell,whole_cell,wholecell,cell_boundary,cytoplasm_boundary",
        help="Comma-separated contour_type aliases for whole-cell",
    )
    parser.add_argument(
        "--cytoplasm-labels",
        default="cytoplasm,cyto",
        help="Comma-separated contour_type aliases for cytoplasm",
    )

    # Schema mapping
    parser.add_argument("--image-name-column", default="image_name")
    parser.add_argument("--cell-id-column", default="cell_id")
    parser.add_argument("--contour-type-column", default="contour_type")
    parser.add_argument(
        "--coords-column",
        default="polygon",
        help="CSV/JSON field for one-row-per-contour coordinates",
    )
    parser.add_argument(
        "--x-column",
        default="",
        help="CSV long-format x column (one row per point)",
    )
    parser.add_argument(
        "--y-column",
        default="",
        help="CSV long-format y column (one row per point)",
    )
    parser.add_argument(
        "--point-order-column",
        default="",
        help="Optional ordering column for long-format CSV points",
    )
    parser.add_argument(
        "--json-records-key",
        default="",
        help="JSON key containing contour records list (if top-level is dict)",
    )

    # Optional behavior
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan images in input-dir",
    )
    parser.add_argument(
        "--exclude-flagged",
        action="store_true",
        help="Exclude rows with QC flags from output CSV",
    )
    parser.add_argument(
        "--flag-border-touching",
        action="store_true",
        help="Add QC flags for border-touching ROIs",
    )
    parser.add_argument(
        "--exclude-border-touching",
        action="store_true",
        help="Flag border-touching ROIs and exclude them from output",
    )
    parser.add_argument(
        "--save-qc-overlays",
        action="store_true",
        help="Save QC overlay images with contours and IDs",
    )
    parser.add_argument(
        "--save-summary-per-image",
        action="store_true",
        help="Write per-image summary CSV",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def _resolve_data_folder(raw_data_folder: Path) -> Path:
    candidates: list[Path] = []
    if raw_data_folder.is_absolute():
        candidates.append(raw_data_folder)
    else:
        base_dir = Path("data/input")
        candidates.append(base_dir / raw_data_folder)
        candidates.append(raw_data_folder)

    search_targets: list[Path] = []
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
        "Dataset folder not found. Checked: "
        + ", ".join(str(c) for c in search_targets)
    )


def _has_supported_images(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    return any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir()
    )


def _infer_quick_input_dir(dataset_root: Path, signal_dir_name: str) -> Path:
    if signal_dir_name.lower() != "auto":
        preferred = dataset_root / signal_dir_name
        if not _has_supported_images(preferred):
            raise FileNotFoundError(
                f"Requested signal folder has no supported images: {preferred}"
            )
        return preferred

    for name in _QUICK_SIGNAL_PRIORITY:
        candidate = dataset_root / name
        if _has_supported_images(candidate):
            return candidate

    if _has_supported_images(dataset_root):
        return dataset_root

    raise RuntimeError(
        f"Cannot infer input image folder under dataset '{dataset_root}'. "
        f"Tried: {', '.join(_QUICK_SIGNAL_PRIORITY)}."
    )


def _finalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    quick_mode = args.data_folder is not None

    if quick_mode:
        dataset_root = _resolve_data_folder(Path(args.data_folder))

        if args.input_dir is None:
            args.input_dir = _infer_quick_input_dir(dataset_root, args.signal_dir_name)
        if args.contours_file is None:
            args.contours_file = Path("data") / "output" / "outline" / dataset_root.name
        if args.output_dir is None and not args.use_main_output_layout:
            args.use_main_output_layout = True
        if args.contour_mode is None:
            # Main merged outlines are nucleus + cytoplasm pairs.
            args.contour_mode = "nucleus_cytoplasm"
    else:
        if args.input_dir is None or args.contours_file is None:
            raise ValueError(
                "Provide either positional data_folder only (quick mode), "
                "or both --input-dir and --contours-file."
            )
        if args.contour_mode is None:
            args.contour_mode = "nucleus_cell"

    if args.txt_second_contour_type is None:
        args.txt_second_contour_type = "cytoplasm"
    if args.cytoplasm_contour_interpretation is None:
        args.cytoplasm_contour_interpretation = "direct"

    args.input_dir = args.input_dir.resolve()
    args.contours_file = args.contours_file.resolve()
    args.image_name_replace_rules = _parse_image_name_replace_rules(
        args.image_name_replace
    )
    return args


# ------------------------------
# Parsing helpers
# ------------------------------
def _parse_image_name_replace_rules(
    raw_rules: Iterable[str] | None,
) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    if raw_rules is None:
        return rules

    for raw in raw_rules:
        token = str(raw).strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid --image-name-replace '{token}'. Expected SRC:DST."
            )
        src, dst = token.split(":", 1)
        src = src.strip()
        dst = dst.strip()
        if not src:
            raise ValueError(
                f"Invalid --image-name-replace '{token}'. SRC cannot be empty."
            )
        rules.append((src, dst))
    return rules


def _as_alias_set(raw: str) -> set[str]:
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _normalize_image_key(name: str) -> tuple[str, str]:
    p = Path(str(name).strip())
    return p.name.lower(), p.stem.lower()


def _candidate_image_keys(
    image_name: str,
    replace_rules: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    variants: set[str] = {str(image_name).strip()}
    if replace_rules:
        # Apply replacements iteratively to allow simple chained rewrites.
        for _ in range(len(replace_rules)):
            expanded = False
            current = list(variants)
            for name in current:
                for src, dst in replace_rules:
                    if src in name:
                        replaced = name.replace(src, dst)
                        if replaced not in variants:
                            variants.add(replaced)
                            expanded = True
            if not expanded:
                break

    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for name in variants:
        key = _normalize_image_key(name)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _resolve_column(df: pd.DataFrame, preferred: str, fallbacks: Iterable[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in fallbacks:
        if col in df.columns:
            return col
    raise ValueError(
        f"Required column not found. preferred='{preferred}', "
        f"fallbacks={list(fallbacks)}, available={list(df.columns)}"
    )


def _parse_points(value: Any) -> np.ndarray:
    """Parse contour points into Nx2 float array.

    Supported examples:
    - [[x1,y1],[x2,y2],...]
    - "[[x1,y1],[x2,y2],...]"
    - "x1,y1;x2,y2;..."
    - "x1,y1,x2,y2,..."
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Missing coordinates")

    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        raise ValueError("List coordinates must be Nx2")

    text = str(value).strip()
    if not text:
        raise ValueError("Empty coordinate string")

    if text.startswith("["):
        arr = np.asarray(json.loads(text), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        raise ValueError("JSON coordinate string must decode to Nx2")

    if ";" in text:
        pairs = [p.strip() for p in text.split(";") if p.strip()]
        pts = []
        for pair in pairs:
            xy = [t.strip() for t in pair.split(",") if t.strip()]
            if len(xy) != 2:
                raise ValueError(f"Invalid pair '{pair}'")
            pts.append([float(xy[0]), float(xy[1])])
        return np.asarray(pts, dtype=float)

    nums = [t.strip() for t in text.split(",") if t.strip()]
    if len(nums) % 2 != 0:
        raise ValueError("Flat coordinate list must contain even number of values")
    flat = np.asarray([float(n) for n in nums], dtype=float)
    return flat.reshape(-1, 2)


def _load_contours_csv(args: argparse.Namespace) -> list[ContourRecord]:
    df = pd.read_csv(args.contours_file)

    image_col = _resolve_column(
        df,
        args.image_name_column,
        ["image_name", "image", "image_file", "image_filename", "filename"],
    )
    cell_col = _resolve_column(df, args.cell_id_column, ["cell_id", "cell", "id"])
    type_col = _resolve_column(
        df,
        args.contour_type_column,
        ["contour_type", "type", "roi_type", "label"],
    )

    records: list[ContourRecord] = []

    # Long format (one vertex per row) if x/y columns are provided.
    if args.x_column and args.y_column:
        x_col = _resolve_column(df, args.x_column, ["x", "coord_x", "px"])
        y_col = _resolve_column(df, args.y_column, ["y", "coord_y", "py"])

        order_col = None
        if args.point_order_column:
            order_col = _resolve_column(
                df,
                args.point_order_column,
                ["point_order", "order", "vertex_index", "idx"],
            )

        group_cols = [image_col, cell_col, type_col]
        for (image_name, cell_id, contour_type), grp in df.groupby(group_cols):
            g = grp.sort_values(order_col) if order_col else grp
            try:
                points = g[[x_col, y_col]].to_numpy(dtype=float)
            except Exception as exc:
                LOGGER.warning(
                    "Skipping malformed long-format contour (image=%s cell=%s type=%s): %s",
                    image_name,
                    cell_id,
                    contour_type,
                    exc,
                )
                continue
            records.append(
                ContourRecord(
                    image_name=str(image_name),
                    cell_id=str(cell_id),
                    contour_type=str(contour_type),
                    points=points,
                )
            )
        return records

    coords_col = _resolve_column(
        df,
        args.coords_column,
        ["polygon", "coordinates", "coords", "points", "vertices", "contour"],
    )

    for _, row in df.iterrows():
        try:
            points = _parse_points(row[coords_col])
        except Exception as exc:
            LOGGER.warning(
                "Skipping malformed contour row (image=%s cell=%s type=%s): %s",
                row.get(image_col),
                row.get(cell_col),
                row.get(type_col),
                exc,
            )
            continue
        records.append(
            ContourRecord(
                image_name=str(row[image_col]),
                cell_id=str(row[cell_col]),
                contour_type=str(row[type_col]),
                points=points,
            )
        )

    return records


def _load_contours_json(args: argparse.Namespace) -> list[ContourRecord]:
    with args.contours_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if args.json_records_key:
            rows = payload.get(args.json_records_key, [])
        elif "contours" in payload and isinstance(payload["contours"], list):
            rows = payload["contours"]
        else:
            raise ValueError(
                "JSON is a dict; provide --json-records-key or use a top-level list"
            )
    else:
        raise ValueError("Unsupported JSON structure for contours")

    records: list[ContourRecord] = []
    for row in rows:
        image_name = row.get(args.image_name_column) or row.get("image_name")
        cell_id = row.get(args.cell_id_column) or row.get("cell_id")
        contour_type = row.get(args.contour_type_column) or row.get("contour_type")

        if image_name is None or cell_id is None or contour_type is None:
            raise ValueError(
                f"JSON row missing required fields: {row}. "
                "Need image_name/cell_id/contour_type (or remapped columns)."
            )

        points_value = (
            row.get(args.coords_column)
            if args.coords_column in row
            else row.get("polygon", row.get("coordinates", row.get("points")))
        )
        try:
            points = _parse_points(points_value)
        except Exception as exc:
            LOGGER.warning(
                "Skipping malformed JSON contour row (image=%s cell=%s type=%s): %s",
                image_name,
                cell_id,
                contour_type,
                exc,
            )
            continue

        records.append(
            ContourRecord(
                image_name=str(image_name),
                cell_id=str(cell_id),
                contour_type=str(contour_type),
                points=points,
            )
        )

    return records


def _parse_txt_polygon_line(line: str) -> np.ndarray | None:
    text = line.strip()
    if not text or text == "-1,-1":
        return None
    nums = [x.strip() for x in text.split(",") if x.strip()]
    if len(nums) < 6 or len(nums) % 2 != 0:
        raise ValueError(f"Invalid txt polygon line: '{line}'")
    coords = np.asarray([float(x) for x in nums], dtype=float).reshape(-1, 2)
    return coords


def _guess_image_name_from_txt_path(txt_path: Path) -> str:
    stem = txt_path.stem
    for suffix in (
        "_merged_cp_outlines",
        "_cyto_seg_cp_outlines",
        "_nuc_seg_cp_outlines",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _iter_txt_paths(base: Path, pattern: str) -> list[Path]:
    if base.is_file():
        return [base]
    if base.is_dir():
        return sorted([p for p in base.glob(pattern) if p.is_file()])
    return []


def _load_contours_txt(args: argparse.Namespace) -> list[ContourRecord]:
    txt_paths = _iter_txt_paths(args.contours_file, args.txt_glob)
    if not txt_paths:
        raise ValueError(
            f"No txt outlines found under '{args.contours_file}' with pattern '{args.txt_glob}'"
        )

    records: list[ContourRecord] = []
    for txt_path in txt_paths:
        with txt_path.open("r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        if len(lines) % 2 != 0:
            LOGGER.warning(
                "TXT file has odd number of lines (ignoring last line): %s", txt_path
            )

        image_name = _guess_image_name_from_txt_path(txt_path)
        n_pairs = len(lines) // 2
        for i in range(n_pairs):
            cell_id = str(i + 1)
            nuc_line = lines[2 * i]
            sec_line = lines[2 * i + 1]

            try:
                nuc_pts = _parse_txt_polygon_line(nuc_line)
            except Exception as exc:
                LOGGER.warning(
                    "Skipping malformed nucleus line in %s pair %d: %s",
                    txt_path.name,
                    i + 1,
                    exc,
                )
                nuc_pts = None

            try:
                sec_pts = _parse_txt_polygon_line(sec_line)
            except Exception as exc:
                LOGGER.warning(
                    "Skipping malformed second contour line in %s pair %d: %s",
                    txt_path.name,
                    i + 1,
                    exc,
                )
                sec_pts = None

            if nuc_pts is not None:
                records.append(
                    ContourRecord(
                        image_name=image_name,
                        cell_id=cell_id,
                        contour_type="nucleus",
                        points=nuc_pts,
                    )
                )
            if sec_pts is not None:
                records.append(
                    ContourRecord(
                        image_name=image_name,
                        cell_id=cell_id,
                        contour_type=args.txt_second_contour_type,
                        points=sec_pts,
                    )
                )

    return records


def load_contours(args: argparse.Namespace) -> list[ContourRecord]:
    contour_format = args.contour_format
    if contour_format == "auto":
        if args.contours_file.is_dir():
            contour_format = "txt"
        else:
            suffix = args.contours_file.suffix.lower()
            if suffix == ".json":
                contour_format = "json"
            elif suffix == ".txt":
                contour_format = "txt"
            else:
                contour_format = "csv"

    if contour_format == "csv":
        return _load_contours_csv(args)
    if contour_format == "json":
        return _load_contours_json(args)
    if contour_format == "txt":
        return _load_contours_txt(args)
    raise ValueError(f"Unsupported contour format: {contour_format}")


# ------------------------------
# Image preprocessing (PyImageJ)
# ------------------------------
def init_pyimagej(fiji_app_path: str):
    try:
        import imagej
        import scyjava
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyimagej/scyjava not available. Install dependencies first."
        ) from exc

    scyjava.config.add_option("-Dscijava.log.level=info")
    if fiji_app_path:
        ij = imagej.init(fiji_app_path, mode="headless", add_legacy=True)
    else:
        ij = imagej.init("sc.fiji:fiji", mode="headless", add_legacy=True)

    if not (ij.legacy and ij.legacy.isActive()):
        raise RuntimeError("ImageJ legacy mode is inactive, cannot run IJ1 macro.")
    return ij


def select_signal_channel(
    image_array: np.ndarray, signal_channel: int, channel_axis: str
) -> np.ndarray:
    if image_array.ndim == 2:
        return image_array.astype(np.float32)

    if image_array.ndim != 3:
        raise ValueError(f"Unsupported image shape {image_array.shape}; expected 2D/3D")

    if channel_axis == "first":
        axis = 0
    elif channel_axis == "last":
        axis = 2
    else:
        if image_array.shape[-1] <= 4:
            axis = 2
        elif image_array.shape[0] <= 4:
            axis = 0
        else:
            raise ValueError(
                "channel-axis auto failed for shape "
                f"{image_array.shape}. Set --channel-axis explicitly."
            )

    num_channels = image_array.shape[axis]
    if not (0 <= signal_channel < num_channels):
        raise ValueError(
            f"signal-channel {signal_channel} out of range [0,{num_channels - 1}]"
        )

    if axis == 0:
        signal = image_array[signal_channel, :, :]
    else:
        signal = image_array[:, :, signal_channel]

    return signal.astype(np.float32)


def preprocess_signal_with_imagej(
    ij,
    signal_2d: np.ndarray,
    rolling_ball_radius: float,
    temp_dir: Path,
    image_stem: str,
) -> np.ndarray:
    tmp_input = temp_dir / f"{image_stem}_signal_input.tif"
    tmp_output = temp_dir / f"{image_stem}_signal_bgsub.tif"

    tifffile.imwrite(tmp_input, signal_2d)

    args = {
        "input_path": str(tmp_input),
        "output_path": str(tmp_output),
        "rolling_ball_radius": float(rolling_ball_radius),
    }
    ij.py.run_macro(_PREPROCESS_MACRO, args=args)

    if not tmp_output.exists():
        raise RuntimeError(f"ImageJ preprocessing failed; no output: {tmp_output}")

    processed = tifffile.imread(tmp_output).astype(np.float32)
    if processed.ndim != 2:
        processed = np.squeeze(processed)
        if processed.ndim != 2:
            raise ValueError(f"Unexpected preprocessed image shape: {processed.shape}")
    return processed


# ------------------------------
# ROI construction and QC
# ------------------------------
def read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        try:
            return np.asarray(tifffile.imread(path))
        except Exception:
            pass

    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required to read non-TIFF images in this environment. "
            "Install opencv-python/opencv-python-headless."
        )

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def sanitize_polygon_points(
    points: np.ndarray, image_shape: tuple[int, int]
) -> tuple[np.ndarray, bool, bool]:
    h, w = image_shape
    out_of_bounds = bool(
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] > (w - 1))
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] > (h - 1))
    )
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h - 1)

    valid = True
    if clipped.shape[0] < 3:
        valid = False

    unique_pts = np.unique(np.round(clipped, 6), axis=0)
    if unique_pts.shape[0] < 3:
        valid = False

    return clipped, out_of_bounds, valid


def polygon_is_valid(points: np.ndarray) -> bool:
    if Polygon is None:
        return True

    poly = Polygon(points)
    if poly.is_empty or poly.area <= 0:
        return False
    if poly.is_valid:
        return True

    repaired = poly.buffer(0)
    if repaired.is_empty:
        return False

    if isinstance(repaired, MultiPolygon):
        repaired = max(repaired.geoms, key=lambda g: g.area)
    return repaired.area > 0 and repaired.is_valid


def polygon_to_bool_mask(
    points: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for polygon mask rasterization in this environment."
        )

    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)
    return mask.astype(bool)


def mask_touches_border(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def classify_contour_type(
    contour_type: str,
    nucleus_alias: set[str],
    cell_alias: set[str],
    cytoplasm_alias: set[str],
) -> str | None:
    key = contour_type.strip().lower()
    if key in nucleus_alias:
        return "nucleus"
    if key in cell_alias:
        return "cell"
    if key in cytoplasm_alias:
        return "cytoplasm"
    return None


def choose_largest_polygon(
    polygons: list[np.ndarray], image_shape: tuple[int, int]
) -> tuple[np.ndarray | None, list[str]]:
    flags: list[str] = []
    candidates: list[tuple[int, np.ndarray]] = []
    for points in polygons:
        clipped, out_of_bounds, basic_valid = sanitize_polygon_points(
            points, image_shape
        )
        if out_of_bounds:
            flags.append("CONTOUR_OUT_OF_BOUNDS")
        if not basic_valid:
            flags.append("INVALID_POLYGON")
            continue
        if not polygon_is_valid(clipped):
            flags.append("INVALID_POLYGON")
            continue
        mask = polygon_to_bool_mask(clipped, image_shape)
        area = int(mask.sum())
        if area <= 0:
            flags.append("ZERO_AREA_POLYGON")
            continue
        candidates.append((area, clipped))

    if not candidates:
        return None, flags
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], flags


def build_cell_roi(
    image_name: str,
    cell_id: str,
    contour_group: dict[str, list[np.ndarray]],
    image_shape: tuple[int, int],
    contour_mode: str,
    cytoplasm_interpretation: str,
    check_border_touching: bool,
) -> CellRoiResult:
    flags: list[str] = []

    nucleus_poly, nuc_flags = choose_largest_polygon(
        contour_group.get("nucleus", []), image_shape
    )
    flags.extend(f"NUCLEUS_{f}" for f in nuc_flags)

    cell_poly: np.ndarray | None = None
    cytoplasm_poly: np.ndarray | None = None

    if contour_mode == "nucleus_cell":
        cell_poly, cell_flags = choose_largest_polygon(
            contour_group.get("cell", []), image_shape
        )
        flags.extend(f"CELL_{f}" for f in cell_flags)
    else:
        cytoplasm_poly, cyto_flags = choose_largest_polygon(
            contour_group.get("cytoplasm", []), image_shape
        )
        flags.extend(f"CYTOPLASM_{f}" for f in cyto_flags)

    if nucleus_poly is None:
        flags.append("MISSING_NUCLEUS")

    nucleus_mask = (
        polygon_to_bool_mask(nucleus_poly, image_shape)
        if nucleus_poly is not None
        else None
    )

    whole_cell_mask: np.ndarray | None = None
    whole_cell_points: np.ndarray | None = None
    cytoplasm_mask: np.ndarray | None = None
    cytoplasm_points: np.ndarray | None = None

    if contour_mode == "nucleus_cell":
        if cell_poly is None:
            flags.append("MISSING_CELL")
        else:
            whole_cell_points = cell_poly
            whole_cell_mask = polygon_to_bool_mask(cell_poly, image_shape)

        if whole_cell_mask is not None and nucleus_mask is not None:
            outside = np.logical_and(nucleus_mask, np.logical_not(whole_cell_mask))
            if np.any(outside):
                flags.append("NUCLEUS_NOT_INSIDE_CELL")
            cytoplasm_mask = np.logical_and(
                whole_cell_mask, np.logical_not(nucleus_mask)
            )
            cytoplasm_points = None  # derived mask

    else:  # contour_mode == nucleus_cytoplasm
        if cytoplasm_poly is None:
            flags.append("MISSING_CYTOPLASM")
        elif cytoplasm_interpretation == "direct":
            cytoplasm_points = cytoplasm_poly
            cytoplasm_mask = polygon_to_bool_mask(cytoplasm_poly, image_shape)
            if nucleus_mask is not None:
                outside = np.logical_and(nucleus_mask, np.logical_not(cytoplasm_mask))
                if np.any(outside):
                    flags.append("NUCLEUS_NOT_INSIDE_CYTOPLASM")
                # Cytoplasm metrics must exclude nucleus pixels.
                cytoplasm_mask = np.logical_and(
                    cytoplasm_mask, np.logical_not(nucleus_mask)
                )
        else:
            whole_cell_points = cytoplasm_poly
            whole_cell_mask = polygon_to_bool_mask(cytoplasm_poly, image_shape)
            if nucleus_mask is not None:
                outside = np.logical_and(nucleus_mask, np.logical_not(whole_cell_mask))
                if np.any(outside):
                    flags.append("NUCLEUS_NOT_INSIDE_CELL")
                cytoplasm_mask = np.logical_and(
                    whole_cell_mask, np.logical_not(nucleus_mask)
                )

    if nucleus_mask is not None and int(nucleus_mask.sum()) == 0:
        flags.append("NUCLEUS_ZERO_AREA")
    if cytoplasm_mask is not None and int(cytoplasm_mask.sum()) == 0:
        flags.append("CYTOPLASM_ZERO_AREA")
    if whole_cell_mask is not None and int(whole_cell_mask.sum()) == 0:
        flags.append("WHOLE_CELL_ZERO_AREA")

    if check_border_touching:
        if nucleus_mask is not None and mask_touches_border(nucleus_mask):
            flags.append("NUCLEUS_BORDER_TOUCH")
        if cytoplasm_mask is not None and mask_touches_border(cytoplasm_mask):
            flags.append("CYTOPLASM_BORDER_TOUCH")
        if whole_cell_mask is not None and mask_touches_border(whole_cell_mask):
            flags.append("WHOLE_CELL_BORDER_TOUCH")

    return CellRoiResult(
        image_name=image_name,
        cell_id=cell_id,
        nucleus_mask=nucleus_mask,
        cytoplasm_mask=cytoplasm_mask,
        whole_cell_mask=whole_cell_mask,
        nucleus_points=nucleus_poly,
        cytoplasm_points=cytoplasm_points,
        whole_cell_points=whole_cell_points,
        flags=sorted(set(flags)),
    )


# ------------------------------
# Measurements
# ------------------------------
def _empty_imagej_measurements() -> dict[str, float]:
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
    out = _empty_imagej_measurements()
    if not path.exists():
        return out

    kv: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    if kv.get("valid") != "1":
        return out

    for key in out:
        if key in kv:
            try:
                out[key] = float(kv[key])
            except Exception:
                out[key] = np.nan

    # Fallback when ImageJ does not emit RawIntDen in some configurations.
    if (
        np.isnan(out["raw_intden"])
        and not np.isnan(out["area"])
        and not np.isnan(out["mean"])
    ):
        out["raw_intden"] = float(out["area"] * out["mean"])

    return out


def measure_roi_with_imagej(
    ij,
    signal_path: Path,
    roi_mask: np.ndarray | None,
    temp_dir: Path,
    slot: str,
) -> dict[str, float]:
    if roi_mask is None or not np.any(roi_mask):
        return _empty_imagej_measurements()

    mask_path = temp_dir / f"{slot}_mask.tif"
    output_path = temp_dir / f"{slot}_measure.txt"
    tifffile.imwrite(mask_path, (roi_mask.astype(np.uint8) * 255))
    ij.py.run_macro(
        _MEASURE_ROI_MACRO,
        args={
            "signal_path": str(signal_path),
            "mask_path": str(mask_path),
            "output_path": str(output_path),
        },
    )
    return _read_kv_measurements(output_path)


def corrected_intden(intden: float, area: float, mean_background: float) -> float:
    """Background-corrected integrated density: IDcor = ID - (Area * mean background)."""
    if any(np.isnan([intden, area, mean_background])):
        return np.nan
    return float(intden - (area * mean_background))


def corrected_mean_intensity(
    intden: float, area: float, mean_background: float
) -> float:
    """Background-corrected mean intensity: (ID - Area*BG) / Area."""
    if any(np.isnan([intden, area, mean_background])) or area <= 0:
        return np.nan
    return float((intden - (area * mean_background)) / area)


def _infer_scale_denominator(image: np.ndarray) -> float:
    if image.size == 0:
        return 1.0
    maxv = float(np.nanmax(image))
    if maxv <= 1.5:
        return 1.0
    if maxv <= 255.5:
        return 255.0
    return 65535.0


def apply_intensity_scale(
    value: float, scale_mode: str, image: np.ndarray
) -> float:
    if np.isnan(value) or scale_mode == "raw":
        return value
    if scale_mode == "zero_to_one_255":
        denom = 255.0
    elif scale_mode == "zero_to_one_65535":
        denom = 65535.0
    else:
        denom = _infer_scale_denominator(image)
    if denom <= 0:
        return np.nan
    return float(value / denom)


def geometry_from_imagej_measurements(m: dict[str, float]) -> dict[str, float]:
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

    # Prefer ImageJ directly-reported shape descriptors where available.
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


# ------------------------------
# QC overlays
# ------------------------------
def _normalize_for_overlay(image: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(image, [1, 99])
    if p99 <= p1:
        p1, p99 = float(np.min(image)), float(np.max(image))
        if p99 <= p1:
            return np.zeros_like(image, dtype=np.uint8)

    scaled = np.clip((image - p1) / (p99 - p1), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _draw_polygon(
    overlay: np.ndarray, points: np.ndarray | None, color: tuple[int, int, int]
):
    if points is None or points.shape[0] < 3:
        return
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        overlay, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA
    )


def save_qc_overlay(
    image: np.ndarray,
    cell_results: list[CellRoiResult],
    out_path: Path,
) -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for QC overlays. Install opencv-python or "
            "opencv-python-headless, or disable --save-qc-overlays."
        )

    base = _normalize_for_overlay(image)
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    for cell in cell_results:
        if cell.whole_cell_mask is not None:
            overlay[cell.whole_cell_mask] = (
                0.7 * overlay[cell.whole_cell_mask] + np.array([255, 80, 0]) * 0.3
            ).astype(np.uint8)
        if cell.cytoplasm_mask is not None:
            overlay[cell.cytoplasm_mask] = (
                0.7 * overlay[cell.cytoplasm_mask] + np.array([0, 255, 0]) * 0.3
            ).astype(np.uint8)
        if cell.nucleus_mask is not None:
            overlay[cell.nucleus_mask] = (
                0.7 * overlay[cell.nucleus_mask] + np.array([0, 0, 255]) * 0.3
            ).astype(np.uint8)

        _draw_polygon(overlay, cell.whole_cell_points, (255, 160, 0))
        _draw_polygon(overlay, cell.cytoplasm_points, (0, 255, 0))
        _draw_polygon(overlay, cell.nucleus_points, (0, 0, 255))

        anchor_mask = (
            cell.cytoplasm_mask
            if cell.cytoplasm_mask is not None
            else (
                cell.whole_cell_mask
                if cell.whole_cell_mask is not None
                else cell.nucleus_mask
            )
        )
        if anchor_mask is not None and np.any(anchor_mask):
            ys, xs = np.where(anchor_mask)
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            text_color = (0, 255, 255) if not cell.flags else (0, 80, 255)
            cv2.putText(
                overlay,
                str(cell.cell_id),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
                cv2.LINE_AA,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


# ------------------------------
# Pipeline
# ------------------------------
def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [
        p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def infer_dataset_root(image_dir: Path) -> Path:
    image_dir = image_dir.resolve()
    if image_dir.name.upper() in {"PC", "DF", "LT", "KI67", "DAPI"}:
        return image_dir.parent
    return image_dir


def _fallback_output_dir(dataset_root: Path, subfolder: str) -> Path:
    out = Path("data") / "output" / subfolder / dataset_root.name
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.use_main_output_layout or args.output_dir is None:
        dataset_root = infer_dataset_root(args.input_dir)
        if main_output_dir is not None:
            results_dir = main_output_dir(dataset_root, "results").resolve()
            qc_dir = main_output_dir(dataset_root, "qc_overlays").resolve()
        else:
            results_dir = _fallback_output_dir(dataset_root, "results")
            qc_dir = _fallback_output_dir(dataset_root, "qc_overlays")
        return results_dir, qc_dir

    results_dir = args.output_dir.resolve()
    qc_dir = (results_dir / "qc_overlays").resolve()
    return results_dir, qc_dir


def group_contours_by_image(
    records: list[ContourRecord],
) -> tuple[dict[str, list[ContourRecord]], dict[str, list[ContourRecord]]]:
    by_name: dict[str, list[ContourRecord]] = {}
    by_stem: dict[str, list[ContourRecord]] = {}
    for rec in records:
        key_name, key_stem = _normalize_image_key(rec.image_name)
        by_name.setdefault(key_name, []).append(rec)
        by_stem.setdefault(key_stem, []).append(rec)
    return by_name, by_stem


def build_image_cell_groups(
    image_records: list[ContourRecord],
    nucleus_alias: set[str],
    cell_alias: set[str],
    cytoplasm_alias: set[str],
) -> dict[str, dict[str, list[np.ndarray]]]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = {}
    for rec in image_records:
        ctype = classify_contour_type(
            rec.contour_type,
            nucleus_alias=nucleus_alias,
            cell_alias=cell_alias,
            cytoplasm_alias=cytoplasm_alias,
        )
        if ctype is None:
            LOGGER.debug(
                "Skip unknown contour type '%s' (image=%s cell=%s)",
                rec.contour_type,
                rec.image_name,
                rec.cell_id,
            )
            continue

        cell_bucket = grouped.setdefault(
            rec.cell_id, {"nucleus": [], "cell": [], "cytoplasm": []}
        )
        cell_bucket[ctype].append(rec.points)

    return grouped


def process_image(
    ij,
    image_path: Path,
    image_records: list[ContourRecord],
    args: argparse.Namespace,
    qc_overlay_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_raw = read_image(image_path)
    signal = select_signal_channel(
        image_raw,
        signal_channel=args.signal_channel,
        channel_axis=args.channel_axis,
    )

    with tempfile.TemporaryDirectory(prefix="pyimagej_bgsub_") as tmp:
        preprocessed = preprocess_signal_with_imagej(
            ij,
            signal,
            rolling_ball_radius=args.rolling_ball_radius,
            temp_dir=Path(tmp),
            image_stem=image_path.stem,
        )

    h, w = preprocessed.shape
    grouped_cells = build_image_cell_groups(
        image_records,
        nucleus_alias=_as_alias_set(args.nucleus_labels),
        cell_alias=_as_alias_set(args.cell_labels),
        cytoplasm_alias=_as_alias_set(args.cytoplasm_labels),
    )

    cell_rois: list[CellRoiResult] = []
    all_cell_masks = np.zeros((h, w), dtype=bool)

    for cell_id, contour_group in grouped_cells.items():
        result = build_cell_roi(
            image_name=image_path.name,
            cell_id=cell_id,
            contour_group=contour_group,
            image_shape=(h, w),
            contour_mode=args.contour_mode,
            cytoplasm_interpretation=args.cytoplasm_contour_interpretation,
            check_border_touching=(
                args.flag_border_touching or args.exclude_border_touching
            ),
        )
        cell_rois.append(result)

        for m in (result.nucleus_mask, result.cytoplasm_mask, result.whole_cell_mask):
            if m is not None:
                all_cell_masks |= m

    bg_mask = np.logical_not(all_cell_masks)
    mean_background = (
        float(np.mean(preprocessed[bg_mask])) if np.any(bg_mask) else np.nan
    )

    rows: list[dict[str, Any]] = []
    exported_cells = 0
    with tempfile.TemporaryDirectory(prefix="pyimagej_measure_") as mtmp:
        mtmp_dir = Path(mtmp)
        signal_measure_path = mtmp_dir / f"{image_path.stem}_signal_bgsub.tif"
        tifffile.imwrite(signal_measure_path, preprocessed.astype(np.float32))

        for idx, result in enumerate(cell_rois, start=1):
            if args.exclude_flagged and result.flags:
                continue
            if args.exclude_border_touching and any(
                f.endswith("BORDER_TOUCH") for f in result.flags
            ):
                continue
            exported_cells += 1

            nuc_m = measure_roi_with_imagej(
                ij,
                signal_measure_path,
                result.nucleus_mask,
                mtmp_dir,
                f"roi_{idx}_nuc",
            )
            cyto_m = measure_roi_with_imagej(
                ij,
                signal_measure_path,
                result.cytoplasm_mask,
                mtmp_dir,
                f"roi_{idx}_cyto",
            )

            nuc_g = geometry_from_imagej_measurements(nuc_m)
            cyto_g = geometry_from_imagej_measurements(cyto_m)
            cyto_id_cor = corrected_intden(
                cyto_m["intden"], cyto_m["area"], mean_background
            )
            cyto_mean_bgsub = corrected_mean_intensity(
                cyto_m["intden"], cyto_m["area"], mean_background
            )
            intden_value = (
                cyto_mean_bgsub if args.intden_mode == "mean_bgsub" else cyto_id_cor
            )
            intden_value = apply_intensity_scale(
                intden_value, args.intensity_scale, preprocessed
            )

            karyoplasmic_ratio = np.nan
            if (
                not np.isnan(cyto_g["area"])
                and not np.isnan(nuc_g["area"])
                and nuc_g["area"] > 0
            ):
                karyoplasmic_ratio = float(cyto_g["area"] / nuc_g["area"])

            rows.append(
                {
                    "Cell_ID": f"{image_path.stem}_{result.cell_id}",
                    "Area_nuc": nuc_g["area"],
                    "Perimeter_nuc": nuc_g["perimeter"],
                    "Convex Perimeter_nuc": nuc_g["convex_perimeter"],
                    "Circular Diameter_nuc": nuc_g["circular_diameter"],
                    "Feret Length_nuc": nuc_g["feret_length"],
                    "Feret Width_nuc": nuc_g["feret_width"],
                    "Aspect Ratio_nuc": nuc_g["aspect_ratio"],
                    "Roundness_nuc": nuc_g["roundness"],
                    "Circularity_nuc": nuc_g["circularity"],
                    "Sphericity_nuc": nuc_g["sphericity"],
                    "Roughness_nuc": nuc_g["roughness"],
                    "Area_cyto": cyto_g["area"],
                    "Perimeter_cyto": cyto_g["perimeter"],
                    "Convex Perimeter_cyto": cyto_g["convex_perimeter"],
                    "Circular Diameter_cyto": cyto_g["circular_diameter"],
                    "Feret Length_cyto": cyto_g["feret_length"],
                    "Feret Width_cyto": cyto_g["feret_width"],
                    "Aspect Ratio_cyto": cyto_g["aspect_ratio"],
                    "Roundness_cyto": cyto_g["roundness"],
                    "Circularity_cyto": cyto_g["circularity"],
                    "Sphericity_cyto": cyto_g["sphericity"],
                    "Roughness_cyto": cyto_g["roughness"],
                    "Karyoplasmic Ratio_cyto": karyoplasmic_ratio,
                    "IntDen": intden_value,
                    "RawIntDen": cyto_m["raw_intden"],
                }
            )

    if args.save_qc_overlays:
        overlay_path = qc_overlay_dir / f"{image_path.stem}_qc.png"
        save_qc_overlay(preprocessed, cell_rois, overlay_path)

    summary = {
        "image_name": image_path.name,
        "n_cells_total": len(cell_rois),
        "n_cells_exported": exported_cells,
        "n_rows_exported": len(rows),
        "n_cells_flagged": int(sum(1 for c in cell_rois if c.flags)),
        "mean_background": mean_background,
    }

    return rows, summary


def run_pipeline(args: argparse.Namespace) -> None:
    args = _finalize_runtime_args(args)
    results_dir, qc_overlay_dir = resolve_output_paths(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input image folder not found: {args.input_dir}")
    if not args.contours_file.exists():
        raise FileNotFoundError(f"Contour file not found: {args.contours_file}")

    LOGGER.info("Step 1/15: Reading contour file")
    records = load_contours(args)
    if not records:
        raise RuntimeError("No contour records loaded")

    LOGGER.info("Step 2/15: Indexing contour records")
    contours_by_name, contours_by_stem = group_contours_by_image(records)
    LOGGER.info("Intensity mode for IntDen column: %s", args.intden_mode)
    LOGGER.info("Intensity scale for IntDen column: %s", args.intensity_scale)
    if args.image_name_replace_rules:
        LOGGER.info(
            "Using image-name replacement rules for contour lookup: %s",
            ", ".join(f"{src}->{dst}" for src, dst in args.image_name_replace_rules),
        )

    LOGGER.info("Step 3/15: Initializing PyImageJ")
    fiji_path = (
        args.fiji_app_path.strip() or os.environ.get("FIJI_APP_PATH", "").strip()
    )
    ij = init_pyimagej(fiji_path)

    LOGGER.info("Step 4/15: Scanning image folder")
    images = list_images(args.input_dir, args.recursive)
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    LOGGER.info("Step 5-15: Processing %d images", len(images))
    for idx, image_path in enumerate(images, start=1):
        image_records: list[ContourRecord] = []
        for key_name, key_stem in _candidate_image_keys(
            image_path.name, args.image_name_replace_rules
        ):
            image_records = contours_by_name.get(key_name)
            if image_records is None:
                image_records = contours_by_stem.get(key_stem, [])
            if image_records:
                break

        if not image_records:
            LOGGER.warning(
                "[%d/%d] No contour entries for image '%s'; skipping",
                idx,
                len(images),
                image_path.name,
            )
            continue

        LOGGER.info("[%d/%d] Processing %s", idx, len(images), image_path.name)
        rows, summary = process_image(
            ij, image_path, image_records, args, qc_overlay_dir
        )
        all_rows.extend(rows)
        all_summaries.append(summary)

    if not all_rows:
        LOGGER.warning("No output rows produced (all skipped or flagged)")

    out_csv = results_dir / "cell_measurements.csv"
    df = pd.DataFrame(all_rows)

    required_order = [
        "Cell_ID",
        "Area_nuc",
        "Perimeter_nuc",
        "Convex Perimeter_nuc",
        "Circular Diameter_nuc",
        "Feret Length_nuc",
        "Feret Width_nuc",
        "Aspect Ratio_nuc",
        "Roundness_nuc",
        "Circularity_nuc",
        "Sphericity_nuc",
        "Roughness_nuc",
        "Area_cyto",
        "Perimeter_cyto",
        "Convex Perimeter_cyto",
        "Circular Diameter_cyto",
        "Feret Length_cyto",
        "Feret Width_cyto",
        "Aspect Ratio_cyto",
        "Roundness_cyto",
        "Circularity_cyto",
        "Sphericity_cyto",
        "Roughness_cyto",
        "Karyoplasmic Ratio_cyto",
        "IntDen",
        "RawIntDen",
    ]
    for col in required_order:
        if col not in df.columns:
            df[col] = np.nan
    df = df[required_order]

    df.to_csv(out_csv, index=False)
    LOGGER.info("Saved ImageJ-style ROI measurements: %s", out_csv)

    if args.save_summary_per_image:
        summary_csv = results_dir / "image_summary.csv"
        pd.DataFrame(all_summaries).to_csv(summary_csv, index=False)
        LOGGER.info("Saved image-level summary: %s", summary_csv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
