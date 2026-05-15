from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PREDICTION_SUFFIX = "_prediction.xlsx"
DEFAULT_SHEET_NAME = "cell_predictions"
OUTLINE_SUFFIX = "_merged_cp_outlines.txt"

# OpenCV uses BGR channel order.
COLORS_BGR = {
    "no_ki67": (255, 255, 255),  # white: pred=0/truth=0 or missing Ki67 label
    "true_positive": (0, 255, 0),  # green: pred=1/truth=1
    "false_positive": (0, 255, 255),  # yellow: pred=1/truth=0
    "false_negative": (0, 0, 255),  # red: pred=0/truth=1
}
LEGEND_ITEMS = [
    ("no_ki67", "White: Pred 0 / Truth 0"),
    ("true_positive", "Green: Pred 1 / Truth 1"),
    ("false_positive", "Yellow: Pred 1 / Truth 0"),
    ("false_negative", "Red: Pred 0 / Truth 1"),
]


@dataclass(frozen=True)
class OutlinePair:
    roi_id: int
    nucleus_xy: np.ndarray
    cytoplasm_xy: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Ki67 prediction contour images from "
            "<folder>_prediction.xlsx and merged cell outlines."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        default="data/output/predict",
        help="Folder containing <folder>_prediction.xlsx files.",
    )
    parser.add_argument(
        "--input-root",
        default="data/input",
        help="Root folder containing source datasets and PC images.",
    )
    parser.add_argument(
        "--outline-root",
        default="data/output/outline",
        help="Root folder containing merged outline files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/output/ki67_contours",
        help="Output folder for generated contour PNGs.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="Only process these source folder names. Default: all prediction workbooks.",
    )
    parser.add_argument(
        "--sheet-name",
        default=DEFAULT_SHEET_NAME,
        help=f"Workbook sheet with cell predictions (default: {DEFAULT_SHEET_NAME}).",
    )
    parser.add_argument(
        "--outline-part",
        choices=["cyto", "nuc", "both"],
        default="both",
        help="Which contour to draw for each cell (default: both).",
    )
    parser.add_argument(
        "--background",
        choices=["pc", "black", "white"],
        default="pc",
        help="Image background for contour output (default: pc).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Cell/cytoplasm contour line width in pixels (default: 2).",
    )
    parser.add_argument(
        "--nucleus-line-width",
        type=int,
        default=1,
        help="Nucleus contour line width in pixels when --outline-part includes nuc.",
    )
    parser.add_argument(
        "--legend-position",
        choices=["right", "none"],
        default="right",
        help="Place color meaning outside the image on the right, or omit it.",
    )
    parser.add_argument(
        "--summary-name",
        default="contour_summary.csv",
        help="Summary CSV filename written under output-root.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path, root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return root / path


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def list_workbooks(prediction_dir: Path, folders: list[str] | None) -> list[Path]:
    if folders:
        return [prediction_dir / f"{folder}{PREDICTION_SUFFIX}" for folder in folders]
    return sorted(
        (
            path
            for path in prediction_dir.glob(f"*{PREDICTION_SUFFIX}")
            if not path.name.startswith("~$")
        ),
        key=natural_key,
    )


def folder_name_from_workbook(workbook_path: Path) -> str:
    name = workbook_path.name
    if name.endswith(PREDICTION_SUFFIX):
        return name[: -len(PREDICTION_SUFFIX)]
    return workbook_path.stem


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def require_columns(df: pd.DataFrame, workbook_path: Path) -> None:
    required = {
        "Image",
        "Cell_ID",
        "predicted_ki67_positive",
        "ki67_ground_truth",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{workbook_path} is missing required columns: {', '.join(missing)}"
        )


def parse_binary(value: object) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() in {"true", "yes", "y"}:
            return 1
        if stripped.lower() in {"false", "no", "n"}:
            return 0
        value = stripped
    try:
        return 1 if float(value) > 0 else 0
    except (TypeError, ValueError):
        return None


def prediction_class(predicted: object, ground_truth: object) -> str:
    pred = parse_binary(predicted)
    truth = parse_binary(ground_truth)
    if pred == 1 and truth == 1:
        return "true_positive"
    if pred == 1 and truth == 0:
        return "false_positive"
    if pred == 0 and truth == 1:
        return "false_negative"
    return "no_ki67"


def parse_roi_id(cell_id: object) -> int | None:
    if pd.isna(cell_id):
        return None
    match = re.search(r"_(\d+)(?:_(?:nuc|cyto))?$", str(cell_id).strip(), re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def load_prediction_lookup(
    workbook_path: Path,
    sheet_name: str,
) -> dict[str, dict[int, str]]:
    df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    df = normalize_prediction_columns(df)
    require_columns(df, workbook_path)

    lookup: dict[str, dict[int, str]] = {}
    for _, row in df.iterrows():
        image_value = row.get("Image")
        if pd.isna(image_value):
            continue
        roi_id = parse_roi_id(row.get("Cell_ID"))
        if roi_id is None:
            continue

        image_stem = Path(str(image_value).strip()).stem
        lookup.setdefault(image_stem, {})[roi_id] = prediction_class(
            row.get("predicted_ki67_positive"),
            row.get("ki67_ground_truth"),
        )
    return lookup


def parse_outline_line(line: str) -> np.ndarray | None:
    line = line.strip()
    if not line or line == "-1,-1":
        return None
    try:
        coords = [int(value) for value in line.split(",")]
    except ValueError:
        return None
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    return np.asarray(coords, dtype=np.int32).reshape(-1, 2)


def parse_outline_pairs(outline_path: Path) -> list[OutlinePair]:
    with outline_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    pairs: list[OutlinePair] = []
    for index in range(len(lines) // 2):
        nucleus_xy = parse_outline_line(lines[2 * index])
        cytoplasm_xy = parse_outline_line(lines[2 * index + 1])
        if nucleus_xy is None or cytoplasm_xy is None:
            continue
        pairs.append(
            OutlinePair(
                roi_id=index + 1,
                nucleus_xy=nucleus_xy,
                cytoplasm_xy=cytoplasm_xy,
            )
        )
    return pairs


def build_pc_image_map(pc_dir: Path) -> dict[str, Path]:
    if not pc_dir.exists() or not pc_dir.is_dir():
        return {}
    return {
        path.stem: path
        for path in pc_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)
    finite_mask = np.isfinite(image_float)
    if not finite_mask.any():
        return np.zeros(image.shape, dtype=np.uint8)

    min_value = float(image_float[finite_mask].min())
    max_value = float(image_float[finite_mask].max())
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image_float - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def read_image_bgr(path: Path) -> np.ndarray | None:
    try:
        image_bytes = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if image_bytes.size == 0:
        return None

    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    image = to_uint8(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return None


def canvas_shape_from_outlines(pairs: list[OutlinePair]) -> tuple[int, int]:
    max_x = 0
    max_y = 0
    for pair in pairs:
        for points in (pair.nucleus_xy, pair.cytoplasm_xy):
            if points.size == 0:
                continue
            max_x = max(max_x, int(points[:, 0].max()))
            max_y = max(max_y, int(points[:, 1].max()))
    return max(max_y + 1, 1), max(max_x + 1, 1)


def make_canvas(
    background: str,
    pc_image_path: Path | None,
    pairs: list[OutlinePair],
) -> tuple[np.ndarray, bool]:
    pc_image = read_image_bgr(pc_image_path) if pc_image_path is not None else None
    if background == "pc" and pc_image is not None:
        return pc_image.copy(), True

    if pc_image is not None:
        height, width = pc_image.shape[:2]
    else:
        height, width = canvas_shape_from_outlines(pairs)

    fill_value = 255 if background == "white" else 0
    return np.full((height, width, 3), fill_value, dtype=np.uint8), pc_image is not None


def draw_outline_pair(
    canvas: np.ndarray,
    pair: OutlinePair,
    contour_class: str,
    outline_part: str,
    line_width: int,
    nucleus_line_width: int,
) -> None:
    color = COLORS_BGR[contour_class]

    def draw_polyline(points: np.ndarray, thickness: int) -> None:
        cv2.polylines(
            canvas,
            [points.astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=max(1, int(thickness)),
            lineType=cv2.LINE_AA,
        )

    if outline_part in {"cyto", "both"}:
        draw_polyline(pair.cytoplasm_xy, line_width)
    if outline_part in {"nuc", "both"}:
        thickness = nucleus_line_width if outline_part == "both" else line_width
        draw_polyline(pair.nucleus_xy, thickness)


def append_legend_panel(
    image: np.ndarray,
    counts: Counter[str],
    outline_part: str,
) -> np.ndarray:
    panel_width = 380
    panel_bg = (28, 28, 28)
    text_color = (235, 235, 235)
    muted_text_color = (180, 180, 180)
    min_height = 310
    height = max(image.shape[0], min_height)
    panel = np.full((height, panel_width, 3), panel_bg, dtype=np.uint8)

    x = 24
    y = 38
    cv2.putText(
        panel,
        "Ki67 color key",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        text_color,
        2,
        cv2.LINE_AA,
    )
    y += 38

    for class_name, label in LEGEND_ITEMS:
        color = COLORS_BGR[class_name]
        cv2.line(panel, (x, y - 9), (x + 42, y - 9), color, 5, cv2.LINE_AA)
        cv2.putText(
            panel,
            label,
            (x + 58, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            text_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"n={int(counts[class_name])}",
            (x + 58, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            muted_text_color,
            1,
            cv2.LINE_AA,
        )
        y += 58

    y += 8
    contour_note = {
        "cyto": "Contour: cell/cytoplasm only",
        "nuc": "Contour: nucleus only",
        "both": "Outer: cell  Inner: nucleus",
    }[outline_part]
    cv2.putText(
        panel,
        contour_note,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        muted_text_color,
        1,
        cv2.LINE_AA,
    )

    combined = np.full(
        (height, image.shape[1] + panel_width, 3),
        0,
        dtype=np.uint8,
    )
    combined[: image.shape[0], : image.shape[1]] = image
    combined[:, image.shape[1] :] = panel
    return combined


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def image_stem_from_outline(outline_path: Path) -> str:
    name = outline_path.name
    if name.endswith(OUTLINE_SUFFIX):
        return name[: -len(OUTLINE_SUFFIX)]
    return outline_path.stem


def process_workbook(
    workbook_path: Path,
    sheet_name: str,
    input_root: Path,
    outline_root: Path,
    output_root: Path,
    background: str,
    outline_part: str,
    line_width: int,
    nucleus_line_width: int,
    legend_position: str,
) -> list[dict[str, object]]:
    folder_name = folder_name_from_workbook(workbook_path)
    if not workbook_path.exists():
        print(f"[WARN] Missing prediction workbook: {workbook_path}")
        return []

    outline_dir = outline_root / folder_name
    if not outline_dir.exists():
        print(f"[WARN] Missing outline folder: {outline_dir}")
        return []

    prediction_lookup = load_prediction_lookup(workbook_path, sheet_name)
    pc_images = build_pc_image_map(input_root / folder_name / "PC")
    outline_paths = sorted(outline_dir.glob(f"*{OUTLINE_SUFFIX}"), key=natural_key)
    if not outline_paths:
        print(f"[WARN] No merged outline files found: {outline_dir}")
        return []

    folder_output_dir = output_root / folder_name
    summary_rows: list[dict[str, object]] = []
    for outline_path in outline_paths:
        image_stem = image_stem_from_outline(outline_path)
        pairs = parse_outline_pairs(outline_path)
        predictions_for_image = prediction_lookup.get(image_stem, {})
        pc_image_path = pc_images.get(image_stem)
        canvas, pc_image_found = make_canvas(background, pc_image_path, pairs)

        counts: Counter[str] = Counter()
        missing_prediction_count = 0
        for pair in pairs:
            contour_class = predictions_for_image.get(pair.roi_id)
            if contour_class is None:
                contour_class = "no_ki67"
                missing_prediction_count += 1
            counts[contour_class] += 1
            draw_outline_pair(
                canvas=canvas,
                pair=pair,
                contour_class=contour_class,
                outline_part=outline_part,
                line_width=line_width,
                nucleus_line_width=nucleus_line_width,
            )

        if legend_position == "right":
            canvas = append_legend_panel(canvas, counts, outline_part)

        output_path = folder_output_dir / f"{image_stem}_ki67_contours.png"
        write_png(output_path, canvas)
        summary_rows.append(
            {
                "source_folder": folder_name,
                "image": image_stem,
                "cell_count": len(pairs),
                "no_ki67_white": counts["no_ki67"],
                "true_positive_green": counts["true_positive"],
                "false_positive_yellow": counts["false_positive"],
                "false_negative_red": counts["false_negative"],
                "missing_prediction_cells": missing_prediction_count,
                "pc_image_found": pc_image_found,
                "outline_part": outline_part,
                "legend_position": legend_position,
                "output_path": str(output_path),
            }
        )

    print(f"[OK] {folder_name}: wrote {len(summary_rows)} contour images")
    return summary_rows


def main() -> int:
    args = parse_args()
    root = repo_root()
    prediction_dir = resolve_path(args.prediction_dir, root)
    input_root = resolve_path(args.input_root, root)
    outline_root = resolve_path(args.outline_root, root)
    output_root = resolve_path(args.output_root, root)

    if args.line_width < 1:
        print("[ERROR] --line-width must be >= 1")
        return 2
    if args.nucleus_line_width < 1:
        print("[ERROR] --nucleus-line-width must be >= 1")
        return 2
    if not prediction_dir.exists():
        print(f"[ERROR] Prediction directory not found: {prediction_dir}")
        return 1

    workbooks = list_workbooks(prediction_dir, args.folders)
    if not workbooks:
        print(f"[INFO] No prediction workbooks found under: {prediction_dir}")
        return 0

    all_summary_rows: list[dict[str, object]] = []
    failures = 0
    for workbook_path in workbooks:
        try:
            all_summary_rows.extend(
                process_workbook(
                    workbook_path=workbook_path,
                    sheet_name=args.sheet_name,
                    input_root=input_root,
                    outline_root=outline_root,
                    output_root=output_root,
                    background=args.background,
                    outline_part=args.outline_part,
                    line_width=args.line_width,
                    nucleus_line_width=args.nucleus_line_width,
                    legend_position=args.legend_position,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures += 1
            print(f"[ERROR] Failed to process {workbook_path}: {exc}")

    if all_summary_rows:
        summary_path = output_root / args.summary_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_summary_rows).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[INFO] Summary written: {summary_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
