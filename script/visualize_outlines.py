import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_line(line: str) -> Optional[List[Tuple[int, int]]]:
    """Convert one outlines line (x1,y1,...) into a list of (x, y) points; skip padding lines."""
    coords = [int(v) for v in line.strip().split(",") if v]
    if len(coords) < 6 or coords[0] == -1:
        return None
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]


def find_image(stem: str, img_dir: Path, exts: Iterable[str]) -> Optional[Path]:
    for ext in exts:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


PINK = (255, 105, 180)  # default cytoplasm color
GREEN = (0, 255, 0)  # ki67=1 & prediction=1
RED = (255, 0, 0)  # ki67=1 & prediction=0


def status_to_color(status: str) -> Tuple[int, int, int]:
    if status == "tp":  # ki67=1 & prediction=1
        return GREEN
    if status == "fn":  # ki67=1 & prediction=0
        return RED
    return PINK


def load_font() -> ImageFont.ImageFont:
    """Try to load a font that supports Chinese; fallback to default."""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msjh.ttc",  # Microsoft JhengHei
        "C:/Windows/Fonts/msjhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, 14)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_legend(img: Image.Image):
    draw = ImageDraw.Draw(img)
    font = load_font()
    entries = [
        (GREEN, "綠色：預測正確"),
        (RED, "紅色：預測錯誤"),
    ]
    margin = 10
    swatch_w, swatch_h = 16, 10
    gap = 6
    line_gap = 4

    text_widths = []
    for _, text in entries:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_widths.append(bbox[2] - bbox[0])
    box_w = margin * 2 + swatch_w + gap + max(text_widths)
    box_h = margin * 2 + len(entries) * (swatch_h + line_gap) - line_gap

    bg_coords = [margin - 6, margin - 6, margin - 6 + box_w, margin - 6 + box_h]
    draw.rectangle(bg_coords, fill=(255, 255, 255), outline=(0, 0, 0))

    y = margin
    for color, text in entries:
        draw.rectangle([margin, y, margin + swatch_w, y + swatch_h], fill=color, outline=(0, 0, 0))
        draw.text((margin + swatch_w + gap, y - 1), text, fill=(0, 0, 0), font=font)
        y += swatch_h + line_gap


def draw_outlines(
    img_path: Path,
    outline_file: Path,
    out_path: Path,
    thickness: int,
    status_map: Dict[Tuple[str, int], str],
    stem: str,
):
    img = Image.open(img_path).convert("RGB")
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    with open(outline_file, "r") as f:
        for idx, line in enumerate(f):
            poly = parse_line(line)
            if poly is None:
                continue
            pair_id = idx // 2 + 1
            is_nucleus = idx % 2 == 0
            status = status_map.get((stem, pair_id), "base")
            color = status_to_color(status) if is_nucleus else PINK
            draw.line(poly + [poly[0]], fill=color, width=thickness)

    draw_legend(overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    return out_path


def build_dirs(args: argparse.Namespace):
    if args.outline_dir:
        outline_dir = Path(args.outline_dir)
    elif args.dataset:
        outline_dir = Path("data/output/outline") / args.dataset
    else:
        raise SystemExit("Please provide --dataset or --outline-dir")

    if args.image_dir:
        image_dir = Path(args.image_dir)
    elif args.dataset:
        image_dir = Path("data/input") / args.dataset / args.img_subdir
    else:
        raise SystemExit("Please provide --dataset or --image-dir")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("data/output/outline_preview") / outline_dir.name

    return outline_dir, image_dir, out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Render merged outlines txt files as color overlays for quick inspection."
    )
    parser.add_argument("--dataset", help="Folder name such as 0819; maps to data/input/<dataset> and data/output/outline/<dataset>.")
    parser.add_argument("--outline-dir", type=Path, help="Custom outline directory that contains *_merged_cp_outlines.txt files.")
    parser.add_argument("--image-dir", type=Path, help="Custom image directory (PC/DAPI/etc.).")
    parser.add_argument("--img-subdir", default="PC", help="Image subfolder used when --dataset is set, default PC.")
    parser.add_argument(
        "--pred-csv",
        type=Path,
        help="Prediction CSV (with columns Cell_ID, ki67_positive, prediction). "
        "If omitted, will try to pick the newest predictions_xgb_tab_<dataset>_cleaned_*_post.csv under predictions/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for overlay PNGs; defaults to data/output/outline_preview/<dataset>.",
    )
    parser.add_argument(
        "--thickness", type=int, default=2, help="Polyline thickness in pixels, default 2."
    )
    parser.add_argument(
        "--exts",
        default=".jpg,.png,.tif,.tiff,.jpeg",
        help="Comma-separated list of candidate image extensions to search.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N outline files for quick preview.")
    args = parser.parse_args()

    outline_dir, image_dir, out_dir = build_dirs(args)
    exts = [e if e.startswith(".") else f".{e}" for e in args.exts.split(",") if e]

    def resolve_pred_file() -> Optional[Path]:
        if args.pred_csv:
            return Path(args.pred_csv)
        if args.dataset:
            pattern = f"predictions_xgb_tab_{args.dataset}_cleaned_*_post.csv"
            candidates = sorted(Path("predictions").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0]
        return None

    def load_status_map(pred_path: Optional[Path]) -> Dict[Tuple[str, int], str]:
        if not pred_path or not pred_path.exists():
            return {}
        df = pd.read_csv(pred_path)
        mapping: Dict[Tuple[str, int], str] = {}

        def to_flag(val) -> int:
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return int(val)
            m = re.search(r"-?\d+\.?\d*", str(val))
            if m:
                try:
                    return int(float(m.group(0)))
                except ValueError:
                    pass
            return 0

        for _, row in df.iterrows():
            cell_id = str(row.get("Cell_ID", ""))
            if "_" not in cell_id:
                continue
            stem, idx_str = cell_id.rsplit("_", 1)
            try:
                idx = int(float(idx_str))
            except ValueError:
                continue
            ki67 = to_flag(row.get("ki67_positive", 0))
            pred = to_flag(row.get("prediction", 0))
            status = "base"
            if ki67 == 1 and pred == 1:
                status = "tp"
            elif ki67 == 1 and pred == 0:
                status = "fn"
            mapping[(stem, idx)] = status
        return mapping

    outline_files = sorted(outline_dir.glob("*_merged_cp_outlines.txt"))
    if args.limit:
        outline_files = outline_files[: args.limit]
    if not outline_files:
        raise SystemExit(f"No outlines found in {outline_dir}")

    pred_file = resolve_pred_file()
    status_map = load_status_map(pred_file)

    print(f"[INFO] image dir: {image_dir}")
    print(f"[INFO] outlines dir: {outline_dir}")
    print(f"[INFO] output dir: {out_dir}")
    if pred_file:
        print(f"[INFO] prediction csv: {pred_file}")
    else:
        print("[INFO] prediction csv: (none found; all cells will be pink)")

    for outline_file in outline_files:
        stem = outline_file.name.replace("_merged_cp_outlines.txt", "")
        img_path = find_image(stem, image_dir, exts)
        if not img_path:
            print(f"[WARN] Missing image for {stem} (looked in {image_dir})")
            continue
        out_path = out_dir / f"{stem}_overlay.png"
        draw_outlines(img_path, outline_file, out_path, args.thickness, status_map, stem)
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
