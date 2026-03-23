import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw


OUTLINE_SUFFIX_PATTERN = re.compile(r"_(?:merge|merged)_cp_outlines$", re.IGNORECASE)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_color(value: str) -> Tuple[int, int, int]:
    """Parse a color string expressed as #RRGGBB or R,G,B."""
    text = value.strip()
    if text.startswith("#"):
        hex_value = text[1:]
        if len(hex_value) != 6:
            raise argparse.ArgumentTypeError(f"Hex color must be 6 digits: {value}")
        try:
            return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid hex color: {value}") from exc

    parts = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip()]
    if len(parts) == 3:
        try:
            rgb = tuple(int(p) for p in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid RGB color: {value}") from exc
        if any(comp < 0 or comp > 255 for comp in rgb):
            raise argparse.ArgumentTypeError(f"RGB components must be 0-255: {value}")
        return rgb

    raise argparse.ArgumentTypeError(
        f"Color must be #RRGGBB or 'R,G,B' format (received {value!r})"
    )


def resolve_directory(path_str: str, search_roots: Iterable[Path]) -> Path:
    """Resolve a directory path, checking optional search roots."""
    candidate = Path(path_str)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Directory not found: {candidate}")
        if not candidate.is_dir():
            raise NotADirectoryError(f"Expected directory, got: {candidate}")
        return candidate

    for root in search_roots:
        resolved = root / candidate
        if resolved.exists() and resolved.is_dir():
            return resolved

    resolved = Path.cwd() / candidate
    if resolved.exists() and resolved.is_dir():
        return resolved

    tried = [str(root / candidate) for root in search_roots]
    tried.append(str(Path.cwd() / candidate))
    raise FileNotFoundError(
        f"Unable to locate directory '{path_str}'. Tried: {', '.join(tried)}"
    )


def load_outlines(outline_path: Path) -> List[List[Tuple[int, int]]]:
    """Load outline polygons from a comma-separated text file."""
    outlines: List[List[Tuple[int, int]]] = []
    with outline_path.open("r", encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = [part for part in line.split(",") if part]
            if len(parts) < 6:
                print(f"[WARN] {outline_path.name} line {row_index}: too few points.")
                continue
            try:
                coords = [int(p) for p in parts]
            except ValueError:
                print(f"[WARN] {outline_path.name} line {row_index}: non-integer data.")
                continue
            if len(coords) % 2 != 0:
                print(f"[WARN] {outline_path.name} line {row_index}: uneven coordinates.")
                continue
            points = list(zip(coords[::2], coords[1::2]))
            outlines.append(points)
    return outlines


def infer_image_key(outline_file: Path) -> str:
    """Infer the image stem that should match an outline file."""
    stem = outline_file.stem
    match = OUTLINE_SUFFIX_PATTERN.search(stem)
    if match:
        return stem[: match.start()]
    return stem


def find_matching_image(image_map: Dict[str, Path], key: str) -> Optional[Path]:
    """Return the image path matching the supplied key, if any."""
    return image_map.get(key.lower())


def prepare_image_map(image_dir: Path) -> Dict[str, Path]:
    """Index images in a directory by lowercase stem."""
    image_map: Dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.stem.lower()
            if key in image_map:
                print(
                    f"[WARN] Duplicate image stem detected; "
                    f"overwriting {image_map[key].name} with {path.name}"
                )
            image_map[key] = path
    if not image_map:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported images found in {image_dir} (accepted: {allowed})."
        )
    return image_map


def draw_outlines(
    image: Image.Image,
    outlines: List[List[Tuple[int, int]]],
    color: Tuple[int, int, int],
    line_width: int,
) -> None:
    """Render each outline polygon on top of the given image."""
    painter = ImageDraw.Draw(image)
    for polygon in outlines:
        if len(polygon) < 2:
            continue
        painter.line(polygon + [polygon[0]], fill=color, width=line_width)


def build_output_dir(
    image_dir: Path, outline_dir: Path, requested_output: Optional[str]
) -> Path:
    """Resolve the output directory for rendered overlays."""
    if requested_output:
        output_dir = Path(requested_output)
        return output_dir if output_dir.is_absolute() else (Path.cwd() / output_dir)

    base = Path("data/output/overlay")
    return base / outline_dir.name / image_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay Cellpose outline text files onto source images without ImageJ.",
    )
    parser.add_argument(
        "--outline-folder",
        required=True,
        help="Folder containing *_merge[d]_cp_outlines.txt files.",
    )
    parser.add_argument(
        "--image-folder",
        required=True,
        help="Folder containing the source images to receive outlines.",
    )
    parser.add_argument(
        "--output-dir",
        help="Destination folder for overlay images "
        "(default: data/output/overlay/<outline>/<image>).",
    )
    parser.add_argument(
        "--color",
        type=parse_color,
        default=(255, 0, 0),
        help="Outline color in #RRGGBB or R,G,B format (default: #FF0000).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Outline stroke width in pixels (default: 2).",
    )

    args = parser.parse_args()

    outline_dir = resolve_directory(
        args.outline_folder, search_roots=[Path("data/output/outline")]
    )
    image_dir = resolve_directory(
        args.image_folder, search_roots=[Path("data/input"), Path("data/output")]
    )
    output_dir = build_output_dir(image_dir, outline_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[INFO] Outline folder: {outline_dir}")
    print(f"[INFO] Image folder:   {image_dir}")
    print(f"[INFO] Output folder:  {output_dir}")
    print(f"[INFO] Outline color:  {args.color}")
    print(f"[INFO] Line width:     {args.line_width}")
    print("=" * 60)

    image_map = prepare_image_map(image_dir)

    outline_files = sorted(
        path
        for path in outline_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )
    if not outline_files:
        raise FileNotFoundError(f"No outline .txt files found in {outline_dir}.")

    processed = 0
    missing = 0
    for outline_file in outline_files:
        outlines = load_outlines(outline_file)
        if not outlines:
            print(f"[WARN] {outline_file.name} contains no valid outlines; skipped.")
            continue

        image_key = infer_image_key(outline_file)
        image_path = find_matching_image(image_map, image_key)
        if image_path is None:
            print(f"[WARN] Missing matching image '{image_key}' for {outline_file.name}.")
            missing += 1
            continue

        with Image.open(image_path) as source_image:
            overlay = source_image.convert("RGB")
            draw_outlines(overlay, outlines, args.color, args.line_width)
            output_name = f"{image_path.stem}_with_outlines{image_path.suffix}"
            destination = output_dir / output_name
            overlay.save(destination)
            processed += 1
            print(f"[DONE] {destination}")

    print("=" * 60)
    print(f"[STATS] Successful overlays: {processed}")
    print(f"[STATS] Missing images:      {missing}")
    print(f"[STATS] Outline files seen:  {len(outline_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
