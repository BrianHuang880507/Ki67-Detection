# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import trange

DEFAULT_IMAGE_DIR = Path("data/input/0819/PC")
DEFAULT_MASKS_DIR = Path("data/output/segment/0819")
DEFAULT_KI67_DIR = Path("data/output/binary/0819")
DEFAULT_OUT_ROOT = Path("data/output/cyto_crops/0819")

KI67_THRESH = 0.10
PAD = 6


def load_masks_safely(npy_path: Path):
    arr = np.load(npy_path, allow_pickle=True)
    outlines = None
    if isinstance(arr, dict) or (arr.dtype == object and arr.shape == ()):  # noqa: E721
        arr = arr.item()
        if "masks" in arr:
            outlines = arr.get("outlines", None)
            arr = arr["masks"]
    return arr, outlines


def to_label_map(masks, hw):
    H, W = hw
    if masks is None:
        return None
    if masks.ndim == 2:
        if masks.shape == (H, W):
            return masks.astype(np.int32)
        squeezed = np.squeeze(masks)
        if squeezed.ndim == 2 and squeezed.shape == (H, W):
            return squeezed.astype(np.int32)
        raise ValueError("2D masks dimensions do not match image size")
    elif masks.ndim == 3:
        if masks.shape[0] == H and masks.shape[1] == W:  # (H,W,N)
            lbl = np.zeros((H, W), dtype=np.int32)
            k = 0
            for i in range(masks.shape[2]):
                bm = masks[..., i] > 0
                if bm.any():
                    k += 1
                    lbl[bm] = k
            return lbl
        if masks.shape[-1] == W and masks.shape[-2] == H:  # (N,H,W)
            lbl = np.zeros((H, W), dtype=np.int32)
            k = 0
            for i in range(masks.shape[0]):
                bm = masks[i] > 0
                if bm.any():
                    k += 1
                    lbl[bm] = k
            return lbl
        squeezed = np.squeeze(masks)
        if squeezed.ndim == 2 and squeezed.shape == (H, W):
            return squeezed.astype(np.int32)
        raise ValueError("3D masks dimensions do not match image size")
    else:
        raise ValueError("Unsupported mask dimensions")


def farthest_interior_point(bm):
    try:
        from scipy import ndimage as ndi

        dist = ndi.distance_transform_edt(bm)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        return int(y), int(x)
    except Exception:
        ys, xs = np.where(bm)
        if ys.size == 0:
            return None
        cy, cx = int(np.median(ys)), int(np.median(xs))
        return cy + 2, cx + 2


def compute_outlines_for_label(bm):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    try:
        from scipy import ndimage as ndi

        er = ndi.binary_erosion(bm, structure=kernel, iterations=1, border_value=0)
    except Exception:
        er = bm.copy()
        H, W = bm.shape
        yy, xx = np.where(bm)
        for y, x in zip(yy, xx):
            for dy, dx in ((0, 1), (1, 0), (-1, 0), (0, -1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W) or not bm[ny, nx]:
                    er[y, x] = False
                    break
    return np.logical_and(bm, np.logical_not(er))


def compute_ki67_positive_ids(label_map, ki67_mask_bool, threshold=0.10):
    if ki67_mask_bool.shape != label_map.shape:
        raise ValueError("Ki-67 mask dimensions do not match label map")
    labels = np.unique(label_map)
    labels = labels[labels != 0]
    pos = set()
    for lb in labels:
        bm = label_map == lb
        roi_area = int(bm.sum())
        if roi_area == 0:
            continue
        overlap_area = int(np.logical_and(bm, ki67_mask_bool).sum())
        if overlap_area / roi_area >= float(threshold):
            pos.add(int(lb))
    return pos


def draw_overlay_with_outlines(img, label_map, positive_ids, out_path):
    overlay = img.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    H, W = label_map.shape
    YELLOW = (255, 255, 0, 255)
    GREEN = (0, 255, 0, 255)

    labels = np.unique(label_map)
    labels = labels[labels != 0]

    for lb in labels:
        bm = label_map == lb
        if not bm.any():
            continue
        boundary = compute_outlines_for_label(bm)
        ys, xs = np.where(boundary)
        ocolor = GREEN if lb in positive_ids else YELLOW
        for y, x in zip(ys, xs):
            if 0 <= x < W and 0 <= y < H:
                overlay.putpixel((x, y), ocolor)

    font = ImageFont.load_default()
    for lb in labels:
        bm = label_map == lb
        if not bm.any():
            continue
        farthest = farthest_interior_point(bm)
        if farthest is None:
            continue
        cy, cx = farthest
        tcolor = GREEN if lb in positive_ids else YELLOW
        if font is not None and (0 <= cx < W and 0 <= cy < H):
            draw.text((cx + 3, cy + 3), str(int(lb)), fill=tcolor, font=font)

    overlay.save(out_path)
    print(f"[INFO] overlay saved to {out_path}")


def process_one(
    image_path: Path,
    masks_dir: Path,
    ki67_dir: Path,
    out_root: Path,
    ki67_thresh: float,
):
    """Generate overlay and cytoplasm crops for a single image."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    mask_path = masks_dir / f"{image_path.stem}_cyto_seg.npy"
    if not mask_path.exists():
        print(f"[WARN] Missing mask: {mask_path}")
        return

    masks, _ = load_masks_safely(mask_path)
    if masks is None:
        return
    label_map = to_label_map(masks, (H, W))
    if label_map is None:
        print(f"[WARN] Unable to build label map for {image_path.name}")
        return

    out_dir = out_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_ids = set()
    ki67_path = ki67_dir / f"Ki67-{image_path.stem}_binary.png"
    if ki67_path.exists():
        from skimage import io as skio

        ki67_mask = skio.imread(str(ki67_path))
        if ki67_mask.ndim == 3:
            ki67_mask = ki67_mask[..., 0]
        ki67_bool = ki67_mask > 0
        positive_ids = compute_ki67_positive_ids(label_map, ki67_bool, ki67_thresh)

    draw_overlay_with_outlines(
        img, label_map, positive_ids, out_dir / f"{image_path.stem}_overlay.png"
    )

    count = 0
    labels = np.unique(label_map)
    labels = labels[labels != 0]
    for i, lb in enumerate(labels, start=1):
        bm = (label_map == lb).astype(np.uint8)
        ys, xs = np.where(bm > 0)
        if xs.size == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD)
        x1, y1 = min(W - 1, x1 + PAD), min(H - 1, y1 + PAD)
        crop = img.crop((x0, y0, x1 + 1, y1 + 1))
        alpha = Image.fromarray((bm[y0 : y1 + 1, x0 : x1 + 1] * 255).astype(np.uint8))
        rgba = crop.copy()
        rgba.putalpha(alpha)
        rgba.save(out_dir / f"{image_path.stem}_cyto_{i:03d}.png")
        count += 1

    print(f"[INFO] {image_path.name}: exported {count} crops")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop cytoplasm regions and overlays using segmentation results."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing source images (.jpg).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=DEFAULT_MASKS_DIR,
        help="Directory containing cytoplasm segmentation npy files.",
    )
    parser.add_argument(
        "--ki67-dir",
        type=Path,
        default=DEFAULT_KI67_DIR,
        help="Directory containing Ki-67 binary masks.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root directory where overlays and crops will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = args.image_dir.expanduser()
    masks_dir = args.masks_dir.expanduser()
    ki67_dir = args.ki67_dir.expanduser()
    out_root = args.out_root.expanduser()

    for path, label in (
        (image_dir, "--image-dir"),
        (masks_dir, "--masks-dir"),
        (ki67_dir, "--ki67-dir"),
    ):
        if not path.is_dir():
            raise SystemExit(f"[ERROR] {label} not found or not a directory: {path}")

    out_root.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.jpg"))
    print(f"[INFO] Found {len(images)} images in {image_dir}")
    if not images:
        return

    for idx in trange(len(images), desc="Processing"):
        process_one(images[idx], masks_dir, ki67_dir, out_root, KI67_THRESH)


if __name__ == "__main__":
    main()
