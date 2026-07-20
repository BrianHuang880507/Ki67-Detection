from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


_PALETTE_RGB = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
)


@dataclass(frozen=True)
class LabelRegion:
    """保存單一 label 的裁切遮罩與全圖輪廓。"""

    label: int
    top: int
    left: int
    mask: np.ndarray
    contours: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PairedLabelRegions:
    """保存一組 SegUI 顯示配對的細胞核與細胞質區域。"""

    cytoplasm: LabelRegion
    nucleus: LabelRegion


@dataclass(frozen=True)
class PairedOverlayData:
    """保存單張影像的 Paired Only 顯示資料與數量統計。"""

    pairs: tuple[PairedLabelRegions, ...]
    raw_nucleus_count: int
    raw_cytoplasm_count: int
    paired_nucleus_count: int
    paired_cytoplasm_count: int


def find_paired_labels(
    cytoplasm_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """依 SegUI 的 nucleus 中心點規則建立顯示配對。

    Args:
        cytoplasm_mask: 細胞質 label mask，背景值為 0。
        nucleus_mask: 細胞核 label mask，背景值為 0。

    Returns:
        依 nucleus label 順序排列的 ``(cytoplasm_label, nucleus_label)``。

    Raises:
        ValueError: mask 不是二維陣列或兩者尺寸不同時拋出。
    """
    cytoplasm_mask = np.asarray(cytoplasm_mask)
    nucleus_mask = np.asarray(nucleus_mask)
    if cytoplasm_mask.ndim != 2 or nucleus_mask.ndim != 2:
        raise ValueError("cytoplasm_mask 與 nucleus_mask 必須是二維陣列")
    if cytoplasm_mask.shape != nucleus_mask.shape:
        raise ValueError("cytoplasm_mask 與 nucleus_mask 尺寸必須相同")

    pairs: list[tuple[int, int]] = []
    for nucleus_label in np.unique(nucleus_mask):
        if nucleus_label == 0:
            continue
        coords = np.argwhere(nucleus_mask == nucleus_label)
        if coords.size == 0:
            continue
        center_y, center_x = coords.mean(axis=0).astype(int)
        cytoplasm_label = int(cytoplasm_mask[center_y, center_x])
        if cytoplasm_label != 0:
            pairs.append((cytoplasm_label, int(nucleus_label)))
    return pairs


def _extract_label_region(mask: np.ndarray, label: int) -> LabelRegion:
    """將單一 label 壓縮成 bounding box 內的布林遮罩。"""
    coords = np.argwhere(mask == label)
    if coords.size == 0:
        raise ValueError(f"mask 不包含 label {label}")

    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0) + 1
    cropped = (mask[top:bottom, left:right] == label)
    contours, _ = cv2.findContours(
        cropped.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    global_contours: list[np.ndarray] = []
    for contour in contours:
        shifted = contour.astype(np.int32, copy=True)
        shifted[:, 0, 0] += int(left)
        shifted[:, 0, 1] += int(top)
        global_contours.append(shifted)

    return LabelRegion(
        label=int(label),
        top=int(top),
        left=int(left),
        mask=cropped,
        contours=tuple(global_contours),
    )


def build_paired_overlay_data(
    cytoplasm_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> PairedOverlayData:
    """從完整 label masks 建立輕量化 Paired Only 顯示資料。

    Args:
        cytoplasm_mask: 細胞質 label mask。
        nucleus_mask: 細胞核 label mask。

    Returns:
        可供 GUI 與 PNG 輸出共用的配對區域資料。
    """
    cytoplasm_mask = np.asarray(cytoplasm_mask)
    nucleus_mask = np.asarray(nucleus_mask)
    label_pairs = find_paired_labels(cytoplasm_mask, nucleus_mask)

    cytoplasm_regions: dict[int, LabelRegion] = {}
    nucleus_regions: dict[int, LabelRegion] = {}
    paired_regions: list[PairedLabelRegions] = []
    for cytoplasm_label, nucleus_label in label_pairs:
        if cytoplasm_label not in cytoplasm_regions:
            cytoplasm_regions[cytoplasm_label] = _extract_label_region(
                cytoplasm_mask, cytoplasm_label
            )
        if nucleus_label not in nucleus_regions:
            nucleus_regions[nucleus_label] = _extract_label_region(
                nucleus_mask, nucleus_label
            )
        paired_regions.append(
            PairedLabelRegions(
                cytoplasm=cytoplasm_regions[cytoplasm_label],
                nucleus=nucleus_regions[nucleus_label],
            )
        )

    raw_nucleus_labels = np.unique(nucleus_mask)
    raw_cytoplasm_labels = np.unique(cytoplasm_mask)
    return PairedOverlayData(
        pairs=tuple(paired_regions),
        raw_nucleus_count=int(np.count_nonzero(raw_nucleus_labels)),
        raw_cytoplasm_count=int(np.count_nonzero(raw_cytoplasm_labels)),
        paired_nucleus_count=len({nucleus for _, nucleus in label_pairs}),
        paired_cytoplasm_count=len({cytoplasm for cytoplasm, _ in label_pairs}),
    )


def _paint_region(
    color_layer: np.ndarray,
    paint_mask: np.ndarray,
    region: LabelRegion,
    color_bgr: tuple[int, int, int],
) -> None:
    """在全圖 color layer 上繪製裁切後的 label 區域。"""
    height, width = region.mask.shape
    row_slice = slice(region.top, region.top + height)
    col_slice = slice(region.left, region.left + width)
    color_view = color_layer[row_slice, col_slice]
    mask_view = paint_mask[row_slice, col_slice]
    color_view[region.mask] = color_bgr
    mask_view[region.mask] = True


def render_paired_overlay_bgr(
    base_bgr: np.ndarray,
    data: PairedOverlayData,
    *,
    show_nucleus: bool = True,
    show_cytoplasm: bool = True,
    alpha: float = 0.5,
) -> np.ndarray:
    """將 Paired Only 核質區域直接繪製在 BGR 原圖上。

    Args:
        base_bgr: OpenCV BGR 原圖。
        data: SegUI 相容的配對顯示資料。
        show_nucleus: 是否顯示已配對細胞核。
        show_cytoplasm: 是否顯示已配對細胞質。
        alpha: 填色透明度，範圍為 0 到 1。

    Returns:
        套用 Paired Only 半透明填色與輪廓後的 BGR 影像。

    Raises:
        ValueError: 原圖格式或透明度不合法時拋出。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必須介於 0 與 1 之間")
    if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("base_bgr 必須是三通道 BGR 影像")

    overlay = base_bgr.copy()
    color_layer = np.zeros_like(overlay)
    paint_mask = np.zeros(overlay.shape[:2], dtype=bool)
    palette_bgr = tuple((blue, green, red) for red, green, blue in _PALETTE_RGB)
    nucleus_fill_bgr = (240, 0, 0)

    for index, pair in enumerate(data.pairs):
        if show_cytoplasm:
            _paint_region(
                color_layer,
                paint_mask,
                pair.cytoplasm,
                palette_bgr[index % len(palette_bgr)],
            )
        if show_nucleus:
            _paint_region(
                color_layer,
                paint_mask,
                pair.nucleus,
                nucleus_fill_bgr,
            )

    overlay[paint_mask] = (
        alpha * color_layer[paint_mask] + (1.0 - alpha) * overlay[paint_mask]
    ).astype(np.uint8)

    for pair in data.pairs:
        if show_cytoplasm:
            cv2.drawContours(overlay, pair.cytoplasm.contours, -1, (255, 190, 0), 1)
        if show_nucleus:
            cv2.drawContours(overlay, pair.nucleus.contours, -1, (255, 0, 0), 1)
    return overlay


def _load_mask(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    """載入 Cellpose mask，必要時以 nearest-neighbor 對齊原圖。"""
    if not mask_path.exists():
        return None
    try:
        payload = np.load(mask_path, allow_pickle=True).item()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict) or "masks" not in payload:
        return None

    mask = np.asarray(payload["masks"])
    height, width = image_shape
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.int32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask.astype(np.int32, copy=False)


def collect_paired_overlay_data(
    image_files: Sequence[Path],
    segment_dir: Path,
) -> dict[str, PairedOverlayData]:
    """在 segmentation 暫存檔清理前收集各影像的配對顯示資料。

    Args:
        image_files: GUI 將顯示的原始影像清單。
        segment_dir: ``*_cyto_seg.npy`` 與 ``*_nuc_seg.npy`` 所在資料夾。

    Returns:
        以影像 stem 為索引的記憶體配對資料；缺少任一 mask 的影像會略過。
    """
    collected: dict[str, PairedOverlayData] = {}
    for image_file in image_files:
        image = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        image_shape = image.shape[:2]
        cytoplasm_mask = _load_mask(
            segment_dir / f"{image_file.stem}_cyto_seg.npy", image_shape
        )
        nucleus_mask = _load_mask(
            segment_dir / f"{image_file.stem}_nuc_seg.npy", image_shape
        )
        if cytoplasm_mask is None or nucleus_mask is None:
            continue
        collected[image_file.stem] = build_paired_overlay_data(
            cytoplasm_mask, nucleus_mask
        )
    return collected
