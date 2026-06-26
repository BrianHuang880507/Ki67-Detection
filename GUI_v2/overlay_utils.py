import os
import cv2
import numpy as np
from pathlib import Path


_PALETTE = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
    [0, 128, 255], [128, 255, 0],
]


def get_overlay_path(file_path, overlay_dir, paired=False):
    suffix = "_overlay_paired.npy" if paired else "_overlay_indep.npy"
    return os.path.join(overlay_dir, Path(file_path).stem + suffix)


def load_mask_file(file_path, suffix):
    mask_path = os.path.splitext(file_path)[0] + suffix
    if not os.path.exists(mask_path):
        return None
    try:
        seg_data = np.load(mask_path, allow_pickle=True).item()
        mask = seg_data.get('masks', None)
        if mask is not None:
            img = cv2.imread(file_path)
            if img is not None and mask.shape != img.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
        return mask
    except Exception:
        return None


def find_paired_labels(cyto_mask, nuc_mask):
    pairs = []
    for nuc_label in np.unique(nuc_mask):
        if nuc_label == 0:
            continue
        coords = np.argwhere(nuc_mask == nuc_label)
        cy, cx = coords.mean(axis=0).astype(int)
        cyto_label = cyto_mask[cy, cx]
        if cyto_label != 0:
            pairs.append((cyto_label, nuc_label))
    return pairs


def apply_overlay(overlay, cyto_mask, nuc_mask, paired=False):
    """Apply mask overlay onto image array in-place."""
    if nuc_mask is not None:
        if paired:
            _apply_paired(overlay, cyto_mask, nuc_mask)
        else:
            _apply_independent(overlay, cyto_mask, nuc_mask)
    else:
        _apply_cyto_only(overlay, cyto_mask)


def _apply_independent(overlay, cyto_mask, nuc_mask):
    """Render all cyto + all nuc masks independently, no pairing required."""
    _apply_cyto_only(overlay, cyto_mask)
    nuc_labels = np.unique(nuc_mask)
    nuc_labels = nuc_labels[nuc_labels != 0]
    if len(nuc_labels) == 0:
        return
    color_layer = np.zeros_like(overlay)
    paint_mask = np.zeros(overlay.shape[:2], dtype=bool)
    for label in nuc_labels:
        region = (nuc_mask == label)
        color_layer[region] = [0, 0, 240]
        paint_mask |= region
    alpha = 0.3
    overlay[paint_mask] = (
        alpha * color_layer[paint_mask] + (1 - alpha) * overlay[paint_mask]
    ).astype(np.uint8)
    for label in nuc_labels:
        nuc_bin = (nuc_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(nuc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)


def _apply_paired(overlay, cyto_mask, nuc_mask):
    pairs = find_paired_labels(cyto_mask, nuc_mask)
    if not pairs:
        return
    color_layer = np.zeros_like(overlay)
    paint_mask = np.zeros(overlay.shape[:2], dtype=bool)
    for idx, (cyto_label, nuc_label) in enumerate(pairs):
        color = _PALETTE[idx % len(_PALETTE)]
        cyto_region = (cyto_mask == cyto_label)
        color_layer[cyto_region] = color
        paint_mask |= cyto_region
        nuc_region = (nuc_mask == nuc_label)
        color_layer[nuc_region] = [0, 0, 240]
        paint_mask |= nuc_region
    alpha = 0.3
    overlay[paint_mask] = (
        alpha * color_layer[paint_mask] + (1 - alpha) * overlay[paint_mask]
    ).astype(np.uint8)
    for cyto_label, nuc_label in pairs:
        cyto_bin = (cyto_mask == cyto_label).astype(np.uint8)
        contours, _ = cv2.findContours(cyto_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 190, 255), 1)
        nuc_bin = (nuc_mask == nuc_label).astype(np.uint8)
        contours, _ = cv2.findContours(nuc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)


def _apply_cyto_only(overlay, cyto_mask):
    unique_labels = np.unique(cyto_mask)
    unique_labels = unique_labels[unique_labels != 0]
    if len(unique_labels) == 0:
        return
    color_layer = np.zeros_like(overlay)
    paint_mask = np.zeros(overlay.shape[:2], dtype=bool)
    for label in unique_labels:
        color = _PALETTE[(label - 1) % len(_PALETTE)]
        region = (cyto_mask == label)
        color_layer[region] = color
        paint_mask |= region
    alpha = 0.3
    overlay[paint_mask] = (
        alpha * color_layer[paint_mask] + (1 - alpha) * overlay[paint_mask]
    ).astype(np.uint8)
    for label in unique_labels:
        cyto_bin = (cyto_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(cyto_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 190, 255), 1)


def render_and_save_overlay(file_path, overlay_dir):
    """Pre-render both overlay modes and save as .npy files. Returns True on success."""
    image = cv2.imread(file_path)
    if image is None:
        return False
    cyto_mask = load_mask_file(file_path, "_cyto_seg.npy")
    if cyto_mask is None:
        return False
    nuc_mask = load_mask_file(file_path, "_nuc_seg.npy")
    base = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    os.makedirs(overlay_dir, exist_ok=True)

    overlay_indep = base.copy()
    apply_overlay(overlay_indep, cyto_mask, nuc_mask, paired=False)
    np.save(get_overlay_path(file_path, overlay_dir, paired=False), overlay_indep)

    overlay_paired = base.copy()
    apply_overlay(overlay_paired, cyto_mask, nuc_mask, paired=True)
    np.save(get_overlay_path(file_path, overlay_dir, paired=True), overlay_paired)

    return True
