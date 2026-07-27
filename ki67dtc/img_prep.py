import shutil
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from tqdm import trange
from cellpose import models, io
from shapely.geometry import Point, Polygon
from skimage.draw import polygon
from PIL import Image


from ki67dtc.utils.io import list_files, output_dir, load_outlines, remove_temp_files

# 預設 Cellpose 模型設定
CYTO_MODEL_PATH = "model/model_BDL6_label_new"
DAPI_NUC_MODEL_PATH = "cyto3"
PC_NUC_MODEL_PATH = "model/model_BDL3_label_dapi"
# 保留舊常數名稱供既有 import 使用；目前代表 DAPI nucleus model。
NUC_MODEL_PATH = DAPI_NUC_MODEL_PATH
CYTO_MODEL_INPUT_SIZE = (1280, 1024)
NUC_MODEL_INPUT_SIZE = (1280, 1024)
PAIRED_FILTER_MIN_REFERENCE_COUNT = 10
PAIRED_FILTER_MIN_NUCLEUS_CONTAINMENT = 0.8


@dataclass(frozen=True)
class PairedAreaFilterResult:
    """儲存配對感知面積過濾的結果與診斷資訊。

    Attributes:
        cytoplasm_mask: 過濾並重新編號後的細胞質 mask。
        nucleus_mask: 過濾並重新編號後的細胞核 mask。
        cytoplasm_threshold: 細胞質最小面積門檻。
        nucleus_threshold: 細胞核最小面積門檻。
        paired_cytoplasm_count: 過濾前成功配對的細胞質數量。
        paired_nucleus_count: 過濾前成功配對的細胞核數量。
        reference_cytoplasm_count: 用於細胞質中位數的標籤數量。
        reference_nucleus_count: 用於細胞核中位數的標籤數量。
        removed_cytoplasm_count: 被移除的未配對細胞質數量。
        removed_nucleus_count: 被移除的未配對細胞核數量。
    """

    cytoplasm_mask: np.ndarray
    nucleus_mask: np.ndarray
    cytoplasm_threshold: float
    nucleus_threshold: float
    paired_cytoplasm_count: int
    paired_nucleus_count: int
    reference_cytoplasm_count: int
    reference_nucleus_count: int
    removed_cytoplasm_count: int
    removed_nucleus_count: int


# ===============================
# 影像分割
# ===============================
def _filter_small_labels(
    masks: np.ndarray,
    min_area_ratio: float,
    min_area_floor: int,
) -> np.ndarray:
    """依標籤面積中位數移除過小物件，並重新連續編號。

    Args:
        masks: Cellpose 輸出的二維標籤 mask，背景值為 0。
        min_area_ratio: 相對於非零標籤面積中位數的最小面積比例。
        min_area_floor: 最小面積的絕對像素下限。

    Returns:
        過濾並重新連續編號的標籤 mask；沒有標籤或中位數無效時原樣回傳。
    """
    mask_array = np.asarray(masks)
    nonzero_values = mask_array[mask_array != 0]
    if nonzero_values.size == 0:
        return masks

    labels, areas = np.unique(nonzero_values, return_counts=True)
    median_area = float(np.median(areas))
    if not np.isfinite(median_area):
        return masks

    threshold = max(float(min_area_floor), float(min_area_ratio) * median_area)
    kept_labels = labels[areas >= threshold]

    unique_labels, inverse = np.unique(mask_array, return_inverse=True)
    remapped_labels = np.zeros(unique_labels.shape, dtype=mask_array.dtype)
    kept_positions = np.flatnonzero(np.isin(unique_labels, kept_labels))
    remapped_labels[kept_positions] = np.arange(
        1,
        kept_positions.size + 1,
        dtype=mask_array.dtype,
    )
    return remapped_labels[inverse].reshape(mask_array.shape)


def _label_area_map(mask: np.ndarray) -> dict[int, int]:
    """計算非背景標籤的像素面積。"""
    mask_array = np.asarray(mask)
    nonzero_values = mask_array[mask_array != 0]
    if nonzero_values.size == 0:
        return {}
    labels, areas = np.unique(nonzero_values, return_counts=True)
    return {
        int(label): int(area)
        for label, area in zip(labels.tolist(), areas.tolist(), strict=True)
    }


def _border_labels(mask: np.ndarray) -> set[int]:
    """找出接觸影像邊界的非背景標籤。"""
    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return set()
    border_values = np.concatenate(
        (
            mask_array[0, :],
            mask_array[-1, :],
            mask_array[:, 0],
            mask_array[:, -1],
        )
    )
    return {int(label) for label in np.unique(border_values) if label != 0}


def _relabel_kept_labels(
    mask: np.ndarray,
    kept_labels: set[int],
) -> np.ndarray:
    """只保留指定標籤，並依原標籤順序重新連續編號。"""
    mask_array = np.asarray(mask)
    unique_labels, inverse = np.unique(mask_array, return_inverse=True)
    remapped_labels = np.zeros(unique_labels.shape, dtype=mask_array.dtype)
    kept_positions = np.flatnonzero(
        np.isin(unique_labels, np.asarray(sorted(kept_labels), dtype=mask_array.dtype))
    )
    remapped_labels[kept_positions] = np.arange(
        1,
        kept_positions.size + 1,
        dtype=mask_array.dtype,
    )
    return remapped_labels[inverse].reshape(mask_array.shape)


def _filter_small_unpaired_labels(
    cytoplasm_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    min_area_ratio: float,
    min_area_floor: int,
    *,
    min_reference_count: int = PAIRED_FILTER_MIN_REFERENCE_COUNT,
    min_nucleus_containment: float = PAIRED_FILTER_MIN_NUCLEUS_CONTAINMENT,
) -> PairedAreaFilterResult:
    """以配對 ROI 建立面積基準，只移除未配對的小物件。

    細胞核質心落在細胞質內即視為配對。所有配對物件均受到保護，不會因
    面積過小而移除。中位數優先使用核位於細胞質內至少指定比例、且兩者
    都未接觸影像邊界的高可信配對；樣本不足時退回全部配對，完全無配對
    時才退回全部標籤。

    Args:
        cytoplasm_mask: 細胞質二維標籤 mask。
        nucleus_mask: 細胞核二維標籤 mask，尺寸須與細胞質一致。
        min_area_ratio: 相對於參考面積中位數的最小面積比例。
        min_area_floor: 最小面積的絕對像素下限。
        min_reference_count: 使用高可信配對計算中位數所需的最少標籤數。
        min_nucleus_containment: 高可信配對所需的最小細胞核包覆比例。

    Returns:
        過濾後的兩種 mask、門檻與配對診斷資訊。

    Raises:
        ValueError: 當 mask 尺寸或過濾參數不合法時拋出。
    """
    cyto_array = np.asarray(cytoplasm_mask)
    nuc_array = np.asarray(nucleus_mask)
    if cyto_array.ndim != 2 or nuc_array.ndim != 2:
        raise ValueError("cytoplasm_mask and nucleus_mask must be two-dimensional.")
    if cyto_array.shape != nuc_array.shape:
        raise ValueError("cytoplasm_mask and nucleus_mask must have the same shape.")
    if min_area_ratio < 0 or min_area_floor < 0:
        raise ValueError("Area filter thresholds must be non-negative.")
    if min_reference_count < 1:
        raise ValueError("min_reference_count must be at least 1.")
    if not 0.0 <= min_nucleus_containment <= 1.0:
        raise ValueError("min_nucleus_containment must be between 0 and 1.")

    cyto_areas = _label_area_map(cyto_array)
    nuc_areas = _label_area_map(nuc_array)
    cyto_border = _border_labels(cyto_array)
    nuc_border = _border_labels(nuc_array)
    paired_cyto: set[int] = set()
    paired_nuc: set[int] = set()
    trusted_cyto: set[int] = set()
    trusted_nuc: set[int] = set()

    if nuc_areas:
        pixel_y, pixel_x = np.nonzero(nuc_array)
        pixel_nuc_labels = nuc_array[pixel_y, pixel_x].astype(np.int64)
        nucleus_labels = np.asarray(sorted(nuc_areas), dtype=np.int64)
        label_counts = np.bincount(pixel_nuc_labels)
        center_y = (
            np.bincount(pixel_nuc_labels, weights=pixel_y)[nucleus_labels]
            / label_counts[nucleus_labels]
        ).astype(int)
        center_x = (
            np.bincount(pixel_nuc_labels, weights=pixel_x)[nucleus_labels]
            / label_counts[nucleus_labels]
        ).astype(int)
        centroid_cyto_labels = cyto_array[center_y, center_x].astype(np.int64)

        assigned_cyto_by_nucleus = np.zeros(label_counts.size, dtype=np.int64)
        assigned_cyto_by_nucleus[nucleus_labels] = centroid_cyto_labels
        pixel_cyto_labels = cyto_array[pixel_y, pixel_x].astype(np.int64)
        contained_pixels = (
            pixel_cyto_labels == assigned_cyto_by_nucleus[pixel_nuc_labels]
        )
        contained_counts = np.bincount(
            pixel_nuc_labels[contained_pixels],
            minlength=label_counts.size,
        )

        for nuc_label, cyto_label in zip(
            nucleus_labels.tolist(),
            centroid_cyto_labels.tolist(),
            strict=True,
        ):
            if cyto_label == 0:
                continue

            paired_cyto.add(cyto_label)
            paired_nuc.add(nuc_label)
            containment = contained_counts[nuc_label] / label_counts[nuc_label]
            if (
                containment >= min_nucleus_containment
                and cyto_label not in cyto_border
                and nuc_label not in nuc_border
            ):
                trusted_cyto.add(cyto_label)
                trusted_nuc.add(nuc_label)

    if (
        len(trusted_cyto) >= min_reference_count
        and len(trusted_nuc) >= min_reference_count
    ):
        reference_cyto = trusted_cyto
        reference_nuc = trusted_nuc
    elif paired_cyto and paired_nuc:
        reference_cyto = paired_cyto
        reference_nuc = paired_nuc
    else:
        reference_cyto = set(cyto_areas)
        reference_nuc = set(nuc_areas)

    def calculate_threshold(
        area_by_label: dict[int, int],
        reference_labels: set[int],
    ) -> float:
        reference_areas = [
            area_by_label[label]
            for label in reference_labels
            if label in area_by_label
        ]
        if not reference_areas:
            return 0.0
        median_area = float(np.median(reference_areas))
        return max(float(min_area_floor), float(min_area_ratio) * median_area)

    cyto_threshold = calculate_threshold(cyto_areas, reference_cyto)
    nuc_threshold = calculate_threshold(nuc_areas, reference_nuc)
    kept_cyto = {
        label
        for label, area in cyto_areas.items()
        if label in paired_cyto or area >= cyto_threshold
    }
    kept_nuc = {
        label
        for label, area in nuc_areas.items()
        if label in paired_nuc or area >= nuc_threshold
    }

    return PairedAreaFilterResult(
        cytoplasm_mask=_relabel_kept_labels(cyto_array, kept_cyto),
        nucleus_mask=_relabel_kept_labels(nuc_array, kept_nuc),
        cytoplasm_threshold=cyto_threshold,
        nucleus_threshold=nuc_threshold,
        paired_cytoplasm_count=len(paired_cyto),
        paired_nucleus_count=len(paired_nuc),
        reference_cytoplasm_count=len(reference_cyto),
        reference_nucleus_count=len(reference_nuc),
        removed_cytoplasm_count=len(cyto_areas) - len(kept_cyto),
        removed_nucleus_count=len(nuc_areas) - len(kept_nuc),
    )


def _filter_paired_segment_files(
    seg_dir: Path,
    stems: list[str],
    *,
    min_area_ratio: float,
    min_area_floor: int,
) -> dict[str, PairedAreaFilterResult]:
    """過濾同名 cyto／nuc segmentation 檔並覆寫其 masks。

    Args:
        seg_dir: 儲存 `*_cyto_seg.npy` 與 `*_nuc_seg.npy` 的資料夾。
        stems: 要處理的共同影像 stem。
        min_area_ratio: 相對於配對參考中位數的最小面積比例。
        min_area_floor: 最小面積的絕對像素下限。

    Returns:
        以影像 stem 為 key 的過濾結果；缺少任一 segmentation 檔時略過。
    """
    results: dict[str, PairedAreaFilterResult] = {}
    for stem in dict.fromkeys(stems):
        cyto_path = seg_dir / f"{stem}_cyto_seg.npy"
        nuc_path = seg_dir / f"{stem}_nuc_seg.npy"
        if not cyto_path.exists() or not nuc_path.exists():
            print(f"[WARN] 略過配對面積過濾，缺少 segmentation 檔：{stem}")
            continue

        cyto_data = np.load(cyto_path, allow_pickle=True).item()
        nuc_data = np.load(nuc_path, allow_pickle=True).item()
        result = _filter_small_unpaired_labels(
            cyto_data["masks"],
            nuc_data["masks"],
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
        )
        cyto_data["masks"] = result.cytoplasm_mask
        nuc_data["masks"] = result.nucleus_mask
        np.save(cyto_path, cyto_data)
        np.save(nuc_path, nuc_data)
        results[stem] = result
        print(
            f"[INFO] 配對面積過濾 {stem}："
            f"cyto 移除 {result.removed_cytoplasm_count}、"
            f"nuc 移除 {result.removed_nucleus_count}；"
            f"保護配對 {result.paired_cytoplasm_count}/"
            f"{result.paired_nucleus_count}。"
        )
    return results


def segment(
    model_path: str,
    img_files: list[Path],
    output_dir: Path,
    suffix: str,
    output_stems: list[str] | None = None,
    channels: list[int] | tuple[int, int] = (0, 0),
    diameter: float | None = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    invert: bool = False,
    model_input_size: tuple[int, int] | None = None,
    *,
    min_area_ratio: float = 0.15,
    min_area_floor: int = 30,
    apply_small_label_filter: bool = True,
):
    """使用指定的 Cellpose 模型分割影像。

    分割結果會先由 Cellpose 輸出為 *_seg.npy，再依照 suffix 與
    output_stems 移到目標資料夾，供後續 outline 轉換使用。

    Args:
        model_path: Cellpose 模型路徑或模型名稱。
        img_files: 要分割的影像路徑清單。
        output_dir: segmentation npy 輸出資料夾。
        suffix: 輸出檔名後綴，例如 `cyto` 或 `nuc`。
        output_stems: 自訂輸出 stem；用於 DAPI 影像對應回 PC 檔名。
        channels: Cellpose 通道設定。
        diameter: Cellpose diameter 參數。
        cellprob_threshold: Cellpose cellprob threshold。
        flow_threshold: Cellpose flow threshold。
        invert: 是否反相後推論。
        model_input_size: 若指定 `(width, height)`，先 resize 到模型訓練尺寸
            推論，再將 mask/flow 還原回原圖尺寸。
        min_area_ratio: 相對於標籤面積中位數的最小面積比例。
        min_area_floor: 最小標籤面積的絕對像素下限。
        apply_small_label_filter: 是否在單一 mask 輸出前套用舊版面積過濾。
            `segment_all` 會關閉此項，待 cyto／nuc 配對後再共同過濾。

    Raises:
        ValueError: 當 `output_stems` 長度不符，或 resize 尺寸不合法時拋出。
    """
    if output_stems is not None and len(output_stems) != len(img_files):
        raise ValueError("output_stems length must match img_files length.")
    if model_input_size is not None:
        target_w, target_h = model_input_size
        if target_w <= 0 or target_h <= 0:
            raise ValueError("model_input_size must contain positive width and height.")

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    for i in trange(len(img_files), desc=f"Segmenting ({suffix})"):
        f = img_files[i]
        img = io.imread(f)
        eval_img = img
        eval_diameter = diameter
        original_size: tuple[int, int] | None = None
        if model_input_size is not None:
            target_w, target_h = model_input_size
            orig_h, orig_w = img.shape[:2]
            original_size = (orig_w, orig_h)
            eval_img = cv2.resize(
                img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            eval_diameter = None

        masks, flows, styles = model.eval(
            eval_img,
            diameter=eval_diameter,
            channels=list(channels),
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            invert=invert,
        )
        if original_size is not None:
            masks = cv2.resize(
                masks.astype(np.int32),
                original_size,
                interpolation=cv2.INTER_NEAREST,
            )
            if flows is not None and len(flows) > 0:
                flows = list(flows)
                flows[0] = cv2.resize(
                    flows[0], original_size, interpolation=cv2.INTER_LINEAR
                )

        if apply_small_label_filter:
            masks = _filter_small_labels(masks, min_area_ratio, min_area_floor)
        io.masks_flows_to_seg(
            img, masks, flows, f, channels=list(channels), diams=eval_diameter
        )
        seg_file = f.with_name(f"{f.stem}_seg.npy")
        target_stem = output_stems[i] if output_stems is not None else f.stem
        target_file = output_dir / f"{target_stem}_{suffix}_seg.npy"
        if target_file.exists():
            target_file.unlink()
        shutil.move(seg_file, target_file)


def _last_numeric_token(stem: str) -> int | None:
    """取出檔名 stem 中最後一段數字。

    Args:
        stem (str): 不含副檔名的檔名。

    Returns:
        int | None: 最後一段數字；若沒有數字則回傳 `None`。
    """
    match = re.search(r"(\d+)(?!.*\d)", stem)
    return int(match.group(1)) if match else None


def _label_centroids(mask: np.ndarray) -> np.ndarray:
    """Return label centroids as ``(x, y)`` coordinates."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    labels = mask[ys, xs].astype(np.int64)
    counts = np.bincount(labels)
    sum_x = np.bincount(labels, weights=xs)
    sum_y = np.bincount(labels, weights=ys)
    valid = np.flatnonzero(counts)[1:]
    if valid.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    return np.column_stack(
        (sum_x[valid] / counts[valid], sum_y[valid] / counts[valid])
    ).astype(np.float32)


def _score_nuc_centers_in_cyto(
    cyto_mask: np.ndarray, nuc_centers: np.ndarray
) -> tuple[int, int, float]:
    """Score how many nucleus centers fall inside any cytoplasm mask."""
    total = int(len(nuc_centers))
    if total == 0:
        return 0, 0, 0.0

    h, w = cyto_mask.shape
    xs = np.rint(nuc_centers[:, 0]).astype(np.int32)
    ys = np.rint(nuc_centers[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    inside = np.zeros(total, dtype=bool)
    inside[valid] = cyto_mask[ys[valid], xs[valid]] > 0
    count = int(np.count_nonzero(inside))
    return count, total, count / total


def _linear_assignment(score_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Maximize row/column assignment scores with scipy when available."""
    try:
        from scipy.optimize import linear_sum_assignment

        row_idx, col_idx = linear_sum_assignment(-score_matrix)
        return list(zip(row_idx.tolist(), col_idx.tolist()))
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
            if (
                len(used_rows) == score_matrix.shape[0]
                or len(used_cols) == score_matrix.shape[1]
            ):
                break
        return pairs


def remap_nuc_segments_to_cyto(
    seg_dir: Path,
    cyto_stems: list[str],
    nuc_stems: list[str],
    min_improvement_fraction: float = 0.05,
) -> dict[str, str]:
    """Reassign DAPI nucleus segmentation files to the best matching PC images.

    Some microscope exports keep PC/DAPI file counts identical but shift the
    numeric suffixes by one frame. Filename matching then produces many
    unpaired nuclei/cytoplasms. This post-segmentation check uses nucleus
    centroids and cytoplasm masks to detect and fix that whole-batch mismatch.
    """
    if len(cyto_stems) < 2 or len(nuc_stems) < 2:
        return {}

    cyto_stems = list(dict.fromkeys(cyto_stems))
    nuc_stems = list(dict.fromkeys(nuc_stems))

    cyto_masks: list[np.ndarray] = []
    valid_cyto_stems: list[str] = []
    for stem in cyto_stems:
        path = seg_dir / f"{stem}_cyto_seg.npy"
        if not path.exists():
            continue
        cyto_masks.append(np.load(path, allow_pickle=True).item()["masks"])
        valid_cyto_stems.append(stem)

    nuc_centers: list[np.ndarray] = []
    valid_nuc_stems: list[str] = []
    for stem in nuc_stems:
        path = seg_dir / f"{stem}_nuc_seg.npy"
        if not path.exists():
            continue
        nuc_mask = np.load(path, allow_pickle=True).item()["masks"]
        nuc_centers.append(_label_centroids(nuc_mask))
        valid_nuc_stems.append(stem)

    if len(valid_cyto_stems) < 2 or len(valid_nuc_stems) < 2:
        return {}

    score_matrix = np.zeros(
        (len(valid_cyto_stems), len(valid_nuc_stems)), dtype=np.float32
    )
    inside_counts = np.zeros_like(score_matrix, dtype=np.int32)
    total_counts = np.zeros_like(score_matrix, dtype=np.int32)
    for row, cyto_mask in enumerate(cyto_masks):
        for col, centers in enumerate(nuc_centers):
            inside, total, fraction = _score_nuc_centers_in_cyto(cyto_mask, centers)
            inside_counts[row, col] = inside
            total_counts[row, col] = total
            # Fraction is the primary signal; count only breaks near ties.
            score_matrix[row, col] = float(fraction * 1000.0 + inside * 0.001)

    assignment = _linear_assignment(score_matrix)
    assigned_score = float(sum(score_matrix[row, col] for row, col in assignment))

    stem_to_nuc_col = {stem: col for col, stem in enumerate(valid_nuc_stems)}
    identity_pairs = [
        (row, stem_to_nuc_col[stem])
        for row, stem in enumerate(valid_cyto_stems)
        if stem in stem_to_nuc_col
    ]
    identity_score = float(sum(score_matrix[row, col] for row, col in identity_pairs))

    changed = any(
        valid_cyto_stems[row] != valid_nuc_stems[col] for row, col in assignment
    )
    if not changed:
        print("[INFO] DAPI nucleus masks match PC filenames; no remap needed.")
        return {}

    required_gain = max(1.0, abs(identity_score) * float(min_improvement_fraction))
    if assigned_score <= identity_score + required_gain:
        print(
            "[INFO] DAPI remap skipped; best assignment did not improve enough "
            f"({assigned_score:.1f} vs {identity_score:.1f})."
        )
        return {}

    sources = {valid_nuc_stems[col] for _, col in assignment}
    staged_paths: dict[str, Path] = {}
    for index, source_stem in enumerate(sorted(sources)):
        source = seg_dir / f"{source_stem}_nuc_seg.npy"
        stage = seg_dir / f"__nuc_remap_stage_{index}_{source.name}"
        if stage.exists():
            stage.unlink()
        shutil.move(str(source), str(stage))
        staged_paths[source_stem] = stage

    remapped: dict[str, str] = {}
    for row, col in assignment:
        target_stem = valid_cyto_stems[row]
        source_stem = valid_nuc_stems[col]
        stage = staged_paths[source_stem]
        target = seg_dir / f"{target_stem}_nuc_seg.npy"
        if target.exists():
            target.unlink()
        shutil.move(str(stage), str(target))
        remapped[target_stem] = source_stem

        inside = int(inside_counts[row, col])
        total = int(total_counts[row, col])
        fraction = inside / total if total else 0.0
        if fraction < 0.50:
            print(
                f"[WARN] Low-confidence DAPI remap: {source_stem} -> {target_stem} "
                f"({inside}/{total} nucleus centers inside cyto)."
            )

    print(
        f"[INFO] Remapped {len(remapped)} DAPI nucleus segmentation files "
        f"(assignment score {assigned_score:.1f}, filename score {identity_score:.1f})."
    )
    return remapped


def segment_all(
    input_dir: str,
    nuc_source: str = "dapi",
    dapi_dir_name: str = "DAPI",
    cyto_model_input_size: tuple[int, int] | None = CYTO_MODEL_INPUT_SIZE,
    nuc_model_input_size: tuple[int, int] | None = NUC_MODEL_INPUT_SIZE,
    *,
    cellprob_threshold: float = 0.0,
    min_area_ratio: float = 0.15,
    min_area_floor: int = 30,
):
    """執行細胞質與細胞核分割。

    - 細胞質一律使用 PC 影像。
    - 細胞核可選擇 PC 或 DAPI；若選 DAPI，模型使用 `cyto3` 且通道使用 [3, 3]。
    - 若選 PC，或 DAPI 不存在而 fallback 到 PC，模型使用 GUI_v2 的 PC nucleus model。

    Args:
        input_dir: 資料集根目錄，預期包含 `PC` 與可選的 `DAPI` 資料夾。
        nuc_source: 細胞核來源，支援 `pc` 或 `dapi`。
        dapi_dir_name: DAPI 資料夾名稱。
        cyto_model_input_size: 細胞質模型推論前的 resize 尺寸；`None` 表示不 resize。
        nuc_model_input_size: 細胞核模型推論前的 resize 尺寸；`None` 表示不 resize。
        cellprob_threshold: Cellpose cellprob threshold。
        min_area_ratio: 相對於配對參考標籤面積中位數的最小面積比例。
        min_area_floor: 最小標籤面積的絕對像素下限。
    """
    root_dir = Path(input_dir)
    pc_dir = root_dir / "PC"
    seg_dir = output_dir(root_dir, "segment")

    img_files = [
        f
        for f in list_files(pc_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
        if "ki67" not in f.stem.lower() and "df" not in f.stem.lower()
    ]
    if not img_files:
        print(f"[WARN] 找不到相位差圖片於 {pc_dir}")
        return

    nuc_source = nuc_source.strip().lower()
    if nuc_source not in {"pc", "dapi"}:
        raise ValueError(f"Unsupported nuc_source: {nuc_source}")

    # 細胞質分割一律使用 PC 影像與灰階通道。
    segment(
        CYTO_MODEL_PATH,
        img_files,
        seg_dir,
        "cyto",
        channels=(0, 0),
        model_input_size=cyto_model_input_size,
        cellprob_threshold=cellprob_threshold,
        min_area_ratio=min_area_ratio,
        min_area_floor=min_area_floor,
        apply_small_label_filter=False,
    )

    if nuc_source == "pc":
        segment(
            PC_NUC_MODEL_PATH,
            img_files,
            seg_dir,
            "nuc",
            channels=(0, 0),
            model_input_size=nuc_model_input_size,
            cellprob_threshold=cellprob_threshold,
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
            apply_small_label_filter=False,
        )
        _filter_paired_segment_files(
            seg_dir,
            [image.stem for image in img_files],
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
        )
        return

    dapi_dir = root_dir / dapi_dir_name
    if not dapi_dir.exists() or not dapi_dir.is_dir():
        print(f"[WARN] 找不到 DAPI 資料夾 {dapi_dir}，細胞核改回使用 PC。")
        segment(
            PC_NUC_MODEL_PATH,
            img_files,
            seg_dir,
            "nuc",
            channels=(0, 0),
            model_input_size=nuc_model_input_size,
            cellprob_threshold=cellprob_threshold,
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
            apply_small_label_filter=False,
        )
        _filter_paired_segment_files(
            seg_dir,
            [image.stem for image in img_files],
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
        )
        return

    dapi_files = list_files(dapi_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    if not dapi_files:
        print(f"[WARN] DAPI 資料夾沒有可用影像 {dapi_dir}，細胞核改回使用 PC。")
        segment(
            PC_NUC_MODEL_PATH,
            img_files,
            seg_dir,
            "nuc",
            channels=(0, 0),
            model_input_size=nuc_model_input_size,
            cellprob_threshold=cellprob_threshold,
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
            apply_small_label_filter=False,
        )
        _filter_paired_segment_files(
            seg_dir,
            [image.stem for image in img_files],
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
        )
        return

    dapi_by_stem: dict[str, list[Path]] = {}
    dapi_by_idx: dict[int, list[Path]] = {}
    for dapi in dapi_files:
        dapi_by_stem.setdefault(dapi.stem.lower(), []).append(dapi)
        idx = _last_numeric_token(dapi.stem)
        if idx is not None:
            dapi_by_idx.setdefault(idx, []).append(dapi)

    used_dapi: set[Path] = set()
    dapi_nuc_img_files: list[Path] = []
    dapi_output_stems: list[str] = []
    fallback_pc_nuc_img_files: list[Path] = []
    fallback_pc_output_stems: list[str] = []
    fallback_pc_count = 0

    for pc in img_files:
        candidates: list[Path] = []
        candidates.extend(dapi_by_stem.get(pc.stem.lower(), []))
        idx = _last_numeric_token(pc.stem)
        if idx is not None:
            candidates.extend(dapi_by_idx.get(idx, []))

        matched: Path | None = None
        for c in candidates:
            if c not in used_dapi:
                matched = c
                used_dapi.add(c)
                break

        if matched is None:
            fallback_pc_count += 1
            fallback_pc_nuc_img_files.append(pc)
            fallback_pc_output_stems.append(pc.stem)
        else:
            dapi_nuc_img_files.append(matched)
            dapi_output_stems.append(pc.stem)

    if fallback_pc_count > 0:
        print(
            f"[WARN] 有 {fallback_pc_count} 張 PC 找不到對應 DAPI，該些影像的細胞核仍使用 PC。"
        )

    if dapi_nuc_img_files:
        segment(
            DAPI_NUC_MODEL_PATH,
            dapi_nuc_img_files,
            seg_dir,
            "nuc",
            output_stems=dapi_output_stems,
            channels=(3, 3),
            model_input_size=nuc_model_input_size,
            cellprob_threshold=cellprob_threshold,
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
            apply_small_label_filter=False,
        )
        remap_nuc_segments_to_cyto(seg_dir, dapi_output_stems, dapi_output_stems)

    if fallback_pc_nuc_img_files:
        segment(
            PC_NUC_MODEL_PATH,
            fallback_pc_nuc_img_files,
            seg_dir,
            "nuc",
            output_stems=fallback_pc_output_stems,
            channels=(0, 0),
            model_input_size=nuc_model_input_size,
            cellprob_threshold=cellprob_threshold,
            min_area_ratio=min_area_ratio,
            min_area_floor=min_area_floor,
            apply_small_label_filter=False,
        )

    _filter_paired_segment_files(
        seg_dir,
        [image.stem for image in img_files],
        min_area_ratio=min_area_ratio,
        min_area_floor=min_area_floor,
    )


# ===============================
# Mask 轉換為 outlines
# ===============================
def mask2txt(npy_files: list[Path], output_dir: Path):
    """將 segmentation npy 檔轉換成 Cellpose outlines 文字檔。"""
    for i in trange(len(npy_files), desc=f"Saving {len(npy_files)} masks"):
        f = npy_files[i]
        data = np.load(f, allow_pickle=True).item()
        masks, flows = data["masks"], data["flows"]
        io.save_masks(
            None,
            masks,
            flows,
            f.stem,
            png=False,
            channels=[0, 0],
            save_txt=True,
            savedir=output_dir,
        )


def mask2txt_all(input_dir: str):
    """
    將細胞質與細胞核的 segmentation npy 檔批次轉成 outlines txt。
    """
    input_dir = Path(input_dir)
    seg_dir = output_dir(input_dir, "segment")
    mask2txt_out = output_dir(input_dir, "outline")
    cyto_files = [f for f in list_files(seg_dir, ".npy") if "_cyto_seg" in f.stem]
    nuc_files = [f for f in list_files(seg_dir, ".npy") if "_nuc_seg" in f.stem]
    mask2txt(cyto_files, mask2txt_out)
    mask2txt(nuc_files, mask2txt_out)


# ===============================
# Outlines 合併
# ===============================
def create_mask(outlines, shape):
    """依據 outlines 座標建立標籤 mask。"""
    mask = np.zeros(shape, dtype=np.int32)
    for i, line in enumerate(outlines, start=1):
        coords = list(map(int, line.split(",")))
        xs, ys = coords[::2], coords[1::2]
        rr, cc = polygon(ys, xs, shape)
        mask[rr, cc] = i
    return mask


from pathlib import Path
from typing import Optional, Tuple


def find_image_and_nuc_file(cyto_file: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """配對細胞質 outlines、細胞核 outlines，並尋找來源 PC 影像。"""
    # 由細胞質 outlines 檔名推得對應的細胞核 outlines 檔名。
    nuc_file = cyto_file.with_name(
        cyto_file.name.replace("_cyto_seg_cp_outlines.txt", "_nuc_seg_cp_outlines.txt")
    )
    if not nuc_file.exists():
        return None, None

    # 從 outlines 所在資料夾取得 dataset 名稱。
    dataset_name = cyto_file.parent.name  # e.g., 2025-07-10-B8-P6...
    index_key = cyto_file.stem.replace("_cyto_seg_cp_outlines", "")

    # 在 data/input/<dataset>/PC 中尋找對應影像。
    for ext in [".jpg", ".png", ".tif", ".tiff"]:
        img_candidate = Path("data/input") / dataset_name / "PC" / f"{index_key}{ext}"
        if img_candidate.exists():
            return nuc_file, img_candidate

    return None, None


def match_and_write(cyto_lines, nuc_lines, cyto_mask, nuc_mask, out_path: Path):
    """將細胞核與細胞質 outlines 配對後寫成 merged outlines。"""
    cyto_ids = np.unique(cyto_mask)[1:]
    nuc_ids = np.unique(nuc_mask)[1:]
    cyto_polygons = {}
    for i, line in enumerate(cyto_lines):
        coords = list(map(int, line.strip().split(",")))
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        poly = Polygon(points)
        if poly.is_valid and not poly.is_empty:
            cyto_polygons[i] = poly
    pairings, used_cyto = {}, set()
    for ni, line in enumerate(nuc_lines):
        coords = list(map(int, line.strip().split(",")))
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        if len(points) < 3:
            continue
        center = np.mean(points, axis=0)
        point = Point(center)
        matched = None
        for ci, poly in cyto_polygons.items():
            if ci in used_cyto:
                continue
            if poly.contains(point):
                matched = ci
                break
        if matched is not None:
            pairings[ni] = matched
            used_cyto.add(matched)
    with open(out_path, "w") as f:
        written_nuc, written_cyto = set(), set()
        for ni, ci in sorted(pairings.items()):
            f.write(nuc_lines[ni].strip() + "\n")
            f.write(cyto_lines[ci].strip() + "\n")
            written_nuc.add(ni)
            written_cyto.add(ci)
        for ni in sorted(set(range(len(nuc_lines))) - written_nuc):
            f.write(nuc_lines[ni].strip() + "\n")
            f.write("-1,-1\n")
        for ci in sorted(set(range(len(cyto_lines))) - written_cyto):
            f.write("-1,-1\n")
            f.write(cyto_lines[ci].strip() + "\n")


def combined(input_dir: str):
    """合併細胞質與細胞核 outlines，輸出成配對後的 merged outlines。"""
    input_dir = Path(input_dir)
    outline_dir = output_dir(input_dir, "outline")
    txt_files = [f for f in list_files(outline_dir, ".txt") if "_cyto_seg" in f.stem]
    if not txt_files:
        print(f"[WARN] 找不到細胞質 outlines 於 {outline_dir}")
        return

    for cyto_file in trange(len(txt_files), desc="Merging outlines"):
        cyto_path = txt_files[cyto_file]
        nuc_path, img_path = find_image_and_nuc_file(cyto_path)
        if not nuc_path or not img_path:
            print(f"[WARN] 缺少細胞核 outlines 或對應影像: {cyto_path.name}")
            continue

        img = Image.open(img_path)
        shape = img.size[::-1]
        cyto_lines = load_outlines(cyto_path)
        nuc_lines = load_outlines(nuc_path)
        cyto_mask = create_mask(cyto_lines, shape)
        nuc_mask = create_mask(nuc_lines, shape)
        out_path = cyto_path.with_name(
            cyto_path.stem.replace("_cyto_seg_cp_outlines", "_merged_cp_outlines.txt")
        )
        match_and_write(cyto_lines, nuc_lines, cyto_mask, nuc_mask, out_path)

    remove_temp_files(outline_dir)
