"""建立 B4 p6 的影像配對、細胞特徵與 IDO response 資料集。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from ki67dtc.cell_anal import (
    _classify_outline_status,
    _geometry_from_measurements,
    _measure_roi_with_python,
    _parse_outline_pairs,
    _polygon_to_mask,
    _read_gray_float,
    _safe_divide,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATTERN = re.compile(
    r"^IFN-r\s*(?P<ifn>\d+(?:\.\d+)?)\s*ng"
    r"(?:_TNF-a\s*(?P<tnf>\d+(?:\.\d+)?)\s*ng)?"
    r"-(?P<channel>phase|DAPI|IDO)-100X-(?P<image_index>\d+)$",
    flags=re.IGNORECASE,
)
CHANNEL_FOLDERS = {"phase": "PC", "dapi": "DAPI", "ido": "IDO"}

MORPHOLOGY_BASE_FEATURES = (
    "Area",
    "Compactness",
    "Eccentricity",
    "Extent",
    "Sphericity",
    "Major Axis Length",
    "Feret Length",
    "Minor Axis Length",
    "Feret Width",
    "Maximum Radius",
    "Mean Radius",
    "Median Radius",
    "Aspect Ratio",
    "Perimeter/Area Ratio",
    "Perimeter",
    "Solidity",
)

GEOMETRY_KEYS = {
    "Area": "area",
    "Compactness": "compactness",
    "Eccentricity": "eccentricity",
    "Extent": "extent",
    "Sphericity": "sphericity",
    "Major Axis Length": "major_axis_length",
    "Feret Length": "feret_length",
    "Minor Axis Length": "minor_axis_length",
    "Feret Width": "feret_width",
    "Maximum Radius": "maximum_radius",
    "Mean Radius": "mean_radius",
    "Median Radius": "median_radius",
    "Aspect Ratio": "aspect_ratio",
    "Perimeter/Area Ratio": "perimeter_area_ratio",
    "Perimeter": "perimeter",
    "Solidity": "solidity",
}


def resolve_project_path(path: str | Path) -> Path:
    """將設定檔中的相對路徑解析為專案根目錄下的絕對路徑。"""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


def parse_image_filename(path: str | Path) -> dict[str, Any]:
    """解析 B4 p6 影像檔名中的刺激劑量、通道與視野編號。

    Args:
        path: 原始影像路徑或檔名。

    Returns:
        劑量、通道、視野編號與唯一配對 key。

    Raises:
        ValueError: 檔名不符合目前 B4 p6 命名規則時拋出。
    """
    candidate = Path(path)
    match = IMAGE_PATTERN.fullmatch(candidate.stem)
    if match is None:
        raise ValueError(f"無法解析影像檔名：{candidate.name}")

    ifn = float(match.group("ifn"))
    tnf = float(match.group("tnf") or 0.0)
    channel = match.group("channel").lower()
    image_index = int(match.group("image_index"))
    image_key = f"IFN{ifn:g}_TNF{tnf:g}_FOV{image_index:02d}"
    return {
        "ifn_dose": ifn,
        "tnf_dose": tnf,
        "channel": channel,
        "image_index": image_index,
        "image_key": image_key,
        "condition": f"IFN{ifn:g}_TNF{tnf:g}",
    }


def _image_size(path: Path) -> tuple[int, int]:
    """讀取影像寬高而不載入完整像素陣列。"""
    with Image.open(path) as image:
        return image.size


def build_manifest(
    input_dir: str | Path,
    expected_triplets: int | None = None,
    expected_conditions: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """建立 phase、DAPI、IDO 一對一配對的 image-level manifest。

    Args:
        input_dir: 包含 ``PC``、``DAPI`` 與 ``IDO`` 的資料集根目錄。
        expected_triplets: 預期配對數；``None`` 表示不檢查固定數量。
        expected_conditions: ``"IFN,TNF"`` 到預期張數的對照。

    Returns:
        每列一組完整 triplet 的資料表。

    Raises:
        FileNotFoundError: 缺少輸入資料夾時拋出。
        ValueError: 發現重複、缺少通道、尺寸不符或數量錯誤時拋出。
    """
    root = resolve_project_path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到輸入資料夾：{root}")

    paired: dict[str, dict[str, Any]] = {}
    for expected_channel, folder_name in CHANNEL_FOLDERS.items():
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到 {expected_channel} 資料夾：{folder}")
        for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff",
            }:
                continue
            parsed = parse_image_filename(path)
            channel = str(parsed["channel"])
            if channel != expected_channel:
                raise ValueError(
                    f"資料夾與檔名通道不一致：{path.name} 應為 {expected_channel}"
                )
            image_key = str(parsed["image_key"])
            row = paired.setdefault(image_key, dict(parsed))
            path_key = f"{channel}_path"
            if path_key in row:
                raise ValueError(f"重複的 {channel} 影像配對：{image_key}")
            row[path_key] = str(path.resolve())
            if channel == "phase":
                row["pc_stem"] = path.stem

    required_paths = {"phase_path", "dapi_path", "ido_path"}
    incomplete = [
        key for key, row in paired.items() if not required_paths.issubset(row.keys())
    ]
    if incomplete:
        raise ValueError(f"有 {len(incomplete)} 組影像缺少通道：{incomplete[:5]}")

    rows = []
    for image_key, row in paired.items():
        sizes = {_image_size(Path(row[key])) for key in required_paths}
        if len(sizes) != 1:
            raise ValueError(f"同一 triplet 影像尺寸不一致：{image_key} -> {sizes}")
        width, height = sizes.pop()
        rows.append({**row, "width": width, "height": height})

    manifest = pd.DataFrame(rows).sort_values(
        ["ifn_dose", "tnf_dose", "image_index"]
    )
    manifest = manifest.reset_index(drop=True)
    if expected_triplets is not None and len(manifest) != int(expected_triplets):
        raise ValueError(
            f"triplet 數量錯誤：預期 {expected_triplets}，實際 {len(manifest)}"
        )

    if expected_conditions:
        actual_counts = manifest.groupby(["ifn_dose", "tnf_dose"]).size()
        for raw_key, expected_count in expected_conditions.items():
            ifn_text, tnf_text = str(raw_key).split(",", maxsplit=1)
            key = (float(ifn_text), float(tnf_text))
            actual = int(actual_counts.get(key, 0))
            if actual != int(expected_count):
                raise ValueError(
                    f"條件 IFN={key[0]:g}, TNF={key[1]:g}："
                    f"預期 {expected_count} 張，實際 {actual} 張"
                )
    return manifest


def ensure_outlines(
    manifest: pd.DataFrame,
    input_dir: str | Path,
    segmentation_config: Mapping[str, Any],
) -> Path:
    """確認所有影像已有合併輪廓，缺少時執行 Cellpose 分割。

    Args:
        manifest: 完整影像配對表。
        input_dir: B4 p6 資料根目錄。
        segmentation_config: Cellpose 與重跑選項。

    Returns:
        合併輪廓的輸出資料夾。

    Raises:
        FileNotFoundError: 未啟用分割且找不到完整輪廓時拋出。
        RuntimeError: 執行後仍缺少輪廓時拋出。
    """
    root = resolve_project_path(input_dir)
    outline_dir = PROJECT_ROOT / "data" / "output" / "outline" / root.name
    expected = [outline_dir / f"{stem}_merged_cp_outlines.txt" for stem in manifest["pc_stem"]]
    force = bool(segmentation_config.get("force", False))
    missing = [path for path in expected if not path.exists()]
    if not force and not missing:
        print(f"[INFO] 重用既有合併輪廓：{outline_dir}")
        return outline_dir

    if not bool(segmentation_config.get("enabled", True)):
        raise FileNotFoundError(
            f"缺少 {len(missing)} 份輪廓且 segmentation.enabled=false：{outline_dir}"
        )

    from ki67dtc.img_prep import combined, mask2txt_all, segment_all

    print("[STEP] 使用 phase 分割細胞外框，使用 DAPI 分割細胞核")
    segment_all(
        root,
        nuc_source=str(segmentation_config.get("nuc_source", "dapi")),
        cellprob_threshold=float(segmentation_config.get("cellprob_threshold", 0.0)),
        min_area_ratio=float(segmentation_config.get("min_area_ratio", 0.15)),
        min_area_floor=int(segmentation_config.get("min_area_floor", 30)),
    )
    mask2txt_all(root)
    combined(root)

    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Cellpose 完成後仍缺少 {len(missing)} 份輪廓：{missing[:3]}")
    return outline_dir


def _geometry_columns(
    geometry: Mapping[str, float], region_suffix: str
) -> dict[str, float]:
    """將內部 geometry keys 轉成實驗資料表欄位。"""
    return {
        f"{feature}_{region_suffix}": float(geometry[key])
        for feature, key in GEOMETRY_KEYS.items()
    }


def extract_cell_level_features(
    manifest: pd.DataFrame,
    outline_dir: str | Path,
    max_nucleus_outside_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """從輪廓計算 morphology 與背景校正 IDO_score。

    Args:
        manifest: 完整影像配對表。
        outline_dir: ``*_merged_cp_outlines.txt`` 所在資料夾。
        max_nucleus_outside_fraction: 核遮罩落在細胞外的最大容許比例。

    Returns:
        Cell-level 特徵表與 image-level segmentation QC 表。

    Raises:
        ValueError: 沒有任何通過 QC 的細胞時拋出。
    """
    outline_root = resolve_project_path(outline_dir)
    cell_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for image in manifest.to_dict(orient="records"):
        pc_stem = str(image["pc_stem"])
        outline_path = outline_root / f"{pc_stem}_merged_cp_outlines.txt"
        phase_signal = _read_gray_float(Path(image["phase_path"]))
        ido_signal = _read_gray_float(Path(image["ido_path"]))
        if phase_signal.shape != ido_signal.shape:
            raise ValueError(f"phase/IDO 尺寸不一致：{image['image_key']}")
        shape = phase_signal.shape
        pairs = _parse_outline_pairs(outline_path, include_unpaired=True)

        objects: list[dict[str, Any]] = []
        cell_union = np.zeros(shape, dtype=bool)
        status_counts: dict[str, int] = {}
        for roi_id, nuc_xy, cell_xy in pairs:
            nuc_mask = (
                _polygon_to_mask(nuc_xy, shape)
                if nuc_xy is not None
                else np.zeros(shape, dtype=bool)
            )
            cell_mask = (
                _polygon_to_mask(cell_xy, shape)
                if cell_xy is not None
                else np.zeros(shape, dtype=bool)
            )
            if np.any(cell_mask):
                cell_union |= cell_mask
            status = _classify_outline_status(nuc_xy, cell_xy, shape)
            status_counts[status] = status_counts.get(status, 0) + 1
            objects.append(
                {
                    "roi_id": int(roi_id),
                    "status": status,
                    "nuc_mask": nuc_mask,
                    "cell_mask": cell_mask,
                }
            )

        background_pixels = ido_signal[~cell_union]
        background_median = (
            float(np.median(background_pixels))
            if background_pixels.size
            else np.nan
        )
        kept = 0
        excluded_outside = 0
        excluded_empty_cytoplasm = 0
        for obj in objects:
            if obj["status"] != "full_cell":
                continue
            nuc_mask = obj["nuc_mask"]
            cell_mask = obj["cell_mask"]
            nuc_area = float(np.count_nonzero(nuc_mask))
            outside_fraction = _safe_divide(
                float(np.count_nonzero(nuc_mask & ~cell_mask)), nuc_area
            )
            if (
                np.isfinite(outside_fraction)
                and outside_fraction > max_nucleus_outside_fraction
            ):
                excluded_outside += 1
                continue

            cytoplasm_mask = cell_mask & ~nuc_mask
            cytoplasm_area = float(np.count_nonzero(cytoplasm_mask))
            if cytoplasm_area <= 0:
                excluded_empty_cytoplasm += 1
                continue

            nuc_geometry = _geometry_from_measurements(
                _measure_roi_with_python(phase_signal, nuc_mask)
            )
            cell_geometry = _geometry_from_measurements(
                _measure_roi_with_python(phase_signal, cell_mask)
            )
            ido_mean = float(np.mean(ido_signal[cytoplasm_mask]))
            ido_score = float(ido_mean - background_median)
            row = {
                "Cell_ID": f"{pc_stem}_{obj['roi_id']}",
                "image_key": image["image_key"],
                "pc_stem": pc_stem,
                "condition": image["condition"],
                "IFN_dose": float(image["ifn_dose"]),
                "TNF_dose": float(image["tnf_dose"]),
                "image_index": int(image["image_index"]),
                "cell_status": obj["status"],
                "nucleus_outside_fraction": outside_fraction,
                "IDO_background_median": background_median,
                "IDO_cytoplasm_mean": ido_mean,
                "IDO_score": ido_score,
                **_geometry_columns(nuc_geometry, "nuc"),
                **_geometry_columns(cell_geometry, "cyto"),
                "Karyoplasmic Ratio": _safe_divide(nuc_geometry["area"], cytoplasm_area),
            }
            cell_rows.append(row)
            kept += 1

        qc_rows.append(
            {
                "image_key": image["image_key"],
                "condition": image["condition"],
                "IFN_dose": float(image["ifn_dose"]),
                "TNF_dose": float(image["tnf_dose"]),
                "outline_pairs": len(objects),
                "kept_full_cells": kept,
                "excluded_nucleus_outside": excluded_outside,
                "excluded_empty_cytoplasm": excluded_empty_cytoplasm,
                "IDO_background_median": background_median,
                **{f"status_{key}": value for key, value in status_counts.items()},
            }
        )

    cells = pd.DataFrame(cell_rows)
    qc = pd.DataFrame(qc_rows)
    if cells.empty:
        raise ValueError("沒有任何通過 segmentation QC 的完整細胞。")
    return cells, qc


def load_cell_level_from_cleaned(
    cleaned_csv: str | Path,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """從 ``main.py`` 的 cleaned CSV 建立 immunity cell-level 資料。

    Args:
        cleaned_csv: ``main.py`` 產生的 ``*_cleaned.csv``。
        manifest: 完整 phase／DAPI／IDO 配對表。

    Returns:
        僅保留完整細胞與 morphology predictors 的資料表，以及每張影像的
        segmentation／資料完整性摘要。

    Raises:
        FileNotFoundError: cleaned CSV 不存在時拋出。
        ValueError: 缺少 IDO target、morphology 欄位或影像對應時拋出。
    """
    source = resolve_project_path(cleaned_csv)
    if not source.is_file():
        raise FileNotFoundError(
            f"找不到 main.py 輸出：{source}。請先執行完整 main.py CLI。"
        )

    raw = pd.read_csv(source)
    required = {
        "Cell_ID",
        "Image",
        "cell_status",
        "IDO_BackgroundMean",
        "IDO_MeanIntensity",
        "IDO_MeanIntensity_BgSub",
        *morphology_feature_columns(),
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            "cleaned CSV 缺少 immunity 所需欄位；請確認 main.py 使用目前程式重跑："
            f"{missing}"
        )

    image_meta = manifest.set_index("pc_stem")[
        ["image_key", "condition", "ifn_dose", "tnf_dose", "image_index"]
    ]
    unknown_images = sorted(set(raw["Image"].astype(str)) - set(image_meta.index.astype(str)))
    if unknown_images:
        raise ValueError(f"cleaned CSV 有無法對應 manifest 的影像：{unknown_images[:5]}")

    full_cells = raw[raw["cell_status"].eq("full_cell")].copy()
    full_cells = full_cells.merge(
        image_meta,
        left_on="Image",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    selected = full_cells[
        [
            "Cell_ID",
            "image_key",
            "Image",
            "condition",
            "ifn_dose",
            "tnf_dose",
            "image_index",
            "cell_status",
            "IDO_BackgroundMean",
            "IDO_MeanIntensity",
            "IDO_MeanIntensity_BgSub",
            *morphology_feature_columns(),
        ]
    ].rename(
        columns={
            "Image": "pc_stem",
            "ifn_dose": "IFN_dose",
            "tnf_dose": "TNF_dose",
            "IDO_BackgroundMean": "IDO_background_median",
            "IDO_MeanIntensity": "IDO_cytoplasm_mean",
            "IDO_MeanIntensity_BgSub": "IDO_score",
        }
    )
    numeric_columns = [
        "IFN_dose",
        "TNF_dose",
        "IDO_background_median",
        "IDO_cytoplasm_mean",
        "IDO_score",
        *morphology_feature_columns(),
    ]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan)
    selected = selected.dropna(subset=["IDO_score"]).reset_index(drop=True)

    status_table = (
        raw.groupby(["Image", "cell_status"]).size().unstack(fill_value=0)
    )
    qc = manifest[["pc_stem", "image_key", "condition", "ifn_dose", "tnf_dose"]].copy()
    qc = qc.merge(status_table, left_on="pc_stem", right_index=True, how="left")
    status_columns = [column for column in status_table.columns]
    for column in status_columns:
        qc[column] = qc[column].fillna(0).astype(int)
        qc = qc.rename(columns={column: f"status_{column}"})
    kept_counts = selected.groupby("image_key").size().rename("kept_full_cells")
    qc = qc.merge(kept_counts, left_on="image_key", right_index=True, how="left")
    qc["kept_full_cells"] = qc["kept_full_cells"].fillna(0).astype(int)
    qc["source_cleaned_csv"] = str(source)
    return selected, qc


def morphology_feature_columns() -> list[str]:
    """回傳 morphology-only 模型允許使用的 cell-level 欄位。"""
    columns = [
        f"{feature}_{suffix}"
        for suffix in ("nuc", "cyto")
        for feature in MORPHOLOGY_BASE_FEATURES
    ]
    return [*columns, "Karyoplasmic Ratio"]


def aggregate_image_level(
    cells: pd.DataFrame,
    manifest: pd.DataFrame,
    min_cells_per_image: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """將 cell-level morphology 與 IDO_score 彙整成 image-level 資料。

    Args:
        cells: 通過 QC 的 cell-level 資料。
        manifest: 完整影像配對表。
        min_cells_per_image: 每張影像至少需保留的完整細胞數。

    Returns:
        Image-level 資料表及 morphology predictor 欄位清單。

    Raises:
        ValueError: 有影像的有效細胞數不足，或彙整結果不是一圖一列時拋出。
    """
    feature_columns = morphology_feature_columns()
    missing = [column for column in feature_columns if column not in cells.columns]
    if missing:
        raise ValueError(f"cell-level 資料缺少 morphology 欄位：{missing}")

    rows: list[dict[str, Any]] = []
    for image_key, group in cells.groupby("image_key", sort=False):
        if len(group) < min_cells_per_image:
            continue
        row: dict[str, Any] = {
            "image_key": image_key,
            "condition": group["condition"].iloc[0],
            "IFN_dose": float(group["IFN_dose"].iloc[0]),
            "TNF_dose": float(group["TNF_dose"].iloc[0]),
            "image_index": int(group["image_index"].iloc[0]),
            "cell_count": int(len(group)),
            "IDO_score": float(group["IDO_score"].median()),
            "IDO_score_IQR": float(
                group["IDO_score"].quantile(0.75)
                - group["IDO_score"].quantile(0.25)
            ),
        }
        for column in feature_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}__median"] = float(values.median())
            row[f"{column}__iqr"] = float(
                values.quantile(0.75) - values.quantile(0.25)
            )
        rows.append(row)

    images = pd.DataFrame(rows)
    expected_keys = set(manifest["image_key"].astype(str))
    actual_keys = set(images.get("image_key", pd.Series(dtype=str)).astype(str))
    missing_images = sorted(expected_keys - actual_keys)
    if missing_images:
        raise ValueError(
            f"有 {len(missing_images)} 張影像少於 {min_cells_per_image} 顆有效細胞："
            f"{missing_images[:5]}"
        )
    if images["image_key"].duplicated().any():
        raise ValueError("image-level dataset 出現重複 image_key。")

    predictors = [
        column
        for column in images.columns
        if column.endswith("__median") or column.endswith("__iqr")
    ]
    return images.sort_values(["IFN_dose", "TNF_dose", "image_index"]), predictors


def calculate_dose_response(
    images: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """計算三條現有劑量曲線及描述性 IDO dose-response AUC。"""
    curves = {
        "IFN_response_AUC": {
            "filter": images["TNF_dose"].eq(0),
            "dose_column": "IFN_dose",
            "expected_doses": (0.0, 25.0, 50.0, 100.0),
        },
        "TNF_response_AUC_IFN0": {
            "filter": images["IFN_dose"].eq(0),
            "dose_column": "TNF_dose",
            "expected_doses": (0.0, 25.0, 50.0),
        },
        "TNF_response_AUC_IFN25": {
            "filter": images["IFN_dose"].eq(25),
            "dose_column": "TNF_dose",
            "expected_doses": (0.0, 25.0, 50.0),
        },
    }
    summary_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    for curve_name, spec in curves.items():
        subset = images.loc[spec["filter"]].copy()
        grouped = subset.groupby(spec["dose_column"])["IDO_score"]
        curve_rows = []
        for dose in spec["expected_doses"]:
            values = grouped.get_group(dose) if dose in grouped.groups else pd.Series(dtype=float)
            if values.empty:
                raise ValueError(f"{curve_name} 缺少 dose={dose:g} 的影像。")
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row = {
                "curve": curve_name,
                "dose": float(dose),
                "n_images": int(len(values)),
                "IDO_score_mean": float(values.mean()),
                "IDO_score_median": float(values.median()),
                "IDO_score_std": std,
                "IDO_score_sem": (
                    float(std / np.sqrt(len(values))) if np.isfinite(std) else np.nan
                ),
            }
            summary_rows.append(row)
            curve_rows.append(row)

        x = np.asarray([row["dose"] for row in curve_rows], dtype=float)
        y = np.asarray([row["IDO_score_median"] for row in curve_rows], dtype=float)
        trapezoid = getattr(np, "trapezoid", np.trapz)
        raw_auc = float(trapezoid(y, x))
        auc_rows.append(
            {
                "curve": curve_name,
                "dose_min": float(x.min()),
                "dose_max": float(x.max()),
                "auc_raw": raw_auc,
                "auc_normalized_by_dose_range": float(raw_auc / (x.max() - x.min())),
                "target_name": "background-corrected IDO_score",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(auc_rows)


def calculate_oof_predicted_auc(
    predictions: pd.DataFrame,
    observed_auc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由 out-of-fold IDO predictions 推導 predicted dose-response AUC。

    Args:
        predictions: 每張影像的 OOF/LOCO observed 與 predicted IDO_score。
        observed_auc: 由實際 image-level IDO_score 計算的三條曲線 AUC。

    Returns:
        每個 validation／repeat／model 的 predicted AUC，以及跨 repeats 的
        median、IQR 與相對 observed AUC 的誤差摘要。
    """
    observed = observed_auc.set_index("curve")
    rows: list[dict[str, Any]] = []
    group_columns = ["validation", "repeat", "model"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        validation, repeat, model = keys
        predicted_images = group[
            ["IFN_dose", "TNF_dose", "predicted_IDO_score"]
        ].rename(columns={"predicted_IDO_score": "IDO_score"})
        _, predicted_auc = calculate_dose_response(predicted_images)
        for auc_row in predicted_auc.to_dict(orient="records"):
            curve = str(auc_row["curve"])
            observed_raw = float(observed.loc[curve, "auc_raw"])
            observed_normalized = float(
                observed.loc[curve, "auc_normalized_by_dose_range"]
            )
            predicted_raw = float(auc_row["auc_raw"])
            predicted_normalized = float(
                auc_row["auc_normalized_by_dose_range"]
            )
            rows.append(
                {
                    "validation": validation,
                    "repeat": int(repeat),
                    "model": model,
                    "curve": curve,
                    "observed_auc_raw": observed_raw,
                    "predicted_auc_raw": predicted_raw,
                    "auc_error_raw": predicted_raw - observed_raw,
                    "absolute_auc_error_raw": abs(predicted_raw - observed_raw),
                    "observed_auc_normalized": observed_normalized,
                    "predicted_auc_normalized": predicted_normalized,
                    "auc_error_normalized": predicted_normalized - observed_normalized,
                    "absolute_auc_error_normalized": abs(
                        predicted_normalized - observed_normalized
                    ),
                }
            )

    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in detail.groupby(["validation", "model", "curve"]):
        validation, model, curve = keys
        row: dict[str, Any] = {
            "validation": validation,
            "model": model,
            "curve": curve,
            "repeats": int(len(group)),
            "observed_auc_raw": float(group["observed_auc_raw"].iloc[0]),
            "observed_auc_normalized": float(
                group["observed_auc_normalized"].iloc[0]
            ),
        }
        for column in (
            "predicted_auc_raw",
            "absolute_auc_error_raw",
            "predicted_auc_normalized",
            "absolute_auc_error_normalized",
        ):
            values = group[column].astype(float)
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_iqr"] = float(
                values.quantile(0.75) - values.quantile(0.25)
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["validation", "model", "curve"]
    )
    return detail, summary
