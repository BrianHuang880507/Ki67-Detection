"""Exp8：把形狀達標的細胞逐顆輸出成去背 PNG。

一個外觀特徵一個資料夾，一顆細胞一張圖，檔名與圖上文字都帶著
「細胞編號 ＋ 刺激條件 ＋ 該特徵的值」，方便生醫同仁直接翻圖、回原片查證。

和 fig06 的差別：fig06 是每個特徵挑 6 顆做版面，這裡是**達標的全部細胞**。
同一顆細胞若同時達標多個特徵，會在各自的資料夾各出現一次。

算繪沿用 `immunity.exp6.gallery` 的去背規則：白底、綠色 IDO、藍色細胞核、
紅色輪廓，並先扣掉該張影像的細胞外背景。所有資料夾共用同一個裁切視窗與
同一個綠色飽和值，跨特徵、跨條件都可以直接比較。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..exp6.dataset import FEATURE_PREFIX
from ..exp6.gallery import (
    GalleryError,
    ImageBundle,
    crop_cell,
    load_image_bundle,
    render_background_removed,
)
from .notable import FeatureThreshold


#: 圖上文字區的高度與四行的基準線位置（像素）。
HEADER_HEIGHT = 76
HEADER_LINES_Y = (17, 35, 53, 71)

#: 亮／不亮的子資料夾名稱，用英文確保路徑安全。
BRIGHT_FOLDER = "IDO_bright"
DIM_FOLDER = "IDO_dim"
HEADER_FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_SCALE = 0.42
HEADER_COLOR = (28, 28, 28)


def folder_name(feature: str) -> str:
    """特徵欄位名轉成資料夾名稱，只留英文以確保路徑安全。"""
    return feature.removeprefix(FEATURE_PREFIX)


def format_value(value: float) -> str:
    """特徵值的顯示格式，和 fig06 一致。"""
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.0f}"


def build_export_table(
    cells: pd.DataFrame,
    thresholds: list[FeatureThreshold],
    *,
    bright_min: float,
    target: str = "IDO_score_ff",
) -> pd.DataFrame:
    """列出每個特徵所有達標的細胞，一列一個 (細胞, 特徵) 組合。

    每個特徵資料夾再依 IDO 分成亮／不亮兩個子資料夾。切點沿用
    `brightness.control_thresholds` 的亮門檻（未刺激對照組的 P99），
    所以和 fig07／fig09 用的是同一個定義。

    這裡刻意只切成兩類而不是三類：`brightness.py` 的三分法會把中間灰帶排除，
    用在統計比較上沒問題，但這批匯出的用途是完整翻閱，不該有細胞憑空消失。
    """
    blocks: list[pd.DataFrame] = []
    for threshold in thresholds:
        qualifying = cells[threshold.qualifies(cells[threshold.feature])]
        if qualifying.empty:
            raise GalleryError(f"{threshold.feature_label} 沒有任何達標細胞")
        value = qualifying[threshold.feature].astype(float)
        is_bright = qualifying[target] > bright_min
        blocks.append(
            qualifying.assign(
                export_feature=threshold.feature,
                export_class=np.where(is_bright, BRIGHT_FOLDER, DIM_FOLDER),
                export_folder=[
                    f"{folder_name(threshold.feature)}/{BRIGHT_FOLDER if flag else DIM_FOLDER}"
                    for flag in is_bright
                ],
                export_label=threshold.feature_label,
                export_value=value,
                export_threshold=threshold.threshold,
                export_bright_min=bright_min,
                export_file=[
                    f"{key}_cell{int(label):03d}_{condition}_{format_value(item)}.png"
                    for key, label, condition, item in zip(
                        qualifying["image_key"],
                        qualifying["cell_label"],
                        qualifying["condition"],
                        value,
                    )
                ],
            )
        )
    return pd.concat(blocks, ignore_index=True)


def export_window_span(table: pd.DataFrame, *, quantile: float = 95.0, margin: float = 1.1) -> int:
    """所有資料夾共用的裁切視窗邊長。

    共用一個視窗，不同特徵、不同條件的圖才能直接互相比較大小。取 P95 之後
    約 1–2% 最長的細胞會略微出界，換取其餘的細胞不要縮得太小。
    """
    column = "cell__MaxFeretDiameter"
    if column not in table.columns:
        raise GalleryError(f"匯出表缺少 {column}")
    return int(max(96, round(float(np.percentile(table[column].dropna(), quantile)) * margin)))


def _subtract_background(bundle: ImageBundle) -> ImageBundle:
    """扣掉細胞外背景中位數，避免刺激孔的背景把暗細胞也染綠。"""
    background = bundle.ido[bundle.cell_mask == 0]
    if background.size == 0:
        return bundle
    return ImageBundle(
        phase=bundle.phase,
        ido=np.clip(bundle.ido - float(np.median(background)), 0.0, None),
        cell_mask=bundle.cell_mask,
        nucleus_mask=bundle.nucleus_mask,
    )


def estimate_vmax(
    data_root: Path, table: pd.DataFrame, span: int, *, sample_images: int = 60, seed: int = 0
) -> float:
    """抽樣估算共用的綠色飽和值，不必先把兩萬張全部裁完。"""
    keys = table["image_key"].drop_duplicates()
    rng = np.random.default_rng(seed)
    picked = keys if len(keys) <= sample_images else pd.Series(
        rng.choice(keys.to_numpy(), size=sample_images, replace=False)
    )
    pooled: list[np.ndarray] = []
    for image_key in picked:
        block = table[table["image_key"] == image_key]
        bundle = _subtract_background(load_image_bundle(data_root, block.iloc[0]))
        for label in block["cell_label"].drop_duplicates():
            crop = crop_cell(
                bundle, int(label), padding=1.0, tile_pixels=span, fixed_span=span
            )
            values = crop["ido"][crop["mask"] > 0.5]
            if values.size:
                pooled.append(values)
    if not pooled:
        raise GalleryError("抽樣不到任何細胞內像素，無法決定顯示範圍")
    return float(max(np.percentile(np.concatenate(pooled), 99.0), 1.0))


def _draw_header(image: np.ndarray, lines: tuple[str, ...]) -> np.ndarray:
    """在圖片上方加一條白色文字區，寫上細胞編號、條件、特徵值與 IDO 值。"""
    height, width = image.shape[:2]
    canvas = np.full((height + HEADER_HEIGHT, width, 3), 255, dtype=np.uint8)
    canvas[HEADER_HEIGHT:] = image
    for text, baseline in zip(lines, HEADER_LINES_Y):
        cv2.putText(
            canvas,
            text,
            (6, baseline),
            HEADER_FONT,
            HEADER_SCALE,
            HEADER_COLOR,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """以 imencode + tofile 寫檔，避開 cv2.imwrite 的路徑編碼問題。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise GalleryError(f"PNG 編碼失敗：{path}")
    encoded.tofile(str(path))


def export_cells(
    data_root: Path,
    table: pd.DataFrame,
    out_root: Path,
    *,
    span: int,
    vmax: float,
    progress: object | None = None,
) -> pd.DataFrame:
    """把匯出表裡的每一列寫成一張 PNG。

    以 `image_key` 分組，一張原始影像只解碼一次；同一顆細胞若達標多個特徵，
    裁切與算繪也只做一次，只是各自加上不同的文字後分別存檔。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    for folder in table["export_folder"].drop_duplicates():
        (out_root / str(folder)).mkdir(parents=True, exist_ok=True)

    written = 0
    records: list[dict[str, object]] = []
    groups = table.groupby("image_key", sort=True)
    for index, (image_key, block) in enumerate(groups, start=1):
        bundle = _subtract_background(load_image_bundle(data_root, block.iloc[0]))
        rendered: dict[int, np.ndarray] = {}
        for row in block.itertuples():
            label = int(row.cell_label)
            if label not in rendered:
                crop = crop_cell(
                    bundle, label, padding=1.0, tile_pixels=span, fixed_span=span
                )
                rgb = render_background_removed(
                    crop["ido"], crop["mask"], crop["nucleus"], vmax=vmax
                )
                rendered[label] = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            value = format_value(float(row.export_value))
            # 資料夾已經分了亮／不亮，圖上也寫出 IDO 值，一眼就能驗證分在哪一邊。
            canvas = _draw_header(
                rendered[label],
                (
                    f"{row.image_key} #{label}",
                    str(row.condition),
                    f"{folder_name(str(row.export_feature))} {value}",
                    f"IDO {float(row.IDO_score_ff):.2f}  {str(row.export_class)}",
                ),
            )
            path = out_root / str(row.export_folder) / str(row.export_file)
            _write_png(path, canvas)
            written += 1
            records.append(
                {
                    "export_folder": row.export_folder,
                    "export_class": row.export_class,
                    "export_file": row.export_file,
                    "image_key": row.image_key,
                    "cell_label": label,
                    "b_id": row.b_id,
                    "passage": row.passage,
                    "condition": row.condition,
                    "ifn_dose": row.ifn_dose,
                    "tnf_dose": row.tnf_dose,
                    "feature": row.export_feature,
                    "feature_label": row.export_label,
                    "value": float(row.export_value),
                    "threshold": float(row.export_threshold),
                    "IDO_score_ff": row.IDO_score_ff,
                }
            )
        if progress is not None and index % 100 == 0:
            progress(f"　　匯出進度 {index}/{len(groups)} 張影像、{written:,} 個檔案")
    return pd.DataFrame(records)
