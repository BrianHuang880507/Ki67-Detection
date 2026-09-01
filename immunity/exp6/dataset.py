"""Exp6 資料組裝：合併 Rui49 外觀特徵、flat-field 校正 IDO 與影像 manifest。

本模組只讀取既有 artifacts，不重跑 Cellpose、不重算特徵，也不覆寫任何
上游檔案。所有輸入皆為 Exp3／Exp4 已鎖定的產物。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CELL_KEY = ["image_key", "cell_label"]
FEATURE_PREFIX = "cell__"

#: 純位置欄位。已知 IDO 通道存在左右照明梯度（見
#: `immunity/outputs/exp4/FINDINGS_2026-08-19.md`），這些欄位描述細胞在
#: 視野中的座標而非外觀，因此不列入「外觀特徵」排名。
POSITION_FEATURES = (
    "cell__Center_X",
    "cell__Center_Y",
    "cell__BoundingBoxMinimum_X",
    "cell__BoundingBoxMinimum_Y",
    "cell__BoundingBoxMaximum_X",
    "cell__BoundingBoxMaximum_Y",
)

#: rolling-ball 背景扣除後恆為 0，Spearman 無定義。
CONSTANT_FEATURES = ("cell__MinIntensity",)

#: 角度為循環量（−π/2 與 +π/2 相同），線性相關係數不可解釋。
CIRCULAR_FEATURES = ("cell__Orientation",)

EXCLUDED_FEATURES = POSITION_FEATURES + CONSTANT_FEATURES + CIRCULAR_FEATURES

MANIFEST_COLUMNS = [
    "image_key",
    "b_id",
    "passage",
    "group_id",
    "condition",
    "condition_index",
    "ifn_dose",
    "tnf_dose",
    "pc_path",
    "ido_path",
]

#: 中文顯示名稱，供圖表使用。未列出的特徵直接去掉 `cell__` 前綴。
FEATURE_LABELS_ZH = {
    "cell__Area": "面積 Area",
    "cell__BoundingBoxArea": "外接矩形面積 BoundingBoxArea",
    "cell__Compactness": "緊緻度 Compactness",
    "cell__ConvexArea": "凸包面積 ConvexArea",
    "cell__Eccentricity": "離心率 Eccentricity",
    "cell__EquivalentDiameter": "等效直徑 EquivalentDiameter",
    "cell__Extent": "填充率 Extent",
    "cell__FormFactor": "圓形度 FormFactor",
    "cell__MajorAxisLength": "長軸長 MajorAxisLength",
    "cell__MaxFeretDiameter": "最大 Feret 徑",
    "cell__MaximumRadius": "最大內接半徑",
    "cell__MeanRadius": "平均內接半徑",
    "cell__MedianRadius": "中位內接半徑",
    "cell__MinFeretDiameter": "最小 Feret 徑",
    "cell__MinorAxisLength": "短軸長 MinorAxisLength",
    "cell__Perimeter": "周長 Perimeter",
    "cell__Solidity": "實心度 Solidity",
    "cell__IntegratedIntensity": "相位總亮度 IntegratedIntensity",
    "cell__MeanIntensity": "相位平均亮度 MeanIntensity",
    "cell__StdIntensity": "相位亮度標準差",
    "cell__MaxIntensity": "相位最大亮度",
    "cell__IntegratedIntensityEdge": "邊緣總亮度",
    "cell__MeanIntensityEdge": "邊緣平均亮度",
    "cell__StdIntensityEdge": "邊緣亮度標準差",
    "cell__MinIntensityEdge": "邊緣最小亮度",
    "cell__MaxIntensityEdge": "邊緣最大亮度",
    "cell__MassDisplacement": "質心位移 MassDisplacement",
    "cell__MADIntensity": "亮度絕對中位差",
    "cell__AngularSecondMoment": "紋理 AngularSecondMoment",
    "cell__Contrast": "紋理 Contrast",
    "cell__Correlation": "紋理 Correlation",
    "cell__Variance": "紋理 Variance",
    "cell__InverseDifferenceMoment": "紋理 InverseDifferenceMoment",
    "cell__SumAverage": "紋理 SumAverage",
    "cell__SumVariance": "紋理 SumVariance",
    "cell__SumEntropy": "紋理 SumEntropy",
    "cell__Entropy": "紋理 Entropy",
    "cell__DifferenceVariance": "紋理 DifferenceVariance",
    "cell__DifferenceEntropy": "紋理 DifferenceEntropy",
    "cell__InfoMeas1": "紋理 InfoMeas1",
    "cell__InfoMeas2": "紋理 InfoMeas2",
}


class DatasetError(RuntimeError):
    """表示 Exp6 輸入資料不符預期。"""


@dataclass(frozen=True)
class InputPaths:
    """Exp6 的三個唯讀輸入。"""

    features: Path
    ido: Path
    manifest: Path

    @classmethod
    def under(cls, data_root: Path) -> "InputPaths":
        """以資料根目錄組出預設輸入路徑。"""
        root = Path(data_root)
        return cls(
            features=root / "immunity/outputs/exp4/cell_level_rui49.csv",
            ido=root
            / "immunity/outputs/exp3/flatfield_2026-08-19/ido_flatfield.csv",
            manifest=root
            / "immunity/outputs/exp3/current_inputs_2026-08-19/data_manifest.csv",
        )

    def missing(self) -> list[Path]:
        """回傳不存在的輸入路徑。"""
        return [path for path in (self.features, self.ido, self.manifest) if not path.exists()]


def feature_columns(frame: pd.DataFrame, *, drop_excluded: bool = True) -> list[str]:
    """取出特徵欄位名稱，預設排除位置、常數與循環量欄位。"""
    columns = [name for name in frame.columns if name.startswith(FEATURE_PREFIX)]
    if drop_excluded:
        columns = [name for name in columns if name not in EXCLUDED_FEATURES]
    return columns


def feature_label(name: str) -> str:
    """回傳特徵的中文顯示名稱。"""
    return FEATURE_LABELS_ZH.get(name, name.removeprefix(FEATURE_PREFIX))


def _collapse_duplicate_cells(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """把同一 (image_key, cell_label) 的多列以中位數收斂成一列。

    Rui49 特徵表與 flat-field IDO 表都是以 (cell_label, nucleus_label) 配對展開，
    一顆細胞若配到多個細胞核就會重複出現。Exp5 以中位數收斂，本模組沿用同一規則。
    """
    grouped = frame.groupby(CELL_KEY, as_index=False)[value_columns].median()
    return grouped


def load_cell_table(paths: InputPaths) -> pd.DataFrame:
    """載入並合併 cell-level 特徵、IDO target 與影像條件。

    Returns:
        每列一顆通過 QC 的細胞，欄位含 `cell__*` 外觀特徵、`IDO_score_ff`、
        `IDO_score_scalar`，以及 donor／passage／condition／劑量。
    """
    missing = paths.missing()
    if missing:
        listed = "\n".join(f"  - {path}" for path in missing)
        raise DatasetError(f"缺少 Exp6 輸入檔案：\n{listed}")

    features = pd.read_csv(paths.features)
    ido = pd.read_csv(paths.ido)
    manifest = pd.read_csv(paths.manifest)

    missing_manifest = [name for name in MANIFEST_COLUMNS if name not in manifest.columns]
    if missing_manifest:
        raise DatasetError(f"manifest 缺少欄位：{missing_manifest}")

    all_features = [name for name in features.columns if name.startswith(FEATURE_PREFIX)]
    if not all_features:
        raise DatasetError("特徵表沒有任何 cell__ 欄位")

    ido_values = [name for name in ("IDO_score_ff", "IDO_score_scalar") if name in ido.columns]
    if "IDO_score_ff" not in ido_values:
        raise DatasetError("IDO 表缺少 IDO_score_ff 欄位")

    features = _collapse_duplicate_cells(features, all_features)
    ido = _collapse_duplicate_cells(ido, ido_values)

    merged = features.merge(ido, on=CELL_KEY, how="inner", validate="one_to_one")
    merged = merged.merge(
        manifest[MANIFEST_COLUMNS], on="image_key", how="inner", validate="many_to_one"
    )
    if merged.empty:
        raise DatasetError("合併後沒有任何細胞，請確認三個輸入是否來自同一批次")

    merged["ifn_dose"] = merged["ifn_dose"].astype(float)
    merged["tnf_dose"] = merged["tnf_dose"].astype(float)
    return merged


def build_fov_table(cells: pd.DataFrame) -> pd.DataFrame:
    """把 cell-level 表彙整成一張影像一列。

    劑量是加在整張影像（整個孔）上，同一影像內的細胞不是獨立樣本；以影像
    中位數為分析單位可避免把 23,012 顆細胞當成 23,012 個獨立觀測。
    """
    value_columns = [name for name in cells.columns if name.startswith(FEATURE_PREFIX)]
    value_columns += [name for name in ("IDO_score_ff", "IDO_score_scalar") if name in cells.columns]
    keys = [
        "image_key",
        "b_id",
        "passage",
        "group_id",
        "condition",
        "condition_index",
        "ifn_dose",
        "tnf_dose",
    ]
    fov = cells.groupby(keys, as_index=False)[value_columns].median()
    counts = cells.groupby("image_key").size().rename("n_cells")
    fov = fov.merge(counts, on="image_key", how="left")
    return fov


def dose_slices(fov: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """回傳兩條可解釋的劑量軸。

    本批次不是完整 factorial design，只有三條一維劑量曲線可用：

    - `ifn`：TNF-α = 0 時，IFN-γ 0／25／50／100。
    - `tnf`：IFN-γ 固定在 0 或 25 時，TNF-α 0／25／50（兩個區塊）。

    直接對全部 693 張影像同時算 IFN 與 TNF 相關係數會把兩個因子互相混淆，
    因此所有排名都在切片內計算。
    """
    ifn_axis = fov[fov["tnf_dose"] == 0].copy()
    tnf_axis = fov[fov["ifn_dose"].isin([0.0, 25.0])].copy()
    return {"ifn": ifn_axis, "tnf": tnf_axis}


def rank_within(frame: pd.DataFrame, column: str, by: list[str]) -> pd.Series:
    """在分組內做百分位排名，回傳 0–1 之間的值。"""
    return frame.groupby(by)[column].rank(pct=True, method="average")


def describe_dataset(cells: pd.DataFrame, fov: pd.DataFrame) -> dict[str, object]:
    """回傳資料規模摘要，寫進 run metadata 與報告。"""
    return {
        "n_cells": int(len(cells)),
        "n_images": int(fov["image_key"].nunique()),
        "n_groups": int(fov["group_id"].nunique()),
        "donors": sorted(fov["b_id"].unique().tolist()),
        "passages": sorted(int(value) for value in fov["passage"].unique()),
        "conditions": int(fov["condition"].nunique()),
        "n_appearance_features": len(feature_columns(cells)),
        "cells_per_image_median": float(np.median(fov["n_cells"])),
        "cells_per_image_min": int(fov["n_cells"].min()),
        "cells_per_image_max": int(fov["n_cells"].max()),
    }
