"""Exp8 形狀特徵定義與劑量關聯性。

只保留 17 個幾何特徵。Exp6 把 41 個外觀特徵混在一起排名時，32 個紋理／
亮度特徵會把形狀擠掉——`bright_dim_features_matched.csv` 裡最好的形狀特徵
只排到第 14 名。生醫所同仁問的是「形狀」，因此本模組全程排除紋理與亮度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

from ..exp6.dataset import FEATURE_PREFIX, feature_label


#: 17 個幾何特徵。其餘 32 個（GLCM 紋理、相位差強度）不屬於「形狀」。
SHAPE_FEATURE_NAMES = (
    "Area",
    "BoundingBoxArea",
    "Compactness",
    "ConvexArea",
    "Eccentricity",
    "EquivalentDiameter",
    "Extent",
    "FormFactor",
    "MajorAxisLength",
    "MaxFeretDiameter",
    "MaximumRadius",
    "MeanRadius",
    "MedianRadius",
    "MinFeretDiameter",
    "MinorAxisLength",
    "Perimeter",
    "Solidity",
)

SHAPE_COLUMNS = tuple(f"{FEATURE_PREFIX}{name}" for name in SHAPE_FEATURE_NAMES)

#: 細胞密度。劑量會改變細胞數，細胞數又會改變形狀，因此每個相關係數都要
#: 附上控制密度後的版本。
DENSITY_COLUMN = "n_cells"


class ShapeError(RuntimeError):
    """表示形狀分析的輸入不符預期。"""


def shape_columns(frame: pd.DataFrame) -> list[str]:
    """取出資料表中實際存在的形狀特徵欄位。"""
    present = [name for name in SHAPE_COLUMNS if name in frame.columns]
    if not present:
        raise ShapeError("資料表沒有任何形狀特徵欄位")
    return present


def partial_spearman(
    frame: pd.DataFrame, x: str, y: str, covariates: tuple[str, ...] = ()
) -> tuple[float, float, int]:
    """控制連續共變量後的 Spearman 相關係數。

    先把 x、y 與共變量各自轉成 rank，再把 x、y 對共變量做線性殘差化，最後
    求殘差的 Pearson r。共變量為空時退化成一般 Spearman。

    Returns:
        `(rho, p, n)`。
    """
    columns = [x, y, *covariates]
    data = frame[columns].dropna()
    if len(data) < len(columns) + 3:
        return float("nan"), float("nan"), len(data)
    if data[x].nunique() < 2 or data[y].nunique() < 2:
        return float("nan"), float("nan"), len(data)

    if not covariates:
        rho, p_value = stats.spearmanr(data[x], data[y])
        return float(rho), float(p_value), len(data)

    ranked = np.column_stack([rankdata(data[name]) for name in columns])
    ranked = ranked - ranked.mean(axis=0)
    design = np.column_stack([ranked[:, 2:], np.ones(len(ranked))])

    def _residual(values: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ beta

    x_residual = _residual(ranked[:, 0])
    y_residual = _residual(ranked[:, 1])
    denominator = np.sqrt(np.dot(x_residual, x_residual) * np.dot(y_residual, y_residual))
    if denominator <= 0:
        return float("nan"), float("nan"), len(data)
    rho = float(np.dot(x_residual, y_residual) / denominator)
    rho = max(-1.0, min(1.0, rho))

    degrees = len(data) - len(covariates) - 2
    if degrees <= 0 or abs(rho) >= 1.0:
        return rho, 0.0 if abs(rho) >= 1.0 else float("nan"), len(data)
    t_stat = rho * np.sqrt(degrees / (1.0 - rho * rho))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), degrees))
    return rho, p_value, len(data)


def correlate_shape_with_dose(
    fov: pd.DataFrame, dose_column: str, covariates: tuple[str, ...]
) -> pd.DataFrame:
    """17 個形狀特徵對某條劑量軸的關聯性，含密度校正前後。

    Args:
        fov: 影像層級資料表。
        dose_column: `ifn_dose` 或 `tnf_dose`。
        covariates: 校正版本要控制的欄位；TNF 軸需同時控制 `ifn_dose`。
    """
    rows: list[dict[str, object]] = []
    for name in shape_columns(fov):
        raw_rho, raw_p, n = partial_spearman(fov, dose_column, name)
        adjusted_rho, adjusted_p, _ = partial_spearman(fov, dose_column, name, covariates)
        retained = (
            adjusted_rho / raw_rho if raw_rho not in (0.0,) and np.isfinite(raw_rho) else float("nan")
        )
        rows.append(
            {
                "feature": name,
                "feature_label": feature_label(name),
                "spearman_rho": raw_rho,
                "abs_rho": abs(raw_rho) if np.isfinite(raw_rho) else float("nan"),
                "p_value": raw_p,
                "rho_density_adjusted": adjusted_rho,
                "p_value_density_adjusted": adjusted_p,
                "retained_fraction": retained,
                "n_images": n,
                "covariates": "、".join(covariates) if covariates else "無",
            }
        )
    table = pd.DataFrame(rows).sort_values("abs_rho", ascending=False, kind="mergesort")
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def shape_dose_response(fov: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """三條劑量曲線上，每個形狀特徵的影像層級摘要。"""
    curves = {
        "IFN-γ（TNF-α = 0）": (fov["tnf_dose"] == 0, "ifn_dose"),
        "TNF-α（IFN-γ = 0）": (fov["ifn_dose"] == 0, "tnf_dose"),
        "TNF-α（IFN-γ = 25）": (fov["ifn_dose"] == 25, "tnf_dose"),
    }
    rows: list[dict[str, object]] = []
    for curve_name, (mask, dose_column) in curves.items():
        subset = fov[mask]
        for dose, block in subset.groupby(dose_column):
            for name in features:
                values = block[name].dropna().to_numpy()
                if values.size == 0:
                    continue
                rows.append(
                    {
                        "curve": curve_name,
                        "dose_ng_ml": float(dose),
                        "feature": name,
                        "feature_label": feature_label(name),
                        "n_images": int(len(block)),
                        "median": float(np.median(values)),
                        "q1": float(np.percentile(values, 25)),
                        "q3": float(np.percentile(values, 75)),
                    }
                )
    return pd.DataFrame(rows)
