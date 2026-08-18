"""比較不同 IFN／TNF 條件下的影像層級 morphology。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


CONTRAST_SPECS = (
    {
        "contrast_id": "IFN25_vs_0_at_TNF0",
        "contrast_label": "IFN 25-0 @ TNF 0",
        "family": "IFN_response_at_TNF0",
        "control": (0.0, 0.0),
        "treated": (25.0, 0.0),
    },
    {
        "contrast_id": "IFN50_vs_0_at_TNF0",
        "contrast_label": "IFN 50-0 @ TNF 0",
        "family": "IFN_response_at_TNF0",
        "control": (0.0, 0.0),
        "treated": (50.0, 0.0),
    },
    {
        "contrast_id": "IFN100_vs_0_at_TNF0",
        "contrast_label": "IFN 100-0 @ TNF 0",
        "family": "IFN_response_at_TNF0",
        "control": (0.0, 0.0),
        "treated": (100.0, 0.0),
    },
    {
        "contrast_id": "TNF25_vs_0_at_IFN0",
        "contrast_label": "TNF 25-0 @ IFN 0",
        "family": "TNF_response_at_IFN0",
        "control": (0.0, 0.0),
        "treated": (0.0, 25.0),
    },
    {
        "contrast_id": "TNF50_vs_0_at_IFN0",
        "contrast_label": "TNF 50-0 @ IFN 0",
        "family": "TNF_response_at_IFN0",
        "control": (0.0, 0.0),
        "treated": (0.0, 50.0),
    },
    {
        "contrast_id": "TNF25_vs_0_at_IFN25",
        "contrast_label": "TNF 25-0 @ IFN 25",
        "family": "TNF_response_at_IFN25",
        "control": (25.0, 0.0),
        "treated": (25.0, 25.0),
    },
    {
        "contrast_id": "TNF50_vs_0_at_IFN25",
        "contrast_label": "TNF 50-0 @ IFN 25",
        "family": "TNF_response_at_IFN25",
        "control": (25.0, 0.0),
        "treated": (25.0, 50.0),
    },
)


def _validate_inputs(
    images: pd.DataFrame, morphology_columns: Sequence[str]
) -> list[str]:
    """驗證條件比較需要的欄位並回傳 predictor 清單。"""
    features = list(dict.fromkeys(morphology_columns))
    if not features:
        raise ValueError("morphology_columns 不可為空。")
    required = {"condition", "IFN_dose", "TNF_dose", *features}
    missing = sorted(required - set(images.columns))
    if missing:
        raise ValueError(f"條件 morphology 分析缺少欄位：{missing}")
    return features


def summarize_condition_morphology(
    images: pd.DataFrame,
    morphology_columns: Sequence[str],
) -> pd.DataFrame:
    """將影像層級 morphology 彙整成條件層級長表。

    Args:
        images: 一張影像一列的資料，需包含條件、劑量與 morphology predictors。
        morphology_columns: 要比較的影像層級 morphology 欄位。

    Returns:
        每列為一個刺激條件與一個 predictor 的描述性摘要。

    Raises:
        ValueError: 缺少必要欄位、predictor 為空或沒有可用數值時拋出。

    Notes:
        此函式以影像為分析單位，不會用相同 ``image_index`` 將不同條件
        的影像配對。取得 Well ID 前，這些摘要只能作探索性比較。
    """
    features = _validate_inputs(images, morphology_columns)
    rows: list[dict[str, Any]] = []
    group_columns = ["IFN_dose", "TNF_dose", "condition"]
    for keys, group in images.groupby(group_columns, sort=True, dropna=False):
        ifn_dose, tnf_dose, condition = keys
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            values = values[np.isfinite(values)]
            if values.empty:
                continue
            rows.append(
                {
                    "condition": str(condition),
                    "IFN_dose": float(ifn_dose),
                    "TNF_dose": float(tnf_dose),
                    "feature": feature,
                    "n_images": int(len(values)),
                    "feature_median": float(values.median()),
                    "feature_q25": float(values.quantile(0.25)),
                    "feature_q75": float(values.quantile(0.75)),
                    "feature_iqr": float(
                        values.quantile(0.75) - values.quantile(0.25)
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("沒有可用的 condition-level morphology 數值。")
    return summary.sort_values(
        ["IFN_dose", "TNF_dose", "feature"]
    ).reset_index(drop=True)


def calculate_morphology_contrasts(
    images: pd.DataFrame,
    morphology_columns: Sequence[str],
) -> pd.DataFrame:
    """計算七組預先定義、條件層級且非配對的 morphology 差異。

    Args:
        images: 一張影像一列的 image-level 資料。
        morphology_columns: 要比較的 morphology predictors。

    Returns:
        每列為一個 contrast 與一個 predictor，包含原始差值及以全體影像
        IQR 標準化的差值。

    Raises:
        ValueError: 缺少必要條件、欄位或有效數值時拋出。

    Notes:
        ``delta_raw`` 定義為處理條件中位數減去對照條件中位數。
        ``delta_scaled_by_global_iqr`` 只用來跨特徵顯示相對變化，不是
        effect size，也不是 paired difference 或生物重複的推論統計。
    """
    features = _validate_inputs(images, morphology_columns)
    summary = summarize_condition_morphology(images, features)
    available_conditions = {
        (float(row.IFN_dose), float(row.TNF_dose))
        for row in summary[["IFN_dose", "TNF_dose"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    required_conditions = {
        condition
        for spec in CONTRAST_SPECS
        for condition in (spec["control"], spec["treated"])
    }
    missing_conditions = sorted(required_conditions - available_conditions)
    if missing_conditions:
        raise ValueError(f"缺少 morphology contrast 所需條件：{missing_conditions}")

    global_iqr: dict[str, float] = {}
    for feature in features:
        values = pd.to_numeric(images[feature], errors="coerce")
        values = values[np.isfinite(values)]
        global_iqr[feature] = (
            float(values.quantile(0.75) - values.quantile(0.25))
            if not values.empty
            else np.nan
        )

    indexed = summary.set_index(["IFN_dose", "TNF_dose", "feature"])
    rows: list[dict[str, Any]] = []
    for spec in CONTRAST_SPECS:
        control_ifn, control_tnf = spec["control"]
        treated_ifn, treated_tnf = spec["treated"]
        for feature in features:
            control = indexed.loc[(control_ifn, control_tnf, feature)]
            treated = indexed.loc[(treated_ifn, treated_tnf, feature)]
            control_median = float(control["feature_median"])
            treated_median = float(treated["feature_median"])
            delta = treated_median - control_median
            scale = global_iqr[feature]
            rows.append(
                {
                    "contrast_id": spec["contrast_id"],
                    "contrast_label": spec["contrast_label"],
                    "family": spec["family"],
                    "comparison_type": "condition_level_unpaired",
                    "control_condition": str(control["condition"]),
                    "treated_condition": str(treated["condition"]),
                    "control_IFN_dose": control_ifn,
                    "control_TNF_dose": control_tnf,
                    "treated_IFN_dose": treated_ifn,
                    "treated_TNF_dose": treated_tnf,
                    "feature": feature,
                    "n_control_images": int(control["n_images"]),
                    "n_treated_images": int(treated["n_images"]),
                    "control_median": control_median,
                    "treated_median": treated_median,
                    "delta_raw": delta,
                    "global_image_iqr": scale,
                    "delta_scaled_by_global_iqr": (
                        float(delta / scale)
                        if np.isfinite(scale) and scale > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _top_features(
    contrasts: pd.DataFrame, top_n: int
) -> list[str]:
    """依最大絕對標準化差異選取圖表用 predictors。"""
    ranked = (
        contrasts.assign(
            absolute_scaled_delta=contrasts[
                "delta_scaled_by_global_iqr"
            ].abs()
        )
        .groupby("feature")["absolute_scaled_delta"]
        .max()
        .sort_values(ascending=False)
    )
    return ranked.head(max(1, int(top_n))).index.tolist()


def plot_morphology_delta_heatmap(
    contrasts: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    """繪製七組 contrasts 的標準化 morphology 差異 heatmap。"""
    features = _top_features(contrasts, top_n)
    labels = [str(spec["contrast_label"]) for spec in CONTRAST_SPECS]
    matrix = (
        contrasts[contrasts["feature"].isin(features)]
        .pivot(
            index="contrast_label",
            columns="feature",
            values="delta_scaled_by_global_iqr",
        )
        .reindex(index=labels, columns=features)
    )
    values = matrix.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    limit = max(limit, 1e-9)

    fig, axis = plt.subplots(
        figsize=(max(12, 0.55 * len(features)), 5.8),
        constrained_layout=True,
    )
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(np.arange(len(features)))
    axis.set_xticklabels(features, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_title("Exploratory morphology differences by stimulation condition")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Median difference / global image IQR")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_top_morphology_boxplots(
    images: pd.DataFrame,
    contrasts: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 4,
) -> None:
    """繪製最大條件差異 predictors 的八條件 boxplots。"""
    features = _top_features(contrasts, top_n)
    conditions = (
        images[["IFN_dose", "TNF_dose", "condition"]]
        .drop_duplicates()
        .sort_values(["IFN_dose", "TNF_dose"])
    )
    n_columns = 2
    n_rows = int(np.ceil(len(features) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(13, max(4.5, 4.2 * n_rows)),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, feature in zip(axes.flat, features, strict=False):
        values = []
        labels = []
        colors = []
        for condition in conditions.itertuples(index=False):
            subset = images[
                images["IFN_dose"].eq(condition.IFN_dose)
                & images["TNF_dose"].eq(condition.TNF_dose)
            ]
            feature_values = pd.to_numeric(
                subset[feature], errors="coerce"
            ).dropna()
            values.append(feature_values.to_numpy(dtype=float))
            labels.append(f"I{condition.IFN_dose:g}\nT{condition.TNF_dose:g}")
            colors.append("#3C78A8" if condition.TNF_dose == 0 else "#E28E2C")
        boxes = axis.boxplot(values, tick_labels=labels, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_title(feature)
        axis.set_xlabel("IFN / TNF dose (ng)")
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    fig.suptitle("Top exploratory morphology features across 8 conditions")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_morphology_pca(
    images: pd.DataFrame,
    morphology_columns: Sequence[str],
    output_path: str | Path,
) -> pd.DataFrame:
    """以 morphology predictors 繪製影像層級 PCA，並回傳 scores。

    PCA 只用來顯示不同刺激條件的形態分布，不參與 ground truth 建立或
    regression 訓練。
    """
    features = _validate_inputs(images, morphology_columns)
    matrix = images[features].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(
        strategy="median", keep_empty_features=True
    ).fit_transform(matrix)
    standardized = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(standardized)
    result = images[
        ["image_key", "condition", "IFN_dose", "TNF_dose"]
    ].reset_index(drop=True)
    result = result.assign(PC1=scores[:, 0], PC2=scores[:, 1])

    fig, axis = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
    markers = ("o", "s", "^", "D", "P", "X", "v", "<")
    conditions = result[
        ["IFN_dose", "TNF_dose", "condition"]
    ].drop_duplicates().sort_values(["IFN_dose", "TNF_dose"])
    for marker, condition in zip(markers, conditions.itertuples(index=False)):
        subset = result[result["condition"].eq(condition.condition)]
        color = "#3C78A8" if condition.TNF_dose == 0 else "#E28E2C"
        axis.scatter(
            subset["PC1"],
            subset["PC2"],
            color=color,
            marker=marker,
            s=42,
            alpha=0.8,
            label=f"IFN {condition.IFN_dose:g} / TNF {condition.TNF_dose:g}",
        )
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axis.set_title("Morphology PCA by stimulation condition (exploratory)")
    axis.axhline(0, color="black", linewidth=0.6, alpha=0.35)
    axis.axvline(0, color="black", linewidth=0.6, alpha=0.35)
    axis.legend(fontsize=8, ncol=2)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return result
