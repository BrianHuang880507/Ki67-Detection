"""Exp4 正式 CV 完成後的唯讀診斷分析。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr


EXPECTED_CELL_COUNT = 19_648
EXPECTED_FOV_COUNT = 693
EXPECTED_ARM_IDS = (
    "geometry_24",
    "rui_48",
    "rui_filtered",
    "rui_48_plus_nucleus",
)
EXPECTED_MODEL_IDS = ("SVR_L", "SVR", "LASSO", "RFR", "GBR", "MLPR")
EXPECTED_GROUP_IDS = (
    "B4_P5",
    "B4_P6",
    "B4_P7",
    "B7_P5",
    "B7_P6",
    "B7_P7",
    "B8_P5",
    "B8_P6",
    "B8_P7",
)
ORIENTATION_RANGE_TOLERANCE = 1e-12
SHRINKAGE_GROUP_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "group_id",
    "n_cells",
    "n_fovs",
    "observed_target",
    "mean_prediction",
)
SHRINKAGE_REPORT_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "n_cells",
    "n_fovs",
    "n_groups",
    "slope",
    "intercept",
    "regression_r_squared",
    "regression_r",
    "regression_p_value",
    "slope_std_error",
    "observed_min",
    "observed_max",
    "observed_range",
    "predicted_min",
    "predicted_max",
    "predicted_range",
    "predicted_to_observed_range_ratio",
)
ORIENTATION_BIN_COLUMNS = (
    "bin_index",
    "left_edge_radians",
    "right_edge_radians",
    "midpoint_radians",
    "left_edge_degrees",
    "right_edge_degrees",
    "midpoint_degrees",
    "n_cells",
    "n_fovs",
    "n_groups",
    "orientation_median_radians",
    "orientation_median_degrees",
    "target_median",
    "target_mean",
    "target_p25",
    "target_p75",
)
ORIENTATION_SUMMARY_COLUMNS = (
    "n_cells",
    "n_fovs",
    "n_groups",
    "n_bins",
    "cell_spearman_r",
    "cell_spearman_p_value",
    "group_spearman_r",
    "group_spearman_p_value",
    "bin_target_median_min",
    "bin_target_median_max",
    "bin_target_median_range",
    "bin_target_mean_min",
    "bin_target_mean_max",
    "bin_target_mean_range",
    "every_bin_covers_all_9_groups",
    "interpretation_scope",
)


class PostCvAnalysisError(ValueError):
    """表示 post-CV 輸入母體、schema 或統計結果不符合契約。"""


@dataclass(frozen=True)
class ShrinkageAnalysisResult:
    """保存 configuration-level OLS 與其 9-group 稽核明細。

    Attributes:
        report: 每個 arm × model 一列的 OLS 與 observed/predicted ranges。
        group_summary: 每個 configuration × group 一列的無權重 group summary。
    """

    report: pd.DataFrame
    group_summary: pd.DataFrame


@dataclass(frozen=True)
class OrientationMarginalResult:
    """保存固定角度 bins 與不帶因果解讀的邊際關聯摘要。

    Attributes:
        bins: 12 個固定 15° bins 的 cell/FOV/group counts 與 target summaries。
        summary: cell/group Spearman、bin target ranges 與 coverage 狀態。
    """

    bins: pd.DataFrame
    summary: pd.DataFrame


def analyze_shrinkage(oof_predictions: pd.DataFrame) -> ShrinkageAnalysisResult:
    """以 pooled OOF 的 group means 估計預測收縮程度。

    每顆 retained cell 在每個 configuration 僅能出現一次。函式先將預測
    聚合成 9 個不加權 group means，再以 ``mean_prediction`` 對
    ``observed_target`` 執行普通最小平方法線性迴歸。

    Args:
        oof_predictions: 正式 CV 的 OOF prediction table。

    Returns:
        24-row OLS 報告及 216-row group-level 稽核明細。

    Raises:
        TypeError: 輸入不是 pandas DataFrame 時拋出。
        PostCvAnalysisError: schema、正式母體或統計結果不符合契約時拋出。
    """
    if not isinstance(oof_predictions, pd.DataFrame):
        raise TypeError("oof_predictions 必須是 pandas DataFrame")
    if oof_predictions.columns.has_duplicates:
        raise PostCvAnalysisError("oof_predictions column labels 必須唯一")
    required = {
        "configuration_id",
        "arm",
        "model",
        "image_key",
        "cell_label",
        "group_id",
        "observed_ido_score",
        "predicted_ido_score",
    }
    missing = sorted(required - set(oof_predictions.columns))
    if missing:
        raise PostCvAnalysisError(f"oof_predictions 缺少欄位：{missing}")
    try:
        numeric_values = oof_predictions.loc[
            :, ["observed_ido_score", "predicted_ido_score"]
        ].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PostCvAnalysisError(
            "OOF observed/predicted values 必須為 numeric 且 finite"
        ) from error
    if not np.isfinite(numeric_values).all():
        raise PostCvAnalysisError(
            "OOF observed/predicted values 必須為 numeric 且 finite"
        )
    identity = oof_predictions.loc[
        :, ["configuration_id", "image_key", "cell_label"]
    ]
    if identity.isna().any().any() or identity.duplicated().any():
        raise PostCvAnalysisError(
            "OOF configuration × cell identity 必須唯一且不可為空"
        )
    expected_rows = (
        EXPECTED_CELL_COUNT * len(EXPECTED_ARM_IDS) * len(EXPECTED_MODEL_IDS)
    )
    if len(oof_predictions) != expected_rows:
        raise PostCvAnalysisError(
            f"OOF row count expected {expected_rows}, actual {len(oof_predictions)}"
        )
    categorical = oof_predictions.loc[
        :, ["configuration_id", "arm", "model", "group_id", "image_key"]
    ]
    if categorical.isna().any().any():
        raise PostCvAnalysisError("OOF categorical identity 不可為空")
    raw_categorical = categorical.astype(str)
    normalized = raw_categorical.apply(lambda series: series.str.strip())
    if normalized.eq("").any().any():
        raise PostCvAnalysisError("OOF categorical identity 不可為空")
    if not raw_categorical.eq(normalized).all().all():
        raise PostCvAnalysisError(
            "OOF canonical identifiers 不可含前後空白"
        )
    derived_configuration = normalized["arm"] + "__" + normalized["model"]
    if not normalized["configuration_id"].eq(derived_configuration).all():
        raise PostCvAnalysisError(
            "configuration_id 必須與每列 arm × model 一致"
        )
    if set(normalized["arm"]) != set(EXPECTED_ARM_IDS):
        raise PostCvAnalysisError("OOF 必須恰含正式 4-arm roster")
    if set(normalized["model"]) != set(EXPECTED_MODEL_IDS):
        raise PostCvAnalysisError("OOF 必須恰含正式 6-model roster")
    expected_configurations = {
        f"{arm}__{model}"
        for arm in EXPECTED_ARM_IDS
        for model in EXPECTED_MODEL_IDS
    }
    if set(normalized["configuration_id"]) != expected_configurations:
        raise PostCvAnalysisError("OOF 必須恰含 4 arms × 6 models")
    try:
        cell_labels = oof_predictions["cell_label"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PostCvAnalysisError("cell_label 必須為正整數") from error
    if (
        not np.isfinite(cell_labels).all()
        or np.any(cell_labels <= 0.0)
        or not np.equal(cell_labels, np.floor(cell_labels)).all()
    ):
        raise PostCvAnalysisError("cell_label 必須為正整數")
    cell_occurrences = oof_predictions.groupby(
        ["image_key", "cell_label"], sort=False, dropna=False
    )["configuration_id"].nunique()
    expected_configuration_count = len(expected_configurations)
    if len(cell_occurrences) != EXPECTED_CELL_COUNT or not cell_occurrences.eq(
        expected_configuration_count
    ).all():
        raise PostCvAnalysisError(
            "OOF cell population 必須是同一組 19,648 cells，且每個 cell "
            "恰有 24 個 configuration predictions"
        )
    if int(oof_predictions["image_key"].nunique()) != EXPECTED_FOV_COUNT:
        raise PostCvAnalysisError("OOF cell population 必須恰含 693 FOV")
    cell_group_mapping = oof_predictions.loc[
        :, ["image_key", "cell_label", "group_id"]
    ].drop_duplicates()
    if cell_group_mapping.duplicated(["image_key", "cell_label"]).any():
        raise PostCvAnalysisError(
            "OOF cell population 的 group_id 必須跨 configurations 恆定"
        )
    expected_groups = set(EXPECTED_GROUP_IDS)
    for configuration_id, configuration in oof_predictions.groupby(
        "configuration_id", sort=False
    ):
        if set(configuration["group_id"].astype(str)) != expected_groups:
            raise PostCvAnalysisError(
                f"{configuration_id} 必須完整覆蓋正式 9 groups"
            )
    target_counts = oof_predictions.groupby("group_id", sort=False)[
        "observed_ido_score"
    ].nunique(dropna=False)
    if not target_counts.eq(1).all():
        raise PostCvAnalysisError(
            "observed target must be constant within each group"
        )

    group_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    for arm in EXPECTED_ARM_IDS:
        for model in EXPECTED_MODEL_IDS:
            configuration_id = f"{arm}__{model}"
            configuration = oof_predictions.loc[
                oof_predictions["configuration_id"].eq(configuration_id)
            ]
            for group_id in EXPECTED_GROUP_IDS:
                group = configuration.loc[configuration["group_id"].eq(group_id)]
                observed = group["observed_ido_score"].to_numpy(dtype=np.float64)
                predicted = group["predicted_ido_score"].to_numpy(dtype=np.float64)
                group_rows.append(
                    {
                        "configuration_id": configuration_id,
                        "arm": arm,
                        "model": model,
                        "group_id": group_id,
                        "n_cells": int(len(group)),
                        "n_fovs": int(group["image_key"].nunique()),
                        "observed_target": float(observed[0]),
                        "mean_prediction": float(np.mean(predicted)),
                    }
                )

            current_groups = group_rows[-len(EXPECTED_GROUP_IDS) :]
            observed_targets = np.asarray(
                [row["observed_target"] for row in current_groups],
                dtype=np.float64,
            )
            mean_predictions = np.asarray(
                [row["mean_prediction"] for row in current_groups],
                dtype=np.float64,
            )
            observed_min = float(np.min(observed_targets))
            observed_max = float(np.max(observed_targets))
            observed_range = observed_max - observed_min
            if observed_range <= 0.0:
                raise PostCvAnalysisError(
                    f"{configuration_id} observed range 必須大於 0"
                )
            regression = linregress(observed_targets, mean_predictions)
            predicted_min = float(np.min(mean_predictions))
            predicted_max = float(np.max(mean_predictions))
            predicted_range = predicted_max - predicted_min
            report_rows.append(
                {
                    "configuration_id": configuration_id,
                    "arm": arm,
                    "model": model,
                    "n_cells": int(len(configuration)),
                    "n_fovs": int(configuration["image_key"].nunique()),
                    "n_groups": int(configuration["group_id"].nunique()),
                    "slope": float(regression.slope),
                    "intercept": float(regression.intercept),
                    "regression_r_squared": float(regression.rvalue**2),
                    "regression_r": float(regression.rvalue),
                    "regression_p_value": float(regression.pvalue),
                    "slope_std_error": float(regression.stderr),
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "observed_range": observed_range,
                    "predicted_min": predicted_min,
                    "predicted_max": predicted_max,
                    "predicted_range": predicted_range,
                    "predicted_to_observed_range_ratio": (
                        predicted_range / observed_range
                    ),
                }
            )

    return ShrinkageAnalysisResult(
        report=pd.DataFrame(report_rows, columns=SHRINKAGE_REPORT_COLUMNS),
        group_summary=pd.DataFrame(group_rows, columns=SHRINKAGE_GROUP_COLUMNS),
    )


def analyze_orientation_marginal(
    retained_cells: pd.DataFrame,
) -> OrientationMarginalResult:
    """描述正式 retained population 的 Orientation 邊際關聯。

    Orientation 使用固定 ``[-pi/2, pi/2]`` 上 12 個等寬 15° bins；內部
    boundary 歸入右側 bin，``pi/2`` 則明確保留於最後一個 bin。所有統計均
    為 descriptive association，不構成因果或生物機制主張。

    Args:
        retained_cells: 含唯一 cell identity、group、Orientation 與 group target
            的 19,648-row 正式母體。

    Returns:
        固定 12-bin table 與單列 machine-readable summary。

    Raises:
        TypeError: 輸入不是 pandas DataFrame 時拋出。
        PostCvAnalysisError: schema、正式母體或統計結果不符合契約時拋出。
    """
    if not isinstance(retained_cells, pd.DataFrame):
        raise TypeError("retained_cells 必須是 pandas DataFrame")
    if retained_cells.columns.has_duplicates:
        raise PostCvAnalysisError("retained_cells column labels 必須唯一")
    required = {
        "image_key",
        "cell_label",
        "group_id",
        "cell__Orientation",
        "group_IDO_score",
    }
    missing = sorted(required - set(retained_cells.columns))
    if missing:
        raise PostCvAnalysisError(f"retained_cells 缺少欄位：{missing}")
    identity = retained_cells.loc[:, ["image_key", "cell_label"]]
    if identity.isna().any().any() or identity.duplicated().any():
        raise PostCvAnalysisError(
            "retained cells require unique (image_key, cell_label) identity"
        )
    if len(retained_cells) != EXPECTED_CELL_COUNT:
        raise PostCvAnalysisError(
            "retained population 必須恰為 19,648 rows："
            f"actual={len(retained_cells)}"
        )
    try:
        cell_labels = retained_cells["cell_label"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PostCvAnalysisError("cell_label must be a positive integer") from error
    if (
        not np.isfinite(cell_labels).all()
        or np.any(cell_labels <= 0.0)
        or not np.equal(cell_labels, np.floor(cell_labels)).all()
    ):
        raise PostCvAnalysisError("cell_label must be a positive integer")
    raw_image_keys = retained_cells["image_key"].astype(str)
    raw_group_ids = retained_cells["group_id"].astype(str)
    image_keys = raw_image_keys.str.strip()
    group_ids = raw_group_ids.str.strip()
    if not raw_image_keys.eq(image_keys).all() or not raw_group_ids.eq(
        group_ids
    ).all():
        raise PostCvAnalysisError(
            "retained population canonical identifiers 不可含前後空白"
        )
    if image_keys.eq("").any() or int(image_keys.nunique()) != EXPECTED_FOV_COUNT:
        raise PostCvAnalysisError("retained population 必須恰含 693 FOV")
    if group_ids.eq("").any() or set(group_ids) != set(EXPECTED_GROUP_IDS):
        raise PostCvAnalysisError("retained population 必須恰含正式 9 groups")
    try:
        numeric_values = retained_cells.loc[
            :, ["cell__Orientation", "group_IDO_score"]
        ].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PostCvAnalysisError(
            "Orientation 與 group target 必須為 numeric 且 finite"
        ) from error
    if not np.isfinite(numeric_values).all():
        raise PostCvAnalysisError(
            "Orientation 與 group target 必須為 numeric 且 finite"
        )
    lower_bound = -np.pi / 2.0
    upper_bound = np.pi / 2.0
    orientations = numeric_values[:, 0]
    if np.any(orientations < lower_bound - ORIENTATION_RANGE_TOLERANCE) or np.any(
        orientations > upper_bound + ORIENTATION_RANGE_TOLERANCE
    ):
        raise PostCvAnalysisError(
            "Orientation 必須位於 axial range [-pi/2, pi/2]"
        )
    orientations = np.clip(orientations, lower_bound, upper_bound)
    targets = numeric_values[:, 1]
    target_counts = pd.DataFrame(
        {"group_id": group_ids.to_numpy(), "target": targets}
    ).groupby("group_id", sort=False)["target"].nunique(dropna=False)
    if not target_counts.eq(1).all():
        raise PostCvAnalysisError(
            "group target must be constant within each group"
        )
    edges = np.linspace(lower_bound, upper_bound, 13, dtype=np.float64)
    bin_indices = np.searchsorted(edges, orientations, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, 11)
    if np.any(np.bincount(bin_indices, minlength=12) == 0):
        raise PostCvAnalysisError(
            "Orientation population must occupy all 12 fixed bins"
        )

    bin_rows: list[dict[str, object]] = []
    for bin_index in range(12):
        selected = bin_indices == bin_index
        bin_cells = retained_cells.loc[selected]
        bin_orientations = orientations[selected]
        bin_targets = targets[selected]
        left = float(edges[bin_index])
        right = float(edges[bin_index + 1])
        midpoint = (left + right) / 2.0
        orientation_median = float(np.median(bin_orientations))
        bin_rows.append(
            {
                "bin_index": bin_index + 1,
                "left_edge_radians": left,
                "right_edge_radians": right,
                "midpoint_radians": midpoint,
                "left_edge_degrees": float(np.degrees(left)),
                "right_edge_degrees": float(np.degrees(right)),
                "midpoint_degrees": float(np.degrees(midpoint)),
                "n_cells": int(np.count_nonzero(selected)),
                "n_fovs": int(bin_cells["image_key"].nunique()),
                "n_groups": int(bin_cells["group_id"].nunique()),
                "orientation_median_radians": orientation_median,
                "orientation_median_degrees": float(
                    np.degrees(orientation_median)
                ),
                "target_median": float(np.median(bin_targets)),
                "target_mean": float(np.mean(bin_targets)),
                "target_p25": float(np.quantile(bin_targets, 0.25)),
                "target_p75": float(np.quantile(bin_targets, 0.75)),
            }
        )
    bins = pd.DataFrame(bin_rows, columns=ORIENTATION_BIN_COLUMNS)

    group_rows: list[dict[str, float]] = []
    for _, group in retained_cells.groupby("group_id", sort=False):
        group_rows.append(
            {
                "orientation_median": float(group["cell__Orientation"].median()),
                "target": float(group["group_IDO_score"].iloc[0]),
            }
        )
    groups = pd.DataFrame(group_rows)
    if (
        groups["orientation_median"].nunique(dropna=False) < 2
        or groups["target"].nunique(dropna=False) < 2
    ):
        raise PostCvAnalysisError(
            "group-level Spearman requires varying group medians and targets"
        )
    cell_spearman = spearmanr(orientations, targets)
    group_spearman = spearmanr(groups["orientation_median"], groups["target"])
    spearman_values = np.asarray(
        [
            cell_spearman.statistic,
            cell_spearman.pvalue,
            group_spearman.statistic,
            group_spearman.pvalue,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(spearman_values).all():
        raise PostCvAnalysisError("Orientation Spearman results 必須全為 finite")
    median_min = float(bins["target_median"].min())
    median_max = float(bins["target_median"].max())
    mean_min = float(bins["target_mean"].min())
    mean_max = float(bins["target_mean"].max())
    summary = pd.DataFrame(
        [
            {
                "n_cells": int(len(retained_cells)),
                "n_fovs": int(retained_cells["image_key"].nunique()),
                "n_groups": int(retained_cells["group_id"].nunique()),
                "n_bins": int(len(bins)),
                "cell_spearman_r": float(cell_spearman.statistic),
                "cell_spearman_p_value": float(cell_spearman.pvalue),
                "group_spearman_r": float(group_spearman.statistic),
                "group_spearman_p_value": float(group_spearman.pvalue),
                "bin_target_median_min": median_min,
                "bin_target_median_max": median_max,
                "bin_target_median_range": median_max - median_min,
                "bin_target_mean_min": mean_min,
                "bin_target_mean_max": mean_max,
                "bin_target_mean_range": mean_max - mean_min,
                "every_bin_covers_all_9_groups": bool(
                    bins["n_groups"].eq(len(EXPECTED_GROUP_IDS)).all()
                ),
                "interpretation_scope": "descriptive_noncausal",
            }
        ],
        columns=ORIENTATION_SUMMARY_COLUMNS,
    )
    return OrientationMarginalResult(bins=bins, summary=summary)


__all__ = [
    "ORIENTATION_BIN_COLUMNS",
    "ORIENTATION_RANGE_TOLERANCE",
    "ORIENTATION_SUMMARY_COLUMNS",
    "SHRINKAGE_GROUP_COLUMNS",
    "SHRINKAGE_REPORT_COLUMNS",
    "OrientationMarginalResult",
    "PostCvAnalysisError",
    "ShrinkageAnalysisResult",
    "analyze_orientation_marginal",
    "analyze_shrinkage",
]
