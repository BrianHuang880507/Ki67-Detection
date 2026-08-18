"""以 image-level morphology 建立 IDO response proxy 預測模型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


MODEL_FEATURE_SET = {
    "dummy_median": "dummy",
    "dose_ridge": "dose",
    "morphology_ridge": "morphology",
    "morphology_elasticnet": "morphology",
    "dose_plus_morphology_ridge": "dose_plus_morphology",
}


def _safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    """計算 Spearman correlation，常數向量時回傳 NaN。"""
    if observed.size < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(spearmanr(observed, predicted).statistic)


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """計算 IDO proxy regression 的主要與次要指標。

    Args:
        observed: 實際 image-level IDO_score。
        predicted: 模型預測 IDO_score。

    Returns:
        MAE、RMSE、R² 與 Spearman correlation。
    """
    y_true = np.asarray(observed, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def _inner_splits(
    conditions: Sequence[str], requested_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """依刺激條件建立可重現的 inner-CV splits。"""
    labels = pd.Series(conditions, dtype=str).reset_index(drop=True)
    min_count = int(labels.value_counts().min())
    n_splits = min(int(requested_splits), min_count)
    if n_splits < 2:
        raise ValueError("訓練 fold 內每個 condition 至少需要兩個樣本。")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def _pipeline(model: RegressorMixin) -> Pipeline:
    """建立所有 preprocessing 都位於 fold 內的 regression pipeline。"""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _fit_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    conditions: Sequence[str],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """在單一 outer fold 內調整超參數並 fit 模型。"""
    if model_name == "dummy_median":
        estimator = DummyRegressor(strategy="median")
        estimator.fit(np.zeros((len(y_train), 1)), y_train)
        return estimator, {}

    if model_name == "morphology_elasticnet":
        estimator = _pipeline(
            ElasticNet(max_iter=int(config.get("elasticnet_max_iter", 50000)), random_state=seed)
        )
        param_grid = {
            "model__alpha": [float(value) for value in config["elasticnet_alphas"]],
            "model__l1_ratio": [float(value) for value in config["elasticnet_l1_ratios"]],
        }
    else:
        estimator = _pipeline(Ridge())
        param_grid = {
            "model__alpha": [float(value) for value in config["ridge_alphas"]]
        }

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=_inner_splits(conditions, int(config.get("inner_splits", 3)), seed),
        n_jobs=int(config.get("n_jobs", 1)),
        error_score="raise",
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, dict(search.best_params_)


def _feature_columns_for_model(
    model_name: str,
    morphology_columns: Sequence[str],
) -> list[str]:
    """取得各比較模型允許使用的 predictors。"""
    dose_columns = ["IFN_dose", "TNF_dose", "IFN_x_TNF"]
    feature_set = MODEL_FEATURE_SET[model_name]
    if feature_set == "dummy":
        return []
    if feature_set == "dose":
        return dose_columns
    if feature_set == "morphology":
        return list(morphology_columns)
    return [*dose_columns, *morphology_columns]


def _predict(estimator: Any, model_name: str, x: pd.DataFrame, count: int) -> np.ndarray:
    """統一 Dummy 與 pipeline 模型的預測介面。"""
    if model_name == "dummy_median":
        return np.asarray(estimator.predict(np.zeros((count, 1))), dtype=float)
    return np.asarray(estimator.predict(x), dtype=float)


def _coefficient_rows(
    estimator: Any,
    model_name: str,
    feature_columns: Sequence[str],
    validation: str,
    repeat: int,
    fold: str,
) -> list[dict[str, Any]]:
    """擷取標準化 predictors 上的線性模型 coefficients。"""
    if model_name == "dummy_median" or not feature_columns:
        return []
    model = estimator.named_steps["model"]
    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    return [
        {
            "validation": validation,
            "repeat": repeat,
            "fold": fold,
            "model": model_name,
            "feature": feature,
            "coefficient_standardized": float(coefficient),
        }
        for feature, coefficient in zip(feature_columns, coefficients, strict=True)
    ]


def _prepare_dataset(images: pd.DataFrame) -> pd.DataFrame:
    """建立 dose interaction 並清理非有限數值。"""
    dataset = images.copy()
    dataset["IFN_x_TNF"] = dataset["IFN_dose"] * dataset["TNF_dose"]
    return dataset.replace([np.inf, -np.inf], np.nan)


def run_repeated_cross_validation(
    images: pd.DataFrame,
    morphology_columns: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """執行 condition-stratified repeated CV。

    Returns:
        Out-of-fold predictions、每個 repeat 的 metrics 與 fold coefficients。
    """
    dataset = _prepare_dataset(images)
    y = dataset["IDO_score"].astype(float)
    conditions = dataset["condition"].astype(str)
    models = list(MODEL_FEATURE_SET)
    seeds = [int(value) for value in config.get("seeds", [42, 314, 2026])]
    outer_splits = int(config.get("outer_splits", 5))
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for repeat, seed in enumerate(seeds):
        splitter = StratifiedKFold(
            n_splits=outer_splits,
            shuffle=True,
            random_state=seed,
        )
        for fold_index, (train_index, test_index) in enumerate(
            splitter.split(dataset, conditions), start=1
        ):
            train_conditions = conditions.iloc[train_index]
            for model_name in models:
                feature_columns = _feature_columns_for_model(
                    model_name, morphology_columns
                )
                x_train = dataset.iloc[train_index][feature_columns]
                x_test = dataset.iloc[test_index][feature_columns]
                estimator, best_params = _fit_model(
                    model_name,
                    x_train,
                    y.iloc[train_index],
                    train_conditions,
                    config,
                    seed + fold_index,
                )
                predictions = _predict(
                    estimator, model_name, x_test, len(test_index)
                )
                for local_index, row_index in enumerate(test_index):
                    prediction_rows.append(
                        {
                            "validation": "repeated_cv",
                            "repeat": repeat,
                            "seed": seed,
                            "fold": str(fold_index),
                            "model": model_name,
                            "image_key": dataset.iloc[row_index]["image_key"],
                            "condition": dataset.iloc[row_index]["condition"],
                            "IFN_dose": dataset.iloc[row_index]["IFN_dose"],
                            "TNF_dose": dataset.iloc[row_index]["TNF_dose"],
                            "observed_IDO_score": float(y.iloc[row_index]),
                            "predicted_IDO_score": float(predictions[local_index]),
                            "best_params": json.dumps(best_params, sort_keys=True),
                        }
                    )
                coefficient_rows.extend(
                    _coefficient_rows(
                        estimator,
                        model_name,
                        feature_columns,
                        "repeated_cv",
                        repeat,
                        str(fold_index),
                    )
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics_rows = []
    for (repeat, model_name), group in predictions.groupby(["repeat", "model"]):
        metrics_rows.append(
            {
                "validation": "repeated_cv",
                "repeat": int(repeat),
                "model": model_name,
                "n_images": int(len(group)),
                **regression_metrics(
                    group["observed_IDO_score"], group["predicted_IDO_score"]
                ),
            }
        )
    return predictions, pd.DataFrame(metrics_rows), pd.DataFrame(coefficient_rows)


def run_leave_one_condition_out(
    images: pd.DataFrame,
    morphology_columns: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """執行 leave-one-condition-out robustness validation。"""
    dataset = _prepare_dataset(images)
    y = dataset["IDO_score"].astype(float)
    conditions = dataset["condition"].astype(str)
    seed = int(config.get("loco_seed", 42))
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for held_out_condition in sorted(conditions.unique()):
        test_mask = conditions.eq(held_out_condition).to_numpy()
        train_index = np.flatnonzero(~test_mask)
        test_index = np.flatnonzero(test_mask)
        train_conditions = conditions.iloc[train_index]
        for model_name in MODEL_FEATURE_SET:
            feature_columns = _feature_columns_for_model(model_name, morphology_columns)
            x_train = dataset.iloc[train_index][feature_columns]
            x_test = dataset.iloc[test_index][feature_columns]
            estimator, best_params = _fit_model(
                model_name,
                x_train,
                y.iloc[train_index],
                train_conditions,
                config,
                seed,
            )
            predictions = _predict(estimator, model_name, x_test, len(test_index))
            for local_index, row_index in enumerate(test_index):
                prediction_rows.append(
                    {
                        "validation": "leave_one_condition_out",
                        "repeat": 0,
                        "seed": seed,
                        "fold": held_out_condition,
                        "model": model_name,
                        "image_key": dataset.iloc[row_index]["image_key"],
                        "condition": dataset.iloc[row_index]["condition"],
                        "IFN_dose": dataset.iloc[row_index]["IFN_dose"],
                        "TNF_dose": dataset.iloc[row_index]["TNF_dose"],
                        "observed_IDO_score": float(y.iloc[row_index]),
                        "predicted_IDO_score": float(predictions[local_index]),
                        "best_params": json.dumps(best_params, sort_keys=True),
                    }
                )
            coefficient_rows.extend(
                _coefficient_rows(
                    estimator,
                    model_name,
                    feature_columns,
                    "leave_one_condition_out",
                    0,
                    held_out_condition,
                )
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics_rows = []
    for model_name, group in predictions.groupby("model"):
        metrics_rows.append(
            {
                "validation": "leave_one_condition_out",
                "repeat": 0,
                "model": model_name,
                "n_images": int(len(group)),
                **regression_metrics(
                    group["observed_IDO_score"], group["predicted_IDO_score"]
                ),
            }
        )
    return predictions, pd.DataFrame(metrics_rows), pd.DataFrame(coefficient_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """將 repeated-CV metrics 彙整為 median 與 IQR。"""
    rows = []
    for (validation, model), group in metrics.groupby(["validation", "model"]):
        row: dict[str, Any] = {
            "validation": validation,
            "model": model,
            "repeats": int(len(group)),
            "n_images": int(group["n_images"].max()),
        }
        for metric in ("mae", "rmse", "r2", "spearman"):
            values = group[metric].astype(float)
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_iqr"] = float(values.quantile(0.75) - values.quantile(0.25))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["validation", "mae_median"])


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """彙整不同 folds 的 standardized coefficients。"""
    if coefficients.empty:
        return coefficients
    grouped = coefficients.groupby(["validation", "model", "feature"])[
        "coefficient_standardized"
    ]
    summary = grouped.agg(coefficient_median="median", coefficient_mean="mean").reset_index()
    iqr = grouped.quantile(0.75) - grouped.quantile(0.25)
    summary = summary.merge(
        iqr.rename("coefficient_iqr").reset_index(),
        on=["validation", "model", "feature"],
    )
    summary["absolute_median"] = summary["coefficient_median"].abs()
    return summary.sort_values(
        ["validation", "model", "absolute_median"], ascending=[True, True, False]
    )


def fit_final_models(
    images: pd.DataFrame,
    morphology_columns: Sequence[str],
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    """以全部影像 fit 最終 morphology models 並保存可重用 pipeline。"""
    dataset = _prepare_dataset(images)
    target = dataset["IDO_score"].astype(float)
    conditions = dataset["condition"].astype(str)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in ("morphology_ridge", "morphology_elasticnet"):
        estimator, best_params = _fit_model(
            model_name,
            dataset[list(morphology_columns)],
            target,
            conditions,
            config,
            int(config.get("final_seed", 42)),
        )
        model_path = destination / f"{model_name}.joblib"
        joblib.dump(
            {
                "estimator": estimator,
                "feature_columns": list(morphology_columns),
                "target": "image-level background-corrected IDO_score",
                "scope": "B4 p6 within-lot exploratory model",
            },
            model_path,
        )
        rows.append(
            {
                "model": model_name,
                "model_path": str(model_path.resolve()),
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def plot_observed_vs_predicted(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """繪製 repeated-CV 的平均 OOF observed-vs-predicted 圖。"""
    subset = predictions[predictions["validation"].eq("repeated_cv")]
    averaged = (
        subset.groupby(["model", "image_key"], as_index=False)
        .agg(
            observed_IDO_score=("observed_IDO_score", "first"),
            predicted_IDO_score=("predicted_IDO_score", "mean"),
        )
    )
    models = list(MODEL_FEATURE_SET)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for axis, model_name in zip(axes.flat, models, strict=False):
        group = averaged[averaged["model"].eq(model_name)]
        axis.scatter(
            group["observed_IDO_score"],
            group["predicted_IDO_score"],
            s=24,
            alpha=0.75,
        )
        low = float(
            min(group["observed_IDO_score"].min(), group["predicted_IDO_score"].min())
        )
        high = float(
            max(group["observed_IDO_score"].max(), group["predicted_IDO_score"].max())
        )
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
        axis.set_title(model_name)
        axis.set_xlabel("Observed IDO score")
        axis.set_ylabel("OOF predicted IDO score")
    for axis in axes.flat[len(models) :]:
        axis.set_visible(False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_dose_response(summary: pd.DataFrame, output_path: str | Path) -> None:
    """繪製三條 IDO dose-response 曲線。"""
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for curve_name, group in summary.groupby("curve", sort=False):
        ordered = group.sort_values("dose")
        axis.errorbar(
            ordered["dose"],
            ordered["IDO_score_median"],
            yerr=ordered["IDO_score_sem"],
            marker="o",
            capsize=3,
            label=curve_name,
        )
    axis.set_xlabel("Stimulation dose (ng)")
    axis.set_ylabel("Background-corrected IDO score")
    axis.set_title("B4 p6 IDO dose-response (descriptive)")
    axis.legend()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
