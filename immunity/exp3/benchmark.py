"""提供 Exp3 可稽核的分組驗證切分與 regression metrics。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from immunity.exp3.feature_sets import PRIMARY_FOV_FEATURES, validate_phase_predictors


_OUTER_FAMILIES = (
    ("leave_one_b_out", "b_id"),
    ("leave_one_passage_out", "passage"),
    ("leave_one_group_out", "group_id"),
    ("leave_one_condition_out", "condition_index"),
)
_REQUIRED_IMAGE_COLUMNS = (
    "image_key",
    "b_id",
    "passage",
    "group_id",
    "condition_index",
)
_MANIFEST_METADATA_COLUMNS = (
    "b_id",
    "passage",
    "group_id",
    "condition_index",
    "condition",
    "ifn_dose",
    "tnf_dose",
    "fov",
)
_PAPER_FEATURES = (
    "cell__perimeter__median",
    "nucleus_cytoplasm_area_ratio__median",
    "cell__feret_length__median",
)
_LINEAR_MODELS = {
    "paper_linear_3f",
    "ridge",
    "elasticnet",
    "dose_ridge",
    "dose_plus_morphology_ridge",
}
_PERMUTATION_MODELS = {
    "rbf_svr",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
}
_PHASE_CANDIDATES = (
    "paper_linear_3f",
    "ridge",
    "elasticnet",
    "rbf_svr",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
)
_RANKING_VALIDATIONS = tuple(name for name, _ in _OUTER_FAMILIES)
_SENSITIVITY_VALIDATIONS = _RANKING_VALIDATIONS[:3]
_RAW_ANALYSES = {"raw", "raw_ido", "raw_target"}
_TARGET_LABEL = "image-level background-corrected IDO proxy"

OOF_COLUMNS = [
    "validation",
    "fold",
    "split_id",
    "model",
    "role",
    "feature_set",
    "image_key",
    "b_id",
    "passage",
    "group_id",
    "condition_index",
    "condition",
    "observed_ido_score",
    "predicted_ido_score",
]
FOLD_METRIC_COLUMNS = [
    "validation",
    "fold",
    "split_id",
    "model",
    "role",
    "feature_set",
    "n_train",
    "n_test",
    "mae",
    "rmse",
    "r2",
    "spearman",
    "observed_sd",
    "prediction_sd",
    "status",
]
HYPERPARAMETER_COLUMNS = [
    "validation",
    "fold",
    "split_id",
    "model",
    "seed",
    "best_params_json",
    "inner_best_mae",
]
FEATURE_IMPORTANCE_COLUMNS = [
    "validation",
    "fold",
    "split_id",
    "model",
    "role",
    "feature_set",
    "feature",
    "importance_type",
    "importance",
    "importance_sd",
]
FAILURE_COLUMNS = [
    "validation",
    "fold",
    "split_id",
    "model",
    "exception_type",
    "message",
]

PARAMETERS: dict[str, dict[str, list[Any]]] = {
    "ridge": {
        "regressor__model__alpha": [
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
            1000,
            10000,
        ]
    },
    "elasticnet": {
        "regressor__model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
        "regressor__model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
    },
    "rbf_svr": {
        "regressor__model__C": [0.1, 1, 10, 100],
        "regressor__model__gamma": ["scale", 0.01, 0.1, 1.0],
        "regressor__model__epsilon": [0.05, 0.1, 0.2],
    },
    "random_forest": {
        "regressor__model__max_depth": [None, 4, 8, 16],
        "regressor__model__min_samples_leaf": [1, 5, 10],
        "regressor__model__max_features": [1.0, "sqrt", 0.5],
    },
    "extra_trees": {
        "regressor__model__max_depth": [None, 4, 8, 16],
        "regressor__model__min_samples_leaf": [1, 5, 10],
        "regressor__model__max_features": [1.0, "sqrt", 0.5],
    },
    "hist_gradient_boosting": {
        "regressor__model__learning_rate": [0.03, 0.1],
        "regressor__model__max_leaf_nodes": [7, 15, 31],
        "regressor__model__min_samples_leaf": [10, 20, 40],
        "regressor__model__l2_regularization": [0, 1, 10],
    },
}


@dataclass(frozen=True)
class ModelSpec:
    """描述一個 benchmark model 的固定 contract。

    Attributes:
        name: Model registry 唯一名稱。
        role: ``candidate`` 或 ``diagnostic``。
        feature_set: 模型使用的 feature set 名稱。
        scaled: Predictor pipeline 是否包含 ``StandardScaler``。
        estimator_factory: 接收 deterministic seed 並建立未 fit estimator 的 factory。
        parameters: 傳給 nested search 的 hyperparameter space。
    """

    name: str
    role: str
    feature_set: str
    scaled: bool
    estimator_factory: Callable[[int], RegressorMixin]
    parameters: dict[str, list[Any]]


@dataclass(frozen=True)
class BenchmarkResult:
    """保存 shared-split nested benchmark 的逐 fold 結果。

    Attributes:
        predictions: Outer-test OOF predictions。
        fold_metrics: 每個 model 與 outer fold 的 regression metrics/status。
        hyperparameters: 每個成功 fold 的 deterministic tuning 結果。
        feature_importance: 只供診斷的 fold-level coefficients/permutation importance。
        failures: 不會中止整體 benchmark 的 fold failures。
    """

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    hyperparameters: pd.DataFrame
    feature_importance: pd.DataFrame
    failures: pd.DataFrame


def build_model_registry(config: Mapping[str, Any]) -> dict[str, ModelSpec]:
    """建立精確七個 candidate 與三個 diagnostic models。

    Args:
        config: Benchmark 設定；測試可覆寫 ``tree_estimators`` 與 ``n_jobs``。

    Returns:
        依固定順序建立的十模型 registry。

    Raises:
        ValueError: Tree estimator 數量或 ``n_jobs`` 不是有效整數時拋出。
    """
    tree_estimators = _positive_int(
        config.get("tree_estimators", 400), "tree_estimators"
    )
    n_jobs = int(config.get("n_jobs", 1))
    if n_jobs == 0:
        raise ValueError("n_jobs 不可為 0")

    def paper_factory(seed: int) -> RegressorMixin:
        return _target_scaled_pipeline(LinearRegression(), scaled=True)

    def ridge_factory(seed: int) -> RegressorMixin:
        return _target_scaled_pipeline(Ridge(), scaled=True)

    def elasticnet_factory(seed: int) -> RegressorMixin:
        return _target_scaled_pipeline(
            ElasticNet(max_iter=50000, random_state=seed), scaled=True
        )

    def svr_factory(seed: int) -> RegressorMixin:
        return _target_scaled_pipeline(SVR(kernel="rbf"), scaled=True)

    def random_forest_factory(seed: int) -> RegressorMixin:
        return _target_scaled_pipeline(
            RandomForestRegressor(
                n_estimators=tree_estimators,
                random_state=seed,
                n_jobs=n_jobs,
            ),
            scaled=False,
        )

    def extra_trees_factory(seed: int) -> RegressorMixin:
        return _target_scaled_pipeline(
            ExtraTreesRegressor(
                n_estimators=tree_estimators,
                random_state=seed,
                n_jobs=n_jobs,
            ),
            scaled=False,
        )

    def hist_gradient_boosting_factory(seed: int) -> RegressorMixin:
        return _target_scaled_pipeline(
            HistGradientBoostingRegressor(random_state=seed), scaled=False
        )

    def dummy_factory(seed: int) -> RegressorMixin:
        return DummyRegressor(strategy="median")

    return {
        "paper_linear_3f": ModelSpec(
            "paper_linear_3f",
            "candidate",
            "paper_linear_3f",
            True,
            paper_factory,
            {},
        ),
        "ridge": ModelSpec(
            "ridge",
            "candidate",
            "basic_median",
            True,
            ridge_factory,
            _parameter_copy("ridge"),
        ),
        "elasticnet": ModelSpec(
            "elasticnet",
            "candidate",
            "basic_median",
            True,
            elasticnet_factory,
            _parameter_copy("elasticnet"),
        ),
        "rbf_svr": ModelSpec(
            "rbf_svr",
            "candidate",
            "basic_median",
            True,
            svr_factory,
            _parameter_copy("rbf_svr"),
        ),
        "random_forest": ModelSpec(
            "random_forest",
            "candidate",
            "basic_median",
            False,
            random_forest_factory,
            _parameter_copy("random_forest"),
        ),
        "extra_trees": ModelSpec(
            "extra_trees",
            "candidate",
            "basic_median",
            False,
            extra_trees_factory,
            _parameter_copy("extra_trees"),
        ),
        "hist_gradient_boosting": ModelSpec(
            "hist_gradient_boosting",
            "candidate",
            "basic_median",
            False,
            hist_gradient_boosting_factory,
            _parameter_copy("hist_gradient_boosting"),
        ),
        "dummy_median": ModelSpec(
            "dummy_median",
            "diagnostic",
            "none",
            False,
            dummy_factory,
            {},
        ),
        "dose_ridge": ModelSpec(
            "dose_ridge",
            "diagnostic",
            "dose",
            True,
            ridge_factory,
            _parameter_copy("ridge"),
        ),
        "dose_plus_morphology_ridge": ModelSpec(
            "dose_plus_morphology_ridge",
            "diagnostic",
            "dose_plus_morphology",
            True,
            ridge_factory,
            _parameter_copy("ridge"),
        ),
    }


def _target_scaled_pipeline(model: RegressorMixin, scaled: bool) -> RegressorMixin:
    """將 predictor preprocessing 與 fold-local target scaling 綁進 estimator。"""
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )
    ]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return TransformedTargetRegressor(
        regressor=Pipeline(steps),
        transformer=StandardScaler(),
    )


def _parameter_copy(name: str) -> dict[str, list[Any]]:
    """複製 parameter space，避免 caller 修改全域正式設定。"""
    return {key: list(values) for key, values in PARAMETERS[name].items()}


def _positive_int(value: Any, name: str) -> int:
    """將設定轉為正整數，拒絕截斷小數。"""
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是正整數") from error
    if converted != value or converted < 1:
        raise ValueError(f"{name} 必須是正整數")
    return converted


@dataclass(frozen=True)
class OuterSplit:
    """保存一個可稽核的 outer train/test split。

    Attributes:
        validation: Outer validation family 名稱。
        fold: 被留出的分組值。
        train_index: 相對於輸入 image frame 的 training positional indices。
        test_index: 相對於輸入 image frame 的 test positional indices。
    """

    validation: str
    fold: str
    train_index: np.ndarray
    test_index: np.ndarray


def make_outer_splits(images: pd.DataFrame) -> list[OuterSplit]:
    """建立四種 deterministic grouped outer split families。

    Args:
        images: 含 image key 與 Exp3 分組 metadata 的 image-level frame。

    Returns:
        依 validation family 與排序後分組值排列的 outer splits。

    Raises:
        ValueError: 輸入為空、必要 metadata 缺失或無法形成安全 split 時拋出。
    """
    _validate_images(images)
    positions = np.arange(len(images), dtype=np.int64)
    splits: list[OuterSplit] = []
    for validation, column in _OUTER_FAMILIES:
        family: list[OuterSplit] = []
        for held_out in _sorted_unique(images[column]):
            test_mask = images[column].eq(held_out).to_numpy(dtype=bool)
            family.append(
                OuterSplit(
                    validation=validation,
                    fold=str(held_out),
                    train_index=positions[~test_mask],
                    test_index=positions[test_mask],
                )
            )
        _validate_split_family(family, len(images), validation)
        splits.extend(family)
    return splits


def outer_split_manifest(
    images: pd.DataFrame,
    splits: Sequence[OuterSplit],
) -> pd.DataFrame:
    """展開 outer splits，並以 positional index 保留每張影像的 metadata。

    Args:
        images: 產生 splits 時使用的原始 image-level frame。
        splits: 欲稽核或保存的 outer splits。

    Returns:
        每個 split、每張影像各一列的 train/test manifest。

    Raises:
        ValueError: Image metadata 不完整，或 split index 越界、重複、重疊、未完整
            覆蓋輸入列時拋出。
    """
    _validate_images(images)
    if not splits:
        raise ValueError("splits 不可為空")

    metadata_columns = [
        column for column in _MANIFEST_METADATA_COLUMNS if column in images.columns
    ]
    positional_metadata = images.iloc[np.arange(len(images))].reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for split in splits:
        train_index = _validated_positions(
            split.train_index, len(images), "train_index"
        )
        test_index = _validated_positions(split.test_index, len(images), "test_index")
        if np.intersect1d(train_index, test_index).size:
            raise ValueError("outer split 的 train_index 與 test_index 不可重疊")
        if train_index.size == 0 or test_index.size == 0:
            raise ValueError("outer split 的 training 與 test 都不可為空")

        roles = np.full(len(images), "", dtype=object)
        roles[train_index] = "train"
        roles[test_index] = "test"
        if np.any(roles == ""):
            raise ValueError("outer split 必須完整覆蓋每個 image row")

        frame = pd.DataFrame(
            {
                "validation": split.validation,
                "fold": split.fold,
                "image_key": positional_metadata["image_key"].to_numpy(),
                "role": roles,
            }
        )
        frame = pd.concat([frame, positional_metadata.loc[:, metadata_columns]], axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_inner_splits(
    training: pd.DataFrame,
    requested: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """以 outer-training frame 的 ``group_id`` 建立 GroupKFold splits。

    Args:
        training: 單一 outer fold 的 training frame。
        requested: 希望建立的 inner fold 數量。

    Returns:
        相對於 ``training`` 的 train/test positional index pairs。

    Raises:
        ValueError: requested 少於 2、group_id 無效，或 training 少於兩群時拋出。
    """
    if "group_id" not in training.columns:
        raise ValueError("training 缺少必要欄位：['group_id']")
    _validate_nonempty_values(training, ("group_id",))
    if int(requested) != requested or int(requested) < 2:
        raise ValueError("requested 必須至少為 2")

    groups = training["group_id"].to_numpy()
    group_count = int(training["group_id"].nunique(dropna=False))
    if group_count < 2:
        raise ValueError("inner validation 至少需要兩個不同的 group_id")
    splitter = GroupKFold(n_splits=min(int(requested), group_count))
    splits = [
        (
            np.asarray(train_index, dtype=np.int64),
            np.asarray(test_index, dtype=np.int64),
        )
        for train_index, test_index in splitter.split(training, groups=groups)
    ]
    _validate_inner_splits(training, splits)
    return splits


def regression_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    """計算具有明確 undefined behavior 的 regression metrics。

    Args:
        observed: 實際 target values。
        predicted: 預測 target values。

    Returns:
        MAE、RMSE、R2 與 Spearman correlation。樣本少於 2 或 observed
        constant 時 R2 為 NaN；樣本少於 2 或任一 vector constant 時
        Spearman 為 NaN。

    Raises:
        ValueError: Vector 非一維、為空、長度不同或含非有限值時拋出。
    """
    y_true = _metric_vector(observed, "observed")
    y_pred = _metric_vector(predicted, "predicted")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("observed 與 predicted 不可為空")
    if y_true.size != y_pred.size:
        raise ValueError("observed 與 predicted 長度必須相同")
    if not np.isfinite(y_true).all():
        raise ValueError("observed 必須全部為有限值")
    if not np.isfinite(y_pred).all():
        raise ValueError("predicted 必須全部為有限值")

    observed_constant = _is_constant(y_true)
    predicted_constant = _is_constant(y_pred)
    r2 = (
        np.nan
        if y_true.size < 2 or observed_constant
        else float(r2_score(y_true, y_pred))
    )
    spearman = (
        np.nan
        if y_true.size < 2 or observed_constant or predicted_constant
        else float(spearmanr(y_true, y_pred).statistic)
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "spearman": spearman,
    }


def run_nested_benchmark(
    images: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    splits: Sequence[OuterSplit],
    config: Mapping[str, Any],
    model_names: Sequence[str] | None = None,
) -> BenchmarkResult:
    """在共用 outer splits 執行 deterministic grouped nested benchmark。

    每個 ``OuterSplit × model`` 獨立執行。單一 fold 失敗時會保留 failed
    metric 與 failure record，並繼續處理其餘 models/folds；不會改用其他 split。

    Args:
        images: 含 metadata、``IDO_score``、dose 與 morphology features 的 frame。
        feature_sets: Feature set 名稱到 predictor columns 的 mapping。
        splits: 所有模型共用且已預先建立的 outer splits。
        config: Seed、inner folds、搜尋上限、平行度與 importance repeats。
        model_names: 可選的 registry model 子集；省略時執行全部十個 models。

    Returns:
        含 OOF predictions、fold metrics、hyperparameters、diagnostic importance
        與 failures 的 ``BenchmarkResult``。

    Raises:
        ValueError: 輸入 metadata、splits、model names 或數值設定無效時拋出。
    """
    _validate_images(images)
    _validate_benchmark_metadata(images)
    if not splits:
        raise ValueError("splits 不可為空")
    _validate_unique_split_ids(splits)

    registry = build_model_registry(config)
    selected_names = list(registry) if model_names is None else list(model_names)
    if not selected_names:
        raise ValueError("model_names 不可為空")
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("model_names 不可重複")
    unknown = sorted(set(selected_names) - set(registry))
    if unknown:
        raise ValueError(f"未知 model_names：{unknown}")
    _validate_candidate_feature_sets(feature_sets, selected_names, registry)

    seed = _random_seed(config.get("seed", 42))
    inner_splits = _positive_int(config.get("inner_splits", 5), "inner_splits")
    if inner_splits < 2:
        raise ValueError("inner_splits 至少為 2")
    candidate_cap = _positive_int(
        config.get("max_hyperparameter_candidates", 24),
        "max_hyperparameter_candidates",
    )
    permutation_repeats = _positive_int(
        config.get("permutation_repeats", 20), "permutation_repeats"
    )
    n_jobs = int(config.get("n_jobs", 1))

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for split in splits:
        train_index, test_index = _validate_benchmark_split(split, len(images))
        training = images.iloc[train_index].reset_index(drop=True)
        testing = images.iloc[test_index].reset_index(drop=True)
        split_id = f"{split.validation}:{split.fold}"

        for model_name in selected_names:
            spec = registry[model_name]
            metric_base = _metric_identity(
                split, split_id, spec, len(training), len(testing)
            )
            try:
                x_train, feature_names = _model_feature_frame(
                    training, spec, feature_sets
                )
                x_test, test_feature_names = _model_feature_frame(
                    testing, spec, feature_sets
                )
                if feature_names != test_feature_names:
                    raise ValueError("outer train/test feature columns 不一致")
                y_train = training["IDO_score"].to_numpy(dtype=float)
                y_test = testing["IDO_score"].to_numpy(dtype=float)
                estimator = spec.estimator_factory(seed)
                fitted, best_params, inner_best_mae = _fit_outer_model(
                    estimator,
                    spec.parameters,
                    x_train,
                    y_train,
                    training,
                    inner_splits,
                    candidate_cap,
                    seed,
                    n_jobs,
                )
                predicted = _finite_predictions(fitted.predict(x_test), len(testing))
                metrics = regression_metrics(y_test, predicted)
                core_prediction_rows = _prediction_records(
                    testing, predicted, split, split_id, spec
                )
                core_metric_row = {
                    **metric_base,
                    **metrics,
                    "observed_sd": float(np.std(y_test)),
                    "prediction_sd": float(np.std(predicted)),
                    "status": "ok",
                }
                core_hyperparameter_row = {
                    "validation": split.validation,
                    "fold": split.fold,
                    "split_id": split_id,
                    "model": model_name,
                    "seed": seed,
                    "best_params_json": json.dumps(
                        best_params, sort_keys=True, separators=(",", ":")
                    ),
                    "inner_best_mae": inner_best_mae,
                }
            except Exception as error:  # noqa: BLE001 - fold isolation is the contract
                metric_rows.append(
                    {
                        **metric_base,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "r2": np.nan,
                        "spearman": np.nan,
                        "observed_sd": _safe_standard_deviation(testing["IDO_score"]),
                        "prediction_sd": np.nan,
                        "status": "failed",
                    }
                )
                failure_rows.append(
                    {
                        "validation": split.validation,
                        "fold": split.fold,
                        "split_id": split_id,
                        "model": model_name,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                continue

            prediction_rows.extend(core_prediction_rows)
            metric_rows.append(core_metric_row)
            hyperparameter_rows.append(core_hyperparameter_row)

            try:
                fold_importance = _fold_feature_importance(
                    fitted,
                    spec,
                    feature_names,
                    x_test,
                    y_test,
                    split,
                    split_id,
                    permutation_repeats,
                    seed,
                    n_jobs,
                )
            except Exception as error:  # noqa: BLE001 - diagnostic only
                warnings.warn(
                    "feature importance diagnostic failed for "
                    f"split_id={split_id}, model={model_name}: "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                importance_rows.extend(fold_importance)

    return BenchmarkResult(
        predictions=pd.DataFrame(prediction_rows, columns=OOF_COLUMNS),
        fold_metrics=pd.DataFrame(metric_rows, columns=FOLD_METRIC_COLUMNS),
        hyperparameters=pd.DataFrame(
            hyperparameter_rows, columns=HYPERPARAMETER_COLUMNS
        ),
        feature_importance=pd.DataFrame(
            importance_rows, columns=FEATURE_IMPORTANCE_COLUMNS
        ),
        failures=pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS),
    )


def rank_phase_models(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    failures: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """以四種 validation 等權排名七個 phase-only candidates 並執行 gates。

    Args:
        fold_metrics: Outer-fold metrics；condition-adjusted rows 會被排除。
        predictions: Raw-target outer-test OOF predictions。
        failures: Outer-fold failure records。
        config: 含 ``simplicity_order``，並可含 ``feature_sets`` 的設定。

    Returns:
        每個 phase-only candidate 一列的可稽核 ranking，包含五項 eligibility
        gates、tie-break 欄位與唯一 winner 標記。

    Raises:
        ValueError: 輸入表格缺少 ranking 必要欄位，或 simplicity order 無效時。
    """
    metric_columns = {"validation", "fold", "model", "mae", "r2", "status"}
    prediction_columns = {
        "validation",
        "model",
        "observed_ido_score",
        "predicted_ido_score",
    }
    missing_metrics = sorted(metric_columns - set(fold_metrics.columns))
    missing_predictions = sorted(prediction_columns - set(predictions.columns))
    if missing_metrics:
        raise ValueError(f"fold_metrics 缺少 ranking 欄位：{missing_metrics}")
    if missing_predictions:
        raise ValueError(f"predictions 缺少 ranking 欄位：{missing_predictions}")

    raw_metrics = _raw_analysis_rows(fold_metrics)
    raw_predictions = _raw_analysis_rows(predictions)
    raw_metrics = _with_split_ids(raw_metrics)
    raw_predictions = _with_split_ids(raw_predictions)
    candidate_metrics = raw_metrics[raw_metrics["model"].isin(_PHASE_CANDIDATES)]
    ok_metrics = candidate_metrics[candidate_metrics["status"].eq("ok")]

    expected_splits = {
        validation: _expected_validation_splits(raw_metrics, validation)
        for validation in _RANKING_VALIDATIONS
    }
    validation_summaries: dict[str, pd.DataFrame] = {}
    for validation in _RANKING_VALIDATIONS:
        family = ok_metrics[ok_metrics["validation"].eq(validation)]
        summary = family.groupby("model", sort=False).agg(
            mae=("mae", "median"),
            r2=("r2", "median"),
            spearman=("spearman", "median")
            if "spearman" in family.columns
            else ("mae", lambda values: np.nan),
        )
        summary["rank"] = summary["mae"].rank(method="average", ascending=True)
        validation_summaries[validation] = summary

    dummy_medians = {
        validation: _dummy_validation_mae(raw_metrics, validation)
        for validation in _RANKING_VALIDATIONS
    }
    simplicity = _simplicity_ranks(config)
    rows: list[dict[str, Any]] = []
    for model in _PHASE_CANDIDATES:
        row: dict[str, Any] = {"model": model}
        mae_wins = 0
        positive_r2 = 0
        prediction_sd_checks: list[bool] = []
        for validation in _RANKING_VALIDATIONS:
            summary = validation_summaries[validation]
            if model in summary.index:
                values = summary.loc[model]
                mae = float(values["mae"])
                r2 = float(values["r2"])
                spearman = float(values["spearman"])
                rank = float(values["rank"])
            else:
                mae = r2 = spearman = rank = np.nan
            dummy_mae = dummy_medians[validation]
            if np.isfinite(mae) and np.isfinite(dummy_mae) and mae < dummy_mae:
                mae_wins += 1
            if np.isfinite(r2) and r2 > 0:
                positive_r2 += 1

            observed_sd, prediction_sd, prediction_gate = _prediction_sd_check(
                raw_predictions, model, validation
            )
            prediction_sd_checks.append(prediction_gate)
            row.update(
                {
                    f"{validation}_mae": mae,
                    f"{validation}_median_r2": r2,
                    f"{validation}_median_spearman": spearman,
                    f"{validation}_rank": rank,
                    f"{validation}_weight": 0.25,
                    f"{validation}_oof_observed_sd": observed_sd,
                    f"{validation}_oof_prediction_sd": prediction_sd,
                    f"{validation}_prediction_sd_gate": prediction_gate,
                }
            )

        ranks = np.asarray(
            [row[f"{validation}_rank"] for validation in _RANKING_VALIDATIONS],
            dtype=float,
        )
        complete_gate = _complete_outer_folds_gate(
            raw_metrics, raw_predictions, failures, model, expected_splits
        )
        feature_gate = _phase_feature_gate(model, raw_metrics, config)
        gates = {
            "mae_beats_dummy_gate": mae_wins >= 3,
            "positive_r2_gate": positive_r2 >= 2,
            "complete_outer_folds_gate": complete_gate,
            "phase_only_feature_gate": feature_gate,
            "prediction_sd_gate": all(prediction_sd_checks),
        }
        reasons = [
            reason
            for gate, reason in (
                (gates["mae_beats_dummy_gate"], "mae_not_better_than_dummy_in_3_validations"),
                (gates["positive_r2_gate"], "positive_median_r2_in_fewer_than_2_validations"),
                (gates["complete_outer_folds_gate"], "failed_or_missing_outer_folds"),
                (gates["phase_only_feature_gate"], "feature_leakage_or_nonexact_whitelist"),
                (gates["prediction_sd_gate"], "prediction_sd_below_5_percent"),
            )
            if not gate
        ]
        finite_ranks = ranks[np.isfinite(ranks)]
        row.update(
            {
                "validations_beating_dummy": mae_wins,
                "validations_with_positive_r2": positive_r2,
                **gates,
                "eligible": not reasons,
                "ineligibility_reasons_json": json.dumps(
                    reasons, ensure_ascii=False, separators=(",", ":")
                ),
                "average_rank": (
                    float(np.mean(ranks)) if finite_ranks.size == len(ranks) else np.nan
                ),
                "worst_validation_rank": (
                    float(np.max(ranks)) if finite_ranks.size == len(ranks) else np.nan
                ),
                "leave_one_b_out_mae": row["leave_one_b_out_mae"],
                "overall_oof_spearman": _overall_oof_spearman(
                    raw_predictions, model
                ),
                "simplicity_rank": simplicity[model],
                "tie_threshold": 0.25,
                "tied_with_best": False,
                "selection_order": np.nan,
                "winner": False,
            }
        )
        rows.append(row)

    ranking = pd.DataFrame(rows)
    ranking["overall_rank"] = ranking["average_rank"]
    eligible = ranking[ranking["eligible"] & ranking["average_rank"].notna()].copy()
    if eligible.empty:
        return ranking

    best_rank = float(eligible["average_rank"].min())
    tied_models = eligible.loc[
        eligible["average_rank"].le(best_rank + 0.25 + 1e-12), "model"
    ].tolist()
    ranking.loc[ranking["model"].isin(tied_models), "tied_with_best"] = True
    ordered_models = _selection_order(ranking, tied_models)
    for order, model in enumerate(ordered_models, start=1):
        ranking.loc[ranking["model"].eq(model), "selection_order"] = order
    ranking.loc[ranking["model"].eq(ordered_models[0]), "winner"] = True
    return ranking


def select_winner(ranking: pd.DataFrame) -> str | None:
    """從 ranking 取得唯一 eligible phase-only winner。

    Args:
        ranking: ``rank_phase_models`` 產生的 ranking。

    Returns:
        Winner model name；沒有 eligible model 時回傳 ``None``。

    Raises:
        ValueError: Ranking 標示多個 winners 或 winner 不是 phase candidate 時。
    """
    required = {"model", "eligible", "winner"}
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"ranking 缺少 winner 欄位：{missing}")
    winners = ranking[ranking["winner"].fillna(False).astype(bool)]
    if winners.empty:
        return None
    if len(winners) != 1:
        raise ValueError("ranking 必須至多標示一個 winner")
    winner = str(winners.iloc[0]["model"])
    if winner not in _PHASE_CANDIDATES or not bool(winners.iloc[0]["eligible"]):
        raise ValueError("winner 必須是 eligible phase-only candidate")
    return winner


def condition_adjust_targets(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """只以 outer-training condition means 建立 training/test residual targets。

    Args:
        train: Outer-training rows，須含 ``condition_index`` 與 ``IDO_score``。
        test: Outer-test rows，須含相同欄位且 condition 已存在於 training。

    Returns:
        與輸入 index 對齊的 training residual 與 test residual。

    Raises:
        ValueError: 必要欄位、target 或 training-derived condition mean 無效時。
    """
    required = {"condition_index", "IDO_score"}
    for name, frame in (("train", train), ("test", test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} 缺少 condition adjustment 欄位：{missing}")
        if frame.empty:
            raise ValueError(f"{name} 不可為空")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} condition 與 IDO_score 不可缺漏")

    means = train.groupby("condition_index", sort=False)["IDO_score"].mean()
    train_means = train["condition_index"].map(means)
    test_means = test["condition_index"].map(means)
    if test_means.isna().any():
        missing_conditions = sorted(
            test.loc[test_means.isna(), "condition_index"].astype(str).unique()
        )
        raise ValueError(
            "outer training 缺少 test condition mean：" f"{missing_conditions}"
        )
    train_target = pd.to_numeric(train["IDO_score"], errors="coerce")
    test_target = pd.to_numeric(test["IDO_score"], errors="coerce")
    train_residual = train_target - train_means
    test_residual = test_target - test_means
    if not np.isfinite(train_residual).all() or not np.isfinite(test_residual).all():
        raise ValueError("condition-adjusted targets 必須全部為有限值")
    return train_residual, test_residual


def run_condition_adjusted_sensitivity(
    images: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    splits: Sequence[OuterSplit],
    config: Mapping[str, Any],
    model_names: Sequence[str],
) -> pd.DataFrame:
    """執行三種 outer validations 的 training-mean residual sensitivity。

    Args:
        images: 含 target、condition、metadata 與 phase features 的 image frame。
        feature_sets: Phase-only feature whitelist mapping。
        splits: 共用 outer splits；LOCO 會明確排除。
        config: Nested benchmark 設定。
        model_names: 要執行的 phase-only candidate names。

    Returns:
        Fold metrics，並以 ``analysis`` 標記 sensitivity；不回傳 ranking。

    Raises:
        ValueError: Model names 含 diagnostic/non-candidate 或 split family 未知時。
    """
    selected_names = list(model_names)
    if not selected_names or any(name not in _PHASE_CANDIDATES for name in selected_names):
        raise ValueError("condition sensitivity 只接受 phase-only candidate models")
    unknown_validations = sorted(
        {split.validation for split in splits} - set(_RANKING_VALIDATIONS)
    )
    if unknown_validations:
        raise ValueError(f"未知 outer validation：{unknown_validations}")

    frames: list[pd.DataFrame] = []
    for split in splits:
        if split.validation not in _SENSITIVITY_VALIDATIONS:
            continue
        train_index, test_index = _validate_benchmark_split(split, len(images))
        training = images.iloc[train_index].reset_index(drop=True)
        testing = images.iloc[test_index].reset_index(drop=True)
        train_residual, test_residual = condition_adjust_targets(training, testing)
        adjusted = images.copy()
        target_position = adjusted.columns.get_loc("IDO_score")
        adjusted.iloc[train_index, target_position] = train_residual.to_numpy(dtype=float)
        adjusted.iloc[test_index, target_position] = test_residual.to_numpy(dtype=float)
        result = run_nested_benchmark(
            adjusted,
            feature_sets,
            [split],
            config,
            model_names=selected_names,
        )
        metrics = result.fold_metrics.copy()
        metrics["analysis"] = "training_condition_mean_residual"
        frames.append(metrics)
    if not frames:
        return pd.DataFrame(columns=[*FOLD_METRIC_COLUMNS, "analysis"])
    return pd.concat(frames, ignore_index=True).loc[
        :, [*FOLD_METRIC_COLUMNS, "analysis"]
    ]


def fit_final_phase_model(
    images: pd.DataFrame,
    winner: str | None,
    feature_sets: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> Path | None:
    """以全 FOV rows 重做 grouped inner tuning 並保存 eligible phase winner。

    Args:
        images: 全部 image-level FOV rows。
        winner: ``select_winner`` 回傳的 candidate name；``None`` 表示 no winner。
        feature_sets: Exact phase-only feature whitelist mapping。
        config: Deterministic tuning 設定；可含 ``eligible_phase_models``。
        output_dir: 已驗證的 ``immunity/outputs/exp3`` 內部目錄。

    Returns:
        Final joblib bundle path；no winner 時回傳 ``None`` 且不建立輸出。

    Raises:
        ValueError: Winner 不合格、feature whitelist 不精確或輸出路徑逸出時。
    """
    if winner is None:
        return None
    registry = build_model_registry(config)
    if winner not in _PHASE_CANDIDATES or registry[winner].role != "candidate":
        raise ValueError("winner 必須是 eligible phase-only candidate")
    declared_eligible = config.get("eligible_phase_models")
    if declared_eligible is not None and winner not in list(declared_eligible):
        raise ValueError("winner 必須是 eligible phase-only candidate")
    _validate_images(images)
    _validate_benchmark_metadata(images)
    _validate_candidate_feature_sets(feature_sets, [winner], registry)
    spec = registry[winner]
    x_train, feature_names = _model_feature_frame(images, spec, feature_sets)
    if tuple(feature_names) not in {
        tuple(_PAPER_FEATURES),
        tuple(PRIMARY_FOV_FEATURES),
    }:
        raise ValueError("final model 必須使用 exact phase-only whitelist")

    from immunity.exp3.run_benchmark import resolve_exp3_output_dir

    safe_output_dir = resolve_exp3_output_dir(output_dir)
    model_dir = _safe_final_path(safe_output_dir, "models", directory=True)
    model_path = _safe_final_path(model_dir, f"{winner}.joblib", directory=False)
    metadata_path = _safe_final_path(
        safe_output_dir, "final_model.json", directory=False
    )

    seed = _random_seed(config.get("seed", 42))
    inner_splits = _positive_int(config.get("inner_splits", 5), "inner_splits")
    if inner_splits < 2:
        raise ValueError("inner_splits 至少為 2")
    candidate_cap = _positive_int(
        config.get("max_hyperparameter_candidates", 24),
        "max_hyperparameter_candidates",
    )
    n_jobs = int(config.get("n_jobs", 1))
    fitted, best_params, inner_best_mae = _fit_outer_model(
        spec.estimator_factory(seed),
        spec.parameters,
        x_train,
        images["IDO_score"].to_numpy(dtype=float),
        images.reset_index(drop=True),
        inner_splits,
        candidate_cap,
        seed,
        n_jobs,
    )
    config_hash = _stable_hash(config)
    manifest_hash = str(
        images.attrs.get("manifest_hash")
        or config.get("manifest_hash")
        or _image_manifest_hash(images)
    )
    bundle = {
        "pipeline": fitted,
        "feature_columns": feature_names,
        "target": _TARGET_LABEL,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "model": winner,
        "training_scope": "all_fov_rows_grouped_inner_tuning",
    }
    metadata = {
        "model": winner,
        "model_path": f"models/{winner}.joblib",
        "feature_columns": feature_names,
        "target": _TARGET_LABEL,
        "training_scope": "all_fov_rows_grouped_inner_tuning",
        "best_params": best_params,
        "inner_best_mae": inner_best_mae,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "seed": seed,
    }

    safe_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=False, exist_ok=True)
    _assert_unlinked_final_path(model_dir, safe_output_dir / "models")
    _write_joblib_atomically(bundle, model_path)
    _write_json_atomically(metadata, metadata_path)
    return model_path


def _raw_analysis_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """排除 condition-adjusted sensitivity rows，避免其進入正式 ranking。"""
    if "analysis" not in frame.columns:
        return frame.copy()
    analysis = frame["analysis"]
    return frame[analysis.isna() | analysis.astype(str).isin(_RAW_ANALYSES)].copy()


def _with_split_ids(metrics: pd.DataFrame) -> pd.DataFrame:
    """在 fixture 未提供 split_id 時建立可稽核 identity。"""
    if "split_id" in metrics.columns:
        return metrics.copy()
    if "fold" not in metrics.columns:
        raise ValueError("ranking input 必須提供 split_id 或 fold")
    result = metrics.copy()
    result["split_id"] = result["validation"].astype(str) + ":" + result["fold"].astype(str)
    return result


def _expected_validation_splits(metrics: pd.DataFrame, validation: str) -> set[str]:
    """以所有 model rows 的 union 定義該 validation 預期 outer folds。"""
    family = metrics[metrics["validation"].eq(validation)]
    return set(family["split_id"].astype(str))


def _dummy_validation_mae(metrics: pd.DataFrame, validation: str) -> float:
    """取得單一 validation 的 successful Dummy median fold MAE。"""
    rows = metrics[
        metrics["validation"].eq(validation)
        & metrics["model"].eq("dummy_median")
        & metrics["status"].eq("ok")
    ]
    values = pd.to_numeric(rows["mae"], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _prediction_sd_check(
    predictions: pd.DataFrame,
    model: str,
    validation: str,
) -> tuple[float, float, bool]:
    """計算單一 validation 的 OOF SD 與 5% non-degeneration gate。"""
    rows = predictions[
        predictions["model"].eq(model) & predictions["validation"].eq(validation)
    ]
    observed = pd.to_numeric(rows["observed_ido_score"], errors="coerce").to_numpy(
        dtype=float
    )
    predicted = pd.to_numeric(rows["predicted_ido_score"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        observed.size == 0
        or predicted.size != observed.size
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
    ):
        return np.nan, np.nan, False
    observed_sd = float(np.std(observed))
    prediction_sd = float(np.std(predicted))
    return observed_sd, prediction_sd, prediction_sd >= 0.05 * observed_sd


def _complete_outer_folds_gate(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    failures: pd.DataFrame,
    model: str,
    expected_splits: Mapping[str, set[str]],
) -> bool:
    """拒絕任何 failed、duplicated 或 missing outer fold。"""
    model_rows = metrics[metrics["model"].eq(model)]
    if not failures.empty and "model" in failures.columns:
        if failures["model"].astype(str).eq(model).any():
            return False
    for validation in _RANKING_VALIDATIONS:
        expected = expected_splits[validation]
        rows = model_rows[model_rows["validation"].eq(validation)]
        actual = set(rows["split_id"].astype(str))
        prediction_rows = predictions[
            predictions["model"].eq(model)
            & predictions["validation"].eq(validation)
        ]
        predicted_splits = set(prediction_rows["split_id"].astype(str))
        if (
            not expected
            or actual != expected
            or predicted_splits != expected
            or len(rows) != len(expected)
            or not rows["status"].eq("ok").all()
        ):
            return False
    return True


def _phase_feature_gate(
    model: str,
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> bool:
    """驗證 model feature set identity 與 exact canonical whitelist。"""
    expected_set = "paper_linear_3f" if model == "paper_linear_3f" else "basic_median"
    rows = metrics[metrics["model"].eq(model)]
    if "feature_set" in rows.columns:
        identities = set(rows["feature_set"].dropna().astype(str))
        if identities and identities != {expected_set}:
            return False
    configured = config.get("feature_sets")
    if configured is not None and not isinstance(configured, Mapping):
        return False
    if model == "paper_linear_3f":
        columns = (
            configured[expected_set]
            if configured is not None and expected_set in configured
            else _PAPER_FEATURES
        )
    elif configured is None:
        columns = PRIMARY_FOV_FEATURES
    elif expected_set in configured:
        columns = configured[expected_set]
    else:
        return False
    columns = tuple(str(column) for column in columns)
    expected = _PAPER_FEATURES if model == "paper_linear_3f" else PRIMARY_FOV_FEATURES
    return tuple(columns) == tuple(expected)


def _overall_oof_spearman(predictions: pd.DataFrame, model: str) -> float:
    """計算四 validation raw OOF predictions 的整體 Spearman。"""
    rows = predictions[predictions["model"].eq(model)]
    observed = pd.to_numeric(rows["observed_ido_score"], errors="coerce").to_numpy(
        dtype=float
    )
    predicted = pd.to_numeric(rows["predicted_ido_score"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        observed.size < 2
        or predicted.size != observed.size
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
        or _is_constant(observed)
        or _is_constant(predicted)
    ):
        return np.nan
    return float(spearmanr(observed, predicted).statistic)


def _simplicity_ranks(config: Mapping[str, Any]) -> dict[str, int]:
    """驗證並建立涵蓋七個 candidates 的 deterministic simplicity ranks。"""
    configured = [str(name) for name in config.get("simplicity_order", [])]
    if len(configured) != len(set(configured)):
        raise ValueError("simplicity_order 不可重複")
    order = [name for name in configured if name in _PHASE_CANDIDATES]
    order.extend(name for name in _PHASE_CANDIDATES if name not in order)
    return {name: index for index, name in enumerate(order)}


def _selection_order(ranking: pd.DataFrame, models: Sequence[str]) -> list[str]:
    """依 worst rank、LOBO MAE、Spearman 與 simplicity 排列 tied models。"""
    tied = ranking[ranking["model"].isin(models)].copy()
    tied["_spearman_sort"] = -pd.to_numeric(
        tied["overall_oof_spearman"], errors="coerce"
    ).fillna(-np.inf)
    tied = tied.sort_values(
        [
            "worst_validation_rank",
            "leave_one_b_out_mae",
            "_spearman_sort",
            "simplicity_rank",
            "model",
        ],
        kind="mergesort",
    )
    return tied["model"].astype(str).tolist()


def _safe_final_path(parent: Path, name: str, *, directory: bool) -> Path:
    """拒絕 final output child 經 symlink 或 junction 改寫目的地。"""
    expected = parent / name
    resolved = expected.resolve(strict=False)
    if resolved != expected:
        kind = "directory" if directory else "file"
        raise ValueError(f"final model {kind} 不可透過 symlink 或 junction 逸出")
    return expected


def _assert_unlinked_final_path(path: Path, expected: Path) -> None:
    """建立目錄後再次驗證實際位置未被 link 置換。"""
    if path.resolve(strict=False) != expected:
        raise ValueError("final model directory 不可透過 symlink 或 junction 逸出")


def _stable_hash(value: object) -> str:
    """以 canonical JSON 建立 reproducible SHA-256 hash。"""
    payload = json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_compatible(value: object) -> object:
    """將常見設定型別正規化為 canonical JSON 可序列化值。"""
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        return converted if np.isfinite(converted) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _image_manifest_hash(images: pd.DataFrame) -> str:
    """由穩定 image metadata 建立 fallback manifest hash。"""
    columns = [
        column
        for column in ("image_key", *_MANIFEST_METADATA_COLUMNS)
        if column in images.columns
    ]
    records = images.loc[:, columns].to_dict(orient="records")
    return _stable_hash(records)


def _write_joblib_atomically(bundle: Mapping[str, Any], path: Path) -> None:
    """在 validated model directory 內原子寫入 joblib bundle。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".joblib", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        joblib.dump(dict(bundle), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomically(payload: Mapping[str, Any], path: Path) -> None:
    """在 validated output directory 內原子寫入 UTF-8 JSON metadata。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                _json_compatible(payload),
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_benchmark_metadata(images: pd.DataFrame) -> None:
    """驗證 benchmark 所有 model 共用的 metadata 與 target。"""
    required = {
        "condition",
        "ifn_dose",
        "tnf_dose",
        "IDO_score",
    }
    missing = sorted(required - set(images.columns))
    if missing:
        raise ValueError(f"images 缺少 benchmark 必要欄位：{missing}")


def _validate_unique_split_ids(splits: Sequence[OuterSplit]) -> None:
    """在任何 model fit 前拒絕重複的 validation/fold identity。"""
    seen: set[str] = set()
    for split in splits:
        split_id = f"{split.validation}:{split.fold}"
        if split_id in seen:
            raise ValueError(f"duplicate split_id: {split_id}")
        seen.add(split_id)


def _validate_candidate_feature_sets(
    feature_sets: Mapping[str, Sequence[str]],
    selected_names: Sequence[str],
    registry: Mapping[str, ModelSpec],
) -> None:
    """在任何 split 前阻擋 candidate feature 缺漏或 target/dose leakage。"""
    needs_basic_median = any(
        registry[name].feature_set == "basic_median" for name in selected_names
    )
    if not needs_basic_median:
        return
    if "basic_median" not in feature_sets:
        raise ValueError("basic_median 必須是精確 33 個 PRIMARY phase-only predictors")

    predictors = tuple(str(column) for column in feature_sets["basic_median"])
    try:
        validate_phase_predictors(predictors)
    except ValueError as error:
        raise ValueError(
            "basic_median 必須是精確 33 個 PRIMARY phase-only predictors"
        ) from error
    if predictors != tuple(PRIMARY_FOV_FEATURES):
        raise ValueError("basic_median 必須是精確 33 個 PRIMARY phase-only predictors")


def _validate_benchmark_split(
    split: OuterSplit, row_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """驗證單一 shared outer split 並回傳 positional indices。"""
    train_index = _validated_positions(split.train_index, row_count, "train_index")
    test_index = _validated_positions(split.test_index, row_count, "test_index")
    if train_index.size == 0 or test_index.size == 0:
        raise ValueError("outer split 的 training 與 test 均不可為空")
    if np.intersect1d(train_index, test_index).size:
        raise ValueError("outer split 的 train/test indices 不可重疊")
    covered = np.sort(np.concatenate([train_index, test_index]))
    if not np.array_equal(covered, np.arange(row_count)):
        raise ValueError("outer split 必須涵蓋所有 image rows")
    return train_index, test_index


def _model_feature_frame(
    frame: pd.DataFrame,
    spec: ModelSpec,
    feature_sets: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, list[str]]:
    """以單一 fold frame 建立 model predictors 與 deterministic interaction。"""
    if spec.name == "dummy_median":
        return pd.DataFrame({"dummy": np.zeros(len(frame))}), ["dummy"]
    if spec.name == "paper_linear_3f":
        columns = list(_PAPER_FEATURES)
    elif spec.name == "dose_ridge":
        columns = ["ifn_dose", "tnf_dose"]
    elif spec.name == "dose_plus_morphology_ridge":
        columns = ["ifn_dose", "tnf_dose", *PRIMARY_FOV_FEATURES]
    else:
        if spec.feature_set not in feature_sets:
            raise ValueError(f"feature_sets 缺少 {spec.feature_set!r}")
        columns = list(feature_sets[spec.feature_set])

    if len(columns) != len(set(columns)):
        raise ValueError(f"{spec.name} predictor columns 不可重複")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.name} 缺少 predictor columns：{missing}")
    predictors = frame.loc[:, columns].copy()
    if spec.name in {"dose_ridge", "dose_plus_morphology_ridge"}:
        predictors.insert(
            2,
            "ifn_x_tnf",
            predictors["ifn_dose"].to_numpy(dtype=float)
            * predictors["tnf_dose"].to_numpy(dtype=float),
        )
    return predictors, predictors.columns.tolist()


def _fit_outer_model(
    estimator: RegressorMixin,
    parameters: Mapping[str, Sequence[Any]],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    training: pd.DataFrame,
    requested_inner_splits: int,
    candidate_cap: int,
    seed: int,
    n_jobs: int,
) -> tuple[RegressorMixin, dict[str, Any], float]:
    """Fit 無 tuning estimator 或執行一次 deterministic RandomizedSearchCV。"""
    if not parameters:
        estimator.fit(x_train, y_train)
        return estimator, {}, np.nan

    cv = make_inner_splits(training, requested_inner_splits)
    total_combinations = len(ParameterGrid(parameters))
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters,
        n_iter=min(candidate_cap, total_combinations),
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        cv=cv,
        refit=True,
        random_state=seed,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, dict(search.best_params_), float(-search.best_score_)


def _finite_predictions(values: Any, expected_length: int) -> np.ndarray:
    """正規化 predictions，並拒絕形狀錯誤或 nonfinite 結果。"""
    predicted = np.asarray(values, dtype=float)
    if predicted.ndim != 1 or predicted.size != expected_length:
        raise ValueError("predicted 必須是一維且與 outer test 長度相同")
    if not np.isfinite(predicted).all():
        raise ValueError("predicted 必須全部為有限值")
    return predicted


def _fold_feature_importance(
    fitted: RegressorMixin,
    spec: ModelSpec,
    feature_names: Sequence[str],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    split: OuterSplit,
    split_id: str,
    repeats: int,
    seed: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    """建立不回饋 tuning/ranking 的 fold-level diagnostic importance。"""
    if spec.name in _LINEAR_MODELS:
        pipeline = fitted.regressor_
        coefficients = np.asarray(
            pipeline.named_steps["model"].coef_, dtype=float
        ).reshape(-1)
        if coefficients.size != len(feature_names):
            raise ValueError("linear coefficient 數量與 predictor columns 不一致")
        means = coefficients
        deviations = np.full(coefficients.size, np.nan)
        importance_type = "standardized_coefficient"
    elif spec.name in _PERMUTATION_MODELS:
        diagnostic = permutation_importance(
            fitted,
            x_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=repeats,
            random_state=seed,
            n_jobs=n_jobs,
        )
        means = diagnostic.importances_mean
        deviations = diagnostic.importances_std
        importance_type = "outer_test_permutation_diagnostic"
    else:
        return []

    return [
        {
            "validation": split.validation,
            "fold": split.fold,
            "split_id": split_id,
            "model": spec.name,
            "role": spec.role,
            "feature_set": spec.feature_set,
            "feature": feature,
            "importance_type": importance_type,
            "importance": float(importance),
            "importance_sd": float(importance_sd),
        }
        for feature, importance, importance_sd in zip(
            feature_names, means, deviations, strict=True
        )
    ]


def _metric_identity(
    split: OuterSplit,
    split_id: str,
    spec: ModelSpec,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    """建立成功與失敗 fold metric 共用的 identity columns。"""
    return {
        "validation": split.validation,
        "fold": split.fold,
        "split_id": split_id,
        "model": spec.name,
        "role": spec.role,
        "feature_set": spec.feature_set,
        "n_train": n_train,
        "n_test": n_test,
    }


def _prediction_records(
    testing: pd.DataFrame,
    predicted: np.ndarray,
    split: OuterSplit,
    split_id: str,
    spec: ModelSpec,
) -> list[dict[str, Any]]:
    """將成功 outer-test predictions 轉為 exact OOF schema records。"""
    records: list[dict[str, Any]] = []
    for row, prediction in zip(testing.itertuples(index=False), predicted, strict=True):
        records.append(
            {
                "validation": split.validation,
                "fold": split.fold,
                "split_id": split_id,
                "model": spec.name,
                "role": spec.role,
                "feature_set": spec.feature_set,
                "image_key": row.image_key,
                "b_id": row.b_id,
                "passage": row.passage,
                "group_id": row.group_id,
                "condition_index": row.condition_index,
                "condition": row.condition,
                "observed_ido_score": float(row.IDO_score),
                "predicted_ido_score": float(prediction),
            }
        )
    return records


def _safe_standard_deviation(values: pd.Series) -> float:
    """Failure record 僅在 target 可完整轉成有限值時保留 observed SD。"""
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.nan
    if vector.size == 0 or not np.isfinite(vector).all():
        return np.nan
    return float(np.std(vector))


def _random_seed(value: Any) -> int:
    """驗證 sklearn random_state 可接受的 deterministic seed。"""
    try:
        seed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("seed 必須是 0 到 2**32 - 1 的整數") from error
    if seed != value or not 0 <= seed < 2**32:
        raise ValueError("seed 必須是 0 到 2**32 - 1 的整數")
    return seed


def _validate_images(images: pd.DataFrame) -> None:
    """驗證 outer splitting 與 manifest 所需的 image metadata。"""
    missing = sorted(set(_REQUIRED_IMAGE_COLUMNS) - set(images.columns))
    if missing:
        raise ValueError(f"images 缺少必要欄位：{missing}")
    if images.empty:
        raise ValueError("images 不可為空")
    _validate_nonempty_values(images, _REQUIRED_IMAGE_COLUMNS)
    if images["image_key"].duplicated().any():
        raise ValueError("image_key 不可重複")


def _validate_nonempty_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """拒絕分組欄位中的 null 或空白值。"""
    for column in columns:
        values = frame[column]
        blank = values.astype("string").str.strip().eq("").fillna(False)
        if values.isna().any() or blank.any():
            raise ValueError(f"{column} 不可含空值或空白值")


def _sorted_unique(values: pd.Series) -> list[object]:
    """以自然順序回傳不重複分組值。"""
    try:
        return sorted(values.unique().tolist())
    except TypeError as error:
        raise ValueError(f"{values.name} 的分組值必須可排序") from error


def _validate_split_family(
    splits: Sequence[OuterSplit],
    row_count: int,
    validation: str,
) -> None:
    """驗證單一 outer family 的 disjointness 與完整 test coverage。"""
    if not splits:
        raise ValueError(f"{validation} 未產生任何 outer split")
    tested: list[np.ndarray] = []
    expected = np.arange(row_count, dtype=np.int64)
    for split in splits:
        train_index = _validated_positions(split.train_index, row_count, "train_index")
        test_index = _validated_positions(split.test_index, row_count, "test_index")
        if train_index.size == 0 or test_index.size == 0:
            raise ValueError(f"{validation} 的 training 與 test 都不可為空")
        if np.intersect1d(train_index, test_index).size:
            raise ValueError(f"{validation} 的 train/test indices 不可重疊")
        covered = np.sort(np.concatenate([train_index, test_index]))
        if not np.array_equal(covered, expected):
            raise ValueError(f"{validation} 的 split 未完整覆蓋全部 image rows")
        tested.append(test_index)
    test_counts = np.bincount(np.concatenate(tested), minlength=row_count)
    if not np.all(test_counts == 1):
        raise ValueError(f"{validation} 中每張影像必須恰為一次 test")


def _validated_positions(
    values: np.ndarray,
    row_count: int,
    name: str,
) -> np.ndarray:
    """驗證一維、唯一且未越界的 positional indices。"""
    positions = np.asarray(values)
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        raise ValueError(f"{name} 必須是一維整數 positional indices")
    positions = positions.astype(np.int64, copy=False)
    if positions.size != np.unique(positions).size:
        raise ValueError(f"{name} 不可含重複 positional index")
    if np.any(positions < 0) or np.any(positions >= row_count):
        raise ValueError(f"{name} 含有越界 positional index")
    return positions


def _validate_inner_splits(
    training: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    """驗證 inner folds 完整覆蓋且沒有 group leakage。"""
    tested: list[np.ndarray] = []
    groups = training["group_id"]
    for train_index, test_index in splits:
        if train_index.size == 0 or test_index.size == 0:
            raise ValueError("inner fold 的 training 與 test 都不可為空")
        if set(groups.iloc[train_index]).intersection(groups.iloc[test_index]):
            raise ValueError("inner GroupKFold 發生 group_id leakage")
        tested.append(test_index)
    tested_positions = np.sort(np.concatenate(tested))
    if not np.array_equal(tested_positions, np.arange(len(training))):
        raise ValueError("inner GroupKFold 未完整且恰一次覆蓋 training rows")


def _metric_vector(values: Sequence[float], name: str) -> np.ndarray:
    """將 metric 輸入正規化為一維 float vector。"""
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是數值 vector") from error
    if vector.ndim != 1:
        raise ValueError(f"{name} 必須是一維 vector")
    return vector


def _is_constant(values: np.ndarray) -> bool:
    """判斷非空 vector 是否全部為相同值。"""
    return bool(np.all(values == values[0]))


__all__ = [
    "BenchmarkResult",
    "FAILURE_COLUMNS",
    "FEATURE_IMPORTANCE_COLUMNS",
    "FOLD_METRIC_COLUMNS",
    "HYPERPARAMETER_COLUMNS",
    "ModelSpec",
    "OOF_COLUMNS",
    "OuterSplit",
    "PARAMETERS",
    "build_model_registry",
    "condition_adjust_targets",
    "fit_final_phase_model",
    "make_inner_splits",
    "make_outer_splits",
    "outer_split_manifest",
    "regression_metrics",
    "rank_phase_models",
    "run_condition_adjusted_sensitivity",
    "run_nested_benchmark",
    "select_winner",
]
