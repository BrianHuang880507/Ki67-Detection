"""提供 Exp3 可稽核的分組驗證切分與 regression metrics。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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

            prediction_rows.extend(
                _prediction_records(testing, predicted, split, split_id, spec)
            )
            metric_rows.append(
                {
                    **metric_base,
                    **metrics,
                    "observed_sd": float(np.std(y_test)),
                    "prediction_sd": float(np.std(predicted)),
                    "status": "success",
                }
            )
            hyperparameter_rows.append(
                {
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
            )
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
    "make_inner_splits",
    "make_outer_splits",
    "outer_split_manifest",
    "regression_metrics",
    "run_nested_benchmark",
]
