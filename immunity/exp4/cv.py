"""Exp4 v5 四 arms、leakage preflight 與 nested cell-level CV 核心。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .cell_dedup import NUCLEUS_FEATURE_COLUMNS
from .rui_features import MORPHOLOGY_FEATURE_NAMES, RUI49_FEATURE_COLUMNS


RANDOM_STATE = 42
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
EXPECTED_CONFIGURATION_COUNT = 24
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
EXPECTED_PHASE2_GROUP_IDS = tuple(
    f"{b_id}_P{passage}_C{condition_index:02d}"
    for b_id in ("B4", "B7", "B8")
    for passage in (5, 6, 7)
    for condition_index in range(1, 9)
)
EXPECTED_MODEL_NAMES = ("SVR_L", "SVR", "LASSO", "RFR", "GBR", "MLPR")
EXPECTED_ARM_NAMES = (
    "geometry_24",
    "rui_48",
    "rui_filtered",
    "rui_48_plus_nucleus",
)
RUI_MODEL_ANCHORS: Mapping[str, Mapping[str, object]] = {
    "SVR_L": {"kernel": "linear", "C": 0.1, "epsilon": 1.0},
    "SVR": {"kernel": "rbf", "C": 5.0, "epsilon": 0.5},
    "LASSO": {"alpha": 0.8172727272727273},
    "RFR": {"n_estimators": 117, "max_depth": 7},
    "GBR": {"n_estimators": 29, "learning_rate": 0.2, "max_depth": 3},
    "MLPR": {
        "activation": "relu",
        "alpha": 1.0,
        "hidden_layer_sizes": (100,),
        "learning_rate": "constant",
        "max_iter": 200,
        "solver": "adam",
    },
}
RUI48_FEATURE_COLUMNS = tuple(
    feature
    for feature in RUI49_FEATURE_COLUMNS
    if feature != "cell__MinIntensity"
)
GEOMETRY24_FEATURE_COLUMNS = tuple(
    f"cell__{name}" for name in MORPHOLOGY_FEATURE_NAMES
)

LEAKAGE_REPORT_COLUMNS = ("check_id", "check_name", "status", "details")
CV_METRIC_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "outer_fold",
    "n_train",
    "n_test",
    "test_group_count",
    "rui_r2",
    "cell_r2",
    "r2_gap_rui_minus_cell",
    "cell_mae",
    "cell_rmse",
    "cell_spearman",
    "cell_spearman_status",
    "best_inner_r2",
    "grid_candidate_count",
    "fit_seconds",
    "checkpoint_status",
)
OOF_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "outer_fold",
    "image_key",
    "cell_label",
    "group_id",
    "confidence_flag",
    "observed_ido_score",
    "predicted_ido_score",
    "residual",
)
HYPERPARAMETER_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "outer_fold",
    "best_params_json",
    "best_inner_r2",
    "grid_candidate_count",
)
PER_GROUP_RESIDUAL_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "outer_fold",
    "group_id",
    "confidence_flag",
    "n_cells",
    "observed_target",
    "mean_prediction",
    "mean_residual",
    "cell_mae",
    "cell_rmse",
)
FEATURE_IMPORTANCE_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "outer_fold",
    "feature",
    "importance_method",
    "importance_mean",
    "importance_std",
    "absolute_importance",
    "rank",
    "is_spatial_feature",
)
CHECKPOINT_MANIFEST_VERSION = 2
SPATIAL_FEATURES = {
    "cell__Center_X",
    "cell__Center_Y",
    "cell__Orientation",
}


class CvContractError(ValueError):
    """表示 formal population、arm 或模型 contract 不成立。"""


class LeakagePreflightError(CvContractError):
    """表示 7 項 leakage preflight 至少一項失敗。"""

    def __init__(self, message: str, report: pd.DataFrame) -> None:
        super().__init__(message)
        self.report = report.copy()


class CheckpointFingerprintError(CvContractError):
    """表示 checkpoint 與本次 data/grid/split/source fingerprints 不一致。"""


class CvExecutionError(RuntimeError):
    """表示 outer fold training、prediction、metric 或 checkpoint 無法完成。"""


@dataclass(frozen=True)
class CvPopulationContract:
    """鎖定 CV population scope，避免 unit fixture 被誤發布成正式結果。

    Attributes:
        scope: ``formal_19648_cells_693_fovs`` 或明示的 ``unit_fixture``。
        expected_cell_count: 必須恰好符合的 unique whole-cell 數。
        expected_fov_count: 必須恰好符合的 FOV 數。
        expected_group_ids: 每個 outer test fold 都必須完整出現的 groups。
        formal: 是否要求正式 Rui estimator roster 與 production publish contract。
    """

    scope: str
    expected_cell_count: int
    expected_fov_count: int
    expected_group_ids: tuple[str, ...]
    formal: bool

    @classmethod
    def formal_exp4(cls) -> "CvPopulationContract":
        """建立不可降級的正式 Exp4 population contract。"""
        return cls(
            scope="formal_19648_cells_693_fovs",
            expected_cell_count=19_648,
            expected_fov_count=693,
            expected_group_ids=EXPECTED_GROUP_IDS,
            formal=True,
        )

    @classmethod
    def formal_exp4_phase2(cls) -> "CvPopulationContract":
        """建立不可降級的 19,648-cell／693-FOV／72-group contract。"""
        return cls(
            scope="formal_phase2_19648_cells_693_fovs_72_groups",
            expected_cell_count=19_648,
            expected_fov_count=693,
            expected_group_ids=EXPECTED_PHASE2_GROUP_IDS,
            formal=True,
        )

    @classmethod
    def fixture(
        cls,
        *,
        expected_cell_count: int,
        expected_fov_count: int,
        expected_group_ids: Sequence[str],
    ) -> "CvPopulationContract":
        """建立只能供 tests 使用、不得發布為 formal 的明示 fixture scope。"""
        return cls(
            scope="unit_fixture_not_publishable",
            expected_cell_count=int(expected_cell_count),
            expected_fov_count=int(expected_fov_count),
            expected_group_ids=tuple(str(value) for value in expected_group_ids),
            formal=False,
        )


@dataclass(frozen=True)
class LeakageContext:
    """保存無法單由 feature names 推得的 leakage provenance。

    Attributes:
        standardization_scope: 必須聲明 scaler 在 outer training Pipeline 內 fit。
        feature_selection_inputs: 必須恰為 ``("X",)``。
        cellpose_invoked: 本輪是否呼叫 Cellpose；正式值必須為 false。
        mask_cache_fingerprint_before: CV 前 mask-cache fingerprint。
        mask_cache_fingerprint_after: CV 前再次唯讀取得的 fingerprint，必須相同。
    """

    standardization_scope: str
    feature_selection_inputs: tuple[str, ...]
    cellpose_invoked: bool
    mask_cache_fingerprint_before: str
    mask_cache_fingerprint_after: str


@dataclass(frozen=True)
class FeatureArmAssembly:
    """保存同一 whole-cell population 上的四個 exact predictor rosters。"""

    cells: pd.DataFrame
    arm_features: tuple[tuple[str, tuple[str, ...]], ...]
    contract: CvPopulationContract

    def as_mapping(self) -> OrderedDict[str, tuple[str, ...]]:
        """依規格順序回傳四個 arm rosters。"""
        return OrderedDict(self.arm_features)


@dataclass(frozen=True)
class LeakagePreflightResult:
    """保存 7 項 fail-closed leakage evidence。"""

    passed: bool
    report: pd.DataFrame


@dataclass(frozen=True)
class ModelSpec:
    """描述單一 sklearn regressor prototype 與 Pipeline parameter grid。"""

    name: str
    estimator: BaseEstimator
    param_grid: Mapping[str, Sequence[object]]


@dataclass(frozen=True)
class CvRunResult:
    """保存 nested CV 所有可發布表與 checkpoint provenance。

    Attributes:
        metrics: 4 arms × 6 models × 5 outer folds 的兩種 R² 與 cell metrics。
        oof_predictions: 每顆細胞在 24 configurations 各一筆 OOF prediction。
        hyperparameters: 每個 outer fold 的最佳 GridSearchCV 參數。
        per_group_residuals: 每 configuration × 9 groups 的 pooled OOF 殘差。
        feature_importance: fold-level feature importance、method 與 absolute rank。
        preflight: 七項全部通過的 leakage evidence。
        checkpoint_manifest: data/grid/split/source 及 combined fingerprints。
        hyperparameter_grids: 可直接 JSON serialization 的實際 grids。
    """

    metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    hyperparameters: pd.DataFrame
    per_group_residuals: pd.DataFrame
    feature_importance: pd.DataFrame
    preflight: LeakagePreflightResult
    checkpoint_manifest: Mapping[str, object]
    hyperparameter_grids: Mapping[str, object]


def assemble_feature_arms(
    retained_cells: pd.DataFrame,
    *,
    filtered_features: Sequence[str],
    contract: CvPopulationContract | None = None,
) -> FeatureArmAssembly:
    """在同一 unique whole-cell population 建立四個 exact feature arms。

    Args:
        retained_cells: 已去重、已觸邊排除且含 48 whole-cell 與 17 nucleus 欄。
        filtered_features: X-only Pearson earlier-wins 後的 canonical subsequence。
        contract: 省略時使用正式 19,648 cells／693 FOV contract。

    Returns:
        含不可混用的四個 predictor rosters 與同一列順序資料。

    Raises:
        TypeError: retained_cells 不是 DataFrame 時拋出。
        CvContractError: population、欄位、順序、有限值或 arm isolation 不合法。
    """
    if not isinstance(retained_cells, pd.DataFrame):
        raise TypeError("retained_cells 必須是 pandas DataFrame")
    active_contract = contract or CvPopulationContract.formal_exp4()
    _validate_population_contract(active_contract)
    _validate_population(retained_cells, active_contract)
    filtered = tuple(str(feature) for feature in filtered_features)
    if not filtered or len(set(filtered)) != len(filtered):
        raise CvContractError("rui_filtered 必須是非空且唯一的 feature roster")
    filtered_set = set(filtered)
    if not filtered_set.issubset(RUI48_FEATURE_COLUMNS):
        raise CvContractError(
            "rui_filtered 只能包含 canonical rui_48 whole-cell features"
        )
    canonical_subsequence = tuple(
        feature for feature in RUI48_FEATURE_COLUMNS if feature in filtered_set
    )
    if filtered != canonical_subsequence:
        raise CvContractError("rui_filtered 必須保留 rui_48 canonical earlier-wins 順序")
    if active_contract.formal and len(filtered) != 30:
        raise CvContractError(
            f"formal rui_filtered expected 30 features, actual {len(filtered)}"
        )

    arm_features = (
        ("geometry_24", GEOMETRY24_FEATURE_COLUMNS),
        ("rui_48", RUI48_FEATURE_COLUMNS),
        ("rui_filtered", filtered),
        (
            "rui_48_plus_nucleus",
            (*RUI48_FEATURE_COLUMNS, *NUCLEUS_FEATURE_COLUMNS),
        ),
    )
    required = set().union(*(set(features) for _, features in arm_features))
    missing = sorted(required - set(retained_cells.columns))
    if missing:
        raise CvContractError(f"retained_cells 缺少 arm predictors：{missing}")
    try:
        numeric = retained_cells.loc[:, sorted(required)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CvContractError("所有 arm predictors 必須是數值") from error
    if not np.isfinite(numeric).all():
        raise CvContractError("所有 arm predictors 必須全為 finite")
    if len(RUI48_FEATURE_COLUMNS) != 48 or len(GEOMETRY24_FEATURE_COLUMNS) != 24:
        raise CvContractError("canonical geometry/rui feature counts 已漂移")
    if len(NUCLEUS_FEATURE_COLUMNS) != 17:
        raise CvContractError("canonical nucleus feature count 已漂移")
    return FeatureArmAssembly(
        cells=retained_cells.copy(),
        arm_features=arm_features,
        contract=active_contract,
    )


def run_leakage_preflight(
    assembly: FeatureArmAssembly,
    context: LeakageContext,
) -> LeakagePreflightResult:
    """執行計畫要求的 7 項 fail-closed leakage 檢查。

    Args:
        assembly: 已完成 exact arm assembly 的同一 population。
        context: scaler、X-only selection、Cellpose 與 mask immutable provenance。

    Returns:
        僅在七項全數通過時回傳 evidence table。

    Raises:
        LeakagePreflightError: 任一檢查失敗時附完整七項 report 拋出。
    """
    arms = assembly.as_mapping()
    all_features = tuple(feature for features in arms.values() for feature in features)
    lower_features = tuple(feature.lower() for feature in all_features)
    forbidden_exact = {
        "b_id",
        "passage",
        "condition",
        "ifn_dose",
        "tnf_dose",
        "image_key",
        "cell_label",
        "group_id",
        "confidence_flag",
    }
    metadata_hits = sorted(
        {
            feature
            for feature in all_features
            if feature.lower() in forbidden_exact
            or "path" in feature.lower()
            or "batch" in feature.lower()
            or "file" in feature.lower()
        }
    )
    ido_hits = sorted(
        {feature for feature in all_features if "ido" in feature.lower()}
    )
    dapi_hits = sorted(
        {feature for feature in all_features if "dapi" in feature.lower()}
    )
    nucleus_set = set(NUCLEUS_FEATURE_COLUMNS)
    isolated = (
        all(
            not set(arms[name]).intersection(nucleus_set)
            for name in ("geometry_24", "rui_48", "rui_filtered")
        )
        and set(arms["rui_48_plus_nucleus"]).intersection(nucleus_set)
        == nucleus_set
        and len(arms["rui_48_plus_nucleus"]) == 65
    )
    before = str(context.mask_cache_fingerprint_before).strip()
    after = str(context.mask_cache_fingerprint_after).strip()
    immutable = (
        not context.cellpose_invoked
        and re.fullmatch(r"[0-9a-fA-F]{64}", before) is not None
        and re.fullmatch(r"[0-9a-fA-F]{64}", after) is not None
        and before == after
    )
    checks = (
        (
            1,
            "scaler_fit_only_within_outer_training_pipeline",
            context.standardization_scope
            == "pipeline_fit_within_outer_training",
            context.standardization_scope,
        ),
        (
            2,
            "feature_selection_uses_x_only",
            tuple(context.feature_selection_inputs) == ("X",),
            repr(tuple(context.feature_selection_inputs)),
        ),
        (3, "metadata_path_batch_predictors_excluded", not metadata_hits, repr(metadata_hits)),
        (4, "ido_predictors_excluded", not ido_hits, repr(ido_hits)),
        (5, "dapi_predictors_excluded", not dapi_hits, repr(dapi_hits)),
        (6, "nucleus_features_isolated_to_arm4", isolated, repr(dict(arms))),
        (
            7,
            "cellpose_not_run_and_mask_cache_immutable",
            immutable,
            (
                f"cellpose_invoked={context.cellpose_invoked};"
                f"before={before};after={after}"
            ),
        ),
    )
    report = pd.DataFrame(
        [
            {
                "check_id": check_id,
                "check_name": name,
                "status": "passed" if passed else "failed",
                "details": details,
            }
            for check_id, name, passed, details in checks
        ],
        columns=LEAKAGE_REPORT_COLUMNS,
    )
    if report["status"].ne("passed").any():
        failed = report.loc[report["status"].eq("failed"), "check_name"].tolist()
        raise LeakagePreflightError(
            f"leakage preflight failed closed: {failed}",
            report,
        )
    return LeakagePreflightResult(passed=True, report=report)


def default_model_specs() -> tuple[ModelSpec, ...]:
    """建立含 Rui anchors、明確 defaults 與固定 seed 的正式六模型規格。

    Returns:
        固定順序 ``SVR_L/SVR/LASSO/RFR/GBR/MLPR`` 的 prototypes 與 grids。
        GridSearchCV 使用時所有 grid keys 已帶 ``regressor__`` prefix。
    """
    return (
        ModelSpec(
            name="SVR_L",
            estimator=SVR(kernel="linear"),
            param_grid={
                "regressor__C": (0.1, 1.0),
                "regressor__epsilon": (0.5, 1.0),
            },
        ),
        ModelSpec(
            name="SVR",
            estimator=SVR(kernel="rbf", gamma="scale"),
            param_grid={
                "regressor__C": (1.0, 5.0),
                "regressor__epsilon": (0.1, 0.5),
            },
        ),
        ModelSpec(
            name="LASSO",
            estimator=Lasso(
                random_state=RANDOM_STATE,
                max_iter=10_000,
                selection="cyclic",
            ),
            param_grid={
                "regressor__alpha": (0.1, 0.8172727272727273, 1.0),
            },
        ),
        ModelSpec(
            name="RFR",
            estimator=RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
                min_samples_leaf=1,
            ),
            param_grid={
                "regressor__n_estimators": (50, 117),
                "regressor__max_depth": (7, None),
            },
        ),
        ModelSpec(
            name="GBR",
            estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
            param_grid={
                "regressor__n_estimators": (29, 100),
                "regressor__learning_rate": (0.05, 0.2),
                "regressor__max_depth": (2, 3),
            },
        ),
        ModelSpec(
            name="MLPR",
            estimator=MLPRegressor(
                activation="relu",
                learning_rate="constant",
                max_iter=200,
                random_state=RANDOM_STATE,
                solver="adam",
            ),
            param_grid={
                "regressor__alpha": (0.1, 1.0),
                "regressor__hidden_layer_sizes": ((50,), (100,)),
            },
        ),
    )


def build_training_pipeline(spec: ModelSpec) -> Pipeline:
    """建立尚未 fit 的 scaler→regressor Pipeline。

    Args:
        spec: 單一模型 prototype；regressor 會先 clone，避免跨 fold state 共用。

    Returns:
        step 順序固定為 ``StandardScaler``、``regressor`` 的未 fit Pipeline。

    Raises:
        TypeError: spec 或 estimator 型別不合法時拋出。
    """
    if not isinstance(spec, ModelSpec) or not isinstance(spec.estimator, BaseEstimator):
        raise TypeError("spec 必須含 sklearn BaseEstimator")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", clone(spec.estimator)),
        ]
    )


def run_rui_cv(
    assembly: FeatureArmAssembly,
    leakage_context: LeakageContext,
    *,
    checkpoint_dir: str | Path,
    source_fingerprint: str,
    model_specs: Sequence[ModelSpec] | None = None,
    n_jobs: int = -1,
    importance_repeats: int = 3,
) -> CvRunResult:
    """執行六模型 × 四 arms 的 outer 5-fold nested CV。

    Outer splits 固定為 ``KFold(5, shuffle=True, random_state=42)`` 且由所有
    24 configurations 共用。每個 outer train 內另建 5-fold GridSearchCV；
    ``StandardScaler`` 位於 Pipeline 內，因此不會看到 outer test。每個 fold
    產生 Rui group-averaged R²、標準 cell R²/MAE/RMSE/Spearman、OOF、
    per-group residual 與明示 method/rank 的 importance。

    Args:
        assembly: 同一 population 的四個 exact arms。
        leakage_context: 七項 leakage provenance。
        checkpoint_dir: fold checkpoints 與 fingerprint manifest 的獨立目錄。
        source_fingerprint: caller 對核心 source/provenance 計算的 SHA256。
        model_specs: 省略時使用正式六模型 grids；fixture scope 可注入同名快速模型。
        n_jobs: GridSearchCV 的 job 數；正式預設使用所有可用 cores。
        importance_repeats: 無 native importance 模型的 test-fold permutation 次數。

    Returns:
        完整且已通過 OOF/shape/group gates 的 ``CvRunResult``。

    Raises:
        CvContractError: population、models、fold groups 或參數不合法時拋出。
        LeakagePreflightError: 任一 leakage preflight 失敗時拋出。
        CheckpointFingerprintError: checkpoint provenance 漂移或內容損毀時拋出。
        CvExecutionError: 任一 fold training/prediction/metric 失敗時拋出。
    """
    if not isinstance(assembly, FeatureArmAssembly):
        raise TypeError("assembly 必須是 FeatureArmAssembly")
    _validate_population_contract(assembly.contract)
    _validate_population(assembly.cells, assembly.contract)
    preflight = run_leakage_preflight(assembly, leakage_context)
    specs = tuple(model_specs) if model_specs is not None else default_model_specs()
    _validate_model_specs(specs, formal=assembly.contract.formal)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise CvContractError("n_jobs 必須是整數")
    if int(n_jobs) == 0:
        raise CvContractError("n_jobs 不可為 0")
    if isinstance(importance_repeats, bool) or int(importance_repeats) <= 0:
        raise CvContractError("importance_repeats 必須是正整數")
    source_hash = str(source_fingerprint).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise CvContractError("source_fingerprint 必須是 64-char SHA256 hex")

    cells = assembly.cells.reset_index(drop=True).copy()
    groups = cells["group_id"].astype(str).to_numpy()
    confidence_flags = _validated_confidence_flags(cells, assembly.contract)
    y = pd.to_numeric(cells["group_IDO_score"], errors="raise").to_numpy(
        dtype=np.float64
    )
    splits = _outer_splits(groups, assembly.contract)
    grid_payload = _model_grids_payload(specs)
    fingerprints = {
        "data_fingerprint": _data_fingerprint(assembly),
        "arms_fingerprint": _sha256_json(assembly.as_mapping()),
        "grid_fingerprint": _sha256_json(grid_payload),
        "split_fingerprint": _split_fingerprint(splits),
        "outer_test_counts": [int(len(test)) for _, test in splits],
        "outer_test_index_fingerprints": {
            str(fold_index): hashlib.sha256(
                np.asarray(test, dtype=np.int64).tobytes(order="C")
            ).hexdigest()
            for fold_index, (_, test) in enumerate(splits, start=1)
        },
        "source_fingerprint": source_hash,
        "sklearn_version": str(sklearn.__version__),
        "importance_repeats": int(importance_repeats),
    }
    fingerprints["run_fingerprint"] = _sha256_json(fingerprints)
    checkpoint_manifest = _prepare_checkpoint_manifest(
        Path(checkpoint_dir),
        fingerprints=fingerprints,
        contract=assembly.contract,
    )

    metrics_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    hyperparameter_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    arms = assembly.as_mapping()
    checkpoint_root = Path(checkpoint_dir)
    for arm, features in arms.items():
        x = cells.loc[:, features].to_numpy(dtype=np.float64)
        for spec in specs:
            configuration_id = f"{arm}__{spec.name}"
            candidate_count = len(ParameterGrid(dict(spec.param_grid)))
            for fold_index, (train_indices, test_indices) in enumerate(splits, start=1):
                checkpoint_path = (
                    checkpoint_root
                    / f"{configuration_id}__fold_{fold_index:02d}.npz"
                )
                try:
                    checkpoint = _load_fold_checkpoint(
                        checkpoint_path,
                        run_fingerprint=str(fingerprints["run_fingerprint"]),
                        configuration_id=configuration_id,
                        outer_fold=fold_index,
                        expected_test_indices=test_indices,
                        expected_feature_count=len(features),
                    )
                    if checkpoint is None:
                        checkpoint = _fit_outer_fold(
                            x=x,
                            y=y,
                            train_indices=train_indices,
                            test_indices=test_indices,
                            spec=spec,
                            outer_fold=fold_index,
                            n_jobs=int(n_jobs),
                            importance_repeats=int(importance_repeats),
                        )
                        _write_fold_checkpoint(
                            checkpoint_path,
                            checkpoint,
                            run_fingerprint=str(fingerprints["run_fingerprint"]),
                            configuration_id=configuration_id,
                            outer_fold=fold_index,
                            test_indices=test_indices,
                        )
                        checkpoint_status = "computed"
                    else:
                        checkpoint_status = "resumed"
                    predictions = np.asarray(
                        checkpoint["predictions"], dtype=np.float64
                    )
                    fold_metrics = _fold_metrics(
                        y_true=y[test_indices],
                        y_pred=predictions,
                        groups=groups[test_indices],
                        expected_group_ids=assembly.contract.expected_group_ids,
                    )
                except CheckpointFingerprintError:
                    raise
                except Exception as error:
                    raise CvExecutionError(
                        f"CV fold failed: {configuration_id}, fold={fold_index}: {error}"
                    ) from error

                metrics_rows.append(
                    {
                        "configuration_id": configuration_id,
                        "arm": arm,
                        "model": spec.name,
                        "outer_fold": fold_index,
                        "n_train": int(len(train_indices)),
                        "n_test": int(len(test_indices)),
                        "test_group_count": int(
                            np.unique(groups[test_indices]).size
                        ),
                        **fold_metrics,
                        "best_inner_r2": float(checkpoint["best_inner_score"]),
                        "grid_candidate_count": candidate_count,
                        "fit_seconds": float(checkpoint["fit_seconds"]),
                        "checkpoint_status": checkpoint_status,
                    }
                )
                best_params_json = str(checkpoint["best_params_json"])
                hyperparameter_rows.append(
                    {
                        "configuration_id": configuration_id,
                        "arm": arm,
                        "model": spec.name,
                        "outer_fold": fold_index,
                        "best_params_json": best_params_json,
                        "best_inner_r2": float(checkpoint["best_inner_score"]),
                        "grid_candidate_count": candidate_count,
                    }
                )
                for position, row_index in enumerate(test_indices):
                    observed = float(y[row_index])
                    predicted = float(predictions[position])
                    group_id = str(groups[row_index])
                    oof_rows.append(
                        {
                            "_row_index": int(row_index),
                            "configuration_id": configuration_id,
                            "arm": arm,
                            "model": spec.name,
                            "outer_fold": fold_index,
                            "image_key": cells.at[row_index, "image_key"],
                            "cell_label": cells.at[row_index, "cell_label"],
                            "group_id": group_id,
                            "confidence_flag": str(confidence_flags[row_index]),
                            "observed_ido_score": observed,
                            "predicted_ido_score": predicted,
                            "residual": predicted - observed,
                        }
                    )
                importance_rows.extend(
                    _importance_report_rows(
                        configuration_id=configuration_id,
                        arm=arm,
                        model=spec.name,
                        outer_fold=fold_index,
                        features=features,
                        method=str(checkpoint["importance_method"]),
                        means=np.asarray(
                            checkpoint["importance_means"], dtype=np.float64
                        ),
                        stds=np.asarray(
                            checkpoint["importance_stds"], dtype=np.float64
                        ),
                    )
                )

    metrics = pd.DataFrame(metrics_rows, columns=CV_METRIC_COLUMNS)
    oof = pd.DataFrame(oof_rows)
    oof = oof.sort_values(
        ["configuration_id", "_row_index"], kind="stable"
    ).drop(columns="_row_index")
    oof = oof.loc[:, OOF_COLUMNS].reset_index(drop=True)
    hyperparameters = pd.DataFrame(
        hyperparameter_rows, columns=HYPERPARAMETER_COLUMNS
    )
    feature_importance = pd.DataFrame(
        importance_rows, columns=FEATURE_IMPORTANCE_COLUMNS
    )
    _validate_cv_completeness(
        assembly,
        specs,
        metrics=metrics,
        oof=oof,
        hyperparameters=hyperparameters,
        feature_importance=feature_importance,
    )
    per_group_residuals = _build_per_group_residuals(
        oof,
        expected_group_ids=assembly.contract.expected_group_ids,
    )
    return CvRunResult(
        metrics=metrics,
        oof_predictions=oof,
        hyperparameters=hyperparameters,
        per_group_residuals=per_group_residuals,
        feature_importance=feature_importance,
        preflight=preflight,
        checkpoint_manifest=checkpoint_manifest,
        hyperparameter_grids=grid_payload,
    )


def _validate_model_specs(specs: tuple[ModelSpec, ...], *, formal: bool) -> None:
    names = tuple(spec.name for spec in specs)
    if names != EXPECTED_MODEL_NAMES:
        raise CvContractError(
            f"model roster/order expected {EXPECTED_MODEL_NAMES}, actual {names}"
        )
    expected_types = {
        "SVR_L": SVR,
        "SVR": SVR,
        "LASSO": Lasso,
        "RFR": RandomForestRegressor,
        "GBR": GradientBoostingRegressor,
        "MLPR": MLPRegressor,
    }
    for spec in specs:
        if not isinstance(spec, ModelSpec) or not isinstance(
            spec.estimator, BaseEstimator
        ):
            raise CvContractError(f"{spec.name} estimator 必須是 sklearn BaseEstimator")
        if any(not str(key).startswith("regressor__") for key in spec.param_grid):
            raise CvContractError(f"{spec.name} grid keys 必須使用 regressor__ prefix")
        pipeline = build_training_pipeline(spec)
        if (
            tuple(pipeline.named_steps) != ("scaler", "regressor")
            or not isinstance(pipeline.named_steps["scaler"], StandardScaler)
            or hasattr(pipeline.named_steps["scaler"], "mean_")
        ):
            raise CvContractError(
                f"{spec.name} Pipeline 必須是未 fit StandardScaler→regressor"
            )
        try:
            candidates = list(ParameterGrid(dict(spec.param_grid)))
        except (TypeError, ValueError) as error:
            raise CvContractError(f"{spec.name} param_grid 不合法：{error}") from error
        if not candidates:
            raise CvContractError(f"{spec.name} param_grid 必須至少一個 candidate")
        if not formal:
            continue
        if type(spec.estimator) is not expected_types[spec.name]:
            raise CvContractError(
                f"formal {spec.name} estimator type 必須是 "
                f"{expected_types[spec.name].__name__}"
            )
        base = spec.estimator.get_params(deep=False)
        anchor = RUI_MODEL_ANCHORS[spec.name]
        contains_anchor = False
        for candidate in candidates:
            effective = dict(base)
            effective.update(
                {
                    str(key).removeprefix("regressor__"): value
                    for key, value in candidate.items()
                }
            )
            if all(effective.get(key) == value for key, value in anchor.items()):
                contains_anchor = True
                break
        if not contains_anchor:
            raise CvContractError(f"{spec.name} grid 缺少 Rui anchor {anchor}")
        if "random_state" in base and base["random_state"] != RANDOM_STATE:
            raise CvContractError(f"{spec.name} random_state 必須為 42")


def _outer_splits(
    groups: np.ndarray,
    contract: CvPopulationContract,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    splitter = KFold(
        n_splits=OUTER_FOLD_COUNT,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = tuple(
        (
            np.asarray(train, dtype=np.int64),
            np.asarray(test, dtype=np.int64),
        )
        for train, test in splitter.split(np.arange(len(groups)))
    )
    observed_test = np.concatenate([test for _, test in splits])
    if (
        observed_test.size != len(groups)
        or np.unique(observed_test).size != len(groups)
        or set(observed_test.tolist()) != set(range(len(groups)))
    ):
        raise CvContractError("outer KFold 未恰一次覆蓋每顆 cell")
    expected_groups = set(contract.expected_group_ids)
    for fold_index, (_, test) in enumerate(splits, start=1):
        actual = set(groups[test].tolist())
        if actual != expected_groups:
            raise CvContractError(
                f"outer test fold {fold_index} 必須含完整 9 groups："
                f"missing={sorted(expected_groups - actual)}, "
                f"extra={sorted(actual - expected_groups)}"
            )
    return splits


def _fit_outer_fold(
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    spec: ModelSpec,
    outer_fold: int,
    n_jobs: int,
    importance_repeats: int,
) -> dict[str, object]:
    inner_cv = KFold(
        n_splits=INNER_FOLD_COUNT,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    pipeline = build_training_pipeline(spec)
    search = GridSearchCV(
        pipeline,
        param_grid=dict(spec.param_grid),
        scoring="r2",
        cv=inner_cv,
        refit=True,
        error_score="raise",
        n_jobs=n_jobs,
        return_train_score=False,
    )
    started = perf_counter()
    search.fit(x[train_indices], y[train_indices])
    fit_seconds = perf_counter() - started
    predictions = np.asarray(search.predict(x[test_indices]), dtype=np.float64)
    if predictions.shape != (len(test_indices),) or not np.isfinite(predictions).all():
        raise CvExecutionError("outer test predictions shape/finite gate failed")
    means, stds, method = _fold_feature_importance(
        search.best_estimator_,
        x[test_indices],
        y[test_indices],
        outer_fold=outer_fold,
        n_repeats=importance_repeats,
    )
    best_inner_score = float(search.best_score_)
    if not np.isfinite(best_inner_score):
        raise CvExecutionError("GridSearchCV best_inner_r2 不是 finite")
    return {
        "predictions": predictions,
        "best_params_json": _canonical_json(search.best_params_),
        "best_inner_score": best_inner_score,
        "fit_seconds": float(fit_seconds),
        "importance_means": means,
        "importance_stds": stds,
        "importance_method": method,
    }


def _fold_feature_importance(
    estimator: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    outer_fold: int,
    n_repeats: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    regressor = estimator.named_steps["regressor"]
    feature_count = x_test.shape[1]
    if hasattr(regressor, "feature_importances_"):
        means = np.asarray(regressor.feature_importances_, dtype=np.float64).reshape(-1)
        if means.size == feature_count and np.isfinite(means).all():
            return (
                means,
                np.full(feature_count, np.nan, dtype=np.float64),
                "native_impurity_decrease",
            )
    if hasattr(regressor, "coef_"):
        coefficients = np.asarray(regressor.coef_, dtype=np.float64).reshape(-1)
        if coefficients.size == feature_count and np.isfinite(coefficients).all():
            return (
                np.abs(coefficients),
                np.full(feature_count, np.nan, dtype=np.float64),
                "absolute_standardized_coefficient",
            )
    permutation = permutation_importance(
        estimator,
        x_test,
        y_test,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE + outer_fold,
        n_jobs=1,
    )
    means = np.asarray(permutation.importances_mean, dtype=np.float64)
    stds = np.asarray(permutation.importances_std, dtype=np.float64)
    if (
        means.shape != (feature_count,)
        or stds.shape != (feature_count,)
        or not np.isfinite(means).all()
        or not np.isfinite(stds).all()
    ):
        raise CvExecutionError("permutation importance shape/finite gate failed")
    return means, stds, "permutation_test_cell_r2"


def _fold_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    expected_group_ids: Sequence[str],
) -> dict[str, float | str]:
    frame = pd.DataFrame(
        {"group_id": groups.astype(str), "observed": y_true, "predicted": y_pred}
    )
    grouped = frame.groupby("group_id", sort=False).agg(
        observed=("observed", "first"),
        predicted=("predicted", "mean"),
        observed_unique=("observed", "nunique"),
    )
    if set(grouped.index) != set(expected_group_ids) or not grouped[
        "observed_unique"
    ].eq(1).all():
        raise CvContractError("Rui group R² fold 缺 group 或 group target 不恆定")
    rui_r2 = float(r2_score(grouped["observed"], grouped["predicted"]))
    cell_r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    constant_spearman_input = (
        np.unique(y_true).size < 2 or np.unique(y_pred).size < 2
    )
    if constant_spearman_input:
        cell_spearman = float("nan")
        cell_spearman_status = "undefined_constant_input"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cell_spearman = float(spearmanr(y_true, y_pred).statistic)
        if not np.isfinite(cell_spearman):
            raise CvExecutionError(
                "cell Spearman 在非常數 finite 輸入下必須為 finite"
            )
        cell_spearman_status = "defined"
    numeric = np.asarray([rui_r2, cell_r2, mae, rmse], dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise CvExecutionError("fold R²/MAE/RMSE 必須全為 finite")
    return {
        "rui_r2": rui_r2,
        "cell_r2": cell_r2,
        "r2_gap_rui_minus_cell": rui_r2 - cell_r2,
        "cell_mae": mae,
        "cell_rmse": rmse,
        "cell_spearman": cell_spearman,
        "cell_spearman_status": cell_spearman_status,
    }


def _importance_report_rows(
    *,
    configuration_id: str,
    arm: str,
    model: str,
    outer_fold: int,
    features: Sequence[str],
    method: str,
    means: np.ndarray,
    stds: np.ndarray,
) -> list[dict[str, object]]:
    feature_tuple = tuple(features)
    if means.shape != (len(feature_tuple),) or stds.shape != (len(feature_tuple),):
        raise CheckpointFingerprintError("checkpoint importance feature count mismatch")
    if not np.isfinite(means).all():
        raise CheckpointFingerprintError("checkpoint importance means 含 nonfinite")
    if not (np.isfinite(stds) | np.isnan(stds)).all():
        raise CheckpointFingerprintError("checkpoint importance stds 不合法")
    order = np.argsort(-np.abs(means), kind="stable")
    ranks = np.empty(len(feature_tuple), dtype=np.int64)
    ranks[order] = np.arange(1, len(feature_tuple) + 1)
    return [
        {
            "configuration_id": configuration_id,
            "arm": arm,
            "model": model,
            "outer_fold": outer_fold,
            "feature": feature,
            "importance_method": method,
            "importance_mean": float(means[index]),
            "importance_std": float(stds[index]),
            "absolute_importance": float(abs(means[index])),
            "rank": int(ranks[index]),
            "is_spatial_feature": feature in SPATIAL_FEATURES,
        }
        for index, feature in enumerate(feature_tuple)
    ]


def _build_per_group_residuals(
    oof: pd.DataFrame,
    *,
    expected_group_ids: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (configuration_id, arm, model, outer_fold, group_id), group in oof.groupby(
        ["configuration_id", "arm", "model", "outer_fold", "group_id"],
        sort=False,
    ):
        observed_values = group["observed_ido_score"].to_numpy(dtype=np.float64)
        predicted_values = group["predicted_ido_score"].to_numpy(dtype=np.float64)
        if np.unique(observed_values).size != 1:
            raise CvContractError(f"{group_id} OOF observed target 不恆定")
        observed = float(observed_values[0])
        residuals = predicted_values - observed_values
        rows.append(
            {
                "configuration_id": configuration_id,
                "arm": arm,
                "model": model,
                "outer_fold": int(outer_fold),
                "group_id": group_id,
                "confidence_flag": _constant_group_confidence(group),
                "n_cells": int(len(group)),
                "observed_target": observed,
                "mean_prediction": float(np.mean(predicted_values)),
                "mean_residual": float(np.mean(residuals)),
                "cell_mae": float(np.mean(np.abs(residuals))),
                "cell_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
            }
        )
    report = pd.DataFrame(rows, columns=PER_GROUP_RESIDUAL_COLUMNS)
    configuration_count = oof["configuration_id"].nunique()
    if len(report) != (
        configuration_count * OUTER_FOLD_COUNT * len(expected_group_ids)
    ):
        raise CvContractError("per-group residual table 不完整")
    for _, configuration_fold in report.groupby(
        ["configuration_id", "outer_fold"], sort=False
    ):
        if set(configuration_fold["group_id"]) != set(expected_group_ids):
            raise CvContractError("per-group residual 缺少 expected group")
    return report


def _validate_cv_completeness(
    assembly: FeatureArmAssembly,
    specs: Sequence[ModelSpec],
    *,
    metrics: pd.DataFrame,
    oof: pd.DataFrame,
    hyperparameters: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    expected_configurations = len(assembly.arm_features) * len(specs)
    if expected_configurations != EXPECTED_CONFIGURATION_COUNT:
        raise CvContractError("CV 必須恰為 24 configurations")
    expected_folds = expected_configurations * OUTER_FOLD_COUNT
    if len(metrics) != expected_folds or len(hyperparameters) != expected_folds:
        raise CvContractError("metrics/hyperparameters 必須各有 24×5 rows")
    if metrics.duplicated(["configuration_id", "outer_fold"]).any() or (
        hyperparameters.duplicated(["configuration_id", "outer_fold"]).any()
    ):
        raise CvContractError("metrics/hyperparameters configuration × fold 必須唯一")
    finite_metric_columns = [
        "rui_r2",
        "cell_r2",
        "r2_gap_rui_minus_cell",
        "cell_mae",
        "cell_rmse",
        "best_inner_r2",
        "fit_seconds",
    ]
    if not np.isfinite(
        metrics.loc[:, finite_metric_columns].to_numpy(dtype=np.float64)
    ).all():
        raise CvContractError("除明示 undefined 的 Spearman 外，metrics 必須全為 finite")
    spearman_status = metrics["cell_spearman_status"].astype(str)
    allowed_spearman_statuses = {"defined", "undefined_constant_input"}
    if not set(spearman_status).issubset(allowed_spearman_statuses):
        raise CvContractError("cell_spearman_status 含未知狀態")
    spearman_values = pd.to_numeric(
        metrics["cell_spearman"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    defined_spearman = spearman_status.eq("defined").to_numpy()
    if not np.isfinite(spearman_values[defined_spearman]).all():
        raise CvContractError("defined cell Spearman 必須為 finite")
    if not np.isnan(spearman_values[~defined_spearman]).all():
        raise CvContractError(
            "undefined_constant_input cell Spearman 的數值必須為 NaN"
        )
    expected_oof = assembly.contract.expected_cell_count * expected_configurations
    if len(oof) != expected_oof:
        raise CvContractError(
            f"OOF row count expected {expected_oof}, actual {len(oof)}"
        )
    identity_columns = ["configuration_id", "image_key", "cell_label"]
    if oof.duplicated(identity_columns).any():
        raise CvContractError("OOF configuration × cell identity 必須唯一")
    if not np.isfinite(
        oof.loc[:, ["observed_ido_score", "predicted_ido_score", "residual"]].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise CvContractError("OOF numeric values 必須全為 finite")
    for _, configuration in oof.groupby("configuration_id", sort=False):
        if len(configuration) != assembly.contract.expected_cell_count:
            raise CvContractError("每個 configuration 必須完整覆蓋所有 cells")
        if configuration.loc[:, ["image_key", "cell_label"]].duplicated().any():
            raise CvContractError("configuration OOF cell keys 不唯一")
    expected_importance = sum(
        len(features) for _, features in assembly.arm_features
    ) * len(specs) * OUTER_FOLD_COUNT
    if len(feature_importance) != expected_importance:
        raise CvContractError(
            "feature importance row count mismatch: "
            f"expected={expected_importance}, actual={len(feature_importance)}"
        )
    if feature_importance.duplicated(
        ["configuration_id", "outer_fold", "feature"]
    ).any():
        raise CvContractError("feature importance fold × feature 必須唯一")


def _model_grids_payload(specs: Sequence[ModelSpec]) -> dict[str, object]:
    return {
        spec.name: {
            "estimator_class": (
                f"{type(spec.estimator).__module__}.{type(spec.estimator).__qualname__}"
            ),
            "estimator_params": _jsonable(spec.estimator.get_params(deep=False)),
            "param_grid": _jsonable(dict(spec.param_grid)),
            "scoring": "r2",
            "inner_cv": {
                "class": "KFold",
                "n_splits": INNER_FOLD_COUNT,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
            "pipeline": ["StandardScaler", "regressor"],
            "refit": True,
            "error_score": "raise",
        }
        for spec in specs
    }


def _data_fingerprint(assembly: FeatureArmAssembly) -> str:
    arms = assembly.as_mapping()
    feature_order = tuple(
        dict.fromkeys(feature for features in arms.values() for feature in features)
    )
    identity_and_target = (
        "image_key",
        "cell_label",
        "group_id",
        "group_IDO_score",
    )
    if (
        assembly.contract != CvPopulationContract.formal_exp4()
        and "confidence_flag" in assembly.cells.columns
    ):
        identity_and_target = (*identity_and_target, "confidence_flag")
    columns = (*identity_and_target, *feature_order)
    hashed = pd.util.hash_pandas_object(
        assembly.cells.loc[:, columns],
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(hashed.tobytes(order="C"))
    digest.update(
        _canonical_json(
            {
                "contract": {
                    "scope": assembly.contract.scope,
                    "expected_cell_count": assembly.contract.expected_cell_count,
                    "expected_fov_count": assembly.contract.expected_fov_count,
                    "expected_group_ids": assembly.contract.expected_group_ids,
                    "formal": assembly.contract.formal,
                },
                "arms": arms,
                "columns": columns,
            }
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _split_fingerprint(
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for train, test in splits:
        digest.update(np.asarray(train, dtype=np.int64).tobytes(order="C"))
        digest.update(b"|")
        digest.update(np.asarray(test, dtype=np.int64).tobytes(order="C"))
        digest.update(b";")
    return digest.hexdigest()


def _prepare_checkpoint_manifest(
    checkpoint_dir: Path,
    *,
    fingerprints: Mapping[str, object],
    contract: CvPopulationContract,
) -> dict[str, object]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"
    expected: dict[str, object] = {
        "manifest_version": CHECKPOINT_MANIFEST_VERSION,
        "contract_scope": contract.scope,
        **dict(fingerprints),
    }
    if manifest_path.exists():
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointFingerprintError(
                f"checkpoint manifest 無法讀取：{error}"
            ) from error
        for key, value in expected.items():
            if actual.get(key) != value:
                raise CheckpointFingerprintError(
                    f"checkpoint {key.replace('_fingerprint', '')} fingerprint mismatch: "
                    f"expected={value}, actual={actual.get(key)}"
                )
        return actual
    existing = [path.name for path in checkpoint_dir.iterdir()]
    if existing:
        raise CheckpointFingerprintError(
            "checkpoint directory 非空但缺 manifest.json：" + repr(sorted(existing))
        )
    _write_json_atomic(expected, manifest_path)
    return expected


def _load_fold_checkpoint(
    path: Path,
    *,
    run_fingerprint: str,
    configuration_id: str,
    outer_fold: int,
    expected_test_indices: np.ndarray,
    expected_feature_count: int,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            required = {
                "run_fingerprint",
                "configuration_id",
                "outer_fold",
                "test_indices",
                "predictions",
                "best_params_json",
                "best_inner_score",
                "fit_seconds",
                "importance_means",
                "importance_stds",
                "importance_method",
            }
            missing = sorted(required - set(checkpoint.files))
            if missing:
                raise CheckpointFingerprintError(
                    f"fold checkpoint 缺欄位 {missing}: {path}"
                )
            actual_run = str(checkpoint["run_fingerprint"].item())
            actual_configuration = str(checkpoint["configuration_id"].item())
            actual_fold = int(checkpoint["outer_fold"].item())
            test_indices = np.asarray(checkpoint["test_indices"], dtype=np.int64)
            predictions = np.asarray(checkpoint["predictions"], dtype=np.float64)
            importance_means = np.asarray(
                checkpoint["importance_means"], dtype=np.float64
            )
            importance_stds = np.asarray(
                checkpoint["importance_stds"], dtype=np.float64
            )
            if actual_run != run_fingerprint:
                raise CheckpointFingerprintError(
                    f"fold checkpoint run fingerprint mismatch: {path}"
                )
            if actual_configuration != configuration_id or actual_fold != outer_fold:
                raise CheckpointFingerprintError(
                    f"fold checkpoint identity mismatch: {path}"
                )
            if not np.array_equal(test_indices, expected_test_indices):
                raise CheckpointFingerprintError(
                    f"fold checkpoint split fingerprint mismatch: {path}"
                )
            if predictions.shape != (len(test_indices),) or not np.isfinite(
                predictions
            ).all():
                raise CheckpointFingerprintError(
                    f"fold checkpoint predictions invalid: {path}"
                )
            if importance_means.shape != (expected_feature_count,) or (
                importance_stds.shape != (expected_feature_count,)
            ):
                raise CheckpointFingerprintError(
                    f"fold checkpoint importance count mismatch: {path}"
                )
            best_inner_score = float(checkpoint["best_inner_score"].item())
            fit_seconds = float(checkpoint["fit_seconds"].item())
            if not np.isfinite(best_inner_score) or not np.isfinite(fit_seconds):
                raise CheckpointFingerprintError(
                    f"fold checkpoint scalar metrics invalid: {path}"
                )
            best_params_json = str(checkpoint["best_params_json"].item())
            json.loads(best_params_json)
            return {
                "predictions": predictions.copy(),
                "best_params_json": best_params_json,
                "best_inner_score": best_inner_score,
                "fit_seconds": fit_seconds,
                "importance_means": importance_means.copy(),
                "importance_stds": importance_stds.copy(),
                "importance_method": str(checkpoint["importance_method"].item()),
            }
    except CheckpointFingerprintError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise CheckpointFingerprintError(
            f"fold checkpoint 無法驗證 {path}: {error}"
        ) from error


def _write_fold_checkpoint(
    path: Path,
    payload: Mapping[str, object],
    *,
    run_fingerprint: str,
    configuration_id: str,
    outer_fold: int,
    test_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".npz",
            prefix=f".{path.stem}.",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                run_fingerprint=np.asarray(run_fingerprint),
                configuration_id=np.asarray(configuration_id),
                outer_fold=np.asarray(outer_fold, dtype=np.int64),
                test_indices=np.asarray(test_indices, dtype=np.int64),
                predictions=np.asarray(payload["predictions"], dtype=np.float64),
                best_params_json=np.asarray(str(payload["best_params_json"])),
                best_inner_score=np.asarray(
                    payload["best_inner_score"], dtype=np.float64
                ),
                fit_seconds=np.asarray(payload["fit_seconds"], dtype=np.float64),
                importance_means=np.asarray(
                    payload["importance_means"], dtype=np.float64
                ),
                importance_stds=np.asarray(
                    payload["importance_stds"], dtype=np.float64
                ),
                importance_method=np.asarray(str(payload["importance_method"])),
            )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json.tmp",
            prefix=f".{path.stem}.",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                payload,
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _validate_population_contract(contract: CvPopulationContract) -> None:
    if not isinstance(contract, CvPopulationContract):
        raise CvContractError("contract 必須是 CvPopulationContract")
    if contract.expected_cell_count <= 0 or contract.expected_fov_count <= 0:
        raise CvContractError("population expected counts 必須為正整數")
    if not contract.expected_group_ids or len(set(contract.expected_group_ids)) != len(
        contract.expected_group_ids
    ):
        raise CvContractError("expected_group_ids 必須非空且唯一")
    formal_contracts = (
        CvPopulationContract.formal_exp4(),
        CvPopulationContract.formal_exp4_phase2(),
    )
    if contract.formal and contract not in formal_contracts:
        raise CvContractError("formal contract 不可覆寫 Phase 1／Phase 2 locks")


def _validate_population(
    cells: pd.DataFrame,
    contract: CvPopulationContract,
) -> None:
    required = {"image_key", "cell_label", "group_id", "group_IDO_score"}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise CvContractError(f"modeling cells 缺少 identity/target 欄位：{missing}")
    if len(cells) != contract.expected_cell_count:
        raise CvContractError(
            "CV cell count expected "
            f"{contract.expected_cell_count}, actual {len(cells)}"
        )
    identity = cells.loc[:, ["image_key", "cell_label"]]
    if identity.isna().any().any() or identity.duplicated().any():
        raise CvContractError("CV input 必須是 unique (image_key, cell_label)")
    image_keys = cells["image_key"].astype(str).str.strip()
    if image_keys.eq("").any() or image_keys.nunique() != contract.expected_fov_count:
        raise CvContractError(
            f"CV FOV count 必須恰為 {contract.expected_fov_count} 且 key 非空"
        )
    groups = tuple(sorted(cells["group_id"].astype(str).unique()))
    if set(groups) != set(contract.expected_group_ids):
        raise CvContractError(
            "CV group roster mismatch: "
            f"expected={contract.expected_group_ids}, actual={groups}"
        )
    targets = pd.to_numeric(cells["group_IDO_score"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(targets).all():
        raise CvContractError("group_IDO_score 必須全為 finite")
    per_group_unique = cells.assign(_target=targets).groupby("group_id", sort=False)[
        "_target"
    ].nunique(dropna=False)
    if not per_group_unique.eq(1).all():
        raise CvContractError("每個 group 的 group_IDO_score 必須恆定")
    if np.unique(targets).size != len(contract.expected_group_ids):
        raise CvContractError(
            "group_IDO_score 相異 target 數必須等於 expected group 數"
        )


def _validated_confidence_flags(
    cells: pd.DataFrame,
    contract: CvPopulationContract,
) -> np.ndarray:
    groups = cells["group_id"].astype(str).to_numpy()
    if contract == CvPopulationContract.formal_exp4():
        return np.where(groups == "B8_P7", "low", "normal")
    if "confidence_flag" not in cells.columns:
        if contract == CvPopulationContract.formal_exp4_phase2():
            raise CvContractError("formal Phase 2 cells 必須提供 per-group confidence_flag")
        return np.where(groups == "B8_P7", "low", "normal")
    flags = cells["confidence_flag"]
    if flags.isna().any():
        raise CvContractError("confidence_flag 不可為空")
    normalized = flags.astype(str)
    if not set(normalized).issubset({"normal", "low"}):
        raise CvContractError("confidence_flag 只能是 normal 或 low")
    per_group = cells.assign(_confidence_flag=normalized).groupby(
        "group_id", sort=False
    )["_confidence_flag"].nunique(dropna=False)
    if not per_group.eq(1).all():
        raise CvContractError("confidence_flag 必須在每個 group 內恆定")
    return normalized.to_numpy(dtype=object)


def _constant_group_confidence(group: pd.DataFrame) -> str:
    flags = group["confidence_flag"].astype(str).unique()
    if len(flags) != 1 or flags[0] not in {"normal", "low"}:
        raise CvContractError("OOF confidence_flag 必須在 group fold 內恆定")
    return str(flags[0])


__all__ = [
    "CV_METRIC_COLUMNS",
    "FEATURE_IMPORTANCE_COLUMNS",
    "HYPERPARAMETER_COLUMNS",
    "OOF_COLUMNS",
    "PER_GROUP_RESIDUAL_COLUMNS",
    "CheckpointFingerprintError",
    "CvContractError",
    "CvExecutionError",
    "CvPopulationContract",
    "CvRunResult",
    "FeatureArmAssembly",
    "LeakageContext",
    "LeakagePreflightError",
    "LeakagePreflightResult",
    "ModelSpec",
    "RUI_MODEL_ANCHORS",
    "RUI48_FEATURE_COLUMNS",
    "assemble_feature_arms",
    "build_training_pipeline",
    "default_model_specs",
    "run_leakage_preflight",
    "run_rui_cv",
]
