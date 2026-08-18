from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import ParameterGrid

import immunity.exp4.cv as cv_module
from immunity.exp4.cell_dedup import NUCLEUS_FEATURE_COLUMNS
from immunity.exp4.cv import (
    CV_METRIC_COLUMNS,
    FEATURE_IMPORTANCE_COLUMNS,
    HYPERPARAMETER_COLUMNS,
    OOF_COLUMNS,
    PER_GROUP_RESIDUAL_COLUMNS,
    CheckpointFingerprintError,
    CvPopulationContract,
    FeatureArmAssembly,
    LeakageContext,
    LeakagePreflightError,
    ModelSpec,
    RUI_MODEL_ANCHORS,
    assemble_feature_arms,
    build_training_pipeline,
    default_model_specs,
    run_leakage_preflight,
    run_rui_cv,
)
from immunity.exp4.rui_features import (
    MORPHOLOGY_FEATURE_NAMES,
    RUI49_FEATURE_COLUMNS,
)


RUI48 = tuple(
    feature
    for feature in RUI49_FEATURE_COLUMNS
    if feature != "cell__MinIntensity"
)
FILTERED30 = tuple(
    feature
    for feature in RUI48
    if feature not in {"cell__Center_X", "cell__Center_Y"}
)[:30]
GROUP_IDS = (
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


class _FirstColumnRegressor(RegressorMixin, BaseEstimator):
    """以第一欄產生確定的非常數預測，供 metrics contract 測試。"""

    def fit(self, x: np.ndarray, y: np.ndarray) -> _FirstColumnRegressor:
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)[:, 0]


class _CoefficientMeanRegressor(RegressorMixin, BaseEstimator):
    """輸出 training mean 並提供 native coefficients，縮短 confidence 回歸測試。"""

    def fit(self, x: np.ndarray, y: np.ndarray) -> _CoefficientMeanRegressor:
        self.n_features_in_ = np.asarray(x).shape[1]
        self.coef_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean_, dtype=np.float64)


def _fixture_contract() -> CvPopulationContract:
    return CvPopulationContract.fixture(
        expected_cell_count=450,
        expected_fov_count=45,
        expected_group_ids=GROUP_IDS,
    )


def _modeling_cells() -> pd.DataFrame:
    rng = np.random.default_rng(20260816)
    row_count = 450
    index = np.arange(row_count)
    group_index = index % len(GROUP_IDS)
    data: dict[str, object] = {
        "image_key": [f"FOV_{value % 45:03d}" for value in index],
        "cell_label": (index // 45) + 1,
        "group_id": [GROUP_IDS[value] for value in group_index],
        "group_IDO_score": 0.5 + group_index.astype(float) * 0.25,
    }
    for offset, feature in enumerate(RUI48, start=1):
        data[feature] = rng.normal(loc=offset, scale=1.0, size=row_count)
    for offset, feature in enumerate(NUCLEUS_FEATURE_COLUMNS, start=1):
        data[feature] = rng.normal(loc=offset + 2.0, scale=0.25, size=row_count)
    return pd.DataFrame(data)


def _leakage_context(*, cellpose_invoked: bool = False) -> LeakageContext:
    return LeakageContext(
        standardization_scope="pipeline_fit_within_outer_training",
        feature_selection_inputs=("X",),
        cellpose_invoked=cellpose_invoked,
        mask_cache_fingerprint_before="b" * 64,
        mask_cache_fingerprint_after="b" * 64,
    )


def _fixture_model_specs() -> tuple[ModelSpec, ...]:
    dummy_specs = tuple(
        ModelSpec(
            name=name,
            estimator=DummyRegressor(strategy="mean"),
            param_grid={},
        )
        for name in ("SVR", "LASSO", "RFR", "GBR", "MLPR")
    )
    return (
        ModelSpec(
            name="SVR_L",
            estimator=_FirstColumnRegressor(),
            param_grid={},
        ),
        *dummy_specs,
    )


def _coefficient_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(
        ModelSpec(
            name=name,
            estimator=_CoefficientMeanRegressor(),
            param_grid={},
        )
        for name in ("SVR_L", "SVR", "LASSO", "RFR", "GBR", "MLPR")
    )


def test_arm_assembly_and_seven_item_leakage_preflight_are_exact() -> None:
    cells = _modeling_cells()
    filtered = FILTERED30

    assembly = assemble_feature_arms(
        cells,
        filtered_features=filtered,
        contract=_fixture_contract(),
    )
    arms = assembly.as_mapping()
    preflight = run_leakage_preflight(assembly, _leakage_context())

    assert tuple(arms) == (
        "geometry_24",
        "rui_48",
        "rui_filtered",
        "rui_48_plus_nucleus",
    )
    assert arms["geometry_24"] == tuple(
        f"cell__{name}" for name in MORPHOLOGY_FEATURE_NAMES
    )
    assert arms["rui_48"] == RUI48
    assert arms["rui_filtered"] == filtered
    assert arms["rui_48_plus_nucleus"] == (*RUI48, *NUCLEUS_FEATURE_COLUMNS)
    assert [len(features) for features in arms.values()] == [24, 48, 30, 65]
    assert all(
        not set(arms[name]).intersection(NUCLEUS_FEATURE_COLUMNS)
        for name in ("geometry_24", "rui_48", "rui_filtered")
    )
    assert set(arms["rui_48_plus_nucleus"]).intersection(
        NUCLEUS_FEATURE_COLUMNS
    ) == set(NUCLEUS_FEATURE_COLUMNS)
    assert preflight.passed is True
    assert preflight.report["check_id"].tolist() == list(range(1, 8))
    assert preflight.report["status"].tolist() == ["passed"] * 7

    with pytest.raises(LeakagePreflightError) as captured:
        run_leakage_preflight(
            assembly,
            _leakage_context(cellpose_invoked=True),
        )
    failed = captured.value.report
    assert failed.loc[failed["check_id"].eq(7), "status"].iloc[0] == "failed"


def test_default_six_model_grids_contain_every_rui_anchor() -> None:
    specs = default_model_specs()

    assert tuple(spec.name for spec in specs) == (
        "SVR_L",
        "SVR",
        "LASSO",
        "RFR",
        "GBR",
        "MLPR",
    )
    assert tuple(type(spec.estimator).__name__ for spec in specs) == (
        "SVR",
        "SVR",
        "Lasso",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "MLPRegressor",
    )
    for spec in specs:
        assert all(key.startswith("regressor__") for key in spec.param_grid)
        candidates = []
        for grid_values in ParameterGrid(dict(spec.param_grid)):
            effective = spec.estimator.get_params(deep=False).copy()
            effective.update(
                {
                    key.removeprefix("regressor__"): value
                    for key, value in grid_values.items()
                }
            )
            candidates.append(effective)
        assert any(
            all(candidate[key] == value for key, value in RUI_MODEL_ANCHORS[spec.name].items())
            for candidate in candidates
        ), spec.name
        if "random_state" in spec.estimator.get_params(deep=False):
            assert spec.estimator.get_params(deep=False)["random_state"] == 42

    for spec in specs:
        pipeline = build_training_pipeline(spec)
        assert tuple(pipeline.named_steps) == ("scaler", "regressor")
        assert type(pipeline.named_steps["scaler"]).__name__ == "StandardScaler"
        assert not hasattr(pipeline.named_steps["scaler"], "mean_")
        assert pipeline.named_steps["regressor"] is not spec.estimator


def test_nested_cv_produces_complete_reused_oof_metrics_residuals_and_checkpoints(
    tmp_path: Path,
) -> None:
    assembly = assemble_feature_arms(
        _modeling_cells(),
        filtered_features=FILTERED30,
        contract=_fixture_contract(),
    )
    kwargs = {
        "checkpoint_dir": tmp_path / "checkpoints",
        "source_fingerprint": "a" * 64,
        "model_specs": _fixture_model_specs(),
        "n_jobs": 1,
        "importance_repeats": 1,
    }

    first = run_rui_cv(assembly, _leakage_context(), **kwargs)

    assert set(first.checkpoint_manifest) >= {
        "data_fingerprint",
        "arms_fingerprint",
        "split_fingerprint",
        "grid_fingerprint",
        "source_fingerprint",
        "sklearn_version",
        "run_fingerprint",
    }
    assert first.metrics.columns.tolist() == list(CV_METRIC_COLUMNS)
    assert first.oof_predictions.columns.tolist() == list(OOF_COLUMNS)
    assert first.hyperparameters.columns.tolist() == list(HYPERPARAMETER_COLUMNS)
    assert first.per_group_residuals.columns.tolist() == list(
        PER_GROUP_RESIDUAL_COLUMNS
    )
    assert first.feature_importance.columns.tolist() == list(
        FEATURE_IMPORTANCE_COLUMNS
    )
    assert len(first.metrics) == 4 * 6 * 5
    assert len(first.oof_predictions) == 450 * 24
    assert len(first.hyperparameters) == 4 * 6 * 5
    assert len(first.per_group_residuals) == 4 * 6 * 5 * 9
    assert set(first.per_group_residuals["outer_fold"]) == {1, 2, 3, 4, 5}
    assert len(first.feature_importance) == (24 + 48 + 30 + 65) * 6 * 5
    assert first.metrics["checkpoint_status"].eq("computed").all()
    assert first.metrics["test_group_count"].eq(9).all()
    assert np.isfinite(first.metrics["rui_r2"]).all()
    assert np.isfinite(first.metrics["cell_r2"]).all()
    assert np.isfinite(first.metrics["cell_mae"]).all()
    assert np.isfinite(first.metrics["cell_rmse"]).all()
    assert first.metrics.loc[
        first.metrics["model"].eq("SVR_L"), "cell_spearman_status"
    ].eq("defined").all()
    assert np.isfinite(
        first.metrics.loc[
            first.metrics["cell_spearman_status"].eq("defined"),
            "cell_spearman",
        ]
    ).all()
    assert first.metrics.loc[
        first.metrics["model"].ne("SVR_L"), "cell_spearman_status"
    ].eq("undefined_constant_input").all()
    assert first.metrics.loc[
        first.metrics["cell_spearman_status"].eq("undefined_constant_input"),
        "cell_spearman",
    ].isna().all()

    per_cell_folds = first.oof_predictions.groupby(
        ["image_key", "cell_label"], sort=False
    )["outer_fold"].nunique()
    assert per_cell_folds.eq(1).all()
    per_configuration_count = first.oof_predictions.groupby(
        "configuration_id", sort=False
    ).size()
    assert len(per_configuration_count) == 24
    assert per_configuration_count.eq(450).all()

    b8 = first.per_group_residuals.loc[
        first.per_group_residuals["group_id"].eq("B8_P7")
    ]
    assert len(b8) == 24 * 5
    assert b8["confidence_flag"].eq("low").all()
    assert first.per_group_residuals.loc[
        first.per_group_residuals["group_id"].ne("B8_P7"),
        "confidence_flag",
    ].eq("normal").all()

    spatial = first.feature_importance.loc[
        first.feature_importance["is_spatial_feature"]
    ]
    assert set(spatial["feature"]) == {
        "cell__Center_X",
        "cell__Center_Y",
        "cell__Orientation",
    }
    assert spatial["rank"].between(1, 65).all()
    assert spatial["importance_method"].eq("permutation_test_cell_r2").all()
    filtered_spatial = spatial.loc[spatial["arm"].eq("rui_filtered")]
    assert set(filtered_spatial["feature"]) == {"cell__Orientation"}

    second = run_rui_cv(assembly, _leakage_context(), **kwargs)
    assert second.metrics["checkpoint_status"].eq("resumed").all()
    pd.testing.assert_frame_equal(
        first.oof_predictions,
        second.oof_predictions,
        check_exact=True,
    )

    with pytest.raises(CheckpointFingerprintError, match="source"):
        run_rui_cv(
            assembly,
            _leakage_context(),
            **{**kwargs, "source_fingerprint": "c" * 64},
        )

    with pytest.raises(CheckpointFingerprintError, match="importance_repeats"):
        run_rui_cv(
            assembly,
            _leakage_context(),
            **{**kwargs, "importance_repeats": 2},
        )


def test_cv_uses_supplied_per_group_confidence_without_phase1_hardcoding(
    tmp_path: Path,
) -> None:
    cells = _modeling_cells()
    cells["confidence_flag"] = np.where(
        cells["group_id"].eq("B4_P5"), "low", "normal"
    )
    assembly = assemble_feature_arms(
        cells,
        filtered_features=FILTERED30,
        contract=_fixture_contract(),
    )

    result = run_rui_cv(
        assembly,
        _leakage_context(),
        checkpoint_dir=tmp_path / "confidence-checkpoints",
        source_fingerprint="d" * 64,
        model_specs=_coefficient_model_specs(),
        n_jobs=1,
        importance_repeats=1,
    )

    assert result.oof_predictions.loc[
        result.oof_predictions["group_id"].eq("B4_P5"), "confidence_flag"
    ].eq("low").all()
    assert result.oof_predictions.loc[
        result.oof_predictions["group_id"].eq("B8_P7"), "confidence_flag"
    ].eq("normal").all()
    assert result.per_group_residuals.loc[
        result.per_group_residuals["group_id"].eq("B4_P5"), "confidence_flag"
    ].eq("low").all()
    assert result.per_group_residuals.loc[
        result.per_group_residuals["group_id"].eq("B8_P7"), "confidence_flag"
    ].eq("normal").all()


def test_formal_phase1_ignores_supplied_confidence_for_legacy_compatibility() -> None:
    fixture = assemble_feature_arms(
        _modeling_cells(),
        filtered_features=FILTERED30,
        contract=_fixture_contract(),
    )
    phase1_contract = CvPopulationContract.formal_exp4()
    phase1_without_confidence = FeatureArmAssembly(
        cells=fixture.cells,
        arm_features=fixture.arm_features,
        contract=phase1_contract,
    )
    supplied = fixture.cells.assign(confidence_flag="low")
    phase1_with_confidence = FeatureArmAssembly(
        cells=supplied,
        arm_features=fixture.arm_features,
        contract=phase1_contract,
    )

    assert cv_module._data_fingerprint(phase1_with_confidence) == (
        cv_module._data_fingerprint(phase1_without_confidence)
    )
    flags = cv_module._validated_confidence_flags(supplied, phase1_contract)
    expected = np.where(
        supplied["group_id"].astype(str).to_numpy() == "B8_P7",
        "low",
        "normal",
    )
    np.testing.assert_array_equal(flags, expected)
