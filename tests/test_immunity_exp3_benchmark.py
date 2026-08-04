from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import immunity.exp3.benchmark as benchmark_module
from immunity.exp3.benchmark import (
    FAILURE_COLUMNS,
    FEATURE_IMPORTANCE_COLUMNS,
    FOLD_METRIC_COLUMNS,
    HYPERPARAMETER_COLUMNS,
    OOF_COLUMNS,
    OuterSplit,
    build_model_registry,
    make_inner_splits,
    make_outer_splits,
    outer_split_manifest,
    regression_metrics,
    run_nested_benchmark,
)
from immunity.exp3.feature_sets import PRIMARY_FOV_FEATURES


def make_grouped_images() -> pd.DataFrame:
    """建立涵蓋全部 Exp3 分組維度的 synthetic image frame。"""
    rows = []
    for b_index, b_id in enumerate(["B4", "B7", "B8"]):
        for passage in [5, 6, 7]:
            for condition_index in range(1, 9):
                rows.append(
                    {
                        "image_key": (f"{b_id}_P{passage}_C{condition_index:02d}_F01"),
                        "b_id": b_id,
                        "passage": passage,
                        "group_id": f"{b_id}_P{passage}",
                        "condition_index": condition_index,
                        "condition": f"condition_{condition_index}",
                        "ifn_dose": float(condition_index - 1),
                        "tnf_dose": 0.0,
                        "fov": 1,
                        "IDO_score": (b_index + passage / 10 + condition_index / 20),
                    }
                )
    images = pd.DataFrame(rows)
    morphology = {
        feature: (
            np.arange(len(images), dtype=float) / 10
            + feature_index
            + images["condition_index"].to_numpy(dtype=float) / 100
        )
        for feature_index, feature in enumerate(PRIMARY_FOV_FEATURES)
    }
    return pd.concat([images, pd.DataFrame(morphology)], axis=1)


def make_tiny_config() -> dict[str, object]:
    """建立可快速執行且維持正式搜尋 contract 的測試設定。"""
    return {
        "seed": 42,
        "inner_splits": 2,
        "max_hyperparameter_candidates": 1,
        "n_jobs": 1,
        "permutation_repeats": 2,
        "tree_estimators": 10,
        "simplicity_order": [
            "paper_linear_3f",
            "ridge",
            "elasticnet",
            "rbf_svr",
            "hist_gradient_boosting",
            "random_forest",
            "extra_trees",
        ],
    }


def test_outer_splits_have_expected_counts_disjoint_groups_and_full_coverage() -> None:
    images = make_grouped_images()

    splits = make_outer_splits(images)

    assert Counter(split.validation for split in splits) == {
        "leave_one_b_out": 3,
        "leave_one_passage_out": 3,
        "leave_one_group_out": 9,
        "leave_one_condition_out": 8,
    }
    grouping_column = {
        "leave_one_b_out": "b_id",
        "leave_one_passage_out": "passage",
        "leave_one_group_out": "group_id",
        "leave_one_condition_out": "condition_index",
    }
    for split in splits:
        assert split.train_index.size > 0
        assert split.test_index.size > 0
        assert set(split.train_index).isdisjoint(split.test_index)
        column = grouping_column[split.validation]
        assert set(images.iloc[split.train_index][column]).isdisjoint(
            images.iloc[split.test_index][column]
        )

    for validation in grouping_column:
        tested = np.concatenate(
            [split.test_index for split in splits if split.validation == validation]
        )
        assert sorted(tested.tolist()) == list(range(len(images)))


def test_outer_splits_are_deterministic_and_fold_names_are_sorted() -> None:
    images = make_grouped_images().sample(frac=1.0, random_state=23)

    first = make_outer_splits(images)
    second = make_outer_splits(images.copy())

    assert [(split.validation, split.fold) for split in first] == [
        ("leave_one_b_out", "B4"),
        ("leave_one_b_out", "B7"),
        ("leave_one_b_out", "B8"),
        ("leave_one_passage_out", "5"),
        ("leave_one_passage_out", "6"),
        ("leave_one_passage_out", "7"),
        *(
            ("leave_one_group_out", group_id)
            for group_id in (
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
        ),
        *(("leave_one_condition_out", str(index)) for index in range(1, 9)),
    ]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.train_index, right.train_index)
        np.testing.assert_array_equal(left.test_index, right.test_index)


def test_outer_split_manifest_preserves_positional_rows_and_metadata() -> None:
    images = make_grouped_images().iloc[[7, 0, 15, 8]].copy()
    images.index = [101, 305, 502, 900]
    split = OuterSplit(
        validation="manual",
        fold="fold-a",
        train_index=np.array([0, 2]),
        test_index=np.array([1, 3]),
    )

    manifest = outer_split_manifest(images, [split])

    assert manifest.columns.tolist() == [
        "validation",
        "fold",
        "image_key",
        "role",
        "b_id",
        "passage",
        "group_id",
        "condition_index",
        "condition",
        "ifn_dose",
        "tnf_dose",
        "fov",
    ]
    assert manifest["image_key"].tolist() == images["image_key"].tolist()
    assert manifest["role"].tolist() == ["train", "test", "train", "test"]
    pd.testing.assert_frame_equal(
        manifest.loc[:, ["image_key", "b_id", "passage", "group_id"]].reset_index(
            drop=True
        ),
        images.loc[:, ["image_key", "b_id", "passage", "group_id"]].reset_index(
            drop=True
        ),
    )


def test_outer_splits_reject_missing_or_ambiguous_grouping_metadata() -> None:
    images = make_grouped_images()

    with pytest.raises(ValueError, match="缺少.*group_id"):
        make_outer_splits(images.drop(columns="group_id"))
    with pytest.raises(ValueError, match="image_key.*重複"):
        make_outer_splits(pd.concat([images, images.iloc[[0]]], ignore_index=True))
    images.loc[0, "b_id"] = None
    with pytest.raises(ValueError, match="b_id.*空值"):
        make_outer_splits(images)


def test_inner_splits_use_groupkfold_without_group_leakage() -> None:
    images = make_grouped_images()

    splits = make_inner_splits(images, requested=5)

    assert len(splits) == 5
    tested = np.concatenate([test_index for _, test_index in splits])
    assert sorted(tested.tolist()) == list(range(len(images)))
    for train_index, test_index in splits:
        assert set(images.iloc[train_index].group_id).isdisjoint(
            images.iloc[test_index].group_id
        )


def test_inner_splits_return_positions_relative_to_non_range_training_frame() -> None:
    training = make_grouped_images().iloc[[0, 8, 16, 24, 32, 40]].copy()
    training.index = [10, 20, 30, 40, 50, 60]

    splits = make_inner_splits(training, requested=20)

    assert len(splits) == training["group_id"].nunique()
    assert all(
        0 <= int(index) < len(training)
        for split in splits
        for indexes in split
        for index in indexes
    )


def test_inner_splits_require_at_least_two_groups_and_two_requested_folds() -> None:
    images = make_grouped_images()

    with pytest.raises(ValueError):
        make_inner_splits(images[images["group_id"].eq("B4_P5")], requested=5)
    with pytest.raises(ValueError, match="requested.*至少為 2"):
        make_inner_splits(images, requested=1)


def test_constant_vectors_keep_errors_but_return_nan_correlations() -> None:
    metrics = regression_metrics([1, 1, 1], [1, 1, 1])

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["spearman"])


def test_metrics_keep_r2_when_only_predictions_are_constant() -> None:
    metrics = regression_metrics([1, 2, 3], [2, 2, 2])

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["r2"] == pytest.approx(0.0)
    assert np.isnan(metrics["spearman"])


def test_single_pair_keeps_errors_but_returns_nan_for_r2_and_spearman() -> None:
    metrics = regression_metrics([3.0], [1.0])

    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["spearman"])


@pytest.mark.parametrize(
    ("observed", "predicted", "message"),
    [
        ([], [], "不可為空"),
        ([1.0, 2.0], [1.0], "長度必須相同"),
        ([1.0, 2.0], [1.0, np.nan], "predicted.*有限值"),
        ([1.0, 2.0], [1.0, np.inf], "predicted.*有限值"),
    ],
)
def test_metrics_reject_empty_unequal_or_nonfinite_predictions(
    observed: list[float], predicted: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        regression_metrics(observed, predicted)


def test_registry_contains_exactly_seven_candidates_and_three_diagnostics() -> None:
    registry = build_model_registry(make_tiny_config())

    assert set(registry) == {
        "paper_linear_3f",
        "ridge",
        "elasticnet",
        "rbf_svr",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "dummy_median",
        "dose_ridge",
        "dose_plus_morphology_ridge",
    }
    assert {name for name, spec in registry.items() if spec.role == "candidate"} == {
        "paper_linear_3f",
        "ridge",
        "elasticnet",
        "rbf_svr",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }
    assert {name for name, spec in registry.items() if spec.role == "diagnostic"} == {
        "dummy_median",
        "dose_ridge",
        "dose_plus_morphology_ridge",
    }


def test_registry_keeps_preprocessing_and_target_scaling_inside_estimators() -> None:
    registry = build_model_registry(make_tiny_config())

    assert isinstance(registry["dummy_median"].estimator_factory(9), DummyRegressor)
    for name, spec in registry.items():
        if name == "dummy_median":
            continue
        estimator = spec.estimator_factory(9)
        assert isinstance(estimator, TransformedTargetRegressor)
        assert isinstance(estimator.transformer, StandardScaler)
        assert isinstance(estimator.regressor, Pipeline)
        assert estimator.regressor.named_steps["imputer"].strategy == "median"
        assert estimator.regressor.named_steps["imputer"].keep_empty_features
        assert ("scaler" in estimator.regressor.named_steps) is spec.scaled

    assert registry["paper_linear_3f"].parameters == {}
    assert registry["dummy_median"].parameters == {}
    assert set(registry["ridge"].parameters) == {"regressor__model__alpha"}
    assert registry["dose_ridge"].parameters == registry["ridge"].parameters
    default_forest = build_model_registry({})["random_forest"].estimator_factory(9)
    assert default_forest.regressor.named_steps["model"].n_estimators == 400
    assert default_forest.regressor.named_steps["model"].random_state == 9
    elasticnet = registry["elasticnet"].estimator_factory(9)
    assert elasticnet.regressor.named_steps["model"].max_iter == 50000


def test_nested_benchmark_uses_shared_splits_and_exact_result_schemas() -> None:
    images = make_grouped_images()
    splits = make_outer_splits(images)

    result = run_nested_benchmark(
        images,
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        splits,
        make_tiny_config(),
    )

    assert result.predictions.columns.tolist() == OOF_COLUMNS
    assert result.fold_metrics.columns.tolist() == FOLD_METRIC_COLUMNS
    assert result.hyperparameters.columns.tolist() == HYPERPARAMETER_COLUMNS
    assert result.feature_importance.columns.tolist() == FEATURE_IMPORTANCE_COLUMNS
    assert result.failures.columns.tolist() == FAILURE_COLUMNS
    expected_split_ids = {f"{split.validation}:{split.fold}" for split in splits}
    by_model = result.fold_metrics.groupby("model")["split_id"].apply(set)
    assert set(by_model.index) == set(build_model_registry(make_tiny_config()))
    assert all(split_ids == expected_split_ids for split_ids in by_model)
    assert result.failures.empty
    assert result.fold_metrics["status"].eq("ok").all()
    assert np.isfinite(result.predictions["predicted_ido_score"]).all()
    importance_features = result.feature_importance.groupby("model")["feature"].apply(
        list
    )
    assert set(importance_features["paper_linear_3f"]) == {
        "cell__perimeter__median",
        "nucleus_cytoplasm_area_ratio__median",
        "cell__feret_length__median",
    }
    assert set(importance_features["dose_ridge"]) == {
        "ifn_dose",
        "tnf_dose",
        "ifn_x_tnf",
    }
    assert set(importance_features["dose_plus_morphology_ridge"]) == {
        "ifn_dose",
        "tnf_dose",
        "ifn_x_tnf",
        *PRIMARY_FOV_FEATURES,
    }


def test_nested_tuning_and_importance_are_deterministic_fold_diagnostics() -> None:
    images = make_grouped_images()
    splits = make_outer_splits(images)[:1]
    model_names = ["paper_linear_3f", "ridge", "rbf_svr", "random_forest"]

    first = run_nested_benchmark(
        images,
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        splits,
        make_tiny_config(),
        model_names=model_names,
    )
    second = run_nested_benchmark(
        images,
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        splits,
        make_tiny_config(),
        model_names=model_names,
    )

    pd.testing.assert_frame_equal(first.hyperparameters, second.hyperparameters)
    assert first.hyperparameters["best_params_json"].map(json.loads).map(dict).map(
        bool
    ).tolist() == [
        False,
        True,
        True,
        True,
    ]
    importance_types = first.feature_importance.groupby("model")[
        "importance_type"
    ].unique()
    assert importance_types["paper_linear_3f"].tolist() == ["standardized_coefficient"]
    assert importance_types["ridge"].tolist() == ["standardized_coefficient"]
    assert importance_types["rbf_svr"].tolist() == ["outer_test_permutation_diagnostic"]
    assert importance_types["random_forest"].tolist() == [
        "outer_test_permutation_diagnostic"
    ]


def test_nonfinite_predictions_are_recorded_as_fold_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = make_grouped_images()
    split = make_outer_splits(images)[:1]

    def predict_nonfinite(self: DummyRegressor, values: object) -> np.ndarray:
        """模擬 estimator 回傳非有限預測。"""
        return np.full(len(values), np.nan)

    monkeypatch.setattr(DummyRegressor, "predict", predict_nonfinite)
    result = run_nested_benchmark(
        images,
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        split,
        make_tiny_config(),
        model_names=["dummy_median", "paper_linear_3f"],
    )

    assert set(result.predictions["model"]) == {"paper_linear_3f"}
    assert result.fold_metrics.set_index("model")["status"].to_dict() == {
        "dummy_median": "failed",
        "paper_linear_3f": "ok",
    }
    assert result.failures[["model", "exception_type"]].to_dict("records") == [
        {"model": "dummy_median", "exception_type": "ValueError"}
    ]
    assert "有限" in result.failures.iloc[0]["message"]


@pytest.mark.parametrize(
    "predictors",
    [
        ["IDO_score"],
        list(PRIMARY_FOV_FEATURES[:-1]),
        [*PRIMARY_FOV_FEATURES[:-1], "ifn_dose"],
    ],
)
def test_benchmark_rejects_leaking_or_nonexact_basic_median_features(
    predictors: list[str],
) -> None:
    images = make_grouped_images()

    with pytest.raises(ValueError, match="basic_median.*33.*phase-only"):
        run_nested_benchmark(
            images,
            {"basic_median": predictors},
            make_outer_splits(images)[:1],
            make_tiny_config(),
            model_names=["ridge"],
        )


def test_importance_failure_warns_without_erasing_successful_core_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = make_grouped_images()
    split = make_outer_splits(images)[:1]

    def fail_importance(*args: object, **kwargs: object) -> list[dict[str, object]]:
        """模擬純 diagnostic importance 計算失敗。"""
        raise RuntimeError("importance exploded")

    monkeypatch.setattr(benchmark_module, "_fold_feature_importance", fail_importance)
    with pytest.warns(
        RuntimeWarning,
        match=(
            "leave_one_b_out:B4.*paper_linear_3f.*RuntimeError.*importance exploded"
        ),
    ):
        result = run_nested_benchmark(
            images,
            {"basic_median": list(PRIMARY_FOV_FEATURES)},
            split,
            make_tiny_config(),
            model_names=["paper_linear_3f"],
        )

    assert len(result.predictions) == len(split[0].test_index)
    assert result.fold_metrics[["model", "status"]].to_dict("records") == [
        {"model": "paper_linear_3f", "status": "ok"}
    ]
    assert result.hyperparameters["model"].tolist() == ["paper_linear_3f"]
    assert result.failures.empty
    assert result.feature_importance.empty


@pytest.mark.parametrize("different_indices", [False, True])
def test_benchmark_rejects_duplicate_split_id_before_fit(
    monkeypatch: pytest.MonkeyPatch,
    different_indices: bool,
) -> None:
    images = make_grouped_images()
    first, second = make_outer_splits(images)[:2]
    duplicate = (
        OuterSplit(
            validation=first.validation,
            fold=first.fold,
            train_index=second.train_index,
            test_index=second.test_index,
        )
        if different_indices
        else first
    )
    fit_calls = 0

    def track_fit(self: DummyRegressor, values: object, target: object) -> object:
        """記錄 duplicate validation 是否錯誤地等到 model fit 後才執行。"""
        nonlocal fit_calls
        fit_calls += 1
        return self

    monkeypatch.setattr(DummyRegressor, "fit", track_fit)
    with pytest.raises(ValueError, match="duplicate split_id.*leave_one_b_out:B4"):
        run_nested_benchmark(
            images,
            {"basic_median": list(PRIMARY_FOV_FEATURES)},
            [first, duplicate],
            make_tiny_config(),
            model_names=["dummy_median"],
        )
    assert fit_calls == 0
