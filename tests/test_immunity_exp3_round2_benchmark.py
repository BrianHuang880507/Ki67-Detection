"""Round 2 Paper93 benchmark adapter 的契約測試。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import immunity.exp3.round2_benchmark as round2_benchmark
from immunity.exp3.benchmark import (
    FAILURE_COLUMNS,
    FEATURE_IMPORTANCE_COLUMNS,
    FOLD_METRIC_COLUMNS,
    HYPERPARAMETER_COLUMNS,
    OOF_COLUMNS,
    BenchmarkResult,
    OuterSplit,
    build_model_registry,
)
from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES, PRIMARY_FOV_FEATURES
from immunity.exp3.round2_benchmark import (
    ROUND2_MODELS,
    Round2Comparison,
    build_round2_comparison,
    normalize_round2_result,
    rank_round2_configurations,
    run_paper93_benchmark,
    select_round2_recommendation,
)
from immunity.exp3.round2_evidence import Round1Evidence


def _config() -> dict[str, object]:
    """建立可檢查 seed、search 與 pipeline identity 的最小設定。"""
    return {
        "seed": 173,
        "inner_splits": 2,
        "max_hyperparameter_candidates": 1,
        "n_jobs": 1,
        "permutation_repeats": 2,
        "tree_estimators": 11,
    }


def _paper93_images() -> pd.DataFrame:
    """建立含精確 93 個 predictors 的正式數量 FOV frame。"""
    count = 693
    rows = pd.DataFrame(
        {
            "image_key": [f"image-{index:03d}" for index in range(count)],
            "b_id": [f"B{index % 3}" for index in range(count)],
            "passage": [index % 3 + 5 for index in range(count)],
            "group_id": [f"group-{index % 9}" for index in range(count)],
            "condition_index": [index % 8 + 1 for index in range(count)],
            "condition": [f"condition-{index % 8 + 1}" for index in range(count)],
            "ifn_dose": np.zeros(count),
            "tnf_dose": np.zeros(count),
            "fov": np.ones(count, dtype=int),
            "IDO_score": np.linspace(0.0, 1.0, count),
        }
    )
    return pd.concat(
        [
            rows,
            pd.DataFrame(
                {
                    feature: np.full(count, float(index))
                    for index, feature in enumerate(PAPER_STYLE_FOV_FEATURES)
                }
            ),
        ],
        axis=1,
    )


def _frozen_splits() -> tuple[OuterSplit, ...]:
    """建立保留 3/3/9/8 家族與 693 筆 membership 的 frozen split fixture。"""
    positions = np.arange(693, dtype=np.int64)
    splits: list[OuterSplit] = []
    for validation, fold_count in (
        ("leave_one_b_out", 3),
        ("leave_one_passage_out", 3),
        ("leave_one_group_out", 9),
        ("leave_one_condition_out", 8),
    ):
        for fold, test_index in enumerate(np.array_split(positions, fold_count), start=1):
            splits.append(
                OuterSplit(
                    validation=validation,
                    fold=str(fold),
                    train_index=positions[~np.isin(positions, test_index)],
                    test_index=test_index,
                )
            )
    return tuple(splits)


def _raw_benchmark_result(*, failed: bool = False) -> BenchmarkResult:
    """建立 secondary adapter 回傳的可稽核結果，不進行昂貴模型訓練。"""
    images = _paper93_images()
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    hyperparameter_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    failed_identity = ("leave_one_b_out", "1", "extra_trees")
    for split_index, split in enumerate(_frozen_splits()):
        split_id = f"{split.validation}:{split.fold}"
        for model_index, model in enumerate(ROUND2_MODELS):
            is_failed = failed and (split.validation, split.fold, model) == failed_identity
            metric_rows.append(
                {
                    "validation": split.validation,
                    "fold": split.fold,
                    "split_id": split_id,
                    "model": model,
                    "role": "candidate",
                    "feature_set": "secondary_raw",
                    "n_train": len(split.train_index),
                    "n_test": len(split.test_index),
                    "mae": np.nan if is_failed else float(split_index + model_index),
                    "rmse": np.nan if is_failed else float(split_index + model_index + 1),
                    "r2": np.nan if is_failed else 0.25,
                    "spearman": np.nan if is_failed else 0.5,
                    "observed_sd": 0.1,
                    "prediction_sd": np.nan if is_failed else 0.2,
                    "status": "failed" if is_failed else "ok",
                }
            )
            if is_failed:
                failure_rows.append(
                    {
                        "validation": split.validation,
                        "fold": split.fold,
                        "split_id": split_id,
                        "model": model,
                        "exception_type": "ValueError",
                        "message": "synthetic failed outer fold",
                    }
                )
                continue
            hyperparameter_rows.append(
                {
                    "validation": split.validation,
                    "fold": split.fold,
                    "split_id": split_id,
                    "model": model,
                    "seed": 173,
                    "best_params_json": "{}",
                    "inner_best_mae": 0.3,
                }
            )
            importance_rows.append(
                {
                    "validation": split.validation,
                    "fold": split.fold,
                    "split_id": split_id,
                    "model": model,
                    "role": "candidate",
                    "feature_set": "secondary_raw",
                    "feature": PAPER_STYLE_FOV_FEATURES[0],
                    "importance_type": "outer_test_permutation_diagnostic",
                    "importance": 0.1,
                    "importance_sd": 0.01,
                }
            )
            for position in split.test_index:
                row = images.iloc[int(position)]
                prediction_rows.append(
                    {
                        "validation": split.validation,
                        "fold": split.fold,
                        "split_id": split_id,
                        "model": model,
                        "role": "candidate",
                        "feature_set": "secondary_raw",
                        "image_key": row.image_key,
                        "b_id": row.b_id,
                        "passage": row.passage,
                        "group_id": row.group_id,
                        "condition_index": row.condition_index,
                        "condition": row.condition,
                        "observed_ido_score": row.IDO_score,
                        "predicted_ido_score": row.IDO_score + 0.1,
                    }
                )
    return BenchmarkResult(
        predictions=pd.DataFrame(prediction_rows, columns=OOF_COLUMNS),
        fold_metrics=pd.DataFrame(metric_rows, columns=FOLD_METRIC_COLUMNS),
        hyperparameters=pd.DataFrame(hyperparameter_rows, columns=HYPERPARAMETER_COLUMNS),
        feature_importance=pd.DataFrame(
            importance_rows, columns=FEATURE_IMPORTANCE_COLUMNS
        ),
        failures=pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS),
    )


def test_round2_benchmark_calls_secondary_adapter_once_with_only_two_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公開入口只能一次委派兩個模型與精確 Paper93 registry。"""
    calls: list[dict[str, object]] = []
    raw_expected = _raw_benchmark_result()

    def fake_adapter(
        images: pd.DataFrame,
        feature_sets: Mapping[str, Sequence[str]],
        splits: Sequence[OuterSplit],
        config: Mapping[str, object],
        model_names: Sequence[str],
        feature_set_name: str,
    ) -> BenchmarkResult:
        calls.append(
            {
                "models": tuple(model_names),
                "feature_set": feature_set_name,
                "predictors": tuple(feature_sets[feature_set_name]),
            }
        )
        return raw_expected

    monkeypatch.setattr(round2_benchmark, "run_phase_feature_set_benchmark", fake_adapter)

    result = run_paper93_benchmark(_paper93_images(), _frozen_splits(), _config())

    assert result is not raw_expected
    assert set(result.fold_metrics["source_round"]) == {"round2_paper93"}
    assert set(result.fold_metrics["feature_set"]) == {"paper_style_median"}
    pd.testing.assert_series_equal(result.fold_metrics["mae"], raw_expected.fold_metrics["mae"])
    assert calls == [
        {
            "models": ("extra_trees", "random_forest"),
            "feature_set": "paper_style_median",
            "predictors": PAPER_STYLE_FOV_FEATURES,
        }
    ]


def test_round2_benchmark_never_refits_33_feature_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 2 不可經由 Round 1 全量 runner 重新擬合 33 特徵 baseline。"""
    forbidden = Mock(side_effect=AssertionError("Round 1 baseline must stay frozen"))
    monkeypatch.setattr(round2_benchmark, "run_nested_benchmark", forbidden, raising=False)
    monkeypatch.setattr(
        round2_benchmark,
        "run_phase_feature_set_benchmark",
        lambda *args, **kwargs: _raw_benchmark_result(),
    )

    run_paper93_benchmark(_paper93_images(), _frozen_splits(), _config())

    forbidden.assert_not_called()


def test_round2_benchmark_uses_exact_frozen_split_ids_and_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """adapter 必須收到原始 frozen splits，不得重建或替換 membership。"""
    received: dict[str, object] = {}
    splits = _frozen_splits()

    def fake_adapter(*args: object, **kwargs: object) -> BenchmarkResult:
        received["splits"] = args[2]
        return _raw_benchmark_result()

    monkeypatch.setattr(round2_benchmark, "run_phase_feature_set_benchmark", fake_adapter)

    run_paper93_benchmark(_paper93_images(), splits, _config())

    assert received["splits"] is splits
    forwarded = received["splits"]
    assert isinstance(forwarded, tuple)
    assert [f"{split.validation}:{split.fold}" for split in forwarded] == [
        f"{split.validation}:{split.fold}" for split in splits
    ]
    for actual, expected in zip(forwarded, splits, strict=True):
        np.testing.assert_array_equal(actual.train_index, expected.train_index)
        np.testing.assert_array_equal(actual.test_index, expected.test_index)


def test_round2_full_shape_has_46_metrics_5544_oof_and_46_hyperparameter_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 Round 2 需保留 23 個 frozen folds 與兩模型的預期列數。"""
    monkeypatch.setattr(
        round2_benchmark,
        "run_phase_feature_set_benchmark",
        lambda *args, **kwargs: _raw_benchmark_result(),
    )

    result = run_paper93_benchmark(_paper93_images(), _frozen_splits(), _config())

    assert len(result.fold_metrics) == 46
    assert len(result.predictions) == 5544
    assert len(result.hyperparameters) == 46


def test_round2_fold_failure_is_retained_and_never_substitutes_a_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失敗 fold 是 evidence，不得以其他 split 補替。"""
    monkeypatch.setattr(
        round2_benchmark,
        "run_phase_feature_set_benchmark",
        lambda *args, **kwargs: _raw_benchmark_result(failed=True),
    )

    result = run_paper93_benchmark(_paper93_images(), _frozen_splits(), _config())

    failed = result.fold_metrics[result.fold_metrics["status"].eq("failed")]
    assert failed[["split_id", "model"]].to_dict("records") == [
        {"split_id": "leave_one_b_out:1", "model": "extra_trees"}
    ]
    assert result.failures[["split_id", "model"]].to_dict("records") == [
        {"split_id": "leave_one_b_out:1", "model": "extra_trees"}
    ]
    assert not (
        result.hyperparameters["split_id"].eq("leave_one_b_out:1")
        & result.hyperparameters["model"].eq("extra_trees")
    ).any()


def test_round2_result_labels_every_row_source_round2_and_paper_style_median() -> None:
    """normalization 必須標示所有結果表，且不變更數值證據。"""
    raw = _raw_benchmark_result(failed=True)

    result = normalize_round2_result(raw)

    for raw_frame, normalized_frame in zip(
        (
            raw.predictions,
            raw.fold_metrics,
            raw.hyperparameters,
            raw.feature_importance,
            raw.failures,
        ),
        (
            result.predictions,
            result.fold_metrics,
            result.hyperparameters,
            result.feature_importance,
            result.failures,
        ),
        strict=True,
    ):
        assert set(normalized_frame["source_round"]) == {"round2_paper93"}
        assert set(normalized_frame["feature_set"]) == {"paper_style_median"}
        assert set(normalized_frame["configuration_id"]) == {
            f"{model}__paper_style_median" for model in normalized_frame["model"]
        }
        pd.testing.assert_frame_equal(
            normalized_frame.loc[:, raw_frame.select_dtypes(include="number").columns],
            raw_frame.select_dtypes(include="number"),
        )


def test_round2_uses_same_seed_tree_search_space_and_fold_local_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wrapper 原樣轉交 config，使既有 tree search 與 fold-local pipeline 生效。"""
    received: dict[str, object] = {}
    config = _config()

    def fake_adapter(*args: object, **kwargs: object) -> BenchmarkResult:
        received["config"] = args[3]
        return _raw_benchmark_result()

    monkeypatch.setattr(round2_benchmark, "run_phase_feature_set_benchmark", fake_adapter)

    run_paper93_benchmark(_paper93_images(), _frozen_splits(), config)

    assert received["config"] is config
    registry = build_model_registry(received["config"])
    for model_name in ROUND2_MODELS:
        spec = registry[model_name]
        estimator = spec.estimator_factory(173)
        assert isinstance(estimator, TransformedTargetRegressor)
        assert isinstance(estimator.transformer, StandardScaler)
        assert isinstance(estimator.regressor, Pipeline)
        assert estimator.regressor.named_steps["imputer"].strategy == "median"
        assert "scaler" not in estimator.regressor.named_steps
        assert estimator.regressor.named_steps["model"].random_state == 173
        assert estimator.regressor.named_steps["model"].n_estimators == 11
        assert spec.parameters


_VALIDATIONS = (
    "leave_one_b_out",
    "leave_one_passage_out",
    "leave_one_group_out",
    "leave_one_condition_out",
)
_CONFIGURATIONS = (
    "extra_trees__basic_median",
    "extra_trees__paper_style_median",
    "random_forest__basic_median",
    "random_forest__paper_style_median",
)


def _expected_split_ids() -> dict[str, list[str]]:
    """建立 3/3/9/8 個合法 split identities。"""
    return {
        validation: [f"{validation}:{fold}" for fold in range(1, count + 1)]
        for validation, count in zip(_VALIDATIONS, (3, 3, 9, 8), strict=True)
    }


def _comparison_rows(
    model: str,
    feature_set: str,
    source_round: str,
    prediction_error: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """建立單一 configuration 的完整 fold、OOF、tuning 與 diagnostic evidence。"""
    configuration_id = f"{model}__{feature_set}"
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    hyperparameters: list[dict[str, object]] = []
    importance: list[dict[str, object]] = []
    for validation, split_ids in _expected_split_ids().items():
        for split_id in split_ids:
            fold = split_id.rsplit(":", maxsplit=1)[1]
            metrics.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "split_id": split_id,
                    "model": model,
                    "role": "candidate",
                    "feature_set": feature_set,
                    "n_train": 44,
                    "n_test": 2,
                    "mae": 1.0,
                    "rmse": 1.2,
                    "r2": 0.4,
                    "spearman": 0.5,
                    "observed_sd": 5.0,
                    "prediction_sd": 5.0,
                    "status": "ok",
                    "source_round": source_round,
                    "configuration_id": configuration_id,
                }
            )
            hyperparameters.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "split_id": split_id,
                    "model": model,
                    "seed": 173,
                    "best_params_json": "{}",
                    "inner_best_mae": 0.5,
                    "source_round": source_round,
                    "feature_set": feature_set,
                    "configuration_id": configuration_id,
                }
            )
            features = (
                PRIMARY_FOV_FEATURES
                if feature_set == "basic_median"
                else PAPER_STYLE_FOV_FEATURES
            )
            for feature in features:
                importance.append(
                    {
                        "validation": validation,
                        "fold": fold,
                        "split_id": split_id,
                        "model": model,
                        "role": "candidate",
                        "feature_set": feature_set,
                        "feature": feature,
                        "importance_type": "outer_test_permutation_diagnostic",
                        "importance": 0.1,
                        "importance_sd": 0.01,
                        "source_round": source_round,
                        "configuration_id": configuration_id,
                    }
                )
            for image_index, observed in enumerate((0.0, 10.0)):
                predictions.append(
                    {
                        "validation": validation,
                        "fold": fold,
                        "split_id": split_id,
                        "model": model,
                        "role": "candidate",
                        "feature_set": feature_set,
                        "image_key": f"{split_id}:image-{image_index}",
                        "observed_ido_score": observed,
                        "predicted_ido_score": observed + prediction_error,
                        "source_round": source_round,
                        "configuration_id": configuration_id,
                    }
                )
    return metrics, predictions, hyperparameters, importance


def _round1_evidence_fixture() -> Round1Evidence:
    """建立只含 frozen ET/RF/Dummy evidence 的 Round 1 fixture。"""
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    hyperparameters: list[dict[str, object]] = []
    importance: list[dict[str, object]] = []
    for model, error in (("extra_trees", 0.2), ("random_forest", 0.6)):
        rows = _comparison_rows(model, "basic_median", "round1", error)
        metrics.extend(rows[0])
        predictions.extend(rows[1])
        hyperparameters.extend(rows[2])
        importance.extend(rows[3])
    for validation, split_ids in _expected_split_ids().items():
        for split_id in split_ids:
            fold = split_id.rsplit(":", maxsplit=1)[1]
            metrics.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "split_id": split_id,
                    "model": "dummy_median",
                    "role": "diagnostic",
                    "feature_set": "none",
                    "n_train": 44,
                    "n_test": 2,
                    "mae": 2.0,
                    "rmse": 2.2,
                    "r2": -0.1,
                    "spearman": 0.0,
                    "observed_sd": 5.0,
                    "prediction_sd": 0.0,
                    "status": "ok",
                    "source_round": "round1",
                    "configuration_id": "dummy_median__none",
                }
            )
            for image_index, observed in enumerate((0.0, 10.0)):
                predictions.append(
                    {
                        "validation": validation,
                        "fold": fold,
                        "split_id": split_id,
                        "model": "dummy_median",
                        "role": "diagnostic",
                        "feature_set": "none",
                        "image_key": f"{split_id}:image-{image_index}",
                        "observed_ido_score": observed,
                        "predicted_ido_score": 5.0,
                        "source_round": "round1",
                        "configuration_id": "dummy_median__none",
                    }
                )
    empty = pd.DataFrame()
    return Round1Evidence(
        root=Path("round1-read-only"),
        manifest=empty,
        segmentation_qc=empty,
        basic_cells=empty,
        basic_images=empty,
        split_manifest=empty,
        selected_metrics=pd.DataFrame(metrics),
        selected_predictions=pd.DataFrame(predictions),
        selected_hyperparameters=pd.DataFrame(hyperparameters),
        selected_importance=pd.DataFrame(importance),
        selected_failures=pd.DataFrame(
            columns=["validation", "fold", "split_id", "model", "source_round", "configuration_id"]
        ),
        model_ranking=pd.DataFrame(
            {
                "model": ["random_forest", "extra_trees"],
                "simplicity_rank": [5, 6],
            }
        ),
        metadata={},
        artifact_hashes={},
    )


def _round2_result_fixture(*, failed: bool = False) -> BenchmarkResult:
    """建立已標示 identity 的兩模型 Paper93 Task 4 輸出。"""
    tables: list[list[dict[str, object]]] = [[], [], [], []]
    for model, error in (("extra_trees", 0.4), ("random_forest", 0.8)):
        rows = _comparison_rows(model, "paper_style_median", "round2_paper93", error)
        for destination, source in zip(tables, rows, strict=True):
            destination.extend(source)
    failures = pd.DataFrame(
        columns=["validation", "fold", "split_id", "model", "source_round", "feature_set", "configuration_id"]
    )
    if failed:
        configuration_id = "extra_trees__paper_style_median"
        split_id = "leave_one_b_out:1"
        failed_metric = (
            pd.DataFrame(tables[0])["configuration_id"].eq(configuration_id)
            & pd.DataFrame(tables[0])["split_id"].eq(split_id)
        )
        metric_frame = pd.DataFrame(tables[0])
        metric_frame.loc[failed_metric, ["mae", "rmse", "r2", "spearman", "prediction_sd"]] = np.nan
        metric_frame.loc[failed_metric, "status"] = "failed"
        tables[0] = metric_frame.to_dict("records")
        for index in (1, 2, 3):
            frame = pd.DataFrame(tables[index])
            tables[index] = frame.loc[
                ~(frame["configuration_id"].eq(configuration_id) & frame["split_id"].eq(split_id))
            ].to_dict("records")
        failures = pd.DataFrame(
            [
                {
                    "validation": "leave_one_b_out",
                    "fold": "1",
                    "split_id": split_id,
                    "model": "extra_trees",
                    "exception_type": "ValueError",
                    "message": "synthetic fold failure",
                    "source_round": "round2_paper93",
                    "feature_set": "paper_style_median",
                    "configuration_id": configuration_id,
                }
            ]
        )
    return BenchmarkResult(
        fold_metrics=pd.DataFrame(tables[0]),
        predictions=pd.DataFrame(tables[1]),
        hyperparameters=pd.DataFrame(tables[2]),
        feature_importance=pd.DataFrame(tables[3]),
        failures=failures,
    )


def _round2_result_with_identity_drift(
    table: str,
    drift: str,
) -> BenchmarkResult:
    """在單一 Task 4 table 追加 rogue 或不一致 identity row。"""
    result = _round2_result_fixture(failed=table == "failures")
    frame = getattr(result, table).copy(deep=True)
    changed = frame.iloc[[0]].copy(deep=True)
    if drift == "rogue_configuration":
        changed["configuration_id"] = "rogue_model__paper_style_median"
    elif drift == "model":
        changed["model"] = "random_forest"
    elif drift == "feature_set":
        changed["feature_set"] = "basic_median"
    else:
        changed["source_round"] = "round1"
    changed = pd.concat([frame, changed], ignore_index=True)
    return replace(result, **{table: changed})


def _predictor_sets() -> dict[str, Sequence[str]]:
    """回傳 33/93 authoritative predictor whitelists。"""
    return {
        "basic_median": PRIMARY_FOV_FEATURES,
        "paper_style_median": PAPER_STYLE_FOV_FEATURES,
    }


def _comparison_fixture(*, failed: bool = False) -> Round2Comparison:
    """組合完整 Round 1 與 Task 4 evidence。"""
    return build_round2_comparison(
        _round1_evidence_fixture(),
        _round2_result_fixture(failed=failed),
        _expected_split_ids(),
    )


def _with_family_errors(
    comparison: Round2Comparison,
    errors: Mapping[str, Sequence[float]],
) -> Round2Comparison:
    """以手算 family MAE literals 置換 candidate OOF predictions。"""
    predictions = comparison.predictions.copy(deep=True)
    for configuration_id, family_errors in errors.items():
        for validation, error in zip(_VALIDATIONS, family_errors, strict=True):
            selected = predictions["configuration_id"].eq(configuration_id) & predictions[
                "validation"
            ].eq(validation)
            predictions.loc[selected, "predicted_ido_score"] = (
                predictions.loc[selected, "observed_ido_score"] + error
            )
    predictions.attrs.update(comparison.predictions.attrs)
    return replace(comparison, predictions=predictions)


def test_round2_comparison_contains_exactly_four_candidate_configurations() -> None:
    evidence = _round1_evidence_fixture()
    before = evidence.selected_metrics.copy(deep=True)

    comparison = build_round2_comparison(
        evidence,
        _round2_result_fixture(),
        _expected_split_ids(),
    )

    assert set(comparison.fold_metrics["configuration_id"]) == set(_CONFIGURATIONS)
    assert set(comparison.dummy_metrics["model"]) == {"dummy_median"}
    assert set(comparison.fold_metrics["source_round"]) == {"round1", "round2_paper93"}
    assert len(comparison.fold_metrics) == 92
    assert len(comparison.hyperparameters) == 92
    assert len(comparison.dummy_metrics) == 23
    pd.testing.assert_frame_equal(evidence.selected_metrics, before)


@pytest.mark.parametrize(
    "table",
    [
        "fold_metrics",
        "predictions",
        "hyperparameters",
        "feature_importance",
        "failures",
    ],
)
@pytest.mark.parametrize(
    "drift",
    ["rogue_configuration", "model", "feature_set", "source_round"],
)
def test_round2_comparison_rejects_rogue_or_mismatched_identity_in_every_table(
    table: str,
    drift: str,
) -> None:
    result = _round2_result_with_identity_drift(table, drift)

    with pytest.raises(ValueError, match=rf"{table}.*identity"):
        build_round2_comparison(
            _round1_evidence_fixture(),
            result,
            _expected_split_ids(),
        )


def test_round2_eligibility_uses_frozen_dummy_fold_median_mae() -> None:
    comparison = _comparison_fixture()
    dummy = comparison.dummy_metrics.copy(deep=True)
    dummy.loc[dummy["validation"].eq("leave_one_condition_out"), "mae"] = 0.5
    comparison = replace(comparison, dummy_metrics=dummy)

    ranking = rank_round2_configurations(comparison, _predictor_sets()).set_index(
        "configuration_id"
    )

    assert ranking.loc["extra_trees__basic_median", "validations_beating_dummy"] == 3
    assert ranking.loc["extra_trees__basic_median", "mae_beats_dummy_gate"]
    assert ranking.loc["random_forest__paper_style_median", "validations_beating_dummy"] == 3


def test_round2_positive_r2_requires_two_validation_families() -> None:
    comparison = _comparison_fixture()
    metrics = comparison.fold_metrics.copy(deep=True)
    selected = metrics["configuration_id"].eq("extra_trees__basic_median")
    metrics.loc[selected, "r2"] = -0.1
    metrics.loc[selected & metrics["validation"].eq("leave_one_b_out"), "r2"] = 0.1
    one_family = rank_round2_configurations(
        replace(comparison, fold_metrics=metrics), _predictor_sets()
    ).set_index("configuration_id")
    metrics.loc[selected & metrics["validation"].eq("leave_one_passage_out"), "r2"] = 0.1
    two_families = rank_round2_configurations(
        replace(comparison, fold_metrics=metrics), _predictor_sets()
    ).set_index("configuration_id")

    assert one_family.loc["extra_trees__basic_median", "validations_with_positive_r2"] == 1
    assert not one_family.loc["extra_trees__basic_median", "positive_r2_gate"]
    assert two_families.loc["extra_trees__basic_median", "validations_with_positive_r2"] == 2
    assert two_families.loc["extra_trees__basic_median", "positive_r2_gate"]


def test_round2_complete_gate_requires_23_folds_and_exact_oof_coverage() -> None:
    comparison = _comparison_fixture()
    predictions = comparison.predictions.copy(deep=True)
    remove = predictions[
        "configuration_id"
    ].eq("random_forest__basic_median") & predictions["split_id"].eq(
        "leave_one_group_out:3"
    )
    predictions = predictions.loc[~remove].copy()
    predictions.attrs.update(comparison.predictions.attrs)

    ranking = rank_round2_configurations(
        replace(comparison, predictions=predictions), _predictor_sets()
    ).set_index("configuration_id")

    assert not ranking.loc["random_forest__basic_median", "complete_outer_folds_gate"]
    assert ranking.loc["extra_trees__basic_median", "complete_outer_folds_gate"]


def test_round2_failed_fold_only_makes_its_configuration_ineligible() -> None:
    comparison = _comparison_fixture(failed=True)

    ranking = rank_round2_configurations(comparison, _predictor_sets()).set_index(
        "configuration_id"
    )

    assert len(comparison.fold_metrics) == 92
    failed = comparison.fold_metrics[comparison.fold_metrics["status"].eq("failed")]
    assert failed[["configuration_id", "split_id"]].to_dict("records") == [
        {
            "configuration_id": "extra_trees__paper_style_median",
            "split_id": "leave_one_b_out:1",
        }
    ]
    assert not ranking.loc[
        "extra_trees__paper_style_median", "complete_outer_folds_gate"
    ]
    assert "leave_one_b_out:1" in ranking.loc[
        "extra_trees__paper_style_median", "completeness_issues_json"
    ]
    assert ranking.drop(index="extra_trees__paper_style_median")["eligible"].all()


def test_round2_missing_feature_importance_is_recorded_without_changing_eligibility() -> None:
    comparison = _comparison_fixture()
    importance = comparison.feature_importance.copy(deep=True)
    missing = importance["configuration_id"].eq(
        "random_forest__paper_style_median"
    ) & importance["split_id"].eq("leave_one_passage_out:2")
    importance = importance.loc[~missing].copy()

    ranking = rank_round2_configurations(
        replace(comparison, feature_importance=importance), _predictor_sets()
    ).set_index("configuration_id")

    assert ranking.loc["random_forest__paper_style_median", "eligible"]
    assert "leave_one_passage_out:2" in ranking.loc[
        "random_forest__paper_style_median", "feature_importance_issues_json"
    ]


def test_round2_complete_feature_importance_roster_has_no_diagnostic_issues() -> None:
    ranking = rank_round2_configurations(
        _comparison_fixture(), _predictor_sets()
    )

    assert ranking["feature_importance_issues_json"].tolist() == ["[]"] * 4


@pytest.mark.parametrize(
    ("drift", "expected_reason"),
    [
        ("missing", "missing_feature_importance"),
        ("duplicate", "duplicate_feature_importance"),
        ("rogue", "unexpected_feature_importance"),
    ],
)
def test_round2_feature_importance_qc_requires_exact_unique_authoritative_roster(
    drift: str,
    expected_reason: str,
) -> None:
    comparison = _comparison_fixture()
    importance = comparison.feature_importance.copy(deep=True)
    configuration_id = "extra_trees__basic_median"
    split_id = "leave_one_b_out:1"
    selected = importance["configuration_id"].eq(
        configuration_id
    ) & importance["split_id"].eq(split_id)
    first_feature = PRIMARY_FOV_FEATURES[0]
    target = selected & importance["feature"].eq(first_feature)
    if drift == "missing":
        importance = importance.loc[~target].copy()
    elif drift == "duplicate":
        importance = pd.concat(
            [importance, importance.loc[target].iloc[[0]].copy()],
            ignore_index=True,
        )
    else:
        rogue = importance.loc[target].iloc[[0]].copy()
        rogue["feature"] = "IDO_score"
        importance = pd.concat([importance, rogue], ignore_index=True)

    ranking = rank_round2_configurations(
        replace(comparison, feature_importance=importance), _predictor_sets()
    ).set_index("configuration_id")

    assert ranking.loc[configuration_id, "eligible"]
    issue = ranking.loc[configuration_id, "feature_importance_issues_json"]
    assert expected_reason in issue
    assert split_id in issue


def test_round2_prediction_sd_gate_requires_five_percent_in_every_family() -> None:
    comparison = _comparison_fixture()
    predictions = comparison.predictions.copy(deep=True)
    selected = predictions["configuration_id"].eq(
        "extra_trees__basic_median"
    ) & predictions["validation"].eq("leave_one_condition_out")
    predictions.loc[selected, "predicted_ido_score"] = 4.0
    predictions.attrs.update(comparison.predictions.attrs)

    ranking = rank_round2_configurations(
        replace(comparison, predictions=predictions), _predictor_sets()
    ).set_index("configuration_id")

    assert not ranking.loc[
        "extra_trees__basic_median", "leave_one_condition_out_prediction_sd_gate"
    ]
    assert not ranking.loc["extra_trees__basic_median", "prediction_sd_gate"]


def test_round2_phase_gate_checks_exact_33_or_93_whitelist() -> None:
    predictors = _predictor_sets()
    predictors["paper_style_median"] = predictors["paper_style_median"][:-1]

    ranking = rank_round2_configurations(_comparison_fixture(), predictors).set_index(
        "configuration_id"
    )

    assert ranking.loc[
        ["extra_trees__basic_median", "random_forest__basic_median"],
        "phase_only_feature_gate",
    ].all()
    assert not ranking.loc[
        ["extra_trees__paper_style_median", "random_forest__paper_style_median"],
        "phase_only_feature_gate",
    ].any()


def test_round2_ranking_uses_pooled_oof_mae_not_fold_median_mae() -> None:
    comparison = _with_family_errors(
        _comparison_fixture(),
        {
            "extra_trees__basic_median": (0.8, 0.8, 0.8, 0.8),
            "extra_trees__paper_style_median": (0.2, 0.2, 0.2, 0.2),
            "random_forest__basic_median": (0.6, 0.6, 0.6, 0.6),
            "random_forest__paper_style_median": (0.4, 0.4, 0.4, 0.4),
        },
    )
    metrics = comparison.fold_metrics.copy(deep=True)
    metrics.loc[
        metrics["configuration_id"].eq("extra_trees__basic_median"), "mae"
    ] = 0.1
    metrics.loc[
        metrics["configuration_id"].eq("extra_trees__paper_style_median"), "mae"
    ] = 1.5

    ranking = rank_round2_configurations(
        replace(comparison, fold_metrics=metrics), _predictor_sets()
    ).set_index("configuration_id")

    assert ranking.loc[
        "extra_trees__paper_style_median", "leave_one_b_out_oof_mae"
    ] == pytest.approx(0.2)
    assert ranking.loc[
        "extra_trees__paper_style_median", "leave_one_b_out_rank"
    ] == 1
    assert ranking.loc[
        "extra_trees__paper_style_median", "leave_one_b_out_median_fold_mae"
    ] == pytest.approx(1.5)


def test_round2_family_ranks_use_method_min_and_equal_weights() -> None:
    comparison = _with_family_errors(
        _comparison_fixture(),
        {
            "extra_trees__basic_median": (0.2, 0.2, 0.2, 0.2),
            "extra_trees__paper_style_median": (0.2, 0.4, 0.4, 0.4),
            "random_forest__basic_median": (0.6, 0.6, 0.6, 0.6),
            "random_forest__paper_style_median": (0.8, 0.8, 0.8, 0.8),
        },
    )

    ranking = rank_round2_configurations(comparison, _predictor_sets()).set_index(
        "configuration_id"
    )

    assert ranking.loc["extra_trees__basic_median", "leave_one_b_out_rank"] == 1
    assert ranking.loc[
        "extra_trees__paper_style_median", "leave_one_b_out_rank"
    ] == 1
    assert ranking.loc["extra_trees__basic_median", "average_rank"] == pytest.approx(1.0)
    for validation in _VALIDATIONS:
        assert ranking[f"{validation}_weight"].eq(0.25).all()


def test_round2_tie_band_is_strict_and_prefers_33_for_same_algorithm() -> None:
    comparison = _with_family_errors(
        _comparison_fixture(),
        {
            "extra_trees__basic_median": (1, 2, 1, 2),
            "extra_trees__paper_style_median": (2, 1, 2, 1),
            "random_forest__basic_median": (3, 3, 4, 4),
            "random_forest__paper_style_median": (4, 4, 3, 3),
        },
    )

    ranking = rank_round2_configurations(comparison, _predictor_sets())
    selected = select_round2_recommendation(ranking)
    indexed = ranking.set_index("configuration_id")

    assert selected == "extra_trees__basic_median"
    assert indexed.loc[
        "extra_trees__paper_style_median", "eliminated_by_simplicity"
    ]
    assert not indexed.loc["random_forest__basic_median", "in_tie_band"]


def test_round2_strict_tie_band_excludes_average_rank_difference_exactly_025() -> None:
    comparison = _with_family_errors(
        _comparison_fixture(),
        {
            "extra_trees__basic_median": (1, 2, 1, 4),
            "extra_trees__paper_style_median": (2, 1, 3, 3),
            "random_forest__basic_median": (4, 3, 2, 2),
            "random_forest__paper_style_median": (3, 4, 4, 1),
        },
    )

    ranking = rank_round2_configurations(comparison, _predictor_sets()).set_index(
        "configuration_id"
    )

    assert ranking.loc["extra_trees__basic_median", "average_rank"] == 2.0
    assert ranking.loc["extra_trees__paper_style_median", "average_rank"] == 2.25
    assert ranking.loc["extra_trees__basic_median", "in_tie_band"]
    assert not ranking.loc["extra_trees__paper_style_median", "in_tie_band"]


def test_round2_tie_break_order_is_worst_lobo_spearman_feature_count_simplicity_name() -> None:
    worst = _with_family_errors(
        _comparison_fixture(),
        {
            "extra_trees__basic_median": (1, 4, 1, 2),
            "extra_trees__paper_style_median": (3, 1, 4, 3),
            "random_forest__basic_median": (4, 3, 3, 4),
            "random_forest__paper_style_median": (2, 2, 2, 2),
        },
    )
    assert select_round2_recommendation(
        rank_round2_configurations(worst, _predictor_sets())
    ) == "random_forest__paper_style_median"

    lobo = _with_family_errors(
        _comparison_fixture(),
        {
            "extra_trees__basic_median": (1, 4, 1, 2),
            "extra_trees__paper_style_median": (3, 2, 2, 3),
            "random_forest__basic_median": (4, 3, 3, 4),
            "random_forest__paper_style_median": (2, 1, 4, 1),
        },
    )
    assert select_round2_recommendation(
        rank_round2_configurations(lobo, _predictor_sets())
    ) == "extra_trees__basic_median"

    spearman = _with_family_errors(
        _comparison_fixture(),
        {configuration_id: (10, 10, 10, 10) for configuration_id in _CONFIGURATIONS},
    )
    predictions = spearman.predictions.copy(deep=True)
    selected = predictions["configuration_id"].eq("random_forest__basic_median")
    predictions.loc[selected, "predicted_ido_score"] = 10.0 - predictions.loc[
        selected, "observed_ido_score"
    ]
    predictions.attrs.update(spearman.predictions.attrs)
    spearman = replace(spearman, predictions=predictions)
    assert select_round2_recommendation(
        rank_round2_configurations(spearman, _predictor_sets())
    ) == "extra_trees__basic_median"

    feature_count = _with_family_errors(
        _comparison_fixture(),
        {
            "extra_trees__basic_median": (0.5, 0.5, 0.5, 0.5),
            "extra_trees__paper_style_median": (1.5, 1.5, 1.5, 1.5),
            "random_forest__basic_median": (2.0, 2.0, 2.0, 2.0),
            "random_forest__paper_style_median": (0.5, 0.5, 0.5, 0.5),
        },
    )
    assert select_round2_recommendation(
        rank_round2_configurations(feature_count, _predictor_sets())
    ) == "extra_trees__basic_median"

    simplicity = _with_family_errors(
        _comparison_fixture(),
        {configuration_id: (0.5, 0.5, 0.5, 0.5) for configuration_id in _CONFIGURATIONS},
    )
    assert select_round2_recommendation(
        rank_round2_configurations(simplicity, _predictor_sets())
    ) == "random_forest__basic_median"
    simplicity.fold_metrics.attrs["round1_simplicity_ranks"] = {
        "extra_trees": 5,
        "random_forest": 5,
    }
    assert select_round2_recommendation(
        rank_round2_configurations(simplicity, _predictor_sets())
    ) == "extra_trees__basic_median"


def test_round2_no_eligible_configuration_returns_none() -> None:
    comparison = _comparison_fixture()
    metrics = comparison.fold_metrics.copy(deep=True)
    metrics["r2"] = -1.0

    ranking = rank_round2_configurations(
        replace(comparison, fold_metrics=metrics), _predictor_sets()
    )

    assert not ranking["eligible"].any()
    assert not ranking["recommended"].any()
    assert select_round2_recommendation(ranking) is None


def test_round2_recommendation_never_changes_round1_final_model_json(
    tmp_path: Path,
) -> None:
    final_model = tmp_path / "final_model.json"
    original = b'{"winner":"round1-extra-trees"}\n'
    final_model.write_bytes(original)

    comparison = _comparison_fixture()
    ranking = rank_round2_configurations(comparison, _predictor_sets())
    select_round2_recommendation(ranking)

    assert final_model.read_bytes() == original
