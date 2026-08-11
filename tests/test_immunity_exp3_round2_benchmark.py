"""Round 2 Paper93 benchmark adapter 的契約測試。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES
from immunity.exp3.round2_benchmark import (
    ROUND2_MODELS,
    normalize_round2_result,
    run_paper93_benchmark,
)


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
