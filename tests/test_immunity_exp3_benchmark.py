from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
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
    condition_adjust_targets,
    fit_final_phase_model,
    make_inner_splits,
    make_outer_splits,
    outer_split_manifest,
    regression_metrics,
    rank_phase_models,
    run_nested_benchmark,
    run_condition_adjusted_sensitivity,
    select_winner,
)
from immunity.exp3.feature_sets import PRIMARY_FOV_FEATURES


VALIDATIONS = (
    "leave_one_b_out",
    "leave_one_passage_out",
    "leave_one_group_out",
    "leave_one_condition_out",
)
CANDIDATE_MODELS = (
    "paper_linear_3f",
    "ridge",
    "elasticnet",
    "rbf_svr",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
)


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
        "feature_sets": {"basic_median": list(PRIMARY_FOV_FEATURES)},
        "expected_outer_split_ids": {
            validation: [f"{validation}:{fold}" for fold in range(1, fold_count + 1)]
            for validation, fold_count in zip(
                VALIDATIONS, (3, 3, 9, 8), strict=True
            )
        },
    }


def make_ranking_fixture(
    fold_counts: tuple[int, int, int, int] = (3, 3, 9, 8),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """建立七個 candidates 與 Dummy 都通過基本完整性的 ranking fixture。"""
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    counts = dict(zip(VALIDATIONS, fold_counts, strict=True))
    for validation, fold_count in counts.items():
        for fold in range(1, fold_count + 1):
            for model_index, model in enumerate((*CANDIDATE_MODELS, "dummy_median")):
                candidate = model != "dummy_median"
                mae = 0.5 + model_index / 10 if candidate else 2.0
                metric_rows.append(
                    {
                        "validation": validation,
                        "fold": str(fold),
                        "split_id": f"{validation}:{fold}",
                        "model": model,
                        "role": "candidate" if candidate else "diagnostic",
                        "feature_set": (
                            "paper_linear_3f"
                            if model == "paper_linear_3f"
                            else "basic_median" if candidate else "none"
                        ),
                        "n_test": 3,
                        "mae": mae,
                        "rmse": mae,
                        "r2": 0.2 if candidate else -0.1,
                        "spearman": 0.5 if candidate else 0.0,
                        "observed_sd": 1.0,
                        "prediction_sd": 0.8 if candidate else 0.0,
                        "status": "ok",
                    }
                )
                predicted = [0.1, 1.0, 1.9] if candidate else [1.0, 1.0, 1.0]
                for observed, prediction in zip([0.0, 1.0, 2.0], predicted, strict=True):
                    prediction_rows.append(
                        {
                            "validation": validation,
                            "fold": str(fold),
                            "split_id": f"{validation}:{fold}",
                            "model": model,
                            "image_key": f"{validation}:{fold}:image-{int(observed)}",
                            "observed_ido_score": observed,
                            "predicted_ido_score": prediction,
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def test_ranking_contains_only_seven_candidates_with_equal_validation_weights() -> None:
    metrics, predictions = make_ranking_fixture((3, 3, 9, 8))

    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), make_tiny_config())

    assert ranking["model"].tolist() == list(CANDIDATE_MODELS)
    assert "dummy_median" not in set(ranking["model"])
    for validation in VALIDATIONS:
        assert ranking[f"{validation}_weight"].eq(0.25).all()
    paper = ranking.set_index("model").loc["paper_linear_3f"]
    assert paper["average_rank"] == pytest.approx(1.0)
    assert paper["overall_rank"] == paper["average_rank"]
    assert paper["eligible"]
    assert paper["winner"]
    assert select_winner(ranking) == "paper_linear_3f"


def test_ranking_fails_feature_gate_without_actual_whitelist_evidence() -> None:
    metrics, predictions = make_ranking_fixture()
    config = make_tiny_config()
    config.pop("feature_sets")

    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), config)

    assert not ranking["phase_only_feature_gate"].any()
    assert not ranking["eligible"].any()
    assert select_winner(ranking) is None


def test_canonical_expected_folds_detect_when_every_model_misses_same_fold() -> None:
    metrics, predictions = make_ranking_fixture()
    missing_split = "leave_one_condition_out:8"
    metrics = metrics[~metrics["split_id"].eq(missing_split)].copy()
    predictions = predictions[~predictions["split_id"].eq(missing_split)].copy()

    ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )

    assert not ranking["complete_outer_folds_gate"].any()
    assert not ranking["eligible"].any()


@pytest.mark.parametrize("dummy_defect", ["missing", "failed"])
def test_incomplete_dummy_fails_beat_dummy_gate(dummy_defect: str) -> None:
    metrics, predictions = make_ranking_fixture()
    selected = metrics["model"].eq("dummy_median") & metrics["split_id"].eq(
        "leave_one_b_out:3"
    )
    if dummy_defect == "missing":
        metrics = metrics.loc[~selected].copy()
    else:
        metrics.loc[selected, "status"] = "failed"

    ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )

    assert not ranking["mae_beats_dummy_gate"].any()
    assert not ranking["eligible"].any()


def test_invalid_expected_fold_evidence_fails_completeness_closed() -> None:
    metrics, predictions = make_ranking_fixture()
    invalid_configs: list[object] = [
        None,
        {},
        {validation: [f"{validation}:1"] for validation in VALIDATIONS[:-1]},
        {
            **make_tiny_config()["expected_outer_split_ids"],
            "leave_one_b_out": ["leave_one_b_out:1", "leave_one_b_out:1"],
        },
    ]

    for invalid in invalid_configs:
        config = make_tiny_config()
        config["expected_outer_split_ids"] = invalid
        ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), config)
        assert not ranking["complete_outer_folds_gate"].any()
        assert not ranking["eligible"].any()


@pytest.mark.parametrize(
    ("validation", "wrong_count"),
    [
        ("leave_one_b_out", 2),
        ("leave_one_b_out", 4),
        ("leave_one_passage_out", 2),
        ("leave_one_passage_out", 4),
        ("leave_one_group_out", 8),
        ("leave_one_group_out", 10),
        ("leave_one_condition_out", 7),
        ("leave_one_condition_out", 9),
    ],
)
def test_expected_fold_evidence_rejects_noncanonical_family_count(
    validation: str, wrong_count: int
) -> None:
    canonical_counts = dict(zip(VALIDATIONS, (3, 3, 9, 8), strict=True))
    canonical_counts[validation] = wrong_count
    fold_counts = tuple(canonical_counts[name] for name in VALIDATIONS)
    metrics, predictions = make_ranking_fixture(fold_counts)
    config = make_tiny_config()
    config["expected_outer_split_ids"] = {
        family: [f"{family}:{fold}" for fold in range(1, count + 1)]
        for family, count in canonical_counts.items()
    }

    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), config)

    assert not ranking["complete_outer_folds_gate"].any()
    assert not ranking["eligible"].any()
    assert select_winner(ranking) is None


def test_one_fold_per_family_evidence_cannot_produce_winner() -> None:
    metrics, predictions = make_ranking_fixture((1, 1, 1, 1))
    config = make_tiny_config()
    config["expected_outer_split_ids"] = {
        validation: [f"{validation}:1"] for validation in VALIDATIONS
    }

    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), config)

    assert not ranking["complete_outer_folds_gate"].any()
    assert not ranking["eligible"].any()
    assert select_winner(ranking) is None


def test_raw_dummy_failure_fails_beat_dummy_gate_despite_ok_metric() -> None:
    metrics, predictions = make_ranking_fixture()
    failures = pd.DataFrame(
        [
            {
                "validation": "leave_one_b_out",
                "fold": "1",
                "split_id": "leave_one_b_out:1",
                "model": "dummy_median",
                "exception_type": "RuntimeError",
                "message": "raw dummy failure evidence",
            }
        ]
    )

    ranking = rank_phase_models(metrics, predictions, failures, make_tiny_config())

    assert not ranking["mae_beats_dummy_gate"].any()
    assert not ranking["eligible"].any()
    assert select_winner(ranking) is None


@pytest.mark.parametrize(
    ("validation", "split_id"),
    [
        ("leave_one_b_out", "leave_one_b_out:99"),
        ("malformed", "not-a-split"),
        (pd.NA, pd.NA),
    ],
)
def test_any_raw_dummy_failure_identity_fails_beat_dummy_gate(
    validation: object, split_id: object
) -> None:
    metrics, predictions = make_ranking_fixture()
    failures = pd.DataFrame(
        [
            {
                "validation": validation,
                "fold": "unknown",
                "split_id": split_id,
                "model": "dummy_median",
                "exception_type": "RuntimeError",
                "message": "raw dummy failure with unusable identity",
            }
        ]
    )

    ranking = rank_phase_models(metrics, predictions, failures, make_tiny_config())

    assert not ranking["mae_beats_dummy_gate"].any()
    assert not ranking["eligible"].any()
    assert select_winner(ranking) is None


def test_missing_oof_row_fails_exact_prediction_completeness() -> None:
    metrics, predictions = make_ranking_fixture()
    missing = predictions["model"].eq("ridge") & predictions["split_id"].eq(
        "leave_one_group_out:9"
    ) & predictions["image_key"].str.endswith("image-2")
    predictions = predictions.loc[~missing].copy()

    ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )
    ridge = ranking.set_index("model").loc["ridge"]

    assert not ridge["complete_outer_folds_gate"]
    assert not ridge["prediction_sd_gate"]
    assert np.isnan(ridge["overall_oof_spearman"])


def test_duplicate_extreme_oof_row_fails_without_distorting_tie_fields() -> None:
    metrics, predictions = make_ranking_fixture()
    duplicate = predictions[
        predictions["model"].eq("ridge")
        & predictions["split_id"].eq("leave_one_b_out:1")
    ].iloc[[0]].copy()
    duplicate["predicted_ido_score"] = 1_000_000.0
    predictions = pd.concat([predictions, duplicate], ignore_index=True)

    ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )
    ridge = ranking.set_index("model").loc["ridge"]

    assert not ridge["complete_outer_folds_gate"]
    assert not ridge["prediction_sd_gate"]
    assert np.isnan(ridge["overall_oof_spearman"])


@pytest.mark.parametrize(
    ("table", "column"),
    [("metrics", "n_test"), ("predictions", "split_id"), ("predictions", "image_key")],
)
def test_missing_oof_evidence_column_fails_prediction_gates_closed(
    table: str, column: str
) -> None:
    metrics, predictions = make_ranking_fixture()
    if table == "metrics":
        metrics = metrics.drop(columns=column)
    else:
        predictions = predictions.drop(columns=column)

    ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )
    ridge = ranking.set_index("model").loc["ridge"]

    assert not ridge["complete_outer_folds_gate"]
    assert not ridge["prediction_sd_gate"]
    assert np.isnan(ridge["overall_oof_spearman"])


def test_five_eligibility_gates_are_individually_auditable() -> None:
    metrics, predictions = make_ranking_fixture()
    bad_model = metrics["model"].eq("ridge")
    metrics.loc[bad_model, ["mae", "rmse"]] = 3.0
    metrics.loc[bad_model, "r2"] = -0.5
    constant = predictions["model"].eq("ridge") & predictions["validation"].eq(
        "leave_one_group_out"
    )
    predictions.loc[constant, "predicted_ido_score"] = 1.0
    missing = metrics["model"].eq("extra_trees") & metrics["validation"].eq(
        "leave_one_condition_out"
    )
    metrics = metrics.loc[~missing].copy()
    config = make_tiny_config()
    config["feature_sets"] = {
        "basic_median": [*PRIMARY_FOV_FEATURES[:-1], "IDO_score"]
    }

    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), config)
    by_model = ranking.set_index("model")

    ridge = by_model.loc["ridge"]
    assert not ridge["mae_beats_dummy_gate"]
    assert not ridge["positive_r2_gate"]
    assert ridge["complete_outer_folds_gate"]
    assert not ridge["phase_only_feature_gate"]
    assert not ridge["prediction_sd_gate"]
    assert not ridge["leave_one_group_out_prediction_sd_gate"]
    assert len(json.loads(ridge["ineligibility_reasons_json"])) == 4
    assert not by_model.loc["paper_linear_3f", "phase_only_feature_gate"]
    assert not by_model.loc["extra_trees", "complete_outer_folds_gate"]


def test_failed_outer_fold_is_ineligible_even_when_metric_values_exist() -> None:
    metrics, predictions = make_ranking_fixture()
    failed = metrics["model"].eq("ridge") & metrics["validation"].eq(
        "leave_one_b_out"
    )
    metrics.loc[failed, "status"] = "failed"
    failures = metrics.loc[failed, ["validation", "fold", "split_id", "model"]].copy()
    failures["exception_type"] = "RuntimeError"
    failures["message"] = "synthetic"
    missing_predictions = predictions["model"].eq("elasticnet") & predictions[
        "validation"
    ].eq("leave_one_passage_out")
    predictions = predictions.loc[~missing_predictions].copy()

    ranking = rank_phase_models(metrics, predictions, failures, make_tiny_config())

    ridge = ranking.set_index("model").loc["ridge"]
    assert not ridge["complete_outer_folds_gate"]
    assert not ridge["eligible"]
    assert "failed_or_missing_outer_folds" in json.loads(
        ridge["ineligibility_reasons_json"]
    )
    assert not ranking.set_index("model").loc[
        "elasticnet", "complete_outer_folds_gate"
    ]


def test_no_eligible_model_does_not_force_a_winner() -> None:
    metrics, predictions = make_ranking_fixture()
    candidate = metrics["role"].eq("candidate")
    metrics.loc[candidate, ["mae", "rmse"]] = 3.0
    metrics.loc[candidate, "r2"] = -0.5
    predictions.loc[
        predictions["model"].isin(CANDIDATE_MODELS), "predicted_ido_score"
    ] = 1.0

    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), make_tiny_config())

    assert not ranking["eligible"].any()
    assert ranking["winner"].eq(False).all()
    assert select_winner(ranking) is None


def test_tie_within_quarter_rank_uses_worst_validation_rank_first() -> None:
    metrics, predictions = make_ranking_fixture()
    patterns = {
        "paper_linear_3f": [0.5, 0.5, 0.5, 0.7],
        "ridge": [0.6, 0.6, 0.6, 0.5],
        "elasticnet": [0.7, 0.7, 0.7, 0.6],
    }
    for model, values in patterns.items():
        for validation, mae in zip(VALIDATIONS, values, strict=True):
            selected = metrics["model"].eq(model) & metrics["validation"].eq(validation)
            metrics.loc[selected, ["mae", "rmse"]] = mae

    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), make_tiny_config())
    by_model = ranking.set_index("model")

    assert by_model.loc["paper_linear_3f", "average_rank"] == pytest.approx(1.5)
    assert by_model.loc["ridge", "average_rank"] == pytest.approx(1.75)
    assert by_model.loc["paper_linear_3f", "worst_validation_rank"] == 3.0
    assert by_model.loc["ridge", "worst_validation_rank"] == 2.0
    assert by_model.loc["ridge", "tied_with_best"]
    assert select_winner(ranking) == "ridge"


def test_tie_break_audit_uses_lobo_then_spearman_then_simplicity() -> None:
    metrics, predictions = make_ranking_fixture()
    rank_patterns = {
        "paper_linear_3f": [0.5, 0.6, 0.5, 0.6],
        "ridge": [0.6, 0.5, 0.6, 0.5],
    }
    for model, values in rank_patterns.items():
        for validation, mae in zip(VALIDATIONS, values, strict=True):
            selected = metrics["model"].eq(model) & metrics["validation"].eq(validation)
            metrics.loc[selected, ["mae", "rmse"]] = mae
    paper_predictions = predictions["model"].eq("paper_linear_3f")
    predictions.loc[paper_predictions, "predicted_ido_score"] = np.tile(
        [0.0, 2.0, 1.0], paper_predictions.sum() // 3
    )

    lobo_ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )
    assert select_winner(lobo_ranking) == "paper_linear_3f"
    audit = lobo_ranking.set_index("model")
    assert audit.loc["paper_linear_3f", "leave_one_b_out_mae"] < audit.loc[
        "ridge", "leave_one_b_out_mae"
    ]
    assert audit.loc["paper_linear_3f", "overall_oof_spearman"] < audit.loc[
        "ridge", "overall_oof_spearman"
    ]

    equal = metrics["model"].isin(["paper_linear_3f", "ridge"])
    metrics.loc[equal, ["mae", "rmse"]] = 0.5
    spearman_ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )
    assert select_winner(spearman_ranking) == "ridge"

    predictions.loc[paper_predictions, "predicted_ido_score"] = np.tile(
        [0.1, 1.0, 1.9], paper_predictions.sum() // 3
    )
    simplicity_ranking = rank_phase_models(
        metrics, predictions, pd.DataFrame(), make_tiny_config()
    )
    assert select_winner(simplicity_ranking) == "paper_linear_3f"
    assert simplicity_ranking.set_index("model").loc[
        "paper_linear_3f", "simplicity_rank"
    ] == 0


def test_condition_adjustment_uses_outer_training_means_only() -> None:
    training = pd.DataFrame(
        {
            "condition_index": [1, 1, 2, 2],
            "IDO_score": [1.0, 3.0, 10.0, 14.0],
        }
    )
    testing = pd.DataFrame(
        {"condition_index": [1, 2], "IDO_score": [5.0, 20.0]}
    )

    train_residual, test_residual = condition_adjust_targets(training, testing)

    assert train_residual.groupby(training["condition_index"]).mean().abs().max() < 1e-12
    np.testing.assert_allclose(test_residual, [3.0, 8.0])
    shifted_test = testing.copy()
    shifted_test["IDO_score"] += 1000.0
    _, shifted_residual = condition_adjust_targets(training, shifted_test)
    np.testing.assert_allclose(shifted_residual, test_residual + 1000.0)


def test_condition_adjustment_rejects_condition_absent_from_outer_training() -> None:
    training = pd.DataFrame({"condition_index": [1, 1], "IDO_score": [1.0, 3.0]})
    testing = pd.DataFrame({"condition_index": [2], "IDO_score": [5.0]})

    with pytest.raises(ValueError, match="outer training.*condition"):
        condition_adjust_targets(training, testing)


def test_condition_sensitivity_runs_three_non_loco_families_without_ranking_input() -> None:
    images = make_grouped_images()
    all_splits = make_outer_splits(images)
    splits = [
        next(split for split in all_splits if split.validation == validation)
        for validation in VALIDATIONS
    ]

    metrics = run_condition_adjusted_sensitivity(
        images,
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        splits,
        make_tiny_config(),
        ["paper_linear_3f"],
    )

    assert set(metrics["validation"]) == set(VALIDATIONS[:3])
    assert "leave_one_condition_out" not in set(metrics["validation"])
    assert metrics["analysis"].eq("training_condition_mean_residual").all()
    assert metrics["model"].eq("paper_linear_3f").all()
    assert metrics["status"].eq("ok").all()

    raw_metrics, predictions = make_ranking_fixture()
    raw_metrics["analysis"] = "raw_ido"
    sensitivity_rows = raw_metrics[
        raw_metrics["model"].eq("paper_linear_3f")
        & raw_metrics["validation"].eq("leave_one_b_out")
    ].copy()
    sensitivity_rows["analysis"] = "training_condition_mean_residual"
    sensitivity_rows[["mae", "rmse"]] = 0.0
    combined = pd.concat([raw_metrics, sensitivity_rows, sensitivity_rows], ignore_index=True)
    ranking = rank_phase_models(combined, predictions, pd.DataFrame(), make_tiny_config())
    assert ranking.set_index("model").loc[
        "paper_linear_3f", "leave_one_b_out_mae"
    ] == pytest.approx(0.5)


def test_condition_sensitivity_failure_does_not_affect_raw_winner() -> None:
    metrics, predictions = make_ranking_fixture()
    failures = pd.DataFrame(
        [
            {
                "validation": "leave_one_b_out",
                "fold": "1",
                "split_id": "leave_one_b_out:1",
                "model": "paper_linear_3f",
                "exception_type": "RuntimeError",
                "message": "sensitivity only",
                "analysis": "training_condition_mean_residual",
            }
        ]
    )

    ranking = rank_phase_models(metrics, predictions, failures, make_tiny_config())

    paper = ranking.set_index("model").loc["paper_linear_3f"]
    assert paper["complete_outer_folds_gate"]
    assert paper["eligible"]
    assert select_winner(ranking) == "paper_linear_3f"


def _patch_exp3_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """將 production output guard 指向測試專用 Exp3 根目錄。"""
    from immunity.exp3 import run_benchmark

    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    return output_root, output_dir


def _create_directory_link(link: Path, target: Path) -> None:
    """建立測試用 directory symlink，Windows 權限不足時改用 junction。"""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        import os
        import subprocess

        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_no_winner_creates_no_final_model_or_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)

    result = fit_final_phase_model(
        make_grouped_images(),
        None,
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        make_tiny_config(),
        output_dir,
    )

    assert result is None
    assert not output_root.exists()


@pytest.mark.parametrize(
    "winner",
    ["dummy_median", "dose_ridge", "dose_plus_morphology_ridge", "unknown"],
)
def test_final_fit_rejects_diagnostics_and_non_candidates_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: str,
) -> None:
    output_root, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="eligible phase-only candidate"):
        fit_final_phase_model(
            make_grouped_images(),
            winner,
            {"basic_median": list(PRIMARY_FOV_FEATURES)},
            make_tiny_config(),
            output_dir,
        )

    assert not output_root.exists()


def test_final_fit_requires_declared_eligibility_and_exact_phase_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)
    config = make_tiny_config()
    config["eligible_phase_models"] = ["paper_linear_3f"]

    with pytest.raises(ValueError, match="eligible_phase_models"):
        fit_final_phase_model(
            make_grouped_images(),
            "ridge",
            {"basic_median": list(PRIMARY_FOV_FEATURES)},
            config,
            output_dir,
        )
    with pytest.raises(ValueError, match="精確 33 個 PRIMARY phase-only"):
        exact_config = make_tiny_config()
        exact_config["eligible_phase_models"] = ["ridge"]
        fit_final_phase_model(
            make_grouped_images(),
            "ridge",
            {"basic_median": [*PRIMARY_FOV_FEATURES[:-1], "delta_IDO_score"]},
            exact_config,
            output_dir,
        )

    assert not output_root.exists()


@pytest.mark.parametrize(
    "eligible_evidence",
    ["missing", None, [], "ridge", {"ridge": True}, ["ridge", "ridge"], ["unknown"]],
)
def test_final_fit_fails_closed_without_valid_eligible_model_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eligible_evidence: object,
) -> None:
    output_root, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)
    config = make_tiny_config()
    if eligible_evidence != "missing":
        config["eligible_phase_models"] = eligible_evidence

    with pytest.raises(ValueError, match="eligible_phase_models"):
        fit_final_phase_model(
            make_grouped_images(),
            "ridge",
            {"basic_median": list(PRIMARY_FOV_FEATURES)},
            config,
            output_dir,
        )

    assert not output_root.exists()


def test_final_fit_writes_auditable_bundle_inside_validated_exp3_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)
    images = make_grouped_images()
    images.attrs["manifest_hash"] = "synthetic-manifest-hash"
    config = make_tiny_config()
    config["eligible_phase_models"] = ["ridge"]

    model_path = fit_final_phase_model(
        images,
        "ridge",
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        config,
        output_dir,
    )

    assert model_path == output_dir / "models" / "ridge.joblib"
    bundle = joblib.load(model_path)
    assert isinstance(bundle["pipeline"], TransformedTargetRegressor)
    assert bundle["model"] == "ridge"
    assert bundle["feature_columns"] == list(PRIMARY_FOV_FEATURES)
    assert bundle["target"] == "image-level background-corrected IDO proxy"
    assert bundle["manifest_hash"] == "synthetic-manifest-hash"
    assert len(bundle["config_hash"]) == 64
    assert bundle["training_scope"] == "all_fov_rows_grouped_inner_tuning"
    metadata = json.loads((output_dir / "final_model.json").read_text("utf-8"))
    assert metadata["model"] == "ridge"
    assert metadata["model_path"] == "models/ridge.joblib"
    assert metadata["best_params"]
    assert metadata["feature_columns"] == list(PRIMARY_FOV_FEATURES)


def test_final_untuned_paper_model_writes_standard_json_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)
    config = make_tiny_config()
    config["eligible_phase_models"] = ["paper_linear_3f"]

    model_path = fit_final_phase_model(
        make_grouped_images(),
        "paper_linear_3f",
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        config,
        output_dir,
    )

    bundle = joblib.load(model_path)
    assert bundle["feature_columns"] == [
        "cell__perimeter__median",
        "nucleus_cytoplasm_area_ratio__median",
        "cell__feret_length__median",
    ]
    metadata = json.loads((output_dir / "final_model.json").read_text("utf-8"))
    assert metadata["best_params"] == {}
    assert metadata["inner_best_mae"] is None


def test_final_fit_rejects_paths_outside_exp3_and_linked_model_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)
    images = make_grouped_images()
    feature_sets = {"basic_median": list(PRIMARY_FOV_FEATURES)}
    config = make_tiny_config()
    config["eligible_phase_models"] = ["ridge"]

    with pytest.raises(ValueError, match="Exp3 output"):
        fit_final_phase_model(images, "ridge", feature_sets, config, tmp_path / "legacy")

    output_dir.mkdir(parents=True)
    outside = tmp_path / "outside-models"
    outside.mkdir()
    _create_directory_link(output_dir / "models", outside)
    with pytest.raises(ValueError, match="symlink|junction"):
        fit_final_phase_model(images, "ridge", feature_sets, config, output_dir)

    assert not any(outside.iterdir())
    assert not (output_dir / "final_model.json").exists()


def test_final_publish_failure_leaves_no_partial_artifact_on_clean_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)
    config = make_tiny_config()
    config["eligible_phase_models"] = ["ridge"]
    model_path = output_dir / "models" / "ridge.joblib"
    metadata_path = output_dir / "final_model.json"
    real_replace = benchmark_module.os.replace
    failed = False

    def fail_metadata_publish_once(source: object, destination: object) -> None:
        """只讓 new metadata publish 失敗一次，rollback 可正常執行。"""
        nonlocal failed
        if Path(destination) == metadata_path and not failed:
            failed = True
            raise OSError("synthetic metadata publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(benchmark_module.os, "replace", fail_metadata_publish_once)
    with pytest.raises(OSError, match="synthetic metadata publish failure"):
        fit_final_phase_model(
            make_grouped_images(),
            "ridge",
            {"basic_median": list(PRIMARY_FOV_FEATURES)},
            config,
            output_dir,
        )

    assert failed
    assert not model_path.exists()
    assert not metadata_path.exists()
    assert not [path for path in output_dir.rglob("*") if path.is_file()]


def test_final_publish_failure_restores_existing_model_and_metadata_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir = _patch_exp3_output_root(tmp_path, monkeypatch)
    config = make_tiny_config()
    config["eligible_phase_models"] = ["ridge"]
    model_path = output_dir / "models" / "ridge.joblib"
    metadata_path = output_dir / "final_model.json"
    model_path.parent.mkdir(parents=True)
    old_model = b"old-model-generation"
    old_metadata = '{"generation":"old"}\n'
    model_path.write_bytes(old_model)
    metadata_path.write_text(old_metadata, encoding="utf-8")
    real_replace = benchmark_module.os.replace
    failed = False

    def fail_metadata_publish_once(source: object, destination: object) -> None:
        """模擬第二個 final replace 失敗，後續 rollback 不再攔截。"""
        nonlocal failed
        if Path(destination) == metadata_path and not failed:
            failed = True
            raise OSError("synthetic metadata publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(benchmark_module.os, "replace", fail_metadata_publish_once)
    with pytest.raises(OSError, match="synthetic metadata publish failure"):
        fit_final_phase_model(
            make_grouped_images(),
            "ridge",
            {"basic_median": list(PRIMARY_FOV_FEATURES)},
            config,
            output_dir,
        )

    assert failed
    assert model_path.read_bytes() == old_model
    assert metadata_path.read_text("utf-8") == old_metadata
    assert sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    ) == ["final_model.json", "models/ridge.joblib"]


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
