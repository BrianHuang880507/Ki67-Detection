from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_zero_variance_column_is_removed_from_healthy_features() -> None:
    from immunity.exp4.feature_health import evaluate_feature_health

    features = pd.DataFrame(
        {
            "cell__Area": [1.0, 2.0, 3.0, 4.0],
            "cell__MinIntensity": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = evaluate_feature_health(
        features,
        fallback_counts={"cell__Area": 0, "cell__MinIntensity": 0},
        eligible_cell_count=4,
        fallback_scope="smoke_master_labels_after_border_exclusion",
    )
    min_row = result.report.set_index("feature").loc["cell__MinIntensity"]

    assert result.blocked is False
    assert result.reasons == ()
    assert result.healthy_feature_columns == ("cell__Area",)
    assert result.removed_columns == ("cell__MinIntensity",)
    assert bool(min_row["is_zero_variance"]) is True
    assert bool(min_row["removed_from_arms"]) is True
    assert min_row["status"] == "removed_zero_variance"
    assert min_row["nunique"] == 1
    assert min_row["mean"] == 0.0
    assert min_row["std"] == 0.0
    assert min_row["relative_std"] == 0.0
    assert min_row["fallback_scope"] == (
        "smoke_master_labels_after_border_exclusion"
    )
    assert min_row["eligible_cell_count"] == 4
    assert result.eligible_cell_count == 4
    assert result.to_metadata_payload()["eligible_cell_count"] == 4
    assert np.isfinite(result.report["fallback_rate"]).all()


@pytest.mark.parametrize(
    ("fallback_count", "expected_blocked"),
    ((1, False), (2, True)),
)
def test_fallback_rate_one_percent_passes_but_above_one_percent_blocks(
    fallback_count: int,
    expected_blocked: bool,
) -> None:
    from immunity.exp4.feature_health import evaluate_feature_health

    features = pd.DataFrame(
        {"cell__Area": np.arange(1.0, 101.0, dtype=np.float64)}
    )

    result = evaluate_feature_health(
        features,
        fallback_counts={"cell__Area": fallback_count},
        eligible_cell_count=100,
    )
    row = result.report.iloc[0]

    assert result.blocked is expected_blocked
    assert row["fallback_rate"] == fallback_count / 100
    assert ("blocked_fallback_rate" in row["status"]) is expected_blocked
    assert any("fallback rate exceeds 1%" in reason for reason in result.reasons) is (
        expected_blocked
    )


def test_more_than_three_zero_variance_columns_is_fail_closed() -> None:
    from immunity.exp4.feature_health import evaluate_feature_health

    features = pd.DataFrame(
        {
            "cell__A": [1.0, 1.0],
            "cell__B": [2.0, 2.0],
            "cell__C": [3.0, 3.0],
            "cell__D": [4.0, 4.0],
            "cell__Healthy": [1.0, 2.0],
        }
    )

    result = evaluate_feature_health(
        features,
        fallback_counts={},
        eligible_cell_count=2,
    )

    assert result.blocked is True
    assert result.removed_columns == (
        "cell__A",
        "cell__B",
        "cell__C",
        "cell__D",
    )
    assert result.healthy_feature_columns == ("cell__Healthy",)
    assert any(
        "zero-variance feature count exceeds 3" in reason
        for reason in result.reasons
    )


def test_exactly_three_zero_variance_columns_are_removed_without_blocking() -> None:
    from immunity.exp4.feature_health import evaluate_feature_health

    features = pd.DataFrame(
        {
            "cell__A": [1.0, 1.0],
            "cell__B": [2.0, 2.0],
            "cell__C": [3.0, 3.0],
            "cell__Healthy": [1.0, 2.0],
        }
    )

    result = evaluate_feature_health(
        features,
        fallback_counts={},
        eligible_cell_count=2,
    )

    assert result.blocked is False
    assert result.removed_columns == ("cell__A", "cell__B", "cell__C")
    assert result.healthy_feature_columns == ("cell__Healthy",)


def test_near_zero_is_record_only_and_zero_mean_is_not_misclassified() -> None:
    from immunity.exp4.feature_health import evaluate_feature_health

    features = pd.DataFrame(
        {
            "cell__NearZero": [1_000_000.0, 1_000_000.1, 999_999.9],
            "cell__ZeroMean": [-1.0, 0.0, 1.0],
        }
    )

    result = evaluate_feature_health(
        features,
        fallback_counts={},
        eligible_cell_count=3,
    )
    report = result.report.set_index("feature")

    assert result.blocked is False
    assert result.removed_columns == ()
    assert result.healthy_feature_columns == ("cell__NearZero", "cell__ZeroMean")
    assert bool(report.loc["cell__NearZero", "is_near_zero_variance"]) is True
    assert report.loc["cell__NearZero", "status"] == "near_zero_variance"
    assert bool(report.loc["cell__ZeroMean", "is_near_zero_variance"]) is False
    assert np.isinf(report.loc["cell__ZeroMean", "relative_std"])
    assert report.loc["cell__ZeroMean", "status"] == "healthy"


def test_v5_near_zero_uses_five_percent_cv_and_never_removes_the_column() -> None:
    from immunity.exp4.feature_health import (
        NEAR_ZERO_RELATIVE_STD_THRESHOLD,
        evaluate_feature_health,
    )

    features = pd.DataFrame(
        {
            "nucleus__solidity": [0.981, 1.019, 0.981, 1.019],
            "cell__AboveThreshold": [0.94, 1.06, 0.94, 1.06],
        }
    )

    result = evaluate_feature_health(
        features,
        fallback_counts={},
        eligible_cell_count=4,
        feature_columns=("nucleus__solidity", "cell__AboveThreshold"),
    )
    report = result.report.set_index("feature")
    metadata = result.to_metadata_payload()

    assert NEAR_ZERO_RELATIVE_STD_THRESHOLD == 0.05
    assert report.loc["nucleus__solidity", "relative_std"] == pytest.approx(0.019)
    assert bool(report.loc["nucleus__solidity", "is_near_zero_variance"]) is True
    assert bool(report.loc["nucleus__solidity", "removed_from_arms"]) is False
    assert report.loc["nucleus__solidity", "status"] == "near_zero_variance"
    assert bool(report.loc["cell__AboveThreshold", "is_near_zero_variance"]) is False
    assert result.healthy_feature_columns == (
        "nucleus__solidity",
        "cell__AboveThreshold",
    )
    assert metadata["near_zero_cv_threshold"] == 0.05
    assert metadata["near_zero_columns"] == ["nucleus__solidity"]
    assert metadata["coefficient_of_variation"]["nucleus__solidity"] == pytest.approx(
        0.019
    )


def test_v5_combined_66_column_health_keeps_both_near_zero_hits_in_final_65() -> None:
    from immunity.exp4.cell_dedup import NUCLEUS_FEATURE_COLUMNS
    from immunity.exp4.feature_health import evaluate_feature_health
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    columns = (*RUI49_FEATURE_COLUMNS, *NUCLEUS_FEATURE_COLUMNS)
    alternating = np.tile(np.array([-1.0, 1.0]), 50)
    features: dict[str, np.ndarray] = {}
    for index, column in enumerate(columns, start=1):
        mean = float(index + 10)
        features[column] = mean * (1.0 + 0.10 * alternating)
    features["cell__MinIntensity"] = np.zeros(100)
    features["cell__InfoMeas2"] = 10.0 * (1.0 + 0.04 * alternating)
    features["nucleus__solidity"] = 10.0 * (1.0 + 0.02 * alternating)

    result = evaluate_feature_health(
        pd.DataFrame(features),
        fallback_counts={},
        eligible_cell_count=100,
        feature_columns=columns,
        fallback_scope="formal_combined_65_predictors_post_border",
    )
    report = result.report.set_index("feature")

    assert result.blocked is False
    assert result.removed_columns == ("cell__MinIntensity",)
    assert len(result.healthy_feature_columns) == 65
    assert set(
        result.report.loc[result.report["is_near_zero_variance"], "feature"]
    ) == {"cell__InfoMeas2", "nucleus__solidity"}
    assert report.loc["cell__InfoMeas2", "relative_std"] == pytest.approx(0.04)
    assert report.loc["nucleus__solidity", "relative_std"] == pytest.approx(0.02)
    assert "cell__InfoMeas2" in result.healthy_feature_columns
    assert "nucleus__solidity" in result.healthy_feature_columns


def test_nonfinite_feature_is_fail_closed() -> None:
    from immunity.exp4.feature_health import evaluate_feature_health

    features = pd.DataFrame({"cell__Area": [1.0, np.inf, 3.0]})

    result = evaluate_feature_health(
        features,
        fallback_counts={"cell__Area": 0},
        eligible_cell_count=3,
    )

    assert result.blocked is True
    assert result.healthy_feature_columns == ()
    assert result.removed_columns == ()
    assert result.report.loc[0, "status"] == "blocked_nonfinite"
    assert any("nonfinite feature columns" in reason for reason in result.reasons)


def test_health_denominator_must_equal_the_actual_population_rows() -> None:
    from immunity.exp4.feature_health import evaluate_feature_health

    with pytest.raises(ValueError, match="eligible_cell_count.*row count"):
        evaluate_feature_health(
            pd.DataFrame({"cell__Area": [1.0, 2.0, 3.0]}),
            fallback_counts={},
            eligible_cell_count=4,
        )
