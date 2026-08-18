from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from immunity.exp4.post_cv_analysis import (
    ORIENTATION_BIN_COLUMNS,
    ORIENTATION_RANGE_TOLERANCE,
    ORIENTATION_SUMMARY_COLUMNS,
    SHRINKAGE_GROUP_COLUMNS,
    SHRINKAGE_REPORT_COLUMNS,
    OrientationMarginalResult,
    PostCvAnalysisError,
    ShrinkageAnalysisResult,
    analyze_orientation_marginal,
    analyze_shrinkage,
)


ARMS = (
    "geometry_24",
    "rui_48",
    "rui_filtered",
    "rui_48_plus_nucleus",
)
MODELS = ("SVR_L", "SVR", "LASSO", "RFR", "GBR", "MLPR")
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
CELL_COUNT = 19_648
FOV_COUNT = 693


@pytest.fixture(scope="module")
def formal_oof_predictions() -> pd.DataFrame:
    cell_index = np.arange(CELL_COUNT, dtype=np.int64)
    group_index = cell_index % len(GROUP_IDS)
    observed_by_group = np.linspace(0.55, 2.46, len(GROUP_IDS))
    image_keys = np.asarray(
        [f"FOV_{value:03d}" for value in cell_index % FOV_COUNT],
        dtype=object,
    )
    cell_labels = (cell_index // FOV_COUNT) + 1
    groups = np.asarray(GROUP_IDS, dtype=object)[group_index]
    observed = observed_by_group[group_index]

    frames: list[pd.DataFrame] = []
    for arm_index, arm in enumerate(ARMS):
        for model_index, model in enumerate(MODELS):
            slope = 0.25 + 0.01 * model_index
            intercept = 1.0 + 0.05 * arm_index
            predicted = intercept + slope * observed
            frames.append(
                pd.DataFrame(
                    {
                        "configuration_id": f"{arm}__{model}",
                        "arm": arm,
                        "model": model,
                        "image_key": image_keys,
                        "cell_label": cell_labels,
                        "group_id": groups,
                        "observed_ido_score": observed,
                        "predicted_ido_score": predicted,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def formal_orientation_cells() -> pd.DataFrame:
    cell_index = np.arange(CELL_COUNT, dtype=np.int64)
    group_index = cell_index % len(GROUP_IDS)
    return pd.DataFrame(
        {
            "image_key": [
                f"FOV_{value:03d}" for value in cell_index % FOV_COUNT
            ],
            "cell_label": (cell_index // FOV_COUNT) + 1,
            "group_id": np.asarray(GROUP_IDS, dtype=object)[group_index],
            "cell__Orientation": np.linspace(
                -np.pi / 2.0,
                np.pi / 2.0,
                CELL_COUNT,
            ),
            "group_IDO_score": 0.5 + group_index * 0.25,
        }
    )


def test_shrinkage_analysis_returns_complete_auditable_ols_report(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    result = analyze_shrinkage(formal_oof_predictions)

    assert isinstance(result, ShrinkageAnalysisResult)
    assert result.report.columns.tolist() == list(SHRINKAGE_REPORT_COLUMNS)
    assert result.group_summary.columns.tolist() == list(SHRINKAGE_GROUP_COLUMNS)
    assert len(result.report) == 4 * 6
    assert len(result.group_summary) == 4 * 6 * 9
    assert result.report["n_cells"].eq(CELL_COUNT).all()
    assert result.report["n_fovs"].eq(FOV_COUNT).all()
    assert result.report["n_groups"].eq(9).all()

    first = result.report.loc[
        result.report["configuration_id"].eq("geometry_24__SVR_L")
    ].iloc[0]
    assert first["slope"] == pytest.approx(0.25)
    assert first["intercept"] == pytest.approx(1.0)
    assert first["regression_r_squared"] == pytest.approx(1.0)
    assert first["regression_r"] == pytest.approx(1.0)
    assert first["observed_min"] == pytest.approx(0.55)
    assert first["observed_max"] == pytest.approx(2.46)
    assert first["observed_range"] == pytest.approx(1.91)
    assert first["predicted_range"] == pytest.approx(0.4775)
    assert first["predicted_to_observed_range_ratio"] == pytest.approx(0.25)
    with pytest.raises(FrozenInstanceError):
        result.report = result.report.copy()


def test_shrinkage_analysis_rejects_duplicate_configuration_cell_key(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    duplicated = pd.concat(
        [formal_oof_predictions, formal_oof_predictions.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(PostCvAnalysisError, match="configuration.*cell"):
        analyze_shrinkage(duplicated)


def test_shrinkage_analysis_rejects_missing_required_column(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    malformed = formal_oof_predictions.drop(columns="predicted_ido_score")

    with pytest.raises(PostCvAnalysisError, match="predicted_ido_score"):
        analyze_shrinkage(malformed)


def test_shrinkage_analysis_rejects_duplicate_column_labels(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    malformed = pd.concat(
        [formal_oof_predictions, formal_oof_predictions[["arm"]]],
        axis=1,
    )

    with pytest.raises(PostCvAnalysisError, match="column labels"):
        analyze_shrinkage(malformed)


def test_shrinkage_analysis_rejects_noncanonical_identifier_whitespace(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    malformed = formal_oof_predictions.copy()
    malformed.loc[0, "arm"] = " geometry_24 "

    with pytest.raises(PostCvAnalysisError, match="canonical identifiers"):
        analyze_shrinkage(malformed)


def test_shrinkage_analysis_rejects_incomplete_oof_population(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    incomplete = formal_oof_predictions.iloc[:-1]

    with pytest.raises(PostCvAnalysisError, match="row count"):
        analyze_shrinkage(incomplete)


def test_shrinkage_analysis_rejects_configuration_arm_model_mismatch(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    mismatched = formal_oof_predictions.copy()
    mismatched.loc[0, "arm"] = "unknown_arm"

    with pytest.raises(PostCvAnalysisError, match="configuration_id.*arm.*model"):
        analyze_shrinkage(mismatched)


def test_shrinkage_analysis_rejects_different_cell_population_between_configurations(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    mismatched = formal_oof_predictions.copy()
    mismatched.loc[len(mismatched) - 1, ["image_key", "cell_label"]] = [
        "FOV_EXTRA",
        1,
    ]

    with pytest.raises(PostCvAnalysisError, match="cell population"):
        analyze_shrinkage(mismatched)


@pytest.mark.parametrize("column", ["observed_ido_score", "predicted_ido_score"])
def test_shrinkage_analysis_rejects_nonfinite_values(
    formal_oof_predictions: pd.DataFrame,
    column: str,
) -> None:
    nonfinite = formal_oof_predictions.copy()
    nonfinite.loc[0, column] = np.inf

    with pytest.raises(PostCvAnalysisError, match="finite"):
        analyze_shrinkage(nonfinite)


def test_shrinkage_analysis_rejects_nonconstant_observed_target_within_group(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    inconsistent = formal_oof_predictions.copy()
    inconsistent.loc[0, "observed_ido_score"] += 0.01

    with pytest.raises(PostCvAnalysisError, match="observed target.*constant"):
        analyze_shrinkage(inconsistent)


def test_shrinkage_analysis_rejects_zero_observed_group_range(
    formal_oof_predictions: pd.DataFrame,
) -> None:
    no_group_variation = formal_oof_predictions.copy()
    no_group_variation["observed_ido_score"] = 1.0

    with pytest.raises(PostCvAnalysisError, match="observed range"):
        analyze_shrinkage(no_group_variation)


def test_orientation_marginal_returns_fixed_bins_and_machine_readable_summary(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    result = analyze_orientation_marginal(formal_orientation_cells)

    assert isinstance(result, OrientationMarginalResult)
    assert result.bins.columns.tolist() == list(ORIENTATION_BIN_COLUMNS)
    assert result.summary.columns.tolist() == list(ORIENTATION_SUMMARY_COLUMNS)
    assert len(result.bins) == 12
    assert len(result.summary) == 1
    assert result.bins["n_cells"].sum() == CELL_COUNT
    assert result.bins["n_groups"].eq(9).all()
    np.testing.assert_allclose(
        result.bins["right_edge_degrees"]
        - result.bins["left_edge_degrees"],
        15.0,
    )
    assert result.bins.iloc[0]["left_edge_radians"] == pytest.approx(
        -np.pi / 2.0
    )
    assert result.bins.iloc[-1]["right_edge_radians"] == pytest.approx(
        np.pi / 2.0
    )
    assert result.bins.iloc[0]["left_edge_degrees"] == pytest.approx(-90.0)
    assert result.bins.iloc[-1]["right_edge_degrees"] == pytest.approx(90.0)

    summary = result.summary.iloc[0]
    assert summary["n_cells"] == CELL_COUNT
    assert summary["n_fovs"] == FOV_COUNT
    assert summary["n_groups"] == 9
    assert summary["n_bins"] == 12
    assert bool(summary["every_bin_covers_all_9_groups"]) is True
    assert np.isfinite(summary["cell_spearman_r"])
    assert np.isfinite(summary["group_spearman_r"])
    assert summary["bin_target_median_range"] == pytest.approx(
        result.bins["target_median"].max()
        - result.bins["target_median"].min()
    )
    assert summary["bin_target_mean_range"] == pytest.approx(
        result.bins["target_mean"].max()
        - result.bins["target_mean"].min()
    )
    assert summary["interpretation_scope"] == "descriptive_noncausal"
    with pytest.raises(FrozenInstanceError):
        result.bins = result.bins.copy()


def test_orientation_marginal_rejects_duplicate_cell_identity(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    duplicated = pd.concat(
        [formal_orientation_cells.iloc[:-1], formal_orientation_cells.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(PostCvAnalysisError, match="unique.*image_key.*cell_label"):
        analyze_orientation_marginal(duplicated)


def test_orientation_marginal_rejects_missing_required_column(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    malformed = formal_orientation_cells.drop(columns="cell__Orientation")

    with pytest.raises(PostCvAnalysisError, match="cell__Orientation"):
        analyze_orientation_marginal(malformed)


def test_orientation_marginal_rejects_duplicate_column_labels(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    malformed = pd.concat(
        [formal_orientation_cells, formal_orientation_cells[["group_id"]]],
        axis=1,
    )

    with pytest.raises(PostCvAnalysisError, match="column labels"):
        analyze_orientation_marginal(malformed)


def test_orientation_marginal_rejects_malformed_cell_label(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    malformed = formal_orientation_cells.copy()
    malformed["cell_label"] = malformed["cell_label"].astype(float)
    malformed.loc[0, "cell_label"] = 1.5

    with pytest.raises(PostCvAnalysisError, match="positive integer"):
        analyze_orientation_marginal(malformed)


def test_orientation_marginal_rejects_noncanonical_group_identifier(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    malformed = formal_orientation_cells.copy()
    malformed.loc[0, "group_id"] = " B4_P5 "

    with pytest.raises(PostCvAnalysisError, match="canonical identifiers"):
        analyze_orientation_marginal(malformed)


def test_orientation_marginal_rejects_incomplete_formal_population(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    incomplete = formal_orientation_cells.iloc[:-1]

    with pytest.raises(PostCvAnalysisError, match="19,648"):
        analyze_orientation_marginal(incomplete)


@pytest.mark.parametrize(
    ("column", "old_value", "new_value", "message"),
    [
        ("image_key", "FOV_692", "FOV_000", "693 FOV"),
        ("group_id", "B8_P7", "B4_P5", "9 groups"),
    ],
)
def test_orientation_marginal_rejects_wrong_fov_or_group_population(
    formal_orientation_cells: pd.DataFrame,
    column: str,
    old_value: str,
    new_value: str,
    message: str,
) -> None:
    invalid = formal_orientation_cells.copy()
    selected = invalid[column].eq(old_value)
    if column == "image_key":
        invalid.loc[selected, "cell_label"] += 1_000
    invalid.loc[selected, column] = new_value

    with pytest.raises(PostCvAnalysisError, match=message):
        analyze_orientation_marginal(invalid)


@pytest.mark.parametrize("column", ["cell__Orientation", "group_IDO_score"])
def test_orientation_marginal_rejects_nonfinite_values(
    formal_orientation_cells: pd.DataFrame,
    column: str,
) -> None:
    nonfinite = formal_orientation_cells.copy()
    nonfinite.loc[0, column] = np.inf

    with pytest.raises(PostCvAnalysisError, match="finite"):
        analyze_orientation_marginal(nonfinite)


def test_orientation_marginal_rejects_nonconstant_group_target(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    inconsistent = formal_orientation_cells.copy()
    inconsistent.loc[0, "group_IDO_score"] += 0.01

    with pytest.raises(PostCvAnalysisError, match="group target.*constant"):
        analyze_orientation_marginal(inconsistent)


def test_orientation_bins_are_left_closed_except_for_the_final_right_edge(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    edges = np.linspace(-np.pi / 2.0, np.pi / 2.0, 13)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    cells = formal_orientation_cells.copy()
    baseline_bins = np.arange(CELL_COUNT) % 12
    cells["cell__Orientation"] = midpoints[baseline_bins]
    cells.loc[:12, "cell__Orientation"] = edges
    expected = np.bincount(baseline_bins[13:], minlength=12)
    expected[0] += 1
    expected[1:11] += 1
    expected[11] += 2

    result = analyze_orientation_marginal(cells)

    assert ORIENTATION_RANGE_TOLERANCE == pytest.approx(1e-12)
    np.testing.assert_array_equal(result.bins["n_cells"], expected)


@pytest.mark.parametrize(
    ("boundary", "direction"),
    [(-np.pi / 2.0, -1.0), (np.pi / 2.0, 1.0)],
)
def test_orientation_range_clips_only_values_within_tolerance(
    formal_orientation_cells: pd.DataFrame,
    boundary: float,
    direction: float,
) -> None:
    within_tolerance = formal_orientation_cells.copy()
    within_tolerance.loc[0, "cell__Orientation"] = (
        boundary + direction * ORIENTATION_RANGE_TOLERANCE / 2.0
    )
    within_result = analyze_orientation_marginal(within_tolerance)
    assert within_result.bins["n_cells"].sum() == CELL_COUNT

    outside_tolerance = formal_orientation_cells.copy()
    outside_tolerance.loc[0, "cell__Orientation"] = (
        boundary + direction * ORIENTATION_RANGE_TOLERANCE * 2.0
    )
    with pytest.raises(PostCvAnalysisError, match="axial range"):
        analyze_orientation_marginal(outside_tolerance)


def test_orientation_marginal_rejects_empty_fixed_bin(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    empty_bins = formal_orientation_cells.copy()
    empty_bins["cell__Orientation"] = 0.0

    with pytest.raises(PostCvAnalysisError, match="all 12 fixed bins"):
        analyze_orientation_marginal(empty_bins)


def test_orientation_marginal_rejects_undefined_group_spearman(
    formal_orientation_cells: pd.DataFrame,
) -> None:
    undefined = formal_orientation_cells.copy()
    for group_id in GROUP_IDS:
        selected = undefined["group_id"].eq(group_id)
        group_count = int(selected.sum())
        positive = np.linspace(
            np.pi / 24.0,
            11.0 * np.pi / 24.0,
            group_count // 2,
        )
        symmetric = np.concatenate((-positive, positive))
        if group_count % 2:
            symmetric = np.concatenate((symmetric, [0.0]))
        undefined.loc[selected, "cell__Orientation"] = symmetric

    with pytest.raises(PostCvAnalysisError, match="group-level Spearman"):
        analyze_orientation_marginal(undefined)
