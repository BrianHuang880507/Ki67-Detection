from __future__ import annotations

import pandas as pd
import pytest


GROUPS = ("B4_P5", "B8_P7")


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_key": ("A_1", "A_2", "B_1", "B_2"),
            "group_id": ("B4_P5", "B4_P5", "B8_P7", "B8_P7"),
            "b_id": ("B4", "B4", "B8", "B8"),
            "passage": (5, 5, 7, 7),
        }
    )


def _deduplicated_cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_key": ("A_1", "A_1", "A_1", "A_2", "A_2", "B_1", "B_1", "B_2"),
            "cell_label": (1, 2, 3, 1, 2, 1, 2, 1),
            "IDO_score": (1.0, 2.0, 3.0, 100.0, 200.0, 10.0, 20.0, 30.0),
        }
    )


def _retained_keys() -> list[tuple[str, int]]:
    return [
        ("A_1", 1),
        ("A_1", 2),
        ("A_1", 3),
        ("A_2", 1),
        ("B_1", 1),
        ("B_1", 2),
        ("B_2", 1),
    ]


def test_group_targets_use_post_border_cell_median_and_distinct_counts() -> None:
    from immunity.exp4.targets import GROUP_TARGET_COLUMNS, build_exp4_targets

    result = build_exp4_targets(
        _deduplicated_cells(),
        _manifest(),
        retained_keys=_retained_keys(),
        expected_pre_border_cells=8,
        expected_post_border_cells=7,
        expected_fov_count=4,
        expected_group_keys=GROUPS,
    )
    targets = result.group_targets.set_index("group_id")
    fov_scores = result.fov_ido_scores.set_index("image_key")

    assert tuple(result.group_targets.columns) == GROUP_TARGET_COLUMNS
    assert targets.loc["B4_P5", "fov_count"] == 2
    assert targets.loc["B4_P5", "cells_before"] == 5
    assert targets.loc["B4_P5", "cells_after"] == 4
    # Direct cell median is 2.5; median(FOV medians) would incorrectly be 51.
    assert targets.loc["B4_P5", "group_IDO_score"] == pytest.approx(2.5)
    assert targets.loc["B4_P5", "group_IDO_score_all_cells"] == pytest.approx(3.0)
    assert targets.loc["B4_P5", "delta"] == pytest.approx(-0.5)
    assert targets.loc["B4_P5", "confidence_flag"] == "normal"
    assert targets.loc["B8_P7", "group_IDO_score"] == pytest.approx(20.0)
    assert targets.loc["B8_P7", "confidence_flag"] == "low"
    assert fov_scores.loc["A_1", "FOV_IDO_score"] == pytest.approx(2.0)
    assert fov_scores.loc["A_2", "FOV_IDO_score"] == pytest.approx(100.0)
    assert result.to_dict() == {
        "pre_border_cell_count": 8,
        "post_border_cell_count": 7,
        "fov_count": 4,
        "group_count": 2,
        "group_target_range": 17.5,
    }


def test_sparse_sensitivity_uses_without_minus_with_and_group_thresholds() -> None:
    from immunity.exp4.targets import (
        SENSITIVITY_REPORT_COLUMNS,
        TargetSensitivityBlockedError,
        build_exp4_targets,
        evaluate_sparse_fov_sensitivity,
    )

    targets = build_exp4_targets(
        _deduplicated_cells(),
        _manifest(),
        retained_keys=_retained_keys(),
        expected_pre_border_cells=8,
        expected_post_border_cells=7,
        expected_fov_count=4,
        expected_group_keys=GROUPS,
    )
    result = evaluate_sparse_fov_sensitivity(
        targets,
        ("A_2", "B_2"),
        expected_sparse_fov_count=2,
        expected_sparse_cell_count=1,
    )
    report = result.report.set_index("group_id")

    assert tuple(result.report.columns) == SENSITIVITY_REPORT_COLUMNS
    assert result.sparse_fov_count == 2
    assert result.baseline_target_range == pytest.approx(17.5)
    assert report.loc["B4_P5", "sparse_fov_count"] == 1
    assert report.loc["B4_P5", "target_with_sparse_fovs"] == pytest.approx(2.5)
    assert report.loc["B4_P5", "target_without_sparse_fovs"] == pytest.approx(2.0)
    assert report.loc["B4_P5", "delta_without_minus_with"] == pytest.approx(-0.5)
    assert report.loc["B4_P5", "absolute_delta"] == pytest.approx(0.5)
    assert report.loc["B4_P5", "absolute_delta_pct_of_range"] == pytest.approx(
        100.0 * 0.5 / 17.5
    )
    assert report.loc["B4_P5", "threshold_pct"] == pytest.approx(5.0)
    assert report.loc["B4_P5", "threshold_absolute"] == pytest.approx(0.875)
    assert report.loc["B4_P5", "status"] == "passed"
    assert report.loc["B8_P7", "delta_without_minus_with"] == pytest.approx(-5.0)
    assert report.loc["B8_P7", "threshold_pct"] == pytest.approx(15.0)
    assert report.loc["B8_P7", "status"] == "blocked"
    assert result.blocked is True
    assert result.blocking_groups == ("B8_P7",)
    with pytest.raises(TargetSensitivityBlockedError, match="B8_P7"):
        result.raise_if_blocked()


def test_post_border_frame_ido_must_match_the_deduplicated_master_exactly() -> None:
    from immunity.exp4.targets import TargetAggregationError, build_exp4_targets

    cells = _deduplicated_cells()
    retained = pd.DataFrame(_retained_keys(), columns=("image_key", "cell_label"))
    post = retained.merge(
        cells,
        on=("image_key", "cell_label"),
        how="left",
        validate="one_to_one",
    )
    post.loc[0, "IDO_score"] += 1e-12

    with pytest.raises(TargetAggregationError, match="IDO_score.*完全相同"):
        build_exp4_targets(
            cells,
            _manifest(),
            post_border_cells=post,
            expected_pre_border_cells=8,
            expected_post_border_cells=7,
            expected_fov_count=4,
            expected_group_keys=GROUPS,
        )


def test_sensitivity_equal_to_threshold_passes() -> None:
    from immunity.exp4.targets import (
        build_exp4_targets,
        evaluate_sparse_fov_sensitivity,
    )

    targets = build_exp4_targets(
        _deduplicated_cells(),
        _manifest(),
        retained_keys=_retained_keys(),
        expected_pre_border_cells=8,
        expected_post_border_cells=7,
        expected_fov_count=4,
        expected_group_keys=GROUPS,
    )
    result = evaluate_sparse_fov_sensitivity(
        targets,
        ("A_2", "B_2"),
        expected_sparse_fov_count=2,
        expected_sparse_cell_count=1,
        general_threshold_pct=100.0 * 0.5 / 17.5,
        low_confidence_threshold_pct=100.0 * 5.0 / 17.5,
    )

    assert result.report["status"].tolist() == ["passed", "passed"]
    assert result.blocked is False
    result.raise_if_blocked()


def test_sparse_fov_count_and_roster_are_fail_closed() -> None:
    from immunity.exp4.targets import (
        TargetAggregationError,
        build_exp4_targets,
        evaluate_sparse_fov_sensitivity,
    )

    targets = build_exp4_targets(
        _deduplicated_cells(),
        _manifest(),
        retained_keys=_retained_keys(),
        expected_pre_border_cells=8,
        expected_post_border_cells=7,
        expected_fov_count=4,
        expected_group_keys=GROUPS,
    )

    with pytest.raises(TargetAggregationError, match="sparse FOV count 不符"):
        evaluate_sparse_fov_sensitivity(
            targets,
            ("A_2",),
            expected_sparse_fov_count=2,
            expected_sparse_cell_count=1,
        )
    with pytest.raises(TargetAggregationError, match="sparse FOV roster 不符"):
        evaluate_sparse_fov_sensitivity(
            targets,
            ("A_2", "NOT_A_FOV"),
            expected_sparse_fov_count=2,
            expected_sparse_cell_count=1,
        )


def test_sparse_roster_rejects_same_count_with_a_non_sparse_fov() -> None:
    from immunity.exp4.targets import (
        TargetAggregationError,
        build_exp4_targets,
        evaluate_sparse_fov_sensitivity,
    )

    targets = build_exp4_targets(
        _deduplicated_cells(),
        _manifest(),
        retained_keys=_retained_keys(),
        expected_pre_border_cells=8,
        expected_post_border_cells=7,
        expected_fov_count=4,
        expected_group_keys=GROUPS,
    )

    # Actual cells_after==1 roster is A_2 + B_2; B_1 exists but has two cells.
    with pytest.raises(TargetAggregationError, match="sparse FOV roster 不符"):
        evaluate_sparse_fov_sensitivity(
            targets,
            ("A_2", "B_1"),
            expected_sparse_fov_count=2,
            expected_sparse_cell_count=1,
        )
