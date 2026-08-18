from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from immunity.exp4.redundancy import (
    REDUNDANCY_REPORT_COLUMNS,
    FeatureRedundancyError,
    analyze_feature_redundancy,
)


def _x48() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    rng = np.random.default_rng(42)
    canonical = tuple(f"cell__F{index:02d}" for index in range(48))
    values = rng.normal(size=(19_648, 48))
    signal = np.linspace(-2.0, 2.0, num=len(values))
    values[:, 0] = signal
    values[:, 1] = signal * 2.0
    values[:, 2] = -signal
    x = pd.DataFrame(values, columns=canonical)
    authoritative = canonical[:28]
    return x, canonical, authoritative


def test_redundancy_is_x_only_earlier_wins_and_traces_rui28_set_relations() -> None:
    x, canonical, authoritative = _x48()

    result = analyze_feature_redundancy(x, canonical, authoritative)

    assert "y" not in inspect.signature(analyze_feature_redundancy).parameters
    assert result.threshold == 0.9
    assert result.selection_inputs == ("X",)
    assert result.report.columns.tolist() == list(REDUNDANCY_REPORT_COLUMNS)
    assert result.removed_features[:2] == (canonical[1], canonical[2])
    assert canonical[0] in result.retained_features
    assert canonical[1] not in result.retained_features
    assert canonical[2] not in result.retained_features

    decisions = result.report.loc[
        result.report["record_type"].eq("feature_decision")
    ].set_index("feature")
    assert len(decisions) == 48
    assert decisions.loc[canonical[0], "decision"] == "retained"
    assert decisions.loc[canonical[1], "decision"] == "removed"
    assert decisions.loc[canonical[1], "paired_feature"] == canonical[0]
    assert decisions.loc[canonical[1], "abs_pearson_r"] == pytest.approx(1.0)
    assert decisions.loc[canonical[2], "paired_feature"] == canonical[0]

    pair_rows = result.report.loc[
        result.report["record_type"].eq("high_correlation_pair")
    ]
    assert {
        (row.paired_feature, row.feature)
        for row in pair_rows.itertuples(index=False)
    } >= {
        (canonical[0], canonical[1]),
        (canonical[0], canonical[2]),
        (canonical[1], canonical[2]),
    }
    assert result.rui28_intersection == tuple(
        feature for feature in authoritative if feature in result.retained_features
    )
    assert result.rui28_only == (canonical[1], canonical[2])
    assert result.filtered_only == tuple(
        feature
        for feature in result.retained_features
        if feature not in authoritative
    )
    assert set(decisions["set_relation"]) >= {
        "intersection",
        "rui28_only",
        "filtered_only",
    }


def test_redundancy_api_rejects_target_argument_by_construction() -> None:
    x, canonical, authoritative = _x48()

    with pytest.raises(TypeError, match="unexpected keyword argument 'y'"):
        analyze_feature_redundancy(  # type: ignore[call-arg]
            x,
            canonical,
            authoritative,
            y=np.arange(len(x)),
        )


def test_redundancy_requires_the_full_post_border_population() -> None:
    x, canonical, authoritative = _x48()

    with pytest.raises(FeatureRedundancyError, match="19,648"):
        analyze_feature_redundancy(x.iloc[:-1], canonical, authoritative)
