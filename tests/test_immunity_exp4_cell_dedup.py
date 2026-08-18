from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _master_rows() -> pd.DataFrame:
    from immunity.exp3.feature_sets import BASIC_GEOMETRY

    rows: list[dict[str, object]] = []
    for nucleus_label, ido_score, nucleus_area, shape_value in (
        (5, 8.0, 10.0, 1.0),
        (2, 4.0, 20.0, 3.0),
    ):
        row: dict[str, object] = {
            "image_key": "FOV_A",
            "cell_label": 1,
            "nucleus_label": nucleus_label,
            "IDO_score": ido_score,
            "cell__area": 100.0,
            "nucleus_cytoplasm_area_ratio": 999.0,
        }
        row.update(
            {
                f"nucleus__{name}": (
                    nucleus_area if name == "area" else shape_value
                )
                for name in BASIC_GEOMETRY
            }
        )
        rows.append(row)

    singleton: dict[str, object] = {
        "image_key": "FOV_B",
        "cell_label": 2,
        "nucleus_label": 7,
        "IDO_score": 9.0,
        "cell__area": 55.0,
        "nucleus_cytoplasm_area_ratio": -1.0,
    }
    singleton.update(
        {
            f"nucleus__{name}": 5.0 if name == "area" else 7.0
            for name in BASIC_GEOMETRY
        }
    )
    rows.append(singleton)
    return pd.DataFrame(rows)


def test_deduplicate_master_cells_aggregates_every_nucleus_feature() -> None:
    from immunity.exp4.cell_dedup import (
        DEDUP_REPORT_COLUMNS,
        NUCLEUS_FEATURE_COLUMNS,
        deduplicate_master_cells,
    )

    result = deduplicate_master_cells(
        _master_rows(),
        expected_raw_rows=3,
        expected_distinct_cells=2,
        expected_multirow_key_count=1,
    )
    cells = result.deduplicated_cells.set_index(["image_key", "cell_label"])
    merged = cells.loc[("FOV_A", 1)]
    singleton = cells.loc[("FOV_B", 2)]

    assert len(result.deduplicated_cells) == 2
    assert result.raw_row_count == 3
    assert result.distinct_cell_count == 2
    assert result.multirow_key_count == 1
    assert result.duplicate_excess_row_count == 1
    assert merged["IDO_score"] == pytest.approx(6.0)
    assert merged["nucleus__area"] == pytest.approx(30.0)
    assert merged["nucleus__compactness"] == pytest.approx(2.0)
    assert merged["nucleus_cytoplasm_area_ratio"] == pytest.approx(3.0 / 7.0)
    assert singleton["nucleus__area"] == pytest.approx(5.0)
    assert singleton["nucleus_cytoplasm_area_ratio"] == pytest.approx(0.1)
    assert set(NUCLEUS_FEATURE_COLUMNS).issubset(result.deduplicated_cells.columns)

    assert tuple(result.dedup_report.columns) == DEDUP_REPORT_COLUMNS
    assert len(result.dedup_report) == 1
    report = result.dedup_report.iloc[0]
    assert report["original_row_count"] == 2
    assert json.loads(report["nucleus_labels_before"]) == [2, 5]
    assert json.loads(report["IDO_score_values_before"]) == [4.0, 8.0]
    assert report["IDO_score_after"] == pytest.approx(6.0)
    assert report["nucleus__area"] == pytest.approx(30.0)
    assert report["nucleus_cytoplasm_area_ratio"] == pytest.approx(3.0 / 7.0)


def test_whole_cell_consistency_reports_the_checked_scope_not_full_by_default() -> None:
    from immunity.exp4.cell_dedup import assert_whole_cell_feature_consistency

    joined_smoke = pd.DataFrame(
        {
            "image_key": ["FOV_A", "FOV_A", "FOV_B"],
            "cell_label": [1, 1, 2],
            "cell__Area": [10.0, 10.0, 20.0],
            "cell__Contrast": [0.25, 0.25, 0.75],
        }
    )

    result = assert_whole_cell_feature_consistency(
        joined_smoke,
        ("cell__Area", "cell__Contrast"),
        scope="smoke_join_5_fovs",
    )

    assert result.scope == "smoke_join_5_fovs"
    assert result.checked_row_count == 3
    assert result.checked_cell_count == 2
    assert result.checked_duplicate_key_count == 1
    assert result.feature_count == 2
    assert result.to_dict() == {
        "scope": "smoke_join_5_fovs",
        "checked_row_count": 3,
        "checked_cell_count": 2,
        "checked_duplicate_key_count": 1,
        "feature_count": 2,
    }


def test_dedup_rejects_first_row_selection_when_whole_cell_features_differ() -> None:
    from immunity.exp4.cell_dedup import CellDedupError, deduplicate_master_cells

    rows = _master_rows()
    rows.loc[1, "cell__area"] = 101.0

    with pytest.raises(CellDedupError, match="feature.*不完全相同"):
        deduplicate_master_cells(
            rows,
            expected_raw_rows=3,
            expected_distinct_cells=2,
            expected_multirow_key_count=1,
        )


def test_whole_cell_consistency_does_not_depend_on_dataframe_index_labels() -> None:
    from immunity.exp4.cell_dedup import assert_whole_cell_feature_consistency

    joined = pd.DataFrame(
        {
            "image_key": ["FOV_A", "FOV_A", "FOV_B"],
            "cell_label": [1, 1, 2],
            "cell__Area": [10.0, 10.0, 20.0],
        },
        index=[7, 7, 7],
    )

    result = assert_whole_cell_feature_consistency(
        joined,
        ("cell__Area",),
        scope="smoke_join",
    )

    assert result.checked_duplicate_key_count == 1


@pytest.mark.parametrize("bad_value", (10.0 + 1e-12, np.inf, np.nan))
def test_whole_cell_consistency_is_exact_and_rejects_nonfinite(
    bad_value: float,
) -> None:
    from immunity.exp4.cell_dedup import (
        CellDedupError,
        assert_whole_cell_feature_consistency,
    )

    joined = pd.DataFrame(
        {
            "image_key": ["FOV_A", "FOV_A"],
            "cell_label": [1, 1],
            "cell__Area": [10.0, bad_value],
        }
    )

    with pytest.raises(CellDedupError):
        assert_whole_cell_feature_consistency(
            joined,
            ("cell__Area",),
            scope="full_join_693_fovs",
        )


@pytest.mark.parametrize(
    ("expected_raw_rows", "expected_distinct_cells"),
    ((23976, 2), (3, 23012)),
)
def test_dedup_snapshot_counts_are_parameterized_fail_closed(
    expected_raw_rows: int,
    expected_distinct_cells: int,
) -> None:
    from immunity.exp4.cell_dedup import CellDedupError, deduplicate_master_cells

    with pytest.raises(CellDedupError, match="count 不符"):
        deduplicate_master_cells(
            _master_rows(),
            expected_raw_rows=expected_raw_rows,
            expected_distinct_cells=expected_distinct_cells,
            expected_multirow_key_count=1,
        )


def test_dedup_rejects_nonpositive_nucleus_cytoplasm_denominator() -> None:
    from immunity.exp4.cell_dedup import CellDedupError, deduplicate_master_cells

    rows = _master_rows()
    rows.loc[rows["image_key"].eq("FOV_A"), "cell__area"] = 30.0

    with pytest.raises(CellDedupError, match="denominator.*大於 0"):
        deduplicate_master_cells(
            rows,
            expected_raw_rows=3,
            expected_distinct_cells=2,
            expected_multirow_key_count=1,
        )


def test_dedup_rejects_wrong_multirow_distribution_with_same_snapshot_counts() -> None:
    from immunity.exp4.cell_dedup import CellDedupError, deduplicate_master_cells

    with pytest.raises(CellDedupError, match="multirow whole-cell key count 不符"):
        deduplicate_master_cells(
            _master_rows(),
            expected_raw_rows=3,
            expected_distinct_cells=2,
            expected_multirow_key_count=2,
        )


def test_full_rui49_consistency_uses_all_canonical_feature_columns() -> None:
    from immunity.exp4.cell_dedup import (
        assert_full_rui49_feature_consistency,
    )
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    joined: dict[str, list[object]] = {
        "image_key": ["FOV_A", "FOV_A"],
        "cell_label": [1, 1],
    }
    joined.update(
        {
            column: [float(index), float(index)]
            for index, column in enumerate(RUI49_FEATURE_COLUMNS, start=1)
        }
    )

    result = assert_full_rui49_feature_consistency(pd.DataFrame(joined))

    assert result.scope == "full_join_693_fovs"
    assert result.feature_count == 49
    assert result.checked_duplicate_key_count == 1


def test_full_rui49_consistency_rejects_one_missing_canonical_column() -> None:
    from immunity.exp4.cell_dedup import (
        CellDedupError,
        assert_full_rui49_feature_consistency,
    )
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    missing_column = RUI49_FEATURE_COLUMNS[-1]
    joined: dict[str, list[object]] = {
        "image_key": ["FOV_A", "FOV_A"],
        "cell_label": [1, 1],
    }
    joined.update(
        {
            column: [float(index), float(index)]
            for index, column in enumerate(RUI49_FEATURE_COLUMNS[:-1], start=1)
        }
    )

    with pytest.raises(CellDedupError, match=missing_column):
        assert_full_rui49_feature_consistency(pd.DataFrame(joined))


def test_full_rui49_canonical_columns_cannot_be_overridden_by_caller() -> None:
    from immunity.exp4.cell_dedup import assert_full_rui49_feature_consistency
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    fake_columns = (*RUI49_FEATURE_COLUMNS[:-1], "cell__fake_replacement")
    joined: dict[str, list[object]] = {
        "image_key": ["FOV_A", "FOV_A"],
        "cell_label": [1, 1],
    }
    joined.update(
        {
            column: [float(index), float(index)]
            for index, column in enumerate(fake_columns, start=1)
        }
    )

    with pytest.raises(TypeError, match="canonical_feature_columns"):
        assert_full_rui49_feature_consistency(
            pd.DataFrame(joined),
            canonical_feature_columns=fake_columns,
        )
