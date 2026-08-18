from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from immunity.exp4.cell_dedup import NUCLEUS_FEATURE_COLUMNS
from immunity.exp4.nucleus_sanity import (
    NUCLEUS_SANITY_COLUMNS,
    build_nucleus_sanity_report,
)


def _retained_cells() -> pd.DataFrame:
    row_count = 19_648
    fov_count = 693
    row_index = np.arange(row_count)
    fov_index = row_index % fov_count
    cell_area = 100.0 + row_index.astype(float)
    nucleus_area = 10.0 + np.sqrt(row_index.astype(float) + 1.0)
    data: dict[str, object] = {
        "image_key": [f"FOV_{index:03d}" for index in fov_index],
        "cell_label": (row_index // fov_count) + 1,
        "cell__Area": cell_area,
    }
    for offset, column in enumerate(NUCLEUS_FEATURE_COLUMNS, start=1):
        if column == "nucleus__area":
            data[column] = nucleus_area
        elif column == "nucleus_cytoplasm_area_ratio":
            data[column] = nucleus_area / (cell_area - nucleus_area)
        else:
            data[column] = offset + (row_index % (offset + 3)) / 100.0
    return pd.DataFrame(data)


def test_nucleus_sanity_reports_full_scope_metrics_and_quintile_fold_changes() -> None:
    retained = _retained_cells()

    report = build_nucleus_sanity_report(retained)

    assert report.columns.tolist() == list(NUCLEUS_SANITY_COLUMNS)
    assert set(report["metric"]) >= {
        "nucleus_area_fraction_gt_one_count",
        "nucleus_area_fraction_quantile",
        "spearman_cell_area_nucleus_area",
        "coefficient_of_variation",
        "pearson_nucleus_ratio_cell_area",
        "nucleus_sphericity_median",
        "cell_area_quintile_median",
        "cell_area_quintile_fold_change",
    }
    impossible = report.loc[
        report["metric"].eq("nucleus_area_fraction_gt_one_count")
    ].iloc[0]
    assert impossible["value"] == 0.0
    assert impossible["n_cells"] == 19_648

    quantiles = report.loc[
        report["metric"].eq("nucleus_area_fraction_quantile"),
        "statistic",
    ].tolist()
    assert quantiles == ["p25", "p50", "p75"]

    cv_rows = report.loc[report["metric"].eq("coefficient_of_variation")]
    assert set(cv_rows["feature"]) == {
        "cell__Area",
        *NUCLEUS_FEATURE_COLUMNS,
    }
    assert len(cv_rows) == 18

    quintile_medians = report.loc[
        report["metric"].eq("cell_area_quintile_median")
    ]
    assert set(quintile_medians["feature"]) == {
        "cell__Area",
        "nucleus__area",
        "nucleus_cytoplasm_area_ratio",
    }
    assert set(quintile_medians["statistic"]) == {
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
    }
    fold_changes = report.loc[
        report["metric"].eq("cell_area_quintile_fold_change")
    ]
    assert set(fold_changes["feature"]) == {
        "cell__Area",
        "nucleus__area",
        "nucleus_cytoplasm_area_ratio",
    }
    expected_cell_fold = (
        quintile_medians.loc[
            quintile_medians["feature"].eq("cell__Area")
            & quintile_medians["statistic"].eq("q5"),
            "value",
        ].iloc[0]
        / quintile_medians.loc[
            quintile_medians["feature"].eq("cell__Area")
            & quintile_medians["statistic"].eq("q1"),
            "value",
        ].iloc[0]
    )
    actual_cell_fold = fold_changes.loc[
        fold_changes["feature"].eq("cell__Area"), "value"
    ].iloc[0]
    assert actual_cell_fold == pytest.approx(expected_cell_fold)

    pearson = report.loc[
        report["metric"].eq("pearson_nucleus_ratio_cell_area"), "value"
    ].iloc[0]
    assert np.isfinite(pearson)

    sphericity = report.loc[
        report["metric"].eq("nucleus_sphericity_median")
    ]
    assert len(sphericity) == 1
    assert sphericity.iloc[0]["feature"] == "nucleus__sphericity"
    assert sphericity.iloc[0]["statistic"] == "p50"
    assert sphericity.iloc[0]["value"] == pytest.approx(
        retained["nucleus__sphericity"].median()
    )
