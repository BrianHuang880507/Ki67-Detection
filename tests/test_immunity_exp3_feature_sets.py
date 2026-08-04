from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immunity.exp3.feature_sets import (
    PRIMARY_CELL_FEATURES,
    PRIMARY_FOV_FEATURES,
    aggregate_fov_features,
    validate_phase_predictors,
)


def _manifest() -> pd.DataFrame:
    """建立兩張影像的最小 synthetic manifest。"""
    return pd.DataFrame(
        [
            {
                "image_key": "img-1",
                "b_id": "B4",
                "passage": 5,
                "group_id": "B4_P5",
                "condition_index": 1,
                "condition": "control",
                "ifn_dose": 0.0,
                "tnf_dose": 0.0,
                "fov": 1,
            },
            {
                "image_key": "img-2",
                "b_id": "B7",
                "passage": 6,
                "group_id": "B7_P6",
                "condition_index": 2,
                "condition": "treated",
                "ifn_dose": 1.0,
                "tnf_dose": 2.0,
                "fov": 2,
            },
        ]
    )


def _cells(cell_count: int = 3) -> pd.DataFrame:
    """建立可手算 median 的 synthetic cell-level features。"""
    rows = []
    for image_index, image_key in enumerate(("img-1", "img-2")):
        for cell_index in range(cell_count):
            value = float(image_index * 10 + cell_index + 1)
            row = {
                "image_key": image_key,
                "IDO_score": value,
                **{feature: value for feature in PRIMARY_CELL_FEATURES},
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_primary_registry_contains_exactly_33_features() -> None:
    assert len(PRIMARY_CELL_FEATURES) == 33
    assert len(PRIMARY_FOV_FEATURES) == 33
    assert "cell__perimeter" in PRIMARY_CELL_FEATURES
    assert "nucleus__feret_length" in PRIMARY_CELL_FEATURES
    assert "nucleus_cytoplasm_area_ratio" in PRIMARY_CELL_FEATURES
    assert all(name.endswith("__median") for name in PRIMARY_FOV_FEATURES)


def test_aggregate_fov_features_uses_medians_only_and_writes_internal_cache(
    tmp_path: Path,
) -> None:
    cells = _cells()
    cells.attrs["min_cells_per_image"] = 3
    cells.attrs["feature_cache_dir"] = str(tmp_path / "feature_cache")

    images, registries = aggregate_fov_features(
        cells, _manifest(), ("basic_median",)
    )

    assert images["image_key"].tolist() == ["img-1", "img-2"]
    assert images["cell_count"].tolist() == [3, 3]
    assert images["IDO_score"].tolist() == [2.0, 12.0]
    assert images.loc[0, "cell__area__median"] == 2.0
    assert registries == {"basic_median": list(PRIMARY_FOV_FEATURES)}
    assert not any(column.endswith("__iqr") for column in images.columns)
    assert (tmp_path / "feature_cache" / "image_level_basic.csv").is_file()


def test_aggregate_fov_features_rejects_any_fov_below_minimum() -> None:
    cells = _cells(cell_count=2)
    cells.attrs["min_cells_per_image"] = 3

    with pytest.raises(ValueError, match="至少 3 顆"):
        aggregate_fov_features(cells, _manifest(), ("basic_median",))


def test_aggregate_fov_features_rejects_duplicate_image_key() -> None:
    manifest = pd.concat([_manifest().iloc[[0]], _manifest()], ignore_index=True)

    with pytest.raises(ValueError, match="重複 image_key"):
        aggregate_fov_features(_cells(), manifest, ("basic_median",))


@pytest.mark.parametrize("bad_targets", [[np.inf, 2.0], [5.0, 5.0]])
def test_aggregate_fov_features_rejects_invalid_or_constant_target(
    bad_targets: list[float],
) -> None:
    cells = _cells()
    for image_key, target in zip(("img-1", "img-2"), bad_targets):
        cells.loc[cells["image_key"] == image_key, "IDO_score"] = target

    with pytest.raises(ValueError, match="IDO_score"):
        aggregate_fov_features(cells, _manifest(), ("basic_median",))


@pytest.mark.parametrize(
    "forbidden",
    [
        "IDO_score",
        "IFN_dose",
        "TNF_dose",
        "dose_group",
        "condition",
        "pc_path",
        "filename",
        "delta__cell__area__median",
    ],
)
def test_phase_predictor_whitelist_rejects_target_and_metadata(
    forbidden: str,
) -> None:
    validate_phase_predictors(["cell__area__median"])
    with pytest.raises(ValueError, match="phase-only"):
        validate_phase_predictors([forbidden])


def test_phase_predictor_whitelist_rejects_unregistered_columns() -> None:
    with pytest.raises(ValueError, match="phase-only"):
        validate_phase_predictors(["cell__invented__median"])
