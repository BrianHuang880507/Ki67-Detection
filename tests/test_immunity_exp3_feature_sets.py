from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from immunity.exp3.feature_sets import (
    BASIC_MEDIAN_IQR_FOV_FEATURES,
    FEATURE_SET_REGISTRY,
    FeatureSetSpec,
    PAPER_STYLE_CELL_FEATURES,
    PAPER_STYLE_EXTRA_FEATURES,
    PAPER_STYLE_FOV_FEATURES,
    PRIMARY_CELL_FEATURES,
    PRIMARY_FOV_FEATURES,
    aggregate_fov_features,
    calculate_delta_signatures,
    validate_phase_predictors,
)
from immunity.exp3.phase_features import (
    extract_features_from_arrays,
    paper_style_extras,
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


def test_secondary_feature_counts_are_fixed() -> None:
    assert FEATURE_SET_REGISTRY["basic_median"].predictor_count == 33
    assert FEATURE_SET_REGISTRY["basic_median_iqr"].predictor_count == 66
    assert len(PAPER_STYLE_CELL_FEATURES) == 93


def test_paper_style_registry_is_exact_label_free_approximation() -> None:
    expected_extras = (
        *(f"cell__zernike_{index:02d}" for index in range(25)),
        *(f"nucleus__zernike_{index:02d}" for index in range(25)),
        "nucleus_cytoplasm_mean_ratio",
        "nucleus_cytoplasm_intden_ratio",
        "nucleus_cytoplasm_raw_intden_ratio",
        "nucleus_cell_intden_ratio",
        "nucleus_cytoplasm_entropy_difference",
        "nucleus_cytoplasm_cv_difference",
        "nucleus_centroid_offset",
        "cell__phase_intensity_cv",
        "nucleus__phase_intensity_cv",
        "cytoplasm__phase_intensity_cv",
    )

    assert PAPER_STYLE_EXTRA_FEATURES == expected_extras
    assert PAPER_STYLE_CELL_FEATURES == (*PRIMARY_CELL_FEATURES, *expected_extras)
    assert PAPER_STYLE_FOV_FEATURES == tuple(
        f"{name}__median" for name in PAPER_STYLE_CELL_FEATURES
    )
    assert (
        FEATURE_SET_REGISTRY["paper_style_median"].replication_scope
        == "partial_label_free_approximation"
    )


def test_paper_style_extras_use_phase_regions_and_normalized_centroid(
    monkeypatch,
) -> None:
    phase = np.arange(144, dtype=float).reshape(12, 12) + 1.0
    cell = np.zeros((12, 12), dtype=bool)
    nucleus = np.zeros((12, 12), dtype=bool)
    cell[2:10, 2:10] = True
    nucleus[4:7, 4:7] = True

    def fake_zernike(_signal: np.ndarray, mask: np.ndarray) -> list[float]:
        offset = 0.0 if np.count_nonzero(mask) == 64 else 100.0
        return [offset + float(index) for index in range(25)]

    monkeypatch.setattr(
        "immunity.exp3.phase_features._zernike_feature_values_python",
        fake_zernike,
    )

    extras = paper_style_extras(phase, cell, nucleus)

    cytoplasm = cell & ~nucleus
    cell_values = phase[cell]
    nucleus_values = phase[nucleus]
    cytoplasm_values = phase[cytoplasm]
    assert tuple(extras) == PAPER_STYLE_EXTRA_FEATURES
    assert extras["cell__zernike_00"] == 0.0
    assert extras["cell__zernike_24"] == 24.0
    assert extras["nucleus__zernike_00"] == 100.0
    assert extras["nucleus__zernike_24"] == 124.0
    assert extras["nucleus_cytoplasm_mean_ratio"] == pytest.approx(
        nucleus_values.mean() / cytoplasm_values.mean()
    )
    assert extras["nucleus_cell_intden_ratio"] == pytest.approx(
        nucleus_values.sum() / cell_values.sum()
    )
    assert extras["nucleus_cytoplasm_entropy_difference"] == pytest.approx(
        np.log2(9.0) - np.log2(55.0)
    )
    expected_offset = np.hypot(5.0 - 5.5, 5.0 - 5.5) / np.sqrt(64.0 / np.pi)
    assert extras["nucleus_centroid_offset"] == pytest.approx(expected_offset)
    assert extras["cell__phase_intensity_cv"] == pytest.approx(
        cell_values.std() / cell_values.mean()
    )
    assert extras["nucleus__phase_intensity_cv"] == pytest.approx(
        nucleus_values.std() / nucleus_values.mean()
    )
    assert extras["cytoplasm__phase_intensity_cv"] == pytest.approx(
        cytoplasm_values.std() / cytoplasm_values.mean()
    )


def test_primary_path_does_not_compute_paper_features(monkeypatch) -> None:
    forbidden = Mock(side_effect=AssertionError("paper extras must stay lazy"))
    monkeypatch.setattr("immunity.exp3.phase_features.paper_style_extras", forbidden)
    phase = np.arange(144, dtype=float).reshape(12, 12)
    ido = np.full((12, 12), 5.0)
    cell = np.zeros((12, 12), dtype=np.int32)
    nucleus = np.zeros((12, 12), dtype=np.int32)
    cell[2:10, 2:10] = 1
    nucleus[4:7, 4:7] = 1

    extract_features_from_arrays(
        "img",
        phase,
        ido,
        cell,
        nucleus,
        0.05,
        enabled_feature_sets=["basic_median"],
    )

    forbidden.assert_not_called()


def test_paper_style_path_adds_exactly_the_registered_extras(monkeypatch) -> None:
    phase = np.arange(144, dtype=float).reshape(12, 12)
    ido = np.full((12, 12), 5.0)
    cell = np.zeros((12, 12), dtype=np.int32)
    nucleus = np.zeros((12, 12), dtype=np.int32)
    cell[2:10, 2:10] = 1
    nucleus[4:7, 4:7] = 1
    expected = {
        feature: float(index + 1)
        for index, feature in enumerate(PAPER_STYLE_EXTRA_FEATURES)
    }
    monkeypatch.setattr(
        "immunity.exp3.phase_features.paper_style_extras",
        lambda *_args: expected,
    )

    rows, _ = extract_features_from_arrays(
        "img",
        phase,
        ido,
        cell,
        nucleus,
        0.05,
        enabled_feature_sets=["paper_style_median"],
    )

    assert len(rows) == 1
    assert {feature: rows[0][feature] for feature in expected} == expected


def test_delta_is_descriptive_and_never_a_model_feature() -> None:
    conditions = [
        (1, 0.0, 0.0),
        (2, 25.0, 0.0),
        (3, 50.0, 0.0),
        (4, 100.0, 0.0),
        (5, 0.0, 25.0),
        (6, 0.0, 50.0),
        (7, 25.0, 25.0),
        (8, 25.0, 50.0),
    ]
    rows = [
        {
            "group_id": group_id,
            "condition_index": index,
            "ifn_dose": ifn,
            "tnf_dose": tnf,
            "cell__area__median": float(index + group_offset),
        }
        for group_offset, group_id in enumerate(["B4_P5", "B7_P5"])
        for index, ifn, tnf in conditions
    ]

    delta = calculate_delta_signatures(
        pd.DataFrame(rows), ["cell__area__median"]
    )

    assert set(delta["comparison_type"]) == {"group_level_unpaired"}
    assert delta.groupby("group_id").size().eq(7).all()
    first = delta.loc[
        (delta["group_id"] == "B4_P5")
        & (delta["contrast_id"] == "IFN25_vs_0_at_TNF0")
    ].iloc[0]
    assert first["control_median"] == 1.0
    assert first["treated_median"] == 2.0
    assert first["delta_raw"] == 1.0
    assert first["global_iqr"] == 4.0
    assert first["delta_scaled_by_global_iqr"] == 0.25
    with pytest.raises(ValueError, match="phase-only"):
        validate_phase_predictors(["delta__cell__area"])


def test_delta_requires_all_eight_predeclared_conditions() -> None:
    rows = [
        {
            "group_id": "B4_P5",
            "ifn_dose": 0.0,
            "tnf_dose": 0.0,
            "cell__area__median": 1.0,
        }
    ]

    with pytest.raises(ValueError, match="缺少.*條件"):
        calculate_delta_signatures(pd.DataFrame(rows), ["cell__area__median"])


def test_delta_emits_seven_contrasts_for_each_of_nine_groups() -> None:
    conditions = (
        (0.0, 0.0),
        (25.0, 0.0),
        (50.0, 0.0),
        (100.0, 0.0),
        (0.0, 25.0),
        (0.0, 50.0),
        (25.0, 25.0),
        (25.0, 50.0),
    )
    rows = [
        {
            "group_id": f"group-{group_index}",
            "ifn_dose": ifn,
            "tnf_dose": tnf,
            "cell__area__median": float(condition_index + group_index),
        }
        for group_index in range(9)
        for condition_index, (ifn, tnf) in enumerate(conditions)
    ]

    delta = calculate_delta_signatures(
        pd.DataFrame(rows), ["cell__area__median"]
    )

    assert len(delta) == 9 * 7
    assert delta.groupby("group_id").size().eq(7).all()


def test_aggregate_fov_features_uses_medians_only_and_writes_internal_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    cells = _cells()
    cells.attrs["min_cells_per_image"] = 3
    cells.attrs["exp3_output_dir"] = str(output_dir)

    images, registries = aggregate_fov_features(
        cells, _manifest(), ("basic_median",)
    )

    assert images["image_key"].tolist() == ["img-1", "img-2"]
    assert images["cell_count"].tolist() == [3, 3]
    assert images["IDO_score"].tolist() == [2.0, 12.0]
    assert images.loc[0, "cell__area__median"] == 2.0
    assert registries == {"basic_median": list(PRIMARY_FOV_FEATURES)}
    assert not any(column.endswith("__iqr") for column in images.columns)
    assert (
        output_dir / "feature_cache" / "image_level_basic.csv"
    ).is_file()


def test_basic_median_iqr_writes_isolated_66_feature_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    cells = _cells()
    cells.attrs["min_cells_per_image"] = 3
    cells.attrs["exp3_output_dir"] = str(output_dir)

    images, registries = aggregate_fov_features(
        cells, _manifest(), ("basic_median_iqr",)
    )

    assert registries == {
        "basic_median_iqr": list(BASIC_MEDIAN_IQR_FOV_FEATURES)
    }
    assert images.loc[0, "cell__area__median"] == 2.0
    assert images.loc[0, "cell__area__iqr"] == 1.0
    assert len(registries["basic_median_iqr"]) == 66
    assert (
        output_dir
        / "feature_cache"
        / "image_level_basic_median_iqr.csv"
    ).is_file()
    assert not (output_dir / "feature_cache" / "image_level_basic.csv").exists()
    metadata = json.loads((output_dir / "feature_sets.json").read_text("utf-8"))
    assert metadata["basic_median_iqr"] == {
        "predictor_columns": list(BASIC_MEDIAN_IQR_FOV_FEATURES),
        "predictor_count": 66,
        "aggregation": "median_iqr",
        "replication_scope": "secondary_phase_only",
    }


def test_paper_style_median_writes_isolated_93_feature_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    cells = _cells()
    for index, feature in enumerate(PAPER_STYLE_EXTRA_FEATURES):
        cells[feature] = float(index + 1)
    cells.attrs["min_cells_per_image"] = 3
    cells.attrs["exp3_output_dir"] = str(output_dir)

    images, registries = aggregate_fov_features(
        cells, _manifest(), ("paper_style_median",)
    )

    assert len(registries["paper_style_median"]) == 93
    assert images.loc[0, "cell__zernike_00__median"] == 1.0
    assert (
        output_dir / "feature_cache" / "image_level_paper_style_median.csv"
    ).is_file()
    assert not (output_dir / "feature_cache" / "image_level_basic.csv").exists()
    metadata = json.loads((output_dir / "feature_sets.json").read_text("utf-8"))
    assert (
        metadata["paper_style_median"]["replication_scope"]
        == "partial_label_free_approximation"
    )
    assert metadata["paper_style_median"]["predictor_count"] == 93


def test_enabled_registry_rejects_duplicate_predictor_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    cells = _cells()
    cells.attrs["exp3_output_dir"] = str(output_dir)
    invalid = FeatureSetSpec(
        name="basic_median",
        cell_features=PRIMARY_CELL_FEATURES,
        predictor_columns=(*PRIMARY_FOV_FEATURES[:-1], PRIMARY_FOV_FEATURES[0]),
        aggregation="median",
        replication_scope="primary_phase_only",
    )
    monkeypatch.setitem(FEATURE_SET_REGISTRY, "basic_median", invalid)

    with pytest.raises(ValueError, match="重複"):
        aggregate_fov_features(cells, _manifest(), ("basic_median",))

    assert not output_dir.exists()


def test_aggregate_fov_features_rejects_legacy_cache_destination(
    tmp_path: Path,
) -> None:
    cells = _cells()
    cells.attrs["exp3_output_dir"] = str(tmp_path / "legacy-results")

    with pytest.raises(ValueError, match="Exp3 output"):
        aggregate_fov_features(cells, _manifest(), ("basic_median",))

    assert not (tmp_path / "legacy-results").exists()


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
