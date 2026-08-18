from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_rui49_feature_columns_have_authoritative_order() -> None:
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    expected_suffixes = (
        "Area",
        "BoundingBoxArea",
        "BoundingBoxMinimum_X",
        "BoundingBoxMinimum_Y",
        "BoundingBoxMaximum_X",
        "BoundingBoxMaximum_Y",
        "Center_X",
        "Center_Y",
        "Compactness",
        "ConvexArea",
        "Eccentricity",
        "EquivalentDiameter",
        "Extent",
        "FormFactor",
        "MajorAxisLength",
        "MaxFeretDiameter",
        "MaximumRadius",
        "MeanRadius",
        "MedianRadius",
        "MinFeretDiameter",
        "MinorAxisLength",
        "Orientation",
        "Perimeter",
        "Solidity",
        "IntegratedIntensity",
        "MeanIntensity",
        "StdIntensity",
        "MinIntensity",
        "MaxIntensity",
        "IntegratedIntensityEdge",
        "MeanIntensityEdge",
        "StdIntensityEdge",
        "MinIntensityEdge",
        "MaxIntensityEdge",
        "MassDisplacement",
        "MADIntensity",
        "AngularSecondMoment",
        "Contrast",
        "Correlation",
        "Variance",
        "InverseDifferenceMoment",
        "SumAverage",
        "SumVariance",
        "SumEntropy",
        "Entropy",
        "DifferenceVariance",
        "DifferenceEntropy",
        "InfoMeas1",
        "InfoMeas2",
    )

    assert RUI49_FEATURE_COLUMNS == tuple(
        f"cell__{suffix}" for suffix in expected_suffixes
    )
    assert len(RUI49_FEATURE_COLUMNS) == 49
    assert all("nucleus" not in column.lower() for column in RUI49_FEATURE_COLUMNS)


def test_preprocess_phase_image_converts_rgb_and_subtracts_background() -> None:
    from immunity.exp4.rui_features import (
        BACKGROUND_RADIUS,
        preprocess_phase_image,
    )

    rgb = np.zeros((21, 21, 3), dtype=np.uint8)
    rgb[..., :] = 20
    rgb[10, 10, 0] = 255

    corrected = preprocess_phase_image(rgb)

    assert BACKGROUND_RADIUS == 50
    assert corrected.shape == rgb.shape[:2]
    assert corrected.dtype == np.float64
    assert np.isfinite(corrected).all()
    assert float(corrected.min()) >= 0.0
    assert corrected[10, 10] > corrected[0, 0]


def test_quantize_masked_texture_reserves_zero_for_padding() -> None:
    from immunity.exp4.rui_features import quantize_masked_texture

    signal = np.array(
        [
            [0.0, 0.5, 99.0],
            [1.0, 2.0, 88.0],
        ],
        dtype=np.float64,
    )
    mask = np.array(
        [
            [True, True, False],
            [True, True, False],
        ]
    )

    quantized = quantize_masked_texture(signal, mask)

    np.testing.assert_array_equal(
        quantized,
        np.array(
            [
                [1, 64, 0],
                [128, 255, 0],
            ],
            dtype=np.uint8,
        ),
    )
    assert np.all(np.diff(quantized[mask]) >= 0)


def test_extract_rui49_features_maps_authoritative_geometry_sources() -> None:
    from immunity.exp4.rui_features import (
        RUI49_FEATURE_COLUMNS,
        extract_rui49_features,
    )
    from ki67dtc.cell_anal import (
        _geometry_from_measurements,
        _measure_roi_with_python,
    )
    from skimage.measure import regionprops

    phase = np.arange(9 * 12, dtype=np.float64).reshape(9, 12) + 1.0
    labels = np.zeros(phase.shape, dtype=np.int32)
    labels[2:6, 3:8] = 7

    features = extract_rui49_features(phase, labels)
    row = features.iloc[0]
    authoritative = _geometry_from_measurements(
        _measure_roi_with_python(phase, labels == 7)
    )
    region = regionprops(labels)[0]

    assert features.columns.tolist() == ["cell_label", *RUI49_FEATURE_COLUMNS]
    assert features["cell_label"].tolist() == [7]
    assert row["cell__Area"] == 20.0
    assert row["cell__BoundingBoxArea"] == 20.0
    assert row["cell__BoundingBoxMinimum_X"] == 3.0
    assert row["cell__BoundingBoxMinimum_Y"] == 2.0
    assert row["cell__BoundingBoxMaximum_X"] == 8.0
    assert row["cell__BoundingBoxMaximum_Y"] == 6.0
    assert row["cell__Center_X"] == pytest.approx(5.0)
    assert row["cell__Center_Y"] == pytest.approx(3.5)
    assert row["cell__ConvexArea"] == 20.0
    assert row["cell__EquivalentDiameter"] == pytest.approx(
        np.sqrt(4.0 * 20.0 / np.pi)
    )
    expected_mapping = {
        "Compactness": "compactness",
        "Extent": "extent",
        "FormFactor": "sphericity",
        "MajorAxisLength": "major_axis_length",
        "MaximumRadius": "maximum_radius",
        "MeanRadius": "mean_radius",
        "MedianRadius": "median_radius",
        "MinorAxisLength": "minor_axis_length",
        "Perimeter": "perimeter",
        "Solidity": "solidity",
    }
    for rui_name, helper_name in expected_mapping.items():
        assert row[f"cell__{rui_name}"] == pytest.approx(
            authoritative[helper_name]
        )
    assert row["cell__Eccentricity"] == pytest.approx(region.eccentricity)
    assert row["cell__MinFeretDiameter"] == pytest.approx(3.0)
    assert row["cell__MaxFeretDiameter"] == pytest.approx(5.0)
    assert 0.0 <= row["cell__MinFeretDiameter"] <= row["cell__MaxFeretDiameter"]


def test_extract_rui49_features_repairs_curved_fitellipse_feret_failure() -> None:
    import cv2

    from immunity.exp4.rui_features import extract_rui49_features

    labels = np.zeros((100, 100), dtype=np.uint8)
    x_coords = np.arange(5, 85, dtype=np.int32)
    y_coords = (5 + (x_coords - 5) ** 2 / (80 * 1.2)).astype(np.int32)
    curve = np.column_stack((x_coords, y_coords)).astype(np.int32)
    cv2.polylines(labels, [curve], False, 1, 1)

    row = extract_rui49_features(
        np.ones(labels.shape, dtype=np.float64),
        labels.astype(np.int32),
    ).iloc[0]

    assert row["cell__MinFeretDiameter"] == pytest.approx(
        13.196033314664493
    )
    assert row["cell__MaxFeretDiameter"] == pytest.approx(
        102.30347012687302
    )
    assert 0.0 <= row["cell__MinFeretDiameter"] <= row["cell__MaxFeretDiameter"]


def test_extract_rui49_features_uses_one_hull_for_both_feret_diameters() -> None:
    from immunity.exp4.rui_features import extract_rui49_features

    labels = np.zeros((40, 40), dtype=np.int32)
    for min_row, min_col in ((2, 2), (2, 30), (30, 15)):
        labels[min_row : min_row + 3, min_col : min_col + 3] = 1

    row = extract_rui49_features(
        np.ones(labels.shape, dtype=np.float64),
        labels,
    ).iloc[0]

    assert row["cell__MinFeretDiameter"] == pytest.approx(
        27.3888419258943
    )
    assert row["cell__MaxFeretDiameter"] == pytest.approx(
        34.48187929913333
    )
    assert 0.0 <= row["cell__MinFeretDiameter"] <= row["cell__MaxFeretDiameter"]


def test_extract_rui49_features_defines_degenerate_feret_without_fallback() -> None:
    from immunity.exp4.rui_features import (
        extract_rui49_features_with_diagnostics,
    )

    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1, 1] = 5
    labels[4, 2:6] = 9

    result = extract_rui49_features_with_diagnostics(
        np.ones(labels.shape, dtype=np.float64),
        labels,
    )

    assert result.features["cell__MinFeretDiameter"].tolist() == [0.0, 0.0]
    assert result.features["cell__MaxFeretDiameter"].tolist() == [0.0, 3.0]
    assert result.fallback_flags["cell__MinFeretDiameter"].tolist() == [
        False,
        False,
    ]
    assert result.fallback_flags["cell__MaxFeretDiameter"].tolist() == [
        False,
        False,
    ]


def test_extract_rui49_features_fails_closed_on_feret_invariant(
    monkeypatch,
) -> None:
    from immunity.exp4 import rui_features

    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1:6, 1:6] = 1
    monkeypatch.setattr(
        rui_features,
        "_feret_diameters",
        lambda _mask: (2.0, 1.0),
    )

    with pytest.raises(
        ValueError,
        match=r"cell_label=1.*MinFeretDiameter=2\.0.*MaxFeretDiameter=1\.0",
    ):
        rui_features.extract_rui49_features(
            np.ones(labels.shape, dtype=np.float64),
            labels,
        )


def test_extract_rui49_features_measures_cell_and_edge_intensity() -> None:
    from immunity.exp4.rui_features import extract_rui49_features

    phase = np.zeros((5, 5), dtype=np.float64)
    phase[1:4, 1:4] = np.arange(1.0, 10.0).reshape(3, 3)
    labels = np.zeros(phase.shape, dtype=np.int32)
    labels[1:4, 1:4] = 3

    row = extract_rui49_features(phase, labels).iloc[0]

    assert row["cell__IntegratedIntensity"] == 45.0
    assert row["cell__MeanIntensity"] == 5.0
    assert row["cell__StdIntensity"] == pytest.approx(np.sqrt(60.0 / 9.0))
    assert row["cell__MinIntensity"] == 1.0
    assert row["cell__MaxIntensity"] == 9.0
    assert row["cell__IntegratedIntensityEdge"] == 40.0
    assert row["cell__MeanIntensityEdge"] == 5.0
    assert row["cell__StdIntensityEdge"] == pytest.approx(np.sqrt(60.0 / 8.0))
    assert row["cell__MinIntensityEdge"] == 1.0
    assert row["cell__MaxIntensityEdge"] == 9.0
    assert row["cell__MassDisplacement"] == pytest.approx(
        np.hypot(2.4 - 2.0, 96.0 / 45.0 - 2.0)
    )
    assert row["cell__MADIntensity"] == 2.0


def test_extract_rui49_features_uses_mahotas_distance_three_four_direction_mean() -> None:
    from immunity.exp4.rui_features import (
        HARALICK_FEATURE_NAMES,
        extract_rui49_features,
    )

    phase = np.arange(1.0, 65.0, dtype=np.float64).reshape(8, 8)
    labels = np.ones(phase.shape, dtype=np.int32)
    expected = np.array(
        [
            0.016250000000000004,
            7180.137499999999,
            0.21420609281871283,
            4778.396071875,
            0.0017745326082746911,
            256.0225,
            11933.446787500005,
            4.982892142331044,
            5.982892142331044,
            0.003229370117187497,
            0.41245102055336447,
            -0.9516389411251871,
            0.9999900649038833,
        ]
    )

    row = extract_rui49_features(phase, labels).iloc[0]
    actual = row[
        [f"cell__{name}" for name in HARALICK_FEATURE_NAMES]
    ].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_haralick_zero_reservation_differs_from_legacy_texture() -> None:
    from immunity.exp4.rui_features import (
        HARALICK_FEATURE_NAMES,
        extract_rui49_features_with_diagnostics,
    )

    phase = np.arange(64, dtype=np.float64).reshape(8, 8)
    labels = np.ones(phase.shape, dtype=np.int32)

    result = extract_rui49_features_with_diagnostics(
        phase,
        labels,
        include_legacy_texture=True,
    )
    columns = [f"cell__{name}" for name in HARALICK_FEATURE_NAMES]
    formal = result.features.loc[0, columns].to_numpy(dtype=np.float64)
    legacy = result.legacy_texture_features.loc[
        0, columns
    ].to_numpy(dtype=np.float64)

    assert result.cell_count == 1
    assert list(result.legacy_texture_features.columns) == [
        "cell_label",
        *columns,
    ]
    assert np.isfinite(formal).all()
    assert np.isfinite(legacy).all()
    assert not np.allclose(formal, legacy)
    assert not any("legacy" in column for column in result.features.columns)


def test_extract_rui49_features_returns_finite_rows_for_degenerate_cells() -> None:
    from immunity.exp4.rui_features import (
        HARALICK_FEATURE_NAMES,
        RUI49_FEATURE_COLUMNS,
        extract_rui49_features,
    )

    phase = np.zeros((7, 7), dtype=np.float64)
    labels = np.zeros(phase.shape, dtype=np.int32)
    labels[1, 1] = 5
    labels[4:6, 4:6] = 9

    features = extract_rui49_features(phase, labels)

    assert features["cell_label"].tolist() == [5, 9]
    assert np.isfinite(
        features.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).all()
    assert (
        features.loc[:, [f"cell__{name}" for name in HARALICK_FEATURE_NAMES]]
        .to_numpy(dtype=np.float64)
        == 0.0
    ).all()


def test_fallback_diagnostics_count_each_rui_feature_exactly() -> None:
    from immunity.exp4.rui_features import (
        FALLBACK_SCOPE_PRE_BORDER,
        extract_rui49_features_with_diagnostics,
    )

    phase = np.zeros((7, 7), dtype=np.float64)
    labels = np.zeros(phase.shape, dtype=np.int32)
    labels[1, 1] = 5
    labels[4:6, 4:6] = 9

    result = extract_rui49_features_with_diagnostics(phase, labels)
    nonzero_counts = {
        feature: count
        for feature, count in result.fallback_counts.items()
        if count
    }

    assert result.cell_count == 2
    assert result.fallback_scope == FALLBACK_SCOPE_PRE_BORDER
    assert nonzero_counts == {
        "cell__FormFactor": 1,
    }
    assert result.fallback_rates["cell__Eccentricity"] == 0.0
    assert result.fallback_rates["cell__MinFeretDiameter"] == 0.0
    assert result.fallback_rates["cell__Area"] == 0.0
    assert result.fallback_flags["cell_label"].tolist() == [5, 9]
    assert result.fallback_flags["cell__Eccentricity"].tolist() == [False, False]
    assert result.fallback_flags["cell__MinFeretDiameter"].tolist() == [
        False,
        False,
    ]
    assert result.fallback_flags["cell__Area"].tolist() == [False, False]


def test_extract_rui49_features_ignores_phase_pixels_outside_cell() -> None:
    from immunity.exp4.rui_features import (
        RUI49_FEATURE_COLUMNS,
        extract_rui49_features,
    )

    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:10, 2:10] = 4
    labels[2, 2] = 0
    baseline = np.arange(144, dtype=np.float64).reshape(12, 12) + 1.0
    changed = baseline.copy()
    changed[labels == 0] = 1_000_000.0

    first = extract_rui49_features(baseline, labels)
    second = extract_rui49_features(changed, labels)

    np.testing.assert_allclose(
        first.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(dtype=np.float64),
        second.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(dtype=np.float64),
    )


def test_extract_rui49_features_is_whole_cell_only_and_fails_closed() -> None:
    from immunity.exp4.rui_features import extract_rui49_features

    assert tuple(inspect.signature(extract_rui49_features).parameters) == (
        "phase_signal",
        "cell_labels",
    )
    with pytest.raises(ValueError, match="二維"):
        extract_rui49_features(
            np.zeros((3, 3, 1), dtype=np.float64),
            np.zeros((3, 3), dtype=np.int32),
        )
    with pytest.raises(ValueError, match="相同尺寸"):
        extract_rui49_features(
            np.zeros((3, 3), dtype=np.float64),
            np.zeros((4, 3), dtype=np.int32),
        )
    with pytest.raises(ValueError, match="非負"):
        extract_rui49_features(
            -np.ones((3, 3), dtype=np.float64),
            np.zeros((3, 3), dtype=np.int32),
        )
    with pytest.raises(ValueError, match="整數"):
        extract_rui49_features(
            np.ones((3, 3), dtype=np.float64),
            np.zeros((3, 3), dtype=np.float64),
        )


def _write_consistency_smoke_fixture(
    tmp_path: Path,
    master_rows: list[dict[str, int | str]],
) -> tuple[Path, Path, Path, Path, Path]:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    manifest_rows = []
    for index in range(1, 6):
        image_key = f"FOV_{index}"
        pc_path = image_dir / f"{image_key}.jpg"
        Image.fromarray(
            np.full((16, 16, 3), 32 + index, dtype=np.uint8),
            mode="RGB",
        ).save(pc_path)
        labels = np.zeros((16, 16), dtype=np.int32)
        labels[3:13, 3:13] = 1
        np.savez_compressed(mask_dir / f"{image_key}.npz", cell_mask=labels)
        manifest_rows.append({"image_key": image_key, "pc_path": str(pc_path)})
    manifest_path = tmp_path / "data_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    cell_level_path = tmp_path / "cell_level_basic.csv"
    pd.DataFrame(master_rows).to_csv(cell_level_path, index=False)
    return (
        manifest_path,
        mask_dir,
        cell_level_path,
        tmp_path / "feature_smoke_report.json",
        tmp_path / "run_metadata.json",
    )


def test_run_rui49_smoke_selects_first_five_and_merges_metadata(
    tmp_path: Path,
) -> None:
    from immunity.exp4.rui_features import (
        HARALICK_FEATURE_NAMES,
        RUI49_FEATURE_COLUMNS,
        run_rui49_smoke,
    )

    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks" / "B4_P5"
    image_dir.mkdir()
    mask_dir.mkdir(parents=True)
    rows = []
    for index in range(6, 0, -1):
        image_key = f"B4_P5_C01_F{index:02d}"
        pc_path = image_dir / f"{image_key}.jpg"
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(16, dtype=np.uint8)[None, :] * 8
        rgb[..., 1] = np.arange(16, dtype=np.uint8)[:, None] * 8
        rgb[..., 2] = index
        Image.fromarray(rgb, mode="RGB").save(pc_path)
        cell_mask = np.zeros((16, 16), dtype=np.int32)
        cell_mask[3:13, 3:13] = index
        np.savez_compressed(
            mask_dir / f"{image_key}.npz",
            cell_mask=cell_mask,
            nucleus_mask=np.full_like(cell_mask, 999),
        )
        rows.append({"image_key": image_key, "pc_path": str(pc_path)})
    manifest_path = tmp_path / "data_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    cell_level_path = tmp_path / "cell_level_basic.csv"
    master_rows = [
        {
            "image_key": f"B4_P5_C01_F{index:02d}",
            "cell_label": index,
            "nucleus_label": index,
        }
        for index in range(1, 6)
    ]
    master_rows.append(
        {
            "image_key": "B4_P5_C01_F01",
            "cell_label": 1,
            "nucleus_label": 99,
        }
    )
    pd.DataFrame(master_rows).to_csv(cell_level_path, index=False)
    report_path = tmp_path / "smoke_report.json"
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "packages": {"mahotas": "keep-me"},
                "status": "snapshot_validated",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "environment_packages.json").write_text(
        json.dumps(
            [
                {"name": "python", "version": "3.10.20"},
                {"name": "mahotas", "version": "1.4.18"},
                {"name": "full-environment-only", "version": "7.0"},
            ]
        ),
        encoding="utf-8-sig",
    )

    report = run_rui49_smoke(
        manifest_path=manifest_path,
        masks_dir=tmp_path / "masks",
        report_path=report_path,
        metadata_path=metadata_path,
        image_limit=5,
        cell_level_path=cell_level_path,
    )

    expected_keys = [f"B4_P5_C01_F{index:02d}" for index in range(1, 6)]
    assert report["status"] == "passed"
    assert report["image_keys"] == expected_keys
    assert [image["image_key"] for image in report["images"]] == expected_keys
    assert [image["cell_count"] for image in report["images"]] == [1] * 5
    assert list(report["overall"]["feature_ranges"]) == list(
        RUI49_FEATURE_COLUMNS
    )
    assert all(
        bounds["finite"]
        for bounds in report["overall"]["feature_ranges"].values()
    )
    texture_columns = [f"cell__{name}" for name in HARALICK_FEATURE_NAMES]
    before = report["overall"][
        "texture_feature_ranges_before_zero_reservation"
    ]
    after = report["overall"][
        "texture_feature_ranges_after_zero_reservation"
    ]
    deltas = report["overall"]["texture_feature_range_deltas"]
    assert list(before) == texture_columns
    assert list(after) == texture_columns
    assert list(deltas) == texture_columns
    assert all(bounds["finite"] for bounds in before.values())
    assert all(bounds["finite"] for bounds in after.values())
    assert after == {
        column: report["overall"]["feature_ranges"][column]
        for column in texture_columns
    }
    assert any(
        delta["min_delta"] != 0.0 or delta["max_delta"] != 0.0
        for delta in deltas.values()
    )
    assert report["overall"]["fallback_diagnostics"]["eligible_cell_count"] == 5
    assert report["overall"]["fallback_diagnostics"]["scope"] == (
        "unique_whole_cell_labels_before_border_exclusion"
    )
    consistency = report["whole_cell_feature_consistency"]
    assert consistency == {
        "status": "passed",
        "scope": "smoke_selected_5_fovs",
        "feature_count": 49,
        "raw_pair_row_count": 6,
        "distinct_label_count": 5,
        "multirow_key_count": 1,
    }
    assert report["full_693_feature_consistency_status"] == "not_run"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["packages"] == {
        "full-environment-only": "7.0",
        "mahotas": "1.4.18",
        "python": "3.10.20",
    }
    assert metadata["python"] == "3.10.20"
    assert metadata["environment_package_count"] == 3
    assert metadata["environment_packages_path"].endswith(
        "environment_packages.json"
    )
    assert metadata["status"] == "snapshot_validated"
    assert metadata["background_subtraction_method"] == "rolling_ball_radius_50"
    assert metadata["haralick_implementation"] == "mahotas.features.haralick"
    assert metadata["haralick_distance"] == 3
    assert metadata["haralick_directions_averaged"] == 4
    assert metadata["haralick_quantization_levels"] == 256
    assert metadata["haralick_padding_level"] == 0
    assert metadata["haralick_in_mask_levels"] == "1..255"
    assert metadata["haralick_zero_reserved_for_padding"] is True
    assert metadata["whole_cell_only"] is True
    assert metadata["validation_status"] == {
        "snapshot": "validated",
        "feature_smoke": "validated",
        "whole_cell_feature_consistency_smoke": "passed",
    }
    assert metadata["smoke_test"]["status"] == "passed"
    assert metadata["smoke_test"]["image_keys"] == expected_keys
    assert metadata["smoke_test"][
        "texture_feature_ranges_before_zero_reservation"
    ] == before
    assert metadata["smoke_test"][
        "texture_feature_ranges_after_zero_reservation"
    ] == after
    assert metadata["smoke_test"]["texture_feature_range_deltas"] == deltas
    assert metadata["smoke_test"]["fallback_diagnostics"] == report[
        "overall"
    ]["fallback_diagnostics"]
    assert metadata["whole_cell_feature_consistency_smoke"] == consistency
    assert metadata["full_693_feature_consistency_status"] == "not_run"


def test_run_rui49_smoke_health_excludes_border_labels_and_fails_after_writes(
    tmp_path: Path,
) -> None:
    from immunity.exp4.rui_features import (
        HARALICK_FEATURE_NAMES,
        SMOKE_HEALTH_SCOPE_POST_BORDER,
        SmokeFeatureHealthError,
        run_rui49_smoke,
    )

    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    manifest_rows = []
    master_rows = []
    for index in range(1, 6):
        image_key = f"FOV_{index}"
        pc_path = image_dir / f"{image_key}.jpg"
        Image.fromarray(
            np.full((32, 32, 3), 64, dtype=np.uint8),
            mode="RGB",
        ).save(pc_path)
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[0, 0] = 100 + index  #每張一個會觸發 fallback 的 border label。
        labels[3:7, 3:7] = 1
        labels[10:14, 10:15] = 2
        if index == 1:
            labels[20, 20] = 3  #唯一保留的退化 label，fallback rate > 1%。
        else:
            labels[20:23, 20:23] = 3
        labels[26, 26] = 4  #mask-only QC object，不得進 health。
        np.savez_compressed(mask_dir / f"{image_key}.npz", cell_mask=labels)
        manifest_rows.append(
            {"image_key": image_key, "pc_path": str(pc_path)}
        )
        master_rows.extend(
            {
                "image_key": image_key,
                "cell_label": label,
                "nucleus_label": label,
            }
            for label in (1, 2, 3, 100 + index)
        )
        if index == 1:
            master_rows.append(
                {
                    "image_key": image_key,
                    "cell_label": 1,
                    "nucleus_label": 99,
                }
            )

    manifest_path = tmp_path / "data_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    cell_level_path = tmp_path / "cell_level_basic.csv"
    pd.DataFrame(master_rows).to_csv(cell_level_path, index=False)
    report_path = tmp_path / "feature_smoke_report.json"
    health_path = tmp_path / "feature_health_report.csv"
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        '{"status": "snapshot_validated"}\n', encoding="utf-8"
    )

    with pytest.raises(SmokeFeatureHealthError, match="provisional"):
        run_rui49_smoke(
            manifest_path=manifest_path,
            masks_dir=mask_dir,
            report_path=report_path,
            metadata_path=metadata_path,
            health_report_path=health_path,
            cell_level_path=cell_level_path,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    health_report = pd.read_csv(health_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    health = report["overall"]["feature_health"]
    consistency = report["whole_cell_feature_consistency"]

    assert report["status"] == "blocked_feature_health"
    assert report["full_693_feature_consistency_status"] == "not_run"
    assert consistency == {
        "status": "passed",
        "scope": "smoke_selected_5_fovs",
        "feature_count": 49,
        "raw_pair_row_count": 21,
        "distinct_label_count": 20,
        "multirow_key_count": 1,
    }
    assert report["texture_scope"] == "all_positive_mask_labels"
    assert report["overall"]["cell_count"] == 25
    assert list(
        report["overall"][
            "texture_feature_ranges_before_zero_reservation"
        ]
    ) == [f"cell__{name}" for name in HARALICK_FEATURE_NAMES]
    assert health["scope"] == SMOKE_HEALTH_SCOPE_POST_BORDER
    assert health["health_scope"] == "master_labels_after_border_exclusion"
    assert health["mask_positive_label_count"] == 25
    assert health["mask_only_label_count"] == 5
    assert health["master_pre_border_label_count"] == 20
    assert health["master_post_border_label_count"] == 15
    assert health["master_border_excluded_label_count"] == 5
    assert health["eligible_cell_count"] == 15
    assert health["blocked"] is True
    assert "cell__MinIntensity" in health["removed_columns"]
    assert health["fallback_counts"]["cell__Eccentricity"] == 0
    assert health["fallback_rates"]["cell__Eccentricity"] == 0.0
    assert health["fallback_counts"]["cell__MinFeretDiameter"] == 0
    assert health["fallback_rates"]["cell__MinFeretDiameter"] == 0.0
    assert health["fallback_counts"]["cell__MaxFeretDiameter"] == 0
    assert health["fallback_rates"]["cell__MaxFeretDiameter"] == 0.0
    assert any("fallback rate exceeds 1%" in reason for reason in health["reasons"])

    area = health_report.set_index("feature").loc["cell__Area"]
    minimum = health_report.set_index("feature").loc["cell__MinIntensity"]
    assert area["mean"] == pytest.approx(217 / 15)
    assert minimum["removed_from_arms"]
    assert set(health_report["fallback_scope"]) == {
        SMOKE_HEALTH_SCOPE_POST_BORDER
    }
    assert set(health_report["eligible_cell_count"]) == {15}

    assert metadata["status"] == "snapshot_validated"
    assert metadata["texture_scope"] == "all_positive_mask_labels"
    assert metadata["feature_health_status"] == "smoke_provisional_blocked"
    assert metadata["validation_status"]["feature_health"] == (
        "smoke_provisional_blocked"
    )
    assert metadata["feature_health_smoke"] == health
    assert metadata["whole_cell_feature_consistency_smoke"] == consistency
    assert metadata["full_693_feature_consistency_status"] == "not_run"
    assert metadata["smoke_test"]["whole_cell_feature_consistency"] == consistency
    assert metadata["smoke_test"]["status"] == "blocked_feature_health"
    assert "feature_health_preflight_status" not in metadata


def test_run_rui49_smoke_rejects_master_label_without_extracted_feature(
    tmp_path: Path,
) -> None:
    from immunity.exp4.rui_features import run_rui49_smoke

    master_rows = [
        {"image_key": f"FOV_{index}", "cell_label": 1, "nucleus_label": 1}
        for index in range(1, 6)
    ]
    master_rows.append(
        {"image_key": "FOV_1", "cell_label": 99, "nucleus_label": 2}
    )
    manifest, masks, cells, report, metadata = _write_consistency_smoke_fixture(
        tmp_path,
        master_rows,
    )

    with pytest.raises(ValueError, match="master label.*不在 mask|缺少 Rui49"):
        run_rui49_smoke(
            manifest_path=manifest,
            masks_dir=masks,
            report_path=report,
            metadata_path=metadata,
            cell_level_path=cells,
        )

    assert not report.exists()
    assert not metadata.exists()


def test_run_rui49_smoke_rejects_duplicate_extracted_feature_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp4 import rui_features

    master_rows = [
        {"image_key": f"FOV_{index}", "cell_label": 1, "nucleus_label": 1}
        for index in range(1, 6)
    ]
    manifest, masks, cells, report, metadata = _write_consistency_smoke_fixture(
        tmp_path,
        master_rows,
    )
    real_extract = rui_features.extract_rui49_features_with_diagnostics

    def duplicate_first_feature(*args, **kwargs):
        extracted = real_extract(*args, **kwargs)
        duplicate = pd.concat(
            [extracted.features, extracted.features.iloc[[0]]],
            ignore_index=True,
        )
        return rui_features.RuiFeatureExtractionResult(
            features=duplicate,
            legacy_texture_features=extracted.legacy_texture_features,
            fallback_flags=extracted.fallback_flags,
            cell_count=len(duplicate),
            fallback_scope=extracted.fallback_scope,
        )

    monkeypatch.setattr(
        rui_features,
        "extract_rui49_features_with_diagnostics",
        duplicate_first_feature,
    )
    with pytest.raises(ValueError, match="feature key 不可重複"):
        rui_features.run_rui49_smoke(
            manifest_path=manifest,
            masks_dir=masks,
            report_path=report,
            metadata_path=metadata,
            cell_level_path=cells,
        )

    assert not report.exists()
    assert not metadata.exists()
