"""測試 frozen-roster Paper-style 93 特徵擷取與 preflight。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from immunity.exp3 import round2_features
from immunity.exp3.feature_sets import (
    PAPER_STYLE_EXTRA_FEATURES,
    PAPER_STYLE_FOV_FEATURES,
    PRIMARY_CELL_FEATURES,
    PRIMARY_FOV_FEATURES,
)
from immunity.exp3.phase_features import extract_features_from_arrays
from immunity.exp3.round2_evidence import Round1Evidence, validate_frozen_masks
from immunity.exp3.round2_features import (
    EXTRACTION_QC_COLUMNS,
    Paper93Bundle,
    ROSTER_COLUMNS,
    VALID_COUNT_COLUMNS,
    extract_locked_paper93,
    require_paper93_preflight,
)


_BASIC_CELL_VALUES = {
    "cell__area": 64.0,
    "cell__compactness": 1.030835089459151,
    "cell__eccentricity": 0.0,
    "cell__extent": 1.0,
    "cell__sphericity": 1.0258261726007487,
    "cell__major_axis_length": 9.16515138991168,
    "cell__feret_length": 9.899494936611665,
    "cell__minor_axis_length": 9.16515138991168,
    "cell__feret_width": 8.359353065490723,
    "cell__maximum_radius": 4.0,
    "cell__mean_radius": 1.875,
    "cell__median_radius": 2.0,
    "cell__aspect_ratio": 1.0,
    "cell__perimeter_area_ratio": 0.4375,
    "cell__perimeter": 28.0,
    "cell__solidity": 1.0,
    "nucleus__area": 9.0,
    "nucleus__compactness": 0.930842267730309,
    "nucleus__eccentricity": 0.0,
    "nucleus__extent": 1.0,
    "nucleus__sphericity": 1.7671458676442586,
    "nucleus__major_axis_length": 3.265986323710904,
    "nucleus__feret_length": 2.8284271247461903,
    "nucleus__minor_axis_length": 3.265986323710904,
    "nucleus__feret_width": 2.58198881149292,
    "nucleus__maximum_radius": 2.0,
    "nucleus__mean_radius": 1.1111111111111112,
    "nucleus__median_radius": 1.0,
    "nucleus__aspect_ratio": 1.0,
    "nucleus__perimeter_area_ratio": 0.8888888888888888,
    "nucleus__perimeter": 8.0,
    "nucleus__solidity": 1.0,
    "nucleus_cytoplasm_area_ratio": 0.16363636363636364,
}


@pytest.fixture
def frozen_evidence(tmp_path: Path) -> Round1Evidence:
    """建立一張 phase、三顆固定 roster cells 的最小 Round 1 證據。"""
    root = tmp_path / "exp3"
    image_key = "B1_P1_C01_F01"
    group_id = "B1_P1"
    phase_path = tmp_path / "phase.png"
    phase = np.arange(900, dtype=np.uint8).reshape(30, 30)
    Image.fromarray(phase).save(phase_path)
    pc_sha256 = hashlib.sha256(phase_path.read_bytes()).hexdigest()

    cell_mask = np.zeros((30, 30), dtype=np.int32)
    nucleus_mask = np.zeros((30, 30), dtype=np.int32)
    for label, (row, column) in enumerate(((2, 2), (2, 14), (15, 8)), 1):
        cell_mask[row : row + 8, column : column + 8] = label
        nucleus_mask[row + 2 : row + 5, column + 2 : column + 5] = label
    mask_path = root / "feature_cache" / "masks" / group_id / f"{image_key}.npz"
    mask_path.parent.mkdir(parents=True)
    provenance_json = json.dumps(
        {
            "pc_sha256": pc_sha256,
            "schema_version": 1,
            "segmentation_config": {
                "max_nucleus_outside_fraction": 0.05,
                "min_cells_per_image": 3,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    provenance_hash = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
    np.savez(
        mask_path,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        provenance_json=np.asarray(provenance_json),
        provenance_hash=np.asarray(provenance_hash),
    )

    manifest = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "b_id": "B1",
                "passage": 1,
                "group_id": group_id,
                "condition_index": 1,
                "condition": "IFN0_TNF0",
                "ifn_dose": 0.0,
                "tnf_dose": 0.0,
                "fov": 1,
                "pc_path": str(phase_path.resolve()),
                "ido_path": str((tmp_path / "missing-ido.png").resolve()),
            }
        ]
    )
    segmentation_qc = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "group_id": group_id,
                "mask_path": str(mask_path.resolve()),
                "pc_sha256": pc_sha256,
                "cache_provenance_hash": provenance_hash,
                "status": "passed",
            }
        ]
    )
    basic_cells = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "cell_label": label,
                "nucleus_label": label,
                **_BASIC_CELL_VALUES,
                "IDO_score": 2.5,
            }
            for label in (1, 2, 3)
        ]
    )
    basic_images = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "b_id": "B1",
                "passage": 1,
                "group_id": group_id,
                "condition_index": 1,
                "condition": "IFN0_TNF0",
                "ifn_dose": 0.0,
                "tnf_dose": 0.0,
                "fov": 1,
                "cell_count": 3,
                "IDO_score": 2.5,
                **{
                    f"{feature}__median": _BASIC_CELL_VALUES[feature]
                    for feature in PRIMARY_CELL_FEATURES
                },
            }
        ]
    )
    empty = pd.DataFrame()
    return Round1Evidence(
        root=root,
        manifest=manifest,
        segmentation_qc=segmentation_qc,
        basic_cells=basic_cells,
        basic_images=basic_images,
        split_manifest=empty,
        selected_metrics=empty,
        selected_predictions=empty,
        selected_hyperparameters=empty,
        selected_importance=empty,
        selected_failures=empty,
        model_ranking=empty,
        metadata={
            "effective_config_snapshot": {
                "segmentation": {"max_nucleus_outside_fraction": 0.05}
            }
        },
        artifact_hashes={},
    )


@pytest.fixture
def verified_mask_qc(frozen_evidence: Round1Evidence) -> pd.DataFrame:
    """建立 Task 2 已通過 bytes 與 provenance 檢查的 mask QC。"""
    return validate_frozen_masks(frozen_evidence)


def _replace_mask_arrays(
    evidence: Round1Evidence,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> None:
    """只替換 synthetic cache arrays，保留已驗證的 provenance identity。"""
    mask_path = Path(evidence.segmentation_qc.loc[0, "mask_path"])
    with np.load(mask_path, allow_pickle=False) as cached:
        arrays = {name: np.asarray(cached[name]) for name in cached.files}
    arrays["cell_mask"] = np.asarray(cell_mask, dtype=np.int32)
    arrays["nucleus_mask"] = np.asarray(nucleus_mask, dtype=np.int32)
    np.savez(mask_path, **arrays)


def _evidence_with_round1_pairs(
    evidence: Round1Evidence,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> Round1Evidence:
    """以 Round 1 extractor 建立與指定 masks 相容的 frozen pair evidence。"""
    _replace_mask_arrays(evidence, cell_mask, nucleus_mask)
    image_key = str(evidence.manifest.loc[0, "image_key"])
    phase = np.asarray(Image.open(evidence.manifest.loc[0, "pc_path"]))
    rows, _ = extract_features_from_arrays(
        image_key,
        phase,
        np.zeros_like(phase, dtype=np.float64),
        cell_mask,
        nucleus_mask,
        max_nucleus_outside_fraction=0.05,
        enabled_feature_sets=("basic_median",),
    )
    basic_cells = pd.DataFrame(rows).loc[
        :, [*ROSTER_COLUMNS, *PRIMARY_CELL_FEATURES, "IDO_score"]
    ]
    basic_images = evidence.basic_images.copy()
    basic_images.loc[0, "cell_count"] = len(basic_cells)
    basic_images.loc[0, "IDO_score"] = float(basic_cells["IDO_score"].median())
    for feature in PRIMARY_CELL_FEATURES:
        basic_images.loc[0, f"{feature}__median"] = float(
            basic_cells[feature].median()
        )
    return replace(
        evidence,
        basic_cells=basic_cells,
        basic_images=basic_images,
    )


def _multi_nucleus_pair_masks() -> tuple[np.ndarray, np.ndarray]:
    """建立三個 unique cells、四個 nucleus-cell pairs 的 synthetic masks。"""
    cells = np.zeros((30, 30), dtype=np.int32)
    nuclei = np.zeros_like(cells)
    cells[2:10, 2:10] = 1
    cells[2:8, 14:20] = 2
    cells[15:19, 8:12] = 3
    nuclei[3:5, 3:5] = 1
    nuclei[4:6, 16:18] = 2
    nuclei[16:18, 9:11] = 3
    nuclei[7:9, 7:9] = 4
    return cells, nuclei


def _outside_boundary_masks(*, exact: bool) -> tuple[np.ndarray, np.ndarray]:
    """建立 1/20 exact boundary 或 1/19 just-above boundary 的 masks。"""
    cells = np.zeros((30, 30), dtype=np.int32)
    nuclei = np.zeros_like(cells)
    for label, (row, column) in enumerate(((2, 2), (2, 14), (15, 8)), 1):
        cells[row : row + 8, column : column + 8] = label
    nuclei[4:8, 4:9] = 1
    nuclei[4:7, 16:19] = 2
    nuclei[17:20, 10:13] = 3
    cells[4, 4] = 0
    if not exact:
        nuclei[7, 8] = 0
    return cells, nuclei


def _evidence_with_embedded_threshold(
    evidence: Round1Evidence,
    threshold: float | None,
) -> Round1Evidence:
    """改寫 synthetic embedded provenance 並同步其 pinned provenance hash。"""
    mask_path = Path(evidence.segmentation_qc.loc[0, "mask_path"])
    with np.load(mask_path, allow_pickle=False) as cached:
        arrays = {name: np.asarray(cached[name]) for name in cached.files}
    provenance = json.loads(str(arrays["provenance_json"].item()))
    segmentation = provenance["segmentation_config"]
    if threshold is None:
        segmentation.pop("max_nucleus_outside_fraction", None)
    else:
        segmentation["max_nucleus_outside_fraction"] = threshold
    provenance_json = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    provenance_hash = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
    arrays["provenance_json"] = np.asarray(provenance_json)
    arrays["provenance_hash"] = np.asarray(provenance_hash)
    np.savez(mask_path, **arrays)
    segmentation_qc = evidence.segmentation_qc.copy()
    segmentation_qc.loc[0, "cache_provenance_hash"] = provenance_hash
    return replace(evidence, segmentation_qc=segmentation_qc)


def test_paper93_bundle_preserves_roster_basic_values_and_target(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    assert len(bundle.predictor_columns) == 93
    assert tuple(bundle.predictor_columns) == PAPER_STYLE_FOV_FEATURES
    assert set(bundle.paper_cells[list(ROSTER_COLUMNS)].itertuples(index=False, name=None)) == set(
        frozen_evidence.basic_cells[list(ROSTER_COLUMNS)].itertuples(index=False, name=None)
    )
    np.testing.assert_allclose(
        bundle.images[list(PRIMARY_FOV_FEATURES)],
        frozen_evidence.basic_images[list(PRIMARY_FOV_FEATURES)],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bundle.images["IDO_score"],
        frozen_evidence.basic_images["IDO_score"],
        rtol=0,
        atol=1e-12,
    )
    require_paper93_preflight(bundle, formal=False)


def test_paper93_preserves_two_nuclei_mapped_to_one_frozen_cell_as_pair_rows(
    frozen_evidence: Round1Evidence,
) -> None:
    cells, nuclei = _multi_nucleus_pair_masks()
    evidence = _evidence_with_round1_pairs(frozen_evidence, cells, nuclei)

    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))

    assert bundle.extraction_qc["status"].tolist() == ["passed"]
    assert set(
        bundle.paper_cells.loc[
            bundle.paper_cells["cell_label"].eq(1),
            ["image_key", "cell_label", "nucleus_label"],
        ].itertuples(index=False, name=None)
    ) == {
        ("B1_P1_C01_F01", 1, 1),
        ("B1_P1_C01_F01", 1, 4),
    }
    assert len(bundle.paper_cells) == 4
    assert bundle.images.loc[0, "cell_count"] == 4
    assert bundle.images.loc[0, "cell__area__median"] == 50.0
    assert bundle.paper_cells.loc[
        bundle.paper_cells["cell_label"].eq(1),
        "nucleus_cytoplasm_area_ratio",
    ].tolist() == pytest.approx([4 / 60, 4 / 60])
    require_paper93_preflight(bundle, formal=False)


def test_paper93_rejects_one_nucleus_mapped_to_two_cells_with_label_evidence(
    frozen_evidence: Round1Evidence,
) -> None:
    cells, nuclei = _multi_nucleus_pair_masks()
    evidence = _evidence_with_round1_pairs(frozen_evidence, cells, nuclei)
    duplicate_nucleus = evidence.basic_cells.iloc[[0]].copy()
    duplicate_nucleus.loc[:, "cell_label"] = 2
    evidence = replace(
        evidence,
        basic_cells=pd.concat(
            [evidence.basic_cells, duplicate_nucleus],
            ignore_index=True,
        ),
    )

    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    reason = bundle.extraction_qc.loc[0, "reason"]
    assert "nucleus_label=1" in reason
    assert "cell_labels=[1, 2]" in reason


def test_paper93_rejects_frozen_pair_when_centroid_maps_to_another_cell(
    frozen_evidence: Round1Evidence,
) -> None:
    cells, nuclei = _multi_nucleus_pair_masks()
    evidence = _evidence_with_round1_pairs(frozen_evidence, cells, nuclei)
    cells[3, 3] = 2
    _replace_mask_arrays(evidence, cells, nuclei)

    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    assert "centroid" in bundle.extraction_qc.loc[0, "reason"]
    assert "expected_cell_label=1" in bundle.extraction_qc.loc[0, "reason"]
    assert "actual_cell_label=2" in bundle.extraction_qc.loc[0, "reason"]


def test_paper93_accepts_exact_round1_outside_fraction_boundary(
    frozen_evidence: Round1Evidence,
) -> None:
    cells, nuclei = _outside_boundary_masks(exact=True)
    evidence = _evidence_with_round1_pairs(frozen_evidence, cells, nuclei)

    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))

    assert bundle.extraction_qc["status"].tolist() == ["passed"]
    assert bundle.paper_cells.loc[
        bundle.paper_cells["nucleus_label"].eq(1),
        "nucleus_outside_fraction",
    ].item() == pytest.approx(0.05)
    qc = bundle.extraction_qc.iloc[0]
    assert qc["retained_outside_pair_count"] == 1
    assert qc["max_retained_outside_fraction"] == pytest.approx(0.05)
    assert qc["max_nucleus_outside_fraction"] == pytest.approx(0.05)
    require_paper93_preflight(bundle, formal=False)


def test_paper93_rejects_just_above_round1_outside_fraction_with_evidence(
    frozen_evidence: Round1Evidence,
) -> None:
    cells, nuclei = _outside_boundary_masks(exact=True)
    evidence = _evidence_with_round1_pairs(frozen_evidence, cells, nuclei)
    cells, nuclei = _outside_boundary_masks(exact=False)
    _replace_mask_arrays(evidence, cells, nuclei)

    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    reason = bundle.extraction_qc.loc[0, "reason"]
    assert "cell_label=1" in reason
    assert "nucleus_label=1" in reason
    assert "outside_fraction=0.052631578947" in reason
    assert "max_nucleus_outside_fraction=0.05" in reason


def test_paper93_requires_pinned_round1_metadata_threshold(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    evidence = replace(frozen_evidence, metadata={})

    with pytest.raises(
        ValueError,
        match="run_metadata.*max_nucleus_outside_fraction",
    ):
        extract_locked_paper93(evidence, verified_mask_qc)


@pytest.mark.parametrize("embedded_threshold", [None, 0.1])
def test_paper93_requires_embedded_provenance_threshold_to_match_metadata(
    frozen_evidence: Round1Evidence,
    embedded_threshold: float | None,
) -> None:
    evidence = _evidence_with_embedded_threshold(
        frozen_evidence,
        embedded_threshold,
    )

    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    reason = bundle.extraction_qc.loc[0, "reason"]
    assert "provenance" in reason
    assert "max_nucleus_outside_fraction" in reason


def test_paper93_mapping_qc_reports_exact_pair_count_relationships(
    frozen_evidence: Round1Evidence,
) -> None:
    cells, nuclei = _multi_nucleus_pair_masks()
    evidence = _evidence_with_round1_pairs(frozen_evidence, cells, nuclei)

    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))

    assert list(bundle.extraction_qc.columns) == EXTRACTION_QC_COLUMNS
    qc = bundle.extraction_qc.iloc[0]
    assert qc["aggregation_unit"] == "frozen_nucleus_cell_pair"
    assert qc["roster_pair_count"] == 4
    assert qc["extracted_pair_count"] == 4
    assert qc["unique_cell_count"] == 3
    assert qc["unique_nucleus_count"] == 4
    assert qc["multi_nucleus_cell_count"] == 1
    assert qc["multi_nucleus_pair_count"] == 2
    assert qc["max_nuclei_per_cell"] == 2
    assert qc["retained_outside_pair_count"] == 0
    assert qc["max_retained_outside_fraction"] == 0.0
    assert qc["max_nucleus_outside_fraction"] == pytest.approx(0.05)
    require_paper93_preflight(bundle, formal=False)


@pytest.mark.parametrize(
    ("column", "forged"),
    [
        ("unique_cell_count", 999),
        ("multi_nucleus_pair_count", 999),
        ("retained_outside_pair_count", 999),
        ("max_retained_outside_fraction", 0.04),
        ("max_nucleus_outside_fraction", 0.1),
        ("aggregation_unit", "unique_cell"),
        ("status", "forged"),
    ],
)
def test_paper93_preflight_rejects_forged_pair_mapping_qc(
    frozen_evidence: Round1Evidence,
    column: str,
    forged: object,
) -> None:
    cells, nuclei = _multi_nucleus_pair_masks()
    evidence = _evidence_with_round1_pairs(frozen_evidence, cells, nuclei)
    bundle = extract_locked_paper93(evidence, validate_frozen_masks(evidence))
    extraction_qc = bundle.extraction_qc.copy()
    extraction_qc.loc[0, column] = forged

    with pytest.raises(ValueError, match="extraction.*QC|pair mapping|status"):
        require_paper93_preflight(
            replace(bundle, extraction_qc=extraction_qc),
            formal=False,
        )


def test_paper93_valid_counts_require_three_finite_cells_per_extra(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {name: 1.0 for name in PAPER_STYLE_EXTRA_FEATURES}
    calls = 0

    def fake_extras(*args: object) -> dict[str, float]:
        nonlocal calls
        calls += 1
        row = dict(values)
        if calls >= 2:
            row["cell__zernike_00"] = np.nan
        return row

    monkeypatch.setattr(round2_features, "paper_style_extras", fake_extras)
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    assert list(bundle.valid_counts.columns) == VALID_COUNT_COLUMNS
    failed = bundle.valid_counts.query(
        "feature == 'cell__zernike_00' and status == 'failed'"
    )
    assert failed["finite_cell_count"].tolist() == [1]
    with pytest.raises(ValueError, match="cell__zernike_00.*finite_cell_count=1"):
        require_paper93_preflight(bundle, formal=True)


def test_paper93_cannot_lower_three_finite_cell_floor(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="min_finite_cells.*精確為 3"):
        extract_locked_paper93(
            frozen_evidence,
            verified_mask_qc,
            min_finite_cells=2,
        )


def test_paper93_recomputes_33_cell_features_and_rejects_mask_array_drift(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    mask_path = Path(verified_mask_qc.loc[0, "mask_path"])
    with np.load(mask_path, allow_pickle=False) as cached:
        arrays = {name: np.asarray(cached[name]) for name in cached.files}
    cell_mask = arrays["cell_mask"]
    cell_mask[cell_mask == 1] = 0
    cell_mask[2:9, 2:10] = 1
    arrays["cell_mask"] = cell_mask
    np.savez(mask_path, **arrays)
    refreshed_mask_qc = validate_frozen_masks(frozen_evidence)

    bundle = extract_locked_paper93(frozen_evidence, refreshed_mask_qc)

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    assert "basic feature drift" in bundle.extraction_qc.loc[0, "reason"]
    with pytest.raises(ValueError, match="basic feature drift"):
        require_paper93_preflight(bundle, formal=False)


@pytest.mark.parametrize("drift", ["missing_extra", "duplicate_roster"])
def test_paper93_rejects_missing_extra_or_duplicate_roster_pair(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    evidence = frozen_evidence
    if drift == "missing_extra":
        values = {name: 1.0 for name in PAPER_STYLE_EXTRA_FEATURES[1:]}
        monkeypatch.setattr(round2_features, "paper_style_extras", lambda *args: values)
    else:
        cells = pd.concat(
            [frozen_evidence.basic_cells, frozen_evidence.basic_cells.iloc[[0]]],
            ignore_index=True,
        )
        evidence = replace(frozen_evidence, basic_cells=cells)

    bundle = extract_locked_paper93(evidence, verified_mask_qc)

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    with pytest.raises(ValueError, match="extra feature schema|roster.*重複"):
        require_paper93_preflight(bundle, formal=False)


def test_paper93_extraction_failure_contains_image_key_pc_path_and_mask_path(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    pc_path = Path(verified_mask_qc.loc[0, "pc_path"])
    mask_path = Path(verified_mask_qc.loc[0, "mask_path"])
    pc_path.unlink()

    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    reason = bundle.extraction_qc.loc[0, "reason"]
    assert "image_key=B1_P1_C01_F01" in reason
    assert str(pc_path) in reason
    assert str(mask_path) in reason
    with pytest.raises(ValueError, match="image_key=B1_P1_C01_F01"):
        require_paper93_preflight(bundle, formal=False)


def test_paper93_constant_feature_is_reported_but_not_failed(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    row = bundle.feature_qc.query("feature == 'cell__area__median'").iloc[0]
    assert bool(row["constant"]) is True
    assert row["status"] == "passed"
    require_paper93_preflight(bundle, formal=False)


@pytest.mark.parametrize(
    "forbidden",
    ["IDO", "IFN", "TNF", "dose", "condition", "path", "filename", "donor", "b_id", "delta"],
)
def test_paper93_rejects_forbidden_predictor_at_exact_count_93(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    forbidden: str,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    columns = (forbidden, *bundle.predictor_columns[1:])

    with pytest.raises(ValueError, match=forbidden):
        require_paper93_preflight(replace(bundle, predictor_columns=columns), formal=False)


@pytest.mark.parametrize("replacement", ["cell__invented__median", "cell__area__median"])
def test_paper93_rejects_duplicate_or_unregistered_predictor_at_count_93(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    replacement: str,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    columns = (*bundle.predictor_columns[:-1], replacement)

    with pytest.raises(ValueError, match="重複|未註冊"):
        require_paper93_preflight(replace(bundle, predictor_columns=columns), formal=False)


def test_paper93_does_not_read_missing_ido_path_or_call_segmentation(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenSegmenter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("Paper93 extractor 不可建立 Segmenter")

    monkeypatch.setattr("immunity.exp3.phase_features.PhaseSegmenter", ForbiddenSegmenter)
    assert not Path(frozen_evidence.manifest.loc[0, "ido_path"]).exists()

    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    require_paper93_preflight(bundle, formal=False)


def test_paper93_formal_preflight_requires_exactly_693_fovs(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    with pytest.raises(ValueError, match="693"):
        require_paper93_preflight(bundle, formal=True)


def test_paper93_preflight_reports_first_canonical_order_mismatch(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    columns = list(bundle.predictor_columns)
    columns[0], columns[1] = columns[1], columns[0]

    with pytest.raises(
        ValueError,
        match="cell__compactness__median.*cell__area__median",
    ):
        require_paper93_preflight(
            replace(bundle, predictor_columns=tuple(columns)), formal=False
        )


@pytest.mark.parametrize("table", ["valid_counts", "feature_qc", "extraction_qc"])
def test_paper93_preflight_revalidates_qc_instead_of_trusting_passed_status(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    table: str,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    changed = getattr(bundle, table).copy()
    if table == "valid_counts":
        changed.loc[0, "finite_cell_count"] = 999
    elif table == "feature_qc":
        changed.loc[0, "finite_fov_count"] = 999
    else:
        changed.loc[0, "extracted_cell_count"] = 999

    with pytest.raises(ValueError, match="QC|count|finite_fov_count"):
        require_paper93_preflight(replace(bundle, **{table: changed}), formal=False)


@pytest.mark.parametrize("changed_table", ["images", "paper_cells"])
def test_paper93_preflight_recomputes_60_extra_medians_from_cells(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    changed_table: str,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    changed = getattr(bundle, changed_table).copy()
    feature = "cell__zernike_00"
    if changed_table == "images":
        changed.loc[0, f"{feature}__median"] += 7.0
    else:
        changed[feature] += 7.0

    with pytest.raises(ValueError, match="median.*cell__zernike_00"):
        require_paper93_preflight(
            replace(bundle, **{changed_table: changed}),
            formal=False,
        )


def test_paper93_extra_aggregation_matches_hand_derived_finite_medians(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    literal_rows = (
        {"cell__zernike_00": 9.0, "cell__zernike_01": np.nan},
        {"cell__zernike_00": 1.0, "cell__zernike_01": 1.0},
        {"cell__zernike_00": 5.0, "cell__zernike_01": 9.0},
    )
    calls = 0

    def literal_extras(*args: object) -> dict[str, float]:
        nonlocal calls
        row = {feature: float(calls * 10 + 10) for feature in PAPER_STYLE_EXTRA_FEATURES}
        row.update(literal_rows[calls])
        calls += 1
        return row

    monkeypatch.setattr(round2_features, "paper_style_extras", literal_extras)

    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    assert bundle.images.loc[0, "cell__zernike_00__median"] == 5.0
    assert bundle.images.loc[0, "cell__zernike_01__median"] == 5.0
    assert bundle.images.loc[0, "cell__zernike_02__median"] == 20.0
    counts = bundle.valid_counts.set_index("feature")
    assert counts.loc["cell__zernike_00", "finite_cell_count"] == 3
    assert counts.loc["cell__zernike_01", "finite_cell_count"] == 2


@pytest.mark.parametrize("atol", [0.0, 1e-9])
def test_paper93_rejects_noncanonical_basic_drift_tolerance(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    atol: float,
) -> None:
    with pytest.raises(ValueError, match="atol.*1e-12"):
        extract_locked_paper93(frozen_evidence, verified_mask_qc, atol=atol)


@pytest.mark.parametrize("mutation", ["synchronized_extra", "basic_and_snapshot", "attrs"])
def test_paper93_creation_semantics_cannot_be_mutated_after_extraction(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    mutation: str,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    changed_images = bundle.images.copy()
    changed_cells = bundle.paper_cells.copy()
    if mutation == "synchronized_extra":
        changed_cells["cell__zernike_00"] += 1.0
        changed_images["cell__zernike_00__median"] += 1.0
    elif mutation == "basic_and_snapshot":
        feature = "cell__area__median"
        changed_images.loc[0, feature] += 0.25
        changed_images.attrs["_paper93_expected_basic_images"].loc[0, feature] += 0.25
    else:
        changed_images.attrs["forged_snapshot"] = "accepted"

    with pytest.raises(ValueError, match="semantic|snapshot|attrs"):
        require_paper93_preflight(
            replace(bundle, images=changed_images, paper_cells=changed_cells),
            formal=False,
        )


@pytest.mark.parametrize("drift", ["forged_qc", "phase_bytes", "mask_provenance"])
def test_paper93_rejects_forged_or_stale_task2_provenance(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    drift: str,
) -> None:
    qc = verified_mask_qc.copy()
    if drift == "forged_qc":
        qc.loc[0, "actual_pc_sha256"] = "0" * 64
    elif drift == "phase_bytes":
        phase_path = Path(qc.loc[0, "pc_path"])
        changed = np.arange(900, dtype=np.uint8).reshape(30, 30)
        changed[0, 0] = 99
        Image.fromarray(changed).save(phase_path)
    else:
        mask_path = Path(qc.loc[0, "mask_path"])
        with np.load(mask_path, allow_pickle=False) as cached:
            arrays = {name: np.asarray(cached[name]) for name in cached.files}
        arrays["provenance_hash"] = np.asarray("0" * 64)
        np.savez(mask_path, **arrays)

    bundle = extract_locked_paper93(frozen_evidence, qc)

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    reason = bundle.extraction_qc.loc[0, "reason"]
    assert "image_key=B1_P1_C01_F01" in reason
    assert str(qc.loc[0, "pc_path"]) in reason
    assert str(qc.loc[0, "mask_path"]) in reason
    assert "SHA-256" in reason or "provenance" in reason


def test_paper93_preflight_rejects_synchronously_lowered_finite_cell_floor(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    counts = bundle.valid_counts.copy()
    counts["required_minimum"] = 1
    counts["status"] = "passed"

    with pytest.raises(ValueError, match="required_minimum.*3"):
        require_paper93_preflight(replace(bundle, valid_counts=counts), formal=False)


def _synchronized_renamed_formal_bundle(bundle: Paper93Bundle) -> Paper93Bundle:
    """建立 693 筆內部 identity 同步但非 production roster 的 adversarial bundle。"""
    keys = [f"RENAMED_{index:03d}" for index in range(693)]
    image_base = bundle.images.copy()
    image_base.attrs = {}
    expected_image_base = bundle.images.attrs["_paper93_expected_basic_images"].copy()
    expected_image_base.attrs = {}
    cell_base = bundle.paper_cells.copy()
    cell_base.attrs = {}
    expected_cell_base = bundle.paper_cells.attrs["_paper93_expected_basic_cells"].copy()
    expected_cell_base.attrs = {}
    valid_base = bundle.valid_counts.copy()
    extraction_base = bundle.extraction_qc.copy()

    def replicated(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for key in keys:
            changed = frame.copy()
            changed["image_key"] = key
            rows.append(changed)
        return pd.concat(rows, ignore_index=True)

    images = replicated(image_base)
    expected_images = replicated(expected_image_base)
    images.attrs["_paper93_expected_basic_images"] = expected_images
    images.attrs["_paper93_atol"] = 1e-12
    cells = replicated(cell_base)
    expected_cells = replicated(expected_cell_base)
    cells.attrs["_paper93_expected_basic_cells"] = expected_cells
    cells.attrs["_paper93_atol"] = 1e-12
    feature_qc = bundle.feature_qc.copy()
    feature_qc["finite_fov_count"] = 693
    return replace(
        bundle,
        images=images,
        paper_cells=cells,
        valid_counts=replicated(valid_base),
        extraction_qc=replicated(extraction_base),
        feature_qc=feature_qc,
    )


def test_paper93_formal_preflight_rejects_synchronized_693_key_rename(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)
    renamed = _synchronized_renamed_formal_bundle(bundle)

    with pytest.raises(ValueError, match="image-key roster SHA-256"):
        require_paper93_preflight(renamed, formal=True)


def test_paper93_rejects_same_shape_mask_pixel_mutation_after_task2_qc(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    mask_path = Path(verified_mask_qc.loc[0, "mask_path"])
    with np.load(mask_path, allow_pickle=False) as cached:
        arrays = {name: np.asarray(cached[name]) for name in cached.files}
    changed = arrays["cell_mask"].copy()
    changed[0, 0] = 99
    arrays["cell_mask"] = changed
    np.savez(mask_path, **arrays)

    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    assert bundle.extraction_qc["status"].tolist() == ["failed"]
    reason = bundle.extraction_qc.loc[0, "reason"]
    assert "cell_mask_sha256" in reason
    assert "image_key=B1_P1_C01_F01" in reason
    assert str(mask_path) in reason


def test_paper93_normalizes_inherited_round1_loader_attrs(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    frozen_evidence.basic_images.attrs["round1_frozen_semantic_sha256"] = {
        "columns": "loader-owned-digest"
    }

    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    assert "round1_frozen_semantic_sha256" not in bundle.images.attrs
    require_paper93_preflight(bundle, formal=False)
