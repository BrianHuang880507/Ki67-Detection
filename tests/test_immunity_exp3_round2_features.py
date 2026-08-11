"""測試 frozen-roster Paper-style 93 特徵擷取與 preflight。"""

from __future__ import annotations

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
from immunity.exp3.round2_evidence import Round1Evidence
from immunity.exp3.round2_features import (
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

    cell_mask = np.zeros((30, 30), dtype=np.int32)
    nucleus_mask = np.zeros((30, 30), dtype=np.int32)
    for label, (row, column) in enumerate(((2, 2), (2, 14), (15, 8)), 1):
        cell_mask[row : row + 8, column : column + 8] = label
        nucleus_mask[row + 2 : row + 5, column + 2 : column + 5] = label
    mask_path = root / "feature_cache" / "masks" / group_id / f"{image_key}.npz"
    mask_path.parent.mkdir(parents=True)
    np.savez(mask_path, cell_mask=cell_mask, nucleus_mask=nucleus_mask)

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
        [{"image_key": image_key, "group_id": group_id, "mask_path": str(mask_path.resolve())}]
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
        metadata={},
        artifact_hashes={},
    )


@pytest.fixture
def verified_mask_qc(frozen_evidence: Round1Evidence) -> pd.DataFrame:
    """建立 Task 2 已通過 bytes 與 provenance 檢查的 mask QC。"""
    row = frozen_evidence.manifest.iloc[0]
    mask_path = frozen_evidence.segmentation_qc.iloc[0]["mask_path"]
    return pd.DataFrame(
        [
            {
                "image_key": row["image_key"],
                "pc_path": row["pc_path"],
                "mask_path": mask_path,
                "status": "passed",
                "reason": "verified",
            }
        ]
    )


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
    with pytest.raises(ValueError, match="min_finite_cells.*至少為 3"):
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
        cell_mask = np.asarray(cached["cell_mask"])
        nucleus_mask = np.asarray(cached["nucleus_mask"])
    cell_mask[cell_mask == 1] = 0
    cell_mask[2:9, 2:10] = 1
    np.savez(mask_path, cell_mask=cell_mask, nucleus_mask=nucleus_mask)

    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

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
