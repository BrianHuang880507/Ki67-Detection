"""Exp4 full Rui49 擷取核心的 public seam 測試。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest


def _write_full_fixture(
    tmp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """建立兩個 FOV、含 multirow 與 mask-only label 的 fixture。"""
    image_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    image_dir.mkdir()
    masks_dir.mkdir()
    manifest_rows: list[dict[str, object]] = []
    for image_key, code in (("FOV_B", 20), ("FOV_A", 10)):
        pc_path = image_dir / f"{image_key}.png"
        Image.fromarray(
            np.full((8, 8, 3), code, dtype=np.uint8),
            mode="RGB",
        ).save(pc_path)
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[4:6, 4:6] = 2
        if image_key == "FOV_B":
            labels[6, 1] = 9
        np.savez_compressed(masks_dir / f"{image_key}.npz", cell_mask=labels)
        manifest_rows.append(
            {"image_key": image_key, "pc_path": str(pc_path)}
        )

    raw_cells = pd.DataFrame(
        [
            {"image_key": "FOV_B", "cell_label": 1, "nucleus_label": 1},
            {"image_key": "FOV_B", "cell_label": 2, "nucleus_label": 2},
            {"image_key": "FOV_A", "cell_label": 1, "nucleus_label": 1},
            {"image_key": "FOV_A", "cell_label": 1, "nucleus_label": 2},
            {"image_key": "FOV_A", "cell_label": 2, "nucleus_label": 3},
        ]
    )
    retained_cells = pd.DataFrame(
        [
            {"image_key": "FOV_B", "cell_label": 2},
            {"image_key": "FOV_A", "cell_label": 2},
            {"image_key": "FOV_A", "cell_label": 1},
        ]
    )
    return pd.DataFrame(manifest_rows), raw_cells, retained_cells, masks_dir


def _install_fake_extractor(monkeypatch, *, fallback_labels=()) -> dict[str, int]:
    """以 deterministic synthetic extractor 取代昂貴影像特徵計算。"""
    from immunity.exp4 import full_rui49
    from immunity.exp4.rui_features import (
        RUI49_FEATURE_COLUMNS,
        RuiFeatureExtractionResult,
    )

    fallback_set = set(fallback_labels)
    calls = {"extract": 0}

    def fake_preprocess(rgb_image: np.ndarray) -> np.ndarray:
        return np.asarray(rgb_image[..., 0], dtype=np.float64)

    def fake_extract(
        phase_signal: np.ndarray,
        cell_labels: np.ndarray,
        *,
        include_legacy_texture: bool = False,
    ) -> RuiFeatureExtractionResult:
        assert include_legacy_texture is False
        calls["extract"] += 1
        image_code = int(phase_signal[0, 0])
        image_key = "FOV_A" if image_code == 10 else "FOV_B"
        labels = [int(value) for value in np.unique(cell_labels) if value > 0]
        feature_rows: list[dict[str, object]] = []
        fallback_rows: list[dict[str, object]] = []
        for label in labels:
            base = float(image_code + label)
            feature_rows.append(
                {
                    "cell_label": label,
                    **{
                        column: (
                            0.0
                            if column == "cell__MinIntensity"
                            else base + index / 100.0
                        )
                        for index, column in enumerate(RUI49_FEATURE_COLUMNS)
                    },
                }
            )
            fallback_rows.append(
                {
                    "cell_label": label,
                    **{
                        column: (image_key, label, column) in fallback_set
                        for column in RUI49_FEATURE_COLUMNS
                    },
                }
            )
        features = pd.DataFrame(
            feature_rows,
            columns=("cell_label", *RUI49_FEATURE_COLUMNS),
        )
        fallback_flags = pd.DataFrame(
            fallback_rows,
            columns=("cell_label", *RUI49_FEATURE_COLUMNS),
        )
        return RuiFeatureExtractionResult(
            features=features,
            legacy_texture_features=None,
            fallback_flags=fallback_flags,
            cell_count=len(features),
        )

    monkeypatch.setattr(full_rui49, "preprocess_phase_image", fake_preprocess)
    monkeypatch.setattr(
        full_rui49,
        "extract_rui49_features_with_diagnostics",
        fake_extract,
    )
    return calls


def _run_full_fixture(tmp_path: Path, monkeypatch, *, fallback_labels=()):
    """以小型 count locks 呼叫 full public seam。"""
    from immunity.exp4.full_rui49 import extract_full_rui49

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    _install_fake_extractor(monkeypatch, fallback_labels=fallback_labels)
    return extract_full_rui49(
        manifest=manifest,
        raw_cells=raw_cells,
        retained_cells=retained_cells,
        masks_dir=masks_dir,
        checkpoint_dir=tmp_path / "output" / ".rui49-checkpoints",
        progress_log_path=tmp_path / "output" / "rui49-progress.jsonl",
        expected_fov_count=2,
        expected_raw_pair_rows=5,
        expected_pre_border_cells=4,
        expected_retained_cells=3,
        expected_raw_multirow_keys=1,
        expected_retained_raw_pair_rows=4,
    )


def test_extract_full_rui49_returns_canonical_scopes_and_health(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    result = _run_full_fixture(tmp_path, monkeypatch)

    assert result.raw_pair_features.columns.tolist() == [
        "image_key",
        "cell_label",
        "nucleus_label",
        *RUI49_FEATURE_COLUMNS,
    ]
    assert result.raw_pair_features.loc[
        :, ["image_key", "cell_label", "nucleus_label"]
    ].values.tolist() == [
        ["FOV_A", 1, 1],
        ["FOV_A", 1, 2],
        ["FOV_A", 2, 3],
        ["FOV_B", 1, 1],
        ["FOV_B", 2, 2],
    ]
    duplicate = result.raw_pair_features.query(
        "image_key == 'FOV_A' and cell_label == 1"
    )
    np.testing.assert_array_equal(
        duplicate.loc[:, RUI49_FEATURE_COLUMNS].iloc[0].to_numpy(),
        duplicate.loc[:, RUI49_FEATURE_COLUMNS].iloc[1].to_numpy(),
    )
    assert len(result.pre_border_features) == 4
    assert ("FOV_B", 9) not in set(
        result.pre_border_features.loc[
            :, ["image_key", "cell_label"]
        ].itertuples(index=False, name=None)
    )
    assert result.retained_features.loc[
        :, ["image_key", "cell_label"]
    ].values.tolist() == [["FOV_A", 1], ["FOV_A", 2], ["FOV_B", 2]]

    assert result.pre_border_consistency.scope == (
        "full_raw_join_5_pairs_4_cells_2_fovs"
    )
    assert result.pre_border_consistency.checked_row_count == 5
    assert result.pre_border_consistency.checked_cell_count == 4
    assert result.pre_border_consistency.checked_duplicate_key_count == 1
    assert result.retained_consistency.scope == (
        "full_retained_raw_join_4_pairs_3_cells_2_fovs"
    )
    assert result.retained_consistency.checked_row_count == 4
    assert result.retained_consistency.checked_cell_count == 3
    assert result.retained_consistency.checked_duplicate_key_count == 1

    assert result.health.eligible_cell_count == 3
    assert result.health.fallback_scope == (
        "full_retained_3_distinct_cells_2_fovs"
    )
    assert result.health.blocked is False
    assert result.health.removed_columns == ("cell__MinIntensity",)
    assert len(result.health.healthy_feature_columns) == 48
    assert list(result.healthy_feature_ranges) == list(
        result.health.healthy_feature_columns
    )
    assert all(
        bounds["finite"]
        for bounds in result.healthy_feature_ranges.values()
    )
    assert len(result.retained_fallback_flags) == 3
    assert all(count == 0 for count in result.retained_fallback_counts.values())
    assert all(rate == 0.0 for rate in result.retained_fallback_rates.values())

    assert result.processed_fov_count == 2
    assert result.resumed_fov_count == 0
    assert [timing.image_key for timing in result.fov_timings] == [
        "FOV_A",
        "FOV_B",
    ]
    assert [item.status for item in result.checkpoint_provenance] == [
        "processed",
        "processed",
    ]
    checkpoints = sorted(
        (tmp_path / "output" / ".rui49-checkpoints").glob("*.npz")
    )
    assert len(checkpoints) == 2
    for checkpoint in checkpoints:
        with np.load(checkpoint, allow_pickle=False) as loaded:
            assert loaded["feature_columns"].dtype.kind == "U"
            assert loaded["features"].dtype == np.float64
    progress = [
        json.loads(line)
        for line in (tmp_path / "output" / "rui49-progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["image_key"] for item in progress] == ["FOV_A", "FOV_B"]
    assert [item["status"] for item in progress] == ["processed", "processed"]


def test_extract_full_rui49_resumes_without_extraction_and_is_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp4.full_rui49 import extract_full_rui49

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    calls = _install_fake_extractor(monkeypatch)
    arguments = {
        "manifest": manifest,
        "raw_cells": raw_cells,
        "retained_cells": retained_cells,
        "masks_dir": masks_dir,
        "checkpoint_dir": tmp_path / "output" / ".rui49-checkpoints",
        "progress_log_path": tmp_path / "output" / "rui49-progress.jsonl",
        "expected_fov_count": 2,
        "expected_raw_pair_rows": 5,
        "expected_pre_border_cells": 4,
        "expected_retained_cells": 3,
        "expected_raw_multirow_keys": 1,
        "expected_retained_raw_pair_rows": 4,
    }

    fresh = extract_full_rui49(**arguments)
    resumed = extract_full_rui49(**arguments)

    assert calls["extract"] == 2
    assert resumed.processed_fov_count == 0
    assert resumed.resumed_fov_count == 2
    assert [timing.status for timing in resumed.fov_timings] == [
        "resumed",
        "resumed",
    ]
    assert [item.status for item in resumed.checkpoint_provenance] == [
        "resumed",
        "resumed",
    ]
    pd.testing.assert_frame_equal(
        resumed.raw_pair_features,
        fresh.raw_pair_features,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        resumed.pre_border_features,
        fresh.pre_border_features,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        resumed.retained_features,
        fresh.retained_features,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        resumed.retained_fallback_flags,
        fresh.retained_fallback_flags,
        check_exact=True,
    )
    progress = [
        json.loads(line)
        for line in arguments["progress_log_path"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["status"] for item in progress] == [
        "processed",
        "processed",
        "resumed",
        "resumed",
    ]


@pytest.mark.parametrize("mutation", ["helper", "constant"])
def test_extract_full_rui49_rejects_transitive_extractor_fingerprint_drift(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    from immunity.exp4 import rui_features
    from immunity.exp4.full_rui49 import (
        FullRui49CheckpointError,
        extract_full_rui49,
    )

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    calls = _install_fake_extractor(monkeypatch)
    arguments = {
        "manifest": manifest,
        "raw_cells": raw_cells,
        "retained_cells": retained_cells,
        "masks_dir": masks_dir,
        "checkpoint_dir": tmp_path / "output" / ".rui49-checkpoints",
        "progress_log_path": tmp_path / "output" / "rui49-progress.jsonl",
        "expected_fov_count": 2,
        "expected_raw_pair_rows": 5,
        "expected_pre_border_cells": 4,
        "expected_retained_cells": 3,
        "expected_raw_multirow_keys": 1,
        "expected_retained_raw_pair_rows": 4,
    }
    extract_full_rui49(**arguments)

    if mutation == "constant":
        monkeypatch.setattr(
            rui_features,
            "BACKGROUND_RADIUS",
            rui_features.BACKGROUND_RADIUS + 1,
        )
    else:
        original_intensity = rui_features._intensity_features

        def changed_intensity(*args, **kwargs):
            return original_intensity(*args, **kwargs)

        monkeypatch.setattr(
            rui_features,
            "_intensity_features",
            changed_intensity,
        )

    with pytest.raises(
        FullRui49CheckpointError,
        match="extractor/source fingerprint stale",
    ):
        extract_full_rui49(**arguments)

    assert calls["extract"] == 2


@pytest.mark.parametrize("working_directory", ["cwd-one", "cwd-two"])
def test_extract_full_rui49_rejects_relative_pc_paths_before_any_write(
    tmp_path: Path,
    monkeypatch,
    working_directory: str,
) -> None:
    from immunity.exp4.full_rui49 import FullRui49Error, extract_full_rui49

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    relative_manifest = manifest.copy()
    relative_manifest["pc_path"] = relative_manifest["pc_path"].map(
        lambda value: Path(str(value)).name
    )
    collision_dir = tmp_path / working_directory
    collision_dir.mkdir()
    for image_key, code in (("FOV_A", 10), ("FOV_B", 20)):
        Image.fromarray(
            np.full((8, 8, 3), code, dtype=np.uint8),
            mode="RGB",
        ).save(collision_dir / f"{image_key}.png")
    monkeypatch.chdir(collision_dir)
    calls = _install_fake_extractor(monkeypatch)
    checkpoint_dir = tmp_path / "output" / ".rui49-checkpoints"
    progress_path = tmp_path / "output" / "rui49-progress.jsonl"

    with pytest.raises(FullRui49Error, match="pc_path 必須是絕對路徑"):
        extract_full_rui49(
            manifest=relative_manifest,
            raw_cells=raw_cells,
            retained_cells=retained_cells,
            masks_dir=masks_dir,
            checkpoint_dir=checkpoint_dir,
            progress_log_path=progress_path,
            expected_fov_count=2,
            expected_raw_pair_rows=5,
            expected_pre_border_cells=4,
            expected_retained_cells=3,
            expected_raw_multirow_keys=1,
            expected_retained_raw_pair_rows=4,
        )

    assert calls["extract"] == 0
    assert not checkpoint_dir.exists()
    assert not progress_path.exists()


def test_extract_full_rui49_rejects_stale_checkpoint_without_reextracting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp4.full_rui49 import (
        FullRui49CheckpointError,
        extract_full_rui49,
    )

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    calls = _install_fake_extractor(monkeypatch)
    arguments = {
        "manifest": manifest,
        "raw_cells": raw_cells,
        "retained_cells": retained_cells,
        "masks_dir": masks_dir,
        "checkpoint_dir": tmp_path / "output" / ".rui49-checkpoints",
        "progress_log_path": tmp_path / "output" / "rui49-progress.jsonl",
        "expected_fov_count": 2,
        "expected_raw_pair_rows": 5,
        "expected_pre_border_cells": 4,
        "expected_retained_cells": 3,
        "expected_raw_multirow_keys": 1,
        "expected_retained_raw_pair_rows": 4,
    }
    extract_full_rui49(**arguments)
    mask_path = masks_dir / "FOV_A.npz"
    current = mask_path.stat()
    os.utime(
        mask_path,
        ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
    )

    with pytest.raises(
        FullRui49CheckpointError,
        match="mask_mtime_ns stale|source fingerprint stale",
    ):
        extract_full_rui49(**arguments)

    assert calls["extract"] == 2
    progress = arguments["progress_log_path"].read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(progress) == 2


def test_extract_full_rui49_propagates_retained_fallbacks_when_health_blocks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _run_full_fixture(
        tmp_path,
        monkeypatch,
        fallback_labels={
            ("FOV_A", 1, "cell__Eccentricity"),
            ("FOV_B", 9, "cell__Eccentricity"),
        },
    )

    assert result.retained_fallback_counts["cell__Eccentricity"] == 1
    assert result.retained_fallback_rates["cell__Eccentricity"] == pytest.approx(
        1 / 3
    )
    assert result.health.blocked is True
    assert result.health.eligible_cell_count == 3
    assert any(
        "fallback rate exceeds 1%" in reason
        for reason in result.health.reasons
    )
    assert result.pre_border_consistency.checked_row_count == 5
    assert result.retained_consistency.checked_row_count == 4
    assert result.retained_fallback_flags.query(
        "image_key == 'FOV_A' and cell_label == 1"
    )["cell__Eccentricity"].item()
    assert not result.retained_fallback_flags.query(
        "image_key == 'FOV_B' and cell_label == 2"
    )["cell__Eccentricity"].item()


def test_extract_full_rui49_rejects_master_label_missing_from_mask(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp4.full_rui49 import FullRui49Error, extract_full_rui49

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    raw_cells.loc[
        (raw_cells["image_key"] == "FOV_B")
        & (raw_cells["cell_label"] == 1),
        "cell_label",
    ] = 8
    _install_fake_extractor(monkeypatch)

    with pytest.raises(
        FullRui49Error,
        match="FOV_B master label 缺少 Rui49 feature.*8",
    ):
        extract_full_rui49(
            manifest=manifest,
            raw_cells=raw_cells,
            retained_cells=retained_cells,
            masks_dir=masks_dir,
            checkpoint_dir=tmp_path / "output" / ".rui49-checkpoints",
            progress_log_path=tmp_path / "output" / "rui49-progress.jsonl",
            expected_fov_count=2,
            expected_raw_pair_rows=5,
            expected_pre_border_cells=4,
            expected_retained_cells=3,
            expected_raw_multirow_keys=1,
            expected_retained_raw_pair_rows=4,
        )

    assert len(list((tmp_path / "output" / ".rui49-checkpoints").glob("*.npz"))) == 1


def test_extract_full_rui49_rejects_nonfinite_canonical_feature(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp4 import full_rui49
    from immunity.exp4.full_rui49 import (
        FullRui49CanonicalBlockedError,
        extract_full_rui49,
    )
    from immunity.exp4.rui_features import RuiFeatureExtractionResult

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    _install_fake_extractor(monkeypatch)
    finite_extractor = full_rui49.extract_rui49_features_with_diagnostics

    def nonfinite_extractor(*args, **kwargs) -> RuiFeatureExtractionResult:
        extraction = finite_extractor(*args, **kwargs)
        features = extraction.features.copy()
        features.loc[0, "cell__Area"] = np.nan
        return RuiFeatureExtractionResult(
            features=features,
            legacy_texture_features=None,
            fallback_flags=extraction.fallback_flags,
            cell_count=extraction.cell_count,
        )

    monkeypatch.setattr(
        full_rui49,
        "extract_rui49_features_with_diagnostics",
        nonfinite_extractor,
    )

    with pytest.raises(
        FullRui49CanonicalBlockedError,
        match="canonical Rui49 blocked",
    ) as captured:
        extract_full_rui49(
            manifest=manifest,
            raw_cells=raw_cells,
            retained_cells=retained_cells,
            masks_dir=masks_dir,
            checkpoint_dir=tmp_path / "output" / ".rui49-checkpoints",
            progress_log_path=tmp_path / "output" / "rui49-progress.jsonl",
            expected_fov_count=2,
            expected_raw_pair_rows=5,
            expected_pre_border_cells=4,
            expected_retained_cells=3,
            expected_raw_multirow_keys=1,
            expected_retained_raw_pair_rows=4,
        )

    diagnostics = captured.value.diagnostics
    assert diagnostics.failure_kind == "blocked_nonfinite"
    assert diagnostics.nonfinite_columns == ("cell__Area",)
    assert diagnostics.health.blocked is True
    assert diagnostics.health.eligible_cell_count == 3
    assert any(
        "nonfinite feature columns" in reason
        for reason in diagnostics.health.reasons
    )
    assert diagnostics.pre_border_consistency.status == "not_run_nonfinite"
    assert diagnostics.pre_border_consistency.scope == (
        "full_raw_join_5_pairs_4_cells_2_fovs"
    )
    assert diagnostics.pre_border_consistency.checked_row_count == 5
    assert diagnostics.pre_border_consistency.checked_cell_count == 4
    assert diagnostics.pre_border_consistency.checked_duplicate_key_count == 1
    assert diagnostics.retained_consistency.status == "not_run_nonfinite"
    assert diagnostics.retained_consistency.scope == (
        "full_retained_raw_join_4_pairs_3_cells_2_fovs"
    )
    assert diagnostics.retained_consistency.checked_row_count == 4
    assert diagnostics.retained_consistency.checked_cell_count == 3
    assert len(diagnostics.raw_pair_features) == 5
    assert len(diagnostics.retained_features) == 3
    assert diagnostics.processed_fov_count == 2
    assert len(diagnostics.checkpoint_provenance) == 2


@pytest.mark.parametrize(
    "malformed_fallback",
    [pytest.param(np.nan, id="nan"), pytest.param("", id="empty"), 2],
)
def test_extract_full_rui49_rejects_malformed_fresh_fallback_flags(
    tmp_path: Path,
    monkeypatch,
    malformed_fallback,
) -> None:
    from immunity.exp4 import full_rui49
    from immunity.exp4.full_rui49 import FullRui49Error, extract_full_rui49
    from immunity.exp4.rui_features import RuiFeatureExtractionResult

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    _install_fake_extractor(monkeypatch)
    valid_extractor = full_rui49.extract_rui49_features_with_diagnostics

    def malformed_extractor(*args, **kwargs) -> RuiFeatureExtractionResult:
        extraction = valid_extractor(*args, **kwargs)
        fallback_flags = extraction.fallback_flags.copy()
        fallback_flags["cell__Area"] = fallback_flags["cell__Area"].astype(
            object
        )
        fallback_flags.loc[0, "cell__Area"] = malformed_fallback
        return RuiFeatureExtractionResult(
            features=extraction.features,
            legacy_texture_features=None,
            fallback_flags=fallback_flags,
            cell_count=extraction.cell_count,
        )

    monkeypatch.setattr(
        full_rui49,
        "extract_rui49_features_with_diagnostics",
        malformed_extractor,
    )
    checkpoint_dir = tmp_path / "output" / ".rui49-checkpoints"

    with pytest.raises(
        FullRui49Error,
        match="fallback flag 必須是 bool 或 numeric 0/1",
    ):
        extract_full_rui49(
            manifest=manifest,
            raw_cells=raw_cells,
            retained_cells=retained_cells,
            masks_dir=masks_dir,
            checkpoint_dir=checkpoint_dir,
            progress_log_path=tmp_path / "output" / "rui49-progress.jsonl",
            expected_fov_count=2,
            expected_raw_pair_rows=5,
            expected_pre_border_cells=4,
            expected_retained_cells=3,
            expected_raw_multirow_keys=1,
            expected_retained_raw_pair_rows=4,
        )

    assert not list(checkpoint_dir.glob("*.npz"))


def test_extract_full_rui49_rejects_malformed_checkpoint_without_reextracting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp4.full_rui49 import (
        FullRui49CheckpointError,
        extract_full_rui49,
    )

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    calls = _install_fake_extractor(monkeypatch)
    arguments = {
        "manifest": manifest,
        "raw_cells": raw_cells,
        "retained_cells": retained_cells,
        "masks_dir": masks_dir,
        "checkpoint_dir": tmp_path / "output" / ".rui49-checkpoints",
        "progress_log_path": tmp_path / "output" / "rui49-progress.jsonl",
        "expected_fov_count": 2,
        "expected_raw_pair_rows": 5,
        "expected_pre_border_cells": 4,
        "expected_retained_cells": 3,
        "expected_raw_multirow_keys": 1,
        "expected_retained_raw_pair_rows": 4,
    }
    fresh = extract_full_rui49(**arguments)
    first_checkpoint = Path(fresh.checkpoint_provenance[0].checkpoint_path)
    np.savez_compressed(
        first_checkpoint,
        image_key=np.asarray("FOV_A", dtype=np.str_),
    )

    with pytest.raises(FullRui49CheckpointError, match="schema 不符"):
        extract_full_rui49(**arguments)

    assert calls["extract"] == 2


@pytest.mark.parametrize(
    ("lock_name", "wrong_value", "message"),
    [
        ("expected_fov_count", 3, "manifest FOV count 不符"),
        ("expected_raw_pair_rows", 6, "raw pair row count 不符"),
        (
            "expected_pre_border_cells",
            5,
            "pre-border distinct cell count 不符",
        ),
        (
            "expected_retained_cells",
            4,
            "retained distinct cell count 不符",
        ),
        (
            "expected_retained_raw_pair_rows",
            5,
            "retained raw pair row count 不符",
        ),
    ],
)
def test_extract_full_rui49_count_locks_fail_closed_before_extraction(
    tmp_path: Path,
    monkeypatch,
    lock_name: str,
    wrong_value: int,
    message: str,
) -> None:
    from immunity.exp4.full_rui49 import FullRui49Error, extract_full_rui49

    manifest, raw_cells, retained_cells, masks_dir = _write_full_fixture(
        tmp_path
    )
    calls = _install_fake_extractor(monkeypatch)
    arguments = {
        "manifest": manifest,
        "raw_cells": raw_cells,
        "retained_cells": retained_cells,
        "masks_dir": masks_dir,
        "checkpoint_dir": tmp_path / "output" / ".rui49-checkpoints",
        "progress_log_path": tmp_path / "output" / "rui49-progress.jsonl",
        "expected_fov_count": 2,
        "expected_raw_pair_rows": 5,
        "expected_pre_border_cells": 4,
        "expected_retained_cells": 3,
        "expected_raw_multirow_keys": 1,
        "expected_retained_raw_pair_rows": 4,
    }
    arguments[lock_name] = wrong_value

    with pytest.raises(FullRui49Error, match=message):
        extract_full_rui49(**arguments)

    assert calls["extract"] == 0
    assert not arguments["checkpoint_dir"].exists()
    assert not arguments["progress_log_path"].exists()
