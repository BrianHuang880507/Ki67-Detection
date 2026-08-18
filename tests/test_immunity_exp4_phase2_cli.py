from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml


def test_phase2_parser_exposes_mutually_exclusive_mode() -> None:
    from immunity.exp4.run_rui2025 import build_parser

    args = build_parser().parse_args(["--config", "config.yaml", "--run-phase2"])

    assert args.run_phase2 is True


def test_phase2_mode_is_mutually_exclusive_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from immunity.exp4 import run_rui2025

    monkeypatch.setattr(
        run_rui2025,
        "_run_phase2_analysis",
        lambda args: (_ for _ in ()).throw(
            AssertionError("phase2 dispatch must not run for a conflicting mode")
        ),
    )

    assert (
        run_rui2025.main(
            ["--config", "config.yaml", "--run-phase2", "--run-cv"]
        )
        == 2
    )


def test_production_config_keeps_phase1_mapping_and_declares_phase2_bundle() -> None:
    config_path = Path(__file__).parents[1] / "immunity" / "configs" / "exp4_rui2025.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["condition_mapping_reversed_hypothesis"][1] == {
        "condition": "IFN100_TNF0",
        "ifn_dose": 100,
        "tnf_dose": 0,
    }
    assert config["condition_mapping_reversed_hypothesis"][4]["condition"] == "IFN0_TNF0"
    assert len(config["phase1_protected_artifact_sha256"]) == 25
    assert config["phase2_cv_checkpoint_dir"].endswith(".phase2_cv_checkpoints")
    assert config["cv_checkpoint_dir"].endswith(".cv_checkpoints")


def test_phase2_publish_mapping_excludes_resumable_checkpoint_cache(
    tmp_path: Path,
) -> None:
    from immunity.exp4.phase2_reporting import (
        phase2_output_paths,
        phase2_publish_paths,
    )

    output_paths = phase2_output_paths(
        {field: str(tmp_path / field) for field in (
            "phase2_cv_checkpoint_dir",
            "phase2_run_metadata_path",
            "phase2_run_log_path",
        )}
    )
    published = phase2_publish_paths(
        output_paths,
        report_path=tmp_path / "REPORT.md",
    )

    assert "phase2_cv_checkpoint_dir" not in published
    assert "phase2_run_metadata_path" in published
    assert "phase2_run_log_path" in published


def test_phase2_publisher_rejects_checkpoint_cache_mapping(tmp_path: Path) -> None:
    from immunity.exp4.phase2_reporting import Phase2PublishError, publish_phase2_bundle

    with pytest.raises(Phase2PublishError, match="checkpoint cache"):
        publish_phase2_bundle(
            {"phase2_cv_checkpoint_dir": tmp_path / "staged"},
            {"phase2_cv_checkpoint_dir": tmp_path / "canonical"},
            phase1_paths={},
            phase1_checkpoint_dir=tmp_path / ".phase1-checkpoints",
            generation_id="checkpoint-rejected",
        )


def test_phase2_checkpoint_status_reports_partial_and_complete_resume() -> None:
    from immunity.exp4.run_rui2025 import _phase2_checkpoint_status

    partial = pd.DataFrame({"checkpoint_status": ["resumed", "computed"]})
    complete = pd.DataFrame({"checkpoint_status": ["resumed", "resumed"]})

    assert _phase2_checkpoint_status(partial) == "partial_resume"
    assert _phase2_checkpoint_status(complete) == "resumed"


def test_phase2_failure_evidence_preserves_partial_checkpoint_manifest(
    tmp_path: Path,
) -> None:
    from immunity.exp4.run_rui2025 import _phase2_checkpoint_cache_evidence

    cache = tmp_path / ".phase2_cv_checkpoints"
    cache.mkdir()
    manifest = cache / "manifest.json"
    manifest.write_text(
        json.dumps({"run_fingerprint": "a" * 64}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fold = cache / "geometry_24__SVR_L__fold_01.npz"
    fold.write_bytes(b"partial-fold")
    before_manifest = manifest.read_bytes()
    before_fold = fold.read_bytes()

    evidence = _phase2_checkpoint_cache_evidence(cache)

    assert evidence["manifest_exists"] is True
    assert evidence["npz_count"] == 1
    assert evidence["manifest"]["run_fingerprint"] == "a" * 64
    assert manifest.read_bytes() == before_manifest
    assert fold.read_bytes() == before_fold


def test_phase2_adapter_preserves_partial_cache_then_resumes_same_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """adapter lifecycle 必須保留 partial cache，第二次沿用同一路徑 resume。"""
    from immunity.exp4 import run_rui2025
    from immunity.exp4.phase2 import PHASE2_MANIFEST_COLUMNS
    from immunity.exp4.phase2_reporting import (
        PHASE2_FIGURE_FIELDS,
        PHASE2_TABLE_FIELDS,
        Phase2DerivedTables,
        PHASE2_OUTPUT_FIELDS,
    )

    output_root = tmp_path / "phase2-output"
    source_root = tmp_path / "source"
    output_root.mkdir()
    source_root.mkdir()
    report_path = tmp_path / "REPORT.md"
    report_path.write_bytes(b"phase1 bytes\n")
    checkpoint_dir = output_root / ".phase2_cv_checkpoints"
    phase2_paths = {
        field: output_root / field for field in PHASE2_OUTPUT_FIELDS
    }
    phase2_paths["phase2_cv_checkpoint_dir"] = checkpoint_dir
    config = {
        "source_root": str(source_root),
        "masks_dir": str(tmp_path / "masks"),
        "manifest_path": str(tmp_path / "manifest.csv"),
        "cell_level_path": str(tmp_path / "cell-level.csv"),
        "phase1_output_root": str(output_root),
        "phase1_protected_artifact_sha256": {},
        "condition_mapping_reversed_hypothesis": {},
    }
    args = SimpleNamespace(config=tmp_path / "fixture.yaml", metadata=None)

    phase1_paths = {
        name: tmp_path / f"phase1-{name.replace('/', '-') }"
        for name in (
            "data_manifest.csv",
            "cell_level_basic.csv",
            "cell_dedup_report.csv",
            "cell_level_rui49.csv",
            "feature_health_report.csv",
            "feature_redundancy_report.csv",
            "oof_predictions.csv",
        )
    }
    full_paths = {
        "run_log_path": tmp_path / "phase1-run.log",
        "cell_level_rui49_path": tmp_path / "phase1-rui49.csv",
        "feature_health_report_path": tmp_path / "phase1-health.csv",
    }
    safe_paths = {
        "cv_checkpoint_dir": tmp_path / ".cv_checkpoints",
        "oof_predictions_path": tmp_path / "phase1-oof.csv",
        "cell_dedup_report_path": tmp_path / "phase1-dedup.csv",
        "feature_redundancy_report_path": tmp_path / "phase1-redundancy.csv",
        "cv_metrics_path": tmp_path / "phase1-metrics.csv",
    }
    exploratory_paths = {"shrinkage_analysis_path": tmp_path / "phase1-shrinkage.csv"}
    immutable_state = SimpleNamespace(
        checkpoint_fingerprint={
            "sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "npz_count": 120,
        },
        mask_cache_fingerprint="c" * 64,
    )
    population = SimpleNamespace(
        cells=pd.DataFrame({"image_key": ["image-1"]}),
        healthy_features=("feature-1",),
        filtered_features=("feature-1",),
    )
    leakage_report = pd.DataFrame(
        [{"check_id": 8, "status": "passed", "details": "stub actual evidence"}]
    )
    leakage = SimpleNamespace(report=leakage_report, passed=True)
    assembly = SimpleNamespace(
        leakage=leakage,
        cv_input_fingerprints={"cv_source_fingerprint": "d" * 64},
    )
    fake_tables = Phase2DerivedTables(
        targets=pd.DataFrame(),
        leakage=pd.DataFrame(),
        metrics=pd.DataFrame(),
        oof_predictions=pd.DataFrame(),
        hyperparameters=pd.DataFrame(),
        hyperparameter_grids={},
        feature_importance=pd.DataFrame(),
        per_group_residuals=pd.DataFrame(),
        shrinkage=pd.DataFrame(),
        dose_response=pd.DataFrame(),
        residual_vs_dose=pd.DataFrame(),
        within_condition=pd.DataFrame(),
        comparison=pd.DataFrame(),
    )
    cv_result = SimpleNamespace(
        metrics=pd.DataFrame({"checkpoint_status": ["resumed"]}),
        leakage=leakage,
        core=SimpleNamespace(checkpoint_manifest={"run_fingerprint": "e" * 64}),
    )
    calls = {
        "cv": 0,
        "resume_observed": False,
        "recomputed": False,
        "published": [],
    }
    initial_cache_bytes: dict[str, bytes] = {}

    def fake_run_phase2_cv(*args: object, **kwargs: object) -> object:
        calls["cv"] += 1
        assert kwargs["checkpoint_dir"] == checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest = checkpoint_dir / "manifest.json"
        fold = checkpoint_dir / "fixture-fold-01.npz"
        if calls["cv"] == 1:
            manifest.write_text(
                json.dumps({"run_fingerprint": "e" * 64}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fold.write_bytes(b"partial-fold-bytes")
            initial_cache_bytes["manifest"] = manifest.read_bytes()
            initial_cache_bytes["fold"] = fold.read_bytes()
            raise RuntimeError("synthetic interruption after first fold")
        calls["resume_observed"] = (
            manifest.read_bytes() == initial_cache_bytes["manifest"]
            and fold.read_bytes() == initial_cache_bytes["fold"]
        )
        calls["recomputed"] = not fold.is_file() or fold.read_bytes() != b"partial-fold-bytes"
        return cv_result

    def fake_write_core(staged: object, tables: object) -> None:
        assert isinstance(staged, dict)
        for field in PHASE2_TABLE_FIELDS:
            Path(staged[field]).write_bytes(b"table")

    def fake_write_dose(staged: object, tables: object) -> None:
        assert isinstance(staged, dict)
        Path(staged["dose_response_check_path"]).write_bytes(b"dose")
        Path(staged["residual_vs_dose_path"]).write_bytes(b"residual")

    def fake_render(staged: object, **kwargs: object) -> None:
        assert isinstance(staged, dict)
        for field in PHASE2_FIGURE_FIELDS:
            Path(staged[field]).write_bytes(b"figure")

    def fake_publish(staged: object, output: object, **kwargs: object) -> dict[str, str]:
        assert isinstance(staged, dict)
        assert isinstance(output, dict)
        calls["published"].append((staged, output))
        assert "phase2_cv_checkpoint_dir" not in staged
        assert "phase2_cv_checkpoint_dir" not in output
        assert all(Path(path).resolve() != checkpoint_dir.resolve() for path in output.values())
        return {}

    monkeypatch.setattr(run_rui2025, "_load_config", lambda path: config)
    monkeypatch.setattr(run_rui2025, "_validate_phase2_mapping_config", lambda value: None)
    monkeypatch.setattr(run_rui2025, "_phase2_output_paths", lambda value: phase2_paths)
    monkeypatch.setattr(run_rui2025, "_output_path", lambda *args: report_path)
    monkeypatch.setattr(run_rui2025, "_metadata_path", lambda value: tmp_path / "phase1-metadata.json")
    monkeypatch.setattr(run_rui2025, "_full_output_paths", lambda value: full_paths)
    monkeypatch.setattr(run_rui2025, "_safe_output_paths", lambda value: safe_paths)
    monkeypatch.setattr(run_rui2025, "_exploratory_output_paths", lambda value: exploratory_paths)
    monkeypatch.setattr(run_rui2025, "_phase2_phase1_artifact_paths", lambda *args, **kwargs: phase1_paths)
    monkeypatch.setattr(run_rui2025, "validate_phase1_hash_lock", lambda *args: {})
    monkeypatch.setattr(run_rui2025, "_sha256_file", lambda path: "f" * 64)
    monkeypatch.setattr(run_rui2025, "_read_json_mapping", lambda path: {"full_rui49": {}})
    monkeypatch.setattr(run_rui2025, "capture_phase1_immutable_state", lambda **kwargs: immutable_state)
    monkeypatch.setattr(run_rui2025, "verify_phase1_immutable_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_rui2025, "validate_phase2_publish_mapping", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_rui2025, "_read_phase2_manifest", lambda path: pd.DataFrame(columns=PHASE2_MANIFEST_COLUMNS))
    monkeypatch.setattr(run_rui2025.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_rui2025, "FrozenPhase1Evidence", SimpleNamespace(from_observed=lambda **kwargs: object()))
    monkeypatch.setattr(run_rui2025, "_phase2_consistency_payload", lambda *args: {})
    monkeypatch.setattr(run_rui2025, "reconstruct_phase2_population", lambda **kwargs: population)
    monkeypatch.setattr(run_rui2025, "assemble_phase2_cv", lambda *args: assembly)
    monkeypatch.setattr(run_rui2025, "run_phase2_cv", fake_run_phase2_cv)
    monkeypatch.setattr(run_rui2025, "derive_phase2_tables", lambda **kwargs: fake_tables)
    monkeypatch.setattr(run_rui2025, "build_dose_response_check", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_rui2025, "analyze_residual_vs_dose", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_rui2025, "write_phase2_core_tables", fake_write_core)
    monkeypatch.setattr(run_rui2025, "write_phase2_dose_tables", fake_write_dose)
    monkeypatch.setattr(run_rui2025, "render_phase2_figures", fake_render)
    monkeypatch.setattr(run_rui2025, "fingerprint_phase2_numeric_artifacts", lambda paths: SimpleNamespace(as_mapping=lambda: {"metrics": "1" * 64}))
    monkeypatch.setattr(run_rui2025, "verify_phase2_numeric_artifact_immutability", lambda *args: SimpleNamespace(passed=True))
    monkeypatch.setattr(run_rui2025, "sha256_file", lambda path: "2" * 64)
    monkeypatch.setattr(run_rui2025, "build_phase2_report_section", lambda **kwargs: "stub phase2 report")
    monkeypatch.setattr(run_rui2025, "publish_phase2_bundle", fake_publish)

    first_status = run_rui2025._run_phase2_analysis(args)
    assert first_status == 2
    manifest_bytes = (checkpoint_dir / "manifest.json").read_bytes()
    fold_bytes = (checkpoint_dir / "fixture-fold-01.npz").read_bytes()
    assert not list(output_root.glob(".staging.phase2.*"))
    failure_evidence = list(output_root.glob(".failed-phase2-generation.*.json"))
    assert len(failure_evidence) == 1
    failure_payload = json.loads(failure_evidence[0].read_text(encoding="utf-8"))
    assert failure_payload["phase2_checkpoint_cache"]["npz_count"] == 1

    second_status = run_rui2025._run_phase2_analysis(args)

    assert second_status == 0
    assert calls["cv"] == 2
    assert calls["resume_observed"] is True
    assert calls["recomputed"] is False
    assert (checkpoint_dir / "manifest.json").read_bytes() == manifest_bytes
    assert (checkpoint_dir / "fixture-fold-01.npz").read_bytes() == fold_bytes
    assert len(calls["published"]) == 1
    assert not list(output_root.glob(".staging.phase2.*"))


def test_phase2_manifest_projection_reconstructs_actual_formal_population_before_cv() -> None:
    """正式 source header 以 allowlist projection 後可在 CV 前重建 population。"""
    from immunity.exp4 import run_rui2025
    from immunity.exp4.phase2 import (
        FrozenPhase1Evidence,
        PHASE2_MANIFEST_COLUMNS,
        reconstruct_phase2_population,
    )

    config_path = Path(__file__).parents[1] / "immunity" / "configs" / "exp4_rui2025.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = run_rui2025._resolve_config_path(config, "source_root", Path.cwd())
    manifest_path = run_rui2025._resolve_config_path(config, "manifest_path", source_root)
    phase1_root = run_rui2025._resolve_config_path(
        {"path": config["phase1_output_root"]}, "path", Path.cwd()
    )
    artifact_paths = run_rui2025._phase2_phase1_artifact_paths(
        config,
        source_root=source_root,
        phase1_output_root=phase1_root,
    )
    metadata_path = run_rui2025._metadata_path(config)
    required = [manifest_path, metadata_path, *artifact_paths.values()]
    if any(not Path(path).is_file() for path in required):
        pytest.skip("canonical Phase 1 artifacts unavailable")

    header = pd.read_csv(manifest_path, nrows=0).columns.tolist()
    assert set(header) >= set(PHASE2_MANIFEST_COLUMNS)
    manifest = run_rui2025._read_phase2_manifest(manifest_path)
    assert tuple(manifest.columns) == PHASE2_MANIFEST_COLUMNS

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evidence = FrozenPhase1Evidence.from_observed(
        artifact_sha256={
            name: run_rui2025._sha256_file(artifact_paths[name])
            for name in (
                "data_manifest.csv",
                "cell_level_basic.csv",
                "cell_dedup_report.csv",
                "cell_level_rui49.csv",
                "feature_health_report.csv",
                "feature_redundancy_report.csv",
                "oof_predictions.csv",
            )
        },
        pre_border_consistency=metadata["full_rui49"]["pre_border_consistency"],
        retained_consistency=metadata["full_rui49"]["retained_consistency"],
    )
    safe_paths = run_rui2025._safe_output_paths(config)
    full_paths = run_rui2025._full_output_paths(config)
    population = reconstruct_phase2_population(
        phase1_oof=pd.read_csv(safe_paths["oof_predictions_path"]),
        cell_level_basic=pd.read_csv(artifact_paths["cell_level_basic.csv"]),
        cell_dedup_report=pd.read_csv(
            artifact_paths["cell_dedup_report.csv"],
            float_precision="round_trip",
        ),
        cell_level_rui49=pd.read_csv(full_paths["cell_level_rui49_path"]),
        feature_health_report=pd.read_csv(full_paths["feature_health_report_path"]),
        feature_redundancy_report=pd.read_csv(
            safe_paths["feature_redundancy_report_path"]
        ),
        manifest=manifest,
        evidence=evidence,
    )

    assert len(population.cells) == 19_648
    assert len(population.manifest) == 693


def test_phase2_run_log_contains_exact_reversed_hypothesis_warning(
    tmp_path: Path,
) -> None:
    from immunity.exp4.phase2_diagnostics import DOSE_MAPPING_NOTE
    from immunity.exp4.phase2_reporting import write_phase2_log

    path = tmp_path / "phase2_run.log"
    write_phase2_log(path, [DOSE_MAPPING_NOTE])

    assert path.read_text(encoding="utf-8") == f"{DOSE_MAPPING_NOTE}\n"


def test_phase2_report_names_core_boundary_and_exact_dose_sha_proof() -> None:
    from immunity.exp4.phase2_reporting import build_phase2_report_section

    targets = pd.DataFrame(
        [
            {
                "group_IDO_score": 0.5,
                "confidence_flag": "normal",
                "condition_index": 4,
            }
        ]
    )
    within = pd.DataFrame(
        [{"condition_index": 4, "configuration_id": "arm__model", "cell_r2": 0.2, "cell_mae": 0.3}]
    )
    dose = pd.DataFrame([{"monotonic_non_decreasing": True}])
    hashes = {name: f"{index:064x}" for index, name in enumerate((
        "metrics",
        "oof_predictions",
        "hyperparameters",
        "feature_importance",
        "per_group_residuals",
    ), start=1)}
    metadata = {
        "status": "validated",
        "mapping_absent_by_core_interface": True,
        "mapping_independence_status": "passed",
        "mapping_independence": {
            "status": "passed",
            "mapping_absent_by_core_interface": True,
            "core_source_components_sha256": {
                "phase2.py": "a" * 64,
                "cv.py": "b" * 64,
            },
            "actual_leakage_check_8": {
                "status": "passed",
                "details": "all_cell_rows_match_manifest=true",
            },
        },
        "phase1_protected_artifact_sha256": {"one": "c" * 64},
        "phase2_numeric_artifact_sha256": {
            "before_dose_diagnostics": hashes,
            "after_dose_diagnostics": hashes,
            "immutability_status": "passed",
        },
        "population": {"retained_cells": 1},
    }

    report = build_phase2_report_section(
        metadata=metadata,
        targets=targets,
        within_condition=within,
        dose_response=dose,
    )

    assert "mapping_absent_by_core_interface=true" in report
    assert "all_cell_rows_match_manifest=true" in report
    assert "exact SHA-256 before/after comparison" in report
    assert "metrics: before=" in report
    assert "with_mapping" not in report
    assert "without_mapping" not in report


def test_phase2_figures_have_identical_sha_across_consecutive_renders(
    tmp_path: Path,
) -> None:
    from immunity.exp4.phase2_reporting import (
        PHASE2_FIGURE_FIELDS,
        render_phase2_figures,
        sha256_file,
    )

    comparison = pd.DataFrame(
        [
            {
                "configuration_id": f"arm__model_{index}",
                "phase1_mean_fold_cell_r2": 0.1 + index,
                "phase2_mean_fold_cell_r2": 0.2 + index,
                "phase1_mean_fold_rui_r2": 0.3 + index,
                "phase2_mean_fold_rui_r2": 0.4 + index,
            }
            for index in range(2)
        ]
    )
    within = pd.DataFrame(
        [
            {
                "condition_index": condition,
                "configuration_id": f"arm__model_{condition}",
                "cell_r2": 0.1 + condition,
            }
            for condition in range(1, 9)
        ]
    )
    dose = pd.DataFrame(
        [
            {
                "donor_passage_id": "B4_P5",
                "ifn_0_tnf_0_target_median": 0.1,
                "ifn_25_tnf_0_target_median": 0.2,
                "ifn_50_tnf_0_target_median": 0.3,
                "ifn_100_tnf_0_target_median": 0.4,
            }
        ]
    )
    staged = []
    hashes = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for index in (1, 2):
            root = tmp_path / f"render-{index}"
            paths = {
                field: root / f"{field}.png" for field in PHASE2_FIGURE_FIELDS
            }
            render_phase2_figures(
                paths,
                comparison=comparison,
                within_condition=within,
                dose_response=dose,
            )
            staged.append(paths)
            hashes.append({field: sha256_file(path) for field, path in paths.items()})

    assert hashes[0] == hashes[1]
    assert [
        str(item.message)
        for item in captured
        if "Glyph" in str(item.message) and "missing from font" in str(item.message)
    ] == []


def test_phase2_figures_fail_closed_without_complete_cjk_font(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from immunity.exp4 import phase2_reporting

    monkeypatch.setattr(
        phase2_reporting,
        "_PHASE2_CJK_FONT_FILENAMES",
        ("phase2-font-that-does-not-exist.ttf",),
    )
    monkeypatch.setattr(
        phase2_reporting,
        "_PHASE2_CJK_FONT_FAMILIES",
        ("Phase2 Font That Does Not Exist",),
    )

    with pytest.raises(
        phase2_reporting.Phase2PublishError,
        match="requires a CJK font covering the exact DOSE_MAPPING_NOTE",
    ):
        phase2_reporting.render_phase2_figures(
            {},
            comparison=pd.DataFrame(),
            within_condition=pd.DataFrame(),
            dose_response=pd.DataFrame(),
        )


@pytest.mark.parametrize("as_directory", [False, True])
def test_phase2_publisher_preserves_backup_when_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_directory: bool,
) -> None:
    from immunity.exp4 import phase2_reporting

    final = tmp_path / ("phase2-dir" if as_directory else "phase2.csv")
    staged = tmp_path / "staging" / final.name
    if as_directory:
        final.mkdir()
        (final / "old.txt").write_bytes(b"old")
        staged.mkdir(parents=True)
        (staged / "new.txt").write_bytes(b"new")
    else:
        final.write_bytes(b"old")
        staged.parent.mkdir()
        staged.write_bytes(b"new")

    real_replace = phase2_reporting.os.replace
    restore = {"enabled": False}

    def fail_restore(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if restore["enabled"] and ".phase2-backup." in source_path.name:
            raise OSError("synthetic restore failure")
        real_replace(source, destination)

    monkeypatch.setattr(phase2_reporting.os, "replace", fail_restore)
    restore["enabled"] = True
    with pytest.raises(phase2_reporting.Phase2PublishError) as captured:
        phase2_reporting.publish_phase2_bundle(
            {"phase2": staged},
            {"phase2": final},
            phase1_paths={},
            phase1_checkpoint_dir=tmp_path / ".phase1-checkpoints",
            generation_id="restore-failure",
            post_publish_gate=lambda: (_ for _ in ()).throw(
                ValueError("synthetic post-publish failure")
            ),
        )

    backup = next(tmp_path.glob(".phase2*.phase2-backup.restore-failure.*"))
    assert backup.exists()
    assert str(backup) in str(captured.value)
    assert str(final) in str(captured.value)


def test_phase2_publisher_quarantines_new_file_when_removal_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from immunity.exp4 import phase2_reporting

    final = tmp_path / "phase2.csv"
    staged = tmp_path / "staging" / "phase2.csv"
    final.write_bytes(b"old")
    staged.parent.mkdir()
    staged.write_bytes(b"new")
    real_remove = phase2_reporting._remove_path

    def fail_new_file_removal(path: Path) -> None:
        if Path(path).resolve() == final.resolve():
            raise OSError("synthetic new-file removal failure")
        real_remove(path)

    monkeypatch.setattr(phase2_reporting, "_remove_path", fail_new_file_removal)
    with pytest.raises(phase2_reporting.Phase2PublishError) as captured:
        phase2_reporting.publish_phase2_bundle(
            {"phase2": staged},
            {"phase2": final},
            phase1_paths={},
            phase1_checkpoint_dir=tmp_path / ".phase1-checkpoints",
            generation_id="quarantine-failure",
            post_publish_gate=lambda: (_ for _ in ()).throw(
                ValueError("synthetic post-publish failure")
            ),
        )

    quarantine = tmp_path / ".phase2.csv.phase2-rollback-quarantine.quarantine-failure"
    assert quarantine.read_bytes() == b"new"
    assert str(quarantine) in captured.value.recovery_paths
    assert final.read_bytes() == b"old"


def test_replace_phase2_report_section_preserves_unmarked_bytes() -> None:
    from immunity.exp4.phase2_reporting import (
        replace_phase2_report_section,
        report_outside_phase2_marker,
    )

    old = b"phase-1\r\ntruth\r\n"
    updated = replace_phase2_report_section(old, "new phase 2")

    assert updated.startswith(old)
    assert b"<!-- EXP4_PHASE2_START -->" in updated
    assert b"<!-- EXP4_PHASE2_END -->" in updated
    assert report_outside_phase2_marker(updated) == old


def test_replace_phase2_report_section_rejects_partial_or_duplicate_markers() -> None:
    from immunity.exp4.phase2_reporting import (
        Phase2PublishError,
        replace_phase2_report_section,
    )

    with pytest.raises(Phase2PublishError, match="marker"):
        replace_phase2_report_section(b"old\n<!-- EXP4_PHASE2_START -->", "new")
    with pytest.raises(Phase2PublishError, match="marker"):
        replace_phase2_report_section(
            b"<!-- EXP4_PHASE2_START -->x<!-- EXP4_PHASE2_END -->"
            b"<!-- EXP4_PHASE2_START -->y<!-- EXP4_PHASE2_END -->",
            "new",
        )


def test_phase2_publisher_rolls_back_everything_after_post_publish_failure(
    tmp_path: Path,
) -> None:
    from immunity.exp4.phase2_reporting import (
        Phase2PublishError,
        publish_phase2_bundle,
    )

    final = tmp_path / "phase2.csv"
    report = tmp_path / "REPORT.md"
    staged = tmp_path / "staging" / "phase2.csv"
    staged_report = tmp_path / "staging" / "REPORT.md"
    final.write_bytes(b"old-phase2")
    report.write_bytes(b"old-report")
    staged.parent.mkdir()
    staged.write_bytes(b"new-phase2")
    staged_report.write_bytes(b"new-report")

    with pytest.raises(Phase2PublishError):
        publish_phase2_bundle(
            {"phase2": staged, "report": staged_report},
            {"phase2": final, "report": report},
            phase1_paths={},
            phase1_checkpoint_dir=tmp_path / ".phase1-checkpoints",
            generation_id="test-generation",
            post_publish_gate=lambda: (_ for _ in ()).throw(
                ValueError("synthetic post-publish failure")
            ),
        )

    assert final.read_bytes() == b"old-phase2"
    assert report.read_bytes() == b"old-report"
    assert not staged.exists()
    assert not staged_report.exists()
    assert not list(tmp_path.glob("*.phase2-backup.*"))


def test_phase2_publisher_rejects_phase1_path_alias(tmp_path: Path) -> None:
    from immunity.exp4.phase2_reporting import Phase2PublishError, publish_phase2_bundle

    phase1 = tmp_path / "cv_metrics.csv"
    phase1.write_bytes(b"phase1")
    staged = tmp_path / "staging" / "phase2.csv"
    staged.parent.mkdir()
    staged.write_bytes(b"new")

    with pytest.raises(Phase2PublishError, match="Phase 1"):
        publish_phase2_bundle(
            {"phase2": staged},
            {"phase2": phase1},
            phase1_paths={"cv_metrics.csv": phase1},
            phase1_checkpoint_dir=tmp_path / ".phase1-checkpoints",
            generation_id="alias-generation",
        )


def test_phase2_publisher_rejects_phase1_checkpoint_overlap(tmp_path: Path) -> None:
    from immunity.exp4.phase2_reporting import Phase2PublishError, publish_phase2_bundle

    phase1_checkpoint = tmp_path / ".cv_checkpoints"
    phase1_checkpoint.mkdir()
    staged = tmp_path / "staging" / "phase2.csv"
    staged.parent.mkdir()
    staged.write_bytes(b"new")

    with pytest.raises(Phase2PublishError, match="checkpoint"):
        publish_phase2_bundle(
            {"phase2": staged},
            {"phase2": phase1_checkpoint / "phase2.csv"},
            phase1_paths={},
            phase1_checkpoint_dir=phase1_checkpoint,
            generation_id="checkpoint-alias-generation",
        )


def test_phase2_publisher_rolls_back_when_commit_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from immunity.exp4 import phase2_reporting

    final = tmp_path / "phase2.csv"
    report = tmp_path / "REPORT.md"
    staged = tmp_path / "staging" / "phase2.csv"
    staged_report = tmp_path / "staging" / "REPORT.md"
    final.write_bytes(b"old-phase2")
    report.write_bytes(b"old-report")
    staged.parent.mkdir()
    staged.write_bytes(b"new-phase2")
    staged_report.write_bytes(b"new-report")
    real_replace = phase2_reporting.os.replace
    state = {"count": 0}

    def fail_on_second_commit(source: object, destination: object) -> None:
        state["count"] += 1
        if state["count"] == 4:
            raise OSError("synthetic staged commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(phase2_reporting.os, "replace", fail_on_second_commit)
    with pytest.raises(phase2_reporting.Phase2PublishError, match="rollback"):
        phase2_reporting.publish_phase2_bundle(
            {"phase2": staged, "report": staged_report},
            {"phase2": final, "report": report},
            phase1_paths={},
            phase1_checkpoint_dir=tmp_path / ".phase1-checkpoints",
            generation_id="commit-failure",
        )
    assert final.read_bytes() == b"old-phase2"
    assert report.read_bytes() == b"old-report"
    assert not list(tmp_path.glob("*.phase2-backup.*"))
