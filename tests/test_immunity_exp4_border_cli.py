from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def test_production_config_locks_exact_canonical_sparse_manifest_stems() -> None:
    """確認正式 sparse roster 直接使用 manifest 的 canonical image_key。"""
    config_path = (
        Path(__file__).parents[1] / "immunity" / "configs" / "exp4_rui2025.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = config["expected"]

    canonical_keys = (
        "B4_P7_C08_F06",
        "B7_P7_C07_F03",
        "B7_P7_C07_F06",
        "B8_P5_C07_F03",
        "B8_P5_C08_F02",
        "B8_P7_C01_F02",
        "B8_P7_C01_F03",
        "B8_P7_C03_F01",
        "B8_P7_C07_F04",
        "B8_P7_C07_F08",
    )

    assert expected["sparse_fov_count"] == 10
    assert tuple(expected["sparse_fov_keys"]) == canonical_keys
    assert all("/" not in image_key for image_key in expected["sparse_fov_keys"])


def test_report_renders_current_feature_smoke_evidence_from_metadata() -> None:
    """validated smoke 必須以 metadata 的 current evidence 取代 stale 占位。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "feature_smoke_status": "validated",
            "smoke_test": {
                "status": "passed",
                "image_count": 5,
                "report_path": "feature_smoke_report.json",
            },
            "whole_cell_feature_consistency_smoke": {
                "status": "passed",
                "scope": "smoke_selected_5_fovs",
                "raw_pair_row_count": 115,
                "distinct_label_count": 110,
                "multirow_key_count": 5,
            },
            "feature_health_smoke": {
                "status": "smoke_provisional_passed",
                "health_scope": "master_labels_after_border_exclusion",
                "master_post_border_label_count": 95,
                "removed_columns": ["cell__MinIntensity"],
                "fallback_rates": {"cell__Area": 0.0, "cell__Variance": 0.0},
            },
            "full_693_feature_consistency_status": "not_run",
        }
    )

    assert "5-image current smoke：passed" in report
    assert (
        "canonical49 consistency：scope=smoke_selected_5_fovs；"
        "115 raw pairs／110 distinct／5 multirow"
    ) in report
    assert "full693 consistency：not_run（尚未驗證）" in report
    assert (
        "health scope=master_labels_after_border_exclusion；"
        "95 post-border cells"
    ) in report
    assert "唯一 zero-variance feature removed：cell__MinIntensity" in report
    assert "fallback max：0%" in report
    assert "texture zero-reservation evidence：feature_smoke_report.json" in report
    assert "新版 source 尚未重跑 49 特徵" not in report


def test_report_keeps_feature_smoke_placeholder_when_not_run() -> None:
    """未執行 smoke 時仍保留不可宣稱 current validated 的占位。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(metadata={"feature_smoke_status": "not_run"})

    assert "新版 source 尚未重跑 49 特徵" in report
    assert "5-image current smoke" not in report


def _write_mask(masks_dir: Path, image_key: str, labels: np.ndarray) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    np.savez(masks_dir / f"{image_key}.npz", cell_mask=labels)


def _fixture(tmp_path: Path, *, blocked: bool = False):
    from immunity.exp3.feature_sets import PRIMARY_CELL_FEATURES

    masks_dir = tmp_path / "masks"
    image_keys = ("FOV_A", "FOV_B")
    manifest = pd.DataFrame(
        {
            "image_key": image_keys,
            "b_id": ("B4", "B4"),
            "passage": (5, 5),
            "group_id": ("B4_P5", "B4_P5"),
        }
    )
    labels_a = np.zeros((6, 6), dtype=np.int32)
    labels_a[1:3, 1:3] = 1
    labels_a[1:3, 3:5] = 2
    labels_a[3:5, 1:3] = 3
    labels_a[0, 3:5] = 4
    labels_b = np.zeros((6, 6), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    labels_b[1:3, 3:5] = 2
    labels_b[3:5, 1:3] = 3
    _write_mask(masks_dir, "FOV_A", labels_a)
    _write_mask(masks_dir, "FOV_B", labels_b)
    rows = []
    for image_key, labels, values in (
        ("FOV_A", (1, 2, 3, 4), (1.0, 3.0, 5.0, 100.0)),
        ("FOV_B", (1, 2, 3), (2.0, 4.0, 6.0)),
    ):
        if blocked and image_key == "FOV_A":
            labels, values = (1, 4), (1.0, 100.0)
            labels_a_blocked = np.zeros((6, 6), dtype=np.int32)
            labels_a_blocked[1:3, 1:3] = 1
            labels_a_blocked[0, 3:5] = 4
            _write_mask(masks_dir, "FOV_A", labels_a_blocked)
        for index, (label, value) in enumerate(zip(labels, values), start=1):
            row = {
                "image_key": image_key,
                "cell_label": label,
                "nucleus_label": index,
                "IDO_score": value,
            }
            row.update(
                {
                    column: (
                        100.0
                        if column == "cell__area"
                        else 1.0 / 99.0
                        if column == "nucleus_cytoplasm_area_ratio"
                        else 1.0
                    )
                    for column in PRIMARY_CELL_FEATURES
                }
            )
            rows.append(row)
    cells = pd.DataFrame(rows)
    manifest_path = tmp_path / "data_manifest.csv"
    cells_path = tmp_path / "cell_level_basic.csv"
    manifest.to_csv(manifest_path, index=False)
    cells.to_csv(cells_path, index=False)
    output_root = Path.cwd() / "immunity" / "outputs" / "exp4" / tmp_path.name
    config = {
        "source_root": str(tmp_path),
        "manifest_path": str(manifest_path),
        "cell_level_path": str(cells_path),
        "masks_dir": str(masks_dir),
        "metadata_path": str(output_root / "run_metadata.json"),
        "border_exclusion_report_path": str(output_root / "border_exclusion_report.csv"),
        "fov_ido_scores_path": str(output_root / "fov_ido_scores.csv"),
        "group_targets_path": str(output_root / "group_targets.csv"),
        "expected": {
            "manifest_image_key_count": 2,
            "cell_row_count": len(cells),
            "cell_image_key_count": 2,
            "dedup_cell_count": 7 if not blocked else 5,
            "post_border_cell_count": 6 if not blocked else 4,
            "duplicate_key_count": 0,
            "sparse_fov_count": 0,
            "sparse_fov_keys": [],
            "group_count": 1,
            "group_keys": ["B4_P5"],
        },
    }
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, config


def test_border_measure_mode_is_mutually_exclusive_with_snapshot_mode(
    tmp_path: Path, capsys
) -> None:
    from immunity.exp4.run_rui2025 import main

    config_path, _ = _fixture(tmp_path)
    exit_code = main(
        [
            "--config",
            str(config_path),
            "--validate-snapshot",
            "--measure-border-exclusion",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "mutually exclusive" in captured.err


def test_border_measure_mode_writes_report_targets_metadata_and_machine_summary(
    tmp_path: Path, capsys
) -> None:
    from immunity.exp4.run_rui2025 import main

    config_path, config = _fixture(tmp_path)
    mask_only = np.zeros((6, 6), dtype=np.int32)
    mask_only[1:3, 1:3] = 1
    mask_only[1:3, 3:5] = 2
    mask_only[3:5, 1:3] = 3
    mask_only[0, 3:5] = 4
    mask_only[3:5, 3:5] = 5
    _write_mask(tmp_path / "masks", "FOV_A", mask_only)
    exit_code = main(["--config", str(config_path), "--measure-border-exclusion"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "diagnostic"
    assert summary["diagnostic_only"] is True
    assert summary["target_status"] == "not_run"
    assert summary["cells_before_border_exclusion"] == 7
    assert summary["cells_excluded_border"] == 1
    assert summary["cells_after_border_exclusion"] == 6
    assert summary["whole_cell_labels_before_border_exclusion"] == 7
    assert summary["whole_cell_labels_excluded_border"] == 1
    assert summary["whole_cell_labels_after_border_exclusion"] == 6
    assert summary["fov_count"] == 2
    assert summary["minimum_whole_cell_labels_after_border_exclusion"] == 3
    assert summary["minimum_cell_gate_status"] == "passed"
    assert summary["mask_only_label_total"] == 1
    assert summary["mask_only_label_max_per_fov"] == 1
    assert summary["output_paths"] == {}
    assert not Path(config["border_exclusion_report_path"]).exists()
    assert not Path(config["group_targets_path"]).exists()
    assert not Path(config["metadata_path"]).exists()


def test_measure_border_mode_is_diagnostic_only_and_does_not_publish_targets(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from immunity.exp4.run_rui2025 import main

    monkeypatch.chdir(tmp_path.parent)
    config_path, config = _fixture(tmp_path)
    target_path = Path(config["group_targets_path"])
    metadata_path = Path(config["metadata_path"])
    target_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)

    exit_code = main(["--config", str(config_path), "--measure-border-exclusion"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "diagnostic"
    assert summary["target_status"] == "not_run"
    assert not target_path.exists()
    assert not metadata_path.exists()


def test_border_measure_keeps_sparse_fov_and_writes_targets(
    tmp_path: Path, capsys
) -> None:
    from immunity.exp4.run_rui2025 import main

    config_path, config = _fixture(tmp_path, blocked=True)
    exit_code = main(["--config", str(config_path), "--measure-border-exclusion"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "diagnostic"
    assert summary["diagnostic_only"] is True
    assert summary["minimum_cell_gate_status"] == "passed"
    assert not Path(config["border_exclusion_report_path"]).exists()
    assert not Path(config["group_targets_path"]).exists()
    assert not Path(config["metadata_path"]).exists()


def test_border_measure_missing_mask_fails_without_report(tmp_path: Path, capsys) -> None:
    from immunity.exp4.run_rui2025 import main

    config_path, config = _fixture(tmp_path)
    (tmp_path / "masks" / "FOV_B.npz").unlink()
    exit_code = main(["--config", str(config_path), "--measure-border-exclusion"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 2
    assert summary["status"] == "blocked"
    assert not Path(config["border_exclusion_report_path"]).exists()


def test_border_measure_existing_lock_mismatch_preserves_metadata(tmp_path: Path, capsys) -> None:
    from immunity.exp4.run_rui2025 import main

    config_path, config = _fixture(tmp_path)
    metadata_path = Path(config["metadata_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        '{"cells_after_border_exclusion": 999, "keep": "original"}\n',
        encoding="utf-8",
    )
    before = metadata_path.read_bytes()
    exit_code = main(["--config", str(config_path), "--measure-border-exclusion"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "diagnostic"
    assert metadata_path.read_bytes() == before


def test_border_measure_rejects_outputs_inside_exp3_source_root(
    tmp_path: Path, capsys
) -> None:
    from immunity.exp4.run_rui2025 import main

    config_path, config = _fixture(tmp_path)
    bad_report = tmp_path / "bad-border.csv"
    config["border_exclusion_report_path"] = str(bad_report)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--measure-border-exclusion"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "diagnostic"
    assert not bad_report.exists()


def test_safe_mode_runs_basic_raw_through_high_target_outputs(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from immunity.exp4.run_rui2025 import main

    monkeypatch.chdir(tmp_path.parent)
    config_path, config = _fixture(tmp_path)
    output_root = Path(config["metadata_path"]).parent
    config.update(
        {
            "cell_dedup_report_path": str(output_root / "cell_dedup_report.csv"),
            "group_target_sensitivity_path": str(
                output_root / "group_target_sensitivity.csv"
            ),
            "report_path": str(output_root / "REPORT.md"),
        }
    )
    config["expected"].update(
        {
            "dedup_cell_count": 7,
            "post_border_cell_count": 6,
            "duplicate_key_count": 0,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--run-safe"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "validated"
    assert summary["dedup_cell_count"] == 7
    assert summary["post_border_cell_count"] == 6
    assert summary["target_status"] == "validated"
    assert Path(config["cell_dedup_report_path"]).is_file()
    assert Path(config["border_exclusion_report_path"]).is_file()
    assert Path(config["group_targets_path"]).is_file()
    assert not Path(config["group_target_sensitivity_path"]).is_file()
    assert Path(output_root / "fov_ido_scores.csv").is_file()
    metadata = json.loads(Path(config["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["target_sensitivity_status"] == "not_run_unit_fixture"
    assert Path(config["report_path"]).is_file()


def test_safe_mode_fails_closed_when_post_border_lock_mismatches(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from immunity.exp4.run_rui2025 import main

    monkeypatch.chdir(tmp_path.parent)
    config_path, config = _fixture(tmp_path)
    output_root = Path(config["metadata_path"]).parent
    config.update(
        {
            "cell_dedup_report_path": str(output_root / "cell_dedup_report.csv"),
            "group_target_sensitivity_path": str(
                output_root / "group_target_sensitivity.csv"
            ),
            "report_path": str(output_root / "REPORT.md"),
        }
    )
    config["expected"].update(
        {
            "dedup_cell_count": 7,
            "post_border_cell_count": 5,
            "duplicate_key_count": 0,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--run-safe"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 2
    assert summary["status"] == "blocked"
    assert "post-border" in summary["error"]
    assert not Path(config["group_targets_path"]).exists()


def test_safe_mode_preserves_old_bundle_and_writes_failed_generation_evidence(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from immunity.exp4.run_rui2025 import main

    monkeypatch.chdir(tmp_path.parent)
    config_path, config = _fixture(tmp_path)
    output_root = Path(config["metadata_path"]).parent
    config.update(
        {
            "cell_dedup_report_path": str(output_root / "cell_dedup_report.csv"),
            "group_target_sensitivity_path": str(
                output_root / "group_target_sensitivity.csv"
            ),
            "report_path": str(output_root / "REPORT.md"),
        }
    )
    config["expected"].update(
        {
            "dedup_cell_count": 999,
            "post_border_cell_count": 6,
            "duplicate_key_count": 0,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    target_path = Path(config["group_targets_path"])
    sensitivity_path = Path(config["group_target_sensitivity_path"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("historical target", encoding="utf-8")
    sensitivity_path.write_text("historical sensitivity", encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--run-safe"])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 2
    assert summary["status"] == "blocked"
    assert target_path.read_text(encoding="utf-8") == "historical target"
    assert sensitivity_path.read_text(encoding="utf-8") == "historical sensitivity"
    evidence_path = Path(summary["failure_evidence_path"])
    assert evidence_path.name.startswith(".failed-generation.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "blocked"
    assert evidence["target_status"] == "blocked"
    assert "diagnostic_error" in evidence


def test_safe_mode_rejects_outputs_outside_current_exp4_output_root(
    tmp_path: Path, capsys
) -> None:
    from immunity.exp4.run_rui2025 import main

    config_path, config = _fixture(tmp_path)
    config["group_targets_path"] = str(tmp_path.parent / "outside-targets.csv")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--run-safe"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "containment" in captured.err or "output" in captured.err
    assert not Path(config["group_targets_path"]).exists()


def _publisher_fixture(tmp_path: Path):
    from immunity.exp4 import run_rui2025

    output_root = tmp_path / "immunity" / "outputs" / "exp4"
    stage_root = output_root / ".staging.generation"
    fields = (
        "metadata_path",
        "cell_dedup_report_path",
        "border_exclusion_report_path",
        "fov_ido_scores_path",
        "group_targets_path",
        "group_target_sensitivity_path",
        "report_path",
    )
    output_paths = {
        field: output_root / f"{field}.out"
        for field in fields
    }
    staged_paths = {}
    for field, output in output_paths.items():
        staged = stage_root / output.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(f"new:{field}", encoding="utf-8")
        staged_paths[field] = staged
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"old:{field}", encoding="utf-8")
    return run_rui2025, output_paths, staged_paths, fields


def test_bundle_publisher_restores_all_old_outputs_on_archive_failure(
    tmp_path: Path, monkeypatch
) -> None:
    run_rui2025, output_paths, staged_paths, fields = _publisher_fixture(tmp_path)
    original_replace = run_rui2025.os.replace
    calls = 0

    def fail_archive_once(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("archive injection")
        return original_replace(source, target)

    monkeypatch.setattr(run_rui2025.os, "replace", fail_archive_once)
    with pytest.raises(Exception, match="archive injection"):
        run_rui2025._publish_bundle(
            staged_paths,
            output_paths,
            fields=fields,
            generation_id="generation",
        )

    assert all(
        path.read_text(encoding="utf-8") == f"old:{field}"
        for field, path in output_paths.items()
    )


def test_bundle_publisher_restores_all_old_outputs_on_commit_failure(
    tmp_path: Path, monkeypatch
) -> None:
    run_rui2025, output_paths, staged_paths, fields = _publisher_fixture(tmp_path)
    original_replace = run_rui2025.os.replace
    calls = 0
    fail_at = len(fields) + 2

    def fail_commit_once(source, target):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("commit injection")
        return original_replace(source, target)

    monkeypatch.setattr(run_rui2025.os, "replace", fail_commit_once)
    with pytest.raises(Exception, match="commit injection"):
        run_rui2025._publish_bundle(
            staged_paths,
            output_paths,
            fields=fields,
            generation_id="generation",
        )

    assert all(
        path.read_text(encoding="utf-8") == f"old:{field}"
        for field, path in output_paths.items()
    )


def test_bundle_publisher_quarantines_new_output_when_first_unlink_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """partial-existing bundle rollback 不得留下原本不存在的 canonical new 檔。"""
    run_rui2025, output_paths, staged_paths, fields = _publisher_fixture(tmp_path)
    missing_field = "metadata_path"
    output_paths[missing_field].unlink()
    original_replace = run_rui2025.os.replace
    original_unlink = run_rui2025.Path.unlink
    calls = 0
    unlink_calls = 0
    fail_at = (len(fields) - 1) + 2

    def fail_second_commit(source, target):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("partial commit injection")
        return original_replace(source, target)

    def fail_first_unlink(path, *args, **kwargs):
        nonlocal unlink_calls
        if unlink_calls == 0:
            unlink_calls += 1
            raise OSError("first unlink injection")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(run_rui2025.os, "replace", fail_second_commit)
    monkeypatch.setattr(run_rui2025.Path, "unlink", fail_first_unlink)
    with pytest.raises(run_rui2025.BundlePublishError, match="quarantined"):
        run_rui2025._publish_bundle(
            staged_paths,
            output_paths,
            fields=fields,
            generation_id="partial-generation",
        )

    quarantine = output_paths[missing_field].with_name(
        f".{output_paths[missing_field].name}.rollback_quarantine.partial-generation"
    )
    assert not output_paths[missing_field].exists()
    assert quarantine.read_text(encoding="utf-8") == f"new:{missing_field}"
    assert all(
        path.read_text(encoding="utf-8") == f"old:{field}"
        for field, path in output_paths.items()
        if field != missing_field
    )


def test_bundle_publisher_marks_fatal_recovery_when_quarantine_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """quarantine 也失敗時必須明確回報 fatal recovery 與 canonical 風險。"""
    run_rui2025, output_paths, staged_paths, fields = _publisher_fixture(tmp_path)
    missing_field = "metadata_path"
    output_paths[missing_field].unlink()
    original_replace = run_rui2025.os.replace
    original_unlink = run_rui2025.Path.unlink
    calls = 0
    unlink_calls = 0
    fail_at = (len(fields) - 1) + 2

    def fail_commit_and_quarantine(source, target):
        nonlocal calls
        calls += 1
        if calls in {fail_at, fail_at + 1}:
            raise OSError("recovery injection")
        return original_replace(source, target)

    def fail_first_unlink(path, *args, **kwargs):
        nonlocal unlink_calls
        if unlink_calls == 0:
            unlink_calls += 1
            raise OSError("first unlink injection")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(run_rui2025.os, "replace", fail_commit_and_quarantine)
    monkeypatch.setattr(run_rui2025.Path, "unlink", fail_first_unlink)
    with pytest.raises(run_rui2025.BundlePublishError, match="FATAL recovery"):
        run_rui2025._publish_bundle(
            staged_paths,
            output_paths,
            fields=fields,
            generation_id="fatal-generation",
        )

    # The explicit fatal error is the evidence that canonical recovery is not
    # guaranteed; the new file is intentionally left visible for diagnosis.
    assert output_paths[missing_field].read_text(encoding="utf-8") == (
        f"new:{missing_field}"
    )


def test_bundle_publisher_restores_old_bundle_when_historical_archive_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """historical archive loop 失敗也必須 rollback 全部 canonical outputs。"""
    run_rui2025, output_paths, staged_paths, fields = _publisher_fixture(tmp_path)
    original_replace = run_rui2025.os.replace
    calls = 0
    fail_at = len(fields) + len(fields) + 1

    def fail_historical_archive(source, target):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("historical archive injection")
        return original_replace(source, target)

    monkeypatch.setattr(run_rui2025.os, "replace", fail_historical_archive)
    with pytest.raises(run_rui2025.BundlePublishError, match="historical archive injection"):
        run_rui2025._publish_bundle(
            staged_paths,
            output_paths,
            fields=fields,
            generation_id="historical-generation",
        )

    assert all(
        path.read_text(encoding="utf-8") == f"old:{field}"
        for field, path in output_paths.items()
    )


def test_bundle_publisher_blocked_subset_omits_old_group_target(
    tmp_path: Path,
) -> None:
    """blocked sensitivity subset 仍發布同 generation、自洽且無 current target。"""
    run_rui2025, output_paths, staged_paths, fields = _publisher_fixture(tmp_path)
    blocked_fields = tuple(field for field in fields if field != "group_targets_path")
    staged_paths["metadata_path"].write_text("status=blocked", encoding="utf-8")
    staged_paths["group_target_sensitivity_path"].write_text(
        "status=blocked", encoding="utf-8"
    )
    staged_paths["report_path"].write_text("status=blocked", encoding="utf-8")

    archived = run_rui2025._publish_bundle(
        staged_paths,
        output_paths,
        fields=blocked_fields,
        generation_id="blocked-generation",
    )

    assert set(archived) == set(fields)
    assert not output_paths["group_targets_path"].exists()
    assert (
        output_paths["group_target_sensitivity_path"].read_text(encoding="utf-8")
        == "status=blocked"
    )
    assert output_paths["metadata_path"].read_text(encoding="utf-8") == "status=blocked"
    assert Path(archived["group_targets_path"]).read_text(encoding="utf-8") == (
        "old:group_targets_path"
    )


def test_bundle_publisher_commits_all_new_outputs_and_archives_old_bundle(
    tmp_path: Path,
) -> None:
    run_rui2025, output_paths, staged_paths, fields = _publisher_fixture(tmp_path)

    archived = run_rui2025._publish_bundle(
        staged_paths,
        output_paths,
        fields=fields,
        generation_id="generation",
    )

    assert set(archived) == set(fields)
    assert all(
        path.read_text(encoding="utf-8") == f"new:{field}"
        for field, path in output_paths.items()
    )
    assert all(
        Path(path).read_text(encoding="utf-8") == f"old:{field}"
        for field, path in archived.items()
    )
