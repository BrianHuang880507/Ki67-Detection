from __future__ import annotations

from dataclasses import replace
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def test_extract_rui49_parser_exposes_dedicated_mode() -> None:
    """CLI parser 必須公開 full Rui49 extraction mode。"""
    from immunity.exp4.run_rui2025 import build_parser

    args = build_parser().parse_args(["--config", "config.yaml", "--extract-rui49"])

    assert args.extract_rui49 is True


def test_run_cv_parser_exposes_single_formal_analysis_mode() -> None:
    """正式 CV 必須有穩定的單一路徑 CLI flag。"""
    from immunity.exp4.run_rui2025 import build_parser

    args = build_parser().parse_args(["--config", "config.yaml", "--run-cv"])

    assert args.run_cv is True


def test_run_exploratory_parser_exposes_post_cv_mode() -> None:
    """post-CV exploratory 必須公開獨立 CLI flag。"""
    from immunity.exp4.run_rui2025 import build_parser

    args = build_parser().parse_args(["--config", "config.yaml", "--run-exploratory"])

    assert args.run_exploratory is True


def test_run_cv_adapter_context_matches_core_standardization_contract() -> None:
    """adapter 傳給 core 的 LeakageContext scope 必須使用 exact contract 字串。"""
    from immunity.exp4.run_rui2025 import _build_cv_leakage_context

    context = _build_cv_leakage_context("a" * 64, "a" * 64)

    assert context.standardization_scope == "pipeline_fit_within_outer_training"
    assert context.feature_selection_inputs == ("X",)
    assert context.cellpose_invoked is False


def test_run_cv_mode_is_mutually_exclusive_with_safe_modes(monkeypatch) -> None:
    """CV 不得與 snapshot、border、safe 或 extraction mode 串接。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.setattr(
        run_rui2025,
        "_run_cv_analysis",
        lambda args: (_ for _ in ()).throw(
            AssertionError("mutually exclusive mode must stop before dispatch")
        ),
        raising=False,
    )

    for conflicting_mode in (
        "--validate-snapshot",
        "--measure-border-exclusion",
        "--run-safe",
        "--extract-rui49",
    ):
        assert (
            run_rui2025.main(
                ["--config", "config.yaml", "--run-cv", conflicting_mode]
            )
            == 2
        )


def test_run_exploratory_mode_is_mutually_exclusive_with_every_other_mode(
    monkeypatch,
) -> None:
    """post-CV exploratory 不得與任何既有 CLI mode 串接。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.setattr(
        run_rui2025,
        "_run_exploratory_analysis",
        lambda args: (_ for _ in ()).throw(
            AssertionError("mutually exclusive mode must stop before dispatch")
        ),
        raising=False,
    )

    for conflicting_mode in (
        "--validate-snapshot",
        "--measure-border-exclusion",
        "--run-safe",
        "--extract-rui49",
        "--run-cv",
    ):
        assert (
            run_rui2025.main(
                ["--config", "config.yaml", "--run-exploratory", conflicting_mode]
            )
            == 2
        )


def test_run_cv_formal_generation_gate_stops_before_core(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """formal generation lock 不符時，CV core 不得被呼叫。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    output_root = tmp_path / "immunity" / "outputs" / "exp4"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    config = {
        "source_root": str(source_root),
        "manifest_path": str(source_root / "manifest.csv"),
        "cell_level_path": str(source_root / "cells.csv"),
        "masks_dir": str(source_root / "masks"),
        "metadata_path": str(output_root / "run_metadata.json"),
        "expected": {
            "manifest_image_key_count": 693,
            "cell_row_count": 23976,
            "cell_image_key_count": 693,
            "group_count": 9,
            "group_keys": [
                "B4_P5", "B4_P6", "B4_P7", "B7_P5", "B7_P6", "B7_P7",
                "B8_P5", "B8_P6", "B8_P7",
            ],
        },
    }
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root.mkdir(parents=True)
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "validated",
                "full_rui49_status": "validated",
                "full_693_feature_consistency_status": "validated",
                "feature_health_status": "validated",
                "full_rui49": {
                    "raw_pair_row_count": 23976,
                    "pre_border_distinct_cell_count": 23012,
                    "retained_cell_count": 19000,
                    "fov_count": 693,
                },
            }
        ),
        encoding="utf-8",
    )
    calls = {"core": 0}
    monkeypatch.setattr(
        run_rui2025,
        "run_rui_cv",
        lambda *args, **kwargs: calls.__setitem__("core", calls["core"] + 1),
        raising=False,
    )

    exit_code = run_rui2025.main(["--config", str(config_path), "--run-cv"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert calls["core"] == 0
    assert "19648" in captured.err or "formal" in captured.err


def test_run_cv_failure_preserves_current_analysis_bundle(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """CV/core 或 publisher 失敗時，既有 canonical analysis files 必須保留。"""
    from types import SimpleNamespace

    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    output_root = tmp_path / "immunity" / "outputs" / "exp4"
    output_root.mkdir(parents=True)
    source_root = tmp_path / "source"
    source_root.mkdir()
    config = {
        "source_root": str(source_root),
        "manifest_path": str(source_root / "manifest.csv"),
        "cell_level_path": str(source_root / "cells.csv"),
        "masks_dir": str(source_root / "masks"),
        "metadata_path": str(output_root / "run_metadata.json"),
        "expected": {"manifest_image_key_count": 693, "cell_row_count": 23976,
                     "cell_image_key_count": 693, "dedup_cell_count": 23012,
                     "post_border_cell_count": 19648, "duplicate_key_count": 938,
                     "retained_raw_pair_rows": 20440, "group_count": 9,
                     "group_keys": ["B4_P5", "B4_P6", "B4_P7", "B7_P5",
                                    "B7_P6", "B7_P7", "B8_P5", "B8_P6", "B8_P7"]},
    }
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    old_metadata = {"status": "validated", "full_rui49_status": "validated",
                    "full_693_feature_consistency_status": "validated",
                    "feature_health_status": "validated",
                    "full_rui49": {"raw_pair_row_count": 23976,
                                   "pre_border_distinct_cell_count": 23012,
                                   "retained_cell_count": 19648, "fov_count": 693}}
    (output_root / "run_metadata.json").write_text(
        json.dumps(old_metadata), encoding="utf-8"
    )
    metrics_path = output_root / "cv_metrics.csv"
    metrics_path.write_text("old-analysis\n", encoding="utf-8")
    monkeypatch.setattr(
        run_rui2025,
        "validate_data_snapshot",
        lambda **_: SimpleNamespace(matched=True, differences=[]),
    )
    monkeypatch.setattr(run_rui2025, "_mask_cache_fingerprint", lambda *_: "b" * 64)
    monkeypatch.setattr(run_rui2025, "_validate_cv_current_generation", lambda **_: None)
    monkeypatch.setattr(
        run_rui2025,
        "_prepare_cv_analysis_inputs",
        lambda **_: (_ for _ in ()).throw(RuntimeError("synthetic cv")),
        raising=False,
    )
    monkeypatch.setattr(
        run_rui2025,
        "run_rui_cv",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic cv")),
        raising=False,
    )

    exit_code = run_rui2025.main(["--config", str(config_path), "--run-cv"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "synthetic cv" in captured.err
    assert metrics_path.read_text(encoding="utf-8") == "old-analysis\n"


def test_run_cv_success_stages_formal_analysis_bundle_without_umap(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """CV seam 成功時應同世代發布 65-health/E3/redundancy/preflight/CV outputs。"""
    from types import SimpleNamespace

    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    output_root = tmp_path / "immunity" / "outputs" / "exp4"
    source_root = tmp_path / "source"
    output_root.mkdir(parents=True)
    source_root.mkdir()
    group_keys = [
        "B4_P5", "B4_P6", "B4_P7", "B7_P5", "B7_P6", "B7_P7",
        "B8_P5", "B8_P6", "B8_P7",
    ]
    config = {
        "source_root": str(source_root),
        "manifest_path": str(source_root / "manifest.csv"),
        "cell_level_path": str(source_root / "cells.csv"),
        "masks_dir": str(source_root / "masks"),
        "metadata_path": str(output_root / "run_metadata.json"),
        "cv_n_jobs": 6,
        "importance_repeats": 3,
        "expected": {
            "manifest_image_key_count": 693,
            "cell_row_count": 23976,
            "cell_image_key_count": 693,
            "dedup_cell_count": 23012,
            "post_border_cell_count": 19648,
            "duplicate_key_count": 938,
            "retained_raw_pair_rows": 20440,
            "group_count": 9,
            "group_keys": group_keys,
        },
    }
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "validated",
                "full_rui49_status": "validated",
                "full_693_feature_consistency_status": "validated",
                "feature_health_status": "validated",
                "target_status": "validated",
                "target_sensitivity_status": "validated",
                "border_exclusion_status": "validated",
                "full_rui49": {
                    "raw_pair_row_count": 23976,
                    "pre_border_distinct_cell_count": 23012,
                    "retained_cell_count": 19648,
                    "retained_raw_pair_row_count": 20440,
                    "fov_count": 693,
                    "canonical_feature_count": 49,
                },
            }
        ),
        encoding="utf-8",
    )

    from immunity.exp4.cv import CV_METRIC_COLUMNS, FEATURE_IMPORTANCE_COLUMNS
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    health_columns = tuple(RUI49_FEATURE_COLUMNS) + tuple(
        f"nucleus__{name}"
        for name in (
            "area", "compactness", "eccentricity", "extent", "sphericity",
            "major_axis_length", "feret_length", "minor_axis_length", "feret_width",
            "maximum_radius", "mean_radius", "median_radius", "aspect_ratio",
            "perimeter_area_ratio", "perimeter", "solidity",
        )
    ) + ("nucleus_cytoplasm_area_ratio",)
    health_report = pd.DataFrame(
        {
            "feature": health_columns,
            "is_near_zero_variance": [False] * len(health_columns),
        }
    )
    health_result = SimpleNamespace(
        report=health_report,
        blocked=False,
        reasons=(),
        removed_columns=("cell__MinIntensity",),
        healthy_feature_columns=tuple(
            col for col in health_columns if col != "cell__MinIntensity"
        ),
        to_metadata_payload=lambda: {
            "blocked": False,
            "removed_columns": ["cell__MinIntensity"],
            "healthy_feature_columns": list(
                col for col in health_columns if col != "cell__MinIntensity"
            ),
        },
    )
    redundancy_report = pd.DataFrame({"record_type": ["feature_decision"]})
    redundancy_result = SimpleNamespace(
        report=redundancy_report,
        retained_features=tuple(RUI49_FEATURE_COLUMNS[:30]),
        to_metadata_payload=lambda: {
            "threshold": 0.9,
            "selection_inputs": ["X"],
            "retained_features": list(RUI49_FEATURE_COLUMNS[:30]),
            "removed_features": list(RUI49_FEATURE_COLUMNS[30:]),
            "rui28_intersection": [],
            "rui28_only": [],
            "filtered_only": [],
        },
    )
    fake_inputs = run_rui2025.CvAnalysisInputs(
        retained_cells=pd.DataFrame(),
        health_result=health_result,
        health_report=health_report,
        nucleus_sanity_report=pd.DataFrame({"metric": ["synthetic"]}),
        redundancy_result=redundancy_result,
        assembly=object(),
        source_fingerprint="a" * 64,
        raw_consistency=object(),
        retained_consistency=object(),
        raw_pair_row_count=23976,
        retained_raw_pair_row_count=20440,
        retained_fov_count=693,
        mask_cache_fingerprint_before="b" * 64,
        mask_cache_fingerprint_after="b" * 64,
    )
    preflight = SimpleNamespace(
        passed=True,
        report=pd.DataFrame(
            {"check_id": list(range(1, 8)), "status": ["passed"] * 7}
        ),
    )
    fake_result = SimpleNamespace(
        metrics=pd.DataFrame(columns=CV_METRIC_COLUMNS),
        oof_predictions=pd.DataFrame(),
        hyperparameters=pd.DataFrame(),
        per_group_residuals=pd.DataFrame(),
        feature_importance=pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS),
        preflight=preflight,
        checkpoint_manifest={"run_fingerprint": "c" * 64},
        hyperparameter_grids={"synthetic": []},
    )
    monkeypatch.setattr(run_rui2025, "validate_data_snapshot", lambda **_: SimpleNamespace(matched=True, differences=[]))
    monkeypatch.setattr(run_rui2025, "_validate_cv_current_generation", lambda **_: None)
    monkeypatch.setattr(run_rui2025, "_mask_cache_fingerprint", lambda *_: "b" * 64)
    monkeypatch.setattr(run_rui2025, "_prepare_cv_analysis_inputs", lambda **_: fake_inputs)
    monkeypatch.setattr(run_rui2025, "run_rui_cv", lambda *args, **kwargs: fake_result)

    exit_code = run_rui2025.main(["--config", str(config_path), "--run-cv"])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    for filename in (
        "feature_health_report.csv", "nucleus_sanity_report.csv",
        "feature_redundancy_report.csv", "leakage_preflight_report.csv",
        "cv_metrics.csv", "oof_predictions.csv", "hyperparameters.csv",
        "hyperparameter_grids.json", "feature_importance.csv",
        "per_group_residuals.csv", "REPORT.md", "run_metadata.json",
    ):
        assert (output_root / filename).is_file(), filename
    assert len(pd.read_csv(output_root / "feature_health_report.csv")) == 66
    metadata = json.loads((output_root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["analysis_status"]["cv"] == "validated"
    assert metadata["analysis_status"]["umap"] == "not_run"
    assert metadata["analysis_status"]["kmeans"] == "not_run"
    assert metadata["feature_health"]["report_row_count"] == 66
    assert metadata["feature_health"]["active_feature_count"] == 65
    assert metadata["cv_n_jobs"] == 6
    assert metadata["importance_repeats"] == 3


def test_extract_rui49_mode_is_mutually_exclusive_with_safe_modes(
    monkeypatch,
) -> None:
    """full extraction 不得與 snapshot、border 或 safe mode 串接。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.setattr(
        run_rui2025,
        "_run_full_rui49",
        lambda args: (_ for _ in ()).throw(
            AssertionError("mutually exclusive mode must stop before dispatch")
        ),
        raising=False,
    )

    for conflicting_mode in (
        "--validate-snapshot",
        "--measure-border-exclusion",
        "--run-safe",
    ):
        exit_code = run_rui2025.main(
            [
                "--config",
                "config.yaml",
                "--extract-rui49",
                conflicting_mode,
            ]
        )
        assert exit_code == 2


def test_production_config_declares_full_rui49_noncanonical_and_canonical_paths() -> None:
    """正式 config 必須鎖定 full CSV、health、log 與 hidden checkpoint 路徑。"""
    config_path = (
        Path(__file__).parents[1] / "immunity" / "configs" / "exp4_rui2025.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["cell_level_rui49_path"] == (
        "immunity/outputs/exp4/cell_level_rui49.csv"
    )
    assert config["feature_health_report_path"] == (
        "immunity/outputs/exp4/feature_health_report.csv"
    )
    assert config["run_log_path"] == "immunity/outputs/exp4/run.log"
    assert config["rui49_checkpoint_dir"] == "immunity/outputs/exp4/.rui49_checkpoints"
    assert config["expected"]["retained_raw_pair_rows"] == 20440
    assert config["dapi_label_masks_dir"] is None
    assert config["nucleus_dapi_validation_path"].endswith(
        "nucleus_dapi_validation.csv"
    )
    assert config["nucleus_dapi_validation_summary_path"].endswith(
        "nucleus_dapi_validation_summary.csv"
    )
    assert config["nucleus_sanity_report_path"].endswith(
        "nucleus_sanity_report.csv"
    )
    assert config["feature_redundancy_report_path"].endswith(
        "feature_redundancy_report.csv"
    )
    assert config["leakage_preflight_report_path"].endswith(
        "leakage_preflight_report.csv"
    )
    assert config["cv_metrics_path"].endswith("cv_metrics.csv")
    assert config["oof_predictions_path"].endswith("oof_predictions.csv")
    assert config["hyperparameters_path"].endswith("hyperparameters.csv")
    assert config["hyperparameter_grids_path"].endswith(
        "hyperparameter_grids.json"
    )
    assert config["feature_importance_path"].endswith("feature_importance.csv")
    assert config["per_group_residuals_path"].endswith("per_group_residuals.csv")
    assert config["cv_checkpoint_dir"].endswith(".cv_checkpoints")
    assert config["cv_n_jobs"] == 6
    assert config["importance_repeats"] == 3
    assert config["ido_background_qc"]["metric"]["pilot_80_fov"] == (
        "IDO_background_median (normalized [0,1])"
    )
    assert config["ido_background_qc"]["metric"]["full_693"] == (
        "extraction_background_median (raw gray)"
    )


def test_production_config_declares_restricted_post_cv_outputs_and_hashes() -> None:
    """正式 config 必須鎖定七項 post-CV outputs 與十個 CV hashes。"""
    config_path = (
        Path(__file__).parents[1] / "immunity" / "configs" / "exp4_rui2025.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["shrinkage_analysis_path"] == (
        "immunity/outputs/exp4/shrinkage_analysis.csv"
    )
    assert config["orientation_target_marginal_path"] == (
        "immunity/outputs/exp4/orientation_target_marginal.csv"
    )
    assert config["umap_embeddings_path"] == "immunity/outputs/exp4/umap_embeddings.csv"
    assert config["kmeans_cluster_features_path"] == (
        "immunity/outputs/exp4/kmeans_cluster_features.csv"
    )
    assert config["umap_by_condition_path"].endswith(
        "figures/umap_by_condition.png"
    )
    assert set(config["expected_cv_artifact_sha256"]) == {
        "cv_metrics.csv",
        "oof_predictions.csv",
        "per_group_residuals.csv",
        "feature_importance.csv",
        "hyperparameters.csv",
        "hyperparameter_grids.json",
        "feature_health_report.csv",
        "nucleus_sanity_report.csv",
        "feature_redundancy_report.csv",
        "leakage_preflight_report.csv",
    }


def test_exploratory_output_mapping_is_restricted_to_seven_artifacts_plus_metadata_report():
    """exploratory publisher mapping 不得攜帶既有 CV numeric outputs。"""
    from immunity.exp4.run_rui2025 import (
        POST_CV_ANALYSIS_FILE_FIELDS,
        _exploratory_output_paths,
    )

    paths = _exploratory_output_paths(
        yaml.safe_load(
            (
                Path(__file__).parents[1]
                / "immunity"
                / "configs"
                / "exp4_rui2025.yaml"
            ).read_text(encoding="utf-8")
        )
    )

    assert tuple(paths) == POST_CV_ANALYSIS_FILE_FIELDS
    assert "cv_metrics_path" not in paths
    assert paths["umap_by_condition_path"].name == "umap_by_condition.png"
def test_extract_mode_rejects_full_writable_paths_outside_current_output_root(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """full mode 必須在 snapshot/core 前拒絕 output containment 漂移。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    config = {
        "source_root": str(tmp_path / "source"),
        "manifest_path": str(tmp_path / "source" / "manifest.csv"),
        "cell_level_path": str(tmp_path / "source" / "cells.csv"),
        "masks_dir": str(tmp_path / "source" / "masks"),
        "metadata_path": str(tmp_path / "immunity" / "outputs" / "exp4" / "run_metadata.json"),
        "report_path": str(tmp_path / "immunity" / "outputs" / "exp4" / "REPORT.md"),
        "cell_level_rui49_path": str(tmp_path / "outside.csv"),
        "feature_health_report_path": str(
            tmp_path / "immunity" / "outputs" / "exp4" / "feature_health_report.csv"
        ),
        "run_log_path": str(tmp_path / "immunity" / "outputs" / "exp4" / "run.log"),
        "rui49_checkpoint_dir": str(
            tmp_path / "immunity" / "outputs" / "exp4" / ".rui49_checkpoints"
        ),
        "expected": {
            "manifest_image_key_count": 0,
            "cell_row_count": 0,
            "cell_image_key_count": 0,
            "group_count": 0,
            "group_keys": [],
        },
    }
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        run_rui2025,
        "validate_data_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("containment must be checked before snapshot")
        ),
    )

    exit_code = run_rui2025.main(
        ["--config", str(config_path), "--extract-rui49"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "containment" in captured.err or "output" in captured.err


def _full_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    """建立小型 CLI fixture；full locks 由 config 明確縮小。"""
    from immunity.exp3.feature_sets import PRIMARY_CELL_FEATURES

    source_root = tmp_path / "source"
    masks_dir = source_root / "masks"
    image_dir = source_root / "images"
    masks_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    image_keys = ("FOV_A", "FOV_B")
    manifest = pd.DataFrame(
        {
            "image_key": image_keys,
            "b_id": ("B4", "B4"),
            "passage": (5, 5),
            "group_id": ("B4_P5", "B4_P5"),
            "pc_path": ("images/FOV_A.png", "images/FOV_B.png"),
        }
    )
    rows: list[dict[str, object]] = []
    labels_a = np.zeros((6, 6), dtype=np.int32)
    labels_a[1:3, 1:3] = 1
    labels_a[1:3, 3:5] = 2
    labels_a[3:5, 1:3] = 3
    labels_a[0, 3:5] = 4
    labels_b = np.zeros((6, 6), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    labels_b[1:3, 3:5] = 2
    labels_b[3:5, 1:3] = 3
    np.savez(masks_dir / "FOV_A.npz", cell_mask=labels_a)
    np.savez(masks_dir / "FOV_B.npz", cell_mask=labels_b)
    for image_key, labels, values in (
        ("FOV_A", (1, 2, 3, 4), (1.0, 3.0, 5.0, 100.0)),
        ("FOV_B", (1, 2, 3), (2.0, 4.0, 6.0)),
    ):
        for nucleus_label, (cell_label, ido_score) in enumerate(
            zip(labels, values), start=1
        ):
            row: dict[str, object] = {
                "image_key": image_key,
                "cell_label": cell_label,
                "nucleus_label": nucleus_label,
                "IDO_score": ido_score,
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
    raw_cells = pd.DataFrame(rows)
    manifest_path = source_root / "data_manifest.csv"
    cell_path = source_root / "cell_level_basic.csv"
    manifest.to_csv(manifest_path, index=False)
    raw_cells.to_csv(cell_path, index=False)
    for image_key in image_keys:
        (image_dir / f"{image_key}.png").write_bytes(b"pc")

    output_root = tmp_path / "immunity" / "outputs" / "exp4"
    config: dict[str, object] = {
        "source_root": str(source_root),
        "manifest_path": str(manifest_path),
        "cell_level_path": str(cell_path),
        "masks_dir": str(masks_dir),
        "metadata_path": str(output_root / "run_metadata.json"),
        "report_path": str(output_root / "REPORT.md"),
        "cell_level_rui49_path": str(output_root / "cell_level_rui49.csv"),
        "feature_health_report_path": str(output_root / "feature_health_report.csv"),
        "run_log_path": str(output_root / "run.log"),
        "rui49_checkpoint_dir": str(output_root / ".rui49_checkpoints"),
        "ido_background_qc": {
            "summary": {
                "pilot_80_fov": "3 distinct normalized values: 8/255, 9/255, 10/255; counts 2/33/45; 78/80=97.5% in 9-10",
                "full_693": "11 distinct raw-gray values: 7-17",
            },
            "source": {
                "pilot_80_fov": "pilot.csv",
                "full_693": "full.csv",
            },
            "metric": {
                "pilot_80_fov": "IDO_background_median (normalized [0,1])",
                "full_693": "extraction_background_median (raw gray)",
            },
        },
        "group_targets_path": str(output_root / "group_targets.csv"),
        "border_exclusion_report_path": str(output_root / "border_exclusion_report.csv"),
        "group_target_sensitivity_path": str(
            output_root / "group_target_sensitivity.csv"
        ),
        "fov_ido_scores_path": str(output_root / "fov_ido_scores.csv"),
        "expected": {
            "manifest_image_key_count": 2,
            "cell_row_count": 7,
            "cell_image_key_count": 2,
            "dedup_cell_count": 7,
            "post_border_cell_count": 6,
            "duplicate_key_count": 0,
            "raw_multirow_key_count": 0,
            "retained_raw_pair_rows": 6,
            "sparse_fov_count": 0,
            "sparse_fov_keys": [],
            "group_count": 1,
            "group_keys": ["B4_P5"],
        },
    }
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, config, output_root


def _fake_full_result(*, blocked: bool = False):
    """建立符合 high dataclass 的小型 full result。"""
    from immunity.exp4.cell_dedup import WholeCellFeatureConsistencyResult
    from immunity.exp4.feature_health import evaluate_feature_health
    from immunity.exp4.full_rui49 import FullRui49Result
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    identity = pd.DataFrame(
        {
            "image_key": ["FOV_A", "FOV_A", "FOV_A", "FOV_A", "FOV_B", "FOV_B", "FOV_B"],
            "cell_label": [1, 2, 3, 4, 1, 2, 3],
            "nucleus_label": [1, 2, 3, 4, 1, 2, 3],
        }
    )
    values = {
        column: (
            np.full(len(identity), 3.0)
            if column == "cell__MinIntensity"
            else np.arange(len(identity), dtype=float) + index + 1.0
        )
        for index, column in enumerate(RUI49_FEATURE_COLUMNS)
    }
    raw_pair_features = pd.concat([identity, pd.DataFrame(values)], axis=1)
    pre_border_features = raw_pair_features.drop(columns="nucleus_label")
    retained_features = pre_border_features.iloc[[0, 1, 2, 4, 5, 6]].reset_index(
        drop=True
    )
    retained_fallback_flags = retained_features.copy()
    retained_fallback_flags.loc[:, RUI49_FEATURE_COLUMNS] = False
    health = evaluate_feature_health(
        retained_features.loc[:, RUI49_FEATURE_COLUMNS],
        fallback_counts={column: 0 for column in RUI49_FEATURE_COLUMNS},
        eligible_cell_count=len(retained_features),
        feature_columns=RUI49_FEATURE_COLUMNS,
        fallback_scope="full_retained_6_distinct_cells_2_fovs",
    )
    if blocked:
        health = replace(health, blocked=True, reasons=("synthetic health blocked",))
    pre_consistency = WholeCellFeatureConsistencyResult(
        scope="full_raw_join_7_pairs_7_cells_2_fovs",
        checked_row_count=7,
        checked_cell_count=7,
        checked_duplicate_key_count=0,
        feature_count=49,
    )
    retained_consistency = WholeCellFeatureConsistencyResult(
        scope="full_retained_raw_join_6_pairs_6_cells_2_fovs",
        checked_row_count=6,
        checked_cell_count=6,
        checked_duplicate_key_count=0,
        feature_count=49,
    )
    return FullRui49Result(
        raw_pair_features=raw_pair_features,
        pre_border_features=pre_border_features,
        retained_features=retained_features,
        retained_fallback_flags=retained_fallback_flags,
        retained_fallback_counts={column: 0 for column in RUI49_FEATURE_COLUMNS},
        retained_fallback_rates={column: 0.0 for column in RUI49_FEATURE_COLUMNS},
        pre_border_consistency=pre_consistency,
        retained_consistency=retained_consistency,
        health=health,
        healthy_feature_ranges={
            column: {"min": 1.0, "max": 9.0, "finite": True}
            for column in health.healthy_feature_columns
        },
        processed_fov_count=2,
        resumed_fov_count=0,
        fov_timings=(),
        total_elapsed_seconds=1.25,
        extractor_fingerprint="synthetic-fingerprint",
        checkpoint_provenance=(),
    )


def _patch_full_snapshot(monkeypatch) -> None:
    from immunity.exp4.data_snapshot import SnapshotResult

    monkeypatch.setattr(
        "immunity.exp4.run_rui2025.validate_data_snapshot",
        lambda **kwargs: SnapshotResult(
            expected={}, actual={}, matched=True, differences=[], observations=[], sources={}
        ),
    )


def test_extract_validated_publishes_full_schema_and_preserves_target_csv(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """validated full mode 發布 canonical full outputs，且不覆寫 target CSV。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    config_path, config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    target_path = output_root / "group_targets.csv"
    target_path.write_text("historical-target\n", encoding="utf-8")
    _patch_full_snapshot(monkeypatch)
    calls: dict[str, object] = {}

    def fake_extract(**kwargs):
        calls.update(kwargs)
        return _fake_full_result()

    monkeypatch.setattr(run_rui2025, "extract_full_rui49", fake_extract, raising=False)

    exit_code = run_rui2025.main(["--config", str(config_path), "--extract-rui49"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["mode"] == "extract-rui49"
    assert summary["status"] == "validated"
    assert all(Path(path).is_absolute() for path in calls["manifest"]["pc_path"])
    raw_output = pd.read_csv(Path(config["cell_level_rui49_path"]))
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    assert list(raw_output.columns) == [
        "image_key",
        "cell_label",
        "nucleus_label",
        *RUI49_FEATURE_COLUMNS,
    ]
    assert len(raw_output) == 7
    assert "IDO_score" not in raw_output.columns
    health_output = pd.read_csv(Path(config["feature_health_report_path"]))
    assert len(health_output) == 49
    assert health_output["eligible_cell_count"].eq(6).all()
    metadata = json.loads(Path(config["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["status"] == "validated"
    assert metadata["full_rui49"]["status"] == "validated"
    assert metadata["full_rui49"]["retained_cell_count"] == 6
    assert metadata["full_rui49"]["fov_count"] == 2
    assert metadata["full_rui49"]["retained_consistency"]["checked_row_count"] == 6
    assert metadata["full_693_feature_consistency_status"] == "validated"
    assert metadata["feature_health_status"] == "validated"
    assert metadata["validation_status"]["feature_health"] == "validated"
    assert metadata["ido_background_qc"]["metric"]["full_693"] == (
        "extraction_background_median (raw gray)"
    )
    assert metadata["background_subtraction_method"] == "rolling_ball_radius_50"
    assert metadata["background_subtraction_status"] == "full_validated"
    assert metadata["haralick_method"] == "mahotas.features.haralick"
    assert metadata["haralick_distance"] == 3
    assert metadata["haralick_zero_reserved_for_padding"] is True
    assert metadata["python"] == platform.python_version()
    assert metadata["python_version"] == platform.python_version()
    assert metadata["package_versions"] == metadata["packages"]
    assert target_path.read_text(encoding="utf-8") == "historical-target\n"
    assert Path(config["report_path"]).is_file()


def test_full_metadata_records_feret_deviation_effective_arms_and_e2_canceled(
    tmp_path: Path,
) -> None:
    """full metadata 必須保留 Feret 偏離、48 欄 arm 與 E2 fail-closed 狀態。"""
    from immunity.exp4 import run_rui2025

    _config_path, config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    result = _fake_full_result()
    metadata = run_rui2025._build_full_metadata(
        {},
        full_result=result,
        output_paths={
            "metadata_path": Path(config["metadata_path"]),
            "report_path": Path(config["report_path"]),
            "cell_level_rui49_path": Path(config["cell_level_rui49_path"]),
            "feature_health_report_path": Path(config["feature_health_report_path"]),
            "run_log_path": Path(config["run_log_path"]),
            "rui49_checkpoint_dir": Path(config["rui49_checkpoint_dir"]),
        },
        current_paths={
            "nucleus_dapi_validation_path": output_root / "nucleus_dapi_validation.csv",
            "nucleus_dapi_validation_summary_path": output_root / "nucleus_dapi_validation_summary.csv",
        },
        generation_id="g",
        expected_pre_border_cells=7,
        expected_retained_cells=6,
        expected_fov_count=2,
        ido_background_qc=None,
    )

    assert metadata["deviations"][0]["id"] == "feret_caliper"
    assert "cell__feret_width" in metadata["deviations"][0]["original_behavior"]
    assert "convex-hull" in metadata["deviations"][0]["new_behavior"]
    assert "MinFeret > MaxFeret" in metadata["deviations"][0]["reason"]
    assert set(metadata["feature_arms"]) == {
        "geometry_24",
        "rui_48",
        "rui_filtered",
        "rui_48_plus_nucleus",
    }
    assert metadata["feature_arms"]["rui_48"]["count"] == 48
    assert metadata["feature_arms"]["rui_48_plus_nucleus"]["count"] == 65
    assert metadata["effective_rui_feature_count"] == 48
    assert metadata["e2"]["status"] == "canceled_v5_no_dapi_acquired"
    assert metadata["e2"]["dapi_label_cache_available"] is False


def test_full_metadata_v5_cancels_e2_and_records_sanity_cv_wiring(
    tmp_path: Path,
) -> None:
    """v5 應取消 E2 並保留無 DAPI provenance，而非宣稱 blocked validation。"""
    from immunity.exp4 import run_rui2025

    _config_path, config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    result = _fake_full_result()
    metadata = run_rui2025._build_full_metadata(
        {
            "e2": {
                "status": "blocked",
                "reason": "legacy missing DAPI evidence",
            }
        },
        full_result=result,
        output_paths={
            "metadata_path": Path(config["metadata_path"]),
            "report_path": Path(config["report_path"]),
            "cell_level_rui49_path": Path(config["cell_level_rui49_path"]),
            "feature_health_report_path": Path(config["feature_health_report_path"]),
            "run_log_path": Path(config["run_log_path"]),
            "rui49_checkpoint_dir": Path(config["rui49_checkpoint_dir"]),
        },
        current_paths={},
        generation_id="g-v5",
        expected_pre_border_cells=7,
        expected_retained_cells=6,
        expected_fov_count=2,
        ido_background_qc=None,
    )

    assert metadata["e2"]["status"] == "canceled_v5_no_dapi_acquired"
    assert metadata["e2"]["dapi_acquired"] is False
    assert metadata["e2"]["reason"]
    assert metadata["e2"]["provenance"]["dataset_count"] == 9
    assert metadata["e2"]["provenance"]["fov_count"] == 693
    assert metadata["validation_status"]["e2"] == "canceled_v5_no_dapi_acquired"
    assert metadata["near_zero_variance_threshold"] == 0.05
    assert metadata["near_zero_variance_action"] == "record_only"


def test_full_mode_blocks_when_validated_reporting_artifact_is_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """validated current target chain 缺 artifact 時必須在 full publish 前 fail-closed。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    config_path, config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(config["metadata_path"]).write_text(
        json.dumps(
            {
                "target_status": "validated",
                "target_sensitivity_status": "validated",
                "border_exclusion_status": "validated",
            }
        ),
        encoding="utf-8",
    )
    _patch_full_snapshot(monkeypatch)
    monkeypatch.setattr(
        run_rui2025,
        "extract_full_rui49",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("validated artifacts must gate before extraction")
        ),
        raising=False,
    )

    exit_code = run_rui2025.main(["--config", str(config_path), "--extract-rui49"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 2
    assert "group_targets_path" in summary["error"]


def test_full_mode_accepts_configured_synthetic_reporting_artifact_counts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """validated reporting gate 使用 config expected counts，fixture 不綁死 9/693。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    config_path, config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "group_id": "B4_P5",
                "b_id": "B4",
                "passage": 5,
                "fov_count": 2,
                "cells_before": 7,
                "cells_after": 6,
                "group_IDO_score": 3.0,
                "group_IDO_score_all_cells": 3.0,
                "delta": 0.0,
                "confidence_flag": "normal",
            }
        ]
    ).to_csv(config["group_targets_path"], index=False)
    pd.DataFrame(
        [
            {
                "group_id": "B4_P5",
                "sparse_fov_count": 0,
                "target_with_sparse_fovs": 3.0,
                "target_without_sparse_fovs": 3.0,
                "delta_without_minus_with": 0.0,
                "absolute_delta": 0.0,
                "baseline_target_range": 0.0,
                "absolute_delta_pct_of_range": 0.0,
                "threshold_pct": 5.0,
                "threshold_absolute": 0.0,
                "status": "passed",
            }
        ]
    ).to_csv(config["group_target_sensitivity_path"], index=False)
    border_rows = []
    for image_key in ("FOV_A", "FOV_B"):
        border_rows.append(
            {
                "image_key": image_key,
                "b_id": "B4",
                "passage": 5,
                "group_id": "B4_P5",
                "cells_before": 4,
                "cells_excluded_border": 1,
                "cells_after": 3,
                "whole_cell_labels_before": 4,
                "whole_cell_labels_excluded_border": 1,
                "whole_cell_labels_after": 3,
                "border_exclusion_fraction": 0.25,
                "border_exclusion_row_fraction": 0.25,
                "minimum_cell_gate_status": "passed",
                "raw_pair_cells_before": 4,
                "raw_pair_cells_excluded_border": 1,
                "raw_pair_cells_after": 3,
                "raw_pair_exclusion_fraction": 0.25,
            }
        )
    pd.DataFrame(border_rows).to_csv(config["border_exclusion_report_path"], index=False)
    Path(config["metadata_path"]).write_text(
        json.dumps(
            {
                "target_status": "validated",
                "target_sensitivity_status": "validated",
                "border_exclusion_status": "validated",
            }
        ),
        encoding="utf-8",
    )
    _patch_full_snapshot(monkeypatch)
    monkeypatch.setattr(
        run_rui2025,
        "extract_full_rui49",
        lambda **kwargs: _fake_full_result(),
        raising=False,
    )

    exit_code = run_rui2025.main(["--config", str(config_path), "--extract-rui49"])

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert json.loads(captured.out)["status"] == "validated"


def test_extract_health_blocked_writes_failure_evidence_without_replacing_bundle(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """health blocked 必須 exit2、寫 noncanonical evidence 並保留舊 full bundle。"""
    from immunity.exp4 import run_rui2025

    monkeypatch.chdir(tmp_path)
    config_path, config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    old_outputs = {
        "cell": Path(config["cell_level_rui49_path"]),
        "health": Path(config["feature_health_report_path"]),
        "metadata": Path(config["metadata_path"]),
        "report": Path(config["report_path"]),
    }
    old_bytes = {}
    for key, path in old_outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"old-" + key.encode("ascii")
        if key == "metadata":
            payload = b'{"status":"old"}\n'
        path.write_bytes(payload)
        old_bytes[key] = payload
    _patch_full_snapshot(monkeypatch)
    monkeypatch.setattr(
        run_rui2025,
        "extract_full_rui49",
        lambda **kwargs: _fake_full_result(blocked=True),
        raising=False,
    )

    exit_code = run_rui2025.main(["--config", str(config_path), "--extract-rui49"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 2
    assert summary["status"] == "blocked"
    evidence_path = Path(summary["failure_evidence_path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["failure_kind"] == "health_blocked"
    assert evidence["health"]["blocked"] is True
    assert evidence["health"]["reasons"] == ["synthetic health blocked"]
    for key, path in old_outputs.items():
        assert path.read_bytes() == old_bytes[key]


def test_extract_typed_nonfinite_failure_writes_structured_evidence(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """high adapter 的 typed nonfinite gate 必須保留完整 structured evidence。"""
    from immunity.exp4 import run_rui2025
    from immunity.exp4.full_rui49 import (
        FullRui49BlockedDiagnostics,
        FullRui49CanonicalBlockedError,
        FullRui49ConsistencyDiagnostic,
    )

    monkeypatch.chdir(tmp_path)
    config_path, _config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _patch_full_snapshot(monkeypatch)
    result = _fake_full_result()
    diagnostic = FullRui49BlockedDiagnostics(
        failure_kind="blocked_nonfinite",
        nonfinite_columns=("cell__MeanIntensity",),
        raw_pair_features=result.raw_pair_features,
        pre_border_features=result.pre_border_features,
        retained_features=result.retained_features,
        retained_fallback_flags=result.retained_fallback_flags,
        retained_fallback_counts=result.retained_fallback_counts,
        retained_fallback_rates=result.retained_fallback_rates,
        pre_border_consistency=FullRui49ConsistencyDiagnostic(
            status="not_run_nonfinite",
            scope="full_raw",
            checked_row_count=7,
            checked_cell_count=7,
            checked_duplicate_key_count=0,
            feature_count=49,
            reason="synthetic nonfinite",
        ),
        retained_consistency=FullRui49ConsistencyDiagnostic(
            status="not_run_nonfinite",
            scope="full_retained",
            checked_row_count=6,
            checked_cell_count=6,
            checked_duplicate_key_count=0,
            feature_count=49,
            reason="synthetic nonfinite",
        ),
        health=result.health,
        healthy_feature_ranges=result.healthy_feature_ranges,
        processed_fov_count=2,
        resumed_fov_count=0,
        fov_timings=(),
        total_elapsed_seconds=1.0,
        extractor_fingerprint="synthetic",
        checkpoint_provenance=(),
    )
    monkeypatch.setattr(
        run_rui2025,
        "extract_full_rui49",
        lambda **kwargs: (_ for _ in ()).throw(
            FullRui49CanonicalBlockedError(diagnostic)
        ),
        raising=False,
    )

    exit_code = run_rui2025.main(["--config", str(config_path), "--extract-rui49"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 2
    evidence = json.loads(
        Path(summary["failure_evidence_path"]).read_text(encoding="utf-8")
    )
    assert evidence["failure_kind"] == "blocked_nonfinite"
    assert evidence["nonfinite_columns"] == ["cell__MeanIntensity"]
    assert evidence["retained_consistency"]["status"] == "not_run_nonfinite"


def test_full_metadata_repoints_smoke_health_to_historical_archive(
    tmp_path: Path,
) -> None:
    """full publish 後 smoke health 不得繼續指向 canonical full CSV。"""
    from immunity.exp4 import run_rui2025

    _config_path, config, output_root = _full_fixture(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_path = Path(config["feature_health_report_path"])
    smoke_path.write_text("smoke-history", encoding="utf-8")
    existing = {
        "feature_smoke_status": "smoke_validated",
        "feature_extraction_status": "smoke_validated",
        "feature_health_smoke": {
            "status": "smoke_provisional_passed",
            "report_path": str(smoke_path),
        },
        "smoke_test": {
            "status": "passed",
            "report_path": str(output_root / "feature_smoke_report.json"),
            "texture_feature_ranges_before_zero_reservation": {
                "cell__Contrast": {"min": 1.0, "max": 2.0}
            },
            "texture_feature_ranges_after_zero_reservation": {
                "cell__Contrast": {"min": 3.0, "max": 4.0}
            },
            "feature_health": {
                "status": "smoke_provisional_passed",
                "report_path": str(smoke_path),
                "texture_feature_ranges_before_zero_reservation": {
                    "cell__Contrast": {"min": 1.0, "max": 2.0}
                },
                "texture_feature_ranges_after_zero_reservation": {
                    "cell__Contrast": {"min": 3.0, "max": 4.0}
                },
            },
        },
    }
    archive_path = smoke_path.with_name("feature_health_report.csv.historical_stale.g")
    metadata = run_rui2025._build_full_metadata(
        existing,
        full_result=_fake_full_result(),
        output_paths={
            "metadata_path": Path(config["metadata_path"]),
            "report_path": Path(config["report_path"]),
            "cell_level_rui49_path": Path(config["cell_level_rui49_path"]),
            "feature_health_report_path": smoke_path,
            "run_log_path": Path(config["run_log_path"]),
            "rui49_checkpoint_dir": Path(config["rui49_checkpoint_dir"]),
        },
        current_paths={},
        generation_id="g",
        expected_pre_border_cells=7,
        expected_retained_cells=6,
        expected_fov_count=2,
        ido_background_qc=config["ido_background_qc"],
        planned_archived_outputs={
            "feature_health_report_path": str(archive_path),
        },
    )

    smoke = metadata["feature_health_smoke"]
    assert smoke["status"] == "historical_stale"
    assert smoke["report_path"] == str(archive_path)
    assert smoke["report_path"] != str(smoke_path)
    nested_smoke = metadata["smoke_test"]
    assert nested_smoke["status"] == "historical_stale"
    assert nested_smoke["feature_health"]["status"] == "historical_stale"
    assert nested_smoke["feature_health"]["report_path"] == str(archive_path)
    assert nested_smoke["feature_health"]["report_path"] != str(smoke_path)
    assert (
        nested_smoke["texture_feature_ranges_before_zero_reservation"][
            "cell__Contrast"
        ]
        == {"min": 1.0, "max": 2.0}
    )
    assert (
        nested_smoke["texture_feature_ranges_after_zero_reservation"]["cell__Contrast"]
        == {"min": 3.0, "max": 4.0}
    )
    assert (
        nested_smoke["feature_health"][
            "texture_feature_ranges_before_zero_reservation"
        ]["cell__Contrast"]
        == {"min": 1.0, "max": 2.0}
    )
    assert metadata["feature_health_status"] == "validated"
    assert metadata["validation_status"]["feature_health"] == "validated"


def test_publish_bundle_rolls_back_when_staged_commit_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """real publisher 的中途 commit 失敗必須恢復舊四檔 bundle。"""
    from immunity.exp4 import run_rui2025

    output_root = tmp_path / "outputs"
    staging_root = tmp_path / "staging"
    output_root.mkdir()
    staging_root.mkdir()
    fields = (
        "cell_level_rui49_path",
        "feature_health_report_path",
        "report_path",
        "metadata_path",
    )
    output_paths = {
        field: output_root / f"{field}.old"
        for field in fields
    }
    staged_paths = {
        field: staging_root / f"{field}.new"
        for field in fields
    }
    old_bytes = {}
    for field in fields:
        old_bytes[field] = f"old-{field}".encode("utf-8")
        output_paths[field].write_bytes(old_bytes[field])
        staged_paths[field].write_text(f"new-{field}", encoding="utf-8")

    real_replace = run_rui2025.os.replace
    calls = {"count": 0}

    def fail_once(source, destination):
        calls["count"] += 1
        if calls["count"] == 6:
            raise OSError("synthetic staged commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(run_rui2025.os, "replace", fail_once)

    with pytest.raises(run_rui2025.BundlePublishError):
        run_rui2025._publish_bundle(
            staged_paths,
            output_paths,
            fields=fields,
            generation_id="synthetic-generation",
        )

    for field, path in output_paths.items():
        assert path.read_bytes() == old_bytes[field]
    assert not staged_paths[fields[0]].exists()
    assert all(staged_paths[field].exists() for field in fields[1:])


def test_publish_bundle_rolls_back_when_post_publish_gate_fails(
    tmp_path: Path,
) -> None:
    """post-publish gate 失敗時必須恢復完整舊 canonical bundle。"""
    from immunity.exp4 import run_rui2025

    output_root = tmp_path / "outputs"
    staging_root = tmp_path / "staging"
    output_root.mkdir()
    staging_root.mkdir()
    fields = (
        "shrinkage_analysis_path",
        "orientation_target_marginal_path",
        "report_path",
        "metadata_path",
    )
    output_paths = {field: output_root / f"{field}.old" for field in fields}
    staged_paths = {field: staging_root / f"{field}.new" for field in fields}
    old_bytes = {}
    for field in fields:
        old_bytes[field] = f"old-{field}".encode("utf-8")
        output_paths[field].write_bytes(old_bytes[field])
        staged_paths[field].write_text(f"new-{field}", encoding="utf-8")

    observed = {"called": False}

    def fail_after_publish() -> None:
        observed["called"] = True
        assert all(path.is_file() for path in output_paths.values())
        raise ValueError("synthetic post-publish gate failure")

    with pytest.raises(run_rui2025.BundlePublishError, match="post-publish"):
        run_rui2025._publish_bundle(
            staged_paths,
            output_paths,
            fields=fields,
            generation_id="post-gate-generation",
            post_publish_gate=fail_after_publish,
        )

    assert observed["called"] is True
    for field, path in output_paths.items():
        assert path.read_bytes() == old_bytes[field]
    assert not any(
        path.name.endswith(".backup.post-gate-generation")
        or path.name.endswith(".historical_stale.post-gate-generation")
        for path in output_root.iterdir()
    )


def test_publish_analysis_bundle_keeps_checkpoint_directory_outside_file_transaction(
    tmp_path: Path,
) -> None:
    """analysis publisher 不得把既有 CV checkpoint directory 搬進 archive。"""
    from immunity.exp4 import run_rui2025

    output_root = tmp_path / "outputs"
    staging_root = tmp_path / "staging"
    output_root.mkdir()
    staging_root.mkdir()
    checkpoint_dir = output_root / ".cv_checkpoints"
    checkpoint_dir.mkdir()
    marker = checkpoint_dir / "checkpoint.bin"
    marker.write_bytes(b"checkpoint")
    output_paths = {
        "cv_metrics_path": output_root / "cv_metrics.csv",
        "cv_checkpoint_dir": checkpoint_dir,
    }
    staged_paths = {
        "cv_metrics_path": staging_root / "cv_metrics.csv",
        "cv_checkpoint_dir": staging_root / ".cv_checkpoints",
    }
    staged_paths["cv_metrics_path"].write_text("model\n", encoding="utf-8")

    archived = run_rui2025.publish_analysis_bundle(
        staged_paths,
        output_paths,
        fields=("cv_metrics_path",),
        generation_id="analysis-g",
    )

    assert output_paths["cv_metrics_path"].read_text(encoding="utf-8") == "model\n"
    assert marker.read_bytes() == b"checkpoint"
    assert "cv_checkpoint_dir" not in archived


def _exploratory_adapter_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    """建立 post-CV adapter 的最小 transaction fixture。"""
    output_root = tmp_path / "immunity" / "outputs" / "exp4"
    source_root = tmp_path / "source"
    output_root.mkdir(parents=True)
    source_root.mkdir()
    (source_root / "masks").mkdir()
    manifest_path = source_root / "manifest.csv"
    pd.DataFrame(
        {
            "image_key": ["FOV1"],
            "b_id": ["B4"],
            "passage": ["P5"],
            "condition": ["synthetic"],
            "ifn_dose": [1],
            "tnf_dose": [0],
            "group_id": ["B4_P5"],
        }
    ).to_csv(manifest_path, index=False)
    (source_root / "cells.csv").write_text(
        "image_key,cell_label\nFOV1,1\n", encoding="utf-8"
    )
    metadata_path = output_root / "run_metadata.json"
    roster = [f"feature_{index}" for index in range(30)]
    metadata = {
        "generation_id": "formal-generation",
        "status": "validated",
        "full_rui49_status": "validated",
        "full_693_feature_consistency_status": "validated",
        "feature_health_status": "validated",
        "target_status": "validated",
        "target_sensitivity_status": "validated",
        "border_exclusion_status": "validated",
        "full_rui49": {
            "raw_pair_row_count": 23976,
            "pre_border_distinct_cell_count": 23012,
            "retained_cell_count": 19648,
            "retained_raw_pair_row_count": 20440,
            "fov_count": 693,
            "canonical_feature_count": 49,
        },
        "analysis_status": {"cv": "validated", "umap": "not_run", "kmeans": "not_run"},
        "feature_arms": {"rui_filtered": {"feature_columns": roster}},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (output_root / "REPORT.md").write_text("old report\n", encoding="utf-8")
    (output_root / "cv_metrics.csv").write_text("protected cv\n", encoding="utf-8")
    config = {
        "source_root": str(source_root),
        "manifest_path": str(manifest_path),
        "cell_level_path": str(source_root / "cells.csv"),
        "masks_dir": str(source_root / "masks"),
        "metadata_path": str(metadata_path),
        "expected": {
            "manifest_image_key_count": 693,
            "cell_row_count": 23976,
            "cell_image_key_count": 693,
            "dedup_cell_count": 23012,
            "post_border_cell_count": 19648,
            "duplicate_key_count": 938,
            "retained_raw_pair_rows": 20440,
            "group_count": 9,
            "group_keys": [
                "B4_P5", "B4_P6", "B4_P7", "B7_P5", "B7_P6", "B7_P7",
                "B8_P5", "B8_P6", "B8_P7",
            ],
        },
    }
    return metadata_path, config, output_root


def test_run_exploratory_success_never_invokes_cv_and_publishes_restricted_bundle(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """post-CV success 不得呼叫 CV，且 publisher mapping 僅含新 artifacts。"""
    from types import SimpleNamespace

    from immunity.exp4 import exploratory, post_cv_analysis, run_rui2025

    metadata_path, config, output_root = _exploratory_adapter_fixture(tmp_path)
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    feature_columns = tuple(f"feature_{index}" for index in range(30))
    retained = pd.DataFrame(
        {
            "image_key": ["FOV1"],
            "cell_label": [1],
            **{name: [1.0] for name in feature_columns},
        }
    )
    fake_inputs = SimpleNamespace(
        retained_cells=retained,
        redundancy_result=SimpleNamespace(retained_features=feature_columns),
    )
    embedding = pd.DataFrame(
        [["FOV1", 1, "B4", "P5", "synthetic", 1, 0, "B4_P5", 0.1, 0.2, True, 0, "IDO_low"]],
        columns=list(exploratory.EMBEDDING_COLUMNS),
    )
    fake_exploratory = exploratory.ExploratoryAnalysisResult(
        embedding=embedding,
        cluster_assignments=embedding.copy(),
        cluster_feature_report=pd.DataFrame(
            [[name, 0, 1, 1, 1, 1, 2, 1, 2, 1, 1] for name in feature_columns],
            columns=list(exploratory.CLUSTER_FEATURE_REPORT_COLUMNS),
        ),
        figures={
            "umap_by_donor.png": b"donor",
            "umap_by_passage.png": b"passage",
            "umap_by_condition.png": b"condition",
        },
        timing={"total_seconds": 0.1},
        provenance={
            "feature_columns": feature_columns,
            "umap_fit_columns": feature_columns,
            "kmeans_fit_columns": ("UMAP1", "UMAP2"),
            "random_state": 42,
            "ifn_selector": "ifn_dose > 0",
            "ifn_selected_cell_count": 1,
            "ifn_selected_fov_count": 1,
            "ifn_selected_group_count": 1,
            "target_median_aggregation_unit": "cell-weighted",
            "standard_scaler_fit_scope": "full_exploratory_population",
            "umap_actual_params": {},
            "kmeans_actual_params": {},
            "report_feature_units": "raw",
            "umap_input_units": "scaled",
        },
    )
    fake_shrinkage = SimpleNamespace(
        report=pd.DataFrame(
            [["geometry_24__RFR", "geometry_24", "RFR", 19648, 693, 9, 0.1, 1.0, 0.1, 0.2, 0.3, 0.01, 0.5, 2.4, 1.9, 1.0, 1.5, 0.5, 0.2]],
            columns=list(post_cv_analysis.SHRINKAGE_REPORT_COLUMNS),
        ),
        group_summary=pd.DataFrame(),
    )
    fake_orientation = SimpleNamespace(
        bins=pd.DataFrame(
            [[index, -1.0, 1.0, 0.0, -57.0, 57.0, 0.0, 1, 1, 1, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] for index in range(1, 13)],
            columns=list(post_cv_analysis.ORIENTATION_BIN_COLUMNS),
        ),
        summary=pd.DataFrame(
            [[19648, 693, 9, 12, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, True, "descriptive_noncausal"]],
            columns=list(post_cv_analysis.ORIENTATION_SUMMARY_COLUMNS),
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_rui2025, "validate_data_snapshot", lambda **_: SimpleNamespace(matched=True, differences=[]))
    monkeypatch.setattr(run_rui2025, "_validate_cv_current_generation", lambda **_: None)
    monkeypatch.setattr(run_rui2025, "_mask_cache_fingerprint", lambda *_: "a" * 64)
    monkeypatch.setattr(run_rui2025, "_prepare_cv_analysis_inputs", lambda **_: fake_inputs)
    monkeypatch.setattr(run_rui2025, "_validate_exploratory_reconstruction", lambda *_: feature_columns)
    monkeypatch.setattr(run_rui2025, "_read_post_cv_inputs", lambda *_: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(run_rui2025, "_validate_post_cv_input_contract", lambda **_: None)
    monkeypatch.setattr(run_rui2025, "_select_post_cv_best_configuration", lambda *_: {"configuration_id": "geometry_24__RFR", "mean_rui_r2": 0.2, "mean_cell_r2": 0.1, "slope": 0.1, "intercept": 1.0, "observed_min": 0.5, "observed_max": 2.4, "predicted_min": 1.0, "predicted_max": 1.5, "observed_range": 1.9})
    monkeypatch.setattr(run_rui2025, "_build_orientation_metadata", lambda *_: {"n_bins": 12, "cell_spearman_r": 0.0, "group_spearman_r": 0.0, "bin_target_median_range": 0.0, "bin_target_mean_range": 0.0})
    monkeypatch.setattr(run_rui2025, "analyze_shrinkage", lambda *_: fake_shrinkage)
    monkeypatch.setattr(run_rui2025, "analyze_orientation_marginal", lambda *_: fake_orientation)
    monkeypatch.setattr(run_rui2025, "run_exploratory_analysis", lambda *_: fake_exploratory)
    monkeypatch.setattr(run_rui2025, "_validate_protected_cv_hashes", lambda *_: {"cv_metrics.csv": "A" * 64})
    monkeypatch.setattr(run_rui2025, "_checkpoint_fingerprint", lambda *_: {"sha256": "B" * 64, "npz_count": 120})
    monkeypatch.setattr(run_rui2025, "run_rui_cv", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CV must not run")))

    assert run_rui2025.main(["--config", str(config_path), "--run-exploratory"]) == 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["generation_id"] == "formal-generation"
    assert metadata["ifn_selector_untrusted"] is True
    assert metadata["post_cv_analysis"]["ifn_selector_untrusted"] is True
    assert metadata["post_cv_analysis"]["umap"]["ifn_selector_untrusted"] is True
    assert (output_root / "umap_embeddings.csv").is_file()
    assert (output_root / "figures" / "umap_by_condition.png").read_bytes() == b"condition"
    assert (output_root / "cv_metrics.csv").read_text(encoding="utf-8") == "protected cv\n"
    assert "run-exploratory" in capsys.readouterr().out


def test_run_exploratory_rejects_post_cv_output_alias_with_protected_artifact(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """post-CV path alias protected CV artifact 時必須在 core 前 fail-closed。"""
    from immunity.exp4 import run_rui2025

    metadata_path, config, output_root = _exploratory_adapter_fixture(tmp_path)
    config["shrinkage_analysis_path"] = str(output_root / "cv_metrics.csv")
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = {"core": 0}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_rui2025,
        "_prepare_cv_analysis_inputs",
        lambda **_: calls.__setitem__("core", calls["core"] + 1),
    )

    assert run_rui2025.main(["--config", str(config_path), "--run-exploratory"]) == 2

    assert calls["core"] == 0
    assert "post-CV output path aliases protected artifact" in capsys.readouterr().err
    assert metadata_path.read_bytes()
    assert (output_root / "REPORT.md").read_text(encoding="utf-8") == "old report\n"


def test_run_exploratory_failure_keeps_previous_post_cv_bundle(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """post-CV core failure 必須保留既有 metadata、REPORT 與 artifacts。"""
    from types import SimpleNamespace

    from immunity.exp4 import run_rui2025

    metadata_path, config, output_root = _exploratory_adapter_fixture(tmp_path)
    old_artifact = output_root / "umap_embeddings.csv"
    old_artifact.write_text("old embedding\n", encoding="utf-8")
    config_path = tmp_path / "exp4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    metadata_before = metadata_path.read_bytes()
    report_before = (output_root / "REPORT.md").read_bytes()
    artifact_before = old_artifact.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_rui2025, "_validate_cv_current_generation", lambda **_: None)
    monkeypatch.setattr(run_rui2025, "_validate_protected_cv_hashes", lambda *_: {"cv_metrics.csv": "A" * 64})
    monkeypatch.setattr(run_rui2025, "_checkpoint_fingerprint", lambda *_: {"sha256": "B" * 64, "npz_count": 120})
    monkeypatch.setattr(run_rui2025, "validate_data_snapshot", lambda **_: SimpleNamespace(matched=True, differences=[]))
    monkeypatch.setattr(run_rui2025, "_mask_cache_fingerprint", lambda *_: "a" * 64)
    monkeypatch.setattr(run_rui2025, "_prepare_cv_analysis_inputs", lambda **_: (_ for _ in ()).throw(RuntimeError("synthetic exploratory")))

    assert run_rui2025.main(["--config", str(config_path), "--run-exploratory"]) == 2
    assert metadata_path.read_bytes() == metadata_before
    assert (output_root / "REPORT.md").read_bytes() == report_before
    assert old_artifact.read_bytes() == artifact_before
    assert "synthetic exploratory" in capsys.readouterr().err
