"""驗證 Exp3 Round 2 CLI 的設定與輸出邊界。"""

from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

import immunity.exp3.run_round2_paper93 as run_module
from immunity.exp3.run_round2_paper93 import (
    load_round2_config,
    resolve_round2_output_dir,
)


def _synthetic_config(tmp_path: Path) -> dict[str, object]:
    """建立 runner orchestration 測試用的最小設定。"""
    return {
        "round1": {
            "dir": str(tmp_path / "round1"),
            "expected": {
                "analyzed_images": 693,
                "valid_cells": 23976,
                "exclusions": 26,
            },
            "roster_sha256": "a" * 64,
            "artifact_sha256": {"source.csv": "b" * 64},
        },
        "features": {
            "min_finite_cells_per_feature": 3,
            "numeric_atol": 1e-12,
        },
        "benchmark": {
            "seed": 20260804,
            "max_hyperparameter_candidates": 24,
            "permutation_repeats": 20,
            "tree_estimators": 400,
        },
        "smoke": {
            "fovs_per_condition": 1,
            "max_hyperparameter_candidates": 1,
            "permutation_repeats": 2,
            "tree_estimators": 10,
        },
        "output": {"dir": str(_formal_output(tmp_path))},
    }


def _formal_output(tmp_path: Path) -> Path:
    """回傳 synthetic Round 2 formal output root。"""
    return tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93"


def _patch_round2_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[str],
) -> dict[str, object]:
    """以完整 public contract doubles 隔離 runner 的外部 stages。"""
    root = _formal_output(tmp_path).resolve()
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", root)
    images = pd.DataFrame(
        {
            "image_key": ["B1_C1_F02", "B1_C1_F01", "B1_C1_F03"],
            "group_id": ["G1", "G1", "G1"],
            "condition_index": [1, 1, 1],
            "fov": [2, 1, 3],
            "IDO_score": [0.2, 0.1, 0.3],
        }
    )
    evidence = SimpleNamespace(
        basic_images=images,
        basic_cells=pd.DataFrame({"image_key": images["image_key"]}),
        split_manifest=pd.DataFrame({"image_key": images["image_key"]}),
        artifact_hashes={"source.csv": "b" * 64},
        metadata={"winner": "extra_trees__basic_median"},
        root=tmp_path / "round1",
        selected_metrics=pd.DataFrame(columns=["model", "status"]),
        selected_predictions=pd.DataFrame(columns=["model", "image_key"]),
    )
    mask_qc = pd.DataFrame(
        {
            "image_key": images["image_key"],
            "pc_path": ["pc-2.tif", "pc-1.tif", "pc-3.tif"],
            "mask_path": ["mask-2.npz", "mask-1.npz", "mask-3.npz"],
            "status": ["passed", "passed", "passed"],
            "reason": ["", "", ""],
        }
    )
    bundle = SimpleNamespace(
        images=images,
        paper_cells=pd.DataFrame(
            {
                "image_key": [
                    "B1_C1_F02",
                    "B1_C1_F02",
                    "B1_C1_F01",
                    "B1_C1_F03",
                ],
                "cell_label": [1, 1, 1, 1],
                "nucleus_label": [1, 2, 1, 1],
                "nucleus_outside_fraction": [0.0, 0.05, 0.0, 0.0],
            }
        ),
        valid_counts=pd.DataFrame({"image_key": images["image_key"]}),
        extraction_qc=pd.DataFrame(
            {
                "image_key": images["image_key"],
                "status": ["passed", "passed", "passed"],
            }
        ),
        feature_qc=pd.DataFrame({"feature": ["f1"], "status": ["passed"]}),
        predictor_columns=("f1",),
        max_nucleus_outside_fraction=0.05,
    )
    result = SimpleNamespace(
        fold_metrics=pd.DataFrame({"model": ["extra_trees"]}),
        predictions=pd.DataFrame(),
        hyperparameters=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        failures=pd.DataFrame(),
    )
    comparison = SimpleNamespace(
        fold_metrics=result.fold_metrics,
        predictions=result.predictions,
        dummy_metrics=pd.DataFrame(),
        dummy_predictions=pd.DataFrame(),
        hyperparameters=result.hyperparameters,
        feature_importance=result.feature_importance,
        failures=result.failures,
    )
    ranking = pd.DataFrame(
        {
            "configuration_id": ["extra_trees__paper_style_median"],
            "eligible": [True],
            "recommended": [True],
        }
    )
    captures: dict[str, object] = {
        "evidence": evidence,
        "bundle": bundle,
        "mask_qc": mask_qc,
        "result": result,
    }

    def stage(name: str, value: object):
        def invoke(*args: object, **kwargs: object) -> object:
            calls.append(name)
            return value

        return invoke

    monkeypatch.setattr(run_module, "load_round1_evidence", stage("load_round1_evidence", evidence))
    monkeypatch.setattr(
        run_module,
        "require_formal_round1_evidence",
        stage("require_formal_round1_evidence", None),
    )
    monkeypatch.setattr(
        run_module,
        "validate_frozen_masks",
        stage("validate_frozen_masks", mask_qc),
    )
    monkeypatch.setattr(
        run_module,
        "extract_locked_paper93",
        stage("extract_locked_paper93", bundle),
    )
    monkeypatch.setattr(
        run_module,
        "require_paper93_preflight",
        stage("require_paper93_preflight", None),
    )
    monkeypatch.setattr(
        run_module,
        "restore_frozen_outer_splits",
        stage("restore_frozen_outer_splits", (object(),)),
    )
    monkeypatch.setattr(run_module, "run_paper93_benchmark", stage("run_paper93_benchmark", result))
    monkeypatch.setattr(
        run_module,
        "build_round2_comparison",
        stage("build_round2_comparison", comparison),
    )
    monkeypatch.setattr(
        run_module,
        "rank_round2_configurations",
        stage("rank_round2_configurations", ranking),
    )
    monkeypatch.setattr(
        run_module,
        "select_round2_recommendation",
        lambda value: str(value.iloc[0]["configuration_id"]),
    )
    monkeypatch.setattr(
        run_module,
        "expected_split_ids",
        lambda splits: {"family": ["family:fold"]},
    )
    monkeypatch.setattr(
        run_module,
        "outer_split_manifest",
        lambda frame, splits: pd.DataFrame({"image_key": frame["image_key"]}),
    )

    def write_bundle(staging: Path, tables: object, payloads: object, context: object) -> Path:
        calls.append("write_round2_bundle")
        captures.update(
            tables=tables,
            payloads=payloads,
            context=context,
            staging=staging,
            preexisting_bundle_artifacts={path.name for path in staging.iterdir()},
        )
        record = staging / "EXPERIMENT_RECORD.md"
        record.write_text("synthetic record\n", encoding="utf-8")
        return record

    def validate_bundle(directory: Path, *, smoke: bool) -> None:
        calls.append("validate_round2_bundle")
        captures["validated_smoke"] = smoke

    def publish(generation: object) -> None:
        calls.append("publish_round2_generation")
        output = generation.output_dir
        output.mkdir(parents=True, exist_ok=True)
        os.replace(generation.staging_dir / "run.log", output / "run.log")
        (output / "EXPERIMENT_RECORD.md").write_text("published\n", encoding="utf-8")

    monkeypatch.setattr(run_module, "write_round2_bundle", write_bundle)
    monkeypatch.setattr(run_module, "validate_round2_bundle", validate_bundle)
    monkeypatch.setattr(run_module, "publish_round2_generation", publish)
    return captures


def _create_directory_link(link: Path, target: Path) -> None:
    """建立目錄 symlink，必要時在 Windows 改用 junction。"""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_round2_orchestrator_runs_locked_stages_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formal runner 必須依鎖定順序完成證據、特徵、benchmark 與發布。"""
    calls: list[str] = []
    _patch_round2_stages(monkeypatch, tmp_path, calls)

    record = run_module.run_round2(_synthetic_config(tmp_path))

    assert calls == [
        "load_round1_evidence",
        "require_formal_round1_evidence",
        "validate_frozen_masks",
        "extract_locked_paper93",
        "require_paper93_preflight",
        "restore_frozen_outer_splits",
        "run_paper93_benchmark",
        "build_round2_comparison",
        "rank_round2_configurations",
        "write_round2_bundle",
        "validate_round2_bundle",
        "publish_round2_generation",
    ]
    assert record == _formal_output(tmp_path).resolve() / "EXPERIMENT_RECORD.md"


def test_round2_success_hands_clean_staging_to_atomic_bundle_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成功流程預寫的 QC 不得與 atomic bundle writer 的 destination 衝突。"""
    calls: list[str] = []
    captures = _patch_round2_stages(monkeypatch, tmp_path, calls)

    run_module.run_round2(_synthetic_config(tmp_path))

    assert captures["preexisting_bundle_artifacts"] == {"run.log"}


def test_round2_preflight_failure_quarantines_qc_and_logs_named_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature preflight 失敗須停止 training，並保留 traceback、影像與路徑。"""
    calls: list[str] = []
    captures = _patch_round2_stages(monkeypatch, tmp_path, calls)
    bundle = captures["bundle"]
    bundle.extraction_qc = pd.DataFrame(
        {
            "image_key": ["B8_P7_C08_F03"],
            "pc_path": ["D:/data/B8_P7_C08_F03_PC.tif"],
            "mask_path": ["D:/cache/B8_P7_C08_F03.npz"],
            "status": ["failed"],
            "reason": ["synthetic drift"],
        }
    )
    monkeypatch.setattr(
        run_module,
        "require_paper93_preflight",
        Mock(side_effect=ValueError("B8_P7_C08_F03 preflight failed")),
    )
    train = Mock(side_effect=AssertionError("model fit must not run"))
    monkeypatch.setattr(run_module, "run_paper93_benchmark", train)

    with pytest.raises(ValueError, match="B8_P7_C08_F03"):
        run_module.run_round2(_synthetic_config(tmp_path))

    train.assert_not_called()
    failed = list(
        (_formal_output(tmp_path).resolve() / "_generations").glob("failed-*")
    )
    assert len(failed) == 1
    assert (failed[0] / "extraction_qc.csv").is_file()
    log = (failed[0] / "run.log").read_text(encoding="utf-8")
    assert "ValueError" in log
    assert "Traceback" in log
    assert "B8_P7_C08_F03" in log
    assert "D:/data/B8_P7_C08_F03_PC.tif" in log
    assert "D:/cache/B8_P7_C08_F03.npz" in log


def test_round2_smoke_uses_subset_diagnostic_splits_and_nonformal_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke 仍驗 source pins，但只跑 subset、diagnostic splits 與縮小後設定。"""
    calls: list[str] = []
    captures = _patch_round2_stages(monkeypatch, tmp_path, calls)
    bundle = captures["bundle"]
    result = captures["result"]

    def extract(
        evidence: object,
        mask_qc: pd.DataFrame,
        image_keys: list[str],
        **kwargs: object,
    ) -> object:
        calls.append("extract_locked_paper93")
        captures["selected_keys"] = list(image_keys)
        bundle.images = bundle.images.loc[
            bundle.images["image_key"].isin(image_keys)
        ].copy()
        bundle.paper_cells = bundle.paper_cells.loc[
            bundle.paper_cells["image_key"].isin(image_keys)
        ].copy()
        return bundle

    def smoke_splits(images: pd.DataFrame, config: object) -> tuple[object, ...]:
        calls.append("make_smoke_outer_splits")
        return (object(),)

    def benchmark(
        images: pd.DataFrame,
        splits: object,
        config: object,
    ) -> object:
        calls.append("run_paper93_benchmark")
        captures["benchmark_config"] = dict(config)
        return result

    monkeypatch.setattr(run_module, "extract_locked_paper93", extract)
    monkeypatch.setattr(run_module, "make_smoke_outer_splits", smoke_splits)
    monkeypatch.setattr(run_module, "run_paper93_benchmark", benchmark)

    record = run_module.run_round2(
        _synthetic_config(tmp_path), smoke_fovs_per_condition=2
    )

    assert record.parent == _formal_output(tmp_path).resolve() / "smoke"
    assert captures["selected_keys"] == ["B1_C1_F01", "B1_C1_F02"]
    assert "require_formal_round1_evidence" in calls
    assert "make_smoke_outer_splits" in calls
    assert "restore_frozen_outer_splits" not in calls
    assert "build_round2_comparison" not in calls
    assert "rank_round2_configurations" not in calls
    assert calls.count("run_paper93_benchmark") == 1
    assert captures["benchmark_config"] == {
        "seed": 20260804,
        "max_hyperparameter_candidates": 1,
        "permutation_repeats": 2,
        "tree_estimators": 10,
    }
    metadata = captures["payloads"]["run_metadata.json"]
    assert metadata["diagnostic_splits"] is True
    assert metadata["non_formal"] is True
    assert metadata["no_scientific_conclusion"] is True
    assert metadata["original_config"]["smoke"]["fovs_per_condition"] == 1
    assert metadata["effective_config"]["smoke"]["fovs_per_condition"] == 2
    eligibility = captures["tables"]["eligibility.csv"]
    assert not eligibility["eligible"].any()
    assert not eligibility["recommended"].any()
    assert set(eligibility["status"]) == {"smoke"}


def test_round2_smoke_emits_parseable_empty_diagnostic_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke 的空 Dummy／comparison artifacts 仍須保留 header 供 validator 讀取。"""
    calls: list[str] = []
    captures = _patch_round2_stages(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        run_module,
        "make_smoke_outer_splits",
        lambda images, config: (object(),),
    )

    run_module.run_round2(
        _synthetic_config(tmp_path), smoke_fovs_per_condition=1
    )

    tables = captures["tables"]
    for name in (
        "dummy_fold_metrics.csv",
        "dummy_oof_predictions.csv",
        "feature_set_comparison.csv",
    ):
        payload = tables[name].to_csv(index=False, lineterminator="\n")
        assert payload.strip(), name
        pd.read_csv(StringIO(payload))


def test_round2_metadata_and_record_context_include_computed_pair_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner 必須由 pair rows 計算 metadata 與 record 共用的 mapping summary。"""
    calls: list[str] = []
    captures = _patch_round2_stages(monkeypatch, tmp_path, calls)

    run_module.run_round2(_synthetic_config(tmp_path))

    expected = {
        "aggregation_unit": "frozen_nucleus_cell_pair",
        "image_count": 3,
        "pair_observation_count": 4,
        "unique_cell_count": 3,
        "unique_nucleus_count": 4,
        "multi_nucleus_cell_count": 1,
        "multi_nucleus_pair_count": 2,
        "max_nuclei_per_cell": 2,
        "retained_outside_pair_count": 1,
        "max_retained_outside_fraction": 0.05,
        "max_nucleus_outside_fraction": 0.05,
    }
    metadata = captures["payloads"]["run_metadata.json"]
    assert metadata["aggregation_unit"] == "frozen_nucleus_cell_pair"
    assert metadata["valid_pair_observations"] == 4
    assert metadata["pair_mapping_summary"] == expected
    assert captures["context"]["valid_pair_observations"] == 4
    assert captures["context"]["pair_mapping_summary"] == expected


def test_round2_smoke_rejects_full_roster_before_downstream_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke count 若選滿 693-row roster，必須在 mask validation 前拒絕。"""
    calls: list[str] = []
    captures = _patch_round2_stages(monkeypatch, tmp_path, calls)
    evidence = captures["evidence"]
    evidence.basic_images = pd.DataFrame(
        {
            "image_key": [f"image-{index:03d}" for index in range(693)],
            "group_id": [f"group-{index // 99}" for index in range(693)],
            "condition_index": [0] * 693,
            "fov": list(range(693)),
        }
    )
    validate = Mock(side_effect=AssertionError("mask validation must not run"))
    extract = Mock(side_effect=AssertionError("feature extraction must not run"))
    benchmark = Mock(side_effect=AssertionError("benchmark must not run"))
    monkeypatch.setattr(run_module, "validate_frozen_masks", validate)
    monkeypatch.setattr(run_module, "extract_locked_paper93", extract)
    monkeypatch.setattr(run_module, "run_paper93_benchmark", benchmark)

    with pytest.raises(ValueError, match="smoke.*subset|strict.*subset|完整"):
        run_module.run_round2(
            _synthetic_config(tmp_path), smoke_fovs_per_condition=100
        )

    validate.assert_not_called()
    extract.assert_not_called()
    benchmark.assert_not_called()


@pytest.mark.parametrize("count", [0, -1])
def test_round2_parser_and_runner_reject_nonpositive_smoke_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    """CLI parser 與 direct runner 都必須在建立 generation 前拒絕非正數。"""
    begin = Mock(side_effect=AssertionError("generation must not begin"))
    monkeypatch.setattr(run_module, "begin_round2_generation", begin)

    with pytest.raises(SystemExit):
        run_module.build_parser().parse_args(
            [
                "--config",
                "round2.yaml",
                "--smoke-fovs-per-condition",
                str(count),
            ]
        )
    with pytest.raises(ValueError, match="正整數"):
        run_module.run_round2(
            _synthetic_config(tmp_path), smoke_fovs_per_condition=count
        )

    begin.assert_not_called()


def test_round2_bundle_validation_failure_restores_qc_before_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic writer 後的 validation 失敗仍須在 failed generation 保留 QC。"""
    calls: list[str] = []
    _patch_round2_stages(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        run_module,
        "validate_round2_bundle",
        Mock(side_effect=ValueError("synthetic bundle validation failure")),
    )

    with pytest.raises(ValueError, match="bundle validation"):
        run_module.run_round2(_synthetic_config(tmp_path))

    assert "publish_round2_generation" not in calls
    failed = list(
        (_formal_output(tmp_path).resolve() / "_generations").glob("failed-*")
    )
    assert len(failed) == 1
    assert (failed[0] / "mask_provenance_qc.csv").is_file()
    assert (failed[0] / "feature_valid_counts.csv").is_file()
    assert (failed[0] / "extraction_qc.csv").is_file()
    assert (failed[0] / "feature_qc.csv").is_file()


def test_round2_quarantine_failure_preserves_original_pipeline_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 3.10 下 quarantine secondary failure 不得取代原始錯誤。"""
    calls: list[str] = []
    _patch_round2_stages(monkeypatch, tmp_path, calls)
    original = ValueError("B8_P7_C08_F03 original preflight failure")
    monkeypatch.setattr(
        run_module,
        "require_paper93_preflight",
        Mock(side_effect=original),
    )
    monkeypatch.setattr(
        run_module,
        "quarantine_round2_generation",
        Mock(side_effect=OSError("secondary quarantine failure")),
    )

    with pytest.raises(ValueError) as caught:
        run_module.run_round2(_synthetic_config(tmp_path))

    assert caught.value is original
    staging = next(
        (_formal_output(tmp_path).resolve() / "_generations").glob("staging-*")
    )
    log = (staging / "run.log").read_text(encoding="utf-8")
    assert "original preflight failure" in log
    assert "secondary quarantine failure" in log


def test_round2_restore_failure_still_appends_original_and_attempts_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QC restore secondary failure 不得阻止 traceback append 或 closed-log move。"""
    calls: list[str] = []
    _patch_round2_stages(monkeypatch, tmp_path, calls)
    original = ValueError("B8_P7_C08_F03 original pipeline failure")
    monkeypatch.setattr(
        run_module,
        "require_paper93_preflight",
        Mock(side_effect=original),
    )
    monkeypatch.setattr(
        run_module,
        "_restore_failure_qc",
        Mock(side_effect=OSError("secondary restore failure")),
    )
    moved: list[Path] = []

    def quarantine(generation: object) -> Path:
        log = generation.staging_dir / "run.log"
        probe = generation.staging_dir / "run-log-closed.probe"
        os.replace(log, probe)
        os.replace(probe, log)
        moved.append(log)
        return generation.staging_dir

    monkeypatch.setattr(run_module, "quarantine_round2_generation", quarantine)

    with pytest.raises(ValueError) as caught:
        run_module.run_round2(_synthetic_config(tmp_path))

    assert caught.value is original
    assert len(moved) == 1
    log = moved[0].read_text(encoding="utf-8")
    assert "original pipeline failure" in log
    assert "secondary restore failure" in log


def test_restore_failure_qc_attempts_remaining_tables_after_one_write_fails(
    tmp_path: Path,
) -> None:
    """單一 QC 寫入失敗後仍須 best-effort 還原其他可用 tables。"""

    class FailingCsvFrame(pd.DataFrame):
        """模擬單一 failure-evidence CSV 無法寫入。"""

        def to_csv(self, *args: object, **kwargs: object) -> None:
            raise OSError("synthetic mask QC restore failure")

    staging = tmp_path / "staging"
    staging.mkdir()
    bundle = SimpleNamespace(
        valid_counts=pd.DataFrame({"image_key": ["image-001"]}),
        extraction_qc=pd.DataFrame({"image_key": ["image-001"]}),
        feature_qc=pd.DataFrame({"feature": ["f1"]}),
    )

    with pytest.raises(OSError, match="mask QC restore failure"):
        run_module._restore_failure_qc(
            staging,
            FailingCsvFrame({"image_key": ["image-001"]}),
            bundle,
        )

    assert (staging / "feature_valid_counts.csv").is_file()
    assert (staging / "extraction_qc.csv").is_file()
    assert (staging / "feature_qc.csv").is_file()


def test_round2_append_failure_still_attempts_quarantine_and_reraises_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traceback append secondary failure 不得阻止 quarantine 或取代原始錯誤。"""
    calls: list[str] = []
    _patch_round2_stages(monkeypatch, tmp_path, calls)
    original = ValueError("original benchmark failure")
    monkeypatch.setattr(
        run_module,
        "require_paper93_preflight",
        Mock(side_effect=original),
    )
    monkeypatch.setattr(
        run_module,
        "_append_failure_evidence",
        Mock(side_effect=OSError("secondary append failure")),
    )
    quarantined: list[Path] = []

    def quarantine(generation: object) -> Path:
        quarantined.append(generation.staging_dir)
        return generation.staging_dir

    monkeypatch.setattr(run_module, "quarantine_round2_generation", quarantine)

    with pytest.raises(ValueError) as caught:
        run_module.run_round2(_synthetic_config(tmp_path))

    assert caught.value is original
    assert len(quarantined) == 1
    log = (quarantined[0] / "run.log").read_text(encoding="utf-8")
    assert "secondary append failure" in log


@pytest.mark.parametrize(("published", "expected"), [(True, 0), (False, 1)])
def test_round2_cli_returns_zero_only_for_existing_published_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published: bool,
    expected: int,
) -> None:
    """CLI 只有在 runner 回傳已存在的 published record 時才可成功。"""
    config = _synthetic_config(tmp_path)
    record = _formal_output(tmp_path) / "EXPERIMENT_RECORD.md"
    if published:
        record.parent.mkdir(parents=True)
        record.write_text("published\n", encoding="utf-8")
    run = Mock(return_value=record)
    monkeypatch.setattr(run_module, "load_round2_config", Mock(return_value=config))
    monkeypatch.setattr(run_module, "run_round2", run)
    monkeypatch.setattr(
        "sys.argv",
        ["run_round2_paper93.py", "--config", "round2.yaml"],
    )

    assert run_module.main() == expected
    run.assert_called_once_with(config, smoke_fovs_per_condition=None)


def test_round2_module_entrypoint_exposes_cli_help() -> None:
    """以 ``python -m`` 執行時必須真的進入 Round 2 CLI parser。"""
    result = subprocess.run(
        [sys.executable, "-m", "immunity.exp3.run_round2_paper93", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--smoke-fovs-per-condition" in result.stdout


def test_round2_config_locks_exact_models_features_seed_and_frozen_hashes() -> None:
    """設定不得偏離 Round 2 的模型、特徵、種子與凍結證據。"""
    config = load_round2_config("immunity/configs/exp3_round2_paper93.yaml")

    assert config["models"] == ["extra_trees", "random_forest"]
    assert config["features"] == {
        "feature_set": "paper_style_median",
        "predictor_count": 93,
        "extra_count": 60,
        "min_finite_cells_per_feature": 3,
        "numeric_atol": 1e-12,
    }
    assert config["benchmark"]["seed"] == 20260804
    assert config["round1"]["expected"]["analyzed_images"] == 693
    assert config["round1"]["expected"]["valid_cells"] == 23976
    assert config["round1"]["roster_sha256"] == (
        "A8333F12E1591D9E4C4174F5C6FE13DD31550124B19AEA3522F4D0F812E0E426"
    )
    assert len(config["round1"]["artifact_sha256"]) == 14


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest_hash",
        "config_hash",
        "roster_sha256",
        "artifact_name",
        "artifact_hash",
    ],
)
def test_round2_config_rejects_mutated_frozen_round1_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    """即使格式正確，Round 1 的凍結 identity 不可遭竄改。"""
    source = Path("immunity/configs/exp3_round2_paper93.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if mutation == "manifest_hash":
        config["round1"]["expected"]["manifest_hash"] = "0" * 64
    elif mutation == "config_hash":
        config["round1"]["expected"]["config_hash"] = "0" * 64
    elif mutation == "roster_sha256":
        config["round1"]["roster_sha256"] = "0" * 64
    elif mutation == "artifact_name":
        artifacts = config["round1"]["artifact_sha256"]
        artifacts["unexpected.csv"] = artifacts.pop("pairing_qc.csv")
    else:
        config["round1"]["artifact_sha256"]["pairing_qc.csv"] = "0" * 64
    path = tmp_path / "mutated-round2.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Round 2"):
        load_round2_config(path)


def test_round2_output_resolver_accepts_only_formal_root_and_smoke_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式根目錄與其 smoke 子目錄是唯一允許的輸出位置。"""
    root = (tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93").resolve()
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", root)

    assert resolve_round2_output_dir(root, smoke=False) == root
    assert resolve_round2_output_dir(root, smoke=True) == root / "smoke"


@pytest.mark.parametrize(
    "relative",
    ["immunity/outputs/exp3", "immunity/outputs/exp3/other", "legacy/results"],
)
def test_round2_output_resolver_rejects_parent_and_legacy_paths(
    relative: str,
) -> None:
    """父目錄、其他實驗與舊輸出路徑均不可作為 Round 2 輸出。"""
    with pytest.raises(ValueError, match="round2_paper93"):
        resolve_round2_output_dir(relative, smoke=False)


def test_round2_output_resolver_rejects_link_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式根目錄本身若為 link/junction，必須拒絕路徑逃逸。"""
    root = tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.parent.mkdir(parents=True)
    _create_directory_link(root, outside)
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", root)

    with pytest.raises(ValueError, match="symlink|junction"):
        resolve_round2_output_dir(root, smoke=False)


def test_round2_output_resolver_rejects_existing_smoke_child_link_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既有 smoke 子目錄若為 link/junction，必須拒絕路徑逃逸。"""
    root = (tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93").resolve()
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    _create_directory_link(root / "smoke", outside)
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", root)

    with pytest.raises(ValueError, match="symlink|junction"):
        resolve_round2_output_dir(root, smoke=True)
