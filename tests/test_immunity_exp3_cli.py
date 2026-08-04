"""驗證 Exp3 CLI、完整 orchestration 與既有主流程的隔離邊界。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import main
from immunity.build_dataset import morphology_feature_columns
import immunity.exp3.run_benchmark as run_module
from immunity.exp3.run_benchmark import load_config, resolve_exp3_output_dir


SYNTHETIC_MAPPING = {
    1: {"condition": "IFN0_TNF0", "ifn_dose": 0.0, "tnf_dose": 0.0},
    2: {"condition": "IFN25_TNF0", "ifn_dose": 25.0, "tnf_dose": 0.0},
    3: {"condition": "IFN50_TNF0", "ifn_dose": 50.0, "tnf_dose": 0.0},
    4: {"condition": "IFN100_TNF0", "ifn_dose": 100.0, "tnf_dose": 0.0},
    5: {"condition": "IFN0_TNF25", "ifn_dose": 0.0, "tnf_dose": 25.0},
    6: {"condition": "IFN0_TNF50", "ifn_dose": 0.0, "tnf_dose": 50.0},
    7: {"condition": "IFN25_TNF25", "ifn_dose": 25.0, "tnf_dose": 25.0},
    8: {"condition": "IFN25_TNF50", "ifn_dose": 25.0, "tnf_dose": 50.0},
}

REQUIRED_TABLE_NAMES = {
    "data_manifest.csv",
    "pairing_qc.csv",
    "segmentation_qc.csv",
    "outer_splits.csv",
    "oof_predictions.csv",
    "fold_metrics.csv",
    "hyperparameters.csv",
    "model_ranking.csv",
    "feature_importance.csv",
    "model_failures.csv",
    "condition_adjusted_metrics.csv",
    "pc_nucleus_dapi_validation.csv",
    "morphology_delta_signatures.csv",
}

REQUIRED_FIGURE_NAMES = {
    "model_validation_rank_heatmap.png",
    "fold_mae_distributions.png",
    "observed_vs_predicted.png",
    "diagnostic_model_comparison.png",
    "feature_importance_stability.png",
    "residuals_by_b_passage_condition.png",
    "morphology_delta_heatmap.png",
    "morphology_delta_pca.png",
}


class SyntheticSegmenter:
    """只以檔名條件建立 deterministic synthetic masks。"""

    def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """回傳一組完全位於 whole-cell 內的 labels。"""
        condition = int(path.stem.split("-", maxsplit=1)[0])
        edge = 9 + condition // 2
        cell = np.zeros((16, 16), dtype=np.int32)
        nucleus = np.zeros((16, 16), dtype=np.int32)
        cell[2:edge, 2:edge] = 1
        nucleus_edge = 4 + max(2, condition // 2)
        nucleus[4:nucleus_edge, 4:nucleus_edge] = 1
        return cell, nucleus


def _tiny_benchmark_config() -> dict[str, object]:
    """建立保留正式模型 contract 的快速 benchmark config。"""
    return {
        "seed": 42,
        "inner_splits": 2,
        "max_hyperparameter_candidates": 1,
        "n_jobs": 1,
        "permutation_repeats": 2,
        "tree_estimators": 10,
        "simplicity_order": [
            "paper_linear_3f",
            "ridge",
            "elasticnet",
            "rbf_svr",
            "hist_gradient_boosting",
            "random_forest",
            "extra_trees",
        ],
    }


def _write_synthetic_pair(
    root: Path,
    condition: int,
    fov: int,
    ido_value: int,
) -> None:
    """寫出一組可由 Exp3 parser 配對的小型 PC／IDO 影像。"""
    (root / "PC").mkdir(parents=True, exist_ok=True)
    (root / "IDO").mkdir(parents=True, exist_ok=True)
    phase = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    ido = np.full((16, 16), 5, dtype=np.uint8)
    edge = 9 + condition // 2
    ido[2:edge, 2:edge] = ido_value
    Image.fromarray(phase).save(root / "PC" / f"{condition}-phase-100X-{fov}.png")
    Image.fromarray(ido).save(root / "IDO" / f"{condition}-IDO-100X-{fov}.png")


def _base_test_config(output_dir: Path) -> dict[str, object]:
    """建立 API synthetic run 所需的完整設定。"""
    return {
        "condition_mapping": SYNTHETIC_MAPPING,
        "segmentation": {
            "device": "cpu",
            "force": True,
            "min_cells_per_image": 1,
        },
        "development_validation": {"enabled": False},
        "feature_sets": {
            "enabled": ["basic_median", "basic_median_iqr"],
            "paper_style_max_features": 93,
        },
        "benchmark": _tiny_benchmark_config(),
        "output": {"dir": str(output_dir)},
        "_output_dir": str(output_dir),
        "_config_path": "synthetic-config-does-not-exist.yaml",
    }


def _make_synthetic_config(
    tmp_path: Path,
    *,
    fovs_per_condition: int,
) -> tuple[dict[str, object], SyntheticSegmenter]:
    """建立 3 B-ID × 3 passage × 8 condition 的 synthetic datasets。"""
    specs = []
    for b_index, b_id in enumerate(("B4", "B7", "B8")):
        for passage in (5, 6, 7):
            root = tmp_path / "data" / f"{b_id}-P{passage}"
            specs.append(
                {"input_dir": str(root), "b_id": b_id, "passage": passage}
            )
            for condition in range(1, 9):
                for fov in range(1, fovs_per_condition + 1):
                    _write_synthetic_pair(
                        root,
                        condition,
                        fov,
                        15 + passage + b_index * 3 + fov,
                    )
    raw_count = 72 * fovs_per_condition
    config = _base_test_config(tmp_path / "output")
    config.update(
        {
            "datasets": specs,
            "expected_totals": {
                "pc": raw_count,
                "ido": raw_count,
                "paired": raw_count,
            },
        }
    )
    return config, SyntheticSegmenter()


def test_exp3_output_must_stay_under_dedicated_root(tmp_path: Path) -> None:
    """Exp3 的輸出目錄只能位於專用根目錄中。"""
    allowed = resolve_exp3_output_dir("immunity/outputs/exp3/smoke")

    assert allowed.as_posix().endswith("immunity/outputs/exp3/smoke")
    with pytest.raises(ValueError, match="immunity/outputs/exp3"):
        resolve_exp3_output_dir("immunity/outputs/b4_p6")
    with pytest.raises(ValueError, match="immunity/outputs/exp3"):
        resolve_exp3_output_dir(tmp_path)


def test_importing_exp3_does_not_expand_main_contract() -> None:
    """Exp3 匯入不可改變主流程 CLI 或 predictor schema。"""
    destinations = {action.dest for action in main.build_parser()._actions}

    assert destinations == {
        "help",
        "data_folder",
        "device",
        "nuc_source",
        "fluor_analy",
        "ki67",
        "ki67_backend",
        "feature_backend",
        "clean_temp",
        "xlsx_version",
    }
    predictors = morphology_feature_columns()
    assert len(predictors) == 33
    assert not any("Zernike" in name or "__median" in name for name in predictors)


def test_exp3_config_loads_without_running_pipeline() -> None:
    """Exp3 設定檔可在不執行 pipeline 的前提下讀取。"""
    config = load_config("immunity/configs/exp3.yaml")

    assert len(config["datasets"]) == 9
    assert config["expected_totals"] == {"pc": 720, "ido": 719, "paired": 719}
    assert config["condition_mapping"] == {}
    assert config["feature_sets"]["enabled"] == ["basic_median"]


def test_empty_mapping_writes_pairing_qc_then_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapping gate 必須晚於 pairing QC、早於 segmentation 與模型。"""
    output = (tmp_path / "output").resolve()
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", output)
    root = tmp_path / "scan" / "B4-P5"
    _write_synthetic_pair(root, 1, 1, 20)
    config = _base_test_config(output)
    config.update(
        {
            "datasets": [
                {"input_dir": str(root), "b_id": "B4", "passage": 5}
            ],
            "expected_totals": {"pc": 1, "ido": 1, "paired": 1},
            "condition_mapping": {},
        }
    )

    with pytest.raises(ValueError, match="condition mapping"):
        run_module.run_benchmark(config)

    pairing_path = output / "pairing_qc.csv"
    assert pairing_path.is_file()
    pairing = pd.read_csv(pairing_path)
    assert pairing["status"].tolist() == ["paired"]
    assert not (output / "fold_metrics.csv").exists()
    assert not (output / "models").exists()
    assert not (output / "EXPERIMENT_RECORD.md").exists()
    assert not list((output / "figures").glob("*.png"))


def test_smoke_pipeline_writes_recomputable_isolated_outputs_without_cellpose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke E2E 只分析 72 images，並保留完整 artifact/evidence contract。"""
    output = (tmp_path / "output").resolve()
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", output)
    config, segmenter = _make_synthetic_config(
        tmp_path,
        fovs_per_condition=2,
    )

    def forbid_real_cellpose(config: object) -> object:
        """若 pipeline 未注入 synthetic Segmenter 就立即失敗。"""
        raise AssertionError("real Cellpose must not run in synthetic E2E")

    monkeypatch.setattr(
        "immunity.exp3.phase_features.PhaseSegmenter",
        forbid_real_cellpose,
    )
    config["_segmenter"] = segmenter

    record = run_module.run_benchmark(config, smoke_fovs_per_condition=1)

    smoke = output / "smoke"
    assert record == smoke / "EXPERIMENT_RECORD.md"
    assert record.is_file() and record.stat().st_size > 0
    assert {path.name for path in smoke.glob("*.csv")} == REQUIRED_TABLE_NAMES
    assert {path.name for path in smoke.glob("*.json")} == {
        "feature_sets.json",
        "run_metadata.json",
    }
    assert {path.name for path in (smoke / "figures").glob("*.png")} == (
        REQUIRED_FIGURE_NAMES
    )
    assert all(path.stat().st_size > 0 for path in (smoke / "figures").glob("*.png"))
    assert not (output / "fold_metrics.csv").exists()
    assert not (output / "model_ranking.csv").exists()
    assert not (output / "EXPERIMENT_RECORD.md").exists()

    manifest_path = smoke / "data_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 72
    assert set(manifest["fov"]) == {1}
    assert manifest.groupby(["group_id", "condition_index"]).size().eq(1).all()

    metrics = pd.read_csv(smoke / "fold_metrics.csv")
    ranking = pd.read_csv(smoke / "model_ranking.csv")
    primary_metrics = metrics[metrics["round"].eq("primary_round_1")]
    assert set(primary_metrics["validation"]) == {
        "leave_one_b_out",
        "leave_one_passage_out",
        "leave_one_group_out",
        "leave_one_condition_out",
    }
    assert ranking["overall_rank"].notna().all()
    assert ranking["complete_outer_folds_gate"].all()
    assert ranking["phase_only_feature_gate"].all()
    recomputed = ranking[
        [
            "leave_one_b_out_rank",
            "leave_one_passage_out_rank",
            "leave_one_group_out_rank",
            "leave_one_condition_out_rank",
        ]
    ].mean(axis=1)
    np.testing.assert_allclose(ranking["overall_rank"], recomputed)
    exploratory = metrics[metrics["round"].eq("exploratory_round_2")]
    assert not exploratory.empty
    assert set(exploratory["feature_set"]) == {"basic_median_iqr"}
    assert set(exploratory["role"]) == {"candidate"}
    assert not exploratory["model"].isin(
        ["dummy_median", "dose_ridge", "dose_plus_morphology_ridge"]
    ).any()

    metadata_text = (smoke / "run_metadata.json").read_text("utf-8")
    assert "NaN" not in metadata_text and "Infinity" not in metadata_text
    metadata = json.loads(metadata_text)
    required_metadata = {
        "started_at_utc",
        "ended_at_utc",
        "elapsed_seconds",
        "git_commit",
        "dirty",
        "config_hash",
        "config_hash_source",
        "manifest_hash",
        "python_version",
        "platform",
        "package_versions",
        "seeds",
        "enabled_feature_sets",
        "smoke_fovs_per_condition",
        "input_counts",
        "cache_hits",
        "failure_count",
    }
    assert required_metadata <= set(metadata)
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["config_hash"])
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["manifest_hash"])
    assert metadata["manifest_hash"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert metadata["config_hash_source"] == "canonical_in_memory_config"
    assert metadata["enabled_feature_sets"] == [
        "basic_median",
        "basic_median_iqr",
    ]
    assert metadata["smoke_fovs_per_condition"] == 1
    assert metadata["input_counts"] == {
        "raw_pc": 144,
        "raw_ido": 144,
        "complete_pairs": 144,
        "analyzed_images": 72,
    }
    assert set(metadata["package_versions"]) >= {
        "numpy",
        "pandas",
        "scikit-learn",
        "cellpose",
    }


def test_model_failure_writes_failure_evidence_then_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """實際 orchestration 偵測 fold failure 時不可發布成功 record。"""
    from immunity.exp3 import benchmark as benchmark_module

    output = (tmp_path / "output").resolve()
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", output)
    config, segmenter = _make_synthetic_config(tmp_path, fovs_per_condition=1)
    config["feature_sets"]["enabled"] = ["basic_median"]
    config["_segmenter"] = segmenter

    failure = pd.DataFrame(
        [
            {
                "validation": "leave_one_b_out",
                "fold": "B4",
                "split_id": "leave_one_b_out:B4",
                "model": "ridge",
                "exception_type": "RuntimeError",
                "message": "synthetic fold failure",
            }
        ],
        columns=benchmark_module.FAILURE_COLUMNS,
    )
    failed_result = benchmark_module.BenchmarkResult(
        predictions=pd.DataFrame(columns=benchmark_module.OOF_COLUMNS),
        fold_metrics=pd.DataFrame(columns=benchmark_module.FOLD_METRIC_COLUMNS),
        hyperparameters=pd.DataFrame(columns=benchmark_module.HYPERPARAMETER_COLUMNS),
        feature_importance=pd.DataFrame(
            columns=benchmark_module.FEATURE_IMPORTANCE_COLUMNS
        ),
        failures=failure,
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_nested_benchmark",
        lambda *args, **kwargs: failed_result,
    )

    with pytest.raises(RuntimeError, match="model.*failure"):
        run_module.run_benchmark(config)

    failures = pd.read_csv(output / "model_failures.csv")
    assert failures["message"].tolist() == ["synthetic fold failure"]
    assert not (output / "EXPERIMENT_RECORD.md").exists()


@pytest.mark.parametrize(
    ("message", "exception_type"),
    [
        ("condition mapping invalid", ValueError),
        ("pairing mismatch", ValueError),
        ("segmentation failed", RuntimeError),
        ("model failed", RuntimeError),
    ],
)
def test_cli_returns_nonzero_and_keeps_run_log_for_pipeline_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    exception_type: type[Exception],
) -> None:
    """Pipeline 任一階段失敗都必須保留正確 run-local log 並回傳非零。"""
    output = (tmp_path / "output").resolve()
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", output)
    config = _base_test_config(output)
    monkeypatch.setattr(run_module, "load_config", lambda path: config)

    def fail_pipeline(config: object, smoke_fovs_per_condition: object = None) -> Path:
        raise exception_type(message)

    monkeypatch.setattr(run_module, "run_benchmark", fail_pipeline, raising=False)
    monkeypatch.setattr(sys, "argv", ["run_benchmark", "--config", "synthetic.yaml"])

    assert run_module.main() != 0
    log_text = (output / "run.log").read_text("utf-8")
    assert message in log_text
    assert not (output / "EXPERIMENT_RECORD.md").exists()


def test_cli_returns_zero_only_after_smoke_record_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI success 必須同時具備 smoke record 與同目錄 run.log。"""
    output = (tmp_path / "output").resolve()
    smoke = output / "smoke"
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", output)
    config = _base_test_config(output)
    monkeypatch.setattr(run_module, "load_config", lambda path: config)

    def write_record(config: object, smoke_fovs_per_condition: object = None) -> Path:
        smoke.mkdir(parents=True, exist_ok=True)
        record = smoke / "EXPERIMENT_RECORD.md"
        record.write_text("synthetic record\n", encoding="utf-8")
        print("synthetic success marker")
        return record

    monkeypatch.setattr(run_module, "run_benchmark", write_record, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark",
            "--config",
            "synthetic.yaml",
            "--smoke-fovs-per-condition",
            "1",
        ],
    )

    assert run_module.main() == 0
    assert (smoke / "EXPERIMENT_RECORD.md").is_file()
    assert "synthetic success marker" in (smoke / "run.log").read_text("utf-8")
    assert not (output / "run.log").exists()


def test_cli_rejects_missing_record_even_when_pipeline_returns_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline 未發布 record 時不可回傳成功。"""
    output = (tmp_path / "output").resolve()
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", output)
    config = _base_test_config(output)
    monkeypatch.setattr(run_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        run_module,
        "run_benchmark",
        lambda config, smoke_fovs_per_condition=None: output / "missing.md",
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["run_benchmark", "--config", "synthetic.yaml"])

    assert run_module.main() != 0
    assert "EXPERIMENT_RECORD" in (output / "run.log").read_text("utf-8")


def test_round_two_top_models_use_task8_tie_break_evidence() -> None:
    """Equal overall rank 不得退回 model-name alphabetic ordering。"""
    ranking = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "overall_rank": 2.0,
                "worst_validation_rank": 4.0,
                "leave_one_b_out_mae": 1.0,
                "overall_oof_spearman": 0.8,
                "simplicity_rank": 6,
            },
            {
                "model": "ridge",
                "overall_rank": 2.0,
                "worst_validation_rank": 3.0,
                "leave_one_b_out_mae": 1.5,
                "overall_oof_spearman": 0.2,
                "simplicity_rank": 2,
            },
            {
                "model": "elasticnet",
                "overall_rank": 3.0,
                "worst_validation_rank": 3.0,
                "leave_one_b_out_mae": 1.0,
                "overall_oof_spearman": 0.9,
                "simplicity_rank": 3,
            },
        ]
    )

    selected = run_module._round_two_top_models(
        ranking,
        ["ridge", "elasticnet", "random_forest"],
    )

    assert selected == ["ridge", "random_forest"]


def test_config_load_failure_writes_smoke_fallback_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output 尚無法由 config 解析時，smoke CLI 仍保留 fallback log。"""
    output = (tmp_path / "output").resolve()
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", output)

    def fail_load(path: object) -> dict[str, object]:
        raise ValueError("synthetic invalid config")

    monkeypatch.setattr(run_module, "load_config", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark",
            "--config",
            "invalid.yaml",
            "--smoke-fovs-per-condition",
            "1",
        ],
    )

    assert run_module.main() != 0
    assert "synthetic invalid config" in (
        output / "smoke" / "run.log"
    ).read_text("utf-8")
