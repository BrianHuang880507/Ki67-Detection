"""測試 Exp3 Round 2 原子報告與 bundle 驗證合約。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import immunity.exp3.run_round2_paper93 as run_module
from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES, PRIMARY_FOV_FEATURES
from immunity.exp3.round2_reporting import (
    ROUND2_TABLE_NAMES,
    begin_round2_generation,
    build_round2_experiment_record,
    publish_round2_generation,
    quarantine_round2_generation,
    validate_round2_bundle,
    write_round2_bundle,
)


_VALIDATIONS = (
    ("leave_one_b_out", 3),
    ("leave_one_passage_out", 3),
    ("leave_one_group_out", 9),
    ("leave_one_condition_out", 8),
)
_CONFIGURATIONS = (
    ("extra_trees__basic_median", "extra_trees", "basic_median", "round1"),
    (
        "extra_trees__paper_style_median",
        "extra_trees",
        "paper_style_median",
        "round2_paper93",
    ),
    ("random_forest__basic_median", "random_forest", "basic_median", "round1"),
    (
        "random_forest__paper_style_median",
        "random_forest",
        "paper_style_median",
        "round2_paper93",
    ),
)
_JSON_NAMES = {
    "feature_sets.json",
    "baseline_provenance.json",
    "run_metadata.json",
}


def _allowed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """將測試限定在暫存的正式 Round 2 根目錄。"""
    output = (tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93").resolve()
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", output)
    return output


def _split_contract() -> list[tuple[str, str, str, list[str]]]:
    """建立 23 個 split 與手算 test membership。"""
    image_keys = [f"image-{index:03d}" for index in range(693)]
    rows: list[tuple[str, str, str, list[str]]] = []
    for validation, count in _VALIDATIONS:
        buckets = np.array_split(np.asarray(image_keys, dtype=object), count)
        for fold, bucket in enumerate(buckets, start=1):
            split_id = f"{validation}:{fold}"
            rows.append((validation, str(fold), split_id, bucket.tolist()))
    return rows


def _formal_tables() -> dict[str, pd.DataFrame]:
    """建立符合 693 FOV、23 folds 與完整 OOF 的正式 bundle fixture。"""
    image_keys = [f"image-{index:03d}" for index in range(693)]
    data_snapshot = pd.DataFrame(
        {
            "image_key": image_keys,
            "IDO_score": np.linspace(0.0, 1.0, 693),
            **{
                feature: np.full(693, float(index))
                for index, feature in enumerate(PAPER_STYLE_FOV_FEATURES)
            },
        }
    )
    split_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    dummy_metric_rows: list[dict[str, object]] = []
    dummy_prediction_rows: list[dict[str, object]] = []
    hyperparameter_rows: list[dict[str, object]] = []
    for validation, fold, split_id, test_keys in _split_contract():
        test_set = set(test_keys)
        for image_key in image_keys:
            split_rows.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "image_key": image_key,
                    "role": "test" if image_key in test_set else "train",
                }
            )
        dummy_metric_rows.append(
            {
                "validation": validation,
                "fold": fold,
                "split_id": split_id,
                "model": "dummy_median",
                "feature_set": "none",
                "source_round": "round1",
                "configuration_id": "dummy_median__none",
                "n_test": len(test_keys),
                "status": "ok",
            }
        )
        for image_key in test_keys:
            dummy_prediction_rows.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "split_id": split_id,
                    "model": "dummy_median",
                    "feature_set": "none",
                    "source_round": "round1",
                    "configuration_id": "dummy_median__none",
                    "image_key": image_key,
                    "observed_ido_score": 0.5,
                    "predicted_ido_score": 0.5,
                }
            )
        for configuration_id, model, feature_set, source_round in _CONFIGURATIONS:
            metric_rows.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "split_id": split_id,
                    "model": model,
                    "feature_set": feature_set,
                    "source_round": source_round,
                    "configuration_id": configuration_id,
                    "n_test": len(test_keys),
                    "mae": 0.1,
                    "rmse": 0.2,
                    "r2": 0.3,
                    "spearman": 0.4,
                    "status": "ok",
                }
            )
            hyperparameter_rows.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "split_id": split_id,
                    "model": model,
                    "feature_set": feature_set,
                    "source_round": source_round,
                    "configuration_id": configuration_id,
                    "seed": 20260804,
                    "best_params_json": "{}",
                }
            )
            for image_key in test_keys:
                prediction_rows.append(
                    {
                        "validation": validation,
                        "fold": fold,
                        "split_id": split_id,
                        "model": model,
                        "feature_set": feature_set,
                        "source_round": source_round,
                        "configuration_id": configuration_id,
                        "image_key": image_key,
                        "observed_ido_score": 0.5,
                        "predicted_ido_score": 0.5,
                    }
                )

    valid_counts = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "feature": feature,
                "roster_cell_count": 4,
                "finite_cell_count": 4,
                "required_minimum": 3,
                "status": "passed",
            }
            for image_key in image_keys
            for feature in PAPER_STYLE_FOV_FEATURES[33:]
        ]
    )
    feature_qc = pd.DataFrame(
        {
            "feature": list(PAPER_STYLE_FOV_FEATURES),
            "feature_group": ["basic"] * 33 + ["extra"] * 60,
            "finite_fov_count": [693] * 93,
            "unique_finite_count": [2] * 93,
            "constant": [False] * 93,
            "forbidden_token": [""] * 93,
            "registered": [True] * 93,
            "status": ["passed"] * 93,
        }
    )
    eligibility = pd.DataFrame(
        [
            {
                "configuration_id": configuration_id,
                "model": model,
                "feature_set": feature_set,
                "source_round": source_round,
                "eligible": True,
                "recommended": configuration_id == "extra_trees__basic_median",
                "status": "formal",
            }
            for configuration_id, model, feature_set, source_round in _CONFIGURATIONS
        ]
    )
    comparison = eligibility.loc[
        :, ["configuration_id", "model", "feature_set", "source_round"]
    ].copy()
    comparison["mae"] = [0.1, 0.2, 0.3, 0.4]

    return {
        "data_snapshot.csv": data_snapshot,
        "outer_splits.csv": pd.DataFrame(split_rows),
        "mask_provenance_qc.csv": pd.DataFrame(
            {"image_key": image_keys, "status": ["passed"] * 693}
        ),
        "feature_valid_counts.csv": valid_counts,
        "extraction_qc.csv": pd.DataFrame(
            {"image_key": image_keys, "status": ["passed"] * 693, "reason": [""] * 693}
        ),
        "feature_qc.csv": feature_qc,
        "fold_metrics.csv": pd.DataFrame(metric_rows),
        "oof_predictions.csv": pd.DataFrame(prediction_rows),
        "dummy_fold_metrics.csv": pd.DataFrame(dummy_metric_rows),
        "dummy_oof_predictions.csv": pd.DataFrame(dummy_prediction_rows),
        "hyperparameters.csv": pd.DataFrame(hyperparameter_rows),
        "feature_importance.csv": pd.DataFrame(
            columns=[
                "validation",
                "fold",
                "split_id",
                "model",
                "feature_set",
                "source_round",
                "configuration_id",
                "feature",
                "importance",
            ]
        ),
        "model_failures.csv": pd.DataFrame(
            columns=[
                "validation",
                "fold",
                "split_id",
                "model",
                "feature_set",
                "source_round",
                "configuration_id",
                "exception_type",
                "message",
            ]
        ),
        "feature_set_comparison.csv": comparison,
        "eligibility.csv": eligibility,
    }


def _formal_tables_with_failed_fold() -> dict[str, pd.DataFrame]:
    """建立一個 candidate outer fold 失敗且證據一致的正式 fixture。"""
    tables = _formal_tables()
    configuration = "extra_trees__paper_style_median"
    split_id = "leave_one_b_out:1"
    failed = tables["fold_metrics.csv"]["configuration_id"].eq(configuration) & tables[
        "fold_metrics.csv"
    ]["split_id"].eq(split_id)
    tables["fold_metrics.csv"].loc[failed, "status"] = "failed"
    tables["fold_metrics.csv"].loc[
        failed, ["mae", "rmse", "r2", "spearman"]
    ] = np.nan
    for name in ("oof_predictions.csv", "hyperparameters.csv", "feature_importance.csv"):
        frame = tables[name]
        keep = ~(
            frame["configuration_id"].eq(configuration)
            & frame["split_id"].eq(split_id)
        )
        tables[name] = frame.loc[keep].copy()
    tables["model_failures.csv"] = pd.DataFrame(
        [
            {
                "validation": "leave_one_b_out",
                "fold": "1",
                "split_id": split_id,
                "model": "extra_trees",
                "feature_set": "paper_style_median",
                "source_round": "round2_paper93",
                "configuration_id": configuration,
                "exception_type": "ValueError",
                "message": "synthetic",
            }
        ]
    )
    tables["eligibility.csv"].loc[
        tables["eligibility.csv"]["configuration_id"].eq(configuration),
        ["eligible", "recommended"],
    ] = [False, False]
    return tables


def _formal_json_payloads() -> dict[str, dict[str, Any]]:
    """建立三個固定 JSON artifacts。"""
    return {
        "feature_sets.json": {
            "paper_style_median": list(PAPER_STYLE_FOV_FEATURES),
            "basic_median": list(PAPER_STYLE_FOV_FEATURES[:33]),
        },
        "baseline_provenance.json": {
            "round1_winner": "extra_trees__basic_median",
            "read_only": True,
        },
        "run_metadata.json": {
            "mode": "formal",
            "seed": 20260804,
            "analyzed_images": 693,
            "valid_cells": 23976,
            "exclusions": 26,
        },
    }


def _formal_record_context() -> dict[str, Any]:
    """建立答案先行的正式 record context。"""
    return {
        "smoke": False,
        "recommendation": "extra_trees__basic_median",
        "paper93_better_than_basic": False,
        "analyzed_images": 693,
        "valid_cells": 23976,
        "exclusions": 26,
        "seed": 20260804,
        "runtime_seconds": 12.5,
        "predictor_columns": list(PAPER_STYLE_FOV_FEATURES),
        "comparison": pd.DataFrame(
            {
                "configuration_id": [row[0] for row in _CONFIGURATIONS],
                "mae": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        "eligibility": pd.DataFrame(
            {
                "configuration_id": [row[0] for row in _CONFIGURATIONS],
                "eligible": [True, True, True, True],
                "recommended": [True, False, False, False],
            }
        ),
        "feature_qc": {"constant_features": 0, "failed_features": 0},
        "extraction_failures": 0,
    }


def _write_bundle(
    output: Path,
    tables: dict[str, pd.DataFrame],
    json_payloads: dict[str, dict[str, Any]],
    record_context: dict[str, Any],
) -> Any:
    """建立 generation 並寫入含已關閉 run.log 的 bundle。"""
    generation = begin_round2_generation(output)
    (generation.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    write_round2_bundle(
        generation.staging_dir,
        tables,
        json_payloads,
        record_context,
    )
    return generation


def _rehash(directory: Path, filename: str) -> None:
    """更新單一 artifact 的測試 hash，讓語意驗證能獨立觸發。"""
    manifest_path = directory / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (directory / filename).read_bytes()
    manifest["artifacts"][filename] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_round2_writer_publishes_exact_required_artifacts_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少或多出固定 artifact 時不能被視為已發布 generation。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )

    validate_round2_bundle(generation.staging_dir, smoke=False)
    publish_round2_generation(generation)

    assert set(path.name for path in generation.output_dir.iterdir() if path.is_file()) == {
        *ROUND2_TABLE_NAMES,
        *_JSON_NAMES,
        "artifact_hashes.json",
        "EXPERIMENT_RECORD.md",
        "run.log",
    }


def test_round2_validator_accepts_published_root_with_managed_generations_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published root 只額外允許安全的 `_generations` 管理目錄。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    publish_round2_generation(generation)

    validate_round2_bundle(generation.output_dir, smoke=False)


def test_round2_published_root_rejects_other_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published root 除 `_generations` 外不可有其他 directory。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    publish_round2_generation(generation)
    (generation.output_dir / "other").mkdir()

    with pytest.raises(ValueError, match="artifact keys|extra|other"):
        validate_round2_bundle(generation.output_dir, smoke=False)


def test_round2_published_root_rejects_reparse_generations_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published `_generations` 若被換成 symlink/junction 必須 fail-closed。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    publish_round2_generation(generation)
    managed = generation.output_dir / "_generations"
    managed.rmdir()
    outside = tmp_path / "outside-generations"
    outside.mkdir()
    try:
        managed.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"此環境無法建立 directory symlink：{error}")

    with pytest.raises(ValueError, match="symlink|junction|reparse|逃逸"):
        validate_round2_bundle(generation.output_dir, smoke=False)


def test_round2_validator_rejects_unexpected_directory_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact artifact gate 必須連額外的 ordinary directory 都拒絕。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (generation.staging_dir / "unexpected_dir").mkdir()

    with pytest.raises(ValueError, match="artifact keys|unexpected_dir|extra"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_round2_writer_preflights_all_destinations_before_first_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一 destination 已存在時，writer 不得先發布其他檔案。"""
    generation = begin_round2_generation(_allowed_output(tmp_path, monkeypatch))
    collision = generation.staging_dir / "eligibility.csv"
    collision.write_text("owned by caller\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="eligibility.csv"):
        write_round2_bundle(
            generation.staging_dir,
            _formal_tables(),
            _formal_json_payloads(),
            _formal_record_context(),
        )

    assert {path.name for path in generation.staging_dir.iterdir()} == {"eligibility.csv"}


def test_round2_writer_rolls_back_staged_files_when_later_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中途 replace 失敗時，writer 要移除本次建立的 destination 與 temp。"""
    import immunity.exp3.round2_reporting as reporting

    generation = begin_round2_generation(_allowed_output(tmp_path, monkeypatch))
    real_replace = reporting.os.replace
    calls = 0

    def fail_second_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        write_round2_bundle(
            generation.staging_dir,
            _formal_tables(),
            _formal_json_payloads(),
            _formal_record_context(),
        )

    assert list(generation.staging_dir.iterdir()) == []


def test_round2_publication_archives_prior_round2_generation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """發布新版只能封存目前 target 的舊 Round 2 檔案。"""
    output = _allowed_output(tmp_path, monkeypatch)
    first = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    publish_round2_generation(first)
    old_record = (output / "EXPERIMENT_RECORD.md").read_text(encoding="utf-8")
    second = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )

    publish_round2_generation(second)

    archives = sorted((output / "_generations").glob("archive-*"))
    assert len(archives) == 1
    assert (archives[0] / "EXPERIMENT_RECORD.md").read_text(encoding="utf-8") == old_record


def test_round2_publication_never_moves_or_overwrites_round1_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 2 發布邊界不能觸及同層的 Round 1 artifact。"""
    output = _allowed_output(tmp_path, monkeypatch)
    output.parent.mkdir(parents=True)
    round1 = output.parent / "data_manifest.csv"
    round1.write_bytes(b"round-one-immutable")
    before = hashlib.sha256(round1.read_bytes()).hexdigest()
    generation = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )

    publish_round2_generation(generation)

    assert hashlib.sha256(round1.read_bytes()).hexdigest() == before


def test_round2_failure_quarantines_staging_and_keeps_qc_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失敗 generation 只搬至 target 內的 failed 目錄並保留 run.log。"""
    generation = begin_round2_generation(_allowed_output(tmp_path, monkeypatch))
    (generation.staging_dir / "run.log").write_text("named-image.tif failed\n", encoding="utf-8")
    (generation.staging_dir / "extraction_qc.csv").write_text("status\nfailed\n", encoding="utf-8")

    failed = quarantine_round2_generation(generation)

    assert failed.parent == generation.output_dir / "_generations"
    assert failed.name.startswith("failed-")
    assert "named-image.tif" in (failed / "run.log").read_text(encoding="utf-8")
    assert not generation.staging_dir.exists()


def test_round2_generation_rejects_symlink_or_junction_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既有 output component 是 reparse point 時必須 fail-closed。"""
    expected = _allowed_output(tmp_path, monkeypatch)
    expected.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        expected.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"此環境無法建立 directory symlink：{error}")

    with pytest.raises(ValueError, match="symlink|junction|reparse|逃逸"):
        begin_round2_generation(expected)


def test_round2_validator_requires_formal_counts_and_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 bundle 必須具有 693 rows、93 predictors、23 folds 與 92 metrics。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )

    validate_round2_bundle(generation.staging_dir, smoke=False)

    assert len(pd.read_csv(generation.staging_dir / "fold_metrics.csv")) == 92


def test_round2_validator_requires_11088_oof_when_all_model_folds_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """所有 fold 成功時，少一筆 candidate OOF 即拒絕。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    path = generation.staging_dir / "oof_predictions.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    _rehash(generation.staging_dir, "oof_predictions.csv")

    with pytest.raises(ValueError, match="11,088|11088|OOF"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_round2_validator_rejects_oof_keys_swapped_between_frozen_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OOF 筆數不變但偏離 frozen test membership 時仍須拒絕。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    path = generation.staging_dir / "oof_predictions.csv"
    predictions = pd.read_csv(path)
    configuration = predictions["configuration_id"].eq("extra_trees__basic_median")
    first = configuration & predictions["split_id"].eq("leave_one_b_out:1")
    second = configuration & predictions["split_id"].eq("leave_one_b_out:2")
    first_keys = predictions.loc[first, "image_key"].to_numpy(copy=True)
    second_keys = predictions.loc[second, "image_key"].to_numpy(copy=True)
    predictions.loc[first, "image_key"] = second_keys
    predictions.loc[second, "image_key"] = first_keys
    predictions.to_csv(path, index=False)
    _rehash(generation.staging_dir, "oof_predictions.csv")

    with pytest.raises(ValueError, match="membership|image_key|frozen"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_round2_validator_reconciles_failed_fold_and_ineligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """failed metric 必須一對一對應 failure、缺少證據與 ineligible 設定。"""
    tables = _formal_tables_with_failed_fold()
    configuration = "extra_trees__paper_style_median"
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    validate_round2_bundle(generation.staging_dir, smoke=False)

    tables["eligibility.csv"].loc[
        tables["eligibility.csv"]["configuration_id"].eq(configuration), "eligible"
    ] = True
    bad = _write_bundle(
        _allowed_output(tmp_path / "other", monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )
    with pytest.raises(ValueError, match="ineligible|eligible"):
        validate_round2_bundle(bad.staging_dir, smoke=False)


def test_failed_fold_validator_rejects_oof_identity_outside_success_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed mode 不得接受任何成功 metric identities 以外的 OOF split。"""
    tables = _formal_tables_with_failed_fold()
    rogue = tables["oof_predictions.csv"].iloc[[0]].copy()
    rogue[["validation", "fold", "split_id", "image_key"]] = [
        "leave_one_b_out",
        "999",
        "leave_one_b_out:999",
        "rogue-image",
    ]
    tables["oof_predictions.csv"] = pd.concat(
        [tables["oof_predictions.csv"], rogue], ignore_index=True
    )
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="OOF.*identity|unknown.*split|成功"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_failed_fold_validator_uses_frozen_n_test_for_failed_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed fold 雖無 OOF，metric n_test 仍須等於 frozen test membership。"""
    tables = _formal_tables_with_failed_fold()
    failed = tables["fold_metrics.csv"]["status"].eq("failed")
    tables["fold_metrics.csv"].loc[failed, "n_test"] = 0
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="n_test.*frozen|frozen.*n_test"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


@pytest.mark.parametrize(
    "drift",
    ["unknown_configuration", "missing_success", "failed_identity_row"],
)
def test_failed_fold_validator_requires_exact_successful_oof_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    """Failed mode 的 OOF 只能且必須對應每個 successful metric identity。"""
    tables = _formal_tables_with_failed_fold()
    predictions = tables["oof_predictions.csv"].copy()
    success = (
        predictions["configuration_id"].eq("extra_trees__basic_median")
        & predictions["split_id"].eq("leave_one_b_out:1")
    )
    if drift == "unknown_configuration":
        predictions.loc[predictions.index[0], "configuration_id"] = "rogue__paper93"
    elif drift == "missing_success":
        predictions = predictions.loc[~success].copy()
    else:
        original = _formal_tables()["oof_predictions.csv"]
        failed_rows = original.loc[
            original["configuration_id"].eq("extra_trees__paper_style_median")
            & original["split_id"].eq("leave_one_b_out:1")
        ]
        predictions = pd.concat([predictions, failed_rows], ignore_index=True)
    tables["oof_predictions.csv"] = predictions
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="OOF|configuration identity"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


class _FailingBinaryHandle:
    """在指定 binary file operation 注入單次失敗。"""

    def __init__(self, handle: Any, operation: str) -> None:
        self._handle = handle
        self._operation = operation

    def __enter__(self) -> "_FailingBinaryHandle":
        self._handle.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self._handle.__exit__(*args)

    def write(self, payload: bytes) -> int:
        if self._operation == "write":
            raise OSError("injected write failure")
        return int(self._handle.write(payload))

    def flush(self) -> None:
        if self._operation == "flush":
            raise OSError("injected flush failure")
        self._handle.flush()

    def fileno(self) -> int:
        return int(self._handle.fileno())


@pytest.mark.parametrize("operation", ["write", "flush", "fsync"])
def test_round2_temp_writer_removes_partial_file_after_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """本次建立的 temp 在 write/flush/fsync 失敗後不可殘留。"""
    import immunity.exp3.round2_reporting as reporting

    generation = begin_round2_generation(_allowed_output(tmp_path, monkeypatch))
    if operation == "fsync":
        monkeypatch.setattr(
            reporting.os,
            "fsync",
            lambda _: (_ for _ in ()).throw(OSError("injected fsync failure")),
        )
    else:
        real_open = Path.open

        def failing_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(path, mode, *args, **kwargs)
            if path.name == ".tmp" and mode == "xb":
                return _FailingBinaryHandle(handle, operation)
            return handle

        monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match=f"injected {operation} failure"):
        write_round2_bundle(
            generation.staging_dir,
            _formal_tables(),
            _formal_json_payloads(),
            _formal_record_context(),
        )

    assert not (generation.staging_dir / ".tmp").exists()
    assert list(generation.staging_dir.iterdir()) == []


def test_round2_temp_writer_never_deletes_preexisting_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusive-create collision 不是本次檔案，失敗 cleanup 不得刪除。"""
    generation = begin_round2_generation(_allowed_output(tmp_path, monkeypatch))
    temp = generation.staging_dir / ".tmp"
    temp.write_bytes(b"caller-owned")

    with pytest.raises(FileExistsError):
        write_round2_bundle(
            generation.staging_dir,
            _formal_tables(),
            _formal_json_payloads(),
            _formal_record_context(),
        )

    assert temp.read_bytes() == b"caller-owned"


def test_round2_validator_requires_dummy_and_four_ranking_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 Dummy 需 23 metrics、2,772 OOF，ranking 需四組設定。"""
    tables = _formal_tables()
    tables["dummy_oof_predictions.csv"] = tables["dummy_oof_predictions.csv"].iloc[:-1]
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="2,772|2772|Dummy"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_round2_validator_rejects_unregistered_feature_importance_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importance feature 必須屬於該 configuration 的 authoritative roster。"""
    tables = _formal_tables()
    tables["feature_importance.csv"] = pd.DataFrame(
        [
            {
                "validation": "leave_one_b_out",
                "fold": "1",
                "split_id": "leave_one_b_out:1",
                "model": "extra_trees",
                "feature_set": "basic_median",
                "source_round": "round1",
                "configuration_id": "extra_trees__basic_median",
                "feature": "rogue_feature",
                "importance": 0.1,
            }
        ]
    )
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="feature_importance.*feature|authoritative"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_round2_record_reports_actual_feature_importance_completeness_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importance 缺漏不阻擋 publication，但 record 必須列出實際 completeness。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )

    validate_round2_bundle(generation.staging_dir, smoke=False)
    record = (generation.staging_dir / "EXPERIMENT_RECORD.md").read_text(
        encoding="utf-8"
    )

    assert "Feature importance diagnostic completeness warning" in record
    assert "actual=0" in record
    assert "expected=5796" in record
    assert "missing=5796" in record


@pytest.mark.parametrize(
    "drift",
    [
        "configuration",
        "model",
        "source",
        "failed_fold",
        "duplicate",
        "too_many",
    ],
)
def test_round2_validator_rejects_invalid_feature_importance_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    """Importance 必須屬於成功 fold、identity 精確、feature 唯一且不超過上限。"""
    tables = (
        _formal_tables_with_failed_fold()
        if drift == "failed_fold"
        else _formal_tables()
    )
    row = {
        "validation": "leave_one_b_out",
        "fold": "1",
        "split_id": "leave_one_b_out:1",
        "model": "extra_trees",
        "feature_set": "basic_median",
        "source_round": "round1",
        "configuration_id": "extra_trees__basic_median",
        "feature": PRIMARY_FOV_FEATURES[0],
        "importance": 0.1,
    }
    if drift == "configuration":
        row["configuration_id"] = "rogue__basic_median"
    elif drift == "model":
        row["model"] = "random_forest"
    elif drift == "source":
        row["source_round"] = "round2_paper93"
    elif drift == "failed_fold":
        row.update(
            {
                "feature_set": "paper_style_median",
                "source_round": "round2_paper93",
                "configuration_id": "extra_trees__paper_style_median",
                "feature": PAPER_STYLE_FOV_FEATURES[0],
            }
        )
    if drift == "too_many":
        importance = pd.DataFrame([row] * 5797)
    elif drift == "duplicate":
        importance = pd.DataFrame([row, row])
    else:
        importance = pd.DataFrame([row])
    tables["feature_importance.csv"] = importance
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(
        ValueError,
        match="feature_importance|configuration identity|successful fold|5796|唯一",
    ):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_round2_validator_rejects_forged_dummy_identity_and_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dummy model、feature、source、split 與成功 status 必須精確鎖定。"""
    tables = _formal_tables()
    tables["dummy_fold_metrics.csv"]["model"] = "forged"
    tables["dummy_fold_metrics.csv"]["feature_set"] = "forged"
    tables["dummy_fold_metrics.csv"]["status"] = "failed"
    tables["dummy_oof_predictions.csv"]["model"] = "forged"
    tables["dummy_oof_predictions.csv"]["feature_set"] = "forged"
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="Dummy.*identity|Dummy.*status"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


@pytest.mark.parametrize("drift", ["duplicate_oof", "missing_fold", "hash"])
def test_round2_validator_rejects_duplicate_missing_fold_or_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    """OOF 重複、fold 缺漏與 bytes drift 都不能發布。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    if drift == "duplicate_oof":
        path = generation.staging_dir / "oof_predictions.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
        _rehash(generation.staging_dir, "oof_predictions.csv")
    elif drift == "missing_fold":
        path = generation.staging_dir / "fold_metrics.csv"
        pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
        _rehash(generation.staging_dir, "fold_metrics.csv")
    else:
        with (generation.staging_dir / "feature_qc.csv").open("ab") as handle:
            handle.write(b"\n")

    with pytest.raises(ValueError, match="duplicate|重複|92|fold|SHA-256|hash"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_smoke_validator_accepts_subset_schema_but_requires_no_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke 只驗證 93 schema 與兩模型流程，且禁止 recommendation。"""
    tables = _formal_tables()
    keep_keys = {f"image-{index:03d}" for index in range(8)}
    tables["data_snapshot.csv"] = tables["data_snapshot.csv"].loc[
        tables["data_snapshot.csv"]["image_key"].isin(keep_keys)
    ]
    for name in (
        "outer_splits.csv",
        "mask_provenance_qc.csv",
        "feature_valid_counts.csv",
        "extraction_qc.csv",
    ):
        tables[name] = tables[name].loc[tables[name]["image_key"].isin(keep_keys)]
    for name in ("fold_metrics.csv", "oof_predictions.csv", "hyperparameters.csv"):
        tables[name] = tables[name].loc[
            tables[name]["feature_set"].eq("paper_style_median")
            & tables[name]["model"].isin(("extra_trees", "random_forest"))
        ].head(16)
    tables["dummy_fold_metrics.csv"] = tables["dummy_fold_metrics.csv"].iloc[0:0]
    tables["dummy_oof_predictions.csv"] = tables["dummy_oof_predictions.csv"].iloc[0:0]
    tables["feature_set_comparison.csv"] = tables["feature_set_comparison.csv"].loc[
        tables["feature_set_comparison.csv"]["feature_set"].eq("paper_style_median")
    ]
    tables["eligibility.csv"] = tables["eligibility.csv"].loc[
        tables["eligibility.csv"]["feature_set"].eq("paper_style_median")
    ].copy()
    tables["eligibility.csv"]["eligible"] = False
    tables["eligibility.csv"]["recommended"] = False
    tables["eligibility.csv"]["status"] = "smoke"
    payloads = _formal_json_payloads()
    payloads["run_metadata.json"] = {"mode": "smoke", "smoke": True, "seed": 20260804}
    context = _formal_record_context()
    context.update({"smoke": True, "recommendation": None})
    output = _allowed_output(tmp_path, monkeypatch)
    generation = _write_bundle(output / "smoke", tables, payloads, context)

    validate_round2_bundle(generation.staging_dir, smoke=True)

    eligibility = pd.read_csv(generation.staging_dir / "eligibility.csv")
    eligibility.loc[eligibility.index[0], "recommended"] = True
    eligibility.to_csv(generation.staging_dir / "eligibility.csv", index=False)
    _rehash(generation.staging_dir, "eligibility.csv")
    with pytest.raises(ValueError, match="smoke.*recommendation|recommendation.*smoke"):
        validate_round2_bundle(generation.staging_dir, smoke=True)


def test_round2_validator_rejects_mode_that_disagrees_with_output_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke child 不得冒充 formal bundle，正式根目錄也不得冒充 smoke。"""
    tables = _formal_tables()
    payloads = _formal_json_payloads()
    context = _formal_record_context()
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch) / "smoke",
        tables,
        payloads,
        context,
    )

    with pytest.raises(ValueError, match="smoke|formal|boundary"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        ["not", "a", "mapping"],
    ],
)
def test_round2_json_writer_rejects_nonfinite_or_non_mapping_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    """JSON root 必須是 mapping，所有巢狀數值必須 finite。"""
    generation = begin_round2_generation(_allowed_output(tmp_path, monkeypatch))
    payloads: dict[str, Any] = _formal_json_payloads()
    payloads["run_metadata.json"] = payload

    with pytest.raises((TypeError, ValueError), match="mapping|finite|NaN|Infinity"):
        write_round2_bundle(
            generation.staging_dir,
            _formal_tables(),
            payloads,
            _formal_record_context(),
        )

    assert list(generation.staging_dir.iterdir()) == []


def test_round2_artifact_hash_manifest_excludes_itself_and_run_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hash manifest 必須只涵蓋已關閉的 CSV、caller JSON 與 record。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )

    manifest = json.loads(
        (generation.staging_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )

    assert set(manifest["artifacts"]) == {
        *ROUND2_TABLE_NAMES,
        *_JSON_NAMES,
        "EXPERIMENT_RECORD.md",
    }
    assert "artifact_hashes.json" not in manifest["artifacts"]
    assert "run.log" not in manifest["artifacts"]
    assert all(set(item) == {"sha256", "bytes"} for item in manifest["artifacts"].values())


def test_round2_record_states_target_scope_and_approximation() -> None:
    """正式 record 必須先回答結論並限制 target 與 predictor 解讀。"""
    text = build_round2_experiment_record(_formal_record_context())

    assert text.index("## 結論") < text.index("## Target 定義")
    assert "IDO 螢光亮度 proxy" in text
    assert "不是免疫力" in text
    assert "phase-only paper-style approximation" in text
    assert "不改寫 Round 1 winner" in text
    predictor_section = text.split("## Predictor 範圍", maxsplit=1)[1].split("##", maxsplit=1)[0]
    assert "donor" not in predictor_section.lower()


def test_smoke_record_starts_with_no_scientific_conclusion() -> None:
    """Smoke record 第一段必須明示不形成 33 vs 93 科學結論。"""
    context = _formal_record_context()
    context.update({"smoke": True, "recommendation": None})

    text = build_round2_experiment_record(context)

    first_paragraph = text.split("\n\n", maxsplit=1)[0]
    assert "僅驗證流程，不是正式實驗結果" in first_paragraph
    assert "不形成 33 vs 93 科學結論" in text
