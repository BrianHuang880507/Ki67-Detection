"""測試 Exp3 Round 2 原子報告與 bundle 驗證合約。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import immunity.exp3.run_round2_paper93 as run_module
from immunity.exp3.benchmark import make_outer_splits, outer_split_manifest
import immunity.exp3.round2_reporting as reporting_module
from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES, PRIMARY_FOV_FEATURES
from immunity.exp3.round2_benchmark import (
    Round2Comparison,
    rank_round2_configurations,
)
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
_FROZEN_ROSTER_SHA256 = (
    "a8333f12e1591d9e4c4174f5c6fe13dd31550124b19aea3522f4d0f812e0e426"
)
_ROUND1_REQUIRED_ARTIFACTS = (
    "pairing_qc.csv",
    "data_manifest.csv",
    "segmentation_qc.csv",
    "outer_splits.csv",
    "feature_cache/image_level_basic.csv",
    "feature_cache/cell_level_basic.csv",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "model_ranking.csv",
    "hyperparameters.csv",
    "feature_importance.csv",
    "model_failures.csv",
    "feature_sets.json",
    "run_metadata.json",
)


def test_round2_reporting_can_be_imported_in_fresh_interpreter() -> None:
    """Fresh interpreter 可直接匯入 standalone validator。"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from immunity.exp3.round2_reporting import "
                "validate_round2_bundle; "
                "assert callable(validate_round2_bundle); "
                "print(validate_round2_bundle.__name__)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "validate_round2_bundle"


def test_default_authority_loader_verifies_real_fixed_round1_when_available() -> None:
    """本機保留正式 artifacts 時，default loader 必須通過固定 14 pins 與語意 gates。"""
    root = Path(__file__).resolve().parents[1] / "immunity" / "outputs" / "exp3"
    if not (root / "feature_cache" / "image_level_basic.csv").is_file():
        pytest.skip("本機未保留 ignored Round 1 authority artifacts")

    authority = reporting_module._load_round1_authority(root)

    assert len(authority["basic_images"]) == 693
    assert authority["basic_images"]["image_key"].nunique() == 693
    assert len(
        authority["outer_splits"].loc[
            :, ["validation", "fold"]
        ].drop_duplicates()
    ) == 23


@pytest.mark.parametrize("bad_count", [3.5, True, -1])
def test_round2_record_rejects_non_strict_display_counts(bad_count: object) -> None:
    """Record count formatter 不得截斷 fraction 或接受 bool/negative。"""
    context = _formal_record_context()
    context["valid_pair_observations"] = bad_count

    with pytest.raises(ValueError, match="count|integer|analyzed"):
        build_round2_experiment_record(context)


def _allowed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """將測試限定在暫存的正式 Round 2 根目錄。"""
    output = (tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93").resolve()
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", output)
    _write_authoritative_round1_fixture(output.parent)
    monkeypatch.setattr(
        reporting_module,
        "_load_round1_authority",
        lambda root: _synthetic_round1_authority(root),
    )
    return output


def _authoritative_basic_images() -> pd.DataFrame:
    """建立 validator 必須從 sibling Round 1 root 讀取的權威 target/basic 表。"""
    image_keys = [f"image-{index:03d}" for index in range(693)]
    return pd.DataFrame(
        {
            "image_key": image_keys,
            "IDO_score": np.linspace(0.0, 1.0, 693),
            "b_id": [f"B{index % 3 + 1}" for index in range(693)],
            "passage": [index % 3 + 5 for index in range(693)],
            "group_id": [f"G{index % 9 + 1}" for index in range(693)],
            "condition_index": [index % 8 + 1 for index in range(693)],
            **{
                feature: np.full(693, float(index))
                for index, feature in enumerate(PRIMARY_FOV_FEATURES)
            },
        }
    )


_SYNTHETIC_AUTHORITY_CACHE: dict[str, Any] | None = None


def _synthetic_round1_authority(root: Path) -> dict[str, Any]:
    """回傳不信任 bundle 的 synthetic Round 1 raw evidence fresh copies。"""
    global _SYNTHETIC_AUTHORITY_CACHE
    if _SYNTHETIC_AUTHORITY_CACHE is None:
        tables = _formal_tables()
        baseline = tables["fold_metrics.csv"][
            tables["fold_metrics.csv"]["source_round"].eq("round1")
        ]
        baseline_oof = tables["oof_predictions.csv"][
            tables["oof_predictions.csv"]["source_round"].eq("round1")
        ]
        baseline_hyper = tables["hyperparameters.csv"][
            tables["hyperparameters.csv"]["source_round"].eq("round1")
        ]
        baseline_importance = tables["feature_importance.csv"][
            tables["feature_importance.csv"]["source_round"].eq("round1")
        ]
        baseline_failures = tables["model_failures.csv"][
            tables["model_failures.csv"]["source_round"].eq("round1")
        ]

        def raw(frame: pd.DataFrame) -> pd.DataFrame:
            result = frame.drop(
                columns=["source_round", "configuration_id"], errors="ignore"
            ).reset_index(drop=True)
            #模擬 authority CSV loader；此 synthetic fixture 的 fold 全為數字。
            if "fold" in result:
                result["fold"] = pd.to_numeric(result["fold"])
            return result

        _SYNTHETIC_AUTHORITY_CACHE = {
            "basic_images": pd.read_csv(
                root / "feature_cache" / "image_level_basic.csv"
            ),
            "outer_splits": pd.read_csv(root / "outer_splits.csv"),
            "selected_metrics": pd.concat(
                [raw(baseline), raw(tables["dummy_fold_metrics.csv"])],
                ignore_index=True,
            ),
            "selected_predictions": pd.concat(
                [raw(baseline_oof), raw(tables["dummy_oof_predictions.csv"])],
                ignore_index=True,
            ),
            "selected_hyperparameters": raw(baseline_hyper),
            "selected_importance": raw(baseline_importance),
            "selected_failures": raw(baseline_failures),
            "roster_sha256": _FROZEN_ROSTER_SHA256,
        }
    return {
        name: value.copy(deep=True) if isinstance(value, pd.DataFrame) else value
        for name, value in _SYNTHETIC_AUTHORITY_CACHE.items()
    }


def _authoritative_outer_splits() -> pd.DataFrame:
    """建立與 formal fixture 相同的權威 Round 1 split membership。"""
    image_keys = [f"image-{index:03d}" for index in range(693)]
    rows: list[dict[str, str]] = []
    for validation, fold, _, test_keys in _split_contract():
        test_set = set(test_keys)
        rows.extend(
            {
                "validation": validation,
                "fold": fold,
                "image_key": image_key,
                "role": "test" if image_key in test_set else "train",
            }
            for image_key in image_keys
        )
    return pd.DataFrame(rows)


def _write_authoritative_round1_fixture(root: Path) -> None:
    """寫出固定 14-artifact pins，其中兩份 CSV 提供 target/split 權威語意。"""
    (root / "feature_cache").mkdir(parents=True, exist_ok=True)
    basic_path = root / "feature_cache" / "image_level_basic.csv"
    split_path = root / "outer_splits.csv"
    if not basic_path.exists():
        _authoritative_basic_images().to_csv(basic_path, index=False)
    if not split_path.exists():
        _authoritative_outer_splits().to_csv(split_path, index=False)
    for relative in _ROUND1_REQUIRED_ARTIFACTS:
        path = root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")


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


def _expected_regression_metrics(
    observed: list[float] | np.ndarray,
    predicted: list[float] | np.ndarray,
) -> dict[str, float]:
    """以手算公式建立 fixture metrics，不呼叫 production metric helper。"""
    actual = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    residual = actual - estimate
    observed_constant = bool(np.all(actual == actual[0]))
    predicted_constant = bool(np.all(estimate == estimate[0]))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": (
            np.nan
            if len(actual) < 2 or observed_constant
            else float(1.0 - np.sum(residual**2) / denominator)
        ),
        "spearman": (
            np.nan
            if len(actual) < 2 or observed_constant or predicted_constant
            else float(pd.Series(actual).corr(pd.Series(estimate), method="spearman"))
        ),
    }


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
    score_by_key = data_snapshot.set_index("image_key")["IDO_score"].to_dict()
    split_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    dummy_metric_rows: list[dict[str, object]] = []
    dummy_prediction_rows: list[dict[str, object]] = []
    hyperparameter_rows: list[dict[str, object]] = []
    for validation, fold, split_id, test_keys in _split_contract():
        test_set = set(test_keys)
        observed = np.asarray([score_by_key[key] for key in test_keys], dtype=float)
        dummy_predicted = np.full(len(test_keys), 0.5, dtype=float)
        dummy_values = _expected_regression_metrics(observed, dummy_predicted)
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
                **dummy_values,
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
                    "observed_ido_score": score_by_key[image_key],
                    "predicted_ido_score": 0.5,
                }
            )
        for configuration_id, model, feature_set, source_round in _CONFIGURATIONS:
            candidate_values = _expected_regression_metrics(observed, observed)
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
                    **candidate_values,
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
                        "observed_ido_score": score_by_key[image_key],
                        "predicted_ido_score": score_by_key[image_key],
                    }
                )

    roster_counts = {
        image_key: 35 if index < 414 else 34
        for index, image_key in enumerate(image_keys)
    }
    assert sum(roster_counts.values()) == 23976
    valid_counts = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "feature": feature,
                "roster_cell_count": roster_counts[image_key],
                "finite_cell_count": roster_counts[image_key],
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
    tables = {
        "data_snapshot.csv": data_snapshot,
        "outer_splits.csv": pd.DataFrame(split_rows),
        "mask_provenance_qc.csv": pd.DataFrame(
            {"image_key": image_keys, "status": ["passed"] * 693}
        ),
        "feature_valid_counts.csv": valid_counts,
        "extraction_qc.csv": pd.DataFrame(
            {
                "image_key": image_keys,
                "aggregation_unit": ["frozen_nucleus_cell_pair"] * 693,
                "roster_pair_count": [roster_counts[key] for key in image_keys],
                "extracted_pair_count": [roster_counts[key] for key in image_keys],
                "unique_cell_count": [roster_counts[key] for key in image_keys],
                "unique_nucleus_count": [roster_counts[key] for key in image_keys],
                "multi_nucleus_cell_count": [0] * 693,
                "multi_nucleus_pair_count": [0] * 693,
                "max_nuclei_per_cell": [1] * 693,
                "retained_outside_pair_count": [0] * 693,
                "max_retained_outside_fraction": [0.0] * 693,
                "max_nucleus_outside_fraction": [0.05] * 693,
                "status": ["passed"] * 693,
                "reason": [""] * 693,
            }
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
        "feature_set_comparison.csv": pd.DataFrame(),
        "eligibility.csv": pd.DataFrame(),
    }
    _refresh_ranking_tables(tables)
    return tables


def _refresh_ranking_tables(tables: dict[str, pd.DataFrame]) -> None:
    """由 raw fold/OOF/dummy evidence 產生 fixture 的完整 derived ranking。"""
    expected_splits = {
        validation: [
            split_id
            for candidate_validation, _, split_id, _ in _split_contract()
            if candidate_validation == validation
        ]
        for validation, _ in _VALIDATIONS
    }
    fold_metrics = tables["fold_metrics.csv"]
    fold_metrics.attrs["round2_expected_splits"] = expected_splits
    comparison = Round2Comparison(
        fold_metrics=fold_metrics,
        predictions=tables["oof_predictions.csv"],
        dummy_metrics=tables["dummy_fold_metrics.csv"],
        dummy_predictions=tables["dummy_oof_predictions.csv"],
        hyperparameters=tables["hyperparameters.csv"],
        feature_importance=tables["feature_importance.csv"],
        failures=tables["model_failures.csv"],
    )
    ranking = rank_round2_configurations(
        comparison,
        {
            "basic_median": list(PRIMARY_FOV_FEATURES),
            "paper_style_median": list(PAPER_STYLE_FOV_FEATURES),
        },
    )
    tables["feature_set_comparison.csv"] = ranking.copy()
    tables["eligibility.csv"] = ranking.copy()


def _synchronize_metrics_from_oof(tables: dict[str, pd.DataFrame]) -> None:
    """讓惡意 bundle 內部自洽，以確認 validator 必須依 raw OOF/外部權威重算。"""
    for metric_name, prediction_name in (
        ("fold_metrics.csv", "oof_predictions.csv"),
        ("dummy_fold_metrics.csv", "dummy_oof_predictions.csv"),
    ):
        metrics = tables[metric_name].copy()
        predictions = tables[prediction_name]
        for index, row in metrics.iterrows():
            selected = predictions.loc[
                predictions["configuration_id"].astype(str).eq(
                    str(row["configuration_id"])
                )
                & predictions["split_id"].astype(str).eq(str(row["split_id"]))
            ]
            values = _expected_regression_metrics(
                selected["observed_ido_score"].to_numpy(dtype=float),
                selected["predicted_ido_score"].to_numpy(dtype=float),
            )
            for field, value in values.items():
                metrics.loc[index, field] = value
        tables[metric_name] = metrics


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
    _refresh_ranking_tables(tables)
    return tables


def _formal_json_payloads() -> dict[str, dict[str, Any]]:
    """建立三個固定 JSON artifacts。"""
    return {
        "feature_sets.json": {
            "paper_style_median": list(PAPER_STYLE_FOV_FEATURES),
            "basic_median": list(PAPER_STYLE_FOV_FEATURES[:33]),
        },
        "baseline_provenance.json": {
            "round1_root": "D:/evidence/immunity/outputs/exp3",
            "read_only": True,
            "artifact_sha256": {"source.csv": "b" * 64},
            "roster_sha256": _FROZEN_ROSTER_SHA256,
        },
        "run_metadata.json": _run_metadata_payload(smoke=False),
    }


def _run_metadata_payload(*, smoke: bool) -> dict[str, Any]:
    """建立含完整可重現性 schema 的 formal/smoke metadata。"""
    return {
        "mode": "smoke" if smoke else "formal",
        "smoke": smoke,
        "diagnostic_splits": smoke,
        "non_formal": smoke,
        "no_scientific_conclusion": smoke,
        "seed": 20260804,
        "analyzed_images": 693,
        "valid_cells": 23976,
        "aggregation_unit": "frozen_nucleus_cell_pair",
        "valid_pair_observations": 23976,
        "pair_mapping_summary": {
            "aggregation_unit": "frozen_nucleus_cell_pair",
            "image_count": 693,
            "pair_observation_count": 23976,
            "unique_cell_count": 23976,
            "unique_nucleus_count": 23976,
            "multi_nucleus_cell_count": 0,
            "multi_nucleus_pair_count": 0,
            "max_nuclei_per_cell": 1,
            "retained_outside_pair_count": 0,
            "max_retained_outside_fraction": 0.0,
            "max_nucleus_outside_fraction": 0.05,
        },
        "exclusions": 26,
        "original_config": {
            "round1": {
                "dir": "immunity/outputs/exp3",
                "artifact_sha256": {"source.csv": "b" * 64},
                "roster_sha256": _FROZEN_ROSTER_SHA256,
            }
        },
        "effective_config": {
            "round1": {
                "dir": "immunity/outputs/exp3",
                "artifact_sha256": {"source.csv": "b" * 64},
                "roster_sha256": _FROZEN_ROSTER_SHA256,
            }
        },
        "reproducibility": {
            "schema_version": 1,
            "git": {
                "commit": "4b3a12093975feb9f6166b7c5fc489434fbebba4",
                "branch": "codex/exp3-multimodel-benchmark",
                "status_porcelain": (
                    " M immunity/configs/exp3.yaml\n"
                    " M tests/test_immunity_exp3_cli.py"
                ),
                "dirty": True,
            },
            "python": {"version": "3.10.11"},
            "packages": {
                "numpy": "1.26.4",
                "pandas": "2.2.3",
                "scikit-learn": "1.5.2",
                "OpenCV": "4.10.0",
                "PyYAML": "6.0.2",
            },
            "timing": {
                "started_at_utc": "2026-08-11T13:09:49.867240Z",
                "completed_at_utc": "2026-08-11T13:10:02.367240Z",
                "runtime_seconds": 12.5,
            },
        },
    }


def _synchronize_smoke_metadata(
    payloads: dict[str, dict[str, Any]],
    tables: dict[str, pd.DataFrame],
) -> None:
    """依 smoke subset 的 extraction QC 重算非科學性 metadata totals。"""
    metadata = _run_metadata_payload(smoke=True)
    extraction = tables["extraction_qc.csv"]
    summary = metadata["pair_mapping_summary"]
    summary.update(
        {
            "image_count": len(tables["data_snapshot.csv"]),
            "pair_observation_count": int(extraction["extracted_pair_count"].sum()),
            "unique_cell_count": int(extraction["unique_cell_count"].sum()),
            "unique_nucleus_count": int(extraction["unique_nucleus_count"].sum()),
            "multi_nucleus_cell_count": int(
                extraction["multi_nucleus_cell_count"].sum()
            ),
            "multi_nucleus_pair_count": int(
                extraction["multi_nucleus_pair_count"].sum()
            ),
            "max_nuclei_per_cell": int(extraction["max_nuclei_per_cell"].max()),
            "retained_outside_pair_count": int(
                extraction["retained_outside_pair_count"].sum()
            ),
            "max_retained_outside_fraction": float(
                extraction["max_retained_outside_fraction"].max()
            ),
        }
    )
    metadata["analyzed_images"] = summary["image_count"]
    metadata["valid_cells"] = summary["pair_observation_count"]
    metadata["valid_pair_observations"] = summary["pair_observation_count"]
    payloads["run_metadata.json"] = metadata


def _formal_record_context() -> dict[str, Any]:
    """建立答案先行的正式 record context。"""
    return {
        "smoke": False,
        "recommendation": "extra_trees__basic_median",
        "paper93_better_than_basic": False,
        "analyzed_images": 693,
        "valid_cells": 23976,
        "valid_pair_observations": 23976,
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
        "pair_mapping_summary": {
            "aggregation_unit": "frozen_nucleus_cell_pair",
            "image_count": 693,
            "pair_observation_count": 23976,
            "unique_cell_count": 23012,
            "unique_nucleus_count": 23976,
            "multi_nucleus_cell_count": 938,
            "multi_nucleus_pair_count": 1902,
            "max_nuclei_per_cell": 3,
            "retained_outside_pair_count": 211,
            "max_retained_outside_fraction": 0.05,
            "max_nucleus_outside_fraction": 0.05,
        },
    }


def _smoke_bundle_inputs() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """建立 split/metric/OOF/hyperparameter 完全對帳的 smoke bundle。"""
    tables = _formal_tables()
    keep_keys = [f"image-{index:03d}" for index in range(72)]
    tables["data_snapshot.csv"] = tables["data_snapshot.csv"].loc[
        tables["data_snapshot.csv"]["image_key"].isin(keep_keys)
    ].copy()
    for name in (
        "mask_provenance_qc.csv",
        "feature_valid_counts.csv",
        "extraction_qc.csv",
    ):
        tables[name] = tables[name].loc[
            tables[name]["image_key"].isin(keep_keys)
        ].copy()

    score_by_key = tables["data_snapshot.csv"].set_index("image_key")["IDO_score"]
    authority_subset = _authoritative_basic_images().loc[
        lambda frame: frame["image_key"].isin(keep_keys)
    ].copy()
    canonical_manifest = outer_split_manifest(
        authority_subset,
        make_outer_splits(authority_subset),
    )
    split_rows = canonical_manifest.loc[
        :, ["validation", "fold", "image_key", "role"]
    ].to_dict("records")
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    hyperparameter_rows: list[dict[str, object]] = []
    paper_configurations = [row for row in _CONFIGURATIONS if row[2] == "paper_style_median"]
    for validation, fold_count in _VALIDATIONS:
        family = canonical_manifest.loc[
            canonical_manifest["validation"].eq(validation)
        ]
        assert family["fold"].nunique() == fold_count
        for fold_value in sorted(family["fold"].astype(str).unique()):
            fold = str(fold_value)
            test_keys = family.loc[
                family["fold"].astype(str).eq(fold)
                & family["role"].eq("test"),
                "image_key",
            ].astype(str).tolist()
            split_id = f"{validation}:{fold}"
            observed = score_by_key.loc[test_keys].to_numpy(dtype=float)
            values = _expected_regression_metrics(observed, observed)
            for configuration_id, model, feature_set, source_round in paper_configurations:
                identity = {
                    "validation": validation,
                    "fold": fold,
                    "split_id": split_id,
                    "model": model,
                    "feature_set": feature_set,
                    "source_round": source_round,
                    "configuration_id": configuration_id,
                }
                metric_rows.append(
                    {**identity, "n_test": len(test_keys), **values, "status": "ok"}
                )
                hyperparameter_rows.append(
                    {**identity, "seed": 20260804, "best_params_json": "{}"}
                )
                prediction_rows.extend(
                    {
                        **identity,
                        "image_key": image_key,
                        "observed_ido_score": float(score_by_key.loc[image_key]),
                        "predicted_ido_score": float(score_by_key.loc[image_key]),
                    }
                    for image_key in test_keys
                )

    tables["outer_splits.csv"] = pd.DataFrame(split_rows)
    tables["fold_metrics.csv"] = pd.DataFrame(metric_rows)
    tables["oof_predictions.csv"] = pd.DataFrame(prediction_rows)
    tables["hyperparameters.csv"] = pd.DataFrame(hyperparameter_rows)
    tables["dummy_fold_metrics.csv"] = tables["dummy_fold_metrics.csv"].iloc[0:0]
    tables["dummy_oof_predictions.csv"] = tables["dummy_oof_predictions.csv"].iloc[0:0]
    tables["feature_set_comparison.csv"] = tables[
        "feature_set_comparison.csv"
    ].iloc[0:0]
    tables["eligibility.csv"] = pd.DataFrame(
        {
            "configuration_id": [row[0] for row in paper_configurations],
            "model": [row[1] for row in paper_configurations],
            "feature_set": [row[2] for row in paper_configurations],
            "source_round": [row[3] for row in paper_configurations],
            "eligible": [False, False],
            "recommended": [False, False],
            "status": ["smoke", "smoke"],
        }
    )
    payloads = _formal_json_payloads()
    _synchronize_smoke_metadata(payloads, tables)
    context = _formal_record_context()
    context.update({"smoke": True, "recommendation": None})
    return tables, payloads, context


def _write_bundle(
    output: Path,
    tables: dict[str, pd.DataFrame],
    json_payloads: dict[str, dict[str, Any]],
    record_context: dict[str, Any],
) -> Any:
    """建立 generation 並寫入含已關閉 run.log 的 bundle。"""
    generation = begin_round2_generation(output)
    payloads = copy.deepcopy(json_payloads)
    formal_output = output.parent if output.name == "smoke" else output
    round1_root = formal_output.parent.resolve()
    pins = {
        relative: hashlib.sha256((round1_root / relative).read_bytes()).hexdigest()
        for relative in _ROUND1_REQUIRED_ARTIFACTS
    }
    provenance = payloads["baseline_provenance.json"]
    provenance["round1_root"] = str(round1_root)
    provenance["artifact_sha256"] = pins
    for config_name in ("original_config", "effective_config"):
        round1 = payloads["run_metadata.json"][config_name]["round1"]
        round1["dir"] = str(round1_root)
        round1["artifact_sha256"] = dict(pins)
    (generation.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    write_round2_bundle(
        generation.staging_dir,
        tables,
        payloads,
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


def _snapshot_files(directory: Path) -> dict[str, bytes]:
    """擷取目錄內所有 regular files 的相對路徑與原始內容。"""
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


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


def test_round2_generation_rejects_same_target_owner_in_real_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """另一個 process 持有同 target generation 時，begin 必須立即失敗。"""
    output = _allowed_output(tmp_path, monkeypatch)
    generation = begin_round2_generation(output)
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "import immunity.exp3.run_round2_paper93 as run_module",
            "from immunity.exp3.round2_reporting import begin_round2_generation",
            "run_module.ROUND2_OUTPUT_ROOT = Path(sys.argv[1]).resolve()",
            "try:",
            "    begin_round2_generation(Path(sys.argv[1]))",
            "except RuntimeError as error:",
            "    print(error)",
            "    raise SystemExit(17)",
            "raise SystemExit(0)",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 17, completed.stderr
    assert "active" in completed.stdout.casefold() or "使用中" in completed.stdout
    (generation.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    quarantine_round2_generation(generation)


def test_round2_generation_allows_formal_and_smoke_owners_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formal 與 smoke 是不同 publication target，彼此不得互相阻塞。"""
    output = _allowed_output(tmp_path, monkeypatch)
    formal = begin_round2_generation(output)
    smoke = begin_round2_generation(output / "smoke")

    assert formal.output_dir == output
    assert smoke.output_dir == output / "smoke"
    for generation in (smoke, formal):
        (generation.staging_dir / "run.log").write_text(
            "closed\n", encoding="utf-8"
        )
        quarantine_round2_generation(generation)


def test_round2_generation_recovers_from_stale_lock_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """殘留 lock marker 沒有 OS owner 時，不得永久阻止新 generation。"""
    output = _allowed_output(tmp_path, monkeypatch)
    generations = output / "_generations"
    generations.mkdir(parents=True)
    marker = generations / ".publication.lock"
    marker.write_bytes(b"stale-marker-from-crashed-process\n")

    generation = begin_round2_generation(output)

    (generation.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    quarantine_round2_generation(generation)
    assert marker.read_bytes() == b"stale-marker-from-crashed-process\n"


def test_round2_generation_process_crash_releases_os_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner process 未執行 cleanup 就終止時，OS 必須自動釋放 publication lock。"""
    output = _allowed_output(tmp_path, monkeypatch)
    script = "\n".join(
        (
            "import os",
            "import sys",
            "from pathlib import Path",
            "import immunity.exp3.run_round2_paper93 as run_module",
            "from immunity.exp3.round2_reporting import begin_round2_generation",
            "run_module.ROUND2_OUTPUT_ROOT = Path(sys.argv[1]).resolve()",
            "begin_round2_generation(Path(sys.argv[1]))",
            "os._exit(23)",
        )
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=Path.cwd(),
        check=False,
        timeout=30,
    )
    assert crashed.returncode == 23
    assert (output / "_generations" / ".publication.lock").is_file()

    generation = begin_round2_generation(output)

    (generation.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    quarantine_round2_generation(generation)


def test_round2_begin_base_exception_releases_acquired_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Begin 取得 OS lock 後中斷，也必須讓下一個 owner 立即接手。"""
    output = _allowed_output(tmp_path, monkeypatch)
    interruption = KeyboardInterrupt("synthetic begin interruption")
    with monkeypatch.context() as patch:
        patch.setattr(
            reporting_module,
            "_generation_suffix",
            lambda: (_ for _ in ()).throw(interruption),
        )
        with pytest.raises(KeyboardInterrupt) as caught:
            begin_round2_generation(output)
    assert caught.value is interruption
    assert (output / "_generations" / ".publication.lock").is_file()

    generation = begin_round2_generation(output)

    (generation.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    quarantine_round2_generation(generation)


def test_round2_forged_and_stale_generations_cannot_publish_or_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只有 begin 回傳的 active object 可完成 lifecycle，copy 與已釋放物件皆拒絕。"""
    output = _allowed_output(tmp_path, monkeypatch)
    publishable = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    forged_publish = replace(publishable)

    with pytest.raises(ValueError, match="owner|ownership|active|持有|已釋放"):
        publish_round2_generation(forged_publish)
    publish_round2_generation(publishable)
    with pytest.raises(ValueError, match="owner|ownership|active|持有|已釋放"):
        publish_round2_generation(publishable)

    quarantinable = begin_round2_generation(output / "smoke")
    (quarantinable.staging_dir / "run.log").write_text(
        "closed\n", encoding="utf-8"
    )
    forged_quarantine = replace(quarantinable)
    with pytest.raises(ValueError, match="owner|ownership|active|持有|已釋放"):
        quarantine_round2_generation(forged_quarantine)
    quarantine_round2_generation(quarantinable)
    with pytest.raises(ValueError, match="owner|ownership|active|持有|已釋放"):
        quarantine_round2_generation(quarantinable)


def test_round2_publish_validation_failure_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staging validation 在首次 move 前失敗，也不得遺留 target ownership。"""
    output = _allowed_output(tmp_path, monkeypatch)
    invalid = begin_round2_generation(output)
    (invalid.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")

    with pytest.raises(ValueError):
        publish_round2_generation(invalid)

    next_generation = begin_round2_generation(output)
    (next_generation.staging_dir / "run.log").write_text(
        "closed\n", encoding="utf-8"
    )
    quarantine_round2_generation(next_generation)


def test_round2_quarantine_base_exception_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quarantine 搬移前遭 BaseException 中斷，也不得遺留 target ownership。"""
    output = _allowed_output(tmp_path, monkeypatch)
    interrupted = begin_round2_generation(output)
    (interrupted.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    interruption = KeyboardInterrupt("synthetic quarantine interruption")
    with monkeypatch.context() as patch:
        patch.setattr(
            reporting_module,
            "_generation_suffix",
            lambda: (_ for _ in ()).throw(interruption),
        )
        with pytest.raises(KeyboardInterrupt) as caught:
            quarantine_round2_generation(interrupted)
    assert caught.value is interruption

    next_generation = begin_round2_generation(output)
    (next_generation.staging_dir / "run.log").write_text(
        "closed\n", encoding="utf-8"
    )
    quarantine_round2_generation(next_generation)


@pytest.mark.parametrize("interruption_type", [RuntimeError, KeyboardInterrupt])
def test_round2_post_publish_check_failure_rolls_back_before_releasing_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
) -> None:
    """Post-publish check 失敗須在同一 ownership 內還原舊 root 並重拋原物件。"""
    output = _allowed_output(tmp_path, monkeypatch)
    first = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (first.staging_dir / "run.log").write_bytes(b"old generation\n")
    publish_round2_generation(first)
    second = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (second.staging_dir / "run.log").write_bytes(b"new generation\n")
    old_root = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    staged = _snapshot_files(second.staging_dir)
    interruption = interruption_type("synthetic post-publish interruption")
    calls: list[tuple[Path, Path | None]] = []

    def reject_published_root(root: Path, archive: Path | None) -> None:
        calls.append((root, archive))
        assert root == output
        assert archive is not None
        assert (root / "run.log").read_bytes() == b"new generation\n"
        assert (archive / "run.log").read_bytes() == b"old generation\n"
        raise interruption

    with pytest.raises(interruption_type) as caught:
        publish_round2_generation(second, post_publish_check=reject_published_root)

    assert caught.value is interruption
    assert len(calls) == 1
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == old_root
    assert _snapshot_files(second.staging_dir) == staged
    assert not list((output / "_generations").glob("archive-*"))

    next_generation = begin_round2_generation(output)
    (next_generation.staging_dir / "run.log").write_text(
        "closed\n", encoding="utf-8"
    )
    quarantine_round2_generation(next_generation)


def test_round2_post_publish_check_receives_matching_archive_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成功 hook 僅執行一次，且收到本 transaction 建立的 matching archive。"""
    output = _allowed_output(tmp_path, monkeypatch)
    first = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (first.staging_dir / "run.log").write_bytes(b"old generation\n")
    publish_round2_generation(first)
    old_root = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    second = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (second.staging_dir / "run.log").write_bytes(b"new generation\n")
    calls: list[tuple[Path, Path | None]] = []

    def verify_transaction(root: Path, archive: Path | None) -> None:
        calls.append((root, archive))
        assert root == output
        assert archive is not None
        assert {
            path.name: path.read_bytes()
            for path in archive.iterdir()
            if path.is_file()
        } == old_root
        assert (root / "run.log").read_bytes() == b"new generation\n"

    publish_round2_generation(second, post_publish_check=verify_transaction)

    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][1].is_dir()


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


def test_round2_validator_accepts_formal_published_root_with_safe_smoke_child_without_recursing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formal published root 可保留安全的 smoke child，且不遞迴驗證其內容。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    publish_round2_generation(generation)
    smoke = generation.output_dir / "smoke"
    smoke.mkdir()
    (smoke / "not-validated-by-formal.txt").write_text("smoke-owned\n", encoding="utf-8")

    validate_round2_bundle(generation.output_dir, smoke=False)


def test_round2_validator_rejects_smoke_directory_in_formal_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formal staging 不得把 smoke child 視為 managed directory。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (generation.staging_dir / "smoke").mkdir()

    with pytest.raises(ValueError, match="artifact keys|extra|smoke"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_round2_validator_rejects_nested_smoke_directory_in_published_smoke_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published smoke root 不得再接受 nested smoke directory。"""
    tables, payloads, context = _smoke_bundle_inputs()
    output = _allowed_output(tmp_path, monkeypatch)
    generation = _write_bundle(output / "smoke", tables, payloads, context)
    publish_round2_generation(generation)
    (generation.output_dir / "smoke").mkdir()

    with pytest.raises(ValueError, match="artifact keys|extra|smoke"):
        validate_round2_bundle(generation.output_dir, smoke=True)


def test_round2_validator_rejects_reparse_smoke_child_in_formal_published_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formal root 的 smoke child 若為 symlink/junction/reparse point 必須 fail-closed。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    publish_round2_generation(generation)
    outside = tmp_path / "outside-smoke"
    outside.mkdir()
    smoke = generation.output_dir / "smoke"
    try:
        smoke.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"此環境無法建立 directory symlink：{error}")

    with pytest.raises(ValueError, match="symlink|junction|reparse|逃逸"):
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
    (managed / ".publication.lock").unlink()
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


def test_round2_publication_rejects_unexpected_root_directory_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publisher 必須在首次 rename 前拒絕任何非 managed root child。"""
    output = _allowed_output(tmp_path, monkeypatch)
    generation = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    unexpected = output / "unmanaged"
    unexpected.mkdir()
    (unexpected / "owned.txt").write_bytes(b"must stay")
    staging_before = _snapshot_files(generation.staging_dir)

    with pytest.raises(ValueError, match="unmanaged|非預期"):
        publish_round2_generation(generation)

    assert _snapshot_files(generation.staging_dir) == staging_before
    assert _snapshot_files(unexpected) == {"owned.txt": b"must stay"}
    assert not any(path.is_file() for path in output.iterdir())


@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_round2_publication_restores_exact_state_after_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
) -> None:
    """非 Exception 中斷發生在 replaces 之間也須精確復原並重拋原物件。"""
    import immunity.exp3.round2_reporting as reporting

    output = _allowed_output(tmp_path, monkeypatch)
    first = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (first.staging_dir / "run.log").write_bytes(b"old generation\n")
    publish_round2_generation(first)
    second = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    (second.staging_dir / "run.log").write_bytes(b"new generation\n")
    old_root = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    staged = _snapshot_files(second.staging_dir)
    assert not list((output / "_generations").glob("archive-*"))

    real_replace = reporting.os.replace
    interrupt_at = len(old_root) + 2
    calls = 0
    interruption = interruption_type("synthetic publication interruption")

    def interrupt_between_replaces(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == interrupt_at:
            raise interruption
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", interrupt_between_replaces)

    with pytest.raises(interruption_type) as caught:
        publish_round2_generation(second)

    assert caught.value is interruption
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == old_root
    assert _snapshot_files(second.staging_dir) == staged
    assert not list((output / "_generations").glob("archive-*"))


def test_round2_publication_validates_published_root_before_removing_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最後一次 replace 後的 corruption 必須由 published-root gate 攔截並復原。"""
    import immunity.exp3.round2_reporting as reporting

    output = _allowed_output(tmp_path, monkeypatch)
    generation = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    real_replace = reporting.os.replace

    def corrupt_after_last_publish(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        real_replace(source, destination)
        destination_path = Path(destination)
        if destination_path.parent == output and destination_path.name == "run.log":
            with (output / "data_snapshot.csv").open("ab") as handle:
                handle.write(b"corrupted-after-replace")

    monkeypatch.setattr(reporting.os, "replace", corrupt_after_last_publish)

    with pytest.raises(ValueError, match="hash|bytes"):
        publish_round2_generation(generation)

    assert generation.staging_dir.exists()
    assert set(_snapshot_files(generation.staging_dir)) == {
        *ROUND2_TABLE_NAMES,
        *_JSON_NAMES,
        "artifact_hashes.json",
        "EXPERIMENT_RECORD.md",
        "run.log",
    }
    assert not any(path.is_file() for path in output.iterdir())


def test_round2_publication_never_moves_or_overwrites_round1_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 2 發布邊界不能觸及同層的 Round 1 artifact。"""
    output = _allowed_output(tmp_path, monkeypatch)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    expected.parent.mkdir(parents=True, exist_ok=True)
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


def test_formal_validator_rehashes_round1_artifacts_at_derived_sibling_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundle pins 正確也不能掩蓋發布後被改寫的 sibling Round 1 bytes。"""
    output = _allowed_output(tmp_path, monkeypatch)
    generation = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    with (output.parent / "outer_splits.csv").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="Round 1|SHA-256|hash|pin"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_formal_validator_rejects_bundle_controlled_alternate_round1_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance/config 即使彼此自洽，也不可把 authority 改指任意替代目錄。"""
    output = _allowed_output(tmp_path, monkeypatch)
    generation = _write_bundle(
        output,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    alternate = tmp_path / "alternate-round1"
    _write_authoritative_round1_fixture(alternate)
    pins = {
        relative: hashlib.sha256((alternate / relative).read_bytes()).hexdigest()
        for relative in _ROUND1_REQUIRED_ARTIFACTS
    }
    provenance_path = generation.staging_dir / "baseline_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["round1_root"] = str(alternate.resolve())
    provenance["artifact_sha256"] = pins
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_path = generation.staging_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for config_name in ("original_config", "effective_config"):
        metadata[config_name]["round1"]["dir"] = str(alternate.resolve())
        metadata[config_name]["round1"]["artifact_sha256"] = dict(pins)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rehash(generation.staging_dir, "baseline_provenance.json")
    _rehash(generation.staging_dir, "run_metadata.json")

    with pytest.raises(ValueError, match="Round 1.*root|sibling|boundary|derived"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_formal_validator_anchors_basic_predictors_to_round1_image_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """內部 finite 的 33-feature drift 仍須由權威 image_level_basic 擋下。"""
    tables = _formal_tables()
    tables["data_snapshot.csv"].loc[0, PRIMARY_FOV_FEATURES[0]] += 0.25
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="Round 1|basic|data_snapshot|authority"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_formal_validator_anchors_target_despite_internally_consistent_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同時竄改 snapshot、OOF、metrics 與 ranking 也不可改寫 Round 1 target。"""
    tables = _formal_tables()
    image_key = "image-000"
    tables["data_snapshot.csv"].loc[
        tables["data_snapshot.csv"]["image_key"].eq(image_key), "IDO_score"
    ] += 0.125
    forged_target = float(
        tables["data_snapshot.csv"].loc[
            tables["data_snapshot.csv"]["image_key"].eq(image_key), "IDO_score"
        ].iloc[0]
    )
    for name in ("oof_predictions.csv", "dummy_oof_predictions.csv"):
        rows = tables[name]["image_key"].eq(image_key)
        tables[name].loc[rows, "observed_ido_score"] = forged_target
    _synchronize_metrics_from_oof(tables)
    _refresh_ranking_tables(tables)
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="Round 1|target|IDO_score|authority"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_formal_validator_anchors_split_despite_internal_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published split 與所有下游 identity 一起交換仍不得偏離 Round 1 authority。"""
    tables = _formal_tables()
    validation = "leave_one_b_out"
    for name in (
        "outer_splits.csv",
        "fold_metrics.csv",
        "oof_predictions.csv",
        "dummy_fold_metrics.csv",
        "dummy_oof_predictions.csv",
        "hyperparameters.csv",
    ):
        frame = tables[name].copy()
        selected = frame["validation"].eq(validation)
        fold = frame.loc[selected, "fold"].astype(str)
        frame.loc[selected, "fold"] = fold.map({"1": "2", "2": "1"}).fillna(fold)
        if "split_id" in frame:
            split_id = frame.loc[selected, "split_id"].astype(str)
            frame.loc[selected, "split_id"] = split_id.map(
                {
                    f"{validation}:1": f"{validation}:2",
                    f"{validation}:2": f"{validation}:1",
                }
            ).fillna(split_id)
        tables[name] = frame
    _refresh_ranking_tables(tables)
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="Round 1|outer_splits|membership|authority"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_formal_validator_rejects_coordinated_round1_baseline_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Basic33 OOF、metrics 與 ranking 同步改寫仍須偏離 pinned Round 1 raw evidence。"""
    tables = _formal_tables()
    baseline = tables["oof_predictions.csv"]["source_round"].eq("round1")
    tables["oof_predictions.csv"].loc[baseline, "predicted_ido_score"] += 0.25
    _synchronize_metrics_from_oof(tables)
    _refresh_ranking_tables(tables)
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="Round 1 baseline raw evidence|pinned baseline"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_smoke_validator_rejects_internally_consistent_noncanonical_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke split 與下游 fold 一起重新標號仍須偏離 authority metadata。"""
    tables, payloads, context = _smoke_bundle_inputs()
    validation = "leave_one_b_out"
    folds = sorted(
        tables["outer_splits.csv"].loc[
            tables["outer_splits.csv"]["validation"].eq(validation), "fold"
        ].astype(str).unique()
    )
    first, second = folds[:2]
    fold_map = {first: second, second: first}
    split_map = {
        f"{validation}:{first}": f"{validation}:{second}",
        f"{validation}:{second}": f"{validation}:{first}",
    }
    for name in (
        "outer_splits.csv",
        "fold_metrics.csv",
        "oof_predictions.csv",
        "hyperparameters.csv",
    ):
        frame = tables[name].copy()
        selected = frame["validation"].eq(validation)
        values = frame.loc[selected, "fold"].astype(str)
        frame.loc[selected, "fold"] = values.map(fold_map).fillna(values)
        if "split_id" in frame:
            split_ids = frame.loc[selected, "split_id"].astype(str)
            frame.loc[selected, "split_id"] = split_ids.map(split_map).fillna(split_ids)
        tables[name] = frame
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch) / "smoke",
        tables,
        payloads,
        context,
    )

    with pytest.raises(ValueError, match="canonical diagnostic|authority.*split"):
        validate_round2_bundle(generation.staging_dir, smoke=True)


def test_round2_validator_rejects_coordinated_roster_repin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance 與兩份 config 同步 repin 仍不得偏離 frozen roster。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    forged = "c" * 64
    provenance_path = generation.staging_dir / "baseline_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["roster_sha256"] = forged
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_path = generation.staging_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for config_name in ("original_config", "effective_config"):
        metadata[config_name]["round1"]["roster_sha256"] = forged
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rehash(generation.staging_dir, "baseline_provenance.json")
    _rehash(generation.staging_dir, "run_metadata.json")

    with pytest.raises(ValueError, match="frozen roster|roster.*authority"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


def test_relative_round1_dir_is_resolved_from_fixed_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone validator 從非 repo CWD 執行仍正確解析凍結相對 root。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    metadata_path = generation.staging_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for config_name in ("original_config", "effective_config"):
        metadata[config_name]["round1"]["dir"] = "immunity/outputs/exp3"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rehash(generation.staging_dir, "run_metadata.json")
    elsewhere = tmp_path / "different-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    validate_round2_bundle(generation.staging_dir, smoke=False)


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
    tables, payloads, context = _smoke_bundle_inputs()
    output = _allowed_output(tmp_path, monkeypatch)
    generation = _write_bundle(output / "smoke", tables, payloads, context)

    validate_round2_bundle(generation.staging_dir, smoke=True)

    eligibility = pd.read_csv(generation.staging_dir / "eligibility.csv")
    eligibility.loc[eligibility.index[0], "recommended"] = True
    eligibility.to_csv(generation.staging_dir / "eligibility.csv", index=False)
    _rehash(generation.staging_dir, "eligibility.csv")
    with pytest.raises(ValueError, match="smoke.*recommendation|recommendation.*smoke"):
        validate_round2_bundle(generation.staging_dir, smoke=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "split_missing_image",
        "split_train_test_overlap",
        "family_test_twice",
        "oof_wrong_test_key",
        "success_missing_hyperparameter",
        "failed_missing_failure",
        "failure_for_success",
        "nonempty_dummy",
        "nonempty_comparison",
        "eligibility_wrong_identity",
        "eligibility_true",
    ],
)
def test_smoke_validator_reconciles_diagnostic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Smoke 只豁免 formal counts/ranking，不豁免 split/status/OOF 證據一致性。"""
    tables, payloads, context = _smoke_bundle_inputs()
    if mutation == "split_missing_image":
        split = tables["outer_splits.csv"]
        first_fold = str(
            split.loc[
                split["validation"].eq("leave_one_b_out"), "fold"
            ].iloc[0]
        )
        tables["outer_splits.csv"] = split.drop(
            split.loc[
                split["validation"].eq("leave_one_b_out")
                & split["fold"].astype(str).eq(first_fold)
            ].index[0]
        )
    elif mutation == "split_train_test_overlap":
        duplicate = tables["outer_splits.csv"].iloc[[0]].copy()
        duplicate["role"] = "test" if duplicate.iloc[0]["role"] == "train" else "train"
        tables["outer_splits.csv"] = pd.concat(
            [tables["outer_splits.csv"], duplicate], ignore_index=True
        )
    elif mutation == "family_test_twice":
        split = tables["outer_splits.csv"]
        image_key = "image-000"
        row = split["validation"].eq("leave_one_b_out") & split[
            "image_key"
        ].eq(image_key)
        split.loc[row, "role"] = "test"
    elif mutation == "oof_wrong_test_key":
        tables["oof_predictions.csv"].loc[0, "image_key"] = "image-007"
    elif mutation == "success_missing_hyperparameter":
        tables["hyperparameters.csv"] = tables["hyperparameters.csv"].iloc[1:]
    elif mutation == "failed_missing_failure":
        failed = tables["fold_metrics.csv"].index[0]
        identity = tables["fold_metrics.csv"].loc[failed]
        tables["fold_metrics.csv"].loc[failed, "status"] = "failed"
        tables["fold_metrics.csv"].loc[
            failed, ["mae", "rmse", "r2", "spearman"]
        ] = np.nan
        for name in ("oof_predictions.csv", "hyperparameters.csv"):
            frame = tables[name]
            tables[name] = frame.loc[
                ~(
                    frame["configuration_id"].eq(identity["configuration_id"])
                    & frame["split_id"].eq(identity["split_id"])
                )
            ]
    elif mutation == "failure_for_success":
        identity = tables["fold_metrics.csv"].iloc[0]
        tables["model_failures.csv"] = pd.DataFrame(
            [
                {
                    **{
                        field: identity[field]
                        for field in (
                            "validation",
                            "fold",
                            "split_id",
                            "model",
                            "feature_set",
                            "source_round",
                            "configuration_id",
                        )
                    },
                    "exception_type": "ValueError",
                    "message": "forged",
                }
            ]
        )
    elif mutation == "nonempty_dummy":
        tables["dummy_fold_metrics.csv"] = _formal_tables()[
            "dummy_fold_metrics.csv"
        ].iloc[[0]]
    elif mutation == "nonempty_comparison":
        tables["feature_set_comparison.csv"] = tables["eligibility.csv"].copy()
    elif mutation == "eligibility_wrong_identity":
        tables["eligibility.csv"].loc[0, "configuration_id"] = "rogue__paper93"
    else:
        tables["eligibility.csv"].loc[0, "eligible"] = True
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch) / "smoke",
        tables,
        payloads,
        context,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Smoke|smoke|split|membership|OOF|hyperparameter|failure|failed|"
            "Dummy|comparison|eligibility|identity"
        ),
    ):
        validate_round2_bundle(generation.staging_dir, smoke=True)


@pytest.mark.parametrize(
    "flag",
    ["diagnostic_splits", "non_formal", "no_scientific_conclusion"],
)
def test_smoke_validator_requires_all_non_scientific_metadata_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    """Smoke metadata 必須逐一明示 diagnostic、non-formal 與無科學結論。"""
    tables, payloads, context = _smoke_bundle_inputs()
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch) / "smoke",
        tables,
        payloads,
        context,
    )
    path = generation.staging_dir / "run_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[flag] = False
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rehash(generation.staging_dir, "run_metadata.json")

    with pytest.raises(ValueError, match="Smoke|smoke|diagnostic|non.formal|科學"):
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
    "mutation",
    [
        "missing_reproducibility",
        "bad_commit",
        "empty_branch",
        "dirty_mismatch",
        "missing_package",
        "non_utc_start",
        "completion_before_start",
        "boolean_runtime",
        "duration_mismatch",
    ],
)
def test_round2_validator_rejects_reproducibility_metadata_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Standalone validator 必須獨立拒絕 provenance schema 與 timing drift。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    path = generation.staging_dir / "run_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    reproducibility = payload["reproducibility"]
    if mutation == "missing_reproducibility":
        payload.pop("reproducibility")
    elif mutation == "bad_commit":
        reproducibility["git"]["commit"] = "0" * 39
    elif mutation == "empty_branch":
        reproducibility["git"]["branch"] = ""
    elif mutation == "dirty_mismatch":
        reproducibility["git"]["dirty"] = False
    elif mutation == "missing_package":
        reproducibility["packages"].pop("OpenCV")
    elif mutation == "non_utc_start":
        reproducibility["timing"]["started_at_utc"] = "2026-08-11T21:09:49.867240+08:00"
    elif mutation == "completion_before_start":
        reproducibility["timing"]["completed_at_utc"] = "2026-08-11T13:09:48.000000Z"
    elif mutation == "boolean_runtime":
        reproducibility["timing"]["runtime_seconds"] = True
    else:
        reproducibility["timing"]["runtime_seconds"] = 99.0
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rehash(generation.staging_dir, "run_metadata.json")

    with pytest.raises(ValueError, match="reproducibility|Git|git|package|UTC|runtime|duration"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "nonnumeric_predictor",
        "infinite_target",
        "missing_mask_image",
        "failed_mask_status",
        "missing_valid_identity",
        "fractional_valid_count",
        "boolean_valid_count",
        "negative_valid_count",
        "valid_minimum_drift",
        "valid_status_drift",
        "fractional_extraction_count",
        "boolean_extraction_count",
        "negative_extraction_count",
        "extraction_roster_mismatch",
        "aggregation_unit_drift",
        "outside_threshold_violation",
        "metadata_total_drift",
        "baseline_read_write",
        "baseline_hash_drift",
    ],
)
def test_round2_validator_rejects_independent_data_qc_and_provenance_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Standalone validator 必須由 raw tables 重算，不得相信 passed/summary。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    directory = generation.staging_dir
    if mutation in {"nonnumeric_predictor", "infinite_target"}:
        name = "data_snapshot.csv"
        frame = pd.read_csv(directory / name)
        field = PAPER_STYLE_FOV_FEATURES[0] if mutation == "nonnumeric_predictor" else "IDO_score"
        if mutation == "nonnumeric_predictor":
            frame[field] = frame[field].astype(object)
            frame.loc[0, field] = "not-numeric"
        else:
            frame.loc[0, field] = np.inf
        frame.to_csv(directory / name, index=False)
    elif mutation in {"missing_mask_image", "failed_mask_status"}:
        name = "mask_provenance_qc.csv"
        frame = pd.read_csv(directory / name)
        if mutation == "missing_mask_image":
            frame = frame.iloc[1:]
        else:
            frame.loc[0, "status"] = "passed"  # summaries仍宣稱 passed
            frame.loc[1, "status"] = "failed"
        frame.to_csv(directory / name, index=False)
    elif mutation.startswith("baseline_"):
        name = "baseline_provenance.json"
        payload = json.loads((directory / name).read_text(encoding="utf-8"))
        if mutation == "baseline_read_write":
            payload["read_only"] = False
        else:
            payload["artifact_sha256"]["source.csv"] = "c" * 64
        (directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif mutation == "metadata_total_drift":
        name = "run_metadata.json"
        payload = json.loads((directory / name).read_text(encoding="utf-8"))
        payload["pair_mapping_summary"]["pair_observation_count"] += 1
        (directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif "valid" in mutation:
        name = "feature_valid_counts.csv"
        frame = pd.read_csv(directory / name)
        if mutation == "missing_valid_identity":
            frame = frame.iloc[1:]
        elif mutation == "fractional_valid_count":
            frame["roster_cell_count"] = frame["roster_cell_count"].astype(object)
            frame.loc[0, "roster_cell_count"] = 35.5
        elif mutation == "boolean_valid_count":
            frame["finite_cell_count"] = frame["finite_cell_count"].astype(object)
            frame.loc[0, "finite_cell_count"] = True
        elif mutation == "negative_valid_count":
            frame.loc[0, "finite_cell_count"] = -1
        elif mutation == "valid_minimum_drift":
            frame.loc[0, "required_minimum"] = 2
        else:
            frame.loc[0, "status"] = "failed"
        frame.to_csv(directory / name, index=False)
    else:
        name = "extraction_qc.csv"
        frame = pd.read_csv(directory / name)
        if mutation == "fractional_extraction_count":
            frame["extracted_pair_count"] = frame["extracted_pair_count"].astype(object)
            frame.loc[0, "extracted_pair_count"] = 35.5
        elif mutation == "boolean_extraction_count":
            frame["max_nuclei_per_cell"] = frame["max_nuclei_per_cell"].astype(object)
            frame.loc[0, "max_nuclei_per_cell"] = True
        elif mutation == "negative_extraction_count":
            frame.loc[0, "retained_outside_pair_count"] = -1
        elif mutation == "extraction_roster_mismatch":
            frame.loc[0, "extracted_pair_count"] -= 1
        elif mutation == "aggregation_unit_drift":
            frame.loc[0, "aggregation_unit"] = "unique_cell"
        else:
            frame.loc[0, "max_retained_outside_fraction"] = 0.051
        frame.to_csv(directory / name, index=False)
    _rehash(directory, name)

    with pytest.raises(
        ValueError,
        match=(
            "data_snapshot|numeric|finite|mask|valid|count|integer|QC|aggregation|"
            "outside|metadata|pair|baseline|provenance|read_only|hash"
        ),
    ):
        validate_round2_bundle(directory, smoke=False)


@pytest.mark.parametrize(
    ("table_name", "field", "value"),
    [
        ("oof_predictions.csv", "predicted_ido_score", np.inf),
        ("oof_predictions.csv", "observed_ido_score", "not-numeric"),
        ("oof_predictions.csv", "observed_ido_score", 999.0),
        ("dummy_oof_predictions.csv", "predicted_ido_score", np.inf),
        ("dummy_oof_predictions.csv", "observed_ido_score", "not-numeric"),
        ("dummy_oof_predictions.csv", "observed_ido_score", 999.0),
    ],
)
def test_round2_validator_rejects_oof_numeric_and_observed_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    table_name: str,
    field: str,
    value: object,
) -> None:
    """Candidate 與 Dummy OOF 必須 finite，observed 必須重現 data target。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    path = generation.staging_dir / table_name
    frame = pd.read_csv(path)
    if isinstance(value, str):
        frame[field] = frame[field].astype(object)
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    _rehash(generation.staging_dir, table_name)

    with pytest.raises(ValueError, match="OOF|observed|predicted|numeric|finite|target"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


@pytest.mark.parametrize(
    ("metric_table", "metric"),
    [
        ("fold_metrics.csv", "mae"),
        ("fold_metrics.csv", "rmse"),
        ("fold_metrics.csv", "r2"),
        ("fold_metrics.csv", "spearman"),
        ("dummy_fold_metrics.csv", "mae"),
        ("dummy_fold_metrics.csv", "rmse"),
        ("dummy_fold_metrics.csv", "r2"),
        ("dummy_fold_metrics.csv", "spearman"),
    ],
)
def test_round2_validator_recomputes_each_successful_fold_metric_from_oof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_table: str,
    metric: str,
) -> None:
    """Rehashed 且 ranking 自洽的假 fold metric 仍須被 raw OOF 重算擋下。"""
    tables = _formal_tables()
    original = tables[metric_table].loc[0, metric]
    tables[metric_table].loc[0, metric] = (
        0.125 if pd.isna(original) else float(original) + 0.125
    )
    _refresh_ranking_tables(tables)
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        tables,
        _formal_json_payloads(),
        _formal_record_context(),
    )

    with pytest.raises(ValueError, match="fold.*metric|metric.*OOF|recomputed|重算"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


@pytest.mark.parametrize("table_name", ["fold_metrics.csv", "dummy_fold_metrics.csv"])
@pytest.mark.parametrize("bad_kind", ["fractional", "boolean", "negative"])
def test_round2_validator_rejects_non_strict_fold_n_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    table_name: str,
    bad_kind: str,
) -> None:
    """Candidate 與 Dummy fold n_test 都不得以 int() 截斷或接受 bool。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    path = generation.staging_dir / table_name
    frame = pd.read_csv(path)
    if bad_kind == "fractional":
        frame["n_test"] = frame["n_test"].astype(object)
        frame.loc[0, "n_test"] = float(frame.loc[0, "n_test"]) + 0.5
    elif bad_kind == "boolean":
        frame["n_test"] = frame["n_test"].astype(object)
        frame.loc[0, "n_test"] = True
    else:
        frame.loc[0, "n_test"] = -1
    frame.to_csv(path, index=False)
    _rehash(generation.staging_dir, table_name)

    with pytest.raises(ValueError, match="n_test|integer|membership"):
        validate_round2_bundle(generation.staging_dir, smoke=False)


@pytest.mark.parametrize(
    "table_name", ["feature_set_comparison.csv", "eligibility.csv"]
)
@pytest.mark.parametrize(
    "field",
    [
        "leave_one_b_out_oof_mae",
        "leave_one_b_out_rank",
        "average_rank",
        "worst_validation_rank",
        "mae_beats_dummy_gate",
        "in_tie_band",
        "eliminated_by_simplicity",
    ],
)
def test_round2_validator_rejects_rehashed_derived_ranking_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    table_name: str,
    field: str,
) -> None:
    """兩份 ranking 的 pooled MAE、min ranks、gates 與 33-feature elimination 都須重算。"""
    generation = _write_bundle(
        _allowed_output(tmp_path, monkeypatch),
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )
    path = generation.staging_dir / table_name
    frame = pd.read_csv(path)
    if pd.api.types.is_bool_dtype(frame[field]):
        frame.loc[0, field] = not bool(frame.loc[0, field])
    else:
        frame.loc[0, field] = float(frame.loc[0, field]) + 0.125
    frame.to_csv(path, index=False)
    _rehash(generation.staging_dir, table_name)

    with pytest.raises(ValueError, match="ranking|derived|gate|rank|comparison|eligibility"):
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
    assert "23,976 frozen nucleus–cell pair observations" in text
    assert "重複 cell label" in text
    assert "依 nucleus 數量加權 whole-cell descriptors" in text
    assert "其他 nuclei 仍保留在 pair-specific cytoplasm" in text
    assert "相同 frozen pair roster" in text
    predictor_section = text.split("## Predictor 範圍", maxsplit=1)[1].split("##", maxsplit=1)[0]
    assert (
        "donor label/identifier、IFN/TNF dose、condition、IDO channel 或 "
        "`IDO_score` target"
    ) in predictor_section
    assert "IDO 僅作 target" in predictor_section


def test_round2_record_states_metric_roles_and_directions() -> None:
    """正式 record 必須明示 primary/secondary 指標與正確方向。"""
    text = build_round2_experiment_record(_formal_record_context())

    assert "MAE 是 primary（越低越好）" in text
    assert "RMSE、R²、Spearman 是 secondary" in text
    assert "RMSE 越低越好" in text
    assert "R² 與 Spearman 越高越好" in text


def test_round2_record_states_strict_tie_decision_rules() -> None:
    """正式 record 必須明示 strict boundary 與同 algorithm 的簡潔性規則。"""
    text = build_round2_experiment_record(_formal_record_context())
    decision_section = text.split(
        "## Eligibility gates 與 tie decision", maxsplit=1
    )[1].split("##", maxsplit=1)[0]

    assert "`average_rank - R* < 0.25`" in decision_section
    assert "等於 `0.25` 不算" in decision_section
    assert "同一 algorithm 的 33-feature 與 93-feature 都在 band" in decision_section
    assert "保留 33-feature、淘汰 93-feature" in decision_section


def test_smoke_record_starts_with_no_scientific_conclusion() -> None:
    """Smoke record 第一段必須明示不形成 33 vs 93 科學結論。"""
    context = _formal_record_context()
    context.update({"smoke": True, "recommendation": None})

    text = build_round2_experiment_record(context)

    first_paragraph = text.split("\n\n", maxsplit=1)[0]
    assert "僅驗證流程，不是正式實驗結果" in first_paragraph
    assert "不形成 33 vs 93 科學結論" in text
