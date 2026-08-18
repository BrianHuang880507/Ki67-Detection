"""Exp4 Rui 2025 replication 的安全 CLI 骨架。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from time import perf_counter
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .border_exclusion import (
    BorderExclusionError,
    BorderExclusionResult,
    apply_border_exclusion,
    write_border_exclusion_report,
    write_cell_dedup_report,
    write_group_targets,
    write_group_target_sensitivity,
    write_fov_ido_scores,
    build_group_target_sensitivity_result,
    build_target_aggregation,
)
from .cell_dedup import NUCLEUS_FEATURE_COLUMNS, assert_full_rui49_feature_consistency
from .cv import (
    CvPopulationContract,
    LeakageContext,
    RUI48_FEATURE_COLUMNS,
    assemble_feature_arms,
    run_rui_cv,
)
from .data_snapshot import SnapshotExpectation, SnapshotResult, validate_data_snapshot
from .feature_health import evaluate_feature_health
from .full_rui49 import (
    FullRui49CanonicalBlockedError,
    RUI49_FEATURE_COLUMNS,
    extract_full_rui49,
)
from .nucleus_sanity import build_nucleus_sanity_report
from .exploratory import (
    CLUSTER_FEATURE_REPORT_COLUMNS,
    CONDITION_WARNING_TEXT,
    EMBEDDING_COLUMNS,
    run_exploratory_analysis,
)
from .post_cv_analysis import (
    ORIENTATION_BIN_COLUMNS,
    ORIENTATION_SUMMARY_COLUMNS,
    SHRINKAGE_REPORT_COLUMNS,
    analyze_orientation_marginal,
    analyze_shrinkage,
)
from .phase2 import (
    FrozenPhase1Evidence,
    PHASE2_MANIFEST_COLUMNS,
    Phase2LeakageContext,
    assemble_phase2_cv,
    phase2_cv_source_components,
    reconstruct_phase2_population,
    run_phase2_cv,
)
from .phase2_reporting import (
    PHASE2_FIGURE_FIELDS,
    PHASE2_TABLE_FIELDS,
    PHASE2_NUMERIC_PATH_FIELDS,
    build_phase2_report_section,
    capture_phase1_immutable_state,
    derive_phase2_tables,
    phase2_output_paths,
    phase2_publish_paths,
    publish_phase2_bundle,
    render_phase2_figures,
    replace_phase2_report_section,
    sha256_file,
    validate_phase1_hash_lock,
    validate_phase2_publish_mapping,
    verify_phase1_immutable_state,
    verify_phase2_numeric_artifact_immutability,
    write_phase2_core_tables,
    write_phase2_dose_tables,
    write_phase2_log,
    write_phase2_metadata,
)
from .phase2_diagnostics import (
    CONDITION_MAPPING_REVERSED_HYPOTHESIS,
    DOSE_MAPPING_NOTE,
    analyze_residual_vs_dose,
    build_dose_response_check,
    fingerprint_phase2_numeric_artifacts,
)
from .redundancy import AUTHORITATIVE_RUI28_FEATURES, analyze_feature_redundancy
from .reporting import write_report_skeleton


PACKAGE_NAMES = (
    "umap-learn",
    "mahotas",
    "scikit-image",
    "scikit-learn",
    "pandas",
    "scipy",
    "pyyaml",
    "joblib",
    "cellpose",
)


FERET_DEVIATION_ID = "feret_caliper"
FERET_DEVIATION = {
    "id": FERET_DEVIATION_ID,
    "original_behavior": (
        "沿用既有 cell__feret_width；其來源為 cv2.fitEllipse 的 minor-axis proxy"
    ),
    "new_behavior": (
        "以 common pixel-center convex-hull caliper 計算 MinFeret/MaxFeret，"
        "並以 skimage.measure.regionprops 的二階矩計算 eccentricity"
    ),
    "reason": (
        "舊 proxy 可能產生 MinFeret > MaxFeret 且極端失真；新計算以 "
        "0 <= MinFeret <= MaxFeret invariant fail-closed"
    ),
    "scope": "whole-cell morphology；本實驗全部四個 arm 共用",
}


FEATURE_ARM_SPECS = {
    "geometry_24": {
        "count": 24,
        "role": "Rui exterior morphology only",
        "nucleus_allowed": False,
    },
    "rui_48": {
        "count": 48,
        "role": "49 canonical whole-cell features minus cell__MinIntensity",
        "nucleus_allowed": False,
    },
    "rui_filtered": {
        "count": "pending_redundancy",
        "role": "rui_48 after Pearson |r| > 0.9 X-only filtering",
        "nucleus_allowed": False,
    },
    "rui_48_plus_nucleus": {
        "count": 65,
        "role": "rui_48 plus 17 aggregated nucleus features",
        "nucleus_allowed": True,
    },
}


E2_V5_STATUS = "canceled_v5_no_dapi_acquired"
E2_V5_REASON = (
    "v5：九個 Exp4 dataset 從未取得 DAPI；資料目錄只含 IDO/PC，"
    "因此 DAPI nucleus validation 不可能執行。"
)
E2_V5_PROVENANCE = {
    "dataset_count": 9,
    "fov_count": 693,
    "channels_observed": ("IDO", "PC"),
    "dapi_acquired": False,
    "source": "v5 data inventory／plan revision",
}


FORMAL_CV_LOCKS = {
    "raw_pair_rows": 23_976,
    "pre_border_cells": 23_012,
    "retained_cells": 19_648,
    "retained_raw_pair_rows": 20_440,
    "fov_count": 693,
    "group_count": 9,
    "duplicate_key_count": 938,
}

CV_ANALYSIS_FILE_FIELDS = (
    "feature_health_report_path",
    "nucleus_sanity_report_path",
    "feature_redundancy_report_path",
    "leakage_preflight_report_path",
    "cv_metrics_path",
    "oof_predictions_path",
    "hyperparameters_path",
    "hyperparameter_grids_path",
    "feature_importance_path",
    "per_group_residuals_path",
    "report_path",
    "metadata_path",
)

POST_CV_ARTIFACT_FIELDS = (
    "shrinkage_analysis_path",
    "orientation_target_marginal_path",
    "umap_embeddings_path",
    "kmeans_cluster_features_path",
    "umap_by_donor_path",
    "umap_by_passage_path",
    "umap_by_condition_path",
)
POST_CV_ANALYSIS_FILE_FIELDS = POST_CV_ARTIFACT_FIELDS + (
    "report_path",
    "metadata_path",
)

PROTECTED_CV_ARTIFACT_FILES = (
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
)


@dataclass(frozen=True)
class CvAnalysisInputs:
    """保存 formal CV 前置階段產生的同世代資料與證據。"""

    retained_cells: pd.DataFrame
    health_result: Any
    health_report: pd.DataFrame
    nucleus_sanity_report: pd.DataFrame
    redundancy_result: Any
    assembly: Any
    source_fingerprint: str
    raw_consistency: Any
    retained_consistency: Any
    raw_pair_row_count: int
    retained_raw_pair_row_count: int
    retained_fov_count: int
    mask_cache_fingerprint_before: str
    mask_cache_fingerprint_after: str


class BundlePublishError(ValueError):
    """表示 Exp4 canonical output bundle 無法完成可回復發布。"""


def build_parser() -> argparse.ArgumentParser:
    """建立 Exp4 CLI parser。"""
    parser = argparse.ArgumentParser(
        description="Rui-style feature replication 的 snapshot/border safety modes"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Exp4 YAML 設定檔"
    )
    parser.add_argument(
        "--validate-snapshot",
        action="store_true",
        help="只執行唯讀資料快照驗證，不啟動後續分析",
    )
    parser.add_argument(
        "--measure-border-exclusion",
        action="store_true",
        help="唯讀量測觸邊排除，輸出 report/targets；不啟動特徵或 Cellpose",
    )
    parser.add_argument(
        "--run-safe",
        "--run-exp4",
        dest="run_safe",
        action="store_true",
        help="安全執行 raw CSV → dedup → border → target/sensitivity；不重跑 source/mask",
    )
    parser.add_argument(
        "--extract-rui49",
        action="store_true",
        help="執行 full 693-FOV Rui49 extraction；完成 canonical49/health 後停止",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help=(
            "執行 formal 19,648-cell analysis：65 欄 health、E3、X-only "
            "redundancy 與 4-arm×6-model CV"
        ),
    )
    parser.add_argument(
        "--run-exploratory",
        action="store_true",
        help=(
            "讀取 validated formal CV generation，執行 post-CV shrinkage、"
            "Orientation marginal、UMAP 與 putative-IFN k-means"
        ),
    )
    parser.add_argument(
        "--run-phase2",
        action="store_true",
        help=(
            "讀取 frozen Phase 1 artifacts，執行 reviewed 72-group Phase 2 "
            "CV adapter 與 restricted publication"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="可選的 run_metadata.json 路徑",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """執行 Exp4 snapshot 或 border-exclusion 的安全唯讀 CLI。"""
    args = build_parser().parse_args(argv)
    selected_modes = sum(
        bool(value)
        for value in (
            args.validate_snapshot,
            args.measure_border_exclusion,
            args.run_safe,
            args.extract_rui49,
            args.run_cv,
            args.run_exploratory,
            args.run_phase2,
        )
    )
    if selected_modes > 1:
        print(
            "STATUS=blocked: --validate-snapshot、--measure-border-exclusion、"
            "--run-safe、--extract-rui49、--run-cv、--run-exploratory、"
            "--run-phase2 mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if selected_modes == 0:
        print(
            "STATUS=blocked: 請明確使用 --validate-snapshot 或 "
            "--measure-border-exclusion、--run-safe、--run-cv 或 --run-phase2。",
            file=sys.stderr,
        )
        return 2

    if args.measure_border_exclusion:
        return _run_border_measurement(args)
    if args.run_safe:
        return _run_safe_pipeline(args)
    if args.extract_rui49:
        return _run_full_rui49(args)
    if args.run_cv:
        return _run_cv_analysis(args)
    if args.run_exploratory:
        return _run_exploratory_analysis(args)
    if args.run_phase2:
        return _run_phase2_analysis(args)

    try:
        config = _load_config(args.config)
        source_root = _resolve_config_path(config, "source_root", Path.cwd())
        manifest_path = _resolve_config_path(config, "manifest_path", source_root)
        cell_level_path = _resolve_config_path(config, "cell_level_path", source_root)
        masks_dir = _resolve_config_path(config, "masks_dir", source_root)
        expectation = SnapshotExpectation.from_mapping(config.get("expected"))
        metadata_path = args.metadata or _metadata_path(config)
        _ensure_output_outside_source_root(metadata_path, source_root, "metadata_path")
        result = validate_data_snapshot(
            source_root=source_root,
            manifest_path=manifest_path,
            cell_level_path=cell_level_path,
            masks_dir=masks_dir,
            expected=expectation,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"STATUS=blocked: cannot validate snapshot: {error}", file=sys.stderr)
        return 2

    try:
        _write_metadata(
            metadata_path,
            status="snapshot_validated" if result.matched else "blocked",
            snapshot=result,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"STATUS=blocked: cannot write metadata: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if not result.matched:
        print(
            "STATUS=blocked: snapshot mismatch; no feature work was started.",
            file=sys.stderr,
        )
        return 2
    return 0




def _run_border_measurement(args: argparse.Namespace) -> int:
    """執行 border diagnostic；絕不發布 canonical report、targets 或 metadata。"""
    result: BorderExclusionResult | None = None
    try:
        config = _load_config(args.config)
        source_root = _resolve_config_path(config, "source_root", Path.cwd())
        manifest_path = _resolve_config_path(config, "manifest_path", source_root)
        cell_level_path = _resolve_config_path(config, "cell_level_path", source_root)
        masks_dir = _resolve_config_path(config, "masks_dir", source_root)
        expectation = SnapshotExpectation.from_mapping(config.get("expected"))
        expected_dedup = _optional_expected_count(config, "dedup_cell_count")
        expected_duplicates = _optional_expected_count(config, "duplicate_key_count")
        expected_post = _optional_expected_count(config, "post_border_cell_count")
        expected_sparse = _optional_expected_sparse_fov_keys(config)
        manifest = pd.read_csv(manifest_path)
        cells = pd.read_csv(cell_level_path)
        result = apply_border_exclusion(
            manifest,
            cells,
            masks_dir,
            expected_manifest_keys=expectation.manifest_image_key_count,
            expected_cell_rows=expectation.cell_row_count,
            expected_cell_keys=expectation.cell_image_key_count,
            expected_dedup_cells=expected_dedup,
            expected_duplicate_keys=expected_duplicates,
            expected_sparse_fov_keys=expected_sparse,
            expected_group_keys=expectation.group_keys,
        )
        summary = _border_summary(
            result,
            report_path=None,
            targets_path=None,
            metadata_path=None,
            status="diagnostic",
        )
        summary.update(
            {
                "diagnostic_only": True,
                "target_status": "not_run",
                "output_paths": {},
            }
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, BorderExclusionError, yaml.YAMLError) as error:
        summary: dict[str, Any] = {
            "mode": "measure-border-exclusion",
            "status": "blocked",
            "diagnostic_only": True,
            "target_status": "not_run",
            "error": str(error),
        }
        if result is not None:
            summary.update(
                _border_summary(
                    result,
                    report_path=None,
                    targets_path=None,
                    metadata_path=None,
                    status="blocked",
                )
            )
            summary["target_status"] = "not_run"
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(
            f"STATUS=blocked: border diagnostic failed: {error}",
            file=sys.stderr,
        )
        return 2



def _run_safe_pipeline(args: argparse.Namespace) -> int:
    """執行 raw basic → high dedup → border → high target/sensitivity。

    所有 canonical outputs 先在同一 generation staging directory 產生；publisher
    只在 bundle 完整後備份舊檔並一次提交，sensitivity evidence 先寫入 staging
    再執行 high result 的 blocked gate。

    Args:
        args: 已由 CLI parser 建立的設定檔、metadata override 與 safe mode 參數。

    Returns:
        ``0`` 表示同一 generation 的 canonical bundle 已 validated 發布；``2``
        表示 snapshot、分析 gate、sensitivity 或 bundle transaction blocked。

    Raises:
        不向 CLI 呼叫端傳播預期的資料或 I/O 例外；所有 failure evidence 會寫入
        noncanonical generation 檔並轉成 machine-readable blocked summary。
    """
    config: Mapping[str, Any] | None = None
    output_paths: dict[str, Path] = {}
    metadata_path: Path | None = None
    source_root: Path | None = None
    result: BorderExclusionResult | None = None
    targets: pd.DataFrame | None = None
    sensitivity: pd.DataFrame | None = None
    generation_id = uuid.uuid4().hex
    staging_root: Path | None = None
    old_metadata: dict[str, Any] = {}
    sensitivity_result = None
    try:
        config = _load_config(args.config)
        source_root = _resolve_config_path(config, "source_root", Path.cwd())
        manifest_path = _resolve_config_path(config, "manifest_path", source_root)
        cell_level_path = _resolve_config_path(config, "cell_level_path", source_root)
        masks_dir = _resolve_config_path(config, "masks_dir", source_root)
        expectation = SnapshotExpectation.from_mapping(config.get("expected"))
        expected_dedup = _expected_count(config, "dedup_cell_count", 23012)
        expected_post = _expected_count(config, "post_border_cell_count", 19648)
        expected_duplicates = _expected_count(config, "duplicate_key_count", 938)
        expected_sparse = _expected_sparse_fov_keys(config)
        output_paths = _safe_output_paths(config)
        metadata_path = args.metadata or output_paths["metadata_path"]
        output_paths["metadata_path"] = metadata_path
        _ensure_safe_output_paths(output_paths, source_root)
        old_metadata = _read_json_mapping(metadata_path)
        planned_archived = _planned_archived_outputs(output_paths, generation_id)
        output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
            strict=False
        )
        staging_root = output_root / f".staging.{generation_id}"
        staging_root.mkdir(parents=True, exist_ok=False)
        staged_paths: dict[str, Path] = {}
        for key, path in output_paths.items():
            relative = Path(path).resolve(strict=False).relative_to(output_root)
            staged = staging_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged_paths[key] = staged

        snapshot = validate_data_snapshot(
            source_root=source_root,
            manifest_path=manifest_path,
            cell_level_path=cell_level_path,
            masks_dir=masks_dir,
            expected=expectation,
        )
        if not snapshot.matched:
            raise BorderExclusionError(
                "snapshot gate failed: " + "; ".join(snapshot.differences)
            )
        manifest = pd.read_csv(manifest_path)
        raw_cells = pd.read_csv(cell_level_path)

        from .border_exclusion import _deduplicate_for_border

        deduplicated, dedup_report = _deduplicate_for_border(
            raw_cells,
            deduplicated_cells=None,
            dedup_report=None,
            expected_raw_rows=expectation.cell_row_count,
            expected_distinct_cells=expected_dedup,
            expected_multirow_keys=expected_duplicates,
        )
        write_cell_dedup_report(
            dedup_report,
            staged_paths["cell_dedup_report_path"],
            expected_duplicate_keys=expected_duplicates,
        )
        result = apply_border_exclusion(
            manifest,
            raw_cells,
            masks_dir,
            expected_manifest_keys=expectation.manifest_image_key_count,
            expected_cell_rows=expectation.cell_row_count,
            expected_cell_keys=expectation.cell_image_key_count,
            expected_dedup_cells=expected_dedup,
            expected_duplicate_keys=expected_duplicates,
            expected_sparse_fov_keys=expected_sparse,
            expected_group_keys=expectation.group_keys,
            deduplicated_cells=deduplicated,
            dedup_report=dedup_report,
        )
        if len(result.eligible_cells) != expected_post:
            raise BorderExclusionError(
                "post-border distinct whole-cell count expected "
                f"{expected_post}, actual {len(result.eligible_cells)}"
            )
        write_border_exclusion_report(
            result,
            staged_paths["border_exclusion_report_path"],
        )
        target_result = build_target_aggregation(
            result,
            expected_pre_border_cells=expected_dedup,
            expected_post_border_cells=expected_post,
            expected_fov_count=expectation.manifest_image_key_count,
        )
        targets = target_result.group_targets.copy()
        write_fov_ido_scores(
            target_result,
            staged_paths["fov_ido_scores_path"],
            expected_fov_count=expectation.manifest_image_key_count,
        )
        if len(expectation.group_keys) == 9:
            sensitivity_result = build_group_target_sensitivity_result(
                result,
                target_result,
            )
            sensitivity = sensitivity_result.report.copy()
            write_group_target_sensitivity(
                sensitivity,
                staged_paths["group_target_sensitivity_path"],
                expected_group_keys=expectation.group_keys,
            )
        else:
            # Small integration fixtures exercise raw-basic → high target only;
            # formal sparse sensitivity is a nine-group production gate.
            sensitivity_result = None
            sensitivity = None

        gate_error: Exception | None = None
        if sensitivity_result is not None:
            try:
                # The high report has already been written to this generation's
                # staging area, so blocked evidence exists before the gate raises.
                sensitivity_result.raise_if_blocked()
            except Exception as error:
                gate_error = error

        final_status = "blocked" if gate_error is not None else "validated"
        final_target_status = "blocked" if gate_error is not None else "validated"
        final_sensitivity_status = (
            "blocked"
            if gate_error is not None and sensitivity_result is not None
            else "validated"
            if sensitivity_result is not None
            else "not_run_unit_fixture"
        )
        final_metadata = _build_safe_metadata(
            old_metadata,
            result=result,
            output_paths=output_paths,
            archived=planned_archived,
            generation_id=generation_id,
            expected_locks={
                "raw_cell_rows": expectation.cell_row_count,
                "dedup_cells": expected_dedup,
                "post_border_cells": expected_post,
                "fov_count": expectation.manifest_image_key_count,
                "duplicate_keys": expected_duplicates,
                "sparse_fov_count": len(expected_sparse),
            },
            status=final_status,
            target_status=final_target_status,
            target_sensitivity_status=final_sensitivity_status,
        )
        if gate_error is None:
            write_group_targets(
                targets,
                staged_paths["group_targets_path"],
                expected_group_keys=expectation.group_keys,
            )
        write_report_skeleton(
            staged_paths["report_path"],
            targets=targets,
            sensitivity=sensitivity,
            border_report=result.border_report,
            metadata=final_metadata,
        )
        _write_json_atomic(final_metadata, staged_paths["metadata_path"])
        publish_fields = [
            "cell_dedup_report_path",
            "border_exclusion_report_path",
            "fov_ido_scores_path",
        ]
        if sensitivity_result is not None:
            publish_fields.append("group_target_sensitivity_path")
        if gate_error is None:
            publish_fields.append("group_targets_path")
        publish_fields.extend(("report_path", "metadata_path"))
        _publish_bundle(
            staged_paths,
            output_paths,
            fields=publish_fields,
            generation_id=generation_id,
        )
        summary = _safe_summary(
            result,
            metadata=final_metadata,
            output_paths=output_paths,
            status=final_status,
        )
        if gate_error is not None:
            summary["error"] = str(gate_error)
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            print(
                f"STATUS=blocked: sparse sensitivity gate failed: {gate_error}",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, BorderExclusionError, yaml.YAMLError) as error:
        previous_feature = str(old_metadata.get("feature_smoke_status", "not_run"))
        previous_extraction = str(
            old_metadata.get("feature_extraction_status", "not_run")
        )
        stale_feature = (
            "historical_stale"
            if {previous_feature, previous_extraction}
            & {"validated", "smoke_validated", "passed"}
            else "not_run"
        )
        diagnostic: dict[str, Any] = {
            "generation_id": generation_id,
            "status": "blocked",
            "target_status": "blocked",
            "target_sensitivity_status": "blocked",
            "feature_smoke_status": stale_feature,
            "feature_extraction_status": stale_feature,
            "source_write_status": "not_written",
            "diagnostic_error": str(error),
        }
        if result is not None:
            diagnostic.update(
                {
                    "raw_cell_count": len(result.raw_cells),
                    "dedup_cell_count": len(result.pre_cells),
                    "post_border_cell_count": len(result.eligible_cells),
                    "fov_count": len(result.manifest),
                }
            )
        failure_evidence_path: Path | None = None
        output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
            strict=False
        )
        if source_root is None or _path_is_outside_source_root(output_root, source_root):
            try:
                failure_evidence_path = _write_failed_generation_evidence(
                    output_root,
                    generation_id=generation_id,
                    diagnostic=diagnostic,
                )
            except (OSError, TypeError, ValueError):
                failure_evidence_path = None
        summary = _safe_summary(
            result,
            metadata=diagnostic,
            output_paths=output_paths,
            status="blocked",
        )
        summary["error"] = str(error)
        if failure_evidence_path is not None:
            summary["failure_evidence_path"] = str(failure_evidence_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(
            f"STATUS=blocked: safe Exp4 pipeline failed: {error}",
            file=sys.stderr,
        )
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)


def _run_full_rui49(args: argparse.Namespace) -> int:
    """執行 full Rui49 adapter，完成 canonical49/health 後立即停止。

    Args:
        args: parser 建立的 config 與 metadata override 參數。

    Returns:
        ``0`` 表示 full canonical bundle 已同 generation 發布；``2`` 表示
        snapshot、F0/border、full core、health 或 publish gate blocked。
    """
    generation_id = uuid.uuid4().hex
    config: Mapping[str, Any] | None = None
    source_root: Path | None = None
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    output_paths: dict[str, Path] = {}
    current_paths: dict[str, Path] = {}
    old_metadata: dict[str, Any] = {}
    staging_root: Path | None = None
    try:
        config = _load_config(args.config)
        source_root = _resolve_config_path(config, "source_root", Path.cwd())
        manifest_path = _resolve_config_path(config, "manifest_path", source_root)
        cell_level_path = _resolve_config_path(config, "cell_level_path", source_root)
        masks_dir = _resolve_config_path(config, "masks_dir", source_root)
        expectation = SnapshotExpectation.from_mapping(config.get("expected"))
        output_paths = _full_output_paths(config)
        output_paths["metadata_path"] = args.metadata or output_paths["metadata_path"]
        _ensure_full_output_paths(output_paths, source_root)
        current_paths = _safe_output_paths(config)
        old_metadata = _read_json_mapping(output_paths["metadata_path"])
        _validate_current_reporting_artifacts(
            metadata=old_metadata,
            current_paths=current_paths,
            expected_group_count=expectation.group_count,
            expected_fov_count=expectation.manifest_image_key_count,
        )

        staging_root = output_root / f".staging.rui49.{generation_id}"
        staging_root.mkdir(parents=True, exist_ok=False)
        staged_paths = _stage_full_paths(staging_root, output_paths)

        snapshot = validate_data_snapshot(
            source_root=source_root,
            manifest_path=manifest_path,
            cell_level_path=cell_level_path,
            masks_dir=masks_dir,
            expected=expectation,
        )
        if not snapshot.matched:
            raise ValueError("snapshot gate failed: " + "; ".join(snapshot.differences))

        manifest = _normalise_manifest_pc_paths(
            pd.read_csv(manifest_path), manifest_path
        )
        raw_cells = pd.read_csv(cell_level_path)
        expected_dedup = _expected_count(config, "dedup_cell_count", 23012)
        expected_post = _expected_count(config, "post_border_cell_count", 19648)
        expected_duplicates = _expected_count(config, "duplicate_key_count", 938)
        expected_sparse = _expected_sparse_fov_keys(config)
        expected_retained_pairs = _expected_count(
            config, "retained_raw_pair_rows", 20440
        )
        from .border_exclusion import _deduplicate_for_border

        deduplicated, dedup_report = _deduplicate_for_border(
            raw_cells,
            deduplicated_cells=None,
            dedup_report=None,
            expected_raw_rows=expectation.cell_row_count,
            expected_distinct_cells=expected_dedup,
            expected_multirow_keys=expected_duplicates,
        )
        border_result = apply_border_exclusion(
            manifest,
            raw_cells,
            masks_dir,
            expected_manifest_keys=expectation.manifest_image_key_count,
            expected_cell_rows=expectation.cell_row_count,
            expected_cell_keys=expectation.cell_image_key_count,
            expected_dedup_cells=expected_dedup,
            expected_duplicate_keys=expected_duplicates,
            expected_sparse_fov_keys=expected_sparse,
            expected_group_keys=expectation.group_keys,
            deduplicated_cells=deduplicated,
            dedup_report=dedup_report,
        )
        if len(border_result.eligible_cells) != expected_post:
            raise ValueError(
                "post-border distinct whole-cell count expected "
                f"{expected_post}, actual {len(border_result.eligible_cells)}"
            )

        full_result = extract_full_rui49(
            manifest=manifest,
            raw_cells=raw_cells,
            retained_cells=border_result.eligible_cells,
            masks_dir=masks_dir,
            checkpoint_dir=output_paths["rui49_checkpoint_dir"],
            progress_log_path=output_paths["run_log_path"],
            expected_fov_count=expectation.manifest_image_key_count,
            expected_raw_pair_rows=expectation.cell_row_count,
            expected_pre_border_cells=expected_dedup,
            expected_retained_cells=expected_post,
            expected_raw_multirow_keys=expected_duplicates,
            expected_retained_raw_pair_rows=expected_retained_pairs,
        )
        if full_result.health.blocked:
            return _finish_full_failure(
                output_root,
                generation_id=generation_id,
                error=ValueError("full retained feature health blocked"),
                full_result=full_result,
                output_paths=output_paths,
            )
        _validate_full_result(
            full_result,
            expected_raw_pair_rows=expectation.cell_row_count,
            expected_pre_border_cells=expected_dedup,
            expected_retained_cells=expected_post,
            expected_raw_multirow_keys=expected_duplicates,
            expected_retained_raw_pair_rows=expected_retained_pairs,
        )
        full_result.raw_pair_features.to_csv(
            staged_paths["cell_level_rui49_path"], index=False
        )
        full_result.health.write_report(staged_paths["feature_health_report_path"])
        planned_archived_outputs = _planned_archived_outputs(
            _full_canonical_output_paths(output_paths),
            generation_id,
        )
        metadata = _build_full_metadata(
            old_metadata,
            full_result=full_result,
            output_paths=output_paths,
            current_paths=current_paths,
            generation_id=generation_id,
            expected_pre_border_cells=expected_dedup,
            expected_retained_cells=expected_post,
            expected_fov_count=expectation.manifest_image_key_count,
            ido_background_qc=config.get("ido_background_qc"),
            planned_archived_outputs=planned_archived_outputs,
        )
        targets = _read_optional_csv(current_paths.get("group_targets_path"))
        sensitivity = _read_optional_csv(
            current_paths.get("group_target_sensitivity_path")
        )
        border_report = _read_optional_csv(
            current_paths.get("border_exclusion_report_path")
        )
        if border_report is None:
            border_report = border_result.border_report
        write_report_skeleton(
            staged_paths["report_path"],
            targets=targets,
            sensitivity=sensitivity,
            border_report=border_report,
            metadata=metadata,
        )
        _write_json_atomic(metadata, staged_paths["metadata_path"])
        canonical_paths = _full_canonical_output_paths(output_paths)
        _publish_bundle(
            staged_paths,
            canonical_paths,
            fields=(
                "cell_level_rui49_path",
                "feature_health_report_path",
                "report_path",
                "metadata_path",
            ),
            generation_id=generation_id,
        )
        summary = _full_summary(
            metadata,
            output_paths=output_paths,
            status="validated",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except FullRui49CanonicalBlockedError as error:
        return _finish_full_failure(
            output_root,
            generation_id=generation_id,
            error=error,
            output_paths=output_paths,
        )
    except Exception as error:
        return _finish_full_failure(
            output_root,
            generation_id=generation_id,
            error=error,
            output_paths=output_paths,
        )
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)


def _run_cv_analysis(args: argparse.Namespace) -> int:
    """執行 formal v5 E3、health、redundancy 與 nested CV analysis bundle。

    這條 CLI 只讀取既有 Rui49 CSV、cell master 與 Exp3 mask cache；所有產物
    先寫入同 generation staging directory，再由 ``publish_analysis_bundle``
    一次提交。任何前置 gate、core 或 publisher 失敗都保留既有 canonical
    analysis bundle，並只寫 noncanonical failure evidence。
    """
    generation_id = uuid.uuid4().hex
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    config: Mapping[str, Any] | None = None
    source_root: Path | None = None
    output_paths: dict[str, Path] = {}
    staging_root: Path | None = None
    old_metadata: dict[str, Any] = {}
    try:
        config = _load_config(args.config)
        source_root = _resolve_config_path(config, "source_root", Path.cwd())
        manifest_path = _resolve_config_path(config, "manifest_path", source_root)
        cell_level_path = _resolve_config_path(config, "cell_level_path", source_root)
        masks_dir = _resolve_config_path(config, "masks_dir", source_root)
        expectation = SnapshotExpectation.from_mapping(config.get("expected"))
        output_paths = _safe_output_paths(config)
        full_paths = _full_output_paths(config)
        output_paths.update(
            {
                field: full_paths[field]
                for field in ("cell_level_rui49_path", "feature_health_report_path")
            }
        )
        output_paths["metadata_path"] = args.metadata or output_paths["metadata_path"]
        _ensure_safe_output_paths(output_paths, source_root)
        old_metadata = _read_json_mapping(output_paths["metadata_path"])
        _validate_cv_config_locks(config, expectation)
        _validate_cv_current_generation(
            metadata=old_metadata,
            current_paths=output_paths,
            expected_group_count=FORMAL_CV_LOCKS["group_count"],
            expected_fov_count=FORMAL_CV_LOCKS["fov_count"],
        )

        snapshot = validate_data_snapshot(
            source_root=source_root,
            manifest_path=manifest_path,
            cell_level_path=cell_level_path,
            masks_dir=masks_dir,
            expected=expectation,
        )
        if not snapshot.matched:
            raise ValueError("snapshot gate failed: " + "; ".join(snapshot.differences))

        mask_before = _mask_cache_fingerprint(masks_dir)
        inputs = _prepare_cv_analysis_inputs(
            config=config,
            manifest_path=manifest_path,
            cell_level_path=cell_level_path,
            masks_dir=masks_dir,
            current_paths=output_paths,
            expectation=expectation,
            metadata=old_metadata,
            mask_cache_fingerprint_before=mask_before,
        )
        mask_after = _mask_cache_fingerprint(masks_dir)
        if mask_before != mask_after:
            raise ValueError(
                "mask cache fingerprint changed during CV input assembly; "
                "fail-closed without publishing"
            )
        if inputs.mask_cache_fingerprint_before != mask_before:
            raise ValueError("CV input mask fingerprint provenance mismatch")

        cv_n_jobs = _positive_or_nonzero_int(config.get("cv_n_jobs", 6), "cv_n_jobs")
        importance_repeats = _positive_int(
            config.get("importance_repeats", 3), "importance_repeats"
        )
        leakage_context = _build_cv_leakage_context(mask_before, mask_after)
        result = run_rui_cv(
            inputs.assembly,
            leakage_context,
            checkpoint_dir=output_paths["cv_checkpoint_dir"],
            source_fingerprint=inputs.source_fingerprint,
            n_jobs=cv_n_jobs,
            importance_repeats=importance_repeats,
        )
        mask_after_cv = _mask_cache_fingerprint(masks_dir)
        if mask_after_cv != mask_before:
            raise ValueError(
                "mask cache fingerprint changed during CV; fail-closed without publishing"
            )

        _validate_cv_result_contract(result)
        staging_root = output_root / f".staging.cv.{generation_id}"
        staging_root.mkdir(parents=True, exist_ok=False)
        staged_paths = _stage_cv_paths(staging_root, output_paths)
        _write_cv_tables(
            staged_paths,
            inputs=inputs,
            result=result,
        )
        targets = _read_optional_csv(output_paths.get("group_targets_path"))
        sensitivity = _read_optional_csv(
            output_paths.get("group_target_sensitivity_path")
        )
        border_report = _read_optional_csv(
            output_paths.get("border_exclusion_report_path")
        )
        metadata = _build_cv_metadata(
            old_metadata,
            inputs=inputs,
            result=result,
            output_paths=output_paths,
            generation_id=generation_id,
            cv_n_jobs=cv_n_jobs,
            importance_repeats=importance_repeats,
            mask_cache_fingerprint_before=mask_before,
            mask_cache_fingerprint_after=mask_after_cv,
        )
        write_report_skeleton(
            staged_paths["report_path"],
            targets=targets,
            sensitivity=sensitivity,
            border_report=border_report,
            metadata=metadata,
        )
        _write_json_atomic(metadata, staged_paths["metadata_path"])
        publish_output_paths = {
            field: output_paths[field] for field in CV_ANALYSIS_FILE_FIELDS
        }
        publish_staged_paths = {
            field: staged_paths[field] for field in CV_ANALYSIS_FILE_FIELDS
        }
        archived = publish_analysis_bundle(
            publish_staged_paths,
            publish_output_paths,
            fields=CV_ANALYSIS_FILE_FIELDS,
            generation_id=generation_id,
        )
        summary = _cv_summary(
            metadata,
            output_paths=output_paths,
            status="validated",
        )
        summary["archived_historical_outputs"] = archived
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        diagnostic = {
            "mode": "run-cv",
            "status": "blocked",
            "generation_id": generation_id,
            "source_write_status": "not_written",
            "error": str(error),
            "analysis_status": {
                "cv": "blocked",
                "nucleus_sanity": "blocked",
                "feature_redundancy": "blocked",
                "feature_health": "blocked",
                "umap": "not_run",
                "kmeans": "not_run",
            },
            "output_paths": {
                key: str(Path(path).resolve(strict=False))
                for key, path in output_paths.items()
            },
        }
        failure_evidence_path: Path | None = None
        try:
            if source_root is None or _path_is_outside_source_root(output_root, source_root):
                failure_evidence_path = _write_failed_generation_evidence(
                    output_root,
                    generation_id=generation_id,
                    diagnostic=diagnostic,
                )
        except (OSError, TypeError, ValueError):
            failure_evidence_path = None
        summary = _cv_summary(
            diagnostic,
            output_paths=output_paths,
            status="blocked",
        )
        summary["error"] = str(error)
        if failure_evidence_path is not None:
            summary["failure_evidence_path"] = str(failure_evidence_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"STATUS=blocked: formal CV analysis failed: {error}", file=sys.stderr)
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)


def _phase2_phase1_artifact_paths(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    phase1_output_root: Path,
) -> dict[str, Path]:
    """解析 Phase 2 需要重驗證的 25 個 Phase 1 artifact paths。"""
    safe = _safe_output_paths(config)
    full = _full_output_paths(config)
    manifest = _resolve_config_path(config, "manifest_path", source_root)
    paths: dict[str, Path] = {
        "border_exclusion_report.csv": safe["border_exclusion_report_path"],
        "cell_dedup_report.csv": safe["cell_dedup_report_path"],
        "cell_level_rui49.csv": full["cell_level_rui49_path"],
        "cv_metrics.csv": safe["cv_metrics_path"],
        "environment_packages.json": phase1_output_root / "environment_packages.json",
        "feature_health_report.csv": full["feature_health_report_path"],
        "feature_importance.csv": safe["feature_importance_path"],
        "feature_redundancy_report.csv": safe["feature_redundancy_report_path"],
        "feature_smoke_report.json": phase1_output_root / "feature_smoke_report.json",
        "figures/umap_by_condition.png": phase1_output_root / "figures" / "umap_by_condition.png",
        "figures/umap_by_donor.png": phase1_output_root / "figures" / "umap_by_donor.png",
        "figures/umap_by_passage.png": phase1_output_root / "figures" / "umap_by_passage.png",
        "fov_ido_scores.csv": safe["fov_ido_scores_path"],
        "group_target_sensitivity.csv": safe["group_target_sensitivity_path"],
        "group_targets.csv": safe["group_targets_path"],
        "hyperparameter_grids.json": safe["hyperparameter_grids_path"],
        "hyperparameters.csv": safe["hyperparameters_path"],
        "kmeans_cluster_features.csv": phase1_output_root / "kmeans_cluster_features.csv",
        "leakage_preflight_report.csv": safe["leakage_preflight_report_path"],
        "nucleus_sanity_report.csv": safe["nucleus_sanity_report_path"],
        "oof_predictions.csv": safe["oof_predictions_path"],
        "orientation_target_marginal.csv": phase1_output_root / "orientation_target_marginal.csv",
        "per_group_residuals.csv": safe["per_group_residuals_path"],
        "shrinkage_analysis.csv": phase1_output_root / "shrinkage_analysis.csv",
        "umap_embeddings.csv": phase1_output_root / "umap_embeddings.csv",
    }
    # The seven frozen-input files use their source/config paths, not metadata
    # claims, and are included in the same 25-artifact lock.
    paths["data_manifest.csv"] = manifest
    paths["cell_level_basic.csv"] = _resolve_config_path(
        config, "cell_level_path", source_root
    )
    configured = config.get("phase1_protected_artifact_paths")
    if configured is not None:
        if not isinstance(configured, Mapping):
            raise ValueError("phase1_protected_artifact_paths 必須是 mapping")
        for name, value in configured.items():
            paths[str(name)] = _resolve_config_path(
                {"path": value}, "path", Path.cwd()
            )
    return paths


def _phase2_consistency_payload(
    metadata: Mapping[str, Any], key: str
) -> Mapping[str, object]:
    """取得 metadata 內的 frozen Rui49 consistency evidence。"""
    full = metadata.get("full_rui49")
    if isinstance(full, Mapping) and isinstance(full.get(key), Mapping):
        return full[key]
    value = metadata.get(key)
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"Phase 1 metadata 缺少 {key} consistency evidence")


def _phase2_stage_paths(
    staging_root: Path,
    output_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """建立不含 resumable checkpoint cache 的 Phase 2 staging mapping。"""
    staged: dict[str, Path] = {}
    for index, (field, final) in enumerate(output_paths.items()):
        if field == "phase2_cv_checkpoint_dir":
            # This directory is a canonical resumable work cache.  It must be
            # passed directly to the core and never staged, published, or
            # archived with the restricted Phase 2 bundle.
            continue
        suffix = Path(final).suffix
        name = f"{index:02d}_{field}{suffix if suffix else ''}"
        candidate = staging_root / name
        candidate.parent.mkdir(parents=True, exist_ok=True)
        staged[field] = candidate
    return staged


def _read_phase2_manifest(path: str | Path) -> pd.DataFrame:
    """以 allowlist 讀 manifest 並明示恢復 canonical Phase 2 欄位順序。"""
    manifest = pd.read_csv(
        path,
        usecols=["image_key", "b_id", "passage", "condition_index"],
    )
    return manifest.loc[:, PHASE2_MANIFEST_COLUMNS]


def _phase2_checkpoint_status(metrics: pd.DataFrame) -> str:
    """從每-fold status 彙總 resumable checkpoint cache 狀態。"""
    if not isinstance(metrics, pd.DataFrame) or "checkpoint_status" not in metrics:
        raise ValueError("Phase 2 CV metrics 缺少 checkpoint_status")
    statuses = set(metrics["checkpoint_status"].astype(str))
    if not statuses or not statuses.issubset({"computed", "resumed"}):
        raise ValueError(f"Phase 2 checkpoint_status 不合法：{sorted(statuses)}")
    if statuses == {"resumed"}:
        return "resumed"
    if statuses == {"computed"}:
        return "computed"
    return "partial_resume"


def _phase2_checkpoint_cache_evidence(path: str | Path) -> dict[str, object]:
    """回報 failure 時保留的 Phase 2 checkpoint cache/provenance。"""
    root = Path(path).expanduser().resolve(strict=False)
    manifest = root / "manifest.json"
    npz_files = sorted(item for item in root.glob("*.npz") if item.is_file())
    evidence: dict[str, object] = {
        "path": str(root),
        "exists": root.is_dir(),
        "manifest_path": str(manifest),
        "manifest_exists": manifest.is_file(),
        "npz_count": len(npz_files),
    }
    if manifest.is_file():
        try:
            evidence["manifest_sha256"] = _sha256_file(manifest)
            evidence["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            evidence["manifest_read_error"] = str(error)
    return evidence


def _phase2_output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """保留 run module 慣例的 Phase 2 output-path adapter seam。"""
    return phase2_output_paths(config)


def _validate_phase2_checkpoint_lock(
    state: Any,
    config: Mapping[str, Any],
) -> None:
    """比對 brief pin 的 Phase 1 checkpoint combined/manifest/NPZ lock。"""
    expected_combined = str(config.get("phase1_checkpoint_sha256", "")).strip().upper()
    expected_manifest = str(
        config.get("phase1_checkpoint_manifest_sha256", "")
    ).strip().upper()
    expected_count = config.get("phase1_checkpoint_npz_count", 120)
    if expected_combined and str(state.checkpoint_fingerprint["sha256"]).upper() != expected_combined:
        raise ValueError("Phase 1 checkpoint combined fingerprint mismatch")
    if expected_manifest and str(state.checkpoint_fingerprint["manifest_sha256"]).upper() != expected_manifest:
        raise ValueError("Phase 1 checkpoint manifest fingerprint mismatch")
    if int(state.checkpoint_fingerprint["npz_count"]) != int(expected_count):
        raise ValueError("Phase 1 checkpoint NPZ count mismatch")


def _validate_phase2_mapping_config(config: Mapping[str, Any]) -> None:
    """驗證 config 的 reversed hypothesis 與 reviewed diagnostics mapping 完全一致。"""
    raw = config.get("condition_mapping_reversed_hypothesis")
    if not isinstance(raw, Mapping):
        raise ValueError("config 缺少 condition_mapping_reversed_hypothesis mapping")
    expected = {
        int(index): {
            "condition": dose.label,
            "ifn_dose": int(dose.ifn_dose),
            "tnf_dose": int(dose.tnf_dose),
        }
        for index, dose in CONDITION_MAPPING_REVERSED_HYPOTHESIS.items()
    }
    normalized: dict[int, dict[str, object]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError("condition_mapping_reversed_hypothesis row 必須是 mapping")
        normalized[int(key)] = {
            "condition": str(value.get("condition", value.get("label", ""))),
            "ifn_dose": int(value.get("ifn_dose")),
            "tnf_dose": int(value.get("tnf_dose")),
        }
    if normalized != expected:
        raise ValueError("condition_mapping_reversed_hypothesis 與 reviewed mapping 不一致")


def _run_phase2_analysis(args: argparse.Namespace) -> int:
    """執行 mapping-free Phase 2 adapter 與 restricted atomic publication。"""
    generation_id = uuid.uuid4().hex
    config: Mapping[str, Any] | None = None
    source_root: Path | None = None
    staging_root: Path | None = None
    output_paths: dict[str, Path] = {}
    report_path: Path | None = None
    phase2_checkpoint_dir: Path | None = None
    phase1_state = None
    try:
        config = _load_config(args.config)
        _validate_phase2_mapping_config(config)
        source_root = _resolve_config_path(config, "source_root", Path.cwd())
        phase2_paths = _phase2_output_paths(config)
        output_paths = phase2_paths
        phase2_checkpoint_dir = phase2_paths["phase2_cv_checkpoint_dir"]
        report_path = _output_path(
            config,
            "report_path",
            "immunity/outputs/exp4/REPORT.md",
        )
        for field, path in phase2_paths.items():
            _ensure_output_outside_source_root(path, source_root, field)
        _ensure_output_outside_source_root(report_path, source_root, "report_path")
        phase1_metadata_path = args.metadata or _metadata_path(config)
        full_paths = _full_output_paths(config)
        safe_paths = _safe_output_paths(config)
        phase1_checkpoint_dir = safe_paths["cv_checkpoint_dir"]
        phase1_output_root_value = config.get(
            "phase1_output_root", phase1_metadata_path.parent
        )
        phase1_output_root = _resolve_config_path(
            {"path": phase1_output_root_value}, "path", Path.cwd()
        )
        phase1_artifact_paths = _phase2_phase1_artifact_paths(
            config,
            source_root=source_root,
            phase1_output_root=phase1_output_root,
        )
        expected_hashes = config.get("phase1_protected_artifact_sha256")
        if not isinstance(expected_hashes, Mapping):
            raise ValueError("config 缺少 phase1_protected_artifact_sha256 mapping")
        expected_hashes = {
            str(name): str(value) for name, value in expected_hashes.items()
        }
        protected_artifact_paths = {
            name: phase1_artifact_paths[name] for name in expected_hashes
        }
        protected_hashes = validate_phase1_hash_lock(
            protected_artifact_paths,
            expected_hashes,
        )
        actual_hashes = {
            **{
                name: _sha256_file(path)
                for name, path in phase1_artifact_paths.items()
            },
            **protected_hashes,
        }
        # The 25-documentation lock intentionally excludes the two frozen
        # source CSVs; they still enter the immutable state and FrozenEvidence.
        if set(actual_hashes) != set(phase1_artifact_paths):
            raise ValueError("Phase 1 actual hash roster mismatch")
        phase1_metadata = _read_json_mapping(phase1_metadata_path)
        run_log_path = full_paths["run_log_path"]
        phase1_state = capture_phase1_immutable_state(
            artifact_paths=phase1_artifact_paths,
            checkpoint_dir=phase1_checkpoint_dir,
            mask_dir=_resolve_config_path(config, "masks_dir", source_root),
            metadata_path=phase1_metadata_path,
            run_log_path=run_log_path,
            report_path=report_path,
        )
        _validate_phase2_checkpoint_lock(phase1_state, config)
        expected_mask = str(config.get("phase1_mask_cache_fingerprint", "")).strip().lower()
        if expected_mask and phase1_state.mask_cache_fingerprint.lower() != expected_mask:
            raise ValueError("Phase 1 mask-cache fingerprint mismatch")
        validate_phase2_publish_mapping(
            {**phase2_paths, "report": report_path},
            phase1_paths={
                **phase1_artifact_paths,
                "run_metadata.json": phase1_metadata_path,
                "run.log": run_log_path,
            },
            phase1_checkpoint_dir=phase1_checkpoint_dir,
        )

        manifest_path = _resolve_config_path(config, "manifest_path", source_root)
        manifest = _read_phase2_manifest(manifest_path)
        phase1_oof = pd.read_csv(safe_paths["oof_predictions_path"])
        cell_level_basic = pd.read_csv(
            _resolve_config_path(config, "cell_level_path", source_root)
        )
        cell_dedup_report = pd.read_csv(
            safe_paths["cell_dedup_report_path"],
            float_precision="round_trip",
        )
        cell_level_rui49 = pd.read_csv(full_paths["cell_level_rui49_path"])
        feature_health = pd.read_csv(full_paths["feature_health_report_path"])
        redundancy = pd.read_csv(safe_paths["feature_redundancy_report_path"])
        evidence = FrozenPhase1Evidence.from_observed(
            artifact_sha256={
                name: actual_hashes[name]
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
            pre_border_consistency=_phase2_consistency_payload(
                phase1_metadata, "pre_border_consistency"
            ),
            retained_consistency=_phase2_consistency_payload(
                phase1_metadata, "retained_consistency"
            ),
        )
        population = reconstruct_phase2_population(
            phase1_oof=phase1_oof,
            cell_level_basic=cell_level_basic,
            cell_dedup_report=cell_dedup_report,
            cell_level_rui49=cell_level_rui49,
            feature_health_report=feature_health,
            feature_redundancy_report=redundancy,
            manifest=manifest,
            evidence=evidence,
        )
        mask_fingerprint = phase1_state.mask_cache_fingerprint
        phase1_leakage = LeakageContext(
            standardization_scope="pipeline_fit_within_outer_training",
            feature_selection_inputs=("X",),
            cellpose_invoked=False,
            mask_cache_fingerprint_before=mask_fingerprint,
            mask_cache_fingerprint_after=mask_fingerprint,
        )
        leakage_context = Phase2LeakageContext.formal(phase1_leakage)
        # The reviewed Phase 2 core has no dose-mapping input.  One assembly is
        # sufficient; the source-component boundary and actual check-8 evidence
        # below are the mapping-absence attestation.  No counterfactual assembly
        # or second CV fit is performed.
        assembly = assemble_phase2_cv(population, leakage_context)
        source_components = dict(phase2_cv_source_components())
        check_eight = assembly.leakage.report.loc[
            assembly.leakage.report["check_id"].eq(8)
        ]
        if len(check_eight) != 1 or str(check_eight.iloc[0]["status"]) != "passed":
            raise ValueError("Phase 2 actual leakage check 8 未通過")
        check_eight_evidence = {
            str(key): _json_safe(value)
            for key, value in check_eight.iloc[0].to_dict().items()
        }

        staging_root = (
            phase2_paths["phase2_run_metadata_path"].parent
            / f".staging.phase2.{generation_id}"
        )
        staging_root.mkdir(parents=True, exist_ok=False)
        staged_paths = _phase2_stage_paths(staging_root, phase2_paths)
        phase2_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cv_result = run_phase2_cv(
            assembly,
            leakage_context,
            checkpoint_dir=phase2_checkpoint_dir,
            phase1_checkpoint_dir=phase1_checkpoint_dir,
            n_jobs=_positive_or_nonzero_int(config.get("cv_n_jobs", 6), "cv_n_jobs"),
            importance_repeats=_positive_int(
                config.get("importance_repeats", 3), "importance_repeats"
            ),
        )
        exploratory_paths = _exploratory_output_paths(config)
        phase1_metrics = pd.read_csv(safe_paths["cv_metrics_path"])
        phase1_shrinkage = pd.read_csv(exploratory_paths["shrinkage_analysis_path"])
        tables = derive_phase2_tables(
            cv_result=cv_result,
            phase1_metrics=phase1_metrics,
            phase1_shrinkage=phase1_shrinkage,
            include_dose=False,
        )
        write_phase2_core_tables(staged_paths, tables)
        numeric_paths = {
            name: staged_paths[field]
            for name, field in PHASE2_NUMERIC_PATH_FIELDS.items()
        }
        numeric_before = fingerprint_phase2_numeric_artifacts(numeric_paths)
        dose_response = build_dose_response_check(
            tables.targets,
            condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
        )
        residual_vs_dose = analyze_residual_vs_dose(
            tables.oof_predictions,
            condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
        )
        tables = replace(
            tables,
            dose_response=dose_response,
            residual_vs_dose=residual_vs_dose,
        )
        write_phase2_dose_tables(staged_paths, tables)
        render_phase2_figures(
            staged_paths,
            comparison=tables.comparison,
            within_condition=tables.within_condition,
            dose_response=tables.dose_response,
        )
        numeric_after = fingerprint_phase2_numeric_artifacts(numeric_paths)
        immutability = verify_phase2_numeric_artifact_immutability(
            numeric_before, numeric_after
        )
        checkpoint_status = _phase2_checkpoint_status(cv_result.metrics)
        checkpoint_manifest = _json_safe(
            getattr(getattr(cv_result, "core", None), "checkpoint_manifest", {})
        )
        metadata: dict[str, Any] = {
            "mode": "run-phase2",
            "status": "validated",
            "generation_id": generation_id,
            "population": {
                "retained_cells": int(len(population.cells)),
                "fov_count": int(population.cells["image_key"].nunique()),
                "group_count": 72,
                "healthy_feature_count": len(population.healthy_features),
                "filtered_feature_count": len(population.filtered_features),
            },
            "phase1_artifact_sha256": actual_hashes,
            "phase1_protected_artifact_sha256": {
                name: actual_hashes[name] for name in expected_hashes
            },
            "phase1_source_artifact_sha256": {
                name: actual_hashes[name]
                for name in ("data_manifest.csv", "cell_level_basic.csv")
            },
            "phase1_checkpoint_fingerprint": phase1_state.checkpoint_fingerprint,
            "phase1_mask_cache_fingerprint": phase1_state.mask_cache_fingerprint,
            "mapping_absent_by_core_interface": True,
            "mapping_independence": {
                "status": "passed",
                "mapping_absent_by_core_interface": True,
                "core_source_components_sha256": source_components,
                "cv_input_fingerprints": dict(assembly.cv_input_fingerprints),
                "actual_leakage_check_8": check_eight_evidence,
                "proof": (
                    "five_phase2_cv_numeric_artifact_sha256_before_after_dose_diagnostics"
                ),
            },
            "mapping_independence_status": "passed",
            "phase2_cv_source_components_sha256": source_components,
            "phase2_cv_input_fingerprints": dict(assembly.cv_input_fingerprints),
            "leakage_preflight": {
                "status": "validated",
                "passed": bool(cv_result.leakage.passed),
                "check_count": int(len(cv_result.leakage.report)),
            },
            "phase2_numeric_artifact_sha256": {
                "before_dose_diagnostics": numeric_before.as_mapping(),
                "after_dose_diagnostics": numeric_after.as_mapping(),
                "immutability_status": "passed" if immutability.passed else "blocked",
            },
            "cv_numbers_bit_for_bit_unchanged_by_dose_mapping": bool(
                immutability.passed
            ),
            "dose_mapping_introduced_after_core_numeric_freeze": True,
            "dose_mapping_status": "post_cv_only_unconfirmed_reversed_hypothesis",
            "dose_mapping_note": DOSE_MAPPING_NOTE,
            "checkpoint_status": checkpoint_status,
            "phase2_checkpoint": {
                "path": str(phase2_checkpoint_dir.resolve(strict=False)),
                "checkpoint_status": checkpoint_status,
                "manifest": checkpoint_manifest,
                "npz_count": int(
                    len(list(phase2_checkpoint_dir.glob("*.npz")))
                ),
            },
            "phase2_run_metadata_path": str(phase2_paths["phase2_run_metadata_path"].resolve(strict=False)),
            "phase2_run_log_path": str(phase2_paths["phase2_run_log_path"].resolve(strict=False)),
        }
        for field, path in phase2_paths.items():
            metadata[field] = str(path.resolve(strict=False))
        write_phase2_log(
            staged_paths["phase2_run_log_path"],
            (
                f"mode=run-phase2 generation_id={generation_id}",
                "status=validated",
                "mapping_independence_status=passed",
                "mapping_absent_by_core_interface=true",
                "dose_mapping_phase=post_cv_only",
                DOSE_MAPPING_NOTE,
                f"checkpoint_status={checkpoint_status}",
                "cv_numeric_sha256_proof=before_dose_diagnostics_equals_after_dose_diagnostics",
                "cv_numbers_bit_for_bit_unchanged_by_dose_mapping=true",
                f"phase1_checkpoint_sha256={phase1_state.checkpoint_fingerprint['sha256']}",
            ),
        )
        metadata["phase2_artifacts"] = {
            field: {
                "path": str(phase2_paths[field].resolve(strict=False)),
                "sha256": sha256_file(staged_paths[field]),
            }
            for field in (*PHASE2_TABLE_FIELDS, *PHASE2_FIGURE_FIELDS)
        }
        metadata["phase2_run_log_sha256"] = sha256_file(
            staged_paths["phase2_run_log_path"]
        )
        write_phase2_metadata(staged_paths["phase2_run_metadata_path"], metadata)
        report_section = build_phase2_report_section(
            metadata=metadata,
            targets=tables.targets,
            within_condition=tables.within_condition,
            dose_response=tables.dose_response,
        )
        staged_report = staging_root / "REPORT.md"
        staged_report.write_bytes(
            replace_phase2_report_section(
                report_path.read_bytes(),
                report_section,
            )
        )
        verify_phase1_immutable_state(
            phase1_state,
            artifact_paths=phase1_artifact_paths,
            checkpoint_dir=phase1_checkpoint_dir,
            mask_dir=_resolve_config_path(config, "masks_dir", source_root),
            metadata_path=phase1_metadata_path,
            run_log_path=run_log_path,
            report_path=report_path,
        )
        publish_output = phase2_publish_paths(output_paths, report_path=report_path)
        publish_staged = {
            **staged_paths,
            "report": staged_report,
        }
        archived = publish_phase2_bundle(
            publish_staged,
            publish_output,
            phase1_paths={
                **phase1_artifact_paths,
                "run_metadata.json": phase1_metadata_path,
                "run.log": run_log_path,
            },
            phase1_checkpoint_dir=phase1_checkpoint_dir,
            generation_id=generation_id,
            post_publish_gate=lambda: verify_phase1_immutable_state(
                phase1_state,
                artifact_paths=phase1_artifact_paths,
                checkpoint_dir=phase1_checkpoint_dir,
                mask_dir=_resolve_config_path(config, "masks_dir", source_root),
                metadata_path=phase1_metadata_path,
                run_log_path=run_log_path,
                report_path=report_path,
            ),
        )
        print(
            json.dumps(
                {
                    "mode": "run-phase2",
                    "status": "validated",
                    "generation_id": generation_id,
                    "mapping_independence_status": "passed",
                    "leakage_check_count": len(cv_result.leakage.report),
                    "archived_historical_outputs": archived,
                    "output_paths": {
                        key: str(path.resolve(strict=False))
                        for key, path in publish_output.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        diagnostic: dict[str, Any] = {
            "mode": "run-phase2",
            "status": "blocked",
            "generation_id": generation_id,
            "source_write_status": "not_written",
            "error": str(error),
            "mapping_independence_status": "not_validated",
            "output_paths": {
                key: str(Path(path).resolve(strict=False))
                for key, path in output_paths.items()
            },
        }
        if phase2_checkpoint_dir is not None:
            diagnostic["phase2_checkpoint_cache"] = _phase2_checkpoint_cache_evidence(
                phase2_checkpoint_dir
            )
        failure_path: Path | None = None
        try:
            if output_paths:
                failure_path = (
                    output_paths["phase2_run_metadata_path"].parent
                    / f".failed-phase2-generation.{generation_id}.json"
                )
                _write_json_atomic(diagnostic, failure_path)
        except (OSError, TypeError, ValueError):
            failure_path = None
        summary = {
            "mode": "run-phase2",
            "status": "blocked",
            "generation_id": generation_id,
            "error": str(error),
            "failure_evidence_path": str(failure_path) if failure_path else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"STATUS=blocked: Phase 2 analysis failed: {error}", file=sys.stderr)
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)


def _run_exploratory_analysis(args: argparse.Namespace) -> int:
    """執行 validated formal CV generation 的 post-CV diagnostics transaction。

    此 adapter 僅重建既有 CV population，呼叫兩個唯讀 pure cores，並把七項
    post-CV artifact、REPORT 與 metadata 以 restricted rollback publisher 一次
    發布；不呼叫 ``run_rui_cv``，也不修改任何 protected CV numeric artifact。

    Args:
        args: parser 產生的 config 與 metadata override。

    Returns:
        ``0`` 表示整包 post-CV artifacts 已發布；``2`` 表示任一 formal gate、
        core、hash、checkpoint 或 transaction 失敗。
    """
    generation_id = uuid.uuid4().hex
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    config: Mapping[str, Any] | None = None
    source_root: Path | None = None
    current_paths: dict[str, Path] = {}
    output_paths: dict[str, Path] = {}
    staging_root: Path | None = None
    old_metadata: dict[str, Any] = {}
    try:
        config = _load_config(args.config)
        source_root = _resolve_config_path(config, "source_root", Path.cwd())
        manifest_path = _resolve_config_path(config, "manifest_path", source_root)
        cell_level_path = _resolve_config_path(config, "cell_level_path", source_root)
        masks_dir = _resolve_config_path(config, "masks_dir", source_root)
        expectation = SnapshotExpectation.from_mapping(config.get("expected"))

        current_paths = _safe_output_paths(config)
        full_paths = _full_output_paths(config)
        current_paths.update(
            {
                field: full_paths[field]
                for field in ("cell_level_rui49_path", "feature_health_report_path")
            }
        )
        current_paths["metadata_path"] = args.metadata or current_paths["metadata_path"]
        _ensure_safe_output_paths(current_paths, source_root)
        output_paths = _exploratory_output_paths(config)
        output_paths["metadata_path"] = current_paths["metadata_path"]
        _ensure_safe_output_paths(output_paths, source_root)
        _ensure_post_cv_output_paths(
            output_paths=output_paths,
            protected_paths=current_paths,
        )
        old_metadata = _read_json_mapping(current_paths["metadata_path"])

        _validate_cv_config_locks(config, expectation)
        _validate_cv_current_generation(
            metadata=old_metadata,
            current_paths=current_paths,
            expected_group_count=FORMAL_CV_LOCKS["group_count"],
            expected_fov_count=FORMAL_CV_LOCKS["fov_count"],
        )
        protected_before = _validate_protected_cv_hashes(config, current_paths)
        checkpoint_before = _checkpoint_fingerprint(current_paths["cv_checkpoint_dir"])

        snapshot = validate_data_snapshot(
            source_root=source_root,
            manifest_path=manifest_path,
            cell_level_path=cell_level_path,
            masks_dir=masks_dir,
            expected=expectation,
        )
        if not snapshot.matched:
            raise ValueError("snapshot gate failed: " + "; ".join(snapshot.differences))
        mask_before = _mask_cache_fingerprint(masks_dir)
        inputs = _prepare_cv_analysis_inputs(
            config=config,
            manifest_path=manifest_path,
            cell_level_path=cell_level_path,
            masks_dir=masks_dir,
            current_paths=current_paths,
            expectation=expectation,
            metadata=old_metadata,
            mask_cache_fingerprint_before=mask_before,
        )
        mask_after = _mask_cache_fingerprint(masks_dir)
        if mask_before != mask_after:
            raise ValueError(
                "mask cache fingerprint changed during exploratory input assembly; "
                "fail-closed without publishing"
            )
        feature_roster = _validate_exploratory_reconstruction(inputs, old_metadata)
        metrics, oof_predictions, feature_importance = _read_post_cv_inputs(
            current_paths
        )
        _validate_post_cv_input_contract(
            metrics=metrics,
            oof_predictions=oof_predictions,
            feature_importance=feature_importance,
            retained_cells=inputs.retained_cells,
        )
        manifest = pd.read_csv(manifest_path)
        started = perf_counter()
        shrinkage = analyze_shrinkage(oof_predictions)
        orientation = analyze_orientation_marginal(inputs.retained_cells)
        post_cv_core_seconds = perf_counter() - started
        exploratory = run_exploratory_analysis(
            inputs.retained_cells,
            feature_roster,
            manifest,
        )
        best = _select_post_cv_best_configuration(metrics, shrinkage.report)
        orientation_payload = _build_orientation_metadata(
            orientation.summary,
            feature_importance,
        )
        checkpoint_after_compute = _checkpoint_fingerprint(
            current_paths["cv_checkpoint_dir"]
        )
        if checkpoint_after_compute != checkpoint_before:
            raise ValueError("CV checkpoint changed during post-CV computation")
        protected_after_compute = _validate_protected_cv_hashes(config, current_paths)
        if protected_after_compute != protected_before:
            raise ValueError("protected CV artifact changed during post-CV computation")

        planned_archived = _planned_archived_outputs(output_paths, generation_id)
        metadata = _build_post_cv_metadata(
            old_metadata,
            output_paths=output_paths,
            generation_id=generation_id,
            inputs=inputs,
            metrics=metrics,
            best=best,
            shrinkage=shrinkage,
            orientation=orientation,
            orientation_payload=orientation_payload,
            exploratory=exploratory,
            post_cv_core_seconds=post_cv_core_seconds,
            protected_hashes_before=protected_before,
            checkpoint_before=checkpoint_before,
            planned_archived=planned_archived,
        )

        staging_root = output_root / f".staging.exploratory.{generation_id}"
        staging_root.mkdir(parents=True, exist_ok=False)
        staged_paths = _stage_post_cv_paths(staging_root, output_paths)
        _write_post_cv_tables(
            staged_paths,
            shrinkage=shrinkage,
            orientation=orientation,
            exploratory=exploratory,
        )
        targets = _read_optional_csv(current_paths.get("group_targets_path"))
        sensitivity = _read_optional_csv(
            current_paths.get("group_target_sensitivity_path")
        )
        border_report = _read_optional_csv(current_paths.get("border_exclusion_report_path"))
        write_report_skeleton(
            staged_paths["report_path"],
            targets=targets,
            sensitivity=sensitivity,
            border_report=border_report,
            metadata=metadata,
        )
        _write_json_atomic(metadata, staged_paths["metadata_path"])

        # Re-check protected state immediately before the restricted publisher.
        protected_before_publish = _validate_protected_cv_hashes(config, current_paths)
        checkpoint_before_publish = _checkpoint_fingerprint(
            current_paths["cv_checkpoint_dir"]
        )
        if protected_before_publish != protected_before:
            raise ValueError("protected CV artifact changed immediately before publish")
        if checkpoint_before_publish != checkpoint_before:
            raise ValueError("CV checkpoint changed immediately before publish")
        archived = publish_analysis_bundle(
            staged_paths,
            output_paths,
            fields=POST_CV_ANALYSIS_FILE_FIELDS,
            generation_id=generation_id,
            post_publish_gate=lambda: _validate_post_publish_protected_state(
                config=config,
                current_paths=current_paths,
                protected_before=protected_before,
                checkpoint_before=checkpoint_before,
            ),
        )
        summary = _exploratory_summary(
            metadata,
            output_paths=output_paths,
            status="validated",
        )
        summary["archived_historical_outputs"] = archived
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        diagnostic = {
            "mode": "run-exploratory",
            "status": "blocked",
            "generation_id": generation_id,
            "source_write_status": "not_written",
            "error": str(error),
            "analysis_status": {
                "shrinkage": "blocked",
                "orientation_marginal": "blocked",
                "umap": "blocked",
                "kmeans": "blocked",
            },
            "output_paths": {
                key: str(Path(path).resolve(strict=False))
                for key, path in output_paths.items()
            },
        }
        failure_evidence_path: Path | None = None
        try:
            if source_root is None or _path_is_outside_source_root(output_root, source_root):
                failure_evidence_path = _write_failed_generation_evidence(
                    output_root,
                    generation_id=generation_id,
                    diagnostic=diagnostic,
                )
        except (OSError, TypeError, ValueError):
            failure_evidence_path = None
        summary = _exploratory_summary(
            diagnostic,
            output_paths=output_paths,
            status="blocked",
        )
        summary["error"] = str(error)
        if failure_evidence_path is not None:
            summary["failure_evidence_path"] = str(failure_evidence_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(
            f"STATUS=blocked: post-CV exploratory analysis failed: {error}",
            file=sys.stderr,
        )
        return 2
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)


def _validate_exploratory_reconstruction(
    inputs: CvAnalysisInputs,
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    """驗證 adapter 重建的 19,648-cell frame 與 exact 30 欄 roster。"""
    frame = inputs.retained_cells
    if len(frame) != FORMAL_CV_LOCKS["retained_cells"]:
        raise ValueError("exploratory retained cell count mismatch")
    if frame.loc[:, ["image_key", "cell_label"]].duplicated().any():
        raise ValueError("exploratory retained cell keys must be unique")
    expected_payload = _as_mapping(_as_mapping(metadata.get("feature_arms")).get("rui_filtered"))
    raw_roster = expected_payload.get("feature_columns")
    if isinstance(raw_roster, (str, bytes)) or not isinstance(raw_roster, Sequence):
        raise ValueError("current metadata rui_filtered feature_columns 缺失")
    expected_roster = tuple(str(value) for value in raw_roster)
    actual_roster = tuple(str(value) for value in inputs.redundancy_result.retained_features)
    if len(expected_roster) != 30 or actual_roster != expected_roster or set(actual_roster) != set(expected_roster):
        raise ValueError(
            "exploratory rui_filtered roster 必須與 current metadata exact match"
        )
    missing_features = [column for column in actual_roster if column not in frame.columns]
    if missing_features:
        raise ValueError("exploratory retained frame 缺少 rui_filtered feature")
    return actual_roster


def _read_post_cv_inputs(
    current_paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """讀取 canonical CV metrics、OOF 與 feature importance tables。"""
    required = (
        "cv_metrics_path",
        "oof_predictions_path",
        "feature_importance_path",
    )
    missing = [field for field in required if field not in current_paths]
    if missing:
        raise ValueError(f"post-CV current paths 缺少欄位：{missing}")
    return tuple(pd.read_csv(current_paths[field]) for field in required)  # type: ignore[return-value]


def _validate_post_cv_input_contract(
    *,
    metrics: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    retained_cells: pd.DataFrame,
) -> None:
    """驗證 canonical CV tables 的 formal row、schema、configuration 與 key set。"""
    if not {"image_key", "cell_label"}.issubset(retained_cells.columns):
        raise ValueError("retained frame 缺少 image_key/cell_label key columns")
    metric_columns = (
        "configuration_id", "arm", "model", "outer_fold", "n_train", "n_test",
        "test_group_count", "rui_r2", "cell_r2", "r2_gap_rui_minus_cell",
        "cell_mae", "cell_rmse", "cell_spearman", "cell_spearman_status",
        "best_inner_r2", "grid_candidate_count", "fit_seconds", "checkpoint_status",
    )
    oof_columns = (
        "configuration_id", "arm", "model", "outer_fold", "image_key", "cell_label",
        "group_id", "confidence_flag", "observed_ido_score", "predicted_ido_score", "residual",
    )
    importance_columns = (
        "configuration_id", "arm", "model", "outer_fold", "feature", "importance_method",
        "importance_mean", "importance_std", "absolute_importance", "rank", "is_spatial_feature",
    )
    if tuple(metrics.columns) != metric_columns:
        raise ValueError("cv_metrics.csv schema 不符")
    if tuple(oof_predictions.columns) != oof_columns:
        raise ValueError("oof_predictions.csv schema 不符")
    if tuple(feature_importance.columns) != importance_columns:
        raise ValueError("feature_importance.csv schema 不符")
    if len(metrics) != 120 or len(oof_predictions) != 471_552:
        raise ValueError("canonical CV row counts 不符 formal contract")
    metric_configs = set(metrics["configuration_id"].astype(str))
    oof_configs = set(oof_predictions["configuration_id"].astype(str))
    importance_configs = set(feature_importance["configuration_id"].astype(str))
    if len(metric_configs) != 24 or metric_configs != oof_configs or metric_configs != importance_configs:
        raise ValueError("canonical CV configuration roster mismatch")
    if metrics.groupby("configuration_id").size().ne(5).any():
        raise ValueError("cv_metrics 每個 configuration 必須恰含 5 folds")
    if metrics.loc[:, ["rui_r2", "cell_r2"]].apply(pd.to_numeric, errors="coerce").isna().any().any():
        raise ValueError("cv_metrics R2 欄位含 nonnumeric")
    retained_keys = set(
        zip(
            retained_cells["image_key"].astype(str),
            retained_cells["cell_label"].astype(str),
        )
    )
    if len(retained_keys) != FORMAL_CV_LOCKS["retained_cells"]:
        raise ValueError("retained frame key count mismatch")
    for configuration_id, group in oof_predictions.groupby("configuration_id", sort=False):
        if len(group) != FORMAL_CV_LOCKS["retained_cells"]:
            raise ValueError(f"OOF configuration row count mismatch: {configuration_id}")
        keys = list(zip(group["image_key"].astype(str), group["cell_label"].astype(str)))
        if len(set(keys)) != len(keys) or set(keys) != retained_keys:
            raise ValueError(f"OOF configuration key set mismatch: {configuration_id}")
    if feature_importance.empty:
        raise ValueError("feature_importance.csv 不可為空")


def _select_post_cv_best_configuration(
    metrics: pd.DataFrame,
    shrinkage_report: pd.DataFrame,
) -> dict[str, object]:
    """依 canonical 五-fold mean Rui R² 選唯一 best configuration。"""
    means = (
        metrics.assign(
            rui_r2=pd.to_numeric(metrics["rui_r2"], errors="coerce"),
            cell_r2=pd.to_numeric(metrics["cell_r2"], errors="coerce"),
        )
        .groupby("configuration_id", sort=False)
        .agg(
            mean_rui_r2=("rui_r2", "mean"),
            mean_cell_r2=("cell_r2", "mean"),
            fold_count=("rui_r2", "size"),
        )
        .reset_index()
    )
    if means["mean_rui_r2"].isna().any():
        raise ValueError("best configuration selection encountered nonfinite R2")
    maximum = float(means["mean_rui_r2"].max())
    winners = means.loc[means["mean_rui_r2"].eq(maximum)]
    if len(winners) != 1:
        raise ValueError("best configuration mean Rui R2 tie；fail-closed")
    selected = winners.iloc[0]
    configuration_id = str(selected["configuration_id"])
    matches = shrinkage_report.loc[
        shrinkage_report["configuration_id"].astype(str).eq(configuration_id)
    ]
    if len(matches) != 1:
        raise ValueError("best configuration missing unique shrinkage row")
    row = matches.iloc[0].to_dict()
    row.update(
        {
            "configuration_id": configuration_id,
            "mean_rui_r2": float(selected["mean_rui_r2"]),
            "mean_cell_r2": float(selected["mean_cell_r2"]),
            "fold_count": int(selected["fold_count"]),
        }
    )
    return _json_safe(row)


def _build_orientation_metadata(
    summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> dict[str, object]:
    """重算 Orientation marginal summary 與三個 RFR mean ranks。"""
    if tuple(summary.columns) != ORIENTATION_SUMMARY_COLUMNS or len(summary) != 1:
        raise ValueError("orientation summary schema 不符")
    payload = _json_safe(summary.iloc[0].to_dict())
    orientation = feature_importance.loc[
        feature_importance["feature"].astype(str).eq("cell__Orientation")
        & feature_importance["model"].astype(str).eq("RFR")
        & feature_importance["importance_method"].astype(str).eq("native_impurity_decrease")
    ].copy()
    ranks: dict[str, float] = {}
    for arm in ("geometry_24", "rui_filtered", "rui_48_plus_nucleus"):
        values = pd.to_numeric(
            orientation.loc[orientation["arm"].astype(str).eq(arm), "rank"],
            errors="coerce",
        )
        if len(values) != 5 or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Orientation RFR rank rows mismatch: {arm}")
        ranks[f"{arm}_mean_rfr_rank"] = float(values.mean())
    payload.update(ranks)
    return payload


def _write_post_cv_tables(
    staged_paths: Mapping[str, Path],
    *,
    shrinkage: Any,
    orientation: Any,
    exploratory: Any,
) -> None:
    """將兩個 pure-core tables 與三個 deterministic PNG 寫入 staging。"""
    if tuple(shrinkage.report.columns) != SHRINKAGE_REPORT_COLUMNS:
        raise ValueError("shrinkage report schema 不符")
    if tuple(orientation.bins.columns) != ORIENTATION_BIN_COLUMNS:
        raise ValueError("orientation bin schema 不符")
    if tuple(exploratory.embedding.columns) != EMBEDDING_COLUMNS:
        raise ValueError("UMAP embedding schema 不符")
    if tuple(exploratory.cluster_feature_report.columns) != CLUSTER_FEATURE_REPORT_COLUMNS:
        raise ValueError("kmeans feature report schema 不符")
    shrinkage.report.to_csv(staged_paths["shrinkage_analysis_path"], index=False)
    orientation.bins.to_csv(
        staged_paths["orientation_target_marginal_path"], index=False
    )
    exploratory.embedding.to_csv(staged_paths["umap_embeddings_path"], index=False)
    exploratory.cluster_feature_report.to_csv(
        staged_paths["kmeans_cluster_features_path"], index=False
    )
    figure_fields = {
        "umap_by_donor.png": "umap_by_donor_path",
        "umap_by_passage.png": "umap_by_passage_path",
        "umap_by_condition.png": "umap_by_condition_path",
    }
    if set(exploratory.figures) != set(figure_fields):
        raise ValueError("exploratory figures 必須恰含 donor/passage/condition")
    for name, field in figure_fields.items():
        staged_paths[field].write_bytes(exploratory.figures[name])


def _build_post_cv_metadata(
    existing: Mapping[str, Any],
    *,
    output_paths: Mapping[str, Path],
    generation_id: str,
    inputs: CvAnalysisInputs,
    metrics: pd.DataFrame,
    best: Mapping[str, object],
    shrinkage: Any,
    orientation: Any,
    orientation_payload: Mapping[str, object],
    exploratory: Any,
    post_cv_core_seconds: float,
    protected_hashes_before: Mapping[str, str],
    checkpoint_before: Mapping[str, object],
    planned_archived: Mapping[str, str],
) -> dict[str, Any]:
    """建立保留 CV generation、加入 post-CV nested generation 的 metadata。"""
    metadata = dict(existing)
    statuses = dict(_as_mapping(existing.get("analysis_status")))
    statuses.update(
        {
            "shrinkage": "validated",
            "orientation_marginal": "validated",
            "umap": "validated",
            "kmeans": "validated",
        }
    )
    outputs = {
        field: str(Path(path).resolve(strict=False))
        for field, path in output_paths.items()
    }
    provenance = _json_safe(exploratory.provenance)
    embedding = exploratory.embedding
    cluster_counts = {
        str(int(cluster)): int(count)
        for cluster, count in embedding.loc[
            embedding["is_putative_ifn_stimulated"].astype(bool), "kmeans_cluster_id"
        ].value_counts(dropna=True).sort_index().items()
    }
    feature_roster = tuple(inputs.redundancy_result.retained_features)
    post = {
        "status": "validated",
        "generation_id": generation_id,
        "analysis_status": {
            "shrinkage": "validated",
            "orientation_marginal": "validated",
            "umap": "validated",
            "kmeans": "validated",
        },
        "outputs": outputs,
        "row_counts": {
            "shrinkage_analysis": int(len(shrinkage.report)),
            "orientation_target_marginal": int(len(orientation.bins)),
            "umap_embeddings": int(len(embedding)),
            "kmeans_cluster_features": int(len(exploratory.cluster_feature_report)),
        },
        "schemas": {
            "shrinkage_analysis": list(SHRINKAGE_REPORT_COLUMNS),
            "orientation_target_marginal": list(ORIENTATION_BIN_COLUMNS),
            "umap_embeddings": list(EMBEDDING_COLUMNS),
            "kmeans_cluster_features": list(CLUSTER_FEATURE_REPORT_COLUMNS),
        },
        "population": {
            "cell_count": FORMAL_CV_LOCKS["retained_cells"],
            "fov_count": FORMAL_CV_LOCKS["fov_count"],
            "group_count": FORMAL_CV_LOCKS["group_count"],
            "feature_count": len(feature_roster),
            "feature_columns": list(feature_roster),
        },
        "feature_roster": list(feature_roster),
        "best_configuration": _json_safe(best),
        "shrinkage": {
            "report_row_count": int(len(shrinkage.report)),
            "group_audit_row_count": int(len(shrinkage.group_summary)),
            "best_configuration_id": str(best["configuration_id"]),
        },
        "orientation": {
            **_json_safe(orientation_payload),
            "bin_target_medians": _json_safe(
                orientation.bins["target_median"].astype(float).tolist()
            ),
            "bin_target_means": _json_safe(
                orientation.bins["target_mean"].astype(float).tolist()
            ),
            "bin_count": int(len(orientation.bins)),
        },
        "umap": {
            "cell_count": int(len(embedding)),
            "fov_count": int(embedding["image_key"].astype(str).nunique()),
            "group_count": int(embedding["group_id"].astype(str).nunique()),
            "feature_count": len(feature_roster),
            "feature_columns": list(feature_roster),
            "standardization_scope": provenance.get(
                "standard_scaler_fit_scope", "full_exploratory_population"
            ),
            "umap_fit_columns": provenance.get("umap_fit_columns", list(feature_roster)),
            "umap_requested_params": provenance.get("umap_requested_params", {"n_components": 2, "random_state": 42}),
            "umap_actual_params": provenance.get("umap_actual_params", {}),
            "umap_defaults": provenance.get("umap_defaults", provenance.get("umap_actual_params", {})),
            "umap_learn_version": provenance.get("umap_learn_version"),
            "kmeans_fit_columns": provenance.get("kmeans_fit_columns", ["UMAP1", "UMAP2"]),
            "kmeans_params": provenance.get("kmeans_params", provenance.get("kmeans_actual_params", {})),
            "kmeans_actual_params": provenance.get("kmeans_actual_params", {}),
            "scikit_learn_version": provenance.get("scikit_learn_version"),
            "random_state": provenance.get("random_state", 42),
            "ifn_selector": provenance.get("ifn_selector", "ifn_dose > 0"),
            "ifn_selector_untrusted": True,
            "ifn_selected_cell_count": provenance.get("ifn_selected_cell_count"),
            "ifn_selected_fov_count": provenance.get("ifn_selected_fov_count"),
            "ifn_selected_group_count": provenance.get("ifn_selected_group_count"),
            "cluster_counts": cluster_counts,
            "target_used_for_fit": False,
            "target_used_for_umap": False,
            "target_used_for_kmeans_fit": False,
            "target_used_only_for_posthoc_cluster_label": True,
            "target_median_aggregation_unit": provenance.get(
                "target_median_aggregation_unit"
            ),
            "report_feature_units": provenance.get("report_feature_units"),
            "umap_input_units": provenance.get("umap_input_units"),
            "condition_warning": CONDITION_WARNING_TEXT,
            "condition_label_trust": "unconfirmed",
            "condition_plot_use": "visualization_only",
            "dose_conclusions_allowed": False,
            "figure_paths": {
                name: outputs[field]
                for name, field in {
                    "umap_by_donor.png": "umap_by_donor_path",
                    "umap_by_passage.png": "umap_by_passage_path",
                    "umap_by_condition.png": "umap_by_condition_path",
                }.items()
            },
        },
        "fit_scopes": {
            "umap": "X-only full-population 30-column rui_filtered feature matrix",
            "kmeans": "UMAP1/UMAP2 coordinates for putative-IFN cells only",
            "target": "post-hoc cluster naming/descriptive diagnostics only",
        },
        "condition_label_trust": "unconfirmed",
        "condition_plot_use": "visualization_only",
        "dose_conclusions_allowed": False,
        "ifn_selector_untrusted": True,
        "putative_ifn_selector": "ifn_dose > 0",
        "target_used_for_umap": False,
        "target_used_for_kmeans_fit": False,
        "target_used_only_for_posthoc_cluster_label": True,
        "target_used_for_fit": False,
        "mlpr_convergence": {
            "max_iter": 200,
            "parameter_unchanged": True,
            "status": "not_converged_at_anchor_limit",
            "warning": "ConvergenceWarning observed during formal CV",
            "warning_count": "not_captured",
        },
        "timing": {
            "post_cv_core_seconds": float(post_cv_core_seconds),
            "exploratory": _json_safe(exploratory.timing),
            "total_seconds": float(post_cv_core_seconds + exploratory.timing.get("total_seconds", 0.0)),
        },
        "protected_cv_artifact_sha256_before": dict(protected_hashes_before),
        "protected_cv_artifact_sha256_after": dict(protected_hashes_before),
        "checkpoint_fingerprint_before": _json_safe(checkpoint_before),
        "checkpoint_fingerprint_after": _json_safe(checkpoint_before),
        "archived_historical_outputs": dict(planned_archived),
    }
    metadata.update(
        {
            # The formal CV generation remains the top-level current generation.
            "generation_id": existing.get("generation_id"),
            "analysis_status": statuses,
            "analysis_no_umap_kmeans": False,
            "post_cv_analysis": post,
            "archived_historical_outputs": {
                **_as_string_mapping(existing.get("archived_historical_outputs")),
                **dict(planned_archived),
            },
        }
    )
    metadata.update(
        {
            "condition_label_trust": "unconfirmed",
            "condition_plot_use": "visualization_only",
            "dose_conclusions_allowed": False,
            "ifn_selector_untrusted": True,
            "putative_ifn_selector": "ifn_dose > 0",
            "target_used_for_umap": False,
            "target_used_for_kmeans_fit": False,
            "target_used_only_for_posthoc_cluster_label": True,
            "target_used_for_fit": False,
        }
    )
    for field, path in output_paths.items():
        metadata[field] = outputs[field]
    metadata["post_cv_analysis_generation_id"] = generation_id
    return _json_safe(metadata)


def _exploratory_summary(
    metadata: Mapping[str, Any],
    *,
    output_paths: Mapping[str, Path],
    status: str,
) -> dict[str, Any]:
    """建立 post-CV CLI 的 machine-readable summary。"""
    post = _as_mapping(metadata.get("post_cv_analysis"))
    population = _as_mapping(post.get("population"))
    return {
        "mode": "run-exploratory",
        "status": status,
        "generation_id": post.get("generation_id"),
        "formal_generation_id": metadata.get("generation_id"),
        "analysis_status": metadata.get("analysis_status", {}),
        "population": population,
        "timing": post.get("timing", {}),
        "output_paths": {
            key: str(Path(path).resolve(strict=False))
            for key, path in output_paths.items()
        },
    }


def _validate_cv_config_locks(
    config: Mapping[str, Any], expectation: SnapshotExpectation
) -> None:
    """驗證 formal CV 不可由 config 降級到 fixture population。"""
    expected = config.get("expected", {})
    if not isinstance(expected, Mapping):
        raise ValueError("config expected 必須是 mapping")
    expected_values = {
        "manifest_image_key_count": FORMAL_CV_LOCKS["fov_count"],
        "cell_row_count": FORMAL_CV_LOCKS["raw_pair_rows"],
        "cell_image_key_count": FORMAL_CV_LOCKS["fov_count"],
        "dedup_cell_count": FORMAL_CV_LOCKS["pre_border_cells"],
        "post_border_cell_count": FORMAL_CV_LOCKS["retained_cells"],
        "duplicate_key_count": FORMAL_CV_LOCKS["duplicate_key_count"],
        "retained_raw_pair_rows": FORMAL_CV_LOCKS["retained_raw_pair_rows"],
        "group_count": FORMAL_CV_LOCKS["group_count"],
    }
    mismatches: list[str] = []
    for key, required in expected_values.items():
        try:
            actual = int(expected.get(key))
        except (TypeError, ValueError):
            actual = None
        if actual != required:
            mismatches.append(f"expected.{key}={actual!r}, required={required}")
    if expectation.group_keys != tuple(
        (
            "B4_P5", "B4_P6", "B4_P7", "B7_P5", "B7_P6", "B7_P7",
            "B8_P5", "B8_P6", "B8_P7",
        )
    ):
        mismatches.append("expected.group_keys 不符 formal 9-group roster")
    if mismatches:
        raise ValueError("formal CV config lock mismatch: " + "; ".join(mismatches))


def _validate_cv_current_generation(
    *,
    metadata: Mapping[str, Any],
    current_paths: Mapping[str, Path],
    expected_group_count: int,
    expected_fov_count: int,
) -> None:
    """驗證目前 full Rui49/target chain 是同一個 formal generation。"""
    required_statuses = (
        "status",
        "full_rui49_status",
        "full_693_feature_consistency_status",
        "feature_health_status",
        "target_status",
        "target_sensitivity_status",
        "border_exclusion_status",
    )
    bad = [
        key
        for key in required_statuses
        if str(metadata.get(key, "not_run")) != "validated"
    ]
    if bad:
        raise ValueError(f"current formal generation status 未 validated：{bad}")
    full = metadata.get("full_rui49")
    if not isinstance(full, Mapping):
        raise ValueError("current metadata 缺少 full_rui49 generation payload")
    lock_fields = {
        "raw_pair_row_count": FORMAL_CV_LOCKS["raw_pair_rows"],
        "pre_border_distinct_cell_count": FORMAL_CV_LOCKS["pre_border_cells"],
        "retained_cell_count": FORMAL_CV_LOCKS["retained_cells"],
        "retained_raw_pair_row_count": FORMAL_CV_LOCKS["retained_raw_pair_rows"],
        "fov_count": expected_fov_count,
        "canonical_feature_count": 49,
    }
    mismatches = [
        f"{key}={full.get(key)!r}, required={required}"
        for key, required in lock_fields.items()
        if full.get(key) != required
    ]
    if mismatches:
        raise ValueError("current full Rui49 lock mismatch: " + "; ".join(mismatches))
    cell_path = current_paths.get("cell_level_rui49_path")
    health_path = current_paths.get("feature_health_report_path")
    if cell_path is None or not Path(cell_path).is_file():
        raise ValueError("current cell_level_rui49.csv 缺失")
    if health_path is None or not Path(health_path).is_file():
        raise ValueError("current feature_health_report.csv 缺失")
    _validate_current_reporting_artifacts(
        metadata=metadata,
        current_paths=current_paths,
        expected_group_count=expected_group_count,
        expected_fov_count=expected_fov_count,
    )


def _positive_int(value: object, name: str) -> int:
    """解析正式設定中的正整數。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必須是正整數")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是正整數") from error
    if number <= 0:
        raise ValueError(f"{name} 必須是正整數")
    return number


def _positive_or_nonzero_int(value: object, name: str) -> int:
    """解析可為負值（代表 joblib all workers）但不可為零的整數。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必須是非零整數")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是非零整數") from error
    if number == 0:
        raise ValueError(f"{name} 不可為 0")
    return number


def _sha256_file(path: Path) -> str:
    """計算單一 canonical artifact 的 SHA256。"""
    candidate = Path(path).expanduser().resolve(strict=False)
    if not candidate.is_file():
        raise ValueError(f"protected artifact missing: {candidate}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest().upper()


def _protected_cv_artifact_paths(
    current_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """建立十個 protected CV artifact 的 filename→path mapping。"""
    field_by_name = {
        "cv_metrics.csv": "cv_metrics_path",
        "oof_predictions.csv": "oof_predictions_path",
        "per_group_residuals.csv": "per_group_residuals_path",
        "feature_importance.csv": "feature_importance_path",
        "hyperparameters.csv": "hyperparameters_path",
        "hyperparameter_grids.json": "hyperparameter_grids_path",
        "feature_health_report.csv": "feature_health_report_path",
        "nucleus_sanity_report.csv": "nucleus_sanity_report_path",
        "feature_redundancy_report.csv": "feature_redundancy_report_path",
        "leakage_preflight_report.csv": "leakage_preflight_report_path",
    }
    missing = [field for field in field_by_name.values() if field not in current_paths]
    if missing:
        raise ValueError(f"protected artifact paths 缺少欄位：{missing}")
    return {
        name: Path(current_paths[field]).expanduser().resolve(strict=False)
        for name, field in field_by_name.items()
    }


def _validate_protected_cv_hashes(
    config: Mapping[str, Any],
    current_paths: Mapping[str, Path],
) -> dict[str, str]:
    """依 config-pinned mapping fail-closed 驗證十個 formal CV hashes。"""
    expected = config.get("expected_cv_artifact_sha256")
    if not isinstance(expected, Mapping):
        raise ValueError("config 缺少 expected_cv_artifact_sha256 mapping")
    expected_keys = set(PROTECTED_CV_ARTIFACT_FILES)
    if set(str(key) for key in expected) != expected_keys:
        raise ValueError("expected_cv_artifact_sha256 必須恰含十個 protected artifacts")
    actual: dict[str, str] = {}
    for name, path in _protected_cv_artifact_paths(current_paths).items():
        value = str(expected.get(name, "")).upper()
        if len(value) != 64 or any(character not in "0123456789ABCDEF" for character in value):
            raise ValueError(f"expected hash 格式錯誤：{name}")
        digest = _sha256_file(path)
        if digest != value:
            raise ValueError(
                f"protected artifact hash mismatch: {name}; expected={value}; actual={digest}"
            )
        actual[name] = digest
    return actual


def _validate_post_publish_protected_state(
    *,
    config: Mapping[str, Any],
    current_paths: Mapping[str, Path],
    protected_before: Mapping[str, str],
    checkpoint_before: Mapping[str, object],
) -> None:
    """驗證 post-CV commit 後 protected CV 與 checkpoint 仍完全不變。

    Args:
        config: 含 pinned protected-artifact hashes 的正式設定。
        current_paths: 目前 formal CV canonical paths。
        protected_before: publisher 前已驗證的十項 protected hashes。
        checkpoint_before: publisher 前已驗證的 checkpoint fingerprint。

    Raises:
        ValueError: 任一 protected artifact 或 checkpoint 在 commit 後改變。
    """
    protected_after = _validate_protected_cv_hashes(config, current_paths)
    checkpoint_after = _checkpoint_fingerprint(current_paths["cv_checkpoint_dir"])
    if protected_after != protected_before:
        raise ValueError("protected CV artifact changed after publish")
    if checkpoint_after != checkpoint_before:
        raise ValueError("CV checkpoint changed after publish")


def _checkpoint_fingerprint(checkpoint_dir: Path) -> dict[str, object]:
    """對 formal CV checkpoint manifest 與 120 NPZ 建立內容 fingerprint。"""
    root = Path(checkpoint_dir).expanduser().resolve(strict=False)
    manifest = root / "manifest.json"
    files = sorted(path for path in root.glob("*.npz") if path.is_file())
    if not root.is_dir() or not manifest.is_file() or len(files) != 120:
        raise ValueError(
            "formal CV checkpoint 必須包含 manifest.json 與恰好 120 個 NPZ"
        )
    digest = hashlib.sha256()
    for path in [manifest, *files]:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "sha256": digest.hexdigest().upper(),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "npz_count": len(files),
    }


def _build_cv_leakage_context(
    mask_cache_fingerprint_before: str,
    mask_cache_fingerprint_after: str,
) -> LeakageContext:
    """建立符合 cv core exact contract 的 formal leakage provenance。"""
    return LeakageContext(
        standardization_scope="pipeline_fit_within_outer_training",
        feature_selection_inputs=("X",),
        cellpose_invoked=False,
        mask_cache_fingerprint_before=str(mask_cache_fingerprint_before),
        mask_cache_fingerprint_after=str(mask_cache_fingerprint_after),
    )


def _mask_cache_fingerprint(masks_dir: Path) -> str:
    """對 mask cache 的 ``*.npz`` 檔名與內容建立 deterministic SHA256。"""
    root = Path(masks_dir).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"mask cache directory 不存在：{root}")
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*.npz") if path.is_file()
    )
    if not files:
        raise ValueError("mask cache 沒有 .npz 檔")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stage_cv_paths(
    staging_root: Path, output_paths: Mapping[str, Path]
) -> dict[str, Path]:
    """依 canonical analysis file paths 建立同 generation staging paths。"""
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    staged: dict[str, Path] = {}
    for field in CV_ANALYSIS_FILE_FIELDS:
        if field not in output_paths:
            raise ValueError(f"CV output paths 缺少欄位：{field}")
        final = Path(output_paths[field]).expanduser().resolve(strict=False)
        try:
            relative = final.relative_to(output_root)
        except ValueError as error:
            raise ValueError(f"{field} 不在 current analysis output root") from error
        target = staging_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        staged[field] = target
    return staged


def _stage_post_cv_paths(
    staging_root: Path, output_paths: Mapping[str, Path]
) -> dict[str, Path]:
    """建立 post-CV 七 artifact、REPORT 與 metadata 的 staging paths。"""
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    staged: dict[str, Path] = {}
    for field in POST_CV_ANALYSIS_FILE_FIELDS:
        if field not in output_paths:
            raise ValueError(f"post-CV output paths 缺少欄位：{field}")
        final = Path(output_paths[field]).expanduser().resolve(strict=False)
        try:
            relative = final.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                f"{field} 不在 current analysis output root"
            ) from error
        target = staging_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        staged[field] = target
    return staged


def _read_metadata_fallback_counts(metadata: Mapping[str, Any]) -> dict[str, int]:
    """讀取 full extraction 的 fallback counts；缺欄時以零填入。"""
    full = metadata.get("full_rui49")
    payload = full.get("fallback_counts", {}) if isinstance(full, Mapping) else {}
    if not isinstance(payload, Mapping):
        payload = {}
    return {
        feature: int(payload.get(feature, 0))
        for feature in RUI49_FEATURE_COLUMNS
    }


def _assert_consistency_lock(
    result: Any,
    *,
    name: str,
    expected_rows: int,
    expected_cells: int,
    expected_duplicates: int,
    expected_fovs: int,
    frame: pd.DataFrame,
) -> None:
    """檢查 high exact consistency result 與 formal scope 的所有鎖定值。"""
    values = {
        "rows": int(result.checked_row_count),
        "cells": int(result.checked_cell_count),
        "duplicate_keys": int(result.checked_duplicate_key_count),
        "features": int(result.feature_count),
        "fovs": int(frame["image_key"].astype(str).nunique()),
    }
    expected = {
        "rows": expected_rows,
        "cells": expected_cells,
        "duplicate_keys": expected_duplicates,
        "features": 49,
        "fovs": expected_fovs,
    }
    mismatch = [
        f"{name}.{key}={values[key]}, required={value}"
        for key, value in expected.items()
        if values[key] != value
    ]
    if mismatch:
        raise ValueError("formal Rui49 consistency lock mismatch: " + "; ".join(mismatch))


def _prepare_cv_analysis_inputs(
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    cell_level_path: Path,
    masks_dir: Path,
    current_paths: Mapping[str, Path],
    expectation: SnapshotExpectation,
    metadata: Mapping[str, Any],
    mask_cache_fingerprint_before: str,
) -> CvAnalysisInputs:
    """以 full current generation 組出 19,648-cell、65-active-feature CV frame。"""
    manifest = pd.read_csv(manifest_path)
    raw_cells = pd.read_csv(cell_level_path)
    expected_dedup = FORMAL_CV_LOCKS["pre_border_cells"]
    expected_duplicates = FORMAL_CV_LOCKS["duplicate_key_count"]
    from .border_exclusion import _deduplicate_for_border

    deduplicated, dedup_report = _deduplicate_for_border(
        raw_cells,
        deduplicated_cells=None,
        dedup_report=None,
        expected_raw_rows=FORMAL_CV_LOCKS["raw_pair_rows"],
        expected_distinct_cells=expected_dedup,
        expected_multirow_keys=expected_duplicates,
    )
    expected_sparse = _expected_sparse_fov_keys(config)
    border_result = apply_border_exclusion(
        manifest,
        raw_cells,
        masks_dir,
        expected_manifest_keys=FORMAL_CV_LOCKS["fov_count"],
        expected_cell_rows=FORMAL_CV_LOCKS["raw_pair_rows"],
        expected_cell_keys=FORMAL_CV_LOCKS["fov_count"],
        expected_dedup_cells=expected_dedup,
        expected_duplicate_keys=expected_duplicates,
        expected_sparse_fov_keys=expected_sparse,
        expected_group_keys=expectation.group_keys,
        deduplicated_cells=deduplicated,
        dedup_report=dedup_report,
    )
    if len(border_result.eligible_cells) != FORMAL_CV_LOCKS["retained_cells"]:
        raise ValueError("formal retained cell count mismatch after border exclusion")

    rui_path = current_paths.get("cell_level_rui49_path")
    if rui_path is None:
        raise ValueError("current cell_level_rui49_path missing")
    rui_raw = pd.read_csv(rui_path)
    expected_schema = ("image_key", "cell_label", "nucleus_label", *RUI49_FEATURE_COLUMNS)
    if tuple(rui_raw.columns) != expected_schema:
        raise ValueError("current cell_level_rui49.csv schema 不符")
    raw_consistency = assert_full_rui49_feature_consistency(
        rui_raw,
        scope="full_rui49_raw_23976_rows_693_fovs",
    )
    _assert_consistency_lock(
        raw_consistency,
        name="raw",
        expected_rows=FORMAL_CV_LOCKS["raw_pair_rows"],
        expected_cells=FORMAL_CV_LOCKS["pre_border_cells"],
        expected_duplicates=FORMAL_CV_LOCKS["duplicate_key_count"],
        expected_fovs=FORMAL_CV_LOCKS["fov_count"],
        frame=rui_raw,
    )

    retained_keys = border_result.eligible_cells.loc[:, ["image_key", "cell_label"]]
    retained_rui_raw = rui_raw.merge(
        retained_keys,
        on=["image_key", "cell_label"],
        how="inner",
        validate="many_to_one",
    )
    retained_consistency = assert_full_rui49_feature_consistency(
        retained_rui_raw,
        scope="full_rui49_retained_20440_rows_19648_cells_693_fovs",
    )
    _assert_consistency_lock(
        retained_consistency,
        name="retained",
        expected_rows=FORMAL_CV_LOCKS["retained_raw_pair_rows"],
        expected_cells=FORMAL_CV_LOCKS["retained_cells"],
        expected_duplicates=769,
        expected_fovs=FORMAL_CV_LOCKS["fov_count"],
        frame=retained_rui_raw,
    )

    rui_distinct = rui_raw.drop_duplicates(
        ["image_key", "cell_label"], keep="first"
    ).drop(columns=["nucleus_label"])
    retained = border_result.eligible_cells.merge(
        manifest.loc[:, ["image_key", "group_id"]],
        on="image_key",
        how="left",
        validate="many_to_one",
    )
    target_path = current_paths.get("group_targets_path")
    targets = _read_optional_csv(target_path)
    if targets is None or "group_IDO_score" not in targets.columns:
        raise ValueError("current group_targets.csv missing group_IDO_score")
    target_columns = ["group_id", "group_IDO_score"]
    if "confidence_flag" in targets.columns:
        target_columns.append("confidence_flag")
    target_map = targets.loc[:, target_columns].copy()
    target_map["group_id"] = target_map["group_id"].astype(str)
    target_map = target_map.drop_duplicates("group_id", keep="first")
    retained = retained.merge(
        target_map,
        on="group_id",
        how="left",
        validate="many_to_one",
    )
    if "confidence_flag" not in retained.columns:
        retained["confidence_flag"] = "normal"
    retained["confidence_flag"] = retained["confidence_flag"].fillna("normal")
    modeling = retained.merge(
        rui_distinct,
        on=["image_key", "cell_label"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_rui"),
    )
    if len(modeling) != FORMAL_CV_LOCKS["retained_cells"]:
        raise ValueError("formal modeling frame row count mismatch")
    if modeling["group_IDO_score"].isna().any():
        raise ValueError("formal modeling frame has cells without group target")

    health_columns = (*RUI49_FEATURE_COLUMNS, *NUCLEUS_FEATURE_COLUMNS)
    health_features = modeling.loc[:, list(health_columns)].copy()
    fallback_counts = _read_metadata_fallback_counts(metadata)
    fallback_counts.update({feature: 0 for feature in NUCLEUS_FEATURE_COLUMNS})
    health_result = evaluate_feature_health(
        health_features,
        fallback_counts=fallback_counts,
        eligible_cell_count=FORMAL_CV_LOCKS["retained_cells"],
        feature_columns=health_columns,
        fallback_scope="cv_formal_retained_19648_cells_65_active_features",
    )
    if health_result.blocked:
        raise ValueError("formal 65-feature health blocked: " + "; ".join(health_result.reasons))
    if tuple(health_result.removed_columns) != ("cell__MinIntensity",):
        raise ValueError(
            "formal zero-variance removal mismatch: "
            f"{list(health_result.removed_columns)}"
        )
    active_health = health_result.report.loc[
        health_result.report["feature"].ne("cell__MinIntensity")
    ].reset_index(drop=True)
    if len(active_health) != 65:
        raise ValueError(f"formal active feature-health row count expected 65, actual {len(active_health)}")

    sanity_report = build_nucleus_sanity_report(modeling)
    redundancy_result = analyze_feature_redundancy(
        modeling.loc[:, list(RUI48_FEATURE_COLUMNS)],
        RUI48_FEATURE_COLUMNS,
        AUTHORITATIVE_RUI28_FEATURES,
    )
    assembly = assemble_feature_arms(
        modeling,
        filtered_features=redundancy_result.retained_features,
        contract=CvPopulationContract.formal_exp4(),
    )
    source_fingerprint = _cv_source_fingerprint(
        config=config,
        metadata=metadata,
        masks_dir=masks_dir,
        input_paths=(
            manifest_path,
            cell_level_path,
            Path(rui_path),
            Path(current_paths["group_targets_path"]),
        ),
        mask_cache_fingerprint=mask_cache_fingerprint_before,
        assembly=assembly,
    )
    return CvAnalysisInputs(
        retained_cells=modeling,
        health_result=health_result,
        # Publish all 66 checked input columns so the removed MinIntensity row
        # remains auditable; only 65 healthy columns enter the four arms.
        health_report=health_result.report.copy(),
        nucleus_sanity_report=sanity_report,
        redundancy_result=redundancy_result,
        assembly=assembly,
        source_fingerprint=source_fingerprint,
        raw_consistency=raw_consistency,
        retained_consistency=retained_consistency,
        raw_pair_row_count=len(rui_raw),
        retained_raw_pair_row_count=len(retained_rui_raw),
        retained_fov_count=int(modeling["image_key"].astype(str).nunique()),
        mask_cache_fingerprint_before=mask_cache_fingerprint_before,
        mask_cache_fingerprint_after=mask_cache_fingerprint_before,
    )


def _cv_source_fingerprint(
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    masks_dir: Path,
    input_paths: Sequence[Path],
    mask_cache_fingerprint: str,
    assembly: Any,
) -> str:
    """hash formal data、algorithm source、config 與 arm roster 供 checkpoint。"""
    digest = hashlib.sha256()
    for label, path in zip(
        ("manifest", "cell_level_basic", "cell_level_rui49", "group_targets"),
        input_paths,
    ):
        candidate = Path(path).expanduser().resolve(strict=False)
        if not candidate.is_file():
            raise ValueError(f"CV source fingerprint input missing: {candidate}")
        digest.update(label.encode("utf-8"))
        digest.update(candidate.read_bytes())
    digest.update(b"mask_cache_fingerprint")
    digest.update(mask_cache_fingerprint.encode("ascii"))
    digest.update(json.dumps(_json_safe(config), sort_keys=True, ensure_ascii=False).encode("utf-8"))
    digest.update(json.dumps(_json_safe(metadata.get("generation_id", "")), ensure_ascii=False).encode("utf-8"))
    for module_name in (
        "cv.py", "nucleus_sanity.py", "feature_health.py", "redundancy.py",
        "cell_dedup.py", "rui_features.py", "run_rui2025.py",
    ):
        module_path = Path(__file__).with_name(module_name)
        digest.update(module_name.encode("utf-8"))
        digest.update(module_path.read_bytes())
    digest.update(json.dumps(_json_safe(assembly.as_mapping()), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _validate_cv_result_contract(result: Any) -> None:
    """確認 core 回傳所有 formal analysis tables 與七項 preflight。"""
    required = (
        "metrics", "oof_predictions", "hyperparameters", "per_group_residuals",
        "feature_importance", "preflight", "checkpoint_manifest",
        "hyperparameter_grids",
    )
    missing = [field for field in required if not hasattr(result, field)]
    if missing:
        raise ValueError(f"CV result 缺少欄位：{missing}")
    preflight = result.preflight
    if not bool(preflight.passed) or len(preflight.report) != 7:
        raise ValueError("leakage preflight 必須七項全數 passed")
    for field in (
        "metrics", "oof_predictions", "hyperparameters", "per_group_residuals",
        "feature_importance",
    ):
        if not isinstance(getattr(result, field), pd.DataFrame):
            raise ValueError(f"CV result.{field} 必須是 DataFrame")


def _write_cv_tables(
    staged_paths: Mapping[str, Path],
    *,
    inputs: CvAnalysisInputs,
    result: Any,
) -> None:
    """把 E3、health、redundancy、preflight 與 CV tables 寫進 staging。"""
    inputs.health_report.to_csv(staged_paths["feature_health_report_path"], index=False)
    inputs.nucleus_sanity_report.to_csv(
        staged_paths["nucleus_sanity_report_path"], index=False
    )
    inputs.redundancy_result.report.to_csv(
        staged_paths["feature_redundancy_report_path"], index=False
    )
    result.preflight.report.to_csv(
        staged_paths["leakage_preflight_report_path"], index=False
    )
    result.metrics.to_csv(staged_paths["cv_metrics_path"], index=False)
    result.oof_predictions.to_csv(staged_paths["oof_predictions_path"], index=False)
    result.hyperparameters.to_csv(
        staged_paths["hyperparameters_path"], index=False
    )
    result.feature_importance.to_csv(
        staged_paths["feature_importance_path"], index=False
    )
    result.per_group_residuals.to_csv(
        staged_paths["per_group_residuals_path"], index=False
    )
    _write_json_atomic(
        _json_safe(result.hyperparameter_grids),
        staged_paths["hyperparameter_grids_path"],
    )


def _build_cv_metadata(
    existing: Mapping[str, Any],
    *,
    inputs: CvAnalysisInputs,
    result: Any,
    output_paths: Mapping[str, Path],
    generation_id: str,
    cv_n_jobs: int,
    importance_repeats: int,
    mask_cache_fingerprint_before: str,
    mask_cache_fingerprint_after: str,
) -> dict[str, Any]:
    """建立只含事實與狀態的 CV generation metadata。"""
    metadata = dict(existing)
    planned_archived = _planned_archived_outputs(
        {field: output_paths[field] for field in CV_ANALYSIS_FILE_FIELDS},
        generation_id,
    )
    health_payload = _json_safe(inputs.health_result.to_metadata_payload())
    full_payload = metadata.get("full_rui49")
    if isinstance(full_payload, Mapping):
        full_payload = dict(full_payload)
        full_payload["feature_health_report_path"] = planned_archived.get(
            "feature_health_report_path", full_payload.get("feature_health_report_path")
        )
        full_payload["feature_health_report_scope"] = "full_extraction_49_columns"
        metadata["full_rui49"] = full_payload
    arm_specs = {
        name: {str(key): _json_safe(value) for key, value in spec.items()}
        for name, spec in FEATURE_ARM_SPECS.items()
    }
    arm_specs["rui_48"]["count"] = 48
    arm_specs["rui_48"]["feature_columns"] = list(RUI48_FEATURE_COLUMNS)
    arm_specs["rui_filtered"]["count"] = len(inputs.redundancy_result.retained_features)
    arm_specs["rui_filtered"]["feature_columns"] = list(
        inputs.redundancy_result.retained_features
    )
    arm_specs["rui_48_plus_nucleus"]["count"] = 65
    checkpoint_manifest = _json_safe(result.checkpoint_manifest)
    metadata.update(
        {
            "generation_id": generation_id,
            "status": "validated",
            "analysis_status": {
                "feature_health": "validated",
                "nucleus_sanity": "validated",
                "feature_redundancy": "validated",
                "leakage_preflight": "validated",
                "cv": "validated",
                "umap": "not_run",
                "kmeans": "not_run",
            },
            "feature_health_status": "validated",
            "feature_health": {
                **health_payload,
                "status": "validated",
                "scope": "cv_formal_retained_19648_cells_65_active_features",
                "report_path": str(Path(output_paths["feature_health_report_path"]).resolve(strict=False)),
                "report_row_count": len(inputs.health_report),
                "input_feature_count": len(inputs.health_result.report),
                "active_feature_count": len(inputs.health_result.healthy_feature_columns),
                "zero_variance_checked_columns": 66,
                "zero_variance_removed_columns": ["cell__MinIntensity"],
                "near_zero_variance_threshold": 0.05,
                "near_zero_variance_action": "record_only",
            },
            "feature_health_report_path": str(Path(output_paths["feature_health_report_path"]).resolve(strict=False)),
            "feature_health_input_feature_count": len(inputs.health_result.report),
            "feature_health_active_feature_count": len(inputs.health_result.healthy_feature_columns),
            "nucleus_sanity": {
                "status": "validated",
                "report_path": str(Path(output_paths["nucleus_sanity_report_path"]).resolve(strict=False)),
                "row_count": len(inputs.nucleus_sanity_report),
                "n_cells": FORMAL_CV_LOCKS["retained_cells"],
                "fov_count": FORMAL_CV_LOCKS["fov_count"],
            },
            "feature_redundancy": {
                "status": "validated",
                "report_path": str(Path(output_paths["feature_redundancy_report_path"]).resolve(strict=False)),
                "report_row_count": len(inputs.redundancy_result.report),
                **_json_safe(inputs.redundancy_result.to_metadata_payload()),
            },
            "leakage_preflight": {
                "status": "validated",
                "passed": True,
                "check_count": len(result.preflight.report),
                "report_path": str(Path(output_paths["leakage_preflight_report_path"]).resolve(strict=False)),
                "cellpose_invoked": False,
                "mask_cache_fingerprint_before": mask_cache_fingerprint_before,
                "mask_cache_fingerprint_after": mask_cache_fingerprint_after,
            },
            "cv": {
                "status": "validated",
                "population_scope": "formal_19648_cells_693_fovs",
                "cell_count": FORMAL_CV_LOCKS["retained_cells"],
                "fov_count": FORMAL_CV_LOCKS["fov_count"],
                "arm_names": [
                    "geometry_24", "rui_48", "rui_filtered", "rui_48_plus_nucleus"
                ],
                "model_count": 6,
                "outer_fold_count": 5,
                "metrics_row_count": len(result.metrics),
                "oof_row_count": len(result.oof_predictions),
                "hyperparameter_row_count": len(result.hyperparameters),
                "feature_importance_row_count": len(result.feature_importance),
                "per_group_residual_row_count": len(result.per_group_residuals),
                "n_jobs": cv_n_jobs,
                "importance_repeats": importance_repeats,
                "checkpoint_manifest": checkpoint_manifest,
                "checkpoint_dir": str(Path(output_paths["cv_checkpoint_dir"]).resolve(strict=False)),
                "metrics_path": str(Path(output_paths["cv_metrics_path"]).resolve(strict=False)),
                "oof_predictions_path": str(Path(output_paths["oof_predictions_path"]).resolve(strict=False)),
                "hyperparameters_path": str(Path(output_paths["hyperparameters_path"]).resolve(strict=False)),
                "hyperparameter_grids_path": str(Path(output_paths["hyperparameter_grids_path"]).resolve(strict=False)),
                "feature_importance_path": str(Path(output_paths["feature_importance_path"]).resolve(strict=False)),
                "per_group_residuals_path": str(Path(output_paths["per_group_residuals_path"]).resolve(strict=False)),
            },
            "cv_metrics_path": str(Path(output_paths["cv_metrics_path"]).resolve(strict=False)),
            "oof_predictions_path": str(Path(output_paths["oof_predictions_path"]).resolve(strict=False)),
            "hyperparameters_path": str(Path(output_paths["hyperparameters_path"]).resolve(strict=False)),
            "hyperparameter_grids_path": str(Path(output_paths["hyperparameter_grids_path"]).resolve(strict=False)),
            "feature_importance_path": str(Path(output_paths["feature_importance_path"]).resolve(strict=False)),
            "per_group_residuals_path": str(Path(output_paths["per_group_residuals_path"]).resolve(strict=False)),
            "nucleus_sanity_report_path": str(Path(output_paths["nucleus_sanity_report_path"]).resolve(strict=False)),
            "feature_redundancy_report_path": str(Path(output_paths["feature_redundancy_report_path"]).resolve(strict=False)),
            "leakage_preflight_report_path": str(Path(output_paths["leakage_preflight_report_path"]).resolve(strict=False)),
            "cv_checkpoint_dir": str(Path(output_paths["cv_checkpoint_dir"]).resolve(strict=False)),
            "cv_n_jobs": cv_n_jobs,
            "importance_repeats": importance_repeats,
            "mask_cache_fingerprint_before": mask_cache_fingerprint_before,
            "mask_cache_fingerprint_after": mask_cache_fingerprint_after,
            "cellpose_invoked": False,
            "full_693_feature_consistency_status": "validated",
            "full_rui49": {
                **(_json_safe(full_payload) if isinstance(full_payload, Mapping) else {}),
                "raw_pair_row_count": FORMAL_CV_LOCKS["raw_pair_rows"],
                "pre_border_distinct_cell_count": FORMAL_CV_LOCKS["pre_border_cells"],
                "retained_cell_count": FORMAL_CV_LOCKS["retained_cells"],
                "retained_raw_pair_row_count": FORMAL_CV_LOCKS["retained_raw_pair_rows"],
                "fov_count": FORMAL_CV_LOCKS["fov_count"],
                "canonical_feature_count": 49,
                "health_report_scope": "full_extraction_49_columns",
            },
            "feature_arms": arm_specs,
            "canonical_rui_feature_count": 49,
            "effective_rui_feature_count": 48,
            "removed_zero_variance_columns": ["cell__MinIntensity"],
            "near_zero_variance_threshold": 0.05,
            "near_zero_variance_action": "record_only",
            "near_zero_variance_columns": [
                str(row.feature)
                for row in inputs.health_report.itertuples(index=False)
                if bool(row.is_near_zero_variance)
            ],
            "deviations": _merge_deviations(
                existing.get("deviations"), FERET_DEVIATION
            ),
            "e2": _e2_status_metadata(
                existing.get("e2"),
                per_fov_path=output_paths.get("nucleus_dapi_validation_path"),
                summary_path=output_paths.get("nucleus_dapi_validation_summary_path"),
            ),
            "e3_arm4_boundary": {
                "status": "mandatory",
                "interpretation": (
                    "Arm 4 只能回答 phase-derived 核代理特徵是否帶入額外資訊；"
                    "未勝過 rui_48 不得推論核沒有用。"
                ),
            },
            "archived_historical_outputs": {
                **_as_string_mapping(existing.get("archived_historical_outputs")),
                **planned_archived,
            },
            "analysis_no_umap_kmeans": True,
        }
    )
    return _json_safe(metadata)


def _cv_summary(
    metadata: Mapping[str, Any],
    *,
    output_paths: Mapping[str, Path],
    status: str,
) -> dict[str, Any]:
    """建立 formal CV CLI 的 machine-readable summary。"""
    cv = metadata.get("cv") if isinstance(metadata.get("cv"), Mapping) else {}
    return {
        "mode": "run-cv",
        "status": status,
        "analysis_status": metadata.get("analysis_status", {}),
        "population_scope": cv.get("population_scope"),
        "cell_count": cv.get("cell_count"),
        "fov_count": cv.get("fov_count"),
        "metrics_row_count": cv.get("metrics_row_count"),
        "output_paths": {
            key: str(Path(path).resolve(strict=False))
            for key, path in output_paths.items()
        },
    }


def _full_canonical_output_paths(
    output_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """取出 full generation 的四個 transactional canonical outputs。"""
    fields = (
        "cell_level_rui49_path",
        "feature_health_report_path",
        "report_path",
        "metadata_path",
    )
    missing = [field for field in fields if field not in output_paths]
    if missing:
        raise ValueError(f"full canonical output paths 缺少欄位：{missing}")
    return {field: Path(output_paths[field]) for field in fields}


def _stage_full_paths(
    staging_root: Path,
    output_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """依 final output 的相對位置建立同 generation staging paths。"""
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    staged: dict[str, Path] = {}
    for field, final_path in _full_canonical_output_paths(output_paths).items():
        final = Path(final_path).expanduser().resolve(strict=False)
        try:
            relative = final.relative_to(output_root)
        except ValueError as error:
            raise ValueError(f"{field} 不在 current full output root") from error
        staged_path = staging_root / relative
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged[field] = staged_path
    return staged


def _normalise_manifest_pc_paths(
    manifest: pd.DataFrame,
    manifest_path: Path,
) -> pd.DataFrame:
    """將 manifest 的 relative PC path 依 CSV parent 正規化成絕對路徑。"""
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest 必須是 pandas DataFrame")
    if "pc_path" not in manifest.columns:
        raise ValueError("manifest 缺少 pc_path")
    normalized = manifest.copy()
    base = Path(manifest_path).expanduser().resolve(strict=False).parent
    values: list[str] = []
    for raw_path in normalized["pc_path"].tolist():
        text = str(raw_path).strip()
        if not text:
            raise ValueError("manifest pc_path 不可為空")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        values.append(str(candidate.resolve(strict=False)))
    normalized["pc_path"] = values
    return normalized


def _read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    """讀取既有 reporting table；缺檔或 malformed table 以 ``None`` 表示。"""
    if path is None or not Path(path).exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
        return None


def _validate_current_reporting_artifacts(
    *,
    metadata: Mapping[str, Any],
    current_paths: Mapping[str, Path],
    expected_group_count: int,
    expected_fov_count: int,
) -> None:
    """驗證已標記 validated 的 target-chain artifacts，缺失即停止 full mode。

    Safe mode 與 full adapter 共用同一組 current tables。若 metadata 已宣稱
    target、sensitivity 或 border validated，full generation 不得悄悄以缺檔、
    parser 可讀但 schema 錯誤的 table 產生新的 canonical feature bundle。
    synthetic fixture 的群組/FOV 數由 caller 依 config expected 傳入。
    """
    checks = (
        (
            "target_status",
            "group_targets_path",
            int(expected_group_count),
            frozenset(
                {
                    "group_id",
                    "b_id",
                    "passage",
                    "fov_count",
                    "cells_before",
                    "cells_after",
                    "group_IDO_score",
                    "group_IDO_score_all_cells",
                    "delta",
                    "confidence_flag",
                }
            ),
            "group_id",
        ),
        (
            "target_sensitivity_status",
            "group_target_sensitivity_path",
            int(expected_group_count),
            frozenset(
                {
                    "group_id",
                    "sparse_fov_count",
                    "target_with_sparse_fovs",
                    "target_without_sparse_fovs",
                    "delta_without_minus_with",
                    "absolute_delta",
                    "baseline_target_range",
                    "absolute_delta_pct_of_range",
                    "threshold_pct",
                    "threshold_absolute",
                    "status",
                }
            ),
            "group_id",
        ),
        (
            "border_exclusion_status",
            "border_exclusion_report_path",
            int(expected_fov_count),
            frozenset(
                {
                    "image_key",
                    "b_id",
                    "passage",
                    "group_id",
                    "cells_before",
                    "cells_excluded_border",
                    "cells_after",
                    "whole_cell_labels_before",
                    "whole_cell_labels_excluded_border",
                    "whole_cell_labels_after",
                    "border_exclusion_fraction",
                    "border_exclusion_row_fraction",
                    "minimum_cell_gate_status",
                    "raw_pair_cells_before",
                    "raw_pair_cells_excluded_border",
                    "raw_pair_cells_after",
                    "raw_pair_exclusion_fraction",
                }
            ),
            "image_key",
        ),
    )
    for status_key, path_key, expected_rows, required_columns, unique_key in checks:
        if str(metadata.get(status_key, "not_run")) != "validated":
            continue
        path = current_paths.get(path_key)
        if path is None:
            raise ValueError(
                f"validated {status_key} 缺少 current output path：{path_key}"
            )
        table = _read_optional_csv(path)
        if table is None:
            raise ValueError(
                f"validated {status_key} 的 {path_key} 缺失或不可讀：{path}"
            )
        missing = sorted(required_columns.difference(table.columns))
        if missing:
            raise ValueError(
                f"validated {status_key} 的 {path_key} schema 缺少欄位：{missing}"
            )
        if len(table) != expected_rows:
            raise ValueError(
                f"validated {status_key} 的 {path_key} row count expected "
                f"{expected_rows}, actual {len(table)}"
            )
        raw_keys = table[unique_key]
        keys = raw_keys.astype(str)
        if raw_keys.isna().any() or keys.str.strip().eq("").any():
            raise ValueError(
                f"validated {status_key} 的 {path_key} key 欄含空值：{unique_key}"
            )
        if keys.nunique(dropna=False) != expected_rows:
            raise ValueError(
                f"validated {status_key} 的 {path_key} key 不唯一：{unique_key}"
            )


def _validate_full_result(
    result: Any,
    *,
    expected_raw_pair_rows: int,
    expected_pre_border_cells: int,
    expected_retained_cells: int,
    expected_raw_multirow_keys: int,
    expected_retained_raw_pair_rows: int,
) -> None:
    """驗證 high result 的 canonical schema、counts、finite 與 health scope。"""
    if tuple(result.raw_pair_features.columns) != (
        "image_key",
        "cell_label",
        "nucleus_label",
        *RUI49_FEATURE_COLUMNS,
    ):
        raise ValueError("full raw canonical schema 不符")
    if len(result.raw_pair_features) != expected_raw_pair_rows:
        raise ValueError("full raw pair row count 不符")
    for frame_name, frame, expected_count in (
        ("pre-border", result.pre_border_features, expected_pre_border_cells),
        ("retained", result.retained_features, expected_retained_cells),
    ):
        if tuple(frame.columns) != ("image_key", "cell_label", *RUI49_FEATURE_COLUMNS):
            raise ValueError(f"full {frame_name} canonical schema 不符")
        if len(frame) != expected_count:
            raise ValueError(f"full {frame_name} cell count 不符")
        try:
            finite = np.isfinite(frame.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(dtype=float))
        except (TypeError, ValueError) as error:
            raise ValueError(f"full {frame_name} feature values 非 numeric") from error
        if not bool(finite.all()):
            raise ValueError(f"full {frame_name} feature values 含 nonfinite")
    try:
        raw_finite = np.isfinite(
            result.raw_pair_features.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(dtype=float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("full raw feature values 非 numeric") from error
    if not bool(raw_finite.all()):
        raise ValueError("full raw feature values 含 nonfinite")

    health_report = result.health.report
    if tuple(health_report.columns) != (
        "feature",
        "nunique",
        "mean",
        "std",
        "relative_std",
        "is_zero_variance",
        "is_near_zero_variance",
        "removed_from_arms",
        "fallback_scope",
        "eligible_cell_count",
        "fallback_count",
        "fallback_rate",
        "status",
    ):
        raise ValueError("feature health report schema 不符")
    if tuple(health_report["feature"].astype(str)) != tuple(RUI49_FEATURE_COLUMNS):
        raise ValueError("feature health report order 不符")
    if len(health_report) != len(RUI49_FEATURE_COLUMNS):
        raise ValueError("feature health report row count 不符")
    if not health_report["eligible_cell_count"].eq(expected_retained_cells).all():
        raise ValueError("feature health denominator 不符")

    pre_consistency = result.pre_border_consistency
    retained_consistency = result.retained_consistency
    if int(pre_consistency.checked_row_count) != expected_raw_pair_rows:
        raise ValueError("pre-border consistency raw row count 不符")
    if int(pre_consistency.checked_cell_count) != expected_pre_border_cells:
        raise ValueError("pre-border consistency cell count 不符")
    if int(pre_consistency.checked_duplicate_key_count) != expected_raw_multirow_keys:
        raise ValueError("pre-border consistency duplicate key count 不符")
    if int(pre_consistency.feature_count) != len(RUI49_FEATURE_COLUMNS):
        raise ValueError("pre-border consistency feature count 不符")
    if int(retained_consistency.checked_row_count) != expected_retained_raw_pair_rows:
        raise ValueError("retained consistency raw row count 不符")
    if int(retained_consistency.checked_cell_count) != expected_retained_cells:
        raise ValueError("retained consistency cell count 不符")
    if int(retained_consistency.feature_count) != len(RUI49_FEATURE_COLUMNS):
        raise ValueError("retained consistency feature count 不符")


def _build_full_metadata(
    existing: Mapping[str, Any],
    *,
    full_result: Any,
    output_paths: Mapping[str, Path],
    current_paths: Mapping[str, Path],
    generation_id: str,
    expected_pre_border_cells: int,
    expected_retained_cells: int,
    expected_fov_count: int,
    ido_background_qc: Mapping[str, Any] | None = None,
    planned_archived_outputs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """建立 full validated metadata，保留 target/border 與歷史 smoke evidence。

    Args:
        existing: 前一 generation metadata；僅作為歷史 evidence seed。
        full_result: high adapter 產生的 full canonical 與 health 結果。
        output_paths: 本 generation 的 canonical 與 noncanonical output paths。
        current_paths: 既有 safe pipeline output paths。
        generation_id: 本 generation 唯一識別碼。
        expected_pre_border_cells: fail-closed 的 border 前 distinct cell lock。
        expected_retained_cells: fail-closed 的 border 後 distinct cell lock。
        expected_fov_count: fail-closed 的 manifest FOV lock。
        ido_background_qc: 已驗證的 pilot/full background provenance mapping。
        planned_archived_outputs: publisher 將歸檔的舊 canonical paths。

    Returns:
        可安全寫入 metadata 與 REPORT 的 full generation mapping。
    """
    metadata: dict[str, Any] = dict(existing)
    planned_archived = {
        str(key): str(value)
        for key, value in (planned_archived_outputs or {}).items()
    }
    previous_smoke = str(metadata.get("feature_smoke_status", "not_run"))
    smoke_status = (
        "historical_stale"
        if previous_smoke in {"validated", "smoke_validated", "passed"}
        or str(metadata.get("feature_extraction_status", "")) == "smoke_validated"
        else previous_smoke
    )
    target_status = str(metadata.get("target_status", "not_run"))
    target_sensitivity_status = str(
        metadata.get("target_sensitivity_status", "not_run")
    )
    validation_status = dict(metadata.get("validation_status", {}))
    if not validation_status:
        validation_status = {
            "snapshot": "validated",
            "border": str(metadata.get("border_exclusion_status", "not_run")),
            "target": target_status,
            "target_sensitivity": target_sensitivity_status,
        }
    validation_status.update(
        {
            "snapshot": "validated",
            "full_rui49": "validated",
            "full_693_feature_consistency": "validated",
            "feature_health": "validated",
            "feature_smoke": smoke_status,
            "e2": E2_V5_STATUS,
        }
    )

    health = full_result.health
    report = health.report
    near_zero = _v5_near_zero_columns(report)
    nonfinite = tuple(
        report.loc[
            report["status"].astype(str).str.contains("blocked_nonfinite"), "feature"
        ].astype(str)
    )
    fallback_rates = {
        str(key): float(value)
        for key, value in full_result.retained_fallback_rates.items()
    }
    full_payload: dict[str, Any] = {
        "status": "validated",
        "raw_pair_row_count": int(len(full_result.raw_pair_features)),
        "canonical_feature_count": len(RUI49_FEATURE_COLUMNS),
        "pre_border_distinct_cell_count": int(len(full_result.pre_border_features)),
        "retained_cell_count": int(len(full_result.retained_features)),
        "retained_raw_pair_row_count": int(
            full_result.retained_consistency.checked_row_count
        ),
        "fov_count": int(expected_fov_count),
        "elapsed_seconds": float(full_result.total_elapsed_seconds),
        "cell_level_path": str(Path(output_paths["cell_level_rui49_path"]).resolve(strict=False)),
        "cell_level_rui49_path": str(Path(output_paths["cell_level_rui49_path"]).resolve(strict=False)),
        "feature_health_report_path": str(Path(output_paths["feature_health_report_path"]).resolve(strict=False)),
        "healthy_feature_count": len(health.healthy_feature_columns),
        "removed_zero_variance_columns": list(health.removed_columns),
        "near_zero_variance_columns": list(near_zero),
        "near_zero_variance_threshold": 0.05,
        "near_zero_variance_action": "record_only",
        "nonfinite_columns": list(nonfinite),
        "max_fallback_rate": max(fallback_rates.values(), default=0.0),
        "fallback_counts": {
            str(key): int(value)
            for key, value in full_result.retained_fallback_counts.items()
        },
        "fallback_rates": fallback_rates,
        "pre_border_consistency": _consistency_diagnostic_payload(
            full_result.pre_border_consistency
        ),
        "retained_consistency": _consistency_diagnostic_payload(
            full_result.retained_consistency
        ),
        "health": _json_safe(health.to_metadata_payload()),
        "healthy_feature_ranges": _json_safe(full_result.healthy_feature_ranges),
        "processed_fov_count": int(full_result.processed_fov_count),
        "resumed_fov_count": int(full_result.resumed_fov_count),
        "fov_timings": _json_safe(full_result.fov_timings),
        "extractor_fingerprint": str(full_result.extractor_fingerprint),
        "checkpoint_provenance": _json_safe(full_result.checkpoint_provenance),
        "generation_id": generation_id,
        "publish_status": "validated",
    }
    smoke_history = metadata.get("feature_health_smoke")
    if isinstance(smoke_history, Mapping):
        smoke_history = dict(smoke_history)
        smoke_history["status"] = "historical_stale"
        smoke_history["report_path"] = planned_archived.get(
            "feature_health_report_path"
        )
        smoke_history["historical_report_path"] = smoke_history["report_path"]
        metadata["feature_health_smoke"] = smoke_history
    elif smoke_status == "historical_stale":
        metadata["feature_health_smoke"] = {
            "status": "historical_stale",
            "report_path": planned_archived.get("feature_health_report_path"),
        }
    smoke_test = metadata.get("smoke_test")
    if isinstance(smoke_test, Mapping):
        smoke_test_history = dict(smoke_test)
        smoke_test_history["status"] = "historical_stale"
        nested_health = smoke_test_history.get("feature_health")
        if isinstance(nested_health, Mapping):
            nested_health_history = dict(nested_health)
            nested_health_history["status"] = "historical_stale"
            nested_health_history["report_path"] = planned_archived.get(
                "feature_health_report_path"
            )
            nested_health_history["historical_report_path"] = (
                nested_health_history["report_path"]
            )
            smoke_test_history["feature_health"] = nested_health_history
        metadata["smoke_test"] = smoke_test_history
    background_qc = ido_background_qc
    if background_qc is None:
        background_qc = existing.get("ido_background_qc")
    if background_qc is not None:
        if not isinstance(background_qc, Mapping):
            raise ValueError("ido_background_qc 必須是 mapping")
        metadata["ido_background_qc"] = _json_safe(background_qc)
    runtime = _runtime_metadata("full_validated")
    runtime_packages = dict(runtime["packages"])
    metadata.update(
        {
            "generation_id": generation_id,
            "status": "validated",
            "snapshot_validation_status": "validated",
            "feature_smoke_status": smoke_status,
            "feature_extraction_status": "validated",
            "full_rui49_status": "validated",
            "full_693_feature_consistency_status": "validated",
            "target_status": target_status,
            "target_sensitivity_status": target_sensitivity_status,
            "source_write_status": "not_written",
            "validation_status": validation_status,
            "full_rui49": full_payload,
            "full_raw_pair_row_count": full_payload["raw_pair_row_count"],
            "full_pre_border_distinct_cell_count": expected_pre_border_cells,
            "full_retained_cell_count": expected_retained_cells,
            "full_retained_raw_pair_row_count": full_payload[
                "retained_raw_pair_row_count"
            ],
            "full_fov_count": expected_fov_count,
            "feature_health_status": "validated",
            "background_subtraction_method": "rolling_ball_radius_50",
            "background_subtraction_status": "full_validated",
            "background_subtraction_scope": "full_693_fov",
            "haralick_method": "mahotas.features.haralick",
            "haralick_implementation": "mahotas.features.haralick",
            "haralick_distance": 3,
            "haralick_directions_averaged": 4,
            "haralick_quantization_levels": 256,
            "haralick_padding_level": 0,
            "haralick_in_mask_levels": "1..255",
            "haralick_zero_reserved_for_padding": True,
            "python": runtime["python"],
            "python_version": runtime["python"],
            "packages": runtime_packages,
            "package_versions": runtime_packages,
            "runtime_executable": str(Path(sys.executable).resolve()),
            "archived_historical_outputs": {
                **_as_string_mapping(existing.get("archived_historical_outputs")),
                **planned_archived,
            },
            "full_rui49_planned_archived_outputs": planned_archived,
            "deviations": _merge_deviations(
                existing.get("deviations"), FERET_DEVIATION
            ),
            "feature_arms": _feature_arm_metadata(health),
            "canonical_rui_feature_count": len(RUI49_FEATURE_COLUMNS),
            "effective_rui_feature_count": len(health.healthy_feature_columns),
            "removed_zero_variance_columns": list(health.removed_columns),
            "nucleus_feature_count": 17,
            "near_zero_variance_threshold": 0.05,
            "near_zero_variance_action": "record_only",
            "near_zero_variance_columns": list(near_zero),
            "nucleus_sanity": _nucleus_sanity_metadata(
                existing.get("nucleus_sanity"),
                report_path=current_paths.get("nucleus_sanity_report_path"),
            ),
            "e3_arm4_boundary": {
                "status": "mandatory",
                "interpretation": (
                    "Arm 4 只能回答 phase-derived 核代理特徵是否帶入額外資訊；"
                    "未勝過 rui_48 不得推論核沒有用。"
                ),
            },
            "e2": _e2_status_metadata(
                existing.get("e2"),
                per_fov_path=current_paths.get("nucleus_dapi_validation_path"),
                summary_path=current_paths.get(
                    "nucleus_dapi_validation_summary_path"
                ),
            ),
        }
    )
    for field, path in output_paths.items():
        metadata[field] = str(Path(path).resolve(strict=False))
    for field in (
        "group_targets_path",
        "group_target_sensitivity_path",
        "border_exclusion_report_path",
        "cell_dedup_report_path",
        "fov_ido_scores_path",
        "report_path",
    ):
        if field in current_paths and (
            Path(current_paths[field]).exists() or field in metadata
        ):
            metadata[field] = str(Path(current_paths[field]).resolve(strict=False))
    for field in (
        "nucleus_sanity_report_path",
        "feature_redundancy_report_path",
        "cv_metrics_path",
        "oof_predictions_path",
        "hyperparameters_path",
        "hyperparameter_grids_path",
        "feature_importance_path",
        "per_group_residuals_path",
        "cv_checkpoint_dir",
    ):
        if field in current_paths:
            metadata[field] = str(Path(current_paths[field]).resolve(strict=False))
    return metadata


def _json_safe(value: Any) -> Any:
    """將 dataclass、mapping、序列與 numpy scalar 轉成 JSON-safe 值。"""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _as_mapping(value: object) -> Mapping[str, object]:
    """將任意 metadata nested value 安全轉為 mapping。"""
    return value if isinstance(value, Mapping) else {}


def _v5_near_zero_columns(report: pd.DataFrame) -> tuple[str, ...]:
    """依 v5 的 CV < 0.05 計算近零變異清單，並排除零變異欄位。

    這是 metadata/reporting adapter 的讀取規則；不會改寫 feature-health
    producer，也不會把命中的欄位從任何 arm 移除。
    """
    required = {"feature", "relative_std", "is_zero_variance"}
    if not required.issubset(report.columns):
        return ()
    relative = pd.to_numeric(report["relative_std"], errors="coerce")
    zero = report["is_zero_variance"].astype(bool)
    finite = np.isfinite(relative.to_numpy(dtype=float))
    mask = (~zero) & (relative < 0.05) & finite
    return tuple(report.loc[mask, "feature"].astype(str))


def _merge_deviations(
    existing: object, deviation: Mapping[str, object]
) -> list[dict[str, object]]:
    """合併產物級 deviation，按 id 去重並保留既有紀錄。"""
    merged: list[dict[str, object]] = []
    if isinstance(existing, Mapping):
        existing_items: Sequence[object] = (existing,)
    elif isinstance(existing, (list, tuple)):
        existing_items = existing
    else:
        existing_items = ()
    for item in existing_items:
        if isinstance(item, Mapping):
            merged.append({str(key): _json_safe(value) for key, value in item.items()})

    new_id = str(deviation.get("id", ""))
    merged = [item for item in merged if str(item.get("id", "")) != new_id]
    merged.append({str(key): _json_safe(value) for key, value in deviation.items()})
    return merged


def _feature_arm_metadata(health: Any) -> dict[str, dict[str, object]]:
    """建立固定四 arm 的實際 feature-count metadata。"""
    specs = {
        name: {str(key): _json_safe(value) for key, value in spec.items()}
        for name, spec in FEATURE_ARM_SPECS.items()
    }
    healthy = tuple(getattr(health, "healthy_feature_columns", ()))
    if healthy:
        specs["rui_48"]["count"] = len(healthy)
        specs["rui_48"]["feature_columns"] = list(healthy)
        specs["rui_48_plus_nucleus"]["count"] = len(healthy) + 17
    return specs


def _e2_status_metadata(
    existing: object,
    *,
    per_fov_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, object]:
    """回傳 v5 E2 cancellation evidence，不把未拍攝 DAPI 當 validation failure。"""
    result = (
        {str(key): _json_safe(value) for key, value in existing.items()}
        if isinstance(existing, Mapping)
        else {}
    )
    result.update(
        {
            "status": E2_V5_STATUS,
            "reason": E2_V5_REASON,
            "provenance": _json_safe(E2_V5_PROVENANCE),
            "expected_fov_count": 693,
            "validated_fov_count": 0,
            "dapi_acquired": False,
            "dapi_label_cache_available": False,
        }
    )
    if per_fov_path is not None:
        result["per_fov_path"] = str(Path(per_fov_path).resolve(strict=False))
    if summary_path is not None:
        result["summary_path"] = str(Path(summary_path).resolve(strict=False))
    return result


def _nucleus_sanity_metadata(
    existing: object,
    *,
    report_path: Path | None = None,
) -> dict[str, object]:
    """建立 v5 nucleus sanity wiring，近零變異只記錄、不移除。"""
    result = (
        {str(key): _json_safe(value) for key, value in existing.items()}
        if isinstance(existing, Mapping)
        else {}
    )
    result.setdefault("status", "not_run")
    result["near_zero_variance_threshold"] = 0.05
    result["near_zero_variance_action"] = "record_only"
    if report_path is not None:
        result["report_path"] = str(Path(report_path).resolve(strict=False))
    return result


def _as_string_mapping(value: object) -> dict[str, str]:
    """取出歷史 archive mapping，忽略非 mapping 或非 scalar 欄位。"""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _consistency_diagnostic_payload(value: Any) -> dict[str, Any]:
    """保留 consistency producer 的 native keys，不假造不存在的 status。"""
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    elif is_dataclass(value):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        return {"status": "not_run"}
    return _json_safe(payload)


def _full_result_diagnostic(result: Any, failure_kind: str) -> dict[str, Any]:
    """建立不含 DataFrame 的 full result noncanonical 診斷。"""
    health = result.health
    report = health.report
    nonfinite = list(
        report.loc[
            report["status"].astype(str).str.contains("blocked_nonfinite"), "feature"
        ].astype(str)
    )
    return {
        "failure_kind": failure_kind,
        "nonfinite_columns": nonfinite,
        "raw_pair_row_count": int(len(result.raw_pair_features)),
        "pre_border_distinct_cell_count": int(len(result.pre_border_features)),
        "retained_cell_count": int(len(result.retained_features)),
        "pre_border_consistency": _consistency_diagnostic_payload(
            result.pre_border_consistency
        ),
        "retained_consistency": _consistency_diagnostic_payload(
            result.retained_consistency
        ),
        "health": _json_safe(health.to_metadata_payload()),
        "processed_fov_count": int(result.processed_fov_count),
        "resumed_fov_count": int(result.resumed_fov_count),
        "total_elapsed_seconds": float(result.total_elapsed_seconds),
    }


def _full_blocked_diagnostic(error: FullRui49CanonicalBlockedError) -> dict[str, Any]:
    """將 high typed blocked diagnostics 壓縮為 JSON-safe evidence。"""
    diagnostics = error.diagnostics
    payload = {
        "failure_kind": str(diagnostics.failure_kind),
        "nonfinite_columns": list(diagnostics.nonfinite_columns),
        "pre_border_consistency": _consistency_diagnostic_payload(
            diagnostics.pre_border_consistency
        ),
        "retained_consistency": _consistency_diagnostic_payload(
            diagnostics.retained_consistency
        ),
        "health": _json_safe(diagnostics.health.to_metadata_payload()),
        "raw_pair_row_count": int(len(diagnostics.raw_pair_features)),
        "pre_border_distinct_cell_count": int(len(diagnostics.pre_border_features)),
        "retained_cell_count": int(len(diagnostics.retained_features)),
        "processed_fov_count": int(diagnostics.processed_fov_count),
        "resumed_fov_count": int(diagnostics.resumed_fov_count),
    }
    return _json_safe(payload)


def _finish_full_failure(
    output_root: Path,
    *,
    generation_id: str,
    error: Exception,
    output_paths: Mapping[str, Path],
    full_result: Any | None = None,
) -> int:
    """寫 full noncanonical failure evidence，保留舊 canonical bundle。"""
    if full_result is not None:
        failure_kind = "health_blocked" if full_result.health.blocked else "full_result_blocked"
        diagnostic = _full_result_diagnostic(full_result, failure_kind)
    elif isinstance(error, FullRui49CanonicalBlockedError):
        diagnostic = _full_blocked_diagnostic(error)
    else:
        diagnostic = {"failure_kind": "full_rui49_error"}
    diagnostic.update(
        {
            "mode": "extract-rui49",
            "status": "blocked",
            "generation_id": generation_id,
            "source_write_status": "not_written",
            "error": str(error),
            "output_paths": {
                key: str(Path(path).resolve(strict=False))
                for key, path in output_paths.items()
            },
        }
    )
    failure_evidence_path: Path | None = None
    try:
        failure_evidence_path = _write_failed_generation_evidence(
            output_root,
            generation_id=generation_id,
            diagnostic=_json_safe(diagnostic),
        )
    except (OSError, TypeError, ValueError):
        pass
    summary = _full_summary(
        diagnostic,
        output_paths=output_paths,
        status="blocked",
    )
    summary["error"] = str(error)
    if failure_evidence_path is not None:
        summary["failure_evidence_path"] = str(failure_evidence_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"STATUS=blocked: full Rui49 extraction failed: {error}", file=sys.stderr)
    return 2


def _full_summary(
    metadata: Mapping[str, Any],
    *,
    output_paths: Mapping[str, Path],
    status: str,
) -> dict[str, Any]:
    """建立 full CLI 的 machine-readable summary。"""
    full = metadata.get("full_rui49")
    full_mapping = full if isinstance(full, Mapping) else {}
    return {
        "mode": "extract-rui49",
        "status": status,
        "target_status": str(metadata.get("target_status", "not_run")),
        "full_rui49_status": str(metadata.get("full_rui49_status", status)),
        "raw_pair_row_count": full_mapping.get("raw_pair_row_count"),
        "pre_border_distinct_cell_count": full_mapping.get(
            "pre_border_distinct_cell_count"
        ),
        "retained_cell_count": full_mapping.get("retained_cell_count"),
        "fov_count": full_mapping.get("fov_count"),
        "output_paths": {
            key: str(Path(path).resolve(strict=False))
            for key, path in output_paths.items()
        },
    }


def _expected_count(config: Mapping[str, Any], key: str, default: int) -> int:
    expected = config.get("expected", {})
    if not isinstance(expected, Mapping):
        raise ValueError("config expected 必須是 mapping")
    value = expected.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"expected.{key} 必須是非負整數")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected.{key} 必須是非負整數") from error
    if integer < 0:
        raise ValueError(f"expected.{key} 必須是非負整數")
    return integer


def _optional_expected_count(config: Mapping[str, Any], key: str) -> int | None:
    """讀取 measurement fixture 可省略的 v4 lock；正式 safe mode 有 defaults。"""
    expected = config.get("expected", {})
    if not isinstance(expected, Mapping) or key not in expected:
        return None
    return _expected_count(config, key, 0)


def _expected_sparse_fov_keys(
    config: Mapping[str, Any],
    *,
    require_explicit_count: bool = True,
) -> tuple[str, ...]:
    """讀取正式 config 的 exact cells_after==2 image-key roster。"""
    expected = config.get("expected", {})
    if not isinstance(expected, Mapping):
        raise ValueError("config expected 必須是 mapping")
    values = expected.get("sparse_fov_keys")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("expected.sparse_fov_keys 必須是 image_key 序列")
    keys = tuple(str(value).strip() for value in values)
    if any(not value for value in keys) or len(keys) != len(set(keys)):
        raise ValueError("expected.sparse_fov_keys 必須不含空值且不重複")
    if "sparse_fov_count" not in expected:
        if require_explicit_count:
            raise ValueError(
                "expected.sparse_fov_count 必須明確鎖定 sparse_fov_keys 長度"
            )
        expected_count = len(keys)
    else:
        expected_count = expected["sparse_fov_count"]
    try:
        expected_count = int(expected_count)
    except (TypeError, ValueError) as error:
        raise ValueError("expected.sparse_fov_count 必須是整數") from error
    if expected_count != len(keys):
        raise ValueError(
            "expected.sparse_fov_count 必須等於 sparse_fov_keys 長度"
        )
    return keys


def _optional_expected_sparse_fov_keys(
    config: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """讀取 diagnostic fixture 可省略的 sparse roster。"""
    expected = config.get("expected", {})
    if not isinstance(expected, Mapping) or "sparse_fov_keys" not in expected:
        return None
    return _expected_sparse_fov_keys(config, require_explicit_count=False)


def _safe_output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """解析 safe mode 的所有 output paths，支援 top-level/outputs mapping。"""
    defaults = {
        "metadata_path": "immunity/outputs/exp4/run_metadata.json",
        "cell_dedup_report_path": "immunity/outputs/exp4/cell_dedup_report.csv",
        "border_exclusion_report_path": "immunity/outputs/exp4/border_exclusion_report.csv",
        "fov_ido_scores_path": "immunity/outputs/exp4/fov_ido_scores.csv",
        "group_targets_path": "immunity/outputs/exp4/group_targets.csv",
        "group_target_sensitivity_path": "immunity/outputs/exp4/group_target_sensitivity.csv",
        "report_path": "immunity/outputs/exp4/REPORT.md",
        "nucleus_dapi_validation_path": (
            "immunity/outputs/exp4/nucleus_dapi_validation.csv"
        ),
        "nucleus_dapi_validation_summary_path": (
            "immunity/outputs/exp4/nucleus_dapi_validation_summary.csv"
        ),
        "nucleus_sanity_report_path": (
            "immunity/outputs/exp4/nucleus_sanity_report.csv"
        ),
        "feature_redundancy_report_path": (
            "immunity/outputs/exp4/feature_redundancy_report.csv"
        ),
        "leakage_preflight_report_path": (
            "immunity/outputs/exp4/leakage_preflight_report.csv"
        ),
        "cv_metrics_path": "immunity/outputs/exp4/cv_metrics.csv",
        "oof_predictions_path": "immunity/outputs/exp4/oof_predictions.csv",
        "hyperparameters_path": "immunity/outputs/exp4/hyperparameters.csv",
        "hyperparameter_grids_path": (
            "immunity/outputs/exp4/hyperparameter_grids.json"
        ),
        "feature_importance_path": (
            "immunity/outputs/exp4/feature_importance.csv"
        ),
        "per_group_residuals_path": (
            "immunity/outputs/exp4/per_group_residuals.csv"
        ),
        "cv_checkpoint_dir": "immunity/outputs/exp4/.cv_checkpoints",
    }
    outputs = config.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        raise ValueError("config outputs 必須是 mapping")
    paths: dict[str, Path] = {}
    for key, default in defaults.items():
        value = config.get(key)
        if value is None and isinstance(outputs, Mapping):
            value = outputs.get(key)
        if value is None:
            value = default
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"config 缺少非空 {key}")
        path = Path(value).expanduser()
        paths[key] = path if path.is_absolute() else (Path.cwd() / path).resolve(strict=False)
    return paths


def _exploratory_output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """解析 post-CV 專用的七項 artifact、REPORT 與 metadata 路徑。

    此 mapping 刻意不包含既有 CV numeric outputs；呼叫 publisher 時只能以
    這個 restricted mapping 建立 post-CV transaction，避免 safe/CV bundle 被
    意外備份或替換。

    Args:
        config: 已由 YAML 載入的 Exp4 設定 mapping。

    Returns:
        依 ``POST_CV_ANALYSIS_FILE_FIELDS`` 固定順序排列的 canonical paths。

    Raises:
        ValueError: 設定不是 mapping、路徑為空或 output path 型別錯誤。
    """
    safe_paths = _safe_output_paths(config)
    defaults = {
        "shrinkage_analysis_path": "immunity/outputs/exp4/shrinkage_analysis.csv",
        "orientation_target_marginal_path": (
            "immunity/outputs/exp4/orientation_target_marginal.csv"
        ),
        "umap_embeddings_path": "immunity/outputs/exp4/umap_embeddings.csv",
        "kmeans_cluster_features_path": (
            "immunity/outputs/exp4/kmeans_cluster_features.csv"
        ),
        "umap_by_donor_path": (
            "immunity/outputs/exp4/figures/umap_by_donor.png"
        ),
        "umap_by_passage_path": (
            "immunity/outputs/exp4/figures/umap_by_passage.png"
        ),
        "umap_by_condition_path": (
            "immunity/outputs/exp4/figures/umap_by_condition.png"
        ),
    }
    outputs = config.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        raise ValueError("config outputs 必須是 mapping")
    paths: dict[str, Path] = {}
    for field in POST_CV_ARTIFACT_FIELDS:
        value = config.get(field)
        if value is None and isinstance(outputs, Mapping):
            value = outputs.get(field)
        if value is None:
            value = defaults[field]
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"config 缺少非空 {field}")
        candidate = Path(value).expanduser()
        paths[field] = (
            candidate
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve(strict=False)
        )
    paths["report_path"] = safe_paths["report_path"]
    paths["metadata_path"] = safe_paths["metadata_path"]
    return paths


def _full_output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """解析 full Rui49 canonical 與 noncanonical writable paths。"""
    defaults = {
        "metadata_path": "immunity/outputs/exp4/run_metadata.json",
        "report_path": "immunity/outputs/exp4/REPORT.md",
        "cell_level_rui49_path": "immunity/outputs/exp4/cell_level_rui49.csv",
        "feature_health_report_path": "immunity/outputs/exp4/feature_health_report.csv",
        "run_log_path": "immunity/outputs/exp4/run.log",
        "rui49_checkpoint_dir": "immunity/outputs/exp4/.rui49_checkpoints",
    }
    outputs = config.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        raise ValueError("config outputs 必須是 mapping")
    paths: dict[str, Path] = {}
    for key, default in defaults.items():
        value = config.get(key)
        if value is None and isinstance(outputs, Mapping):
            value = outputs.get(key)
        if value is None:
            value = default
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"config 缺少非空 {key}")
        path = Path(value).expanduser()
        paths[key] = (
            path if path.is_absolute() else (Path.cwd() / path).resolve(strict=False)
        )
    return paths


def _ensure_full_output_paths(
    output_paths: Mapping[str, Path], source_root: Path
) -> None:
    """將 full mode 所有 writable paths 限制在 current Exp4 output root。"""
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    for field, candidate in output_paths.items():
        resolved = Path(candidate).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                f"{field} output containment failed: {resolved} not under {output_root}"
            ) from error
        if not _path_is_outside_source_root(resolved, source_root):
            raise ValueError(
                f"{field} output containment failed: output is inside source_root"
            )
    checkpoint = Path(output_paths["rui49_checkpoint_dir"]).resolve(strict=False)
    if not checkpoint.name.startswith("."):
        raise ValueError("rui49_checkpoint_dir 必須是 hidden directory")


def _ensure_safe_output_paths(
    output_paths: Mapping[str, Path],
    source_root: Path,
) -> None:
    """將 safe outputs fail-closed 限制在 current worktree Exp4 root。"""
    output_root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    for field, candidate in output_paths.items():
        resolved = Path(candidate).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                f"{field} output containment failed: {resolved} not under {output_root}"
            ) from error
        if not _path_is_outside_source_root(resolved, source_root):
            raise ValueError(
                f"{field} output containment failed: output is inside source_root"
            )


def _ensure_post_cv_output_paths(
    *,
    output_paths: Mapping[str, Path],
    protected_paths: Mapping[str, Path],
) -> None:
    """拒絕 post-CV output 與任何 protected CV artifact 的 path alias。

    Args:
        output_paths: post-CV 七項 artifact、REPORT 與 metadata 的 writable paths。
        protected_paths: current formal CV output paths，必須包含十項 protected
            artifact fields。

    Raises:
        ValueError: post-CV path 與 protected artifact resolve 到同一 canonical path。
    """
    protected = _protected_cv_artifact_paths(protected_paths)
    protected_by_path = {
        path: name for name, path in protected.items()
    }
    for field, candidate in output_paths.items():
        resolved = Path(candidate).expanduser().resolve(strict=False)
        protected_name = protected_by_path.get(resolved)
        if protected_name is not None:
            raise ValueError(
                "post-CV output path aliases protected artifact: "
                f"{field}={resolved} aliases {protected_name}"
            )


def _is_under_safe_output_root(path: Path) -> bool:
    """判斷 path 是否位於 current worktree 的 Exp4 output root。"""
    root = (Path.cwd() / "immunity" / "outputs" / "exp4").resolve(
        strict=False
    )
    try:
        Path(path).expanduser().resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _planned_archived_outputs(
    output_paths: Mapping[str, Path],
    generation_id: str,
) -> dict[str, str]:
    """預覽本 generation 成功後的 historical archive 路徑，不改變檔案。"""
    return {
        field: str(
            Path(path)
            .expanduser()
            .with_name(f"{Path(path).name}.historical_stale.{generation_id}")
            .resolve(strict=False)
        )
        for field, path in output_paths.items()
        if Path(path).expanduser().exists()
    }


def _rollback_quarantine_path(final: Path, generation_id: str) -> Path:
    """取得 rollback 失敗時使用的 generation-specific noncanonical 路徑。"""
    return final.with_name(f".{final.name}.rollback_quarantine.{generation_id}")


def _publish_bundle(
    staged_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    *,
    fields: Sequence[str],
    generation_id: str,
    post_publish_gate: Callable[[], None] | None = None,
) -> dict[str, str]:
    """以可 rollback 的單一 transaction 發布 Exp4 canonical outputs。

    Args:
        staged_paths: 本 generation staging files；每個 ``fields`` 都必須存在。
        output_paths: canonical output field 到固定 worktree 路徑的 mapping；所有
            fields 都會先備份，包含 blocked bundle 不發布的舊 target 檔。
        fields: 本次要成為 current bundle 的 staged field；未列出的 canonical
            field 會保持不存在，讓 blocked sensitivity 不會沿用舊 target。
        generation_id: 唯一 generation id，用於暫存 backup 與 historical archive。
        post_publish_gate: 已提交新 bundle 後、transaction 返回前執行的驗證 callback；
            callback 失敗時，publisher 會以同一組 backup/archive 還原完整舊 bundle。

    Returns:
        成功歸檔的舊 output field 到 historical archive 路徑。

    Raises:
        BundlePublishError: archive、staged commit 或 historical archive 任一步驟
            失敗；函式會嘗試恢復完整舊 bundle。若移除已 commit 的新檔失敗，
            會先將其移至 generation-specific noncanonical quarantine；連
            quarantine 也失敗時，例外會明確標示 ``FATAL recovery``，不可宣稱
            canonical 狀態已完整恢復。
    """
    canonical_fields = tuple(output_paths)
    publish_fields = tuple(fields)
    if not canonical_fields or any(field not in output_paths for field in publish_fields):
        raise BundlePublishError("bundle fields 不完整或包含未知 output")
    if len(set(Path(path).resolve(strict=False) for path in output_paths.values())) != len(
        output_paths
    ):
        raise BundlePublishError("canonical output paths 不可重複")

    moved: list[tuple[str, Path, Path, Path]] = []
    committed: list[str] = []
    archived: dict[str, str] = {}
    try:
        # First create same-generation backups.  No current pointer changes to
        # new content occur until every existing canonical has been moved.
        for field in canonical_fields:
            final = Path(output_paths[field]).expanduser().resolve(strict=False)
            if not final.exists():
                continue
            if final.is_dir():
                # Checkpoint directories are noncanonical working state; never
                # move or archive them as part of a CSV/JSON bundle transaction.
                continue
            backup = final.with_name(f"{final.name}.backup.{generation_id}")
            archive = final.with_name(
                f"{final.name}.historical_stale.{generation_id}"
            )
            if backup.exists() or archive.exists():
                raise BundlePublishError(
                    f"bundle backup/archive path already exists: {final.name}"
                )
            os.replace(final, backup)
            moved.append((field, final, backup, archive))

        # Commit staged files only after all old canonical files are safely in
        # this transaction's backup set.
        for field in publish_fields:
            staged = Path(staged_paths[field]).expanduser()
            final = Path(output_paths[field]).expanduser().resolve(strict=False)
            if not staged.is_file():
                raise BundlePublishError(f"staged output missing: {field}")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            committed.append(field)

        # Once the full new bundle is current, rename backups to the durable
        # historical names.  A failure here also rolls back the new bundle.
        for field, final, backup, archive in moved:
            os.replace(backup, archive)
            archived[field] = str(archive)
        if post_publish_gate is not None:
            post_publish_gate()
        return archived
    except Exception as error:
        rollback_errors: list[str] = []
        # Remove every newly committed current file before restoring old files.
        for field in reversed(committed):
            final = Path(output_paths[field]).expanduser().resolve(strict=False)
            try:
                final.unlink(missing_ok=True)
            except OSError as rollback_error:
                quarantine = _rollback_quarantine_path(final, generation_id)
                try:
                    if quarantine.exists():
                        raise OSError(
                            f"rollback quarantine path already exists: {quarantine}"
                        )
                    os.replace(final, quarantine)
                    rollback_errors.append(
                        f"remove {field}: {rollback_error}; "
                        f"quarantined new output at {quarantine}"
                    )
                except OSError as quarantine_error:
                    rollback_errors.append(
                        f"FATAL recovery remove {field}: {rollback_error}; "
                        f"rollback quarantine failed: {quarantine_error}; "
                        "canonical may still contain new output"
                    )
        # Restore from either the still-live backup or an already renamed archive.
        for field, final, backup, archive in reversed(moved):
            try:
                source = archive if archive.exists() else backup
                if source.exists():
                    os.replace(source, final)
            except OSError as rollback_error:
                rollback_errors.append(f"restore {field}: {rollback_error}")
        detail = str(error)
        if rollback_errors:
            detail += "; rollback errors: " + " | ".join(rollback_errors)
        if isinstance(error, BundlePublishError) and not rollback_errors:
            raise
        raise BundlePublishError(f"bundle publish failed: {detail}") from error


def publish_analysis_bundle(
    staged_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    *,
    fields: Sequence[str],
    generation_id: str,
    post_publish_gate: Callable[[], None] | None = None,
) -> dict[str, str]:
    """以既有 rollback transaction 發布 high analysis 產物。

    這是 E3 sanity、Pearson redundancy 與 CV adapter 共用的 public seam；
    high 只需把已驗證的 staged tables/metadata/report 與 config-resolved
    paths 傳入，publisher 會維持同一 generation 的原子提交。此 wrapper 不
    執行任何分析，也不允許 reporting layer 偽造結果。
    """
    return _publish_bundle(
        staged_paths,
        output_paths,
        fields=fields,
        generation_id=generation_id,
        post_publish_gate=post_publish_gate,
    )


def _build_safe_metadata(
    existing: Mapping[str, Any],
    *,
    result: BorderExclusionResult,
    output_paths: Mapping[str, Path],
    archived: Mapping[str, str],
    generation_id: str,
    expected_locks: Mapping[str, int],
    status: str,
    target_status: str,
    target_sensitivity_status: str,
) -> dict[str, Any]:
    """建立 safe generation metadata，分離 target 與 feature-smoke 狀態。

    Args:
        existing: 既有 metadata seed；只保留非 current 的歷史欄位。
        result: 本 generation 的 border result，作為所有 count/gate current source。
        output_paths: canonical outputs，用於記錄已發布路徑。
        archived: 本 generation 成功歸檔的舊 output 路徑。
        generation_id: 本 generation 唯一識別碼。
        expected_locks: snapshot、dedup、border 與 FOV 鎖定值。
        status: 本 generation safe pipeline status。
        target_status: current target status。
        target_sensitivity_status: current sensitivity status。

    Returns:
        已以 current border result 覆寫 count/gate 欄位的 metadata mapping。

    Raises:
        ValueError: border result 缺少 raw inclusion mask 或 report schema 不完整。
    """
    raw_mask = result.raw_inclusion_mask
    if raw_mask is None or len(raw_mask) != len(result.raw_cells):
        raise ValueError("current border result 缺少與 raw rows 對齊的 inclusion mask")
    border_report = result.border_report
    try:
        row_before = int(len(result.raw_cells))
        row_excluded = int((~raw_mask).sum())
        row_after = int(raw_mask.sum())
        distinct_before = int(len(result.pre_cells))
        distinct_excluded = int(len(result.excluded_cells))
        distinct_after = int(len(result.eligible_cells))
        whole_before = int(border_report["whole_cell_labels_before"].sum())
        whole_excluded = int(
            border_report["whole_cell_labels_excluded_border"].sum()
        )
        whole_after = int(border_report["whole_cell_labels_after"].sum())
        minimum_whole_after = int(border_report["whole_cell_labels_after"].min())
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("current border result report schema 不完整") from error
    previous_feature = str(existing.get("feature_smoke_status", "not_run"))
    previous_extraction = str(
        existing.get("feature_extraction_status", "not_run")
    )
    previous_report = existing.get("feature_smoke_report")
    previous_report_status = (
        str(previous_report.get("status", ""))
        if isinstance(previous_report, Mapping)
        else ""
    )
    feature_status = (
        "historical_stale"
        if any(
            status in {"validated", "smoke_validated", "passed"}
            for status in (previous_feature, previous_extraction, previous_report_status)
        )
        else "not_run"
    )
    metadata: dict[str, Any] = dict(existing)
    metadata.update(
        {
            "generation_id": generation_id,
            "status": status,
            "border_exclusion_status": "validated",
            "border_exclusion_gate_status": result.minimum_cell_gate_status,
            "minimum_cell_gate_status": result.minimum_cell_gate_status,
            "target_status": target_status,
            "target_sensitivity_status": target_sensitivity_status,
            "feature_smoke_status": feature_status,
            "feature_extraction_status": feature_status,
            "source_write_status": "not_written",
            "expected_locks": dict(expected_locks),
            "raw_cell_count": row_before,
            "cells_before_border_exclusion": row_before,
            "cells_excluded_border": row_excluded,
            "cells_after_border_exclusion_raw_pairs": row_after,
            "distinct_cells_before_border_exclusion": distinct_before,
            "distinct_cells_excluded_border": distinct_excluded,
            "distinct_cells_after_border_exclusion": distinct_after,
            "cells_after_border_exclusion": distinct_after,
            "dedup_cell_count": distinct_before,
            "post_border_cell_count": distinct_after,
            "whole_cell_labels_before_border_exclusion": whole_before,
            "whole_cell_labels_excluded_border": whole_excluded,
            "whole_cell_labels_after_border_exclusion": whole_after,
            "minimum_whole_cell_labels_after_border_exclusion": minimum_whole_after,
            "duplicate_whole_cell_key_count": result.duplicate_key_count,
            "duplicate_key_count": result.duplicate_key_count,
            "duplicate_excess_row_count": result.duplicate_excess_row_count,
            "mask_only_label_total": result.mask_only_label_total,
            "mask_only_label_max_per_fov": result.mask_only_label_max_per_fov,
            "fov_count": len(result.manifest),
            "sparse_fov_keys": list(
                result.expected_sparse_fov_keys or _inferred_sparse_keys(result)
            ),
            "archived_historical_outputs": dict(archived),
            "validation_status": {
                "snapshot": "validated",
                "border": "validated",
                "target": target_status,
                "target_sensitivity": target_sensitivity_status,
                "feature_smoke": feature_status,
            },
        }
    )
    for field in (
        "cell_dedup_report_path",
        "border_exclusion_report_path",
        "fov_ido_scores_path",
        "group_target_sensitivity_path",
        "report_path",
    ):
        if (
            field == "group_target_sensitivity_path"
            and target_sensitivity_status == "not_run_unit_fixture"
        ):
            continue
        if field in output_paths:
            metadata[field] = str(Path(output_paths[field]).resolve(strict=False))
    if target_status == "validated" and "group_targets_path" in output_paths:
        metadata["group_targets_path"] = str(
            Path(output_paths["group_targets_path"]).resolve(strict=False)
        )
    else:
        metadata.pop("group_targets_path", None)
        metadata["stale_group_targets_status"] = "historical_stale"
    return metadata


def _write_failed_generation_evidence(
    output_root: Path,
    *,
    generation_id: str,
    diagnostic: Mapping[str, object],
) -> Path:
    """寫入唯一 noncanonical failure evidence，不覆寫 current bundle。

    Args:
        output_root: current worktree 的 Exp4 output root。
        generation_id: 失敗 generation 的唯一識別碼。
        diagnostic: 可序列化的 blocked 診斷內容。

    Returns:
        noncanonical ``.failed-generation.<id>.json`` 路徑。

    Raises:
        ValueError: JSON atomic write 失敗。
    """
    evidence_path = output_root / f".failed-generation.{generation_id}.json"
    _write_json_atomic(diagnostic, evidence_path)
    return evidence_path


def _inferred_sparse_keys(result: BorderExclusionResult) -> tuple[str, ...]:
    """由 border report 推導 cells_after==2 roster。"""
    return tuple(
        result.border_report.loc[
            result.border_report["cells_after"].astype(int).eq(2), "image_key"
        ].astype(str)
    )


def _path_is_outside_source_root(path: Path, source_root: Path) -> bool:
    """回傳 output 是否不在唯讀 source root 子樹內。"""
    try:
        path.expanduser().resolve(strict=False).relative_to(
            source_root.expanduser().resolve(strict=False)
        )
    except ValueError:
        return True
    return False


def _read_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata JSON root 必須是 object")
    return dict(value)


def _write_json_atomic(value: Mapping[str, object], path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"metadata atomic write failed: {path}") from error


def _safe_summary(
    result: BorderExclusionResult | None,
    *,
    metadata: Mapping[str, object],
    output_paths: Mapping[str, Path],
    status: str,
) -> dict[str, Any]:
    """建立 safe mode machine-readable summary。"""
    summary: dict[str, Any] = {
        "mode": "run-safe",
        "status": status,
        "target_status": str(metadata.get("target_status", "blocked")),
        "feature_smoke_status": str(metadata.get("feature_smoke_status", "not_run")),
        "metadata_path": str(output_paths["metadata_path"].resolve(strict=False))
        if "metadata_path" in output_paths
        else None,
        "output_paths": {
            key: str(path.resolve(strict=False)) for key, path in output_paths.items()
        },
    }
    if result is not None:
        summary.update(
            {
                "raw_cell_count": len(result.raw_cells),
                "dedup_cell_count": len(result.pre_cells),
                "duplicate_key_count": result.duplicate_key_count,
                "post_border_cell_count": len(result.eligible_cells),
                "fov_count": len(result.manifest),
                "minimum_cell_gate_status": result.minimum_cell_gate_status,
            }
        )
    return summary


def _border_summary(
    result: BorderExclusionResult,
    *,
    report_path: Path | None,
    targets_path: Path | None,
    metadata_path: Path | None,
    status: str,
) -> dict[str, Any]:
    """建立 stdout 用的 machine-readable border summary。"""
    report = result.border_report
    return {
        "mode": "measure-border-exclusion",
        "status": status,
        "cells_before_border_exclusion": int(len(result.pre_cells)),
        "cells_excluded_border": int(len(result.excluded_cells)),
        "cells_after_border_exclusion": int(len(result.eligible_cells)),
        "whole_cell_labels_before_border_exclusion": int(
            report["whole_cell_labels_before"].sum()
        ),
        "whole_cell_labels_excluded_border": int(
            report["whole_cell_labels_excluded_border"].sum()
        ),
        "whole_cell_labels_after_border_exclusion": int(
            report["whole_cell_labels_after"].sum()
        ),
        "fov_count": int(len(result.manifest)),
        "minimum_whole_cell_labels_after_border_exclusion": int(
            report["whole_cell_labels_after"].min()
        ),
        "minimum_cell_gate_status": result.minimum_cell_gate_status,
        "mask_only_label_total": int(result.mask_only_label_total),
        "mask_only_label_max_per_fov": int(result.mask_only_label_max_per_fov),
        "report_path": str(report_path.resolve(strict=False))
        if report_path is not None
        else None,
        "border_exclusion_report_path": str(report_path.resolve(strict=False))
        if report_path is not None
        else None,
        "targets_path": str(targets_path.resolve(strict=False))
        if targets_path is not None
        else None,
        "group_targets_path": str(targets_path.resolve(strict=False))
        if targets_path is not None
        else None,
        "metadata_path": str(metadata_path.resolve(strict=False))
        if metadata_path is not None
        else None,
    }


def _load_config(path: Path) -> Mapping[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("config root 必須是 mapping")
    return loaded


def _resolve_config_path(config: Mapping[str, Any], key: str, base: Path) -> Path:
    value = config.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"config 缺少非空 {key}")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve(strict=False)


def _metadata_path(config: Mapping[str, Any]) -> Path:
    value = config.get("metadata_path", "immunity/outputs/exp4/run_metadata.json")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve(strict=False)


def _output_path(config: Mapping[str, Any], key: str, default: str) -> Path:
    """解析 border output path，兼容 top-level 與 outputs mapping。"""
    outputs = config.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        raise ValueError("config outputs 必須是 mapping")
    value = config.get(key)
    if value is None and isinstance(outputs, Mapping):
        value = outputs.get(key)
    if value is None:
        value = default
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"config 缺少非空 {key}")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve(strict=False)


def _ensure_output_outside_source_root(
    path: Path, source_root: Path, field_name: str
) -> None:
    """拒絕將 Exp4 產物寫入唯讀 Exp3 source root 或其子樹。"""
    candidate = path.expanduser().resolve(strict=False)
    root = source_root.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    raise ValueError(
        f"{field_name} must be outside source_root: "
        f"{candidate} is under {root}"
    )


def _runtime_metadata(status: str) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "rui_sklearn_reference": "1.3.1",
        "seed": 42,
        "background_subtraction_method": None,
        "background_subtraction_status": "pending",
        "status": status,
    }


def _write_metadata(path: Path, status: str, snapshot: SnapshotResult) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("run_metadata.json root 必須是 object")
        existing = loaded
    runtime = _runtime_metadata(status)
    if existing.get("background_subtraction_method"):
        runtime["background_subtraction_method"] = existing[
            "background_subtraction_method"
        ]
        runtime["background_subtraction_status"] = existing.get(
            "background_subtraction_status", "implemented"
        )
    previous_feature_status = str(existing.get("feature_smoke_status", ""))
    feature_smoke_status = (
        "historical_stale"
        if previous_feature_status in {"validated", "smoke_validated", "passed"}
        or existing.get("feature_extraction_status") == "smoke_validated"
        else "not_run"
    )
    metadata = {
        **existing,
        **runtime,
        "snapshot_validation_status": (
            "validated" if snapshot.matched else "blocked"
        ),
        "snapshot_validation": snapshot.to_dict(),
        "validation_status": {
            "snapshot": "validated" if snapshot.matched else "blocked",
            "feature_smoke": feature_smoke_status,
        },
        "target_status": "not_run",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
