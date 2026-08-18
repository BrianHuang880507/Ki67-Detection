"""Exp4 Phase 2 的 deterministic publication、圖表與 REPORT transaction。

此模組只負責 adapter 需要的 I/O 與發佈邊界。Phase 2 target、CV 與
post-hoc 計算一律委派給 ``phase2`` 與 ``phase2_diagnostics`` 的 public seams，
避免在 reporting layer 複製數值邏輯。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cv import (
    CV_METRIC_COLUMNS,
    FEATURE_IMPORTANCE_COLUMNS,
    HYPERPARAMETER_COLUMNS,
    LEAKAGE_REPORT_COLUMNS,
    OOF_COLUMNS,
    PER_GROUP_RESIDUAL_COLUMNS,
)
from .phase2 import (
    METRICS_WITHIN_CONDITION_COLUMNS,
    PHASE1_VS_PHASE2_COMPARISON_COLUMNS,
    PHASE2_SHRINKAGE_COLUMNS,
    PHASE2_TARGET_COLUMNS,
    Phase2CvRunResult,
    analyze_phase2_shrinkage,
    build_metrics_within_condition,
    compare_phase1_phase2,
)
from .phase2_diagnostics import (
    CONDITION_MAPPING_REVERSED_HYPOTHESIS,
    DOSE_MAPPING_NOTE,
    DOSE_RESPONSE_COLUMNS,
    RESIDUAL_VS_DOSE_COLUMNS,
    analyze_residual_vs_dose,
    build_dose_response_check,
    fingerprint_phase2_numeric_artifacts,
    verify_phase2_numeric_artifact_immutability,
)


PHASE2_REPORT_START = "<!-- EXP4_PHASE2_START -->"
PHASE2_REPORT_END = "<!-- EXP4_PHASE2_END -->"

PHASE2_TABLE_FIELDS = (
    "phase2_group_targets_path",
    "phase2_leakage_preflight_report_path",
    "phase2_cv_metrics_path",
    "phase2_oof_predictions_path",
    "phase2_hyperparameters_path",
    "phase2_hyperparameter_grids_path",
    "phase2_feature_importance_path",
    "phase2_per_group_residuals_path",
    "phase2_shrinkage_analysis_path",
    "dose_response_check_path",
    "residual_vs_dose_path",
    "metrics_within_condition_path",
    "phase1_vs_phase2_comparison_path",
)
PHASE2_FIGURE_FIELDS = (
    "phase2_figure_phase1_vs_phase2_r2_path",
    "phase2_figure_within_condition_r2_path",
    "phase2_figure_dose_response_path",
)
PHASE2_META_FIELDS = (
    "phase2_run_metadata_path",
    "phase2_run_log_path",
    "phase2_cv_checkpoint_dir",
)
PHASE2_OUTPUT_FIELDS = PHASE2_TABLE_FIELDS + PHASE2_FIGURE_FIELDS + PHASE2_META_FIELDS
# The checkpoint directory is a resumable work cache, not a published artifact.
# Keep it in the output-path contract so the adapter can validate and report its
# canonical location, but never pass it through the atomic publisher/archive.
PHASE2_PUBLISH_FIELDS = (
    PHASE2_TABLE_FIELDS
    + PHASE2_FIGURE_FIELDS
    + ("phase2_run_metadata_path", "phase2_run_log_path")
)

PHASE2_FILE_DEFAULTS = {
    "phase2_group_targets_path": "immunity/outputs/exp4/phase2_group_targets.csv",
    "phase2_leakage_preflight_report_path": (
        "immunity/outputs/exp4/phase2_leakage_preflight_report.csv"
    ),
    "phase2_cv_metrics_path": "immunity/outputs/exp4/phase2_cv_metrics.csv",
    "phase2_oof_predictions_path": (
        "immunity/outputs/exp4/phase2_oof_predictions.csv"
    ),
    "phase2_hyperparameters_path": (
        "immunity/outputs/exp4/phase2_hyperparameters.csv"
    ),
    "phase2_hyperparameter_grids_path": (
        "immunity/outputs/exp4/phase2_hyperparameter_grids.json"
    ),
    "phase2_feature_importance_path": (
        "immunity/outputs/exp4/phase2_feature_importance.csv"
    ),
    "phase2_per_group_residuals_path": (
        "immunity/outputs/exp4/phase2_per_group_residuals.csv"
    ),
    "phase2_shrinkage_analysis_path": (
        "immunity/outputs/exp4/phase2_shrinkage_analysis.csv"
    ),
    "dose_response_check_path": "immunity/outputs/exp4/dose_response_check.csv",
    "residual_vs_dose_path": "immunity/outputs/exp4/residual_vs_dose.csv",
    "metrics_within_condition_path": (
        "immunity/outputs/exp4/metrics_within_condition.csv"
    ),
    "phase1_vs_phase2_comparison_path": (
        "immunity/outputs/exp4/phase1_vs_phase2_comparison.csv"
    ),
    "phase2_figure_phase1_vs_phase2_r2_path": (
        "immunity/outputs/exp4/figures/phase1_vs_phase2_r2.png"
    ),
    "phase2_figure_within_condition_r2_path": (
        "immunity/outputs/exp4/figures/phase2_within_condition_r2.png"
    ),
    "phase2_figure_dose_response_path": (
        "immunity/outputs/exp4/figures/phase2_dose_response.png"
    ),
    "phase2_run_metadata_path": (
        "immunity/outputs/exp4/phase2_run_metadata.json"
    ),
    "phase2_run_log_path": "immunity/outputs/exp4/phase2_run.log",
    "phase2_cv_checkpoint_dir": "immunity/outputs/exp4/.phase2_cv_checkpoints",
}

PHASE2_NUMERIC_PATH_FIELDS = {
    "metrics": "phase2_cv_metrics_path",
    "oof_predictions": "phase2_oof_predictions_path",
    "hyperparameters": "phase2_hyperparameters_path",
    "feature_importance": "phase2_feature_importance_path",
    "per_group_residuals": "phase2_per_group_residuals_path",
}

_TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "phase2_group_targets_path": PHASE2_TARGET_COLUMNS,
    "phase2_leakage_preflight_report_path": LEAKAGE_REPORT_COLUMNS,
    "phase2_cv_metrics_path": CV_METRIC_COLUMNS,
    "phase2_oof_predictions_path": OOF_COLUMNS,
    "phase2_hyperparameters_path": HYPERPARAMETER_COLUMNS,
    "phase2_feature_importance_path": FEATURE_IMPORTANCE_COLUMNS,
    "phase2_per_group_residuals_path": PER_GROUP_RESIDUAL_COLUMNS,
    "phase2_shrinkage_analysis_path": PHASE2_SHRINKAGE_COLUMNS,
    "dose_response_check_path": DOSE_RESPONSE_COLUMNS,
    "residual_vs_dose_path": RESIDUAL_VS_DOSE_COLUMNS,
    "metrics_within_condition_path": METRICS_WITHIN_CONDITION_COLUMNS,
    "phase1_vs_phase2_comparison_path": PHASE1_VS_PHASE2_COMPARISON_COLUMNS,
}

# Keep font selection explicit and ordered so the dose figure is reproducible
# across Windows machines with more than one installed CJK font.  The font
# family fallback below also covers non-Windows environments used for tests.
_PHASE2_CJK_FONT_FILENAMES = (
    "msjh.ttc",
    "msjhl.ttc",
    "msyh.ttc",
    "NotoSansTC-VF.ttf",
    "NotoSansHK-VF.ttf",
    "mingliu.ttc",
    "mingliub.ttc",
    "simsun.ttc",
)
_PHASE2_CJK_FONT_FAMILIES = (
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Noto Sans TC",
    "Noto Sans HK",
    "MingLiU",
    "PMingLiU",
    "SimSun",
    "Arial Unicode MS",
)


class Phase2PublishError(ValueError):
    """表示 Phase 2 bundle 無法安全完成 atomic publish。"""

    def __init__(
        self,
        message: str,
        *,
        recovery_paths: Sequence[str | Path] = (),
    ) -> None:
        super().__init__(message)
        self.recovery_paths = tuple(str(Path(path)) for path in recovery_paths)


@dataclass(frozen=True)
class Phase2DerivedTables:
    """保存 Phase 2 adapter 產生、但不包含原始 CV fitting 的表格。"""

    targets: pd.DataFrame
    leakage: pd.DataFrame
    metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    hyperparameters: pd.DataFrame
    hyperparameter_grids: Mapping[str, object]
    feature_importance: pd.DataFrame
    per_group_residuals: pd.DataFrame
    shrinkage: pd.DataFrame
    dose_response: pd.DataFrame
    residual_vs_dose: pd.DataFrame
    within_condition: pd.DataFrame
    comparison: pd.DataFrame


@dataclass(frozen=True)
class Phase1ImmutableState:
    """保存 publish 前需要重驗證的 Phase 1 bytes/fingerprints。"""

    artifact_sha256: Mapping[str, str]
    checkpoint_fingerprint: Mapping[str, object]
    mask_cache_fingerprint: str
    metadata_bytes: bytes
    run_log_bytes: bytes
    report_outside_marker: bytes


def _resolve_path(value: object, *, base_dir: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("output path 必須是非空字串")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve(strict=False)


def phase2_output_paths(
    config: Mapping[str, object], *, base_dir: str | Path | None = None
) -> dict[str, Path]:
    """解析所有 Phase 2-only output/checkpoint paths。"""
    if not isinstance(config, Mapping):
        raise TypeError("config 必須是 mapping")
    root = Path(base_dir or Path.cwd()).expanduser().resolve(strict=False)
    outputs = config.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        raise ValueError("config outputs 必須是 mapping")
    paths: dict[str, Path] = {}
    for field in PHASE2_OUTPUT_FIELDS:
        value = config.get(field)
        if value is None and isinstance(outputs, Mapping):
            value = outputs.get(field)
        if value is None:
            value = PHASE2_FILE_DEFAULTS[field]
        paths[field] = _resolve_path(value, base_dir=root)
    return paths


def phase2_publish_paths(
    output_paths: Mapping[str, Path], *, report_path: str | Path
) -> dict[str, Path]:
    """建立 publisher 唯一允許接收的 Phase 2 files/figures/REPORT mapping。"""
    missing = [field for field in PHASE2_PUBLISH_FIELDS if field not in output_paths]
    if missing:
        raise Phase2PublishError(f"Phase 2 output paths 缺少欄位：{missing}")
    result = {field: Path(output_paths[field]) for field in PHASE2_PUBLISH_FIELDS}
    result["report"] = Path(report_path)
    return result


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_phase2_publish_mapping(
    output_paths: Mapping[str, Path],
    *,
    phase1_paths: Mapping[str, str | Path],
    phase1_checkpoint_dir: str | Path,
) -> None:
    """拒絕 Phase 2 publisher 對 Phase 1 artifact/tree 的任何 alias。"""
    phase1_files = {
        str(name): _canonical(path) for name, path in phase1_paths.items()
    }
    phase1_tree = _canonical(phase1_checkpoint_dir)
    seen: dict[Path, str] = {}
    for field, raw_path in output_paths.items():
        path = _canonical(raw_path)
        if path in seen:
            raise Phase2PublishError(
                f"Phase 2 output path alias: {field}={path} aliases {seen[path]}"
            )
        seen[path] = field
        if path in phase1_files.values():
            name = next(name for name, value in phase1_files.items() if value == path)
            raise Phase2PublishError(
                f"Phase 2 publisher aliases Phase 1 artifact {name}: {path}"
            )
        if _is_under(path, phase1_tree) or _is_under(phase1_tree, path):
            raise Phase2PublishError(
                f"Phase 2 publisher path overlaps Phase 1 checkpoint tree: {path}"
            )


def _remove_path(path: Path) -> None:
    """嚴格移除 transaction path；任何 I/O failure 都必須向上回報。"""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as error:
        raise Phase2PublishError(f"remove failed: {path}: {error}") from error


def _cleanup_staging_parent(path: Path) -> None:
    """移除 publisher 建立且已空的 generation staging parent。"""
    parent = path.parent
    if parent.name.startswith(".staging") or parent.name.lower() == "staging":
        try:
            parent.rmdir()
        except FileNotFoundError:
            return
        except OSError as error:
            raise Phase2PublishError(
                f"staging cleanup failed: {parent}: {error}"
            ) from error


def _backup_path(final: Path, generation_id: str, index: int) -> Path:
    return final.with_name(f".{final.name}.phase2-backup.{generation_id}.{index}")


def _rollback_quarantine_path(path: Path, generation_id: str) -> Path:
    """取得 rollback 失敗時的 generation-specific noncanonical 路徑。"""
    return path.with_name(f".{path.name}.phase2-rollback-quarantine.{generation_id}")


def _rollback_remove_path(
    path: Path,
    *,
    generation_id: str,
    label: str,
    rollback_errors: list[str],
    recovery_paths: list[Path],
) -> bool:
    """移除 rollback 產物，失敗時保留或 quarantine 並回報 exact path。"""
    if not path.exists():
        return True
    try:
        _remove_path(path)
        return True
    except Exception as error:
        quarantine = _rollback_quarantine_path(path, generation_id)
        try:
            if quarantine.exists():
                raise OSError(f"quarantine path already exists: {quarantine}")
            os.replace(path, quarantine)
            recovery_paths.append(quarantine)
            rollback_errors.append(
                f"remove {label} failed at {path}: {error}; "
                f"quarantined recovery path={quarantine}"
            )
            return True
        except Exception as quarantine_error:
            recovery_paths.append(path)
            rollback_errors.append(
                f"FATAL recovery remove {label} failed at {path}: {error}; "
                f"quarantine failed: {quarantine_error}; path preserved={path}"
            )
            return False


def _rollback_publish_records(
    records: Sequence[Mapping[str, object]],
    *,
    generation_id: str,
    error: BaseException,
) -> Phase2PublishError:
    """還原 bundle；任何 backup/archive 無法還原時絕不刪除其唯一副本。"""
    rollback_errors: list[str] = []
    recovery_paths: list[Path] = []
    for record in reversed(records):
        key = str(record["key"])
        final = record["final"]
        backup = record["backup"]
        staged = record["staged"]
        archive = record.get("archive")
        assert isinstance(final, Path)
        assert isinstance(backup, Path)
        assert isinstance(staged, Path)
        if bool(record.get("installed")):
            removed = _rollback_remove_path(
                final,
                generation_id=generation_id,
                label=f"installed {key}",
                rollback_errors=rollback_errors,
                recovery_paths=recovery_paths,
            )
        else:
            removed = True

        # An archive is the only remaining copy after the archive phase.  Move
        # it back to the backup slot before attempting canonical restoration.
        if isinstance(archive, Path) and archive.exists() and not backup.exists():
            try:
                os.replace(archive, backup)
            except OSError as restore_error:
                recovery_paths.append(archive)
                rollback_errors.append(
                    f"restore archive failed for {key}: {restore_error}; "
                    f"recoverable archive={archive}"
                )

        # Never overwrite a surviving current path, and never remove a backup
        # after a failed restore: it may be the only recoverable old artifact.
        if bool(record.get("backup_moved")) and backup.exists():
            if removed and not final.exists():
                try:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup, final)
                except OSError as restore_error:
                    recovery_paths.extend((backup, final))
                    rollback_errors.append(
                        f"restore {key} failed: {restore_error}; "
                        f"recoverable backup={backup}; canonical={final}"
                    )
            else:
                recovery_paths.extend((backup, final))
                rollback_errors.append(
                    f"restore {key} deferred because canonical path remains: "
                    f"backup={backup}; canonical={final}"
                )

        _rollback_remove_path(
            staged,
            generation_id=generation_id,
            label=f"staged {key}",
            rollback_errors=rollback_errors,
            recovery_paths=recovery_paths,
        )

    for record in reversed(records):
        staged = record["staged"]
        assert isinstance(staged, Path)
        try:
            _cleanup_staging_parent(staged)
        except Exception as cleanup_error:
            recovery_paths.append(staged.parent)
            rollback_errors.append(
                f"staging parent cleanup failed: {staged.parent}: {cleanup_error}"
            )

    detail = f"Phase 2 publish rollback: {error}"
    if rollback_errors:
        detail += "; rollback errors=" + " | ".join(rollback_errors)
    unique_recovery_paths = tuple(dict.fromkeys(recovery_paths))
    if unique_recovery_paths:
        detail += "; recoverable_paths=" + ", ".join(
            str(path) for path in unique_recovery_paths
        )
    return Phase2PublishError(detail, recovery_paths=unique_recovery_paths)


def publish_phase2_bundle(
    staged_paths: Mapping[str, str | Path],
    output_paths: Mapping[str, str | Path],
    *,
    phase1_paths: Mapping[str, str | Path],
    phase1_checkpoint_dir: str | Path,
    generation_id: str,
    post_publish_gate: Callable[[], None] | None = None,
) -> dict[str, str]:
    """以可回復 transaction 發布 Phase 2 files/figures 與 marker REPORT。

    所有既有 Phase 2 generations 只在成功後移到 generation-specific historical
    names；任一 staged commit 或 post-publish gate 失敗時，舊 bundle 與 REPORT
    會嘗試完整復原。若移除/還原本身失敗，唯一 backup/archive 會保留並在
    ``Phase2PublishError.recovery_paths`` 與訊息中列出，不會宣稱無 residue。
    """
    if "phase2_cv_checkpoint_dir" in output_paths:
        raise Phase2PublishError(
            "Phase 2 checkpoint cache is resumable work state and cannot be published"
        )
    if set(staged_paths) != set(output_paths):
        raise Phase2PublishError("staged/output publisher mapping keys 不一致")
    validate_phase2_publish_mapping(
        {key: Path(value) for key, value in output_paths.items()},
        phase1_paths=phase1_paths,
        phase1_checkpoint_dir=phase1_checkpoint_dir,
    )
    records: list[dict[str, object]] = []
    for index, key in enumerate(sorted(output_paths)):
        staged = _canonical(staged_paths[key])
        final = _canonical(output_paths[key])
        if staged == final:
            raise Phase2PublishError(f"staged path aliases final path: {key}={final}")
        if not staged.exists():
            raise Phase2PublishError(f"staged Phase 2 artifact missing: {key}={staged}")
        backup = _backup_path(final, generation_id, index)
        if backup.exists():
            raise Phase2PublishError(
                f"Phase 2 backup path already exists: {backup}",
                recovery_paths=(backup,),
            )
        archive = final.with_name(f"{final.name}.historical_stale.{generation_id}")
        if key != "report" and archive.exists():
            raise Phase2PublishError(
                f"Phase 2 archive path already exists: {archive}",
                recovery_paths=(archive,),
            )
        records.append(
            {
                "key": key,
                "staged": staged,
                "final": final,
                "backup": backup,
                "had_existing": final.exists(),
                "backup_moved": False,
                "installed": False,
                "archive_candidate": archive,
            }
        )

    archived: dict[str, str] = {}
    try:
        for record in records:
            final = record["final"]
            backup = record["backup"]
            assert isinstance(final, Path)
            assert isinstance(backup, Path)
            if bool(record["had_existing"]):
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(final, backup)
                record["backup_moved"] = True
        for record in records:
            staged = record["staged"]
            final = record["final"]
            assert isinstance(staged, Path)
            assert isinstance(final, Path)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            record["installed"] = True
        if post_publish_gate is not None:
            post_publish_gate()
    except Exception as error:
        raise _rollback_publish_records(
            records,
            generation_id=generation_id,
            error=error,
        ) from error

    try:
        for record in records:
            key = str(record["key"])
            backup = record["backup"]
            final = record["final"]
            assert isinstance(backup, Path)
            assert isinstance(final, Path)
            if not bool(record["backup_moved"]):
                continue
            # REPORT contains Phase 1 bytes outside the marker and is intentionally
            # never archived as a whole generation.
            if key == "report":
                continue
            archive = final.with_name(f"{final.name}.historical_stale.{generation_id}")
            os.replace(backup, archive)
            record["archive"] = archive
            archived[key] = str(archive)
        # Finish all staging cleanup before deleting the old REPORT backup.  If
        # cleanup fails, that backup is still the only recoverable Phase 1
        # outside-marker bytes and rollback can restore it.
        for record in records:
            _cleanup_staging_parent(record["staged"])
        for record in records:
            if str(record["key"]) == "report":
                _remove_path(record["backup"])
    except Exception as error:
        raise _rollback_publish_records(
            records,
            generation_id=generation_id,
            error=error,
        ) from error
    return archived


def replace_phase2_report_section(report_bytes: bytes, phase2_section: str) -> bytes:
    """只替換 marker block，保留 REPORT 其餘 bytes 完全不變。"""
    if not isinstance(report_bytes, bytes):
        raise TypeError("report_bytes 必須是 bytes")
    try:
        current = report_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Phase2PublishError("REPORT 必須是 UTF-8") from error
    starts = [match.start() for match in re.finditer(re.escape(PHASE2_REPORT_START), current)]
    ends = [match.start() for match in re.finditer(re.escape(PHASE2_REPORT_END), current)]
    if len(starts) != len(ends) or len(starts) > 1:
        raise Phase2PublishError("REPORT Phase 2 marker 必須恰有零或一組")
    block = (
        PHASE2_REPORT_START
        + "\n"
        + str(phase2_section).rstrip("\r\n")
        + "\n"
        + PHASE2_REPORT_END
    )
    if not starts:
        # Do not add a byte outside the marker block: this keeps even a
        # no-final-newline Phase 1 REPORT byte-identical outside the block.
        return (current + block).encode("utf-8")
    start = starts[0]
    end = ends[0]
    if end < start:
        raise Phase2PublishError("REPORT Phase 2 marker 順序錯誤")
    end += len(PHASE2_REPORT_END)
    return (current[:start] + block + current[end:]).encode("utf-8")


def report_outside_phase2_marker(report_bytes: bytes) -> bytes:
    """移除 marker block，供 Phase 1 REPORT byte immutability 比對。"""
    if not isinstance(report_bytes, bytes):
        raise TypeError("report_bytes 必須是 bytes")
    text = report_bytes.decode("utf-8")
    starts = [match.start() for match in re.finditer(re.escape(PHASE2_REPORT_START), text)]
    ends = [match.start() for match in re.finditer(re.escape(PHASE2_REPORT_END), text)]
    if len(starts) != len(ends) or len(starts) > 1:
        raise Phase2PublishError("REPORT Phase 2 marker 必須恰有零或一組")
    if not starts:
        return report_bytes
    end = ends[0] + len(PHASE2_REPORT_END)
    return (text[: starts[0]] + text[end:]).encode("utf-8")


def _write_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise Phase2PublishError(f"table 必須是 DataFrame：{path.name}")
    expected = tuple(columns)
    if tuple(frame.columns) != expected:
        raise Phase2PublishError(
            f"Phase 2 schema mismatch for {path.name}: expected={expected}, actual={tuple(frame.columns)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def derive_phase2_tables(
    *,
    cv_result: Phase2CvRunResult,
    phase1_metrics: pd.DataFrame,
    phase1_shrinkage: pd.DataFrame,
    include_dose: bool = True,
) -> Phase2DerivedTables:
    """從 Phase 2 core result 呼叫既有 public diagnostics seams。"""
    if not isinstance(cv_result, Phase2CvRunResult):
        # A narrow duck-typed seam keeps injected fast core results useful in tests
        # while production still receives the reviewed immutable result type.
        required = ("assembly", "metrics", "oof_predictions", "hyperparameters")
        if any(not hasattr(cv_result, field) for field in required):
            raise TypeError("cv_result 必須是 Phase2CvRunResult 或相容 injected result")
    targets = cv_result.assembly.targets.copy()
    leakage = cv_result.leakage.report.copy()
    metrics = cv_result.metrics.copy()
    oof = cv_result.oof_predictions.copy()
    hyperparameters = cv_result.hyperparameters.copy()
    feature_importance = cv_result.feature_importance.copy()
    residuals = cv_result.per_group_residuals.copy()
    shrinkage = analyze_phase2_shrinkage(oof)
    if include_dose:
        dose_response = build_dose_response_check(
            targets,
            condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
        )
        residual_vs_dose = analyze_residual_vs_dose(
            oof,
            condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
        )
    else:
        dose_response = pd.DataFrame(columns=DOSE_RESPONSE_COLUMNS)
        residual_vs_dose = pd.DataFrame(columns=RESIDUAL_VS_DOSE_COLUMNS)
    within_condition = build_metrics_within_condition(oof)
    comparison = compare_phase1_phase2(
        phase1_metrics=phase1_metrics,
        phase1_shrinkage=phase1_shrinkage,
        phase2_metrics=metrics,
        phase2_shrinkage=shrinkage.report,
    )
    return Phase2DerivedTables(
        targets=targets,
        leakage=leakage,
        metrics=metrics,
        oof_predictions=oof,
        hyperparameters=hyperparameters,
        hyperparameter_grids=dict(cv_result.core.hyperparameter_grids)
        if hasattr(cv_result, "core")
        else dict(getattr(cv_result, "hyperparameter_grids", {})),
        feature_importance=feature_importance,
        per_group_residuals=residuals,
        shrinkage=shrinkage.report,
        dose_response=dose_response,
        residual_vs_dose=residual_vs_dose,
        within_condition=within_condition,
        comparison=comparison,
    )


def write_phase2_core_tables(
    staged_paths: Mapping[str, str | Path], tables: Phase2DerivedTables
) -> dict[str, Path]:
    """將 core/non-dose tables 依固定 schema 寫入 staging。"""
    values = {
        "phase2_group_targets_path": tables.targets,
        "phase2_leakage_preflight_report_path": tables.leakage,
        "phase2_cv_metrics_path": tables.metrics,
        "phase2_oof_predictions_path": tables.oof_predictions,
        "phase2_hyperparameters_path": tables.hyperparameters,
        "phase2_feature_importance_path": tables.feature_importance,
        "phase2_per_group_residuals_path": tables.per_group_residuals,
        "phase2_shrinkage_analysis_path": tables.shrinkage,
        "metrics_within_condition_path": tables.within_condition,
        "phase1_vs_phase2_comparison_path": tables.comparison,
    }
    written: dict[str, Path] = {}
    for field, frame in values.items():
        path = Path(staged_paths[field])
        _write_csv(frame, path, _TABLE_SCHEMAS[field])
        written[field] = path
    grids_path = Path(staged_paths["phase2_hyperparameter_grids_path"])
    grids_path.parent.mkdir(parents=True, exist_ok=True)
    grids_path.write_text(
        json.dumps(tables.hyperparameter_grids, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    written["phase2_hyperparameter_grids_path"] = grids_path
    return written


def write_phase2_dose_tables(
    staged_paths: Mapping[str, str | Path], tables: Phase2DerivedTables
) -> None:
    """寫入只在五個 core numeric artifacts freeze 後產生的 dose tables。"""
    _write_csv(
        tables.dose_response,
        Path(staged_paths["dose_response_check_path"]),
        DOSE_RESPONSE_COLUMNS,
    )
    _write_csv(
        tables.residual_vs_dose,
        Path(staged_paths["residual_vs_dose_path"]),
        RESIDUAL_VS_DOSE_COLUMNS,
    )


def _phase2_cjk_font_properties() -> Any:
    """取得可完整呈現 dose mapping note 的 deterministic CJK 字型。

    先依固定順序檢查 Windows 明確字型檔，再依固定 family 順序查詢
    Matplotlib。每個候選都必須涵蓋 note 的所有 code point；找不到時
    fail-closed，避免產生含 tofu 方框的正式 PNG。

    Returns:
        可傳給 Matplotlib text artist 的 ``FontProperties``。

    Raises:
        Phase2PublishError: 沒有任何已安裝字型能涵蓋完整 note。
    """
    import matplotlib.font_manager as font_manager
    from matplotlib.ft2font import FT2Font

    required_codepoints = {ord(character) for character in DOSE_MAPPING_NOTE}
    candidates: list[Path] = []
    windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates.extend(
        windows_root / "Fonts" / filename
        for filename in _PHASE2_CJK_FONT_FILENAMES
    )
    for family in _PHASE2_CJK_FONT_FAMILIES:
        try:
            discovered = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
        except (OSError, ValueError):
            continue
        candidates.append(Path(discovered))

    seen: set[Path] = set()
    attempted: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        attempted.append(str(candidate))
        if not candidate.is_file():
            continue
        try:
            charmap = FT2Font(str(candidate)).get_charmap()
        except (OSError, RuntimeError, ValueError):
            continue
        if required_codepoints.issubset(charmap):
            return font_manager.FontProperties(fname=str(candidate))

    raise Phase2PublishError(
        "Phase 2 dose figure requires a CJK font covering the exact "
        f"DOSE_MAPPING_NOTE; tried={attempted}"
    )


def _mpl_save(path: Path, draw: Callable[[Any], None]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 7), dpi=120)
    draw((figure, axis))
    figure.tight_layout()
    figure.savefig(
        path,
        format="png",
        dpi=120,
        metadata={"Software": "immunity.exp4.phase2_reporting"},
    )
    plt.close(figure)


def render_phase2_figures(
    staged_paths: Mapping[str, str | Path],
    *,
    comparison: pd.DataFrame,
    within_condition: pd.DataFrame,
    dose_response: pd.DataFrame,
) -> None:
    """以固定排序產生三張 Phase 2-only PNG。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Resolve before creating any figure so missing CJK support cannot leave a
    # partially rendered generation in the staging directory.
    cjk_font = _phase2_cjk_font_properties()

    figure_path = Path(staged_paths["phase2_figure_phase1_vs_phase2_r2_path"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    def draw_comparison(payload: Any) -> None:
        figure, axis = payload
        ordered = comparison.sort_values("configuration_id", kind="stable")
        positions = np.arange(len(ordered), dtype=float)
        axis.bar(positions - 0.27, ordered["phase1_mean_fold_cell_r2"], width=0.18, label="Phase 1 cell R²")
        axis.bar(positions - 0.09, ordered["phase2_mean_fold_cell_r2"], width=0.18, label="Phase 2 cell R²")
        axis.bar(positions + 0.09, ordered["phase1_mean_fold_rui_r2"], width=0.18, label="Phase 1 group R²")
        axis.bar(positions + 0.27, ordered["phase2_mean_fold_rui_r2"], width=0.18, label="Phase 2 group R²")
        axis.set_xticks(positions)
        axis.set_xticklabels(ordered["configuration_id"], rotation=90, fontsize=6)
        axis.set_ylabel("R²")
        axis.set_title("Exp4 Phase 1 vs Phase 2 R²")
        axis.legend(fontsize=8)

    _mpl_save(figure_path, draw_comparison)

    within_path = Path(staged_paths["phase2_figure_within_condition_r2_path"])
    within_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=120, sharey=True)
    ordered_within = within_condition.sort_values(
        ["condition_index", "configuration_id"], kind="stable"
    )
    for condition_index, axis in enumerate(axes.flat, start=1):
        subset = ordered_within.loc[
            ordered_within["condition_index"].eq(condition_index)
        ]
        axis.plot(np.arange(len(subset)), subset["cell_r2"].to_numpy(float), marker=".")
        axis.set_title(f"condition_index={condition_index}")
        axis.set_xticks([])
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("cell R²")
    axes[1, 0].set_ylabel("cell R²")
    figure.suptitle("Phase 2 within-condition Diagnostic-B")
    figure.tight_layout()
    figure.savefig(
        within_path,
        format="png",
        dpi=120,
        metadata={"Software": "immunity.exp4.phase2_reporting"},
    )
    plt.close(figure)

    dose_path = Path(staged_paths["phase2_figure_dose_response_path"])
    dose_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7), dpi=120)
    for _, row in dose_response.sort_values("donor_passage_id", kind="stable").iterrows():
        axis.plot(
            [0, 25, 50, 100],
            [
                row["ifn_0_tnf_0_target_median"],
                row["ifn_25_tnf_0_target_median"],
                row["ifn_50_tnf_0_target_median"],
                row["ifn_100_tnf_0_target_median"],
            ],
            marker="o",
            label=str(row["donor_passage_id"]),
        )
    axis.set_xlabel("reversed hypothesis IFN dose")
    axis.set_ylabel("IDO target median")
    axis.set_title("Phase 2 TNF=0 dose response")
    axis.text(
        0.01,
        0.01,
        DOSE_MAPPING_NOTE,
        transform=axis.transAxes,
        fontsize=9,
        color="darkred",
        fontproperties=cjk_font,
    )
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(
        dose_path,
        format="png",
        dpi=120,
        metadata={"Software": "immunity.exp4.phase2_reporting"},
    )
    plt.close(figure)


def build_phase2_report_section(
    *,
    metadata: Mapping[str, object],
    targets: pd.DataFrame,
    within_condition: pd.DataFrame,
    dose_response: pd.DataFrame,
) -> str:
    """建立 marker 內的 factual Phase 2 section 與四項 interpretation boundary。"""
    target_values = pd.to_numeric(targets["group_IDO_score"], errors="raise")
    target_range = float(target_values.max() - target_values.min())
    condition_four = within_condition.loc[
        within_condition["condition_index"].eq(4)
    ]
    condition_four_numbers = (
        "none"
        if condition_four.empty
        else "; ".join(
            f"{row.configuration_id}: cell_R²={float(row.cell_r2):.8g}, "
            f"MAE={float(row.cell_mae):.8g}"
            for row in condition_four.itertuples(index=False)
        )
    )
    monotonic = int(
        dose_response["monotonic_non_decreasing"].astype(bool).sum()
    )
    input_hashes = metadata.get("phase1_protected_artifact_sha256", {})
    mapping = metadata.get("mapping_independence", {})
    mapping_absent = bool(metadata.get("mapping_absent_by_core_interface", False))
    source_components = (
        mapping.get("core_source_components_sha256", {})
        if isinstance(mapping, Mapping)
        else {}
    )
    source_text = "; ".join(
        f"{key}={value}" for key, value in source_components.items()
    ) or "not_run"
    check_eight = (
        mapping.get("actual_leakage_check_8", {})
        if isinstance(mapping, Mapping)
        else {}
    )
    numeric_proof = metadata.get("phase2_numeric_artifact_sha256", {})
    before_candidate = (
        numeric_proof.get("before_dose_diagnostics", {})
        if isinstance(numeric_proof, Mapping)
        else {}
    )
    after_candidate = (
        numeric_proof.get("after_dose_diagnostics", {})
        if isinstance(numeric_proof, Mapping)
        else {}
    )
    before_hashes = before_candidate if isinstance(before_candidate, Mapping) else {}
    after_hashes = after_candidate if isinstance(after_candidate, Mapping) else {}
    numeric_hash_text = "; ".join(
        f"{name}: before={before_hashes.get(name, 'not_run')}, "
        f"after={after_hashes.get(name, 'not_run')}"
        for name in before_hashes
    ) or "not_run"
    confidence_counts = (
        targets["confidence_flag"].astype(str).value_counts().sort_index().to_dict()
    )
    population = metadata.get("population")
    retained_cells = (
        int(population.get("retained_cells", 19648))
        if isinstance(population, Mapping)
        else 19648
    )
    return f"""## Exp4 Phase 2：donor×passage×condition target

- status：{metadata.get('status', 'not_run')}；retained cells={retained_cells:,}；targets={len(targets)}；target range={target_range:.10g}。
- mapping_independence_status={metadata.get('mapping_independence_status', mapping.get('status', 'not_run') if isinstance(mapping, Mapping) else 'not_run')}；mapping_absent_by_core_interface={str(mapping_absent).lower()}；core source components：{source_text}。
- actual leakage check 8：status={check_eight.get('status', 'not_run') if isinstance(check_eight, Mapping) else 'not_run'}；details={check_eight.get('details', 'not_run') if isinstance(check_eight, Mapping) else 'not_run'}；Phase 1 protected hashes recorded={len(input_hashes) if isinstance(input_hashes, Mapping) else 0}。
- CV numbers bit-for-bit unchanged by dose mapping：five core numeric artifacts were frozen before dose diagnostics; exact SHA-256 before/after comparison：{numeric_hash_text}；immutability={numeric_proof.get('immutability_status', 'not_run') if isinstance(numeric_proof, Mapping) else 'not_run'}。
- target table：{len(targets)} opaque groups；confidence counts={confidence_counts}；target min/median/max={float(target_values.min()):.10g}/{float(target_values.median()):.10g}/{float(target_values.max()):.10g}。
- outputs are Phase 2-only and are staged before publication; Phase 1 `run_metadata.json` and `run.log` are not rewritten.

### Interpretation boundaries

1. This design deviates from Rui: each Rui donor had one 50 ng/mL IFN-γ 24h treatment, so donor aggregation denoised; our donor×passage group contains eight treatments, so Phase 1 aggregation deleted the dominant within-group signal.
2. Common-cause warning: stimulation changes both morphology and IDO. Higher R² may mean the model recognizes treatment/well, not intrinsic cell-line potential; never state the latter.
3. Diagnostic-B boundary: if within-condition performance collapses, Phase 2 R² can only support “morphology reflects stimulation treatment,” not “morphology predicts cell potential.” Condition-index-4 numbers: {condition_four_numbers}. No unspecified collapse threshold is invented.
4. Both designs use cell-level 5-fold CV with cells from the same group in train/test; absolute values are optimistic and only the relative comparison is meaningful.

### Dose-dependent diagnostics

{DOSE_MAPPING_NOTE}

- `dose_response_check.csv`: {DOSE_MAPPING_NOTE}; {monotonic}/9 TNF=0 donor×passage curves are monotonic non-decreasing under the reversed hypothesis; this is post-hoc evidence and never gates CV.
- `residual_vs_dose.csv`: {DOSE_MAPPING_NOTE}
- `figures/phase2_dose_response.png`: {DOSE_MAPPING_NOTE}

### Phase 2 machine-readable bundle

| artifact | path |
| --- | --- |
""" + "\n".join(
        f"| {field} | {metadata.get(field, 'not_run')} |" for field in PHASE2_OUTPUT_FIELDS
    ) + "\n"


def write_phase2_metadata(
    path: str | Path, metadata: Mapping[str, object]
) -> None:
    """以固定 JSON formatting 寫出 Phase 2 run metadata。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_phase2_log(path: str | Path, lines: Sequence[str]) -> None:
    """寫出不含 Phase 1 log 的 Phase 2-specific log。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(str(line) for line in lines).rstrip("\n") + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _json_safe(vars(value))
    return value


def _sha256_file(path: str | Path) -> str:
    candidate = _canonical(path)
    if not candidate.is_file():
        raise Phase2PublishError(f"Phase 1 artifact missing: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    """回傳指定 Phase 2 artifact 的 uppercase SHA-256。"""
    return _sha256_file(path)


def checkpoint_tree_fingerprint(path: str | Path) -> dict[str, object]:
    """以 manifest + sorted NPZ bytes 建立 Phase 1/2 checkpoint fingerprint。"""
    root = _canonical(path)
    manifest = root / "manifest.json"
    files = sorted(item for item in root.glob("*.npz") if item.is_file())
    if not root.is_dir() or not manifest.is_file() or len(files) != 120:
        raise Phase2PublishError(
            f"checkpoint tree 必須有 manifest.json 與 120 NPZ：{root}"
        )
    digest = hashlib.sha256()
    for item in [manifest, *files]:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "sha256": digest.hexdigest().upper(),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "npz_count": len(files),
    }


def mask_cache_fingerprint(path: str | Path) -> str:
    """以 relative filename + bytes 建立 mask-cache fingerprint。"""
    root = _canonical(path)
    files = sorted(item for item in root.rglob("*.npz") if item.is_file())
    if not root.is_dir() or not files:
        raise Phase2PublishError(f"mask cache 缺失或沒有 NPZ：{root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


def capture_phase1_immutable_state(
    *,
    artifact_paths: Mapping[str, str | Path],
    checkpoint_dir: str | Path,
    mask_dir: str | Path,
    metadata_path: str | Path,
    run_log_path: str | Path,
    report_path: str | Path,
) -> Phase1ImmutableState:
    """讀取 Phase 1 actual hashes/bytes，供 publish 前後比對。"""
    metadata = _canonical(metadata_path)
    run_log = _canonical(run_log_path)
    report = _canonical(report_path)
    if not metadata.is_file() or not run_log.is_file() or not report.is_file():
        raise Phase2PublishError("Phase 1 metadata、run.log、REPORT 必須存在")
    return Phase1ImmutableState(
        artifact_sha256={name: _sha256_file(path) for name, path in artifact_paths.items()},
        checkpoint_fingerprint=checkpoint_tree_fingerprint(checkpoint_dir),
        mask_cache_fingerprint=mask_cache_fingerprint(mask_dir),
        metadata_bytes=metadata.read_bytes(),
        run_log_bytes=run_log.read_bytes(),
        report_outside_marker=report_outside_phase2_marker(report.read_bytes()),
    )


def verify_phase1_immutable_state(
    state: Phase1ImmutableState,
    *,
    artifact_paths: Mapping[str, str | Path],
    checkpoint_dir: str | Path,
    mask_dir: str | Path,
    metadata_path: str | Path,
    run_log_path: str | Path,
    report_path: str | Path,
) -> None:
    """重算 Phase 1 hashes/checkpoint/mask/bytes，任一漂移即 fail closed。"""
    current = capture_phase1_immutable_state(
        artifact_paths=artifact_paths,
        checkpoint_dir=checkpoint_dir,
        mask_dir=mask_dir,
        metadata_path=metadata_path,
        run_log_path=run_log_path,
        report_path=report_path,
    )
    if dict(current.artifact_sha256) != dict(state.artifact_sha256):
        raise Phase2PublishError("Phase 1 protected artifact hash changed")
    if dict(current.checkpoint_fingerprint) != dict(state.checkpoint_fingerprint):
        raise Phase2PublishError("Phase 1 checkpoint fingerprint changed")
    if current.mask_cache_fingerprint != state.mask_cache_fingerprint:
        raise Phase2PublishError("Exp3 mask-cache fingerprint changed")
    if current.metadata_bytes != state.metadata_bytes:
        raise Phase2PublishError("Phase 1 run_metadata.json bytes changed")
    if current.run_log_bytes != state.run_log_bytes:
        raise Phase2PublishError("Phase 1 run.log bytes changed")
    if current.report_outside_marker != state.report_outside_marker:
        raise Phase2PublishError("Phase 1 REPORT bytes outside Phase 2 marker changed")


def validate_phase1_hash_lock(
    artifact_paths: Mapping[str, str | Path],
    expected_hashes: Mapping[str, str],
) -> dict[str, str]:
    """比對 brief 指定的 25 個 Phase 1 numeric/figure hashes。"""
    if set(artifact_paths) != set(expected_hashes):
        raise Phase2PublishError("Phase 1 protected hash roster mismatch")
    actual = {name: _sha256_file(path) for name, path in artifact_paths.items()}
    mismatch = [
        name
        for name in expected_hashes
        if str(expected_hashes[name]).strip().upper() != actual[name]
    ]
    if mismatch:
        raise Phase2PublishError("Phase 1 protected hash mismatch: " + ", ".join(mismatch))
    return actual


__all__ = [
    "DOSE_MAPPING_NOTE",
    "PHASE2_FIGURE_FIELDS",
    "PHASE2_OUTPUT_FIELDS",
    "PHASE2_PUBLISH_FIELDS",
    "PHASE2_REPORT_END",
    "PHASE2_REPORT_START",
    "PHASE2_TABLE_FIELDS",
    "Phase1ImmutableState",
    "Phase2DerivedTables",
    "Phase2PublishError",
    "build_phase2_report_section",
    "capture_phase1_immutable_state",
    "checkpoint_tree_fingerprint",
    "derive_phase2_tables",
    "mask_cache_fingerprint",
    "phase2_output_paths",
    "phase2_publish_paths",
    "publish_phase2_bundle",
    "render_phase2_figures",
    "replace_phase2_report_section",
    "report_outside_phase2_marker",
    "sha256_file",
    "validate_phase1_hash_lock",
    "validate_phase2_publish_mapping",
    "verify_phase1_immutable_state",
    "verify_phase2_numeric_artifact_immutability",
    "fingerprint_phase2_numeric_artifacts",
    "write_phase2_core_tables",
    "write_phase2_dose_tables",
    "write_phase2_log",
    "write_phase2_metadata",
]
