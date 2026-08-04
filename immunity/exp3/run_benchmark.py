"""提供 Exp3 benchmark 的設定與命令列介面。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP3_OUTPUT_ROOT = (PROJECT_ROOT / "immunity" / "outputs" / "exp3").resolve()

_FORMAL_TABLE_NAMES = (
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
)
_FORMAL_FILE_NAMES = (
    *_FORMAL_TABLE_NAMES,
    "feature_sets.json",
    "run_metadata.json",
    "final_model.json",
    "EXPERIMENT_RECORD.md",
)
_FORMAL_DIRECTORY_NAMES = ("figures", "models")
_FAILURE_EVIDENCE_NAMES = {
    "data_manifest.csv",
    "pairing_qc.csv",
    "segmentation_qc.csv",
    "model_failures.csv",
    "condition_adjusted_metrics.csv",
}


def resolve_exp3_output_dir(path: str | Path) -> Path:
    """解析並驗證 Exp3 專用輸出目錄。

    Args:
        path: 欲使用的輸出目錄，可為相對或絕對路徑。

    Returns:
        已解析且位於 Exp3 專用輸出根目錄內的絕對路徑。

    Raises:
        ValueError: 當路徑不在 ``immunity/outputs/exp3`` 內時拋出。
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != EXP3_OUTPUT_ROOT and EXP3_OUTPUT_ROOT not in resolved.parents:
        raise ValueError("Exp3 output 必須位於 immunity/outputs/exp3 之下")
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    """讀取並驗證 Exp3 YAML 設定，不執行任何 pipeline。

    Args:
        path: Exp3 YAML 設定檔的相對或絕對路徑。

    Returns:
        已通過必要欄位與輸出目錄檢查的設定內容。

    Raises:
        ValueError: 當 YAML 根節點不是 mapping，或缺少必要欄位時拋出。
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Exp3 config 必須是 mapping")

    required = {
        "datasets",
        "expected_totals",
        "condition_mapping",
        "segmentation",
        "development_validation",
        "feature_sets",
        "benchmark",
        "output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Exp3 config 缺少必要欄位：{missing}")

    config["_config_path"] = str(config_path.resolve())
    config["_output_dir"] = str(resolve_exp3_output_dir(config["output"]["dir"]))
    return config


def build_parser() -> argparse.ArgumentParser:
    """建立 Exp3 獨立 benchmark 命令列解析器。

    Returns:
        設定完成的 ``ArgumentParser``。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-fovs-per-condition", type=int, default=None)
    return parser


def run_benchmark(
    config: Mapping[str, Any],
    smoke_fovs_per_condition: int | None = None,
) -> Path:
    """隔離舊執行世代後，執行並發布單一 Exp3 benchmark 世代。

    Args:
        config: 經 ``load_config`` 驗證的完整 Exp3 設定。
        smoke_fovs_per_condition: 每個 group/condition 保留的 smoke FOV 數。

    Returns:
        完成本世代發布的 ``EXPERIMENT_RECORD.md`` 路徑。

    Raises:
        TypeError: ``config`` 不是 mapping 時拋出。
        ValueError: output boundary 或輸入設定不合法時拋出。
        RuntimeError: 任一 pipeline stage 無法完成時拋出。
    """
    if not isinstance(config, Mapping):
        raise TypeError("Exp3 config 必須是 mapping")
    config_evidence = _capture_config_evidence(config)
    generation_config = dict(config)
    generation_config["_config_evidence"] = config_evidence
    output_dir = _resolve_run_output(config, smoke_fovs_per_condition)
    _archive_previous_generation(output_dir)
    try:
        return _run_benchmark_generation(generation_config, smoke_fovs_per_condition)
    except BaseException as error:
        try:
            _quarantine_failed_generation(output_dir)
        except BaseException as quarantine_error:
            error.add_note(
                "Exp3 failed-generation quarantine error: "
                f"{type(quarantine_error).__name__}: {quarantine_error}"
            )
        raise


def _run_benchmark_generation(
    config: Mapping[str, Any],
    smoke_fovs_per_condition: int | None = None,
) -> Path:
    """依固定 stage 順序執行 Exp3 phase-only benchmark。

    Args:
        config: 經 ``load_config`` 驗證的完整 Exp3 設定。測試可用私有
            ``_segmenter`` 注入符合 ``Segmenter`` protocol 的 synthetic adapter。
        smoke_fovs_per_condition: 每個 ``group_id × condition_index`` 保留的前 N
            個 FOV；提供時所有結果隔離於唯一 ``smoke/`` child。

    Returns:
        完成發布的 ``EXPERIMENT_RECORD.md`` 路徑。

    Raises:
        TypeError: ``config`` 不是 mapping 時拋出。
        ValueError: 輸入設定、mapping、feature set 或 output boundary 無效時拋出。
        RuntimeError: Segmentation／extraction／model evidence 不完整時拋出。
    """
    if not isinstance(config, Mapping):
        raise TypeError("Exp3 config 必須是 mapping")
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    config_evidence = _required_config_evidence(config)
    output_dir = _resolve_run_output(config, smoke_fovs_per_condition)
    run_config = dict(config)
    run_config["_output_dir"] = str(output_dir)
    feature_cache = _prepare_child_directory(output_dir, "feature_cache")

    #延遲匯入可避免 feature/reporting modules 反向依賴 resolver 的 circular import。
    from immunity.exp3.benchmark import (
        FAILURE_COLUMNS,
        FEATURE_IMPORTANCE_COLUMNS,
        FOLD_METRIC_COLUMNS,
        HYPERPARAMETER_COLUMNS,
        OOF_COLUMNS,
        BenchmarkResult,
        fit_final_phase_model,
        make_outer_splits,
        outer_split_manifest,
        rank_phase_models,
        run_condition_adjusted_sensitivity,
        run_nested_benchmark,
        run_phase_feature_set_benchmark,
        select_winner,
    )
    from immunity.exp3.feature_sets import (
        FEATURE_SET_REGISTRY,
        PRIMARY_FOV_FEATURES,
        aggregate_fov_features,
        calculate_delta_signatures,
    )
    from immunity.exp3.manifest import (
        apply_condition_mapping,
        scan_datasets,
        validate_expected_totals,
    )
    from immunity.exp3.phase_features import (
        DEVELOPMENT_VALIDATION_COLUMNS,
        PhaseSegmenter,
        cache_phase_masks,
        extract_basic_cell_features,
        run_development_nucleus_validation,
    )

    raw_manifest, pairing_qc = scan_datasets(_required_sequence(config, "datasets"))
    _write_csv_atomically(pairing_qc, output_dir / "pairing_qc.csv")
    validate_expected_totals(
        pairing_qc,
        _required_mapping(config, "expected_totals"),
    )
    sampled_raw = _limit_smoke_manifest(raw_manifest, smoke_fovs_per_condition)
    manifest = apply_condition_mapping(
        sampled_raw,
        _required_mapping(config, "condition_mapping"),
    )
    manifest_path = output_dir / "data_manifest.csv"
    _write_csv_atomically(manifest, manifest_path)
    manifest_hash = _sha256_bytes(manifest_path.read_bytes())

    injected_segmenter = config.get("_segmenter")
    active_segmenter = (
        injected_segmenter
        if injected_segmenter is not None
        else PhaseSegmenter(_required_mapping(config, "segmentation"))
    )
    segmentation_qc = cache_phase_masks(
        manifest,
        feature_cache,
        run_config,
        segmenter=active_segmenter,
    )
    _write_csv_atomically(segmentation_qc, output_dir / "segmentation_qc.csv")
    failed_segmentation = segmentation_qc[~segmentation_qc["status"].eq("passed")]
    if not failed_segmentation.empty:
        raise RuntimeError(
            "segmentation failed for image keys: "
            f"{failed_segmentation['image_key'].astype(str).tolist()}"
        )

    development_validation = run_development_nucleus_validation(
        run_config,
        feature_cache,
    )
    if development_validation.empty:
        development_validation = pd.DataFrame(columns=DEVELOPMENT_VALIDATION_COLUMNS)
    cells, extraction_qc = extract_basic_cell_features(
        manifest,
        segmentation_qc,
        run_config,
    )
    combined_qc = _combine_segmentation_and_extraction_qc(
        segmentation_qc,
        extraction_qc,
    )
    _write_csv_atomically(combined_qc, output_dir / "segmentation_qc.csv")
    failed_extraction = extraction_qc[~extraction_qc["status"].eq("passed")]
    if not failed_extraction.empty:
        raise RuntimeError(
            "feature extraction failed for image keys: "
            f"{failed_extraction['image_key'].astype(str).tolist()}"
        )

    enabled_feature_sets = _enabled_feature_sets(config)
    if "basic_median" not in enabled_feature_sets:
        raise ValueError("Round 1 必須明確啟用 basic_median")
    images, feature_sets = aggregate_fov_features(
        cells,
        manifest,
        enabled_feature_sets,
    )
    images.attrs["manifest_hash"] = manifest_hash
    delta_features = list(
        dict.fromkeys(
            column
            for name in enabled_feature_sets
            for column in feature_sets[name]
        )
    )
    morphology_delta = calculate_delta_signatures(images, delta_features)

    splits = make_outer_splits(images)
    outer_splits = outer_split_manifest(images, splits)
    benchmark_config = dict(_required_mapping(config, "benchmark"))
    candidate_names = [
        "paper_linear_3f",
        "ridge",
        "elasticnet",
        "rbf_svr",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ]
    diagnostic_names = [
        "dummy_median",
        "dose_ridge",
        "dose_plus_morphology_ridge",
    ]
    primary = run_nested_benchmark(
        images,
        feature_sets,
        splits,
        benchmark_config,
        model_names=[*candidate_names, *diagnostic_names],
    )
    condition_adjusted = run_condition_adjusted_sensitivity(
        images,
        feature_sets,
        splits,
        benchmark_config,
        model_names=candidate_names,
    )

    evidence_config = dict(benchmark_config)
    evidence_config["feature_sets"] = {
        "basic_median": list(PRIMARY_FOV_FEATURES)
    }
    evidence_config["expected_outer_split_ids"] = _canonical_split_ids(splits)
    ranking = rank_phase_models(
        primary.fold_metrics,
        primary.predictions,
        primary.failures,
        evidence_config,
    )
    ranking["role"] = "candidate"
    eligible_phase_models = ranking.loc[
        ranking["eligible"].fillna(False).astype(bool), "model"
    ].astype(str).tolist()
    winner = select_winner(ranking)

    round_two = _run_round_two(
        images=images,
        feature_sets=feature_sets,
        splits=splits,
        benchmark_config=benchmark_config,
        ranking=ranking,
        enabled_feature_sets=enabled_feature_sets,
        primary_candidates=candidate_names,
        run_phase_feature_set_benchmark=run_phase_feature_set_benchmark,
        benchmark_result_type=BenchmarkResult,
        oof_columns=OOF_COLUMNS,
        metric_columns=FOLD_METRIC_COLUMNS,
        hyperparameter_columns=HYPERPARAMETER_COLUMNS,
        importance_columns=FEATURE_IMPORTANCE_COLUMNS,
        failure_columns=FAILURE_COLUMNS,
    )
    predictions = _combine_round_frames(
        primary.predictions,
        round_two.predictions,
    )
    fold_metrics = _combine_round_frames(
        primary.fold_metrics,
        round_two.fold_metrics,
    )
    hyperparameters = _combine_round_frames(
        primary.hyperparameters,
        round_two.hyperparameters,
    )
    feature_importance = _combine_round_frames(
        primary.feature_importance,
        round_two.feature_importance,
    )
    failures = _combine_round_frames(primary.failures, round_two.failures)

    from immunity.exp3.reporting import (
        write_experiment_record,
        write_figures,
        write_result_tables,
    )

    tables = {
        "data_manifest.csv": manifest,
        "pairing_qc.csv": pairing_qc,
        "segmentation_qc.csv": combined_qc,
        "outer_splits.csv": outer_splits,
        "oof_predictions.csv": predictions,
        "fold_metrics.csv": fold_metrics,
        "hyperparameters.csv": hyperparameters,
        "model_ranking.csv": ranking,
        "feature_importance.csv": feature_importance,
        "model_failures.csv": failures,
        "condition_adjusted_metrics.csv": condition_adjusted,
        "pc_nucleus_dapi_validation.csv": development_validation,
        "morphology_delta_signatures.csv": morphology_delta,
    }
    write_result_tables(output_dir, tables)
    status = "winner_selected" if winner is not None else "no_eligible_phase_only_model"
    primary_artifacts = {
        "status": status,
        "winner": winner,
        "ranking": ranking,
        "fold_metrics": primary.fold_metrics,
        "oof_predictions": primary.predictions,
        "feature_importance": primary.feature_importance,
        "morphology_delta_signatures": morphology_delta,
    }
    write_figures(output_dir, primary_artifacts)
    final_config = dict(evidence_config)
    final_config["eligible_phase_models"] = eligible_phase_models
    final_config["config_hash"] = config_evidence["effective_config_hash"]
    final_config["manifest_hash"] = manifest_hash
    fit_final_phase_model(
        images,
        winner,
        feature_sets,
        final_config,
        output_dir,
    )

    ended_at = datetime.now(timezone.utc)
    elapsed_seconds = max(0.0, time.perf_counter() - started_clock)
    metadata = _run_metadata(
        config=config,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed_seconds,
        config_evidence=config_evidence,
        manifest_hash=manifest_hash,
        enabled_feature_sets=enabled_feature_sets,
        smoke_fovs_per_condition=smoke_fovs_per_condition,
        pairing_qc=pairing_qc,
        analyzed_images=len(manifest),
        segmentation_qc=segmentation_qc,
        failure_count=len(failures),
    )
    feature_registry = json.loads(
        (output_dir / "feature_sets.json").read_text(encoding="utf-8")
    )
    context = {
        **primary_artifacts,
        "raw_pc": metadata["input_counts"]["raw_pc"],
        "raw_ido": metadata["input_counts"]["raw_ido"],
        "complete_pairs": metadata["input_counts"]["complete_pairs"],
        "exclusions": pairing_qc.loc[
            ~pairing_qc["status"].eq("paired"), "detail"
        ].astype(str).tolist(),
        "segmentation_pass_rate": float(segmentation_qc["status"].eq("passed").mean()),
        "condition_adjusted_metrics": condition_adjusted,
        "pc_nucleus_dapi_validation": development_validation,
        "limitations": [
            "IDO fluorescence is an image-level background-corrected IDO proxy.",
            "B-ID 身分尚未確認，結果只代表跨 B-ID exploratory validation。",
            "Round 2 secondary feature sets 屬探索性，不改寫 Primary winner。",
        ],
        "metadata": metadata,
        "feature_sets": feature_registry,
    }
    record = write_experiment_record(output_dir, context)
    if not record.is_file():
        raise RuntimeError("EXPERIMENT_RECORD.md 未成功發布")
    return record


def main() -> int:
    """執行 Exp3 CLI，並將 stdout/stderr 同步保存至 run-local log。

    Returns:
        Record 存在時回傳 0；設定或 pipeline 失敗時回傳非零。
    """
    arguments = build_parser().parse_args()
    original_directory = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        try:
            config = load_config(arguments.config)
            output_dir = _resolve_run_output(
                config,
                arguments.smoke_fovs_per_condition,
            )
        except Exception:
            error_info = sys.exc_info()
            fallback = Path(EXP3_OUTPUT_ROOT)
            if arguments.smoke_fovs_per_condition is not None:
                fallback = fallback / "smoke"
            try:
                fallback = _prepare_output_directory(fallback)
                log_path = _safe_run_file(fallback, "run.log")
                with log_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as log_handle:
                    stderr_tee = _Tee(sys.stderr, log_handle)
                    traceback.print_exception(*error_info, file=stderr_tee)
            except Exception:
                traceback.print_exception(*error_info)
            return 2
        log_path = _safe_run_file(output_dir, "run.log")
        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            stdout_tee = _Tee(sys.stdout, log_handle)
            stderr_tee = _Tee(sys.stderr, log_handle)
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(
                stderr_tee
            ):
                try:
                    record = run_benchmark(
                        config,
                        smoke_fovs_per_condition=arguments.smoke_fovs_per_condition,
                    )
                    if not record.is_file() or record.name != "EXPERIMENT_RECORD.md":
                        raise RuntimeError("EXPERIMENT_RECORD.md 不存在，CLI 不可回傳成功")
                    print(f"Exp3 record: {record}")
                    return 0
                except Exception:  # noqa: BLE001 - CLI 必須保留完整錯誤 evidence
                    traceback.print_exc()
                    return 1
    finally:
        os.chdir(original_directory)


class _Tee:
    """將文字同步寫入 console stream 與 run log。"""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self._stream = stream
        self._log = log

    def write(self, value: str) -> int:
        """同步寫入兩個 streams。"""
        self._stream.write(value)
        self._log.write(value)
        self._log.flush()
        return len(value)

    def flush(self) -> None:
        """同步刷新兩個 streams。"""
        self._stream.flush()
        self._log.flush()


def _resolve_run_output(
    config: Mapping[str, Any],
    smoke_fovs_per_condition: int | None,
) -> Path:
    """解析 full/smoke 唯一輸出位置並建立安全 run 目錄。"""
    if smoke_fovs_per_condition is not None:
        if isinstance(smoke_fovs_per_condition, bool) or int(
            smoke_fovs_per_condition
        ) != smoke_fovs_per_condition:
            raise ValueError("smoke_fovs_per_condition 必須是正整數")
        if int(smoke_fovs_per_condition) < 1:
            raise ValueError("smoke_fovs_per_condition 必須是正整數")
    configured = config.get("_output_dir")
    if configured is None:
        output = _required_mapping(config, "output")
        configured = output.get("dir")
    base = resolve_exp3_output_dir(configured)
    if smoke_fovs_per_condition is None:
        run_output = base
    else:
        run_output = base if base.name == "smoke" else base / "smoke"
        run_output = resolve_exp3_output_dir(run_output)
    return _prepare_output_directory(run_output)


def _prepare_output_directory(path: Path) -> Path:
    """建立且重驗證未被 symlink/junction 改寫的 output directory。"""
    expected = Path(path)
    if expected.exists() and not expected.is_dir():
        raise ValueError(f"Exp3 output 必須是目錄：{expected}")
    expected.mkdir(parents=True, exist_ok=True)
    if expected.resolve(strict=False) != expected:
        raise ValueError("Exp3 output 不可透過 symlink 或 junction 改寫")
    return expected


def _prepare_child_directory(parent: Path, name: str) -> Path:
    """在 run root 建立單一固定 child directory。"""
    child = parent / name
    if child.exists() and not child.is_dir():
        raise ValueError(f"Exp3 child 必須是目錄：{child}")
    child.mkdir(parents=False, exist_ok=True)
    if child.resolve(strict=False) != child:
        raise ValueError(f"Exp3 child {name!r} 不可透過 symlink 或 junction 改寫")
    return child


def _safe_run_file(parent: Path, name: str) -> Path:
    """驗證 run-local final file 沒有 link escape 或目錄型別衝突。"""
    safe_parent = _prepare_output_directory(parent)
    destination = safe_parent / name
    if destination.exists() and not destination.is_file():
        raise ValueError(f"Exp3 output file 必須是一般檔案：{destination}")
    if destination.resolve(strict=False) != destination:
        raise ValueError(f"Exp3 output file {name!r} 不可透過 symlink 改寫")
    return destination


def _directory_identity(path: Path) -> tuple[int, int, Path]:
    """擷取目錄 identity，供跨 preflight/move 的替換偵測。"""
    stat = path.stat()
    return int(stat.st_dev), int(stat.st_ino), path.resolve(strict=True)


def _assert_directory_identity(
    path: Path,
    identity: tuple[int, int, Path],
) -> None:
    """確認目錄未在 archive transaction 期間被置換。"""
    if _directory_identity(path) != identity:
        raise ValueError("Exp3 archive parent 在 transaction 期間已被替換")


def _archive_previous_generation(output_dir: Path) -> Path | None:
    """將上一世代所有固定正式 artifacts 搬入可復原 archive。"""
    return _archive_generation_targets(
        output_dir,
        (*_FORMAL_FILE_NAMES, *_FORMAL_DIRECTORY_NAMES),
        prefix="previous",
    )


def _quarantine_failed_generation(output_dir: Path) -> Path | None:
    """隔離失敗世代的成功 artifacts，只保留當次失敗/QC evidence。"""
    names = tuple(
        name
        for name in (*_FORMAL_FILE_NAMES, *_FORMAL_DIRECTORY_NAMES)
        if name not in _FAILURE_EVIDENCE_NAMES
    )
    return _archive_generation_targets(output_dir, names, prefix="failed")


def _archive_generation_targets(
    output_dir: Path,
    names: Sequence[str],
    *,
    prefix: str,
) -> Path | None:
    """以完整 preflight、identity checks 與 rollback 搬移固定 targets。

    Args:
        output_dir: 已限制於 Exp3 專用根目錄內的本次 run root。
        names: 僅允許直接位於 run root 的固定檔案或目錄名稱。
        prefix: Archive child 的世代類型標記。

    Returns:
        有 targets 時回傳 archive child；否則回傳 ``None``。

    Raises:
        ValueError: 任一 target、archive root 或 parent identity 不安全時拋出。
        OSError: 搬移失敗且 rollback 完成後重新拋出原始錯誤。
    """
    safe_root = _prepare_output_directory(output_dir)
    root_identity = _directory_identity(safe_root)
    planned: list[tuple[Path, tuple[int, int, bool]]] = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Exp3 archive target 必須是固定單層名稱")
        target = safe_root / name
        if target.resolve(strict=False) != target:
            raise ValueError(f"Exp3 archive target {name!r} 不可是 link/junction")
        if not target.exists():
            continue
        expected_directory = name in _FORMAL_DIRECTORY_NAMES
        if expected_directory != target.is_dir():
            expected = "目錄" if expected_directory else "一般檔案"
            raise ValueError(f"Exp3 archive target {name!r} 必須是{expected}")
        stat = target.stat()
        planned.append(
            (target, (int(stat.st_dev), int(stat.st_ino), expected_directory))
        )

    generations = safe_root / "_generations"
    if generations.resolve(strict=False) != generations:
        raise ValueError("Exp3 _generations 不可是 symlink 或 junction")
    if generations.exists() and not generations.is_dir():
        raise ValueError("Exp3 _generations 必須是目錄")
    if not planned:
        return None

    _assert_directory_identity(safe_root, root_identity)
    generations = _prepare_child_directory(safe_root, "_generations")
    generations_identity = _directory_identity(generations)
    archive_name = (
        f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{uuid.uuid4().hex[:12]}"
    )
    archive_dir = generations / archive_name
    if archive_dir.resolve(strict=False) != archive_dir or archive_dir.exists():
        raise ValueError("Exp3 archive child 路徑不安全或已存在")
    archive_dir.mkdir(parents=False, exist_ok=False)
    archive_identity = _directory_identity(archive_dir)
    moved: list[tuple[Path, Path]] = []
    try:
        for target, expected_identity in planned:
            _assert_directory_identity(safe_root, root_identity)
            _assert_directory_identity(generations, generations_identity)
            _assert_directory_identity(archive_dir, archive_identity)
            if not target.exists() or target.resolve(strict=False) != target:
                raise ValueError(f"Exp3 archive target {target.name!r} 已被替換")
            stat = target.stat()
            actual_identity = (
                int(stat.st_dev),
                int(stat.st_ino),
                target.is_dir(),
            )
            if actual_identity != expected_identity:
                raise ValueError(f"Exp3 archive target {target.name!r} identity 已改變")
            destination = archive_dir / target.name
            if destination.resolve(strict=False) != destination or destination.exists():
                raise ValueError("Exp3 archive destination 不安全或已存在")
            os.replace(target, destination)
            moved.append((target, destination))
    except Exception as error:
        rollback_errors: list[Exception] = []
        for target, destination in reversed(moved):
            try:
                _assert_directory_identity(safe_root, root_identity)
                _assert_directory_identity(archive_dir, archive_identity)
                if target.exists() or not destination.exists():
                    raise ValueError("Exp3 archive rollback target 狀態不一致")
                os.replace(destination, target)
            except Exception as rollback_error:  #保留原始錯誤並附加 rollback evidence。
                rollback_errors.append(rollback_error)
        if rollback_errors:
            error.add_note(
                "Exp3 archive rollback failures: "
                + "; ".join(str(item) for item in rollback_errors)
            )
        raise
    finally:
        if archive_dir.exists() and not any(archive_dir.iterdir()):
            archive_dir.rmdir()
    return archive_dir


def _write_csv_atomically(frame: pd.DataFrame, destination: Path) -> None:
    """在重新驗證 parent identity 後原子發布 early QC／manifest CSV。"""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("early Exp3 CSV 必須是 pandas DataFrame")
    final_path = _safe_run_file(destination.parent, destination.name)
    parent_stat = final_path.parent.stat()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv.tmp",
            dir=final_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
        current_stat = final_path.parent.stat()
        if (current_stat.st_dev, current_stat.st_ino) != (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ):
            raise ValueError("Exp3 output parent 在 atomic replace 前已被替換")
        _safe_run_file(final_path.parent, final_path.name)
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """讀取必要 mapping 設定。"""
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Exp3 config {key!r} 必須是 mapping")
    return value


def _required_sequence(
    config: Mapping[str, Any], key: str
) -> Sequence[Mapping[str, Any]]:
    """讀取必要 mapping sequence 設定。"""
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Exp3 config {key!r} 必須是 sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"Exp3 config {key!r} 每一項都必須是 mapping")
    return value


def _limit_smoke_manifest(
    raw_manifest: pd.DataFrame,
    smoke_fovs_per_condition: int | None,
) -> pd.DataFrame:
    """依 group/condition/FOV deterministic 選取 smoke image identities。"""
    if smoke_fovs_per_condition is None:
        return raw_manifest.copy().reset_index(drop=True)
    limit = int(smoke_fovs_per_condition)
    required = {"group_id", "condition_index", "fov", "pc_path", "ido_path"}
    missing = sorted(required - set(raw_manifest.columns))
    if missing:
        raise ValueError(f"smoke manifest 缺少欄位：{missing}")
    ordered = raw_manifest.sort_values(
        ["group_id", "condition_index", "fov", "pc_path", "ido_path"],
        kind="stable",
    )
    sampled = ordered.groupby(
        ["group_id", "condition_index"],
        sort=False,
        group_keys=False,
    ).head(limit)
    identities = sampled.loc[
        :, ["group_id", "condition_index", "fov", "pc_path", "ido_path"]
    ]
    if identities.duplicated().any():
        raise ValueError("smoke sampling 產生重複 image identity")
    return sampled.reset_index(drop=True)


def _combine_segmentation_and_extraction_qc(
    segmentation_qc: pd.DataFrame,
    extraction_qc: pd.DataFrame,
) -> pd.DataFrame:
    """將兩階段逐圖 QC 合併為單一正式 evidence table。"""
    renamed = extraction_qc.rename(
        columns={
            column: f"extraction_{column}"
            for column in extraction_qc.columns
            if column != "image_key"
        }
    )
    return segmentation_qc.merge(
        renamed,
        on="image_key",
        how="left",
        validate="one_to_one",
    )


def _enabled_feature_sets(config: Mapping[str, Any]) -> tuple[str, ...]:
    """讀取且正規化明確啟用的 feature-set names。"""
    feature_config = _required_mapping(config, "feature_sets")
    enabled = feature_config.get("enabled")
    if not isinstance(enabled, Sequence) or isinstance(enabled, (str, bytes)):
        raise ValueError("feature_sets.enabled 必須是非空 sequence")
    names = tuple(str(name) for name in enabled)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_sets.enabled 不可為空或重複")
    return names


def _canonical_split_ids(splits: Sequence[Any]) -> dict[str, list[str]]:
    """由唯一 shared splits 建立 Task 8 canonical evidence mapping。"""
    result: dict[str, list[str]] = {}
    for split in splits:
        result.setdefault(str(split.validation), []).append(
            f"{split.validation}:{split.fold}"
        )
    return result


def _run_round_two(
    *,
    images: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    splits: Sequence[Any],
    benchmark_config: Mapping[str, Any],
    ranking: pd.DataFrame,
    enabled_feature_sets: Sequence[str],
    primary_candidates: Sequence[str],
    run_phase_feature_set_benchmark: Any,
    benchmark_result_type: Any,
    oof_columns: Sequence[str],
    metric_columns: Sequence[str],
    hyperparameter_columns: Sequence[str],
    importance_columns: Sequence[str],
    failure_columns: Sequence[str],
) -> Any:
    """只對 overall rank 前二執行 opt-in secondary exploratory benchmark。"""
    secondary_sets = [name for name in enabled_feature_sets if name != "basic_median"]
    empty = benchmark_result_type(
        predictions=pd.DataFrame(columns=oof_columns),
        fold_metrics=pd.DataFrame(columns=metric_columns),
        hyperparameters=pd.DataFrame(columns=hyperparameter_columns),
        feature_importance=pd.DataFrame(columns=importance_columns),
        failures=pd.DataFrame(columns=failure_columns),
    )
    if not secondary_sets:
        return empty
    top_models = _round_two_top_models(ranking, primary_candidates)
    if not top_models:
        return empty
    frames: dict[str, list[pd.DataFrame]] = {
        "predictions": [],
        "fold_metrics": [],
        "hyperparameters": [],
        "feature_importance": [],
        "failures": [],
    }
    for feature_set_name in secondary_sets:
        nonpaper = [name for name in top_models if name != "paper_linear_3f"]
        if nonpaper:
            result = run_phase_feature_set_benchmark(
                images,
                feature_sets,
                splits,
                benchmark_config,
                model_names=nonpaper,
                feature_set_name=feature_set_name,
            )
            for key in frames:
                frames[key].append(getattr(result, key))
        if "paper_linear_3f" in top_models:
            frames["fold_metrics"].append(
                _paper_not_applicable_metrics(
                    images,
                    splits,
                    feature_set_name,
                    metric_columns,
                )
            )
    return benchmark_result_type(
        predictions=_concat_frames(frames["predictions"], oof_columns),
        fold_metrics=_concat_frames(frames["fold_metrics"], metric_columns),
        hyperparameters=_concat_frames(
            frames["hyperparameters"], hyperparameter_columns
        ),
        feature_importance=_concat_frames(
            frames["feature_importance"], importance_columns
        ),
        failures=_concat_frames(frames["failures"], failure_columns),
    )


def _round_two_top_models(
    ranking: pd.DataFrame,
    primary_candidates: Sequence[str],
) -> list[str]:
    """依 Task 8 完整 tie-break evidence 選取 Round 2 前兩名。"""
    required = {
        "model",
        "overall_rank",
        "worst_validation_rank",
        "leave_one_b_out_mae",
        "overall_oof_spearman",
        "simplicity_rank",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"Round 2 ranking 缺少 tie-break evidence：{missing}")
    ranked = ranking[ranking["model"].isin(primary_candidates)].copy()
    numeric_columns = [
        "overall_rank",
        "worst_validation_rank",
        "leave_one_b_out_mae",
        "overall_oof_spearman",
        "simplicity_rank",
    ]
    for column in numeric_columns:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked[ranked["overall_rank"].notna()]
    if ranked.empty:
        return []
    best_rank = float(ranked["overall_rank"].min())
    first_band = ranked[
        (ranked["overall_rank"] - best_rank).lt(0.25)
    ].copy()
    first_band["_spearman_sort"] = -first_band["overall_oof_spearman"]
    evidence_columns = [
        "worst_validation_rank",
        "leave_one_b_out_mae",
        "_spearman_sort",
        "simplicity_rank",
        "model",
    ]
    ordered_band = first_band.sort_values(
        evidence_columns,
        ascending=True,
        na_position="last",
        kind="stable",
    )
    selected = ordered_band["model"].astype(str).head(2).tolist()
    if len(selected) == 2:
        return selected

    outside = ranked[~ranked["model"].isin(first_band["model"])].copy()
    outside["_spearman_sort"] = -outside["overall_oof_spearman"]
    ordered_outside = outside.sort_values(
        ["overall_rank", *evidence_columns],
        ascending=True,
        na_position="last",
        kind="stable",
    )
    for model in ordered_outside["model"].astype(str):
        if model not in selected:
            selected.append(model)
        if len(selected) == 2:
            break
    return selected


def _paper_not_applicable_metrics(
    images: pd.DataFrame,
    splits: Sequence[Any],
    feature_set_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """以明確 not_applicable rows 保留 paper fixed-3f Round 2 決策。"""
    rows = []
    for split in splits:
        rows.append(
            {
                "validation": split.validation,
                "fold": split.fold,
                "split_id": f"{split.validation}:{split.fold}",
                "model": "paper_linear_3f",
                "role": "candidate",
                "feature_set": feature_set_name,
                "n_train": len(split.train_index),
                "n_test": len(split.test_index),
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "spearman": float("nan"),
                "observed_sd": float("nan"),
                "prediction_sd": float("nan"),
                "status": "not_applicable",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _concat_frames(
    frames: Sequence[pd.DataFrame], columns: Sequence[str]
) -> pd.DataFrame:
    """以固定 columns 合併零至多個同 contract frames。"""
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, list(columns)]


def _combine_round_frames(
    primary: pd.DataFrame,
    exploratory: pd.DataFrame,
) -> pd.DataFrame:
    """標記 Round 1/2 後合併公開結果表。"""
    primary_round = primary.copy()
    primary_round["round"] = "primary_round_1"
    exploratory_round = exploratory.copy()
    exploratory_round["round"] = "exploratory_round_2"
    if primary_round.empty:
        return exploratory_round.reset_index(drop=True)
    if exploratory_round.empty:
        return primary_round.reset_index(drop=True)
    return pd.concat([primary_round, exploratory_round], ignore_index=True)


def _sha256_bytes(value: bytes) -> str:
    """計算 bytes 的小寫 SHA-256。"""
    return hashlib.sha256(value).hexdigest()


def _config_hash(config: Mapping[str, Any]) -> tuple[str, str]:
    """依 sanitized effective in-memory config 計算 SHA-256。"""
    evidence = _capture_config_evidence(config)
    return (
        str(evidence["effective_config_hash"]),
        str(evidence["config_hash_source"]),
    )


def _capture_config_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    """在 run entry 凍結 effective config 與來源檔案證據。

    Args:
        config: 呼叫當下的完整 Exp3 設定；runtime 私有欄位不會發布。

    Returns:
        可由 strict JSON 重建的 sanitized snapshot、effective hash，以及入口時
        原始設定檔的路徑與 bytes hash。
    """
    if not isinstance(config, Mapping):
        raise TypeError("Exp3 config 必須是 mapping")
    canonical = _canonical_json_value(config)
    if not isinstance(canonical, dict):
        raise TypeError("effective config snapshot 必須是 mapping")
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path_value = config.get("_config_path")
    entry_source = "in_memory_config"
    entry_file_hash: str | None = None
    if isinstance(path_value, (str, Path)):
        config_path = Path(path_value).resolve(strict=False)
        entry_source = str(config_path)
        if config_path.is_file():
            entry_file_hash = _sha256_bytes(config_path.read_bytes())
    return {
        "effective_config_snapshot": canonical,
        "effective_config_hash": _sha256_bytes(payload),
        "config_hash_source": "canonical_in_memory_config",
        "entry_config_file_hash": entry_file_hash,
        "entry_config_source": entry_source,
    }


def _required_config_evidence(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """取得入口凍結證據；直接呼叫 generation 時在該入口補捕捉。"""
    evidence = config.get("_config_evidence")
    if evidence is None:
        return _capture_config_evidence(config)
    if not isinstance(evidence, Mapping):
        raise TypeError("_config_evidence 必須是 mapping")
    required = {
        "effective_config_snapshot",
        "effective_config_hash",
        "config_hash_source",
        "entry_config_file_hash",
        "entry_config_source",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ValueError(f"_config_evidence 缺少欄位：{missing}")
    return evidence


def _canonical_json_value(value: Any) -> Any:
    """將 config 正規化成不含 runtime objects 的 deterministic JSON value。"""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not str(key).startswith("_")
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("config hash 不接受 NaN 或 Infinity")
        return value
    raise TypeError(f"config hash 不支援型別：{type(value).__name__}")


def _run_metadata(
    *,
    config: Mapping[str, Any],
    started_at: datetime,
    ended_at: datetime,
    elapsed_seconds: float,
    config_evidence: Mapping[str, Any],
    manifest_hash: str,
    enabled_feature_sets: Sequence[str],
    smoke_fovs_per_condition: int | None,
    pairing_qc: pd.DataFrame,
    analyzed_images: int,
    segmentation_qc: pd.DataFrame,
    failure_count: int,
) -> dict[str, Any]:
    """建立 strict-JSON-compatible reproducibility metadata。"""
    benchmark = _required_mapping(config, "benchmark")
    effective_snapshot = _canonical_json_value(
        config_evidence["effective_config_snapshot"]
    )
    effective_hash = str(config_evidence["effective_config_hash"])
    git_commit, dirty = _git_state()
    raw_pc = int(pd.to_numeric(pairing_qc["pc_count"], errors="raise").sum())
    raw_ido = int(pd.to_numeric(pairing_qc["ido_count"], errors="raise").sum())
    complete_pairs = int(pairing_qc["status"].eq("paired").sum())
    cache_hits = int(segmentation_qc["cache_status"].eq("reused").sum())
    return {
        "started_at_utc": _utc_text(started_at),
        "ended_at_utc": _utc_text(ended_at),
        "elapsed_seconds": float(elapsed_seconds),
        "runtime_seconds": float(elapsed_seconds),
        "git_commit": git_commit,
        "dirty": dirty,
        "config_hash": effective_hash,
        "config_hash_source": str(config_evidence["config_hash_source"]),
        "effective_config_hash": effective_hash,
        "effective_config_snapshot": effective_snapshot,
        "entry_config_file_hash": config_evidence["entry_config_file_hash"],
        "entry_config_source": str(config_evidence["entry_config_source"]),
        "manifest_hash": manifest_hash,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            name: _package_version(distribution)
            for name, distribution in (
                ("numpy", "numpy"),
                ("pandas", "pandas"),
                ("scikit-learn", "scikit-learn"),
                ("cellpose", "cellpose"),
            )
        },
        "seed": int(benchmark.get("seed", 42)),
        "seeds": {"benchmark": int(benchmark.get("seed", 42))},
        "enabled_feature_sets": list(enabled_feature_sets),
        "smoke_fovs_per_condition": smoke_fovs_per_condition,
        "input_counts": {
            "raw_pc": raw_pc,
            "raw_ido": raw_ido,
            "complete_pairs": complete_pairs,
            "analyzed_images": int(analyzed_images),
        },
        "cache_hits": {"mask_reused": cache_hits},
        "failure_count": int(failure_count),
    }


def _git_state() -> tuple[str, bool]:
    """讀取目前 worktree commit 與 dirty evidence。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, bool(status.strip())


def _package_version(distribution: str) -> str:
    """讀取 distribution version，不 import runtime package。"""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _utc_text(value: datetime) -> str:
    """以含 Z suffix 的 ISO-8601 表示 UTC timestamp。"""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
