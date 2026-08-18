"""原子寫入、發布並驗證 Exp3 Round 2 Paper93 結果。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from immunity.exp3.benchmark import make_outer_splits, outer_split_manifest
from immunity.exp3.feature_sets import (
    PAPER_STYLE_EXTRA_FEATURES,
    PAPER_STYLE_FOV_FEATURES,
    PRIMARY_FOV_FEATURES,
)
from immunity.exp3.round2_benchmark import (
    Round2Comparison,
    rank_round2_configurations,
    select_round2_recommendation,
)


ROUND2_TABLE_NAMES = (
    "data_snapshot.csv",
    "outer_splits.csv",
    "mask_provenance_qc.csv",
    "feature_valid_counts.csv",
    "extraction_qc.csv",
    "feature_qc.csv",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "dummy_fold_metrics.csv",
    "dummy_oof_predictions.csv",
    "hyperparameters.csv",
    "feature_importance.csv",
    "model_failures.csv",
    "feature_set_comparison.csv",
    "eligibility.csv",
)
ROUND2_JSON_NAMES = (
    "feature_sets.json",
    "baseline_provenance.json",
    "run_metadata.json",
)
_HASHED_ARTIFACT_NAMES = (*ROUND2_TABLE_NAMES, *ROUND2_JSON_NAMES, "EXPERIMENT_RECORD.md")
_BUNDLE_FILE_NAMES = (*_HASHED_ARTIFACT_NAMES, "artifact_hashes.json", "run.log")
_VALIDATIONS = (
    "leave_one_b_out",
    "leave_one_passage_out",
    "leave_one_group_out",
    "leave_one_condition_out",
)
_FOLD_COUNTS = dict(zip(_VALIDATIONS, (3, 3, 9, 8), strict=True))
_CONFIGURATIONS = {
    "extra_trees__basic_median": ("extra_trees", "basic_median", "round1"),
    "extra_trees__paper_style_median": (
        "extra_trees",
        "paper_style_median",
        "round2_paper93",
    ),
    "random_forest__basic_median": ("random_forest", "basic_median", "round1"),
    "random_forest__paper_style_median": (
        "random_forest",
        "paper_style_median",
        "round2_paper93",
    ),
}
_ROUND2_MODELS = {"extra_trees", "random_forest"}
_FROZEN_ROUND1_ROSTER_SHA256 = (
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
_METRIC_COLUMNS = ("mae", "rmse", "r2", "spearman")
_PUBLICATION_LOCK_NAME = ".publication.lock"


@dataclass(frozen=True)
class Round2Generation:
    """保存單次 Round 2 generation 的安全發布邊界。

    Attributes:
        output_dir: resolver 核准的正式根目錄或 smoke 子目錄。
        staging_dir: 位於 output boundary 內、尚未發布的唯一 staging 目錄。
    """

    output_dir: Path
    staging_dir: Path
    _owner_token: str = field(repr=False, compare=False)


@dataclass
class _PublicationOwnership:
    """保存單一 process 內 active generation 的 OS lock ownership。"""

    generation: Round2Generation
    output_dir: Path
    staging_dir: Path
    lock_handle: BinaryIO
    released: bool = False


_ACTIVE_PUBLICATIONS: dict[str, _PublicationOwnership] = {}


def begin_round2_generation(output_dir: Path) -> Round2Generation:
    """在核准的 Round 2 target 內建立唯一 staging generation。

    Args:
        output_dir: 正式 Round 2 根目錄或其 ``smoke`` 子目錄。

    Returns:
        含核准 target 與新 staging 目錄的 generation。

    Raises:
        ValueError: 路徑不在固定 target，或含 symlink/junction/reparse point。
    """
    target = _resolve_allowed_output(Path(output_dir))
    target.mkdir(parents=True, exist_ok=True)
    _assert_safe_directory(target, target)
    generations = target / "_generations"
    _assert_safe_child(target, generations)
    generations.mkdir(exist_ok=True)
    _assert_safe_directory(target, generations)
    lock_path = generations / _PUBLICATION_LOCK_NAME
    _assert_safe_child(target, lock_path)
    lock_handle = _acquire_publication_lock(target, lock_path)
    try:
        staging = generations / f"staging-{_generation_suffix()}"
        _assert_safe_child(target, staging)
        staging.mkdir()
        _assert_safe_directory(target, staging)
        owner_token = uuid.uuid4().hex
        generation = Round2Generation(
            output_dir=target,
            staging_dir=staging,
            _owner_token=owner_token,
        )
        _ACTIVE_PUBLICATIONS[owner_token] = _PublicationOwnership(
            generation=generation,
            output_dir=target,
            staging_dir=staging,
            lock_handle=lock_handle,
        )
        return generation
    except BaseException:
        _close_publication_lock(lock_handle)
        raise


def publish_round2_generation(
    generation: Round2Generation,
    *,
    post_publish_check: Callable[[Path, Path | None], None] | None = None,
    on_publish_failure: (
        Callable[[Round2Generation, BaseException], None] | None
    ) = None,
) -> None:
    """封存 target 內舊版 Round 2 檔案，並發布已驗證 staging。

    Args:
        generation: ``begin_round2_generation`` 建立的 active generation。
        post_publish_check: 可選的唯讀 transaction hook。Built-in published-root
            validation 完成後、刪除 staging 與釋放 OS lock 前，精確呼叫一次並傳入
            published root 與本次建立的 matching archive（沒有舊 bundle 時為
            ``None``）。Hook 拋出任何 ``BaseException`` 都會觸發原有 rollback。
        on_publish_failure: 可選的失敗 lifecycle hook。任何 publication
            ``BaseException`` 完成 rollback 後、仍持有相同 OS lock 時精確呼叫一次；
            可寫入 staging failure evidence 並呼叫
            ``quarantine_round2_generation``。Hook 自身失敗不會取代原始例外。

    Raises:
        ValueError: Boundary、bundle 或現有 destination 不符合固定合約。
        OSError: 檔案搬移失敗且已完成安全 rollback 時拋出。
    """
    ownership = _require_generation_ownership(generation)
    archive: Path | None = None
    archived: list[str] = []
    published: list[str] = []
    try:
        output, staging = _validate_generation(generation, ownership)
        if post_publish_check is not None and not callable(post_publish_check):
            raise TypeError("post_publish_check 必須是 callable 或 None")
        if on_publish_failure is not None and not callable(on_publish_failure):
            raise TypeError("on_publish_failure 必須是 callable 或 None")
        validate_round2_bundle(staging, smoke=output.name == "smoke")
        existing_files = _preflight_published_root(output)
        if existing_files:
            archive = output / "_generations" / f"archive-{_generation_suffix()}"
            _assert_safe_child(output, archive)
            archive.mkdir()
            _assert_safe_directory(output, archive)
            for name in _BUNDLE_FILE_NAMES:
                source = output / name
                if source.exists() or source.is_symlink():
                    os.replace(source, archive / name)
                    archived.append(name)
        for name in _BUNDLE_FILE_NAMES:
            source = staging / name
            destination = output / name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Round 2 destination 已存在：{name}")
            os.replace(source, destination)
            published.append(name)
        validate_round2_bundle(output, smoke=output.name == "smoke")
        if post_publish_check is not None:
            post_publish_check(output, archive)
        staging.rmdir()
    except BaseException as original_error:
        for name in reversed(published):
            try:
                destination = output / name
                if destination.exists() and not (staging / name).exists():
                    os.replace(destination, staging / name)
            except BaseException:  # noqa: BLE001 - rollback 不可掩蓋原始中斷
                pass
        if archive is not None:
            for name in reversed(archived):
                try:
                    source = archive / name
                    if source.exists() and not (output / name).exists():
                        os.replace(source, output / name)
                except BaseException:  # noqa: BLE001 - rollback 不可掩蓋原始中斷
                    pass
            try:
                if archive.exists() and not any(archive.iterdir()):
                    archive.rmdir()
            except BaseException:  # noqa: BLE001 - rollback 不可掩蓋原始中斷
                pass
        if callable(on_publish_failure):
            try:
                on_publish_failure(generation, original_error)
            except BaseException:  # noqa: BLE001 - cleanup 不可取代原始 publish error
                pass
        raise
    finally:
        _release_publication_ownership(ownership)


def _preflight_published_root(output: Path) -> list[Path]:
    """在首次 rename 前驗證 published root 的完整 child 類型契約。

    Args:
        output: 已核准的 formal root 或 smoke root。

    Returns:
        目前位於 root 的既有 bundle regular files。

    Raises:
        ValueError: 當 root 含未知 child、symlink 或不安全目錄時拋出。
    """
    children = list(output.iterdir())
    allowed_directories = {"_generations"}
    if output.name != "smoke":
        allowed_directories.add("smoke")
    existing_files: list[Path] = []
    unexpected: list[str] = []
    for child in children:
        if child.name in _BUNDLE_FILE_NAMES:
            _assert_regular_file(output, child)
            existing_files.append(child)
        elif child.name in allowed_directories:
            _assert_safe_directory(output, child)
        else:
            unexpected.append(child.name)
    if unexpected:
        raise ValueError(f"Round 2 output 含非預期 child：{sorted(unexpected)}")
    return existing_files


def quarantine_round2_generation(generation: Round2Generation) -> Path:
    """將失敗 staging 搬至相同 target boundary 的具名 quarantine。

    Args:
        generation: 尚未發布且含 ``run.log`` 的 generation。

    Returns:
        保留失敗 artifacts 與 QC log 的 ``failed-*`` 目錄。

    Raises:
        ValueError: Generation 越界、已消失，或 ``run.log`` 不安全。
    """
    ownership = _require_generation_ownership(generation)
    try:
        output, staging = _validate_generation(generation, ownership)
        _assert_regular_file(staging, staging / "run.log")
        failed = output / "_generations" / f"failed-{_generation_suffix()}"
        _assert_safe_child(output, failed)
        if failed.exists() or failed.is_symlink():
            raise FileExistsError(f"Quarantine destination 已存在：{failed}")
        os.replace(staging, failed)
        _assert_safe_directory(output, failed)
        return failed
    finally:
        _release_publication_ownership(ownership)


def write_round2_bundle(
    staging_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    json_payloads: Mapping[str, Mapping[str, Any]],
    record_context: Mapping[str, Any],
) -> Path:
    """以 sibling temp 與 replace 寫入固定 Round 2 bundle。

    Args:
        staging_dir: ``begin_round2_generation`` 建立的 staging 目錄。
        tables: Key 精確等於 ``ROUND2_TABLE_NAMES`` 的資料表 mapping。
        json_payloads: 三個固定 JSON artifact 的 mapping。
        record_context: 產生繁體中文實驗記錄所需的 context。

    Returns:
        寫入完成的 ``EXPERIMENT_RECORD.md`` 路徑。

    Raises:
        TypeError: Table 或 JSON root 類型錯誤。
        ValueError: Key、boundary 或 JSON finite 合約不符。
        OSError: 寫入失敗；本次已發布的檔案會先 rollback。
    """
    staging, output = _validate_staging_directory(Path(staging_dir))
    _require_exact_keys(tables, ROUND2_TABLE_NAMES, "Round 2 tables")
    _require_exact_keys(json_payloads, ROUND2_JSON_NAMES, "Round 2 JSON payloads")
    if not isinstance(record_context, Mapping):
        raise TypeError("record_context 必須是 mapping")

    payloads: dict[str, bytes] = {}
    for name in ROUND2_TABLE_NAMES:
        frame = tables[name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} 必須是 pandas DataFrame")
        payloads[name] = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    for name in ROUND2_JSON_NAMES:
        payloads[name] = _strict_json_bytes(json_payloads[name], name)
    record_data = dict(record_context)
    record_data["feature_importance_completeness"] = (
        _feature_importance_completeness(
            tables["fold_metrics.csv"],
            tables["feature_importance.csv"],
        )
    )
    payloads["EXPERIMENT_RECORD.md"] = build_round2_experiment_record(
        record_data
    ).encode("utf-8")

    destinations = [staging / name for name in (*_HASHED_ARTIFACT_NAMES, "artifact_hashes.json")]
    for destination in destinations:
        _assert_safe_child(output, destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Round 2 staging destination 已存在：{destination.name}")

    written: list[Path] = []
    temporary: list[Path] = []
    try:
        for name in _HASHED_ARTIFACT_NAMES:
            destination = staging / name
            temp = _write_sibling_temp(destination, payloads[name])
            temporary.append(temp)
            os.replace(temp, destination)
            temporary.remove(temp)
            written.append(destination)
        hash_payload = {
            "artifacts": {
                name: {
                    "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                    "bytes": len(payloads[name]),
                }
                for name in _HASHED_ARTIFACT_NAMES
            }
        }
        destination = staging / "artifact_hashes.json"
        temp = _write_sibling_temp(
            destination,
            _strict_json_bytes(hash_payload, "artifact_hashes.json"),
        )
        temporary.append(temp)
        os.replace(temp, destination)
        temporary.remove(temp)
        written.append(destination)
    except Exception:
        for path in temporary:
            path.unlink(missing_ok=True)
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    _assert_safe_directory(output, staging)
    return staging / "EXPERIMENT_RECORD.md"


def validate_round2_bundle(directory: Path, *, smoke: bool) -> None:
    """驗證 formal 或 smoke Round 2 bundle 的完整性與語意。

    Args:
        directory: 已發布 target 或 target 內的 staging 目錄。
        smoke: 是否套用只驗證流程、不形成科學結論的 smoke 合約。

    Raises:
        ValueError: Boundary、artifact、hash、schema、identity 或 row coverage 不符。
    """
    bundle, output = _validate_bundle_directory(Path(directory))
    if (output.name == "smoke") != smoke:
        raise ValueError("Bundle mode 必須與 formal root 或 smoke child boundary 一致")
    entries = {path.name for path in bundle.iterdir()}
    expected_entries = set(_BUNDLE_FILE_NAMES)
    if bundle == output:
        expected_entries.add("_generations")
        smoke_directory = output / "smoke"
        if not smoke and (smoke_directory.exists() or smoke_directory.is_symlink()):
            expected_entries.add("smoke")
            _assert_safe_directory(output, smoke_directory)
    if entries != expected_entries:
        missing = sorted(expected_entries - entries)
        extra = sorted(entries - expected_entries)
        raise ValueError(f"Round 2 bundle artifact keys 不符；missing={missing}, extra={extra}")
    if bundle == output:
        _assert_safe_directory(output, output / "_generations")
    for name in _BUNDLE_FILE_NAMES:
        _assert_regular_file(output, bundle / name)
    _validate_hash_manifest(bundle)

    tables = {name: pd.read_csv(bundle / name) for name in ROUND2_TABLE_NAMES}
    feature_sets = _read_json_mapping(bundle / "feature_sets.json")
    baseline_provenance = _read_json_mapping(bundle / "baseline_provenance.json")
    metadata = _read_json_mapping(bundle / "run_metadata.json")
    _validate_reproducibility_metadata(metadata)
    authority = _validate_baseline_provenance(
        baseline_provenance,
        metadata,
        output,
    )
    _validate_predictor_registry(feature_sets)
    _validate_common_tables(tables, metadata)
    _validate_authoritative_data_snapshot(
        tables["data_snapshot.csv"],
        authority["basic_images"],
        smoke=smoke,
    )
    if smoke:
        _validate_smoke_tables(
            tables,
            metadata,
            bundle,
            authority_basic_images=authority["basic_images"],
        )
    else:
        _validate_formal_tables(
            tables,
            metadata,
            authority_split_manifest=authority["outer_splits"],
        )
        _validate_authoritative_round1_baselines(tables, authority)


def build_round2_experiment_record(context: Mapping[str, Any]) -> str:
    """建立答案先行、限制科學解讀的繁體中文實驗記錄。

    Args:
        context: Recommendation、比較表、QC、provenance 與 runtime context。

    Returns:
        固定以結論開頭的 Markdown；smoke 會明示不形成科學結論。

    Raises:
        TypeError: ``context`` 不是 mapping 時拋出。
    """
    if not isinstance(context, Mapping):
        raise TypeError("record context 必須是 mapping")
    smoke = bool(context.get("smoke", False))
    recommendation = context.get("recommendation")
    if smoke:
        opening = (
            "僅驗證流程，不是正式實驗結果。此 smoke run 不形成 33 vs 93 科學結論，"
            "也不產生 recommendation。"
        )
        conclusion = "Smoke bundle 僅確認資料、特徵、模型與報告管線可完成。"
        plain = "本次不比較 93 與 33 predictors 的科學表現。"
    else:
        opening = "# Exp3 Round 2 實驗記錄"
        conclusion = (
            f"Recommendation：{recommendation}。"
            if recommendation
            else "沒有符合 gates 的 eligible recommendation。"
        )
        better = context.get("paper93_better_than_basic")
        if better is True:
            plain = "預先定義的 gates 支持 93 predictors 優於 33 predictors。"
        elif better is False:
            plain = "預先定義的 gates 不支持 93 predictors 優於 33 predictors。"
        else:
            plain = "是否優於 33 predictors 只依 eligibility 與 strict tie rule 判定。"

    images = _display(context.get("analyzed_images", 693))
    pairs = _display_count(
        context.get(
            "valid_pair_observations",
            context.get("valid_cells", 23976),
        )
    )
    exclusions = _display(context.get("exclusions", 26))
    seed = _display(context.get("seed", 20260804))
    runtime = _display(context.get("runtime_seconds", "未提供"))
    comparison = _markdown_context(context.get("comparison"))
    eligibility = _markdown_context(context.get("eligibility"))
    feature_qc = _markdown_context(context.get("feature_qc"))
    extraction_failures = _display(context.get("extraction_failures", "未提供"))
    pair_mapping = _markdown_context(context.get("pair_mapping_summary"))
    importance_completeness = _importance_completeness_text(
        context.get("feature_importance_completeness")
    )

    sections = [
        opening,
        "## 結論",
        conclusion,
        plain,
        "本次 secondary benchmark 不改寫 Round 1 winner。",
        "## Target 定義",
        (
            "Target 是 frozen FOV-level IDO 螢光亮度 proxy；它不是免疫力、功能性 assay "
            "或個體層級標籤，不可延伸作上述解讀。"
        ),
        "## Predictor 範圍",
        (
            "模型只使用 93 個已註冊的 phase-derived morphology predictors；這是 "
            "phase-only paper-style approximation。Target、實驗條件、路徑與批次資訊均不進入 "
            "predictor matrix。Predictors 不包含 donor label/identifier、IFN/TNF dose、"
            "condition、IDO channel 或 `IDO_score` target；IDO 僅作 target。"
        ),
        "## Data lock",
        (
            f"固定資料：{images} FOV、{pairs} frozen nucleus–cell pair observations、"
            f"{exclusions} exclusions；seed={seed}。\n\nPair mapping summary：{pair_mapping}"
        ),
        "## Pair aggregation limitation",
        (
            "每列 aggregation unit 是 frozen nucleus–cell pair。重複 cell label 會依 "
            "nucleus 數量加權 whole-cell descriptors；pair-specific cytoplasm 僅排除該列 "
            "current nucleus，因此同一 cell 的其他 nuclei 仍保留在 pair-specific cytoplasm。"
        ),
        "33 vs 93 的解讀只限於相同 frozen pair roster，不代表 unique cells 的獨立樣本比較。",
        "## 評估指標",
        (
            "MAE 是 primary（越低越好）；RMSE、R²、Spearman 是 secondary（RMSE "
            "越低越好，R² 與 Spearman 越高越好）。"
        ),
        "## 33 vs 93 比較",
        comparison,
        "## Eligibility gates 與 tie decision",
        (
            "Strict tie：只有 `average_rank - R* < 0.25` 才屬同一 band；等於 `0.25` 不算。"
            "若同一 algorithm 的 33-feature 與 93-feature 都在 band，保留 33-feature、"
            "淘汰 93-feature。"
        ),
        eligibility,
        "## Feature QC 與 extraction",
        f"Feature QC：{feature_qc}\n\nExtraction failures：{extraction_failures}。",
        "## Provenance、hashes、runtime 與限制",
        (
            f"Runtime seconds：{runtime}。完整 SHA-256 與 byte length 見 artifact_hashes.json；"
            f"Round 1 artifacts 為唯讀。{importance_completeness}"
        ),
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def _resolve_allowed_output(candidate: Path) -> Path:
    """以 Task 1 resolver 驗證 formal root 或 smoke child。"""
    import immunity.exp3.run_round2_paper93 as run_module

    root = Path(run_module.ROUND2_OUTPUT_ROOT)
    candidate_absolute = Path(os.path.abspath(candidate))
    root_absolute = Path(os.path.abspath(root))
    smoke_absolute = root_absolute / "smoke"
    if candidate_absolute == root_absolute:
        return run_module.resolve_round2_output_dir(root, smoke=False)
    if candidate_absolute == smoke_absolute:
        return run_module.resolve_round2_output_dir(root, smoke=True)
    raise ValueError("Round 2 generation 必須位於固定 formal root 或 smoke child")


def _acquire_publication_lock(output: Path, lock_path: Path) -> BinaryIO:
    """以 OS-held nonblocking exclusive lock 取得 target publication ownership。

    Lock file 只是位於 ``_generations`` 的穩定 inode；真正 ownership 由 OS file
    lock 持有，因此 process crash 會自動釋放，殘留 marker 不會永久阻塞。

    Args:
        output: 已核准的 formal 或 smoke target。
        lock_path: Target ``_generations`` 內的固定 lock file。

    Returns:
        持有 exclusive lock、必須由 lifecycle 終點關閉的 binary handle。

    Raises:
        RuntimeError: 同一 target 已有 active owner，或 OS 不支援所需 lock。
        ValueError: Lock path 不是安全的 target regular file。
    """
    try:
        handle = lock_path.open("a+b")
    except OSError as error:
        raise RuntimeError(f"無法開啟 Round 2 publication lock：{lock_path}") from error
    try:
        _assert_regular_file(output, lock_path)
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    f"Round 2 publication target 已有 active owner：{output}"
                ) from error
        elif os.name == "posix":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as error:
                raise RuntimeError(
                    f"Round 2 publication target 已有 active owner：{output}"
                ) from error
        else:
            raise RuntimeError(f"此平台不支援 Round 2 publication lock：{os.name}")
        return handle
    except BaseException:
        handle.close()
        raise


def _close_publication_lock(handle: BinaryIO) -> None:
    """釋放 OS lock 並關閉 handle；重複呼叫不產生副作用。"""
    if handle.closed:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _require_generation_ownership(
    generation: Round2Generation,
) -> _PublicationOwnership:
    """取得 active ownership，拒絕 forged、copied 或已完成的 generation。"""
    if not isinstance(generation, Round2Generation):
        raise TypeError("generation 必須是 Round2Generation")
    ownership = _ACTIVE_PUBLICATIONS.get(generation._owner_token)
    if (
        ownership is None
        or ownership.released
        or ownership.generation is not generation
        or ownership.lock_handle.closed
    ):
        raise ValueError("Round 2 generation 沒有 active publication ownership")
    return ownership


def _release_publication_ownership(ownership: _PublicationOwnership) -> None:
    """從 registry 移除 ownership，並精確一次釋放其 OS lock。"""
    if ownership.released:
        return
    ownership.released = True
    token = ownership.generation._owner_token
    if _ACTIVE_PUBLICATIONS.get(token) is ownership:
        del _ACTIVE_PUBLICATIONS[token]
    _close_publication_lock(ownership.lock_handle)


def _validate_generation(
    generation: Round2Generation,
    ownership: _PublicationOwnership | None = None,
) -> tuple[Path, Path]:
    """驗證 generation identity 與 staging boundary。"""
    if not isinstance(generation, Round2Generation):
        raise TypeError("generation 必須是 Round2Generation")
    active = ownership or _require_generation_ownership(generation)
    output = _resolve_allowed_output(generation.output_dir)
    staging = Path(generation.staging_dir)
    if active.output_dir != output or active.staging_dir != staging:
        raise ValueError("Round 2 generation ownership identity 不合法")
    if staging.parent != output / "_generations" or not staging.name.startswith("staging-"):
        raise ValueError("Round 2 staging identity 不合法")
    _assert_safe_directory(output, output)
    _assert_safe_directory(output, staging)
    return output, staging


def _validate_staging_directory(staging: Path) -> tuple[Path, Path]:
    """從 staging path 還原並驗證 output boundary。"""
    if staging.parent.name != "_generations" or not staging.name.startswith("staging-"):
        raise ValueError("Round 2 writer 只能寫入 staging generation")
    output = _resolve_allowed_output(staging.parent.parent)
    _assert_safe_directory(output, staging)
    return staging, output


def _validate_bundle_directory(directory: Path) -> tuple[Path, Path]:
    """接受已發布 target 或 target 內的 staging directory。"""
    import immunity.exp3.run_round2_paper93 as run_module

    absolute = Path(os.path.abspath(directory))
    root = Path(os.path.abspath(run_module.ROUND2_OUTPUT_ROOT))
    smoke = root / "smoke"
    if absolute in (root, smoke):
        output = _resolve_allowed_output(absolute)
    elif absolute.parent.name == "_generations" and absolute.name.startswith("staging-"):
        output = _resolve_allowed_output(absolute.parent.parent)
    else:
        raise ValueError("Round 2 bundle directory 不在核准 boundary")
    _assert_safe_directory(output, absolute)
    return absolute, output


def _assert_safe_child(root: Path, candidate: Path) -> None:
    """拒絕 lexical boundary 外或既有 reparse component。"""
    root_absolute = Path(os.path.abspath(root))
    candidate_absolute = Path(os.path.abspath(candidate))
    try:
        candidate_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError("Round 2 path 逃逸 output boundary") from error
    resolved_root = root_absolute.resolve(strict=False)
    resolved_candidate = candidate_absolute.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("Round 2 path 經 symlink/junction 逃逸 output boundary") from error
    for component in _existing_components(candidate_absolute):
        if _is_reparse_point(component):
            raise ValueError(f"Round 2 path 不可包含 symlink/junction/reparse point：{component}")


def _assert_safe_directory(root: Path, directory: Path) -> None:
    """驗證 boundary 內的真實 directory。"""
    _assert_safe_child(root, directory)
    if not directory.exists() or not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Round 2 directory 不存在或不安全：{directory}")


def _assert_regular_file(root: Path, path: Path) -> None:
    """驗證 boundary 內且非 reparse point 的 regular file。"""
    _assert_safe_child(root, path)
    try:
        status = os.lstat(path)
    except OSError as error:
        raise ValueError(f"Round 2 artifact 不存在：{path.name}") from error
    if _is_reparse_point(path) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"Round 2 artifact 必須是 regular file：{path.name}")


def _existing_components(path: Path) -> list[Path]:
    """回傳 path 中由 anchor 起連續存在的 lexical components。"""
    current = Path(path.anchor)
    components: list[Path] = []
    if current.exists() or current.is_symlink():
        components.append(current)
    for part in path.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        components.append(current)
    return components


def _is_reparse_point(path: Path) -> bool:
    """判斷 path 是否為 symlink 或 Windows junction/reparse point。"""
    try:
        status = os.lstat(path)
    except OSError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse)


def _generation_suffix() -> str:
    """產生可排序 UTC timestamp 與不可碰撞 UUID。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def _require_exact_keys(mapping: Mapping[str, Any], names: Sequence[str], label: str) -> None:
    """要求 mapping keys 精確吻合固定 artifact names。"""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{label} 必須是 mapping")
    actual = set(mapping)
    expected = set(names)
    if actual != expected:
        raise ValueError(
            f"{label} keys 必須精確吻合；missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _strict_json_bytes(payload: Mapping[str, Any], name: str) -> bytes:
    """序列化 root mapping 且拒絕所有 non-finite number。"""
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} JSON root 必須是 mapping")
    _require_finite_json(payload, name)
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是 finite JSON mapping：{error}") from error
    return (text + "\n").encode("utf-8")


def _require_finite_json(value: Any, name: str) -> None:
    """遞迴拒絕 JSON payload 內的 NaN 與 Infinity。"""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} JSON 數值必須 finite，不可為 NaN 或 Infinity")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} JSON mapping key 必須是字串")
            _require_finite_json(child, name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite_json(child, name)


def _write_sibling_temp(destination: Path, payload: bytes) -> Path:
    """以 exclusive create 寫入 destination 同層 temp。"""
    #Windows 未啟用 long-path policy 時仍有 MAX_PATH；staging 已唯一，可安全重用短名稱。
    temp = destination.with_name(".tmp")
    created = False
    try:
        handle = temp.open("xb")
        created = True
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if created:
            temp.unlink(missing_ok=True)
        raise
    return temp


def _validate_hash_manifest(directory: Path) -> None:
    """驗證 hash manifest 精確排除自身與 run.log。"""
    manifest = _read_json_mapping(directory / "artifact_hashes.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifact_hashes.json 缺少 artifacts mapping")
    if set(artifacts) != set(_HASHED_ARTIFACT_NAMES):
        raise ValueError("artifact hash keys 必須排除自身與 run.log，且精確涵蓋 bundle")
    for name in _HASHED_ARTIFACT_NAMES:
        item = artifacts[name]
        if not isinstance(item, Mapping) or set(item) != {"sha256", "bytes"}:
            raise ValueError(f"artifact hash entry 不合法：{name}")
        payload = (directory / name).read_bytes()
        if item["sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"{name} SHA-256 hash mismatch")
        if item["bytes"] != len(payload):
            raise ValueError(f"{name} byte length mismatch")


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    """讀取 strict finite JSON mapping。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path.name} 不是合法 finite JSON：{error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} JSON root 必須是 mapping")
    _require_finite_json(payload, path.name)
    return payload


def _reject_json_constant(value: str) -> None:
    """拒絕 JSON parser 的 NaN/Infinity extension。"""
    raise ValueError(f"JSON 數值必須 finite：{value}")


def _validate_predictor_registry(feature_sets: Mapping[str, Any]) -> None:
    """驗證 Paper93 與 basic predictor identities。"""
    paper = feature_sets.get("paper_style_median")
    basic = feature_sets.get("basic_median")
    if not isinstance(paper, list) or paper != list(PAPER_STYLE_FOV_FEATURES):
        raise ValueError("feature_sets.json 必須含精確 93 predictor identity")
    if not isinstance(basic, list) or basic != list(PRIMARY_FOV_FEATURES):
        raise ValueError("feature_sets.json 必須含精確 33 predictor identity")


def _validate_reproducibility_metadata(metadata: Mapping[str, Any]) -> None:
    """驗證 Git、runtime environment 與 UTC timing 的完整結構化證據。"""
    reproducibility = metadata.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise ValueError("run_metadata reproducibility 必須是 mapping")
    if set(reproducibility) != {"schema_version", "git", "python", "packages", "timing"}:
        raise ValueError("reproducibility schema keys 不合法")
    schema_version = reproducibility["schema_version"]
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("reproducibility schema_version 必須是 1")

    git = reproducibility["git"]
    if not isinstance(git, Mapping) or set(git) != {
        "commit",
        "branch",
        "status_porcelain",
        "dirty",
    }:
        raise ValueError("Git reproducibility schema 不合法")
    commit = git["commit"]
    if not (
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdefABCDEF" for character in commit)
    ):
        raise ValueError("Git commit 必須是完整 40 位元十六進位 identity")
    if not isinstance(git["branch"], str) or not git["branch"].strip():
        raise ValueError("Git branch 不可為空")
    status = git["status_porcelain"]
    dirty = git["dirty"]
    if not isinstance(status, str) or not isinstance(dirty, bool):
        raise ValueError("Git status_porcelain/dirty 型別不合法")
    if dirty is not bool(status):
        raise ValueError("Git dirty 必須與 status_porcelain 一致")

    python = reproducibility["python"]
    if (
        not isinstance(python, Mapping)
        or set(python) != {"version"}
        or not isinstance(python["version"], str)
        or not python["version"].strip()
    ):
        raise ValueError("Python reproducibility version 不合法")
    packages = reproducibility["packages"]
    expected_packages = {"numpy", "pandas", "scikit-learn", "OpenCV", "PyYAML"}
    if not isinstance(packages, Mapping) or set(packages) != expected_packages:
        raise ValueError("reproducibility package keys 不合法")
    if any(not isinstance(value, str) or not value.strip() for value in packages.values()):
        raise ValueError("reproducibility package versions 必須是非空字串")

    timing = reproducibility["timing"]
    if not isinstance(timing, Mapping) or set(timing) != {
        "started_at_utc",
        "completed_at_utc",
        "runtime_seconds",
    }:
        raise ValueError("reproducibility timing schema 不合法")
    started = _parse_utc_timestamp(timing["started_at_utc"], "started_at_utc")
    completed = _parse_utc_timestamp(timing["completed_at_utc"], "completed_at_utc")
    runtime = timing["runtime_seconds"]
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        raise ValueError("reproducibility runtime_seconds 必須是 finite nonnegative number")
    duration = (completed - started).total_seconds()
    if duration < 0.0:
        raise ValueError("UTC completion 不得早於 start")
    if abs(duration - float(runtime)) > 1.0:
        raise ValueError(
            "reproducibility UTC duration 與 runtime_seconds 不一致："
            f"duration={duration}, runtime={runtime}"
        )


def _parse_utc_timestamp(value: object, name: str) -> datetime:
    """解析 canonical ``Z`` UTC timestamp。"""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"UTC {name} 必須是 Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"UTC {name} 格式不合法") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"UTC {name} offset 不合法")
    return parsed


def _validate_baseline_provenance(
    provenance: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """由核准 output 推導 Round 1 authority，重驗 bytes 並載入語意錨點。

    Args:
        provenance: Bundle 聲明的 Round 1 provenance。
        metadata: Bundle 的 original/effective config evidence。
        output: 已由 boundary resolver 核准的 formal root 或 smoke child。

    Returns:
        權威 ``image_level_basic`` 與 ``outer_splits`` 表格。

    Raises:
        ValueError: Root、pin、bytes 或必要權威表格不一致時拋出。
    """
    expected_keys = {
        "round1_root",
        "read_only",
        "artifact_sha256",
        "roster_sha256",
    }
    if set(provenance) != expected_keys or provenance.get("read_only") is not True:
        raise ValueError("baseline provenance schema/read_only 不合法")
    root = provenance.get("round1_root")
    artifacts = provenance.get("artifact_sha256")
    roster = provenance.get("roster_sha256")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("baseline provenance round1_root 不合法")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        _ROUND1_REQUIRED_ARTIFACTS
    ):
        raise ValueError("baseline provenance artifact_sha256 必須精確列出 14 個 artifacts")
    if not all(
        isinstance(name, str) and name
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdefABCDEF" for character in digest)
        for name, digest in artifacts.items()
    ):
        raise ValueError("baseline provenance artifact hash identity 不合法")
    if not (
        isinstance(roster, str)
        and len(roster) == 64
        and all(character in "0123456789abcdefABCDEF" for character in roster)
    ):
        raise ValueError("baseline provenance roster hash identity 不合法")
    if roster.lower() != _FROZEN_ROUND1_ROSTER_SHA256:
        raise ValueError("baseline provenance roster 必須等於 frozen roster authority")
    formal_output = output.parent if output.name == "smoke" else output
    authority_root = formal_output.parent.resolve(strict=True)
    project_root = authority_root.parents[2]
    try:
        declared_root = _resolve_round1_declaration(root, project_root)
    except OSError as error:
        raise ValueError("baseline provenance Round 1 root 不存在") from error
    if declared_root != authority_root:
        raise ValueError("baseline provenance Round 1 root 必須是 output 推導的 sibling root")

    for config_name in ("original_config", "effective_config"):
        config = metadata.get(config_name)
        round1 = config.get("round1") if isinstance(config, Mapping) else None
        if not isinstance(round1, Mapping):
            raise ValueError(f"baseline provenance 缺少 {config_name}.round1 evidence")
        if dict(round1.get("artifact_sha256", {})) != dict(artifacts):
            raise ValueError(f"baseline provenance artifact hashes 與 {config_name} 不一致")
        config_roster = round1.get("roster_sha256")
        if not isinstance(config_roster, str) or config_roster.lower() != roster.lower():
            raise ValueError(f"baseline provenance roster hash 與 {config_name} 不一致")
        configured_root = round1.get("dir")
        if not isinstance(configured_root, str) or not configured_root.strip():
            raise ValueError(f"baseline provenance {config_name} round1.dir 不合法")
        try:
            resolved_config_root = _resolve_round1_declaration(
                configured_root,
                project_root,
            )
        except OSError as error:
            raise ValueError(
                f"baseline provenance {config_name} Round 1 root 不存在"
            ) from error
        if resolved_config_root != authority_root:
            raise ValueError(
                f"baseline provenance {config_name} Round 1 root 不是推導的 sibling root"
            )

    paths: dict[str, Path] = {}
    for relative in _ROUND1_REQUIRED_ARTIFACTS:
        path = authority_root / Path(relative)
        _assert_regular_file(authority_root, path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(artifacts[relative]).lower():
            raise ValueError(
                f"Round 1 artifact {relative} SHA-256 不符合 pin："
                f"expected={artifacts[relative]}, actual={actual}"
            )
        paths[relative] = path
    authority = _load_round1_authority(authority_root)
    expected_authority_keys = {
        "basic_images",
        "outer_splits",
        "selected_metrics",
        "selected_predictions",
        "selected_hyperparameters",
        "selected_importance",
        "selected_failures",
        "roster_sha256",
    }
    if set(authority) != expected_authority_keys:
        raise ValueError("Round 1 authority loader contract 不合法")
    if str(authority["roster_sha256"]).lower() != roster.lower():
        raise ValueError("baseline provenance roster 偏離 Round 1 authority")
    return authority


def _resolve_round1_declaration(value: str, project_root: Path) -> Path:
    """安全解析 Round 1 root 聲明。

    Args:
        value: Bundle 或 config 宣告的 Round 1 root。
        project_root: 由核准 output boundary 推導的固定專案根。

    Returns:
        已存在且解析完成的絕對路徑。

    Raises:
        ValueError: Relative path 含 traversal、Windows 特殊 root 或 reparse
            component，或解析後逃離專案根時拋出。
        OSError: 路徑不存在或無法解析時拋出。
    """
    windows_path = PureWindowsPath(value)
    if windows_path.drive and not windows_path.root:
        raise ValueError("Round 1 root 不可使用 drive-relative 路徑")
    if windows_path.root and not windows_path.drive:
        raise ValueError("Round 1 root 不可使用 Windows rooted-relative 路徑")
    declared = Path(value)
    if declared.is_absolute():
        return declared.resolve(strict=True)
    if any(part == ".." for part in windows_path.parts):
        raise ValueError("Round 1 relative root 不可含 .. traversal")

    project_absolute = Path(os.path.abspath(project_root))
    project_resolved = project_absolute.resolve(strict=True)
    lexical = project_absolute / declared
    current = project_absolute
    for part in declared.parts:
        if part in ("", "."):
            continue
        current /= part
        if (current.exists() or current.is_symlink()) and _is_reparse_point(current):
            raise ValueError(
                "Round 1 relative root 不可包含 symlink/junction/reparse component"
            )
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(project_resolved)
    except ValueError as error:
        raise ValueError("Round 1 relative root 經解析後逃逸 project boundary") from error
    return resolved


def _load_round1_authority(root: Path) -> dict[str, Any]:
    """以程式碼固定 pins 載入完整 Round 1 evidence，拒絕 bundle 自行 repin。"""
    from immunity.exp3 import round2_evidence as evidence_module

    expected = {
        "raw_pc": 720,
        "raw_ido": 719,
        "paired": 719,
        "exclusions": 26,
        "analyzed_images": 693,
        "valid_cells": 23976,
        "outer_folds": 23,
        "seed": 20260804,
        "manifest_hash": evidence_module._FROZEN_ARTIFACT_HASHES[
            "data_manifest.csv"
        ],
        "config_hash": "b7590ed3438cdb5062b2588e525e603dd0b7610d397322c761facdaea5b54bc0",
    }
    evidence = evidence_module.load_round1_evidence(
        {
            "round1": {
                "dir": root,
                "expected": expected,
                "roster_sha256": evidence_module._FROZEN_ROSTER_HASH,
                "artifact_sha256": dict(
                    evidence_module._FROZEN_ARTIFACT_HASHES
                ),
            }
        }
    )
    evidence_module.require_formal_round1_evidence(evidence)
    return {
        "basic_images": evidence.basic_images.copy(deep=True),
        "outer_splits": evidence.split_manifest.copy(deep=True),
        "selected_metrics": evidence.selected_metrics.copy(deep=True),
        "selected_predictions": evidence.selected_predictions.copy(deep=True),
        "selected_hyperparameters": evidence.selected_hyperparameters.copy(deep=True),
        "selected_importance": evidence.selected_importance.copy(deep=True),
        "selected_failures": evidence.selected_failures.copy(deep=True),
        "roster_sha256": evidence_module._FROZEN_ROSTER_HASH,
    }


def _validate_authoritative_round1_baselines(
    tables: Mapping[str, pd.DataFrame],
    authority: Mapping[str, Any],
) -> None:
    """將正式 bundle 的 Basic33／Dummy raw evidence 對回 frozen Round 1。"""
    models = {"extra_trees", "random_forest"}

    def selected(frame: pd.DataFrame, names: set[str]) -> pd.DataFrame:
        return frame.loc[frame["model"].astype(str).isin(names)].copy()

    published_metrics = tables["fold_metrics.csv"].loc[
        tables["fold_metrics.csv"]["source_round"].astype(str).eq("round1")
    ]
    published_predictions = tables["oof_predictions.csv"].loc[
        tables["oof_predictions.csv"]["source_round"].astype(str).eq("round1")
    ]
    published_hyperparameters = tables["hyperparameters.csv"].loc[
        tables["hyperparameters.csv"]["source_round"].astype(str).eq("round1")
    ]
    published_importance = tables["feature_importance.csv"].loc[
        tables["feature_importance.csv"]["source_round"].astype(str).eq("round1")
    ]
    published_failures = tables["model_failures.csv"].loc[
        tables["model_failures.csv"]["source_round"].astype(str).eq("round1")
    ]
    comparisons = (
        (
            published_metrics,
            selected(authority["selected_metrics"], models),
            "candidate metrics",
        ),
        (
            tables["dummy_fold_metrics.csv"],
            selected(authority["selected_metrics"], {"dummy_median"}),
            "Dummy metrics",
        ),
        (
            published_predictions,
            selected(authority["selected_predictions"], models),
            "candidate OOF",
        ),
        (
            tables["dummy_oof_predictions.csv"],
            selected(authority["selected_predictions"], {"dummy_median"}),
            "Dummy OOF",
        ),
        (
            published_hyperparameters,
            selected(authority["selected_hyperparameters"], models),
            "hyperparameters",
        ),
        (
            published_importance,
            selected(authority["selected_importance"], models),
            "feature importance",
        ),
        (
            published_failures,
            selected(authority["selected_failures"], models),
            "model failures",
        ),
    )
    for published, expected, name in comparisons:
        _compare_round1_raw_evidence(published, expected, name)


def _compare_round1_raw_evidence(
    published: pd.DataFrame,
    expected: pd.DataFrame,
    name: str,
) -> None:
    """以 canonical dtype 與 exact values 比對 Round 1 raw evidence。"""
    columns = list(expected.columns)
    _require_columns(published, columns, f"Round 1 baseline {name}")
    #Round 2 publisher 會把已載入的 Round 1 table 再寫成一次 CSV；只將
    #authority 投影成該 canonical emission，published values 不可再次 round-trip。
    emitted_authority = pd.read_csv(StringIO(expected.to_csv(index=False)))
    schema = emitted_authority.dtypes.to_dict()
    actual = _canonical_round1_raw_frame(
        published.loc[:, columns],
        schema,
        f"published {name}",
    )
    canonical = _canonical_round1_raw_frame(
        emitted_authority.loc[:, columns],
        schema,
        f"authority {name}",
    )
    sort_columns = [
        column
        for column in (
            "validation",
            "fold",
            "split_id",
            "model",
            "image_key",
            "feature",
        )
        if column in columns
    ]
    actual = actual.sort_values(
        sort_columns,
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    canonical = canonical.sort_values(
        sort_columns,
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual,
            canonical,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            f"Round 1 baseline raw evidence 偏離 pinned baseline（{name}）：{error}"
        ) from error


def _canonical_round1_raw_frame(
    frame: pd.DataFrame,
    schema: Mapping[str, Any],
    name: str,
) -> pd.DataFrame:
    """依 authority dtype 將 CSV loader 差異正規化為 exact comparison frame。

    Args:
        frame: 要正規化的 raw evidence table。
        schema: Authority 每欄的 pandas dtype。
        name: 錯誤訊息使用的表格名稱。

    Returns:
        Text 統一為 pandas string、integer 統一為 int64、floating point
        統一為 float64 的新表格。

    Raises:
        ValueError: Numeric 欄含 bool、non-numeric、non-finite integer、fractional
            integer 或不支援的 dtype 時拋出。
    """
    normalized: dict[str, pd.Series] = {}
    for column, dtype in schema.items():
        values = frame[column].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(dtype):
            if not pd.api.types.is_bool_dtype(values.dtype):
                raise ValueError(f"{name} {column} canonical boolean dtype 不一致")
            normalized[column] = values.astype(bool)
            continue
        if pd.api.types.is_integer_dtype(dtype):
            if pd.api.types.is_bool_dtype(values.dtype) or values.map(
                lambda value: isinstance(value, (bool, np.bool_))
            ).any():
                raise ValueError(f"{name} {column} canonical integer 不可接受 boolean")
            try:
                numeric = pd.to_numeric(values, errors="raise")
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{name} {column} canonical integer dtype 不合法"
                ) from error
            array = numeric.to_numpy(dtype=float)
            if (
                not np.isfinite(array).all()
                or not np.equal(array, np.floor(array)).all()
            ):
                raise ValueError(f"{name} {column} canonical integer value 不合法")
            try:
                normalized[column] = numeric.astype("int64")
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{name} {column} canonical integer 超出 int64"
                ) from error
            continue
        if pd.api.types.is_float_dtype(dtype):
            if pd.api.types.is_bool_dtype(values.dtype) or values.map(
                lambda value: isinstance(value, (bool, np.bool_))
            ).any():
                raise ValueError(f"{name} {column} canonical numeric 不可接受 boolean")
            try:
                normalized[column] = pd.to_numeric(values, errors="raise").astype(
                    "float64"
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{name} {column} canonical floating dtype 不合法"
                ) from error
            continue
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            normalized[column] = values.astype("string")
            continue
        raise ValueError(f"{name} {column} authority dtype 不支援：{dtype}")
    return pd.DataFrame(normalized, columns=list(schema))


def _validate_authoritative_data_snapshot(
    data: pd.DataFrame,
    authority: pd.DataFrame,
    *,
    smoke: bool,
) -> None:
    """以 Round 1 image cache 鎖定 image keys、target 與 33 basic predictors。"""
    columns = ("image_key", "IDO_score", *PRIMARY_FOV_FEATURES)
    _require_columns(authority, columns, "Round 1 image_level_basic authority")
    authority_keys = authority["image_key"].astype(str)
    data_keys = data["image_key"].astype(str)
    if authority_keys.duplicated().any() or data_keys.duplicated().any():
        raise ValueError("Round 1 authority/data_snapshot image_key 不可重複")
    if smoke:
        if not set(data_keys) < set(authority_keys):
            raise ValueError("Smoke data_snapshot 必須是 Round 1 authority strict subset")
    elif set(data_keys) != set(authority_keys) or len(data) != len(authority):
        raise ValueError("Formal data_snapshot image keys 必須等於 Round 1 authority")
    expected = authority.assign(_image_key=authority_keys).set_index("_image_key")
    actual = data.assign(_image_key=data_keys).set_index("_image_key")
    ordered = expected.loc[actual.index, ["IDO_score", *PRIMARY_FOV_FEATURES]]
    actual_numeric = actual.loc[:, ["IDO_score", *PRIMARY_FOV_FEATURES]].apply(
        pd.to_numeric, errors="coerce"
    )
    expected_numeric = ordered.apply(pd.to_numeric, errors="coerce")
    if not np.isclose(
        actual_numeric.to_numpy(dtype=float),
        expected_numeric.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ).all():
        raise ValueError("data_snapshot target/basic predictors 偏離 Round 1 authority")


def _strict_integer(value: object, name: str, *, minimum: int = 0) -> int:
    """解析 QC integer 並拒絕 bool、非有限值、fraction 與低於下限。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必須是 >= {minimum} 的 strict integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必須是 >= {minimum} 的 strict integer") from error
    if not math.isfinite(numeric) or numeric < minimum or numeric != math.floor(numeric):
        raise ValueError(f"{name} 必須是 >= {minimum} 的 strict integer")
    return int(numeric)


def _finite_float(value: object, name: str) -> float:
    """解析非 bool 的 finite 浮點 QC 值。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必須是 finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必須是 finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} 必須是 finite number")
    return numeric


def _canonical_csv_float(value: object, name: str) -> float:
    """將 metadata float 正規化為 Round 2 CSV 輸出後的 pandas parse 值。"""
    numeric = _finite_float(value, name)
    payload = pd.DataFrame({"value": [numeric]}).to_csv(
        index=False, lineterminator="\n"
    )
    return float(pd.read_csv(StringIO(payload)).iloc[0, 0])


def _validate_common_tables(
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> None:
    """驗證 formal/smoke 共用的 93-feature schema 與 QC tables。"""
    data = tables["data_snapshot.csv"]
    _require_columns(data, ("image_key", "IDO_score", *PAPER_STYLE_FOV_FEATURES), "data_snapshot")
    if data.empty or data["image_key"].duplicated().any():
        raise ValueError("data_snapshot image_key 必須非空且唯一")
    numeric_columns = ["IDO_score", *PAPER_STYLE_FOV_FEATURES]
    numeric = data.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("data_snapshot 的 target 與 93 predictors 必須 numeric finite")
    image_keys = set(data["image_key"].astype(str))

    mask_qc = tables["mask_provenance_qc.csv"]
    _require_columns(mask_qc, ("image_key", "status"), "mask_provenance_qc")
    mask_keys = mask_qc["image_key"].astype(str)
    if (
        len(mask_qc) != len(image_keys)
        or mask_keys.duplicated().any()
        or set(mask_keys) != image_keys
        or not mask_qc["status"].astype(str).eq("passed").all()
    ):
        raise ValueError("mask provenance QC 必須逐一涵蓋 data images 且全部 passed")

    valid_rosters = _validate_published_valid_counts(
        tables["feature_valid_counts.csv"], image_keys
    )
    _validate_published_extraction_qc(
        tables["extraction_qc.csv"],
        image_keys,
        valid_rosters,
        metadata,
    )
    feature_qc = tables["feature_qc.csv"]
    _require_columns(feature_qc, ("feature", "status"), "feature_qc")
    if (
        set(feature_qc["feature"].astype(str)) != set(PAPER_STYLE_FOV_FEATURES)
        or len(feature_qc) != 93
    ):
        raise ValueError("feature_qc 必須精確包含 93 predictor identity")
    if not feature_qc["status"].astype(str).eq("passed").all():
        raise ValueError("feature_qc 的 93 predictors 必須全部 passed")
    _require_columns(
        tables["fold_metrics.csv"],
        _IDENTITY_COLUMNS + ("n_test", "status"),
        "fold_metrics",
    )
    _require_columns(
        tables["oof_predictions.csv"],
        _IDENTITY_COLUMNS
        + ("image_key", "observed_ido_score", "predicted_ido_score"),
        "oof_predictions",
    )
    _validate_oof_values(tables["oof_predictions.csv"], data, "candidate OOF")
    _require_columns(tables["hyperparameters.csv"], _IDENTITY_COLUMNS, "hyperparameters")
    _require_columns(
        tables["eligibility.csv"],
        ("configuration_id", "eligible", "recommended"),
        "eligibility",
    )


def _validate_published_valid_counts(
    frame: pd.DataFrame,
    image_keys: set[str],
) -> dict[str, int]:
    """驗證 data FOV × 60 extras 的 strict integer count identities。"""
    required = (
        "image_key",
        "feature",
        "roster_cell_count",
        "finite_cell_count",
        "required_minimum",
        "status",
    )
    _require_columns(frame, required, "feature_valid_counts")
    extras = set(PAPER_STYLE_EXTRA_FEATURES)
    identities = list(
        frame.loc[:, ["image_key", "feature"]].astype(str).itertuples(
            index=False, name=None
        )
    )
    expected = {(image_key, feature) for image_key in image_keys for feature in extras}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("feature valid-count 必須精確涵蓋 data FOV × 60 extras")
    roster_by_image: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        image_key = str(row.image_key)
        roster = _strict_integer(row.roster_cell_count, "roster_cell_count", minimum=1)
        finite = _strict_integer(row.finite_cell_count, "finite_cell_count", minimum=0)
        minimum = _strict_integer(row.required_minimum, "required_minimum", minimum=1)
        if minimum != 3:
            raise ValueError("feature valid-count required_minimum 必須精確為 3")
        if finite > roster:
            raise ValueError("feature valid-count finite_cell_count 不可大於 roster")
        expected_status = "passed" if finite >= minimum else "failed"
        if str(row.status) != expected_status or expected_status != "passed":
            raise ValueError("feature valid-count minimum/status/count 不一致或未通過")
        previous = roster_by_image.setdefault(image_key, roster)
        if previous != roster:
            raise ValueError("同一 data FOV 的 valid-count roster 必須全表一致")
    return roster_by_image


def _validate_published_extraction_qc(
    frame: pd.DataFrame,
    image_keys: set[str],
    valid_rosters: Mapping[str, int],
    metadata: Mapping[str, Any],
) -> None:
    """重算 extraction pair counts、outside boundary 與 metadata totals。"""
    count_fields = (
        "roster_pair_count",
        "extracted_pair_count",
        "unique_cell_count",
        "unique_nucleus_count",
        "multi_nucleus_cell_count",
        "multi_nucleus_pair_count",
        "max_nuclei_per_cell",
        "retained_outside_pair_count",
    )
    required = (
        "image_key",
        "aggregation_unit",
        *count_fields,
        "max_retained_outside_fraction",
        "max_nucleus_outside_fraction",
        "status",
    )
    _require_columns(frame, required, "extraction_qc")
    keys = frame["image_key"].astype(str)
    if (
        len(frame) != len(image_keys)
        or keys.duplicated().any()
        or set(keys) != image_keys
        or not frame["status"].astype(str).eq("passed").all()
    ):
        raise ValueError("extraction QC 必須逐一涵蓋 data images 且全部 passed")
    totals = {field: 0 for field in count_fields}
    maximum_nuclei = 0
    maximum_outside = 0.0
    for row in frame.itertuples(index=False):
        image_key = str(row.image_key)
        if str(row.aggregation_unit) != "frozen_nucleus_cell_pair":
            raise ValueError("extraction QC aggregation unit 必須是 frozen nucleus-cell pair")
        counts = {
            field: _strict_integer(
                getattr(row, field),
                field,
                minimum=(1 if field in {
                    "roster_pair_count",
                    "extracted_pair_count",
                    "unique_cell_count",
                    "unique_nucleus_count",
                    "max_nuclei_per_cell",
                } else 0),
            )
            for field in count_fields
        }
        if (
            counts["roster_pair_count"] != counts["extracted_pair_count"]
            or counts["roster_pair_count"] != valid_rosters[image_key]
        ):
            raise ValueError("extraction QC extracted=roster 與 valid-count identity 不一致")
        extracted = counts["extracted_pair_count"]
        multi_cells = counts["multi_nucleus_cell_count"]
        multi_pairs = counts["multi_nucleus_pair_count"]
        if (
            counts["unique_cell_count"] > extracted
            or counts["unique_nucleus_count"] != extracted
            or multi_cells > counts["unique_cell_count"]
            or multi_pairs > extracted
            or (multi_cells == 0) != (multi_pairs == 0)
            or (multi_cells > 0 and multi_pairs < 2 * multi_cells)
            or counts["retained_outside_pair_count"] > extracted
        ):
            raise ValueError("extraction QC pair topology counts 不一致")
        retained = _finite_float(
            row.max_retained_outside_fraction,
            "max_retained_outside_fraction",
        )
        threshold = _finite_float(
            row.max_nucleus_outside_fraction,
            "max_nucleus_outside_fraction",
        )
        if threshold != 0.05 or retained < 0.0 or retained > threshold:
            raise ValueError("extraction QC outside fraction 必須位於 frozen <=0.05 boundary")
        for field, value in counts.items():
            totals[field] += value
        maximum_nuclei = max(maximum_nuclei, counts["max_nuclei_per_cell"])
        maximum_outside = max(maximum_outside, retained)
    _validate_pair_metadata_totals(
        metadata,
        image_count=len(image_keys),
        totals=totals,
        maximum_nuclei=maximum_nuclei,
        maximum_outside=maximum_outside,
    )


def _validate_pair_metadata_totals(
    metadata: Mapping[str, Any],
    *,
    image_count: int,
    totals: Mapping[str, int],
    maximum_nuclei: int,
    maximum_outside: float,
) -> None:
    """以 extraction rows 對帳 metadata 與 pair summary。"""
    summary = metadata.get("pair_mapping_summary")
    expected_keys = {
        "aggregation_unit",
        "image_count",
        "pair_observation_count",
        "unique_cell_count",
        "unique_nucleus_count",
        "multi_nucleus_cell_count",
        "multi_nucleus_pair_count",
        "max_nuclei_per_cell",
        "retained_outside_pair_count",
        "max_retained_outside_fraction",
        "max_nucleus_outside_fraction",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected_keys:
        raise ValueError("metadata pair_mapping_summary schema 不合法")
    expected_counts = {
        "image_count": image_count,
        "pair_observation_count": totals["extracted_pair_count"],
        "unique_cell_count": totals["unique_cell_count"],
        "unique_nucleus_count": totals["unique_nucleus_count"],
        "multi_nucleus_cell_count": totals["multi_nucleus_cell_count"],
        "multi_nucleus_pair_count": totals["multi_nucleus_pair_count"],
        "max_nuclei_per_cell": maximum_nuclei,
        "retained_outside_pair_count": totals["retained_outside_pair_count"],
    }
    for field, expected in expected_counts.items():
        if _strict_integer(summary[field], f"pair_mapping_summary.{field}") != expected:
            raise ValueError(f"metadata pair summary {field} 與 extraction QC 不一致")
    if summary["aggregation_unit"] != "frozen_nucleus_cell_pair":
        raise ValueError("metadata aggregation unit 不合法")
    if (
        _canonical_csv_float(summary["max_retained_outside_fraction"], "summary outside")
        != maximum_outside
        or _finite_float(summary["max_nucleus_outside_fraction"], "summary threshold")
        != 0.05
    ):
        raise ValueError("metadata outside fraction summary 與 extraction QC 不一致")
    for field, expected in (
        ("analyzed_images", image_count),
        ("valid_cells", totals["extracted_pair_count"]),
        ("valid_pair_observations", totals["extracted_pair_count"]),
    ):
        if _strict_integer(metadata.get(field), field) != expected:
            raise ValueError(f"metadata {field} 與 extraction QC 不一致")
    if metadata.get("aggregation_unit") != "frozen_nucleus_cell_pair":
        raise ValueError("metadata aggregation_unit 不合法")


_IDENTITY_COLUMNS = (
    "validation",
    "fold",
    "split_id",
    "model",
    "feature_set",
    "source_round",
    "configuration_id",
)


def _validate_formal_tables(
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    *,
    authority_split_manifest: pd.DataFrame,
) -> None:
    """驗證正式 693-FOV publication gates。"""
    data = tables["data_snapshot.csv"]
    if len(data) != 693:
        raise ValueError("Formal data_snapshot 必須精確為 693 rows")
    if (
        metadata.get("mode") != "formal"
        or metadata.get("smoke") is not False
        or metadata.get("seed") != 20260804
        or metadata.get("diagnostic_splits") is not False
        or metadata.get("non_formal") is not False
        or metadata.get("no_scientific_conclusion") is not False
    ):
        raise ValueError("Formal run_metadata mode/seed/non-scientific flags 不合法")
    split_membership = _formal_split_membership(
        tables["outer_splits.csv"], set(data["image_key"].astype(str))
    )
    authority_membership = _formal_split_membership(
        authority_split_manifest,
        set(data["image_key"].astype(str)),
    )
    if split_membership != authority_membership:
        raise ValueError("Published outer_splits membership 偏離 Round 1 authority")
    splits = set(split_membership)
    expected_metric_identities = {
        (configuration, split_id)
        for configuration in _CONFIGURATIONS
        for split_id in splits
    }
    metrics = tables["fold_metrics.csv"].copy()
    if len(metrics) != 92:
        raise ValueError("Formal fold_metrics 必須精確包含 92 個 metric identities")
    _validate_candidate_identities(metrics, "fold_metrics")
    actual_metric_identities = _identity_pairs(metrics)
    if (
        len(actual_metric_identities) != len(metrics)
        or actual_metric_identities != expected_metric_identities
    ):
        raise ValueError("Formal fold_metrics 有 duplicate、missing fold 或非預期 identity")
    statuses = set(metrics["status"].astype(str))
    if not statuses.issubset({"ok", "failed"}):
        raise ValueError("fold_metrics status 只能是 ok/failed")
    failed_rows = metrics[metrics["status"].astype(str).eq("failed")]
    failed = _identity_pairs(failed_rows)
    success = expected_metric_identities - failed

    failures = tables["model_failures.csv"]
    _require_columns(failures, _IDENTITY_COLUMNS, "model_failures")
    if not failures.empty:
        _validate_candidate_identities(failures, "model_failures")
    failure_ids = _identity_pairs(failures)
    if len(failure_ids) != len(failures) or failure_ids != failed:
        raise ValueError("failed fold 必須與 model_failures 一對一對齊")

    predictions = tables["oof_predictions.csv"]
    _validate_candidate_identities(predictions, "oof_predictions")
    _validate_oof_coverage(
        metrics,
        predictions,
        failed,
        split_membership,
        expected_total=11088,
    )
    verified_metrics = _validate_and_recompute_fold_metrics(
        metrics,
        predictions,
        "candidate fold_metrics",
    )
    hyperparameters = tables["hyperparameters.csv"]
    _validate_candidate_identities(hyperparameters, "hyperparameters")
    hyper_ids = _identity_pairs(hyperparameters)
    if len(hyper_ids) != len(hyperparameters) or hyper_ids != success:
        raise ValueError("hyperparameters 必須逐一對應成功 fold，且只能缺 failed identity")
    if not failed and (len(predictions) != 11088 or len(hyperparameters) != 92):
        raise ValueError("Failure-free formal bundle 必須有 11,088 OOF 與 92 hyperparameters")

    _validate_feature_importance(
        metrics,
        tables["feature_importance.csv"],
        failed,
    )

    _validate_oof_values(
        tables["dummy_oof_predictions.csv"],
        data,
        "Dummy OOF",
    )
    _validate_dummy_tables(tables, split_membership)
    verified_dummy_metrics = _validate_and_recompute_fold_metrics(
        tables["dummy_fold_metrics.csv"],
        tables["dummy_oof_predictions.csv"],
        "Dummy fold_metrics",
    )
    _validate_formal_ranking(
        tables,
        failed,
        split_membership,
        verified_metrics=verified_metrics,
        verified_dummy_metrics=verified_dummy_metrics,
    )


def _formal_split_membership(
    frame: pd.DataFrame, image_keys: set[str]
) -> dict[str, set[str]]:
    """驗證 23 folds，並回傳每個 split 的 frozen test image keys。"""
    _require_columns(frame, ("validation", "fold", "image_key", "role"), "outer_splits")
    working = frame.copy()
    working["validation"] = working["validation"].astype(str)
    working["fold"] = working["fold"].astype(str)
    working["image_key"] = working["image_key"].astype(str)
    if working.duplicated(["validation", "fold", "image_key"]).any():
        raise ValueError("outer_splits 含 duplicate membership")
    identities = working[["validation", "fold"]].drop_duplicates()
    counts = identities.groupby("validation").size().to_dict()
    if counts != _FOLD_COUNTS or len(identities) != 23:
        raise ValueError("outer_splits 必須精確包含 23 folds（3/3/9/8）")
    test_membership: dict[str, set[str]] = {}
    for row in identities.itertuples(index=False):
        subset = working[
            working["validation"].eq(row.validation) & working["fold"].eq(row.fold)
        ]
        if set(subset["image_key"]) != image_keys or len(subset) != 693:
            raise ValueError("每個 formal outer fold 必須完整覆蓋 693 image keys")
        if set(subset["role"].astype(str)) != {"train", "test"}:
            raise ValueError("outer_splits role 必須含 train/test")
        split_id = f"{row.validation}:{row.fold}"
        test_keys = set(
            subset.loc[subset["role"].astype(str).eq("test"), "image_key"].astype(str)
        )
        if not test_keys:
            raise ValueError(f"outer_splits 的 {split_id} test membership 不可為空")
        test_membership[split_id] = test_keys
    return test_membership


def _validate_candidate_identities(frame: pd.DataFrame, name: str) -> None:
    """驗證四組 candidate 的 model/feature/source identity。"""
    _require_columns(frame, _IDENTITY_COLUMNS, name)
    for row in frame.loc[:, _IDENTITY_COLUMNS].itertuples(index=False):
        contract = _CONFIGURATIONS.get(str(row.configuration_id))
        actual = (str(row.model), str(row.feature_set), str(row.source_round))
        if contract != actual:
            raise ValueError(f"{name} configuration identity 不合法：{row.configuration_id}")
        if str(row.split_id) != f"{row.validation}:{row.fold}":
            raise ValueError(f"{name} split_id identity 不合法")


def _identity_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """回傳 configuration/split identity set。"""
    if frame.empty:
        return set()
    return set(
        zip(
            frame["configuration_id"].astype(str),
            frame["split_id"].astype(str),
            strict=True,
        )
    )


def _validate_oof_values(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    name: str,
) -> None:
    """驗證 OOF observed/predicted finite，且 observed 重現 data target。"""
    _require_columns(
        predictions,
        ("image_key", "observed_ido_score", "predicted_ido_score"),
        name,
    )
    if predictions.empty:
        return
    numeric = predictions.loc[
        :, ["observed_ido_score", "predicted_ido_score"]
    ].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} observed/predicted 必須 numeric finite")
    targets = data.set_index(data["image_key"].astype(str))["IDO_score"]
    expected = predictions["image_key"].astype(str).map(targets)
    if expected.isna().any() or not np.isclose(
        numeric["observed_ido_score"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise ValueError(
            f"{name} identity/image_key membership 的 observed 必須精確重現 data target"
        )


def _validate_oof_coverage(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    failed: set[tuple[str, str]],
    expected_membership: Mapping[str, set[str]],
    *,
    expected_total: int,
) -> None:
    """逐 metric 對帳 OOF membership，failed identity 必須完全缺席。"""
    if predictions.duplicated(["configuration_id", "validation", "image_key"]).any():
        raise ValueError("OOF predictions 含 duplicate configuration/family/image identity")
    prediction_ids = _identity_pairs(predictions)
    metric_ids = _identity_pairs(metrics)
    successful_ids = metric_ids - failed
    if prediction_ids != successful_ids:
        missing = sorted(successful_ids - prediction_ids)
        extra = sorted(prediction_ids - successful_ids)
        raise ValueError(
            "OOF identity set 必須精確等於 successful metric identities；"
            f"missing={missing}, extra={extra}"
        )
    for row in metrics.itertuples(index=False):
        identity = (str(row.configuration_id), str(row.split_id))
        selected = predictions[
            predictions["configuration_id"].astype(str).eq(identity[0])
            & predictions["split_id"].astype(str).eq(identity[1])
        ]
        frozen_keys = expected_membership[identity[1]]
        if (
            _strict_integer(
                row.n_test,
                "fold n_test frozen membership",
                minimum=1,
            )
            != len(frozen_keys)
        ):
            raise ValueError(
                f"fold n_test 與 frozen test membership 不一致：{identity}"
            )
        expected_keys = set() if identity in failed else frozen_keys
        actual_keys = set(selected["image_key"].astype(str))
        expected_count = len(expected_keys)
        if len(selected) != expected_count or actual_keys != expected_keys:
            raise ValueError(
                f"OOF frozen test membership 不符：{identity}; "
                f"expected={expected_count}, actual={len(selected)}; "
                f"failure-free total={expected_total:,}"
            )


def _validate_and_recompute_fold_metrics(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """由 successful fold OOF 獨立重算四項 regression metrics。

    Args:
        metrics: 含 status 與四項 published metrics 的 fold table。
        predictions: 與 successful metric identities 對齊的 OOF rows。
        name: 錯誤訊息使用的表格名稱。

    Returns:
        將 successful rows 替換為已驗證重算值的 table 副本。

    Raises:
        ValueError: OOF 不存在、數值非法或 published metric 漂移時拋出。
    """
    _require_columns(metrics, ("status", *_METRIC_COLUMNS), name)
    verified = metrics.copy()
    for index, row in verified.iterrows():
        if str(row["status"]) != "ok":
            continue
        selected = predictions[
            predictions["configuration_id"].astype(str).eq(
                str(row["configuration_id"])
            )
            & predictions["split_id"].astype(str).eq(str(row["split_id"]))
        ]
        if selected.empty:
            raise ValueError(f"{name} successful fold 缺少 OOF rows")
        observed = pd.to_numeric(
            selected["observed_ido_score"], errors="coerce"
        ).to_numpy(dtype=float)
        predicted = pd.to_numeric(
            selected["predicted_ido_score"], errors="coerce"
        ).to_numpy(dtype=float)
        recomputed = _independent_regression_metrics(observed, predicted)
        for field, expected in recomputed.items():
            try:
                actual = float(row[field])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{name} {field} 必須與 OOF 重算 metric 一致"
                ) from error
            equal = (
                math.isnan(expected) and math.isnan(actual)
            ) or (
                math.isfinite(expected)
                and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            )
            if not equal:
                raise ValueError(
                    f"{name} {field} 與 OOF 重算 metric 不一致："
                    f"published={actual}, recomputed={expected}"
                )
            verified.loc[index, field] = expected
    return verified


def _independent_regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """以 raw vectors 計算 MAE、RMSE、R² 與 Spearman。"""
    if (
        observed.ndim != 1
        or predicted.ndim != 1
        or observed.size == 0
        or observed.size != predicted.size
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("fold OOF vectors 必須等長、非空且 finite")
    residual = observed - predicted
    observed_constant = bool(np.all(observed == observed[0]))
    predicted_constant = bool(np.all(predicted == predicted[0]))
    r2 = (
        np.nan
        if observed.size < 2 or observed_constant
        else float(
            1.0
            - np.sum(residual**2)
            / np.sum((observed - np.mean(observed)) ** 2)
        )
    )
    spearman = (
        np.nan
        if observed.size < 2 or observed_constant or predicted_constant
        else float(spearmanr(observed, predicted).statistic)
    )
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": r2,
        "spearman": spearman,
    }


def _validate_dummy_tables(
    tables: Mapping[str, pd.DataFrame],
    expected_membership: Mapping[str, set[str]],
) -> None:
    """驗證 frozen Dummy identity、23 metrics 與 2,772 OOF。"""
    metrics = tables["dummy_fold_metrics.csv"]
    predictions = tables["dummy_oof_predictions.csv"]
    required = _IDENTITY_COLUMNS + ("n_test", "status")
    _require_columns(metrics, required, "dummy_fold_metrics")
    _require_columns(
        predictions,
        _IDENTITY_COLUMNS + ("image_key",),
        "dummy_oof_predictions",
    )
    split_ids = set(expected_membership)
    if len(metrics) != 23 or set(metrics["split_id"].astype(str)) != split_ids:
        raise ValueError("Formal Dummy 必須有 23 個 fold metrics")
    for frame, name in (
        (metrics, "dummy_fold_metrics"),
        (predictions, "dummy_oof_predictions"),
    ):
        identity = frame.loc[
            :, ["model", "feature_set", "source_round", "configuration_id"]
        ].astype(str)
        expected = ("dummy_median", "none", "round1", "dummy_median__none")
        if not all(tuple(row) == expected for row in identity.itertuples(index=False, name=None)):
            raise ValueError(f"Dummy identity 不合法：{name}")
        split_identity = (
            frame["validation"].astype(str) + ":" + frame["fold"].astype(str)
        )
        if not split_identity.eq(frame["split_id"].astype(str)).all():
            raise ValueError(f"Dummy split identity 不合法：{name}")
    if not metrics["status"].astype(str).eq("ok").all():
        raise ValueError("Dummy status 必須全部為 ok")
    if len(predictions) != 2772:
        raise ValueError("Formal Dummy 必須有 2,772 OOF predictions")
    if predictions.duplicated(["validation", "image_key"]).any():
        raise ValueError("Dummy OOF predictions 含 duplicate membership")
    for row in metrics.itertuples(index=False):
        selected = predictions[
            predictions["split_id"].astype(str).eq(str(row.split_id))
        ]
        expected_keys = expected_membership[str(row.split_id)]
        if (
            _strict_integer(row.n_test, "Dummy fold n_test", minimum=1)
            != len(expected_keys)
            or len(selected) != len(expected_keys)
            or set(selected["image_key"].astype(str)) != expected_keys
        ):
            raise ValueError("Dummy OOF frozen test membership 與 fold 不一致")


def _validate_feature_importance(
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    failed: set[tuple[str, str]],
) -> None:
    """驗證 importance identity；缺漏只保留為 diagnostic warning。"""
    _require_columns(
        importance,
        _IDENTITY_COLUMNS + ("feature",),
        "feature_importance",
    )
    if len(importance) > 5796:
        raise ValueError("feature_importance 總列數不可超過 5796")
    if importance.empty:
        return
    _validate_candidate_identities(importance, "feature_importance")
    successful_ids = _identity_pairs(metrics) - failed
    importance_ids = _identity_pairs(importance)
    unexpected_ids = importance_ids - successful_ids
    if unexpected_ids:
        raise ValueError(
            "feature_importance 只能屬於 successful fold identities："
            f"{sorted(unexpected_ids)}"
        )
    if importance.duplicated(["configuration_id", "split_id", "feature"]).any():
        raise ValueError("feature_importance 每個 successful fold/feature 必須唯一")
    feature_rosters = {
        "basic_median": set(PRIMARY_FOV_FEATURES),
        "paper_style_median": set(PAPER_STYLE_FOV_FEATURES),
    }
    invalid = importance[
        ~importance.apply(
            lambda row: str(row["feature"])
            in feature_rosters[str(row["feature_set"])],
            axis=1,
        )
    ]
    if not invalid.empty:
        raise ValueError(
            "feature_importance feature 不在 configuration authoritative roster："
            f"{invalid.iloc[0]['feature']}"
        )


def _feature_importance_completeness(
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict[str, int]:
    """依實際成功 metrics 計算 diagnostic completeness 摘要。"""
    required_metrics = {"configuration_id", "split_id", "feature_set", "status"}
    required_importance = {"configuration_id", "split_id", "feature"}
    if not required_metrics.issubset(metrics.columns):
        return {"actual": len(importance), "expected": 0, "missing": 0, "incomplete": 0}
    successful = metrics[metrics["status"].astype(str).eq("ok")]
    feature_counts = {"basic_median": 33, "paper_style_median": 93}
    expected = int(
        successful["feature_set"].astype(str).map(feature_counts).fillna(0).sum()
    )
    if not required_importance.issubset(importance.columns):
        return {
            "actual": len(importance),
            "expected": expected,
            "missing": max(expected - len(importance), 0),
            "incomplete": len(successful),
        }
    actual_counts = importance.groupby(
        ["configuration_id", "split_id"], dropna=False
    )["feature"].nunique()
    incomplete = 0
    for row in successful.itertuples(index=False):
        expected_count = feature_counts.get(str(row.feature_set), 0)
        actual_count = int(
            actual_counts.get((str(row.configuration_id), str(row.split_id)), 0)
        )
        if actual_count != expected_count:
            incomplete += 1
    actual = len(importance)
    return {
        "actual": actual,
        "expected": expected,
        "missing": max(expected - actual, 0),
        "incomplete": incomplete,
    }


def _importance_completeness_text(value: Any) -> str:
    """依實際 completeness 摘要產生明確 diagnostic 訊息。"""
    if not isinstance(value, Mapping):
        return "Feature importance diagnostic completeness 未提供。"
    actual = _strict_integer(value.get("actual", 0), "importance actual")
    expected = _strict_integer(value.get("expected", 0), "importance expected")
    missing = _strict_integer(
        value.get("missing", max(expected - actual, 0)),
        "importance missing",
    )
    incomplete = _strict_integer(
        value.get("incomplete", 0),
        "importance incomplete",
    )
    if missing or incomplete or actual != expected:
        return (
            "Feature importance diagnostic completeness warning："
            f"actual={actual}, expected={expected}, missing={missing}, "
            f"incomplete_fold_identities={incomplete}；缺漏不作 publication hard gate。"
        )
    return (
        "Feature importance diagnostic completeness complete："
        f"actual={actual}, expected={expected}。"
    )


def _validate_formal_ranking(
    tables: Mapping[str, pd.DataFrame],
    failed: set[tuple[str, str]],
    split_membership: Mapping[str, set[str]],
    *,
    verified_metrics: pd.DataFrame,
    verified_dummy_metrics: pd.DataFrame,
) -> None:
    """由 raw evidence 重算並逐欄驗證 ranking、gates 與 recommendation。"""
    comparison = tables["feature_set_comparison.csv"]
    eligibility = tables["eligibility.csv"]
    _require_columns(
        comparison,
        ("configuration_id", "model", "feature_set", "source_round"),
        "feature_set_comparison",
    )
    _require_columns(
        eligibility,
        (
            "configuration_id",
            "model",
            "feature_set",
            "source_round",
            "eligible",
            "recommended",
        ),
        "eligibility",
    )
    for frame, name in ((comparison, "feature_set_comparison"), (eligibility, "eligibility")):
        if len(frame) != 4 or set(frame["configuration_id"].astype(str)) != set(_CONFIGURATIONS):
            raise ValueError(f"{name} 必須精確包含四個 ranking rows")
        for row in frame.itertuples(index=False):
            if _CONFIGURATIONS[str(row.configuration_id)] != (
                str(row.model),
                str(row.feature_set),
                str(row.source_round),
            ):
                raise ValueError(f"{name} identity 不合法")
    recommended = eligibility[_boolean_series(eligibility["recommended"], "recommended")]
    if len(recommended) > 1:
        raise ValueError("Formal eligibility 最多只能有一個 recommendation")
    if not recommended.empty and not bool(
        _boolean_series(recommended["eligible"], "eligible").iloc[0]
    ):
        raise ValueError("Recommendation 必須 eligible")
    failed_configurations = {configuration for configuration, _ in failed}
    eligible = _boolean_series(eligibility["eligible"], "eligible")
    bad = eligibility.loc[eligible, "configuration_id"].astype(str).isin(failed_configurations)
    if bad.any():
        raise ValueError("含 failed fold 的 configuration 必須 ineligible")
    expected_splits = {
        validation: sorted(
            split_id
            for split_id in split_membership
            if split_id.startswith(f"{validation}:")
        )
        for validation in _VALIDATIONS
    }
    raw_metrics = verified_metrics.copy()
    raw_metrics.attrs["round2_expected_splits"] = expected_splits
    recomputed = rank_round2_configurations(
        Round2Comparison(
            fold_metrics=raw_metrics,
            predictions=tables["oof_predictions.csv"].copy(),
            dummy_metrics=verified_dummy_metrics.copy(),
            dummy_predictions=tables["dummy_oof_predictions.csv"].copy(),
            hyperparameters=tables["hyperparameters.csv"].copy(),
            feature_importance=tables["feature_importance.csv"].copy(),
            failures=tables["model_failures.csv"].copy(),
        ),
        {
            "basic_median": list(PRIMARY_FOV_FEATURES),
            "paper_style_median": list(PAPER_STYLE_FOV_FEATURES),
        },
    )
    expected_recommendation = select_round2_recommendation(recomputed)
    for published, name in (
        (comparison, "feature_set_comparison"),
        (eligibility, "eligibility"),
    ):
        _compare_recomputed_ranking(published, recomputed, name)
    actual_recommendation = select_round2_recommendation(eligibility)
    if actual_recommendation != expected_recommendation:
        raise ValueError("derived ranking deterministic recommendation 不一致")


def _compare_recomputed_ranking(
    published: pd.DataFrame,
    recomputed: pd.DataFrame,
    name: str,
) -> None:
    """以 configuration stable order 比對每個 ranking derived field。"""
    if list(published.columns) != list(recomputed.columns):
        raise ValueError(f"{name} derived ranking columns 不一致")
    actual = published.sort_values("configuration_id", kind="stable").reset_index(
        drop=True
    )
    expected = recomputed.sort_values(
        "configuration_id", kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{name} derived ranking/gate drift：{error}") from error


def _validate_smoke_tables(
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    directory: Path,
    *,
    authority_basic_images: pd.DataFrame,
) -> None:
    """驗證 smoke diagnostic split、fold、OOF 與 failure evidence 一致性。"""
    data = tables["data_snapshot.csv"]
    if data.empty or len(data) >= 693:
        raise ValueError("Smoke data_snapshot 必須是非空 subset")
    if (
        metadata.get("mode") != "smoke"
        or metadata.get("smoke") is not True
        or metadata.get("diagnostic_splits") is not True
        or metadata.get("non_formal") is not True
        or metadata.get("no_scientific_conclusion") is not True
    ):
        raise ValueError("Smoke metadata 必須明示 diagnostic/non-formal/無科學結論")
    metrics = tables["fold_metrics.csv"]
    if metrics.empty:
        raise ValueError("Smoke fold_metrics 不可為空")
    if set(metrics["model"].astype(str)) != _ROUND2_MODELS:
        raise ValueError("Smoke 必須只含 extra_trees 與 random_forest")
    if set(metrics["feature_set"].astype(str)) != {"paper_style_median"}:
        raise ValueError("Smoke 必須只執行 paper_style_median")
    if set(metrics["source_round"].astype(str)) != {"round2_paper93"}:
        raise ValueError("Smoke source_round identity 不合法")
    split_membership = _smoke_split_membership(
        tables["outer_splits.csv"],
        set(data["image_key"].astype(str)),
    )
    _validate_canonical_smoke_splits(
        tables["outer_splits.csv"],
        data,
        authority_basic_images,
    )
    split_ids = set(split_membership)
    smoke_configurations = {
        "extra_trees__paper_style_median",
        "random_forest__paper_style_median",
    }
    expected_metrics = {
        (configuration, split_id)
        for configuration in smoke_configurations
        for split_id in split_ids
    }
    _validate_candidate_identities(metrics, "Smoke fold_metrics")
    metric_ids = _identity_pairs(metrics)
    if len(metrics) != 46 or len(metric_ids) != len(metrics) or metric_ids != expected_metrics:
        raise ValueError("Smoke 必須精確包含兩組 Paper93 configurations × 23 splits")
    statuses = set(metrics["status"].astype(str))
    if not statuses.issubset({"ok", "failed"}):
        raise ValueError("Smoke fold_metrics status 只能是 ok/failed")
    failed = _identity_pairs(metrics[metrics["status"].astype(str).eq("failed")])

    failures = tables["model_failures.csv"]
    _require_columns(failures, _IDENTITY_COLUMNS, "Smoke model_failures")
    if not failures.empty:
        _validate_candidate_identities(failures, "Smoke model_failures")
    failure_ids = _identity_pairs(failures)
    if len(failure_ids) != len(failures) or failure_ids != failed:
        raise ValueError("Smoke failed folds 必須與 model_failures 一對一對齊")

    predictions = tables["oof_predictions.csv"]
    _validate_candidate_identities(predictions, "Smoke OOF")
    _validate_oof_coverage(
        metrics,
        predictions,
        failed,
        split_membership,
        expected_total=576,
    )
    _validate_and_recompute_fold_metrics(metrics, predictions, "Smoke fold_metrics")

    hyperparameters = tables["hyperparameters.csv"]
    _validate_candidate_identities(hyperparameters, "Smoke hyperparameters")
    hyper_ids = _identity_pairs(hyperparameters)
    successful = expected_metrics - failed
    if len(hyper_ids) != len(hyperparameters) or hyper_ids != successful:
        raise ValueError("Smoke hyperparameters 必須逐一對應 successful folds")
    _validate_feature_importance(metrics, tables["feature_importance.csv"], failed)

    if not tables["dummy_fold_metrics.csv"].empty or not tables[
        "dummy_oof_predictions.csv"
    ].empty:
        raise ValueError("Smoke Dummy artifacts 必須為空")
    if not tables["feature_set_comparison.csv"].empty:
        raise ValueError("Smoke feature-set comparison 必須為空")
    eligibility = tables["eligibility.csv"]
    expected_eligibility = {
        configuration: _CONFIGURATIONS[configuration]
        for configuration in smoke_configurations
    }
    if len(eligibility) != 2 or set(
        eligibility["configuration_id"].astype(str)
    ) != smoke_configurations:
        raise ValueError("Smoke eligibility 必須精確包含兩組 Paper93 identities")
    for row in eligibility.itertuples(index=False):
        expected = expected_eligibility[str(row.configuration_id)]
        if (str(row.model), str(row.feature_set), str(row.source_round)) != expected:
            raise ValueError("Smoke eligibility configuration identity 不合法")
    if _boolean_series(eligibility["eligible"], "eligible").any():
        raise ValueError("Smoke eligibility 不可標示 eligible")
    if _boolean_series(eligibility["recommended"], "recommended").any():
        raise ValueError("smoke 不可產生 recommendation")
    if "status" not in eligibility or set(eligibility["status"].astype(str)) != {"smoke"}:
        raise ValueError("Smoke eligibility 必須明示 smoke status")
    record = (directory / "EXPERIMENT_RECORD.md").read_text(encoding="utf-8")
    if "僅驗證流程，不是正式實驗結果" not in record or "不形成 33 vs 93 科學結論" not in record:
        raise ValueError("Smoke record 必須明示無科學結論")


def _smoke_split_membership(
    frame: pd.DataFrame,
    image_keys: set[str],
) -> dict[str, set[str]]:
    """驗證 23 個 diagnostic splits 與每-family 一次 test coverage。"""
    _require_columns(frame, ("validation", "fold", "image_key", "role"), "Smoke splits")
    working = frame.copy()
    for column in ("validation", "fold", "image_key", "role"):
        working[column] = working[column].astype(str)
    if working.duplicated(["validation", "fold", "image_key"]).any():
        raise ValueError("Smoke split membership 含 duplicate 或 train/test overlap")
    identities = working.loc[:, ["validation", "fold"]].drop_duplicates()
    counts = identities.groupby("validation").size().to_dict()
    if counts != _FOLD_COUNTS or len(identities) != 23:
        raise ValueError("Smoke diagnostic splits 必須精確包含四 families 的 3/3/9/8 folds")
    membership: dict[str, set[str]] = {}
    for row in identities.itertuples(index=False):
        subset = working[
            working["validation"].eq(row.validation)
            & working["fold"].eq(row.fold)
        ]
        if len(subset) != len(image_keys) or set(subset["image_key"]) != image_keys:
            raise ValueError("Smoke 每個 split 必須完整 partition data_snapshot")
        if set(subset["role"]) != {"train", "test"}:
            raise ValueError("Smoke split 必須同時含非空 train/test roles")
        split_id = f"{row.validation}:{row.fold}"
        membership[split_id] = set(
            subset.loc[subset["role"].eq("test"), "image_key"]
        )
    for validation in _VALIDATIONS:
        test_rows = working[
            working["validation"].eq(validation)
            & working["role"].eq("test")
        ]
        counts_by_image = test_rows["image_key"].value_counts()
        if set(counts_by_image.index) != image_keys or not counts_by_image.eq(1).all():
            raise ValueError("Smoke 每個 family 的每張 image 必須恰好 test 一次")
    return membership


def _validate_canonical_smoke_splits(
    published: pd.DataFrame,
    data: pd.DataFrame,
    authority: pd.DataFrame,
) -> None:
    """由權威 metadata 重建 smoke grouped splits 並逐 membership 比對。"""
    image_keys = set(data["image_key"].astype(str))
    authority_keys = authority["image_key"].astype(str)
    subset = authority.loc[authority_keys.isin(image_keys)].copy()
    if len(subset) != len(image_keys) or set(subset["image_key"].astype(str)) != image_keys:
        raise ValueError("Smoke authority subset image identity 不完整")
    canonical = outer_split_manifest(subset, make_outer_splits(subset))
    columns = ["validation", "fold", "image_key", "role"]

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, columns, "Smoke canonical diagnostic splits")
        result = frame.loc[:, columns].copy()
        for column in columns:
            result[column] = result[column].astype(str)
        return result.sort_values(columns, kind="stable").reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            normalized(published),
            normalized(canonical),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "Smoke canonical diagnostic splits 偏離 authority metadata"
        ) from error


def _boolean_series(series: pd.Series, name: str) -> pd.Series:
    """嚴格解析 CSV boolean 欄位。"""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{name} 必須是 boolean")
    return normalized.eq("true")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """要求資料表包含指定欄位。"""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少必要欄位：{missing}")


def _markdown_context(value: Any) -> str:
    """將 record context 轉為簡潔且穩定的 Markdown 文字。"""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return "（無資料）"
        return "```text\n" + value.to_csv(index=False, lineterminator="\n").rstrip() + "\n```"
    if isinstance(value, Mapping):
        return "、".join(f"{key}={_display(item)}" for key, item in value.items()) or "（無資料）"
    return _display(value if value is not None else "未提供")


def _display(value: Any) -> str:
    """格式化 record 中的 scalar。"""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6g}"
    return str(value)


def _display_count(value: Any) -> str:
    """嚴格解析 observation count 後以千分位顯示。"""
    numeric = _strict_integer(value, "display count")
    return f"{numeric:,}"


__all__ = [
    "ROUND2_JSON_NAMES",
    "ROUND2_TABLE_NAMES",
    "Round2Generation",
    "begin_round2_generation",
    "build_round2_experiment_record",
    "publish_round2_generation",
    "quarantine_round2_generation",
    "validate_round2_bundle",
    "write_round2_bundle",
]
