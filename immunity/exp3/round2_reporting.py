"""原子寫入、發布並驗證 Exp3 Round 2 Paper93 結果。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES, PRIMARY_FOV_FEATURES


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


@dataclass(frozen=True)
class Round2Generation:
    """保存單次 Round 2 generation 的安全發布邊界。

    Attributes:
        output_dir: resolver 核准的正式根目錄或 smoke 子目錄。
        staging_dir: 位於 output boundary 內、尚未發布的唯一 staging 目錄。
    """

    output_dir: Path
    staging_dir: Path


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
    staging = generations / f"staging-{_generation_suffix()}"
    _assert_safe_child(target, staging)
    staging.mkdir()
    _assert_safe_directory(target, staging)
    return Round2Generation(output_dir=target, staging_dir=staging)


def publish_round2_generation(generation: Round2Generation) -> None:
    """封存 target 內舊版 Round 2 檔案，並發布已驗證 staging。

    Args:
        generation: ``begin_round2_generation`` 建立的 generation。

    Raises:
        ValueError: Boundary、bundle 或現有 destination 不符合固定合約。
        OSError: 檔案搬移失敗且已完成安全 rollback 時拋出。
    """
    output, staging = _validate_generation(generation)
    validate_round2_bundle(staging, smoke=output.name == "smoke")
    existing_files = [path for path in output.iterdir() if path.is_file() or path.is_symlink()]
    unexpected = sorted(path.name for path in existing_files if path.name not in _BUNDLE_FILE_NAMES)
    if unexpected:
        raise ValueError(f"Round 2 output 含非預期檔案：{unexpected}")
    for path in existing_files:
        _assert_regular_file(output, path)

    archive: Path | None = None
    archived: list[str] = []
    published: list[str] = []
    try:
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
        staging.rmdir()
    except Exception:
        for name in reversed(published):
            destination = output / name
            if destination.exists() and not (staging / name).exists():
                os.replace(destination, staging / name)
        if archive is not None:
            for name in reversed(archived):
                source = archive / name
                if source.exists() and not (output / name).exists():
                    os.replace(source, output / name)
            if archive.exists() and not any(archive.iterdir()):
                archive.rmdir()
        raise


def quarantine_round2_generation(generation: Round2Generation) -> Path:
    """將失敗 staging 搬至相同 target boundary 的具名 quarantine。

    Args:
        generation: 尚未發布且含 ``run.log`` 的 generation。

    Returns:
        保留失敗 artifacts 與 QC log 的 ``failed-*`` 目錄。

    Raises:
        ValueError: Generation 越界、已消失，或 ``run.log`` 不安全。
    """
    output, staging = _validate_generation(generation)
    _assert_regular_file(staging, staging / "run.log")
    failed = output / "_generations" / f"failed-{_generation_suffix()}"
    _assert_safe_child(output, failed)
    if failed.exists() or failed.is_symlink():
        raise FileExistsError(f"Quarantine destination 已存在：{failed}")
    os.replace(staging, failed)
    _assert_safe_directory(output, failed)
    return failed


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
    _read_json_mapping(bundle / "baseline_provenance.json")
    metadata = _read_json_mapping(bundle / "run_metadata.json")
    _validate_predictor_registry(feature_sets)
    _validate_common_tables(tables)
    if smoke:
        _validate_smoke_tables(tables, metadata, bundle)
    else:
        _validate_formal_tables(tables, metadata)


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
            "phase-only paper-style approximation。Target、實驗條件、路徑與批次資訊均不進入 predictor matrix。"
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
        "## 33 vs 93 比較",
        comparison,
        "## Eligibility gates 與 tie decision",
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


def _validate_generation(generation: Round2Generation) -> tuple[Path, Path]:
    """驗證 generation identity 與 staging boundary。"""
    if not isinstance(generation, Round2Generation):
        raise TypeError("generation 必須是 Round2Generation")
    output = _resolve_allowed_output(generation.output_dir)
    staging = Path(generation.staging_dir)
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


def _validate_common_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    """驗證 formal/smoke 共用的 93-feature schema 與 QC tables。"""
    data = tables["data_snapshot.csv"]
    _require_columns(data, ("image_key", "IDO_score", *PAPER_STYLE_FOV_FEATURES), "data_snapshot")
    if data.empty or data["image_key"].duplicated().any():
        raise ValueError("data_snapshot image_key 必須非空且唯一")
    feature_qc = tables["feature_qc.csv"]
    _require_columns(feature_qc, ("feature", "status"), "feature_qc")
    if (
        set(feature_qc["feature"].astype(str)) != set(PAPER_STYLE_FOV_FEATURES)
        or len(feature_qc) != 93
    ):
        raise ValueError("feature_qc 必須精確包含 93 predictor identity")
    _require_columns(
        tables["fold_metrics.csv"],
        _IDENTITY_COLUMNS + ("n_test", "status"),
        "fold_metrics",
    )
    _require_columns(
        tables["oof_predictions.csv"],
        _IDENTITY_COLUMNS + ("image_key",),
        "oof_predictions",
    )
    _require_columns(tables["hyperparameters.csv"], _IDENTITY_COLUMNS, "hyperparameters")
    _require_columns(
        tables["eligibility.csv"],
        ("configuration_id", "eligible", "recommended"),
        "eligibility",
    )


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
    tables: Mapping[str, pd.DataFrame], metadata: Mapping[str, Any]
) -> None:
    """驗證正式 693-FOV publication gates。"""
    data = tables["data_snapshot.csv"]
    if len(data) != 693:
        raise ValueError("Formal data_snapshot 必須精確為 693 rows")
    if metadata.get("mode", "formal") != "formal" or metadata.get("seed") != 20260804:
        raise ValueError("Formal run_metadata mode/seed 不合法")
    split_membership = _formal_split_membership(
        tables["outer_splits.csv"], set(data["image_key"].astype(str))
    )
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

    _validate_dummy_tables(tables, split_membership)
    _validate_formal_ranking(tables, failed)


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
        if int(row.n_test) != len(frozen_keys):
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
            int(row.n_test) != len(expected_keys)
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
    actual = int(value.get("actual", 0))
    expected = int(value.get("expected", 0))
    missing = int(value.get("missing", max(expected - actual, 0)))
    incomplete = int(value.get("incomplete", 0))
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
    tables: Mapping[str, pd.DataFrame], failed: set[tuple[str, str]]
) -> None:
    """驗證四組比較、eligibility 與 failed configuration 對帳。"""
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


def _validate_smoke_tables(
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    directory: Path,
) -> None:
    """驗證 smoke subset schema、兩模型、status 與無 recommendation。"""
    data = tables["data_snapshot.csv"]
    if data.empty or len(data) >= 693:
        raise ValueError("Smoke data_snapshot 必須是非空 subset")
    mode_ok = metadata.get("mode") == "smoke" or metadata.get("smoke") is True
    if not mode_ok:
        raise ValueError("Smoke run_metadata 必須明示 smoke status")
    metrics = tables["fold_metrics.csv"]
    if metrics.empty:
        raise ValueError("Smoke fold_metrics 不可為空")
    if set(metrics["model"].astype(str)) != _ROUND2_MODELS:
        raise ValueError("Smoke 必須只含 extra_trees 與 random_forest")
    if set(metrics["feature_set"].astype(str)) != {"paper_style_median"}:
        raise ValueError("Smoke 必須只執行 paper_style_median")
    if set(metrics["source_round"].astype(str)) != {"round2_paper93"}:
        raise ValueError("Smoke source_round identity 不合法")
    eligibility = tables["eligibility.csv"]
    if _boolean_series(eligibility["recommended"], "recommended").any():
        raise ValueError("smoke 不可產生 recommendation")
    if "status" not in eligibility or set(eligibility["status"].astype(str)) != {"smoke"}:
        raise ValueError("Smoke eligibility 必須明示 smoke status")
    record = (directory / "EXPERIMENT_RECORD.md").read_text(encoding="utf-8")
    if "僅驗證流程，不是正式實驗結果" not in record or "不形成 33 vs 93 科學結論" not in record:
        raise ValueError("Smoke record 必須明示無科學結論")


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
    """以千分位顯示整數 observation count，其他值沿用 scalar formatter。"""
    if isinstance(value, bool):
        return str(value)
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return _display(value)
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
