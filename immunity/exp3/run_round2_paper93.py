"""載入並驗證 Exp3 Round 2 paper-style 實驗設定。"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import stat
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import yaml

if __name__ == "__main__":
    #避免 ``python -m`` 下 reporting 反向匯入 canonical module 造成第二份初始化。
    sys.modules.setdefault("immunity.exp3.run_round2_paper93", sys.modules[__name__])

from immunity.exp3.benchmark import outer_split_manifest
from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES, PRIMARY_FOV_FEATURES
from immunity.exp3.round2_benchmark import (
    build_round2_comparison,
    rank_round2_configurations,
    run_paper93_benchmark,
    select_round2_recommendation,
)
from immunity.exp3.round2_evidence import (
    expected_split_ids,
    load_round1_evidence,
    make_smoke_outer_splits,
    require_formal_round1_evidence,
    restore_frozen_outer_splits,
    validate_frozen_masks,
)
from immunity.exp3.round2_features import (
    extract_locked_paper93,
    require_paper93_preflight,
)
from immunity.exp3.round2_reporting import (
    ROUND2_JSON_NAMES,
    ROUND2_TABLE_NAMES,
    begin_round2_generation,
    publish_round2_generation,
    quarantine_round2_generation,
    validate_round2_bundle,
    write_round2_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUND2_OUTPUT_ROOT = (
    PROJECT_ROOT / "immunity" / "outputs" / "exp3" / "round2_paper93"
).resolve()
ROUND2_MODELS = ("extra_trees", "random_forest")
ROUND2_FEATURE_SET = "paper_style_median"
_ROUND2_FEATURES = {
    "feature_set": ROUND2_FEATURE_SET,
    "predictor_count": 93,
    "extra_count": 60,
    "min_finite_cells_per_feature": 3,
    "numeric_atol": 1e-12,
}
_EXPECTED_ROUND1_COUNTS = {
    "raw_pc": 720,
    "raw_ido": 719,
    "paired": 719,
    "exclusions": 26,
    "analyzed_images": 693,
    "valid_cells": 23976,
    "outer_folds": 23,
    "seed": 20260804,
}
_FROZEN_ROUND1_EXPECTED = {
    **_EXPECTED_ROUND1_COUNTS,
    "manifest_hash": "b4508aaf830f4c456759d1f02458606aa28ed385856575c6fb99a5ac2e163771",
    "config_hash": "b7590ed3438cdb5062b2588e525e603dd0b7610d397322c761facdaea5b54bc0",
}
_FROZEN_ROUND1_ROSTER_SHA256 = (
    "a8333f12e1591d9e4c4174f5c6fe13dd31550124b19aea3522f4d0f812e0e426"
)
_FROZEN_ROUND1_ARTIFACT_SHA256 = {
    "pairing_qc.csv": "0fa953eceb35678aab0209214eb1ee267e844a0b81e649b0c0acca487246e0eb",
    "data_manifest.csv": "b4508aaf830f4c456759d1f02458606aa28ed385856575c6fb99a5ac2e163771",
    "segmentation_qc.csv": "2a1e8969fc8d0a81d3dd737ef99df57f58ce3e1faf0206e833463f2dc9d28c1e",
    "outer_splits.csv": "c7e9e26a9ad0c6f73d8f14c5fa4c73747f6e58865aac963bf8e16e527fa05a3e",
    "feature_cache/image_level_basic.csv": "dad479258d08846d05a43f551b0001a52d96fa5e2e8ea6a98645723931aa0d14",
    "feature_cache/cell_level_basic.csv": "7d999a728d848ccb61a3910cb5230c9efde6d7e1fa14f22de792990ee4ad7850",
    "fold_metrics.csv": "8ce0df3866822a54c98cb1cd95dbec3c3a8ffa130af54890692b240250091884",
    "oof_predictions.csv": "637242dc0f45c4dc436643214c21ac451fc560d7ae36b25a6db04aafe5d6a2c8",
    "model_ranking.csv": "d279ae973d5ab7508e97460bb1b6f94dd95345ce27d4cca0ad2f587516cb72e4",
    "hyperparameters.csv": "5da9c936b573126d21417752d0fd140306562ed387b299364a5cd356268bfe00",
    "feature_importance.csv": "17cf4bafdb1d7b87654d7efd58ef79584f1324c40abac3cb7bdfc617a510b9f6",
    "model_failures.csv": "b1ce311d21be365f1e5a010d9eb16aaea997e5d843e5b630aef77c91b52d98ea",
    "feature_sets.json": "49e65affebbab6285170e45048eee3991bfc70e1a53454d7e6bbc94a092e28cd",
    "run_metadata.json": "d0cdfabea5f9243eeadf7eb2278019f9497a75c3e82ba129b4077e1f62566b9f",
}


def load_round2_config(path: str | Path) -> dict[str, Any]:
    """載入並驗證 Round 2 的不可變實驗設定。

    Args:
        path: Round 2 YAML 設定檔的絕對或專案相對路徑。

    Returns:
        經過合約驗證、附加設定檔與正式輸出目錄資訊的設定。

    Raises:
        ValueError: 當 YAML 不是 mapping，或偏離 Round 2 的凍結合約時拋出。
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Round 2 config 必須是 mapping")

    _validate_round2_config(config)
    config["round1"]["roster_sha256"] = config["round1"]["roster_sha256"].upper()
    config["_config_path"] = str(config_path.resolve())
    config["_output_dir"] = str(
        resolve_round2_output_dir(config["output"]["dir"], smoke=False)
    )
    return config


def resolve_round2_output_dir(path: str | Path, *, smoke: bool) -> Path:
    """解析 Round 2 唯一允許的正式輸出目錄或 smoke 子目錄。

    Args:
        path: 必須精確指向 Round 2 正式根目錄的路徑。
        smoke: 為 ``True`` 時回傳正式根目錄下的 ``smoke`` 子目錄。

    Returns:
        已解析且不含 link/junction 逃逸的正式輸出位置。

    Raises:
        ValueError: 當路徑不是正式根目錄，或既有元件為 symlink/junction 時拋出。
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    expected = Path(ROUND2_OUTPUT_ROOT).resolve(strict=False)
    if resolved != expected:
        raise ValueError(
            "Round 2 output 必須是 immunity/outputs/exp3/round2_paper93"
        )

    target = candidate / "smoke" if smoke else candidate
    if any(_is_reparse_point(part) for part in _existing_path_components(target)):
        raise ValueError("Round 2 output path 不可包含 symlink 或 junction")
    expected_target = expected / "smoke" if smoke else expected
    if target.resolve(strict=False) != expected_target:
        raise ValueError("Round 2 output target 不可逃逸正式目錄")
    return expected_target


def build_parser() -> argparse.ArgumentParser:
    """建立 Round 2 CLI 的參數解析器。

    Returns:
        包含設定檔與 smoke FOV 覆寫參數的解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke-fovs-per-condition",
        type=_positive_int_argument,
        default=None,
    )
    return parser


def run_round2(
    config: Mapping[str, Any],
    smoke_fovs_per_condition: int | None = None,
) -> Path:
    """執行並原子發布獨立的 Exp3 Round 2 Paper93 generation。

    Args:
        config: 已驗證且含 Round 1 pins、feature、benchmark 與 output 的設定。
        smoke_fovs_per_condition: 每個 ``group_id × condition_index`` 保留的
            smoke FOV 數；省略時執行正式 frozen membership benchmark。

    Returns:
        已發布且存在的 ``EXPERIMENT_RECORD.md`` 路徑。

    Raises:
        TypeError: ``config`` 不是 mapping 時拋出。
        ValueError: 設定、來源證據、mask、特徵或 bundle validation 失敗時拋出。
        RuntimeError: 發布後 record 不存在時拋出。
    """
    if not isinstance(config, Mapping):
        raise TypeError("Round 2 config 必須是 mapping")
    smoke = smoke_fovs_per_condition is not None
    smoke_count = _smoke_count(smoke_fovs_per_condition) if smoke else None
    output_value = _required_mapping(config, "output").get("dir")
    if not isinstance(output_value, (str, Path)):
        raise ValueError("Round 2 output.dir 必須是路徑")
    output_dir = resolve_round2_output_dir(output_value, smoke=smoke)
    generation = begin_round2_generation(output_dir)
    log_path = generation.staging_dir / "run.log"
    started = time.perf_counter()
    mask_qc: pd.DataFrame | None = None
    bundle: Any = None

    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            stdout_tee = _Tee(sys.stdout, log_handle)
            stderr_tee = _Tee(sys.stderr, log_handle)
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(
                stderr_tee
            ):
                evidence = load_round1_evidence(config)
                #Smoke 仍必須先驗證完整 14-artifact source pins 與正式 roster。
                require_formal_round1_evidence(evidence)
                image_keys = (
                    _select_smoke_image_keys(evidence.basic_images, smoke_count)
                    if smoke
                    else evidence.basic_images["image_key"].astype(str).tolist()
                )
                mask_qc = validate_frozen_masks(evidence, image_keys)
                mask_qc.to_csv(
                    generation.staging_dir / "mask_provenance_qc.csv",
                    index=False,
                    lineterminator="\n",
                )
                _require_mask_qc(mask_qc)

                feature_config = _required_mapping(config, "features")
                bundle = extract_locked_paper93(
                    evidence,
                    mask_qc,
                    image_keys,
                    min_finite_cells=int(
                        feature_config["min_finite_cells_per_feature"]
                    ),
                    atol=float(feature_config["numeric_atol"]),
                )
                _write_feature_qc(generation.staging_dir, bundle)
                require_paper93_preflight(bundle, formal=not smoke)

                splits = (
                    make_smoke_outer_splits(bundle.images, config)
                    if smoke
                    else restore_frozen_outer_splits(
                        bundle.images, evidence.split_manifest
                    )
                )
                effective_config = _effective_config(
                    config,
                    smoke=smoke,
                    smoke_fovs_per_condition=smoke_count,
                )
                paper93_result = run_paper93_benchmark(
                    bundle.images,
                    splits,
                    _required_mapping(effective_config, "benchmark"),
                )
                if smoke:
                    comparison = None
                    ranking = _smoke_eligibility()
                    recommendation = None
                else:
                    comparison = build_round2_comparison(
                        evidence,
                        paper93_result,
                        expected_split_ids(splits),
                    )
                    ranking = rank_round2_configurations(
                        comparison,
                        {
                            "basic_median": list(PRIMARY_FOV_FEATURES),
                            "paper_style_median": list(PAPER_STYLE_FOV_FEATURES),
                        },
                    )
                    recommendation = select_round2_recommendation(ranking)

                tables = _round2_tables(
                    evidence=evidence,
                    bundle=bundle,
                    mask_qc=mask_qc,
                    splits=splits,
                    paper93_result=paper93_result,
                    comparison=comparison,
                    ranking=ranking,
                    smoke=smoke,
                )
                payloads = _round2_json_payloads(
                    config=config,
                    effective_config=effective_config,
                    evidence=evidence,
                    bundle=bundle,
                    smoke=smoke,
                )
                context = _round2_record_context(
                    config=config,
                    bundle=bundle,
                    ranking=ranking,
                    recommendation=recommendation,
                    smoke=smoke,
                    runtime_seconds=time.perf_counter() - started,
                )
                _remove_preflight_qc(generation.staging_dir)
                write_round2_bundle(
                    generation.staging_dir,
                    tables,
                    payloads,
                    context,
                )
                validate_round2_bundle(generation.staging_dir, smoke=smoke)

        #Windows 上已開啟的 handle 不可安全搬移；發布只能發生在 with 之外。
        publish_round2_generation(generation)
        record = output_dir / "EXPERIMENT_RECORD.md"
        if not record.is_file():
            raise RuntimeError("EXPERIMENT_RECORD.md 未成功發布")
        return record
    except BaseException:
        error_info = sys.exc_info()
        try:
            _restore_failure_qc(generation.staging_dir, mask_qc, bundle)
        except BaseException as restore_error:
            _append_secondary_failure_evidence(
                log_path,
                "restore_failure_qc",
                restore_error,
            )
        try:
            _append_failure_evidence(log_path, error_info, generation.staging_dir)
        except BaseException as append_error:
            _append_secondary_failure_evidence(
                log_path,
                "append_failure_evidence",
                append_error,
            )
        try:
            if generation.staging_dir.exists():
                quarantine_round2_generation(generation)
        except BaseException as quarantine_error:
            _append_secondary_failure_evidence(
                log_path,
                "quarantine_round2_generation",
                quarantine_error,
            )
        raise


def main() -> int:
    """執行 Round 2 CLI 並將 pipeline 失敗轉為非零 exit code。

    Returns:
        Bundle 完成 validation、發布且 record 存在時回傳 0；否則回傳非零。
    """
    arguments = build_parser().parse_args()
    try:
        config = load_round2_config(arguments.config)
        record = run_round2(
            config,
            smoke_fovs_per_condition=arguments.smoke_fovs_per_condition,
        )
        if record.name != "EXPERIMENT_RECORD.md" or not record.is_file():
            raise RuntimeError("EXPERIMENT_RECORD.md 不存在，CLI 不可回傳成功")
        print(f"Exp3 Round 2 record: {record}")
        return 0
    except Exception:  # noqa: BLE001 - CLI 必須把完整 traceback 輸出到 stderr
        traceback.print_exc()
        return 1


class _Tee:
    """將 console stream 同步寫入 run-local log。"""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self._stream = stream
        self._log = log

    def write(self, value: str) -> int:
        """同步寫入 console 與 log，並立即刷新 log。"""
        self._stream.write(value)
        self._log.write(value)
        self._log.flush()
        return len(value)

    def flush(self) -> None:
        """同步刷新 console 與 log。"""
        self._stream.flush()
        self._log.flush()


def _positive_int_argument(value: str) -> int:
    """將 CLI 值解析為正整數，供 argparse 在 pipeline 前拒絕無效輸入。"""
    try:
        converted = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必須是正整數") from error
    if converted < 1:
        raise argparse.ArgumentTypeError("必須是正整數")
    return converted


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """取得必要 mapping 設定。"""
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 2 {key} 必須是 mapping")
    return value


def _smoke_count(value: int | None) -> int:
    """驗證 smoke 每組 FOV 數是正整數。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("smoke_fovs_per_condition 必須是正整數")
    return value


def _select_smoke_image_keys(images: pd.DataFrame, count: int | None) -> list[str]:
    """依 group、condition 與 FOV stable sort 選取 deterministic smoke subset。"""
    if count is None:
        raise ValueError("smoke subset 缺少每組 FOV 數")
    required = {"image_key", "group_id", "condition_index", "fov"}
    missing = sorted(required - set(images.columns))
    if missing:
        raise ValueError(f"smoke images 缺少必要欄位：{missing}")
    source_keys = images["image_key"].astype(str).tolist()
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("smoke source image_key 必須唯一")
    ordered = images.assign(_image_key=source_keys).sort_values(
        ["group_id", "condition_index", "fov", "_image_key"], kind="stable"
    )
    selected = ordered.groupby(
        ["group_id", "condition_index"], sort=False, dropna=False
    ).head(count)
    if selected.empty:
        raise ValueError("smoke image subset 不可為空")
    selected_keys = selected["_image_key"].tolist()
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("smoke selected image_key 必須唯一")
    if len(selected_keys) >= len(source_keys):
        raise ValueError("smoke 必須是完整 frozen roster 的 strict subset")
    return selected_keys


def _require_mask_qc(mask_qc: pd.DataFrame) -> None:
    """在 feature extraction 前阻擋所有 mask provenance failures。"""
    if "status" not in mask_qc:
        raise ValueError("mask provenance QC 缺少 status")
    failed = mask_qc[~mask_qc["status"].astype(str).eq("passed")]
    if not failed.empty:
        columns = [
            name
            for name in ("image_key", "pc_path", "mask_path", "reason")
            if name in failed
        ]
        evidence = failed.loc[:, columns].to_dict(orient="records")
        raise ValueError(f"mask provenance failure：{evidence}")


def _write_feature_qc(staging: Path, bundle: Any) -> None:
    """在 preflight 前保留可用的 feature extraction QC。"""
    for name, frame in (
        ("feature_valid_counts.csv", bundle.valid_counts),
        ("extraction_qc.csv", bundle.extraction_qc),
        ("feature_qc.csv", bundle.feature_qc),
    ):
        frame.to_csv(staging / name, index=False, lineterminator="\n")


def _remove_preflight_qc(staging: Path) -> None:
    """成功 preflight 後移除暫存 QC，交由 atomic bundle writer 重寫。"""
    for name in (
        "mask_provenance_qc.csv",
        "feature_valid_counts.csv",
        "extraction_qc.csv",
        "feature_qc.csv",
    ):
        (staging / name).unlink()


def _restore_failure_qc(
    staging: Path,
    mask_qc: pd.DataFrame | None,
    bundle: Any,
) -> None:
    """在晚期 failure quarantine 前補回尚未由 writer 留下的 QC tables。"""
    frames = {
        "mask_provenance_qc.csv": mask_qc,
        "feature_valid_counts.csv": getattr(bundle, "valid_counts", None),
        "extraction_qc.csv": getattr(bundle, "extraction_qc", None),
        "feature_qc.csv": getattr(bundle, "feature_qc", None),
    }
    failures: list[BaseException] = []
    for name, frame in frames.items():
        path = staging / name
        if path.exists() or not isinstance(frame, pd.DataFrame):
            continue
        try:
            frame.to_csv(path, index=False, lineterminator="\n")
        except BaseException as error:  # noqa: BLE001 - 其餘 QC 仍須 best-effort 寫入
            failures.append(error)
    if failures:
        raise failures[0]


def _effective_config(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    smoke_fovs_per_condition: int | None,
) -> dict[str, Any]:
    """建立保留原始設定、只縮小 smoke 計算量的 effective config。"""
    effective = copy.deepcopy(dict(config))
    if smoke:
        benchmark = dict(_required_mapping(effective, "benchmark"))
        smoke_config = _required_mapping(config, "smoke")
        for key in (
            "max_hyperparameter_candidates",
            "permutation_repeats",
            "tree_estimators",
        ):
            benchmark[key] = smoke_config[key]
        effective["benchmark"] = benchmark
        effective["smoke"] = {
            **dict(smoke_config),
            "fovs_per_condition": smoke_fovs_per_condition,
        }
    return effective


def _smoke_eligibility() -> pd.DataFrame:
    """建立不含排名或 recommendation 的兩模型 smoke status 表。"""
    return pd.DataFrame(
        {
            "configuration_id": [
                "extra_trees__paper_style_median",
                "random_forest__paper_style_median",
            ],
            "model": ["extra_trees", "random_forest"],
            "feature_set": [ROUND2_FEATURE_SET, ROUND2_FEATURE_SET],
            "source_round": ["round2_paper93", "round2_paper93"],
            "eligible": [False, False],
            "recommended": [False, False],
            "status": ["smoke", "smoke"],
        }
    )


def _round2_tables(
    *,
    evidence: Any,
    bundle: Any,
    mask_qc: pd.DataFrame,
    splits: Any,
    paper93_result: Any,
    comparison: Any,
    ranking: pd.DataFrame,
    smoke: bool,
) -> dict[str, pd.DataFrame]:
    """依 formal/smoke contract 建立固定 Round 2 CSV tables。"""
    if smoke:
        sources = {
            "fold_metrics.csv": paper93_result.fold_metrics,
            "oof_predictions.csv": paper93_result.predictions,
            "dummy_fold_metrics.csv": evidence.selected_metrics.iloc[0:0].copy(),
            "dummy_oof_predictions.csv": evidence.selected_predictions.iloc[
                0:0
            ].copy(),
            "hyperparameters.csv": paper93_result.hyperparameters,
            "feature_importance.csv": paper93_result.feature_importance,
            "model_failures.csv": paper93_result.failures,
            "feature_set_comparison.csv": pd.DataFrame(
                columns=["configuration_id", "model", "feature_set", "source_round"]
            ),
            "eligibility.csv": ranking,
        }
    else:
        sources = {
            "fold_metrics.csv": comparison.fold_metrics,
            "oof_predictions.csv": comparison.predictions,
            "dummy_fold_metrics.csv": comparison.dummy_metrics,
            "dummy_oof_predictions.csv": comparison.dummy_predictions,
            "hyperparameters.csv": comparison.hyperparameters,
            "feature_importance.csv": comparison.feature_importance,
            "model_failures.csv": comparison.failures,
            "feature_set_comparison.csv": ranking,
            "eligibility.csv": ranking,
        }
    tables = {
        "data_snapshot.csv": bundle.images,
        "outer_splits.csv": outer_split_manifest(bundle.images, splits),
        "mask_provenance_qc.csv": mask_qc,
        "feature_valid_counts.csv": bundle.valid_counts,
        "extraction_qc.csv": bundle.extraction_qc,
        "feature_qc.csv": bundle.feature_qc,
        **sources,
    }
    if set(tables) != set(ROUND2_TABLE_NAMES):
        raise RuntimeError("Round 2 table assembly 未符合固定 artifact contract")
    return tables


def _round2_json_payloads(
    *,
    config: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    evidence: Any,
    bundle: Any,
    smoke: bool,
) -> dict[str, Mapping[str, Any]]:
    """建立 predictor、來源 provenance 與執行模式 JSON artifacts。"""
    expected = _required_mapping(_required_mapping(config, "round1"), "expected")
    metadata = {
        "mode": "smoke" if smoke else "formal",
        "smoke": smoke,
        "seed": int(_required_mapping(effective_config, "benchmark")["seed"]),
        "analyzed_images": len(bundle.images),
        "valid_cells": len(bundle.paper_cells),
        "exclusions": int(expected.get("exclusions", 26)),
        "diagnostic_splits": smoke,
        "non_formal": smoke,
        "no_scientific_conclusion": smoke,
        "original_config": _json_ready(config),
        "effective_config": _json_ready(effective_config),
    }
    payloads = {
        "feature_sets.json": {
            "basic_median": list(PRIMARY_FOV_FEATURES),
            "paper_style_median": list(PAPER_STYLE_FOV_FEATURES),
        },
        "baseline_provenance.json": {
            "round1_root": str(evidence.root),
            "read_only": True,
            "artifact_sha256": dict(evidence.artifact_hashes),
            "roster_sha256": _required_mapping(config, "round1").get(
                "roster_sha256"
            ),
        },
        "run_metadata.json": metadata,
    }
    if set(payloads) != set(ROUND2_JSON_NAMES):
        raise RuntimeError("Round 2 JSON assembly 未符合固定 artifact contract")
    return payloads


def _round2_record_context(
    *,
    config: Mapping[str, Any],
    bundle: Any,
    ranking: pd.DataFrame,
    recommendation: str | None,
    smoke: bool,
    runtime_seconds: float,
) -> dict[str, Any]:
    """建立正式結論或 smoke 無科學結論的繁體中文 record context。"""
    expected = _required_mapping(_required_mapping(config, "round1"), "expected")
    failed = bundle.extraction_qc[
        ~bundle.extraction_qc["status"].astype(str).eq("passed")
    ]
    return {
        "smoke": smoke,
        "recommendation": recommendation,
        "paper93_better_than_basic": (
            recommendation.endswith("__paper_style_median")
            if recommendation is not None
            else None
        ),
        "analyzed_images": len(bundle.images),
        "valid_cells": len(bundle.paper_cells),
        "exclusions": int(expected.get("exclusions", 26)),
        "seed": int(_required_mapping(config, "benchmark")["seed"]),
        "runtime_seconds": runtime_seconds,
        "comparison": "smoke diagnostic only" if smoke else ranking,
        "eligibility": "smoke: no ranking/recommendation" if smoke else ranking,
        "feature_qc": bundle.feature_qc,
        "extraction_failures": len(failed),
    }


def _json_ready(value: Any) -> Any:
    """將 caller config 複製成可稽核且可 strict JSON 序列化的值。"""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _append_failure_evidence(
    log_path: Path,
    error_info: tuple[Any, Any, Any],
    staging: Path,
) -> None:
    """以短暫 handle 寫入 traceback 與已落盤的具名 QC evidence。"""
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
            traceback.print_exception(*error_info, file=log_handle)
            for name in ("mask_provenance_qc.csv", "extraction_qc.csv"):
                path = staging / name
                if not path.is_file():
                    continue
                log_handle.write(f"\n[{name}]\n")
                log_handle.write(path.read_text(encoding="utf-8"))
            log_handle.flush()
            os.fsync(log_handle.fileno())
    except OSError:
        traceback.print_exception(*error_info)


def _append_secondary_failure_evidence(
    log_path: Path,
    stage: str,
    error: BaseException,
) -> None:
    """以 Python 3.10-safe 方式記錄 cleanup error，且永不掩蓋原始錯誤。"""
    message = f"secondary cleanup failure [{stage}]: {type(error).__name__}: {error}\n"
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
            log_handle.write(message)
    except BaseException:  # noqa: BLE001 - secondary evidence 不得取代 pipeline error
        try:
            sys.stderr.write(message)
        except BaseException:  # noqa: BLE001 - 原始 exception 永遠優先
            pass


def _validate_round2_config(config: dict[str, Any]) -> None:
    """驗證設定中的凍結 Round 2 合約。"""
    required = {"experiment_name", "round1", "models", "features", "benchmark", "smoke", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Round 2 config 缺少必要欄位：{missing}")
    if config["models"] != list(ROUND2_MODELS):
        raise ValueError(f"Round 2 models 必須是 {list(ROUND2_MODELS)}")
    if config["features"] != _ROUND2_FEATURES:
        raise ValueError("Round 2 features 不符合 paper_style_median 93 predictors 合約")
    if not isinstance(config["benchmark"], Mapping) or config["benchmark"].get("seed") != 20260804:
        raise ValueError("Round 2 benchmark seed 必須是 20260804")
    if not isinstance(config["round1"], Mapping):
        raise ValueError("Round 2 round1 必須是 mapping")
    expected = config["round1"].get("expected")
    if not isinstance(expected, Mapping) or dict(expected) != _FROZEN_ROUND1_EXPECTED:
        raise ValueError("Round 2 round1 expected 不符合凍結 identity")
    roster_sha256 = config["round1"].get("roster_sha256")
    if (
        not _is_sha256(roster_sha256)
        or roster_sha256.lower() != _FROZEN_ROUND1_ROSTER_SHA256
    ):
        raise ValueError("Round 2 roster_sha256 不符合凍結 identity")
    artifacts = config["round1"].get("artifact_sha256")
    if (
        not isinstance(artifacts, Mapping)
        or dict(artifacts) != _FROZEN_ROUND1_ARTIFACT_SHA256
    ):
        raise ValueError("Round 2 artifact_sha256 不符合凍結 identity")
    output = config["output"]
    if not isinstance(output, Mapping) or not isinstance(output.get("dir"), str):
        raise ValueError("Round 2 output.dir 必須是字串")
    resolve_round2_output_dir(output["dir"], smoke=False)


def _is_sha256(value: object) -> bool:
    """判斷值是否為 64 位元十六進位 SHA-256 字串。"""
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _existing_path_components(path: Path) -> Iterator[Path]:
    """依序產生目標路徑中已存在的 lexical 元件。"""
    anchor = Path(path.anchor)
    current = anchor
    if current.exists() or current.is_symlink():
        yield current
    for part in path.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        yield current


def _is_reparse_point(path: Path) -> bool:
    """判斷路徑是否為 symlink 或 Windows reparse point/junction。"""
    try:
        status = os.lstat(path)
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(status, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


if __name__ == "__main__":
    raise SystemExit(main())
