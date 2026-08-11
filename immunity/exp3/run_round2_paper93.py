"""載入並驗證 Exp3 Round 2 paper-style 實驗設定。"""

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


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
    parser.add_argument("--smoke-fovs-per-condition", type=int, default=None)
    return parser


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
    if not _is_sha256(roster_sha256):
        raise ValueError("Round 2 roster_sha256 必須是 SHA-256")
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
