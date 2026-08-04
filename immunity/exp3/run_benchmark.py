"""提供 Exp3 benchmark 的設定與命令列介面。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP3_OUTPUT_ROOT = (PROJECT_ROOT / "immunity" / "outputs" / "exp3").resolve()


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
