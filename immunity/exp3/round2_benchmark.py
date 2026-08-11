"""Round 2 Paper93 frozen-split benchmark 的薄型 adapter。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from immunity.exp3.benchmark import (
    BenchmarkResult,
    OuterSplit,
    run_phase_feature_set_benchmark,
)
from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES


ROUND2_MODELS = ("extra_trees", "random_forest")
_ROUND2_SOURCE = "round2_paper93"
_ROUND2_FEATURE_SET = "paper_style_median"


def run_paper93_benchmark(
    images: pd.DataFrame,
    splits: Sequence[OuterSplit],
    config: Mapping[str, Any],
) -> BenchmarkResult:
    """以 frozen splits 執行 Paper93 的兩模型 secondary benchmark。

    Args:
        images: Task 3 建立且具精確 93 個 Paper-style predictors 的 FOV 資料。
        splits: Task 2 還原的 Round 1 outer splits；會原樣轉交既有 adapter。
        config: Round 1 已鎖定的 seed、search 與 estimator pipeline 設定。

    Returns:
        加上 Round 2 identity 的 benchmark 結果；所有數值證據均保留。
    """
    raw = run_phase_feature_set_benchmark(
        images,
        {_ROUND2_FEATURE_SET: list(PAPER_STYLE_FOV_FEATURES)},
        splits,
        config,
        model_names=list(ROUND2_MODELS),
        feature_set_name=_ROUND2_FEATURE_SET,
    )
    return normalize_round2_result(raw)


def normalize_round2_result(result: BenchmarkResult) -> BenchmarkResult:
    """複製 benchmark 結果並統一附加 Round 2 identity 欄位。

    Args:
        result: 既有 secondary adapter 產生的原始 benchmark 結果。

    Returns:
        不修改輸入資料表與數值證據的新 ``BenchmarkResult``；每個結果表皆含
        ``source_round``、``feature_set`` 與 ``configuration_id``。
    """
    return BenchmarkResult(
        predictions=_normalize_round2_frame(result.predictions),
        fold_metrics=_normalize_round2_frame(result.fold_metrics),
        hyperparameters=_normalize_round2_frame(result.hyperparameters),
        feature_importance=_normalize_round2_frame(result.feature_importance),
        failures=_normalize_round2_frame(result.failures),
    )


def _normalize_round2_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """回傳保留列順序及原值、附加固定 identity 的結果表副本。"""
    normalized = frame.copy(deep=True)
    normalized["source_round"] = _ROUND2_SOURCE
    normalized["feature_set"] = _ROUND2_FEATURE_SET
    normalized["configuration_id"] = (
        normalized["model"].astype(str) + f"__{_ROUND2_FEATURE_SET}"
    )
    return normalized


__all__ = ["ROUND2_MODELS", "normalize_round2_result", "run_paper93_benchmark"]
