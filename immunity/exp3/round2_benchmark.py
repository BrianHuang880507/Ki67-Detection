"""Round 2 Paper93 frozen-split benchmark 的薄型 adapter。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from immunity.exp3.benchmark import (
    BenchmarkResult,
    OuterSplit,
    run_phase_feature_set_benchmark,
)
from immunity.exp3.feature_sets import PAPER_STYLE_FOV_FEATURES
from immunity.exp3.feature_sets import PRIMARY_FOV_FEATURES
from immunity.exp3.round2_evidence import Round1Evidence


ROUND2_MODELS = ("extra_trees", "random_forest")
_ROUND2_SOURCE = "round2_paper93"
_ROUND2_FEATURE_SET = "paper_style_median"
_ROUND1_FEATURE_SET = "basic_median"
_VALIDATIONS = (
    "leave_one_b_out",
    "leave_one_passage_out",
    "leave_one_group_out",
    "leave_one_condition_out",
)
_EXPECTED_FOLD_COUNTS = dict(zip(_VALIDATIONS, (3, 3, 9, 8), strict=True))
_CONFIGURATION_CONTRACT = {
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
_DEFAULT_SIMPLICITY_RANKS = {"random_forest": 5, "extra_trees": 6}


@dataclass(frozen=True)
class Round2Comparison:
    """保存 33/93 四組 candidate 與獨立 Dummy 證據。

    Attributes:
        fold_metrics: 四組 candidate 的逐 fold metrics；failed row 仍保留。
        predictions: 四組 candidate 的 outer-test OOF predictions。
        dummy_metrics: 唯讀 Round 1 Dummy fold metrics。
        dummy_predictions: 唯讀 Round 1 Dummy OOF predictions。
        hyperparameters: 四組 candidate 成功 folds 的 tuning evidence。
        feature_importance: 四組 candidate 的 diagnostic importance evidence。
        failures: 四組 candidate 的 fold failure evidence。
    """

    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    dummy_metrics: pd.DataFrame
    dummy_predictions: pd.DataFrame
    hyperparameters: pd.DataFrame
    feature_importance: pd.DataFrame
    failures: pd.DataFrame


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


def build_round2_comparison(
    evidence: Round1Evidence,
    paper93_result: BenchmarkResult,
    expected_splits: Mapping[str, Sequence[str]],
) -> Round2Comparison:
    """組合 frozen Round 1 與 Paper93 結果，但不改寫來源證據。

    Args:
        evidence: Task 2 載入的唯讀 Round 1 ET/RF/Dummy 證據。
        paper93_result: Task 4 已正規化的 ET/RF Paper93 結果。
        expected_splits: 四個 validation families 的 canonical split IDs。

    Returns:
        Dummy 與精確四組 candidate 分離的比較證據副本。

    Raises:
        ValueError: Split contract 或四組 candidate identity 不完整時拋出。
    """
    canonical_splits = _canonical_expected_splits(expected_splits)
    baseline_metrics = _round1_candidate_rows(evidence.selected_metrics)
    baseline_predictions = _round1_candidate_rows(evidence.selected_predictions)
    baseline_hyperparameters = _round1_candidate_rows(
        evidence.selected_hyperparameters
    )
    baseline_importance = _round1_candidate_rows(evidence.selected_importance)
    baseline_failures = _round1_candidate_rows(evidence.selected_failures)

    fold_metrics = _combine_candidate_frames(
        baseline_metrics, paper93_result.fold_metrics
    )
    predictions = _combine_candidate_frames(
        baseline_predictions, paper93_result.predictions
    )
    hyperparameters = _combine_candidate_frames(
        baseline_hyperparameters, paper93_result.hyperparameters
    )
    feature_importance = _combine_candidate_frames(
        baseline_importance, paper93_result.feature_importance
    )
    failures = _combine_candidate_frames(
        baseline_failures, paper93_result.failures
    )
    actual_configurations = set(
        fold_metrics.get("configuration_id", pd.Series(dtype=str)).astype(str)
    )
    if actual_configurations != set(_CONFIGURATION_CONTRACT):
        raise ValueError(
            "Round 2 comparison 必須精確包含四組 candidate configurations："
            f"{sorted(actual_configurations)}"
        )

    dummy_metrics = _round1_dummy_rows(evidence.selected_metrics)
    dummy_predictions = _round1_dummy_rows(evidence.selected_predictions)
    attrs = {
        "round2_expected_splits": canonical_splits,
        "round1_simplicity_ranks": _round1_simplicity_ranks(
            evidence.model_ranking
        ),
    }
    for frame in (
        fold_metrics,
        predictions,
        dummy_metrics,
        dummy_predictions,
        hyperparameters,
        feature_importance,
        failures,
    ):
        frame.attrs.update(attrs)
    return Round2Comparison(
        fold_metrics=fold_metrics,
        predictions=predictions,
        dummy_metrics=dummy_metrics,
        dummy_predictions=dummy_predictions,
        hyperparameters=hyperparameters,
        feature_importance=feature_importance,
        failures=failures,
    )


def rank_round2_configurations(
    comparison: Round2Comparison,
    predictor_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """以五項 gates 與四-family pooled OOF MAE 排名四組設定。

    Gate metrics 使用逐 fold median MAE/R²；family ranks 則只使用精確完整的
    pooled OOF predictions，兩者刻意以不同欄名保存，避免混用。單一 failed
    outer fold 只會讓所屬 configuration 失去完整性，不影響其他設定。

    Args:
        comparison: ``build_round2_comparison`` 產生的候選與 Dummy 證據。
        predictor_sets: ``basic_median`` 與 ``paper_style_median`` predictors。

    Returns:
        固定四列的可稽核 ranking，含 eligibility、strict tie band 與唯一推薦。

    Raises:
        ValueError: 比較表缺欄、缺少 canonical split metadata 或 identity 不合法時拋出。
    """
    _require_columns(
        comparison.fold_metrics,
        (
            "validation",
            "fold",
            "split_id",
            "model",
            "feature_set",
            "configuration_id",
            "n_test",
            "mae",
            "r2",
            "status",
        ),
        "fold_metrics",
    )
    _require_columns(
        comparison.predictions,
        (
            "validation",
            "split_id",
            "configuration_id",
            "image_key",
            "observed_ido_score",
            "predicted_ido_score",
        ),
        "predictions",
    )
    expected_splits = comparison.fold_metrics.attrs.get("round2_expected_splits")
    if not isinstance(expected_splits, Mapping):
        raise ValueError("Round2Comparison 缺少 canonical expected split metadata")
    canonical_splits = _canonical_expected_splits(expected_splits)
    dummy_reference = {
        validation: _validated_dummy_predictions(
            comparison.dummy_metrics,
            comparison.dummy_predictions,
            validation,
            set(canonical_splits[validation]),
        )
        for validation in _VALIDATIONS
    }
    dummy_complete = all(
        dummy_reference[validation] is not None for validation in _VALIDATIONS
    )
    dummy_medians = {
        validation: _median_fold_value(
            comparison.dummy_metrics,
            "dummy_median__none",
            validation,
            "mae",
        )
        for validation in _VALIDATIONS
    }
    simplicity_ranks = _comparison_simplicity_ranks(comparison.fold_metrics)
    rows: list[dict[str, Any]] = []
    validated_by_configuration: dict[str, dict[str, pd.DataFrame | None]] = {}

    for configuration_id, (model, feature_set, source_round) in (
        _CONFIGURATION_CONTRACT.items()
    ):
        validated_predictions = {
            validation: _validated_candidate_predictions(
                comparison,
                configuration_id,
                validation,
                set(canonical_splits[validation]),
                dummy_reference[validation],
            )
            for validation in _VALIDATIONS
        }
        validated_by_configuration[configuration_id] = validated_predictions
        row: dict[str, Any] = {
            "configuration_id": configuration_id,
            "model": model,
            "feature_set": feature_set,
            "source_round": source_round,
            "feature_count": _feature_count(predictor_sets, feature_set),
        }
        mae_wins = 0
        positive_r2 = 0
        prediction_checks: list[bool] = []
        for validation in _VALIDATIONS:
            median_mae = _median_fold_value(
                comparison.fold_metrics,
                configuration_id,
                validation,
                "mae",
            )
            median_r2 = _median_fold_value(
                comparison.fold_metrics,
                configuration_id,
                validation,
                "r2",
            )
            pooled_oof_mae = _pooled_oof_mae(
                validated_predictions[validation]
            )
            observed_sd, prediction_sd, sd_gate = _prediction_sd_check(
                validated_predictions[validation]
            )
            if (
                np.isfinite(median_mae)
                and np.isfinite(dummy_medians[validation])
                and median_mae < dummy_medians[validation]
            ):
                mae_wins += 1
            if np.isfinite(median_r2) and median_r2 > 0:
                positive_r2 += 1
            prediction_checks.append(sd_gate)
            row.update(
                {
                    f"{validation}_median_fold_mae": median_mae,
                    f"{validation}_median_fold_r2": median_r2,
                    f"{validation}_oof_mae": pooled_oof_mae,
                    f"{validation}_weight": 0.25,
                    f"{validation}_oof_observed_sd": observed_sd,
                    f"{validation}_oof_prediction_sd": prediction_sd,
                    f"{validation}_prediction_sd_gate": sd_gate,
                }
            )

        complete_gate = _configuration_complete(
            comparison,
            configuration_id,
            validated_predictions,
        )
        gates = {
            "mae_beats_dummy_gate": dummy_complete and mae_wins >= 3,
            "positive_r2_gate": positive_r2 >= 2,
            "complete_outer_folds_gate": complete_gate,
            "phase_only_feature_gate": _phase_whitelist_gate(
                comparison.fold_metrics,
                predictor_sets,
                configuration_id,
                feature_set,
            ),
            "prediction_sd_gate": all(prediction_checks),
        }
        reasons = [
            reason
            for passed, reason in (
                (
                    gates["mae_beats_dummy_gate"],
                    "mae_not_better_than_dummy_in_3_validations",
                ),
                (
                    gates["positive_r2_gate"],
                    "positive_median_r2_in_fewer_than_2_validations",
                ),
                (
                    gates["complete_outer_folds_gate"],
                    "failed_or_missing_outer_folds_or_oof",
                ),
                (
                    gates["phase_only_feature_gate"],
                    "nonexact_phase_feature_whitelist",
                ),
                (
                    gates["prediction_sd_gate"],
                    "prediction_sd_below_5_percent",
                ),
            )
            if not passed
        ]
        row.update(
            {
                "validations_beating_dummy": mae_wins,
                "validations_with_positive_r2": positive_r2,
                **gates,
                "eligible": not reasons,
                "reasons_json": json.dumps(
                    reasons, ensure_ascii=False, separators=(",", ":")
                ),
                "overall_oof_spearman": _overall_oof_spearman(
                    validated_predictions
                ),
                "simplicity_rank": simplicity_ranks[model],
                "completeness_issues_json": json.dumps(
                    _completeness_issues(
                        comparison,
                        configuration_id,
                        canonical_splits,
                        validated_predictions,
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "feature_importance_issues_json": json.dumps(
                    _feature_importance_issues(comparison, configuration_id),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        rows.append(row)

    ranking = pd.DataFrame(rows)
    for validation in _VALIDATIONS:
        ranking[f"{validation}_rank"] = ranking[
            f"{validation}_oof_mae"
        ].rank(method="min", ascending=True)
    rank_columns = [f"{validation}_rank" for validation in _VALIDATIONS]
    complete_ranks = ranking[rank_columns].notna().all(axis=1)
    ranking["average_rank"] = np.nan
    ranking.loc[complete_ranks, "average_rank"] = (
        ranking.loc[complete_ranks, rank_columns].sum(axis=1) * 0.25
    )
    ranking["worst_validation_rank"] = np.nan
    ranking.loc[complete_ranks, "worst_validation_rank"] = ranking.loc[
        complete_ranks, rank_columns
    ].max(axis=1)
    ranking["in_tie_band"] = False
    ranking["eliminated_by_simplicity"] = False
    ranking["recommended"] = False

    eligible = ranking[ranking["eligible"] & ranking["average_rank"].notna()]
    if eligible.empty:
        return ranking
    best_rank = float(eligible["average_rank"].min())
    in_band = ranking["eligible"] & (
        ranking["average_rank"] - best_rank < 0.25
    )
    ranking.loc[in_band, "in_tie_band"] = True
    for model in ROUND2_MODELS:
        basic_id = f"{model}__basic_median"
        paper_id = f"{model}__paper_style_median"
        ids_in_band = set(
            ranking.loc[
                ranking["in_tie_band"] & ranking["model"].eq(model),
                "configuration_id",
            ]
        )
        if {basic_id, paper_id}.issubset(ids_in_band):
            ranking.loc[
                ranking["configuration_id"].eq(paper_id),
                "eliminated_by_simplicity",
            ] = True
    finalists = ranking[
        ranking["in_tie_band"] & ~ranking["eliminated_by_simplicity"]
    ].copy()
    finalists = finalists.sort_values(
        [
            "worst_validation_rank",
            "leave_one_b_out_oof_mae",
            "overall_oof_spearman",
            "feature_count",
            "simplicity_rank",
            "model",
        ],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
        na_position="last",
    )
    recommended_id = str(finalists.iloc[0]["configuration_id"])
    ranking.loc[
        ranking["configuration_id"].eq(recommended_id), "recommended"
    ] = True
    return ranking


def select_round2_recommendation(ranking: pd.DataFrame) -> str | None:
    """取得唯一 eligible Round 2 recommendation。

    Args:
        ranking: ``rank_round2_configurations`` 產生的四列 ranking。

    Returns:
        唯一 recommended ``configuration_id``；無 eligible 設定時回傳 ``None``。

    Raises:
        ValueError: Ranking 含多個推薦或推薦列並非 eligible 時拋出。
    """
    _require_columns(
        ranking,
        ("configuration_id", "eligible", "recommended"),
        "round2 ranking",
    )
    recommended = ranking[ranking["recommended"].astype(bool)]
    if len(recommended) > 1:
        raise ValueError("Round 2 ranking 最多只能有一個 recommendation")
    if recommended.empty:
        return None
    if not bool(recommended.iloc[0]["eligible"]):
        raise ValueError("Round 2 recommendation 必須為 eligible")
    return str(recommended.iloc[0]["configuration_id"])


def _canonical_expected_splits(
    expected_splits: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """驗證並凍結四-family 3/3/9/8 split contract。"""
    if set(expected_splits) != set(_VALIDATIONS):
        raise ValueError("expected_splits 必須精確包含四個 validation families")
    canonical: dict[str, tuple[str, ...]] = {}
    for validation in _VALIDATIONS:
        values = expected_splits[validation]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"{validation} split IDs 必須為 sequence")
        split_ids = tuple(str(value) for value in values)
        if (
            len(split_ids) != _EXPECTED_FOLD_COUNTS[validation]
            or len(split_ids) != len(set(split_ids))
            or any(
                not split_id.startswith(f"{validation}:")
                or split_id == f"{validation}:"
                for split_id in split_ids
            )
        ):
            raise ValueError(f"{validation} split IDs 不符合 canonical fold contract")
        canonical[validation] = split_ids
    return canonical


def _round1_candidate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """複製 Round 1 ET/RF rows 並補齊明確 comparison identity。"""
    if frame.empty or "model" not in frame.columns:
        return frame.copy(deep=True)
    result = frame[frame["model"].isin(ROUND2_MODELS)].copy(deep=True)
    result["source_round"] = "round1"
    result["feature_set"] = _ROUND1_FEATURE_SET
    result["configuration_id"] = (
        result["model"].astype(str) + "__basic_median"
    )
    return result.reset_index(drop=True)


def _round1_dummy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """複製 Round 1 Dummy rows，避免其進入 candidate tables。"""
    if frame.empty or "model" not in frame.columns:
        return frame.copy(deep=True)
    result = frame[frame["model"].eq("dummy_median")].copy(deep=True)
    result["source_round"] = "round1"
    result["feature_set"] = "none"
    result["configuration_id"] = "dummy_median__none"
    return result.reset_index(drop=True)


def _combine_candidate_frames(
    baseline: pd.DataFrame, paper93: pd.DataFrame
) -> pd.DataFrame:
    """依來源順序組合兩輪 candidate tables 的獨立副本。"""
    return pd.concat(
        [baseline.copy(deep=True), paper93.copy(deep=True)],
        ignore_index=True,
        sort=False,
    )


def _round1_simplicity_ranks(model_ranking: pd.DataFrame) -> dict[str, int]:
    """讀取既有 Round 1 simplicity rank，缺欄時使用固定 published order。"""
    ranks = dict(_DEFAULT_SIMPLICITY_RANKS)
    if {"model", "simplicity_rank"}.issubset(model_ranking.columns):
        for model in ROUND2_MODELS:
            values = pd.to_numeric(
                model_ranking.loc[
                    model_ranking["model"].eq(model), "simplicity_rank"
                ],
                errors="coerce",
            ).dropna()
            if len(values) == 1:
                ranks[model] = int(values.iloc[0])
    return ranks


def _comparison_simplicity_ranks(metrics: pd.DataFrame) -> dict[str, int]:
    """從 comparison metadata 取得兩模型的既有 simplicity ranks。"""
    value = metrics.attrs.get("round1_simplicity_ranks")
    if not isinstance(value, Mapping):
        return dict(_DEFAULT_SIMPLICITY_RANKS)
    return {
        model: int(value.get(model, _DEFAULT_SIMPLICITY_RANKS[model]))
        for model in ROUND2_MODELS
    }


def _metric_rows_complete(
    metrics: pd.DataFrame,
    configuration_id: str,
    validation: str,
    expected: set[str],
) -> bool:
    """驗證單一 configuration/family 的成功 metric identities。"""
    rows = metrics[
        metrics["configuration_id"].eq(configuration_id)
        & metrics["validation"].eq(validation)
    ]
    if rows.empty:
        return False
    split_ids = rows["split_id"].astype(str)
    expected_identity = (
        rows["validation"].astype(str) + ":" + rows["fold"].astype(str)
    )
    n_test = pd.to_numeric(rows["n_test"], errors="coerce").to_numpy(float)
    return bool(
        len(rows) == len(expected)
        and not split_ids.duplicated().any()
        and set(split_ids) == expected
        and split_ids.reset_index(drop=True).equals(
            expected_identity.reset_index(drop=True)
        )
        and rows["status"].eq("ok").all()
        and np.isfinite(n_test).all()
        and np.all(n_test >= 1)
        and np.equal(n_test, np.floor(n_test)).all()
    )


def _validated_dummy_predictions(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    validation: str,
    expected: set[str],
) -> pd.DataFrame | None:
    """驗證 frozen Dummy metric/OOF 完整性並回傳 canonical membership。"""
    required = {
        "configuration_id",
        "validation",
        "split_id",
        "image_key",
        "observed_ido_score",
    }
    if not required.issubset(predictions.columns) or not _metric_rows_complete(
        metrics, "dummy_median__none", validation, expected
    ):
        return None
    rows = predictions[
        predictions["configuration_id"].eq("dummy_median__none")
        & predictions["validation"].eq(validation)
    ].copy()
    return _validate_oof_rows(
        metrics,
        rows,
        "dummy_median__none",
        validation,
        expected,
    )


def _validated_candidate_predictions(
    comparison: Round2Comparison,
    configuration_id: str,
    validation: str,
    expected: set[str],
    dummy_reference: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """驗證 candidate OOF count、membership 與 frozen target identity。"""
    if dummy_reference is None or not _metric_rows_complete(
        comparison.fold_metrics, configuration_id, validation, expected
    ):
        return None
    rows = comparison.predictions[
        comparison.predictions["configuration_id"].eq(configuration_id)
        & comparison.predictions["validation"].eq(validation)
    ].copy()
    validated = _validate_oof_rows(
        comparison.fold_metrics,
        rows,
        configuration_id,
        validation,
        expected,
    )
    if validated is None:
        return None
    keys = ["split_id", "image_key"]
    actual = validated.sort_values(keys, kind="stable").reset_index(drop=True)
    frozen = dummy_reference.sort_values(keys, kind="stable").reset_index(drop=True)
    if not actual[keys].astype(str).equals(frozen[keys].astype(str)):
        return None
    actual_observed = pd.to_numeric(
        actual["observed_ido_score"], errors="coerce"
    ).to_numpy(float)
    frozen_observed = pd.to_numeric(
        frozen["observed_ido_score"], errors="coerce"
    ).to_numpy(float)
    if not np.allclose(
        actual_observed, frozen_observed, rtol=0, atol=1e-12, equal_nan=False
    ):
        return None
    return validated


def _validate_oof_rows(
    metrics: pd.DataFrame,
    rows: pd.DataFrame,
    configuration_id: str,
    validation: str,
    expected: set[str],
) -> pd.DataFrame | None:
    """套用逐 split n_test、唯一 membership 與 finite values 檢查。"""
    if rows.empty or rows[["split_id", "image_key"]].isna().any().any():
        return None
    rows["split_id"] = rows["split_id"].astype(str)
    rows["image_key"] = rows["image_key"].astype(str)
    observed = pd.to_numeric(rows["observed_ido_score"], errors="coerce")
    predicted = (
        pd.to_numeric(rows["predicted_ido_score"], errors="coerce")
        if "predicted_ido_score" in rows.columns
        else observed
    )
    if (
        set(rows["split_id"]) != expected
        or rows["image_key"].str.strip().eq("").any()
        or rows.duplicated(["split_id", "image_key"]).any()
        or not np.isfinite(observed.to_numpy(float)).all()
        or not np.isfinite(predicted.to_numpy(float)).all()
    ):
        return None
    metric_rows = metrics[
        metrics["configuration_id"].eq(configuration_id)
        & metrics["validation"].eq(validation)
    ].set_index("split_id")
    for split_id in expected:
        if int(rows["split_id"].eq(split_id).sum()) != int(
            metric_rows.loc[split_id, "n_test"]
        ):
            return None
    return rows


def _configuration_complete(
    comparison: Round2Comparison,
    configuration_id: str,
    validated_predictions: Mapping[str, pd.DataFrame | None],
) -> bool:
    """拒絕所屬 failure 或任一 family metric/OOF 不完整的設定。"""
    if not comparison.failures.empty and "configuration_id" in comparison.failures:
        if comparison.failures["configuration_id"].eq(configuration_id).any():
            return False
    return all(
        validated_predictions.get(validation) is not None
        for validation in _VALIDATIONS
    )


def _completeness_issues(
    comparison: Round2Comparison,
    configuration_id: str,
    expected_splits: Mapping[str, Sequence[str]],
    validated_predictions: Mapping[str, pd.DataFrame | None],
) -> list[str]:
    """列出 failed/missing metric 與 OOF split identities，供 QC 稽核。"""
    issues: list[str] = []
    metrics = comparison.fold_metrics[
        comparison.fold_metrics["configuration_id"].eq(configuration_id)
    ]
    predictions = comparison.predictions[
        comparison.predictions["configuration_id"].eq(configuration_id)
    ]
    for validation in _VALIDATIONS:
        family_metrics = metrics[metrics["validation"].eq(validation)]
        family_predictions = predictions[predictions["validation"].eq(validation)]
        for split_id in expected_splits[validation]:
            metric_rows = family_metrics[
                family_metrics["split_id"].astype(str).eq(split_id)
            ]
            if metric_rows.empty:
                issues.append(f"missing_metric:{split_id}")
                continue
            if len(metric_rows) != 1:
                issues.append(f"duplicate_metric:{split_id}")
                continue
            if not metric_rows["status"].eq("ok").all():
                issues.append(f"failed_metric:{split_id}")
            expected_count = pd.to_numeric(
                metric_rows["n_test"], errors="coerce"
            ).iloc[0]
            actual_count = int(
                family_predictions["split_id"].astype(str).eq(split_id).sum()
            )
            if not np.isfinite(expected_count) or actual_count != int(expected_count):
                issues.append(
                    f"oof_count:{split_id}:expected={expected_count}:actual={actual_count}"
                )
        if validated_predictions.get(validation) is None and not any(
            validation in issue for issue in issues
        ):
            issues.append(f"invalid_oof_membership:{validation}")
    if not comparison.failures.empty and {
        "configuration_id",
        "split_id",
    }.issubset(comparison.failures.columns):
        failures = comparison.failures[
            comparison.failures["configuration_id"].eq(configuration_id)
        ]
        for split_id in failures["split_id"].astype(str):
            marker = f"model_failure:{split_id}"
            if marker not in issues:
                issues.append(marker)
    return issues


def _feature_importance_issues(
    comparison: Round2Comparison, configuration_id: str
) -> list[str]:
    """列出成功 folds 缺少 diagnostic importance 的 identities。"""
    successful = comparison.fold_metrics[
        comparison.fold_metrics["configuration_id"].eq(configuration_id)
        & comparison.fold_metrics["status"].eq("ok")
    ]
    if {
        "configuration_id",
        "split_id",
    }.issubset(comparison.feature_importance.columns):
        importance_ids = set(
            comparison.feature_importance.loc[
                comparison.feature_importance["configuration_id"].eq(
                    configuration_id
                ),
                "split_id",
            ].astype(str)
        )
    else:
        importance_ids = set()
    return [
        f"missing_feature_importance:{split_id}"
        for split_id in successful["split_id"].astype(str)
        if split_id not in importance_ids
    ]


def _median_fold_value(
    metrics: pd.DataFrame,
    configuration_id: str,
    validation: str,
    column: str,
) -> float:
    """計算成功 fold 的 median gate metric。"""
    rows = metrics[
        metrics["configuration_id"].eq(configuration_id)
        & metrics["validation"].eq(validation)
        & metrics["status"].eq("ok")
    ]
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _feature_count(
    predictor_sets: Mapping[str, Sequence[str]], feature_set: str
) -> int:
    """取得實際 predictor 數量；無效 sequence 以零保留 audit row。"""
    value = predictor_sets.get(feature_set)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return 0
    return len(value)


def _phase_whitelist_gate(
    metrics: pd.DataFrame,
    predictor_sets: Mapping[str, Sequence[str]],
    configuration_id: str,
    feature_set: str,
) -> bool:
    """檢查 configuration identity 與 authoritative 33/93 whitelist。"""
    configured = predictor_sets.get(feature_set)
    if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
        return False
    expected = (
        PRIMARY_FOV_FEATURES
        if feature_set == _ROUND1_FEATURE_SET
        else PAPER_STYLE_FOV_FEATURES
    )
    rows = metrics[metrics["configuration_id"].eq(configuration_id)]
    if rows.empty or set(rows["feature_set"].astype(str)) != {feature_set}:
        return False
    return tuple(str(value) for value in configured) == tuple(expected)


def _pooled_oof_mae(predictions: pd.DataFrame | None) -> float:
    """由完整 OOF rows 計算 pooled MAE，而非 fold MAE 的平均。"""
    if predictions is None:
        return np.nan
    observed = pd.to_numeric(
        predictions["observed_ido_score"], errors="coerce"
    ).to_numpy(float)
    predicted = pd.to_numeric(
        predictions["predicted_ido_score"], errors="coerce"
    ).to_numpy(float)
    return float(np.mean(np.abs(observed - predicted)))


def _prediction_sd_check(
    predictions: pd.DataFrame | None,
) -> tuple[float, float, bool]:
    """計算 pooled OOF SD 並套用 prediction >= observed 5% gate。"""
    if predictions is None:
        return np.nan, np.nan, False
    observed = pd.to_numeric(
        predictions["observed_ido_score"], errors="coerce"
    ).to_numpy(float)
    predicted = pd.to_numeric(
        predictions["predicted_ido_score"], errors="coerce"
    ).to_numpy(float)
    observed_sd = float(np.std(observed))
    prediction_sd = float(np.std(predicted))
    return observed_sd, prediction_sd, prediction_sd >= 0.05 * observed_sd


def _overall_oof_spearman(
    validated_predictions: Mapping[str, pd.DataFrame | None],
) -> float:
    """計算四-family 完整 raw OOF rows 的整體 Spearman。"""
    if any(validated_predictions.get(validation) is None for validation in _VALIDATIONS):
        return np.nan
    rows = pd.concat(
        [
            validated_predictions[validation]
            for validation in _VALIDATIONS
            if validated_predictions[validation] is not None
        ],
        ignore_index=True,
    )
    observed = pd.to_numeric(rows["observed_ido_score"], errors="coerce")
    predicted = pd.to_numeric(rows["predicted_ido_score"], errors="coerce")
    if observed.nunique() < 2 or predicted.nunique() < 2:
        return np.nan
    return float(observed.corr(predicted, method="spearman"))


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], table: str
) -> None:
    """拒絕缺少比較 contract 欄位的資料表。"""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table} 缺少 Round 2 comparison 欄位：{missing}")


__all__ = [
    "ROUND2_MODELS",
    "Round2Comparison",
    "build_round2_comparison",
    "normalize_round2_result",
    "rank_round2_configurations",
    "run_paper93_benchmark",
    "select_round2_recommendation",
]
