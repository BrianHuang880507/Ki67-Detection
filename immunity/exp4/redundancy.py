"""Exp4 Rui 48 欄的 X-only Pearson 去冗餘。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


PEARSON_REDUNDANCY_THRESHOLD = 0.9
EXPECTED_RUI_FEATURE_COUNT = 48
EXPECTED_AUTHORITATIVE_COUNT = 28
EXPECTED_POST_BORDER_CELL_COUNT = 19_648
REDUNDANCY_REPORT_COLUMNS = (
    "record_type",
    "feature",
    "canonical_index",
    "decision",
    "paired_feature",
    "paired_canonical_index",
    "pearson_r",
    "abs_pearson_r",
    "is_authoritative_rui28",
    "set_relation",
    "threshold",
    "selection_inputs",
)

AUTHORITATIVE_RUI28_FEATURES = tuple(
    f"cell__{name}"
    for name in (
        "BoundingBoxArea",
        "Center_X",
        "Center_Y",
        "Compactness",
        "Eccentricity",
        "EquivalentDiameter",
        "Extent",
        "FormFactor",
        "MaxFeretDiameter",
        "MedianRadius",
        "MinorAxisLength",
        "Orientation",
        "Perimeter",
        "Solidity",
        "IntegratedIntensity",
        "IntegratedIntensityEdge",
        "MeanIntensityEdge",
        "StdIntensityEdge",
        "MassDisplacement",
        "MADIntensity",
        "Contrast",
        "Correlation",
        "Variance",
        "SumAverage",
        "SumEntropy",
        "DifferenceEntropy",
        "InfoMeas1",
        "InfoMeas2",
    )
)


class FeatureRedundancyError(ValueError):
    """表示 48 欄 X-only Pearson contract 不成立。"""


@dataclass(frozen=True)
class FeatureRedundancyResult:
    """保存 earlier-wins 決策與 Rui 28 集合差異。

    Attributes:
        retained_features: 依 canonical 順序保留的欄位。
        removed_features: 依 canonical 順序剔除的欄位。
        rui28_intersection: authoritative Rui 28 與本資料保留欄的交集。
        rui28_only: authoritative Rui 28 中被本資料移除的欄位。
        filtered_only: 本資料保留但不在 authoritative Rui 28 的欄位。
        report: 每欄決策及所有超門檻 pair 的長格式 evidence。
        threshold: 固定的 absolute Pearson gate。
        selection_inputs: 明示本 API 只使用 X。
    """

    retained_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    rui28_intersection: tuple[str, ...]
    rui28_only: tuple[str, ...]
    filtered_only: tuple[str, ...]
    report: pd.DataFrame
    threshold: float = PEARSON_REDUNDANCY_THRESHOLD
    selection_inputs: tuple[str, ...] = ("X",)

    def to_metadata_payload(self) -> dict[str, object]:
        """回傳可直接寫入 run metadata 的 X-only provenance。"""
        return {
            "threshold": self.threshold,
            "selection_inputs": list(self.selection_inputs),
            "canonical_rule": "upper_triangle_any_earlier_wins",
            "retained_features": list(self.retained_features),
            "removed_features": list(self.removed_features),
            "rui28_intersection": list(self.rui28_intersection),
            "rui28_only": list(self.rui28_only),
            "filtered_only": list(self.filtered_only),
        }


def analyze_feature_redundancy(
    x48: pd.DataFrame,
    canonical_order: Sequence[str],
    authoritative_rui28: Sequence[str] = AUTHORITATIVE_RUI28_FEATURES,
) -> FeatureRedundancyResult:
    """以 X-only Pearson upper triangle 移除 48 欄中的冗餘欄位。

    對 canonical index ``i < j`` 的所有 pair 計算 Pearson r；只要 later
    feature ``j`` 與任何 earlier feature ``i`` 滿足 ``abs(r) > 0.9``，就
    移除 ``j``。即使 ``i`` 自己被更早欄移除，仍可觸發 ``j``；決策 row 記錄
    最早觸發欄，另以 pair rows 保存所有超門檻關係。

    Args:
        x48: 完整正式母體的 48 個健康 whole-cell predictors；API 不接受 y。
        canonical_order: Rui 49 原始順序扣除零變異 MinIntensity 後的 48 欄。
        authoritative_rui28: 從 Rui xlsx 確認的 28 欄權威集合。

    Returns:
        保留／移除欄、Rui 28 交差集與完整 pair evidence。

    Raises:
        TypeError: ``x48`` 不是 DataFrame 時拋出。
        FeatureRedundancyError: 欄數、順序、有限值、變異或 Rui 28 roster 不合法。
    """
    if not isinstance(x48, pd.DataFrame):
        raise TypeError("x48 必須是 pandas DataFrame")
    if len(x48) != EXPECTED_POST_BORDER_CELL_COUNT:
        raise FeatureRedundancyError(
            "X48 必須使用完整 19,648-cell post-border population："
            f"actual={len(x48)}"
        )
    canonical = tuple(str(feature) for feature in canonical_order)
    authoritative = tuple(str(feature) for feature in authoritative_rui28)
    _validate_rosters(x48, canonical, authoritative)
    try:
        values = x48.loc[:, canonical].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FeatureRedundancyError("X48 必須全為數值") from error
    if values.shape[0] < 3:
        raise FeatureRedundancyError("Pearson 去冗餘至少需要 3 rows")
    if not np.isfinite(values).all():
        raise FeatureRedundancyError("X48 必須全為 finite")
    if np.any(np.std(values, axis=0, ddof=0) == 0.0):
        constants = [
            canonical[index]
            for index in np.flatnonzero(np.std(values, axis=0, ddof=0) == 0.0)
        ]
        raise FeatureRedundancyError(
            f"X48 不可含 zero-variance columns：{constants}"
        )

    correlations = np.corrcoef(values, rowvar=False)
    if correlations.shape != (EXPECTED_RUI_FEATURE_COUNT, EXPECTED_RUI_FEATURE_COUNT):
        raise FeatureRedundancyError("Pearson correlation matrix shape 不合法")
    if not np.isfinite(correlations).all():
        raise FeatureRedundancyError("Pearson correlation matrix 含 nonfinite")

    high_pairs: list[tuple[int, int, float]] = []
    triggers: dict[int, tuple[int, float]] = {}
    for later in range(1, len(canonical)):
        for earlier in range(later):
            correlation = float(correlations[earlier, later])
            if abs(correlation) > PEARSON_REDUNDANCY_THRESHOLD:
                high_pairs.append((earlier, later, correlation))
                triggers.setdefault(later, (earlier, correlation))

    removed = tuple(canonical[index] for index in sorted(triggers))
    removed_set = set(removed)
    retained = tuple(feature for feature in canonical if feature not in removed_set)
    authoritative_set = set(authoritative)
    retained_set = set(retained)
    intersection = tuple(
        feature for feature in authoritative if feature in retained_set
    )
    rui_only = tuple(
        feature for feature in authoritative if feature not in retained_set
    )
    filtered_only = tuple(
        feature for feature in retained if feature not in authoritative_set
    )

    rows: list[dict[str, object]] = []
    for index, feature in enumerate(canonical):
        trigger = triggers.get(index)
        decision = "removed" if trigger is not None else "retained"
        paired_index = trigger[0] if trigger is not None else None
        correlation = trigger[1] if trigger is not None else float("nan")
        rows.append(
            _report_row(
                record_type="feature_decision",
                feature=feature,
                canonical_index=index,
                decision=decision,
                paired_feature=(
                    canonical[paired_index] if paired_index is not None else ""
                ),
                paired_canonical_index=paired_index,
                correlation=correlation,
                authoritative_set=authoritative_set,
                retained_set=retained_set,
            )
        )
    for earlier, later, correlation in high_pairs:
        rows.append(
            _report_row(
                record_type="high_correlation_pair",
                feature=canonical[later],
                canonical_index=later,
                decision="above_threshold",
                paired_feature=canonical[earlier],
                paired_canonical_index=earlier,
                correlation=correlation,
                authoritative_set=authoritative_set,
                retained_set=retained_set,
            )
        )
    report = pd.DataFrame(rows, columns=REDUNDANCY_REPORT_COLUMNS)
    return FeatureRedundancyResult(
        retained_features=retained,
        removed_features=removed,
        rui28_intersection=intersection,
        rui28_only=rui_only,
        filtered_only=filtered_only,
        report=report,
    )


def _validate_rosters(
    x48: pd.DataFrame,
    canonical: tuple[str, ...],
    authoritative: tuple[str, ...],
) -> None:
    if len(canonical) != EXPECTED_RUI_FEATURE_COUNT or len(set(canonical)) != len(
        canonical
    ):
        raise FeatureRedundancyError("canonical_order 必須是 48 個唯一欄名")
    if len(authoritative) != EXPECTED_AUTHORITATIVE_COUNT or len(
        set(authoritative)
    ) != len(authoritative):
        raise FeatureRedundancyError("authoritative_rui28 必須是 28 個唯一欄名")
    if set(x48.columns) != set(canonical) or len(x48.columns) != len(canonical):
        raise FeatureRedundancyError(
            "X48 columns 必須與 canonical_order 為同一 48 欄集合"
        )
    missing_authoritative = sorted(set(authoritative) - set(canonical))
    if missing_authoritative:
        raise FeatureRedundancyError(
            f"authoritative_rui28 不在 X48：{missing_authoritative}"
        )


def _set_relation(
    feature: str,
    *,
    authoritative_set: set[str],
    retained_set: set[str],
) -> str:
    in_rui = feature in authoritative_set
    retained = feature in retained_set
    if in_rui and retained:
        return "intersection"
    if in_rui:
        return "rui28_only"
    if retained:
        return "filtered_only"
    return "neither"


def _report_row(
    *,
    record_type: str,
    feature: str,
    canonical_index: int,
    decision: str,
    paired_feature: str,
    paired_canonical_index: int | None,
    correlation: float,
    authoritative_set: set[str],
    retained_set: set[str],
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "feature": feature,
        "canonical_index": canonical_index,
        "decision": decision,
        "paired_feature": paired_feature,
        "paired_canonical_index": (
            float("nan")
            if paired_canonical_index is None
            else paired_canonical_index
        ),
        "pearson_r": correlation,
        "abs_pearson_r": abs(correlation),
        "is_authoritative_rui28": feature in authoritative_set,
        "set_relation": _set_relation(
            feature,
            authoritative_set=authoritative_set,
            retained_set=retained_set,
        ),
        "threshold": PEARSON_REDUNDANCY_THRESHOLD,
        "selection_inputs": "X",
    }


__all__ = [
    "AUTHORITATIVE_RUI28_FEATURES",
    "EXPECTED_RUI_FEATURE_COUNT",
    "EXPECTED_POST_BORDER_CELL_COUNT",
    "FeatureRedundancyError",
    "FeatureRedundancyResult",
    "PEARSON_REDUNDANCY_THRESHOLD",
    "REDUNDANCY_REPORT_COLUMNS",
    "analyze_feature_redundancy",
]
