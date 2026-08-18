"""Exp4 arm assembly 前的 fail-closed 特徵健康檢查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


NEAR_ZERO_RELATIVE_STD_THRESHOLD = 0.05
MAX_ZERO_VARIANCE_COLUMNS = 3
MAX_FALLBACK_RATE = 0.01
DEFAULT_HEALTH_SCOPE = "eligible_rows_supplied_by_caller"
FEATURE_HEALTH_COLUMNS = (
    "feature",
    "nunique",
    "mean",
    "std",
    "relative_std",
    "is_zero_variance",
    "is_near_zero_variance",
    "removed_from_arms",
    "fallback_scope",
    "eligible_cell_count",
    "fallback_count",
    "fallback_rate",
    "status",
)


@dataclass(frozen=True)
class FeatureHealthResult:
    """保存特徵健康 gate 結果與 arm allowlist。

    Attributes:
        blocked: 是否因非有限值、過多常數欄或 fallback rate 而停止。
        reasons: 可供人員確認的 fail-closed 原因。
        healthy_feature_columns: 可進入後續 arm assembly 的欄位順序。
        removed_columns: 因零變異而從全部 arm 移除的欄位。
        report: 每個 feature 一列的健康指標。
        fallback_scope: fallback denominator 的 row scope。
        eligible_cell_count: fallback rate 使用的 eligible row 分母。
    """

    blocked: bool
    reasons: tuple[str, ...]
    healthy_feature_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    report: pd.DataFrame
    fallback_scope: str
    eligible_cell_count: int

    def write_report(self, path: str | Path) -> None:
        """將逐欄健康報告寫成 CSV。

        Args:
            path: CSV 輸出路徑。
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.report.to_csv(output, index=False)

    def to_metadata_payload(self) -> dict[str, object]:
        """回傳可合併至 run metadata 的 gate 摘要。"""
        return {
            "blocked": self.blocked,
            "reasons": list(self.reasons),
            "healthy_feature_columns": list(self.healthy_feature_columns),
            "removed_columns": list(self.removed_columns),
            "fallback_scope": self.fallback_scope,
            "eligible_cell_count": self.eligible_cell_count,
            "near_zero_cv_threshold": NEAR_ZERO_RELATIVE_STD_THRESHOLD,
            "near_zero_columns": [
                str(row.feature)
                for row in self.report.itertuples(index=False)
                if bool(row.is_near_zero_variance)
            ],
            "coefficient_of_variation": {
                str(row.feature): float(row.relative_std)
                for row in self.report.itertuples(index=False)
            },
            "fallback_counts": {
                str(row.feature): int(row.fallback_count)
                for row in self.report.itertuples(index=False)
            },
            "fallback_rates": {
                str(row.feature): float(row.fallback_rate)
                for row in self.report.itertuples(index=False)
            },
        }


def evaluate_feature_health(
    features: pd.DataFrame,
    *,
    fallback_counts: Mapping[str, int],
    eligible_cell_count: int,
    feature_columns: Sequence[str] | None = None,
    fallback_scope: str = DEFAULT_HEALTH_SCOPE,
) -> FeatureHealthResult:
    """檢查非有限值、零／近零變異與逐欄 fallback rate。

    v5 將近零變異定義為 coefficient of variation
    ``std(ddof=0) / abs(mean) < 0.05``。命中只記錄、不移除；mean 為 0 且
    std 非 0 時 CV 明確記為 infinity，因此不會誤判為 near-zero。零變異仍
    依既有規格移除。

    Args:
        features: arm assembly 前的 feature DataFrame。
        fallback_counts: 已依 eligible rows 對齊後的逐欄 fallback 次數。
        eligible_cell_count: fallback rate 的明確分母。
        feature_columns: 要檢查的欄位順序；省略時使用所有 ``cell__`` 欄。
        fallback_scope: 描述分母涵蓋哪些 rows 的 machine-readable 字串。

    Returns:
        包含 report、健康 allowlist、移除欄位與 fail-closed 原因的結果。

    Raises:
        TypeError: ``features`` 不是 DataFrame 時拋出。
        ValueError: 欄位、分母、scope 或 fallback counts 不合法時拋出。
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features 必須是 pandas DataFrame")
    if isinstance(eligible_cell_count, bool) or not isinstance(
        eligible_cell_count, (int, np.integer)
    ):
        raise ValueError("eligible_cell_count 必須是正整數")
    denominator = int(eligible_cell_count)
    if denominator <= 0:
        raise ValueError("eligible_cell_count 必須是正整數")
    if len(features) != denominator:
        raise ValueError(
            "eligible_cell_count 必須等於 features row count："
            f"eligible_cell_count={denominator}, row count={len(features)}"
        )
    scope = str(fallback_scope).strip()
    if not scope:
        raise ValueError("fallback_scope 不可為空")

    if feature_columns is None:
        columns = tuple(
            str(column)
            for column in features.columns
            if str(column).startswith("cell__")
        )
    else:
        columns = tuple(str(column) for column in feature_columns)
    if not columns:
        raise ValueError("沒有可檢查的 feature columns")
    if len(set(columns)) != len(columns):
        raise ValueError("feature_columns 不可重複")
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(f"features 缺少欄位：{missing}")

    normalized_counts: dict[str, int] = {}
    for column in columns:
        raw_count = fallback_counts.get(column, 0)
        if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)):
            raise ValueError(f"{column} fallback_count 必須是非負整數")
        count = int(raw_count)
        if count < 0 or count > denominator:
            raise ValueError(
                f"{column} fallback_count 必須介於 0 與 eligible_cell_count"
            )
        normalized_counts[column] = count

    rows: list[dict[str, object]] = []
    reasons: list[str] = []
    removed: list[str] = []
    healthy: list[str] = []
    nonfinite_columns: list[str] = []
    excessive_fallback_columns: list[str] = []
    for column in columns:
        try:
            values = features[column].to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            values = np.full(len(features), np.nan, dtype=np.float64)
        finite = bool(values.size > 0 and np.isfinite(values).all())
        fallback_count = normalized_counts[column]
        fallback_rate = fallback_count / denominator

        if finite:
            nunique = int(pd.Series(values).nunique(dropna=False))
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            is_zero = bool(nunique == 1 or std == 0.0)
            if mean == 0.0:
                relative_std = 0.0 if std == 0.0 else float("inf")
                is_near_zero = False
            else:
                relative_std = float(std / abs(mean))
                is_near_zero = bool(
                    not is_zero
                    and relative_std < NEAR_ZERO_RELATIVE_STD_THRESHOLD
                )
        else:
            nunique = int(pd.Series(values).nunique(dropna=False))
            mean = float("nan")
            std = float("nan")
            relative_std = float("nan")
            is_zero = False
            is_near_zero = False
            nonfinite_columns.append(column)

        statuses: list[str] = []
        if not finite:
            statuses.append("blocked_nonfinite")
        if is_zero:
            statuses.append("removed_zero_variance")
            removed.append(column)
        elif is_near_zero:
            statuses.append("near_zero_variance")
        if fallback_rate > MAX_FALLBACK_RATE:
            statuses.append("blocked_fallback_rate")
            excessive_fallback_columns.append(column)
        if not statuses:
            statuses.append("healthy")
        if finite and not is_zero and fallback_rate <= MAX_FALLBACK_RATE:
            healthy.append(column)

        rows.append(
            {
                "feature": column,
                "nunique": nunique,
                "mean": mean,
                "std": std,
                "relative_std": relative_std,
                "is_zero_variance": is_zero,
                "is_near_zero_variance": is_near_zero,
                "removed_from_arms": is_zero,
                "fallback_scope": scope,
                "eligible_cell_count": denominator,
                "fallback_count": fallback_count,
                "fallback_rate": fallback_rate,
                "status": ";".join(statuses),
            }
        )

    if nonfinite_columns:
        reasons.append(f"nonfinite feature columns: {nonfinite_columns}")
    if len(removed) > MAX_ZERO_VARIANCE_COLUMNS:
        reasons.append(
            "zero-variance feature count exceeds 3: "
            f"count={len(removed)}, columns={removed}"
        )
    if excessive_fallback_columns:
        reasons.append(
            "fallback rate exceeds 1%: "
            f"columns={excessive_fallback_columns}"
        )

    report = pd.DataFrame(rows, columns=FEATURE_HEALTH_COLUMNS)
    return FeatureHealthResult(
        blocked=bool(reasons),
        reasons=tuple(reasons),
        healthy_feature_columns=tuple(healthy),
        removed_columns=tuple(removed),
        report=report,
        fallback_scope=scope,
        eligible_cell_count=denominator,
    )
