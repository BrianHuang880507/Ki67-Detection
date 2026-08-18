"""Exp4 v4 group target 與 sparse-FOV sensitivity 純邏輯介面。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .cell_dedup import CELL_KEY_COLUMNS
from .data_snapshot import DEFAULT_GROUP_KEYS


GROUP_TARGET_COLUMNS = (
    "group_id",
    "b_id",
    "passage",
    "fov_count",
    "cells_before",
    "cells_after",
    "group_IDO_score",
    "group_IDO_score_all_cells",
    "delta",
    "confidence_flag",
)
FOV_IDO_SCORE_COLUMNS = (
    "image_key",
    "group_id",
    "b_id",
    "passage",
    "cells_after",
    "FOV_IDO_score",
)
SENSITIVITY_REPORT_COLUMNS = (
    "group_id",
    "sparse_fov_count",
    "target_with_sparse_fovs",
    "target_without_sparse_fovs",
    "delta_without_minus_with",
    "absolute_delta",
    "baseline_target_range",
    "absolute_delta_pct_of_range",
    "threshold_pct",
    "threshold_absolute",
    "status",
)


class TargetAggregationError(ValueError):
    """表示 v4 target 輸入或 fail-closed invariant 不符合規格。"""


class TargetSensitivityBlockedError(TargetAggregationError):
    """表示 sparse-FOV sensitivity 超過允許 threshold。"""


@dataclass(frozen=True)
class TargetAggregationResult:
    """保存正式 v4 targets、描述性 FOV scores 與已驗證 cell roster。

    Attributes:
        group_targets: 固定 ``GROUP_TARGET_COLUMNS`` 的正式九組 target。
        fov_ido_scores: 只供 QC 的 post-border FOV median，不進入 target chain。
        manifest: 通過 group/FOV gate 的 manifest copy。
        pre_border_cells: 唯一 whole-cell 的去重 master copy。
        post_border_cells: 依 retained keys 從 master 篩出的 trusted copy。
        expected_group_keys: 本次 fail-closed 的 exact group 順序。
    """

    group_targets: pd.DataFrame
    fov_ido_scores: pd.DataFrame
    manifest: pd.DataFrame
    pre_border_cells: pd.DataFrame
    post_border_cells: pd.DataFrame
    expected_group_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, int | float]:
        """回傳 CLI 可記錄的 cell/FOV/group 計數與 target range。"""
        scores = self.group_targets["group_IDO_score"].to_numpy(dtype=float)
        return {
            "pre_border_cell_count": len(self.pre_border_cells),
            "post_border_cell_count": len(self.post_border_cells),
            "fov_count": len(self.manifest),
            "group_count": len(self.group_targets),
            "group_target_range": float(np.max(scores) - np.min(scores)),
        }


@dataclass(frozen=True)
class SparseFovSensitivityResult:
    """保存 sparse-FOV 反事實 target 表與 fail-closed gate 狀態。

    ``report`` 永遠完整涵蓋 expected groups，讓 CLI 可先落盤證據，再呼叫
    ``raise_if_blocked`` 停止後續建模。

    Attributes:
        report: 固定 ``SENSITIVITY_REPORT_COLUMNS`` 的逐組敏感度表。
        sparse_fov_count: 本次驗證且正式 target 確實保留的 sparse FOV 數。
        baseline_target_range: 九組 with-sparse targets 的 max-minus-min。
        blocked: 是否至少一組超過其 threshold。
        blocking_groups: 按正式 group 順序列出的阻擋群組。
    """

    report: pd.DataFrame
    sparse_fov_count: int
    baseline_target_range: float
    blocked: bool
    blocking_groups: tuple[str, ...]

    def raise_if_blocked(self) -> None:
        """在完整 report 可被 caller 保存後，執行 sensitivity fail-closed gate。

        Raises:
            TargetSensitivityBlockedError: 任一一般組超過 5%，或 B8_P7 超過
                15% 時拋出。
        """
        if self.blocked:
            raise TargetSensitivityBlockedError(
                "sparse-FOV target sensitivity blocked groups: "
                + ", ".join(self.blocking_groups)
            )

    def to_dict(self) -> dict[str, object]:
        """回傳 CLI 可記錄的 sensitivity gate 摘要。"""
        return {
            "sparse_fov_count": self.sparse_fov_count,
            "baseline_target_range": self.baseline_target_range,
            "blocked": self.blocked,
            "blocking_groups": list(self.blocking_groups),
        }


def build_exp4_targets(
    deduplicated_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    retained_keys: Iterable[tuple[object, object]] | pd.DataFrame | None = None,
    post_border_cells: pd.DataFrame | None = None,
    expected_pre_border_cells: int = 23012,
    expected_post_border_cells: int = 19648,
    expected_fov_count: int = 693,
    expected_group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> TargetAggregationResult:
    """以去重 cells 建立 v4 cell-median targets 與描述性 FOV QC。

    呼叫端須以 ``retained_keys`` 或 ``post_border_cells`` 二擇一描述 border
    exclusion 結果；後者也只作 key roster，IDO 一律取自已驗證的 deduplicated
    master。正式 group target 直接取 group 內所有 post-border cell 的 median，
    不經過 FOV median。FOV median 僅由回傳的 ``fov_ido_scores`` 揭露。

    Args:
        deduplicated_cells: F0 後、每個 whole-cell key 唯一的 pre-border master。
        manifest: 每個 FOV 一列，含 ``image_key``、group、donor 與 passage。
        retained_keys: border exclusion 後保留的 composite keys 或 key frame。
        post_border_cells: border exclusion 後 frame；與 ``retained_keys`` 二擇一。
        expected_pre_border_cells: fail-closed pre-border distinct cell count。
        expected_post_border_cells: fail-closed post-border distinct cell count。
        expected_fov_count: fail-closed manifest/FOV count。
        expected_group_keys: fail-closed exact group ids 與輸出順序。

    Returns:
        正式 group targets、描述性 FOV scores 與兩份 trusted cell roster。

    Raises:
        TargetAggregationError: 任一 schema、key、finite、count 或 group gate 失敗。
    """
    for name, value in (
        ("expected_pre_border_cells", expected_pre_border_cells),
        ("expected_post_border_cells", expected_post_border_cells),
        ("expected_fov_count", expected_fov_count),
    ):
        _validate_expected_count(value, name)
    groups = _normalise_expected_groups(expected_group_keys)
    manifest_frame = _normalise_manifest(
        manifest,
        expected_fov_count=expected_fov_count,
        expected_group_keys=groups,
    )
    pre = _normalise_unique_cells(
        deduplicated_cells,
        name="deduplicated_cells",
        expected_count=expected_pre_border_cells,
    )
    if set(pre["image_key"]) != set(manifest_frame["image_key"]):
        raise TargetAggregationError(
            "pre-border cells 與 manifest 的 image_key 集合必須完全相同"
        )
    retained = _normalise_retained_keys(
        retained_keys=retained_keys,
        post_border_cells=post_border_cells,
    )
    if len(retained) != expected_post_border_cells:
        raise TargetAggregationError(
            "post-border distinct cell count 不符："
            f"expected {expected_post_border_cells}, actual {len(retained)}"
        )

    pre_index = pd.MultiIndex.from_frame(pre.loc[:, CELL_KEY_COLUMNS])
    retained_index = pd.MultiIndex.from_frame(retained.loc[:, CELL_KEY_COLUMNS])
    missing_keys = retained_index.difference(pre_index)
    if len(missing_keys):
        preview = list(missing_keys[:5])
        raise TargetAggregationError(
            "retained keys 不可超出 deduplicated master: " + repr(preview)
        )
    keep_mask = pre_index.isin(retained_index)
    post = pre.loc[keep_mask].reset_index(drop=True)
    if len(post) != len(retained):
        raise TargetAggregationError("retained keys 未能一對一對齊 deduplicated master")
    if post_border_cells is not None and "IDO_score" in post_border_cells.columns:
        _assert_post_border_ido_consistency(post_border_cells, post)
    if set(post["image_key"]) != set(manifest_frame["image_key"]):
        raise TargetAggregationError(
            "post-border cells 必須保留全部 manifest FOV，不可暗中丟棄 sparse FOV"
        )

    context_columns = ("image_key", "group_id", "b_id", "passage")
    cell_context = manifest_frame.loc[:, context_columns]
    pre_context = pre.loc[:, (*CELL_KEY_COLUMNS, "IDO_score")].merge(
        cell_context,
        on="image_key",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    post_context = post.loc[:, (*CELL_KEY_COLUMNS, "IDO_score")].merge(
        cell_context,
        on="image_key",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    fov_stats = (
        post_context.groupby("image_key", sort=False)["IDO_score"]
        .agg(cells_after="size", FOV_IDO_score="median")
        .reset_index()
    )
    fov_scores = manifest_frame.loc[:, context_columns].merge(
        fov_stats,
        on="image_key",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    fov_scores = fov_scores.loc[:, FOV_IDO_SCORE_COLUMNS]
    if len(fov_scores) != expected_fov_count or fov_scores.isna().any().any():
        raise TargetAggregationError("descriptive FOV_IDO_score 未完整覆蓋 manifest")
    if not np.isfinite(
        fov_scores["FOV_IDO_score"].to_numpy(dtype=float)
    ).all():
        raise TargetAggregationError("descriptive FOV_IDO_score 含非 finite 值")

    target_rows: list[dict[str, object]] = []
    for group_id in groups:
        group_manifest = manifest_frame.loc[
            manifest_frame["group_id"].eq(group_id)
        ]
        before = pre_context.loc[pre_context["group_id"].eq(group_id)]
        after = post_context.loc[post_context["group_id"].eq(group_id)]
        if group_manifest.empty or before.empty or after.empty:
            raise TargetAggregationError(f"group 缺少 FOV/cells: {group_id}")
        group_score = float(np.median(after["IDO_score"].to_numpy(dtype=float)))
        all_score = float(np.median(before["IDO_score"].to_numpy(dtype=float)))
        target_rows.append(
            {
                "group_id": group_id,
                "b_id": str(group_manifest.iloc[0]["b_id"]),
                "passage": int(group_manifest.iloc[0]["passage"]),
                "fov_count": int(len(group_manifest)),
                "cells_before": int(len(before)),
                "cells_after": int(len(after)),
                "group_IDO_score": group_score,
                "group_IDO_score_all_cells": all_score,
                "delta": group_score - all_score,
                "confidence_flag": "low" if group_id == "B8_P7" else "normal",
            }
        )
    targets = pd.DataFrame(target_rows, columns=GROUP_TARGET_COLUMNS)
    target_values = targets.loc[
        :, ("group_IDO_score", "group_IDO_score_all_cells", "delta")
    ].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise TargetAggregationError("group targets 含非 finite 值")

    return TargetAggregationResult(
        group_targets=targets,
        fov_ido_scores=fov_scores.reset_index(drop=True),
        manifest=manifest_frame.reset_index(drop=True),
        pre_border_cells=pre.reset_index(drop=True),
        post_border_cells=post,
        expected_group_keys=groups,
    )


def evaluate_sparse_fov_sensitivity(
    targets: TargetAggregationResult,
    sparse_fov_keys: Sequence[str],
    *,
    expected_sparse_fov_count: int = 10,
    expected_sparse_cell_count: int = 2,
    general_threshold_pct: float = 5.0,
    low_confidence_group: str = "B8_P7",
    low_confidence_threshold_pct: float = 15.0,
) -> SparseFovSensitivityResult:
    """比較保留與反事實移除 sparse FOV 的 v4 cell-median targets。

    正式 target 使用 ``targets.post_border_cells`` 的全部 FOV；without 版本只在
    此敏感度計算中移除指定 FOV。delta 固定為 ``without - with``，percentage
    denominator 固定為 with-sparse 九組 target 的 max-minus-min。一般組門檻
    5%，B8_P7 門檻 15%；absolute delta 等於門檻時通過。

    Args:
        targets: ``build_exp4_targets`` 產生的已驗證正式 target 結果。
        sparse_fov_keys: 觸邊後低細胞數 FOV 的 exact image keys。
        expected_sparse_fov_count: fail-closed sparse FOV 數，正式值為 10。
        expected_sparse_cell_count: sparse FOV 的 exact post-border cell 數，
            正式值為 2。
        general_threshold_pct: 一般組相對九組全距的百分比門檻。
        low_confidence_group: 使用放寬門檻的既定 low-confidence group。
        low_confidence_threshold_pct: low-confidence group 百分比門檻。

    Returns:
        完整九組 sensitivity report 與可顯式執行的 blocked gate。

    Raises:
        TargetAggregationError: sparse roster、正式 target、全距或 threshold 無效。
    """
    _validate_expected_count(
        expected_sparse_fov_count,
        "expected_sparse_fov_count",
    )
    _validate_expected_count(
        expected_sparse_cell_count,
        "expected_sparse_cell_count",
    )
    if expected_sparse_cell_count < 1:
        raise TargetAggregationError("expected_sparse_cell_count 必須是正整數")
    sparse = _normalise_sparse_fov_keys(sparse_fov_keys)
    if len(sparse) != expected_sparse_fov_count:
        raise TargetAggregationError(
            "sparse FOV count 不符："
            f"expected {expected_sparse_fov_count}, actual {len(sparse)}"
        )
    general_pct = _normalise_threshold(general_threshold_pct, "general_threshold_pct")
    low_pct = _normalise_threshold(
        low_confidence_threshold_pct,
        "low_confidence_threshold_pct",
    )
    low_group = str(low_confidence_group).strip()
    if not low_group:
        raise TargetAggregationError("low_confidence_group 不可為空")

    manifest = targets.manifest.copy().reset_index(drop=True)
    post = targets.post_border_cells.copy().reset_index(drop=True)
    fov_qc = targets.fov_ido_scores.copy().reset_index(drop=True)
    _require_columns(
        fov_qc,
        ("image_key", "cells_after"),
        "fov_ido_scores",
    )
    if fov_qc["image_key"].duplicated().any():
        raise TargetAggregationError("fov_ido_scores image_key 必須唯一")
    fov_qc["image_key"] = fov_qc["image_key"].astype(str)
    cell_counts = pd.to_numeric(fov_qc["cells_after"], errors="coerce")
    if (
        not np.isfinite(cell_counts.to_numpy(dtype=float)).all()
        or not np.equal(cell_counts, np.floor(cell_counts)).all()
        or bool((cell_counts < 1).any())
    ):
        raise TargetAggregationError("fov_ido_scores cells_after 必須是 positive finite 整數")
    inferred_sparse = tuple(
        fov_qc.loc[
            cell_counts.eq(expected_sparse_cell_count),
            "image_key",
        ]
    )
    if len(inferred_sparse) != expected_sparse_fov_count:
        raise TargetAggregationError(
            "由 fov_ido_scores 推導的 sparse FOV count 不符："
            f"expected {expected_sparse_fov_count}, actual {len(inferred_sparse)}"
        )
    if set(sparse) != set(inferred_sparse):
        missing = sorted(set(inferred_sparse) - set(sparse))
        unexpected = sorted(set(sparse) - set(inferred_sparse))
        raise TargetAggregationError(
            "sparse FOV roster 不符："
            f"missing={missing}, unexpected={unexpected}"
        )
    manifest_keys = set(manifest["image_key"])
    missing_manifest = sorted(set(sparse) - manifest_keys)
    if missing_manifest:
        raise TargetAggregationError(
            "sparse FOV 不在正式 manifest: " + ", ".join(missing_manifest)
        )
    post_fov_keys = set(post["image_key"])
    missing_post = sorted(set(sparse) - post_fov_keys)
    if missing_post:
        raise TargetAggregationError(
            "sparse FOV 必須全數保留於正式 post-border target: "
            + ", ".join(missing_post)
        )

    group_targets = targets.group_targets.copy().reset_index(drop=True)
    _require_columns(
        group_targets,
        ("group_id", "group_IDO_score"),
        "group_targets",
    )
    if group_targets["group_id"].duplicated().any():
        raise TargetAggregationError("group_targets group_id 必須唯一")
    if tuple(group_targets["group_id"].astype(str)) != targets.expected_group_keys:
        raise TargetAggregationError(
            "group_targets 順序/集合必須等於 expected_group_keys"
        )
    with_values = pd.to_numeric(
        group_targets["group_IDO_score"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(with_values).all():
        raise TargetAggregationError("with-sparse group targets 必須全為 finite")
    baseline_range = float(np.max(with_values) - np.min(with_values))
    if not np.isfinite(baseline_range) or baseline_range <= 0.0:
        raise TargetAggregationError(
            "baseline target range 必須為 finite 且大於 0，才能計算 sensitivity"
        )

    post_context = post.loc[:, (*CELL_KEY_COLUMNS, "IDO_score")].merge(
        manifest.loc[:, ("image_key", "group_id")],
        on="image_key",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    without = post_context.loc[~post_context["image_key"].isin(sparse)]
    sparse_manifest = manifest.loc[manifest["image_key"].isin(sparse)]
    with_lookup = group_targets.set_index("group_id")["group_IDO_score"]

    rows: list[dict[str, object]] = []
    blocking_groups: list[str] = []
    for group_id in targets.expected_group_keys:
        group_without = without.loc[without["group_id"].eq(group_id), "IDO_score"]
        if group_without.empty:
            raise TargetAggregationError(
                f"移除 sparse FOV 後 group 無 retained cells: {group_id}"
            )
        target_with = float(with_lookup.loc[group_id])
        target_without = float(
            np.median(group_without.to_numpy(dtype=np.float64))
        )
        delta = target_without - target_with
        absolute_delta = abs(delta)
        delta_pct = 100.0 * absolute_delta / baseline_range
        threshold_pct = low_pct if group_id == low_group else general_pct
        threshold_absolute = baseline_range * threshold_pct / 100.0
        passed = absolute_delta <= threshold_absolute
        status = "passed" if passed else "blocked"
        if not passed:
            blocking_groups.append(group_id)
        rows.append(
            {
                "group_id": group_id,
                "sparse_fov_count": int(
                    sparse_manifest["group_id"].eq(group_id).sum()
                ),
                "target_with_sparse_fovs": target_with,
                "target_without_sparse_fovs": target_without,
                "delta_without_minus_with": delta,
                "absolute_delta": absolute_delta,
                "baseline_target_range": baseline_range,
                "absolute_delta_pct_of_range": delta_pct,
                "threshold_pct": threshold_pct,
                "threshold_absolute": threshold_absolute,
                "status": status,
            }
        )
    report = pd.DataFrame(rows, columns=SENSITIVITY_REPORT_COLUMNS)
    numeric_columns = tuple(
        column
        for column in SENSITIVITY_REPORT_COLUMNS
        if column not in {"group_id", "status"}
    )
    if not np.isfinite(report.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise TargetAggregationError("sparse-FOV sensitivity report 含非 finite 值")
    return SparseFovSensitivityResult(
        report=report,
        sparse_fov_count=len(sparse),
        baseline_target_range=baseline_range,
        blocked=bool(blocking_groups),
        blocking_groups=tuple(blocking_groups),
    )


def _validate_expected_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TargetAggregationError(f"{name} 必須是非負整數")
    if int(value) < 0:
        raise TargetAggregationError(f"{name} 必須是非負整數")


def _normalise_sparse_fov_keys(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TargetAggregationError("sparse_fov_keys 必須是非字串序列")
    sparse = tuple(str(value).strip() for value in values)
    if any(not value for value in sparse):
        raise TargetAggregationError("sparse_fov_keys 不可含空值")
    if len(sparse) != len(set(sparse)):
        raise TargetAggregationError("sparse_fov_keys 不可重複")
    return sparse


def _normalise_threshold(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TargetAggregationError(f"{name} 必須是 finite 非負數") from error
    if not np.isfinite(numeric) or numeric < 0.0:
        raise TargetAggregationError(f"{name} 必須是 finite 非負數")
    return numeric


def _normalise_expected_groups(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TargetAggregationError("expected_group_keys 必須是非字串序列")
    groups = tuple(str(value).strip() for value in values)
    if not groups or any(not group for group in groups):
        raise TargetAggregationError("expected_group_keys 不可為空")
    if len(groups) != len(set(groups)):
        raise TargetAggregationError("expected_group_keys 不可重複")
    return groups


def _normalise_manifest(
    manifest: pd.DataFrame,
    *,
    expected_fov_count: int,
    expected_group_keys: tuple[str, ...],
) -> pd.DataFrame:
    required = ("image_key", "group_id", "b_id", "passage")
    _require_columns(manifest, required, "manifest")
    frame = manifest.loc[:, required].copy().reset_index(drop=True)
    if len(frame) != expected_fov_count:
        raise TargetAggregationError(
            "manifest FOV count 不符："
            f"expected {expected_fov_count}, actual {len(frame)}"
        )
    for column in ("image_key", "group_id", "b_id"):
        if frame[column].isna().any():
            raise TargetAggregationError(f"manifest {column} 不可為空")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise TargetAggregationError(f"manifest {column} 不可為空字串")
    if frame["image_key"].duplicated().any():
        raise TargetAggregationError("manifest image_key 必須一 FOV 一列且唯一")
    passage = pd.to_numeric(frame["passage"], errors="coerce")
    if (
        not np.isfinite(passage.to_numpy(dtype=float)).all()
        or not np.equal(passage, np.floor(passage)).all()
    ):
        raise TargetAggregationError("manifest passage 必須是 finite 整數")
    frame["passage"] = passage.astype(int)
    canonical = frame["b_id"] + "_P" + frame["passage"].astype(str)
    if not canonical.equals(frame["group_id"]):
        raise TargetAggregationError("manifest group_id 必須等於 b_id + '_P' + passage")
    actual_groups = set(frame["group_id"])
    if actual_groups != set(expected_group_keys):
        raise TargetAggregationError(
            "manifest group ids 不符："
            f"expected {list(expected_group_keys)}, actual {sorted(actual_groups)}"
        )
    return frame


def _normalise_unique_cells(
    cells: pd.DataFrame,
    *,
    name: str,
    expected_count: int,
) -> pd.DataFrame:
    required = (*CELL_KEY_COLUMNS, "IDO_score")
    _require_columns(cells, required, name)
    frame = cells.copy().reset_index(drop=True)
    _normalise_key_columns(frame, name)
    if frame.duplicated(list(CELL_KEY_COLUMNS)).any():
        raise TargetAggregationError(f"{name} 必須已去重為 unique whole-cell keys")
    if len(frame) != expected_count:
        raise TargetAggregationError(
            f"{name} distinct cell count 不符："
            f"expected {expected_count}, actual {len(frame)}"
        )
    frame["IDO_score"] = pd.to_numeric(frame["IDO_score"], errors="coerce")
    if not np.isfinite(frame["IDO_score"].to_numpy(dtype=float)).all():
        raise TargetAggregationError(f"{name} IDO_score 必須全為 finite")
    return frame


def _normalise_retained_keys(
    *,
    retained_keys: Iterable[tuple[object, object]] | pd.DataFrame | None,
    post_border_cells: pd.DataFrame | None,
) -> pd.DataFrame:
    if (retained_keys is None) == (post_border_cells is None):
        raise TargetAggregationError(
            "retained_keys 與 post_border_cells 必須恰好提供一個"
        )
    source: object
    name: str
    if post_border_cells is not None:
        source = post_border_cells
        name = "post_border_cells"
    else:
        source = retained_keys
        name = "retained_keys"
    if isinstance(source, pd.DataFrame):
        _require_columns(source, CELL_KEY_COLUMNS, name)
        frame = source.loc[:, CELL_KEY_COLUMNS].copy().reset_index(drop=True)
    else:
        if isinstance(source, (str, bytes)):
            raise TargetAggregationError("retained_keys 必須是 composite key iterable")
        try:
            values = list(source)  # type: ignore[arg-type]
        except TypeError as error:
            raise TargetAggregationError(
                "retained_keys 必須是 composite key iterable"
            ) from error
        if any(
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 2
            for value in values
        ):
            raise TargetAggregationError(
                "retained_keys 每筆必須是 (image_key, cell_label)"
            )
        frame = pd.DataFrame(values, columns=CELL_KEY_COLUMNS)
    _normalise_key_columns(frame, name)
    if frame.duplicated(list(CELL_KEY_COLUMNS)).any():
        raise TargetAggregationError(f"{name} 不可含重複 whole-cell keys")
    return frame


def _assert_post_border_ido_consistency(
    provided: pd.DataFrame,
    trusted: pd.DataFrame,
) -> None:
    _require_columns(
        provided,
        (*CELL_KEY_COLUMNS, "IDO_score"),
        "post_border_cells",
    )
    actual = provided.loc[:, (*CELL_KEY_COLUMNS, "IDO_score")].copy()
    actual = actual.reset_index(drop=True)
    _normalise_key_columns(actual, "post_border_cells")
    actual["IDO_score"] = pd.to_numeric(actual["IDO_score"], errors="coerce")
    if not np.isfinite(actual["IDO_score"].to_numpy(dtype=float)).all():
        raise TargetAggregationError("post_border_cells IDO_score 必須全為 finite")
    expected = trusted.loc[:, (*CELL_KEY_COLUMNS, "IDO_score")]
    comparison = actual.merge(
        expected,
        on=list(CELL_KEY_COLUMNS),
        how="inner",
        suffixes=("_provided", "_master"),
        validate="one_to_one",
        sort=False,
    )
    if len(comparison) != len(expected):
        raise TargetAggregationError(
            "post_border_cells IDO roster 必須等於 retained master roster"
        )
    provided_values = comparison["IDO_score_provided"].to_numpy(dtype=float)
    master_values = comparison["IDO_score_master"].to_numpy(dtype=float)
    if not np.array_equal(provided_values, master_values, equal_nan=False):
        raise TargetAggregationError(
            "post_border_cells IDO_score 必須與 deduplicated master 完全相同"
        )


def _normalise_key_columns(frame: pd.DataFrame, name: str) -> None:
    if frame["image_key"].isna().any() or frame["cell_label"].isna().any():
        raise TargetAggregationError(f"{name} whole-cell keys 不可為空")
    frame["image_key"] = frame["image_key"].astype(str).str.strip()
    if frame["image_key"].eq("").any():
        raise TargetAggregationError(f"{name} image_key 不可為空字串")
    labels = pd.to_numeric(frame["cell_label"], errors="coerce")
    if (
        not np.isfinite(labels.to_numpy(dtype=float)).all()
        or not np.equal(labels, np.floor(labels)).all()
        or bool((labels <= 0).any())
    ):
        raise TargetAggregationError(f"{name} cell_label 必須是 positive finite 整數")
    frame["cell_label"] = labels.astype(int)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TargetAggregationError(f"{name} 缺少欄位: " + ", ".join(missing))
    duplicates = [
        column for column in columns if list(frame.columns).count(column) != 1
    ]
    if duplicates:
        raise TargetAggregationError(
            f"{name} 欄名必須唯一: " + ", ".join(dict.fromkeys(duplicates))
        )
