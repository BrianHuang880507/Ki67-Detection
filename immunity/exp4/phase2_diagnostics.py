"""Exp4 Phase 2 reversed-dose diagnostics 與 post-CV immutability gates。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .phase2 import (
    PHASE1_ORDER_CONFIGURATION_ID,
    PHASE2_CELL_COUNT,
    PHASE2_FOV_COUNT,
    PHASE2_GROUP_COUNT,
    PHASE2_TARGET_COLUMNS,
    Phase2ContractError,
    validate_phase2_oof,
)


DOSE_MAPPING_NOTE = "劑量對應為未經確認之反轉假設，待取像端確認"
DOSE_RESPONSE_COLUMNS = (
    "donor_passage_id",
    "b_id",
    "passage",
    "ifn_0_tnf_0_target_median",
    "ifn_25_tnf_0_target_median",
    "ifn_50_tnf_0_target_median",
    "ifn_100_tnf_0_target_median",
    "monotonic_non_decreasing",
    "dose_mapping_note",
)
RESIDUAL_VS_DOSE_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "n_groups",
    "mean_residual_vs_ifn_rho",
    "mean_residual_vs_ifn_p_value",
    "mean_residual_vs_tnf_rho",
    "mean_residual_vs_tnf_p_value",
    "abs_mean_residual_vs_ifn_rho",
    "abs_mean_residual_vs_ifn_p_value",
    "abs_mean_residual_vs_tnf_rho",
    "abs_mean_residual_vs_tnf_p_value",
    "dose_mapping_note",
)
PHASE2_NUMERIC_ARTIFACT_NAMES = (
    "metrics",
    "oof_predictions",
    "hyperparameters",
    "feature_importance",
    "per_group_residuals",
)

_ARM_NAMES = (
    "geometry_24",
    "rui_48",
    "rui_filtered",
    "rui_48_plus_nucleus",
)
_MODEL_NAMES = ("SVR_L", "SVR", "LASSO", "RFR", "GBR", "MLPR")
_CONFIGURATION_IDS = tuple(
    f"{arm}__{model}" for arm in _ARM_NAMES for model in _MODEL_NAMES
)
_GROUP_IDS = frozenset(
    f"{b_id}_P{passage}_C{condition_index:02d}"
    for b_id in ("B4", "B7", "B8")
    for passage in (5, 6, 7)
    for condition_index in range(1, 9)
)
_DONOR_PASSAGE_ORDER = tuple(
    (b_id, passage)
    for b_id in ("B4", "B7", "B8")
    for passage in (5, 6, 7)
)


class Phase2ArtifactMutationError(Phase2ContractError):
    """表示 dose diagnostics 前後的 post-CV numeric artifacts 已改變。"""


@dataclass(frozen=True)
class DoseCondition:
    """保存 reversed-hypothesis condition 的明示 label 與 doses。"""

    label: str
    ifn_dose: int
    tnf_dose: int


CONDITION_MAPPING_REVERSED_HYPOTHESIS: Mapping[int, DoseCondition] = (
    MappingProxyType(
        {
            1: DoseCondition("IFN100_TNF0", 100, 0),
            2: DoseCondition("IFN50_TNF0", 50, 0),
            3: DoseCondition("IFN25_TNF0", 25, 0),
            4: DoseCondition("IFN0_TNF0", 0, 0),
            5: DoseCondition("IFN0_TNF25", 0, 25),
            6: DoseCondition("IFN0_TNF50", 0, 50),
            7: DoseCondition("IFN25_TNF25", 25, 25),
            8: DoseCondition("IFN25_TNF50", 25, 50),
        }
    )
)


@dataclass(frozen=True)
class Phase2NumericArtifactFingerprints:
    """保存五個 frozen post-CV numeric artifacts 的 exact SHA-256。"""

    artifact_sha256: tuple[tuple[str, str], ...]

    def as_mapping(self) -> dict[str, str]:
        """依正式 artifact roster 回傳一份 hash mapping。"""
        return dict(self.artifact_sha256)


@dataclass(frozen=True)
class Phase2ArtifactImmutabilityResult:
    """保存 dose diagnostics 前後五個 artifacts 的 exact 比對結果。"""

    passed: bool
    before_sha256: Mapping[str, str]
    after_sha256: Mapping[str, str]


def build_dose_response_check(
    phase2_targets: pd.DataFrame,
    *,
    condition_mapping: Mapping[int, DoseCondition],
) -> pd.DataFrame:
    """以 explicit reversed hypothesis 記錄九組 TNF=0 dose response。

    Args:
        phase2_targets: ``build_phase2_targets`` 的正式 72-row table。
        condition_mapping: 明示 reversed-dose hypothesis；不影響 CV。

    Returns:
        九個 donor×passage 的 IFN 0/25/50/100 target medians 與 monotonic flag。

    Raises:
        TypeError: targets 或 mapping 型別不合法時拋出。
        Phase2ContractError: target schema 或 mapping 不符合正式 contract。
    """
    targets = _validated_phase2_target_table(phase2_targets)
    mapping = _validated_dose_mapping(condition_mapping)
    tnf_zero_by_ifn = {
        dose.ifn_dose: condition_index
        for condition_index, dose in mapping.items()
        if dose.tnf_dose == 0
    }
    if set(tnf_zero_by_ifn) != {0, 25, 50, 100}:
        raise Phase2ContractError("TNF=0 mapping 必須恰含 IFN 0/25/50/100")
    rows: list[dict[str, object]] = []
    for b_id, passage in _DONOR_PASSAGE_ORDER:
        group = targets.loc[
            targets["b_id"].eq(b_id) & targets["passage"].eq(passage)
        ].set_index("condition_index")
        values = [
            float(group.at[tnf_zero_by_ifn[dose], "group_IDO_score"])
            for dose in (0, 25, 50, 100)
        ]
        rows.append(
            {
                "donor_passage_id": f"{b_id}_P{passage}",
                "b_id": b_id,
                "passage": passage,
                "ifn_0_tnf_0_target_median": values[0],
                "ifn_25_tnf_0_target_median": values[1],
                "ifn_50_tnf_0_target_median": values[2],
                "ifn_100_tnf_0_target_median": values[3],
                "monotonic_non_decreasing": bool(
                    np.greater_equal(np.diff(values), 0.0).all()
                ),
                "dose_mapping_note": DOSE_MAPPING_NOTE,
            }
        )
    return pd.DataFrame(rows, columns=DOSE_RESPONSE_COLUMNS)


def analyze_residual_vs_dose(
    oof_predictions: pd.DataFrame,
    *,
    condition_mapping: Mapping[int, DoseCondition],
) -> pd.DataFrame:
    """直接由正式 OOF pool 72 個 group residual means 並對 doses 做 Spearman。

    Args:
        oof_predictions: 完整 24×19,648 Phase 2 OOF table。
        condition_mapping: 明示 reversed-dose hypothesis；只在 post-hoc 使用。

    Returns:
        24-row residual/absolute-residual 對 IFN/TNF 的 rho 與 p-values。

    Raises:
        TypeError: OOF 或 mapping 型別不合法時拋出。
        Phase2ContractError: configuration cell roster、cell→group mapping 或
            formal completeness 不成立時拋出。
    """
    oof = validate_phase2_oof(oof_predictions)
    _validate_constant_cell_group_mapping(oof)
    mapping = _validated_dose_mapping(condition_mapping)
    grouped = oof.groupby(
        ["configuration_id", "arm", "model", "group_id"],
        sort=False,
        observed=True,
    ).agg(
        n_cells=("cell_label", "size"),
        mean_residual=("residual", "mean"),
    ).reset_index()
    if len(grouped) != len(_CONFIGURATION_IDS) * PHASE2_GROUP_COUNT:
        raise Phase2ContractError("OOF residual pooling 必須恰為 24×72 means")
    grouped["condition_index"] = _condition_indices_from_group_ids(
        grouped["group_id"]
    )
    grouped["ifn_dose"] = grouped["condition_index"].map(
        {index: dose.ifn_dose for index, dose in mapping.items()}
    )
    grouped["tnf_dose"] = grouped["condition_index"].map(
        {index: dose.tnf_dose for index, dose in mapping.items()}
    )
    rows: list[dict[str, object]] = []
    for configuration_id in _CONFIGURATION_IDS:
        arm, model = configuration_id.split("__", maxsplit=1)
        configuration = grouped.loc[
            grouped["configuration_id"].eq(configuration_id)
        ]
        if (
            len(configuration) != PHASE2_GROUP_COUNT
            or set(configuration["group_id"].astype(str)) != _GROUP_IDS
            or int(configuration["n_cells"].sum()) != PHASE2_CELL_COUNT
        ):
            raise Phase2ContractError(
                f"{configuration_id} residual pooling 必須恰含 19,648 cells／72 groups"
            )
        mean_residual = configuration["mean_residual"].to_numpy(np.float64)
        absolute = np.abs(mean_residual)
        ifn = configuration["ifn_dose"].to_numpy(np.float64)
        tnf = configuration["tnf_dose"].to_numpy(np.float64)
        mean_ifn = _finite_spearman(mean_residual, ifn, configuration_id)
        mean_tnf = _finite_spearman(mean_residual, tnf, configuration_id)
        absolute_ifn = _finite_spearman(absolute, ifn, configuration_id)
        absolute_tnf = _finite_spearman(absolute, tnf, configuration_id)
        rows.append(
            {
                "configuration_id": configuration_id,
                "arm": arm,
                "model": model,
                "n_groups": PHASE2_GROUP_COUNT,
                "mean_residual_vs_ifn_rho": mean_ifn[0],
                "mean_residual_vs_ifn_p_value": mean_ifn[1],
                "mean_residual_vs_tnf_rho": mean_tnf[0],
                "mean_residual_vs_tnf_p_value": mean_tnf[1],
                "abs_mean_residual_vs_ifn_rho": absolute_ifn[0],
                "abs_mean_residual_vs_ifn_p_value": absolute_ifn[1],
                "abs_mean_residual_vs_tnf_rho": absolute_tnf[0],
                "abs_mean_residual_vs_tnf_p_value": absolute_tnf[1],
                "dose_mapping_note": DOSE_MAPPING_NOTE,
            }
        )
    return pd.DataFrame(rows, columns=RESIDUAL_VS_DOSE_COLUMNS)


def fingerprint_phase2_numeric_artifacts(
    artifact_paths: Mapping[str, str | Path],
) -> Phase2NumericArtifactFingerprints:
    """計算五個 Phase 2 post-CV numeric artifact 檔案的 exact SHA-256。

    Args:
        artifact_paths: 以正式 artifact 名稱對應實際檔案路徑的 mapping。

    Returns:
        依正式 roster 排列的 immutable fingerprints。

    Raises:
        TypeError: ``artifact_paths`` 不是 mapping 時拋出。
        Phase2ContractError: roster 不精確或任一檔案不存在時拋出。
    """
    if not isinstance(artifact_paths, Mapping):
        raise TypeError("artifact_paths 必須是 Mapping")
    if set(artifact_paths) != set(PHASE2_NUMERIC_ARTIFACT_NAMES):
        raise Phase2ContractError(
            "Phase 2 numeric artifact roster mismatch: "
            f"expected={PHASE2_NUMERIC_ARTIFACT_NAMES}, actual={tuple(artifact_paths)}"
        )
    fingerprints: list[tuple[str, str]] = []
    for name in PHASE2_NUMERIC_ARTIFACT_NAMES:
        path = Path(artifact_paths[name])
        if not path.is_file():
            raise Phase2ContractError(f"Phase 2 numeric artifact 不存在：{name}={path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprints.append((name, digest.hexdigest()))
    return Phase2NumericArtifactFingerprints(tuple(fingerprints))


def verify_phase2_numeric_artifact_immutability(
    before: Phase2NumericArtifactFingerprints,
    after: Phase2NumericArtifactFingerprints,
) -> Phase2ArtifactImmutabilityResult:
    """比較所有 dose diagnostics 前後五個 artifact 的 exact SHA-256。

    Args:
        before: 執行任何 dose diagnostics 前取得的 fingerprints。
        after: 所有 dose diagnostics 完成後重新取得的 fingerprints。

    Returns:
        五個 hashes 完全一致的 machine-readable result。

    Raises:
        TypeError: before/after 不是正式 fingerprint 型別時拋出。
        Phase2ArtifactMutationError: 任一 artifact hash 改變時 fail closed。
    """
    if not isinstance(before, Phase2NumericArtifactFingerprints):
        raise TypeError("before 必須是 Phase2NumericArtifactFingerprints")
    if not isinstance(after, Phase2NumericArtifactFingerprints):
        raise TypeError("after 必須是 Phase2NumericArtifactFingerprints")
    before_hashes = before.as_mapping()
    after_hashes = after.as_mapping()
    if tuple(before_hashes) != PHASE2_NUMERIC_ARTIFACT_NAMES or tuple(
        after_hashes
    ) != PHASE2_NUMERIC_ARTIFACT_NAMES:
        raise Phase2ContractError("Phase 2 numeric artifact fingerprint roster mismatch")
    changed = [
        name
        for name in PHASE2_NUMERIC_ARTIFACT_NAMES
        if before_hashes[name] != after_hashes[name]
    ]
    if changed:
        raise Phase2ArtifactMutationError(
            "dose diagnostics 修改了 frozen Phase 2 numeric artifacts: "
            + repr(changed)
        )
    return Phase2ArtifactImmutabilityResult(
        passed=True,
        before_sha256=MappingProxyType(before_hashes),
        after_sha256=MappingProxyType(after_hashes),
    )


def _validated_dose_mapping(
    condition_mapping: Mapping[int, DoseCondition],
) -> dict[int, DoseCondition]:
    if not isinstance(condition_mapping, Mapping):
        raise TypeError("condition_mapping 必須是 Mapping")
    actual = dict(condition_mapping)
    expected = dict(CONDITION_MAPPING_REVERSED_HYPOTHESIS)
    if actual != expected:
        raise Phase2ContractError(
            "condition_mapping 必須是 exact reversed-dose hypothesis"
        )
    return actual


def _validated_phase2_target_table(targets: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(targets, pd.DataFrame):
        raise TypeError("phase2_targets 必須是 pandas DataFrame")
    if tuple(targets.columns) != PHASE2_TARGET_COLUMNS:
        raise Phase2ContractError("phase2_targets schema mismatch")
    if len(targets) != PHASE2_GROUP_COUNT or targets["group_id"].duplicated().any():
        raise Phase2ContractError("phase2_targets 必須恰含 72 unique groups")
    expected_group_ids = (
        targets["b_id"].astype(str)
        + "_P"
        + targets["passage"].astype(str)
        + "_C"
        + targets["condition_index"].astype(int).astype(str).str.zfill(2)
    )
    if not targets["group_id"].astype(str).eq(expected_group_ids).all():
        raise Phase2ContractError("phase2_targets group_id metadata mismatch")
    if set(targets["group_id"].astype(str)) != _GROUP_IDS:
        raise Phase2ContractError("phase2_targets formal group roster mismatch")
    try:
        numeric = targets.loc[
            :,
            [
                "passage",
                "condition_index",
                "fov_count",
                "cells_after",
                "group_IDO_score",
            ],
        ].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError("phase2_targets numeric values 型別不合法") from error
    if not np.isfinite(numeric).all():
        raise Phase2ContractError("phase2_targets numeric values 必須 finite")
    if targets["cells_after"].sum() != PHASE2_CELL_COUNT:
        raise Phase2ContractError("phase2_targets cells_after 必須總和 19,648")
    if targets["fov_count"].sum() != PHASE2_FOV_COUNT:
        raise Phase2ContractError("phase2_targets fov_count 必須總和 693")
    return targets


def _validate_constant_cell_group_mapping(oof: pd.DataFrame) -> None:
    key_columns = ["image_key", "cell_label"]
    reference = oof.loc[
        oof["configuration_id"].eq(PHASE1_ORDER_CONFIGURATION_ID),
        [*key_columns, "group_id"],
    ]
    if len(reference) != PHASE2_CELL_COUNT:
        raise Phase2ContractError("authoritative OOF cell roster 必須恰為 19,648")
    reference_map = reference.set_index(key_columns)["group_id"].astype(str)
    reference_index = reference_map.index
    for configuration_id in _CONFIGURATION_IDS:
        configuration = oof.loc[
            oof["configuration_id"].eq(configuration_id),
            [*key_columns, "group_id"],
        ]
        actual_map = configuration.set_index(key_columns)["group_id"].astype(str)
        if (
            len(actual_map) != PHASE2_CELL_COUNT
            or len(reference_index.difference(actual_map.index))
            or len(actual_map.index.difference(reference_index))
        ):
            raise Phase2ContractError(
                f"{configuration_id} 未共享 exact 19,648 cell roster"
            )
        aligned = actual_map.reindex(reference_index)
        if not aligned.equals(reference_map):
            raise Phase2ContractError(
                f"{configuration_id} cell→group mapping 與 authoritative OOF 不一致"
            )


def _condition_indices_from_group_ids(group_ids: pd.Series) -> pd.Series:
    parsed = group_ids.astype(str).str.extract(
        r"^B[478]_P[567]_C(?P<condition_index>0[1-8])$"
    )["condition_index"]
    if parsed.isna().any():
        raise Phase2ContractError("Phase 2 group_id 必須符合 Bx_Px_C01..C08")
    return parsed.astype(int).set_axis(group_ids.index)


def _finite_spearman(
    first: np.ndarray,
    second: np.ndarray,
    configuration_id: str,
) -> tuple[float, float]:
    result = spearmanr(first, second)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite([rho, p_value]).all():
        raise Phase2ContractError(
            f"{configuration_id} residual-vs-dose Spearman 必須 finite"
        )
    return rho, p_value


__all__ = [
    "CONDITION_MAPPING_REVERSED_HYPOTHESIS",
    "DOSE_MAPPING_NOTE",
    "DOSE_RESPONSE_COLUMNS",
    "PHASE2_NUMERIC_ARTIFACT_NAMES",
    "RESIDUAL_VS_DOSE_COLUMNS",
    "DoseCondition",
    "Phase2ArtifactImmutabilityResult",
    "Phase2ArtifactMutationError",
    "Phase2NumericArtifactFingerprints",
    "analyze_residual_vs_dose",
    "build_dose_response_check",
    "fingerprint_phase2_numeric_artifacts",
    "verify_phase2_numeric_artifact_immutability",
]
