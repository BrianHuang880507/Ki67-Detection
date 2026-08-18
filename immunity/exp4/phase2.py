"""Exp4 Phase 2 frozen population、formal CV 與唯讀 post-hoc 核心。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from .cell_dedup import NUCLEUS_FEATURE_COLUMNS
from .cv import (
    CvPopulationContract,
    CvRunResult,
    FeatureArmAssembly,
    LeakageContext,
    LeakagePreflightError,
    LeakagePreflightResult,
    ModelSpec,
    assemble_feature_arms,
    run_rui_cv,
    run_leakage_preflight,
)
from .rui_features import RUI49_FEATURE_COLUMNS


PHASE2_CELL_COUNT = 19_648
PHASE2_FOV_COUNT = 693
PHASE2_GROUP_COUNT = 72
PHASE2_GROUP_KEY_FIELDS = ("b_id", "passage", "condition_index")
PHASE2_MANIFEST_COLUMNS = ("image_key", *PHASE2_GROUP_KEY_FIELDS)
PHASE2_TARGET_COLUMNS = (
    "group_id",
    "b_id",
    "passage",
    "condition_index",
    "fov_count",
    "cells_after",
    "group_IDO_score",
    "confidence_flag",
)
PHASE2_SHRINKAGE_GROUP_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "group_id",
    "n_cells",
    "n_fovs",
    "observed_target",
    "mean_prediction",
)
PHASE2_SHRINKAGE_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "n_cells",
    "n_fovs",
    "n_groups",
    "slope",
    "intercept",
    "regression_r_squared",
    "regression_r",
    "regression_p_value",
    "slope_std_error",
    "observed_min",
    "observed_max",
    "observed_range",
    "predicted_min",
    "predicted_max",
    "predicted_range",
    "predicted_to_observed_range_ratio",
)
METRICS_WITHIN_CONDITION_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "condition_index",
    "n_cells",
    "n_groups",
    "cell_r2",
    "cell_mae",
    "cell_spearman",
    "cell_spearman_p_value",
    "cell_spearman_status",
)
PHASE1_VS_PHASE2_COMPARISON_COLUMNS = (
    "configuration_id",
    "arm",
    "model",
    "phase1_mean_fold_cell_r2",
    "phase2_mean_fold_cell_r2",
    "phase1_mean_fold_rui_r2",
    "phase2_mean_fold_rui_r2",
    "phase1_shrinkage_slope",
    "phase2_shrinkage_slope",
)
PHASE1_RETAINED_SEQUENCE_SHA256 = (
    "9726b2712c777ab83a77378e792cbffc8b57545d53430cac547958a85e762a74"
)
PHASE1_ORDER_CONFIGURATION_ID = "geometry_24__SVR_L"
PHASE1_DEDUP_PARSED_NUMERIC_SHA256 = (
    "65d4de276265b7390c1b541f957b3b3d64bce9eb2b0cae4c4afbe584723b42b7"
)
PHASE1_PINNED_ARTIFACT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "data_manifest.csv": (
            "b4508aaf830f4c456759d1f02458606aa28ed385856575c6fb99a5ac2e163771"
        ),
        "cell_level_basic.csv": (
            "7d999a728d848ccb61a3910cb5230c9efde6d7e1fa14f22de792990ee4ad7850"
        ),
        "cell_dedup_report.csv": (
            "3ff03e66faca2df2cd3ba7caa33b60f7924f9ad328e87219ec47459f0f87f209"
        ),
        "cell_level_rui49.csv": (
            "5c6197cc002d406daaa0b4ddf60824d1f37fc0411f8e11f765df8db186e25966"
        ),
        "feature_health_report.csv": (
            "969bcd073846c635d665d6bf38a78bc2ea5d63531085bc46887c1b27b572fc5b"
        ),
        "feature_redundancy_report.csv": (
            "2534decda02fc123ba75bd109f683aa44d1a31a4426d7a53bc25a6c5bc94d1c4"
        ),
        "oof_predictions.csv": (
            "69947b3ba3d5da5b5e7e58648db58bc21849b5cf9ab602269519daae9fddeb2a"
        ),
    }
)

_PHASE1_ARM_NAMES = (
    "geometry_24",
    "rui_48",
    "rui_filtered",
    "rui_48_plus_nucleus",
)
_PHASE1_MODEL_NAMES = ("SVR_L", "SVR", "LASSO", "RFR", "GBR", "MLPR")
_PHASE1_CONFIGURATION_IDS = frozenset(
    f"{arm}__{model}"
    for arm in _PHASE1_ARM_NAMES
    for model in _PHASE1_MODEL_NAMES
)
_IDENTITY_COLUMNS = ("image_key", "cell_label")
_PHASE2_GROUP_IDS = tuple(
    f"{b_id}_P{passage}_C{condition_index:02d}"
    for b_id in ("B4", "B7", "B8")
    for passage in (5, 6, 7)
    for condition_index in range(1, 9)
)
_IMAGE_KEY_PATTERN = re.compile(
    r"^(?P<b_id>B[478])_P(?P<passage>[567])_C(?P<condition_index>0[1-8])_"
)


class Phase2ContractError(ValueError):
    """表示 Phase 2 frozen-input、target、CV 或 post-hoc contract 不成立。"""


class Phase2LeakageError(Phase2ContractError):
    """表示 Phase 2 八項 leakage preflight 至少一項失敗。"""

    def __init__(self, message: str, report: pd.DataFrame) -> None:
        super().__init__(message)
        self.report = report.copy()


@dataclass(frozen=True)
class Phase2PopulationContract:
    """不可降級的 Phase 2 formal population contract。"""

    scope: str
    expected_cell_count: int
    expected_fov_count: int
    expected_group_ids: tuple[str, ...]

    @classmethod
    def formal(cls) -> "Phase2PopulationContract":
        """建立唯一允許的 19,648-cell／693-FOV／72-group contract。"""
        return cls(
            scope="formal_phase2_19648_cells_693_fovs_72_groups",
            expected_cell_count=PHASE2_CELL_COUNT,
            expected_fov_count=PHASE2_FOV_COUNT,
            expected_group_ids=_PHASE2_GROUP_IDS,
        )

    def as_cv_contract(self) -> CvPopulationContract:
        """轉成既有 CV module 可執行的 formal Phase 2 contract。"""
        return CvPopulationContract.formal_exp4_phase2()


@dataclass(frozen=True)
class Phase2LeakageContext:
    """保存 Phase 1 七項 provenance；第八項只使用實際資料證據。"""

    phase1: LeakageContext

    @classmethod
    def formal(cls, phase1: LeakageContext) -> "Phase2LeakageContext":
        """建立 Phase 2 leakage context。"""
        if not isinstance(phase1, LeakageContext):
            raise TypeError("phase1 必須是 LeakageContext")
        return cls(phase1=phase1)


@dataclass(frozen=True)
class Phase2CvAssembly:
    """保存 Phase 2 targets、四 arms、八項 preflight 與 mapping-free fingerprints。"""

    population: "Phase2FrozenPopulation"
    targets: pd.DataFrame
    feature_assembly: FeatureArmAssembly
    contract: Phase2PopulationContract
    leakage: LeakagePreflightResult
    cv_input_fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class Phase2CvRunResult:
    """保存既有 CV engine 的 frozen 結果與 Phase 2 assembly provenance。"""

    core: CvRunResult
    assembly: Phase2CvAssembly
    checkpoint_dir: Path

    @property
    def metrics(self) -> pd.DataFrame:
        """回傳 120-row fold metrics。"""
        return self.core.metrics

    @property
    def oof_predictions(self) -> pd.DataFrame:
        """回傳 471,552-row OOF predictions。"""
        return self.core.oof_predictions

    @property
    def hyperparameters(self) -> pd.DataFrame:
        """回傳 120-row hyperparameter table。"""
        return self.core.hyperparameters

    @property
    def per_group_residuals(self) -> pd.DataFrame:
        """回傳 24×5×72 residual pieces。"""
        return self.core.per_group_residuals

    @property
    def feature_importance(self) -> pd.DataFrame:
        """回傳完整 fold-level feature importance。"""
        return self.core.feature_importance

    @property
    def leakage(self) -> LeakagePreflightResult:
        """回傳 Phase 2 八項 leakage evidence。"""
        return self.assembly.leakage


@dataclass(frozen=True)
class Phase2ShrinkageResult:
    """保存 24-row Phase 2 shrinkage report 與 1,728-row group audit。"""

    report: pd.DataFrame
    group_summary: pd.DataFrame


@dataclass(frozen=True)
class Phase1ConsistencyEvidence:
    """保存 Phase 1 已完成的 Rui49 exact-consistency 證據。"""

    scope: str
    checked_row_count: int
    checked_cell_count: int
    checked_duplicate_key_count: int
    feature_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Phase1ConsistencyEvidence":
        """從 Phase 1 metadata mapping 建立 immutable evidence。

        Args:
            value: ``pre_border_consistency`` 或 ``retained_consistency`` mapping。

        Returns:
            正規化後的 consistency evidence。

        Raises:
            Phase2ContractError: 欄位遺漏或型別不合法時拋出。
        """
        required = {
            "scope",
            "checked_row_count",
            "checked_cell_count",
            "checked_duplicate_key_count",
            "feature_count",
        }
        missing = sorted(required - set(value))
        if missing:
            raise Phase2ContractError(f"Rui49 consistency evidence 缺欄：{missing}")
        try:
            return cls(
                scope=str(value["scope"]),
                checked_row_count=int(value["checked_row_count"]),
                checked_cell_count=int(value["checked_cell_count"]),
                checked_duplicate_key_count=int(
                    value["checked_duplicate_key_count"]
                ),
                feature_count=int(value["feature_count"]),
            )
        except (TypeError, ValueError) as error:
            raise Phase2ContractError(
                "Rui49 consistency evidence counts 必須為整數"
            ) from error


@dataclass(frozen=True)
class FrozenPhase1Evidence:
    """鎖定七個 Phase 1 inputs 與兩段 Rui49 consistency evidence。"""

    artifact_sha256: tuple[tuple[str, str], ...]
    pre_border_consistency: Phase1ConsistencyEvidence
    retained_consistency: Phase1ConsistencyEvidence

    @classmethod
    def from_observed(
        cls,
        *,
        artifact_sha256: Mapping[str, str],
        pre_border_consistency: Mapping[str, object],
        retained_consistency: Mapping[str, object],
    ) -> "FrozenPhase1Evidence":
        """將 adapter 量測到的 hashes 與 metadata 轉成 immutable evidence。

        Args:
            artifact_sha256: 七個 formal input 檔案的實測 SHA-256。
            pre_border_consistency: Phase 1 full pre-border consistency mapping。
            retained_consistency: Phase 1 retained consistency mapping。

        Returns:
            保留 canonical artifact-name order 的 frozen evidence。

        Raises:
            Phase2ContractError: artifact roster 或 hash 格式不合法時拋出。
        """
        if set(artifact_sha256) != set(PHASE1_PINNED_ARTIFACT_SHA256):
            raise Phase2ContractError(
                "Phase 1 artifact fingerprint roster mismatch: "
                f"expected={tuple(PHASE1_PINNED_ARTIFACT_SHA256)}, "
                f"actual={tuple(artifact_sha256)}"
            )
        normalized: list[tuple[str, str]] = []
        for name in PHASE1_PINNED_ARTIFACT_SHA256:
            fingerprint = str(artifact_sha256[name]).strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                raise Phase2ContractError(f"{name} fingerprint 不是 SHA-256")
            normalized.append((name, fingerprint))
        return cls(
            artifact_sha256=tuple(normalized),
            pre_border_consistency=Phase1ConsistencyEvidence.from_mapping(
                pre_border_consistency
            ),
            retained_consistency=Phase1ConsistencyEvidence.from_mapping(
                retained_consistency
            ),
        )

    def as_mapping(self) -> dict[str, str]:
        """依 pinned order 回傳 artifact hashes 的新 mapping。"""
        return dict(self.artifact_sha256)


@dataclass(frozen=True)
class Phase2FrozenPopulation:
    """保存只由 frozen Phase 1 artifacts 重建的正式 Phase 2 population。

    Attributes:
        cells: 依 authoritative OOF key sequence 排列的 19,648 cells。
        manifest: 僅保留四欄 allowlist 的 693-row manifest。
        healthy_features: 僅由 Phase 1 health report 讀出的 48 欄 roster。
        filtered_features: 僅由 Phase 1 redundancy report 讀出的 30 欄 roster。
        retained_sequence_sha256: authoritative cell-key sequence fingerprint。
        frozen_source_fingerprint: 不含 dose mapping 的 Phase 1 source fingerprint。
        evidence: caller 實測且已通過 pinned gates 的 immutable evidence。
    """

    cells: pd.DataFrame
    manifest: pd.DataFrame
    healthy_features: tuple[str, ...]
    filtered_features: tuple[str, ...]
    retained_sequence_sha256: str
    frozen_source_fingerprint: str
    evidence: FrozenPhase1Evidence


def reconstruct_phase2_population(
    *,
    phase1_oof: pd.DataFrame,
    cell_level_basic: pd.DataFrame,
    cell_dedup_report: pd.DataFrame,
    cell_level_rui49: pd.DataFrame,
    feature_health_report: pd.DataFrame,
    feature_redundancy_report: pd.DataFrame,
    manifest: pd.DataFrame,
    evidence: FrozenPhase1Evidence,
) -> Phase2FrozenPopulation:
    """從明示 Phase 1 DataFrames 重建 frozen 19,648-cell population。

    此函式不執行 feature extraction、dedup aggregation、border scanning、
    feature health 或 Pearson selection。Multirow IDO/nucleus 值只取自既有
    dedup report；whole-cell features 在 pinned consistency 通過後只取首列。

    Args:
        phase1_oof: Phase 1 24 configurations 的 canonical OOF table。
        cell_level_basic: Phase 1 23,976-row raw basic feature table。
        cell_dedup_report: Phase 1 已聚合的 938-key dedup report。
        cell_level_rui49: Phase 1 frozen 23,976-row Rui49 table。
        feature_health_report: Phase 1 frozen health report。
        feature_redundancy_report: Phase 1 frozen redundancy report。
        manifest: 只含四欄 allowlist 的 manifest。
        evidence: adapter 實測的 pinned hashes 與 Rui49 consistency evidence。

    Returns:
        可供 target builder 與 formal CV assembly 使用的 frozen population。

    Raises:
        TypeError: DataFrame 或 evidence 型別不合法時拋出。
        Phase2ContractError: 任一 pinned/schema/count/order/reuse gate 失敗。
    """
    _validate_frozen_evidence(evidence)
    labels = _validated_manifest(manifest)
    retained_keys, sequence_fingerprint = _authoritative_retained_keys(phase1_oof)
    basic_values = _reconstruct_basic_values(
        cell_level_basic,
        cell_dedup_report,
    )
    healthy_features = _healthy_roster(feature_health_report)
    filtered_features = _filtered_roster(
        feature_redundancy_report,
        healthy_features=healthy_features,
    )
    whole_cell_values = _first_rui49_values(
        cell_level_rui49,
        cell_level_basic=cell_level_basic,
    )

    ordered = retained_keys.assign(_retained_order=np.arange(len(retained_keys)))
    ordered = ordered.merge(
        basic_values,
        on=list(_IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    ordered = ordered.merge(
        labels,
        on="image_key",
        how="left",
        validate="many_to_one",
    )
    ordered = ordered.merge(
        whole_cell_values,
        on=list(_IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    ).sort_values("_retained_order", kind="stable")
    required_values = [
        "IDO_score",
        *NUCLEUS_FEATURE_COLUMNS,
        *PHASE2_GROUP_KEY_FIELDS,
        *RUI49_FEATURE_COLUMNS,
    ]
    if ordered.loc[:, required_values].isna().any().any():
        raise Phase2ContractError("frozen Phase 1 joins 未完整覆蓋 retained OOF keys")
    if len(ordered) != PHASE2_CELL_COUNT or ordered["image_key"].nunique() != PHASE2_FOV_COUNT:
        raise Phase2ContractError("reconstructed population 必須恰為 19,648 cells／693 FOV")
    numeric_columns = ["IDO_score", *NUCLEUS_FEATURE_COLUMNS, *RUI49_FEATURE_COLUMNS]
    try:
        numeric = ordered.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError("reconstructed frozen values 必須是 numeric") from error
    if not np.isfinite(numeric).all():
        raise Phase2ContractError("reconstructed frozen values 必須全為 finite")

    columns = (
        *_IDENTITY_COLUMNS,
        *PHASE2_GROUP_KEY_FIELDS,
        "IDO_score",
        *NUCLEUS_FEATURE_COLUMNS,
        *RUI49_FEATURE_COLUMNS,
    )
    cells = ordered.loc[:, columns].reset_index(drop=True)
    source_payload = {
        "artifact_sha256": evidence.artifact_sha256,
        "pre_border_consistency": evidence.pre_border_consistency.__dict__,
        "retained_consistency": evidence.retained_consistency.__dict__,
        "retained_sequence_sha256": sequence_fingerprint,
        "healthy_features": healthy_features,
        "filtered_features": filtered_features,
        "manifest_columns": PHASE2_MANIFEST_COLUMNS,
    }
    frozen_source_fingerprint = hashlib.sha256(
        json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return Phase2FrozenPopulation(
        cells=cells,
        manifest=labels.reset_index(drop=True),
        healthy_features=healthy_features,
        filtered_features=filtered_features,
        retained_sequence_sha256=sequence_fingerprint,
        frozen_source_fingerprint=frozen_source_fingerprint,
        evidence=evidence,
    )


def assemble_phase2_cv(
    population: Phase2FrozenPopulation,
    leakage_context: Phase2LeakageContext,
) -> Phase2CvAssembly:
    """在 frozen population 上建立正式 72-group 四-arm CV assembly。

    Args:
        population: ``reconstruct_phase2_population`` 的正式結果。
        leakage_context: Phase 1 七項與 Phase 2 check 8 provenance。

    Returns:
        含 targets、formal CV adapter、八項 report 與五個 input fingerprints。

    Raises:
        TypeError: 輸入型別不合法時拋出。
        Phase2ContractError: population、target 或 formal contract 不成立。
        Phase2LeakageError: 八項 leakage check 任一失敗時拋出。
    """
    if not isinstance(population, Phase2FrozenPopulation):
        raise TypeError("population 必須是 Phase2FrozenPopulation")
    if not isinstance(leakage_context, Phase2LeakageContext):
        raise TypeError("leakage_context 必須是 Phase2LeakageContext")
    contract = Phase2PopulationContract.formal()
    targets = build_phase2_targets(population.cells, population.manifest)
    target_values = targets.loc[
        :,
        [
            *PHASE2_GROUP_KEY_FIELDS,
            "group_id",
            "group_IDO_score",
            "confidence_flag",
        ],
    ]
    modeling_cells = population.cells.merge(
        target_values,
        on=list(PHASE2_GROUP_KEY_FIELDS),
        how="left",
        validate="many_to_one",
    )
    if modeling_cells.loc[
        :, ["group_id", "group_IDO_score", "confidence_flag"]
    ].isna().any().any():
        raise Phase2ContractError("Phase 2 targets 未完整覆蓋 reconstructed cells")
    feature_assembly = assemble_feature_arms(
        modeling_cells,
        filtered_features=population.filtered_features,
        contract=contract.as_cv_contract(),
    )
    _validate_modeling_cells_against_targets(
        feature_assembly.cells,
        targets,
    )
    leakage = run_phase2_leakage_preflight(
        feature_assembly,
        leakage_context,
        manifest=population.manifest,
    )
    fingerprints = _phase2_cv_input_fingerprints(
        feature_assembly,
        frozen_source_fingerprint=population.frozen_source_fingerprint,
        targets=targets,
        manifest=population.manifest,
    )
    return Phase2CvAssembly(
        population=population,
        targets=targets,
        feature_assembly=feature_assembly,
        contract=contract,
        leakage=leakage,
        cv_input_fingerprints=MappingProxyType(fingerprints),
    )


def run_phase2_cv(
    assembly: Phase2CvAssembly,
    leakage_context: Phase2LeakageContext,
    *,
    checkpoint_dir: str | Path,
    phase1_checkpoint_dir: str | Path,
    model_specs: Sequence[ModelSpec] | None = None,
    n_jobs: int = -1,
    importance_repeats: int = 3,
) -> Phase2CvRunResult:
    """以既有 nested-CV engine 執行 Phase 2，並隔離 Phase 1 checkpoints。

    Args:
        assembly: 已通過 8-row leakage preflight 的 formal Phase 2 assembly。
        leakage_context: 與 assembly 相同的 formal leakage provenance。
        checkpoint_dir: Phase 2 專用 checkpoint directory。
        phase1_checkpoint_dir: Phase 1 checkpoint directory，只用於 distinct gate。
        model_specs: 省略時沿用既有 ``default_model_specs`` 六模型 grids。
        n_jobs: 傳給既有 GridSearchCV runner 的 job 數。
        importance_repeats: permutation importance 重複次數。

    Returns:
        完整且 frozen 的 Phase 2 CV result；不寫 canonical CSV。

    Raises:
        TypeError: assembly/context 型別不合法時拋出。
        Phase2ContractError: checkpoint 目錄未與 Phase 1 分離時拋出。
        Phase2LeakageError: 執行前重新驗證的八項 report 失敗時拋出。
    """
    if not isinstance(assembly, Phase2CvAssembly):
        raise TypeError("assembly 必須是 Phase2CvAssembly")
    if not isinstance(leakage_context, Phase2LeakageContext):
        raise TypeError("leakage_context 必須是 Phase2LeakageContext")
    _validate_modeling_cells_against_targets(
        assembly.feature_assembly.cells,
        assembly.targets,
    )
    phase2_path = Path(checkpoint_dir)
    phase1_path = Path(phase1_checkpoint_dir)
    resolved_phase2 = phase2_path.resolve(strict=False)
    resolved_phase1 = phase1_path.resolve(strict=False)
    if (
        resolved_phase2 == resolved_phase1
        or resolved_phase2 in resolved_phase1.parents
        or resolved_phase1 in resolved_phase2.parents
    ):
        raise Phase2ContractError(
            "Phase 2 與 Phase 1 checkpoint directory trees 必須 disjoint"
        )
    run_phase2_leakage_preflight(
        assembly.feature_assembly,
        leakage_context,
        manifest=assembly.population.manifest,
    )
    current_fingerprints = _phase2_cv_input_fingerprints(
        assembly.feature_assembly,
        frozen_source_fingerprint=assembly.population.frozen_source_fingerprint,
        targets=assembly.targets,
        manifest=assembly.population.manifest,
    )
    if current_fingerprints != dict(assembly.cv_input_fingerprints):
        raise Phase2ContractError("Phase 2 assembly 在 frozen CV 前已被修改")
    core = run_rui_cv(
        assembly.feature_assembly,
        leakage_context.phase1,
        checkpoint_dir=phase2_path,
        source_fingerprint=assembly.cv_input_fingerprints[
            "cv_source_fingerprint"
        ],
        model_specs=model_specs,
        n_jobs=n_jobs,
        importance_repeats=importance_repeats,
    )
    return Phase2CvRunResult(
        core=core,
        assembly=assembly,
        checkpoint_dir=phase2_path,
    )


def analyze_phase2_shrinkage(
    oof_predictions: pd.DataFrame,
) -> Phase2ShrinkageResult:
    """以 pooled OOF 的 72 個無權重 group means 計算 shrinkage。

    Args:
        oof_predictions: 完整 24 configurations × 19,648 cells OOF table。

    Returns:
        24-row regression report 與 24×72 group summary。

    Raises:
        TypeError: 輸入不是 DataFrame 時拋出。
        Phase2ContractError: formal OOF、group target 或 regression 不合法。
    """
    oof = validate_phase2_oof(oof_predictions)
    target_counts = oof.groupby(
        ["configuration_id", "group_id"], sort=False, observed=True
    )["observed_ido_score"].nunique(dropna=False)
    if not target_counts.eq(1).all():
        raise Phase2ContractError("Phase 2 OOF target 必須在 configuration/group 內恆定")
    grouped = oof.groupby(
        ["configuration_id", "arm", "model", "group_id"],
        sort=False,
        observed=True,
    ).agg(
        n_cells=("cell_label", "size"),
        n_fovs=("image_key", "nunique"),
        observed_target=("observed_ido_score", "first"),
        mean_prediction=("predicted_ido_score", "mean"),
    ).reset_index()
    group_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    for arm in _PHASE1_ARM_NAMES:
        for model in _PHASE1_MODEL_NAMES:
            configuration_id = f"{arm}__{model}"
            configuration = grouped.loc[
                grouped["configuration_id"].eq(configuration_id)
            ].set_index("group_id")
            if set(configuration.index) != set(_PHASE2_GROUP_IDS):
                raise Phase2ContractError(
                    f"{configuration_id} shrinkage 必須完整含 72 groups"
                )
            configuration = configuration.loc[list(_PHASE2_GROUP_IDS)]
            for group_id, row in configuration.iterrows():
                group_rows.append(
                    {
                        "configuration_id": configuration_id,
                        "arm": arm,
                        "model": model,
                        "group_id": group_id,
                        "n_cells": int(row["n_cells"]),
                        "n_fovs": int(row["n_fovs"]),
                        "observed_target": float(row["observed_target"]),
                        "mean_prediction": float(row["mean_prediction"]),
                    }
                )
            observed = configuration["observed_target"].to_numpy(dtype=np.float64)
            predicted = configuration["mean_prediction"].to_numpy(dtype=np.float64)
            observed_min = float(np.min(observed))
            observed_max = float(np.max(observed))
            observed_range = observed_max - observed_min
            if observed_range <= 0.0:
                raise Phase2ContractError(
                    f"{configuration_id} shrinkage observed range 必須大於 0"
                )
            regression = linregress(observed, predicted)
            regression_values = np.asarray(
                [
                    regression.slope,
                    regression.intercept,
                    regression.rvalue,
                    regression.pvalue,
                    regression.stderr,
                ],
                dtype=np.float64,
            )
            if not np.isfinite(regression_values).all():
                raise Phase2ContractError(
                    f"{configuration_id} shrinkage regression 必須全為 finite"
                )
            predicted_min = float(np.min(predicted))
            predicted_max = float(np.max(predicted))
            predicted_range = predicted_max - predicted_min
            report_rows.append(
                {
                    "configuration_id": configuration_id,
                    "arm": arm,
                    "model": model,
                    "n_cells": int(configuration["n_cells"].sum()),
                    "n_fovs": int(
                        oof.loc[
                            oof["configuration_id"].eq(configuration_id),
                            "image_key",
                        ].nunique()
                    ),
                    "n_groups": PHASE2_GROUP_COUNT,
                    "slope": float(regression.slope),
                    "intercept": float(regression.intercept),
                    "regression_r_squared": float(regression.rvalue**2),
                    "regression_r": float(regression.rvalue),
                    "regression_p_value": float(regression.pvalue),
                    "slope_std_error": float(regression.stderr),
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "observed_range": observed_range,
                    "predicted_min": predicted_min,
                    "predicted_max": predicted_max,
                    "predicted_range": predicted_range,
                    "predicted_to_observed_range_ratio": (
                        predicted_range / observed_range
                    ),
                }
            )
    report = pd.DataFrame(report_rows, columns=PHASE2_SHRINKAGE_COLUMNS)
    group_summary = pd.DataFrame(
        group_rows, columns=PHASE2_SHRINKAGE_GROUP_COLUMNS
    )
    if len(report) != 24 or len(group_summary) != 24 * PHASE2_GROUP_COUNT:
        raise Phase2ContractError("Phase 2 shrinkage output row counts 不完整")
    return Phase2ShrinkageResult(report=report, group_summary=group_summary)


def build_metrics_within_condition(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """由 OOF-only predictions 計算 24 configurations × 8 opaque conditions。

    Args:
        oof_predictions: 完整 formal Phase 2 OOF table。

    Returns:
        192-row cell R²、MAE、Spearman、p-value 與 undefined status table。

    Raises:
        TypeError: 輸入不是 DataFrame 時拋出。
        Phase2ContractError: OOF completeness 或 numeric contract 不成立。
    """
    oof = validate_phase2_oof(oof_predictions)
    conditions = _condition_indices_from_group_ids(oof["group_id"])
    rows: list[dict[str, object]] = []
    for arm in _PHASE1_ARM_NAMES:
        for model in _PHASE1_MODEL_NAMES:
            configuration_id = f"{arm}__{model}"
            configuration_mask = oof["configuration_id"].eq(configuration_id)
            for condition_index in range(1, 9):
                subset = oof.loc[configuration_mask & conditions.eq(condition_index)]
                if subset["group_id"].nunique() != 9 or subset.empty:
                    raise Phase2ContractError(
                        f"{configuration_id}/C{condition_index:02d} 必須含 9 groups"
                    )
                observed = subset["observed_ido_score"].to_numpy(np.float64)
                predicted = subset["predicted_ido_score"].to_numpy(np.float64)
                cell_r2 = float(r2_score(observed, predicted))
                cell_mae = float(mean_absolute_error(observed, predicted))
                if np.unique(observed).size < 2 or np.unique(predicted).size < 2:
                    rho = float("nan")
                    p_value = float("nan")
                    status = "undefined_constant_input"
                else:
                    correlation = spearmanr(observed, predicted)
                    rho = float(correlation.statistic)
                    p_value = float(correlation.pvalue)
                    if not np.isfinite([rho, p_value]).all():
                        raise Phase2ContractError(
                            f"{configuration_id}/C{condition_index:02d} Spearman 非 finite"
                        )
                    status = "defined"
                if not np.isfinite([cell_r2, cell_mae]).all():
                    raise Phase2ContractError("within-condition R²/MAE 必須 finite")
                rows.append(
                    {
                        "configuration_id": configuration_id,
                        "arm": arm,
                        "model": model,
                        "condition_index": condition_index,
                        "n_cells": int(len(subset)),
                        "n_groups": 9,
                        "cell_r2": cell_r2,
                        "cell_mae": cell_mae,
                        "cell_spearman": rho,
                        "cell_spearman_p_value": p_value,
                        "cell_spearman_status": status,
                    }
                )
    report = pd.DataFrame(rows, columns=METRICS_WITHIN_CONDITION_COLUMNS)
    if len(report) != 192:
        raise Phase2ContractError("metrics_within_condition 必須恰為 192 rows")
    return report


def compare_phase1_phase2(
    *,
    phase1_metrics: pd.DataFrame,
    phase1_shrinkage: pd.DataFrame,
    phase2_metrics: pd.DataFrame,
    phase2_shrinkage: pd.DataFrame,
) -> pd.DataFrame:
    """依 exact arm/model 比較兩 phase 的 fold R² means 與 shrinkage slope。

    Args:
        phase1_metrics: Phase 1 120-row fold metrics。
        phase1_shrinkage: Phase 1 24-row shrinkage report。
        phase2_metrics: Phase 2 120-row fold metrics。
        phase2_shrinkage: Phase 2 24-row shrinkage report。

    Returns:
        24-row Phase 1-vs-Phase 2 comparison。

    Raises:
        TypeError: 任一輸入不是 DataFrame 時拋出。
        Phase2ContractError: configuration/fold/group roster 不完整時拋出。
    """
    phase1_summary = _metrics_summary(phase1_metrics, label="Phase 1")
    phase2_summary = _metrics_summary(phase2_metrics, label="Phase 2")
    phase1_slopes = _shrinkage_slopes(
        phase1_shrinkage, label="Phase 1", expected_group_count=9
    )
    phase2_slopes = _shrinkage_slopes(
        phase2_shrinkage, label="Phase 2", expected_group_count=72
    )
    merged = phase1_summary.merge(
        phase2_summary,
        on=["configuration_id", "arm", "model"],
        how="inner",
        validate="one_to_one",
        suffixes=("_phase1", "_phase2"),
    ).merge(
        phase1_slopes,
        on=["configuration_id", "arm", "model"],
        how="inner",
        validate="one_to_one",
    ).merge(
        phase2_slopes,
        on=["configuration_id", "arm", "model"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    indexed = merged.set_index("configuration_id")
    for arm in _PHASE1_ARM_NAMES:
        for model in _PHASE1_MODEL_NAMES:
            configuration_id = f"{arm}__{model}"
            if configuration_id not in indexed.index:
                raise Phase2ContractError("comparison 缺 formal configuration")
            row = indexed.loc[configuration_id]
            rows.append(
                {
                    "configuration_id": configuration_id,
                    "arm": arm,
                    "model": model,
                    "phase1_mean_fold_cell_r2": float(row["mean_fold_cell_r2_phase1"]),
                    "phase2_mean_fold_cell_r2": float(row["mean_fold_cell_r2_phase2"]),
                    "phase1_mean_fold_rui_r2": float(row["mean_fold_rui_r2_phase1"]),
                    "phase2_mean_fold_rui_r2": float(row["mean_fold_rui_r2_phase2"]),
                    "phase1_shrinkage_slope": float(row["phase1_shrinkage_slope"]),
                    "phase2_shrinkage_slope": float(row["phase2_shrinkage_slope"]),
                }
            )
    return pd.DataFrame(rows, columns=PHASE1_VS_PHASE2_COMPARISON_COLUMNS)


def validate_phase2_oof(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """驗證正式 Phase 2 OOF 的 schema、數量與 fold/group 完整性。

    Args:
        oof_predictions: 預期為 24×19,648 列的 Phase 2 OOF table。

    Returns:
        通過所有 fail-closed gates 的原始 DataFrame。

    Raises:
        TypeError: 輸入不是 pandas DataFrame 時拋出。
        Phase2ContractError: 任一正式 OOF contract 不成立時拋出。
    """
    if not isinstance(oof_predictions, pd.DataFrame):
        raise TypeError("oof_predictions 必須是 pandas DataFrame")
    required = {
        "configuration_id",
        "arm",
        "model",
        "outer_fold",
        "image_key",
        "cell_label",
        "group_id",
        "confidence_flag",
        "observed_ido_score",
        "predicted_ido_score",
        "residual",
    }
    missing = sorted(required - set(oof_predictions.columns))
    if missing:
        raise Phase2ContractError(f"Phase 2 OOF 缺少欄位：{missing}")
    if len(oof_predictions) != PHASE2_CELL_COUNT * 24:
        raise Phase2ContractError("Phase 2 OOF 必須恰為 471,552 rows")
    categorical = oof_predictions.loc[
        :, ["configuration_id", "arm", "model", "image_key", "group_id"]
    ]
    if categorical.isna().any().any():
        raise Phase2ContractError("Phase 2 OOF identity 不可為空")
    normalized = categorical.astype(str)
    if normalized.apply(lambda column: column.str.strip().eq("")).any().any():
        raise Phase2ContractError("Phase 2 OOF identity 不可為空字串")
    derived = normalized["arm"] + "__" + normalized["model"]
    if not normalized["configuration_id"].eq(derived).all():
        raise Phase2ContractError("OOF configuration_id 必須等於 arm__model")
    if set(normalized["configuration_id"]) != _PHASE1_CONFIGURATION_IDS:
        raise Phase2ContractError("Phase 2 OOF 必須恰含 4 arms × 6 models")
    identity = oof_predictions.loc[
        :, ["configuration_id", "image_key", "cell_label"]
    ]
    if identity.isna().any().any() or identity.duplicated().any():
        raise Phase2ContractError("Phase 2 OOF configuration × cell 必須唯一")
    try:
        numeric = oof_predictions.loc[
            :, ["observed_ido_score", "predicted_ido_score", "residual"]
        ].to_numpy(dtype=np.float64)
        folds = pd.to_numeric(
            oof_predictions["outer_fold"], errors="raise"
        ).to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError("Phase 2 OOF numeric 欄位型別不合法") from error
    if not np.isfinite(numeric).all() or not set(folds).issubset({1, 2, 3, 4, 5}):
        raise Phase2ContractError("Phase 2 OOF numeric/fold values 不合法")
    if not np.allclose(
        numeric[:, 1] - numeric[:, 0], numeric[:, 2], rtol=0.0, atol=1e-12
    ):
        raise Phase2ContractError("Phase 2 OOF residual 必須等於 predicted-observed")
    condition_indices = _condition_indices_from_group_ids(
        oof_predictions["group_id"]
    )
    if not condition_indices.between(1, 8).all():
        raise Phase2ContractError("Phase 2 OOF condition index 必須為 1-8")
    if set(oof_predictions["group_id"].astype(str)) != set(_PHASE2_GROUP_IDS):
        raise Phase2ContractError("Phase 2 OOF 必須完整含 72 formal groups")
    if oof_predictions["image_key"].nunique() != PHASE2_FOV_COUNT:
        raise Phase2ContractError("Phase 2 OOF 必須恰含 693 FOV")
    group_target_counts = oof_predictions.groupby(
        "group_id", sort=False, observed=True
    )["observed_ido_score"].nunique(dropna=False)
    if not group_target_counts.eq(1).all() or len(group_target_counts) != 72:
        raise Phase2ContractError("Phase 2 OOF 每個 group target 必須恆定")
    confidence_counts = oof_predictions.groupby(
        "group_id", sort=False, observed=True
    )["confidence_flag"].nunique(dropna=False)
    if not confidence_counts.eq(1).all() or not set(
        oof_predictions["confidence_flag"].astype(str)
    ).issubset({"normal", "low"}):
        raise Phase2ContractError("Phase 2 OOF confidence 必須 per-group 恆定")
    expected_groups = set(_PHASE2_GROUP_IDS)
    for configuration_id, configuration in oof_predictions.groupby(
        "configuration_id", sort=False, observed=True
    ):
        if len(configuration) != PHASE2_CELL_COUNT:
            raise Phase2ContractError(
                f"{configuration_id} OOF 必須覆蓋 19,648 cells"
            )
        if set(configuration["outer_fold"].astype(int)) != {1, 2, 3, 4, 5}:
            raise Phase2ContractError(f"{configuration_id} OOF 必須含 folds 1-5")
        fold_groups = configuration.groupby("outer_fold", sort=False)["group_id"].agg(
            lambda values: set(values.astype(str))
        )
        if not all(groups == expected_groups for groups in fold_groups):
            raise Phase2ContractError(
                f"{configuration_id} 每個 outer fold 必須含完整 72 groups"
            )
    return oof_predictions


def _condition_indices_from_group_ids(group_ids: pd.Series) -> pd.Series:
    parsed = group_ids.astype(str).str.extract(
        r"^B[478]_P[567]_C(?P<condition_index>0[1-8])$"
    )["condition_index"]
    if parsed.isna().any():
        raise Phase2ContractError("Phase 2 group_id 必須符合 Bx_Px_C01..C08")
    return parsed.astype(int).set_axis(group_ids.index)


def _metrics_summary(metrics: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError(f"{label} metrics 必須是 pandas DataFrame")
    required = {
        "configuration_id",
        "arm",
        "model",
        "outer_fold",
        "cell_r2",
        "rui_r2",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise Phase2ContractError(f"{label} metrics 缺少欄位：{missing}")
    if len(metrics) != 120 or metrics.duplicated(
        ["configuration_id", "outer_fold"]
    ).any():
        raise Phase2ContractError(f"{label} metrics 必須恰含 24×5 unique rows")
    derived = metrics["arm"].astype(str) + "__" + metrics["model"].astype(str)
    if not metrics["configuration_id"].astype(str).eq(derived).all() or set(
        derived
    ) != _PHASE1_CONFIGURATION_IDS:
        raise Phase2ContractError(f"{label} metrics configuration roster mismatch")
    numeric = metrics.loc[:, ["cell_r2", "rui_r2"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise Phase2ContractError(f"{label} fold R² 必須 finite")
    fold_sets = metrics.groupby("configuration_id", sort=False)["outer_fold"].agg(
        lambda values: set(values.astype(int))
    )
    if not all(folds == {1, 2, 3, 4, 5} for folds in fold_sets):
        raise Phase2ContractError(f"{label} metrics 每個 configuration 必須含 folds 1-5")
    return metrics.groupby(
        ["configuration_id", "arm", "model"], sort=False, observed=True
    ).agg(
        mean_fold_cell_r2=("cell_r2", "mean"),
        mean_fold_rui_r2=("rui_r2", "mean"),
    ).reset_index()


def _shrinkage_slopes(
    shrinkage: pd.DataFrame,
    *,
    label: str,
    expected_group_count: int,
) -> pd.DataFrame:
    if not isinstance(shrinkage, pd.DataFrame):
        raise TypeError(f"{label} shrinkage 必須是 pandas DataFrame")
    required = {"configuration_id", "arm", "model", "n_groups", "slope"}
    missing = sorted(required - set(shrinkage.columns))
    if missing:
        raise Phase2ContractError(f"{label} shrinkage 缺少欄位：{missing}")
    if len(shrinkage) != 24 or shrinkage["configuration_id"].duplicated().any():
        raise Phase2ContractError(f"{label} shrinkage 必須恰含 24 unique rows")
    derived = shrinkage["arm"].astype(str) + "__" + shrinkage["model"].astype(str)
    if not shrinkage["configuration_id"].astype(str).eq(derived).all() or set(
        derived
    ) != _PHASE1_CONFIGURATION_IDS:
        raise Phase2ContractError(f"{label} shrinkage configuration roster mismatch")
    if not pd.to_numeric(shrinkage["n_groups"], errors="coerce").eq(
        expected_group_count
    ).all():
        raise Phase2ContractError(
            f"{label} shrinkage n_groups 必須為 {expected_group_count}"
        )
    slopes = pd.to_numeric(shrinkage["slope"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(slopes).all():
        raise Phase2ContractError(f"{label} shrinkage slope 必須 finite")
    column = "phase1_shrinkage_slope" if label == "Phase 1" else "phase2_shrinkage_slope"
    result = shrinkage.loc[:, ["configuration_id", "arm", "model"]].copy()
    result[column] = slopes
    return result


def run_phase2_leakage_preflight(
    feature_assembly: FeatureArmAssembly,
    context: Phase2LeakageContext,
    *,
    manifest: pd.DataFrame,
) -> LeakagePreflightResult:
    """執行 Phase 1-equivalent 七項與實證式 opaque-key check 8。

    Args:
        feature_assembly: 正式 72-group feature-arm assembly。
        context: Phase 1 七項 leakage provenance。
        manifest: 實際載入的四欄 Phase 2 manifest projection。

    Returns:
        恰含八列且全部 ``passed`` 的 machine-readable report。

    Raises:
        TypeError: context 型別不合法時拋出。
        Phase2LeakageError: 任一項失敗時附完整八列 report 拋出。
    """
    if not isinstance(context, Phase2LeakageContext):
        raise TypeError("context 必須是 Phase2LeakageContext")
    try:
        phase1 = run_leakage_preflight(feature_assembly, context.phase1)
        base_report = phase1.report.copy()
    except LeakagePreflightError as error:
        base_report = error.report.copy()

    arms = feature_assembly.as_mapping()
    predictors = tuple(
        feature for features in arms.values() for feature in features
    )
    lower_predictors = tuple(feature.lower() for feature in predictors)
    forbidden_exact = {
        "condition",
        "ifn_dose",
        "tnf_dose",
        "group_id",
        *(field.lower() for field in PHASE2_GROUP_KEY_FIELDS),
    }
    forbidden_predictors = sorted(
        {
            feature
            for feature, lower in zip(predictors, lower_predictors, strict=True)
            if lower in forbidden_exact or "dose" in lower
        }
    )
    manifest_allowlist_matches = (
        isinstance(manifest, pd.DataFrame)
        and tuple(manifest.columns) == PHASE2_MANIFEST_COLUMNS
    )
    group_key_tuple_matches = (
        manifest_allowlist_matches
        and tuple(manifest.columns[1:]) == PHASE2_GROUP_KEY_FIELDS
    )
    filename_condition_indices_match = False
    within_image_group_fields_unique = False
    all_cell_rows_match_manifest = False
    manifest_cell_labels_match = False
    manifest_error = ""
    try:
        loaded_manifest = _validated_manifest(manifest)
        if tuple(loaded_manifest.columns[1:]) != PHASE2_GROUP_KEY_FIELDS:
            raise Phase2ContractError("manifest group-key tuple mismatch")
        filename_condition_indices_match = True
        cell_metadata = feature_assembly.cells.loc[
            :, PHASE2_MANIFEST_COLUMNS
        ].copy()
        per_image_unique = cell_metadata.groupby(
            "image_key", sort=False, observed=True
        )[list(PHASE2_GROUP_KEY_FIELDS)].nunique(dropna=False)
        within_image_group_fields_unique = bool(
            per_image_unique.eq(1).all(axis=None)
        )
        comparison = cell_metadata.merge(
            loaded_manifest,
            on="image_key",
            how="left",
            validate="many_to_one",
            suffixes=("_cell", "_manifest"),
        )
        row_matches = np.ones(len(comparison), dtype=bool)
        for field in PHASE2_GROUP_KEY_FIELDS:
            row_matches &= comparison[f"{field}_cell"].eq(
                comparison[f"{field}_manifest"]
            ).to_numpy(dtype=bool)
        all_cell_rows_match_manifest = bool(
            len(comparison) == len(cell_metadata) and row_matches.all()
        )
        manifest_cell_labels_match = bool(
            within_image_group_fields_unique and all_cell_rows_match_manifest
        )
        if not manifest_cell_labels_match:
            raise Phase2ContractError(
                "每列 cell metadata 必須與 canonical manifest 一致，且每個 image "
                "的 group fields 必須唯一"
            )
    except (KeyError, TypeError, ValueError, Phase2ContractError) as error:
        manifest_error = str(error)
    check_eight_passed = (
        manifest_allowlist_matches
        and group_key_tuple_matches
        and filename_condition_indices_match
        and within_image_group_fields_unique
        and all_cell_rows_match_manifest
        and manifest_cell_labels_match
        and not forbidden_predictors
    )
    details = json.dumps(
        {
            "group_key_fields": list(PHASE2_GROUP_KEY_FIELDS),
            "manifest_columns": (
                list(manifest.columns) if isinstance(manifest, pd.DataFrame) else []
            ),
            "manifest_allowlist_matches": manifest_allowlist_matches,
            "group_key_tuple_matches": group_key_tuple_matches,
            "filename_condition_indices_match": filename_condition_indices_match,
            "within_image_group_fields_unique": within_image_group_fields_unique,
            "all_cell_rows_match_manifest": all_cell_rows_match_manifest,
            "manifest_cell_labels_match": manifest_cell_labels_match,
            "manifest_error": manifest_error,
            "forbidden_predictors": forbidden_predictors,
            "group_id_excluded_from_x": "group_id" not in predictors,
            "condition_index_excluded_from_x": "condition_index" not in predictors,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    check_eight = pd.DataFrame(
        [
            {
                "check_id": 8,
                "check_name": "phase2_opaque_keys_manifest_and_x_isolation",
                "status": "passed" if check_eight_passed else "failed",
                "details": details,
            }
        ],
        columns=base_report.columns,
    )
    report = pd.concat([base_report, check_eight], ignore_index=True)
    if len(report) != 8 or report["check_id"].tolist() != list(range(1, 9)):
        raise Phase2ContractError("Phase 2 leakage report schema 必須恰為 checks 1-8")
    if report["status"].ne("passed").any():
        failed = report.loc[report["status"].eq("failed"), "check_name"].tolist()
        raise Phase2LeakageError(
            f"Phase 2 leakage preflight failed closed: {failed}",
            report,
        )
    return LeakagePreflightResult(passed=True, report=report)


def _phase2_cv_input_fingerprints(
    feature_assembly: FeatureArmAssembly,
    *,
    frozen_source_fingerprint: str,
    targets: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, str]:
    arms = feature_assembly.as_mapping()
    feature_order = tuple(
        dict.fromkeys(feature for features in arms.values() for feature in features)
    )
    cells = feature_assembly.cells.reset_index(drop=True)
    x_hashes = pd.util.hash_pandas_object(
        cells.loc[:, feature_order],
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64)
    x_digest = hashlib.sha256()
    x_digest.update(x_hashes.tobytes(order="C"))
    x_digest.update(
        json.dumps(
            {name: list(features) for name, features in arms.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    y = pd.to_numeric(cells["group_IDO_score"], errors="raise").to_numpy(
        dtype=np.float64
    )
    y_fingerprint = hashlib.sha256(y.tobytes(order="C")).hexdigest()
    group_hashes = pd.util.hash_pandas_object(
        cells.loc[:, ["group_id", "confidence_flag"]],
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64)
    group_ids_fingerprint = hashlib.sha256(
        group_hashes.tobytes(order="C")
    ).hexdigest()
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = tuple(
        (
            np.asarray(train, dtype=np.int64),
            np.asarray(test, dtype=np.int64),
        )
        for train, test in splitter.split(np.arange(len(cells)))
    )
    expected_groups = set(_PHASE2_GROUP_IDS)
    for fold_index, (_, test) in enumerate(splits, start=1):
        actual = set(cells.iloc[test]["group_id"].astype(str))
        if actual != expected_groups:
            raise Phase2ContractError(
                f"Phase 2 outer fold {fold_index} 未完整包含 72 groups"
            )
    split_digest = hashlib.sha256()
    for train, test in splits:
        split_digest.update(train.tobytes(order="C"))
        split_digest.update(b"|")
        split_digest.update(test.tobytes(order="C"))
        split_digest.update(b";")
    split_fingerprint = split_digest.hexdigest()
    target_hashes = pd.util.hash_pandas_object(
        targets.loc[:, PHASE2_TARGET_COLUMNS],
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64)
    manifest_projection = _validated_manifest(manifest)
    manifest_hashes = pd.util.hash_pandas_object(
        manifest_projection.loc[:, PHASE2_MANIFEST_COLUMNS],
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64)
    manifest_projection_fingerprint = hashlib.sha256(
        manifest_hashes.tobytes(order="C")
    ).hexdigest()
    source_components = dict(phase2_cv_source_components())
    if set(source_components) != {"phase2.py", "cv.py"} or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in source_components.values()
    ):
        raise Phase2ContractError("Phase 2 CV source component hashes 不合法")
    cell_target_metadata_digest = hashlib.sha256()
    cell_target_metadata_digest.update(y.tobytes(order="C"))
    cell_target_metadata_digest.update(group_hashes.tobytes(order="C"))
    source_payload = {
        "frozen_source_fingerprint": frozen_source_fingerprint,
        "target_fingerprint": hashlib.sha256(
            target_hashes.tobytes(order="C")
        ).hexdigest(),
        "manifest_projection_fingerprint": manifest_projection_fingerprint,
        "cell_target_metadata_fingerprint": (
            cell_target_metadata_digest.hexdigest()
        ),
        "source_components": source_components,
        "group_key_fields": PHASE2_GROUP_KEY_FIELDS,
        "manifest_columns": PHASE2_MANIFEST_COLUMNS,
        "arms": {name: list(features) for name, features in arms.items()},
    }
    cv_source_fingerprint = hashlib.sha256(
        json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "x_fingerprint": x_digest.hexdigest(),
        "y_fingerprint": y_fingerprint,
        "group_ids_fingerprint": group_ids_fingerprint,
        "split_fingerprint": split_fingerprint,
        "cv_source_fingerprint": cv_source_fingerprint,
    }


def phase2_cv_source_components() -> Mapping[str, str]:
    """回傳 mapping-free Phase 2 core 與 CV/model/arm source hashes。

    Returns:
        只含 ``phase2.py`` 與 ``cv.py`` 內容 SHA-256 的 immutable mapping；
        dose diagnostics 不屬於 CV provenance。
    """
    module_directory = Path(__file__).resolve().parent
    components = {
        name: hashlib.sha256((module_directory / name).read_bytes()).hexdigest()
        for name in ("phase2.py", "cv.py")
    }
    return MappingProxyType(components)


def build_phase2_targets(
    retained_cells: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """以 opaque donor、passage、condition index 建立 72 個正式 targets。

    此介面只讀取 retained cell 的 identity／``IDO_score`` 與四欄 manifest，
    不接受 condition label 或 dose mapping。每組少於 10 cells 時 fail closed，
    10–29 cells 則保留並標為 ``low``。

    Args:
        retained_cells: Phase 1 frozen retained cells；每個 whole-cell key 一列。
        manifest: 欄位必須恰為 ``image_key,b_id,passage,condition_index``。

    Returns:
        依 donor、passage、condition index 排序的 72-row target table。

    Raises:
        TypeError: 輸入不是 pandas DataFrame 時拋出。
        Phase2ContractError: schema、正式 counts、key 或 target contract 不成立。
    """
    cells = _validated_target_cells(retained_cells)
    labels = _validated_manifest(manifest)
    cells = cells.merge(labels, on="image_key", how="left", validate="many_to_one")
    if cells.loc[:, PHASE2_GROUP_KEY_FIELDS].isna().any().any():
        raise Phase2ContractError("retained cells 含 manifest allowlist 外的 image_key")

    grouped = cells.groupby(list(PHASE2_GROUP_KEY_FIELDS), sort=True, observed=True)
    targets = grouped.agg(
        fov_count=("image_key", "nunique"),
        cells_after=("cell_label", "size"),
        group_IDO_score=("IDO_score", "median"),
    ).reset_index()
    targets.insert(
        0,
        "group_id",
        targets["b_id"].astype(str)
        + "_P"
        + targets["passage"].astype(str)
        + "_C"
        + targets["condition_index"].astype(int).astype(str).str.zfill(2),
    )
    targets["confidence_flag"] = np.where(
        targets["cells_after"].lt(30), "low", "normal"
    )

    below_minimum = targets.loc[targets["cells_after"].lt(10), "group_id"].tolist()
    if below_minimum:
        raise Phase2ContractError(
            f"Phase 2 group below 10 cells: {below_minimum}"
        )
    if len(cells) != PHASE2_CELL_COUNT:
        raise Phase2ContractError(
            f"Phase 2 cell count expected {PHASE2_CELL_COUNT}, actual {len(cells)}"
        )
    if cells["image_key"].nunique() != PHASE2_FOV_COUNT:
        raise Phase2ContractError(
            f"Phase 2 FOV count expected {PHASE2_FOV_COUNT}, "
            f"actual {cells['image_key'].nunique()}"
        )
    if len(targets) != PHASE2_GROUP_COUNT:
        raise Phase2ContractError(
            f"Phase 2 group count expected {PHASE2_GROUP_COUNT}, actual {len(targets)}"
        )
    expected_indices = tuple(range(1, 9))
    for donor_passage, group in targets.groupby(
        ["b_id", "passage"], sort=False, observed=True
    ):
        actual = tuple(group["condition_index"].astype(int))
        if actual != expected_indices:
            raise Phase2ContractError(
                f"{donor_passage} condition indices expected 1-8, actual {actual}"
            )
    if targets["group_IDO_score"].nunique(dropna=False) != PHASE2_GROUP_COUNT:
        raise Phase2ContractError("Phase 2 必須有 72 個相異 finite targets")
    numeric = targets.loc[
        :, ["fov_count", "cells_after", "group_IDO_score"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise Phase2ContractError("Phase 2 target numeric values 必須全為 finite")
    return targets.loc[:, PHASE2_TARGET_COLUMNS].reset_index(drop=True)


def _validate_modeling_cells_against_targets(
    modeling_cells: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """逐列驗證 modeling target metadata 等於 canonical Phase 2 targets。

    Args:
        modeling_cells: 已組裝、準備送入 CV 的 19,648-cell frame。
        targets: 由 opaque group keys 建立的 canonical 72-row target table。

    Raises:
        TypeError: 任一輸入不是 pandas DataFrame 時拋出。
        Phase2ContractError: target table 或任一 cell 的 group ID、target exact
            bits、confidence 不符合 canonical targets 時拋出。
    """
    if not isinstance(modeling_cells, pd.DataFrame):
        raise TypeError("modeling_cells 必須是 pandas DataFrame")
    if not isinstance(targets, pd.DataFrame):
        raise TypeError("targets 必須是 pandas DataFrame")
    if tuple(targets.columns) != PHASE2_TARGET_COLUMNS:
        raise Phase2ContractError("canonical Phase 2 targets schema mismatch")
    target_keys = list(PHASE2_GROUP_KEY_FIELDS)
    if (
        len(targets) != PHASE2_GROUP_COUNT
        or targets.duplicated(target_keys).any()
        or targets["group_id"].duplicated().any()
    ):
        raise Phase2ContractError(
            "canonical Phase 2 targets 必須恰含 72 unique groups"
        )
    derived_group_ids = (
        targets["b_id"].astype(str)
        + "_P"
        + targets["passage"].astype(str)
        + "_C"
        + targets["condition_index"].astype(int).astype(str).str.zfill(2)
    )
    expected_confidence = np.where(
        pd.to_numeric(targets["cells_after"], errors="raise").to_numpy(
            dtype=np.int64
        )
        < 30,
        "low",
        "normal",
    )
    if not targets["group_id"].astype(str).eq(derived_group_ids).all() or not np.array_equal(
        targets["confidence_flag"].astype(str).to_numpy(dtype=object),
        expected_confidence,
    ):
        raise Phase2ContractError(
            "canonical Phase 2 targets group_id/confidence metadata mismatch"
        )
    required_cell_columns = {
        *PHASE2_GROUP_KEY_FIELDS,
        "group_id",
        "group_IDO_score",
        "confidence_flag",
    }
    missing = sorted(required_cell_columns - set(modeling_cells.columns))
    if missing:
        raise Phase2ContractError(
            f"modeling cells 缺 canonical Phase 2 target columns：{missing}"
        )
    expected = targets.loc[
        :,
        [
            *PHASE2_GROUP_KEY_FIELDS,
            "group_id",
            "group_IDO_score",
            "confidence_flag",
        ],
    ]
    comparison = modeling_cells.loc[
        :,
        [
            *PHASE2_GROUP_KEY_FIELDS,
            "group_id",
            "group_IDO_score",
            "confidence_flag",
        ],
    ].merge(
        expected,
        on=target_keys,
        how="left",
        validate="many_to_one",
        suffixes=("_cell", "_target"),
    )
    group_ids_match = comparison["group_id_cell"].astype(str).eq(
        comparison["group_id_target"].astype(str)
    )
    confidence_matches = comparison["confidence_flag_cell"].astype(str).eq(
        comparison["confidence_flag_target"].astype(str)
    )
    try:
        actual_targets = pd.to_numeric(
            comparison["group_IDO_score_cell"], errors="raise"
        ).to_numpy(dtype=np.float64)
        expected_targets = pd.to_numeric(
            comparison["group_IDO_score_target"], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError(
            "modeling cells/canonical Phase 2 targets 必須是 numeric"
        ) from error
    exact_target_bits_match = (
        np.isfinite(actual_targets).all()
        and np.isfinite(expected_targets).all()
        and np.array_equal(
            actual_targets.view(np.uint64),
            expected_targets.view(np.uint64),
        )
    )
    if (
        len(comparison) != len(modeling_cells)
        or not group_ids_match.all()
        or not confidence_matches.all()
        or not exact_target_bits_match
    ):
        mismatches = []
        if not group_ids_match.all():
            mismatches.append("group_id")
        if not exact_target_bits_match:
            mismatches.append("group_IDO_score exact bits")
        if not confidence_matches.all():
            mismatches.append("confidence_flag")
        raise Phase2ContractError(
            "modeling cells 不符合 canonical Phase 2 targets：" + repr(mismatches)
        )


def _validate_frozen_evidence(evidence: FrozenPhase1Evidence) -> None:
    if not isinstance(evidence, FrozenPhase1Evidence):
        raise TypeError("evidence 必須是 FrozenPhase1Evidence")
    actual = evidence.as_mapping()
    if actual != dict(PHASE1_PINNED_ARTIFACT_SHA256):
        differences = {
            name: {
                "expected": expected,
                "actual": actual.get(name),
            }
            for name, expected in PHASE1_PINNED_ARTIFACT_SHA256.items()
            if actual.get(name) != expected
        }
        raise Phase2ContractError(
            f"Phase 1 pinned artifact fingerprints mismatch: {differences}"
        )
    expected_pre = Phase1ConsistencyEvidence(
        scope="full_raw_join_23976_pairs_23012_cells_693_fovs",
        checked_row_count=23_976,
        checked_cell_count=23_012,
        checked_duplicate_key_count=938,
        feature_count=49,
    )
    expected_retained = Phase1ConsistencyEvidence(
        scope="full_retained_raw_join_20440_pairs_19648_cells_693_fovs",
        checked_row_count=20_440,
        checked_cell_count=19_648,
        checked_duplicate_key_count=769,
        feature_count=49,
    )
    if evidence.pre_border_consistency != expected_pre:
        raise Phase2ContractError("Phase 1 pre-border Rui49 consistency evidence mismatch")
    if evidence.retained_consistency != expected_retained:
        raise Phase2ContractError("Phase 1 retained Rui49 consistency evidence mismatch")


def _authoritative_retained_keys(
    phase1_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if not isinstance(phase1_oof, pd.DataFrame):
        raise TypeError("phase1_oof 必須是 pandas DataFrame")
    required = {"configuration_id", *_IDENTITY_COLUMNS}
    missing = sorted(required - set(phase1_oof.columns))
    if missing:
        raise Phase2ContractError(f"phase1_oof 缺少欄位：{missing}")
    if len(phase1_oof) != PHASE2_CELL_COUNT * 24:
        raise Phase2ContractError(
            "Phase 1 OOF row count 必須恰為 19,648×24"
        )
    configuration_ids = phase1_oof["configuration_id"].astype(str)
    if set(configuration_ids) != _PHASE1_CONFIGURATION_IDS:
        raise Phase2ContractError("Phase 1 OOF configuration roster 必須恰為 4×6")
    authoritative = phase1_oof.loc[
        configuration_ids.eq(PHASE1_ORDER_CONFIGURATION_ID), _IDENTITY_COLUMNS
    ].reset_index(drop=True)
    if (
        len(authoritative) != PHASE2_CELL_COUNT
        or authoritative.isna().any().any()
        or authoritative.duplicated().any()
    ):
        raise Phase2ContractError("authoritative Phase 1 OOF cell keys 不完整或重複")
    for configuration_id, configuration in phase1_oof.groupby(
        "configuration_id", sort=False
    ):
        keys = configuration.loc[:, _IDENTITY_COLUMNS].reset_index(drop=True)
        if not keys.equals(authoritative):
            raise Phase2ContractError(
                f"Phase 1 OOF retained key sequence 漂移：{configuration_id}"
            )
    hashed = pd.util.hash_pandas_object(
        authoritative,
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64)
    fingerprint = hashlib.sha256(hashed.tobytes(order="C")).hexdigest()
    if fingerprint != PHASE1_RETAINED_SEQUENCE_SHA256:
        raise Phase2ContractError(
            "Phase 1 OOF retained sequence SHA-256 mismatch: "
            f"expected={PHASE1_RETAINED_SEQUENCE_SHA256}, actual={fingerprint}"
        )
    return authoritative, fingerprint


def _reconstruct_basic_values(
    cell_level_basic: pd.DataFrame,
    cell_dedup_report: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(cell_level_basic, pd.DataFrame):
        raise TypeError("cell_level_basic 必須是 pandas DataFrame")
    if not isinstance(cell_dedup_report, pd.DataFrame):
        raise TypeError("cell_dedup_report 必須是 pandas DataFrame")
    basic_required = {
        *_IDENTITY_COLUMNS,
        "IDO_score",
        "cell__area",
        *NUCLEUS_FEATURE_COLUMNS,
    }
    missing_basic = sorted(basic_required - set(cell_level_basic.columns))
    if missing_basic:
        raise Phase2ContractError(f"cell_level_basic 缺少欄位：{missing_basic}")
    dedup_required = {
        *_IDENTITY_COLUMNS,
        "original_row_count",
        "IDO_score_after",
        *NUCLEUS_FEATURE_COLUMNS,
    }
    missing_dedup = sorted(dedup_required - set(cell_dedup_report.columns))
    if missing_dedup:
        raise Phase2ContractError(f"cell_dedup_report 缺少欄位：{missing_dedup}")
    if len(cell_level_basic) != 23_976:
        raise Phase2ContractError("cell_level_basic row count 必須恰為 23,976")
    key_counts = cell_level_basic.groupby(
        list(_IDENTITY_COLUMNS), sort=False, dropna=False
    ).size()
    if len(key_counts) != 23_012:
        raise Phase2ContractError("cell_level_basic unique key count 必須恰為 23,012")
    multi_counts = key_counts.loc[key_counts.gt(1)]
    if len(multi_counts) != 938:
        raise Phase2ContractError("cell_level_basic multirow key count 必須恰為 938")
    dedup_keys = cell_dedup_report.loc[:, _IDENTITY_COLUMNS]
    if (
        len(cell_dedup_report) != 938
        or dedup_keys.isna().any().any()
        or dedup_keys.duplicated().any()
    ):
        raise Phase2ContractError("cell_dedup_report 必須恰含 938 unique keys")
    dedup_index = pd.MultiIndex.from_frame(dedup_keys)
    if set(dedup_index) != set(multi_counts.index):
        raise Phase2ContractError("cell_dedup_report keys 必須恰為 raw multirow keys")
    reported_counts = pd.Series(
        pd.to_numeric(
            cell_dedup_report["original_row_count"], errors="raise"
        ).to_numpy(dtype=np.int64),
        index=dedup_index,
    )
    if not reported_counts.sort_index().equals(multi_counts.astype(np.int64).sort_index()):
        raise Phase2ContractError("cell_dedup_report original_row_count mismatch")
    parsed_numeric_fingerprint = _dedup_parsed_numeric_fingerprint(
        cell_dedup_report
    )
    if parsed_numeric_fingerprint != PHASE1_DEDUP_PARSED_NUMERIC_SHA256:
        raise Phase2ContractError(
            "cell_dedup_report round-trip parsed-numeric fingerprint mismatch: "
            f"expected={PHASE1_DEDUP_PARSED_NUMERIC_SHA256}, "
            f"actual={parsed_numeric_fingerprint}"
        )

    first_rows = cell_level_basic.drop_duplicates(
        list(_IDENTITY_COLUMNS), keep="first"
    ).loc[
        :,
        [
            *_IDENTITY_COLUMNS,
            "IDO_score",
            "cell__area",
            *NUCLEUS_FEATURE_COLUMNS,
        ],
    ]
    first_index = pd.MultiIndex.from_frame(first_rows.loc[:, _IDENTITY_COLUMNS])
    singleton_values = first_rows.loc[~first_index.isin(dedup_index)].copy()
    try:
        singleton_cell_area = pd.to_numeric(
            singleton_values["cell__area"], errors="raise"
        ).to_numpy(dtype=np.float64)
        singleton_nucleus_area = pd.to_numeric(
            singleton_values["nucleus__area"], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError(
            "singleton cell/nucleus area 必須為 numeric"
        ) from error
    singleton_denominator = singleton_cell_area - singleton_nucleus_area
    if (
        not np.isfinite(singleton_cell_area).all()
        or not np.isfinite(singleton_nucleus_area).all()
        or not np.isfinite(singleton_denominator).all()
        or np.less_equal(singleton_denominator, 0.0).any()
    ):
        raise Phase2ContractError(
            "singleton nucleus/cytoplasm denominator 必須為 finite 且大於 0"
        )
    singleton_values["nucleus_cytoplasm_area_ratio"] = (
        singleton_nucleus_area / singleton_denominator
    )
    singleton_values = singleton_values.loc[
        :, [*_IDENTITY_COLUMNS, "IDO_score", *NUCLEUS_FEATURE_COLUMNS]
    ]
    dedup_values = cell_dedup_report.loc[
        :, [*_IDENTITY_COLUMNS, "IDO_score_after", *NUCLEUS_FEATURE_COLUMNS]
    ].rename(columns={"IDO_score_after": "IDO_score"})
    canonical = pd.concat([singleton_values, dedup_values], ignore_index=True)
    if len(canonical) != 23_012 or canonical.duplicated(list(_IDENTITY_COLUMNS)).any():
        raise Phase2ContractError("reconstructed basic values 必須恰含 23,012 keys")
    try:
        values = canonical.loc[
            :, ["IDO_score", *NUCLEUS_FEATURE_COLUMNS]
        ].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError("basic/dedup canonical values 必須為 numeric") from error
    if not np.isfinite(values).all():
        raise Phase2ContractError("basic/dedup canonical values 必須全為 finite")
    return canonical


def _dedup_parsed_numeric_fingerprint(cell_dedup_report: pd.DataFrame) -> str:
    """計算 canonical dedup keys 與 18 個 parsed numeric 欄的 exact fingerprint。

    Args:
        cell_dedup_report: 已通過 formal 938-row schema/key gates 的 dedup report。

    Returns:
        依 keys 排序、以 little-endian float64 bits 計算的 SHA-256。

    Raises:
        Phase2ContractError: key label 或 numeric values 無法精確正規化時拋出。
    """
    numeric_columns = ("IDO_score_after", *NUCLEUS_FEATURE_COLUMNS)
    frame = cell_dedup_report.loc[
        :, [*_IDENTITY_COLUMNS, *numeric_columns]
    ].copy()
    frame["image_key"] = frame["image_key"].astype(str)
    try:
        labels = pd.to_numeric(
            frame["cell_label"], errors="raise"
        ).to_numpy(dtype=np.float64)
        values = frame.loc[:, numeric_columns].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError(
            "cell_dedup_report parsed-numeric fingerprint inputs 必須為 numeric"
        ) from error
    if (
        not np.isfinite(labels).all()
        or not np.equal(labels, np.floor(labels)).all()
        or not np.isfinite(values).all()
    ):
        raise Phase2ContractError(
            "cell_dedup_report parsed-numeric fingerprint inputs 必須 finite"
        )
    frame["_cell_label_int"] = labels.astype(np.int64)
    frame = frame.sort_values(
        ["image_key", "_cell_label_int"], kind="stable"
    ).reset_index(drop=True)
    values = frame.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)
    header = {
        "columns": [*_IDENTITY_COLUMNS, *numeric_columns],
        "row_count": len(frame),
    }
    key_payload = list(
        zip(
            frame["image_key"].tolist(),
            frame["_cell_label_int"].astype(int).tolist(),
        )
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        json.dumps(
            key_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(values.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _first_rui49_values(
    cell_level_rui49: pd.DataFrame,
    *,
    cell_level_basic: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(cell_level_rui49, pd.DataFrame):
        raise TypeError("cell_level_rui49 必須是 pandas DataFrame")
    required = {*_IDENTITY_COLUMNS, *RUI49_FEATURE_COLUMNS}
    missing = sorted(required - set(cell_level_rui49.columns))
    if missing:
        raise Phase2ContractError(f"cell_level_rui49 缺少欄位：{missing}")
    if len(cell_level_rui49) != 23_976:
        raise Phase2ContractError("cell_level_rui49 row count 必須恰為 23,976")
    rui_key_counts = cell_level_rui49.groupby(
        list(_IDENTITY_COLUMNS), sort=False, dropna=False
    ).size()
    basic_key_counts = cell_level_basic.groupby(
        list(_IDENTITY_COLUMNS), sort=False, dropna=False
    ).size()
    if not rui_key_counts.sort_index().equals(basic_key_counts.sort_index()):
        raise Phase2ContractError("Rui49/raw basic whole-cell key multiset mismatch")
    first = cell_level_rui49.drop_duplicates(
        list(_IDENTITY_COLUMNS), keep="first"
    ).loc[:, [*_IDENTITY_COLUMNS, *RUI49_FEATURE_COLUMNS]]
    if len(first) != 23_012 or first.duplicated(list(_IDENTITY_COLUMNS)).any():
        raise Phase2ContractError("Rui49 first-row table 必須恰含 23,012 keys")
    try:
        values = first.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError("Rui49 values 必須為 numeric") from error
    if not np.isfinite(values).all():
        raise Phase2ContractError("Rui49 values 必須全為 finite")
    return first


def _healthy_roster(feature_health_report: pd.DataFrame) -> tuple[str, ...]:
    if not isinstance(feature_health_report, pd.DataFrame):
        raise TypeError("feature_health_report 必須是 pandas DataFrame")
    required = {"feature", "removed_from_arms"}
    missing = sorted(required - set(feature_health_report.columns))
    if missing:
        raise Phase2ContractError(f"feature_health_report 缺少欄位：{missing}")
    if len(feature_health_report) != 66:
        raise Phase2ContractError("feature_health_report 必須恰含 66 rows")
    features = feature_health_report["feature"].astype(str)
    if features.duplicated().any():
        raise Phase2ContractError("feature_health_report feature 必須唯一")
    cell_report = feature_health_report.loc[features.str.startswith("cell__")].copy()
    if len(cell_report) != 49 or tuple(cell_report["feature"]) != tuple(
        RUI49_FEATURE_COLUMNS
    ):
        raise Phase2ContractError("health report 必須依 Rui49 canonical order 含 49 欄")
    removed_mask = cell_report["removed_from_arms"]
    if not pd.api.types.is_bool_dtype(removed_mask):
        raise Phase2ContractError("removed_from_arms 必須是 boolean")
    removed = tuple(cell_report.loc[removed_mask, "feature"].astype(str))
    if removed != ("cell__MinIntensity",):
        raise Phase2ContractError("health report 只能移除 cell__MinIntensity")
    healthy = tuple(cell_report.loc[~removed_mask, "feature"].astype(str))
    if len(healthy) != 48:
        raise Phase2ContractError("health report 必須提供 frozen 48-feature roster")
    return healthy


def _filtered_roster(
    feature_redundancy_report: pd.DataFrame,
    *,
    healthy_features: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(feature_redundancy_report, pd.DataFrame):
        raise TypeError("feature_redundancy_report 必須是 pandas DataFrame")
    required = {
        "record_type",
        "feature",
        "canonical_index",
        "decision",
        "selection_inputs",
    }
    missing = sorted(required - set(feature_redundancy_report.columns))
    if missing:
        raise Phase2ContractError(
            f"feature_redundancy_report 缺少欄位：{missing}"
        )
    decisions = feature_redundancy_report.loc[
        feature_redundancy_report["record_type"].eq("feature_decision")
    ].copy()
    if len(decisions) != 48 or decisions["feature"].duplicated().any():
        raise Phase2ContractError("redundancy report 必須恰含 48 unique feature decisions")
    if set(decisions["feature"].astype(str)) != set(healthy_features):
        raise Phase2ContractError("redundancy decisions 必須恰覆蓋 health 48 roster")
    if not decisions["selection_inputs"].astype(str).eq("X").all():
        raise Phase2ContractError("redundancy report selection_inputs 必須只含 X")
    try:
        decisions["canonical_index"] = pd.to_numeric(
            decisions["canonical_index"], errors="raise"
        ).astype(int)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError("canonical_index 必須為整數") from error
    decisions = decisions.sort_values("canonical_index", kind="stable")
    if tuple(decisions["canonical_index"]) != tuple(range(48)):
        raise Phase2ContractError("redundancy canonical_index 必須恰為 0-47")
    if not set(decisions["decision"].astype(str)).issubset({"retained", "removed"}):
        raise Phase2ContractError("redundancy decision 含未知值")
    filtered = tuple(
        decisions.loc[decisions["decision"].eq("retained"), "feature"].astype(str)
    )
    canonical_filtered = tuple(
        feature for feature in healthy_features if feature in set(filtered)
    )
    if len(filtered) != 30 or filtered != canonical_filtered:
        raise Phase2ContractError("redundancy report 必須提供 canonical 30-feature roster")
    return filtered


def _validated_target_cells(retained_cells: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(retained_cells, pd.DataFrame):
        raise TypeError("retained_cells 必須是 pandas DataFrame")
    if retained_cells.columns.has_duplicates:
        raise Phase2ContractError("retained_cells column labels 必須唯一")
    required = ("image_key", "cell_label", "IDO_score")
    missing = sorted(set(required) - set(retained_cells.columns))
    if missing:
        raise Phase2ContractError(f"retained_cells 缺少欄位：{missing}")
    frame = retained_cells.loc[:, required].copy()
    if frame.loc[:, ["image_key", "cell_label"]].isna().any().any():
        raise Phase2ContractError("retained whole-cell identity 不可為空")
    frame["image_key"] = frame["image_key"].astype(str)
    if frame["image_key"].str.strip().ne(frame["image_key"]).any() or frame[
        "image_key"
    ].eq("").any():
        raise Phase2ContractError("retained image_key 必須是無前後空白的非空字串")
    if frame.duplicated(["image_key", "cell_label"]).any():
        raise Phase2ContractError("retained whole-cell keys 必須唯一")
    try:
        frame["IDO_score"] = pd.to_numeric(frame["IDO_score"], errors="raise")
    except (TypeError, ValueError) as error:
        raise Phase2ContractError("IDO_score 必須是 finite numeric") from error
    if not np.isfinite(frame["IDO_score"].to_numpy(dtype=np.float64)).all():
        raise Phase2ContractError("IDO_score 必須是 finite numeric")
    return frame


def _validated_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest 必須是 pandas DataFrame")
    if tuple(manifest.columns) != PHASE2_MANIFEST_COLUMNS:
        raise Phase2ContractError(
            "manifest columns 必須恰為四欄 allowlist："
            f"{PHASE2_MANIFEST_COLUMNS}"
        )
    if len(manifest) != PHASE2_FOV_COUNT:
        raise Phase2ContractError(
            f"manifest row count expected {PHASE2_FOV_COUNT}, actual {len(manifest)}"
        )
    frame = manifest.copy()
    if frame.isna().any().any() or frame["image_key"].duplicated().any():
        raise Phase2ContractError("manifest 四欄不可為空且 image_key 必須唯一")
    frame["image_key"] = frame["image_key"].astype(str)
    parsed = frame["image_key"].str.extract(_IMAGE_KEY_PATTERN)
    if parsed.isna().any().any():
        raise Phase2ContractError("manifest image_key 不符合 Bx_Px_Cxx_ 格式")
    try:
        passages = pd.to_numeric(frame["passage"], errors="raise").astype(int)
        conditions = pd.to_numeric(
            frame["condition_index"], errors="raise"
        ).astype(int)
    except (TypeError, ValueError) as error:
        raise Phase2ContractError(
            "manifest passage/condition_index 必須是整數"
        ) from error
    matches = (
        parsed["b_id"].eq(frame["b_id"].astype(str))
        & parsed["passage"].astype(int).eq(passages)
        & parsed["condition_index"].astype(int).eq(conditions)
    )
    if not matches.all():
        raise Phase2ContractError(
            "manifest filename-derived condition indices 必須 693/693 相符"
        )
    if not conditions.between(1, 8).all():
        raise Phase2ContractError("manifest condition_index 必須全為 1-8")
    frame["passage"] = passages
    frame["condition_index"] = conditions
    return frame


__all__ = [
    "METRICS_WITHIN_CONDITION_COLUMNS",
    "PHASE1_VS_PHASE2_COMPARISON_COLUMNS",
    "PHASE1_DEDUP_PARSED_NUMERIC_SHA256",
    "PHASE1_ORDER_CONFIGURATION_ID",
    "PHASE1_PINNED_ARTIFACT_SHA256",
    "PHASE1_RETAINED_SEQUENCE_SHA256",
    "PHASE2_CELL_COUNT",
    "PHASE2_FOV_COUNT",
    "PHASE2_GROUP_COUNT",
    "PHASE2_GROUP_KEY_FIELDS",
    "PHASE2_MANIFEST_COLUMNS",
    "PHASE2_SHRINKAGE_COLUMNS",
    "PHASE2_SHRINKAGE_GROUP_COLUMNS",
    "PHASE2_TARGET_COLUMNS",
    "FrozenPhase1Evidence",
    "Phase1ConsistencyEvidence",
    "Phase2ContractError",
    "Phase2CvAssembly",
    "Phase2CvRunResult",
    "Phase2FrozenPopulation",
    "Phase2LeakageContext",
    "Phase2LeakageError",
    "Phase2PopulationContract",
    "Phase2ShrinkageResult",
    "analyze_phase2_shrinkage",
    "assemble_phase2_cv",
    "build_phase2_targets",
    "build_metrics_within_condition",
    "compare_phase1_phase2",
    "reconstruct_phase2_population",
    "run_phase2_cv",
    "run_phase2_leakage_preflight",
    "phase2_cv_source_components",
    "validate_phase2_oof",
]
