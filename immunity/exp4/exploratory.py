"""Exp4 UMAP/k-means 探索性分析的唯讀、fail-closed 核心介面。

本模組刻意不接觸 Exp4 canonical output，也不負責寫檔。呼叫端可將
``ExploratoryAnalysisResult`` 的 DataFrame 與 PNG bytes 交給後續 runner，
由 runner 以自己的 transactional staging 流程落盤。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from io import BytesIO
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


EXPECTED_CELL_COUNT = 19_648
EXPECTED_FOV_COUNT = 693
EXPECTED_GROUP_COUNT = 9
RANDOM_STATE = 42
KMEANS_CLUSTER_COUNT = 2
KMEANS_N_INIT = 10
CONDITION_WARNING_TEXT = (
    "condition 標籤可信度未確認；僅作視覺化，不得據以做劑量相關結論。"
)

EMBEDDING_COLUMNS = (
    "image_key",
    "cell_label",
    "b_id",
    "passage",
    "condition",
    "ifn_dose",
    "tnf_dose",
    "group_id",
    "UMAP1",
    "UMAP2",
    "is_putative_ifn_stimulated",
    "kmeans_cluster_id",
    "kmeans_cluster_label",
)
CLUSTER_FEATURE_REPORT_COLUMNS = (
    "feature",
    "ido_low_cluster_id",
    "ido_high_cluster_id",
    "low_cell_count",
    "high_cell_count",
    "low_target_median",
    "high_target_median",
    "median_ido_low",
    "median_ido_high",
    "high_minus_low",
    "absolute_difference",
)
# Cluster assignments use the same exact 13-column schema as the embedding;
# the result exposes only the putative-IFN rows through ``cluster_assignments``.
CLUSTER_COLUMNS = EMBEDDING_COLUMNS
_METADATA_COLUMNS = frozenset(
    {
        "image_key",
        "cell_label",
        "nucleus_label",
        "b_id",
        "passage",
        "condition",
        "ifn_dose",
        "tnf_dose",
        "group_id",
    }
)
_TARGET_ALIASES = (
    "group_IDO_score",
    "observed_ido_score",
    "target",
)
_MANIFEST_LABEL_COLUMNS = (
    "image_key",
    "b_id",
    "passage",
    "condition",
    "ifn_dose",
    "tnf_dose",
    "group_id",
)


class ExploratoryContractError(ValueError):
    """表示 exploratory population、schema 或 leakage contract 不成立。"""


@dataclass(frozen=True)
class ExploratoryAnalysisResult:
    """保存 UMAP、putative-IFN clusters、特徵報告與可 stage artifact。

    Attributes:
        embedding: 每顆 retained cell 一列的 UMAP 表。
        cluster_assignments: 僅 putative IFN subset 的 UMAP/k-means 表。
        cluster_feature_report: 30 個 X-only feature 的兩群中位數與差異。
        figures: 三個 deterministic PNG 的檔名到 bytes 對照；本模組不寫檔。
        timing: 各分析階段與總耗時（秒）。
        provenance: seed、實際 estimator defaults、population gate 與 warning。
    """

    embedding: pd.DataFrame
    cluster_assignments: pd.DataFrame
    cluster_feature_report: pd.DataFrame
    figures: Mapping[str, bytes]
    timing: Mapping[str, float]
    provenance: Mapping[str, object]

    @property
    def embedding_table(self) -> pd.DataFrame:
        """回傳 embedding table 的唯讀語意別名。"""
        return self.embedding

    @property
    def clusters(self) -> pd.DataFrame:
        """回傳 IFN cluster assignments 的語意別名。"""
        return self.cluster_assignments

    @property
    def kmeans_table(self) -> pd.DataFrame:
        """回傳 IFN k-means assignments 的明確別名。"""
        return self.cluster_assignments

    @property
    def feature_report(self) -> pd.DataFrame:
        """回傳 cluster feature report 的語意別名。"""
        return self.cluster_feature_report

    @property
    def cluster_report(self) -> pd.DataFrame:
        """回傳 cluster feature report 的完整語意別名。"""
        return self.cluster_feature_report

    @property
    def cluster_features(self) -> pd.DataFrame:
        """回傳 cluster feature report 的 runner-friendly 別名。"""
        return self.cluster_feature_report

    @property
    def report(self) -> pd.DataFrame:
        """回傳 cluster feature report 的簡短別名。"""
        return self.cluster_feature_report

    @property
    def figure_pngs(self) -> Mapping[str, bytes]:
        """回傳 PNG bytes 的語意別名。"""
        return self.figures

    @property
    def pngs(self) -> Mapping[str, bytes]:
        """回傳 PNG bytes 的簡短別名。"""
        return self.figures

    @property
    def figure_bytes(self) -> Mapping[str, bytes]:
        """回傳 PNG bytes 的明確別名。"""
        return self.figures

    @property
    def timings(self) -> Mapping[str, float]:
        """回傳 timing mapping 的複數別名。"""
        return self.timing

    def to_dict(self) -> dict[str, object]:
        """回傳不含大型資料與 bytes 的 machine-readable 摘要。"""
        return {
            "embedding_row_count": int(len(self.embedding)),
            "cluster_row_count": int(len(self.cluster_assignments)),
            "cluster_feature_report_row_count": int(
                len(self.cluster_feature_report)
            ),
            "figure_names": list(self.figures),
            "timing": dict(self.timing),
            "provenance": dict(self.provenance),
        }


# Some callers use the shorter result name when staging analysis bundles.
ExploratoryResult = ExploratoryAnalysisResult


def run_exploratory_analysis(
    retained_cells: pd.DataFrame | None = None,
    feature_columns: Sequence[str] | None = None,
    manifest: pd.DataFrame | None = None,
    *,
    expected_cell_count: int = EXPECTED_CELL_COUNT,
    expected_fov_count: int = EXPECTED_FOV_COUNT,
    expected_group_count: int = EXPECTED_GROUP_COUNT,
    target_column: str = "group_IDO_score",
    cells: pd.DataFrame | None = None,
    rui_filtered_features: Sequence[str] | None = None,
    rui_filtered: Sequence[str] | None = None,
    manifest_labels: pd.DataFrame | None = None,
) -> ExploratoryAnalysisResult:
    """執行 deterministic UMAP 與 putative-IFN k-means 探索性分析。

    Args:
        retained_cells: border-retained、每個 whole-cell 一列的 feature table。
        feature_columns: 恰 30 個 ``rui_filtered`` X-only 欄位，順序會保留。
        manifest: 以 ``image_key`` 一列一 FOV 提供 metadata labels 的 manifest。
        expected_cell_count: exploratory population 的 exact unique cell count。
        expected_fov_count: exploratory population 的 exact FOV count。
        expected_group_count: exploratory population 的 exact group count。
        target_column: retained table（或 manifest）內的 observed group target 欄位。
        cells: ``retained_cells`` 的 keyword alias。
        rui_filtered_features: ``feature_columns`` 的 keyword alias。
        rui_filtered: ``feature_columns`` 的簡短 alias。
        manifest_labels: ``manifest`` 的 keyword alias。

    Returns:
        含 embedding、IFN clusters、30-feature report、PNG bytes、timing 與
        provenance 的 immutable result object。

    Raises:
        ExploratoryContractError: schema、population、join、finite、X-only、
            k-means 或 post-hoc naming contract 任何一項不成立時。

    Notes:
        StandardScaler 會在完整 exploratory population fit；k-means 只看
        putative IFN subset 的 UMAP1/UMAP2。``ifn_dose > 0`` 只是不可信的
        selector，且 target 永遠在 k-means fit 完成後才用於 high/low 命名。
    """
    started = perf_counter()
    cells_frame = _resolve_alias(
        retained_cells,
        cells,
        primary_name="retained_cells",
        alias_name="cells",
    )
    filtered_alias = _resolve_alias(
        rui_filtered_features,
        rui_filtered,
        primary_name="rui_filtered_features",
        alias_name="rui_filtered",
    )
    feature_roster = _resolve_alias(
        feature_columns,
        filtered_alias,
        primary_name="feature_columns",
        alias_name="rui_filtered_features",
    )
    manifest_frame = _resolve_alias(
        manifest,
        manifest_labels,
        primary_name="manifest",
        alias_name="manifest_labels",
    )
    if cells_frame is None or feature_roster is None or manifest_frame is None:
        raise ExploratoryContractError(
            "retained_cells、feature_columns 與 manifest 都必須提供"
        )
    expected = _validate_expected_counts(
        expected_cell_count,
        expected_fov_count,
        expected_group_count,
    )

    prepared, labels, target_name, population = _prepare_population(
        cells_frame,
        manifest_frame,
        feature_roster,
        target_column=target_column,
        expected=expected,
    )
    validation_seconds = perf_counter() - started

    feature_names = tuple(str(feature) for feature in feature_roster)
    x = prepared.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)

    import umap  # type: ignore[import-not-found]

    umap_model = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    umap_params = _estimator_params(umap_model)
    umap_started = perf_counter()
    # UMAP 0.5.x reports that its random state forces n_jobs=1. This is an
    # expected deterministic configuration detail, not a failed analysis.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value .* overridden to 1 by setting random_state",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"n_neighbors is larger than the dataset size; truncating to X.shape\[0\] - 1",
        )
        coordinates = np.asarray(umap_model.fit_transform(scaled_x), dtype=np.float64)
    umap_seconds = perf_counter() - umap_started
    if coordinates.shape != (len(prepared), 2) or not np.isfinite(coordinates).all():
        raise ExploratoryContractError("UMAP embedding 必須是 finite 的 N×2 表")

    embedding = labels.loc[
        :, ["image_key", "cell_label", *_MANIFEST_LABEL_COLUMNS[1:]]
    ].copy()
    embedding["UMAP1"] = coordinates[:, 0]
    embedding["UMAP2"] = coordinates[:, 1]
    embedding = embedding.reset_index(drop=True)

    ifn_mask = labels["ifn_dose"].to_numpy(dtype=np.float64) > 0.0
    ifn_count = int(ifn_mask.sum())
    if ifn_count < KMEANS_CLUSTER_COUNT:
        raise ExploratoryContractError(
            "putative IFN subset 必須至少包含兩顆細胞才能執行 k-means"
        )
    kmeans = KMeans(
        n_clusters=KMEANS_CLUSTER_COUNT,
        random_state=RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    kmeans_params = _estimator_params(kmeans)
    kmeans_started = perf_counter()
    try:
        cluster_ids = kmeans.fit_predict(coordinates[ifn_mask, :2]).astype(int)
    except Exception as error:  # pragma: no cover - sklearn message varies by version
        raise ExploratoryContractError("UMAP 座標無法完成兩群 k-means") from error
    kmeans_seconds = perf_counter() - kmeans_started
    if len(np.unique(cluster_ids)) != KMEANS_CLUSTER_COUNT:
        raise ExploratoryContractError("k-means 必須產生恰好兩個 cluster")

    # This is the first point at which the observed target is allowed to affect
    # analysis output: it names the already-fitted geometric clusters only.
    target_values = prepared[target_name].to_numpy(dtype=np.float64)
    ifn_target = target_values[ifn_mask]
    target_medians = {
        cluster: float(np.median(ifn_target[cluster_ids == cluster]))
        for cluster in range(KMEANS_CLUSTER_COUNT)
    }
    if np.isclose(
        target_medians[0], target_medians[1], rtol=0.0, atol=1e-12
    ):
        raise ExploratoryContractError(
            "post-hoc cluster target medians tied；無法安全命名 high/low"
        )
    high_cluster = max(target_medians, key=target_medians.__getitem__)
    low_cluster = min(target_medians, key=target_medians.__getitem__)
    names = np.where(cluster_ids == high_cluster, "IDO_high", "IDO_low")

    embedding["is_putative_ifn_stimulated"] = ifn_mask
    embedding["kmeans_cluster_id"] = pd.Series(
        pd.array([pd.NA] * len(embedding), dtype="Int64")
    )
    embedding["kmeans_cluster_label"] = pd.Series(
        [None] * len(embedding), dtype="object"
    )
    ifn_indices = np.flatnonzero(ifn_mask)
    embedding.loc[ifn_indices, "kmeans_cluster_id"] = cluster_ids
    embedding.loc[ifn_indices, "kmeans_cluster_label"] = names
    embedding = embedding.loc[:, list(EMBEDDING_COLUMNS)].reset_index(drop=True)
    cluster_table = embedding.loc[ifn_mask, :].reset_index(drop=True)

    feature_report = _build_feature_report(
        prepared,
        feature_names,
        ifn_mask,
        cluster_ids,
        target_medians,
        high_cluster=high_cluster,
        low_cluster=low_cluster,
    )

    figure_started = perf_counter()
    figures = {
        "umap_by_donor.png": _render_umap_png(embedding, "b_id"),
        "umap_by_passage.png": _render_umap_png(embedding, "passage"),
        "umap_by_condition.png": _render_umap_png(
            embedding,
            "condition",
            annotation=CONDITION_WARNING_TEXT,
        ),
    }
    figure_seconds = perf_counter() - figure_started
    total_seconds = perf_counter() - started

    provenance: dict[str, object] = {
        "population_scope": "formal_19648_cells_693_fovs_9_groups"
        if expected == (EXPECTED_CELL_COUNT, EXPECTED_FOV_COUNT, EXPECTED_GROUP_COUNT)
        else "explicit_expected_population",
        "expected_cell_count": expected[0],
        "expected_fov_count": expected[1],
        "expected_group_count": expected[2],
        "actual_cell_count": population["cell_count"],
        "actual_fov_count": population["fov_count"],
        "actual_group_count": population["group_count"],
        "feature_columns": feature_names,
        "feature_count": len(feature_names),
        "standard_scaler_fit_scope": "full_exploratory_population",
        "standard_scaler_params": _estimator_params(scaler),
        "umap_fit_columns": feature_names,
        "umap_requested_params": {
            "n_components": 2,
            "random_state": RANDOM_STATE,
        },
        "umap_actual_params": umap_params,
        "umap_defaults": umap_params,
        "umap_params": umap_params,
        "kmeans_actual_params": kmeans_params,
        "kmeans_params": kmeans_params,
        "kmeans_fit_columns": ("UMAP1", "UMAP2"),
        "random_state": RANDOM_STATE,
        "ifn_selector": "ifn_dose > 0",
        "ifn_selector_untrusted": True,
        "ifn_selected_cell_count": ifn_count,
        "ifn_selected_fov_count": int(labels.loc[ifn_mask, "image_key"].nunique()),
        "ifn_selected_group_count": int(labels.loc[ifn_mask, "group_id"].nunique()),
        "manifest_join": "cell_to_manifest_many_to_one_by_image_key",
        "target_column": target_name,
        "target_used_for_fit": False,
        "target_used_for_posthoc_naming": True,
        "target_median_aggregation_unit": (
            "cell-weighted group_IDO_score over putative-IFN cells within each "
            "fitted cluster"
        ),
        "target_aggregation_unit": (
            "putative-IFN cell rows within each fitted cluster"
        ),
        "aggregation_unit": "putative-IFN cell rows within each fitted cluster",
        "raw_feature_medians_in_report": True,
        "report_feature_units": "raw retained feature units",
        "umap_input_units": "StandardScaler-transformed retained feature values",
        "cluster_target_medians": target_medians,
        "high_cluster": int(high_cluster),
        "low_cluster": int(low_cluster),
        "condition_warning": CONDITION_WARNING_TEXT,
        "figure_annotations": {
            "umap_by_condition.png": CONDITION_WARNING_TEXT,
        },
        "umap_learn_version": _package_version("umap-learn"),
        "scikit_learn_version": _package_version("scikit-learn"),
    }
    timing = {
        "validation_seconds": float(validation_seconds),
        "umap_seconds": float(umap_seconds),
        "kmeans_seconds": float(kmeans_seconds),
        "figures_seconds": float(figure_seconds),
        "total_seconds": float(total_seconds),
    }
    return ExploratoryAnalysisResult(
        embedding=embedding,
        cluster_assignments=cluster_table,
        cluster_feature_report=feature_report,
        figures=figures,
        timing=timing,
        provenance=provenance,
    )


def _resolve_alias(
    primary: Any,
    alias: Any,
    *,
    primary_name: str,
    alias_name: str,
) -> Any:
    if primary is not None and alias is not None:
        raise ExploratoryContractError(
            f"{primary_name} 與 {alias_name} 不可同時提供"
        )
    return primary if primary is not None else alias


def _validate_expected_counts(
    expected_cell_count: int,
    expected_fov_count: int,
    expected_group_count: int,
) -> tuple[int, int, int]:
    values = (expected_cell_count, expected_fov_count, expected_group_count)
    names = ("expected_cell_count", "expected_fov_count", "expected_group_count")
    normalised: list[int] = []
    for name, value in zip(names, values):
        if isinstance(value, bool):
            raise ExploratoryContractError(f"{name} 必須是正整數")
        try:
            integer = int(value)
        except (TypeError, ValueError) as error:
            raise ExploratoryContractError(f"{name} 必須是正整數") from error
        if integer != value or integer <= 0:
            raise ExploratoryContractError(f"{name} 必須是正整數")
        normalised.append(integer)
    return tuple(normalised)  # type: ignore[return-value]


def _prepare_population(
    retained_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str,
    expected: tuple[int, int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, int]]:
    if not isinstance(retained_cells, pd.DataFrame):
        raise ExploratoryContractError("retained_cells 必須是 pandas DataFrame")
    if not isinstance(manifest, pd.DataFrame):
        raise ExploratoryContractError("manifest 必須是 pandas DataFrame")
    feature_names = _validate_feature_roster(
        feature_columns,
        retained_cells,
        target_column=target_column,
    )
    _require_columns(retained_cells, ("image_key", "cell_label"), "retained_cells")
    _require_columns(manifest, _MANIFEST_LABEL_COLUMNS, "manifest")

    cells = retained_cells.copy()
    labels = manifest.copy()
    _normalise_image_keys(cells, "retained_cells")
    _normalise_image_keys(labels, "manifest")
    _validate_unique_cell_keys(cells)
    if labels["image_key"].duplicated().any():
        raise ExploratoryContractError("manifest image_key 必須 unique")
    _validate_labels(labels)

    target_name = _find_target_column(
        cells,
        labels,
        target_column=target_column,
    )
    if target_name not in cells.columns:
        cells = cells.merge(
            labels.loc[:, ["image_key", target_name]],
            on="image_key",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    if target_name not in cells.columns:
        raise ExploratoryContractError(
            f"retained_cells/manifest 缺少 observed target 欄位 {target_column!r}"
        )
    if target_name != "group_IDO_score":
        cells["group_IDO_score"] = cells[target_name]

    cell_keys = set(cells["image_key"])
    manifest_keys = set(labels["image_key"])
    if cell_keys != manifest_keys:
        raise ExploratoryContractError(
            "retained_cells 與 manifest 必須以 image_key many_to_one 完整 join: "
            f"retained_only={sorted(cell_keys - manifest_keys)}, "
            f"manifest_only={sorted(manifest_keys - cell_keys)}"
        )
    if len(cells) != expected[0]:
        raise ExploratoryContractError(
            f"retained unique cell count expected {expected[0]}, actual {len(cells)}"
        )
    if len(cell_keys) != expected[1] or len(labels) != expected[1]:
        raise ExploratoryContractError(
            f"FOV count expected {expected[1]}, actual retained={len(cell_keys)}, "
            f"manifest={len(labels)}"
        )
    group_count = int(labels["group_id"].nunique())
    if group_count != expected[2]:
        raise ExploratoryContractError(
            f"group count expected {expected[2]}, actual {group_count}"
        )

    _finite_numeric_values(cells, feature_names, "rui_filtered X")
    target_values = _finite_numeric_values(cells, (target_name,), "observed target")
    cells[target_name] = target_values[:, 0]
    if target_name != "group_IDO_score":
        cells["group_IDO_score"] = cells[target_name]

    # Manifest labels are authoritative for coloring and IFN selection. Drop
    # any duplicated metadata columns carried by the retained cell table so a
    # stale cell-side label cannot survive the join under a ``_manifest`` name.
    cell_values = cells.drop(
        columns=[
            column
            for column in _MANIFEST_LABEL_COLUMNS
            if column != "image_key" and column in cells.columns
        ]
    )
    joined = cell_values.merge(
        labels.loc[:, list(_MANIFEST_LABEL_COLUMNS)],
        on="image_key",
        how="left",
        validate="many_to_one",
        sort=False,
        suffixes=("", "_manifest"),
    )
    if len(joined) != expected[0] or joined.loc[:, list(_MANIFEST_LABEL_COLUMNS)].isna().any().any():
        raise ExploratoryContractError("manifest labels join 未完整覆蓋 retained cells")
    output_labels = joined.loc[
        :, ["image_key", "cell_label", *_MANIFEST_LABEL_COLUMNS[1:]]
    ].copy()
    population = {
        "cell_count": int(len(joined)),
        "fov_count": int(joined["image_key"].nunique()),
        "group_count": int(joined["group_id"].nunique()),
    }
    return joined, output_labels, target_name, population


def _validate_feature_roster(
    feature_columns: Sequence[str],
    retained_cells: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[str, ...]:
    if isinstance(feature_columns, (str, bytes)):
        raise ExploratoryContractError("rui_filtered feature list 必須是欄位序列")
    try:
        names = tuple(str(feature) for feature in feature_columns)
    except TypeError as error:
        raise ExploratoryContractError("rui_filtered feature list 必須是欄位序列") from error
    if len(names) != 30 or len(set(names)) != 30:
        raise ExploratoryContractError(
            f"rui_filtered feature list 必須恰有 30 個唯一欄位，實際 {len(names)}"
        )
    leakage = sorted(
        {
            name
            for name in names
            if name in _METADATA_COLUMNS
            or name in _TARGET_ALIASES
            or name == target_column
            or any(token in name.lower() for token in ("ido", "dapi", "nucleus"))
        }
    )
    if leakage:
        raise ExploratoryContractError(
            "rui_filtered X 不得包含 metadata/target 或 IDO/DAPI/nucleus 欄位: "
            + repr(leakage)
        )
    missing = [name for name in names if name not in retained_cells.columns]
    if missing:
        raise ExploratoryContractError(
            "rui_filtered feature 欄位不存在於 retained_cells: " + repr(missing)
        )
    return names


def _require_columns(
    frame: pd.DataFrame, required: Sequence[str], name: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ExploratoryContractError(f"{name} schema 缺少欄位: {missing}")


def _normalise_image_keys(frame: pd.DataFrame, name: str) -> None:
    if frame["image_key"].isna().any():
        raise ExploratoryContractError(f"{name} image_key 不可為空")
    frame["image_key"] = frame["image_key"].astype(str).str.strip()
    if (frame["image_key"] == "").any():
        raise ExploratoryContractError(f"{name} image_key 不可為空")


def _validate_unique_cell_keys(cells: pd.DataFrame) -> None:
    if cells["cell_label"].isna().any():
        raise ExploratoryContractError("retained_cells cell_label 不可為空")
    keys = pd.MultiIndex.from_frame(cells.loc[:, ["image_key", "cell_label"]])
    if keys.duplicated().any():
        raise ExploratoryContractError(
            "retained_cells 必須是 unique whole-cell keys (image_key, cell_label)"
        )


def _validate_labels(labels: pd.DataFrame) -> None:
    if labels.loc[:, list(_MANIFEST_LABEL_COLUMNS)].isna().any().any():
        raise ExploratoryContractError("manifest labels 不可為 null")
    for column in ("b_id", "condition", "group_id"):
        if labels[column].astype(str).str.strip().eq("").any():
            raise ExploratoryContractError(f"manifest {column} 不可為空")
    for column in ("passage", "ifn_dose", "tnf_dose"):
        try:
            values = pd.to_numeric(labels[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ExploratoryContractError(f"manifest {column} 必須是 numeric") from error
        if not np.isfinite(values).all():
            raise ExploratoryContractError(f"manifest {column} 必須是 finite")
        if column == "passage" and not np.equal(values, np.floor(values)).all():
            raise ExploratoryContractError("manifest passage 必須是整數")
        labels[column] = values.astype(int) if column == "passage" else values


def _find_target_column(
    cells: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    target_column: str,
) -> str:
    candidates = (target_column, *_TARGET_ALIASES)
    for candidate in candidates:
        if candidate in cells.columns or candidate in labels.columns:
            return candidate
    raise ExploratoryContractError(
        f"retained_cells/manifest 缺少 observed target 欄位 {target_column!r}"
    )


def _finite_numeric_values(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> np.ndarray:
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ExploratoryContractError(f"{label} 必須全部是 numeric finite") from error
    if not np.isfinite(values).all():
        raise ExploratoryContractError(f"{label} 必須全部是 finite")
    return values


def _build_feature_report(
    prepared: pd.DataFrame,
    feature_names: Sequence[str],
    ifn_mask: np.ndarray,
    cluster_ids: np.ndarray,
    target_medians: Mapping[int, float],
    *,
    high_cluster: int,
    low_cluster: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in feature_names:
        values = prepared[feature].to_numpy(dtype=np.float64)[ifn_mask]
        cluster_values = {
            cluster: values[cluster_ids == cluster]
            for cluster in range(KMEANS_CLUSTER_COUNT)
        }
        medians = {
            cluster: float(np.median(cluster_values[cluster]))
            for cluster in range(KMEANS_CLUSTER_COUNT)
        }
        high_median = medians[high_cluster]
        low_median = medians[low_cluster]
        rows.append(
            {
                "feature": feature,
                "ido_low_cluster_id": int(low_cluster),
                "ido_high_cluster_id": int(high_cluster),
                "low_cell_count": int(len(cluster_values[low_cluster])),
                "high_cell_count": int(len(cluster_values[high_cluster])),
                "low_target_median": float(target_medians[low_cluster]),
                "high_target_median": float(target_medians[high_cluster]),
                "median_ido_low": low_median,
                "median_ido_high": high_median,
                "high_minus_low": high_median - low_median,
                "absolute_difference": abs(high_median - low_median),
            }
        )
    report = pd.DataFrame(rows, columns=list(CLUSTER_FEATURE_REPORT_COLUMNS))
    return report


def _render_umap_png(
    embedding: pd.DataFrame,
    color_column: str,
    *,
    annotation: str | None = None,
) -> bytes:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.font_manager import FontProperties
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - environment-specific
        raise ExploratoryContractError("matplotlib 無法產生 exploratory PNG") from error

    figure, axis = plt.subplots(figsize=(7.0, 5.0), dpi=100)
    font = FontProperties(family="Microsoft JhengHei")
    values = embedding[color_column]
    categories = sorted(pd.unique(values), key=lambda value: (type(value).__name__, str(value)))
    colour_map = plt.get_cmap("tab20", max(len(categories), 1))
    for index, category in enumerate(categories):
        mask = values.eq(category).to_numpy()
        axis.scatter(
            embedding.loc[mask, "UMAP1"],
            embedding.loc[mask, "UMAP2"],
            s=5,
            alpha=0.75,
            color=colour_map(index),
            label=str(category),
            linewidths=0,
            rasterized=True,
        )
    axis.set_xlabel("UMAP1", fontproperties=font)
    axis.set_ylabel("UMAP2", fontproperties=font)
    axis.set_title(f"UMAP by {color_column}", fontproperties=font)
    axis.legend(
        title=color_column,
        loc="best",
        fontsize=7,
        markerscale=1.5,
        prop=font,
    )
    if annotation is not None:
        figure.text(
            0.02,
            0.01,
            annotation,
            ha="left",
            va="bottom",
            fontsize=7,
            fontproperties=font,
        )
        figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output = BytesIO()
    try:
        figure.savefig(
            output,
            format="png",
            dpi=100,
            metadata={
                "Software": "Ki67-Detection Exp4 exploratory",
                "Description": annotation or "",
            },
        )
        return output.getvalue()
    finally:
        plt.close(figure)


def _estimator_params(estimator: Any) -> dict[str, object]:
    try:
        params = estimator.get_params(deep=False)
    except AttributeError:  # pragma: no cover - defensive typing fallback
        return {}
    return {str(key): _normalise_metadata_value(value) for key, value in params.items()}


def _normalise_metadata_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return tuple(_normalise_metadata_value(item) for item in value)
    if isinstance(value, list):
        return [_normalise_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise_metadata_value(item) for key, item in value.items()}
    return value


def _package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


# Explicit aliases keep the seam discoverable without creating a second implementation.
run_umap_kmeans = run_exploratory_analysis
build_exploratory_analysis = run_exploratory_analysis
build_exploratory_result = run_exploratory_analysis


__all__ = [
    "CLUSTER_COLUMNS",
    "CLUSTER_FEATURE_REPORT_COLUMNS",
    "CONDITION_WARNING_TEXT",
    "EMBEDDING_COLUMNS",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_FOV_COUNT",
    "EXPECTED_GROUP_COUNT",
    "ExploratoryAnalysisResult",
    "ExploratoryContractError",
    "ExploratoryResult",
    "build_exploratory_analysis",
    "build_exploratory_result",
    "run_exploratory_analysis",
    "run_umap_kmeans",
]
