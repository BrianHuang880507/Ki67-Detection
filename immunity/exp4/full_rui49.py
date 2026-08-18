"""Exp4 完整 Rui49 擷取、checkpoint 與 F0 evidence 核心。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import inspect
import json
from numbers import Real
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from skimage.io import imread

from immunity.exp4 import rui_features as rui_features_module
from immunity.exp4.cell_dedup import (
    WholeCellFeatureConsistencyResult,
    assert_full_rui49_feature_consistency,
)
from immunity.exp4.feature_health import (
    FeatureHealthResult,
    evaluate_feature_health,
)
from immunity.exp4.rui_features import (
    RUI49_FEATURE_COLUMNS,
    RuiFeatureExtractionResult,
    extract_rui49_features_with_diagnostics,
    preprocess_phase_image,
)
from ki67dtc import cell_anal as cell_anal_module


IDENTITY_COLUMNS = ("image_key", "cell_label")
RAW_IDENTITY_COLUMNS = (*IDENTITY_COLUMNS, "nucleus_label")
CHECKPOINT_FORMAT_VERSION = 1
_RUI_TRANSITIVE_HELPERS = (
    "quantize_masked_texture",
    "_morphology_features",
    "_finite_or",
    "_intensity_features",
    "_haralick_features",
    "_has_four_direction_pairs",
)
_GEOMETRY_TRANSITIVE_HELPERS = (
    "_measure_roi_with_python",
    "_geometry_from_measurements",
    "_quantize_to_levels",
)
_RESULT_DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-image",
    "mahotas",
    "imageio",
    "pillow",
)


class FullRui49Error(ValueError):
    """表示 full Rui49 輸入或結果違反 fail-closed contract。"""


class FullRui49CheckpointError(FullRui49Error):
    """表示 full Rui49 checkpoint 遺失必要證據、損毀或已 stale。"""


@dataclass(frozen=True)
class FovRui49Timing:
    """記錄單一 FOV 的載入、前處理與擷取時間。

    Attributes:
        image_key: manifest 的唯一 FOV key。
        status: 本次為 ``processed`` 或 ``resumed``。
        load_seconds: 讀取 PC 與 mask 所需秒數。
        preprocess_seconds: phase 前處理所需秒數。
        extraction_seconds: Rui49 擷取所需秒數。
        total_seconds: 單一 FOV 完整處理所需秒數。
    """

    image_key: str
    status: str
    load_seconds: float
    preprocess_seconds: float
    extraction_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class FovCheckpointProvenance:
    """保存單一 FOV checkpoint 的來源與載入狀態。

    Attributes:
        image_key: manifest 的唯一 FOV key。
        checkpoint_path: noncanonical NPZ checkpoint 絕對路徑。
        status: 本次為 ``processed`` 或 ``resumed``。
        source_fingerprint: PC 與 mask size/mtime 證據的 SHA-256。
        pc_size: PC 影像 byte size。
        pc_mtime_ns: PC 影像 modification time，單位為 ns。
        mask_size: whole-cell NPZ byte size。
        mask_mtime_ns: whole-cell NPZ modification time，單位為 ns。
    """

    image_key: str
    checkpoint_path: str
    status: str
    source_fingerprint: str
    pc_size: int
    pc_mtime_ns: int
    mask_size: int
    mask_mtime_ns: int


@dataclass(frozen=True)
class FullRui49Result:
    """保存 full Rui49 canonical payload 與 correctness evidence。

    Attributes:
        raw_pair_features: raw whole-cell/nucleus pairs 加 canonical 49 欄。
        pre_border_features: border 前 distinct master whole-cells 加 49 欄。
        retained_features: border 後 retained distinct whole-cells 加 49 欄。
        retained_fallback_flags: retained cells 的逐欄 fallback 布林旗標。
        retained_fallback_counts: retained scope 的逐欄 fallback 次數。
        retained_fallback_rates: retained scope 的逐欄 fallback 比率。
        pre_border_consistency: full raw join 的 canonical exact gate 證據。
        retained_consistency: retained raw join 的 canonical exact gate 證據。
        health: retained distinct-cell scope 的完整 feature health 結果。
        healthy_feature_ranges: health allowlist 各欄的有限 min/max。
        processed_fov_count: 本次真正執行擷取的 FOV 數。
        resumed_fov_count: 本次由 checkpoint 恢復的 FOV 數。
        fov_timings: manifest stable order 的逐 FOV timing。
        total_elapsed_seconds: 本次 public call 的 wall-clock 秒數。
        extractor_fingerprint: 前處理、擷取器與 canonical schema fingerprint。
        checkpoint_provenance: manifest stable order 的 checkpoint 證據。
    """

    raw_pair_features: pd.DataFrame
    pre_border_features: pd.DataFrame
    retained_features: pd.DataFrame
    retained_fallback_flags: pd.DataFrame
    retained_fallback_counts: Mapping[str, int]
    retained_fallback_rates: Mapping[str, float]
    pre_border_consistency: WholeCellFeatureConsistencyResult
    retained_consistency: WholeCellFeatureConsistencyResult
    health: FeatureHealthResult
    healthy_feature_ranges: Mapping[str, Mapping[str, float | bool]]
    processed_fov_count: int
    resumed_fov_count: int
    fov_timings: tuple[FovRui49Timing, ...]
    total_elapsed_seconds: float
    extractor_fingerprint: str
    checkpoint_provenance: tuple[FovCheckpointProvenance, ...]


@dataclass(frozen=True)
class FullRui49ConsistencyDiagnostic:
    """描述 exact consistency gate 已通過或因 nonfinite 未執行。

    Attributes:
        status: ``passed`` 或 ``not_run_nonfinite``。
        scope: 明確的 full raw/retained count scope。
        checked_row_count: 此 scope 的 raw pair rows。
        checked_cell_count: 此 scope 的 distinct whole-cells。
        checked_duplicate_key_count: 此 scope 實測的 multirow keys。
        feature_count: canonical feature 欄數，固定為 49。
        reason: 未執行 exact gate 的結構化原因；passed 時為 ``None``。
    """

    status: str
    scope: str
    checked_row_count: int
    checked_cell_count: int
    checked_duplicate_key_count: int
    feature_count: int
    reason: str | None


@dataclass(frozen=True)
class FullRui49BlockedDiagnostics:
    """保存禁止 canonical success 時可供 CLI 落盤的完整診斷。

    Attributes:
        failure_kind: machine-readable failure 類型。
        nonfinite_columns: raw canonical payload 中含 nonfinite 的欄位。
        raw_pair_features: noncanonical raw-pair diagnostic frame。
        pre_border_features: noncanonical pre-border distinct-cell frame。
        retained_features: noncanonical retained distinct-cell frame。
        retained_fallback_flags: retained scope 逐欄 fallback flags。
        retained_fallback_counts: retained scope 逐欄 fallback counts。
        retained_fallback_rates: retained scope 逐欄 fallback rates。
        pre_border_consistency: raw exact gate status 與 count evidence。
        retained_consistency: retained exact gate status 與 count evidence。
        health: full retained-scope structured health result。
        healthy_feature_ranges: health allowlist 中仍 finite 欄位的 ranges。
        processed_fov_count: 本次實際擷取 FOV 數。
        resumed_fov_count: 本次 resume FOV 數。
        fov_timings: manifest stable order 的 timing。
        total_elapsed_seconds: 本次 call wall-clock 秒數。
        extractor_fingerprint: 完整 extractor/source fingerprint。
        checkpoint_provenance: 逐 FOV noncanonical checkpoint evidence。
    """

    failure_kind: str
    nonfinite_columns: tuple[str, ...]
    raw_pair_features: pd.DataFrame
    pre_border_features: pd.DataFrame
    retained_features: pd.DataFrame
    retained_fallback_flags: pd.DataFrame
    retained_fallback_counts: Mapping[str, int]
    retained_fallback_rates: Mapping[str, float]
    pre_border_consistency: FullRui49ConsistencyDiagnostic
    retained_consistency: FullRui49ConsistencyDiagnostic
    health: FeatureHealthResult
    healthy_feature_ranges: Mapping[str, Mapping[str, float | bool]]
    processed_fov_count: int
    resumed_fov_count: int
    fov_timings: tuple[FovRui49Timing, ...]
    total_elapsed_seconds: float
    extractor_fingerprint: str
    checkpoint_provenance: tuple[FovCheckpointProvenance, ...]


class FullRui49CanonicalBlockedError(FullRui49Error):
    """表示 nonfinite 已禁止 canonical success，但 full 診斷可用。"""

    def __init__(self, diagnostics: FullRui49BlockedDiagnostics) -> None:
        """建立攜帶 full-scope noncanonical diagnostics 的 typed failure。

        Args:
            diagnostics: CLI 可安全落盤的完整 structured failure evidence。
        """
        self.diagnostics = diagnostics
        columns = ", ".join(diagnostics.nonfinite_columns)
        super().__init__(f"canonical Rui49 blocked by nonfinite columns: {columns}")


@dataclass(frozen=True)
class _SourceSnapshot:
    """保存 checkpoint stale gate 所需的不可變檔案 metadata。"""

    image_key: str
    pc_path: Path
    mask_path: Path
    pc_size: int
    pc_mtime_ns: int
    mask_size: int
    mask_mtime_ns: int
    fingerprint: str


def extract_full_rui49(
    *,
    manifest: pd.DataFrame,
    raw_cells: pd.DataFrame,
    retained_cells: pd.DataFrame,
    masks_dir: str | Path,
    checkpoint_dir: str | Path,
    progress_log_path: str | Path,
    expected_fov_count: int = 693,
    expected_raw_pair_rows: int = 23_976,
    expected_pre_border_cells: int = 23_012,
    expected_retained_cells: int = 19_648,
    expected_raw_multirow_keys: int = 938,
    expected_retained_raw_pair_rows: int = 20_440,
) -> FullRui49Result:
    """擷取完整 Rui49，建立 resumable checkpoints 與 exact F0 evidence。

    此 deep core 只讀 manifest 指定的 PC 影像與既有 ``.npz`` whole-cell
    masks。它不發布 canonical CSV/metadata/report，也不執行 segmentation、
    predictor、Pearson、CV 或模型。mask-only labels 可擷取供 checkpoint
    完整性使用，但不會進入 master payload、health 或 fallback denominator。

    Args:
        manifest: 含唯一 ``image_key`` 與絕對 ``pc_path`` 的 FOV roster。
        raw_cells: raw ``image_key × cell_label × nucleus_label`` pair rows。
        retained_cells: border 後每個 whole-cell 恰一列的 key roster。
        masks_dir: 既有 whole-cell NPZ 根目錄，只會索引 ``.npz``。
        checkpoint_dir: Exp4 output 下名稱以 ``.`` 開頭的 hidden 目錄。
        progress_log_path: 每完成或恢復一個 FOV 即 append/flush 的 JSONL。
        expected_fov_count: manifest 與 retained FOV 的 fail-closed lock。
        expected_raw_pair_rows: raw pair row count lock。
        expected_pre_border_cells: border 前 distinct whole-cell count lock。
        expected_retained_cells: border 後 distinct whole-cell count lock。
        expected_raw_multirow_keys: full raw join multirow key count lock。
        expected_retained_raw_pair_rows: retained raw join row count lock。

    Returns:
        canonical raw/pre-border/retained frames、fallback、兩次 exact gate、
        full retained health、ranges、timing 與 checkpoint provenance。

    Raises:
        TypeError: 三個 roster 輸入不是 pandas DataFrame。
        FullRui49Error: schema、count、coverage、finite 或 join contract 失敗。
        FullRui49CheckpointError: checkpoint stale、malformed 或 schema 不符。
        OSError: PC、mask、checkpoint 或 progress log 無法安全讀寫。
    """
    started = perf_counter()
    expected = _validate_expected_counts(
        expected_fov_count=expected_fov_count,
        expected_raw_pair_rows=expected_raw_pair_rows,
        expected_pre_border_cells=expected_pre_border_cells,
        expected_retained_cells=expected_retained_cells,
        expected_raw_multirow_keys=expected_raw_multirow_keys,
        expected_retained_raw_pair_rows=expected_retained_raw_pair_rows,
    )
    normalized_manifest = _normalize_manifest(
        manifest,
        expected_fov_count=expected["expected_fov_count"],
    )
    normalized_raw = _normalize_raw_cells(
        raw_cells,
        normalized_manifest,
        expected_raw_pair_rows=expected["expected_raw_pair_rows"],
        expected_pre_border_cells=expected["expected_pre_border_cells"],
        expected_raw_multirow_keys=expected["expected_raw_multirow_keys"],
    )
    normalized_retained = _normalize_retained_cells(
        retained_cells,
        normalized_manifest,
        normalized_raw,
        expected_retained_cells=expected["expected_retained_cells"],
    )
    retained_key_index = pd.MultiIndex.from_frame(
        normalized_retained.loc[:, IDENTITY_COLUMNS]
    )
    raw_key_index = pd.MultiIndex.from_frame(
        normalized_raw.loc[:, IDENTITY_COLUMNS]
    )
    retained_raw_pair_count = int(raw_key_index.isin(retained_key_index).sum())
    if retained_raw_pair_count != expected["expected_retained_raw_pair_rows"]:
        raise FullRui49Error(
            "retained raw pair row count 不符："
            f"expected {expected['expected_retained_raw_pair_rows']}, "
            f"actual {retained_raw_pair_count}"
        )

    masks_root = Path(masks_dir).expanduser().resolve(strict=True)
    if not masks_root.is_dir():
        raise FullRui49Error(f"masks_dir 不是目錄：{masks_root}")
    mask_index = _index_npz_masks(masks_root)
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve(strict=False)
    progress_path = Path(progress_log_path).expanduser().resolve(strict=False)
    _validate_output_paths(checkpoint_root, progress_path, masks_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    extractor_fingerprint = _extractor_fingerprint()
    feature_frames: list[pd.DataFrame] = []
    fallback_frames: list[pd.DataFrame] = []
    timings: list[FovRui49Timing] = []
    provenance: list[FovCheckpointProvenance] = []
    processed_fov_count = 0
    resumed_fov_count = 0
    raw_labels_by_key = {
        image_key: set(group["cell_label"].tolist())
        for image_key, group in normalized_raw.groupby("image_key", sort=False)
    }

    for row in normalized_manifest.itertuples(index=False):
        image_key = str(row.image_key)
        if image_key not in mask_index:
            raise FullRui49Error(f"manifest FOV 缺少 whole-cell mask：{image_key}")
        pc_path = Path(str(row.pc_path)).expanduser().resolve(strict=True)
        source = _snapshot_source(image_key, pc_path, mask_index[image_key])
        checkpoint_path = _checkpoint_path(checkpoint_root, image_key)
        if checkpoint_path.exists():
            features, fallback_flags, timing = _load_checkpoint(
                checkpoint_path,
                source=source,
                extractor_fingerprint=extractor_fingerprint,
            )
            _validate_source_unchanged(source)
            extracted_labels = set(features["cell_label"].tolist())
            missing_master_labels = sorted(
                raw_labels_by_key[image_key] - extracted_labels
            )
            if missing_master_labels:
                raise FullRui49CheckpointError(
                    f"{image_key} checkpoint 缺少 master label："
                    f"{missing_master_labels[:10]}"
                )
            keyed_features = features.copy()
            keyed_features.insert(0, "image_key", image_key)
            keyed_fallback = fallback_flags.copy()
            keyed_fallback.insert(0, "image_key", image_key)
            checkpoint_item = _checkpoint_provenance(
                source,
                checkpoint_path,
                status="resumed",
            )
            _append_progress(
                progress_path,
                image_key=image_key,
                status="resumed",
                checkpoint_path=checkpoint_path,
                source_fingerprint=source.fingerprint,
                timing=timing,
            )
            feature_frames.append(keyed_features)
            fallback_frames.append(keyed_fallback)
            timings.append(timing)
            provenance.append(checkpoint_item)
            resumed_fov_count += 1
            continue

        fov_started = perf_counter()
        load_started = perf_counter()
        rgb_image = np.asarray(imread(pc_path))
        cell_labels = _load_whole_cell_labels(source.mask_path)
        load_seconds = perf_counter() - load_started
        preprocess_started = perf_counter()
        phase_signal = preprocess_phase_image(rgb_image)
        preprocess_seconds = perf_counter() - preprocess_started
        extraction_started = perf_counter()
        extraction = extract_rui49_features_with_diagnostics(
            phase_signal,
            cell_labels,
            include_legacy_texture=False,
        )
        extraction_seconds = perf_counter() - extraction_started
        _validate_source_unchanged(source)
        features, fallback_flags = _validate_extraction(
            image_key,
            extraction,
        )
        extracted_labels = set(features["cell_label"].tolist())
        missing_master_labels = sorted(
            raw_labels_by_key[image_key] - extracted_labels
        )
        if missing_master_labels:
            raise FullRui49Error(
                f"{image_key} master label 缺少 Rui49 feature："
                f"{missing_master_labels[:10]}"
            )

        keyed_features = features.copy()
        keyed_features.insert(0, "image_key", image_key)
        keyed_fallback = fallback_flags.copy()
        keyed_fallback.insert(0, "image_key", image_key)
        total_seconds = perf_counter() - fov_started
        timing = FovRui49Timing(
            image_key=image_key,
            status="processed",
            load_seconds=float(load_seconds),
            preprocess_seconds=float(preprocess_seconds),
            extraction_seconds=float(extraction_seconds),
            total_seconds=float(total_seconds),
        )
        _write_checkpoint_atomic(
            checkpoint_path,
            source=source,
            extractor_fingerprint=extractor_fingerprint,
            features=features,
            fallback_flags=fallback_flags,
            timing=timing,
        )
        checkpoint_item = _checkpoint_provenance(
            source,
            checkpoint_path,
            status="processed",
        )
        _append_progress(
            progress_path,
            image_key=image_key,
            status="processed",
            checkpoint_path=checkpoint_path,
            source_fingerprint=source.fingerprint,
            timing=timing,
        )
        feature_frames.append(keyed_features)
        fallback_frames.append(keyed_fallback)
        timings.append(timing)
        provenance.append(checkpoint_item)
        processed_fov_count += 1

    all_features = pd.concat(feature_frames, ignore_index=True)
    all_fallback_flags = pd.concat(fallback_frames, ignore_index=True)
    _validate_unique_feature_keys(all_features, name="Rui49 features")
    _validate_unique_feature_keys(all_fallback_flags, name="fallback flags")

    pre_border = _join_distinct_features(normalized_raw, all_features)
    pre_border_fallback = _join_distinct_fallback(
        normalized_raw,
        all_fallback_flags,
    )
    raw_pair_features = normalized_raw.merge(
        pre_border,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="many_to_one",
        sort=False,
    ).loc[:, (*RAW_IDENTITY_COLUMNS, *RUI49_FEATURE_COLUMNS)]
    if len(raw_pair_features) != expected["expected_raw_pair_rows"]:
        raise FullRui49Error("full raw Rui49 join 未保留全部 pair rows")
    pre_scope = (
        f"full_raw_join_{expected['expected_raw_pair_rows']}_pairs_"
        f"{expected['expected_pre_border_cells']}_cells_"
        f"{expected['expected_fov_count']}_fovs"
    )
    pre_consistency, pre_consistency_diagnostic = _run_consistency_gate(
        raw_pair_features,
        scope=pre_scope,
    )
    if (
        pre_consistency_diagnostic.checked_duplicate_key_count
        != expected["expected_raw_multirow_keys"]
    ):
        raise FullRui49Error(
            "full raw Rui49 multirow key count 不符："
            f"expected {expected['expected_raw_multirow_keys']}, "
            "actual "
            f"{pre_consistency_diagnostic.checked_duplicate_key_count}"
        )

    retained = normalized_retained.merge(
        pre_border,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    ).loc[:, (*IDENTITY_COLUMNS, *RUI49_FEATURE_COLUMNS)]
    retained_fallback_flags = normalized_retained.merge(
        pre_border_fallback,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    ).loc[:, (*IDENTITY_COLUMNS, *RUI49_FEATURE_COLUMNS)]
    if retained_fallback_flags.loc[:, RUI49_FEATURE_COLUMNS].isna().any().any():
        raise FullRui49Error("retained whole-cell 缺少 fallback flag")
    retained_fallback_flags.loc[:, RUI49_FEATURE_COLUMNS] = (
        retained_fallback_flags.loc[:, RUI49_FEATURE_COLUMNS].astype(bool)
    )

    raw_key_index = pd.MultiIndex.from_frame(
        raw_pair_features.loc[:, IDENTITY_COLUMNS]
    )
    retained_raw = raw_pair_features.loc[
        raw_key_index.isin(retained_key_index)
    ].reset_index(drop=True)
    retained_scope = (
        f"full_retained_raw_join_"
        f"{expected['expected_retained_raw_pair_rows']}_pairs_"
        f"{expected['expected_retained_cells']}_cells_"
        f"{expected['expected_fov_count']}_fovs"
    )
    retained_consistency, retained_consistency_diagnostic = _run_consistency_gate(
        retained_raw,
        scope=retained_scope,
    )

    fallback_counts = {
        column: int(retained_fallback_flags[column].sum())
        for column in RUI49_FEATURE_COLUMNS
    }
    fallback_rates = {
        column: count / expected["expected_retained_cells"]
        for column, count in fallback_counts.items()
    }
    health_scope = (
        f"full_retained_{expected['expected_retained_cells']}_"
        f"distinct_cells_{expected['expected_fov_count']}_fovs"
    )
    health = evaluate_feature_health(
        retained.loc[:, RUI49_FEATURE_COLUMNS],
        fallback_counts=fallback_counts,
        eligible_cell_count=expected["expected_retained_cells"],
        feature_columns=RUI49_FEATURE_COLUMNS,
        fallback_scope=health_scope,
    )
    healthy_ranges = _feature_ranges(
        retained,
        health.healthy_feature_columns,
    )
    nonfinite_columns = _nonfinite_feature_columns(raw_pair_features)
    if nonfinite_columns:
        diagnostics = FullRui49BlockedDiagnostics(
            failure_kind="blocked_nonfinite",
            nonfinite_columns=nonfinite_columns,
            raw_pair_features=raw_pair_features.reset_index(drop=True),
            pre_border_features=pre_border.reset_index(drop=True),
            retained_features=retained.reset_index(drop=True),
            retained_fallback_flags=retained_fallback_flags.reset_index(
                drop=True
            ),
            retained_fallback_counts=fallback_counts,
            retained_fallback_rates=fallback_rates,
            pre_border_consistency=pre_consistency_diagnostic,
            retained_consistency=retained_consistency_diagnostic,
            health=health,
            healthy_feature_ranges=healthy_ranges,
            processed_fov_count=processed_fov_count,
            resumed_fov_count=resumed_fov_count,
            fov_timings=tuple(timings),
            total_elapsed_seconds=float(perf_counter() - started),
            extractor_fingerprint=extractor_fingerprint,
            checkpoint_provenance=tuple(provenance),
        )
        raise FullRui49CanonicalBlockedError(diagnostics)
    if pre_consistency is None or retained_consistency is None:
        raise AssertionError("finite Rui49 scope 必須完成 exact consistency gate")
    return FullRui49Result(
        raw_pair_features=raw_pair_features.reset_index(drop=True),
        pre_border_features=pre_border.reset_index(drop=True),
        retained_features=retained.reset_index(drop=True),
        retained_fallback_flags=retained_fallback_flags.reset_index(drop=True),
        retained_fallback_counts=fallback_counts,
        retained_fallback_rates=fallback_rates,
        pre_border_consistency=pre_consistency,
        retained_consistency=retained_consistency,
        health=health,
        healthy_feature_ranges=healthy_ranges,
        processed_fov_count=processed_fov_count,
        resumed_fov_count=resumed_fov_count,
        fov_timings=tuple(timings),
        total_elapsed_seconds=float(perf_counter() - started),
        extractor_fingerprint=extractor_fingerprint,
        checkpoint_provenance=tuple(provenance),
    )


def _validate_expected_counts(**values: int) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise FullRui49Error(f"{name} 必須是整數")
        numeric = int(value)
        minimum = 0 if name == "expected_raw_multirow_keys" else 1
        if numeric < minimum:
            raise FullRui49Error(f"{name} 必須大於或等於 {minimum}")
        normalized[name] = numeric
    return normalized


def _normalize_manifest(
    manifest: pd.DataFrame,
    *,
    expected_fov_count: int,
) -> pd.DataFrame:
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest 必須是 pandas DataFrame")
    _require_columns(manifest, ("image_key", "pc_path"), "manifest")
    frame = manifest.loc[:, ["image_key", "pc_path"]].copy()
    keys = frame["image_key"].astype("string").str.strip()
    if keys.isna().any() or keys.eq("").any():
        raise FullRui49Error("manifest image_key 不可為空")
    if keys.duplicated().any():
        raise FullRui49Error("manifest image_key 不可重複")
    paths = frame["pc_path"].astype("string").str.strip()
    if paths.isna().any() or paths.eq("").any():
        raise FullRui49Error("manifest pc_path 不可為空")
    relative_paths = [
        str(value) for value in paths.tolist() if not Path(str(value)).is_absolute()
    ]
    if relative_paths:
        raise FullRui49Error(
            "manifest pc_path 必須是絕對路徑：" + repr(relative_paths[:10])
        )
    frame = frame.assign(image_key=keys.astype(str), pc_path=paths.astype(str))
    frame = frame.sort_values("image_key", kind="stable").reset_index(drop=True)
    if len(frame) != expected_fov_count:
        raise FullRui49Error(
            "manifest FOV count 不符："
            f"expected {expected_fov_count}, actual {len(frame)}"
        )
    return frame


def _normalize_raw_cells(
    raw_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    expected_raw_pair_rows: int,
    expected_pre_border_cells: int,
    expected_raw_multirow_keys: int,
) -> pd.DataFrame:
    if not isinstance(raw_cells, pd.DataFrame):
        raise TypeError("raw_cells 必須是 pandas DataFrame")
    _require_columns(raw_cells, RAW_IDENTITY_COLUMNS, "raw_cells")
    frame = raw_cells.loc[:, RAW_IDENTITY_COLUMNS].copy()
    frame["image_key"] = _normalize_image_keys(frame["image_key"], "raw_cells")
    frame["cell_label"] = _positive_integer_labels(
        frame["cell_label"],
        "raw_cells cell_label",
    )
    frame["nucleus_label"] = _positive_integer_labels(
        frame["nucleus_label"],
        "raw_cells nucleus_label",
    )
    if frame.duplicated(list(RAW_IDENTITY_COLUMNS)).any():
        raise FullRui49Error("raw_cells pair key 不可重複")
    if len(frame) != expected_raw_pair_rows:
        raise FullRui49Error(
            "raw pair row count 不符："
            f"expected {expected_raw_pair_rows}, actual {len(frame)}"
        )
    manifest_keys = set(manifest["image_key"])
    if set(frame["image_key"]) != manifest_keys:
        raise FullRui49Error("raw_cells FOV coverage 必須與 manifest 完全相同")
    distinct_count = len(frame.loc[:, IDENTITY_COLUMNS].drop_duplicates())
    if distinct_count != expected_pre_border_cells:
        raise FullRui49Error(
            "pre-border distinct cell count 不符："
            f"expected {expected_pre_border_cells}, actual {distinct_count}"
        )
    multirow_count = int(
        (
            frame.groupby(list(IDENTITY_COLUMNS), sort=False).size()
            > 1
        ).sum()
    )
    if multirow_count != expected_raw_multirow_keys:
        raise FullRui49Error(
            "raw multirow key count 不符："
            f"expected {expected_raw_multirow_keys}, actual {multirow_count}"
        )
    order = {key: index for index, key in enumerate(manifest["image_key"])}
    frame = frame.assign(
        _manifest_order=frame["image_key"].map(order),
        _raw_order=np.arange(len(frame)),
    ).sort_values(
        ["_manifest_order", "cell_label", "_raw_order"],
        kind="stable",
    )
    return frame.loc[:, RAW_IDENTITY_COLUMNS].reset_index(drop=True)


def _normalize_retained_cells(
    retained_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    raw_cells: pd.DataFrame,
    *,
    expected_retained_cells: int,
) -> pd.DataFrame:
    if not isinstance(retained_cells, pd.DataFrame):
        raise TypeError("retained_cells 必須是 pandas DataFrame")
    _require_columns(retained_cells, IDENTITY_COLUMNS, "retained_cells")
    frame = retained_cells.loc[:, IDENTITY_COLUMNS].copy()
    frame["image_key"] = _normalize_image_keys(
        frame["image_key"],
        "retained_cells",
    )
    frame["cell_label"] = _positive_integer_labels(
        frame["cell_label"],
        "retained_cells cell_label",
    )
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise FullRui49Error("retained_cells whole-cell key 不可重複")
    if len(frame) != expected_retained_cells:
        raise FullRui49Error(
            "retained distinct cell count 不符："
            f"expected {expected_retained_cells}, actual {len(frame)}"
        )
    manifest_keys = set(manifest["image_key"])
    if set(frame["image_key"]) != manifest_keys:
        raise FullRui49Error("retained_cells 必須涵蓋全部 manifest FOV")
    raw_keys = pd.MultiIndex.from_frame(raw_cells.loc[:, IDENTITY_COLUMNS])
    retained_keys = pd.MultiIndex.from_frame(frame.loc[:, IDENTITY_COLUMNS])
    missing = retained_keys.difference(raw_keys)
    if len(missing):
        raise FullRui49Error(
            "retained_cells 含 raw master 外的 key："
            + repr(list(missing[:10]))
        )
    order = {key: index for index, key in enumerate(manifest["image_key"])}
    frame = frame.assign(_manifest_order=frame["image_key"].map(order)).sort_values(
        ["_manifest_order", "cell_label"],
        kind="stable",
    )
    return frame.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)


def _normalize_image_keys(values: pd.Series, name: str) -> pd.Series:
    keys = values.astype("string").str.strip()
    if keys.isna().any() or keys.eq("").any():
        raise FullRui49Error(f"{name} image_key 不可為空")
    return keys.astype(str)


def _positive_integer_labels(values: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or np.any(numeric <= 0)
    ):
        raise FullRui49Error(f"{name} 必須是 positive integers")
    return numeric.astype(np.int64)


def _index_npz_masks(masks_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(
        (
            candidate
            for candidate in masks_root.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == ".npz"
        ),
        key=lambda candidate: candidate.as_posix().lower(),
    ):
        if path.stem in index:
            raise FullRui49Error(
                f"mask stem 重複：{path.stem}: {index[path.stem]}, {path}"
            )
        index[path.stem] = path.resolve(strict=True)
    return index


def _validate_output_paths(
    checkpoint_root: Path,
    progress_path: Path,
    masks_root: Path,
) -> None:
    if not checkpoint_root.name.startswith("."):
        raise FullRui49CheckpointError(
            "checkpoint_dir 必須是名稱以 '.' 開頭的 Exp4 hidden output dir"
        )
    if _is_relative_to(checkpoint_root, masks_root):
        raise FullRui49CheckpointError("checkpoint_dir 不得位於 mask cache 內")
    if _is_relative_to(progress_path, masks_root):
        raise FullRui49CheckpointError("progress_log_path 不得位於 mask cache 內")
    if checkpoint_root.exists() and not checkpoint_root.is_dir():
        raise FullRui49CheckpointError("checkpoint_dir 已存在但不是目錄")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _snapshot_source(
    image_key: str,
    pc_path: Path,
    mask_path: Path,
) -> _SourceSnapshot:
    pc_stat = pc_path.stat()
    mask_stat = mask_path.stat()
    payload = {
        "image_key": image_key,
        "pc_path": str(pc_path),
        "pc_size": int(pc_stat.st_size),
        "pc_mtime_ns": int(pc_stat.st_mtime_ns),
        "mask_path": str(mask_path),
        "mask_size": int(mask_stat.st_size),
        "mask_mtime_ns": int(mask_stat.st_mtime_ns),
    }
    fingerprint = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return _SourceSnapshot(
        image_key=image_key,
        pc_path=pc_path,
        mask_path=mask_path,
        pc_size=payload["pc_size"],
        pc_mtime_ns=payload["pc_mtime_ns"],
        mask_size=payload["mask_size"],
        mask_mtime_ns=payload["mask_mtime_ns"],
        fingerprint=fingerprint,
    )


def _validate_source_unchanged(source: _SourceSnapshot) -> None:
    current = _snapshot_source(
        source.image_key,
        source.pc_path,
        source.mask_path,
    )
    if current.fingerprint != source.fingerprint:
        raise FullRui49CheckpointError(
            f"{source.image_key} PC/mask 在擷取期間變更，拒絕寫 checkpoint"
        )


def _load_whole_cell_labels(mask_path: Path) -> np.ndarray:
    with np.load(mask_path, allow_pickle=False) as loaded:
        if "cell_mask" not in loaded:
            raise FullRui49Error(f"mask NPZ 缺少 cell_mask：{mask_path}")
        return np.asarray(loaded["cell_mask"]).copy()


def _validate_extraction(
    image_key: str,
    extraction: RuiFeatureExtractionResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(extraction, RuiFeatureExtractionResult):
        raise FullRui49Error(f"{image_key} extractor 回傳型別不符")
    expected_columns = ["cell_label", *RUI49_FEATURE_COLUMNS]
    features = extraction.features.copy()
    fallback_flags = extraction.fallback_flags.copy()
    if features.columns.tolist() != expected_columns:
        raise FullRui49Error(f"{image_key} extractor canonical 欄位不符")
    if fallback_flags.columns.tolist() != expected_columns:
        raise FullRui49Error(f"{image_key} fallback canonical 欄位不符")
    feature_labels = _positive_integer_labels(
        features["cell_label"],
        f"{image_key} feature cell_label",
    )
    fallback_labels = _positive_integer_labels(
        fallback_flags["cell_label"],
        f"{image_key} fallback cell_label",
    )
    if len(features) == 0:
        raise FullRui49Error(f"{image_key} mask 沒有 positive whole-cell label")
    if extraction.cell_count != len(features):
        raise FullRui49Error(f"{image_key} extractor cell_count 不一致")
    if len(fallback_flags) != len(features):
        raise FullRui49Error(f"{image_key} feature/fallback row count 不一致")
    if not np.array_equal(feature_labels, fallback_labels):
        raise FullRui49Error(f"{image_key} feature/fallback label 順序不一致")
    if pd.Series(feature_labels).duplicated().any():
        raise FullRui49Error(f"{image_key} extractor cell_label 不可重複")
    features["cell_label"] = feature_labels
    fallback_flags["cell_label"] = fallback_labels
    try:
        numeric = features.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise FullRui49Error(f"{image_key} Rui49 feature 必須是數值") from error
    features.loc[:, RUI49_FEATURE_COLUMNS] = numeric
    fallback_flags.loc[:, RUI49_FEATURE_COLUMNS] = _validated_fallback_values(
        fallback_flags,
        image_key=image_key,
    )
    return features, fallback_flags


def _validated_fallback_values(
    fallback_flags: pd.DataFrame,
    *,
    image_key: str,
) -> np.ndarray:
    """逐值驗證 fresh extractor flags 並回傳 canonical boolean matrix。"""
    values = fallback_flags.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(
        dtype=object
    )
    validated = np.empty(values.shape, dtype=bool)
    for row_index, column_index in np.ndindex(values.shape):
        value = values[row_index, column_index]
        if isinstance(value, (bool, np.bool_)):
            validated[row_index, column_index] = bool(value)
            continue
        if isinstance(value, Real):
            numeric = float(value)
            if np.isfinite(numeric) and numeric in (0.0, 1.0):
                validated[row_index, column_index] = bool(numeric)
                continue
        column = RUI49_FEATURE_COLUMNS[column_index]
        raise FullRui49Error(
            f"{image_key} fallback flag 必須是 bool 或 numeric 0/1："
            f"row={row_index}, column={column}, value={value!r}"
        )
    return validated


def _extractor_fingerprint() -> str:
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "canonical_columns": list(RUI49_FEATURE_COLUMNS),
        "python": platform.python_version(),
        "dependencies": {
            distribution: _distribution_version(distribution)
            for distribution in _RESULT_DEPENDENCY_DISTRIBUTIONS
        },
        "rui_constants": {
            "background_radius": rui_features_module.BACKGROUND_RADIUS,
            "morphology_feature_names": list(
                rui_features_module.MORPHOLOGY_FEATURE_NAMES
            ),
            "intensity_feature_names": list(
                rui_features_module.INTENSITY_FEATURE_NAMES
            ),
            "haralick_feature_names": list(
                rui_features_module.HARALICK_FEATURE_NAMES
            ),
        },
        "preprocess": _callable_source(preprocess_phase_image),
        "extractor": _callable_source(
            extract_rui49_features_with_diagnostics
        ),
        "rui_transitive_helpers": {
            name: _callable_source(getattr(rui_features_module, name))
            for name in _RUI_TRANSITIVE_HELPERS
        },
        "geometry_transitive_helpers": {
            name: _callable_source(getattr(cell_anal_module, name))
            for name in _GEOMETRY_TRANSITIVE_HELPERS
        },
        "source_module_sha256": {
            "immunity.exp4.rui_features": _module_source_digest(
                rui_features_module
            ),
            "ki67dtc.cell_anal": _module_source_digest(cell_anal_module),
        },
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _callable_source(function: Any) -> str:
    identity = f"{function.__module__}.{function.__qualname__}"
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        if code is None:
            raise FullRui49CheckpointError(
                f"無法 fingerprint callable：{identity}"
            )
        source = json.dumps(
            {
                "bytecode": code.co_code.hex(),
                "names": list(code.co_names),
                "consts": [repr(value) for value in code.co_consts],
            },
            sort_keys=True,
        )
    return identity + "\n" + source


def _distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _module_source_digest(module: Any) -> str:
    source_path = Path(str(module.__file__)).resolve(strict=True)
    return sha256(source_path.read_bytes()).hexdigest()


def _checkpoint_path(checkpoint_root: Path, image_key: str) -> Path:
    digest = sha256(image_key.encode("utf-8")).hexdigest()
    return checkpoint_root / f"fov-{digest}.npz"


def _write_checkpoint_atomic(
    path: Path,
    *,
    source: _SourceSnapshot,
    extractor_fingerprint: str,
    features: pd.DataFrame,
    fallback_flags: pd.DataFrame,
    timing: FovRui49Timing,
) -> None:
    """以同目錄 temp file + replace 原子寫入 non-pickle NPZ checkpoint。"""
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                format_version=np.asarray(
                    CHECKPOINT_FORMAT_VERSION,
                    dtype=np.int64,
                ),
                image_key=np.asarray(source.image_key, dtype=np.str_),
                pc_path=np.asarray(str(source.pc_path), dtype=np.str_),
                mask_path=np.asarray(str(source.mask_path), dtype=np.str_),
                pc_size=np.asarray(source.pc_size, dtype=np.int64),
                pc_mtime_ns=np.asarray(source.pc_mtime_ns, dtype=np.int64),
                mask_size=np.asarray(source.mask_size, dtype=np.int64),
                mask_mtime_ns=np.asarray(source.mask_mtime_ns, dtype=np.int64),
                source_fingerprint=np.asarray(
                    source.fingerprint,
                    dtype=np.str_,
                ),
                extractor_fingerprint=np.asarray(
                    extractor_fingerprint,
                    dtype=np.str_,
                ),
                feature_columns=np.asarray(
                    RUI49_FEATURE_COLUMNS,
                    dtype=np.str_,
                ),
                cell_labels=features["cell_label"].to_numpy(dtype=np.int64),
                features=features.loc[:, RUI49_FEATURE_COLUMNS].to_numpy(
                    dtype=np.float64
                ),
                fallback_flags=fallback_flags.loc[
                    :, RUI49_FEATURE_COLUMNS
                ].to_numpy(dtype=np.uint8),
                timing_seconds=np.asarray(
                    [
                        timing.load_seconds,
                        timing.preprocess_seconds,
                        timing.extraction_seconds,
                        timing.total_seconds,
                    ],
                    dtype=np.float64,
                ),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_checkpoint(
    path: Path,
    *,
    source: _SourceSnapshot,
    extractor_fingerprint: str,
) -> tuple[pd.DataFrame, pd.DataFrame, FovRui49Timing]:
    """安全載入並完整驗證單一 FOV NPZ checkpoint。

    Args:
        path: noncanonical checkpoint 路徑。
        source: 本次呼叫重新量測的 PC/mask size 與 mtime。
        extractor_fingerprint: 本次執行中的 extractor/source fingerprint。

    Returns:
        canonical feature frame、fallback flags 與標記為 resumed 的 timing。

    Raises:
        FullRui49CheckpointError: checkpoint 缺欄、stale、非安全 dtype、
            canonical schema 或數值 contract 不符。
    """
    required = {
        "format_version",
        "image_key",
        "pc_path",
        "mask_path",
        "pc_size",
        "pc_mtime_ns",
        "mask_size",
        "mask_mtime_ns",
        "source_fingerprint",
        "extractor_fingerprint",
        "feature_columns",
        "cell_labels",
        "features",
        "fallback_flags",
        "timing_seconds",
    }
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != required:
                missing = sorted(required - set(loaded.files))
                unexpected = sorted(set(loaded.files) - required)
                raise FullRui49CheckpointError(
                    "checkpoint schema 不符："
                    f"missing={missing}, unexpected={unexpected}"
                )
            arrays = {name: np.asarray(loaded[name]) for name in required}
        if any(array.dtype.kind == "O" for array in arrays.values()):
            raise FullRui49CheckpointError("checkpoint 不得含 object/pickle dtype")
        if _checkpoint_scalar_int(arrays["format_version"], "format_version") != (
            CHECKPOINT_FORMAT_VERSION
        ):
            raise FullRui49CheckpointError("checkpoint format_version 不支援")
        if _checkpoint_scalar_str(arrays["image_key"], "image_key") != (
            source.image_key
        ):
            raise FullRui49CheckpointError("checkpoint image_key stale")
        if _checkpoint_scalar_str(arrays["pc_path"], "pc_path") != str(
            source.pc_path
        ):
            raise FullRui49CheckpointError("checkpoint PC path stale")
        if _checkpoint_scalar_str(arrays["mask_path"], "mask_path") != str(
            source.mask_path
        ):
            raise FullRui49CheckpointError("checkpoint mask path stale")
        source_integer_fields = {
            "pc_size": source.pc_size,
            "pc_mtime_ns": source.pc_mtime_ns,
            "mask_size": source.mask_size,
            "mask_mtime_ns": source.mask_mtime_ns,
        }
        for name, expected_value in source_integer_fields.items():
            actual = _checkpoint_scalar_int(arrays[name], name)
            if actual != expected_value:
                raise FullRui49CheckpointError(
                    f"checkpoint {name} stale："
                    f"expected {expected_value}, actual {actual}"
                )
        if _checkpoint_scalar_str(
            arrays["source_fingerprint"],
            "source_fingerprint",
        ) != source.fingerprint:
            raise FullRui49CheckpointError("checkpoint source fingerprint stale")
        if _checkpoint_scalar_str(
            arrays["extractor_fingerprint"],
            "extractor_fingerprint",
        ) != extractor_fingerprint:
            raise FullRui49CheckpointError(
                "checkpoint extractor/source fingerprint stale"
            )
        columns = arrays["feature_columns"]
        if columns.ndim != 1 or tuple(columns.astype(str).tolist()) != tuple(
            RUI49_FEATURE_COLUMNS
        ):
            raise FullRui49CheckpointError(
                "checkpoint canonical feature_columns 不符"
            )
        labels = arrays["cell_labels"]
        feature_values = arrays["features"]
        fallback_values = arrays["fallback_flags"]
        if labels.ndim != 1 or labels.dtype.kind not in "iu":
            raise FullRui49CheckpointError(
                "checkpoint cell_labels 必須是一維 integer"
            )
        row_count = len(labels)
        expected_shape = (row_count, len(RUI49_FEATURE_COLUMNS))
        if feature_values.shape != expected_shape:
            raise FullRui49CheckpointError("checkpoint features shape 不符")
        if feature_values.dtype.kind not in "fiu":
            raise FullRui49CheckpointError("checkpoint features 必須是 numeric")
        if fallback_values.shape != expected_shape:
            raise FullRui49CheckpointError("checkpoint fallback_flags shape 不符")
        if fallback_values.dtype.kind not in "biu":
            raise FullRui49CheckpointError(
                "checkpoint fallback_flags 必須是 bool/integer"
            )
        if not np.isin(fallback_values, (0, 1)).all():
            raise FullRui49CheckpointError(
                "checkpoint fallback_flags 只能是 0/1"
            )
        timing_values = arrays["timing_seconds"]
        if (
            timing_values.shape != (4,)
            or timing_values.dtype.kind not in "fiu"
            or not np.isfinite(timing_values).all()
            or np.any(timing_values < 0.0)
        ):
            raise FullRui49CheckpointError(
                "checkpoint timing_seconds 必須是四個非負 finite 數"
            )
        features = pd.DataFrame(
            feature_values.astype(np.float64, copy=False),
            columns=RUI49_FEATURE_COLUMNS,
        )
        features.insert(0, "cell_label", labels.astype(np.int64, copy=False))
        fallback_flags = pd.DataFrame(
            fallback_values.astype(bool, copy=False),
            columns=RUI49_FEATURE_COLUMNS,
        )
        fallback_flags.insert(
            0,
            "cell_label",
            labels.astype(np.int64, copy=False),
        )
        extraction = RuiFeatureExtractionResult(
            features=features,
            legacy_texture_features=None,
            fallback_flags=fallback_flags,
            cell_count=row_count,
        )
        validated_features, validated_fallback = _validate_extraction(
            source.image_key,
            extraction,
        )
        timing = FovRui49Timing(
            image_key=source.image_key,
            status="resumed",
            load_seconds=float(timing_values[0]),
            preprocess_seconds=float(timing_values[1]),
            extraction_seconds=float(timing_values[2]),
            total_seconds=float(timing_values[3]),
        )
        return validated_features, validated_fallback, timing
    except FullRui49CheckpointError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise FullRui49CheckpointError(
            f"checkpoint malformed，拒絕 resume：{path}: {error}"
        ) from error


def _checkpoint_scalar_int(array: np.ndarray, name: str) -> int:
    if array.shape != () or array.dtype.kind not in "iu":
        raise FullRui49CheckpointError(f"checkpoint {name} 必須是 integer scalar")
    return int(array.item())


def _checkpoint_scalar_str(array: np.ndarray, name: str) -> str:
    if array.shape != () or array.dtype.kind not in "US":
        raise FullRui49CheckpointError(f"checkpoint {name} 必須是 string scalar")
    return str(array.item())


def _checkpoint_provenance(
    source: _SourceSnapshot,
    checkpoint_path: Path,
    *,
    status: str,
) -> FovCheckpointProvenance:
    return FovCheckpointProvenance(
        image_key=source.image_key,
        checkpoint_path=str(checkpoint_path),
        status=status,
        source_fingerprint=source.fingerprint,
        pc_size=source.pc_size,
        pc_mtime_ns=source.pc_mtime_ns,
        mask_size=source.mask_size,
        mask_mtime_ns=source.mask_mtime_ns,
    )


def _append_progress(
    path: Path,
    *,
    image_key: str,
    status: str,
    checkpoint_path: Path,
    source_fingerprint: str,
    timing: FovRui49Timing,
) -> None:
    """Append 並 fsync 單一 FOV progress record，供中斷後人工追蹤。"""
    payload = {
        "image_key": image_key,
        "status": status,
        "checkpoint_path": str(checkpoint_path),
        "source_fingerprint": source_fingerprint,
        "timing_seconds": {
            "load": timing.load_seconds,
            "preprocess": timing.preprocess_seconds,
            "extraction": timing.extraction_seconds,
            "total": timing.total_seconds,
        },
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _validate_unique_feature_keys(frame: pd.DataFrame, *, name: str) -> None:
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise FullRui49Error(f"{name} whole-cell key 不可重複")


def _join_distinct_features(
    raw_cells: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    keys = raw_cells.loc[:, IDENTITY_COLUMNS].drop_duplicates().reset_index(
        drop=True
    )
    master_index = pd.MultiIndex.from_frame(keys)
    feature_index = pd.MultiIndex.from_frame(
        features.loc[:, IDENTITY_COLUMNS]
    )
    missing = master_index.difference(feature_index)
    if len(missing):
        raise FullRui49Error(
            "master label 缺少 Rui49 feature：" + repr(list(missing[:10]))
        )
    joined = keys.merge(
        features,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    ).loc[:, (*IDENTITY_COLUMNS, *RUI49_FEATURE_COLUMNS)]
    return joined


def _join_distinct_fallback(
    raw_cells: pd.DataFrame,
    fallback_flags: pd.DataFrame,
) -> pd.DataFrame:
    keys = raw_cells.loc[:, IDENTITY_COLUMNS].drop_duplicates().reset_index(
        drop=True
    )
    joined = keys.merge(
        fallback_flags,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    ).loc[:, (*IDENTITY_COLUMNS, *RUI49_FEATURE_COLUMNS)]
    if joined.loc[:, RUI49_FEATURE_COLUMNS].isna().any().any():
        raise FullRui49Error("master label 缺少 fallback flag")
    joined.loc[:, RUI49_FEATURE_COLUMNS] = joined.loc[
        :, RUI49_FEATURE_COLUMNS
    ].astype(bool)
    return joined


def _feature_ranges(
    features: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, float | bool]]:
    ranges: dict[str, dict[str, float | bool]] = {}
    for column in columns:
        values = features[column].to_numpy(dtype=np.float64)
        finite = bool(values.size > 0 and np.isfinite(values).all())
        if not finite:
            raise FullRui49Error(f"healthy feature range 非 finite：{column}")
        ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "finite": finite,
        }
    return ranges


def _run_consistency_gate(
    joined_cells: pd.DataFrame,
    *,
    scope: str,
) -> tuple[
    WholeCellFeatureConsistencyResult | None,
    FullRui49ConsistencyDiagnostic,
]:
    """在 finite 時執行 exact gate，否則回傳明確的未執行證據。"""
    nonfinite_columns = _nonfinite_feature_columns(joined_cells)
    if nonfinite_columns:
        key_sizes = joined_cells.groupby(
            list(IDENTITY_COLUMNS),
            sort=False,
            dropna=False,
        ).size()
        diagnostic = FullRui49ConsistencyDiagnostic(
            status="not_run_nonfinite",
            scope=scope,
            checked_row_count=len(joined_cells),
            checked_cell_count=int(len(key_sizes)),
            checked_duplicate_key_count=int((key_sizes > 1).sum()),
            feature_count=len(RUI49_FEATURE_COLUMNS),
            reason="nonfinite columns: " + ", ".join(nonfinite_columns),
        )
        return None, diagnostic
    evidence = assert_full_rui49_feature_consistency(
        joined_cells,
        scope=scope,
    )
    return evidence, FullRui49ConsistencyDiagnostic(
        status="passed",
        scope=evidence.scope,
        checked_row_count=evidence.checked_row_count,
        checked_cell_count=evidence.checked_cell_count,
        checked_duplicate_key_count=evidence.checked_duplicate_key_count,
        feature_count=evidence.feature_count,
        reason=None,
    )


def _nonfinite_feature_columns(features: pd.DataFrame) -> tuple[str, ...]:
    """回傳依 canonical 順序含任一 nonfinite 值的 feature 欄名。"""
    return tuple(
        column
        for column in RUI49_FEATURE_COLUMNS
        if not np.isfinite(features[column].to_numpy(dtype=np.float64)).all()
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FullRui49Error(f"{name} 缺少欄位：{missing}")
