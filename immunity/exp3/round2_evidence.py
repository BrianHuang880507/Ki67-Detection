"""載入並驗證 Exp3 Round 2 使用的凍結 Round 1 證據。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from immunity.exp3.benchmark import (
    OuterSplit,
    make_outer_splits,
    outer_split_manifest,
)
from immunity.exp3.feature_sets import PRIMARY_CELL_FEATURES, PRIMARY_FOV_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ROUND2_OUTPUT_ROOT = (
    PROJECT_ROOT / "immunity" / "outputs" / "exp3" / "round2_paper93"
).resolve(strict=False)
_REQUIRED_ARTIFACTS = (
    "pairing_qc.csv",
    "data_manifest.csv",
    "segmentation_qc.csv",
    "outer_splits.csv",
    "feature_cache/image_level_basic.csv",
    "feature_cache/cell_level_basic.csv",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "model_ranking.csv",
    "hyperparameters.csv",
    "feature_importance.csv",
    "model_failures.csv",
    "feature_sets.json",
    "run_metadata.json",
)
_FROZEN_ARTIFACT_HASHES = {
    "pairing_qc.csv": "0fa953eceb35678aab0209214eb1ee267e844a0b81e649b0c0acca487246e0eb",
    "data_manifest.csv": "b4508aaf830f4c456759d1f02458606aa28ed385856575c6fb99a5ac2e163771",
    "segmentation_qc.csv": "2a1e8969fc8d0a81d3dd737ef99df57f58ce3e1faf0206e833463f2dc9d28c1e",
    "outer_splits.csv": "c7e9e26a9ad0c6f73d8f14c5fa4c73747f6e58865aac963bf8e16e527fa05a3e",
    "feature_cache/image_level_basic.csv": "dad479258d08846d05a43f551b0001a52d96fa5e2e8ea6a98645723931aa0d14",
    "feature_cache/cell_level_basic.csv": "7d999a728d848ccb61a3910cb5230c9efde6d7e1fa14f22de792990ee4ad7850",
    "fold_metrics.csv": "8ce0df3866822a54c98cb1cd95dbec3c3a8ffa130af54890692b240250091884",
    "oof_predictions.csv": "637242dc0f45c4dc436643214c21ac451fc560d7ae36b25a6db04aafe5d6a2c8",
    "model_ranking.csv": "d279ae973d5ab7508e97460bb1b6f94dd95345ce27d4cca0ad2f587516cb72e4",
    "hyperparameters.csv": "5da9c936b573126d21417752d0fd140306562ed387b299364a5cd356268bfe00",
    "feature_importance.csv": "17cf4bafdb1d7b87654d7efd58ef79584f1324c40abac3cb7bdfc617a510b9f6",
    "model_failures.csv": "b1ce311d21be365f1e5a010d9eb16aaea997e5d843e5b630aef77c91b52d98ea",
    "feature_sets.json": "49e65affebbab6285170e45048eee3991bfc70e1a53454d7e6bbc94a092e28cd",
    "run_metadata.json": "d0cdfabea5f9243eeadf7eb2278019f9497a75c3e82ba129b4077e1f62566b9f",
}
_FROZEN_ROSTER_HASH = (
    "a8333f12e1591d9e4c4174f5c6fe13dd31550124b19aea3522f4d0f812e0e426"
)
_FROZEN_IMAGE_KEYS_SHA256 = (
    "efc55b45033a3923f1467dce86802f6514a0e9c182de77e94a2c155d47503232"
)
_MODEL_CONTRACT = {
    "extra_trees": ("candidate", "basic_median"),
    "random_forest": ("candidate", "basic_median"),
    "dummy_median": ("diagnostic", "none"),
}
_FOLD_COUNTS = {
    "leave_one_b_out": 3,
    "leave_one_passage_out": 3,
    "leave_one_group_out": 9,
    "leave_one_condition_out": 8,
}
_IMAGE_METADATA = (
    "b_id",
    "passage",
    "group_id",
    "condition_index",
    "condition",
    "ifn_dose",
    "tnf_dose",
    "fov",
)
_MASK_QC_COLUMNS = (
    "image_key",
    "pc_path",
    "mask_path",
    "expected_pc_sha256",
    "actual_pc_sha256",
    "expected_provenance_hash",
    "actual_provenance_hash",
    "status",
    "reason",
)
_SEMANTIC_DIGEST_ATTR = "round1_frozen_semantic_sha256"


class _EvidenceMetadata(dict[str, Any]):
    """保存 mutable metadata 與其 loader-time canonical semantic digest。"""

    def __init__(self, value: Mapping[str, Any]) -> None:
        super().__init__(value)
        self.frozen_semantic_sha256 = _metadata_semantic_sha256(self)


@dataclass(frozen=True)
class Round1Evidence:
    """保存已通過 bytes 與語意驗證的 Round 1 唯讀證據。

    Attributes:
        root: Round 1 Exp3 輸出根目錄。
        manifest: 已分析影像 manifest。
        segmentation_qc: 凍結 segmentation 與 cache 證據。
        basic_cells: 33-feature cell-level 表格。
        basic_images: 33-feature image-level median 表格。
        split_manifest: 凍結 outer split membership。
        selected_metrics: ET、RF 與 Dummy 的 Round 1 fold metrics。
        selected_predictions: ET、RF 與 Dummy 的 Round 1 OOF 預測。
        selected_hyperparameters: 指定模型的 fold hyperparameters。
        selected_importance: ET 與 RF 的 feature importance。
        selected_failures: 指定模型的 failure evidence。
        model_ranking: Round 1 model ranking。
        metadata: Round 1 run metadata。
        artifact_hashes: 實際計算的來源 artifact SHA-256。
    """

    root: Path
    manifest: pd.DataFrame
    segmentation_qc: pd.DataFrame
    basic_cells: pd.DataFrame
    basic_images: pd.DataFrame
    split_manifest: pd.DataFrame
    selected_metrics: pd.DataFrame
    selected_predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    selected_importance: pd.DataFrame
    selected_failures: pd.DataFrame
    model_ranking: pd.DataFrame
    metadata: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]


def canonical_roster_sha256(cells: pd.DataFrame) -> str:
    """計算 cell identity roster 的 canonical SHA-256。

    Args:
        cells: 含 image_key、cell_label 與 nucleus_label 的 cell 表格。

    Returns:
        排序並轉成固定 CSV bytes 後的小寫 SHA-256。

    Raises:
        ValueError: 必要欄位缺失或 label 不能精確轉為整數時拋出。
    """
    _require_columns(cells, ("image_key", "cell_label", "nucleus_label"), "basic cells")
    roster = cells.loc[:, ["image_key", "cell_label", "nucleus_label"]].copy()
    roster["image_key"] = roster["image_key"].astype(str)
    for column in ("cell_label", "nucleus_label"):
        numeric = pd.to_numeric(roster[column], errors="raise")
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"roster {column} 必須是有限整數")
        roster[column] = numeric.astype(int)
    roster = roster.sort_values(
        ["image_key", "cell_label", "nucleus_label"], kind="stable"
    )
    payload = roster.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_round1_evidence(config: Mapping[str, Any]) -> Round1Evidence:
    """載入固定 bytes，並在回傳前驗證 Round 1 語意 identity。

    Args:
        config: 含 round1.dir、artifact_sha256、roster_sha256 與 expected 的設定。

    Returns:
        僅保留 ET、RF 與 Dummy primary Round 1 結果的證據物件。

    Raises:
        ValueError: 路徑、bytes hash、schema 或跨 artifact 語意不一致時拋出。
    """
    round1 = _mapping(config, "round1")
    root_value = round1.get("dir")
    if not isinstance(root_value, (str, Path)):
        raise ValueError("round1.dir 必須是路徑")
    root = Path(root_value)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve(strict=True)
    if root == _ROUND2_OUTPUT_ROOT or _ROUND2_OUTPUT_ROOT in root.parents:
        raise ValueError("round1.dir 必須是 Exp3 parent root，不可指向 Round 2 output")
    if not root.is_dir():
        raise ValueError(f"round1.dir 不是目錄：{root}")

    pins = round1.get("artifact_sha256")
    if not isinstance(pins, Mapping) or set(pins) != set(_REQUIRED_ARTIFACTS):
        raise ValueError("round1 artifact_sha256 必須精確列出 14 個 artifacts")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for relative in _REQUIRED_ARTIFACTS:
        lexical_path = root / Path(relative)
        if _contains_reparse_component(root, lexical_path):
            raise ValueError(
                f"Round 1 artifact {relative} 必須是 root 內 regular file"
            )
        try:
            path = lexical_path.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"Round 1 artifact {relative} 必須是 root 內 regular file"
            ) from error
        if root not in path.parents or not _is_regular_file(path):
            raise ValueError(f"Round 1 artifact {relative} 必須是 root 內 regular file")
        actual = _sha256_file(path)
        expected_hash = pins[relative]
        if not isinstance(expected_hash, str) or actual != expected_hash.lower():
            raise ValueError(
                f"Round 1 artifact {relative} SHA-256 不符合 pin："
                f"expected={expected_hash}, actual={actual}"
            )
        paths[relative] = path
        hashes[relative] = actual

    manifest = pd.read_csv(paths["data_manifest.csv"])
    segmentation_qc = pd.read_csv(paths["segmentation_qc.csv"])
    basic_cells = pd.read_csv(paths["feature_cache/cell_level_basic.csv"])
    basic_images = pd.read_csv(paths["feature_cache/image_level_basic.csv"])
    split_manifest = pd.read_csv(paths["outer_splits.csv"])
    metrics = pd.read_csv(paths["fold_metrics.csv"])
    predictions = pd.read_csv(paths["oof_predictions.csv"])
    hyperparameters = pd.read_csv(paths["hyperparameters.csv"])
    importance = pd.read_csv(paths["feature_importance.csv"])
    failures = pd.read_csv(paths["model_failures.csv"])
    model_ranking = pd.read_csv(paths["model_ranking.csv"])
    feature_sets = _read_json(paths["feature_sets.json"])
    metadata = _read_json(paths["run_metadata.json"])

    _validate_basic_tables(manifest, segmentation_qc, basic_cells, basic_images, feature_sets)
    roster_hash = canonical_roster_sha256(basic_cells)
    roster_pin = round1.get("roster_sha256")
    if not isinstance(roster_pin, str) or roster_hash != roster_pin.lower():
        raise ValueError(
            f"Round 1 roster SHA-256 不符合 pin：expected={roster_pin}, actual={roster_hash}"
        )
    _validate_basic_medians(basic_cells, basic_images)
    _validate_metadata(round1, metadata, hashes, manifest)

    selected_metrics = _select_primary(metrics, include_dummy=True, table="fold_metrics")
    selected_predictions = _select_primary(predictions, include_dummy=True, table="oof_predictions")
    selected_hyperparameters = _select_primary(
        hyperparameters, include_dummy=True, table="hyperparameters"
    )
    selected_importance = _select_primary(
        importance, include_dummy=False, table="feature_importance"
    )
    selected_failures = _select_primary(failures, include_dummy=True, table="model_failures")
    _validate_selected_evidence(
        selected_metrics,
        selected_predictions,
        basic_images,
        split_manifest,
    )
    _freeze_semantic_frames(
        {
            "manifest": manifest,
            "segmentation_qc": segmentation_qc,
            "basic_cells": basic_cells,
            "basic_images": basic_images,
            "split_manifest": split_manifest,
            "selected_metrics": selected_metrics,
            "selected_predictions": selected_predictions,
            "selected_hyperparameters": selected_hyperparameters,
            "selected_importance": selected_importance,
            "selected_failures": selected_failures,
            "model_ranking": model_ranking,
        }
    )
    frozen_metadata = _EvidenceMetadata(metadata)
    return Round1Evidence(
        root=root,
        manifest=manifest,
        segmentation_qc=segmentation_qc,
        basic_cells=basic_cells,
        basic_images=basic_images,
        split_manifest=split_manifest,
        selected_metrics=selected_metrics,
        selected_predictions=selected_predictions,
        selected_hyperparameters=selected_hyperparameters,
        selected_importance=selected_importance,
        selected_failures=selected_failures,
        model_ranking=model_ranking,
        metadata=frozen_metadata,
        artifact_hashes=hashes,
    )


def require_formal_round1_evidence(evidence: Round1Evidence) -> None:
    """套用不可由 config 放寬的正式 Round 1 production gate。

    Args:
        evidence: 已通過一般 loader 驗證的完整 Round 1 證據。

    Raises:
        ValueError: 來源 hash、row counts、roster、fold 或模型證據偏離正式快照時拋出。
    """
    if dict(evidence.artifact_hashes) != _FROZEN_ARTIFACT_HASHES:
        raise ValueError("formal Round 1 source SHA-256 不符合固定 14-artifact snapshot")
    _require_semantic_frames_unchanged(
        {
            "manifest": evidence.manifest,
            "segmentation_qc": evidence.segmentation_qc,
            "basic_cells": evidence.basic_cells,
            "basic_images": evidence.basic_images,
            "split_manifest": evidence.split_manifest,
            "selected_metrics": evidence.selected_metrics,
            "selected_predictions": evidence.selected_predictions,
            "selected_hyperparameters": evidence.selected_hyperparameters,
            "selected_importance": evidence.selected_importance,
            "selected_failures": evidence.selected_failures,
            "model_ranking": evidence.model_ranking,
        }
    )
    _require_metadata_unchanged(evidence.metadata)
    if len(evidence.basic_images) != 693 or evidence.basic_images["image_key"].nunique() != 693:
        raise ValueError("formal Round 1 必須含 693 unique images")
    if len(evidence.basic_cells) != 23976:
        raise ValueError("formal Round 1 必須含 23,976 cells")
    if canonical_roster_sha256(evidence.basic_cells) != _FROZEN_ROSTER_HASH:
        raise ValueError("formal Round 1 roster SHA-256 不符合固定 snapshot")
    if _metadata_seed(evidence.metadata) != 20260804:
        raise ValueError("formal Round 1 seed 必須是 20260804")

    pairs = evidence.split_manifest.loc[:, ["validation", "fold"]].drop_duplicates()
    counts = pairs.groupby("validation", sort=False)["fold"].size().to_dict()
    if counts != _FOLD_COUNTS or len(pairs) != 23:
        raise ValueError(f"formal Round 1 fold contract 必須是 3/3/9/8，共 23 folds：{counts}")
    _validate_selected_evidence(
        evidence.selected_metrics,
        evidence.selected_predictions,
        evidence.basic_images,
        evidence.split_manifest,
    )
    restore_frozen_outer_splits(evidence.basic_images, evidence.split_manifest)
    expected_ids = {
        f"{str(row.validation)}:{str(row.fold)}" for row in pairs.itertuples(index=False)
    }
    for model in _MODEL_CONTRACT:
        model_metrics = evidence.selected_metrics[evidence.selected_metrics["model"].eq(model)]
        metric_ids = set(model_metrics["split_id"].astype(str))
        if len(model_metrics) != 23 or metric_ids != expected_ids or not model_metrics["status"].eq("ok").all():
            raise ValueError(f"formal Round 1 {model} 必須有 23 個 successful folds")
        model_oof = evidence.selected_predictions[evidence.selected_predictions["model"].eq(model)]
        if len(model_oof) != 2772:
            raise ValueError(f"formal Round 1 {model} 必須有 2,772 OOF rows")
    if not evidence.selected_failures.empty:
        raise ValueError("formal Round 1 不允許 model failure evidence")


def validate_frozen_masks(
    evidence: Round1Evidence,
    image_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """唯讀驗證 PC bytes、mask cache 與 embedded provenance。

    Args:
        evidence: 已驗證的 Round 1 evidence。
        image_keys: 要驗證的 image keys；省略時驗證完整 manifest。

    Returns:
        每張影像一列且不直接 raise cache failure 的 provenance QC。

    Raises:
        ValueError: 指定 key 重複、未知或 evidence 表格 identity 不完整時拋出。
    """
    _require_columns(evidence.manifest, ("image_key", "group_id", "pc_path"), "manifest")
    _require_columns(
        evidence.segmentation_qc,
        ("image_key", "mask_path", "pc_sha256", "cache_provenance_hash"),
        "segmentation_qc",
    )
    manifest = evidence.manifest.set_index(evidence.manifest["image_key"].astype(str), drop=False)
    qc = evidence.segmentation_qc.set_index(
        evidence.segmentation_qc["image_key"].astype(str), drop=False
    )
    if manifest.index.has_duplicates or qc.index.has_duplicates:
        raise ValueError("mask evidence image_key 不可重複")
    keys = list(manifest.index) if image_keys is None else [str(key) for key in image_keys]
    if len(keys) != len(set(keys)):
        raise ValueError("image_keys 不可重複")
    unknown = sorted(set(keys) - set(manifest.index))
    if unknown:
        raise ValueError(f"image_keys 含未知 image_key：{unknown}")
    missing_qc = sorted(set(keys) - set(qc.index))
    if missing_qc:
        raise ValueError(f"segmentation_qc 缺少 image_key：{missing_qc}")

    rows: list[dict[str, Any]] = []
    for image_key in keys:
        manifest_row = manifest.loc[image_key]
        qc_row = qc.loc[image_key]
        group_id = str(manifest_row["group_id"])
        _safe_path_component(group_id, "group_id")
        _safe_path_component(image_key, "image_key")
        pc_path = Path(str(manifest_row["pc_path"])).resolve(strict=False)
        lexical_mask_path = (
            evidence.root
            / "feature_cache"
            / "masks"
            / group_id
            / f"{image_key}.npz"
        )
        mask_path = lexical_mask_path.resolve(strict=False)
        expected_pc = str(qc_row["pc_sha256"])
        expected_provenance = str(qc_row["cache_provenance_hash"])
        result = {
            "image_key": image_key,
            "pc_path": str(pc_path),
            "mask_path": str(mask_path),
            "expected_pc_sha256": expected_pc,
            "actual_pc_sha256": "",
            "expected_provenance_hash": expected_provenance,
            "actual_provenance_hash": "",
            "status": "failed",
            "reason": "",
        }
        try:
            if _contains_reparse_component(evidence.root, lexical_mask_path):
                raise ValueError("mask path 不可包含 symlink 或 junction")
            if Path(str(qc_row["mask_path"])).resolve(strict=False) != mask_path:
                raise ValueError("segmentation_qc.mask_path 與 root 推導路徑不一致")
            actual_pc = _sha256_file(pc_path)
            result["actual_pc_sha256"] = actual_pc
            if actual_pc != expected_pc:
                raise ValueError("PC SHA-256 不一致")
            with np.load(mask_path, allow_pickle=False) as cached:
                if "cell_mask" not in cached or "nucleus_mask" not in cached:
                    raise ValueError("mask cache 缺少 cell_mask 或 nucleus_mask")
                cell_mask = np.asarray(cached["cell_mask"])
                nucleus_mask = np.asarray(cached["nucleus_mask"])
                if cell_mask.ndim != 2 or nucleus_mask.ndim != 2 or cell_mask.shape != nucleus_mask.shape:
                    raise ValueError("mask arrays 必須是相同 shape 的二維陣列")
                provenance_json = _npz_text(cached, "provenance_json")
                embedded_hash = _npz_text(cached, "provenance_hash")
            result["actual_provenance_hash"] = embedded_hash
            calculated_hash = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
            if embedded_hash != calculated_hash or embedded_hash != expected_provenance:
                raise ValueError("mask provenance hash 不一致")
            provenance = _strict_json_object(provenance_json, "mask provenance")
            if _canonical_json(provenance) != provenance_json:
                raise ValueError("mask provenance 必須是 canonical JSON")
            if provenance.get("pc_sha256") != actual_pc:
                raise ValueError("mask embedded pc_sha256 與 PC bytes 不一致")
            result["status"] = "passed"
            result["reason"] = "verified"
        except Exception as error:  # noqa: BLE001 - 每張影像須保留完整 failure QC
            result["reason"] = (
                f"image_key={image_key}; pc_path={pc_path}; mask_path={mask_path}; "
                f"{type(error).__name__}: {error}"
            )
        rows.append(result)
    return pd.DataFrame(rows, columns=_MASK_QC_COLUMNS)


def restore_frozen_outer_splits(
    images: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> tuple[OuterSplit, ...]:
    """從 693-image artifact 還原並精確比對正式 outer splits。

    Args:
        images: 完整 693-image formal snapshot。
        split_manifest: Round 1 outer_splits.csv 內容。

    Returns:
        依 source artifact 首次出現順序排列的 positional splits。

    Raises:
        ValueError: Schema、fold counts、membership 或 metadata 偏離凍結證據時拋出。
    """
    _require_columns(images, ("image_key", *_IMAGE_METADATA), "images")
    _require_columns(
        split_manifest,
        ("validation", "fold", "image_key", "role", *_IMAGE_METADATA),
        "split manifest",
    )
    if len(images) != 693 or images["image_key"].astype(str).nunique() != 693:
        raise ValueError("frozen split restoration 只接受完整 693 unique images")
    image_keys = images["image_key"].astype(str)
    if image_keys.duplicated().any():
        raise ValueError("images 出現重複 image_key")
    image_key_hash = _canonical_image_keys_sha256(image_keys)
    if image_key_hash != _FROZEN_IMAGE_KEYS_SHA256:
        raise ValueError(
            "formal Round 1 image-key roster SHA-256 不符合固定 snapshot："
            f"actual={image_key_hash}"
        )
    source = split_manifest.copy()
    source["validation"] = source["validation"].astype(str)
    source["fold"] = source["fold"].astype(str)
    source["image_key"] = source["image_key"].astype(str)
    identity = ["validation", "fold", "image_key"]
    if source.duplicated(identity).any():
        raise ValueError("split manifest 出現重複 membership row")
    pairs = source.loc[:, ["validation", "fold"]].drop_duplicates()
    counts = pairs.groupby("validation", sort=False)["fold"].size().to_dict()
    if counts != _FOLD_COUNTS:
        raise ValueError(f"split family fold counts 必須是 3/3/9/8：{counts}")

    expected_key_set = set(image_keys)
    positions = {key: position for position, key in enumerate(image_keys)}
    image_by_key = images.copy()
    image_by_key.index = image_keys
    splits: list[OuterSplit] = []
    for pair in pairs.itertuples(index=False):
        validation, fold = str(pair.validation), str(pair.fold)
        rows = source[source["validation"].eq(validation) & source["fold"].eq(fold)]
        actual_keys = set(rows["image_key"])
        if len(rows) != 693 or actual_keys != expected_key_set:
            unknown = sorted(actual_keys - expected_key_set)
            raise ValueError(
                f"split {validation}:{fold} 未完整覆蓋 693 image_key；unknown={unknown}"
            )
        if not set(rows["role"]) <= {"train", "test"}:
            raise ValueError(f"split {validation}:{fold} role 只允許 train/test")
        for column in _IMAGE_METADATA:
            expected = rows["image_key"].map(image_by_key[column])
            try:
                pd.testing.assert_series_equal(
                    rows[column].reset_index(drop=True),
                    expected.reset_index(drop=True),
                    check_names=False,
                    check_dtype=False,
                )
            except AssertionError as error:
                raise ValueError(
                    f"split {validation}:{fold} metadata {column} 不符合 images"
                ) from error
        train_keys = rows.loc[rows["role"].eq("train"), "image_key"]
        test_keys = rows.loc[rows["role"].eq("test"), "image_key"]
        if train_keys.empty or test_keys.empty or set(train_keys) & set(test_keys):
            raise ValueError(f"split {validation}:{fold} train/test membership 不合法")
        splits.append(
            OuterSplit(
                validation=validation,
                fold=fold,
                train_index=np.asarray([positions[key] for key in train_keys], dtype=np.int64),
                test_index=np.asarray([positions[key] for key in test_keys], dtype=np.int64),
            )
        )
    for family in _FOLD_COUNTS:
        family_test = source.loc[source["validation"].eq(family) & source["role"].eq("test"), "image_key"]
        if Counter(family_test) != Counter({key: 1 for key in image_keys}):
            raise ValueError(f"{family} 每個 image_key 必須恰好一次 test membership")

    restored = outer_split_manifest(images, splits)
    compare_columns = ["validation", "fold", "image_key", "role", *_IMAGE_METADATA]
    pair_order = {
        (str(row.validation), str(row.fold)): index
        for index, row in enumerate(pairs.itertuples(index=False))
    }
    expected = source.loc[:, compare_columns].copy()
    expected["_pair_order"] = [
        pair_order[(validation, fold)]
        for validation, fold in zip(
            expected["validation"], expected["fold"], strict=True
        )
    ]
    expected["_image_order"] = expected["image_key"].map(positions)
    expected = (
        expected.sort_values(["_pair_order", "_image_order"], kind="stable")
        .drop(columns=["_pair_order", "_image_order"])
        .reset_index(drop=True)
    )
    actual = restored.loc[:, compare_columns].copy()
    actual["fold"] = actual["fold"].astype(str)
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError("restored split membership 與 frozen source artifact 不一致") from error
    canonical = outer_split_manifest(images, make_outer_splits(images)).loc[
        :, compare_columns
    ]
    canonical["validation"] = canonical["validation"].astype(str)
    canonical["fold"] = canonical["fold"].astype(str)
    canonical["image_key"] = canonical["image_key"].astype(str)
    source_membership = source.loc[:, compare_columns].sort_values(
        ["validation", "fold", "image_key"], kind="stable"
    )
    canonical_membership = canonical.sort_values(
        ["validation", "fold", "image_key"], kind="stable"
    )
    try:
        pd.testing.assert_frame_equal(
            source_membership.reset_index(drop=True),
            canonical_membership.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError(
            "frozen source 不符合 canonical grouped outer membership"
        ) from error
    return tuple(splits)


def make_smoke_outer_splits(
    images: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[OuterSplit, ...]:
    """建立 deterministic、non-formal 的 smoke diagnostic splits。

    Args:
        images: 已由完整正式證據擷取的 smoke image subset。
        config: 含 smoke 設定的 Round 2 config；僅用來確認呼叫者明示 smoke mode。

    Returns:
        四個 validation families 的 diagnostic OuterSplit tuple。

    Raises:
        ValueError: Smoke 設定、split identity 或 train/test 安全性不完整時拋出。

    Note:
        回傳結果是 ``diagnostic_splits``、``non_formal``，不得用於
        ``scientific_conclusion``。
    """
    if not isinstance(config.get("smoke"), Mapping):
        raise ValueError("smoke diagnostic splits 需要明確 smoke config")
    splits = tuple(make_outer_splits(images))
    ids = [f"{split.validation}:{split.fold}" for split in splits]
    if len(ids) != len(set(ids)):
        raise ValueError("smoke split_id 不可重複")
    if set(split.validation for split in splits) != set(_FOLD_COUNTS):
        raise ValueError("smoke splits 必須包含四個 validation families")
    for split in splits:
        if not len(split.train_index) or not len(split.test_index):
            raise ValueError("smoke split train/test 不可為空")
        if np.intersect1d(split.train_index, split.test_index).size:
            raise ValueError("smoke split 不可有 train/test leakage")
    return splits


def expected_split_ids(splits: Sequence[OuterSplit]) -> dict[str, list[str]]:
    """依輸入順序將 split IDs 分組成 validation family mapping。

    Args:
        splits: Outer splits。

    Returns:
        每個 family 對應的 ``validation:fold`` ID 清單。

    Raises:
        ValueError: Split identity 重複時拋出。
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for split in splits:
        split_id = f"{split.validation}:{split.fold}"
        if split_id in seen:
            raise ValueError(f"split_id 重複：{split_id}")
        seen.add(split_id)
        grouped[str(split.validation)].append(split_id)
    return dict(grouped)


def _validate_basic_tables(
    manifest: pd.DataFrame,
    segmentation_qc: pd.DataFrame,
    cells: pd.DataFrame,
    images: pd.DataFrame,
    feature_sets: Any,
) -> None:
    """驗證基礎影像、cell 與 feature-set schema。"""
    if any(frame.empty for frame in (manifest, segmentation_qc, cells, images)):
        raise ValueError("Round 1 image/cell/basic evidence 不可為空")
    _require_columns(manifest, ("image_key", *_IMAGE_METADATA, "pc_path"), "manifest")
    _require_columns(segmentation_qc, ("image_key", "group_id"), "segmentation_qc")
    expected_cell_columns = {"image_key", "cell_label", "nucleus_label", *PRIMARY_CELL_FEATURES, "IDO_score"}
    if set(cells.columns) != expected_cell_columns:
        raise ValueError("cell-level basic 必須精確包含 33 basic columns 與 identity/target")
    expected_image_columns = {"image_key", *_IMAGE_METADATA, "cell_count", "IDO_score", *PRIMARY_FOV_FEATURES}
    if set(images.columns) != expected_image_columns:
        raise ValueError("image-level basic 必須精確包含 33 basic median columns")
    expected_feature_set = {
        "predictor_columns": list(PRIMARY_FOV_FEATURES),
        "predictor_count": 33,
        "aggregation": "median",
        "replication_scope": "primary_phase_only",
    }
    if feature_sets != {"basic_median": expected_feature_set}:
        raise ValueError("feature_sets basic_median 不符合固定 33-feature schema")
    frame_sets = []
    for name, frame in (("manifest", manifest), ("segmentation_qc", segmentation_qc), ("basic images", images)):
        keys = frame["image_key"].astype(str)
        if keys.duplicated().any():
            raise ValueError(f"{name} image_key 不可重複")
        frame_sets.append(set(keys))
    if not all(keys == frame_sets[0] for keys in frame_sets[1:]):
        raise ValueError("manifest、segmentation_qc 與 basic image_key identity 不一致")
    manifest_by_key = manifest.set_index(manifest["image_key"].astype(str))
    images_by_key = images.set_index(images["image_key"].astype(str))
    ordered_keys = list(images_by_key.index)
    for column in _IMAGE_METADATA:
        try:
            pd.testing.assert_series_equal(
                images_by_key.loc[ordered_keys, column].reset_index(drop=True),
                manifest_by_key.loc[ordered_keys, column].reset_index(drop=True),
                check_names=False,
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"manifest 與 image-level basic metadata {column} 不一致"
            ) from error
    qc_by_key = segmentation_qc.set_index(
        segmentation_qc["image_key"].astype(str)
    )
    if not qc_by_key.loc[ordered_keys, "group_id"].astype(str).eq(
        images_by_key.loc[ordered_keys, "group_id"].astype(str)
    ).all():
        raise ValueError("segmentation_qc 與 image-level basic metadata group_id 不一致")
    roster = cells.loc[:, ["image_key", "cell_label", "nucleus_label"]]
    if roster.duplicated().any():
        raise ValueError("cell roster identity 不可重複")
    cell_counts = cells.groupby(cells["image_key"].astype(str), sort=False).size()
    if set(cell_counts.index) != frame_sets[0] or (cell_counts < 3).any():
        raise ValueError("每個 basic image_key 必須至少有 3 cells")
    raw_counts = pd.to_numeric(
        images_by_key.loc[cell_counts.index, "cell_count"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(raw_counts).all() or not np.equal(
        raw_counts, np.floor(raw_counts)
    ).all():
        raise ValueError("image-level basic cell_count 必須是有限整數")
    if not np.array_equal(raw_counts.astype(int), cell_counts.to_numpy()):
        raise ValueError("image-level basic cell_count 與 cell roster 不一致")
    for name, values in (("cell IDO_score", cells["IDO_score"]), ("image IDO_score", images["IDO_score"])):
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} 必須全部有限")


def _validate_basic_medians(cells: pd.DataFrame, images: pd.DataFrame) -> None:
    """以 cell table 重算 image-level basic median 與 target。"""
    grouped = cells.groupby(cells["image_key"].astype(str), sort=False)
    by_key = images.set_index(images["image_key"].astype(str))
    for image_key, rows in grouped:
        image_row = by_key.loc[image_key]
        target_median = float(pd.to_numeric(rows["IDO_score"], errors="raise").median())
        if not np.isclose(
            target_median,
            float(image_row["IDO_score"]),
            rtol=0,
            atol=1e-12,
            equal_nan=False,
        ):
            raise ValueError(f"image_key={image_key} 的 IDO_score median 與 image target 不一致")
        for cell_column, image_column in zip(PRIMARY_CELL_FEATURES, PRIMARY_FOV_FEATURES, strict=True):
            actual = float(pd.to_numeric(rows[cell_column], errors="raise").median())
            expected = float(image_row[image_column])
            if not np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=False):
                raise ValueError(f"image_key={image_key} 的 basic median {image_column} 不一致")


def _validate_metadata(
    round1: Mapping[str, Any],
    metadata: Any,
    hashes: Mapping[str, str],
    manifest: pd.DataFrame,
) -> None:
    """驗證 run metadata 與 artifacts/config pins 的內部 identity。"""
    if not isinstance(metadata, Mapping):
        raise ValueError("run_metadata 必須是 JSON object")
    expected = round1.get("expected")
    if not isinstance(expected, Mapping):
        raise ValueError("round1.expected 必須是 mapping")
    counts = metadata.get("input_counts")
    if not isinstance(counts, Mapping) or int(counts.get("analyzed_images", -1)) != len(manifest):
        raise ValueError("run_metadata analyzed_images 與 manifest 不一致")
    if "analyzed_images" in expected and int(expected["analyzed_images"]) != len(manifest):
        raise ValueError("round1.expected analyzed_images 與 manifest 不一致")
    if _metadata_seed(metadata) != int(expected.get("seed", -1)):
        raise ValueError("run_metadata seed 與 round1.expected 不一致")
    manifest_hash = hashes["data_manifest.csv"]
    if metadata.get("manifest_hash") != manifest_hash or expected.get("manifest_hash") != manifest_hash:
        raise ValueError("run_metadata manifest_hash 與 data_manifest bytes 不一致")
    if metadata.get("config_hash") != expected.get("config_hash"):
        raise ValueError("run_metadata config_hash 與 round1.expected 不一致")


def _select_primary(
    frame: pd.DataFrame,
    *,
    include_dummy: bool,
    table: str,
) -> pd.DataFrame:
    """只保留固定模型的 primary Round 1 rows 並加入穩定 identity。"""
    _require_columns(frame, ("model", "round"), table)
    models = set(_MODEL_CONTRACT) if include_dummy else {"extra_trees", "random_forest"}
    selected = frame[frame["round"].eq("primary_round_1") & frame["model"].isin(models)].copy()
    if not frame.empty:
        required_models = models & set(frame["model"])
        if set(selected["model"]) != required_models:
            raise ValueError(f"{table} 指定模型必須來自 primary_round_1")
    selected["source_round"] = "round1"
    selected["configuration_id"] = selected["model"].map(
        {model: f"{model}__{contract[1]}" for model, contract in _MODEL_CONTRACT.items()}
    )
    return selected.reset_index(drop=True)


def _validate_selected_evidence(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    images: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> None:
    """驗證指定模型角色、fold/OOF identity 與 observed target。"""
    expected_models = set(_MODEL_CONTRACT)
    if set(metrics["model"]) != expected_models or set(predictions["model"]) != expected_models:
        raise ValueError("Round 1 必須精確包含 extra_trees、random_forest 與 dummy_median")
    for table_name, frame in (("fold metrics", metrics), ("OOF", predictions)):
        _require_columns(frame, ("validation", "fold", "split_id", "model", "role", "feature_set"), table_name)
        for model, (role, feature_set) in _MODEL_CONTRACT.items():
            rows = frame[frame["model"].eq(model)]
            if not rows["role"].eq(role).all() or not rows["feature_set"].eq(feature_set).all():
                raise ValueError(f"{model} role/feature_set 不符合固定 Round 1 合約")
            expected_ids = rows["validation"].astype(str) + ":" + rows["fold"].astype(str)
            if not expected_ids.eq(rows["split_id"].astype(str)).all():
                raise ValueError(f"{table_name} split_id identity 不一致")
    if metrics.duplicated(["validation", "fold", "model"]).any():
        raise ValueError("fold metrics identity 不可重複")
    _require_columns(predictions, ("image_key", "observed_ido_score"), "OOF")
    if predictions.duplicated(["validation", "fold", "model", "image_key"]).any():
        raise ValueError("OOF identity 不可重複")
    _require_columns(
        split_manifest,
        ("validation", "fold", "image_key", "role"),
        "split manifest",
    )
    test_rows = split_manifest[split_manifest["role"].eq("test")]
    expected_membership = {
        (str(row.validation), str(row.fold), str(row.image_key))
        for row in test_rows.itertuples(index=False)
    }
    for model in _MODEL_CONTRACT:
        model_rows = predictions[predictions["model"].eq(model)]
        actual_membership = {
            (str(row.validation), str(row.fold), str(row.image_key))
            for row in model_rows.itertuples(index=False)
        }
        if actual_membership != expected_membership:
            raise ValueError(
                f"{model} OOF identity 必須精確符合 frozen test membership"
            )
    targets = images.set_index(images["image_key"].astype(str))["IDO_score"]
    expected_targets = predictions["image_key"].astype(str).map(targets)
    observed = pd.to_numeric(predictions["observed_ido_score"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(expected_targets, errors="coerce").to_numpy(float)
    if not np.allclose(observed, expected, rtol=0, atol=1e-12, equal_nan=False):
        raise ValueError("OOF observed IDO_score 與 frozen image target 不一致")


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """取得必要 mapping。"""
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} 必須是 mapping")
    return value


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """拒絕缺少必要欄位的 frame。"""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少欄位：{missing}")


def _sha256_file(path: Path) -> str:
    """串流計算檔案 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_regular_file(path: Path) -> bool:
    """確認 lexical target 本身是 regular file 而非 link/reparse point。"""
    status = os.lstat(path)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISREG(status.st_mode) and not path.is_symlink() and not (
        getattr(status, "st_file_attributes", 0) & reparse
    )


def _contains_reparse_component(root: Path, candidate: Path) -> bool:
    """檢查 root 以下已存在 lexical components 是否含 link/junction。"""
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = root
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for part in relative.parts:
        current = current / part
        try:
            status = os.lstat(current)
        except OSError:
            break
        if current.is_symlink() or (
            getattr(status, "st_file_attributes", 0) & reparse
        ):
            return True
    return False


def _frame_semantic_sha256(frame: pd.DataFrame) -> str:
    """計算解析後 DataFrame 的欄位、dtype、index 與 values digest。"""
    schema = json.dumps(
        {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "index_dtype": str(frame.index.dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    values = pd.util.hash_pandas_object(
        frame,
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256(schema)
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_image_keys_sha256(image_keys: Sequence[str]) -> str:
    """計算 production image-key roster 的 canonical SHA-256。"""
    keys = sorted(str(key) for key in image_keys)
    payload = ("\n".join(keys) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata_semantic_sha256(metadata: Mapping[str, Any]) -> str:
    """計算 run metadata strict canonical JSON 的 SHA-256。"""
    payload = json.dumps(
        dict(metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_metadata_unchanged(metadata: Mapping[str, Any]) -> None:
    """拒絕缺少 snapshot digest 或 loader 後被改寫的 run metadata。"""
    expected = getattr(metadata, "frozen_semantic_sha256", None)
    if not isinstance(expected, str):
        raise ValueError("formal Round 1 metadata semantic digest 不可缺少")
    actual = _metadata_semantic_sha256(metadata)
    if actual != expected:
        raise ValueError(
            "formal Round 1 metadata semantic digest 不符合 loader snapshot"
        )


def _freeze_semantic_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    """將 loader-time canonical semantic digest 保存於各 evidence frame attrs。"""
    for name, frame in frames.items():
        frame.attrs[_SEMANTIC_DIGEST_ATTR] = {
            "name": name,
            "sha256": _frame_semantic_sha256(frame),
        }


def _require_semantic_frames_unchanged(
    frames: Mapping[str, pd.DataFrame],
) -> None:
    """拒絕 loader 回傳後被原地或 copy 後改寫的 evidence frames。"""
    for name, frame in frames.items():
        frozen = frame.attrs.get(_SEMANTIC_DIGEST_ATTR)
        if frozen is None:
            raise ValueError(f"formal Round 1 {name} semantic digest 不可缺少")
        if not isinstance(frozen, Mapping) or frozen.get("name") != name:
            raise ValueError(f"formal Round 1 {name} semantic digest metadata 不合法")
        actual = _frame_semantic_sha256(frame)
        if frozen.get("sha256") != actual:
            raise ValueError(
                f"formal Round 1 {name} semantic digest 不符合 loader snapshot"
            )


def _read_json(path: Path) -> Any:
    """讀取拒絕 NaN 與 Infinity 的 JSON。"""
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"JSON 不允許 constant：{value}")
        ),
    )


def _metadata_seed(metadata: Mapping[str, Any]) -> int:
    """取得並交叉確認 metadata 的 benchmark seed。"""
    seed = int(metadata.get("seed", -1))
    seeds = metadata.get("seeds")
    if isinstance(seeds, Mapping) and int(seeds.get("benchmark", seed)) != seed:
        raise ValueError("run_metadata seed 與 seeds.benchmark 不一致")
    return seed


def _npz_text(cached: Any, name: str) -> str:
    """從 allow_pickle=False 的 NPZ 讀取文字 scalar。"""
    if name not in cached:
        raise ValueError(f"mask cache 缺少 {name}")
    value = np.asarray(cached[name])
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"mask cache {name} 必須是文字 scalar")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _strict_json_object(payload: str, name: str) -> dict[str, Any]:
    """解析 strict JSON object。"""
    value = json.loads(
        payload,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"不允許 JSON constant：{constant}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必須是 JSON object")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    """產生 provenance 使用的 canonical JSON。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _safe_path_component(value: str, name: str) -> None:
    """拒絕可能讓 derived mask path 逃逸 root 的 component。"""
    if not value or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{name} 不是安全 path component")
