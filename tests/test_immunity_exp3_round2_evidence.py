"""測試 Round 2 對 Round 1 凍結證據的唯讀合約。"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from immunity.exp3.benchmark import outer_split_manifest
from immunity.exp3.round2_evidence import (
    Round1Evidence,
    expected_split_ids,
    load_round1_evidence,
    make_smoke_outer_splits,
    require_formal_round1_evidence,
    restore_frozen_outer_splits,
    validate_frozen_masks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASIC_FEATURES = (
    "cell__area",
    "cell__compactness",
    "cell__eccentricity",
    "cell__extent",
    "cell__sphericity",
    "cell__major_axis_length",
    "cell__feret_length",
    "cell__minor_axis_length",
    "cell__feret_width",
    "cell__maximum_radius",
    "cell__mean_radius",
    "cell__median_radius",
    "cell__aspect_ratio",
    "cell__perimeter_area_ratio",
    "cell__perimeter",
    "cell__solidity",
    "nucleus__area",
    "nucleus__compactness",
    "nucleus__eccentricity",
    "nucleus__extent",
    "nucleus__sphericity",
    "nucleus__major_axis_length",
    "nucleus__feret_length",
    "nucleus__minor_axis_length",
    "nucleus__feret_width",
    "nucleus__maximum_radius",
    "nucleus__mean_radius",
    "nucleus__median_radius",
    "nucleus__aspect_ratio",
    "nucleus__perimeter_area_ratio",
    "nucleus__perimeter",
    "nucleus__solidity",
    "nucleus_cytoplasm_area_ratio",
)
FOV_FEATURES = tuple(f"{name}__median" for name in BASIC_FEATURES)
REQUIRED_ARTIFACTS = (
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _roster_sha256(cells: pd.DataFrame) -> str:
    roster = cells.loc[:, ["image_key", "cell_label", "nucleus_label"]].copy()
    roster["image_key"] = roster["image_key"].astype(str)
    roster["cell_label"] = roster["cell_label"].astype(int)
    roster["nucleus_label"] = roster["nucleus_label"].astype(int)
    roster = roster.sort_values(
        ["image_key", "cell_label", "nucleus_label"], kind="stable"
    )
    payload = roster.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _write_round1_fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "exp3"
    (root / "feature_cache" / "masks" / "B1_P1").mkdir(parents=True)
    image_keys = ["B1_P1_C01_F01", "B1_P1_C02_F01"]
    pc_paths = []
    mask_paths = []
    pc_hashes = []
    provenance_hashes = []
    for index, image_key in enumerate(image_keys, start=1):
        pc_path = tmp_path / f"pc-{index}.bin"
        pc_path.write_bytes(f"phase-{index}".encode())
        pc_hash = _sha256(pc_path)
        provenance = {
            "pc_sha256": pc_hash,
            "schema_version": 1,
            "segmentation_config": {"min_cells_per_image": 3},
        }
        provenance_json = _canonical_json(provenance)
        provenance_hash = hashlib.sha256(provenance_json.encode()).hexdigest()
        mask_path = root / "feature_cache" / "masks" / "B1_P1" / f"{image_key}.npz"
        labels = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        np.savez(
            mask_path,
            cell_mask=labels,
            nucleus_mask=labels,
            provenance_json=np.asarray(provenance_json),
            provenance_hash=np.asarray(provenance_hash),
        )
        pc_paths.append(str(pc_path.resolve()))
        mask_paths.append(str(mask_path.resolve()))
        pc_hashes.append(pc_hash)
        provenance_hashes.append(provenance_hash)

    metadata = pd.DataFrame(
        {
            "b_id": ["B1", "B1"],
            "passage": [1, 1],
            "condition_index": [1, 2],
            "fov": [1, 1],
            "group_id": ["B1_P1", "B1_P1"],
            "pc_path": pc_paths,
            "ido_path": [str(tmp_path / "ido-1"), str(tmp_path / "ido-2")],
            "condition": ["IFN0_TNF0", "IFN25_TNF0"],
            "ifn_dose": [0.0, 25.0],
            "tnf_dose": [0.0, 0.0],
            "image_key": image_keys,
        }
    )
    metadata.to_csv(root / "data_manifest.csv", index=False)
    pd.DataFrame(
        {
            "b_id": ["B1", "B1"],
            "passage": [1, 1],
            "condition_index": [1, 2],
            "fov": [1, 1],
            "status": ["paired", "paired"],
            "pc_count": [1, 1],
            "ido_count": [1, 1],
            "pc_path": pc_paths,
            "ido_path": [str(tmp_path / "ido-1"), str(tmp_path / "ido-2")],
            "detail": ["", ""],
        }
    ).to_csv(root / "pairing_qc.csv", index=False)
    pd.DataFrame(
        {
            "image_key": image_keys,
            "group_id": ["B1_P1", "B1_P1"],
            "mask_path": mask_paths,
            "cache_status": ["reused", "reused"],
            "cache_reason": ["provenance_match", "provenance_match"],
            "pc_sha256": pc_hashes,
            "cache_provenance_hash": provenance_hashes,
            "status": ["passed", "passed"],
        }
    ).to_csv(root / "segmentation_qc.csv", index=False)

    cell_rows = []
    image_rows = []
    for image_index, image_key in enumerate(image_keys):
        target = float(image_index + 1)
        for label in (1, 2, 3):
            row = {
                "image_key": image_key,
                "cell_label": label,
                "nucleus_label": label,
                "IDO_score": target,
            }
            row.update(
                {feature: float(image_index * 100 + label + offset) for offset, feature in enumerate(BASIC_FEATURES)}
            )
            cell_rows.append(row)
        image_row = {
            "image_key": image_key,
            "b_id": "B1",
            "passage": 1,
            "group_id": "B1_P1",
            "condition_index": image_index + 1,
            "condition": ["IFN0_TNF0", "IFN25_TNF0"][image_index],
            "ifn_dose": [0.0, 25.0][image_index],
            "tnf_dose": 0.0,
            "fov": 1,
            "cell_count": 3,
            "IDO_score": target,
        }
        image_row.update(
            {f"{feature}__median": float(image_index * 100 + 2 + offset) for offset, feature in enumerate(BASIC_FEATURES)}
        )
        image_rows.append(image_row)
    cells = pd.DataFrame(cell_rows)
    images = pd.DataFrame(image_rows)
    cells.to_csv(root / "feature_cache" / "cell_level_basic.csv", index=False)
    images.to_csv(root / "feature_cache" / "image_level_basic.csv", index=False)

    split_rows = []
    split_specs = [("synthetic", "first", image_keys[0]), ("synthetic", "second", image_keys[1])]
    for validation, fold, held_out in split_specs:
        for row in image_rows:
            split_rows.append(
                {
                    "validation": validation,
                    "fold": fold,
                    "image_key": row["image_key"],
                    "role": "test" if row["image_key"] == held_out else "train",
                    **{key: row[key] for key in ("b_id", "passage", "group_id", "condition_index", "condition", "ifn_dose", "tnf_dose", "fov")},
                }
            )
    pd.DataFrame(split_rows).to_csv(root / "outer_splits.csv", index=False)

    metric_rows = []
    prediction_rows = []
    hyperparameter_rows = []
    importance_rows = []
    model_contract = {
        "extra_trees": ("candidate", "basic_median"),
        "random_forest": ("candidate", "basic_median"),
        "dummy_median": ("diagnostic", "none"),
    }
    for validation, fold, held_out in split_specs:
        target = float(image_keys.index(held_out) + 1)
        split_id = f"{validation}:{fold}"
        for model, (role, feature_set) in model_contract.items():
            metric_rows.append(
                {"validation": validation, "fold": fold, "split_id": split_id, "model": model, "role": role, "feature_set": feature_set, "n_train": 1, "n_test": 1, "mae": 0.1, "rmse": 0.1, "r2": 0.0, "spearman": np.nan, "observed_sd": 0.0, "prediction_sd": 0.0, "status": "ok", "round": "primary_round_1"}
            )
            prediction_rows.append(
                {"validation": validation, "fold": fold, "split_id": split_id, "model": model, "role": role, "feature_set": feature_set, "image_key": held_out, "b_id": "B1", "passage": 1, "group_id": "B1_P1", "condition_index": image_keys.index(held_out) + 1, "condition": ["IFN0_TNF0", "IFN25_TNF0"][image_keys.index(held_out)], "observed_ido_score": target, "predicted_ido_score": target, "round": "primary_round_1"}
            )
            hyperparameter_rows.append(
                {"validation": validation, "fold": fold, "split_id": split_id, "model": model, "seed": 7, "best_params_json": "{}", "inner_best_mae": np.nan, "round": "primary_round_1"}
            )
            if model != "dummy_median":
                importance_rows.append(
                    {"validation": validation, "fold": fold, "split_id": split_id, "model": model, "role": role, "feature_set": feature_set, "feature": FOV_FEATURES[0], "importance_type": "outer_test_permutation_diagnostic", "importance": 0.0, "importance_sd": 0.0, "round": "primary_round_1"}
                )
    pd.DataFrame(metric_rows).to_csv(root / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(root / "oof_predictions.csv", index=False)
    pd.DataFrame(hyperparameter_rows).to_csv(root / "hyperparameters.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(root / "feature_importance.csv", index=False)
    pd.DataFrame(columns=["validation", "fold", "split_id", "model", "exception_type", "message", "round"]).to_csv(root / "model_failures.csv", index=False)
    pd.DataFrame({"model": ["extra_trees", "random_forest"], "role": ["candidate", "candidate"]}).to_csv(root / "model_ranking.csv", index=False)
    (root / "feature_sets.json").write_text(
        json.dumps({"basic_median": {"predictor_columns": list(FOV_FEATURES), "predictor_count": 33, "aggregation": "median", "replication_scope": "primary_phase_only"}}),
        encoding="utf-8",
    )
    manifest_hash = _sha256(root / "data_manifest.csv")
    config_hash = "c" * 64
    (root / "run_metadata.json").write_text(
        json.dumps({"input_counts": {"analyzed_images": 2}, "seed": 7, "manifest_hash": manifest_hash, "config_hash": config_hash}),
        encoding="utf-8",
    )
    hashes = {relative: _sha256(root / relative) for relative in REQUIRED_ARTIFACTS}
    return {
        "round1": {
            "dir": str(root),
            "expected": {"analyzed_images": 2, "seed": 7, "manifest_hash": manifest_hash, "config_hash": config_hash},
            "roster_sha256": _roster_sha256(cells),
            "artifact_sha256": hashes,
        }
    }


def _refresh_hash(config: dict[str, Any], relative: str) -> None:
    root = Path(config["round1"]["dir"])
    config["round1"]["artifact_sha256"][relative] = _sha256(root / relative)


def _mutate_fixture_and_refresh_hash(config: dict[str, Any], drift: str) -> None:
    root = Path(config["round1"]["dir"])
    if drift == "image_key":
        relative = "data_manifest.csv"
        frame = pd.read_csv(root / relative)
        frame.loc[0, "image_key"] = "unexpected"
    elif drift == "target":
        relative = "feature_cache/image_level_basic.csv"
        frame = pd.read_csv(root / relative)
        frame.loc[0, "IDO_score"] += 1.0
    elif drift == "basic_value":
        relative = "feature_cache/image_level_basic.csv"
        frame = pd.read_csv(root / relative)
        frame.loc[0, FOV_FEATURES[0]] += 1.0
    else:
        relative = "feature_cache/cell_level_basic.csv"
        frame = pd.read_csv(root / relative)
        frame.loc[0, "cell_label"] = 99
    frame.to_csv(root / relative, index=False)
    _refresh_hash(config, relative)
    if relative == "data_manifest.csv":
        config["round1"]["expected"]["manifest_hash"] = _sha256(root / relative)


def test_round1_evidence_loads_pinned_artifacts_and_semantic_identity(tmp_path: Path) -> None:
    evidence = load_round1_evidence(_write_round1_fixture(tmp_path))

    assert evidence.basic_images["image_key"].is_unique
    assert len(evidence.basic_cells) == 6
    assert set(evidence.selected_metrics["model"]) == {"extra_trees", "random_forest", "dummy_median"}
    assert set(evidence.selected_metrics["source_round"]) == {"round1"}
    assert set(evidence.selected_metrics["configuration_id"]) == {
        "extra_trees__basic_median",
        "random_forest__basic_median",
        "dummy_median__none",
    }


def test_round1_evidence_rejects_bytes_drift_before_semantic_use(tmp_path: Path) -> None:
    config = _write_round1_fixture(tmp_path)
    target = Path(config["round1"]["dir"]) / "outer_splits.csv"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="outer_splits.csv.*SHA-256"):
        load_round1_evidence(config)


def test_round1_evidence_rejects_missing_required_regular_file(tmp_path: Path) -> None:
    config = _write_round1_fixture(tmp_path)
    target = Path(config["round1"]["dir"]) / "pairing_qc.csv"
    target.unlink()

    with pytest.raises(ValueError, match="pairing_qc.csv.*regular file"):
        load_round1_evidence(config)


def test_round1_evidence_rejects_lexical_artifact_symlink(tmp_path: Path) -> None:
    config = _write_round1_fixture(tmp_path)
    root = Path(config["round1"]["dir"])
    target = root / "pairing_qc.csv"
    backing = root / "pairing_qc.backing.csv"
    target.rename(backing)
    try:
        target.symlink_to(backing)
    except OSError as error:
        pytest.skip(f"此平台無法建立 symlink：{error}")

    with pytest.raises(ValueError, match="pairing_qc.csv.*regular file"):
        load_round1_evidence(config)


@pytest.mark.parametrize("drift", ["image_key", "target", "basic_value", "roster"])
def test_round1_evidence_rejects_semantic_drift_after_hash_refresh(tmp_path: Path, drift: str) -> None:
    config = _write_round1_fixture(tmp_path)
    _mutate_fixture_and_refresh_hash(config, drift)

    with pytest.raises(ValueError, match="image_key|IDO_score|basic|roster"):
        load_round1_evidence(config)


@pytest.mark.parametrize(
    ("column", "value"),
    [("role", "diagnostic"), ("feature_set", "none"), ("round", "exploratory_round_2")],
)
def test_round1_evidence_requires_exact_et_rf_dummy_roles_feature_sets_and_primary_round(
    tmp_path: Path, column: str, value: str
) -> None:
    config = _write_round1_fixture(tmp_path)
    root = Path(config["round1"]["dir"])
    metrics = pd.read_csv(root / "fold_metrics.csv")
    metrics.loc[metrics["model"].eq("extra_trees"), column] = value
    metrics.to_csv(root / "fold_metrics.csv", index=False)
    _refresh_hash(config, "fold_metrics.csv")

    with pytest.raises(ValueError, match="extra_trees|primary_round_1|role|feature_set"):
        load_round1_evidence(config)


@pytest.mark.parametrize("drift", ["metadata", "cell_count", "fractional_count"])
def test_round1_evidence_rejects_cross_table_image_identity_drift(
    tmp_path: Path, drift: str
) -> None:
    config = _write_round1_fixture(tmp_path)
    root = Path(config["round1"]["dir"])
    images = pd.read_csv(root / "feature_cache/image_level_basic.csv")
    if drift == "metadata":
        images.loc[0, "condition"] = "changed"
    elif drift == "cell_count":
        images.loc[0, "cell_count"] = 4
    else:
        images["cell_count"] = images["cell_count"].astype(float)
        images.loc[0, "cell_count"] = 3.5
    images.to_csv(root / "feature_cache/image_level_basic.csv", index=False)
    _refresh_hash(config, "feature_cache/image_level_basic.csv")

    with pytest.raises(ValueError, match="metadata|cell_count"):
        load_round1_evidence(config)


def test_round1_evidence_rejects_oof_image_outside_frozen_test_membership(
    tmp_path: Path,
) -> None:
    config = _write_round1_fixture(tmp_path)
    root = Path(config["round1"]["dir"])
    predictions = pd.read_csv(root / "oof_predictions.csv")
    row = predictions.index[
        predictions["model"].eq("extra_trees")
        & predictions["fold"].eq("first")
    ][0]
    predictions.loc[row, "image_key"] = "B1_P1_C02_F01"
    predictions.loc[row, "observed_ido_score"] = 2.0
    predictions.to_csv(root / "oof_predictions.csv", index=False)
    _refresh_hash(config, "oof_predictions.csv")

    with pytest.raises(ValueError, match="OOF.*test membership"):
        load_round1_evidence(config)


def test_frozen_mask_validation_accepts_exact_pc_and_embedded_provenance(tmp_path: Path) -> None:
    evidence = load_round1_evidence(_write_round1_fixture(tmp_path))

    qc = validate_frozen_masks(evidence)

    assert qc.columns.tolist() == [
        "image_key",
        "pc_path",
        "mask_path",
        "expected_pc_sha256",
        "actual_pc_sha256",
        "expected_provenance_hash",
        "actual_provenance_hash",
        "cell_mask_sha256",
        "nucleus_mask_sha256",
        "status",
        "reason",
    ]
    assert qc["status"].tolist() == ["passed", "passed"]
    assert qc["reason"].tolist() == ["verified", "verified"]
    assert qc["actual_pc_sha256"].tolist() == qc["expected_pc_sha256"].tolist()
    assert qc["actual_provenance_hash"].tolist() == qc["expected_provenance_hash"].tolist()


@pytest.mark.parametrize("drift", ["missing_pc", "changed_pc", "missing_mask", "corrupt_mask", "provenance"])
def test_frozen_mask_validation_fails_closed_for_missing_changed_or_corrupt_cache(tmp_path: Path, drift: str) -> None:
    evidence = load_round1_evidence(_write_round1_fixture(tmp_path))
    pc_path = Path(evidence.manifest.loc[0, "pc_path"])
    mask_path = evidence.root / "feature_cache" / "masks" / "B1_P1" / "B1_P1_C01_F01.npz"
    if drift == "missing_pc":
        pc_path.unlink()
    elif drift == "changed_pc":
        pc_path.write_bytes(b"changed")
    elif drift == "missing_mask":
        mask_path.unlink()
    elif drift == "corrupt_mask":
        mask_path.write_bytes(b"not-an-npz")
    else:
        with np.load(mask_path, allow_pickle=False) as cached:
            arrays = {name: cached[name] for name in cached.files}
        arrays["provenance_hash"] = np.asarray("0" * 64)
        np.savez(mask_path, **arrays)

    qc = validate_frozen_masks(evidence, ["B1_P1_C01_F01"])

    assert qc.loc[0, "status"] == "failed"
    assert "B1_P1_C01_F01" in qc.loc[0, "reason"]
    assert str(pc_path.resolve()) in qc.loc[0, "reason"]
    assert str(mask_path.resolve()) in qc.loc[0, "reason"]


def test_frozen_mask_failure_does_not_change_cache_bytes_or_create_segmenter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = load_round1_evidence(_write_round1_fixture(tmp_path))
    mask_path = evidence.root / "feature_cache" / "masks" / "B1_P1" / "B1_P1_C01_F01.npz"
    before = mask_path.read_bytes()
    Path(evidence.manifest.loc[0, "pc_path"]).write_bytes(b"changed")
    calls = 0

    def forbidden_segmenter(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("mask 驗證不得建立 segmenter")

    monkeypatch.setattr("immunity.exp3.phase_features.PhaseSegmenter", forbidden_segmenter)

    qc = validate_frozen_masks(evidence, ["B1_P1_C01_F01"])

    assert qc.loc[0, "status"] == "failed"
    assert mask_path.read_bytes() == before
    assert calls == 0


def test_frozen_mask_qc_binds_dtype_shape_and_contiguous_array_bytes(
    tmp_path: Path,
) -> None:
    evidence = load_round1_evidence(_write_round1_fixture(tmp_path))
    image_key = "B1_P1_C01_F01"

    original = validate_frozen_masks(evidence, [image_key])

    assert original.loc[0, "cell_mask_sha256"] == (
        "5c561f876f432cee885d05c80d723da8e7d104a6ae33631428b1b4a5cd20bf6c"
    )
    assert original.loc[0, "nucleus_mask_sha256"] == (
        "5c561f876f432cee885d05c80d723da8e7d104a6ae33631428b1b4a5cd20bf6c"
    )
    mask_path = Path(original.loc[0, "mask_path"])
    with np.load(mask_path, allow_pickle=False) as cached:
        arrays = {name: np.asarray(cached[name]) for name in cached.files}
    changed = arrays["cell_mask"].copy()
    changed[0, 0] = 9
    arrays["cell_mask"] = changed
    np.savez(mask_path, **arrays)

    fresh = validate_frozen_masks(evidence, [image_key])

    assert fresh.loc[0, "status"] == "passed"
    assert fresh.loc[0, "cell_mask_sha256"] == (
        "6b85a4b5b20f14c66d3a8d1c062d97435145a22b8781493e261e3564e0e38e65"
    )
    assert fresh.loc[0, "nucleus_mask_sha256"] == original.loc[
        0, "nucleus_mask_sha256"
    ]


@pytest.fixture(scope="module")
def formal_evidence() -> Round1Evidence:
    config = yaml.safe_load((PROJECT_ROOT / "immunity/configs/exp3_round2_paper93.yaml").read_text(encoding="utf-8"))
    return load_round1_evidence(config)


def test_formal_round1_gate_accepts_only_693_23976_exact_hashes_roster_and_23_folds(formal_evidence: Round1Evidence) -> None:
    require_formal_round1_evidence(formal_evidence)


@pytest.mark.parametrize("field", ["config_hash", "input_counts"])
def test_formal_round1_gate_rejects_metadata_mutation_after_load(
    formal_evidence: Round1Evidence,
    field: str,
) -> None:
    canonical = json.dumps(
        dict(formal_evidence.metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == (
        "b10871a7f6b86dff8a86ebbe0cd604a4d599b84aa955646adc7db2cea46fd4a4"
    )
    metadata = copy.deepcopy(formal_evidence.metadata)
    if field == "config_hash":
        metadata["config_hash"] = "0" * 64
    else:
        metadata["input_counts"]["analyzed_images"] = 692
    changed = replace(formal_evidence, metadata=metadata)

    with pytest.raises(ValueError, match="metadata.*semantic digest"):
        require_formal_round1_evidence(changed)


@pytest.mark.parametrize(
    "drift",
    [
        "count",
        "roster",
        "hash",
        "fold",
        "oof",
        "metric_role",
        "split_membership",
        "basic_feature",
        "hyperparameter",
        "attrs_missing",
    ],
)
def test_formal_round1_gate_rejects_count_roster_hash_fold_or_oof_drift(formal_evidence: Round1Evidence, drift: str) -> None:
    evidence = formal_evidence
    if drift == "count":
        evidence = replace(evidence, basic_images=evidence.basic_images.iloc[:-1].copy())
    elif drift == "roster":
        cells = evidence.basic_cells.copy()
        cells.loc[cells.index[0], "cell_label"] = 999999
        evidence = replace(evidence, basic_cells=cells)
    elif drift == "hash":
        hashes = dict(evidence.artifact_hashes)
        hashes["outer_splits.csv"] = "0" * 64
        evidence = replace(evidence, artifact_hashes=hashes)
    elif drift == "fold":
        first = evidence.split_manifest[["validation", "fold"]].drop_duplicates().iloc[0]
        keep = ~(
            evidence.split_manifest["validation"].eq(first["validation"])
            & evidence.split_manifest["fold"].astype(str).eq(str(first["fold"]))
        )
        evidence = replace(evidence, split_manifest=evidence.split_manifest.loc[keep].copy())
    elif drift == "metric_role":
        metrics = evidence.selected_metrics.copy()
        metrics.loc[metrics.index[0], "role"] = "diagnostic"
        evidence = replace(evidence, selected_metrics=metrics)
    elif drift == "split_membership":
        manifest = evidence.split_manifest.copy()
        family = manifest[manifest["validation"].eq("leave_one_b_out")]
        first_test = family[family["fold"].astype(str).eq("B4") & family["role"].eq("test")].iloc[0]
        second_test = family[family["fold"].astype(str).eq("B7") & family["role"].eq("test")].iloc[0]
        for fold, key, role in (
            ("B4", first_test["image_key"], "train"),
            ("B4", second_test["image_key"], "test"),
            ("B7", second_test["image_key"], "train"),
            ("B7", first_test["image_key"], "test"),
        ):
            mask = manifest["validation"].eq("leave_one_b_out") & manifest["fold"].astype(str).eq(fold) & manifest["image_key"].eq(key)
            manifest.loc[mask, "role"] = role
        evidence = replace(evidence, split_manifest=manifest)
    elif drift == "basic_feature":
        cells = evidence.basic_cells.copy()
        cells.loc[cells.index[0], BASIC_FEATURES[0]] += 1.0
        evidence = replace(evidence, basic_cells=cells)
    elif drift == "hyperparameter":
        hyperparameters = evidence.selected_hyperparameters.copy()
        hyperparameters.loc[hyperparameters.index[0], "best_params_json"] = '{"changed":true}'
        evidence = replace(evidence, selected_hyperparameters=hyperparameters)
    elif drift == "attrs_missing":
        hyperparameters = evidence.selected_hyperparameters.copy()
        hyperparameters.attrs.clear()
        evidence = replace(evidence, selected_hyperparameters=hyperparameters)
    else:
        evidence = replace(evidence, selected_predictions=evidence.selected_predictions.iloc[:-1].copy())

    with pytest.raises(ValueError, match="693|23,976|roster|SHA-256|fold|OOF|role|membership|semantic"):
        require_formal_round1_evidence(evidence)


def test_restore_frozen_splits_round_trips_outer_split_manifest_exactly(formal_evidence: Round1Evidence) -> None:
    splits = restore_frozen_outer_splits(formal_evidence.basic_images, formal_evidence.split_manifest)

    restored = outer_split_manifest(formal_evidence.basic_images, splits)
    expected = formal_evidence.split_manifest.loc[:, restored.columns]
    pd.testing.assert_frame_equal(restored.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    assert {family: len(ids) for family, ids in expected_split_ids(splits).items()} == {
        "leave_one_b_out": 3,
        "leave_one_passage_out": 3,
        "leave_one_group_out": 9,
        "leave_one_condition_out": 8,
    }


def test_restore_frozen_splits_canonicalizes_rows_within_each_split(
    formal_evidence: Round1Evidence,
) -> None:
    reordered_groups = []
    for _, rows in formal_evidence.split_manifest.groupby(
        ["validation", "fold"], sort=False
    ):
        reordered_groups.append(rows.iloc[::-1])
    reordered = pd.concat(reordered_groups, ignore_index=True)

    splits = restore_frozen_outer_splits(formal_evidence.basic_images, reordered)

    restored = outer_split_manifest(formal_evidence.basic_images, splits)
    expected = formal_evidence.split_manifest.loc[:, restored.columns]
    pd.testing.assert_frame_equal(
        restored.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_restore_frozen_splits_rejects_coordinated_cross_fold_membership_swap(
    formal_evidence: Round1Evidence,
) -> None:
    manifest = formal_evidence.split_manifest.copy()
    family = manifest[manifest["validation"].eq("leave_one_b_out")]
    first_test = family[
        family["fold"].astype(str).eq("B4") & family["role"].eq("test")
    ].iloc[0]
    second_test = family[
        family["fold"].astype(str).eq("B7") & family["role"].eq("test")
    ].iloc[0]
    for fold, key, role in (
        ("B4", first_test["image_key"], "train"),
        ("B4", second_test["image_key"], "test"),
        ("B7", second_test["image_key"], "train"),
        ("B7", first_test["image_key"], "test"),
    ):
        mask = (
            manifest["validation"].eq("leave_one_b_out")
            & manifest["fold"].astype(str).eq(fold)
            & manifest["image_key"].eq(key)
        )
        manifest.loc[mask, "role"] = role

    with pytest.raises(ValueError, match="canonical.*membership"):
        restore_frozen_outer_splits(formal_evidence.basic_images, manifest)


def test_restore_frozen_splits_rejects_synchronized_nonproduction_image_keys(
    formal_evidence: Round1Evidence,
) -> None:
    image_keys = sorted(formal_evidence.basic_images["image_key"].astype(str))
    payload = ("\n".join(image_keys) + "\n").encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == (
        "efc55b45033a3923f1467dce86802f6514a0e9c182de77e94a2c155d47503232"
    )
    renamed = {key: f"RENAMED_{key}" for key in image_keys}
    images = formal_evidence.basic_images.copy()
    images["image_key"] = images["image_key"].map(renamed)
    manifest = formal_evidence.split_manifest.copy()
    manifest["image_key"] = manifest["image_key"].map(renamed)

    with pytest.raises(ValueError, match="image-key roster SHA-256"):
        restore_frozen_outer_splits(images, manifest)


@pytest.mark.parametrize("drift", ["missing", "duplicate", "unknown", "membership", "metadata"])
def test_restore_frozen_splits_rejects_missing_duplicate_unknown_or_changed_membership(formal_evidence: Round1Evidence, drift: str) -> None:
    manifest = formal_evidence.split_manifest.copy()
    if drift == "missing":
        manifest = manifest.iloc[1:].copy()
    elif drift == "duplicate":
        manifest = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    elif drift == "unknown":
        manifest.loc[0, "image_key"] = "UNKNOWN"
    elif drift == "membership":
        manifest.loc[0, "role"] = "test" if manifest.loc[0, "role"] == "train" else "train"
    else:
        manifest.loc[0, "group_id"] = "changed"

    with pytest.raises(ValueError, match="完整|重複|unknown|membership|metadata|test|image_key"):
        restore_frozen_outer_splits(formal_evidence.basic_images, manifest)


def test_smoke_splits_have_four_validation_families_without_frozen_membership_gate(formal_evidence: Round1Evidence) -> None:
    smoke_images = (
        formal_evidence.basic_images.sort_values(["group_id", "condition_index", "fov"], kind="stable")
        .groupby(["group_id", "condition_index"], sort=False)
        .head(1)
        .reset_index(drop=True)
    )

    splits = make_smoke_outer_splits(smoke_images, {"smoke": {"fovs_per_condition": 1}})

    assert set(expected_split_ids(splits)) == {
        "leave_one_b_out",
        "leave_one_passage_out",
        "leave_one_group_out",
        "leave_one_condition_out",
    }
    assert len({f"{split.validation}:{split.fold}" for split in splits}) == len(splits)
    assert all(len(split.train_index) and len(split.test_index) for split in splits)
    assert all(not np.intersect1d(split.train_index, split.test_index).size for split in splits)
