from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import immunity.exp4.cv as cv_module
import immunity.exp4.phase2 as phase2_module
from immunity.exp4.cell_dedup import NUCLEUS_FEATURE_COLUMNS
from immunity.exp4.cv import (
    CvPopulationContract,
    FeatureArmAssembly,
    LeakageContext,
    assemble_feature_arms,
)
from immunity.exp4.phase2 import (
    METRICS_WITHIN_CONDITION_COLUMNS,
    PHASE1_VS_PHASE2_COMPARISON_COLUMNS,
    PHASE1_RETAINED_SEQUENCE_SHA256,
    FrozenPhase1Evidence,
    PHASE2_SHRINKAGE_COLUMNS,
    PHASE2_TARGET_COLUMNS,
    Phase2ContractError,
    Phase2LeakageContext,
    Phase2LeakageError,
    Phase2PopulationContract,
    analyze_phase2_shrinkage,
    assemble_phase2_cv,
    build_metrics_within_condition,
    compare_phase1_phase2,
    build_phase2_targets,
    reconstruct_phase2_population,
    run_phase2_cv,
    run_phase2_leakage_preflight,
    phase2_cv_source_components,
)
from immunity.exp4.phase2_diagnostics import (
    CONDITION_MAPPING_REVERSED_HYPOTHESIS,
    DOSE_MAPPING_NOTE,
    DOSE_RESPONSE_COLUMNS,
    PHASE2_NUMERIC_ARTIFACT_NAMES,
    RESIDUAL_VS_DOSE_COLUMNS,
    Phase2ArtifactMutationError,
    analyze_residual_vs_dose,
    build_dose_response_check,
    fingerprint_phase2_numeric_artifacts,
    verify_phase2_numeric_artifact_immutability,
)
from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS


DONOR_PASSAGES = tuple(
    (b_id, passage)
    for b_id in ("B4", "B7", "B8")
    for passage in (5, 6, 7)
)
LOW_GROUP_COUNTS = {
    "B8_P7_C01": 29,
    "B8_P7_C07": 27,
    "B8_P7_C08": 22,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formal_reconstruction() -> Iterator[dict[str, object]]:
    """唯讀載入 canonical Phase 1 artifacts，供 formal seam integration 共用。"""
    paths = _formal_phase1_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        pytest.skip(f"formal Phase 1 artifacts unavailable: {missing}")
    metadata = json.loads(paths["run_metadata.json"].read_text(encoding="utf-8"))
    evidence = FrozenPhase1Evidence.from_observed(
        artifact_sha256={
            name: _sha256_file(path)
            for name, path in paths.items()
            if name != "run_metadata.json"
        },
        pre_border_consistency=metadata["full_rui49"]["pre_border_consistency"],
        retained_consistency=metadata["full_rui49"]["retained_consistency"],
    )
    frames = {
        "oof": pd.read_csv(paths["oof_predictions.csv"]),
        "basic": pd.read_csv(paths["cell_level_basic.csv"]),
        "dedup": pd.read_csv(
            paths["cell_dedup_report.csv"], float_precision="round_trip"
        ),
        "rui49": pd.read_csv(paths["cell_level_rui49.csv"]),
        "health": pd.read_csv(paths["feature_health_report.csv"]),
        "redundancy": pd.read_csv(paths["feature_redundancy_report.csv"]),
        "manifest": pd.read_csv(paths["data_manifest.csv"]).loc[
            :, ["image_key", "b_id", "passage", "condition_index"]
        ],
    }
    population = reconstruct_phase2_population(
        phase1_oof=frames["oof"],
        cell_level_basic=frames["basic"],
        cell_dedup_report=frames["dedup"],
        cell_level_rui49=frames["rui49"],
        feature_health_report=frames["health"],
        feature_redundancy_report=frames["redundancy"],
        manifest=frames["manifest"],
        evidence=evidence,
    )
    yield {
        "paths": paths,
        "metadata": metadata,
        "evidence": evidence,
        "frames": frames,
        "population": population,
    }


@pytest.fixture(scope="module")
def formal_posthoc_inputs(
    formal_reconstruction: dict[str, object],
) -> Iterator[dict[str, pd.DataFrame]]:
    """以 frozen Phase 1 OOF predictions 組成不重訓的 Phase 2 post-hoc fixture。"""
    population = formal_reconstruction["population"]
    phase1_oof = formal_reconstruction["frames"]["oof"]
    targets = build_phase2_targets(population.cells, population.manifest)
    cell_groups = population.cells.loc[
        :, ["image_key", "cell_label", "b_id", "passage", "condition_index"]
    ].copy()
    cell_groups["group_id"] = (
        cell_groups["b_id"]
        + "_P"
        + cell_groups["passage"].astype(str)
        + "_C"
        + cell_groups["condition_index"].astype(str).str.zfill(2)
    )
    cell_groups = cell_groups.merge(
        targets.loc[:, ["group_id", "group_IDO_score", "confidence_flag"]],
        on="group_id",
        how="left",
        validate="many_to_one",
    )
    oof = phase1_oof.drop(
        columns=[
            "group_id",
            "confidence_flag",
            "observed_ido_score",
            "residual",
        ]
    ).merge(
        cell_groups.loc[
            :,
            [
                "image_key",
                "cell_label",
                "group_id",
                "group_IDO_score",
                "confidence_flag",
            ],
        ],
        on=["image_key", "cell_label"],
        how="left",
        validate="many_to_one",
    )
    oof = oof.rename(columns={"group_IDO_score": "observed_ido_score"})
    oof["residual"] = oof["predicted_ido_score"] - oof["observed_ido_score"]
    oof = oof.loc[
        :,
        [
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
        ],
    ]
    output_root = PROJECT_ROOT / "immunity/outputs/exp4"
    phase1_metrics = pd.read_csv(output_root / "cv_metrics.csv")
    phase1_shrinkage = pd.read_csv(output_root / "shrinkage_analysis.csv")
    phase2_metrics = phase1_metrics.copy()
    phase2_metrics["cell_r2"] = phase2_metrics["cell_r2"] + 0.05
    phase2_metrics["rui_r2"] = phase2_metrics["rui_r2"] + 0.10
    yield {
        "targets": targets,
        "oof": oof,
        "phase1_metrics": phase1_metrics,
        "phase1_shrinkage": phase1_shrinkage,
        "phase2_metrics": phase2_metrics,
    }


@pytest.fixture(scope="module")
def formal_manifest_and_cells() -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """建立符合 Phase 2 正式 count contract 的 deterministic synthetic population。"""
    manifest_rows: list[dict[str, object]] = []
    images_by_group: dict[str, list[str]] = {}
    for b_id, passage in DONOR_PASSAGES:
        fov_serial = 0
        for condition_index in range(1, 9):
            fov_count = 10 if condition_index <= 5 else 9
            group_id = f"{b_id}_P{passage}_C{condition_index:02d}"
            images_by_group[group_id] = []
            for _ in range(fov_count):
                fov_serial += 1
                image_key = f"{group_id}_F{fov_serial:02d}"
                images_by_group[group_id].append(image_key)
                manifest_rows.append(
                    {
                        "image_key": image_key,
                        "b_id": b_id,
                        "passage": passage,
                        "condition_index": condition_index,
                    }
                )

    regular_groups = [
        group_id for group_id in images_by_group if group_id not in LOW_GROUP_COUNTS
    ]
    remaining = 19_648 - sum(LOW_GROUP_COUNTS.values())
    base_count, extra = divmod(remaining, len(regular_groups))
    group_counts = {
        group_id: base_count + int(index < extra)
        for index, group_id in enumerate(regular_groups)
    }
    group_counts.update(LOW_GROUP_COUNTS)

    cell_rows: list[dict[str, object]] = []
    for group_position, (group_id, image_keys) in enumerate(images_by_group.items()):
        count = group_counts[group_id]
        labels_by_image = {image_key: 0 for image_key in image_keys}
        for offset in range(count):
            image_key = image_keys[offset % len(image_keys)]
            labels_by_image[image_key] += 1
            cell_rows.append(
                {
                    "image_key": image_key,
                    "cell_label": labels_by_image[image_key],
                    "IDO_score": float(group_position + 1),
                }
            )

    manifest = pd.DataFrame(
        manifest_rows,
        columns=("image_key", "b_id", "passage", "condition_index"),
    )
    cells = pd.DataFrame(cell_rows)
    assert len(manifest) == 693
    assert len(cells) == 19_648
    yield manifest, cells


def test_build_phase2_targets_uses_only_opaque_condition_keys(
    formal_manifest_and_cells: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, cells = formal_manifest_and_cells

    targets = build_phase2_targets(cells, manifest)

    assert targets.columns.tolist() == list(PHASE2_TARGET_COLUMNS)
    assert len(targets) == 72
    assert not targets.duplicated(["b_id", "passage", "condition_index"]).any()
    assert targets["group_id"].str.fullmatch(r"B[478]_P[567]_C0[1-8]").all()
    assert targets.groupby(["b_id", "passage"])["condition_index"].apply(
        lambda values: tuple(values) == tuple(range(1, 9))
    ).all()
    assert targets["cells_after"].sum() == 19_648
    assert targets["fov_count"].sum() == 693
    assert targets["group_IDO_score"].nunique() == 72
    assert targets.loc[
        targets["confidence_flag"].eq("low"), ["group_id", "cells_after"]
    ].set_index("group_id")["cells_after"].to_dict() == LOW_GROUP_COUNTS
    assert targets.loc[
        targets["confidence_flag"].eq("normal"), "cells_after"
    ].ge(30).all()
    assert not {
        "condition",
        "ifn_dose",
        "tnf_dose",
    }.intersection(targets.columns)


def test_build_phase2_targets_rejects_non_allowlisted_manifest_column(
    formal_manifest_and_cells: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, cells = formal_manifest_and_cells
    leaked = manifest.assign(ifn_dose=100)

    with pytest.raises(Phase2ContractError, match="allowlist"):
        build_phase2_targets(cells, leaked)


def test_build_phase2_targets_fails_closed_below_ten_cells(
    formal_manifest_and_cells: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    manifest, cells = formal_manifest_and_cells
    blocked_group = "B8_P7_C08"
    blocked_keys = set(
        manifest.loc[
            manifest["image_key"].str.startswith(blocked_group), "image_key"
        ]
    )
    blocked = cells.loc[
        ~cells["image_key"].isin(blocked_keys)
        | cells.groupby("image_key").cumcount().lt(1)
    ].copy()
    blocked = blocked.loc[
        ~blocked["image_key"].isin(blocked_keys)
        | blocked.groupby("image_key").cumcount().lt(1)
    ].copy()

    with pytest.raises(Phase2ContractError, match="below 10"):
        build_phase2_targets(blocked, manifest)


def test_reconstruct_phase2_population_reuses_frozen_phase1_artifacts(
    formal_reconstruction: dict[str, object],
) -> None:
    population = formal_reconstruction["population"]
    frames = formal_reconstruction["frames"]
    dedup = frames["dedup"]
    rui49 = frames["rui49"]
    targets = build_phase2_targets(population.cells, population.manifest)

    assert len(population.cells) == 19_648
    assert population.cells["image_key"].nunique() == 693
    assert population.retained_sequence_sha256 == PHASE1_RETAINED_SEQUENCE_SHA256
    assert len(population.healthy_features) == 48
    assert "cell__MinIntensity" not in population.healthy_features
    assert len(population.filtered_features) == 30
    assert set(population.filtered_features).issubset(population.healthy_features)
    assert tuple(population.cells.columns[-49:]) == tuple(RUI49_FEATURE_COLUMNS)

    retained_keys = population.cells.loc[:, ["image_key", "cell_label"]]
    retained_dedup = dedup.merge(
        retained_keys, on=["image_key", "cell_label"], how="inner"
    ).iloc[0]
    reconstructed_dedup = population.cells.loc[
        population.cells["image_key"].eq(retained_dedup["image_key"])
        & population.cells["cell_label"].eq(retained_dedup["cell_label"])
    ].iloc[0]
    assert reconstructed_dedup["IDO_score"] == pytest.approx(
        retained_dedup["IDO_score_after"], abs=0.0
    )
    np.testing.assert_array_equal(
        reconstructed_dedup.loc[list(NUCLEUS_FEATURE_COLUMNS)].to_numpy(float),
        retained_dedup.loc[list(NUCLEUS_FEATURE_COLUMNS)].to_numpy(float),
    )
    first_rui = rui49.loc[
        rui49["image_key"].eq(retained_dedup["image_key"])
        & rui49["cell_label"].eq(retained_dedup["cell_label"])
    ].iloc[0]
    np.testing.assert_array_equal(
        reconstructed_dedup.loc[list(RUI49_FEATURE_COLUMNS)].to_numpy(float),
        first_rui.loc[list(RUI49_FEATURE_COLUMNS)].to_numpy(float),
    )

    assert targets["cells_after"].agg(["min", "median", "max"]).to_dict() == {
        "min": 22.0,
        "median": 226.0,
        "max": 1_159.0,
    }
    assert targets.loc[
        targets["confidence_flag"].eq("low"), ["group_id", "cells_after"]
    ].set_index("group_id")["cells_after"].to_dict() == LOW_GROUP_COUNTS
    assert targets["group_IDO_score"].nunique() == 72
    assert (
        targets["group_IDO_score"].max() - targets["group_IDO_score"].min()
    ) == pytest.approx(17.1620334292, abs=1e-9)


def test_reconstruct_phase2_nucleus_arm_matches_phase1_for_all_retained_cells_exactly(
    formal_reconstruction: dict[str, object],
) -> None:
    """以 Phase 1 dedup 規則逐格重建正式 19,648×17 nucleus evidence。"""
    population = formal_reconstruction["population"]
    frames = formal_reconstruction["frames"]
    raw = frames["basic"]
    dedup = frames["dedup"]
    retained = frames["oof"].loc[
        frames["oof"]["configuration_id"].eq("geometry_24__SVR_L"),
        ["image_key", "cell_label"],
    ].reset_index(drop=True)

    key_columns = ["image_key", "cell_label"]
    first = raw.drop_duplicates(key_columns, keep="first").loc[
        :, [*key_columns, "cell__area", *NUCLEUS_FEATURE_COLUMNS]
    ].copy()
    dedup_index = pd.MultiIndex.from_frame(dedup.loc[:, key_columns])
    first_index = pd.MultiIndex.from_frame(first.loc[:, key_columns])
    singleton_mask = ~first_index.isin(dedup_index)
    singleton = first.loc[singleton_mask].copy()
    singleton["nucleus_cytoplasm_area_ratio"] = (
        singleton["nucleus__area"]
        / (singleton["cell__area"] - singleton["nucleus__area"])
    )
    canonical = pd.concat(
        [
            singleton.loc[:, [*key_columns, *NUCLEUS_FEATURE_COLUMNS]],
            dedup.loc[:, [*key_columns, *NUCLEUS_FEATURE_COLUMNS]],
        ],
        ignore_index=True,
    )
    expected = retained.merge(
        canonical,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    actual = population.cells.loc[:, NUCLEUS_FEATURE_COLUMNS]

    assert expected.loc[:, NUCLEUS_FEATURE_COLUMNS].shape == (19_648, 17)
    assert actual.shape == (19_648, 17)
    np.testing.assert_array_equal(
        actual.to_numpy(dtype=np.float64),
        expected.loc[:, NUCLEUS_FEATURE_COLUMNS].to_numpy(dtype=np.float64),
    )


def test_reconstruct_phase2_rejects_default_parsed_dedup_numeric_values(
    formal_reconstruction: dict[str, object],
) -> None:
    frames = formal_reconstruction["frames"]
    paths = formal_reconstruction["paths"]
    default_parsed = pd.read_csv(paths["cell_dedup_report.csv"])
    numeric_columns = ["IDO_score_after", *NUCLEUS_FEATURE_COLUMNS]
    assert not np.array_equal(
        default_parsed.loc[:, numeric_columns].to_numpy(dtype=np.float64),
        frames["dedup"].loc[:, numeric_columns].to_numpy(dtype=np.float64),
    )

    with pytest.raises(Phase2ContractError, match="round-trip parsed-numeric"):
        reconstruct_phase2_population(
            phase1_oof=frames["oof"],
            cell_level_basic=frames["basic"],
            cell_dedup_report=default_parsed,
            cell_level_rui49=frames["rui49"],
            feature_health_report=frames["health"],
            feature_redundancy_report=frames["redundancy"],
            manifest=frames["manifest"],
            evidence=formal_reconstruction["evidence"],
        )


def test_formal_phase1_data_fingerprint_matches_existing_checkpoint_golden(
    formal_reconstruction: dict[str, object],
) -> None:
    population = formal_reconstruction["population"]
    output_root = PROJECT_ROOT / "immunity/outputs/exp4"
    phase1_oof = formal_reconstruction["frames"]["oof"]
    targets = (
        phase1_oof.loc[
            phase1_oof["configuration_id"].eq("geometry_24__SVR_L"),
            ["group_id", "observed_ido_score"],
        ]
        .drop_duplicates("group_id")
        .set_index("group_id")["observed_ido_score"]
    )
    cells = population.cells.copy()
    cells["group_id"] = (
        cells["b_id"].astype(str) + "_P" + cells["passage"].astype(str)
    )
    cells["group_IDO_score"] = cells["group_id"].map(targets)
    assembly = assemble_feature_arms(
        cells,
        filtered_features=population.filtered_features,
        contract=CvPopulationContract.formal_exp4(),
    )
    checkpoint_manifest = json.loads(
        (output_root / ".cv_checkpoints/manifest.json").read_text(encoding="utf-8")
    )

    assert checkpoint_manifest["data_fingerprint"] == (
        "cfb1d3e0638e7fb9c0f173c4167a64af64ab96ed26909066640e43a51559ee0b"
    )
    assert cv_module._data_fingerprint(assembly) == checkpoint_manifest[
        "data_fingerprint"
    ]
    supplied = FeatureArmAssembly(
        cells=assembly.cells.assign(confidence_flag="low"),
        arm_features=assembly.arm_features,
        contract=assembly.contract,
    )
    assert cv_module._data_fingerprint(supplied) == checkpoint_manifest[
        "data_fingerprint"
    ]


def test_assemble_phase2_cv_has_formal_72_group_contract_and_eight_checks(
    formal_reconstruction: dict[str, object],
) -> None:
    population = formal_reconstruction["population"]
    metadata = formal_reconstruction["metadata"]
    leakage_context = Phase2LeakageContext.formal(
        LeakageContext(
            standardization_scope="pipeline_fit_within_outer_training",
            feature_selection_inputs=("X",),
            cellpose_invoked=False,
            mask_cache_fingerprint_before=metadata["mask_cache_fingerprint_before"],
            mask_cache_fingerprint_after=metadata["mask_cache_fingerprint_after"],
        )
    )

    assembly = assemble_phase2_cv(population, leakage_context)

    assert assembly.contract == Phase2PopulationContract.formal()
    assert assembly.feature_assembly.contract.formal is True
    assert len(assembly.feature_assembly.contract.expected_group_ids) == 72
    assert len(assembly.targets) == 72
    assert assembly.leakage.report["check_id"].tolist() == list(range(1, 9))
    assert assembly.leakage.report["status"].tolist() == ["passed"] * 8
    assert set(assembly.cv_input_fingerprints) == {
        "x_fingerprint",
        "y_fingerprint",
        "group_ids_fingerprint",
        "split_fingerprint",
        "cv_source_fingerprint",
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for value in assembly.cv_input_fingerprints.values()
    )
    assert (
        assembly.cv_input_fingerprints["split_fingerprint"]
        == "997fc8459dedbd49db6730211ca5555af7b7025c2ead6cc911d81463bd1be951"
    )
    predictors = {
        feature
        for features in assembly.feature_assembly.as_mapping().values()
        for feature in features
    }
    assert "group_id" not in predictors
    assert not any(
        feature.lower() in {"condition", "ifn_dose", "tnf_dose"}
        or "dose" in feature.lower()
        for feature in predictors
    )
    low = assembly.feature_assembly.cells.loc[
        assembly.feature_assembly.cells["confidence_flag"].eq("low")
    ]
    assert set(low["group_id"]) == set(LOW_GROUP_COUNTS)

    source_components = phase2_cv_source_components()
    assert set(source_components) == {"phase2.py", "cv.py"}
    assert source_components["phase2.py"] == _sha256_file(
        PROJECT_ROOT / "immunity/exp4/phase2.py"
    )
    assert source_components["cv.py"] == _sha256_file(
        PROJECT_ROOT / "immunity/exp4/cv.py"
    )
    assert "phase2_diagnostics.py" not in source_components


@pytest.mark.parametrize("evidence_kind", ["manifest", "filename", "x_roster"])
def test_phase2_leakage_check_eight_fails_closed_on_actual_evidence(
    formal_reconstruction: dict[str, object],
    evidence_kind: str,
) -> None:
    population = formal_reconstruction["population"]
    metadata = formal_reconstruction["metadata"]
    valid = Phase2LeakageContext.formal(
        LeakageContext(
            standardization_scope="pipeline_fit_within_outer_training",
            feature_selection_inputs=("X",),
            cellpose_invoked=False,
            mask_cache_fingerprint_before=metadata["mask_cache_fingerprint_before"],
            mask_cache_fingerprint_after=metadata["mask_cache_fingerprint_after"],
        )
    )
    assembly = assemble_phase2_cv(population, valid)
    manifest = population.manifest.copy()
    feature_assembly = assembly.feature_assembly
    if evidence_kind == "manifest":
        manifest = manifest.assign(ifn_dose=100)
    elif evidence_kind == "filename":
        manifest.loc[manifest.index[0], "condition_index"] = 8
    else:
        arm_features = list(feature_assembly.arm_features)
        name, features = arm_features[0]
        arm_features[0] = (name, (*features, "condition_index"))
        feature_assembly = FeatureArmAssembly(
            cells=feature_assembly.cells,
            arm_features=tuple(arm_features),
            contract=feature_assembly.contract,
        )

    with pytest.raises(Phase2LeakageError) as captured:
        run_phase2_leakage_preflight(
            feature_assembly,
            valid,
            manifest=manifest,
        )

    report = captured.value.report
    assert len(report) == 8
    assert report.loc[report["check_id"].eq(8), "status"].item() == "failed"


def test_phase2_leakage_check_eight_rejects_non_first_cell_manifest_drift(
    formal_reconstruction: dict[str, object],
) -> None:
    population = formal_reconstruction["population"]
    metadata = formal_reconstruction["metadata"]
    context = Phase2LeakageContext.formal(
        LeakageContext(
            standardization_scope="pipeline_fit_within_outer_training",
            feature_selection_inputs=("X",),
            cellpose_invoked=False,
            mask_cache_fingerprint_before=metadata["mask_cache_fingerprint_before"],
            mask_cache_fingerprint_after=metadata["mask_cache_fingerprint_after"],
        )
    )
    assembly = assemble_phase2_cv(population, context)
    cells = assembly.feature_assembly.cells.copy()
    non_first_index = cells.index[cells["image_key"].duplicated(keep="first")][0]
    assert cells.loc[: non_first_index - 1, "image_key"].eq(
        cells.at[non_first_index, "image_key"]
    ).any()
    cells.at[non_first_index, "condition_index"] = (
        int(cells.at[non_first_index, "condition_index"]) % 8 + 1
    )
    drifted = replace(assembly.feature_assembly, cells=cells)

    with pytest.raises(Phase2LeakageError) as captured:
        run_phase2_leakage_preflight(
            drifted,
            context,
            manifest=population.manifest,
        )

    check_eight = captured.value.report.loc[
        captured.value.report["check_id"].eq(8)
    ].iloc[0]
    assert check_eight["status"] == "failed"
    assert "cell metadata" in check_eight["details"]


def test_phase2_runner_uses_distinct_checkpoint_tree(
    formal_reconstruction: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    population = formal_reconstruction["population"]
    metadata = formal_reconstruction["metadata"]
    context = Phase2LeakageContext.formal(
        LeakageContext(
            standardization_scope="pipeline_fit_within_outer_training",
            feature_selection_inputs=("X",),
            cellpose_invoked=False,
            mask_cache_fingerprint_before=metadata["mask_cache_fingerprint_before"],
            mask_cache_fingerprint_after=metadata["mask_cache_fingerprint_after"],
        )
    )
    assembly = assemble_phase2_cv(population, context)
    captured: dict[str, object] = {}
    frozen_core = object()

    def fake_run_rui_cv(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return frozen_core

    monkeypatch.setattr(phase2_module, "run_rui_cv", fake_run_rui_cv)
    phase1_checkpoints = tmp_path / ".cv_checkpoints"
    phase2_checkpoints = tmp_path / ".phase2_cv_checkpoints"

    result = run_phase2_cv(
        assembly,
        context,
        checkpoint_dir=phase2_checkpoints,
        phase1_checkpoint_dir=phase1_checkpoints,
        n_jobs=1,
        importance_repeats=1,
    )
    assert result.core is frozen_core
    assert captured["args"] == (assembly.feature_assembly, context.phase1)
    assert captured["kwargs"]["checkpoint_dir"] == phase2_checkpoints
    assert captured["kwargs"]["source_fingerprint"] == (
        assembly.cv_input_fingerprints["cv_source_fingerprint"]
    )
    invalid_paths = (
        phase1_checkpoints,
        phase1_checkpoints.parent,
        phase1_checkpoints / "phase2",
    )
    for invalid_path in invalid_paths:
        with pytest.raises(Phase2ContractError, match="disjoint"):
            run_phase2_cv(
                assembly,
                context,
                checkpoint_dir=invalid_path,
                phase1_checkpoint_dir=phase1_checkpoints,
                n_jobs=1,
                importance_repeats=1,
            )


@pytest.mark.parametrize(
    "drift_kind", ["confidence_flag", "group_IDO_score", "group_id"]
)
def test_phase2_runner_rejects_per_cell_target_metadata_drift_before_core_fit(
    formal_reconstruction: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drift_kind: str,
) -> None:
    population = formal_reconstruction["population"]
    metadata = formal_reconstruction["metadata"]
    context = Phase2LeakageContext.formal(
        LeakageContext(
            standardization_scope="pipeline_fit_within_outer_training",
            feature_selection_inputs=("X",),
            cellpose_invoked=False,
            mask_cache_fingerprint_before=metadata["mask_cache_fingerprint_before"],
            mask_cache_fingerprint_after=metadata["mask_cache_fingerprint_after"],
        )
    )
    assembly = assemble_phase2_cv(population, context)
    cells = assembly.feature_assembly.cells.copy()
    group_mask = cells["group_id"].eq("B8_P7_C01")
    assert int(group_mask.sum()) == 29
    assert cells.loc[group_mask, "confidence_flag"].eq("low").all()
    if drift_kind == "confidence_flag":
        cells.loc[group_mask, "confidence_flag"] = "normal"
    elif drift_kind == "group_IDO_score":
        changed_index = cells.index[group_mask][0]
        cells.at[changed_index, "group_IDO_score"] = np.nextafter(
            float(cells.at[changed_index, "group_IDO_score"]), np.inf
        )
    else:
        changed_index = cells.index[group_mask][0]
        cells.at[changed_index, "group_id"] = "B8_P7_C02"
    drifted_feature_assembly = replace(assembly.feature_assembly, cells=cells)
    drifted_assembly = replace(
        assembly,
        feature_assembly=drifted_feature_assembly,
    )
    core_calls = 0

    def fake_run_rui_cv(*args: object, **kwargs: object) -> object:
        nonlocal core_calls
        core_calls += 1
        return object()

    monkeypatch.setattr(phase2_module, "run_rui_cv", fake_run_rui_cv)

    with pytest.raises(Phase2ContractError, match="canonical Phase 2 targets"):
        run_phase2_cv(
            drifted_assembly,
            context,
            checkpoint_dir=tmp_path / ".phase2_cv_checkpoints",
            phase1_checkpoint_dir=tmp_path / ".cv_checkpoints",
            n_jobs=1,
            importance_repeats=1,
        )

    assert core_calls == 0


def test_phase2_cv_provenance_changes_with_core_source_or_manifest_projection(
    formal_reconstruction: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    population = formal_reconstruction["population"]
    metadata = formal_reconstruction["metadata"]
    context = Phase2LeakageContext.formal(
        LeakageContext(
            standardization_scope="pipeline_fit_within_outer_training",
            feature_selection_inputs=("X",),
            cellpose_invoked=False,
            mask_cache_fingerprint_before=metadata["mask_cache_fingerprint_before"],
            mask_cache_fingerprint_after=metadata["mask_cache_fingerprint_after"],
        )
    )
    baseline = assemble_phase2_cv(population, context)

    manifest_reordered = population.manifest.iloc[::-1].reset_index(drop=True)
    reordered = assemble_phase2_cv(
        replace(population, manifest=manifest_reordered), context
    )
    assert reordered.cv_input_fingerprints["cv_source_fingerprint"] != (
        baseline.cv_input_fingerprints["cv_source_fingerprint"]
    )
    for name in (
        "x_fingerprint",
        "y_fingerprint",
        "group_ids_fingerprint",
        "split_fingerprint",
    ):
        assert reordered.cv_input_fingerprints[name] == (
            baseline.cv_input_fingerprints[name]
        )

    monkeypatch.setattr(
        phase2_module,
        "phase2_cv_source_components",
        lambda: {"phase2.py": "a" * 64, "cv.py": "b" * 64},
    )
    with pytest.raises(Phase2ContractError, match="frozen CV"):
        run_phase2_cv(
            baseline,
            context,
            checkpoint_dir=tmp_path / ".phase2_cv_checkpoints",
            phase1_checkpoint_dir=tmp_path / ".cv_checkpoints",
            n_jobs=1,
            importance_repeats=1,
        )
    source_changed = assemble_phase2_cv(population, context)
    assert source_changed.cv_input_fingerprints["cv_source_fingerprint"] != (
        baseline.cv_input_fingerprints["cv_source_fingerprint"]
    )


def test_phase2_core_has_no_dose_diagnostics_dependency() -> None:
    source = (PROJECT_ROOT / "immunity/exp4/phase2.py").read_text(encoding="utf-8")

    assert "phase2_diagnostics" not in source
    assert "CONDITION_MAPPING_REVERSED_HYPOTHESIS" not in source
    assert "DOSE_MAPPING_NOTE" not in source


def test_phase2_shrinkage_and_within_condition_are_formal_oof_only(
    formal_posthoc_inputs: dict[str, pd.DataFrame],
) -> None:
    oof = formal_posthoc_inputs["oof"]

    shrinkage = analyze_phase2_shrinkage(oof)
    within = build_metrics_within_condition(oof)

    assert shrinkage.report.columns.tolist() == list(PHASE2_SHRINKAGE_COLUMNS)
    assert len(shrinkage.report) == 24
    assert shrinkage.report["n_groups"].eq(72).all()
    assert len(shrinkage.group_summary) == 24 * 72
    assert within.columns.tolist() == list(METRICS_WITHIN_CONDITION_COLUMNS)
    assert len(within) == 24 * 8
    assert within["n_groups"].eq(9).all()
    assert within.groupby("configuration_id")["n_cells"].sum().eq(19_648).all()
    assert set(within["cell_spearman_status"]) <= {
        "defined",
        "undefined_constant_input",
    }
    defined = within["cell_spearman_status"].eq("defined")
    assert np.isfinite(within.loc[defined, "cell_spearman"]).all()
    assert np.isfinite(within.loc[defined, "cell_spearman_p_value"]).all()


def test_dose_response_and_residual_diagnostics_use_explicit_reversed_mapping(
    formal_posthoc_inputs: dict[str, pd.DataFrame],
) -> None:
    dose_response = build_dose_response_check(
        formal_posthoc_inputs["targets"],
        condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
    )
    residual = analyze_residual_vs_dose(
        formal_posthoc_inputs["oof"],
        condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
    )

    assert dose_response.columns.tolist() == list(DOSE_RESPONSE_COLUMNS)
    assert len(dose_response) == 9
    assert dose_response["monotonic_non_decreasing"].sum() == 8
    non_monotonic = dose_response.loc[
        ~dose_response["monotonic_non_decreasing"]
    ].iloc[0]
    assert non_monotonic["donor_passage_id"] == "B7_P6"
    np.testing.assert_allclose(
        non_monotonic.loc[
            [
                "ifn_0_tnf_0_target_median",
                "ifn_25_tnf_0_target_median",
                "ifn_50_tnf_0_target_median",
                "ifn_100_tnf_0_target_median",
            ]
        ].to_numpy(float),
        [0.036036, 1.194734, 0.901134, 2.525080],
        atol=1e-6,
    )
    assert dose_response["dose_mapping_note"].eq(DOSE_MAPPING_NOTE).all()

    assert residual.columns.tolist() == list(RESIDUAL_VS_DOSE_COLUMNS)
    assert len(residual) == 24
    assert residual["n_groups"].eq(72).all()
    assert residual["dose_mapping_note"].eq(DOSE_MAPPING_NOTE).all()
    correlation_columns = [
        column
        for column in residual.columns
        if column.endswith("_rho") or column.endswith("_p_value")
    ]
    assert np.isfinite(residual.loc[:, correlation_columns]).all().all()


def test_residual_vs_dose_rejects_cell_to_group_mapping_drift(
    formal_posthoc_inputs: dict[str, pd.DataFrame],
) -> None:
    oof = formal_posthoc_inputs["oof"].copy()
    configuration_ids = oof["configuration_id"].drop_duplicates().tolist()
    changed_configuration = configuration_ids[1]
    changed_index = oof.index[oof["configuration_id"].eq(changed_configuration)][0]
    destination = oof.loc[
        oof["configuration_id"].eq(changed_configuration)
        & oof["group_id"].ne(oof.at[changed_index, "group_id"])
    ].iloc[0]
    oof.at[changed_index, "group_id"] = destination["group_id"]
    oof.at[changed_index, "confidence_flag"] = destination["confidence_flag"]
    oof.at[changed_index, "observed_ido_score"] = destination["observed_ido_score"]
    oof.at[changed_index, "residual"] = (
        oof.at[changed_index, "predicted_ido_score"]
        - oof.at[changed_index, "observed_ido_score"]
    )

    with pytest.raises(Phase2ContractError, match="cell.*group mapping"):
        analyze_residual_vs_dose(
            oof,
            condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
        )


def test_dose_diagnostics_leave_all_post_cv_numeric_artifacts_bit_identical(
    formal_posthoc_inputs: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    artifact_paths: dict[str, Path] = {}
    for index, name in enumerate(PHASE2_NUMERIC_ARTIFACT_NAMES):
        path = tmp_path / f"{name}.csv"
        path.write_bytes(f"artifact-{index}\n".encode("ascii"))
        artifact_paths[name] = path
    before = fingerprint_phase2_numeric_artifacts(artifact_paths)

    build_dose_response_check(
        formal_posthoc_inputs["targets"],
        condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
    )
    analyze_residual_vs_dose(
        formal_posthoc_inputs["oof"],
        condition_mapping=CONDITION_MAPPING_REVERSED_HYPOTHESIS,
    )
    after = fingerprint_phase2_numeric_artifacts(artifact_paths)

    result = verify_phase2_numeric_artifact_immutability(before, after)
    assert result.passed is True
    assert dict(result.before_sha256) == dict(result.after_sha256)

    artifact_paths["oof_predictions"].write_bytes(b"mutated\n")
    mutated = fingerprint_phase2_numeric_artifacts(artifact_paths)
    with pytest.raises(Phase2ArtifactMutationError, match="oof_predictions"):
        verify_phase2_numeric_artifact_immutability(before, mutated)


def test_phase1_vs_phase2_comparison_joins_exact_arm_model_roster(
    formal_posthoc_inputs: dict[str, pd.DataFrame],
) -> None:
    phase2_shrinkage = analyze_phase2_shrinkage(
        formal_posthoc_inputs["oof"]
    ).report

    comparison = compare_phase1_phase2(
        phase1_metrics=formal_posthoc_inputs["phase1_metrics"],
        phase1_shrinkage=formal_posthoc_inputs["phase1_shrinkage"],
        phase2_metrics=formal_posthoc_inputs["phase2_metrics"],
        phase2_shrinkage=phase2_shrinkage,
    )

    assert comparison.columns.tolist() == list(PHASE1_VS_PHASE2_COMPARISON_COLUMNS)
    assert len(comparison) == 24
    assert not comparison.duplicated(["arm", "model"]).any()
    assert np.allclose(
        comparison["phase2_mean_fold_cell_r2"]
        - comparison["phase1_mean_fold_cell_r2"],
        0.05,
    )
    assert np.allclose(
        comparison["phase2_mean_fold_rui_r2"]
        - comparison["phase1_mean_fold_rui_r2"],
        0.10,
    )


def _formal_phase1_paths() -> dict[str, Path]:
    config = (PROJECT_ROOT / "immunity/configs/exp4_rui2025.yaml").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^source_root:\s*'([^']+)'", config, flags=re.MULTILINE)
    if match is None:
        raise AssertionError("exp4_rui2025.yaml 缺少 source_root")
    source_root = Path(match.group(1))
    output_root = PROJECT_ROOT / "immunity/outputs/exp4"
    return {
        "data_manifest.csv": source_root / "immunity/outputs/exp3/data_manifest.csv",
        "cell_level_basic.csv": (
            source_root / "immunity/outputs/exp3/feature_cache/cell_level_basic.csv"
        ),
        "cell_dedup_report.csv": output_root / "cell_dedup_report.csv",
        "cell_level_rui49.csv": output_root / "cell_level_rui49.csv",
        "feature_health_report.csv": output_root / "feature_health_report.csv",
        "feature_redundancy_report.csv": output_root / "feature_redundancy_report.csv",
        "oof_predictions.csv": output_root / "oof_predictions.csv",
        "run_metadata.json": output_root / "run_metadata.json",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
