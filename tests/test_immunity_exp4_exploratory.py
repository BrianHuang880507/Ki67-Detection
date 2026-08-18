from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


FEATURES = tuple(f"cell__exploratory_{index:02d}" for index in range(30))


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """建立小型但完整的 exploratory seam fixture。"""
    image_keys = ("FOV_A", "FOV_B", "FOV_C")
    manifest = pd.DataFrame(
        {
            "image_key": image_keys,
            "b_id": ("B4", "B7", "B8"),
            "passage": (5, 6, 7),
            "condition": ("control", "ifn", "ifn"),
            "ifn_dose": (0.0, 10.0, 100.0),
            "tnf_dose": (0.0, 0.0, 0.0),
            "group_id": ("G1", "G2", "G3"),
        }
    )
    rows: list[dict[str, object]] = []
    for fov_index, image_key in enumerate(image_keys):
        for cell_label in range(1, 5):
            row: dict[str, object] = {
                "image_key": image_key,
                "cell_label": cell_label,
                "group_IDO_score": float((fov_index + 1) * 10),
            }
            row.update(
                {
                    feature: float(fov_index * 2 + cell_label + offset / 100.0)
                    for offset, feature in enumerate(FEATURES)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), manifest


def _run(cells: pd.DataFrame, manifest: pd.DataFrame):
    from immunity.exp4.exploratory import run_exploratory_analysis

    return run_exploratory_analysis(
        cells,
        FEATURES,
        manifest,
        expected_cell_count=len(cells),
        expected_fov_count=manifest["image_key"].nunique(),
        expected_group_count=manifest["group_id"].nunique(),
    )


def test_exploratory_result_has_embedding_clusters_report_and_pngs() -> None:
    cells, manifest = _inputs()

    result = _run(cells, manifest)

    assert tuple(result.embedding.columns) == (
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
    assert result.embedding.shape == (12, 13)
    assert result.embedding["is_putative_ifn_stimulated"].sum() == 8
    assert result.embedding.loc[
        ~result.embedding["is_putative_ifn_stimulated"],
        ["kmeans_cluster_id", "kmeans_cluster_label"],
    ].isna().all().all()
    assert result.cluster_assignments["image_key"].nunique() == 2
    assert set(result.cluster_assignments["kmeans_cluster_label"]) == {
        "IDO_high",
        "IDO_low",
    }
    assert tuple(result.cluster_feature_report.columns) == (
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
    assert result.cluster_feature_report["feature"].tolist() == list(FEATURES)
    assert len(result.cluster_feature_report) == 30
    assert set(result.figures) == {
        "umap_by_donor.png",
        "umap_by_passage.png",
        "umap_by_condition.png",
    }
    assert all(result.figures[name].startswith(b"\x89PNG") for name in result.figures)


def test_exploratory_uses_exact_x_only_features_and_rejects_metadata_leakage() -> None:
    cells, manifest = _inputs()
    from immunity.exp4.exploratory import ExploratoryContractError

    with pytest.raises(ExploratoryContractError, match="metadata/target"):
        from immunity.exp4.exploratory import run_exploratory_analysis

        run_exploratory_analysis(
            cells,
            (*FEATURES[:-1], "group_IDO_score"),
            manifest,
            expected_cell_count=len(cells),
            expected_fov_count=3,
            expected_group_count=3,
        )


def test_exploratory_fails_closed_on_nonfinite_x_and_non_one_to_one_manifest_join() -> None:
    cells, manifest = _inputs()
    from immunity.exp4.exploratory import ExploratoryContractError

    cells.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ExploratoryContractError, match="finite"):
        _run(cells, manifest)

    cells, manifest = _inputs()
    duplicate = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(ExploratoryContractError, match="unique"):
        _run(cells, duplicate)


def test_exploratory_umap_and_kmeans_are_deterministic_and_target_is_posthoc_only() -> None:
    cells, manifest = _inputs()
    first = _run(cells, manifest)

    altered = cells.copy()
    altered["group_IDO_score"] = altered["group_IDO_score"] + 1000.0
    second = _run(altered, manifest)

    np.testing.assert_allclose(
        first.embedding[["UMAP1", "UMAP2"]],
        second.embedding[["UMAP1", "UMAP2"]],
    )
    assert first.cluster_assignments["kmeans_cluster_id"].tolist() == second.cluster_assignments[
        "kmeans_cluster_id"
    ].tolist()
    assert first.figures == second.figures
    assert first.provenance["random_state"] == 42
    assert first.provenance["kmeans_fit_columns"] == ("UMAP1", "UMAP2")
    assert first.provenance["target_used_for_fit"] is False
    assert first.provenance["ifn_selector_untrusted"] is True
    assert first.provenance["ifn_selected_cell_count"] == 8


def test_exploratory_uses_manifest_labels_when_cell_table_carries_stale_metadata() -> None:
    cells, manifest = _inputs()
    cells["condition"] = "stale"
    cells["ifn_dose"] = 0.0
    cells["group_id"] = "stale"

    result = _run(cells, manifest)

    assert result.embedding["condition"].tolist() == [
        "control",
        "control",
        "control",
        "control",
        "ifn",
        "ifn",
        "ifn",
        "ifn",
        "ifn",
        "ifn",
        "ifn",
        "ifn",
    ]
    assert result.embedding["is_putative_ifn_stimulated"].sum() == 8


def test_exploratory_defaults_fail_closed_on_nonformal_population() -> None:
    cells, manifest = _inputs()
    from immunity.exp4.exploratory import ExploratoryContractError

    with pytest.raises(ExploratoryContractError, match="expected 19648"):
        from immunity.exp4.exploratory import run_exploratory_analysis

        run_exploratory_analysis(cells, FEATURES, manifest)


def test_exploratory_condition_warning_is_exact_and_recorded_as_visible_annotation() -> None:
    cells, manifest = _inputs()
    result = _run(cells, manifest)
    expected = "condition 標籤可信度未確認；僅作視覺化，不得據以做劑量相關結論。"

    assert result.provenance["condition_warning"] == expected
    assert result.provenance["figure_annotations"]["umap_by_condition.png"] == expected
    assert expected.encode("utf-8") in result.figures["umap_by_condition.png"]


def test_exploratory_fails_closed_when_posthoc_target_medians_tie() -> None:
    cells, manifest = _inputs()
    cells["group_IDO_score"] = 1.0
    from immunity.exp4.exploratory import ExploratoryContractError

    with pytest.raises(ExploratoryContractError, match="tied"):
        _run(cells, manifest)
