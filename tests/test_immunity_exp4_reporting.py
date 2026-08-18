from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _current_targets() -> pd.DataFrame:
    """建立涵蓋 target range、相鄰 gap 與 border delta 的 current fixture。"""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 2.00995757995635]
    deltas = {
        "B4_P6": 0.01,
        "B4_P7": 0.02,
        "B7_P6": 0.03,
        "B7_P7": 0.04,
        "B8_P5": 0.05,
        "B8_P6": 0.1269336253,
        "B8_P7": -0.01,
        "B9_P6": -0.02,
        "B9_P7": 0.06,
    }
    rows = []
    for index, (group_id, score) in enumerate(zip(deltas, scores)):
        delta = deltas[group_id]
        rows.append(
            {
                "group_id": group_id,
                "b_id": group_id.split("_")[0],
                "passage": int(group_id.split("_")[1][1:]),
                "fov_count": 80,
                "cells_before": 100 + index,
                "cells_after": 90 + index,
                "group_IDO_score": score,
                "group_IDO_score_all_cells": score - delta,
                "delta": delta,
                "confidence_flag": "low" if group_id == "B8_P7" else "normal",
            }
        )
    return pd.DataFrame(rows)


def test_report_derives_current_target_scale_and_border_bias() -> None:
    """REPORT 必須從 current target table 計算 range、gap 與 delta 符號。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        targets=_current_targets(),
        metadata={"target_status": "validated"},
    )

    assert "1.90995758" in report
    assert "約 1.91" in report
    assert "gaps <0.2：6/8" in report
    assert "sorted targets（ascending）" in report
    assert "delta 定義：post-border − pre-border；符號：7 正、2 負、0 零" in report
    assert "B8_P6" in report
    assert "+0.1269336253" in report
    assert "6.6459%" in report
    assert "7 donor × 323 cells" in report
    assert "cell-level 5-fold CV 會被 B4_P6 與" in report
    assert "B7_P6 主導" in report
    assert "不能把高樣本數組別的影響當成 donor-level 證據" in report


def test_report_uses_current_target_range_in_r2_boundary_and_not_run_without_targets() -> None:
    """R² 解讀的 target range 必須跟著 current table 或 not_run 變化。"""
    from immunity.exp4.reporting import build_report_skeleton

    changed_targets = _current_targets()
    changed_targets["group_IDO_score"] *= 2.5
    changed_report = build_report_skeleton(targets=changed_targets)
    assert (
        "未來無論 R² 高低，均須與 4.77489395 灰階 target range"
        in changed_report
    )

    missing_report = build_report_skeleton()
    assert "未來無論 R² 高低，均須與 not_run target range" in missing_report


def test_report_renders_background_provenance_and_full_feature_evidence() -> None:
    """full metadata 與 pilot/full background provenance 必須獨立呈現。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "feature_smoke_status": "validated",
            "smoke_test": {"status": "passed", "image_count": 5},
            "feature_health_smoke": {
                "health_scope": "master_labels_after_border_exclusion",
                "master_post_border_label_count": 95,
                "removed_columns": ["cell__MinIntensity"],
                "fallback_rates": {"cell__Area": 0.0},
            },
            "full_rui49": {
                "status": "validated",
                "raw_pair_row_count": 23976,
                "canonical_feature_count": 49,
                "pre_border_distinct_cell_count": 23012,
                "retained_cell_count": 19648,
                "fov_count": 693,
                "elapsed_seconds": 12.345,
                "cell_level_path": "cell_level_rui49.csv",
                "feature_health_report_path": "feature_health_report.csv",
                "healthy_feature_count": 48,
                "removed_zero_variance_columns": ["cell__MinIntensity"],
                "near_zero_variance_columns": [],
                "nonfinite_columns": [],
                "max_fallback_rate": 0.0,
                "retained_consistency": {
                    "status": "passed",
                    "raw_pair_row_count": 20440,
                    "checked_cell_count": 19648,
                    "fov_count": 693,
                    "canonical_feature_count": 49,
                },
            },
            "ido_background_qc": {
                "summary": {
                    "pilot_80_fov": (
                        "3 distinct normalized values: 8/255, 9/255, 10/255; "
                        "78/80=97.5% in 9-10"
                    ),
                    "full_693": "11 distinct raw-gray values: 7-17",
                },
                "source": {
                    "pilot_80_fov": "immunity/outputs/b4_p6/cell_level_features.csv",
                    "full_693": "immunity/outputs/exp3/segmentation_qc.csv",
                },
                "metric": {
                    "pilot_80_fov": "IDO_background_median (normalized [0,1])",
                    "full_693": "extraction_background_median (raw gray)",
                },
            },
        }
    )

    assert "full extraction status：validated" in report
    assert "23,976 raw output rows × 49 canonical features" in report
    assert "retained 19,648/693" in report
    assert "full retained exact consistency：passed" in report
    assert "唯一 removed zero-variance：cell__MinIntensity" in report
    assert "48 healthy" in report
    assert "nonfinite columns：none" in report
    assert "near-zero variance columns：none" in report
    assert "max fallback rate：0%" in report
    assert "elapsed：12.345 s" in report
    assert "cell_level_rui49.csv" in report
    assert "feature_health_report.csv" in report
    assert (
        "pilot 80-FOV evidence：summary=3 distinct normalized values: 8/255, 9/255, "
        "10/255; 78/80=97.5% in 9-10；"
        "source=immunity/outputs/b4_p6/cell_level_features.csv；"
        "metric=IDO_background_median (normalized [0,1])" in report
    )
    assert (
        "full 693-FOV evidence：summary=11 distinct raw-gray values: 7-17；"
        "source=immunity/outputs/exp3/segmentation_qc.csv；"
        "metric=extraction_background_median (raw gray)" in report
    )
    assert "IDO_background_median" in report
    assert "extraction_background_median" in report


def test_report_renders_native_retained_consistency_payload() -> None:
    """REPORT 必須可直接呈現 consistency producer 的 native ``to_dict``。"""
    from immunity.exp4.cell_dedup import WholeCellFeatureConsistencyResult
    from immunity.exp4.reporting import build_report_skeleton

    native_payload = WholeCellFeatureConsistencyResult(
        scope="full_retained_19648_693",
        checked_row_count=20440,
        checked_cell_count=19648,
        checked_duplicate_key_count=792,
        feature_count=49,
    ).to_dict()
    report = build_report_skeleton(
        metadata={
            "full_rui49": {
                "status": "validated",
                "retained_consistency": native_payload,
            }
        }
    )

    assert (
        "full retained exact consistency：passed（scope=full_retained_19648_693；"
        "20,440 raw pairs／19,648 cells／792 duplicate keys／49 features）"
        in report
    )


def test_report_keeps_full_feature_evidence_not_run_without_smoke_substitution() -> None:
    """full 尚未執行時不得把 smoke health 當作 full evidence。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "feature_smoke_status": "validated",
            "smoke_test": {"status": "passed", "image_count": 5},
            "feature_health_smoke": {
                "health_scope": "master_labels_after_border_exclusion",
                "master_post_border_label_count": 95,
            },
            "full_rui49": {
                "status": "not_run",
                "raw_pair_row_count": 23976,
                "canonical_feature_count": 49,
                "retained_cell_count": 19648,
                "cell_level_path": "stale-full.csv",
                "retained_consistency": {
                    "scope": "stale_full_payload",
                    "checked_row_count": 20440,
                    "checked_cell_count": 19648,
                    "feature_count": 49,
                },
            },
        }
    )

    full_section = report.split("## Full Rui49 feature evidence", 1)[1].split(
        "## 尚待完成", 1
    )[0]
    assert "full extraction status：not_run" in full_section
    assert "full 693-FOV feature evidence：not_run；尚未完成" in full_section
    assert "23,976 raw output rows × 49 canonical features" not in full_section
    assert "stale-full.csv" not in full_section
    assert "stale_full_payload" not in full_section
    assert "5-image current smoke：passed" in report
    assert "95 post-border cells" in report


def test_report_discloses_feret_deviation_and_effective_feature_arms() -> None:
    """REPORT 必須揭露 Feret 偏離與 MinIntensity 移除後的四個 arm。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "deviations": [
                {
                    "id": "feret_caliper",
                    "original_behavior": (
                        "沿用既有 cell__feret_width；cv2.fitEllipse minor-axis proxy"
                    ),
                    "new_behavior": (
                        "common pixel-center convex-hull caliper Min/Max Feret + "
                        "regionprops eccentricity"
                    ),
                    "reason": (
                        "舊 proxy 可產生 MinFeret > MaxFeret 且極端失真；"
                        "新 invariant fail-closed"
                    ),
                }
            ],
            "feature_arms": {
                "geometry_24": {"count": 24},
                "rui_48": {"count": 48},
                "rui_filtered": {"count": "pending"},
                "rui_48_plus_nucleus": {"count": 65},
            },
            "removed_zero_variance_columns": ["cell__MinIntensity"],
        }
    )

    assert "Feret deviation" in report
    assert "cell__feret_width" in report
    assert "convex-hull caliper" in report
    assert "MinFeret > MaxFeret" in report
    assert "geometry_24" in report
    assert "rui_48" in report
    assert "rui_filtered" in report
    assert "rui_48_plus_nucleus" in report
    assert "49→48" in report
    assert "cell__MinIntensity" in report


def test_report_marks_missing_dapi_label_cache_as_e2_blocked() -> None:
    """缺少 DAPI label cache 時 REPORT 不得產生虛構 Dice／IoU。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "e2": {
                "status": "blocked",
                "expected_fov_count": 693,
                "validated_fov_count": 0,
                "reason": "precomputed DAPI label masks unavailable",
            }
        }
    )

    assert "E2 status：blocked" in report
    assert "validated FOV=0/693" in report
    assert "precomputed DAPI label masks unavailable" in report
    assert "未產生 Dice／IoU 數值" in report


def test_report_uses_v5_canceled_e2_and_mandatory_arm4_boundary() -> None:
    """v5 REPORT 必須呈現取消 E2、sanity threshold 與 arm4 不對稱邊界。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "e2": {
                "status": "canceled_v5_no_dapi_acquired",
                "expected_fov_count": 693,
                "validated_fov_count": 0,
                "reason": "the nine Exp4 datasets never acquired DAPI",
                "provenance": {
                    "dataset_count": 9,
                    "fov_count": 693,
                    "channels_observed": ["IDO", "PC"],
                },
            },
            "near_zero_variance_threshold": 0.05,
            "near_zero_variance_action": "record_only",
            "near_zero_variance_columns": ["nucleus__solidity"],
            "e3_arm4_boundary": {
                "status": "mandatory",
                "text": (
                    "Arm 4 若勝過 rui_48，只能說 phase-derived 核代理特徵帶入額外資訊；"
                    "若未勝過，不可推論核沒有用。"
                ),
            },
        }
    )

    assert "canceled_v5_no_dapi_acquired" in report
    assert "the nine Exp4 datasets never acquired DAPI" in report
    assert "dataset_count=9" in report
    assert "near-zero variance threshold：CV < 0.05" in report
    assert "record_only" in report
    assert "nucleus__solidity" in report
    assert "Arm 4 結論邊界（mandatory）" in report
    assert "不可推論核沒有用" in report


def test_report_records_v5_analysis_status_without_cv_interpretation() -> None:
    """CV 完成後 REPORT 只列 status/path，不在 reporting layer 解讀結果。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "analysis_status": {
                "feature_health": "validated",
                "nucleus_sanity": "validated",
                "feature_redundancy": "validated",
                "leakage_preflight": "validated",
                "cv": "validated",
                "umap": "not_run",
                "kmeans": "not_run",
            },
            "feature_health_report_path": "feature_health_report.csv",
            "nucleus_sanity_report_path": "nucleus_sanity_report.csv",
            "feature_redundancy_report_path": "feature_redundancy_report.csv",
            "leakage_preflight_report_path": "leakage_preflight_report.csv",
            "cv_metrics_path": "cv_metrics.csv",
        }
    )

    assert "model/CV 結果：validated" in report
    assert "feature_health：status=validated" in report
    assert "cv：status=validated；path=cv_metrics.csv" in report
    assert "UMAP：status=not_run；k-means：status=not_run" in report
    assert "本報告不寫解讀結論" in report


def test_report_renders_validated_post_cv_evidence_and_replaces_placeholders() -> None:
    """validated post-CV metadata 必須呈現證據並移除 stale pending wording。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "analysis_status": {
                "cv": "validated",
                "umap": "validated",
                "kmeans": "validated",
                "shrinkage": "validated",
                "orientation_marginal": "validated",
            },
            "analysis_no_umap_kmeans": False,
            "cv_metrics_path": "cv_metrics.csv",
            "deviations": {
                "feret_caliper": {"id": "feret_caliper"},
            },
            "post_cv_analysis": {
                "status": "validated",
                "best_configuration": {
                    "configuration_id": "rui_48_plus_nucleus__MLPR",
                    "mean_rui_r2": 0.1888487171,
                    "mean_cell_r2": 0.184,
                    "slope": 0.1888487171,
                    "intercept": 1.64,
                    "observed_min": 0.54782921996,
                    "observed_max": 2.45778679992,
                    "predicted_min": 1.36198978007,
                    "predicted_max": 1.81942595568,
                    "observed_range": 1.90995758,
                    "predicted_range": 0.45743618,
                },
                "orientation": {
                    "geometry_24_mean_rfr_rank": 6.4,
                    "rui_filtered_mean_rfr_rank": 7.0,
                    "rui_48_plus_nucleus_mean_rfr_rank": 6.2,
                    "cell_spearman_r": 0.0079756,
                    "group_spearman_r": 0.2166667,
                    "bin_target_median_range": 0.0,
                    "bin_target_mean_range": 0.06530196,
                    "n_bins": 12,
                },
                "umap": {
                    "cell_count": 19648,
                    "fov_count": 693,
                    "feature_count": 30,
                    "ifn_selected_cell_count": 12705,
                    "ifn_selector_untrusted": True,
                    "cluster_counts": {"0": 6000, "1": 6705},
                    "condition_warning": "condition label trust unconfirmed; visualization only",
                    "figure_paths": {
                        "umap_by_condition.png": "figures/umap_by_condition.png"
                    },
                },
                "outputs": {
                    "umap_embeddings_path": "umap_embeddings.csv",
                    "kmeans_cluster_features_path": "kmeans_cluster_features.csv",
                },
                "mlpr_convergence": {
                    "max_iter": 200,
                    "status": "not_converged_at_anchor_limit",
                    "warning": "ConvergenceWarning observed during formal CV",
                },
            },
        }
    )

    assert "模型只還原了約 19% 的組間差異。" in report
    assert "1.90995758" in report
    assert "ConvergenceWarning" in report
    assert "Orientation" in report
    assert "bounded" in report or "有界" in report
    assert "umap_by_condition.png" in report
    assert "condition label trust unconfirmed; visualization only" in report
    assert "putative-IFN mapping is unconfirmed and untrusted" in report
    assert "visualization only; no dose conclusions are allowed" in report
    assert "尚待完成：UMAP、k-means" not in report
    assert "本報告不寫解讀結論" not in report
    assert "no meaningful univariate marginal directionality detected" in report
    assert "flat 12-bin medians and near-zero cell rho" in report
    assert "fit noise and/or multivariate interaction" in report
    assert "not evidence of true directionality" in report
    assert "no causal/biological inference" in report
    post_cv = report.split("## Post-CV diagnostics", 1)[1]
    shrinkage = post_cv.split("- shrinkage OLS", 1)[1]
    assert shrinkage.splitlines()[1].startswith("- Arm 4 boundary")
    r2_line_index = next(
        index
        for index, line in enumerate(report.splitlines())
        if "formal CV 的 best configuration 為" in line
    )
    assert "Arm 4 boundary" in report.splitlines()[r2_line_index + 1]
    assert "phase-derived nucleus proxy" in report
    assert "true nucleus features are useful" in report
    assert "cannot distinguish biology from a failed proxy" in report
    assert "Klinker" not in report.split("## Post-CV diagnostics", 1)[1].split(
        "### Orientation marginal", 1
    )[0]
    for preserved in ("E2", "Feret", "target", "background", "B8_P7"):
        assert preserved in report


def test_report_uses_current_formal_health_instead_of_archived_full_health() -> None:
    """正式 CV health 必須使用 current canonical evidence，不混用 archived full path。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "analysis_status": {
                "cv": "validated",
                "umap": "not_run",
                "kmeans": "not_run",
            },
            "full_rui49": {
                "status": "validated",
                "raw_pair_row_count": 23976,
                "canonical_feature_count": 49,
                "pre_border_distinct_cell_count": 23012,
                "retained_cell_count": 19648,
                "fov_count": 693,
                "feature_health_report_path": (
                    "feature_health_report.csv.historical_stale.full"
                ),
                "healthy_feature_count": 48,
                "removed_zero_variance_columns": ["cell__MinIntensity"],
                "near_zero_variance_columns": [],
            },
            "feature_health_status": "validated",
            "feature_health_report_path": "feature_health_report.csv",
            "feature_health": {
                "status": "validated",
                "scope": "cv_formal_retained_19648_cells_65_active_features",
                "input_feature_count": 66,
                "active_feature_count": 65,
                "zero_variance_removed_columns": ["cell__MinIntensity"],
                "near_zero_variance_columns": [
                    "cell__InfoMeas2",
                    "nucleus__solidity",
                ],
                "fallback_rates": {"cell__Area": 0.0},
            },
        }
    )

    full_section = report.split("## Full Rui49 feature evidence", 1)[1].split(
        "## 已知實作偏離", 1
    )[0]
    assert "feature_health_report.csv。" in full_section
    assert "feature_health_report.csv.historical_stale.full" not in full_section
    assert "66 checked→65 active" in full_section
    assert "cell__InfoMeas2、nucleus__solidity" in full_section
    assert "near-zero variance columns：none" not in full_section
    assert "max fallback rate：0%" in full_section


def test_report_pending_section_excludes_validated_cv_and_conditions_are_future_only() -> None:
    """CV 完成後尚待完成只列 UMAP/k-means，condition boundary 必須是 future conditional。"""
    from immunity.exp4.reporting import build_report_skeleton

    report = build_report_skeleton(
        metadata={
            "analysis_status": {
                "cv": "validated",
                "umap": "not_run",
                "kmeans": "not_run",
            }
        }
    )
    pending = report.split("## 尚待完成", 1)[1]
    assert "model、OOF、CV" not in pending
    assert "UMAP" in pending and "k-means" in pending
    assert "若後續產生 umap_by_condition.png" in report


def _formal_refresh_fixture(tmp_path: Path) -> dict[str, Path]:
    """建立 reporting-only refresh 的 current canonical fixture。"""
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    output_root = tmp_path / "outputs" / "exp4"
    output_root.mkdir(parents=True)
    metadata_path = output_root / "run_metadata.json"
    report_path = output_root / "REPORT.md"
    health_path = output_root / "feature_health_report.csv"
    metrics_path = output_root / "cv_metrics.csv"
    stale_health_path = output_root / "feature_health_report.csv.historical_stale.full"

    nucleus_features = tuple(
        f"nucleus__{name}"
        for name in (
            "area", "compactness", "eccentricity", "extent", "sphericity",
            "major_axis_length", "feret_length", "minor_axis_length", "feret_width",
            "maximum_radius", "mean_radius", "median_radius", "aspect_ratio",
            "perimeter_area_ratio", "perimeter", "solidity",
        )
    ) + ("nucleus_cytoplasm_area_ratio",)
    features = list(RUI49_FEATURE_COLUMNS) + list(nucleus_features)
    near_zero = {"cell__InfoMeas2", "nucleus__solidity"}
    health = pd.DataFrame(
        {
            "feature": features,
            "relative_std": [
                0.01 if feature in near_zero else 0.1 for feature in features
            ],
            "is_zero_variance": [feature == "cell__MinIntensity" for feature in features],
            "is_near_zero_variance": [feature in near_zero for feature in features],
            "fallback_rate": [0.0] * len(features),
        }
    )
    health.to_csv(health_path, index=False)
    pd.DataFrame({"configuration_id": range(120)}).to_csv(metrics_path, index=False)
    report_path.write_text("old report\n", encoding="utf-8")
    stale_health_path.write_text("historical full health\n", encoding="utf-8")

    metadata = {
        "status": "validated",
        "report_path": str(report_path),
        "feature_health_report_path": str(health_path),
        "cv_metrics_path": str(metrics_path),
        "feature_health_status": "validated",
        "near_zero_variance_check_status": "not_run",
        "near_zero_variance_columns": [],
        "analysis_status": {
            "feature_health": "validated",
            "nucleus_sanity": "validated",
            "feature_redundancy": "validated",
            "leakage_preflight": "validated",
            "cv": "validated",
            "umap": "not_run",
            "kmeans": "not_run",
        },
        "cv": {
            "status": "validated",
            "cell_count": 19648,
            "fov_count": 693,
            "metrics_row_count": 120,
            "metrics_path": str(metrics_path),
        },
        "full_retained_cell_count": 19648,
        "full_fov_count": 693,
        "full_rui49": {
            "status": "validated",
            "feature_health_report_path": str(stale_health_path),
            "feature_health_report_scope": "full_extraction_49_columns",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "root": output_root,
        "metadata": metadata_path,
        "report": report_path,
        "health": health_path,
        "metrics": metrics_path,
    }


def test_refresh_formal_metadata_and_report_reconciles_current_health_without_numeric_writes(
    tmp_path: Path,
) -> None:
    """refresh 只更新 metadata/REPORT，並把 formal health 與歷史 scope 分離。"""
    from immunity.exp4.reporting import refresh_formal_metadata_and_report

    paths = _formal_refresh_fixture(tmp_path)
    health_before = paths["health"].read_bytes()
    metrics_before = paths["metrics"].read_bytes()

    result = refresh_formal_metadata_and_report(
        metadata_path=paths["metadata"],
        feature_health_path=paths["health"],
        report_path=paths["report"],
    )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")
    assert result["status"] == "validated"
    assert metadata["near_zero_variance_check_status"] == "validated"
    assert metadata["near_zero_variance_columns"] == [
        "cell__InfoMeas2",
        "nucleus__solidity",
    ]
    assert metadata["feature_health"]["report_path"] == str(paths["health"].resolve())
    assert metadata["feature_health"]["input_feature_count"] == 66
    assert metadata["feature_health"]["active_feature_count"] == 65
    assert metadata["feature_health"]["near_zero_variance_columns"] == [
        "cell__InfoMeas2",
        "nucleus__solidity",
    ]
    assert metadata["full_rui49"]["feature_health_report_status"] == "historical_stale"
    assert metadata["full_rui49"]["feature_health_report_scope"] == (
        "historical_stale_full_extraction_49_columns"
    )
    assert "66 checked→65 active" in report
    assert "feature_health_report.csv。" in report
    assert "historical_stale.full" not in report
    assert "尚待完成：UMAP、k-means" in report
    assert health_before == paths["health"].read_bytes()
    assert metrics_before == paths["metrics"].read_bytes()


def test_refresh_formal_metadata_and_report_persists_untrusted_ifn_mapping(
    tmp_path: Path,
) -> None:
    """validated post-CV refresh 必須持久化 untrusted putative-IFN provenance。"""
    from immunity.exp4.reporting import refresh_formal_metadata_and_report

    paths = _formal_refresh_fixture(tmp_path)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["analysis_status"].update(
        {
            "umap": "validated",
            "kmeans": "validated",
            "shrinkage": "validated",
            "orientation_marginal": "validated",
        }
    )
    metadata["post_cv_analysis"] = {
        "status": "validated",
        "umap": {},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    refresh_formal_metadata_and_report(
        metadata_path=paths["metadata"],
        feature_health_path=paths["health"],
        report_path=paths["report"],
    )

    refreshed = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert refreshed["ifn_selector_untrusted"] is True
    assert refreshed["post_cv_analysis"]["ifn_selector_untrusted"] is True
    assert refreshed["post_cv_analysis"]["umap"]["ifn_selector_untrusted"] is True
    report = paths["report"].read_text(encoding="utf-8")
    assert "putative-IFN mapping is unconfirmed and untrusted" in report
    assert "no dose conclusions are allowed" in report


def test_refresh_formal_metadata_and_report_rolls_back_on_report_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第二個 replace 失敗時 metadata 與 REPORT 必須都回復原內容。"""
    from immunity.exp4 import reporting

    paths = _formal_refresh_fixture(tmp_path)
    metadata_before = paths["metadata"].read_bytes()
    report_before = paths["report"].read_bytes()
    real_replace = reporting.os.replace
    state = {"failed": False}

    def fail_report_commit(source: object, destination: object) -> None:
        destination_path = Path(destination)
        source_name = Path(source).name
        if (
            destination_path.resolve() == paths["report"].resolve()
            and ".refresh." in source_name
            and not state["failed"]
        ):
            state["failed"] = True
            raise OSError("synthetic REPORT replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", fail_report_commit)
    with pytest.raises(OSError, match="synthetic REPORT replace failure"):
        reporting.refresh_formal_metadata_and_report(
            metadata_path=paths["metadata"],
            feature_health_path=paths["health"],
            report_path=paths["report"],
        )
    assert paths["metadata"].read_bytes() == metadata_before
    assert paths["report"].read_bytes() == report_before


def test_refresh_formal_metadata_and_report_rejects_unvalidated_cv(
    tmp_path: Path,
) -> None:
    """refresh 必須 fail-closed，不得替未完成 CV 生成 metadata。"""
    from immunity.exp4.reporting import refresh_formal_metadata_and_report

    paths = _formal_refresh_fixture(tmp_path)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata["analysis_status"]["cv"] = "not_run"
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cv validated"):
        refresh_formal_metadata_and_report(
            metadata_path=paths["metadata"],
            feature_health_path=paths["health"],
            report_path=paths["report"],
        )


def test_refresh_formal_cli_exposes_reporting_only_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI path 必須只呼叫 reporting-only refresh seam。"""
    from immunity.exp4.reporting import _refresh_cli

    paths = _formal_refresh_fixture(tmp_path)
    assert (
        _refresh_cli(
            [
                "--refresh-formal",
                "--metadata",
                str(paths["metadata"]),
                "--feature-health",
                str(paths["health"]),
                "--report",
                str(paths["report"]),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["numeric_artifacts_modified"] is False


def test_refresh_formal_cli_resolves_explicit_paths_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI 明確給定的相對路徑必須依 cwd 解讀，不可重複拼 metadata.parent。"""
    from immunity.exp4.reporting import _refresh_cli

    paths = _formal_refresh_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative = Path("outputs") / "exp4"
    assert (
        _refresh_cli(
            [
                "--refresh-formal",
                "--metadata",
                str(relative / "run_metadata.json"),
                "--feature-health",
                str(relative / "feature_health_report.csv"),
                "--report",
                str(relative / "REPORT.md"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def test_refresh_formal_subprocess_cli_accepts_worktree_relative_paths(
    tmp_path: Path,
) -> None:
    """真正的 ``python -m`` CLI 也必須接受 worktree/cwd-relative paths。"""
    import os
    import subprocess
    import sys

    paths = _formal_refresh_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    relative = Path("outputs") / "exp4"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "immunity.exp4.reporting",
            "--refresh-formal",
            "--metadata",
            str(relative / "run_metadata.json"),
            "--feature-health",
            str(relative / "feature_health_report.csv"),
            "--report",
            str(relative / "REPORT.md"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "validated"
