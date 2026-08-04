"""Exp3 reporting artifacts 的固定契約與隔離測試。"""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import immunity.exp3.run_benchmark as run_module
from immunity.exp3.reporting import (
    REQUIRED_FIGURES,
    REQUIRED_TABLES,
    write_experiment_record,
    write_figures,
    write_result_tables,
)


def _allowed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """建立測試專用的 Exp3 output root 與 run 目錄。"""
    root = (tmp_path / "immunity" / "outputs" / "exp3").resolve()
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", root)
    return root, root / "synthetic-run"


def _report_tables() -> dict[str, pd.DataFrame]:
    """建立涵蓋完整固定表格集合的最小資料。"""
    return {
        name: pd.DataFrame([{"status": "synthetic"}]) for name in REQUIRED_TABLES
    }


def _formal_feature_registry() -> dict[str, object]:
    """鏡像 feature aggregation 寫出的 formal registry schema。"""
    return {
        "basic_median": {
            "predictor_columns": ["cell__area__median"],
            "predictor_count": 33,
            "aggregation": "median",
            "replication_scope": "image",
        }
    }


def _no_winner_context() -> dict[str, object]:
    """建立技術記錄用的 no-winner synthetic context。"""
    return {
        "status": "no_eligible_phase_only_model",
        "winner": None,
        "raw_pc": 720,
        "raw_ido": 719,
        "complete_pairs": 719,
        "exclusions": ["B7-P5/C04/F10: missing IDO pair"],
        "segmentation_pass_rate": 0.98,
        "ranking": pd.DataFrame(
            [
                {
                    "model": "ridge",
                    "eligible": False,
                    "average_rank": 1.5,
                    "ineligibility_reasons_json": '["MAE 未優於 Dummy"]',
                }
            ]
        ),
        "fold_metrics": pd.DataFrame(
            [
                {
                    "validation": "leave_one_b_out",
                    "model": "ridge",
                    "mae": 1.2,
                    "rmse": 1.6,
                    "r2": -0.1,
                    "spearman": 0.2,
                }
            ]
        ),
        "limitations": ["IDO fluorescence is an IDO proxy."],
        "metadata": {
            "git_commit": "synthetic",
            "dirty": False,
            "config_hash": "config-sha",
            "manifest_hash": "manifest-sha",
            "python_version": "3.10.synthetic",
            "package_versions": {"pandas": "synthetic"},
            "seed": 42,
            "runtime_seconds": 1.25,
        },
    }


def _figure_artifacts() -> dict[str, object]:
    """建立八張圖皆可繪製的 deterministic synthetic artifact。"""
    validations = (
        "leave_one_b_out",
        "leave_one_passage_out",
        "leave_one_group_out",
        "leave_one_condition_out",
    )
    models = ("ridge", "dummy_median", "dose_ridge", "dose_plus_morphology_ridge")
    ranking = pd.DataFrame(
        [
            {
                "model": model,
                "role": "candidate" if model == "ridge" else "diagnostic",
                "eligible": False,
                "average_rank": index + 1.0,
                **{
                    f"{validation}_rank": float(index + offset + 1)
                    for offset, validation in enumerate(validations)
                },
            }
            for index, model in enumerate(models[:2])
        ]
    )
    fold_metrics = pd.DataFrame(
        [
            {
                "validation": validation,
                "fold": fold,
                "model": model,
                "role": "candidate" if model == "ridge" else "diagnostic",
                "mae": 0.8 + model_index * 0.2 + fold * 0.04,
                "rmse": 1.0 + model_index * 0.2,
                "r2": 0.2 - model_index * 0.05,
                "spearman": 0.4 - model_index * 0.03,
            }
            for model_index, model in enumerate(models)
            for validation in validations
            for fold in range(3)
        ]
    )
    observed = np.linspace(5.0, 20.0, 24)
    predictions = pd.DataFrame(
        {
            "model": ["ridge"] * len(observed),
            "observed_ido_score": observed,
            "predicted_ido_score": observed * 0.8 + 1.5,
            "b_id": ["B4", "B7", "B8"] * 8,
            "passage": [5, 6, 7, 5] * 6,
            "condition": [f"C{index % 8 + 1:02d}" for index in range(24)],
        }
    )
    importance = pd.DataFrame(
        [
            {
                "model": "ridge",
                "fold": fold,
                "feature": feature,
                "importance": (feature_index + 1) * (1 if fold % 2 == 0 else 0.8),
                "importance_sd": 0.1,
            }
            for fold in range(4)
            for feature_index, feature in enumerate(("area", "perimeter", "solidity"))
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "group_id": f"B{group + 4}-P{group % 3 + 5}",
                "contrast": f"C{contrast + 2:02d}-C01",
                "feature": feature,
                "scaled_delta": np.sin(group + contrast + feature_index),
            }
            for group in range(9)
            for contrast in range(7)
            for feature_index, feature in enumerate(("area", "perimeter", "solidity"))
        ]
    )
    return {
        "status": "no_eligible_phase_only_model",
        "winner": None,
        "ranking": ranking,
        "fold_metrics": fold_metrics,
        "oof_predictions": predictions,
        "feature_importance": importance,
        "morphology_delta_signatures": deltas,
    }


def _create_directory_link(link: Path, target: Path) -> None:
    """建立 directory symlink；Windows 無權限時改用 junction。"""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )


def _png_title(path: Path) -> str:
    """讀取 shipped PNG 的 reader-facing title metadata。"""
    with Image.open(path) as image:
        return str(image.info.get("Title", ""))


def test_result_writer_creates_exact_required_tables_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """避免漏表、額外表或直接覆寫 final CSV。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    import immunity.exp3.reporting as reporting

    real_replace = reporting.os.replace
    replacements: list[tuple[Path, Path]] = []

    def observe_replace(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        replacements.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(reporting.os, "replace", observe_replace)
    write_result_tables(output, _report_tables())

    assert {path.name for path in output.glob("*.csv")} == set(REQUIRED_TABLES)
    assert len(replacements) == len(REQUIRED_TABLES)
    assert all(source.parent == destination.parent for source, destination in replacements)
    assert all(source.suffix == ".tmp" for source, _ in replacements)
    assert not list(output.glob("*.tmp"))


def test_result_writer_preflights_every_destination_before_first_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """後段 table destination 無效時不得新增或覆寫前段 finals。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    first = output / REQUIRED_TABLES[0]
    first.write_text("old-generation\n", encoding="utf-8")
    (output / REQUIRED_TABLES[-1]).mkdir()

    with pytest.raises(ValueError, match="一般檔案"):
        write_result_tables(output, _report_tables())

    assert first.read_text(encoding="utf-8") == "old-generation\n"
    assert {path.name for path in output.iterdir()} == {
        REQUIRED_TABLES[0],
        REQUIRED_TABLES[-1],
    }


def test_result_writer_rolls_back_when_later_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中途 replace 失敗時，所有既有 tables 必須回復舊 generation。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    for name in REQUIRED_TABLES:
        (output / name).write_text(f"old:{name}\n", encoding="utf-8")
    import immunity.exp3.reporting as reporting

    real_replace = reporting.os.replace
    failed = False

    def fail_third_publish_once(source: object, destination: object) -> None:
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path.name == REQUIRED_TABLES[2]
            and source_path.suffix == ".tmp"
            and not failed
        ):
            failed = True
            raise OSError("synthetic publication failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(reporting.os, "replace", fail_third_publish_once)
    with pytest.raises(OSError, match="synthetic publication failure"):
        write_result_tables(output, _report_tables())

    assert failed
    assert all(
        (output / name).read_text(encoding="utf-8") == f"old:{name}\n"
        for name in REQUIRED_TABLES
    )
    assert not list(output.glob("*.tmp"))
    assert not list(output.glob("*.backup"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda tables: tables.pop(REQUIRED_TABLES[0]), "missing"),
        (lambda tables: tables.update({"surprise.csv": pd.DataFrame()}), "unexpected"),
    ],
)
def test_result_writer_rejects_nonexact_table_keys_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    """固定輸出 contract 有差異時不得留下部分結果。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    tables = _report_tables()
    mutation(tables)

    with pytest.raises(ValueError, match=message):
        write_result_tables(output, tables)

    assert not output.exists()


def test_writers_reject_escaped_or_linked_output_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """output root 與 figures child 都不可經 link 逸出。"""
    root, output = _allowed_output(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="Exp3 output"):
        write_result_tables(tmp_path / "legacy", _report_tables())

    root.mkdir(parents=True)
    output.mkdir()
    _create_directory_link(output / "figures", outside)
    with pytest.raises(ValueError, match="symlink|junction"):
        write_figures(output, {})
    assert not list(outside.iterdir())


def test_record_states_no_winner_and_preserves_technical_section_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """no-winner 結論必須保守，且十段技術記錄可依序稽核。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    path = write_experiment_record(output, _no_winner_context())
    text = path.read_text(encoding="utf-8")
    headings = (
        "## 1. 結論狀態",
        "## 2. 資料與 QC",
        "## 3. Primary benchmark",
        "## 4. Eligibility gate",
        "## 5. Dose confounding diagnostics",
        "## 6. Condition-adjusted sensitivity",
        "## 7. ΔMorphology",
        "## 8. 證據、推論與限制",
        "## 9. Reproducibility",
        "## 10. 下一批資料需求",
    )

    assert [text.index(heading) for heading in headings] == sorted(
        text.index(heading) for heading in headings
    )
    assert "沒有符合門檻的 phase-only 模型" in text
    assert "IDO proxy" in text
    assert "整體免疫抑制能力" not in text
    assert "719" in text and "9 個 biological groups" in text
    assert "描述性" in text and "不納入排名" in text
    assert "### 證據" in text and "### 推論" in text and "### 限制" in text
    assert "解讀" in text and "注意" in text
    assert (output / "run_metadata.json").is_file()
    assert not (output / "feature_sets.json").exists()


def test_record_preflights_all_destinations_before_metadata_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record path 無效時不得先覆寫 run metadata。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text('{"generation":"old"}\n', encoding="utf-8")
    (output / "EXPERIMENT_RECORD.md").mkdir()

    with pytest.raises(ValueError, match="一般檔案"):
        write_experiment_record(output, _no_winner_context())

    assert metadata_path.read_text("utf-8") == '{"generation":"old"}\n'


def test_record_validates_existing_feature_registry_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Formal feature registry 只能由 feature aggregation 擁有與寫入。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    registry = _formal_feature_registry()
    registry_text = json.dumps(registry, ensure_ascii=False, separators=(",", ":"))
    registry_path = output / "feature_sets.json"
    registry_path.write_text(registry_text, encoding="utf-8")
    context = _no_winner_context()
    context["feature_sets"] = registry

    write_experiment_record(output, context)

    assert registry_path.read_text("utf-8") == registry_text


def test_record_rejects_feature_registry_mismatch_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Context registry 與 formal registry 不一致時不得發布 record/metadata。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    existing = _formal_feature_registry()
    registry_path = output / "feature_sets.json"
    registry_text = json.dumps(existing, allow_nan=False)
    registry_path.write_text(registry_text, encoding="utf-8")
    context = _no_winner_context()
    context["feature_sets"] = _formal_feature_registry()
    context["feature_sets"]["basic_median"]["predictor_count"] = 34

    with pytest.raises(ValueError, match="feature_sets.json.*mismatch"):
        write_experiment_record(output, context)

    assert registry_path.read_text("utf-8") == registry_text
    assert not (output / "run_metadata.json").exists()
    assert not (output / "EXPERIMENT_RECORD.md").exists()


def test_record_uses_strict_json_and_leaves_no_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非有限 metadata 不得被序列化成非標準 JSON。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    context = _no_winner_context()
    context["metadata"] = {"runtime_seconds": np.nan}

    with pytest.raises(ValueError, match="JSON"):
        write_experiment_record(output, context)

    assert not (output / "run_metadata.json").exists()
    assert not (output / "EXPERIMENT_RECORD.md").exists()


def test_figure_writer_creates_eight_nonempty_pngs_and_closes_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """八種固定圖形在 synthetic data 上皆可產出且不洩漏 figure。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    paths = write_figures(output, _figure_artifacts())

    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.parent == output / "figures" for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)
    assert plt.get_fignums() == []


def test_figure_writer_preflights_every_destination_before_first_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """後段 figure destination 無效時不得新增或覆寫前段 PNG。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    figures = output / "figures"
    figures.mkdir(parents=True)
    first = figures / REQUIRED_FIGURES[0]
    first.write_bytes(b"old-png-generation")
    (figures / REQUIRED_FIGURES[-1]).mkdir()

    with pytest.raises(ValueError, match="一般檔案"):
        write_figures(output, _figure_artifacts())

    assert first.read_bytes() == b"old-png-generation"
    assert {path.name for path in figures.iterdir()} == {
        REQUIRED_FIGURES[0],
        REQUIRED_FIGURES[-1],
    }
    assert plt.get_fignums() == []


def test_malformed_nonfinite_frames_emit_eight_placeholders_without_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inf/非有限資料不得中止 figure bundle 或留下 partial finals。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    artifacts = _figure_artifacts()
    artifacts["morphology_delta_signatures"] = artifacts[
        "morphology_delta_signatures"
    ].assign(scaled_delta=np.inf)
    artifacts["fold_metrics"] = artifacts["fold_metrics"].assign(mae=np.inf)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        paths = write_figures(output, artifacts)

    assert [path.name for path in paths] == list(REQUIRED_FIGURES)
    assert all(path.stat().st_size > 0 for path in paths)
    assert "Insufficient" in _png_title(output / "figures" / "fold_mae_distributions.png")
    assert "Insufficient" in _png_title(output / "figures" / "morphology_delta_pca.png")
    assert plt.get_fignums() == []


def test_no_winner_scatter_is_explicitly_exploratory_and_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """最佳但不合格的候選模型不得在散點圖被誤標為 winner。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    write_figures(output, _figure_artifacts())

    title = _png_title(output / "figures" / "observed_vs_predicted.png")
    assert "Exploratory — ineligible" in title
    assert "winner" not in title.lower()
    assert plt.get_fignums() == []


def test_model_specific_figures_do_not_mix_other_model_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Residual 與 importance 圖不得把其他模型的 rows 混入候選模型。"""
    _, output = _allowed_output(tmp_path, monkeypatch)

    artifacts = _figure_artifacts()
    predictions = artifacts["oof_predictions"]
    other_predictions = predictions.assign(
        model="other_model",
        predicted_ido_score=predictions["predicted_ido_score"] + 100,
    )
    artifacts["oof_predictions"] = pd.concat(
        [predictions, other_predictions], ignore_index=True
    )
    importance = artifacts["feature_importance"]
    artifacts["feature_importance"] = pd.concat(
        [importance, importance.assign(model="other_model", importance=1000.0)],
        ignore_index=True,
    )
    write_figures(output, artifacts)

    importance_title = _png_title(
        output / "figures" / "feature_importance_stability.png"
    )
    residual_title = _png_title(
        output / "figures" / "residuals_by_b_passage_condition.png"
    )
    assert "ridge" in importance_title
    assert "n=12 fold-feature rows" in importance_title
    assert "ridge" in residual_title
    assert "n=24 image rows" in residual_title
    assert plt.get_fignums() == []


def test_rank_heatmap_excludes_diagnostics_using_explicit_role_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank heatmap 的 candidate n 與 rows 不得包含 diagnostic model。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    write_figures(output, _figure_artifacts())

    title = _png_title(output / "figures" / "model_validation_rank_heatmap.png")
    assert "n=1 candidate rows" in title
    assert plt.get_fignums() == []


def test_rank_heatmap_uses_unambiguous_fold_role_for_legacy_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """舊 ranking 無 role 時，以 fold metrics 的唯一 role 證據篩 candidate。"""
    _, output = _allowed_output(tmp_path, monkeypatch)

    artifacts = _figure_artifacts()
    artifacts["ranking"] = artifacts["ranking"].drop(columns="role")
    write_figures(output, artifacts)

    title = _png_title(output / "figures" / "model_validation_rank_heatmap.png")
    assert "n=1 candidate rows" in title
    assert plt.get_fignums() == []


def test_primary_record_and_mae_png_exclude_diagnostics_by_explicit_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary evidence 只能包含 explicit candidate rows。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    artifacts = _figure_artifacts()
    context = _no_winner_context()
    context["fold_metrics"] = artifacts["fold_metrics"]

    record = write_experiment_record(output, context)
    write_figures(output, artifacts)

    text = record.read_text("utf-8")
    primary = text.split("## 3. Primary benchmark", 1)[1].split(
        "## 4. Eligibility gate", 1
    )[0]
    assert "ridge" in primary
    assert "dummy_median" not in primary
    mae_title = _png_title(output / "figures" / "fold_mae_distributions.png")
    assert "n=12 model-fold rows" in mae_title


def test_primary_evidence_fails_closed_without_role_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Role evidence 缺失時 Primary table/MAE plot 不得從 model name 猜測。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    artifacts = _figure_artifacts()
    artifacts["fold_metrics"] = artifacts["fold_metrics"].drop(columns="role")
    context = _no_winner_context()
    context["fold_metrics"] = artifacts["fold_metrics"]

    record = write_experiment_record(output, context)
    write_figures(output, artifacts)

    text = record.read_text("utf-8")
    primary = text.split("## 3. Primary benchmark", 1)[1].split(
        "## 4. Eligibility gate", 1
    )[0]
    assert "No validation metrics reported." in primary
    assert "Insufficient" in _png_title(
        output / "figures" / "fold_mae_distributions.png"
    )


@pytest.mark.parametrize("invalid_selection", ["missing_rank", "ineligible_winner"])
def test_observed_vs_predicted_fails_closed_for_untrusted_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_selection: str,
) -> None:
    """無有限 rank 或不合格 winner 都不得被畫成有效候選模型。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    artifacts = _figure_artifacts()
    if invalid_selection == "missing_rank":
        artifacts["ranking"] = artifacts["ranking"].assign(average_rank=np.nan)
    else:
        artifacts["status"] = "winner_selected"
        artifacts["winner"] = "ridge"
        artifacts["ranking"] = artifacts["ranking"].assign(eligible=False)

    write_figures(output, artifacts)

    title = _png_title(output / "figures" / "observed_vs_predicted.png")
    assert "Insufficient selection evidence" in title
    assert "Eligible model" not in title
    assert plt.get_fignums() == []
