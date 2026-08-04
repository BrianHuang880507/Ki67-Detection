"""Exp3 reporting artifacts 的固定契約與隔離測試。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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
        "feature_sets": {"enabled": ["basic_median"]},
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
    assert json.loads((output / "feature_sets.json").read_text("utf-8")) == {
        "enabled": ["basic_median"]
    }


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


def test_no_winner_scatter_is_explicitly_exploratory_and_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """最佳但不合格的候選模型不得在散點圖被誤標為 winner。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    import immunity.exp3.reporting as reporting

    observed_titles: list[str] = []
    real_savefig = reporting._save_figure_atomically

    def observe_title(figure, path: Path) -> None:
        if path.name == "observed_vs_predicted.png":
            observed_titles.append(figure.axes[0].get_title(loc="left"))
        real_savefig(figure, path)

    monkeypatch.setattr(reporting, "_save_figure_atomically", observe_title)
    write_figures(output, _figure_artifacts())

    assert len(observed_titles) == 1
    assert "Exploratory — ineligible" in observed_titles[0]
    assert "winner" not in observed_titles[0].lower()
    assert plt.get_fignums() == []


def test_model_specific_figures_do_not_mix_other_model_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Residual 與 importance 圖不得把其他模型的 rows 混入候選模型。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    import immunity.exp3.reporting as reporting

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
    observed_titles: dict[str, str] = {}
    real_savefig = reporting._save_figure_atomically

    def observe_titles(figure, path: Path) -> None:
        if path.name == "feature_importance_stability.png":
            observed_titles["importance"] = figure.axes[0].get_title(loc="left")
        if path.name == "residuals_by_b_passage_condition.png":
            observed_titles["residual"] = figure._suptitle.get_text()
        real_savefig(figure, path)

    monkeypatch.setattr(reporting, "_save_figure_atomically", observe_titles)
    write_figures(output, artifacts)

    assert "ridge" in observed_titles["importance"]
    assert "n=12 fold-feature rows" in observed_titles["importance"]
    assert "ridge" in observed_titles["residual"]
    assert "n=24 image rows" in observed_titles["residual"]
    assert plt.get_fignums() == []


def test_rank_heatmap_excludes_diagnostics_using_explicit_role_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank heatmap 的 candidate n 與 rows 不得包含 diagnostic model。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    import immunity.exp3.reporting as reporting

    observed: dict[str, object] = {}
    real_savefig = reporting._save_figure_atomically

    def observe_rank_heatmap(figure, path: Path) -> None:
        if path.name == "model_validation_rank_heatmap.png":
            axis = figure.axes[0]
            observed["title"] = axis.get_title(loc="left")
            observed["rows"] = [label.get_text() for label in axis.get_yticklabels()]
        real_savefig(figure, path)

    monkeypatch.setattr(reporting, "_save_figure_atomically", observe_rank_heatmap)
    write_figures(output, _figure_artifacts())

    assert "n=1 candidate rows" in str(observed["title"])
    assert observed["rows"] == ["ridge"]
    assert plt.get_fignums() == []


def test_rank_heatmap_uses_unambiguous_fold_role_for_legacy_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """舊 ranking 無 role 時，以 fold metrics 的唯一 role 證據篩 candidate。"""
    _, output = _allowed_output(tmp_path, monkeypatch)
    import immunity.exp3.reporting as reporting

    artifacts = _figure_artifacts()
    artifacts["ranking"] = artifacts["ranking"].drop(columns="role")
    observed_rows: list[str] = []
    real_savefig = reporting._save_figure_atomically

    def observe_rank_heatmap(figure, path: Path) -> None:
        if path.name == "model_validation_rank_heatmap.png":
            observed_rows.extend(
                label.get_text() for label in figure.axes[0].get_yticklabels()
            )
        real_savefig(figure, path)

    monkeypatch.setattr(reporting, "_save_figure_atomically", observe_rank_heatmap)
    write_figures(output, artifacts)

    assert observed_rows == ["ridge"]
    assert plt.get_fignums() == []
