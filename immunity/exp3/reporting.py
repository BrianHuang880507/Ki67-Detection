"""Exp3 固定結果表、靜態圖表與技術實驗記錄。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from immunity.exp3.run_benchmark import resolve_exp3_output_dir


REQUIRED_TABLES = (
    "data_manifest.csv",
    "pairing_qc.csv",
    "segmentation_qc.csv",
    "outer_splits.csv",
    "oof_predictions.csv",
    "fold_metrics.csv",
    "hyperparameters.csv",
    "model_ranking.csv",
    "feature_importance.csv",
    "model_failures.csv",
    "condition_adjusted_metrics.csv",
    "pc_nucleus_dapi_validation.csv",
    "morphology_delta_signatures.csv",
)

REQUIRED_FIGURES = (
    "model_validation_rank_heatmap.png",
    "fold_mae_distributions.png",
    "observed_vs_predicted.png",
    "diagnostic_model_comparison.png",
    "feature_importance_stability.png",
    "residuals_by_b_passage_condition.png",
    "morphology_delta_heatmap.png",
    "morphology_delta_pca.png",
)

_VALIDATIONS = (
    "leave_one_b_out",
    "leave_one_passage_out",
    "leave_one_group_out",
    "leave_one_condition_out",
)
_BLUE = "#356E9F"
_BLUE_LIGHT = "#DCE9F3"
_ORANGE = "#D9822B"
_ORANGE_LIGHT = "#F6E4D1"
_INK = "#24313A"
_NEUTRAL = "#68747D"
_GRID = "#D8DEE3"
_BACKGROUND = "#FCFCFB"
_DIVERGING = LinearSegmentedColormap.from_list(
    "exp3_blue_orange", (_BLUE, "#F5F3EE", _ORANGE)
)
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "exp3_rank_blue", (_BLUE, _BLUE_LIGHT)
)


@dataclass(frozen=True)
class _DirectoryIdentity:
    """發布期間用來偵測 parent directory 被置換的 snapshot。"""

    path: Path
    resolved: Path
    device: int
    inode: int


@dataclass
class _StagedArtifact:
    """尚未發布的同層暫存檔與 final destination。"""

    temporary: Path
    destination: Path


def write_result_tables(
    output_dir: Path, tables: Mapping[str, pd.DataFrame]
) -> None:
    """原子寫出固定的 13 個 Exp3 CSV 結果表。

    Args:
        output_dir: `immunity/outputs/exp3` 下的單次實驗目錄。
        tables: 以完整 CSV 檔名為 key 的 DataFrame mapping；key 必須與
            `REQUIRED_TABLES` 完全相同。

    Raises:
        TypeError: `tables` 不是 mapping，或值不是 DataFrame。
        ValueError: 缺少／多出固定表格，或輸出位置不安全。
    """
    if not isinstance(tables, Mapping):
        raise TypeError("tables 必須是 filename-to-DataFrame mapping")
    actual = set(tables)
    required = set(REQUIRED_TABLES)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError("結果表 key 必須完全符合固定 contract：" + "; ".join(details))
    invalid = sorted(name for name, frame in tables.items() if not isinstance(frame, pd.DataFrame))
    if invalid:
        raise TypeError(f"結果表必須是 pandas DataFrame：{invalid}")

    safe_output = _prepare_output_dir(output_dir)
    identity = _snapshot_directory(safe_output)
    destinations = _preflight_files(safe_output, REQUIRED_TABLES, identity)
    staged: list[_StagedArtifact] = []
    try:
        for name, destination in zip(REQUIRED_TABLES, destinations, strict=True):
            staged.append(
                _StagedArtifact(_stage_csv(tables[name], destination), destination)
            )
        _publish_staged(staged, identity)
    finally:
        _cleanup_staged(staged)


def write_figures(output_dir: Path, artifacts: Mapping[str, Any]) -> list[Path]:
    """產生八張固定名稱的 Exp3 靜態 PNG 圖。

    缺少或沒有足夠資料時，仍會輸出清楚標示 insufficient data 的圖，避免
    pipeline 因純報表資料不足而失去其他可稽核產物。

    Args:
        output_dir: `immunity/outputs/exp3` 下的單次實驗目錄。
        artifacts: reporting tables 與 run status；支援 `ranking`／
            `model_ranking`、`fold_metrics`、`oof_predictions`／`predictions`、
            `feature_importance` 與 `morphology_delta_signatures`。

    Returns:
        依 `REQUIRED_FIGURES` 排序的八個 PNG 路徑。

    Raises:
        TypeError: `artifacts` 不是 mapping。
        ValueError: 輸出位置或 figures 目錄不安全。
    """
    if not isinstance(artifacts, Mapping):
        raise TypeError("artifacts 必須是 mapping")
    safe_output = _prepare_output_dir(output_dir)
    figures_dir = _prepare_child_directory(safe_output, "figures")
    identity = _snapshot_directory(figures_dir)
    destinations = _preflight_files(figures_dir, REQUIRED_FIGURES, identity)
    builders = (
        _rank_heatmap,
        _mae_distributions,
        _observed_vs_predicted,
        _diagnostic_comparison,
        _importance_stability,
        _residual_comparison,
        _delta_heatmap,
        _delta_pca,
    )
    staged: list[_StagedArtifact] = []
    try:
        for name, builder, destination in zip(
            REQUIRED_FIGURES, builders, destinations, strict=True
        ):
            figure = _safe_build_figure(builder, artifacts, name)
            try:
                staged.append(
                    _StagedArtifact(_stage_figure(figure, destination), destination)
                )
            finally:
                plt.close(figure)
        _publish_staged(staged, identity)
    finally:
        _cleanup_staged(staged)
    return destinations


def write_experiment_record(output_dir: Path, context: Mapping[str, Any]) -> Path:
    """建立 answer-first、可稽核的 Exp3 技術實驗記錄。

    記錄固定保留 brief 指定的十段順序，並將描述性、診斷性與 predictive
    證據分開。`metadata` 會同步寫成 strict `run_metadata.json`；formal
    `feature_sets.json` 只由 feature aggregation 擁有，本函式最多驗證既有內容。

    Args:
        output_dir: `immunity/outputs/exp3` 下的單次實驗目錄。
        context: 結論狀態、QC、benchmark、diagnostics、限制與 reproducibility
            metadata。

    Returns:
        `EXPERIMENT_RECORD.md` 的完整路徑。

    Raises:
        TypeError: `context` 或 JSON payload 不是 mapping。
        ValueError: JSON 含非有限值，或輸出位置不安全。
    """
    if not isinstance(context, Mapping):
        raise TypeError("context 必須是 mapping")
    metadata = context.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("context['metadata'] 必須是 mapping")
    feature_sets = context.get("feature_sets")
    if feature_sets is not None and not isinstance(feature_sets, Mapping):
        raise TypeError("context['feature_sets'] 必須是 mapping")

    #先序列化可確保 NaN/Infinity 在任何 final file 建立前就被拒絕。
    metadata_text = _strict_json_text(metadata, "run_metadata.json")
    record_text = _build_experiment_record(context)

    safe_output = _prepare_output_dir(output_dir)
    identity = _snapshot_directory(safe_output)
    if feature_sets is not None:
        _validate_existing_feature_registry(safe_output, feature_sets, identity)
    names = ("run_metadata.json", "EXPERIMENT_RECORD.md")
    destinations = _preflight_files(safe_output, names, identity)
    staged: list[_StagedArtifact] = []
    try:
        for text, destination in zip(
            (metadata_text, record_text), destinations, strict=True
        ):
            staged.append(
                _StagedArtifact(_stage_text(text, destination), destination)
            )
        _publish_staged(staged, identity)
    finally:
        _cleanup_staged(staged)
    return destinations[1]


def _prepare_output_dir(output_dir: Path) -> Path:
    """驗證並建立 dedicated Exp3 run 目錄。"""
    safe_output = resolve_exp3_output_dir(output_dir)
    safe_output.mkdir(parents=True, exist_ok=True)
    if not safe_output.is_dir() or safe_output.resolve(strict=False) != safe_output:
        raise ValueError("Exp3 output 不可透過 symlink 或 junction 逸出")
    return safe_output


def _prepare_child_directory(parent: Path, name: str) -> Path:
    """建立並再次驗證單層 output child directory。"""
    directory = _safe_child(parent, name, kind="directory")
    directory.mkdir(exist_ok=True)
    if not directory.is_dir() or directory.resolve(strict=False) != directory:
        raise ValueError(f"{name} 不可透過 symlink 或 junction 逸出")
    return directory


def _snapshot_directory(path: Path) -> _DirectoryIdentity:
    """記錄 parent identity；用於合理偵測 Windows path replacement。"""
    resolved = path.resolve(strict=True)
    stat = path.stat()
    return _DirectoryIdentity(path, resolved, int(stat.st_dev), int(stat.st_ino))


def _assert_directory_identity(identity: _DirectoryIdentity) -> None:
    """確認 parent 自 preflight 後未被 symlink/junction 或新目錄置換。"""
    try:
        resolved = identity.path.resolve(strict=True)
        stat = identity.path.stat()
    except OSError as error:
        raise ValueError("輸出 parent 在發布期間消失或被置換") from error
    if (
        resolved != identity.resolved
        or int(stat.st_dev) != identity.device
        or int(stat.st_ino) != identity.inode
    ):
        raise ValueError("輸出 parent 在發布期間被置換")


def _preflight_files(
    parent: Path, names: Sequence[str], identity: _DirectoryIdentity
) -> list[Path]:
    """在 first publish 前驗證完整 final destination set。"""
    _assert_directory_identity(identity)
    destinations = [_safe_child(parent, name, kind="file") for name in names]
    _assert_directory_identity(identity)
    return destinations


def _safe_child(parent: Path, name: str, *, kind: str) -> Path:
    """拒絕 child path 經 symlink 或 junction 改寫目的地。"""
    expected = parent / name
    if expected.resolve(strict=False) != expected:
        raise ValueError(f"{kind} {name!r} 不可透過 symlink 或 junction 逸出")
    if expected.exists():
        if kind == "file" and not expected.is_file():
            raise ValueError(f"{name!r} 必須是一般檔案")
        if kind == "directory" and not expected.is_dir():
            raise ValueError(f"{name!r} 必須是目錄")
    return expected


def _stage_csv(frame: pd.DataFrame, destination: Path) -> Path:
    """在 final 同層完成 CSV 暫存，但不發布。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
        return temporary_path
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _stage_text(text: str, destination: Path) -> Path:
    """在 final 同層完成 UTF-8 暫存，但不發布。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        return temporary_path
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _stage_figure(figure: Figure, destination: Path) -> Path:
    """將 PNG 完整寫入同層暫存檔，但不發布。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        figure.savefig(
            temporary_path,
            format="png",
            dpi=160,
            facecolor=_BACKGROUND,
            bbox_inches="tight",
            metadata={
                "Software": "Ki67-Detection Exp3 reporting",
                "Title": _figure_title(figure),
            },
        )
        if temporary_path.stat().st_size == 0:
            raise OSError(f"figure {destination.name!r} 暫存檔為空")
        return temporary_path
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _publish_staged(
    staged: Sequence[_StagedArtifact], identity: _DirectoryIdentity
) -> None:
    """以完整 preflight、backup 與 rollback 發布同一 artifact bundle。"""
    if not staged:
        return
    names = [item.destination.name for item in staged]
    _preflight_files(identity.path, names, identity)
    backups: dict[Path, Path] = {}
    published: list[_StagedArtifact] = []
    try:
        for item in staged:
            if item.destination.exists():
                backup = _reserve_sibling(item.destination, ".backup")
                shutil.copy2(item.destination, backup)
                backups[item.destination] = backup
        _preflight_files(identity.path, names, identity)
        for item in staged:
            _assert_directory_identity(identity)
            _safe_child(identity.path, item.destination.name, kind="file")
            os.replace(item.temporary, item.destination)
            published.append(item)
        _assert_directory_identity(identity)
    except Exception:
        rollback_errors: list[Exception] = []
        for item in reversed(published):
            try:
                backup = backups.pop(item.destination, None)
                if backup is not None and backup.exists():
                    os.replace(backup, item.destination)
                elif item.destination.exists() and item.destination.is_file():
                    item.destination.unlink()
            except Exception as rollback_error:  # pragma: no cover - catastrophic OS failure
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError("artifact publication rollback failed") from rollback_errors[0]
        raise
    finally:
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def _reserve_sibling(destination: Path, suffix: str) -> Path:
    """保留同層唯一暫存路徑，讓 backup/stage 不跨 filesystem。"""
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=suffix,
        dir=destination.parent,
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def _cleanup_staged(staged: Sequence[_StagedArtifact]) -> None:
    """清除尚未被 replace 消耗的 staged files。"""
    for item in staged:
        if item.temporary.exists():
            item.temporary.unlink()


def _validate_existing_feature_registry(
    output_dir: Path,
    expected: Mapping[str, Any],
    identity: _DirectoryIdentity,
) -> None:
    """只驗證 feature aggregation 擁有的 formal registry，絕不寫入。"""
    _assert_directory_identity(identity)
    path = _safe_child(output_dir, "feature_sets.json", kind="file")
    if not path.is_file():
        raise ValueError("feature_sets.json 不存在；reporting 不可建立 formal registry")
    try:
        actual = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"feature_sets.json 不是 strict JSON registry：{error}") from error
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ValueError("feature_sets.json registry mismatch")
    _assert_directory_identity(identity)


def _figure_title(figure: Figure) -> str:
    """產生 PNG metadata title，包含 visible insufficient-data cue。"""
    parts: list[str] = []
    if figure._suptitle is not None:
        parts.append(figure._suptitle.get_text())
    for axis in figure.axes:
        title = axis.get_title(loc="left") or axis.get_title()
        if title and title not in parts:
            parts.append(title)
        parts.extend(
            text.get_text()
            for text in axis.texts
            if "Insufficient" in text.get_text()
        )
    return "\n".join(parts) or "Exp3 figure"


def _strict_json_text(payload: Mapping[str, Any], name: str) -> str:
    """產生拒絕 NaN/Infinity 的標準 JSON 文字。"""
    try:
        return json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是 strict JSON：{error}") from error


def _frame(artifacts: Mapping[str, Any], *names: str) -> pd.DataFrame:
    """從支援的 alias 取得 DataFrame；不合法資料視為 unavailable。"""
    for name in names:
        value = artifacts.get(name)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    return pd.DataFrame()


def _new_figure(*, width: float = 9.0, height: float = 5.5) -> tuple[Figure, Any]:
    """建立符合 Exp3 靜態圖表色彩與背景規範的單一 axes。"""
    figure, axis = plt.subplots(figsize=(width, height), facecolor=_BACKGROUND)
    axis.set_facecolor(_BACKGROUND)
    return figure, axis


def _style_axis(axis: Any, *, grid_axis: str | None = None) -> None:
    """套用 quiet grid、深色文字與可見軸錨點。"""
    axis.tick_params(colors=_INK, labelsize=8)
    for side, spine in axis.spines.items():
        spine.set_color(_NEUTRAL if side in {"left", "bottom"} else _GRID)
        spine.set_linewidth(0.8)
    if grid_axis:
        axis.grid(axis=grid_axis, color=_GRID, linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
    axis.xaxis.label.set_color(_INK)
    axis.yaxis.label.set_color(_INK)
    axis.title.set_color(_INK)


def _placeholder(axis: Any, title: str, message: str) -> None:
    """以明確文字呈現不足資料，而非繪製誤導性 marks。"""
    axis.set_title(title, loc="left", fontsize=12, fontweight="semibold")
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=_NEUTRAL,
        fontsize=10,
        wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    _style_axis(axis)


def _safe_build_figure(
    builder: Any, artifacts: Mapping[str, Any], filename: str
) -> Figure:
    """將 malformed inputs 降級成 placeholder，並關閉 builder 洩漏的 figures。"""
    before = set(plt.get_fignums())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                figure = builder(artifacts)
        if not isinstance(figure, Figure):
            raise TypeError("figure builder 必須回傳 matplotlib Figure")
    except Exception:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        figure, axis = _new_figure()
        _placeholder(
            axis,
            _figure_label(filename),
            "Insufficient data — malformed or non-finite inputs.",
        )
        return figure
    for number in set(plt.get_fignums()) - before - {figure.number}:
        plt.close(number)
    return figure


def _figure_label(filename: str) -> str:
    """提供 malformed-data placeholder 的固定 neutral title。"""
    return {
        "model_validation_rank_heatmap.png": "Model rank by validation family",
        "fold_mae_distributions.png": "Outer-fold MAE distributions",
        "observed_vs_predicted.png": "Observed vs predicted IDO proxy",
        "diagnostic_model_comparison.png": "Diagnostic model MAE comparison",
        "feature_importance_stability.png": "Feature importance stability",
        "residuals_by_b_passage_condition.png": "OOF residuals by B-ID, passage, and condition",
        "morphology_delta_heatmap.png": "ΔMorphology signature heatmap",
        "morphology_delta_pca.png": "ΔMorphology PCA",
    }[filename]


def _rank_heatmap(artifacts: Mapping[str, Any]) -> Figure:
    """繪製 model × validation rank matrix。"""
    ranking = _candidate_ranking(artifacts)
    figure, axis = _new_figure(width=10, height=5.8)
    rank_columns = [f"{validation}_rank" for validation in _VALIDATIONS]
    if ranking.empty or "model" not in ranking or not set(rank_columns).issubset(ranking):
        _placeholder(
            axis,
            "Model rank by validation family",
            "Insufficient data — validation-level ranks are unavailable.",
        )
        return figure
    matrix = ranking[rank_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if matrix.size == 0 or not np.isfinite(matrix).any():
        _placeholder(axis, "Model rank by validation family", "Insufficient finite rank data.")
        return figure
    display = axis.imshow(matrix, cmap=_SEQUENTIAL, aspect="auto")
    axis.set_xticks(range(len(rank_columns)), [_human_validation(v) for v in _VALIDATIONS])
    axis.set_yticks(range(len(ranking)), ranking["model"].astype(str))
    axis.set_title(
        f"Model rank by validation family\nLower rank is better; n={len(ranking)} candidate rows",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                "NA" if not np.isfinite(value) else f"{value:.1f}",
                ha="center",
                va="center",
                color=(
                    "#FFFFFF"
                    if np.isfinite(value)
                    and value <= float(np.nanmin(matrix)) + 0.35 * float(np.nanmax(matrix) - np.nanmin(matrix))
                    else _INK
                ),
                fontsize=8,
            )
    colorbar = figure.colorbar(display, ax=axis, pad=0.02)
    colorbar.set_label("Rank", color=_INK)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def _mae_distributions(artifacts: Mapping[str, Any]) -> Figure:
    """繪製各模型 outer-fold MAE 分布。"""
    metrics = _explicit_role_rows(_frame(artifacts, "fold_metrics"), "candidate")
    figure, axis = _new_figure(width=10, height=6)
    if metrics.empty or not {"model", "mae"}.issubset(metrics):
        _placeholder(axis, "Outer-fold MAE distributions", "Insufficient fold-level MAE data.")
        return figure
    clean = metrics.assign(mae=_finite_numeric(metrics["mae"])).dropna(
        subset=["model", "mae"]
    )
    groups = [group for _, group in clean.groupby("model", sort=True)]
    labels = [str(name) for name in sorted(clean["model"].astype(str).unique())]
    if not groups:
        _placeholder(
            axis,
            "Outer-fold MAE distributions",
            "Insufficient data — no finite fold-level MAE values.",
        )
        return figure
    box = axis.boxplot(
        [group["mae"].to_numpy(float) for group in groups],
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": _ORANGE, "markeredgecolor": _INK},
        medianprops={"color": _INK, "linewidth": 1.5},
    )
    for patch in box["boxes"]:
        patch.set(facecolor=_BLUE_LIGHT, edgecolor=_BLUE, hatch="//")
    axis.set_title(
        f"Outer-fold MAE distributions\nIDO proxy units; n={len(clean)} model-fold rows",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    axis.set_ylabel("MAE (IDO proxy units)")
    axis.tick_params(axis="x", rotation=30)
    _style_axis(axis, grid_axis="y")
    figure.tight_layout()
    return figure


def _observed_vs_predicted(artifacts: Mapping[str, Any]) -> Figure:
    """繪製 eligible winner 或明確標示 ineligible candidate 的 OOF scatter。"""
    predictions = _frame(artifacts, "oof_predictions", "predictions")
    ranking = _candidate_ranking(artifacts)
    winner = artifacts.get("winner")
    no_winner = winner in {None, ""} or artifacts.get("status") == "no_eligible_phase_only_model"
    selected = (
        _best_ineligible_model(ranking)
        if no_winner
        else _validated_winner(artifacts)
    )
    label = (
        "Exploratory — ineligible"
        if no_winner and selected is not None
        else "Eligible model"
        if selected is not None
        else "Insufficient selection evidence"
    )
    title = f"Observed vs predicted IDO proxy — {selected or 'model unavailable'}\n{label}"
    figure, axis = _new_figure(width=7.2, height=6.5)
    required = {"model", "observed_ido_score", "predicted_ido_score"}
    if predictions.empty or not required.issubset(predictions) or not selected:
        _placeholder(axis, title, "Insufficient OOF prediction data.")
        return figure
    selected_rows = predictions[predictions["model"].astype(str).eq(selected)].copy()
    selected_rows["observed_ido_score"] = _finite_numeric(
        selected_rows["observed_ido_score"]
    )
    selected_rows["predicted_ido_score"] = _finite_numeric(
        selected_rows["predicted_ido_score"]
    )
    selected_rows = selected_rows.dropna(
        subset=["observed_ido_score", "predicted_ido_score"]
    )
    if len(selected_rows) < 12:
        _placeholder(
            axis,
            title,
            f"Insufficient data for an honest scatter (n={len(selected_rows)}; need ≥12 points).",
        )
        return figure
    observed = selected_rows["observed_ido_score"].to_numpy(float)
    predicted = selected_rows["predicted_ido_score"].to_numpy(float)
    lower = float(min(observed.min(), predicted.min()))
    upper = float(max(observed.max(), predicted.max()))
    axis.scatter(
        observed,
        predicted,
        s=38,
        facecolors=_BLUE_LIGHT,
        edgecolors=_BLUE,
        linewidths=1.0,
        marker="o",
        label=selected,
    )
    axis.plot([lower, upper], [lower, upper], linestyle="--", color=_NEUTRAL, label="Identity")
    axis.set_title(
        f"{title}; n={len(selected_rows)} OOF image rows",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    axis.set_xlabel("Observed IDO proxy (image-level background-corrected intensity)")
    axis.set_ylabel("Predicted IDO proxy (same units)")
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return figure


def _diagnostic_comparison(artifacts: Mapping[str, Any]) -> Figure:
    """比較 Dummy、dose-only 與 dose-plus-morphology diagnostic MAE。"""
    metrics = _frame(artifacts, "fold_metrics", "diagnostic_metrics")
    figure, axis = _new_figure(width=8.5, height=5.5)
    expected = ("dummy_median", "dose_ridge", "dose_plus_morphology_ridge")
    if metrics.empty or not {"model", "mae"}.issubset(metrics):
        _placeholder(axis, "Diagnostic model MAE comparison", "Insufficient diagnostic MAE data.")
        return figure
    clean = _explicit_role_rows(metrics, "diagnostic")
    clean = clean[clean["model"].astype(str).isin(expected)].copy()
    clean["mae"] = _finite_numeric(clean["mae"])
    summary = clean.dropna(subset=["mae"]).groupby("model")["mae"].median().reindex(expected)
    available = summary.dropna()
    if available.empty:
        _placeholder(axis, "Diagnostic model MAE comparison", "No finite diagnostic MAE values.")
        return figure
    style = {
        "dummy_median": (_NEUTRAL, _INK, ".."),
        "dose_ridge": (_ORANGE_LIGHT, _ORANGE, "//"),
        "dose_plus_morphology_ridge": (_BLUE_LIGHT, _BLUE, "xx"),
    }
    bars = axis.bar(
        range(len(available)),
        available.to_numpy(),
        color=[style[model][0] for model in available.index],
    )
    for bar, model in zip(bars, available.index, strict=True):
        _, edge, hatch = style[model]
        bar.set_edgecolor(edge)
        bar.set_hatch(hatch)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3g}",
            ha="center",
            va="bottom",
            color=_INK,
            fontsize=8,
        )
    axis.set_xticks(range(len(available)), available.index, rotation=20, ha="right")
    axis.set_ylim(bottom=0)
    axis.set_ylabel("Median outer-fold MAE (IDO proxy units)")
    axis.set_title(
        f"Diagnostic model MAE comparison\nDummy, dose-only, and dose-plus-morphology; n={len(clean)} fold rows",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axis(axis, grid_axis="y")
    figure.tight_layout()
    return figure


def _importance_stability(artifacts: Mapping[str, Any]) -> Figure:
    """繪製 fold 間 feature importance 的平均與範圍。"""
    importance = _frame(artifacts, "feature_importance")
    figure, axis = _new_figure(width=9, height=6)
    if importance.empty or not {"feature", "importance"}.issubset(importance):
        _placeholder(axis, "Feature importance stability", "Insufficient fold-level importance data.")
        return figure
    selected = _selected_model(artifacts, importance)
    clean = importance.copy()
    if selected is not None and "model" in clean:
        clean = clean[clean["model"].astype(str).eq(selected)]
    clean["importance"] = _finite_numeric(clean["importance"])
    clean = clean.dropna(subset=["feature", "importance"])
    if clean.empty:
        _placeholder(axis, "Feature importance stability", "No finite feature importance values.")
        return figure
    summary = clean.groupby("feature")["importance"].agg(["mean", "std", "count"])
    summary["abs_mean"] = summary["mean"].abs()
    summary = summary.nlargest(12, "abs_mean").sort_values("mean")
    error = summary["std"].fillna(0.0).to_numpy(float)
    y = np.arange(len(summary))
    axis.errorbar(
        summary["mean"],
        y,
        xerr=error,
        fmt="o",
        color=_BLUE,
        markerfacecolor=_BLUE_LIGHT,
        markeredgecolor=_BLUE,
        ecolor=_NEUTRAL,
        capsize=3,
    )
    axis.axvline(0, color=_INK, linewidth=0.8, linestyle="--")
    axis.set_yticks(y, summary.index.astype(str))
    axis.set_xlabel("Mean importance across folds (±1 SD; model-specific units)")
    axis.set_title(
        f"Feature importance stability — {selected or 'model unavailable'}\n"
        f"Top {len(summary)} features; n={len(clean)} fold-feature rows",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axis(axis, grid_axis="x")
    figure.tight_layout()
    return figure


def _residual_comparison(artifacts: Mapping[str, Any]) -> Figure:
    """依 B-ID、passage 與 condition 顯示 OOF residual 分布。"""
    predictions = _frame(artifacts, "oof_predictions", "predictions")
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), facecolor=_BACKGROUND)
    for axis in axes:
        axis.set_facecolor(_BACKGROUND)
    required = {"observed_ido_score", "predicted_ido_score"}
    if predictions.empty or not required.issubset(predictions):
        for axis in axes:
            axis.set_visible(False)
        axes[0].set_visible(True)
        _placeholder(
            axes[0],
            "OOF residuals by B-ID, passage, and condition",
            "Insufficient OOF residual data.",
        )
        return figure
    selected = _selected_model(artifacts, predictions)
    clean = predictions.copy()
    if selected is not None and "model" in clean:
        clean = clean[clean["model"].astype(str).eq(selected)]
    if "residual" not in clean:
        clean["residual"] = _finite_numeric(clean["observed_ido_score"]) - _finite_numeric(
            clean["predicted_ido_score"]
        )
    else:
        clean["residual"] = _finite_numeric(clean["residual"])
    clean = clean.dropna(subset=["residual"])
    dimensions = (("b_id", "B-ID"), ("passage", "Passage"), ("condition", "Condition"))
    plotted = False
    for axis, (column, label) in zip(axes, dimensions, strict=True):
        if column not in clean or clean[column].dropna().empty:
            _placeholder(axis, label, "Unavailable")
            continue
        grouped = [(str(name), group["residual"].to_numpy(float)) for name, group in clean.groupby(column)]
        boxes = axis.boxplot(
            [values for _, values in grouped],
            tick_labels=[name for name, _ in grouped],
            patch_artist=True,
            medianprops={"color": _ORANGE, "linewidth": 1.4},
        )
        for patch in boxes["boxes"]:
            patch.set(facecolor=_BLUE_LIGHT, edgecolor=_BLUE, hatch="//")
        axis.axhline(0, color=_INK, linestyle="--", linewidth=0.8)
        axis.set_title(label, loc="left", fontsize=10, fontweight="semibold")
        axis.tick_params(axis="x", rotation=45)
        _style_axis(axis, grid_axis="y")
        plotted = True
    if not plotted:
        _placeholder(axes[0], "OOF residuals by B-ID, passage, and condition", "No grouping columns available.")
    figure.suptitle(
        f"OOF residuals by B-ID, passage, and condition — {selected or 'model unavailable'}\n"
        f"Observed minus predicted IDO proxy; n={len(clean)} image rows",
        x=0.04,
        ha="left",
        color=_INK,
        fontsize=12,
        fontweight="semibold",
    )
    axes[0].set_ylabel("Residual (IDO proxy units)")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure


def _delta_heatmap(artifacts: Mapping[str, Any]) -> Figure:
    """繪製 descriptive ΔMorphology signature matrix。"""
    deltas = _frame(artifacts, "morphology_delta_signatures", "morphology_deltas")
    figure, axis = _new_figure(width=10, height=10)
    required = {
        "group_id",
        "contrast_id",
        "feature",
        "delta_scaled_by_global_iqr",
    }
    if deltas.empty or not required.issubset(deltas):
        _placeholder(axis, "ΔMorphology signature heatmap", "Insufficient descriptive signature data.")
        return figure
    clean = deltas.copy()
    clean["delta_scaled_by_global_iqr"] = _finite_numeric(
        clean["delta_scaled_by_global_iqr"]
    )
    clean["row_label"] = (
        clean["group_id"].astype(str) + " | " + clean["contrast_id"].astype(str)
    )
    pivot = clean.pivot_table(
        index="row_label",
        columns="feature",
        values="delta_scaled_by_global_iqr",
        aggfunc="mean",
    )
    if pivot.empty or not np.isfinite(pivot.to_numpy(float)).any():
        _placeholder(axis, "ΔMorphology signature heatmap", "No finite descriptive deltas.")
        return figure
    matrix = pivot.to_numpy(float)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = float(finite.max()) if finite.size else 1.0
    limit = max(limit, 1e-12)
    display = axis.imshow(matrix, cmap=_DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index.astype(str), fontsize=6)
    if matrix.size <= 240:
        for row, column in np.ndindex(matrix.shape):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                "NA" if not np.isfinite(value) else f"{value:.1f}",
                ha="center",
                va="center",
                color=("#FFFFFF" if np.isfinite(value) and abs(value) >= 0.6 * limit else _INK),
                fontsize=5,
            )
    axis.set_title(
        f"ΔMorphology signature heatmap\nDescriptive scaled deltas; {pivot.index.nunique()} group-contrast rows",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    colorbar = figure.colorbar(display, ax=axis, pad=0.02)
    colorbar.set_label("Scaled ΔMorphology (descriptive)", color=_INK)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def _delta_pca(artifacts: Mapping[str, Any]) -> Figure:
    """以 SVD 建立九個 biological-group signature 的 descriptive PCA。"""
    deltas = _frame(artifacts, "morphology_delta_signatures", "morphology_deltas")
    figure, axis = _new_figure(width=8, height=6.5)
    required = {
        "group_id",
        "contrast_id",
        "feature",
        "delta_scaled_by_global_iqr",
    }
    if deltas.empty or not required.issubset(deltas):
        _placeholder(axis, "ΔMorphology PCA", "Insufficient descriptive signature data.")
        return figure
    clean = deltas.copy()
    clean["delta_scaled_by_global_iqr"] = _finite_numeric(
        clean["delta_scaled_by_global_iqr"]
    )
    clean["dimension"] = (
        clean["contrast_id"].astype(str) + " | " + clean["feature"].astype(str)
    )
    matrix = clean.pivot_table(
        index="group_id",
        columns="dimension",
        values="delta_scaled_by_global_iqr",
        aggfunc="mean",
    )
    matrix = matrix.dropna(axis=1, how="all").fillna(0.0)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        _placeholder(
            axis,
            "ΔMorphology PCA",
            f"Insufficient data for PCA (n={matrix.shape[0]} biological groups).",
        )
        return figure
    values = matrix.to_numpy(float)
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ right[:2].T
    denominator = float(np.square(singular_values).sum())
    explained = np.square(singular_values[:2]) / denominator if denominator > 0 else np.zeros(2)
    axis.scatter(
        scores[:, 0],
        scores[:, 1],
        s=52,
        marker="o",
        facecolors=_ORANGE_LIGHT,
        edgecolors=_ORANGE,
        linewidths=1.2,
    )
    for group_id, x, y in zip(matrix.index.astype(str), scores[:, 0], scores[:, 1], strict=True):
        axis.annotate(group_id, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=_INK)
    axis.axhline(0, color=_GRID, linewidth=0.8)
    axis.axvline(0, color=_GRID, linewidth=0.8)
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% descriptive variance)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% descriptive variance)")
    axis.set_title(
        f"ΔMorphology PCA\nDescriptive signatures; n={matrix.shape[0]} biological groups (not independent images)",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return figure


def _first_present(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    """回傳第一個存在的欄名。"""
    return next((name for name in names if name in frame), None)


def _finite_numeric(values: pd.Series) -> pd.Series:
    """將 numeric-like values 正規化，並把 ±Inf 視為 unavailable。"""
    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _explicit_role_rows(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """只保留明確 role rows；缺少 role evidence 時 fail closed。"""
    if frame.empty or "role" not in frame:
        return pd.DataFrame(columns=frame.columns)
    roles = frame["role"].astype("string").str.strip().str.lower()
    return frame[roles.eq(role).fillna(False)].copy()


def _best_ranked_model(ranking: pd.DataFrame) -> str | None:
    """取得 no-winner 圖可使用的最佳 ineligible candidate 名稱。"""
    if ranking.empty or "model" not in ranking:
        return None
    rank_column = _first_present(ranking, ("overall_rank", "average_rank", "selection_order"))
    if rank_column is None:
        return None
    ordered = ranking.assign(
        _report_rank=_finite_numeric(ranking[rank_column])
    ).dropna(subset=["_report_rank"])
    ordered = ordered.sort_values(
        ["_report_rank", "model"], na_position="last", kind="mergesort"
    )
    return str(ordered.iloc[0]["model"]) if not ordered.empty else None


def _explicit_eligibility_rows(
    ranking: pd.DataFrame, *, eligible: bool
) -> pd.DataFrame:
    """只接受 literal bool/np.bool_ eligibility evidence。"""
    if ranking.empty or "eligible" not in ranking:
        return pd.DataFrame(columns=ranking.columns)
    mask = ranking["eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and bool(value) is eligible
    )
    return ranking[mask].copy()


def _best_ineligible_model(ranking: pd.DataFrame) -> str | None:
    """只從明確 ineligible 且具有有限 rank 的 candidates 選 exploratory row。"""
    return _best_ranked_model(
        _explicit_eligibility_rows(ranking, eligible=False)
    )


def _validated_winner(artifacts: Mapping[str, Any]) -> str | None:
    """確認 provided winner 同時有 candidate role 與 explicit eligible evidence。"""
    winner = artifacts.get("winner")
    if winner is None or not str(winner).strip():
        return None
    candidate = str(winner)
    ranking = _explicit_eligibility_rows(
        _candidate_ranking(artifacts), eligible=True
    )
    if ranking.empty or "model" not in ranking:
        return None
    return candidate if ranking["model"].astype(str).eq(candidate).any() else None


def _candidate_ranking(artifacts: Mapping[str, Any]) -> pd.DataFrame:
    """只保留有明確 candidate role 證據的 ranking rows。

    Ranking 自帶 `role` 時以該欄為準；舊有 ranking contract 沒有 role 時，
    只接受 fold metrics 中 role 唯一且明確為 `candidate` 的模型。兩者都沒有
    時回傳空表，讓 candidate-only 圖 fail closed，而不是從模型名稱猜測。
    """
    ranking = _frame(artifacts, "ranking", "model_ranking")
    if ranking.empty or "model" not in ranking:
        return pd.DataFrame()
    if "role" in ranking:
        roles = ranking["role"].astype("string").str.strip().str.lower()
        return ranking[roles.eq("candidate").fillna(False)].copy()

    metrics = _frame(artifacts, "fold_metrics")
    if metrics.empty or not {"model", "role"}.issubset(metrics):
        return pd.DataFrame(columns=ranking.columns)
    evidence = metrics[["model", "role"]].dropna().copy()
    evidence["model"] = evidence["model"].astype(str)
    evidence["role"] = evidence["role"].astype(str).str.strip().str.lower()
    unique_roles = evidence.groupby("model")["role"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    candidates = unique_roles[
        unique_roles.map(lambda roles: roles == ("candidate",))
    ].index
    return ranking[ranking["model"].astype(str).isin(candidates)].copy()


def _selected_model(
    artifacts: Mapping[str, Any], frame: pd.DataFrame
) -> str | None:
    """選擇 winner 或最佳 ranked candidate，並確認該表確實有此模型。"""
    ranking = _candidate_ranking(artifacts)
    candidate = _validated_winner(artifacts) or _best_ineligible_model(ranking)
    if "model" not in frame or frame.empty:
        return candidate
    available = frame["model"].dropna().astype(str)
    if candidate is not None and available.eq(candidate).any():
        return candidate
    return candidate


def _human_validation(value: str) -> str:
    """將 validation identifier 轉成短標籤。"""
    return {
        "leave_one_b_out": "Leave-one-B-out",
        "leave_one_passage_out": "Leave-one-passage-out",
        "leave_one_group_out": "Leave-one-group-out",
        "leave_one_condition_out": "Leave-one-condition-out",
    }.get(value, value)


def _build_experiment_record(context: Mapping[str, Any]) -> str:
    """組合固定十段的 reader-facing Markdown。"""
    status = str(context.get("status", "unknown"))
    winner = _validated_winner(context)
    no_winner = winner is None or status == "no_eligible_phase_only_model"
    conclusion = (
        "沒有符合門檻的 phase-only 模型；本次不發布 winner。"
        if no_winner
        else f"符合預先定義門檻的 phase-only 模型為 `{winner}`。"
    )
    raw_pc = _display(context.get("raw_pc"))
    raw_ido = _display(context.get("raw_ido"))
    complete_pairs = _display(context.get("complete_pairs"))
    exclusions = _string_list(context.get("exclusions"))
    segmentation_rate = _format_rate(context.get("segmentation_pass_rate"))
    ranking = _context_frame(context, "ranking", "model_ranking")
    metrics = _explicit_role_rows(
        _context_frame(context, "fold_metrics"), "candidate"
    )
    adjusted = _context_frame(context, "condition_adjusted_metrics")
    metadata = context.get("metadata", {})
    limitations = _string_list(context.get("limitations"))
    if not any("IDO proxy" in item for item in limitations):
        limitations.append(
            "IDO fluorescence is an IDO proxy；本實驗不量測整體功能性免疫反應。"
        )
    limitations = list(dict.fromkeys(limitations))

    sections = [
        "# Exp3 Phase-only Morphology Benchmark 實驗記錄",
        "",
        "## 1. 結論狀態",
        "",
        f"**技術摘要：{conclusion}** 目標量是 image-level background-corrected IDO proxy；"
        "結果只回答 morphology 對此 proxy 的 out-of-fold predictive evidence，不能解讀為因果或功能性免疫結論。",
        "",
        "**解讀：** Eligibility gate 是發布決策的依據；排名較前但未通過 gate 的模型只能作 exploratory diagnostic。",
        "",
        "**注意：** Negative result 不證明 morphology 完全沒有訊號，只表示在本資料、特徵、切分與門檻下證據不足。",
        "",
        "## 2. 資料與 QC",
        "",
        f"- Raw PC images：{raw_pc}",
        f"- Raw IDO images：{raw_ido}",
        f"- Complete PC–IDO pairs：{complete_pairs}",
        f"- Segmentation pass rate：{segmentation_rate}",
        f"- Exclusions：{'; '.join(exclusions) if exclusions else 'none reported'}",
        "",
        _development_validation_text(context),
        "",
        "**解讀：** 每個 complete pair 是一個 image-level target/predictor observation；上述計數界定 benchmark 分母。",
        "",
        "**注意：** Pairing 完整不等於 segmentation 或 condition metadata 正確；兩者需由各自 QC 證據確認。",
        "",
        "## 3. Primary benchmark",
        "",
        "四個 grouped validation family 使用 MAE、RMSE、R² 與 Spearman；MAE/RMSE 單位為 IDO proxy intensity。",
        "",
        _validation_tables(metrics),
        "",
        "![Model rank by validation family](figures/model_validation_rank_heatmap.png)",
        "",
        "**解讀：** Rank heatmap 比較同一候選模型跨四種 holdout family 的相對位置；數字越小越前。"
        " **注意：** 相對名次不等同效果大小，仍須和 Dummy、絕對誤差及 eligibility gates 一起讀。",
        "",
        "![Outer-fold MAE distributions](figures/fold_mae_distributions.png)",
        "",
        "**解讀：** MAE distribution 顯示模型在 outer folds 的誤差範圍與中位數。"
        " **注意：** Folds 共享有限 biological groups，不能當成大量獨立重複實驗。",
        "",
        "![Observed versus predicted IDO proxy](figures/observed_vs_predicted.png)",
        "",
        "**解讀：** Scatter 只在至少 12 個同粒度 OOF image rows 時顯示，虛線為 identity。"
        " **注意：** No-winner 狀態下圖只標示 `Exploratory — ineligible`，不構成 winner 或部署證據。",
        "",
        "![Feature importance stability](figures/feature_importance_stability.png)",
        "",
        "**解讀：** Importance 圖比較 fold 間方向與變異，僅用於診斷 model dependence。"
        " **注意：** Coefficient/permutation importance 不是 causal effect，也不證明生物機制。",
        "",
        "## 4. Eligibility gate",
        "",
        _eligibility_table(ranking),
        "",
        "**解讀：** 每個候選模型都必須通過 beat-Dummy、positive-R²、complete-fold、phase-only feature 與 prediction-SD gates。",
        "",
        "**注意：** Gate 是預先定義的發布規則；不可因某一張圖看似良好而事後放寬。",
        "",
        "## 5. Dose confounding diagnostics",
        "",
        "![Dummy, dose-only, and dose-plus-morphology diagnostics](figures/diagnostic_model_comparison.png)",
        "",
        "**解讀：** Dummy、dose-only 與 dose-plus-morphology 只用來診斷 dose confounding 與 morphology 的增量資訊。"
        " **注意：** Diagnostic model 不參與 phase-only winner 排名，也不能建立 dose 的因果效果。",
        "",
        "## 6. Condition-adjusted sensitivity",
        "",
        "Condition-adjusted sensitivity 是 robustness check，**明確不納入排名**。",
        "",
        _compact_metrics_table(adjusted),
        "",
        "![OOF residuals by B-ID, passage, and condition](figures/residuals_by_b_passage_condition.png)",
        "",
        "**解讀：** Residual panels 檢查 error 是否集中於特定 B-ID、passage 或 condition。"
        " **注意：** 這是診斷性比較；adjustment 對 feature/target 關係的敏感度不會改寫 primary rank。",
        "",
        "## 7. ΔMorphology",
        "",
        "ΔMorphology **只作描述性分析**。九組 signature 代表 **9 個 biological groups**，"
        "不是 719 個 independent samples；不得據此進行 image-level inferential claim。",
        "",
        "![Descriptive morphology delta heatmap](figures/morphology_delta_heatmap.png)",
        "",
        "**解讀：** Heatmap 顯示各 biological group/contrast 的 standardized morphology change pattern。"
        " **注意：** 顏色與格內數值描述 pattern，不提供獨立樣本的 p-value 或 causal direction。",
        "",
        "![Descriptive morphology delta PCA](figures/morphology_delta_pca.png)",
        "",
        "**解讀：** PCA 將九組高維 descriptive signatures 投影到兩軸，僅供查看相近或離群 pattern。"
        " **注意：** PCA 分離不代表 condition effect，且 biological-group n=9。",
        "",
        "## 8. 證據、推論與限制",
        "",
        "### 證據",
        "",
        f"- Fixed outer-fold metrics、OOF predictions 與 eligibility rows 支持本次結論狀態 `{status}`。",
        f"- Pairing evidence 包含 {complete_pairs} 個 complete pairs；ΔMorphology evidence 以 9 個 biological groups 為粒度。",
        "",
        "### 推論",
        "",
        f"- {conclusion}",
        "- Dose diagnostics 與 condition-adjusted sensitivity 只說明結果對 confounding/adjustment 的敏感度。",
        "- ΔMorphology 只描述 observed group-level morphology patterns。",
        "",
        "### 限制",
        "",
        "\n".join(f"- {item}" for item in limitations),
        "",
        "## 9. Reproducibility",
        "",
        _metadata_table(metadata if isinstance(metadata, Mapping) else {}),
        "",
        "**解讀：** Commit、dirty status、hashes、versions、seeds 與 runtime 用於重建同一分析環境。",
        "",
        "**注意：** 相同 seed 不能補償資料、套件或 condition mapping 改變；hash 與 dirty status 必須一起核對。",
        "",
        "## 10. 下一批資料需求",
        "",
        "1. 鎖定並人工驗證 condition mapping，保存可追溯版本。",
        "2. 補齊 donor／cell lot／batch metadata，以分離 biological 與 technical variation。",
        "3. 收集獨立 functional assay，驗證 IDO proxy 與功能性 readout 的關聯。",
        "",
        "**下一步：** 先完成 verified condition mapping，再以新增 donor/lot 的獨立批次重跑同一 frozen protocol。",
        "",
        "**待回答問題：** Negative eligibility 結果在獨立 biological batches、不同 segmentation QC threshold 與 functional assay 下是否仍成立？",
        "",
    ]
    return "\n".join(sections)


def _context_frame(context: Mapping[str, Any], *names: str) -> pd.DataFrame:
    """從 context 讀取 DataFrame alias。"""
    return _frame(context, *names)


def _display(value: object) -> str:
    """以可讀文字呈現未知或 scalar 值。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "not reported"
    return str(value)


def _format_rate(value: object) -> str:
    """以百分比顯示 0–1 pass rate，其他值明確標示 unavailable。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not reported"
    if not np.isfinite(numeric):
        return "not reported"
    return f"{numeric * 100:.1f}%" if 0 <= numeric <= 1 else f"{numeric:.1f}%"


def _string_list(value: object) -> list[str]:
    """安全正規化 reader-facing 字串清單。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _development_validation_text(context: Mapping[str, Any]) -> str:
    """呈現 optional development-only PC-vs-DAPI nucleus metrics。"""
    value = context.get("pc_nucleus_dapi_validation")
    if isinstance(value, pd.DataFrame) and not value.empty:
        return (
            "Development-only PC-vs-DAPI nucleus validation（不進入模型選擇）：\n\n"
            + _markdown_table(value.head(10))
        )
    if isinstance(value, Mapping) and value:
        return "Development-only PC-vs-DAPI nucleus validation：" + ", ".join(
            f"{key}={_display(item)}" for key, item in value.items()
        )
    return "Development-only PC-vs-DAPI nucleus validation：not run or not reported."


def _validation_tables(metrics: pd.DataFrame) -> str:
    """依固定四種 validation 產生 exact metric lookup tables。"""
    blocks: list[str] = []
    columns = ("model", "mae", "rmse", "r2", "spearman")
    for validation in _VALIDATIONS:
        blocks.extend((f"### {_human_validation(validation)}", ""))
        if metrics.empty or "validation" not in metrics:
            blocks.append("No validation metrics reported.")
        else:
            subset = metrics[metrics["validation"].astype(str).eq(validation)]
            available = [column for column in columns if column in subset]
            if subset.empty or not available:
                blocks.append("No validation metrics reported.")
            else:
                numeric = [column for column in available if column != "model"]
                if "model" in available and numeric:
                    summary = subset.groupby("model", as_index=False)[numeric].median(numeric_only=True)
                else:
                    summary = subset[available]
                blocks.append(_markdown_table(summary[list(available)] if set(available).issubset(summary) else summary))
        blocks.extend(
            (
                "",
                "**解讀：** 此表只比較同一 holdout family 下的 outer-fold performance。 "
                "**注意：** 不同 family 的 fold 數與 test cohort 不同，不應把列數當成獨立重複樣本數。",
                "",
            )
        )
    return "\n".join(blocks).rstrip()


def _eligibility_table(ranking: pd.DataFrame) -> str:
    """產生每個候選模型的 pass/fail 與原因表。"""
    if ranking.empty or "model" not in ranking:
        return "No candidate eligibility rows reported."
    rows = []
    for _, row in ranking.iterrows():
        eligible = bool(row.get("eligible", False))
        reason_value = row.get("ineligibility_reasons_json", "[]")
        try:
            parsed = json.loads(reason_value) if isinstance(reason_value, str) else reason_value
        except json.JSONDecodeError:
            parsed = [str(reason_value)]
        reasons = _string_list(parsed)
        rows.append(
            {
                "model": row["model"],
                "gate": "PASS" if eligible else "FAIL",
                "reason": "—" if eligible else "; ".join(reasons) or "unspecified gate failure",
            }
        )
    return _markdown_table(pd.DataFrame(rows))


def _compact_metrics_table(frame: pd.DataFrame) -> str:
    """呈現 sensitivity exact metrics 或 unavailable 說明。"""
    if frame.empty:
        return "No condition-adjusted metrics reported."
    columns = [column for column in ("validation", "model", "mae", "rmse", "r2", "spearman") if column in frame]
    return _markdown_table(frame[columns].head(30)) if columns else "No recognized sensitivity metrics reported."


def _metadata_table(metadata: Mapping[str, Any]) -> str:
    """將 reproducibility metadata 正規化為可稽核表格。"""
    keys = (
        "git_commit",
        "dirty",
        "config_hash",
        "manifest_hash",
        "python_version",
        "package_versions",
        "seed",
        "seeds",
        "runtime_seconds",
    )
    rows = []
    for key in keys:
        value = metadata.get(key, "not reported")
        if isinstance(value, Mapping):
            value = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
        rows.append({"field": key, "value": value})
    return _markdown_table(pd.DataFrame(rows))


def _markdown_table(frame: pd.DataFrame) -> str:
    """在不依賴 optional tabulate 套件下產生簡單 Markdown table。"""
    if frame.empty:
        return "No rows reported."
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(_escape_markdown(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_escape_markdown(_format_cell(value)) for value in values) + " |")
    return "\n".join((header, divider, *rows))


def _format_cell(value: object) -> str:
    """以短而穩定的形式格式化 Markdown cell。"""
    if value is None:
        return "NA"
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _escape_markdown(value: str) -> str:
    """避免 table cell 中的 pipe 或換行破壞 Markdown。"""
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


__all__ = [
    "REQUIRED_FIGURES",
    "REQUIRED_TABLES",
    "write_experiment_record",
    "write_figures",
    "write_result_tables",
]
