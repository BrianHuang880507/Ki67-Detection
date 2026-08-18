"""Exp4 v5 可重複更新的繁中報告 writer。"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .exploratory import CONDITION_WARNING_TEXT


def build_report_skeleton(
    *,
    targets: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
    border_report: pd.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """建立不虛構 model 結果、且嵌入 current v5 tables 的繁中報告。

    Args:
        targets: high target API 回傳的 current group target table；尚未成功時
            可為 ``None``，writer 會保留明確的 not-run 占位訊息。
        sensitivity: high sparse-FOV API 回傳的 current evidence table；不得由
            reporting layer 自行補造 threshold 或 status。
        border_report: 每個 manifest FOV 一列的 distinct whole-cell border QC。
        metadata: current generation 的 status、lock 與 feature-smoke freshness。

    Returns:
        可重複產生的繁中 Markdown 報告內容，不包含尚未完成的 model 結果。
    """
    metadata = metadata or {}
    status = str(metadata.get("status", "not_run"))
    feature_status = str(metadata.get("feature_smoke_status", "not_run"))
    target_status = str(metadata.get("target_status", "not_run"))
    analysis_execution = _analysis_execution_line(metadata)
    r2_boundary = _r2_boundary_line(metadata, targets)
    return f"""# Exp4：Rui-style feature replication on non-quantitative phase images

> 本文件由 immunity.exp4.reporting 產生，可在新的實驗輸出後重複更新。
> 本版只記錄資料組裝與分析邊界；model 尚未執行時不填入任何 model 結果。

## 執行狀態

- run status：{status}
- feature smoke status：{feature_status}（不得與 target status 混用）
- target status：{target_status}
- model/CV 結果：{analysis_execution}

## E2 PC-derived nucleus 對 DAPI 驗證

{_e2_summary(metadata)}

## Nucleus sanity（v5 E2 replacement）

{_nucleus_sanity_summary(metadata)}

## v4 資料單位與鎖定

- raw cell_level_basic.csv：23,976 rows（snapshot fail-closed）
- F0 distinct whole cells：23,012；duplicate keys：938
- post-border distinct whole cells：19,648（由當代 generation 實測 lock 驗證）
- FOV：693；每個 post-border FOV 均保留，包括只有 2 顆細胞的稀疏 FOV
- pre-border group counts 與 cell-level CV 主導關係如下；大／小組別不平衡必須揭露。

{_group_table(targets)}

## Border、FOV QC 與 target

Border report 的 cells_before／cells_after／cells_excluded_border 全部是
distinct whole-cell counts；raw nucleus-pair counts 若存在只作觀測欄。正式
group_IDO_score 是每組所有保留細胞的 median；FOV_IDO_score 只作 QC，
不進 target chain。

{_border_summary(border_report)}

{_b8_summary(targets, border_report, sensitivity)}

## Group target（current generation）

{_target_table(targets)}

## Target 尺度與相鄰間距（current post-border）

{_target_scale_summary(targets)}

## Border target bias（current post-border − pre-border）

{_border_target_bias_summary(targets)}

## Background quantization ceiling（provenance 分離）

{_background_qc_summary(metadata)}

## Sparse-FOV sensitivity（current generation）

一般組 gate 為 5%，B8_P7 gate 為 15%，equality pass；以下表格直接嵌入 high
sensitivity result，不由 reporting layer 合成 threshold/status。

{_sensitivity_table(sensitivity)}

{_sensitivity_summary(sensitivity)}

## 解讀邊界

1. 本實驗是 Rui-style feature replication on non-quantitative phase images，不是
   Rui et al. 完整 reproduction，也不是 qDPC-based prediction。
2. target 是 IDO 免疫螢光代理值，不是論文的 L-KYN assay；影像是 8-bit JPEG phase
   contrast，不是定量相位；倍率是 100X，不是論文 10X。
{_condition_visualization_boundary(metadata)}
4. FOV_IDO_score 只供 QC；Center_X、Center_Y、Orientation 若進入
   feature model，需單獨標示視野位置依賴。
5. Rui reference 是 7 donor × 323 cells 的完全平衡 cohort；本資料去重後各組
   為 458–6,152 cells（約 13 倍不平衡），cell-level 5-fold CV 會被 B4_P6 與
   B7_P6 主導，不能把高樣本數組別的影響當成 donor-level 證據。
{r2_boundary}

## Full Rui49 feature evidence

{_full_rui49_summary(metadata)}

## 已知實作偏離與有效 feature arms

{_deviations_summary(metadata)}

{_feature_arm_summary(metadata)}

## Arm 4 結論邊界（mandatory）

{_e3_boundary_summary(metadata)}

## v5 analysis artifacts

{_analysis_artifacts_summary(metadata)}

{_post_cv_analysis_summary(metadata)}

## 尚待完成

{_feature_smoke_summary(metadata)}
{_pending_summary(metadata)}
"""


def write_report_skeleton(
    path: str | Path,
    *,
    targets: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
    border_report: pd.DataFrame | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """以 atomic replace 寫出可重複更新的 REPORT.md。

    Args:
        path: staging 或 canonical REPORT.md 路徑。
        targets: current high target table；可為 ``None`` 表示尚未驗證。
        sensitivity: current high sensitivity evidence；可為 ``None`` 表示尚未
            執行正式九組 gate。
        border_report: current border QC table。
        metadata: current generation status mapping。

    Raises:
        ValueError: Markdown 暫存檔或 atomic replace 失敗。
    """
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            build_report_skeleton(
                targets=targets,
                sensitivity=sensitivity,
                border_report=border_report,
                metadata=metadata,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"REPORT.md atomic write failed: {output}") from error


def refresh_formal_metadata_and_report(
    *,
    metadata_path: str | Path,
    feature_health_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """修正既有 formal CV metadata/REPORT 的 freshness 標記。

    這是 reporting-only reconciliation seam；只讀取 current
    ``feature_health_report.csv``、``cv_metrics.csv`` 與既有 metadata，並以
    rollback transaction 只替換 ``run_metadata.json`` 與 ``REPORT.md``。
    不會重跑 CV，也不會改寫任何 numeric analysis artifact 或 checkpoint。

    Args:
        metadata_path: current ``run_metadata.json`` 路徑。
        feature_health_path: current formal health CSV；省略時取 metadata。
        report_path: current ``REPORT.md``；省略時取 metadata。

    Returns:
        refresh status、formal health counts 與已驗證的 current paths。

    Raises:
        ValueError: current generation 不符合 formal CV lock 或路徑不一致。
        OSError: metadata/REPORT atomic replacement 失敗；失敗時回復兩者。
    """
    metadata_file = Path(metadata_path).expanduser().resolve(strict=False)
    if not metadata_file.is_file():
        raise ValueError(f"metadata path 不存在：{metadata_file}")
    try:
        loaded = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"metadata 不是可讀 JSON：{metadata_file}") from error
    if not isinstance(loaded, dict):
        raise ValueError("run_metadata.json root 必須是 object")

    metadata = dict(loaded)
    feature_file = _refresh_path(
        feature_health_path or metadata.get("feature_health_report_path"),
        metadata_file.parent,
        "feature_health_path",
    )
    report_file = _refresh_path(
        report_path or metadata.get("report_path"),
        metadata_file.parent,
        "report_path",
    )
    if feature_file == report_file:
        raise ValueError("feature_health_path 與 report_path 不可相同")
    formal = _validate_formal_refresh_inputs(
        metadata,
        metadata_file=metadata_file,
        feature_file=feature_file,
    )
    health_report = formal["health_report"]
    metrics_rows = int(formal["metrics_rows"])

    updated = _reconcile_formal_metadata(
        metadata,
        feature_file=feature_file,
        formal_health=health_report,
    )
    targets = _refresh_optional_csv(updated, "group_targets_path")
    sensitivity = _refresh_optional_csv(updated, "group_target_sensitivity_path")
    border_report = _refresh_optional_csv(updated, "border_exclusion_report_path")
    report = build_report_skeleton(
        targets=targets,
        sensitivity=sensitivity,
        border_report=border_report,
        metadata=updated,
    )
    _replace_refresh_bundle(
        metadata_file,
        report_file,
        json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        report,
    )
    zero = _refresh_bool(health_report["is_zero_variance"])
    near = _refresh_bool(health_report["is_near_zero_variance"]) & ~zero
    return {
        "status": "validated",
        "metadata_path": str(metadata_file),
        "report_path": str(report_file),
        "feature_health_report_path": str(feature_file),
        "feature_health_rows": int(len(health_report)),
        "feature_health_active_rows": int((~zero).sum()),
        "near_zero_variance_columns": list(
            health_report.loc[near, "feature"].astype(str)
        ),
        "cv_metrics_rows": metrics_rows,
        "numeric_artifacts_modified": False,
    }


def _refresh_path(value: object, base: Path, label: str) -> Path:
    """解析 refresh path；禁止 historical archive 作為 current input。"""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} 缺少 current path")
    candidate = Path(value).expanduser()
    resolved = candidate if candidate.is_absolute() else (base / candidate)
    resolved = resolved.resolve(strict=False)
    if ".historical_stale." in resolved.name:
        raise ValueError(f"{label} 不可指向 historical_stale artifact：{resolved}")
    return resolved


def _refresh_bool(values: pd.Series) -> pd.Series:
    """將 CSV boolean 欄位穩定轉成 bool，避免字串 False 被視為真。"""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _validate_formal_refresh_inputs(
    metadata: Mapping[str, object],
    *,
    metadata_file: Path,
    feature_file: Path,
) -> dict[str, object]:
    """驗證 refresh 的 formal CV、health rows 與 current path contract。"""
    statuses = _as_mapping(metadata.get("analysis_status"))
    cv = _as_mapping(metadata.get("cv"))
    if str(statuses.get("cv", cv.get("status", "not_run"))) != "validated":
        raise ValueError("refresh 需要 cv validated generation")
    expected = {"cell_count": 19648, "fov_count": 693, "metrics_row_count": 120}
    for key, expected_value in expected.items():
        actual = cv.get(key, metadata.get(key))
        if int(actual) != expected_value:
            raise ValueError(
                f"formal CV lock mismatch: {key}={actual}, expected {expected_value}"
            )

    current_feature_value = metadata.get("feature_health_report_path")
    if current_feature_value is not None:
        current_feature = _refresh_path(
            current_feature_value,
            metadata_file.parent,
            "metadata.feature_health_report_path",
        )
        if current_feature != feature_file:
            raise ValueError(
                "feature_health_report_path 不是 metadata 宣告的 current path"
            )
    if not feature_file.is_file():
        raise ValueError(f"current feature health CSV 不存在：{feature_file}")
    health_report = pd.read_csv(feature_file)
    required_health = {
        "feature",
        "relative_std",
        "is_zero_variance",
        "is_near_zero_variance",
        "fallback_rate",
    }
    missing_health = sorted(required_health.difference(health_report.columns))
    if len(health_report) != 66 or missing_health:
        raise ValueError(
            "current formal feature health schema/row lock mismatch: "
            f"rows={len(health_report)}, missing={missing_health}"
        )

    metrics_value = cv.get("metrics_path", metadata.get("cv_metrics_path"))
    metrics_file = _refresh_path(metrics_value, metadata_file.parent, "cv_metrics_path")
    if not metrics_file.is_file():
        raise ValueError(f"current CV metrics CSV 不存在：{metrics_file}")
    metrics_rows = len(pd.read_csv(metrics_file))
    if metrics_rows != 120:
        raise ValueError(f"current CV metrics rows={metrics_rows}, expected 120")
    return {"health_report": health_report, "metrics_rows": metrics_rows}


def _reconcile_formal_metadata(
    metadata: Mapping[str, object],
    *,
    feature_file: Path,
    formal_health: pd.DataFrame,
) -> dict[str, object]:
    """建立 refresh 後 metadata mapping，不觸碰 numeric artifacts。"""
    updated = dict(metadata)
    zero = _refresh_bool(formal_health["is_zero_variance"])
    near = _refresh_bool(formal_health["is_near_zero_variance"]) & ~zero
    zero_columns = formal_health.loc[zero, "feature"].astype(str).tolist()
    near_columns = formal_health.loc[near, "feature"].astype(str).tolist()
    fallback_rates = {
        str(row.feature): float(row.fallback_rate)
        for row in formal_health.itertuples(index=False)
    }
    formal_health_payload = dict(_as_mapping(updated.get("feature_health")))
    formal_health_payload.update(
        {
            "status": "validated",
            "scope": "cv_formal_retained_19648_cells_65_active_features",
            "report_path": str(feature_file),
            "report_row_count": len(formal_health),
            "input_feature_count": len(formal_health),
            "active_feature_count": int((~zero).sum()),
            "zero_variance_checked_columns": len(formal_health),
            "zero_variance_removed_columns": zero_columns,
            "removed_columns": zero_columns,
            "near_zero_variance_check_status": "validated",
            "near_zero_variance_threshold": 0.05,
            "near_zero_variance_action": "record_only",
            "near_zero_variance_columns": near_columns,
            "near_zero_columns": near_columns,
            "fallback_rates": fallback_rates,
            "max_fallback_rate": max(fallback_rates.values(), default=0.0),
        }
    )
    updated.update(
        {
            "feature_health_status": "validated",
            "feature_health_report_path": str(feature_file),
            "feature_health_input_feature_count": len(formal_health),
            "feature_health_active_feature_count": int((~zero).sum()),
            "feature_health": formal_health_payload,
            "near_zero_variance_check_status": "validated",
            "near_zero_variance_threshold": 0.05,
            "near_zero_variance_action": "record_only",
            "near_zero_variance_columns": near_columns,
        }
    )
    full = _as_mapping(updated.get("full_rui49"))
    if full:
        full_payload = dict(full)
        full_payload["feature_health_report_status"] = "historical_stale"
        full_payload["feature_health_report_scope"] = (
            "historical_stale_full_extraction_49_columns"
        )
        updated["full_rui49"] = full_payload
    if _has_validated_post_cv(updated):
        post = dict(_as_mapping(updated.get("post_cv_analysis")))
        umap = dict(_as_mapping(post.get("umap")))
        post["ifn_selector_untrusted"] = True
        umap["ifn_selector_untrusted"] = True
        post["umap"] = umap
        updated["post_cv_analysis"] = post
        updated["ifn_selector_untrusted"] = True
    return updated


def _refresh_optional_csv(
    metadata: Mapping[str, object], key: str
) -> pd.DataFrame | None:
    """讀取 current reporting CSV；缺少時保留 not-run semantics。"""
    value = metadata.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(value).expanduser().resolve(strict=False)
    if ".historical_stale." in path.name or not path.is_file():
        return None
    return pd.read_csv(path)


def _replace_refresh_bundle(
    metadata_file: Path,
    report_file: Path,
    metadata_text: str,
    report_text: str,
) -> None:
    """兩檔 replace transaction；第二檔失敗時回復兩檔舊內容。"""
    token = uuid.uuid4().hex
    metadata_tmp = metadata_file.parent / f".{metadata_file.name}.refresh.{token}.tmp"
    report_tmp = report_file.parent / f".{report_file.name}.refresh.{token}.tmp"
    metadata_backup = metadata_file.parent / f".{metadata_file.name}.refresh.{token}.backup"
    report_backup = report_file.parent / f".{report_file.name}.refresh.{token}.backup"
    pairs = (
        (metadata_file, metadata_tmp, metadata_backup),
        (report_file, report_tmp, report_backup),
    )
    moved: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        metadata_tmp.write_text(metadata_text, encoding="utf-8")
        report_tmp.write_text(report_text, encoding="utf-8")
        for final, _temporary, backup in pairs:
            if final.exists():
                os.replace(final, backup)
                moved.append((final, backup))
        os.replace(metadata_tmp, metadata_file)
        committed.append(metadata_file)
        os.replace(report_tmp, report_file)
        committed.append(report_file)
    except Exception:
        for final in reversed(committed):
            final.unlink(missing_ok=True)
        for final, backup in reversed(moved):
            if backup.exists():
                os.replace(backup, final)
        raise
    finally:
        metadata_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)
        metadata_backup.unlink(missing_ok=True)
        report_backup.unlink(missing_ok=True)


def _refresh_cli(argv: list[str] | None = None) -> int:
    """執行 reporting-only formal reconciliation CLI。"""
    parser = argparse.ArgumentParser(
        description="Refresh Exp4 formal metadata/REPORT only"
    )
    parser.add_argument("--refresh-formal", action="store_true", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--feature-health")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    result = refresh_formal_metadata_and_report(
        metadata_path=_cli_absolute_path(args.metadata),
        feature_health_path=_cli_absolute_path(args.feature_health),
        report_path=_cli_absolute_path(args.report),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cli_absolute_path(value: str | None) -> str | None:
    """將 CLI 明確 path 依 cwd 正規化；省略值保留 API metadata-relative semantics。"""
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return str(candidate.resolve(strict=False))


def _group_table(targets: pd.DataFrame | None) -> str:
    if targets is None or targets.empty:
        return "_current target table 尚未寫出；計畫範圍為 458–6,152 顆、約 13 倍。_"
    values = pd.to_numeric(targets["cells_before"], errors="coerce").dropna()
    ratio_line = ""
    if not values.empty and float(values.min()) > 0:
        ratio_line = (
            f"實測 pre-border count range：{int(values.min()):,}–"
            f"{int(values.max()):,}，約 {float(values.max() / values.min()):.0f} 倍。\n\n"
        )
    return (
        ratio_line
        + "實測 pre-border group counts（最小／最大與 ratio 由 current table 計算）：\n\n"
        + _markdown_table(
            targets,
            (
                "group_id",
                "cells_before",
                "cells_after",
                "group_IDO_score",
                "group_IDO_score_all_cells",
                "delta",
                "confidence_flag",
            ),
        )
    )


def _target_table(targets: pd.DataFrame | None) -> str:
    if targets is None or targets.empty:
        return "_尚未寫出 9-row group target table。_"
    columns = (
        "group_id",
        "b_id",
        "passage",
        "fov_count",
        "cells_before",
        "cells_after",
        "group_IDO_score",
        "group_IDO_score_all_cells",
        "delta",
        "confidence_flag",
    )
    return _markdown_table(targets, columns)


def _target_range_text(targets: pd.DataFrame | None) -> str:
    """回傳 current target range 的解讀文字，缺少 target 時為 not_run。"""
    target_range = _target_range_value(targets)
    if target_range is None:
        return "not_run target range"
    return f"{target_range:.8f} 灰階 target range"


def _target_range_value(targets: pd.DataFrame | None) -> float | None:
    """從 current target table 回傳 finite ``group_IDO_score`` 全距。"""
    if targets is None or targets.empty or "group_IDO_score" not in targets.columns:
        return None
    scores = pd.to_numeric(targets["group_IDO_score"], errors="coerce")
    values = scores.to_numpy(dtype=float)
    finite = scores.notna().to_numpy() & np.isfinite(values)
    if not finite.any():
        return None
    return float(values[finite].max() - values[finite].min())


def _target_scale_summary(targets: pd.DataFrame | None) -> str:
    """從 current target table 計算 target 全距與相鄰間距。

    Args:
        targets: current group target table；分數欄應為 post-border median。

    Returns:
        顯示 target 全距、升冪排序值、8 個相鄰間距與小於 0.2 的數量，或
        在輸入尚未完成時顯示明確的 not-run 訊息。
    """
    if targets is None or targets.empty or "group_IDO_score" not in targets.columns:
        return "- current post-border target scale：not_run（缺少 group_IDO_score）。"

    scores = pd.to_numeric(targets["group_IDO_score"], errors="coerce")
    finite = scores.notna() & np.isfinite(scores.to_numpy(dtype=float))
    if not finite.any():
        return "- current post-border target scale：not_run（沒有 finite target）。"

    frame = targets.loc[finite].copy()
    frame["_target_score"] = scores.loc[finite].astype(float)
    if "group_id" in frame.columns:
        frame["_target_group"] = frame["group_id"].astype(str)
    else:
        frame["_target_group"] = frame.index.astype(str)
    frame = frame.sort_values(
        ["_target_score", "_target_group"], kind="mergesort"
    )
    values = frame["_target_score"].to_numpy(dtype=float)
    target_range = _target_range_value(targets)
    if target_range is None:
        return "- current post-border target scale：not_run（沒有 finite target）。"
    gaps = np.diff(values)
    small_gap_count = int(np.count_nonzero(gaps < 0.2))
    sorted_text = "、".join(
        f"{group}={value:.8f}"
        for group, value in zip(frame["_target_group"], values)
    )
    gap_text = "、".join(f"{gap:.8f}" for gap in gaps) or "not_run"
    return "\n".join(
        (
            f"- current post-border target range：{target_range:.8f} 灰階 "
            f"（約 {target_range:.2f}；exact {target_range:.14f}）。",
            f"- sorted targets（ascending）：{sorted_text}（rows={len(values)}）。",
            f"- adjacent gaps（{len(gaps)}）：{gap_text}。",
            f"- gaps <0.2：{small_gap_count}/{len(gaps)}。",
        )
    )


def _border_target_bias_summary(targets: pd.DataFrame | None) -> str:
    """從 target table 的 ``delta`` 計算 border target 系統性偏移。

    Args:
        targets: current group target table；``delta`` 定義為
            post-border 減 pre-border。

    Returns:
        每組 delta、符號數、占 target range 百分比與最大絕對偏移的 Markdown，
        或在輸入缺欄時顯示 not-run。
    """
    required = {"group_IDO_score", "delta"}
    if targets is None or targets.empty or not required.issubset(targets.columns):
        return "- border target bias：not_run（缺少 group_IDO_score 或 delta）。"

    scores = pd.to_numeric(targets["group_IDO_score"], errors="coerce")
    deltas = pd.to_numeric(targets["delta"], errors="coerce")
    finite = (
        scores.notna()
        & deltas.notna()
        & np.isfinite(scores.to_numpy(dtype=float))
        & np.isfinite(deltas.to_numpy(dtype=float))
    )
    if not finite.any():
        return "- border target bias：not_run（沒有 finite target delta）。"

    frame = targets.loc[finite].copy()
    frame["_target_score"] = scores.loc[finite].astype(float)
    frame["_delta"] = deltas.loc[finite].astype(float)
    if "group_id" in frame.columns:
        frame["_target_group"] = frame["group_id"].astype(str)
    else:
        frame["_target_group"] = frame.index.astype(str)
    target_range = float(
        frame["_target_score"].max() - frame["_target_score"].min()
    )
    delta_values = frame["_delta"].to_numpy(dtype=float)
    positive_count = int(np.count_nonzero(delta_values > 0))
    negative_count = int(np.count_nonzero(delta_values < 0))
    zero_count = int(np.count_nonzero(delta_values == 0))
    absolute_delta = frame["_delta"].abs()
    max_index = absolute_delta.idxmax()
    max_row = frame.loc[max_index]
    max_delta = float(max_row["_delta"])
    max_pct = (
        abs(max_delta) / target_range * 100.0 if target_range > 0 else float("nan")
    )
    max_pct_text = f"{max_pct:.4f}%" if np.isfinite(max_pct) else "not_run"

    rows = [
        "| group_id | delta (post-border - pre-border) | abs(delta)/target range (%) |",
        "| --- | --- | --- |",
    ]
    for group_value, delta_value in frame.loc[:, ["_target_group", "_delta"]].itertuples(
        index=False, name=None
    ):
        group = str(group_value)
        delta = float(delta_value)
        pct = abs(delta) / target_range * 100.0 if target_range > 0 else float("nan")
        pct_text = f"{pct:.4f}%" if np.isfinite(pct) else "not_run"
        rows.append(f"| {group} | {_format_signed_delta(delta)} | {pct_text} |")

    return "\n".join(
        (
            f"- delta 定義：post-border − pre-border；符號：{positive_count} 正、"
            f"{negative_count} 負、{zero_count} 零。",
            f"- target range denominator：{target_range:.8f} 灰階。",
            f"- 最大絕對偏移：{max_row['_target_group']} "
            f"{_format_signed_delta(max_delta)}（{max_pct_text} target range）。",
            *rows,
        )
    )


def _format_signed_delta(value: float) -> str:
    """格式化帶正負號的 target delta。"""
    return f"{value:+.10f}"


def _sensitivity_table(sensitivity: pd.DataFrame | None) -> str:
    if sensitivity is None or sensitivity.empty:
        return "_尚未寫出 9-row sensitivity table。_"
    columns = (
        "group_id",
        "sparse_fov_count",
        "target_with_sparse_fovs",
        "target_without_sparse_fovs",
        "delta_without_minus_with",
        "absolute_delta",
        "baseline_target_range",
        "absolute_delta_pct_of_range",
        "threshold_pct",
        "threshold_absolute",
        "status",
    )
    return _markdown_table(sensitivity, columns)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return f"_table schema incomplete; missing: {', '.join(missing)}_"
    rows = frame.loc[:, list(columns)].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body: list[str] = []
    for row in rows.itertuples(index=False, name=None):
        values = [_format_cell(value) for value in row]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join((header, separator, *body))


def _format_cell(value: object) -> str:
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{key}: {_format_cell(item)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(_format_cell(item) for item in value) + "]"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _border_summary(border_report: pd.DataFrame | None) -> str:
    if border_report is None or border_report.empty:
        return "- border report：尚未寫出；預期 693 rows。"
    if "cells_after" not in border_report.columns:
        return f"- border report：{len(border_report):,} rows（缺少 cells_after）。"
    after = pd.to_numeric(border_report["cells_after"], errors="coerce")
    return (
        f"- border report：{len(border_report):,} FOV rows；"
        f"distinct post-border cells 合計 {int(after.sum()):,}；"
        f"最小 FOV {int(after.min()):,} 顆。"
    )


def _b8_summary(
    targets: pd.DataFrame | None,
    border_report: pd.DataFrame | None,
    sensitivity: pd.DataFrame | None = None,
) -> str:
    """摘要 B8_P7 的 current target 不穩定性與實測 sensitivity。

    Args:
        targets: current 9-group target table。
        border_report: current FOV-level distinct whole-cell border report。
        sensitivity: current high sparse-FOV sensitivity evidence；可為 ``None``。

    Returns:
        包含 low-confidence、pre/post count、retained-cell median 與可用時
        sensitivity 的繁中 Markdown 條列。
    """
    if targets is None or targets.empty or "group_id" not in targets.columns:
        return "- B8_P7：low confidence；current target 尚未寫出。"
    b8 = targets.loc[targets["group_id"].astype(str).eq("B8_P7")]
    if b8.empty:
        return "- B8_P7：low confidence；current table 尚未包含該組。"
    lines = [
        f"- B8_P7 target 較其餘八組不穩定；"
        f"confidence_flag={b8.iloc[0].get('confidence_flag', 'low')}；"
        f"pre-border cells={_format_cell(b8.iloc[0].get('cells_before'))}；"
        f"post-border cells={_format_cell(b8.iloc[0].get('cells_after'))}。"
    ]
    if (
        border_report is not None
        and not border_report.empty
        and "group_id" in border_report.columns
        and "cells_after" in border_report.columns
    ):
        values = pd.to_numeric(
            border_report.loc[
                border_report["group_id"].astype(str).eq("B8_P7"),
                "cells_after",
            ],
            errors="coerce",
        ).dropna()
        if not values.empty:
            lines.append(
                f"- B8_P7 retained cells/FOV median：{float(values.median()):.12g}。"
            )
    if (
        sensitivity is not None
        and not sensitivity.empty
        and "group_id" in sensitivity.columns
        and "absolute_delta_pct_of_range" in sensitivity.columns
    ):
        measured = pd.to_numeric(
            sensitivity.loc[
                sensitivity["group_id"].astype(str).eq("B8_P7"),
                "absolute_delta_pct_of_range",
            ],
            errors="coerce",
        ).dropna()
        if not measured.empty:
            lines.append(
                f"- B8_P7 實測 sensitivity：{float(measured.iloc[0]):.12g}%。"
            )
    return "\n".join(lines)


def _sensitivity_summary(sensitivity: pd.DataFrame | None) -> str:
    if sensitivity is None or sensitivity.empty:
        return "- B8_P7 實測 sensitivity：尚未寫出。"
    if "group_id" not in sensitivity.columns:
        return "- sensitivity report schema 不完整。"
    value = pd.to_numeric(
        sensitivity.loc[
            sensitivity["group_id"].astype(str).eq("B8_P7"),
            "absolute_delta_pct_of_range",
        ],
        errors="coerce",
    ).dropna()
    if value.empty:
        return "- B8_P7 實測 sensitivity：尚未寫出。"
    return (
        f"- B8_P7 實測 sensitivity：{float(value.iloc[0]):.12g}%（以 high output 為準）。"
    )


def _background_qc_summary(metadata: Mapping[str, object]) -> str:
    """呈現 background quantization ceiling 並保留 pilot/full provenance。

    Args:
        metadata: current generation metadata；預期含
            ``ido_background_qc`` nested mapping。

    Returns:
        由 metadata 直接組成的 background QC 摘要。缺少 metadata 時只輸出
        not-run，不猜測任何 background 數值或 scope。
    """
    qc = _as_mapping(metadata.get("ido_background_qc"))
    if not qc:
        return (
            "- background QC：not_run；尚未提供 target-side quantization ceiling "
            "與 provenance。"
        )

    lines = ["- background QC status：provided。"]
    summary = qc.get("summary", "not_run")
    source = qc.get("source", "not_run")
    metric = qc.get("metric", "not_run")
    scoped_fields = any(
        isinstance(qc.get(field), Mapping) for field in ("summary", "source", "metric")
    )
    if not scoped_fields:
        lines.extend(
            (
                f"- summary：{summary}。",
                f"- source：{source}。",
                f"- metric：{metric}。",
            )
        )
    scope = qc.get("scope")
    if scope is not None:
        lines.append(f"- scope：{scope}。")

    if scoped_fields:
        for key, label in (("pilot_80_fov", "pilot 80-FOV"), ("full_693", "full 693-FOV")):
            evidence_summary = _scoped_background_value(summary, key)
            evidence_source = _scoped_background_value(source, key)
            evidence_metric = _scoped_background_value(metric, key)
            lines.append(
                f"- {label} evidence：summary={evidence_summary}；"
                f"source={evidence_source}；metric={evidence_metric}。"
            )
    else:
        for key, label in (("pilot_80_fov", "pilot 80-FOV"), ("full_693", "full 693-FOV")):
            evidence = _as_mapping(qc.get(key))
            if not evidence:
                continue
            evidence_summary = evidence.get("summary", "not_run")
            evidence_source = evidence.get("source", "not_run")
            evidence_metric = evidence.get("metric", "not_run")
            lines.append(
                f"- {label} evidence：summary={evidence_summary}；"
                f"source={evidence_source}；metric={evidence_metric}。"
            )

    lines.append(
        "- provenance boundary：pilot 80-FOV normalized evidence 與 full-693 "
        "raw-gray QC 必須分開，不得把 pilot 冒稱 full data。"
    )
    return "\n".join(lines)


def _scoped_background_value(value: object, scope: str) -> object:
    """取出 scoped background metadata；缺欄時以 not_run 表示。"""
    if isinstance(value, Mapping):
        return value.get(scope, "not_run")
    return value


def _full_rui49_summary(metadata: Mapping[str, object]) -> str:
    """呈現 full Rui49 extraction evidence，與 smoke evidence 分離。

    Args:
        metadata: current generation metadata；預期含 ``full_rui49`` nested
            mapping。缺少欄位時以 ``not_run`` 表示，不由 smoke scope 補值。

    Returns:
        full extraction status、row/cell/FOV scope、exact consistency、health、
        timing 與 canonical paths 的 Markdown 摘要。
    """
    full = _as_mapping(metadata.get("full_rui49"))
    status = str(full.get("status", "not_run"))
    if status == "not_run":
        return "\n".join(
            (
                "- full extraction status：not_run。",
                "- full 693-FOV feature evidence：not_run；尚未完成完整 extraction，"
                "不可用 smoke 95 cells 冒充。",
            )
        )

    raw_rows = _metadata_display(full, "raw_pair_row_count")
    canonical_count = _metadata_display(full, "canonical_feature_count")
    pre_border = _metadata_display(full, "pre_border_distinct_cell_count")
    retained = _metadata_display(full, "retained_cell_count")
    fov_count = _metadata_display(full, "fov_count")
    elapsed = _elapsed_display(full.get("elapsed_seconds", "not_run"))
    csv_path = _metadata_display(full, "cell_level_path")
    health_path = _metadata_display(full, "feature_health_report_path")
    healthy_count = _metadata_display(full, "healthy_feature_count")
    removed = _sequence_display(full.get("removed_zero_variance_columns"))
    near_zero = _sequence_display(full.get("near_zero_variance_columns"))
    nonfinite = _sequence_display(full.get("nonfinite_columns"))
    fallback = _fallback_rate_display(full.get("max_fallback_rate", "not_run"))
    consistency_status, consistency_details = _consistency_display(
        full.get("retained_consistency", "not_run"),
        parent_status=status,
    )
    removed_label = (
        "唯一 removed zero-variance"
        if len(_string_sequence(full.get("removed_zero_variance_columns"))) == 1
        else "removed zero-variance"
    )

    current_health = _current_formal_health(metadata)
    if current_health:
        current_scope = str(current_health.get("scope", "unknown"))
        current_input = _metadata_display(current_health, "input_feature_count")
        current_active = _metadata_display(current_health, "active_feature_count")
        current_removed = _sequence_display(
            current_health.get("zero_variance_removed_columns", ())
        )
        current_near_zero = _sequence_display(
            current_health.get("near_zero_variance_columns", ())
        )
        current_nonfinite = _sequence_display(
            current_health.get("nonfinite_columns", ())
        )
        current_fallback = _fallback_rate_display(
            current_health.get("fallback_rates", current_health.get("max_fallback_rate", "not_run"))
        )
        current_path = str(
            metadata.get(
                "feature_health_report_path",
                current_health.get("report_path", "not_run"),
            )
        )
        return "\n".join(
            (
                f"- full extraction status：{status}。",
                f"- {raw_rows} raw output rows × {canonical_count} canonical features。",
                f"- retained {retained}/{fov_count}（distinct cells/FOV）。",
                f"- pre-border distinct whole cells：{pre_border}。",
                f"- full retained exact consistency：{consistency_status}"
                f"{consistency_details}。",
                f"- current formal feature health：scope={current_scope}；"
                f"{current_input} checked→{current_active} active；"
                f"zero-variance removed：{current_removed}；"
                f"near-zero variance columns：{current_near_zero}；"
                f"nonfinite columns：{current_nonfinite}；"
                f"max fallback rate：{current_fallback}。",
                f"- elapsed：{elapsed}。",
                f"- canonical CSV：{csv_path}。",
                f"- current formal feature health report：{current_path}。",
            )
        )

    return "\n".join(
        (
            f"- full extraction status：{status}。",
            f"- {raw_rows} raw output rows × {canonical_count} canonical features。",
            f"- retained {retained}/{fov_count}（distinct cells/FOV）。",
            f"- pre-border distinct whole cells：{pre_border}。",
            f"- full retained exact consistency：{consistency_status}"
            f"{consistency_details}。",
            f"- feature health：{healthy_count} healthy；"
            f"{removed_label}：{removed}；"
            f"near-zero variance columns：{near_zero}；"
            f"nonfinite columns：{nonfinite}。",
            f"- max fallback rate：{fallback}。",
            f"- elapsed：{elapsed}。",
            f"- canonical CSV：{csv_path}。",
            f"- feature health report：{health_path}。",
        )
    )


def _current_formal_health(metadata: Mapping[str, object]) -> Mapping[str, object]:
    """取得 current formal feature-health evidence，排除 historical smoke。"""
    health = _as_mapping(metadata.get("feature_health"))
    status = str(health.get("status", metadata.get("feature_health_status", "not_run")))
    if status != "validated":
        return {}
    current = dict(health)
    current.setdefault(
        "scope", "cv_formal_retained_19648_cells_65_active_features"
    )
    current.setdefault(
        "input_feature_count", metadata.get("feature_health_input_feature_count", "not_run")
    )
    current.setdefault(
        "active_feature_count", metadata.get("feature_health_active_feature_count", "not_run")
    )
    current.setdefault(
        "zero_variance_removed_columns",
        health.get("removed_columns", metadata.get("removed_zero_variance_columns", ())),
    )
    current.setdefault(
        "near_zero_variance_columns",
        health.get("near_zero_columns", metadata.get("near_zero_variance_columns", ())),
    )
    current.setdefault("nonfinite_columns", ())
    return current


def _metadata_display(metadata: Mapping[str, object], key: str) -> str:
    """以 report 慣用千分位格式呈現 metadata scalar，缺欄則 not_run。"""
    value = metadata.get(key, "not_run")
    if value == "not_run":
        return "not_run"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.12g}"
    return str(value)


def _elapsed_display(value: object) -> str:
    """格式化 full extraction elapsed seconds。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not_run"
    return f"{number:.3f} s" if np.isfinite(number) else "not_run"


def _sequence_display(value: object) -> str:
    """格式化 metadata 的 feature name sequence，空序列顯示 none。"""
    names = _string_sequence(value)
    return "、".join(names) if names else "none"


def _fallback_rate_display(value: object) -> str:
    """格式化 fraction-based fallback rate 為百分比。"""
    candidates: list[object]
    if isinstance(value, Mapping):
        candidates = list(value.values())
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        candidates = [value]
    rates: list[float] = []
    for candidate in candidates:
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            rates.append(number)
    if not rates:
        return "not_run"
    return f"{max(rates) * 100:.12g}%"


def _consistency_display(
    value: object, *, parent_status: str = "not_run"
) -> tuple[str, str]:
    """格式化 native retained consistency status 與 scope counts。

    ``WholeCellFeatureConsistencyResult.to_dict()`` 沒有 status；只有在
    validated full wrapper 下才可將這份 exact payload 顯示為 passed。
    """
    if not isinstance(value, Mapping):
        return str(value), ""
    if "status" in value:
        status = str(value["status"])
    elif parent_status == "validated":
        status = "passed"
    else:
        status = "not_run"
    details: list[str] = []
    if "scope" in value:
        details.append(f"scope={value['scope']}")
    detail_fields = (
        (("raw_pair_row_count", "checked_row_count"), "raw pairs"),
        (("checked_cell_count",), "cells"),
        (("checked_duplicate_key_count",), "duplicate keys"),
        (("fov_count", "checked_fov_count"), "FOV"),
        (("canonical_feature_count", "feature_count"), "features"),
    )
    for keys, label in detail_fields:
        key = next((candidate for candidate in keys if candidate in value), None)
        if key is not None:
            details.append(f"{_metadata_display(value, key)} {label}")
    if details and details[0].startswith("scope="):
        suffix = f"（{details[0]}；{'／'.join(details[1:])}）"
    else:
        suffix = f"（{'／'.join(details)}）" if details else ""
    return status, suffix


def _feature_smoke_summary(metadata: Mapping[str, object]) -> str:
    """依 current metadata 產生 feature smoke 與 consistency evidence。

    Args:
        metadata: current generation 的 run metadata；nested smoke、health 與
            whole-cell consistency 欄位均直接來自 feature smoke writer。

    Returns:
        validated 時的 5-image smoke 摘要，或未執行時保留的 stale/not-run
        占位訊息。此 helper 不會推定 full 693-FOV 已完成。
    """
    feature_status = str(metadata.get("feature_smoke_status", "not_run"))
    if feature_status not in {"validated", "smoke_validated", "passed"}:
        return (
            "- 在新版 source 尚未重跑 49 特徵前，feature smoke 只能標為 "
            "historical_stale／not_run，不可宣稱 current validated。"
        )

    smoke = _as_mapping(metadata.get("smoke_test"))
    consistency = _as_mapping(
        metadata.get("whole_cell_feature_consistency_smoke")
    )
    health = _as_mapping(metadata.get("feature_health_smoke"))

    smoke_status = str(smoke.get("status", "not_run"))
    image_count = _format_cell(smoke.get("image_count", "unknown"))
    consistency_status = str(consistency.get("status", "not_run"))
    consistency_scope = str(consistency.get("scope", "unknown"))
    raw_pairs = _format_cell(consistency.get("raw_pair_row_count", "unknown"))
    distinct_cells = _format_cell(
        consistency.get("distinct_label_count", "unknown")
    )
    multirow_keys = _format_cell(consistency.get("multirow_key_count", "unknown"))

    health_scope = str(
        health.get("health_scope", health.get("scope", "unknown"))
    )
    post_border_cells = _format_cell(
        health.get(
            "master_post_border_label_count",
            health.get("eligible_cell_count", "unknown"),
        )
    )
    removed_columns = _string_sequence(health.get("removed_columns"))
    if len(removed_columns) == 1:
        removed_text = f"唯一 zero-variance feature removed：{removed_columns[0]}"
    elif removed_columns:
        removed_text = (
            "zero-variance features removed：" + "、".join(removed_columns)
        )
    else:
        removed_text = "zero-variance feature removed：none reported"

    fallback_rates = _as_mapping(health.get("fallback_rates"))
    numeric_rates: list[float] = []
    for rate in fallback_rates.values():
        try:
            numeric_rate = float(rate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_rate):
            numeric_rates.append(numeric_rate)
    fallback_text = (
        f"{max(numeric_rates) * 100:.12g}%" if numeric_rates else "unknown"
    )
    smoke_report = str(
        smoke.get("report_path", metadata.get("feature_smoke_report", "unknown"))
    )
    full_status = str(
        metadata.get("full_693_feature_consistency_status", "not_run")
    )
    full_text = (
        f"{full_status}（尚未驗證）"
        if full_status == "not_run"
        else f"{full_status}（本報告不將其視為 current validated）"
    )
    return "\n".join(
        (
            f"- 5-image current smoke：{smoke_status}（image_count={image_count}）。",
            f"- canonical49 consistency：scope={consistency_scope}；"
            f"{raw_pairs} raw pairs／"
            f"{distinct_cells} distinct／{multirow_keys} multirow；"
            f"status={consistency_status}。",
            f"- full693 consistency：{full_text}。",
            f"- feature health：health scope={health_scope}；"
            f"{post_border_cells} post-border cells。",
            f"- {removed_text}。",
            f"- fallback max：{fallback_text}。",
            f"- texture zero-reservation evidence：{smoke_report}。",
        )
    )


def _deviations_summary(metadata: Mapping[str, object]) -> str:
    """呈現產物級的實作偏離，避免只依賴對話紀錄。

    Args:
        metadata: current run metadata；``deviations`` 可為 mapping 或 mapping
            sequence。

    Returns:
        每項偏離的原本行為、實際行為與原因；缺少資料時明確標示 not-run。
    """
    raw = metadata.get("deviations")
    if isinstance(raw, Mapping):
        candidates: list[Mapping[str, object]] = [raw]
    elif isinstance(raw, (list, tuple)):
        candidates = [item for item in raw if isinstance(item, Mapping)]
    else:
        candidates = []
    if not candidates:
        return "- deviations：not_run。"

    lines: list[str] = []
    for item in candidates:
        deviation_id = str(item.get("id", "unnamed"))
        lines.extend(
            (
                f"- {deviation_id}（Feret deviation）：",
                f"  - 原本的行為：{item.get('original_behavior', 'not_run')}。",
                f"  - 實際改成：{item.get('new_behavior', 'not_run')}。",
                f"  - 原因：{item.get('reason', 'not_run')}。",
            )
        )
    return "\n".join(lines)


def _e2_summary(metadata: Mapping[str, object]) -> str:
    """呈現 E2 label-cache 驗證狀態，不以缺失資料偽造 Dice／IoU。"""
    e2 = _as_mapping(metadata.get("e2"))
    if not e2:
        return "- E2 status：not_run；尚未提供 DAPI label-cache 驗證 evidence。"
    status = str(e2.get("status", "not_run"))
    reason = str(e2.get("reason", "not_run"))
    reason_text = reason if reason.endswith(("。", ".", "!", "?")) else reason + "。"
    expected = e2.get("expected_fov_count", "not_run")
    validated = e2.get("validated_fov_count", "not_run")
    lines = [
        f"- E2 status：{status}；validated FOV={validated}/{expected}。",
        f"- E2 evidence：{reason_text}",
    ]
    provenance = _as_mapping(e2.get("provenance"))
    if provenance:
        details = "；".join(
            f"{key}={_format_cell(value)}" for key, value in provenance.items()
        )
        lines.append(f"- E2 provenance：{details}。")
    if status == "canceled_v5_no_dapi_acquired":
        lines.append(
            "- v5 裁決：九個 Exp4 dataset 從未取得 DAPI；E2 不執行，"
            "不把取消視為 Dice／IoU 失敗。"
        )
    if status != "validated":
        lines.append(
            "- 未產生 Dice／IoU 數值；DAPI 只允許作 E2 驗證，未進入任何 predictor。"
        )
    return "\n".join(lines)


def _nucleus_sanity_summary(metadata: Mapping[str, object]) -> str:
    """呈現 v5 不需 DAPI 的 nucleus sanity evidence 與近零變異規則。"""
    sanity = _as_mapping(metadata.get("nucleus_sanity"))
    status = str(sanity.get("status", "not_run"))
    path = sanity.get("report_path", "not_run")
    threshold = metadata.get(
        "near_zero_variance_threshold",
        sanity.get("near_zero_variance_threshold", 0.05),
    )
    action = metadata.get(
        "near_zero_variance_action",
        sanity.get("near_zero_variance_action", "record_only"),
    )
    columns = metadata.get(
        "near_zero_variance_columns",
        sanity.get("near_zero_variance_columns", ()),
    )
    names = _string_sequence(columns)
    column_text = "、".join(names) if names else "none reported"
    lines = [
        f"- nucleus sanity status：{status}；report={path}。",
        f"- near-zero variance threshold：CV < {float(threshold):.2f}；"
        f"action={action}（不自動移除）。",
        f"- near-zero variance columns：{column_text}。",
    ]
    if sanity:
        for key in (
            "nucleus_area_cell_area_ratio_gt1_count",
            "nucleus_area_cell_area_ratio_p50",
            "nucleus_sphericity_median",
            "spearman_cell_area_nucleus_area",
            "nucleus_area_cv",
            "nucleus_solidity_cv",
        ):
            if key in sanity:
                lines.append(f"- {key}：{_format_cell(sanity[key])}。")
    return "\n".join(lines)


def _e3_boundary_summary(metadata: Mapping[str, object]) -> str:
    """輸出 v5 Arm 4 mandatory boundary template，不寫 CV 解釋結論。"""
    boundary = _as_mapping(metadata.get("e3_arm4_boundary"))
    return "\n".join(
        (
            f"- status：{boundary.get('status', 'mandatory')}。",
            "- Arm 4 回答的是「這套 phase-derived 核代理特徵有沒有加到訊號」，"
            "不是「核特徵有沒有用」。",
            "- 若 Arm 4 勝過 rui_48，只能說這組代理特徵帶入額外資訊。",
            "- 若 Arm 4 未勝過 rui_48，不可推論「核沒有用」；"
            "不可推論核沒有用；無法區分核真的沒用與核代理分割抓不到核形態。",
            "- 不引用 Klinker 的 Hoechst 真實核證據支持本 arm。",
        )
    )


def _feature_arm_summary(metadata: Mapping[str, object]) -> str:
    """呈現 CV 使用的四個固定 arm 與零變異欄位移除紀錄。"""
    raw_arms = metadata.get("feature_arms")
    if not isinstance(raw_arms, Mapping):
        return "- feature arms：not_run。"

    required = (
        "geometry_24",
        "rui_48",
        "rui_filtered",
        "rui_48_plus_nucleus",
    )
    lines = ["- feature arms（名稱固定）："]
    for name in required:
        spec = raw_arms.get(name)
        if isinstance(spec, Mapping):
            count = spec.get("count", "not_run")
            role = spec.get("role", "")
            suffix = f"；{role}" if role else ""
            lines.append(f"  - `{name}`：{count} 欄{suffix}。")
        else:
            lines.append(f"  - `{name}`：not_run。")

    removed = _string_sequence(metadata.get("removed_zero_variance_columns"))
    if not removed:
        full = _as_mapping(metadata.get("full_rui49"))
        removed = _string_sequence(full.get("removed_zero_variance_columns"))
    removed_text = "、".join(removed) if removed else "none"
    healthy_count = metadata.get("healthy_rui_feature_count")
    if healthy_count is None:
        full = _as_mapping(metadata.get("full_rui49"))
        healthy_count = full.get("healthy_feature_count", "not_run")
    lines.append(
        f"- cell__MinIntensity 為零變異欄位，從所有 arm 移除；"
        f"canonical 49→48 effective Rui features（健康欄位={healthy_count}；"
        f"removed={removed_text}）。"
    )
    return "\n".join(lines)


def _analysis_execution_line(metadata: Mapping[str, object]) -> str:
    """回報 v5 analysis 是否完成；不在此解讀任何 R² 或模型優劣。"""
    if _has_validated_post_cv(metadata):
        return "validated（formal CV 與 post-CV diagnostics 已發布）"
    statuses = _as_mapping(metadata.get("analysis_status"))
    status = str(statuses.get("cv", "not_run"))
    if status == "validated":
        return "validated（metrics/OOF 與 provenance tables 已發布；本報告不寫解讀結論）"
    if status == "blocked":
        return "blocked（未發布 canonical analysis bundle）"
    return "**尚未執行**"


def _condition_visualization_boundary(metadata: Mapping[str, object]) -> str:
    """以 future-conditional wording 限制 condition visualization 的解讀。"""
    statuses = _as_mapping(metadata.get("analysis_status"))
    umap_status = str(statuses.get("umap", "not_run"))
    if umap_status == "validated":
        return (
            "3. 已產生 umap_by_condition.png；"
            f"{CONDITION_WARNING_TEXT}"
            f"（目前 UMAP status={umap_status}。）"
        )
    return (
        "3. 若後續產生 umap_by_condition.png，condition 標籤的可信度未確認；"
        "該圖僅作視覺化，**不得據此推導任何劑量結論**。"
        f"（目前 UMAP status={umap_status}。）"
    )


def _pending_summary(metadata: Mapping[str, object]) -> str:
    """只列出尚未執行的 analysis artifact，不重複已 validated 的 CV。"""
    statuses = _as_mapping(metadata.get("analysis_status"))
    pending: list[str] = []
    if str(statuses.get("cv", "not_run")) != "validated":
        pending.append("model、OOF、CV")
    if str(statuses.get("umap", "not_run")) == "not_run":
        pending.append("UMAP")
    if str(statuses.get("kmeans", "not_run")) == "not_run":
        pending.append("k-means")
    if not pending:
        return "- 尚待完成：none。"
    return f"- 尚待完成：{'、'.join(pending)}。"


def _r2_boundary_line(
    metadata: Mapping[str, object], targets: pd.DataFrame | None
) -> str:
    """呈現 R² 的事實狀態與固定尺度邊界，不解讀模型結果。"""
    statuses = _as_mapping(metadata.get("analysis_status"))
    if _has_validated_post_cv(metadata):
        best = _as_mapping(_as_mapping(metadata.get("post_cv_analysis")).get("best_configuration"))
        line = (
            "6. formal CV 的 best configuration 為 "
            f"{best.get('configuration_id', 'not_run')}；"
            f"mean fold Rui R²={_format_number(best.get('mean_rui_r2'))}，"
            f"cell-level R²={_format_number(best.get('mean_cell_r2'))}。"
            "R² 仍須與 "
            f"{_target_range_text(targets)}、密集相鄰間距與 background "
            "quantization ceiling 一起解讀。"
        )
        if str(best.get("configuration_id", "")).startswith("rui_48_plus_nucleus__"):
            line += "\n" + _post_cv_arm4_boundary_line(metadata)
        return line
    if str(statuses.get("cv", "not_run")) == "validated":
        return (
            "6. R²/model 已發布於 machine-readable CV tables；本報告不寫 R² 解讀"
            f"結論，仍須與 {_target_range_text(targets)}、密集相鄰間距與 background "
            "quantization ceiling 一起解讀。"
        )
    return (
        "6. R²／model 尚未執行；未來無論 R² 高低，均須與 "
        f"{_target_range_text(targets)}、密集相鄰間距與 background quantization "
        "ceiling 一起解讀。"
    )


def _analysis_artifacts_summary(metadata: Mapping[str, object]) -> str:
    """列出 analysis tables 的狀態與路徑，不產生 CV 解釋性結論。"""
    statuses = _as_mapping(metadata.get("analysis_status"))
    rows = [
        ("feature_health", "feature_health"),
        ("nucleus_sanity", "nucleus_sanity"),
        ("feature_redundancy", "feature_redundancy"),
        ("leakage_preflight", "leakage_preflight"),
        ("cv", "cv"),
    ]
    path_fields = {
        "feature_health": "feature_health_report_path",
        "nucleus_sanity": "nucleus_sanity_report_path",
        "feature_redundancy": "feature_redundancy_report_path",
        "leakage_preflight": "leakage_preflight_report_path",
        "cv": "cv_metrics_path",
    }
    lines = []
    if not _has_validated_post_cv(metadata):
        lines.append(
            "- 本節只記錄 machine-readable artifact 的 status/path；不填寫 model、R²、"
            "feature importance 或 per-group residual 的解釋性結論。"
        )
    else:
        lines.append("- formal CV 與 post-CV artifacts 均保留 machine-readable status/path。")
    for label, key in rows:
        status = str(statuses.get(key, "not_run"))
        path = metadata.get(path_fields[key], "not_run")
        lines.append(f"- {label}：status={status}；path={path}。")
    health = _as_mapping(metadata.get("feature_health"))
    if health:
        lines.append(
            "- feature health scope："
            f"input={health.get('input_feature_count', 'not_run')} 欄；"
            f"active={health.get('active_feature_count', 'not_run')} 欄；"
            f"zero-variance removed={_sequence_display(health.get('zero_variance_removed_columns'))}。"
        )
    lines.append(
        f"- UMAP：status={statuses.get('umap', 'not_run')}；"
        f"k-means：status={statuses.get('kmeans', 'not_run')}。"
    )
    if _has_validated_post_cv(metadata):
        post = _as_mapping(metadata.get("post_cv_analysis"))
        outputs = _as_mapping(post.get("outputs"))
        post_path_fields = {
            "shrinkage": "shrinkage_analysis_path",
            "orientation_marginal": "orientation_target_marginal_path",
        }
        for key in ("shrinkage", "orientation_marginal"):
            lines.append(
                f"- {key}：status={statuses.get(key, 'not_run')}；"
                f"path={metadata.get(post_path_fields[key], outputs.get(post_path_fields[key], 'not_run'))}。"
            )
    return "\n".join(lines)


def _post_cv_analysis_summary(metadata: Mapping[str, object]) -> str:
    """呈現 validated post-CV 證據、限制與探索性 artifact 路徑。"""
    if not _has_validated_post_cv(metadata):
        return ""
    post = _as_mapping(metadata.get("post_cv_analysis"))
    best = _as_mapping(post.get("best_configuration"))
    orientation = _as_mapping(post.get("orientation"))
    umap = _as_mapping(post.get("umap"))
    outputs = _as_mapping(post.get("outputs"))
    convergence = _as_mapping(post.get("mlpr_convergence"))
    best_is_arm4 = str(best.get("configuration_id", "")).startswith(
        "rui_48_plus_nucleus__"
    )
    lines = [
        "## Post-CV diagnostics（探索性；不影響 formal model）",
        "",
        "### Best configuration 與 shrinkage",
        "",
        f"- configuration：{best.get('configuration_id', 'not_run')}；"
        f"mean fold Rui R²={_format_number(best.get('mean_rui_r2'))}；"
        f"cell-level R²={_format_number(best.get('mean_cell_r2'))}。",
        f"- shrinkage OLS：slope={_format_number(best.get('slope'))}；"
        f"intercept={_format_number(best.get('intercept'))}；"
        f"observed group-mean range="
        f"[{_format_number(best.get('observed_min'))}, {_format_number(best.get('observed_max'))}]；"
        f"predicted range="
        f"[{_format_number(best.get('predicted_min'))}, {_format_number(best.get('predicted_max'))}]。",
    ]
    if best_is_arm4:
        lines.append(_post_cv_arm4_boundary_line(metadata))
    lines.extend(
        [
        "- 模型只還原了約 19% 的組間差異。",
        f"- observed target range={_format_number(best.get('observed_range'))}；"
        "與密集相鄰組別間距及 background quantization 一起看，"
        "單一 R² 不足以代表組間還原程度；強烈 regression-to-the-mean 仍存在。",
        f"- MLPR convergence boundary：max_iter={convergence.get('max_iter', 200)}；"
        f"status={convergence.get('status', 'not_converged_at_anchor_limit')}；"
        "formal execution observed ConvergenceWarning，參數未調整。",
        "",
        "### Orientation marginal（描述性、非因果）",
        "",
        f"- 12 bins；cell Spearman r={_format_number(orientation.get('cell_spearman_r'))}；"
        f"group-median Spearman r={_format_number(orientation.get('group_spearman_r'))}；"
        f"target median range={_format_number(orientation.get('bin_target_median_range'))}；"
        f"target mean range={_format_number(orientation.get('bin_target_mean_range'))}。",
        f"- 12-bin target medians：{_sequence_display(orientation.get('bin_target_medians'))}。",
        f"- RFR mean rank（Orientation）：geometry_24={_format_number(orientation.get('geometry_24_mean_rfr_rank'))}；"
        f"rui_filtered={_format_number(orientation.get('rui_filtered_mean_rfr_rank'))}；"
        f"rui_48_plus_nucleus={_format_number(orientation.get('rui_48_plus_nucleus_mean_rfr_rank'))}。",
        "- Evidence-bounded conclusion：no meaningful univariate marginal directionality detected；"
        "flat 12-bin medians and near-zero cell rho，故 RFR rank is more consistent with "
        "fit noise and/or multivariate interaction, not evidence of true directionality；"
        "no causal/biological inference。",
        "",
        "### UMAP / putative-IFN k-means（探索性 only）",
        "",
        f"- population={umap.get('cell_count', 19648)} cells、"
        f"{umap.get('fov_count', 693)} FOV；X-only feature count={umap.get('feature_count', 30)}；"
        f"putative IFN selector=`ifn_dose > 0`，selected={umap.get('ifn_selected_cell_count', 'not_run')} cells；"
        f"cluster counts={_mapping_display(umap.get('cluster_counts'))}。",
        f"- UMAP parameters：n_components=2、random_state={_format_number(_as_mapping(umap.get('umap_requested_params')).get('random_state', 42))}；"
        f"KMeans：n_clusters={_format_number(_as_mapping(umap.get('kmeans_actual_params')).get('n_clusters', 2))}、"
        f"n_init={_format_number(_as_mapping(umap.get('kmeans_actual_params')).get('n_init', 10))}、"
        f"random_state={_format_number(_as_mapping(umap.get('kmeans_actual_params')).get('random_state', 42))}。",
        "- UMAP 先以 full population 的 StandardScaler-transformed 30 欄 fit；"
        "k-means 只以 UMAP1/UMAP2 fit，target 僅作 post-hoc cluster naming，"
        "不影響 formal model。",
        f"- artifacts：embeddings={outputs.get('umap_embeddings_path', metadata.get('umap_embeddings_path', 'not_run'))}；"
        f"cluster features={outputs.get('kmeans_cluster_features_path', metadata.get('kmeans_cluster_features_path', 'not_run'))}。",
        f"- {umap.get('condition_warning', CONDITION_WARNING_TEXT)}",
        "- condition plot 僅作 visualization；不得由 condition 或 putative IFN labels 推導劑量結論。",
        "- condition/putative-IFN mapping is unconfirmed and untrusted; "
        "visualization only; no dose conclusions are allowed。",
        ]
    )
    figure_paths = _as_mapping(umap.get("figure_paths"))
    for key in ("umap_by_donor.png", "umap_by_passage.png", "umap_by_condition.png"):
        if key in figure_paths:
            lines.append(f"- {key}：{figure_paths[key]}。")
    return "\n".join(lines)


def _post_cv_arm4_boundary_line(metadata: Mapping[str, object]) -> str:
    """緊鄰 best Arm 4 結果呈現 phase-derived proxy 的非對稱邊界。"""
    return (
        "- Arm 4 boundary：Arm 4 tests whether our phase-derived nucleus proxy adds "
        "signal, not whether true nucleus features are useful; if it loses, cannot "
        "distinguish biology from a failed proxy。"
    )


def _has_validated_post_cv(metadata: Mapping[str, object]) -> bool:
    """判斷 post-CV 是否已完成整包 validated 發布。"""
    post = _as_mapping(metadata.get("post_cv_analysis"))
    statuses = _as_mapping(metadata.get("analysis_status"))
    return str(post.get("status", "")) == "validated" and all(
        str(statuses.get(key, "not_run")) == "validated"
        for key in ("shrinkage", "orientation_marginal", "umap", "kmeans")
    )


def _format_number(value: object) -> str:
    """以穩定 compact 格式呈現 numeric metadata。"""
    if value is None:
        return "not_run"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nonfinite"
    return f"{number:.12g}"


def _mapping_display(value: object) -> str:
    """以穩定順序呈現簡單 machine-readable mapping。"""
    mapping = _as_mapping(value)
    if not mapping:
        return "not_run"
    return "{" + ", ".join(
        f"{key}={_format_number(item)}" for key, item in sorted(mapping.items())
    ) + "}"


def _as_mapping(value: object) -> Mapping[str, object]:
    """將 metadata nested value 安全視為 mapping。"""
    return value if isinstance(value, Mapping) else {}


def _string_sequence(value: object) -> tuple[str, ...]:
    """將 metadata sequence 轉為穩定字串 tuple。"""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


if __name__ == "__main__":  # pragma: no cover - CLI seam
    raise SystemExit(_refresh_cli())
