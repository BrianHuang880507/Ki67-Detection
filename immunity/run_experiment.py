"""執行 B4 p6 morphology → IDO response proxy 完整實驗。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np
import pandas as pd
import sklearn
import yaml

from .build_dataset import (
    PROJECT_ROOT,
    aggregate_image_level,
    build_manifest,
    calculate_dose_response,
    calculate_oof_predicted_auc,
    ensure_outlines,
    extract_cell_level_features,
    load_cell_level_from_cleaned,
    resolve_project_path,
)
from .train_ido_proxy import (
    fit_final_models,
    plot_dose_response,
    plot_observed_vs_predicted,
    run_leave_one_condition_out,
    run_repeated_cross_validation,
    summarize_coefficients,
    summarize_metrics,
)


class _Tee:
    """將 stdout/stderr 同時寫到終端機與 run.log。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        """寫入所有目標 stream。"""
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        """同步刷新所有目標 stream。"""
        for stream in self.streams:
            stream.flush()


def load_config(path: str | Path) -> dict[str, Any]:
    """讀取並驗證 immunity experiment YAML 設定。

    Args:
        path: YAML 設定檔路徑。

    Returns:
        實驗設定字典。

    Raises:
        FileNotFoundError: 找不到設定檔時拋出。
        ValueError: 設定不是 mapping 或缺少必要區塊時拋出。
    """
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到實驗設定：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("實驗設定最外層必須是 YAML mapping。")
    for section in ("dataset", "segmentation", "analysis", "modeling"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"實驗設定缺少 `{section}` mapping。")
    config["_config_path"] = str(config_path)
    return config


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """不依賴 tabulate，建立簡單 Markdown table。"""
    selected = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4g}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    cells: pd.DataFrame,
    images: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    dose_auc: pd.DataFrame,
    predicted_auc_summary: pd.DataFrame,
) -> Path:
    """產生白話且不超出證據邊界的實驗結果報告。"""
    repeated = metrics_summary[
        metrics_summary["validation"].eq("repeated_cv")
    ].copy()
    morphology = repeated[repeated["model"].eq("morphology_ridge")]
    dummy = repeated[repeated["model"].eq("dummy_median")]
    signal_statement = "尚無法判定"
    if not morphology.empty and not dummy.empty:
        morph_mae = float(morphology["mae_median"].iloc[0])
        dummy_mae = float(dummy["mae_median"].iloc[0])
        morph_spearman = float(morphology["spearman_median"].iloc[0])
        if morph_mae < dummy_mae and morph_spearman > 0:
            signal_statement = "目前資料顯示 morphology 具有 IDO proxy 預測訊號"
        else:
            signal_statement = "目前資料尚未顯示穩定優於 Dummy 的 morphology 預測訊號"

    top_features = coefficient_summary[
        coefficient_summary["validation"].eq("repeated_cv")
        & coefficient_summary["model"].eq("morphology_ridge")
    ].head(15)
    morphology_predicted_auc = predicted_auc_summary[
        predicted_auc_summary["validation"].eq("repeated_cv")
        & predicted_auc_summary["model"].eq("morphology_ridge")
    ]
    report = [
        "# B4 p6 Morphology → IDO Response 實驗結果",
        "",
        f"> 結論：{signal_statement}。這是 B4 p6 單一批次內的探索性結果，不能稱為已驗證的免疫抑制能力。",
        "",
        "## 資料與分析單位",
        "",
        f"- 完整 phase／DAPI／IDO triplets：{len(manifest)} 張視野。",
        f"- 通過 segmentation QC 的完整細胞：{len(cells)} 顆。",
        f"- Image-level 模型樣本：{len(images)} 張視野。",
        "- phase 用於細胞外框；DAPI 用於細胞核；IDO 只用於 target。",
        "- Morphology-only predictors 不含 IDO intensity、IFN/TNF 劑量或 Ki67 資訊。",
        "",
        "## Repeated cross-validation",
        "",
        _markdown_table(
            repeated,
            [
                "model",
                "n_images",
                "mae_median",
                "rmse_median",
                "r2_median",
                "spearman_median",
            ],
        ),
        "",
        "MAE 越低越好；R² 與 Spearman 越高越好。主要判讀是 morphology-only Ridge 是否穩定優於 Dummy，而不是只看單次 train fit。",
        "",
        "## Leave-one-condition-out",
        "",
        _markdown_table(
            metrics_summary[
                metrics_summary["validation"].eq("leave_one_condition_out")
            ],
            [
                "model",
                "n_images",
                "mae_median",
                "r2_median",
                "spearman_median",
            ],
        ),
        "",
        "這個測試較嚴格：每次把一種 IFN/TNF 條件整組留到測試集。若隨機分層 CV 表現好、但此處失效，代表模型可能主要辨認已知刺激條件，尚不能推論可泛化能力。",
        "",
        "## Morphology Ridge 主要 coefficients",
        "",
        (
            _markdown_table(
                top_features,
                ["feature", "coefficient_median", "coefficient_iqr"],
            )
            if not top_features.empty
            else "沒有可用 coefficient。"
        ),
        "",
        "Coefficient 正負只表示其他 standardized predictors 固定時的模型方向，不等於生物因果關係。",
        "",
        "## 描述性 IDO dose-response AUC",
        "",
        _markdown_table(
            dose_auc,
            ["curve", "auc_raw", "auc_normalized_by_dose_range"],
        ),
        "",
        "這些值應稱為 IDO dose-response AUC；目前只有一個 lot，因此不作為 machine-learning ground truth，也不能稱為 overall immunosuppressive capacity AUC。",
        "",
        "## Morphology OOF predictions 推導的 AUC",
        "",
        _markdown_table(
            morphology_predicted_auc,
            [
                "curve",
                "observed_auc_raw",
                "predicted_auc_raw_median",
                "absolute_auc_error_raw_median",
            ],
        ),
        "",
        "模型仍以每張影像的 IDO_score 為訓練目標；此表是在完成 out-of-fold 預測後，才依劑量重建曲線並積分。因此 predicted AUC 是模型結果的下游摘要，不是用單一 B4 p6 AUC 反覆訓練。",
        "",
        "## 限制",
        "",
        "- 只有 B4 p6 一個 lot／passage，80 張視野是技術影像，不是 80 個獨立生物樣本。",
        "- IDO 螢光是 immunosuppressive proxy；尚未經 PBMC／T-cell functional assay 驗證。",
        "- JPG 缺少曝光 metadata，結果只適合目前批次內比較。",
        "- IFN×TNF 並非完整全排列，因此只呈現三條現有的一維 dose-response 曲線。",
        "",
        "## 圖表",
        "",
        "- `figures/observed_vs_predicted.png`：repeated-CV 的 out-of-fold 預測。",
        "- `figures/dose_response.png`：三條描述性 IDO dose-response 曲線。",
    ]
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def run_experiment(config: Mapping[str, Any]) -> Path:
    """依設定執行資料整理、cross-validation、模型保存與報告輸出。"""
    dataset_config = config["dataset"]
    segmentation_config = config["segmentation"]
    analysis_config = config["analysis"]
    modeling_config = config["modeling"]
    input_dir = resolve_project_path(dataset_config["input_dir"])
    output_dir = resolve_project_path(analysis_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Experiment config: {config['_config_path']}")
    print(f"[INFO] Input: {input_dir}")
    print(f"[INFO] Output: {output_dir}")

    manifest = build_manifest(
        input_dir,
        expected_triplets=dataset_config.get("expected_triplets"),
        expected_conditions=dataset_config.get("expected_conditions"),
    )
    manifest.to_csv(output_dir / "data_manifest.csv", index=False)
    print(f"[OK] 影像配對完成：{len(manifest)} triplets")

    feature_source = str(
        analysis_config.get("feature_source", "cleaned_csv")
    ).strip().lower()
    if feature_source == "cleaned_csv":
        cells, segmentation_qc = load_cell_level_from_cleaned(
            analysis_config["cleaned_csv"], manifest
        )
        print(f"[OK] 已讀取 main.py cleaned CSV：{analysis_config['cleaned_csv']}")
    elif feature_source == "outlines":
        outline_dir = ensure_outlines(manifest, input_dir, segmentation_config)
        cells, segmentation_qc = extract_cell_level_features(
            manifest,
            outline_dir,
            max_nucleus_outside_fraction=float(
                analysis_config.get("max_nucleus_outside_fraction", 0.05)
            ),
        )
    else:
        raise ValueError(
            "analysis.feature_source 必須是 `cleaned_csv` 或 `outlines`。"
        )
    cells.to_csv(output_dir / "cell_level_features.csv", index=False)
    segmentation_qc.to_csv(output_dir / "segmentation_qc.csv", index=False)
    print(f"[OK] Cell-level morphology/IDO：{len(cells)} cells")

    images, morphology_columns = aggregate_image_level(
        cells,
        manifest,
        min_cells_per_image=int(analysis_config.get("min_cells_per_image", 3)),
    )
    images.to_csv(output_dir / "image_level_dataset.csv", index=False)
    (output_dir / "morphology_predictors.json").write_text(
        json.dumps(morphology_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not np.isfinite(images["IDO_score"]).all() or images["IDO_score"].nunique() < 2:
        raise ValueError("image-level IDO_score 無效或沒有變異，無法訓練模型。")
    print(
        f"[OK] Image-level dataset：{len(images)} images，"
        f"{len(morphology_columns)} morphology predictors"
    )

    dose_summary, dose_auc = calculate_dose_response(images)
    dose_summary.to_csv(output_dir / "dose_response_summary.csv", index=False)
    dose_auc.to_csv(output_dir / "dose_response_auc.csv", index=False)

    repeated_predictions, repeated_metrics, repeated_coefficients = (
        run_repeated_cross_validation(images, morphology_columns, modeling_config)
    )
    loco_predictions, loco_metrics, loco_coefficients = run_leave_one_condition_out(
        images, morphology_columns, modeling_config
    )
    predictions = pd.concat(
        [repeated_predictions, loco_predictions], ignore_index=True
    )
    metrics = pd.concat([repeated_metrics, loco_metrics], ignore_index=True)
    coefficients = pd.concat(
        [repeated_coefficients, loco_coefficients], ignore_index=True
    )
    metrics_summary = summarize_metrics(metrics)
    coefficient_summary = summarize_coefficients(coefficients)
    predicted_auc, predicted_auc_summary = calculate_oof_predicted_auc(
        predictions, dose_auc
    )

    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    metrics_summary.to_csv(output_dir / "cv_metrics_summary.csv", index=False)
    coefficients.to_csv(output_dir / "fold_coefficients.csv", index=False)
    coefficient_summary.to_csv(output_dir / "feature_coefficients.csv", index=False)
    predicted_auc.to_csv(output_dir / "predicted_dose_response_auc.csv", index=False)
    predicted_auc_summary.to_csv(
        output_dir / "predicted_dose_response_auc_summary.csv", index=False
    )
    print("[OK] Repeated CV 與 leave-one-condition-out 完成")

    final_models = fit_final_models(
        images,
        morphology_columns,
        modeling_config,
        output_dir / "models",
    )
    final_models.to_csv(output_dir / "final_models.csv", index=False)
    plot_observed_vs_predicted(
        predictions, figures_dir / "observed_vs_predicted.png"
    )
    plot_dose_response(dose_summary, figures_dir / "dose_response.png")

    report_path = write_report(
        output_dir,
        manifest,
        cells,
        images,
        metrics_summary,
        coefficient_summary,
        dose_auc,
        predicted_auc_summary,
    )
    metadata = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config["_config_path"],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "triplets": int(len(manifest)),
        "cells": int(len(cells)),
        "images": int(len(images)),
        "target": "image-level background-corrected IDO_score",
        "scope": "B4 p6 within-lot exploratory model",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] 實驗報告：{report_path}")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment config")
    return parser


def main() -> int:
    """命令列入口。"""
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_dir = resolve_project_path(config["analysis"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)
    log_path = output_dir / "run.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        stdout = _Tee(sys.stdout, log_handle)
        stderr = _Tee(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print("\n" + "=" * 72)
            print(f"[START] {datetime.now(timezone.utc).isoformat()}")
            run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
