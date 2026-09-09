"""Exp8 入口：刺激濃度、IDO 亮暗與 donor 之間的細胞形狀差異。

用法（在專案根目錄）：

    conda run --no-capture-output -n ki67dtc python -m immunity.exp8.run_experiment

只讀取 Exp3／Exp4 既有 artifacts，輸出寫進 `immunity/outputs/exp8/`。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from ..exp6.dataset import InputPaths, build_fov_table, describe_dataset, load_cell_table
from ..exp6.figures import configure_fonts
from ..exp6.gallery import GalleryConfig
from . import figures as fig
from .brightness import (
    BRIGHT_LABEL,
    DARK_LABEL,
    PRIMARY_CONDITION,
    control_thresholds,
    images_with_both_groups,
    label_brightness,
    paired_shape_contrast,
    positive_fraction,
)
from .donor_response import delta_matrix, donor_spread, overall_magnitude, shape_delta
from .reporting import render_report
from .notable import (
    build_thresholds,
    cell_caption,
    combined_ranking,
    notable_fraction,
    row_label,
    select_notable_cells,
    select_typical_cells,
)
from .selection import fixed_window_span, paired_cells_within_image, render_selection
from .shape import (
    DENSITY_COLUMN,
    correlate_shape_with_dose,
    shape_columns,
    shape_dose_response,
)


HIGH_DOSE_CONDITION = "IFN100_TNF0"


def _timestamp() -> str:
    """回傳含時區的 ISO 8601 時間。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RunLog:
    """同時寫到 stdout 與 run.log 的簡易紀錄器。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        self.lines.append(line)
        print(line, flush=True)

    def flush(self) -> None:
        """把累積的紀錄寫檔。"""
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="Exp8：形狀、IDO 亮暗與 donor 差異")
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--primary-condition",
        default=PRIMARY_CONDITION,
        help=f"亮暗比較的主力條件（預設 {PRIMARY_CONDITION}，亮暗人數最平衡）",
    )
    parser.add_argument("--cells-per-dose", type=int, default=6, help="每一列放幾顆細胞")
    parser.add_argument(
        "--notable-features",
        type=int,
        default=5,
        help="用兩條劑量軸合併排名的前幾個形狀特徵當篩選條件（預設 5）",
    )
    parser.add_argument(
        "--notable-percentile",
        type=float,
        default=90.0,
        help="形狀「特別」的門檻取對照組的第幾百分位（預設 90）",
    )
    parser.add_argument("--paired-images", type=int, default=10, help="亮暗配對影像庫用幾張影像")
    parser.add_argument("--detail-features", type=int, default=3, help="劑量曲線畫幾個形狀特徵")
    parser.add_argument("--skip-images", action="store_true", help="不讀原始影像，跳過兩張影像庫")
    return parser.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path, log: RunLog) -> None:
    """輸出 CSV 並記錄列數。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"寫出 {path.name}（{len(frame)} 列）")


def main(argv: list[str] | None = None) -> int:
    """執行完整 Exp8 流程。"""
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    out_root = (args.output_root or data_root / "immunity/outputs/exp8").resolve()
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log = RunLog(out_root / "run.log")
    started = perf_counter()
    log(f"Exp8 開始；data_root={data_root}")
    log(f"圖表字型：{configure_fonts()}")

    paths = InputPaths.under(data_root)
    cells = load_cell_table(paths)
    fov = build_fov_table(cells)
    summary = describe_dataset(cells, fov)
    features = shape_columns(fov)
    log(f"資料：{summary['n_cells']} 顆細胞、{summary['n_images']} 張影像、{len(features)} 個形狀特徵")

    thresholds = control_thresholds(cells)
    log(thresholds.describe())
    labelled_cells = label_brightness(cells, thresholds)
    counts = labelled_cells["ido_class"].value_counts()
    log(
        f"亮暗分組：{BRIGHT_LABEL} {int(counts.get(BRIGHT_LABEL, 0)):,} 顆、"
        f"{DARK_LABEL} {int(counts.get(DARK_LABEL, 0)):,} 顆、"
        f"灰帶 {int(counts.get('灰帶', 0)):,} 顆"
    )

    figure_paths: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}

    figure_paths["fig01"] = fig.plot_ido_dose_response(fov, figures_dir / "fig01_ido_dose.png")

    positives = positive_fraction(labelled_cells, thresholds)
    tables["ido_positive_fraction"] = positives
    figure_paths["fig02"] = fig.plot_positive_fraction(
        positives, figures_dir / "fig02_ido_positive_fraction.png"
    )
    for row in positives.itertuples():
        log(f"IDO 陽性比例 {row.condition}：{row.positive_fraction:.1%}（n={row.n_cells:,} 顆）")

    ifn_slice = fov[fov["tnf_dose"] == 0]
    tnf_slice = fov[fov["ifn_dose"].isin([0.0, 25.0])]
    ifn_table = correlate_shape_with_dose(ifn_slice, "ifn_dose", (DENSITY_COLUMN,))
    tnf_table = correlate_shape_with_dose(tnf_slice, "tnf_dose", (DENSITY_COLUMN, "ifn_dose"))
    tables["shape_ifn_correlations"] = ifn_table
    tables["shape_tnf_correlations"] = tnf_table
    figure_paths["fig03"] = fig.plot_shape_dose_correlation(
        ifn_table, figures_dir / "fig03_shape_ifn.png", "細胞形狀與 IFN-γ 濃度的關聯性"
    )
    figure_paths["fig04"] = fig.plot_shape_dose_correlation(
        tnf_table, figures_dir / "fig04_shape_tnf.png", "細胞形狀與 TNF-α 濃度的關聯性"
    )
    log(
        f"IFN 軸密度校正後平均保留 {ifn_table['retained_fraction'].median():.0%}；"
        f"TNF 軸 {tnf_table['retained_fraction'].median():.0%}"
    )

    top = ifn_table.nsmallest(args.detail_features, "rank")
    top_features = top["feature"].tolist()
    top_labels = top["feature_label"].tolist()
    tables["shape_dose_response"] = shape_dose_response(fov, features)
    figure_paths["fig05"] = fig.plot_shape_dose_response(
        fov, top_features, top_labels, figures_dir / "fig05_shape_dose_response.png"
    )

    contrast, contrast_detail = paired_shape_contrast(
        labelled_cells, features, condition=args.primary_condition
    )
    tables["bright_dim_shape"] = contrast
    tables["bright_dim_shape_per_image"] = contrast_detail
    figure_paths["fig07"] = fig.plot_bright_dim_shape(
        contrast, figures_dir / "fig07_bright_dim_shape.png"
    )
    best = contrast.nsmallest(1, "rank").iloc[0]
    log(
        f"亮暗形狀差異（{args.primary_condition}，{int(best['n_images'])} 張影像配對）："
        f"最大是 {best['feature_label']} 效果量 {best['effect_median']:+.3f}、"
        f"{int(best['images_same_sign'])}/{int(best['n_images'])} 張同號"
    )

    distribution_features = contrast.nsmallest(4, "rank")
    figure_paths["fig08"] = fig.plot_bright_dim_distribution(
        labelled_cells[labelled_cells["condition"] == args.primary_condition],
        distribution_features["feature"].tolist(),
        distribution_features["feature_label"].tolist(),
        figures_dir / "fig08_bright_dim_distribution.png",
    )

    delta = shape_delta(fov, features)
    tables["shape_delta"] = delta
    matrix = delta_matrix(delta, axis="ifn")
    figure_paths["fig10"] = fig.plot_donor_delta_heatmap(
        matrix, figures_dir / "fig10_donor_delta.png"
    )
    slope_features = ifn_table.nsmallest(4, "rank")
    figure_paths["fig11"] = fig.plot_donor_dose_slopes(
        delta,
        slope_features["feature"].tolist(),
        slope_features["feature_label"].tolist(),
        figures_dir / "fig11_donor_slopes.png",
    )
    magnitude = overall_magnitude(delta, condition=HIGH_DOSE_CONDITION)
    tables["donor_magnitude"] = magnitude
    tables["donor_spread"] = donor_spread(delta, axis="ifn")
    figure_paths["fig12"] = fig.plot_donor_magnitude(
        magnitude, figures_dir / "fig12_donor_magnitude.png"
    )
    for row in magnitude.drop_duplicates("b_id").itertuples():
        log(
            f"{HIGH_DOSE_CONDITION} 整體形狀變化幅度 {row.b_id}："
            f"中位數 {row.median:.3f}（{row.min:.3f}–{row.max:.3f}，n={int(row.n_passages)} 個 passage）"
        )

    ranking = combined_ranking(ifn_table, tnf_table, top_n=args.notable_features)
    shape_thresholds = build_thresholds(cells, ranking, percentile=args.notable_percentile)
    tables["notable_feature_ranking"] = ranking
    fractions = notable_fraction(cells, shape_thresholds)
    tables["notable_fraction"] = fractions
    for threshold in shape_thresholds:
        log(f"形狀門檻　{threshold.describe()}")
        block = fractions[fractions["feature"] == threshold.feature]
        low = block[block["condition"] == "IFN0_TNF0"]["notable_fraction"].iloc[0]
        high = block[block["condition"] == HIGH_DOSE_CONDITION]["notable_fraction"].iloc[0]
        log(f"　　達標比例　IFN0_TNF0 {low:.1%} → {HIGH_DOSE_CONDITION} {high:.1%}")
    figure_paths["fig13"] = fig.plot_notable_fraction(
        fractions, figures_dir / "fig13_notable_fraction.png", "形狀特別的細胞比例"
    )

    if not args.skip_images:
        blocks: list[tuple[object, pd.DataFrame]] = [
            (None, select_typical_cells(cells, shape_thresholds, count=args.cells_per_dose))
        ]
        blocks += [
            (threshold, select_notable_cells(cells, threshold, count=args.cells_per_dose))
            for threshold in shape_thresholds
        ]
        notable_selection = pd.concat([block for _, block in blocks], ignore_index=True)
        # 這張圖同時放典型細胞與長軸特別長的細胞，若用 P95 當視窗，典型細胞
        # 會縮成小點看不清楚。取 P65 讓多數細胞填滿版面，少數最長的略微出界。
        notable_span = fixed_window_span(notable_selection, quantile=0.65, margin=1.2)
        rendered, _ = render_selection(
            data_root,
            notable_selection,
            # tile 邊長對齊裁切視窗，避免降採樣把細長的細胞糊掉。
            GalleryConfig(
                tile_pixels=notable_span, fixed_span=notable_span, subtract_background=True
            ),
        )
        log(f"形狀影像庫固定裁切視窗 {notable_span} 像素，圖上相對大小為真實比例")

        rows = []
        for threshold, block in blocks:
            images = [
                rendered[(str(row.image_key), int(row.cell_label))] for row in block.itertuples()
            ]
            captions = [cell_caption(row) for _, row in block.iterrows()]
            rows.append((row_label(threshold), images, captions))
        figure_paths["fig06"] = fig.plot_notable_cells(
            rows, figures_dir / "fig06_notable_cells.png", "形狀特別的細胞"
        )
        tables["notable_cells"] = notable_selection[
            [
                "image_key",
                "cell_label",
                "b_id",
                "passage",
                "condition",
                "ifn_dose",
                "tnf_dose",
                "notable_feature",
                "notable_label",
                "notable_value",
                "IDO_score_ff",
            ]
        ]

        candidates = images_with_both_groups(
            labelled_cells, condition=args.primary_condition, min_per_group=3
        )
        image_keys = candidates["image_key"].head(args.paired_images).tolist()
        paired = paired_cells_within_image(labelled_cells, top_features[:2], image_keys)
        paired_span = fixed_window_span(paired)
        paired_rendered, _ = render_selection(
            data_root,
            paired,
            GalleryConfig(tile_pixels=180, fixed_span=paired_span, subtract_background=True),
        )
        log(f"亮暗配對固定裁切視窗 {paired_span} 像素")
        pairs = []
        for image_key in image_keys:
            block = paired[paired["pair_image"] == image_key]
            bright = block[block["pair_group"] == BRIGHT_LABEL]
            dim = block[block["pair_group"] == DARK_LABEL]
            if bright.empty or dim.empty:
                continue
            pairs.append(
                (
                    image_key,
                    paired_rendered[(image_key, int(bright["cell_label"].iloc[0]))],
                    paired_rendered[(image_key, int(dim["cell_label"].iloc[0]))],
                )
            )
        figure_paths["fig09"] = fig.plot_paired_cells(
            pairs, figures_dir / "fig09_paired_cells.png", "同一張影像中的 IDO 亮細胞與暗細胞"
        )
        tables["paired_cells"] = paired[
            ["image_key", "cell_label", "b_id", "passage", "pair_group", "IDO_score_ff"]
        ]
        log(f"影像庫完成：濃度列 {len(rows)} 列、亮暗配對 {len(pairs)} 組")
    else:
        log("已略過影像庫（--skip-images）")

    for name, frame in tables.items():
        _write_csv(frame, out_root / f"{name}.csv", log)

    elapsed = perf_counter() - started
    metadata = {
        "generated_at": _timestamp(),
        "elapsed_seconds": round(elapsed, 2),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "data_root": str(data_root),
        "output_root": str(out_root),
        "dataset": summary,
        "shape_features": features,
        "primary_condition": args.primary_condition,
        "thresholds": {
            "dark_max": thresholds.dark_max,
            "bright_min": thresholds.bright_min,
            "control_cells": thresholds.control_cells,
            "control_median": thresholds.control_median,
        },
        "notable_percentile": args.notable_percentile,
        "notable_thresholds": [
            {
                "feature": item.feature,
                "feature_label": item.feature_label,
                "direction": item.direction,
                "threshold": item.threshold,
                "artifact_cutoff": item.artifact_cutoff,
                "control_median": item.control_median,
            }
            for item in shape_thresholds
        ],
        "figures": {key: value.name for key, value in sorted(figure_paths.items())},
    }
    (out_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("寫出 run_metadata.json")

    (out_root / "REPORT.md").write_text(
        render_report(
            summary=summary,
            thresholds=thresholds,
            tables=tables,
            metadata=metadata,
            primary_condition=args.primary_condition,
            high_dose_condition=HIGH_DOSE_CONDITION,
        ),
        encoding="utf-8",
    )
    log("寫出 REPORT.md")
    log(f"Exp8 完成，耗時 {elapsed:.1f} 秒")
    log.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
