"""Exp6 入口：劑量關聯分析 ＋ IDO 亮／暗細胞影像庫。

用法（在專案根目錄）：

    conda run --no-capture-output -n ki67dtc python -m immunity.exp6.run_experiment

只讀取 Exp3／Exp4 既有 artifacts，所有輸出寫進 `immunity/outputs/exp6/`，
不覆寫任何上游檔案。
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

from .dataset import (
    EXCLUDED_FEATURES,
    InputPaths,
    build_fov_table,
    describe_dataset,
    dose_slices,
    feature_columns,
    load_cell_table,
)
from .dose_association import DOSE_AXES, build_all_tables, top_features
from .figures import (
    configure_fonts,
    plot_features_vs_ido,
    plot_ido_dose_response,
    plot_top_features,
    plot_top_features_combined,
)
from .gallery import GalleryConfig, MATCH_STRATA, build_split
from .reporting import render_report


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
    parser = argparse.ArgumentParser(description="Exp6：劑量關聯與 IDO 亮暗細胞影像庫")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.cwd(),
        help="Exp3／Exp4 產物所在的專案根目錄（預設為目前工作目錄）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="輸出目錄（預設為 <data-root>/immunity/outputs/exp6）",
    )
    parser.add_argument("--top-n", type=int, default=10, help="長條圖取前幾名（預設 10）")
    parser.add_argument(
        "--decile-pct", type=float, default=10.0, help="亮／暗各取上下多少百分比（預設 10）"
    )
    parser.add_argument(
        "--tiles-per-group", type=int, default=30, help="影像庫每組放幾顆細胞（預設 30）"
    )
    parser.add_argument(
        "--skip-gallery", action="store_true", help="只跑第一部分的關聯分析，不讀原始影像"
    )
    parser.add_argument(
        "--skip-cell-crops", action="store_true", help="不輸出逐顆細胞的四格 PNG"
    )
    return parser.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path, log: RunLog) -> None:
    """輸出 CSV 並記錄列數。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"寫出 {path.name}（{len(frame)} 列）")


def main(argv: list[str] | None = None) -> int:
    """執行完整 Exp6 流程。"""
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    out_root = (args.output_root or data_root / "immunity/outputs/exp6").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log = RunLog(out_root / "run.log")
    started = perf_counter()
    log(f"Exp6 開始；data_root={data_root}")

    font = configure_fonts()
    log(f"圖表字型：{font}")

    paths = InputPaths.under(data_root)
    cells = load_cell_table(paths)
    fov = build_fov_table(cells)
    slices = dose_slices(fov)
    summary = describe_dataset(cells, fov)
    log(
        f"資料：{summary['n_cells']} 顆細胞、{summary['n_images']} 張影像、"
        f"{summary['n_appearance_features']} 個外觀特徵"
    )
    log(f"IFN 軸 n={len(slices['ifn'])} 張影像；TNF 軸 n={len(slices['tnf'])} 張影像")

    tables = build_all_tables(cells, fov, slices)
    for name, frame in tables.items():
        _write_csv(frame, out_root / f"{name}.csv", log)
    _write_csv(fov, out_root / "image_level_dataset.csv", log)

    ido_ff = tables["ido_dose_correlations"]
    ido_ff = ido_ff[ido_ff["target"] == "IDO_score_ff"]
    for row in ido_ff.itertuples():
        log(
            f"IDO vs {row.dose_axis}［{row.block}］：rho={row.spearman_rho:+.3f}、"
            f"p={row.p_value:.3g}、n={row.n_images}、"
            f"同號區塊 {row.blocks_same_sign}/{row.n_blocks}、"
            f"中位數 {row.median_at_min:.2f}→{row.median_at_max:.2f}"
        )

    figure_paths: dict[str, Path] = {}
    figure_paths["ido_dose_response"] = plot_ido_dose_response(
        fov, tables["ido_dose_correlations"], figures_dir / "fig01_ido_vs_dose.png"
    )
    for index, axis in enumerate(DOSE_AXES, start=2):
        figure_paths[f"top_{axis.name.lower()}"] = plot_top_features(
            tables["feature_dose_correlations"],
            axis.name,
            axis.title,
            figures_dir / f"fig0{index}_top{args.top_n}_features_{axis.name.lower()}.png",
            top_n=args.top_n,
        )
    figure_paths["top_combined"] = plot_top_features_combined(
        tables["feature_dose_correlations"],
        figures_dir / f"fig04_top{args.top_n}_features_combined.png",
        top_n=args.top_n,
    )
    figure_paths["features_vs_ido"] = plot_features_vs_ido(
        tables["feature_ido_correlations"],
        figures_dir / f"fig05_top{args.top_n}_features_vs_ido.png",
        top_n=args.top_n,
    )
    log(f"第一部分圖表完成（{len(figure_paths)} 張）")

    splits: dict[str, dict[str, object]] = {}
    if not args.skip_gallery:
        config = GalleryConfig(
            decile_pct=args.decile_pct,
            tiles_per_group=args.tiles_per_group,
            seed=0,
        )
        split_specs = (
            (
                "global",
                None,
                "IDO 亮 vs. 不亮的細胞影像（不分刺激條件）",
                f"全部 {summary['n_cells']} 顆細胞取 IDO_score_ff 上／下 {args.decile_pct:.0f}%",
                not args.skip_cell_crops,
            ),
            (
                "matched",
                MATCH_STRATA,
                "同一刺激條件下，IDO 亮 vs. 不亮的細胞影像",
                f"在每個 donor×passage×條件內各取上／下 {args.decile_pct:.0f}%",
                False,
            ),
        )
        for split_name, strata, title, subtitle, write_crops in split_specs:
            log(f"影像庫 {split_name} 開始")
            result = build_split(
                cells,
                data_root,
                out_root,
                config,
                split_name=split_name,
                strata=strata,
                title=title,
                subtitle=subtitle,
                write_crops=write_crops,
            )
            _write_csv(result["contrast"], out_root / f"bright_dim_features_{split_name}.csv", log)
            _write_csv(result["selection"], out_root / f"bright_dim_selected_cells_{split_name}.csv", log)
            log(
                f"影像庫 {split_name}：亮 {result['labelled_counts']['bright']} 顆、"
                f"暗 {result['labelled_counts']['dim']} 顆；"
                f"逐顆 PNG {result['n_crop_files']} 張"
            )
            splits[split_name] = result

        from .figures import plot_bright_dim_contrast

        figure_paths["bright_dim_matched"] = plot_bright_dim_contrast(
            splits["matched"]["contrast"],
            figures_dir / f"fig06_top{args.top_n}_bright_vs_dim_matched.png",
            top_n=args.top_n,
        )
        log("第二部分圖表完成")
    else:
        log("已略過影像庫（--skip-gallery）")

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
        "inputs": {
            "features": str(paths.features),
            "ido": str(paths.ido),
            "manifest": str(paths.manifest),
        },
        "dataset": summary,
        "excluded_features": list(EXCLUDED_FEATURES),
        "top_n": args.top_n,
        "decile_pct": args.decile_pct,
        "gallery": {
            name: {
                "counts": result["labelled_counts"],
                "n_crop_files": result["n_crop_files"],
            }
            for name, result in splits.items()
        },
    }
    (out_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("寫出 run_metadata.json")

    report = render_report(
        summary=summary,
        slices=slices,
        tables=tables,
        splits=splits,
        top_n=args.top_n,
        decile_pct=args.decile_pct,
        metadata=metadata,
    )
    (out_root / "REPORT.md").write_text(report, encoding="utf-8")
    log("寫出 REPORT.md")
    log(f"Exp6 完成，耗時 {elapsed:.1f} 秒")
    log.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
