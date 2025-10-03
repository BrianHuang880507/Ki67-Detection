#!/usr/bin/env python
"""Compare DF vs LT fluorescence metrics.

Usage
-----
python analysis/compare_fluor.py --input data/output/results/20251003-LC3/20251003-LC3_cleaned.csv

The script will print a similarity table and (optionally) save per-pair stats to CSV.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def list_metric_pairs(columns: Iterable[str]) -> list[str]:
    bases: set[str] = set()
    for col in columns:
        if col.endswith("_x"):
            bases.add(col[:-2])
        elif col.endswith("_y"):
            bases.add(col[:-2])
    return sorted(bases)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="DF/LT 螢光參數相似度比對")
    parser.add_argument("--input", required=True, help="輸入 cleaned CSV 路徑")
    parser.add_argument(
        "--output", help="(選用) 將每個指標的比較結果輸出成 CSV", default=None
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    pairs = list_metric_pairs(df.columns)
    if not pairs:
        raise SystemExit("找不到 *_x / *_y 欄位，請確認輸入檔案是否包含 DF/LT 資料。")

    rows = []
    for base in pairs:
        col_x = f"{base}_x"
        col_y = f"{base}_y"
        if col_x not in df.columns or col_y not in df.columns:
            continue

        sub = df[[col_x, col_y]].dropna()
        if sub.empty:
            continue
        x = sub[col_x].to_numpy()
        y = sub[col_y].to_numpy()

        pearson = float(sub[col_x].corr(sub[col_y], method="pearson"))
        spearman = float(sub[col_x].corr(sub[col_y], method="spearman"))
        mae = float(np.mean(np.abs(x - y)))
        mape = float(np.mean(np.abs((x - y) / x))) if np.all(x != 0) else math.nan
        cos_sim = cosine_similarity(x, y)
        rows.append(
            {
                "metric": base,
                "pearson": pearson,
                "spearman": spearman,
                "cosine": cos_sim,
                "mae": mae,
                "mape": mape,
            }
        )

    if not rows:
        raise SystemExit("所有 *_x / *_y 欄位都沒有可比較的有效資料。")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("pearson", ascending=False)

    print("=== DF / LT 螢光相似度指標 ===")
    print(result_df.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
        print(f"\n[資訊] 已輸出比較結果：{out_path}")


if __name__ == "__main__":
    main()
