#!/usr/bin/env python
"""Compare DF vs LT fluorescence metrics.

Usage
-----
python analysis/compare_fluor.py --input data/output/results/20251003-LC3/20251003-LC3_cleaned.csv

The script prints similarity statistics and (optionally) writes a CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_METRIC_PATTERN = re.compile(r"^(?P<prefix>IntDen|RawIntDen)[-_]?(?P<index>\d+)$")
_DISPLAY_COLUMNS = ["metric", "pearson", "spearman", "cosine", "mae", "mape"]


def list_metric_pairs(columns: Iterable[str]) -> list[str]:
    bases: set[str] = set()
    for col in columns:
        if col.endswith("_x"):
            bases.add(col[:-2])
        elif col.endswith("_y"):
            bases.add(col[:-2])
    return sorted(bases)


def parse_metric_name(name: str) -> tuple[str | None, int | None]:
    match = _METRIC_PATTERN.match(name)
    if not match:
        return None, None
    prefix = match.group("prefix")
    index = int(match.group("index"))
    return prefix, index


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def sort_and_relabel(rows: list[dict[str, object]], prefix: str) -> list[dict[str, object]]:
    def sort_key(row: dict[str, object]) -> tuple[float, str]:
        idx = row.get("__index")
        if isinstance(idx, int):
            return float(idx), str(row.get("metric", ""))
        return math.inf, str(row.get("metric", ""))

    rows = sorted(rows, key=sort_key)
    for i, row in enumerate(rows, start=1):
        row["metric"] = f"{prefix}-{i}"
        row.pop("__index", None)
        row.pop("__prefix", None)
    return rows


def cleanup_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cleaned: list[dict[str, object]] = []
    for row in rows:
        cleaned_row = {name: row.get(name) for name in _DISPLAY_COLUMNS}
        cleaned.append(cleaned_row)
    return cleaned


def write_wide_csv(out_path: Path, groups: list[list[dict[str, object]]]) -> None:
    if not groups:
        return

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header: list[str] = []
        for _ in groups:
            header.extend(_DISPLAY_COLUMNS)
        writer.writerow(header)

        max_len = max(len(group) for group in groups)
        for idx in range(max_len):
            row: list[str] = []
            for group in groups:
                if idx < len(group):
                    row.extend(format_value(group[idx].get(col)) for col in _DISPLAY_COLUMNS)
                else:
                    row.extend([""] * len(_DISPLAY_COLUMNS))
            writer.writerow(row)


def print_table(title: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=_DISPLAY_COLUMNS)
    if df.empty:
        return
    print(f"\n=== {title} ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DF/LT similarity metrics")
    parser.add_argument("--input", required=True, help="Path to the cleaned CSV file")
    parser.add_argument(
        "--output",
        help="Optional path for the summary CSV (wide format)",
        default=None,
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    pairs = list_metric_pairs(df.columns)
    if not pairs:
        raise SystemExit("No *_x / *_y columns were found. Check the input file.")

    int_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    other_rows: list[dict[str, object]] = []

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

        prefix, index = parse_metric_name(base)
        row = {
            "metric": base,
            "pearson": pearson,
            "spearman": spearman,
            "cosine": cos_sim,
            "mae": mae,
            "mape": mape,
            "__prefix": prefix,
            "__index": index,
        }

        if prefix == "IntDen":
            int_rows.append(row)
        elif prefix == "RawIntDen":
            raw_rows.append(row)
        else:
            other_rows.append(row)

    if not (int_rows or raw_rows or other_rows):
        raise SystemExit("No metric pairs could be compared.")

    if int_rows:
        int_rows = sort_and_relabel(int_rows, "IntDen")
    if raw_rows:
        raw_rows = sort_and_relabel(raw_rows, "RawIntDen")
    if other_rows:
        for row in other_rows:
            row.pop("__prefix", None)
            row.pop("__index", None)
        other_rows = sorted(other_rows, key=lambda r: str(r.get("metric", "")))

    print_table("IntDen similarity", cleanup_rows(int_rows))
    print_table("RawIntDen similarity", cleanup_rows(raw_rows))
    print_table("Other metric similarity", cleanup_rows(other_rows))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        groups: list[list[dict[str, object]]] = []
        if int_rows:
            groups.append(cleanup_rows(int_rows))
        if raw_rows:
            groups.append(cleanup_rows(raw_rows))
        if other_rows:
            groups.append(cleanup_rows(other_rows))
        write_wide_csv(out_path, groups)
        print(f"\n[INFO] Saved summary CSV to: {out_path}")


if __name__ == "__main__":
    main()
