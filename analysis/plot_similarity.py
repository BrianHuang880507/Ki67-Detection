#!/usr/bin/env python
"""Plot DF vs LT similarity metrics as line charts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import math
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rcParams

_METRIC_PATTERN = re.compile(r"^(?P<prefix>IntDen|RawIntDen)[-_]?(?P<index>\d+)$")
_BASE_COLUMNS = ["metric", "pearson", "spearman", "cosine", "mae", "mape"]
_IMAGE_SUFFIX = "png"
_PRIORITY = {"IntDen": 0, "RawIntDen": 1}


def _configure_font(preferred: str | None = None) -> str | None:
    """Configure Matplotlib font, preferring CJK-capable options."""
    candidates = [preferred] if preferred else []
    candidates.extend(
        [
            "Microsoft JhengHei",
            "Microsoft YaHei",
            "SimHei",
            "PingFang TC",
            "PingFang HK",
            "PingFang SC",
            "Noto Sans CJK TC",
            "Noto Sans CJK JP",
            "Noto Sans CJK SC",
            "Source Han Sans TW",
        ]
    )
    for name in candidates:
        if not name:
            continue
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        rcParams["font.family"] = name
        rcParams["axes.unicode_minus"] = False
        return name
    return None


def _parse_metric_name(name: str) -> Tuple[str | None, int | None]:
    match = _METRIC_PATTERN.match(name)
    if not match:
        return None, None
    prefix = match.group("prefix")
    index = int(match.group("index"))
    return prefix, index


def _prepare_long_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    df = df.dropna(subset=["metric"])
    df["metric"] = df["metric"].astype(str).str.strip()
    df = df[df["metric"] != ""]
    if df.empty:
        return {}

    for col in _BASE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    prefixes = []
    indices = []
    for value in df["metric"]:
        prefix, idx = _parse_metric_name(value)
        if prefix is None:
            prefix = "Other"
        prefixes.append(prefix)
        indices.append(math.inf if idx is None else idx)
    df["__prefix"] = prefixes
    df["__index"] = indices

    groups: Dict[str, pd.DataFrame] = {}
    for prefix in df["__prefix"].unique():
        sub = df[df["__prefix"] == prefix].copy()
        if prefix in {"IntDen", "RawIntDen"}:
            sub = sub.sort_values(["__index", "metric"], kind="mergesort").reset_index(drop=True)
            sub["metric"] = [f"{prefix}-{i}" for i in range(1, len(sub) + 1)]
        else:
            sub = sub.sort_values("metric", kind="mergesort").reset_index(drop=True)
        sub = sub.drop(columns=["__prefix", "__index"])
        groups[prefix] = sub
    return groups


def _split_wide_frame(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups: Dict[str, pd.DataFrame] = {}
    suffix_idx = 0
    while True:
        suffix = "" if suffix_idx == 0 else f".{suffix_idx}"
        columns = [f"{col}{suffix}" for col in _BASE_COLUMNS]
        if not all(col in df.columns for col in columns):
            break
        sub = df[columns].copy()
        sub.columns = _BASE_COLUMNS
        sub_groups = _prepare_long_groups(sub)
        for key, value in sub_groups.items():
            if key in groups:
                groups[key] = pd.concat([groups[key], value], ignore_index=True)
            else:
                groups[key] = value
        suffix_idx += 1
    return groups


def _load_groups(csv_path: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    # Treat as long-form only when there are exactly the base columns with unique names
    if df.columns.is_unique and set(_BASE_COLUMNS) == set(df.columns) and len(df.columns) == len(_BASE_COLUMNS):
        return _prepare_long_groups(df[_BASE_COLUMNS])

    return _split_wide_frame(df)


def _plot_group(
    df: pd.DataFrame,
    metrics: list[str],
    group_label: str,
    output_dir: Path | None,
    should_show: bool,
) -> None:
    if df.empty:
        return

    x = range(len(df))
    plt.figure(figsize=(10, 5))
    for col in metrics:
        if col not in df.columns:
            continue
        plt.plot(x, df[col], marker="o", label=col)

    plt.xticks(x, df["metric"], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Similarity")
    plt.title(f"DF vs LT similarity - {group_label}")
    if metrics:
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        outfile = output_dir / f"similarity_{group_label}.{_IMAGE_SUFFIX}"
        plt.savefig(outfile, dpi=300)
        print(f"[INFO] Saved plot: {outfile}")

    if should_show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DF/LT similarity metrics")
    parser.add_argument("--input", required=True, help="CSV produced by compare_fluor")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["pearson", "spearman", "cosine"],
        help="Similarity columns to plot (default: pearson spearman cosine)",
    )
    parser.add_argument("--font", help="Preferred font name (optional)")
    parser.add_argument(
        "--output",
        help="Directory for exported plots (files named similarity_<group>.png)",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    font_name = _configure_font(args.font)
    if font_name:
        print(f"[INFO] Using font: {font_name}")
    else:
        print("[WARN] No preferred font found; using Matplotlib default.")

    metrics = [m for m in args.metrics if m in _BASE_COLUMNS[1:]]
    if not metrics:
        raise SystemExit("No valid similarity columns were requested.")

    groups = _load_groups(csv_path)
    if not groups:
        raise SystemExit("No metric groups available for plotting.")

    ordered_labels = sorted(
        groups.keys(),
        key=lambda label: (_PRIORITY.get(label, 99), label.lower()),
    )

    output_dir = Path(args.output) if args.output else None
    should_show = args.show or (output_dir is None)

    for label in ordered_labels:
        _plot_group(groups[label], metrics, label, output_dir, should_show)


if __name__ == "__main__":
    main()
