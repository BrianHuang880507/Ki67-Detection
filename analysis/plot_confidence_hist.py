#!/usr/bin/env python3
"""Plot confidence histogram (percentage vs. cell counts) from prediction CSV files."""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_prediction_csv(pred_dir: Path, dataset_stem: str) -> Path | None:
    """Return the most recent prediction CSV whose name contains dataset_stem."""
    candidates = sorted(
        pred_dir.glob(f"*{dataset_stem}*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_probabilities(pred_path: Path) -> pd.Series:
    """Load the probability column from prediction CSV."""
    df = pd.read_csv(pred_path, usecols=["probability"])
    return df["probability"]


def build_histogram(
    probs: pd.Series,
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return histogram counts and bin edges for probabilities."""
    bins = np.arange(0.0, 1.0 + bin_size, bin_size)
    counts, edges = np.histogram(probs, bins=bins)
    return counts, edges


def format_bin_labels(edges: np.ndarray) -> List[str]:
    """Convert bin edges (0-1) into percentage range labels."""
    labels: List[str] = []
    for start, end in zip(edges[:-1], edges[1:]):
        labels.append(f"{int(round(start * 100)):02d}-{int(round(end * 100)):02d}%")
    return labels


def plot_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Create and save the bar chart."""
    centers = (edges[:-1] + edges[1:]) / 2.0
    plt.figure(figsize=(12, 6))
    plt.bar(centers * 100.0, counts, width=(edges[1] - edges[0]) * 100.0 * 0.9)
    plt.xlabel("Confidence (%)")
    plt.ylabel("Cell Count")
    plt.title(title)
    plt.xticks(centers * 100.0, format_bin_labels(edges), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histogram of prediction confidences for test CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("data/output/test"),
        help="Directory containing *_cleaned.csv files.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("predictions"),
        help="Directory containing prediction CSV files with probability column.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.05,
        help="Histogram bin size (default 0.05 for 5%% steps).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/confidence_hist.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Prediction Confidence Histogram (Test Sets)",
        help="Title for the generated chart.",
    )

    args = parser.parse_args()

    if args.bin_size <= 0 or args.bin_size > 1:
        raise ValueError("--bin-size must be between 0 and 1.")

    csv_paths = sorted(args.csv_dir.rglob("*_cleaned.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_cleaned.csv found under {args.csv_dir}")

    all_probs: List[pd.Series] = []
    missing: List[str] = []

    for csv_path in csv_paths:
        dataset_stem = csv_path.stem.replace("_cleaned", "")
        pred_csv = find_prediction_csv(args.pred_dir, dataset_stem)
        if pred_csv is None:
            missing.append(dataset_stem)
            continue
        try:
            probs = load_probabilities(pred_csv)
        except ValueError:
            missing.append(dataset_stem)
            continue
        all_probs.append(probs)
        print(f"Loaded {len(probs):5d} rows from {pred_csv.name}")

    if not all_probs:
        raise RuntimeError(
            "No prediction files with probability column were found for the given datasets."
        )

    combined = pd.concat(all_probs, ignore_index=True)
    counts, edges = build_histogram(combined, args.bin_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_histogram(counts, edges, args.output, args.title)
    print(f"Saved histogram to {args.output}")

    if missing:
        print(
            "Warning: missing predictions for datasets:",
            ", ".join(sorted(set(missing))),
        )


if __name__ == "__main__":
    main()
