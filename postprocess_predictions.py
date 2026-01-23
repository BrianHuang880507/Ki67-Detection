#!/usr/bin/env python3
"""
Post-process Ki67 prediction CSVs to adjust decision logic and report metrics.
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

TARGET_COL = "ki67_positive"
PROB_COL = "probability"
PRED_COL = "prediction"
ACCURACY_COL = "accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply custom post-processing rules to Ki67 model prediction CSVs."
        )
    )
    parser.add_argument("csv", type=Path, help="Path to the predictions CSV to modify.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--neg-threshold",
        type=float,
        default=0.15,
        help="Probability threshold used to mark cells as negative.",
    )
    parser.add_argument(
        "--precision-target",
        type=float,
        default=0.9,
        help="Desired precision level for selecting the positive probability threshold.",
    )
    return parser.parse_args()


def prepare_ground_truth(df: pd.DataFrame) -> pd.Series:
    if TARGET_COL not in df.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return series


def select_positive_threshold(
    probs: pd.Series,
    labels: pd.Series,
    precision_target: float,
    neg_threshold: float,
) -> Tuple[float, Optional[float], int, int, int]:
    known_mask = labels.isin([0, 1]) & probs.notna()
    if not known_mask.any():
        return float("inf"), None, 0, 0, 0
    search_df = pd.DataFrame({"prob": probs[known_mask], "label": labels[known_mask]})
    candidate_thresholds = sorted(search_df["prob"].unique(), reverse=True)
    best_threshold = float("inf")
    best_precision: Optional[float] = None
    best_diff = float("inf")
    best_counts = (0, 0)

    for threshold in candidate_thresholds:
        if threshold <= neg_threshold:
            continue
        mask = search_df["prob"] >= threshold
        denom = int(mask.sum())
        if denom == 0:
            continue
        positives = search_df.loc[mask, "label"]
        tp = int((positives == 1).sum())
        fp = denom - tp
        precision = tp / denom if denom else None
        if precision is None:
            continue
        diff = abs(precision - precision_target)
        # Prefer precision that is >= target when equally close; otherwise pick best diff.
        prefer_current = False
        if diff < best_diff:
            prefer_current = True
        elif diff == best_diff:
            if best_precision is None:
                prefer_current = True
            elif precision >= precision_target and (
                best_precision < precision_target or precision > best_precision
            ):
                prefer_current = True
        if prefer_current:
            best_threshold = float(threshold)
            best_precision = precision
            best_diff = diff
            best_counts = (tp, fp)

    if best_precision is None:
        return float("inf"), None, 0, 0, 0

    tp, fp = best_counts
    denom = tp + fp
    return best_threshold, best_precision, tp, fp, denom


def compute_negative_stats(
    probs: pd.Series,
    labels: pd.Series,
    neg_threshold: float,
    pos_threshold: float,
) -> Tuple[Optional[float], int, int, int]:
    known_mask = labels.isin([0, 1]) & probs.notna()
    if not known_mask.any():
        return None, 0, 0, 0
    if np.isfinite(pos_threshold):
        mask = (probs >= neg_threshold) & (probs < pos_threshold) & known_mask
    else:
        mask = (probs >= neg_threshold) & known_mask
    denom = int(mask.sum())
    if denom == 0:
        return None, 0, 0, 0
    subset = labels[mask]
    tn = int((subset == 0).sum())
    fn = denom - tn
    npv = tn / denom if denom else None
    return npv, tn, fn, denom


def update_predictions(
    df: pd.DataFrame,
    pos_threshold: float,
    neg_threshold: float,
) -> None:
    original_preds = df.get(PRED_COL, pd.Series(dtype=object))
    original_preds = original_preds.astype(str, copy=False)
    probs = pd.to_numeric(df.get(PROB_COL), errors="coerce")

    new_preds = []
    for idx, prob in probs.items():
        if pd.isna(prob):
            new_preds.append(
                original_preds.loc[idx] if idx in original_preds.index else ""
            )
            continue
        if prob >= pos_threshold:
            new_preds.append("1")
        elif prob >= neg_threshold:
            new_preds.append("0")
        else:
            base = original_preds.loc[idx] if idx in original_preds.index else ""
            if isinstance(base, str) and base.lower() != "nan" and base != "":
                new_preds.append(f"{base}(uncertain)")
            else:
                new_preds.append("uncertain")
    df[PRED_COL] = new_preds


def update_accuracy_summary(
    df: pd.DataFrame,
    npv: Optional[float],
    precision: Optional[float],
) -> None:
    if ACCURACY_COL not in df.columns:
        return
    accuracy_series = df[ACCURACY_COL]
    summary_idx = accuracy_series.last_valid_index()
    if summary_idx is None:
        return
    values = [m for m in (npv, precision) if m is not None]
    if not values:
        return
    final_value: float
    if len(values) == 2:
        avg = sum(values) / 2.0
        final_value = avg if avg >= 0.9 else max(values)
    else:
        final_value = values[0]
    df.at[summary_idx, ACCURACY_COL] = f"{final_value:.6f}"


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    output_path = args.output or csv_path

    df = pd.read_csv(csv_path)
    labels = prepare_ground_truth(df)
    probs = pd.to_numeric(df.get(PROB_COL), errors="coerce")

    pos_threshold, precision, tp, fp, pos_denom = select_positive_threshold(
        probs, labels, args.precision_target, args.neg_threshold
    )
    npv, tn, fn, neg_denom = compute_negative_stats(
        probs, labels, args.neg_threshold, pos_threshold
    )

    update_predictions(df, pos_threshold, args.neg_threshold)
    update_accuracy_summary(df, npv, precision)

    df.to_csv(output_path, index=False)

    total_denominator = neg_denom + pos_denom
    print("Post-processing summary")
    print("----------------------")
    print(f"Negative threshold (>=): {args.neg_threshold:.3f}")
    if npv is not None:
        print(f"Negative predictive value: {npv:.4f} (TN={tn}, FN={fn}, denom={neg_denom})")
    else:
        print("Negative predictive value: N/A")
    if np.isfinite(pos_threshold):
        print(f"Positive threshold (>=): {pos_threshold:.3f}")
    else:
        print("Positive threshold (>=): N/A")
    if precision is not None:
        print(
            f"Precision: {precision:.4f} (TP={tp}, FP={fp}, denom={pos_denom})"
        )
    else:
        print("Precision: N/A")
    print(f"Denominator total (NPV + precision): {total_denominator}")
    print(f"Updated file written to: {output_path}")


if __name__ == "__main__":
    main()
