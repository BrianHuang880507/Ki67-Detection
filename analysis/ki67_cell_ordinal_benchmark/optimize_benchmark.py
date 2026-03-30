#!/usr/bin/env python
"""Model tuning and ensemble experiments for Ki67 cell benchmark."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_benchmark as rb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune tabular models for Ki67 benchmark")
    parser.add_argument("--results-root", type=Path, default=Path("data/output/results"))
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--allow-missing-classes", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def metric_sort_key(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["quadratic_weighted_kappa"]),
        float(row["macro_f1"]),
        float(row["balanced_accuracy"]),
        float(row["accuracy"]),
    )


def fit_and_score(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
    y_eval: np.ndarray,
) -> dict[str, float]:
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    return rb.compute_metrics(y_eval, pred, labels=rb.ORDINAL_LABELS)


def build_tuning_spaces(random_state: int, num_classes: int) -> dict[str, list[dict[str, Any]]]:
    spaces: dict[str, list[dict[str, Any]]] = {
        "logistic_regression": [
            {"C": 0.25, "class_weight": "balanced"},
            {"C": 1.0, "class_weight": "balanced"},
            {"C": 2.0, "class_weight": None},
        ],
        "random_forest": [
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 800, "max_depth": 18, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 1000, "max_depth": 24, "min_samples_leaf": 1, "max_features": 0.7},
        ],
        "extra_trees": [
            {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 1000, "max_depth": 22, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 1200, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.8},
        ],
        "hist_gradient_boosting": [
            {"max_iter": 400, "learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 20},
            {"max_iter": 700, "learning_rate": 0.03, "max_depth": 8, "min_samples_leaf": 20},
            {"max_iter": 900, "learning_rate": 0.02, "max_depth": 8, "min_samples_leaf": 40},
        ],
    }
    try:
        from xgboost import XGBClassifier  # noqa: F401

        spaces["xgboost"] = [
            {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 800,
                "max_depth": 7,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_lambda": 1.5,
            },
        ]
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier  # noqa: F401

        spaces["catboost"] = [
            {"iterations": 500, "depth": 6, "learning_rate": 0.05},
            {"iterations": 800, "depth": 7, "learning_rate": 0.03},
        ]
    except Exception:
        pass

    return spaces


def build_model(model_name: str, params: dict[str, Any], random_state: int, num_classes: int) -> Pipeline:
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=random_state,
                        C=params["C"],
                        class_weight=params["class_weight"],
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=random_state,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        **params,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        random_state=random_state,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        **params,
                    ),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=random_state,
                        **params,
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=num_classes,
                        eval_metric="mlogloss",
                        random_state=random_state,
                        n_jobs=-1,
                        **params,
                    ),
                ),
            ]
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        loss_function="MultiClass",
                        random_seed=random_state,
                        verbose=False,
                        **params,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_and_export_test(
    output_dir: Path,
    variant: str,
    tag: str,
    model_name: str,
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    row = rb.evaluate_model_on_split(
        model=model,
        model_name=f"{model_name}__{tag}",
        variant_name=variant,
        split_name="test",
        split_frame=test_df,
        feature_cols=feature_cols,
        output_dir=output_dir,
    )
    row["tag"] = tag
    return row


def fit_refit_and_eval_test(
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Pipeline, dict[str, float]]:
    train_val = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    model.fit(train_val[feature_cols], train_val["label"].to_numpy())
    pred = model.predict(test_df[feature_cols])
    metrics = rb.compute_metrics(test_df["label"].to_numpy(), pred, labels=rb.ORDINAL_LABELS)
    return model, metrics


def build_soft_voting_model(fitted_models: list[tuple[str, Pipeline]]) -> Pipeline:
    estimators = [(name, model) for name, model in fitted_models]
    voting = VotingClassifier(estimators=estimators, voting="soft")
    return Pipeline(steps=[("voter", voting)])


def run() -> Path:
    args = parse_args()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("analysis/ki67_cell_ordinal_benchmark") / f"run_tuned_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, manifest = rb.load_all_features(args.results_root)
    labeled_df, label_info = rb.attach_labels(raw_df, args)
    split_df, split_map = rb.build_grouped_split(
        labeled_df=labeled_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    manifest.to_csv(output_dir / "source_manifest.csv", index=False)
    split_map.sort_values(["dataset", "split", "Image"]).to_csv(output_dir / "image_split_map.csv", index=False)
    split_df[rb.KEY_COLUMNS + ["P", "label", "split"]].to_csv(output_dir / "cell_split_detail.csv", index=False)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    tuning_spaces = build_tuning_spaces(args.random_state, len(rb.ORDINAL_LABELS))
    candidate_rows: list[dict[str, Any]] = []
    tuned_rows: list[dict[str, Any]] = []
    best_configs: list[dict[str, Any]] = []

    for variant, include_passage in [("without_passage", False), ("with_passage", True)]:
        feature_cols = rb.resolve_feature_columns(split_df, explicit_features=None, include_passage=include_passage)
        (output_dir / f"{variant}_feature_columns.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

        x_train = train_df[feature_cols]
        y_train = train_df["label"].to_numpy()
        x_val = val_df[feature_cols]
        y_val = val_df["label"].to_numpy()

        # baseline from existing registry, ranked on validation.
        baseline_models, skipped_optional = rb.build_model_registry(args.random_state, len(rb.ORDINAL_LABELS))
        for model_name, model in baseline_models.items():
            try:
                m = fit_and_score(model, x_train, y_train, x_val, y_val)
                candidate_rows.append(
                    {
                        "variant": variant,
                        "family": model_name,
                        "candidate_id": "baseline_default",
                        "params_json": "{}",
                        "status": "ok",
                        **m,
                    }
                )
                # Refit on train+val then evaluate test.
                refit_model, _ = fit_refit_and_eval_test(model, train_df, val_df, test_df, feature_cols)
                trow = evaluate_and_export_test(
                    output_dir=output_dir,
                    variant=variant,
                    tag="baseline_refit",
                    model_name=model_name,
                    model=refit_model,
                    test_df=test_df,
                    feature_cols=feature_cols,
                )
                tuned_rows.append(trow)
            except Exception as exc:
                candidate_rows.append(
                    {
                        "variant": variant,
                        "family": model_name,
                        "candidate_id": "baseline_default",
                        "params_json": "{}",
                        "status": "fail",
                        "reason": str(exc),
                        "accuracy": math.nan,
                        "macro_f1": math.nan,
                        "balanced_accuracy": math.nan,
                        "quadratic_weighted_kappa": math.nan,
                    }
                )

        for model_name, configs in tuning_spaces.items():
            family_candidates: list[dict[str, Any]] = []
            for idx, params in enumerate(configs):
                cid = f"cfg_{idx+1}"
                try:
                    model = build_model(model_name, params, args.random_state, len(rb.ORDINAL_LABELS))
                    m = fit_and_score(model, x_train, y_train, x_val, y_val)
                    row = {
                        "variant": variant,
                        "family": model_name,
                        "candidate_id": cid,
                        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                        "status": "ok",
                        "reason": "",
                        **m,
                    }
                except Exception as exc:
                    row = {
                        "variant": variant,
                        "family": model_name,
                        "candidate_id": cid,
                        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                        "status": "fail",
                        "reason": str(exc),
                        "accuracy": math.nan,
                        "macro_f1": math.nan,
                        "balanced_accuracy": math.nan,
                        "quadratic_weighted_kappa": math.nan,
                    }
                candidate_rows.append(row)
                family_candidates.append(row)

            ok_candidates = [r for r in family_candidates if r["status"] == "ok"]
            if not ok_candidates:
                continue
            best = sorted(
                ok_candidates,
                key=lambda r: (
                    r["quadratic_weighted_kappa"],
                    r["macro_f1"],
                    r["balanced_accuracy"],
                    r["accuracy"],
                ),
                reverse=True,
            )[0]
            best_configs.append(best)
            best_params = json.loads(best["params_json"])
            best_model = build_model(model_name, best_params, args.random_state, len(rb.ORDINAL_LABELS))
            best_model, _ = fit_refit_and_eval_test(best_model, train_df, val_df, test_df, feature_cols)
            trow = evaluate_and_export_test(
                output_dir=output_dir,
                variant=variant,
                tag="tuned_refit",
                model_name=model_name,
                model=best_model,
                test_df=test_df,
                feature_cols=feature_cols,
            )
            tuned_rows.append(trow)

        # Optional soft voting ensemble over top-3 tuned model families.
        variant_best = [r for r in best_configs if r["variant"] == variant]
        if len(variant_best) >= 2:
            top = sorted(
                variant_best,
                key=lambda r: (
                    r["quadratic_weighted_kappa"],
                    r["macro_f1"],
                    r["balanced_accuracy"],
                    r["accuracy"],
                ),
                reverse=True,
            )[:3]
            fitted: list[tuple[str, Pipeline]] = []
            for i, item in enumerate(top):
                family = item["family"]
                params = json.loads(item["params_json"])
                m = build_model(family, params, args.random_state + i + 7, len(rb.ORDINAL_LABELS))
                train_val = pd.concat([train_df, val_df], axis=0, ignore_index=True)
                m.fit(train_val[feature_cols], train_val["label"].to_numpy())
                fitted.append((f"{family}_{i+1}", m))
            try:
                ensemble = build_soft_voting_model(fitted)
                # VotingClassifier needs fitting wrapper; pass train_val once.
                train_val = pd.concat([train_df, val_df], axis=0, ignore_index=True)
                ensemble.fit(train_val[feature_cols], train_val["label"].to_numpy())
                trow = evaluate_and_export_test(
                    output_dir=output_dir,
                    variant=variant,
                    tag="soft_voting_refit",
                    model_name="ensemble",
                    model=ensemble,
                    test_df=test_df,
                    feature_cols=feature_cols,
                )
                tuned_rows.append(trow)
            except Exception:
                pass

    candidates_df = pd.DataFrame(candidate_rows)
    tuned_df = pd.DataFrame(tuned_rows)
    best_df = pd.DataFrame(best_configs)

    candidates_df.to_csv(output_dir / "tuning_candidates_val.csv", index=False)
    best_df.to_csv(output_dir / "best_configs_by_family.csv", index=False)
    tuned_df.to_csv(output_dir / "tuned_test_metrics.csv", index=False)

    if not tuned_df.empty:
        rank = tuned_df.sort_values(
            ["quadratic_weighted_kappa", "macro_f1", "balanced_accuracy", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)
        rank.to_csv(output_dir / "tuned_model_ranking_test.csv", index=False)

    config = {
        "output_dir": str(output_dir),
        "results_root": str(args.results_root),
        "label_info": label_info,
        "random_state": args.random_state,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    out = run()
    print(f"[OK] Optimization complete. Output: {out}")
