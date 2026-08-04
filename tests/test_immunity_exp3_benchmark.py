from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from immunity.exp3.benchmark import (
    OuterSplit,
    make_inner_splits,
    make_outer_splits,
    outer_split_manifest,
    regression_metrics,
)


def make_grouped_images() -> pd.DataFrame:
    """建立涵蓋全部 Exp3 分組維度的 synthetic image frame。"""
    rows = []
    for b_index, b_id in enumerate(["B4", "B7", "B8"]):
        for passage in [5, 6, 7]:
            for condition_index in range(1, 9):
                rows.append(
                    {
                        "image_key": (
                            f"{b_id}_P{passage}_C{condition_index:02d}_F01"
                        ),
                        "b_id": b_id,
                        "passage": passage,
                        "group_id": f"{b_id}_P{passage}",
                        "condition_index": condition_index,
                        "condition": f"condition_{condition_index}",
                        "ifn_dose": float(condition_index - 1),
                        "tnf_dose": 0.0,
                        "fov": 1,
                        "IDO_score": (
                            b_index + passage / 10 + condition_index / 20
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_outer_splits_have_expected_counts_disjoint_groups_and_full_coverage(
) -> None:
    images = make_grouped_images()

    splits = make_outer_splits(images)

    assert Counter(split.validation for split in splits) == {
        "leave_one_b_out": 3,
        "leave_one_passage_out": 3,
        "leave_one_group_out": 9,
        "leave_one_condition_out": 8,
    }
    grouping_column = {
        "leave_one_b_out": "b_id",
        "leave_one_passage_out": "passage",
        "leave_one_group_out": "group_id",
        "leave_one_condition_out": "condition_index",
    }
    for split in splits:
        assert split.train_index.size > 0
        assert split.test_index.size > 0
        assert set(split.train_index).isdisjoint(split.test_index)
        column = grouping_column[split.validation]
        assert set(images.iloc[split.train_index][column]).isdisjoint(
            images.iloc[split.test_index][column]
        )

    for validation in grouping_column:
        tested = np.concatenate(
            [split.test_index for split in splits if split.validation == validation]
        )
        assert sorted(tested.tolist()) == list(range(len(images)))


def test_outer_splits_are_deterministic_and_fold_names_are_sorted() -> None:
    images = make_grouped_images().sample(frac=1.0, random_state=23)

    first = make_outer_splits(images)
    second = make_outer_splits(images.copy())

    assert [(split.validation, split.fold) for split in first] == [
        ("leave_one_b_out", "B4"),
        ("leave_one_b_out", "B7"),
        ("leave_one_b_out", "B8"),
        ("leave_one_passage_out", "5"),
        ("leave_one_passage_out", "6"),
        ("leave_one_passage_out", "7"),
        *(
            ("leave_one_group_out", group_id)
            for group_id in (
                "B4_P5",
                "B4_P6",
                "B4_P7",
                "B7_P5",
                "B7_P6",
                "B7_P7",
                "B8_P5",
                "B8_P6",
                "B8_P7",
            )
        ),
        *(("leave_one_condition_out", str(index)) for index in range(1, 9)),
    ]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.train_index, right.train_index)
        np.testing.assert_array_equal(left.test_index, right.test_index)


def test_outer_split_manifest_preserves_positional_rows_and_metadata() -> None:
    images = make_grouped_images().iloc[[7, 0, 15, 8]].copy()
    images.index = [101, 305, 502, 900]
    split = OuterSplit(
        validation="manual",
        fold="fold-a",
        train_index=np.array([0, 2]),
        test_index=np.array([1, 3]),
    )

    manifest = outer_split_manifest(images, [split])

    assert manifest.columns.tolist() == [
        "validation",
        "fold",
        "image_key",
        "role",
        "b_id",
        "passage",
        "group_id",
        "condition_index",
        "condition",
        "ifn_dose",
        "tnf_dose",
        "fov",
    ]
    assert manifest["image_key"].tolist() == images["image_key"].tolist()
    assert manifest["role"].tolist() == ["train", "test", "train", "test"]
    pd.testing.assert_frame_equal(
        manifest.loc[:, ["image_key", "b_id", "passage", "group_id"]]
        .reset_index(drop=True),
        images.loc[:, ["image_key", "b_id", "passage", "group_id"]]
        .reset_index(drop=True),
    )


def test_outer_splits_reject_missing_or_ambiguous_grouping_metadata() -> None:
    images = make_grouped_images()

    with pytest.raises(ValueError, match="缺少.*group_id"):
        make_outer_splits(images.drop(columns="group_id"))
    with pytest.raises(ValueError, match="image_key.*重複"):
        make_outer_splits(
            pd.concat([images, images.iloc[[0]]], ignore_index=True)
        )
    images.loc[0, "b_id"] = None
    with pytest.raises(ValueError, match="b_id.*空值"):
        make_outer_splits(images)


def test_inner_splits_use_groupkfold_without_group_leakage() -> None:
    images = make_grouped_images()

    splits = make_inner_splits(images, requested=5)

    assert len(splits) == 5
    tested = np.concatenate([test_index for _, test_index in splits])
    assert sorted(tested.tolist()) == list(range(len(images)))
    for train_index, test_index in splits:
        assert set(images.iloc[train_index].group_id).isdisjoint(
            images.iloc[test_index].group_id
        )


def test_inner_splits_return_positions_relative_to_non_range_training_frame() -> None:
    training = make_grouped_images().iloc[[0, 8, 16, 24, 32, 40]].copy()
    training.index = [10, 20, 30, 40, 50, 60]

    splits = make_inner_splits(training, requested=20)

    assert len(splits) == training["group_id"].nunique()
    assert all(
        0 <= int(index) < len(training)
        for split in splits
        for indexes in split
        for index in indexes
    )


def test_inner_splits_require_at_least_two_groups_and_two_requested_folds() -> None:
    images = make_grouped_images()

    with pytest.raises(ValueError):
        make_inner_splits(images[images["group_id"].eq("B4_P5")], requested=5)
    with pytest.raises(ValueError, match="requested.*至少為 2"):
        make_inner_splits(images, requested=1)


def test_constant_vectors_keep_errors_but_return_nan_correlations() -> None:
    metrics = regression_metrics([1, 1, 1], [1, 1, 1])

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["spearman"])


def test_metrics_keep_r2_when_only_predictions_are_constant() -> None:
    metrics = regression_metrics([1, 2, 3], [2, 2, 2])

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["r2"] == pytest.approx(0.0)
    assert np.isnan(metrics["spearman"])


def test_single_pair_keeps_errors_but_returns_nan_for_r2_and_spearman() -> None:
    metrics = regression_metrics([3.0], [1.0])

    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["spearman"])


@pytest.mark.parametrize(
    ("observed", "predicted", "message"),
    [
        ([], [], "不可為空"),
        ([1.0, 2.0], [1.0], "長度必須相同"),
        ([1.0, 2.0], [1.0, np.nan], "predicted.*有限值"),
        ([1.0, 2.0], [1.0, np.inf], "predicted.*有限值"),
    ],
)
def test_metrics_reject_empty_unequal_or_nonfinite_predictions(
    observed: list[float], predicted: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        regression_metrics(observed, predicted)
