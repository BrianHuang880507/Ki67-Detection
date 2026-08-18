from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _write_mask(masks_dir: Path, image_key: str, labels: np.ndarray) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    np.savez(masks_dir / f"{image_key}.npz", cell_mask=labels)


def _manifest(image_keys: tuple[str, ...], groups: tuple[str, ...] | None = None) -> pd.DataFrame:
    groups = groups or tuple("B4_P5" for _ in image_keys)
    return pd.DataFrame(
        {
            "image_key": image_keys,
            "b_id": [group.split("_")[0] for group in groups],
            "passage": [int(group.split("_P")[1]) for group in groups],
            "group_id": groups,
        }
    )


def _cells(image_keys: tuple[str, ...], labels: tuple[int, ...], ido: tuple[float, ...] | None = None) -> pd.DataFrame:
    from immunity.exp3.feature_sets import PRIMARY_CELL_FEATURES

    rows = []
    ido = ido or tuple(float(index + 1) for index in range(len(labels)))
    for image_key in image_keys:
        for index, (label, value) in enumerate(zip(labels, ido), start=1):
            row = {
                "image_key": image_key,
                "cell_label": label,
                "nucleus_label": index,
                "IDO_score": value,
            }
            row.update(
                {
                    column: (
                        100.0
                        if column == "cell__area"
                        else 1.0 / 99.0
                        if column == "nucleus_cytoplasm_area_ratio"
                        else 1.0
                    )
                    for column in PRIMARY_CELL_FEATURES
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _fill_high_features(frame: pd.DataFrame) -> pd.DataFrame:
    """補齊 synthetic raw rows 的 high F0 欄位。"""
    from immunity.exp3.feature_sets import PRIMARY_CELL_FEATURES

    for column in PRIMARY_CELL_FEATURES:
        default = (
            100.0
            if column == "cell__area"
            else 0.001
            if column == "nucleus__area"
            else 1.0 / 99.0
            if column == "nucleus_cytoplasm_area_ratio"
            else 1.0
        )
        frame[column] = frame[column].fillna(default)
    return frame


def _valid_fixture(tmp_path: Path):
    masks_dir = tmp_path / "masks"
    image_keys = ("FOV_A", "FOV_B")
    labels_a = np.zeros((6, 6), dtype=np.int32)
    labels_a[1:3, 1:3] = 1
    labels_a[1:3, 3:5] = 2
    labels_a[3:5, 1:3] = 3
    labels_a[0, 3:5] = 4
    labels_b = np.zeros((6, 6), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    labels_b[1:3, 3:5] = 2
    labels_b[3:5, 1:3] = 3
    _write_mask(masks_dir, "FOV_A", labels_a)
    _write_mask(masks_dir, "FOV_B", labels_b)
    manifest = _manifest(image_keys)
    cells = pd.concat(
        [
            _cells(("FOV_A",), (1, 2, 3, 4), (1.0, 3.0, 5.0, 100.0)),
            _cells(("FOV_B",), (1, 2, 3), (2.0, 4.0, 6.0)),
        ],
        ignore_index=True,
    )
    return manifest, cells, masks_dir


def test_border_label_detection_hits_all_four_edges_but_not_interior() -> None:
    from immunity.exp4.border_exclusion import find_border_labels

    labels = np.zeros((5, 5), dtype=np.int32)
    labels[0, 1] = 1
    labels[4, 2] = 2
    labels[2, 0] = 3
    labels[2, 4] = 4
    labels[2, 2] = 5

    assert find_border_labels(labels) == {1, 2, 3, 4}


def test_same_cell_label_is_aligned_by_image_key_and_cell_label(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import apply_border_exclusion

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )

    assert set(result.excluded_composite_keys) == {("FOV_A", 4)}
    assert len(result.eligible_cells) == 6
    assert (
        (result.eligible_cells["image_key"] == "FOV_B")
        & (result.eligible_cells["cell_label"] == 1)
    ).any()


def test_v4_border_uses_distinct_dedup_cells_and_keeps_two_cell_fov(
    tmp_path: Path,
) -> None:
    """v4 以 whole-cell frame 計數，稀疏 FOV 不套用 min-3 gate。"""
    from immunity.exp4.border_exclusion import apply_border_exclusion

    masks_dir = tmp_path / "masks"
    labels_a = np.zeros((6, 6), dtype=np.int32)
    labels_a[1:3, 1:3] = 1
    labels_a[3:5, 1:3] = 2
    labels_a[0, 3:5] = 3
    labels_b = np.zeros((6, 6), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    labels_b[1:3, 3:5] = 2
    labels_b[3:5, 1:3] = 3
    _write_mask(masks_dir, "FOV_A", labels_a)
    _write_mask(masks_dir, "FOV_B", labels_b)
    manifest = _manifest(("FOV_A", "FOV_B"))
    raw_cells = pd.DataFrame(
        [
            {"image_key": "FOV_A", "cell_label": 1, "nucleus_label": 1, "IDO_score": 1.0},
            {"image_key": "FOV_A", "cell_label": 1, "nucleus_label": 2, "IDO_score": 3.0},
            {"image_key": "FOV_A", "cell_label": 1, "nucleus_label": 5, "IDO_score": 7.0},
            {"image_key": "FOV_A", "cell_label": 2, "nucleus_label": 3, "IDO_score": 5.0},
            {"image_key": "FOV_A", "cell_label": 3, "nucleus_label": 4, "IDO_score": 99.0},
            {"image_key": "FOV_B", "cell_label": 1, "nucleus_label": 1, "IDO_score": 2.0},
            {"image_key": "FOV_B", "cell_label": 2, "nucleus_label": 2, "IDO_score": 4.0},
            {"image_key": "FOV_B", "cell_label": 3, "nucleus_label": 3, "IDO_score": 6.0},
        ]
    )
    from immunity.exp3.feature_sets import PRIMARY_CELL_FEATURES

    for column in PRIMARY_CELL_FEATURES:
        raw_cells[column] = (
            100.0
            if column == "cell__area"
            else 1.0 / 99.0
            if column == "nucleus_cytoplasm_area_ratio"
            else 1.0
        )

    result = apply_border_exclusion(
        manifest,
        raw_cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=8,
        expected_cell_keys=2,
        expected_dedup_cells=6,
        expected_duplicate_keys=1,
        expected_group_keys=("B4_P5",),
    )

    assert len(result.raw_cells) == 8
    assert len(result.pre_cells) == 6
    assert result.border_report["cells_after"].tolist() == [2, 3]
    assert result.border_report["whole_cell_labels_after"].tolist() == [2, 3]
    assert result.minimum_cell_gate_status == "passed"
    assert result.duplicate_key_count == 1
    assert result.duplicate_excess_row_count == 2


def test_sparse_roster_lock_is_checked_against_inferred_two_cell_fovs(
    tmp_path: Path,
) -> None:
    from immunity.exp4.border_exclusion import BorderExclusionError, apply_border_exclusion

    masks_dir = tmp_path / "masks"
    labels_a = np.zeros((4, 4), dtype=np.int32)
    labels_a[1:3, 1:3] = 1
    labels_a[0, 1:3] = 2
    labels_b = np.zeros((4, 4), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    labels_b[1:3, 2:4] = 2
    labels_b[2:4, 1:3] = 3
    _write_mask(masks_dir, "FOV_A", labels_a)
    _write_mask(masks_dir, "FOV_B", labels_b)
    manifest = _manifest(("FOV_A", "FOV_B"))
    cells = _cells(("FOV_A",), (1, 2), (1.0, 2.0))._append(
        _cells(("FOV_B",), (1, 2, 3), (3.0, 4.0, 5.0)),
        ignore_index=True,
    )

    with pytest.raises(BorderExclusionError, match="sparse FOV roster"):
        apply_border_exclusion(
            manifest,
            cells,
            masks_dir,
            expected_manifest_keys=2,
            expected_cell_rows=5,
            expected_cell_keys=2,
            expected_dedup_cells=5,
            expected_duplicate_keys=0,
            expected_sparse_fov_keys=("FOV_B",),
            expected_group_keys=("B4_P5",),
        )


def test_basic_23976_row_adapter_does_not_require_rui49_gate(monkeypatch) -> None:
    """37 欄 basic raw 只走 F0 dedup；Rui49 gate 留給 feature join。"""
    from immunity.exp3.feature_sets import PRIMARY_CELL_FEATURES
    from immunity.exp4 import border_exclusion as border

    rows: list[dict[str, object]] = []
    for index in range(23976):
        image_key = "FOV_A" if index % 2 == 0 else "FOV_B"
        cell_label = 1 if index % 2 == 0 else 2
        row: dict[str, object] = {
            "image_key": image_key,
            "cell_label": cell_label,
            "nucleus_label": index + 1,
            "IDO_score": float(index % 11),
        }
        row.update(
            {
                column: (
                    100.0
                    if column == "cell__area"
                    else 0.001
                    if column == "nucleus__area"
                    else 1.0
                    if column.startswith("cell__")
                    else 1.0 / 99.0
                    if column == "nucleus_cytoplasm_area_ratio"
                    else float(index % 7 + 1)
                )
                for column in PRIMARY_CELL_FEATURES
            }
        )
        rows.append(row)
    raw = pd.DataFrame(rows)
    result, report = border._deduplicate_for_border(
        raw,
        deduplicated_cells=None,
        dedup_report=None,
        expected_raw_rows=23976,
        expected_distinct_cells=2,
        expected_multirow_keys=2,
    )

    assert len(result) == 2
    assert len(report) == 2


def test_report_skeleton_discloses_v4_boundaries_without_model_results(tmp_path: Path) -> None:
    from immunity.exp4.reporting import write_report_skeleton

    output = tmp_path / "REPORT.md"
    write_report_skeleton(output)
    content = output.read_text(encoding="utf-8")
    assert "458" in content and "6,152" in content
    assert "B8_P7" in content and "low confidence" in content
    assert "FOV_IDO" in content and "QC" in content
    assert "可信度未確認" in content
    assert "尚未執行" in content


def test_canonical_49_join_calls_high_gate_without_removed_override(
    tmp_path: Path, monkeypatch
) -> None:
    from immunity.exp4 import border_exclusion as border
    from immunity.exp4.rui_features import RUI49_FEATURE_COLUMNS

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = border.apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    features = result.pre_cells.loc[:, ["image_key", "cell_label"]].drop_duplicates()
    features = features.assign(
        **{column: 1.0 for column in RUI49_FEATURE_COLUMNS}
    )
    called: dict[str, object] = {}

    def fake_gate(joined: pd.DataFrame, *, scope: str):
        called.update(rows=len(joined), scope=scope)
        return None

    monkeypatch.setattr(border, "assert_full_rui49_feature_consistency", fake_gate)
    x, y = border.assemble_feature_xy(
        result,
        features,
        feature_columns=RUI49_FEATURE_COLUMNS,
    )

    assert len(x) == len(y) == len(result.eligible_cells)
    assert called == {"rows": len(result.raw_cells), "scope": "full_join_693_fovs"}


def test_dedup_seam_rejects_non_high_result(monkeypatch) -> None:
    from immunity.exp4 import border_exclusion as border

    raw = _cells(("FOV_A",), (1,), (1.0,))

    monkeypatch.setattr(
        border,
        "deduplicate_master_cells",
        lambda *args, **kwargs: SimpleNamespace(
            deduplicated_cells=raw,
            dedup_report=pd.DataFrame(),
        ),
    )

    with pytest.raises(Exception, match="CellDedupResult"):
        border._deduplicate_for_border(
            raw,
            deduplicated_cells=None,
            dedup_report=None,
            expected_raw_rows=1,
            expected_distinct_cells=1,
            expected_multirow_keys=0,
        )


def test_dedup_writer_rejects_noncanonical_report_schema(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import BorderExclusionError, write_cell_dedup_report

    with pytest.raises(BorderExclusionError, match="schema"):
        write_cell_dedup_report(
            pd.DataFrame(columns=("image_key", "unexpected")),
            tmp_path / "dedup.csv",
            expected_duplicate_keys=0,
        )


def test_report_embeds_measured_group_and_sensitivity_tables(tmp_path: Path) -> None:
    from immunity.exp4.reporting import write_report_skeleton

    groups = (
        "B4_P5", "B4_P6", "B4_P7", "B7_P5", "B7_P6", "B7_P7",
        "B8_P5", "B8_P6", "B8_P7",
    )
    targets = pd.DataFrame(
        {
            "group_id": groups,
            "b_id": [group.split("_")[0] for group in groups],
            "passage": [int(group.split("_P")[1]) for group in groups],
            "fov_count": [61] * 9,
            "cells_before": [1802, 6152, 1492, 3469, 3584, 1685, 1465, 2905, 458],
            "cells_after": [1700, 6000, 1400, 3400, 3500, 1600, 1400, 2800, 387],
            "group_IDO_score": [float(index + 1) for index in range(9)],
            "group_IDO_score_all_cells": [float(index + 1) for index in range(9)],
            "delta": [0.0] * 9,
            "confidence_flag": ["low" if group == "B8_P7" else "normal" for group in groups],
        }
    )
    sensitivity = pd.DataFrame(
        {
            "group_id": groups,
            "sparse_fov_count": [0] * 8 + [5],
            "target_with_sparse_fovs": [1.0] * 9,
            "target_without_sparse_fovs": [1.0] * 8 + [1.0436500752998],
            "delta_without_minus_with": [0.0] * 8 + [0.0436500752998],
            "absolute_delta": [0.0] * 8 + [0.0436500752998],
            "baseline_target_range": [1.90995757995635] * 9,
            "absolute_delta_pct_of_range": [0.0] * 8 + [4.36500752998],
            "threshold_pct": [5.0] * 8 + [15.0],
            "threshold_absolute": [0.095] * 8 + [0.285],
            "status": ["passed"] * 9,
        }
    )
    border = pd.DataFrame(
        {
            "image_key": ["B8_P7/C01_F02", "B8_P7/C01_F03"],
            "group_id": ["B8_P7", "B8_P7"],
            "cells_after": [2, 10],
        }
    )
    output = tmp_path / "REPORT.md"
    write_report_skeleton(output, targets=targets, sensitivity=sensitivity, border_report=border)
    content = output.read_text(encoding="utf-8")

    assert "B8_P7" in content
    assert "4.36500752998%" in content
    assert "B8_P7 target 較其餘八組不穩定" in content
    assert "confidence_flag=low" in content
    assert "458" in content and "387" in content and "median：6" in content
    assert "13 倍" in content
    assert "FOV_IDO" in content and "QC" in content


def test_duplicate_whole_cell_label_rows_keep_master_rows_and_share_border_flag(
    tmp_path: Path,
) -> None:
    from immunity.exp4.border_exclusion import apply_border_exclusion

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    cells = _fill_high_features(pd.concat(
        [
            cells,
            pd.DataFrame(
                [
                    {
                        "image_key": "FOV_A",
                        "cell_label": 1,
                        "nucleus_label": 98,
                        "IDO_score": 9.0,
                    },
                    {
                        "image_key": "FOV_A",
                        "cell_label": 4,
                        "nucleus_label": 99,
                        "IDO_score": 50.0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    ))
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=9,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=2,
        expected_group_keys=("B4_P5",),
    )

    assert len(result.raw_cells) == 9
    assert len(result.pre_cells) == 7
    assert len(result.excluded_cells) == 1
    assert result.excluded_composite_keys == (("FOV_A", 4),)
    assert len(
        result.eligible_cells.loc[
            (result.eligible_cells["image_key"] == "FOV_A")
            & (result.eligible_cells["cell_label"] == 1)
        ]
    ) == 1


def test_mask_and_csv_positive_labels_must_match(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import apply_border_exclusion, BorderExclusionError

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    cells.loc[(cells["image_key"] == "FOV_B") & (cells["cell_label"] == 3), "cell_label"] = 9

    with pytest.raises(BorderExclusionError, match="label|composite"):
        apply_border_exclusion(
            manifest,
            cells,
            masks_dir,
            expected_manifest_keys=2,
            expected_cell_rows=7,
            expected_cell_keys=2,
            expected_dedup_cells=7,
            expected_duplicate_keys=0,
            expected_group_keys=("B4_P5",),
        )


def test_mask_only_labels_are_ignored_but_csv_only_labels_fail_closed(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import apply_border_exclusion

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[1:3, 3:5] = 2
    labels[3:5, 1:3] = 3
    labels[0, 3:5] = 4
    labels[3:5, 3:5] = 5  # mask-only QC object; not in locked cell roster.
    _write_mask(masks_dir, "FOV_A", labels)

    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )

    assert result.mask_only_label_total == 1
    assert result.mask_only_label_max_per_fov == 1
    assert len(result.pre_cells) == 7
    assert len(result.eligible_cells) == 6
    assert result.excluded_composite_keys == (("FOV_A", 4),)


@pytest.mark.parametrize(
    ("expected_manifest_keys", "expected_cell_rows", "expected_cell_keys"),
    [(693, 7, 2), (2, 23976, 2), (2, 7, 693)],
)
def test_snapshot_counts_are_fail_closed_and_parameterized(
    tmp_path: Path,
    expected_manifest_keys: int,
    expected_cell_rows: int,
    expected_cell_keys: int,
) -> None:
    from immunity.exp4.border_exclusion import BorderExclusionError, apply_border_exclusion

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    with pytest.raises(BorderExclusionError, match="expected|count"):
        apply_border_exclusion(
            manifest,
            cells,
            masks_dir,
            expected_manifest_keys=expected_manifest_keys,
            expected_cell_rows=expected_cell_rows,
            expected_cell_keys=expected_cell_keys,
            expected_dedup_cells=7,
            expected_duplicate_keys=0,
            expected_group_keys=("B4_P5",),
        )


def test_after_less_than_three_is_retained_and_targets_are_allowed(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import (
        apply_border_exclusion,
        build_group_targets,
        write_border_exclusion_report,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    cells = cells.loc[
        ~((cells["image_key"] == "FOV_A") & cells["cell_label"].isin((2, 3)))
    ].copy()
    cells = cells.loc[cells["image_key"] != "FOV_B"].copy()
    cells = pd.concat(
        [cells, _cells(("FOV_B",), (1,), (2.0,))], ignore_index=True
    )
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[0, 3:5] = 4
    _write_mask(masks_dir, "FOV_A", labels)
    labels_b = np.zeros((6, 6), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    _write_mask(masks_dir, "FOV_B", labels_b)

    output = tmp_path / "border_exclusion_report.csv"
    target_output = tmp_path / "group_targets.csv"
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=3,
        expected_cell_keys=2,
        expected_dedup_cells=3,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    assert result.minimum_cell_gate_status == "passed"
    assert len(result.border_report) == 2
    write_border_exclusion_report(result, output)
    assert output.exists()
    targets = build_group_targets(result)
    from immunity.exp4.border_exclusion import write_group_targets

    write_group_targets(targets, target_output, expected_group_keys=("B4_P5",))
    assert target_output.exists()


def test_filter_and_y_use_the_same_inclusion_mask_and_targets_are_literal(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import (
        apply_border_exclusion,
        assemble_xy,
        build_group_targets,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    result.pre_cells["feature"] = np.arange(len(result.pre_cells), dtype=float)
    result.eligible_cells["feature"] = result.pre_cells.loc[
        result.eligible_cells.index, "feature"
    ]
    x, y = assemble_xy(result, feature_columns=("feature",))
    targets = build_group_targets(result)

    assert len(x) == len(y) == 6
    assert len(x) == len(result.eligible_cells)
    excluded_feature = result.pre_cells.loc[list(result.excluded_indices), "feature"].iloc[0]
    assert not (x["feature"] == excluded_feature).any()
    # FOV medians: A eligible=(1,3,5)->3, B=(2,4,6)->4; group median=3.5.
    assert targets.loc[0, "group_IDO_score"] == pytest.approx(3.5)
    # Before: A median=4, B median=4; group median=4.
    assert targets.loc[0, "group_IDO_score_all_cells"] == pytest.approx(4.0)
    assert y.tolist() == pytest.approx([3.5] * 6)


def test_label_features_many_to_one_join_preserves_duplicate_master_rows(
    tmp_path: Path,
) -> None:
    from immunity.exp4.border_exclusion import (
        apply_border_exclusion,
        assemble_feature_xy,
        join_label_features,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    cells = _fill_high_features(pd.concat(
        [
            cells,
            pd.DataFrame(
                [
                    {
                        "image_key": "FOV_A",
                        "cell_label": 1,
                        "nucleus_label": 98,
                        "IDO_score": 9.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ))
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=8,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=1,
        expected_group_keys=("B4_P5",),
    )
    features = pd.DataFrame(
        [
            {"image_key": image_key, "cell_label": label, "rui_feature": float(label)}
            for image_key in ("FOV_A", "FOV_B")
            for label in (1, 2, 3, 4) if not (image_key == "FOV_B" and label == 4)
        ]
    )
    joined = join_label_features(result, features)
    x, y = assemble_feature_xy(result, features, feature_columns=("rui_feature",))
    assert len(joined) == len(result.pre_cells) == 7
    assert len(x) == len(y) == len(result.eligible_cells) == 6
    assert x.loc[x["rui_feature"] == 1.0].shape[0] == 2


def test_label_features_extra_key_fails_closed(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import BorderExclusionError, apply_border_exclusion, join_label_features

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    features = pd.DataFrame(
        [
            {"image_key": image_key, "cell_label": label, "rui_feature": float(label)}
            for image_key, labels in (("FOV_A", (1, 2, 3, 4)), ("FOV_B", (1, 2, 3)))
            for label in labels
        ]
        + [{"image_key": "QC_EXTRA", "cell_label": 1, "rui_feature": 99.0}]
    )

    with pytest.raises(BorderExclusionError, match="extra|集合|key"):
        join_label_features(result, features)


def test_many_to_one_join_preserves_three_key_order_and_rejects_nan_feature(
    tmp_path: Path,
) -> None:
    from immunity.exp4.border_exclusion import BorderExclusionError, apply_border_exclusion, join_label_features

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    cells = _fill_high_features(pd.concat(
        [
            cells,
            pd.DataFrame(
                [{"image_key": "FOV_A", "cell_label": 1, "nucleus_label": 98, "IDO_score": 9.0}]
            ),
        ],
        ignore_index=True,
    ))
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=8,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=1,
        expected_group_keys=("B4_P5",),
    )
    features = pd.DataFrame(
        [
            {"image_key": image_key, "cell_label": label, "rui_feature": float(label)}
            for image_key, labels in (("FOV_A", (1, 2, 3, 4)), ("FOV_B", (1, 2, 3)))
            for label in labels
        ]
    )
    joined = join_label_features(result, features, feature_columns=("rui_feature",))
    before_keys = pd.MultiIndex.from_frame(
        result.pre_cells[["image_key", "cell_label", "nucleus_label"]]
    )
    joined_keys = pd.MultiIndex.from_frame(
        joined[["image_key", "cell_label", "nucleus_label"]]
    )
    assert len(joined) == len(result.pre_cells) == 7
    assert joined_keys.equals(before_keys)
    assert joined["IDO_score"].tolist() == result.pre_cells["IDO_score"].tolist()
    assert joined["rui_feature"].notna().all()
    assert joined.loc[
        (joined["image_key"] == "FOV_A") & (joined["cell_label"] == 1), "rui_feature"
    ].tolist() == [1.0]

    features.loc[0, "rui_feature"] = np.nan
    with pytest.raises(BorderExclusionError, match="feature"):
        join_label_features(result, features, feature_columns=("rui_feature",))


def test_report_and_targets_have_deterministic_schema_and_order(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import (
        BORDER_REPORT_COLUMNS,
        GROUP_TARGET_COLUMNS,
        apply_border_exclusion,
        build_group_targets,
        write_border_exclusion_report,
        write_group_targets,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    targets = build_group_targets(result)
    assert list(result.border_report.columns) == list(BORDER_REPORT_COLUMNS)
    assert list(targets.columns) == list(GROUP_TARGET_COLUMNS)
    assert result.border_report["image_key"].tolist() == ["FOV_A", "FOV_B"]

    border_path = tmp_path / "border_exclusion_report.csv"
    target_path = tmp_path / "group_targets.csv"
    write_border_exclusion_report(result, border_path)
    write_group_targets(targets, target_path, expected_group_keys=("B4_P5",))
    assert pd.read_csv(border_path).columns.tolist() == list(BORDER_REPORT_COLUMNS)
    assert pd.read_csv(target_path).columns.tolist() == list(GROUP_TARGET_COLUMNS)


def test_group_target_writer_rejects_wrong_group_count_before_writing(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import (
        GROUP_TARGET_COLUMNS,
        BorderExclusionError,
        write_group_targets,
    )

    targets = pd.DataFrame(
        [
            {
                "group_id": "B4_P5",
                "b_id": "B4",
                "passage": 5,
                "fov_count": 2,
                "cells_before": 7,
                "cells_after": 6,
                "group_IDO_score": 1.0,
                "group_IDO_score_all_cells": 2.0,
                "delta": -1.0,
            }
        ],
        columns=GROUP_TARGET_COLUMNS,
    )
    output = tmp_path / "group_targets.csv"
    with pytest.raises(BorderExclusionError, match="group|row"):
        write_group_targets(targets, output)
    assert not output.exists()


def test_metadata_first_lock_preserves_existing_fields_and_is_idempotent(
    tmp_path: Path,
) -> None:
    from immunity.exp4.border_exclusion import (
        apply_border_exclusion,
        build_group_targets,
        update_border_exclusion_metadata,
        write_border_exclusion_report,
        write_group_targets,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    report_path = tmp_path / "border_exclusion_report.csv"
    target_path = tmp_path / "group_targets.csv"
    write_border_exclusion_report(result, report_path)
    write_group_targets(
        build_group_targets(result), target_path, expected_group_keys=("B4_P5",)
    )
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        '{"status": "snapshot_validated", "packages": {"pandas": "x"}}\n',
        encoding="utf-8",
    )

    first = update_border_exclusion_metadata(
        result,
        metadata_path,
        report_path,
        target_path,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    assert first["cells_after_border_exclusion"] == 6
    assert first["whole_cell_labels_before_border_exclusion"] == 7
    assert first["whole_cell_labels_after_border_exclusion"] == 6
    assert first["cells_excluded_border"] == 1
    assert first["fov_count"] == 2
    assert first["minimum_cell_gate_status"] == "passed"
    assert first["status"] == "snapshot_validated"
    before_second = metadata_path.read_bytes()
    second = update_border_exclusion_metadata(
        result,
        metadata_path,
        report_path,
        target_path,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    assert second["cells_after_border_exclusion"] == first["cells_after_border_exclusion"]
    assert metadata_path.read_bytes() != b""
    assert metadata_path.exists()


def test_metadata_lock_mismatch_fails_without_overwrite(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import (
        BorderExclusionError,
        apply_border_exclusion,
        build_group_targets,
        update_border_exclusion_metadata,
        write_border_exclusion_report,
        write_group_targets,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    report_path = tmp_path / "border.csv"
    target_path = tmp_path / "targets.csv"
    write_border_exclusion_report(result, report_path)
    write_group_targets(
        build_group_targets(result), target_path, expected_group_keys=("B4_P5",)
    )
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        '{"cells_after_border_exclusion": 999, "keep": "original"}\n',
        encoding="utf-8",
    )
    before = metadata_path.read_bytes()
    with pytest.raises(BorderExclusionError, match="lock|mismatch|999"):
        update_border_exclusion_metadata(
            result,
            metadata_path,
            report_path,
            target_path,
            expected_manifest_keys=2,
            expected_cell_rows=7,
            expected_cell_keys=2,
            expected_dedup_cells=7,
            expected_duplicate_keys=0,
            expected_group_keys=("B4_P5",),
        )
    assert metadata_path.read_bytes() == before


def test_metadata_records_distinct_post_lock_for_sparse_fov(tmp_path: Path) -> None:
    from immunity.exp4.border_exclusion import (
        apply_border_exclusion,
        build_group_targets,
        update_border_exclusion_metadata,
        write_border_exclusion_report,
        write_group_targets,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    cells = cells.loc[
        ~((cells["image_key"] == "FOV_A") & cells["cell_label"].isin((2, 3)))
    ].copy()
    cells = cells.loc[cells["image_key"] != "FOV_B"].copy()
    cells = pd.concat([cells, _cells(("FOV_B",), (1,), (2.0,))], ignore_index=True)
    labels_a = np.zeros((6, 6), dtype=np.int32)
    labels_a[1:3, 1:3] = 1
    labels_a[0, 3:5] = 4
    _write_mask(masks_dir, "FOV_A", labels_a)
    labels_b = np.zeros((6, 6), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    _write_mask(masks_dir, "FOV_B", labels_b)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=3,
        expected_cell_keys=2,
        expected_dedup_cells=3,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    report_path = tmp_path / "border.csv"
    write_border_exclusion_report(result, report_path)
    targets_path = tmp_path / "targets.csv"
    write_group_targets(
        build_group_targets(result), targets_path, expected_group_keys=("B4_P5",)
    )
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text('{"status": "snapshot_validated"}\n', encoding="utf-8")

    metadata = update_border_exclusion_metadata(
        result,
        metadata_path,
        report_path,
        targets_path,
        expected_manifest_keys=2,
        expected_cell_rows=3,
        expected_cell_keys=2,
        expected_dedup_cells=3,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    assert metadata["minimum_cell_gate_status"] == "passed"
    assert metadata["border_exclusion_status"] == "validated"
    assert metadata["cells_after_border_exclusion"] == 2
    assert metadata["target_status"] == "validated"


def test_safe_metadata_refreshes_current_border_gate_over_stale_seed(
    tmp_path: Path,
) -> None:
    """current safe metadata 不得沿用 seed generation 的 blocked gate/count。"""
    from immunity.exp4.border_exclusion import apply_border_exclusion
    from immunity.exp4.run_rui2025 import _build_safe_metadata

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=7,
        expected_cell_keys=2,
        expected_dedup_cells=7,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )

    metadata = _build_safe_metadata(
        {
            "border_exclusion_gate_status": "blocked_after_lt_3",
            "minimum_cell_gate_status": "blocked_after_lt_3",
            "raw_cell_count": 999,
            "dedup_cell_count": 999,
            "post_border_cell_count": 999,
            "duplicate_key_count": 999,
            "fov_count": 999,
            "sparse_fov_keys": ["stale-key"],
        },
        result=result,
        output_paths={},
        archived={},
        generation_id="current-generation",
        expected_locks={
            "raw_cell_rows": 7,
            "dedup_cells": 7,
            "post_border_cells": 6,
            "duplicate_keys": 0,
            "fov_count": 2,
            "sparse_fov_count": 0,
        },
        status="validated",
        target_status="validated",
        target_sensitivity_status="validated",
    )

    assert metadata["border_exclusion_gate_status"] == "passed"
    assert metadata["minimum_cell_gate_status"] == "passed"
    assert metadata["raw_cell_count"] == 7
    assert metadata["dedup_cell_count"] == 7
    assert metadata["post_border_cell_count"] == 6
    assert metadata["duplicate_key_count"] == 0
    assert metadata["fov_count"] == 2
    assert metadata["sparse_fov_keys"] == []


def test_metadata_separates_historical_feature_smoke_from_current_targets(
    tmp_path: Path,
) -> None:
    from immunity.exp4.border_exclusion import (
        apply_border_exclusion,
        update_border_exclusion_metadata,
        write_border_exclusion_report,
    )

    manifest, cells, masks_dir = _valid_fixture(tmp_path)
    cells = cells.loc[
        ~((cells["image_key"] == "FOV_A") & cells["cell_label"].isin((2, 3)))
    ].copy()
    cells = cells.loc[cells["image_key"] != "FOV_B"].copy()
    cells = pd.concat([cells, _cells(("FOV_B",), (1,), (2.0,))], ignore_index=True)
    labels_a = np.zeros((6, 6), dtype=np.int32)
    labels_a[1:3, 1:3] = 1
    labels_a[0, 3:5] = 4
    _write_mask(masks_dir, "FOV_A", labels_a)
    labels_b = np.zeros((6, 6), dtype=np.int32)
    labels_b[1:3, 1:3] = 1
    _write_mask(masks_dir, "FOV_B", labels_b)
    result = apply_border_exclusion(
        manifest,
        cells,
        masks_dir,
        expected_manifest_keys=2,
        expected_cell_rows=3,
        expected_cell_keys=2,
        expected_dedup_cells=3,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )
    report_path = tmp_path / "border.csv"
    write_border_exclusion_report(result, report_path)
    old_targets = tmp_path / "old-group-targets.csv"
    old_targets.write_text("historical", encoding="utf-8")
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
                {
                    "keep": "unrelated",
                    "status": "snapshot_validated",
                    "feature_smoke_status": "validated",
                "feature_extraction_status": "smoke_validated",
                "feature_smoke_report": "old-feature-smoke.json",
                "validation_status": {
                    "snapshot": "validated",
                    "feature_smoke": "validated",
                    "other": "preserve",
                },
                "smoke_test": {
                    "status": "passed",
                    "report_path": "old-feature-smoke.json",
                },
                "group_targets_path": str(old_targets),
            }
        ),
        encoding="utf-8",
    )

    metadata = update_border_exclusion_metadata(
        result,
        metadata_path,
        report_path,
        old_targets,
        expected_manifest_keys=2,
        expected_cell_rows=3,
        expected_cell_keys=2,
        expected_dedup_cells=3,
        expected_duplicate_keys=0,
        expected_group_keys=("B4_P5",),
    )

    assert metadata["keep"] == "unrelated"
    assert metadata["status"] == "snapshot_validated"
    assert metadata["feature_smoke_status"] == "historical_stale"
    assert metadata["feature_extraction_status"] == "historical_stale"
    assert metadata["validation_status"]["feature_smoke"] == "historical_stale"
    assert metadata["validation_status"]["snapshot"] == "validated"
    assert metadata["validation_status"]["other"] == "preserve"
    assert metadata["smoke_test"]["status"] == "historical_stale"
    assert metadata["target_status"] == "validated"
    assert metadata["group_targets_path"] == str(old_targets.resolve())
    assert old_targets.read_text(encoding="utf-8") == "historical"
