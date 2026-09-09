from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from immunity.exp6 import gallery as gl
from immunity.exp8 import brightness as br
from immunity.exp8 import donor_response as dr
from immunity.exp8 import notable as nb
from immunity.exp8 import selection as sel
from immunity.exp8 import shape as sh


CONDITIONS = [
    ("IFN0_TNF0", 0.0, 0.0),
    ("IFN25_TNF0", 25.0, 0.0),
    ("IFN100_TNF0", 100.0, 0.0),
    ("IFN0_TNF50", 0.0, 50.0),
]


def _cell_table(seed: int = 3) -> pd.DataFrame:
    """合成 cell-level 表：形狀隨劑量拉長，IDO 隨劑量變亮。"""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for donor_index, donor in enumerate(("B4", "B7", "B8")):
        for passage in (5, 6, 7):
            for condition, ifn, tnf in CONDITIONS:
                for fov in range(1, 5):
                    image_key = f"{donor}_P{passage}_{condition}_F{fov:02d}"
                    # 上限刻意壓在 0.5，讓最高劑量也還有一半細胞是暗的，
                    # 才測得到「同一張影像內同時有亮細胞與暗細胞」。
                    dose_effect = ifn / 200.0
                    for cell in range(1, 9):
                        # 亮暗由劑量與細胞內雜訊共同決定，讓同一張影像內有兩群。
                        lit = rng.random() < dose_effect
                        rows.append(
                            {
                                "image_key": image_key,
                                "cell_label": cell,
                                "b_id": donor,
                                "passage": passage,
                                "group_id": f"{donor}_P{passage}",
                                "condition": condition,
                                "condition_index": CONDITIONS.index((condition, ifn, tnf)) + 1,
                                "ifn_dose": ifn,
                                "tnf_dose": tnf,
                                "pc_path": f"pc/{image_key}.jpg",
                                "ido_path": f"ido/{image_key}.jpg",
                                "IDO_score_ff": (
                                    rng.normal(6.0, 1.0) if lit else rng.normal(0.3, 0.2)
                                ),
                                # 每個形狀特徵都要有雜訊，否則對照組的 IQR 會是 0，
                                # 標準化時整欄變成 NaN。
                                "cell__Area": 1000.0 + 200 * dose_effect + rng.normal(0, 30),
                                "cell__BoundingBoxArea": 1400.0 + 300 * dose_effect + rng.normal(0, 40),
                                "cell__Compactness": 2.0 + dose_effect + rng.normal(0, 0.05),
                                "cell__ConvexArea": 1100.0 + 220 * dose_effect + rng.normal(0, 35),
                                "cell__Eccentricity": 0.7 + 0.2 * dose_effect + rng.normal(0, 0.01),
                                "cell__EquivalentDiameter": 35.0 + 4 * dose_effect + rng.normal(0, 1),
                                "cell__Extent": 0.6 - 0.15 * dose_effect + rng.normal(0, 0.02),
                                "cell__FormFactor": 0.6 - 0.2 * dose_effect + rng.normal(0, 0.02),
                                "cell__MajorAxisLength": 90.0 + 40 * dose_effect + rng.normal(0, 2),
                                "cell__MaxFeretDiameter": 100.0 + 45 * dose_effect + rng.normal(0, 3),
                                "cell__MaximumRadius": 20.0 - 2 * dose_effect + rng.normal(0, 0.6),
                                "cell__MeanRadius": 9.0 - 1.0 * dose_effect + rng.normal(0, 0.3),
                                "cell__MedianRadius": 8.0 - 0.8 * dose_effect + rng.normal(0, 0.3),
                                "cell__MinFeretDiameter": 30.0 - 5 * dose_effect + rng.normal(0, 1),
                                "cell__MinorAxisLength": 28.0 - 6 * dose_effect + rng.normal(0, 1),
                                "cell__Perimeter": 250.0 + 60 * dose_effect + rng.normal(0, 6),
                                "cell__Solidity": 0.9 - 0.05 * dose_effect + rng.normal(0, 0.01),
                                # 非形狀欄位，必須被排除。
                                "cell__Contrast": rng.normal(500, 20),
                                "cell__Center_X": rng.uniform(0, 1280),
                            }
                        )
    return pd.DataFrame(rows)


@pytest.fixture()
def cells() -> pd.DataFrame:
    return _cell_table()


@pytest.fixture()
def fov(cells: pd.DataFrame) -> pd.DataFrame:
    from immunity.exp6.dataset import build_fov_table

    return build_fov_table(cells)


def test_shape_columns_keeps_only_the_seventeen_geometry_features(cells: pd.DataFrame) -> None:
    columns = sh.shape_columns(cells)
    assert len(columns) == 17
    assert "cell__Contrast" not in columns
    assert "cell__Center_X" not in columns
    assert set(columns) == set(sh.SHAPE_COLUMNS)


def test_shape_columns_rejects_a_table_without_shape_features() -> None:
    with pytest.raises(sh.ShapeError):
        sh.shape_columns(pd.DataFrame({"a": [1, 2, 3]}))


def test_partial_spearman_matches_plain_spearman_without_covariates() -> None:
    from scipy.stats import spearmanr

    frame = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6], "y": [2.0, 1, 4, 3, 6, 5]})
    expected, _ = spearmanr(frame["x"], frame["y"])
    rho, _, n = sh.partial_spearman(frame, "x", "y")
    assert rho == pytest.approx(float(expected))
    assert n == 6


def test_partial_spearman_removes_a_shared_driver() -> None:
    """x 與 y 都只由 z 驅動；控制 z 之後相關性必須塌掉。"""
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    frame = pd.DataFrame(
        {"z": z, "x": z + rng.normal(0, 0.05, 400), "y": z + rng.normal(0, 0.05, 400)}
    )
    raw, _, _ = sh.partial_spearman(frame, "x", "y")
    adjusted, _, _ = sh.partial_spearman(frame, "x", "y", ("z",))
    assert raw > 0.9
    assert abs(adjusted) < 0.3


def test_correlate_shape_with_dose_reports_both_versions(fov: pd.DataFrame) -> None:
    table = sh.correlate_shape_with_dose(fov[fov["tnf_dose"] == 0], "ifn_dose", ("n_cells",))
    assert len(table) == 17
    assert list(table["rank"]) == sorted(table["rank"])
    assert {"spearman_rho", "rho_density_adjusted", "retained_fraction"} <= set(table.columns)
    top = table.sort_values("rank").iloc[0]
    assert abs(top["spearman_rho"]) >= abs(table["spearman_rho"]).median()


def test_control_thresholds_come_from_the_unstimulated_condition(cells: pd.DataFrame) -> None:
    thresholds = br.control_thresholds(cells)
    control = cells.loc[cells["condition"] == br.CONTROL_CONDITION, "IDO_score_ff"]
    assert thresholds.dark_max == pytest.approx(float(np.percentile(control, 95)))
    assert thresholds.bright_min == pytest.approx(float(np.percentile(control, 99)))
    assert thresholds.bright_min > thresholds.dark_max
    assert "P95" in thresholds.describe()


def test_control_thresholds_reject_a_tiny_control_group(cells: pd.DataFrame) -> None:
    small = cells[cells["condition"] == br.CONTROL_CONDITION].head(20)
    with pytest.raises(br.BrightnessError):
        br.control_thresholds(small)


def test_label_brightness_splits_into_three_classes(cells: pd.DataFrame) -> None:
    thresholds = br.control_thresholds(cells)
    labelled = br.label_brightness(cells, thresholds)
    dark = labelled[labelled["ido_class"] == br.DARK_LABEL]
    bright = labelled[labelled["ido_class"] == br.BRIGHT_LABEL]
    assert dark["IDO_score_ff"].max() <= thresholds.dark_max
    assert bright["IDO_score_ff"].min() > thresholds.bright_min
    assert set(labelled["ido_class"]) <= {br.DARK_LABEL, br.BRIGHT_LABEL, "灰帶"}


def test_positive_fraction_is_about_five_percent_in_the_control(cells: pd.DataFrame) -> None:
    thresholds = br.control_thresholds(cells)
    labelled = br.label_brightness(cells, thresholds)
    table = br.positive_fraction(labelled, thresholds)
    control = table[table["condition"] == br.CONTROL_CONDITION].iloc[0]
    assert control["positive_fraction"] == pytest.approx(0.05, abs=0.01)
    stimulated = table[table["condition"] == "IFN100_TNF0"].iloc[0]
    assert stimulated["positive_fraction"] > control["positive_fraction"]
    assert (table["positive_fraction_min"] <= table["positive_fraction_max"]).all()


def test_paired_shape_contrast_runs_within_images(cells: pd.DataFrame) -> None:
    thresholds = br.control_thresholds(cells)
    labelled = br.label_brightness(cells, thresholds)
    summary, detail = br.paired_shape_contrast(
        labelled, sh.shape_columns(cells), condition="IFN100_TNF0", min_per_group=2
    )
    assert len(summary) == 17
    assert list(summary["rank"]) == sorted(summary["rank"])
    assert (summary["effect_median"].abs() <= 1.0).all()
    assert detail["image_key"].nunique() == int(summary["n_images"].iloc[0])
    assert (summary["images_same_sign"] <= summary["n_images"]).all()


def test_paired_shape_contrast_rejects_a_condition_without_both_groups(
    cells: pd.DataFrame,
) -> None:
    thresholds = br.control_thresholds(cells)
    labelled = br.label_brightness(cells, thresholds)
    with pytest.raises(br.BrightnessError):
        br.paired_shape_contrast(
            labelled, sh.shape_columns(cells), condition=br.CONTROL_CONDITION, min_per_group=2
        )


def test_images_with_both_groups_spreads_across_donors(cells: pd.DataFrame) -> None:
    thresholds = br.control_thresholds(cells)
    labelled = br.label_brightness(cells, thresholds)
    table = br.images_with_both_groups(labelled, condition="IFN100_TNF0", min_per_group=2)
    assert not table.empty
    assert {"b_id", "n_paired"} <= set(table.columns)
    # 前三名應該來自不同 donor，而不是同一個 donor 連續佔滿。
    assert table.head(3)["b_id"].nunique() >= 2


def test_shape_delta_is_measured_against_each_own_control(fov: pd.DataFrame) -> None:
    delta = dr.shape_delta(fov)
    assert br.CONTROL_CONDITION not in set(delta["condition"])
    assert {"delta", "delta_standardized", "control_value"} <= set(delta.columns)
    row = delta.iloc[0]
    assert row["delta"] == pytest.approx(row["value"] - row["control_value"])
    # 高劑量下長軸長必須變長。
    major = delta[(delta["feature"] == "cell__MajorAxisLength") & (delta["ifn_dose"] == 100)]
    assert major["delta"].median() > 0


def test_shape_delta_rejects_a_block_without_a_control(fov: pd.DataFrame) -> None:
    broken = fov[~((fov["b_id"] == "B4") & (fov["condition"] == br.CONTROL_CONDITION))]
    with pytest.raises(dr.ShapeError, match="缺少未刺激對照"):
        dr.shape_delta(broken)


def test_delta_matrix_has_one_column_per_donor_and_dose(fov: pd.DataFrame) -> None:
    matrix = dr.delta_matrix(dr.shape_delta(fov), axis="ifn")
    assert matrix.shape[0] == 17
    donors = {donor for donor, _ in matrix.columns}
    doses = {dose for _, dose in matrix.columns}
    assert donors == {"B4", "B7", "B8"}
    assert doses == {25.0, 100.0}
    # 列依最大絕對變化排序。
    magnitudes = matrix.abs().max(axis=1).to_numpy()
    assert np.all(np.diff(magnitudes) <= 1e-9)


def test_overall_magnitude_is_positive_and_per_passage(fov: pd.DataFrame) -> None:
    table = dr.overall_magnitude(dr.shape_delta(fov), condition="IFN100_TNF0")
    assert (table["magnitude"] >= 0).all()
    assert set(table["b_id"]) == {"B4", "B7", "B8"}
    assert (table["n_passages"] == 3).all()
    for donor, block in table.groupby("b_id"):
        assert block["median"].iloc[0] == pytest.approx(float(block["magnitude"].median()))


def test_donor_spread_compares_donor_gap_against_the_effect(fov: pd.DataFrame) -> None:
    table = dr.donor_spread(dr.shape_delta(fov), axis="ifn")
    assert (table["donor_range"] >= 0).all()
    assert (table["n_donors"] == 3).all()
    assert {"range_vs_effect", "donor_min", "donor_max"} <= set(table.columns)


def test_representative_cells_picks_the_median_not_the_extreme(cells: pd.DataFrame) -> None:
    block = cells[cells["condition"] == "IFN100_TNF0"]
    chosen = sel.representative_cells(block, ["cell__MajorAxisLength"], count=5)
    assert len(chosen) == 5
    centre = float(block["cell__MajorAxisLength"].median())
    chosen_gap = (chosen["cell__MajorAxisLength"] - centre).abs().max()
    overall_gap = (block["cell__MajorAxisLength"] - centre).abs().max()
    assert chosen_gap < overall_gap


def test_representative_cells_spreads_across_a_key(cells: pd.DataFrame) -> None:
    block = cells[cells["condition"] == "IFN100_TNF0"]
    chosen = sel.representative_cells(
        block, ["cell__MajorAxisLength"], count=3, spread_key="b_id"
    )
    assert chosen["b_id"].nunique() == 3


def test_fixed_window_span_scales_with_the_biggest_cell(cells: pd.DataFrame) -> None:
    span = sel.fixed_window_span(cells)
    largest = float(np.percentile(cells["cell__MaxFeretDiameter"], 95))
    assert span >= largest
    assert span >= 96


def test_fixed_window_span_requires_the_feret_column() -> None:
    with pytest.raises(gl.GalleryError):
        sel.fixed_window_span(pd.DataFrame({"a": [1.0]}))


def _two_cell_bundle() -> gl.ImageBundle:
    """一大一小兩顆細胞，用來檢查固定視窗是否保留相對大小。

    兩顆邊長分別是 100 與 50，真實面積比 4；兩者都夠大，不會碰到
    `crop_cell` 的最小視窗下限，才測得出兩種裁切方式的差別。
    """
    cell_mask = np.zeros((300, 300), dtype=np.int32)
    cell_mask[20:120, 20:120] = 1
    cell_mask[200:250, 200:250] = 2
    nucleus_mask = np.zeros((300, 300), dtype=np.int32)
    nucleus_mask[55:85, 55:85] = 1
    nucleus_mask[215:235, 215:235] = 2
    ido = np.full((300, 300), 5.0, dtype=np.float32)
    ido[cell_mask > 0] = 90.0
    return gl.ImageBundle(
        phase=np.full((300, 300), 40.0, dtype=np.float32),
        ido=ido,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
    )


def test_fixed_span_preserves_relative_cell_size() -> None:
    bundle = _two_cell_bundle()
    big = gl.crop_cell(bundle, 1, padding=1.3, tile_pixels=100, fixed_span=150)
    small = gl.crop_cell(bundle, 2, padding=1.3, tile_pixels=100, fixed_span=150)
    ratio = float(big["mask"].sum()) / float(small["mask"].sum())
    # 真實面積比是 (100/50)^2 = 4。
    assert ratio == pytest.approx(4.0, rel=0.15)


def test_without_fixed_span_relative_size_is_lost() -> None:
    bundle = _two_cell_bundle()
    big = gl.crop_cell(bundle, 1, padding=1.3, tile_pixels=100)
    small = gl.crop_cell(bundle, 2, padding=1.3, tile_pixels=100)
    ratio = float(big["mask"].sum()) / float(small["mask"].sum())
    # 各自貼合外框再縮放，兩顆在畫面上看起來一樣大。
    assert ratio == pytest.approx(1.0, rel=0.2)


def test_fixed_span_pads_at_the_image_border() -> None:
    bundle = _two_cell_bundle()
    crop = gl.crop_cell(bundle, 1, padding=1.3, tile_pixels=None, fixed_span=200)
    # 視窗中心在 (69.5, 69.5)，左上會超出邊界，仍必須回傳完整 200 見方。
    assert crop["ido"].shape == (200, 200)
    assert crop["mask"].shape == (200, 200)


# --- 形狀特別的細胞（fig06 / fig13） -----------------------------------------


@pytest.fixture()
def ranking(fov: pd.DataFrame) -> pd.DataFrame:
    ifn_table = sh.correlate_shape_with_dose(fov[fov["tnf_dose"] == 0], "ifn_dose", ("n_cells",))
    tnf_table = sh.correlate_shape_with_dose(fov[fov["ifn_dose"] == 0], "tnf_dose", ("n_cells",))
    return nb.combined_ranking(ifn_table, tnf_table, top_n=3)


def test_combined_ranking_keeps_only_same_sign_features(ranking: pd.DataFrame) -> None:
    assert len(ranking) == 3
    assert (np.sign(ranking["rho_ifn"]) == np.sign(ranking["rho_tnf"])).all()
    assert list(ranking["combined_rank"]) == [1, 2, 3]
    assert list(ranking["mean_rank"]) == sorted(ranking["mean_rank"])
    assert set(ranking["direction"]) <= {-1, 1}


def test_combined_ranking_rejects_tables_with_no_agreement() -> None:
    left = pd.DataFrame(
        {"feature": ["a"], "feature_label": ["A"], "rank": [1], "spearman_rho": [0.5]}
    )
    right = pd.DataFrame({"feature": ["a"], "rank": [1], "spearman_rho": [-0.5]})
    with pytest.raises(sh.ShapeError, match="同號"):
        nb.combined_ranking(left, right, top_n=1)


def test_build_thresholds_uses_the_control_percentile(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    thresholds = nb.build_thresholds(cells, ranking, percentile=90.0)
    assert len(thresholds) == 3
    control = cells[cells["condition"] == nb.CONTROL_CONDITION]
    for item in thresholds:
        wanted = 90.0 if item.direction > 0 else 10.0
        assert item.threshold == pytest.approx(
            float(np.percentile(control[item.feature], wanted))
        )
        assert "對照組" in item.describe()


def test_threshold_qualifies_excludes_the_artifact_tail(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    item = nb.build_thresholds(cells, ranking, percentile=90.0, artifact_trim=0.05)[0]
    values = cells[item.feature]
    qualifies = item.qualifies(values)
    assert qualifies.any()
    if item.direction > 0:
        assert (values[qualifies] > item.threshold).all()
        # 最極端的那一群被排除，不會出現在挑選結果裡。
        assert not qualifies[values > item.artifact_cutoff].any()
    else:
        assert (values[qualifies] < item.threshold).all()


def test_notable_fraction_is_about_the_percentile_in_the_control(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    thresholds = nb.build_thresholds(cells, ranking, percentile=90.0)
    table = nb.notable_fraction(cells, thresholds)
    assert len(table) == 3 * cells["condition"].nunique()
    control = table[table["condition"] == nb.CONTROL_CONDITION]
    assert control["notable_fraction"].between(0.04, 0.12).all()
    assert (table["notable_fraction_min"] <= table["notable_fraction_max"]).all()
    assert list(table["feature_order"]) == sorted(table["feature_order"])


def test_select_notable_cells_spreads_over_the_qualifying_range(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    item = nb.build_thresholds(cells, ranking, percentile=90.0)[0]
    chosen = nb.select_notable_cells(cells, item, count=8)
    assert len(chosen) == 8
    assert item.qualifies(chosen[item.feature]).all()
    # 等分位取樣：值必須展開，不能全部擠在中位數附近。
    values = chosen["notable_value"].to_numpy()
    qualifying = cells[item.qualifies(cells[item.feature])][item.feature]
    covered = (values.max() - values.min()) / (qualifying.max() - qualifying.min())
    assert covered > 0.5
    assert set(chosen["notable_feature"]) == {item.feature}


def test_select_notable_cells_rejects_an_unreachable_threshold(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    item = nb.build_thresholds(cells, ranking, percentile=90.0)[0]
    impossible = replace(item, threshold=float(cells[item.feature].max()) * 10)
    with pytest.raises(sh.ShapeError, match="門檻"):
        nb.select_notable_cells(cells, impossible, count=4)


def test_select_typical_cells_are_all_below_every_threshold(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    thresholds = nb.build_thresholds(cells, ranking, percentile=90.0)
    chosen = nb.select_typical_cells(cells, thresholds, count=8)
    assert len(chosen) == 8
    assert (chosen["condition"] == nb.CONTROL_CONDITION).all()
    for item in thresholds:
        assert not item.qualifies(chosen[item.feature]).any()


def test_cell_caption_carries_the_cell_id_and_value(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    item = nb.build_thresholds(cells, ranking, percentile=90.0)[0]
    chosen = nb.select_notable_cells(cells, item, count=3)
    row = chosen.iloc[0]
    identifier, condition, value = nb.cell_caption(row)
    assert identifier == f"{row['image_key']} #{int(row['cell_label'])}"
    assert condition == str(row["condition"])
    # 特徵值必須是可讀的數字，且對得回原始值。
    assert float(value) == pytest.approx(float(row["notable_value"]), rel=1e-3)


def test_row_label_shows_the_feature_and_threshold(
    cells: pd.DataFrame, ranking: pd.DataFrame
) -> None:
    item = nb.build_thresholds(cells, ranking, percentile=90.0)[0]
    label = nb.row_label(item)
    assert item.feature_label in label
    assert "\n" in label
    assert (">" if item.direction > 0 else "<") in label
    assert "典型細胞" in nb.row_label(None)
