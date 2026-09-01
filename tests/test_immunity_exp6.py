from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immunity.exp6 import dataset as ds
from immunity.exp6 import dose_association as da
from immunity.exp6 import gallery as gl


FEATURES = ["cell__Area", "cell__Eccentricity", "cell__MeanIntensity"]


def _feature_frame(n_images: int = 12, cells_per_image: int = 4) -> pd.DataFrame:
    """做出可預測的 cell-level 特徵表，每顆細胞刻意重複兩列。"""
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for image_index in range(n_images):
        image_key = f"IMG{image_index:02d}"
        for cell in range(1, cells_per_image + 1):
            base = {
                "cell__Area": 100.0 + image_index * 10 + cell,
                "cell__Eccentricity": 0.3 + 0.01 * image_index,
                "cell__MeanIntensity": float(rng.normal(0.1, 0.01)),
                "cell__Center_X": float(rng.uniform(0, 1280)),
                "cell__MinIntensity": 0.0,
                "cell__Orientation": float(rng.uniform(-1.5, 1.5)),
            }
            for nucleus_label in (cell, cell + 100):
                rows.append(
                    {
                        "image_key": image_key,
                        "cell_label": cell,
                        "nucleus_label": nucleus_label,
                        **base,
                    }
                )
    return pd.DataFrame(rows)


def _ido_frame(features: pd.DataFrame) -> pd.DataFrame:
    """IDO 表與特徵表共用 key，兩列同值以測試中位數收斂。"""
    frame = features[["image_key", "cell_label", "nucleus_label"]].copy()
    image_index = frame["image_key"].str.removeprefix("IMG").astype(int)
    frame["IDO_score_ff"] = image_index * 0.5 + frame["cell_label"] * 0.1
    frame["IDO_score_scalar"] = frame["IDO_score_ff"] + 1.0
    return frame


def _manifest_frame(n_images: int = 12) -> pd.DataFrame:
    """四種 IFN×TNF 組合輪流指派，兩個 donor 各一個 passage。"""
    conditions = [
        ("IFN0_TNF0", 0.0, 0.0, 4),
        ("IFN25_TNF0", 25.0, 0.0, 3),
        ("IFN0_TNF25", 0.0, 25.0, 5),
        ("IFN25_TNF25", 25.0, 25.0, 7),
    ]
    rows: list[dict[str, object]] = []
    for image_index in range(n_images):
        condition, ifn, tnf, condition_index = conditions[image_index % len(conditions)]
        donor = "B4" if image_index % 2 == 0 else "B7"
        rows.append(
            {
                "image_key": f"IMG{image_index:02d}",
                "b_id": donor,
                "passage": 5,
                "group_id": f"{donor}_P5",
                "condition": condition,
                "condition_index": condition_index,
                "ifn_dose": ifn,
                "tnf_dose": tnf,
                "pc_path": f"pc/{image_index}.jpg",
                "ido_path": f"ido/{image_index}.jpg",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def cell_table(tmp_path: Path) -> pd.DataFrame:
    features = _feature_frame()
    paths = ds.InputPaths(
        features=tmp_path / "features.csv",
        ido=tmp_path / "ido.csv",
        manifest=tmp_path / "manifest.csv",
    )
    features.to_csv(paths.features, index=False)
    _ido_frame(features).to_csv(paths.ido, index=False)
    _manifest_frame().to_csv(paths.manifest, index=False)
    return ds.load_cell_table(paths)


def test_feature_columns_drops_position_constant_and_circular() -> None:
    frame = _feature_frame()
    kept = ds.feature_columns(frame)
    assert set(kept) == set(FEATURES)
    for excluded in ("cell__Center_X", "cell__MinIntensity", "cell__Orientation"):
        assert excluded not in kept
    assert excluded in ds.feature_columns(frame, drop_excluded=False)


def test_load_cell_table_collapses_duplicate_nucleus_rows(cell_table: pd.DataFrame) -> None:
    assert len(cell_table) == 12 * 4
    assert not cell_table.duplicated(subset=ds.CELL_KEY).any()
    assert {"ifn_dose", "tnf_dose", "IDO_score_ff", "pc_path"} <= set(cell_table.columns)


def test_load_cell_table_reports_missing_inputs(tmp_path: Path) -> None:
    paths = ds.InputPaths(
        features=tmp_path / "nope_features.csv",
        ido=tmp_path / "nope_ido.csv",
        manifest=tmp_path / "nope_manifest.csv",
    )
    with pytest.raises(ds.DatasetError, match="缺少 Exp6 輸入檔案"):
        ds.load_cell_table(paths)


def test_build_fov_table_is_one_row_per_image(cell_table: pd.DataFrame) -> None:
    fov = ds.build_fov_table(cell_table)
    assert len(fov) == cell_table["image_key"].nunique()
    assert (fov["n_cells"] == 4).all()
    first = fov.loc[fov["image_key"] == "IMG00", "cell__Area"].iloc[0]
    assert first == pytest.approx(np.median([101.0, 102.0, 103.0, 104.0]))


def test_dose_slices_keep_only_interpretable_axes(cell_table: pd.DataFrame) -> None:
    slices = ds.dose_slices(ds.build_fov_table(cell_table))
    assert (slices["ifn"]["tnf_dose"] == 0).all()
    assert set(slices["ifn"]["ifn_dose"]) == {0.0, 25.0}
    assert set(slices["tnf"]["ifn_dose"]) <= {0.0, 25.0}


def test_stratified_spearman_matches_plain_spearman_without_strata() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 1.0, 4.0, 3.0, 5.0]})
    from scipy.stats import spearmanr

    expected, _ = spearmanr(frame["x"], frame["y"])
    rho, _, n = da.stratified_spearman(frame, "x", "y")
    assert rho == pytest.approx(float(expected))
    assert n == 5


def test_stratified_spearman_removes_a_block_confound() -> None:
    """兩個區塊內 x 與 y 反向，但區塊平均值同向；分層後 rho 必須翻負。"""
    frame = pd.DataFrame(
        {
            "block": ["A"] * 4 + ["B"] * 4,
            "x": [1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0],
            "y": [4.0, 3.0, 2.0, 1.0, 14.0, 13.0, 12.0, 11.0],
        }
    )
    pooled, _, _ = da.stratified_spearman(frame, "x", "y")
    within, _, _ = da.stratified_spearman(frame, "x", "y", ("block",))
    assert pooled > 0.4
    assert within == pytest.approx(-1.0)


def test_stratified_spearman_returns_nan_for_constant_column() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [7.0, 7.0, 7.0, 7.0]})
    rho, p_value, _ = da.stratified_spearman(frame, "x", "y")
    assert np.isnan(rho)
    assert np.isnan(p_value)


def test_benjamini_hochberg_is_monotone_and_bounded() -> None:
    p_values = np.array([0.001, 0.01, 0.04, 0.2, np.nan])
    q_values = da.benjamini_hochberg(p_values)
    finite = q_values[:-1]
    assert np.isnan(q_values[-1])
    assert np.all(finite >= p_values[:-1])
    assert np.all(finite <= 1.0)
    assert np.all(np.diff(finite) >= -1e-12)


def test_correlate_ido_with_dose_splits_tnf_by_ifn_block(cell_table: pd.DataFrame) -> None:
    slices = ds.dose_slices(ds.build_fov_table(cell_table))
    table = da.correlate_ido_with_dose(slices)
    tnf = table[(table["dose_axis"] == "TNF") & (table["target"] == "IDO_score_ff")]
    assert set(tnf["block"]) == {"全部", "IFN-γ = 0", "IFN-γ = 25"}
    assert {"median_at_min", "median_at_max", "delta_grey_levels"} <= set(table.columns)


def test_correlate_features_with_dose_ranks_and_adjusts(cell_table: pd.DataFrame) -> None:
    fov = ds.build_fov_table(cell_table)
    slices = ds.dose_slices(fov)
    table = da.correlate_features_with_dose(slices, ds.feature_columns(cell_table))
    assert set(table["dose_axis"]) == {"IFN", "TNF"}
    for _, block in table.groupby("dose_axis"):
        assert list(block["rank"]) == sorted(block["rank"])
        ordered = block.sort_values("rank")["abs_rho"].dropna().to_numpy()
        assert np.all(np.diff(ordered) <= 1e-12)
    assert (table["q_value_bh"].dropna() <= 1.0).all()


def test_top_features_returns_requested_count(cell_table: pd.DataFrame) -> None:
    slices = ds.dose_slices(ds.build_fov_table(cell_table))
    table = da.correlate_features_with_dose(slices, ds.feature_columns(cell_table))
    top = da.top_features(table, "IFN", top_n=2)
    assert len(top) == 2
    assert list(top["rank"]) == [1, 2]


def test_assign_brightness_groups_global_and_stratified(cell_table: pd.DataFrame) -> None:
    plain = gl.assign_brightness_groups(cell_table, decile_pct=25.0, strata=None)
    assert set(plain["brightness_group"]) == {"bright", "dim", "mid"}
    assert plain.loc[plain["brightness_group"] == "bright", "IDO_score_ff"].min() > plain.loc[
        plain["brightness_group"] == "dim", "IDO_score_ff"
    ].max()

    stratified = gl.assign_brightness_groups(
        cell_table, decile_pct=25.0, strata=["condition_index"]
    )
    # 分層後每個條件都必須各自有亮組，不會整個條件被判成暗。
    per_condition = stratified.groupby("condition_index")["brightness_group"].apply(
        lambda values: "bright" in set(values)
    )
    assert per_condition.all()


def test_contrast_features_sign_follows_the_bright_group(cell_table: pd.DataFrame) -> None:
    labelled = gl.assign_brightness_groups(cell_table, decile_pct=25.0, strata=None)
    table = gl.contrast_features(labelled, ds.feature_columns(cell_table), decile_pct=25.0)
    row = table[table["feature"] == "cell__Area"].iloc[0]
    assert row["median_bright"] > row["median_dim"]
    assert row["rank_biserial"] > 0
    assert (table["rank_biserial"].abs() <= 1.0).all()


def test_contrast_features_rejects_an_empty_group(cell_table: pd.DataFrame) -> None:
    labelled = gl.assign_brightness_groups(cell_table, decile_pct=25.0, strata=None)
    labelled["brightness_group"] = "mid"
    with pytest.raises(gl.GalleryError):
        gl.contrast_features(labelled, ds.feature_columns(cell_table), decile_pct=25.0)


def test_sample_cells_is_deterministic_and_capped(cell_table: pd.DataFrame) -> None:
    labelled = gl.assign_brightness_groups(cell_table, decile_pct=40.0, strata=None)
    first = gl.sample_cells(labelled, "bright", 5, seed=0, strata=None)
    second = gl.sample_cells(labelled, "bright", 5, seed=0, strata=None)
    assert len(first) == 5
    assert list(first["cell_label"]) == list(second["cell_label"])


def test_crop_cell_masks_everything_outside_the_label() -> None:
    cell_mask = np.zeros((60, 60), dtype=np.int32)
    cell_mask[20:40, 25:35] = 3
    phase = np.full((60, 60), 50.0, dtype=np.float32)
    ido = np.full((60, 60), 7.0, dtype=np.float32)
    ido[20:40, 25:35] = 200.0
    bundle = gl.ImageBundle(phase=phase, ido=ido, cell_mask=cell_mask)

    crop = gl.crop_cell(bundle, 3, padding=1.4, tile_pixels=48)
    assert crop["ido"].shape == (48, 48)
    outside = crop["mask"] < 0.5
    assert np.allclose(crop["ido_segmented"][outside], 0.0)
    assert crop["ido_segmented"][crop["mask"] > 0.5].max() == pytest.approx(200.0)


def test_crop_cell_rejects_a_missing_label() -> None:
    bundle = gl.ImageBundle(
        phase=np.zeros((10, 10), dtype=np.float32),
        ido=np.zeros((10, 10), dtype=np.float32),
        cell_mask=np.zeros((10, 10), dtype=np.int32),
    )
    with pytest.raises(gl.GalleryError, match="cell_label"):
        gl.crop_cell(bundle, 9, padding=1.2, tile_pixels=16)


def test_load_image_bundle_reports_a_missing_mask(tmp_path: Path) -> None:
    row = pd.Series({"group_id": "B4_P5", "image_key": "IMG99", "pc_path": "x", "ido_path": "y"})
    with pytest.raises(gl.GalleryError, match="mask cache"):
        gl.load_image_bundle(tmp_path, row)


def test_render_report_contains_both_questions(cell_table: pd.DataFrame) -> None:
    from immunity.exp6.reporting import render_report

    fov = ds.build_fov_table(cell_table)
    slices = ds.dose_slices(fov)
    tables = da.build_all_tables(cell_table, fov, slices)
    summary = ds.describe_dataset(cell_table, fov)
    report = render_report(
        summary=summary,
        slices=slices,
        tables=tables,
        splits={},
        top_n=3,
        decile_pct=10.0,
        metadata={"generated_at": "2026-09-01T00:00:00+08:00", "elapsed_seconds": 1.0},
    )
    assert "問題 1a" in report
    assert "問題 1b" in report
    assert "限制" in report
    assert "flat-field" in report
