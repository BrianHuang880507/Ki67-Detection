from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immunity.exp4.nucleus_validation import (
    E2ValidationError,
    load_dapi_label_mask,
    validate_nucleus_dapi,
)


def test_validate_nucleus_dapi_aggregates_hungarian_matches_by_fov_and_group(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        {
            "image_key": ["B4_P5_C01_F01", "B8_P7_C01_F01"],
            "group_id": ["B4_P5", "B8_P7"],
        }
    )
    pc_masks = {
        "B4_P5_C01_F01": np.array(
            [[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 2]], dtype=np.int32
        ),
        "B8_P7_C01_F01": np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32
        ),
    }
    dapi_masks = {
        "B4_P5_C01_F01": np.array(
            [[9, 9, 0, 7], [9, 9, 0, 7], [0, 0, 0, 7]], dtype=np.int32
        ),
        "B8_P7_C01_F01": np.array(
            [[0, 0, 8, 8], [0, 0, 8, 8], [0, 0, 0, 0]], dtype=np.int32
        ),
    }
    paths = {key: tmp_path / f"{key}.npz" for key in dapi_masks}

    result = validate_nucleus_dapi(
        manifest,
        pc_masks_dir=tmp_path / "pc",
        dapi_mask_paths=paths,
        expected_fov_count=2,
        pc_mask_loader=lambda _root, key, _group: pc_masks[key],
        dapi_mask_loader=lambda path: dapi_masks[path.stem],
    )

    assert result.per_fov["image_key"].tolist() == manifest["image_key"].tolist()
    perfect = result.per_fov.set_index("image_key").loc["B4_P5_C01_F01"]
    assert perfect["pc_nucleus_count"] == 2
    assert perfect["dapi_nucleus_count"] == 2
    assert perfect["matched_count"] == 2
    assert perfect["matched_cell_coverage"] == pytest.approx(1.0)
    assert perfect["dice_median"] == pytest.approx(1.0)
    assert perfect["iou_median"] == pytest.approx(1.0)
    assert perfect["status"] == "matched"

    no_overlap = result.per_fov.set_index("image_key").loc["B8_P7_C01_F01"]
    assert no_overlap["matched_count"] == 0
    assert no_overlap["matched_cell_coverage"] == pytest.approx(0.0)
    assert np.isnan(no_overlap["dice_median"])
    assert np.isnan(no_overlap["iou_median"])
    assert no_overlap["status"] == "no_overlap"

    summary = result.summary.set_index("scope")
    assert summary.loc["overall", "fov_count"] == 2
    assert summary.loc["overall", "matched_fov_count"] == 1
    assert summary.loc["overall", "matched_pair_count"] == 2
    assert summary.loc["overall", "dice_median"] == pytest.approx(1.0)
    assert summary.loc["overall", "iou_median"] == pytest.approx(1.0)
    assert set(result.summary["scope"]) == {
        "overall",
        "group:B4_P5",
        "group:B8_P7",
    }


def test_validate_nucleus_dapi_fails_closed_before_loading_when_key_is_missing(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        {
            "image_key": ["B4_P5_C01_F01", "B4_P5_C01_F02"],
            "group_id": ["B4_P5", "B4_P5"],
        }
    )
    loaded: list[str] = []

    with pytest.raises(
        E2ValidationError,
        match=r"DAPI label mask.*B4_P5_C01_F02",
    ):
        validate_nucleus_dapi(
            manifest,
            pc_masks_dir=tmp_path / "pc",
            dapi_mask_paths={
                "B4_P5_C01_F01": tmp_path / "B4_P5_C01_F01.npz"
            },
            expected_fov_count=2,
            pc_mask_loader=lambda _root, key, _group: loaded.append(key),
        )

    assert loaded == []


def test_validate_nucleus_dapi_requires_exact_unique_manifest_scope(
    tmp_path: Path,
) -> None:
    duplicate = pd.DataFrame(
        {
            "image_key": ["B4_P5_C01_F01", "B4_P5_C01_F01"],
            "group_id": ["B4_P5", "B4_P5"],
        }
    )

    with pytest.raises(E2ValidationError, match="image_key.*唯一"):
        validate_nucleus_dapi(
            duplicate,
            pc_masks_dir=tmp_path,
            dapi_mask_paths={"B4_P5_C01_F01": tmp_path / "mask.npz"},
            expected_fov_count=2,
        )


def test_load_dapi_label_mask_accepts_only_precomputed_label_cache(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "dapi.npz"
    expected = np.array([[0, 3], [3, 0]], dtype=np.int32)
    np.savez_compressed(cache, dapi_nucleus_mask=expected)

    actual = load_dapi_label_mask(cache)

    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(E2ValidationError, match="precomputed.*npz"):
        load_dapi_label_mask(tmp_path / "raw-dapi.jpg")


def test_validate_nucleus_dapi_rejects_non_label_values(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {"image_key": ["B4_P5_C01_F01"], "group_id": ["B4_P5"]}
    )

    with pytest.raises(E2ValidationError, match="non-negative integer labels"):
        validate_nucleus_dapi(
            manifest,
            pc_masks_dir=tmp_path,
            dapi_mask_paths={"B4_P5_C01_F01": tmp_path / "dapi.npz"},
            expected_fov_count=1,
            pc_mask_loader=lambda _root, _key, _group: np.zeros(
                (2, 2), dtype=np.int32
            ),
            dapi_mask_loader=lambda _path: np.array(
                [[0.0, 0.5], [0.0, 0.0]], dtype=float
            ),
        )
