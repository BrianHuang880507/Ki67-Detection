from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from immunity.exp3 import phase_features
from immunity.exp3.phase_features import (
    PhaseSegmenter,
    cache_phase_masks,
    compare_nucleus_masks,
    load_cached_masks,
    run_development_nucleus_validation,
)


class FakeSegmenter:
    """提供可追蹤呼叫路徑的 synthetic segmentation。"""

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """回傳一組具有單一配對細胞的 masks。"""
        self.paths.append(path)
        cell = np.zeros((12, 12), dtype=np.int32)
        nucleus = np.zeros((12, 12), dtype=np.int32)
        cell[2:10, 2:10] = 1
        nucleus[4:7, 4:7] = 1
        return cell, nucleus


def _manifest_row(tmp_path: Path, image_key: str, group_id: str) -> dict[str, str]:
    """建立不需要真實影像內容的 Exp3 manifest row。"""
    pc_path = tmp_path / f"{image_key}.jpg"
    pc_path.touch()
    return {
        "image_key": image_key,
        "group_id": group_id,
        "pc_path": str(pc_path),
        "ido_path": str(tmp_path / f"{image_key}-ido.jpg"),
    }


def test_cache_phase_masks_uses_only_pc_path(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F01", "B4_P5")]
    )
    fake = FakeSegmenter()

    qc = cache_phase_masks(manifest, tmp_path / "feature_cache", {}, fake)

    assert fake.paths == [Path(manifest.loc[0, "pc_path"])]
    assert qc.loc[0, "cache_status"] == "created"
    assert qc.loc[0, "status"] == "passed"
    assert qc.loc[0, "paired_cells"] == 1
    assert Path(qc.loc[0, "mask_path"]) == (
        tmp_path
        / "feature_cache"
        / "masks"
        / "B4_P5"
        / "B4_P5_C01_F01.npz"
    )
    cell, nucleus = load_cached_masks(qc.loc[0, "mask_path"])
    assert cell.shape == nucleus.shape == (12, 12)


def test_cache_phase_masks_reuses_or_replaces_cache_deterministically(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F02", "B4_P5")]
    )
    original = FakeSegmenter()
    first = cache_phase_masks(manifest, tmp_path / "cache", {}, original)
    cached_path = Path(first.loc[0, "mask_path"])
    first_bytes = cached_path.read_bytes()

    unused = FakeSegmenter()
    reused = cache_phase_masks(manifest, tmp_path / "cache", {}, unused)

    assert unused.paths == []
    assert reused.loc[0, "cache_status"] == "reused"
    assert cached_path.read_bytes() == first_bytes

    replacement = FakeSegmenter()
    replaced = cache_phase_masks(
        manifest,
        tmp_path / "cache",
        {"force": True},
        replacement,
    )
    assert replacement.paths == [Path(manifest.loc[0, "pc_path"])]
    assert replaced.loc[0, "cache_status"] == "replaced"


def test_cache_phase_masks_records_invalid_shape_and_continues(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            _manifest_row(tmp_path, "B4_P5_C01_F03", "B4_P5"),
            _manifest_row(tmp_path, "B4_P5_C01_F04", "B4_P5"),
        ]
    )

    class ShapeFailingSegmenter(FakeSegmenter):
        def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
            if path.name.endswith("F03.jpg"):
                self.paths.append(path)
                return (
                    np.zeros((12, 12), dtype=np.int32),
                    np.zeros((11, 12), dtype=np.int32),
                )
            return super().segment(path)

    fake = ShapeFailingSegmenter()
    qc = cache_phase_masks(manifest, tmp_path / "cache", {}, fake)

    assert qc["status"].tolist() == ["failed", "passed"]
    assert "相同尺寸" in qc.loc[0, "error"]
    assert qc.loc[1, "paired_cells"] == 1
    assert len(fake.paths) == 2


def test_cache_phase_masks_rejects_zero_paired_cells(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F05", "B4_P5")]
    )

    class UnpairedSegmenter:
        def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
            cell = np.zeros((12, 12), dtype=np.int32)
            nucleus = np.zeros((12, 12), dtype=np.int32)
            cell[1:4, 1:4] = 1
            nucleus[8:10, 8:10] = 1
            return cell, nucleus

    qc = cache_phase_masks(
        manifest,
        tmp_path / "cache",
        {},
        UnpairedSegmenter(),
    )

    assert qc.loc[0, "status"] == "failed"
    assert qc.loc[0, "paired_cells"] == 0
    assert "配對細胞數為 0" in qc.loc[0, "error"]
    assert not Path(qc.loc[0, "mask_path"]).exists()


def test_eval_label_mask_uses_required_resize_and_cellpose_arguments() -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    class FakeModel:
        def __init__(self) -> None:
            self.image_shape: tuple[int, ...] | None = None
            self.kwargs: dict[str, object] = {}

        def eval(self, resized_image: np.ndarray, **kwargs: object):
            self.image_shape = resized_image.shape
            self.kwargs = kwargs
            mask = np.zeros((2, 3), dtype=np.int64)
            mask[:, 1:] = 4
            return mask, None, None

    model = FakeModel()

    result = phase_features._eval_label_mask(
        model,
        image,
        (3, 2),
        0.25,
    )

    assert model.image_shape == (2, 3, 3)
    assert model.kwargs == {
        "diameter": None,
        "channels": [0, 0],
        "cellprob_threshold": 0.25,
        "flow_threshold": 0.4,
        "invert": False,
    }
    assert result.shape == (5, 7)
    assert result.dtype == np.int32
    assert set(np.unique(result)) == {0, 4}


def test_phase_segmenter_loads_models_once_and_reads_only_given_pc_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cellpose import io, models
    from ki67dtc import img_prep

    model_paths: list[str] = []
    eval_calls: list[dict[str, object]] = []
    read_paths: list[Path] = []

    class FakeModel:
        def __init__(self, *, gpu: bool, pretrained_model: str) -> None:
            model_paths.append(pretrained_model)

        def eval(self, image: np.ndarray, **kwargs: object):
            eval_calls.append(kwargs)
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            mask[1:-1, 1:-1] = 1
            return mask, None, None

    monkeypatch.setattr(models, "CellposeModel", FakeModel)
    monkeypatch.setattr(
        io,
        "imread",
        lambda path: read_paths.append(Path(path)) or np.zeros((8, 8, 3), np.uint8),
    )
    monkeypatch.setattr(
        img_prep,
        "CYTO_MODEL_INPUT_SIZE",
        (8, 8),
    )
    monkeypatch.setattr(
        img_prep,
        "NUC_MODEL_INPUT_SIZE",
        (8, 8),
    )
    pc_path = tmp_path / "pc.jpg"

    segmenter = PhaseSegmenter({"device": "cpu", "min_area_floor": 0})
    cell, nucleus = segmenter.segment(pc_path)

    assert model_paths == [img_prep.CYTO_MODEL_PATH, img_prep.PC_NUC_MODEL_PATH]
    assert read_paths == [pc_path]
    assert len(eval_calls) == 2
    assert all(call["channels"] == [0, 0] for call in eval_calls)
    assert cell.shape == nucleus.shape == (8, 8)


def test_pc_nucleus_reference_comparison_reports_perfect_overlap() -> None:
    pc = np.zeros((12, 12), dtype=np.int32)
    dapi = np.zeros((12, 12), dtype=np.int32)
    pc[3:8, 4:9] = 1
    dapi[3:8, 4:9] = 7

    comparison = compare_nucleus_masks(pc, dapi)

    assert comparison.loc[0, "pc_label"] == 1
    assert comparison.loc[0, "dapi_label"] == 7
    assert comparison.loc[0, "dice"] == 1.0
    assert comparison.loc[0, "iou"] == 1.0
    assert comparison.loc[0, "area_ratio_pc_to_dapi"] == 1.0
    assert comparison.loc[0, "pc_feret_length"] == comparison.loc[
        0, "dapi_feret_length"
    ]
    assert comparison.loc[0, "matched_cell_coverage"] == 1.0


def test_development_validation_is_disabled_without_touching_input(
    tmp_path: Path,
) -> None:
    result = run_development_nucleus_validation(
        {
            "development_validation": {
                "enabled": False,
                "input_dir": str(tmp_path / "missing-dapi-dataset"),
            }
        },
        tmp_path / "cache",
    )

    assert result.empty


def test_development_validation_uses_pc_and_dapi_models_only_after_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cellpose import io, models
    from ki67dtc import img_prep

    input_dir = tmp_path / "B4 p6"
    filenames = {
        "PC": "IFN-r 0 ng-phase-100X-1.png",
        "DAPI": "IFN-r 0 ng-DAPI-100X-1.png",
        "IDO": "IFN-r 0 ng-IDO-100X-1.png",
    }
    for folder, filename in filenames.items():
        image_path = input_dir / folder / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)

    model_paths: list[str] = []
    eval_channels: list[list[int]] = []

    class FakeModel:
        def __init__(self, *, gpu: bool, pretrained_model: str) -> None:
            model_paths.append(pretrained_model)

        def eval(self, image: np.ndarray, **kwargs: object):
            eval_channels.append(list(kwargs["channels"]))
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            mask[2:6, 2:6] = 1
            return mask, None, None

    monkeypatch.setattr(models, "CellposeModel", FakeModel)
    monkeypatch.setattr(io, "imread", lambda path: np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(img_prep, "NUC_MODEL_INPUT_SIZE", (8, 8))

    result = run_development_nucleus_validation(
        {
            "segmentation": {"device": "cpu"},
            "development_validation": {
                "enabled": True,
                "input_dir": str(input_dir),
                "sample_size": 1,
            },
        },
        tmp_path / "feature_cache",
    )

    assert model_paths == [img_prep.PC_NUC_MODEL_PATH, img_prep.DAPI_NUC_MODEL_PATH]
    assert eval_channels == [[0, 0], [3, 3]]
    assert result["role"].tolist() == ["development_only_dapi_reference"]
    assert result.loc[0, "image_key"] == "IFN0_TNF0_FOV01"
    validation_cache = tmp_path / "feature_cache" / "development_validation"
    assert sorted(path.name for path in validation_cache.glob("*.npz")) == [
        "IFN0_TNF0_FOV01.npz"
    ]
