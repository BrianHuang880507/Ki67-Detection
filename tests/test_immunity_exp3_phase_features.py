from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from immunity.exp3 import phase_features
from immunity.exp3.phase_features import (
    PhaseSegmenter,
    cache_phase_masks,
    compare_nucleus_masks,
    extract_basic_cell_features,
    extract_features_from_arrays,
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


def _corrupt_stored_npz_member(mask_path: Path, member_name: str) -> None:
    """翻轉未壓縮 ZIP member 的資料位元，保留結構但破壞 CRC。"""
    with zipfile.ZipFile(mask_path) as archive:
        member = archive.getinfo(member_name)
        assert member.compress_type == zipfile.ZIP_STORED
    with mask_path.open("r+b") as stream:
        stream.seek(member.header_offset)
        local_header = stream.read(30)
        filename_length = int.from_bytes(local_header[26:28], "little")
        extra_length = int.from_bytes(local_header[28:30], "little")
        last_data_byte = (
            member.header_offset
            + 30
            + filename_length
            + extra_length
            + member.file_size
            - 1
        )
        stream.seek(last_data_byte)
        original = stream.read(1)
        stream.seek(last_data_byte)
        stream.write(bytes([original[0] ^ 0xFF]))


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


def test_cache_phase_masks_rejects_path_components_before_any_cache_access(
    tmp_path: Path,
) -> None:
    """Traversal component 不得觸及 masks root 外的檔案。"""
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F01", "../../../escaped")]
    )
    cache_dir = tmp_path / "feature_cache"
    escaped = (cache_dir / "masks" / "../../../escaped").resolve()
    fake = FakeSegmenter()

    with pytest.raises(ValueError, match="group_id"):
        cache_phase_masks(manifest, cache_dir, {}, fake)

    assert fake.paths == []
    assert not escaped.exists()


def test_cache_phase_masks_rejects_linked_group_before_segmentation(
    tmp_path: Path,
) -> None:
    """Group junction/symlink 不得將 cache 寫到 masks root 外。"""
    cache_dir = tmp_path / "feature_cache"
    masks_root = cache_dir / "masks"
    outside = tmp_path / "outside"
    masks_root.mkdir(parents=True)
    outside.mkdir()
    _create_directory_link(masks_root / "B4_P5", outside)
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F01", "B4_P5")]
    )
    fake = FakeSegmenter()

    with pytest.raises(ValueError, match="mask_path"):
        cache_phase_masks(manifest, cache_dir, {}, fake)

    assert fake.paths == []
    assert not (outside / "B4_P5_C01_F01.npz").exists()


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


def test_cache_phase_masks_replaces_cache_when_pc_bytes_change(
    tmp_path: Path,
) -> None:
    """相同 image key 的 PC 內容改變時不得重用舊 masks。"""
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F08", "B4_P5")]
    )
    pc_path = Path(manifest.loc[0, "pc_path"])
    pc_path.write_bytes(b"first")
    cache_phase_masks(manifest, tmp_path / "cache", {}, FakeSegmenter())
    pc_path.write_bytes(b"second")
    replacement = FakeSegmenter()

    qc = cache_phase_masks(manifest, tmp_path / "cache", {}, replacement)

    assert replacement.paths == [pc_path]
    assert qc.loc[0, "cache_status"] == "replaced"
    assert qc.loc[0, "cache_reason"] == "pc_content_changed"
    assert len(qc.loc[0, "pc_sha256"]) == 64
    assert len(qc.loc[0, "cache_provenance_hash"]) == 64


def test_cache_phase_masks_replaces_cache_when_segmentation_config_changes(
    tmp_path: Path,
) -> None:
    """影響 segmentation 的 threshold 改變時不得重用舊 masks。"""
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F09", "B4_P5")]
    )
    cache_phase_masks(
        manifest,
        tmp_path / "cache",
        {"cellprob_threshold": 0.0},
        FakeSegmenter(),
    )
    replacement = FakeSegmenter()

    qc = cache_phase_masks(
        manifest,
        tmp_path / "cache",
        {"cellprob_threshold": 0.5},
        replacement,
    )

    assert replacement.paths == [Path(manifest.loc[0, "pc_path"])]
    assert qc.loc[0, "cache_status"] == "replaced"
    assert qc.loc[0, "cache_reason"] == "segmentation_config_changed"


def test_cache_phase_masks_replaces_cache_when_segmenter_signature_changes(
    tmp_path: Path,
) -> None:
    """Injected segmenter 的明確穩定 signature 改變時不得重用。"""

    class SignedSegmenter(FakeSegmenter):
        def __init__(self, signature: str) -> None:
            super().__init__()
            self.cache_signature = signature

    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F10", "B4_P5")]
    )
    cache_phase_masks(
        manifest,
        tmp_path / "cache",
        {},
        SignedSegmenter("model-v1"),
    )
    replacement = SignedSegmenter("model-v2")

    qc = cache_phase_masks(manifest, tmp_path / "cache", {}, replacement)

    assert replacement.paths == [Path(manifest.loc[0, "pc_path"])]
    assert qc.loc[0, "cache_status"] == "replaced"
    assert qc.loc[0, "cache_reason"] == "segmenter_signature_changed"


def test_cache_phase_masks_replaces_legacy_cache_without_provenance(
    tmp_path: Path,
) -> None:
    """沒有 provenance 的 legacy NPZ 不可視為可重用 cache。"""
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F11", "B4_P5")]
    )
    mask_path = (
        tmp_path / "cache" / "masks" / "B4_P5" / "B4_P5_C01_F11.npz"
    )
    mask_path.parent.mkdir(parents=True)
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:10, 2:10] = 1
    np.savez(mask_path, cell_mask=labels, nucleus_mask=labels)
    replacement = FakeSegmenter()

    qc = cache_phase_masks(manifest, tmp_path / "cache", {}, replacement)

    assert replacement.paths == [Path(manifest.loc[0, "pc_path"])]
    assert qc.loc[0, "cache_status"] == "replaced"
    assert qc.loc[0, "cache_reason"] == "legacy_cache"
    with np.load(mask_path, allow_pickle=False) as cached:
        assert {"provenance_json", "provenance_hash"} <= set(cached.files)


def test_cache_phase_masks_replaces_crc_corrupt_cache(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [_manifest_row(tmp_path, "B4_P5_C01_F03", "B4_P5")]
    )
    mask_path = (
        tmp_path / "cache" / "masks" / "B4_P5" / "B4_P5_C01_F03.npz"
    )
    mask_path.parent.mkdir(parents=True)
    valid = np.zeros((12, 12), dtype=np.int32)
    np.savez(mask_path, cell_mask=valid, nucleus_mask=valid)
    _corrupt_stored_npz_member(mask_path, "cell_mask.npy")
    with pytest.raises(zipfile.BadZipFile, match="Bad CRC-32"):
        load_cached_masks(mask_path)
    replacement = FakeSegmenter()

    qc = cache_phase_masks(manifest, tmp_path / "cache", {}, replacement)

    assert replacement.paths == [Path(manifest.loc[0, "pc_path"])]
    assert qc.loc[0, "cache_status"] == "replaced"
    assert qc.loc[0, "status"] == "passed"
    cell, nucleus = load_cached_masks(mask_path)
    assert cell.shape == nucleus.shape == (12, 12)


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


def test_cache_phase_masks_records_inference_error_and_continues(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            _manifest_row(tmp_path, "B4_P5_C01_F06", "B4_P5"),
            _manifest_row(tmp_path, "B4_P5_C01_F07", "B4_P5"),
        ]
    )

    class InferenceFailingSegmenter(FakeSegmenter):
        def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
            if path.name.endswith("F06.jpg"):
                self.paths.append(path)
                raise RuntimeError("synthetic inference failure")
            return super().segment(path)

    fake = InferenceFailingSegmenter()

    qc = cache_phase_masks(manifest, tmp_path / "cache", {}, fake)

    assert qc["status"].tolist() == ["failed", "passed"]
    assert qc.loc[0, "error"] == "synthetic inference failure"
    assert qc.loc[1, "error"] == ""
    assert fake.paths == [
        Path(manifest.loc[0, "pc_path"]),
        Path(manifest.loc[1, "pc_path"]),
    ]


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


@pytest.mark.parametrize("label_side", ["pc", "dapi", "neither"])
def test_pc_nucleus_reference_comparison_keeps_zero_label_summary(
    label_side: str,
) -> None:
    pc = np.zeros((12, 12), dtype=np.int32)
    dapi = np.zeros((12, 12), dtype=np.int32)
    if label_side == "pc":
        pc[2:5, 2:5] = 1
    elif label_side == "dapi":
        dapi[7:10, 7:10] = 7

    comparison = compare_nucleus_masks(pc, dapi)

    assert len(comparison) == 1
    assert pd.isna(comparison.loc[0, "pc_label"])
    assert pd.isna(comparison.loc[0, "dapi_label"])
    assert pd.isna(comparison.loc[0, "dice"])
    assert pd.isna(comparison.loc[0, "iou"])
    assert comparison.loc[0, "matched_cell_coverage"] == 0.0


def test_pc_nucleus_reference_comparison_keeps_zero_overlap_summary() -> None:
    pc = np.zeros((12, 12), dtype=np.int32)
    dapi = np.zeros((12, 12), dtype=np.int32)
    pc[1:4, 1:4] = 1
    dapi[8:11, 8:11] = 7

    comparison = compare_nucleus_masks(pc, dapi)

    assert len(comparison) == 1
    assert comparison.loc[0, "matched_cell_coverage"] == 0.0
    empty_metrics = comparison.loc[
        0,
        ["pc_label", "dapi_label", "dice", "iou"],
    ]
    assert empty_metrics.isna().all()


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
    assert result.loc[0, "matched_cell_coverage"] == 0.0
    assert pd.isna(result.loc[0, "pc_label"])
    assert pd.isna(result.loc[0, "dapi_label"])
    validation_cache = tmp_path / "feature_cache" / "development_validation"
    assert sorted(path.name for path in validation_cache.glob("*.npz")) == [
        "IFN0_TNF0_FOV01.npz"
    ]


def test_feature_extraction_uses_true_cytoplasm_for_ido_target() -> None:
    phase = np.arange(100, dtype=float).reshape(10, 10)
    ido = np.full((10, 10), 5.0)
    cell = np.zeros((10, 10), dtype=np.int32)
    nucleus = np.zeros((10, 10), dtype=np.int32)
    cell[2:8, 2:8] = 1
    nucleus[4:6, 4:6] = 1
    ido[cell == 1] = 20.0
    ido[nucleus == 1] = 100.0

    rows, qc = extract_features_from_arrays(
        "img", phase, ido, cell, nucleus, 0.05
    )

    assert len(rows) == 1
    assert rows[0]["IDO_score"] == pytest.approx(15.0)
    assert rows[0]["nucleus_cytoplasm_area_ratio"] == pytest.approx(4 / 32)
    assert qc["kept_full_cells"] == 1


def test_feature_extraction_records_outside_nucleus_and_empty_cytoplasm() -> None:
    phase = np.ones((12, 12), dtype=float)
    ido = np.zeros((12, 12), dtype=float)
    cell = np.zeros((12, 12), dtype=np.int32)
    nucleus = np.zeros((12, 12), dtype=np.int32)

    cell[2:7, 2:7] = 1
    nucleus[3:5, 3:5] = 1
    nucleus[9, 9] = 1
    cell[8, 2] = 2
    nucleus[8, 2] = 2

    rows, qc = extract_features_from_arrays(
        "img", phase, ido, cell, nucleus, 0.05
    )

    assert rows == []
    assert qc["paired_cells"] == 2
    assert qc["excluded_nucleus_outside"] == 1
    assert qc["excluded_empty_cytoplasm"] == 1
    assert qc["kept_full_cells"] == 0


def test_feature_extraction_rejects_mismatched_array_dimensions() -> None:
    phase = np.zeros((8, 8), dtype=float)
    ido = np.zeros((8, 7), dtype=float)
    mask = np.zeros((8, 8), dtype=np.int32)

    with pytest.raises(ValueError, match="相同尺寸"):
        extract_features_from_arrays("img", phase, ido, mask, mask, 0.05)


def test_extract_basic_cell_features_writes_internal_cell_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    image_key = "B4_P5_C01_F01"
    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    phase_path = tmp_path / "phase.png"
    ido_path = tmp_path / "ido.png"
    Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10)).save(phase_path)
    Image.fromarray(np.full((10, 10), 5, dtype=np.uint8)).save(ido_path)

    cell = np.zeros((10, 10), dtype=np.int32)
    nucleus = np.zeros((10, 10), dtype=np.int32)
    cell[2:8, 2:8] = 1
    nucleus[4:6, 4:6] = 1
    mask_path = (
        output_dir
        / "feature_cache"
        / "masks"
        / "B4_P5"
        / f"{image_key}.npz"
    )
    mask_path.parent.mkdir(parents=True)
    np.savez(mask_path, cell_mask=cell, nucleus_mask=nucleus)
    manifest = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "b_id": "B4",
                "passage": 5,
                "group_id": "B4_P5",
                "condition_index": 1,
                "condition": "control",
                "ifn_dose": 0.0,
                "tnf_dose": 0.0,
                "fov": 1,
                "pc_path": str(phase_path),
                "ido_path": str(ido_path),
            }
        ]
    )
    segmentation_qc = pd.DataFrame(
        [{"image_key": image_key, "mask_path": str(mask_path), "status": "passed"}]
    )

    cells, qc = extract_basic_cell_features(
        manifest,
        segmentation_qc,
        {
            "_output_dir": str(output_dir),
            "segmentation": {
                "max_nucleus_outside_fraction": 0.05,
                "min_cells_per_image": 1,
            },
            "feature_sets": {"enabled": ["basic_median"]},
        },
    )

    assert len(cells) == 1
    assert qc.loc[0, "status"] == "passed"
    assert cells.attrs["min_cells_per_image"] == 1
    assert (
        output_dir / "feature_cache" / "cell_level_basic.csv"
    ).is_file()


def test_extract_basic_cell_features_rejects_legacy_output_destination(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        columns=["image_key", "group_id", "pc_path", "ido_path"]
    )
    segmentation_qc = pd.DataFrame(
        columns=["image_key", "mask_path", "status"]
    )
    legacy_dir = tmp_path / "legacy-results"

    with pytest.raises(ValueError, match="Exp3 output"):
        extract_basic_cell_features(
            manifest,
            segmentation_qc,
            {"_output_dir": str(legacy_dir)},
        )

    assert not legacy_dir.exists()


def test_extract_basic_cell_features_rejects_repository_root_destination() -> None:
    from immunity.exp3.run_benchmark import PROJECT_ROOT

    manifest = pd.DataFrame(
        columns=["image_key", "group_id", "pc_path", "ido_path"]
    )
    segmentation_qc = pd.DataFrame(
        columns=["image_key", "mask_path", "status"]
    )

    with pytest.raises(ValueError, match="Exp3 output"):
        extract_basic_cell_features(
            manifest,
            segmentation_qc,
            {"_output_dir": str(PROJECT_ROOT)},
        )


def test_extract_basic_cell_features_rejects_na_output_without_writing_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = pd.DataFrame(
        columns=["image_key", "group_id", "pc_path", "ido_path"]
    )
    segmentation_qc = pd.DataFrame(
        columns=["image_key", "mask_path", "status"]
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="_output_dir"):
        extract_basic_cell_features(
            manifest,
            segmentation_qc,
            {"_output_dir": pd.NA},
        )

    assert not (tmp_path / "cell_level_basic.csv").exists()


@pytest.mark.parametrize("invalid_mask_path", [pd.NA, "", 123])
def test_extract_basic_cell_features_rejects_invalid_mask_cache_path(
    invalid_mask_path: object,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    image_key = "B4_P5_C01_F01"
    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    monkeypatch.chdir(tmp_path)
    phase_path = tmp_path / "phase.png"
    ido_path = tmp_path / "ido.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(phase_path)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(ido_path)
    manifest = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "group_id": "B4_P5",
                "pc_path": str(phase_path),
                "ido_path": str(ido_path),
            }
        ]
    )
    segmentation_qc = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "mask_path": invalid_mask_path,
                "status": "passed",
            }
        ]
    )

    with pytest.raises(ValueError, match="mask_path"):
        extract_basic_cell_features(
            manifest,
            segmentation_qc,
            {"_output_dir": str(output_dir)},
        )

    assert not (tmp_path / "cell_level_basic.csv").exists()


def test_extract_basic_cell_features_rejects_unexpected_mask_cache_structure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    image_key = "B4_P5_C01_F01"
    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    manifest = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "group_id": "B4_P5",
                "pc_path": str(tmp_path / "phase.png"),
                "ido_path": str(tmp_path / "ido.png"),
            }
        ]
    )
    segmentation_qc = pd.DataFrame(
        [
            {
                "image_key": image_key,
                "mask_path": str(tmp_path / "legacy" / f"{image_key}.npz"),
                "status": "passed",
            }
        ]
    )

    with pytest.raises(ValueError, match="mask_path"):
        extract_basic_cell_features(
            manifest,
            segmentation_qc,
            {"_output_dir": str(output_dir)},
        )

    assert not (tmp_path / "legacy").exists()


@pytest.mark.parametrize(
    ("component_name", "invalid_value"),
    [
        ("group_id", "."),
        ("group_id", ".."),
        ("group_id", " B4_P5"),
        ("group_id", "B4_P5 "),
        ("group_id", "B4/P5"),
        ("group_id", "B4\\P5"),
        ("group_id", "C:B4_P5"),
        ("image_key", "."),
        ("image_key", ".."),
        ("image_key", " B4_P5_C01_F01"),
        ("image_key", "B4_P5_C01_F01 "),
        ("image_key", "B4/P5_C01_F01"),
        ("image_key", "B4\\P5_C01_F01"),
        ("image_key", "C:B4_P5_C01_F01"),
    ],
)
def test_extract_basic_cell_features_rejects_unsafe_cache_components(
    component_name: str,
    invalid_value: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from immunity.exp3 import run_benchmark

    output_root = tmp_path / "immunity" / "outputs" / "exp3"
    output_dir = output_root / "test-run"
    monkeypatch.setattr(run_benchmark, "EXP3_OUTPUT_ROOT", output_root.resolve())
    values = {
        "group_id": "B4_P5",
        "image_key": "B4_P5_C01_F01",
    }
    values[component_name] = invalid_value
    mask_path = (
        output_dir
        / "feature_cache"
        / "masks"
        / values["group_id"]
        / f"{values['image_key']}.npz"
    )
    manifest = pd.DataFrame(
        [
            {
                **values,
                "pc_path": str(tmp_path / "phase.png"),
                "ido_path": str(tmp_path / "ido.png"),
            }
        ]
    )
    segmentation_qc = pd.DataFrame(
        [
            {
                "image_key": values["image_key"],
                "mask_path": str(mask_path),
                "status": "passed",
            }
        ]
    )

    with pytest.raises(ValueError, match=component_name):
        extract_basic_cell_features(
            manifest,
            segmentation_qc,
            {"_output_dir": str(output_dir)},
        )


def test_mask_cache_path_rejects_resolved_parent_outside_masks_root(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "feature_cache"
    escaped_path = cache_dir / "escaped.npz"

    with pytest.raises(ValueError, match="mask_path"):
        phase_features._validate_exp3_mask_cache_path(
            escaped_path,
            cache_dir,
            "..",
            "escaped",
        )


def _create_directory_link(link: Path, target: Path) -> None:
    """建立測試用 directory symlink，Windows 權限不足時改用 junction。"""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        import os
        import subprocess

        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_mask_cache_path_rejects_symlinked_group_outside_masks_root(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "feature_cache"
    masks_root = cache_dir / "masks"
    outside = tmp_path / "outside"
    masks_root.mkdir(parents=True)
    outside.mkdir()
    linked_group = masks_root / "B4_P5"
    _create_directory_link(linked_group, outside)
    escaped_path = outside / "B4_P5_C01_F01.npz"

    with pytest.raises(ValueError, match="mask_path"):
        phase_features._validate_exp3_mask_cache_path(
            escaped_path,
            cache_dir,
            "B4_P5",
            "B4_P5_C01_F01",
        )


def test_mask_cache_path_rejects_masks_root_linked_outside_cache(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "feature_cache"
    outside = tmp_path / "outside-masks"
    cache_dir.mkdir()
    outside.mkdir()
    _create_directory_link(cache_dir / "masks", outside)
    group_dir = outside / "B4_P5"
    group_dir.mkdir()
    escaped_path = group_dir / "B4_P5_C01_F01.npz"

    with pytest.raises(ValueError, match="mask_path"):
        phase_features._validate_exp3_mask_cache_path(
            escaped_path,
            cache_dir,
            "B4_P5",
            "B4_P5_C01_F01",
        )
