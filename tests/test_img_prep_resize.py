import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np

from ki67dtc import img_prep


EXPECTED_PC_NUC_MODEL_PATH = "model/model_BDL3_label_dapi"
EXPECTED_DAPI_NUC_MODEL_PATH = "cyto3"


class SegmentResizeTest(unittest.TestCase):
    def test_segment_resizes_model_input_and_restores_mask_to_original_shape(
        self,
    ) -> None:
        original = np.zeros((6, 8, 3), dtype=np.uint8)
        model_input_shapes: list[tuple[int, ...]] = []
        saved_shapes: dict[str, tuple[int, ...]] = {}

        class FakeModel:
            def __init__(self, gpu: bool, pretrained_model: str) -> None:
                self.gpu = gpu
                self.pretrained_model = pretrained_model

            def eval(self, img, **kwargs):
                model_input_shapes.append(img.shape)
                masks = np.ones(img.shape[:2], dtype=np.int32)
                flows = [np.zeros((*img.shape[:2], 3), dtype=np.float32)]
                return masks, flows, None

        def fake_save(img, masks, flows, filename, channels, diams):
            saved_shapes["img"] = img.shape
            saved_shapes["masks"] = masks.shape
            saved_shapes["flow"] = flows[0].shape
            saved_shapes["channels"] = tuple(channels)
            saved_shapes["diams"] = diams

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.jpg"
            output_dir = root / "segment"
            output_dir.mkdir()

            with (
                patch.object(img_prep.io, "imread", return_value=original),
                patch.object(img_prep.models, "CellposeModel", FakeModel),
                patch.object(img_prep.io, "masks_flows_to_seg", fake_save),
                patch.object(img_prep.shutil, "move"),
            ):
                img_prep.segment(
                    "model/path",
                    [image_path],
                    output_dir,
                    "cyto",
                    channels=(0, 0),
                    model_input_size=(4, 3),
                )

        self.assertEqual(model_input_shapes, [(3, 4, 3)])
        self.assertEqual(saved_shapes["img"], original.shape)
        self.assertEqual(saved_shapes["masks"], original.shape[:2])
        self.assertEqual(saved_shapes["flow"], original.shape)
        self.assertEqual(saved_shapes["channels"], (0, 0))
        self.assertIsNone(saved_shapes["diams"])

    def _run_segment_capturing_gpu_flag(self, **segment_kwargs) -> list[bool]:
        """執行 segment() 並回傳每次建立 CellposeModel 時收到的 gpu 旗標。"""
        original = np.zeros((6, 8, 3), dtype=np.uint8)
        gpu_flags: list[bool] = []

        class FakeModel:
            def __init__(self, gpu: bool, pretrained_model: str) -> None:
                gpu_flags.append(gpu)

            def eval(self, img, **kwargs):
                masks = np.ones(img.shape[:2], dtype=np.int32)
                flows = [np.zeros((*img.shape[:2], 3), dtype=np.float32)]
                return masks, flows, None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "segment"
            output_dir.mkdir()

            with (
                patch.object(img_prep.io, "imread", return_value=original),
                patch.object(img_prep.models, "CellposeModel", FakeModel),
                patch.object(img_prep.io, "masks_flows_to_seg"),
                patch.object(img_prep.shutil, "move"),
            ):
                img_prep.segment(
                    "model/path",
                    [root / "sample.jpg"],
                    output_dir,
                    "cyto",
                    channels=(0, 0),
                    **segment_kwargs,
                )

        return gpu_flags

    def test_segment_defaults_to_gpu(self) -> None:
        self.assertEqual(self._run_segment_capturing_gpu_flag(), [True])

    def test_segment_forces_cpu_when_use_gpu_is_false(self) -> None:
        self.assertEqual(self._run_segment_capturing_gpu_flag(use_gpu=False), [False])

    def test_filter_small_labels_removes_fragments_and_relabels_contiguously(
        self,
    ) -> None:
        masks = np.zeros((20, 30), dtype=np.int32)
        masks[0:10, 0:10] = 5
        masks[0:10, 12:22] = 9
        masks[15:17, 0:5] = 42

        filtered = img_prep._filter_small_labels(
            masks,
            min_area_ratio=0.15,
            min_area_floor=30,
        )

        np.testing.assert_array_equal(np.unique(filtered), np.array([0, 1, 2]))
        self.assertEqual(np.count_nonzero(filtered == 1), 100)
        self.assertEqual(np.count_nonzero(filtered == 2), 100)
        self.assertTrue(np.all(filtered[15:17, 0:5] == 0))

    def test_paired_filter_keeps_small_pairs_and_removes_unpaired_noise(
        self,
    ) -> None:
        cyto_masks = np.zeros((50, 80), dtype=np.int32)
        nuc_masks = np.zeros_like(cyto_masks)

        for label, x_start in enumerate((5, 25, 45), start=1):
            cyto_masks[5:15, x_start : x_start + 10] = label
            nuc_masks[8:12, x_start + 3 : x_start + 7] = label

        cyto_masks[25:28, 5:8] = 4
        nuc_masks[26:28, 6:8] = 4
        cyto_masks[25:28, 25:28] = 5
        nuc_masks[25:27, 45:47] = 5

        result = img_prep._filter_small_unpaired_labels(
            cyto_masks,
            nuc_masks,
            min_area_ratio=0.5,
            min_area_floor=0,
            min_reference_count=3,
        )

        self.assertEqual(result.cytoplasm_threshold, 50.0)
        self.assertEqual(result.nucleus_threshold, 8.0)
        self.assertEqual(result.paired_cytoplasm_count, 4)
        self.assertEqual(result.paired_nucleus_count, 4)
        self.assertNotEqual(result.cytoplasm_mask[26, 6], 0)
        self.assertNotEqual(result.nucleus_mask[26, 6], 0)
        self.assertEqual(result.cytoplasm_mask[26, 26], 0)
        self.assertEqual(result.nucleus_mask[25, 45], 0)

    def test_paired_segment_file_filter_updates_masks_and_preserves_payload(
        self,
    ) -> None:
        cyto_masks = np.zeros((20, 30), dtype=np.int32)
        nuc_masks = np.zeros_like(cyto_masks)
        cyto_masks[2:12, 2:12] = 1
        nuc_masks[5:9, 5:9] = 1
        cyto_masks[15:18, 2:5] = 2
        nuc_masks[15:17, 15:17] = 2

        with tempfile.TemporaryDirectory() as tmp:
            seg_dir = Path(tmp)
            cyto_path = seg_dir / "sample_cyto_seg.npy"
            nuc_path = seg_dir / "sample_nuc_seg.npy"
            np.save(cyto_path, {"masks": cyto_masks, "flows": ["cyto-flow"]})
            np.save(nuc_path, {"masks": nuc_masks, "flows": ["nuc-flow"]})

            results = img_prep._filter_paired_segment_files(
                seg_dir,
                ["sample"],
                min_area_ratio=0.15,
                min_area_floor=30,
            )

            saved_cyto = np.load(cyto_path, allow_pickle=True).item()
            saved_nuc = np.load(nuc_path, allow_pickle=True).item()

        self.assertEqual(results["sample"].removed_cytoplasm_count, 1)
        self.assertEqual(results["sample"].removed_nucleus_count, 1)
        np.testing.assert_array_equal(np.unique(saved_cyto["masks"]), [0, 1])
        np.testing.assert_array_equal(np.unique(saved_nuc["masks"]), [0, 1])
        self.assertEqual(saved_cyto["flows"], ["cyto-flow"])
        self.assertEqual(saved_nuc["flows"], ["nuc-flow"])

    def test_segment_all_passes_default_resize_sizes_to_cyto_and_pc_nuc(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pc_dir = root / "PC"
            pc_dir.mkdir()
            image_path = pc_dir / "sample.jpg"
            image_path.write_bytes(b"placeholder")
            seg_dir = root / "data" / "output" / "segment"

            def fake_output_dir(input_dir, subfolder):
                self.assertEqual(Path(input_dir), root)
                self.assertEqual(subfolder, "segment")
                return seg_dir

            with (
                patch.object(img_prep, "output_dir", side_effect=fake_output_dir),
                patch.object(img_prep, "list_files", return_value=[image_path]),
                patch.object(img_prep, "segment") as segment_mock,
                patch.object(img_prep, "_filter_paired_segment_files") as filter_mock,
            ):
                img_prep.segment_all(root, nuc_source="pc")

        self.assertEqual(
            segment_mock.call_args_list,
            [
                call(
                    img_prep.CYTO_MODEL_PATH,
                    [image_path],
                    seg_dir,
                    "cyto",
                    channels=(0, 0),
                    model_input_size=img_prep.CYTO_MODEL_INPUT_SIZE,
                    cellprob_threshold=0.0,
                    min_area_ratio=0.15,
                    min_area_floor=30,
                    apply_small_label_filter=False,
                    use_gpu=True,
                ),
                call(
                    EXPECTED_PC_NUC_MODEL_PATH,
                    [image_path],
                    seg_dir,
                    "nuc",
                    channels=(0, 0),
                    model_input_size=img_prep.NUC_MODEL_INPUT_SIZE,
                    cellprob_threshold=0.0,
                    min_area_ratio=0.15,
                    min_area_floor=30,
                    apply_small_label_filter=False,
                    use_gpu=True,
                ),
            ],
        )
        filter_mock.assert_called_once_with(
            seg_dir,
            [image_path.stem],
            min_area_ratio=0.15,
            min_area_floor=30,
        )

    def test_segment_all_forwards_use_gpu_to_every_segment_call(self) -> None:
        """use_gpu 必須傳到每一個 segment() 呼叫，漏掉的分支會靜默回到 GPU。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pc_dir = root / "PC"
            pc_dir.mkdir()
            image_path = pc_dir / "sample.jpg"
            image_path.write_bytes(b"placeholder")

            with (
                patch.object(
                    img_prep, "output_dir", return_value=root / "segment"
                ),
                patch.object(img_prep, "list_files", return_value=[image_path]),
                patch.object(img_prep, "segment") as segment_mock,
                patch.object(img_prep, "_filter_paired_segment_files"),
            ):
                img_prep.segment_all(root, nuc_source="pc", use_gpu=False)

        self.assertGreater(len(segment_mock.call_args_list), 0)
        for segment_call in segment_mock.call_args_list:
            self.assertIs(segment_call.kwargs["use_gpu"], False)

    def test_segment_all_uses_dapi_nuc_model_when_dapi_source_is_available(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pc_dir = root / "PC"
            dapi_dir = root / "DAPI"
            pc_dir.mkdir()
            dapi_dir.mkdir()
            pc_image = pc_dir / "sample.jpg"
            dapi_image = dapi_dir / "sample.jpg"
            pc_image.write_bytes(b"pc")
            dapi_image.write_bytes(b"dapi")
            seg_dir = root / "data" / "output" / "segment"

            def fake_output_dir(input_dir, subfolder):
                self.assertEqual(Path(input_dir), root)
                self.assertEqual(subfolder, "segment")
                return seg_dir

            def fake_list_files(folder, extensions):
                folder = Path(folder)
                if folder == pc_dir:
                    return [pc_image]
                if folder == dapi_dir:
                    return [dapi_image]
                return []

            with (
                patch.object(img_prep, "output_dir", side_effect=fake_output_dir),
                patch.object(img_prep, "list_files", side_effect=fake_list_files),
                patch.object(img_prep, "segment") as segment_mock,
                patch.object(img_prep, "remap_nuc_segments_to_cyto"),
                patch.object(img_prep, "_filter_paired_segment_files") as filter_mock,
            ):
                img_prep.segment_all(root, nuc_source="dapi")

        self.assertEqual(
            segment_mock.call_args_list,
            [
                call(
                    img_prep.CYTO_MODEL_PATH,
                    [pc_image],
                    seg_dir,
                    "cyto",
                    channels=(0, 0),
                    model_input_size=img_prep.CYTO_MODEL_INPUT_SIZE,
                    cellprob_threshold=0.0,
                    min_area_ratio=0.15,
                    min_area_floor=30,
                    apply_small_label_filter=False,
                    use_gpu=True,
                ),
                call(
                    EXPECTED_DAPI_NUC_MODEL_PATH,
                    [dapi_image],
                    seg_dir,
                    "nuc",
                    output_stems=[pc_image.stem],
                    channels=(3, 3),
                    model_input_size=img_prep.NUC_MODEL_INPUT_SIZE,
                    cellprob_threshold=0.0,
                    min_area_ratio=0.15,
                    min_area_floor=30,
                    apply_small_label_filter=False,
                    use_gpu=True,
                ),
            ],
        )
        filter_mock.assert_called_once_with(
            seg_dir,
            [pc_image.stem],
            min_area_ratio=0.15,
            min_area_floor=30,
        )

    def test_segment_all_falls_back_to_pc_nuc_model_when_dapi_folder_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pc_dir = root / "PC"
            pc_dir.mkdir()
            image_path = pc_dir / "sample.jpg"
            image_path.write_bytes(b"placeholder")
            seg_dir = root / "data" / "output" / "segment"

            def fake_output_dir(input_dir, subfolder):
                self.assertEqual(Path(input_dir), root)
                self.assertEqual(subfolder, "segment")
                return seg_dir

            with (
                patch.object(img_prep, "output_dir", side_effect=fake_output_dir),
                patch.object(img_prep, "list_files", return_value=[image_path]),
                patch.object(img_prep, "segment") as segment_mock,
                patch.object(img_prep, "_filter_paired_segment_files") as filter_mock,
            ):
                img_prep.segment_all(root, nuc_source="dapi")

        self.assertEqual(
            segment_mock.call_args_list,
            [
                call(
                    img_prep.CYTO_MODEL_PATH,
                    [image_path],
                    seg_dir,
                    "cyto",
                    channels=(0, 0),
                    model_input_size=img_prep.CYTO_MODEL_INPUT_SIZE,
                    cellprob_threshold=0.0,
                    min_area_ratio=0.15,
                    min_area_floor=30,
                    apply_small_label_filter=False,
                    use_gpu=True,
                ),
                call(
                    EXPECTED_PC_NUC_MODEL_PATH,
                    [image_path],
                    seg_dir,
                    "nuc",
                    channels=(0, 0),
                    model_input_size=img_prep.NUC_MODEL_INPUT_SIZE,
                    cellprob_threshold=0.0,
                    min_area_ratio=0.15,
                    min_area_floor=30,
                    apply_small_label_filter=False,
                    use_gpu=True,
                ),
            ],
        )
        filter_mock.assert_called_once_with(
            seg_dir,
            [image_path.stem],
            min_area_ratio=0.15,
            min_area_floor=30,
        )


if __name__ == "__main__":
    unittest.main()
