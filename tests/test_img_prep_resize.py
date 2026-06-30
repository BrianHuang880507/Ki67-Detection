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
                ),
                call(
                    EXPECTED_PC_NUC_MODEL_PATH,
                    [image_path],
                    seg_dir,
                    "nuc",
                    channels=(0, 0),
                    model_input_size=img_prep.NUC_MODEL_INPUT_SIZE,
                ),
            ],
        )

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
                ),
                call(
                    EXPECTED_DAPI_NUC_MODEL_PATH,
                    [dapi_image],
                    seg_dir,
                    "nuc",
                    output_stems=[pc_image.stem],
                    channels=(3, 3),
                    model_input_size=img_prep.NUC_MODEL_INPUT_SIZE,
                ),
            ],
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
                ),
                call(
                    EXPECTED_PC_NUC_MODEL_PATH,
                    [image_path],
                    seg_dir,
                    "nuc",
                    channels=(0, 0),
                    model_input_size=img_prep.NUC_MODEL_INPUT_SIZE,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
