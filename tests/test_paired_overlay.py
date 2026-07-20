import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from ki67dtc.app_pipeline import PipelineResult, load_paired_merged_outlines
from ki67dtc.gui.main_window import save_pipeline_fill_overlays
from ki67dtc.paired_overlay import (
    build_paired_overlay_data,
    find_paired_labels,
    render_paired_overlay_bgr,
)


class PairedOverlayTest(unittest.TestCase):
    def test_find_paired_labels_matches_segui_centroid_rule(self) -> None:
        cytoplasm_mask = np.zeros((12, 14), dtype=np.int32)
        cytoplasm_mask[1:10, 1:9] = 4
        cytoplasm_mask[1:3, 11:13] = 9

        nucleus_mask = np.zeros_like(cytoplasm_mask)
        nucleus_mask[3:5, 3:5] = 2
        nucleus_mask[6:8, 5:7] = 7
        nucleus_mask[8:10, 11:13] = 8

        pairs = find_paired_labels(cytoplasm_mask, nucleus_mask)

        self.assertEqual(pairs, [(4, 2), (4, 7)])

    def test_render_hides_unpaired_regions_without_mutating_masks(self) -> None:
        base = np.full((12, 14, 3), 120, dtype=np.uint8)
        cytoplasm_mask = np.zeros((12, 14), dtype=np.int32)
        cytoplasm_mask[1:10, 1:9] = 1
        cytoplasm_mask[1:4, 10:13] = 2
        nucleus_mask = np.zeros_like(cytoplasm_mask)
        nucleus_mask[4:6, 4:6] = 1
        nucleus_mask[8:10, 10:12] = 2
        cytoplasm_before = cytoplasm_mask.copy()
        nucleus_before = nucleus_mask.copy()

        paired_data = build_paired_overlay_data(cytoplasm_mask, nucleus_mask)
        rendered = render_paired_overlay_bgr(base, paired_data, alpha=0.5)

        self.assertEqual(paired_data.raw_cytoplasm_count, 2)
        self.assertEqual(paired_data.raw_nucleus_count, 2)
        self.assertEqual(paired_data.paired_cytoplasm_count, 1)
        self.assertEqual(paired_data.paired_nucleus_count, 1)
        self.assertFalse(np.array_equal(rendered[2, 2], base[2, 2]))
        np.testing.assert_array_equal(rendered[2, 11], base[2, 11])
        np.testing.assert_array_equal(rendered[8, 10], base[8, 10])
        np.testing.assert_array_equal(cytoplasm_mask, cytoplasm_before)
        np.testing.assert_array_equal(nucleus_mask, nucleus_before)

    def test_saved_overlay_uses_same_paired_only_data(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "data" / "input" / "demo" / "PC"
            results_dir = root / "data" / "output" / "results" / "demo"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample.png"
            base = np.full((12, 14, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), base))

            cytoplasm_mask = np.zeros((12, 14), dtype=np.int32)
            cytoplasm_mask[1:10, 1:9] = 1
            cytoplasm_mask[1:4, 10:13] = 2
            nucleus_mask = np.zeros_like(cytoplasm_mask)
            nucleus_mask[4:6, 4:6] = 1
            paired_data = build_paired_overlay_data(cytoplasm_mask, nucleus_mask)
            result = PipelineResult(
                data_folder=root / "data" / "input" / "demo",
                image_files=[image_path],
                results_dir=results_dir,
                paired_overlays={image_path.stem: paired_data},
            )

            saved_paths = save_pipeline_fill_overlays(result, alpha=0.5)

            self.assertEqual(saved_paths, [results_dir / "sample_overlay.png"])
            saved = cv2.imread(str(saved_paths[0]), cv2.IMREAD_COLOR)
            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertFalse(np.array_equal(saved[2, 2], base[2, 2]))
            np.testing.assert_array_equal(saved[2, 11], base[2, 11])
            self.assertEqual(
                sorted(path.name for path in results_dir.glob("*overlay*")),
                ["sample_overlay.png"],
            )

    def test_merged_outline_fallback_keeps_only_complete_pairs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            merged_path = Path(temporary_directory) / "sample_merged_cp_outlines.txt"
            merged_path.write_text(
                "1,1,3,1,3,3,1,3\n"
                "0,0,5,0,5,5,0,5\n"
                "7,7,8,7,8,8,7,8\n"
                "-1,-1\n"
                "-1,-1\n"
                "9,9,11,9,11,11,9,11\n",
                encoding="utf-8",
            )

            polygons = load_paired_merged_outlines(merged_path)

            self.assertEqual(len(polygons.nuc_polygons), 1)
            self.assertEqual(len(polygons.cyto_polygons), 1)
            np.testing.assert_array_equal(
                polygons.nuc_polygons[0],
                np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=np.int32),
            )


if __name__ == "__main__":
    unittest.main()
