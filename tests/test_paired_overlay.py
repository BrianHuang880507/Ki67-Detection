import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from ki67dtc.app_pipeline import (
    PipelineResult,
    load_merged_outlines,
    load_paired_merged_outlines,
)
from ki67dtc.gui.main_window import save_pipeline_fill_overlays
from ki67dtc.paired_overlay import (
    build_paired_overlay_data,
    find_paired_labels,
    render_all_overlay_bgr,
    render_paired_overlay_bgr,
)


class PairedOverlayTest(unittest.TestCase):
    def test_paired_renderer_preserves_pair_interleaved_overlap_order(self) -> None:
        base = np.full((13, 13, 3), 120, dtype=np.uint8)
        cytoplasm_mask = np.zeros((13, 13), dtype=np.int32)
        cytoplasm_mask[1:12, 1:12] = 1
        nucleus_mask = np.zeros_like(cytoplasm_mask)
        nucleus_mask[3:6, 3:6] = 1
        nucleus_mask[7:10, 7:10] = 2

        data = build_paired_overlay_data(cytoplasm_mask, nucleus_mask)
        rendered = render_paired_overlay_bgr(base, data, alpha=1.0)

        self.assertEqual(len(data.pairs), 2)
        np.testing.assert_array_equal(rendered[4, 4], np.array([0, 255, 0]))
        np.testing.assert_array_equal(rendered[8, 8], np.array([240, 0, 0]))

    def test_saved_overlay_preserves_pair_interleaved_overlap_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "data" / "input" / "demo" / "PC"
            results_dir = root / "data" / "output" / "results" / "demo"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample.png"
            base = np.full((13, 13, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), base))

            cytoplasm_mask = np.zeros((13, 13), dtype=np.int32)
            cytoplasm_mask[1:12, 1:12] = 1
            nucleus_mask = np.zeros_like(cytoplasm_mask)
            nucleus_mask[3:6, 3:6] = 1
            nucleus_mask[7:10, 7:10] = 2
            data = build_paired_overlay_data(cytoplasm_mask, nucleus_mask)
            result = PipelineResult(
                data_folder=root / "data" / "input" / "demo",
                image_files=[image_path],
                results_dir=results_dir,
                paired_overlays={image_path.stem: data},
            )

            saved_paths = save_pipeline_fill_overlays(result, alpha=1.0)

            saved = cv2.imread(str(saved_paths[0]), cv2.IMREAD_COLOR)
            self.assertIsNotNone(saved)
            assert saved is not None
            np.testing.assert_array_equal(saved[4, 4], np.array([0, 255, 0]))
            np.testing.assert_array_equal(saved[8, 8], np.array([240, 0, 0]))

    def test_all_regions_are_cached_and_rendered_without_mutating_inputs(self) -> None:
        base = np.full((12, 14, 3), 120, dtype=np.uint8)
        cytoplasm_mask = np.zeros((12, 14), dtype=np.int32)
        cytoplasm_mask[1:10, 1:9] = 1
        cytoplasm_mask[1:4, 10:13] = 2
        nucleus_mask = np.zeros_like(cytoplasm_mask)
        nucleus_mask[4:6, 4:6] = 1
        nucleus_mask[8:10, 10:12] = 2
        base_before = base.copy()
        cytoplasm_before = cytoplasm_mask.copy()
        nucleus_before = nucleus_mask.copy()

        data = build_paired_overlay_data(cytoplasm_mask, nucleus_mask)
        paired = render_paired_overlay_bgr(base, data, alpha=0.5)
        all_regions = render_all_overlay_bgr(base, data, alpha=0.5)

        self.assertEqual(
            [region.label for region in data.all_cytoplasm_regions],
            [1, 2],
        )
        self.assertEqual(
            [region.label for region in data.all_nucleus_regions],
            [1, 2],
        )
        np.testing.assert_array_equal(paired[2, 11], base[2, 11])
        np.testing.assert_array_equal(paired[8, 10], base[8, 10])
        self.assertFalse(np.array_equal(all_regions[2, 11], base[2, 11]))
        self.assertFalse(np.array_equal(all_regions[8, 10], base[8, 10]))
        hidden = render_all_overlay_bgr(
            base,
            data,
            show_nucleus=False,
            show_cytoplasm=False,
            alpha=0.5,
        )
        lower_alpha = render_all_overlay_bgr(base, data, alpha=0.25)
        np.testing.assert_array_equal(hidden, base)
        full_distance = np.linalg.norm(
            all_regions[2, 11].astype(int) - base[2, 11].astype(int)
        )
        lower_distance = np.linalg.norm(
            lower_alpha[2, 11].astype(int) - base[2, 11].astype(int)
        )
        self.assertLess(lower_distance, full_distance)
        np.testing.assert_array_equal(base, base_before)
        np.testing.assert_array_equal(cytoplasm_mask, cytoplasm_before)
        np.testing.assert_array_equal(nucleus_mask, nucleus_before)

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

    def test_merged_outline_loaders_separate_paired_and_all_records(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            merged_path = Path(temporary_directory) / "sample_merged_cp_outlines.txt"
            merged_path.write_text(
                "1,1,5,1,5,5,1,5\n"
                "0,0,6,0,6,6,0,6\n"
                "8,8,12,8,12,12,8,12\n"
                "-1,-1\n"
                "-1,-1\n"
                "8,1,13,1,13,6,8,6\n"
                "not,a,polygon\n"
                "14,1,18,1,18,5,14,5\n"
                "20,20,24,20,24,24,20,24\n",
                encoding="utf-8",
            )

            paired = load_paired_merged_outlines(merged_path)
            all_polygons = load_merged_outlines(merged_path)

            self.assertEqual(len(paired.nuc_polygons), 1)
            self.assertEqual(len(paired.cyto_polygons), 1)
            self.assertEqual(len(all_polygons.nuc_polygons), 2)
            self.assertEqual(len(all_polygons.cyto_polygons), 3)
            np.testing.assert_array_equal(
                all_polygons.nuc_polygons[1],
                np.array([[8, 8], [12, 8], [12, 12], [8, 12]], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                all_polygons.cyto_polygons[-1],
                np.array([[14, 1], [18, 1], [18, 5], [14, 5]], dtype=np.int32),
            )

    def test_merged_outline_loaders_return_empty_for_unreadable_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing_merged_cp_outlines.txt"

            paired = load_paired_merged_outlines(missing_path)
            all_polygons = load_merged_outlines(missing_path)

            self.assertEqual(paired.nuc_polygons, [])
            self.assertEqual(paired.cyto_polygons, [])
            self.assertEqual(all_polygons.nuc_polygons, [])
            self.assertEqual(all_polygons.cyto_polygons, [])

    def test_merged_outline_loaders_return_empty_for_existing_empty_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            merged_path = Path(temporary_directory) / "empty_merged_cp_outlines.txt"
            merged_path.touch()

            paired = load_paired_merged_outlines(merged_path)
            all_polygons = load_merged_outlines(merged_path)

            self.assertEqual(paired.nuc_polygons, [])
            self.assertEqual(paired.cyto_polygons, [])
            self.assertEqual(paired.pair_record_indices, [])
            self.assertEqual(all_polygons.nuc_polygons, [])
            self.assertEqual(all_polygons.cyto_polygons, [])

    def test_merged_outline_loaders_reject_all_invalid_and_odd_polygons(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            merged_path = Path(temporary_directory) / "invalid_merged_cp_outlines.txt"
            merged_path.write_text(
                "not,a,polygon\n"
                "1,2,3,4,5\n"
                "1,2,3,4\n"
                "-1,-1\n",
                encoding="utf-8",
            )

            paired = load_paired_merged_outlines(merged_path)
            all_polygons = load_merged_outlines(merged_path)

            self.assertEqual(paired.nuc_polygons, [])
            self.assertEqual(paired.cyto_polygons, [])
            self.assertEqual(paired.pair_record_indices, [])
            self.assertEqual(all_polygons.nuc_polygons, [])
            self.assertEqual(all_polygons.cyto_polygons, [])

    def test_merged_outline_loaders_reject_out_of_range_coordinates_per_side(
        self,
    ) -> None:
        int32_max = np.iinfo(np.int32).max
        int32_min = np.iinfo(np.int32).min
        with TemporaryDirectory() as temporary_directory:
            merged_path = Path(temporary_directory) / "overflow_merged_cp_outlines.txt"
            merged_path.write_text(
                f"{int32_max + 1},1,5,1,5,5,1,5\n"
                "0,0,6,0,6,6,0,6\n"
                "20,20,24,20,24,24,20,24\n"
                f"{int32_min - 1},16,28,16,28,28,16,28\n",
                encoding="utf-8",
            )

            paired = load_paired_merged_outlines(merged_path)
            all_polygons = load_merged_outlines(merged_path)

            self.assertEqual(paired.nuc_polygons, [])
            self.assertEqual(paired.cyto_polygons, [])
            self.assertEqual(paired.pair_record_indices, [])
            self.assertEqual(len(all_polygons.nuc_polygons), 1)
            self.assertEqual(len(all_polygons.cyto_polygons), 1)
            np.testing.assert_array_equal(
                all_polygons.nuc_polygons[0],
                np.array([[20, 20], [24, 20], [24, 24], [20, 24]], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                all_polygons.cyto_polygons[0],
                np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=np.int32),
            )

    def test_saved_outline_fallback_remains_paired_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "data" / "input" / "demo" / "PC"
            outline_dir = root / "data" / "output" / "outline" / "demo"
            results_dir = root / "data" / "output" / "results" / "demo"
            image_dir.mkdir(parents=True)
            outline_dir.mkdir(parents=True)
            image_path = image_dir / "sample.png"
            base = np.full((18, 18, 3), 100, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), base))
            (outline_dir / "sample_merged_cp_outlines.txt").write_text(
                "5,5,9,5,9,9,5,9\n"
                "2,2,11,2,11,11,2,11\n"
                "12,10,16,10,16,15,12,15\n"
                "-1,-1\n"
                "-1,-1\n"
                "12,1,16,1,16,6,12,6\n",
                encoding="utf-8",
            )
            result = PipelineResult(
                data_folder=root / "data" / "input" / "demo",
                image_files=[image_path],
                results_dir=results_dir,
            )

            saved_path = save_pipeline_fill_overlays(result, alpha=0.5)[0]
            saved = cv2.imread(str(saved_path), cv2.IMREAD_COLOR)

            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertFalse(np.array_equal(saved[3, 3], base[3, 3]))
            np.testing.assert_array_equal(saved[12, 14], base[12, 14])
            np.testing.assert_array_equal(saved[3, 14], base[3, 14])


if __name__ == "__main__":
    unittest.main()
