import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import cv2

from ki67dtc.cell_anal_plot import plot_global_area_analysis


class CellAreaPlotTest(unittest.TestCase):
    def test_global_area_analysis_saves_scatter_and_histogram(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "sample_cleaned.csv"
            outdir = root / "results"
            pd.DataFrame(
                {
                    "Cell_ID": ["sample_1", "sample_2", "sample_3"],
                    "Area_nuc": [100.0, 150.0, 80.0],
                    "Area_cyto": [1000.0, 1800.0, 900.0],
                }
            ).to_csv(csv_path, index=False)

            scatter_path, histogram_path = plot_global_area_analysis(
                csv_path,
                outdir,
                thres=6.0,
                width_um_per_px=2.0,
                height_um_per_px=3.0,
            )

            self.assertEqual(scatter_path, outdir / "all_cell_nucleus_area.png")
            self.assertEqual(
                histogram_path,
                outdir / "all_log_cell_area_distribution.png",
            )
            self.assertGreater(scatter_path.stat().st_size, 0)
            self.assertGreater(histogram_path.stat().st_size, 0)

    def test_global_area_analysis_writes_opaque_white_background_images(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "sample_cleaned.csv"
            outdir = root / "results"
            pd.DataFrame(
                {
                    "Cell_ID": ["sample_1", "sample_2", "sample_3"],
                    "Area_nuc": [100.0, 150.0, 80.0],
                    "Area_cyto": [1000.0, 1800.0, 900.0],
                }
            ).to_csv(csv_path, index=False)

            scatter_path, histogram_path = plot_global_area_analysis(csv_path, outdir)

            for image_path in [scatter_path, histogram_path]:
                with self.subTest(image_path=image_path.name):
                    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                    self.assertIsNotNone(image)
                    assert image is not None
                    if image.shape[2] == 4:
                        self.assertEqual(int(image[0, 0, 3]), 255)
                    self.assertGreaterEqual(int(image[0, 0, 0]), 245)
                    self.assertGreaterEqual(int(image[0, 0, 1]), 245)
                    self.assertGreaterEqual(int(image[0, 0, 2]), 245)


if __name__ == "__main__":
    unittest.main()
