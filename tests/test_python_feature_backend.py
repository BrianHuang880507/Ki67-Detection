import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ki67dtc.cell_anal import (
    _advanced_texture_feature_values_python,
    _measure_roi_with_python,
    _nucleolus_feature_values,
    flour_anal,
    merged_excel,
    param_anal,
)


class PythonFeatureBackendTest(unittest.TestCase):
    def test_circle_uses_standard_circularity_formula(self) -> None:
        signal = np.ones((64, 64), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (32, 32), 12, 1, -1)

        measurement = _measure_roi_with_python(signal, mask.astype(bool))
        expected = (
            4.0
            * np.pi
            * measurement["area"]
            / measurement["perimeter"] ** 2
        )

        self.assertAlmostEqual(measurement["circ"], expected)
        self.assertGreater(measurement["circ"], 0.8)

    def test_advanced_texture_features_are_finite_and_normalized(self) -> None:
        y_grid, x_grid = np.indices((96, 96))
        signal = (
            0.2
            + 0.004 * x_grid
            + 0.08 * np.sin(x_grid / 4.0)
            + 0.05 * np.cos(y_grid / 7.0)
        ).astype(np.float32)
        mask = (x_grid - 48) ** 2 + (y_grid - 48) ** 2 <= 30**2

        values = _advanced_texture_feature_values_python(
            signal, mask, erode_px=2
        )

        self.assertEqual(len(values), 56)
        self.assertTrue(np.isfinite(values).all())
        for radius_index in range(3):
            start = radius_index * 10
            self.assertAlmostEqual(sum(values[start : start + 10]), 1.0)
        self.assertGreater(values[30], 0.0)

    def test_nucleolus_detection_finds_bright_foci(self) -> None:
        y_grid, x_grid = np.indices((96, 96))
        mask = (x_grid - 48) ** 2 + (y_grid - 48) ** 2 <= 26**2
        signal = np.full((96, 96), 0.2, dtype=np.float32)
        signal[mask] = 0.35
        for center_y, center_x in [(40, 40), (56, 50), (41, 60)]:
            signal += (
                0.5
                * np.exp(
                    -(
                        (x_grid - center_x) ** 2
                        + (y_grid - center_y) ** 2
                    )
                    / (2.0 * 1.5**2)
                )
            ).astype(np.float32)

        count, mean_area, max_area = _nucleolus_feature_values(signal, mask)

        self.assertEqual(count, 3.0)
        self.assertGreater(mean_area, 0.0)
        self.assertGreaterEqual(max_area, mean_area)

    def test_param_analysis_writes_expanded_schema_and_merges_cell_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.tile(
                np.linspace(20, 220, 160, dtype=np.uint8),
                (128, 1),
            )
            cv2.circle(image, (42, 64), 26, 170, -1)
            cv2.circle(image, (42, 64), 11, 75, -1)
            cv2.circle(image, (112, 64), 23, 155, -1)
            cv2.circle(image, (112, 64), 9, 65, -1)
            for center in [(38, 61), (46, 67), (109, 61)]:
                cv2.circle(image, center, 2, 245, -1)
            image_path = root / "sample.png"
            cv2.imwrite(str(image_path), image)

            outline_path = root / "sample_merged_cp_outlines.txt"
            outline_path.write_text(
                "31,53,53,53,53,75,31,75\n"
                "16,38,68,38,68,90,16,90\n"
                "103,55,121,55,121,73,103,73\n"
                "89,41,135,41,135,87,89,87\n",
                encoding="utf-8",
            )
            output_path = root / "params.csv"
            merged_path = root / "params_merged.csv"

            param_anal(
                image_path,
                outline_path,
                output_path,
                feature_backend="python",
            )

            result = pd.read_csv(output_path)
            self.assertEqual(
                result["Cell_ID"].tolist(),
                [
                    "sample_1_nuc",
                    "sample_1_cyto",
                    "sample_2_nuc",
                    "sample_2_cyto",
                ],
            )
            for column in [
                "Mean",
                "GLCM Contrast",
                "LBP Entropy",
                "LBP Uniform R3 Hist Bin 09",
                "Tamura Coarseness",
                "Zernike Moment 24",
                "Whole Cell Mean",
                "Whole Cell GLCM Entropy",
                "Nucleolus Count",
                "Nuc Cell IntDen Ratio",
                "Halo Outer CV",
                "Halo Angular Variance",
                "Halo Radial Gradient",
                "Halo Width",
                "Neighbour Area Ratio",
                "Mitotic Index",
                "Mean Protrusion Length Norm",
                "Debris Count",
                "Mitotic Score",
            ]:
                self.assertIn(column, result.columns)
            cyto_rows = result[result["Cell_ID"].str.endswith("_cyto")]
            for column in [
                "Whole Cell Mean",
                "Whole Cell GLCM Entropy",
                "Nucleolus Count",
                "Halo Angular Variance",
                "Halo Radial Gradient",
                "Neighbour Area Ratio",
                "Mitotic Index",
                "Mitotic Score",
            ]:
                self.assertTrue(np.isfinite(cyto_rows[column]).all(), column)

            for radius in (1, 2, 3):
                columns = [
                    f"LBP Uniform R{radius} Hist Bin {idx:02d}"
                    for idx in range(10)
                ]
                sums = result[columns].sum(axis=1)
                np.testing.assert_allclose(sums, 1.0, atol=1e-6)

            merged_excel(output_path, merged_path)
            merged = pd.read_csv(merged_path)
            self.assertEqual(len(merged), 2)
            self.assertIn("Whole Cell Mean", merged.columns)
            self.assertIn("Tamura Coarseness_nuc", merged.columns)
            self.assertIn("Tamura Coarseness_cyto", merged.columns)
            self.assertNotIn("Whole Cell Mean_nuc", merged.columns)
            self.assertNotIn("Halo Angular Variance_cyto", merged.columns)

    def test_python_fluorescence_extraction_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y_grid, x_grid = np.indices((96, 96))
            image = (
                20
                + 180
                * np.exp(
                    -((x_grid - 48) ** 2 + (y_grid - 48) ** 2)
                    / (2.0 * 18**2)
                )
            ).astype(np.uint8)
            image_path = root / "fluor.png"
            cv2.imwrite(str(image_path), image)
            outline_path = root / "sample_merged_cp_outlines.txt"
            outline_path.write_text(
                "38,38,58,38,58,58,38,58\n"
                "22,22,74,22,74,74,22,74\n",
                encoding="utf-8",
            )
            output_path = root / "fluor.csv"

            flour_anal(
                image_path,
                outline_path,
                output_path,
                max_expand_steps=3,
                feature_backend="python",
            )

            result = pd.read_csv(output_path)
            self.assertGreater(len(result), 0)
            self.assertEqual(
                result.columns.tolist(), ["Label", "IntDen", "RawIntDen"]
            )
            self.assertTrue(np.isfinite(result["IntDen"]).all())


if __name__ == "__main__":
    unittest.main()
