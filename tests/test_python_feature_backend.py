import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

from ki67dtc.cell_anal import (
    _advanced_texture_feature_values_python,
    _geometry_from_measurements,
    _measure_roi_with_python,
    _nucleolus_feature_values,
    _polygon_to_mask,
    _texture_feature_parameter_values_python,
    flour_anal,
    merged_excel,
    param_anal,
)


class PythonFeatureBackendTest(unittest.TestCase):
    def test_python_measurement_uses_opencv_geometry(self) -> None:
        signal = np.ones((80, 80), dtype=np.float32)
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.ellipse(mask, (40, 40), (18, 8), 20, 0, 360, 1, -1)
        mask_bool = mask.astype(bool)

        measurement = _measure_roi_with_python(signal, mask_bool)
        geometry = _geometry_from_measurements(measurement)
        y_coords, x_coords = np.nonzero(mask_bool)
        area = float(np.count_nonzero(mask_bool))
        x_centroid = float(np.mean(x_coords))
        y_centroid = float(np.mean(y_coords))
        contours, _ = cv2.findContours(
            mask * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        convex_perimeter = float(cv2.arcLength(hull, True))
        hull_points = hull[:, 0, :].astype(np.float64)
        differences = hull_points[:, None, :] - hull_points[None, :, :]
        feret = float(np.sqrt(np.sum(differences**2, axis=2)).max())
        _, ellipse_axes, _ = cv2.fitEllipse(contour)
        major = float(max(ellipse_axes))
        minor = float(min(ellipse_axes))
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))
        expected_compactness = float(
            2.0
            * np.pi
            * np.mean(
                (y_coords.astype(float) - y_centroid) ** 2
                + (x_coords.astype(float) - x_centroid) ** 2
            )
            / area
        )
        region = regionprops(mask.astype(np.uint8))[0]
        distance = distance_transform_edt(np.pad(mask_bool, 1))[1:-1, 1:-1]
        radius_values = distance[mask_bool]

        self.assertEqual(measurement["area"], area)
        self.assertAlmostEqual(measurement["x_centroid"], x_centroid)
        self.assertAlmostEqual(measurement["y_centroid"], y_centroid)
        self.assertAlmostEqual(measurement["perimeter"], perimeter)
        self.assertAlmostEqual(measurement["convex_perimeter"], convex_perimeter)
        self.assertAlmostEqual(measurement["major"], major)
        self.assertAlmostEqual(measurement["minor"], minor)
        self.assertAlmostEqual(measurement["feret"], feret)
        self.assertAlmostEqual(measurement["minferet"], minor)
        self.assertAlmostEqual(measurement["eccentricity"], eccentricity)
        self.assertAlmostEqual(geometry["compactness"], expected_compactness)
        self.assertAlmostEqual(geometry["extent"], region.extent)
        self.assertAlmostEqual(geometry["major_axis_length"], region.axis_major_length)
        self.assertAlmostEqual(geometry["minor_axis_length"], region.axis_minor_length)
        self.assertAlmostEqual(geometry["maximum_radius"], radius_values.max())
        self.assertAlmostEqual(geometry["mean_radius"], radius_values.mean())
        self.assertAlmostEqual(geometry["median_radius"], np.median(radius_values))
        self.assertAlmostEqual(geometry["perimeter_area_ratio"], perimeter / area)
        self.assertAlmostEqual(geometry["solidity"], region.solidity)
        self.assertAlmostEqual(
            geometry["circularity"],
            2.0 * np.sqrt(np.pi * area) / perimeter**2,
        )
        self.assertAlmostEqual(
            geometry["sphericity"],
            4.0 * np.pi * area / perimeter**2,
        )
        self.assertAlmostEqual(
            geometry["roughness"],
            1.0 - convex_perimeter / perimeter,
        )

    def test_python_glcm_quantizes_to_64_gray_levels(self) -> None:
        signal = np.array(
            [
                [0, 24, 64, 90, 120],
                [8, 32, 70, 96, 128],
                [12, 36, 75, 104, 136],
                [16, 40, 80, 112, 144],
                [20, 44, 84, 116, 152],
            ],
            dtype=np.float32,
        ) / 255.0
        mask = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=bool,
        )
        gray_levels = 64
        roi_img = np.clip(np.rint(signal * 255.0), 0, 255).astype(np.uint8)
        roi_img[~mask] = 0
        roi_img = np.floor(
            roi_img.astype(np.float64) / 256.0 * gray_levels
        ).astype(np.uint8)
        glcm = graycomatrix(
            roi_img,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=gray_levels,
            symmetric=True,
            normed=True,
        )
        diff_indices = np.abs(
            np.arange(gray_levels)[:, None] - np.arange(gray_levels)[None, :]
        )
        diff_variances = []
        for distance_index in range(glcm.shape[2]):
            for angle_index in range(glcm.shape[3]):
                matrix = glcm[:, :, distance_index, angle_index]
                probabilities = np.bincount(
                    diff_indices.ravel(),
                    weights=matrix.ravel(),
                    minlength=gray_levels,
                ).astype(np.float64)
                index = np.arange(gray_levels, dtype=np.float64)
                mean = float(np.sum(index * probabilities))
                diff_variances.append(
                    float(np.sum(((index - mean) ** 2) * probabilities))
                )
        expected = [
            float(graycoprops(glcm, "ASM").mean()),
            float(graycoprops(glcm, "contrast").mean()),
            float(graycoprops(glcm, "correlation").mean()),
            float(np.mean(diff_variances)),
            float((-glcm[glcm > 0] * np.log(glcm[glcm > 0])).sum() / 4.0),
            float(graycoprops(glcm, "homogeneity").mean()),
        ]

        values = _texture_feature_parameter_values_python(
            signal,
            mask,
            erode_px=2,
        )

        np.testing.assert_allclose(values[:6], expected, rtol=1e-10, atol=1e-10)

    def test_circle_uses_table_circularity_and_sphericity_formula(self) -> None:
        signal = np.ones((64, 64), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (32, 32), 12, 1, -1)

        measurement = _measure_roi_with_python(signal, mask.astype(bool))
        expected_circularity = (
            2.0
            * np.sqrt(np.pi * measurement["area"])
            / measurement["perimeter"] ** 2
        )
        expected_sphericity = (
            4.0 * np.pi * measurement["area"] / measurement["perimeter"] ** 2
        )

        self.assertAlmostEqual(measurement["circ"], expected_circularity)
        self.assertAlmostEqual(measurement["sphericity"], expected_sphericity)
        self.assertGreater(measurement["sphericity"], 0.8)

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
                "Compactness",
                "Extent",
                "Major Axis Length",
                "Minor Axis Length",
                "Maximum Radius",
                "Mean Radius",
                "Median Radius",
                "Perimeter/Area Ratio",
                "Solidity",
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
            first_cyto = result[result["Cell_ID"] == "sample_1_cyto"].iloc[0]
            nuc_mask = _polygon_to_mask(
                np.array([[31, 53], [53, 53], [53, 75], [31, 75]]),
                image.shape,
            )
            cell_mask = _polygon_to_mask(
                np.array([[16, 38], [68, 38], [68, 90], [16, 90]]),
                image.shape,
            )
            expected_cell_area = float(np.count_nonzero(cell_mask))
            expected_cytoplasm_area = expected_cell_area - float(
                np.count_nonzero(nuc_mask)
            )
            self.assertEqual(first_cyto["Area"], expected_cell_area)
            self.assertAlmostEqual(
                first_cyto["Karyoplasmic Ratio"],
                float(np.count_nonzero(nuc_mask)) / expected_cytoplasm_area,
            )
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
