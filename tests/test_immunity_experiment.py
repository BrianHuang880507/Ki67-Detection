import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from immunity.build_dataset import (
    aggregate_image_level,
    build_manifest,
    calculate_dose_response,
    calculate_oof_predicted_auc,
    load_cell_level_from_cleaned,
    morphology_feature_columns,
    parse_image_filename,
)
from immunity.train_ido_proxy import run_repeated_cross_validation


class ImmunityExperimentTest(unittest.TestCase):
    """驗證 immunity 資料解析、彙整與 leakage-safe CV。"""

    def test_parse_image_filename_extracts_condition_channel_and_index(self) -> None:
        parsed = parse_image_filename(
            "IFN-r 25ng_TNF-a 50ng-DAPI-100X-12.jpg"
        )

        self.assertEqual(parsed["ifn_dose"], 25.0)
        self.assertEqual(parsed["tnf_dose"], 50.0)
        self.assertEqual(parsed["channel"], "dapi")
        self.assertEqual(parsed["image_index"], 12)
        self.assertEqual(parsed["image_key"], "IFN25_TNF50_FOV12")

    def test_manifest_requires_complete_exact_triplets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("PC", "DAPI", "IDO"):
                (root / folder).mkdir()
            names = {
                "PC": "IFN-r 0ng-phase-100X-1.jpg",
                "DAPI": "IFN-r 0ng-DAPI-100X-1.jpg",
                "IDO": "IFN-r 0ng-IDO-100X-1.jpg",
            }
            for folder, name in names.items():
                Image.fromarray(np.zeros((16, 20), dtype=np.uint8)).save(
                    root / folder / name
                )

            manifest = build_manifest(
                root,
                expected_triplets=1,
                expected_conditions={"0,0": 1},
            )

            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest.loc[0, "width"], 20)
            self.assertEqual(manifest.loc[0, "height"], 16)
            self.assertEqual(manifest.loc[0, "condition"], "IFN0_TNF0")

    def test_image_aggregation_outputs_median_and_iqr_only(self) -> None:
        feature_columns = morphology_feature_columns()
        rows = []
        for image_index, image_key in enumerate(("A", "B"), start=1):
            for cell_index in range(4):
                row = {
                    "image_key": image_key,
                    "condition": f"C{image_index}",
                    "IFN_dose": float(image_index),
                    "TNF_dose": 0.0,
                    "image_index": image_index,
                    "IDO_score": float(cell_index + image_index),
                }
                row.update(
                    {
                        column: float(cell_index + offset)
                        for offset, column in enumerate(feature_columns)
                    }
                )
                rows.append(row)
        cells = pd.DataFrame(rows)
        manifest = pd.DataFrame({"image_key": ["A", "B"]})

        images, predictors = aggregate_image_level(
            cells, manifest, min_cells_per_image=3
        )

        self.assertEqual(len(images), 2)
        self.assertEqual(len(predictors), len(feature_columns) * 2)
        self.assertTrue(all(name.endswith(("__median", "__iqr")) for name in predictors))
        self.assertNotIn("IDO_score", predictors)
        self.assertNotIn("IFN_dose", predictors)

    def test_load_cell_level_from_main_cleaned_csv_filters_full_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            morphology = morphology_feature_columns()
            common = {
                "Image": "sample-phase",
                "IDO_BackgroundMean": 0.1,
                "IDO_MeanIntensity": 0.4,
                "IDO_MeanIntensity_BgSub": 0.3,
                **{column: float(index + 1) for index, column in enumerate(morphology)},
            }
            cleaned = pd.DataFrame(
                [
                    {"Cell_ID": "sample_1", "cell_status": "full_cell", **common},
                    {"Cell_ID": "sample_2", "cell_status": "cyto_cut", **common},
                ]
            )
            cleaned_path = root / "sample_cleaned.csv"
            cleaned.to_csv(cleaned_path, index=False)
            manifest = pd.DataFrame(
                [
                    {
                        "pc_stem": "sample-phase",
                        "image_key": "IFN0_TNF0_FOV01",
                        "condition": "IFN0_TNF0",
                        "ifn_dose": 0.0,
                        "tnf_dose": 0.0,
                        "image_index": 1,
                    }
                ]
            )

            cells, qc = load_cell_level_from_cleaned(cleaned_path, manifest)

            self.assertEqual(cells["Cell_ID"].tolist(), ["sample_1"])
            self.assertAlmostEqual(cells.loc[0, "IDO_score"], 0.3)
            self.assertEqual(qc.loc[0, "kept_full_cells"], 1)
            self.assertEqual(qc.loc[0, "status_cyto_cut"], 1)

    def test_dose_response_auc_uses_existing_three_curves(self) -> None:
        conditions = [
            (0, 0, 1.0),
            (25, 0, 2.0),
            (50, 0, 3.0),
            (100, 0, 5.0),
            (0, 25, 2.0),
            (0, 50, 4.0),
            (25, 25, 3.0),
            (25, 50, 5.0),
        ]
        images = pd.DataFrame(
            [
                {"IFN_dose": ifn, "TNF_dose": tnf, "IDO_score": score}
                for ifn, tnf, score in conditions
            ]
        )

        summary, auc = calculate_dose_response(images)

        self.assertEqual(summary["curve"].nunique(), 3)
        self.assertEqual(len(auc), 3)
        ifn_auc = auc.loc[auc["curve"].eq("IFN_response_AUC"), "auc_raw"].iloc[0]
        self.assertAlmostEqual(ifn_auc, 300.0)

    def test_oof_predicted_auc_matches_observed_when_predictions_are_exact(self) -> None:
        conditions = [
            (0, 0, 1.0),
            (25, 0, 2.0),
            (50, 0, 3.0),
            (100, 0, 5.0),
            (0, 25, 2.0),
            (0, 50, 4.0),
            (25, 25, 3.0),
            (25, 50, 5.0),
        ]
        images = pd.DataFrame(
            [
                {"IFN_dose": ifn, "TNF_dose": tnf, "IDO_score": score}
                for ifn, tnf, score in conditions
            ]
        )
        _, observed_auc = calculate_dose_response(images)
        predictions = pd.DataFrame(
            [
                {
                    "validation": "repeated_cv",
                    "repeat": 0,
                    "model": "morphology_ridge",
                    "IFN_dose": ifn,
                    "TNF_dose": tnf,
                    "predicted_IDO_score": score,
                }
                for ifn, tnf, score in conditions
            ]
        )

        detail, summary = calculate_oof_predicted_auc(predictions, observed_auc)

        np.testing.assert_allclose(detail["absolute_auc_error_raw"], 0.0)
        np.testing.assert_allclose(summary["absolute_auc_error_raw_median"], 0.0)

    def test_repeated_cv_generates_one_oof_prediction_per_image_and_model(self) -> None:
        rng = np.random.default_rng(42)
        rows = []
        morphology_columns = ["Area_nuc__median", "Solidity_cyto__iqr"]
        conditions = [
            (0, 0),
            (0, 25),
            (0, 50),
            (25, 0),
            (25, 25),
            (25, 50),
            (50, 0),
            (100, 0),
        ]
        for condition_index, (ifn, tnf) in enumerate(conditions):
            for replicate in range(4):
                feature = condition_index + rng.normal(scale=0.1)
                rows.append(
                    {
                        "image_key": f"C{condition_index}_R{replicate}",
                        "condition": f"IFN{ifn}_TNF{tnf}",
                        "IFN_dose": float(ifn),
                        "TNF_dose": float(tnf),
                        "IDO_score": 0.7 * feature + rng.normal(scale=0.05),
                        morphology_columns[0]: feature,
                        morphology_columns[1]: rng.normal(),
                    }
                )
        images = pd.DataFrame(rows)
        config = {
            "outer_splits": 2,
            "inner_splits": 2,
            "seeds": [42],
            "ridge_alphas": [0.1, 1.0],
            "elasticnet_alphas": [0.01],
            "elasticnet_l1_ratios": [0.5],
            "elasticnet_max_iter": 5000,
            "n_jobs": 1,
        }

        predictions, metrics, coefficients = run_repeated_cross_validation(
            images, morphology_columns, config
        )

        self.assertEqual(predictions["model"].nunique(), 5)
        self.assertTrue(
            (predictions.groupby("model")["image_key"].nunique() == len(images)).all()
        )
        self.assertEqual(len(metrics), 5)
        self.assertFalse(coefficients.empty)


if __name__ == "__main__":
    unittest.main()
