import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ki67dtc.utils.io import merge_all_final_csvs


@contextmanager
def _temp_cwd():
    """在暫存目錄下執行測試。

    `output_dir()` 以 `./data/output/<subfolder>` 這個相對路徑組合輸出位置，
    因此測試必須切換工作目錄才不會污染專案的實際輸出資料夾。
    """
    original = Path.cwd()
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            yield Path(tmp_dir)
        finally:
            os.chdir(original)


def _write_final_csv(results_dir: Path, stem: str, statuses: list[str]) -> None:
    """依 cell_status 清單寫出一個 `*_final.csv`。"""
    pd.DataFrame(
        {
            "Cell_ID": [f"{stem}_{index}" for index in range(len(statuses))],
            "Area_nuc": [100.0] * len(statuses),
            "Area_cyto": [1000.0] * len(statuses),
            "cell_status": statuses,
        }
    ).to_csv(results_dir / f"{stem}_final.csv", index=False)


class MergeFinalCsvTest(unittest.TestCase):
    def test_keeps_all_six_outline_statuses(self) -> None:
        with _temp_cwd():
            dataset = Path("sample_dataset")
            results_dir = Path("data/output/results") / dataset.name
            results_dir.mkdir(parents=True, exist_ok=True)
            _write_final_csv(
                results_dir,
                "img1",
                [
                    "full_cell",
                    "nuc_only",
                    "cyto_cut",
                    "cyto_only",
                    "nuc_cut",
                    "both_cut",
                ],
            )

            output_path = merge_all_final_csvs(dataset)

            self.assertIsNotNone(output_path)
            assert output_path is not None
            merged = pd.read_csv(output_path)
            self.assertEqual(
                sorted(merged["cell_status"].tolist()),
                sorted(
                    [
                        "both_cut",
                        "cyto_cut",
                        "cyto_only",
                        "full_cell",
                        "nuc_cut",
                        "nuc_only",
                    ]
                ),
            )

    def test_drops_empty_and_unknown_statuses(self) -> None:
        with _temp_cwd():
            dataset = Path("sample_dataset")
            results_dir = Path("data/output/results") / dataset.name
            results_dir.mkdir(parents=True, exist_ok=True)
            _write_final_csv(
                results_dir, "img1", ["full_cell", "empty", "unknown", "both_cut"]
            )

            output_path = merge_all_final_csvs(dataset)

            assert output_path is not None
            merged = pd.read_csv(output_path)
            self.assertEqual(
                sorted(merged["cell_status"].tolist()), ["both_cut", "full_cell"]
            )

    def test_row_count_matches_sum_of_final_csvs(self) -> None:
        """沒有 empty/unknown 時，cleaned 筆數應等於所有 final 筆數加總。"""
        with _temp_cwd():
            dataset = Path("sample_dataset")
            results_dir = Path("data/output/results") / dataset.name
            results_dir.mkdir(parents=True, exist_ok=True)
            _write_final_csv(results_dir, "img1", ["full_cell", "cyto_only", "nuc_cut"])
            _write_final_csv(results_dir, "img2", ["both_cut", "nuc_only", "cyto_cut"])

            output_path = merge_all_final_csvs(dataset)

            assert output_path is not None
            final_total = sum(
                len(pd.read_csv(path))
                for path in sorted(results_dir.glob("*_final.csv"))
            )
            merged = pd.read_csv(output_path)
            self.assertEqual(len(merged), final_total)

    def test_falls_back_to_dropna_without_cell_status_column(self) -> None:
        with _temp_cwd():
            dataset = Path("sample_dataset")
            results_dir = Path("data/output/results") / dataset.name
            results_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "Cell_ID": ["a", "b", "c"],
                    "Area_nuc": [100.0, None, 120.0],
                    "Area_cyto": [1000.0, 1200.0, None],
                }
            ).to_csv(results_dir / "img1_final.csv", index=False)

            output_path = merge_all_final_csvs(dataset)

            assert output_path is not None
            merged = pd.read_csv(output_path)
            self.assertEqual(merged["Cell_ID"].tolist(), ["a"])

    def test_returns_none_when_no_final_csv(self) -> None:
        with _temp_cwd():
            dataset = Path("sample_dataset")
            (Path("data/output/results") / dataset.name).mkdir(
                parents=True, exist_ok=True
            )

            self.assertIsNone(merge_all_final_csvs(dataset))


if __name__ == "__main__":
    unittest.main()
