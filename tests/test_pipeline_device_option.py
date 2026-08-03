import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ki67dtc.app_pipeline import run_pipeline


class RunPipelineDeviceTest(unittest.TestCase):
    """驗證 run_pipeline 的 device 字串正確轉成 segment_all 的 use_gpu 布林值。"""

    def _run_and_capture_use_gpu(self, **pipeline_kwargs) -> bool:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            (root / "PC").mkdir(parents=True)

            with (
                patch("ki67dtc.img_prep.segment_all") as segment_all_mock,
                patch("ki67dtc.img_prep.mask2txt_all"),
                patch("ki67dtc.img_prep.combined"),
                patch("ki67dtc.cell_anal.run_all"),
                patch("ki67dtc.app_pipeline.collect_paired_overlay_data",
                      return_value={}),
                patch("ki67dtc.app_pipeline.output_dir",
                      return_value=root / "out"),
            ):
                run_pipeline(root, **pipeline_kwargs)

            segment_all_mock.assert_called_once()
            return segment_all_mock.call_args.kwargs["use_gpu"]

    def test_cpu_disables_gpu(self) -> None:
        self.assertIs(self._run_and_capture_use_gpu(device="cpu"), False)

    def test_gpu_enables_gpu(self) -> None:
        self.assertIs(self._run_and_capture_use_gpu(device="gpu"), True)

    def test_defaults_to_gpu(self) -> None:
        self.assertIs(self._run_and_capture_use_gpu(), True)


if __name__ == "__main__":
    unittest.main()
