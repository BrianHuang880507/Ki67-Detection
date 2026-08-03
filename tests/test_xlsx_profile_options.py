import unittest
from argparse import Namespace
from pathlib import Path

from main import build_parser
from scripts.run_all_data_input import build_command


class XlsxProfileOptionsTest(unittest.TestCase):
    def test_cli_defaults_to_engineer_profile(self) -> None:
        args = build_parser().parse_args(["--data_folder", "demo"])

        self.assertEqual(args.xlsx_version, "engineer")

    def test_cli_accepts_both_profiles_and_underscore_alias(self) -> None:
        parser = build_parser()

        biomedical = parser.parse_args(
            ["--data_folder", "demo", "--xlsx-version", "biomedical"]
        )
        engineer = parser.parse_args(
            ["--data_folder", "demo", "--xlsx_version", "engineer"]
        )
        both = parser.parse_args(
            ["--data_folder", "demo", "--xlsx-version", "both"]
        )

        self.assertEqual(biomedical.xlsx_version, "biomedical")
        self.assertEqual(engineer.xlsx_version, "engineer")
        self.assertEqual(both.xlsx_version, "both")

    def test_batch_command_forwards_xlsx_profile(self) -> None:
        args = Namespace(
            device="cpu",
            nuc_source="dapi",
            ki67_backend="opencv",
            feature_backend="python",
            xlsx_version="biomedical",
            fluor_analy=False,
            ki67=False,
            clean_temp=False,
        )

        command = build_command(
            "python",
            Path("main.py"),
            Path("data/input/demo"),
            args,
        )

        option_index = command.index("--xlsx-version")
        self.assertEqual(command[option_index + 1], "biomedical")

    def test_batch_command_forwards_device(self) -> None:
        args = Namespace(
            device="cpu",
            nuc_source="pc",
            ki67_backend="opencv",
            feature_backend="python",
            xlsx_version="biomedical",
            fluor_analy=False,
            ki67=False,
            clean_temp=False,
        )

        command = build_command(
            "python",
            Path("main.py"),
            Path("data/input/demo"),
            args,
        )

        option_index = command.index("--device")
        self.assertEqual(command[option_index + 1], "cpu")


if __name__ == "__main__":
    unittest.main()
