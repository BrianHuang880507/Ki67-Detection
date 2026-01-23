"""Batch runner that executes the Ki-67 CLI pipeline for multiple folders."""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_FOLDERS = [
    "2025-06-19-B4-P6-P10-P14-Ki67-P10-1",
    "2025-06-19-B4-P6-P10-P14-Ki67-P10-2",
    "2025-06-19-B4-P6-P10-P14-Ki67-P14-1",
    "2025-06-19-B4-P6-P10-P14-Ki67-P14-2",
    "2025-06-19-B4-P6-P10-P14-Ki67-P6-1",
    "2025-06-19-B4-P6-P10-P14-Ki67-P6-2",
    "2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P10",
    "2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P14",
    "2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P6",
]


@dataclass
class StageResult:
    command: Sequence[str]
    returncode: int
    stage: str
    folder: str


def run_command(cmd: Sequence[str], cwd: Path, dry_run: bool) -> int:
    cmd_display = " ".join(str(part) for part in cmd)
    print(f"\n[INFO] Running in {cwd}:\n       {cmd_display}")
    if dry_run:
        print("[INFO] Dry-run enabled; command not executed.")
        return 0
    completed = subprocess.run(cmd, cwd=str(cwd))
    if completed.returncode != 0:
        print(f"[ERROR] Command exited with code {completed.returncode}")
    return completed.returncode


def ensure_directories_exist(paths: Iterable[Path], labels: Iterable[str]):
    for path, label in zip(paths, labels):
        if not path.is_dir():
            raise SystemExit(
                f"[ERROR] {label} does not exist or is not a directory: {path}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Execute main.py, crop_nuclei_from_npy.py, and predict.py sequentially "
            "for multiple folders."
        )
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=DEFAULT_FOLDERS,
        help="List of folder names to process (default: predefined production order).",
    )
    parser.add_argument(
        "--python-exe",
        default=None,
        help="Python interpreter to use for invoking the CLI commands.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root used as working directory when invoking commands.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/input"),
        help="Root directory that contains the input folders.",
    )
    parser.add_argument(
        "--image-subdir",
        default="PC",
        help="Sub-directory inside each input folder containing source images.",
    )
    parser.add_argument(
        "--segment-root",
        type=Path,
        default=Path("data/output/segment"),
        help="Root directory containing cytoplasm segmentation npy files.",
    )
    parser.add_argument(
        "--binary-root",
        type=Path,
        default=Path("data/output/binary"),
        help="Root directory containing Ki-67 binary masks.",
    )
    parser.add_argument(
        "--crops-root",
        type=Path,
        default=Path("data/output/cyto_crops"),
        help="Root directory where cropped cytoplasm outputs are stored.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/output/results"),
        help="Root directory containing cleaned CSV outputs for prediction.",
    )
    parser.add_argument(
        "--csv-name-template",
        default="{folder}_cleaned.csv",
        help=(
            "Filename template for the cleaned CSV. Use {folder} as placeholder. "
            "Final path: <results-root>/<folder>/<filename>."
        ),
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the directory containing the trained model (predict.py --model-dir).",
    )
    parser.add_argument(
        "--model-key",
        default="xgb_concat",
        help="Model key argument forwarded to predict.py.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/output/cyto_crops"),
        help="Root directory passed to predict.py --image-root.",
    )
    parser.add_argument(
        "--predict-output-root",
        type=Path,
        default=Path("predictions"),
        help="Directory where prediction CSV files will be written.",
    )
    parser.add_argument(
        "--predict-output-template",
        default="predictions_{folder}.csv",
        help="Filename template for predict.py --output (supports {folder}).",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Skip running main.py for each folder.",
    )
    parser.add_argument(
        "--skip-crop",
        action="store_true",
        help="Skip running crop_nuclei_from_npy.py for each folder.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip running predict.py for each folder.",
    )
    parser.add_argument(
        "--main-fluor-analy",
        action="store_true",
        help="Forward --fluor_analy flag to main.py.",
    )
    parser.add_argument(
        "--main-ki67",
        action="store_true",
        help="Forward --ki67 flag to main.py.",
    )
    parser.add_argument(
        "--main-clean-temp",
        action="store_true",
        help="Forward --clean_temp flag to main.py.",
    )
    parser.add_argument(
        "--main-extra-args",
        default="",
        help="Additional arguments appended to main.py (space-separated).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining folders even if a command fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def build_command_summaries(args) -> Sequence[str]:
    summaries = []
    if not args.skip_main:
        main_summary = "main.py --data_folder <input-root>/<folder>"
        main_extras = []
        if args.main_fluor_analy:
            main_extras.append("--fluor_analy")
        if args.main_ki67:
            main_extras.append("--ki67")
        if args.main_clean_temp:
            main_extras.append("--clean_temp")
        if args.main_extra_args:
            escaped = args.main_extra_args.replace("{", "{{").replace("}", "}}")
            main_extras.append(escaped)
        if main_extras:
            main_summary += " " + " ".join(main_extras)
        summaries.append(main_summary)
    if not args.skip_crop:
        summaries.append(
            "crop_nuclei_from_npy.py --image-dir <input-root>/<folder>/<image-subdir> "
            "--masks-dir <segment-root>/<folder> --ki67-dir <binary-root>/<folder> "
            "--out-root <crops-root>/<folder>"
        )
    if not args.skip_predict:
        summaries.append(
            "predict.py --model-dir {model_dir} --model-key {model_key} "
            "--csv <results-root>/<folder>/<csv-name> --image-root {image_root} "
            "--output <predict-output-root>/<predict-output-template>"
        )
    return summaries


def main():
    args = parse_args()
    python_exe = args.python_exe or sys.executable or "python"
    project_root = args.project_root.resolve()
    image_root_path = (project_root / args.image_root).resolve()
    predict_output_root = (project_root / args.predict_output_root).resolve()

    print("=" * 80)
    print("[INFO] Batch pipeline runner")
    print(f"[INFO] Using Python executable: {python_exe}")
    print(f"[INFO] Working directory: {project_root}")
    print(f"[INFO] Folders: {', '.join(args.folders)}")
    for summary in build_command_summaries(args):
        print(
            f"[PLAN] {summary.format(model_dir=args.model_dir, model_key=args.model_key, image_root=args.image_root)}"
        )
    print("=" * 80)

    if not args.dry_run:
        predict_output_root.mkdir(parents=True, exist_ok=True)

    command_history: List[StageResult] = []
    for folder in args.folders:
        print("\n" + "-" * 80)
        print(f"[INFO] Processing folder: {folder}")
        print("-" * 80)
        folder_input_dir = (project_root / args.input_root / folder).resolve()
        image_dir = folder_input_dir / args.image_subdir
        masks_dir = (project_root / args.segment_root / folder).resolve()
        ki67_dir = (project_root / args.binary_root / folder).resolve()
        out_root = (project_root / args.crops_root / folder).resolve()
        csv_filename = args.csv_name_template.format(folder=folder)
        csv_path = (project_root / args.results_root / folder / csv_filename).resolve()
        predict_output_name = args.predict_output_template.format(folder=folder)
        predict_output_path = (predict_output_root / predict_output_name).resolve()

        if not args.skip_crop:
            ensure_directories_exist(
                [image_dir.parent, masks_dir.parent, ki67_dir.parent],
                ["Image folder root", "Segment root", "Binary root"],
            )

        if not args.skip_main:
            cmd_main = [
                python_exe,
                "main.py",
                "--data_folder",
                str(folder_input_dir),
            ]
            if args.main_fluor_analy:
                cmd_main.append("--fluor_analy")
            if args.main_ki67:
                cmd_main.append("--ki67")
            if args.main_clean_temp:
                cmd_main.append("--clean_temp")
            if args.main_extra_args:
                cmd_main.extend(shlex.split(args.main_extra_args))
            rc = run_command(cmd_main, project_root, args.dry_run)
            command_history.append(StageResult(cmd_main, rc, "main.py", folder))
            if rc != 0 and not args.continue_on_error:
                break

        if not args.skip_crop and (
            args.continue_on_error
            or (not command_history or command_history[-1].returncode == 0)
        ):
            ensure_directories_exist(
                [image_dir, masks_dir, ki67_dir],
                ["Image directory", "Masks directory", "Ki-67 directory"],
            )
            if not args.dry_run:
                out_root.mkdir(parents=True, exist_ok=True)
            cmd_crop = [
                python_exe,
                "crop_nuclei_from_npy.py",
                "--image-dir",
                str(image_dir),
                "--masks-dir",
                str(masks_dir),
                "--ki67-dir",
                str(ki67_dir),
                "--out-root",
                str(out_root),
            ]
            rc = run_command(cmd_crop, project_root, args.dry_run)
            command_history.append(
                StageResult(cmd_crop, rc, "crop_nuclei_from_npy.py", folder)
            )
            if rc != 0 and not args.continue_on_error:
                break

        if not args.skip_predict and (
            args.continue_on_error
            or (not command_history or command_history[-1].returncode == 0)
        ):
            ensure_directories_exist(
                [csv_path.parent, image_root_path],
                ["CSV directory", "Image root"],
            )
            if not args.dry_run:
                predict_output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd_predict = [
                python_exe,
                "predict.py",
                "--model-dir",
                args.model_dir,
                "--model-key",
                args.model_key,
                "--csv",
                str(csv_path),
                "--image-root",
                str(image_root_path),
                "--output",
                str(predict_output_path),
            ]
            rc = run_command(cmd_predict, project_root, args.dry_run)
            command_history.append(StageResult(cmd_predict, rc, "predict.py", folder))
            if rc != 0 and not args.continue_on_error:
                break

    print("\n" + "=" * 80)
    print("[INFO] Execution summary:")
    for result in command_history:
        status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
        print(f" - {result.stage} [{result.folder}]: {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
