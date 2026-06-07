from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyImageJ and Python feature backends and compare CSV values."
    )
    parser.add_argument("--data-folder", required=True, help="Dataset folder containing PC/.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Comparison output folder. Defaults to data/output/backend_comparison/<dataset>.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Only compare the first N images.",
    )
    parser.add_argument(
        "--worker-backend",
        choices=["pyimagej", "python"],
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    data_folder = Path(args.data_folder).resolve()
    pc_dir = data_folder / "PC"
    if not pc_dir.is_dir():
        raise FileNotFoundError(f"PC folder not found: {pc_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT
        / "data"
        / "output"
        / "backend_comparison"
        / data_folder.name
    )
    images = sorted(
        path
        for path in pc_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]
    return data_folder, output_dir, images


def run_worker(
    backend: str,
    data_folder: Path,
    output_dir: Path,
    images: list[Path],
) -> int:
    from ki67dtc.cell_anal import param_anal

    outline_dir = REPO_ROOT / "data" / "output" / "outline" / data_folder.name
    backend_dir = output_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for image_path in images:
        outline_path = outline_dir / f"{image_path.stem}_merged_cp_outlines.txt"
        if not outline_path.exists():
            print(f"[WARN] Missing outline: {outline_path}")
            continue
        output_path = backend_dir / f"{image_path.stem}_params.csv"
        param_anal(
            image_path,
            outline_path,
            output_path,
            feature_backend=backend,
        )
        processed += 1
    print(f"[INFO] {backend}: processed {processed} image(s)")
    return 0 if processed > 0 else 1


def launch_worker(
    backend: str,
    args: argparse.Namespace,
    data_folder: Path,
    output_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-folder",
        str(data_folder),
        "--output-dir",
        str(output_dir),
        "--worker-backend",
        backend,
    ]
    if args.max_images is not None:
        command.extend(["--max-images", str(args.max_images)])
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator <= 0:
        return np.nan
    return float(np.sum(left_centered * right_centered) / denominator)


def build_comparison(output_dir: Path) -> Path:
    imagej_dir = output_dir / "pyimagej"
    python_dir = output_dir / "python"
    rows: list[dict[str, float | int | str]] = []

    for imagej_path in sorted(imagej_dir.glob("*_params.csv")):
        python_path = python_dir / imagej_path.name
        if not python_path.exists():
            continue
        imagej_df = pd.read_csv(imagej_path)
        python_df = pd.read_csv(python_path)
        merged = imagej_df.merge(
            python_df,
            on="Cell_ID",
            how="inner",
            suffixes=("_pyimagej", "_python"),
        )
        common_features = [
            column
            for column in imagej_df.columns
            if column != "Cell_ID"
            and column in python_df.columns
            and pd.api.types.is_numeric_dtype(imagej_df[column])
            and pd.api.types.is_numeric_dtype(python_df[column])
        ]
        for feature in common_features:
            left = pd.to_numeric(
                merged[f"{feature}_pyimagej"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            right = pd.to_numeric(
                merged[f"{feature}_python"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            valid = np.isfinite(left) & np.isfinite(right)
            if not np.any(valid):
                continue
            left = left[valid]
            right = right[valid]
            delta = right - left
            rows.append(
                {
                    "Image": imagej_path.name.removesuffix("_params.csv"),
                    "Feature": feature,
                    "Valid Pairs": int(len(delta)),
                    "PyImageJ Mean": float(np.mean(left)),
                    "Python Mean": float(np.mean(right)),
                    "Mean Signed Difference": float(np.mean(delta)),
                    "Mean Absolute Difference": float(np.mean(np.abs(delta))),
                    "Median Absolute Difference": float(np.median(np.abs(delta))),
                    "Max Absolute Difference": float(np.max(np.abs(delta))),
                    "Pearson Correlation": pearson_correlation(left, right),
                }
            )

    comparison_path = output_dir / "feature_backend_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    return comparison_path


def main() -> int:
    args = parse_args()
    data_folder, output_dir, images = resolve_paths(args)
    if not images:
        raise FileNotFoundError(f"No PC images found in: {data_folder / 'PC'}")

    if args.worker_backend:
        return run_worker(
            args.worker_backend,
            data_folder,
            output_dir,
            images,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    launch_worker("pyimagej", args, data_folder, output_dir)
    launch_worker("python", args, data_folder, output_dir)
    comparison_path = build_comparison(output_dir)
    print(f"[INFO] Comparison report: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
