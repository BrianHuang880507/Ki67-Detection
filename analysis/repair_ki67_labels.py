"""Regenerate Ki67 masks and labels without rerunning PC feature extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ki67dtc.cell_anal import (
    add_outline_status_column,
    detect_ki67_positive,
    ki67_binarize,
    merge_ki67_labels,
)
from ki67dtc.utils.io import merge_all_final_csvs


DEFAULT_DATASETS = [
    "ki67trainset-20250806",
    "ki67trainset-20250807",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate Ki67 binary masks, cell labels, cell_status, and "
            "cleaned CSV files while reusing existing PC feature tables."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset folder names under data/input.",
    )
    parser.add_argument(
        "--ki67-backend",
        choices=["pyimagej", "opencv"],
        default="pyimagej",
    )
    return parser.parse_args()


def repair_dataset(dataset: str, backend: str) -> None:
    data_dir = PROJECT_ROOT / "data" / "input" / dataset
    mapping_path = data_dir / "image_mapping.csv"
    pc_dir = data_dir / "PC"
    ki67_dir = data_dir / "KI67"
    outline_dir = PROJECT_ROOT / "data" / "output" / "outline" / dataset
    result_dir = PROJECT_ROOT / "data" / "output" / "results" / dataset
    binary_dir = PROJECT_ROOT / "data" / "output" / "binary" / dataset

    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)

    mapping = pd.read_csv(mapping_path)
    repaired = 0
    for _, row in mapping.iterrows():
        pc_name = str(row.get("PC_Name", "")).strip()
        ki67_name = str(row.get("KI67_Name", "")).strip()
        if not pc_name or not ki67_name or pc_name == "nan" or ki67_name == "nan":
            continue

        pc_path = pc_dir / pc_name
        ki67_path = ki67_dir / ki67_name
        outline_path = (
            outline_dir / f"{Path(pc_name).stem}_merged_cp_outlines.txt"
        )
        final_csv = result_dir / f"{Path(pc_name).stem}_final.csv"
        if not all(
            path.exists()
            for path in (pc_path, ki67_path, outline_path, final_csv)
        ):
            print(f"[WARN] Skip incomplete mapping row: {pc_name} / {ki67_name}")
            continue

        mask_path = ki67_binarize(ki67_path, backend=backend)
        label_path = binary_dir / f"{Path(pc_name).stem}_label.txt"
        detect_ki67_positive(outline_path, mask_path, label_path)
        merge_ki67_labels(final_csv, label_path, final_csv)
        add_outline_status_column(final_csv, outline_path, pc_path)
        repaired += 1

    merge_all_final_csvs(data_dir)
    print(f"[INFO] {dataset}: repaired {repaired} images")


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        repair_dataset(dataset, args.ki67_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
