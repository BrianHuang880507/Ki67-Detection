#!/usr/bin/env python3
"""
Batch-run Cellpose segmentation on specified dataset folders.

Each dataset folder should contain a PC/ 子資料夾，與現有 segment_all 預期一致。
Example:
    python script/run_segment_subset.py data/input/0819 data/input/2025-06-19-B4-P6-P10-P14-Ki67-P6-1
"""
import argparse
import os
import sys
from pathlib import Path

if os.name == "nt":
    # Keep behavior consistent with other entry scripts on Windows.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure project root is importable when running as: python script/run_segment_subset.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ki67dtc.img_prep import segment_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Cellpose segmentation (cyto + nuc) on selected datasets. "
            "Nucleus segmentation uses DAPI by default and writes segmentation outputs."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset root(s) that contain PC/ (e.g. data/input/0819)",
    )
    parser.add_argument(
        "--nuc-source",
        type=str,
        default="dapi",
        choices=["pc", "dapi"],
        help="Nucleus segmentation source. Default: dapi",
    )
    parser.add_argument(
        "--dapi-dir-name",
        type=str,
        default="DAPI",
        help="DAPI folder name under each dataset root (used when --nuc-source=dapi).",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        root = Path(ds)
        if not root.exists():
            print(f"[WARN] 路徑不存在，略過：{root}")
            continue
        print(
            f"[INFO] 開始分割：{root} "
            f"(nuc_source={args.nuc_source}, dapi_dir_name={args.dapi_dir_name})"
        )
        segment_all(
            str(root),
            nuc_source=args.nuc_source,
            dapi_dir_name=args.dapi_dir_name,
        )
    print("[INFO] 全部處理完成")


if __name__ == "__main__":
    main()
