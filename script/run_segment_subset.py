#!/usr/bin/env python3
"""
Batch-run Cellpose segmentation on specified dataset folders.

Each dataset folder should contain a PC/ 子資料夾，與現有 segment_all 預期一致。
Example:
    python script/run_segment_subset.py data/input/0819 data/input/2025-06-19-B4-P6-P10-P14-Ki67-P6-1
"""
import argparse
from pathlib import Path

from ki67dtc.img_prep import segment_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cellpose segmentation (cyto + nuc) on selected datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset root(s) that contain PC/ (e.g. data/input/0819)",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        root = Path(ds)
        if not root.exists():
            print(f"[WARN] 路徑不存在，略過：{root}")
            continue
        print(f"[INFO] 開始分割：{root}")
        segment_all(str(root))
    print("[INFO] 全部處理完成")


if __name__ == "__main__":
    main()
