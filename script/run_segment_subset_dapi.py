#!/usr/bin/env python3
"""
Batch-run Cellpose segmentation (cyto + nuc) on datasets using DAPI images.

資料夾列表已寫在程式裡（DEFAULT_DATASETS），每個資料夾需包含 DAPI/ 子資料夾。
如需改列表，可直接編輯 DEFAULT_DATASETS，或在命令列另外指定資料夾。

Example:
    python script/run_segment_subset_dapi.py              # 使用內建列表
    python script/run_segment_subset_dapi.py data/input/0819 data/input/P7-P10-P7  # 覆寫列表
"""
import argparse
import sys
from pathlib import Path

# 保證可以匯入本專案模組
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ki67dtc.img_prep import (  # noqa: E402
    CYTO_MODEL_PATH,
    NUC_MODEL_PATH,
    segment,
    output_dir,
    list_files,
)


DEFAULT_DATASETS = [
    # 依你的資料夾列表編輯
    "data/input/2025-06-19-B4-P6-P10-P14-Ki67-P6-1",
    "data/input/2025-06-19-B4-P6-P10-P14-Ki67-P6-2",
    "data/input/2025-06-19-B4-P6-P10-P14-Ki67-P10-1",
    "data/input/2025-06-19-B4-P6-P10-P14-Ki67-P10-2",
    "data/input/2025-06-19-B4-P6-P10-P14-Ki67-P14-1",
    "data/input/2025-06-19-B4-P6-P10-P14-Ki67-P14-2",
    "data/input/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P6",
    "data/input/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P10",
    "data/input/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P14",
    "data/input/P7-P10-P7",
    "data/input/P7-P10-P8",
    "data/input/P7-P10-P9",
    "data/input/P7-P10-P10",
    "data/input/P11-P13-P11",
    "data/input/P11-P13-P12",
    "data/input/P11-P13-P13",
    "data/input/0819",
]


def segment_dapi_folder(dataset_root: Path) -> None:
    dapi_dir = dataset_root / "DAPI"
    if not dapi_dir.exists():
        print(f"[WARN] 找不到 DAPI 資料夾，略過：{dapi_dir}")
        return
    img_files = list_files(dapi_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    if not img_files:
        print(f"[WARN] DAPI 資料夾中無影像，略過：{dapi_dir}")
        return
    seg_dir = output_dir(dataset_root, "segment")
    print(f"[INFO] DAPI 分割（nuc only）：{dataset_root}")
    # DAPI 在彩色圖通常在藍/綠通道，這裡用綠通道 (channels=(1,0))，必要時可改成 (2,0)。
    segment(
        NUC_MODEL_PATH,
        img_files,
        seg_dir,
        "nuc",
        channels=(1, 0),
        cellprob_threshold=-3.0,
        flow_threshold=0.2,
        diameter=None,
        invert=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cellpose segmentation (cyto + nuc) on DAPI folders."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset root(s) that contain DAPI/ (e.g. data/input/0819); empty = use DEFAULT_DATASETS",
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS

    for ds in datasets:
        root = Path(ds)
        if not root.exists():
            print(f"[WARN] 路徑不存在，略過：{root}")
            continue
        segment_dapi_folder(root)
    print("[INFO] 全部處理完成")


if __name__ == "__main__":
    main()
