#!/usr/bin/env python3
"""
Batch-run Cellpose counting on DAPI images.

資料夾列表已寫在程式裡（DEFAULT_DATASETS），每個資料夾需包含 DAPI/ 子資料夾。
如需改列表，可直接編輯 DEFAULT_DATASETS，或在命令列另外指定資料夾。

Example:
    python script/run_segment_subset_dapi.py
    python script/run_segment_subset_dapi.py data/input/0819 data/input/P7-P10-P7
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from cellpose import io, models

# 保證可以匯入本專案模組
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ki67dtc.utils.io import list_files  # noqa: E402


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
]

MODEL_TYPE = "cyto3"
CHANNELS = (3, 3)
# Use GUI-like defaults for mask thresholding.
CELLPROB_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4
DIAMETER = None
INVERT = False


def count_cells_in_image(model: models.Cellpose, image_path: Path) -> int:
    img = io.imread(image_path)
    masks, _, _, _ = model.eval(
        img,
        diameter=DIAMETER,
        channels=list(CHANNELS),
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        invert=INVERT,
    )
    if masks is None:
        return 0
    labels = np.unique(masks)
    return int(np.sum(labels > 0))


def count_dapi_folder(
    dataset_root: Path,
    model: models.Cellpose,
) -> list[tuple[str, str, int]]:
    results: list[tuple[str, str, int]] = []
    dapi_dir = dataset_root / "DAPI"
    if not dapi_dir.exists():
        print(f"[WARN] 找不到 DAPI 資料夾，略過：{dapi_dir}")
        return results

    img_files = list_files(dapi_dir, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    if not img_files:
        print(f"[WARN] DAPI 資料夾中無影像，略過：{dapi_dir}")
        return results

    print(f"[INFO] 計數 DAPI 細胞：{dataset_root}")
    for img_path in img_files:
        count = count_cells_in_image(model, img_path)
        print(f"  - {img_path.name}: {count}")
        results.append((dataset_root.name, img_path.name, count))
    return results


def write_results_txt(
    output_path: Path,
    rows: list[tuple[str, str, int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("model=cyto3\n")
        f.write("channels=(3,3)\n")
        f.write("cellprob_threshold=0.0\n")
        f.write("flow_threshold=0.4\n")
        f.write("diameter=None (auto-estimate)\n")
        f.write("dataset\timage\tcell_count\n")
        for dataset, image, count in rows:
            f.write(f"{dataset}\t{image}\t{count}\n")

        total = sum(c for _, _, c in rows)
        f.write(f"\nTOTAL_IMAGES={len(rows)}\n")
        f.write(f"TOTAL_CELLS={total}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cellpose counting on DAPI folders and save one TXT summary."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset root(s) that contain DAPI/ (e.g. data/input/0819); empty = use DEFAULT_DATASETS",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("logs") / "dapi_cell_counts.txt",
        help="Single TXT file path to store all counting results.",
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    model = models.Cellpose(gpu=True, model_type=MODEL_TYPE)

    all_rows: list[tuple[str, str, int]] = []

    for ds in datasets:
        root = Path(ds)
        if not root.exists():
            print(f"[WARN] 路徑不存在，略過：{root}")
            continue
        all_rows.extend(count_dapi_folder(root, model))

    write_results_txt(args.output_txt, all_rows)
    print(f"[INFO] 統計結果已輸出：{args.output_txt.resolve()}")
    print("[INFO] 全部處理完成")


if __name__ == "__main__":
    main()
