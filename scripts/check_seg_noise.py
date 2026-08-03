"""分割雜訊診斷：回報環境版本 + 同一張影像的細胞質標籤統計。

背景：同一份程式在不同機器上跑出雜訊量差很多時，要先確認差異來自
「環境版本」「程式版本」還是「影像本身」。這支腳本把三者一次印出來，
兩台機器跑同一張影像後直接比對數字即可。

CPU/GPU 本身不是原因：本專案已在同機器上驗證，Cellpose 3.1.1.1 的
CPU 與 GPU 路徑對同一張影像產生的標籤數相同（面積僅差 1-3 px）。

用法（在專案根目錄，啟用專案環境後）：
    python scripts/check_seg_noise.py data/input/<資料集> --limit 3

把整段輸出貼回來比對。
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print_versions() -> None:
    """印出會影響 Cellpose 結果的套件版本與程式版本。"""
    print("=" * 60)
    print("環境版本")
    print("=" * 60)
    print(f"python      : {sys.version.split()[0]}")
    print(f"executable  : {sys.executable}")

    for module_name, label in (
        ("torch", "torch"),
        ("cellpose", "cellpose"),
        ("numpy", "numpy"),
        ("numba", "numba"),
        ("llvmlite", "llvmlite"),
        ("cv2", "opencv"),
    ):
        try:
            module = __import__(module_name)
        except Exception as error:  # noqa: BLE001 - 只是回報，不中斷診斷
            print(f"{label:<12}: 匯入失敗 ({error})")
            continue
        version = getattr(module, "__version__", None) or getattr(
            module, "version", "unknown"
        )
        print(f"{label:<12}: {version}")

    try:
        import torch

        print(f"cuda 可用   : {torch.cuda.is_available()}")
    except Exception:  # noqa: BLE001
        pass

    for model_name in ("model_BDL6_label_new", "model_BDL3_label_dapi"):
        model_file = REPO_ROOT / "model" / model_name
        if not model_file.is_file():
            print(f"{model_name:<12}: 檔案不存在")
            continue
        digest = hashlib.md5(model_file.read_bytes()).hexdigest()
        print(f"{model_name}: {model_file.stat().st_size} bytes  md5={digest}")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        print(f"程式版本    : {commit}{' (有未提交變更)' if dirty else ''}")
    except Exception as error:  # noqa: BLE001
        print(f"程式版本    : 取不到 ({error})")
    print()


def _label_stats(masks: np.ndarray) -> dict[str, float]:
    """統計 mask 的標籤數量與面積分佈。"""
    nonzero = masks[masks > 0]
    if nonzero.size == 0:
        return {"n": 0, "median": 0.0, "lt30": 0, "lt15pct": 0}
    _, areas = np.unique(nonzero, return_counts=True)
    median = float(np.median(areas))
    return {
        "n": int(areas.size),
        "median": median,
        "lt30": int(np.sum(areas < 30)),
        "lt15pct": int(np.sum(areas < 0.15 * median)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="含 PC 子資料夾的資料集目錄")
    parser.add_argument("--limit", type=int, default=3, help="診斷幾張影像")
    args = parser.parse_args()

    _print_versions()

    import cv2
    from cellpose import io as cp_io
    from cellpose import models

    from ki67dtc.img_prep import CYTO_MODEL_INPUT_SIZE, CYTO_MODEL_PATH

    pc_dir = Path(args.dataset_dir) / "PC"
    if not pc_dir.is_dir():
        print(f"[ERROR] 找不到 {pc_dir}")
        return 1

    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    img_files = sorted(
        f
        for f in pc_dir.iterdir()
        if f.suffix.lower() in suffixes
        and "ki67" not in f.stem.lower()
        and "df" not in f.stem.lower()
    )[: args.limit]
    if not img_files:
        print(f"[ERROR] {pc_dir} 沒有可用的 PC 影像")
        return 1

    print("=" * 60)
    print("細胞質分割統計（未套用面積過濾）")
    print("=" * 60)
    print(f"model       : {CYTO_MODEL_PATH}")
    print(f"resize      : {CYTO_MODEL_INPUT_SIZE}")
    print()

    model = models.CellposeModel(gpu=True, pretrained_model=CYTO_MODEL_PATH)
    print(f"cellpose device: {model.device}\n")

    header = (
        f"{'image':<24}{'原始尺寸':>14}{'labels':>8}"
        f"{'median_area':>13}{'area<30':>9}{'area<15%':>10}"
    )
    print(header)
    for image_path in img_files:
        img = cp_io.imread(str(image_path))
        orig_h, orig_w = img.shape[:2]
        target_w, target_h = CYTO_MODEL_INPUT_SIZE
        eval_img = cv2.resize(
            img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        masks, _, _ = model.eval(
            eval_img,
            diameter=None,
            channels=[0, 0],
            cellprob_threshold=0.0,
            flow_threshold=0.4,
        )
        masks = cv2.resize(
            masks.astype(np.int32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        stats = _label_stats(masks)
        print(
            f"{image_path.name:<24}{f'{orig_w}x{orig_h}':>14}{stats['n']:>8}"
            f"{stats['median']:>13.1f}{stats['lt30']:>9}{stats['lt15pct']:>10}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
