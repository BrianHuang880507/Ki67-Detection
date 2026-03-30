import argparse
from pathlib import Path

from ki67dtc.img_prep import segment_all, mask2txt_all, combined
from ki67dtc.cell_anal import run_all


def main():
    parser = argparse.ArgumentParser(description="細胞影像分析 Pipeline")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="輸入資料夾名稱或路徑"
    )
    parser.add_argument(
        "--nuc_source",
        type=str,
        default="dapi",
        choices=["pc", "dapi"],
        help="nucleus segmentation 來源（pc 或 dapi，預設 dapi）",
    )
    parser.add_argument("--fluor_analy", action="store_true", help="是否執行螢光分析")
    parser.add_argument("--ki67", action="store_true", help="是否執行 Ki67 判斷")
    parser.add_argument(
        "--ki67_backend",
        type=str,
        default="pyimagej",
        choices=["pyimagej", "opencv"],
        help="Ki67 二值化方法（預設 pyimagej）",
    )
    parser.add_argument("--clean_temp", action="store_true", help="是否清理暫存資料")

    args = parser.parse_args()

    raw_data_arg = Path(args.data_folder)
    candidates = []
    if raw_data_arg.is_absolute():
        candidates.append(raw_data_arg)
    else:
        base_dir = Path("data/input")
        candidates.append(base_dir / raw_data_arg)
        candidates.append(raw_data_arg)

    search_targets = []
    seen = set()
    for candidate in candidates:
        absolute = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        key = str(absolute.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            search_targets.append(absolute)

    data_folder = None
    for candidate in search_targets:
        if candidate.exists() and candidate.is_dir():
            data_folder = candidate
            break

    if data_folder is None:
        print("[錯誤] 找不到資料夾，請確認以下路徑是否存在：")
        for candidate in search_targets:
            print(f" - {candidate}")
        exit(1)

    print("=" * 50)
    print(f"[資訊] 使用資料夾：{data_folder}")
    print(f"[資訊] nucleus 來源：{args.nuc_source}")
    print(f"[資訊] 啟用螢光分析：{args.fluor_analy}")
    print(f"[資訊] 啟用 Ki67 分析：{args.ki67}")
    print(f"[資訊] Ki67 backend：{args.ki67_backend}")
    print(f"[資訊] 清理暫存檔：{args.clean_temp}")
    print("=" * 50)

    # Step 1: segmentation
    print("\n[STEP 1] 執行 segmentation (cyto & nuc)")
    segment_all(data_folder, nuc_source=args.nuc_source)

    # Step 2: mask -> outlines
    print("\n[STEP 2] 將 segmentation npy 轉成 outlines txt")
    mask2txt_all(data_folder)

    # Step 3: combine outlines
    print("\n[STEP 3] 合併 nucleus 與 cytoplasm outlines")
    combined(data_folder)

    # Step 4: geometry & intensity analysis
    print("\n[STEP 4] 幾何參數與螢光/陽性分析")
    run_all(
        data_folder,
        fluor_analy=args.fluor_analy,
        ki67=args.ki67,
        ki67_backend=args.ki67_backend,
        clean_temp=args.clean_temp,
    )

    print("\n[資訊] Pipeline 完成！請檢查輸出結果。")


if __name__ == "__main__":
    main()
