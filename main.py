import argparse
from pathlib import Path
from ki67dtc.img_prep import segment_all, mask2txt_all, combined
from ki67dtc.cell_anal import run_all


def main():
    parser = argparse.ArgumentParser(description="細胞影像分析 Pipeline")
    parser.add_argument("--data_folder", type=str, required=True, help="輸入資料夾路徑")
    parser.add_argument("--fluor_analy", action="store_true", help="是否進行螢光分析")
    parser.add_argument("--ki67", action="store_true", help="是否進行 Ki67 判斷")
    parser.add_argument("--clean_temp", action="store_true", help="是否清理暫存資料")

    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    if not data_folder.exists() or not data_folder.is_dir():
        print(f"[ERROR] 找不到資料夾: {data_folder}")
        exit(1)

    print("=" * 50)
    print(f"[INFO] 測試資料夾: {data_folder}")
    print(f"[INFO] 螢光分析: {args.fluor_analy}")
    print(f"[INFO] Ki67 判斷: {args.ki67}")
    print(f"[INFO] 清理暫存: {args.clean_temp}")
    print("=" * 50)

    # Step 1: segmentation
    print("\n[STEP 1] 執行 segmentation (cyto & nuc)")
    segment_all(data_folder)

    # Step 2: mask -> outlines
    print("\n[STEP 2] 將 segmentation npy 轉成 outlines txt")
    mask2txt_all(data_folder)

    # Step 3: 合併 nucleus & cytoplasm outlines
    print("\n[STEP 3] 合併 nucleus & cytoplasm outlines")
    combined(data_folder)

    # Step 4: 細胞參數 & 螢光強度分析
    print("\n[STEP 4] 執行參數分析 + 螢光強度分析")
    run_all(
        data_folder,
        fluor_analy=args.fluor_analy,
        ki67=args.ki67,
        clean_temp=args.clean_temp,
    )

    print("\n[INFO] Pipeline 執行完成！請檢查輸出結果。")


if __name__ == "__main__":
    main()
