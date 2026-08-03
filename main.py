import argparse
from pathlib import Path


def run_preprocessing(data_folder: Path, nuc_source: str, device: str = "gpu") -> None:
    """Run segmentation and outline preparation.

    Args:
        data_folder: 輸入資料集資料夾。
        nuc_source: nucleus segmentation 來源，``pc`` 或 ``dapi``。
        device: 分割使用的運算裝置，``gpu`` 或 ``cpu``。
    """
    from ki67dtc.img_prep import combined, mask2txt_all, segment_all

    # Step 1: segmentation
    print("\n[STEP 1] 執行 segmentation (cyto & nuc)")
    segment_all(data_folder, nuc_source=nuc_source, use_gpu=device != "cpu")

    # Step 2: mask -> outlines
    print("\n[STEP 2] 將 segmentation npy 轉成 outlines txt")
    mask2txt_all(data_folder)

    # Step 3: combine outlines
    print("\n[STEP 3] 合併 nucleus 與 cytoplasm outlines")
    combined(data_folder)


def build_parser() -> argparse.ArgumentParser:
    """建立主流程命令列解析器。

    Returns:
        設定完成的 ``ArgumentParser``。
    """
    parser = argparse.ArgumentParser(description="細胞影像分析 Pipeline")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="輸入資料夾名稱或路徑"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="分割使用的運算裝置（gpu 或 cpu，預設 gpu）",
    )
    parser.add_argument(
        "--nuc_source",
        type=str,
        default="pc",
        choices=["pc", "dapi"],
        help="nucleus segmentation 來源（pc 或 dapi，預設 pc）",
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
    parser.add_argument(
        "--feature_backend",
        type=str,
        default="python",
        choices=["pyimagej", "python"],
        help="特徵提取方法（預設 python）",
    )
    parser.add_argument("--clean_temp", action="store_true", help="是否清理暫存資料")
    parser.add_argument(
        "--xlsx-version",
        "--xlsx_version",
        dest="xlsx_version",
        choices=["engineer", "biomedical", "both"],
        default="engineer",
        help=(
            "cleaned CSV 的 XLSX 版本：engineer 產生 *_cleaned_se.xlsx；"
            "biomedical 產生 *_cleaned.xlsx；both 同時產生兩者（預設 engineer）"
        ),
    )
    return parser


def main() -> None:
    """解析命令列參數並執行 Ki67 影像分析主流程。"""
    parser = build_parser()

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
    print(f"[資訊] 運算裝置：{args.device}")
    print(f"[資訊] nucleus 來源：{args.nuc_source}")
    print(f"[資訊] 啟用螢光分析：{args.fluor_analy}")
    print(f"[資訊] 啟用 Ki67 分析：{args.ki67}")
    print(f"[資訊] Ki67 backend：{args.ki67_backend}")
    print(f"[資訊] Feature backend：{args.feature_backend}")
    print(f"[資訊] 清理暫存檔：{args.clean_temp}")
    print(f"[資訊] XLSX 版本：{args.xlsx_version}")
    print("=" * 50)

    run_preprocessing(data_folder, args.nuc_source, args.device)

    # Step 4: geometry & intensity analysis
    print("\n[STEP 4] 幾何參數與螢光/陽性分析")
    from ki67dtc.cell_anal import run_all

    run_all(
        data_folder,
        fluor_analy=args.fluor_analy,
        ki67=args.ki67,
        ki67_backend=args.ki67_backend,
        feature_backend=args.feature_backend,
        clean_temp=args.clean_temp,
        xlsx_profile=args.xlsx_version,
    )

    print("\n[資訊] Pipeline 完成！請檢查輸出結果。")


if __name__ == "__main__":
    main()
