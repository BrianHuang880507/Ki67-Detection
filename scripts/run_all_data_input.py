from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    """單一資料集執行結果。"""

    dataset: str
    return_code: int
    elapsed_sec: float


def parse_args() -> argparse.Namespace:
    """解析批次執行 `main.py` 的命令列參數。

    Returns:
        argparse.Namespace: 使用者指定的資料根目錄、執行選項與篩選條件。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the full main.py pipeline, including segmentation, for every "
            "first-level folder under data/input."
        )
    )
    parser.add_argument(
        "--input-root",
        default="data/input",
        help="Root folder that contains datasets (default: data/input).",
    )
    parser.add_argument(
        "--main-script",
        default="main.py",
        help="Full pipeline script path, including segmentation (default: main.py).",
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable used to launch each run.",
    )
    parser.add_argument(
        "--nuc_source",
        type=str,
        default="dapi",
        choices=["pc", "dapi"],
        help="nucleus segmentation source (pc or dapi).",
    )
    parser.add_argument("--fluor_analy", action="store_true", help="Enable fluor analysis.")
    parser.add_argument("--ki67", action="store_true", help="Enable Ki67 analysis.")
    parser.add_argument(
        "--ki67_backend",
        type=str,
        default="pyimagej",
        choices=["pyimagej", "opencv"],
        help="Ki67 binarization backend.",
    )
    parser.add_argument("--clean_temp", action="store_true", help="Clean temp files.")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only run selected dataset folder names.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Skip dataset folder names.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when any dataset run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned commands without executing.",
    )
    return parser.parse_args()


def normalize_names(names: list[str] | None) -> set[str]:
    """將資料夾名稱轉為小寫集合，供 only/exclude 比對使用。

    Args:
        names (list[str] | None): 使用者輸入的資料夾名稱。

    Returns:
        set[str]: 去除空白並轉小寫後的名稱集合。
    """
    if not names:
        return set()
    return {name.strip().lower() for name in names if name.strip()}


def collect_datasets(input_root: Path, only: list[str] | None, exclude: list[str] | None) -> list[Path]:
    """收集要批次執行的第一層資料集資料夾。

    Args:
        input_root (Path): `data/input` 或其他資料集根目錄。
        only (list[str] | None): 只執行的資料夾名稱清單。
        exclude (list[str] | None): 要略過的資料夾名稱清單。

    Returns:
        list[Path]: 排序後的資料集資料夾路徑。

    Raises:
        FileNotFoundError: 當輸入根目錄不存在或不是資料夾時拋出。
    """
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    all_dirs = sorted((p for p in input_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower())

    only_set = normalize_names(only)
    exclude_set = normalize_names(exclude)

    datasets = all_dirs
    if only_set:
        datasets = [p for p in all_dirs if p.name.lower() in only_set]
        selected = {p.name.lower() for p in datasets}
        missing = sorted(only_set - selected)
        if missing:
            print(f"[WARN] --only targets not found: {', '.join(missing)}")

    if exclude_set:
        datasets = [p for p in datasets if p.name.lower() not in exclude_set]

    return datasets


def build_command(
    python_exec: str,
    main_script: Path,
    dataset: Path,
    args: argparse.Namespace,
) -> list[str]:
    """組合執行單一資料集的 `main.py` 命令。

    Args:
        python_exec (str): Python 執行檔路徑。
        main_script (Path): 主流程 `main.py` 路徑。
        dataset (Path): 目標資料集資料夾。
        args (argparse.Namespace): 批次執行參數。

    Returns:
        list[str]: 可交給 `subprocess.run()` 的命令參數。
    """
    cmd = [
        python_exec,
        str(main_script),
        "--data_folder",
        str(dataset),
        "--nuc_source",
        args.nuc_source,
        "--ki67_backend",
        args.ki67_backend,
    ]
    if args.fluor_analy:
        cmd.append("--fluor_analy")
    if args.ki67:
        cmd.append("--ki67")
    if args.clean_temp:
        cmd.append("--clean_temp")
    return cmd


def main() -> int:
    """批次執行 `data/input` 下多個資料集的完整主流程。

    Returns:
        int: 程式結束碼，0 表示全部成功或 dry-run，1 表示至少一個資料集失敗。
    """
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = repo_root / input_root

    main_script = Path(args.main_script)
    if not main_script.is_absolute():
        main_script = repo_root / main_script
    if not main_script.exists():
        print(f"[ERROR] Main script not found: {main_script}")
        return 1

    try:
        datasets = collect_datasets(input_root, args.only, args.exclude)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1
    if not datasets:
        print(f"[INFO] No dataset folders to run under: {input_root}")
        return 0

    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Main script: {main_script}")
    print(f"[INFO] Datasets to run: {len(datasets)}")
    for index, dataset in enumerate(datasets, start=1):
        print(f"  {index:>3}. {dataset.name}")

    results: list[RunResult] = []
    for index, dataset in enumerate(datasets, start=1):
        cmd = build_command(args.python_exec, main_script, dataset, args)
        print("\n" + "=" * 70)
        print(f"[RUN] {index}/{len(datasets)} -> {dataset.name}")
        print(f"[CMD] {' '.join(cmd)}")
        print("=" * 70)

        if args.dry_run:
            results.append(RunResult(dataset=dataset.name, return_code=0, elapsed_sec=0.0))
            continue

        start = time.perf_counter()
        completed = subprocess.run(cmd, cwd=repo_root)
        elapsed = time.perf_counter() - start

        results.append(
            RunResult(
                dataset=dataset.name,
                return_code=completed.returncode,
                elapsed_sec=elapsed,
            )
        )

        if completed.returncode != 0:
            print(f"[FAIL] {dataset.name} (exit={completed.returncode}, {elapsed:.1f}s)")
            if args.stop_on_error:
                print("[INFO] Stop early because --stop-on-error is enabled.")
                break
        else:
            print(f"[OK] {dataset.name} ({elapsed:.1f}s)")

    failures = [r for r in results if r.return_code != 0]
    successes = [r for r in results if r.return_code == 0]

    print("\n" + "#" * 70)
    print("[SUMMARY]")
    print(f"Total planned: {len(datasets)}")
    print(f"Executed: {len(results)}")
    print(f"Success: {len(successes)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("Failed datasets:")
        for item in failures:
            print(f"  - {item.dataset} (exit={item.return_code}, {item.elapsed_sec:.1f}s)")
    print("#" * 70)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
