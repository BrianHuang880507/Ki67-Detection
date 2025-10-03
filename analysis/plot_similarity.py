#!/usr/bin/env python
"""Plot DF vs LT similarity metrics as line charts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


def _configure_font(preferred: str | None = None) -> str | None:
    """設定 Matplotlib 字型，避免中文缺字。"""
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang TC",
        "PingFang HK",
        "PingFang SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Source Han Sans TW",
    ])
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        rcParams["font.family"] = name
        rcParams["axes.unicode_minus"] = False
        return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="繪製 DF / LT 螢光相似度折線圖")
    parser.add_argument("--input", required=True, help="compare_fluor 產出的 CSV")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["pearson", "spearman", "cosine"],
        help="要繪製的相似度欄位 (預設: pearson spearman cosine)",
    )
    parser.add_argument("--font", help="自訂字型名稱 (選填)")
    parser.add_argument("--output", help="(選用) 存成圖檔的路徑，例如 analysis/similarity.png")
    parser.add_argument("--show", action="store_true", help="顯示互動視窗")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    font_name = _configure_font(args.font)
    if font_name:
        print(f"[資訊] 使用字型：{font_name}")
    else:
        print("[警告] 找不到可用的中文字型，圖中文字可能無法完整顯示。")

    metrics = [m for m in args.metrics if m in df.columns]
    if not metrics:
        raise SystemExit("在 CSV 中找不到指定的相似度欄位。")

    df = df.sort_values("metric")
    x = range(len(df))

    plt.figure(figsize=(10, 5))
    for col in metrics:
        plt.plot(x, df[col], marker="o", label=col)

    plt.xticks(x, df["metric"], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("相似度")
    plt.title("DF vs LT 螢光指標相似度")
    plt.legend()
    plt.tight_layout()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"[資訊] 已輸出圖檔：{out_path}")

    if args.show or not args.output:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
