"""Exp6 報告輸出：把表格轉成生醫同仁能直接讀的中文 Markdown。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


LIMITATIONS = """1. **相關不等於因果，也不等於「形態能預測 IDO」。** 本實驗只做關聯性排名，沒有做 out-of-fold 預測。要回答「能不能預測」請看 Exp3／Exp4 的交叉驗證結果。
2. **只有 3 個 donor（B4／B7／B8）、3 個 passage。** 693 張影像是技術重複，不是 693 個獨立生物樣本。所有 p 值只描述這批影像內的技術不確定性。
3. **不是完整 factorial design。** 現有條件只夠切出三條一維劑量曲線；IFN 軸固定 TNF=0，TNF 軸固定 IFN=0 或 25。任何「IFN×TNF 交互作用」的說法都超出本批資料。
4. **IDO 影像是 8-bit JPEG。** 灰階量化是硬上限，不同批次的 IDO_score 絕對值不可直接比較。
5. **IDO 通道原本有左右照明梯度，本實驗使用 flat-field 校正後的 `IDO_score_ff`。** 校正把細胞層級 X 位置解釋變異由 23.4% 降到 0.3%，但**沒有空白孔參考影像**可獨立驗證擬合曲面。分析已排除所有純位置欄位（Center_X／Y、BoundingBox 座標）。
6. **外觀特徵中的 intensity 與 texture 來自相位差（PC）通道，不是 IDO 通道。** 因此不存在「用 IDO 預測 IDO」的洩漏；但相位差亮度會受細胞密度與貼附狀態影響。
7. **劑量同時改變形態與 IDO。** 形態與 IDO 的相關可能只是共同原因造成，`rho_within_condition` 欄位就是為了分辨這件事。
8. **少數細胞的 phase 輪廓與 IDO 亮點沒有完全對齊。** 逐顆 PNG 中看得到個別例子（分割輪廓旁邊有一團 IDO 訊號沒被框進去）。已抽 160 張影像量測：在 IFN 刺激條件下，高於背景 5×MAD 的 IDO 像素只有 **9–18%** 落在 cell mask 之外，而 cell mask 只覆蓋畫面的 9–15%，代表**通道之間沒有系統性位移**（相位互相關的 |dx|、|dy| 中位數約 1 px），問題是 Cellpose 在個別視野漏抓或合併細胞。未刺激條件的 83–96% 只是在量測雜訊（該條件的高訊號像素幾乎為 0）。**看單顆細胞圖時請對照輪廓，不要只看亮度。**
9. **IFN 處理時長在 repo 中沒有紀錄**，待生醫所確認後才能與文獻條件對齊。
10. **本結果定位為探索性描述**，不得作為放行依據、品質認證或臨床決策依據。"""


def _fmt(value: float, digits: int = 3) -> str:
    """數值格式化，NaN 顯示為 n/a。"""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_p(value: float) -> str:
    """p／q 值格式化。"""
    if value is None or not np.isfinite(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def _feature_table(frame: pd.DataFrame, top_n: int) -> str:
    """把特徵排名轉成 Markdown 表。"""
    subset = frame.nsmallest(top_n, "rank").sort_values("rank")
    lines = [
        "| 名次 | 外觀特徵 | Spearman ρ | q (BH) | 區塊同號 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in subset.itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.feature_label} | {row.spearman_rho:+.3f} | "
            f"{_fmt_p(row.q_value_bh)} | {int(row.blocks_same_sign)}/{int(row.n_blocks)} |"
        )
    return "\n".join(lines)


def _ido_feature_table(frame: pd.DataFrame, top_n: int) -> str:
    """外觀特徵對 IDO 的排名表。"""
    subset = frame.nsmallest(top_n, "rank").sort_values("rank")
    lines = [
        "| 名次 | 外觀特徵 | ρ（全部影像） | ρ（同一條件內） | q (BH) |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in subset.itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.feature_label} | {row.rho_image_level:+.3f} | "
            f"{row.rho_within_condition:+.3f} | {_fmt_p(row.q_value_bh)} |"
        )
    return "\n".join(lines)


def _dose_response_table(frame: pd.DataFrame) -> str:
    """IDO 劑量曲線表。"""
    lines = [
        "| 曲線 | 劑量 (ng/mL) | 影像數 | 細胞數 | IDO 中位數 | Q1 | Q3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.curve} | {row.dose_ng_ml:.0f} | {int(row.n_images)} | {int(row.n_cells)} | "
            f"{row.ido_median:.2f} | {row.ido_q1:.2f} | {row.ido_q3:.2f} |"
        )
    return "\n".join(lines)


def _bright_dim_table(frame: pd.DataFrame, top_n: int) -> str:
    """亮暗細胞外觀差異表。"""
    subset = frame.nsmallest(top_n, "rank").sort_values("rank")
    lines = [
        "| 名次 | 外觀特徵 | rank-biserial | 亮組中位數 | 暗組中位數 | q (BH) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in subset.itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.feature_label} | {row.rank_biserial:+.3f} | "
            f"{row.median_bright:.3g} | {row.median_dim:.3g} | {_fmt_p(row.q_value_bh)} |"
        )
    return "\n".join(lines)


def _headline_ido(ido_correlations: pd.DataFrame) -> str:
    """IDO 對兩條劑量軸的結論表，rho 與灰階變化量並列。"""
    ff = ido_correlations[ido_correlations["target"] == "IDO_score_ff"]
    lines = [
        "| 劑量軸 | 區塊 | Spearman ρ | p | 影像數 | 區塊同號 | IDO 中位數變化（灰階） |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in ff.itertuples():
        lines.append(
            f"| {row.dose_axis_title} | {row.block} | **{row.spearman_rho:+.3f}** | "
            f"{_fmt_p(row.p_value)} | {int(row.n_images)} | "
            f"{int(row.blocks_same_sign)}/{int(row.n_blocks)} | "
            f"{row.median_at_min:.2f} → {row.median_at_max:.2f}"
            f"（{row.delta_grey_levels:+.2f}，{row.dose_min:.0f}→{row.dose_max:.0f} ng/mL） |"
        )
    return "\n".join(lines)


def render_report(
    *,
    summary: Mapping[str, Any],
    slices: Mapping[str, pd.DataFrame],
    tables: Mapping[str, pd.DataFrame],
    splits: Mapping[str, Mapping[str, Any]],
    top_n: int,
    decile_pct: float,
    metadata: Mapping[str, Any],
) -> str:
    """組出完整的 REPORT.md 內容。"""
    ido_correlations = tables["ido_dose_correlations"]
    feature_dose = tables["feature_dose_correlations"]
    scalar_rows = ido_correlations[
        (ido_correlations["target"] == "IDO_score_scalar")
        & (ido_correlations["block"] == "全部")
    ]
    scalar_note = "、".join(
        f"{row.dose_axis} ρ = {row.spearman_rho:+.3f}" for row in scalar_rows.itertuples()
    )

    sections: list[str] = []
    sections.append(
        f"""# Exp6：IDO／外觀特徵與 IFN-γ／TNF-α 的關聯性，以及 IDO 亮暗細胞影像庫

> 產生時間：{metadata['generated_at']}　|　耗時 {metadata['elapsed_seconds']} 秒
> 本實驗**只讀取** Exp3／Exp4 既有 artifacts，不重跑 Cellpose、不重算特徵、不覆寫上游檔案。

## 這份報告回答什麼

生醫所同仁提出的兩個問題：

1. IDO 跟 IFN-γ／TNF-α 有多強的關聯？細胞外觀特徵跟 IFN-γ／TNF-α 又有多強的關聯（前 {top_n} 名長條圖）？
2. 把 IDO 亮的細胞與不亮的細胞分別切出來看。

## 資料

| 項目 | 數值 |
|---|---|
| 細胞數 | {summary['n_cells']:,} |
| 影像數 | {summary['n_images']} |
| donor × passage | {len(summary['donors'])} × {len(summary['passages'])}（{'、'.join(summary['donors'])}；P{'、P'.join(str(value) for value in summary['passages'])}） |
| 刺激條件 | {summary['conditions']} |
| 外觀特徵數 | {summary['n_appearance_features']} |
| 每張影像細胞數 | 中位數 {summary['cells_per_image_median']:.0f}（{summary['cells_per_image_min']}–{summary['cells_per_image_max']}） |

**分析單位是「一張影像」**：劑量施加在整個孔上，同一張影像內的細胞共享同一個劑量。
把 {summary['n_cells']:,} 顆細胞當成獨立觀測會把 p 值灌到沒有意義，因此所有相關係數都先把
每張影像的細胞取中位數。

**劑量軸不是完整 factorial design**，只能切出兩條乾淨的軸：

- IFN 軸：TNF-α 固定 0，IFN-γ 0／25／50／100，n = {len(slices['ifn'])} 張影像。
- TNF 軸：IFN-γ 固定 0 或 25，TNF-α 0／25／50，n = {len(slices['tnf'])} 張影像；
  相關係數在 IFN 區塊內計算（partial Spearman），避免拿 IFN 差異冒充 TNF 效果。

## 問題 1a：IDO 對 IFN-γ／TNF-α 的關聯性

{_headline_ido(ido_correlations)}

對照組（僅扣純量背景、未做 flat-field 校正的 `IDO_score_scalar`）：{scalar_note}。

![IDO 劑量反應](figures/fig01_ido_vs_dose.png)

{_dose_response_table(tables['ido_dose_response'])}

## 問題 1b：外觀特徵對 IFN-γ／TNF-α 的關聯性（前 {top_n} 名）

![前十名合併](figures/fig04_top{top_n}_features_combined.png)

### 對 IFN-γ 劑量（TNF-α = 0）

![IFN 前十名](figures/fig02_top{top_n}_features_ifn.png)

{_feature_table(feature_dose[feature_dose['dose_axis'] == 'IFN'], top_n)}

### 對 TNF-α 劑量（固定 IFN-γ）

![TNF 前十名](figures/fig03_top{top_n}_features_tnf.png)

{_feature_table(feature_dose[feature_dose['dose_axis'] == 'TNF'], top_n)}

### 附帶：外觀特徵對 IDO 本身的關聯性

`ρ（同一條件內）` 控制刺激條件後才算。若某個特徵在左邊很高、右邊掉到接近 0，
代表它跟的是劑量，不是 IDO 本身。

![特徵對 IDO](figures/fig05_top{top_n}_features_vs_ido.png)

{_ido_feature_table(tables['feature_ido_correlations'], top_n)}
"""
    )

    if splits:
        global_counts = splits.get("global", {}).get("labelled_counts", {})
        matched_counts = splits.get("matched", {}).get("labelled_counts", {})
        sections.append(
            f"""## 問題 2：IDO 亮／不亮的細胞影像

兩種切法都做，因為它們回答的不是同一件事：

| 切法 | 定義 | 亮 | 暗 | 回答什麼 |
|---|---|---:|---:|---|
| `global` | 全部細胞取 IDO_score_ff 上／下 {decile_pct:.0f}% | {global_counts.get('bright', 0):,} | {global_counts.get('dim', 0):,} | 最直觀的「亮 vs 不亮」，但亮的那群幾乎必然來自 IFN-γ 刺激組 |
| `matched` | 在每個 donor×passage×條件內取上／下 {decile_pct:.0f}% | {matched_counts.get('bright', 0):,} | {matched_counts.get('dim', 0):,} | 劑量被固定住，剩下的差異才是「同樣刺激下，為什麼有些細胞亮」 |

### 去背版本（背景已移除）

白底＝背景已移除，綠色＝IDO 強度，藍色＝細胞核，紅線＝分割輪廓。
綠色飽和值只由細胞內像素決定，所以不會被大片白色背景拉暗。

![global 去背](figures/fig_gallery_global_ido_nobg.png)

![matched 去背](figures/fig_gallery_matched_ido_nobg.png)

### 整張視野去背：未刺激 vs. 強刺激

保留細胞在視野中的原始位置，看得到密度與排列。只畫通過 QC、有配到細胞核的細胞，
所以畫面上看到的細胞就是統計用到的細胞。

![整張視野去背](figures/fig07_fov_background_removed.png)

全尺寸單張 PNG 放在 `fov_background_removed/`，清單見 `fov_background_removed_selected.csv`。

### global：不分刺激條件

![global IDO 影像庫](figures/fig_gallery_global_ido.png)

![global 相位差影像庫](figures/fig_gallery_global_phase.png)

![global 分割後 IDO](figures/fig_gallery_global_ido_segmented.png)

每顆細胞另外輸出一張五格 PNG（phase 原圖／phase 分割／IDO 原圖／IDO 分割（黑底）／
IDO 去背（白底）），放在 `cell_crops_global/`，共 {splits.get('global', {}).get('n_crop_files', 0)} 張。
對應的細胞清單與原始影像路徑見 `bright_dim_selected_cells_global.csv`。

### matched：同一刺激條件內

![matched IDO 影像庫](figures/fig_gallery_matched_ido.png)

![matched 相位差影像庫](figures/fig_gallery_matched_phase.png)

### 同一條件下，亮細胞與暗細胞的外觀差多少

rank-biserial 效果量：+1 代表亮組該特徵幾乎總是比暗組大，0 代表兩組完全重疊。

![亮暗外觀差異](figures/fig06_top{top_n}_bright_vs_dim_matched.png)

{_bright_dim_table(splits['matched']['contrast'], top_n)}
"""
        )

    sections.append(
        f"""## 輸出檔案

| 檔案 | 內容 |
|---|---|
| `ido_dose_correlations.csv` | IDO（校正前後）對兩條劑量軸的相關係數 |
| `ido_dose_response.csv` | 三條描述性劑量曲線的影像層級摘要 |
| `feature_dose_correlations.csv` | 全部 {summary['n_appearance_features']} 個外觀特徵 × 2 條劑量軸的完整排名 |
| `feature_ido_correlations.csv` | 外觀特徵對 IDO 的關聯性（含條件內版本） |
| `image_level_dataset.csv` | 影像層級資料（一張影像一列） |
| `bright_dim_features_*.csv` | 亮／暗細胞的外觀特徵比較 |
| `bright_dim_selected_cells_*.csv` | 影像庫實際用到的細胞清單與原始影像路徑 |
| `fov_background_removed_selected.csv` | 整張視野去背圖用到的視野清單 |
| `cell_crops_global/` | 逐顆細胞的五格 PNG（含白底去背） |
| `fov_background_removed/` | 全尺寸整張視野去背 PNG |
| `figures/` | 全部圖表 |
| `run_metadata.json`、`run.log` | 執行環境與紀錄 |

## 限制

{LIMITATIONS}
"""
    )
    return "\n".join(sections)
