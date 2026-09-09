"""Exp8 報告輸出。圖上沒有的說明文字全部放在這裡。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .brightness import BRIGHT_LABEL, CONTROL_CONDITION, DARK_LABEL


LIMITATIONS = """1. **只有 3 個 donor。** 任何「某個 donor 反應比較大」的說法都是 n=3，只能當觀察，不能當結論。Exp5 已證明 B4 與 B8 在 IDO 上不可分。
2. **passage 不是生物重複。** P5／P6／P7 是同一個 lot 的連續傳代，傳代本身就會改變形狀。圖上三個點是傳代間變異，不是生物信賴區間，所以不畫誤差棒。
3. **單位是像素，不是微米。** 目前沒有 µm/pixel 轉換比例，長度類特徵只能在本批次內比較。
4. **細胞密度是真實的混淆因子。** 濃度會改變細胞數，細胞數又會改變形狀。fig03／fig04 已同時給出扣除密度前後的相關係數；IFN 軸幾乎不受影響，TNF 軸有一部分是密度造成的。
5. **強刺激下沒亮的細胞不一定是真的沒反應。** 也可能是焦平面偏移或分割品質較差。若形狀差異主要出現在與影像品質相關的特徵上，要先排除這個可能。
6. **IDO 影像是 8-bit JPEG，且經過 flat-field 校正。** 校正把細胞層級的左右照明梯度由 23.4% 降到 0.3%，但沒有空白孔參考影像可獨立驗證。
7. **只做關聯與差異描述，沒有做 out-of-fold 預測。** 要回答「形狀能不能預測 IDO」請看 Exp3／Exp4 的交叉驗證結果。
8. **IFN 處理時長在 repo 中沒有紀錄**，待生醫所確認後才能與文獻條件對齊。
9. **本結果定位為探索性描述**，不得作為放行依據、品質認證或臨床決策依據。"""


def _fmt_p(value: float) -> str:
    """p／q 值格式化。"""
    if value is None or not np.isfinite(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def _correlation_table(frame: pd.DataFrame) -> str:
    """形狀與濃度的關聯性表。"""
    lines = [
        "| 名次 | 形狀特徵 | 相關係數 | 扣除細胞密度後 | 保留比例 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in frame.sort_values("rank").itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.feature_label} | {row.spearman_rho:+.3f} | "
            f"{row.rho_density_adjusted:+.3f} | {row.retained_fraction:.0%} |"
        )
    return "\n".join(lines)


def _positive_table(frame: pd.DataFrame) -> str:
    """IDO 陽性比例表。"""
    lines = [
        "| 條件 | 細胞數 | 陽性比例 | 各 donor×passage 範圍 |",
        "|---|---:|---:|---|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.condition} | {int(row.n_cells):,} | {row.positive_fraction:.1%} | "
            f"{row.positive_fraction_min:.1%} – {row.positive_fraction_max:.1%} |"
        )
    return "\n".join(lines)


def _contrast_table(frame: pd.DataFrame) -> str:
    """亮暗形狀差異表。"""
    lines = [
        "| 名次 | 形狀特徵 | 效果量 | 四分位範圍 | 同號影像 | q 值 |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for row in frame.sort_values("rank").itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.feature_label} | {row.effect_median:+.3f} | "
            f"{row.effect_q1:+.2f} – {row.effect_q3:+.2f} | "
            f"{int(row.images_same_sign)}/{int(row.n_images)} | {_fmt_p(row.q_value_bh)} |"
        )
    return "\n".join(lines)


def _passage_table(frame: pd.DataFrame) -> str:
    """donor × passage 的變化幅度，用來分辨 donor 效應與傳代效應。"""
    pivot = frame.pivot_table(index="b_id", columns="passage", values="magnitude")
    header = "| donor | " + " | ".join(f"P{int(column)}" for column in pivot.columns) + " |"
    divider = "|---|" + "---:|" * len(pivot.columns)
    lines = [header, divider]
    for donor, row in pivot.iterrows():
        lines.append(f"| {donor} | " + " | ".join(f"{value:.3f}" for value in row) + " |")
    return "\n".join(lines)


def _threshold_list(metadata: Mapping[str, Any]) -> str:
    """形狀「特別」的門檻清單。"""
    lines = []
    for item in metadata.get("notable_thresholds", []):
        arrow = "高於" if item["direction"] > 0 else "低於"
        lines.append(
            f"- **{item['feature_label']}**：{arrow} {item['threshold']:.4g}"
            f"（對照組中位數 {item['control_median']:.4g}）"
        )
    return "\n".join(lines)


def _notable_fraction_table(frame: pd.DataFrame) -> str:
    """各條件下形狀達標的細胞比例。"""
    labels = list(dict.fromkeys(frame.sort_values("feature_order")["feature_label"]))
    conditions = (
        frame.drop_duplicates("condition").sort_values(["ifn_dose", "tnf_dose"])["condition"].tolist()
    )
    lines = ["| 條件 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for condition in conditions:
        block = frame[frame["condition"] == condition].set_index("feature_label")
        cells = " | ".join(f"{block.loc[label, 'notable_fraction']:.1%}" for label in labels)
        lines.append(f"| {condition} | {cells} |")
    return "\n".join(lines)


def _magnitude_table(frame: pd.DataFrame) -> str:
    """各 donor 的整體形狀變化幅度表。"""
    lines = ["| donor | 中位數 | 最小 | 最大 | passage 數 |", "|---|---:|---:|---:|---:|"]
    for row in frame.drop_duplicates("b_id").sort_values("b_id").itertuples():
        lines.append(
            f"| {row.b_id} | {row.median:.3f} | {row.min:.3f} | {row.max:.3f} | "
            f"{int(row.n_passages)} |"
        )
    return "\n".join(lines)


def render_report(
    *,
    summary: Mapping[str, Any],
    thresholds: Any,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    primary_condition: str,
    high_dose_condition: str,
) -> str:
    """組出 REPORT.md 內容。"""
    contrast = tables["bright_dim_shape"]
    best = contrast.sort_values("rank").iloc[0]
    magnitude = tables["donor_magnitude"].drop_duplicates("b_id").sort_values("median")
    spread = tables["donor_spread"]

    notable_thresholds = metadata.get("notable_thresholds", [])
    n_notable = len(notable_thresholds)
    notable_percentile = float(metadata.get("notable_percentile", 90.0))
    primary_feature = notable_thresholds[0]["feature"] if notable_thresholds else None
    dose_response = tables["shape_dose_response"]
    ifn_curve = dose_response[
        (dose_response["feature"] == primary_feature)
        & (dose_response["curve"].str.startswith("IFN"))
    ].sort_values("dose_ng_ml")
    ecc_control = float(ifn_curve["median"].iloc[0]) if not ifn_curve.empty else float("nan")
    ecc_high = float(ifn_curve["median"].iloc[-1]) if not ifn_curve.empty else float("nan")

    fractions = tables["notable_fraction"]
    per_condition = fractions.groupby("condition")["notable_fraction"].mean()
    top_condition = str(per_condition.idxmax())
    top_fraction = float(per_condition.max())
    ifn100_fraction = float(per_condition.get("IFN100_TNF0", float("nan")))
    tnf_retained = float(tables["shape_tnf_correlations"]["retained_fraction"].median())

    blocks = tables["donor_magnitude"]
    last_passage = int(blocks["passage"].max())
    early = blocks[blocks["passage"] < last_passage]["magnitude"]
    late = blocks[blocks["passage"] == last_passage]["magnitude"]
    passage_low, passage_high = float(early.min()), float(early.max())
    p7_low, p7_high = float(late.min()), float(late.max())

    return f"""# Exp8：刺激濃度、IDO 亮暗與 donor 之間的細胞形狀差異

> 產生時間：{metadata['generated_at']}　|　耗時 {metadata['elapsed_seconds']} 秒
> 只讀取 Exp3／Exp4 既有 artifacts，不重跑 Cellpose、不重算特徵。

## 這份報告回答什麼

1. 刺激濃度對細胞形狀的影響。
2. IDO 有亮或沒亮，在細胞形狀上的差異。
3. 不同 donor 對相同刺激濃度的形狀變化量。

**全程只用 {len(metadata['shape_features'])} 個形狀（幾何）特徵**，不含紋理與亮度。
Exp6 把 41 個特徵混在一起排名時，32 個紋理特徵會把形狀擠掉——最好的形狀特徵
只排到第 14 名。生醫所同仁問的是形狀，因此本實驗全程排除紋理。

## 資料

| 項目 | 數值 |
|---|---|
| 細胞數 | {summary['n_cells']:,} |
| 影像數 | {summary['n_images']} |
| donor × passage | {len(summary['donors'])} × {len(summary['passages'])}（{'、'.join(summary['donors'])}） |
| 刺激條件 | {summary['conditions']} |
| 形狀特徵 | {len(metadata['shape_features'])} |

## 亮與暗怎麼定義

不用百分位硬切。{thresholds.describe()}

- **{DARK_LABEL}**：`IDO 強度 <= {thresholds.dark_max:.2f}`，和完全沒刺激的細胞分不出來。
- **{BRIGHT_LABEL}**：`IDO 強度 > {thresholds.bright_min:.2f}`，在對照組裡只有 1% 會到這個程度。
- 中間灰帶不納入比較。

## 問題 1：刺激濃度對細胞形狀的影響

![IDO 對刺激濃度的反應](figures/fig01_ido_dose.png)

![IDO 陽性細胞比例](figures/fig02_ido_positive_fraction.png)

{_positive_table(tables['ido_positive_fraction'])}

未刺激組本身有 5% 落在門檻之上，這是門檻的定義使然。**TNF-α 單獨作用的兩個條件
只有 6–7%，和未刺激幾乎相同**——TNF-α 單獨不誘導 IDO，只有在 IFN-γ 存在時才放大。

### 形狀與濃度的關聯性

![細胞形狀與 IFN-γ 濃度的關聯性](figures/fig03_shape_ifn.png)

{_correlation_table(tables['shape_ifn_correlations'])}

![細胞形狀與 TNF-α 濃度的關聯性](figures/fig04_shape_tnf.png)

{_correlation_table(tables['shape_tnf_correlations'])}

「扣除細胞密度後」是控制每張影像細胞數之後的偏相關係數。濃度會改變細胞數，
細胞數又會改變形狀，不控制就無法分辨是刺激讓細胞變形，還是細胞變少所以攤得開。
TNF-α 軸另外也控制了 IFN-γ 濃度。

![細胞形狀隨刺激濃度的變化](figures/fig05_shape_dose_response.png)

### 形狀特別的細胞

原本每個濃度都取最接近中位數的細胞，但濃度之間的中位數位移很小
（離心率 {ecc_control:.3f} → {ecc_high:.3f}），肉眼看不出差別。改成用形狀門檻挑細胞：
把達標的那群和典型細胞擺在一起，差異才看得出來。

篩選條件取**兩條劑量軸合併排名的前 {n_notable} 個形狀特徵**（fig03 與 fig04 的名次相加），
且要求兩軸同號；門檻取未刺激對照組的第 {notable_percentile:.0f} 百分位：

{_threshold_list(metadata)}

![形狀特別的細胞](figures/fig06_notable_cells.png)

每格上方兩行是**細胞編號**與刺激條件＋該特徵的值，可以直接回原圖找到同一顆細胞；
完整清單見 `notable_cells.csv`。每一列都在達標區間上**等分位取樣**，所以是從
「剛過門檻」漸變到「明顯特別」，不是全部挑最極端的。

**刻意不挑最極端的那一端。** 實測全資料離心率前 40 名落在 0.9975–0.9989、
長軸長落在 416–541 像素，多數是分割把相鄰細胞併成一個物件的結果，而且沒有
劑量梯度（IFN0 佔 12–25%，和無關聯時的期望值差不多）。挑那一群等於在展示
分割失敗，不是展示生物差異。

![形狀特別的細胞比例](figures/fig13_notable_fraction.png)

{_notable_fraction_table(tables['notable_fraction'])}

影像庫只放得下少數幾顆細胞，容易被質疑是挑出來的；上面的比例才是量化證據。

**這張表的排序值得注意。** 達標比例最高的是 {top_condition}（{top_fraction:.0%}），
不是濃度最高的 IFN100_TNF0（{ifn100_fraction:.0%}）；而且 TNF-α 單獨作用的
IFN0_TNF25／IFN0_TNF50 都高於 IFN25_TNF0。也就是說**形狀變化跟著 TNF-α 走的成分
比跟著 IFN-γ 更多**，這和 fig04 的相關係數一致（離心率對 TNF 0.481、對 IFN 0.479，
緊緻度對 TNF 0.469、對 IFN 0.441）。

但要注意這裡有一部分是細胞密度：TNF-α 讓細胞數下降得比 IFN-γ 明顯
（相關係數 −0.31 對 −0.11），而細胞少的視野裡細胞攤得比較開。fig04 顯示
TNF 軸扣掉密度後只保留約 {tnf_retained:.0%}，IFN 軸則幾乎不受影響。
**IDO 的誘導由 IFN-γ 主導，形狀的變化則和 TNF-α 與細胞密度關係更深，兩者不是同一回事。**

## 問題 2：IDO 亮與暗在形狀上的差異

比較在 **{primary_condition}** 進行，因為這個條件的亮暗人數最平衡。
所有比較都在**同一張影像內**完成：同一張影像的細胞共享 donor、passage、濃度、
拍攝時間、照明與焦平面，把這些配對掉之後剩下的才是細胞本身的差異。
每張影像各算一次效果量，再對這些影像層級的效果量做 Wilcoxon 檢定。

![IDO 亮細胞與暗細胞的形狀差異](figures/fig07_bright_dim_shape.png)

{_contrast_table(tables['bright_dim_shape'])}

差異最大的是 **{best['feature_label']}**，效果量 {best['effect_median']:+.3f}，
{int(best['images_same_sign'])}/{int(best['n_images'])} 張影像方向一致。
效果量 +1 代表亮細胞的該特徵幾乎總是比暗細胞大，0 代表兩群完全重疊。

![IDO 亮細胞與暗細胞的形狀分布](figures/fig08_bright_dim_distribution.png)

![同一張影像中的 IDO 亮細胞與暗細胞](figures/fig09_paired_cells.png)

## 問題 3：不同 donor 的形狀變化量

每個 donor 先減掉**自己同一個 passage 的未刺激對照**（{CONTROL_CONDITION}），
因為 donor 的基線形狀本來就不同。變化量再除以對照組的四分位距，
不同特徵才能放在同一張圖上比較。

![各 donor 的形狀變化量](figures/fig10_donor_delta.png)

![各 donor 的形狀變化量與濃度的關係](figures/fig11_donor_slopes.png)

![各 donor 的整體形狀變化幅度](figures/fig12_donor_magnitude.png)

{high_dose_condition} 下的整體形狀變化幅度（{len(metadata['shape_features'])} 個特徵標準化變化量的絕對值平均）：

{_magnitude_table(tables['donor_magnitude'])}

三個 donor 的中位數落在 {magnitude['median'].min():.3f}–{magnitude['median'].max():.3f}，
而同一個 donor 三個 passage 之間的範圍是
{(magnitude['max'] - magnitude['min']).min():.3f}–{(magnitude['max'] - magnitude['min']).max():.3f}。
**同一個 donor 傳代之間的差距，比三個 donor 之間的差距大好幾倍**，
所以本批資料不足以宣稱哪個 donor 的形狀反應比較強。

拆到 passage 層級就看得很清楚：

{_passage_table(tables['donor_magnitude'])}

**P7 在三個 donor 都是離群值**（P5 與 P6 全部落在 {passage_low:.2f}–{passage_high:.2f}，
P7 則跳到 {p7_low:.2f}–{p7_high:.2f}）。三個 donor 同時出現同一個方向，
表示這是**傳代次數的效應，不是 donor 的差異**。要比較 donor，必須先把 passage
控制住，或改用同一個 passage 的資料。

各特徵×濃度下 donor 之間的差距見 `donor_spread.csv`，
其中 `range_vs_effect` 是「三個 donor 的極差」除以「變化量本身」；
這個比值大於 1 代表 donor 之間的分歧比刺激造成的變化還大。
目前有 {int((spread['range_vs_effect'].abs() > 1).sum())} / {len(spread)} 個組合落在這個範圍。

## 輸出檔案

| 檔案 | 內容 |
|---|---|
| `ido_positive_fraction.csv` | 各條件的 IDO 陽性細胞比例 |
| `shape_ifn_correlations.csv`、`shape_tnf_correlations.csv` | 形狀與濃度的關聯性，含密度校正 |
| `shape_dose_response.csv` | 每個形狀特徵在三條劑量曲線上的影像層級摘要 |
| `bright_dim_shape.csv` | 亮暗形狀差異彙總 |
| `bright_dim_shape_per_image.csv` | 每張影像每個特徵的效果量明細 |
| `shape_delta.csv` | 每個 donor×passage×條件的形狀變化量 |
| `donor_magnitude.csv` | 各 donor 的整體形狀變化幅度 |
| `donor_spread.csv` | 各特徵×濃度下 donor 之間的差距 |
| `notable_feature_ranking.csv` | 兩條劑量軸合併排名，篩選條件的來源 |
| `notable_fraction.csv` | 各條件下形狀達標的細胞比例 |
| `notable_cells.csv` | fig06 用到的細胞編號、條件與特徵值 |
| `paired_cells.csv` | fig09 用到的細胞清單 |
| `figures/` | fig01–fig12 |

## 限制

{LIMITATIONS}
"""
