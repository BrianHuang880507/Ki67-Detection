# Ki67 特徵與特徵參數定義

本文件整理 Ki67 預測流程中「特徵」與「特徵參數」的命名、意義與公式。

---

## 命名原則

**特徵**指的是具有生物或影像意義的概念群，例如 `Intensity Distribution Features`、`Nuclear Sub-Region Features`、`Edge / Halo Features`。  
**特徵參數**指的是實際被量測、寫入 CSV、後續可進模型的數值欄位，例如 `Mean_nuc`、`CV_cyto`、`Nuc Cyto Mean Ratio`。

---

## 一、Intensity Distribution Features

此特徵描述 ROI 內像素強度分布形狀。Phase contrast 中像素強度可視為細胞厚度、折射率、乾質量與核/質紋理狀態的間接 proxy。每個參數可分別量測於 nucleus 與 cytoplasm。

### 基礎強度參數

| 特徵參數    | 定義                                 | 公式 / 來源                                   |
| ----------- | ------------------------------------ | --------------------------------------------- |
| `Mean`      | ROI 內平均灰階值                     | ImageJ `Mean`；概念式：`μ = mean(I_i)`        |
| `StdDev`    | ROI 內灰階標準差                     | ImageJ `StdDev`                               |
| `Min`       | ROI 內最小灰階值                     | ImageJ `Min`                                  |
| `Max`       | ROI 內最大灰階值                     | ImageJ `Max`                                  |
| `IntDen`    | Integrated Density                   | ImageJ `IntDen`，通常對應 `Area × Mean`       |
| `RawIntDen` | Raw Integrated Density               | ImageJ `RawIntDen`，對應 ROI 內原始像素值總和 |
| `CV`        | 變異係數，表示相對於平均值的強度變異 | ImageJ macro：`CV = StdDev / Mean`            |
| `Range`     | 強度範圍                             | ImageJ macro：`Range = Max - Min`             |

---

### 分布形狀參數

| 特徵參數   | 定義                           | 公式 / 來源                 |
| ---------- | ------------------------------ | --------------------------- |
| `P10`      | ROI intensity 第 10 百分位     | ImageJ histogram percentile |
| `P25`      | ROI intensity 第 25 百分位     | ImageJ histogram percentile |
| `P75`      | ROI intensity 第 75 百分位     | ImageJ histogram percentile |
| `P90`      | ROI intensity 第 90 百分位     | ImageJ histogram percentile |
| `IQR80`    | robust intensity range         | `IQR80 = P90 - P10`         |
| `Entropy`  | 灰階 histogram Shannon entropy | `H = -Σ p_k log2(p_k)`      |
| `Skewness` | 強度分布偏態                   | ImageJ `Skew`               |
| `Kurtosis` | 強度分布峰態                   | ImageJ `Kurt`               |

### 核質強度比與差值

這些參數由 nucleus 與 cytoplasm 的 ImageJ 量測值組合而成，用來描述核與細胞質之間的強度差異。

| 特徵參數                      | 定義                              | 公式                             |
| ----------------------------- | --------------------------------- | -------------------------------- |
| `Nuc Cyto Mean Ratio`         | 核區平均強度 / 細胞質平均強度     | `Mean_nuc / Mean_cyto`           |
| `Nuc Cyto IntDen Ratio`       | 核區 IntDen / 細胞質 IntDen       | `IntDen_nuc / IntDen_cyto`       |
| `Nuc Cyto RawIntDen Ratio`    | 核區 RawIntDen / 細胞質 RawIntDen | `RawIntDen_nuc / RawIntDen_cyto` |
| `Nuc Cyto Entropy Difference` | 核區 entropy 與細胞質 entropy 差  | `Entropy_nuc - Entropy_cyto`     |
| `Nuc Cyto CV Difference`      | 核區 CV 與細胞質 CV 差            | `CV_nuc - CV_cyto`               |

---

## 二、Nuclear Sub-Region Features

此特徵描述 nucleus mask 的幾何與強度狀態。Ki67 是核內蛋白，因此核區相關參數通常比單純細胞質形狀更接近 Ki67 分子訊號。

| 特徵參數                  | 定義                             | 公式 / 來源                                     |
| ------------------------- | -------------------------------- | ----------------------------------------------- |
| `Area_nuc`                | 核面積                           | ImageJ `Area`                                   |
| `Circular Diameter_nuc`   | 核等效圓直徑                     | `2 × sqrt(Area_nuc / π)`                        |
| `Feret Length_nuc`        | 核最大 Feret 直徑                | ImageJ `Feret`                                  |
| `Feret Width_nuc`         | 核最小 Feret 直徑                | ImageJ `MinFeret`                               |
| `Aspect Ratio_nuc`        | 核長寬比                         | ImageJ `AR` 或 `Major / Minor`                  |
| `Eccentricity_nuc`        | 核橢圓離心率                     | `sqrt(1 - (Minor / Major)^2)`                   |
| `Sphericity_nuc`          | 核圓形程度                       | ImageJ `Circ.`                                  |
| `Roughness_nuc`           | 核輪廓粗糙度 proxy               | `Convex Perimeter_nuc / Perimeter_nuc`          |
| `CV_nuc`                  | 核區強度變異係數                 | `StdDev_nuc / Mean_nuc`                         |
| `Entropy_nuc`             | 核區強度 entropy                 | `-Σ p_k log2(p_k)`                              |
| `Nucleus Centroid Offset` | 核中心偏離 whole-cell 中心的程度 | `distance(C_nuc, C_cell) / sqrt(Area_cell / π)` |

`Nucleus Centroid Offset` 越大代表 nucleus 越偏離細胞中心，可能反映細胞極性、遷移或貼附狀態差異。

---

## 三、Edge / Halo Features

此特徵描述 phase contrast 中細胞邊緣的 halo、邊界明暗與銳利程度。Halo 可反映細胞厚度、貼附狀態、圓化程度與焦距差異。

目前設計會以 cell mask 產生 inner ring 與 outer ring，並使用 ImageJ 量測 ring intensity。

| 特徵參數                | 定義                                  | 公式 / 來源                                                           |
| ----------------------- | ------------------------------------- | --------------------------------------------------------------------- |
| `Halo Outer Mean`       | cell 外側 halo ring 平均強度          | ImageJ 量測 outer ring `Mean`                                         |
| `Halo Outer StdDev`     | cell 外側 halo ring 強度變異          | ImageJ 量測 outer ring `StdDev`                                       |
| `Halo Inner Mean`       | cell 內側邊緣 ring 平均強度           | ImageJ 量測 inner ring `Mean`                                         |
| `Halo Inner StdDev`     | cell 內側邊緣 ring 強度變異           | ImageJ 量測 inner ring `StdDev`                                       |
| `Halo Inner Outer Diff` | 內外 halo 強度差                      | `abs(Halo Outer Mean - Halo Inner Mean)`                              |
| `Halo Width`            | halo 回到背景強度附近所需的 ring 寬度 | 最小 `w`，使 ring mean `<= background mean + 1.5 × background StdDev` |
| `Edge Sharpness`        | 細胞邊界銳利程度                      | ImageJ `Find Edges` 後，在 boundary ring 量測 `Mean`                  |

---

## 四、Texture Features

此特徵描述 ROI 內像素強度的空間排列，而不只是分布形狀。PDF 建議可在 whole-cell、nucleus、cytoplasm 分別量測，並先對 mask 做 erosion 以避開 phase contrast 邊界 halo。

備份 pipeline 目前已接入 GLCM / Haralick 與 LBP 參數。GLCM 使用 PyImageJ `imagej-ops` 計算四方向平均值；LBP 使用 ImageJ macro 依 `P=8, R=1` 計算 8-neighbor binary code，再輸出 ROI 內 histogram 與 summary。

---

### GLCM / Haralick

GLCM 是 gray-level co-occurrence matrix，用來描述相鄰像素灰階組合的出現頻率。

| 特徵參數                   | 定義             | 常見公式                     |
| -------------------------- | ---------------- | ---------------------------- |
| `GLCM Contrast`            | 灰階局部變化強度 | `Σ(i-j)^2 P(i,j)`            |
| `GLCM ASM`                 | 紋理均勻度       | `Σ P(i,j)^2`                 |
| `GLCM Entropy`             | 紋理複雜度       | `-Σ P(i,j) log(P(i,j))`      |
| `GLCM Homogeneity`         | 相鄰像素相似程度 | `Σ P(i,j) / (1 + abs(i-j))`  |
| `GLCM Correlation`         | 灰階方向性相關   | GLCM correlation             |
| `GLCM Difference Variance` | 相鄰灰階差異變異 | Haralick difference variance |

### LBP

LBP 是 local binary pattern，描述每個像素與周圍鄰居的局部紋理關係。

| 特徵參數            | 定義                                  |
| ------------------- | ------------------------------------- |
| `LBP Hist Bin_*`    | ROI 內 LBP code histogram 各 bin 比例 |
| `LBP Uniform Ratio` | uniform pattern 佔比                  |
| `LBP Mean`          | LBP code 平均值                       |
| `LBP StdDev`        | LBP code 標準差                       |
| `LBP Entropy`       | LBP code histogram entropy            |

目前 `LBP Hist Bin 00` 到 `LBP Hist Bin 15` 會把 0-255 的 LBP code 以每 16 個 code 合併成一個 bin，以降低欄位數與小 ROI 的 histogram 稀疏度。

### 其他紋理參數

| 特徵參數            | 定義                         | 目前狀態 |
| ------------------- | ---------------------------- | -------- |
| `Tamura Coarseness` | 紋理粗細尺度                 | 未導入   |
| `Zernike Moments`   | 圓形區域內形狀與紋理的矩特徵 | 未導入   |

---

## 五、Population / Neighbourhood Context Features

此特徵描述單張影像 field of view 內的整體細胞狀態與局部鄰近關係。PDF 指出 population-level heterogeneity 可能比單顆細胞參數更能反映增殖能力。

| 特徵參數                         | 定義                            | 公式                                               |
| -------------------------------- | ------------------------------- | -------------------------------------------------- |
| `Image Confluency`               | 影像內細胞覆蓋比例              | `Σ Area_cell / Image Area`                         |
| `Population Area CV`             | 同張影像 cell area 的變異係數   | `std(Area_cell) / mean(Area_cell)`                 |
| `Population Circularity CV`      | 同張影像 circularity 的變異係數 | `std(Circularity_cell) / mean(Circularity_cell)`   |
| `Nearest Neighbor Distance`      | 最近鄰細胞距離                  | `min distance(C_i, C_j)`                           |
| `Nearest Neighbor Distance Norm` | 標準化最近鄰距離                | `Nearest Neighbor Distance / median cell diameter` |
| `Local Neighbor Count`           | 固定半徑內鄰近細胞數            | `count(distance <= radius)`                        |
| `Local Density`                  | 局部細胞密度                    | `Local Neighbor Count / (πr^2)`                    |
| `Cluster Size`                   | 該細胞所屬 cluster 的細胞數     | 根據 centroid distance graph 分群                  |
| `Cluster Size Norm`              | 標準化 cluster size             | `Cluster Size / total cell count`                  |
| `Largest Cluster Ratio`          | 最大 cluster 佔比               | `largest cluster size / total cell count`          |

---

## 六、Protrusion / Shape Complexity Features

此特徵描述 whole-cell boundary 的突起、凹陷與輪廓複雜度。細胞進入 mitosis 前常會收回突起並圓化；貼附良好或遷移中的細胞則可能有更複雜的邊界。

| 特徵參數                    | 定義                 | 公式 / 方法                                      |
| --------------------------- | -------------------- | ------------------------------------------------ |
| `Protrusion Count`          | 凸包缺陷數量         | `convexity defects depth > threshold` 的數量     |
| `Mean Convex Defect Depth`  | 平均凸包缺陷深度     | `mean(valid defect depths)`                      |
| `Max Convex Defect Depth`   | 最大凸包缺陷深度     | `max(valid defect depths)`                       |
| `Fractal Dimension`         | 邊界自相似複雜度     | box-counting：`slope(log(N), log(1 / box size))` |
| `Boundary Inflection Count` | 邊界曲率方向變化次數 | contour curvature sign changes                   |

---

## 七、Debris Quantification

此特徵描述 background mask 中的小碎片或死亡細胞碎屑，作為培養狀態、凋亡或 necrosis 的影像 proxy。備份 pipeline 會讀取 `ki67dtc/debris_feature_config.json`，以固定的 ImageJ threshold 規則與 `Analyze Particles` 面積範圍提取參數。

| 特徵參數                  | 定義                                        | 公式 / 方法                                  |
| ------------------------- | ------------------------------------------- | -------------------------------------------- |
| `Debris Count`            | background 中小碎片數量                     | background mask 內 `Analyze Particles` count |
| `Debris Area Fraction`    | debris 面積佔 background 比例               | `Σ Area_debris / Area_background`            |
| `Debris Mean Area`        | debris 平均面積                             | `mean(Area_debris)`                          |
| `Debris Density`          | background 中 debris 數量密度               | `Debris Count / Area_background`             |
| `Nearest Debris Distance` | cell centroid 到最近 debris centroid 的距離 | `min distance(C_cell, C_debris)`             |

---

## 八、Mitosis-Specific Features

此特徵不是單一量測，而是由多個特徵參數組合成與 mitosis 狀態相關的分數。Ki67 在 late G1 到 mitosis 期間表現，因此 mitosis proxy 可作為 Ki67 判斷的輔助訊號。

| 特徵參數                      | 定義                     | 公式 / 方法                                          |
| ----------------------------- | ------------------------ | ---------------------------------------------------- |
| `Mitotic Score`               | mitosis-like 狀態分數    | roundness、small area、低 entropy 等參數的組合       |
| `Daughter Pair Flag`          | 疑似剛分裂 daughter pair | `NND < 0.5 × mean diameter` 且面積差 `< 25%`         |
| `Protrusion Retraction Score` | 突起收回與圓化程度       | high circularity + small area + low protrusion count |

目前備份版的 composite score 是初版規則，後續應以 labeled data 重新校正權重與閾值。
