# Ki67 特徵與特徵參數定義

本文件整理 Ki67 預測流程中「特徵」與「特徵參數」的命名、意義與公式。

---

## 命名原則

**特徵**指的是具有生物或影像意義的概念群，例如 `Intensity Distribution Features`、`Nuclear Sub-Region Features`、`Edge / Halo Features`。  
**特徵參數**指的是實際被量測、寫入 CSV、後續可進模型的數值欄位，例如 `Mean_nuc`、`CV_cyto`、`Nuc Cyto Mean Ratio`。

---

## 一、Intensity Distribution Features

此特徵描述 ROI 內像素強度分布形狀。Phase contrast 中像素強度可視為細胞厚度、折射率、乾質量與核/質紋理狀態的間接 proxy。
每個參數可分別量測於 nucleus 與 cytoplasm。

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
| `Nuc Cell IntDen Ratio`       | 核區 IOD / whole-cell IOD         | `IntDen_nuc / Whole Cell IntDen` |
| `Nuc Cyto Entropy Difference` | 核區 entropy 與細胞質 entropy 差  | `Entropy_nuc - Entropy_cyto`     |
| `Nuc Cyto CV Difference`      | 核區 CV 與細胞質 CV 差            | `CV_nuc - CV_cyto`               |

同一組 16 個強度參數也會以 `Whole Cell <參數>` 寫入 cell-level 欄位，補齊 PDF 要求的 whole-cell、nucleus、cytoplasm 三個區域。

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
| `Nucleolus Count`         | 核內亮點候選數量                 | 核內局部背景扣除後的 local maxima component 數  |
| `Mean Nucleolus Area`     | 候選核仁平均面積                 | `mean(candidate component areas)`               |
| `Max Nucleolus Area`      | 最大候選核仁面積                 | `max(candidate component areas)`                |

`Nucleus Centroid Offset` 越大代表 nucleus 越偏離細胞中心，可能反映細胞極性、遷移或貼附狀態差異。
核仁偵測屬於 PDF 標示的 partial feature，仍需依細胞株、倍率與焦距人工抽查門檻。

---

## 三、Edge / Halo Features

此特徵描述 phase contrast 中細胞邊緣的 halo、邊界明暗與銳利程度。Halo 可反映細胞厚度、貼附狀態、圓化程度與焦距差異。

| 特徵參數                | 定義                                  | 公式 / 來源                                                           |
| ----------------------- | ------------------------------------- | --------------------------------------------------------------------- |
| `Halo Outer Mean`       | cell 外側 halo ring 平均強度          | ImageJ 量測 outer ring `Mean`                                         |
| `Halo Outer StdDev`     | cell 外側 halo ring 強度變異          | ImageJ 量測 outer ring `StdDev`                                       |
| `Halo Outer CV`         | halo ring 相對變異                    | `Halo Outer StdDev / Halo Outer Mean`                                 |
| `Halo Inner Mean`       | cell 內側邊緣 ring 平均強度           | ImageJ 量測 inner ring `Mean`                                         |
| `Halo Inner StdDev`     | cell 內側邊緣 ring 強度變異           | ImageJ 量測 inner ring `StdDev`                                       |
| `Halo Inner Outer Diff` | 內外 halo 強度差                      | `abs(Halo Outer Mean - Halo Inner Mean)`                              |
| `Halo Angular Variance` | halo 的角向不對稱程度                 | 36 個角向 sector mean 的 variance                                     |
| `Halo Radial Gradient`  | cell edge 向外 15 px 的平均強度斜率   | 16 個方向 radial profile slope 的平均                                |
| `Halo Width`            | halo 回到背景強度附近所需的 ring 寬度 | 最小 `w`，使 ring mean `<= background mean + 1.5 × background StdDev` |
| `Edge Sharpness`        | 細胞邊界銳利程度                      | ImageJ `Find Edges` 後，在 boundary ring 量測 `Mean`                  |

---

## 四、Texture Features

此特徵描述 ROI 內像素強度的空間排列，而不只是分布形狀。

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

Python backend 使用 64 gray levels、distance `[1, 3, 5]`、四個方向，最後對方向與距離取平均。nucleus、cytoplasm 與 whole-cell 皆會量測，mask 先 erosion 2 px。

### LBP

LBP 是 local binary pattern，描述每個像素與周圍鄰居的局部紋理關係。

| 特徵參數            | 定義                                  |
| ------------------- | ------------------------------------- |
| `LBP Hist Bin_*`    | ROI 內 LBP code histogram 各 bin 比例 |
| `LBP Uniform Ratio` | uniform pattern 佔比                  |
| `LBP Mean`          | LBP code 平均值                       |
| `LBP StdDev`        | LBP code 標準差                       |
| `LBP Entropy`       | LBP code histogram entropy            |
| `LBP Uniform R1/R2/R3 Hist Bin 00-09` | radius 1、2、3 的 rotation-invariant uniform LBP histogram |

目前 `LBP Hist Bin 00` 到 `LBP Hist Bin 15` 會把 0-255 的 LBP code 以每 16 個 code 合併成一個 bin，以降低欄位數與小 ROI 的 histogram 稀疏度。
PDF 指定的 multi-radius uniform LBP 另以每個 radius 10 bins 輸出；舊 16-bin 欄位保留供既有資料比對。

### 其他紋理參數

| 特徵參數                  | 定義                         | 實作 |
| ------------------------- | ---------------------------- | ---- |
| `Tamura Coarseness`       | 紋理粗細尺度                 | 6 個 box-filter scales 的最佳反應尺度平均 |
| `Zernike Moment 00-24`    | 圓形區域內形狀與紋理的矩特徵 | `mahotas.features.zernike_moments(..., degree=8)` |

上述 texture 參數都會輸出 nucleus、cytoplasm 與 `Whole Cell` 版本。

---

## 五、Population / Neighbourhood Context Features

此特徵描述單張影像 field of view 內的整體細胞狀態與局部鄰近關係。

| 特徵參數                         | 定義                            | 公式                                               |
| -------------------------------- | ------------------------------- | -------------------------------------------------- |
| `Image Confluency`               | 影像內細胞覆蓋比例              | `Σ Area_cell / Image Area`                         |
| `Population Area CV`             | 同張影像 cell area 的變異係數   | `std(Area_cell) / mean(Area_cell)`                 |
| `Population Circularity CV`      | 同張影像 circularity 的變異係數 | `std(Circularity_cell) / mean(Circularity_cell)`   |
| `Nearest Neighbor Distance`      | 最近鄰細胞距離                  | `min distance(C_i, C_j)`                           |
| `Nearest Neighbor Distance Norm` | 標準化最近鄰距離                | `Nearest Neighbor Distance / median cell diameter` |
| `Local Neighbor Count`           | 固定半徑內鄰近細胞數            | `count(distance <= radius)`                        |
| `Local Density`                  | 局部細胞密度                    | `Local Neighbor Count / (πr^2)`                    |
| `Neighbour Area Ratio`           | 相對於最近三個鄰居的面積比      | `Area_cell / mean(Area_3_nearest)`                 |
| `Cluster Size`                   | 該細胞所屬 cluster 的細胞數     | 根據 centroid distance graph 分群                  |
| `Cluster Size Norm`              | 標準化 cluster size             | `Cluster Size / total cell count`                  |
| `Largest Cluster Ratio`          | 最大 cluster 佔比               | `largest cluster size / total cell count`          |
| `Mitotic Index`                  | FOV 疑似 M-phase 細胞比例        | `mean(circularity > 0.85 and area < 0.6 × median)` |

---

## 六、Protrusion / Shape Complexity Features

此特徵描述 whole-cell boundary 的突起、凹陷與輪廓複雜度。
細胞進入 mitosis 前常會收回突起並圓化；貼附良好或遷移中的細胞則可能有更複雜的邊界。

| 特徵參數                    | 定義                 | 公式 / 方法                                      |
| --------------------------- | -------------------- | ------------------------------------------------ |
| `Protrusion Count`          | 凸包缺陷數量         | `convexity defects depth > threshold` 的數量     |
| `Mean Convex Defect Depth`  | 平均凸包缺陷深度     | `mean(valid defect depths)`                      |
| `Mean Protrusion Length Norm` | 尺度標準化平均突起長度 | `Mean Convex Defect Depth / equivalent diameter` |
| `Max Convex Defect Depth`   | 最大凸包缺陷深度     | `max(valid defect depths)`                       |
| `Fractal Dimension`         | 邊界自相似複雜度     | box-counting：`slope(log(N), log(1 / box size))` |
| `Boundary Inflection Count` | 邊界曲率方向變化次數 | contour curvature sign changes                   |

---

## 七、Debris Quantification

此特徵描述 background mask 中的小碎片或死亡細胞碎屑，作為培養狀態、凋亡或 necrosis 的影像 proxy。

| 特徵參數                  | 定義                                        | 公式 / 方法                                  |
| ------------------------- | ------------------------------------------- | -------------------------------------------- |
| `Debris Count`            | background 中小碎片數量                     | background mask 內 `Analyze Particles` count |
| `Debris Area Fraction`    | debris 面積佔 background 比例               | `Σ Area_debris / Area_background`            |
| `Debris Mean Area`        | debris 平均面積                             | `mean(Area_debris)`                          |
| `Debris Density`          | background 中 debris 數量密度               | `Debris Count / Area_background`             |
| `Nearest Debris Distance` | cell centroid 到最近 debris centroid 的距離 | `min distance(C_cell, C_debris)`             |

---

## 八、Mitosis-Specific Features

此特徵不是單一量測，而是由多個特徵參數組合成與 mitosis 狀態相關的分數。
Ki67 在 late G1 到 mitosis 期間表現，因此 mitosis proxy 可作為 Ki67 判斷的輔助訊號。

| 特徵參數                      | 定義                     | 公式 / 方法                                          |
| ----------------------------- | ------------------------ | ---------------------------------------------------- |
| `Mitotic Score`               | mitosis-like 狀態分數    | circularity × small-area × low whole-cell GLCM entropy × halo symmetry |
| `Daughter Pair Flag`          | 疑似剛分裂 daughter pair | `NND < 0.5 × mean diameter` 且面積差 `< 25%`         |
| `Protrusion Retraction Score` | 突起收回與圓化程度       | high circularity + small area + low protrusion count |

---

## 九、未導入項目

目前只保留需要多時間點與 tracking 的 Dynamic / Temporal Features 未導入，包括 migration speed、shape change rate 與 population Delta-CV。
