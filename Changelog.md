# Changelog

## 2026.06.30

  * 對齊 SegmentationUI 主畫面風格，加入深色主題、選單列入口與右側四區資訊面板。
  * 將輸入資料夾與分析選項移到選單列，支援核來源、Ki67 backend、分析方法與清理暫存檔案等選項。
  * 調整主畫面版面配置：左側保留主要影像顯示區，右側分為終端輸出、影像清單、特徵參數與細胞面積分析圖。
  * 將開始、中止與重新開始改為置中的圖示按鈕，並補上執行中與停用狀態的視覺回饋。
  * 將影像 overlay 控制列移到 `Image File Name` 同列右側，並移除底部狀態列顯示。
  * 在 Cellpose 分割前加入固定模型輸入尺寸 resize，推論後將 mask 與 flow 還原到原圖尺寸。
  * 依細胞核來源選用對應模型：DAPI 使用 `cyto3`，PC 或 DAPI fallback 使用 `model/model_BDL3_label_dapi`。
  * 補上 UI layout contract 與分割 resize/model 選擇測試，降低後續介面與 pipeline 行為回歸風險。

## 2026.06.06

  * 新增 `feature_backend=pyimagej|python`，支援以 NumPy、SciPy、scikit-image 與 OpenCV 提取既有特徵欄位
  * Debris 特徵改由 phase-contrast 影像計算，避免使用 Ki67 影像造成標籤洩漏
  * 修正 circularity 公式為 `4πA/P²`
  * 新增兩種特徵後端的批次差異比較腳本
  * 導入 PDF 中除 Dynamic/time-lapse 外的剩餘特徵：multi-distance GLCM、multi-radius uniform LBP、Tamura、Zernike、whole-cell texture/intensity、核仁候選、Halo angular/radial、Neighbour Area Ratio 與 Mitotic Index
  * 新增 Python backend 的多尺度紋理、核仁、expanded schema、欄位合併與螢光提取測試

## 2026.06.02

  * 新增 Ki67 特徵參數擴充主流程，導入 Intensity distribution、Nuclear sub-region、Halo 基礎量、Population context、Protrusion/Shape complexity、Debris 與 Mitosis proxy 參數
  * 新增 Texture Features 參數，支援 PyImageJ GLCM/Haralick 與 ImageJ macro LBP 量測，作為 Ki67 判讀模型的紋理訊號補充
  * 新增 Eccentricity 幾何參數，輸出 `Eccentricity_nuc` 與 `Eccentricity_cyto`，補足核區與細胞質輪廓形狀描述
  * 新增 `cell_status` 分類與 final/cleaned CSV 分流規則，`*_final.csv` 保留所有細胞，`*_cleaned.csv` 依狀態保留 `full_cell`、`nuc_only`、`cyto_cut`
  * 調整 cell-level 特徵參數輸出欄位，`Nuc Cyto Mean Ratio`、Debris、Population、Mitosis proxy 等不再附加 `_nuc` 或 `_cyto` suffix
  * 修復 PyImageJ Ki67 二值化路徑解析，避免 ImageJ macro 因相對路徑或 Windows 路徑格式而無法產出可讀二值圖
  * 修復凸缺陷輪廓在 OpenCV `convexityDefects` 遇到自交或非單調 hull index 時中斷流程的問題，改以缺值保留並讓主流程繼續
  * 改善暫存檔清理規則，主流程 clean temp 會移除 `*_seg.npy`、`*_cyto_seg.npy` 與 `*_nuc_seg.npy`，降低輸出資料夾體積
  * 新增 `ki67dtc/feature_param.md`，以繁體中文整理特徵與特徵參數定義、公式、命名規則與目前導入狀態
  * 補齊 Python 函式 docstring 與重要流程註解，統一使用繁體中文說明提高後續維護可讀性
  * 標註尚未導入參數：Tamura Coarseness、Zernike Moments、Halo Radial Gradient、Halo Angular Variance 與 Neighbour Area Ratio
