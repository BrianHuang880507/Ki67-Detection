# Changelog
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
