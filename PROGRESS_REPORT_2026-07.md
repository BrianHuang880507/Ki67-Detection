# Ki67 Detection 近一個月進度報告

> 報告期間：2026-06-27～2026-07-27

## 本月重點

本月主要完成 GUI 對齊 SegUI、Cellpose 分割流程穩定化、特徵分析擴充，
以及 cleaned 結果的 XLSX 匯出。另完成第一版 morphology → IDO response
proxy 探索流程，供後續免疫相關研究評估。

## 已完成項目

### 1. GUI 與結果顯示

- 主畫面改為 SegmentationUI 風格，整合選單、右側資訊區與 overlay 控制。
- 新增核、細胞質及 Ki67 半透明填色，pipeline 完成後可自動儲存 overlay。
- GUI 採用 Paired Only 顯示，只呈現細胞核與細胞質有效配對。

### 2. Cellpose 與 ROI 過濾

- 分割前統一 resize 至模型輸入尺寸，完成後將 mask 與 flow 還原至原圖。
- DAPI 與 PC 核分割依來源選用對應模型，並補上 DAPI fallback／remap。
- 小面積過濾改用配對 ROI 建立參考中位數；已配對小 ROI 受到保護，
  只移除未配對雜訊。
- 10 張實驗影像驗證結果：CPU 為 333 個 cyto／303 個 nucleus，
  GPU 為 332 個 cyto／304 個 nucleus。

### 3. 特徵分析與輸出

- Python backend 的幾何量測改用 OpenCV，並降低 GLCM 灰階層數。
- 新增 Extent、主次軸、半徑統計、Solidity、Perimeter/Area Ratio
  與 CellProfiler 相容 Compactness。
- cleaned CSV 可輸出工程版與生醫版 XLSX；CLI、批次腳本與 GUI
  均可選擇版本。

### 4. Immunity 探索分析

- 建立 B4 p6 phase／DAPI／IDO 共 80 組影像的資料配對與 QC 流程。
- 完成 morphology-only、dose-only、ElasticNet 與組合模型的交叉驗證。
- 目前 morphology Ridge 在批次內優於 Dummy baseline，但
  leave-one-condition-out 表現下降，因此結果仍定位為單一批次探索，
  尚不能視為跨批次免疫抑制能力模型。

## 下一階段

- 以更多影像批次驗證配對感知面積過濾的穩定性。
- 整理 Cellpose CPU／GPU 與 DAPI／PC 的參數比較紀錄。
- 累積不同 donor、lot 或 passage，重新評估 morphology → IDO 模型泛化能力。
