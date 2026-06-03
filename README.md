# Ki67 Detection

Ki67 Detection是一套自動化細胞影像分析系統，用於細胞分割、形態/螢光特徵分析，以及無染色 Ki67 預測，降低人工圈選失誤與細胞染色成本。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Cellpose](https://img.shields.io/badge/Cellpose-3.1.1.1-purple)
![授權狀態](https://img.shields.io/badge/%E6%8E%88%E6%AC%8A-%E5%B0%9A%E6%9C%AA%E6%8C%87%E5%AE%9A-lightgrey)

---

## 目錄

- [專案簡介](#專案簡介)
- [功能特色](#功能特色)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [專案架構](#專案架構)
- [使用方式](#使用方式)
- [常見問題](#常見問題)
- [貢獻指南](#貢獻指南)
- [授權條款](#授權條款)

---

## 專案簡介

本專案將細胞影像分析拆成可重複執行的流程：

1. 使用 Phase Contrast 影像分割細胞質/細胞核。
2. 將 Cellpose `*_seg.npy` 轉成輪廓文字檔，合併細胞核與細胞質 ROI。
3. 量測細胞形態、細胞核/細胞質幾何參數與螢光強度。
4. 產生 Ki67 二值化遮罩與細胞層級 `ki67_positive` 標記。
5. 以已訓練模型輸出細胞、影像、資料夾層級的 Ki67 預測結果與報表。(開發中)

**本專案目前只適用於MSC**

---

## 環境需求

- Python 3.10+
- Mamba 或 Conda
- OpenJDK 11，供 PyImageJ/Fiji 後端使用
- NVIDIA GPU/CUDA 建議使用；目前 Cellpose 影像分割以 `gpu=True` 初始化模型
- 專案 Cellpose 模型，例如 `model/model_BDL6_label_new`
- 影像資料放在 `data/input/<資料集>/`

主要 Python 套件列在 [requirements.txt](requirements.txt)。圖形介面入口使用 PyQt6；若環境尚未安裝，請另外安裝 `PyQt6`。

---

## 安裝步驟

```bash
# 複製專案
git clone https://github.com/BrianHuang880507/Ki67-Detection.git

# 進入專案目錄
cd Ki67-Detection

# 建立環境，先安裝 PyImageJ 與 Java
mamba create -n ki67 python=3.10 pyimagej openjdk=11 pip -c conda-forge -y
mamba activate ki67

# 若無 mamba
conda install -n base -c conda-forge mamba

# 安裝其餘相依套件
pip install -r requirements.txt

# 如需使用圖形介面
pip install PyQt6
```

若使用既有 Fiji 安裝，可設定 `FIJI_APP_PATH` 指向 Fiji 應用程式路徑；若未設定，程式會嘗試透過 PyImageJ 初始化 Fiji。

---

## 專案架構

資料集請放在 `data/input/<資料集>/`，其中 `PC/` 為必要資料夾；`DAPI/`、`IDO/`、`DF/`、`LT/`、`KI67/` 依分析需求提供。分析後的中間檔、CSV 與視覺化結果會輸出到 `data/output/` 對應子資料夾。

Ki67 特徵與特徵參數定義請見 [ki67dtc/feature_param.md](ki67dtc/feature_param.md)。

```text
Ki67-Detection/
├── main.py                          # 命令列分析管線入口
├── app.py                           # 圖形介面入口
├── requirements.txt                 # Python 相依套件
├── model/                           # Cellpose 分割模型
├── data/
│   ├── input/
│   │   └── <資料集>/
│   │       ├── PC/                  # 必要；Phase Contrast 影像
│   │       ├── DAPI/                # 選用；細胞核螢光影像
│   │       ├── IDO/                 # 選用；螢光通道
│   │       ├── DF/                  # 選用；螢光/影像通道
│   │       ├── LT/                  # 選用；螢光/影像通道
│   │       └── KI67/                # 選用；Ki67 染色影像
│   └── output/
│       ├── segment/<資料集>/        # Cellpose 分割遮罩
│       ├── outline/<資料集>/        # 細胞質、細胞核、合併輪廓 txt
│       ├── binary/<資料集>/         # Ki67 二值化遮罩與陽性標記
│       ├── results/<資料集>/        # 每張影像 final CSV 與 <資料集>_cleaned.csv
│       ├── train
│       │     └──ki67_pred/
│       │        └──model/           # 訓練後模型輸出檔
│       │        └──report/          # 訓練報告
│       ├── predict/                 # Ki67 正式推論 CSV/XLSX/圖表
│       ├── cell_outlines/<資料集>/   # 細胞輪廓 PNG
│       └── ki67_contours/<資料集>/   # Ki67 預測輪廓 PNG
├── ki67dtc/
│   ├── app_pipeline.py              # GUI 使用的分析流程封裝
│   ├── img_prep.py                  # 影像前處理與分割準備
│   ├── cell_anal.py                 # 細胞分析與特徵量測主程式
│   ├── cell_anal_backup.py          # 備份/實驗版分析流程
│   ├── feature_param.md             # Ki67 特徵與特徵參數定義
│   ├── debris_feature_config.json   # debris 特徵設定
│   ├── gui/
│   │   └── main_window.py           # PyQt6 圖形介面
│   └── utils/
│       └── io.py                    # 檔案與路徑工具
├── analysis/
│   ├── ki67_pred_training.py        # Ki67 預測模型訓練
│   ├── ki67_pred.py                 # Ki67 預測流程
│   ├── ki67_pred_utils.py           # 預測工具函式
│   └── ki67_pdf_feature_analysis.py  # PDF 特徵分析輔助工具
├── scripts/
│   ├── run_all_data_input.py                 # 批次處理 data/input 內資料集
│   ├── train_ki67_pred.bat                   # Windows 訓練批次檔
│   └── predict_ki67.bat                      # Windows 推論批次檔
├── docs/
│   ├── ki67_prediction_workflow.md  # Ki67 預測流程文件
│   └── ki67_prediction_workflow.png # Ki67 預測流程圖
└── README.md                        # 專案說明文件
```

---

## 使用方式

### 單一資料集分析

```bash
python main.py --data_folder example_data --nuc_source dapi --fluor_analy --ki67
```

也可以傳入相對或絕對路徑：

```bash
python main.py --data_folder data/input/example_data --nuc_source dapi --ki67_backend opencv
```

常用參數：

- `--data_folder`：待測資料集名稱或路徑；若只給名稱，會優先搜尋 `data/input/<資料集>`。
- `--nuc_source`：細胞核分割來源，可用 `dapi` 或 `pc`。
- `--fluor_analy`：啟用螢光強度分析。
- `--ki67`：啟用 Ki67 陽性分析。
- `--ki67_backend`：Ki67 二值化後端，可用 `pyimagej` 或 `opencv`。
- `--clean_temp`：清理中間暫存檔。

### 批次處理

```bash
# 跑完 data/input 底下所有第一層資料集
python scripts/run_all_data_input.py --fluor_analy --ki67 --clean_temp

# 只處理指定資料夾
python scripts/run_all_data_input.py --only P6-1 P6-2 --fluor_analy --ki67

# 先預覽將要執行的命令
python scripts/run_all_data_input.py --dry-run --fluor_analy --ki67
```

### 圖形介面

```bash
python app.py
```

圖形介面會呼叫 `ki67dtc.app_pipeline.run_pipeline()`，完成後讀取 `data/output/results/<資料集>/<資料集>_cleaned.csv` 並在介面中顯示結果。

### Ki67 模型訓練與推論

開發中...

### 視覺化輸出

```bash
# 產生細胞輪廓圖
python scripts/generate_cell_outline_images.py

# 產生 Ki67 預測輪廓圖
python scripts/generate_ki67_prediction_contours.py

# 產生比例長條圖與混淆矩陣
python scripts/generate_ki67_prediction_plots.py
```

---

## 常見問題

--

---

## 貢獻指南

--

---

## 授權條款

--
