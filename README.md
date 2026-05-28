# Ki67 Detection

Ki67 Detection是一套自動化細胞影像分析系統，用於細胞分割、輪廓合併、形態與螢光特徵量測，以及無染色 Ki67 陽性比例預測，降低人工圈選與統計多通道影像的重複成本。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Cellpose](https://img.shields.io/badge/Cellpose-3.1.1.1-purple)
![授權狀態](https://img.shields.io/badge/%E6%8E%88%E6%AC%8A-%E5%B0%9A%E6%9C%AA%E6%8C%87%E5%AE%9A-lightgrey)

## 目錄

- [專案簡介](#專案簡介)
- [功能特色](#功能特色)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [使用方式](#使用方式)
- [常見問題](#常見問題)
- [貢獻指南](#貢獻指南)
- [授權條款](#授權條款)

## 專案簡介

本專案整合 Cellpose、PyImageJ、OpenCV、scikit-learn、LightGBM、XGBoost 與 PyQt6，將 Ki67 細胞影像分析拆成可重複執行的流程：

1. 使用 PC 影像分割細胞質，並可使用 DAPI 或 PC 影像分割細胞核。
2. 將 Cellpose `*_seg.npy` 轉成輪廓文字檔，合併細胞核與細胞質 ROI。
3. 量測細胞形態、細胞核/細胞質幾何參數與螢光強度。
4. 產生 Ki67 二值化遮罩與細胞層級 `ki67_positive` 標記。
5. 以已訓練模型輸出細胞、影像、資料夾層級的 Ki67 預測結果與報表。

主要入口包含命令列管線 `main.py`、圖形介面 `app.py`、批次處理腳本 `scripts/run_all_data_input.py`，以及模型訓練與推論腳本 `analysis/ki67_pred_training.py`、`analysis/ki67_pred.py`。

## 功能特色

- **細胞影像分析**：一步完成影像分割、遮罩轉輪廓、細胞核/細胞質合併、幾何量測與螢光量測。

- **無染色 Ki67 判斷**：支援 `pyimagej` 與 `opencv` 後端產生 Ki67 二值化遮罩，並以細胞核 ROI 重疊比例標記陽性細胞。

- **圖形介面操作**：提供 GUI 視窗介面，可執行管線、瀏覽影像與檢視 cleaned CSV。

## 環境需求

- Python 3.10+
- Mamba 或 Conda
- OpenJDK 11，供 PyImageJ/Fiji 後端使用
- NVIDIA GPU/CUDA 建議使用；目前 Cellpose 影像分割以 `gpu=True` 初始化模型
- 專案 Cellpose 模型，例如 `model/model_BDL6_label_new`
- 影像資料放在 `data/input/<資料集>/`

主要 Python 套件列在 [requirements.txt](requirements.txt)。圖形介面入口使用 PyQt6；若環境尚未安裝，請另外安裝 `PyQt6`。

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

## 使用方式

### 輸入資料夾格式

`PC/` 為必要資料夾；`DAPI/`、`IDO/`、`DF/`、`LT/`、`KI67/` 依分析需求提供。

```text
data/
  input/
    example_data/
      PC/
      DAPI/
      IDO/
      KI67/
  output/
```

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

訓練會從 `data/output/results/**/**_cleaned.csv` 載入資料，輸出模型檔與報告到 `data/output/train/ki67_pred/`。

```bash
python analysis/ki67_pred_training.py
```

正式推論會讀取 `data/output/train/ki67_pred/model/` 中的模型，輸出細胞、影像、資料夾層級結果到 `data/output/predict/`。

```bash
python analysis/ki67_pred.py
```

Windows 批次檔也可使用：

```bat
train_ki67_pred.bat
predict_ki67.bat
```

注意：這兩個 `.bat` 檔目前寫死 `D:\anaconda3\envs\ki67dtc\python.exe`，若本機環境不同，請先修改 `PYTHON_EXE`。

### 視覺化輸出

```bash
# 產生細胞輪廓圖
python scripts/generate_cell_outline_images.py

# 產生 Ki67 預測輪廓圖
python scripts/generate_ki67_prediction_contours.py

# 產生比例長條圖與混淆矩陣
python scripts/generate_ki67_prediction_plots.py
```

### 輸出資料夾

```text
data/output/
  segment/<資料集>/               # Cellpose 分割遮罩
  outline/<資料集>/               # 細胞質、細胞核、合併輪廓 txt
  binary/<資料集>/                # Ki67 二值化遮罩與陽性標記
  results/<資料集>/               # 每張影像 final CSV 與 <資料集>_cleaned.csv
  train/ki67_pred/model/          # 訓練後模型輸出檔
  train/ki67_pred/report/         # 訓練報告
  predict/                        # Ki67 正式推論 CSV/XLSX/圖表
  cell_outlines/<資料集>/          # 細胞輪廓 PNG
  ki67_contours/<資料集>/          # Ki67 預測輪廓 PNG
```

## 常見問題

**Q: 程式找不到資料集資料夾怎麼辦？**

A: `main.py --data_folder example_data` 會先找 `data/input/example_data`，再找目前目錄下的 `example_data`。請確認資料夾存在，且至少包含 `PC/` 影像資料夾。

**Q: PyImageJ 或 Java 初始化失敗怎麼辦？**

A: 先確認環境中有 `pyimagej`、`scyjava`、`jpype1` 與 OpenJDK 11。若有既有 Fiji 安裝，可設定 `FIJI_APP_PATH`；若只想避開 ImageJ 後端，可在 Ki67 分析時使用 `--ki67_backend opencv`。

**Q: 圖形介面啟動時找不到 PyQt6？**

A: 目前 `requirements.txt` 未列入 PyQt6。請在同一個 Python 環境執行 `pip install PyQt6`。

**Q: `.bat` 檔顯示 Python not found？**

A: `train_ki67_pred.bat` 與 `predict_ki67.bat` 內的 `PYTHON_EXE` 是本機絕對路徑，請改成你的 conda/mamba 環境 Python 路徑。

**Q: 複製專案後缺少 `data/` 或 `model/` 怎麼辦？**

A: `.gitignore` 目前會忽略 `data/` 與 `model/`。請從團隊共享位置或實驗輸出檔補齊資料與 Cellpose 模型後再執行分析。

**Q: 沒有 GPU 可以跑嗎？**

A: 目前 `ki67dtc/img_prep.py` 以 `CellposeModel(gpu=True, ...)` 初始化。若環境沒有可用 GPU，請先調整 Cellpose 初始化設定，或改在具備 CUDA 的環境執行影像分割。

## 貢獻指南

--

## 授權條款

--
