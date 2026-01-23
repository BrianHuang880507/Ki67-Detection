## 安裝與環境

建議使用 Python 3.10+，並安裝必要套件：

```bash
pip install -r requirements.txt
```

## GUI（PyQt6）

本專案包含一個用於操作 Ki67 pipeline 的圖形介面（可瀏覽影像、顯示 outlines overlay、檢視 `<dataset>_cleaned.csv` 並點選 Cell_ID 高亮對應 cell）。

### 如何開啟 GUI

- 從專案根目錄執行：

  ```bash
  python app.py
  ```

### 待測圖片要放哪裡

- 建議放在：`data/input/<dataset_name>/`
- GUI 內請直接選擇資料夾：`data/input/<dataset_name>/`

> `<dataset_name>` 通常會用資料夾名稱當作 dataset id，後續輸出也會以它為子目錄。

### 輸出檔案會在哪裡

- 主要結果（CSV）：
  - `data/output/results/<dataset_name>/<dataset_name>_cleaned.csv`
- outlines（GUI overlay 會讀取）：
  - 專案會產生 `*_merged_cp_outlines.txt`（每兩行一組：nuc/cyto；`-1,-1` 表示缺失）。
  - 檔案通常會與影像同資料夾或位於對應的輸出資料夾（依 pipeline 實作/設定而定）。
- segmentation masks（`.npy`）：
  - 常見會輸出在 `masks/` 或 `data/output/` 相關子資料夾（依 pipeline 設定而定）。

- 模型選擇：
  - 預設會掃描 `./model` 目錄。
  - 本專案的模型檔可能是「無副檔名」檔案（例如 `model_BDL3_label_dapi`），屬正常。

## 使用流程總覽

1. **主流程產生量測結果**

   ```bash
   python main.py --data_folder ./data/input/sample1
   ```

   這一步會依設定輸出每張影像對應的 `_final.csv`，並在彙整後於 `data/output/results/<dataset>/` 產生 `<dataset>_cleaned.csv`。

2. **裁切細胞核/質影像**

   ```bash
   python crop_nuclei_from_npy.py --config ./configs/crop_config.yaml
   ```

   - 讀取前一步的 NPY/CSV 參數
   - 產生對應的 cyto/nuc 裁切圖片

3. **模型訓練與預測**
   - 訓練：
     ```bash
     python train.py --csv-root data/output/results --image-root data/output/cyto_crops
     ```
   - 預測：
     ```bash
     python predict.py --model-dir outputs_models/<timestamp> --model-key xgb_concat \
                       --csv data/output/results/<batch>/<file>_cleaned.csv \
                       --image-root data/output/cyto_crops --output predictions.csv
     ```

## main.py 常用參數

| 參數            | 類別 | 預設值 | 說明               |
| --------------- | ---- | ------ | ------------------ |
| `--data_folder` | str  | 必填   | 輸入資料夾路徑     |
| `--fluor_analy` | bool | True   | 是否執行螢光分析   |
| `--ki67`        | bool | True   | 是否進行 Ki67 判斷 |
| `--clean_temp`  | bool | True   | 是否清除暫存資料   |
