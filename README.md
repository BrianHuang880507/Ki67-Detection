## 安裝與環境

建議使用 Python 3.10+，並安裝必要套件：

```bash
pip install -r requirements.txt
```

## 使用流程總覽

1. **主流程產生量測結果**

   ```bash
   python main.py --data_folder ./data/input/sample1
   ```

   這一步會依設定輸出每張影像對應的 `_final.csv`。

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
