import subprocess

# main.py
# python main.py --data_folder path/to/data --fluor_analy --ki67 --clean_temp

# test.py
# python train.py --csv-root data/output/results --image-root data/output/cyto_crops
# 若置換測試過重，加入 --skip-permutation

# predict.py
# python predict.py --model-dir outputs_models/<timestamp> --model-key xgb_concat
#  --csv path/to/new_cleaned.csv --image-root data/output/cyto_crops --output predictions.csv
