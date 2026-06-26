import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
from pathlib import Path
import shutil
from ki67dtc.img_prep import segment_all, mask2txt_all, combined
from ki67dtc.cell_anal import run_all
from ki67dtc.cell_anal_plot import plot_area_analysis, plot_global_area_analysis
from ki67dtc.utils.io import list_files, output_dir

def analyze_cell(data_folder, data_folder_out, thres_logarea, CYTO_MODEL_PATH, NUC_MODEL_PATH, progress_callback=None,
                 status_callback=None, width_um_per_px=1.5896, height_um_per_px=1.5876):
    #****缺細胞數量預測功能****
    
    parser = argparse.ArgumentParser(description="細胞影像分析 Pipeline")
    '''
    parser.add_argument("--data_folder", type=str, default ="./data2/input", required=False, help="輸入資料夾路徑")
    parser.add_argument("--data_folder_out", type=str, default ="./data2/output", required=False, help="輸出資料夾路徑")
    parser.add_argument("--thres_logarea", type=float, default = 6, help="質面積分佈圖閥值")
    '''
    parser.add_argument("--fluor_analy", type=bool, default=False, help="是否進行螢光分析 (預設: True)")
    parser.add_argument("--ki67", type=bool, default=False, help="是否進行 Ki67 判斷 (預設: True)")
    parser.add_argument("--clean_temp", type=bool, default=False, help="是否清理暫存資料 (預設: True)")
    
    parser.add_argument("--plot", type=bool, default=True, help="是否輸出分析圖表 (預設: True)")
    args = parser.parse_args()
       
    print("=" * 50)
    print(f"[INFO] 測試資料夾: {data_folder}")
    print(f"[INFO] 螢光分析: {args.fluor_analy}")
    print(f"[INFO] Ki67 判斷: {args.ki67}")
    print(f"[INFO] 清理暫存: {args.clean_temp}")
    print("=" * 50)

    # Step 1: segmentation
    print("\n[STEP 1] 執行 segmentation (cyto & nuc)")
    # Count images for progress bar
    img_files = [
        f for f in list_files(data_folder, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
        if "ki67" not in f.stem.lower() and "df" not in f.stem.lower()
    ]
    total_images = len(img_files)
    processed = 0

    def progress(idx):
        if progress_callback:
            progress_callback(idx)

    # segment_all should call progress_callback(idx) after each image
    segment_all(data_folder, data_folder_out, CYTO_MODEL_PATH, NUC_MODEL_PATH,
                progress_callback=progress, status_callback=status_callback)

    if status_callback:
        status_callback("Converting masks to outlines...")
    # Step 2: mask -> outlines
    print("\n[STEP 2] 將 segmentation npy 轉成 outlines txt")
    mask2txt_all(data_folder, data_folder_out,
                 progress_callback=progress_callback, base_offset=total_images * 2)
    if status_callback:
        status_callback("Merging outlines...")
    # Step 3: 合併 nucleus & cytoplasm outlines
    print("\n[STEP 3] 合併 nucleus & cytoplasm outlines")
    combined(data_folder, data_folder_out,
             progress_callback=progress_callback, offset=total_images * 4)

    # Step 4: 細胞參數 & 螢光強度分析
    print("\n[STEP 4] 執行參數分析 + 螢光強度分析")
    # clear previous data
    folder_path = output_dir(data_folder_out, "results")
    figure_path = output_dir(data_folder_out, "figure")

    for clean_dir in [folder_path, figure_path]:
        if os.path.exists(clean_dir) and os.path.isdir(clean_dir):
            shutil.rmtree(clean_dir)
            os.makedirs(clean_dir)  # Recreate empty directory

    if status_callback:
        status_callback("Analyzing cell parameters...")
    run_all(
        data_folder,
        data_folder_out,
        fluor_analy=args.fluor_analy,
        ki67=args.ki67,
        clean_temp=args.clean_temp,
        plot=args.plot,
        thres_logarea=thres_logarea,
        progress_callback=progress_callback,
        offset=total_images * 5,
    )

    analy_dir = output_dir(data_folder_out, "results")
    figure_dir = output_dir(data_folder_out, "figure")
        
    # function 3 :全部 細胞質 vs 細胞核面積散布圖 & 細胞質面積對數分布圖（含閾值紅線）
    if status_callback:
        status_callback("Generating plots...")
    ALL_para_file = [f for f in list_files(analy_dir, ".csv") if "ALL_" in f.stem]
    #ALL_para_file = Path(ALL_para_file)
    plot_global_area_analysis(ALL_para_file[0], figure_dir, thres_logarea,
                               width_um_per_px=width_um_per_px, height_um_per_px=height_um_per_px)

    print("\n[INFO] Pipeline 執行完成！")


if __name__ == "__main__":
    #輸入
    #------------------------
    #模型路徑固定
    CYTO_MODEL_PATH = "model/model_BDL6_label_new"
    NUC_MODEL_PATH = "model/model_BDL3_label_dapi"  
    data_folder = "./data2/input"
    data_folder_out = "./data2/output"
    thres_logarea = 7.0
    #------------------------ 
  
      
    data_folder = Path(data_folder)
    data_folder_out = Path(data_folder_out)
    if not data_folder.exists() or not data_folder.is_dir():
        print(f"[ERROR] 找不到資料夾: {data_folder}")
        exit(1)
        
    # function 1 :分割細胞 & 分析細胞參數    
    analyze_cell(data_folder, data_folder_out, thres_logarea, CYTO_MODEL_PATH, NUC_MODEL_PATH)
    
    analy_dir = output_dir(data_folder_out, "results")
    figure_dir = output_dir(data_folder_out, "figure")
    
    
    # function 2 :細胞質 vs 細胞核面積散布圖 & 細胞質面積對數分布圖（含閾值紅線）
    print("\n[STEP 5] 產出散布圖和分布圖")
    para_files = [f for f in list_files(analy_dir, ".csv") if "_final" in f.stem]
    for para_file in para_files:
        name = para_file.stem.replace("_final", "")
        plot_area_analysis(para_file, figure_dir, name, thres_logarea)
        
    # function 3 :全部 細胞質 vs 細胞核面積散布圖 & 細胞質面積對數分布圖（含閾值紅線）
    ALL_para_file = [f for f in list_files(analy_dir, ".csv") if "ALL_" in f.stem]
    #ALL_para_file = Path(ALL_para_file)    
    plot_global_area_analysis(ALL_para_file[0], figure_dir, thres_logarea)

    print("\n[INFO] 全部執行完成！請檢查輸出結果。")