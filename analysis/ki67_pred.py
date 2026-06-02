"""Ki67 正式預測流程。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ki67_pred_utils import (
    DEFAULT_BDL_RATIO_REFERENCE,
    DEFAULT_DATA_PATTERN,
    DEFAULT_HOLDOUT_SOURCE_FOLDERS,
    DEFAULT_LABEL_CANDIDATES,
    DEFAULT_PASSAGE_CANDIDATES,
    aggregate_to_image_features,
    apply_probability_calibration,
    build_image_design_matrices,
    compare_folder_ratio_to_reference,
    find_cleaned_csv_files,
    load_ki67_dataset,
    predict_stage1_probability,
    save_folder_ratio_comparison_plot,
    save_json,
    summarize_folder_ratio,
    transform_features,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "output" / "results"
MODEL_DIR = PROJECT_ROOT / "data" / "output" / "train" / "ki67_pred" / "model"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "predict"

DATA_PATTERN = DEFAULT_DATA_PATTERN
PREDICT_SOURCE_FOLDERS = list(DEFAULT_HOLDOUT_SOURCE_FOLDERS)
LABEL_CANDIDATES = list(DEFAULT_LABEL_CANDIDATES)
PASSAGE_CANDIDATES = list(DEFAULT_PASSAGE_CANDIDATES)
BDL_RATIO_REFERENCE = dict(DEFAULT_BDL_RATIO_REFERENCE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PREDICTION_THRESHOLD = 0.5
FOLDER_VS_BDL_COLUMNS = [
    "source_folder",
    "image_count",
    "cell_count",
    "pred_ratio",
    "input_image_count",
    "used_image_count",
    "bdl_true_ratio",
    "abs_error",
    "within_0p05",
    "within_0p10",
]
FOLDER_VS_ANSWER_COLUMNS = [
    "source_folder",
    "image_count",
    "cell_count",
    "pred_ratio",
    "answer_ratio",
    "abs_error",
    "within_0p05",
    "within_0p10",
    "input_image_count",
    "used_image_count",
    "missing_image_count",
    "extra_image_count",
    "missing_images",
    "extra_images",
]
CELL_PREDICTION_COLUMNS = [
    "source_folder",
    "Image",
    "Cell_ID",
    "_source_row_id",
    "raw_cell_prob",
    "cell_prob",
    "predicted_ki67_positive",
    "ki67_ground_truth",
    "prediction_correct",
]
IMAGE_PREDICTION_DROP_COLUMNS = ["image_key", "passage", "true_ratio"]
WORKBOOK_SHEET_FOLDER = "predictions vs answer"
WORKBOOK_SHEET_IMAGE = "image_predictions"
WORKBOOK_SHEET_CELL = "cell_predictions"


def ensure_clean_predict_dir() -> None:
    """清空並重建正式預測輸出目錄。"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_input_pc_image_stems(source_folder: str) -> list[str]:
    """列出 data/input/<source_folder>/PC 內的原始 PC 影像 stem。"""
    pc_dir = PROJECT_ROOT / "data" / "input" / source_folder / "PC"
    if not pc_dir.exists() or not pc_dir.is_dir():
        return []
    return sorted(
        p.stem
        for p in pc_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def add_image_count_diagnostics(
    folder_df: pd.DataFrame,
    image_df: pd.DataFrame,
) -> pd.DataFrame:
    """補上原始影像數、實際用於預測的影像數，以及缺漏影像清單。"""
    if len(folder_df) == 0:
        return folder_df

    rows = []
    for _, row in folder_df.iterrows():
        source_folder = str(row["source_folder"])
        expected_images = set(list_input_pc_image_stems(source_folder))
        used_images = set(
            image_df.loc[
                image_df["source_folder"].astype(str) == source_folder, "image_name"
            ].astype(str)
        )
        missing_images = sorted(expected_images - used_images)
        extra_images = sorted(used_images - expected_images)

        row_dict = row.to_dict()
        row_dict["input_image_count"] = len(expected_images)
        row_dict["used_image_count"] = int(row_dict.get("image_count", 0))
        row_dict["missing_image_count"] = len(missing_images)
        row_dict["extra_image_count"] = len(extra_images)
        row_dict["missing_images"] = ";".join(missing_images)
        row_dict["extra_images"] = ";".join(extra_images)
        rows.append(row_dict)

    return pd.DataFrame(rows)


def answer_path_candidates(cleaned_csv_path: Path) -> list[Path]:
    """產生 cleaned CSV 對應的人工答案檔候選路徑。

    Args:
        cleaned_csv_path (Path): 單一資料集的 `*_cleaned.csv` 路徑。

    Returns:
        list[Path]: 依優先順序排列且不重複的答案檔候選路徑。
    """
    candidates = []
    if cleaned_csv_path.name.endswith("_cleaned.csv"):
        candidates.append(
            cleaned_csv_path.with_name(
                cleaned_csv_path.name.replace("_cleaned.csv", "_answer.csv")
            )
        )
    candidates.append(
        cleaned_csv_path.parent / f"{cleaned_csv_path.parent.name}_answer.csv"
    )

    unique_candidates = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)
    return unique_candidates


def pick_ground_truth_column(columns: pd.Index) -> str | None:
    """從答案表欄位中挑選 Ki67 標籤欄位。

    Args:
        columns (pd.Index): 答案表的欄位索引。

    Returns:
        str | None: 找到的標籤欄位名稱；找不到時回傳 `None`。
    """
    column_lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in LABEL_CANDIDATES:
        found = column_lookup.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


def load_answer_labels(csv_files: list[Path]) -> pd.DataFrame:
    """讀取多個 cleaned CSV 對應的人工答案標籤。

    Args:
        csv_files (list[Path]): 預測輸入的 cleaned CSV 檔案清單。

    Returns:
        pd.DataFrame: 包含 `source_file`、`_source_row_id` 與答案標籤的表格。
    """
    frames = []
    for csv_path in csv_files:
        answer_path = next(
            (path for path in answer_path_candidates(csv_path) if path.exists()),
            None,
        )
        if answer_path is None:
            continue

        answer_df = pd.read_csv(answer_path)
        answer_df = answer_df.loc[:, ~answer_df.columns.duplicated()].copy()
        ground_truth_col = pick_ground_truth_column(answer_df.columns)
        if ground_truth_col is None:
            continue

        ground_truth = pd.to_numeric(answer_df[ground_truth_col], errors="coerce")
        labels = pd.DataFrame(
            {
                "source_file": str(csv_path),
                "_source_row_id": np.arange(len(answer_df)),
                "ki67_ground_truth_answer": np.where(
                    ground_truth.notna(),
                    (ground_truth > 0).astype(int),
                    np.nan,
                ),
            }
        )
        frames.append(labels)

    if not frames:
        return pd.DataFrame(
            columns=["source_file", "_source_row_id", "ki67_ground_truth_answer"]
        )
    return pd.concat(frames, ignore_index=True)


def attach_ground_truth_from_answers(
    predict_df: pd.DataFrame,
    csv_files: list[Path],
) -> pd.DataFrame:
    """將人工答案標籤合併回 cell-level 預測資料。

    Args:
        predict_df (pd.DataFrame): 模型預測前後的 cell-level 資料表。
        csv_files (list[Path]): 與資料列來源對應的 cleaned CSV 清單。

    Returns:
        pd.DataFrame: 補上 `ki67_ground_truth` 的預測資料表。
    """
    work_df = predict_df.copy()
    work_df["_source_row_id"] = work_df.groupby("source_file", sort=False).cumcount()

    if "ki67_label" in work_df.columns:
        work_df["ki67_ground_truth"] = pd.to_numeric(
            work_df["ki67_label"],
            errors="coerce",
        )
    else:
        work_df["ki67_ground_truth"] = np.nan

    answer_labels = load_answer_labels(csv_files)
    if len(answer_labels) > 0:
        work_df = work_df.merge(
            answer_labels,
            on=["source_file", "_source_row_id"],
            how="left",
        )
        work_df["ki67_ground_truth"] = work_df[
            "ki67_ground_truth_answer"
        ].combine_first(work_df["ki67_ground_truth"])
        work_df = work_df.drop(columns=["ki67_ground_truth_answer"])

    ground_truth = pd.to_numeric(work_df["ki67_ground_truth"], errors="coerce")
    work_df["ki67_ground_truth"] = np.where(
        ground_truth.notna(),
        (ground_truth > 0).astype(int),
        np.nan,
    )
    if pd.notna(work_df["ki67_ground_truth"]).any():
        work_df["ki67_label"] = work_df["ki67_ground_truth"]
    return work_df


def build_cell_prediction_output(predict_df: pd.DataFrame) -> pd.DataFrame:
    """建立 cell-level 預測輸出表。

    Args:
        predict_df (pd.DataFrame): 含 `cell_prob` 與 ground truth 的預測資料。

    Returns:
        pd.DataFrame: 只保留報表需要欄位的 cell-level 預測表。
    """
    cell_df = predict_df.copy()
    cell_df["predicted_ki67_positive"] = (
        cell_df["cell_prob"] >= PREDICTION_THRESHOLD
    ).astype(int)

    ground_truth = pd.to_numeric(cell_df["ki67_ground_truth"], errors="coerce")
    cell_df["prediction_correct"] = np.where(
        ground_truth.notna(),
        cell_df["predicted_ki67_positive"].to_numpy() == ground_truth.to_numpy(),
        pd.NA,
    )

    output_columns = [
        column for column in CELL_PREDICTION_COLUMNS if column in cell_df.columns
    ]
    return cell_df.loc[:, output_columns].copy()


def build_image_prediction_output(image_df: pd.DataFrame) -> pd.DataFrame:
    """建立 image-level 預測輸出表。

    Args:
        image_df (pd.DataFrame): 以影像彙總後的預測資料。

    Returns:
        pd.DataFrame: 移除內部欄位並補上誤差欄位的影像預測表。
    """
    image_output_df = image_df.copy()
    if "true_ratio" in image_output_df.columns:
        image_output_df["answer_ratio"] = image_output_df["true_ratio"]
        image_output_df["abs_error"] = (
            image_output_df["pred_ratio"] - image_output_df["answer_ratio"]
        ).abs()
    else:
        image_output_df["answer_ratio"] = np.nan
        image_output_df["abs_error"] = np.nan

    return image_output_df.drop(
        columns=IMAGE_PREDICTION_DROP_COLUMNS,
        errors="ignore",
    )


def build_folder_vs_answer_output(folder_df: pd.DataFrame) -> pd.DataFrame:
    """建立資料夾層級 prediction-vs-answer 比對表。

    Args:
        folder_df (pd.DataFrame): 以資料夾彙總的預測比例與答案比例。

    Returns:
        pd.DataFrame: 固定欄位順序的資料夾層級比對表。
    """
    folder_output_df = folder_df.copy()
    if "true_ratio" in folder_output_df.columns:
        folder_output_df["answer_ratio"] = folder_output_df["true_ratio"]
    else:
        folder_output_df["answer_ratio"] = np.nan

    folder_output_df["abs_error"] = (
        folder_output_df["pred_ratio"] - folder_output_df["answer_ratio"]
    ).abs()
    folder_output_df["within_0p05"] = (folder_output_df["abs_error"] <= 0.05).astype(
        "boolean"
    )
    folder_output_df["within_0p10"] = (folder_output_df["abs_error"] <= 0.10).astype(
        "boolean"
    )
    folder_output_df.loc[
        folder_output_df["abs_error"].isna(),
        ["within_0p05", "within_0p10"],
    ] = pd.NA

    return folder_output_df.reindex(columns=FOLDER_VS_ANSWER_COLUMNS)


def save_prediction_workbooks(
    folder_output_df: pd.DataFrame,
    image_output_df: pd.DataFrame,
    cell_output_df: pd.DataFrame,
) -> list[Path]:
    """依資料夾輸出 prediction Excel 工作簿。

    Args:
        folder_output_df (pd.DataFrame): 資料夾層級 prediction-vs-answer 表。
        image_output_df (pd.DataFrame): 影像層級預測表。
        cell_output_df (pd.DataFrame): 細胞層級預測表。

    Returns:
        list[Path]: 已輸出的 Excel 工作簿路徑清單。
    """
    workbook_paths = []
    source_folders = sorted(
        image_output_df["source_folder"].dropna().astype(str).unique()
    )
    for source_folder in source_folders:
        workbook_path = OUTPUT_DIR / f"{source_folder}_prediction.xlsx"
        folder_sheet_df = folder_output_df[
            folder_output_df["source_folder"].astype(str) == source_folder
        ].copy()
        image_sheet_df = image_output_df[
            image_output_df["source_folder"].astype(str) == source_folder
        ].copy()
        cell_sheet_df = cell_output_df[
            cell_output_df["source_folder"].astype(str) == source_folder
        ].copy()

        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            folder_sheet_df.to_excel(
                writer,
                sheet_name=WORKBOOK_SHEET_FOLDER,
                index=False,
            )
            image_sheet_df.to_excel(
                writer,
                sheet_name=WORKBOOK_SHEET_IMAGE,
                index=False,
            )
            cell_sheet_df.to_excel(
                writer,
                sheet_name=WORKBOOK_SHEET_CELL,
                index=False,
            )
        workbook_paths.append(workbook_path)
    return workbook_paths


def main() -> None:
    """執行正式預測。"""
    ensure_clean_predict_dir()

    preprocess_bundle = joblib.load(MODEL_DIR / "preprocess_bundle.joblib")
    stage1_model = joblib.load(MODEL_DIR / "stage1_cell_model.joblib")
    stage1_calibrator = joblib.load(MODEL_DIR / "stage1_calibrator.joblib")
    stage2_model = joblib.load(MODEL_DIR / "stage2_ratio_model.joblib")

    ratio_feature_columns = json.loads(
        (MODEL_DIR / "ratio_feature_columns.json").read_text(encoding="utf-8")
    )["feature_names"]
    training_meta = json.loads(
        (MODEL_DIR / "training_meta.json").read_text(encoding="utf-8")
    )

    csv_files = find_cleaned_csv_files(
        results_dir=RESULTS_DIR,
        pattern=DATA_PATTERN,
        include_folders=PREDICT_SOURCE_FOLDERS,
        excluded_source_folders=[],
    )
    predict_df = load_ki67_dataset(
        csv_files=csv_files,
        label_candidates=LABEL_CANDIDATES,
        passage_candidates=PASSAGE_CANDIDATES,
        require_label=False,
    )
    predict_df = attach_ground_truth_from_answers(predict_df, csv_files)
    predict_df["image_key"] = (
        predict_df["source_folder"].astype(str) + "::" + predict_df["Image"].astype(str)
    )

    x_all = transform_features(
        df=predict_df,
        feature_cols=training_meta["raw_feature_columns"],
        prep=preprocess_bundle,
    )
    x_final = x_all.loc[:, preprocess_bundle["selected_features"]].copy()
    raw_prob = predict_stage1_probability(stage1_model, x_final)
    predict_df["raw_cell_prob"] = raw_prob
    predict_df["cell_prob"] = apply_probability_calibration(stage1_calibrator, raw_prob)
    cell_output_df = build_cell_prediction_output(predict_df)
    cell_output_df.to_csv(
        OUTPUT_DIR / "cell_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    image_df = aggregate_to_image_features(predict_df, prob_col="cell_prob")
    design_result = build_image_design_matrices(
        image_tables={"Predict": image_df},
        feature_names=ratio_feature_columns,
    )
    image_df["pred_ratio"] = np.clip(
        stage2_model.predict(design_result["matrices"]["Predict"]),
        0.0,
        1.0,
    )
    image_output_df = build_image_prediction_output(image_df)
    image_output_df.to_csv(
        OUTPUT_DIR / "image_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    folder_df = summarize_folder_ratio(image_df)
    folder_df = add_image_count_diagnostics(folder_df, image_df)
    folder_answer_output_df = build_folder_vs_answer_output(folder_df)
    folder_answer_output_df.to_csv(
        OUTPUT_DIR / "folder_predictions_vs_answer.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compare_df = compare_folder_ratio_to_reference(
        folder_ratio_df=folder_df,
        reference_map=BDL_RATIO_REFERENCE,
        reference_col_name="bdl_true_ratio",
    )
    compare_output_df = compare_df.reindex(columns=FOLDER_VS_BDL_COLUMNS)

    if len(compare_df) > 0:
        compare_output_df.to_csv(
            OUTPUT_DIR / "folder_ratio_vs_bdl.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_folder_ratio_comparison_plot(
            compare_df=compare_output_df,
            save_path=OUTPUT_DIR / "folder_ratio_vs_bdl.png",
            reference_col="bdl_true_ratio",
            reference_label="BDL Ground Truth",
        )

    workbook_paths = save_prediction_workbooks(
        folder_output_df=folder_answer_output_df,
        image_output_df=image_output_df,
        cell_output_df=cell_output_df,
    )

    print(f"Prediction workbooks: {len(workbook_paths)}")

    print("===== Ki67 正式預測完成 =====")
    print(f"輸出目錄: {OUTPUT_DIR}")
    if len(compare_df) > 0:
        print("===== 與 BDL Ground Truth 比較 =====")
        print(compare_output_df.to_string(index=False))


if __name__ == "__main__":
    main()
