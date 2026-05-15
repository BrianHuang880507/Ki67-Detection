"""Ki67 正式訓練與預測共用工具。

此模組集中處理:
1. cleaned.csv 載入與欄位整理。
2. 特徵前處理與特徵篩選。
3. 以影像為單位的資料切分。
4. cell probability 與 image ratio 模型所需的共用函式。
"""

from __future__ import annotations

import json
import re
import matplotlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:  # pylint: disable=broad-except
    LGBMRegressor = None

DEFAULT_LABEL_CANDIDATES = ["ki67_positive", "ki67_label", "label", "target"]
DEFAULT_PASSAGE_CANDIDATES = ["passage", "Passage", "passage_id"]
DEFAULT_DATA_PATTERN = "*_cleaned.csv"
DEFAULT_EXCLUDED_SOURCE_FOLDERS = {"0819"}
DEFAULT_HOLDOUT_SOURCE_FOLDERS = ["P6-1", "P6-2", "P6-3", "P12-1", "P12-2", "P12-3"]
DEFAULT_BDL_RATIO_REFERENCE = {
    "P6-1": 0.732,
    "P6-2": 0.773,
    "P6-3": 0.617,
    "P12-1": 0.492,
    "P12-2": 0.426,
    "P12-3": 0.471,
}
BASE_RATIO_FEATURE_COLUMNS = [
    "mean_cell_prob",
    "median_cell_prob",
    "p90_cell_prob",
    "frac_prob_gt_0_3",
    "frac_prob_gt_0_5",
    "frac_prob_gt_0_7",
    "cell_count",
    "passage",
]


class IdentityProbabilityCalibrator:
    """不做任何變換的機率校正器。"""

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> "IdentityProbabilityCalibrator":
        """相容 sklearn 介面的 fit。"""
        _ = raw_prob, y_true
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        """直接回傳原始機率。"""
        return np.clip(np.asarray(raw_prob, dtype=np.float64), 0.0, 1.0)


def infer_passage_from_text(text: str) -> str:
    """從文字中抓出最後一個 passage。

    Args:
        text: 來源字串。

    Returns:
        passage 文字，例如 `P6`。若找不到則回傳 `P_UNKNOWN`。
    """
    matches = re.findall(r"(?i)\bP(?:assage)?[-_ ]?(\d{1,2})\b", str(text))
    if matches:
        return f"P{int(matches[-1])}"
    return "P_UNKNOWN"


def pick_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """從候選欄位中挑出第一個存在欄位。

    Args:
        columns: 實際欄位名稱。
        candidates: 候選欄位名稱。

    Returns:
        匹配到的欄位名稱，若無則為 `None`。
    """
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def find_cleaned_csv_files(
    results_dir: Path,
    pattern: str = DEFAULT_DATA_PATTERN,
    include_folders: Optional[Sequence[str]] = None,
    excluded_source_folders: Optional[Sequence[str]] = None,
    source_folder_keyword: Optional[str] = None,
) -> List[Path]:
    """搜尋 cleaned.csv 檔案。

    Args:
        results_dir: `data/output/results` 目錄。
        pattern: 檔案比對樣式。
        include_folders: 若指定，只保留這些 source_folder。
        excluded_source_folders: 排除的 source_folder。
        source_folder_keyword: source_folder 名稱需包含的關鍵字。

    Returns:
        CSV 檔案路徑清單。
    """
    include_set = None
    if include_folders:
        include_set = {str(item).strip().lower() for item in include_folders if str(item).strip()}

    excluded_set = {
        str(item).strip().lower()
        for item in (excluded_source_folders or DEFAULT_EXCLUDED_SOURCE_FOLDERS)
        if str(item).strip()
    }
    keyword = None if source_folder_keyword is None else str(source_folder_keyword).strip().lower()

    csv_files: List[Path] = []
    for csv_path in sorted(results_dir.rglob(pattern)):
        if not csv_path.is_file():
            continue
        source_folder = csv_path.parent.name
        source_folder_lower = source_folder.lower()
        if source_folder_lower in excluded_set:
            continue
        if include_set is not None and source_folder_lower not in include_set:
            continue
        if keyword and keyword not in source_folder_lower:
            continue
        csv_files.append(csv_path)
    return csv_files


def load_ki67_dataset(
    csv_files: Sequence[Path],
    label_candidates: Sequence[str],
    passage_candidates: Sequence[str],
    require_label: bool,
) -> pd.DataFrame:
    """載入 Ki67 cleaned.csv 資料集。

    Args:
        csv_files: cleaned.csv 路徑清單。
        label_candidates: Ki67 標籤候選欄位。
        passage_candidates: passage 候選欄位。
        require_label: 是否要求標籤欄位存在。

    Returns:
        合併後的資料表。

    Raises:
        ValueError: 沒有找到可用資料時拋出。
    """
    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as error:  # pylint: disable=broad-except
            print(f"[警告] 無法讀取 {csv_path}: {error}")
            continue

        local_df = df.copy()
        local_df = local_df.loc[:, ~local_df.columns.duplicated()].copy()

        label_col = pick_existing_column(local_df.columns, label_candidates)
        if require_label and label_col is None:
            continue
        if label_col is not None:
            local_df = local_df.rename(columns={label_col: "ki67_label"})

        passage_col = pick_existing_column(local_df.columns, passage_candidates)
        if passage_col is not None:
            local_df["passage"] = local_df[passage_col].astype(str).map(infer_passage_from_text)
        else:
            local_df["passage"] = infer_passage_from_text(csv_path.parent.name)

        if "Image" not in local_df.columns:
            local_df["Image"] = csv_path.name

        local_df["source_file"] = str(csv_path)
        local_df["source_folder"] = csv_path.parent.name
        frames.append(local_df)

    if not frames:
        raise ValueError("找不到可用的 cleaned.csv 資料。")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.loc[:, ~dataset.columns.duplicated()].copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    if "ki67_label" in dataset.columns:
        dataset["ki67_label"] = pd.to_numeric(dataset["ki67_label"], errors="coerce")
        dataset = dataset.dropna(subset=["ki67_label"]).copy()
        dataset["ki67_label"] = (dataset["ki67_label"] > 0).astype(int)

    return dataset.reset_index(drop=True)


def detect_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """找出數值特徵欄位。

    Args:
        df: 資料表。

    Returns:
        可作為特徵的數值欄位名稱。

    Raises:
        ValueError: 沒有數值特徵時拋出。
    """
    exclude_cols = {"ki67_label", "passage", "source_file", "source_folder"}
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]
    if not feature_cols:
        raise ValueError("找不到可用的數值特徵欄位。")
    return feature_cols


def build_image_key(df: pd.DataFrame) -> pd.Series:
    """建立影像群組鍵值。

    Args:
        df: cell-level 資料表。

    Returns:
        每列對應的 `image_key`。
    """
    image_name = df["Image"].astype(str) if "Image" in df.columns else df["source_file"].astype(str).map(lambda x: Path(x).name)
    return df["source_folder"].astype(str) + "::" + image_name.astype(str)


def fit_preprocessor(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, Any]:
    """擬合前處理器。

    Args:
        train_df: 訓練資料。
        feature_cols: 原始特徵欄位。

    Returns:
        前處理器字典。
    """
    imputer = SimpleImputer(strategy="median")
    variance_selector = VarianceThreshold(threshold=0.0)
    scaler = StandardScaler()

    x_train = train_df.loc[:, feature_cols]
    x_train_imp = imputer.fit_transform(x_train)
    x_train_var = variance_selector.fit_transform(x_train_imp)
    scaler.fit(x_train_var)

    keep_indices = variance_selector.get_support(indices=True)
    kept_features = [feature_cols[idx] for idx in keep_indices]

    return {
        "imputer": imputer,
        "variance_selector": variance_selector,
        "scaler": scaler,
        "kept_features": kept_features,
    }


def transform_features(df: pd.DataFrame, feature_cols: Sequence[str], prep: Mapping[str, Any]) -> pd.DataFrame:
    """套用前處理器並回傳 DataFrame。

    Args:
        df: 要轉換的資料。
        feature_cols: 原始特徵欄位。
        prep: 前處理器字典。

    Returns:
        轉換後特徵表。
    """
    x_values = df.loc[:, feature_cols]
    x_values = prep["imputer"].transform(x_values)
    x_values = prep["variance_selector"].transform(x_values)
    x_values = prep["scaler"].transform(x_values)
    return pd.DataFrame(x_values, columns=prep["kept_features"], index=df.index)


def compute_feature_target_correlation(x_df: pd.DataFrame, y: pd.Series) -> pd.Series:
    """計算特徵與 Ki67 的 Spearman 相關。

    Args:
        x_df: 特徵表。
        y: Ki67 標籤。

    Returns:
        由高到低排序的相關係數。
    """
    corr_values: Dict[str, float] = {}
    y_values = y.to_numpy(dtype=np.float32)
    for col in x_df.columns:
        x_values = x_df[col].to_numpy(dtype=np.float32)
        if np.nanstd(x_values) == 0:
            corr_values[col] = 0.0
            continue
        corr = spearmanr(x_values, y_values, nan_policy="omit").correlation
        corr_values[col] = 0.0 if pd.isna(corr) else float(corr)
    return pd.Series(corr_values).sort_values(ascending=False)


def filter_similar_features(x_train: pd.DataFrame, y_train: pd.Series, threshold: float) -> Dict[str, Any]:
    """移除彼此過度相似的特徵。

    Args:
        x_train: 訓練特徵。
        y_train: 訓練標籤。
        threshold: 特徵間相似度上限。

    Returns:
        保留與刪除特徵資訊。
    """
    corr_target = compute_feature_target_correlation(x_train, y_train)
    corr_matrix = x_train.corr(method="spearman").abs()
    features = list(x_train.columns)
    drop_set = set()

    for index_i, feat_i in enumerate(features):
        if feat_i in drop_set:
            continue
        for index_j in range(index_i + 1, len(features)):
            feat_j = features[index_j]
            if feat_j in drop_set:
                continue
            similarity = corr_matrix.loc[feat_i, feat_j]
            if pd.isna(similarity) or similarity < threshold:
                continue

            score_i = abs(corr_target.get(feat_i, 0.0))
            score_j = abs(corr_target.get(feat_j, 0.0))
            if score_i < score_j:
                drop_set.add(feat_i)
                break
            drop_set.add(feat_j)

    kept_features = [feature for feature in features if feature not in drop_set]
    return {
        "kept_features": kept_features,
        "dropped_features": sorted(drop_set),
        "corr_matrix": corr_matrix,
    }


def select_positive_correlation_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    min_corr: float,
) -> Dict[str, Any]:
    """保留與 Ki67 正相關的特徵。

    Args:
        x_train: 訓練特徵。
        y_train: 訓練標籤。
        min_corr: 最低正相關門檻。

    Returns:
        最終選用特徵與報表。
    """
    corr_series = compute_feature_target_correlation(x_train, y_train)
    selected_features = corr_series[corr_series > min_corr].index.tolist()
    if not selected_features and len(corr_series) > 0:
        selected_features = [corr_series.index[0]]

    report_df = pd.DataFrame(
        {
            "feature": corr_series.index,
            "corr_to_ki67": corr_series.values,
            "is_selected": [item in selected_features for item in corr_series.index],
        }
    )
    return {
        "selected_features": selected_features,
        "correlation_report": report_df,
    }


def split_dev_and_holdout(
    dataset: pd.DataFrame,
    holdout_folders: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """將資料切成開發集與 holdout 集。

    Args:
        dataset: 完整資料集。
        holdout_folders: holdout source_folder 清單。

    Returns:
        `dev_df` 與 `holdout_df`。
    """
    holdout_set = {str(item).strip().lower() for item in holdout_folders if str(item).strip()}
    if not holdout_set:
        return {
            "dev_df": dataset.copy().reset_index(drop=True),
            "holdout_df": dataset.iloc[0:0].copy().reset_index(drop=True),
        }

    is_holdout = dataset["source_folder"].astype(str).str.lower().isin(holdout_set)
    return {
        "dev_df": dataset.loc[~is_holdout].copy().reset_index(drop=True),
        "holdout_df": dataset.loc[is_holdout].copy().reset_index(drop=True),
    }


def _allocate_split_counts(n_groups: int, test_size: float, val_size: float) -> Tuple[int, int, int]:
    """依群組數配置 train/val/test 比例。"""
    if n_groups <= 2:
        return n_groups, 0, 0

    n_test = int(round(n_groups * test_size))
    n_val = int(round(n_groups * val_size))
    if test_size > 0:
        n_test = max(1, n_test)
    if val_size > 0:
        n_val = max(1, n_val)

    while n_groups - n_test - n_val < 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    return n_groups - n_test - n_val, n_val, n_test


def split_by_image_within_passage(
    dataset: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """以 image 為單位、在各 passage 內切分 train/val/test。

    Args:
        dataset: 開發集資料。
        test_size: test 比例。
        val_size: val 比例。
        random_state: 隨機種子。

    Returns:
        切分後資料與 source_folder 使用摘要。
    """
    work_df = dataset.copy()
    work_df["image_key"] = build_image_key(work_df)
    rng = np.random.default_rng(random_state)
    split_map: Dict[str, str] = {}

    for passage, part_df in work_df.groupby("passage", sort=True):
        image_keys = part_df["image_key"].drop_duplicates().tolist()
        rng.shuffle(image_keys)
        n_train, n_val, n_test = _allocate_split_counts(len(image_keys), test_size, val_size)

        train_keys = image_keys[:n_train]
        val_keys = image_keys[n_train : n_train + n_val]
        test_keys = image_keys[n_train + n_val : n_train + n_val + n_test]

        for image_key in train_keys:
            split_map[image_key] = "Train"
        for image_key in val_keys:
            split_map[image_key] = "Val"
        for image_key in test_keys:
            split_map[image_key] = "Test"

    work_df["split_name"] = work_df["image_key"].map(split_map).fillna("Train")
    train_df = work_df[work_df["split_name"] == "Train"].copy().reset_index(drop=True)
    val_df = work_df[work_df["split_name"] == "Val"].copy().reset_index(drop=True)
    test_df = work_df[work_df["split_name"] == "Test"].copy().reset_index(drop=True)

    usage_rows: List[Dict[str, Any]] = []
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        for source_folder, folder_df in split_df.groupby("source_folder", sort=True):
            usage_rows.append(
                {
                    "split_name": split_name,
                    "source_folder": source_folder,
                    "image_count": int(folder_df["image_key"].nunique()),
                    "cell_count": int(len(folder_df)),
                    "positive_ratio": float(folder_df["ki67_label"].mean()),
                }
            )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "source_folder_usage_df": pd.DataFrame(usage_rows),
    }


def fit_stage1_cell_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    use_class_weight: bool,
) -> SGDClassifier:
    """擬合第一階段 cell probability 模型。

    Args:
        x_train: 訓練特徵。
        y_train: 訓練標籤。
        random_state: 隨機種子。
        use_class_weight: 是否使用 `balanced` 權重。

    Returns:
        已訓練的 SGDClassifier。
    """
    class_weight = "balanced" if use_class_weight else None
    model = SGDClassifier(
        loss="log_loss",
        max_iter=3000,
        tol=1e-3,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(x_train.to_numpy(dtype=np.float32), y_train.to_numpy(dtype=np.int64))
    return model


def fit_stage1_cell_model_with_oof(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    image_keys: pd.Series,
    random_state: int,
    use_class_weight: bool,
) -> Dict[str, Any]:
    """用 GroupKFold 建立 OOF cell probability。

    Args:
        x_train: 訓練特徵。
        y_train: 訓練標籤。
        image_keys: 影像群組鍵值。
        random_state: 隨機種子。
        use_class_weight: 是否使用 `balanced` 權重。

    Returns:
        最終模型與 OOF probability。
    """
    unique_group_count = int(image_keys.nunique())
    n_splits = min(5, unique_group_count)
    x_np = x_train.to_numpy(dtype=np.float32)
    y_np = y_train.to_numpy(dtype=np.int64)

    if n_splits < 2:
        model = fit_stage1_cell_model(x_train, y_train, random_state, use_class_weight)
        train_prob = model.predict_proba(x_np)[:, 1]
        return {"model": model, "train_oof_prob": train_prob, "n_splits": 1}

    oof_prob = np.zeros(len(x_train), dtype=np.float32)
    group_kfold = GroupKFold(n_splits=n_splits)
    for fold_index, (fit_index, val_index) in enumerate(group_kfold.split(x_np, y_np, groups=image_keys)):
        model = fit_stage1_cell_model(
            x_train=pd.DataFrame(x_np[fit_index], columns=x_train.columns),
            y_train=pd.Series(y_np[fit_index]),
            random_state=random_state + fold_index,
            use_class_weight=use_class_weight,
        )
        oof_prob[val_index] = model.predict_proba(x_np[val_index])[:, 1]

    final_model = fit_stage1_cell_model(x_train, y_train, random_state, use_class_weight)
    return {"model": final_model, "train_oof_prob": oof_prob, "n_splits": n_splits}


def predict_stage1_probability(model: SGDClassifier, x_df: pd.DataFrame) -> np.ndarray:
    """預測 cell-level 陽性機率。"""
    return model.predict_proba(x_df.to_numpy(dtype=np.float32))[:, 1]


def fit_probability_calibrator(
    raw_prob: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
) -> Any:
    """擬合機率校正器。

    Args:
        raw_prob: 原始機率。
        y_true: 真實標籤。
        method: 校正方法，支援 `none` 與 `isotonic`。

    Returns:
        已訓練的校正器。
    """
    method = str(method).strip().lower()
    raw_prob = np.asarray(raw_prob, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    if method in {"", "none"}:
        return IdentityProbabilityCalibrator().fit(raw_prob, y_true)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_prob, y_true)
        return calibrator

    raise ValueError(f"不支援的 probability calibration 方法: {method}")


def apply_probability_calibration(calibrator: Any, raw_prob: np.ndarray) -> np.ndarray:
    """套用機率校正器。

    Args:
        calibrator: 已訓練校正器。
        raw_prob: 原始機率。

    Returns:
        校正後機率。
    """
    raw_prob = np.asarray(raw_prob, dtype=np.float64)
    if hasattr(calibrator, "predict"):
        calibrated = calibrator.predict(raw_prob)
    else:
        calibrated = raw_prob
    return np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, 1.0)


def aggregate_to_image_features(cell_df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """將 cell probability 聚合成 image-level 特徵。

    Args:
        cell_df: cell-level 資料。
        prob_col: 機率欄位名稱。

    Returns:
        image-level 特徵表。
    """
    work_df = cell_df.copy()
    if "image_key" not in work_df.columns:
        work_df["image_key"] = build_image_key(work_df)

    grouped = work_df.groupby("image_key", as_index=False)
    agg_dict = {
        "source_folder": ("source_folder", "first"),
        "image_name": ("Image", "first") if "Image" in work_df.columns else ("source_file", "first"),
        "passage": ("passage", "first"),
        "cell_count": (prob_col, "size"),
        "mean_cell_prob": (prob_col, "mean"),
        "median_cell_prob": (prob_col, "median"),
    }
    if "ki67_label" in work_df.columns:
        agg_dict["true_ratio"] = ("ki67_label", "mean")

    image_df = grouped.agg(**agg_dict)
    extra_rows: List[Dict[str, Any]] = []
    for image_key, image_part in grouped:
        probs = image_part[prob_col].to_numpy(dtype=np.float32)
        extra_rows.append(
            {
                "image_key": image_key,
                "p90_cell_prob": float(np.quantile(probs, 0.90)),
                "frac_prob_gt_0_3": float((probs > 0.30).mean()),
                "frac_prob_gt_0_5": float((probs > 0.50).mean()),
                "frac_prob_gt_0_7": float((probs > 0.70).mean()),
            }
        )
    extra_df = pd.DataFrame(extra_rows)
    return image_df.merge(extra_df, on="image_key", how="left")


def build_image_design_matrices(
    image_tables: Mapping[str, pd.DataFrame],
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """建立第二階段 image ratio 設計矩陣。

    Args:
        image_tables: 以 split 名稱為 key 的 image-level 表格。
        feature_names: 若指定，會依既有欄位順序對齊。

    Returns:
        設計矩陣與最終欄位名稱。
    """
    parts: List[pd.DataFrame] = []
    split_names = list(image_tables.keys())
    for split_name, image_df in image_tables.items():
        if image_df is None or len(image_df) == 0:
            continue
        part_df = image_df.loc[:, BASE_RATIO_FEATURE_COLUMNS].copy()
        part_df["__split_name__"] = split_name
        parts.append(part_df)

    if not parts:
        final_feature_names = list(feature_names or [])
        empty_matrices = {
            split_name: pd.DataFrame(columns=final_feature_names)
            for split_name in split_names
        }
        return {"feature_names": final_feature_names, "matrices": empty_matrices}

    merged = pd.concat(parts, axis=0, ignore_index=True)
    merged = pd.get_dummies(merged, columns=["passage"], dtype=float)

    if feature_names is None:
        final_feature_names = [col for col in merged.columns if col != "__split_name__"]
    else:
        final_feature_names = list(feature_names)
        for column in final_feature_names:
            if column not in merged.columns:
                merged[column] = 0.0

    matrices: Dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        split_part = merged[merged["__split_name__"] == split_name].copy()
        matrices[split_name] = split_part.reindex(columns=final_feature_names, fill_value=0.0).reset_index(drop=True)

    return {"feature_names": final_feature_names, "matrices": matrices}


def fit_ratio_models(random_state: int) -> Dict[str, Any]:
    """建立第二階段 image ratio 候選模型。"""
    models: Dict[str, Any] = {
        "Ridge": Ridge(alpha=1.0),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if LGBMRegressor is not None:
        models["LightGBMRegressor"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return models


def evaluate_ratio_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """評估 image-level ratio 預測表現。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    abs_error = np.abs(y_pred - y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_0p05": float((abs_error <= 0.05).mean()),
        "within_0p10": float((abs_error <= 0.10).mean()),
    }


def summarize_folder_ratio(image_pred_df: pd.DataFrame) -> pd.DataFrame:
    """將 image-level 預測彙整成 source_folder 層級比例。

    Args:
        image_pred_df: 至少含 `source_folder`、`cell_count`、`pred_ratio` 的資料表。

    Returns:
        source_folder 層級比例表。
    """
    rows: List[Dict[str, Any]] = []
    for source_folder, folder_df in image_pred_df.groupby("source_folder", sort=True):
        weights = folder_df["cell_count"].to_numpy(dtype=np.float64)
        pred_ratio = np.average(folder_df["pred_ratio"].to_numpy(dtype=np.float64), weights=weights)
        row: Dict[str, Any] = {
            "source_folder": source_folder,
            "image_count": int(folder_df["image_key"].nunique()),
            "cell_count": int(folder_df["cell_count"].sum()),
            "pred_ratio": float(pred_ratio),
        }
        if "true_ratio" in folder_df.columns:
            true_ratio = np.average(folder_df["true_ratio"].to_numpy(dtype=np.float64), weights=weights)
            row["true_ratio"] = float(true_ratio)
            row["abs_error"] = float(abs(pred_ratio - true_ratio))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("source_folder").reset_index(drop=True)


def compare_folder_ratio_to_reference(
    folder_ratio_df: pd.DataFrame,
    reference_map: Mapping[str, float],
    reference_col_name: str,
) -> pd.DataFrame:
    """將 folder-level 預測與外部 ground truth 比較。

    Args:
        folder_ratio_df: folder-level 預測表。
        reference_map: 外部真值對照。
        reference_col_name: 真值欄位名稱。

    Returns:
        比較後的資料表。
    """
    compare_df = folder_ratio_df.copy()
    compare_df[reference_col_name] = compare_df["source_folder"].map(reference_map)
    compare_df = compare_df.dropna(subset=[reference_col_name]).copy()
    compare_df["abs_error"] = (compare_df["pred_ratio"] - compare_df[reference_col_name]).abs()
    compare_df["within_0p05"] = compare_df["abs_error"] <= 0.05
    compare_df["within_0p10"] = compare_df["abs_error"] <= 0.10
    return compare_df.reset_index(drop=True)


def save_feature_correlation_plot(report_df: pd.DataFrame, save_path: Path, top_k: int = 20) -> None:
    """儲存特徵與 Ki67 相關性長條圖。

    Args:
        report_df: 特徵報表，需含 `feature` 與 `corr_to_ki67`。
        save_path: 圖檔路徑。
        top_k: 顯示前幾個特徵。
    """
    if len(report_df) == 0:
        return

    plot_df = report_df.sort_values("corr_to_ki67", ascending=False).head(top_k).copy()
    colors = ["#c44e52" if flag else "#4c72b0" for flag in plot_df["is_selected"].astype(bool)]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.barh(plot_df["feature"], plot_df["corr_to_ki67"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation to Ki67")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Correlation Ranking")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_config_comparison_plot(config_df: pd.DataFrame, save_path: Path) -> None:
    """儲存組態比較圖。

    Args:
        config_df: 組態比較表，需含 `config_name` 與數值指標欄位。
        save_path: 圖檔路徑。
    """
    if len(config_df) == 0:
        return

    plot_cols = [col for col in ["lfso_folder_mae", "lfso_image_mae", "holdout_bdl_mae"] if col in config_df.columns]
    if not plot_cols:
        return

    x_labels = config_df["config_name"].astype(str).tolist()
    x_pos = np.arange(len(x_labels))
    width = 0.22 if len(plot_cols) >= 3 else 0.32
    offsets = np.linspace(-(len(plot_cols) - 1) / 2.0, (len(plot_cols) - 1) / 2.0, len(plot_cols)) * width
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(max(10, len(x_labels) * 1.4), 6), dpi=180)
    for index, column in enumerate(plot_cols):
        ax.bar(x_pos + offsets[index], config_df[column].to_numpy(dtype=float), width=width, color=colors[index], label=column)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Configuration Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_calibration_curve_plot(
    y_true: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    save_path: Path,
) -> None:
    """儲存 calibration curve 圖。

    Args:
        y_true: 真實標籤。
        raw_prob: 原始機率。
        calibrated_prob: 校正後機率。
        save_path: 圖檔路徑。
    """
    if len(y_true) == 0:
        return

    y_true = np.asarray(y_true, dtype=np.int64)
    raw_prob = np.asarray(raw_prob, dtype=np.float64)
    calibrated_prob = np.asarray(calibrated_prob, dtype=np.float64)

    raw_true, raw_pred = calibration_curve(y_true, raw_prob, n_bins=10, strategy="quantile")
    cal_true, cal_pred = calibration_curve(y_true, calibrated_prob, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="Ideal")
    ax.plot(raw_pred, raw_true, marker="o", color="#dd8452", label="Raw")
    ax.plot(cal_pred, cal_true, marker="o", color="#4c72b0", label="Calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive ratio")
    ax.set_title("Stage1 Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_folder_ratio_comparison_plot(
    compare_df: pd.DataFrame,
    save_path: Path,
    reference_col: str,
    reference_label: str,
) -> None:
    """儲存 folder-level 預測比例與真值對照圖。

    Args:
        compare_df: 比較表。
        save_path: 圖檔路徑。
        reference_col: 真值欄位名稱。
        reference_label: 真值圖例名稱。
    """
    if len(compare_df) == 0:
        return

    plot_df = compare_df.copy()
    x_labels = plot_df["source_folder"].astype(str).tolist()
    x_pos = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.4), 6), dpi=180)
    ax.bar(x_pos - width / 2.0, plot_df[reference_col].to_numpy(dtype=float), width=width, color="#4c72b0", label=reference_label)
    ax.bar(x_pos + width / 2.0, plot_df["pred_ratio"].to_numpy(dtype=float), width=width, color="#dd8452", label="Predicted")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Ki67 ratio")
    ax.set_title("Folder-level Ki67 Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_json(save_path: Path, payload: Mapping[str, Any]) -> None:
    """以 UTF-8 儲存 JSON。"""
    save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
