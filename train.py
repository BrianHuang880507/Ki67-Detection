#!/usr/bin/env python3
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier
import xgboost
import sklearn
import warnings
import re


TARGET_COL = "ki67_positive"
ID_COL = "Cell_ID"
SUPPORTED_BACKBONES = {"resnet18", "resnet50"}
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]


def log(message: str) -> None:
    """
    這個函式會在終端輸出帶有時間戳記的訊息。

    參數:
    message (str): 要顯示的文字內容。

    回傳值:
    None: 此函式沒有回傳值。
    """
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class MultiBatchImageDataset(Dataset):
    def __init__(self, img_root: Path, size: int = 224, include_overlay: bool = False):
        """
        這個方法會初始化資料集，收集每個批次下所有符合條件的影像路徑。

        參數:
        img_root (Path): 影像資料的根目錄。
        size (int): 影像縮放後的邊長。
        include_overlay (bool): 是否保留檔名結尾為 `_overlay` 的影像。

        回傳值:
        None: 此建構子沒有回傳值。
        """
        self.img_root = Path(img_root)
        self.size = size
        self.include_overlay = include_overlay
        if not self.img_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.img_root}")
        self.paths: List[Path] = []
        self.batches: List[str] = []
        for batch_dir in sorted(self.img_root.glob("*")):
            if not batch_dir.is_dir():
                continue
            batch_name = batch_dir.name
            for img_path in sorted(batch_dir.rglob("*")):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                if not include_overlay and img_path.stem.lower().endswith("_overlay"):
                    continue
                self.paths.append(img_path)
                self.batches.append(batch_name)
        self.transform = T.Compose(
            [
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_STD),
            ]
        )

    def __len__(self) -> int:
        """
        這個方法會回傳資料集中可用影像的數量。

        參數:
        無。

        回傳值:
        int: 影像檔案的總數。
        """
        return len(self.paths)

    def __getitem__(self, idx: int):
        """
        這個方法會讀取指定索引的影像並轉為模型可用的張量。

        參數:
        idx (int): 目標影像的索引值。

        回傳值:
        tuple: 包含影像張量、影像檔名（小寫，不含副檔名）及批次名稱。
        """
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, path.stem.lower(), self.batches[idx]


def build_backbone(arch: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """
    這個函式會依照指定架構建立 ResNet 主幹網路並移除分類層。

    參數:
    arch (str): Backbone 名稱，目前支援 `resnet18` 與 `resnet50`。
    pretrained (bool): 是否載入預訓練權重。

    回傳值:
    tuple: 包含處理後的模型與輸出特徵維度。
    """
    arch = arch.lower()
    if arch not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {arch}")
    if arch == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except AttributeError:
            backbone = models.resnet18(pretrained=pretrained)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif arch == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        except AttributeError:
            backbone = models.resnet50(pretrained=pretrained)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {arch}")
    return backbone, feat_dim


@torch.no_grad()
def extract_all_embeddings(
    dataset: Dataset,
    arch: str,
    pretrained: bool,
    batch_size: int,
    device: Optional[str],
) -> Tuple[pd.DataFrame, int]:
    """
    這個函式會批次抽取所有影像的特徵向量並組成資料表。

    參數:
    dataset (Dataset): 來源影像資料集。
    arch (str): 指定的 ResNet 架構名稱。
    pretrained (bool): 是否使用預訓練權重。
    batch_size (int): DataLoader 的批次大小。
    device (Optional[str]): 指定使用的裝置，預設自動偵測。

    回傳值:
    tuple: 包含特徵 DataFrame 與特徵維度大小。
    """
    total = len(dataset)
    log(
        f"Extracting embeddings from {total} images using {arch} (pretrained={pretrained})"
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    if device:
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, feat_dim = build_backbone(arch, pretrained)
    backbone.to(torch_device).eval()
    features: List[np.ndarray] = []
    stems: List[str] = []
    batches: List[str] = []
    for batch_idx, (images, stem_batch, batch_names) in enumerate(dataloader):
        images = images.to(torch_device, non_blocking=True)
        outputs = backbone(images).cpu().numpy()
        features.append(outputs)
        stems.extend([s.lower() for s in stem_batch])
        batches.extend([b for b in batch_names])
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            log(f"  processed {len(stems)}/{total} images")
    if features:
        array = np.concatenate(features, axis=0).astype(np.float32)
    else:
        array = np.empty((0, feat_dim), dtype=np.float32)
    emb_df = pd.DataFrame(array)
    emb_df["_stem"] = stems
    emb_df["_batch"] = batches
    return emb_df, feat_dim


def normalize_pathlike_to_stem(series: pd.Series) -> pd.Series:
    """
    這個函式會把路徑字串標準化為檔名（不含副檔名）的小寫格式。

    參數:
    series (pd.Series): 含有路徑或檔名的字串序列。

    回傳值:
    pd.Series: 標準化後的檔名序列。
    """
    return (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
        .str.strip()
        .str.replace(r"\\.(png|jpg|jpeg)$", "", regex=True, case=False)
        .str.lower()
    )


def cellid_to_stem(series: pd.Series, channel: str) -> pd.Series:
    """
    這個函式會將 Cell_ID 轉換為對應的影像檔名 stem。

    參數:
    series (pd.Series): 含有 Cell_ID 的欄位資料。
    channel (str): 要套用的通道名稱（例如 `cyto`）。

    回傳值:
    pd.Series: 轉換後的檔名 stem 序列。
    """
    channel = channel.lower().strip()

    def _convert(value: object) -> str:
        text = str(value).strip().lower()
        if not text:
            return f"unknown_{channel}_000"

        parts = [p for p in text.split("_") if p != ""]
        base_parts = []
        chan_part = None
        idx_num = None

        for part in parts:
            if part in {"cyto", "nuc"} and chan_part is None:
                chan_part = part
            elif part.isdigit() and idx_num is None:
                try:
                    idx_num = int(part)
                except ValueError:
                    idx_num = None
            else:
                base_parts.append(part)

        if not base_parts:
            base = text.replace("_", "-") or "unknown"
        else:
            base = "_".join(base_parts)

        chan = chan_part or channel
        idx_num = 0 if idx_num is None else max(idx_num, 0)

        return f"{base}_{chan}_{idx_num:03d}"

    return series.apply(_convert)


def make_logreg() -> Pipeline:
    """
    這個函式會建立帶有補值與標準化流程的邏輯迴歸管線。

    參數:
    無。

    回傳值:
    Pipeline: 可直接訓練的 sklearn 管線模型。
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, solver="liblinear", class_weight="balanced"
                ),
            ),
        ]
    )


def make_xgb(scale_pos_weight: Optional[float] = None) -> XGBClassifier:
    """
    這個函式會建立固定超參數的 XGBoost 分類器。

    參數:
    scale_pos_weight (Optional[float]): 額外的類別不平衡權重，預設為 None。

    回傳值:
    XGBClassifier: 已設定好參數的 XGBoost 模型。
    """
    params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "eval_metric": "logloss",
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    init_params = XGBClassifier.__init__.__code__.co_varnames
    if "use_label_encoder" in init_params:
        params["use_label_encoder"] = False
    return XGBClassifier(**params)


def pick_cv(
    y: np.ndarray,
    groups: Optional[Sequence[str]] = None,
    max_splits: int = 5,
    random_state: int = 42,
):
    """
    這個函式會根據資料情況挑選合適的交叉驗證策略。

    參數:
    y (np.ndarray): 目標值陣列。
    groups (Optional[Sequence[str]]): 分組資訊，供 GroupKFold 使用。
    max_splits (int): 允許的最大分割數。
    random_state (int): StratifiedKFold 的隨機種子。

    回傳值:
    tuple: 包含 CV 物件、額外參數 dict 與模式名稱。
    """
    y_array = np.asarray(y)
    if groups is not None:
        groups_array = np.asarray(groups)
        unique_groups = pd.Series(groups_array).nunique()
        if unique_groups >= 2:
            n_splits = min(max_splits, unique_groups)
            if n_splits >= 2:
                return GroupKFold(n_splits=n_splits), {"groups": groups_array}, "group"
    pos = int((y_array == 1).sum())
    neg = int((y_array == 0).sum())
    max_by_class = max(1, min(pos, neg))
    n_splits = max(2, min(max_splits, max_by_class))
    return (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        {},
        "strat",
    )


def xgb_gain_series(
    trained_xgb: XGBClassifier, feature_names: Sequence[str]
) -> pd.Series:
    """
    這個函式會整理 XGBoost 模型的 gain 重要度成為排序後的 Series。

    參數:
    trained_xgb (XGBClassifier): 已訓練完成的 XGB 模型。
    feature_names (Sequence[str]): 特徵名稱列表。

    回傳值:
    pd.Series: 依重要度排序的數值序列。
    """
    booster = trained_xgb.get_booster()
    raw = booster.get_score(importance_type="gain")
    mapped: Dict[str, float] = {}
    feat_list = list(feature_names)
    feat_set = set(feat_list)
    for key, value in raw.items():
        if isinstance(key, str) and key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if 0 <= idx < len(feat_list):
                mapped[str(feat_list[idx])] = value
        elif key in feat_set:
            mapped[str(key)] = value
    series = pd.Series(mapped, name="gain_importance").sort_values(ascending=False)
    return series


def permutation_importance_df(
    trained_pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 20,
) -> pd.DataFrame:
    """
    這個函式會計算 permutation importance 並回傳資料表。

    參數:
    trained_pipeline (Pipeline): 已訓練的管線模型。
    X (pd.DataFrame): 特徵資料。
    y (np.ndarray): 目標值陣列。
    feature_names (Sequence[str]): 特徵名稱。
    n_repeats (int): 隨機打亂次數。

    回傳值:
    pd.DataFrame: 包含平均值與標準差的特徵重要度表。
    """
    perm = permutation_importance(
        trained_pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring="roc_auc",
    )
    df_imp = pd.DataFrame(
        {
            "feature": list(feature_names),
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    ).sort_values("perm_mean", ascending=False)
    return df_imp


def average_rank_from_bucket(bucket: Dict[str, List[float]]) -> pd.Series:
    """
    這個函式會計算同一特徵跨檔案平均排名。

    參數:
    bucket (Dict[str, List[float]]): 特徵名稱對應排名列表的字典。

    回傳值:
    pd.Series: 依平均排名排序的結果。
    """
    data = {f: np.nanmean(vals) for f, vals in bucket.items() if len(vals) > 0}
    series = pd.Series(data, name="avg_rank").dropna().sort_values()
    if not series.empty:
        mask = series.index.to_series().map(
            lambda x: x is not None and str(x).strip() != "" and str(x).lower() != "nan"
        )
        series = series[mask]
    return series


def analyze_csv_files(
    csv_paths: Sequence[Path],
    csv_root: Path,
    channel: str,
    out_root: Path,
    skip_permutation: bool,
    max_splits: int,
):
    """
    這個函式會逐一分析指定的 CSV 檔案並輸出交叉驗證與重要度結果。

    參數:
    csv_paths (Sequence[Path]): 所有待處理的 CSV 路徑列表。
    csv_root (Path): CSV 根目錄，用於產生相對路徑。
    channel (str): 轉換影像檔名時使用的通道名稱。
    out_root (Path): 分析結果輸出的根資料夾。
    skip_permutation (bool): 是否略過 permutation importance。
    max_splits (int): 交叉驗證的最大分割數。

    回傳值:
    tuple: 包含併表後的 DataFrame、摘要表與平均排名結果。
    """
    csv_root = Path(csv_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    gain_rank_bucket: Dict[str, List[float]] = defaultdict(list)
    perm_rank_bucket: Dict[str, List[float]] = defaultdict(list)
    all_rows: List[pd.DataFrame] = []
    for path in csv_paths:
        log(f"Processing tabular file: {path}")
        df = pd.read_csv(path)
        if TARGET_COL not in df.columns:
            log(f"  [skip] missing target column '{TARGET_COL}'")
            continue
        df["_batch"] = path.parent.name
        tmp = df.copy()
        image_col = None
        for col in df.columns:
            if col.lower() in {
                "image",
                "image_name",
                "filename",
                "file",
                "img",
                "name",
            }:
                image_col = col
        if image_col is None:
            tmp["__image_fallback__"] = np.arange(len(tmp))
            image_col = "__image_fallback__"
        if ID_COL in tmp.columns:
            tmp["_join_key"] = cellid_to_stem(tmp[ID_COL], channel=channel)
        else:
            tmp["_join_key"] = normalize_pathlike_to_stem(tmp[image_col])
        tmp["_join_key"] = tmp["_join_key"].astype(str).str.lower().str.strip()
        tmp["_batch"] = tmp["_batch"].astype(str)
        tmp["_image_group"] = tmp[image_col].astype(str)
        tmp["_source_csv"] = path.as_posix()
        all_rows.append(tmp)
        drop_cols = [c for c in [ID_COL, TARGET_COL, image_col] if c in tmp.columns]
        X = tmp.drop(columns=drop_cols, errors="ignore")
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            log("  [skip] no numeric features after preprocessing")
            continue
        X_num = tmp[num_cols]
        y = tmp[TARGET_COL].astype(int).values
        groups = tmp[image_col].astype(str).values if image_col in tmp.columns else None
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos == 0 or neg == 0:
            log("  [warn] target has only one class, skipping CV/importance")
            continue
        spw = neg / pos if pos > 0 else 1.0
        cv, cv_kwargs, cv_mode = pick_cv(y, groups=groups, max_splits=max_splits)
        log_model = make_logreg()
        xgb_model = make_xgb(scale_pos_weight=spw)
        acc_log = cross_val_score(
            log_model, X_num, y, cv=cv, scoring="accuracy", **cv_kwargs
        )
        auc_log = cross_val_score(
            log_model, X_num, y, cv=cv, scoring="roc_auc", **cv_kwargs
        )
        acc_xgb = cross_val_score(
            xgb_model, X_num, y, cv=cv, scoring="accuracy", **cv_kwargs
        )
        auc_xgb = cross_val_score(
            xgb_model, X_num, y, cv=cv, scoring="roc_auc", **cv_kwargs
        )
        log(
            f"  {path.name} ({cv_mode}) LogReg Acc={acc_log.mean():.3f}±{acc_log.std():.3f} AUROC={auc_log.mean():.3f}±{auc_log.std():.3f}"
        )
        log(
            f"  {path.name} ({cv_mode}) XGB    Acc={acc_xgb.mean():.3f}±{acc_xgb.std():.3f} AUROC={auc_xgb.mean():.3f}±{auc_xgb.std():.3f}"
        )
        try:
            rel_folder = path.parent.relative_to(csv_root)
        except ValueError:
            rel_folder = path.parent
        out_subdir = out_root / rel_folder / path.stem
        out_subdir.mkdir(parents=True, exist_ok=True)
        cv_df = pd.DataFrame(
            {
                "model": ["LogReg", "XGB"],
                "acc_mean": [float(acc_log.mean()), float(acc_xgb.mean())],
                "acc_std": [float(acc_log.std()), float(acc_xgb.std())],
                "auc_mean": [float(auc_log.mean()), float(auc_xgb.mean())],
                "auc_std": [float(auc_log.std()), float(auc_xgb.std())],
                "cv_mode": [cv_mode, cv_mode],
                "group_label": [rel_folder.as_posix(), rel_folder.as_posix()],
            }
        )
        cv_df.to_csv(out_subdir / "cv_scores.csv", index=False)
        xgb_model.fit(X_num, y)
        gain_series = xgb_gain_series(xgb_model, num_cols)
        gain_series.to_csv(out_subdir / "xgb_importance_gain.csv")
        top_gain = "; ".join(
            [f"{feat}:{score:.3f}" for feat, score in gain_series.head(5).items()]
        )
        granks = gain_series.rank(ascending=False, method="average")
        for feat, rank in granks.items():
            if pd.isna(feat) or str(feat).strip() == "" or str(feat).lower() == "nan":
                continue
            gain_rank_bucket[str(feat)].append(float(rank))
        perm_df = None
        top_perm = ""
        if not skip_permutation:
            try:
                log_model.fit(X_num, y)
                perm_df = permutation_importance_df(
                    log_model, X_num, y, num_cols, n_repeats=20
                )
                perm_df.to_csv(
                    out_subdir / "permutation_importance_logreg.csv", index=False
                )
                top_perm = "; ".join(
                    [
                        f"{row.feature}:{row.perm_mean:.3f}"
                        for _, row in perm_df.head(5).iterrows()
                    ]
                )
                tmp_perm = perm_df.sort_values("perm_mean", ascending=False).copy()
                tmp_perm["rank"] = np.arange(1, len(tmp_perm) + 1)
                for _, row in tmp_perm.iterrows():
                    feat = row["feature"]
                    if (
                        pd.isna(feat)
                        or str(feat).strip() == ""
                        or str(feat).lower() == "nan"
                    ):
                        continue
                    perm_rank_bucket[str(feat)].append(float(row["rank"]))
            except Exception as exc:
                log(f"  [warn] permutation importance failed for {path.name}: {exc}")
        summary_rows.append(
            {
                "folder": rel_folder.as_posix(),
                "file": path.name,
                "logreg_acc_mean": float(acc_log.mean()),
                "logreg_auc_mean": float(auc_log.mean()),
                "xgb_acc_mean": float(acc_xgb.mean()),
                "xgb_auc_mean": float(auc_xgb.mean()),
                "top5_xgb_gain": top_gain,
                "top5_perm": top_perm,
            }
        )
    csv_big = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["folder", "xgb_auc_mean"], ascending=[True, False]
        )
        summary_df.to_csv(out_root / "summary_across_files.csv", index=False)
    gain_avg_rank = average_rank_from_bucket(gain_rank_bucket)
    if not gain_avg_rank.empty:
        gain_avg_rank.to_csv(out_root / "global_gain_avg_rank.csv")
    perm_avg_rank = average_rank_from_bucket(perm_rank_bucket)
    if not perm_avg_rank.empty:
        perm_avg_rank.to_csv(out_root / "global_perm_avg_rank.csv")
    log(f"Tabular analysis summary saved to {out_root}")
    return csv_big, summary_df, gain_avg_rank, perm_avg_rank


def merge_with_embeddings(csv_big: pd.DataFrame, emb_df: pd.DataFrame):
    """
    這個函式會將表格資料與影像嵌入合併，並回傳不同特徵矩陣。

    參數:
    csv_big (pd.DataFrame): 整合後的表格資料。
    emb_df (pd.DataFrame): 影像特徵資料表。

    回傳值:
    tuple: 包含表格特徵、影像特徵、組合特徵、後設資訊與分組標籤等內容。
    """
    if csv_big.empty:
        raise RuntimeError("No tabular data available for training.")
    emb_df_key = emb_df.copy()
    emb_df_key["_join_key"] = emb_df_key["_stem"].astype(str).str.lower().str.strip()
    duplicates = emb_df_key.duplicated(subset=["_join_key", "_batch"], keep=False).sum()
    if duplicates:
        log(
            f"[warn] {duplicates} duplicate embedding rows detected for the same (_join_key, _batch)"
        )
    merged = pd.merge(
        csv_big,
        emb_df_key,
        on=["_join_key", "_batch"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise RuntimeError(
            "Merging csv data with embeddings resulted in an empty dataframe."
        )
    y = merged[TARGET_COL].astype(int).values
    emb_cols = [col for col in merged.columns if re.fullmatch(r"\d+", str(col))]
    emb_cols_sorted = sorted(emb_cols, key=lambda c: int(str(c)))
    X_emb = (
        merged[emb_cols_sorted].to_numpy(dtype=np.float32)
        if emb_cols_sorted
        else np.empty((len(merged), 0), dtype=np.float32)
    )
    tab_drop_cols = {
        ID_COL,
        TARGET_COL,
        "_batch",
        "_stem",
        "_join_key",
        "_image_group",
        "_source_csv",
    }
    tab_drop_cols.update(emb_cols_sorted)
    tab_df_cols = [col for col in merged.columns if col not in tab_drop_cols]
    tab_df = merged[tab_df_cols].copy()
    num_cols = tab_df.select_dtypes(include=[np.number]).columns.tolist()
    X_tab = (
        tab_df[num_cols].to_numpy(dtype=np.float32)
        if num_cols
        else np.empty((len(merged), 0), dtype=np.float32)
    )
    if X_tab.size and X_emb.size:
        X_concat = np.hstack([X_tab, X_emb])
    elif X_tab.size:
        X_concat = X_tab.copy()
    else:
        X_concat = X_emb.copy()
    groups_image = merged["_join_key"].astype(str).values
    groups_batch = merged["_batch"].astype(str).values
    feature_meta = {
        "tab_cols_all": [str(c) for c in tab_df_cols],
        "num_cols": [str(c) for c in num_cols],
        "emb_cols": [str(c) for c in emb_cols_sorted],
    }
    return X_tab, X_emb, X_concat, feature_meta, y, groups_image, groups_batch, merged


def compute_class_balance(y: np.ndarray) -> Dict[str, float]:
    """
    這個函式會計算資料中的正負樣本數與建議權重。

    參數:
    y (np.ndarray): 二元分類的標籤陣列。

    回傳值:
    Dict[str, float]: 包含正樣本、負樣本、總數與 scale_pos_weight。
    """
    y_array = np.asarray(y)
    pos = int((y_array == 1).sum())
    neg = int((y_array == 0).sum())
    total = int(y_array.size)
    if pos == 0:
        raise RuntimeError("Training data does not contain positive samples.")
    if neg == 0:
        raise RuntimeError("Training data does not contain negative samples.")
    scale_pos_weight = neg / pos
    return {
        "pos": pos,
        "neg": neg,
        "total": total,
        "scale_pos_weight": float(scale_pos_weight),
    }


def evaluate_feature_sets(
    X_tab: np.ndarray,
    X_emb: np.ndarray,
    X_concat: np.ndarray,
    y: np.ndarray,
    groups_image: np.ndarray,
    groups_batch: np.ndarray,
    max_splits: int,
    skip_cv: bool,
    skip_logo: bool,
    scale_pos_weight: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    這個函式會針對不同特徵組合進行交叉驗證與 Leave-One-Batch-Out 評估。

    參數:
    X_tab (np.ndarray): 純表格特徵矩陣。
    X_emb (np.ndarray): 純影像特徵矩陣。
    X_concat (np.ndarray): 表格與影像組合特徵矩陣。
    y (np.ndarray): 標籤資料。
    groups_image (np.ndarray): 影像分組資訊。
    groups_batch (np.ndarray): 批次分組資訊。
    max_splits (int): 最大交叉驗證分割數。
    skip_cv (bool): 是否略過交叉驗證。
    skip_logo (bool): 是否略過 Leave-One-Batch-Out。
    scale_pos_weight (float): XGBoost 的類別權重。

    回傳值:
    Tuple[pd.DataFrame, pd.DataFrame]: 分別為交叉驗證摘要與 LOBO 結果。
    """
    cv_records: List[Dict[str, float]] = []
    if not skip_cv:
        feature_sets = [
            ("Embeddings only", X_emb),
            ("Tabular only", X_tab),
            ("Concat (Tab+Emb)", X_concat),
        ]
        for name, X in feature_sets:
            if X.size == 0:
                log(f"[CV] {name} skipped (no features)")
                continue
            cv, cv_kwargs, mode = pick_cv(y, groups=groups_image, max_splits=max_splits)
            logreg = make_logreg()
            xgb = make_xgb(scale_pos_weight=scale_pos_weight)
            acc_log = cross_val_score(
                logreg, X, y, cv=cv, scoring="accuracy", **cv_kwargs
            )
            auc_log = cross_val_score(
                logreg, X, y, cv=cv, scoring="roc_auc", **cv_kwargs
            )
            acc_xgb = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy", **cv_kwargs)
            auc_xgb = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc", **cv_kwargs)
            log(
                f"[CV] {name:16s} ({mode}) LogReg Acc={acc_log.mean():.3f}±{acc_log.std():.3f} AUROC={auc_log.mean():.3f}±{auc_log.std():.3f}"
            )
            log(
                f"[CV] {name:16s} ({mode}) XGB    Acc={acc_xgb.mean():.3f}±{acc_xgb.std():.3f} AUROC={auc_xgb.mean():.3f}±{auc_xgb.std():.3f}"
            )
            cv_records.append(
                {
                    "feature_set": name,
                    "model": "logreg",
                    "metric": "accuracy",
                    "mean": float(acc_log.mean()),
                    "std": float(acc_log.std()),
                    "cv_mode": mode,
                }
            )
            cv_records.append(
                {
                    "feature_set": name,
                    "model": "logreg",
                    "metric": "roc_auc",
                    "mean": float(auc_log.mean()),
                    "std": float(auc_log.std()),
                    "cv_mode": mode,
                }
            )
            cv_records.append(
                {
                    "feature_set": name,
                    "model": "xgb",
                    "metric": "accuracy",
                    "mean": float(acc_xgb.mean()),
                    "std": float(acc_xgb.std()),
                    "cv_mode": mode,
                }
            )
            cv_records.append(
                {
                    "feature_set": name,
                    "model": "xgb",
                    "metric": "roc_auc",
                    "mean": float(auc_xgb.mean()),
                    "std": float(auc_xgb.std()),
                    "cv_mode": mode,
                }
            )
    cv_df = pd.DataFrame(cv_records)
    lobo_records: List[Dict[str, float]] = []
    if not skip_logo and groups_batch.size >= 1:
        unique_batches = pd.Series(groups_batch).nunique()
        if unique_batches >= 2:
            feature_sets = [
                ("Embeddings only", X_emb),
                ("Tabular only", X_tab),
                ("Concat (Tab+Emb)", X_concat),
            ]
            for name, X in feature_sets:
                if X.size == 0:
                    log(f"[LOBO] {name} skipped (no features)")
                    continue
                logreg = make_logreg()
                xgb = make_xgb(scale_pos_weight=scale_pos_weight)
                logo = LeaveOneGroupOut()
                for train_idx, test_idx in logo.split(X, y, groups=groups_batch):
                    batch_name = pd.Series(groups_batch[test_idx]).iloc[0]
                    logreg.fit(X[train_idx], y[train_idx])
                    xgb.fit(X[train_idx], y[train_idx])
                    prob_log = logreg.predict_proba(X[test_idx])[:, 1]
                    prob_xgb = xgb.predict_proba(X[test_idx])[:, 1]
                    pred_log = (prob_log >= 0.5).astype(int)
                    pred_xgb = (prob_xgb >= 0.5).astype(int)
                    auc_log = (
                        roc_auc_score(y[test_idx], prob_log)
                        if len(np.unique(y[test_idx])) > 1
                        else np.nan
                    )
                    auc_xgb = (
                        roc_auc_score(y[test_idx], prob_xgb)
                        if len(np.unique(y[test_idx])) > 1
                        else np.nan
                    )
                    acc_log = accuracy_score(y[test_idx], pred_log)
                    acc_xgb = accuracy_score(y[test_idx], pred_xgb)
                    lobo_records.append(
                        {
                            "feature_set": name,
                            "model": "logreg",
                            "test_batch": str(batch_name),
                            "acc": float(acc_log),
                            "roc_auc": (
                                float(auc_log) if not np.isnan(auc_log) else np.nan
                            ),
                        }
                    )
                    lobo_records.append(
                        {
                            "feature_set": name,
                            "model": "xgb",
                            "test_batch": str(batch_name),
                            "acc": float(acc_xgb),
                            "roc_auc": (
                                float(auc_xgb) if not np.isnan(auc_xgb) else np.nan
                            ),
                        }
                    )
    lobo_df = pd.DataFrame(lobo_records)
    return cv_df, lobo_df


def train_final_models(
    X_tab: np.ndarray,
    X_emb: np.ndarray,
    X_concat: np.ndarray,
    y: np.ndarray,
    scale_pos_weight: float,
) -> Dict[str, object]:
    """
    這個函式會以全量資料訓練最終的各式模型。

    參數:
    X_tab (np.ndarray): 純表格特徵。
    X_emb (np.ndarray): 純影像特徵。
    X_concat (np.ndarray): 表格與影像合併特徵。
    y (np.ndarray): 標籤陣列。
    scale_pos_weight (float): XGBoost 的不平衡權重。

    回傳值:
    Dict[str, object]: 模型名稱對應訓練完成的模型物件。
    """
    models: Dict[str, object] = {}
    if X_emb.size:
        models["logreg_emb"] = make_logreg().fit(X_emb, y)
        models["xgb_emb"] = make_xgb(scale_pos_weight=scale_pos_weight).fit(X_emb, y)
    if X_tab.size:
        models["logreg_tab"] = make_logreg().fit(X_tab, y)
        models["xgb_tab"] = make_xgb(scale_pos_weight=scale_pos_weight).fit(X_tab, y)
    if X_concat.size:
        models["logreg_concat"] = make_logreg().fit(X_concat, y)
        models["xgb_concat"] = make_xgb(scale_pos_weight=scale_pos_weight).fit(
            X_concat, y
        )
    return models


def save_models(models: Dict[str, object], model_dir: Path) -> Dict[str, Path]:
    """
    這個函式會將模型序列化並寫入指定目錄。

    參數:
    models (Dict[str, object]): 模型名稱與模型物件的對應字典。
    model_dir (Path): 輸出模型的目錄。

    回傳值:
    Dict[str, Path]: 模型名稱對應的檔案路徑。
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, model in models.items():
        path = model_dir / f"final_{name}.pkl"
        joblib.dump(model, path)
        paths[name] = path
    return paths


def compute_train_scores(
    models: Dict[str, object],
    X_tab: np.ndarray,
    X_emb: np.ndarray,
    X_concat: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    """
    這個函式會計算訓練資料上的準確率與 AUROC。

    參數:
    models (Dict[str, object]): 各個已訓練的模型。
    X_tab (np.ndarray): 純表格特徵矩陣。
    X_emb (np.ndarray): 純影像特徵矩陣。
    X_concat (np.ndarray): 合併特徵矩陣。
    y (np.ndarray): 標籤資料。

    回傳值:
    pd.DataFrame: 各模型的評估指標摘要。
    """
    records: List[Dict[str, float]] = []
    feature_map = {
        "tab": X_tab,
        "emb": X_emb,
        "concat": X_concat,
    }
    for name, model in models.items():
        key = None
        if name.endswith("_tab"):
            key = "tab"
        elif name.endswith("_emb"):
            key = "emb"
        elif name.endswith("_concat"):
            key = "concat"
        if key is None:
            continue
        X = feature_map[key]
        if X.size == 0:
            continue
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, prob)
        log(f"[TRAIN] {name:16s} Acc={acc:.3f} AUROC={auc:.3f}")
        records.append({"model": name, "accuracy": float(acc), "roc_auc": float(auc)})
    return pd.DataFrame(records)


def save_manifest(model_dir: Path, manifest: Dict[str, object]) -> Path:
    """
    這個函式會將訓練設定與產出資訊寫入 manifest 檔案。

    參數:
    model_dir (Path): 模型輸出目錄。
    manifest (Dict[str, object]): 需要儲存的資訊內容。

    回傳值:
    Path: 實際寫入的 manifest 檔案路徑。
    """
    manifest_path = model_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def parse_args() -> argparse.Namespace:
    """
    這個函式會定義並解析訓練腳本所需的命令列參數。

    參數:
    無。

    回傳值:
    argparse.Namespace: 解析後的參數集合。
    """
    parser = argparse.ArgumentParser(description="訓練 Ki67 分類流程。")
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("data/output/results"),
        help="包含 *_cleaned.csv 檔案的根目錄。",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/output/cyto_crops"),
        help="依批次存放影像切塊的根目錄。",
    )
    parser.add_argument(
        "--channel", default="cyto", help="轉換 Cell_ID 為檔名時使用的通道字尾。"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs_models"),
        help="儲存訓練後模型的輸出目錄。",
    )
    parser.add_argument(
        "--tabular-output-root",
        type=Path,
        default=Path("analyze_results"),
        help="輸出表格分析結果的目錄。",
    )
    parser.add_argument(
        "--backbone",
        choices=sorted(SUPPORTED_BACKBONES),
        default="resnet18",
        help="用來抽取影像特徵的 CNN 架構。",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="影像縮放後的邊長像素。"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="抽取特徵時的批次大小。"
    )
    parser.add_argument(
        "--device", default=None, help="PyTorch 使用的裝置，例如 'cuda:0'。"
    )
    parser.add_argument(
        "--max-splits", type=int, default=5, help="交叉驗證允許的最大分割數。"
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="略過計算 permutation importance。",
    )
    parser.add_argument(
        "--skip-cv", action="store_true", help="略過合併資料的交叉驗證。"
    )
    parser.add_argument(
        "--skip-logo", action="store_true", help="略過分批 Leave-One-Batch-Out 評估。"
    )
    parser.add_argument("--timestamp", default=None, help="自訂輸出資料夾的時間戳記。")
    return parser.parse_args()


def main() -> None:
    """
    這個函式是腳本進入點，串接資料處理、模型訓練與輸出流程。

    參數:
    無。

    回傳值:
    None: 此函式不會回傳值。
    """
    args = parse_args()
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", message="No positive samples in y_true")
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    csv_paths = sorted(Path(args.csv_root).rglob("*_cleaned.csv"))
    if not csv_paths:
        log(f"No *_cleaned.csv files found under {args.csv_root}")
        sys.exit(1)
    log(f"Found {len(csv_paths)} cleaned CSV files")
    csv_big, summary_df, gain_avg_rank, perm_avg_rank = analyze_csv_files(
        csv_paths=csv_paths,
        csv_root=args.csv_root,
        channel=args.channel,
        out_root=args.tabular_output_root,
        skip_permutation=args.skip_permutation,
        max_splits=args.max_splits,
    )
    dataset = MultiBatchImageDataset(args.image_root, size=args.image_size)
    if len(dataset) == 0:
        log(f"No images found under {args.image_root}")
        sys.exit(1)
    emb_df, feat_dim = extract_all_embeddings(
        dataset=dataset,
        arch=args.backbone,
        pretrained=True,
        batch_size=args.batch_size,
        device=args.device,
    )
    log(f"Embeddings dataframe shape: {emb_df.shape}")
    X_tab, X_emb, X_concat, feature_meta, y, groups_image, groups_batch, merged = (
        merge_with_embeddings(csv_big, emb_df)
    )
    class_balance = compute_class_balance(y)
    log(
        "Training samples: total={total} pos={pos} neg={neg} scale_pos_weight={spw:.3f}".format(
            total=class_balance["total"],
            pos=class_balance["pos"],
            neg=class_balance["neg"],
            spw=class_balance["scale_pos_weight"],
        )
    )
    cv_df, lobo_df = evaluate_feature_sets(
        X_tab=X_tab,
        X_emb=X_emb,
        X_concat=X_concat,
        y=y,
        groups_image=groups_image,
        groups_batch=groups_batch,
        max_splits=args.max_splits,
        skip_cv=args.skip_cv,
        skip_logo=args.skip_logo,
        scale_pos_weight=class_balance["scale_pos_weight"],
    )
    ts = args.timestamp or time.strftime("%Y%m%d-%H%M%S")
    model_dir = args.output_root / ts
    models = train_final_models(
        X_tab, X_emb, X_concat, y, scale_pos_weight=class_balance["scale_pos_weight"]
    )
    artifact_paths = save_models(models, model_dir)
    train_scores_df = compute_train_scores(models, X_tab, X_emb, X_concat, y)
    if not train_scores_df.empty:
        train_scores_df.to_csv(model_dir / "train_scores.csv", index=False)
    if not cv_df.empty:
        cv_df.to_csv(model_dir / "cv_summary.csv", index=False)
    if not lobo_df.empty:
        lobo_df.to_csv(model_dir / "lobo_summary.csv", index=False)
    manifest = {
        "pipeline": "resnet_embedding_extractor + classical_classifier",
        "resnet": {
            "arch": args.backbone,
            "input_size": args.image_size,
            "pretrained": True,
            "normalize_mean": IMAGE_NORMALIZE_MEAN,
            "normalize_std": IMAGE_NORMALIZE_STD,
            "feat_dim": int(X_emb.shape[1]),
        },
        "data": {
            "channel": args.channel,
            "csv_root": str(Path(args.csv_root).resolve()),
            "image_root": str(Path(args.image_root).resolve()),
        },
        "features": {
            "tab_cols_all": feature_meta["tab_cols_all"],
            "num_cols": feature_meta["num_cols"],
            "emb_cols": feature_meta["emb_cols"],
            "emb_dim": int(X_emb.shape[1]),
            "concat_dim": int(X_concat.shape[1]),
        },
        "class_balance": class_balance,
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "timestamp": ts,
        "tabular_analysis_dir": str(Path(args.tabular_output_root).resolve()),
        "source_csv_files": [str(p.resolve()) for p in csv_paths],
        "merged_rows": int(len(merged)),
        "embedding_feature_dim": int(feat_dim),
    }
    manifest_path = save_manifest(model_dir, manifest)
    log(f"Manifest saved to {manifest_path}")
    log("Artifacts:")
    for name, path in artifact_paths.items():
        log(f"  {name}: {path}")
    log("Training complete.")


if __name__ == "__main__":
    main()
