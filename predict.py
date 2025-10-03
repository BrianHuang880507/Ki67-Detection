#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import warnings
from torch.utils.data import DataLoader, Dataset


TARGET_COL = "ki67_positive"
ID_COL = "Cell_ID"
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]
SUPPORTED_BACKBONES = {"resnet18", "resnet50"}


def log(message: str) -> None:
    """
    這個函式會在終端輸出附帶時間戳記的訊息。

    參數:
    message (str): 要顯示的訊息內容。

    回傳值:
    None: 此函式不會回傳值。
    """
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def build_backbone(arch: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """
    這個函式會依指定架構建立 ResNet 主幹並回傳其特徵維度。

    參數:
    arch (str): ResNet 架構名稱 (resnet18 或 resnet50)。
    pretrained (bool): 是否載入預訓練權重。

    回傳值:
    tuple: 包含修改後的模型以及輸出特徵維度。
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


class InferenceImageDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]], size: int):
        """
        這個方法會初始化推論用資料集並建立轉換流程。

        參數:
        records (Sequence[Dict[str, object]]): 影像路徑與批次資訊的列表。
        size (int): 影像縮放後的邊長。

        回傳值:
        None: 建構子不會回傳值。
        """
        self.records = list(records)
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
        int: 可迭代的影像筆數。
        """
        return len(self.records)

    def __getitem__(self, idx: int):
        """
        這個方法會讀取指定索引的影像並回傳張量與關聯資訊。

        參數:
        idx (int): 目標元素的索引。

        回傳值:
        tuple: 包含影像張量、join key 與批次名稱。
        """
        record = self.records[idx]
        path: Path = record["path"]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, record["join_key"], record["batch"]


@torch.no_grad()
def extract_embeddings(
    records: Sequence[Dict[str, object]],
    arch: str,
    pretrained: bool,
    batch_size: int,
    device: Optional[str],
    image_size: int,
) -> Tuple[pd.DataFrame, int]:
    """
    這個函式會批次將影像轉換為特徵向量並回傳 DataFrame。

    參數:
    records (Sequence[Dict[str, object]]): 影像路徑與識別資訊。
    arch (str): ResNet 架構名稱。
    pretrained (bool): 是否使用預訓練權重。
    batch_size (int): DataLoader 的批次大小。
    device (Optional[str]): 指定裝置字串，預設自動偵測。
    image_size (int): 影像縮放尺寸。

    回傳值:
    tuple: 包含特徵 DataFrame 與特徵維度。
    """
    dataset = InferenceImageDataset(records, image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    total = len(dataset)
    if total == 0:
        return pd.DataFrame(columns=["_join_key", "_batch"]), 0
    if device:
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, feat_dim = build_backbone(arch, pretrained)
    backbone.to(torch_device).eval()
    features: List[np.ndarray] = []
    join_keys: List[str] = []
    batches: List[str] = []
    for batch_idx, (images, join_batch, batch_names) in enumerate(dataloader):
        images = images.to(torch_device, non_blocking=True)
        outputs = backbone(images).cpu().numpy()
        features.append(outputs)
        join_keys.extend([str(j).lower() for j in join_batch])
        batches.extend([str(b) for b in batch_names])
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            log(f"  extracted embeddings for {len(join_keys)}/{total} images")
    array = np.concatenate(features, axis=0).astype(np.float32)
    emb_df = pd.DataFrame(array, columns=[str(i) for i in range(array.shape[1])])
    emb_df["_join_key"] = join_keys
    emb_df["_batch"] = batches
    return emb_df, feat_dim


def normalize_pathlike_to_stem(series: pd.Series) -> pd.Series:
    """
    這個函式會把路徑字串轉換成小寫且移除副檔名的檔名。

    參數:
    series (pd.Series): 含有影像路徑或檔名的序列。

    回傳值:
    pd.Series: 標準化後的檔名 stem 序列。
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


def find_image_column(df: pd.DataFrame) -> Optional[str]:
    """
    這個函式會嘗試在資料表中尋找影像檔名欄位。

    參數:
    df (pd.DataFrame): 原始輸入資料表。

    回傳值:
    Optional[str]: 找到的欄位名稱，若無則回傳 None。
    """
    for col in df.columns:
        if col.lower() in {"image", "image_name", "filename", "file", "img", "name"}:
            return col
    return None


def load_csv_inputs(csv_paths: Sequence[Path], channel: str) -> pd.DataFrame:
    """
    這個函式會讀取多個 CSV 並附加 join key、批次與索引資訊。

    參數:
    csv_paths (Sequence[Path]): 需要讀取的 CSV 路徑列表。
    channel (str): 生成 join key 時使用的通道名稱。

    回傳值:
    pd.DataFrame: 合併後的整體資料表。
    """
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(path)
        df["_batch"] = path.parent.name
        df["_source_csv"] = path.as_posix()
        if ID_COL in df.columns:
            df["_join_key"] = cellid_to_stem(df[ID_COL], channel=channel)
        else:
            image_col = find_image_column(df)
            if image_col is None:
                raise ValueError(
                    f"Cannot infer image column for {path}; provide 'Cell_ID' or an image filename column."
                )
            df["_join_key"] = normalize_pathlike_to_stem(df[image_col])
        df["_join_key"] = df["_join_key"].astype(str).str.lower().str.strip()
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSV data loaded.")
    data_df = pd.concat(frames, ignore_index=True)
    data_df["_orig_index"] = np.arange(len(data_df))
    return data_df


def build_image_index(
    image_root: Path, batches: Iterable[str]
) -> Dict[Tuple[str, str], Path]:
    """
    這個函式會建立 (批次, 檔名) 對應實體路徑的索引。

    參數:
    image_root (Path): 影像資料的根目錄。
    batches (Iterable[str]): 需要搜尋的批次名稱。

    回傳值:
    Dict[Tuple[str, str], Path]: 查詢時使用的索引字典。
    """
    index: Dict[Tuple[str, str], Path] = {}
    for batch in sorted({str(b) for b in batches}):
        batch_dir = image_root / batch
        if not batch_dir.exists():
            log(f"[warn] image batch directory not found: {batch_dir}")
            continue
        for path in batch_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            stem = path.stem.lower()
            if stem.endswith("_overlay"):
                continue
            index[(batch, stem)] = path
    return index


def prepare_image_records(
    df: pd.DataFrame, image_root: Path
) -> Tuple[List[Dict[str, object]], List[Tuple[str, str]]]:
    """
    這個函式會根據資料表準備影像讀取所需的紀錄並回報缺漏。

    參數:
    df (pd.DataFrame): 帶有 `_batch` 與 `_join_key` 欄位的資料表。
    image_root (Path): 影像根目錄。

    回傳值:
    tuple: 可成功對應的紀錄列表與缺漏的批次/檔名組合。
    """
    batches = df["_batch"].astype(str).tolist()
    join_keys = df["_join_key"].astype(str).tolist()
    index = build_image_index(image_root, batches)
    required_pairs = {(b, j) for b, j in zip(batches, join_keys)}
    missing = [pair for pair in sorted(required_pairs) if pair not in index]
    records = [
        {"path": index[pair], "batch": pair[0], "join_key": pair[1]}
        for pair in required_pairs
        if pair in index
    ]
    return records, missing


def load_manifest(model_dir: Path) -> Dict[str, object]:
    """
    這個函式會讀取模型輸出目錄中的 manifest 檔案。

    參數:
    model_dir (Path): 模型與 manifest 所在的目錄。

    回傳值:
    Dict[str, object]: 解析後的 manifest 內容。
    """
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(model_dir: Path, artifact_path: str) -> Path:
    """
    這個函式會將 manifest 中的模型路徑解析成實體路徑。

    參數:
    model_dir (Path): 模型輸出目錄。
    artifact_path (str): manifest 中記錄的相對或絕對路徑。

    回傳值:
    Path: 解析後的檔案路徑。
    """
    path = Path(artifact_path)
    if path.is_absolute():
        return path

    if path.exists():
        return path

    candidate = model_dir / path
    if candidate.exists():
        return candidate

    return candidate


def prepare_feature_matrices(
    merged: pd.DataFrame, num_cols: Sequence[str], emb_cols: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    這個函式會依欄位清單建立表格、影像與組合特徵矩陣。

    參數:
    merged (pd.DataFrame): 已完成合併的資料表。
    num_cols (Sequence[str]): 表格數值欄位名稱。
    emb_cols (Sequence[str]): 影像特徵欄位名稱。

    回傳值:
    tuple: 依序為表格特徵、影像特徵與組合特徵矩陣。
    """
    X_tab = np.empty((len(merged), 0), dtype=np.float32)
    X_emb = np.empty((len(merged), 0), dtype=np.float32)
    if num_cols:
        tab_df = merged.loc[:, list(num_cols)].apply(pd.to_numeric, errors="coerce")
        X_tab = tab_df.to_numpy(dtype=np.float32)
    if emb_cols:
        emb_df = merged.loc[:, list(emb_cols)].apply(pd.to_numeric, errors="coerce")
        X_emb = emb_df.to_numpy(dtype=np.float32)
    if X_tab.size and X_emb.size:
        X_concat = np.hstack([X_tab, X_emb])
    elif X_tab.size:
        X_concat = X_tab.copy()
    else:
        X_concat = X_emb.copy()
    return X_tab, X_emb, X_concat


def select_feature_matrix(
    model_key: str, matrices: Dict[str, np.ndarray]
) -> Tuple[str, np.ndarray]:
    """
    這個函式會根據模型鍵選擇合適的特徵矩陣。

    參數:
    model_key (str): manifest 中的模型代稱。
    matrices (Dict[str, np.ndarray]): 預先準備好的特徵矩陣字典。

    回傳值:
    tuple: 指示使用的特徵類型與實際矩陣。
    """
    if model_key.endswith("_tab"):
        return "tab", matrices["tab"]
    if model_key.endswith("_emb"):
        return "emb", matrices["emb"]
    if model_key.endswith("_concat"):
        return "concat", matrices["concat"]
    raise ValueError(f"Unrecognized model key: {model_key}")


def run_inference(args: argparse.Namespace) -> Path:
    """
    這個函式會依命令列參數載入模型、整理資料並輸出預測結果。

    參數:
    args (argparse.Namespace): 使用者輸入的參數集合。

    回傳值:
    Path: 預測結果 CSV 的輸出路徑。
    """
    model_dir = args.model_dir.resolve()
    manifest = load_manifest(model_dir)
    artifacts = manifest.get("artifacts", {})
    if args.model_key not in artifacts:
        raise KeyError(
            f"Model key '{args.model_key}' not found in manifest artifacts list: {list(artifacts.keys())}"
        )
    model_path = resolve_model_path(model_dir, artifacts[args.model_key])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    channel = args.channel or manifest.get("data", {}).get("channel", "cyto")
    image_root = args.image_root or Path(manifest.get("data", {}).get("image_root", ""))
    backbone = args.backbone or manifest.get("resnet", {}).get("arch", "resnet18")
    image_size = args.image_size or int(
        manifest.get("resnet", {}).get("input_size", 224)
    )
    pretrained = bool(manifest.get("resnet", {}).get("pretrained", True))
    num_cols = manifest.get("features", {}).get("num_cols", [])
    emb_cols = manifest.get("features", {}).get("emb_cols", [])
    emb_dim = (
        int(manifest.get("features", {}).get("emb_dim", len(emb_cols)))
        if emb_cols
        else int(manifest.get("features", {}).get("emb_dim", 0))
    )

    csv_paths = [Path(p) for p in args.csv]
    log(f"Loading CSV files ({len(csv_paths)}):")
    for p in csv_paths:
        log(f"  {p}")
    data_df = load_csv_inputs(csv_paths, channel=channel)

    for col in num_cols:
        if col not in data_df.columns:
            data_df[col] = np.nan

    requires_emb = args.model_key.endswith("_emb") or args.model_key.endswith("_concat")
    merged = data_df.copy()
    dropped_pairs: List[Tuple[str, str]] = []
    if requires_emb:
        if not image_root:
            raise ValueError("Image root must be provided for embedding-based models.")
        image_root = Path(image_root)
        records, missing = prepare_image_records(data_df, image_root)
        if not records:
            raise RuntimeError("No image records matched the requested CSV rows.")
        if missing:
            log(
                f"[warn] Missing {len(missing)} image(s); corresponding rows will be skipped."
            )
        missing_set = set(missing)
        mask = [
            (row_batch, row_join) not in missing_set
            for row_batch, row_join in zip(data_df["_batch"], data_df["_join_key"])
        ]
        merged = data_df.loc[mask].reset_index(drop=True)
        dropped_pairs = missing
        emb_df, feat_dim = extract_embeddings(
            records,
            arch=backbone,
            pretrained=pretrained,
            batch_size=args.batch_size,
            device=args.device,
            image_size=image_size,
        )
        if emb_df.empty:
            raise RuntimeError("Embedding dataframe is empty after extraction.")
        merged = merged.merge(
            emb_df, on=["_join_key", "_batch"], how="inner", validate="many_to_one"
        )
        if merged.empty:
            raise RuntimeError(
                "No rows left after merging embeddings; check join keys and image availability."
            )
        feature_cols = sorted(
            [c for c in emb_df.columns if str(c).isdigit()], key=lambda x: int(str(x))
        )
        if not feature_cols:
            raise RuntimeError(
                "No embedding feature columns found in extracted embeddings."
            )
        if emb_cols:
            missing_cols = [c for c in emb_cols if c not in feature_cols]
            if missing_cols:
                log(
                    f"[warn] Manifest embedding columns missing in extracted data: {missing_cols}; using extracted order instead."
                )
                emb_cols = feature_cols
            else:
                emb_cols = [c for c in emb_cols if c in feature_cols]
        else:
            emb_cols = feature_cols
        emb_dim = len(emb_cols)
    else:
        emb_cols = []
        emb_dim = 0
        feat_dim = 0

    merged = merged.sort_values("_orig_index").reset_index(drop=True)

    X_tab, X_emb, X_concat = prepare_feature_matrices(merged, num_cols, emb_cols)
    matrices = {"tab": X_tab, "emb": X_emb, "concat": X_concat}
    feature_type, feature_matrix = select_feature_matrix(args.model_key, matrices)
    if feature_matrix.size == 0:
        raise RuntimeError(
            f"Feature matrix for '{feature_type}' is empty; check input data and manifest."
        )

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model '{args.model_key}' does not support predict_proba()."
        )
    probs = model.predict_proba(feature_matrix)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    result_df = merged.drop(columns=[c for c in emb_cols], errors="ignore")
    result_df["probability"] = probs
    result_df["prediction"] = preds
    result_df["model_key"] = args.model_key
    result_df = result_df.drop(columns=["_orig_index"], errors="ignore")

    output_path = args.output
    if output_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = model_dir / f"predictions_{args.model_key}_{ts}.csv"
    else:
        output_path = Path(output_path)
    result_df.to_csv(output_path, index=False)

    log(f"Wrote predictions to {output_path}")
    log(f"Rows scored: {len(result_df)}")
    log(f"Positive predictions: {int(preds.sum())}")
    if dropped_pairs:
        log(f"Rows skipped due to missing images: {len(dropped_pairs)}")
    return output_path


def parse_args() -> argparse.Namespace:
    """
    這個函式會定義並解析推論腳本的命令列參數。

    參數:
    無。

    回傳值:
    argparse.Namespace: 解析後的參數集合。
    """
    parser = argparse.ArgumentParser(description="使用訓練好的 Ki67 模型進行推論。")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="包含 manifest.json 與模型檔案的目錄。",
    )
    parser.add_argument(
        "--model-key",
        default="xgb_concat",
        help="要使用的模型鍵，例如 xgb_concat 或 logreg_tab。",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        help="一個或多個待預測的 *_cleaned.csv 檔案。",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="覆寫影像根目錄，未提供則使用 manifest 設定。",
    )
    parser.add_argument(
        "--channel", default=None, help="覆寫 join key 使用的通道字尾。"
    )
    parser.add_argument("--backbone", default=None, help="覆寫推論時的 CNN 架構設定。")
    parser.add_argument(
        "--image-size", type=int, default=None, help="覆寫影像縮放尺寸。"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="抽取影像特徵時的批次大小。"
    )
    parser.add_argument("--device", default=None, help="指定 PyTorch 推論裝置。")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="機率大於此值視為陽性。"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="指定預測結果輸出的 CSV 路徑。"
    )
    return parser.parse_args()


def main() -> None:
    """
    這個函式是推論腳本的進入點，負責啟動整體流程。

    參數:
    無。

    回傳值:
    None: 此函式不會回傳值。
    """
    args = parse_args()
    warnings.filterwarnings("ignore", message="Palette images with transparency")
    output_path = run_inference(args)
    log(f"Done. Predictions stored at {output_path}")


if __name__ == "__main__":
    main()
