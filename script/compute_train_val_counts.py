#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


TARGET_COL = "ki67_positive"
ID_COL = "Cell_ID"


def log(msg: str) -> None:
    print(msg, flush=True)


def find_image_col(cols: List[str]) -> Optional[str]:
    lc = [c.lower() for c in cols]
    for name in ["image", "image_name", "filename", "file", "img", "name"]:
        if name in lc:
            return cols[lc.index(name)]
    return None


def build_csv_big(csv_root: Path, channel: str) -> pd.DataFrame:
    import train as train_mod  # reuse helpers without running main

    rows: List[pd.DataFrame] = []
    for path in sorted(csv_root.rglob("*_cleaned.csv")):
        df = pd.read_csv(path)
        # Skip if target missing
        if TARGET_COL not in df.columns:
            continue
        tmp = df.copy()
        image_col = find_image_col(list(tmp.columns))
        if image_col is None:
            # no image column, fall back to synthetic key if needed
            tmp["__image_fallback__"] = np.arange(len(tmp))
            image_col = "__image_fallback__"
        if ID_COL in tmp.columns:
            tmp["_join_key"] = train_mod.cellid_to_stem(tmp[ID_COL], channel=channel)
        else:
            tmp["_join_key"] = train_mod.normalize_pathlike_to_stem(tmp[image_col])
        tmp["_join_key"] = tmp["_join_key"].astype(str).str.lower().str.strip()
        tmp["_batch"] = path.parent.name
        tmp["_image_group"] = tmp[image_col].astype(str)
        tmp["_source_csv"] = path.as_posix()
        rows.append(tmp[["_join_key", "_batch", TARGET_COL]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_image_keys(image_root: Path) -> pd.DataFrame:
    # Images are under image_root/<batch>/**/* with valid suffixes
    exts = {".png", ".jpg", ".jpeg"}
    records: List[Tuple[str, str]] = []
    for batch_dir in sorted(image_root.glob("*")):
        if not batch_dir.is_dir():
            continue
        batch_name = batch_dir.name
        for p in batch_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                records.append((p.stem.lower(), batch_name))
    if not records:
        return pd.DataFrame(columns=["_join_key", "_batch"])  
    df = pd.DataFrame(records, columns=["_join_key", "_batch"]).drop_duplicates()
    return df


def main(model_dir: Path) -> None:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found in {model_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_root = Path(manifest["data"]["csv_root"]) if isinstance(manifest["data"]["csv_root"], str) else Path(manifest["data"]["csv_root"])  # type: ignore
    image_root = Path(manifest["data"]["image_root"]) if isinstance(manifest["data"]["image_root"], str) else Path(manifest["data"]["image_root"])  # type: ignore
    channel = str(manifest["data"].get("channel", "cyto"))

    log(f"csv_root={csv_root}")
    log(f"image_root={image_root}")
    log(f"channel={channel}")

    csv_big = build_csv_big(csv_root, channel)
    if csv_big.empty:
        raise SystemExit("No CSV rows with target found under csv_root")

    img_keys = build_image_keys(image_root)
    if img_keys.empty:
        raise SystemExit("No images found under image_root")

    merged = pd.merge(
        csv_big,
        img_keys,
        on=["_join_key", "_batch"],
        how="inner",
        validate="many_to_one",
    )

    total = len(merged)
    pos = int((merged[TARGET_COL].astype(int) == 1).sum())
    neg = int((merged[TARGET_COL].astype(int) == 0).sum())
    log("")
    log(f"Total training samples (cells): {total} (pos={pos}, neg={neg})")

    # Cross-validation (GroupKFold or StratifiedKFold), replicate pick_cv logic
    import train as train_mod
    y = merged[TARGET_COL].astype(int).values
    groups_image = merged["_join_key"].astype(str).values
    cv, cv_kwargs, mode = train_mod.pick_cv(y, groups=groups_image, max_splits=5)

    fold_sizes: List[Tuple[int, int]] = []  # (train_count, val_count)
    for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, **cv_kwargs)):
        fold_sizes.append((len(tr), len(te)))
    if fold_sizes:
        log("")
        log(f"CV mode: {mode} | folds={len(fold_sizes)}")
        for i, (trc, tec) in enumerate(fold_sizes, 1):
            log(f"  Fold {i}: train={trc}, val={tec}")

    # LOBO by batch
    from sklearn.model_selection import LeaveOneGroupOut
    groups_batch = merged["_batch"].astype(str).values
    logo = LeaveOneGroupOut()
    lobo_counts: Dict[str, int] = {}
    for tr, te in logo.split(np.zeros(len(y)), y, groups=groups_batch):
        batch_name = pd.Series(groups_batch[te]).iloc[0]
        lobo_counts[str(batch_name)] = len(te)
    if lobo_counts:
        log("")
        log("LOBO validation sizes per batch:")
        for k, v in sorted(lobo_counts.items()):
            log(f"  {k}: {v}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: compute_train_val_counts.py <model_dir>")
    main(Path(sys.argv[1]))

