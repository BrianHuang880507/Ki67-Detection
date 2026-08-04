"""提供 Exp3 可稽核的分組驗證切分與 regression metrics。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


_OUTER_FAMILIES = (
    ("leave_one_b_out", "b_id"),
    ("leave_one_passage_out", "passage"),
    ("leave_one_group_out", "group_id"),
    ("leave_one_condition_out", "condition_index"),
)
_REQUIRED_IMAGE_COLUMNS = (
    "image_key",
    "b_id",
    "passage",
    "group_id",
    "condition_index",
)
_MANIFEST_METADATA_COLUMNS = (
    "b_id",
    "passage",
    "group_id",
    "condition_index",
    "condition",
    "ifn_dose",
    "tnf_dose",
    "fov",
)


@dataclass(frozen=True)
class OuterSplit:
    """保存一個可稽核的 outer train/test split。

    Attributes:
        validation: Outer validation family 名稱。
        fold: 被留出的分組值。
        train_index: 相對於輸入 image frame 的 training positional indices。
        test_index: 相對於輸入 image frame 的 test positional indices。
    """

    validation: str
    fold: str
    train_index: np.ndarray
    test_index: np.ndarray


def make_outer_splits(images: pd.DataFrame) -> list[OuterSplit]:
    """建立四種 deterministic grouped outer split families。

    Args:
        images: 含 image key 與 Exp3 分組 metadata 的 image-level frame。

    Returns:
        依 validation family 與排序後分組值排列的 outer splits。

    Raises:
        ValueError: 輸入為空、必要 metadata 缺失或無法形成安全 split 時拋出。
    """
    _validate_images(images)
    positions = np.arange(len(images), dtype=np.int64)
    splits: list[OuterSplit] = []
    for validation, column in _OUTER_FAMILIES:
        family: list[OuterSplit] = []
        for held_out in _sorted_unique(images[column]):
            test_mask = images[column].eq(held_out).to_numpy(dtype=bool)
            family.append(
                OuterSplit(
                    validation=validation,
                    fold=str(held_out),
                    train_index=positions[~test_mask],
                    test_index=positions[test_mask],
                )
            )
        _validate_split_family(family, len(images), validation)
        splits.extend(family)
    return splits


def outer_split_manifest(
    images: pd.DataFrame,
    splits: Sequence[OuterSplit],
) -> pd.DataFrame:
    """展開 outer splits，並以 positional index 保留每張影像的 metadata。

    Args:
        images: 產生 splits 時使用的原始 image-level frame。
        splits: 欲稽核或保存的 outer splits。

    Returns:
        每個 split、每張影像各一列的 train/test manifest。

    Raises:
        ValueError: Image metadata 不完整，或 split index 越界、重複、重疊、未完整
            覆蓋輸入列時拋出。
    """
    _validate_images(images)
    if not splits:
        raise ValueError("splits 不可為空")

    metadata_columns = [
        column for column in _MANIFEST_METADATA_COLUMNS if column in images.columns
    ]
    positional_metadata = images.iloc[np.arange(len(images))].reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for split in splits:
        train_index = _validated_positions(
            split.train_index, len(images), "train_index"
        )
        test_index = _validated_positions(split.test_index, len(images), "test_index")
        if np.intersect1d(train_index, test_index).size:
            raise ValueError("outer split 的 train_index 與 test_index 不可重疊")
        if train_index.size == 0 or test_index.size == 0:
            raise ValueError("outer split 的 training 與 test 都不可為空")

        roles = np.full(len(images), "", dtype=object)
        roles[train_index] = "train"
        roles[test_index] = "test"
        if np.any(roles == ""):
            raise ValueError("outer split 必須完整覆蓋每個 image row")

        frame = pd.DataFrame(
            {
                "validation": split.validation,
                "fold": split.fold,
                "image_key": positional_metadata["image_key"].to_numpy(),
                "role": roles,
            }
        )
        frame = pd.concat(
            [frame, positional_metadata.loc[:, metadata_columns]], axis=1
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_inner_splits(
    training: pd.DataFrame,
    requested: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """以 outer-training frame 的 ``group_id`` 建立 GroupKFold splits。

    Args:
        training: 單一 outer fold 的 training frame。
        requested: 希望建立的 inner fold 數量。

    Returns:
        相對於 ``training`` 的 train/test positional index pairs。

    Raises:
        ValueError: requested 少於 2、group_id 無效，或 training 少於兩群時拋出。
    """
    if "group_id" not in training.columns:
        raise ValueError("training 缺少必要欄位：['group_id']")
    _validate_nonempty_values(training, ("group_id",))
    if int(requested) != requested or int(requested) < 2:
        raise ValueError("requested 必須至少為 2")

    groups = training["group_id"].to_numpy()
    group_count = int(training["group_id"].nunique(dropna=False))
    if group_count < 2:
        raise ValueError("inner validation 至少需要兩個不同的 group_id")
    splitter = GroupKFold(n_splits=min(int(requested), group_count))
    splits = [
        (
            np.asarray(train_index, dtype=np.int64),
            np.asarray(test_index, dtype=np.int64),
        )
        for train_index, test_index in splitter.split(training, groups=groups)
    ]
    _validate_inner_splits(training, splits)
    return splits


def regression_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    """計算具有明確 undefined behavior 的 regression metrics。

    Args:
        observed: 實際 target values。
        predicted: 預測 target values。

    Returns:
        MAE、RMSE、R2 與 Spearman correlation。樣本少於 2 或 observed
        constant 時 R2 為 NaN；樣本少於 2 或任一 vector constant 時
        Spearman 為 NaN。

    Raises:
        ValueError: Vector 非一維、為空、長度不同或含非有限值時拋出。
    """
    y_true = _metric_vector(observed, "observed")
    y_pred = _metric_vector(predicted, "predicted")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("observed 與 predicted 不可為空")
    if y_true.size != y_pred.size:
        raise ValueError("observed 與 predicted 長度必須相同")
    if not np.isfinite(y_true).all():
        raise ValueError("observed 必須全部為有限值")
    if not np.isfinite(y_pred).all():
        raise ValueError("predicted 必須全部為有限值")

    observed_constant = _is_constant(y_true)
    predicted_constant = _is_constant(y_pred)
    r2 = (
        np.nan
        if y_true.size < 2 or observed_constant
        else float(r2_score(y_true, y_pred))
    )
    spearman = (
        np.nan
        if y_true.size < 2 or observed_constant or predicted_constant
        else float(spearmanr(y_true, y_pred).statistic)
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "spearman": spearman,
    }


def _validate_images(images: pd.DataFrame) -> None:
    """驗證 outer splitting 與 manifest 所需的 image metadata。"""
    missing = sorted(set(_REQUIRED_IMAGE_COLUMNS) - set(images.columns))
    if missing:
        raise ValueError(f"images 缺少必要欄位：{missing}")
    if images.empty:
        raise ValueError("images 不可為空")
    _validate_nonempty_values(images, _REQUIRED_IMAGE_COLUMNS)
    if images["image_key"].duplicated().any():
        raise ValueError("image_key 不可重複")


def _validate_nonempty_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """拒絕分組欄位中的 null 或空白值。"""
    for column in columns:
        values = frame[column]
        blank = values.astype("string").str.strip().eq("").fillna(False)
        if values.isna().any() or blank.any():
            raise ValueError(f"{column} 不可含空值或空白值")


def _sorted_unique(values: pd.Series) -> list[object]:
    """以自然順序回傳不重複分組值。"""
    try:
        return sorted(values.unique().tolist())
    except TypeError as error:
        raise ValueError(f"{values.name} 的分組值必須可排序") from error


def _validate_split_family(
    splits: Sequence[OuterSplit],
    row_count: int,
    validation: str,
) -> None:
    """驗證單一 outer family 的 disjointness 與完整 test coverage。"""
    if not splits:
        raise ValueError(f"{validation} 未產生任何 outer split")
    tested: list[np.ndarray] = []
    expected = np.arange(row_count, dtype=np.int64)
    for split in splits:
        train_index = _validated_positions(split.train_index, row_count, "train_index")
        test_index = _validated_positions(split.test_index, row_count, "test_index")
        if train_index.size == 0 or test_index.size == 0:
            raise ValueError(f"{validation} 的 training 與 test 都不可為空")
        if np.intersect1d(train_index, test_index).size:
            raise ValueError(f"{validation} 的 train/test indices 不可重疊")
        covered = np.sort(np.concatenate([train_index, test_index]))
        if not np.array_equal(covered, expected):
            raise ValueError(f"{validation} 的 split 未完整覆蓋全部 image rows")
        tested.append(test_index)
    test_counts = np.bincount(np.concatenate(tested), minlength=row_count)
    if not np.all(test_counts == 1):
        raise ValueError(f"{validation} 中每張影像必須恰為一次 test")


def _validated_positions(
    values: np.ndarray,
    row_count: int,
    name: str,
) -> np.ndarray:
    """驗證一維、唯一且未越界的 positional indices。"""
    positions = np.asarray(values)
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        raise ValueError(f"{name} 必須是一維整數 positional indices")
    positions = positions.astype(np.int64, copy=False)
    if positions.size != np.unique(positions).size:
        raise ValueError(f"{name} 不可含重複 positional index")
    if np.any(positions < 0) or np.any(positions >= row_count):
        raise ValueError(f"{name} 含有越界 positional index")
    return positions


def _validate_inner_splits(
    training: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    """驗證 inner folds 完整覆蓋且沒有 group leakage。"""
    tested: list[np.ndarray] = []
    groups = training["group_id"]
    for train_index, test_index in splits:
        if train_index.size == 0 or test_index.size == 0:
            raise ValueError("inner fold 的 training 與 test 都不可為空")
        if set(groups.iloc[train_index]).intersection(groups.iloc[test_index]):
            raise ValueError("inner GroupKFold 發生 group_id leakage")
        tested.append(test_index)
    tested_positions = np.sort(np.concatenate(tested))
    if not np.array_equal(tested_positions, np.arange(len(training))):
        raise ValueError("inner GroupKFold 未完整且恰一次覆蓋 training rows")


def _metric_vector(values: Sequence[float], name: str) -> np.ndarray:
    """將 metric 輸入正規化為一維 float vector。"""
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是數值 vector") from error
    if vector.ndim != 1:
        raise ValueError(f"{name} 必須是一維 vector")
    return vector


def _is_constant(values: np.ndarray) -> bool:
    """判斷非空 vector 是否全部為相同值。"""
    return bool(np.all(values == values[0]))


__all__ = [
    "OuterSplit",
    "make_inner_splits",
    "make_outer_splits",
    "outer_split_manifest",
    "regression_metrics",
]
