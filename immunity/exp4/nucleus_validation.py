"""Exp4 E2：既有 PC nucleus 與預先計算 DAPI label masks 的描述性驗證。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from immunity.exp3.phase_features import compare_nucleus_masks, load_cached_masks


PER_FOV_COLUMNS = (
    "image_key",
    "group_id",
    "pc_nucleus_count",
    "dapi_nucleus_count",
    "matched_count",
    "matched_cell_coverage",
    "dice_min",
    "dice_q25",
    "dice_median",
    "dice_q75",
    "dice_max",
    "dice_mean",
    "iou_min",
    "iou_q25",
    "iou_median",
    "iou_q75",
    "iou_max",
    "iou_mean",
    "status",
)
SUMMARY_COLUMNS = (
    "scope",
    "group_id",
    "fov_count",
    "matched_fov_count",
    "pc_nucleus_count",
    "dapi_nucleus_count",
    "matched_pair_count",
    "matched_cell_coverage_mean",
    "dice_min",
    "dice_q25",
    "dice_median",
    "dice_q75",
    "dice_max",
    "dice_mean",
    "iou_min",
    "iou_q25",
    "iou_median",
    "iou_q75",
    "iou_max",
    "iou_mean",
)


class E2ValidationError(ValueError):
    """表示 E2 的完整 FOV scope 或 label-mask contract 不成立。"""


@dataclass(frozen=True)
class NucleusDapiValidationResult:
    """保存 E2 的逐 FOV 與整體／群組描述性統計。

    Attributes:
        per_fov: 每個 manifest FOV 恰一列的 Dice／IoU 摘要。
        summary: matched nucleus pair 為統計單位的 overall 與 group 分布。
    """

    per_fov: pd.DataFrame
    summary: pd.DataFrame


PcMaskLoader = Callable[[Path, str, str], np.ndarray]
DapiMaskLoader = Callable[[Path], np.ndarray]


def load_dapi_label_mask(path: str | Path) -> np.ndarray:
    """唯讀載入預先計算的 DAPI nucleus label cache。

    此介面刻意不接受 raw DAPI 影像，避免 Exp4 隱含啟動 Cellpose 或把灰階
    intensity 當成 instance labels。cache 必須是 ``.npz``，且包含
    ``dapi_nucleus_mask``。

    Args:
        path: 預先計算的 DAPI label cache 路徑。

    Returns:
        二維、非負整數的 DAPI nucleus label mask。

    Raises:
        E2ValidationError: 副檔名、cache key 或 label values 不合法時拋出。
        OSError: cache 無法讀取時拋出。
    """
    cache_path = Path(path)
    if cache_path.suffix.lower() != ".npz":
        raise E2ValidationError(
            "DAPI validation 只接受 precomputed .npz label cache；不接受 raw 影像"
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        if "dapi_nucleus_mask" not in cache:
            raise E2ValidationError(
                f"DAPI label cache 缺少 dapi_nucleus_mask：{cache_path}"
            )
        mask = np.asarray(cache["dapi_nucleus_mask"])
    return _validated_label_mask(mask, source=f"DAPI cache {cache_path}")


def validate_nucleus_dapi(
    manifest: pd.DataFrame,
    *,
    pc_masks_dir: str | Path,
    dapi_mask_paths: Mapping[str, str | Path],
    expected_fov_count: int = 693,
    pc_mask_loader: PcMaskLoader | None = None,
    dapi_mask_loader: DapiMaskLoader = load_dapi_label_mask,
) -> NucleusDapiValidationResult:
    """在完整 manifest scope 比較 PC-derived 與 DAPI nucleus labels。

    函式沿用 Exp3 的 ``compare_nucleus_masks``，因此 matching 為 Hungarian
    assignment，Dice／IoU 定義也與既有流程一致。DAPI 僅透過既有 label cache
    讀取；本函式不匯入或呼叫 Cellpose。任一 manifest key 缺 DAPI mapping、
    任一 mask 無法讀取或 shape 不一致時皆 fail-closed。沒有 overlap 則保留
    描述性 sentinel，不視為 gate failure。

    Args:
        manifest: 必須含唯一 ``image_key`` 與 ``group_id`` 的正式 FOV manifest。
        pc_masks_dir: Exp3 PC-derived mask cache 根目錄。
        dapi_mask_paths: ``image_key`` 到既有 DAPI label cache 的完整 mapping。
        expected_fov_count: 本次必須恰好涵蓋的 FOV 數；正式執行為 693。
        pc_mask_loader: 可測試替換的 PC nucleus loader；省略時唯讀 Exp3 cache。
        dapi_mask_loader: 可測試替換的 DAPI label loader。

    Returns:
        每 FOV 一列與 overall／各 group 分布的不可變結果物件。

    Raises:
        E2ValidationError: scope、mapping、label values 或逐張比較不合法時拋出。
        TypeError: manifest 或 DAPI mapping 型別不合法時拋出。
    """
    normalized_manifest = _validated_manifest(manifest, expected_fov_count)
    normalized_paths = {str(key): Path(value) for key, value in dapi_mask_paths.items()}
    expected_keys = tuple(normalized_manifest["image_key"].tolist())
    missing = sorted(set(expected_keys) - set(normalized_paths))
    if missing:
        raise E2ValidationError(
            "DAPI label mask mapping 缺少 manifest image_key：" + ", ".join(missing)
        )

    load_pc = pc_mask_loader or _load_pc_nucleus_mask
    pc_root = Path(pc_masks_dir)
    per_fov_rows: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []
    for row in normalized_manifest.itertuples(index=False):
        image_key = str(row.image_key)
        group_id = str(row.group_id)
        try:
            pc_mask = _validated_label_mask(
                load_pc(pc_root, image_key, group_id),
                source=f"PC cache {image_key}",
            )
            dapi_mask = _validated_label_mask(
                dapi_mask_loader(normalized_paths[image_key]),
                source=f"DAPI cache {image_key}",
            )
            compared = compare_nucleus_masks(pc_mask, dapi_mask)
        except E2ValidationError:
            raise
        except Exception as error:
            raise E2ValidationError(
                f"E2 mask comparison failed for {image_key}: {error}"
            ) from error

        pc_count = _positive_label_count(pc_mask)
        dapi_count = _positive_label_count(dapi_mask)
        dice = pd.to_numeric(compared["dice"], errors="coerce").to_numpy(float)
        iou = pd.to_numeric(compared["iou"], errors="coerce").to_numpy(float)
        matched = np.isfinite(dice) & np.isfinite(iou)
        matched_count = int(np.count_nonzero(matched))
        coverage_values = pd.to_numeric(
            compared["matched_cell_coverage"], errors="coerce"
        ).to_numpy(float)
        finite_coverage = coverage_values[np.isfinite(coverage_values)]
        coverage = float(finite_coverage[0]) if finite_coverage.size else 0.0

        per_fov_rows.append(
            {
                "image_key": image_key,
                "group_id": group_id,
                "pc_nucleus_count": pc_count,
                "dapi_nucleus_count": dapi_count,
                "matched_count": matched_count,
                "matched_cell_coverage": coverage,
                **_metric_statistics("dice", dice[matched]),
                **_metric_statistics("iou", iou[matched]),
                "status": _fov_status(pc_count, dapi_count, matched_count),
            }
        )
        if matched_count:
            pair_frames.append(
                pd.DataFrame(
                    {
                        "group_id": group_id,
                        "dice": dice[matched],
                        "iou": iou[matched],
                    }
                )
            )

    per_fov = pd.DataFrame(per_fov_rows, columns=PER_FOV_COLUMNS)
    pairs = (
        pd.concat(pair_frames, ignore_index=True)
        if pair_frames
        else pd.DataFrame(columns=("group_id", "dice", "iou"))
    )
    summary = _build_summary(per_fov, pairs)
    return NucleusDapiValidationResult(per_fov=per_fov, summary=summary)


def _validated_manifest(manifest: pd.DataFrame, expected_fov_count: int) -> pd.DataFrame:
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest 必須是 pandas DataFrame")
    required = {"image_key", "group_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise E2ValidationError(f"manifest 缺少欄位：{missing}")
    if isinstance(expected_fov_count, bool) or int(expected_fov_count) <= 0:
        raise E2ValidationError("expected_fov_count 必須是正整數")
    expected = int(expected_fov_count)
    if len(manifest) != expected:
        raise E2ValidationError(
            f"E2 manifest FOV count expected {expected}, actual {len(manifest)}"
        )
    normalized = manifest.loc[:, ["image_key", "group_id"]].copy()
    normalized["image_key"] = normalized["image_key"].astype(str).str.strip()
    normalized["group_id"] = normalized["group_id"].astype(str).str.strip()
    if normalized["image_key"].eq("").any() or normalized["group_id"].eq("").any():
        raise E2ValidationError("manifest image_key/group_id 不可為空")
    if normalized["image_key"].duplicated().any():
        duplicates = sorted(
            normalized.loc[normalized["image_key"].duplicated(False), "image_key"].unique()
        )
        raise E2ValidationError(f"manifest image_key 必須唯一：{duplicates}")
    return normalized.reset_index(drop=True)


def _load_pc_nucleus_mask(root: Path, image_key: str, group_id: str) -> np.ndarray:
    path = root / group_id / f"{image_key}.npz"
    _cell_mask, nucleus_mask = load_cached_masks(path)
    return nucleus_mask


def _validated_label_mask(mask: np.ndarray, *, source: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise E2ValidationError(f"{source} must contain two-dimensional labels")
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise E2ValidationError(
            f"{source} must contain non-negative integer labels"
        ) from error
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0)
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise E2ValidationError(
            f"{source} must contain non-negative integer labels"
        )
    if numeric.max(initial=0.0) > np.iinfo(np.int32).max:
        raise E2ValidationError(f"{source} labels exceed int32 range")
    return numeric.astype(np.int32, copy=False)


def _positive_label_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(mask) > 0))


def _metric_statistics(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_mean": float("nan"),
        }
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_q25": float(np.quantile(finite, 0.25)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_q75": float(np.quantile(finite, 0.75)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
    }


def _fov_status(pc_count: int, dapi_count: int, matched_count: int) -> str:
    if matched_count:
        return "matched"
    if pc_count == 0 and dapi_count == 0:
        return "empty_both"
    if pc_count == 0:
        return "empty_pc"
    if dapi_count == 0:
        return "empty_dapi"
    return "no_overlap"


def _build_summary(per_fov: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str | None]] = [("overall", None)]
    scopes.extend(
        (f"group:{group_id}", str(group_id))
        for group_id in per_fov["group_id"].drop_duplicates().tolist()
    )
    for scope, group_id in scopes:
        fov_rows = (
            per_fov
            if group_id is None
            else per_fov.loc[per_fov["group_id"].eq(group_id)]
        )
        pair_rows = (
            pairs if group_id is None else pairs.loc[pairs["group_id"].eq(group_id)]
        )
        rows.append(
            {
                "scope": scope,
                "group_id": "ALL" if group_id is None else group_id,
                "fov_count": int(len(fov_rows)),
                "matched_fov_count": int(fov_rows["matched_count"].gt(0).sum()),
                "pc_nucleus_count": int(fov_rows["pc_nucleus_count"].sum()),
                "dapi_nucleus_count": int(fov_rows["dapi_nucleus_count"].sum()),
                "matched_pair_count": int(len(pair_rows)),
                "matched_cell_coverage_mean": float(
                    fov_rows["matched_cell_coverage"].mean()
                ),
                **_metric_statistics(
                    "dice", pair_rows["dice"].to_numpy(dtype=float)
                ),
                **_metric_statistics("iou", pair_rows["iou"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


__all__ = [
    "E2ValidationError",
    "NucleusDapiValidationResult",
    "PER_FOV_COLUMNS",
    "SUMMARY_COLUMNS",
    "load_dapi_label_mask",
    "validate_nucleus_dapi",
]
