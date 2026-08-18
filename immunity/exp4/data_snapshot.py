"""Exp4 輸入資料快照的唯讀、fail-closed 驗證介面。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


DEFAULT_GROUP_KEYS = tuple(
    f"{donor}_P{passage}"
    for donor in ("B4", "B7", "B8")
    for passage in (5, 6, 7)
)


@dataclass(frozen=True)
class SnapshotExpectation:
    """描述 snapshot gate 的明確 manifest/cell/group 期待值。

    `cell_row_count` 是 `cell_level_basic.csv` 的資料列數，
    `cell_image_key_count` 則是其中相異的 `image_key` 數。mask `.npz` 總數
    與不在 manifest 的 mask stem 都是 observation，不會因數量不同而阻擋。
    """

    manifest_image_key_count: int = 693
    cell_row_count: int = 23976
    cell_image_key_count: int = 693
    group_count: int = 9
    group_keys: tuple[str, ...] = DEFAULT_GROUP_KEYS

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SnapshotExpectation":
        """從 YAML mapping 建立快照期待值。

        Args:
            values: 可選的設定 mapping；缺少欄位時使用 Exp4 鎖定值。

        Returns:
            正規化後的快照期待值。

        Raises:
            ValueError: 當數量不是非負整數，或群組 keys 不是序列時。
        """
        if values is not None and not isinstance(values, Mapping):
            raise ValueError("snapshot expected 必須是 mapping")
        values = values or {}
        defaults = cls()
        counts: dict[str, int] = {}
        for name in (
            "manifest_image_key_count",
            "cell_row_count",
            "cell_image_key_count",
            "group_count",
        ):
            try:
                counts[name] = int(values.get(name, getattr(defaults, name)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"snapshot expected {name} 必須是整數") from error
            if counts[name] < 0:
                raise ValueError(f"snapshot expected {name} 不可為負數")

        raw_keys = values.get("group_keys", defaults.group_keys)
        if isinstance(raw_keys, (str, bytes)):
            raise ValueError("snapshot expected group_keys 必須是序列")
        try:
            keys = tuple(str(key) for key in raw_keys)
        except TypeError as error:
            raise ValueError("snapshot expected group_keys 必須是序列") from error
        return cls(group_keys=keys, **counts)

    def to_dict(self) -> dict[str, Any]:
        """轉成可 JSON 序列化的期待值 mapping。"""
        return {
            "manifest_image_key_count": self.manifest_image_key_count,
            "cell_row_count": self.cell_row_count,
            "cell_image_key_count": self.cell_image_key_count,
            "group_count": self.group_count,
            "group_keys": list(self.group_keys),
        }


@dataclass(frozen=True)
class SnapshotResult:
    """保存 snapshot gate、觀察值、差異與唯讀來源。"""

    expected: Mapping[str, Any]
    actual: Mapping[str, Any]
    matched: bool
    differences: list[str]
    observations: list[str]
    sources: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """轉成 CLI 可輸出的 JSON mapping。"""
        return {
            "expected": dict(self.expected),
            "actual": dict(self.actual),
            "matched": self.matched,
            "differences": list(self.differences),
            "observations": list(self.observations),
            "sources": dict(self.sources),
        }


def validate_data_snapshot(
    *,
    manifest_path: str | Path,
    cell_level_path: str | Path,
    masks_dir: str | Path,
    expected: SnapshotExpectation | Mapping[str, Any] | None = None,
    source_root: str | Path | None = None,
) -> SnapshotResult:
    """唯讀驗證 Exp4 manifest、cell cache 與 mask cache 的 key-level snapshot。

    Args:
        manifest_path: Exp3 `data_manifest.csv` 的來源路徑。
        cell_level_path: Exp3 `cell_level_basic.csv` 的來源路徑。
        masks_dir: Exp3 mask cache 根目錄；只掃描遞迴找到的 `.npz` 檔案。
        expected: 預期 manifest FOV key 數與 cell distinct key 數。
        source_root: 明確的 Exp3 source root，僅記錄於結果，不會寫入該目錄。

    Returns:
        包含 gate expected/actual、observations、differences 與絕對來源路徑的結果。

    Note:
        mask `.npz` 總數與不在 manifest 的 mask stem 是 QC observation，不會改變
        `matched`；manifest 缺少對應 mask 則是 fail-closed gate。
    """
    expectation = _coerce_expectation(expected)
    manifest = _read_csv(Path(manifest_path))
    cells = _read_csv(Path(cell_level_path))
    mask_files = _list_mask_files(Path(masks_dir))

    actual, differences, observations = _compute_actual(manifest, cells, mask_files)
    expected_dict = expectation.to_dict()
    _compare_gate(expected_dict, actual, differences)

    sources = {
        "manifest_path": str(Path(manifest_path).expanduser().resolve(strict=False)),
        "cell_level_path": str(
            Path(cell_level_path).expanduser().resolve(strict=False)
        ),
        "masks_dir": str(Path(masks_dir).expanduser().resolve(strict=False)),
    }
    if source_root is not None:
        sources["source_root"] = str(
            Path(source_root).expanduser().resolve(strict=False)
        )

    return SnapshotResult(
        expected=expected_dict,
        actual=actual,
        matched=not differences,
        differences=differences,
        observations=observations,
        sources=sources,
    )


def _coerce_expectation(
    expected: SnapshotExpectation | Mapping[str, Any] | None,
) -> SnapshotExpectation:
    if expected is None:
        return SnapshotExpectation()
    if isinstance(expected, SnapshotExpectation):
        return expected
    return SnapshotExpectation.from_mapping(expected)


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
        return None


def _list_mask_files(path: Path) -> list[Path]:
    """只列出 mask cache 內副檔名為 `.npz` 的檔案。"""
    if not path.is_dir():
        return []
    return sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == ".npz"
        ),
        key=lambda candidate: candidate.as_posix().lower(),
    )


def _key_values(frame: pd.DataFrame | None) -> tuple[list[str], dict[str, int]]:
    if frame is None or "image_key" not in frame.columns:
        return [], {"null": 0, "empty": 0, "duplicates": 0}
    series = frame["image_key"]
    null_count = int(series.isna().sum())
    values = [str(value).strip() for value in series.dropna()]
    empty_count = sum(value == "" for value in values)
    keys = [value for value in values if value]
    unique_keys = sorted(set(keys))
    return unique_keys, {
        "null": null_count,
        "empty": empty_count,
        "duplicates": len(keys) - len(unique_keys),
    }


def _compute_actual(
    manifest: pd.DataFrame | None,
    cells: pd.DataFrame | None,
    mask_files: list[Path],
) -> tuple[dict[str, Any], list[str], list[str]]:
    differences: list[str] = []
    observations: list[str] = []
    manifest_keys, manifest_stats = _key_values(manifest)
    cell_keys, cell_stats = _key_values(cells)
    manifest_set = set(manifest_keys)
    cell_set = set(cell_keys)
    mask_stems = sorted({path.stem for path in mask_files})
    mask_set = set(mask_stems)
    manifest_only = sorted(manifest_set - cell_set)
    cell_only = sorted(cell_set - manifest_set)
    missing_masks = sorted(manifest_set - mask_set)
    extra_masks = sorted(mask_set - manifest_set)

    actual: dict[str, Any] = {
        "manifest_image_key_count": len(manifest_set),
        "cell_row_count": int(len(cells)) if cells is not None else 0,
        "cell_image_key_count": len(cell_set),
        "manifest_cell_key_sets_equal": not manifest_only and not cell_only,
        "manifest_only_keys": manifest_only,
        "cell_only_keys": cell_only,
        "manifest_missing_mask_count": len(missing_masks),
        "manifest_missing_mask_keys": missing_masks,
        "mask_npz_count": len(mask_files),
        "mask_stem_not_manifest_count": len(extra_masks),
        "mask_stem_not_manifest_keys": extra_masks,
        "mask_extension_distribution": dict(
            sorted(Counter(path.suffix.lower() for path in mask_files).items())
        ),
    }

    required_manifest_columns = {"image_key", "b_id", "passage"}
    if manifest is None:
        differences.append("manifest_path: missing or unreadable CSV")
    else:
        missing_columns = sorted(required_manifest_columns - set(manifest.columns))
        if missing_columns:
            differences.append(f"manifest schema missing columns: {missing_columns}")
        actual["manifest_row_count"] = int(len(manifest))
        actual["manifest_null_image_key_count"] = manifest_stats["null"]
        actual["manifest_empty_image_key_count"] = manifest_stats["empty"]
        actual["manifest_duplicate_image_key_count"] = manifest_stats["duplicates"]
        if "image_key" in manifest.columns:
            if manifest_stats["null"] or manifest_stats["empty"]:
                differences.append(
                    "manifest image_key contains null/empty values: "
                    f"null={manifest_stats['null']}, empty={manifest_stats['empty']}"
                )
            if manifest_stats["duplicates"]:
                differences.append(
                    "manifest image_key is not unique: "
                    f"{manifest_stats['duplicates']} duplicate rows"
                )

        group_keys: set[str] = set()
        if {"b_id", "passage"}.issubset(manifest.columns):
            for row in manifest[["b_id", "passage"]].itertuples(index=False):
                if pd.isna(row.b_id) or pd.isna(row.passage):
                    continue
                try:
                    passage = int(row.passage)
                except (TypeError, ValueError):
                    continue
                group_keys.add(f"{str(row.b_id).strip().upper()}_P{passage}")
        actual["group_keys"] = sorted(group_keys)
        actual["group_count"] = len(group_keys)

    if cells is None:
        differences.append("cell_level_path: missing or unreadable CSV")
    else:
        if "image_key" not in cells.columns:
            differences.append("cell schema missing column: ['image_key']")
        actual["cell_null_image_key_count"] = cell_stats["null"]
        actual["cell_empty_image_key_count"] = cell_stats["empty"]
        if cell_stats["null"] or cell_stats["empty"]:
            differences.append(
                "cell image_key contains null/empty values: "
                f"null={cell_stats['null']}, empty={cell_stats['empty']}"
            )

    if manifest_only or cell_only:
        differences.append(
            "key_sets_equal: expected True, actual False; "
            f"manifest_only_keys={manifest_only}, cell_only_keys={cell_only}"
        )
    if missing_masks:
        differences.append(
            "manifest_missing_mask_count: expected 0, "
            f"actual {len(missing_masks)}; keys={missing_masks}"
        )

    observations.append(f"cell_row_count: observed {actual['cell_row_count']}")
    observations.append(f"group_count: observed {actual.get('group_count', 0)}")
    observations.append(f"mask_npz_count: observed {len(mask_files)} (not a gate)")
    observations.append(
        "mask_stem_not_manifest_count: "
        f"observed {len(extra_masks)} (not a gate)"
    )
    return actual, differences, observations


def _compare_gate(
    expected: Mapping[str, Any], actual: Mapping[str, Any], differences: list[str]
) -> None:
    for key in (
        "manifest_image_key_count",
        "cell_row_count",
        "cell_image_key_count",
        "group_count",
    ):
        if actual.get(key) != expected[key]:
            differences.append(
                f"{key}: expected {expected[key]}, actual {actual.get(key)}"
            )
    expected_groups = sorted(str(key) for key in expected.get("group_keys", ()))
    actual_groups = sorted(str(key) for key in actual.get("group_keys", ()))
    if actual_groups != expected_groups:
        differences.append(
            "group_keys: expected "
            f"{expected_groups}, actual {actual_groups}"
        )
    if actual.get("manifest_missing_mask_count") != 0 and not any(
        difference.startswith("manifest_missing_mask_count:")
        for difference in differences
    ):
        differences.append(
            "manifest_missing_mask_count: expected 0, "
            f"actual {actual.get('manifest_missing_mask_count')}"
        )
