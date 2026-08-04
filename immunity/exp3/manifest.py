"""建立 Exp3 P5 至 P7 影像 manifest，並執行 PC/IDO 配對品質檢查。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


P5_PATTERN = re.compile(
    r"^(?P<condition>[1-8])-(?P<channel>phase|IDO)-100X-(?P<fov>\d{1,2})$",
    re.IGNORECASE,
)
P67_PATTERN = re.compile(
    r"^B\d+-P[67]-10X-(?P<condition>[1-8])\.?-"
    r"(?:(?P<ido>IDO)\.?-)?(?P<fov>\d{1,2})$",
    re.IGNORECASE,
)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

MANIFEST_COLUMNS = [
    "b_id",
    "passage",
    "condition_index",
    "fov",
    "group_id",
    "pc_path",
    "ido_path",
]
PAIRING_QC_COLUMNS = [
    "b_id",
    "passage",
    "condition_index",
    "fov",
    "status",
    "pc_count",
    "ido_count",
    "pc_path",
    "ido_path",
    "detail",
]


def parse_image_name(name: str, expected_channel: str) -> tuple[int, int]:
    """解析 Exp3 影像檔名的 condition index 與 FOV。

    Args:
        name: 影像檔名或路徑；僅使用檔名主體進行解析。
        expected_channel: 檔案所在 channel 資料夾所預期的 ``"pc"`` 或 ``"ido"``。

    Returns:
        依序為 condition index 與 FOV 的整數 tuple。

    Raises:
        ValueError: 檔名不符合 P5/P6/P7 規則、channel 不符或 FOV 不在 1 至 10。
    """
    normalized_channel = expected_channel.lower()
    if normalized_channel not in {"pc", "ido"}:
        raise ValueError(f"不支援的 expected channel：{expected_channel}")

    stem = Path(name).stem
    match = P5_PATTERN.fullmatch(stem) or P67_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"無法解析 Exp3 影像檔名：{name}")

    parsed_channel = "ido" if match.groupdict().get("ido") else "pc"
    if match.groupdict().get("channel"):
        parsed_channel = (
            "pc" if match.group("channel").lower() == "phase" else "ido"
        )
    if parsed_channel != normalized_channel:
        raise ValueError(
            f"檔名 channel 與資料夾不符：{name} 預期 {normalized_channel}"
        )

    condition = int(match.group("condition"))
    fov = int(match.group("fov"))
    if not 1 <= fov <= 10:
        raise ValueError(f"FOV 必須介於 1 至 10：{name}")
    return condition, fov


def scan_datasets(
    specs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """唯讀掃描 Exp3 datasets，建立完整 pair manifest 與 pairing QC。

    Args:
        specs: 每個 dataset 的 ``input_dir``、``b_id`` 與 ``passage`` 設定。

    Returns:
        僅包含一對一 PC/IDO pair 的 raw manifest，以及所有配對結果的 QC 表。

    Raises:
        FileNotFoundError: dataset 根目錄或必要的 ``PC``、``IDO`` 子目錄不存在。
        KeyError: dataset 設定缺少必要欄位。
    """
    grouped_paths: dict[tuple[str, int, int, int], dict[str, list[Path]]] = {}
    parse_error_rows: list[dict[str, Any]] = []

    for spec in specs:
        root = Path(spec["input_dir"]).expanduser().resolve(strict=False)
        b_id = str(spec["b_id"])
        passage = int(spec["passage"])
        if not root.is_dir():
            raise FileNotFoundError(f"找不到 Exp3 dataset：{root}")

        for channel, folder_name in (("pc", "PC"), ("ido", "IDO")):
            folder = root / folder_name
            if not folder.is_dir():
                raise FileNotFoundError(f"找不到 {channel.upper()} 資料夾：{folder}")

            for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    continue
                resolved_path = path.resolve(strict=False)
                try:
                    condition_index, fov = parse_image_name(path.name, channel)
                except ValueError as error:
                    parse_error_rows.append(
                        {
                            "b_id": b_id,
                            "passage": passage,
                            "condition_index": None,
                            "fov": None,
                            "status": "parse_error",
                            "pc_count": 0,
                            "ido_count": 0,
                            "pc_path": None,
                            "ido_path": None,
                            "detail": f"{resolved_path}: {error}",
                        }
                    )
                    continue

                key = (b_id, passage, condition_index, fov)
                grouped_paths.setdefault(key, {"pc": [], "ido": []})[channel].append(
                    resolved_path
                )

    manifest_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for (b_id, passage, condition_index, fov), paths in sorted(grouped_paths.items()):
        pc_paths = sorted(paths["pc"], key=lambda item: item.as_posix().lower())
        ido_paths = sorted(paths["ido"], key=lambda item: item.as_posix().lower())
        pc_count = len(pc_paths)
        ido_count = len(ido_paths)
        group_id = f"{b_id}_P{passage}"
        pc_path = _format_paths(pc_paths)
        ido_path = _format_paths(ido_paths)

        if pc_count == ido_count == 1:
            manifest_rows.append(
                {
                    "b_id": b_id,
                    "passage": passage,
                    "condition_index": condition_index,
                    "fov": fov,
                    "group_id": group_id,
                    "pc_path": pc_path,
                    "ido_path": ido_path,
                }
            )
            status = "paired"
            detail = ""
        elif pc_count > 1:
            status = "duplicate_pc"
            detail = "同一 key 有多個 PC 影像"
        elif ido_count > 1:
            status = "duplicate_ido"
            detail = "同一 key 有多個 IDO 影像"
        elif pc_count == 0:
            status = "missing_pc"
            detail = "缺少 PC 影像"
        else:
            status = "missing_ido"
            detail = "缺少 IDO 影像"

        qc_rows.append(
            {
                "b_id": b_id,
                "passage": passage,
                "condition_index": condition_index,
                "fov": fov,
                "status": status,
                "pc_count": pc_count,
                "ido_count": ido_count,
                "pc_path": pc_path,
                "ido_path": ido_path,
                "detail": detail,
            }
        )

    manifest = _sorted_frame(manifest_rows, MANIFEST_COLUMNS, include_status=False)
    qc = _sorted_frame(
        [*qc_rows, *parse_error_rows], PAIRING_QC_COLUMNS, include_status=True
    )
    return manifest, qc


def validate_expected_totals(qc: pd.DataFrame, expected: Mapping[str, int]) -> None:
    """驗證 pairing QC 計數是否符合 Exp3 設定的總數 gate。

    Args:
        qc: ``scan_datasets`` 產生的 pairing QC 表。
        expected: 必須含有 ``pc``、``ido``、``paired`` 的預期總數。

    Raises:
        ValueError: QC 欄位或預期總數設定不完整，或實際值不符合預期。
    """
    required = {"pc", "ido", "paired"}
    if set(expected) != required:
        raise ValueError("expected totals 必須且只能包含 pc、ido、paired")
    missing_columns = {"status", "pc_count", "ido_count"} - set(qc.columns)
    if missing_columns:
        raise ValueError(f"expected totals 無法驗證，QC 缺少欄位：{sorted(missing_columns)}")

    actual = {
        "pc": int(pd.to_numeric(qc["pc_count"], errors="raise").sum()),
        "ido": int(pd.to_numeric(qc["ido_count"], errors="raise").sum()),
        "paired": int(qc["status"].eq("paired").sum()),
    }
    normalized_expected = {name: int(value) for name, value in expected.items()}
    if actual != normalized_expected:
        raise ValueError(
            f"expected totals 不符：預期 {normalized_expected}，實際 {actual}"
        )


def apply_condition_mapping(
    raw_manifest: pd.DataFrame,
    mapping: Mapping[Any, Any],
) -> pd.DataFrame:
    """將嚴格的一至八 condition mapping 加入完整 pair manifest。

    Args:
        raw_manifest: 僅包含完整 PC/IDO pair 的 raw manifest。
        mapping: 以 condition index 為 key，含 ``condition``、``ifn_dose``、
            ``tnf_dose`` 的一至八完整 mapping。

    Returns:
        含 condition 名稱、兩種劑量與穩定 ``image_key`` 的排序 manifest。

    Raises:
        ValueError: mapping 未恰好覆蓋一至八，或名稱與 dose 組合不唯一。
    """
    normalized = _normalize_condition_mapping(mapping)
    rows: list[dict[str, Any]] = []
    for row in raw_manifest.to_dict(orient="records"):
        condition_index = int(row["condition_index"])
        values = normalized[condition_index]
        rows.append(
            {
                **row,
                "condition": str(values["condition"]),
                "ifn_dose": float(values["ifn_dose"]),
                "tnf_dose": float(values["tnf_dose"]),
                "image_key": (
                    f"{row['b_id']}_P{int(row['passage'])}_"
                    f"C{condition_index:02d}_F{int(row['fov']):02d}"
                ),
            }
        )

    columns = [
        *raw_manifest.columns.tolist(),
        *[
            column
            for column in ("condition", "ifn_dose", "tnf_dose", "image_key")
            if column not in raw_manifest.columns
        ],
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["b_id", "passage", "condition_index", "fov"])
        .reset_index(drop=True)
    )


def _format_paths(paths: Sequence[Path]) -> str | None:
    """將同一 key 的零到多個檔案路徑轉為穩定字串。"""
    if not paths:
        return None
    return ";".join(str(path) for path in paths)


def _sorted_frame(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str], include_status: bool
) -> pd.DataFrame:
    """以固定欄位與排序建立可重現的 manifest 或 QC DataFrame。"""
    frame = pd.DataFrame(rows, columns=columns)
    sort_columns = ["b_id", "passage", "condition_index", "fov"]
    if include_status:
        sort_columns.append("status")
    return frame.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def _normalize_condition_mapping(
    mapping: Mapping[Any, Any],
) -> dict[int, Mapping[str, Any]]:
    """驗證並正規化一至八的 condition mapping。"""
    try:
        normalized = {int(key): value for key, value in mapping.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("condition mapping 的 key 必須是 1 至 8") from error
    if len(normalized) != len(mapping) or set(normalized) != set(range(1, 9)):
        raise ValueError("condition mapping 必須完整且唯一地包含 1 至 8")

    required = {"condition", "ifn_dose", "tnf_dose"}
    for index, values in normalized.items():
        if not isinstance(values, Mapping) or not required.issubset(values):
            raise ValueError(
                f"condition mapping {index} 缺少欄位：{sorted(required)}"
            )

    dose_pairs = {
        (float(values["ifn_dose"]), float(values["tnf_dose"]))
        for values in normalized.values()
    }
    labels = {str(values["condition"]) for values in normalized.values()}
    if len(dose_pairs) != 8 or len(labels) != 8:
        raise ValueError("condition mapping 的 condition 與 dose 組合必須各自唯一")
    return normalized
