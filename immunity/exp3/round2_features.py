"""以凍結 roster 擷取隔離的 Exp3 Round 2 Paper-style 93 特徵。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from immunity.exp3.feature_sets import (
    BASIC_GEOMETRY,
    PAPER_STYLE_EXTRA_FEATURES,
    PAPER_STYLE_FOV_FEATURES,
    PRIMARY_CELL_FEATURES,
    PRIMARY_FOV_FEATURES,
    validate_phase_predictors,
)
from immunity.exp3.phase_features import (
    _geometry_values,
    _read_grayscale_image,
    load_cached_masks,
    paper_style_extras,
)
from immunity.exp3.round2_evidence import Round1Evidence


ROSTER_COLUMNS = ("image_key", "cell_label", "nucleus_label")
VALID_COUNT_COLUMNS = [
    "image_key",
    "feature",
    "roster_cell_count",
    "finite_cell_count",
    "required_minimum",
    "status",
]
EXTRACTION_QC_COLUMNS = [
    "image_key",
    "pc_path",
    "mask_path",
    "roster_cell_count",
    "extracted_cell_count",
    "status",
    "reason",
]
FEATURE_QC_COLUMNS = [
    "feature",
    "feature_group",
    "finite_fov_count",
    "unique_finite_count",
    "constant",
    "forbidden_token",
    "registered",
    "status",
]
_FORBIDDEN_PREDICTOR_TOKENS = (
    "ido",
    "ifn",
    "tnf",
    "dose",
    "condition",
    "path",
    "filename",
    "donor",
    "b_id",
    "delta",
)
_EXPECTED_BASIC_ATTR = "_paper93_expected_basic_images"
_EXPECTED_CELLS_ATTR = "_paper93_expected_basic_cells"
_ATOL_ATTR = "_paper93_atol"


@dataclass(frozen=True)
class Paper93Bundle:
    """保存隔離的 93-feature 資料與 fail-closed 品質證據。

    Attributes:
        images: 凍結 metadata、33 個 Round 1 predictors、target 與 60 個新 median。
        paper_cells: 僅含凍結 roster 的重算 cell-level phase features。
        valid_counts: 每張 FOV、每個額外特徵的有限 cell 數量。
        extraction_qc: 每張 FOV 的擷取結果與具路徑的失敗原因。
        feature_qc: 93 個 predictors 的有限性、常數與 leakage 檢查。
        predictor_columns: canonical Paper-style 93 predictor 順序。
    """

    images: pd.DataFrame
    paper_cells: pd.DataFrame
    valid_counts: pd.DataFrame
    extraction_qc: pd.DataFrame
    feature_qc: pd.DataFrame
    predictor_columns: tuple[str, ...]


def extract_locked_paper93(
    evidence: Round1Evidence,
    mask_qc: pd.DataFrame,
    image_keys: Sequence[str] | None = None,
    *,
    min_finite_cells: int = 3,
    atol: float = 1e-12,
) -> Paper93Bundle:
    """只讀 phase 與已驗證 masks，依凍結 roster 擷取 93 個 predictors。

    Args:
        evidence: 已通過 Task 2 驗證的 Round 1 frozen evidence。
        mask_qc: ``validate_frozen_masks`` 產生的逐圖驗證結果。
        image_keys: 選用的 image keys；省略時處理完整凍結 image roster。
        min_finite_cells: 每張 FOV、每個額外特徵要求的有限 cell 數。
        atol: 重算 basic cell features 與 frozen values 的絕對容忍值。

    Returns:
        含資料、valid counts 與 QC 的 immutable bundle contract。

    Raises:
        ValueError: 輸入 schema、image identity、路徑或數值參數不合法時拋出。

    Notes:
        單張影像的讀檔、roster 或 feature failure 會寫入 ``extraction_qc``，
        不會以常數填補；由 ``require_paper93_preflight`` 統一阻擋。
    """
    minimum = _positive_integer(min_finite_cells, "min_finite_cells")
    if minimum < 3:
        raise ValueError("min_finite_cells 至少為 3，不可放寬 frozen Paper93 gate")
    tolerance = float(atol)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("atol 必須是非負有限值")

    _require_columns(evidence.manifest, ("image_key", "group_id", "pc_path"), "manifest")
    _require_columns(
        evidence.basic_cells,
        (*ROSTER_COLUMNS, *PRIMARY_CELL_FEATURES, "IDO_score"),
        "basic_cells",
    )
    _require_columns(
        evidence.basic_images,
        ("image_key", *PRIMARY_FOV_FEATURES, "IDO_score"),
        "basic_images",
    )
    _require_columns(mask_qc, ("image_key", "pc_path", "mask_path", "status"), "mask_qc")

    manifest = _unique_by_image(evidence.manifest, "manifest")
    basic_images = _unique_by_image(evidence.basic_images, "basic_images")
    verified = _unique_by_image(mask_qc, "mask_qc")
    if set(manifest.index) != set(basic_images.index):
        raise ValueError("manifest 與 basic_images image_key identity 不一致")
    keys = _selected_image_keys(list(basic_images.index), image_keys)
    missing_qc = [key for key in keys if key not in verified.index]
    if missing_qc:
        raise ValueError(f"mask_qc 缺少 image_key：{missing_qc}")

    selected_images = basic_images.loc[keys].reset_index(drop=True).copy()
    selected_cells = evidence.basic_cells[
        evidence.basic_cells["image_key"].astype(str).isin(keys)
    ].copy()
    selected_cells["image_key"] = selected_cells["image_key"].astype(str)
    cell_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []

    for image_key in keys:
        manifest_row = manifest.loc[image_key]
        qc_row = verified.loc[image_key]
        pc_path = Path(str(qc_row["pc_path"])).resolve(strict=False)
        mask_path = Path(str(qc_row["mask_path"])).resolve(strict=False)
        roster = selected_cells[selected_cells["image_key"].eq(image_key)].copy()
        qc_result: dict[str, Any] = {
            "image_key": image_key,
            "pc_path": str(pc_path),
            "mask_path": str(mask_path),
            "roster_cell_count": len(roster),
            "extracted_cell_count": 0,
            "status": "failed",
            "reason": "",
        }
        try:
            _require_verified_paths(evidence, manifest_row, qc_row, image_key)
            if str(qc_row["status"]) != "passed":
                raise ValueError(
                    f"mask_qc status={qc_row['status']}; reason={qc_row.get('reason', '')}"
                )
            image_rows = _extract_roster_image(
                image_key,
                pc_path,
                mask_path,
                roster,
                tolerance,
            )
            cell_rows.extend(image_rows)
            qc_result["extracted_cell_count"] = len(image_rows)
            qc_result["status"] = "passed"
            qc_result["reason"] = "verified frozen roster extraction"
        except Exception as error:  # noqa: BLE001 - 每張 FOV 必須保留完整 failure evidence
            qc_result["reason"] = (
                f"image_key={image_key}; pc_path={pc_path}; mask_path={mask_path}; "
                f"{type(error).__name__}: {error}"
            )
        extraction_rows.append(qc_result)

    paper_cells = pd.DataFrame(
        cell_rows,
        columns=[
            *ROSTER_COLUMNS,
            *PRIMARY_CELL_FEATURES,
            *PAPER_STYLE_EXTRA_FEATURES,
            "IDO_score",
        ],
    )
    valid_counts, extra_medians = _aggregate_extras(
        paper_cells,
        keys,
        selected_cells,
        minimum,
    )
    images = selected_images.copy()
    for feature in PAPER_STYLE_EXTRA_FEATURES:
        images[f"{feature}__median"] = [
            extra_medians[(image_key, feature)] for image_key in keys
        ]
    images = images.reset_index(drop=True)

    expected_images = selected_images.loc[
        :, ["image_key", *PRIMARY_FOV_FEATURES, "IDO_score"]
    ].copy()
    expected_images.attrs = {}
    expected_cells = selected_cells.loc[
        :, [*ROSTER_COLUMNS, *PRIMARY_CELL_FEATURES, "IDO_score"]
    ].copy()
    expected_cells.attrs = {}
    images.attrs[_EXPECTED_BASIC_ATTR] = expected_images
    images.attrs[_ATOL_ATTR] = tolerance
    paper_cells.attrs[_EXPECTED_CELLS_ATTR] = expected_cells
    paper_cells.attrs[_ATOL_ATTR] = tolerance

    feature_qc = _build_feature_qc(images, valid_counts)
    return Paper93Bundle(
        images=images,
        paper_cells=paper_cells,
        valid_counts=valid_counts,
        extraction_qc=pd.DataFrame(extraction_rows, columns=EXTRACTION_QC_COLUMNS),
        feature_qc=feature_qc,
        predictor_columns=tuple(PAPER_STYLE_FOV_FEATURES),
    )


def require_paper93_preflight(bundle: Paper93Bundle, *, formal: bool) -> None:
    """驗證 Paper93 bundle 可安全進入 smoke 或 formal benchmark。

    Args:
        bundle: ``extract_locked_paper93`` 產生的隔離資料與 QC。
        formal: ``True`` 時額外要求完整 693 FOV；``False`` 允許非空 subset。

    Raises:
        ValueError: predictor、leakage、roster、凍結值、finite count 或 QC 失敗時拋出。
    """
    predictors = tuple(str(column) for column in bundle.predictor_columns)
    if len(predictors) != 93:
        raise ValueError(f"Paper93 predictor 必須精確為 93 欄，目前為 {len(predictors)}")
    duplicates = sorted({name for name in predictors if predictors.count(name) > 1})
    if duplicates:
        raise ValueError(f"Paper93 predictor 順序不可重複：{duplicates}")
    forbidden = [(name, _forbidden_token(name)) for name in predictors]
    forbidden = [(name, token) for name, token in forbidden if token]
    if forbidden:
        details = ", ".join(f"{name}({token})" for name, token in forbidden)
        raise ValueError(f"Paper93 predictor 含 forbidden token：{details}")
    unregistered = [name for name in predictors if name not in PAPER_STYLE_FOV_FEATURES]
    if unregistered:
        raise ValueError(f"Paper93 predictor 未註冊：{unregistered}")
    if predictors != tuple(PAPER_STYLE_FOV_FEATURES):
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(predictors, PAPER_STYLE_FOV_FEATURES, strict=True)
            )
            if actual != expected
        )
        raise ValueError(
            "Paper93 predictor 必須符合固定 33+60 identity 與 canonical order；"
            f"index={mismatch}, actual={predictors[mismatch]}, "
            f"expected={PAPER_STYLE_FOV_FEATURES[mismatch]}"
        )
    validate_phase_predictors(predictors)

    _require_columns(
        bundle.extraction_qc,
        EXTRACTION_QC_COLUMNS,
        "extraction_qc",
        exact=True,
    )
    failed_extraction = bundle.extraction_qc[
        ~bundle.extraction_qc["status"].astype(str).eq("passed")
    ]
    if not failed_extraction.empty:
        raise ValueError(
            "Paper93 extraction failure："
            + " | ".join(failed_extraction["reason"].astype(str).tolist())
        )

    _validate_extraction_qc(bundle)
    _validate_frozen_images(bundle.images)
    _validate_frozen_cells(bundle.paper_cells)
    _validate_valid_counts(bundle)
    _validate_feature_qc(bundle)

    image_count = len(bundle.images)
    if formal and image_count != 693:
        raise ValueError(f"formal Paper93 必須精確包含 693 FOV，目前為 {image_count}")
    if not formal and image_count == 0:
        raise ValueError("smoke Paper93 subset 不可為空")


def _extract_roster_image(
    image_key: str,
    pc_path: Path,
    mask_path: Path,
    roster: pd.DataFrame,
    atol: float,
) -> list[dict[str, Any]]:
    """擷取一張影像的 frozen roster，不進行配對、排除或重建 roster。"""
    if roster.empty:
        raise ValueError("frozen roster 不可為空")
    if roster.loc[:, list(ROSTER_COLUMNS)].duplicated().any():
        raise ValueError("frozen roster identity 重複")
    if roster["cell_label"].duplicated().any() or roster["nucleus_label"].duplicated().any():
        raise ValueError("frozen roster label mapping 重複")

    phase = _read_grayscale_image(pc_path)
    cell_mask, nucleus_mask = load_cached_masks(mask_path)
    if phase.ndim != 2 or phase.shape != cell_mask.shape or cell_mask.shape != nucleus_mask.shape:
        raise ValueError("phase 與 verified mask arrays 必須是相同 shape 的二維陣列")

    rows: list[dict[str, Any]] = []
    for frozen in roster.to_dict(orient="records"):
        cell_label = _label(frozen["cell_label"], "cell_label")
        nucleus_label = _label(frozen["nucleus_label"], "nucleus_label")
        cell_region = cell_mask == cell_label
        nucleus_region = nucleus_mask == nucleus_label
        if not np.any(cell_region):
            raise ValueError(f"frozen roster missing cell_label={cell_label}")
        if not np.any(nucleus_region):
            raise ValueError(f"frozen roster missing nucleus_label={nucleus_label}")
        if np.any(nucleus_region & ~cell_region):
            raise ValueError(
                f"frozen roster nucleus outside cell_label={cell_label}, nucleus_label={nucleus_label}"
            )
        cytoplasm = cell_region & ~nucleus_region
        if not np.any(cytoplasm):
            raise ValueError(
                f"frozen roster empty cytoplasm cell_label={cell_label}, nucleus_label={nucleus_label}"
            )
        basic = {
            **_geometry_values(phase, cell_region, "cell"),
            **_geometry_values(phase, nucleus_region, "nucleus"),
            "nucleus_cytoplasm_area_ratio": float(
                np.count_nonzero(nucleus_region) / np.count_nonzero(cytoplasm)
            ),
        }
        expected_basic = (
            *(f"cell__{name}" for name in BASIC_GEOMETRY),
            *(f"nucleus__{name}" for name in BASIC_GEOMETRY),
            "nucleus_cytoplasm_area_ratio",
        )
        if tuple(basic) != expected_basic or expected_basic != tuple(PRIMARY_CELL_FEATURES):
            raise ValueError("basic feature schema 偏離固定 33 欄")
        for feature in PRIMARY_CELL_FEATURES:
            actual = float(basic[feature])
            expected = float(frozen[feature])
            if not np.isclose(actual, expected, rtol=0, atol=atol, equal_nan=False):
                raise ValueError(
                    f"basic feature drift cell_label={cell_label}, feature={feature}, "
                    f"expected={expected}, actual={actual}"
                )
        extras = paper_style_extras(phase, cell_region, nucleus_region)
        if tuple(extras) != tuple(PAPER_STYLE_EXTRA_FEATURES):
            missing = sorted(set(PAPER_STYLE_EXTRA_FEATURES) - set(extras))
            unexpected = sorted(set(extras) - set(PAPER_STYLE_EXTRA_FEATURES))
            raise ValueError(
                f"extra feature schema 偏離固定 60 欄；missing={missing}; unexpected={unexpected}"
            )
        rows.append(
            {
                "image_key": image_key,
                "cell_label": cell_label,
                "nucleus_label": nucleus_label,
                **basic,
                **{feature: float(extras[feature]) for feature in PAPER_STYLE_EXTRA_FEATURES},
                "IDO_score": float(frozen["IDO_score"]),
            }
        )
    return rows


def _aggregate_extras(
    paper_cells: pd.DataFrame,
    image_keys: Sequence[str],
    frozen_cells: pd.DataFrame,
    minimum: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    """以有限值計算 60 個額外特徵 medians 與逐圖 valid counts。"""
    rows: list[dict[str, Any]] = []
    medians: dict[tuple[str, str], float] = {}
    for image_key in image_keys:
        extracted = paper_cells[paper_cells["image_key"].astype(str).eq(image_key)]
        roster_count = int(
            frozen_cells["image_key"].astype(str).eq(image_key).sum()
        )
        for feature in PAPER_STYLE_EXTRA_FEATURES:
            values = pd.to_numeric(extracted[feature], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            finite_count = int(finite.size)
            status = "passed" if finite_count >= minimum else "failed"
            rows.append(
                {
                    "image_key": image_key,
                    "feature": feature,
                    "roster_cell_count": roster_count,
                    "finite_cell_count": finite_count,
                    "required_minimum": minimum,
                    "status": status,
                }
            )
            medians[(image_key, feature)] = (
                float(np.median(finite)) if finite_count else np.nan
            )
    return pd.DataFrame(rows, columns=VALID_COUNT_COLUMNS), medians


def _build_feature_qc(images: pd.DataFrame, valid_counts: pd.DataFrame) -> pd.DataFrame:
    """建立固定 93-row feature QC；常數只回報，不視為失敗。"""
    rows: list[dict[str, Any]] = []
    failed_extras = set(
        valid_counts.loc[~valid_counts["status"].eq("passed"), "feature"].astype(str)
    )
    for feature in PAPER_STYLE_FOV_FEATURES:
        values = pd.to_numeric(images[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        finite_count = int(finite.size)
        unique_count = int(np.unique(finite).size)
        token = _forbidden_token(feature)
        registered = feature in PAPER_STYLE_FOV_FEATURES
        cell_feature = feature.removesuffix("__median")
        finite_ok = finite_count == len(images)
        counts_ok = cell_feature not in failed_extras
        rows.append(
            {
                "feature": feature,
                "feature_group": (
                    "basic" if feature in PRIMARY_FOV_FEATURES else "paper_style_extra"
                ),
                "finite_fov_count": finite_count,
                "unique_finite_count": unique_count,
                "constant": bool(unique_count <= 1),
                "forbidden_token": token,
                "registered": bool(registered),
                "status": (
                    "passed" if finite_ok and counts_ok and not token and registered else "failed"
                ),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_QC_COLUMNS)


def _validate_frozen_images(images: pd.DataFrame) -> None:
    """檢查 FOV schema、有限性與 frozen 33 predictors／target tolerance。"""
    _require_columns(
        images,
        ("image_key", "IDO_score", *PAPER_STYLE_FOV_FEATURES),
        "images",
    )
    if images.empty:
        raise ValueError("Paper93 images 不可為空")
    if images["image_key"].astype(str).duplicated().any():
        raise ValueError("Paper93 images image_key 不可重複")
    matrix = images.loc[:, list(PAPER_STYLE_FOV_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        locations = np.argwhere(~np.isfinite(matrix))
        row_index, column_index = locations[0]
        raise ValueError(
            f"Paper93 predictor 非有限：image_key={images.iloc[row_index]['image_key']}; "
            f"feature={PAPER_STYLE_FOV_FEATURES[column_index]}"
        )
    expected = images.attrs.get(_EXPECTED_BASIC_ATTR)
    tolerance = images.attrs.get(_ATOL_ATTR)
    if not isinstance(expected, pd.DataFrame) or tolerance is None:
        raise ValueError("Paper93 images 缺少 frozen basic/target comparison evidence")
    actual = images.loc[:, ["image_key", *PRIMARY_FOV_FEATURES, "IDO_score"]]
    _assert_aligned_frozen_values(actual, expected, float(tolerance), "FOV basic/target")


def _validate_frozen_cells(cells: pd.DataFrame) -> None:
    """檢查 paper cell roster 與 frozen cell basic／target identity。"""
    _require_columns(
        cells,
        (*ROSTER_COLUMNS, *PRIMARY_CELL_FEATURES, *PAPER_STYLE_EXTRA_FEATURES, "IDO_score"),
        "paper_cells",
    )
    expected = cells.attrs.get(_EXPECTED_CELLS_ATTR)
    tolerance = cells.attrs.get(_ATOL_ATTR)
    if not isinstance(expected, pd.DataFrame) or tolerance is None:
        raise ValueError("Paper93 cells 缺少 frozen roster/basic/target comparison evidence")
    actual = cells.loc[:, [*ROSTER_COLUMNS, *PRIMARY_CELL_FEATURES, "IDO_score"]]
    _assert_aligned_frozen_values(actual, expected, float(tolerance), "cell roster/basic/target")


def _assert_aligned_frozen_values(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    atol: float,
    label: str,
) -> None:
    """依 identity 排序後比較 frozen 數值，不接受 NaN 相等。"""
    identity = [column for column in ROSTER_COLUMNS if column in actual.columns]
    if identity == ["image_key"] or "cell_label" not in actual.columns:
        identity = ["image_key"]
    actual_sorted = actual.sort_values(identity, kind="stable").reset_index(drop=True)
    expected_sorted = expected.sort_values(identity, kind="stable").reset_index(drop=True)
    if len(actual_sorted) != len(expected_sorted):
        raise ValueError(
            f"{label} row count 不一致：actual={len(actual_sorted)}, expected={len(expected_sorted)}"
        )
    if not actual_sorted.loc[:, identity].astype(str).equals(
        expected_sorted.loc[:, identity].astype(str)
    ):
        raise ValueError(f"{label} identity 不一致")
    value_columns = [column for column in expected_sorted.columns if column not in identity]
    actual_values = actual_sorted.loc[:, value_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    expected_values = expected_sorted.loc[:, value_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=0,
        atol=atol,
        equal_nan=False,
    ):
        mismatch = np.argwhere(
            ~np.isclose(
                actual_values,
                expected_values,
                rtol=0,
                atol=atol,
                equal_nan=False,
            )
        )[0]
        row_index, column_index = (int(mismatch[0]), int(mismatch[1]))
        raise ValueError(
            f"{label} drift：identity={actual_sorted.loc[row_index, identity].to_dict()}; "
            f"feature={value_columns[column_index]}"
        )


def _validate_valid_counts(bundle: Paper93Bundle) -> None:
    """驗證每張 FOV × 60 extras 的 exact finite-count schema。"""
    _require_columns(bundle.valid_counts, VALID_COUNT_COLUMNS, "valid_counts", exact=True)
    keys = bundle.images["image_key"].astype(str).tolist()
    expected_pairs = {
        (image_key, feature) for image_key in keys for feature in PAPER_STYLE_EXTRA_FEATURES
    }
    actual_pairs = list(
        bundle.valid_counts.loc[:, ["image_key", "feature"]].astype(str).itertuples(
            index=False, name=None
        )
    )
    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError("Paper93 valid_counts image/feature pair 不可重複")
    if set(actual_pairs) != expected_pairs:
        raise ValueError("Paper93 valid_counts 必須精確涵蓋每張 FOV × 60 extras")
    counts = bundle.valid_counts.set_index(["image_key", "feature"], drop=False)
    required_values: set[int] = set()
    for image_key, feature in expected_pairs:
        row = counts.loc[(image_key, feature)]
        required = _positive_integer(row["required_minimum"], "required_minimum")
        required_values.add(required)
        cells = bundle.paper_cells[
            bundle.paper_cells["image_key"].astype(str).eq(image_key)
        ]
        values = pd.to_numeric(cells[feature], errors="coerce").to_numpy(dtype=float)
        expected_roster_count = len(cells)
        expected_finite_count = int(np.isfinite(values).sum())
        try:
            roster_count = int(row["roster_cell_count"])
            finite_count = int(row["finite_cell_count"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Paper93 valid-count QC 非整數：image_key={image_key}, feature={feature}"
            ) from error
        expected_status = "passed" if expected_finite_count >= required else "failed"
        if (
            roster_count != expected_roster_count
            or finite_count != expected_finite_count
            or str(row["status"]) != expected_status
        ):
            raise ValueError(
                f"Paper93 valid-count QC 不一致：image_key={image_key}, feature={feature}, "
                f"roster_cell_count={roster_count}/{expected_roster_count}, "
                f"finite_cell_count={finite_count}/{expected_finite_count}, "
                f"status={row['status']}/{expected_status}"
            )
    if len(required_values) != 1:
        raise ValueError("Paper93 valid-count QC required_minimum 必須全表一致")
    failed = bundle.valid_counts[~bundle.valid_counts["status"].astype(str).eq("passed")]
    if not failed.empty:
        details = "; ".join(
            f"image_key={row.image_key}, feature={row.feature}, "
            f"finite_cell_count={row.finite_cell_count}, required_minimum={row.required_minimum}"
            for row in failed.itertuples(index=False)
        )
        raise ValueError(f"Paper93 finite cell count 不足：{details}")


def _validate_feature_qc(bundle: Paper93Bundle) -> None:
    """驗證 feature QC 與 canonical 93 predictors 一致且全部通過。"""
    _require_columns(bundle.feature_qc, FEATURE_QC_COLUMNS, "feature_qc", exact=True)
    features = bundle.feature_qc["feature"].astype(str).tolist()
    if features != list(PAPER_STYLE_FOV_FEATURES):
        raise ValueError("Paper93 feature_qc 必須是 canonical 93-row order")
    failed = bundle.feature_qc[~bundle.feature_qc["status"].astype(str).eq("passed")]
    if not failed.empty:
        raise ValueError(f"Paper93 feature_qc failed：{failed['feature'].astype(str).tolist()}")
    if not bundle.feature_qc["registered"].astype(bool).all():
        raise ValueError("Paper93 feature_qc 含未註冊 predictor")
    if bundle.feature_qc["forbidden_token"].astype(str).ne("").any():
        raise ValueError("Paper93 feature_qc 含 forbidden predictor")
    expected = _build_feature_qc(bundle.images, bundle.valid_counts)
    try:
        pd.testing.assert_frame_equal(
            bundle.feature_qc.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError(f"Paper93 feature QC 與資料重算結果不一致：{error}") from error


def _validate_extraction_qc(bundle: Paper93Bundle) -> None:
    """驗證 extraction QC identity 與 frozen／實際 roster counts。"""
    qc = bundle.extraction_qc.copy()
    qc["image_key"] = qc["image_key"].astype(str)
    if qc["image_key"].duplicated().any():
        raise ValueError("Paper93 extraction QC image_key 不可重複")
    image_keys = bundle.images["image_key"].astype(str).tolist()
    if set(qc["image_key"]) != set(image_keys):
        raise ValueError("Paper93 extraction QC image_key identity 不一致")
    expected_cells = bundle.paper_cells.attrs.get(_EXPECTED_CELLS_ATTR)
    if not isinstance(expected_cells, pd.DataFrame):
        raise ValueError("Paper93 extraction QC 缺少 frozen roster evidence")
    qc_by_key = qc.set_index("image_key", drop=False)
    for image_key in image_keys:
        row = qc_by_key.loc[image_key]
        expected_roster_count = int(
            expected_cells["image_key"].astype(str).eq(image_key).sum()
        )
        expected_extracted_count = int(
            bundle.paper_cells["image_key"].astype(str).eq(image_key).sum()
        )
        try:
            roster_count = int(row["roster_cell_count"])
            extracted_count = int(row["extracted_cell_count"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Paper93 extraction QC count 非整數：image_key={image_key}") from error
        if (
            roster_count != expected_roster_count
            or extracted_count != expected_extracted_count
        ):
            raise ValueError(
                f"Paper93 extraction QC count 不一致：image_key={image_key}, "
                f"roster_cell_count={roster_count}/{expected_roster_count}, "
                f"extracted_cell_count={extracted_count}/{expected_extracted_count}"
            )


def _require_verified_paths(
    evidence: Round1Evidence,
    manifest_row: pd.Series,
    qc_row: pd.Series,
    image_key: str,
) -> None:
    """確認 extractor 使用 Task 2 已驗證的 phase 與固定 mask 路徑。"""
    pc_path = Path(str(qc_row["pc_path"])).resolve(strict=False)
    manifest_pc = Path(str(manifest_row["pc_path"])).resolve(strict=False)
    group_id = str(manifest_row["group_id"])
    expected_mask = (
        evidence.root / "feature_cache" / "masks" / group_id / f"{image_key}.npz"
    ).resolve(strict=False)
    mask_path = Path(str(qc_row["mask_path"])).resolve(strict=False)
    if pc_path != manifest_pc:
        raise ValueError("mask_qc.pc_path 與 frozen manifest 不一致")
    if mask_path != expected_mask:
        raise ValueError("mask_qc.mask_path 與 frozen cache path 不一致")


def _selected_image_keys(
    available: Sequence[str], image_keys: Sequence[str] | None
) -> list[str]:
    """驗證並回傳不重複、已知的指定 image roster。"""
    keys = list(available) if image_keys is None else [str(key) for key in image_keys]
    if len(keys) != len(set(keys)):
        raise ValueError("image_keys 不可重複")
    unknown = sorted(set(keys) - set(available))
    if unknown:
        raise ValueError(f"image_keys 含未知 image_key：{unknown}")
    return keys


def _unique_by_image(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """建立字串 image_key index 並拒絕 ambiguous rows。"""
    indexed = frame.copy()
    indexed["image_key"] = indexed["image_key"].astype(str)
    if indexed["image_key"].duplicated().any():
        raise ValueError(f"{name} image_key 不可重複")
    return indexed.set_index("image_key", drop=False)


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
    *,
    exact: bool = False,
) -> None:
    """驗證 DataFrame 必要欄位，必要時要求 exact order。"""
    expected = list(columns)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} 缺少欄位：{missing}")
    if exact and list(frame.columns) != expected:
        raise ValueError(f"{name} schema 必須精確為：{expected}")


def _positive_integer(value: object, name: str) -> int:
    """解析至少為一的整數參數，拒絕布林與截斷。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必須是至少為 1 的整數")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是至少為 1 的整數") from error
    if not np.isfinite(numeric) or numeric < 1 or numeric != np.floor(numeric):
        raise ValueError(f"{name} 必須是至少為 1 的整數")
    return int(numeric)


def _label(value: object, name: str) -> int:
    """解析正整數 mask label，不允許 float truncation。"""
    label = _positive_integer(value, name)
    return label


def _forbidden_token(feature: str) -> str:
    """回傳 feature 名稱中第一個 leakage token，沒有則回傳空字串。"""
    lowered = str(feature).lower()
    return next((token for token in _FORBIDDEN_PREDICTOR_TOKENS if token in lowered), "")


__all__ = [
    "EXTRACTION_QC_COLUMNS",
    "FEATURE_QC_COLUMNS",
    "Paper93Bundle",
    "ROSTER_COLUMNS",
    "VALID_COUNT_COLUMNS",
    "extract_locked_paper93",
    "require_paper93_preflight",
]
