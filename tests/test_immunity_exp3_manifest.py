"""驗證 Exp3 跨 passage 影像 manifest、配對 QC 與 condition mapping。"""

from pathlib import Path

import pandas as pd
import pytest

from immunity.exp3.manifest import (
    apply_condition_mapping,
    parse_image_name,
    scan_datasets,
    validate_expected_totals,
)


def test_parser_accepts_all_three_naming_styles() -> None:
    """三種已知命名格式應解析出相同的 condition 與 FOV。"""
    assert parse_image_name("4-phase-100X-10.jpg", "pc") == (4, 10)
    assert parse_image_name("B4-P6-10x-4-10.jpg", "pc") == (4, 10)
    assert parse_image_name("B8-P7-10X-8.-10.jpg", "pc") == (8, 10)
    assert parse_image_name("B8-P7-10X-8-IDO.-10.jpg", "ido") == (8, 10)


def test_parser_rejects_channel_mismatch_and_invalid_fov() -> None:
    """錯誤 channel 或 FOV 範圍不得進入配對流程。"""
    with pytest.raises(ValueError, match="channel"):
        parse_image_name("1-IDO-100X-1.jpg", "pc")
    with pytest.raises(ValueError, match="FOV"):
        parse_image_name("B4-P6-10X-1-11.jpg", "pc")


def test_scan_keeps_pair_and_records_missing_ido(tmp_path: Path) -> None:
    """缺少 IDO 的 key 應留在 QC，完整 pair 才能進入 raw manifest。"""
    root = tmp_path / "B7-P5"
    (root / "PC").mkdir(parents=True)
    (root / "IDO").mkdir()
    (root / "PC" / "1-phase-100X-1.jpg").touch()
    (root / "IDO" / "1-IDO-100X-1.jpg").touch()
    (root / "PC" / "4-phase-100X-10.jpg").touch()

    manifest, qc = scan_datasets([{"input_dir": root, "b_id": "B7", "passage": 5}])

    assert list(manifest[["condition_index", "fov"]].itertuples(index=False, name=None)) == [
        (1, 1)
    ]
    missing = qc[qc["status"].eq("missing_ido")]
    assert list(missing[["condition_index", "fov"]].itertuples(index=False, name=None)) == [
        (4, 10)
    ]


def test_scan_records_duplicate_and_parse_error_deterministically(tmp_path: Path) -> None:
    """重複與無法解析檔案應保留在 QC，且列順序不可依掃描順序漂移。"""
    root = tmp_path / "B4-P6"
    (root / "PC").mkdir(parents=True)
    (root / "IDO").mkdir()
    (root / "PC" / "B4-P6-10X-2-01.jpg").touch()
    (root / "PC" / "B4-P6-10X-2-01.png").touch()
    (root / "IDO" / "B4-P6-10X-2-IDO-01.jpg").touch()
    (root / "IDO" / "unparseable.jpg").touch()

    manifest, qc = scan_datasets([{"input_dir": root, "b_id": "B4", "passage": 6}])

    assert manifest.empty
    assert qc["status"].tolist() == ["duplicate_pc", "parse_error"]
    duplicate = qc.iloc[0]
    assert (duplicate["pc_count"], duplicate["ido_count"]) == (2, 1)
    assert "unparseable.jpg" in qc.iloc[1]["detail"]


def test_condition_mapping_is_mandatory_and_exact() -> None:
    """condition mapping 必須完整覆蓋 1 至 8，且每個 condition 唯一。"""
    raw = pd.DataFrame(
        [
            {
                "b_id": "B4",
                "passage": 5,
                "condition_index": 1,
                "fov": 1,
                "group_id": "B4_P5",
                "pc_path": "pc.jpg",
                "ido_path": "ido.jpg",
            }
        ]
    )
    with pytest.raises(ValueError, match="condition mapping"):
        apply_condition_mapping(raw, {})
    mapping = {
        index: {
            "ifn_dose": float(index - 1),
            "tnf_dose": 0.0,
            "condition": f"condition_{index}",
        }
        for index in range(1, 9)
    }

    mapped = apply_condition_mapping(raw, mapping)

    assert mapped.loc[0, "condition"] == "condition_1"
    assert mapped.loc[0, "image_key"] == "B4_P5_C01_F01"


def test_expected_totals_are_enforced() -> None:
    """總數 gate 應以 QC 計數拒絕不符合 config 的資料集。"""
    qc = pd.DataFrame(
        [
            {"status": "paired", "pc_count": 1, "ido_count": 1},
            {"status": "missing_ido", "pc_count": 1, "ido_count": 0},
        ]
    )

    validate_expected_totals(qc, {"pc": 2, "ido": 1, "paired": 1})
    with pytest.raises(ValueError, match="expected totals"):
        validate_expected_totals(qc, {"pc": 2, "ido": 2, "paired": 2})
