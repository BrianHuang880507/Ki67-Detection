from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def _write_snapshot_inputs(
    root: Path,
    *,
    manifest_keys: tuple[str, ...] = ("FOV_A", "FOV_B"),
    cell_keys: tuple[str, ...] = ("FOV_A", "FOV_B"),
    mask_keys: tuple[str, ...] = ("FOV_A", "FOV_B"),
    cell_rows: int | None = None,
    group_rows: tuple[tuple[str, int], ...] | None = None,
) -> tuple[Path, Path, Path]:
    """建立可控制 key 集合的最小 snapshot fixture。"""
    manifest_path = root / "data_manifest.csv"
    cells_path = root / "feature_cache" / "cell_level_basic.csv"
    masks_path = root / "feature_cache" / "masks"
    masks_path.mkdir(parents=True)

    if group_rows is None:
        group_rows = tuple(("B4", 5) for _ in manifest_keys)
    pd.DataFrame(
        [
            {"image_key": key, "b_id": b_id, "passage": passage}
            for key, (b_id, passage) in zip(manifest_keys, group_rows)
        ]
    ).to_csv(manifest_path, index=False)
    cell_values = list(cell_keys)
    if cell_rows is not None:
        if not cell_values:
            cell_values = ["FOV_A"]
        cell_values = [cell_values[index % len(cell_values)] for index in range(cell_rows)]
    pd.DataFrame({"image_key": cell_values}).to_csv(cells_path, index=False)
    for key in mask_keys:
        (masks_path / f"{key}.npz").write_bytes(b"fixture")
    return manifest_path, cells_path, masks_path


def _expected(
    count: int = 2,
    *,
    cell_rows: int | None = None,
    groups=None,
    group_count: int | None = None,
):
    from immunity.exp4.data_snapshot import SnapshotExpectation

    if cell_rows is None:
        cell_rows = count
    if groups is None:
        groups = ("B4_P5",)
    return SnapshotExpectation(
        manifest_image_key_count=count,
        cell_row_count=cell_rows,
        cell_image_key_count=count,
        group_count=len(groups) if group_count is None else group_count,
        group_keys=tuple(groups),
    )


def test_non_npz_files_and_directories_are_ignored(tmp_path: Path) -> None:
    from immunity.exp4.data_snapshot import validate_data_snapshot

    manifest, cells, masks = _write_snapshot_inputs(tmp_path)
    (masks / "ignored.txt").write_text("not a mask", encoding="utf-8")
    (masks / "nested").mkdir()
    (masks / "nested" / "ignored.csv").write_text("not a mask", encoding="utf-8")

    result = validate_data_snapshot(
        manifest_path=manifest,
        cell_level_path=cells,
        masks_dir=masks,
        expected=_expected(),
    )

    assert result.matched is True
    assert result.actual["mask_npz_count"] == 2
    assert result.actual["mask_extension_distribution"] == {".npz": 2}


def test_manifest_key_without_mask_is_fail_closed(tmp_path: Path) -> None:
    from immunity.exp4.data_snapshot import validate_data_snapshot

    manifest, cells, masks = _write_snapshot_inputs(tmp_path, mask_keys=("FOV_A",))

    result = validate_data_snapshot(
        manifest_path=manifest,
        cell_level_path=cells,
        masks_dir=masks,
        expected=_expected(),
    )

    assert result.matched is False
    assert result.actual["manifest_missing_mask_keys"] == ["FOV_B"]
    assert result.actual["manifest_missing_mask_count"] == 1
    assert any("manifest_missing_mask" in difference for difference in result.differences)


def test_manifest_and_cells_same_count_but_different_keys_fail_closed(
    tmp_path: Path,
) -> None:
    from immunity.exp4.data_snapshot import validate_data_snapshot

    manifest, cells, masks = _write_snapshot_inputs(
        tmp_path,
        cell_keys=("FOV_A", "FOV_C"),
        mask_keys=("FOV_A", "FOV_B"),
    )

    result = validate_data_snapshot(
        manifest_path=manifest,
        cell_level_path=cells,
        masks_dir=masks,
        expected=_expected(),
    )

    assert result.matched is False
    assert result.actual["manifest_only_keys"] == ["FOV_B"]
    assert result.actual["cell_only_keys"] == ["FOV_C"]
    assert any("key_sets_equal" in difference for difference in result.differences)


def test_extra_mask_and_mask_count_are_observations_only(tmp_path: Path) -> None:
    from immunity.exp4.data_snapshot import validate_data_snapshot

    manifest, cells, masks = _write_snapshot_inputs(
        tmp_path,
        mask_keys=("FOV_A", "FOV_B", "QC_EXTRA"),
    )

    result = validate_data_snapshot(
        manifest_path=manifest,
        cell_level_path=cells,
        masks_dir=masks,
        expected=_expected(),
    )

    assert result.matched is True
    assert result.differences == []
    assert result.actual["mask_npz_count"] == 3
    assert result.actual["mask_stem_not_manifest_keys"] == ["QC_EXTRA"]
    assert result.actual["mask_stem_not_manifest_count"] == 1
    assert any("mask_stem_not_manifest" in observation for observation in result.observations)


def test_cell_row_count_is_a_fail_closed_gate(tmp_path: Path) -> None:
    from immunity.exp4.data_snapshot import validate_data_snapshot

    manifest, cells, masks = _write_snapshot_inputs(tmp_path, cell_rows=3)

    result = validate_data_snapshot(
        manifest_path=manifest,
        cell_level_path=cells,
        masks_dir=masks,
        expected=_expected(cell_rows=2),
    )

    assert result.matched is False
    assert result.actual["cell_row_count"] == 3
    assert any("cell_row_count" in difference for difference in result.differences)


def test_manifest_group_count_and_exact_keys_are_fail_closed_gates(
    tmp_path: Path,
) -> None:
    from immunity.exp4.data_snapshot import validate_data_snapshot

    group_rows = (("B4", 5), ("B7", 6))
    manifest, cells, masks = _write_snapshot_inputs(
        tmp_path,
        group_rows=group_rows,
    )

    result = validate_data_snapshot(
        manifest_path=manifest,
        cell_level_path=cells,
        masks_dir=masks,
        expected=_expected(
            groups=("B4_P5", "B8_P7"),
            group_count=9,
        ),
    )

    assert result.matched is False
    assert result.actual["group_count"] == 2
    assert result.actual["group_keys"] == ["B4_P5", "B7_P6"]
    assert any("group_count" in difference for difference in result.differences)
    assert any("group_keys" in difference for difference in result.differences)


def test_manifest_unique_image_key_count_is_fail_closed_gate(tmp_path: Path) -> None:
    from immunity.exp4.data_snapshot import validate_data_snapshot

    manifest, cells, masks = _write_snapshot_inputs(
        tmp_path,
        manifest_keys=("FOV_A", "FOV_A"),
        cell_keys=("FOV_A",),
        mask_keys=("FOV_A",),
    )

    result = validate_data_snapshot(
        manifest_path=manifest,
        cell_level_path=cells,
        masks_dir=masks,
        expected=_expected(count=2, cell_rows=1),
    )

    assert result.matched is False
    assert result.actual["manifest_duplicate_image_key_count"] == 1
    assert any("manifest image_key is not unique" in difference for difference in result.differences)


def test_cli_reports_real_snapshot_and_returns_zero_for_nonblocking_extra_masks(
    tmp_path: Path, capsys
) -> None:
    from immunity.exp4.run_rui2025 import main

    manifest, cells, masks = _write_snapshot_inputs(
        tmp_path,
        mask_keys=("FOV_A", "FOV_B", "QC_EXTRA"),
    )
    config_path = tmp_path / "exp4.yaml"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-run_metadata.json"
    config_path.write_text(
        yaml.safe_dump(
            {
                "source_root": str(tmp_path),
                "manifest_path": str(manifest),
                "cell_level_path": str(cells),
                "masks_dir": str(masks),
                "metadata_path": str(metadata_path),
                "expected": {
                    "manifest_image_key_count": 2,
                    "cell_row_count": 2,
                    "cell_image_key_count": 2,
                    "group_count": 1,
                    "group_keys": ["B4_P5"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path), "--validate-snapshot"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["matched"] is True
    assert payload["actual"]["mask_npz_count"] == 3
    assert payload["actual"]["mask_stem_not_manifest_keys"] == ["QC_EXTRA"]
    assert captured.err == ""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["snapshot_validation_status"] == "validated"
    assert metadata["snapshot_validation"]["matched"] is True
    assert metadata["snapshot_validation"]["actual"]["mask_npz_count"] == 3
    assert metadata["snapshot_validation"]["actual"][
        "mask_stem_not_manifest_keys"
    ] == ["QC_EXTRA"]
