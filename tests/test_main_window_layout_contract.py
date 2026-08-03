import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QStatusBar, QToolButton

from ki67dtc.app_pipeline import PipelineResult
from ki67dtc.gui.main_window import MainWindow
from ki67dtc.gui.theme import APP_QSS
from ki67dtc.paired_overlay import build_paired_overlay_data


class MainWindowLayoutContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def _display_all_regions_with_unpaired_data(
        self,
        root: Path,
    ) -> np.ndarray:
        """載入含未配對區域的實際影像，並切換為顯示全部 regions。"""
        image_path = root / "sample.png"
        base = np.full((12, 14, 3), 120, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), base))

        cytoplasm_mask = np.zeros((12, 14), dtype=np.int32)
        cytoplasm_mask[1:10, 1:9] = 1
        cytoplasm_mask[1:4, 10:13] = 2
        nucleus_mask = np.zeros_like(cytoplasm_mask)
        nucleus_mask[4:7, 4:7] = 1
        data = build_paired_overlay_data(cytoplasm_mask, nucleus_mask)
        self.window._pipeline_result = PipelineResult(
            data_folder=root,
            image_files=[image_path],
            paired_overlays={image_path.stem: data},
        )
        self.window._current_image_index = 0
        self.window._load_image_and_overlays(image_path)
        self.window.chk_paired_only.setChecked(False)
        QApplication.processEvents()
        return base

    def test_menu_bar_contains_file_and_analysis_options(self) -> None:
        menu_titles = [action.text() for action in self.window.menuBar().actions()]

        self.assertIn("檔案", menu_titles)
        self.assertIn("分析選項", menu_titles)
        self.assertIn("輸出", menu_titles)

    def test_export_menu_defaults_to_biomedical_and_exposes_two_profiles(self) -> None:
        actions = self.window.export_profile_actions

        self.assertEqual(
            [(action.text(), action.data()) for action in actions],
            [("生醫版本", "biomedical"), ("工程版本", "engineer")],
        )
        self.assertTrue(actions[0].isChecked())
        self.assertFalse(actions[1].isChecked())
        self.assertEqual(self.window._xlsx_profile, "biomedical")

    def test_export_menu_can_switch_to_engineer_profile(self) -> None:
        engineer_action = self.window.export_profile_actions[1]
        engineer_action.setChecked(True)

        self.window._on_export_profile_changed()

        self.assertEqual(self.window._xlsx_profile, "engineer")

    def test_file_menu_exposes_open_action(self) -> None:
        self.assertEqual(self.window.action_open_input.text(), "開啟")

    def test_window_title_and_icon_match_segmentation_ui_branding(self) -> None:
        self.assertEqual(self.window.windowTitle(), "ITRI CytoScope")
        self.assertFalse(self.window.windowIcon().isNull())

    def test_analysis_options_dialog_exposes_expected_controls(self) -> None:
        dialog = self.window._build_analysis_options_dialog()

        self.assertEqual(dialog.windowTitle(), "分析選項")
        for widget_type, object_name in [
            (QtWidgets.QComboBox, "deviceCombo"),
            (QtWidgets.QComboBox, "nucSourceCombo"),
            (QtWidgets.QComboBox, "ki67BackendCombo"),
            (QtWidgets.QComboBox, "featureBackendCombo"),
            (QtWidgets.QCheckBox, "fluorAnalysisCheck"),
            (QtWidgets.QCheckBox, "ki67AnalysisCheck"),
            (QtWidgets.QCheckBox, "cleanTempCheck"),
            (QtWidgets.QDoubleSpinBox, "widthPixelScaleSpin"),
            (QtWidgets.QDoubleSpinBox, "heightPixelScaleSpin"),
        ]:
            with self.subTest(object_name=object_name):
                self.assertIsNotNone(dialog.findChild(widget_type, object_name))

    def test_analysis_options_defaults_to_gpu_pc_and_python(self) -> None:
        self.assertEqual(self.window._device, "gpu")
        self.assertEqual(self.window._nuc_source, "pc")
        self.assertEqual(self.window._feature_backend, "python")

        dialog = self.window._build_analysis_options_dialog()
        for object_name, expected in [
            ("deviceCombo", "gpu"),
            ("nucSourceCombo", "pc"),
            ("featureBackendCombo", "python"),
        ]:
            combo = dialog.findChild(QtWidgets.QComboBox, object_name)
            self.assertIsNotNone(combo)
            assert combo is not None
            self.assertEqual(combo.currentData(), expected)

    def test_analysis_options_dialog_puts_device_first(self) -> None:
        dialog = self.window._build_analysis_options_dialog()
        form = dialog.findChild(QtWidgets.QFormLayout)
        self.assertIsNotNone(form)
        assert form is not None

        label = form.itemAt(0, QtWidgets.QFormLayout.ItemRole.LabelRole)
        field = form.itemAt(0, QtWidgets.QFormLayout.ItemRole.FieldRole)
        self.assertIsNotNone(label)
        self.assertIsNotNone(field)
        assert label is not None and field is not None
        self.assertEqual(label.widget().text(), "運算裝置")
        self.assertEqual(field.widget().objectName(), "deviceCombo")

    def test_analysis_options_dialog_exposes_backend_values(self) -> None:
        dialog = self.window._build_analysis_options_dialog()
        option_groups = {
            "deviceCombo": [("GPU", "gpu"), ("CPU", "cpu")],
            "nucSourceCombo": [("DAPI", "dapi"), ("PC", "pc")],
            "ki67BackendCombo": [("PyImageJ", "pyimagej"), ("OpenCV", "opencv")],
            "featureBackendCombo": [("PyImageJ", "pyimagej"), ("Python", "python")],
        }

        for object_name, expected in option_groups.items():
            combo = dialog.findChild(QtWidgets.QComboBox, object_name)
            self.assertIsNotNone(combo)
            assert combo is not None
            actual = [
                (combo.itemText(index), combo.itemData(index))
                for index in range(combo.count())
            ]

            self.assertEqual(actual, expected)

    def test_analysis_options_dialog_applies_pipeline_values(self) -> None:
        dialog = self.window._build_analysis_options_dialog()
        # 每個值都刻意選成與預設不同，套用失敗時測試才會失敗。
        for object_name, value in [
            ("deviceCombo", "cpu"),
            ("nucSourceCombo", "dapi"),
            ("ki67BackendCombo", "opencv"),
            ("featureBackendCombo", "pyimagej"),
        ]:
            combo = dialog.findChild(QtWidgets.QComboBox, object_name)
            self.assertIsNotNone(combo)
            assert combo is not None
            combo.setCurrentIndex(combo.findData(value))

        fluor_check = dialog.findChild(QtWidgets.QCheckBox, "fluorAnalysisCheck")
        ki67_check = dialog.findChild(QtWidgets.QCheckBox, "ki67AnalysisCheck")
        clean_check = dialog.findChild(QtWidgets.QCheckBox, "cleanTempCheck")
        width_spin = dialog.findChild(QtWidgets.QDoubleSpinBox, "widthPixelScaleSpin")
        height_spin = dialog.findChild(QtWidgets.QDoubleSpinBox, "heightPixelScaleSpin")
        assert fluor_check is not None
        assert ki67_check is not None
        assert clean_check is not None
        assert width_spin is not None
        assert height_spin is not None
        fluor_check.setChecked(False)
        ki67_check.setChecked(False)
        clean_check.setChecked(False)
        width_spin.setValue(2.5)
        height_spin.setValue(3.5)

        self.window._apply_analysis_options_dialog(dialog)

        self.assertEqual(self.window._device, "cpu")
        self.assertEqual(self.window._nuc_source, "dapi")
        self.assertEqual(self.window._ki67_backend, "opencv")
        self.assertEqual(self.window._feature_backend, "pyimagej")
        self.assertFalse(self.window._fluor_analy)
        self.assertFalse(self.window._ki67)
        self.assertFalse(self.window._clean_temp)
        self.assertEqual(self.window._width_um_per_px, 2.5)
        self.assertEqual(self.window._height_um_per_px, 3.5)

    def test_right_side_has_four_named_panels(self) -> None:
        expected_names = [
            "terminalPanel",
            "imageListPanel",
            "areaScatterPanel",
            "areaHistogramPanel",
        ]

        for object_name in expected_names:
            with self.subTest(object_name=object_name):
                self.assertIsNotNone(self.window.findChild(object, object_name))

    def test_right_panel_initial_sizes_prioritize_results_and_chart(self) -> None:
        self.window.resize(1400, 900)
        self.window.show()
        QApplication.processEvents()

        sizes = self.window.right_splitter.sizes()

        self.assertEqual(len(sizes), 4)
        self.assertGreater(sizes[2], sizes[0])
        self.assertGreater(sizes[2], sizes[1])
        self.assertGreater(sizes[3], sizes[0])
        self.assertGreater(sizes[3], sizes[1])

    def test_status_bar_is_not_part_of_main_window_layout(self) -> None:
        self.assertEqual(self.window.findChildren(QStatusBar), [])

    def test_overlay_controls_share_image_file_header_row(self) -> None:
        self.assertEqual(self.window.image_header_widget.objectName(), "imageHeaderRow")
        self.assertIs(self.window.image_file_label.parent(), self.window.image_header_widget)
        self.assertIs(self.window.chk_show_nuc.parent(), self.window.image_header_widget)
        self.assertIs(self.window.chk_show_cyto.parent(), self.window.image_header_widget)
        self.assertIs(
            self.window.chk_paired_only.parent(), self.window.image_header_widget
        )
        self.assertEqual(self.window.chk_paired_only.objectName(), "pairedOnlyCheck")
        self.assertEqual(self.window.chk_paired_only.text(), "僅顯示核質配對")
        self.assertTrue(self.window.chk_paired_only.isChecked())
        self.assertFalse(self.window.chk_paired_only.isEnabled())
        self.assertTrue(self.window._paired_only_preference)
        self.assertIs(self.window.chk_show_ki67.parent(), self.window.image_header_widget)
        self.assertIs(self.window.alpha_slider.parent(), self.window.image_header_widget)

    def test_control_buttons_are_icon_only_and_centered_contract(self) -> None:
        buttons = self.window.control_button_row.findChildren(QToolButton)

        self.assertEqual(
            [button.objectName() for button in buttons],
            ["startButton", "stopButton", "restartButton"],
        )
        self.assertTrue(all(button.text() == "" for button in buttons))
        self.assertEqual(
            self.window.control_button_row.property("alignmentRole"),
            "centeredIconControls",
        )

    def test_start_button_icon_state_changes_when_running(self) -> None:
        self.window._set_running_state(False)
        self.assertEqual(self.window.start_button.property("iconTone"), "black")

        self.window._set_running_state(True)
        self.assertEqual(self.window.start_button.property("iconTone"), "white")

    def test_area_chart_panels_have_scatter_and_histogram_labels(self) -> None:
        self.assertEqual(self.window.area_scatter_title.text(), "3. 細胞面積點狀分布圖")
        self.assertEqual(self.window.area_histogram_title.text(), "4. 細胞面積長條圖")

    def test_segmentation_theme_is_applied_to_main_window(self) -> None:
        self.assertIn("#1C2030", APP_QSS)
        self.assertIn("#1C2030", self.window.styleSheet())

    def test_control_button_theme_uses_cyan_default_and_subdued_disabled(self) -> None:
        self.assertIn("QToolButton", APP_QSS)
        self.assertIn("background-color: #00AEEF", APP_QSS)
        self.assertIn("QToolButton:disabled", APP_QSS)
        self.assertIn("background-color: #2A3142", APP_QSS)

    def test_area_chart_labels_use_white_background(self) -> None:
        self.assertIn("QLabel#areaScatterLabel", APP_QSS)
        self.assertIn("QLabel#areaHistogramLabel", APP_QSS)
        self.assertIn("background-color: #FFFFFF", APP_QSS)

    def test_progress_updates_are_appended_to_terminal_output(self) -> None:
        self.window._on_progress_changed(1, 4, "正在分析影像")

        self.assertEqual(self.window.progress_bar.value(), 25)
        self.assertIn(
            "[INFO] 正在分析影像 (1/4)",
            self.window.terminal_output.toPlainText(),
        )

    def test_image_list_selection_loads_image_and_updates_file_label(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            image = QImage(4, 4, QImage.Format.Format_RGB888)
            image.fill(0x336699)
            self.assertTrue(image.save(str(image_path)))

            self.window._pipeline_result = PipelineResult(
                data_folder=Path(tmp_dir),
                image_files=[image_path],
            )

            self.window._populate_image_list([image_path])

            self.assertEqual(self.window.image_list.count(), 1)
            self.assertEqual(self.window.image_list.currentRow(), 0)
            self.assertEqual(self.window._current_image_index, 0)
            self.assertEqual(self.window.image_file_label.text(), image_path.name)
            self.assertGreater(len(self.window.image_scene.items()), 0)

    def test_load_area_charts_without_existing_files_clears_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self.window._current_data_folder = Path(tmp_dir) / "data" / "input" / "demo"

            self.window._load_area_charts()

            self.assertIsNone(getattr(self.window, "_area_scatter_pixmap", None))
            self.assertIsNone(getattr(self.window, "_area_histogram_pixmap", None))
            self.assertEqual(self.window.area_scatter_label.text(), "尚無細胞面積點狀分布圖")
            self.assertEqual(self.window.area_histogram_label.text(), "尚無細胞面積長條圖")

    def test_load_area_charts_ignores_fallback_when_current_dataset_is_set(self) -> None:
        original_cwd = os.getcwd()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fallback_chart = (
                tmp_path
                / "data"
                / "output"
                / "figure"
                / "all_log_cell_area_distribution.png"
            )
            fallback_chart.parent.mkdir(parents=True)
            image = QImage(4, 4, QImage.Format.Format_RGB888)
            image.fill(0x669933)
            self.assertTrue(image.save(str(fallback_chart)))

            try:
                os.chdir(tmp_path)
                self.window._current_data_folder = (
                    tmp_path / "other_root" / "input" / "selected_dataset"
                )

                self.window._load_area_charts()
            finally:
                os.chdir(original_cwd)

            self.assertIsNone(getattr(self.window, "_area_histogram_pixmap", None))
            self.assertEqual(self.window.area_histogram_label.text(), "尚無細胞面積長條圖")

    def test_load_area_charts_uses_pipeline_result_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            scatter_path = tmp_path / "all_cell_nucleus_area.png"
            histogram_path = tmp_path / "all_log_cell_area_distribution.png"
            image = QImage(4, 4, QImage.Format.Format_RGB888)
            image.fill(0x669933)
            self.assertTrue(image.save(str(scatter_path)))
            self.assertTrue(image.save(str(histogram_path)))

            self.window._pipeline_result = PipelineResult(
                data_folder=tmp_path,
                image_files=[],
                area_scatter_plot=scatter_path,
                area_histogram_plot=histogram_path,
            )

            self.window._load_area_charts()

            self.assertIsNotNone(getattr(self.window, "_area_scatter_pixmap", None))
            self.assertIsNotNone(getattr(self.window, "_area_histogram_pixmap", None))
            self.assertEqual(self.window.area_scatter_label.text(), "")
            self.assertEqual(self.window.area_histogram_label.text(), "")

    def test_analysis_complete_dialog_warns_and_centers_confirm_button(self) -> None:
        dialog = self.window._build_analysis_complete_dialog()
        button = dialog.findChild(QtWidgets.QPushButton, "analysisCompleteConfirmButton")
        button_row = dialog.findChild(QtWidgets.QWidget, "analysisCompleteButtonRow")

        self.assertEqual(dialog.windowTitle(), "警告")
        self.assertEqual(dialog.findChild(QtWidgets.QLabel, "analysisCompleteMessage").text(), "分析完畢!!!")
        self.assertIsNotNone(button)
        self.assertIsNotNone(button_row)
        assert button is not None
        assert button_row is not None
        self.assertEqual(button.text(), "確認")
        self.assertEqual(
            button_row.layout().itemAt(0).alignment(),
            QtCore.Qt.AlignmentFlag.AlignCenter,
        )

    def test_outline_fallback_fills_polygons_and_honors_alpha(self) -> None:
        base = np.full((16, 16, 3), 100, dtype=np.uint8)
        cyto_polys = [
            np.array([[1, 1], [14, 1], [14, 14], [1, 14]], dtype=np.int32)
        ]
        nuc_polys = [
            np.array([[5, 5], [10, 5], [10, 10], [5, 10]], dtype=np.int32)
        ]
        self.window._show_nuc = True
        self.window._show_cyto = True
        self.window._overlay_alpha = 0.5

        overlay = self.window._create_outline_overlay_bgr(
            base,
            nuc_polys=nuc_polys,
            cyto_polys=cyto_polys,
        )

        np.testing.assert_array_equal(overlay[2, 2], np.array([50, 50, 178]))
        np.testing.assert_array_equal(overlay[7, 7], np.array([170, 50, 50]))

        self.window._overlay_alpha = 0.25
        lower_alpha = self.window._create_outline_overlay_bgr(
            base,
            nuc_polys=nuc_polys,
            cyto_polys=cyto_polys,
        )
        np.testing.assert_array_equal(lower_alpha[2, 2], np.array([75, 75, 139]))

        self.window._show_nuc = False
        self.window._show_cyto = False
        hidden = self.window._create_outline_overlay_bgr(
            base,
            nuc_polys=nuc_polys,
            cyto_polys=cyto_polys,
        )
        np.testing.assert_array_equal(hidden, base)

    def test_segmentation_masks_are_loaded_and_rendered_as_colored_overlay(self) -> None:
        original_cwd = os.getcwd()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_dir = tmp_path / "data" / "input" / "demo" / "PC"
            segment_dir = tmp_path / "data" / "output" / "segment" / "demo"
            image_dir.mkdir(parents=True)
            segment_dir.mkdir(parents=True)
            image_path = image_dir / "sample.png"
            base = np.full((8, 8, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), base))

            cyto_mask = np.zeros((8, 8), dtype=np.int32)
            cyto_mask[1:7, 1:7] = 1
            nuc_mask = np.zeros((8, 8), dtype=np.int32)
            nuc_mask[3:5, 3:5] = 1
            np.save(segment_dir / "sample_cyto_seg.npy", {"masks": cyto_mask})
            np.save(segment_dir / "sample_nuc_seg.npy", {"masks": nuc_mask})

            try:
                os.chdir(tmp_path)
                self.window._pipeline_result = PipelineResult(
                    data_folder=tmp_path / "data" / "input" / "demo",
                    image_files=[image_path],
                )
                self.window._current_image_index = 0

                self.window._load_image_and_overlays(image_path)
                self.window._update_display_pixmap()
            finally:
                os.chdir(original_cwd)

            masks = self.window._current_overlay_masks.get(image_path)
            self.assertIsNotNone(masks)
            assert masks is not None
            nuc_loaded, cyto_loaded = masks
            self.assertIsNotNone(nuc_loaded)
            self.assertIsNotNone(cyto_loaded)
            self.assertFalse(np.array_equal(self.window._current_overlay_image_array, base))
            self.assertTrue(self.window.chk_paired_only.isEnabled())
            self.assertIn(image_path.stem, self.window._pipeline_result.paired_overlays)

    def test_paired_only_checkbox_switches_display_without_starting_pipeline(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            base = np.full((12, 14, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), base))

            cyto_mask = np.zeros((12, 14), dtype=np.int32)
            cyto_mask[1:10, 1:9] = 1
            cyto_mask[1:4, 10:13] = 2
            nuc_mask = np.zeros_like(cyto_mask)
            nuc_mask[4:6, 4:6] = 1
            data = build_paired_overlay_data(cyto_mask, nuc_mask)
            self.window._pipeline_result = PipelineResult(
                data_folder=Path(tmp_dir),
                image_files=[image_path],
                paired_overlays={image_path.stem: data},
            )
            self.window._current_image_index = 0

            self.window._load_image_and_overlays(image_path)
            self.window._update_display_pixmap()
            paired = self.window._current_overlay_image_array.copy()
            self.window._selected_cell_id = "sample_1"
            self.window._highlight_enabled = True

            self.window.chk_paired_only.setChecked(False)
            QApplication.processEvents()
            all_regions = self.window._current_overlay_image_array

            self.assertTrue(self.window.chk_paired_only.isEnabled())
            self.assertFalse(self.window._paired_only_preference)
            np.testing.assert_array_equal(paired[2, 11], base[2, 11])
            self.assertFalse(np.array_equal(all_regions[2, 11], base[2, 11]))
            self.assertTrue(self.window._highlight_enabled)
            self.assertEqual(self.window._selected_cell_id, "sample_1")
            self.assertIsNone(self.window._pipeline_thread)

    def test_missing_full_data_forces_paired_only_and_restores_preference(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            available_path = root / "available.png"
            missing_path = root / "missing.png"
            base = np.full((12, 14, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(available_path), base))
            self.assertTrue(cv2.imwrite(str(missing_path), base))

            cyto_mask = np.zeros((12, 14), dtype=np.int32)
            cyto_mask[1:10, 1:9] = 1
            nuc_mask = np.zeros_like(cyto_mask)
            nuc_mask[4:6, 4:6] = 1
            data = build_paired_overlay_data(cyto_mask, nuc_mask)
            self.window._pipeline_result = PipelineResult(
                data_folder=root,
                image_files=[available_path, missing_path],
                paired_overlays={available_path.stem: data},
            )

            self.window._current_image_index = 0
            self.window._load_image_and_overlays(available_path)
            self.window.chk_paired_only.setChecked(False)
            self.window._current_image_index = 1
            self.window._load_image_and_overlays(missing_path)

            self.assertFalse(self.window.chk_paired_only.isEnabled())
            self.assertTrue(self.window.chk_paired_only.isChecked())
            self.assertFalse(self.window._paired_only_preference)
            self.assertIn("僅能顯示核質配對", self.window._last_status_message)

            self.window._current_image_index = 0
            self.window._load_image_and_overlays(available_path)

            self.assertTrue(self.window.chk_paired_only.isEnabled())
            self.assertFalse(self.window.chk_paired_only.isChecked())
            self.assertFalse(self.window._paired_only_preference)

    def test_reset_restores_paired_only_default(self) -> None:
        self.window.chk_paired_only.setEnabled(True)
        self.window.chk_paired_only.setChecked(False)

        self.window._on_reset_clicked()

        self.assertTrue(self.window._paired_only_preference)
        self.assertTrue(self.window.chk_paired_only.isChecked())
        self.assertFalse(self.window.chk_paired_only.isEnabled())

    def test_loading_new_dataset_restores_paired_only_default(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset = Path(tmp_dir) / "data" / "input" / "demo"
            pc_dir = dataset / "PC"
            pc_dir.mkdir(parents=True)
            image_path = pc_dir / "sample.png"
            image = np.full((8, 8, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.window._paired_only_preference = False

            self.window._load_images_from_folder(dataset)

            self.assertTrue(self.window._paired_only_preference)
            self.assertTrue(self.window.chk_paired_only.isChecked())

    def test_running_new_dataset_restores_paired_only_default(self) -> None:
        """Run 路徑啟動新資料集前應恢復 Paired Only 顯示偏好。"""
        self.window.chk_paired_only.setEnabled(True)
        self.window.chk_paired_only.setChecked(False)
        self.window.input_dir_edit.setText("new-dataset")

        with patch("ki67dtc.gui.main_window.PipelineThread"):
            self.window._on_run_clicked()

        self.assertTrue(self.window._paired_only_preference)
        self.assertTrue(self.window.chk_paired_only.isChecked())
        self.assertFalse(self.window.chk_paired_only.isEnabled())

    def test_run_redraws_existing_image_as_paired_only(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base = self._display_all_regions_with_unpaired_data(root)
            self.assertFalse(
                np.array_equal(
                    self.window._current_overlay_image_array[2, 11],
                    base[2, 11],
                )
            )
            self.window.input_dir_edit.setText(str(root))

            with patch("ki67dtc.gui.main_window.PipelineThread"):
                self.window._on_run_clicked()

            self.assertTrue(self.window._paired_only_preference)
            self.assertTrue(self.window.chk_paired_only.isChecked())
            self.assertFalse(self.window.chk_paired_only.isEnabled())
            np.testing.assert_array_equal(
                self.window._current_overlay_image_array[2, 11], base[2, 11]
            )

    def test_pipeline_failure_restores_paired_only_availability(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base = self._display_all_regions_with_unpaired_data(root)
            self.window.input_dir_edit.setText(str(root))
            with patch("ki67dtc.gui.main_window.PipelineThread"):
                self.window._on_run_clicked()

            self.window._on_pipeline_failed("測試失敗")

            self.assertTrue(self.window.chk_paired_only.isEnabled())
            self.assertTrue(self.window.chk_paired_only.isChecked())
            np.testing.assert_array_equal(
                self.window._current_overlay_image_array[2, 11], base[2, 11]
            )

    def test_stopping_running_pipeline_restores_paired_only_availability(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base = self._display_all_regions_with_unpaired_data(root)
            self.window.input_dir_edit.setText(str(root))

            with patch("ki67dtc.gui.main_window.PipelineThread"):
                self.window._on_run_clicked()
                self.window._on_stop_clicked()

            self.assertTrue(self.window.chk_paired_only.isEnabled())
            self.assertTrue(self.window.chk_paired_only.isChecked())
            np.testing.assert_array_equal(
                self.window._current_overlay_image_array[2, 11], base[2, 11]
            )


if __name__ == "__main__":
    unittest.main()
