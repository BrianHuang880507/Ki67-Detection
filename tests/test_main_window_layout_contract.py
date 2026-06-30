import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QStatusBar, QToolButton

from ki67dtc.app_pipeline import PipelineResult
from ki67dtc.gui.main_window import MainWindow
from ki67dtc.gui.theme import APP_QSS


class MainWindowLayoutContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_menu_bar_contains_file_and_analysis_options(self) -> None:
        menu_titles = [action.text() for action in self.window.menuBar().actions()]

        self.assertIn("檔案", menu_titles)
        self.assertIn("分析選項", menu_titles)

    def test_file_menu_exposes_open_action(self) -> None:
        self.assertEqual(self.window.action_open_input.text(), "開啟")

    def test_analysis_menu_exposes_expected_options(self) -> None:
        expected = [
            "核來源",
            "Ki67 Backend",
            "分析方法",
            "螢光分析",
            "Ki67 分析",
            "清理暫存檔案",
        ]
        actual = [action.text() for action in self.window.analysis_option_actions]

        self.assertEqual(actual, expected)

    def test_analysis_option_submenus_expose_backend_values(self) -> None:
        option_groups = [
            (
                self.window.nuc_source_actions,
                [("DAPI", "dapi", True), ("PC", "pc", False)],
            ),
            (
                self.window.ki67_backend_actions,
                [("PyImageJ", "pyimagej", True), ("OpenCV", "opencv", False)],
            ),
            (
                self.window.feature_backend_actions,
                [("PyImageJ", "pyimagej", True), ("Python", "python", False)],
            ),
        ]

        for actions, expected in option_groups:
            with self.subTest(actions=[action.text() for action in actions]):
                actual = [
                    (action.text(), action.data(), action.isChecked())
                    for action in actions
                ]

                self.assertEqual(actual, expected)

    def test_analysis_option_submenus_are_exclusive(self) -> None:
        self.window.nuc_source_actions[1].setChecked(True)
        self.window.ki67_backend_actions[1].setChecked(True)
        self.window.feature_backend_actions[1].setChecked(True)

        self.assertEqual(
            self.window._selected_action_value(
                self.window.nuc_source_actions,
                "dapi",
            ),
            "pc",
        )
        self.assertEqual(
            self.window._selected_action_value(
                self.window.ki67_backend_actions,
                "pyimagej",
            ),
            "opencv",
        )
        self.assertEqual(
            self.window._selected_action_value(
                self.window.feature_backend_actions,
                "pyimagej",
            ),
            "python",
        )
        self.assertFalse(self.window.nuc_source_actions[0].isChecked())
        self.assertFalse(self.window.ki67_backend_actions[0].isChecked())
        self.assertFalse(self.window.feature_backend_actions[0].isChecked())

    def test_right_side_has_four_named_panels(self) -> None:
        expected_names = [
            "terminalPanel",
            "imageListPanel",
            "featureTablePanel",
            "areaChartPanel",
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

    def test_area_chart_panel_keeps_cell_area_analysis_label(self) -> None:
        self.assertEqual(self.window.area_chart_title.text(), "細胞面積分析")

    def test_segmentation_theme_is_applied_to_main_window(self) -> None:
        self.assertIn("#1C2030", APP_QSS)
        self.assertIn("#1C2030", self.window.styleSheet())

    def test_control_button_theme_uses_cyan_default_and_subdued_disabled(self) -> None:
        self.assertIn("QToolButton", APP_QSS)
        self.assertIn("background-color: #00AEEF", APP_QSS)
        self.assertIn("QToolButton:disabled", APP_QSS)
        self.assertIn("background-color: #2A3142", APP_QSS)

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

    def test_load_area_chart_without_existing_file_clears_label(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self.window._current_data_folder = Path(tmp_dir) / "data" / "input" / "demo"

            self.window._load_area_chart()

            self.assertIsNone(getattr(self.window, "_area_chart_pixmap", None))
            self.assertEqual(self.window.area_chart_label.text(), "尚無細胞面積分析圖")

    def test_load_area_chart_ignores_fallback_when_current_dataset_is_set(self) -> None:
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

                self.window._load_area_chart()
            finally:
                os.chdir(original_cwd)

            self.assertIsNone(getattr(self.window, "_area_chart_pixmap", None))
            self.assertEqual(self.window.area_chart_label.text(), "尚無細胞面積分析圖")


if __name__ == "__main__":
    unittest.main()
