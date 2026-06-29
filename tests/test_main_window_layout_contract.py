import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QToolButton

from ki67dtc.gui.main_window import MainWindow


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


if __name__ == "__main__":
    unittest.main()
