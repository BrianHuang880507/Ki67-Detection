import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore
from CellimageSegmentation_v6 import Ui_MainWindow
from main_test2 import analyze_cell
from ki67dtc.cell_anal_plot import plot_global_area_analysis
from PyQt5.QtCore import QTimer
from overlay_utils import render_and_save_overlay, get_overlay_path, apply_overlay, load_mask_file
import shutil
import json

OVERLAY_DIR = "./data2/output/overlays"

_QSS_DARK = """
QMainWindow, QWidget {
    background-color: #1C2030;
    color: #E8EDF5;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QFrame { background-color: transparent; }
QLabel { background-color: transparent; color: #E8EDF5; }

QLabel#Image {
    background-color: #0F1218;
    border: 1px solid #2E3548;
    border-radius: 3px;
}
QLabel#ImageFileName {
    color: #8A9AB0;
    font-size: 14pt;
}
QLabel#AreaScalablePlot, QLabel#AreaScatteringPlot {
    background-color: #E8EFF7;
    border: 1px solid #2E3548;
}
QLabel#widthScaleAppliedLabel, QLabel#heightScaleAppliedLabel {
    color: #4A5868;
}
QLabel#SlideNumber {
    background-color: #0F1218;
    color: #00AEEF;
    border: 1px solid #2E3548;
    border-radius: 3px;
    font-family: "Consolas";
    font-size: 12pt;
    padding: 2px 6px;
}
QLabel#CellNumberLabel {
    color: #7A8899;
    font-size: 12pt;
    font-weight: bold;
    background-color: transparent;
}
QLCDNumber#CellNumberShow {
    background-color: #0F1218;
    color: #00AEEF;
    border: 1px solid #2E3548;
    border-radius: 3px;
}

QPushButton {
    background-color: #252A3A;
    color: #C0C8D8;
    border: 1px solid #353D55;
    border-radius: 4px;
    padding: 5px 12px;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QPushButton:hover {
    background-color: #00AEEF;
    color: #0A0E18;
    border: 1px solid #00AEEF;
}
QPushButton:pressed {
    background-color: #0090CC;
    border: 1px solid #0090CC;
}
QPushButton:disabled {
    background-color: #181C28;
    color: #353D55;
    border: 1px solid #252A3A;
}
QPushButton#SegmentButton {
    background-color: #00AEEF;
    color: #0A0E18;
    border: none;
    border-radius: 5px;
    font-size: 16pt;
    font-weight: bold;
    padding: 8px;
}
QPushButton#SegmentButton:hover {
    background-color: #22C8FF;
    color: #0A0E18;
}
QPushButton#SegmentButton:pressed {
    background-color: #0090CC;
}
QPushButton#BrowseFileButton {
    background-color: transparent;
    color: #00AEEF;
    border: 1px solid #00AEEF;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 11pt;
}
QPushButton#BrowseFileButton:hover {
    background-color: #00AEEF;
    color: #0A0E18;
}
QPushButton#SaveFileButton {
    background-color: transparent;
    color: #00D4B4;
    border: 1px solid #00D4B4;
    border-radius: 4px;
    padding: 5px 12px;
}
QPushButton#SaveFileButton:hover {
    background-color: #00D4B4;
    color: #0A0E18;
}
QPushButton#PreviousButton, QPushButton#NextButton {
    background-color: #1C2030;
    color: #7A8899;
    border: 1px solid #2E3548;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11pt;
    min-width: 88px;
}
QPushButton#PreviousButton:hover, QPushButton#NextButton:hover {
    background-color: #252A3A;
    color: #E8EDF5;
    border: 1px solid #3A4258;
}

QScrollArea {
    background-color: #161A26;
    border: 1px solid #2E3548;
    border-radius: 3px;
}
QScrollArea > QWidget > QWidget {
    background-color: #161A26;
}
QCheckBox {
    color: #A0AABB;
    font-size: 10pt;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #3A4258;
    border-radius: 2px;
    background: #252A3A;
}
QCheckBox::indicator:checked {
    background: #00AEEF;
    border: 1px solid #0090CC;
}
QCheckBox::indicator:hover {
    border: 1px solid #00AEEF;
}

QRadioButton {
    color: #C0C8D8;
    font-size: 11pt;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1.5px solid #3A4258;
    border-radius: 7px;
    background: #252A3A;
}
QRadioButton::indicator:checked {
    background: #00AEEF;
    border: 2px solid #0090CC;
}
QRadioButton::indicator:hover {
    border: 1.5px solid #00AEEF;
}

QGroupBox {
    background-color: transparent;
    border: 1px solid #2E3548;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 6px;
    font-size: 8pt;
    font-weight: bold;
    letter-spacing: 1px;
    color: #4A5868;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: #4A5868;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #2E3548;
    border-radius: 2px;
    margin: 0;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: #00AEEF;
    border: 2px solid #1C2030;
}
QSlider::handle:horizontal:hover { background: #22C8FF; }
QSlider::sub-page:horizontal {
    background: #00AEEF;
    border-radius: 2px;
}

QLineEdit {
    background-color: #161A26;
    color: #E8EDF5;
    border: 1px solid #353D55;
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 10pt;
    selection-background-color: #00AEEF;
    selection-color: #0A0E18;
}
QLineEdit:focus { border: 1px solid #00AEEF; }

QScrollBar:vertical {
    background: #1C2030;
    width: 7px;
    margin: 0;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #353D55;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #00AEEF; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #1C2030;
    height: 7px;
    margin: 0;
    border-radius: 3px;
}
QScrollBar::handle:horizontal {
    background: #353D55;
    border-radius: 3px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover { background: #00AEEF; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QProgressDialog {
    background-color: #22283A;
    color: #E8EDF5;
    border: 1px solid #2E3548;
    border-radius: 6px;
}
QProgressDialog QLabel { color: #E8EDF5; font-size: 11pt; }
QProgressBar {
    background-color: #1C2030;
    border: 1px solid #2E3548;
    border-radius: 3px;
    text-align: center;
    color: #E8EDF5;
    font-size: 9pt;
    height: 8px;
}
QProgressBar::chunk {
    background-color: #00AEEF;
    border-radius: 2px;
}
QMessageBox {
    background-color: #22283A;
    color: #E8EDF5;
}
QMessageBox QPushButton { min-width: 80px; padding: 6px 16px; }
"""

_QSS_CLEAN = """
QMainWindow, QWidget {
    background-color: #F4F6F9;
    color: #1A2030;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QFrame { background-color: transparent; }
QLabel { background-color: transparent; color: #1A2030; }

QLabel#Image {
    background-color: #E4E8EF;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
}
QLabel#ImageFileName {
    color: #5A6A88;
    font-size: 14pt;
}
QLabel#AreaScalablePlot, QLabel#AreaScatteringPlot {
    background-color: #FFFFFF;
    border: 1px solid #C8D0DC;
}
QLabel#widthScaleAppliedLabel, QLabel#heightScaleAppliedLabel {
    color: #8898B0;
}
QLabel#SlideNumber {
    background-color: #FFFFFF;
    color: #0090CC;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
    font-family: "Consolas";
    font-size: 12pt;
    padding: 2px 6px;
}
QLabel#CellNumberLabel {
    color: #6B7A99;
    font-size: 12pt;
    font-weight: bold;
    background-color: transparent;
}
QLCDNumber#CellNumberShow {
    background-color: #FFFFFF;
    color: #0090CC;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
}
QPushButton {
    background-color: #FFFFFF;
    color: #3A4258;
    border: 1px solid #C8D0DC;
    border-radius: 4px;
    padding: 5px 12px;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QPushButton:hover {
    background-color: #0090CC;
    color: #FFFFFF;
    border: 1px solid #0090CC;
}
QPushButton:pressed {
    background-color: #006E9E;
    border: 1px solid #006E9E;
}
QPushButton:disabled {
    background-color: #F0F2F5;
    color: #B0B8C8;
    border: 1px solid #D8DDE8;
}
QPushButton#SegmentButton {
    background-color: #0090CC;
    color: #FFFFFF;
    border: none;
    border-radius: 5px;
    font-size: 16pt;
    font-weight: bold;
    padding: 8px;
}
QPushButton#SegmentButton:hover { background-color: #00AEEF; }
QPushButton#SegmentButton:pressed { background-color: #006E9E; }
QPushButton#BrowseFileButton {
    background-color: transparent;
    color: #0090CC;
    border: 1px solid #0090CC;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 11pt;
}
QPushButton#BrowseFileButton:hover { background-color: #0090CC; color: #FFFFFF; }
QPushButton#SaveFileButton {
    background-color: transparent;
    color: #00A890;
    border: 1px solid #00A890;
    border-radius: 4px;
    padding: 5px 12px;
}
QPushButton#SaveFileButton:hover { background-color: #00A890; color: #FFFFFF; }
QPushButton#PreviousButton, QPushButton#NextButton {
    background-color: #F0F2F5;
    color: #6B7A99;
    border: 1px solid #C8D0DC;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11pt;
    min-width: 88px;
}
QPushButton#PreviousButton:hover, QPushButton#NextButton:hover {
    background-color: #E0E5EF;
    color: #1A2030;
    border: 1px solid #A8B4C8;
}
QScrollArea {
    background-color: #FFFFFF;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
}
QScrollArea > QWidget > QWidget { background-color: #FFFFFF; }
QCheckBox { color: #4A5868; font-size: 10pt; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #A8B4C8; border-radius: 2px; background: #FFFFFF;
}
QCheckBox::indicator:checked { background: #0090CC; border: 1px solid #0070A0; }
QCheckBox::indicator:hover { border: 1px solid #0090CC; }
QRadioButton { color: #3A4258; font-size: 11pt; spacing: 6px; }
QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1.5px solid #A8B4C8; border-radius: 7px; background: #FFFFFF;
}
QRadioButton::indicator:checked { background: #0090CC; border: 2px solid #0070A0; }
QRadioButton::indicator:hover { border: 1.5px solid #0090CC; }
QGroupBox {
    background-color: transparent;
    border: 1px solid #C8D0DC; border-radius: 5px;
    margin-top: 10px; padding-top: 6px;
    font-size: 8pt; font-weight: bold; letter-spacing: 1px; color: #8898B0;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px; color: #8898B0;
}
QSlider::groove:horizontal { height: 4px; background: #D0D8E8; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0;
    border-radius: 8px; background: #0090CC; border: 2px solid #F4F6F9;
}
QSlider::handle:horizontal:hover { background: #00AEEF; }
QSlider::sub-page:horizontal { background: #0090CC; border-radius: 2px; }
QLineEdit {
    background-color: #FFFFFF; color: #1A2030;
    border: 1px solid #C8D0DC; border-radius: 3px;
    padding: 3px 6px; font-size: 10pt;
    selection-background-color: #0090CC; selection-color: #FFFFFF;
}
QLineEdit:focus { border: 1px solid #0090CC; }
QScrollBar:vertical { background: #F0F2F5; width: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:vertical { background: #C0C8D8; border-radius: 3px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #0090CC; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #F0F2F5; height: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:horizontal { background: #C0C8D8; border-radius: 3px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background: #0090CC; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QProgressDialog {
    background-color: #FFFFFF; color: #1A2030;
    border: 1px solid #C8D0DC; border-radius: 6px;
}
QProgressDialog QLabel { color: #1A2030; font-size: 11pt; }
QProgressBar {
    background-color: #E8EDF5; border: 1px solid #C8D0DC;
    border-radius: 3px; text-align: center; color: #3A4258;
    font-size: 9pt; height: 8px;
}
QProgressBar::chunk { background-color: #0090CC; border-radius: 2px; }
QMessageBox { background-color: #FFFFFF; color: #1A2030; }
QMessageBox QPushButton { min-width: 80px; padding: 6px 16px; }
"""

_QSS_STEEL = """
QMainWindow, QWidget {
    background-color: #2C3347;
    color: #D8E0F0;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QFrame { background-color: transparent; }
QLabel { background-color: transparent; color: #D8E0F0; }

QLabel#Image {
    background-color: #1E2535;
    border: 1px solid #3A4560;
    border-radius: 3px;
}
QLabel#ImageFileName { color: #8A9ABB; font-size: 14pt; }
QLabel#AreaScalablePlot, QLabel#AreaScatteringPlot {
    background-color: #EBF0F8;
    border: 1px solid #3A4560;
}
QLabel#widthScaleAppliedLabel, QLabel#heightScaleAppliedLabel {
    color: #6B7A99;
}
QLabel#SlideNumber {
    background-color: #1E2535;
    color: #5B8BE8;
    border: 1px solid #3A4560;
    border-radius: 3px;
    font-family: "Consolas";
    font-size: 12pt;
    padding: 2px 6px;
}
QLabel#CellNumberLabel {
    color: #7A8899; font-size: 12pt; font-weight: bold;
    background-color: transparent;
}
QLCDNumber#CellNumberShow {
    background-color: #1E2535; color: #5B8BE8;
    border: 1px solid #3A4560; border-radius: 3px;
}
QPushButton {
    background-color: #343C55; color: #A8B8D0;
    border: 1px solid #3A4560; border-radius: 4px;
    padding: 5px 12px; font-family: "Segoe UI"; font-size: 11pt;
}
QPushButton:hover { background-color: #5B8BE8; color: #FFFFFF; border: 1px solid #5B8BE8; }
QPushButton:pressed { background-color: #3A6BC8; border: 1px solid #3A6BC8; }
QPushButton:disabled { background-color: #242B3E; color: #3A4560; border: 1px solid #2A3248; }
QPushButton#SegmentButton {
    background-color: #00AEEF; color: #0A0E18;
    border: none; border-radius: 5px;
    font-size: 16pt; font-weight: bold; padding: 8px;
}
QPushButton#SegmentButton:hover { background-color: #22C8FF; }
QPushButton#SegmentButton:pressed { background-color: #0090CC; }
QPushButton#BrowseFileButton {
    background-color: transparent; color: #5B8BE8;
    border: 1px solid #5B8BE8; border-radius: 4px;
    padding: 6px 12px; font-size: 11pt;
}
QPushButton#BrowseFileButton:hover { background-color: #5B8BE8; color: #FFFFFF; }
QPushButton#SaveFileButton {
    background-color: transparent; color: #3BD4C0;
    border: 1px solid #3BD4C0; border-radius: 4px; padding: 5px 12px;
}
QPushButton#SaveFileButton:hover { background-color: #3BD4C0; color: #0A0E18; }
QPushButton#PreviousButton, QPushButton#NextButton {
    background-color: #2C3347; color: #6B7A99;
    border: 1px solid #3A4560; border-radius: 4px;
    padding: 4px 10px; font-size: 11pt; min-width: 88px;
}
QPushButton#PreviousButton:hover, QPushButton#NextButton:hover {
    background-color: #343C55; color: #D8E0F0; border: 1px solid #4A5878;
}
QScrollArea { background-color: #242B3E; border: 1px solid #3A4560; border-radius: 3px; }
QScrollArea > QWidget > QWidget { background-color: #242B3E; }
QCheckBox { color: #8A9BC0; font-size: 10pt; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #3A4560; border-radius: 2px; background: #343C55;
}
QCheckBox::indicator:checked { background: #5B8BE8; border: 1px solid #3A6BC8; }
QCheckBox::indicator:hover { border: 1px solid #5B8BE8; }
QRadioButton { color: #A8B8D0; font-size: 11pt; spacing: 6px; }
QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1.5px solid #3A4560; border-radius: 7px; background: #343C55;
}
QRadioButton::indicator:checked { background: #5B8BE8; border: 2px solid #3A6BC8; }
QRadioButton::indicator:hover { border: 1.5px solid #5B8BE8; }
QGroupBox {
    background-color: transparent;
    border: 1px solid #3A4560; border-radius: 5px;
    margin-top: 10px; padding-top: 6px;
    font-size: 8pt; font-weight: bold; letter-spacing: 1px; color: #4A5878;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px; color: #4A5878;
}
QSlider::groove:horizontal { height: 4px; background: #3A4560; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0;
    border-radius: 8px; background: #5B8BE8; border: 2px solid #2C3347;
}
QSlider::handle:horizontal:hover { background: #7AAAF8; }
QSlider::sub-page:horizontal { background: #5B8BE8; border-radius: 2px; }
QLineEdit {
    background-color: #1E2535; color: #D8E0F0;
    border: 1px solid #3A4560; border-radius: 3px;
    padding: 3px 6px; font-size: 10pt;
    selection-background-color: #5B8BE8; selection-color: #FFFFFF;
}
QLineEdit:focus { border: 1px solid #5B8BE8; }
QScrollBar:vertical { background: #2C3347; width: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:vertical { background: #3A4560; border-radius: 3px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #5B8BE8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #2C3347; height: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:horizontal { background: #3A4560; border-radius: 3px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background: #5B8BE8; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QProgressDialog {
    background-color: #343C55; color: #D8E0F0;
    border: 1px solid #3A4560; border-radius: 6px;
}
QProgressDialog QLabel { color: #D8E0F0; font-size: 11pt; }
QProgressBar {
    background-color: #2C3347; border: 1px solid #3A4560;
    border-radius: 3px; text-align: center; color: #D8E0F0;
    font-size: 9pt; height: 8px;
}
QProgressBar::chunk { background-color: #5B8BE8; border-radius: 2px; }
QMessageBox { background-color: #343C55; color: #D8E0F0; }
QMessageBox QPushButton { min-width: 80px; padding: 6px 16px; }
"""

THEMES = {
    "dark": _QSS_DARK,
    "clean": _QSS_CLEAN,
    "steel": _QSS_STEEL,
}
_THEME_ACCENT = {
    "dark": "#00AEEF",
    "clean": "#0090CC",
    "steel": "#5B8BE8",
}

CONFIG_PATH = "./config.json"

class AnalysisWorker(QtCore.QThread):
    progress_changed = QtCore.pyqtSignal(int)
    status_changed   = QtCore.pyqtSignal(str)
    finished         = QtCore.pyqtSignal()

    def __init__(self, data_folder, data_folder_out, thres_logarea, CYTO_MODEL_PATH, NUC_MODEL_PATH,
                 selected_files=None, width_um_per_px=1.5896, height_um_per_px=1.5876, total_images=0):
        super().__init__()
        self.data_folder = data_folder
        self.data_folder_out = data_folder_out
        self.thres_logarea = thres_logarea
        self.CYTO_MODEL_PATH = CYTO_MODEL_PATH
        self.NUC_MODEL_PATH = NUC_MODEL_PATH
        self.selected_files = selected_files or []
        self.width_um_per_px = width_um_per_px
        self.height_um_per_px = height_um_per_px
        self.total_images = total_images

    def run(self):
        analyze_cell(
            self.data_folder,
            self.data_folder_out,
            self.thres_logarea,
            self.CYTO_MODEL_PATH,
            self.NUC_MODEL_PATH,
            progress_callback=self.progress_changed.emit,
            status_callback=self.status_changed.emit,
            width_um_per_px=self.width_um_per_px,
            height_um_per_px=self.height_um_per_px
        )
        # Pre-render overlays so image switching is instant after analysis
        overlay_dir = os.path.join(self.data_folder_out, "overlays")
        _n_overlays = len(self.selected_files)
        for _i, file_path in enumerate(self.selected_files):
            self.status_changed.emit(f"Preparing overlay {_i+1}/{_n_overlays}: {Path(file_path).name}")
            render_and_save_overlay(file_path, overlay_dir)
            self.progress_changed.emit(self.total_images * 6 + _i + 1)
        self.finished.emit()

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Load persisted theme and apply before any signal fires
        _saved_theme = "dark"
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f:
                    _saved_theme = json.load(f).get("theme", "dark")
                if _saved_theme not in THEMES:
                    _saved_theme = "dark"
            except Exception:
                pass
        self._apply_theme(_saved_theme)

        # Window icon and title
        _logo_path = self.get_resource_path("itri_EL_C-png/itri_EL_C.png")
        if os.path.exists(_logo_path):
            self.setWindowIcon(QtGui.QIcon(_logo_path))
        self.setWindowTitle("ITRI CytoScope")

        self.setup_control()
        self.selected_files = []
        self.current_index = -1
        self.mask_on = True
        self.paired_mode = False
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.timeout.connect(self._update_slider_plot)
        self._pending_slider_value = None
        self._image_cache = {}   # file_path -> image_rgb ndarray

    def setup_control(self):
        self.ui.BrowseFileButton.clicked.connect(self.open_files)
        self.ui.SegmentButton.clicked.connect(self.run_analysis)
        self.ui.PreviousButton.clicked.connect(self.show_previous_image)
        self.ui.NextButton.clicked.connect(self.show_next_image)
        self.ui.MaskOnButton.clicked.connect(self.toggle_mask)
        self.ui.horizontalSlider.valueChanged.connect(self.slider)
        self.ui.SaveFileButton.clicked.connect(self.save_results)
        self.ui.overlayAllButton.toggled.connect(self._on_overlay_mode_changed)
        self.ui.actionDarkLab.triggered.connect(lambda: self._apply_theme("dark"))
        self.ui.actionCleanLab.triggered.connect(lambda: self._apply_theme("clean"))
        self.ui.actionBlueSteelLab.triggered.connect(lambda: self._apply_theme("steel"))
        self.ui.CellNumberShow.display("0")

    def _on_overlay_mode_changed(self, all_cells_checked):
        self.paired_mode = not all_cells_checked
        self.show_image()

    def _apply_theme(self, name: str):
        self._current_theme = name
        self.setStyleSheet(THEMES.get(name, _QSS_DARK))
        self._save_config(name)
        _action_map = {
            "dark": self.ui.actionDarkLab,
            "clean": self.ui.actionCleanLab,
            "steel": self.ui.actionBlueSteelLab,
        }
        if name in _action_map:
            _action_map[name].setChecked(True)

    def _save_config(self, name: str):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump({"theme": name}, f)
        except Exception:
            pass

    def open_files(self):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Files",
            "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
        )
        if file_names:
            layout = self.ui.fileCheckboxContainerLayout
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            for fname in file_names:
                cb = QtWidgets.QCheckBox(fname)
                cb.setChecked(True)
                cb.stateChanged.connect(self.on_checkbox_state_changed)
                layout.addWidget(cb)
            layout.addStretch()
            self.selected_files = file_names
            self.current_index = 0 if self.selected_files else -1

    def on_checkbox_state_changed(self, state):
        # checkbox = self.sender()  # which checkbox sent the signal
        # if state == QtCore.Qt.Checked:
        #     print(f"Selected: {checkbox.text()}")
        # else:
        #     print(f"Deselected: {checkbox.text()}")
        pass

    def get_selected_files(self):
        layout = self.ui.fileCheckboxContainerLayout
        self.selected_files = [
            layout.itemAt(i).widget().text()
            for i in range(layout.count() - 1)
            if isinstance(layout.itemAt(i).widget(), QtWidgets.QCheckBox) and layout.itemAt(i).widget().isChecked()
        ]
        self.current_index = 0
        return self.selected_files

    def prepare_input_folder(self):
        input_folder = "./data2/input" 
        if os.path.exists(input_folder):
            for fname in os.listdir(input_folder):
                fpath = os.path.join(input_folder, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
        else:
            os.makedirs(input_folder)
        new_selected_files = []
        for src_path in self.selected_files:
            dst_path = os.path.join(input_folder, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            new_selected_files.append(dst_path)
        self.selected_files = new_selected_files

    
    def get_resource_path(self,relative_path):
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)
    
    def run_analysis(self):
        selected_files = self.get_selected_files()
        if not selected_files:
            QtWidgets.QMessageBox.warning(self, "No Files Selected", "Please select at least one file to analyze.")
            return
        self.prepare_input_folder()
        # Clear image cache and delete pre-rendered overlays from previous batch
        self._image_cache.clear()
        if os.path.exists(OVERLAY_DIR):
            shutil.rmtree(OVERLAY_DIR)
        total_images = len(selected_files)
        self.progress = QtWidgets.QProgressDialog("Processing images...", "Cancel", 0, total_images * 7, self)
        self.progress.setWindowModality(QtCore.Qt.ApplicationModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        self.setEnabled(False)
        # 讀取 UI 輸入的倍率，空白或非數字時退回預設值
        DEFAULT_W, DEFAULT_H = 1.5896, 1.5876
        try:
            width_val = float(self.ui.widthScaleInput.text().strip() or DEFAULT_W)
        except ValueError:
            width_val = DEFAULT_W
        try:
            height_val = float(self.ui.heightScaleInput.text().strip() or DEFAULT_H)
        except ValueError:
            height_val = DEFAULT_H
        # 儲存本次套用的值，分析完成後顯示
        self._applied_width = width_val
        self._applied_height = height_val

        self.worker = AnalysisWorker(
            data_folder="./data2/input",
            data_folder_out="./data2/output",
            thres_logarea=7,
            CYTO_MODEL_PATH=self.get_resource_path("model/model_BDL6_label_new"),
            NUC_MODEL_PATH=self.get_resource_path("model/model_BDL3_label_dapi"),
            selected_files=self.selected_files,
            width_um_per_px=width_val,
            height_um_per_px=height_val,
            total_images=total_images
        )
        self.worker.progress_changed.connect(self.progress.setValue)
        self.worker.status_changed.connect(self.progress.setLabelText)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

    def analysis_finished(self):
        self.progress.close()
        self.setEnabled(True)
        # 更新「套用」顯示標籤
        w = getattr(self, '_applied_width', 1.5896)
        h = getattr(self, '_applied_height', 1.5876)
        accent = _THEME_ACCENT.get(getattr(self, "_current_theme", "dark"), "#00AEEF")
        self.ui.widthScaleAppliedLabel.setText(f"套用：{w}")
        self.ui.widthScaleAppliedLabel.setStyleSheet(f"color: {accent};")
        self.ui.heightScaleAppliedLabel.setText(f"套用：{h}")
        self.ui.heightScaleAppliedLabel.setStyleSheet(f"color: {accent};")
        QtWidgets.QMessageBox.information(self, "Done", "Analysis finished.")
        if self.selected_files:
            # Show cell number from ALL_para_combine.csv
            csv_path = r".\data2\output\results\ALL_para_combine.csv"
            cell_count = self.get_cell_count(csv_path)
            self.ui.CellNumberShow.display(cell_count)
            self.show_image()

    def get_cell_count(self, csv_path):
        import pandas as pd
        if not os.path.exists(csv_path):
            return 0
        df = pd.read_csv(csv_path)
        # ALL_para_combine.csv 經 merged_excel() 處理後，nuc/cyto 已合併為一行
        # 所以每行就代表一顆配對細胞，直接回傳總行數即為實際細胞數
        return df.shape[0]

    def show_image(self):
        self.ui.Image.clear()
        if not self.selected_files or self.current_index < 0:
            return
        file_path = self.selected_files[self.current_index]
        self.ui.ImageFileName.setText(os.path.basename(file_path))

        # --- 讀取原圖（cache 起來，切換時不重複讀磁碟）---
        if file_path not in self._image_cache:
            image = cv2.imread(file_path)
            if image is None:
                self.show_message("Error", f"Cannot open {file_path}")
                return
            self._image_cache[file_path] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = self._image_cache[file_path]

        if self.mask_on:
            overlay_path = get_overlay_path(file_path, OVERLAY_DIR, self.paired_mode)
            if os.path.exists(overlay_path):
                overlay = np.load(overlay_path)
            else:
                # Masks not yet available (no analysis run), render on the fly
                overlay = image_rgb.copy()
                cyto_mask = load_mask_file(file_path, "_cyto_seg.npy")
                if cyto_mask is not None:
                    nuc_mask = load_mask_file(file_path, "_nuc_seg.npy")
                    apply_overlay(overlay, cyto_mask, nuc_mask, paired=self.paired_mode)
        else:
            overlay = image_rgb.copy()

        self.display_image_on_label(self.ui.Image, overlay)
        self.display_plot_on_label(self.ui.AreaScatteringPlot, r".\data2\output\figure\all_cell_nucleus_area.png")
        self.display_plot_on_label(self.ui.AreaScalablePlot, r".\data2\output\figure\all_log_cell_area_distribution.png")

    def display_image_on_label(self, label, image):
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)

            # print(f"Label size before (in display method): {label.size()}")
  
            label.setPixmap(pixmap.scaled(
                label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation
            ))

            label.update()

            # print(f"Label size after (in display method): {label.size()}")


    def display_plot_on_label(self, label, plot_path): 
        pixmap = QtGui.QPixmap(plot_path) 
        if pixmap.isNull(): 
            self.show_message("Plot Error", f"Could not load plot image:\n{plot_path}") 
        else: 
            label.setPixmap(pixmap.scaled( label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation ))

    def show_message(self, title, text):
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        msg.exec_()

    def show_previous_image(self):
        if self.selected_files and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.selected_files and self.current_index < len(self.selected_files) - 1:
            self.current_index += 1
            self.show_image()

    def toggle_mask(self, checked):
        self.mask_on = checked
        self.show_image()

    def slider(self, value):
        value=value/10
        # Update label text immediately    
        self.ui.SlideNumber.setText(str(value))
        # Debounce: store the value and start/restart the timer
        self._pending_slider_value = value
        self.slider_timer.start(200)  # Debounced plot update

    def _update_slider_plot(self):
        QtGui.QPixmapCache.clear()
        value = self._pending_slider_value
        all_final_csv = r"data2\output\results\ALL_para_combine.csv"
        outdir = Path(r"data2/output/figure")
        w = getattr(self, '_applied_width', 1.5896)
        h = getattr(self, '_applied_height', 1.5876)
        plot_global_area_analysis(all_final_csv, outdir, thres=float(value),
                                  width_um_per_px=w, height_um_per_px=h)
        hist_path = outdir / "all_log_cell_area_distribution.png"
        # print(hist_path)
        pixmap = QtGui.QPixmap(str(hist_path))
        if pixmap.isNull():
            self.show_message("Plot Error", f"Could not load plot image:\n{hist_path}")
        else:
            self.ui.AreaScalablePlot.setPixmap(pixmap.scaled(
                self.ui.AreaScalablePlot.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            ))
            print(f"Updated plot with threshold: {value}")

    def save_results(self):
        # Ask user for target directory
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Save Directory")
        if not target_dir:
            return

        # Create a results folder with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = os.path.join(target_dir, f"AnalysisResults_{timestamp}")
        os.makedirs(results_folder, exist_ok=True)

        # Save analyzed images (as shown in GUI)
        images_folder = os.path.join(results_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        for idx, file_path in enumerate(self.selected_files):
            image = cv2.imread(file_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlay_path = get_overlay_path(file_path, OVERLAY_DIR, self.paired_mode)
            if self.mask_on and os.path.exists(overlay_path):
                overlay = np.load(overlay_path)
            else:
                overlay = image_rgb.copy()
            save_path = os.path.join(images_folder, f"{idx:03d}_{os.path.basename(file_path)}")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save plots
        plot_files = [
            (r".\data2\output\figure\all_cell_nucleus_area.png", "all_cell_nucleus_area.png"),
            (r".\data2\output\figure\all_log_cell_area_distribution.png", "all_log_cell_area_distribution.png"),
        ]
        for src, dst in plot_files:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(results_folder, dst))

        # Save parameter CSV
        csv_src = r".\data2\output\results\ALL_para_combine.csv"
        if os.path.exists(csv_src):
            shutil.copy2(csv_src, os.path.join(results_folder, "ALL_para_combine.csv"))

        QtWidgets.QMessageBox.information(self, "Save Complete", f"Results saved to:\n{results_folder}")