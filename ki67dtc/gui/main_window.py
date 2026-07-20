from pathlib import Path
from typing import Callable, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QLabel,
    QFileDialog,
    QCheckBox,
    QListWidget,
    QGraphicsView,
    QGraphicsScene,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
)

import cv2
import numpy as np
import csv

from ..app_pipeline import (
    run_pipeline,
    PipelineResult,
    _resolve_data_folder,
    find_merged_outline_for_image,
    load_paired_merged_outlines,
)
from ..paired_overlay import (
    PairedOverlayData,
    build_paired_overlay_data,
    render_paired_overlay_bgr,
)
from .icons import standard_icon
from .theme import APP_QSS

_SEGMENTATION_PALETTE_RGB = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """將 RGB 色碼轉成 OpenCV BGR 色碼。"""
    red, green, blue = color
    return blue, green, red


def render_fill_overlay_bgr(
    base_bgr: np.ndarray,
    nuc_polys: list[np.ndarray] | None,
    cyto_polys: list[np.ndarray] | None,
    *,
    show_nuc: bool = True,
    show_cyto: bool = True,
    alpha: float = 0.5,
) -> np.ndarray:
    """以 OpenCV 在原圖上產生 nucleus 與 cytoplasm 半透明填色疊圖。

    Args:
        base_bgr: OpenCV BGR 原圖。
        nuc_polys: nucleus polygon 清單。
        cyto_polys: cytoplasm polygon 清單。
        show_nuc: 是否繪製 nucleus。
        show_cyto: 是否繪製 cytoplasm。
        alpha: 填色疊圖透明度，範圍為 0 到 1。

    Returns:
        完成半透明填色與邊界繪製的 BGR 影像。

    Raises:
        ValueError: 當 alpha 不在 0 到 1 範圍時拋出。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必須介於 0 與 1 之間")

    overlay = base_bgr.copy()
    nuc_color = (255, 0, 0)
    cyto_color = (0, 0, 255)
    thickness = 1

    if cyto_polys and show_cyto:
        palette_bgr = [_rgb_to_bgr(color) for color in _SEGMENTATION_PALETTE_RGB]
        for index, poly in enumerate(cyto_polys):
            pts = poly.reshape(-1, 1, 2)
            cv2.fillPoly(
                overlay,
                [pts],
                palette_bgr[index % len(palette_bgr)],
            )

    if nuc_polys and show_nuc:
        nuc_fill_color = _rgb_to_bgr((0, 0, 240))
        for poly in nuc_polys:
            pts = poly.reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], nuc_fill_color)

    blended = cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)

    if nuc_polys and show_nuc:
        for poly in nuc_polys:
            pts = poly.reshape(-1, 1, 2)
            cv2.polylines(
                blended, [pts], isClosed=True, color=nuc_color, thickness=thickness
            )

    if cyto_polys and show_cyto:
        for poly in cyto_polys:
            pts = poly.reshape(-1, 1, 2)
            cv2.polylines(
                blended, [pts], isClosed=True, color=cyto_color, thickness=thickness
            )

    return blended


def save_pipeline_fill_overlays(
    result: PipelineResult,
    *,
    alpha: float = 0.5,
) -> list[Path]:
    """批次產生並儲存 pipeline 影像的 Paired Only fill-overlay。

    Args:
        result: Pipeline 完成後的資料集、影像與結果路徑。
        alpha: 填色疊圖透明度，範圍為 0 到 1。

    Returns:
        成功寫入的 overlay PNG 路徑清單。

    Raises:
        FileNotFoundError: 找不到原圖或 merged outline 時拋出。
        OSError: OpenCV 無法寫入 overlay PNG 時拋出。
        ValueError: 當 alpha 不在 0 到 1 範圍時拋出。
    """
    data_folder = Path(result.data_folder).resolve()
    results_dir = (
        Path(result.results_dir)
        if result.results_dir is not None
        else data_folder.parent.parent / "output" / "results" / data_folder.name
    )
    outline_dir = data_folder.parent.parent / "output" / "outline" / data_folder.name
    results_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image_file in result.image_files:
        image_path = Path(image_file)
        base_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if base_bgr is None:
            raise FileNotFoundError(f"無法載入原始影像：{image_path}")
        if base_bgr.ndim == 2:
            base_bgr = cv2.cvtColor(base_bgr, cv2.COLOR_GRAY2BGR)

        paired_data = result.paired_overlays.get(image_path.stem)
        if paired_data is not None:
            overlay_bgr = render_paired_overlay_bgr(
                base_bgr,
                paired_data,
                alpha=alpha,
            )
        else:
            merged_path = outline_dir / f"{image_path.stem}_merged_cp_outlines.txt"
            if not merged_path.exists():
                raise FileNotFoundError(f"找不到 merged outline：{merged_path}")
            polygons = load_paired_merged_outlines(merged_path)
            overlay_bgr = render_fill_overlay_bgr(
                base_bgr,
                polygons.nuc_polygons,
                polygons.cyto_polygons,
                alpha=alpha,
            )
        output_path = results_dir / f"{image_path.stem}_overlay.png"
        if not cv2.imwrite(str(output_path), overlay_bgr):
            raise OSError(f"無法寫入 fill-overlay：{output_path}")
        saved_paths.append(output_path)

    return saved_paths


class PipelineThread(QThread):
    """背景執行 pipeline 的 QThread。

    目前只用步驟級的進度更新，未來可擴充為每張影像的進度。
    """

    progress_changed = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(PipelineResult)
    failed = pyqtSignal(str)

    def __init__(
        self,
        data_folder: Path,
        nuc_source: str,
        fluor_analy: bool,
        ki67: bool,
        ki67_backend: str,
        feature_backend: str,
        clean_temp: bool,
        width_um_per_px: float,
        height_um_per_px: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        """初始化背景 pipeline 執行緒。"""
        super().__init__(parent)
        self._data_folder = data_folder
        self._nuc_source = nuc_source
        self._fluor_analy = fluor_analy
        self._ki67 = ki67
        self._ki67_backend = ki67_backend
        self._feature_backend = feature_backend
        self._clean_temp = clean_temp
        self._width_um_per_px = width_um_per_px
        self._height_um_per_px = height_um_per_px

    def _progress_callback(self, done: int, total: int, message: str) -> None:
        """將 pipeline 進度轉送為 Qt signal。"""
        self.progress_changed.emit(done, total, message)

    def run(self) -> None:  # type: ignore[override]
        """在背景執行 Ki67 pipeline 並回報成功或錯誤。"""
        try:
            result = run_pipeline(
                self._data_folder,
                nuc_source=self._nuc_source,
                fluor_analy=self._fluor_analy,
                ki67=self._ki67,
                ki67_backend=self._ki67_backend,
                feature_backend=self._feature_backend,
                clean_temp=self._clean_temp,
                width_um_per_px=self._width_um_per_px,
                height_um_per_px=self._height_um_per_px,
                progress_callback=self._progress_callback,
            )
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        """建立可滾輪縮放與拖曳的影像檢視元件。"""
        super().__init__(parent)
        self._zoom_factor = 1.15
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """依滑鼠滾輪方向縮放影像視圖。"""
        angle = event.angleDelta().y()
        if angle == 0:
            return
        if angle > 0:
            factor = self._zoom_factor
        else:
            factor = 1.0 / self._zoom_factor
        self.scale(factor, factor)

    def reset_view(self):
        """重設影像視圖縮放矩陣。"""
        self.resetTransform()


class MainWindow(QMainWindow):
    """最小可用的主視窗骨架。

    - 左側：輸入資料夾路徑 + Run / Stop / Reset + 進度列
    - 右側：暫時留白，未來放影像 viewer 與統計資訊
    """

    def __init__(self):
        """初始化主視窗狀態並建立 UI。"""
        super().__init__()
        self._apply_window_branding()
        # 預設視窗更大
        self.resize(1400, 900)
        self.setStyleSheet(APP_QSS)
        self._pipeline_thread: PipelineThread | None = None
        self._pipeline_result: PipelineResult | None = None
        self._current_image_index: int | None = None
        self._current_image_array: np.ndarray | None = None
        self._current_overlay_polygons: dict[
            Path, tuple[list[np.ndarray], list[np.ndarray]]
        ] = {}
        self._current_overlay_masks: dict[
            Path, tuple[np.ndarray | None, np.ndarray | None]
        ] = {}
        self._current_overlay_image: QPixmap | None = None
        self._current_overlay_image_array: np.ndarray | None = None
        self._show_nuc: bool = True
        self._show_cyto: bool = True
        self._show_ki67: bool = False
        self._overlay_alpha: float = 0.5
        self._current_data_folder: Path | None = None
        self._area_scatter_pixmap: QPixmap | None = None
        self._area_histogram_pixmap: QPixmap | None = None
        self._last_status_message: str = ""
        self._nuc_source: str = "dapi"
        self._ki67_backend: str = "pyimagej"
        self._feature_backend: str = "pyimagej"
        self._fluor_analy: bool = True
        self._ki67: bool = True
        self._clean_temp: bool = True
        self._width_um_per_px: float = 1.5896
        self._height_um_per_px: float = 1.5876
        # 新增：目前選中的 Cell_ID（例如 "1_3"）
        self._selected_cell_id: str | None = None
        self._highlight_enabled: bool = False
        # 用來判斷「再次點同一列」
        self._last_selected_row: int | None = None
        self._cleaned_csv_rows: list[list[str]] = []
        self._cleaned_csv_header: list[str] | None = None
        self._build_menu_bar()
        self._build_ui()

        # selection change 只會在 selection 真的改變時觸發；再點同一列不一定會觸發
        # 因此用 clicked 事件確保每次點擊都能切換
        self.results_table.cellClicked.connect(self._on_results_table_cell_clicked)

    def _apply_window_branding(self) -> None:
        """套用與 SegmentationUI 一致的視窗標題與 icon。"""
        self.setWindowTitle("ITRI CytoScope")
        icon_path = Path(__file__).with_name("assets") / "itri_EL_C.png"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

    def _set_running_state(self, running: bool) -> None:
        """設定執行狀態與控制按鈕圖示。"""
        self._is_running = running
        tone = "white" if running else "black"
        self.start_button.setProperty("iconTone", tone)
        self.start_button.setIcon(
            standard_icon(
                self,
                QtWidgets.QStyle.StandardPixmap.SP_MediaPlay,
                tone,
            )
        )
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.restart_button.setEnabled(not running)

    def _selected_action_value(
        self,
        actions: list[QtGui.QAction],
        default: str,
    ) -> str:
        """取得目前選取的 action 資料值。

        Args:
            actions: 同一選項群組中的 action 清單。
            default: 沒有任何 action 被選取時使用的預設值。

        Returns:
            目前勾選 action 的 `data()` 字串，或預設值。
        """
        for action in actions:
            if action.isChecked():
                return str(action.data())
        return default

    def _add_exclusive_option_actions(
        self,
        menu: QtWidgets.QMenu,
        options: list[tuple[str, str, bool]],
    ) -> list[QtGui.QAction]:
        """在 submenu 中加入互斥的選項 actions。

        Args:
            menu: 要放入選項的 submenu。
            options: `(顯示文字, backend value, 是否預設勾選)` 清單。

        Returns:
            已建立的 actions。
        """
        group = QtGui.QActionGroup(menu)
        group.setExclusive(True)
        actions: list[QtGui.QAction] = []
        for label, value, checked in options:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(checked)
            action.setData(value)
            group.addAction(action)
            actions.append(action)
        return actions

    def _on_results_table_cell_clicked(self, row: int, col: int) -> None:
        """滑鼠點擊表格任一格：若點到同一列，切換高亮 on/off。"""
        self._toggle_highlight_by_row(row)

    def _on_results_table_selection_changed(self) -> None:
        """鍵盤或程式改變 selection 時同步高亮狀態。"""
        sel = self.results_table.selectionModel()
        if sel is None or not sel.hasSelection():
            self._selected_cell_id = None
            self._highlight_enabled = False
            self._last_selected_row = None
            self._update_display_pixmap()
            return

        row = sel.selectedRows()[0].row()
        self._toggle_highlight_by_row(row)

    def _toggle_highlight_by_row(self, row: int) -> None:
        """依照 row 讀取 Cell_ID，並依規則切換 highligh。"""
        item = self.results_table.item(row, 0)
        cell_id = item.text().strip() if item is not None else ""
        if not cell_id:
            self._selected_cell_id = None
            self._highlight_enabled = False
            self._last_selected_row = None
            self._update_display_pixmap()
            return

        # 若再次點到同一個 row + 同一個 Cell_ID，切換 on/off
        if self._last_selected_row == row and self._selected_cell_id == cell_id:
            self._highlight_enabled = not self._highlight_enabled
        else:
            self._selected_cell_id = cell_id
            self._highlight_enabled = True
            self._last_selected_row = row

        # 為了讓使用者「再點同一列」也能觸發 selection change（部分平台不會觸發）
        # 先保留顯示上的 row 選取，但不依賴 selection change。
        self._update_display_pixmap()
    # --- UI 組裝 ---

    def _build_menu_bar(self) -> None:
        """建立主視窗選單列與分析選項 actions。"""
        file_menu = self.menuBar().addMenu("檔案")
        self.action_open_input = file_menu.addAction("開啟")
        self.action_open_input.triggered.connect(self._on_browse_input)

        self.action_analysis_options = self.menuBar().addAction("分析選項")
        self.action_analysis_options.triggered.connect(self._open_analysis_options_dialog)

    def _new_option_combo(
        self,
        parent: QWidget,
        object_name: str,
        options: list[tuple[str, str]],
        current_value: str,
    ) -> QtWidgets.QComboBox:
        """建立分析選項子視窗使用的下拉選單。"""
        combo = QtWidgets.QComboBox(parent)
        combo.setObjectName(object_name)
        for label, value in options:
            combo.addItem(label, value)
        index = combo.findData(current_value)
        if index >= 0:
            combo.setCurrentIndex(index)
        return combo

    def _build_analysis_options_dialog(self) -> QtWidgets.QDialog:
        """建立分析選項子視窗。"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("分析選項")
        dialog.setObjectName("analysisOptionsDialog")
        dialog.setModal(True)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        root.addLayout(form)

        form.addRow(
            "核來源",
            self._new_option_combo(
                dialog,
                "nucSourceCombo",
                [("DAPI", "dapi"), ("PC", "pc")],
                self._nuc_source,
            ),
        )
        form.addRow(
            "Ki67 Backend",
            self._new_option_combo(
                dialog,
                "ki67BackendCombo",
                [("PyImageJ", "pyimagej"), ("OpenCV", "opencv")],
                self._ki67_backend,
            ),
        )
        form.addRow(
            "分析方法",
            self._new_option_combo(
                dialog,
                "featureBackendCombo",
                [("PyImageJ", "pyimagej"), ("Python", "python")],
                self._feature_backend,
            ),
        )

        fluor_check = QtWidgets.QCheckBox("螢光分析", dialog)
        fluor_check.setObjectName("fluorAnalysisCheck")
        fluor_check.setChecked(self._fluor_analy)
        form.addRow("", fluor_check)

        ki67_check = QtWidgets.QCheckBox("Ki67 分析", dialog)
        ki67_check.setObjectName("ki67AnalysisCheck")
        ki67_check.setChecked(self._ki67)
        form.addRow("", ki67_check)

        clean_check = QtWidgets.QCheckBox("清理暫存檔案", dialog)
        clean_check.setObjectName("cleanTempCheck")
        clean_check.setChecked(self._clean_temp)
        form.addRow("", clean_check)

        width_spin = QtWidgets.QDoubleSpinBox(dialog)
        width_spin.setObjectName("widthPixelScaleSpin")
        width_spin.setDecimals(4)
        width_spin.setRange(0.0001, 1000.0)
        width_spin.setValue(self._width_um_per_px)
        width_spin.setSuffix(" µm/pixel")
        form.addRow("WIDTH", width_spin)

        height_spin = QtWidgets.QDoubleSpinBox(dialog)
        height_spin.setObjectName("heightPixelScaleSpin")
        height_spin.setDecimals(4)
        height_spin.setRange(0.0001, 1000.0)
        height_spin.setValue(self._height_um_per_px)
        height_spin.setSuffix(" µm/pixel")
        form.addRow("HEIGHT", height_spin)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        confirm_button = QPushButton("確認", dialog)
        confirm_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("取消", dialog)
        cancel_button.clicked.connect(dialog.reject)
        button_row.addWidget(confirm_button)
        button_row.addWidget(cancel_button)
        root.addLayout(button_row)

        return dialog

    def _apply_analysis_options_dialog(self, dialog: QtWidgets.QDialog) -> None:
        """套用分析選項子視窗目前值。"""
        nuc_combo = dialog.findChild(QtWidgets.QComboBox, "nucSourceCombo")
        ki67_backend_combo = dialog.findChild(QtWidgets.QComboBox, "ki67BackendCombo")
        feature_backend_combo = dialog.findChild(QtWidgets.QComboBox, "featureBackendCombo")
        fluor_check = dialog.findChild(QtWidgets.QCheckBox, "fluorAnalysisCheck")
        ki67_check = dialog.findChild(QtWidgets.QCheckBox, "ki67AnalysisCheck")
        clean_check = dialog.findChild(QtWidgets.QCheckBox, "cleanTempCheck")
        width_spin = dialog.findChild(QtWidgets.QDoubleSpinBox, "widthPixelScaleSpin")
        height_spin = dialog.findChild(QtWidgets.QDoubleSpinBox, "heightPixelScaleSpin")

        if nuc_combo is not None:
            self._nuc_source = str(nuc_combo.currentData())
        if ki67_backend_combo is not None:
            self._ki67_backend = str(ki67_backend_combo.currentData())
        if feature_backend_combo is not None:
            self._feature_backend = str(feature_backend_combo.currentData())
        if fluor_check is not None:
            self._fluor_analy = fluor_check.isChecked()
        if ki67_check is not None:
            self._ki67 = ki67_check.isChecked()
        if clean_check is not None:
            self._clean_temp = clean_check.isChecked()
        if width_spin is not None:
            self._width_um_per_px = float(width_spin.value())
        if height_spin is not None:
            self._height_um_per_px = float(height_spin.value())

    def _open_analysis_options_dialog(self) -> None:
        """開啟分析選項子視窗並套用確認後的設定。"""
        dialog = self._build_analysis_options_dialog()
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._apply_analysis_options_dialog(dialog)

    def _new_panel(self, object_name: str, title: str) -> QtWidgets.QFrame:
        """建立右側等分 panel。"""
        panel = QtWidgets.QFrame(self)
        panel.setObjectName(object_name)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        label = QLabel(title, panel)
        label.setObjectName(f"{object_name}Title")
        layout.addWidget(label)
        return panel

    def _build_terminal_panel(self) -> QtWidgets.QFrame:
        """建立輸出主控台與置中的圖示控制按鈕列。"""
        panel = self._new_panel("terminalPanel", "1. 輸出主控台")
        layout = panel.layout()
        assert layout is not None

        self.terminal_output = QtWidgets.QTextEdit(panel)
        self.terminal_output.setObjectName("terminalOutput")
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setText("[INFO] 等待開啟資料夾...")
        layout.addWidget(self.terminal_output, stretch=1)

        control_container = QtWidgets.QWidget(panel)
        control_container.setObjectName("controlButtonRow")
        control_container.setProperty("alignmentRole", "centeredIconControls")
        self.control_button_row = control_container

        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addStretch(1)

        self.start_button = QtWidgets.QToolButton(control_container)
        self.start_button.setObjectName("startButton")
        self.start_button.setText("")
        self.start_button.clicked.connect(self._on_run_clicked)

        self.stop_button = QtWidgets.QToolButton(control_container)
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setText("")
        self.stop_button.setProperty("iconTone", "black")
        self.stop_button.setIcon(
            standard_icon(
                self,
                QtWidgets.QStyle.StandardPixmap.SP_MediaStop,
                "black",
            )
        )
        self.stop_button.clicked.connect(self._on_stop_clicked)

        self.restart_button = QtWidgets.QToolButton(control_container)
        self.restart_button.setObjectName("restartButton")
        self.restart_button.setText("")
        self.restart_button.setProperty("iconTone", "black")
        self.restart_button.setIcon(
            standard_icon(
                self,
                QtWidgets.QStyle.StandardPixmap.SP_BrowserReload,
                "black",
            )
        )
        self.restart_button.clicked.connect(self._on_reset_clicked)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.restart_button)
        control_layout.addStretch(1)

        layout.addWidget(control_container)
        self._set_running_state(False)
        return panel

    def _build_image_list_panel(self) -> QtWidgets.QFrame:
        """建立影像清單 panel。"""
        panel = self._new_panel("imageListPanel", "2. 影像清單")
        layout = panel.layout()
        assert layout is not None

        self.image_list = QtWidgets.QListWidget(panel)
        self.image_list.setObjectName("folderImageList")
        self.image_list.currentRowChanged.connect(self._on_image_selection_changed)
        layout.addWidget(self.image_list, stretch=1)
        return panel

    def _create_hidden_results_table(self) -> None:
        """保留 cleaned CSV 與 cell highlight 邏輯使用的隱藏表格。"""
        self.results_table = QTableWidget(self)
        self.results_table.setObjectName("featureParameterTable")
        self.results_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.results_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Cell_ID", "Area", "Mean"])
        self.results_table.hide()

    def _build_area_scatter_panel(self) -> QtWidgets.QFrame:
        """建立細胞面積點狀分布圖 panel。"""
        panel = self._new_panel("areaScatterPanel", "3. 細胞面積點狀分布圖")
        self.area_scatter_title = panel.findChild(QLabel, "areaScatterPanelTitle")
        layout = panel.layout()
        assert layout is not None
        self.area_scatter_label = QLabel(panel)
        self.area_scatter_label.setObjectName("areaScatterLabel")
        self.area_scatter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.area_scatter_label.setText("")
        layout.addWidget(self.area_scatter_label, stretch=1)
        return panel

    def _build_area_histogram_panel(self) -> QtWidgets.QFrame:
        """建立細胞面積長條圖 panel。"""
        panel = self._new_panel("areaHistogramPanel", "4. 細胞面積長條圖")
        self.area_histogram_title = panel.findChild(QLabel, "areaHistogramPanelTitle")
        layout = panel.layout()
        assert layout is not None
        self.area_histogram_label = QLabel(panel)
        self.area_histogram_label.setObjectName("areaHistogramLabel")
        self.area_histogram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.area_histogram_label.setText("")
        layout.addWidget(self.area_histogram_label, stretch=1)
        return panel

    def _build_ui(self) -> None:
        """建立主視窗所有輸入、控制、影像與結果表元件。"""
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        self.input_dir_edit = QLineEdit(self)
        self.input_dir_edit.hide()
        self._create_hidden_results_table()

        self.btn_run = QPushButton("Run", self)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_run.hide()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self.image_panel = QWidget(self)
        self.image_panel.setObjectName("imagePanel")
        viewer_layout = QVBoxLayout(self.image_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(6)

        self.graphics_view = ZoomableGraphicsView(self)
        self.graphics_view.setMinimumSize(400, 300)
        self.graphics_view.setStyleSheet("background-color: #202020;")
        self._scene = QGraphicsScene(self)
        self.graphics_view.setScene(self._scene)
        self.image_view = self.graphics_view
        self.image_scene = self._scene
        viewer_layout.addWidget(self.graphics_view, stretch=1)

        self.image_header_widget = QWidget(self.image_panel)
        self.image_header_widget.setObjectName("imageHeaderRow")
        image_header_layout = QHBoxLayout(self.image_header_widget)
        image_header_layout.setContentsMargins(0, 0, 0, 0)
        image_header_layout.setSpacing(8)

        self.image_file_label = QLabel("Image File Name", self.image_header_widget)
        self.image_file_label.setObjectName("imageFileName")
        self.file_label = self.image_file_label
        image_header_layout.addWidget(self.image_file_label, stretch=1)

        overlay_controls = QtWidgets.QHBoxLayout()
        overlay_controls.setContentsMargins(0, 0, 0, 0)
        overlay_controls.setSpacing(6)
        self.chk_show_nuc = QtWidgets.QCheckBox("顯示核輪廓", self.image_header_widget)
        self.chk_show_nuc.setChecked(True)
        self.chk_show_cyto = QtWidgets.QCheckBox("顯示質輪廓", self.image_header_widget)
        self.chk_show_cyto.setChecked(True)
        self.chk_show_ki67 = QtWidgets.QCheckBox("顯示ki67", self.image_header_widget)
        self.chk_show_ki67.setChecked(False)

        self.alpha_slider = QtWidgets.QSlider(
            Qt.Orientation.Horizontal, self.image_header_widget
        )
        self.alpha_slider.setMinimum(10)  # 0.1
        self.alpha_slider.setMaximum(100)  # 1.0
        self.alpha_slider.setValue(int(self._overlay_alpha * 100))
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.alpha_slider.setFixedWidth(90)
        alpha_label = QtWidgets.QLabel("透明度", self.image_header_widget)

        overlay_controls.addWidget(self.chk_show_nuc)
        overlay_controls.addWidget(self.chk_show_cyto)
        overlay_controls.addWidget(self.chk_show_ki67)
        overlay_controls.addWidget(alpha_label)
        overlay_controls.addWidget(self.alpha_slider)
        image_header_layout.addLayout(overlay_controls)
        viewer_layout.addWidget(self.image_header_widget)

        self.main_splitter.addWidget(self.image_panel)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.right_splitter.setObjectName("rightPanelSplitter")

        self.right_splitter.addWidget(self._build_terminal_panel())
        self.right_splitter.addWidget(self._build_image_list_panel())
        self.right_splitter.addWidget(self._build_area_scatter_panel())
        self.right_splitter.addWidget(self._build_area_histogram_panel())

        for index in range(4):
            self.right_splitter.setStretchFactor(index, 1)
        self.right_splitter.setSizes([190, 155, 245, 235])
        self.main_splitter.addWidget(self.right_splitter)

        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([900, 450])

        root.addWidget(self.main_splitter)

        # 保留最後狀態訊息，不建立底部 QStatusBar。
        self._show_status_message("就緒")

        # 連接 overlay 控制訊號
        self.chk_show_nuc.toggled.connect(self._on_overlay_controls_changed)
        self.chk_show_cyto.toggled.connect(self._on_overlay_controls_changed)
        self.chk_show_ki67.toggled.connect(self._on_overlay_controls_changed)
        self.alpha_slider.valueChanged.connect(self._on_overlay_controls_changed)

    # --- 事件處理 ---

    def _show_status_message(self, message: str, timeout: int = 0) -> None:
        """記錄短狀態訊息，不在底部建立 QStatusBar。

        Args:
            message: 要保留的狀態訊息。
            timeout: 保留相容 Qt `showMessage` 呼叫的逾時參數。
        """
        _ = timeout
        self._last_status_message = message

    def _build_analysis_complete_dialog(self) -> QtWidgets.QDialog:
        """建立分析完成提示子視窗。"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("警告")
        dialog.setObjectName("analysisCompleteDialog")
        dialog.setModal(True)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(14)

        message = QLabel("分析完畢!!!", dialog)
        message.setObjectName("analysisCompleteMessage")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(message)

        button_row = QtWidgets.QWidget(dialog)
        button_row.setObjectName("analysisCompleteButtonRow")
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        confirm_button = QPushButton("確認", button_row)
        confirm_button.setObjectName("analysisCompleteConfirmButton")
        confirm_button.clicked.connect(dialog.accept)
        button_layout.addWidget(confirm_button, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addWidget(button_row)

        return dialog

    def _show_analysis_complete_dialog(self) -> None:
        """顯示分析完成提示子視窗。"""
        self._build_analysis_complete_dialog().exec()

    def _on_browse_input(self) -> None:
        """開啟資料夾選擇器並載入影像與 cleaned CSV。"""
        base = Path.cwd() / "data" / "input"
        start_dir = str(base) if base.exists() else ""
        directory = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾", start_dir)
        if directory:
            self.input_dir_edit.setText(directory)
            self._load_images_from_folder(Path(directory))
            # 嘗試載入對應 cleaned CSV
            self._load_cleaned_csv_for_dataset()

    def _on_run_clicked(self) -> None:
        """使用目前 UI 設定啟動背景 pipeline。"""
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            return

        path_text = self.input_dir_edit.text().strip()
        if not path_text:
            self._show_status_message("請先選擇輸入資料夾")
            return

        data_folder = Path(path_text)

        self._pipeline_thread = PipelineThread(
            data_folder,
            nuc_source=self._nuc_source,
            fluor_analy=self._fluor_analy,
            ki67=self._ki67,
            ki67_backend=self._ki67_backend,
            feature_backend=self._feature_backend,
            clean_temp=self._clean_temp,
            width_um_per_px=self._width_um_per_px,
            height_um_per_px=self._height_um_per_px,
            parent=self,
        )
        self._pipeline_thread.progress_changed.connect(self._on_progress_changed)
        self._pipeline_thread.finished_ok.connect(self._on_pipeline_finished)
        self._pipeline_thread.failed.connect(self._on_pipeline_failed)
        self._pipeline_thread.start()

        self._set_running_state(True)
        self._show_status_message("Pipeline 執行中...")

    def _on_stop_clicked(self) -> None:
        """停止目前背景 pipeline 執行緒。"""
        # 目前簡單呼叫 thread.terminate()，未來可實作更溫和的中止機制
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            self._pipeline_thread.terminate()
            self._pipeline_thread.wait()
            self._show_status_message("Pipeline 已中止")
            self._set_running_state(False)

    def _on_reset_clicked(self) -> None:
        """重設 UI 狀態、影像清單、結果表與 overlay。"""
        self.progress_bar.setValue(0)
        if hasattr(self, "terminal_output"):
            self.terminal_output.clear()
            self.terminal_output.append("[INFO] 等待開啟資料夾...")
        self.image_list.clear()
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.image_file_label.setText("Image File Name")
        self.area_scatter_label.clear()
        self.area_scatter_label.setText("")
        self.area_histogram_label.clear()
        self.area_histogram_label.setText("")
        self._area_scatter_pixmap = None
        self._area_histogram_pixmap = None
        self._pipeline_result = None
        self._current_image_index = None
        self._current_image_array = None
        self._current_overlay_polygons.clear()
        self._current_overlay_masks.clear()
        self._current_overlay_image = None
        self._current_overlay_image_array = None
        self._current_data_folder = None
        self._selected_cell_id = None
        self._highlight_enabled = False
        self._last_selected_row = None
        self._cleaned_csv_rows = []
        self._cleaned_csv_header = None
        self._scene.clear()
        self._set_running_state(False)
        self._show_status_message("已重設")

    def _append_terminal_line(self, level: str, message: str) -> None:
        """將訊息追加到終端輸出面板。"""
        if hasattr(self, "terminal_output"):
            self.terminal_output.append(f"[{level}] {message}")

    def _on_progress_changed(self, done: int, total: int, message: str) -> None:
        """更新進度條與狀態列文字。"""
        percent = int(done / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self._show_status_message(f"{message} ({done}/{total})")
        suffix = f" ({done}/{total})" if total > 0 else ""
        self._append_terminal_line("INFO", f"{message}{suffix}")

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
        """處理 pipeline 成功結束後的影像清單與結果載入。"""
        self._set_running_state(False)
        self.progress_bar.setValue(100)
        self._pipeline_result = result
        # 記錄目前資料夾
        self._current_data_folder = result.data_folder

        if result.paired_overlays:
            raw_cytoplasm_count = sum(
                data.raw_cytoplasm_count for data in result.paired_overlays.values()
            )
            paired_cytoplasm_count = sum(
                data.paired_cytoplasm_count for data in result.paired_overlays.values()
            )
            paired_nucleus_count = sum(
                data.paired_nucleus_count for data in result.paired_overlays.values()
            )
            raw_nucleus_count = sum(
                data.raw_nucleus_count for data in result.paired_overlays.values()
            )
            self._append_terminal_line(
                "INFO",
                "GUI 強制使用 Paired Only："
                f"cytoplasm {paired_cytoplasm_count}/{raw_cytoplasm_count}，"
                f"nucleus {paired_nucleus_count}/{raw_nucleus_count}",
            )

        try:
            saved_overlays = save_pipeline_fill_overlays(
                result, alpha=self._overlay_alpha
            )
            self._append_terminal_line(
                "INFO", f"已自動儲存 {len(saved_overlays)} 張 fill-overlay PNG"
            )
        except Exception as exc:  # noqa: BLE001
            self._append_terminal_line(
                "ERROR", f"自動儲存 fill-overlay 失敗：{exc}"
            )

        self._populate_image_list(result.image_files)

        # pipeline 結束後載入 cleaned CSV
        self._load_cleaned_csv_for_dataset()
        self._load_area_charts()

        self._show_status_message(
            f"Pipeline 完成，共處理 {len(result.image_files)} 張影像"
        )

        self._append_terminal_line(
            "INFO", f"Pipeline 完成，載入 {len(result.image_files)} 張影像"
        )
        self._show_analysis_complete_dialog()

    def _on_pipeline_failed(self, message: str) -> None:
        """處理 pipeline 失敗訊息並恢復 Run 按鈕。"""
        self._set_running_state(False)
        self._show_status_message(f"錯誤：{message}")

        self._append_terminal_line("ERROR", message)

    def _on_image_selection_changed(self, row: int) -> None:
        """切換目前選取影像並更新 overlay 與結果表。"""
        if (
            not self._pipeline_result
            or row < 0
            or row >= len(self._pipeline_result.image_files)
        ):
            return
        self._current_image_index = row
        img_path = self._pipeline_result.image_files[row]
        self._load_image_and_overlays(img_path)
        self._refresh_results_table_for_current_image()
        self._update_display_pixmap()

    def _load_image_and_overlays(self, img_path: Path) -> None:
        """載入原圖、segmentation masks 與 merged outlines。"""
        # 使用 OpenCV 讀圖，保留灰階或彩色
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            self._show_status_message(f"無法載入影像：{img_path}")
            self._current_image_array = None
            self._current_overlay_image_array = None
            self.image_file_label.setText("無法載入影像")
            return

        if img_bgr.ndim == 2:  # 灰階 -> 轉成 BGR
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        self._current_image_array = img_bgr
        self.image_file_label.setText(img_path.name)

        nuc_mask, cyto_mask = self._load_segmentation_masks_for_image(
            img_path, img_bgr.shape[:2]
        )
        if nuc_mask is not None or cyto_mask is not None:
            self._current_overlay_masks[img_path] = (nuc_mask, cyto_mask)
        else:
            self._current_overlay_masks.pop(img_path, None)

        paired_data = self._paired_overlay_data_for_image(img_path)

        # 嘗試載入 outlines
        merged_path = find_merged_outline_for_image(img_path)
        nuc_polys: list[np.ndarray] = []
        cyto_polys: list[np.ndarray] = []
        if merged_path is None:
            self._current_overlay_polygons.pop(img_path, None)
            if nuc_mask is None and cyto_mask is None and paired_data is None:
                self._current_overlay_image = None
                self._current_overlay_image_array = None
                self._show_status_message(f"沒有找到 outlines：{img_path.name}")
                return
        else:
            polygons = load_paired_merged_outlines(merged_path)
            nuc_polys = polygons.nuc_polygons
            cyto_polys = polygons.cyto_polygons
            self._current_overlay_polygons[img_path] = (nuc_polys, cyto_polys)

        overlay_bgr = self._create_overlay_bgr(
            img_bgr,
            nuc_polys,
            cyto_polys,
            nuc_mask=nuc_mask,
            cyto_mask=cyto_mask,
            paired_data=paired_data,
        )
        self._current_overlay_image_array = overlay_bgr
        self._current_overlay_image = self._pixmap_from_bgr(overlay_bgr)

    def _segment_output_dir_for_image(self, image_path: Path) -> Path:
        """推導指定影像對應的 segmentation 輸出資料夾。

        Args:
            image_path: 目前 GUI 顯示的原始影像路徑。

        Returns:
            `data/output/segment/<folder_name>` 類型的資料夾路徑。
        """
        if self._pipeline_result is not None:
            data_folder = Path(self._pipeline_result.data_folder)
            return data_folder.parent.parent / "output" / "segment" / data_folder.name
        if self._current_data_folder is not None:
            data_folder = Path(self._current_data_folder)
            return data_folder.parent.parent / "output" / "segment" / data_folder.name

        image_path = image_path.resolve()
        dataset_dir = (
            image_path.parent.parent
            if image_path.parent.name.lower() == "pc"
            else image_path.parent
        )
        if dataset_dir.parent.name.lower() == "input":
            return dataset_dir.parent.parent / "output" / "segment" / dataset_dir.name
        return Path("data") / "output" / "segment" / dataset_dir.name

    def _load_segmentation_mask_for_image(
        self,
        image_path: Path,
        suffix: str,
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """讀取 Cellpose segmentation label mask。

        Args:
            image_path: 原始影像路徑。
            suffix: `cyto` 或 `nuc`。
            image_shape: 原始影像的 `(height, width)`，用於必要時 resize。

        Returns:
            讀取成功時回傳 label mask；找不到或格式不符時回傳 `None`。
        """
        mask_path = (
            self._segment_output_dir_for_image(image_path)
            / f"{image_path.stem}_{suffix}_seg.npy"
        )
        if not mask_path.exists():
            return None
        try:
            data = np.load(mask_path, allow_pickle=True).item()
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(data, dict) or "masks" not in data:
            return None

        mask = np.asarray(data["masks"])
        height, width = image_shape
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask.astype(np.int32, copy=False)

    def _load_segmentation_masks_for_image(
        self,
        image_path: Path,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """讀取目前影像的 nucleus 與 cytoplasm segmentation masks。"""
        nuc_mask = self._load_segmentation_mask_for_image(
            image_path, "nuc", image_shape
        )
        cyto_mask = self._load_segmentation_mask_for_image(
            image_path, "cyto", image_shape
        )
        return nuc_mask, cyto_mask

    def _paired_overlay_data_for_image(
        self,
        image_path: Path,
    ) -> PairedOverlayData | None:
        """取得 pipeline 在暫存清理前保留的 SegUI 配對顯示資料。"""
        if self._pipeline_result is None:
            return None
        return self._pipeline_result.paired_overlays.get(image_path.stem)

    def _blend_label_regions(
        self,
        overlay: np.ndarray,
        label_mask: np.ndarray,
        color_for_label: Callable[[int], tuple[int, int, int]],
    ) -> None:
        """將 label mask 的每個物件以半透明顏色填入 overlay。"""
        labels = np.unique(label_mask)
        labels = labels[labels != 0]
        if labels.size == 0:
            return

        color_layer = np.zeros_like(overlay)
        paint_mask = np.zeros(overlay.shape[:2], dtype=bool)
        for label in labels:
            region = label_mask == label
            color_layer[region] = color_for_label(int(label))
            paint_mask |= region

        alpha = self._overlay_alpha
        overlay[paint_mask] = (
            alpha * color_layer[paint_mask] + (1 - alpha) * overlay[paint_mask]
        ).astype(np.uint8)

    def _draw_label_contours(
        self,
        overlay: np.ndarray,
        label_mask: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        """依 label mask 邊界繪製物件輪廓。"""
        labels = np.unique(label_mask)
        labels = labels[labels != 0]
        for label in labels:
            binary = (label_mask == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 1)

    def _apply_segmentation_mask_overlay(
        self,
        base_bgr: np.ndarray,
        nuc_mask: np.ndarray | None,
        cyto_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        """依 SegmentationUI 風格產生彩色 segmentation mask 疊圖。

        Args:
            base_bgr: OpenCV BGR 原圖。
            nuc_mask: nucleus label mask。
            cyto_mask: cytoplasm label mask。

        Returns:
            兩種 mask 都存在時回傳 Paired Only 疊圖；否則回傳 `None`。
        """
        if nuc_mask is None or cyto_mask is None:
            return None
        paired_data = build_paired_overlay_data(cyto_mask, nuc_mask)
        return render_paired_overlay_bgr(
            base_bgr,
            paired_data,
            show_nucleus=self._show_nuc,
            show_cytoplasm=self._show_cyto,
            alpha=self._overlay_alpha,
        )

    def _pixmap_from_bgr(self, bgr: np.ndarray) -> QPixmap:
        """將 OpenCV BGR 影像轉為 Qt QPixmap。"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _create_overlay_bgr(
        self,
        base_bgr: np.ndarray,
        nuc_polys: list[np.ndarray] | None,
        cyto_polys: list[np.ndarray] | None,
        nuc_mask: np.ndarray | None = None,
        cyto_mask: np.ndarray | None = None,
        paired_data: PairedOverlayData | None = None,
    ) -> np.ndarray:
        """根據目前設定，在 base 影像上畫出 mask 或輪廓，回傳 BGR 影像。"""
        if paired_data is not None:
            blended = render_paired_overlay_bgr(
                base_bgr,
                paired_data,
                show_nucleus=self._show_nuc,
                show_cytoplasm=self._show_cyto,
                alpha=self._overlay_alpha,
            )
        else:
            mask_overlay = self._apply_segmentation_mask_overlay(
                base_bgr, nuc_mask=nuc_mask, cyto_mask=cyto_mask
            )
            if mask_overlay is not None:
                blended = mask_overlay
            else:
                blended = self._create_outline_overlay_bgr(
                    base_bgr, nuc_polys=nuc_polys, cyto_polys=cyto_polys
                )

        if self._show_ki67:
            ki67_color = (203, 0, 255)  # 粉色
            for idx in self._ki67_positive_indices_for_current_image():
                if cyto_polys and 0 <= idx < len(cyto_polys):
                    pts = cyto_polys[idx].reshape(-1, 1, 2)
                elif nuc_polys and 0 <= idx < len(nuc_polys):
                    pts = nuc_polys[idx].reshape(-1, 1, 2)
                else:
                    continue
                cv2.polylines(
                    blended, [pts], isClosed=True, color=ki67_color, thickness=3
                )

        return blended

    def _create_outline_overlay_bgr(
        self,
        base_bgr: np.ndarray,
        nuc_polys: list[np.ndarray] | None,
        cyto_polys: list[np.ndarray] | None,
    ) -> np.ndarray:
        """使用 outline polygon 產生半透明填色的 fallback 疊圖。"""
        return render_fill_overlay_bgr(
            base_bgr,
            nuc_polys,
            cyto_polys,
            show_nuc=self._show_nuc,
            show_cyto=self._show_cyto,
            alpha=self._overlay_alpha,
        )

    def _create_overlay_pixmap(
        self,
        base_bgr: np.ndarray,
        nuc_polys: list[np.ndarray] | None,
        cyto_polys: list[np.ndarray] | None,
        nuc_mask: np.ndarray | None = None,
        cyto_mask: np.ndarray | None = None,
    ) -> QPixmap:
        """根據目前設定，在 base 影像上畫出疊圖，回傳 QPixmap。"""
        return self._pixmap_from_bgr(
            self._create_overlay_bgr(
                base_bgr,
                nuc_polys,
                cyto_polys,
                nuc_mask=nuc_mask,
                cyto_mask=cyto_mask,
            )
        )

    def _update_display_pixmap(self) -> None:
        """更新顯示影像。"""
        if self._current_image_array is None:
            return

        img_path = self._current_image_path()
        if img_path is None:
            self._set_pixmap_in_view(self._pixmap_from_bgr(self._current_image_array))
            return

        polys = self._current_overlay_polygons.get(img_path)
        masks = self._current_overlay_masks.get(img_path)
        if polys is None and masks is None:
            self._set_pixmap_in_view(self._pixmap_from_bgr(self._current_image_array))
            return

        nuc_polys, cyto_polys = polys if polys is not None else ([], [])
        nuc_mask, cyto_mask = masks if masks is not None else (None, None)
        paired_data = self._paired_overlay_data_for_image(img_path)
        display_bgr = self._create_overlay_bgr(
            self._current_image_array,
            nuc_polys,
            cyto_polys,
            nuc_mask=nuc_mask,
            cyto_mask=cyto_mask,
            paired_data=paired_data,
        )

        if self._highlight_enabled and self._selected_cell_id is not None:
            idx = self._cell_index_from_cell_id(self._selected_cell_id)
            if idx is not None:
                if 0 <= idx < len(cyto_polys):
                    cyto_pts = cyto_polys[idx].reshape(-1, 1, 2)
                    cv2.polylines(
                        display_bgr,
                        [cyto_pts],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=4,
                    )
                if 0 <= idx < len(nuc_polys):
                    nuc_pts = nuc_polys[idx].reshape(-1, 1, 2)
                    cv2.polylines(
                        display_bgr,
                        [nuc_pts],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=3,
                    )

        self._current_overlay_image_array = display_bgr
        self._set_pixmap_in_view(self._pixmap_from_bgr(display_bgr))

    def _set_pixmap_in_view(self, pixmap: QPixmap) -> None:
        """將 QPixmap 顯示到 GraphicsView 並自動 fit。"""
        self._scene.clear()
        if pixmap.isNull():
            return
        item = self._scene.addPixmap(pixmap)
        rect = QRectF(pixmap.rect())
        self._scene.setSceneRect(rect)
        # fit and reset transform so initial view fits
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """視窗尺寸改變時重新 fit 目前影像。"""
        super().resizeEvent(event)
        # 視窗縮放時重新 fit 一次當前圖（保持大致適配）
        if self._scene.items():
            item = self._scene.items()[0]
            self.graphics_view.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
        self._scale_area_chart_pixmaps()

    def _scale_chart_pixmap(self, pixmap: QPixmap | None, label: QLabel) -> None:
        """依照目前 label 大小縮放圖表。"""
        if pixmap is None or pixmap.isNull():
            return
        target_size = label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        label.setPixmap(
            pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _scale_area_chart_pixmaps(self) -> None:
        """縮放右側兩張細胞面積分析圖。"""
        self._scale_chart_pixmap(self._area_scatter_pixmap, self.area_scatter_label)
        self._scale_chart_pixmap(
            self._area_histogram_pixmap,
            self.area_histogram_label,
        )

    def _first_existing_path(self, candidates: list[Path]) -> Path | None:
        """回傳候選路徑中第一個存在的檔案。"""
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists():
                return candidate
        return None

    def _load_one_area_chart(
        self,
        chart_path: Path | None,
        label: QLabel,
        empty_text: str,
    ) -> QPixmap | None:
        """載入單張圖表並更新對應 QLabel。"""
        if chart_path is None:
            label.clear()
            label.setText(empty_text)
            return
        pixmap = QPixmap(str(chart_path))
        if pixmap.isNull():
            label.clear()
            label.setText(empty_text)
            return None
        label.setText("")
        self._scale_chart_pixmap(pixmap, label)
        return pixmap

    def _area_chart_candidates(self, filename: str) -> list[Path]:
        """建立指定圖檔名稱的候選路徑。"""
        candidates: list[Path] = []
        if self._pipeline_result is not None:
            result_path = getattr(
                self._pipeline_result,
                "area_scatter_plot"
                if filename == "all_cell_nucleus_area.png"
                else "area_histogram_plot",
                None,
            )
            if result_path is not None:
                candidates.append(Path(result_path))
        if self._current_data_folder is not None:
            dataset_name = self._current_data_folder.name
            candidates.append(
                self._current_data_folder.parent.parent
                / "output"
                / "results"
                / dataset_name
                / filename
            )
        return candidates

    def _load_area_charts(self) -> None:
        """載入細胞面積散點圖與長條圖。"""
        scatter_path = self._first_existing_path(
            self._area_chart_candidates("all_cell_nucleus_area.png")
        )
        histogram_path = self._first_existing_path(
            self._area_chart_candidates("all_log_cell_area_distribution.png")
        )

        self._area_scatter_pixmap = self._load_one_area_chart(
            scatter_path,
            self.area_scatter_label,
            "尚無細胞面積點狀分布圖",
        )
        self._area_histogram_pixmap = self._load_one_area_chart(
            histogram_path,
            self.area_histogram_label,
            "尚無細胞面積長條圖",
        )

    def _load_area_chart(self) -> None:
        """相容舊呼叫；改為載入兩張細胞面積圖。"""
        self._load_area_charts()

    def _on_overlay_controls_changed(self) -> None:
        """當 overlay checkbox / alpha / view mode 改變時刷新顯示。

        若目前正在高亮指定 cell，切換模式不應該取消高亮，只需重繪。
        """
        self._show_nuc = self.chk_show_nuc.isChecked()
        self._show_cyto = self.chk_show_cyto.isChecked()
        self._show_ki67 = self.chk_show_ki67.isChecked()
        self._overlay_alpha = self.alpha_slider.value() / 100.0
        self._update_display_pixmap()

    def _load_images_from_folder(self, raw_folder: Path) -> None:
        """從使用者選擇的資料夾載入影像列表（不跑 pipeline）。"""
        try:
            data_folder = _resolve_data_folder(raw_folder)
        except FileNotFoundError as e:
            self._show_status_message(str(e))
            return

        # 記錄目前資料夾，供載入 cleaned CSV 使用
        self._current_data_folder = data_folder

        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        pc_dir = data_folder / "PC"
        search_dir = pc_dir if pc_dir.exists() and pc_dir.is_dir() else data_folder

        image_files: list[Path] = []
        for p in sorted(search_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                image_files.append(p)

        result_dir = data_folder.parent.parent / "output" / "results" / data_folder.name
        scatter_path = result_dir / "all_cell_nucleus_area.png"
        histogram_path = result_dir / "all_log_cell_area_distribution.png"
        self._pipeline_result = PipelineResult(
            data_folder=data_folder,
            image_files=image_files,
            results_dir=result_dir,
            area_scatter_plot=scatter_path if scatter_path.exists() else None,
            area_histogram_plot=histogram_path if histogram_path.exists() else None,
        )

        self._populate_image_list(image_files)
        self._load_area_charts()

        if image_files:
            self.image_list.setCurrentRow(0)
            self._append_terminal_line("INFO", f"載入資料夾：{data_folder}")
            self._append_terminal_line("INFO", f"找到 {len(image_files)} 張影像")
            self._show_status_message(f"載入 {len(image_files)} 張影像")
        else:
            self._scene.clear()
            self._show_status_message("找不到可顯示的影像檔")

    def _populate_image_list(self, image_files: list[Path]) -> None:
        """根據給定檔案列表更新右側 QListWidget。"""
        self.image_list.clear()
        for p in image_files:
            self.image_list.addItem(p.name)
        if image_files:
            self.image_list.setCurrentRow(0)
        else:
            self._current_image_index = None
            self._current_image_array = None
            self._current_overlay_polygons.clear()
            self._current_overlay_masks.clear()
            self._current_overlay_image = None
            self._current_overlay_image_array = None
            self.image_file_label.setText("Image File Name")
            self._scene.clear()

    def _load_cleaned_csv_for_dataset(self) -> None:
        """嘗試載入 data/output/results/<dataset>/<dataset>_cleaned.csv 並填表"""
        if not self._current_data_folder:
            return
        dataset_name = self._current_data_folder.name
        csv_path = (
            self._current_data_folder.parent.parent
            / "output"
            / "results"
            / dataset_name
            / f"{dataset_name}_cleaned.csv"
        )
        if not csv_path.exists():
            self._show_status_message(f"找不到 cleaned CSV: {csv_path}", 5000)
            self._cleaned_csv_rows = []
            self._cleaned_csv_header = None
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self._update_display_pixmap()
            return

        try:
            text = csv_path.read_text(encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            self._show_status_message(f"讀取 cleaned CSV 失敗: {e}", 5000)
            return

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            self._show_status_message("cleaned CSV 為空", 3000)
            self._cleaned_csv_rows = []
            self._cleaned_csv_header = None
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return

        # 自動偵測分隔符（優先 tab，其次 comma，否則用 whitespace）
        sample = lines[0]
        if "\t" in sample:
            delim = "\t"
            header = lines[0].split(delim)
            data_rows = [ln.split(delim) for ln in lines[1:]]
        elif "," in sample:
            # 用 csv.reader 正式解析 comma（避免帶引號的欄位問題）
            reader = csv.reader(lines)
            rows = list(reader)
            header = rows[0]
            data_rows = rows[1:]
        else:
            # 空白分隔（你的貼文是這種顯示，但實檔多半是 tab）
            header = lines[0].split()
            data_rows = [ln.split() for ln in lines[1:]]

        self._cleaned_csv_header = header
        self._cleaned_csv_rows = data_rows

        self._refresh_results_table_for_current_image()
        self._update_display_pixmap()
        self._show_status_message(f"載入 cleaned CSV: {csv_path}", 3000)

    def _column_index(self, column_name: str) -> int | None:
        """在 cleaned CSV header 中尋找欄位索引。"""
        if not self._cleaned_csv_header:
            return None
        target = column_name.strip().lower()
        for idx, name in enumerate(self._cleaned_csv_header):
            if name.strip().lower() == target:
                return idx
        return None

    def _current_image_path(self) -> Path | None:
        """取得目前影像清單選取的影像路徑。"""
        if (
            self._pipeline_result is None
            or self._current_image_index is None
            or self._current_image_index < 0
            or self._current_image_index >= len(self._pipeline_result.image_files)
        ):
            return None
        return self._pipeline_result.image_files[self._current_image_index]

    def _cell_id_to_image_stem(self, cell_id: str) -> str | None:
        """由 Cell_ID 解析對應影像 stem。"""
        image_stem, sep, cell_number = cell_id.strip().rpartition("_")
        if sep and cell_number.isdigit() and image_stem:
            return image_stem
        return None

    def _cell_index_from_cell_id(self, cell_id: str) -> int | None:
        """由 Cell_ID 解析 0-based cell index。"""
        _, sep, cell_number = cell_id.strip().rpartition("_")
        if not sep:
            return None
        try:
            index = int(cell_number) - 1
        except ValueError:
            return None
        return index if index >= 0 else None

    def _row_matches_image(self, row: list[str], image_path: Path) -> bool:
        """判斷 cleaned CSV row 是否屬於目前影像。"""
        image_idx = self._column_index("Image")
        if image_idx is not None and image_idx < len(row):
            return row[image_idx].strip() == image_path.stem

        cell_idx = self._column_index("Cell_ID")
        if cell_idx is None:
            cell_idx = 0
        if cell_idx < len(row):
            image_stem = self._cell_id_to_image_stem(row[cell_idx])
            return image_stem == image_path.stem
        return False

    def _rows_for_current_image(self) -> list[list[str]]:
        """取得目前影像對應的 cleaned CSV rows。"""
        image_path = self._current_image_path()
        if image_path is None:
            return []
        return [
            row
            for row in self._cleaned_csv_rows
            if self._row_matches_image(row, image_path)
        ]

    def _is_positive_value(self, value: str) -> bool:
        """將文字或數值型 Ki67 欄位轉為布林陽性判斷。"""
        text = value.strip().lower()
        if text in {"true", "yes", "y", "positive", "pos"}:
            return True
        if text in {"", "false", "no", "n", "negative", "neg"}:
            return False
        try:
            return float(text) > 0
        except ValueError:
            return False

    def _ki67_positive_indices_for_current_image(self) -> set[int]:
        """取得目前影像中 Ki67 positive cell 的 0-based index 集合。"""
        ki67_idx = self._column_index("ki67_positive")
        if ki67_idx is None:
            return set()

        cell_idx = self._column_index("Cell_ID")
        if cell_idx is None:
            cell_idx = 0

        positive_indices: set[int] = set()
        for row in self._rows_for_current_image():
            if ki67_idx >= len(row) or cell_idx >= len(row):
                continue
            if not self._is_positive_value(row[ki67_idx]):
                continue
            cell_index = self._cell_index_from_cell_id(row[cell_idx])
            if cell_index is not None:
                positive_indices.add(cell_index)
        return positive_indices

    def _refresh_results_table_for_current_image(self) -> None:
        """依目前影像重建右側結果表格內容。"""
        self._selected_cell_id = None
        self._highlight_enabled = False
        self._last_selected_row = None

        self.results_table.blockSignals(True)
        try:
            self.results_table.clear()
            if not self._cleaned_csv_header:
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
                return

            rows = self._rows_for_current_image()
            self.results_table.setColumnCount(len(self._cleaned_csv_header))
            self.results_table.setRowCount(len(rows))
            self.results_table.setHorizontalHeaderLabels(self._cleaned_csv_header)

            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    self.results_table.setItem(r, c, QTableWidgetItem(str(val)))
        finally:
            self.results_table.blockSignals(False)

        self.results_table.resizeColumnsToContents()

    def _on_results_table_selection_changed(self) -> None:
        """點選表格列：

        - 第一次選到某個 Cell_ID：啟用高亮
        - 再次點同一個 Cell_ID（同一列）時：取消高亮
        """
        # 取得目前選取的 row
        sel = self.results_table.selectionModel()
        if sel is None or not sel.hasSelection():
            self._selected_cell_id = None
            self._highlight_enabled = False
            self._update_display_pixmap()
            return

        row = sel.selectedRows()[0].row()
        item = self.results_table.item(row, 0)  # Cell_ID 欄
        cell_id = item.text().strip() if item is not None else ""

        if not cell_id:
            self._selected_cell_id = None
            self._highlight_enabled = False
            self._update_display_pixmap()
            return

        # 如果同一個 Cell_ID 已經啟用高亮，再點一次同一列 => 關閉
        if self._highlight_enabled and self._selected_cell_id == cell_id:
            self._highlight_enabled = False
        else:
            self._selected_cell_id = cell_id
            self._highlight_enabled = True

        self._update_display_pixmap()
