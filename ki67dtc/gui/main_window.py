from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QLabel,
    QFileDialog,
    QStatusBar,
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
    load_merged_outlines,
)


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
        self.setWindowTitle("Ki67 細胞影像分析 GUI (Early Prototype)")
        # 預設視窗更大
        self.resize(1400, 900)
        self._pipeline_thread: PipelineThread | None = None
        self._pipeline_result: PipelineResult | None = None
        self._current_image_index: int | None = None
        self._current_image_array: np.ndarray | None = None
        self._current_overlay_polygons: dict[
            Path, tuple[list[np.ndarray], list[np.ndarray]]
        ] = {}
        self._current_overlay_image: QPixmap | None = None
        self._show_nuc: bool = True
        self._show_cyto: bool = True
        self._show_ki67: bool = False
        self._overlay_alpha: float = 0.5
        self._current_data_folder: Path | None = None
        # 新增：目前選中的 Cell_ID（例如 "1_3"）
        self._selected_cell_id: str | None = None
        self._highlight_enabled: bool = False
        # 用來判斷「再次點同一列」
        self._last_selected_row: int | None = None
        self._cleaned_csv_rows: list[list[str]] = []
        self._cleaned_csv_header: list[str] | None = None
        self._build_ui()

        # selection change 只會在 selection 真的改變時觸發；再點同一列不一定會觸發
        # 因此用 clicked 事件確保每次點擊都能切換
        self.results_table.cellClicked.connect(self._on_results_table_cell_clicked)

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

    def _build_ui(self) -> None:
        """建立主視窗所有輸入、控制、影像與結果表元件。"""
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        # ===== 最上方：工具列表 =====
        top_panel = QWidget(self)
        top_layout = QVBoxLayout(top_panel)

        form = QFormLayout()
        self.input_dir_edit = QLineEdit(self)
        btn_browse_input = QPushButton("選擇輸入資料夾", self)
        btn_browse_input.clicked.connect(self._on_browse_input)

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_dir_edit)
        input_row.addWidget(btn_browse_input)
        form.addRow("輸入資料夾", input_row)

        self.chk_fluor = QCheckBox("螢光分析", self)
        self.chk_fluor.setChecked(True)
        self.chk_ki67 = QCheckBox("Ki67 分析", self)
        self.chk_ki67.setChecked(True)
        self.chk_clean = QCheckBox("清理暫存資料", self)
        self.chk_clean.setChecked(True)

        # 分析選項同一行
        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("核來源", self))
        self.nuc_source_combo = QtWidgets.QComboBox(self)
        self.nuc_source_combo.addItem("DAPI", "dapi")
        self.nuc_source_combo.addItem("PC", "pc")
        options_row.addWidget(self.nuc_source_combo)

        options_row.addSpacing(16)
        options_row.addWidget(QLabel("特徵後端", self))
        self.feature_backend_combo = QtWidgets.QComboBox(self)
        self.feature_backend_combo.addItem("PyImageJ", "pyimagej")
        self.feature_backend_combo.addItem("Python", "python")
        options_row.addWidget(self.feature_backend_combo)

        options_row.addSpacing(16)
        options_row.addWidget(QLabel("Ki67 後端", self))
        self.ki67_backend_combo = QtWidgets.QComboBox(self)
        self.ki67_backend_combo.addItem("PyImageJ", "pyimagej")
        self.ki67_backend_combo.addItem("OpenCV", "opencv")
        options_row.addWidget(self.ki67_backend_combo)

        options_row.addSpacing(16)

        options_row.addWidget(self.chk_fluor)
        options_row.addWidget(self.chk_ki67)
        options_row.addWidget(self.chk_clean)

        options_row.addStretch(1)
        form.addRow("分析選項", options_row)

        top_layout.addLayout(form)

        controls_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run", self)
        self.btn_stop = QPushButton("Stop", self)
        self.btn_reset = QPushButton("Reset", self)

        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_reset.clicked.connect(self._on_reset_clicked)

        controls_layout.addWidget(self.btn_run)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_reset)
        top_layout.addLayout(controls_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        top_layout.addWidget(self.progress_bar)

        root.addWidget(top_panel, stretch=0)

        # ===== 下方：用 splitter 嚴格控制 2/3 vs 1/3 以及右側上下比例 =====
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # 左側 viewer（含 overlay 控制）
        viewer_panel = QWidget(self)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_view = ZoomableGraphicsView(self)
        self.graphics_view.setMinimumSize(400, 300)
        self.graphics_view.setStyleSheet("background-color: #202020;")
        self._scene = QGraphicsScene(self)
        self.graphics_view.setScene(self._scene)
        viewer_layout.addWidget(self.graphics_view, stretch=1)

        overlay_controls = QtWidgets.QHBoxLayout()
        self.chk_show_nuc = QtWidgets.QCheckBox("顯示核輪廓")
        self.chk_show_nuc.setChecked(True)
        self.chk_show_cyto = QtWidgets.QCheckBox("顯示質輪廓")
        self.chk_show_cyto.setChecked(True)
        self.chk_show_ki67 = QtWidgets.QCheckBox("顯示ki67")
        self.chk_show_ki67.setChecked(False)

        self.alpha_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(10)  # 0.1
        self.alpha_slider.setMaximum(100)  # 1.0
        self.alpha_slider.setValue(int(self._overlay_alpha * 100))
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        alpha_label = QtWidgets.QLabel("透明度")

        overlay_controls.addWidget(self.chk_show_nuc)
        overlay_controls.addWidget(self.chk_show_cyto)
        overlay_controls.addWidget(self.chk_show_ki67)
        overlay_controls.addWidget(alpha_label)
        overlay_controls.addWidget(self.alpha_slider)
        overlay_controls.addStretch(1)
        viewer_layout.addLayout(overlay_controls)

        bottom_splitter.addWidget(viewer_panel)

        # 右側：上下 splitter（上 image list / 下 results table）
        right_splitter = QSplitter(Qt.Orientation.Vertical, self)

        self.image_list = QtWidgets.QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selection_changed)
        right_splitter.addWidget(self.image_list)

        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.results_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        # 注意：不要再用 currentCellChanged 綁到 selection handler，避免重複觸發/行為不一致
        right_splitter.addWidget(self.results_table)

        # 右側上下各半（可再依 layout.png 微調）
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        bottom_splitter.addWidget(right_splitter)

        # 左右 2/3 : 1/3
        bottom_splitter.setStretchFactor(0, 2)
        bottom_splitter.setStretchFactor(1, 1)

        root.addWidget(bottom_splitter, stretch=1)

        # 狀態列
        status = QStatusBar(self)
        self.setStatusBar(status)
        self.statusBar().showMessage("就緒")

        # 連接 overlay 控制訊號
        self.chk_show_nuc.toggled.connect(self._on_overlay_controls_changed)
        self.chk_show_cyto.toggled.connect(self._on_overlay_controls_changed)
        self.chk_show_ki67.toggled.connect(self._on_overlay_controls_changed)
        self.alpha_slider.valueChanged.connect(self._on_overlay_controls_changed)

    # --- 事件處理 ---

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
            self.statusBar().showMessage("請先選擇輸入資料夾")
            return

        data_folder = Path(path_text)

        nuc_source = str(self.nuc_source_combo.currentData() or "dapi")
        feature_backend = str(
            self.feature_backend_combo.currentData() or "pyimagej"
        )
        ki67_backend = str(
            self.ki67_backend_combo.currentData() or "pyimagej"
        )
        fluor_analy = self.chk_fluor.isChecked()
        ki67 = self.chk_ki67.isChecked()
        clean_temp = self.chk_clean.isChecked()

        self._pipeline_thread = PipelineThread(
            data_folder,
            nuc_source=nuc_source,
            fluor_analy=fluor_analy,
            ki67=ki67,
            ki67_backend=ki67_backend,
            feature_backend=feature_backend,
            clean_temp=clean_temp,
            parent=self,
        )
        self._pipeline_thread.progress_changed.connect(self._on_progress_changed)
        self._pipeline_thread.finished_ok.connect(self._on_pipeline_finished)
        self._pipeline_thread.failed.connect(self._on_pipeline_failed)
        self._pipeline_thread.start()

        self.btn_run.setEnabled(False)
        self.statusBar().showMessage("Pipeline 執行中...")

    def _on_stop_clicked(self) -> None:
        """停止目前背景 pipeline 執行緒。"""
        # 目前簡單呼叫 thread.terminate()，未來可實作更溫和的中止機制
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            self._pipeline_thread.terminate()
            self._pipeline_thread.wait()
            self.statusBar().showMessage("Pipeline 已中止")
            self.btn_run.setEnabled(True)

    def _on_reset_clicked(self) -> None:
        """重設 UI 狀態、影像清單、結果表與 overlay。"""
        self.progress_bar.setValue(0)
        self.image_list.clear()
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self._pipeline_result = None
        self._current_image_index = None
        self._current_image_array = None
        self._current_overlay_polygons.clear()
        self._current_overlay_image = None
        self._current_data_folder = None
        self._selected_cell_id = None
        self._highlight_enabled = False
        self._last_selected_row = None
        self._cleaned_csv_rows = []
        self._cleaned_csv_header = None
        self._scene.clear()
        self.statusBar().showMessage("已重設")

    def _on_progress_changed(self, done: int, total: int, message: str) -> None:
        """更新進度條與狀態列文字。"""
        percent = int(done / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.statusBar().showMessage(f"{message} ({done}/{total})")

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
        """處理 pipeline 成功結束後的影像清單與結果載入。"""
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self._pipeline_result = result
        # 記錄目前資料夾
        self._current_data_folder = result.data_folder

        self._populate_image_list(result.image_files)

        # pipeline 結束後載入 cleaned CSV
        self._load_cleaned_csv_for_dataset()

        self.statusBar().showMessage(
            f"Pipeline 完成，共處理 {len(result.image_files)} 張影像"
        )

    def _on_pipeline_failed(self, message: str) -> None:
        """處理 pipeline 失敗訊息並恢復 Run 按鈕。"""
        self.btn_run.setEnabled(True)
        self.statusBar().showMessage(f"錯誤：{message}")

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
        """載入原圖為 numpy 陣列，並嘗試載入對應的 merged outlines。"""
        # 使用 OpenCV 讀圖，保留灰階或彩色
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            self.statusBar().showMessage(f"無法載入影像：{img_path}")
            self._current_image_array = None
            self.image_label.setText("無法載入影像")
            return

        if img_bgr.ndim == 2:  # 灰階 -> 轉成 BGR
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        self._current_image_array = img_bgr

        # 嘗試載入 outlines
        merged_path = find_merged_outline_for_image(img_path)
        if merged_path is None:
            self._current_overlay_polygons.pop(img_path, None)
            self._current_overlay_image = None
            self.statusBar().showMessage(f"沒有找到 outlines：{img_path.name}")
            return

        polygons = load_merged_outlines(merged_path)
        self._current_overlay_polygons[img_path] = (
            polygons.nuc_polygons,
            polygons.cyto_polygons,
        )
        # 先產生一張預設 overlay（兩者皆顯示）
        self._current_overlay_image = self._create_overlay_pixmap(
            img_bgr, polygons.nuc_polygons, polygons.cyto_polygons
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
    ) -> np.ndarray:
        """根據目前設定，在 base 影像上畫出輪廓，回傳 BGR 影像。"""
        overlay = base_bgr.copy()

        # 顏色：BGR
        nuc_color = (255, 0, 0)  # 藍色
        cyto_color = (0, 0, 255)  # 紅色
        thickness = 1

        if nuc_polys and self._show_nuc:
            for poly in nuc_polys:
                pts = poly.reshape(-1, 1, 2)
                cv2.polylines(
                    overlay, [pts], isClosed=True, color=nuc_color, thickness=thickness
                )

        if cyto_polys and self._show_cyto:
            for poly in cyto_polys:
                pts = poly.reshape(-1, 1, 2)
                cv2.polylines(
                    overlay, [pts], isClosed=True, color=cyto_color, thickness=thickness
                )

        alpha = self._overlay_alpha
        blended = cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)

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

    def _create_overlay_pixmap(
        self,
        base_bgr: np.ndarray,
        nuc_polys: list[np.ndarray] | None,
        cyto_polys: list[np.ndarray] | None,
    ) -> QPixmap:
        """根據目前設定，在 base 影像上畫出輪廓，回傳 QPixmap。"""
        return self._pixmap_from_bgr(
            self._create_overlay_bgr(base_bgr, nuc_polys, cyto_polys)
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
        if polys is None:
            self._set_pixmap_in_view(self._pixmap_from_bgr(self._current_image_array))
            return

        nuc_polys, cyto_polys = polys
        display_bgr = self._create_overlay_bgr(
            self._current_image_array, nuc_polys, cyto_polys
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
            self.statusBar().showMessage(str(e))
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

        self._pipeline_result = PipelineResult(
            data_folder=data_folder, image_files=image_files
        )

        self._populate_image_list(image_files)

        if image_files:
            self.image_list.setCurrentRow(0)
            self.statusBar().showMessage(f"載入 {len(image_files)} 張影像")
        else:
            self._scene.clear()
            self.statusBar().showMessage("找不到可顯示的影像檔")

    def _populate_image_list(self, image_files: list[Path]) -> None:
        """根據給定檔案列表更新右側 QListWidget。"""
        self.image_list.clear()
        for p in image_files:
            self.image_list.addItem(p.name)
        if image_files:
            self.image_list.setCurrentRow(0)

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
            self.statusBar().showMessage(f"找不到 cleaned CSV: {csv_path}", 5000)
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
            self.statusBar().showMessage(f"讀取 cleaned CSV 失敗: {e}", 5000)
            return

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            self.statusBar().showMessage("cleaned CSV 為空", 3000)
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
        self.statusBar().showMessage(f"載入 cleaned CSV: {csv_path}", 3000)

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
