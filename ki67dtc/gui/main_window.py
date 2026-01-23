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
        fluor_analy: bool,
        ki67: bool,
        clean_temp: bool,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._data_folder = data_folder
        self._fluor_analy = fluor_analy
        self._ki67 = ki67
        self._clean_temp = clean_temp

    def _progress_callback(self, done: int, total: int, message: str) -> None:
        self.progress_changed.emit(done, total, message)

    def run(self) -> None:  # type: ignore[override]
        try:
            result = run_pipeline(
                self._data_folder,
                fluor_analy=self._fluor_analy,
                ki67=self._ki67,
                clean_temp=self._clean_temp,
                progress_callback=self._progress_callback,
            )
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_factor = 1.15
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        angle = event.angleDelta().y()
        if angle == 0:
            return
        if angle > 0:
            factor = self._zoom_factor
        else:
            factor = 1.0 / self._zoom_factor
        self.scale(factor, factor)

    def reset_view(self):
        self.resetTransform()


class MainWindow(QMainWindow):
    """最小可用的主視窗骨架。

    - 左側：輸入資料夾路徑 + Run / Stop / Reset + 進度列
    - 右側：暫時留白，未來放影像 viewer 與統計資訊
    """

    def __init__(self):
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
        self._overlay_alpha: float = 0.5
        self._view_mode: str = "overlay"  # "raw" or "overlay"
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
        self.results_table.itemSelectionChanged.connect(
            self._on_results_table_selection_changed
        )

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

        # 分析選項同一行 + 右側 model 選擇
        options_row = QHBoxLayout()
        options_row.addWidget(self.chk_fluor)
        options_row.addWidget(self.chk_ki67)
        options_row.addWidget(self.chk_clean)

        options_row.addSpacing(16)

        options_row.addWidget(QLabel("CYTO model", self))
        self.cyto_model_combo = QtWidgets.QComboBox(self)
        options_row.addWidget(self.cyto_model_combo)

        options_row.addWidget(QLabel("NUC model", self))
        self.nuc_model_combo = QtWidgets.QComboBox(self)
        options_row.addWidget(self.nuc_model_combo)

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

        self.alpha_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(10)  # 0.1
        self.alpha_slider.setMaximum(100)  # 1.0
        self.alpha_slider.setValue(int(self._overlay_alpha * 100))
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        alpha_label = QtWidgets.QLabel("透明度")

        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItems(["原圖", "原圖 + 輪廓"])
        self.view_mode_combo.setCurrentIndex(1)

        overlay_controls.addWidget(self.chk_show_nuc)
        overlay_controls.addWidget(self.chk_show_cyto)
        overlay_controls.addWidget(alpha_label)
        overlay_controls.addWidget(self.alpha_slider)
        overlay_controls.addWidget(self.view_mode_combo)
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
        self.alpha_slider.valueChanged.connect(self._on_overlay_controls_changed)
        self.view_mode_combo.currentIndexChanged.connect(
            self._on_overlay_controls_changed
        )

        # 在 build_ui 結尾或建立完 combo 後載入 ./model 內容
        self._populate_model_combos()

    # --- 事件處理 ---

    def _on_browse_input(self) -> None:
        base = Path.cwd() / "data" / "input"
        start_dir = str(base) if base.exists() else ""
        directory = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾", start_dir)
        if directory:
            self.input_dir_edit.setText(directory)
            self._load_images_from_folder(Path(directory))
            # 嘗試載入對應 cleaned CSV
            self._load_cleaned_csv_for_dataset()

    def _on_run_clicked(self) -> None:
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            return

        path_text = self.input_dir_edit.text().strip()
        if not path_text:
            self.statusBar().showMessage("請先選擇輸入資料夾")
            return

        data_folder = Path(path_text)

        fluor_analy = self.chk_fluor.isChecked()
        ki67 = self.chk_ki67.isChecked()
        clean_temp = self.chk_clean.isChecked()

        self._pipeline_thread = PipelineThread(
            data_folder,
            fluor_analy=fluor_analy,
            ki67=ki67,
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
        # 目前簡單呼叫 thread.terminate()，未來可實作更溫和的中止機制
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            self._pipeline_thread.terminate()
            self._pipeline_thread.wait()
            self.statusBar().showMessage("Pipeline 已中止")
            self.btn_run.setEnabled(True)

    def _on_reset_clicked(self) -> None:
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
        self._scene.clear()
        self.statusBar().showMessage("已重設")

    def _on_progress_changed(self, done: int, total: int, message: str) -> None:
        percent = int(done / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.statusBar().showMessage(f"{message} ({done}/{total})")

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
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
        self.btn_run.setEnabled(True)
        self.statusBar().showMessage(f"錯誤：{message}")

    def _on_image_selection_changed(self, row: int) -> None:
        if (
            not self._pipeline_result
            or row < 0
            or row >= len(self._pipeline_result.image_files)
        ):
            return
        self._current_image_index = row
        img_path = self._pipeline_result.image_files[row]
        self._load_image_and_overlays(img_path)
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

    def _create_overlay_pixmap(
        self,
        base_bgr: np.ndarray,
        nuc_polys: list[np.ndarray] | None,
        cyto_polys: list[np.ndarray] | None,
    ) -> QPixmap:
        """根據目前設定，在 base 影像上畫出輪廓，回傳 QPixmap。"""
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

        # 轉成 QPixmap
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _update_display_pixmap(self) -> None:
        """更新顯示影像。"""
        if self._current_image_array is None:
            return

        def _pixmap_from_bgr(bgr: np.ndarray) -> QPixmap:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)

        # 若高亮啟用且有選中 cell -> 僅顯示該 cell（核+質）
        if (
            self._highlight_enabled
            and self._selected_cell_id is not None
            and self._current_image_index is not None
            and self._pipeline_result is not None
        ):
            img_path = self._pipeline_result.image_files[self._current_image_index]
            polys = self._current_overlay_polygons.get(img_path)

            if polys is None:
                self._set_pixmap_in_view(_pixmap_from_bgr(self._current_image_array))
                return

            nuc_polys, cyto_polys = polys

            # Cell_ID: "1_3" -> pair index = 3
            idx = -1
            parts = self._selected_cell_id.split("_")
            if len(parts) == 2:
                try:
                    idx = int(parts[1]) - 1
                except ValueError:
                    idx = -1

            if 0 <= idx < len(nuc_polys) and 0 <= idx < len(cyto_polys):
                bgr = self._current_image_array.copy()
                cyto_pts = cyto_polys[idx].reshape(-1, 1, 2)
                nuc_pts = nuc_polys[idx].reshape(-1, 1, 2)

                cv2.polylines(
                    bgr,
                    [cyto_pts],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=4,
                )
                cv2.polylines(
                    bgr,
                    [nuc_pts],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=3,
                )

                self._set_pixmap_in_view(_pixmap_from_bgr(bgr))
                return

            self._set_pixmap_in_view(_pixmap_from_bgr(self._current_image_array))
            return

        # === 未高亮時：依模式顯示 ===
        if self.view_mode_combo.currentIndex() == 0:  # 原圖
            self._set_pixmap_in_view(_pixmap_from_bgr(self._current_image_array))
            return

        # 原圖 + 輪廓
        if self._current_image_index is None or not self._pipeline_result:
            return

        img_path = self._pipeline_result.image_files[self._current_image_index]
        polys = self._current_overlay_polygons.get(img_path)

        if polys is None:
            self._set_pixmap_in_view(_pixmap_from_bgr(self._current_image_array))
            return

        nuc_polys, cyto_polys = polys
        pixmap = self._create_overlay_pixmap(
            self._current_image_array, nuc_polys, cyto_polys
        )
        self._set_pixmap_in_view(pixmap)

    def _set_pixmap_in_view(self, pixmap: QPixmap) -> None:
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
            return

        try:
            text = csv_path.read_text(encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"讀取 cleaned CSV 失敗: {e}", 5000)
            return

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            self.statusBar().showMessage("cleaned CSV 為空", 3000)
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

        self.results_table.clear()
        self.results_table.setColumnCount(len(header))
        self.results_table.setRowCount(len(data_rows))
        self.results_table.setHorizontalHeaderLabels(header)

        for r, row in enumerate(data_rows):
            for c, val in enumerate(row):
                self.results_table.setItem(r, c, QTableWidgetItem(str(val)))

        self.results_table.resizeColumnsToContents()
        self.statusBar().showMessage(f"載入 cleaned CSV: {csv_path}", 3000)

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

    def _populate_model_combos(self) -> None:
        """掃描 ./model 目錄，填入 CYTO/NUC model 下拉選單。

        本專案的模型目前看起來是「無副檔名的檔案」（例如 model_BDL3_label_dapi），
        因此這裡同時支援：
        - 常見副檔名模型檔
        - 無副檔名但檔名以 model_ 開頭的檔案
        """
        model_dir = Path.cwd() / "model"
        self.cyto_model_combo.clear()
        self.nuc_model_combo.clear()

        if not model_dir.exists() or not model_dir.is_dir():
            self.cyto_model_combo.addItem("(找不到 ./model)")
            self.nuc_model_combo.addItem("(找不到 ./model)")
            return

        exts = {".pth", ".pt", ".onnx", ".pkl", ".h5", ".ckpt"}

        files: list[Path] = []
        for p in sorted(model_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() in exts:
                files.append(p)
                continue
            # 無副檔名模型檔（例如 model_BDL3_label_dapi）
            if p.suffix == "" and p.name.lower().startswith("model_"):
                files.append(p)

        if not files:
            self.cyto_model_combo.addItem("(model 目錄無模型檔)")
            self.nuc_model_combo.addItem("(model 目錄無模型檔)")
            return

        for p in files:
            self.cyto_model_combo.addItem(p.name, str(p))
            self.nuc_model_combo.addItem(p.name, str(p))

        # 預設：若檔名含 cyto/nuc，嘗試自動選（找不到就維持第一個）
        for i in range(self.cyto_model_combo.count()):
            name = self.cyto_model_combo.itemText(i).lower()
            if "cyto" in name:
                self.cyto_model_combo.setCurrentIndex(i)
                break

        for i in range(self.nuc_model_combo.count()):
            name = self.nuc_model_combo.itemText(i).lower()
            if "nuc" in name or "nucleus" in name:
                self.nuc_model_combo.setCurrentIndex(i)
                break
