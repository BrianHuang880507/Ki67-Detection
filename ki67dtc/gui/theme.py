"""SegmentationUI 深色主題樣式。"""

APP_QSS = """
QMainWindow,
QWidget {
    background-color: #1C2030;
    color: #E8EDF5;
    font-family: "Segoe UI";
    font-size: 11pt;
}

QMenuBar {
    background-color: #151927;
    color: #E8EDF5;
    border-bottom: 1px solid #2A3142;
}

QMenuBar::item {
    background: transparent;
    padding: 6px 10px;
}

QMenuBar::item:selected,
QMenu::item:selected {
    background-color: #00AEEF;
    color: #071018;
}

QMenu {
    background-color: #202638;
    color: #E8EDF5;
    border: 1px solid #30384C;
}

QMenu::item {
    padding: 6px 24px 6px 12px;
}

QFrame#terminalPanel,
QFrame#imageListPanel,
QFrame#featureTablePanel,
QFrame#areaChartPanel {
    background-color: #242B3D;
    border: 1px solid #344057;
    border-radius: 6px;
}

QTextEdit#terminalOutput {
    background-color: #0F1320;
    color: #A8FFDC;
    border: 1px solid #344057;
    border-radius: 4px;
    font-family: "Consolas";
}

QListWidget#folderImageList,
QTableWidget#featureParameterTable {
    background-color: #181D2B;
    color: #E8EDF5;
    border: 1px solid #344057;
    gridline-color: #344057;
    selection-background-color: #00AEEF;
    selection-color: #071018;
}

QToolButton {
    background-color: #E8EDF5;
    color: #111827;
    border: 1px solid #C6D0DF;
    border-radius: 4px;
    padding: 6px;
}

QToolButton:hover {
    background-color: #00AEEF;
    color: #071018;
    border-color: #00AEEF;
}

QToolButton:pressed {
    background-color: #0089C0;
    border-color: #0089C0;
}

QLabel#areaChartLabel {
    color: #E8EDF5;
    font-size: 12pt;
    font-weight: 600;
}
"""
