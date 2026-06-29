"""提供 GUI 標準圖示建立與套色工具。"""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyle, QWidget


ICON_SIZE = QSize(20, 20)


def standard_icon(widget: QWidget, pixmap: QStyle.StandardPixmap, color: str) -> QIcon:
    """建立套用指定顏色的 Qt 標準圖示。

    Args:
        widget: 用來取得目前樣式與標準圖示的 QWidget。
        pixmap: Qt 內建標準圖示類型。
        color: 要套用到圖示上的顏色名稱或色碼。

    Returns:
        已套色完成的 QIcon。
    """
    base = widget.style().standardIcon(pixmap).pixmap(ICON_SIZE)
    tinted = QPixmap(base.size())
    tinted.fill(Qt.GlobalColor.transparent)

    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, base)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), QColor(color))
    painter.end()

    return QIcon(tinted)
