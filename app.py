import sys
import os

if os.name == "nt":
    # Work around OpenMP runtime conflict on Windows (libomp vs libiomp5md).
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PyQt6.QtWidgets import QApplication

from ki67dtc.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
