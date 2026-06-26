import os
import sys
import traceback
import builtins
import faulthandler
import functools

# Dump C-level stack trace to stderr on segfault/crash
faulthandler.enable()

# Make every print auto-flush so we don't lose buffered output on crash
print = functools.partial(builtins.print, flush=True)

print(f"[DEBUG] Python {sys.version}")
print(f"[DEBUG] Executable: {sys.executable}")

# Determine base path for normal and PyInstaller frozen modes
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    print(f"[DEBUG] Running as PyInstaller frozen app, MEIPASS={base_path}")
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
    print(f"[DEBUG] Running from source, base_path={base_path}")

# Add common DLL search paths for conda/Windows environments
dll_paths = [
    base_path,
    os.path.join(base_path, "Library", "bin"),
    os.path.join(base_path, "Library", "mingw-w64", "bin"),
    os.path.join(base_path, "Library", "usr", "bin"),
    os.path.join(base_path, "Scripts"),
    os.path.join(base_path, "bin"),
]

for p in dll_paths:
    if os.path.exists(p) and p not in os.environ["PATH"]:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        print(f"[DEBUG] Added to PATH: {p}")

# Allow duplicate OpenMP libraries (for PyTorch/NumPy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if getattr(sys, 'frozen', False):
    # Add MEIPASS to PATH for PyInstaller
    os.environ['PATH'] = base_path + os.pathsep + os.environ['PATH']

print("[DEBUG] Importing PyQt5...")
try:
    from PyQt5 import QtWidgets
    print("[DEBUG] PyQt5 imported OK")
except Exception as e:
    print(f"[ERROR] Failed to import PyQt5: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[DEBUG] Importing controller...")
try:
    from controller import MainWindow_controller
    print("[DEBUG] controller imported OK")
except Exception as e:
    print(f"[ERROR] Failed to import controller: {e}")
    traceback.print_exc()
    sys.exit(1)

print("[DEBUG] Importing torch...")
try:
    import torch
    print(f"[DEBUG] torch {torch.__version__} imported OK, CUDA available: {torch.cuda.is_available()}")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")
except Exception as e:
    print(f"[ERROR] Failed to import/configure torch: {e}")
    traceback.print_exc()
    sys.exit(1)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    print("[DEBUG] Creating QApplication...")
    app = QtWidgets.QApplication(sys.argv)

    print("[DEBUG] Creating MainWindow_controller...")
    try:
        window = MainWindow_controller()
    except Exception as e:
        print(f"[ERROR] Failed to create MainWindow_controller: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("[DEBUG] Showing window...")
    window.show()

    print("[DEBUG] Entering event loop...")
    exit_code = app.exec_()
    print(f"[DEBUG] App exited with code {exit_code}")
    sys.exit(exit_code)
