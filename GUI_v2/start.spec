# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# Prevent OpenMP DLL conflict (PyTorch libiomp5md.dll vs OpenBLAS)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === Collect all binaries + data + hidden imports for each major dependency ===
cv2_datas,        cv2_binaries,        cv2_hiddenimports        = collect_all('cv2')
scipy_datas,      scipy_binaries,      scipy_hiddenimports      = collect_all('scipy')
tifffile_datas,   tifffile_binaries,   tifffile_hiddenimports   = collect_all('tifffile')
imageio_datas,    imageio_binaries,    imageio_hiddenimports    = collect_all('imageio')
skimage_datas,    skimage_binaries,    skimage_hiddenimports    = collect_all('skimage')
cellpose_datas,   cellpose_binaries,   cellpose_hiddenimports   = collect_all('cellpose')
torch_datas,      torch_binaries,      torch_hiddenimports      = collect_all('torch')
torchvision_datas, torchvision_binaries, torchvision_hiddenimports = collect_all('torchvision')

a = Analysis(
    ['start.py'],
    pathex=['.'],
    binaries=(
        cv2_binaries + scipy_binaries + tifffile_binaries + imageio_binaries
        + skimage_binaries + cellpose_binaries + torch_binaries + torchvision_binaries
    ),
    datas=[
        ('model', 'model'),
    ] + cv2_datas + scipy_datas + tifffile_datas + imageio_datas
      + skimage_datas + cellpose_datas + torch_datas + torchvision_datas,
    hiddenimports=[
        # Local modules
        'controller',
        'overlay_utils',
        'main_test2',
        'CellimageSegmentation_v6',
        'ki67dtc',
        'ki67dtc.cell_anal',
        'ki67dtc.cell_anal_plot',
        'ki67dtc.img_prep',
        'ki67dtc.utils',
        'ki67dtc.utils.io',
        # Data / analysis
        'pandas',
        'pandas._libs',
        'pandas._libs.tslibs',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.timedeltas',
        'natsort',
        'shapely',
        'shapely.geometry',
        'tqdm',
        'tqdm.auto',
        # Plotting
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        'matplotlib.figure',
        # PyQt5
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtSvg',
        'PyQt5.sip',
        # multiprocessing (used in start.py)
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.managers',
        'multiprocessing.forkserver',
        'multiprocessing.popen_spawn_win32',
    ] + cv2_hiddenimports + scipy_hiddenimports + tifffile_hiddenimports
      + imageio_hiddenimports + skimage_hiddenimports + cellpose_hiddenimports
      + torch_hiddenimports + torchvision_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ITRICytoScope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ITRICytoScope',
)
