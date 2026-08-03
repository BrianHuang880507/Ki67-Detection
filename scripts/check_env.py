"""環境健檢：確認 Ki67-Detection 的原生相依沒有互相衝突。

用法：
    python scripts/check_env.py

背景：conda 的 MKL 與 pip 安裝的 PyTorch 都需要一個叫 libiomp5md.dll 的檔案，
而 Windows 每個行程只能載入一份同名模組。兩者相爭時，torch 匯入會失敗並丟出：

    [WinError 127] Error loading "...\\torch\\lib\\shm.dll" or one of its dependencies.

這個錯誤其實和 shm.dll 無關 —— shm.dll 只是 torch\\lib 裡字母序最前面、
且相依於 torch_cpu.dll 的檔案，所以由它代為報錯。

衝突嚴重時整個行程會直接被作業系統終止（延遲載入例外 0xC06D007F），連
traceback 都沒有，因此本腳本把會觸發問題的匯入放進子行程執行，父行程只讀
子行程回報的進度，這樣就算子行程當掉也還能指出死在哪一步。設計理由見
environment.yml 的註解。
"""

import json
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    # 終端機不一定能表示所有字元，換成可替代模式以免健檢自己噴 UnicodeEncodeError。
    sys.stdout.reconfigure(errors="backslashreplace")

_failures = []

# 子行程被作業系統終止時的常見狀態碼。0xC06D007F 是 MSVC 延遲載入機制專用的
# 例外，代表 DLL 找到了但裡面缺少需要的函式 —— 正是同名 DLL 被別人搶走的徵狀。
_CRASH_CODES = {
    0xC06D007F: "延遲載入失敗：DLL 有載到，但裡面缺少需要的函式（ERROR_PROC_NOT_FOUND）",
    0xC06D007E: "延遲載入失敗：找不到 DLL（ERROR_MOD_NOT_FOUND）",
    0xC0000005: "存取違規（ACCESS_VIOLATION）",
    0xC0000135: "找不到相依的 DLL（STATUS_DLL_NOT_FOUND）",
    0xC0000139: "DLL 缺少進入點（STATUS_ENTRYPOINT_NOT_FOUND）",
}


def _ok(message: str) -> None:
    print(f"  [OK]   {message}", flush=True)


def _fail(message: str) -> None:
    print(f"  [FAIL] {message}", flush=True)
    _failures.append(message)


def _note(message: str) -> None:
    print(f"         {message}", flush=True)


# 在子行程執行的探測程式。每完成一步就送出一行 KI67| 標記，父行程據此判斷
# 進行到哪裡；若子行程被作業系統終止，最後一個標記就是斷點。
_PROBE = r'''
import json, os, sys


def emit(tag, **payload):
    sys.stdout.write("KI67|" + tag + "|" + json.dumps(payload) + "\n")
    sys.stdout.flush()


def loaded_modules():
    if os.name != "nt":
        return []
    import ctypes
    import ctypes.wintypes as wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # GetCurrentProcess 回傳 64 位元偽 handle，restype 若沿用預設 c_int 會被截斷。
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    psapi.EnumProcessModules.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, wintypes.DWORD,
    ]

    handle = kernel32.GetCurrentProcess()
    slots = (ctypes.c_void_p * 4096)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModules(handle, slots, ctypes.sizeof(slots), ctypes.byref(needed)):
        return []
    count = needed.value // ctypes.sizeof(ctypes.c_void_p)
    name = ctypes.create_unicode_buffer(1024)
    return [
        (psapi.GetModuleFileNameExW(handle, slots[i], name, 1024), name.value)[1]
        for i in range(count)
    ]


import numpy as np
emit("numpy", version=np.__version__)

# 真的呼叫一次 LAPACK，強迫 BLAS 後端載入。skimage.color 匯入時就會做同樣的
# 事，那是當初把 MKL 帶進行程的源頭。
from scipy import linalg
linalg.inv(np.eye(3))
names = [os.path.basename(p) for p in loaded_modules()]
emit(
    "blas",
    mkl=sorted({n for n in names if n.lower().startswith("mkl_")}),
    openblas=sorted({n for n in names if "openblas" in n.lower()}),
)

import skimage.color
emit("skimage")

from cellpose import io, models
emit("cellpose")

import torch
emit(
    "torch",
    version=torch.__version__,
    cuda=torch.cuda.is_available(),
    threads=torch.get_num_threads(),
)

known = {"libiomp5md.dll", "libomp.dll", "vcomp140.dll", "libgomp-1.dll"}
names = [os.path.basename(p) for p in loaded_modules()]
emit("openmp", runtimes=sorted({n for n in names if n.lower() in known}))

import imagej
emit("imagej")

emit("done")
'''


def check_interpreter() -> None:
    print("\n[1] 直譯器")
    _note(f"executable : {sys.executable}")
    _note(f"version    : {sys.version.split()[0]}")
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        _fail(
            f"Python {sys.version_info.major}.{sys.version_info.minor} 不在支援範圍 3.10-3.12"
        )
    else:
        _ok("Python 版本在支援範圍內")


def check_files_on_disk() -> None:
    """只看檔案不載入，因此即使環境已壞掉也一定跑得完。"""
    print("\n[2] 環境檔案（靜態檢查）")
    prefix = sys.prefix
    library_bin = os.path.join(prefix, "Library", "bin")
    torch_lib = os.path.join(prefix, "Lib", "site-packages", "torch", "lib")

    mkl = []
    if os.path.isdir(library_bin):
        mkl = sorted(
            name
            for name in os.listdir(library_bin)
            if name.lower().startswith("mkl_") and name.lower().endswith(".dll")
        )
    if mkl:
        _fail(f"環境內含 MKL（{len(mkl)} 個 DLL，例如 {mkl[0]}）")
        _note('修正：conda install -n ki67dtc "libblas=*=*openblas" -c conda-forge')
    else:
        _ok("環境內沒有 MKL")

    copies = [
        path
        for path in (
            os.path.join(library_bin, "libiomp5md.dll"),
            os.path.join(torch_lib, "libiomp5md.dll"),
        )
        if os.path.isfile(path)
    ]
    if len(copies) > 1:
        message = "環境裡有兩份 libiomp5md.dll，Windows 只會載入其中一份"
        if mkl:
            _fail(message)
        else:
            # MKL 已移除就沒有人會去要 conda 那份，torch 會載到自己的副本。
            _ok(message + "（但 MKL 已移除，不會有人搶）")
        for path in copies:
            _note(f"{os.path.getsize(path):>9,} bytes  {path}")
    elif copies:
        _ok(f"只有一份 libiomp5md.dll：{copies[0]}")


def check_import_chain() -> None:
    print("\n[3] 匯入鏈（當初崩潰的順序，於子行程執行）")
    proc = subprocess.run(
        [sys.executable, "-u", "-c", _PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    steps = {}
    for line in proc.stdout.splitlines():
        if line.startswith("KI67|"):
            _, tag, payload = line.split("|", 2)
            steps[tag] = json.loads(payload)

    if "numpy" in steps:
        version = steps["numpy"]["version"]
        if tuple(int(part) for part in version.split(".")[:2]) >= (2, 1):
            _fail(f"numpy {version} 過新，cellpose 3.1.1.1 需要 <2.1")
        else:
            _ok(f"numpy {version} 符合 cellpose 的 <2.1 限制")

    if "blas" in steps:
        blas = steps["blas"]
        if blas["mkl"]:
            _fail(f"呼叫 LAPACK 後載入了 MKL：{'、'.join(blas['mkl'])}")
        elif blas["openblas"]:
            _ok(f"BLAS 後端為 OpenBLAS（{'、'.join(blas['openblas'])}）")
        else:
            _note("未偵測到具名的 BLAS 模組，可能是靜態連結，非錯誤")

    for tag, label in (
        ("skimage", "skimage.color"),
        ("cellpose", "cellpose"),
        ("torch", "torch"),
        ("imagej", "imagej（未啟動 JVM）"),
    ):
        if tag in steps:
            _ok(f"{label} 匯入成功")

    if "torch" in steps:
        torch_info = steps["torch"]
        _note(f"torch {torch_info['version']}")
        _note(f"CUDA 可用 : {torch_info['cuda']}")
        _note(f"執行緒數  : {torch_info['threads']}")

    if "openmp" in steps:
        runtimes = steps["openmp"]["runtimes"]
        if not runtimes:
            _note("未偵測到 OpenMP 執行期")
        else:
            _note(f"OpenMP 執行期：{'、'.join(runtimes)}")
            if {"libiomp5md.dll", "libomp.dll"} <= set(runtimes):
                _fail("同時載入 Intel 與 LLVM 的 OpenMP 執行期，行為未定義")
            else:
                _ok("沒有偵測到 OpenMP 執行期衝突")

    if "done" in steps:
        return

    order = ["numpy", "blas", "skimage", "cellpose", "torch", "openmp", "imagej"]
    completed = [tag for tag in order if tag in steps]
    stalled = order[len(completed)] if len(completed) < len(order) else "done"
    code = proc.returncode & 0xFFFFFFFF
    _fail(f"匯入鏈中斷於 {stalled}（子行程 exit code 0x{code:08X}）")
    if code in _CRASH_CODES:
        _note(_CRASH_CODES[code])

    stderr = proc.stderr.strip()
    if stderr:
        for line in stderr.splitlines()[-6:]:
            _note(line)
    else:
        _note("子行程沒有留下 traceback，屬於作業系統層級的 DLL 載入失敗")
        _note("典型成因是 MKL 與 PyTorch 搶 libiomp5md.dll，見上方靜態檢查")

    if not os.environ.get("CONDA_PREFIX"):
        _note("提醒：CONDA_PREFIX 未設定，代表沒有先 conda/mamba activate")


def main() -> int:
    print("=" * 70)
    print("Ki67-Detection 環境健檢")
    print("=" * 70)

    for check in (check_interpreter, check_files_on_disk, check_import_chain):
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            _fail(f"{check.__name__} 執行時發生非預期錯誤：{exc}")

    print("\n" + "=" * 70)
    if _failures:
        print(f"不合格 —— {len(_failures)} 項未通過：")
        for item in _failures:
            print(f"  - {item}")
        print("=" * 70)
        return 1

    print("全部通過。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
