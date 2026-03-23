#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run main.py for a list of data folders on Windows.
- Sequential execution (stops on failure by default).
- Logs stdout/stderr to ./logs/run_<timestamp>.log
- Skips any folder that doesn't exist.
Customize FOLDER_NAMES below if needed.
"""

from pathlib import Path
import subprocess
import time
import sys
import os

# ====== USER SETTINGS ======
# Root under which your folders live (e.g., "data/input")
DATA_ROOT = Path("data") / "input"

# Your folder names (edit this list as needed)
FOLDER_NAMES = [
    "2025-06-19-B4-P6-P10-P14-Ki67-P6-1",
    "2025-06-19-B4-P6-P10-P14-Ki67-P6-2",
    "2025-06-19-B4-P6-P10-P14-Ki67-P10-1",
    "2025-06-19-B4-P6-P10-P14-Ki67-P10-2",
    "2025-06-19-B4-P6-P10-P14-Ki67-P14-1",
    "2025-06-19-B4-P6-P10-P14-Ki67-P14-2",
    "2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P6",
    "2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P10",
    "2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P14",
]

# Base Python executable and script to call
PY_EXE = sys.executable  # current python; change to r"C:\path\to\python.exe" if needed
MAIN_SCRIPT = Path("main.py")

# Extra args that stay the same every run
EXTRA_ARGS = ["--nuc_source", "dapi", "--ki67", "--ki67_backend", "pyimagej", "--clean_temp"]

# Stop the whole batch when one fails?
STOP_ON_FAILURE = True
# ====== END SETTINGS ======


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{ts}.log"

    with log_path.open("a", encoding="utf-8") as log:

        def log_write(s):
            print(s)
            print(s, file=log, flush=True)

        log_write("=" * 80)
        log_write(f"Batch start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_write(f"Working dir: {Path.cwd()}")
        log_write(f"Python: {PY_EXE}")
        log_write(f"Script: {MAIN_SCRIPT}")

        for name in FOLDER_NAMES:
            target = DATA_ROOT / name
            if not target.exists():
                log_write(f"[SKIP] Folder not found: {target}")
                continue

            # Run with -u (unbuffered) so child prints flush through immediately
            cmd = [
                PY_EXE,
                "-u",
                str(MAIN_SCRIPT),
                "--data_folder",
                str(target),
            ] + EXTRA_ARGS
            log_write("-" * 80)
            log_write(f"[RUN] {' '.join(cmd)}")
            start = time.time()

            try:
                # Stream output live to console and log. Merge stderr into stdout.
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                # Read lines as they arrive and tee to console + log
                assert p.stdout is not None
                for line in p.stdout:
                    line = line.rstrip("\n")
                    if line:
                        print(line)
                        print(line, file=log, flush=True)
                p.wait()
            except Exception as e:
                log_write(f"[ERROR] Failed to start process: {e!r}")
                if STOP_ON_FAILURE:
                    sys.exit(1)
                else:
                    continue

            elapsed = time.time() - start
            log_write(f"[EXIT] code={p.returncode} time={elapsed:.1f}s")

            if p.returncode != 0 and STOP_ON_FAILURE:
                log_write("[ABORT] Stop on failure is enabled.")
                sys.exit(p.returncode)

        log_write(f"Batch end: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_write("=" * 80)


if __name__ == "__main__":
    main()
