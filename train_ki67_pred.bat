@echo off
setlocal

cd /d "%~dp0"
set "PYTHON_EXE=D:\anaconda3\envs\ki67dtc\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

echo [INFO] Running Ki67 formal training...
"%PYTHON_EXE%" "analysis\ki67_pred_training.py"
if errorlevel 1 (
  echo [ERROR] ki67_pred_training failed.
  exit /b 1
)

echo.
echo [DONE] Ki67 formal training finished.
exit /b 0
