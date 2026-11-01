@echo off
rem Starts the Viewer from the source (needs Python 3.11 or 3.12; installs the 3 small libraries once).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || (echo ERROR: Python 3 not found. & pause & exit /b 1)
)
.venv\Scripts\python.exe -c "import PySide6, cv2, numpy" >nul 2>&1 || .venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe photovec_viewer.py %*
