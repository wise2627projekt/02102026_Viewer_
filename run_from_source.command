#!/bin/bash
# PhotoVec Viewer on a Mac, from the source: double-click (first time: right-click -> Open). Needs Python 3.9 or newer
# (python.org, or `brew install python@3.12`). Installs the 3 small libraries once into .venv-mac next to this file.
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3.12 python3.11 python3.13 python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "ERROR: Python 3.9 or newer was not found (install it from python.org)."; read -n 1 -s -r -p "Press any key to close."; exit 1; }
[ -x .venv-mac/bin/python ] || "$PY" -m venv .venv-mac || { echo "ERROR: could not create the environment"; exit 1; }
if ! .venv-mac/bin/python -c "import PySide6, cv2, numpy" >/dev/null 2>&1; then
  .venv-mac/bin/python -m pip install --upgrade pip
  .venv-mac/bin/python -m pip install -r requirements.txt || .venv-mac/bin/python -m pip install numpy opencv-python-headless PySide6-Essentials || {
    echo "ERROR: install failed - see the messages above."; read -n 1 -s -r -p "Press any key to close."; exit 1; }
fi
exec .venv-mac/bin/python photovec_viewer.py "$@"
