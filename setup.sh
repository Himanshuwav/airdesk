#!/bin/sh
# Creates .venv inside this folder and installs everything into it. No sudo,
# nothing installed system-wide.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-/usr/local/bin/python3.12}"
if ! "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
  echo "Need Python 3.12 (mediapipe 0.10.21 has no wheels for newer versions)." >&2
  echo "Set PYTHON=/path/to/python3.12 and rerun." >&2
  exit 1
fi
"$PY" -m venv .venv
export PIP_NO_CACHE_DIR=1
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps face_recognition==1.3.0
.venv/bin/python -W ignore -c "import cv2, mediapipe, face_recognition, pyautogui; print('AirDesk dependencies OK')"
