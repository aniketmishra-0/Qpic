#!/usr/bin/env bash
#
# Build the Qt (PySide6) Qpic desktop app (macOS .app / Linux binary).
#
# This variant shows the UI in a Qt WebEngine (Chromium) window instead of the
# OS webview used by build_desktop.sh, so rendering is identical across
# platforms. The trade-off is a larger bundle (~150-200 MB).
#
# Usage:
#   ./build_desktop_qt.sh
#
# Output:
#   dist/Qpic.app   (macOS)
#   dist/Qpic/      (Linux/Windows folder build)
#
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "==> Installing build dependencies"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -r requirements-desktop-qt.txt

echo "==> Vendoring Tesseract (for offline OCR in the bundle)"
if ! "$PY" scripts/vendor_tesseract.py; then
  echo "    WARNING: could not vendor Tesseract. The app will still build, but"
  echo "    OCR for scanned PDFs will need a system Tesseract install."
  echo "    Install it first:  brew install tesseract tesseract-lang"
fi

echo "==> Cleaning previous build"
rm -rf build dist

echo "==> Building with PyInstaller (Qt variant)"
"$PY" -m PyInstaller desktop_qt.spec --noconfirm

echo
echo "==> Done."
if [ -d "dist/Qpic.app" ]; then
  echo "    App:  dist/Qpic.app"
  echo "    Tip:  first launch -> right-click the app -> Open (unsigned build)."
else
  echo "    Build output is in: dist/"
fi
