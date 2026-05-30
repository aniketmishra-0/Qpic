#!/usr/bin/env bash
#
# Build the Qpic desktop app (macOS .app / Linux binary).
#
# Usage:
#   ./build_desktop.sh
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
"$PY" -m pip install -r requirements-desktop.txt

echo "==> Vendoring Tesseract (for offline OCR in the bundle)"
if ! "$PY" scripts/vendor_tesseract.py; then
  echo "    WARNING: could not vendor Tesseract. The app will still build, but"
  echo "    OCR for scanned PDFs will need a system Tesseract install."
  echo "    Install it first:  brew install tesseract tesseract-lang"
fi

echo "==> Cleaning previous build"
rm -rf build dist

echo "==> Building with PyInstaller"
"$PY" -m PyInstaller desktop.spec --noconfirm

echo
echo "==> Done."
if [ -d "dist/Qpic.app" ]; then
  echo "    App:  dist/Qpic.app"
  echo "    Tip:  first launch -> right-click the app -> Open (unsigned build)."
else
  echo "    Build output is in: dist/"
fi
