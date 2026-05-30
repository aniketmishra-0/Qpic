# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Qt (PySide6) Qpic desktop build.

Same bundling as desktop.spec (FastAPI backend + static UI + vendored
Tesseract), but the window layer is Qt WebEngine instead of pywebview, and the
entry point is ``desktop_qt.py``.

Build:
    pyinstaller desktop_qt.spec --noconfirm
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

import os

block_cipher = None

# Ship the web UI and the app package source alongside the binary.
datas = [
    ("static", "static"),
    ("app", "app"),
]

# Bundle a self-contained Tesseract if it was vendored.
if os.path.isdir("vendor/tesseract"):
    datas += [("vendor/tesseract", "tesseract")]
    print("desktop_qt.spec: bundling vendored Tesseract from vendor/tesseract")
else:
    print(
        "desktop_qt.spec: WARNING no vendor/tesseract found — OCR will require a "
        "system Tesseract install. Run scripts/vendor_tesseract.py to bundle it."
    )

# Pull in dynamically-imported modules PyInstaller can't see by static analysis.
hiddenimports = []
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("fastapi")
hiddenimports += collect_submodules("anthropic")
hiddenimports += ["h11", "anyio", "click", "pydantic_settings"]

# pymupdf / pillow / cv2 occasionally need their data files bundled.
datas += collect_data_files("fitz", include_py_files=False)

a = Analysis(
    ["desktop_qt.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # tkinter not needed; keep matplotlib/pytest out. PySide6's own PyInstaller
    # hook pulls in the required Qt WebEngine resources automatically.
    excludes=["tkinter", "pytest", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Qpic",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX can corrupt Qt's shared libs — leave it off for the Qt build.
    upx=False,
    console=False,  # windowed app — no terminal
    disable_windowed_traceback=False,
    argv_emulation=True,  # macOS: handle file-open events cleanly
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
    upx=False,
    upx_exclude=[],
    name="Qpic",
)

# macOS .app bundle wrapper.
app = BUNDLE(
    coll,
    name="Qpic.app",
    icon=None,
    bundle_identifier="com.qpic.desktop",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSBackgroundOnly": False,
    },
)
