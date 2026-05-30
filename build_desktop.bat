@echo off
REM Build the Qpic desktop app on Windows (.exe).
REM
REM Usage (from this folder, in a Command Prompt):
REM     build_desktop.bat
REM
REM Output:
REM     dist\Qpic\Qpic.exe

setlocal
cd /d "%~dp0"

set PY=python

echo ==> Installing build dependencies
%PY% -m pip install -r requirements.txt || goto :error
%PY% -m pip install -r requirements-desktop.txt || goto :error

echo ==> Vendoring Tesseract (for offline OCR in the bundle)
%PY% scripts\vendor_tesseract.py
if errorlevel 1 (
    echo    WARNING: could not vendor Tesseract. The app will still build, but
    echo    OCR for scanned PDFs will need a system Tesseract install.
    echo    Install it from: https://github.com/UB-Mannheim/tesseract/wiki
)

echo ==> Cleaning previous build
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo ==> Building with PyInstaller
%PY% -m PyInstaller desktop.spec --noconfirm || goto :error

echo.
echo ==> Done. App is in: dist\Qpic\
goto :eof

:error
echo.
echo Build failed. See the output above.
exit /b 1
