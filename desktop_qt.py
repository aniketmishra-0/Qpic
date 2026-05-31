"""Qt (PySide6) desktop launcher for Qpic.

Same idea as ``desktop.py`` — run the FastAPI app on a private localhost port in
a background thread and show the existing web UI in a native window — but the
window here is a Qt ``QWebEngineView`` (Qt's bundled Chromium) instead of
pywebview's OS webview.

Why a Qt variant?
  * Consistent Chromium rendering on every OS (the UI looks identical on macOS
    and Windows, no native-webview quirks).
  * A real Qt main window: native menu bar, window state, zoom shortcuts.

The web server, services and the static UI are completely unchanged; only the
window layer differs. The cheap server-bootstrap helpers are reused from
``desktop.py`` so there's a single source of truth for "start the backend".

Run from source:
    python desktop_qt.py

Bundle:
    pyinstaller desktop_qt.spec --noconfirm
"""

from __future__ import annotations

import os
import sys
import threading

# Reuse the exact server-bootstrap logic from the pywebview launcher so the two
# entry points can never drift apart.
from desktop import (
    _find_free_port,
    _resource_dir,
    _run_server,
    _wait_until_up,
    _writable_data_dir,
)

APP_NAME = "Qpic"
WINDOW_W, WINDOW_H = 1080, 760
MIN_W, MIN_H = 380, 560


def _build_window(url: str):
    """Create the Qt main window hosting the web UI.

    Imports of the Qt modules are deferred to here so importing this module
    (e.g. by PyInstaller's analysis) doesn't hard-require Qt at module load.
    """

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QFileDialog, QMainWindow

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(APP_NAME)
            self.resize(WINDOW_W, WINDOW_H)
            self.setMinimumSize(MIN_W, MIN_H)

            self.view = QWebEngineView(self)
            self.view.load(url)  # QUrl-coercible str
            self.setCentralWidget(self.view)

            # QWebEngineView has no built-in download manager, so a plain
            # `<a download>` or a `blob:` link would silently save nothing.
            # Catch every download request and route it through a native
            # "Save As" dialog instead.
            self.view.page().profile().downloadRequested.connect(self._on_download)

            # In-app Help menu (no external links — just the built-in guide).
            self._build_help_menu()

            # Familiar zoom shortcuts (Ctrl/Cmd +/-/0), mirroring a browser.
            mod = Qt.ControlModifier
            for keys, fn in (
                (QKeySequence.ZoomIn, self._zoom_in),
                (QKeySequence("Ctrl+="), self._zoom_in),
                (QKeySequence.ZoomOut, self._zoom_out),
                (QKeySequence("Ctrl+0"), self._zoom_reset),
            ):
                QShortcut(keys, self, activated=fn)

        def _build_help_menu(self) -> None:
            """Add a Help menu whose items only open the in-app walkthrough.

            Each action drives the page's existing "How to Use" modal by
            clicking the elements the web UI already wires up — nothing here
            leaves the app (no website links, no update checks).
            """

            menu = self.menuBar().addMenu("Help")

            act_guide = menu.addAction("How to Use Qpic")
            act_guide.setShortcut(QKeySequence("Ctrl+?"))
            act_guide.triggered.connect(lambda: self._open_help(None))

            menu.addSeparator()
            menu.addAction("How to Crop").triggered.connect(
                lambda: self._open_help("crop")
            )
            menu.addAction("How to Rename Batch").triggered.connect(
                lambda: self._open_help("rename")
            )

        def _open_help(self, tab_key: "str | None") -> None:
            """Open the in-app How-to-Use modal, optionally on a given tab."""

            js = "var b=document.getElementById('howToBtn'); if(b)b.click();"
            if tab_key:
                js += (
                    "var t=document.querySelector("
                    f"'.ht-tab[data-ht-tab=\"{tab_key}\"]'); if(t)t.click();"
                )
            self.view.page().runJavaScript(js)

        def _on_download(self, download: "QWebEngineDownloadRequest") -> None:
            """Pop a native Save-As dialog and write the downloaded bytes."""

            suggested = download.suggestedFileName() or "download.zip"
            target, _ = QFileDialog.getSaveFileName(self, "Save", suggested)
            if not target:
                download.cancel()
                return

            # PySide6 6.x splits the destination into a directory + filename.
            directory, _, filename = target.rpartition("/")
            download.setDownloadDirectory(directory or ".")
            download.setDownloadFileName(filename or suggested)
            download.accept()

        def _zoom_in(self) -> None:
            self.view.setZoomFactor(min(self.view.zoomFactor() + 0.1, 3.0))

        def _zoom_out(self) -> None:
            self.view.setZoomFactor(max(self.view.zoomFactor() - 0.1, 0.4))

        def _zoom_reset(self) -> None:
            self.view.setZoomFactor(1.0)

    return MainWindow()


def main() -> int:
    # Make bundled resources importable / discoverable (mirrors desktop.py).
    res = _resource_dir()
    if str(res) not in sys.path:
        sys.path.insert(0, str(res))

    # Keep crop jobs in a writable per-user folder.
    os.environ.setdefault("TEMP_DIR", str(_writable_data_dir() / "temp"))

    host = "127.0.0.1"
    port = _find_free_port()

    server_thread = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    server_thread.start()

    if not _wait_until_up(host, port):
        sys.stderr.write("Qpic: server failed to start.\n")
        return 1

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    window = _build_window(f"http://{host}:{port}/")
    window.show()

    # Blocks until the window is closed; the daemon server thread dies with it.
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
