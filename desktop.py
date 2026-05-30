"""Desktop launcher for Qpic.

Runs the FastAPI app on a private localhost port inside a background thread and
shows the existing web UI in a native desktop window (via ``pywebview``). No
terminal, no browser tab — double-click the bundled app and a normal window
opens.

This is what gets bundled by PyInstaller into the ``.app`` / ``.exe``. It is the
program's entry point in the packaged build; the web server still exists, it is
just hidden inside the app and started/stopped automatically.
"""

from __future__ import annotations

import base64
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional


def _resource_dir() -> Path:
    """Directory that holds bundled resources (static/, app/).

    Under PyInstaller the files are extracted to ``sys._MEIPASS``; when running
    from source it's just this file's folder.
    """

    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else Path(__file__).resolve().parent


def _writable_data_dir() -> Path:
    """A per-user, writable folder for temp crop jobs.

    The bundle itself is read-only (and wiped on exit), so cropped images and
    zips must live somewhere persistent and user-writable instead.
    """

    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support" / "Qpic"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", home)) / "Qpic"
    else:
        base = home / ".local" / "share" / "qpic"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _find_free_port() -> int:
    """Grab an unused localhost port so multiple launches don't clash."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_up(host: str, port: int, timeout: float = 30.0) -> bool:
    """Block until the server accepts connections (or we give up)."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def _run_server(host: str, port: int) -> None:
    """Start uvicorn in this thread.

    The native loop/http extras from ``uvicorn[standard]`` (uvloop/httptools)
    are skipped in favour of the pure-python ``asyncio``/``h11`` stack so the
    PyInstaller bundle stays portable and doesn't need those native modules.
    """

    import uvicorn

    from app.main import app

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
        http="h11",
        ws="none",
    )
    uvicorn.Server(config).run()


class SaveBridge:
    """Python side of the JS ``window.pywebview.api`` save bridge.

    A ``pywebview`` window has no browser download manager, so plain
    ``<a download>`` links and ``blob:`` URLs save nothing — clicking a download
    button just silently does nothing. The web UI therefore calls these methods
    instead: they pop a native "Save As" dialog and write the bytes to the path
    the user picks. ``base_url`` lets the UI hand us a server-relative path
    (``/api/crop/download/...``) that we fetch over the private localhost port.
    """

    def __init__(self) -> None:
        self.base_url = ""
        self._window: Any = None

    def attach(self, window: Any, base_url: str) -> None:
        self._window = window
        self.base_url = base_url.rstrip("/")

    def _ask_save_path(self, suggested_name: str) -> Optional[str]:
        """Show the native Save-As dialog; return the chosen path or None."""

        import webview

        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=suggested_name or "download.zip",
        )
        # pywebview returns a str (newer) or a (path,) tuple/list (older); both
        # collapse to a single path here. None/empty means the user cancelled.
        if not result:
            return None
        if isinstance(result, (list, tuple)):
            return str(result[0]) if result else None
        return str(result)

    def save_url(self, url: str, suggested_name: str) -> dict:
        """Download a server URL and save it via a native Save-As dialog.

        ``url`` may be absolute or server-relative (it's joined onto the private
        localhost base). Returns a small status dict the JS side can surface.
        """

        try:
            target = self._ask_save_path(suggested_name)
            if not target:
                return {"ok": False, "cancelled": True}

            full = url if url.startswith("http") else f"{self.base_url}{url}"
            with urllib.request.urlopen(full) as resp:  # noqa: S310 (localhost only)
                data = resp.read()
            Path(target).write_bytes(data)
            return {"ok": True, "path": target}
        except Exception as exc:  # surface a readable message to the UI
            return {"ok": False, "error": str(exc)}

    def save_base64(self, b64: str, suggested_name: str) -> dict:
        """Save raw bytes (base64-encoded by JS) via a native Save-As dialog.

        Used by the Rename tool, whose ZIP is built from an in-memory blob
        rather than a downloadable URL.
        """

        try:
            target = self._ask_save_path(suggested_name)
            if not target:
                return {"ok": False, "cancelled": True}

            Path(target).write_bytes(base64.b64decode(b64))
            return {"ok": True, "path": target}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def main() -> int:
    # Make bundled resources importable / discoverable.
    res = _resource_dir()
    if str(res) not in sys.path:
        sys.path.insert(0, str(res))

    # Keep crop jobs in a writable per-user folder (absolute path overrides the
    # app's default relative TEMP_DIR).
    os.environ.setdefault("TEMP_DIR", str(_writable_data_dir() / "temp"))

    host = "127.0.0.1"
    port = _find_free_port()

    server_thread = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    server_thread.start()

    if not _wait_until_up(host, port):
        sys.stderr.write("Qpic: server failed to start.\n")
        return 1

    import webview  # pywebview

    base_url = f"http://{host}:{port}"
    bridge = SaveBridge()
    window = webview.create_window(
        "Qpic",
        f"{base_url}/",
        js_api=bridge,
        width=1280,
        height=860,
        min_size=(720, 600),
    )
    # Hand the window + base URL to the bridge so its Save-As dialogs can fetch
    # job ZIPs over the private localhost port.
    bridge.attach(window, base_url)
    # Blocks until the window is closed; the daemon server thread dies with it.
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
