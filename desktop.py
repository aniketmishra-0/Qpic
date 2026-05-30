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

import os
import socket
import sys
import threading
import time
from pathlib import Path


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

    webview.create_window(
        "Qpic",
        f"http://{host}:{port}/",
        width=1080,
        height=760,
        min_size=(380, 560),
    )
    # Blocks until the window is closed; the daemon server thread dies with it.
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
