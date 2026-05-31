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
import shutil
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Module-level reference to the pywebview window, used by the native macOS menus
# (installed after the GUI loop starts) so their actions can drive the web UI.
_MENU_WINDOW: Any = None
# The Help-menu action target must be retained for the lifetime of the app, or
# AppKit's weak reference to it is collected and the menu items stop working.
_HELP_TARGET: Any = None


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
        localhost base). The response is streamed to disk in chunks so even a
        multi-gigabyte ZIP is saved with a flat memory footprint. Returns a
        small status dict the JS side can surface.
        """

        try:
            target = self._ask_save_path(suggested_name)
            if not target:
                return {"ok": False, "cancelled": True}

            full = url if url.startswith("http") else f"{self.base_url}{url}"
            with urllib.request.urlopen(full) as resp:  # noqa: S310 (localhost only)
                with open(target, "wb") as out:
                    shutil.copyfileobj(resp, out, length=1024 * 1024)
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


def _run_ui_js(js: str) -> None:
    """Run a snippet of JavaScript in the web UI, if the window is ready.

    The native menus reach the UI only through this helper, so the menu actions
    can stay tied to features that already exist in the page (no new web code,
    no external links).
    """

    win = _MENU_WINDOW
    if win is None:
        return
    try:
        win.evaluate_js(js)
    except Exception:  # pragma: no cover - best-effort; menu just no-ops
        pass


def _make_help_target():
    """Build the AppKit object whose methods back the Help menu items.

    Each item just drives the page's built-in "How to Use" walkthrough by
    clicking the elements the web UI already wires up:
      * ``#howToBtn``                       opens the walkthrough modal
      * ``.ht-tab[data-ht-tab="crop"]``     jumps to the "How to Crop" tab
      * ``.ht-tab[data-ht-tab="rename"]``   jumps to the "How to Rename Batch" tab

    Returns ``None`` if PyObjC/Foundation isn't importable.
    """

    try:
        from Foundation import NSObject
    except Exception:  # pragma: no cover - Foundation always present on a mac build
        return None

    # Clicking #howToBtn runs the UI's own openHowTo(); switching tabs reuses
    # the tab buttons' existing click handlers.
    _OPEN = "var b=document.getElementById('howToBtn'); if(b)b.click();"
    _TAB = (
        "var b=document.getElementById('howToBtn'); if(b)b.click();"
        "var t=document.querySelector('.ht-tab[data-ht-tab=\"{key}\"]'); if(t)t.click();"
    )

    class HelpTarget(NSObject):
        def showHowToUse_(self, _sender) -> None:
            _run_ui_js(_OPEN)

        def showCropHelp_(self, _sender) -> None:
            _run_ui_js(_TAB.format(key="crop"))

        def showRenameHelp_(self, _sender) -> None:
            _run_ui_js(_TAB.format(key="rename"))

    return HelpTarget.alloc().init()


def _install_macos_edit_menu() -> None:
    """Give the macOS window standard Edit + Help menus.

    pywebview's Cocoa backend never installs an application main menu. On macOS
    the usual editing shortcuts (Select All, Copy, Paste…) are delivered through
    the Edit menu's key equivalents, so without that menu Cmd+A does nothing —
    both in the UI's text fields and in the native file-open panel shown when
    you browse images for the Rename tool. Build a minimal main menu with the
    standard responder selectors; the keys then reach the first responder
    (text field or open panel) automatically.

    A Help menu is added too, but it stays strictly in-app: every item opens the
    page's existing "How to Use" walkthrough (overall guide, cropping, renaming).
    No website links, no "check for updates" — only what already lives in Qpic.

    No-op on non-macOS platforms or if AppKit isn't importable.
    """

    if sys.platform != "darwin":
        return

    try:
        from AppKit import NSApplication, NSMenu, NSMenuItem
    except Exception:  # pragma: no cover - AppKit always present on a mac build
        return

    app = NSApplication.sharedApplication()
    main_menu = NSMenu.alloc().init()

    # First menu is the "application" menu; it must exist for the menu bar to
    # render, and it carries the standard Quit shortcut (Cmd+Q).
    app_item = NSMenuItem.alloc().init()
    main_menu.addItem_(app_item)
    app_menu = NSMenu.alloc().init()
    app_item.setSubmenu_(app_menu)
    app_menu.addItem_(
        NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Qpic", "terminate:", "q"
        )
    )

    # The Edit menu is what actually fixes Cmd+A & friends.
    edit_item = NSMenuItem.alloc().init()
    main_menu.addItem_(edit_item)
    edit_menu = NSMenu.alloc().initWithTitle_("Edit")
    edit_item.setSubmenu_(edit_menu)
    for title, action, key in (
        ("Undo", "undo:", "z"),
        ("Redo", "redo:", "Z"),
        (None, None, None),
        ("Cut", "cut:", "x"),
        ("Copy", "copy:", "c"),
        ("Paste", "paste:", "v"),
        ("Select All", "selectAll:", "a"),
    ):
        if title is None:
            edit_menu.addItem_(NSMenuItem.separatorItem())
            continue
        edit_menu.addItem_(
            NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        )

    # The Help menu — in-app only. Items map 1:1 to the page's "How to Use"
    # walkthrough tabs, so nothing here leaves the app.
    global _HELP_TARGET
    _HELP_TARGET = _make_help_target()
    if _HELP_TARGET is not None:
        help_item = NSMenuItem.alloc().init()
        main_menu.addItem_(help_item)
        help_menu = NSMenu.alloc().initWithTitle_("Help")
        help_item.setSubmenu_(help_menu)
        for title, selector, key in (
            ("How to Use Qpic", "showHowToUse:", "?"),
            (None, None, None),
            ("How to Crop", "showCropHelp:", ""),
            ("How to Rename Batch", "showRenameHelp:", ""),
        ):
            if title is None:
                help_menu.addItem_(NSMenuItem.separatorItem())
                continue
            mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, selector, key
            )
            mi.setTarget_(_HELP_TARGET)
            help_menu.addItem_(mi)
        # Registering it as *the* Help menu gives the standard search field and
        # the system-standard menu position.
        app.setHelpMenu_(help_menu)

    app.setMainMenu_(main_menu)


def _on_gui_started() -> None:
    """Runs once the GUI loop is up; installs the macOS Edit menu on the main thread."""

    if sys.platform != "darwin":
        return
    try:
        from PyObjCTools import AppHelper

        # NSApp lives on the main thread; schedule the menu install there.
        AppHelper.callAfter(_install_macos_edit_menu)
    except Exception:  # pragma: no cover - best-effort; app still works without it
        pass


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
    # Expose the window to the native macOS menus so the Help menu items can
    # drive the in-app "How to Use" walkthrough.
    global _MENU_WINDOW
    _MENU_WINDOW = window
    # Blocks until the window is closed; the daemon server thread dies with it.
    # ``func`` fires once the GUI loop is running, where we add the macOS Edit
    # menu that makes Cmd+A/C/V/X/Z work in the UI and native file dialogs.
    webview.start(_on_gui_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
