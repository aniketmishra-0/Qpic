"""Collect a self-contained Tesseract into ``vendor/tesseract`` for bundling.

PyInstaller never bundles Tesseract on its own, so the packaged desktop app
would have no OCR unless the end user separately installed it. This script
copies an *installed* Tesseract (binary + its shared libraries + ``tessdata``
language files) into ``vendor/tesseract`` as a relocatable folder. ``desktop.spec``
then ships that folder inside the app, and ``tesseract_locator`` finds it at
runtime.

Run it once on each build machine *before* PyInstaller:

    python scripts/vendor_tesseract.py            # auto-detect installed tesseract
    python scripts/vendor_tesseract.py --src /opt/homebrew/bin/tesseract

Platforms:
* macOS  — copies the binary + every non-system dylib it depends on (found via
  ``otool -L``, walked transitively) and rewrites their load paths to
  ``@loader_path`` so the bundle runs on a machine without Homebrew.
* Windows — copies ``tesseract.exe`` + the DLLs sitting next to it (the
  UB-Mannheim installer ships them together) + ``tessdata``.
* Linux  — copies the binary + its non-system shared objects (via ``ldd``);
  mainly useful for a Linux onedir build (CI primarily targets mac + Windows).

It is intentionally dependency-free (stdlib only) so CI can run it before the
project's own deps are installed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEST = PROJECT_ROOT / "vendor" / "tesseract"

# Library prefixes that always exist on the target OS — never copy or relocate
# these (doing so can break the bundle on a different OS patch level).
_MAC_SYSTEM_PREFIXES = ("/usr/lib/", "/System/")
_LINUX_SYSTEM_PREFIXES = (
    "linux-vdso",
    "/lib/",
    "/lib64/",
    "/usr/lib/x86_64-linux-gnu/libc",
    "/usr/lib/x86_64-linux-gnu/libm",
    "/usr/lib/x86_64-linux-gnu/libpthread",
    "/usr/lib/x86_64-linux-gnu/libdl",
)


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _detect_binary(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"--src path does not exist: {p}")
        return p
    name = "tesseract.exe" if os.name == "nt" else "tesseract"
    found = shutil.which(name)
    if not found and os.name == "nt":
        # UB-Mannheim default install dir isn't always on PATH.
        for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
            base = os.environ.get(env_key)
            if base:
                cand = Path(base) / "Tesseract-OCR" / name
                if cand.is_file():
                    found = str(cand)
                    break
    if not found:
        sys.exit(
            "Could not find an installed 'tesseract'. Install it first "
            "(brew install tesseract / apt install tesseract-ocr / "
            "UB-Mannheim installer) or pass --src."
        )
    return Path(found)


def _find_tessdata(binary: Path) -> Path | None:
    """Locate the tessdata folder for an installed Tesseract."""

    env = os.environ.get("TESSDATA_PREFIX")
    cands: list[Path] = []
    if env:
        cands += [Path(env), Path(env) / "tessdata"]
    cands += [
        binary.parent / "tessdata",
        binary.parent.parent / "share" / "tessdata",
        binary.parent.parent / "share" / "tesseract-ocr" / "tessdata",
    ]
    # Homebrew keeps tessdata under the versioned cellar; resolve symlinks.
    try:
        real = binary.resolve()
        cands += [
            real.parent.parent / "share" / "tessdata",
            real.parent.parent / "share" / "tesseract-ocr" / "tessdata",
        ]
    except OSError:
        pass
    # Homebrew "tesseract-lang" formula installs extra languages here.
    if sys.platform == "darwin":
        for pref in ("/opt/homebrew", "/usr/local"):
            cands.append(Path(pref) / "share" / "tessdata")

    for c in cands:
        if c.is_dir() and any(c.glob("*.traineddata")):
            return c
    return None


# --------------------------------------------------------------------------- #
# macOS
# --------------------------------------------------------------------------- #
def _mac_deps(target: Path) -> list[str]:
    """Non-system dylibs a Mach-O file links against (one level)."""

    out = _run(["otool", "-L", str(target)])
    deps: list[str] = []
    for line in out.splitlines()[1:]:
        m = re.match(r"\s+(\S+)\s+\(", line)
        if not m:
            continue
        dep = m.group(1)
        if dep.startswith(_MAC_SYSTEM_PREFIXES):
            continue
        if dep.startswith("@"):  # already relocated reference
            continue
        deps.append(dep)
    return deps


def _vendor_macos(binary: Path, dest: Path) -> None:
    bin_dir = dest
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the full dependency closure of the binary.
    collected: dict[str, Path] = {}  # basename -> real source path
    queue = list(_mac_deps(binary))
    while queue:
        dep = queue.pop()
        src = Path(dep)
        if not src.is_file():
            # Some deps are referenced by an install-name that isn't a real path
            # (e.g. @rpath). Best-effort: skip; the linker found it at build time.
            continue
        name = src.name
        if name in collected:
            continue
        collected[name] = src
        queue.extend(_mac_deps(src))

    # Copy binary + dylibs side by side.
    dst_bin = bin_dir / "tesseract"
    shutil.copy2(binary, dst_bin)
    dst_bin.chmod(0o755)

    for name, src in collected.items():
        shutil.copy2(src, bin_dir / name)
        (bin_dir / name).chmod(0o644)

    # Rewrite load paths to @loader_path so everything resolves within the folder.
    def _relocate(target: Path, is_lib: bool) -> None:
        if is_lib:
            subprocess.run(
                ["install_name_tool", "-id", f"@loader_path/{target.name}", str(target)],
                check=False,
                capture_output=True,
            )
        for dep in _mac_deps(target):
            base = Path(dep).name
            if base in collected:
                subprocess.run(
                    ["install_name_tool", "-change", dep, f"@loader_path/{base}", str(target)],
                    check=False,
                    capture_output=True,
                )

    _relocate(dst_bin, is_lib=False)
    for name in collected:
        _relocate(bin_dir / name, is_lib=True)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def _vendor_windows(binary: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, dest / binary.name)
    # The UB-Mannheim build keeps all required DLLs next to tesseract.exe.
    for dll in binary.parent.glob("*.dll"):
        shutil.copy2(dll, dest / dll.name)


# --------------------------------------------------------------------------- #
# Linux
# --------------------------------------------------------------------------- #
def _vendor_linux(binary: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    dst_bin = dest / "tesseract"
    shutil.copy2(binary, dst_bin)
    dst_bin.chmod(0o755)
    try:
        out = _run(["ldd", str(binary)])
    except Exception:
        out = ""
    for line in out.splitlines():
        m = re.search(r"=>\s+(\S+)\s+\(", line)
        if not m:
            continue
        lib = m.group(1)
        if lib.startswith(_LINUX_SYSTEM_PREFIXES) or not Path(lib).is_file():
            continue
        shutil.copy2(lib, dest / Path(lib).name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", help="Path to an installed tesseract binary.")
    parser.add_argument(
        "--langs",
        default="eng,hin,osd",
        help="Comma-separated traineddata languages to include (default: eng,hin,osd).",
    )
    args = parser.parse_args()

    binary = _detect_binary(args.src)
    print(f"==> Using tesseract: {binary}")

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)

    if sys.platform == "darwin":
        _vendor_macos(binary, DEST)
    elif os.name == "nt":
        _vendor_windows(binary, DEST)
    else:
        _vendor_linux(binary, DEST)

    # Copy language data into vendor/tesseract/tessdata.
    tessdata = _find_tessdata(binary)
    if not tessdata:
        sys.exit("Could not find tessdata (the *.traineddata language files).")
    wanted = {l.strip() for l in args.langs.split(",") if l.strip()}
    dst_tessdata = DEST / "tessdata"
    dst_tessdata.mkdir(parents=True, exist_ok=True)
    copied = []
    for td in tessdata.glob("*.traineddata"):
        if not wanted or td.stem in wanted:
            shutil.copy2(td, dst_tessdata / td.name)
            copied.append(td.stem)
    if not copied:
        sys.exit(f"None of the requested languages {sorted(wanted)} found in {tessdata}")
    print(f"==> Vendored languages: {', '.join(sorted(copied))}")
    print(f"==> Done. Self-contained Tesseract at: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
