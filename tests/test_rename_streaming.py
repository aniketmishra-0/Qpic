"""Tests for the streamed (large-batch) rename flow.

Covers the disk-based ZIP builder directly and the full session API
(create → upload chunks → finalize → download) so a multi-thousand-file,
multi-gigabyte batch can be packed without holding it all in memory.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import app
from app.services.rename_service import write_rename_zip_from_paths


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_write_zip_from_paths_verbatim(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.jpg"
    raw_a = _png_bytes((255, 0, 0))
    a.write_bytes(raw_a)
    b.write_bytes(_jpg_bytes((0, 255, 0)))

    zip_path = tmp_path / "out.zip"
    count = write_rename_zip_from_paths(
        zip_path,
        [("a.png", a), ("b.jpg", b)],
        pattern="Q#",
        start=1,
        padding=2,
    )

    assert count == 2
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["Q01.png", "Q02.jpg"]
        # Verbatim copy keeps the exact bytes.
        assert zf.read("Q01.png") == raw_a
        assert Image.open(io.BytesIO(zf.read("Q02.jpg"))).format == "JPEG"


def test_write_zip_from_paths_reencodes_to_png(tmp_path: Path) -> None:
    src = tmp_path / "photo.jpg"
    src.write_bytes(_jpg_bytes((0, 0, 255)))

    zip_path = tmp_path / "out.zip"
    write_rename_zip_from_paths(
        zip_path,
        [("photo.jpg", src)],
        pattern="#",
        start=1,
        padding=0,
        output_format="png",
    )

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["1.png"]
        assert Image.open(io.BytesIO(zf.read("1.png"))).format == "PNG"


def test_write_zip_from_paths_explicit_stems(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(_png_bytes((1, 2, 3)))
    b.write_bytes(_png_bytes((4, 5, 6)))

    zip_path = tmp_path / "out.zip"
    write_rename_zip_from_paths(
        zip_path,
        [("a.png", a), ("b.png", b)],
        pattern="#",
        start=1,
        padding=0,
        explicit_stems=["cat", "dog"],
    )

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["cat.png", "dog.png"]


def _client(tmp_path: Path) -> TestClient:
    client = TestClient(app)
    # The lifespan sets app.state on enter; override temp_root to a temp dir so
    # the test never touches the real ./temp folder.
    client.__enter__()
    app.state.settings = Settings(ANTHROPIC_API_KEY=None)
    app.state.temp_root = str(tmp_path)
    return client


def test_session_flow_packs_all_chunks(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        sid = client.post("/api/rename/session").json()["session_id"]

        # Upload 5 files across two chunks; order must be preserved.
        chunk1 = [
            ("files", (f"img{i}.png", _png_bytes((i, i, i)), "image/png"))
            for i in range(3)
        ]
        r1 = client.post(f"/api/rename/session/{sid}/files", files=chunk1)
        assert r1.status_code == 200
        assert r1.json()["total"] == 3

        chunk2 = [
            ("files", (f"img{i}.png", _png_bytes((i, i, i)), "image/png"))
            for i in range(3, 5)
        ]
        r2 = client.post(f"/api/rename/session/{sid}/files", files=chunk2)
        assert r2.json()["total"] == 5

        # Finalize with a pattern.
        fr = client.post(
            f"/api/rename/session/{sid}/finalize",
            data={"pattern": "Q#", "start": "1", "padding": "2", "output_format": "original"},
        )
        assert fr.status_code == 200
        body = fr.json()
        assert body["count"] == 5
        assert body["download_url"].endswith(f"/{sid}/download")

        # Download streams the ZIP back.
        dl = client.get(body["download_url"])
        assert dl.status_code == 200
        with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
            assert zf.namelist() == ["Q01.png", "Q02.png", "Q03.png", "Q04.png", "Q05.png"]

        # Cleanup removes the session.
        client.delete(f"/api/rename/session/{sid}")
        assert client.get(body["download_url"]).status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_session_explicit_names_via_json(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        sid = client.post("/api/rename/session").json()["session_id"]
        files = [
            ("files", ("one.png", _png_bytes((1, 1, 1)), "image/png")),
            ("files", ("two.png", _png_bytes((2, 2, 2)), "image/png")),
        ]
        client.post(f"/api/rename/session/{sid}/files", files=files)

        fr = client.post(
            f"/api/rename/session/{sid}/finalize",
            data={"names": '["alpha", "beta"]', "output_format": "original"},
        )
        assert fr.status_code == 200
        dl = client.get(fr.json()["download_url"])
        with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
            assert zf.namelist() == ["alpha.png", "beta.png"]
    finally:
        client.__exit__(None, None, None)


def test_session_rejects_unknown_session(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        files = [("files", ("x.png", _png_bytes((0, 0, 0)), "image/png"))]
        r = client.post("/api/rename/session/deadbeef/files", files=files)
        assert r.status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_session_name_count_mismatch_errors(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        sid = client.post("/api/rename/session").json()["session_id"]
        files = [("files", ("a.png", _png_bytes((0, 0, 0)), "image/png"))]
        client.post(f"/api/rename/session/{sid}/files", files=files)
        fr = client.post(
            f"/api/rename/session/{sid}/finalize",
            data={"names": '["a", "b", "c"]'},
        )
        assert fr.status_code == 400
    finally:
        client.__exit__(None, None, None)
