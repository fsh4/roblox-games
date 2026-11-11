import asyncio
import io
import os
import socket
import time

import pytest
from PIL import Image

from bot.delivery import LinkSigner
from utils.filesystem import UnsafePathError, safe_join
from utils.images import UploadRejected, check_upload


def png_bytes(w, h, mode="RGB"):
    b = io.BytesIO()
    Image.new(mode, (w, h)).save(b, "PNG")
    return b.getvalue()


def test_valid_png_accepted():
    assert check_upload("a.png", png_bytes(200, 120), 10_000_000, 40_000_000).kind == "png"


@pytest.mark.parametrize("name,data,why", [
    ("a.exe", png_bytes(100, 100), "extension"),
    ("a.png", b"MZ\x90\x00 executable", "magic"),
    ("a.png", b"", "empty"),
    ("a.png", png_bytes(20, 20), "too small"),
    ("a.svg", b"<svg onload=alert(1)>", "svg"),
])
def test_bad_uploads_rejected(name, data, why):
    with pytest.raises(UploadRejected):
        check_upload(name, data, 10_000_000, 40_000_000)


def test_oversized_and_bomb_images_rejected():
    with pytest.raises(UploadRejected):
        check_upload("a.png", png_bytes(300, 300), 100, 40_000_000)
    bomb = png_bytes(9000, 9000, "1")
    assert len(bomb) < 2_000_000
    with pytest.raises(UploadRejected):
        check_upload("bomb.png", bomb, 10_000_000, 40_000_000)


def test_safe_join_blocks_traversal(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    assert safe_join(base, "projects", "abc") == (base / "projects" / "abc").resolve()
    for bad in ("../x", "/etc/passwd", "a/../../x"):
        with pytest.raises(UnsafePathError):
            safe_join(base, bad)
    os.symlink(tmp_path, base / "escape")
    with pytest.raises(UnsafePathError):
        safe_join(base, "escape", "secret")


def test_signed_links(tmp_path):
    data = tmp_path / "data"
    (data / "projects" / "p1").mkdir(parents=True)
    f = data / "projects" / "p1" / "Model_v1.zip"
    f.write_bytes(b"zip")
    signer = LinkSigner("s3cret", "https://example.test", data, 1)
    url = signer.sign(f)
    exp, mac, rel = url.split("/d/")[1].split("/", 2)
    assert signer.verify(rel, exp, mac) == f.resolve()
    assert signer.verify(rel, exp, "0" * 32) is None
    assert signer.verify("../outside.txt", exp, mac) is None
    assert signer.verify(rel, str(int(time.time()) - 5), mac) is None
    assert signer.sign(tmp_path / "elsewhere.zip") is None


def test_download_server_serves_only_signed_files(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from bot.delivery import start_download_server

    data = tmp_path / "data"
    (data / "projects" / "p1").mkdir(parents=True)
    f = data / "projects" / "p1" / "Model_v1.zip"
    f.write_bytes(b"package bytes")
    (tmp_path / "secret.txt").write_text("outside")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    signer = LinkSigner("s3cret", f"http://127.0.0.1:{port}", data, 1)

    async def run():
        runner = await start_download_server(signer, "127.0.0.1", port)
        try:
            async with aiohttp.ClientSession() as http:
                async with http.get(signer.sign(f)) as r:
                    assert r.status == 200 and await r.read() == b"package bytes"
                    assert "attachment" in r.headers["Content-Disposition"]
                exp = int(time.time()) + 600
                for bad in (f"/d/{exp}/{'0' * 32}/projects/p1/Model_v1.zip", f"/d/{exp}/{'0' * 32}/../secret.txt"):
                    async with http.get(f"http://127.0.0.1:{port}{bad}") as r:
                        assert r.status == 404
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_worker_environment_has_no_secrets():
    from workers.queue import worker_env

    env = worker_env({"DISCORD_TOKEN": "x", "DOWNLOAD_SECRET": "y", "PATH": "/usr/bin", "HOME": "/home/bot"})
    assert "DISCORD_TOKEN" not in env and "DOWNLOAD_SECRET" not in env
    assert env["PATH"] == "/usr/bin" and env["OMP_NUM_THREADS"] == "2"


def test_log_lines_never_contain_a_token():
    import logging

    from utils.logging import RedactingFilter

    token = "M" + "a" * 24 + "." + "b" * 6 + "." + "c" * 30
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    handler.addFilter(RedactingFilter())
    log = logging.getLogger("redaction-test")
    log.addHandler(handler)
    log.propagate = False
    log.error("login failed with %s", token)
    log.removeHandler(handler)
    assert token not in records[0].getMessage() and "[redacted-token]" in records[0].getMessage()
