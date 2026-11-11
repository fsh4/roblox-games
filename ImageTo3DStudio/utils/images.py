"""Validation of untrusted uploads before they reach the engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"RIFF", "webp"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)


class UploadRejected(ValueError):
    pass


@dataclass
class UploadCheck:
    kind: str
    width: int
    height: int


def sniff(data: bytes) -> Optional[str]:
    for magic, kind in MAGIC:
        if data.startswith(magic):
            if kind == "webp" and data[8:12] != b"WEBP":
                return None
            return kind
    return None


def check_upload(filename: str, data: bytes, max_bytes: int, max_pixels: int) -> UploadCheck:
    """Reject anything that is not a reasonably sized raster image.

    The decision uses the file's own bytes (magic number + a full decode with
    Pillow's decompression-bomb guard), never the extension or the content type
    claimed by the client. Files are never executed or extracted.
    """
    name = (filename or "").lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise UploadRejected(f"`{filename}` is not an image file (allowed: PNG, JPG, WEBP, BMP, TIFF, GIF)")
    if len(data) == 0:
        raise UploadRejected(f"`{filename}` is empty")
    if len(data) > max_bytes:
        raise UploadRejected(f"`{filename}` is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB")
    kind = sniff(data)
    if kind is None:
        raise UploadRejected(f"`{filename}` does not look like a supported image")
    import io

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w * h > max_pixels:
                raise UploadRejected(f"`{filename}` is {w}x{h}; the limit is {max_pixels // 1_000_000} megapixels")
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except UploadRejected:
        raise
    except Image.DecompressionBombError as exc:
        raise UploadRejected(f"`{filename}` was rejected as a decompression bomb") from exc
    except Exception as exc:
        raise UploadRejected(f"`{filename}` could not be decoded ({exc.__class__.__name__})") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    if min(w, h) < 48:
        raise UploadRejected(f"`{filename}` is too small ({w}x{h}); use at least 48 px on the short side")
    return UploadCheck(kind, w, h)
