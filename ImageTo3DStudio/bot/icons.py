"""Code-drawn icon set, uploaded once as application emojis.

Every icon is drawn with Pillow primitives at 4x resolution and downsampled, in
the studio's palette (amber accent on light strokes). Uploading them as
application-owned emojis lets buttons and embeds show a consistent custom icon
set without any stock emoji.
"""
from __future__ import annotations

import io
import logging
from typing import Callable, Dict

from PIL import Image, ImageDraw

log = logging.getLogger("studio.icons")

S = 512
INK = (232, 238, 246, 255)
AMBER = (240, 168, 64, 255)
W = 30


def _canvas():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _cube(d):
    top = [(256, 70), (436, 160), (256, 250), (76, 160)]
    d.polygon(top, fill=AMBER)
    d.line([(76, 160), (76, 352), (256, 442), (436, 352), (436, 160)], fill=INK, width=W, joint="curve")
    d.line([(256, 250), (256, 442)], fill=INK, width=W)
    d.line(top + [top[0]], fill=INK, width=W, joint="curve")


def _layers(d):
    for i, y in enumerate((300, 220, 140)):
        pts = [(256, y - 80), (440, y), (256, y + 80), (72, y)]
        d.polygon(pts, fill=AMBER if i == 2 else (0, 0, 0, 0), outline=INK)
        d.line(pts + [pts[0]], fill=INK, width=W, joint="curve")


def _download(d):
    d.line([(256, 70), (256, 330)], fill=INK, width=W + 10)
    d.polygon([(150, 250), (362, 250), (256, 370)], fill=AMBER)
    d.line([(90, 380), (90, 440), (422, 440), (422, 380)], fill=INK, width=W, joint="curve")


def _sliders(d):
    for x, knob in ((140, 330), (256, 170), (372, 290)):
        d.line([(x, 80), (x, 432)], fill=INK, width=W - 6)
        d.ellipse([x - 48, knob - 48, x + 48, knob + 48], fill=AMBER, outline=INK, width=W - 10)


def _gear(d):
    import math

    cx = cy = 256
    for k in range(8):
        a = k * math.pi / 4
        x, y = cx + math.cos(a) * 170, cy + math.sin(a) * 170
        d.ellipse([x - 42, y - 42, x + 42, y + 42], fill=INK)
    d.ellipse([cx - 150, cy - 150, cx + 150, cy + 150], fill=INK)
    d.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], fill=AMBER)


def _clock(d):
    d.ellipse([70, 70, 442, 442], outline=INK, width=W)
    d.line([(256, 256), (256, 140)], fill=AMBER, width=W + 6)
    d.line([(256, 256), (346, 300)], fill=INK, width=W)
    d.ellipse([236, 236, 276, 276], fill=INK)


def _book(d):
    d.polygon([(256, 140), (256, 430), (80, 380), (80, 90)], outline=INK, fill=(0, 0, 0, 0))
    d.line([(256, 140), (256, 430), (80, 380), (80, 90), (256, 140)], fill=INK, width=W, joint="curve")
    d.polygon([(256, 140), (432, 90), (432, 380), (256, 430)], fill=AMBER)
    d.line([(256, 140), (432, 90), (432, 380), (256, 430)], fill=INK, width=W, joint="curve")


def _mesh(d):
    pts = [(256, 70), (440, 400), (72, 400)]
    mids = [((pts[i][0] + pts[(i + 1) % 3][0]) // 2, (pts[i][1] + pts[(i + 1) % 3][1]) // 2) for i in range(3)]
    d.polygon(mids, fill=AMBER)
    d.line(pts + [pts[0]], fill=INK, width=W, joint="curve")
    d.line(mids + [mids[0]], fill=INK, width=W - 8, joint="curve")


def _palette(d):
    d.ellipse([70, 80, 442, 430], outline=INK, width=W)
    for (x, y), col in zip(((170, 190), (260, 150), (350, 200)), (AMBER, INK, AMBER)):
        d.ellipse([x - 36, y - 36, x + 36, y + 36], fill=col)
    d.ellipse([300, 300, 380, 380], fill=(0, 0, 0, 0), outline=INK, width=W - 8)


def _ruler(d):
    d.polygon([(90, 330), (330, 90), (422, 182), (182, 422)], outline=INK, fill=(0, 0, 0, 0))
    d.line([(90, 330), (330, 90), (422, 182), (182, 422), (90, 330)], fill=INK, width=W, joint="curve")
    for k in range(1, 6):
        x = 90 + k * 40
        y = 330 - k * 40
        L = 60 if k % 2 else 36
        d.line([(x, y), (x + L * 0.7, y + L * 0.7)], fill=AMBER, width=W - 10)


def _refresh(d):
    d.arc([90, 90, 422, 422], 30, 320, fill=INK, width=W)
    d.polygon([(380, 70), (440, 200), (300, 190)], fill=AMBER)


def _cross(d):
    d.line([(120, 120), (392, 392)], fill=INK, width=W + 12)
    d.line([(392, 120), (120, 392)], fill=AMBER, width=W + 12)


ICONS: Dict[str, Callable] = {
    "cube": _cube, "layers": _layers, "download": _download, "sliders": _sliders, "gear": _gear,
    "clock": _clock, "book": _book, "mesh": _mesh, "palette": _palette, "ruler": _ruler,
    "refresh": _refresh, "cross": _cross,
}
PREFIX = "i3d_"


def draw_icon(name: str, size: int = 128) -> bytes:
    img, d = _canvas()
    ICONS[name](d)
    out = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()


class IconSet:
    """Resolves icon names to application emojis when they are available."""

    def __init__(self):
        self.emojis: Dict[str, object] = {}

    def get(self, name: str):
        return self.emojis.get(name)

    async def sync(self, client) -> int:
        try:
            existing = {e.name: e for e in await client.fetch_application_emojis()}
        except Exception as exc:
            log.warning("could not list application emojis: %s", exc)
            return 0
        created = 0
        for name in ICONS:
            key = PREFIX + name
            emoji = existing.get(key)
            if emoji is None:
                try:
                    emoji = await client.create_application_emoji(name=key, image=draw_icon(name))
                    created += 1
                except Exception as exc:
                    log.warning("could not upload icon %s: %s", key, exc)
                    continue
            self.emojis[name] = emoji
        log.info("icon set ready (%d icons, %d uploaded now)", len(self.emojis), created)
        return created
