"""Draw the flat emblem test image (no Blender needed).

    python tools/make_drawing.py assets/test_images/emblem_drawing.png
"""
import math
import sys

from PIL import Image, ImageDraw

SS = 4
SIZE = 800


def star(cx: float, cy: float, r_out: float, r_in: float, points: int = 5):
    pts = []
    for k in range(points * 2):
        r = r_out if k % 2 == 0 else r_in
        a = -math.pi / 2 + k * math.pi / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def main(out: str) -> None:
    s = SIZE * SS
    img = Image.new("RGB", (s, s), (250, 250, 247))
    d = ImageDraw.Draw(img)
    c = s / 2
    d.polygon(star(c, c * 1.04, s * 0.40, s * 0.18), fill=(58, 42, 18))
    d.polygon(star(c, c * 1.04, s * 0.40 - 14 * SS, s * 0.18 - 7 * SS), fill=(232, 178, 58))
    hole = s * 0.07
    d.ellipse([c - hole - 10 * SS, c * 1.04 - hole - 10 * SS, c + hole + 10 * SS, c * 1.04 + hole + 10 * SS], fill=(58, 42, 18))
    d.ellipse([c - hole, c * 1.04 - hole, c + hole, c * 1.04 + hole], fill=(250, 250, 247))
    img.resize((SIZE, SIZE), Image.LANCZOS).save(out, optimize=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "emblem_drawing.png")
