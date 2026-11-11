"""Colour for the unseen back of a single-photo solid.

Mirroring the photo onto the back repeats every feature (a strawberry, a face)
on both sides and shows a visible fold at the silhouette. Instead the surface
is described in cylindrical coordinates around the vertical axis: per image
row the object spans a half width ``a`` (from the outline) and a half depth
``b`` (from the inflated thickness), and a surface point has an azimuth
``theta`` (0 = facing the camera, +-90 = the outline, 180 = the back).

The visible front (|theta| <= 75 degrees) is unwrapped into a strip
``rows x angle``. The back is filled by image quilting (Efros and Freeman
2001) from that strip:

* patches are only taken from rows close to the target row, so horizontal
  bands (cake layers, a belt, a shirt) continue around the object;
* patches from compact colour regions (an eye, a logo, a single fruit) are
  penalised, while regions that run around the object (icing, fur, a
  stripe) are not, so the back repeats the materials rather than the
  unique details; a source position already used nearby is penalised too,
  so nothing is stamped in a row;
* the strip between 60 and 75 degrees on both sides is copied unchanged and
  the quilt has to join it, so the photo and the synthesised back meet
  without a seam;
* the strip and its horizontal mirror are both used as source, which doubles
  the variety without copying the photo as a whole;
* patches are copied without the photo's lighting (divided by a heavily
  blurred brightness), and the back is re-lit smoothly, so lit and shaded
  patches never end up next to each other as blotches;
* overlaps are joined along minimum-error boundary cuts.

The synthesis runs at reduced resolution and records, for every pixel, where
it was copied from. The full-resolution back is then assembled from the
full-resolution strip with the same placement, so it keeps the photo's
sharpness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

FRONT_LIMIT = 75.0
JOIN_START = 60.0
BLEND = (64.0, 74.0)
LOW_ROWS = 160
PATCH = 28
OVERLAP = 9
ROW_REACH = 0.035
VERTICAL_WEIGHT = 900.0
REUSE_WEIGHT = 2500.0
SOURCE_LIMIT = 58.0
RARITY_WEIGHT = 2500.0
TOLERANCE = 0.15
UNIQUE_LIMIT = 0.08


@dataclass
class RowModel:
    """Per image row: axis position, half width and half depth, all in pixels."""

    y0: int
    center: np.ndarray
    half_width: np.ndarray
    half_depth: np.ndarray

    def lookup(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = np.clip(np.round(y).astype(np.int64) - self.y0, 0, len(self.center) - 1)
        return self.center[i], self.half_width[i], self.half_depth[i]


def row_model(inside: np.ndarray, h2: Optional[np.ndarray], depth_scale: float = 1.0) -> RowModel:
    """Row axis, width and depth from the object mask (full image) and its thickness map.

    Without a thickness map the cross-sections are circles (a lathe object).
    """
    rows = np.flatnonzero(inside.any(axis=1))
    y0, y1 = int(rows[0]), int(rows[-1])
    sub = inside[y0:y1 + 1]
    n = sub.shape[0]
    any_row = sub.any(axis=1)
    left = np.where(any_row, np.argmax(sub, axis=1), 0).astype(np.float64)
    right = np.where(any_row, sub.shape[1] - 1 - np.argmax(sub[:, ::-1], axis=1), 0).astype(np.float64)
    center = (left + right) / 2.0
    half_width = (right - left + 1) / 2.0
    if h2 is None:
        half_depth = half_width.copy() * depth_scale
    else:
        hd = np.sqrt(np.clip(h2, 0, None)).max(axis=1)
        hd = np.interp(np.linspace(0, len(hd) - 1, n), np.arange(len(hd)), hd)
        xs = np.flatnonzero(inside.any(axis=0))
        scale = (xs[-1] - xs[0] + 1) / max(h2.shape[1], 1)
        half_depth = hd * scale * depth_scale
    valid = any_row & (half_width > 0.5)
    idx = np.arange(n)
    if valid.any() and not valid.all():
        for arr in (center, half_width, half_depth):
            arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    k = max(1, n // 120) * 2 + 1
    smooth = lambda v: cv2.GaussianBlur(v.astype(np.float32).reshape(-1, 1), (1, k), 0).ravel().astype(np.float64)
    return RowModel(y0, smooth(center), np.maximum(smooth(half_width), 0.5), np.maximum(smooth(half_depth), 0.5))


def ring_offset(rows: np.ndarray, depth: np.ndarray, tilt: float, cos_theta: np.ndarray, n: int) -> np.ndarray:
    """Photo row minus ring row for a ring point; capped so it never leaves the object's rows."""
    off = depth * tilt * cos_theta
    down = 0.75 * np.clip(n - 1 - rows, 0, None)
    up = 0.75 * np.clip(rows, 0, None)
    return np.clip(off, -up, down)


def _unwrap(image: np.ndarray, model: RowModel, angles_deg: np.ndarray, tilt: float = 0.0) -> np.ndarray:
    """Front strip: rings of the object x azimuths, sampled from the photo.

    ``tilt`` is the sine of the camera elevation. Seen from above, a
    horizontal ring of the object shows up lower in the photo at the front
    than at the sides (by ``half_depth * tilt * cos(theta)``); sampling along
    that curve makes every ring one straight row of the strip.
    """
    n = len(model.center)
    th = np.radians(angles_deg)[None, :]
    x = model.center[:, None] + model.half_width[:, None] * np.sin(th)
    rows = np.arange(n, dtype=np.float64)[:, None]
    y = rows + model.y0 + ring_offset(rows, model.half_depth[:, None], tilt, np.cos(th), n)
    return cv2.remap(image, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _vertical_cut(err: np.ndarray) -> np.ndarray:
    """Column index per row of the minimum-cost top-to-bottom path through ``err``."""
    h, w = err.shape
    cost = err.copy()
    for i in range(1, h):
        prev = cost[i - 1]
        left = np.concatenate([[np.inf], prev[:-1]])
        right = np.concatenate([prev[1:], [np.inf]])
        cost[i] += np.minimum(np.minimum(left, prev), right)
    path = np.zeros(h, np.int64)
    path[-1] = int(np.argmin(cost[-1]))
    for i in range(h - 2, -1, -1):
        j = path[i + 1]
        lo, hi = max(0, j - 1), min(w, j + 2)
        path[i] = lo + int(np.argmin(cost[i, lo:hi]))
    return path


def rarity(strip: np.ndarray, clusters: int = 8, wide: float = 0.55) -> np.ndarray:
    """0..1 per pixel: how much its colour region is a one-off detail.

    The strip is clustered in CIELAB; every connected region of one cluster is
    measured by how far it extends around the object (its width relative to
    the strip). A band that runs around the object (icing, fur, a stripe) is
    material and scores 0; a compact blob (a fruit, an eye, a logo) is a
    detail and scores close to 1.
    """
    h, w = strip.shape[:2]
    lab = cv2.cvtColor(np.clip(strip / 255.0, 0, 1).astype(np.float32) ** (1 / 2.2), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    cv2.setRNGSeed(5)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _, labels, _ = cv2.kmeans(lab * np.array([0.3, 1.4, 1.4], np.float32), clusters, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)
    out = np.zeros((h, w), np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for k in range(clusters):
        m = cv2.morphologyEx((labels == k).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        extent = stats[:, cv2.CC_STAT_WIDTH] / float(w)
        score = np.clip(1.0 - extent / wide, 0.0, 1.0)
        score[0] = 0.0
        sel = (labels == k)
        out[sel] = np.maximum(out[sel], score[comp[sel]])
    return cv2.GaussianBlur(out, (0, 0), 1.0)


def quilt(source: np.ndarray, detail: np.ndarray, target_w: int, fixed_left: np.ndarray, fixed_right: np.ndarray,
          seed: int = 7, source_cols: Optional[list] = None, mirrored_source: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fill a ``rows x target_w`` canvas from ``source``; returns (source row, source col) per pixel.

    ``fixed_left`` / ``fixed_right`` are the source column indices whose
    content is pinned at the first / last columns of the canvas. Pinned
    columns are never replaced, except inside the overlap strip next to the
    synthesised area, where the minimum-error cut decides.
    """
    rng = np.random.default_rng(seed)
    h, sw = source.shape[:2]
    ranges = source_cols or [(0, sw)]
    widest = max(hi - lo for lo, hi in ranges)
    p = int(max(4, min(PATCH, h, widest, target_w - len(fixed_left) - len(fixed_right) + 2 * OVERLAP)))
    o = int(max(1, min(OVERLAP, p // 3)))
    reach = max(2, int(round(ROW_REACH * h)))
    kl, kr = len(fixed_left), len(fixed_right)
    canvas = np.zeros((h, target_w, 3), np.float32)
    filled = np.zeros((h, target_w), bool)
    protected = np.zeros((h, target_w), bool)
    src_r = np.zeros((h, target_w), np.int32)
    src_c = np.zeros((h, target_w), np.int32)
    rows_idx = np.arange(h)[:, None]
    for cols, start in ((fixed_left, 0), (fixed_right, target_w - kr)):
        k = len(cols)
        canvas[:, start:start + k] = source[:, cols]
        filled[:, start:start + k] = True
        src_r[:, start:start + k] = rows_idx
        src_c[:, start:start + k] = cols[None, :]
    protected[:, : max(kl - o, 0)] = True
    protected[:, target_w - kr + o:] = True
    windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(source, (p, p), axis=(0, 1)), 2, -1)
    det = cv2.boxFilter(detail.astype(np.float32), -1, (p, p), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    det = det[: h - p + 1, : sw - p + 1]
    blob = cv2.boxFilter((detail > 0.6).astype(np.float32), -1, (p, p), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    blob = blob[: h - p + 1, : sw - p + 1]
    step = p - o
    tops = list(range(0, max(h - p, 0) + 1, step))
    if tops[-1] != h - p:
        tops.append(h - p)
    first, last = max(kl - o, 0), max(target_w - kr - p + o, 0)
    lefts = list(range(first, last + 1, step))
    if not lefts or lefts[-1] != last:
        lefts.append(last)
    stride = 1 if sw < 240 else 2
    cand_cols = np.concatenate([np.arange(lo, max(lo + 1, min(hi, sw) - p + 1), stride) for lo, hi in ranges])
    used: list = []
    half = sw / 2.0 if mirrored_source else sw * 4.0
    for ty in tops:
        r_lo, r_hi = max(0, ty - reach), min(h - p, ty + reach)
        cand_rows = np.arange(r_lo, r_hi + 1)
        cand = windows[cand_rows][:, cand_cols]
        vertical = VERTICAL_WEIGHT * ((cand_rows - ty) / max(reach, 1)) ** 2
        rare = RARITY_WEIGHT * det[cand_rows][:, cand_cols] ** 2
        share = blob[cand_rows][:, cand_cols]
        unique = np.where(share > UNIQUE_LIMIT, 1e7, 0.0) if (share <= UNIQUE_LIMIT).any() else 0.0
        used = [u for u in used if abs(u[0] - ty) <= 1.5 * p]
        for tx in lefts:
            tgt = canvas[ty:ty + p, tx:tx + p]
            have = filled[ty:ty + p, tx:tx + p]
            if have.any():
                diff = ((cand - tgt[None, None]) ** 2).sum(-1)
                ssd = (diff * have[None, None]).sum((-1, -2)) / have.sum()
            else:
                ssd = np.zeros(cand.shape[:2], np.float32)
            cost = ssd + rare + vertical[:, None] + unique
            if used:
                ux = np.array([u[1] for u in used], np.float64)
                twin = np.where(ux < half, 2 * half - p - ux, 2 * half - p - ux)
                both = np.concatenate([ux, twin])
                near = np.exp(-(((cand_cols[None, :] - both[:, None]) / (0.8 * p)) ** 2)).sum(0)
                cost = cost + REUSE_WEIGHT * near[None, :]
            best = float(cost.min())
            ok = np.argwhere(cost <= best * (1 + TOLERANCE) + 1e-6)
            ri, ci = ok[rng.integers(len(ok))]
            sy, sx = int(cand_rows[ri]), int(cand_cols[ci])
            used.append((ty, sx))
            patch = source[sy:sy + p, sx:sx + p]
            err = ((patch - tgt) ** 2).sum(-1)
            keep_old = np.zeros((p, p), bool)
            if have[:, :o].all():
                cut = _vertical_cut(err[:, :o])
                keep_old |= np.arange(p)[None, :] < cut[:, None]
            if have[:o, :].all():
                cut = _vertical_cut(err[:o, :].T)
                keep_old |= np.arange(p)[:, None] < cut[None, :]
            if have[:, -o:].all():
                cut = _vertical_cut(err[:, -o:])
                keep_old |= np.arange(p)[None, :] > (p - o + cut[:, None])
            new = ~(have & keep_old) & ~protected[ty:ty + p, tx:tx + p]
            yy, xx = np.nonzero(new)
            canvas[ty + yy, tx + xx] = patch[yy, xx]
            src_r[ty + yy, tx + xx] = sy + yy
            src_c[ty + yy, tx + xx] = sx + xx
            filled[ty:ty + p, tx:tx + p] = True
    return src_r, src_c


@dataclass
class Surround:
    """Colour lookup for any point on a single-photo solid, front and back."""

    camera: object
    model: RowModel
    ppm: float
    front: np.ndarray
    back: np.ndarray
    back_start: float
    cols_per_degree: float
    tilt: float = 0.0

    def _coords(self, pos: np.ndarray):
        """Image position and azimuth of surface points.

        On the front the azimuth is taken from the photo column itself
        (``sin(theta) = u``), so photo and synthesised back agree exactly in
        the join band; towards the sides it changes over to the geometric
        angle around the row's ellipse, which is continuous across the
        outline even where the mesh does not reach the row's full width.
        """
        xy, _, _ = self.camera.project(pos)
        c, a, b = self.model.lookup(xy[:, 1])
        u = (xy[:, 0] - c) / a
        w = pos[:, 2] * self.ppm / b
        geometric = np.degrees(np.arctan2(u, -w))
        exact = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))
        t = np.clip((np.abs(geometric) - BLEND[1]) / (88.0 - BLEND[1]), 0.0, 1.0)
        t = np.where(w > 0, 1.0, t * t * (3 - 2 * t))
        theta = exact * (1 - t) + geometric * t
        return xy, theta

    def sample(self, pos: np.ndarray, front: Optional[np.ndarray] = None, back: Optional[np.ndarray] = None) -> np.ndarray:
        front = self.front if front is None else front
        back = self.back if back is None else back
        xy, theta = self._coords(pos)
        _, _, depth = self.model.lookup(xy[:, 1])
        n = len(self.model.center)
        mesh_row = xy[:, 1] - self.model.y0
        ring_row = mesh_row - ring_offset(mesh_row, depth, self.tilt, np.cos(np.radians(theta)), n)
        fh, fw = front.shape[:2]
        sx = xy[:, 0] * fw / self.camera.width
        sy = xy[:, 1] * fh / self.camera.height
        f = _bilinear(front, sx, sy)
        phi = np.mod(theta, 360.0)
        bx = (phi - self.back_start) * self.cols_per_degree
        by = ring_row * back.shape[0] / len(self.model.center)
        b = _bilinear(back, bx, by)
        t = np.clip((np.abs(theta) - BLEND[0]) / (BLEND[1] - BLEND[0]), 0.0, 1.0)
        t = t * t * (3 - 2 * t)
        if f.ndim == 1:
            return f * (1 - t) + b * t
        return f * (1 - t)[:, None] + b * t[:, None]


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)
    fy = (y - y0)
    if img.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def smooth_shading(strip: np.ndarray) -> np.ndarray:
    """Low-frequency brightness of the strip (lighting and the object's own shading)."""
    lum = strip[..., :3].mean(axis=2).astype(np.float32)
    sigma = max(2.0, min(strip.shape[:2]) / 10.0)
    shade = cv2.GaussianBlur(lum, (0, 0), sigma)
    return np.maximum(shade, 0.02)


def _back_shading(shading: np.ndarray, angles: np.ndarray, cpd: float, width: int) -> np.ndarray:
    """Smooth lighting for the back, equal to the photo's lighting in the join bands.

    Quilting moves patches around, so lit and shaded patches would end up next
    to each other as blotches. Patches are therefore copied without their
    lighting; the back gets lighting that runs smoothly from the photo's
    lighting at one join band, through the row's typical brightness, to the
    other join band.
    """
    phi = JOIN_START + np.arange(width) / cpd

    def column(t):
        return shading[:, int(np.clip(np.round((t - angles[0]) * cpd), 0, len(angles) - 1))]

    centre = np.median(shading[:, np.abs(angles) <= 45.0], axis=1)
    left_edge, right_edge = column(FRONT_LIMIT), column(-FRONT_LIMIT)
    out = np.empty((shading.shape[0], width), np.float32)
    for j, p in enumerate(phi):
        t = p - 360.0 if p > 180.0 else p
        if abs(t) <= FRONT_LIMIT:
            out[:, j] = column(t)
            continue
        if p <= 180.0:
            u = (p - FRONT_LIMIT) / (180.0 - FRONT_LIMIT)
            a, b = left_edge, centre
        else:
            u = (p - 180.0) / (180.0 - FRONT_LIMIT)
            a, b = centre, right_edge
        u = u * u * (3 - 2 * u)
        out[:, j] = a * (1 - u) + b * u
    return out


def estimate_tilt(image: np.ndarray, model: RowModel, steps: int = 13, limit: float = 0.6) -> float:
    """Sine of the camera elevation that makes the object's horizontal structure straightest.

    Bands that run around the object (layers, rims, stripes, seams) curve
    downwards towards the middle of a photo taken from above. Each candidate
    elevation straightens the strip differently; the one whose columns agree
    best (lowest spread of lightness across each row) is chosen. Objects
    without horizontal structure score about the same for every candidate
    and keep a level camera.
    """
    angles = np.linspace(-SOURCE_LIMIT, SOURCE_LIMIT, 48)
    rows = len(model.center)
    take = np.linspace(0, rows - 1, min(rows, 160)).astype(np.int64)
    sub = RowModel(model.y0, model.center, model.half_width, model.half_depth)
    gray = cv2.cvtColor(np.clip(image[..., :3], 0, 1).astype(np.float32) ** (1 / 2.2), cv2.COLOR_RGB2LAB)[..., 0]
    scores = []
    candidates = np.linspace(0.0, limit, steps)
    for t in candidates:
        strip = _unwrap(gray, sub, angles, t)[take]
        strip = cv2.GaussianBlur(strip, (0, 0), 1.0)
        scores.append(float(np.mean(np.std(strip, axis=1))))
    scores = np.asarray(scores)
    best = int(np.argmin(scores))
    if scores[0] - scores[best] < 0.04 * scores[0]:
        return 0.0
    return float(candidates[best])


def build_surround(image: np.ndarray, inside: np.ndarray, h2: Optional[np.ndarray], camera, ppm: float, depth_scale: float = 1.0) -> Surround:
    """Synthesise the back of a single-photo solid; ``image`` is float (H, W, C)."""
    model = row_model(inside, h2, depth_scale)
    tilt = estimate_tilt(image, model)
    n = len(model.center)
    perimeter = 2 * np.pi * np.sqrt((model.half_width.max() ** 2 + model.half_depth.max() ** 2) / 2)
    cpd = float(np.clip(perimeter / 360.0, 0.7, 4.5))
    angles = np.arange(-FRONT_LIMIT, FRONT_LIMIT + 1e-6, 1.0 / cpd)
    strip = _unwrap(image.astype(np.float32), model, angles, tilt)
    shading = smooth_shading(strip)
    strip = strip / shading[..., None]
    s = max(1, int(np.ceil(n / LOW_ROWS)))
    low_h = max(PATCH + 2, n // s)
    low_w = max(PATCH + 2, int(round(len(angles) / s)))
    low = cv2.resize(strip, (low_w, low_h), interpolation=cv2.INTER_AREA)
    if low.ndim == 2:
        low = low[..., None]
    low255 = low[..., :3] * (255.0 if low.max() <= 1.5 else 1.0)
    detail = rarity(low255)
    low_cpd = low_w / len(angles) * cpd
    span = 360.0 - 2 * JOIN_START
    target_w = int(round(span * low_cpd))
    join_cols = int(round((FRONT_LIMIT - JOIN_START) * low_cpd))
    right_cols = np.arange(low_w - join_cols, low_w)
    left_cols = np.arange(0, join_cols)
    margin = int(round((FRONT_LIMIT - SOURCE_LIMIT) * low_cpd))
    pool = np.concatenate([low255, low255[:, ::-1]], axis=1)
    pool_detail = np.concatenate([detail, detail[:, ::-1]], axis=1)
    src_r, src_c = quilt(pool, pool_detail, target_w, fixed_left=right_cols, fixed_right=left_cols,
                         source_cols=[(margin, low_w - margin), (low_w + margin, 2 * low_w - margin)], mirrored_source=True)
    strip = np.concatenate([strip, strip[:, ::-1]], axis=1)
    big_w = int(round(span * cpd))
    back_shading = _back_shading(shading, angles, cpd, big_w)
    fy = low_h / n
    fx = target_w / big_w
    yy, xx = np.mgrid[0:n, 0:big_w].astype(np.float32)
    ly = np.clip((yy * fy).astype(np.int64), 0, low_h - 1)
    lx = np.clip((xx * fx).astype(np.int64), 0, target_w - 1)
    sy = (src_r[ly, lx] - ly) / fy + yy
    sx = src_c[ly, lx] * (len(angles) / low_w) + (xx - lx / fx)
    back = cv2.remap(strip, sx.astype(np.float32), np.clip(sy, 0, n - 1).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    back = back * back_shading[..., None]
    return Surround(camera, model, ppm, image, back, JOIN_START, cpd, tilt)


def preview_strip(surround_obj: "Surround") -> np.ndarray:
    """The synthesised back as an image (linear colour), for previews and debugging."""
    return surround_obj.back
