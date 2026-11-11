"""Rounded volumes from one silhouette (medial-ball inflation).

The solid is the union of all balls whose equatorial disc fits inside the
silhouette (Twarog, Tappen and Adelson 2012, "Puffball"). A circular outline
becomes a sphere, a bar becomes a cylinder with round ends, and a part that is
narrow in the photo (a handle, a cone on top of a cake) stays as thin in depth
as it is wide. The squared half thickness is

    H2(p) = max over inside pixels c of  D(c)^2 - |p - c|^2

with D the distance to the outline. That maximum is the lower envelope of
parabolas, computed exactly with the two-pass transform of Felzenszwalb and
Huttenlocher (2012).

Objects that stand on the ground end in a straight bottom edge. The outline is
then mirrored below the ground line before inflating, so the base keeps its
full depth and is cut flat instead of tapering to a point.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

WORK_SIDE = 288
FLAT_BAND = 0.02
FLAT_SHARE = 0.3


def smooth_outline(mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove pixel stairs and segmentation burrs without moving the outline."""
    m = mask.astype(np.uint8)
    side = max(m.shape)
    sigma = max(0.8, side / 220.0) * float(strength)
    blurred = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    out = blurred > 0.5
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
    if n > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = labels == keep
    return out


def flat_edge_share(mask: np.ndarray, side: str = "bottom") -> float:
    """Share of the widest row that the object's straight bottom (or top) edge covers."""
    m = mask if side == "bottom" else mask[::-1]
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if len(rows) < 8 or len(cols) < 8:
        return 0.0
    height = rows[-1] - rows[0] + 1
    band = max(2, int(round(FLAT_BAND * height)))
    sub = m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    lowest = sub.shape[0] - 1 - np.argmax(sub[::-1], axis=0)
    occupied = sub.any(axis=0)
    touching = occupied & (lowest >= sub.shape[0] - band)
    widest = float(sub.sum(axis=1).max())
    return float(touching.sum() / max(widest, 1.0))


def _envelope_1d(f: np.ndarray) -> np.ndarray:
    """min over q of (p - q)^2 + f[q] for every p; f may contain +inf."""
    n = len(f)
    out = np.full(n, np.inf)
    finite = np.flatnonzero(np.isfinite(f))
    if len(finite) == 0:
        return out
    v = np.zeros(len(finite), np.int64)
    z = np.zeros(len(finite) + 1)
    k = 0
    v[0] = finite[0]
    z[0] = -np.inf
    z[1] = np.inf
    for q in finite[1:]:
        fq = f[q]
        while True:
            p = v[k]
            s = ((fq + q * q) - (f[p] + p * p)) / (2.0 * (q - p))
            if s <= z[k] and k > 0:
                k -= 1
            else:
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    idx = np.searchsorted(z[1:k + 1], np.arange(n), side="left")
    src = v[idx]
    d = np.arange(n) - src
    return d * d + f[src]


def ball_field(mask: np.ndarray) -> np.ndarray:
    """Squared half thickness in px^2: positive inside the outline, negative outside."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float64)
    radius = np.maximum(dist - 0.5, 0.0)
    f = np.where(m > 0, -(radius * radius), np.inf)
    tmp = np.empty_like(f)
    for x in range(f.shape[1]):
        tmp[:, x] = _envelope_1d(f[:, x])
    out = np.empty_like(f)
    for y in range(f.shape[0]):
        out[y] = _envelope_1d(tmp[y])
    return -out


def thickness_field(mask: np.ndarray, flat_base: bool, flat_top: bool = False) -> Tuple[np.ndarray, float]:
    """Half-thickness-squared map at working resolution and its px scale relative to the mask."""
    h, w = mask.shape
    scale = min(1.0, WORK_SIDE / max(h, w))
    small = cv2.resize(mask.astype(np.uint8), (max(8, int(round(w * scale))), max(8, int(round(h * scale)))), interpolation=cv2.INTER_AREA) > 0 if scale < 1 else mask.astype(bool)
    sh = small.shape[0]
    pad = 2
    body = np.pad(small, ((0, 0), (pad, pad)))
    stack = [body]
    top_rows = 0
    if flat_top:
        stack.insert(0, body[::-1])
        top_rows = sh
    else:
        stack.insert(0, np.zeros((pad, body.shape[1]), bool))
        top_rows = pad
    if flat_base:
        stack.append(body[::-1])
    else:
        stack.append(np.zeros((pad, body.shape[1]), bool))
    field = ball_field(np.vstack(stack))
    return field[top_rows:top_rows + sh, pad:-pad], scale


def volume_from_thickness(h2_px: np.ndarray, dims, res: int, depth_scale: float = 1.0):
    """Inside-positive field of the inflated silhouette on a voxel grid.

    ``h2_px`` is the map from ``thickness_field`` (rows top-down, columns
    left-right); image left is the object's +X because the camera looks at the
    front (-Z). Returns (field[x, y, z], origin, voxel step) in studs.
    """
    width, height = float(dims[0]), float(dims[1])
    step = max(width, height) / res
    nx = max(6, int(round(width / step)))
    ny = max(6, int(round(height / step)))
    px = height / max(h2_px.shape[0], 1)
    interp = cv2.INTER_AREA if nx < h2_px.shape[1] else cv2.INTER_LINEAR
    h2 = cv2.resize(h2_px.astype(np.float32), (nx, ny), interpolation=interp).astype(np.float64) * px * px
    peak = float(h2.max())
    if peak <= 0:
        raise ValueError("empty silhouette")
    half_depth = np.sqrt(peak) * depth_scale
    nz = max(8, int(np.ceil(2 * half_depth / step)) + 4)
    zc = (np.arange(nz) - (nz - 1) / 2) * step
    fld = (h2[:, :, None] - (zc[None, None, :] / depth_scale) ** 2) / (2.0 * np.sqrt(peak))
    vol = np.ascontiguousarray(np.transpose(fld[::-1, ::-1, :], (1, 0, 2)), dtype=np.float32)
    return vol, [-width / 2, 0.0, float(zc[0])], step


def extend_colours(rgb: np.ndarray, inside: np.ndarray, inset: float = 0.012) -> np.ndarray:
    """Photo whose pixels at and beyond the outline repeat the nearest object colour.

    Outline pixels are a blend of object and background. They are replaced by
    the colour found a few pixels further in, so the rim and the mirrored back
    of the model never show a halo of the background.
    """
    h, w = inside.shape
    margin = max(2, int(round(inset * max(np.ptp(np.nonzero(inside)[0]) + 1, np.ptp(np.nonzero(inside)[1]) + 1))))
    core = cv2.erode(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))) > 0
    if core.sum() < 16:
        core = inside
    _, labels = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(core)
    lut = np.zeros((labels.max() + 1, 2), np.int64)
    lut[labels[ys, xs]] = np.stack([ys, xs], axis=1)
    src = lut[labels]
    out = rgb[src[..., 0], src[..., 1]]
    soft = cv2.GaussianBlur(out, (0, 0), max(1.0, margin / 2.0))
    out = np.where(core[..., None], rgb, soft)
    return out.astype(rgb.dtype)


def photo_relief(rgb: np.ndarray, inside: np.ndarray, height_studs: float, amount: float = 0.006) -> np.ndarray:
    """Fine surface relief (studs) read from the photo's local brightness.

    Small bright details (sprinkles, stitches, rims, highlights on ridges)
    stand out from their surroundings, small dark ones (grooves, gaps) sink in.
    Only the band-passed lightness is used, so broad shading and large colour
    areas stay flat; the result feeds the normal map, never the mesh.
    """
    ys, xs = np.nonzero(inside)
    if len(ys) == 0:
        return np.zeros(inside.shape, np.float32)
    size = float(max(np.ptp(ys), np.ptp(xs)) + 1)
    light = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0
    fine = cv2.GaussianBlur(light, (0, 0), max(0.8, size / 400.0))
    broad = cv2.GaussianBlur(light, (0, 0), max(2.5, size / 45.0))
    band = np.tanh((fine - broad) / 0.08)
    return (band * amount * float(height_studs)).astype(np.float32)
