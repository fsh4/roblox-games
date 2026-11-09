"""Texture production: UV atlas, material synthesis, photo projection and baking.

Pipeline
--------
1. ``pack_atlas``: split every part into UV islands (connected faces in the
   physical-unit UV layout), choose one texel density for the whole model and
   shelf-pack the islands into a square atlas with padding.
2. Rasterize all triangles in UV space so every texel knows its triangle,
   3D position, normal, tangent frame, part and physical UV.
3. Evaluate materials per texel (solid wood, fabric weave, brushed metal ...)
   or synthesise from a photo exemplar with histogram-preserving tiling.
4. Optionally project the source photograph (visibility tested with a depth
   map from the estimated camera, mirrored for hidden sides).
5. Bake ambient occlusion with many orthographic depth maps, bake shape detail
   from a high-resolution mesh into the normal map, add material micro-relief.
6. Fill the gutters (push-pull) so mip-mapping never bleeds background.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import render
from .geometry import Mesh
from .materials import MaterialSpec, linear_to_srgb, srgb_to_linear
from .scene import Part, Scene

Progress = Optional[Callable[[str, float], None]]


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

class Noise:
    """Seeded Perlin gradient noise, fBm and Worley noise (vectorised)."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        p = rng.permutation(256)
        self.perm = np.concatenate([p, p]).astype(np.int64)
        ang = rng.uniform(0, 2 * np.pi, 256)
        self.g2 = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        g3 = rng.normal(size=(256, 3))
        self.g3 = g3 / np.linalg.norm(g3, axis=1, keepdims=True)
        self.points = rng.uniform(0, 1, size=(256, 2))

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def noise2(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        xf, yf = x - xi, y - yi
        u, v = self._fade(xf), self._fade(yf)
        p = self.perm
        X, Y = xi & 255, yi & 255

        def grad(h, dx, dy):
            g = self.g2[h & 255]
            return g[..., 0] * dx + g[..., 1] * dy

        aa = p[p[X] + Y]
        ab = p[p[X] + Y + 1]
        ba = p[p[X + 1] + Y]
        bb = p[p[X + 1] + Y + 1]
        x1 = grad(aa, xf, yf) * (1 - u) + grad(ba, xf - 1, yf) * u
        x2 = grad(ab, xf, yf - 1) * (1 - u) + grad(bb, xf - 1, yf - 1) * u
        return (x1 * (1 - v) + x2 * v) * 1.41

    def noise3(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
        xf, yf, zf = x - xi, y - yi, z - zi
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        p = self.perm
        X, Y, Z = xi & 255, yi & 255, zi & 255

        def grad(h, dx, dy, dz):
            g = self.g3[h & 255]
            return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

        A, B = p[X] + Y, p[X + 1] + Y
        AA, AB, BA, BB = p[A] + Z, p[A + 1] + Z, p[B] + Z, p[B + 1] + Z
        x1 = grad(p[AA], xf, yf, zf) * (1 - u) + grad(p[BA], xf - 1, yf, zf) * u
        x2 = grad(p[AB], xf, yf - 1, zf) * (1 - u) + grad(p[BB], xf - 1, yf - 1, zf) * u
        y1 = x1 * (1 - v) + x2 * v
        x3 = grad(p[AA + 1], xf, yf, zf - 1) * (1 - u) + grad(p[BA + 1], xf - 1, yf, zf - 1) * u
        x4 = grad(p[AB + 1], xf, yf - 1, zf - 1) * (1 - u) + grad(p[BB + 1], xf - 1, yf - 1, zf - 1) * u
        y2 = x3 * (1 - v) + x4 * v
        return (y1 * (1 - w) + y2 * w) * 1.15

    def fbm2(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total = total + amp * self.noise2(x * freq, y * freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def fbm3(self, x, y, z, octaves=4, lacunarity=2.0, gain=0.5):
        total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total = total + amp * self.noise3(x * freq, y * freq, z * freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def worley2(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        best = np.full(x.shape, 9.0)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cx, cy = xi + dx, yi + dy
                h = self.perm[(self.perm[cx & 255] + cy) & 255]
                pt = self.points[h]
                d = np.hypot(cx + pt[..., 0] - x, cy + pt[..., 1] - y)
                best = np.minimum(best, d)
        return best


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Atlas packing
# ---------------------------------------------------------------------------

@dataclass
class Island:
    part: int
    faces: np.ndarray
    vmin: np.ndarray
    size: np.ndarray
    rotated: bool = False
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class Atlas:
    resolution: int
    density: float
    padding: int
    islands: List[Island]
    phys_uvs: List[np.ndarray]
    utilization: float


def find_islands(mesh: Mesh) -> List[np.ndarray]:
    n_v = len(mesh.vertices)
    if len(mesh.faces) == 0:
        return []
    f = mesh.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_v, n_v))
    _, labels = connected_components(adj, directed=False)
    face_label = labels[f[:, 0]]
    order = np.argsort(face_label, kind="stable")
    splits = np.flatnonzero(np.diff(face_label[order])) + 1
    return np.split(order, splits)


def _shelf_pack(sizes: np.ndarray, res: int) -> Optional[np.ndarray]:
    """Pack (w,h) integer rectangles; returns (N,2) positions or None."""
    order = np.lexsort((-sizes[:, 0], -sizes[:, 1]))
    pos = np.zeros((len(sizes), 2), dtype=np.int64)
    x = y = shelf_h = 0
    for i in order:
        w, h = int(sizes[i, 0]), int(sizes[i, 1])
        if w > res or h > res:
            return None
        if x + w > res:
            y += shelf_h
            x = 0
            shelf_h = 0
        if y + h > res:
            return None
        pos[i] = (x, y)
        x += w
        shelf_h = max(shelf_h, h)
    return pos


def pack_atlas(meshes: Sequence[Mesh], resolution: int, padding: int = 4) -> Atlas:
    islands: List[Island] = []
    phys = []
    for pi, m in enumerate(meshes):
        phys.append(m.uvs.copy())
        for faces in find_islands(m):
            vid = np.unique(m.faces[faces])
            uv = m.uvs[vid]
            lo, hi = uv.min(0), uv.max(0)
            size = np.maximum(hi - lo, 1e-4)
            islands.append(Island(pi, faces, lo, size, rotated=bool(size[1] > size[0] * 1.0001)))
    if not islands:
        return Atlas(resolution, 1.0, padding, [], phys, 0.0)
    sizes = np.array([[i.size[1], i.size[0]] if i.rotated else [i.size[0], i.size[1]] for i in islands])
    total_area = float((sizes[:, 0] * sizes[:, 1]).sum())
    hi_d = np.sqrt(resolution * resolution / max(total_area, 1e-9))
    lo_d = 0.0
    best = None
    for _ in range(24):
        d = 0.5 * (lo_d + hi_d) if best is not None or lo_d > 0 else hi_d * 0.7
        px = np.maximum(np.ceil(sizes * d), 1).astype(np.int64) + 2 * padding
        pos = _shelf_pack(px, resolution)
        if pos is not None:
            best = (d, pos, px)
            lo_d = d
        else:
            hi_d = d
        if hi_d - lo_d < hi_d * 0.004:
            break
    if best is None:
        d = lo_d if lo_d > 0 else 1e-3
        px = np.maximum(np.ceil(sizes * d), 1).astype(np.int64) + 2 * padding
        pos = _shelf_pack(px, resolution)
        if pos is None:
            raise RuntimeError("UV atlas packing failed: too many islands for the resolution")
        best = (d, pos, px)
    d, pos, px = best
    used = 0.0
    for isl, p, s in zip(islands, pos, px):
        isl.rect = (int(p[0]), int(p[1]), int(s[0]), int(s[1]))
        used += (s[0] - 2 * padding) * (s[1] - 2 * padding)
    atlas = Atlas(resolution, float(d), padding, islands, phys, float(used / (resolution * resolution)))
    return atlas


def apply_atlas_uvs(meshes: Sequence[Mesh], atlas: Atlas) -> None:
    res = float(atlas.resolution)
    for isl in atlas.islands:
        m = meshes[isl.part]
        vid = np.unique(m.faces[isl.faces])
        local = atlas.phys_uvs[isl.part][vid] - isl.vmin
        if isl.rotated:
            local = np.stack([local[:, 1], isl.size[0] - local[:, 0]], axis=1)
        x0, y0, w, h = isl.rect
        px = x0 + atlas.padding + local[:, 0] * atlas.density
        py = y0 + atlas.padding + local[:, 1] * atlas.density
        m.uvs[vid] = np.stack([px / res, 1.0 - py / res], axis=1)


# ---------------------------------------------------------------------------
# Texel attributes
# ---------------------------------------------------------------------------

@dataclass
class Texels:
    res: int
    mask: np.ndarray
    part: np.ndarray
    face: np.ndarray
    pos: np.ndarray
    nrm: np.ndarray
    tangent: np.ndarray
    bitangent: np.ndarray
    phys_uv: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def texel_attributes(meshes: Sequence[Mesh], atlas: Atlas) -> Texels:
    res = atlas.resolution
    verts, faces, nrms, uvs, puv, pid, tans = [], [], [], [], [], [], []
    off = 0
    for i, m in enumerate(meshes):
        verts.append(m.vertices)
        faces.append(m.faces + off)
        nrms.append(m.normals)
        uvs.append(m.uvs)
        puv.append(atlas.phys_uvs[i])
        pid.append(np.full(len(m.faces), i, dtype=np.int64))
        tans.append(render.vertex_tangents(m.vertices, m.faces, m.normals, m.uvs))
        off += len(m.vertices)
    V = np.concatenate(verts)
    F = np.concatenate(faces)
    N = np.concatenate(nrms)
    UV = np.concatenate(uvs)
    PUV = np.concatenate(puv)
    PID = np.concatenate(pid)
    TAN = np.concatenate(tans)
    xy = np.stack([UV[:, 0] * res, (1.0 - UV[:, 1]) * res], axis=1)
    ras = render.rasterize(xy, np.zeros(len(V)), F, res, res)
    mask = ras.mask
    rows, cols = np.nonzero(mask)
    fid = ras.face_id[rows, cols]
    b = ras.bary[rows, cols].astype(np.float64)
    tri = F[fid]

    def interp(a):
        return np.einsum("nk,nkc->nc", b, a[tri])

    pos = interp(V)
    nrm = interp(N)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    t4 = interp(TAN)
    t = t4[:, :3] - nrm * np.sum(nrm * t4[:, :3], axis=1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    sign = np.where(t4[:, 3] < 0, -1.0, 1.0)[:, None]
    bt = np.cross(nrm, t) * sign
    return Texels(res, mask, PID[fid], fid, pos, nrm, t, bt, interp(PUV), rows, cols)


# ---------------------------------------------------------------------------
# Material evaluation
# ---------------------------------------------------------------------------

@dataclass
class Channels:
    albedo: np.ndarray
    height: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    emissive: np.ndarray
    alpha: np.ndarray


def _empty_channels(n: int) -> Channels:
    return Channels(np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n))


def _tone(base_lin: np.ndarray, t: np.ndarray, variation: float) -> np.ndarray:
    """Blend around the measured colour so the average stays on ``base``."""
    light = np.clip(base_lin * (1.0 + variation), 0, 1)
    dark = np.clip(base_lin * (1.0 - 1.5 * variation), 0, 1)
    t = np.clip(t, 0, 1)[:, None]
    return light * (1 - t) + dark * t


def eval_material(mat: MaterialSpec, local: np.ndarray, uv: np.ndarray, grain_axis: int, seed: int) -> Channels:
    n = len(uv)
    ch = _empty_channels(n)
    base = srgb_to_linear(np.asarray(mat.base_color))
    noise = Noise(seed * 7919 + zlib.crc32(mat.key.encode()) % 1000)
    kind = mat.kind
    var = mat.color_variation
    ch.roughness[:] = mat.roughness
    ch.metallic[:] = mat.metallic
    s = max(mat.pattern_scale, 1e-3)
    if kind == "wood":
        a = local[:, grain_axis]
        others = [k for k in range(3) if k != grain_axis]
        b, c = local[:, others[0]], local[:, others[1]]
        rng = np.random.default_rng(seed + 11)
        b0 = rng.uniform(-3.0, 3.0) + np.sign(rng.uniform(-1, 1)) * 1.5
        c0 = rng.uniform(-3.0, 3.0) - 2.0
        warp = noise.fbm3(a * 0.45, b * 1.4, c * 1.4, 3) * 0.06 * s
        r = np.hypot(b - b0, c - c0) + warp + noise.noise3(a * 0.08, b * 0.3, c * 0.3) * 0.15
        spacing = 0.085 * s
        ring = (r / spacing) % 1.0
        late = _smoothstep(0.55, 0.78, ring) * (1 - _smoothstep(0.88, 1.0, ring))
        fibers = noise.noise3(a * 1.5 / s, b * 55 / s, c * 55 / s) * 0.5 + 0.5
        pores = np.clip(noise.noise3(a * 6 / s, b * 140 / s, c * 140 / s), 0, 1)
        tone = 0.62 * late + 0.28 * fibers + 0.25 * pores - 0.18
        tone += noise.fbm3(a * 0.25, b * 0.6, c * 0.6, 2) * 0.25
        ch.albedo = _tone(base, tone, var)
        if mat.secondary_color is not None:
            sec = srgb_to_linear(np.asarray(mat.secondary_color))
            ch.albedo = ch.albedo * 0.85 + sec[None, :] * 0.15 * (late[:, None])
        ch.height = -late * 0.004 - pores * 0.0015 + fibers * 0.0008
        ch.roughness = np.clip(mat.roughness + (late - 0.3) * 0.08 + pores * 0.05, 0.05, 1)
    elif kind == "fabric":
        u, v = uv[:, 0], uv[:, 1]
        k = 2 * np.pi / max(0.035 * s / 0.06, 1e-3)
        weave = np.sin(u * k) * np.sin(v * k)
        mottling = noise.fbm2(u * 1.6, v * 1.6, 3)
        fuzz = noise.noise2(u * 90, v * 90)
        tone = 0.5 + 0.35 * mottling + 0.15 * fuzz + 0.1 * weave
        h = weave * 0.0006 + fuzz * 0.0003
        if mat.quilted:
            cell = 0.75
            du = np.abs(((u + v * 0.0) / cell) % 1.0 - 0.5)
            dv = np.abs(((v) / cell) % 1.0 - 0.5)
            seam = np.maximum(_smoothstep(0.43, 0.5, du), _smoothstep(0.43, 0.5, dv))
            puff = (0.5 - du) * (0.5 - dv)
            h = h + puff * 0.04 - seam * 0.01
            tone = tone + seam * 0.35
        ch.albedo = _tone(base, (tone - 0.5) * 2 * 0.5 + 0.5, var)
        ch.height = h
        ch.roughness = np.clip(mat.roughness + fuzz * 0.03, 0.05, 1)
    elif kind == "metal":
        u, v = uv[:, 0], uv[:, 1]
        streak = noise.noise2(u * 0.6, v * 70) * 0.6 + noise.noise2(u * 3, v * 180) * 0.4
        blotch = noise.fbm2(u * 0.8, v * 0.8, 3)
        ch.albedo = _tone(base, 0.5 + 0.25 * streak + 0.2 * blotch, var)
        ch.height = streak * 0.0003
        ch.roughness = np.clip(mat.roughness + streak * 0.06 + blotch * 0.05, 0.05, 1)
    elif kind == "leather":
        u, v = uv[:, 0], uv[:, 1]
        cell = noise.worley2(u * 30 / max(s / 0.05, 1e-3), v * 30 / max(s / 0.05, 1e-3))
        crease = _smoothstep(0.0, 0.35, cell)
        mott = noise.fbm2(u * 1.2, v * 1.2, 3)
        ch.albedo = _tone(base, 0.5 + 0.3 * mott + 0.3 * (1 - crease), var)
        ch.height = crease * 0.002
        ch.roughness = np.clip(mat.roughness - crease * 0.06 + 0.04, 0.05, 1)
    elif kind == "stone":
        f = noise.fbm3(local[:, 0] * 1.3 / s, local[:, 1] * 1.3 / s, local[:, 2] * 1.3 / s, 5)
        cell = noise.worley2(uv[:, 0] * 6 / s, uv[:, 1] * 6 / s)
        ch.albedo = _tone(base, 0.5 + 0.45 * f + 0.2 * cell, var)
        ch.height = f * 0.006 + cell * 0.002
        ch.roughness = np.clip(mat.roughness + f * 0.05, 0.05, 1)
    elif kind == "emissive":
        ch.albedo = np.tile(base, (n, 1))
        ch.emissive[:] = 1.0 if mat.emissive <= 0 else min(mat.emissive, 1.0)
    elif kind == "glass":
        ch.albedo = np.tile(base, (n, 1))
        ch.alpha[:] = mat.opacity
    else:
        f = noise.fbm3(local[:, 0] * 0.9, local[:, 1] * 0.9, local[:, 2] * 0.9, 3)
        micro = noise.noise3(local[:, 0] * 25, local[:, 1] * 25, local[:, 2] * 25)
        ch.albedo = _tone(base, 0.5 + 0.4 * f + 0.1 * micro, var)
        ch.height = micro * 0.0004
        ch.roughness = np.clip(mat.roughness + f * 0.04, 0.05, 1)
    if mat.emissive > 0 and kind != "emissive":
        ch.emissive[:] = min(mat.emissive, 1.0)
    if mat.opacity < 0.999:
        ch.alpha[:] = mat.opacity
    ch.height = ch.height * mat.normal_strength / 0.5 if mat.normal_strength > 0 else ch.height * 0
    return ch


# ---------------------------------------------------------------------------
# Exemplar-based synthesis (histogram-preserving stochastic tiling)
# ---------------------------------------------------------------------------

class Exemplar:
    """Heitz & Neyret (2018) by-example noise on a triangle grid.

    The exemplar is Gaussianised per decorrelated colour channel. Every texel
    blends three copies of the exemplar, each centred on a vertex of a triangle
    grid with its own random offset, using variance-preserving weights; the
    result goes back through the inverse histogram. Each copy only covers the
    hexagon around its vertex and offsets keep that footprint inside the
    exemplar, so non-tileable photo patches never show wrap seams.
    """

    SPAN = 0.3

    def __init__(self, image_lin: np.ndarray, px_per_stud: float, seed: int = 0):
        from scipy.special import ndtri

        img = np.asarray(image_lin, dtype=np.float64)
        h, w = img.shape[:2]
        self.h, self.w = h, w
        flat = img.reshape(-1, 3)
        self.mean = flat.mean(0)
        cov = np.cov((flat - self.mean).T) + np.eye(3) * 1e-8
        _, vecs = np.linalg.eigh(cov)
        self.basis = vecs
        dec = (flat - self.mean) @ vecs
        n = len(dec)
        gvals = ndtri((np.arange(n) + 0.5) / n)
        gauss = np.zeros_like(dec)
        self.lut = []
        for c in range(3):
            order = np.argsort(dec[:, c], kind="stable")
            gauss[order, c] = gvals
            self.lut.append(np.sort(dec[:, c]))
        self.gauss = gauss.reshape(h, w, 3)
        self.tile = self.SPAN * np.array([w, h], dtype=np.float64) / max(px_per_stud, 1e-6)
        self.seed = int(seed)

    def _offset(self, ix, iy):
        key = ((ix * 73856093) ^ (iy * 19349663) ^ (self.seed * 83492791)) & 0x7FFFFFFF
        ox = ((key * 2654435761) % 1000003) / 1000003.0
        oy = ((key * 2246822519) % 999983) / 999983.0
        lo = self.SPAN * 1.2
        return lo + ox * (1 - 2 * lo), lo + oy * (1 - 2 * lo)

    def sample(self, uv_studs: np.ndarray) -> np.ndarray:
        from scipy.special import ndtr

        p = uv_studs / self.tile
        s = np.stack([p[:, 0] - p[:, 1] * 0.57735027, p[:, 1] * 1.15470054], axis=1)
        base = np.floor(s).astype(np.int64)
        fr = s - base
        upper = (fr[:, 0] + fr[:, 1]) >= 1.0
        v1 = np.where(upper[:, None], base + 1, base)
        v2 = base + np.array([1, 0])
        v3 = base + np.array([0, 1])
        w1 = np.where(upper, fr[:, 0] + fr[:, 1] - 1.0, 1.0 - fr[:, 0] - fr[:, 1])
        w2 = np.where(upper, 1.0 - fr[:, 1], fr[:, 0])
        w3 = np.where(upper, 1.0 - fr[:, 0], fr[:, 1])
        acc = np.zeros((len(p), 3))
        wsq = np.zeros(len(p))
        for vert, wt in ((v1, w1), (v2, w2), (v3, w3)):
            pv = np.stack([vert[:, 0] + vert[:, 1] * 0.5, vert[:, 1] * 0.8660254], axis=1)
            local = (p - pv) * self.SPAN
            ox, oy = self._offset(vert[:, 0], vert[:, 1])
            sx = (ox + local[:, 0]) * (self.w - 1)
            sy = (oy + local[:, 1]) * (self.h - 1)
            g = render.sample_image_xy(self.gauss, np.stack([sx, sy], axis=1))
            acc += g * wt[:, None]
            wsq += wt * wt
        gblend = acc / np.sqrt(np.maximum(wsq, 1e-9))[:, None]
        q = ndtr(gblend)
        out = np.zeros_like(gblend)
        for c in range(3):
            lut = self.lut[c]
            idx = np.clip((q[:, c] * (len(lut) - 1)).astype(np.int64), 0, len(lut) - 1)
            out[:, c] = lut[idx]
        return np.clip(out @ self.basis.T + self.mean, 0, 1)


def load_exemplar(path: str, base_color: Sequence[float]) -> Optional[np.ndarray]:
    """Load an exemplar patch, flatten its illumination and recentre its colour."""
    try:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    except Exception:
        return None
    if min(img.shape[:2]) < 24:
        return None
    lin = srgb_to_linear(img)
    import cv2

    k = max(3, (min(lin.shape[:2]) // 3) | 1)
    blur = cv2.GaussianBlur(lin.astype(np.float32), (k, k), 0).astype(np.float64)
    flat = lin / np.maximum(blur, 1e-3)
    target = srgb_to_linear(np.asarray(base_color))
    out = np.clip(flat * target[None, None, :], 0, 1)
    return out


# ---------------------------------------------------------------------------
# Photo projection
# ---------------------------------------------------------------------------

@dataclass
class PhotoProjection:
    """A source photograph and the camera that maps world points into it."""

    image_lin: np.ndarray
    camera: render.Camera
    mask: Optional[np.ndarray] = None
    mirror_axis: Optional[int] = 2
    mirror_center: float = 0.0
    weight_floor: float = 0.12
    extra: List["PhotoProjection"] = field(default_factory=list)
    pre: Optional[np.ndarray] = None


def project_photo(texels: Texels, sel: np.ndarray, proj: PhotoProjection, all_verts: np.ndarray, all_faces: np.ndarray):
    """Return (colour (n,3) linear, confidence (n,)) for the selected texels.

    Each view contributes where the texel is visible (depth test against a depth
    map rendered from that camera) and faces the camera. Hidden texels fall back
    to the mirror image across the symmetry plane. Contributions are blended with
    the fourth power of their weight so the most frontal view dominates.
    """
    pos = texels.pos[sel]
    nrm = texels.nrm[sel]
    num = np.zeros((len(pos), 3))
    den = np.zeros(len(pos))
    conf = np.zeros(len(pos))
    diam = float(np.linalg.norm(all_verts.max(0) - all_verts.min(0)))
    for view in [proj] + list(proj.extra):
        cam = view.camera
        if view.pre is not None:
            mv = all_verts @ view.pre[:3, :3].T + view.pre[:3, 3]
            vpos = pos @ view.pre[:3, :3].T + view.pre[:3, 3]
            vnrm = nrm @ np.linalg.inv(view.pre[:3, :3])
            vnrm /= np.maximum(np.linalg.norm(vnrm, axis=1, keepdims=True), 1e-9)
        else:
            mv, vpos, vnrm = all_verts, pos, nrm
        depth = render.depth_map(cam, mv, all_faces)
        h, w = depth.shape
        ih, iw = view.image_lin.shape[:2]
        for mirrored in (False, True):
            if mirrored and view.mirror_axis is None:
                continue
            p = vpos.copy()
            n = vnrm.copy()
            if mirrored:
                ax = view.mirror_axis
                p[:, ax] = 2 * view.mirror_center - p[:, ax]
                n[:, ax] = -n[:, ax]
            xy, d, _ = cam.project(p)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            xi = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, w - 1)
            yi = np.clip(np.floor(xy[:, 1]).astype(np.int64), 0, h - 1)
            vis = inside & (d <= depth[yi, xi] + 0.012 * diam)
            facing = np.sum(n * cam.view_dirs(p), axis=1)
            wgt = _smoothstep(view.weight_floor, 0.55, facing) * vis
            if view.mask is not None:
                mh, mw = view.mask.shape[:2]
                mx = np.clip((xy[:, 0] * mw / w).astype(np.int64), 0, mw - 1)
                my = np.clip((xy[:, 1] * mh / h).astype(np.int64), 0, mh - 1)
                wgt = wgt * (view.mask[my, mx] > 0)
            if mirrored:
                wgt = wgt * 0.8
            if not np.any(wgt > 0):
                continue
            col = render.sample_image_xy(view.image_lin, np.stack([xy[:, 0] * iw / w, xy[:, 1] * ih / h], axis=1))
            w4 = wgt ** 4
            num += col * w4[:, None]
            den += w4
            conf = np.maximum(conf, wgt)
    color = num / np.maximum(den, 1e-12)[:, None]
    return color, conf


# ---------------------------------------------------------------------------
# Ambient occlusion and normal baking
# ---------------------------------------------------------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=1)


def bake_ao(points: np.ndarray, normals: np.ndarray, vertices: np.ndarray, faces: np.ndarray, directions: int = 64, res: int = 384, ground_y: Optional[float] = None) -> np.ndarray:
    lo, hi = vertices.min(0), vertices.max(0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.linalg.norm(hi - lo)) * 0.5, 1e-3)
    acc = np.zeros(len(points))
    wsum = np.zeros(len(points))
    bias = radius * 0.012
    offset = points + normals * bias * 0.5
    for d in fibonacci_sphere(directions):
        w = np.clip(np.sum(normals * d, axis=1), 0.0, 1.0)
        active = w > 0
        if not active.any():
            continue
        eye = center + d * radius * 3.0
        cam = render.Camera(render.look_at(eye, center, (0, 1, 0) if abs(d[1]) < 0.99 else (0, 0, 1)), res, res, ortho_scale=res / (2.05 * radius))
        depth = render.depth_map(cam, vertices, faces)
        xy, z, _ = cam.project(offset[active])
        xi = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, res - 1)
        yi = np.clip(np.floor(xy[:, 1]).astype(np.int64), 0, res - 1)
        visible = z <= depth[yi, xi] + bias
        if ground_y is not None and d[1] < -0.05:
            visible &= points[active, 1] < ground_y + bias
        acc[active] += visible * w[active]
        wsum[active] += w[active]
    return acc / np.maximum(wsum, 1e-9)


def bake_detail_normals(texels: Texels, sel: np.ndarray, high: Mesh, samples: int = 400_000) -> np.ndarray:
    """Transfer normals of a high-detail mesh into tangent space of the low mesh."""
    area = high.face_areas()
    prob = area / max(area.sum(), 1e-12)
    rng = np.random.default_rng(5)
    count = int(min(samples, max(20_000, len(high.faces) * 30)))
    fi = rng.choice(len(high.faces), size=count, p=prob)
    r1, r2 = rng.random(count), rng.random(count)
    s1 = np.sqrt(r1)
    bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], axis=1)
    tri = high.vertices[high.faces[fi]]
    pts = np.einsum("nk,nkc->nc", bary, tri)
    ntri = high.normals[high.faces[fi]]
    nrm = np.einsum("nk,nkc->nc", bary, ntri)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    tree = cKDTree(pts)
    _, idx = tree.query(texels.pos[sel], k=4)
    nh = nrm[idx].mean(axis=1)
    n_low = texels.nrm[sel]
    nh = np.where(np.sum(nh * n_low, axis=1, keepdims=True) < 0.2, n_low, nh)
    nh /= np.maximum(np.linalg.norm(nh, axis=1, keepdims=True), 1e-9)
    t, b = texels.tangent[sel], texels.bitangent[sel]
    ts = np.stack([np.sum(nh * t, 1), np.sum(nh * b, 1), np.sum(nh * n_low, 1)], axis=1)
    return ts / np.maximum(np.linalg.norm(ts, axis=1, keepdims=True), 1e-9)


def height_to_normal(height_img: np.ndarray, mask: np.ndarray, density: float, strength: float = 1.0) -> np.ndarray:
    import cv2

    h = push_pull_fill(height_img[..., None], mask)[..., 0].astype(np.float32)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    su = gx * density * strength
    sv = -gy * density * strength
    n = np.stack([-su, -sv, np.ones_like(su)], axis=2)
    return n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-9)


def push_pull_fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill unmasked pixels from a mask-weighted image pyramid (seamless gutters)."""
    img = img.astype(np.float64)
    m = mask.astype(np.float64)
    levels = []
    cur_i, cur_m = img * m[..., None], m
    while min(cur_m.shape) > 1:
        levels.append((cur_i, cur_m))
        h, w = cur_m.shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        pi = np.zeros((h2 * 2, w2 * 2, img.shape[2]))
        pm = np.zeros((h2 * 2, w2 * 2))
        pi[:h, :w] = cur_i
        pm[:h, :w] = cur_m
        si = pi.reshape(h2, 2, w2, 2, -1).sum(axis=(1, 3))
        sm = pm.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        cur_i = np.where(sm[..., None] > 0, si / np.maximum(sm[..., None], 1e-9), 0)
        cur_m = (sm > 0).astype(np.float64)
    filled = cur_i
    for li, lm in reversed(levels):
        h, w = lm.shape
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)[:h, :w]
        mean_i = np.where(lm[..., None] > 0, li / np.maximum(lm[..., None], 1e-9), 0)
        filled = np.where(lm[..., None] > 0, mean_i, up)
    out = img.copy()
    out[~mask] = filled[~mask]
    return out


# ---------------------------------------------------------------------------
# Main bake
# ---------------------------------------------------------------------------

@dataclass
class BakeOptions:
    resolution: int = 1024
    padding: int = 6
    ambient_occlusion: bool = True
    ao_in_color: float = 0.55
    ao_directions: int = 64
    ground_occlusion: bool = True
    detail_normals: bool = True
    texture_style: str = "auto"


@dataclass
class TextureSet:
    maps: Dict[str, np.ndarray]
    resolution: int
    density: float
    utilization: float
    island_count: int
    notes: List[str] = field(default_factory=list)

    def save(self, folder: str) -> Dict[str, str]:
        import os

        os.makedirs(folder, exist_ok=True)
        paths = {}
        for name, arr in self.maps.items():
            path = os.path.join(folder, f"{name}.png")
            mode = "L" if arr.ndim == 2 else ("RGBA" if arr.shape[2] == 4 else "RGB")
            Image.fromarray(arr, mode).save(path, optimize=True)
            paths[name] = path
        return paths


def _to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def bake_scene(scene: Scene, options: BakeOptions, photo: Optional[PhotoProjection] = None, progress: Progress = None) -> TextureSet:
    """Pack UVs, synthesise all maps and write atlas UVs into the scene meshes."""
    def report(stage, frac):
        if progress:
            progress(stage, frac)

    meshes = [p.mesh for p in scene.parts]
    res = int(options.resolution)
    atlas = pack_atlas(meshes, res, options.padding)
    apply_atlas_uvs(meshes, atlas)
    report("uv atlas packed", 0.1)
    tx = texel_attributes(meshes, atlas)
    n = len(tx.rows)
    albedo = np.zeros((n, 3))
    height = np.zeros(n)
    rough = np.zeros(n)
    metal = np.zeros(n)
    emis = np.zeros(n)
    alpha = np.ones(n)
    notes: List[str] = []
    all_v = np.concatenate([m.vertices for m in meshes])
    offs = np.cumsum([0] + [len(m.vertices) for m in meshes])[:-1]
    all_f = np.concatenate([m.faces + o for m, o in zip(meshes, offs)])
    exemplar_cache: Dict[str, Optional[Exemplar]] = {}
    for pi, part in enumerate(scene.parts):
        sel = np.flatnonzero(tx.part == pi)
        if len(sel) == 0:
            continue
        mat = scene.materials.get(part.material)
        if mat is None:
            continue
        inv = np.linalg.inv(part.frame)
        local = tx.pos[sel] @ inv[:3, :3].T + inv[:3, 3]
        uv = tx.phys_uv[sel]
        if part.uv_grain == "v":
            uv = uv[:, ::-1]
        ch = eval_material(mat, local, uv, part.grain_axis, part.seed)
        use_exemplar = options.texture_style in ("auto", "photo") and mat.exemplar and mat.kind not in ("emissive", "glass")
        if use_exemplar:
            if mat.key not in exemplar_cache:
                ex_img = load_exemplar(mat.exemplar, mat.base_color)
                exemplar_cache[mat.key] = Exemplar(ex_img, px_per_stud=mat.exemplar_density, seed=part.seed) if ex_img is not None else None
            ex = exemplar_cache[mat.key]
            if ex is not None:
                syn = ex.sample(uv)
                mix = 0.8 if options.texture_style == "photo" else 0.6
                ch.albedo = ch.albedo * (1 - mix) + syn * mix
        if part.photo and photo is not None:
            col, wgt = project_photo(tx, sel, photo, all_v, all_f)
            fallback = ch.albedo
            if np.any(wgt > 0.3):
                seen = col[wgt > 0.3]
                med = np.median(seen, axis=0)
                fallback = ch.albedo / np.maximum(srgb_to_linear(np.asarray(mat.base_color)), 1e-3) * med
            ch.albedo = col * wgt[:, None] + fallback * (1 - wgt[:, None])
        albedo[sel] = ch.albedo
        height[sel] = ch.height
        rough[sel] = ch.roughness
        metal[sel] = ch.metallic
        emis[sel] = ch.emissive
        alpha[sel] = ch.alpha
    report("materials synthesised", 0.45)

    ao = np.ones(n)
    if options.ambient_occlusion and n:
        step = 2 if res >= 1024 else 1
        sub = np.flatnonzero(((tx.rows % step) == 0) & ((tx.cols % step) == 0))
        ground = float(all_v[:, 1].min()) if options.ground_occlusion else None
        ao_sub = bake_ao(tx.pos[sub], tx.nrm[sub], all_v, all_f, options.ao_directions, 384, ground)
        if step > 1:
            tree = cKDTree(np.stack([tx.rows[sub], tx.cols[sub]], 1))
            _, idx = tree.query(np.stack([tx.rows, tx.cols], 1), k=1)
            same_part = tx.part[sub][idx] == tx.part
            ao = np.where(same_part, ao_sub[idx], 1.0)
        else:
            ao = ao_sub
        ao = np.clip(ao * 1.08, 0, 1)
    report("ambient occlusion baked", 0.75)

    normal_ts = np.zeros((n, 3))
    normal_ts[:, 2] = 1.0
    if options.detail_normals:
        for pi, part in enumerate(scene.parts):
            if part.detail is None:
                continue
            sel = np.flatnonzero(tx.part == pi)
            if len(sel):
                normal_ts[sel] = bake_detail_normals(tx, sel, part.detail)
    report("detail normals baked", 0.85)

    def img(values, channels):
        out = np.zeros((res, res, channels))
        out[tx.rows, tx.cols] = values.reshape(n, channels)
        return out

    mask = tx.mask
    color_lin = albedo * (1.0 - options.ao_in_color * (1.0 - ao))[:, None] if options.ambient_occlusion else albedo
    base_img = push_pull_fill(img(linear_to_srgb(color_lin), 3), mask)
    maps: Dict[str, np.ndarray] = {}
    if np.any(alpha < 0.999):
        a_img = push_pull_fill(img(alpha, 1), mask)
        maps["basecolor"] = _to_u8(np.concatenate([base_img, a_img], axis=2))
        maps["opacity"] = _to_u8(a_img[..., 0])
    else:
        maps["basecolor"] = _to_u8(base_img)
    h_img = img(height, 1)[..., 0]
    n_detail = height_to_normal(h_img, mask, atlas.density, 1.0)
    n_shape = push_pull_fill(img(normal_ts, 3), mask)
    n_shape /= np.maximum(np.linalg.norm(n_shape, axis=2, keepdims=True), 1e-9)
    nb = np.concatenate([n_shape[..., :2] + n_detail[..., :2], (n_shape[..., 2:3] * n_detail[..., 2:3])], axis=2)
    nb /= np.maximum(np.linalg.norm(nb, axis=2, keepdims=True), 1e-9)
    if float(np.abs(nb[mask][:, :2]).max(initial=0.0)) > 0.01:
        maps["normal"] = _to_u8(nb * 0.5 + 0.5)
    else:
        notes.append("Normal map skipped: the surface has no relief detail at this resolution.")
    maps["roughness"] = _to_u8(push_pull_fill(img(rough, 1), mask)[..., 0])
    if np.any(metal > 0.01):
        maps["metallic"] = _to_u8(push_pull_fill(img(metal, 1), mask)[..., 0])
    if options.ambient_occlusion:
        if float(ao.std()) > 0.01:
            maps["ambient_occlusion"] = _to_u8(push_pull_fill(img(ao, 1), mask)[..., 0])
        else:
            notes.append("Ambient occlusion map skipped: occlusion is uniform for this shape.")
    if np.any(emis > 0.01):
        maps["emissive"] = _to_u8(push_pull_fill(img(emis, 1), mask)[..., 0])
    report("maps encoded", 1.0)
    return TextureSet(maps, res, atlas.density, atlas.utilization, len(atlas.islands), notes)
