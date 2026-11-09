"""Software rasterizer used for previews, texture baking and model fitting.

The rasterizer is fully vectorised with NumPy: triangles are bucketed by the
size of their screen bounding box and every bucket is evaluated as one batch of
candidate pixels; a single lexicographic sort then resolves the depth test.
It produces per-pixel triangle ids and barycentric coordinates, from which any
vertex attribute can be interpolated (perspective-correct when ``inv_w`` is
given).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

EPS = 1e-12
_BUCKETS = (2, 4, 8, 16, 32, 64)


@dataclass
class Raster:
    face_id: np.ndarray
    bary: np.ndarray
    depth: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return self.face_id >= 0


def rasterize(
    xy: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    inv_w: Optional[np.ndarray] = None,
    cull_back: bool = False,
) -> Raster:
    """Rasterize triangles given in pixel coordinates.

    ``xy`` (V,2) pixel coordinates (x right, y down), ``z`` (V,) depth where smaller
    is closer. Pixel centres sit at integer + 0.5.
    """
    faces = np.asarray(faces, dtype=np.int64)
    face_id = np.full((height, width), -1, dtype=np.int64)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    if len(faces) == 0:
        return Raster(face_id, bary, depth)
    tri = xy[faces]
    area2 = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    valid = np.abs(area2) > 1e-12
    if cull_back:
        valid &= area2 < 0
    finite = np.isfinite(tri).all(axis=(1, 2)) & np.isfinite(z[faces]).all(axis=1)
    valid &= finite
    lo = np.floor(tri.min(axis=1) - 0.5).astype(np.int64)
    hi = np.ceil(tri.max(axis=1) - 0.5).astype(np.int64)
    lo[:, 0] = np.clip(lo[:, 0], 0, width - 1)
    lo[:, 1] = np.clip(lo[:, 1], 0, height - 1)
    hi[:, 0] = np.clip(hi[:, 0], 0, width - 1)
    hi[:, 1] = np.clip(hi[:, 1], 0, height - 1)
    valid &= (hi[:, 0] >= lo[:, 0]) & (hi[:, 1] >= lo[:, 1])
    span = np.maximum(hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1]) + 1
    idx_all = np.flatnonzero(valid)
    pix_list, z_list, f_list, b_list = [], [], [], []
    prev = 0
    for bsize in _BUCKETS:
        sel = idx_all[(span[idx_all] > prev) & (span[idx_all] <= bsize)]
        prev = bsize
        for chunk in np.array_split(sel, max(1, int(np.ceil(len(sel) * bsize * bsize / 4_000_000)))):
            if len(chunk):
                _raster_batch(chunk, bsize, xy, z, faces, lo, hi, width, inv_w, pix_list, z_list, f_list, b_list)
    big = idx_all[span[idx_all] > _BUCKETS[-1]]
    for fi in big:
        _raster_large(int(fi), xy, z, faces, lo, hi, width, inv_w, pix_list, z_list, f_list, b_list)
    if not pix_list:
        return Raster(face_id, bary, depth)
    pix = np.concatenate(pix_list)
    zz = np.concatenate(z_list)
    ff = np.concatenate(f_list)
    bb = np.concatenate(b_list)
    order = np.lexsort((zz, pix))
    pix_s = pix[order]
    first = np.ones(len(pix_s), dtype=bool)
    first[1:] = pix_s[1:] != pix_s[:-1]
    win = order[first]
    flat_f = face_id.reshape(-1)
    flat_f[pix[win]] = ff[win]
    bary.reshape(-1, 3)[pix[win]] = bb[win]
    depth.reshape(-1)[pix[win]] = zz[win]
    return Raster(face_id, bary, depth)


def _bary(px, py, tri):
    ax, ay = tri[..., 0, 0], tri[..., 0, 1]
    bx, by = tri[..., 1, 0], tri[..., 1, 1]
    cx, cy = tri[..., 2, 0], tri[..., 2, 1]
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    den = np.where(np.abs(den) < EPS, EPS, den)
    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
    return w0, w1, 1.0 - w0 - w1


def _emit(fids, px, py, w0, w1, w2, xy_faces_z, inv_w_f, width, pix_list, z_list, f_list, b_list):
    tol = -1e-6
    inside = (w0 >= tol) & (w1 >= tol) & (w2 >= tol)
    if not inside.any():
        return
    w0, w1, w2 = w0[inside], w1[inside], w2[inside]
    zf = xy_faces_z[inside]
    zz = w0 * zf[:, 0] + w1 * zf[:, 1] + w2 * zf[:, 2]
    if inv_w_f is not None:
        iw = inv_w_f[inside]
        p0, p1, p2 = w0 * iw[:, 0], w1 * iw[:, 1], w2 * iw[:, 2]
        s = p0 + p1 + p2
        s = np.where(np.abs(s) < EPS, EPS, s)
        w0, w1, w2 = p0 / s, p1 / s, p2 / s
    pix_list.append((py[inside] * width + px[inside]).astype(np.int64))
    z_list.append(zz)
    f_list.append(fids[inside])
    b_list.append(np.stack([w0, w1, w2], axis=1).astype(np.float32))


def _raster_batch(chunk, bsize, xy, z, faces, lo, hi, width, inv_w, pix_list, z_list, f_list, b_list):
    oy, ox = np.meshgrid(np.arange(bsize), np.arange(bsize), indexing="ij")
    ox = ox.reshape(-1)
    oy = oy.reshape(-1)
    px = lo[chunk, 0][:, None] + ox[None, :]
    py = lo[chunk, 1][:, None] + oy[None, :]
    keep = (px <= hi[chunk, 0][:, None]) & (py <= hi[chunk, 1][:, None])
    tri = xy[faces[chunk]]
    w0, w1, w2 = _bary(px + 0.5, py + 0.5, tri[:, None, :, :])
    w0 = np.where(keep, w0, -1.0)
    fids = np.broadcast_to(chunk[:, None], px.shape)
    zf = np.broadcast_to(z[faces[chunk]][:, None, :], px.shape + (3,))
    iwf = None if inv_w is None else np.broadcast_to(inv_w[faces[chunk]][:, None, :], px.shape + (3,)).reshape(-1, 3)
    _emit(fids.reshape(-1), px.reshape(-1), py.reshape(-1), w0.reshape(-1), w1.reshape(-1), w2.reshape(-1), zf.reshape(-1, 3), iwf, width, pix_list, z_list, f_list, b_list)


def _raster_large(fi, xy, z, faces, lo, hi, width, inv_w, pix_list, z_list, f_list, b_list):
    xs = np.arange(lo[fi, 0], hi[fi, 0] + 1)
    ys = np.arange(lo[fi, 1], hi[fi, 1] + 1)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    px = gx.reshape(-1)
    py = gy.reshape(-1)
    tri = xy[faces[fi]][None, :, :]
    w0, w1, w2 = _bary(px + 0.5, py + 0.5, tri)
    n = len(px)
    zf = np.broadcast_to(z[faces[fi]], (n, 3))
    iwf = None if inv_w is None else np.broadcast_to(inv_w[faces[fi]], (n, 3))
    _emit(np.full(n, fi, dtype=np.int64), px, py, w0, w1, w2, zf, iwf, width, pix_list, z_list, f_list, b_list)


def interpolate(raster: Raster, faces: np.ndarray, attr: np.ndarray) -> np.ndarray:
    """Interpolate a per-vertex attribute (V,C) across covered pixels -> (H,W,C)."""
    h, w = raster.face_id.shape
    out = np.zeros((h, w, attr.shape[1]), dtype=np.float64)
    m = raster.mask
    if not m.any():
        return out
    fid = raster.face_id[m]
    b = raster.bary[m].astype(np.float64)
    tri_attr = attr[faces[fid]]
    out[m] = np.einsum("nk,nkc->nc", b, tri_attr)
    return out


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """Pinhole or orthographic camera. View space: +X right, +Y up, looking -Z."""

    view: np.ndarray
    width: int
    height: int
    focal: float = 0.0
    ortho_scale: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    @property
    def is_ortho(self) -> bool:
        return self.ortho_scale > 0

    @property
    def position(self) -> np.ndarray:
        inv = np.linalg.inv(self.view)
        return inv[:3, 3]

    def project(self, points: np.ndarray):
        """Return (pixel_xy, depth, inv_w) for world points."""
        p = points @ self.view[:3, :3].T + self.view[:3, 3]
        depth = -p[:, 2]
        cx = self.cx or self.width / 2.0
        cy = self.cy or self.height / 2.0
        if self.is_ortho:
            s = self.ortho_scale
            x = cx + p[:, 0] * s
            y = cy - p[:, 1] * s
            return np.stack([x, y], axis=1), depth, None
        d = np.where(depth > 1e-6, depth, 1e-6)
        x = cx + self.focal * p[:, 0] / d
        y = cy - self.focal * p[:, 1] / d
        return np.stack([x, y], axis=1), depth, 1.0 / d

    def view_dirs(self, points: np.ndarray) -> np.ndarray:
        """Unit vectors from points toward the camera."""
        if self.is_ortho:
            inv = np.linalg.inv(self.view)
            d = inv[:3, 2]
            return np.broadcast_to(d / np.linalg.norm(d), points.shape).copy()
        v = self.position[None, :] - points
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-12)
    up = np.asarray(up, dtype=np.float64)
    if abs(np.dot(f, up / np.linalg.norm(up))) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r /= max(np.linalg.norm(r), 1e-12)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def orbit_camera(target, radius, azimuth_deg, elevation_deg, width, height, fov_deg=35.0, ortho_extent: float = 0.0) -> Camera:
    """Camera orbiting ``target``. Azimuth 0 looks at the model's front (-Z side)."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    target = np.asarray(target, dtype=np.float64)
    offset = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)]) * radius
    view = look_at(target + offset, target)
    if ortho_extent > 0:
        return Camera(view, width, height, ortho_scale=min(width, height) / ortho_extent)
    focal = 0.5 * height / np.tan(np.radians(fov_deg) / 2.0)
    return Camera(view, width, height, focal=focal)


def depth_map(camera: Camera, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    xy, z, iw = camera.project(vertices)
    return rasterize(xy, z, faces, camera.width, camera.height, iw).depth


# ---------------------------------------------------------------------------
# Texture sampling
# ---------------------------------------------------------------------------

def sample_bilinear(image: np.ndarray, uv: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Sample (H,W,C) float image at UV (N,2) with V pointing up (OpenGL style)."""
    h, w = image.shape[:2]
    img = image if image.ndim == 3 else image[..., None]
    x = uv[:, 0] * w - 0.5
    y = (1.0 - uv[:, 1]) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    if wrap:
        xi0, xi1 = x0 % w, (x0 + 1) % w
        yi0, yi1 = y0 % h, (y0 + 1) % h
    else:
        xi0, xi1 = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
        yi0, yi1 = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    c00 = img[yi0, xi0]
    c10 = img[yi0, xi1]
    c01 = img[yi1, xi0]
    c11 = img[yi1, xi1]
    out = (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy
    return out if image.ndim == 3 else out[:, 0]


def sample_image_xy(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H,W,C) image at pixel coordinates (x right, y down)."""
    h, w = image.shape[:2]
    uv = np.stack([(xy[:, 0]) / w, 1.0 - (xy[:, 1]) / h], axis=1)
    return sample_bilinear(image, uv)


# ---------------------------------------------------------------------------
# Tangent frames
# ---------------------------------------------------------------------------

def vertex_tangents(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (xyz + handedness w), Lengyel's method.

    For the axis-aligned UV islands produced by this engine the result matches
    MikkTSpace, which is what Roblox and Blender use for tangent-space maps.
    """
    tri = vertices[faces]
    tuv = uvs[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    d1 = tuv[:, 1] - tuv[:, 0]
    d2 = tuv[:, 2] - tuv[:, 0]
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-14, 1e-14, r)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    tan1 = np.zeros_like(vertices)
    tan2 = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(tan1, faces[:, k], sdir)
        np.add.at(tan2, faces[:, k], tdir)
    n = normals
    t = tan1 - n * np.sum(n * tan1, axis=1, keepdims=True)
    tl = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.cross(n, np.array([0.0, 1.0, 0.0]))
    fallback = np.where(np.linalg.norm(fallback, axis=1, keepdims=True) < 1e-6, np.cross(n, np.array([1.0, 0.0, 0.0])), fallback)
    t = np.where(tl < 1e-12, fallback, t)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    w = np.where(np.sum(np.cross(n, t) * tan2, axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], axis=1)


# ---------------------------------------------------------------------------
# Shaded preview rendering
# ---------------------------------------------------------------------------

@dataclass
class Materials:
    basecolor: np.ndarray
    normal: Optional[np.ndarray] = None
    roughness: Optional[np.ndarray] = None
    metallic: Optional[np.ndarray] = None
    emissive: Optional[np.ndarray] = None


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def shade(
    camera: Camera,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uvs: Optional[np.ndarray],
    textures: Optional[Materials],
    flat_color: Sequence[float] = (0.75, 0.75, 0.75),
    light_dir: Sequence[float] = (-0.45, 0.8, -0.55),
    shadows: bool = True,
    background: Optional[np.ndarray] = None,
    floor_y: Optional[float] = None,
) -> np.ndarray:
    """Physically-inspired preview shading. Returns sRGB float image (H,W,3)."""
    h, w = camera.height, camera.width
    xy, z, iw = camera.project(vertices)
    ras = rasterize(xy, z, faces, w, h, iw)
    mask = ras.mask
    if background is None:
        background = _studio_background(w, h)
    out = background.copy()
    pos = interpolate(ras, faces, vertices)
    nrm = interpolate(ras, faces, normals)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-9)
    ldir = np.asarray(light_dir, dtype=np.float64)
    ldir /= np.linalg.norm(ldir)
    light_cam = None
    if shadows:
        light_cam = _light_camera(vertices, ldir)
        light_depth = depth_map(light_cam, vertices, faces)
    if mask.any():
        p = pos[mask]
        n = nrm[mask]
        if uvs is not None and textures is not None:
            uv = interpolate(ras, faces, uvs)[mask]
            albedo = _srgb_to_linear(sample_bilinear(textures.basecolor[..., :3], uv))
            rough = sample_bilinear(textures.roughness, uv) if textures.roughness is not None else np.full(len(p), 0.6)
            metal = sample_bilinear(textures.metallic, uv) if textures.metallic is not None else np.zeros(len(p))
            if textures.normal is not None:
                tang = vertex_tangents(vertices, faces, normals, uvs)
                t4 = interpolate(ras, faces, tang)[mask]
                tn = sample_bilinear(textures.normal, uv) * 2.0 - 1.0
                t = t4[:, :3]
                t = t - n * np.sum(n * t, axis=1, keepdims=True)
                t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
                bsign = np.where(t4[:, 3] < 0, -1.0, 1.0)[:, None]
                b = np.cross(n, t) * bsign
                n = t * tn[:, 0:1] + b * tn[:, 1:2] + n * tn[:, 2:3]
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            emis = sample_bilinear(textures.emissive, uv) if textures.emissive is not None else None
        else:
            albedo = np.tile(_srgb_to_linear(np.asarray(flat_color, dtype=np.float64)), (len(p), 1))
            rough = np.full(len(p), 0.6)
            metal = np.zeros(len(p))
            emis = None
        vdir = camera.view_dirs(p)
        facing = np.sum(n * vdir, axis=1) < 0
        n[facing] = -n[facing]
        ndl = np.clip(np.sum(n * ldir, axis=1), 0.0, 1.0)
        if shadows:
            # Normal offset and slope-scaled bias in shadow-map texels: a fixed
            # bias prints acne stripes on curved surfaces lit at grazing angles.
            texel = 1.0 / light_cam.ortho_scale
            slope = np.minimum(np.sqrt(np.maximum(1.0 - ndl * ndl, 0.0)) / np.maximum(ndl, 1e-3), 6.0)
            sh = _shadow_lookup(light_cam, light_depth, p + n * (1.5 * texel), texel * (1.0 + 2.0 * slope))
        else:
            sh = np.ones(len(p))
        fill_dir = np.array([0.6, 0.35, 0.7])
        fill_dir /= np.linalg.norm(fill_dir)
        ndf = np.clip(np.sum(n * fill_dir, axis=1), 0.0, 1.0)
        hemi = 0.5 + 0.5 * n[:, 1]
        hvec = ldir[None, :] + vdir
        hvec /= np.maximum(np.linalg.norm(hvec, axis=1, keepdims=True), 1e-9)
        ndh = np.clip(np.sum(n * hvec, axis=1), 0.0, 1.0)
        rough_c = np.clip(rough, 0.05, 1.0)
        shininess = 2.0 / (rough_c ** 4 + 1e-3)
        spec_norm = (shininess + 8.0) / (8.0 * np.pi)
        f0 = 0.04 * (1 - metal)[:, None] + albedo * metal[:, None]
        spec = (spec_norm * ndh ** shininess * ndl)[:, None] * f0 * 0.6
        diffuse = albedo * (1 - metal)[:, None]
        key = np.array([1.0, 0.96, 0.9]) * 2.2
        fill = np.array([0.55, 0.65, 0.85]) * 0.45
        sky = np.array([0.36, 0.40, 0.48])
        ground = np.array([0.20, 0.18, 0.16])
        ambient = sky[None, :] * hemi[:, None] + ground[None, :] * (1 - hemi)[:, None]
        lit = diffuse * (key[None, :] * (ndl * sh)[:, None] / np.pi * 1.6 + fill[None, :] * ndf[:, None] + ambient)
        lit += spec * key[None, :] * sh[:, None]
        rim = np.clip(1.0 - np.sum(n * vdir, axis=1), 0, 1) ** 3
        lit += albedo * rim[:, None] * 0.12
        if emis is not None:
            lit += albedo * emis[:, None] * 2.0
        color = _tonemap(lit)
        out[mask] = color
    if floor_y is not None and shadows:
        out = _floor_shadow(out, mask, camera, light_cam, light_depth, floor_y)
    return out


def _tonemap(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    mapped = x * (1.0 + x / 6.0) / (1.0 + x)
    return _linear_to_srgb(mapped)


def _studio_background(w: int, h: int) -> np.ndarray:
    """Dark slate backdrop with a soft vignette (house style of the previews)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    top = np.array([0.105, 0.125, 0.150])
    bottom = np.array([0.050, 0.058, 0.070])
    t = (yy / max(h - 1, 1))[..., None]
    img = top * (1 - t) + bottom * t
    cx, cy = w / 2, h * 0.55
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    glow = np.clip(1.0 - r * 1.6, 0, 1)[..., None] ** 2
    img = img + glow * np.array([0.05, 0.065, 0.075])
    return img


def _light_camera(vertices: np.ndarray, ldir: np.ndarray, res: int = 768) -> Camera:
    c = 0.5 * (vertices.min(0) + vertices.max(0))
    radius = max(np.linalg.norm(vertices.max(0) - vertices.min(0)) * 0.5, 1e-3)
    floor_pad = radius * 1.6
    eye = c + ldir * radius * 4.0
    view = look_at(eye, c)
    return Camera(view, res, res, ortho_scale=res / (2.0 * floor_pad))


def _shadow_lookup(light_cam: Camera, light_depth: np.ndarray, points: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """3x3 percentage-closer lookup; ``bias`` is the per-point depth tolerance."""
    xy, d, _ = light_cam.project(points)
    h, w = light_depth.shape
    if bias is None:
        bias = 0.03 * (1 + 0.02 * d)
    total = np.zeros(len(points))
    count = 0
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            xi = np.clip(np.floor(xy[:, 0] + dx).astype(np.int64), 0, w - 1)
            yi = np.clip(np.floor(xy[:, 1] + dy).astype(np.int64), 0, h - 1)
            occ = light_depth[yi, xi]
            total += (d <= occ + bias).astype(np.float64)
            count += 1
    return total / count


def _floor_shadow(img, mask, camera: Camera, light_cam, light_depth, floor_y: float) -> np.ndarray:
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    inv = np.linalg.inv(camera.view)
    cx, cy = camera.width / 2.0, camera.height / 2.0
    if camera.is_ortho:
        dirs = np.broadcast_to(-inv[:3, 2], (h, w, 3))
        local = np.stack([(xx + 0.5 - cx) / camera.ortho_scale, -(yy + 0.5 - cy) / camera.ortho_scale, np.zeros_like(xx)], axis=2)
        origins = local @ inv[:3, :3].T + inv[:3, 3]
    else:
        local = np.stack([(xx + 0.5 - cx) / camera.focal, -(yy + 0.5 - cy) / camera.focal, -np.ones_like(xx)], axis=2)
        dirs = local @ inv[:3, :3].T
        origins = np.broadcast_to(inv[:3, 3], (h, w, 3))
    dy = dirs[..., 1]
    t = np.where(np.abs(dy) > 1e-9, (floor_y - origins[..., 1]) / np.where(np.abs(dy) > 1e-9, dy, 1.0), -1.0)
    hit = (t > 0) & ~mask
    if not hit.any():
        return img
    pts = origins[hit] + dirs[hit] * t[hit][:, None]
    lit = _shadow_lookup(light_cam, light_depth, pts + np.array([0, 0.01, 0]))
    out = img.copy()
    shade_amt = (1.0 - lit) * 0.55
    out[hit] = out[hit] * (1.0 - shade_amt[:, None])
    return out


def wireframe_overlay(image: np.ndarray, camera: Camera, vertices: np.ndarray, faces: np.ndarray, color=(0.95, 0.72, 0.30), alpha: float = 0.85) -> np.ndarray:
    """Draw visible triangle edges (depth-tested) over a rendered image."""
    h, w = image.shape[:2]
    xy, z, iw = camera.project(vertices)
    depth = rasterize(xy, z, faces, w, h, iw).depth
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = xy[edges[:, 0]], xy[edges[:, 1]]
    za, zb = z[edges[:, 0]], z[edges[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    steps = np.clip(np.ceil(length * 1.5).astype(np.int64), 2, 2000)
    total = int(steps.sum())
    edge_idx = np.repeat(np.arange(len(edges)), steps)
    offsets = np.arange(total) - np.repeat(np.cumsum(steps) - steps, steps)
    t = offsets / np.maximum(np.repeat(steps, steps) - 1, 1)
    p = a[edge_idx] + (b[edge_idx] - a[edge_idx]) * t[:, None]
    zz = za[edge_idx] + (zb[edge_idx] - za[edge_idx]) * t
    xi = np.floor(p[:, 0]).astype(np.int64)
    yi = np.floor(p[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi, yi, zz = xi[inside], yi[inside], zz[inside]
    tol = 0.01 * (1.0 + np.abs(zz)) + 0.02
    visible = zz <= depth[yi, xi] + tol
    out = image.copy()
    col = np.asarray(color, dtype=np.float64)
    out[yi[visible], xi[visible]] = out[yi[visible], xi[visible]] * (1 - alpha) + col * alpha
    return out
