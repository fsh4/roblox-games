"""Mesh data structure and procedural primitive builders.

Coordinate system (identical to Roblox): right-handed, +Y up, the model's
front faces -Z, the model's right side is +X. Units are studs.

Every builder returns a closed, outward-wound triangle mesh with:
  * per-vertex normals (vertices are split where shading must be hard),
  * per-vertex UVs expressed in physical units (1 UV unit = 1 stud) so that
    the texture atlas can later assign a uniform texel density to all parts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

EPS = 1e-9


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    face_material: Optional[np.ndarray] = None
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64).reshape(-1, 3)
        if self.uvs is not None:
            self.uvs = np.asarray(self.uvs, dtype=np.float64).reshape(-1, 2)
        if self.face_material is not None:
            self.face_material = np.asarray(self.face_material, dtype=np.int64).reshape(-1)

    @property
    def triangle_count(self) -> int:
        return int(len(self.faces))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    def copy(self) -> "Mesh":
        return Mesh(
            self.vertices.copy(),
            self.faces.copy(),
            None if self.normals is None else self.normals.copy(),
            None if self.uvs is None else self.uvs.copy(),
            None if self.face_material is None else self.face_material.copy(),
            self.name,
        )

    def bounds(self) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.zeros((2, 3))
        return np.stack([self.vertices.min(0), self.vertices.max(0)])

    def extents(self) -> np.ndarray:
        b = self.bounds()
        return b[1] - b[0]

    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(length, EPS)

    def face_areas(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    def surface_area(self) -> float:
        return float(self.face_areas().sum())

    def signed_volume(self) -> float:
        tri = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        out = self.copy()
        out.apply_transform(matrix)
        return out

    def apply_transform(self, matrix: np.ndarray) -> None:
        m = np.asarray(matrix, dtype=np.float64)
        self.vertices = self.vertices @ m[:3, :3].T + m[:3, 3]
        if self.normals is not None:
            nm = np.linalg.inv(m[:3, :3]).T
            n = self.normals @ nm.T
            self.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), EPS)
        if np.linalg.det(m[:3, :3]) < 0:
            self.faces = self.faces[:, ::-1].copy()

    def translated(self, offset: Sequence[float]) -> "Mesh":
        out = self.copy()
        out.vertices = out.vertices + np.asarray(offset, dtype=np.float64)
        return out


def merge(meshes: Iterable[Mesh], name: str = "merged") -> Mesh:
    meshes = [m for m in meshes if m is not None and len(m.faces)]
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0, dtype=np.int64), name)
    verts, faces, norms, uvs, mats = [], [], [], [], []
    offset = 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + offset)
        norms.append(m.normals if m.normals is not None else vertex_normals(m))
        uvs.append(m.uvs if m.uvs is not None else np.zeros((len(m.vertices), 2)))
        mats.append(m.face_material if m.face_material is not None else np.zeros(len(m.faces), dtype=np.int64))
        offset += len(m.vertices)
    return Mesh(np.concatenate(verts), np.concatenate(faces), np.concatenate(norms), np.concatenate(uvs), np.concatenate(mats), name)


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Area-weighted vertex normals over the existing vertex indexing."""
    tri = mesh.vertices[mesh.faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(out, mesh.faces[:, k], fn)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), EPS)


def recompute_normals(mesh: Mesh, smooth_angle_deg: float = 35.0) -> Mesh:
    """Rebuild per-corner normals with hard edges above ``smooth_angle_deg``.

    Corners that share a position are averaged only with neighbouring faces whose
    normal is within the threshold (auto-smooth). Vertices are split as needed and
    existing UVs are preserved per corner.
    """
    fn = mesh.face_normals()
    fa = mesh.face_areas()
    corners_pos = mesh.vertices[mesh.faces].reshape(-1, 3)
    corner_face = np.repeat(np.arange(len(mesh.faces)), 3)
    key = np.round(corners_pos / 1e-6).astype(np.int64)
    _, pos_id = np.unique(key, axis=0, return_inverse=True)
    pos_id = pos_id.reshape(-1)
    order = np.argsort(pos_id, kind="stable")
    boundaries = np.flatnonzero(np.diff(pos_id[order])) + 1
    groups = np.split(order, boundaries)
    cos_t = np.cos(np.radians(smooth_angle_deg))
    corner_normal = np.zeros((len(corners_pos), 3))
    if smooth_angle_deg <= 0:
        corner_normal = fn[corner_face]
    else:
        for g in groups:
            f = corner_face[g]
            n = fn[f]
            w = fa[f][:, None]
            if len(g) == 1:
                corner_normal[g] = n
                continue
            sim = (n @ n.T) >= cos_t
            acc = sim.astype(np.float64) @ (n * w)
            corner_normal[g] = acc
        corner_normal /= np.maximum(np.linalg.norm(corner_normal, axis=1, keepdims=True), EPS)
    corner_uv = mesh.uvs[mesh.faces].reshape(-1, 2) if mesh.uvs is not None else np.zeros((len(corners_pos), 2))
    return _rebuild_from_corners(corners_pos, corner_normal, corner_uv, mesh.face_material, mesh.name)


def _rebuild_from_corners(pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray, face_material, name: str) -> Mesh:
    data = np.concatenate([
        np.round(pos / 1e-6),
        np.round(nrm / 1e-4),
        np.round(uv / 1e-6),
    ], axis=1).astype(np.int64)
    _, first, inverse = np.unique(data, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return Mesh(pos[first], inverse.reshape(-1, 3), nrm[first], uv[first], face_material, name)


def weld(mesh: Mesh) -> Mesh:
    """Merge vertices that are identical in position, normal and UV."""
    corners_pos = mesh.vertices[mesh.faces].reshape(-1, 3)
    corner_nrm = mesh.normals[mesh.faces].reshape(-1, 3) if mesh.normals is not None else np.zeros_like(corners_pos)
    corner_uv = mesh.uvs[mesh.faces].reshape(-1, 2) if mesh.uvs is not None else np.zeros((len(corners_pos), 2))
    return _rebuild_from_corners(corners_pos, corner_nrm, corner_uv, mesh.face_material, mesh.name)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def translation(offset: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = offset
    return m


def scaling(factors: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = factors
    return m


def rotation(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), EPS)
    x, y, z = a
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    r = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    return m


def compose(*matrices: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    for m in matrices:
        out = out @ m
    return out


def align_y_to(direction: Sequence[float]) -> np.ndarray:
    """Rotation that maps +Y onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(np.linalg.norm(d), EPS)
    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(y, d))
    if c > 1 - 1e-9:
        return np.eye(4)
    if c < -1 + 1e-9:
        return rotation([1, 0, 0], np.pi)
    axis = np.cross(y, d)
    return rotation(axis, np.arccos(np.clip(c, -1, 1)))


# ---------------------------------------------------------------------------
# Rounded / bevelled box
# ---------------------------------------------------------------------------

_FACE_DEFS = [
    (0, +1, 2, 1),
    (0, -1, 2, 1),
    (1, +1, 0, 2),
    (1, -1, 0, 2),
    (2, +1, 0, 1),
    (2, -1, 0, 1),
]


def _axis_samples(half: float, radius: float, segments: int, flat_divisions: int = 0) -> np.ndarray:
    if segments <= 0 or radius <= EPS:
        return np.linspace(-half, half, flat_divisions + 2)
    radius = min(radius, half * 0.999)
    band = np.linspace(0.0, radius, segments + 1)
    left = -half + band
    right = half - band[::-1]
    if radius >= half * 0.999:
        return np.concatenate([left[:-1], [0.0], right[1:]])
    middle = np.linspace(left[-1], right[0], flat_divisions + 2)[1:-1]
    return np.concatenate([left, middle, right])


def box(
    size: Sequence[float],
    center: Sequence[float] = (0.0, 0.0, 0.0),
    bevel: float = 0.0,
    bevel_segments: int = 0,
    grain_axis: Optional[int] = None,
    name: str = "box",
    flat_divisions: int = 0,
) -> Mesh:
    """Axis-aligned box with optional rounded edges.

    ``bevel`` is the edge radius in studs; ``bevel_segments`` the subdivisions per
    rounded band. With ``bevel_segments == 0`` the box is a 12-triangle cuboid with
    hard edges. ``grain_axis`` (0, 1, 2) orients the U texture axis along that world
    axis on every face that contains it, so wood grain follows the part's length.
    ``flat_divisions`` adds interior grid lines (needed before deforming the box).
    """
    size = np.maximum(np.asarray(size, dtype=np.float64), 1e-4)
    half = size / 2.0
    radius = float(min(bevel, half.min() * 0.95)) if bevel_segments > 0 else 0.0
    segs = bevel_segments if radius > EPS else 0
    samples = [_axis_samples(half[i], radius, segs, flat_divisions) for i in range(3)]
    inner = half - radius
    verts, norms, uvs, faces = [], [], [], []
    base = 0
    for axis, sign, ua, va in _FACE_DEFS:
        su, sv = samples[ua], samples[va]
        gu, gv = np.meshgrid(su, sv, indexing="ij")
        p = np.zeros(gu.shape + (3,))
        p[..., axis] = sign * half[axis]
        p[..., ua] = gu
        p[..., va] = gv
        p = p.reshape(-1, 3)
        if radius > EPS:
            core = np.clip(p, -inner, inner)
            d = p - core
            dl = np.linalg.norm(d, axis=1, keepdims=True)
            n = d / np.maximum(dl, EPS)
            p = core + n * radius
        else:
            n = np.zeros_like(p)
            n[:, axis] = sign
        flat = np.zeros_like(n)
        flat[:, axis] = sign
        n = np.where(np.linalg.norm(n, axis=1, keepdims=True) < 0.5, flat, n)
        u_axis, v_axis = ua, va
        if grain_axis is not None and grain_axis != axis and grain_axis == va:
            u_axis, v_axis = va, ua
        pu = gu.reshape(-1) if u_axis == ua else gv.reshape(-1)
        pv = gv.reshape(-1) if v_axis == va else gu.reshape(-1)
        uv = np.stack([pu + half[u_axis], pv + half[v_axis]], axis=1)
        nu, nv = len(su), len(sv)
        idx = np.arange(nu * nv).reshape(nu, nv)
        a = idx[:-1, :-1].reshape(-1)
        b = idx[1:, :-1].reshape(-1)
        c = idx[1:, 1:].reshape(-1)
        d_ = idx[:-1, 1:].reshape(-1)
        quad = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d_], 1)], 1).reshape(-1, 3)
        expected = np.zeros(3)
        expected[axis] = sign
        tri = p[quad]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if np.dot(cross.sum(0), expected) < 0:
            quad = quad[:, ::-1]
        verts.append(p)
        norms.append(n)
        uvs.append(uv)
        faces.append(quad + base)
        base += len(p)
    mesh = Mesh(np.concatenate(verts) + np.asarray(center, dtype=np.float64), np.concatenate(faces), np.concatenate(norms), np.concatenate(uvs), None, name)
    return _drop_degenerate(mesh)


def _drop_degenerate(mesh: Mesh) -> Mesh:
    areas = mesh.face_areas()
    keep = areas > 1e-12
    if keep.all():
        return mesh
    out = mesh.copy()
    out.faces = out.faces[keep]
    if out.face_material is not None:
        out.face_material = out.face_material[keep]
    return out


def box_triangles(bevel_segments: int) -> int:
    s = max(0, int(bevel_segments))
    return 12 * (2 * s + 1) ** 2


# ---------------------------------------------------------------------------
# Lathe (surface of revolution): cylinders, cones, spheres, vases, lampshades
# ---------------------------------------------------------------------------

def lathe(
    profile: Sequence[Sequence[float]],
    segments: int = 24,
    smooth_profile: bool = True,
    hard_angle_deg: float = 50.0,
    closed_profile: bool = False,
    name: str = "lathe",
    start_angle: float = 0.0,
) -> Mesh:
    """Revolve a 2D profile ``[(radius, y), ...]`` around the +Y axis.

    The profile runs bottom to top. Points with radius 0 become poles (fans).
    Profile corners sharper than ``hard_angle_deg`` get split normals so caps and
    rims stay crisp. UVs: U = arc length at the widest radius, V = profile length.
    """
    prof = np.asarray(profile, dtype=np.float64)
    if closed_profile and np.linalg.norm(prof[0] - prof[-1]) > 1e-9:
        prof = np.vstack([prof, prof[:1]])
    segments = max(3, int(segments))
    seg_dir = np.diff(prof, axis=0)
    seg_len = np.linalg.norm(seg_dir, axis=1)
    keep = np.concatenate([[True], seg_len > 1e-9])
    prof = prof[keep]
    seg_dir = np.diff(prof, axis=0)
    seg_len = np.linalg.norm(seg_dir, axis=1)
    tang = seg_dir / np.maximum(seg_len[:, None], EPS)
    seg_n2 = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    rmax = max(float(prof[:, 0].max()), 1e-3)
    v_coord = np.concatenate([[0.0], np.cumsum(seg_len)])
    # Duplicate profile points at hard corners: each segment gets its own pair of
    # endpoints when its neighbour bends more than the threshold.
    cos_h = np.cos(np.radians(hard_angle_deg))
    rings = []
    for i in range(len(prof)):
        prev_n = seg_n2[i - 1] if i > 0 else None
        next_n = seg_n2[i] if i < len(seg_n2) else None
        if closed_profile and i == 0:
            prev_n = seg_n2[-1]
        if prev_n is not None and next_n is not None:
            if smooth_profile and float(np.dot(prev_n, next_n)) >= cos_h:
                n = prev_n + next_n
                rings.append((i, n / max(np.linalg.norm(n), EPS), "both"))
            else:
                rings.append((i, prev_n, "prev"))
                rings.append((i, next_n, "next"))
        else:
            rings.append((i, prev_n if prev_n is not None else next_n, "both"))
    angles = start_angle + np.linspace(0.0, 2.0 * np.pi, segments + 1)
    ca, sa = np.cos(angles), np.sin(angles)
    verts, norms, uvs = [], [], []
    ring_index = []
    for (i, n2, role) in rings:
        r, y = prof[i]
        ring_index.append((i, role))
        x = r * ca
        z = -r * sa
        verts.append(np.stack([x, np.full_like(x, y), z], axis=1))
        nr, ny = n2
        norms.append(np.stack([nr * ca, np.full_like(ca, ny), -nr * sa], axis=1))
        uvs.append(np.stack([angles * rmax, np.full_like(angles, v_coord[i])], axis=1))
    faces = []
    width = segments + 1
    for s in range(len(prof) - 1):
        r0 = _ring_for(ring_index, s, "next")
        r1 = _ring_for(ring_index, s + 1, "prev")
        a0, a1 = prof[s][0], prof[s + 1][0]
        b0, b1 = r0 * width, r1 * width
        for k in range(segments):
            p00, p01 = b0 + k, b0 + k + 1
            p10, p11 = b1 + k, b1 + k + 1
            if a0 <= 1e-9 and a1 <= 1e-9:
                continue
            if a0 <= 1e-9:
                faces.append((p00, p11, p10))
            elif a1 <= 1e-9:
                faces.append((p00, p01, p10))
            else:
                faces.append((p00, p01, p11))
                faces.append((p00, p11, p10))
    mesh = Mesh(np.concatenate(verts), np.asarray(faces, dtype=np.int64), np.concatenate(norms), np.concatenate(uvs), None, name)
    mesh = _drop_degenerate(mesh)
    if mesh.signed_volume() < 0:
        mesh.faces = mesh.faces[:, ::-1].copy()
        mesh.normals = -mesh.normals
    return mesh


def _ring_for(ring_index, profile_i: int, want: str) -> int:
    candidates = [k for k, (i, role) in enumerate(ring_index) if i == profile_i]
    if len(candidates) == 1:
        return candidates[0]
    for k in candidates:
        if ring_index[k][1] == want:
            return k
    return candidates[0]


def _cap(y: float, radius: float, segments: int, up: bool, rmax: float) -> Mesh:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(angles), np.full(segments, y), -radius * np.sin(angles)], axis=1)
    verts = np.vstack([[0.0, y, 0.0], ring])
    n = np.zeros_like(verts)
    n[:, 1] = 1.0 if up else -1.0
    idx = np.arange(1, segments + 1)
    nxt = np.roll(idx, -1)
    faces = np.stack([np.zeros(segments, dtype=np.int64), idx, nxt], axis=1)
    if not up:
        faces = faces[:, ::-1]
    uv = np.stack([verts[:, 0] + rmax, verts[:, 2] + rmax], axis=1)
    return Mesh(verts, faces, n, uv, None, "cap")


def cylinder(
    radius: float,
    height: float,
    segments: int = 16,
    radius_top: Optional[float] = None,
    bevel: float = 0.0,
    bevel_segments: int = 0,
    name: str = "cylinder",
) -> Mesh:
    """Capped cylinder / frustum standing on y=0 with optional rounded rims."""
    rt = radius if radius_top is None else radius_top
    rb = radius
    if bevel_segments > 0 and bevel > EPS:
        b = min(bevel, 0.45 * min(rb, rt, height / 2))
        prof = [(0.0, 0.0)]
        for k in range(bevel_segments + 1):
            t = k / bevel_segments * (np.pi / 2)
            prof.append((rb - b + b * np.sin(t), b - b * np.cos(t)))
        for k in range(bevel_segments + 1):
            t = k / bevel_segments * (np.pi / 2)
            prof.append((rt - b + b * np.cos(t), height - b + b * np.sin(t)))
        prof.append((0.0, height))
        return lathe(prof, segments, smooth_profile=True, hard_angle_deg=60.0, name=name)
    side = lathe([(rb, 0.0), (rt, height)], segments, name=name)
    rmax = max(rb, rt)
    parts = [side]
    if rb > EPS:
        parts.append(_cap(0.0, rb, segments, up=False, rmax=rmax))
    if rt > EPS:
        parts.append(_cap(height, rt, segments, up=True, rmax=rmax))
    out = merge(parts, name)
    out.face_material = None
    return out


def cylinder_triangles(segments: int, bevel_segments: int = 0) -> int:
    if bevel_segments > 0:
        rings = 2 * (bevel_segments + 1)
        return segments * (2 * (rings - 1) + 2)
    return segments * 4


def sphere(radius: float, segments: int = 16, rings: int = 10, name: str = "sphere") -> Mesh:
    t = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    prof = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    prof[0, 0] = 0.0
    prof[-1, 0] = 0.0
    return lathe(prof, segments, smooth_profile=True, hard_angle_deg=89.0, name=name)


def torus(major: float, minor: float, segments: int = 24, tube_segments: int = 10, name: str = "torus") -> Mesh:
    t = np.linspace(0.0, 2 * np.pi, tube_segments + 1)
    prof = np.stack([major + minor * np.cos(t), minor * np.sin(t)], axis=1)
    return lathe(prof, segments, smooth_profile=True, hard_angle_deg=89.0, closed_profile=True, name=name)


# ---------------------------------------------------------------------------
# Extrusion of 2D polygons (with holes)
# ---------------------------------------------------------------------------

def _ensure_ccw(ring: np.ndarray, ccw: bool = True) -> np.ndarray:
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if (area > 0) != ccw:
        return ring[::-1].copy()
    return ring


def triangulate_polygon(outer: np.ndarray, holes: Sequence[np.ndarray] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Ear-cut triangulation. Returns (points (N,2), triangles (T,3)) CCW."""
    import mapbox_earcut as earcut

    outer = _ensure_ccw(np.asarray(outer, dtype=np.float64), True)
    rings = [outer] + [_ensure_ccw(np.asarray(h, dtype=np.float64), False) for h in holes]
    pts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tris = np.asarray(earcut.triangulate_float64(pts, ends), dtype=np.int64).reshape(-1, 3)
    if len(tris):
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flip = cross < 0
        tris[flip] = tris[flip][:, ::-1]
    return pts, tris


def _offset_ring(ring: np.ndarray, distance: float) -> np.ndarray:
    """Mitered inward offset for a CCW ring (positive distance shrinks)."""
    prev = np.roll(ring, 1, axis=0)
    nxt = np.roll(ring, -1, axis=0)
    e0 = ring - prev
    e1 = nxt - ring
    n0 = np.stack([-e0[:, 1], e0[:, 0]], 1)
    n1 = np.stack([-e1[:, 1], e1[:, 0]], 1)
    n0 /= np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), EPS)
    n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), EPS)
    bis = n0 + n1
    bl = np.linalg.norm(bis, axis=1, keepdims=True)
    bis = np.where(bl < 1e-6, n0, bis / np.maximum(bl, EPS))
    cosang = np.clip(np.sum(bis * n0, axis=1, keepdims=True), 0.35, 1.0)
    return ring + bis * (distance / cosang)


def _degenerate_vertices(ring: np.ndarray, tol: float) -> np.ndarray:
    """Vertices that duplicate their predecessor or lie on the line through their neighbours."""
    prev = np.roll(ring, 1, axis=0)
    nxt = np.roll(ring, -1, axis=0)
    cross = np.abs((ring[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1]) - (ring[:, 1] - prev[:, 1]) * (nxt[:, 0] - prev[:, 0]))
    dist = cross / np.maximum(np.linalg.norm(nxt - prev, axis=1), EPS)
    return (dist < tol) | (np.linalg.norm(ring - prev, axis=1) < tol)


def _clean_ring(ring: np.ndarray, tol: float, inset: float = 0.0) -> np.ndarray:
    """Drop vertices the ear-cutter would discard, in the ring and in its mitred inset.

    Ear clipping silently removes collinear and repeated points; the side wall
    would still use them, leaving T-junctions (open edges) along the caps.
    """
    r = np.asarray(ring, dtype=np.float64)
    for _ in range(len(r)):
        if len(r) <= 3:
            break
        bad = _degenerate_vertices(r, tol)
        if inset > EPS:
            bad |= _degenerate_vertices(_offset_ring(r, inset), tol)
        if not bad.any():
            break
        bad &= ~np.roll(bad, 1)
        r = r[~bad]
    return r


def _rings_cross(rings: Sequence[np.ndarray]) -> bool:
    """True if any two edges of the rings properly intersect (self or mutual)."""
    p = np.concatenate([r for r in rings])
    q = np.concatenate([np.roll(r, -1, axis=0) for r in rings])

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    for start in range(0, len(p), 512):
        pa, qa = p[start:start + 512, None], q[start:start + 512, None]
        o1 = orient(pa, qa, p[None])
        o2 = orient(pa, qa, q[None])
        o3 = orient(p[None], q[None], pa)
        o4 = orient(p[None], q[None], qa)
        if np.any((o1 * o2 < 0) & (o3 * o4 < 0)):
            return True
    return False


def extrude_polygon(
    outer: np.ndarray,
    holes: Sequence[np.ndarray] = (),
    depth: float = 1.0,
    bevel: float = 0.0,
    name: str = "extrusion",
) -> Mesh:
    """Extrude a polygon in the XY plane symmetrically along Z (front face at -Z).

    A small ``bevel`` produces a chamfer ring on both sides using a mitered inset,
    which keeps vertex correspondence so the wall stays watertight. Where the
    inset would fold over itself (thin spikes, short edges) the bevel is
    halved, and dropped after three tries.
    """
    outer = _ensure_ccw(np.asarray(outer, dtype=np.float64), True)
    holes = [_ensure_ccw(np.asarray(h, dtype=np.float64), False) for h in holes if len(h) >= 3]
    half = depth / 2.0
    tol = 1e-6 * max(float(np.ptp(outer, axis=0).max()), 1e-3)
    b = float(min(bevel, half * 0.8))
    for _ in range(4):
        rings = [_clean_ring(r, tol, b) for r in [outer] + holes]
        rings = [rings[0]] + [h for h in rings[1:] if len(h) >= 3]
        inset_rings = [_offset_ring(r, b) for r in rings] if b > EPS else rings
        if b <= EPS or not _rings_cross(inset_rings):
            break
        b *= 0.5
    else:
        b = 0.0
        rings = [_clean_ring(r, tol) for r in [outer] + holes]
        rings = [rings[0]] + [h for h in rings[1:] if len(h) >= 3]
        inset_rings = rings
    cap_pts, cap_tris = triangulate_polygon(inset_rings[0], inset_rings[1:])
    meshes = []
    for z, sign in ((-half, -1.0), (half, 1.0)):
        v = np.column_stack([cap_pts, np.full(len(cap_pts), z)])
        n = np.zeros_like(v)
        n[:, 2] = sign
        f = cap_tris if sign > 0 else cap_tris[:, ::-1]
        meshes.append(Mesh(v, f, n, cap_pts.copy(), None, name + "_cap"))
    for ring, inset in zip(rings, inset_rings):
        if b > EPS:
            layers = [(inset, -half), (ring, -half + b), (ring, half - b), (inset, half)]
        else:
            layers = [(ring, -half), (ring, half)]
        meshes.append(_wall(layers, name))
    out = merge(meshes, name)
    out.face_material = None
    if out.signed_volume() < 0:
        out.faces = out.faces[:, ::-1].copy()
        out.normals = -out.normals
    return out


def _wall(layers, name: str) -> Mesh:
    """Side wall between consecutive (ring, z) layers with hard vertical edges."""
    meshes = []
    for (ra, za), (rb, zb) in zip(layers[:-1], layers[1:]):
        n_pts = len(ra)
        ra_n = np.roll(ra, -1, axis=0)
        rb_n = np.roll(rb, -1, axis=0)
        quads_v = np.stack([
            np.column_stack([ra, np.full(n_pts, za)]),
            np.column_stack([ra_n, np.full(n_pts, za)]),
            np.column_stack([rb_n, np.full(n_pts, zb)]),
            np.column_stack([rb, np.full(n_pts, zb)]),
        ], axis=1)
        e1 = quads_v[:, 1] - quads_v[:, 0]
        e2 = quads_v[:, 3] - quads_v[:, 0]
        fn = np.cross(e1, e2)
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), EPS)
        verts = quads_v.reshape(-1, 3)
        norms = np.repeat(fn, 4, axis=0)
        edge_len = np.linalg.norm(ra_n - ra, axis=1)
        u0 = np.concatenate([[0.0], np.cumsum(edge_len)[:-1]])
        u1 = u0 + edge_len
        uv = np.stack([
            np.stack([u0, np.full(n_pts, za)], 1),
            np.stack([u1, np.full(n_pts, za)], 1),
            np.stack([u1, np.full(n_pts, zb)], 1),
            np.stack([u0, np.full(n_pts, zb)], 1),
        ], axis=1).reshape(-1, 2)
        base = np.arange(n_pts) * 4
        f = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)])
        meshes.append(Mesh(verts, f, norms, uv, None, name + "_wall"))
    return merge(meshes, name + "_wall")


# ---------------------------------------------------------------------------
# Loft between cross sections (vehicle bodies, cushions, arms)
# ---------------------------------------------------------------------------

def loft(sections: Sequence[np.ndarray], cap_ends: bool = True, smooth: bool = True, name: str = "loft") -> Mesh:
    """Skin a list of closed 3D rings with equal point counts.

    The rings are ordered along the sweep; each ring's points run in a consistent
    rotational direction. Ends are closed with ear-cut caps.
    """
    rings = [np.asarray(s, dtype=np.float64) for s in sections]
    n = len(rings[0])
    if any(len(r) != n for r in rings):
        raise ValueError("loft sections need equal point counts")
    m = len(rings)
    grid = np.stack(rings)  # (m, n, 3)
    gridw = np.concatenate([grid, grid[:, :1]], axis=1)  # wrap seam duplicate
    verts = gridw.reshape(-1, 3)
    w = n + 1
    faces = []
    for i in range(m - 1):
        for k in range(n):
            a = i * w + k
            b = i * w + k + 1
            c = (i + 1) * w + k + 1
            d = (i + 1) * w + k
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.asarray(faces, dtype=np.int64)
    arc = np.concatenate([np.zeros((m, 1)), np.cumsum(np.linalg.norm(np.diff(gridw, axis=1), axis=2), axis=1)], axis=1)
    centers = grid.mean(axis=1)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))])
    uv = np.stack([arc.reshape(-1), np.repeat(along, w)], axis=1)
    body = Mesh(verts, faces, None, uv, None, name)
    if body.signed_volume() < 0:
        body.faces = body.faces[:, ::-1].copy()
    body.normals = vertex_normals(body) if smooth else None
    parts = [body]
    if cap_ends:
        dir_along = centers[-1] - centers[0]
        for idx, outward in ((0, -1.0), (m - 1, 1.0)):
            ring = rings[idx]
            c = ring.mean(axis=0)
            fit_n = _plane_normal(ring)
            if np.dot(fit_n, dir_along) * outward < 0:
                fit_n = -fit_n
            u_ax = ring[0] - c
            u_ax -= fit_n * np.dot(u_ax, fit_n)
            u_ax /= max(np.linalg.norm(u_ax), EPS)
            v_ax = np.cross(fit_n, u_ax)
            pts2 = np.stack([(ring - c) @ u_ax, (ring - c) @ v_ax], axis=1)
            p2, tris = triangulate_polygon(pts2)
            pts3 = c + p2[:, :1] * u_ax + p2[:, 1:] * v_ax
            cap = Mesh(pts3, tris, np.tile(fit_n, (len(pts3), 1)), p2 - p2.min(0), None, name + "_cap")
            tri3 = pts3[tris]
            cn = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]).sum(0)
            if np.dot(cn, fit_n) < 0:
                cap.faces = cap.faces[:, ::-1].copy()
            parts.append(cap)
    out = merge(parts, name)
    out.face_material = None
    return out


def _plane_normal(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c)
    n = vt[2]
    return n / max(np.linalg.norm(n), EPS)


# ---------------------------------------------------------------------------
# Repetition helpers
# ---------------------------------------------------------------------------

def array_linear(mesh: Mesh, count: int, step: Sequence[float]) -> Mesh:
    step = np.asarray(step, dtype=np.float64)
    copies = [mesh.translated(step * i) for i in range(int(count))]
    return merge(copies, mesh.name)


def mirror_x(mesh: Mesh) -> Mesh:
    return mesh.transformed(scaling([-1.0, 1.0, 1.0]))
