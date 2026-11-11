"""Mesh and asset validation with safe automatic repairs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .geometry import Mesh
from .scene import Scene
from .settings import ROBLOX_MESH_TRIANGLE_LIMIT

MAX_PART_STUDS = 2048.0


@dataclass
class MeshReport:
    name: str
    triangles: int
    vertices: int
    degenerate: int
    duplicate_faces: int
    open_edges: int
    nonmanifold_edges: int
    inconsistent_edges: int
    volume: float
    nan_values: bool
    uv_out_of_range: int
    normal_flips: int

    @property
    def watertight(self) -> bool:
        return self.open_edges == 0 and self.nonmanifold_edges == 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["watertight"] = self.watertight
        return d


@dataclass
class ValidationReport:
    meshes: List[MeshReport]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "repairs": self.repairs, "meshes": [m.to_dict() for m in self.meshes]}


def _edge_stats(mesh: Mesh) -> Tuple[int, int, int]:
    key = np.round(mesh.vertices / 1e-5).astype(np.int64)
    _, pid = np.unique(key, axis=0, return_inverse=True)
    f = pid.reshape(-1)[mesh.faces]
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, ucount = np.unique(undirected, axis=0, return_counts=True)
    _, dcount = np.unique(directed, axis=0, return_counts=True)
    return int((ucount == 1).sum()), int((ucount > 2).sum()), int((dcount > 1).sum())


def inspect_mesh(mesh: Mesh) -> MeshReport:
    areas = mesh.face_areas()
    degenerate = int((areas <= 1e-12).sum())
    sorted_faces = np.sort(mesh.faces, axis=1)
    _, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    dup = int((counts > 1).sum())
    open_e, nonman, incons = _edge_stats(mesh)
    nan = bool(not np.isfinite(mesh.vertices).all() or (mesh.normals is not None and not np.isfinite(mesh.normals).all()))
    uv_bad = 0
    if mesh.uvs is not None:
        uv_bad = int(((mesh.uvs < -1e-6) | (mesh.uvs > 1 + 1e-6)).any(axis=1).sum())
    flips = 0
    if mesh.normals is not None and len(mesh.faces):
        fn = mesh.face_normals()
        vn = mesh.normals[mesh.faces].mean(axis=1)
        flips = int((np.einsum("ij,ij->i", fn, vn) < -0.2).sum())
    return MeshReport(mesh.name, mesh.triangle_count, mesh.vertex_count, degenerate, dup, open_e, nonman, incons,
                      mesh.signed_volume(), nan, uv_bad, flips)


def flip_slivers(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, int]:
    """Remove zero-area triangles with three distinct, collinear corners by edge flips.

    Deleting such a triangle would leave a T-junction (open edges). Instead its
    longest edge is flipped with the neighbouring triangle across that edge,
    which keeps the surface closed and leaves two proper triangles.
    """
    f = faces.copy()
    flipped = 0

    def area(t):
        a, b, c = vertices[t[0]], vertices[t[1]], vertices[t[2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    for _ in range(4):
        tri = vertices[f]
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        bad = np.flatnonzero((areas <= 1e-12) & distinct)
        if not len(bad):
            break
        edge_face = {}
        for fi, (a, b, c) in enumerate(f.tolist()):
            edge_face[(a, b)] = fi
            edge_face[(b, c)] = fi
            edge_face[(c, a)] = fi
        progress = False
        for fi in bad.tolist():
            a, b, c = f[fi].tolist()
            corners = [(a, b, c), (b, c, a), (c, a, b)]
            p, q, r = max(corners, key=lambda t: float(np.linalg.norm(vertices[t[1]] - vertices[t[0]])))
            g = edge_face.get((q, p))
            if g is None or g == fi:
                continue
            s = next((v for v in f[g].tolist() if v not in (p, q)), None)
            if s is None or s == r or (r, s) in edge_face or (s, r) in edge_face:
                continue
            t1, t2 = (p, s, r), (s, q, r)
            if area(t1) <= 1e-12 or area(t2) <= 1e-12:
                continue
            for e in ((p, q), (q, r), (r, p), (q, p), (p, s), (s, q)):
                edge_face.pop(e, None)
            f[fi], f[g] = t1, t2
            for e in ((p, s), (s, r), (r, p)):
                edge_face[e] = fi
            for e in ((s, q), (q, r), (r, s)):
                edge_face[e] = g
            flipped += 1
            progress = True
        if not progress:
            break
    return f, flipped


def repair_mesh(mesh: Mesh) -> Tuple[Mesh, List[str]]:
    notes = []
    out = mesh.copy()
    out.faces, flipped = flip_slivers(out.vertices, out.faces)
    if flipped:
        notes.append(f"{mesh.name}: flipped {flipped} zero-area sliver triangles")
    f = out.faces
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    if (~keep).any():
        notes.append(f"{mesh.name}: removed {int((~keep).sum())} degenerate triangles")
        out.faces = out.faces[keep]
        if out.face_material is not None:
            out.face_material = out.face_material[keep]
    sorted_faces = np.sort(out.faces, axis=1)
    _, first = np.unique(sorted_faces, axis=0, return_index=True)
    if len(first) < len(out.faces):
        notes.append(f"{mesh.name}: removed {len(out.faces) - len(first)} duplicate triangles")
        first = np.sort(first)
        out.faces = out.faces[first]
        if out.face_material is not None:
            out.face_material = out.face_material[first]
    used = np.unique(out.faces)
    if len(used) < len(out.vertices):
        remap = np.full(len(out.vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.vertices = out.vertices[used]
        if out.normals is not None:
            out.normals = out.normals[used]
        if out.uvs is not None:
            out.uvs = out.uvs[used]
        out.faces = remap[out.faces]
    if out.signed_volume() < 0:
        out.faces = out.faces[:, ::-1].copy()
        if out.normals is not None:
            out.normals = -out.normals
        notes.append(f"{mesh.name}: flipped inward-facing winding")
    if out.uvs is not None and ((out.uvs < 0) | (out.uvs > 1)).any():
        out.uvs = np.clip(out.uvs, 0.0, 1.0)
        notes.append(f"{mesh.name}: clamped UVs into 0..1")
    if out.normals is not None:
        n = np.linalg.norm(out.normals, axis=1, keepdims=True)
        out.normals = np.where(n > 1e-9, out.normals / np.maximum(n, 1e-9), np.array([0.0, 1.0, 0.0]))
    return out, notes


def validate_scene(scene: Scene, budget: int, texture_maps: Dict[str, np.ndarray] = None) -> ValidationReport:
    reports = []
    rep = ValidationReport(reports)
    for part in scene.parts:
        fixed, notes = repair_mesh(part.mesh)
        part.mesh = fixed
        rep.repairs.extend(notes)
        r = inspect_mesh(fixed)
        reports.append(r)
        if r.nan_values:
            rep.errors.append(f"{part.name}: NaN or infinite coordinates")
        if r.triangles == 0:
            rep.errors.append(f"{part.name}: no triangles")
        if r.nonmanifold_edges:
            rep.warnings.append(f"{part.name}: {r.nonmanifold_edges} non-manifold edges")
        if r.open_edges:
            rep.warnings.append(f"{part.name}: {r.open_edges} open edges (not watertight)")
        if r.inconsistent_edges:
            rep.warnings.append(f"{part.name}: {r.inconsistent_edges} edges with inconsistent winding")
        if r.normal_flips:
            rep.warnings.append(f"{part.name}: {r.normal_flips} triangles whose normals oppose their winding")
    total = scene.triangle_count
    if total > budget * 1.1:
        rep.warnings.append(f"triangle count {total} exceeds the requested budget {budget}")
    lo, hi = scene.bounds()
    ext = hi - lo
    if (ext > MAX_PART_STUDS).any():
        rep.errors.append(f"model is {ext.max():.0f} studs long; Roblox parts are limited to {MAX_PART_STUDS:.0f} studs")
    if (ext < 0.01).any():
        rep.errors.append("model has no thickness along one axis")
    for p in scene.parts:
        if p.mesh.triangle_count > ROBLOX_MESH_TRIANGLE_LIMIT:
            rep.warnings.append(f"{p.name}: {p.mesh.triangle_count} triangles exceeds the Roblox per-mesh limit; it will be split on export")
    if texture_maps:
        for name, arr in texture_maps.items():
            h, w = arr.shape[:2]
            if h != w or (h & (h - 1)):
                rep.warnings.append(f"texture {name} is {w}x{h}; square power-of-two sizes are recommended")
            if max(h, w) > 4096:
                rep.errors.append(f"texture {name} exceeds 4096 px")
    return rep
