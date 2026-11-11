"""Triangle-budget solving, decimation and Roblox-specific optimisation.

Budget strategy, in order of preference:
  1. Level of detail by construction: rebuild the parametric scene at the
     highest tessellation level ``q`` that fits the budget (binary search). This
     keeps clean topology, UV islands and exact silhouettes.
  2. Quadric-error-metric decimation (Garland-Heckbert edge collapse, via the
     MIT-licensed fast-simplification library) per part, with the budget split
     in proportion to each part's surface detail, followed by normal and UV
     regeneration. Used for scanned/carved meshes and for budgets below the
     coarsest procedural level.
  3. When the requested budget cannot be met without destroying the shape,
     the result is clamped at the smallest safe count and the user is told.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as g
from .scene import Part, Scene
from .settings import ROBLOX_MESH_TRIANGLE_LIMIT

MIN_PART_TRIANGLES = 12


@dataclass
class BudgetReport:
    requested: int
    achieved: int
    original_triangles: int
    original_vertices: int
    final_vertices: int
    method: str
    lod_q: float
    reachable: bool
    minimum_safe: int
    quality_impact: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def reduction_percent(self) -> float:
        if self.original_triangles <= 0:
            return 0.0
        return 100.0 * (1.0 - self.achieved / self.original_triangles)

    def to_dict(self) -> dict:
        return {
            "requested_triangles": self.requested, "final_triangles": self.achieved,
            "original_triangles": self.original_triangles, "original_vertices": self.original_vertices,
            "final_vertices": self.final_vertices, "reduction_percent": round(self.reduction_percent, 2),
            "method": self.method, "lod_q": round(self.lod_q, 3), "budget_reachable": self.reachable,
            "minimum_safe_triangles": self.minimum_safe, "quality_impact": self.quality_impact, "notes": self.notes,
        }


def solve_lod(build: Callable[[dict], Scene], budget: int, q_cap: float = 1.0, iterations: int = 8) -> Tuple[Scene, dict]:
    """Highest level of detail that fits ``budget``.

    First a binary search on the shared quality ``q``; then each sub-level
    (cushion subdivision, round segments, bevel segments) is raised on its own
    while the budget allows, so the triangle count lands close to the request.
    """
    lod = {"q": 0.0}
    best = build(lod)
    if best.triangle_count > budget:
        return best, lod
    top = build({"q": q_cap})
    if top.triangle_count <= budget:
        return top, {"q": q_cap}
    lo, hi = 0.0, q_cap
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        sc = build({"q": mid})
        if sc.triangle_count <= budget:
            best, lod, lo = sc, {"q": mid}, mid
        else:
            hi = mid
    for key in ("cushion_q", "round_q", "bevel_q"):
        probe = build({**lod, key: q_cap})
        if probe.triangle_count == best.triangle_count:
            continue
        if probe.triangle_count <= budget:
            best, lod = probe, {**lod, key: q_cap}
            continue
        a, b = lod["q"], q_cap
        for _ in range(6):
            mid = 0.5 * (a + b)
            sc = build({**lod, key: mid})
            if sc.triangle_count <= budget:
                best, a = sc, mid
                lod = {**lod, key: mid}
            else:
                b = mid
    return best, lod


def decimate_mesh(mesh: g.Mesh, target: int, smooth_angle: float = 40.0) -> g.Mesh:
    """QEM decimation on welded positions; returns a mesh with recomputed normals."""
    import fast_simplification

    target = int(max(target, 4))
    if mesh.triangle_count <= target:
        return mesh
    key = np.round(mesh.vertices / 1e-6).astype(np.int64)
    uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pts = mesh.vertices[first]
    tris = inverse.reshape(-1)[mesh.faces]
    good = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[good].astype(np.int64)
    best = None
    # Aggressive collapses can pinch thin parts (chair slats) into edges shared
    # by more than two triangles. Retry gentler or slightly lower targets and
    # keep the first manifold result within the budget.
    for agg, scale in ((7.0, 1.0), (5.0, 1.0), (7.0, 0.97), (5.0, 0.97), (7.0, 0.93), (5.0, 0.93), (3.0, 0.9)):
        out_pts, out_tris = fast_simplification.simplify(pts, tris, target_count=max(4, int(target * scale)), agg=agg)
        out = _drop_bad_faces(g.Mesh(out_pts, out_tris, None, np.zeros((len(out_pts), 2)), None, mesh.name))
        if out.triangle_count > target:
            continue
        defects = _topology_defects(out.faces)
        if best is None or defects < best[0]:
            best = (defects, out)
        if defects == 0:
            break
    out = best[1] if best is not None else out
    if _topology_defects(out.faces):
        out.vertices, out.faces = split_pinches(out.vertices, out.faces)
        out.uvs = np.zeros((len(out.vertices), 2))
    if out.signed_volume() < 0:
        out.faces = out.faces[:, ::-1].copy()
    return g.recompute_normals(out, smooth_angle)


def split_pinches(vertices: np.ndarray, faces: np.ndarray, gap: float = 0.002) -> Tuple[np.ndarray, np.ndarray]:
    """Separate surface sheets that touch along an edge or at a vertex.

    Triangle corners are joined only across edges shared by exactly two
    triangles; every resulting corner group becomes its own vertex. Where a
    vertex splits, each copy moves ``gap`` studs along the normal of its own
    sheet, so the sheets no longer touch and every edge borders two triangles.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    corner_vertex = faces.reshape(-1)
    c0 = np.arange(3 * n_faces)
    c1 = (c0 // 3) * 3 + (c0 % 3 + 1) % 3
    va, vb = corner_vertex[c0], corner_vertex[c1]
    key = np.minimum(va, vb).astype(np.int64) * (len(vertices) + 1) + np.maximum(va, vb)
    order = np.argsort(key, kind="stable")
    _, start, counts = np.unique(key[order], return_index=True, return_counts=True)
    pair = counts == 2
    i1, i2 = order[start[pair]], order[start[pair] + 1]
    same = va[i1] == va[i2]
    rows = np.concatenate([c0[i1], c1[i1]])
    cols = np.concatenate([np.where(same, c0[i2], c1[i2]), np.where(same, c1[i2], c0[i2])])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * n_faces, 3 * n_faces))
    n_groups, group = connected_components(graph, directed=False)
    positions = np.zeros((n_groups, 3))
    positions[group] = vertices[corner_vertex]
    source = np.zeros(n_groups, dtype=np.int64)
    source[group] = corner_vertex
    copies = np.bincount(source, minlength=len(vertices))
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    group_n = np.zeros((n_groups, 3))
    np.add.at(group_n, group, np.repeat(face_n, 3, axis=0))
    group_n /= np.maximum(np.linalg.norm(group_n, axis=1, keepdims=True), 1e-12)
    split = copies[source] > 1
    positions[split] += group_n[split] * gap
    return positions, group.reshape(-1, 3)


def _topology_defects(faces: np.ndarray) -> int:
    """Edges shared by more than two triangles plus edges used twice in the same direction."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, ucount = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    _, dcount = np.unique(directed, axis=0, return_counts=True)
    return int((ucount > 2).sum() + (dcount > 1).sum())


def transfer_normals(source: g.Mesh, target: g.Mesh, k: int = 6) -> np.ndarray:
    """Shading normals for ``target`` interpolated from a denser ``source`` of the same surface.

    Decimated organic surfaces keep the smooth shading of the full-resolution
    surface instead of the facet normals of their sliver triangles. Where the
    transferred normal leans more than 25 degrees away from the target's own
    crease-aware normal (a rim or other hard edge), the own normal is kept so
    the edge stays crisp instead of smearing into wedge-shaped highlights.
    """
    own = target.normals if target.normals is not None else g.vertex_normals(target)
    if source.normals is None or len(source.vertices) < k:
        return own
    dist, idx = cKDTree(source.vertices).query(target.vertices, k=k)
    w = 1.0 / np.maximum(dist, 1e-9)
    n = np.einsum("nk,nkc->nc", w, source.normals[idx])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    agree = np.sum(n * own, axis=1) > np.cos(np.radians(25.0))
    return np.where(agree[:, None], n, own)


def _drop_bad_faces(mesh: g.Mesh) -> g.Mesh:
    f = mesh.faces
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    keep &= mesh.face_areas() > 1e-12
    out = mesh.copy()
    out.faces = f[keep]
    if out.face_material is not None:
        out.face_material = out.face_material[keep]
    return out


def allocate_budget(parts: List[Part], budget: int) -> List[int]:
    """Split a triangle budget over parts by area-weighted detail with floors."""
    counts = np.array([p.mesh.triangle_count for p in parts], dtype=np.float64)
    areas = np.array([max(p.mesh.surface_area(), 1e-6) for p in parts])
    weight = np.sqrt(counts) * np.sqrt(areas)
    floor = np.minimum(counts, MIN_PART_TRIANGLES)
    spare = max(budget - floor.sum(), 0)
    alloc = floor + spare * weight / max(weight.sum(), 1e-9)
    alloc = np.minimum(alloc, counts)
    for _ in range(6):
        left = budget - alloc.sum()
        room = counts - alloc
        if left <= 1 or room.sum() <= 0:
            break
        alloc += np.minimum(room, left * room / room.sum())
    return [int(max(MIN_PART_TRIANGLES if c >= MIN_PART_TRIANGLES else c, a)) for a, c in zip(alloc, counts)]


def surface_samples(meshes: List[g.Mesh], n: int, seed: int = 3) -> np.ndarray:
    verts = []
    areas = []
    for m in meshes:
        tri = m.vertices[m.faces]
        verts.append(tri)
        areas.append(m.face_areas())
    tri = np.concatenate(verts)
    a = np.concatenate(areas)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tri), size=n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    w = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], axis=1)
    return np.einsum("nk,nkc->nc", w, tri[idx])


def quality_impact(reference: List[g.Mesh], result: List[g.Mesh], samples: int = 40_000) -> Dict[str, float]:
    """Symmetric surface deviation between a reference and an optimised model."""
    if not reference or not result:
        return {}
    a = surface_samples(reference, samples)
    b = surface_samples(result, samples)
    diag = float(np.linalg.norm(a.max(0) - a.min(0))) or 1.0
    d_ab, _ = cKDTree(b).query(a, k=1)
    d_ba, _ = cKDTree(a).query(b, k=1)
    d = np.concatenate([d_ab, d_ba])
    mean = float(d.mean() / diag * 100)
    p95 = float(np.percentile(d, 95) / diag * 100)
    rating = "negligible" if p95 < 0.6 else "low" if p95 < 1.5 else "moderate" if p95 < 4.0 else "high"
    return {"mean_deviation_percent": round(mean, 3), "p95_deviation_percent": round(p95, 3), "rating": rating}


def enforce_budget(scene: Scene, budget: int, smooth_angle: float) -> Tuple[Scene, List[str]]:
    notes: List[str] = []
    total = scene.triangle_count
    if total <= budget:
        return scene, notes
    alloc = allocate_budget(scene.parts, budget)
    parts = []
    for p, a in zip(scene.parts, alloc):
        if p.mesh.triangle_count > a:
            from .reconstruction import _assign_chart_uvs

            m = decimate_mesh(p.mesh, a, smooth_angle)
            if scene.meta.get("smooth_surface"):
                m.normals = transfer_normals(p.mesh, m)
            _assign_chart_uvs(m)
            parts.append(Part(p.name, m, p.material, p.frame, p.grain_axis, "u", p.seed, p.photo, p.native, p.detail if p.detail is not None else p.mesh, p.group))
        else:
            parts.append(p)
    out = Scene(scene.name, parts, scene.materials, dict(scene.meta))
    if out.triangle_count > budget * 1.05:
        notes.append(f"Requested {budget} triangles; the smallest safe result is {out.triangle_count} because every part keeps a closed minimum shape.")
    return out, notes


def split_for_roblox(meshes: List[Tuple[str, g.Mesh]], limit: int = ROBLOX_MESH_TRIANGLE_LIMIT) -> List[Tuple[str, g.Mesh]]:
    """Ensure no exported mesh exceeds Roblox's per-MeshPart triangle limit."""
    out = []
    for name, m in meshes:
        if m.triangle_count <= limit:
            out.append((name, m))
            continue
        order = np.argsort(m.vertices[m.faces].mean(axis=1) @ np.array([1.0, 0.001, 0.000001]))
        chunks = int(np.ceil(m.triangle_count / limit))
        for i, idx in enumerate(np.array_split(order, chunks)):
            faces = m.faces[idx]
            used, inv = np.unique(faces, return_inverse=True)
            sub = g.Mesh(m.vertices[used], inv.reshape(-1, 3), m.normals[used] if m.normals is not None else None,
                         m.uvs[used] if m.uvs is not None else None, m.face_material[idx] if m.face_material is not None else None, f"{name}_{i + 1}")
            out.append((f"{name}_{i + 1}", sub))
    return out


def apply_shading(scene: Scene, shading: str, smooth_angle: float) -> None:
    if shading == "flat":
        for p in scene.parts:
            p.mesh = g.recompute_normals(p.mesh, 0.0)


def place_pivot(scene: Scene, pivot: str) -> np.ndarray:
    """Translate the scene so the requested pivot sits at the origin."""
    lo, hi = scene.bounds()
    if pivot == "center":
        offset = -(lo + hi) / 2
    elif pivot == "back":
        offset = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -hi[2]])
    elif pivot == "origin":
        offset = np.zeros(3)
    else:
        offset = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    if np.linalg.norm(offset) > 0:
        scene.transform(g.translation(offset))
    return offset


def scale_scene(scene: Scene, factors) -> None:
    f = np.asarray(factors, dtype=np.float64)
    m = g.scaling(f)
    for p in scene.parts:
        p.mesh.apply_transform(m)
        if p.mesh.uvs is not None:
            p.mesh.uvs = p.mesh.uvs * float(np.cbrt(np.prod(np.abs(f))))
        if p.detail is not None:
            p.detail.apply_transform(m)
        p.frame = m @ p.frame
        for prim in p.native:
            rot = prim.cframe[:3, :3]
            local_scale = [float(np.linalg.norm(f * rot[:, i])) for i in range(3)]
            prim.size = tuple(float(s * ls) for s, ls in zip(prim.size, local_scale))
            pos = prim.cframe[:3, 3] * f
            cf = np.eye(4)
            cf[:3, :3] = rot
            cf[:3, 3] = pos
            prim.cframe = cf
