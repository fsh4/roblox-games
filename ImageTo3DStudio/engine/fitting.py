"""Analysis-by-synthesis fitting of parametric templates to a single photo.

A template hypothesis (shape parameters + viewpoint) is rendered as a label map
with the software rasterizer and compared to the photo with three classical
terms:
  * zone agreement: rendered pixels on certain background, certain-object pixels
    left uncovered (undecided pixels are ignored),
  * chamfer distance from rendered part/silhouette boundaries to image edges,
  * reverse chamfer from image edges inside the object to rendered boundaries.
The search is a coarse viewpoint grid followed by deterministic pattern search
(coordinate descent with step halving) over continuous and discrete parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import render
from .analyzer import LoadedImage, Segmentation, edge_map
from .procedural import LOD, Template

TAU = 6.0


@dataclass
class FitTarget:
    width: int
    height: int
    zones: np.ndarray
    edges: np.ndarray
    edge_dt: np.ndarray
    bbox: Tuple[float, float, float, float]
    to_work: float
    probable: np.ndarray


def _vertical_envelope(zones: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Probable-object pixels: per column, from the top-most undecided edge above
    the certain-object pixels down to the lowest certain-object pixel.

    This captures surfaces that match the background colour (white bedding in
    front of a white wall) but are outlined by edges above solid parts.
    """
    h, w = zones.shape
    out = np.zeros((h, w), bool)
    fg = zones == 2
    cand = edges & (zones == 1)
    for x in range(w):
        rows = np.flatnonzero(fg[:, x])
        if len(rows) == 0:
            continue
        top_fg, bottom = rows.min(), rows.max()
        above = np.flatnonzero(cand[:top_fg, x])
        top = above.min() if len(above) else top_fg
        span = slice(top, bottom + 1)
        out[span, x] = zones[span, x] != 0
    return out & (zones == 1)


@dataclass
class View:
    azimuth: float
    elevation: float
    fov: float
    scale: float
    tx: float
    ty: float

    def to_dict(self) -> dict:
        return {k: round(float(v), 5) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "View":
        return View(**{k: float(d[k]) for k in ("azimuth", "elevation", "fov", "scale", "tx", "ty")})


@dataclass
class FitResult:
    template: str
    params: Dict[str, float]
    view: View
    score: float
    components: Dict[str, float]
    evaluations: int
    counts: Dict[str, Tuple[int, float]] = field(default_factory=dict)


def make_target(img: LoadedImage, seg: Segmentation, max_side: int = 320) -> FitTarget:
    h, w = img.rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    fw, fh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    zones = cv2.resize(seg.zones, (fw, fh), interpolation=cv2.INTER_NEAREST)
    edges_full = edge_map(img.rgb)
    edges = cv2.resize(cv2.dilate(edges_full, np.ones((2, 2), np.uint8)), (fw, fh), interpolation=cv2.INTER_AREA) > 40
    dt = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    near_edges = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    obj = (zones == 2) | ((zones == 1) & near_edges)
    ys, xs = np.nonzero(obj)
    if len(xs) == 0:
        bbox = (0.0, 0.0, float(fw), float(fh))
    else:
        bbox = (float(np.percentile(xs, 0.5)), float(np.percentile(ys, 0.5)), float(np.percentile(xs, 99.5)), float(np.percentile(ys, 99.5)))
    probable = _vertical_envelope(zones, edges)
    return FitTarget(fw, fh, zones, edges, dt, bbox, 1.0 / s, probable)


def _camera_for(bounds: np.ndarray, view: View, width: int, height: int) -> render.Camera:
    center = 0.5 * (bounds[0] + bounds[1])
    radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.5, 1e-3)
    dist = radius / np.tan(np.radians(view.fov) / 2.0) * 1.15
    return render.orbit_camera(center, dist, view.azimuth, view.elevation, width, height, view.fov)


def project_scene(vertices: np.ndarray, bounds: np.ndarray, view: View, width: int, height: int):
    cam = _camera_for(bounds, view, width, height)
    xy, z, iw = cam.project(vertices)
    c = np.array([width / 2.0, height / 2.0])
    xy = (xy - c) * view.scale + c + np.array([view.tx, view.ty])
    return xy, z, iw, cam


def label_map(template: Template, params: Dict[str, float], view: View, width: int, height: int, lod: Optional[LOD] = None):
    scene = template.build(params, lod or LOD(0.0, fast=True))
    keys = list(scene.materials.keys())
    verts, faces, labs = [], [], []
    off = 0
    for p in scene.parts:
        verts.append(p.mesh.vertices)
        faces.append(p.mesh.faces + off)
        labs.append(np.full(len(p.mesh.faces), keys.index(p.material)))
        off += len(p.mesh.vertices)
    V = np.concatenate(verts)
    F = np.concatenate(faces)
    L = np.concatenate(labs)
    bounds = scene.bounds()
    xy, z, iw, cam = project_scene(V, bounds, view, width, height)
    ras = render.rasterize(xy, z, F, width, height, iw)
    lab = np.where(ras.face_id >= 0, L[np.clip(ras.face_id, 0, None)], -1)
    return lab, scene, keys, (xy, z, cam, bounds)


def _boundary(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, bool)
    d1 = lab[:, 1:] != lab[:, :-1]
    d2 = lab[1:, :] != lab[:-1, :]
    b[:, 1:] |= d1
    b[:, :-1] |= d1
    b[1:, :] |= d2
    b[:-1, :] |= d2
    return b & ((lab >= 0) | np.roll(lab >= 0, 1, 0) | np.roll(lab >= 0, 1, 1))


def score_labels(lab: np.ndarray, target: FitTarget) -> Tuple[float, Dict[str, float]]:
    R = lab >= 0
    Z = target.zones
    n_r = int(R.sum())
    if n_r < 10:
        return 10.0, {"empty": 1.0}
    p_bg = float((R & (Z == 0)).sum()) / n_r
    fg = Z == 2
    r_fg = float((fg & ~R).sum()) / max(int(fg.sum()), 1)
    B = _boundary(lab)
    cham = float(np.minimum(target.edge_dt[B], TAU).mean() / TAU) if B.any() else 1.0
    region = cv2.dilate(R.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    E = target.edges & region
    if E.any() and B.any():
        dtb = cv2.distanceTransform((~B).astype(np.uint8), cv2.DIST_L2, 3)
        rev = float(np.minimum(dtb[E], TAU).mean() / TAU)
    else:
        rev = 1.0
    prob = target.probable
    r_prob = float((prob & ~R).sum()) / max(int(prob.sum()), 1) if prob.any() else 0.0
    total = 1.3 * p_bg + 1.6 * r_fg + 0.4 * r_prob + 0.9 * cham + 0.45 * rev
    return total, {"background_overlap": p_bg, "object_missed": r_fg, "probable_missed": r_prob, "edge_chamfer": cham, "edge_coverage": rev}


def _prior_penalty(template: Template, params: Dict[str, float]) -> float:
    pen = 0.0
    for k, d in template.params.items():
        if not d.fit or d.kind != "length":
            continue
        v = max(params.get(k, d.default), 1e-6)
        ratio = np.log(v / max(d.default, 1e-6))
        pen += (ratio / np.log(2.5)) ** 2
    return 0.04 * pen


@dataclass
class _Dim:
    name: str
    step: float
    lo: float
    hi: float
    discrete: bool = False
    log: bool = False


class _Evaluator:
    """Builds, projects and scores hypotheses; keeps the 2D placement stable."""

    def __init__(self, template: Template, target: FitTarget):
        self.t = template
        self.target = target
        self.count = 0

    def geometry(self, params):
        scene = self.t.build(params, LOD(0.0, fast=True))
        keys = list(scene.materials.keys())
        V = np.concatenate([p.mesh.vertices for p in scene.parts])
        offs = np.cumsum([0] + [len(p.mesh.vertices) for p in scene.parts])[:-1]
        F = np.concatenate([p.mesh.faces + o for p, o in zip(scene.parts, offs)])
        L = np.concatenate([np.full(len(p.mesh.faces), keys.index(p.material)) for p in scene.parts])
        return V, F, L, scene.bounds()

    def raw_bbox(self, V, bounds, view: View):
        base = View(view.azimuth, view.elevation, view.fov, 1.0, 0.0, 0.0)
        xy, _, _, _ = project_scene(V, bounds, base, self.target.width, self.target.height)
        return xy.min(0), xy.max(0)

    def place(self, V, bounds, view: View, box) -> View:
        lo, hi = self.raw_bbox(V, bounds, view)
        (x0, y0), (x1, y1) = box
        bw, bh = max(hi[0] - lo[0], 1e-3), max(hi[1] - lo[1], 1e-3)
        s = float(np.exp(0.5 * (np.log(max(x1 - x0, 1e-3) / bw) + np.log(max(y1 - y0, 1e-3) / bh))))
        c = np.array([self.target.width / 2.0, self.target.height / 2.0])
        t = 0.5 * (np.array([x0, y0]) + np.array([x1, y1])) - ((0.5 * (lo + hi) - c) * s + c)
        return View(view.azimuth, view.elevation, view.fov, s, float(t[0]), float(t[1]))

    def placed_box(self, V, bounds, view: View):
        lo, hi = self.raw_bbox(V, bounds, view)
        c = np.array([self.target.width / 2.0, self.target.height / 2.0])
        t = np.array([view.tx, view.ty])
        return ((lo - c) * view.scale + c + t), ((hi - c) * view.scale + c + t)

    def score(self, params, view: View, geo=None):
        V, F, L, bounds = geo if geo is not None else self.geometry(params)
        xy, z, iw, _ = project_scene(V, bounds, view, self.target.width, self.target.height)
        ras = render.rasterize(xy, z, F, self.target.width, self.target.height, iw)
        lab = np.where(ras.face_id >= 0, L[np.clip(ras.face_id, 0, None)], -1)
        sc, comp = score_labels(lab, self.target)
        self.count += 1
        return sc + _prior_penalty(self.t, params), comp


def fit_template(template: Template, target: FitTarget, init_params: Optional[Dict[str, float]] = None, max_evals: int = 360, coarse_only: bool = False, fov: float = 40.0) -> FitResult:
    params = template.clamp(init_params or template.defaults())
    ev = _Evaluator(template, target)
    geo0 = ev.geometry(params)
    box = ((target.bbox[0], target.bbox[1]), (target.bbox[2], target.bbox[3]))
    seeds = []
    for f in (fov - 10.0, fov + 10.0):
        for az in range(-70, 71, 10):
            for el in (8, 16, 24, 32):
                v = ev.place(geo0[0], geo0[3], View(az, el, f, 1.0, 0.0, 0.0), box)
                sc, comp = ev.score(params, v, geo0)
                seeds.append((sc, v, comp))
    seeds.sort(key=lambda t: t[0])
    if coarse_only:
        sc, v, comp = seeds[0]
        return FitResult(template.name, params, v, sc, comp, ev.count)

    anchor = max(template.size_params, key=lambda k: template.params[k].default)
    dims: List[_Dim] = [
        _Dim("@azimuth", 8.0, -90.0, 90.0), _Dim("@elevation", 6.0, 0.0, 60.0), _Dim("@fov", 8.0, 15.0, 75.0),
        _Dim("@scale", 0.06, 0.2, 5.0, log=True), _Dim("@tx", 3.0, -400, 400), _Dim("@ty", 3.0, -400, 400),
    ]
    for k, d in template.params.items():
        if k == anchor:
            continue
        if d.fit:
            dims.append(_Dim(k, 0.15, d.lo, d.hi, log=True))

    def get(p, v, dim):
        return getattr(v, dim.name[1:]) if dim.name.startswith("@") else p[dim.name]

    def setv(p, v, dim, val):
        val = float(np.clip(val, dim.lo, dim.hi))
        if dim.discrete:
            val = float(round(val))
        if dim.name.startswith("@"):
            return p, View(**{**v.__dict__, dim.name[1:]: val})
        p = dict(p)
        p[dim.name] = val
        return p, v

    overall = None
    picked: List[View] = []
    for sc0, v0, _ in seeds:
        if any(abs(v0.azimuth - q.azimuth) < 20 and abs(v0.elevation - q.elevation) < 15 for q in picked):
            continue
        picked.append(v0)
        if len(picked) >= 3:
            break
    budget = max(80, (max_evals - ev.count) // max(len(picked), 1))
    for v0 in picked:
        p, v = dict(params), v0
        geo = ev.geometry(p)
        cur, comp = ev.score(p, v, geo)
        steps = {d.name: d.step for d in dims}
        start = ev.count
        while ev.count - start < budget:
            improved = False
            for dim in dims:
                st = steps[dim.name]
                if st < (0.6 if dim.discrete else 1e-3):
                    continue
                for sign in (1, -1):
                    base = get(p, v, dim)
                    val = base * float(np.exp(sign * st)) if dim.log else base + sign * st
                    if dim.discrete and round(val) == round(base):
                        continue
                    np_, nv = setv(p, v, dim, val)
                    reshape = not dim.name.startswith("@") or dim.name in ("@azimuth", "@elevation", "@fov")
                    ngeo = ev.geometry(np_) if not dim.name.startswith("@") else geo
                    if reshape:
                        nv = ev.place(ngeo[0], ngeo[3], nv, ev.placed_box(geo[0], geo[3], v))
                    sc, cc = ev.score(np_, nv, ngeo)
                    if sc < cur - 1e-5:
                        p, v, cur, comp, geo = np_, nv, sc, cc, ngeo
                        improved = True
                        break
                if ev.count - start >= budget:
                    break
            if not improved:
                left = False
                for dim in dims:
                    if not dim.discrete:
                        steps[dim.name] *= 0.5
                        left |= steps[dim.name] > (0.01 if dim.log else 0.25)
                if not left:
                    break
        if overall is None or cur < overall[0]:
            overall = (cur, p, v, comp)
    cur, p, v, comp = overall
    return FitResult(template.name, p, v, float(cur), comp, ev.count)


# ---------------------------------------------------------------------------
# Counting repeated elements along a projected 3D segment
# ---------------------------------------------------------------------------

COUNT_HOOKS = {
    "bed": [("headboard_slats", "bars", lambda p: ((-p["width"] / 2 + p["post_size"] * 1.1, p["headboard_height"] * 0.62, p["length"] / 2 - p["post_size"] / 2), (p["width"] / 2 - p["post_size"] * 1.1, p["headboard_height"] * 0.62, p["length"] / 2 - p["post_size"] / 2)), 0, 14)],
    "chair": [("back_slats", "bars", lambda p: ((-p["width"] / 2 + p["leg_size"] * 1.2, (p["seat_height"] + p["back_height"]) / 2, p["depth"] / 2 - p["leg_size"]), (p["width"] / 2 - p["leg_size"] * 1.2, (p["seat_height"] + p["back_height"]) / 2, p["depth"] / 2 - p["leg_size"])), 0, 8)],
    "cabinet": [("rows", "seams", lambda p: ((0.0, p["plinth_height"] + 0.1, -p["depth"] / 2), (0.0, p["height"] - 0.12, -p["depth"] / 2)), 1, 8)],
    "shelf": [("shelves", "seams", lambda p: ((0.0, p["plinth_height"] + 0.15, -p["depth"] / 2 + 0.1), (0.0, p["height"] - 0.15, -p["depth"] / 2 + 0.1)), 0, 10)],
    "sofa": [("cushions", "seams", lambda p: ((-p["width"] / 2 + p["arm_width"] * 1.1, p["seat_height"] + 0.12, -p["depth"] / 2 + 0.2), (p["width"] / 2 - p["arm_width"] * 1.1, p["seat_height"] + 0.12, -p["depth"] / 2 + 0.2)), 1, 5)],
}


def _runs(binary: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(binary):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(binary) - start >= min_len:
        runs.append((start, len(binary)))
    return runs


def count_repeats(template: Template, fit: FitResult, img: LoadedImage, target: FitTarget, seg: Optional[Segmentation] = None) -> Dict[str, Tuple[int, float]]:
    """Count repeated elements (slats, drawer seams, cushions) along projected lines.

    "bars": object-coloured runs separated by background (slats against a wall),
    measured on the distance-to-background-colour signal. "seams": thin dark
    gaps between panels; the element count is seams + 1.
    """
    hooks = COUNT_HOOKS.get(template.name, [])
    if not hooks:
        return {}
    scene = template.build(fit.params, LOD(0.0, fast=True))
    h, w = img.rgb.shape[:2]
    lab = cv2.cvtColor(img.rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = lab[..., 0]
    if seg is not None and len(seg.background_lab):
        wts = np.array([0.55, 1.0, 1.0], np.float32)
        bgd = np.min(np.linalg.norm((lab[:, :, None, :] - seg.background_lab[None, None]) * wts, axis=3), axis=2)
    else:
        bgd = 255.0 - gray
    view = scaled_view(fit.view, w / target.width, w, h)
    out = {}
    for name, mode, seg_fn, lo, hi in hooks:
        a, b = seg_fn(fit.params)
        xy, *_ = project_scene(np.array([a, b], dtype=np.float64), scene.bounds(), view, w, h)
        n = int(max(32, np.linalg.norm(xy[1] - xy[0])))
        line = np.linspace(xy[0], xy[1], n)
        signal = bgd if mode == "bars" else 255.0 - gray
        prof = np.mean([render.sample_image_xy(signal[..., None], line + np.array([0, off]))[:, 0] for off in (-3, 0, 3)], axis=0)
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        p10, p90 = np.percentile(prof, 10), np.percentile(prof, 90)
        if p90 - p10 < 6:
            continue
        thr = 0.5 * (p10 + p90)
        runs = _runs(prof > thr, max(2, n // 120))
        inner = [r for r in runs if r[0] > 1 and r[1] < n - 1] if mode == "bars" else [r for r in runs if (r[1] - r[0]) < n * 0.08]
        if mode == "bars":
            count = len(inner)
        else:
            count = len(inner) + 1
        widths = np.array([r[1] - r[0] for r in inner], dtype=np.float64)
        regular = float(1.0 - np.std(widths) / (np.mean(widths) + 1e-9)) if len(widths) > 1 else 0.5
        if lo <= count <= hi and regular > 0.35:
            out[name] = (int(count), regular)
    return out


def scaled_view(view: View, factor: float, width: int, height: int) -> View:
    """Express a fit-resolution view at another image resolution."""
    return View(view.azimuth, view.elevation, view.fov, view.scale, view.tx * factor, view.ty * factor)
