"""Reconstruction strategies and their selection.

Single image (Mode A):
  * ``template``  - parametric template fitted by analysis-by-synthesis (Mode C)
  * ``revolve``   - surface of revolution from a mirror-symmetric silhouette
  * ``extrude``   - flat object: contour extruded with a bevel (signs, drawings)
  * ``inflate``   - organic object: silhouette inflated by its distance field
Multiple images (Mode B):
  * ``visual_hull`` - silhouettes from labelled reference views carved together
  * ``turntable``   - unordered photos around the object, camera yaw recovered
                      from SIFT matches and essential matrices, then carved

Every strategy produces a JSON-serialisable *spec* (the editable intermediate
representation) plus optional arrays saved next to it. ``build_scene`` turns a
spec back into geometry at any level of detail, so later edits never repeat the
image analysis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from . import analyzer as an
from . import geometry as g
from . import render
from .fitting import FitTarget, View, count_repeats, fit_template, label_map, make_target, scaled_view
from .materials import MaterialSpec, infer_kind, make_material, srgb_to_linear
from .procedural import LOD, TEMPLATES, Template, get_template, m2s
from .scene import NativePrimitive, Part, Scene
from .textures import PhotoProjection

Progress = Optional[Callable[[str, float], None]]

CATEGORY_TEMPLATES = {
    "bed": "bed", "chair": "chair", "stool": "chair", "table": "table", "desk": "table",
    "sofa": "sofa", "couch": "sofa", "cabinet": "cabinet", "dresser": "cabinet", "drawer": "cabinet",
    "wardrobe": "cabinet", "shelf": "shelf", "bookcase": "shelf", "lamp": "lamp", "house": "house",
    "building": "house", "car": "car", "vehicle": "car", "box": "box", "crate": "box",
}
GENERIC = ("revolve", "extrude", "inflate")
TYPICAL_HEIGHT_M = {"revolve": 0.6, "extrude": 1.0, "inflate": 0.8, "box": 0.6}


class ReconstructionError(RuntimeError):
    pass


def _register_box_template() -> None:
    from .procedural import MaterialSlot, P, Template as T, _register

    def _box(b, p):
        b.board("Body", "surface", (p["width"], p["height"], p["depth"]), (0, p["height"] / 2, 0), p["bevel"], 0)

    if "box" not in TEMPLATES:
        _register(T("box", "Box", {
            "width": P(m2s(0.5), 0.3, 40.0, "Width", fit=True),
            "height": P(m2s(0.5), 0.3, 40.0, "Height", fit=True),
            "depth": P(m2s(0.5), 0.3, 40.0, "Depth", fit=True),
            "bevel": P(0.03, 0.0, 0.3, "Edge radius"),
        }, [MaterialSlot("surface", "Surface", ("photo",), "photo", (0.7, 0.7, 0.7))], _box, ("width", "height", "depth"),
            "Cuboid with the photograph projected onto its visible faces."))


_register_box_template()


@dataclass
class Source:
    path: str
    image: an.LoadedImage
    seg: an.Segmentation
    features: an.Features
    view_label: str = "front"


@dataclass
class Reconstruction:
    spec: dict
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    preview_overlay: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Analysis entry point
# ---------------------------------------------------------------------------

def analyse(paths: Sequence[str], labels: Optional[Sequence[str]] = None, sensitivity: float = 0.5) -> List[Source]:
    out = []
    for i, p in enumerate(paths):
        img = an.load_image(p)
        seg = an.segment(img, sensitivity)
        feats = an.compute_features(img, seg)
        lbl = (labels[i] if labels and i < len(labels) else ("front" if i == 0 else "auto"))
        out.append(Source(p, img, seg, feats, lbl))
    return out


def rank_categories(src: Source) -> List[Tuple[str, float]]:
    """Deterministic prior over categories from silhouette and line features."""
    f = src.features
    s: Dict[str, float] = {}
    straight = min(1.0, (f.vertical_lines + f.horizontal_lines) / 20.0)
    s["bed"] = 0.5 * straight + (0.4 if 1.2 < f.aspect < 3.5 else 0.0) + (0.2 if f.fill_ratio < 0.75 else 0.0)
    s["sofa"] = 0.3 * straight + (0.4 if 1.4 < f.aspect < 4.0 else 0.0) + (0.3 if f.fill_ratio > 0.55 else 0.0)
    s["table"] = 0.5 * straight + (0.3 if f.aspect > 0.9 else 0.0) + (0.3 if f.fill_ratio < 0.55 else 0.0)
    s["chair"] = 0.5 * straight + (0.4 if 0.35 < f.aspect < 1.2 else 0.0) + (0.2 if f.fill_ratio < 0.6 else 0.0)
    s["cabinet"] = 0.4 * straight + (0.4 if f.rectangularity > 0.8 else 0.0) + (0.2 if f.fill_ratio > 0.75 else 0.0)
    s["shelf"] = 0.4 * straight + (0.4 if f.aspect < 0.8 else 0.0) + (0.2 if f.rectangularity > 0.75 else 0.0)
    s["lamp"] = (0.6 if f.symmetry > 0.85 else 0.0) + (0.4 if f.aspect < 0.9 else 0.0)
    s["house"] = 0.4 * straight + (0.3 if f.solidity > 0.85 else 0.0)
    s["car"] = (0.4 if 1.6 < f.aspect < 4.5 else 0.0) + (0.3 if f.solidity > 0.8 else 0.0) + 0.2 * (1 - straight)
    s["box"] = (0.6 if f.rectangularity > 0.85 and f.solidity > 0.9 else 0.0) + 0.2 * straight
    s["revolve"] = (0.9 if f.symmetry > 0.9 else 0.0) + (0.2 if f.aspect < 1.4 else 0.0)
    s["extrude"] = 0.8 * f.drawing_likeness + (0.2 if f.hole_count > 0 else 0.0)
    s["inflate"] = 0.4 + 0.3 * (1 - straight)
    return sorted(s.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------
# Template strategy
# ---------------------------------------------------------------------------

def _lab(color) -> np.ndarray:
    c = (np.clip(np.asarray(color, dtype=np.float64), 0, 1) * 255).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(c, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float64)


def _specific_regions(rgb: np.ndarray, regions: Dict[str, np.ndarray], families: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Assign photo colour clusters to template materials, family-aware.

    Fitting error makes projected regions overlap neighbouring materials (the
    duvet region also covers the wooden rail). The pixels of all regions are
    clustered together in CIELAB with a low lightness weight (so shading of one
    surface stays in one cluster). Materials are processed from the largest
    region down; each takes the cluster most of its pixels vote for, except
    clusters already claimed by a material of a different family (wood vs
    fabric). The region is then restricted to pixels whose chroma matches the
    chosen cluster, keeping both lit and shaded pixels of that surface. The
    returned confidence is the share of the region that voted for that cluster.
    """
    if len(regions) < 2:
        return regions, {kk: 1.0 for kk in regions}
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    wts = np.array([0.35, 1.0, 1.0], np.float32)
    rng = np.random.default_rng(4)
    pooled = []
    for reg in regions.values():
        px = lab_img[reg]
        if len(px) > 15000:
            px = px[rng.choice(len(px), 15000, replace=False)]
        pooled.append(px)
    pool = np.concatenate(pooled) * wts
    k = int(min(6, 2 + len(regions)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.3)
    _, _, centers = cv2.kmeans(pool.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    families = families or {}
    votes_by_key = {}
    for key, reg in regions.items():
        px = lab_img[reg] * wts
        assign = np.argmin(np.linalg.norm(px[:, None, :] - centers[None], axis=2), axis=1)
        votes_by_key[key] = np.bincount(assign, minlength=k).astype(np.float64)
    fams = sorted({families.get(kk, kk) for kk in regions})
    density = np.zeros((len(fams), k))
    for fi, fam in enumerate(fams):
        members = [kk for kk in regions if families.get(kk, kk) == fam]
        total = sum(float(regions[kk].sum()) for kk in members)
        density[fi] = sum(votes_by_key[kk] for kk in members) / max(total, 1.0)
    allowed = density >= 0.7 * density.max(axis=0, keepdims=True)
    out = {}
    conf = {}
    for key, reg in regions.items():
        fi = fams.index(families.get(key, key))
        votes = votes_by_key[key].copy()
        votes[~allowed[fi]] = -1.0
        best = int(np.argmax(votes))
        if votes[best] <= 0:
            out[key] = reg
            conf[key] = 0.0
            continue
        chroma_c = centers[best][1:]
        chroma_d = np.linalg.norm(lab_img[..., 1:] - chroma_c, axis=2)
        light_d = np.abs(lab_img[..., 0] * wts[0] - centers[best][0]) / wts[0]
        sel = reg & (chroma_d < 10.0) & (light_d < 70)
        conf[key] = float(sel.sum() / max(reg.sum(), 1))
        out[key] = sel if sel.sum() >= 80 else reg
    return out, conf


def _materials_from_fit(template: Template, params, view: View, src: Source, target: FitTarget, work_dir: str) -> Tuple[Dict[str, MaterialSpec], List[str]]:
    """Measure each template material on the photo pixels its parts cover.

    Regions are eroded to stay away from boundaries; colours use the robust
    albedo estimate. A material whose colour matches a larger material of a
    different family (fabric reading as the wooden frame) is treated as
    contaminated by fitting error and borrows the colour of a confidently
    measured material of its own family, or the template default. Materials
    of one family with the same chroma but different brightness are taken to
    be the same surface in light and shadow and share the lit colour.
    """
    rgb = src.image.rgb
    h, w = rgb.shape[:2]
    v = scaled_view(view, w / target.width, w, h)
    lab, scene, keys, (xy, z, cam, bounds) = label_map(template, params, v, w, h)
    zones = src.seg.zones
    notes = []
    measured: Dict[str, dict] = {}
    slots = {s.key: s for s in template.materials}
    erode = max(3, int(round(min(h, w) / 160)))
    regions = {}
    for i, key in enumerate(keys):
        region = (lab == i) & (zones != 0)
        region = cv2.erode(region.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)).astype(bool)
        if key in slots and region.sum() >= 150:
            regions[key] = region
    families = {key: slots[key].default_kind for key in regions}
    picked, confidence = _specific_regions(rgb, regions, families)
    for key, region in picked.items():
        if region.sum() >= 80:
            measured[key] = an.region_stats(rgb, region)
            measured[key]["confidence"] = confidence.get(key, 0.0)
    decided: Dict[str, Tuple[str, tuple, str]] = {}
    for key, st in measured.items():
        slot = slots[key]
        kind = infer_kind(st["color"], st["texture"], st["coherence"], st["specular"], slot.candidates, prior=slot.default_kind) if len(slot.candidates) > 1 else slot.candidates[0]
        color = tuple(st["color_lit"] if kind in ("fabric", "painted", "plastic", "ceramic") else st["color"])
        decided[key] = (kind, color, "measured from photo")
    for key, st in measured.items():
        slot = slots[key]
        kind, color, source = decided[key]
        reason = None
        if st.get("confidence", 1.0) < 0.4:
            reason = f"only {st.get('confidence', 0):.0%} of its photo region agreed on one colour"
        for other, ost in measured.items():
            if other == key or slots[other].default_kind == slot.default_kind or ost["count"] <= st["count"]:
                continue
            if np.linalg.norm(_lab(decided[other][1])[1:] - _lab(color)[1:]) < 14:
                reason = f"its photo region overlapped '{slots[other].label}'"
        if reason:
            peers = [k for k in measured if k != key and slots[k].default_kind == slot.default_kind and decided[k][2].startswith("measured")]
            if peers:
                best = max(peers, key=lambda k: _lab(decided[k][1])[0])
                decided[key] = (decided[best][0], decided[best][1], f"same material as '{slots[best].label}' ({reason})")
            else:
                decided[key] = (slot.default_kind, slot.default_color, f"template default ({reason})")
    family_members: Dict[str, List[str]] = {}
    for key in measured:
        family_members.setdefault(slots[key].default_kind, []).append(key)
    for fam, members in family_members.items():
        for key in members:
            kind, color, source = decided[key]
            for other in members:
                ocolor = decided[other][1]
                if other != key and np.linalg.norm(_lab(ocolor)[1:] - _lab(color)[1:]) < 8 and _lab(ocolor)[0] > _lab(color)[0] + 8:
                    color = ocolor
                    source = f"{source}; lit colour taken from '{slots[other].label}' (same fabric in shadow)"
            decided[key] = (kind, color, source)
    mats: Dict[str, MaterialSpec] = {}
    for i, key in enumerate(keys):
        slot = slots.get(key)
        if slot is None:
            continue
        st = measured.get(key)
        if st is None:
            mats[key] = make_material(key, slot.default_kind, slot.default_color, quilted=slot.quilted, source="template default (not visible)", name=slot.label)
            continue
        kind, color, source = decided[key]
        mat = make_material(key, kind, color, quilted=slot.quilted, source=source, name=slot.label)
        if source.startswith("measured") and (kind in ("wood", "stone", "leather") or (kind == "fabric" and st["texture"] > 0.045)):
            ex = _extract_exemplar(scene, key, cam, v, rgb, lab, i, work_dir, w, h)
            if ex is not None:
                mat.exemplar, mat.exemplar_density = ex
        mats[key] = mat
        notes.append(f"{slot.label}: {kind}, colour {mat.hex_color} ({source}, {int(st['count'])} px)")
    return mats, notes


def _face_quads(prim: NativePrimitive):
    sx, sy, sz = [s / 2 for s in prim.size]
    corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    faces = {
        (0, 1): [5, 7, 3, 1], (0, -1): [0, 2, 6, 4], (1, 1): [2, 3, 7, 6], (1, -1): [0, 4, 5, 1],
        (2, 1): [1, 3, 2, 0], (2, -1): [4, 6, 7, 5],
    }
    world = corners @ prim.cframe[:3, :3].T + prim.cframe[:3, 3]
    out = []
    for (ax, sign), idx in faces.items():
        n = prim.cframe[:3, ax] * sign
        out.append((ax, n, world[idx]))
    return out


def _extract_exemplar(scene: Scene, key: str, cam, view: View, rgb, lab, lab_index, work_dir, w, h):
    best = None
    c = np.array([w / 2.0, h / 2.0])
    for part in scene.parts:
        if part.material != key:
            continue
        for prim in part.native:
            if prim.shape != "Block":
                continue
            for ax, n, quad in _face_quads(prim):
                center = quad.mean(0)
                to_cam = cam.position - center
                if np.dot(n, to_cam / np.linalg.norm(to_cam)) < 0.35:
                    continue
                xy, _, _ = cam.project(quad)
                xy = (xy - c) * view.scale + c + np.array([view.tx, view.ty])
                area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
                cx, cy = xy.mean(0).astype(int)
                if not (0 <= cx < w and 0 <= cy < h) or lab[cy, cx] != lab_index:
                    continue
                if best is None or area > best[0]:
                    best = (area, ax, quad, xy, part)
    if best is None or best[0] < 900:
        return None
    area, ax, quad, xy, part = best
    len_a = np.linalg.norm(quad[1] - quad[0])
    len_b = np.linalg.norm(quad[3] - quad[0])
    ppm = float(np.sqrt(area / max(len_a * len_b, 1e-6)))
    ow = int(np.clip(len_a * ppm, 24, 512))
    oh = int(np.clip(len_b * ppm, 24, 512))
    grain_dir = part.frame[:3, part.grain_axis]
    edge_a = (quad[1] - quad[0]) / max(len_a, 1e-9)
    patch = an.rectify_quad(rgb, xy, ow, oh)
    if abs(np.dot(edge_a, grain_dir)) < 0.7:
        patch = np.ascontiguousarray(np.rot90(patch))
    inset = max(2, int(min(patch.shape[:2]) * 0.06))
    patch = patch[inset:-inset, inset:-inset]
    if min(patch.shape[:2]) < 20:
        return None
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, f"exemplar_{key}.png")
    Image.fromarray(patch).save(path)
    return path, ppm


def reconstruct_template(src: Source, template_name: str, work_dir: str, progress: Progress = None, fit_budget: int = 420) -> Reconstruction:
    template = get_template(template_name)
    target = make_target(src.image, src.seg)
    if progress:
        progress(f"fitting {template.label.lower()} template", 0.1)
    fit = fit_template(template, target, max_evals=fit_budget)
    counts = count_repeats(template, fit, src.image, target, src.seg)
    params = dict(fit.params)
    for k, (n, conf) in counts.items():
        if fit.score < 0.95:
            params[k] = float(n)
    if progress:
        progress("measuring materials", 0.8)
    mats, mat_notes = _materials_from_fit(template, params, fit.view, src, target, work_dir)
    quality = float(np.clip(1.25 - fit.score * 0.75, 0.0, 1.0))
    spec = {
        "strategy": "template",
        "template": template.name,
        "category": template.name,
        "params": {k: round(float(v), 4) for k, v in params.items()},
        "view": fit.view.to_dict(),
        "fit": {"score": round(fit.score, 4), "components": {k: round(v, 4) for k, v in fit.components.items()}, "evaluations": fit.evaluations,
                "counts": {k: [int(n), round(float(c), 3)] for k, (n, c) in counts.items()}},
        "materials": {k: m.to_dict() for k, m in mats.items()},
        "confidence": round(quality * (0.6 + 0.4 * src.seg.confidence), 3),
        "notes": mat_notes,
        "photo_projection": template.name == "box",
        "fit_resolution": [target.width, target.height],
    }
    h, w = src.image.rgb.shape[:2]
    v = scaled_view(fit.view, w / target.width, w, h)
    lab, *_ = label_map(template, params, v, w, h)
    overlay = src.image.rgb.astype(np.float32).copy()
    msk = lab >= 0
    overlay[msk] = overlay[msk] * 0.5 + np.array([240, 150, 40]) * 0.5
    return Reconstruction(spec, {"model_mask": msk.astype(np.uint8)}, overlay.astype(np.uint8))


# ---------------------------------------------------------------------------
# Generic single-view strategies
# ---------------------------------------------------------------------------

def _px_to_studs(src: Source, height_studs: float) -> float:
    x0, y0, x1, y1 = src.features.bbox
    return height_studs / max(y1 - y0 + 1, 1)


def _front_camera(src: Source, k: float, origin_px: Tuple[float, float]) -> dict:
    """Orthographic front camera parameters mapping object studs to image px."""
    return {"type": "ortho", "px_per_stud": 1.0 / k, "origin_px": [float(origin_px[0]), float(origin_px[1])], "image_size": [int(src.image.rgb.shape[1]), int(src.image.rgb.shape[0])]}


def reconstruct_revolve(src: Source, height_studs: float) -> Reconstruction:
    mask = src.seg.filled
    x0, y0, x1, y1 = src.features.bbox
    axis = src.features.symmetry_axis
    k = _px_to_studs(src, height_studs)
    rows = np.arange(y0, y1 + 1)
    radius = []
    for y in rows:
        xs = np.flatnonzero(mask[y])
        if len(xs) == 0:
            radius.append(0.0)
            continue
        radius.append(max(axis - xs.min(), xs.max() - axis))
    r = np.asarray(radius, dtype=np.float64)
    r = cv2.GaussianBlur(r.reshape(-1, 1).astype(np.float32), (1, 9), 0).ravel()
    y_studs = (y1 - rows) * k
    order = np.argsort(y_studs)
    profile = np.stack([r[order] * k, y_studs[order]], axis=1)
    spec = {
        "strategy": "revolve", "category": "revolve",
        "dimensions": [float(2 * r.max() * k), float(height_studs), float(2 * r.max() * k)],
        "camera": _front_camera(src, k, (axis, y1)),
        "confidence": round(0.5 + 0.5 * src.features.symmetry * src.seg.confidence, 3),
        "notes": ["Rotationally symmetric object: the profile was revolved around the vertical axis; the back is a mirror of the front photo."],
        "photo_projection": True,
    }
    return Reconstruction(spec, {"profile": profile})


def _contours(mask: np.ndarray):
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ReconstructionError("no contour found")
    hier = hier[0]
    outer_i = max((i for i in range(len(cnts)) if hier[i][3] < 0), key=lambda i: cv2.contourArea(cnts[i]))
    holes = [cnts[i] for i in range(len(cnts)) if hier[i][3] == outer_i and cv2.contourArea(cnts[i]) > 0.002 * cv2.contourArea(cnts[outer_i])]
    return cnts[outer_i].reshape(-1, 2).astype(np.float64), [h.reshape(-1, 2).astype(np.float64) for h in holes]


def reconstruct_extrude(src: Source, height_studs: float, thickness_ratio: float = 0.12) -> Reconstruction:
    outer, holes = _contours(src.seg.mask)
    x0, y0, x1, y1 = src.features.bbox
    k = _px_to_studs(src, height_studs)
    cx = (x0 + x1) / 2.0
    width = (x1 - x0 + 1) * k
    spec = {
        "strategy": "extrude", "category": "extrude",
        "dimensions": [float(width), float(height_studs), float(max(width, height_studs) * thickness_ratio)],
        "thickness_ratio": thickness_ratio,
        "camera": _front_camera(src, k, (cx, y1)),
        "confidence": round(0.45 + 0.4 * src.seg.confidence, 3),
        "notes": ["Flat object: the outline was extruded; front and back use the photo, the edge colour is sampled from the outline."],
        "photo_projection": True,
    }
    arrays = {"outer": outer}
    for i, h in enumerate(holes[:16]):
        arrays[f"hole{i}"] = h
    return Reconstruction(spec, arrays)


def reconstruct_inflate(src: Source, height_studs: float, depth_ratio: float = 0.55) -> Reconstruction:
    x0, y0, x1, y1 = src.features.bbox
    k = _px_to_studs(src, height_studs)
    cx = (x0 + x1) / 2.0
    width = (x1 - x0 + 1) * k
    mask = src.seg.filled[y0:y1 + 1, x0:x1 + 1]
    spec = {
        "strategy": "inflate", "category": "inflate",
        "dimensions": [float(width), float(height_studs), float(min(width, height_studs) * depth_ratio)],
        "depth_ratio": depth_ratio,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "camera": _front_camera(src, k, (cx, y1)),
        "confidence": round(0.35 + 0.4 * src.seg.confidence, 3),
        "notes": ["Organic object: the silhouette was inflated with a rounded thickness profile. Unseen geometry (the back) is estimated as a mirror of the front."],
        "photo_projection": True,
    }
    return Reconstruction(spec, {"mask": mask.astype(np.uint8)})


# ---------------------------------------------------------------------------
# Multi-view strategies
# ---------------------------------------------------------------------------

VIEW_DIRECTIONS = {
    "front": (0.0, 0.0), "right": (90.0, 0.0), "side": (90.0, 0.0), "back": (180.0, 0.0),
    "left": (-90.0, 0.0), "top": (0.0, 89.0),
}


def reconstruct_visual_hull(sources: Sequence[Source], height_studs: float, resolution: int = 112) -> Reconstruction:
    """Carve a voxel volume with orthographic reference-view silhouettes."""
    views = []
    for s in sources:
        lbl = s.view_label if s.view_label in VIEW_DIRECTIONS else None
        if lbl is None:
            raise ReconstructionError("reference-view mode needs every image labelled front/side/back/left/top")
        views.append((lbl, s))
    front = next((s for l, s in views if l == "front"), None)
    if front is None:
        raise ReconstructionError("a front view is required")
    fx0, fy0, fx1, fy1 = front.features.bbox
    k = height_studs / max(fy1 - fy0 + 1, 1)
    width = (fx1 - fx0 + 1) * k
    depth = None
    for l, s in views:
        if l in ("right", "side", "left"):
            x0, y0, x1, y1 = s.features.bbox
            depth = (x1 - x0 + 1) * height_studs / max(y1 - y0 + 1, 1)
        elif l == "top" and depth is None:
            x0, y0, x1, y1 = s.features.bbox
            depth = (y1 - y0 + 1) * width / max(x1 - x0 + 1, 1)
    if depth is None:
        depth = 0.6 * width
    dims = np.array([width, height_studs, depth])
    n = resolution
    step = dims.max() / n
    gx = np.arange(-dims[0] / 2, dims[0] / 2 + step, step)
    gy = np.arange(0, dims[1] + step, step)
    gz = np.arange(-dims[2] / 2, dims[2] / 2 + step, step)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    occ = np.ones(X.shape, bool)
    cams = []
    for lbl, s in views:
        x0, y0, x1, y1 = s.features.bbox
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if lbl == "front":
            u = (-X / dims[0] + 0.5) * bw + x0
            v = y1 - Y / dims[1] * bh
        elif lbl == "back":
            u = (X / dims[0] + 0.5) * bw + x0
            v = y1 - Y / dims[1] * bh
        elif lbl in ("right", "side"):
            u = (-Z / dims[2] + 0.5) * bw + x0
            v = y1 - Y / dims[1] * bh
        elif lbl == "left":
            u = (Z / dims[2] + 0.5) * bw + x0
            v = y1 - Y / dims[1] * bh
        else:
            u = (-X / dims[0] + 0.5) * bw + x0
            v = (Z / dims[2] + 0.5) * bh + y0
        mask = cv2.dilate(s.seg.filled.astype(np.uint8), np.ones((3, 3), np.uint8))
        ui = np.clip(np.round(u).astype(np.int64), 0, mask.shape[1] - 1)
        vi = np.clip(np.round(v).astype(np.int64), 0, mask.shape[0] - 1)
        occ &= mask[vi, ui] > 0
        cams.append({"label": lbl, "bbox": [int(x0), int(y0), int(x1), int(y1)], "path_index": sources.index(s)})
    if occ.sum() < 20:
        raise ReconstructionError("silhouettes do not intersect; check the view labels")
    spec = {
        "strategy": "visual_hull", "category": "visual_hull",
        "dimensions": [float(d) for d in dims], "voxel_step": float(step),
        "grid_origin": [float(gx[0]), float(gy[0]), float(gz[0])],
        "views": cams,
        "confidence": round(0.5 + 0.1 * len(views), 3),
        "notes": [f"Visual hull carved from {len(views)} labelled views ({', '.join(l for l, _ in views)}). Concavities that no silhouette reveals cannot be recovered."],
        "photo_projection": True,
    }
    return Reconstruction(spec, {"occupancy": occ.astype(np.uint8)})


def _intrinsics(img: an.LoadedImage) -> np.ndarray:
    h, w = img.rgb.shape[:2]
    if img.exif_focal_35mm:
        f = img.exif_focal_35mm / 36.0 * max(w, h)
    else:
        f = 1.15 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def estimate_turntable(sources: Sequence[Source]) -> Tuple[List[float], List[str]]:
    """Recover relative yaw between consecutive photos with SIFT + essential matrix."""
    notes = []
    sift = cv2.SIFT_create(nfeatures=4000)
    feats = []
    for s in sources:
        gray = cv2.cvtColor(s.image.rgb, cv2.COLOR_RGB2GRAY)
        m = cv2.dilate(s.seg.filled.astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
        kp, des = sift.detectAndCompute(gray, m)
        feats.append((kp, des))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    yaws = []
    ok = 0
    for i in range(len(sources) - 1):
        kp1, d1 = feats[i]
        kp2, d2 = feats[i + 1]
        if d1 is None or d2 is None or len(kp1) < 30 or len(kp2) < 30:
            yaws.append(None)
            continue
        good = [m for m, n in matcher.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 25:
            yaws.append(None)
            continue
        p1 = np.float64([kp1[m.queryIdx].pt for m in good])
        p2 = np.float64([kp2[m.trainIdx].pt for m in good])
        K = _intrinsics(sources[i].image)
        E, inl = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.5)
        if E is None or inl is None or inl.sum() < 20:
            yaws.append(None)
            continue
        _, R, t, _ = cv2.recoverPose(E[:3], p1, p2, K, mask=inl)
        rvec, _ = cv2.Rodrigues(R)
        angle = float(np.degrees(np.linalg.norm(rvec)))
        axis = rvec.ravel() / max(np.linalg.norm(rvec), 1e-9)
        if abs(axis[1]) < 0.7:
            notes.append(f"Pair {i + 1}-{i + 2}: rotation axis is not vertical; the object may not have been circled.")
        yaws.append(angle * (1 if axis[1] >= 0 else -1))
        ok += 1
    if ok < max(1, (len(sources) - 1) * 0.5):
        raise ReconstructionError(f"only {ok} of {len(sources) - 1} photo pairs could be matched; the photo set lacks overlap or texture")
    valid = [abs(y) for y in yaws if y is not None]
    mean_step = float(np.median(valid))
    angles = [0.0]
    for y in yaws:
        angles.append(angles[-1] + (abs(y) if y is not None else mean_step))
    notes.append(f"Recovered yaw steps: {', '.join('?' if y is None else f'{abs(y):.0f}' for y in yaws)} degrees.")
    return angles, notes


def reconstruct_turntable(sources: Sequence[Source], height_studs: float, resolution: int = 96) -> Reconstruction:
    angles, notes = estimate_turntable(sources)
    first = sources[0]
    x0, y0, x1, y1 = first.features.bbox
    aspect = (x1 - x0 + 1) / max(y1 - y0 + 1, 1)
    half = max(aspect, 1.0) * height_studs * 0.6
    n = resolution
    gx = np.linspace(-half, half, n)
    gy = np.linspace(0, height_studs * 1.05, n)
    gz = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    votes = np.zeros(len(pts), np.int32)
    cams = []
    for s, ang in zip(sources, angles):
        h, w = s.image.rgb.shape[:2]
        K = _intrinsics(s.image)
        bx0, by0, bx1, by1 = s.features.bbox
        obj_h_px = by1 - by0 + 1
        dist = K[0, 0] * height_studs / max(obj_h_px, 1) * 1.0
        cam = render.orbit_camera((0, height_studs / 2, 0), dist, ang, 10.0, w, h, 2 * np.degrees(np.arctan(h / 2 / K[1, 1])))
        xy, d, _ = cam.project(pts)
        shift = np.array([(bx0 + bx1) / 2 - w / 2, (by0 + by1) / 2 - h / 2])
        xy = xy + shift
        m = cv2.dilate(s.seg.filled.astype(np.uint8), np.ones((5, 5), np.uint8))
        xi = np.clip(xy[:, 0].astype(np.int64), 0, w - 1)
        yi = np.clip(xy[:, 1].astype(np.int64), 0, h - 1)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        votes += (inside & (m[yi, xi] > 0)).astype(np.int32)
        cams.append({"azimuth": float(ang), "distance": float(dist), "shift": shift.tolist(), "path_index": sources.index(s)})
    need = max(len(sources) - 1, 2)
    occ = (votes >= need).reshape(X.shape)
    if occ.sum() < 30:
        raise ReconstructionError("photo silhouettes did not agree on a common volume")
    spec = {
        "strategy": "turntable", "category": "turntable",
        "dimensions": [float(2 * half), float(height_studs * 1.05), float(2 * half)],
        "grid_origin": [float(gx[0]), float(gy[0]), float(gz[0])], "voxel_step": float(gx[1] - gx[0]),
        "views": cams, "confidence": 0.45,
        "notes": notes + ["Photo-set reconstruction assumes the camera circled the object at a similar height and distance."],
        "photo_projection": True,
    }
    return Reconstruction(spec, {"occupancy": occ.astype(np.uint8)})


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def reconstruct(paths: Sequence[str], work_dir: str, mode: str = "auto", category: str = "auto", view_labels: Optional[Sequence[str]] = None,
                target_height: Optional[float] = None, sensitivity: float = 0.5, progress: Progress = None) -> Tuple[Reconstruction, List[Source]]:
    def report(msg, frac):
        if progress:
            progress(msg, frac)

    report("analysing image" + ("s" if len(paths) > 1 else ""), 0.02)
    sources = analyse(paths, view_labels, sensitivity)
    notes: List[str] = []
    for s in sources:
        notes.extend(s.seg.notes)
    if len(sources) > 1 and mode in ("auto", "multi"):
        labelled = view_labels and all(l in VIEW_DIRECTIONS for l in view_labels[: len(sources)])
        h = target_height or 6.0
        try:
            if labelled:
                report("carving visual hull from reference views", 0.3)
                rec = reconstruct_visual_hull(sources, h)
            else:
                report("estimating camera motion between photos", 0.3)
                rec = reconstruct_turntable(sources, h)
            rec.spec["notes"] = notes + rec.spec["notes"]
            rec.spec["mode"] = "B"
            return rec, sources
        except ReconstructionError as exc:
            notes.append(f"Multi-image reconstruction was not possible ({exc}); the first image was reconstructed on its own.")
    src = sources[0]
    ranking = rank_categories(src)
    chosen = None
    alternatives = []
    cat = (category or "auto").lower()
    if cat != "auto":
        chosen = CATEGORY_TEMPLATES.get(cat, cat)
        if chosen not in TEMPLATES and chosen not in GENERIC:
            notes.append(f"Unknown category '{category}', detecting automatically instead.")
            chosen = None
    if chosen is None and mode == "single":
        chosen = "inflate"
    if chosen is None:
        report("comparing object templates", 0.08)
        target = make_target(src.image, src.seg)
        candidates = [c for c, _ in ranking if c in TEMPLATES][:5]
        scored = []
        for c in candidates:
            res = fit_template(get_template(c), target, coarse_only=True)
            scored.append((res.score, c))
            alternatives.append({"template": c, "coarse_score": round(res.score, 4)})
        scored.sort()
        best_score, best = scored[0]
        sym = src.features.symmetry
        if best_score < 1.3:
            chosen = best
        elif sym > 0.9:
            chosen = "revolve"
        elif src.features.drawing_likeness > 0.55:
            chosen = "extrude"
        else:
            chosen = "inflate"
        notes.append(f"Detected category: {chosen} (template scores: " + ", ".join(f"{c} {s:.2f}" for s, c in scored) + ").")
    report(f"reconstructing as {chosen}", 0.15)
    if chosen in TEMPLATES:
        def sub(msg, f):
            report(msg, 0.15 + 0.6 * f)
        rec = reconstruct_template(src, chosen, work_dir, sub)
    else:
        h = target_height or m2s(TYPICAL_HEIGHT_M.get(chosen, 0.8))
        if chosen == "revolve":
            rec = reconstruct_revolve(src, h)
        elif chosen == "extrude":
            rec = reconstruct_extrude(src, h)
        else:
            rec = reconstruct_inflate(src, h)
    rec.spec["notes"] = notes + rec.spec.get("notes", [])
    rec.spec["alternatives"] = alternatives
    rec.spec["mode"] = "C" if chosen in TEMPLATES else "A"
    rec.spec["segmentation"] = {"method": src.seg.method, "confidence": round(src.seg.confidence, 3)}
    rec.spec["features"] = src.features.to_dict()
    return rec, sources


# ---------------------------------------------------------------------------
# Building geometry from a spec (used for every edit and LOD)
# ---------------------------------------------------------------------------

def _voxel_mesh(occ: np.ndarray, origin, step: float, smooth_iters: int = 8) -> g.Mesh:
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    vol = gaussian_filter(np.pad(occ.astype(np.float32), 2), 0.7)
    verts, faces, _, _ = marching_cubes(vol, 0.5)
    verts = (verts - 2.0) * step + np.asarray(origin) + step * 0.5
    mesh = g.Mesh(verts, faces.astype(np.int64))
    if mesh.signed_volume() < 0:
        mesh.faces = mesh.faces[:, ::-1].copy()
    mesh = taubin_smooth(mesh, smooth_iters)
    mesh.normals = g.vertex_normals(mesh)
    return mesh


def taubin_smooth(mesh: g.Mesh, iterations: int = 8, lam: float = 0.5, mu: float = -0.53) -> g.Mesh:
    from scipy.sparse import coo_matrix, diags

    n = len(mesh.vertices)
    f = mesh.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    norm = diags(1.0 / np.maximum(deg, 1)) @ adj
    v = mesh.vertices.copy()
    for _ in range(iterations):
        v = v + lam * (norm @ v - v)
        v = v + mu * (norm @ v - v)
    out = mesh.copy()
    out.vertices = v
    return out


def _inflate_volume(mask: np.ndarray, dims, depth: float, res: int) -> Tuple[np.ndarray, list, float]:
    """Occupancy volume of a silhouette inflated by sqrt(distance) (Teddy-like).

    The mask is the cropped front view (rows top-down, columns left-right). Image
    left is the object's right (+X) because the camera looks at the front (-Z).
    """
    step = max(dims[0], dims[1]) / res
    nx = max(4, int(round(dims[0] / step)))
    ny = max(4, int(round(dims[1] / step)))
    m = cv2.resize(mask.astype(np.uint8) * 255, (nx, ny), interpolation=cv2.INTER_AREA) > 127
    dt = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    if dt.max() <= 0:
        raise ReconstructionError("empty silhouette")
    half_t = np.sqrt(np.clip(dt / dt.max(), 0, 1)) * depth / 2
    nz = max(4, int(round(depth / step)) + 2)
    zc = (np.arange(nz) - (nz - 1) / 2) * step
    rc = m[:, :, None] & (np.abs(zc[None, None, :]) <= half_t[:, :, None])
    vol = np.transpose(rc[::-1, ::-1, :], (1, 0, 2))
    origin = [-dims[0] / 2, 0.0, float(zc[0])]
    return vol, origin, step


def build_generic_scene(spec: dict, arrays: Dict[str, np.ndarray], lod: LOD) -> Scene:
    strategy = spec["strategy"]
    mats = {k: MaterialSpec.from_dict(v) for k, v in spec.get("materials", {}).items()}
    if "surface" not in mats:
        mats["surface"] = make_material("surface", "photo", (0.7, 0.7, 0.7), name="Photo surface", source="photo projection")
    dims = spec["dimensions"]
    if strategy == "revolve":
        prof = np.asarray(arrays["profile"], dtype=np.float64)
        n = int(np.clip(12 + lod.q * 52, 12, 64))
        idx = np.linspace(0, len(prof) - 1, n).round().astype(int)
        p = prof[idx]
        p = np.vstack([[0.0, p[0, 1]], p, [0.0, p[-1, 1]]])
        mesh = g.lathe(p, max(8, lod.round_segments + 8), smooth_profile=True, hard_angle_deg=60.0, name="Body")
        native = [NativePrimitive("Cylinder", (max(y1 - y0, 0.05), 2 * max(r, 0.03), 2 * max(r, 0.03)), g.translation((0, (y0 + y1) / 2, 0)) @ g.rotation([0, 0, 1], np.pi / 2), "surface", f"Body{i + 1}")
                  for i, (r, y0, y1) in enumerate(_profile_bands(p, 6))]
        parts = [Part("Body", mesh, "surface", np.eye(4), 1, "v", 1, True, native)]
    elif strategy == "extrude":
        outer = np.asarray(arrays["outer"], dtype=np.float64)
        holes = [np.asarray(arrays[k], dtype=np.float64) for k in sorted(arrays) if k.startswith("hole")]
        cam = spec["camera"]
        k = 1.0 / cam["px_per_stud"]
        ox, oy = cam["origin_px"]
        eps = float(np.clip(2.5 - lod.q * 2.2, 0.3, 2.5))

        def to_studs(c):
            c2 = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(np.float64)
            return np.stack([-(c2[:, 0] - ox) * k, (oy - c2[:, 1]) * k], axis=1)

        o = to_studs(outer)
        hs = [to_studs(h) for h in holes if len(h) >= 8]
        depth = dims[2]
        mesh = g.extrude_polygon(o, [h for h in hs if len(h) >= 3], depth, depth * 0.08 * lod.bevel_scale, "Body")
        lo, hi = mesh.bounds()
        native = [NativePrimitive("Block", tuple((hi - lo).tolist()), g.translation((lo + hi) / 2), "surface", "Body")]
        parts = [Part("Body", mesh, "surface", np.eye(4), 0, "u", 1, True, native)]
    elif strategy == "inflate":
        mask = np.asarray(arrays["mask"]) > 0
        res = int(np.clip(40 + lod.q * 110, 40, 150))
        vol, origin, step = _inflate_volume(mask, dims, dims[2], res)
        mesh = _voxel_mesh(vol, origin, step, 6)
        parts = [Part("Body", mesh, "surface", np.eye(4), 0, "u", 1, True, _voxel_natives(vol, origin, step, "surface"))]
    elif strategy in ("visual_hull", "turntable"):
        occ = np.asarray(arrays["occupancy"]) > 0
        factor = 1 if lod.q > 0.6 else 2
        if factor > 1:
            occ = occ[::factor, ::factor, ::factor]
        mesh = _voxel_mesh(occ, spec["grid_origin"], spec["voxel_step"] * factor, 6)
        parts = [Part("Body", mesh, "surface", np.eye(4), 0, "u", 1, True, _voxel_natives(occ, spec["grid_origin"], spec["voxel_step"] * factor, "surface"))]
    else:
        raise ReconstructionError(f"unknown strategy {strategy}")
    for p in parts:
        _assign_chart_uvs(p.mesh)
    return Scene(spec.get("name", "Model"), parts, mats, {"strategy": strategy})


def _profile_bands(profile: np.ndarray, n: int):
    y_lo, y_hi = profile[:, 1].min(), profile[:, 1].max()
    edges = np.linspace(y_lo, y_hi, n + 1)
    out = []
    order = np.argsort(profile[:, 1])
    py, pr = profile[order, 1], profile[order, 0]
    for a, b in zip(edges[:-1], edges[1:]):
        ys = np.linspace(a, b, 6)
        out.append((float(np.interp(ys, py, pr).mean()), float(a), float(b)))
    return out


def _voxel_natives(occ: np.ndarray, origin, step: float, mat: str, max_parts: int = 120) -> List[NativePrimitive]:
    """Greedy box decomposition of an occupancy grid into Roblox blocks."""
    occ = occ.copy()
    factor = 1
    while occ.sum() > 0 and _estimate_boxes(occ) > max_parts and factor < 8:
        factor *= 2
        sh = [max(1, s // 2) for s in occ.shape]
        occ = occ[: sh[0] * 2, : sh[1] * 2, : sh[2] * 2].reshape(sh[0], 2, sh[1], 2, sh[2], 2).mean(axis=(1, 3, 5)) >= 0.5
    st = step * factor
    boxes = []
    rem = occ.copy()
    while rem.any() and len(boxes) < max_parts:
        i, j, k = np.argwhere(rem)[0]
        i1 = i
        while i1 + 1 < rem.shape[0] and rem[i1 + 1, j, k]:
            i1 += 1
        j1 = j
        while j1 + 1 < rem.shape[1] and rem[i:i1 + 1, j1 + 1, k].all():
            j1 += 1
        k1 = k
        while k1 + 1 < rem.shape[2] and rem[i:i1 + 1, j:j1 + 1, k1 + 1].all():
            k1 += 1
        rem[i:i1 + 1, j:j1 + 1, k:k1 + 1] = False
        lo = np.asarray(origin) + np.array([i, j, k]) * st
        size = np.array([i1 - i + 1, j1 - j + 1, k1 - k + 1]) * st
        boxes.append(NativePrimitive("Block", tuple(size.tolist()), g.translation(lo + size / 2), mat, f"Voxel{len(boxes) + 1}"))
    return boxes


def _estimate_boxes(occ: np.ndarray) -> int:
    return int(min(occ.sum(), 10_000) ** 0.75)


def _assign_chart_uvs(mesh: g.Mesh) -> None:
    """Cube-projection charts: faces grouped by dominant normal axis, split at chart borders."""
    fn = mesh.face_normals()
    axis = np.argmax(np.abs(fn), axis=1)
    sign = np.sign(fn[np.arange(len(fn)), axis]).astype(int)
    chart = axis * 2 + (sign > 0)
    corners = mesh.faces.reshape(-1)
    face_chart = np.repeat(chart, 3)
    key = corners * 6 + face_chart
    uniq, inverse = np.unique(key, return_inverse=True)
    vid = uniq // 6
    cid = uniq % 6
    verts = mesh.vertices[vid]
    norms = mesh.normals[vid] if mesh.normals is not None else None
    ax = cid // 2
    uv = np.zeros((len(vid), 2))
    for a in range(3):
        sel = ax == a
        others = [i for i in range(3) if i != a]
        uv[sel] = verts[sel][:, others]
    mesh.vertices = verts
    mesh.normals = norms
    mesh.faces = inverse.reshape(-1, 3)
    mesh.uvs = uv - uv.min(0)


def photo_projection_for(spec: dict, sources: Sequence[Source], scene: Scene) -> Optional[PhotoProjection]:
    """Camera(s) that map model space onto the source photo(s)."""
    if not spec.get("photo_projection"):
        return None
    strategy = spec["strategy"]
    if strategy in ("revolve", "extrude", "inflate"):
        src = sources[0]
        cam_spec = spec["camera"]
        w, h = cam_spec["image_size"]
        ppm = cam_spec["px_per_stud"]
        ox, oy = cam_spec["origin_px"]
        view = render.look_at((0, 0, -50), (0, 0, 0))
        cam = render.Camera(view, w, h, ortho_scale=ppm, cx=ox, cy=oy)
        img = srgb_to_linear(src.image.rgb.astype(np.float64) / 255.0)
        return PhotoProjection(img, cam, src.seg.mask.astype(np.uint8), mirror_axis=2, mirror_center=0.0)
    if strategy == "template":
        src = sources[0]
        template = get_template(spec["template"])
        h, w = src.image.rgb.shape[:2]
        fw, fh = spec.get("fit_resolution", [320, 175])
        v = scaled_view(View.from_dict(spec["view"]), w / fw, w, h)
        cam = _view_camera(template.build(spec["params"], LOD(0.0, fast=True)).bounds(), v, w, h)
        img = srgb_to_linear(src.image.rgb.astype(np.float64) / 255.0)
        return PhotoProjection(img, cam, None, mirror_axis=0, mirror_center=0.0)
    if strategy == "visual_hull":
        views = []
        dims = spec["dimensions"]
        for cam_spec in spec["views"]:
            s = sources[cam_spec["path_index"]]
            x0, y0, x1, y1 = cam_spec["bbox"]
            lbl = cam_spec["label"]
            az, el = VIEW_DIRECTIONS[lbl]
            bh = y1 - y0 + 1
            ppm = bh / dims[1] if lbl != "top" else (x1 - x0 + 1) / dims[0]
            eye_dir = np.array([np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el)), -np.cos(np.radians(az)) * np.cos(np.radians(el))])
            center = np.array([0.0, dims[1] / 2, 0.0])
            vm = render.look_at(center + eye_dir * 100, center, (0, 1, 0) if lbl != "top" else (0, 0, 1))
            img = srgb_to_linear(s.image.rgb.astype(np.float64) / 255.0)
            h, w = s.image.rgb.shape[:2]
            cam = render.Camera(vm, w, h, ortho_scale=ppm, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2)
            views.append(PhotoProjection(img, cam, s.seg.mask.astype(np.uint8), mirror_axis=None))
        first = views[0]
        first.extra = views[1:]
        if not any(c["label"] == "back" for c in spec["views"]):
            first.mirror_axis = 2
        return first
    if strategy == "turntable":
        views = []
        for cam_spec in spec["views"]:
            s = sources[cam_spec["path_index"]]
            h, w = s.image.rgb.shape[:2]
            K = _intrinsics(s.image)
            fov = 2 * np.degrees(np.arctan(h / 2 / K[1, 1]))
            cam = render.orbit_camera((0, spec["dimensions"][1] / 2 / 1.05, 0), cam_spec["distance"], cam_spec["azimuth"], 10.0, w, h, fov)
            cam.cx = w / 2 + cam_spec["shift"][0]
            cam.cy = h / 2 + cam_spec["shift"][1]
            img = srgb_to_linear(s.image.rgb.astype(np.float64) / 255.0)
            views.append(PhotoProjection(img, cam, s.seg.mask.astype(np.uint8), mirror_axis=None))
        first = views[0]
        first.extra = views[1:]
        return first
    return None


def _view_camera(bounds: np.ndarray, view: View, w: int, h: int) -> render.Camera:
    """Fold the 2D similarity of a fitted view into a pinhole camera."""
    center = 0.5 * (bounds[0] + bounds[1])
    radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.5, 1e-3)
    dist = radius / np.tan(np.radians(view.fov) / 2.0) * 1.15
    cam = render.orbit_camera(center, dist, view.azimuth, view.elevation, w, h, view.fov)
    cam.focal *= view.scale
    cam.cx = w / 2.0 + view.tx
    cam.cy = h / 2.0 + view.ty
    return cam


def _json_safe(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def save_reconstruction(rec: Reconstruction, folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "spec.json"), "w", encoding="utf-8") as fh:
        json.dump(rec.spec, fh, indent=2, default=_json_safe)
    if rec.arrays:
        np.savez_compressed(os.path.join(folder, "spec_data.npz"), **rec.arrays)
    if rec.preview_overlay is not None:
        Image.fromarray(rec.preview_overlay).save(os.path.join(folder, "fit_overlay.png"))


def load_arrays(folder: str) -> Dict[str, np.ndarray]:
    path = os.path.join(folder, "spec_data.npz")
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
