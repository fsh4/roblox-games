"""Box body with attached parts, measured from one photo.

Many props are a box with things on it: a parcel with a cake on top, a crate
with a lamp, a chest with a lock. No object category is guessed. The body is
recognised by its outline alone (a convex polygon of four to seven straight
sides), then a cuboid and a perspective camera are fitted to that outline by
analysis-by-synthesis (``engine.fitting``). Parts sticking out of the body in
the box outline are found after a first fit; the box is fitted again
without them, and each one is grown over the box faces by colour (an object
standing on the top face hides part of that face), inflated from its own
outline and placed in 3D where its foot meets the box: a ray from the fitted camera
through the foot pixel is intersected with the box, and the part is stood
upright there, facing the camera, at the scale the camera gives at that
distance.

Texturing: the visible faces of the box take the photo through the fitted
camera; faces and spots the photo does not show take the face opposite them
(the same photo seen through the box turned half way round, which keeps
writing readable), and only there. Each part keeps its own front-on photo projection with a synthesised
back, like any inflated object.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import analyzer as an
from . import geometry as g
from . import render, sculpt
from .fitting import fit_template, label_map, make_target, scaled_view
from .materials import make_material, srgb_to_linear
from .procedural import LOD, get_template

MAX_SPILL = 0.06
MAX_MISSED = 0.2
MIN_SOLIDITY = 0.93
MIN_STRAIGHT = 0.4
PART_PREFIX = "part{}_"


class _Zones:
    def __init__(self, zones: np.ndarray):
        self.zones = zones


def boxy(body: np.ndarray) -> bool:
    """True when the body outline looks like a box seen from any side: convex, four to seven straight sides."""
    cnts, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    if hull <= 0 or area / hull < MIN_SOLIDITY:
        return False
    sides = len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True))
    return 4 <= sides <= 7 and an.outline_straightness(c) >= MIN_STRAIGHT


def camera_ray(cam: render.Camera, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
    """World origin and unit direction of the ray through pixel (x, y)."""
    inv = np.linalg.inv(cam.view)
    cx = cam.cx or cam.width / 2.0
    cy = cam.cy or cam.height / 2.0
    if cam.is_ortho:
        local = np.array([(x - cx) / cam.ortho_scale, -(y - cy) / cam.ortho_scale, 0.0, 1.0])
        return (inv @ local)[:3], -inv[:3, 2] / np.linalg.norm(inv[:3, 2])
    d = inv[:3, :3] @ np.array([(x - cx) / cam.focal, -(y - cy) / cam.focal, -1.0])
    return cam.position, d / np.linalg.norm(d)


def ray_box(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Optional[np.ndarray]:
    """First point where the ray enters the axis-aligned box [lo, hi]."""
    inv = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    t0 = (lo - origin) * inv
    t1 = (hi - origin) * inv
    near = np.max(np.minimum(t0, t1))
    far = np.min(np.maximum(t0, t1))
    if far < max(near, 0.0):
        return None
    return origin + direction * max(near, 0.0)


def _fit_box(src, zones: np.ndarray, fit_budget: int):
    target = make_target(src.image, _Zones(zones))
    fit = fit_template(get_template("box"), target, max_evals=fit_budget)
    return fit, target


def _box_mask(params: dict, view, w: int, h: int) -> np.ndarray:
    lab, *_ = label_map(get_template("box"), params, view, w, h)
    return lab >= 0


def _parts_on_box(rgb: np.ndarray, obj: np.ndarray, box: np.ndarray, min_area: float) -> Tuple[np.ndarray, int]:
    """Objects standing on the box: what sticks out of the box outline, followed down over the box faces.

    A part standing on the top face hides some of that face in the photo, so
    only its upper piece sticks out of the box outline. That piece seeds a
    graph cut (GrabCut) in a window below and around it: the piece is
    certain object, box pixels far from it are certain box, and the colour
    models decide the rest, so the whole part is cut out even where its
    colours are close to the box's. Thin slivers along the box edges (fit
    error) are not parts.
    """
    h, w = obj.shape
    grown = cv2.dilate(box.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    thick = max(5, int(round(0.015 * max(h, w)))) | 1
    out = cv2.morphologyEx((obj & ~grown).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick, thick))) > 0
    count, comp, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
    labels = np.zeros((h, w), np.int32)
    k = 0
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < min_area or area < 0.3 * bw * bh or min(bw, bh) < 0.015 * max(h, w):
            continue
        seed = comp == i
        rx0, rx1 = max(0, int(x - 0.6 * bw)), min(w, int(x + 1.6 * bw))
        ry0, ry1 = max(0, int(y - 0.1 * bh)), min(h, int(y + bh + 1.6 * max(bh, bw)))
        roi = (slice(ry0, ry1), slice(rx0, rx1))
        gc = np.full((ry1 - ry0, rx1 - rx0), cv2.GC_PR_BGD, np.uint8)
        yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
        cx = x + 0.5 * bw
        near = np.abs(xx - cx) <= 0.75 * bw
        gc[near & obj[roi]] = cv2.GC_PR_FGD
        gc[~obj[roi]] = cv2.GC_BGD
        gc[(np.abs(xx - cx) > 1.1 * bw) & obj[roi]] = cv2.GC_BGD
        gc[cv2.erode(seed[roi].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = cv2.GC_FGD
        try:
            cv2.setRNGSeed(an.RNG_SEED)
            cv2.grabCut(np.ascontiguousarray(rgb[roi]), gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & obj[roi]
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        n2, comp2 = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
        ids = np.unique(comp2[seed[roi] & fg])
        region = np.zeros((h, w), bool)
        region[roi] = np.isin(comp2, ids[ids > 0])
        region = sculpt.fill_outline(region) & obj
        region &= labels == 0
        if region.sum() >= min_area:
            k += 1
            labels[region] = k
    return labels, k


def reconstruct(src, crop: np.ndarray, body: np.ndarray, fit_budget: int = 420) -> Optional[Tuple[dict, Dict[str, np.ndarray], np.ndarray]]:
    """Fit a box to the body (crop coordinates) and place every part standing on it.

    Returns (spec, arrays, overlay) or None when no box explains the body.
    """
    x0, y0 = src.features.bbox[0], src.features.bbox[1]
    rgb = src.image.rgb
    h, w = rgb.shape[:2]

    def full(m):
        out = np.zeros((h, w), bool)
        hh, ww = m.shape
        out[y0:y0 + hh, x0:x0 + ww] = m
        return out

    obj = full(crop)
    hull = np.zeros(body.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(body.astype(np.uint8))), 1)
    zones = np.where(full(body), 2, np.where(full(crop & (hull > 0)), 1, 0)).astype(np.uint8)
    fit, target = _fit_box(src, zones, fit_budget)
    view = scaled_view(fit.view, w / target.width, w, h)
    sticking = obj & ~(cv2.dilate(_box_mask(fit.params, view, w, h).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    if sticking.sum() > 0.004 * obj.sum():
        ring = cv2.dilate(sticking.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        zones = np.where(obj & ~ring, 2, np.where(ring & obj & ~sticking, 1, 0)).astype(np.uint8)
        fit, target = _fit_box(src, zones, fit_budget)
        view = scaled_view(fit.view, w / target.width, w, h)
    spill = float(fit.components.get("background_overlap", 1.0))
    missed = float(fit.components.get("object_missed", 1.0))
    if spill > MAX_SPILL or missed > MAX_MISSED:
        return None
    params = dict(fit.params)
    box = get_template("box")
    bounds = box.build(params, LOD(0.0, fast=True)).bounds()
    cam = box_camera(bounds, view, w, h)
    lo, hi = bounds[0], bounds[1]
    box_px = _box_mask(params, view, w, h)
    labels, count = _parts_on_box(rgb, obj, box_px, max(40.0, 0.003 * obj.sum()))
    parts: List[dict] = []
    arrays: Dict[str, np.ndarray] = {}
    skipped = 0
    for i in range(1, count + 1):
        m = labels == i
        ys, xs = np.nonzero(m)
        bottom = ys >= ys.max() - max(2, int(0.08 * (ys.max() - ys.min() + 1)))
        foot = (float(xs[bottom].mean()), float(ys.max()))
        ray_o, ray_d = camera_ray(cam, *foot)
        anchor = ray_box(ray_o, ray_d, lo, hi)
        if anchor is None:
            skipped += 1
            continue
        part = _place_part(src, cam, m, foot, anchor, len(parts) + 1)
        parts.append(part[0])
        arrays.update(part[1])
    colour = tuple(float(c) for c in np.median(rgb[box_px & obj], axis=0) / 255.0)
    dims = (hi - lo).tolist()
    notes = [f"Box-shaped body: a {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} stud box and the camera "
             f"({view.elevation:.0f} degrees above, {abs(view.azimuth):.0f} degrees to the side) were fitted to its outline; "
             "hidden faces repeat the opposite visible face."]
    if parts:
        notes.append(f"{len(parts)} object{'s' if len(parts) > 1 else ''} standing on the box rebuilt from {'their' if len(parts) > 1 else 'its'} own outline "
                     f"and placed where {'they stand' if len(parts) > 1 else 'it stands'}.")
    if skipped:
        notes.append(f"{skipped} part(s) beside the box were left out.")
    spec = {
        "strategy": "composite", "category": "composite", "template": "box",
        "params": {k: round(float(v), 4) for k, v in params.items()},
        "view": fit.view.to_dict(), "fit_resolution": [target.width, target.height],
        "fit": {"score": round(fit.score, 4), "components": {k: round(v, 4) for k, v in fit.components.items()}, "evaluations": fit.evaluations},
        "materials": {"surface": make_material("surface", "photo", colour, name="Photo surface", source="photo projection").to_dict()},
        "parts": parts,
        "confidence": round(float(np.clip(1.1 - fit.score * 0.6, 0.2, 0.95)) * (0.6 + 0.4 * src.seg.confidence), 3),
        "notes": notes, "photo_projection": True,
    }
    overlay = rgb.astype(np.float32).copy()
    overlay[box_px] = overlay[box_px] * 0.5 + np.array([240, 150, 40]) * 0.5
    for i in range(1, count + 1):
        overlay[labels == i] = overlay[labels == i] * 0.5 + np.array([60, 140, 240]) * 0.5
    return spec, arrays, overlay.astype(np.uint8)


def box_camera(bounds: np.ndarray, view, w: int, h: int) -> render.Camera:
    """The fitted view as a pinhole camera at full image resolution."""
    centre = 0.5 * (bounds[0] + bounds[1])
    radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.5, 1e-3)
    dist = radius / np.tan(np.radians(view.fov) / 2.0) * 1.15
    cam = render.orbit_camera(centre, dist, view.azimuth, view.elevation, w, h, view.fov)
    cam.focal *= view.scale
    cam.cx = w / 2.0 + view.tx
    cam.cy = h / 2.0 + view.ty
    return cam


def _place_part(src, cam: render.Camera, m: np.ndarray, foot: Tuple[float, float], anchor: np.ndarray,
                index: int) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Inflate one part from its outline (full-image mask) and stand it upright on ``anchor``, facing the camera."""
    x0 = y0 = 0
    ys, xs = np.nonzero(m)
    ax0, ax1, ay0, ay1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    sub = m[ay0:ay1 + 1, ax0:ax1 + 1]
    toward = cam.position - anchor
    toward[1] = 0.0
    toward /= max(np.linalg.norm(toward), 1e-9)
    z_axis = -toward
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    a, _, _ = cam.project(anchor[None])
    b, _, _ = cam.project((anchor + x_axis * 0.1)[None])
    c, _, _ = cam.project((anchor + y_axis * 0.1)[None])
    px_h = float(np.linalg.norm(b - a)) / 0.1
    px_v = float(np.linalg.norm(c - a)) / 0.1
    k = 1.0 / max(px_h, 1e-6)
    stretch = px_h / max(px_v, 1e-6)
    flat = sculpt.flat_edge_share(sub, "bottom") >= sculpt.FLAT_SHARE
    h2, _ = sculpt.thickness_field(sub, flat)
    height = sub.shape[0] * k
    rows = h2.shape[0] / sub.shape[0]
    depth = 2.0 * float(np.sqrt(max(h2.max(), 0.0))) / rows * k
    centre_col = (sub.shape[1] - 1) / 2.0
    fx, fy = foot[0] - ax0, foot[1] - ay0
    row = int(np.clip(round(fy * rows), 0, h2.shape[0] - 1))
    half = float(np.sqrt(max(h2[row].max(), 0.0))) / rows * k
    foot_model = np.array([-(fx - centre_col) * k, (sub.shape[0] - 1 - fy) * k, -half])
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = x_axis, y_axis, z_axis
    sink = np.array([0.0, -0.03 * height * stretch, 0.0])
    placement = g.translation(anchor + sink) @ frame @ g.scaling((1.0, stretch, 1.0)) @ g.translation(-foot_model)
    prefix = PART_PREFIX.format(index)
    spec = {
        "strategy": "inflate", "category": "inflate", "inflate_model": "balls",
        "dimensions": [float(sub.shape[1] * k), float(height), float(depth)],
        "depth_scale": 1.0, "flat_base": bool(flat), "flat_top": False,
        "bbox": [int(x0 + ax0), int(y0 + ay0), int(x0 + ax1), int(y0 + ay1)],
        "camera": {"type": "ortho", "px_per_stud": 1.0 / k, "origin_px": [float(x0 + ax0 + centre_col), float(y0 + ay1)],
                   "image_size": [int(src.image.rgb.shape[1]), int(src.image.rgb.shape[0])]},
        "photo_projection": True, "placement": placement.tolist(), "array_prefix": prefix, "name": f"Part{index}",
    }
    return spec, {prefix + "mask": sub.astype(np.uint8), prefix + "thickness": h2.astype(np.float32)}


def part_arrays(arrays: Dict[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    return {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}


HALF_TURN = np.diag([-1.0, 1.0, -1.0, 1.0])


def projection(spec: dict, sources, arrays: Dict[str, np.ndarray], part_projection):
    """Photo projection of a composite: box through the fitted camera, parts through their own."""
    from .textures import PhotoProjection

    src = sources[0]
    box = get_template(spec["template"])
    h, w = src.image.rgb.shape[:2]
    fw, fh = spec["fit_resolution"]
    from .fitting import View

    view = scaled_view(View.from_dict(spec["view"]), w / fw, w, h)
    cam = box_camera(box.build(spec["params"], LOD(0.0, fast=True)).bounds(), view, w, h)
    img = srgb_to_linear(src.image.rgb.astype(np.float64) / 255.0)
    proj = PhotoProjection(img, cam, None, mirror_axis=None)
    proj.extra = [PhotoProjection(img, cam, None, mirror_axis=None, post=HALF_TURN, fill_only=True)]
    for sub in spec.get("parts", []):
        sp = part_projection(sub, sources, None, part_arrays(arrays, sub["array_prefix"]))
        if sp is not None:
            sp.post = np.linalg.inv(np.asarray(sub["placement"], dtype=np.float64))
            proj.per_part[sub["name"]] = sp
    return proj
