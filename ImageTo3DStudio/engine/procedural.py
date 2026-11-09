"""Parametric object templates (Mode C: procedural object reconstruction).

Each template turns a small dictionary of physical parameters (in studs) into a
set of named parts built from reusable modelling components: bevelled boards,
turned/tapered legs, cushions with domed faces, lathed shades, extruded roofs and
lofted bodies. Tessellation density comes from an ``LOD`` object so the same
parameters can be rebuilt at any triangle budget without re-analysing the image.

Coordinates: floor at y = 0, object centred on x = z = 0, front faces -Z.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as g
from .materials import MaterialSpec, make_material
from .scene import NativePrimitive, Part, Scene

STUD = 0.28  # metres per stud (Roblox physics units)


def m2s(metres: float) -> float:
    return metres / STUD


@dataclass
class LOD:
    """Tessellation controls derived from one quality scalar ``q`` in [0, 1]."""

    q: float = 0.6
    bevel_scale: float = 1.0
    keep_details: bool = True
    fast: bool = False
    bevel_q: Optional[float] = None
    round_q: Optional[float] = None
    cushion_q: Optional[float] = None

    def _q(self, override: Optional[float]) -> float:
        return self.q if override is None else override

    @property
    def bevel_segments(self) -> int:
        q = self._q(self.bevel_q)
        if q < 0.08:
            return 0
        return int(np.clip(round(1 + q * 3), 1, 4))

    @property
    def round_segments(self) -> int:
        return int(np.clip(round(6 + self._q(self.round_q) * 26), 6, 32))

    @property
    def rings(self) -> int:
        return int(np.clip(round(4 + self._q(self.round_q) * 12), 4, 16))

    @property
    def cushion_divisions(self) -> int:
        return int(np.clip(round(self._q(self.cushion_q) * 7), 0, 7))


@dataclass
class ParamDef:
    default: float
    lo: float
    hi: float
    label: str
    kind: str = "length"
    fit: bool = False


@dataclass
class MaterialSlot:
    key: str
    label: str
    candidates: Tuple[str, ...]
    default_kind: str
    default_color: Tuple[float, float, float]
    quilted: bool = False


@dataclass
class Template:
    name: str
    label: str
    params: Dict[str, ParamDef]
    materials: List[MaterialSlot]
    builder: Callable[["Builder", Dict[str, float]], None]
    size_params: Tuple[str, str, str]
    description: str = ""

    def defaults(self) -> Dict[str, float]:
        return {k: v.default for k, v in self.params.items()}

    def clamp(self, params: Dict[str, float]) -> Dict[str, float]:
        out = self.defaults()
        for k, v in params.items():
            if k in self.params:
                d = self.params[k]
                out[k] = float(np.clip(float(v), d.lo, d.hi))
        return out

    def default_materials(self) -> Dict[str, MaterialSpec]:
        return {s.key: make_material(s.key, s.default_kind, s.default_color, quilted=s.quilted, source="template default", name=s.label) for s in self.materials}

    def build(self, params: Dict[str, float], lod: LOD, materials: Optional[Dict[str, MaterialSpec]] = None, with_detail: bool = False) -> Scene:
        p = self.clamp(params)
        b = Builder(lod, with_detail)
        self.builder(b, p)
        mats = materials if materials is not None else self.default_materials()
        for slot in self.materials:
            if slot.key not in mats:
                mats[slot.key] = make_material(slot.key, slot.default_kind, slot.default_color, name=slot.label)
        used = {part.material for part in b.parts}
        mats = {k: v for k, v in mats.items() if k in used}
        return Scene(self.label, b.parts, mats, {"template": self.name, "params": p, "lod": lod.q})


class Builder:
    """Collects parts; every helper also records a Roblox-native primitive."""

    def __init__(self, lod: LOD, with_detail: bool):
        self.lod = lod
        self.with_detail = with_detail
        self.parts: List[Part] = []
        self._seed = 0

    def _next_seed(self) -> int:
        self._seed += 1
        return self._seed

    def board(self, name: str, mat: str, size, center, bevel: float = 0.02, grain_axis: int = 0, rot: Optional[np.ndarray] = None, group: str = "") -> Part:
        size = np.maximum(np.asarray(size, dtype=np.float64), 0.02)
        r = bevel * self.lod.bevel_scale
        segs = self.lod.bevel_segments if r > 1e-4 else 0
        mesh = g.box(size, (0, 0, 0), r, segs, grain_axis, name)
        frame = g.translation(center) @ (rot if rot is not None else np.eye(4))
        mesh.apply_transform(frame)
        detail = None
        if self.with_detail and r > 1e-4 and segs < 4:
            detail = g.box(size, (0, 0, 0), r, 4, grain_axis, name)
            detail.apply_transform(frame)
        part = Part(name, mesh, mat, frame, grain_axis, "u", self._next_seed(), False,
                    [NativePrimitive("Block", tuple(size), frame.copy(), mat, name)], detail, group or name)
        self.parts.append(part)
        return part

    def cushion(self, name: str, mat: str, size, center, puff: float = 0.12, bevel: float = 0.25, sag: float = 0.0, group: str = "") -> Part:
        size = np.maximum(np.asarray(size, dtype=np.float64), 0.05)
        lod = self.lod
        r = min(bevel * lod.bevel_scale, size.min() * 0.45)
        segs = max(lod.bevel_segments, 1) if r > 1e-4 else 0
        div = lod.cushion_divisions

        def make(segs_, div_):
            m = g.box(size, (0, 0, 0), r, segs_, 0, name, flat_divisions=div_)
            half = size / 2.0
            x = np.clip(m.vertices[:, 0] / half[0], -1, 1)
            z = np.clip(m.vertices[:, 2] / half[2], -1, 1)
            dome = (1 - x * x) * (1 - z * z)
            yn = m.vertices[:, 1] / half[1]
            m.vertices[:, 1] += np.sign(yn) * np.abs(yn) ** 2 * dome * puff * size[1]
            if sag:
                m.vertices[:, 1] -= sag * size[1] * (1 - x * x) * 0.5
            return m if lod.fast else g.recompute_normals(m, 60.0)

        mesh = make(segs, div)
        frame = g.translation(center)
        mesh.apply_transform(frame)
        detail = None
        if self.with_detail and (segs < 3 or div < 5):
            detail = make(4, 7)
            detail.apply_transform(frame)
        part = Part(name, mesh, mat, frame, 0, "u", self._next_seed(), False,
                    [NativePrimitive("Block", tuple(size * np.array([1, 1 + puff * 0.5, 1])), frame.copy(), mat, name)], detail, group or name)
        self.parts.append(part)
        return part

    def rod(self, name: str, mat: str, radius: float, start, end, taper: float = 1.0, bevel: float = 0.0, group: str = "") -> Part:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        axis = end - start
        length = float(np.linalg.norm(axis))
        lod = self.lod
        segs = lod.round_segments
        r_b = bevel * lod.bevel_scale
        bsegs = lod.bevel_segments if r_b > 1e-4 else 0
        mesh = g.cylinder(radius, length, segs, radius * taper, r_b, bsegs, name)
        frame = g.translation(start) @ g.align_y_to(axis)
        mesh.apply_transform(frame)
        detail = None
        if self.with_detail and (segs < 24 or (r_b > 1e-4 and bsegs < 4)):
            detail = g.cylinder(radius, length, 32, radius * taper, r_b, 4 if r_b > 1e-4 else 0, name)
            detail.apply_transform(frame)
        mid = 0.5 * (start + end)
        rot = g.align_y_to(axis) @ g.rotation([0, 0, 1], np.pi / 2)
        cf = g.translation(mid) @ rot
        d = 2 * radius * (1 + taper) / 2
        part = Part(name, mesh, mat, frame, 1, "v", self._next_seed(), False,
                    [NativePrimitive("Cylinder", (length, d, d), cf, mat, name)], detail, group or name)
        self.parts.append(part)
        return part

    def turned(self, name: str, mat: str, profile: Sequence[Tuple[float, float]], base, group: str = "", native_shape: str = "Cylinder") -> Part:
        prof = [(max(r, 0.0), y) for r, y in profile]
        mesh = g.lathe(prof, self.lod.round_segments, smooth_profile=True, hard_angle_deg=55.0, name=name)
        frame = g.translation(base)
        mesh.apply_transform(frame)
        detail = None
        if self.with_detail and self.lod.round_segments < 24:
            detail = g.lathe(prof, 32, smooth_profile=True, hard_angle_deg=55.0, name=name)
            detail.apply_transform(frame)
        rmax = max(r for r, _ in prof)
        ys = [y for _, y in prof]
        native = []
        if native_shape == "Ball":
            cf = g.translation(np.asarray(base) + np.array([0, (max(ys) + min(ys)) / 2, 0]))
            native.append(NativePrimitive("Ball", (2 * rmax, 2 * rmax, 2 * rmax), cf, mat, name))
        else:
            segments = _stack_profile(prof, 4)
            for i, (r, y0, y1) in enumerate(segments):
                cf = g.translation(np.asarray(base) + np.array([0, (y0 + y1) / 2, 0])) @ g.rotation([0, 0, 1], np.pi / 2)
                native.append(NativePrimitive("Cylinder", (max(y1 - y0, 0.05), 2 * max(r, 0.03), 2 * max(r, 0.03)), cf, mat, f"{name}{i + 1}"))
        part = Part(name, mesh, mat, frame, 1, "v", self._next_seed(), False, native, detail, group or name)
        self.parts.append(part)
        return part

    def prism(self, name: str, mat: str, outline: np.ndarray, depth: float, center, rot: Optional[np.ndarray] = None, bevel: float = 0.0, group: str = "", native_boxes: Optional[List[Tuple[tuple, np.ndarray]]] = None) -> Part:
        mesh = g.extrude_polygon(outline, (), depth, bevel * self.lod.bevel_scale, name)
        frame = g.translation(center) @ (rot if rot is not None else np.eye(4))
        mesh.apply_transform(frame)
        native = []
        if native_boxes:
            for i, (size, local_cf) in enumerate(native_boxes):
                native.append(NativePrimitive("Block", tuple(size), frame @ local_cf, mat, f"{name}{i + 1}"))
        else:
            lo, hi = outline.min(0), outline.max(0)
            cf = frame @ g.translation([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0])
            native.append(NativePrimitive("Block", (hi[0] - lo[0], hi[1] - lo[1], depth), cf, mat, name))
        part = Part(name, mesh, mat, frame, 0, "u", self._next_seed(), False, native, None, group or name)
        self.parts.append(part)
        return part

    def loft_part(self, name: str, mat: str, sections: List[np.ndarray], group: str = "", native_size=None, native_center=None) -> Part:
        mesh = g.loft(sections, cap_ends=True, smooth=True, name=name)
        mesh = g.recompute_normals(mesh, 45.0)
        native = []
        if native_size is not None:
            native.append(NativePrimitive("Block", tuple(native_size), g.translation(native_center), mat, name))
        part = Part(name, mesh, mat, np.eye(4), 0, "u", self._next_seed(), False, native, None, group or name)
        self.parts.append(part)
        return part


def _stack_profile(profile, max_segments: int):
    """Approximate a lathe profile by a few stacked cylinders for native parts."""
    pts = np.asarray(profile, dtype=np.float64)
    y_lo, y_hi = pts[:, 1].min(), pts[:, 1].max()
    edges = np.linspace(y_lo, y_hi, max_segments + 1)
    out = []
    for y0, y1 in zip(edges[:-1], edges[1:]):
        ys = np.linspace(y0, y1, 5)
        rs = np.interp(ys, pts[:, 1], pts[:, 0]) if np.all(np.diff(pts[:, 1]) >= 0) else np.full(5, pts[:, 0].max())
        out.append((float(rs.mean()), float(y0), float(y1)))
    return out


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

WOOD_FRAME = ("wood", "painted", "metal", "plastic")
SOFT = ("fabric", "leather", "plastic")


def _bed(b: Builder, p: Dict[str, float]) -> None:
    W, L = p["width"], p["length"]
    leg_h, rail_h, rail_t = p["leg_height"], p["rail_height"], p["rail_thickness"]
    post = p["post_size"]
    head_h, foot_h = p["headboard_height"], p["footboard_height"]
    mat_t = p["mattress_thickness"]
    bev = p["bevel"]
    hx, hz = W / 2, L / 2
    rail_top = leg_h + rail_h
    foot_top = max(foot_h, rail_top)
    for sx in (-1, 1):
        x = sx * (hx - post / 2)
        b.board("HeadPost" + ("R" if sx > 0 else "L"), "frame", (post, head_h, post), (x, head_h / 2, hz - post / 2), bev, 1, group="Headboard")
        b.board("FootPost" + ("R" if sx > 0 else "L"), "frame", (post, foot_top, post), (x, foot_top / 2, -hz + post / 2), bev, 1, group="Footboard")
        b.board("SideRail" + ("R" if sx > 0 else "L"), "frame", (rail_t, rail_h, L - 2 * post), (sx * (hx - rail_t / 2 - (post - rail_t) / 2), leg_h + rail_h / 2, 0), bev, 2, group="Frame")
    b.board("FootRail", "frame", (W - 2 * post, rail_h, rail_t), (0, leg_h + rail_h / 2, -hz + post / 2), bev, 0, group="Frame")
    b.board("HeadRail", "frame", (W - 2 * post, rail_h, rail_t), (0, leg_h + rail_h / 2, hz - post / 2), bev, 0, group="Frame")
    if p["center_legs"] >= 0.5:
        for sx in (-1, 1):
            b.board("CenterLeg" + ("R" if sx > 0 else "L"), "frame", (post * 0.9, leg_h, post * 0.9), (sx * (hx - post / 2), leg_h / 2, 0), bev, 1, group="Frame")
    if foot_h > rail_top + 0.1:
        panel_h = foot_top - rail_top
        b.board("FootPanel", "frame", (W - 2 * post, panel_h, rail_t), (0, rail_top + panel_h / 2, -hz + post / 2), bev, 0, group="Footboard")
        b.board("FootCap", "frame", (W + 0.06, post * 0.45, post * 1.15), (0, foot_top + post * 0.2, -hz + post / 2), bev, 0, group="Footboard")
    board_bottom = rail_top + mat_t * 0.35
    top_rail_h = max(post * 1.1, 0.25)
    b.board("HeadTopRail", "frame", (W + 0.08, top_rail_h, post * 1.2), (0, head_h - top_rail_h / 2, hz - post / 2), bev, 0, group="Headboard")
    b.board("HeadLowRail", "frame", (W - 2 * post, top_rail_h * 0.8, rail_t), (0, board_bottom, hz - post / 2), bev, 0, group="Headboard")
    slat_zone_lo = board_bottom + top_rail_h * 0.4
    slat_zone_hi = head_h - top_rail_h
    n_slats = int(round(p["headboard_slats"]))
    inner_w = W - 2 * post
    if n_slats > 0 and slat_zone_hi - slat_zone_lo > 0.1:
        slat_w = min(p["slat_width"], inner_w / (n_slats * 1.3))
        gap = (inner_w - n_slats * slat_w) / (n_slats + 1)
        for i in range(n_slats):
            x = -inner_w / 2 + gap + slat_w / 2 + i * (slat_w + gap)
            b.board(f"Slat{i + 1}", "frame", (slat_w, slat_zone_hi - slat_zone_lo, rail_t * 0.8), (x, (slat_zone_lo + slat_zone_hi) / 2, hz - post / 2), bev * 0.5, 1, group="Headboard")
    elif slat_zone_hi - slat_zone_lo > 0.1:
        b.board("HeadPanel", "frame", (inner_w, slat_zone_hi - slat_zone_lo, rail_t * 0.8), (0, (slat_zone_lo + slat_zone_hi) / 2, hz - post / 2), bev, 0, group="Headboard")
    mat_w = W - 2 * rail_t - 0.04
    mat_l = L - 2 * post - 0.04
    mat_y = rail_top - p["mattress_drop"] + mat_t / 2
    b.cushion("Mattress", "mattress", (mat_w, mat_t, mat_l), (0, mat_y, 0.0), puff=0.05, bevel=min(0.12, mat_t * 0.3), group="Bedding")
    if p["duvet"] >= 0.5:
        duvet_l = mat_l * p["duvet_coverage"]
        duvet_t = 0.16
        b.cushion("Duvet", "duvet", (mat_w + 0.12, duvet_t + mat_t * 0.55, duvet_l), (0, mat_y + mat_t / 2 - (mat_t * 0.55) / 2 + duvet_t / 2 + 0.02, -mat_l / 2 + duvet_l / 2), puff=0.06, bevel=0.14, group="Bedding")
        b.cushion("DuvetFold", "duvet", (mat_w + 0.14, duvet_t * 1.2, 0.5), (0, mat_y + mat_t / 2 + duvet_t * 0.75, -mat_l / 2 + duvet_l - 0.18), puff=0.2, bevel=0.12, group="Bedding")
    n_p = int(round(p["pillows"]))
    if n_p > 0:
        pw = min(p["pillow_width"], (mat_w - 0.1) / n_p)
        for i in range(n_p):
            x = -mat_w / 2 + mat_w * (i + 0.5) / n_p
            b.cushion(f"Pillow{i + 1}", "pillow", (pw * 0.92, 0.38, 1.45), (x, mat_y + mat_t / 2 + 0.17, mat_l / 2 - 0.9), puff=0.55, bevel=0.17, group="Bedding")


def _chair(b: Builder, p: Dict[str, float]) -> None:
    W, D = p["width"], p["depth"]
    seat_h, seat_t = p["seat_height"], p["seat_thickness"]
    back_h = p["back_height"]
    leg = p["leg_size"]
    bev = p["bevel"]
    hx, hz = W / 2, D / 2
    round_legs = p["round_legs"] >= 0.5
    taper = p["leg_taper"]
    for sx in (-1, 1):
        for sz in (-1, 1):
            x = sx * (hx - leg / 2 - 0.02)
            z = sz * (hz - leg / 2 - 0.02)
            top = seat_h - seat_t if sz < 0 else back_h
            nm = ("Back" if sz > 0 else "Front") + "Leg" + ("R" if sx > 0 else "L")
            if round_legs:
                b.rod(nm, "frame", leg / 2, (x, 0, z), (x, top, z), taper=taper if sz < 0 else 1.0, bevel=leg * 0.1, group="Legs")
            else:
                b.board(nm, "frame", (leg, top, leg), (x, top / 2, z), bev, 1, group="Legs")
    if p["cushion"] >= 0.5:
        b.board("SeatBase", "frame", (W, seat_t * 0.5, D), (0, seat_h - seat_t * 0.75, 0), bev, 0, group="Seat")
        b.cushion("SeatCushion", "seat", (W - 0.06, seat_t * 0.9, D - 0.06), (0, seat_h - seat_t * 0.05, 0), puff=0.25, bevel=0.1, group="Seat")
    else:
        b.board("Seat", "seat", (W, seat_t, D), (0, seat_h - seat_t / 2, 0), bev * 1.5, 0, group="Seat")
    apron_h = p["apron_height"]
    if apron_h > 0.02:
        for sz in (-1, 1):
            b.board("Apron" + ("Back" if sz > 0 else "Front"), "frame", (W - 2 * leg, apron_h, leg * 0.5), (0, seat_h - seat_t - apron_h / 2, sz * (hz - leg / 2 - 0.02)), bev, 0, group="Frame")
        for sx in (-1, 1):
            b.board("Apron" + ("R" if sx > 0 else "L"), "frame", (leg * 0.5, apron_h, D - 2 * leg), (sx * (hx - leg / 2 - 0.02), seat_h - seat_t - apron_h / 2, 0), bev, 2, group="Frame")
    if p["stretchers"] >= 0.5:
        y = seat_h * 0.28
        for sx in (-1, 1):
            b.board("Stretcher" + ("R" if sx > 0 else "L"), "frame", (leg * 0.6, leg * 0.6, D - 2 * leg), (sx * (hx - leg / 2 - 0.02), y, 0), bev, 2, group="Frame")
    zb = hz - leg / 2 - 0.02
    top_rail = p["back_rail"]
    b.board("BackTopRail", "frame", (W - 0.04, top_rail, leg * 0.9), (0, back_h - top_rail / 2, zb), bev, 0, group="Back")
    n = int(round(p["back_slats"]))
    lo = seat_h + 0.12
    hi = back_h - top_rail
    if hi > lo + 0.1:
        if n > 0:
            inner = W - 2 * leg - 0.04
            sw = min(p["slat_width"], inner / (n * 1.4))
            gap = (inner - n * sw) / (n + 1)
            for i in range(n):
                x = -inner / 2 + gap + sw / 2 + i * (sw + gap)
                b.board(f"BackSlat{i + 1}", "frame", (sw, hi - lo, leg * 0.45), (x, (lo + hi) / 2, zb), bev * 0.6, 1, group="Back")
        else:
            b.board("BackPanel", "seat" if p["cushion"] >= 0.5 else "frame", (W - 2 * leg - 0.04, hi - lo, leg * 0.5), (0, (lo + hi) / 2, zb), bev * 1.5, 0, group="Back")
    if p["arms"] >= 0.5:
        arm_y = seat_h + p["arm_height"]
        for sx in (-1, 1):
            x = sx * (hx - leg / 2)
            b.board("Arm" + ("R" if sx > 0 else "L"), "frame", (leg * 1.2, leg * 0.7, D * 0.95), (x, arm_y, -0.02), bev, 2, group="Arms")
            b.board("ArmPost" + ("R" if sx > 0 else "L"), "frame", (leg * 0.8, arm_y - seat_h, leg * 0.8), (x, seat_h + (arm_y - seat_h) / 2, -hz + leg), bev, 1, group="Arms")


def _table(b: Builder, p: Dict[str, float]) -> None:
    W, D, H = p["width"], p["depth"], p["height"]
    top_t, leg, bev = p["top_thickness"], p["leg_size"], p["bevel"]
    over = p["overhang"]
    round_top = p["round_top"] >= 0.5
    if round_top:
        r = min(W, D) / 2
        prof = [(0.0, 0.0), (r - top_t * 0.3, 0.0), (r, top_t * 0.3), (r, top_t * 0.7), (r - top_t * 0.3, top_t), (0.0, top_t)]
        b.turned("Top", "top", prof, (0, H - top_t, 0), group="Top")
    else:
        b.board("Top", "top", (W, top_t, D), (0, H - top_t / 2, 0), bev * 1.5, 0, group="Top")
    if p["pedestal"] >= 0.5:
        col_r = max(leg * 0.9, 0.2)
        b.rod("Column", "frame", col_r, (0, 0.15, 0), (0, H - top_t, 0), taper=0.85, group="Base")
        b.turned("Foot", "frame", [(0.0, 0.0), (min(W, D) * 0.33, 0.0), (min(W, D) * 0.3, 0.12), (col_r * 1.3, 0.2), (0.0, 0.2)], (0, 0, 0), group="Base")
        return
    hx, hz = W / 2 - over, D / 2 - over
    round_legs = p["round_legs"] >= 0.5
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * (hx - leg / 2), sz * (hz - leg / 2)
            nm = ("Back" if sz > 0 else "Front") + "Leg" + ("R" if sx > 0 else "L")
            if round_legs:
                b.rod(nm, "frame", leg / 2, (x, 0, z), (x, H - top_t, z), taper=p["leg_taper"], bevel=leg * 0.08, group="Legs")
            else:
                b.board(nm, "frame", (leg, H - top_t, leg), (x, (H - top_t) / 2, z), bev, 1, group="Legs")
    ah = p["apron_height"]
    if ah > 0.02:
        for sz in (-1, 1):
            b.board("Apron" + ("Back" if sz > 0 else "Front"), "frame", (2 * hx - 2 * leg, ah, leg * 0.35), (0, H - top_t - ah / 2, sz * (hz - leg / 2)), bev, 0, group="Frame")
        for sx in (-1, 1):
            b.board("Apron" + ("R" if sx > 0 else "L"), "frame", (leg * 0.35, ah, 2 * hz - 2 * leg), (sx * (hx - leg / 2), H - top_t - ah / 2, 0), bev, 2, group="Frame")
    if p["shelf"] >= 0.5:
        b.board("LowerShelf", "top", (2 * hx - leg * 0.5, top_t * 0.7, 2 * hz - leg * 0.5), (0, H * 0.22, 0), bev, 0, group="Frame")


def _sofa(b: Builder, p: Dict[str, float]) -> None:
    W, D, H = p["width"], p["depth"], p["height"]
    seat_h, arm_w, arm_h = p["seat_height"], p["arm_width"], p["arm_height"]
    leg_h = p["leg_height"]
    bev = p["bevel"]
    n = int(round(p["cushions"]))
    back_t = p["back_thickness"]
    base_top = seat_h - 0.45
    b.cushion("Base", "body", (W, base_top - leg_h, D), (0, leg_h + (base_top - leg_h) / 2, 0), puff=0.02, bevel=0.12, group="Body")
    for sx in (-1, 1):
        b.cushion("Arm" + ("R" if sx > 0 else "L"), "body", (arm_w, arm_h - leg_h, D), (sx * (W / 2 - arm_w / 2), leg_h + (arm_h - leg_h) / 2, 0), puff=0.04, bevel=min(0.3, arm_w * 0.35), group="Arms")
    b.cushion("BackFrame", "body", (W - 2 * arm_w + 0.02, H - leg_h, back_t * 0.6), (0, leg_h + (H - leg_h) / 2, D / 2 - back_t * 0.3), puff=0.03, bevel=0.15, group="Body")
    inner = W - 2 * arm_w
    cw = inner / max(n, 1)
    seat_d = D - back_t
    for i in range(max(n, 1)):
        x = -inner / 2 + cw * (i + 0.5)
        b.cushion(f"SeatCushion{i + 1}", "cushion", (cw - 0.05, 0.5, seat_d - 0.05), (x, base_top + 0.25, -D / 2 + seat_d / 2), puff=0.18, bevel=0.14, group="Cushions")
        b.cushion(f"BackCushion{i + 1}", "cushion", (cw - 0.05, H - base_top - 0.6, back_t * 0.55), (x, base_top + 0.5 + (H - base_top - 0.6) / 2, D / 2 - back_t * 0.6 - back_t * 0.27), puff=0.22, bevel=0.16, group="Cushions")
    if leg_h > 0.05:
        for sx in (-1, 1):
            for sz in (-1, 1):
                b.board(("Back" if sz > 0 else "Front") + "Leg" + ("R" if sx > 0 else "L"), "legs", (0.22, leg_h, 0.22), (sx * (W / 2 - 0.25), leg_h / 2, sz * (D / 2 - 0.25)), bev, 1, group="Legs")


def _cabinet(b: Builder, p: Dict[str, float]) -> None:
    W, D, H = p["width"], p["depth"], p["height"]
    t, plinth, bev = p["panel_thickness"], p["plinth_height"], p["bevel"]
    rows = int(round(p["rows"]))
    cols = int(round(p["columns"]))
    body_lo = plinth
    b.board("Top", "body", (W + 0.06, t, D + 0.04), (0, H - t / 2, 0), bev, 0, group="Carcass")
    for sx in (-1, 1):
        b.board("Side" + ("R" if sx > 0 else "L"), "body", (t, H - t - body_lo, D), (sx * (W / 2 - t / 2), body_lo + (H - t - body_lo) / 2, 0), bev, 1, group="Carcass")
    b.board("Bottom", "body", (W - 2 * t, t, D), (0, body_lo + t / 2, 0), bev, 0, group="Carcass")
    b.board("Back", "body", (W - 2 * t, H - t - body_lo, t * 0.5), (0, body_lo + (H - t - body_lo) / 2, D / 2 - t * 0.25), 0.0, 1, group="Carcass")
    if plinth > 0.05:
        if p["legs"] >= 0.5:
            for sx in (-1, 1):
                for sz in (-1, 1):
                    b.board(("Back" if sz > 0 else "Front") + "Foot" + ("R" if sx > 0 else "L"), "frame", (0.2, plinth, 0.2), (sx * (W / 2 - 0.2), plinth / 2, sz * (D / 2 - 0.2)), bev, 1, group="Base")
        else:
            b.board("Plinth", "body", (W - 0.1, plinth, D - 0.12), (0, plinth / 2, 0.04), bev, 0, group="Base")
    gap = 0.03
    fw = (W - 2 * t) / max(cols, 1)
    fh = (H - t - body_lo - t) / max(rows, 1)
    z_front = -D / 2 + t * 0.4
    for r in range(max(rows, 1)):
        for c in range(max(cols, 1)):
            x = -W / 2 + t + fw * (c + 0.5)
            y = body_lo + t + fh * (r + 0.5)
            name = f"Front{r + 1}_{c + 1}"
            b.board(name, "front", (fw - gap, fh - gap, t * 0.8), (x, y, z_front), bev, 0 if fw >= fh else 1, group="Fronts")
            hs = p["handle_size"]
            if hs > 0.01:
                drawer = fw >= fh
                hx = x if drawer else x + (fw / 2 - 0.25) * (1 if c % 2 == 0 else -1)
                hy = y + (fh * 0.25 if drawer else 0)
                if drawer:
                    b.board(f"Handle{r + 1}_{c + 1}", "handle", (hs * 3, hs * 0.6, hs * 0.6), (hx, hy, z_front - t * 0.4 - hs * 0.3), hs * 0.2, 0, group="Handles")
                else:
                    b.board(f"Handle{r + 1}_{c + 1}", "handle", (hs * 0.6, hs * 3, hs * 0.6), (hx, hy, z_front - t * 0.4 - hs * 0.3), hs * 0.2, 1, group="Handles")


def _shelf(b: Builder, p: Dict[str, float]) -> None:
    W, D, H = p["width"], p["depth"], p["height"]
    t, bev = p["panel_thickness"], p["bevel"]
    n = int(round(p["shelves"]))
    for sx in (-1, 1):
        b.board("Side" + ("R" if sx > 0 else "L"), "body", (t, H, D), (sx * (W / 2 - t / 2), H / 2, 0), bev, 1, group="Carcass")
    b.board("Top", "body", (W - 2 * t, t, D), (0, H - t / 2, 0), bev, 0, group="Carcass")
    b.board("Bottom", "body", (W - 2 * t, t, D), (0, p["plinth_height"] + t / 2, 0), bev, 0, group="Carcass")
    if p["back_panel"] >= 0.5:
        b.board("Back", "back", (W - 2 * t, H - t, t * 0.4), (0, H / 2, D / 2 - t * 0.2), 0.0, 1, group="Carcass")
    lo, hi = p["plinth_height"] + t, H - t
    for i in range(n):
        y = lo + (hi - lo) * (i + 1) / (n + 1)
        b.board(f"Shelf{i + 1}", "body", (W - 2 * t, t, D - 0.06), (0, y, -0.02), bev, 0, group="Shelves")


def _lamp(b: Builder, p: Dict[str, float]) -> None:
    H = p["height"]
    base_r, base_h = p["base_radius"], p["base_height"]
    stem_r = p["stem_radius"]
    sh_bot, sh_top, sh_h = p["shade_bottom_radius"], p["shade_top_radius"], p["shade_height"]
    b.turned("Base", "base", [(0.0, 0.0), (base_r, 0.0), (base_r, base_h * 0.6), (base_r * 0.85, base_h), (0.0, base_h)], (0, 0, 0), group="Base")
    shade_bottom_y = H - sh_h
    b.rod("Stem", "base", stem_r, (0, base_h, 0), (0, shade_bottom_y + sh_h * 0.35, 0), taper=1.0, group="Stem")
    wall = max(0.04, sh_bot * 0.04)
    prof = [(sh_bot - wall, 0.0), (sh_bot, 0.0), (sh_top, sh_h), (sh_top - wall, sh_h), (sh_bot - wall, 0.0)]
    mesh = g.lathe(prof, b.lod.round_segments, smooth_profile=False, hard_angle_deg=30.0, closed_profile=True, name="Shade")
    frame = g.translation((0, shade_bottom_y, 0))
    mesh.apply_transform(frame)
    native = [NativePrimitive("Cylinder", (sh_h, 2 * (sh_bot + sh_top) / 2, 2 * (sh_bot + sh_top) / 2), g.translation((0, shade_bottom_y + sh_h / 2, 0)) @ g.rotation([0, 0, 1], np.pi / 2), "shade", "Shade")]
    b.parts.append(Part("Shade", mesh, "shade", frame, 1, "v", b._next_seed(), False, native, None, "Shade"))
    if p["bulb"] >= 0.5:
        br = min(sh_bot, sh_top) * 0.45
        b.turned("Bulb", "bulb", [(0.0, 0.0)] + [(br * np.sin(t), br - br * np.cos(t)) for t in np.linspace(0.3, np.pi, 8)[1:]] + [(0.0, 2 * br)], (0, shade_bottom_y + sh_h * 0.35, 0), group="Bulb", native_shape="Ball")


def _house(b: Builder, p: Dict[str, float]) -> None:
    W, D, H = p["width"], p["depth"], p["wall_height"]
    roof_h, over = p["roof_height"], p["roof_overhang"]
    b.board("Walls", "walls", (W, H, D), (0, H / 2, 0), 0.05, 0, group="Walls")
    b.board("Foundation", "trim", (W + 0.3, 0.4, D + 0.3), (0, 0.2, 0), 0.04, 0, group="Walls")
    half = W / 2 + over
    gable = np.array([[-W / 2, 0.0], [W / 2, 0.0], [0.0, roof_h * (W / 2) / half]])
    b.prism("Gable", "walls", gable, D - 0.02, (0, H, 0), group="Roof")
    slope_len = float(np.hypot(half, roof_h))
    ang = float(np.arctan2(roof_h, half))
    for sx in (-1, 1):
        rot = g.rotation([0, 0, 1], -sx * ang)
        cx = sx * half / 2
        cy = H + roof_h / 2 + 0.12
        b.board("Roof" + ("R" if sx > 0 else "L"), "roof", (slope_len + 0.1, 0.25, D + 2 * over), (cx, cy, 0), 0.03, 0, rot=rot, group="Roof")
    dw, dh = p["door_width"], p["door_height"]
    b.board("Door", "door", (dw, dh, 0.15), (0, dh / 2 + 0.4, -D / 2 - 0.06), 0.03, 1, group="Openings")
    n = int(round(p["windows"]))
    for i in range(n):
        side = -1 if i % 2 == 0 else 1
        k = i // 2
        x = side * (W / 4 + k * 0.1)
        b.board(f"Window{i + 1}", "glass", (p["window_size"], p["window_size"] * 0.9, 0.12), (x, H * 0.6, -D / 2 - 0.05), 0.02, 0, group="Openings")
        b.board(f"WindowFrame{i + 1}", "trim", (p["window_size"] + 0.3, 0.15, 0.25), (x, H * 0.6 - p["window_size"] * 0.45 - 0.07, -D / 2 - 0.1), 0.02, 0, group="Openings")
    if p["chimney"] >= 0.5:
        b.board("Chimney", "trim", (0.8, roof_h * 0.9 + 0.6, 0.8), (W * 0.25, H + roof_h * 0.55 + 0.3, D * 0.15), 0.03, 1, group="Roof")


def _car(b: Builder, p: Dict[str, float]) -> None:
    L, W, H = p["length"], p["width"], p["height"]
    wr = p["wheel_radius"]
    clearance = p["ground_clearance"]
    cabin_len = p["cabin_length"] * L
    cabin_h = p["cabin_height"]
    body_h = H - cabin_h - clearance
    body_lo = clearance
    hood = p["hood_slope"]
    nseg = 10
    zs = np.linspace(-L / 2, L / 2, nseg)
    sections = []
    for i, z in enumerate(zs):
        t = i / (nseg - 1)
        end = min(t, 1 - t) * 2
        tip = 0.82 + 0.18 * np.clip(end * 3, 0, 1)
        w = W / 2 * tip
        top = body_lo + body_h * (1.0 - (hood * (1 - np.clip(end * 3, 0, 1)) if t < 0.5 else 0.12 * (1 - np.clip(end * 3, 0, 1))))
        ring = _rounded_rect_ring(w, body_lo + 0.1, top, 0.25, 6)
        sections.append(np.column_stack([ring[:, 0], ring[:, 1], np.full(len(ring), z)]))
    b.loft_part("Body", "body", sections, group="Body", native_size=(W, body_h, L), native_center=(0, body_lo + body_h / 2, 0))
    cz = p["cabin_offset"] * L
    c_lo = body_lo + body_h - 0.05
    outline = np.array([[-cabin_len / 2, 0.0], [cabin_len / 2, 0.0], [cabin_len / 2 - cabin_len * 0.22, cabin_h], [-cabin_len / 2 + cabin_len * 0.3, cabin_h]])
    rot = g.rotation([0, 1, 0], np.pi / 2)
    b.prism("Cabin", "glass", outline, W * 0.86, (0, c_lo, cz), rot=rot, bevel=0.06, group="Cabin")
    b.board("Roof", "body", (W * 0.82, 0.12, cabin_len * 0.5), (0, c_lo + cabin_h + 0.04, cz + cabin_len * 0.04), 0.05, 2, group="Cabin")
    for sz in (-1, 1):
        for sx in (-1, 1):
            z = sz * (L / 2 - wr * 1.6)
            x = sx * (W / 2 - p["wheel_width"] / 2 + 0.05)
            start = (x - p["wheel_width"] / 2, wr, z)
            end = (x + p["wheel_width"] / 2, wr, z)
            nm = ("Rear" if sz > 0 else "Front") + "Wheel" + ("R" if sx > 0 else "L")
            b.rod(nm, "tire", wr, start, end, taper=1.0, bevel=wr * 0.2, group="Wheels")
            b.rod(nm + "Rim", "rim", wr * 0.55, (x + sx * p["wheel_width"] / 2, wr, z), (x + sx * (p["wheel_width"] / 2 + 0.04), wr, z), taper=1.0, group="Wheels")
    for sx in (-1, 1):
        b.board("Headlight" + ("R" if sx > 0 else "L"), "light", (0.5, 0.22, 0.08), (sx * W * 0.32, body_lo + body_h * 0.62, -L / 2 + 0.04), 0.04, 0, group="Lights")


def _rounded_rect_ring(half_w: float, y0: float, y1: float, r: float, seg: int) -> np.ndarray:
    r = min(r, half_w * 0.9, (y1 - y0) * 0.45)
    pts = []
    corners = [(half_w - r, y0 + r, -np.pi / 2), (half_w - r, y1 - r, 0.0), (-half_w + r, y1 - r, np.pi / 2), (-half_w + r, y0 + r, np.pi)]
    for cx, cy, a0 in corners:
        for k in range(seg):
            a = a0 + (np.pi / 2) * k / (seg - 1)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.asarray(pts)


def P(default, lo, hi, label, kind="length", fit=False):
    return ParamDef(default, lo, hi, label, kind, fit)


TEMPLATES: Dict[str, Template] = {}


def _register(t: Template) -> None:
    TEMPLATES[t.name] = t


_register(Template(
    "bed", "Bed",
    {
        "width": P(m2s(1.05), 2.0, 12.0, "Frame width", fit=True),
        "length": P(m2s(2.10), 4.0, 14.0, "Frame length", fit=True),
        "leg_height": P(m2s(0.17), 0.15, 2.5, "Leg clearance", fit=True),
        "rail_height": P(m2s(0.14), 0.2, 1.5, "Rail height", fit=True),
        "rail_thickness": P(m2s(0.035), 0.06, 0.6, "Rail thickness"),
        "post_size": P(m2s(0.065), 0.12, 0.8, "Post size"),
        "headboard_height": P(m2s(1.0), 1.0, 7.0, "Headboard height", fit=True),
        "footboard_height": P(m2s(0.42), 0.5, 5.0, "Footboard height", fit=True),
        "headboard_slats": P(6, 0, 16, "Headboard slats", "count"),
        "slat_width": P(m2s(0.065), 0.08, 0.8, "Slat width"),
        "mattress_thickness": P(m2s(0.2), 0.3, 2.0, "Mattress thickness", fit=True),
        "mattress_drop": P(m2s(0.05), 0.0, 1.0, "Mattress sink into frame"),
        "duvet": P(1, 0, 1, "Duvet", "toggle"),
        "duvet_coverage": P(0.72, 0.3, 1.0, "Duvet coverage", "ratio"),
        "pillows": P(1, 0, 4, "Pillows", "count"),
        "pillow_width": P(m2s(0.62), 1.0, 4.0, "Pillow width"),
        "center_legs": P(1, 0, 1, "Centre support legs", "toggle"),
        "bevel": P(0.02, 0.0, 0.15, "Wood edge radius"),
    },
    [
        MaterialSlot("frame", "Bed frame", WOOD_FRAME, "wood", (0.85, 0.69, 0.48)),
        MaterialSlot("mattress", "Mattress / sheet", SOFT, "fabric", (0.93, 0.93, 0.92)),
        MaterialSlot("duvet", "Duvet", SOFT, "fabric", (0.95, 0.95, 0.94)),
        MaterialSlot("pillow", "Pillow", SOFT, "fabric", (0.96, 0.96, 0.95)),
    ],
    _bed, ("width", "headboard_height", "length"),
    "Frame with posts, rails, slatted headboard, footboard, mattress, duvet and pillows.",
))

_register(Template(
    "chair", "Chair",
    {
        "width": P(m2s(0.46), 1.0, 6.0, "Seat width", fit=True),
        "depth": P(m2s(0.48), 1.0, 6.0, "Seat depth", fit=True),
        "seat_height": P(m2s(0.46), 0.8, 4.0, "Seat height", fit=True),
        "seat_thickness": P(m2s(0.04), 0.06, 1.0, "Seat thickness", fit=True),
        "back_height": P(m2s(0.9), 1.5, 7.0, "Back height", fit=True),
        "back_rail": P(m2s(0.08), 0.08, 1.0, "Back top rail"),
        "back_slats": P(3, 0, 10, "Back slats", "count"),
        "slat_width": P(m2s(0.05), 0.06, 0.6, "Slat width"),
        "leg_size": P(m2s(0.04), 0.08, 0.6, "Leg thickness", fit=True),
        "leg_taper": P(0.8, 0.4, 1.0, "Leg taper", "ratio"),
        "round_legs": P(0, 0, 1, "Round legs", "toggle"),
        "apron_height": P(m2s(0.06), 0.0, 0.8, "Apron height"),
        "stretchers": P(0, 0, 1, "Stretchers", "toggle"),
        "cushion": P(0, 0, 1, "Upholstered seat", "toggle"),
        "arms": P(0, 0, 1, "Armrests", "toggle"),
        "arm_height": P(m2s(0.22), 0.4, 2.0, "Arm height above seat"),
        "bevel": P(0.02, 0.0, 0.12, "Edge radius"),
    },
    [
        MaterialSlot("frame", "Frame", WOOD_FRAME, "wood", (0.55, 0.38, 0.24)),
        MaterialSlot("seat", "Seat", WOOD_FRAME + ("fabric", "leather"), "wood", (0.6, 0.42, 0.27)),
    ],
    _chair, ("width", "back_height", "depth"),
    "Four legs, seat (board or cushion), slatted or panel back, optional arms and stretchers.",
))

_register(Template(
    "table", "Table",
    {
        "width": P(m2s(1.4), 1.5, 16.0, "Top width", fit=True),
        "depth": P(m2s(0.8), 1.5, 12.0, "Top depth", fit=True),
        "height": P(m2s(0.75), 1.0, 5.0, "Height", fit=True),
        "top_thickness": P(m2s(0.035), 0.06, 0.8, "Top thickness", fit=True),
        "leg_size": P(m2s(0.06), 0.08, 0.8, "Leg thickness", fit=True),
        "leg_taper": P(0.85, 0.4, 1.0, "Leg taper", "ratio"),
        "round_legs": P(0, 0, 1, "Round legs", "toggle"),
        "overhang": P(m2s(0.05), 0.0, 1.0, "Top overhang"),
        "apron_height": P(m2s(0.08), 0.0, 1.0, "Apron height"),
        "shelf": P(0, 0, 1, "Lower shelf", "toggle"),
        "round_top": P(0, 0, 1, "Round top", "toggle"),
        "pedestal": P(0, 0, 1, "Pedestal base", "toggle"),
        "bevel": P(0.02, 0.0, 0.12, "Edge radius"),
    },
    [
        MaterialSlot("top", "Table top", WOOD_FRAME + ("stone", "glass"), "wood", (0.62, 0.45, 0.3)),
        MaterialSlot("frame", "Legs and frame", WOOD_FRAME, "wood", (0.55, 0.4, 0.27)),
    ],
    _table, ("width", "height", "depth"),
    "Top with legs and apron, round or pedestal variants, optional lower shelf.",
))

_register(Template(
    "sofa", "Sofa",
    {
        "width": P(m2s(2.0), 3.0, 16.0, "Width", fit=True),
        "depth": P(m2s(0.9), 2.0, 6.0, "Depth", fit=True),
        "height": P(m2s(0.85), 1.5, 5.0, "Back height", fit=True),
        "seat_height": P(m2s(0.45), 0.8, 3.0, "Seat height", fit=True),
        "arm_width": P(m2s(0.18), 0.3, 1.5, "Arm width", fit=True),
        "arm_height": P(m2s(0.62), 1.0, 3.5, "Arm height", fit=True),
        "back_thickness": P(m2s(0.25), 0.4, 2.0, "Back thickness"),
        "leg_height": P(m2s(0.08), 0.0, 1.0, "Leg height"),
        "cushions": P(3, 1, 5, "Cushions", "count"),
        "bevel": P(0.03, 0.0, 0.12, "Edge radius"),
    },
    [
        MaterialSlot("body", "Upholstery", SOFT, "fabric", (0.45, 0.47, 0.5)),
        MaterialSlot("cushion", "Cushions", SOFT, "fabric", (0.5, 0.52, 0.55)),
        MaterialSlot("legs", "Legs", WOOD_FRAME, "wood", (0.3, 0.2, 0.12)),
    ],
    _sofa, ("width", "height", "depth"),
    "Upholstered base, arms, back frame, seat and back cushions, short legs.",
))

_register(Template(
    "cabinet", "Cabinet",
    {
        "width": P(m2s(0.9), 1.5, 14.0, "Width", fit=True),
        "depth": P(m2s(0.45), 1.0, 6.0, "Depth", fit=True),
        "height": P(m2s(0.85), 1.0, 10.0, "Height", fit=True),
        "panel_thickness": P(m2s(0.025), 0.06, 0.4, "Panel thickness"),
        "plinth_height": P(m2s(0.08), 0.0, 1.5, "Plinth / feet height", fit=True),
        "rows": P(3, 1, 8, "Drawer rows", "count"),
        "columns": P(1, 1, 4, "Columns", "count"),
        "handle_size": P(m2s(0.025), 0.0, 0.3, "Handle size"),
        "legs": P(0, 0, 1, "Feet instead of plinth", "toggle"),
        "bevel": P(0.015, 0.0, 0.1, "Edge radius"),
    },
    [
        MaterialSlot("body", "Carcass", WOOD_FRAME, "wood", (0.7, 0.55, 0.38)),
        MaterialSlot("front", "Fronts", WOOD_FRAME, "wood", (0.72, 0.57, 0.4)),
        MaterialSlot("handle", "Handles", ("metal", "wood", "plastic"), "metal", (0.75, 0.75, 0.72)),
        MaterialSlot("frame", "Feet", WOOD_FRAME, "wood", (0.4, 0.3, 0.2)),
    ],
    _cabinet, ("width", "height", "depth"),
    "Carcass with drawer/door grid, handles and plinth or feet.",
))

_register(Template(
    "shelf", "Shelf",
    {
        "width": P(m2s(0.8), 1.5, 14.0, "Width", fit=True),
        "depth": P(m2s(0.3), 0.6, 4.0, "Depth", fit=True),
        "height": P(m2s(1.8), 1.5, 14.0, "Height", fit=True),
        "panel_thickness": P(m2s(0.025), 0.06, 0.4, "Panel thickness"),
        "plinth_height": P(m2s(0.05), 0.0, 1.0, "Plinth"),
        "shelves": P(4, 0, 10, "Shelves", "count"),
        "back_panel": P(1, 0, 1, "Back panel", "toggle"),
        "bevel": P(0.01, 0.0, 0.08, "Edge radius"),
    },
    [
        MaterialSlot("body", "Body", WOOD_FRAME, "wood", (0.78, 0.66, 0.5)),
        MaterialSlot("back", "Back panel", WOOD_FRAME, "painted", (0.9, 0.9, 0.88)),
    ],
    _shelf, ("width", "height", "depth"),
    "Bookcase with sides, shelves and optional back panel.",
))

_register(Template(
    "lamp", "Lamp",
    {
        "height": P(m2s(0.55), 1.0, 8.0, "Height", fit=True),
        "base_radius": P(m2s(0.08), 0.15, 2.0, "Base radius", fit=True),
        "base_height": P(m2s(0.03), 0.05, 1.5, "Base height", fit=True),
        "stem_radius": P(m2s(0.012), 0.03, 0.4, "Stem radius"),
        "shade_bottom_radius": P(m2s(0.17), 0.2, 3.0, "Shade bottom radius", fit=True),
        "shade_top_radius": P(m2s(0.12), 0.1, 3.0, "Shade top radius", fit=True),
        "shade_height": P(m2s(0.22), 0.2, 3.0, "Shade height", fit=True),
        "bulb": P(1, 0, 1, "Glowing bulb", "toggle"),
    },
    [
        MaterialSlot("base", "Base and stem", ("metal", "ceramic", "wood", "plastic"), "metal", (0.3, 0.3, 0.3)),
        MaterialSlot("shade", "Shade", ("fabric", "plastic", "metal", "glass"), "fabric", (0.93, 0.9, 0.82)),
        MaterialSlot("bulb", "Bulb", ("emissive",), "emissive", (1.0, 0.93, 0.75)),
    ],
    _lamp, ("shade_bottom_radius", "height", "shade_bottom_radius"),
    "Turned base, stem, open lathed shade and emissive bulb.",
))

_register(Template(
    "house", "House",
    {
        "width": P(m2s(8.0), 8.0, 80.0, "Width", fit=True),
        "depth": P(m2s(6.0), 6.0, 60.0, "Depth", fit=True),
        "wall_height": P(m2s(3.0), 4.0, 40.0, "Wall height", fit=True),
        "roof_height": P(m2s(2.2), 1.0, 30.0, "Roof height", fit=True),
        "roof_overhang": P(m2s(0.35), 0.0, 4.0, "Roof overhang"),
        "door_width": P(m2s(1.0), 1.5, 8.0, "Door width"),
        "door_height": P(m2s(2.1), 3.0, 14.0, "Door height"),
        "windows": P(2, 0, 8, "Front windows", "count"),
        "window_size": P(m2s(1.0), 1.0, 8.0, "Window size"),
        "chimney": P(1, 0, 1, "Chimney", "toggle"),
    },
    [
        MaterialSlot("walls", "Walls", ("painted", "stone", "wood", "plastic"), "painted", (0.92, 0.88, 0.8)),
        MaterialSlot("roof", "Roof", ("painted", "stone", "wood", "metal"), "stone", (0.55, 0.22, 0.18)),
        MaterialSlot("door", "Door", WOOD_FRAME, "wood", (0.45, 0.28, 0.16)),
        MaterialSlot("glass", "Windows", ("glass",), "glass", (0.55, 0.7, 0.8)),
        MaterialSlot("trim", "Trim", ("painted", "stone", "wood"), "painted", (0.95, 0.95, 0.95)),
    ],
    _house, ("width", "wall_height", "depth"),
    "Walls, gable roof with overhang, door, windows and chimney.",
))

_register(Template(
    "car", "Car",
    {
        "length": P(m2s(4.3), 6.0, 40.0, "Length", fit=True),
        "width": P(m2s(1.8), 3.0, 16.0, "Width", fit=True),
        "height": P(m2s(1.45), 2.5, 14.0, "Height", fit=True),
        "wheel_radius": P(m2s(0.32), 0.5, 4.0, "Wheel radius", fit=True),
        "wheel_width": P(m2s(0.22), 0.3, 2.0, "Wheel width"),
        "ground_clearance": P(m2s(0.18), 0.2, 3.0, "Ground clearance"),
        "cabin_length": P(0.5, 0.25, 0.8, "Cabin length", "ratio", fit=True),
        "cabin_height": P(m2s(0.55), 0.8, 6.0, "Cabin height", fit=True),
        "cabin_offset": P(0.05, -0.3, 0.3, "Cabin offset", "ratio"),
        "hood_slope": P(0.18, 0.0, 0.5, "Hood slope", "ratio"),
    },
    [
        MaterialSlot("body", "Paint", ("metal", "painted", "plastic"), "painted", (0.75, 0.12, 0.1)),
        MaterialSlot("glass", "Windows", ("glass",), "glass", (0.2, 0.25, 0.3)),
        MaterialSlot("tire", "Tyres", ("rubber",), "rubber", (0.08, 0.08, 0.08)),
        MaterialSlot("rim", "Rims", ("metal",), "metal", (0.75, 0.75, 0.75)),
        MaterialSlot("light", "Lights", ("emissive", "glass"), "emissive", (1.0, 0.95, 0.85)),
    ],
    _car, ("width", "height", "length"),
    "Lofted body, cabin with glass, four wheels with rims and headlights.",
))


def get_template(name: str) -> Template:
    if name not in TEMPLATES:
        raise KeyError(f"unknown template '{name}'")
    return TEMPLATES[name]


def scale_params_to_dims(template: Template, params: Dict[str, float], target: Sequence[Optional[float]]) -> Dict[str, float]:
    """Scale all length parameters so the built object's bounding box matches ``target``.

    ``target`` is (x, y, z) in studs; ``None`` keeps that axis proportional.
    """
    probe = template.build(params, LOD(0.0)).bounds()
    ext = np.maximum(probe[1] - probe[0], 1e-6)
    factors = [None if t is None else float(t) / e for t, e in zip(target, ext)]
    known = [f for f in factors if f is not None]
    if not known:
        return dict(params)
    uniform = float(np.mean(known))
    fx, fy, fz = [uniform if f is None else f for f in factors]
    out = dict(params)
    axis_hint = _axis_hints(template)
    for k, d in template.params.items():
        if d.kind != "length":
            continue
        ax = axis_hint.get(k)
        f = {0: fx, 1: fy, 2: fz}.get(ax, (fx + fy + fz) / 3.0)
        out[k] = float(params.get(k, d.default)) * f
    return template.clamp(out)


def _axis_hints(template: Template) -> Dict[str, int]:
    hints = {}
    for k in template.params:
        lk = k.lower()
        if lk in ("width",) or "width" in lk and "slat" not in lk and "wheel" not in lk and "door" not in lk and "arm" not in lk and "pillow" not in lk:
            hints[k] = 0
        elif lk in ("length",):
            hints[k] = 2
        elif lk in ("depth",):
            hints[k] = 2
        elif "height" in lk or lk.endswith("_h"):
            hints[k] = 1
    return hints
