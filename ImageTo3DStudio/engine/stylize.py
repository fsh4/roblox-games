"""Roblox art styles applied on top of a reconstruction.

``realistic``  the photo-textured model (default).
``cartoon``    a handful of flat, saturated colours taken from the photo, no
               photographed light or shadow, no relief map, light occlusion.
``blocky``     the hand-built block look of many Roblox games: the solid is
               rebuilt from cubes, same-coloured cubes are grouped into
               slabs of up to 4 x 1 x 4 cubes with staggered seams, every
               block border is a dark groove with a bevel in the normal map,
               the body is exactly symmetric, and attached parts (a cone, a
               flower, a handle) get their own colours. No studs and no baked
               occlusion (it greys the colours).

Colours are clustered in CIELAB, near-duplicates (light and shadow of one
surface) are merged, small strongly different features (eyes) are kept as
accent colours, and everything is slightly saturated: two or three main
colours plus accents, as in the project's model guide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import geometry as g
from .materials import linear_to_srgb, make_material, srgb_to_linear
from .reconstruction import _assign_chart_uvs
from .scene import NativePrimitive, Part, Scene

ART_STYLES = ("realistic", "cartoon", "blocky")
CARTOON_COLOURS = 8
BLOCK_COLOURS = 6
ACCESSORY_COLOURS = 2
BLOCK_CHROMA = 1.45
MIN_CELLS = 10
PREFERRED_CELLS = 16
BOX_LIMITS = (4, 1, 4)
GROOVE = 0.07
MAX_NATIVE_PARTS = 600


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def _to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)


def vivid(lab: np.ndarray, chroma: float = 1.18, light: float = 4.0) -> np.ndarray:
    """Toy-like colour: chroma pushed outwards from grey, lightness lifted a little."""
    out = lab.astype(np.float32).copy()
    out[:, 1:] = 128 + (out[:, 1:] - 128) * chroma
    out[:, 0] = np.clip(out[:, 0] + light, 0, 255)
    return out


def palette(colours_u8: np.ndarray, k: int, weights: Optional[np.ndarray] = None, chroma: float = 1.18) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-means palette in CIELAB (lightness weighted low).

    Returns (vivid palette RGB u8, labels, plain palette centres in CIELAB).
    """
    lab = _to_lab(colours_u8)
    k = int(max(1, min(k, len(np.unique(colours_u8.reshape(-1, 3), axis=0)))))
    scale = np.array([0.45, 1.0, 1.0], np.float32)
    cv2.setRNGSeed(11)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3)
    data = lab * scale
    if weights is not None and len(data) > 4 * k:
        reps = np.clip(np.round(weights / max(weights.mean(), 1e-9)), 1, 6).astype(np.int64)
        data_fit = np.repeat(data, reps, axis=0)
    else:
        data_fit = data
    _, _, centres = cv2.kmeans(data_fit.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = np.argmin(np.linalg.norm(data[:, None, :] - centres[None], axis=2), axis=1)
    pal_lab = centres / scale
    return _from_lab(vivid(pal_lab, chroma)), labels, pal_lab


def with_accents(sample_u8: np.ndarray, pal_rgb: np.ndarray, centres: np.ndarray, extra: int = 3, min_share: float = 0.003, min_count: int = 20) -> tuple:
    """Add small, strongly coloured regions the main palette missed (a fruit, a button, a light).

    Pixels far from every palette colour and clearly saturated are clustered
    on their own; each such cluster that covers at least ``min_share`` of the
    object becomes an accent colour.
    """
    scale_w = np.array([0.45, 1.0, 1.0], np.float32)
    lab = _to_lab(sample_u8)
    dist = np.min(np.linalg.norm((lab[:, None, :] - centres[None]) * scale_w, axis=2), axis=1)
    chroma = np.linalg.norm(lab[:, 1:] - 128.0, axis=1)
    odd = (dist > 22.0) & ((chroma > 25.0) | (np.abs(lab[:, 0] - np.median(lab[:, 0])) > 70.0))
    if odd.sum() < max(min_count, min_share * len(lab)):
        return pal_rgb, centres
    k = int(min(extra, max(1, odd.sum() // max(min_count, int(min_share * len(lab))))))
    acc_rgb, acc_labels, acc_centres = palette(sample_u8[odd], k, chroma=1.25)
    keep = [i for i in range(len(acc_centres)) if (acc_labels == i).sum() >= max(min_count, min_share * len(lab))]
    if not keep:
        return pal_rgb, centres
    return np.concatenate([pal_rgb, acc_rgb[keep]]), np.concatenate([centres, acc_centres[keep]])


def flatten_lighting(rgb_u8: np.ndarray, inside: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Remove most of the photographed light and shadow (broad lightness changes)."""
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    ys, xs = np.nonzero(inside)
    if len(ys) == 0:
        return rgb_u8
    size = float(max(np.ptp(ys), np.ptp(xs)) + 1)
    broad = cv2.GaussianBlur(L, (0, 0), max(3.0, size / 8.0))
    target = float(np.median(L[inside]))
    lab[..., 0] = np.clip(L - strength * (broad - target), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def cartoonize(rgb_u8: np.ndarray, inside: np.ndarray, k: int = CARTOON_COLOURS, pal: Optional[tuple] = None) -> Tuple[np.ndarray, tuple]:
    """Flat-colour version of a photo: smoothed, light removed, mapped to a small palette.

    Returns (image, palette). Pass ``pal`` to reuse the palette of another
    image (the synthesised back uses the palette of the front).
    """
    h, w = rgb_u8.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(rgb_u8, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb_u8.copy()
    small_in = cv2.resize(inside.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    flat = flatten_lighting(small, small_in if small_in.any() else np.ones(small.shape[:2], bool), strength=0.6)
    smooth = cv2.pyrMeanShiftFiltering(flat, 10, 24)
    smooth = cv2.bilateralFilter(smooth, 9, 30, 9)
    px = smooth[small_in] if small_in.any() else smooth.reshape(-1, 3)
    if pal is None:
        sample = px if len(px) <= 60000 else px[np.random.default_rng(3).choice(len(px), 60000, replace=False)]
        _, first, centres = palette(sample, k)
        centres = merge_close(centres, np.bincount(first, minlength=len(centres)).astype(np.float64), limit=10.0)
        pal = with_accents(sample, _from_lab(vivid(centres)), centres)
    pal_rgb, centres = pal
    scale_w = np.array([0.6, 1.0, 1.0], np.float32)
    lab = _to_lab(smooth.reshape(-1, 3)) * scale_w
    labels = np.argmin(np.linalg.norm(lab[:, None, :] - (centres * scale_w)[None], axis=2), axis=1)
    labels = labels.reshape(small.shape[:2]).astype(np.uint8)
    labels = cv2.medianBlur(labels, 7)
    labels = cv2.medianBlur(labels, 5)
    full = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    full = cv2.medianBlur(full, 5)
    return pal_rgb[full], pal


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def guide_pitch(extent: np.ndarray) -> float:
    """Starting block size: ``PREFERRED_CELLS`` blocks along the longest side.

    ``choose_grid`` makes the blocks larger until the triangle budget fits.
    """
    return float(np.max(extent)) / PREFERRED_CELLS


def voxelize(meshes: List[g.Mesh], pitch: float, shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Solid occupancy of watertight meshes; returns (grid[x, y, z], centre of cell 0).

    Cell centres lie on multiples of ``pitch`` after moving the meshes by
    ``-shift`` along X, so a mirror plane at ``x = shift`` runs through a
    column of cell centres and mirrored cells line up exactly.
    """
    import trimesh

    verts = np.concatenate([m.vertices for m in meshes]) - np.array([shift, 0.0, 0.0])
    offs = np.cumsum([0] + [len(m.vertices) for m in meshes])[:-1]
    faces = np.concatenate([m.faces + o for m, o in zip(meshes, offs)])
    tm = trimesh.Trimesh(verts, faces, process=True)
    vox = tm.voxelized(pitch=pitch).fill()
    occ = np.asarray(vox.matrix, bool)
    origin = np.asarray(vox.transform)[:3, 3] + np.array([shift, 0.0, 0.0])
    return occ, origin


def fix_edge_contacts(occ: np.ndarray, rounds: int = 4) -> np.ndarray:
    """Fill cells so no two cubes touch only along an edge (keeps the surface manifold)."""
    occ = np.pad(occ, 1)
    for _ in range(rounds):
        fill = np.zeros_like(occ)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            c10 = np.roll(occ, -1, a)
            c01 = np.roll(occ, -1, b)
            c11 = np.roll(c10, -1, b)
            fill |= np.roll(occ & c11 & ~c10 & ~c01, 1, a)
            fill |= c10 & c01 & ~occ & ~c11
        fill &= ~occ
        if not fill.any():
            break
        occ |= fill
    return occ[1:-1, 1:-1, 1:-1]


_DIRS = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]


def _exposed(occ: np.ndarray, axis: int, sign: int) -> np.ndarray:
    pad = np.pad(occ, 1)
    nb = np.roll(pad, -sign, axis=axis)[1:-1, 1:-1, 1:-1]
    return occ & ~nb


def cube_surface(occ: np.ndarray, origin: np.ndarray, pitch: float) -> g.Mesh:
    """Outer faces of the cubes; vertices shared within one face direction (flat shading)."""
    verts: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    base = 0
    for axis, sign in _DIRS:
        cells = np.argwhere(_exposed(occ, axis, sign))
        if len(cells) == 0:
            continue
        u, v = [i for i in range(3) if i != axis]
        corner = cells.copy()
        corner[:, axis] += 1 if sign > 0 else 0
        quads = []
        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            c = corner.copy()
            c[:, u] += du
            c[:, v] += dv
            quads.append(c)
        quads = np.stack(quads, axis=1)
        keys, inv = np.unique(quads.reshape(-1, 3), axis=0, return_inverse=True)
        inv = inv.reshape(-1, 4) + base
        n = np.zeros(3)
        n[axis] = sign
        tri = np.concatenate([inv[:, [0, 1, 2]], inv[:, [0, 2, 3]]])
        if np.cross(np.eye(3)[u], np.eye(3)[v])[axis] * sign < 0:
            tri = tri[:, ::-1]
        verts.append(origin + (keys - 0.5) * pitch)
        faces.append(tri)
        normals.append(np.repeat(n[None], len(keys), axis=0))
        base += len(keys)
    mesh = g.Mesh(np.concatenate(verts), np.concatenate(faces).astype(np.int64), name="Blocks")
    mesh.normals = np.concatenate(normals)
    return mesh


def _estimate_triangles(occ: np.ndarray) -> int:
    return 2 * sum(int(_exposed(occ, a, s).sum()) for a, s in _DIRS)


def _mirror_index(origin: np.ndarray, pitch: float, plane: float) -> int:
    """Twice the index of the cell column the mirror plane runs through."""
    return int(round(2 * (plane - origin[0]) / pitch))


def mirror_cells(occ: np.ndarray, twice_centre: int) -> Tuple[np.ndarray, np.ndarray]:
    """Index of every cell's mirror partner along X (-1 where it falls outside the grid)."""
    i = np.arange(occ.shape[0])
    partner = twice_centre - i
    partner[(partner < 0) | (partner >= occ.shape[0])] = -1
    return i, partner


def symmetrise(occ: np.ndarray, keep: np.ndarray, twice_centre: int) -> np.ndarray:
    """Add the mirror image of the cells in ``keep`` (the body), so the body is exactly symmetric."""
    i, partner = mirror_cells(occ, twice_centre)
    out = occ.copy()
    ok = partner >= 0
    out[partner[ok]] |= (occ & keep)[i[ok]]
    return out


def choose_grid(meshes: List[g.Mesh], extent: np.ndarray, budget: int, plane: Optional[float] = None,
                body_of=None) -> Tuple[np.ndarray, np.ndarray, float]:
    pitch = guide_pitch(extent)
    for _ in range(16):
        occ, origin = voxelize(meshes, pitch, plane or 0.0)
        if plane is not None:
            keep = body_of(occ, origin, pitch) if body_of is not None else occ
            occ = symmetrise(occ, keep, _mirror_index(origin, pitch, plane))
        occ = fix_edge_contacts(occ)
        if _estimate_triangles(occ) <= 0.97 * budget or pitch > float(np.max(extent)) / MIN_CELLS:
            return occ, origin, pitch
        pitch *= 1.12
    return occ, origin, pitch


def block_colours(tree: cKDTree, srgb: np.ndarray, faces: np.ndarray, total: int, contrast: float = 0.3, share: float = 0.15) -> np.ndarray:
    """Colour of each block from the texture samples under its outer face.

    The median colour is used, except when a clearly lighter or darker part
    (an eye, a mouth, an outline) covers at least ``share`` of the face: then
    the feature wins, because a block model draws an eye with one block.
    """
    k = int(min(160, total))
    _, idx = tree.query(faces, k=k)
    cols = srgb[idx]
    med = np.median(cols, axis=1)
    diff = cols.mean(axis=2) - med.mean(axis=1, keepdims=True)
    out = med.copy()
    for sign in (-1.0, 1.0):
        sel = (diff * sign) > contrast
        pick = sel.mean(axis=1) >= share
        if pick.any():
            masked = np.where(sel[..., None], cols, np.nan)
            out[pick] = np.nanmedian(masked[pick], axis=1)
    return out


def tidy_labels(labels: np.ndarray, mask: np.ndarray, centres: np.ndarray, passes: int = 3, similar: float = 42.0) -> np.ndarray:
    """Majority filter over each block's 26 neighbours inside ``mask``.

    A block takes its neighbours' colour only when the two colours are close
    (light and shadow of one surface); a clearly different colour (an eye, a
    button) is a feature and stays.
    """
    from scipy.ndimage import convolve

    count = len(centres)
    w = np.array([0.45, 1.0, 1.0])
    close = np.linalg.norm((centres[:, None, :] - centres[None]) * w, axis=2) < similar
    kernel = np.ones((3, 3, 3), np.float32)
    kernel[1, 1, 1] = 3.0
    out = labels.copy()
    for _ in range(passes):
        votes = np.stack([convolve(((out == c) & mask).astype(np.float32), kernel, mode="constant") for c in range(count)])
        best = np.argmax(votes, axis=0)
        cur = np.clip(out, 0, count - 1)
        change = mask & (out >= 0) & close[cur, best]
        out = np.where(change, best, out)
    return out


def merge_close(centres: np.ndarray, counts: np.ndarray, limit: float = 16.0) -> np.ndarray:
    """Merge palette colours closer than ``limit`` (CIELAB, lightness weighted low)."""
    w = np.array([0.45, 1.0, 1.0])
    cs = [c.astype(np.float64) for c in centres]
    ns = [float(n) for n in counts]
    while len(cs) > 1:
        arr = np.array(cs)
        d = np.linalg.norm((arr[:, None, :] - arr[None]) * w, axis=2)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] >= limit:
            break
        total = ns[i] + ns[j]
        cs[i] = (cs[i] * ns[i] + cs[j] * ns[j]) / max(total, 1e-9)
        ns[i] = total
        del cs[j], ns[j]
    return np.array(cs, np.float32)


def nearest_labels(colours_u8: np.ndarray, centres: np.ndarray) -> np.ndarray:
    w = np.array([0.45, 1.0, 1.0], np.float32)
    lab = _to_lab(colours_u8)
    return np.argmin(np.linalg.norm((lab[:, None, :] - centres[None]) * w, axis=2), axis=1)


def region_palette(colours_u8: np.ndarray, k: int, accents: bool) -> np.ndarray:
    """Palette (CIELAB centres) for one region: clustered, near-duplicates merged, accents added."""
    if len(colours_u8) == 0:
        return np.zeros((0, 3), np.float32)
    _, first, centres = palette(colours_u8, min(k, len(colours_u8)), chroma=BLOCK_CHROMA)
    centres = merge_close(centres, np.bincount(first, minlength=len(centres)).astype(np.float64), limit=20.0)
    if accents:
        _, centres = with_accents(colours_u8, _from_lab(vivid(centres, BLOCK_CHROMA)), centres, min_count=2)
    used = np.unique(nearest_labels(colours_u8, centres))
    return centres[used]


def block_boxes(occ: np.ndarray, labels: np.ndarray, limits: Tuple[int, int, int] = BOX_LIMITS) -> np.ndarray:
    """Group same-colour cubes into blocks of at most ``limits`` cubes; returns a block id per cell.

    Blocks are grown from cells in a fixed order, alternating the order of the
    horizontal axes per layer so seams stagger like stacked bricks.
    """
    ids = np.full(occ.shape, -1, np.int64)
    lab = np.where(occ, labels, -1)
    next_id = 0
    for cell in np.argwhere(occ):
        i, j, k = cell
        if ids[i, j, k] >= 0:
            continue
        c = lab[i, j, k]
        free = (ids == -1) & (lab == c)
        ax = (0, 2) if j % 2 == 0 else (2, 0)
        lo = [i, j, k]
        hi = [i, j, k]
        for axis in (ax[0], ax[1], 1):
            while hi[axis] - lo[axis] + 1 < limits[axis] and hi[axis] + 1 < occ.shape[axis]:
                sl = [slice(lo[a], hi[a] + 1) for a in range(3)]
                sl[axis] = slice(hi[axis] + 1, hi[axis] + 2)
                if not free[tuple(sl)].all():
                    break
                hi[axis] += 1
        ids[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = next_id
        next_id += 1
    return ids


@dataclass
class BlockLook:
    """Colour and relief lookup for the block model.

    Every point takes the colour of its cube; near the border of its block it
    is darkened and lowered (a groove between blocks), and just inside the
    groove slightly lifted, which reads as a bevelled edge in the normal map.
    """

    tree: cKDTree
    colours_lin: np.ndarray
    box_lo: np.ndarray
    box_hi: np.ndarray
    box_of: np.ndarray
    pitch: float
    centres: Optional[np.ndarray] = None
    eye: Optional[np.ndarray] = None

    def _edge(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, idx = self.tree.query(pos, k=1)
        b = self.box_of[idx]
        d = np.minimum(pos - self.box_lo[b], self.box_hi[b] - pos)
        d = np.sort(np.abs(d), axis=1)
        return idx, d[:, 1] / self.pitch

    def sample(self, pos: np.ndarray, front=None, back=None) -> np.ndarray:
        idx, e = self._edge(pos)
        groove = np.clip((e - GROOVE) / GROOVE, 0.0, 1.0)
        shade = 0.55 + 0.45 * groove
        lift = 1.0 + 0.10 * np.clip(1.0 - np.abs(e - 2.5 * GROOVE) / (1.5 * GROOVE), 0.0, 1.0)
        out = np.clip(self.colours_lin[idx] * (shade * lift)[:, None], 0.0, 1.0)
        if self.eye is not None and self.eye.any():
            out[self._glint(pos, idx)] = 1.0
        return out

    def _glint(self, pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Eye highlight: a large and a small white square on the front face of an eye block.

        Seen from the front, image left is +X, so the large square sits at the
        upper left of the face (+X, +Y) and the small one below and right of it.
        """
        local = (pos - self.centres[idx]) / self.pitch
        front = self.eye[idx] & (local[:, 2] < -0.45)
        x, y = local[:, 0], local[:, 1]
        big = (x > 0.08) & (x < 0.36) & (y > 0.08) & (y < 0.36)
        small = (x > -0.06) & (x < 0.06) & (y > -0.06) & (y < 0.06)
        return front & (big | small)

    def height(self, pos: np.ndarray) -> np.ndarray:
        _, e = self._edge(pos)
        return -self.pitch * 0.06 * (1.0 - np.clip(e / (3.0 * GROOVE), 0.0, 1.0)) ** 2

    def move(self, shift: np.ndarray) -> None:
        shift = np.asarray(shift, dtype=np.float64)
        self.tree = cKDTree(self.tree.data + shift)
        self.box_lo = self.box_lo + shift
        self.box_hi = self.box_hi + shift
        if self.centres is not None:
            self.centres = self.centres + shift


def blocky_scene(scene: Scene, sample_pos: np.ndarray, sample_lin: np.ndarray, budget: int, name: str = "Model",
                 plane: Optional[float] = None, region_of=None) -> Tuple[Scene, BlockLook, List[str]]:
    """Rebuild ``scene`` from coloured blocks, like hand-built Roblox block models.

    ``plane`` is the X position of the object's mirror plane (the body is then
    made exactly symmetric in shape and, when the photo agrees, in colour).
    ``region_of(points)`` returns 0 for body points and 1..n for accessories;
    every region gets its own colours so a cone or a fruit is not painted in
    the body's colours.
    """
    meshes = [p.mesh for p in scene.parts]
    lo, hi = scene.bounds()

    def body_of(occ, origin, pitch):
        if region_of is None:
            return occ
        cells = np.argwhere(occ)
        out = np.zeros_like(occ)
        out[tuple(cells.T)] = region_of(origin + cells * pitch) == 0
        return out

    occ, origin, pitch = choose_grid(meshes, hi - lo, budget, plane, body_of)
    cells = np.argwhere(occ)
    centres = origin + cells * pitch
    region = np.zeros(occ.shape, np.int64)
    if region_of is not None:
        region[tuple(cells.T)] = np.maximum(region_of(centres), 0)
    surface = np.zeros(occ.shape, bool)
    for a, s in _DIRS:
        surface |= _exposed(occ, a, s)
    shell = np.argwhere(surface)
    outward = np.zeros((len(shell), 3))
    for a, sgn in _DIRS:
        outward[_exposed(occ, a, sgn)[tuple(shell.T)], a] += sgn
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
    srgb = np.clip(linear_to_srgb(sample_lin), 0, 1)
    shell_u8 = (block_colours(cKDTree(sample_pos), srgb, origin + shell * pitch + outward * 0.5 * pitch, len(sample_pos)) * 255).astype(np.uint8)
    shell_region = region[tuple(shell.T)]
    labels = np.full(occ.shape, -1, np.int64)
    pal_lab = []
    for r in np.unique(shell_region):
        sel = shell_region == r
        cent = region_palette(shell_u8[sel], BLOCK_COLOURS if r == 0 else ACCESSORY_COLOURS, accents=(r == 0))
        lab_r = nearest_labels(shell_u8[sel], cent) + len(pal_lab)
        labels[tuple(shell[sel].T)] = lab_r
        pal_lab.extend(list(cent))
    pal_lab, labels = unify_palette(np.array(pal_lab, np.float32), labels)
    for r in np.unique(shell_region):
        labels = np.where(region == r, tidy_labels(labels, surface & (region == r), pal_lab), labels)
    mirrored = 0
    if plane is not None:
        mirrored = _mirror_colours(labels, surface & (region == 0), _mirror_index(origin, pitch, plane))
    inner = occ & ~surface
    if inner.any():
        _, nn = cKDTree(origin + shell * pitch).query(origin + np.argwhere(inner) * pitch, k=1)
        labels[tuple(np.argwhere(inner).T)] = labels[tuple(shell.T)][nn]
    pal = _from_lab(vivid(pal_lab, BLOCK_CHROMA))
    pal_lin = srgb_to_linear(pal.astype(np.float64) / 255.0)
    ids = block_boxes(occ, labels)
    nbox = int(ids.max()) + 1
    box_lo = np.full((nbox, 3), np.inf)
    box_hi = np.full((nbox, 3), -np.inf)
    cell_lo, cell_hi = centres - 0.5 * pitch, centres + 0.5 * pitch
    cell_box = ids[tuple(cells.T)]
    np.minimum.at(box_lo, cell_box, cell_lo)
    np.maximum.at(box_hi, cell_box, cell_hi)
    eyes = find_eyes(occ, labels, pal_lab, region, _mirror_index(origin, pitch, plane) if plane is not None else None)
    look = BlockLook(cKDTree(centres), pal_lin[labels[tuple(cells.T)]], box_lo, box_hi, cell_box, pitch, centres.copy(), eyes[tuple(cells.T)])
    body = cube_surface(occ, origin, pitch)
    _assign_chart_uvs(body)
    mats = {"surface": make_material("surface", "plastic", (0.7, 0.7, 0.7), name="Block colours (texture)", source="one colour per block")}
    for i, c in enumerate(pal):
        mats[f"block{i + 1}"] = make_material(f"block{i + 1}", "plastic", tuple(float(x) / 255 for x in c), name=f"Block colour {i + 1}", source="clustered from the photo")
    box_label = np.zeros(nbox, np.int64)
    box_label[cell_box] = labels[tuple(cells.T)]
    natives = [NativePrimitive("Block", tuple((box_hi[b] - box_lo[b]).tolist()), g.translation((box_lo[b] + box_hi[b]) / 2), f"block{box_label[b] + 1}", f"Block{b + 1}")
               for b in range(min(nbox, MAX_NATIVE_PARTS))]
    out = Scene(name, [Part("Blocks", body, "surface", np.eye(4), 0, "u", 1, True, natives)], mats, {"strategy": "blocky", "smooth_surface": False})
    hexes = ", ".join("#%02X%02X%02X" % tuple(int(x) for x in c) for c in pal)
    notes = [f"Block style: {len(cells)} cubes of {pitch:.2f} studs grouped into {nbox} blocks, {len(pal)} colours from the photo ({hexes})."]
    if eyes.any():
        n_eyes = int(cv2.connectedComponents(eyes.any(axis=2).astype(np.uint8).T)[0] - 1)
        notes.append(f"{n_eyes} eye{'s' if n_eyes != 1 else ''} found; each got a white highlight.")
    if plane is not None:
        notes.append("The body is built symmetric" + (f"; {mirrored} block colours were matched to their mirror partner." if mirrored else "."))
    if region_of is not None and int(region.max()) > 0:
        notes.append(f"{int(region.max())} attached part(s) got their own colours.")
    if nbox > MAX_NATIVE_PARTS:
        notes.append(f"The native-part model was limited to {MAX_NATIVE_PARTS} blocks.")
    return out, look, notes


def unify_palette(pal_lab: np.ndarray, labels: np.ndarray, limit: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """Give regions one shared colour where their colours are practically the same."""
    w = np.array([0.45, 1.0, 1.0])
    target = np.arange(len(pal_lab))
    for j in range(len(pal_lab)):
        for i in range(j):
            if target[i] == i and np.linalg.norm((pal_lab[i] - pal_lab[j]) * w) < limit:
                target[j] = i
                break
    keep = np.unique(target)
    remap = np.searchsorted(keep, target)
    out = np.where(labels >= 0, remap[np.clip(labels, 0, None)], labels)
    return pal_lab[keep], out


def find_eyes(occ: np.ndarray, labels: np.ndarray, pal_lab: np.ndarray, region: np.ndarray, twice_centre: Optional[int] = None,
              max_cells: int = 6, surround: float = 0.55) -> np.ndarray:
    """Front-facing blocks that carry an eye highlight.

    Eyes are two small spots (at most ``max_cells`` blocks each, seen from
    the front) whose colour is far darker than the most common body colour,
    at the same height in the upper two thirds of the object, each mostly
    surrounded by the body colour, and mirror images of each other when the
    body has a mirror plane. Only the upper-left block (+X, +Y, as seen from
    the front) of each eye is returned, so every eye gets one highlight.
    """
    eyes = np.zeros(occ.shape, bool)
    front = _exposed(occ, 2, -1) & (region == 0)
    if not front.any():
        return eyes
    lab_front = labels[front]
    main = np.bincount(lab_front[lab_front >= 0]).argmax()
    dark = (pal_lab[:, 0] < 0.55 * pal_lab[main, 0]) & (pal_lab[:, 0] < 95)
    cand = front & dark[np.clip(labels, 0, len(pal_lab) - 1)] & (labels >= 0)
    ys = np.flatnonzero(occ.any(axis=(0, 2)))
    if len(ys) == 0:
        return eyes
    cand[:, : ys[0] + (ys[-1] - ys[0]) // 3, :] = False
    face = cand.any(axis=2)
    first = np.argmax(front, axis=2)
    face_label = np.where(front.any(axis=2), labels[np.arange(occ.shape[0])[:, None], np.arange(occ.shape[1])[None, :], first], -1)
    count, comp, stats, cen = cv2.connectedComponentsWithStats(face.astype(np.uint8), connectivity=4)
    spots = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] > max_cells:
            continue
        ring = (cv2.dilate((comp == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (comp != i)
        if ring.sum() == 0 or (face_label[ring] == main).mean() < surround:
            continue
        spots.append((i, cen[i][0], cen[i][1]))
    pairs = set()
    for a_i, (ia, ya, xa) in enumerate(spots):
        for ib, yb, xb in spots[a_i + 1:]:
            level = abs(ya - yb) <= 1.0 and abs(xa - xb) >= 2.0
            mirrored = twice_centre is None or abs((xa + xb) - twice_centre) <= 1.5
            if level and mirrored:
                pairs.update((ia, ib))
    for i in pairs:
        cells = np.argwhere(comp == i)
        x, y = cells[np.lexsort((-cells[:, 0], -cells[:, 1]))[0]]
        eyes[x, y] = cand[x, y]
    return eyes


def _mirror_colours(labels: np.ndarray, mask: np.ndarray, twice_centre: int, agree: float = 0.7) -> int:
    """Make mirror partners share a colour when the photo's colours are already mostly symmetric.

    Where partners differ, the rarer colour wins (an eye seen on one side
    becomes a pair of eyes). Returns the number of changed blocks.
    """
    i, partner = mirror_cells(labels, twice_centre)
    ok = partner >= 0
    a = labels[i[ok]]
    b = labels[partner[ok]]
    both = mask[i[ok]] & mask[partner[ok]]
    if not both.any():
        return 0
    same = (a == b)[both].mean()
    if same < agree:
        return 0
    counts = np.bincount(labels[mask], minlength=int(labels.max()) + 1)
    rare = np.where(counts[np.clip(a, 0, None)] <= counts[np.clip(b, 0, None)], a, b)
    changed = both & (a != b)
    a_new = np.where(changed, rare, a)
    labels[i[ok]] = np.where(both, a_new, labels[i[ok]])
    labels[partner[ok]] = np.where(both, a_new, labels[partner[ok]])
    return int(changed.sum())


# ---------------------------------------------------------------------------
# Cartoon
# ---------------------------------------------------------------------------

def _u8(img_lin: np.ndarray) -> np.ndarray:
    return (np.clip(linear_to_srgb(np.clip(img_lin[..., :3], 0, None)), 0, 1) * 255 + 0.5).astype(np.uint8)


def _lin(img_u8: np.ndarray) -> np.ndarray:
    return srgb_to_linear(img_u8.astype(np.float64) / 255.0)


def cartoon_projection(photo) -> None:
    """Replace every photo of a projection (and a synthesised back) by its flat-colour version."""
    shared = None
    done = {}
    for view in photo.views():
        inside = view.mask > 0 if view.mask is not None else np.ones(view.image_lin.shape[:2], bool)
        key = id(view.image_lin)
        if key not in done:
            done[key] = cartoonize(_u8(view.image_lin), inside, pal=shared)
            shared = done[key][1]
        flat = done[key][0]
        sur = getattr(view, "surround", None)
        view.image_lin = _lin(flat)
        view.relief = None
        view.relief_back = None
        if sur is not None:
            back, _ = cartoonize(_u8(sur.back), np.ones(sur.back.shape[:2], bool), pal=shared)
            sur.front = view.image_lin
            sur.back = _lin(back).astype(np.float32)


def flatten_projection(photo, strength: float = 0.8) -> None:
    """Remove most of the photographed light and shadow from every photo of a projection.

    Used before block colours are sampled, so one surface in light and in
    shadow gives one block colour, not two.
    """
    done = {}
    for view in photo.views():
        inside = view.mask > 0 if view.mask is not None else np.ones(view.image_lin.shape[:2], bool)
        key = id(view.image_lin)
        if key not in done:
            done[key] = _lin(flatten_lighting(_u8(view.image_lin), inside, strength))
        view.image_lin = done[key]
        sur = getattr(view, "surround", None)
        if sur is not None:
            sur.front = view.image_lin
            sur.back = _lin(flatten_lighting(_u8(sur.back), np.ones(sur.back.shape[:2], bool), strength)).astype(np.float32)


def cartoon_materials(scene: Scene) -> None:
    """Procedural materials (wood, fabric, metal ...) become flat, vivid plastic of the same colour."""
    for key, mat in list(scene.materials.items()):
        if mat.kind in ("photo", "glass", "emissive"):
            continue
        rgb = (np.clip(np.asarray(mat.base_color), 0, 1) * 255).astype(np.uint8).reshape(1, 3)
        bright = _from_lab(vivid(_to_lab(rgb))).reshape(3) / 255.0
        scene.materials[key] = make_material(key, "plastic", tuple(float(c) for c in bright), name=mat.name, source=f"cartoon style ({mat.kind})")
