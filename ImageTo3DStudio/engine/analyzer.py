"""Classical image analysis: validation, segmentation and geometric features.

No learned models are used. Foreground extraction combines four classical cues:
  1. a background colour model fitted (k-means, CIELAB) to the image border,
  2. closure of strong Canny edges (regions enclosed by edges and not connected
     to the border are object candidates),
  3. shadow suppression (pixels with background chromaticity but lower
     brightness are treated as cast shadows),
  4. GrabCut graph-cut refinement (Rother et al. 2004) initialised from 1-3.
If the input has an alpha channel the alpha matte is used directly.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 64_000_000

WORK_MAX_SIDE = 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"}


class ImageValidationError(ValueError):
    pass


@dataclass
class LoadedImage:
    rgb: np.ndarray
    alpha: Optional[np.ndarray]
    original_size: Tuple[int, int]
    format: str
    exif_focal_35mm: Optional[float] = None


def load_image(path: str, max_pixels: int = 40_000_000, max_side: int = WORK_MAX_SIDE) -> LoadedImage:
    """Open an untrusted image safely and return a working-resolution copy."""
    try:
        with Image.open(path) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError(f"unsupported image format '{fmt or 'unknown'}'")
            w, h = probe.size
            if w * h > max_pixels:
                raise ImageValidationError(f"image too large ({w}x{h}); limit is {max_pixels // 1_000_000} MP")
            if min(w, h) < 48:
                raise ImageValidationError("image too small (minimum 48 px on the short side)")
            probe.verify()
        with Image.open(path) as im:
            focal = None
            try:
                exif = im.getexif()
                sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
                f35 = sub.get(0xA405) if sub else None
                if f35:
                    focal = float(f35)
            except Exception:
                focal = None
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
            im = im.convert("RGBA" if has_alpha else "RGB")
            scale = min(1.0, max_side / max(im.size))
            if scale < 1.0:
                im = im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))), Image.LANCZOS)
            arr = np.asarray(im)
    except ImageValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("image rejected as a decompression bomb") from exc
    except Exception as exc:
        raise ImageValidationError(f"not a readable image ({exc.__class__.__name__})") from exc
    rgb = arr[..., :3].copy()
    alpha = arr[..., 3].copy() if arr.shape[2] == 4 else None
    return LoadedImage(rgb, alpha, (w, h), fmt, focal)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@dataclass
class Segmentation:
    """Foreground estimate.

    ``mask`` is the best binary cut-out, ``filled`` the same without holes and
    ``zones`` a three-level map used by model fitting: 0 = certainly background,
    1 = undecided, 2 = certainly object. Undecided pixels (e.g. a white mattress in
    front of a white wall) never penalise a model hypothesis.
    """

    mask: np.ndarray
    filled: np.ndarray
    zones: np.ndarray
    confidence: float
    method: str
    background_lab: np.ndarray
    notes: List[str] = field(default_factory=list)


def _border_band(h: int, w: int, frac: float = 0.03) -> np.ndarray:
    b = max(3, int(round(frac * min(h, w))))
    band = np.zeros((h, w), bool)
    band[:b] = band[-b:] = True
    band[:, :b] = band[:, -b:] = True
    return band


def edge_map(rgb: np.ndarray) -> np.ndarray:
    """Canny edges on lightness and chroma with data-driven thresholds.

    A bilateral pre-filter removes JPEG block noise on flat walls while keeping
    object outlines; lightness thresholds come from the gradient-magnitude
    distribution, chroma thresholds are absolute CIELAB steps.
    """
    smooth = cv2.bilateralFilter(rgb, 7, 28, 7)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    L = cv2.GaussianBlur(lab[..., 0], (3, 3), 0)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    hi = float(max(70.0, np.percentile(mag, 88)))
    out = cv2.Canny(L, hi * 0.45, hi, L2gradient=True)
    for c in (1, 2):
        ch = cv2.GaussianBlur(lab[..., c], (3, 3), 0)
        out |= cv2.Canny(ch, 28, 60, L2gradient=True)
    return out


def _largest_components(mask: np.ndarray, keep_ratio: float = 0.08, seed: Optional[np.ndarray] = None) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    keep = np.zeros(n, bool)
    keep[1:] = areas >= keep_ratio * big
    if seed is not None:
        hits = np.bincount(labels[seed], minlength=n)
        keep[1:] &= hits[1:] > 0
        if not keep.any():
            keep[1 + int(np.argmax(areas))] = True
    return keep[labels]


def fill_holes(mask: np.ndarray, max_hole_ratio: float = 1.0) -> np.ndarray:
    m = mask.astype(np.uint8)
    h, w = m.shape
    inv = (1 - m).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy().astype(bool)
    total = max(int(mask.sum()), 1)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        touches = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches and area <= max_hole_ratio * total:
            out[labels == i] = True
    return out


def _hull_mask(mask: np.ndarray) -> np.ndarray:
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is not None and len(pts) >= 3:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def segment(img: LoadedImage, sensitivity: float = 0.5, debug: Optional[dict] = None) -> Segmentation:
    rgb = img.rgb
    h, w = rgb.shape[:2]
    notes: List[str] = []
    if img.alpha is not None and (img.alpha < 128).mean() > 0.02:
        mask = _largest_components(img.alpha >= 128, 0.02)
        zones = np.where(mask, 2, 0).astype(np.uint8)
        return Segmentation(mask, fill_holes(mask), zones, 0.98, "alpha matte", np.zeros((1, 3)), notes)

    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    band = _border_band(sh, sw)
    samples = lab[band]
    k = 4 if len(samples) > 400 else 2
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, lbl, centers = cv2.kmeans(samples, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lbl.ravel(), minlength=k) / len(lbl)
    centers = centers[share > 0.04]
    weights = np.array([0.55, 1.0, 1.0], np.float32)
    dist = np.min(np.linalg.norm((lab[:, :, None, :] - centers[None, None]) * weights, axis=3), axis=2)
    spread = float(np.percentile(dist[band], 90))
    plain = spread < 9.0 and float(share.max()) > 0.85
    edges = edge_map(small)

    rgbf = small.astype(np.float32) + 1.0
    chroma = rgbf / rgbf.sum(axis=2, keepdims=True)
    bright = rgbf.sum(axis=2)
    bg_rgb = cv2.cvtColor(centers.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32) + 1.0
    bg_chroma = bg_rgb / bg_rgb.sum(axis=1, keepdims=True)
    bg_bright = bg_rgb.sum(axis=1)
    bottom = np.zeros((sh, sw), bool)
    bottom[-max(3, sh // 30):] = True
    floor_ids = np.unique(np.argmin(np.linalg.norm((lab[bottom][:, None, :] - centers[None]) * weights, axis=2), axis=1))
    cd = np.min(np.linalg.norm(chroma[:, :, None, :] - bg_chroma[floor_ids][None, None], axis=3), axis=2)
    ratio = np.max(bright[:, :, None] / bg_bright[floor_ids][None, None], axis=2)
    lower = np.zeros((sh, sw), bool)
    lower[int(sh * 0.45):] = True
    shadow = (cd < 0.016) & (ratio > 0.25) & (ratio < 0.88) & lower

    thr = max(10.0, float(cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))
    thr *= 1.25 - 0.5 * float(np.clip(sensitivity, 0, 1))
    sure_fg = (dist > thr * (1.3 if plain else 1.6)) & ~shadow & ~band
    sure_fg = cv2.morphologyEx(sure_fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(sure_fg.astype(np.uint8), connectivity=8)
    if n_c > 1:
        keep = np.zeros(n_c, bool)
        keep[1:] = st_c[1:, cv2.CC_STAT_AREA] > 0.0015 * sh * sw
        sure_fg = keep[lab_c]
    if sure_fg.sum() < 0.002 * sh * sw:
        y0, x0 = int(sh * 0.3), int(sw * 0.3)
        sure_fg = np.zeros((sh, sw), bool)
        sure_fg[y0:sh - y0, x0:sw - x0] = True
        notes.append("Weak foreground cues; assumed the object sits in the centre of the frame.")
    barrier = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    floor_ab = lab[bottom][:, 1:]
    ab_med = np.median(floor_ab, axis=0)
    ab_tol = max(6.0, float(np.percentile(np.linalg.norm(floor_ab - ab_med, axis=1), 90)) * 1.6)
    floor_like = (np.linalg.norm(lab[..., 1:] - ab_med, axis=2) < ab_tol) & ~barrier & lower
    n_fl, lab_fl, _, _ = cv2.connectedComponentsWithStats(floor_like.astype(np.uint8), connectivity=4)
    seeds = np.unique(lab_fl[bottom])
    floor = np.isin(lab_fl, seeds[seeds > 0])
    floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & ~barrier
    if floor.mean() < 0.6:
        sure_fg &= ~floor
        shadow |= floor
    hull = _hull_mask(sure_fg)
    hull_d = cv2.dilate(hull.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    passable = (~barrier & (dist < thr)).astype(np.uint8)
    n_f, lab_f, _, _ = cv2.connectedComponentsWithStats(passable, connectivity=4)
    bl = np.unique(np.concatenate([lab_f[0], lab_f[-1], lab_f[:, 0], lab_f[:, -1]]))
    flood = np.isin(lab_f, bl[bl > 0])
    sure_bg = (flood & ~hull_d) | (band & (dist < thr)) | (floor & ~hull_d)
    if plain:
        sure_bg |= (dist < thr * 0.5) & ~hull
    zones = np.ones((sh, sw), np.uint8)
    zones[sure_bg] = 0
    zones[sure_fg] = 2

    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    gc[hull & ~sure_bg] = cv2.GC_PR_FGD
    gc[(dist > thr) & ~shadow & ~sure_bg] = cv2.GC_PR_FGD
    gc[cv2.erode(sure_bg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = cv2.GC_BGD
    gc[cv2.erode(sure_fg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = cv2.GC_FGD
    method = "plain-background graph cut" if plain else "zone-seeded graph cut"
    if debug is not None:
        debug.update(floor=floor, dist=dist, edges=edges, shadow=shadow, sure_fg=sure_fg, sure_bg=sure_bg, init=gc.copy(), plain=plain, spread=spread, thr=thr)
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
        fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    except cv2.error:
        fg = sure_fg | (hull & ~sure_bg)
        method = "zone threshold (graph cut failed)"
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    fg = _largest_components(fg, 0.06, seed=sure_fg)
    if fg.sum() < 0.002 * sh * sw:
        fg = sure_fg.copy()
        notes.append("Graph cut produced an empty mask; kept the colour-distinct region.")

    def up(m):
        return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1 else m.astype(np.uint8)

    mask = up(fg).astype(bool)
    zones_full = up(zones)
    filled = fill_holes(mask)
    boundary = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge_support = float((cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0)[boundary].mean()) if boundary.any() else 0.0
    decided = float((zones != 1).mean())
    coverage = float(fg.mean())
    conf = 0.45 * edge_support + 0.4 * decided + 0.15 * (1.0 if 0.03 < coverage < 0.9 else 0.0)
    if plain:
        conf = min(1.0, conf + 0.1)
    if coverage > 0.85:
        notes.append("The object fills almost the whole frame; leave some background around it for a cleaner cut-out.")
    if conf < 0.5:
        notes.append("Low segmentation confidence: a plain background or a PNG with transparency gives better results.")
    return Segmentation(mask, filled, zones_full, float(np.clip(conf, 0, 1)), method, centers, notes)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass
class Features:
    bbox: Tuple[int, int, int, int]
    aspect: float
    fill_ratio: float
    solidity: float
    hole_ratio: float
    hole_count: int
    symmetry: float
    symmetry_axis: float
    rectangularity: float
    vertical_lines: int
    horizontal_lines: int
    oblique_lines: int
    line_length_ratio: float
    color_count: int
    palette: List[Tuple[Tuple[float, float, float], float]]
    drawing_likeness: float
    contour: np.ndarray
    width_profile: np.ndarray

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox), "aspect": round(self.aspect, 4), "fill_ratio": round(self.fill_ratio, 4),
            "solidity": round(self.solidity, 4), "hole_ratio": round(self.hole_ratio, 4), "hole_count": self.hole_count,
            "symmetry": round(self.symmetry, 4), "rectangularity": round(self.rectangularity, 4),
            "vertical_lines": self.vertical_lines, "horizontal_lines": self.horizontal_lines,
            "oblique_lines": self.oblique_lines, "line_length_ratio": round(self.line_length_ratio, 4),
            "color_count": self.color_count, "drawing_likeness": round(self.drawing_likeness, 4),
            "palette": [{"color": [round(c, 4) for c in col], "share": round(s, 4)} for col, s in self.palette],
        }


def mirror_symmetry(mask: np.ndarray) -> Tuple[float, float]:
    """Best vertical mirror axis and its IoU score."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    x0, x1 = xs.min(), xs.max()
    best = (0.0, float((x0 + x1) / 2))
    for axis in np.linspace(x0 + (x1 - x0) * 0.35, x0 + (x1 - x0) * 0.65, 13):
        shift = int(round(2 * axis))
        flipped = np.zeros_like(mask)
        src_x = shift - np.arange(mask.shape[1])
        valid = (src_x >= 0) & (src_x < mask.shape[1])
        flipped[:, valid] = mask[:, src_x[valid]]
        inter = np.logical_and(mask, flipped).sum()
        union = np.logical_or(mask, flipped).sum()
        score = inter / max(union, 1)
        if score > best[0]:
            best = (float(score), float(axis))
    return best


def line_segments(gray: np.ndarray, min_len: float) -> np.ndarray:
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(gray)[0]
        segs = lines.reshape(-1, 4) if lines is not None else np.zeros((0, 4))
    except (cv2.error, AttributeError):
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=int(min_len), maxLineGap=4)
        segs = lines.reshape(-1, 4).astype(np.float64) if lines is not None else np.zeros((0, 4))
    if len(segs) == 0:
        return segs
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[length >= min_len]


def palette(rgb: np.ndarray, mask: np.ndarray, k: int = 5) -> List[Tuple[Tuple[float, float, float], float]]:
    px = rgb[mask].astype(np.float32)
    if len(px) < k:
        return [((0.7, 0.7, 0.7), 1.0)]
    if len(px) > 60000:
        px = px[np.random.default_rng(0).choice(len(px), 60000, replace=False)]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lbl, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k) / len(lbl)
    rgbc = cv2.cvtColor(centers.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3) / 255.0
    order = np.argsort(-counts)
    return [(tuple(float(c) for c in rgbc[i]), float(counts[i])) for i in order]


def compute_features(img: LoadedImage, seg: Segmentation) -> Features:
    mask = seg.mask
    filled = seg.filled
    ys, xs = np.nonzero(filled)
    if len(xs) == 0:
        raise ImageValidationError("no object found in the image")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    area = float(filled.sum())
    contours, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    rect = cv2.minAreaRect(cnt)
    rect_area = float(rect[1][0] * rect[1][1])
    holes = filled & ~mask
    n_h, _, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    hole_count = int(np.sum(st[1:, cv2.CC_STAT_AREA] > 0.002 * area)) if n_h > 1 else 0
    sym, axis = mirror_symmetry(filled)
    gray = cv2.cvtColor(img.rgb, cv2.COLOR_RGB2GRAY)
    segs = line_segments(gray, 0.06 * max(bw, bh))
    near = cv2.dilate(filled.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    v = h_ = o = 0
    total_len = 0.0
    for sx0, sy0, sx1, sy1 in segs:
        mx, my = int((sx0 + sx1) / 2), int((sy0 + sy1) / 2)
        if not (0 <= my < near.shape[0] and 0 <= mx < near.shape[1] and near[my, mx]):
            continue
        ang = np.degrees(np.arctan2(abs(sy1 - sy0), abs(sx1 - sx0)))
        total_len += np.hypot(sx1 - sx0, sy1 - sy0)
        if ang > 75:
            v += 1
        elif ang < 15:
            h_ += 1
        else:
            o += 1
    pal = palette(img.rgb, mask)
    significant = sum(1 for _, s in pal if s > 0.08)
    px = img.rgb[mask].astype(np.float32) / 255.0
    sat = np.max(px, 1) - np.min(px, 1)
    flatness = 1.0 - min(1.0, float(np.std(cv2.Laplacian(gray, cv2.CV_32F)[mask])) / 40.0)
    dark_outline = float(((gray < 50) & cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)).astype(bool)).mean() * 50)
    drawing = float(np.clip(0.5 * flatness + 0.3 * min(dark_outline, 1.0) + 0.2 * float(np.mean(sat > 0.35)), 0, 1))
    prof = np.array([filled[y, x0:x1 + 1].sum() for y in range(y0, y1 + 1)], dtype=np.float64) / max(bw, 1)
    return Features(
        (x0, y0, x1, y1), bw / max(bh, 1), area / (bw * bh), area / max(hull_area, 1.0),
        float(holes.sum()) / max(area, 1.0), hole_count, sym, axis, area / max(rect_area, 1.0),
        v, h_, o, total_len / max(bw + bh, 1), significant, pal, drawing, cnt.reshape(-1, 2), prof,
    )


def region_stats(rgb: np.ndarray, region: np.ndarray) -> Dict[str, float]:
    """Colour and texture statistics used to infer material kinds."""
    if region.sum() < 30:
        return {"count": int(region.sum())}
    px = rgb[region].astype(np.float64) / 255.0
    lin = np.where(px <= 0.04045, px / 12.92, ((px + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    keep = (lum > np.percentile(lum, 8)) & (lum < np.percentile(lum, 92))
    med_lin = np.median(lin[keep], axis=0) if keep.any() else np.median(lin, axis=0)
    med = np.where(med_lin <= 0.0031308, med_lin * 12.92, 1.055 * med_lin ** (1 / 2.4) - 0.055)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    inner = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if inner.sum() < 20:
        inner = region
    jxx, jyy, jxy = float((gx[inner] ** 2).mean()), float((gy[inner] ** 2).mean()), float((gx[inner] * gy[inner]).mean())
    tr = jxx + jyy
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tr) if tr > 1e-9 else 0.0
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    texture = float(np.std((gray - blur)[inner]))
    spec = float(np.mean(lum > np.percentile(lum, 50) * 2.2)) if lum.size else 0.0
    h, l, s = colorsys.rgb_to_hls(*[float(c) for c in med])
    return {
        "count": int(region.sum()), "color": [float(c) for c in med], "lightness": l, "saturation": s, "hue": h * 360,
        "texture": texture, "coherence": coherence, "specular": spec, "grain_angle": float(0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))),
    }


def rectify_quad(rgb: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Perspective-rectify an image quadrilateral (TL, TR, BR, BL) to a rectangle."""
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    hmat = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(rgb, hmat, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def periodic_count(profile: np.ndarray, lo: int = 2, hi: int = 16) -> Tuple[int, float]:
    """Estimate how many repeated bars a 1D intensity profile contains (FFT peak)."""
    p = np.asarray(profile, dtype=np.float64)
    if len(p) < 16:
        return 0, 0.0
    p = p - np.convolve(p, np.ones(max(3, len(p) // 6)) / max(3, len(p) // 6), mode="same")
    p *= np.hanning(len(p))
    spec = np.abs(np.fft.rfft(p))
    freqs = np.arange(len(spec))
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any() or spec[1:].sum() <= 0:
        return 0, 0.0
    k = int(freqs[band][np.argmax(spec[band])])
    strength = float(spec[k] / (np.median(spec[1:]) + 1e-9))
    return k, strength
