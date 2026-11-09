"""Preview renders of the exported asset (orthographic views, 3/4 view,
wireframe, turntable GIF, contact sheet) and a local interactive viewer."""
from __future__ import annotations

import base64
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import render
from engine.geometry import Mesh, merge

ACCENT = (240, 168, 64)
INK = (226, 232, 240)
MUTED = (140, 152, 168)
PANEL = (22, 27, 34)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _materials(maps: Dict[str, np.ndarray]) -> Optional[render.Materials]:
    if "basecolor" not in maps:
        return None
    f = lambda k: (maps[k].astype(np.float64) / 255.0) if k in maps else None
    bc = f("basecolor")
    return render.Materials(bc[..., :3], f("normal"), f("roughness"), f("metallic"), f("emissive"))


def _render_view(mesh: Mesh, mats, azimuth: float, elevation: float, size: Tuple[int, int], ortho: bool, ss: int = 2, wire: bool = False) -> Image.Image:
    lo, hi = mesh.bounds()
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    w, h = size[0] * ss, size[1] * ss
    if ortho:
        cam = render.orbit_camera(center, radius * 4, azimuth, elevation, w, h, ortho_extent=radius * 2.35)
    else:
        fov = 32.0
        dist = radius / np.tan(np.radians(fov) / 2) * 1.12
        cam = render.orbit_camera(center, dist, azimuth, elevation, w, h, fov)
    img = render.shade(cam, mesh.vertices, mesh.faces, mesh.normals, mesh.uvs, mats, floor_y=float(lo[1]))
    if wire:
        img = render.wireframe_overlay(img, cam, mesh.vertices, mesh.faces, color=(0.98, 0.72, 0.30), alpha=0.75)
    pil = _to_img(img)
    return pil.resize(size, Image.LANCZOS) if ss > 1 else pil


def _label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    f = _font(max(12, img.height // 26))
    pad = max(6, img.height // 60)
    d.rectangle([pad, pad, pad + 4, pad + f.size + 4], fill=ACCENT)
    d.text((pad + 12, pad), text.upper(), font=f, fill=INK)
    return img


def render_previews(meshes: Sequence[Tuple[str, Mesh]], maps: Dict[str, np.ndarray], out_dir: str, title: str, stats_lines: List[str], turntable: bool = True) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    merged = merge([m for _, m in meshes], title)
    mats = _materials(maps)
    paths = {}
    views = [("front", 0.0, 6.0, True), ("side", 90.0, 6.0, True), ("back", 180.0, 6.0, True), ("three_quarter", 38.0, 22.0, False)]
    tiles = {}
    for name, az, el, ortho in views:
        img = _render_view(merged, mats, az, el, (640, 640), ortho)
        tiles[name] = img
        out = _label(img.copy(), name.replace("_", " "))
        p = os.path.join(out_dir, f"{name}.png")
        out.save(p, optimize=True)
        paths[name] = p
    wire = _label(_render_view(merged, mats, 38.0, 22.0, (640, 640), False, wire=True), "wireframe")
    p = os.path.join(out_dir, "wireframe.png")
    wire.save(p, optimize=True)
    paths["wireframe"] = p
    sheet = contact_sheet(tiles, title, stats_lines)
    p = os.path.join(out_dir, "preview.png")
    sheet.save(p, optimize=True)
    paths["preview"] = p
    if turntable:
        frames = []
        for k in range(24):
            fr = _render_view(merged, mats, k * 15.0, 18.0, (320, 320), False, ss=1)
            frames.append(fr.convert("P", palette=Image.ADAPTIVE, colors=200))
        p = os.path.join(out_dir, "turntable.gif")
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True, disposal=2)
        paths["turntable"] = p
    return paths


def contact_sheet(tiles: Dict[str, Image.Image], title: str, stats_lines: List[str]) -> Image.Image:
    cell = 360
    width = cell * 3
    header = 64
    sheet = Image.new("RGB", (width, header + cell * 2), PANEL)
    d = ImageDraw.Draw(sheet)
    d.rectangle([0, 0, width, header], fill=(15, 19, 25))
    d.rectangle([20, 18, 26, 46], fill=ACCENT)
    d.text((38, 12), title, font=_font(24), fill=INK)
    d.text((38, 42), "IMAGE-TO-3D STUDIO  |  procedural modelling + classical vision", font=_font(12), fill=MUTED)
    big = tiles["three_quarter"].resize((cell * 2, cell * 2), Image.LANCZOS)
    sheet.paste(_label(big, "three quarter"), (0, header))
    for i, name in enumerate(("front", "side")):
        sheet.paste(_label(tiles[name].resize((cell, cell), Image.LANCZOS), name), (cell * 2, header + i * cell))
    overlay = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    f = _font(13)
    line_h = 18
    box_h = 14 + line_h * len(stats_lines)
    od.rectangle([0, sheet.height - box_h, cell * 2, sheet.height], fill=(10, 13, 18, 190))
    y = sheet.height - box_h + 7
    for line in stats_lines:
        od.text((14, y), line, font=f, fill=INK + (255,))
        y += line_h
    return Image.alpha_composite(sheet.convert("RGBA"), overlay).convert("RGB")


VIEWER_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - 3D preview</title>
<style>
:root {{ --bg: #0f1319; --panel: #161b22; --ink: #e2e8f0; --muted: #8c98a8; --accent: #f0a840; }}
html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--ink); font: 14px/1.4 system-ui, sans-serif; }}
#bar {{ position: fixed; inset: 0 0 auto 0; display: flex; gap: 16px; align-items: center; padding: 12px 16px; background: var(--panel); border-bottom: 2px solid var(--accent); }}
#bar b {{ font-size: 16px; }} #bar span {{ color: var(--muted); }}
#hint {{ position: fixed; bottom: 12px; left: 16px; color: var(--muted); }}
canvas {{ display: block; }}
</style>
</head>
<body>
<div id="bar"><b>{title}</b><span>{stats}</span></div>
<div id="hint">Drag to orbit, scroll or pinch to zoom. Needs an internet connection for the viewer library.</div>
<script type="importmap">{{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ GLTFLoader }} from "three/addons/loaders/GLTFLoader.js";
const glb = Uint8Array.from(atob("{glb}"), c => c.charCodeAt(0)).buffer;
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1319);
scene.add(new THREE.HemisphereLight(0xdde6f0, 0x2a2420, 1.4));
const key = new THREE.DirectionalLight(0xfff2e0, 2.4); key.position.set(-4, 8, -6); scene.add(key);
const camera = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.01, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
new GLTFLoader().parse(glb, "", gltf => {{
  const root = gltf.scene; scene.add(root);
  const box = new THREE.Box3().setFromObject(root), size = box.getSize(new THREE.Vector3()), mid = box.getCenter(new THREE.Vector3());
  const radius = size.length() / 2;
  const vfov = THREE.MathUtils.degToRad(camera.fov), hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const dist = radius / Math.sin(Math.min(vfov, hfov) / 2) * 1.05;
  camera.position.copy(mid).addScaledVector(new THREE.Vector3(0.6, 0.45, -0.75).normalize(), dist);
  camera.near = dist / 200; camera.far = dist * 50; camera.updateProjectionMatrix();
  controls.target.copy(mid); controls.minDistance = radius * 0.3; controls.maxDistance = dist * 4; controls.update();
  const grid = new THREE.GridHelper(Math.max(size.x, size.z) * 2.5, 20, 0x3a4250, 0x232a33); grid.position.y = box.min.y; scene.add(grid);
}});
addEventListener("resize", () => {{ camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); }});
renderer.setAnimationLoop(() => {{ controls.update(); renderer.render(scene, camera); }});
</script>
</body>
</html>
"""


def write_viewer(path: str, glb_path: str, title: str, stats: str, max_bytes: int = 12_000_000) -> Optional[str]:
    size = os.path.getsize(glb_path)
    if size > max_bytes:
        return None
    with open(glb_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    html = VIEWER_TEMPLATE.format(title=title.replace("<", ""), stats=stats.replace("<", ""), glb=b64)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
