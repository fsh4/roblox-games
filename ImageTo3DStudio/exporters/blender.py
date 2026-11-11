"""Optional Blender bridge: independent verification of exported files.

The studio writes FBX itself. When Blender is available (a Blender install via
BLENDER_PATH, or the ``bpy`` module from PyPI in some Python interpreter), this
module imports an exported file in a headless Blender process and reports what
Blender sees: objects, triangle counts, world-space bounds, UV layers and the
textures wired into the material. Tests and the quality report use it as an
external check of the FBX writer.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

SCRIPT = r'''
import bpy, json, sys
path = sys.argv[sys.argv.index("--") + 1]
out = sys.argv[sys.argv.index("--") + 2]
bpy.ops.wm.read_factory_settings(use_empty=True)
if path.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=path)
else:
    bpy.ops.import_scene.gltf(filepath=path)
report = {"objects": [], "images": []}
for ob in bpy.context.scene.objects:
    if ob.type != "MESH":
        continue
    me = ob.data
    mw = ob.matrix_world
    co = [mw @ v.co for v in me.vertices]
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    links = []
    for slot in ob.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for l in mat.node_tree.links:
                if l.from_node.type == "TEX_IMAGE":
                    links.append([l.from_node.image.name if l.from_node.image else None, l.to_node.type, l.to_socket.name])
    report["objects"].append({
        "name": ob.name, "triangles": tris, "vertices": len(me.vertices), "uv_layers": [u.name for u in me.uv_layers],
        "min": [min(c[i] for c in co) for i in range(3)] if co else None, "max": [max(c[i] for c in co) for i in range(3)] if co else None,
        "texture_links": links,
    })
for im in bpy.data.images:
    report["images"].append({"name": im.name, "size": list(im.size), "packed": im.packed_file is not None})
json.dump(report, open(out, "w"))
'''


def find_blender(configured: str = "") -> Optional[List[str]]:
    """Command prefix that runs a Python script inside Blender, or None."""
    if configured and os.path.exists(configured):
        if os.path.basename(configured).lower().startswith("python"):
            return [configured]
        return [configured, "--background", "--factory-startup", "--python"]
    exe = shutil.which("blender")
    if exe:
        return [exe, "--background", "--factory-startup", "--python"]
    for candidate in (os.environ.get("BPY_PYTHON", ""),):
        if candidate and os.path.exists(candidate):
            return [candidate]
    try:
        import importlib.util

        if importlib.util.find_spec("bpy") is not None:
            return [sys.executable]
    except Exception:
        pass
    return None


def inspect_with_blender(path: str, configured: str = "", timeout: int = 240) -> Optional[dict]:
    cmd = find_blender(configured)
    if cmd is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "studio_blender_probe.py")
        out = os.path.join(tmp, "report.json")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(SCRIPT)
        proc = subprocess.run(cmd + [script, "--", os.path.abspath(path), out], capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"Blender failed to import {os.path.basename(path)} (exit {proc.returncode}): {proc.stderr[:700]} ... {proc.stderr[-300:]}")
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)
