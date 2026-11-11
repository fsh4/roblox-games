"""Convert an existing package to another format without rebuilding it.

The FBX in the package is the source of truth: it is parsed back with the FBX
reader (so the conversion reflects exactly what was delivered) and written out
as OBJ or STL; other formats are copied from the package.
"""
from __future__ import annotations

import os
import shutil
from typing import List, Tuple

import numpy as np

from engine.geometry import Mesh, weld

from . import fbx, interchange, packaging

FORMATS = {
    "fbx": "model.fbx", "glb": "model.glb", "rbxm": "model.rbxm", "rbxmx": "model.rbxmx",
    "obj": None, "stl": None, "textures": None, "zip": None,
}


def meshes_from_fbx(path: str) -> List[Tuple[str, Mesh]]:
    summary = fbx.summarize_fbx(path)
    out = []
    for gid, gd in summary["geometries"].items():
        tri = gd["triangles"]
        if tri is None:
            continue
        v_local = gd["vertices"]
        world = np.stack([-v_local[:, 0], v_local[:, 2], v_local[:, 1]], axis=1)
        corner_n = gd["normals"]
        corner_n = np.stack([-corner_n[:, 0], corner_n[:, 2], corner_n[:, 1]], axis=1)
        corner_uv = gd["uv"] if gd["uv"] is not None else np.zeros((len(tri) * 3, 2))
        corner_v = world[tri.reshape(-1)]
        mesh = Mesh(corner_v, np.arange(len(corner_v)).reshape(-1, 3), corner_n, corner_uv, None, gd["name"])
        out.append((gd["name"], weld(mesh)))
    return out


def export_format(package_dir: str, fmt: str, output_dir: str) -> str:
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"unsupported format '{fmt}'")
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(package_dir))
    if FORMATS.get(fmt):
        src = os.path.join(package_dir, FORMATS[fmt])
        if not os.path.exists(src):
            raise FileNotFoundError(f"this version has no {fmt.upper()} file")
        dst = os.path.join(output_dir, f"{name}.{fmt}")
        shutil.copyfile(src, dst)
        return dst
    if fmt == "zip":
        return packaging.zip_directory(package_dir, os.path.join(output_dir, f"{name}.zip"), name)
    if fmt == "textures":
        tex = os.path.join(package_dir, "textures")
        return packaging.zip_directory(tex, os.path.join(output_dir, f"{name}_textures.zip"), f"{name}_textures")
    meshes = meshes_from_fbx(os.path.join(package_dir, "model.fbx"))
    if fmt == "stl":
        dst = os.path.join(output_dir, f"{name}.stl")
        interchange.write_stl(dst, meshes, name)
        return dst
    obj_dir = os.path.join(output_dir, f"{name}_obj")
    os.makedirs(obj_dir, exist_ok=True)
    tex_src = os.path.join(package_dir, "textures")
    if os.path.isdir(tex_src):
        shutil.copytree(tex_src, os.path.join(obj_dir, "textures"), dirs_exist_ok=True)
    maps = {f[:-4]: f"textures/{f}" for f in os.listdir(os.path.join(obj_dir, "textures"))} if os.path.isdir(os.path.join(obj_dir, "textures")) else {}
    interchange.write_obj(os.path.join(obj_dir, "model.obj"), meshes, maps)
    return packaging.zip_directory(obj_dir, os.path.join(output_dir, f"{name}_obj.zip"), f"{name}_obj")
