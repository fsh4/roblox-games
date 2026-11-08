"""glTF 2.0 binary (.glb), Wavefront OBJ/MTL and binary STL writers."""
from __future__ import annotations

import io
import json
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from engine.geometry import Mesh

MeshList = Sequence[Tuple[str, Mesh]]


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    mode = "L" if arr.ndim == 2 else ("RGBA" if arr.shape[2] == 4 else "RGB")
    Image.fromarray(arr, mode).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def write_glb(path: str, meshes: MeshList, maps: Dict[str, np.ndarray], material_name: str = "Atlas", roughness: float = 1.0) -> dict:
    """Write meshes with one PBR metallic-roughness material.

    Roughness, metalness and occlusion are packed into one ORM texture
    (R = occlusion, G = roughness, B = metallic) as the glTF spec defines. glTF
    places the UV origin at the top-left, so V is flipped.
    """
    blobs: List[bytes] = []
    views: List[dict] = []
    accessors: List[dict] = []

    def add_view(data: bytes, target: Optional[int] = None) -> int:
        offset = sum(len(b) for b in blobs)
        pad = (-offset) % 4
        if pad:
            blobs.append(b"\x00" * pad)
            offset += pad
        blobs.append(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    def add_accessor(arr: np.ndarray, comp: int, kind: str, target: int, minmax: bool = False) -> int:
        v = add_view(arr.tobytes(), target)
        acc = {"bufferView": v, "componentType": comp, "count": int(arr.shape[0]), "type": kind}
        if minmax:
            acc["min"] = arr.min(axis=0).astype(float).tolist()
            acc["max"] = arr.max(axis=0).astype(float).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    images, textures = [], []

    def add_image(arr: np.ndarray) -> int:
        v = add_view(_png(arr))
        images.append({"bufferView": v, "mimeType": "image/png"})
        textures.append({"source": len(images) - 1, "sampler": 0})
        return len(textures) - 1

    mat: dict = {"name": material_name, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
    if "basecolor" in maps:
        mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": add_image(maps["basecolor"])}
        if maps["basecolor"].ndim == 3 and maps["basecolor"].shape[2] == 4:
            mat["alphaMode"] = "BLEND"
    res = next(iter(maps.values())).shape[0] if maps else 1
    r = maps.get("roughness", np.full((res, res), int(roughness * 255), np.uint8))
    m = maps.get("metallic", np.zeros_like(r))
    ao = maps.get("ambient_occlusion", np.full_like(r, 255))
    orm = np.stack([ao, r, m], axis=2)
    orm_tex = add_image(orm)
    mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": orm_tex}
    mat["pbrMetallicRoughness"]["metallicFactor"] = 1.0 if "metallic" in maps else 0.0
    if "ambient_occlusion" in maps:
        mat["occlusionTexture"] = {"index": orm_tex}
    if "normal" in maps:
        mat["normalTexture"] = {"index": add_image(maps["normal"])}
    if "emissive" in maps:
        e = maps["emissive"]
        base = maps.get("basecolor")
        emis_rgb = (base[..., :3].astype(np.float32) * (e[..., None] / 255.0)).astype(np.uint8) if base is not None else np.stack([e] * 3, axis=2)
        mat["emissiveTexture"] = {"index": add_image(emis_rgb)}
        mat["emissiveFactor"] = [1.0, 1.0, 1.0]
    nodes, gl_meshes = [], []
    for name, mesh in meshes:
        pos = mesh.vertices.astype(np.float32)
        nrm = mesh.normals.astype(np.float32)
        uv = mesh.uvs.astype(np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        idx = mesh.faces.astype(np.uint32).reshape(-1)
        prim = {
            "attributes": {
                "POSITION": add_accessor(pos, 5126, "VEC3", 34962, True),
                "NORMAL": add_accessor(nrm, 5126, "VEC3", 34962),
                "TEXCOORD_0": add_accessor(uv, 5126, "VEC2", 34962),
            },
            "indices": add_accessor(idx.reshape(-1, 1), 5125, "SCALAR", 34963),
            "material": 0,
        }
        accessors[prim["indices"]]["type"] = "SCALAR"
        gl_meshes.append({"name": name, "primitives": [prim]})
        nodes.append({"name": name, "mesh": len(gl_meshes) - 1})
    bin_data = b"".join(blobs)
    bin_data += b"\x00" * ((-len(bin_data)) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "Image-to-3D Studio"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": gl_meshes,
        "materials": [mat], "textures": textures, "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(bin_data)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_data)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(bin_data), 0x004E4942) + bin_data)
    return {"bytes": total, "meshes": len(gl_meshes)}


def write_obj(path: str, meshes: MeshList, maps: Dict[str, str], mtl_name: str = "model.mtl", material_name: str = "Atlas", base_color=(1.0, 1.0, 1.0)) -> dict:
    """OBJ with an MTL using the PBR extension keys (map_Pr, map_Pm, norm)."""
    lines = ["# Image-to-3D Studio", f"mtllib {mtl_name}"]
    v_off = 1
    for name, m in meshes:
        lines.append(f"o {name}")
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.vertices)
        lines.extend(f"vt {u:.6f} {v:.6f}" for u, v in m.uvs)
        lines.extend(f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in m.normals)
        lines.append(f"usemtl {material_name}")
        lines.append("s 1")
        f = m.faces + v_off
        lines.extend(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in f)
        v_off += len(m.vertices)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    mtl = [f"newmtl {material_name}", "Ka 0 0 0", f"Kd {base_color[0]:.4f} {base_color[1]:.4f} {base_color[2]:.4f}", "Ks 0.04 0.04 0.04", "Ns 50", "d 1", "illum 2"]
    keys = {"basecolor": "map_Kd", "normal": "norm", "roughness": "map_Pr", "metallic": "map_Pm", "emissive": "map_Ke", "opacity": "map_d"}
    for k, tag in keys.items():
        if k in maps:
            mtl.append(f"{tag} {maps[k]}")
    import os

    with open(os.path.join(os.path.dirname(path), mtl_name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(mtl) + "\n")
    return {"vertices": v_off - 1}


def write_stl(path: str, meshes: MeshList, name: str = "model") -> dict:
    tris = []
    for _, m in meshes:
        t = m.vertices[m.faces]
        tris.append(t)
    t = np.concatenate(tris).astype(np.float32)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = t
    header = (f"Image-to-3D Studio {name}"[:80]).encode("ascii", "replace").ljust(80, b" ")
    with open(path, "wb") as fh:
        fh.write(header + struct.pack("<I", len(t)) + rec.tobytes())
    return {"triangles": int(len(t))}
