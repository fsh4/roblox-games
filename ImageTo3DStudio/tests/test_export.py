import io
import json
import os
import shutil
import struct
import subprocess
import zipfile

import numpy as np
import pytest
import trimesh

from engine import procedural as pr
from engine import textures as tx
from exporters import blender, convert, fbx, interchange, packaging, rbxm

RBXCHECK = os.environ.get("RBXCHECK") or shutil.which("rbxcheck")


@pytest.fixture(scope="module")
def baked_chair():
    t = pr.get_template("chair")
    scene = t.build(t.defaults(), pr.LOD(0.3))
    ts = tx.bake_scene(scene, tx.BakeOptions(resolution=256, ao_directions=8))
    return scene, ts


def _png(arr):
    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def test_fbx_roundtrip(tmp_path, baked_chair):
    scene, ts = baked_chair
    m = scene.merged()
    mat = fbx.FbxMaterial("Chair", {k: (f"textures/{k}.png", _png(v)) for k, v in ts.maps.items() if k in fbx.TEXTURE_SLOTS})
    path = tmp_path / "chair.fbx"
    fbx.write_fbx(str(path), [fbx.FbxMesh("Chair", m.vertices, m.faces, m.normals, m.uvs)], mat)
    s = fbx.summarize_fbx(str(path))
    assert s["version"] == 7400
    assert s["global_settings"]["UpAxis"] == [1] and s["global_settings"]["UnitScaleFactor"] == [100.0]
    geo = next(iter(s["geometries"].values()))
    assert len(geo["triangles"]) == m.triangle_count
    assert np.allclose(np.sort(s["world_vertices"], 0), np.sort(m.vertices, 0), atol=1e-9)
    assert s["texture_slots"]["DiffuseColor"] == "textures/basecolor.png"
    assert s["texture_slots"]["ShininessExponent"] == "textures/roughness.png"
    assert all(n > 100 for n in s["embedded_bytes"].values())


def test_fbx_imports_in_blender(tmp_path, baked_chair):
    if blender.find_blender(os.environ.get("BLENDER_PATH", "")) is None:
        pytest.skip("Blender / bpy not available")
    scene, ts = baked_chair
    m = scene.merged()
    mat = fbx.FbxMaterial("Chair", {k: (f"textures/{k}.png", _png(v)) for k, v in ts.maps.items() if k in fbx.TEXTURE_SLOTS})
    path = tmp_path / "chair.fbx"
    fbx.write_fbx(str(path), [fbx.FbxMesh("Chair", m.vertices, m.faces, m.normals, m.uvs)], mat)
    rep = blender.inspect_with_blender(str(path), os.environ.get("BLENDER_PATH", ""))
    ob = rep["objects"][0]
    assert ob["triangles"] == m.triangle_count
    lo, hi = m.bounds()
    expected_min = [-hi[0], lo[2], lo[1]]
    expected_max = [-lo[0], hi[2], hi[1]]
    assert np.allclose(ob["min"], expected_min, atol=1e-4) and np.allclose(ob["max"], expected_max, atol=1e-4)
    sockets = {link[2] for link in ob["texture_links"]}
    assert {"Base Color", "Roughness"} <= sockets
    assert all(im["packed"] for im in rep["images"])


def test_rbxm_and_rbxmx_structure(tmp_path, baked_chair):
    scene, _ = baked_chair
    root = rbxm.build_part_model(scene, "Chair")
    b = tmp_path / "chair.rbxm"
    x = tmp_path / "chair.rbxmx"
    rbxm.write_rbxm(str(b), root)
    rbxm.write_rbxmx(str(x), root)
    data = b.read_bytes()
    assert data[:8] == b"<roblox!" and data[8:14] == b"\x89\xff\x0d\x0a\x1a\x0a"
    classes, instances = struct.unpack_from("<ii", data, 16)
    assert classes == 2 and instances == 1 + len(root.children)
    assert data.endswith(b"END\x00" + struct.pack("<III", 0, 9, 0) + b"</roblox>")
    names = []
    off = 32
    while off < len(data):
        name = data[off:off + 4]
        comp, size = struct.unpack_from("<II", data, off + 4)
        names.append(name.rstrip(b"\x00").decode())
        off += 16 + (comp or size)
    assert names[0] == "META" and names[-1] == "END" and "PRNT" in names and names.count("INST") == 2
    import xml.etree.ElementTree as ET

    tree = ET.parse(x)
    items = tree.getroot().findall(".//Item")
    assert len(items) == instances
    if RBXCHECK:
        out_b = subprocess.run([RBXCHECK, str(b)], capture_output=True, text=True, check=True).stdout
        out_x = subprocess.run([RBXCHECK, str(x)], capture_output=True, text=True, check=True).stdout
        assert out_b == out_x, "binary and XML decode to different trees"
        assert out_b.count("Part '") == len(root.children)


def test_glb_is_valid_gltf(tmp_path, baked_chair):
    scene, ts = baked_chair
    m = scene.merged()
    p = tmp_path / "chair.glb"
    interchange.write_glb(str(p), [("Chair", m)], ts.maps)
    raw = p.read_bytes()
    magic, version, length = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67 and version == 2 and length == len(raw)
    jlen = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + jlen])
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] >= 0
    loaded = trimesh.load(str(p), force="mesh")
    assert len(loaded.faces) == m.triangle_count


def test_obj_and_stl_roundtrip(tmp_path, baked_chair):
    scene, _ = baked_chair
    m = scene.merged()
    interchange.write_obj(str(tmp_path / "m.obj"), [("Chair", m)], {"basecolor": "textures/basecolor.png"})
    interchange.write_stl(str(tmp_path / "m.stl"), [("Chair", m)])
    assert len(trimesh.load(str(tmp_path / "m.obj"), force="mesh").faces) == m.triangle_count
    assert len(trimesh.load(str(tmp_path / "m.stl")).faces) == m.triangle_count
    assert "map_Kd textures/basecolor.png" in (tmp_path / "model.mtl").read_text()


def test_zip_contains_only_package_files(tmp_path):
    pkg = tmp_path / "Pkg"
    (pkg / "textures").mkdir(parents=True)
    (pkg / "model.fbx").write_bytes(b"x")
    (pkg / "textures" / "a.png").write_bytes(b"y")
    outside = tmp_path / "secret.txt"
    outside.write_text("no")
    os.symlink(outside, pkg / "link.txt")
    z = packaging.zip_directory(str(pkg), str(tmp_path / "out.zip"), "Pkg")
    names = zipfile.ZipFile(z).namelist()
    assert sorted(names) == ["Pkg/model.fbx", "Pkg/textures/a.png"]
    assert all(not n.startswith("/") and ".." not in n for n in names)


def test_export_conversion_from_package(tmp_path, bed_version):
    proj, res = bed_version
    out = tmp_path / "exports"
    stl = convert.export_format(res.package_dir, "stl", str(out))
    assert len(trimesh.load(stl).faces) == res.stats["triangles"]
    obj_zip = convert.export_format(res.package_dir, "obj", str(out))
    assert any(n.endswith("model.obj") for n in zipfile.ZipFile(obj_zip).namelist())
    with pytest.raises(ValueError):
        convert.export_format(res.package_dir, "exe", str(out))
