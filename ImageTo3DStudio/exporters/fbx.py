"""Native binary FBX 7.4 writer and reader.

The writer reproduces the structure that Blender's FBX exporter emits with the
settings Roblox documents for Studio import (Forward: Z, Up: Y, Apply Scalings:
FBX Unit Scale, Path Mode: Copy + Embed Textures):
  * GlobalSettings axes Up=+Y, Front=-Z, Coord=-X and UnitScaleFactor 100,
  * geometry stored in Blender-local axes with the model node rotated
    (90, 0, 180) degrees, so world coordinates equal Roblox coordinates,
  * one Phong material whose texture slots match Blender's mapping
    (DiffuseColor, NormalMap, ShininessExponent = roughness,
    ReflectionFactor = metallic, EmissiveColor), and
  * PNG files embedded in Video.Content plus relative paths to the copies in
    the package's textures folder.
The reader parses any binary FBX into a node tree; tests use it to verify our
own output independently of the writer's code paths.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HEAD_MAGIC = b"Kaydara FBX Binary\x20\x20\x00\x1a\x00"
FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
TIME_ID = b"1970-01-01 10:00:00:000"
FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
VERSION = 7400
SENTINEL = b"\x00" * 13

TEXTURE_SLOTS = {
    "basecolor": "DiffuseColor",
    "normal": "NormalMap",
    "roughness": "ShininessExponent",
    "metallic": "ReflectionFactor",
    "emissive": "EmissiveColor",
}


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, *props, children: Optional[List["Node"]] = None):
        self.name = name
        self.props = list(props)
        self.children = children or []

    def add(self, name: str, *props) -> "Node":
        child = Node(name, *props)
        self.children.append(child)
        return child


class I32(int):
    pass


class I64(int):
    pass


class F64(float):
    pass


class Raw(bytes):
    pass


def _encode_prop(v) -> bytes:
    if isinstance(v, bool):
        return b"C" + struct.pack("<?", v)
    if isinstance(v, I64):
        return b"L" + struct.pack("<q", int(v))
    if isinstance(v, I32):
        return b"I" + struct.pack("<i", int(v))
    if isinstance(v, int):
        return b"I" + struct.pack("<i", v)
    if isinstance(v, float):
        return b"D" + struct.pack("<d", v)
    if isinstance(v, Raw):
        return b"R" + struct.pack("<I", len(v)) + bytes(v)
    if isinstance(v, bytes):
        return b"S" + struct.pack("<I", len(v)) + v
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(v, np.ndarray):
        code = {np.dtype(np.float64): b"d", np.dtype(np.float32): b"f", np.dtype(np.int32): b"i", np.dtype(np.int64): b"l"}[v.dtype]
        data = np.ascontiguousarray(v).astype(v.dtype.newbyteorder("<")).tobytes()
        if len(data) > 128:
            comp = zlib.compress(data, 6)
            return code + struct.pack("<3I", v.size, 1, len(comp)) + comp
        return code + struct.pack("<3I", v.size, 0, len(data)) + data
    raise TypeError(f"unsupported FBX property {type(v)}")


def _encode_node(node: Node, offset: int, is_last: bool) -> bytes:
    props = b"".join(_encode_prop(p) for p in node.props)
    name = node.name.encode("ascii")
    head_len = 12 + 1 + len(name)
    body = bytearray()
    cursor = offset + head_len + len(props)
    if node.children:
        for i, child in enumerate(node.children):
            enc = _encode_node(child, cursor, i == len(node.children) - 1)
            body += enc
            cursor += len(enc)
        body += SENTINEL
    elif not node.props and not is_last:
        body += SENTINEL
    end = offset + head_len + len(props) + len(body)
    return struct.pack("<3I", end, len(node.props), len(props)) + bytes([len(name)]) + name + props + bytes(body)


def write_nodes(path: str, nodes: List[Node]) -> None:
    out = bytearray(HEAD_MAGIC + struct.pack("<I", VERSION))
    for i, n in enumerate(nodes):
        out += _encode_node(n, len(out), False)
    out += SENTINEL
    out += FOOT_ID + b"\x00" * 4
    pad = ((len(out) + 15) & ~15) - len(out)
    out += b"\x00" * (pad if pad else 16)
    out += struct.pack("<I", VERSION) + b"\x00" * 120 + FOOT_MAGIC
    with open(path, "wb") as fh:
        fh.write(out)


def _p(name: str, typ: str, label: str, flags: str, *values) -> Node:
    return Node("P", name, typ, label, flags, *values)


def _props70(*items: Node) -> Node:
    return Node("Properties70", children=list(items))


@dataclass
class FbxMesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray


@dataclass
class FbxMaterial:
    name: str
    textures: Dict[str, Tuple[str, bytes]] = field(default_factory=dict)
    diffuse: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    roughness: float = 0.6
    metallic: float = 0.0


class _Ids:
    def __init__(self):
        self.n = 1_000_000_000

    def next(self) -> I64:
        self.n += 7919
        return I64(self.n)


def to_blender_local(points: np.ndarray) -> np.ndarray:
    """Roblox axes (X, Y, Z) -> Blender object axes (-X, Z, Y)."""
    return np.stack([-points[:, 0], points[:, 2], points[:, 1]], axis=1)


def _geometry_node(gid: I64, mesh: FbxMesh) -> Node:
    v = to_blender_local(mesh.vertices).astype(np.float64)
    n = to_blender_local(mesh.normals).astype(np.float64)
    faces = mesh.faces.astype(np.int64)
    poly = faces.copy()
    poly[:, 2] = -poly[:, 2] - 1
    corner_v = faces.reshape(-1)
    nrm_corner = n[corner_v]
    uv_corner = mesh.uvs[corner_v]
    nrm_unique, nrm_index = np.unique(np.round(nrm_corner, 6), axis=0, return_inverse=True)
    uv_unique, uv_index = np.unique(np.round(uv_corner, 7), axis=0, return_inverse=True)
    geo = Node("Geometry", gid, mesh.name + "\x00\x01Geometry", "Mesh")
    geo.children.append(_props70())
    geo.add("GeometryVersion", I32(124))
    geo.add("Vertices", v.reshape(-1))
    geo.add("PolygonVertexIndex", poly.reshape(-1).astype(np.int32))
    le_n = geo.add("LayerElementNormal", I32(0))
    le_n.add("Version", I32(101))
    le_n.add("Name", "")
    le_n.add("MappingInformationType", "ByPolygonVertex")
    le_n.add("ReferenceInformationType", "IndexToDirect")
    le_n.add("Normals", nrm_unique.astype(np.float64).reshape(-1))
    le_n.add("NormalsIndex", nrm_index.reshape(-1).astype(np.int32))
    le_uv = geo.add("LayerElementUV", I32(0))
    le_uv.add("Version", I32(101))
    le_uv.add("Name", "UVMap")
    le_uv.add("MappingInformationType", "ByPolygonVertex")
    le_uv.add("ReferenceInformationType", "IndexToDirect")
    le_uv.add("UV", uv_unique.astype(np.float64).reshape(-1))
    le_uv.add("UVIndex", uv_index.reshape(-1).astype(np.int32))
    le_m = geo.add("LayerElementMaterial", I32(0))
    le_m.add("Version", I32(101))
    le_m.add("Name", "")
    le_m.add("MappingInformationType", "AllSame")
    le_m.add("ReferenceInformationType", "IndexToDirect")
    le_m.add("Materials", np.zeros(1, dtype=np.int32))
    layer = geo.add("Layer", I32(0))
    layer.add("Version", I32(100))
    for t in ("LayerElementNormal", "LayerElementUV", "LayerElementMaterial"):
        le = layer.add("LayerElement")
        le.add("Type", t)
        le.add("TypedIndex", I32(0))
    return geo


def _model_node(mid: I64, name: str) -> Node:
    m = Node("Model", mid, name + "\x00\x01Model", "Mesh")
    m.add("Version", I32(232))
    m.children.append(_props70(
        _p("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0),
        _p("Lcl Rotation", "Lcl Rotation", "", "A", 90.0, 0.0, 180.0),
        _p("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0),
        _p("DefaultAttributeIndex", "int", "Integer", "", I32(0)),
        _p("InheritType", "enum", "", "", I32(1)),
    ))
    m.add("MultiLayer", I32(0))
    m.add("MultiTake", I32(0))
    m.add("Shading", True)
    m.add("Culling", "CullingOff")
    return m


def _material_node(mat_id: I64, mat: FbxMaterial) -> Node:
    shin = float(((1.0 - mat.roughness) * 10.0) ** 2)
    node = Node("Material", mat_id, mat.name + "\x00\x01Material", "")
    node.add("Version", I32(102))
    node.add("ShadingModel", "Phong")
    node.add("MultiLayer", I32(0))
    d = mat.diffuse
    node.children.append(_props70(
        _p("DiffuseColor", "Color", "", "A", float(d[0]), float(d[1]), float(d[2])),
        _p("EmissiveColor", "Color", "", "A", 1.0, 1.0, 1.0),
        _p("EmissiveFactor", "Number", "", "A", 1.0 if "emissive" in mat.textures else 0.0),
        _p("AmbientColor", "Color", "", "A", 0.0, 0.0, 0.0),
        _p("AmbientFactor", "Number", "", "A", 0.0),
        _p("SpecularColor", "Color", "", "A", 0.8, 0.8, 0.8),
        _p("SpecularFactor", "Number", "", "A", 0.25),
        _p("Shininess", "Number", "", "A", shin),
        _p("ShininessExponent", "Number", "", "A", shin),
        _p("ReflectionColor", "Color", "", "A", 0.8, 0.8, 0.8),
        _p("ReflectionFactor", "Number", "", "A", float(mat.metallic)),
    ))
    return node


def _texture_nodes(tid: I64, vid: I64, slot_name: str, rel_path: str, data: bytes) -> Tuple[Node, Node]:
    stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    tex = Node("Texture", tid, f"{slot_name}_texture\x00\x01Texture", "")
    tex.add("Type", "TextureVideoClip")
    tex.add("Version", I32(202))
    tex.add("TextureName", f"{slot_name}_texture\x00\x01Texture")
    tex.add("Media", f"{stem}\x00\x01Video")
    tex.add("FileName", rel_path)
    tex.add("RelativeFilename", rel_path)
    tex.children.append(_props70(_p("UseMaterial", "bool", "", "", I32(1))))
    vid_node = Node("Video", vid, f"{stem}\x00\x01Video", "Clip")
    vid_node.add("Type", "Clip")
    vid_node.children.append(_props70(_p("Path", "KString", "Url", "", rel_path)))
    vid_node.add("UseMipMap", I32(0))
    vid_node.add("Filename", rel_path)
    vid_node.add("RelativeFilename", rel_path)
    vid_node.add("Content", Raw(data))
    return tex, vid_node


def _definitions(n_geo: int, n_model: int, n_mat: int, n_tex: int) -> Node:
    d = Node("Definitions")
    d.add("Version", I32(100))
    d.add("Count", I32(1 + n_geo + n_model + n_mat + 2 * n_tex))
    gs = d.add("ObjectType", "GlobalSettings")
    gs.add("Count", I32(1))
    og = d.add("ObjectType", "Geometry")
    og.add("Count", I32(n_geo))
    pt = og.add("PropertyTemplate", "FbxMesh")
    pt.children.append(_props70(
        _p("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8),
        _p("BBoxMin", "Vector3D", "Vector", "", 0.0, 0.0, 0.0),
        _p("BBoxMax", "Vector3D", "Vector", "", 0.0, 0.0, 0.0),
        _p("Primary Visibility", "bool", "", "", I32(1)),
        _p("Casts Shadows", "bool", "", "", I32(1)),
        _p("Receive Shadows", "bool", "", "", I32(1)),
    ))
    om = d.add("ObjectType", "Model")
    om.add("Count", I32(n_model))
    pt = om.add("PropertyTemplate", "FbxNode")
    pt.children.append(_props70(
        _p("RotationOrder", "enum", "", "", I32(0)),
        _p("InheritType", "enum", "", "", I32(0)),
        _p("ScalingMax", "Vector3D", "Vector", "", 1.0, 1.0, 1.0),
        _p("GeometricTranslation", "Vector3D", "Vector", "", 0.0, 0.0, 0.0),
        _p("GeometricRotation", "Vector3D", "Vector", "", 0.0, 0.0, 0.0),
        _p("GeometricScaling", "Vector3D", "Vector", "", 1.0, 1.0, 1.0),
        _p("Show", "bool", "", "", I32(1)),
        _p("DefaultAttributeIndex", "int", "Integer", "", I32(-1)),
        _p("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0),
        _p("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0),
        _p("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0),
        _p("Visibility", "Visibility", "", "A", 1.0),
        _p("Visibility Inheritance", "Visibility Inheritance", "", "", I32(1)),
    ))
    if n_mat:
        omt = d.add("ObjectType", "Material")
        omt.add("Count", I32(n_mat))
        pt = omt.add("PropertyTemplate", "FbxSurfacePhong")
        pt.children.append(_props70(
            _p("ShadingModel", "KString", "", "", "Phong"),
            _p("EmissiveColor", "Color", "", "A", 0.0, 0.0, 0.0),
            _p("EmissiveFactor", "Number", "", "A", 1.0),
            _p("DiffuseColor", "Color", "", "A", 0.8, 0.8, 0.8),
            _p("DiffuseFactor", "Number", "", "A", 1.0),
            _p("NormalMap", "Vector3D", "Vector", "", 0.0, 0.0, 0.0),
            _p("BumpFactor", "double", "Number", "", 1.0),
            _p("SpecularColor", "Color", "", "A", 0.2, 0.2, 0.2),
            _p("ShininessExponent", "Number", "", "A", 20.0),
            _p("ReflectionFactor", "Number", "", "A", 1.0),
        ))
    if n_tex:
        ot = d.add("ObjectType", "Texture")
        ot.add("Count", I32(n_tex))
        pt = ot.add("PropertyTemplate", "FbxFileTexture")
        pt.children.append(_props70(
            _p("TextureTypeUse", "enum", "", "", I32(0)),
            _p("AlphaSource", "enum", "", "", I32(2)),
            _p("Texture alpha", "double", "Number", "", 1.0),
            _p("PremultiplyAlpha", "bool", "", "", I32(1)),
            _p("CurrentTextureBlendMode", "enum", "", "", I32(1)),
            _p("UVSet", "KString", "", "", "default"),
            _p("WrapModeU", "enum", "", "", I32(0)),
            _p("WrapModeV", "enum", "", "", I32(0)),
            _p("UseMaterial", "bool", "", "", I32(0)),
            _p("UseMipMap", "bool", "", "", I32(0)),
        ))
        ov = d.add("ObjectType", "Video")
        ov.add("Count", I32(n_tex))
        pt = ov.add("PropertyTemplate", "FbxVideo")
        pt.children.append(_props70(
            _p("Path", "KString", "Url", "", ""),
            _p("AccessMode", "enum", "", "", I32(0)),
            _p("ImageSequence", "bool", "", "", I32(0)),
        ))
    return d


def _header(creator: str, now) -> List[Node]:
    ext = Node("FBXHeaderExtension")
    ext.add("FBXHeaderVersion", I32(1003))
    ext.add("FBXVersion", I32(VERSION))
    ext.add("EncryptionType", I32(0))
    ts = ext.add("CreationTimeStamp")
    for k, v in (("Version", 1000), ("Year", now[0]), ("Month", now[1]), ("Day", now[2]), ("Hour", now[3]), ("Minute", now[4]), ("Second", now[5]), ("Millisecond", 0)):
        ts.add(k, I32(v))
    ext.add("Creator", creator)
    si = ext.add("SceneInfo", "GlobalInfo\x00\x01SceneInfo", "UserData")
    si.add("Type", "UserData")
    si.add("Version", I32(100))
    md = si.add("MetaData")
    md.add("Version", I32(100))
    for k in ("Title", "Subject", "Author", "Keywords", "Revision", "Comment"):
        md.add(k, "")
    si.children.append(_props70(
        _p("DocumentUrl", "KString", "Url", "", "/model.fbx"),
        _p("SrcDocumentUrl", "KString", "Url", "", "/model.fbx"),
        _p("Original", "Compound", "", ""),
        _p("Original|ApplicationVendor", "KString", "", "", "Image-to-3D Studio"),
        _p("Original|ApplicationName", "KString", "", "", "Image-to-3D Studio"),
        _p("Original|ApplicationVersion", "KString", "", "", "1.0"),
    ))
    nodes = [ext, Node("FileId", Raw(FILE_ID)), Node("CreationTime", TIME_ID), Node("Creator", creator)]
    gs = Node("GlobalSettings")
    gs.add("Version", I32(1000))
    gs.children.append(_props70(
        _p("UpAxis", "int", "Integer", "", I32(1)),
        _p("UpAxisSign", "int", "Integer", "", I32(1)),
        _p("FrontAxis", "int", "Integer", "", I32(2)),
        _p("FrontAxisSign", "int", "Integer", "", I32(-1)),
        _p("CoordAxis", "int", "Integer", "", I32(0)),
        _p("CoordAxisSign", "int", "Integer", "", I32(-1)),
        _p("OriginalUpAxis", "int", "Integer", "", I32(-1)),
        _p("OriginalUpAxisSign", "int", "Integer", "", I32(1)),
        _p("UnitScaleFactor", "double", "Number", "", 100.0),
        _p("OriginalUnitScaleFactor", "double", "Number", "", 100.0),
        _p("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0),
        _p("DefaultCamera", "KString", "", "", "Producer Perspective"),
        _p("TimeMode", "enum", "", "", I32(11)),
        _p("TimeSpanStart", "KTime", "Time", "", I64(0)),
        _p("TimeSpanStop", "KTime", "Time", "", I64(46186158000)),
        _p("CustomFrameRate", "double", "Number", "", 24.0),
    ))
    nodes.append(gs)
    return nodes


def write_fbx(path: str, meshes: Sequence[FbxMesh], material: FbxMaterial, creator: str = "Image-to-3D Studio FBX writer") -> dict:
    """Write meshes sharing one textured material. Returns a small summary."""
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    ids = _Ids()
    nodes = _header(creator, (now.year, now.month, now.day, now.hour, now.minute, now.second))
    doc_id = ids.next()
    docs = Node("Documents")
    docs.add("Count", I32(1))
    doc = docs.add("Document", doc_id, "Scene", "Scene")
    doc.children.append(_props70(_p("SourceObject", "object", "", ""), _p("ActiveAnimStackName", "KString", "", "", "")))
    doc.add("RootNode", I64(0))
    nodes.append(docs)
    nodes.append(Node("References"))
    tex_items = [(k, material.textures[k]) for k in TEXTURE_SLOTS if k in material.textures]
    nodes.append(_definitions(len(meshes), len(meshes), 1, len(tex_items)))
    objects = Node("Objects")
    conns = Node("Connections")
    mat_id = ids.next()
    for m in meshes:
        gid, mid = ids.next(), ids.next()
        objects.children.append(_geometry_node(gid, m))
        objects.children.append(_model_node(mid, m.name))
        conns.add("C", "OO", mid, I64(0))
        conns.add("C", "OO", gid, mid)
        conns.add("C", "OO", mat_id, mid)
    objects.children.append(_material_node(mat_id, material))
    for key, (rel, data) in tex_items:
        tid, vid = ids.next(), ids.next()
        tex, vid_node = _texture_nodes(tid, vid, key, rel, data)
        objects.children.append(tex)
        objects.children.append(vid_node)
        conns.add("C", "OP", tid, mat_id, TEXTURE_SLOTS[key])
        conns.add("C", "OO", vid, tid)
    nodes.append(objects)
    nodes.append(conns)
    takes = Node("Takes")
    takes.add("Current", "")
    nodes.append(takes)
    write_nodes(path, nodes)
    return {"meshes": len(meshes), "triangles": int(sum(len(m.faces) for m in meshes)), "textures": [k for k, _ in tex_items]}


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

@dataclass
class ReadNode:
    name: str
    props: list
    children: List["ReadNode"]

    def find(self, name: str) -> Optional["ReadNode"]:
        return next((c for c in self.children if c.name == name), None)

    def find_all(self, name: str) -> List["ReadNode"]:
        return [c for c in self.children if c.name == name]


def read_fbx(path: str) -> Tuple[int, List[ReadNode]]:
    data = open(path, "rb").read()
    if not data.startswith(HEAD_MAGIC[:20]):
        raise ValueError("not a binary FBX file")
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500

    def prop(off):
        t = chr(data[off])
        off += 1
        if t in "YCIFDL":
            fmt = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t]
            return struct.unpack_from(fmt, data, off)[0], off + struct.calcsize(fmt)
        if t in "fdlib":
            n, enc, clen = struct.unpack_from("<3I", data, off)
            off += 12
            raw = data[off:off + clen]
            off += clen
            if enc:
                raw = zlib.decompress(raw)
            dt = {"f": np.float32, "d": np.float64, "l": np.int64, "i": np.int32, "b": np.bool_}[t]
            arr = np.frombuffer(raw, dt)
            if len(arr) != n:
                raise ValueError("array length mismatch")
            return arr, off
        if t in "SR":
            n = struct.unpack_from("<I", data, off)[0]
            off += 4
            return bytes(data[off:off + n]), off + n
        raise ValueError(f"unknown property type {t!r}")

    def node(off):
        if wide:
            end, nprops, _ = struct.unpack_from("<3Q", data, off)
            off += 24
        else:
            end, nprops, _ = struct.unpack_from("<3I", data, off)
            off += 12
        if end == 0:
            return None, off + 1
        nl = data[off]
        name = data[off + 1:off + 1 + nl].decode("ascii")
        off += 1 + nl
        props = []
        for _ in range(nprops):
            v, off = prop(off)
            props.append(v)
        kids = []
        while off < end:
            k, off2 = node(off)
            if k is None:
                off = off2
                break
            kids.append(k)
            off = off2
        if off != end:
            raise ValueError(f"node {name} size mismatch")
        return ReadNode(name, props, kids), end

    off = 27
    nodes = []
    while True:
        n, off = node(off)
        if n is None:
            break
        nodes.append(n)
    if FOOT_MAGIC not in data[-40:]:
        raise ValueError("missing FBX footer")
    return version, nodes


def summarize_fbx(path: str) -> dict:
    """Decode meshes, materials, textures and connections for verification."""
    version, nodes = read_fbx(path)
    top = {n.name: n for n in nodes}
    gs = {p.props[0].decode(): p.props[4:] for p in top["GlobalSettings"].find("Properties70").children}
    objects = top["Objects"]
    geoms = {}
    for gnode in objects.find_all("Geometry"):
        v = gnode.find("Vertices").props[0].reshape(-1, 3)
        pvi = gnode.find("PolygonVertexIndex").props[0]
        ends = pvi < 0
        idx = np.where(ends, -pvi - 1, pvi)
        tri = idx.reshape(-1, 3) if ends[2::3].all() and len(idx) % 3 == 0 else None
        uv_node = gnode.find("LayerElementUV")
        n_node = gnode.find("LayerElementNormal")
        uv = uv_node.find("UV").props[0].reshape(-1, 2)[uv_node.find("UVIndex").props[0]] if uv_node else None
        nrm = n_node.find("Normals").props[0].reshape(-1, 3)
        ni = n_node.find("NormalsIndex")
        nrm = nrm[ni.props[0]] if ni is not None else nrm
        geoms[gnode.props[0]] = {"name": gnode.props[1].split(b"\x00")[0].decode(), "vertices": v, "triangles": tri, "uv": uv, "normals": nrm}
    models = {}
    for mnode in objects.find_all("Model"):
        p70 = {p.props[0].decode(): p.props[4:] for p in mnode.find("Properties70").children}
        models[mnode.props[0]] = {"name": mnode.props[1].split(b"\x00")[0].decode(), "rotation": p70.get("Lcl Rotation"), "translation": p70.get("Lcl Translation")}
    textures = {}
    videos = {}
    for t in objects.find_all("Texture"):
        textures[t.props[0]] = t.find("RelativeFilename").props[0].decode()
    for vn in objects.find_all("Video"):
        content = vn.find("Content")
        videos[vn.props[0]] = len(content.props[0]) if content is not None else 0
    slots = {}
    conn = top["Connections"]
    for c in conn.children:
        kind = c.props[0].decode()
        if kind == "OP":
            slots[c.props[3].decode()] = textures.get(c.props[1])
    world = []
    for gid, gd in geoms.items():
        rot = None
        for c in conn.children:
            if c.props[0] == b"OO" and c.props[1] == gid and c.props[2] in models:
                rot = models[c.props[2]]["rotation"]
        r = np.radians(np.asarray(rot if rot else (0, 0, 0), dtype=np.float64))
        rx = np.array([[1, 0, 0], [0, np.cos(r[0]), -np.sin(r[0])], [0, np.sin(r[0]), np.cos(r[0])]])
        ry = np.array([[np.cos(r[1]), 0, np.sin(r[1])], [0, 1, 0], [-np.sin(r[1]), 0, np.cos(r[1])]])
        rz = np.array([[np.cos(r[2]), -np.sin(r[2]), 0], [np.sin(r[2]), np.cos(r[2]), 0], [0, 0, 1]])
        world.append(gd["vertices"] @ (rz @ ry @ rx).T)
    return {
        "version": version, "global_settings": {k: list(v) for k, v in gs.items() if k in ("UpAxis", "FrontAxis", "FrontAxisSign", "CoordAxis", "CoordAxisSign", "UnitScaleFactor")},
        "geometries": geoms, "models": models, "texture_slots": slots, "embedded_bytes": videos,
        "world_vertices": np.concatenate(world) if world else np.zeros((0, 3)),
    }
