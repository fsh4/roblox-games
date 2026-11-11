"""Roblox model files (.rbxm binary and .rbxmx XML) built from native parts.

A Roblox ``MeshPart`` can only reference meshes stored on Roblox's servers, so
a model file cannot embed our triangle mesh. These files therefore contain the
part-based version of the reconstruction: every component's Roblox-native
primitives (Block, Cylinder, Ball) with size, CFrame, colour and a matching
built-in material. They open with File > Insert from file in Studio without any
upload. The textured mesh itself is delivered as FBX/GLB for the 3D Importer.

Binary layout follows the rbx-dom specification (format version 0, chunks
META, INST, PROP, PRNT, END, stored uncompressed).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np

from engine.scene import Scene

PART_SHAPE = {"Ball": 0, "Block": 1, "Cylinder": 2}
SURFACE_SMOOTH = 0

T_STRING, T_BOOL, T_FLOAT32, T_CFRAME, T_ENUM, T_REF, T_VECTOR3, T_COLOR3U8, T_OPTCFRAME = 0x01, 0x02, 0x04, 0x10, 0x12, 0x13, 0x0E, 0x1A, 0x1E


@dataclass
class Instance:
    class_name: str
    props: Dict[str, Tuple[str, Any]]
    children: List["Instance"] = field(default_factory=list)
    referent: int = -1

    @property
    def name(self) -> str:
        return self.props["Name"][1]


def _clean_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = m[:3, 3]
    return out


def build_part_model(scene: Scene, name: str, offset: Sequence[float] = (0.0, 0.0, 0.0), anchored: bool = True) -> Instance:
    off = np.asarray(offset, dtype=np.float64)
    children = []
    used_names: Dict[str, int] = {}
    for part in scene.parts:
        mat = scene.materials.get(part.material)
        for prim in part.native:
            if prim.shape not in PART_SHAPE:
                continue
            cf = _clean_rotation(prim.cframe)
            cf[:3, 3] += off
            color = mat.base_color if mat else (0.7, 0.7, 0.7)
            rgb = tuple(int(round(np.clip(c, 0, 1) * 255)) for c in color)
            nm = prim.name or part.name
            used_names[nm] = used_names.get(nm, 0) + 1
            if used_names[nm] > 1:
                nm = f"{nm}{used_names[nm]}"
            size = tuple(float(max(s, 0.05)) for s in prim.size)
            if prim.shape == "Ball":
                d = float(max(size))
                size = (d, d, d)
            children.append(Instance("Part", {
                "Name": ("string", nm),
                "Anchored": ("bool", bool(anchored)),
                "CanCollide": ("bool", True),
                "CastShadow": ("bool", True),
                "Locked": ("bool", False),
                "CFrame": ("CFrame", cf),
                "size": ("Vector3", size),
                "Color3uint8": ("Color3uint8", rgb),
                "Material": ("Enum", int(mat.roblox_material_enum if mat else 272)),
                "shape": ("Enum", PART_SHAPE[prim.shape]),
                "TopSurface": ("Enum", SURFACE_SMOOTH),
                "BottomSurface": ("Enum", SURFACE_SMOOTH),
                "Transparency": ("float", float(1.0 - (mat.opacity if mat else 1.0)) if mat and mat.kind == "glass" else 0.0),
                "Reflectance": ("float", 0.0),
            }))
    pivot = np.eye(4)
    pivot[:3, 3] = off
    return Instance("Model", {"Name": ("string", name), "WorldPivotData": ("OptionalCFrame", pivot)}, children)


def _assign(root: Instance) -> List[Instance]:
    order: List[Instance] = []

    def visit(inst):
        inst.referent = len(order)
        order.append(inst)
        for c in inst.children:
            visit(c)

    visit(root)
    return order


# ---------------------------------------------------------------------------
# Binary encoding helpers
# ---------------------------------------------------------------------------

def _interleave(data: bytes, width: int) -> bytes:
    if not data:
        return b""
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, width)
    return arr.T.copy().tobytes()


def _zigzag(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.int64)
    return np.where(v >= 0, 2 * v, -2 * v - 1).astype(np.uint32)


def _i32_array(values: Sequence[int]) -> bytes:
    return _interleave(_zigzag(np.asarray(values, dtype=np.int64)).astype(">u4").tobytes(), 4)


def _ref_array(values: Sequence[int]) -> bytes:
    v = np.asarray(values, dtype=np.int64)
    delta = np.diff(v, prepend=0)
    return _i32_array(delta)


def _rbx_float_array(values: Sequence[float]) -> bytes:
    bits = np.asarray(values, dtype=">f4").view(">u4").astype(np.uint64)
    rot = ((bits << np.uint64(1)) | (bits >> np.uint64(31))) & np.uint64(0xFFFFFFFF)
    return _interleave(rot.astype(">u4").tobytes(), 4)


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _cframe_values(cfs: Sequence[np.ndarray]) -> bytes:
    head = bytearray()
    for cf in cfs:
        r = cf[:3, :3]
        if np.allclose(r, np.eye(3), atol=1e-7):
            head += b"\x02"
        else:
            head += b"\x00" + struct.pack("<9f", *[float(x) for x in r.reshape(-1)])
    pos = np.array([cf[:3, 3] for cf in cfs], dtype=np.float64)
    return bytes(head) + _rbx_float_array(pos[:, 0]) + _rbx_float_array(pos[:, 1]) + _rbx_float_array(pos[:, 2])


def _encode_values(kind: str, values: List[Any]) -> Tuple[int, bytes]:
    if kind == "string":
        return T_STRING, b"".join(_string(v) for v in values)
    if kind == "bool":
        return T_BOOL, bytes(1 if v else 0 for v in values)
    if kind == "float":
        return T_FLOAT32, _rbx_float_array(values)
    if kind == "Enum":
        return T_ENUM, _interleave(np.asarray(values, dtype=">u4").tobytes(), 4)
    if kind == "Vector3":
        a = np.asarray(values, dtype=np.float64)
        return T_VECTOR3, _rbx_float_array(a[:, 0]) + _rbx_float_array(a[:, 1]) + _rbx_float_array(a[:, 2])
    if kind == "Color3uint8":
        a = np.asarray(values, dtype=np.uint8)
        return T_COLOR3U8, a[:, 0].tobytes() + a[:, 1].tobytes() + a[:, 2].tobytes()
    if kind == "CFrame":
        return T_CFRAME, _cframe_values(values)
    if kind == "OptionalCFrame":
        present = [v is not None for v in values]
        cfs = [v if v is not None else np.eye(4) for v in values]
        return T_OPTCFRAME, bytes([T_CFRAME]) + _cframe_values(cfs) + bytes([T_BOOL]) + bytes(1 if p else 0 for p in present)
    if kind == "Ref":
        return T_REF, _ref_array([v if v is not None else -1 for v in values])
    raise TypeError(f"unsupported property type {kind}")


def _chunk(name: bytes, data: bytes) -> bytes:
    return name.ljust(4, b"\x00") + struct.pack("<III", 0, len(data), 0) + data


def write_rbxm(path: str, root: Instance) -> dict:
    order = _assign(root)
    by_class: Dict[str, List[Instance]] = {}
    for inst in order:
        by_class.setdefault(inst.class_name, []).append(inst)
    class_names = sorted(by_class)
    out = bytearray(b"<roblox!" + b"\x89\xff\x0d\x0a\x1a\x0a" + struct.pack("<H", 0) + struct.pack("<ii", len(class_names), len(order)) + b"\x00" * 8)
    out += _chunk(b"META", struct.pack("<I", 1) + _string("ExplicitAutoJoints") + _string("true"))
    for cid, cname in enumerate(class_names):
        insts = by_class[cname]
        data = struct.pack("<I", cid) + _string(cname) + b"\x00" + struct.pack("<I", len(insts)) + _ref_array([i.referent for i in insts])
        out += _chunk(b"INST", data)
    for cid, cname in enumerate(class_names):
        insts = by_class[cname]
        prop_names = sorted({k for i in insts for k in i.props})
        for pname in prop_names:
            kinds = {i.props[pname][0] for i in insts if pname in i.props}
            if len(kinds) != 1 or any(pname not in i.props for i in insts):
                raise ValueError(f"property {cname}.{pname} must be set on every instance with one type")
            kind = kinds.pop()
            type_id, payload = _encode_values(kind, [i.props[pname][1] for i in insts])
            out += _chunk(b"PROP", struct.pack("<I", cid) + _string(pname) + bytes([type_id]) + payload)
    children, parents = [], []

    def post(inst, parent):
        for c in inst.children:
            post(c, inst)
        children.append(inst.referent)
        parents.append(parent.referent if parent is not None else -1)

    post(root, None)
    out += _chunk(b"PRNT", b"\x00" + struct.pack("<I", len(children)) + _ref_array(children) + _ref_array(parents))
    out += _chunk(b"END\x00", b"</roblox>")
    with open(path, "wb") as fh:
        fh.write(out)
    return {"instances": len(order), "classes": class_names, "bytes": len(out)}


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------

def _f32(v: float) -> str:
    """Shortest text that parses back to the same float32 the binary file stores."""
    return f"{float(np.float32(v)):.9g}"


def _xml_cframe(tag: str, name: Optional[str], cf: np.ndarray, indent: str) -> str:
    r = cf[:3, :3]
    p = cf[:3, 3]
    comps = [("X", p[0]), ("Y", p[1]), ("Z", p[2])] + [(f"R{i}{j}", r[i, j]) for i in range(3) for j in range(3)]
    inner = "".join(f"<{k}>{_f32(v)}</{k}>" for k, v in comps)
    attr = f' name="{name}"' if name else ""
    return f"{indent}<{tag}{attr}>{inner}</{tag}>\n"


def _xml_prop(pname: str, kind: str, value: Any, indent: str) -> str:
    if kind == "string":
        return f'{indent}<string name="{pname}">{escape(value)}</string>\n'
    if kind == "bool":
        return f'{indent}<bool name="{pname}">{"true" if value else "false"}</bool>\n'
    if kind == "float":
        return f'{indent}<float name="{pname}">{_f32(value)}</float>\n'
    if kind == "Enum":
        return f'{indent}<token name="{pname}">{int(value)}</token>\n'
    if kind == "Vector3":
        x, y, z = value
        return f'{indent}<Vector3 name="{pname}"><X>{_f32(x)}</X><Y>{_f32(y)}</Y><Z>{_f32(z)}</Z></Vector3>\n'
    if kind == "Color3uint8":
        r, g, b = value
        packed = (0xFF << 24) | (r << 16) | (g << 8) | b
        return f'{indent}<Color3uint8 name="{pname}">{packed}</Color3uint8>\n'
    if kind == "CFrame":
        return _xml_cframe("CoordinateFrame", pname, value, indent)
    if kind == "OptionalCFrame":
        if value is None:
            return f'{indent}<OptionalCoordinateFrame name="{pname}"></OptionalCoordinateFrame>\n'
        return f'{indent}<OptionalCoordinateFrame name="{pname}">\n' + _xml_cframe("CFrame", None, value, indent + "\t") + f"{indent}</OptionalCoordinateFrame>\n"
    if kind == "Ref":
        return f'{indent}<Ref name="{pname}">{"null" if value is None else "RBX" + str(value)}</Ref>\n'
    raise TypeError(kind)


def write_rbxmx(path: str, root: Instance) -> dict:
    order = _assign(root)
    lines = ['<roblox xmlns:xmime="http://www.w3.org/2005/05/xmlmime" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.roblox.com/roblox.xsd" version="4">\n',
             '\t<Meta name="ExplicitAutoJoints">true</Meta>\n', "\t<External>null</External>\n", "\t<External>nil</External>\n"]

    def emit(inst: Instance, depth: int):
        ind = "\t" * depth
        lines.append(f'{ind}<Item class="{inst.class_name}" referent="RBX{inst.referent}">\n{ind}\t<Properties>\n')
        for pname in sorted(inst.props):
            kind, value = inst.props[pname]
            lines.append(_xml_prop(pname, kind, value, ind + "\t\t"))
        lines.append(f"{ind}\t</Properties>\n")
        for c in inst.children:
            emit(c, depth + 1)
        lines.append(f"{ind}</Item>\n")

    emit(root, 1)
    lines.append("</roblox>\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    return {"instances": len(order)}
