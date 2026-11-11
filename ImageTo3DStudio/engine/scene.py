"""Scene containers shared by the reconstruction, texturing and export stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .geometry import Mesh, merge
from .materials import MaterialSpec


@dataclass
class NativePrimitive:
    """A Roblox-native part approximating (part of) a component.

    ``shape`` is one of Block, Cylinder, Ball, Wedge. ``size`` uses Roblox's
    convention (Cylinder: length along local X, diameter on Y/Z). ``cframe`` is a
    4x4 world transform whose rotation is orthonormal.
    """

    shape: str
    size: tuple
    cframe: np.ndarray
    material: str
    name: str = ""


@dataclass
class Part:
    name: str
    mesh: Mesh
    material: str
    frame: np.ndarray = field(default_factory=lambda: np.eye(4))
    grain_axis: int = 0
    uv_grain: str = "u"
    seed: int = 0
    photo: bool = False
    native: List[NativePrimitive] = field(default_factory=list)
    detail: Optional[Mesh] = None
    group: str = ""

    @property
    def triangle_count(self) -> int:
        return self.mesh.triangle_count


@dataclass
class Scene:
    name: str
    parts: List[Part]
    materials: Dict[str, MaterialSpec]
    meta: dict = field(default_factory=dict)

    @property
    def triangle_count(self) -> int:
        return int(sum(p.mesh.triangle_count for p in self.parts))

    @property
    def vertex_count(self) -> int:
        return int(sum(p.mesh.vertex_count for p in self.parts))

    def bounds(self) -> np.ndarray:
        if not self.parts:
            return np.zeros((2, 3))
        b = np.stack([p.mesh.bounds() for p in self.parts])
        return np.stack([b[:, 0].min(0), b[:, 1].max(0)])

    def merged(self) -> Mesh:
        meshes = []
        keys = list(self.materials.keys())
        for p in self.parts:
            m = p.mesh.copy()
            m.face_material = np.full(len(m.faces), keys.index(p.material) if p.material in keys else 0, dtype=np.int64)
            meshes.append(m)
        return merge(meshes, self.name)

    def transform(self, matrix: np.ndarray) -> None:
        for p in self.parts:
            p.mesh.apply_transform(matrix)
            if p.detail is not None:
                p.detail.apply_transform(matrix)
            p.frame = matrix @ p.frame
            for prim in p.native:
                prim.cframe = matrix @ prim.cframe
