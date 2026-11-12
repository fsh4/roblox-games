"""User-editable build settings (one copy is stored with every project version)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

ROBLOX_MESH_TRIANGLE_LIMIT = 20_000
TRIANGLE_PRESETS = (500, 1000, 2000, 5000, 10000, 20000)
TEXTURE_SIZES = (256, 512, 1024, 2048)
QUALITY_LEVELS = ("draft", "standard", "high")
COLLISION_MODES = ("box", "hull", "default", "precise", "none")
PIVOTS = ("bottom", "center", "back", "origin")
GROUPINGS = ("merged", "parts")
SHADING = ("smooth", "flat")
TEXTURE_STYLES = ("auto", "procedural", "photo")
ART_STYLES = ("realistic", "cartoon", "blocky")

_LIMITS = {
    "triangle_budget": (60, 60_000),
    "texture_resolution": (128, 2048),
    "scale": (0.05, 50.0),
    "bevel": (0.0, 3.0),
    "smooth_angle": (0.0, 89.0),
    "ao_in_color": (0.0, 1.0),
    "sensitivity": (0.0, 1.0),
}


@dataclass
class BuildSettings:
    triangle_budget: int = 5000
    texture_resolution: int = 1024
    quality: str = "standard"
    dimensions: Optional[List[float]] = None
    scale: float = 1.0
    shading: str = "smooth"
    smooth_angle: float = 40.0
    bevel: float = 1.0
    collision: str = "hull"
    pivot: str = "bottom"
    mesh_grouping: str = "merged"
    texture_style: str = "auto"
    art_style: str = "realistic"
    ao_in_color: float = 0.55
    low_poly: bool = False
    sensitivity: float = 0.5
    param_overrides: Dict[str, float] = field(default_factory=dict)
    material_overrides: Dict[str, dict] = field(default_factory=dict)

    def validated(self) -> "BuildSettings":
        s = BuildSettings(**asdict(self))
        for key, (lo, hi) in _LIMITS.items():
            val = getattr(s, key)
            setattr(s, key, type(val)(min(max(val, lo), hi)))
        s.triangle_budget = int(s.triangle_budget)
        s.texture_resolution = min(TEXTURE_SIZES, key=lambda r: abs(r - s.texture_resolution))
        if s.quality not in QUALITY_LEVELS:
            s.quality = "standard"
        if s.collision not in COLLISION_MODES:
            s.collision = "hull"
        if s.pivot not in PIVOTS:
            s.pivot = "bottom"
        if s.mesh_grouping not in GROUPINGS:
            s.mesh_grouping = "merged"
        if s.shading not in SHADING:
            s.shading = "smooth"
        if s.texture_style not in TEXTURE_STYLES:
            s.texture_style = "auto"
        if s.art_style == "studs":
            s.art_style = "blocky"
        if s.art_style not in ART_STYLES:
            s.art_style = "realistic"
        if s.dimensions is not None:
            dims = [None if d in (None, 0) else float(min(max(float(d), 0.1), 2048.0)) for d in list(s.dimensions)[:3]]
            s.dimensions = dims if any(d is not None for d in dims) else None
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Optional[dict]) -> "BuildSettings":
        if not data:
            return BuildSettings()
        allowed = set(BuildSettings.__dataclass_fields__)
        return BuildSettings(**{k: v for k, v in data.items() if k in allowed}).validated()

    def diff(self, other: "BuildSettings") -> Dict[str, tuple]:
        a, b = self.to_dict(), other.to_dict()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}
