"""Material definitions, presets and Roblox material mapping."""
from __future__ import annotations

import colorsys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

KINDS = (
    "wood", "fabric", "metal", "painted", "plastic", "leather",
    "ceramic", "glass", "stone", "rubber", "emissive", "photo", "generic",
)

ROBLOX_MATERIAL = {
    "wood": ("Wood", 512),
    "fabric": ("Fabric", 1312),
    "metal": ("Metal", 1088),
    "painted": ("SmoothPlastic", 272),
    "plastic": ("SmoothPlastic", 272),
    "leather": ("Leather", 2309),
    "ceramic": ("SmoothPlastic", 272),
    "glass": ("Glass", 1568),
    "stone": ("Concrete", 816),
    "rubber": ("Rubber", 2311),
    "emissive": ("Neon", 288),
    "photo": ("SmoothPlastic", 272),
    "generic": ("SmoothPlastic", 272),
}

_DEFAULTS = {
    "wood": dict(roughness=0.62, metallic=0.0, pattern_scale=0.9, normal_strength=0.55, color_variation=0.22),
    "fabric": dict(roughness=0.92, metallic=0.0, pattern_scale=0.06, normal_strength=0.35, color_variation=0.05),
    "metal": dict(roughness=0.32, metallic=1.0, pattern_scale=0.4, normal_strength=0.15, color_variation=0.05),
    "painted": dict(roughness=0.45, metallic=0.0, pattern_scale=1.0, normal_strength=0.08, color_variation=0.03),
    "plastic": dict(roughness=0.40, metallic=0.0, pattern_scale=1.0, normal_strength=0.05, color_variation=0.03),
    "leather": dict(roughness=0.55, metallic=0.0, pattern_scale=0.05, normal_strength=0.45, color_variation=0.08),
    "ceramic": dict(roughness=0.18, metallic=0.0, pattern_scale=1.0, normal_strength=0.03, color_variation=0.02),
    "glass": dict(roughness=0.06, metallic=0.0, pattern_scale=1.0, normal_strength=0.0, color_variation=0.0),
    "stone": dict(roughness=0.86, metallic=0.0, pattern_scale=0.5, normal_strength=0.5, color_variation=0.12),
    "rubber": dict(roughness=0.9, metallic=0.0, pattern_scale=0.3, normal_strength=0.1, color_variation=0.03),
    "emissive": dict(roughness=0.5, metallic=0.0, pattern_scale=1.0, normal_strength=0.0, color_variation=0.0),
    "photo": dict(roughness=0.6, metallic=0.0, pattern_scale=1.0, normal_strength=0.1, color_variation=0.0),
    "generic": dict(roughness=0.55, metallic=0.0, pattern_scale=1.0, normal_strength=0.05, color_variation=0.04),
}

DISPLAY_NAMES = {
    "wood": "Wood", "fabric": "Fabric", "metal": "Metal", "painted": "Painted",
    "plastic": "Plastic", "leather": "Leather", "ceramic": "Ceramic", "glass": "Glass",
    "stone": "Stone", "rubber": "Rubber", "emissive": "Light emitter",
    "photo": "Photo-projected", "generic": "Generic",
}


@dataclass
class MaterialSpec:
    key: str
    kind: str = "generic"
    base_color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    roughness: float = 0.55
    metallic: float = 0.0
    color_variation: float = 0.05
    pattern_scale: float = 1.0
    normal_strength: float = 0.1
    emissive: float = 0.0
    opacity: float = 1.0
    quilted: bool = False
    exemplar: Optional[str] = None
    exemplar_density: float = 60.0
    secondary_color: Optional[Tuple[float, float, float]] = None
    source: str = "default"
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            self.kind = "generic"
        self.base_color = tuple(float(np.clip(c, 0.0, 1.0)) for c in self.base_color)
        if self.secondary_color is not None:
            self.secondary_color = tuple(float(np.clip(c, 0.0, 1.0)) for c in self.secondary_color)
        self.roughness = float(np.clip(self.roughness, 0.02, 1.0))
        self.metallic = float(np.clip(self.metallic, 0.0, 1.0))
        self.opacity = float(np.clip(self.opacity, 0.05, 1.0))
        if not self.name:
            self.name = f"{DISPLAY_NAMES.get(self.kind, 'Material')} ({self.key})"

    @property
    def roblox_material(self) -> str:
        return ROBLOX_MATERIAL.get(self.kind, ("SmoothPlastic", 272))[0]

    @property
    def roblox_material_enum(self) -> int:
        return ROBLOX_MATERIAL.get(self.kind, ("SmoothPlastic", 272))[1]

    @property
    def hex_color(self) -> str:
        return "#" + "".join(f"{int(round(c * 255)):02X}" for c in self.base_color)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "MaterialSpec":
        allowed = {f for f in MaterialSpec.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in allowed}
        for key in ("base_color", "secondary_color"):
            if clean.get(key) is not None:
                clean[key] = tuple(clean[key])
        return MaterialSpec(**clean)


def make_material(key: str, kind: str, color: Sequence[float], **overrides) -> MaterialSpec:
    params = dict(_DEFAULTS.get(kind, _DEFAULTS["generic"]))
    params.update(overrides)
    return MaterialSpec(key=key, kind=kind, base_color=tuple(color), **params)


def parse_hex_color(text: str) -> Optional[Tuple[float, float, float]]:
    t = text.strip().lstrip("#")
    if len(t) == 3:
        t = "".join(ch * 2 for ch in t)
    if len(t) != 6:
        return None
    try:
        return tuple(int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None


def infer_kind(color_srgb: Sequence[float], texture_strength: float, orientation_coherence: float, specular_ratio: float, candidates: Sequence[str], prior: Optional[str] = None) -> str:
    """Pick a material kind for a region using colour and texture statistics.

    ``texture_strength`` is the normalised local contrast, ``orientation_coherence``
    the structure-tensor coherence (grain), ``specular_ratio`` the share of near-white
    highlight pixels. The choice is limited to the template's plausible candidates.
    """
    r, g, b = [float(c) for c in color_srgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hue = h * 360.0
    scores: Dict[str, float] = {}
    for kind in candidates:
        score = 0.0
        if kind == "wood":
            score += 1.2 if 15 <= hue <= 50 and 0.15 <= s <= 0.85 and 0.15 <= l <= 0.85 else -0.6
            score += 1.0 * orientation_coherence + 0.5 * min(texture_strength * 6, 1.0)
        elif kind == "fabric":
            score += 0.6 + 0.4 * (1.0 - min(texture_strength * 5, 1.0))
            score += 0.3 if specular_ratio < 0.02 else -0.3
        elif kind == "metal":
            score += 1.0 if s < 0.15 else -0.8
            score += 1.2 * min(specular_ratio * 20, 1.0)
        elif kind in ("painted", "plastic"):
            score += 0.5 + 0.3 * (1.0 - min(texture_strength * 8, 1.0))
            score += 0.2 if s > 0.25 else 0.0
        elif kind == "leather":
            score += 0.4 if (hue < 40 or l < 0.25) else 0.0
            score += 0.3 * min(specular_ratio * 15, 1.0)
        elif kind == "ceramic":
            score += 0.4 + 0.6 * min(specular_ratio * 15, 1.0) if l > 0.6 else 0.0
        elif kind == "glass":
            score += 0.1
        elif kind == "stone":
            score += 0.5 if s < 0.2 else 0.0
            score += 0.4 * min(texture_strength * 6, 1.0)
        else:
            score += 0.2
        if kind == prior:
            score += 0.45
        scores[kind] = score
    return max(scores.items(), key=lambda kv: kv[1])[0] if scores else "generic"


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def describe(materials: Dict[str, MaterialSpec]) -> str:
    lines = []
    for m in materials.values():
        lines.append(f"- **{m.name}** - kind `{m.kind}`, colour `{m.hex_color}`, roughness {m.roughness:.2f}, metallic {m.metallic:.2f}, Roblox material `{m.roblox_material}` (source: {m.source})")
    return "\n".join(lines)
