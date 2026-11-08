"""Deliverable package layout, metadata files and ZIP archives."""
from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from typing import Dict, Iterable, List, Optional

COLLISION_ENUM = {"box": "Box", "hull": "Hull", "default": "Default", "precise": "PreciseConvexDecomposition", "none": "Box"}


def safe_name(text: str, fallback: str = "Model", limit: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", (text or "").replace(" ", "_"))
    cleaned = cleaned.strip("_-")[:limit]
    return cleaned or fallback


def write_json(path: str, data) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def zip_directory(src_dir: str, zip_path: str, root_name: Optional[str] = None) -> str:
    """Deterministic ZIP of a folder; only regular files inside ``src_dir``."""
    src_dir = os.path.realpath(src_dir)
    root_name = root_name or os.path.basename(src_dir)
    tmp = zip_path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder, dirs, files in os.walk(src_dir):
            dirs.sort()
            for f in sorted(files):
                full = os.path.realpath(os.path.join(folder, f))
                if not full.startswith(src_dir + os.sep) or os.path.islink(os.path.join(folder, f)):
                    continue
                arc = os.path.join(root_name, os.path.relpath(full, src_dir)).replace(os.sep, "/")
                info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(full, "rb") as fh:
                    zf.writestr(info, fh.read())
    os.replace(tmp, zip_path)
    return zip_path


def luau_import_helper(model_name: str, collision: str, anchored: bool = True) -> str:
    """Studio command-bar script that applies the recommended MeshPart settings."""
    fidelity = COLLISION_ENUM.get(collision, "Hull")
    can_collide = "false" if collision == "none" else "true"
    return f'''local Selection = game:GetService("Selection")

local SETTINGS = {{
	CollisionFidelity = Enum.CollisionFidelity.{fidelity},
	RenderFidelity = Enum.RenderFidelity.Automatic,
	Anchored = {"true" if anchored else "false"},
	CanCollide = {can_collide},
	CastShadow = true,
}}

local target = Selection:Get()[1]
if not target then
	warn("Select the imported {model_name} model in the Explorer, then run this again.")
	return
end

local updated = 0
local function apply(part)
	for key, value in SETTINGS do
		local ok = pcall(function()
			part[key] = value
		end)
		if not ok then
			warn(("Could not set %s on %s"):format(key, part:GetFullName()))
		end
	end
	updated += 1
end

if target:IsA("MeshPart") then
	apply(target)
end
for _, item in target:GetDescendants() do
	if item:IsA("MeshPart") then
		apply(item)
	end
end
print(("Image-to-3D Studio: updated %d MeshPart(s) in %s"):format(updated, target.Name))
'''


def material_info_markdown(name: str, materials: Dict[str, dict], maps: Iterable[str], density: float, resolution: int) -> str:
    lines = [f"# Materials - {name}", "",
             "All parts share **one texture atlas** so the model imports as a single SurfaceAppearance-ready material.", "",
             f"- Atlas resolution: {resolution} x {resolution} px", f"- Texel density: {density:.1f} px per stud", "",
             "| Slot | Kind | Colour | Roughness | Metallic | Roblox material (part version) | Source |", "|---|---|---|---|---|---|---|"]
    for key, m in materials.items():
        color = "#" + "".join(f"{int(round(c * 255)):02X}" for c in m["base_color"])
        lines.append(f"| {m.get('name', key)} | {m['kind']} | `{color}` | {m['roughness']:.2f} | {m['metallic']:.2f} | {m.get('roblox_material', '')} | {m.get('source', '')} |")
    lines += ["", "## Texture maps", ""]
    meaning = {
        "basecolor": "Albedo with mild ambient occlusion baked in (Roblox has no AO slot). sRGB.",
        "normal": "Tangent-space normal map, OpenGL convention (+Y up), as Roblox requires.",
        "roughness": "Grayscale, black = glossy, white = matte.",
        "metallic": "Grayscale, white = metal. Only present when a metal surface exists.",
        "ambient_occlusion": "Grayscale cavity/contact shadowing, kept separate for other engines.",
        "emissive": "Grayscale emissive mask (SurfaceAppearance EmissiveMaskContent).",
        "opacity": "Grayscale opacity; also stored in the base colour alpha channel.",
    }
    for m in maps:
        lines.append(f"- `textures/{m}.png` - {meaning.get(m, '')}")
    lines += ["", "## Roblox SurfaceAppearance mapping", "",
              "| SurfaceAppearance property | File |", "|---|---|",
              "| ColorMap | textures/basecolor.png |", "| NormalMap | textures/normal.png |",
              "| RoughnessMap | textures/roughness.png |", "| MetalnessMap | textures/metallic.png (if present) |",
              "| EmissiveMaskContent | textures/emissive.png (if present) |", ""]
    return "\n".join(lines)


def package_readme(info: dict) -> str:
    s = info
    files = "\n".join(f"- `{f}`" for f in s["files"])
    notes = "\n".join(f"- {n}" for n in s.get("notes", [])) or "- none"
    warn = "\n".join(f"- {w}" for w in s.get("warnings", [])) or "- none"
    return f"""# {s['title']}

Generated by **Image-to-3D Studio** (project `{s['project_id']}`, version {s['version']}).
Reconstruction: **{s['strategy_label']}**, confidence {s['confidence']:.0%}.
No generative AI was used: geometry comes from parametric modelling and classical computer vision.

## Model

| | |
|---|---|
| Size (studs) | {s['size'][0]:.2f} x {s['size'][1]:.2f} x {s['size'][2]:.2f} (W x H x D) |
| Size (metres, 1 stud = 0.28 m) | {s['size'][0] * 0.28:.2f} x {s['size'][1] * 0.28:.2f} x {s['size'][2] * 0.28:.2f} |
| Triangles | {s['triangles']:,} (requested {s['budget']:,}) |
| Vertices | {s['vertices']:,} |
| Meshes in FBX | {s['mesh_count']} |
| Texture atlas | {s['texture_resolution']} px |
| Pivot | {s['pivot']} |

## Files

{files}

## Import into Roblox Studio

**Textured mesh (recommended):** File > Import 3D > choose `model.fbx`
(or `model.glb`). Keep *File Geometry > Scale Unit = Studs*; the file is
authored in studs. Textures are embedded in the FBX and also provided in
`textures/` if you want to assign them manually to a SurfaceAppearance.
Afterwards select the imported model and paste `roblox/apply_import_settings.luau`
into the Command Bar to apply the recommended collision and render settings
(CollisionFidelity {s['collision']}).

**Instant part version (no upload needed):** File > Insert from file > `model.rbxm`
(or `model.rbxmx`). This is a native-part build of the same object using
Roblox materials, useful as a collider, a blockout or a fallback.

## Reconstruction notes

{notes}

## Validation warnings

{warn}

Unseen surfaces (for example the back of an object photographed from the
front) are estimated, not measured.
"""
