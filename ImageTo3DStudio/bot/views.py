"""Buttons, select menus and modals.

Result-card buttons are ``DynamicItem``s whose custom IDs encode the action,
project and version, so they keep working after the bot restarts. Panels that
only live for one session (edit panel, pickers) are ordinary views with a
timeout; every handler re-checks ownership before acting.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import discord

from engine.materials import KINDS, parse_hex_color
from engine.procedural import TEMPLATES
from engine.settings import COLLISION_MODES, PIVOTS, QUALITY_LEVELS, TEXTURE_SIZES, TEXTURE_STYLES, BuildSettings
from storage.projects import UserDefaults

from . import embeds

if TYPE_CHECKING:
    from .interactions import StudioService

CATEGORY_CHOICES = ["auto", "bed", "chair", "table", "sofa", "cabinet", "shelf", "lamp", "house", "car", "box", "revolve", "extrude", "inflate"]


def _svc(interaction: discord.Interaction) -> "StudioService":
    return interaction.client.studio


def _num(text: str, lo: float, hi: float, cast=float) -> Optional[float]:
    t = (text or "").strip().replace(",", ".").lower().rstrip("x")
    if not t:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k)?", t)
    if not m:
        raise ValueError(f"'{text}' is not a number")
    v = float(m.group(1)) * (1000 if m.group(2) else 1)
    if not lo <= v <= hi:
        raise ValueError(f"{v:g} is outside {lo:g}-{hi:g}")
    return cast(v)


def _select(options: List[str], current: str, placeholder: str) -> discord.ui.Select:
    return discord.ui.Select(placeholder=placeholder, options=[discord.SelectOption(label=o, value=o, default=(o == current)) for o in options])


# ---------------------------------------------------------------------- job message

class JobView(discord.ui.View):
    def __init__(self, svc: "StudioService", job_id: str):
        super().__init__(timeout=None)
        self.add_item(CancelJobButton(job_id, svc.icons.get("cross")))


class CancelJobButton(discord.ui.DynamicItem[discord.ui.Button], template=r"i3dc:(?P<job>[0-9a-f]{12})"):
    def __init__(self, job_id: str, emoji=None):
        super().__init__(discord.ui.Button(label="Cancel", style=discord.ButtonStyle.secondary, emoji=emoji, custom_id=f"i3dc:{job_id}"))
        self.job_id = job_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["job"], item.emoji)

    async def callback(self, interaction: discord.Interaction):
        await _svc(interaction).cancel_job(interaction, self.job_id)


class RetryView(discord.ui.View):
    def __init__(self, svc: "StudioService", job_id: str):
        super().__init__(timeout=None)
        self.add_item(RetryButton(job_id, svc.icons.get("refresh")))


class RetryButton(discord.ui.DynamicItem[discord.ui.Button], template=r"i3dr:(?P<job>[0-9a-f]{12})"):
    def __init__(self, job_id: str, emoji=None):
        super().__init__(discord.ui.Button(label="Retry", style=discord.ButtonStyle.primary, emoji=emoji, custom_id=f"i3dr:{job_id}"))
        self.job_id = job_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["job"], item.emoji)

    async def callback(self, interaction: discord.Interaction):
        await _svc(interaction).retry_job(interaction, self.job_id)


# ---------------------------------------------------------------------- result card

ACTIONS = {
    "zip": ("Download ZIP", discord.ButtonStyle.success, "download"),
    "files": ("Files", discord.ButtonStyle.secondary, "layers"),
    "edit": ("Edit", discord.ButtonStyle.primary, "sliders"),
    "versions": ("Versions", discord.ButtonStyle.secondary, "clock"),
    "export": ("Formats", discord.ButtonStyle.secondary, "cube"),
}


class ProjectAction(discord.ui.DynamicItem[discord.ui.Button], template=r"i3d:(?P<action>[a-z]+):(?P<pid>[0-9a-f]{8}):(?P<ver>[0-9]+)"):
    def __init__(self, action: str, pid: str, ver: int, emoji=None):
        label, style, _ = ACTIONS.get(action, (action.title(), discord.ButtonStyle.secondary, None))
        super().__init__(discord.ui.Button(label=label, style=style, emoji=emoji, custom_id=f"i3d:{action}:{pid}:{ver}"))
        self.action, self.pid, self.ver = action, pid, int(ver)

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], match["pid"], int(match["ver"]), item.emoji)

    async def callback(self, interaction: discord.Interaction):
        await handle_action(interaction, self.action, self.pid, self.ver)


class ResultView(discord.ui.View):
    def __init__(self, svc: "StudioService", pid: str, ver: int):
        super().__init__(timeout=None)
        for action, (_, _, icon) in ACTIONS.items():
            self.add_item(ProjectAction(action, pid, ver, svc.icons.get(icon)))


async def handle_action(interaction: discord.Interaction, action: str, pid: str, ver: int) -> None:
    svc = _svc(interaction)
    project = await svc.load_project(interaction, pid, edit=action == "edit")
    if project is None:
        return
    if action == "zip":
        v = await asyncio.to_thread(svc.projects.get_version, pid, ver)
        if v is None or not v.zip_path:
            await svc.reply(interaction, content="This version has no package (it may have been archived; rebuild it from Versions).", ephemeral=True)
            return
        await svc.send_file(interaction, Path(v.zip_path), f"{project.name} v{ver}")
    elif action == "files":
        v = await asyncio.to_thread(svc.projects.get_version, pid, ver)
        if v is None or not v.package_dir:
            await svc.reply(interaction, content="This version has no package.", ephemeral=True)
            return
        await svc.reply(interaction, content=f"Pick a file from **{project.name} v{ver}**:", view=FilePicker(Path(v.package_dir)), ephemeral=True)
    elif action == "edit":
        await open_edit_panel(interaction, project, ver)
    elif action == "versions":
        await open_versions(interaction, project)
    elif action == "export":
        await svc.reply(interaction, content=f"Export **{project.name} v{ver}** as:", view=ExportPicker(project.id, ver), ephemeral=True)


class FilePicker(discord.ui.View):
    FILES = [("model.fbx", "FBX mesh (Roblox Import 3D)"), ("model.rbxm", "Roblox part model"), ("model.rbxmx", "Roblox part model (XML)"),
             ("model.glb", "glTF binary"), ("textures/basecolor.png", "Base colour"), ("textures/normal.png", "Normal map"),
             ("textures/roughness.png", "Roughness"), ("textures/metallic.png", "Metalness"), ("textures/ambient_occlusion.png", "Ambient occlusion"),
             ("textures/emissive.png", "Emissive mask"), ("previews/preview.png", "Preview sheet"), ("previews/turntable.gif", "Turntable"),
             ("previews/viewer.html", "Interactive viewer"), ("materials/materials.json", "Materials"), ("geometry/mesh_statistics.json", "Mesh statistics"),
             ("roblox/apply_import_settings.luau", "Studio import helper"), ("README.md", "Read me")]

    def __init__(self, package_dir: Path):
        super().__init__(timeout=600)
        self.package_dir = package_dir
        opts = [discord.SelectOption(label=desc, value=rel, description=rel) for rel, desc in self.FILES if (package_dir / rel).exists()]
        sel = discord.ui.Select(placeholder="Choose a file", options=opts[:25])
        sel.callback = self._pick
        self.sel = sel
        self.add_item(sel)

    async def _pick(self, interaction: discord.Interaction):
        rel = self.sel.values[0]
        if rel not in {r for r, _ in self.FILES}:
            return
        await _svc(interaction).send_file(interaction, self.package_dir / rel)


class ExportPicker(discord.ui.View):
    FORMATS = [("fbx", "FBX - Roblox Import 3D, Blender, Unity"), ("glb", "glTF binary - PBR, Roblox, web"), ("obj", "OBJ + MTL + textures (zip)"),
               ("stl", "STL - geometry only, 3D printing"), ("rbxm", "RBXM - native Roblox parts"), ("rbxmx", "RBXMX - native parts, XML"),
               ("textures", "All texture maps (zip)"), ("zip", "Complete package (zip)")]

    def __init__(self, pid: str, ver: int):
        super().__init__(timeout=600)
        self.pid, self.ver = pid, ver
        sel = discord.ui.Select(placeholder="Choose a format", options=[discord.SelectOption(label=f.upper(), value=f, description=d) for f, d in self.FORMATS])
        sel.callback = self._pick
        self.sel = sel
        self.add_item(sel)

    async def _pick(self, interaction: discord.Interaction):
        svc = _svc(interaction)
        project = await svc.load_project(interaction, self.pid)
        if project:
            await svc.export(interaction, project, self.ver, self.sel.values[0])


# ---------------------------------------------------------------------- upload modal

class UploadModal(discord.ui.Modal, title="New 3D model"):
    images = discord.ui.Label(text="Photos", description="1 to 6 images of the object (PNG with transparency works best)",
                              component=discord.ui.FileUpload(min_values=1, max_values=6, required=True))
    model_name = discord.ui.Label(text="Name", description="Optional", component=discord.ui.TextInput(required=False, max_length=60, placeholder="Oak bed"))
    category = discord.ui.Label(text="Object type", component=discord.ui.Select(options=[discord.SelectOption(label=c, value=c, default=c == "auto") for c in CATEGORY_CHOICES], required=False))
    views = discord.ui.Label(text="View labels (multi-photo)", description="e.g. front,side,back - leave empty for a walk-around", component=discord.ui.TextInput(required=False, max_length=60))

    async def on_submit(self, interaction: discord.Interaction):
        files = list(self.images.component.values)
        cat = (self.category.component.values or ["auto"])[0]
        await _svc(interaction).start_create(interaction, files, category=cat, views=self.views.component.value or "", name=self.model_name.component.value or None)


class UploadPrompt(discord.ui.View):
    def __init__(self, svc: "StudioService"):
        super().__init__(timeout=600)
        btn = discord.ui.Button(label="Upload photos", style=discord.ButtonStyle.primary, emoji=svc.icons.get("cube"))
        btn.callback = self._open
        self.add_item(btn)

    async def _open(self, interaction: discord.Interaction):
        await interaction.response.send_modal(UploadModal())


# ---------------------------------------------------------------------- edit panel

EDIT_OPTIONS = [
    ("budget", "Triangle budget", "Exact or preset triangle count"),
    ("size", "Size and scale", "Width / height / depth in studs, or a scale factor"),
    ("textures", "Textures", "Resolution, style, ambient occlusion"),
    ("surface", "Shading and bevel", "Smooth/flat, edge rounding"),
    ("materials", "Materials and colours", "Colour, kind, roughness, metalness"),
    ("shape", "Shape parameters", "Template dimensions, counts, toggles"),
    ("roblox", "Roblox options", "Collision, pivot, mesh grouping"),
    ("lower", "Lower-poly version", "Half the triangles, flat low-poly look"),
    ("higher", "Higher-detail version", "Double the triangles, high quality"),
    ("rebuild", "Rebuild UVs and textures", "Same settings, fresh atlas"),
    ("reanalyse", "Re-analyse photo", "Different object type, mode or sensitivity"),
]


async def open_edit_panel(interaction: discord.Interaction, project, version: Optional[int] = None) -> None:
    svc = _svc(interaction)
    v = await asyncio.to_thread(svc.projects.get_version, project.id, version or project.current_version) if (version or project.current_version) else None
    if v is None:
        await svc.reply(interaction, content="This project has no finished version yet.", ephemeral=True)
        return
    emb = embeds.info_embed(f"Edit {project.name}  |  based on v{v.number}", embeds.settings_summary(v.settings))
    await svc.reply(interaction, embed=emb, view=EditPanel(svc, project.id, v.number, v.settings), ephemeral=True)


class EditPanel(discord.ui.View):
    def __init__(self, svc: "StudioService", pid: str, ver: int, settings: BuildSettings):
        super().__init__(timeout=900)
        self.pid, self.ver, self.settings = pid, ver, settings
        sel = discord.ui.Select(placeholder="What do you want to change?", options=[discord.SelectOption(label=l, value=k, description=d) for k, l, d in EDIT_OPTIONS])
        sel.callback = self._choose
        self.sel = sel
        self.add_item(sel)
        for action in ("versions", "zip"):
            self.add_item(ProjectAction(action, pid, ver, svc.icons.get(ACTIONS[action][2])))

    async def _choose(self, interaction: discord.Interaction):
        svc = _svc(interaction)
        project = await svc.load_project(interaction, self.pid, edit=True)
        if project is None:
            return
        key = self.sel.values[0]
        s = self.settings
        if key == "budget":
            await interaction.response.send_modal(BudgetModal(self.pid, self.ver, s))
        elif key == "size":
            await interaction.response.send_modal(SizeModal(self.pid, self.ver, s))
        elif key == "textures":
            await interaction.response.send_modal(TextureModal(self.pid, self.ver, s))
        elif key == "surface":
            await interaction.response.send_modal(SurfaceModal(self.pid, self.ver, s))
        elif key == "roblox":
            await interaction.response.send_modal(RobloxModal(self.pid, self.ver, s))
        elif key == "reanalyse":
            await interaction.response.send_modal(ReanalyseModal(self.pid, self.ver, s))
        elif key == "materials":
            spec = svc.read_spec(self.pid) or {}
            mats = spec.get("materials", {})
            if not mats:
                await svc.reply(interaction, content="This reconstruction has a single photo-projected surface; its colours come from the photo.", ephemeral=True)
                return
            await svc.reply(interaction, content="Choose a material:", view=MaterialPicker(self.pid, self.ver, s, mats), ephemeral=True)
        elif key == "shape":
            spec = svc.read_spec(self.pid) or {}
            if spec.get("strategy") != "template":
                await svc.reply(interaction, content="Shape parameters exist for template reconstructions (furniture, buildings, vehicles). For this model use Size and scale: the three sizes are width, height and depth, so a flatter or deeper body is set there.", ephemeral=True)
                return
            await svc.reply(interaction, content="Choose a parameter:", view=ParamPicker(self.pid, self.ver, s, spec), ephemeral=True)
        elif key == "lower":
            ns = _copy(s, triangle_budget=max(60, s.triangle_budget // 2), low_poly=True)
            await svc.queue_build(interaction, project, self.ver, ns, f"lower-poly from v{self.ver}")
        elif key == "higher":
            ns = _copy(s, triangle_budget=min(60000, s.triangle_budget * 2), quality="high", low_poly=False)
            await svc.queue_build(interaction, project, self.ver, ns, f"higher detail from v{self.ver}")
        elif key == "rebuild":
            await svc.queue_build(interaction, project, self.ver, s, f"rebuild of v{self.ver}")


def _copy(s: BuildSettings, **changes) -> BuildSettings:
    d = s.to_dict()
    d.update(changes)
    return BuildSettings.from_dict(d)


class _EditModal(discord.ui.Modal):
    def __init__(self, pid: str, ver: int, settings: BuildSettings, title: str):
        super().__init__(title=title, timeout=900)
        self.pid, self.ver, self.settings = pid, ver, settings

    async def submit(self, interaction: discord.Interaction, changes: dict, note: str, kind: str = "edit", reanalyse: Optional[dict] = None):
        svc = _svc(interaction)
        project = await svc.load_project(interaction, self.pid, edit=True)
        if project is None:
            return
        await svc.queue_build(interaction, project, self.ver, _copy(self.settings, **changes), note, kind, reanalyse)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        msg = str(error) if isinstance(error, ValueError) else "Something went wrong while applying the change."
        await _svc(interaction).reply(interaction, embed=embeds.error_embed("Not applied", msg), ephemeral=True)


class BudgetModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings):
        super().__init__(pid, ver, s, "Triangle budget")
        self.preset = discord.ui.Label(text="Preset", component=discord.ui.Select(required=False, options=[
            discord.SelectOption(label=f"{n:,} triangles", value=str(n), default=n == s.triangle_budget) for n in (500, 1000, 2000, 5000, 10000, 20000)]))
        self.custom = discord.ui.Label(text="Custom count", description="60 - 60,000. Overrides the preset. Meshes above 20,000 are split for Roblox.",
                                       component=discord.ui.TextInput(required=False, max_length=7, placeholder=str(s.triangle_budget)))
        self.add_item(self.preset)
        self.add_item(self.custom)

    async def on_submit(self, interaction: discord.Interaction):
        custom = _num(self.custom.component.value, 60, 60000, int)
        preset = self.preset.component.values[0] if self.preset.component.values else None
        budget = custom if custom is not None else (int(preset) if preset else self.settings.triangle_budget)
        await self.submit(interaction, {"triangle_budget": budget}, f"triangle budget {budget:,}")


class SizeModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings):
        super().__init__(pid, ver, s, "Size and scale")
        dims = s.dimensions or [None, None, None]
        self.w = discord.ui.Label(text="Width (studs, X)", description="Empty = keep proportions", component=discord.ui.TextInput(required=False, max_length=8, default=f"{dims[0]:g}" if dims[0] else None))
        self.h = discord.ui.Label(text="Height (studs, Y)", component=discord.ui.TextInput(required=False, max_length=8, default=f"{dims[1]:g}" if dims[1] else None))
        self.d = discord.ui.Label(text="Depth (studs, Z)", component=discord.ui.TextInput(required=False, max_length=8, default=f"{dims[2]:g}" if dims[2] else None))
        self.scale = discord.ui.Label(text="Scale factor", description="Multiplies the result, 1 = unchanged (1 stud = 0.28 m)", component=discord.ui.TextInput(required=False, max_length=6, default=f"{s.scale:g}"))
        for item in (self.w, self.h, self.d, self.scale):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        dims = [_num(x.component.value, 0.1, 2048) for x in (self.w, self.h, self.d)]
        scale = _num(self.scale.component.value, 0.05, 50) or 1.0
        await self.submit(interaction, {"dimensions": dims if any(dims) else None, "scale": scale}, "size " + " x ".join("auto" if d is None else f"{d:g}" for d in dims) + f", scale {scale:g}")


class TextureModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings):
        super().__init__(pid, ver, s, "Textures")
        self.res = discord.ui.Label(text="Atlas resolution", description="Roblox recommends up to 1024 for most props", component=discord.ui.Select(options=[discord.SelectOption(label=f"{r} x {r}", value=str(r), default=r == s.texture_resolution) for r in TEXTURE_SIZES]))
        self.style = discord.ui.Label(text="Texture source", description="auto: photo samples when reliable, photo: prefer photo, procedural: synthetic only", component=_select(list(TEXTURE_STYLES), s.texture_style, "Style"))
        self.ao = discord.ui.Label(text="Ambient occlusion in colour (0-1)", component=discord.ui.TextInput(required=False, max_length=4, default=f"{s.ao_in_color:g}"))
        for item in (self.res, self.style, self.ao):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        res = int(self.res.component.values[0]) if self.res.component.values else self.settings.texture_resolution
        style = self.style.component.values[0] if self.style.component.values else self.settings.texture_style
        ao = _num(self.ao.component.value, 0, 1)
        await self.submit(interaction, {"texture_resolution": res, "texture_style": style, "ao_in_color": self.settings.ao_in_color if ao is None else ao}, f"textures {res}px {style}")


class SurfaceModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings):
        super().__init__(pid, ver, s, "Shading and bevel")
        self.shading = discord.ui.Label(text="Shading", component=_select(["smooth", "flat"], s.shading, "Shading"))
        self.bevel = discord.ui.Label(text="Bevel intensity (0-3)", description="0 = sharp edges, 1 = measured, 2+ = softer", component=discord.ui.TextInput(required=False, max_length=4, default=f"{s.bevel:g}"))
        self.quality = discord.ui.Label(text="Geometry quality", component=_select(list(QUALITY_LEVELS), s.quality, "Quality"))
        for item in (self.shading, self.bevel, self.quality):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        bevel = _num(self.bevel.component.value, 0, 3)
        await self.submit(interaction, {
            "shading": (self.shading.component.values or [self.settings.shading])[0],
            "bevel": self.settings.bevel if bevel is None else bevel,
            "quality": (self.quality.component.values or [self.settings.quality])[0],
        }, "shading / bevel / quality")


class RobloxModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings):
        super().__init__(pid, ver, s, "Roblox options")
        self.coll = discord.ui.Label(text="CollisionFidelity", component=_select(list(COLLISION_MODES), s.collision, "Collision"))
        self.pivot = discord.ui.Label(text="Pivot", description="bottom = sits on the floor", component=_select(list(PIVOTS), s.pivot, "Pivot"))
        self.group = discord.ui.Label(text="Meshes", description="merged = one MeshPart, parts = one per component group", component=_select(["merged", "parts"], s.mesh_grouping, "Grouping"))
        for item in (self.coll, self.pivot, self.group):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        await self.submit(interaction, {
            "collision": (self.coll.component.values or [self.settings.collision])[0],
            "pivot": (self.pivot.component.values or [self.settings.pivot])[0],
            "mesh_grouping": (self.group.component.values or [self.settings.mesh_grouping])[0],
        }, "Roblox options")


class ReanalyseModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings):
        super().__init__(pid, ver, s, "Re-analyse the photo")
        self.cat = discord.ui.Label(text="Object type", component=_select(CATEGORY_CHOICES, "auto", "Object type"))
        self.mode = discord.ui.Label(text="Mode", description="auto, single (generic shape) or multi (several photos)", component=_select(["auto", "single", "multi"], "auto", "Mode"))
        self.sens = discord.ui.Label(text="Background sensitivity (0-1)", description="Higher keeps more of the object when it resembles the background", component=discord.ui.TextInput(required=False, max_length=4, default="0.5"))
        for item in (self.cat, self.mode, self.sens):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        opts = {"category": (self.cat.component.values or ["auto"])[0], "mode": (self.mode.component.values or ["auto"])[0], "sensitivity": _num(self.sens.component.value, 0, 1) or 0.5}
        await self.submit(interaction, {}, f"re-analysed as {opts['category']}", "reanalyze", opts)


class MaterialPicker(discord.ui.View):
    def __init__(self, pid: str, ver: int, s: BuildSettings, mats: dict):
        super().__init__(timeout=600)
        self.pid, self.ver, self.s, self.mats = pid, ver, s, mats
        sel = discord.ui.Select(placeholder="Material", options=[discord.SelectOption(label=(m.get("name") or k)[:100], value=k, description=f"{m.get('kind')} #{''.join(f'{int(c * 255):02X}' for c in m.get('base_color', (0, 0, 0)))}") for k, m in list(mats.items())[:25]])
        sel.callback = self._pick
        self.sel = sel
        self.add_item(sel)

    async def _pick(self, interaction: discord.Interaction):
        key = self.sel.values[0]
        base = dict(self.mats.get(key, {}))
        base.update(self.s.material_overrides.get(key, {}))
        await interaction.response.send_modal(MaterialModal(self.pid, self.ver, self.s, key, base))


class MaterialModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings, key: str, mat: dict):
        super().__init__(pid, ver, s, f"Material: {(mat.get('name') or key)[:30]}")
        self.key = key
        col = mat.get("color") if isinstance(mat.get("color"), str) else "#" + "".join(f"{int(round(c * 255)):02X}" for c in mat.get("base_color", (0.7, 0.7, 0.7)))
        self.color = discord.ui.Label(text="Colour (hex)", component=discord.ui.TextInput(max_length=7, default=col))
        self.kind = discord.ui.Label(text="Kind", component=_select([k for k in KINDS if k != "photo"], mat.get("kind", "generic"), "Kind"))
        self.rough = discord.ui.Label(text="Roughness (0 glossy - 1 matte)", component=discord.ui.TextInput(required=False, max_length=4, default=f"{float(mat.get('roughness', 0.5)):.2f}"))
        self.metal = discord.ui.Label(text="Metalness (0-1)", component=discord.ui.TextInput(required=False, max_length=4, default=f"{float(mat.get('metallic', 0.0)):.2f}"))
        for item in (self.color, self.kind, self.rough, self.metal):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        if parse_hex_color(self.color.component.value) is None:
            raise ValueError("Colour must look like #D9B07A")
        ov = dict(self.settings.material_overrides)
        entry = {"color": self.color.component.value.strip()}
        if self.kind.component.values:
            entry["kind"] = self.kind.component.values[0]
        r = _num(self.rough.component.value, 0, 1)
        m = _num(self.metal.component.value, 0, 1)
        if r is not None:
            entry["roughness"] = r
        if m is not None:
            entry["metallic"] = m
        ov[self.key] = entry
        await self.submit(interaction, {"material_overrides": ov}, f"material {self.key} {entry['color']}")


class ParamPicker(discord.ui.View):
    def __init__(self, pid: str, ver: int, s: BuildSettings, spec: dict):
        super().__init__(timeout=600)
        self.pid, self.ver, self.s = pid, ver, s
        t = TEMPLATES[spec["template"]]
        self.template = t
        values = {**spec.get("params", {}), **s.param_overrides}
        opts = []
        for k, d in list(t.params.items())[:25]:
            v = values.get(k, d.default)
            shown = f"{v:.2f} studs" if d.kind == "length" else (f"{int(round(v))}" if d.kind in ("count", "toggle") else f"{v:.2f}")
            opts.append(discord.SelectOption(label=d.label[:100], value=k, description=f"now {shown}  (range {d.lo:g}-{d.hi:g})"))
        self.values = values
        sel = discord.ui.Select(placeholder="Parameter", options=opts)
        sel.callback = self._pick
        self.sel = sel
        self.add_item(sel)

    async def _pick(self, interaction: discord.Interaction):
        k = self.sel.values[0]
        await interaction.response.send_modal(ParamModal(self.pid, self.ver, self.s, self.template, k, self.values.get(k, self.template.params[k].default)))


class ParamModal(_EditModal):
    def __init__(self, pid, ver, s: BuildSettings, template, key: str, current: float):
        d = template.params[key]
        super().__init__(pid, ver, s, d.label[:45])
        self.key, self.d = key, d
        hint = {"length": "studs", "count": "whole number", "toggle": "1 = on, 0 = off", "ratio": "ratio"}.get(d.kind, "")
        self.val = discord.ui.Label(text=f"{d.label} ({hint})", description=f"Allowed {d.lo:g} - {d.hi:g}", component=discord.ui.TextInput(max_length=8, default=f"{current:g}"))
        self.add_item(self.val)

    async def on_submit(self, interaction: discord.Interaction):
        v = _num(self.val.component.value, self.d.lo, self.d.hi)
        if self.d.kind in ("count", "toggle"):
            v = float(round(v))
        ov = dict(self.settings.param_overrides)
        ov[self.key] = v
        await self.submit(interaction, {"param_overrides": ov}, f"{self.d.label} = {v:g}")


# ---------------------------------------------------------------------- versions

async def open_versions(interaction: discord.Interaction, project) -> None:
    svc = _svc(interaction)
    versions = await asyncio.to_thread(svc.projects.list_versions, project.id, 25)
    if not versions:
        await svc.reply(interaction, content="No versions yet.", ephemeral=True)
        return
    lines = [f"`v{v.number}` **{v.status}** - {v.note[:70] or 'build'} - <t:{int(v.created_at)}:R>" for v in versions[:12]]
    await svc.reply(interaction, embed=embeds.info_embed(f"Versions of {project.name}", "\n".join(lines)), view=VersionPicker(project.id, versions), ephemeral=True)


class VersionPicker(discord.ui.View):
    def __init__(self, pid: str, versions):
        super().__init__(timeout=600)
        self.pid = pid
        sel = discord.ui.Select(placeholder="Pick a version", options=[discord.SelectOption(label=f"v{v.number} ({v.status})", value=str(v.number), description=(v.note or "build")[:100]) for v in versions[:25]])
        sel.callback = self._pick
        self.sel = sel
        self.add_item(sel)

    async def _pick(self, interaction: discord.Interaction):
        ver = int(self.sel.values[0])
        svc = _svc(interaction)
        v = await asyncio.to_thread(svc.projects.get_version, self.pid, ver)
        if v is None:
            return
        emb = embeds.info_embed(f"Version {ver}", embeds.settings_summary(v.settings) + (f"\n\n{v.stats.get('triangles', 0):,} triangles" if v.stats else ""))
        await svc.reply(interaction, embed=emb, view=VersionActions(svc, self.pid, ver, v.status == "ready"), ephemeral=True)


class VersionActions(discord.ui.View):
    def __init__(self, svc: "StudioService", pid: str, ver: int, ready: bool):
        super().__init__(timeout=600)
        self.pid, self.ver = pid, ver
        restore = discord.ui.Button(label=f"Restore v{ver} as new version", style=discord.ButtonStyle.primary, emoji=svc.icons.get("refresh"))
        restore.callback = self._restore
        self.add_item(restore)
        if ready:
            self.add_item(ProjectAction("zip", pid, ver, svc.icons.get("download")))

    async def _restore(self, interaction: discord.Interaction):
        svc = _svc(interaction)
        project = await svc.load_project(interaction, self.pid, edit=True)
        if project is None:
            return
        v = await asyncio.to_thread(svc.projects.get_version, self.pid, self.ver)
        await svc.queue_build(interaction, project, self.ver, v.settings, f"restored from v{self.ver}")


# ---------------------------------------------------------------------- history, settings, delete

class HistoryView(discord.ui.View):
    def __init__(self, svc: "StudioService", user_id: int, projects, page: int, pages: int):
        super().__init__(timeout=600)
        self.user_id, self.page, self.pages = user_id, page, pages
        if projects:
            sel = discord.ui.Select(placeholder="Open a project", options=[discord.SelectOption(label=p.name[:100], value=p.id, description=f"{p.id}  |  {p.category or p.status}  |  v{p.current_version}") for p in projects[:25]])
            sel.callback = self._open
            self.sel = sel
            self.add_item(sel)
        prev = discord.ui.Button(label="Previous", disabled=page <= 0, style=discord.ButtonStyle.secondary)
        nxt = discord.ui.Button(label="Next", disabled=page >= pages - 1, style=discord.ButtonStyle.secondary)
        prev.callback = lambda i: self._go(i, -1)
        nxt.callback = lambda i: self._go(i, 1)
        self.add_item(prev)
        self.add_item(nxt)

    async def _go(self, interaction: discord.Interaction, delta: int):
        from .commands import history_page

        emb, view = await history_page(_svc(interaction), self.user_id, self.page + delta)
        await interaction.response.edit_message(embed=emb, view=view)

    async def _open(self, interaction: discord.Interaction):
        svc = _svc(interaction)
        project = await svc.load_project(interaction, self.sel.values[0])
        if project:
            await show_project(interaction, project)


async def show_project(interaction: discord.Interaction, project) -> None:
    svc = _svc(interaction)
    versions = await asyncio.to_thread(svc.projects.list_versions, project.id, 8)
    latest = next((v for v in versions if v.status == "ready"), None)
    emb = embeds.project_embed(project, versions, latest.stats if latest else None, svc.read_spec(project.id))
    preview = Path(latest.package_dir) / "previews" / "preview.png" if latest and latest.package_dir else None
    kwargs = {}
    if preview and preview.exists():
        emb.set_image(url="attachment://preview.png")
        kwargs["file"] = discord.File(str(preview), filename="preview.png")
    view = ResultView(svc, project.id, latest.number) if latest else None
    await svc.reply(interaction, embed=emb, view=view, ephemeral=True, **kwargs)


class DefaultsModal(discord.ui.Modal, title="Default settings"):
    def __init__(self, d: UserDefaults):
        super().__init__(timeout=900)
        self.tris = discord.ui.Label(text="Triangle budget", description="60 - 60,000", component=discord.ui.TextInput(max_length=7, default=str(d.triangle_budget)))
        self.tex = discord.ui.Label(text="Texture resolution", component=discord.ui.Select(options=[discord.SelectOption(label=str(r), value=str(r), default=r == d.texture_resolution) for r in TEXTURE_SIZES]))
        self.quality = discord.ui.Label(text="Quality", component=_select(list(QUALITY_LEVELS), d.quality, "Quality"))
        self.privacy = discord.ui.Label(text="Results", description="private = only you see progress and files", component=_select(["public", "private"], "private" if d.private_results else "public", "Visibility"))
        self.coll = discord.ui.Label(text="Collision", component=_select(list(COLLISION_MODES), d.collision, "Collision"))
        for item in (self.tris, self.tex, self.quality, self.privacy, self.coll):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        svc = _svc(interaction)
        d = UserDefaults(
            triangle_budget=_num(self.tris.component.value, 60, 60000, int) or 5000,
            texture_resolution=int((self.tex.component.values or ["1024"])[0]),
            quality=(self.quality.component.values or ["standard"])[0],
            private_results=(self.privacy.component.values or ["public"])[0] == "private",
            collision=(self.coll.component.values or ["hull"])[0],
        )
        await asyncio.to_thread(svc.projects.set_defaults, interaction.user.id, d)
        await svc.reply(interaction, embed=embeds.info_embed("Defaults saved", f"Triangles {d.triangle_budget:,}, texture {d.texture_resolution}px, quality {d.quality}, results {'private' if d.private_results else 'public'}, collision {d.collision}."), ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await _svc(interaction).reply(interaction, embed=embeds.error_embed("Not saved", str(error)), ephemeral=True)


class SettingsView(discord.ui.View):
    def __init__(self, svc: "StudioService", d: UserDefaults):
        super().__init__(timeout=600)
        self.d = d
        btn = discord.ui.Button(label="Change defaults", style=discord.ButtonStyle.primary, emoji=svc.icons.get("gear"))
        btn.callback = self._open
        self.add_item(btn)

    async def _open(self, interaction: discord.Interaction):
        await interaction.response.send_modal(DefaultsModal(self.d))


class ConfirmDelete(discord.ui.View):
    def __init__(self, pid: str):
        super().__init__(timeout=120)
        self.pid = pid

    @discord.ui.button(label="Delete permanently", style=discord.ButtonStyle.danger)
    async def yes(self, interaction: discord.Interaction, button: discord.ui.Button):
        svc = _svc(interaction)
        project = await svc.load_project(interaction, self.pid, edit=True)
        if project is None:
            return
        await asyncio.to_thread(svc.projects.soft_delete, self.pid)
        await interaction.response.edit_message(embed=embeds.info_embed("Deleted", f"Project `{self.pid}` and its files were removed."), view=None)

    @discord.ui.button(label="Keep", style=discord.ButtonStyle.secondary)
    async def no(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(embed=embeds.info_embed("Kept", "Nothing was deleted."), view=None)


class CancelPicker(discord.ui.View):
    def __init__(self, jobs):
        super().__init__(timeout=300)
        sel = discord.ui.Select(placeholder="Cancel a job", options=[discord.SelectOption(label=f"{j.kind} v{j.version} ({j.status})", value=j.id, description=j.stage[:100] or None) for j in jobs[:25]])
        sel.callback = self._pick
        self.sel = sel
        self.add_item(sel)

    async def _pick(self, interaction: discord.Interaction):
        await _svc(interaction).cancel_job(interaction, self.sel.values[0])
