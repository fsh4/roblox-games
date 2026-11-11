"""Embed builders with the studio's visual language (compact, mobile friendly)."""
from __future__ import annotations

from typing import List, Optional

import discord

from engine.procedural import TEMPLATES
from engine.settings import BuildSettings

AMBER = 0xF0A840
TEAL = 0x2EC4A6
RED = 0xE5534B
SLATE = 0x5B6B7F

STAGES = ("analysing", "fitting", "measuring", "building", "uv", "materials", "occlusion", "normals", "maps", "export", "previews", "package")


def bar(fraction: float, width: int = 18) -> str:
    fraction = max(0.0, min(1.0, fraction))
    full = int(round(fraction * width))
    return "`" + "█" * full + "░" * (width - full) + f"` {fraction * 100:4.0f}%"


def _ago(ts: float) -> str:
    return f"<t:{int(ts)}:R>"


def progress_embed(project_name: str, project_id: str, version: int, stage: str, progress: float, position: int = 0, status: str = "running", note: str = "") -> discord.Embed:
    title = f"{project_name}  |  v{version}"
    if status == "queued":
        desc = f"Waiting in the queue (position **{position}**)." if position else "Waiting in the queue."
    else:
        desc = f"**{stage.capitalize() or 'Working'}**\n{bar(progress)}"
    e = discord.Embed(title=title, description=desc, color=AMBER)
    if note:
        e.add_field(name="Note", value=note[:1000], inline=False)
    e.set_footer(text=f"Image-to-3D Studio  |  project {project_id}")
    return e


def result_embed(project_name: str, project_id: str, version: int, result: dict, attachment_name: Optional[str]) -> discord.Embed:
    s = result.get("stats", {})
    b = s.get("budget_report", {})
    size = s.get("size_studs", [0, 0, 0])
    conf = float(s.get("confidence", result.get("confidence") or 0.5))
    e = discord.Embed(title=f"{project_name}  |  v{version} ready", color=TEAL,
                      description=f"{s.get('strategy_label', '')}\nConfidence **{conf:.0%}** - unseen sides are estimated, not measured.")
    e.add_field(name="Triangles", value=f"**{s.get('triangles', 0):,}** / {s.get('budget', 0):,} budget\n{b.get('reduction_percent', 0):.0f}% below full detail", inline=True)
    impact = b.get("quality_impact", {})
    e.add_field(name="Shape impact", value=f"{impact.get('rating', 'n/a')}\np95 {impact.get('p95_deviation_percent', 0):.2f}%", inline=True)
    e.add_field(name="Size", value=f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} studs", inline=True)
    maps = ", ".join(s.get("texture_maps", []))
    e.add_field(name="Textures", value=f"{s.get('texture_resolution', 0)} px atlas\n{maps}", inline=True)
    e.add_field(name="Roblox", value=f"{s.get('mesh_count', 1)} MeshPart(s), collision {s.get('roblox', {}).get('collision_fidelity', 'Hull')}\nPart model: {s.get('native_parts', 0)} parts", inline=True)
    secs = s.get("timings_seconds", {}).get("total") or result.get("seconds")
    e.add_field(name="Build time", value=f"{secs:.0f} s" if secs else "-", inline=True)
    warnings = result.get("warnings") or []
    notes = [n for n in (result.get("notes") or []) if n]
    extra = (warnings + notes)[:3]
    if extra:
        e.add_field(name="Notes", value="\n".join(f"- {x[:180]}" for x in extra), inline=False)
    if attachment_name:
        e.set_image(url=f"attachment://{attachment_name}")
    e.set_footer(text=f"project {project_id}  |  /3d edit to change budget, size, materials or textures")
    return e


def error_embed(title: str, message: str) -> discord.Embed:
    return discord.Embed(title=title, description=message[:3900], color=RED)


def info_embed(title: str, description: str = "") -> discord.Embed:
    return discord.Embed(title=title, description=description[:3900], color=SLATE)


def project_embed(project, versions: List, latest_stats: Optional[dict], spec: Optional[dict]) -> discord.Embed:
    e = discord.Embed(title=f"{project.name}", color=SLATE, description=f"Project `{project.id}`  |  created {_ago(project.created_at)}")
    if spec:
        if spec.get("strategy") == "template":
            t = TEMPLATES.get(spec.get("template", ""))
            e.add_field(name="Reconstruction", value=f"Template **{t.label if t else spec.get('template')}**, fit score {spec.get('fit', {}).get('score', 0):.2f}", inline=False)
        else:
            e.add_field(name="Reconstruction", value=spec.get("strategy", "-"), inline=False)
    if latest_stats:
        size = latest_stats.get("size_studs", [0, 0, 0])
        e.add_field(name="Current version", value=f"v{project.current_version}: {latest_stats.get('triangles', 0):,} tris, {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} studs", inline=False)
    if versions:
        lines = [f"`v{v.number}` {v.status} - {v.note[:60] or 'build'}" for v in versions[:6]]
        e.add_field(name="Versions", value="\n".join(lines), inline=False)
    return e


def settings_summary(s: BuildSettings) -> str:
    dims = "auto" if not s.dimensions else " x ".join("-" if d is None else f"{d:g}" for d in s.dimensions)
    return (f"Triangles **{s.triangle_budget:,}**  |  texture **{s.texture_resolution}px** ({s.texture_style})\n"
            f"Quality **{s.quality}**  |  shading **{s.shading}**  |  bevel x{s.bevel:g}\n"
            f"Size {dims} studs, scale x{s.scale:g}  |  pivot {s.pivot}  |  collision {s.collision}  |  meshes {s.mesh_grouping}")


def help_embed() -> discord.Embed:
    e = discord.Embed(title="Image-to-3D Studio", color=AMBER, description=(
        "Turns photos into Roblox-ready 3D models with **classical computer vision and procedural modelling** - no AI models or AI services.\n"
        "Results are estimates of what the photo shows; sides that are not visible are inferred."))
    e.add_field(name="/3d create", value="Upload 1-6 photos. One photo: the object is matched against furniture/vehicle/building templates or rebuilt by revolution, extrusion or inflation. Several photos: label them `front,side,back,top` for a visual hull, or leave unlabelled for a photo walk-around.", inline=False)
    e.add_field(name="/3d edit", value="Change triangle budget, size, materials, textures, bevel, shading, collision, pivot or shape parameters. Every change is a new version; earlier versions stay restorable.", inline=False)
    e.add_field(name="/3d export", value="Get FBX, GLB, OBJ, STL, RBXM, RBXMX, textures or the whole ZIP of any version.", inline=False)
    e.add_field(name="/3d history  |  /3d status  |  /3d settings", value="Your projects, running jobs and your default settings.", inline=False)
    e.add_field(name="Best photos", value="Whole object in frame, plain background, even light, about 30 degrees above. A PNG with transparency skips background removal.", inline=False)
    e.add_field(name="Roblox", value="Import `model.fbx` with File > Import 3D (scale unit Studs). `model.rbxm` is an instant native-part version for Insert from file.", inline=False)
    return e
