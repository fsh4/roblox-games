"""The /3d slash command group."""
from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, List, Optional

import discord
from discord import app_commands

from engine.settings import ART_STYLES

from . import embeds
from .views import (CATEGORY_CHOICES, CancelPicker, ConfirmDelete, ExportPicker, HistoryView, SettingsView, UploadPrompt,
                    open_edit_panel, open_versions, show_project)

if TYPE_CHECKING:
    from .interactions import StudioService

PAGE = 10

STYLE_LABELS = {"realistic": "Realistic (photo texture)", "cartoon": "Cartoon (flat bright colours)", "blocky": "Blocky (stacked blocks, symmetric)"}

CATEGORY_LABELS = {
    "auto": "Auto (model the shape as it is)", "bed": "Bed", "chair": "Chair / stool", "table": "Table / desk", "sofa": "Sofa", "cabinet": "Cabinet / dresser",
    "shelf": "Shelf / bookcase", "lamp": "Lamp", "house": "House / building", "car": "Car / vehicle", "box": "Box / crate (photo projected)",
    "revolve": "Round object (vase, bottle, barrel)", "extrude": "Flat object / drawing", "inflate": "Free-form object (toy, food, character, prop)",
}


async def history_page(svc: "StudioService", user_id: int, page: int):
    total = await asyncio.to_thread(svc.projects.count_for_user, user_id)
    pages = max(1, math.ceil(total / PAGE))
    page = max(0, min(page, pages - 1))
    projects = await asyncio.to_thread(svc.projects.list_for_user, user_id, PAGE, page * PAGE)
    if not projects:
        return embeds.info_embed("Your projects", "Nothing yet. Start with `/3d create`."), None
    lines = [f"`{p.id}` **{p.name}** - {p.category or p.status}, v{p.current_version} - <t:{int(p.updated_at)}:R>" for p in projects]
    emb = embeds.info_embed(f"Your projects ({total})", "\n".join(lines))
    emb.set_footer(text=f"page {page + 1} of {pages}")
    return emb, HistoryView(svc, user_id, projects, page, pages)


class StudioCommands(app_commands.Group):
    def __init__(self, svc: "StudioService"):
        super().__init__(name="3d", description="Image-to-3D Studio: photos to Roblox-ready 3D models")
        self.svc = svc

    async def project_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        projects = await asyncio.to_thread(self.svc.projects.list_for_user, interaction.user.id, 50, 0)
        cur = current.lower().strip()
        out = []
        for p in projects:
            if not cur or cur in p.name.lower() or p.id.startswith(cur):
                out.append(app_commands.Choice(name=f"{p.name[:70]} ({p.id}, v{p.current_version})", value=p.id))
            if len(out) >= 25:
                break
        return out

    @app_commands.command(name="create", description="Turn photos into a 3D model")
    @app_commands.describe(
        image="Photo of the object (leave empty to open the upload form)", image2="Extra view", image3="Extra view", image4="Extra view",
        category="Leave on auto: the shape is modelled as it is. Pick a furniture template only if you want one",
        mode="auto, single (one-photo generic reconstruction) or multi (several photos)",
        views="For several photos: their directions, e.g. front,side,back,top",
        triangles="Triangle budget (60 - 60,000)", texture="Texture atlas size", quality="Geometry and baking quality", name="Model name",
        style="Look of the model: photo texture, cartoon colours or stacked blocks")
    @app_commands.choices(
        category=[app_commands.Choice(name=CATEGORY_LABELS[c], value=c) for c in CATEGORY_CHOICES],
        mode=[app_commands.Choice(name=n, value=n) for n in ("auto", "single", "multi")],
        texture=[app_commands.Choice(name=f"{r} x {r}", value=r) for r in (256, 512, 1024, 2048)],
        quality=[app_commands.Choice(name=q, value=q) for q in ("draft", "standard", "high")],
        style=[app_commands.Choice(name=STYLE_LABELS[s], value=s) for s in ART_STYLES])
    async def create(self, interaction: discord.Interaction, image: Optional[discord.Attachment] = None, image2: Optional[discord.Attachment] = None,
                     image3: Optional[discord.Attachment] = None, image4: Optional[discord.Attachment] = None, category: str = "auto", mode: str = "auto",
                     views: Optional[str] = None, triangles: Optional[app_commands.Range[int, 60, 60000]] = None, texture: Optional[int] = None,
                     quality: Optional[str] = None, name: Optional[app_commands.Range[str, 1, 60]] = None, style: Optional[str] = None):
        files = [a for a in (image, image2, image3, image4) if a is not None]
        if not files:
            emb = embeds.info_embed("New 3D model", "Upload one photo, or several photos of the same object from different sides.\n"
                                    "Tip: plain background, whole object visible, light from the front.")
            await interaction.response.send_message(embed=emb, view=UploadPrompt(self.svc), ephemeral=True)
            return
        await self.svc.start_create(interaction, files, mode=mode, category=category, views=views or "", triangles=triangles, texture=texture, quality=quality, name=name, style=style)

    @app_commands.command(name="edit", description="Change a model without re-uploading the photo")
    @app_commands.describe(project="Project to edit (latest if empty)")
    async def edit(self, interaction: discord.Interaction, project: Optional[str] = None):
        proj = await self._resolve(interaction, project, edit=True)
        if proj:
            await open_edit_panel(interaction, proj)

    @app_commands.command(name="export", description="Download a model in another format")
    @app_commands.describe(project="Project", version="Version number (latest if empty)")
    @app_commands.choices(format=[app_commands.Choice(name=f.upper(), value=f) for f, _ in ExportPicker.FORMATS])
    async def export(self, interaction: discord.Interaction, project: Optional[str] = None, format: Optional[str] = None, version: Optional[int] = None):
        proj = await self._resolve(interaction, project)
        if not proj:
            return
        ver = version or proj.current_version
        if not format:
            await self.svc.reply(interaction, content=f"Export **{proj.name} v{ver}** as:", view=ExportPicker(proj.id, ver), ephemeral=True)
            return
        await self.svc.export(interaction, proj, ver, format)

    @app_commands.command(name="history", description="Your generated models")
    async def history(self, interaction: discord.Interaction):
        emb, view = await history_page(self.svc, interaction.user.id, 0)
        await self.svc.reply(interaction, embed=emb, view=view, ephemeral=True) if view else await self.svc.reply(interaction, embed=emb, ephemeral=True)

    @app_commands.command(name="versions", description="Versions of a project; restore any of them")
    @app_commands.describe(project="Project (latest if empty)")
    async def versions(self, interaction: discord.Interaction, project: Optional[str] = None):
        proj = await self._resolve(interaction, project)
        if proj:
            await open_versions(interaction, proj)

    @app_commands.command(name="open", description="Show a project card with its preview and files")
    @app_commands.describe(project="Project (latest if empty)")
    async def open(self, interaction: discord.Interaction, project: Optional[str] = None):
        proj = await self._resolve(interaction, project)
        if proj:
            await show_project(interaction, proj)

    @app_commands.command(name="status", description="Running and queued jobs")
    async def status(self, interaction: discord.Interaction):
        jobs = await asyncio.to_thread(self.svc.jobs.active_for_user, interaction.user.id)
        lines = []
        for j in jobs:
            if j.status == "queued":
                lines.append(f"`{j.id}` {j.kind} v{j.version} - queued, position {self.svc.queue.position(j.id) or '?'}")
            else:
                lines.append(f"`{j.id}` {j.kind} v{j.version} - {j.stage}\n{embeds.bar(j.progress)}")
        desc = "\n".join(lines) or "No jobs running."
        desc += f"\n\nServer queue: {self.svc.queue.running} running, {self.svc.queue.waiting} waiting, {self.svc.queue.concurrency} slot(s)."
        recent = await asyncio.to_thread(self.svc.jobs.recent_for_user, interaction.user.id, 5)
        done = [j for j in recent if not j.active]
        if done:
            desc += "\n\nRecent: " + ", ".join(f"v{j.version} {j.status}" for j in done)
        view = CancelPicker(jobs) if jobs else None
        await self.svc.reply(interaction, embed=embeds.info_embed("Jobs", desc), view=view, ephemeral=True) if view else await self.svc.reply(interaction, embed=embeds.info_embed("Jobs", desc), ephemeral=True)

    @app_commands.command(name="settings", description="Your default settings for new models")
    async def settings(self, interaction: discord.Interaction):
        d = await asyncio.to_thread(self.svc.projects.get_defaults, interaction.user.id)
        desc = (f"Triangle budget **{d.triangle_budget:,}**\nTexture **{d.texture_resolution}px**\nQuality **{d.quality}**\n"
                f"Results **{'private' if d.private_results else 'public'}**\nCollision **{d.collision}**")
        await self.svc.reply(interaction, embed=embeds.info_embed("Default settings", desc), view=SettingsView(self.svc, d), ephemeral=True)

    @app_commands.command(name="delete", description="Delete a project and its files")
    @app_commands.describe(project="Project to delete")
    async def delete(self, interaction: discord.Interaction, project: str):
        proj = await self.svc.load_project(interaction, project, edit=True)
        if proj:
            await self.svc.reply(interaction, embed=embeds.error_embed(f"Delete {proj.name}?", "All versions and files of this project will be removed. This cannot be undone."), view=ConfirmDelete(proj.id), ephemeral=True)

    @app_commands.command(name="help", description="How Image-to-3D Studio works")
    async def help(self, interaction: discord.Interaction):
        await self.svc.reply(interaction, embed=embeds.help_embed(), ephemeral=True)

    async def _resolve(self, interaction: discord.Interaction, project: Optional[str], edit: bool = False):
        if project:
            return await self.svc.load_project(interaction, project, edit)
        latest = await asyncio.to_thread(self.svc.projects.list_for_user, interaction.user.id, 1, 0, False)
        if not latest:
            await self.svc.reply(interaction, embed=embeds.info_embed("No projects yet", "Create one with `/3d create`."), ephemeral=True)
            return None
        return latest[0]

    edit.autocomplete("project")(project_autocomplete)
    export.autocomplete("project")(project_autocomplete)
    versions.autocomplete("project")(project_autocomplete)
    open.autocomplete("project")(project_autocomplete)
    delete.autocomplete("project")(project_autocomplete)
