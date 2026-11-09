"""StudioService: the glue between Discord interactions, storage and the job queue.

Commands and UI components stay thin; every flow (create, edit, restore,
export, delivery, job progress) lives here so it is written once.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import discord

from config import Config
from engine.settings import BuildSettings
from exporters.convert import export_format
from storage.jobs import Job, JobStore
from storage.projects import Project, ProjectStore, UserDefaults
from utils.images import UploadRejected, check_upload
from workers.queue import JobEvent, JobQueue

from . import embeds
from .delivery import LinkSigner, describe_plan, plan_package, to_files, upload_limit
from .icons import IconSet

log = logging.getLogger("studio.service")
TOKEN_LIFETIME = 14 * 60


@dataclass
class Tracker:
    interaction: Optional[discord.Interaction]
    message: Optional[discord.abc.Messageable]
    user_id: int
    private: bool
    created: float = field(default_factory=time.time)
    last_edit: float = 0.0


class StudioService:
    def __init__(self, bot: discord.Client, cfg: Config, projects: ProjectStore, jobs: JobStore, icons: IconSet):
        self.bot = bot
        self.cfg = cfg
        self.projects = projects
        self.jobs = jobs
        self.icons = icons
        self.queue = JobQueue(jobs, projects, self.on_job_event, cfg.max_concurrent_jobs, cfg.job_timeout_seconds, cfg.worker_memory_mb)
        self.trackers: Dict[str, Tracker] = {}
        self.signer: Optional[LinkSigner] = None
        if cfg.download_server and cfg.download_secret and cfg.download_base_url:
            self.signer = LinkSigner(cfg.download_secret, cfg.download_base_url, cfg.data_dir, cfg.download_ttl_hours)

    # ------------------------------------------------------------------ access

    def is_admin(self, user_id: int) -> bool:
        return user_id in self.cfg.admin_user_ids

    async def load_project(self, interaction: discord.Interaction, project_id: str, edit: bool = False) -> Optional[Project]:
        pid = (project_id or "").strip().lower().lstrip("#")
        m = re.search(r"[0-9a-f]{8}", pid)
        proj = await asyncio.to_thread(self.projects.get, m.group(0)) if m else None
        allowed = proj is not None and (proj.can_edit(interaction.user.id, self.cfg.admin_user_ids) if edit else proj.can_view(interaction.user.id, self.cfg.admin_user_ids))
        if not allowed:
            await self.reply(interaction, embed=embeds.error_embed("Project not found", "No project with that ID belongs to you. Use `/3d history` to see yours."), ephemeral=True)
            return None
        return proj

    async def reply(self, interaction: discord.Interaction, ephemeral: bool = True, **kwargs):
        try:
            if interaction.response.is_done():
                return await interaction.followup.send(ephemeral=ephemeral, wait=True, **kwargs)
            await interaction.response.send_message(ephemeral=ephemeral, **kwargs)
            return await interaction.original_response()
        except discord.HTTPException as exc:
            log.warning("reply failed: %s", exc)
            return None

    # ------------------------------------------------------------------ create

    async def start_create(self, interaction: discord.Interaction, attachments: Sequence[discord.Attachment], *, mode: str = "auto",
                           category: str = "auto", views: str = "", triangles: Optional[int] = None, texture: Optional[int] = None,
                           quality: Optional[str] = None, name: Optional[str] = None) -> None:
        defaults: UserDefaults = await asyncio.to_thread(self.projects.get_defaults, interaction.user.id)
        private = bool(defaults.private_results)
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True, ephemeral=private)
        attachments = [a for a in attachments if a is not None]
        if not attachments:
            await self.reply(interaction, embed=embeds.error_embed("No image", "Attach at least one photo."), ephemeral=True)
            return
        if len(attachments) > self.cfg.max_images_per_job:
            await self.reply(interaction, embed=embeds.error_embed("Too many images", f"Use up to {self.cfg.max_images_per_job} images per model."), ephemeral=True)
            return
        active = await asyncio.to_thread(self.jobs.active_for_user, interaction.user.id)
        if len(active) >= self.cfg.max_jobs_per_user:
            await self.reply(interaction, embed=embeds.error_embed("Busy", f"You already have {len(active)} job(s) running or queued. Check `/3d status`."), ephemeral=True)
            return
        max_bytes = int(self.cfg.max_upload_mb * 1024 * 1024)
        blobs = []
        try:
            for att in attachments:
                if att.size > max_bytes:
                    raise UploadRejected(f"`{att.filename}` is {att.size / 1e6:.1f} MB; the limit is {self.cfg.max_upload_mb:.0f} MB")
                data = await att.read()
                await asyncio.to_thread(check_upload, att.filename, data, max_bytes, self.cfg.max_image_pixels)
                blobs.append((att.filename, data))
        except UploadRejected as exc:
            await self.reply(interaction, embed=embeds.error_embed("Image rejected", str(exc)), ephemeral=True)
            return
        except discord.HTTPException:
            await self.reply(interaction, embed=embeds.error_embed("Download failed", "Discord did not deliver the attachment. Please try again."), ephemeral=True)
            return
        title = (name or Path(attachments[0].filename).stem.replace("_", " ").replace("-", " ").title() or "Model")[:60]
        project = await asyncio.to_thread(self.projects.create_project, interaction.user.id, interaction.guild_id, interaction.channel_id, title)
        source = self.projects.project_dir(project.id) / "source"
        names = []
        for i, (fname, data) in enumerate(blobs):
            ext = Path(fname).suffix.lower() if Path(fname).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif") else ".png"
            safe = f"image_{i + 1}{ext}"
            (source / safe).write_bytes(data)
            names.append(safe)
        labels = [v.strip().lower() for v in views.replace(";", ",").split(",") if v.strip()] if views else None
        settings = BuildSettings(
            triangle_budget=int(triangles or defaults.triangle_budget), texture_resolution=int(texture or defaults.texture_resolution),
            quality=quality or defaults.quality, shading=defaults.shading, collision=defaults.collision).validated()
        payload_extra = {"images": names, "mode": mode, "category": category, "view_labels": labels, "sensitivity": 0.5}
        await self._enqueue(interaction, project, settings, None, "initial generation", "create", payload_extra, private)

    # ------------------------------------------------------------------ edits

    async def queue_build(self, interaction: discord.Interaction, project: Project, base_version: int, settings: BuildSettings, note: str,
                          kind: str = "edit", reanalyse: Optional[dict] = None) -> None:
        defaults = await asyncio.to_thread(self.projects.get_defaults, interaction.user.id)
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True, ephemeral=bool(defaults.private_results))
        active = await asyncio.to_thread(self.jobs.active_for_user, interaction.user.id)
        if len(active) >= self.cfg.max_jobs_per_user:
            await self.reply(interaction, embed=embeds.error_embed("Busy", "Wait for your running jobs to finish (see `/3d status`)."), ephemeral=True)
            return
        extra = {}
        if kind == "reanalyze":
            spec = self.read_spec(project.id)
            extra = {"images": (spec or {}).get("images", []), "mode": reanalyse.get("mode", "auto"), "category": reanalyse.get("category", "auto"),
                     "view_labels": (spec or {}).get("view_labels"), "sensitivity": float(reanalyse.get("sensitivity", 0.5))}
        await self._enqueue(interaction, project, settings.validated(), base_version, note, kind, extra, bool(defaults.private_results))

    async def _enqueue(self, interaction, project: Project, settings: BuildSettings, parent: Optional[int], note: str, kind: str, extra: dict, private: bool) -> None:
        version = await asyncio.to_thread(self.projects.new_version, project.id, settings, parent, note)
        payload = {
            "kind": kind, "project_id": project.id, "project_dir": str(self.projects.project_dir(project.id)),
            "version_dir": str(self.projects.version_dir(project.id, version.number)), "version": version.number,
            "title": project.name, "settings": settings.to_dict(), **extra,
        }
        job = await asyncio.to_thread(self.jobs.create, project.id, version.number, interaction.user.id, kind, payload, interaction.channel_id)
        position = await self.queue.submit(job)
        status = "queued" if position > 1 or self.queue.running >= self.queue.concurrency else "running"
        emb = embeds.progress_embed(project.name, project.id, version.number, "queued", 0.0, position, status, note if kind != "create" else "")
        from .views import JobView

        msg = await self.reply(interaction, embed=emb, view=JobView(self, job.id), ephemeral=private)
        if msg is not None:
            await asyncio.to_thread(self.jobs.set_message, job.id, getattr(msg.channel, "id", None), msg.id)
        self.trackers[job.id] = Tracker(interaction, msg, interaction.user.id, private)

    # ------------------------------------------------------------------ job events

    async def on_job_event(self, ev: JobEvent) -> None:
        job = ev.job
        tr = self.trackers.get(job.id)
        project = await asyncio.to_thread(self.projects.get, job.project_id)
        if project is None:
            return
        if ev.kind in ("progress", "started"):
            if tr is None or time.time() - tr.last_edit < 2.0:
                return
            tr.last_edit = time.time()
            emb = embeds.progress_embed(project.name, project.id, job.version, ev.data.get("stage", job.stage), float(ev.data.get("progress", job.progress)))
            await self._edit(job, tr, embed=emb)
            return
        if ev.kind == "done":
            await self._edit(job, tr, embed=embeds.info_embed(f"{project.name}  |  v{job.version}", "Finished - results below."), view=None)
            await self.deliver_result(job, project, ev.data, tr)
        elif ev.kind == "failed":
            err = ev.data.get("error", job.error or "unknown error")
            from .views import RetryView

            await self._edit(job, tr, embed=embeds.error_embed(f"{project.name}  |  v{job.version} failed", f"{err}\n\nNothing from earlier versions was changed."), view=RetryView(self, job.id))
        elif ev.kind == "cancelled":
            await self._edit(job, tr, embed=embeds.info_embed(f"{project.name}  |  v{job.version}", "Cancelled."), view=None)
        if ev.kind in ("done", "failed", "cancelled"):
            self.trackers.pop(job.id, None)

    async def _edit(self, job: Job, tr: Optional[Tracker], **kwargs) -> None:
        try:
            if tr is not None and tr.interaction is not None and time.time() - tr.created < TOKEN_LIFETIME and tr.message is not None:
                await tr.interaction.followup.edit_message(tr.message.id, **kwargs)
                return
            if job.channel_id and job.message_id:
                channel = self.bot.get_channel(job.channel_id)
                if channel is not None:
                    await channel.get_partial_message(job.message_id).edit(**kwargs)
        except discord.HTTPException as exc:
            log.debug("progress edit failed for %s: %s", job.id, exc)

    async def _send_followup(self, job: Job, tr: Optional[Tracker], **kwargs) -> Optional[discord.Message]:
        private = tr.private if tr else False
        try:
            if tr is not None and tr.interaction is not None and time.time() - tr.created < TOKEN_LIFETIME:
                return await tr.interaction.followup.send(wait=True, ephemeral=private, **kwargs)
        except discord.HTTPException as exc:
            log.info("followup failed for %s: %s", job.id, exc)
        user = self.bot.get_user(job.user_id) or await self.bot.fetch_user(job.user_id)
        channel = self.bot.get_channel(job.channel_id) if job.channel_id and not private else None
        try:
            if channel is not None:
                return await channel.send(content=f"<@{job.user_id}>", allowed_mentions=discord.AllowedMentions(users=True), **kwargs)
            if user is not None:
                return await user.send(**kwargs)
        except discord.HTTPException as exc:
            log.warning("could not deliver result for %s: %s", job.id, exc)
        return None

    async def deliver_result(self, job: Job, project: Project, result: dict, tr: Optional[Tracker]) -> None:
        from .views import ResultView

        package = Path(result.get("package_dir", ""))
        zip_path = Path(result.get("zip", ""))
        limit = upload_limit(tr.interaction if tr else None, self.cfg.default_upload_limit_mb)
        plan = plan_package(package, zip_path, limit, self.signer.sign if self.signer else None)
        preview = package / "previews" / "preview.png"
        files = []
        if preview.exists() and preview.stat().st_size <= limit:
            files.append(discord.File(str(preview), filename="preview.png"))
        files += [f for f in to_files(plan.files) if f.filename != "preview.png"]
        emb = embeds.result_embed(project.name, project.id, job.version, result, "preview.png" if preview.exists() else None)
        note = describe_plan(plan)
        await self._send_followup(job, tr, content=note or None, embed=emb, files=files[:10], view=ResultView(self, project.id, job.version))

    # ------------------------------------------------------------------ files and exports

    def read_spec(self, project_id: str) -> Optional[dict]:
        p = self.projects.project_dir(project_id) / "source" / "spec.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    async def send_file(self, interaction: discord.Interaction, path: Path, note: str = "") -> None:
        limit = upload_limit(interaction, self.cfg.default_upload_limit_mb)
        if not path.exists():
            await self.reply(interaction, embed=embeds.error_embed("Missing file", "That file is not part of this version."), ephemeral=True)
            return
        if path.stat().st_size > limit:
            link = self.signer.sign(path) if self.signer else None
            msg = f"`{path.name}` is larger than this channel allows." + (f" Private link (expires): {link}" if link else " Lower the texture resolution or download individual files.")
            await self.reply(interaction, content=msg, ephemeral=True)
            return
        await self.reply(interaction, content=note or None, file=discord.File(str(path), filename=path.name), ephemeral=True)

    async def export(self, interaction: discord.Interaction, project: Project, version: int, fmt: str) -> None:
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True, ephemeral=True)
        v = await asyncio.to_thread(self.projects.get_version, project.id, version)
        if v is None or v.status != "ready" or not v.package_dir:
            await self.reply(interaction, embed=embeds.error_embed("Not available", f"Version {version} has no finished package (status: {v.status if v else 'missing'})."), ephemeral=True)
            return
        out_dir = self.projects.version_dir(project.id, version) / "exports"
        try:
            path = await asyncio.wait_for(asyncio.to_thread(export_format, v.package_dir, fmt, str(out_dir)), timeout=120)
        except Exception as exc:
            log.warning("export %s failed for %s v%s: %s", fmt, project.id, version, exc)
            await self.reply(interaction, embed=embeds.error_embed("Export failed", str(exc)[:500]), ephemeral=True)
            return
        await self.send_file(interaction, Path(path), f"{project.name} v{version} as {fmt.upper()}")

    async def cancel_job(self, interaction: discord.Interaction, job_id: str) -> None:
        job = await asyncio.to_thread(self.jobs.get, job_id)
        if job is None or (job.user_id != interaction.user.id and not self.is_admin(interaction.user.id)):
            await self.reply(interaction, content="That job is not yours.", ephemeral=True)
            return
        ok = await self.queue.cancel(job_id)
        await self.reply(interaction, content="Cancelling..." if ok else "That job already finished.", ephemeral=True)

    async def retry_job(self, interaction: discord.Interaction, job_id: str) -> None:
        job = await asyncio.to_thread(self.jobs.get, job_id)
        if job is None or job.user_id != interaction.user.id:
            await self.reply(interaction, content="That job is not yours.", ephemeral=True)
            return
        project = await asyncio.to_thread(self.projects.get, job.project_id)
        if project is None:
            await self.reply(interaction, content="The project no longer exists.", ephemeral=True)
            return
        settings = BuildSettings.from_dict(job.payload.get("settings"))
        analysed = (self.projects.project_dir(project.id) / "source" / "spec.json").exists()
        kind = "edit" if analysed and job.kind == "edit" else ("create" if not analysed else job.kind)
        extra = {} if kind == "edit" else {k: job.payload[k] for k in ("images", "mode", "category", "view_labels", "sensitivity") if k in job.payload}
        defaults = await asyncio.to_thread(self.projects.get_defaults, interaction.user.id)
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True, ephemeral=bool(defaults.private_results))
        await self._enqueue(interaction, project, settings, job.version, f"retry of v{job.version}", kind, extra, bool(defaults.private_results))
