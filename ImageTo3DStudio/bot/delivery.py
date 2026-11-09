"""Delivering packages within Discord's upload limits.

Discord checks the size of every attachment against the limit reported in the
interaction (``attachment_size_limit``; 20 MiB by default since 2026-09, more in
boosted servers). The ZIP is attached when it fits; otherwise the most useful
files are attached individually and, if the optional download server is
enabled, the user gets a signed, expiring link that only works for that file.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import discord

from utils.filesystem import UnsafePathError, human_size, safe_join

log = logging.getLogger("studio.delivery")

CORE_ORDER = ("model.fbx", "model.rbxm", "model.glb", "previews/preview.png", "README.md")


@dataclass
class Plan:
    files: List[Path]
    skipped: List[Path]
    links: List[Tuple[str, str]]


def upload_limit(interaction: Optional[discord.Interaction], default_mb: float) -> int:
    limit = getattr(interaction, "filesize_limit", None) if interaction is not None else None
    if not limit:
        guild = getattr(interaction, "guild", None) if interaction is not None else None
        limit = getattr(guild, "filesize_limit", None)
    limit = int(limit or default_mb * 1024 * 1024)
    return int(limit * 0.97)


def plan_package(package_dir: Path, zip_path: Path, limit: int, linker=None) -> Plan:
    if zip_path.exists() and zip_path.stat().st_size <= limit:
        return Plan([zip_path], [], [])
    files, skipped, links = [], [], []
    for rel in CORE_ORDER:
        p = package_dir / rel
        if not p.exists():
            continue
        if p.stat().st_size <= limit and len(files) < 10:
            files.append(p)
        else:
            skipped.append(p)
    if linker is not None and zip_path.exists():
        url = linker(zip_path)
        if url:
            links.append((zip_path.name, url))
    return Plan(files, skipped, links)


def describe_plan(plan: Plan) -> str:
    if not plan.skipped and not plan.links:
        return ""
    parts = []
    if plan.skipped:
        parts.append("Too large for this channel's upload limit: " + ", ".join(f"`{p.name}` ({human_size(p.stat().st_size)})" for p in plan.skipped) + ".")
    for name, url in plan.links:
        parts.append(f"Private download link for `{name}` (expires): {url}")
    if not plan.links:
        parts.append("Use `/3d export` for single files, or lower the texture resolution in `/3d edit` to shrink the package.")
    return "\n".join(parts)


class LinkSigner:
    """HMAC-signed, expiring download URLs for the optional HTTP server."""

    def __init__(self, secret: str, base_url: str, data_dir: Path, ttl_hours: int):
        self.secret = secret.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.data_dir = Path(data_dir).resolve()
        self.ttl = ttl_hours * 3600

    def sign(self, path: Path) -> Optional[str]:
        try:
            rel = Path(path).resolve().relative_to(self.data_dir).as_posix()
        except ValueError:
            return None
        exp = int(time.time()) + self.ttl
        mac = hmac.new(self.secret, f"{rel}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        from urllib.parse import quote

        return f"{self.base_url}/d/{exp}/{mac}/{quote(rel)}"

    def verify(self, rel: str, exp: str, mac: str) -> Optional[Path]:
        if not exp.isdigit() or int(exp) < time.time():
            return None
        good = hmac.new(self.secret, f"{rel}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(good, mac):
            return None
        try:
            p = safe_join(self.data_dir, rel)
        except UnsafePathError:
            return None
        return p if p.is_file() else None


async def start_download_server(signer: LinkSigner, host: str, port: int):
    from aiohttp import web

    async def handle(request: web.Request) -> web.StreamResponse:
        p = signer.verify(request.match_info["rel"], request.match_info["exp"], request.match_info["mac"])
        if p is None:
            return web.Response(status=404, text="link expired or invalid")
        return web.FileResponse(p, headers={"Content-Disposition": f'attachment; filename="{p.name}"', "Cache-Control": "private, no-store"})

    app = web.Application()
    app.router.add_get("/d/{exp}/{mac}/{rel:.+}", handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("download server listening on %s:%d", host, port)
    return runner


def to_files(paths: List[Path]) -> List[discord.File]:
    out = []
    for p in paths:
        name = p.name
        if p.parent.name == "previews" and name == "preview.png":
            name = "preview.png"
        out.append(discord.File(str(p), filename=name))
    return out
