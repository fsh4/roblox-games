"""Image-to-3D Studio - Discord bot entry point.

    python main.py            start the bot
    python main.py --check    validate configuration, storage and dependencies, then exit
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys

import discord
from discord.ext import commands

from config import Config, load_config
from storage.database import Database
from storage.jobs import JobStore
from storage.projects import ProjectStore
from utils.logging import setup_logging

log = logging.getLogger("studio")


class StudioBot(commands.Bot):
    def __init__(self, cfg: Config):
        intents = discord.Intents.default()
        super().__init__(command_prefix=commands.when_mentioned, intents=intents, help_command=None,
                         activity=discord.Activity(type=discord.ActivityType.watching, name="/3d create"))
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.projects = ProjectStore(self.db, cfg.projects_dir)
        self.jobs = JobStore(self.db)
        from bot.icons import IconSet
        from bot.interactions import StudioService

        self.icons = IconSet()
        self.studio = StudioService(self, cfg, self.projects, self.jobs, self.icons)
        self._download_runner = None

    async def setup_hook(self) -> None:
        from bot.commands import StudioCommands
        from bot.delivery import start_download_server
        from bot.views import CancelJobButton, ProjectAction, RetryButton

        self.add_dynamic_items(ProjectAction, CancelJobButton, RetryButton)
        self.tree.add_command(StudioCommands(self.studio))
        self.tree.on_error = self.on_app_command_error
        if self.cfg.dev_guild_id:
            guild = discord.Object(id=self.cfg.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info("synced %d command group(s) to development guild %s", len(synced), self.cfg.dev_guild_id)
        else:
            synced = await self.tree.sync()
            log.info("synced %d global command group(s) (may take a while to appear)", len(synced))
        recovered = await self.studio.queue.start()
        for job in recovered["interrupted"]:
            log.info("job %s was interrupted by the restart; the user can retry it from /3d status", job.id)
        if self.cfg.download_server and self.studio.signer is not None:
            self._download_runner = await start_download_server(self.studio.signer, self.cfg.download_host, self.cfg.download_port)

    async def on_ready(self) -> None:
        log.info("logged in as %s (%s) in %d server(s)", self.user, self.user.id if self.user else "?", len(self.guilds))
        if self.cfg.register_app_emojis and not self.icons.emojis:
            await self.icons.sync(self)

    async def on_app_command_error(self, interaction: discord.Interaction, error: discord.app_commands.AppCommandError) -> None:
        log.exception("command error", exc_info=error)
        from bot import embeds

        msg = "Something went wrong on our side; the error was logged. Please try again."
        if isinstance(error, discord.app_commands.CommandOnCooldown):
            msg = f"Slow down - try again in {error.retry_after:.0f}s."
        await self.studio.reply(interaction, embed=embeds.error_embed("Error", msg), ephemeral=True)

    async def close(self) -> None:
        await self.studio.queue.stop()
        if self._download_runner is not None:
            await self._download_runner.cleanup()
        self.db.close()
        await super().close()


def check(cfg: Config) -> int:
    """Startup self-test that needs no Discord connection."""
    problems = []
    if not cfg.discord_token:
        problems.append("DISCORD_TOKEN is not set (copy .env.example to .env and fill it in)")
    versions = {}
    for name in ("numpy", "cv2", "scipy", "PIL", "skimage", "trimesh", "fast_simplification", "mapbox_earcut"):
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "installed")
        except ImportError:
            problems.append(f"missing dependency: {name} (run pip install -r requirements.txt)")
    if versions:
        log.info("dependencies: %s, discord.py %s", ", ".join(f"{k} {v}" for k, v in versions.items()), discord.__version__)
    db = Database(cfg.db_path)
    if not db.integrity_ok():
        problems.append("database integrity check failed")
    db.close()
    for p in problems:
        log.error(p)
    if not problems:
        log.info("configuration OK - data folder %s", cfg.data_dir)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Image-to-3D Studio Discord bot")
    parser.add_argument("--check", action="store_true", help="validate configuration and exit")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.data_dir / "logs")
    if args.check:
        return check(cfg)
    if not cfg.discord_token:
        log.error("DISCORD_TOKEN is not set. Copy .env.example to .env and add your bot token.")
        return 1
    bot = StudioBot(cfg)
    try:
        bot.run(cfg.discord_token, log_handler=None)
    except discord.LoginFailure:
        log.error("Discord rejected the token. Reset it in the Developer Portal and update .env.")
        return 1
    except discord.PrivilegedIntentsRequired:
        log.error("The bot asked for a privileged intent that is not enabled in the Developer Portal.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
