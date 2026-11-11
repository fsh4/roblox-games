"""Application configuration from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, optional quotes, # comments.

    Values already present in the real environment win, so secrets can be
    injected by the host without editing files.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _ids(name: str) -> List[int]:
    out = []
    for part in os.environ.get(name, "").replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


@dataclass
class Config:
    discord_token: str = ""
    dev_guild_id: Optional[int] = None
    data_dir: Path = ROOT / "data"
    max_concurrent_jobs: int = 2
    max_jobs_per_user: int = 2
    job_timeout_seconds: int = 900
    worker_memory_mb: int = 6144
    max_upload_mb: float = 20.0
    max_image_pixels: int = 40_000_000
    max_images_per_job: int = 6
    default_upload_limit_mb: float = 10.0
    admin_user_ids: List[int] = field(default_factory=list)
    register_app_emojis: bool = True
    keep_versions: int = 30
    log_level: str = "INFO"
    download_server: bool = False
    download_host: str = "0.0.0.0"
    download_port: int = 8765
    download_base_url: str = ""
    download_secret: str = ""
    download_ttl_hours: int = 24
    blender_path: str = ""

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "studio.db"


def load_config(env_file: Optional[Path] = None) -> Config:
    load_dotenv(env_file or ROOT / ".env")
    guild = os.environ.get("DISCORD_GUILD_ID", "").strip()
    cpu = os.cpu_count() or 2
    return Config(
        discord_token=os.environ.get("DISCORD_TOKEN", "").strip(),
        dev_guild_id=int(guild) if guild.isdigit() else None,
        data_dir=Path(os.environ.get("DATA_DIR", str(ROOT / "data"))).expanduser().resolve(),
        max_concurrent_jobs=max(1, _int("MAX_CONCURRENT_JOBS", max(1, cpu // 2))),
        max_jobs_per_user=max(1, _int("MAX_JOBS_PER_USER", 2)),
        job_timeout_seconds=max(60, _int("JOB_TIMEOUT_SECONDS", 900)),
        worker_memory_mb=max(512, _int("WORKER_MEMORY_MB", 6144)),
        max_upload_mb=max(1.0, _float("MAX_UPLOAD_MB", 20.0)),
        max_image_pixels=max(1_000_000, _int("MAX_IMAGE_PIXELS", 40_000_000)),
        max_images_per_job=min(10, max(1, _int("MAX_IMAGES_PER_JOB", 6))),
        default_upload_limit_mb=max(8.0, _float("DEFAULT_UPLOAD_LIMIT_MB", 10.0)),
        admin_user_ids=_ids("ADMIN_USER_IDS"),
        register_app_emojis=_bool("REGISTER_APP_EMOJIS", True),
        keep_versions=max(3, _int("KEEP_VERSIONS", 30)),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        download_server=_bool("DOWNLOAD_SERVER", False),
        download_host=os.environ.get("DOWNLOAD_HOST", "0.0.0.0"),
        download_port=_int("DOWNLOAD_PORT", 8765),
        download_base_url=os.environ.get("DOWNLOAD_BASE_URL", "").rstrip("/"),
        download_secret=os.environ.get("DOWNLOAD_SECRET", ""),
        download_ttl_hours=max(1, _int("DOWNLOAD_TTL_HOURS", 24)),
        blender_path=os.environ.get("BLENDER_PATH", "").strip(),
    )
