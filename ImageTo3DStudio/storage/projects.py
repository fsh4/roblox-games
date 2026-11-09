"""Projects, versions and per-user defaults."""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from engine.settings import BuildSettings
from utils.filesystem import remove_tree, safe_join

from .database import Database


@dataclass
class Project:
    id: str
    owner_id: int
    guild_id: Optional[int]
    channel_id: Optional[int]
    name: str
    category: Optional[str]
    strategy: Optional[str]
    mode: Optional[str]
    status: str
    current_version: int
    shared_with: List[int]
    created_at: float
    updated_at: float

    def can_view(self, user_id: int, admins: List[int]) -> bool:
        return user_id == self.owner_id or user_id in self.shared_with or user_id in admins

    def can_edit(self, user_id: int, admins: List[int]) -> bool:
        return user_id == self.owner_id or user_id in admins


@dataclass
class Version:
    project_id: str
    number: int
    parent: Optional[int]
    status: str
    note: str
    settings: BuildSettings
    stats: Dict
    package_dir: Optional[str]
    zip_path: Optional[str]
    created_at: float


@dataclass
class UserDefaults:
    triangle_budget: int = 5000
    texture_resolution: int = 1024
    quality: str = "standard"
    shading: str = "smooth"
    private_results: bool = False
    collision: str = "hull"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _project(row) -> Project:
    return Project(row["id"], row["owner_id"], row["guild_id"], row["channel_id"], row["name"], row["category"], row["strategy"],
                   row["mode"], row["status"], row["current_version"], json.loads(row["shared_with"] or "[]"), row["created_at"], row["updated_at"])


def _version(row) -> Version:
    return Version(row["project_id"], row["number"], row["parent"], row["status"], row["note"] or "",
                   BuildSettings.from_dict(json.loads(row["settings"])), json.loads(row["stats"]) if row["stats"] else {},
                   row["package_dir"], row["zip_path"], row["created_at"])


class ProjectStore:
    def __init__(self, db: Database, projects_dir: Path):
        self.db = db
        self.root = Path(projects_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        return safe_join(self.root, project_id)

    def version_dir(self, project_id: str, number: int) -> Path:
        return safe_join(self.root, project_id, "versions", f"v{number:04d}")

    def create_project(self, owner_id: int, guild_id: Optional[int], channel_id: Optional[int], name: str) -> Project:
        now = time.time()
        for _ in range(5):
            pid = secrets.token_hex(4)
            if self.db.query_one("SELECT 1 FROM projects WHERE id=?", (pid,)) is None:
                break
        clean = " ".join(name.split())[:60] or "Model"
        self.db.execute(
            "INSERT INTO projects(id, owner_id, guild_id, channel_id, name, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (pid, owner_id, guild_id, channel_id, clean, "analysing", now, now))
        (self.project_dir(pid) / "source").mkdir(parents=True, exist_ok=True)
        return self.get(pid)

    def get(self, project_id: str) -> Optional[Project]:
        if not project_id or not all(c in "0123456789abcdef" for c in project_id.lower()):
            return None
        row = self.db.query_one("SELECT * FROM projects WHERE id=? AND deleted=0", (project_id.lower(),))
        return _project(row) if row else None

    def list_for_user(self, user_id: int, limit: int = 25, offset: int = 0, include_shared: bool = True) -> List[Project]:
        rows = self.db.query(
            "SELECT * FROM projects WHERE deleted=0 AND (owner_id=? OR (? AND shared_with LIKE ?)) ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (user_id, 1 if include_shared else 0, f"%{user_id}%", limit, offset))
        return [p for p in (_project(r) for r in rows) if p.owner_id == user_id or user_id in p.shared_with]

    def count_for_user(self, user_id: int) -> int:
        row = self.db.query_one("SELECT COUNT(*) FROM projects WHERE deleted=0 AND owner_id=?", (user_id,))
        return int(row[0]) if row else 0

    def update_project(self, project_id: str, **fields) -> None:
        allowed = {"name", "category", "strategy", "mode", "status", "current_version", "channel_id"}
        cols = {k: v for k, v in fields.items() if k in allowed}
        if "shared_with" in fields:
            cols["shared_with"] = json.dumps(sorted(set(int(x) for x in fields["shared_with"])))
        if not cols:
            return
        cols["updated_at"] = time.time()
        sets = ", ".join(f"{k}=?" for k in cols)
        self.db.execute(f"UPDATE projects SET {sets} WHERE id=?", (*cols.values(), project_id))

    def soft_delete(self, project_id: str) -> None:
        self.db.execute("UPDATE projects SET deleted=1, status='deleted', updated_at=? WHERE id=?", (time.time(), project_id))
        remove_tree(self.project_dir(project_id), self.root)

    def new_version(self, project_id: str, settings: BuildSettings, parent: Optional[int], note: str) -> Version:
        def tx(conn):
            row = conn.execute("SELECT COALESCE(MAX(number), 0) FROM versions WHERE project_id=?", (project_id,)).fetchone()
            number = int(row[0]) + 1
            conn.execute("INSERT INTO versions(project_id, number, parent, status, note, settings, created_at) VALUES (?,?,?,?,?,?,?)",
                         (project_id, number, parent, "pending", note[:200], json.dumps(settings.to_dict()), time.time()))
            return number

        number = self.db.transaction(tx)
        self.version_dir(project_id, number).mkdir(parents=True, exist_ok=True)
        return self.get_version(project_id, number)

    def get_version(self, project_id: str, number: int) -> Optional[Version]:
        row = self.db.query_one("SELECT * FROM versions WHERE project_id=? AND number=?", (project_id, number))
        return _version(row) if row else None

    def list_versions(self, project_id: str, limit: int = 25) -> List[Version]:
        rows = self.db.query("SELECT * FROM versions WHERE project_id=? ORDER BY number DESC LIMIT ?", (project_id, limit))
        return [_version(r) for r in rows]

    def latest_ready(self, project_id: str) -> Optional[Version]:
        row = self.db.query_one("SELECT * FROM versions WHERE project_id=? AND status='ready' ORDER BY number DESC LIMIT 1", (project_id,))
        return _version(row) if row else None

    def finish_version(self, project_id: str, number: int, stats: dict, package_dir: str, zip_path: str) -> None:
        self.db.execute("UPDATE versions SET status='ready', stats=?, package_dir=?, zip_path=? WHERE project_id=? AND number=?",
                        (json.dumps(stats), package_dir, zip_path, project_id, number))
        self.update_project(project_id, current_version=number, status="ready")

    def fail_version(self, project_id: str, number: int, error: str) -> None:
        self.db.execute("UPDATE versions SET status='failed', note=? WHERE project_id=? AND number=?", (error[:200], project_id, number))
        proj = self.get(project_id)
        if proj and proj.current_version == 0:
            self.update_project(project_id, status="failed")

    def prune_versions(self, project_id: str, keep: int) -> int:
        """Delete packages of old versions; their settings stay so they can be rebuilt."""
        rows = self.db.query("SELECT number FROM versions WHERE project_id=? AND status='ready' ORDER BY number DESC", (project_id,))
        removed = 0
        for r in rows[keep:]:
            vdir = self.version_dir(project_id, r["number"])
            if vdir.exists():
                remove_tree(vdir, self.root)
                removed += 1
            self.db.execute("UPDATE versions SET status='archived', package_dir=NULL, zip_path=NULL WHERE project_id=? AND number=?", (project_id, r["number"]))
        return removed

    def get_defaults(self, user_id: int) -> UserDefaults:
        row = self.db.query_one("SELECT defaults FROM user_settings WHERE user_id=?", (user_id,))
        if not row:
            return UserDefaults()
        data = json.loads(row["defaults"])
        d = UserDefaults()
        for k, v in data.items():
            if hasattr(d, k):
                setattr(d, k, v)
        return d

    def set_defaults(self, user_id: int, defaults: UserDefaults) -> None:
        self.db.execute("INSERT INTO user_settings(user_id, defaults, updated_at) VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET defaults=excluded.defaults, updated_at=excluded.updated_at",
                        (user_id, json.dumps(defaults.to_dict()), time.time()))
