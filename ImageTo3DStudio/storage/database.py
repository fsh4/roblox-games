"""SQLite storage with WAL journaling and schema migrations.

All access goes through one connection guarded by a lock; callers in the bot's
event loop use ``run`` so queries execute in a worker thread and never block
Discord's gateway heartbeat.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        owner_id INTEGER NOT NULL,
        guild_id INTEGER,
        channel_id INTEGER,
        name TEXT NOT NULL,
        category TEXT,
        strategy TEXT,
        mode TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        current_version INTEGER NOT NULL DEFAULT 0,
        shared_with TEXT NOT NULL DEFAULT '[]',
        deleted INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id, deleted, updated_at)",
    """
    CREATE TABLE IF NOT EXISTS versions (
        project_id TEXT NOT NULL REFERENCES projects(id),
        number INTEGER NOT NULL,
        parent INTEGER,
        status TEXT NOT NULL DEFAULT 'pending',
        note TEXT,
        settings TEXT NOT NULL,
        stats TEXT,
        package_dir TEXT,
        zip_path TEXT,
        created_at REAL NOT NULL,
        PRIMARY KEY (project_id, number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        stage TEXT,
        progress REAL NOT NULL DEFAULT 0,
        error TEXT,
        payload TEXT NOT NULL,
        channel_id INTEGER,
        message_id INTEGER,
        created_at REAL NOT NULL,
        started_at REAL,
        finished_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS user_settings (
        user_id INTEGER PRIMARY KEY,
        defaults TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
]
SCHEMA_VERSION = "1"


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate()

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            for stmt in SCHEMA:
                cur.execute(stmt)
            cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return cur.rowcount

    def query(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def transaction(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                out = fn(self._conn)
                self._conn.execute("COMMIT")
                return out
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    async def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        return await asyncio.to_thread(fn, *args, **kwargs)

    def integrity_ok(self) -> bool:
        row = self.query_one("PRAGMA integrity_check")
        return bool(row) and row[0] == "ok"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
