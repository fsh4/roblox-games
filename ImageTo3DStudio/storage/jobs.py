"""Job records: queued -> running -> done | failed | cancelled | interrupted."""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .database import Database

ACTIVE = ("queued", "running")


@dataclass
class Job:
    id: str
    project_id: str
    version: int
    user_id: int
    kind: str
    status: str
    stage: str
    progress: float
    error: Optional[str]
    payload: Dict
    channel_id: Optional[int]
    message_id: Optional[int]
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]

    @property
    def active(self) -> bool:
        return self.status in ACTIVE


def _job(row) -> Job:
    return Job(row["id"], row["project_id"], row["version"], row["user_id"], row["kind"], row["status"], row["stage"] or "",
               row["progress"], row["error"], json.loads(row["payload"]), row["channel_id"], row["message_id"],
               row["created_at"], row["started_at"], row["finished_at"])


class JobStore:
    def __init__(self, db: Database):
        self.db = db

    def create(self, project_id: str, version: int, user_id: int, kind: str, payload: dict, channel_id: Optional[int] = None) -> Job:
        jid = secrets.token_hex(6)
        self.db.execute("INSERT INTO jobs(id, project_id, version, user_id, kind, status, stage, progress, payload, channel_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (jid, project_id, version, user_id, kind, "queued", "waiting in queue", 0.0, json.dumps(payload), channel_id, time.time()))
        return self.get(jid)

    def get(self, job_id: str) -> Optional[Job]:
        row = self.db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return _job(row) if row else None

    def set_message(self, job_id: str, channel_id: Optional[int], message_id: Optional[int]) -> None:
        self.db.execute("UPDATE jobs SET channel_id=?, message_id=? WHERE id=?", (channel_id, message_id, job_id))

    def start(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET status='running', started_at=?, stage='starting' WHERE id=?", (time.time(), job_id))

    def progress(self, job_id: str, stage: str, progress: float) -> None:
        self.db.execute("UPDATE jobs SET stage=?, progress=? WHERE id=? AND status='running'", (stage[:120], float(progress), job_id))

    def finish(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        self.db.execute("UPDATE jobs SET status=?, error=?, finished_at=?, progress=CASE WHEN ?='done' THEN 1.0 ELSE progress END WHERE id=?",
                        (status, (error or "")[:500] or None, time.time(), status, job_id))

    def active_for_user(self, user_id: int) -> List[Job]:
        rows = self.db.query("SELECT * FROM jobs WHERE user_id=? AND status IN ('queued','running') ORDER BY created_at", (user_id,))
        return [_job(r) for r in rows]

    def active(self) -> List[Job]:
        return [_job(r) for r in self.db.query("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created_at")]

    def recent_for_user(self, user_id: int, limit: int = 10) -> List[Job]:
        return [_job(r) for r in self.db.query("SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (user_id, limit))]

    def recover_after_restart(self) -> Dict[str, List[Job]]:
        """Running jobs died with the old process; queued ones can resume."""
        running = [_job(r) for r in self.db.query("SELECT * FROM jobs WHERE status='running'")]
        for j in running:
            self.finish(j.id, "interrupted", "the bot restarted while this job was running")
        queued = [_job(r) for r in self.db.query("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at")]
        return {"interrupted": running, "queued": queued}

    def stats(self) -> Dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        return {r["status"]: int(r["n"]) for r in rows}
