"""Asynchronous job queue that runs every job in an isolated worker process.

* A fixed number of consumer tasks bounds CPU use (MAX_CONCURRENT_JOBS).
* Each job runs ``python -m workers.processor job.json`` with a timeout, a memory
  limit (Linux) and capped BLAS threads; its stdout is parsed line by line.
* Events (progress, done, failed, cancelled) go to one async callback, which the
  bot uses to update Discord messages. Nothing here touches Discord directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from storage.jobs import Job, JobStore
from storage.projects import ProjectStore

log = logging.getLogger("studio.queue")
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class JobEvent:
    kind: str
    job: Job
    data: dict = field(default_factory=dict)


EventHandler = Callable[[JobEvent], Awaitable[None]]


def _limit_resources(memory_mb: int):
    def apply():
        try:
            import resource

            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            os.nice(5)
        except Exception:
            pass

    return apply if sys.platform.startswith("linux") else None


def worker_env(base: Optional[dict] = None) -> dict:
    """Environment for a worker process: capped math threads, no tokens or secrets."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if "TOKEN" not in k.upper() and "SECRET" not in k.upper()}
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
    return env


class JobQueue:
    def __init__(self, jobs: JobStore, projects: ProjectStore, handler: EventHandler, concurrency: int = 2,
                 timeout_s: int = 900, memory_mb: int = 6144, python: Optional[str] = None):
        self.jobs = jobs
        self.projects = projects
        self.handler = handler
        self.concurrency = max(1, concurrency)
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.python = python or sys.executable
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._order: List[str] = []
        self._procs: Dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set = set()
        self._tasks: List[asyncio.Task] = []
        self._last_progress: Dict[str, float] = {}

    async def start(self) -> Dict[str, List[Job]]:
        recovered = await asyncio.to_thread(self.jobs.recover_after_restart)
        for job in recovered["queued"]:
            self._order.append(job.id)
            self._queue.put_nowait(job.id)
        self._tasks = [asyncio.create_task(self._consumer(i), name=f"job-consumer-{i}") for i in range(self.concurrency)]
        log.info("job queue started with %d worker slot(s); resumed %d queued, %d interrupted", self.concurrency, len(recovered["queued"]), len(recovered["interrupted"]))
        return recovered

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for jid, proc in list(self._procs.items()):
            if proc.returncode is None:
                proc.kill()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def submit(self, job: Job) -> int:
        self._order.append(job.id)
        await self._queue.put(job.id)
        return self.position(job.id)

    def position(self, job_id: str) -> int:
        try:
            return self._order.index(job_id) + 1
        except ValueError:
            return 0

    @property
    def waiting(self) -> int:
        return len(self._order)

    @property
    def running(self) -> int:
        return len(self._procs)

    async def cancel(self, job_id: str) -> bool:
        job = await asyncio.to_thread(self.jobs.get, job_id)
        if job is None or not job.active:
            return False
        self._cancelled.add(job_id)
        proc = self._procs.get(job_id)
        if proc is not None and proc.returncode is None:
            proc.kill()
        elif job_id in self._order:
            self._order.remove(job_id)
            await asyncio.to_thread(self.jobs.finish, job_id, "cancelled", "cancelled by the user")
            await self._emit("cancelled", job_id)
        return True

    async def _emit(self, kind: str, job_id: str, data: Optional[dict] = None) -> None:
        job = await asyncio.to_thread(self.jobs.get, job_id)
        if job is None:
            return
        try:
            await self.handler(JobEvent(kind, job, data or {}))
        except Exception:
            log.exception("event handler failed for %s %s", kind, job_id)

    async def _consumer(self, slot: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                if job_id in self._order:
                    self._order.remove(job_id)
                if job_id in self._cancelled:
                    continue
                job = await asyncio.to_thread(self.jobs.get, job_id)
                if job is None or job.status != "queued":
                    continue
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job %s crashed in the queue", job_id)
                await asyncio.to_thread(self.jobs.finish, job_id, "failed", "internal error in the job queue")
                await self._emit("failed", job_id, {"error": "internal error in the job queue"})
            finally:
                self._queue.task_done()

    async def _run(self, job: Job) -> None:
        payload = dict(job.payload)
        work_dir = Path(payload["version_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        job_file = work_dir / f"job_{job.id}.json"
        job_file.write_text(json.dumps(payload), encoding="utf-8")
        await asyncio.to_thread(self.jobs.start, job.id)
        await self._emit("started", job.id)
        env = worker_env()
        started = time.time()
        proc = await asyncio.create_subprocess_exec(
            self.python, "-m", "workers.processor", str(job_file), cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, preexec_fn=_limit_resources(self.memory_mb),
            limit=16 * 1024 * 1024)
        self._procs[job.id] = proc
        result: Optional[dict] = None
        error: Optional[str] = None
        stderr_tail: List[str] = []

        async def read_stdout():
            nonlocal result, error
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                tag, _, body = line.partition(" ")
                try:
                    data = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    continue
                if tag == "PROGRESS":
                    await asyncio.to_thread(self.jobs.progress, job.id, data.get("stage", ""), float(data.get("progress", 0)))
                    now = time.time()
                    if now - self._last_progress.get(job.id, 0) > 1.0:
                        self._last_progress[job.id] = now
                        await self._emit("progress", job.id, data)
                elif tag == "RESULT":
                    result = data
                elif tag == "ERROR":
                    error = data.get("error", "unknown error")

        async def read_stderr():
            assert proc.stderr is not None
            async for raw in proc.stderr:
                stderr_tail.append(raw.decode("utf-8", "replace").rstrip())
                del stderr_tail[:-40]

        try:
            await asyncio.wait_for(asyncio.gather(read_stdout(), read_stderr(), proc.wait()), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            error = f"timed out after {self.timeout_s // 60} minutes"
        finally:
            self._procs.pop(job.id, None)
            self._last_progress.pop(job.id, None)
            try:
                job_file.unlink()
            except OSError:
                pass
        elapsed = time.time() - started
        if job.id in self._cancelled:
            self._cancelled.discard(job.id)
            await asyncio.to_thread(self.jobs.finish, job.id, "cancelled", "cancelled by the user")
            await asyncio.to_thread(self.projects.fail_version, job.project_id, job.version, "cancelled")
            await self._emit("cancelled", job.id)
            return
        if result is not None and proc.returncode == 0:
            await asyncio.to_thread(self.projects.finish_version, job.project_id, job.version, result.get("stats", {}), result.get("package_dir", ""), result.get("zip", ""))
            await asyncio.to_thread(self.projects.update_project, job.project_id, category=result.get("category"), strategy=result.get("strategy"))
            await asyncio.to_thread(self.jobs.finish, job.id, "done")
            log.info("job %s (%s) finished in %.1fs", job.id, job.kind, elapsed)
            await self._emit("done", job.id, result)
            return
        if error is None:
            error = f"worker exited with code {proc.returncode}"
        if stderr_tail:
            log.warning("job %s failed: %s\n%s", job.id, error, "\n".join(stderr_tail[-15:]))
        await asyncio.to_thread(self.jobs.finish, job.id, "failed", error)
        await asyncio.to_thread(self.projects.fail_version, job.project_id, job.version, error)
        await self._emit("failed", job.id, {"error": error})
