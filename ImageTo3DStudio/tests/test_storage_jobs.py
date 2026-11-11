import asyncio
import shutil
import sys
import time

import pytest

from engine.settings import BuildSettings
from storage.database import Database
from storage.jobs import JobStore
from storage.projects import ProjectStore, UserDefaults
from workers.queue import JobQueue

from .conftest import image


@pytest.fixture()
def stores(tmp_path):
    db = Database(tmp_path / "studio.db")
    return db, ProjectStore(db, tmp_path / "projects"), JobStore(db)


def test_projects_versions_and_ownership(stores):
    db, ps, _ = stores
    p = ps.create_project(42, 1, 2, "  My   Chair ")
    assert p.name == "My Chair" and len(p.id) == 8
    assert ps.get(p.id).can_edit(42, []) and not ps.get(p.id).can_edit(7, [])
    assert ps.get(p.id).can_edit(7, [7]), "admins can edit"
    v1 = ps.new_version(p.id, BuildSettings(triangle_budget=900), None, "first")
    v2 = ps.new_version(p.id, BuildSettings(triangle_budget=1800), 1, "second")
    assert (v1.number, v2.number) == (1, 2)
    ps.finish_version(p.id, 2, {"triangles": 1700}, "/x", "/x.zip")
    assert ps.get(p.id).current_version == 2
    assert ps.get_version(p.id, 1).settings.triangle_budget == 900
    assert [v.number for v in ps.list_versions(p.id)] == [2, 1]
    assert ps.get("../../etc") is None and ps.get("zzzzzzzz") is None
    ps.set_defaults(42, UserDefaults(triangle_budget=777, private_results=True))
    assert ps.get_defaults(42).triangle_budget == 777 and ps.get_defaults(42).private_results
    assert db.integrity_ok()


def test_restart_recovery(stores):
    _, ps, js = stores
    p = ps.create_project(1, None, None, "R")
    a = js.create(p.id, 1, 1, "create", {"version_dir": "x"})
    b = js.create(p.id, 2, 1, "edit", {"version_dir": "y"})
    js.start(a.id)
    rec = js.recover_after_restart()
    assert [j.id for j in rec["interrupted"]] == [a.id]
    assert [j.id for j in rec["queued"]] == [b.id]
    assert js.get(a.id).status == "interrupted"


def _payload(ps, p, version, settings, images):
    return {"kind": "create", "project_id": p.id, "project_dir": str(ps.project_dir(p.id)), "version_dir": str(ps.version_dir(p.id, version)),
            "version": version, "title": p.name, "settings": settings.to_dict(), "images": images, "mode": "auto", "category": "bed"}


def test_queue_isolates_failures_and_runs_concurrently(stores):
    _, ps, js = stores
    events = []

    async def handler(ev):
        events.append(ev.kind)

    async def run():
        q = JobQueue(js, ps, handler, concurrency=2, timeout_s=600, python=sys.executable)
        await q.start()
        good = ps.create_project(1, None, None, "Good")
        shutil.copy(image("bed_photo.jpg"), ps.project_dir(good.id) / "source" / "image_1.jpg")
        s = BuildSettings(triangle_budget=800, texture_resolution=256, quality="draft")
        ps.new_version(good.id, s, None, "x")
        j1 = js.create(good.id, 1, 1, "create", _payload(ps, good, 1, s, ["image_1.jpg"]))
        bad = ps.create_project(1, None, None, "Bad")
        (ps.project_dir(bad.id) / "source" / "image_1.png").write_bytes(b"garbage")
        ps.new_version(bad.id, s, None, "x")
        j2 = js.create(bad.id, 1, 1, "create", _payload(ps, bad, 1, s, ["image_1.png"]))
        await q.submit(j1)
        await q.submit(j2)
        t = time.time()
        while (js.get(j1.id).active or js.get(j2.id).active) and time.time() - t < 600:
            await asyncio.sleep(0.5)
        await q.stop()
        return js.get(j1.id), js.get(j2.id)

    r1, r2 = asyncio.run(run())
    assert r1.status == "done"
    assert r2.status == "failed" and "image" in (r2.error or "")
    assert "progress" in events and "done" in events and "failed" in events


def test_queue_timeout_and_cancel(stores):
    _, ps, js = stores

    async def handler(ev):
        pass

    async def run():
        q = JobQueue(js, ps, handler, concurrency=1, timeout_s=2, python=sys.executable)
        await q.start()
        p = ps.create_project(1, None, None, "Slow")
        shutil.copy(image("bed_photo.jpg"), ps.project_dir(p.id) / "source" / "image_1.jpg")
        s = BuildSettings(triangle_budget=800, texture_resolution=256, quality="draft")
        ps.new_version(p.id, s, None, "x")
        ps.new_version(p.id, s, None, "y")
        slow = js.create(p.id, 1, 1, "create", _payload(ps, p, 1, s, ["image_1.jpg"]))
        queued = js.create(p.id, 2, 1, "create", _payload(ps, p, 2, s, ["image_1.jpg"]))
        await q.submit(slow)
        await q.submit(queued)
        assert await q.cancel(queued.id)
        t = time.time()
        while js.get(slow.id).active and time.time() - t < 60:
            await asyncio.sleep(0.2)
        await q.stop()
        return js.get(slow.id), js.get(queued.id)

    slow, queued = asyncio.run(run())
    assert slow.status == "failed" and "timed out" in slow.error
    assert queued.status == "cancelled"
