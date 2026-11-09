"""Drive StudioService flows with stand-in Discord objects (no network, no token)."""
import asyncio
import itertools
import time

import discord
import pytest

from .conftest import image

_ids = itertools.count(1000)


class FakeChannel:
    id = 555


class FakeMessage:
    def __init__(self, **kwargs):
        self.id = next(_ids)
        self.channel = FakeChannel()
        self.kwargs = kwargs


class FakeResponse:
    def __init__(self, log):
        self.log = log
        self.done = False

    def is_done(self):
        return self.done

    async def defer(self, thinking=False, ephemeral=False):
        self.done = True
        self.log.append(("defer", {"ephemeral": ephemeral}))

    async def send_message(self, **kwargs):
        self.done = True
        self.log.append(("send_message", kwargs))

    async def send_modal(self, modal):
        self.done = True
        self.log.append(("modal", {"modal": modal}))

    async def edit_message(self, **kwargs):
        self.done = True
        self.log.append(("edit_message", kwargs))


class FakeFollowup:
    def __init__(self, log):
        self.log = log

    async def send(self, wait=True, ephemeral=False, **kwargs):
        msg = FakeMessage(**kwargs)
        self.log.append(("followup", kwargs))
        return msg

    async def edit_message(self, message_id, **kwargs):
        self.log.append(("followup_edit", kwargs))


class FakeUser:
    def __init__(self, uid):
        self.id = uid


class FakeInteraction:
    def __init__(self, client, uid=4242):
        self.log = []
        self.client = client
        self.user = FakeUser(uid)
        self.guild_id = 1
        self.channel_id = 555
        self.guild = None
        self.filesize_limit = 25 * 1024 * 1024
        self.response = FakeResponse(self.log)
        self.followup = FakeFollowup(self.log)

    async def original_response(self):
        return FakeMessage()


class FakeAttachment:
    def __init__(self, path):
        self.filename = path.name
        self.data = path.read_bytes()
        self.size = len(self.data)

    async def read(self):
        return self.data


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    from config import load_config
    from main import StudioBot

    bot = StudioBot(load_config())
    yield bot
    bot.db.close()


async def _wait_idle(svc, timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        if not svc.jobs.active():
            return
        await asyncio.sleep(0.5)
    raise TimeoutError("jobs did not finish")


def test_create_edit_export_flow(studio):
    svc = studio.studio

    async def run():
        await svc.queue.start()
        svc.projects.set_defaults(4242, __import__("storage.projects", fromlist=["UserDefaults"]).UserDefaults(triangle_budget=1500, texture_resolution=256, quality="draft"))
        it = FakeInteraction(studio)
        await svc.start_create(it, [FakeAttachment(image("bed_photo.jpg"))], category="bed", name="Test Bed")
        await _wait_idle(svc)
        project = svc.projects.list_for_user(4242, 5, 0)[0]
        kinds = [k for k, _ in it.log]
        assert kinds[0] == "defer"
        progress_msg = next(kw for k, kw in it.log if k == "followup" and "view" in kw)
        assert progress_msg["embed"].title.startswith("Test Bed")
        result = [kw for k, kw in it.log if k == "followup" and kw.get("files")]
        assert result, "the result message with files was not sent"
        names = {f.filename for f in result[-1]["files"]}
        assert "preview.png" in names and any(n.endswith(".zip") for n in names)
        assert result[-1]["embed"].title.endswith("ready")
        assert project.current_version == 1

        from bot.views import handle_action
        from engine.settings import BuildSettings

        edit = FakeInteraction(studio)
        v1 = svc.projects.get_version(project.id, 1)
        new = BuildSettings.from_dict({**v1.settings.to_dict(), "triangle_budget": 700})
        await svc.queue_build(edit, svc.projects.get(project.id), 1, new, "budget 700")
        await _wait_idle(svc)
        assert svc.projects.get(project.id).current_version == 2
        assert svc.projects.get_version(project.id, 2).stats["triangles"] <= 735

        exp = FakeInteraction(studio)
        await svc.export(exp, svc.projects.get(project.id), 2, "stl")
        sent = [kw for k, kw in exp.log if k == "followup" and kw.get("file")]
        assert sent and sent[0]["file"].filename.endswith(".stl")

        stranger = FakeInteraction(studio, uid=1)
        await handle_action(stranger, "edit", project.id, 2)
        refused = [kw for k, kw in stranger.log if k in ("send_message", "followup")]
        assert refused and "not found" in refused[0]["embed"].title.lower()

        files = FakeInteraction(studio)
        await handle_action(files, "files", project.id, 2)
        assert any(k == "send_message" and kw.get("view") is not None for k, kw in files.log)
        await svc.queue.stop()

    asyncio.run(run())


def test_rejects_non_image_upload(studio, tmp_path):
    svc = studio.studio
    fake = tmp_path / "virus.png"
    fake.write_bytes(b"MZ not an image at all")

    async def run():
        it = FakeInteraction(studio)
        await svc.start_create(it, [FakeAttachment(fake)])
        msgs = [kw for k, kw in it.log if k == "followup"]
        assert msgs and "rejected" in msgs[0]["embed"].title.lower()
        assert not svc.projects.list_for_user(4242, 5, 0)

    asyncio.run(run())
