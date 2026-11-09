import asyncio
import re

import pytest


@pytest.fixture()
def bot(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config import load_config
    from main import StudioBot

    b = StudioBot(load_config())
    yield b
    b.db.close()


def walk(cmd):
    assert re.fullmatch(r"[-_\w]{1,32}", cmd["name"]) and cmd["name"] == cmd["name"].lower()
    assert 1 <= len(cmd.get("description", "x")) <= 100
    opts = cmd.get("options", [])
    assert len(opts) <= 25
    required_seen_optional = False
    for o in opts:
        if o.get("type") in (1, 2):
            walk(o)
            continue
        if not o.get("required", False):
            required_seen_optional = True
        else:
            assert not required_seen_optional, "required options must come first"
        assert len(o.get("choices", [])) <= 25
        for c in o.get("choices", []):
            assert 1 <= len(c["name"]) <= 100


def test_slash_commands_are_valid(bot):
    from bot.commands import StudioCommands

    bot.tree.add_command(StudioCommands(bot.studio))
    payload = [c.to_dict(bot.tree) for c in bot.tree.get_commands()]
    assert [p["name"] for p in payload] == ["3d"]
    subs = {o["name"] for o in payload[0]["options"]}
    assert {"create", "edit", "export", "history", "status", "settings", "help", "versions", "open", "delete"} <= subs
    for p in payload:
        walk(p)


def test_result_buttons_are_persistent(bot):
    from bot.views import CancelJobButton, ProjectAction, ResultView, RetryButton

    async def build():
        return ResultView(bot.studio, "0123abcd", 3)

    view = asyncio.run(build())
    ids = [item.custom_id for item in view.children]
    assert len(ids) == 5
    for cid in ids:
        assert re.fullmatch(ProjectAction.__discord_ui_compiled_template__, cid)
    assert re.fullmatch(CancelJobButton.__discord_ui_compiled_template__, "i3dc:0123456789ab")
    assert re.fullmatch(RetryButton.__discord_ui_compiled_template__, "i3dr:0123456789ab")


def test_modals_fit_discord_limits(bot):
    from engine.settings import BuildSettings
    from bot import views

    async def build():
        s = BuildSettings()
        modals = [views.BudgetModal("0123abcd", 1, s), views.SizeModal("0123abcd", 1, s), views.TextureModal("0123abcd", 1, s),
                  views.SurfaceModal("0123abcd", 1, s), views.RobloxModal("0123abcd", 1, s), views.ReanalyseModal("0123abcd", 1, s),
                  views.MaterialModal("0123abcd", 1, s, "frame", {"base_color": (0.8, 0.6, 0.4), "kind": "wood"}), views.UploadModal()]
        for m in modals:
            assert len(m.children) <= 5
            m.to_dict()
        panel = views.EditPanel(bot.studio, "0123abcd", 1, s)
        assert len(views.EDIT_OPTIONS) <= 25 and len(panel.children) <= 25

    asyncio.run(build())


def test_icons_render_as_png():
    from bot.icons import ICONS, draw_icon

    for name in ICONS:
        data = draw_icon(name)
        assert data.startswith(b"\x89PNG") and len(data) < 256_000


def test_embeds_respect_limits():
    from bot import embeds

    e = embeds.result_embed("Bed", "0123abcd", 2, {"stats": {"triangles": 4500, "budget": 5000, "size_studs": [3, 3, 7], "texture_maps": ["basecolor"],
                                                             "budget_report": {"reduction_percent": 77, "quality_impact": {"rating": "low", "p95_deviation_percent": 0.6}}},
                                                   "warnings": ["w" * 500], "notes": ["n" * 500]}, "preview.png")
    total = len(e.title or "") + len(e.description or "") + sum(len(f.name) + len(f.value) for f in e.fields)
    assert total < 6000 and all(len(f.value) <= 1024 for f in e.fields)
    assert "█" in embeds.bar(0.5)
