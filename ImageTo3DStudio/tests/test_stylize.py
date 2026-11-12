import shutil

import cv2
import numpy as np
import pytest

from engine import pipeline, sculpt, stylize
from engine.settings import BuildSettings

from .conftest import image


def test_old_studs_setting_becomes_blocky():
    assert BuildSettings(art_style="studs").validated().art_style == "blocky"
    assert BuildSettings(art_style="unknown").validated().art_style == "realistic"


def test_nearly_symmetric_body_becomes_exactly_symmetric_and_keeps_its_accessory():
    mask = np.zeros((200, 160), np.uint8)
    cv2.ellipse(mask, (78, 115), (60, 75), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (90, 110), (55, 70), 0, 0, 360, 1, -1)
    cv2.fillPoly(mask, [np.array([[95, 45], [118, 8], [112, 50]])], 1)
    parts = sculpt.shape_parts(mask > 0)
    assert parts.symmetric
    assert parts.accessories == 1
    body = parts.regions == 0
    assert (body == body[:, ::-1]).mean() > 0.995
    assert (parts.regions > 0).sum() > 100


def test_parts_made_of_thin_pieces_stay_one_piece():
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (95, 10), (105, 190), 1, -1)
    cv2.rectangle(mask, (60, 140), (140, 150), 1, -1)
    parts = sculpt.shape_parts(mask > 0)
    assert parts.accessories == 0


def test_blocks_merge_same_colour_cubes_into_slabs():
    occ = np.ones((6, 2, 6), bool)
    labels = np.zeros(occ.shape, np.int64)
    ids = stylize.block_boxes(occ, labels)
    assert ids.min() == 0
    sizes = np.bincount(ids.ravel())
    assert sizes.max() <= np.prod(stylize.BOX_LIMITS)
    for b in range(ids.max() + 1):
        assert len(np.unique(np.argwhere(ids == b)[:, 1])) == 1, "a slab is one cube tall"


def test_palette_unification_merges_identical_colours():
    pal = np.array([[180, 128, 160], [181, 129, 161], [60, 128, 128]], np.float32)
    labels = np.array([0, 1, 2, -1])
    new_pal, new_labels = stylize.unify_palette(pal, labels)
    assert len(new_pal) == 2
    assert list(new_labels) == [0, 0, 1, -1]


@pytest.fixture(scope="module")
def plush_blocky(tmp_path_factory):
    proj = tmp_path_factory.mktemp("plush_blocky")
    (proj / "source").mkdir()
    shutil.copy(image("plush_render.png"), proj / "source" / "image_1.png")
    pipeline.analyse_project(str(proj), [str(proj / "source" / "image_1.png")])
    spec, arrays = pipeline.load_spec(str(proj))
    res = pipeline.build_version(str(proj), str(proj / "v1"), spec, arrays, BuildSettings(triangle_budget=3000, texture_resolution=512, quality="draft", art_style="blocky"), "Plush", "abcd1234", 1)
    return spec, res


def test_blocky_model_is_valid_and_has_no_studs(plush_blocky):
    spec, res = plush_blocky
    assert spec["mirror_plane"]
    assert res.stats["validation"]["ok"]
    assert not res.stats["validation"]["warnings"]
    assert res.stats["triangles"] <= 3000
    assert any(n.startswith("Block style") for n in res.notes)
    assert res.stats["native_parts"] > 10


def test_blocky_plush_gets_eye_highlights(plush_blocky):
    _, res = plush_blocky
    assert any("eye" in n and "highlight" in n for n in res.notes), res.notes


def _face_grid(spots):
    occ = np.ones((12, 12, 6), bool)
    labels = np.zeros(occ.shape, np.int64)
    for x, y in spots:
        labels[x, y, 0] = 1
    pal = np.array([[200, 128, 128], [40, 128, 128]], np.float32)
    return occ, labels, pal, np.zeros(occ.shape, np.int64)


def test_eyes_need_a_mirrored_pair():
    occ, labels, pal, region = _face_grid([(4, 8), (7, 8)])
    eyes = stylize.find_eyes(occ, labels, pal, region, twice_centre=11)
    assert eyes.sum() == 2
    occ, labels, pal, region = _face_grid([(4, 8)])
    assert stylize.find_eyes(occ, labels, pal, region, twice_centre=11).sum() == 0
    occ, labels, pal, region = _face_grid([(4, 8), (7, 5)])
    assert stylize.find_eyes(occ, labels, pal, region, twice_centre=11).sum() == 0
