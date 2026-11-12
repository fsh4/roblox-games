import shutil

import numpy as np
import pytest

from engine import composite, pipeline
from engine.settings import BuildSettings

from .conftest import image


def test_ray_enters_the_box_on_its_top():
    lo, hi = np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0])
    hit = composite.ray_box(np.array([0.3, 5.0, 0.2]), np.array([0.0, -1.0, 0.0]), lo, hi)
    assert np.allclose(hit, [0.3, 2.0, 0.2])
    assert composite.ray_box(np.array([3.0, 5.0, 0.0]), np.array([0.0, -1.0, 0.0]), lo, hi) is None


def test_boxy_outline_test():
    import cv2

    hexagon = np.zeros((300, 300), np.uint8)
    cv2.fillPoly(hexagon, [np.array([[150, 30], [270, 90], [270, 210], [150, 270], [30, 210], [30, 90]])], 1)
    disc = np.zeros((300, 300), np.uint8)
    cv2.circle(disc, (150, 150), 120, 1, -1)
    assert composite.boxy(hexagon > 0)
    assert not composite.boxy(disc > 0)


@pytest.fixture(scope="module")
def box_project(tmp_path_factory):
    proj = tmp_path_factory.mktemp("box_project")
    (proj / "source").mkdir()
    shutil.copy(image("box_photo.jpg"), proj / "source" / "image_1.jpg")
    spec = pipeline.analyse_project(str(proj), [str(proj / "source" / "image_1.jpg")])
    return proj, spec


def test_parcel_photo_is_a_box_with_the_dessert_on_top(box_project):
    _, spec = box_project
    assert spec["strategy"] == "composite", spec["notes"]
    assert spec["template"] == "box"
    assert not any("sofa" in n.lower() for n in spec["notes"])
    assert len(spec["parts"]) == 1
    placement = np.asarray(spec["parts"][0]["placement"])
    height = spec["params"]["height"]
    assert abs(placement[1, 3] - height) < 0.25 * height, "the dessert stands on the top face"


def test_parcel_package_is_valid(box_project):
    proj, _ = box_project
    spec, arrays = pipeline.load_spec(str(proj))
    res = pipeline.build_version(str(proj), str(proj / "v1"), spec, arrays, BuildSettings(triangle_budget=3000, texture_resolution=256, quality="draft"), "Parcel", "abcd1234", 1)
    assert res.stats["validation"]["ok"]
    assert res.stats["parts"] >= 2


@pytest.mark.parametrize("name", ["chair_render.png", "table_render.png"])
def test_furniture_is_never_guessed(tmp_path, name):
    (tmp_path / "source").mkdir()
    shutil.copy(image(name), tmp_path / "source" / name)
    spec = pipeline.analyse_project(str(tmp_path), [str(tmp_path / "source" / name)])
    assert spec["strategy"] != "template", spec["notes"]
