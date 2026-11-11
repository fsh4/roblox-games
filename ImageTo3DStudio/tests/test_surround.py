import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from engine import analyzer as an
from engine import pipeline, surround
from engine import reconstruction as rc
from engine.procedural import LOD
from engine.settings import BuildSettings

from .conftest import image


def _shaded(mask: np.ndarray, colour, path) -> None:
    h, w = mask.shape
    yy = np.mgrid[0:h, 0:w][0]
    bg = np.repeat((120 + 25 * yy / h)[..., None], 3, axis=2)
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    light = 0.55 + 0.45 * np.sqrt(d / max(d.max(), 1))
    img = np.where(mask[..., None], np.asarray(colour, np.float64)[None, None] * light[..., None], bg)
    img = img + np.random.default_rng(1).normal(0, 2.5, img.shape)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=92)


def test_quilt_keeps_pinned_columns():
    rng = np.random.default_rng(0)
    src = rng.random((60, 120, 3)).astype(np.float32) * 255
    rows, cols = surround.quilt(src, np.zeros((60, 120), np.float32), 200, np.arange(108, 120), np.arange(0, 12))
    keep = 12 - surround.OVERLAP
    assert (cols[:, :keep] == np.arange(108, 108 + keep)[None]).all()
    assert (cols[:, -keep:] == np.arange(12 - keep, 12)[None]).all()
    assert rows.min() >= 0 and rows.max() < 60 and cols.max() < 120


def test_letterbox_bars_are_removed(tmp_path):
    img = np.full((300, 400, 3), 128, np.uint8)
    img[:40] = 0
    img[-40:] = 0
    cv2.circle(img, (200, 150), 60, (200, 60, 50), -1)
    Image.fromarray(img).save(tmp_path / "bars.png")
    loaded = an.load_image(str(tmp_path / "bars.png"))
    assert loaded.rgb.shape[0] == 220


def test_see_through_hole_is_kept(tmp_path):
    mask = np.zeros((480, 640), np.uint8)
    cv2.circle(mask, (320, 240), 170, 1, -1)
    cv2.circle(mask, (320, 240), 70, 0, -1)
    (tmp_path / "source").mkdir()
    _shaded(mask > 0, (230, 160, 90), tmp_path / "source" / "donut.jpg")
    spec = pipeline.analyse_project(str(tmp_path), [str(tmp_path / "source" / "donut.jpg")])
    assert spec["strategy"] == "inflate", spec["notes"]
    spec, arrays = pipeline.load_spec(str(tmp_path))
    mesh = rc.build_generic_scene(spec, arrays, LOD(0.4)).parts[0].mesh
    _, weld = np.unique(np.round(mesh.vertices, 6), axis=0, return_inverse=True)
    faces = weld.reshape(-1)[mesh.faces]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    e = len(np.unique(edges, axis=0))
    used = len(np.unique(faces))
    assert used - e + len(faces) == 0, "a ring has Euler characteristic 0 (one hole)"


@pytest.fixture(scope="module")
def egg(tmp_path_factory):
    proj = tmp_path_factory.mktemp("egg_surround")
    (proj / "source").mkdir()
    shutil.copy(image("egg_photo.jpg"), proj / "source" / "image_1.jpg")
    pipeline.analyse_project(str(proj), [str(proj / "source" / "image_1.jpg")])
    spec, arrays = pipeline.load_spec(str(proj))
    sources = pipeline._light_sources(str(proj), spec)
    proj_obj = rc.photo_projection_for(spec, sources, None, arrays)
    mesh = rc.build_generic_scene(spec, arrays, LOD(0.5)).parts[0].mesh
    return proj, spec, arrays, proj_obj, mesh


def test_camera_elevation_is_found(egg):
    *_, proj_obj, _ = egg
    elevation = np.degrees(np.arcsin(proj_obj.surround.tilt))
    assert 15 <= elevation <= 40, elevation


def test_back_is_synthesised_not_mirrored(egg):
    *_, proj_obj, mesh = egg
    v = mesh.vertices
    front = v[v[:, 2] < -0.3 * np.abs(v[:, 2]).max()]
    mirrored = front * np.array([1.0, 1.0, -1.0])
    a = proj_obj.surround.sample(front)
    b = proj_obj.surround.sample(mirrored)
    assert np.mean(np.linalg.norm(a - b, axis=1)) > 0.05, "the back must not be a copy of the front"


def test_texture_is_continuous_across_the_outline(egg):
    *_, proj_obj, mesh = egg
    xy, theta = proj_obj.surround._coords(mesh.vertices)
    edges = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    h = mesh.vertices[:, 1]
    mid = (h[edges[:, 0]] > 0.2 * h.max()) & (h[edges[:, 0]] < 0.8 * h.max())
    jump = np.abs(((theta[edges[:, 0]] - theta[edges[:, 1]]) + 180) % 360 - 180)
    assert jump[mid].max() < 20.0, "the azimuth must not jump on the body of the object"


def test_egg_package_with_synthesised_back(egg, tmp_path):
    proj, spec, arrays, *_ = egg
    res = pipeline.build_version(str(proj), str(tmp_path / "v1"), spec, arrays, BuildSettings(triangle_budget=2000, texture_resolution=256, quality="draft"), "Egg", "abcd1234", 1)
    assert res.stats["validation"]["ok"]
