import shutil

import cv2
import numpy as np
import pytest

from engine import analyzer as an
from engine import pipeline, sculpt
from engine import reconstruction as rc
from engine.procedural import LOD
from engine.settings import BuildSettings

from .conftest import image


def _disc(side: int = 200, radius: int = 80) -> np.ndarray:
    m = np.zeros((side, side), np.uint8)
    cv2.circle(m, (side // 2, side // 2), radius, 1, -1)
    return m > 0


def test_ball_field_of_a_disc_is_a_sphere():
    mask = _disc()
    h2 = sculpt.ball_field(mask)
    yy, xx = np.mgrid[:200, :200]
    expected = 80.0 ** 2 - (xx - 100) ** 2 - (yy - 100) ** 2
    inner = expected > 400
    assert np.abs(np.sqrt(h2[inner]) - np.sqrt(expected[inner])).max() < 2.5
    assert (h2[~mask] <= 0).all()


def test_ball_field_keeps_thin_parts_thin():
    mask = np.zeros((220, 220), np.uint8)
    cv2.circle(mask, (110, 140), 70, 1, -1)
    cv2.rectangle(mask, (104, 10), (116, 80), 1, -1)
    h2 = sculpt.ball_field(mask > 0)
    assert np.sqrt(h2[140, 110]) > 60
    assert np.sqrt(max(h2[30, 110], 0)) < 9


def test_flat_base_detection():
    box = np.zeros((120, 120), bool)
    box[20:100, 30:90] = True
    assert sculpt.flat_edge_share(box, "bottom") > 0.9
    assert sculpt.flat_edge_share(_disc(), "bottom") < sculpt.FLAT_SHARE


def test_outline_straightness_separates_curves_from_polygons():
    def contour(mask):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(cs, key=cv2.contourArea)

    box = np.zeros((200, 200), bool)
    box[40:160, 30:170] = True
    assert an.outline_straightness(contour(box)) > 0.9
    assert an.outline_straightness(contour(_disc())) < 0.1


def test_inflated_disc_is_a_closed_ball():
    mask = _disc(160, 70)[10:151, 10:151]
    h2, _ = sculpt.thickness_field(mask, False)
    spec = {"strategy": "inflate", "inflate_model": "balls", "dimensions": [4.0, 4.0, 4.0], "depth_scale": 1.0}
    scene = rc.build_generic_scene(spec, {"mask": mask.astype(np.uint8), "thickness": h2.astype(np.float32)}, LOD(0.5))
    mesh = scene.parts[0].mesh
    lo, hi = mesh.bounds()
    size = hi - lo
    assert np.allclose(size, 4.0, atol=0.3), size
    sphere = 4.0 / 3.0 * np.pi * 2.0 ** 3
    assert abs(abs(mesh.signed_volume()) - sphere) / sphere < 0.12


@pytest.fixture(scope="module")
def egg_project(tmp_path_factory):
    proj = tmp_path_factory.mktemp("egg_project")
    (proj / "source").mkdir()
    for i in (1, 2):
        shutil.copy(image("egg_photo.jpg"), proj / "source" / f"image_{i}.jpg")
    return proj


def test_decorated_egg_is_not_furniture(egg_project):
    spec = pipeline.analyse_project(str(egg_project), [str(egg_project / "source" / "image_1.jpg")])
    assert spec["strategy"] == "inflate", spec["notes"]
    assert spec["inflate_model"] == "balls"
    assert spec["alternatives"] == [], "no furniture template may be tried on a curved outline"
    w, h, d = spec["dimensions"]
    assert 0.75 < d / w < 1.1, "an egg is about as deep as it is wide"
    spec, arrays = pipeline.load_spec(str(egg_project))
    res = pipeline.build_version(str(egg_project), str(egg_project / "v1"), spec, arrays, BuildSettings(triangle_budget=3000, texture_resolution=256, quality="draft"), "Egg", "abcd1234", 1)
    assert res.stats["triangles"] <= 3000 * 1.05
    assert res.stats["validation"]["ok"]
    assert all(m["watertight"] for m in res.stats["validation"]["meshes"])


def test_repeated_photo_does_not_start_a_photo_set(egg_project):
    paths = [str(egg_project / "source" / f"image_{i}.jpg") for i in (1, 2)]
    spec = pipeline.analyse_project(str(egg_project), paths)
    assert spec["strategy"] == "inflate", spec["notes"]
    assert spec["images"] == ["image_1.jpg"]
    assert any("same view" in n for n in spec["notes"])


def test_wrap_projection_colours_every_texel(egg_project):
    spec = pipeline.analyse_project(str(egg_project), [str(egg_project / "source" / "image_1.jpg")])
    spec, arrays = pipeline.load_spec(str(egg_project))
    sources = pipeline._light_sources(str(egg_project), spec)
    proj = rc.photo_projection_for(spec, sources, None, arrays)
    assert proj.wrap and proj.relief is not None
    x0, y0, x1, y1 = spec["bbox"]
    inside = np.zeros(sources[0].image.rgb.shape[:2], bool)
    inside[y0:y1 + 1, x0:x1 + 1] = np.asarray(arrays["mask"]) > 0
    background = sources[0].image.rgb[5, 5].astype(np.float64)
    extended = sculpt.extend_colours(sources[0].image.rgb, inside).astype(np.float64)
    rim = inside & ~(cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    before = np.linalg.norm(sources[0].image.rgb[rim].astype(np.float64) - background, axis=1)
    after = np.linalg.norm(extended[rim] - background, axis=1)
    assert np.mean(after < 25) < 0.5 * np.mean(before < 25) + 0.02, "outline pixels must not keep the backdrop colour"
    assert np.abs(proj.relief).max() > 0
