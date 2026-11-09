import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine import analyzer as an
from engine import pipeline
from engine.settings import BuildSettings

from .conftest import image

REQUIRED = ["model.fbx", "model.glb", "model.rbxm", "model.rbxmx", "README.md", "textures/basecolor.png", "textures/normal.png",
            "textures/roughness.png", "materials/materials.json", "materials/material_info.md", "geometry/mesh_statistics.json",
            "geometry/geometry_settings.json", "geometry/mesh_metadata.json", "previews/front.png", "previews/side.png", "previews/back.png",
            "previews/three_quarter.png", "previews/preview.png", "previews/turntable.gif", "source/generation_parameters.json",
            "roblox/roblox_import_settings.json", "roblox/apply_import_settings.luau"]


def test_bed_photo_is_reconstructed_as_a_bed(bed_project):
    _, spec = bed_project
    assert spec["strategy"] == "template"
    assert spec["template"] == "bed"
    assert spec["fit"]["score"] < 1.0
    assert spec["materials"]["frame"]["kind"] == "wood"


def test_package_is_complete_and_within_budget(bed_version):
    proj, res = bed_version
    pkg = Path(res.package_dir)
    for rel in REQUIRED:
        assert (pkg / rel).exists(), f"missing {rel}"
    assert res.stats["triangles"] <= 4000
    assert res.stats["validation"]["ok"]
    assert all(m["watertight"] for m in res.stats["validation"]["meshes"])
    assert Path(res.zip_path).stat().st_size > 50_000


def test_edit_reuses_analysis_and_applies_changes(bed_project, bed_version, tmp_path):
    proj, _ = bed_project
    spec_path = proj / "source" / "spec.json"
    before = spec_path.stat().st_mtime
    spec, arrays = pipeline.load_spec(str(proj))
    s = BuildSettings(triangle_budget=1200, texture_resolution=256, quality="draft", dimensions=[5.0, None, None],
                      material_overrides={"frame": {"color": "#3A2A1E", "kind": "wood"}}, pivot="center")
    t0 = time.time()
    res = pipeline.build_version(str(proj), str(tmp_path / "v2"), spec, arrays, s, "Edited Bed", "abcd1234", 2)
    assert spec_path.stat().st_mtime == before, "an edit must not re-run the image analysis"
    assert res.stats["triangles"] <= 1200 * 1.05
    assert abs(res.stats["size_studs"][0] - 5.0) < 0.3
    mats = json.loads((Path(res.package_dir) / "materials" / "materials.json").read_text())["materials"]
    assert mats["frame"]["source"] == "edited by user"
    lo = np.array(json.loads((Path(res.package_dir) / "geometry" / "mesh_metadata.json").read_text())["bounds_studs"]["min"])
    hi = np.array(json.loads((Path(res.package_dir) / "geometry" / "mesh_metadata.json").read_text())["bounds_studs"]["max"])
    assert np.allclose((lo + hi) / 2, 0, atol=1e-6), "centre pivot"
    assert time.time() - t0 < 120


def test_unreachable_budget_is_reported(bed_project, tmp_path):
    proj, _ = bed_project
    spec, arrays = pipeline.load_spec(str(proj))
    res = pipeline.build_version(str(proj), str(tmp_path / "v3"), spec, arrays, BuildSettings(triangle_budget=100, texture_resolution=256, quality="draft"), "Tiny", "abcd1234", 3)
    br = res.stats["budget_report"]
    assert br["final_triangles"] <= 400
    assert any("budget" in n.lower() or "triangles" in n.lower() for n in res.notes) or br["budget_reachable"]


@pytest.mark.parametrize("name,expected", [("vase_render.png", "revolve"), ("plush_render.png", "inflate")])
def test_generic_single_view_strategies(tmp_path, name, expected):
    src = image(name)
    (tmp_path / "source").mkdir()
    shutil.copy(src, tmp_path / "source" / name)
    spec = pipeline.analyse_project(str(tmp_path), [str(tmp_path / "source" / name)], category=expected)
    assert spec["strategy"] == expected
    spec, arrays = pipeline.load_spec(str(tmp_path))
    res = pipeline.build_version(str(tmp_path), str(tmp_path / "v1"), spec, arrays, BuildSettings(triangle_budget=3000, texture_resolution=256, quality="draft"), expected, "abcd1234", 1)
    assert res.stats["triangles"] <= 3000 * 1.05
    assert res.stats["validation"]["ok"]


def test_multi_view_visual_hull(tmp_path):
    names = ["mug_front.png", "mug_side.png", "mug_back.png"]
    srcs = [image(n) for n in names]
    (tmp_path / "source").mkdir()
    for s in srcs:
        shutil.copy(s, tmp_path / "source" / s.name)
    spec = pipeline.analyse_project(str(tmp_path), [str(tmp_path / "source" / n) for n in names], mode="multi", view_labels=["front", "side", "back"])
    assert spec["strategy"] == "visual_hull"
    spec, arrays = pipeline.load_spec(str(tmp_path))
    res = pipeline.build_version(str(tmp_path), str(tmp_path / "v1"), spec, arrays, BuildSettings(triangle_budget=4000, texture_resolution=256, quality="draft"), "Mug", "abcd1234", 1)
    size = res.stats["size_studs"]
    assert size[0] > size[2] * 0.9, "the handle makes the mug wider than deep"


def test_large_image_is_downscaled(tmp_path):
    big = tmp_path / "big.jpg"
    Image.new("RGB", (6000, 4000), (200, 200, 200)).save(big, quality=60)
    img = an.load_image(str(big))
    assert max(img.rgb.shape[:2]) == an.WORK_MAX_SIDE
    assert img.original_size == (6000, 4000)


def test_invalid_image_is_rejected(tmp_path):
    bad = tmp_path / "fake.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nthis is not really a png")
    with pytest.raises(an.ImageValidationError):
        an.load_image(str(bad))
