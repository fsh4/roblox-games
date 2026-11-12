import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_IMAGES = ROOT / "assets" / "test_images"


def image(name: str) -> Path:
    p = TEST_IMAGES / name
    if not p.exists():
        pytest.skip(f"test image {name} not present")
    return p


@pytest.fixture(scope="session")
def bed_project(tmp_path_factory):
    """One analysed bed project shared by the export and pipeline tests."""
    from engine import pipeline

    proj = tmp_path_factory.mktemp("bed_project")
    (proj / "source").mkdir()
    shutil.copy(image("bed_photo.jpg"), proj / "source" / "image_1.jpg")
    spec = pipeline.analyse_project(str(proj), [str(proj / "source" / "image_1.jpg")], category="bed")
    return proj, spec


@pytest.fixture(scope="session")
def bed_version(bed_project):
    from engine import pipeline
    from engine.settings import BuildSettings

    proj, _ = bed_project
    spec, arrays = pipeline.load_spec(str(proj))
    settings = BuildSettings(triangle_budget=4000, texture_resolution=512, quality="draft")
    res = pipeline.build_version(str(proj), str(proj / "versions" / "v0001"), spec, arrays, settings, "Test Bed", "abcd1234", 1)
    return proj, res
