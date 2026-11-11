import numpy as np

from engine import procedural as pr
from engine import render
from engine import textures as tx
from engine.materials import make_material


def test_atlas_islands_do_not_overlap():
    scene = pr.get_template("chair").build(pr.get_template("chair").defaults(), pr.LOD(0.4))
    meshes = [p.mesh for p in scene.parts]
    atlas = tx.pack_atlas(meshes, 512, padding=4)
    tx.apply_atlas_uvs(meshes, atlas)
    cover = np.zeros((512, 512), np.int32)
    for m in meshes:
        assert m.uvs.min() >= 0 and m.uvs.max() <= 1
        xy = np.stack([m.uvs[:, 0] * 512, (1 - m.uvs[:, 1]) * 512], axis=1)
        ras = render.rasterize(xy, np.zeros(len(xy)), m.faces, 512, 512)
        cover += ras.mask
    island_cover = np.zeros((512, 512), np.int32)
    for isl in atlas.islands:
        x, y, w, h = isl.rect
        island_cover[y:y + h, x:x + w] += 1
    assert island_cover.max() == 1, "packed island rectangles overlap"
    assert 0.2 < atlas.utilization <= 1.0


def test_bake_produces_meaningful_maps():
    t = pr.get_template("lamp")
    scene = t.build(t.defaults(), pr.LOD(0.4))
    ts = tx.bake_scene(scene, tx.BakeOptions(resolution=256, ao_directions=12))
    assert ts.maps["basecolor"].shape == (256, 256, 3)
    assert ts.maps["roughness"].ndim == 2
    assert "metallic" in ts.maps, "the lamp base is metal"
    assert "emissive" in ts.maps, "the bulb emits light"
    for arr in ts.maps.values():
        assert arr.dtype == np.uint8


def test_metallic_map_skipped_without_metal():
    t = pr.get_template("shelf")
    scene = t.build(t.defaults(), pr.LOD(0.2))
    ts = tx.bake_scene(scene, tx.BakeOptions(resolution=128, ao_directions=8))
    assert "metallic" not in ts.maps
    assert "emissive" not in ts.maps


def test_push_pull_fills_every_pixel():
    img = np.zeros((64, 64, 3))
    mask = np.zeros((64, 64), bool)
    mask[20:30, 20:30] = True
    img[mask] = [0.2, 0.4, 0.6]
    out = tx.push_pull_fill(img, mask)
    assert np.allclose(out[~mask], [0.2, 0.4, 0.6], atol=1e-6)


def test_exemplar_synthesis_keeps_colour_statistics():
    rng = np.random.default_rng(0)
    base = np.array([0.6, 0.4, 0.2])
    ex = np.clip(base + rng.normal(0, 0.05, (64, 96, 3)), 0, 1)
    sampler = tx.Exemplar(ex, px_per_stud=40)
    uv = rng.uniform(0, 10, (20000, 2))
    out = sampler.sample(uv)
    assert np.allclose(out.mean(0), ex.reshape(-1, 3).mean(0), atol=0.02)
    assert abs(out.std() - ex.std()) < 0.02


def test_wood_material_varies_but_keeps_mean():
    mat = make_material("w", "wood", (0.7, 0.5, 0.3))
    local = np.random.default_rng(1).uniform(-1, 1, (5000, 3))
    ch = tx.eval_material(mat, local, local[:, :2], 0, 3)
    assert ch.albedo.std() > 0.01
    assert np.abs(ch.albedo.mean(0) - np.array([0.7, 0.5, 0.3]) ** 2.2).max() < 0.15
