import numpy as np
import pytest

from engine import geometry as g
from engine import procedural as pr


def edge_report(m: g.Mesh):
    key = np.round(m.vertices / 1e-6).astype(np.int64)
    _, pid = np.unique(key, axis=0, return_inverse=True)
    f = pid.reshape(-1)[m.faces]
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, ucount = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    _, dcount = np.unique(directed, axis=0, return_counts=True)
    return int((ucount == 1).sum()), int((ucount > 2).sum()), int((dcount > 1).sum())


def assert_closed_solid(m: g.Mesh):
    open_e, nonman, incons = edge_report(m)
    assert open_e == 0, "mesh has holes"
    assert nonman == 0, "mesh has non-manifold edges"
    assert incons == 0, "winding is inconsistent"
    assert m.signed_volume() > 0, "faces point inward"
    fn = m.face_normals()
    vn = m.normals[m.faces].mean(axis=1)
    assert (np.einsum("ij,ij->i", fn, vn) > 0).mean() > 0.99, "vertex normals disagree with winding"


STAR = np.array([[np.cos(a) * (1 if i % 2 == 0 else 0.45), np.sin(a) * (1 if i % 2 == 0 else 0.45)] for i, a in enumerate(np.linspace(0, 2 * np.pi, 11)[:-1])])


@pytest.mark.parametrize("mesh", [
    g.box((2, 1, 3)),
    g.box((2, 1, 3), bevel=0.1, bevel_segments=3, grain_axis=2),
    g.box((1, 1, 1), bevel=0.2, bevel_segments=2, flat_divisions=4),
    g.cylinder(0.5, 2, 16),
    g.cylinder(0.5, 2, 24, radius_top=0.1),
    g.cylinder(0.5, 2, 16, bevel=0.1, bevel_segments=3),
    g.sphere(1, 20, 12),
    g.torus(1, 0.25, 24, 12),
    g.lathe([(0, 0), (0.6, 0), (0.7, 0.3), (0.4, 1.2), (0.45, 1.5), (0, 1.5)], 24),
    g.extrude_polygon(STAR, depth=0.3, bevel=0.03),
    g.extrude_polygon(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float), [np.array([[-.4, -.4], [.4, -.4], [.4, .4], [-.4, .4]])], depth=0.5, bevel=0.05),
    g.loft([np.array([[np.cos(a) * r, np.sin(a) * r * 0.6, z] for a in np.linspace(0, 2 * np.pi, 13)[:-1]]) for z, r in [(0, .5), (1, .8), (2, .7), (3, .3)]]),
], ids=["box", "bevel-box", "subdivided-box", "cylinder", "frustum", "rounded-cylinder", "sphere", "torus", "lathe", "bevelled-extrusion", "extrusion-with-hole", "loft"])
def test_primitives_are_closed_solids(mesh):
    assert_closed_solid(mesh)
    assert mesh.uvs is not None and np.isfinite(mesh.uvs).all()


def test_box_volume_and_counts():
    b = g.box((2, 3, 4))
    assert b.triangle_count == 12
    assert abs(b.signed_volume() - 24.0) < 1e-9
    assert g.box((1, 1, 1), bevel=0.1, bevel_segments=2).triangle_count == g.box_triangles(2)


def test_mirror_transform_keeps_outward_winding():
    m = g.box((1, 2, 3), center=(1, 0, 0)).transformed(g.scaling([-1, 1, 1]))
    assert m.signed_volume() > 0
    assert_closed_solid(m)


def test_flat_and_smooth_normals():
    sphere = g.sphere(1, 16, 10)
    flat = g.recompute_normals(sphere, 0.0)
    smooth = g.recompute_normals(sphere, 80.0)
    assert flat.vertex_count > smooth.vertex_count
    fn = flat.face_normals()
    assert np.allclose(flat.normals[flat.faces[:, 0]], fn, atol=1e-6)


@pytest.mark.parametrize("name", sorted(pr.TEMPLATES))
def test_templates_build_closed_parts(name):
    t = pr.get_template(name)
    scene = t.build(t.defaults(), pr.LOD(0.5))
    assert scene.parts, "template produced no parts"
    for part in scene.parts:
        assert_closed_solid(part.mesh)
        assert part.native, f"{part.name} has no Roblox-native primitive"
    counts = [t.build(t.defaults(), pr.LOD(q)).triangle_count for q in (0.0, 0.5, 1.0)]
    assert counts == sorted(counts), "triangle count must grow with level of detail"


def test_template_resizing_hits_target_dimensions():
    t = pr.get_template("table")
    params = pr.scale_params_to_dims(t, t.defaults(), [8.0, 3.0, None])
    lo, hi = t.build(params, pr.LOD(0.0)).bounds()
    ext = hi - lo
    assert abs(ext[0] - 8.0) / 8.0 < 0.06
    assert abs(ext[1] - 3.0) / 3.0 < 0.06


def test_split_for_roblox_respects_meshpart_limit():
    from engine.optimization import split_for_roblox

    m = g.sphere(2.0, segments=160, rings=150)
    assert m.triangle_count > 2 * 20_000
    parts = split_for_roblox([("Body", m)], 20_000)
    assert len(parts) == int(np.ceil(m.triangle_count / 20_000))
    assert all(p.triangle_count <= 20_000 for _, p in parts)
    assert sum(p.triangle_count for _, p in parts) == m.triangle_count
    merged = np.sort(np.concatenate([p.vertices[p.faces].reshape(-1, 9) for _, p in parts]), axis=0)
    assert np.allclose(merged, np.sort(m.vertices[m.faces].reshape(-1, 9), axis=0)), "no triangle may be lost or changed"


def test_pinched_sheets_are_separated():
    from engine.optimization import _topology_defects, split_pinches

    def tet(a, b, c, d):
        return [[a, c, b], [a, b, d], [b, c, d], [a, d, c]]

    verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.4, 1], [0.5, -1, 0], [0.5, -0.4, -1]], float)
    faces = np.array(tet(0, 1, 2, 3) + tet(1, 0, 4, 5))
    m = g.Mesh(verts, faces)
    assert _topology_defects(m.faces) > 0, "two solids sharing an edge"
    v, f = split_pinches(m.vertices, m.faces)
    out = g.Mesh(v, f)
    out.normals = g.vertex_normals(out)
    assert _topology_defects(out.faces) == 0
    assert edge_report(out) == (0, 0, 0)
    assert len(v) == 8
