"""End-to-end generation: reconstruction spec -> optimised, textured, exported package.

``analyse_project`` runs the expensive image analysis once and stores the spec in
the project's ``source`` folder. ``build_version`` turns a spec plus build
settings into a complete deliverable package. Edits (triangle budget, size,
materials, textures, pivot ...) only call ``build_version`` again, so the photo is
never re-analysed for an edit.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import analyzer as an
from . import geometry as g
from . import optimization as opt
from . import reconstruction as rc
from . import textures as tx
from .materials import MaterialSpec, make_material, parse_hex_color
from .procedural import LOD, get_template, scale_params_to_dims
from .scene import Scene
from .settings import ROBLOX_MESH_TRIANGLE_LIMIT, BuildSettings
from .validation import validate_scene

Progress = Optional[Callable[[str, float], None]]

STRATEGY_LABELS = {
    "template": "Mode C - procedural object reconstruction (fitted template)",
    "revolve": "Mode A - single image, surface of revolution",
    "extrude": "Mode A - single image, extruded outline",
    "inflate": "Mode A - single image, inflated silhouette",
    "visual_hull": "Mode B - multiple labelled views, visual hull",
    "turntable": "Mode B - photo set, recovered camera yaw + visual hull",
}
QUALITY = {
    "draft": {"q_cap": 0.35, "ao_dirs": 24, "fit_budget": 220, "detail": False},
    "standard": {"q_cap": 0.8, "ao_dirs": 48, "fit_budget": 420, "detail": True},
    "high": {"q_cap": 1.0, "ao_dirs": 96, "fit_budget": 700, "detail": True},
}


@dataclass
class VersionResult:
    package_dir: str
    zip_path: str
    files: Dict[str, str]
    stats: dict
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


def _staged(progress: Progress, lo: float, hi: float) -> Progress:
    if progress is None:
        return None
    return lambda msg, f: progress(msg, lo + (hi - lo) * float(np.clip(f, 0, 1)))


def analyse_project(project_dir: str, image_paths: Sequence[str], mode: str = "auto", category: str = "auto",
                    view_labels: Optional[Sequence[str]] = None, target_height: Optional[float] = None,
                    sensitivity: float = 0.5, progress: Progress = None) -> dict:
    """Run image analysis + reconstruction and persist the editable spec."""
    source = os.path.join(project_dir, "source")
    os.makedirs(source, exist_ok=True)
    rec, sources = rc.reconstruct(image_paths, source, mode, category, view_labels, target_height, sensitivity, progress)
    for i, s in enumerate(sources):
        Image.fromarray((s.seg.mask * 255).astype(np.uint8)).save(os.path.join(source, f"mask_{i}.png"))
        Image.fromarray((s.seg.zones * 120).astype(np.uint8)).save(os.path.join(source, f"zones_{i}.png"))
    rec.spec["images"] = [os.path.basename(p) for p in image_paths]
    rec.spec["view_labels"] = [s.view_label for s in sources]
    rc.save_reconstruction(rec, source)
    return rec.spec


def load_spec(project_dir: str) -> Tuple[dict, Dict[str, np.ndarray]]:
    source = os.path.join(project_dir, "source")
    with open(os.path.join(source, "spec.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    return spec, rc.load_arrays(source)


def _light_sources(project_dir: str, spec: dict) -> List[rc.Source]:
    """Reload source photos and stored masks for photo projection (no re-analysis)."""
    source = os.path.join(project_dir, "source")
    out = []
    for i, name in enumerate(spec.get("images", [])):
        img = an.load_image(os.path.join(source, name))
        mpath = os.path.join(source, f"mask_{i}.png")
        mask = np.asarray(Image.open(mpath)) > 127 if os.path.exists(mpath) else np.ones(img.rgb.shape[:2], bool)
        seg = an.Segmentation(mask, an.fill_holes(mask), np.where(mask, 2, 0).astype(np.uint8), 1.0, "stored", np.zeros((1, 3)))
        feats = an.compute_features(img, seg)
        labels = spec.get("view_labels") or []
        out.append(rc.Source(os.path.join(source, name), img, seg, feats, labels[i] if i < len(labels) else "front"))
    return out


def _apply_material_overrides(materials: Dict[str, MaterialSpec], overrides: Dict[str, dict]) -> Dict[str, MaterialSpec]:
    out = {}
    for key, mat in materials.items():
        o = overrides.get(key)
        if not o:
            out[key] = mat
            continue
        d = mat.to_dict()
        if "kind" in o and o["kind"] != d["kind"]:
            base = make_material(key, o["kind"], tuple(d["base_color"]), name=d["name"], quilted=d["quilted"])
            d.update({k: v for k, v in base.to_dict().items() if k in ("roughness", "metallic", "pattern_scale", "normal_strength", "color_variation", "kind")})
            d["exemplar"] = None
        if "color" in o:
            c = parse_hex_color(str(o["color"])) if isinstance(o["color"], str) else tuple(o["color"])
            if c:
                d["base_color"] = c
                d["exemplar"] = None if d.get("kind") != "wood" else d.get("exemplar")
        for k in ("roughness", "metallic", "emissive", "opacity"):
            if k in o:
                d[k] = float(o[k])
        d["source"] = "edited by user"
        out[key] = MaterialSpec.from_dict(d)
    return out


def _scene_builder(spec: dict, arrays: Dict[str, np.ndarray], settings: BuildSettings):
    """Return a builder ``build(lod_args) -> Scene``; size and material edits are applied here."""
    bevel_scale = 0.0 if settings.low_poly else settings.bevel
    detail = QUALITY[settings.quality]["detail"] and not settings.low_poly
    if spec["strategy"] == "template":
        template = get_template(spec["template"])
        params = dict(spec["params"])
        params.update({k: float(v) for k, v in settings.param_overrides.items() if k in template.params})
        if settings.dimensions:
            params = scale_params_to_dims(template, params, settings.dimensions)
        if settings.scale != 1.0:
            params = scale_params_to_dims(template, params, _scaled_extent(template, params, settings.scale))
        base_mats = {k: MaterialSpec.from_dict(v) for k, v in spec["materials"].items()}
        mats = _apply_material_overrides(base_mats, settings.material_overrides)

        def build(lod_args: dict) -> Scene:
            m = {k: MaterialSpec.from_dict(v.to_dict()) for k, v in mats.items()}
            return template.build(params, LOD(bevel_scale=bevel_scale, **lod_args), m, with_detail=detail)

        return build
    base_mats = {k: MaterialSpec.from_dict(v) for k, v in spec.get("materials", {}).items()}
    spec_local = dict(spec)

    def build(lod_args: dict) -> Scene:
        sc = rc.build_generic_scene(spec_local, arrays, LOD(lod_args.get("q", 0.6), bevel_scale=bevel_scale))
        sc.materials = _apply_material_overrides({**sc.materials, **base_mats}, settings.material_overrides)
        lo, hi = sc.bounds()
        ext = hi - lo
        factors = np.ones(3) * settings.scale
        if settings.dimensions:
            dims = settings.dimensions
            known = [d / e for d, e in zip(dims, ext) if d]
            uniform = float(np.mean(known)) if known else 1.0
            factors = np.array([(d / e) if d else uniform for d, e in zip(dims, ext)]) * settings.scale
        if not np.allclose(factors, 1.0):
            opt.scale_scene(sc, factors)
        return sc

    return build


def _scaled_extent(template, params, scale: float):
    lo, hi = template.build(params, LOD(0.0, fast=True)).bounds()
    return list((hi - lo) * scale)


def build_version(project_dir: str, version_dir: str, spec: dict, arrays: Dict[str, np.ndarray], settings: BuildSettings,
                  title: str, project_id: str, version: int, progress: Progress = None) -> VersionResult:
    from exporters import fbx as fbx_writer
    from exporters import interchange, packaging, previews, rbxm

    t0 = time.time()
    timings: Dict[str, float] = {}
    settings = settings.validated()
    q_cap = QUALITY[settings.quality]["q_cap"]
    report = _staged(progress, 0.0, 1.0)
    if report:
        report("building geometry", 0.02)
    build = _scene_builder(spec, arrays, settings)
    budget = int(settings.triangle_budget)
    reference = build({"q": q_cap})
    original_tris, original_verts = reference.triangle_count, reference.vertex_count
    scene, lod_used = opt.solve_lod(build, budget, q_cap)
    q = float(lod_used["q"])
    notes: List[str] = []
    method = "procedural level of detail" if spec["strategy"] == "template" else "level of detail"
    if scene.triangle_count > budget:
        scene, dec_notes = opt.enforce_budget(scene, budget, settings.smooth_angle)
        notes.extend(dec_notes)
        method = "quadric edge-collapse decimation" if q == 0.0 else method
    min_safe = build({"q": 0.0}).triangle_count if spec["strategy"] == "template" else 0
    reachable = scene.triangle_count <= budget * 1.05
    if not reachable:
        notes.append(f"The {budget}-triangle budget is below what this shape needs; the closest safe result has {scene.triangle_count} triangles. Try 'Low-poly version' or a part-based RBXM for very low budgets.")
    if settings.shading == "flat" or settings.low_poly:
        opt.apply_shading(scene, "flat", 0.0)
    timings["geometry"] = time.time() - t0
    if report:
        report("optimised geometry", 0.18)
    lo_before, hi_before = scene.bounds()
    offset = opt.place_pivot(scene, settings.pivot)
    if scene.triangle_count < original_tris:
        impact = opt.quality_impact([p.mesh for p in reference.parts], [p.mesh for p in scene.parts])
    else:
        impact = {"mean_deviation_percent": 0.0, "p95_deviation_percent": 0.0, "rating": "none (full detail kept)"}
    del reference
    budget_report = opt.BudgetReport(budget, scene.triangle_count, original_tris, original_verts, scene.vertex_count, method, q, reachable, min_safe, impact, [])

    t1 = time.time()
    photo = None
    if spec.get("photo_projection"):
        sources = _light_sources(project_dir, spec)
        photo = rc.photo_projection_for(spec, sources, scene)
        if photo is not None:
            base = _base_extent(spec, arrays)
            final = hi_before - lo_before
            factors = np.where(base > 1e-9, final / np.maximum(base, 1e-9), 1.0)
            pre = g.scaling(1.0 / factors) @ g.translation(-offset)
            for view in [photo] + list(photo.extra):
                view.pre = pre
    style = settings.texture_style
    ao_dirs = QUALITY[settings.quality]["ao_dirs"]
    tex_set = tx.bake_scene(scene, tx.BakeOptions(resolution=settings.texture_resolution, ao_directions=ao_dirs, ao_in_color=settings.ao_in_color,
                                                  detail_normals=QUALITY[settings.quality]["detail"], texture_style=style), photo, _staged(progress, 0.2, 0.62))
    timings["textures"] = time.time() - t1
    notes.extend(tex_set.notes)

    validation = validate_scene(scene, budget, tex_set.maps)
    pkg_name = packaging.safe_name(title)
    package_dir = os.path.join(version_dir, pkg_name)
    if os.path.isdir(package_dir):
        shutil.rmtree(package_dir)
    for sub in ("textures", "materials", "geometry", "previews", "source", "roblox"):
        os.makedirs(os.path.join(package_dir, sub), exist_ok=True)
    tex_paths = tex_set.save(os.path.join(package_dir, "textures"))

    if settings.mesh_grouping == "parts":
        groups: Dict[str, List[g.Mesh]] = {}
        for p in scene.parts:
            groups.setdefault(packaging.safe_name(p.group or p.name, "Part"), []).append(p.mesh)
        export_meshes = [(f"{pkg_name}_{k}", g.merge(v, k)) for k, v in groups.items()]
    else:
        export_meshes = [(pkg_name, scene.merged())]
    export_meshes = opt.split_for_roblox(export_meshes, ROBLOX_MESH_TRIANGLE_LIMIT)
    for _, m in export_meshes:
        m.face_material = None
    if report:
        report("exporting FBX, GLB and Roblox files", 0.66)
    t2 = time.time()

    def png_bytes(key):
        with open(tex_paths[key], "rb") as fh:
            return fh.read()

    mat_name = f"{pkg_name}_Atlas"
    fbx_mat = fbx_writer.FbxMaterial(mat_name, {k: (f"textures/{k}.png", png_bytes(k)) for k in tex_paths if k in fbx_writer.TEXTURE_SLOTS},
                                     roughness=float(np.mean([m.roughness for m in scene.materials.values()])), metallic=float(max([m.metallic for m in scene.materials.values()] + [0.0])))
    files: Dict[str, str] = {}
    fbx_path = os.path.join(package_dir, "model.fbx")
    fbx_writer.write_fbx(fbx_path, [fbx_writer.FbxMesh(n, m.vertices, m.faces, m.normals, m.uvs) for n, m in export_meshes], fbx_mat)
    files["fbx"] = fbx_path
    glb_path = os.path.join(package_dir, "model.glb")
    interchange.write_glb(glb_path, export_meshes, tex_set.maps, mat_name)
    files["glb"] = glb_path
    native = rbxm.build_part_model(scene, pkg_name, offset=(0.0, 0.0, 0.0), anchored=True)
    if native.children:
        files["rbxm"] = os.path.join(package_dir, "model.rbxm")
        files["rbxmx"] = os.path.join(package_dir, "model.rbxmx")
        rbxm.write_rbxm(files["rbxm"], native)
        rbxm.write_rbxmx(files["rbxmx"], native)
    if settings.collision != "none":
        coll = _collision_proxy(scene, settings.collision)
        if coll is not None:
            files["collision"] = os.path.join(package_dir, "roblox", "collision.obj")
            interchange.write_obj(files["collision"], [("Collision", coll)], {}, mtl_name="collision.mtl", material_name="Collision")
    timings["export"] = time.time() - t2

    lo, hi = scene.bounds()
    size = (hi - lo).tolist()
    stats = {
        "title": title, "project_id": project_id, "version": version,
        "strategy": spec["strategy"], "strategy_label": STRATEGY_LABELS.get(spec["strategy"], spec["strategy"]),
        "category": spec.get("category"), "confidence": float(spec.get("confidence", 0.5)),
        "triangles": scene.triangle_count, "vertices": scene.vertex_count, "budget": budget,
        "parts": len(scene.parts), "mesh_count": len(export_meshes), "native_parts": len(native.children),
        "size_studs": [round(s, 3) for s in size], "size_metres": [round(s * 0.28, 3) for s in size],
        "pivot": settings.pivot, "texture_resolution": tex_set.resolution, "texel_density": round(tex_set.density, 2),
        "atlas_utilization": round(tex_set.utilization, 3), "uv_islands": tex_set.island_count,
        "texture_maps": sorted(tex_set.maps.keys()), "materials": len(scene.materials),
        "budget_report": budget_report.to_dict(), "validation": validation.to_dict(),
        "roblox": {"per_mesh_limit": ROBLOX_MESH_TRIANGLE_LIMIT, "meshes_over_limit": 0, "collision_fidelity": packaging.COLLISION_ENUM.get(settings.collision, "Hull"),
                   "recommended_texture": _recommended_texture(size)},
    }
    stats_lines = [
        f"{stats['triangles']:,} tris   {stats['vertices']:,} verts   {len(export_meshes)} mesh(es)",
        f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} studs   atlas {tex_set.resolution}px",
        f"{stats['strategy_label'][:60]}",
    ]
    if report:
        report("rendering previews", 0.74)
    t3 = time.time()
    prev = previews.render_previews(export_meshes, tex_set.maps, os.path.join(package_dir, "previews"), title, stats_lines, turntable=True)
    viewer = previews.write_viewer(os.path.join(package_dir, "previews", "viewer.html"), glb_path, title, stats_lines[0])
    timings["previews"] = time.time() - t3
    for k, v in prev.items():
        files[f"preview_{k}"] = v
    if viewer:
        files["viewer"] = viewer

    source_dir = os.path.join(project_dir, "source")
    for i, name in enumerate(spec.get("images", [])):
        src = os.path.join(source_dir, name)
        if os.path.exists(src):
            dst = os.path.join(package_dir, "source", "original_image" + ("" if i == 0 else f"_{i + 1}") + os.path.splitext(name)[1].lower())
            shutil.copyfile(src, dst)
    overlay = os.path.join(source_dir, "fit_overlay.png")
    if os.path.exists(overlay):
        shutil.copyfile(overlay, os.path.join(package_dir, "source", "template_fit_overlay.png"))
    packaging.write_json(os.path.join(package_dir, "source", "generation_parameters.json"), {
        "project_id": project_id, "version": version, "settings": settings.to_dict(),
        "reconstruction": {k: v for k, v in spec.items() if k not in ("features",)}, "features": spec.get("features"),
    })
    mats_dict = {k: {**m.to_dict(), "roblox_material": m.roblox_material} for k, m in scene.materials.items()}
    packaging.write_json(os.path.join(package_dir, "materials", "materials.json"), {
        "atlas": {"resolution": tex_set.resolution, "texel_density_px_per_stud": round(tex_set.density, 3), "maps": {k: f"textures/{k}.png" for k in tex_set.maps}},
        "materials": mats_dict,
    })
    with open(os.path.join(package_dir, "materials", "material_info.md"), "w", encoding="utf-8") as fh:
        fh.write(packaging.material_info_markdown(title, mats_dict, sorted(tex_set.maps), tex_set.density, tex_set.resolution))
    packaging.write_json(os.path.join(package_dir, "geometry", "mesh_statistics.json"), {
        "triangles": stats["triangles"], "vertices": stats["vertices"], "meshes": [{"name": n, "triangles": m.triangle_count, "vertices": m.vertex_count} for n, m in export_meshes],
        "parts": [{"name": p.name, "group": p.group, "material": p.material, "triangles": p.mesh.triangle_count} for p in scene.parts],
        "budget": budget_report.to_dict(), "validation": validation.to_dict(),
    })
    packaging.write_json(os.path.join(package_dir, "geometry", "geometry_settings.json"), {
        "units": "studs (1 stud = 0.28 m)", "up_axis": "+Y", "front": "-Z", "pivot": settings.pivot, "pivot_offset_applied": offset.tolist(),
        "lod_q": round(q, 3), "shading": "flat" if (settings.shading == "flat" or settings.low_poly) else "smooth",
        "smooth_angle_degrees": settings.smooth_angle, "bevel_intensity": settings.bevel, "mesh_grouping": settings.mesh_grouping,
        "template_params": spec.get("params"), "param_overrides": settings.param_overrides,
    })
    packaging.write_json(os.path.join(package_dir, "geometry", "mesh_metadata.json"), {
        "bounds_studs": {"min": lo.tolist(), "max": hi.tolist()}, "size_studs": size,
        "fbx": {"version": 7400, "axes": "Y up, -Z front (Blender: Forward Z, Up Y)", "unit_scale_factor": 100, "embedded_textures": sorted(fbx_mat.textures)},
        "native_part_model": {"instances": len(native.children), "classes": ["Model", "Part"]},
    })
    packaging.write_json(os.path.join(package_dir, "roblox", "roblox_import_settings.json"), {
        "importer": {"Scale Unit": "Studs", "Anchored": True, "Set Pivot to Scene Origin": True, "Merge Meshes": settings.mesh_grouping == "merged"},
        "meshpart": {"CollisionFidelity": packaging.COLLISION_ENUM.get(settings.collision, "Hull"), "RenderFidelity": "Automatic", "CanCollide": settings.collision != "none"},
        "surface_appearance": {"ColorMap": "textures/basecolor.png", "NormalMap": "textures/normal.png" if "normal" in tex_set.maps else None,
                               "RoughnessMap": "textures/roughness.png", "MetalnessMap": "textures/metallic.png" if "metallic" in tex_set.maps else None,
                               "EmissiveMaskContent": "textures/emissive.png" if "emissive" in tex_set.maps else None,
                               "AlphaMode": "Transparency" if "opacity" in tex_set.maps else "Overlay"},
        "triangles_per_mesh": [m.triangle_count for _, m in export_meshes], "per_mesh_limit": ROBLOX_MESH_TRIANGLE_LIMIT,
    })
    with open(os.path.join(package_dir, "roblox", "apply_import_settings.luau"), "w", encoding="utf-8") as fh:
        fh.write(packaging.luau_import_helper(pkg_name, settings.collision))
    rel_files = sorted(os.path.relpath(os.path.join(dp, f), package_dir).replace(os.sep, "/") for dp, _, fs in os.walk(package_dir) for f in fs)
    readme_info = dict(stats)
    readme_info.update(size=size, files=rel_files + ["README.md"], notes=spec.get("notes", []) + notes, warnings=validation.warnings, collision=packaging.COLLISION_ENUM.get(settings.collision, "Hull"))
    with open(os.path.join(package_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(packaging.package_readme(readme_info))
    zip_path = os.path.join(version_dir, f"{pkg_name}_v{version}.zip")
    packaging.zip_directory(package_dir, zip_path, pkg_name)
    files["zip"] = zip_path
    files["readme"] = os.path.join(package_dir, "README.md")
    files.update({f"texture_{k}": v for k, v in tex_paths.items()})
    timings["total"] = time.time() - t0
    stats["timings_seconds"] = {k: round(v, 2) for k, v in timings.items()}
    stats["notes"] = notes
    packaging.write_json(os.path.join(version_dir, "stats.json"), stats)
    if report:
        report("package ready", 1.0)
    return VersionResult(package_dir, zip_path, files, stats, validation.warnings, notes, timings)


def _recommended_texture(size) -> int:
    m = max(size[0], size[2], size[1])
    return 256 if m <= 5 else 512 if m <= 10 else 1024


def _base_extent(spec: dict, arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Extent of the unedited reconstruction; projection cameras live in its space."""
    if spec["strategy"] == "template":
        lo, hi = get_template(spec["template"]).build(spec["params"], LOD(0.0, fast=True)).bounds()
    else:
        lo, hi = rc.build_generic_scene(spec, arrays, LOD(0.0)).bounds()
    return hi - lo


def _collision_proxy(scene: Scene, mode: str) -> Optional[g.Mesh]:
    import trimesh

    verts = np.concatenate([p.mesh.vertices for p in scene.parts])
    if mode == "box":
        lo, hi = verts.min(0), verts.max(0)
        return g.box(hi - lo, (lo + hi) / 2, name="Collision")
    if mode in ("hull", "default"):
        hull = trimesh.Trimesh(verts).convex_hull
        m = g.Mesh(hull.vertices, hull.faces, None, np.zeros((len(hull.vertices), 2)), None, "Collision")
        m.normals = g.vertex_normals(m)
        return m
    hulls = []
    for p in scene.parts:
        h = trimesh.Trimesh(p.mesh.vertices).convex_hull
        m = g.Mesh(h.vertices, h.faces, None, np.zeros((len(h.vertices), 2)), None, p.name)
        m.normals = g.vertex_normals(m)
        hulls.append(m)
    return g.merge(hulls, "Collision")
