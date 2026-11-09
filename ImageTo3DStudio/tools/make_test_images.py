"""Render photo-like test images with Blender Cycles (run with Blender's Python).

    python -m pip install bpy==5.2.3    (or use a Blender install)
    python tools/make_test_images.py <output_dir>

The objects are modelled here independently of the engine's templates (turned
legs, curved backrest, metaball plush ...) so the tests do not grade the engine
on its own geometry. Every image is a synthetic render and is labelled as such
in assets/test_images/SOURCES.md.
"""
from __future__ import annotations

import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.82, 0.85, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    return scene


def material(name, color, rough=0.5, metal=0.0, wood=False, noise_scale=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if wood or noise_scale:
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.inputs["Scale"].default_value = 3.0 if wood else noise_scale
        wave.inputs["Distortion"].default_value = 6.0
        wave.inputs["Detail"].default_value = 4.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*[c * 0.72 for c in color], 1)
        ramp.color_ramp.elements[1].color = (*color, 1)
        nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def box(name, size, loc, mat, bevel=0.01):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


def turned_leg(name, x, y, h, r, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=r, depth=h, location=(x, y, h / 2))
    ob = bpy.context.active_object
    ob.name = name
    for v in ob.data.vertices:
        t = (v.co.z + h / 2) / h
        k = 1.0 - 0.35 * t + 0.12 * math.sin(t * math.pi * 3)
        v.co.x *= k
        v.co.y *= k
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


def room(floor_color=(0.55, 0.42, 0.3), wall_color=(0.86, 0.85, 0.82)):
    floor = material("floor", floor_color, 0.6, wood=True)
    wall = material("wall", wall_color, 0.9)
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    bpy.context.active_object.data.materials.append(floor)
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 3.0, 5), rotation=(math.pi / 2, 0, 0))
    bpy.context.active_object.data.materials.append(wall)


def studio_backdrop(color=(0.92, 0.92, 0.92)):
    mat = material("backdrop", color, 0.9)
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    bpy.context.active_object.data.materials.append(mat)
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 8, 8), rotation=(math.pi / 2, 0, 0))
    bpy.context.active_object.data.materials.append(mat)


def light_and_camera(target, distance, azimuth_deg, elevation_deg, lens=40, ortho=None):
    scene = bpy.context.scene
    bpy.ops.object.light_add(type="AREA", location=(target[0] - 2.5, target[1] - 3.0, target[2] + 4.0))
    key = bpy.context.active_object
    key.data.energy = 900
    key.data.size = 3.0
    key.rotation_euler = (math.radians(40), 0, math.radians(-35))
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    loc = Vector(target) + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * distance
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    direction = Vector(target) - loc
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho
    else:
        cam.data.lens = lens
    scene.camera = cam
    return cam


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("rendered", path, flush=True)


def build_chair():
    wood = material("walnut", (0.42, 0.27, 0.16), 0.45, wood=True)
    seat = material("seat", (0.62, 0.15, 0.12), 0.85)
    w, d, sh = 0.46, 0.44, 0.46
    for sx in (-1, 1):
        for sy in (-1, 1):
            turned_leg(f"leg{sx}{sy}", sx * (w / 2 - 0.03), sy * (d / 2 - 0.03), sh - 0.05, 0.022, wood)
    box("seat", (w + 0.02, d + 0.02, 0.06), (0, 0, sh - 0.02), seat, 0.02)
    for sx in (-1, 1):
        box(f"post{sx}", (0.035, 0.035, 0.5), (sx * (w / 2 - 0.03), d / 2 - 0.03, sh + 0.25), wood, 0.008)
    box("toprail", (w - 0.02, 0.03, 0.09), (0, d / 2 - 0.02, sh + 0.45), wood, 0.01)
    for i in range(3):
        box(f"slat{i}", (0.03, 0.02, 0.3), ((i - 1) * 0.11, d / 2 - 0.03, sh + 0.22), wood, 0.006)


def build_table():
    oak = material("oak", (0.72, 0.52, 0.32), 0.5, wood=True)
    box("top", (1.4, 0.8, 0.04), (0, 0, 0.73), oak, 0.012)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"leg{sx}{sy}", (0.06, 0.06, 0.71), (sx * 0.62, sy * 0.32, 0.355), oak, 0.006)
    for sy in (-1, 1):
        box(f"apron{sy}", (1.18, 0.02, 0.08), (0, sy * 0.33, 0.67), oak, 0.004)


def build_vase():
    glaze = material("glaze", (0.12, 0.32, 0.55), 0.18)
    prof = [(0.0, 0.0), (0.09, 0.0), (0.12, 0.05), (0.15, 0.16), (0.13, 0.28), (0.07, 0.36), (0.065, 0.42), (0.085, 0.45), (0.0, 0.45)]
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0, z)) for r, z in prof]
    for a, b in zip(verts[:-1], verts[1:]):
        bm.edges.new((a, b))
    bmesh.ops.spin(bm, geom=bm.verts[:] + bm.edges[:], angle=math.tau, steps=64, axis=(0, 0, 1), cent=(0, 0, 0))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    me = bpy.data.meshes.new("vase")
    bm.to_mesh(me)
    ob = bpy.data.objects.new("vase", me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(glaze)
    for p in me.polygons:
        p.use_smooth = True


def build_plush():
    fur = material("fur", (0.86, 0.55, 0.18), 0.95, noise_scale=40.0)
    blobs = [((0, 0, 0.28), (0.24, 0.2, 0.26)), ((0, 0, 0.58), (0.2, 0.18, 0.18)), ((-0.12, 0, 0.75), (0.06, 0.05, 0.1)),
             ((0.12, 0, 0.75), (0.06, 0.05, 0.1)), ((-0.24, 0, 0.32), (0.07, 0.07, 0.13)), ((0.24, 0, 0.32), (0.07, 0.07, 0.13)),
             ((-0.11, 0, 0.06), (0.09, 0.09, 0.08)), ((0.11, 0, 0.06), (0.09, 0.09, 0.08))]
    parts = []
    for loc, scale in blobs:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=loc)
        ob = bpy.context.active_object
        ob.scale = scale
        parts.append(ob)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in parts:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    body = bpy.context.active_object
    bpy.ops.object.transform_apply(scale=True)
    rem = body.modifiers.new("remesh", "REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = 0.012
    sm = body.modifiers.new("smooth", "SMOOTH")
    sm.iterations = 12
    sm.factor = 0.8
    body.data.materials.append(fur)
    bpy.ops.object.shade_smooth()
    eye = material("eye", (0.02, 0.02, 0.02), 0.2)
    for sx in (-1, 1):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.028, location=(sx * 0.07, -0.17, 0.6))
        bpy.context.active_object.data.materials.append(eye)


def build_mug():
    ceramic = material("ceramic", (0.85, 0.2, 0.18), 0.25)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.05, depth=0.1, location=(0, 0, 0.05))
    bpy.context.active_object.data.materials.append(ceramic)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.03, minor_radius=0.009, location=(0.055, 0, 0.055), rotation=(math.pi / 2, 0, 0))
    bpy.context.active_object.data.materials.append(ceramic)


def main(out, only=None):
    os.makedirs(out, exist_ok=True)
    if only == "plush":
        reset()
        bpy.context.scene.render.film_transparent = True
        build_plush(); light_and_camera((0, 0, 0.4), 2.2, 0, 4, 60); render(os.path.join(out, "plush_render.png"))
        return
    reset(); room(); build_chair(); light_and_camera((0, 0, 0.45), 2.3, 32, 22, 45); render(os.path.join(out, "chair_render.png"))
    reset(); studio_backdrop(); build_table(); light_and_camera((0, 0, 0.4), 3.4, -28, 24, 40); render(os.path.join(out, "table_render.png"))
    reset(); studio_backdrop((0.95, 0.95, 0.95)); build_vase(); light_and_camera((0, 0, 0.22), 1.6, 0, 6, 70); render(os.path.join(out, "vase_render.png"))
    reset()
    bpy.context.scene.render.film_transparent = True
    build_plush(); light_and_camera((0, 0, 0.4), 2.2, 0, 4, 60); render(os.path.join(out, "plush_render.png"))
    for label, az in (("front", 0), ("side", -90), ("back", 180)):
        reset(); studio_backdrop((0.96, 0.96, 0.96)); build_mug(); light_and_camera((0, 0, 0.05), 3, az, 0, ortho=0.22); render(os.path.join(out, f"mug_{label}.png"))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(args[0], args[1] if len(args) > 1 else None)
