"""Generate the sample assets and measure them for docs/TEST_REPORT.md.

    python tools/quality_report.py [--out samples] [--blender /path/to/python-with-bpy] [--rbxcheck /path/to/rbxcheck]

Every case runs in its own worker process (the same entry point the bot uses),
so the reported time and peak memory are what a Discord job costs. For each
result the script also measures silhouette agreement between the delivered
model and the photo, re-imports the FBX in Blender when available and decodes
the RBXM with rbx-dom when available.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import render  # noqa: E402
from engine.fitting import View, scaled_view  # noqa: E402
from engine.procedural import LOD, get_template  # noqa: E402
from exporters import blender, convert  # noqa: E402

CASES = [
    {"name": "Bed", "images": ["bed_photo.jpg"], "category": "auto", "triangles": 5000, "texture": 1024, "note": "real photo supplied by the owner"},
    {"name": "Chair", "images": ["chair_render.png"], "category": "auto", "triangles": 4000, "texture": 1024, "note": "synthetic Cycles render"},
    {"name": "Table", "images": ["table_render.png"], "category": "auto", "triangles": 2000, "texture": 512, "note": "synthetic Cycles render"},
    {"name": "Vase", "images": ["vase_render.png"], "category": "revolve", "triangles": 3000, "texture": 512, "note": "synthetic Cycles render"},
    {"name": "Plush", "images": ["plush_render.png"], "category": "inflate", "triangles": 5000, "texture": 512, "note": "synthetic render, PNG with alpha"},
    {"name": "Mug", "images": ["mug_front.png", "mug_side.png", "mug_back.png"], "views": ["front", "side", "back"], "mode": "multi", "triangles": 4000, "texture": 512, "note": "three synthetic orthographic views"},
]


def run_worker(project_dir: Path, case: dict) -> dict:
    from engine.settings import BuildSettings

    version_dir = project_dir / "versions" / "v0001"
    version_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "create", "project_id": "sample00", "project_dir": str(project_dir), "version_dir": str(version_dir), "version": 1,
        "title": case["name"], "images": [f"image_{i + 1}{Path(n).suffix}" for i, n in enumerate(case["images"])],
        "mode": case.get("mode", "auto"), "category": case.get("category", "auto"), "view_labels": case.get("views"),
        "settings": BuildSettings(triangle_budget=case["triangles"], texture_resolution=case["texture"], quality="standard").to_dict(),
    }
    job = project_dir / "job.json"
    job.write_text(json.dumps(payload))
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "workers.processor", str(job)], cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
        elif line.startswith("ERROR "):
            raise RuntimeError(line[6:])
    if result is None:
        raise RuntimeError(proc.stderr[-2000:])
    result["wall_seconds"] = round(elapsed, 1)
    result["peak_rss_mb"] = round(max(peak_kb, before) / 1024, 0)
    return result


def silhouette_iou(project_dir: Path, result: dict) -> float:
    """IoU between the delivered model's silhouette and the photo's object mask."""
    spec = json.loads((project_dir / "source" / "spec.json").read_text())
    mask = np.asarray(Image.open(project_dir / "source" / "mask_0.png")) > 127
    h, w = mask.shape
    meshes = convert.meshes_from_fbx(str(Path(result["package_dir"]) / "model.fbx"))
    V = np.concatenate([m.vertices for _, m in meshes])
    offs = np.cumsum([0] + [len(m.vertices) for _, m in meshes])[:-1]
    F = np.concatenate([m.faces + o for (_, m), o in zip(meshes, offs)])
    geo = json.loads((Path(result["package_dir"]) / "geometry" / "geometry_settings.json").read_text())
    V = V - np.asarray(geo["pivot_offset_applied"])
    if spec["strategy"] == "template":
        t = get_template(spec["template"])
        fw, fh = spec["fit_resolution"]
        view = scaled_view(View.from_dict(spec["view"]), w / fw, w, h)
        bounds = t.build(spec["params"], LOD(0.0, fast=True)).bounds()
        from engine.reconstruction import _view_camera

        cam = _view_camera(bounds, view, w, h)
    elif spec["strategy"] in ("revolve", "extrude", "inflate"):
        c = spec["camera"]
        cam = render.Camera(render.look_at((0, 0, -50), (0, 0, 0)), w, h, ortho_scale=c["px_per_stud"], cx=c["origin_px"][0], cy=c["origin_px"][1])
    elif spec["strategy"] == "visual_hull":
        v0 = spec["views"][0]
        x0, y0, x1, y1 = v0["bbox"]
        ppm = (y1 - y0 + 1) / spec["dimensions"][1]
        center = np.array([0.0, spec["dimensions"][1] / 2, 0.0])
        cam = render.Camera(render.look_at(center + np.array([0, 0, -100]), center), w, h, ortho_scale=ppm, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2)
    else:
        return float("nan")
    xy, z, iw = cam.project(V)
    sil = render.rasterize(xy, z, F, w, h, iw).mask
    target = mask if spec["strategy"] != "template" else np.asarray(Image.open(project_dir / "source" / "zones_0.png")) > 0
    inter = np.logical_and(sil, target).sum()
    union = np.logical_or(sil, target).sum()
    return float(inter / max(union, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "samples"))
    ap.add_argument("--work", default=str(ROOT / "data" / "quality_runs"))
    ap.add_argument("--blender", default=os.environ.get("BPY_PYTHON", ""))
    ap.add_argument("--rbxcheck", default=os.environ.get("RBXCHECK", ""))
    args = ap.parse_args()
    out = Path(args.out)
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        proj = work / case["name"]
        (proj / "source").mkdir(parents=True)
        for i, n in enumerate(case["images"]):
            shutil.copy(ROOT / "assets" / "test_images" / n, proj / "source" / f"image_{i + 1}{Path(n).suffix}")
        row = {"case": case["name"], "input": ", ".join(case["images"]), "input_note": case["note"], "budget": case["triangles"]}
        try:
            res = run_worker(proj, case)
        except Exception as exc:
            row.update(status="failed", error=str(exc)[:300])
            rows.append(row)
            print(case["name"], "FAILED", exc, flush=True)
            continue
        st = res["stats"]
        row.update(status="ok", strategy=st["strategy"], category=st.get("category"), confidence=st.get("confidence"),
                   triangles=st["triangles"], vertices=st["vertices"], meshes=st["mesh_count"], native_parts=st["native_parts"],
                   size_studs=st["size_studs"], maps=st["texture_maps"], seconds=res["wall_seconds"], peak_rss_mb=res["peak_rss_mb"],
                   reduction_percent=st["budget_report"]["reduction_percent"], impact=st["budget_report"]["quality_impact"],
                   validation_ok=st["validation"]["ok"], watertight=all(m["watertight"] for m in st["validation"]["meshes"]),
                   warnings=st["validation"]["warnings"], fit=json.loads((proj / "source" / "spec.json").read_text()).get("fit", {}).get("components"))
        try:
            row["silhouette_iou"] = round(silhouette_iou(proj, res), 3)
        except Exception as exc:
            row["silhouette_iou"] = None
            row["iou_error"] = str(exc)[:200]
        pkg = Path(res["package_dir"])
        if args.blender and os.path.exists(args.blender):
            try:
                rep = blender.inspect_with_blender(str(pkg / "model.fbx"), args.blender)
                row["blender_fbx"] = {"objects": len(rep["objects"]), "triangles": sum(o["triangles"] for o in rep["objects"]),
                                      "packed_images": sum(1 for im in rep["images"] if im["packed"]),
                                      "texture_sockets": sorted({l[2] for o in rep["objects"] for l in o["texture_links"]})}
            except Exception as exc:
                row["blender_fbx"] = {"error": str(exc)[:200]}
        if args.rbxcheck and os.path.exists(args.rbxcheck):
            p = subprocess.run([args.rbxcheck, str(pkg / "model.rbxm")], capture_output=True, text=True)
            q = subprocess.run([args.rbxcheck, str(pkg / "model.rbxmx")], capture_output=True, text=True)
            row["rbxm_check"] = {"ok": p.returncode == 0 and q.returncode == 0, "parts": p.stdout.count("Part '"), "binary_equals_xml": p.stdout == q.stdout}
        dest = out / case["name"]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pkg, dest)
        shutil.copy(res["zip"], out / Path(res["zip"]).name)
        rows.append(row)
        print(case["name"], row["status"], row.get("strategy"), row.get("triangles"), f"{row.get('seconds')}s", f"IoU {row.get('silhouette_iou')}", flush=True)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "quality_report.json").write_text(json.dumps(rows, indent=2, default=str))
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
