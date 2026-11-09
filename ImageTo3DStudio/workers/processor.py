"""Worker process: ``python -m workers.processor <job.json>``.

Runs one job in its own process so a crash, a memory blow-up or a timeout can
never take the Discord bot down. Progress is streamed to stdout as single-line
JSON records prefixed with PROGRESS, RESULT or ERROR.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def emit(tag: str, data: dict) -> None:
    sys.stdout.write(f"{tag} {json.dumps(data, default=str)}\n")
    sys.stdout.flush()


def run_job(payload: dict) -> dict:
    from engine import pipeline
    from engine.settings import BuildSettings

    kind = payload["kind"]
    project_dir = payload["project_dir"]
    analysed = False
    t0 = time.time()

    def progress_a(msg: str, frac: float) -> None:
        emit("PROGRESS", {"stage": msg, "progress": round(0.5 * frac, 4)})

    def progress_b(msg: str, frac: float) -> None:
        base, span = (0.5, 0.5) if analysed else (0.0, 1.0)
        emit("PROGRESS", {"stage": msg, "progress": round(base + span * frac, 4)})

    if kind in ("create", "reanalyze"):
        source = os.path.join(project_dir, "source")
        images = [os.path.join(source, n) for n in payload["images"]]
        pipeline.analyse_project(project_dir, images, payload.get("mode", "auto"), payload.get("category", "auto"),
                                 payload.get("view_labels"), payload.get("target_height"), float(payload.get("sensitivity", 0.5)), progress_a)
        analysed = True
    spec, arrays = pipeline.load_spec(project_dir)
    settings = BuildSettings.from_dict(payload.get("settings"))
    res = pipeline.build_version(project_dir, payload["version_dir"], spec, arrays, settings, payload["title"], payload["project_id"], int(payload["version"]), progress_b)
    return {
        "package_dir": res.package_dir, "zip": res.zip_path, "files": res.files, "stats": res.stats,
        "warnings": res.warnings, "notes": res.notes, "strategy": spec.get("strategy"), "category": spec.get("category"),
        "spec_notes": spec.get("notes", []), "confidence": spec.get("confidence"), "seconds": round(time.time() - t0, 2),
    }


def main(argv) -> int:
    if len(argv) < 2:
        emit("ERROR", {"error": "usage: python -m workers.processor <job.json>"})
        return 2
    try:
        with open(argv[1], encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        emit("ERROR", {"error": f"cannot read job file: {exc}"})
        return 2
    try:
        result = run_job(payload)
    except MemoryError:
        emit("ERROR", {"error": "the job ran out of memory; try a smaller image or a lower quality setting", "type": "MemoryError"})
        return 3
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("ERROR", {"error": str(exc) or exc.__class__.__name__, "type": exc.__class__.__name__})
        return 1
    result_path = os.path.join(payload["version_dir"], "result.json")
    try:
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
    except OSError:
        pass
    emit("RESULT", result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
