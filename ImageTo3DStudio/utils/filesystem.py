"""Filesystem helpers with path-containment checks."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List


class UnsafePathError(ValueError):
    pass


def safe_join(base: Path, *parts: str) -> Path:
    """Join below ``base`` and refuse anything that escapes it (.., symlinks, absolute)."""
    base_r = Path(base).resolve()
    target = base_r.joinpath(*parts).resolve()
    if target != base_r and base_r not in target.parents:
        raise UnsafePathError(f"path escapes {base_r}")
    return target


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dir_size(path: Path) -> int:
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def remove_tree(path: Path, within: Path) -> None:
    path = Path(path).resolve()
    within = Path(within).resolve()
    if within not in path.parents:
        raise UnsafePathError("refusing to delete outside the data folder")
    shutil.rmtree(path, ignore_errors=True)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def files_by_size(paths: Iterable[Path]) -> List[Path]:
    return sorted((p for p in paths if p.is_file()), key=lambda p: p.stat().st_size)
