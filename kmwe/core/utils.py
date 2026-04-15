from __future__ import annotations

import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Iterable


MARKER_FILES = ("pyproject.toml",)
MARKER_DIRS = (".git",)
MARKER_RELATIVE_FILES = ("config/default.yaml",)


def is_colab_env() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


def find_upwards(start_dir: Path, rel_path: str) -> Path | None:
    current = start_dir.resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / rel_path
        if candidate.exists():
            return candidate
    return None


def find_project_root_from_default(start_dir: Path) -> Path:
    default_path = find_upwards(start_dir, "config/default.yaml")
    if default_path is None:
        raise FileNotFoundError(
            "config/default.yaml를 찾을 수 없습니다. "
            f"탐색 시작 디렉터리: {start_dir}"
        )
    return default_path.parent.parent.resolve()


def resolve_project_root_auto(start_dir: Path) -> Path:
    current = start_dir.resolve()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in MARKER_FILES):
            return parent
        if any((parent / marker).is_dir() for marker in MARKER_DIRS):
            return parent
        if any((parent / marker).exists() for marker in MARKER_RELATIVE_FILES):
            return parent
    import warnings

    warnings.warn(
        "project_root='__AUTO__' 해석 실패: 마커를 찾지 못했습니다. "
        f"탐색 시작점({current})을 project_root로 사용합니다."
    )
    return current


def ensure_absolute(path_value: str, base_dir: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{ts}_{rand}"


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def flatten_list(items: Iterable[Iterable[str]] | None) -> list[str]:
    if not items:
        return []
    result: list[str] = []
    for chunk in items:
        if chunk:
            result.extend(chunk)
    return result
