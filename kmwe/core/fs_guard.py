from __future__ import annotations

from pathlib import Path


def assert_under_dir(path: Path, allowed_root: Path, what: str) -> None:
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(
            f"{what} 경로가 허용된 범위를 벗어났습니다: {resolved} (allowed_root={allowed})"
        ) from exc
