from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


def dumps_artifact(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def write_json(path: Path, obj: Any, *, indent: int | None = 2) -> None:
    path.write_text(dumps_artifact(obj, indent=indent), encoding="utf-8")


def write_jsonl_line(fp: TextIO, obj: Any) -> None:
    fp.write(dumps_artifact(obj) + "\n")
