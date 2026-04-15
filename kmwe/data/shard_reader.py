from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl_shards(
    shard_paths: Iterable[Path | str], *, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    count = 0
    for path in shard_paths:
        shard_path = Path(path)
        if not shard_path.exists():
            continue
        with shard_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return
