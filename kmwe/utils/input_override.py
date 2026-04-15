from __future__ import annotations

import os
from typing import Any


def get_forced_input_jsonl(cfg: dict[str, Any]) -> str | None:
    forced = os.getenv("KMWE_FORCE_INPUT_JSONL")
    if forced:
        return forced
    return cfg.get("paths", {}).get("force_ingest_corpus_jsonl")


def apply_forced_input_jsonl(
    cfg: dict[str, Any], *, stage: str
) -> tuple[dict[str, Any], str | None, str | None]:
    forced = get_forced_input_jsonl(cfg)
    if not forced:
        return cfg, None, None
    cfg.setdefault("infer", {})
    cfg.setdefault("silver", {})
    cfg["infer"]["input_jsonl"] = forced
    cfg["silver"]["input_jsonl"] = forced
    source = "env" if os.getenv("KMWE_FORCE_INPUT_JSONL") else "cfg.paths.force_ingest_corpus_jsonl"
    return cfg, forced, source
