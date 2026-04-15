from __future__ import annotations

from typing import Any

from kmwe.data.types import BaseTextRecord
from kmwe.data.utils_uid import build_uid


def ingested_record_to_text(
    record: dict[str, Any], *, corpus: str, split: str | None = None
) -> BaseTextRecord:
    text = record.get("raw_sentence") or record.get("target_sentence") or record.get("text") or ""
    meta = {k: v for k, v in record.items() if k not in {"raw_sentence", "target_sentence"}}
    uid = build_uid(record, corpus=corpus, text=text)
    return {
        "uid": uid,
        "corpus": corpus,
        "split": split,
        "text": text,
        "meta": meta,
        "morph_tokens": record.get("morph_tokens"),
        "dep": record.get("dep"),
        "sense": record.get("sense"),
    }
