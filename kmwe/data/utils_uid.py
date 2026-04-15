from __future__ import annotations

import hashlib
from typing import Any


def build_uid(record: dict[str, Any], *, corpus: str | None = None, text: str | None = None) -> str:
    example_id = record.get("example_id")
    instance_id = record.get("instance_id")
    if example_id is not None and instance_id is not None:
        return f"{example_id}#{instance_id}"
    doc_id = record.get("doc_id")
    sent_index = record.get("sent_index")
    if doc_id is not None and sent_index is not None:
        return f"{doc_id}:{sent_index}"
    base_text = text or record.get("raw_sentence") or record.get("target_sentence") or ""
    base_corpus = corpus or str(record.get("corpus") or "")
    payload = f"{base_corpus}\n{base_text}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:16]
