from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class BaseTextRecord(TypedDict):
    uid: str
    corpus: str
    split: str | None
    text: str
    meta: dict[str, Any]
    morph_tokens: Any | None
    dep: Any | None
    sense: Any | None


@dataclass(frozen=True)
class SpanSupervisionExample:
    uid: str
    text: str
    context_left: str
    context_right: str
    candidate_e_id: str
    span_segments: list[tuple[int, int]]
    label: int
    weight: float = 1.0
    allowed_e_ids: list[str] | None = None
    split: str | None = None
    gold_example_role: str | None = None
    role: str | None = None
    meta: dict[str, Any] | None = None
