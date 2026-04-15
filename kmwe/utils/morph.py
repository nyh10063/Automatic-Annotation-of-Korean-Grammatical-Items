from __future__ import annotations

from typing import Any
import re

_KIWI_CACHE: dict[str, Any] = {}


def analyze_with_kiwi(sentence: str, *, model: str = "cong-global") -> list[dict[str, Any]]:
    kiwi = _get_kiwi(model)
    if kiwi is None:
        return _fallback_analyze(sentence)
    try:
        analyses = kiwi.analyze(sentence)
    except Exception:
        return _fallback_analyze(sentence)
    if not analyses:
        return []
    tokens = []
    for token in analyses[0][0]:
        start = int(getattr(token, "start", 0))
        length = int(getattr(token, "len", 0))
        tokens.append(
            {
                "surface": str(getattr(token, "form", "")),
                "lemma": str(getattr(token, "lemma", "")),
                "pos": str(getattr(token, "tag", "")),
                "start": start,
                "end": start + length,
            }
        )
    return tokens


def _get_kiwi(model: str) -> Any | None:
    if model in _KIWI_CACHE:
        return _KIWI_CACHE[model]
    try:
        from kiwipiepy import Kiwi
    except Exception:
        _KIWI_CACHE[model] = None
        return None
    try:
        kiwi = Kiwi(model_type=model)
    except Exception:
        _KIWI_CACHE[model] = None
        return None
    _KIWI_CACHE[model] = kiwi
    return kiwi


def _fallback_analyze(sentence: str) -> list[dict[str, Any]]:
    tokens = []
    for match in re.finditer(r"\S+", sentence):
        surface = match.group(0)
        tokens.append(
            {
                "surface": surface,
                "lemma": surface,
                "pos": "UNK",
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens
