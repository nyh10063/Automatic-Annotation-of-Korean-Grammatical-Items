from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from kmwe.core.config_loader import ConfigError
from kmwe.stages.build_bgroup_sft import _normalize_row, _validate_row
from kmwe.stages.infer_step2_rerank import _build_marked_sentence


BGROUP_INPUT_MODE = "bgroup_cross_encoder_pair_v1"
BGROUP_SPAN_MARKER_STYLE = "[SPAN]...[/SPAN]"
BGROUP_TEXT_B_FORMAT = "canonical_form_plus_gloss_plain"
BGROUP_CANDIDATE_SCORING = "shared_cross_encoder_local_softmax_ce"


def _normalize_span_segments(span_segments: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not span_segments:
        return out
    for item in span_segments:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            start = int(item[0])
            end = int(item[1])
        except Exception:
            continue
        if end <= start:
            continue
        out.append((start, end))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _inject_span_markers(target_sentence: str, span_segments: Any) -> str:
    spans = _normalize_span_segments(span_segments)
    return _build_marked_sentence(target_sentence, spans)


def _build_bgroup_text_a(marked_sentence: str) -> str:
    return str(marked_sentence or "").strip()


def _strip_canonical_form_suffix(canonical_form: str) -> str:
    text = str(canonical_form or "").strip()
    if not text:
        return ""
    return re.sub(r"\s*[0-9]+\s*$", "", text).strip()


def _build_bgroup_text_b(canonical_form: str, gloss: str) -> str:
    canonical = _strip_canonical_form_suffix(canonical_form)
    gloss_text = str(gloss or "").strip()
    if not canonical:
        raise ConfigError("B-group candidate canonical_form이 비어 있습니다.")
    if gloss_text:
        return canonical + "\n" + gloss_text
    return canonical


def build_bgroup_cross_encoder_input(example: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    marked_sentence = _inject_span_markers(
        str(example.get("target_sentence") or ""),
        example.get("span_segments") or [],
    )
    text_a = _build_bgroup_text_a(marked_sentence)
    text_b = _build_bgroup_text_b(
        str(candidate.get("canonical_form") or ""),
        str(candidate.get("gloss") or ""),
    )
    meta = {
        "candidate_e_id": str(candidate.get("e_id") or "").strip(),
        "canonical_form": _strip_canonical_form_suffix(candidate.get("canonical_form") or ""),
        "gloss": str(candidate.get("gloss") or "").strip(),
        "polyset_id": str(example.get("polyset_id") or "").strip(),
        "group_key": str(example.get("group_key") or "").strip(),
        "span_segments": [[int(s), int(e)] for s, e in _normalize_span_segments(example.get("span_segments") or [])],
    }
    return {"text_a": text_a, "text_b": text_b, "meta": meta}


def _load_expredict_candidate_meta(dict_xlsx: Path, sheet_name: str = "expredict") -> dict[str, dict[str, str]]:
    if not dict_xlsx.exists():
        raise ConfigError(f"dict_xlsx 경로가 존재하지 않습니다: {dict_xlsx}")
    df = pd.read_excel(dict_xlsx, sheet_name=sheet_name, engine="openpyxl")
    meta: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        eid = str(row.get("e_id") or "").strip()
        if not eid:
            continue
        meta[eid] = {
            "canonical_form": str(row.get("canonical_form") or "").strip(),
            "gloss": str(row.get("gloss") or "").strip(),
            "polyset_id": str(row.get("polyset_id") or "").strip(),
            "group": str(row.get("group") or "").strip(),
        }
    return meta


def load_bgroup_cross_encoder_examples(
    *,
    cfg: dict[str, Any],
    logger: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    stage_cfg = cfg.get("bgroup_encoder_ce", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    gold_xlsx = Path(str(paths_cfg.get("gold_b_xlsx") or paths_cfg.get("gold_xlsx") or "")).expanduser()
    dict_xlsx = Path(str(paths_cfg.get("dict_xlsx") or "")).expanduser()
    gold_sheet_name = str(stage_cfg.get("gold_sheet_name") or "gold")
    allow_multiple = bool(stage_cfg.get("allow_multiple", False))
    if allow_multiple:
        raise ConfigError("현재 B-group encoder CE baseline은 allow_multiple=false만 지원합니다.")
    if not gold_xlsx.exists():
        raise ConfigError(f"gold_b_xlsx 경로가 유효하지 않습니다: {gold_xlsx}")
    if not dict_xlsx.exists():
        raise ConfigError(f"dict_xlsx 경로가 유효하지 않습니다: {dict_xlsx}")

    df = pd.read_excel(gold_xlsx, sheet_name=gold_sheet_name, engine="openpyxl")
    rows = [_normalize_row(r) for r in df.to_dict(orient="records")]
    expredict_meta = _load_expredict_candidate_meta(dict_xlsx)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    issue_counts = Counter()
    split_counts_raw = Counter(r.get("split") or "" for r in rows)
    role_counts_raw = Counter(r.get("gold_example_role") or "" for r in rows)
    candidate_count_distribution = Counter()
    sample_rows: list[dict[str, Any]] = []

    for row in rows:
        errors, warnings = _validate_row(row, allow_multiple=False)
        if warnings:
            for w in warnings:
                issue_counts[f"warning:{w}"] += 1
        if errors:
            for e in errors:
                issue_counts[f"error:{e}"] += 1
            continue

        split = str(row.get("split") or "").strip().lower()
        if split not in by_split:
            issue_counts[f"error:unsupported_split:{split or '__missing__'}"] += 1
            continue

        candidate_e_ids = [str(x).strip() for x in (row.get("candidate_e_ids") or []) if str(x).strip()]
        effective_gold_e_ids = [str(x).strip() for x in (row.get("effective_gold_e_ids") or []) if str(x).strip()]
        gold_e_id = effective_gold_e_ids[0] if effective_gold_e_ids else "__NONE__"
        label_index = candidate_e_ids.index(gold_e_id) if gold_e_id in candidate_e_ids else len(candidate_e_ids)
        span_segments = _normalize_span_segments(row.get("span_segments_parsed") or [])
        polyset_candidates = []
        missing_meta: list[str] = []
        polyset_ids = []
        groups = []

        example_base = {
            "example_id": str(row.get("example_id") or "").strip(),
            "instance_id": str(row.get("instance_id") or "").strip(),
            "group_key": str(row.get("example_key_full") or "").strip(),
            "target_sentence": str(row.get("target_sentence") or ""),
            "context_left": str(row.get("context_left") or ""),
            "context_right": str(row.get("context_right") or ""),
            "span_segments": [[int(s), int(e)] for s, e in span_segments],
            "gold_example_role": str(row.get("gold_example_role") or "").strip(),
            "pattern_type": str(row.get("pattern_type") or "").strip(),
            "source": str(row.get("source") or "").strip(),
            "note": str(row.get("note") or ""),
        }
        for eid in candidate_e_ids:
            meta = expredict_meta.get(eid) or {}
            canonical_form = str(meta.get("canonical_form") or "").strip()
            gloss = str(meta.get("gloss") or "").strip()
            polyset_id = str(meta.get("polyset_id") or "").strip()
            group = str(meta.get("group") or "").strip()
            if not canonical_form:
                missing_meta.append(eid)
                continue
            polyset_ids.append(polyset_id)
            groups.append(group)
            built = build_bgroup_cross_encoder_input(
                {
                    **example_base,
                    "polyset_id": polyset_id,
                },
                {
                    "e_id": eid,
                    "canonical_form": canonical_form,
                    "gloss": gloss,
                },
            )
            polyset_candidates.append(
                {
                    "candidate_e_id": eid,
                    "canonical_form": _strip_canonical_form_suffix(canonical_form),
                    "gloss": gloss,
                    "polyset_id": polyset_id,
                    "group": group,
                    "text_a": built["text_a"],
                    "text_b": built["text_b"],
                    "meta": built["meta"],
                }
            )
        if missing_meta:
            issue_counts["error:missing_candidate_meta"] += 1
            continue
        if not polyset_candidates:
            issue_counts["error:empty_polyset_candidates"] += 1
            continue
        if label_index > len(polyset_candidates):
            issue_counts["error:invalid_label_index"] += 1
            continue

        polyset_id = next((x for x in polyset_ids if x), "")
        polyset_group = next((x for x in groups if x), "")
        example = {
            **example_base,
            "split": split,
            "polyset_id": polyset_id,
            "polyset_group": polyset_group,
            "candidate_inputs": polyset_candidates,
            "candidate_e_ids": [c["candidate_e_id"] for c in polyset_candidates],
            "gold_e_id": gold_e_id,
            "label_index": int(label_index),
            "decision_type": str(row.get("effective_decision_type") or row.get("decision_type") or "").strip(),
            "has_none_label": True,
            "input_mode": BGROUP_INPUT_MODE,
            "span_marker_style": BGROUP_SPAN_MARKER_STYLE,
            "text_b_format": BGROUP_TEXT_B_FORMAT,
            "candidate_scoring": BGROUP_CANDIDATE_SCORING,
        }
        by_split[split].append(example)
        candidate_count_distribution[len(polyset_candidates)] += 1
        if len(sample_rows) < 5:
            sample_rows.append(
                {
                    "group_key": example["group_key"],
                    "split": split,
                    "role": example["gold_example_role"],
                    "polyset_id": polyset_id,
                    "gold_e_id": gold_e_id,
                    "label_index": example["label_index"],
                    "candidate_e_ids": example["candidate_e_ids"],
                    "text_a": polyset_candidates[0]["text_a"],
                    "candidate_text_bs": [c["text_b"] for c in polyset_candidates],
                }
            )

    summary = {
        "gold_xlsx": str(gold_xlsx),
        "dict_xlsx": str(dict_xlsx),
        "gold_sheet_name": gold_sheet_name,
        "input_mode": BGROUP_INPUT_MODE,
        "span_marker_style": BGROUP_SPAN_MARKER_STYLE,
        "text_b_format": BGROUP_TEXT_B_FORMAT,
        "candidate_scoring": BGROUP_CANDIDATE_SCORING,
        "n_rows_input": len(rows),
        "split_counts_raw": dict(split_counts_raw),
        "role_counts_raw": dict(role_counts_raw),
        "issue_counts": dict(issue_counts),
        "candidate_count_distribution": {str(k): int(v) for k, v in sorted(candidate_count_distribution.items())},
        "n_examples_by_split": {k: len(v) for k, v in by_split.items()},
    }
    logger.info("[bgroup_encoder_ce][data] summary=%s", summary)
    return by_split, summary, sample_rows
