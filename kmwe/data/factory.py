from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterator
from collections import Counter

import pandas as pd

from kmwe.core.config_loader import ConfigError
from kmwe.data.adapter_ingested_record import ingested_record_to_text
from kmwe.data.ingested_index import list_shard_paths
from kmwe.data.mix_sampler import WeightedMixtureSampler
from kmwe.data.shard_reader import iter_jsonl_shards
from kmwe.data.types import BaseTextRecord, SpanSupervisionExample
from kmwe.data.utils_uid import build_uid


# legacy single-text A-group input
AGROUP_INPUT_CONSTRUCTION_VERSION = "agroup_binary_span_v1"
AGROUP_SPAN_MARKER_STYLE = "[SPAN]...[/SPAN]"

# v2 pair-style A-group input
AGROUP_INPUT_CONSTRUCTION_VERSION_V2 = "agroup_binary_pair_v2"
AGROUP_TEXT_B_FORMAT = "canonical_form_plus_gloss_plain"


def _normalize_span_segments(
    span_segments: list[list[int]] | list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in span_segments or []:
        try:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            start = int(item[0])
            end = int(item[1])
        except Exception:
            continue
        if end <= start:
            continue
        out.append((start, end))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _inject_span_markers(
    target_sentence: str,
    span_segments: list[tuple[int, int]],
) -> str:
    sent_text = str(target_sentence or "")
    if not sent_text or not span_segments:
        return sent_text
    text = sent_text
    markers = sorted(span_segments, key=lambda x: (x[0], x[1]), reverse=True)
    for start, end in markers:
        start_i = max(0, min(len(text), int(start)))
        end_i = max(start_i, min(len(text), int(end)))
        text = text[:end_i] + "[/SPAN]" + text[end_i:]
        text = text[:start_i] + "[SPAN]" + text[start_i:]
    return text


def _build_agroup_encoder_input_text(
    *,
    e_id: str,
    target_sentence: str,
    span_segments: list[list[int]] | list[tuple[int, int]] | None,
    context_left: str = "",
    context_right: str = "",
) -> str:
    e_id_text = str(e_id or "").strip()
    sent_text = str(target_sentence or "")
    left_text = str(context_left or "")
    right_text = str(context_right or "")
    normalized_segments = _normalize_span_segments(span_segments)
    marked_sentence = _inject_span_markers(sent_text, normalized_segments)
    try:
        spans_text = json.dumps(
            [[int(s), int(e)] for s, e in normalized_segments],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        spans_text = "[]"
    return "\n".join(
        [
            f"candidate_e_id={e_id_text}",
            f"target_sentence={marked_sentence}",
            f"context_left={left_text}",
            f"context_right={right_text}",
            f"span_segments={spans_text}",
            f"input_construction_version={AGROUP_INPUT_CONSTRUCTION_VERSION}",
            f"span_marker_style={AGROUP_SPAN_MARKER_STYLE}",
        ]
    )


def _build_agroup_pair_text_a(
    *,
    target_sentence: str,
    span_segments: list[list[int]] | list[tuple[int, int]] | None,
) -> str:
    sent_text = str(target_sentence or "")
    normalized_segments = _normalize_span_segments(span_segments)
    return _inject_span_markers(sent_text, normalized_segments).strip()


def _build_agroup_pair_text_b(
    *,
    canonical_form: str,
    gloss: str,
) -> str:
    canonical = str(canonical_form or "").strip()
    gloss_text = str(gloss or "").strip()
    if not canonical:
        raise ConfigError("A-group candidate canonical_form이 비어 있습니다.")
    if gloss_text:
        return canonical + "\n" + gloss_text
    return canonical


def build_agroup_pair_encoder_input(
    example: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    text_a = _build_agroup_pair_text_a(
        target_sentence=str(example.get("target_sentence") or ""),
        span_segments=example.get("span_segments") or [],
    )
    text_b = _build_agroup_pair_text_b(
        canonical_form=str(candidate.get("canonical_form") or ""),
        gloss=str(candidate.get("gloss") or ""),
    )
    normalized_segments = _normalize_span_segments(example.get("span_segments") or [])
    meta = {
        "candidate_e_id": str(candidate.get("e_id") or "").strip(),
        "canonical_form": str(candidate.get("canonical_form") or "").strip(),
        "gloss": str(candidate.get("gloss") or "").strip(),
        "span_segments": [[int(s), int(e)] for s, e in normalized_segments],
        "input_construction_version": AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
        "span_marker_style": AGROUP_SPAN_MARKER_STYLE,
        "text_b_format": AGROUP_TEXT_B_FORMAT,
    }
    return {"text_a": text_a, "text_b": text_b, "meta": meta}


def _load_agroup_candidate_meta(
    dict_xlsx: Path,
    sheet_name: str = "expredict",
) -> dict[str, dict[str, str]]:
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
            "group": str(row.get("group") or "").strip(),
        }
    return meta


def format_encoder_input(
    *,
    e_id: str,
    target_sentence: str,
    span_segments: list[list[int]] | list[tuple[int, int]],
    context_left: str = "",
    context_right: str = "",
) -> str:
    return _build_agroup_encoder_input_text(
        e_id=e_id,
        target_sentence=target_sentence,
        span_segments=span_segments,
        context_left=context_left,
        context_right=context_right,
    )


def build_tapt_stream(
    cfg: dict[str, Any],
    ingested_index: dict[str, Any],
    *,
    index_path: Path,
    max_examples: int | None = None,
) -> Iterator[BaseTextRecord]:
    tapt_cfg = cfg.get("tapt", {}) or {}
    data_mix = tapt_cfg.get("data_mix", {}) or {}
    seed = int(cfg.get("runtime", {}).get("seed", 0) or 0)
    deterministic = bool(cfg.get("runtime", {}).get("deterministic", True))
    shards_by_corpus = list_shard_paths(ingested_index, index_path=index_path)

    iterators: dict[str, Iterator[BaseTextRecord]] = {}
    for corpus, paths in shards_by_corpus.items():
        it = iter_jsonl_shards(paths)
        iterators[corpus] = (ingested_record_to_text(r, corpus=corpus) for r in it)

    sampler = WeightedMixtureSampler(iterators, data_mix, seed=seed, deterministic=deterministic)
    curriculum = tapt_cfg.get("learner_curriculum", {}) or {}
    if curriculum.get("enabled"):
        max_steps = int(tapt_cfg.get("max_steps") or (max_examples or 0))
        stage2_ratio = float(curriculum.get("stage2", {}).get("steps_ratio", 0.0) or 0.0)
        learner_ratio = float(curriculum.get("stage2", {}).get("learner_mix_ratio", 0.0) or 0.0)
        stage2_start = int(max_steps * stage2_ratio) if max_steps else 0
        base_weights = dict(data_mix)
        stage2_weights = dict(data_mix)
        if "learner_5_6" in stage2_weights:
            stage2_weights["learner_5_6"] = learner_ratio
        if "learner_5_6" in base_weights:
            base_weights["learner_5_6"] = 0.0
    else:
        stage2_start = None
        base_weights = None
        stage2_weights = None

    count = 0
    while max_examples is None or count < max_examples:
        if stage2_start is not None and base_weights is not None and stage2_weights is not None:
            sampler.set_weights(stage2_weights if count >= stage2_start else base_weights)
        sampled = sampler.sample()
        if sampled is None:
            break
        _, record = sampled
        yield record
        count += 1


def build_mtl_streams(
    cfg: dict[str, Any],
    ingested_index: dict[str, Any],
    *,
    index_path: Path,
    max_examples: int | None = None,
) -> dict[str, Iterator[BaseTextRecord]]:
    shards_by_corpus = list_shard_paths(ingested_index, index_path=index_path)

    def _stream(corpus: str) -> Iterator[BaseTextRecord]:
        paths = shards_by_corpus.get(corpus, [])
        it = iter_jsonl_shards(paths, limit=max_examples)
        return (ingested_record_to_text(r, corpus=corpus) for r in it)

    streams: dict[str, Iterator[BaseTextRecord]] = {}
    streams["structural"] = _stream("dependency")
    streams["sense"] = _stream("semantic")
    streams["pos_morph"] = _stream("morph")
    if bool(cfg.get("mtl", {}).get("pos_morph", {}).get("learner_mix_enabled", False)):
        streams["pos_morph_learner"] = _stream("learner_5_6")
    return streams


def build_weak_span_examples(
    cfg: dict[str, Any],
    silver_jsonl_path: Path,
    *,
    max_examples: int | None = None,
) -> Iterator[SpanSupervisionExample]:
    hold_weight = float(cfg.get("silver", {}).get("partial_labels", {}).get("hold_weight", 0.2))
    with silver_jsonl_path.open("r", encoding="utf-8") as fp:
        count = 0
        for line in fp:
            if max_examples is not None and count >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("target_sentence") or record.get("raw_sentence") or ""
            if not text:
                continue
            uid = build_uid(record, corpus="silver", text=text)
            candidates = record.get("candidates") or []
            for cand in candidates:
                triage = cand.get("triage")
                label = 1 if triage in {"confirm", "hold"} else 0
                weight = 1.0
                allowed = None
                if triage == "hold":
                    weight = hold_weight
                    allowed = cand.get("allowed_e_ids")
                span_segments = cand.get("span_segments") or []
                example = SpanSupervisionExample(
                    uid=uid,
                    text=format_encoder_input(
                        e_id=str(cand.get("e_id") or ""),
                        target_sentence=text,
                        span_segments=[(int(s), int(e)) for s, e in span_segments],
                        context_left="",
                        context_right="",
                    ),
                    context_left="",
                    context_right="",
                    candidate_e_id=str(cand.get("e_id") or ""),
                    span_segments=[(int(s), int(e)) for s, e in span_segments],
                    label=int(label),
                    weight=float(weight),
                    allowed_e_ids=allowed,
                )
                yield example
                count += 1
                if max_examples is not None and count >= max_examples:
                    break


def build_finetune_span_examples(
    cfg: dict[str, Any],
    gold_xlsx_path: Path,
    *,
    max_examples: int | None = None,
    allowed_splits_override: set[str] | None = None,
    emit_role_meta: bool = True,
    fail_on_zero_neg_target_absent_given_span: bool = False,
) -> Iterator[SpanSupervisionExample]:
    import logging
    import random

    logger = logging.getLogger(__name__)
    df = pd.read_excel(gold_xlsx_path, sheet_name="gold", engine="openpyxl")
    finetune_cfg = cfg.get("finetune", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    sheet_names_cfg = cfg.get("sheet_names", {}) or {}
    input_construction_version = str(
        finetune_cfg.get("input_construction_version") or AGROUP_INPUT_CONSTRUCTION_VERSION
    ).strip() or AGROUP_INPUT_CONSTRUCTION_VERSION
    pair_mode = input_construction_version == AGROUP_INPUT_CONSTRUCTION_VERSION_V2
    input_builder_name = (
        "build_agroup_pair_encoder_input" if pair_mode else "format_encoder_input"
    )
    agroup_candidate_meta: dict[str, dict[str, str]] = {}
    if pair_mode:
        dict_xlsx_raw = str(paths_cfg.get("dict_xlsx") or "").strip()
        if not dict_xlsx_raw:
            raise ConfigError(
                "A-group pair mode에서는 paths.dict_xlsx가 필요합니다."
            )
        dict_xlsx = Path(dict_xlsx_raw).expanduser()
        expredict_sheet_name = str(sheet_names_cfg.get("expredict") or "expredict")
        agroup_candidate_meta = _load_agroup_candidate_meta(dict_xlsx, expredict_sheet_name)
        logger.info(
            "[finetune][agroup_pair] enabled=true dict_xlsx=%s expredict_sheet=%s",
            dict_xlsx,
            expredict_sheet_name,
        )
    if allowed_splits_override is not None:
        allowed_splits = {
            str(x).strip().lower() for x in allowed_splits_override if str(x).strip()
        }
    else:
        allowed_splits = _parse_allowed_splits(finetune_cfg.get("allowed_splits"))
    split_filter_enabled = bool(allowed_splits)
    split_filter_fail_on_missing = bool(finetune_cfg.get("split_filter_fail_on_missing", True))
    split_seen = 0
    split_kept = 0
    split_dropped = 0
    split_missing = 0
    split_kept_by: Counter[str] = Counter()
    split_dropped_by: Counter[str] = Counter()
    if split_filter_enabled:
        logger.info(
            "[finetune][split] allowed_splits=%s fail_on_missing=%s",
            sorted(allowed_splits),
            split_filter_fail_on_missing,
        )
    n_yielded = 0
    n_role_pos = 0
    n_role_neg_boundary = 0
    n_role_neg_confusable = 0
    n_role_neg_target_absent = 0
    n_drop_neg_target_absent = 0
    n_drop_neg_confusable_no_conf = 0
    n_drop_span_missing = 0
    n_drop_unknown_role = 0
    n_neg_target_absent_given_span = 0
    n_neg_target_absent_random_span = 0
    n_neg_target_absent_fallback_parse_fail = 0

    def _attach_meta(
        example: SpanSupervisionExample,
        *,
        split: str,
        role: str,
        extra: dict[str, Any] | None = None,
    ) -> SpanSupervisionExample:
        if not emit_role_meta:
            return example
        payload: dict[str, Any] = {
            "split": split,
            "gold_example_role": role,
            "role": role,
            "input_construction_version": input_construction_version,
            "span_marker_style": AGROUP_SPAN_MARKER_STYLE,
            "input_builder": input_builder_name,
        }
        if pair_mode:
            payload["text_b_format"] = AGROUP_TEXT_B_FORMAT
        if extra:
            payload.update(extra)
        try:
            meta = getattr(example, "meta", None)
            if isinstance(meta, dict):
                meta.update(payload)
            else:
                object.__setattr__(example, "meta", dict(payload))
        except Exception:
            pass
        for key in ("split", "gold_example_role", "role"):
            try:
                object.__setattr__(example, key, payload.get(key))
            except Exception:
                pass
        return example

    def _safe_instance_id(value: Any) -> int:
        if value is None:
            return 0
        if pd.isna(value):
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    def _build_pair_meta_extra(
        *,
        candidate_e_id: str,
        target_sentence: str,
        span_segments: list[tuple[int, int]] | list[list[int]] | None,
    ) -> dict[str, Any]:
        if not pair_mode:
            return {}
        eid = str(candidate_e_id or "").strip()
        meta_row = agroup_candidate_meta.get(eid) or {}
        canonical_form = str(meta_row.get("canonical_form") or "").strip()
        gloss = str(meta_row.get("gloss") or "").strip()
        if not canonical_form:
            raise ConfigError(
                f"A-group pair mode candidate meta missing canonical_form: e_id={eid}"
            )
        built = build_agroup_pair_encoder_input(
            {
                "target_sentence": str(target_sentence or ""),
                "span_segments": span_segments or [],
            },
            {
                "e_id": eid,
                "canonical_form": canonical_form,
                "gloss": gloss,
            },
        )
        return {
            "canonical_form": canonical_form,
            "gloss": gloss,
            "text_a": str(built.get("text_a") or ""),
            "text_b": str(built.get("text_b") or ""),
            "text_b_format": AGROUP_TEXT_B_FORMAT,
            "group": str(meta_row.get("group") or "").strip(),
        }

    def _apply_pair_representative_text(example: SpanSupervisionExample) -> SpanSupervisionExample:
        if not pair_mode:
            return example
        meta = getattr(example, "meta", None)
        if not isinstance(meta, dict):
            return example
        text_a = str(meta.get("text_a") or "").strip()
        if not text_a:
            return example
        try:
            object.__setattr__(example, "text", text_a)
        except Exception:
            pass
        return example

    try:
        for _, row in df.iterrows():
            if max_examples is not None and n_yielded >= max_examples:
                break
            if split_filter_enabled:
                split_seen += 1
                split_name = str(row.get("split") or "").strip().lower()
                if not split_name:
                    split_missing += 1
                    if split_filter_fail_on_missing:
                        raise ConfigError(
                            "finetune.allowed_splits가 설정되었지만 gold.split이 비어 있습니다. "
                            "split_filter_fail_on_missing=true"
                        )
                    split_dropped += 1
                    split_dropped_by["__missing__"] += 1
                    continue
                if split_name not in allowed_splits:
                    split_dropped += 1
                    split_dropped_by[split_name] += 1
                    continue
                split_kept += 1
                split_kept_by[split_name] += 1

            role = str(row.get("gold_example_role", "") or "").strip()
            if role in {"pos_conti", "pos_disconti"}:
                label = 1
            elif role == "neg_boundary":
                label = 0
            elif role == "neg_confusable":
                label = 0
                conf_e_id = str(row.get("conf_e_id", "") or "").strip()
                if not conf_e_id:
                    n_drop_neg_confusable_no_conf += 1
                    continue
            elif role == "neg_target_absent":
                # create fake negative span anchor
                sent = str(row.get("target_sentence") or "")
                L = len(sent)
                if L < 1:
                    n_drop_neg_target_absent += 1
                    continue

                fake_span: list[tuple[int, int]]
                given_segments: list[tuple[int, int]] | None = None
                span_raw = row.get("span_segments")
                if not pd.isna(span_raw) and str(span_raw).strip() != "":
                    try:
                        parsed = ast.literal_eval(str(span_raw))
                        parsed_segments = [(int(s), int(e)) for s, e in parsed]
                        if parsed_segments:
                            given_segments = parsed_segments
                    except Exception:
                        n_neg_target_absent_fallback_parse_fail += 1

                if given_segments:
                    fake_span = given_segments
                    n_neg_target_absent_given_span += 1
                else:
                    start = random.randint(0, L - 1)
                    end = min(L, start + random.randint(1, 3))
                    fake_span = [(start, end)]
                    n_neg_target_absent_random_span += 1
                record = {
                    "example_id": row.get("example_id"),
                    "instance_id": row.get("instance_id"),
                    "doc_id": row.get("doc_id"),
                    "sent_index": row.get("sent_index"),
                }
                uid = build_uid(record, corpus="gold", text=sent)
                context_left = str(row.get("context_left") or "")
                context_right = str(row.get("context_right") or "")
                candidate_e_id = str(row.get("e_id") or "")
                split_value = str(row.get("split", "") or "").strip() or "UNK_SPLIT"
                policy = "given" if given_segments else "random"
                example = SpanSupervisionExample(
                    uid=uid,
                    text=format_encoder_input(
                        e_id=candidate_e_id,
                        target_sentence=sent,
                        span_segments=fake_span,
                        context_left=context_left,
                        context_right=context_right,
                    ),
                    context_left=context_left,
                    context_right=context_right,
                    candidate_e_id=candidate_e_id,
                    span_segments=fake_span,
                    label=0,
                    weight=1.0,
                    allowed_e_ids=None,
                )
                example = _attach_meta(
                    example,
                    split=split_value,
                    role=role,
                    extra={
                        "e_id": candidate_e_id,
                        "example_id": str(row.get("example_id", "") or ""),
                        "instance_id": _safe_instance_id(row.get("instance_id")),
                        "neg_target_absent_span_policy": policy,
                        **_build_pair_meta_extra(
                            candidate_e_id=candidate_e_id,
                            target_sentence=sent,
                            span_segments=fake_span,
                        ),
                    },
                )
                example = _apply_pair_representative_text(example)
                n_yielded += 1
                n_role_neg_target_absent += 1
                yield example
                continue
            else:
                n_drop_unknown_role += 1
                continue

            span_raw = row.get("span_segments")
            if pd.isna(span_raw):
                n_drop_span_missing += 1
                continue
            try:
                span_segments = ast.literal_eval(str(span_raw))
            except Exception:
                n_drop_span_missing += 1
                continue

            text = str(row.get("target_sentence") or "")
            if not text:
                continue
            record = {
                "example_id": row.get("example_id"),
                "instance_id": row.get("instance_id"),
                "doc_id": row.get("doc_id"),
                "sent_index": row.get("sent_index"),
            }
            uid = build_uid(record, corpus="gold", text=text)
            context_left = str(row.get("context_left") or "")
            context_right = str(row.get("context_right") or "")
            candidate_e_id = str(row.get("e_id") or "")
            split_value = str(row.get("split", "") or "").strip() or "UNK_SPLIT"
            example = SpanSupervisionExample(
                uid=uid,
                text=format_encoder_input(
                    e_id=candidate_e_id,
                    target_sentence=text,
                    span_segments=[(int(s), int(e)) for s, e in span_segments],
                    context_left=context_left,
                    context_right=context_right,
                ),
                context_left=context_left,
                context_right=context_right,
                candidate_e_id=candidate_e_id,
                span_segments=[(int(s), int(e)) for s, e in span_segments],
                label=int(label),
                weight=1.0,
                allowed_e_ids=None,
            )
            example = _attach_meta(
                example,
                split=split_value,
                role=role,
                extra={
                    "e_id": candidate_e_id,
                    "example_id": str(row.get("example_id", "") or ""),
                    "instance_id": _safe_instance_id(row.get("instance_id")),
                    **_build_pair_meta_extra(
                        candidate_e_id=candidate_e_id,
                        target_sentence=text,
                        span_segments=[(int(s), int(e)) for s, e in span_segments],
                    ),
                },
            )
            example = _apply_pair_representative_text(example)
            if role in {"pos_conti", "pos_disconti"}:
                n_role_pos += 1
            elif role == "neg_boundary":
                n_role_neg_boundary += 1
            elif role == "neg_confusable":
                n_role_neg_confusable += 1
            n_yielded += 1
            yield example
    finally:
        logger.info("finetune_examples n_yielded=%s", n_yielded)
        logger.info(
            "finetune_examples role_counts pos=%s neg_boundary=%s neg_confusable=%s neg_target_absent=%s",
            n_role_pos,
            n_role_neg_boundary,
            n_role_neg_confusable,
            n_role_neg_target_absent,
        )
        logger.info(
            "finetune_examples drops neg_target_absent=%s span_missing=%s unknown_role=%s",
            n_drop_neg_target_absent,
            n_drop_span_missing,
            n_drop_unknown_role,
        )
        logger.info(
            "finetune_examples neg_target_absent_span_policy given=%s random=%s fallback_parse_fail=%s",
            n_neg_target_absent_given_span,
            n_neg_target_absent_random_span,
            n_neg_target_absent_fallback_parse_fail,
        )
        summary_prefix = (
            "[finetune][split][summary]"
            if split_filter_enabled
            else "[finetune][split][summary][disabled]"
        )
        logger.info(
            "%s seen=%d kept=%d dropped=%d missing_split=%d kept_by_split=%s dropped_by_split=%s",
            summary_prefix,
            split_seen,
            split_kept,
            split_dropped,
            split_missing,
            dict(split_kept_by),
            dict(split_dropped_by),
        )
        if (
            fail_on_zero_neg_target_absent_given_span
            and n_role_neg_target_absent > 0
            and n_neg_target_absent_given_span == 0
        ):
            raise ConfigError(
                "guardrails: neg_target_absent 예제가 존재하지만 given span 사용 수가 0입니다. "
                "gold span 또는 split/role 전달 경로를 점검하세요."
            )


def _parse_allowed_splits(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out = {str(x).strip().lower() for x in value if str(x).strip()}
        return {x for x in out if x}
    text = str(value).strip()
    if text == "":
        return set()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ConfigError(f"finetune.allowed_splits JSON 파싱 실패: {text!r}") from exc
        if not isinstance(parsed, list):
            raise ConfigError(f"finetune.allowed_splits는 list 또는 문자열이어야 합니다: {type(parsed)}")
        out = {str(x).strip().lower() for x in parsed if str(x).strip()}
        return {x for x in out if x}
    out = {s.strip().lower() for s in text.split(",") if s.strip()}
    return {x for x in out if x}
