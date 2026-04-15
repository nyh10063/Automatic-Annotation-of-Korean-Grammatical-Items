from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kmwe.data.rule_eval import (
    RuleEvalInstance,
    RuleEvalPrediction,
    compute_strict_metrics,
    dedupe_keep_order,
    load_gold_instances,
    write_predictions_csv,
)
from kmwe.stages.build_bgroup_sft import _build_prompt_core
from kmwe.stages.train_llm_sft import _render_chat_messages, parse_decision_line


@dataclass
class RuleE2EEvalConfig:
    gold_path: str
    dict_path: str
    gold_sheet_name: str = "gold"
    split_name: str = "test"
    mode: str = "gate_only"
    max_examples: int | None = None
    a_checkpoint: str | None = None
    b_checkpoint: str | None = None
    b_llm_model_name_or_path: str | None = None
    b_llm_backend: str = "hf"
    b_llm_allow_multiple: bool = False
    b_llm_max_input_len: int = 2048
    b_llm_max_new_tokens: int = 8
    b_llm_do_sample: bool = False
    b_llm_temperature: float = 1.0
    b_llm_top_p: float = 1.0
    b_llm_api_key_env: str = "OPENAI_API_KEY"
    a_group_accept_threshold: float = 0.55
    candidate_scoring_batch_size: int = 32
    b_group_max_seq_len: int = 256


@dataclass
class RuleGateDecision:
    example_key: str
    split: str | None
    group: str | None
    polyset_id: str | None
    sentence: str
    gold_e_id: str
    gold_example_role: str | None
    gold_span_segments: Any | None
    rule_gate_status: str
    rule_candidate_e_ids: list[str]
    rule_candidate_count: int
    matched_rule_ids: list[str]
    gold_in_candidates: bool
    downstream_mode: str | None
    downstream_pred_e_id: str | None
    final_status: str
    final_error_reason: str | None
    miss_stage: str
    candidate_source: str = "rule_detect"
    raw_downstream_output: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def load_test_instances(
    gold_path: str,
    *,
    sheet_name: str,
    expredict_map: dict[str, dict[str, Any]] | None = None,
    split_name: str = "test",
    max_examples: int | None = None,
) -> tuple[list[RuleEvalInstance], dict[str, Any]]:
    instances, gold_meta = load_gold_instances(
        gold_path,
        sheet_name=sheet_name,
        expredict_map=expredict_map or {},
    )
    wanted = str(split_name or "test").strip().lower()
    filtered = [inst for inst in instances if str((inst.meta or {}).get("split") or "").strip().lower() == wanted]
    if max_examples is not None:
        filtered = filtered[: int(max_examples)]
    meta = dict(gold_meta)
    meta.update(
        {
            "requested_split": wanted,
            "n_instances_before_split_filter": len(instances),
            "n_instances_after_split_filter": len(filtered),
            "max_examples": max_examples,
        }
    )
    return filtered, meta


def filter_instances_for_mode(instances: list[RuleEvalInstance], mode: str) -> list[RuleEvalInstance]:
    mode_l = str(mode or "gate_only").strip().lower()
    if mode_l == "a_group":
        return [inst for inst in instances if str(inst.gold_group or "").lower() == "a"]
    if mode_l in {"b_group_encoder", "b_group_llm"}:
        return [inst for inst in instances if str(inst.gold_group or "").lower() == "b"]
    return instances


def _parse_multi_text(raw: Any) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for chunk in text.replace(",", ";").split(";"):
        item = chunk.strip()
        if not item or item.lower() in {"nan", "none", "null"}:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_span_segments_any(raw: Any) -> list[tuple[int, int]]:
    import ast

    if raw is None or raw == "":
        return []
    parsed = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
    out: list[tuple[int, int]] = []
    for item in parsed or []:
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


def _derive_decision_type_from_gold(gold_e_ids: list[str]) -> str:
    if not gold_e_ids:
        return "none"
    if len(gold_e_ids) == 1:
        return "one"
    return "multi"


def build_bgroup_llm_prompt_row(decision: RuleGateDecision, *, allow_multiple: bool) -> dict[str, Any]:
    meta = dict(decision.meta or {})
    forced_single = _parse_multi_text(meta.get("gold_e_ids_single_if_forced"))
    gold_e_ids = _parse_multi_text(meta.get("gold_e_ids"))
    anchor_eid = str(meta.get("anchor_eid") or "").strip()
    if decision.gold_e_id != "__NONE__":
        effective_gold_e_ids = [str(decision.gold_e_id).strip()]
    else:
        effective_gold_e_ids = []
    if not gold_e_ids and anchor_eid and decision.gold_e_id != "__NONE__":
        gold_e_ids = [anchor_eid]
    if not forced_single and decision.gold_e_id != "__NONE__":
        forced_single = [str(decision.gold_e_id).strip()]
    span_segments = _parse_span_segments_any(decision.gold_span_segments)
    example_id = str(meta.get("example_id") or "").strip()
    instance_id = str(meta.get("instance_id") or "").strip()
    if (not example_id) and decision.example_key:
        parts = str(decision.example_key).split("#", 1)
        example_id = parts[0]
        if len(parts) > 1 and not instance_id:
            instance_id = parts[1]
    row = {
        "example_id": example_id,
        "instance_id": instance_id,
        "example_key_full": str(decision.example_key or "").strip(),
        "target_sentence": str(decision.sentence or ""),
        "context_left": "",
        "context_right": "",
        "gold_example_role": str(decision.gold_example_role or "").strip().lower(),
        "split": str(decision.split or meta.get("split") or "").strip().lower(),
        "pattern_type": str(meta.get("pattern_type") or "").strip(),
        "source": str(meta.get("source") or "").strip(),
        "note": str(meta.get("note") or ""),
        "anchor_eid": anchor_eid,
        "candidate_e_ids": list(decision.rule_candidate_e_ids or []),
        "gold_e_ids": gold_e_ids,
        "gold_e_ids_single_if_forced": forced_single,
        "effective_gold_e_ids": effective_gold_e_ids,
        "decision_type_raw": str(meta.get("decision_type") or "").strip().lower(),
        "decision_type": _derive_decision_type_from_gold(gold_e_ids),
        "effective_decision_type": _derive_decision_type_from_gold(effective_gold_e_ids),
        "span_segments_raw": str(decision.gold_span_segments or ""),
        "span_segments_parsed": span_segments,
        "gold_policy": "single_if_forced",
        "effective_gold_source": "gold_e_ids_single_if_forced",
        "allow_multiple": bool(allow_multiple),
    }
    return row


def build_bgroup_llm_prompt_payload(
    decision: RuleGateDecision,
    *,
    expredict_meta: dict[str, dict[str, Any]],
    allow_multiple: bool,
) -> dict[str, Any]:
    row = build_bgroup_llm_prompt_row(decision, allow_multiple=allow_multiple)
    system_prompt, user_prompt = _build_prompt_core(row, expredict_meta, allow_multiple)
    candidate_e_ids = list(row.get("candidate_e_ids") or [])
    candidate_number_to_eid = {str(i + 1): eid for i, eid in enumerate(candidate_e_ids)}
    metadata = {
        "example_key_full": row.get("example_key_full"),
        "gold_example_role": row.get("gold_example_role"),
        "candidate_e_ids": candidate_e_ids,
        "candidate_number_to_eid": candidate_number_to_eid,
        "gold_e_ids": list(row.get("gold_e_ids") or []),
        "gold_e_ids_single_if_forced": list(row.get("gold_e_ids_single_if_forced") or []),
        "effective_gold_e_ids": list(row.get("effective_gold_e_ids") or []),
        "effective_decision_type": str(row.get("effective_decision_type") or ""),
        "decision_type": str(row.get("decision_type") or ""),
        "split": row.get("split"),
        "gold_policy": row.get("gold_policy"),
        "effective_gold_source": row.get("effective_gold_source"),
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return {
        "row": row,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "candidate_number_to_eid": candidate_number_to_eid,
        "messages": messages,
        "metadata": metadata,
    }


def render_bgroup_llm_prompt_text(
    tokenizer: Any,
    payload: dict[str, Any],
    *,
    add_generation_prompt: bool = True,
) -> str:
    messages = list(payload.get("messages") or [])
    if len(messages) != 2:
        raise ValueError(f"B-group LLM prompt messages 길이는 2여야 합니다. got={len(messages)}")
    return _render_chat_messages(tokenizer, messages, add_generation_prompt=add_generation_prompt)


def build_bgroup_llm_generation_config(cfg: RuleE2EEvalConfig) -> dict[str, Any]:
    """Mirror the baseline HF LLM generation contract for isolated B-group e2e eval."""
    max_input_len = int(cfg.b_llm_max_input_len)
    max_new_tokens = int(cfg.b_llm_max_new_tokens)
    do_sample = bool(cfg.b_llm_do_sample)
    gen_cfg: dict[str, Any] = {
        "max_input_len": max_input_len,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": float(cfg.b_llm_temperature),
        "top_p": float(cfg.b_llm_top_p),
    }
    return gen_cfg


def parse_bgroup_llm_raw_output(
    raw_text: str,
    candidate_e_ids: list[str],
    *,
    allow_multiple: bool,
) -> dict[str, Any]:
    parsed = parse_decision_line(
        raw_text=raw_text,
        candidate_e_ids=list(candidate_e_ids or []),
        allow_multiple=allow_multiple,
    )
    pred_e_ids = [str(x).strip() for x in (parsed.get("pred_e_ids") or []) if str(x).strip()]
    pred_e_id = pred_e_ids[0] if pred_e_ids else "__NONE__"
    status = str(parsed.get("status") or "parse_failure").strip()
    ok = status == "ok"
    return {
        "status": status,
        "ok": ok,
        "pred_e_ids": pred_e_ids,
        "pred_e_id": pred_e_id,
        "decision_line": str(parsed.get("decision_line") or ""),
        "error_type": parsed.get("error_type"),
        "raw_text": str(parsed.get("raw_text") or raw_text or ""),
        "parser_policy": "baseline_parse_decision_line",
        "normalization_policy": "baseline_parse_decision_line",
    }


def build_rule_gate_decision(
    instance: RuleEvalInstance,
    *,
    candidates: list[dict[str, Any]],
    downstream_mode: str | None = None,
) -> RuleGateDecision:
    candidate_e_ids = dedupe_keep_order([str(c.get("e_id") or "").strip() for c in candidates])
    matched_rule_ids = dedupe_keep_order(
        [
            str(rule_id)
            for cand in candidates
            for rule_id in ((cand.get("stage_hits") or {}).get("detect") or [])
            if str(rule_id).strip()
        ]
    )
    gold_in_candidates = instance.gold_e_id != "__NONE__" and instance.gold_e_id in candidate_e_ids
    split = str((instance.meta or {}).get("split") or "") or None

    if instance.gold_e_id != "__NONE__":
        if gold_in_candidates:
            rule_gate_status = "positive_rule_pass"
            final_status = "rule_pass_to_downstream"
            final_error_reason = None
            miss_stage = "passed"
        else:
            rule_gate_status = "positive_rule_fail"
            final_status = "fn_rule_miss"
            final_error_reason = "rule_failed_to_generate_gold_eid"
            miss_stage = "no_candidate" if not candidate_e_ids else "unknown"
    else:
        if candidate_e_ids:
            rule_gate_status = "negative_rule_pass"
            final_status = "rule_pass_to_downstream"
            final_error_reason = None
            miss_stage = "passed"
        else:
            rule_gate_status = "negative_rule_fail"
            final_status = "tn_rule_blocked"
            final_error_reason = None
            miss_stage = "blocked"

    meta = dict(instance.meta)
    if instance.anchor_eid is not None:
        meta["anchor_eid"] = instance.anchor_eid

    return RuleGateDecision(
        example_key=instance.example_key,
        split=split,
        group=instance.gold_group,
        polyset_id=instance.gold_polyset_id,
        sentence=instance.sentence,
        gold_e_id=instance.gold_e_id,
        gold_example_role=instance.gold_example_role,
        gold_span_segments=instance.gold_span_segments,
        rule_gate_status=rule_gate_status,
        rule_candidate_e_ids=candidate_e_ids,
        rule_candidate_count=len(candidate_e_ids),
        matched_rule_ids=matched_rule_ids,
        gold_in_candidates=gold_in_candidates,
        downstream_mode=downstream_mode,
        downstream_pred_e_id=None,
        final_status=final_status,
        final_error_reason=final_error_reason,
        miss_stage=miss_stage,
        meta=meta,
    )


def apply_downstream_result(
    decision: RuleGateDecision,
    *,
    pred_e_id: str | None,
    raw_output: Any | None = None,
    error_reason: str | None = None,
) -> RuleGateDecision:
    guess = str(pred_e_id or "").strip() or None
    decision.downstream_pred_e_id = guess
    decision.raw_downstream_output = raw_output

    if decision.gold_e_id != "__NONE__":
        if guess is None:
            decision.final_status = "fn_downstream_none"
            decision.final_error_reason = error_reason or "downstream_returned_none"
        elif guess == decision.gold_e_id:
            decision.final_status = "tp_exact_match"
            decision.final_error_reason = None
        else:
            decision.final_status = "fp_fn_wrong_positive"
            decision.final_error_reason = error_reason or "downstream_wrong_positive"
    else:
        if guess is None:
            decision.final_status = "tn_recovered_none"
            decision.final_error_reason = None
        else:
            decision.final_status = "fp_negative_to_positive"
            decision.final_error_reason = error_reason or "negative_promoted_to_positive"
    return decision


def compute_rule_gate_metrics(decisions: list[RuleGateDecision]) -> dict[str, Any]:
    positive = [d for d in decisions if d.gold_e_id != "__NONE__"]
    negative = [d for d in decisions if d.gold_e_id == "__NONE__"]

    pos_pass = sum(1 for d in positive if d.rule_gate_status == "positive_rule_pass")
    pos_fail = sum(1 for d in positive if d.rule_gate_status == "positive_rule_fail")
    neg_pass = sum(1 for d in negative if d.rule_gate_status == "negative_rule_pass")
    neg_fail = sum(1 for d in negative if d.rule_gate_status == "negative_rule_fail")

    tp = pos_pass
    fn = pos_fail
    fp = neg_pass
    tn = neg_fail

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision or recall else 0.0

    return {
        "n_total": len(decisions),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "n_rule_passed": pos_pass + neg_pass,
        "n_rule_failed": pos_fail + neg_fail,
        "total_gate_coverage": _safe_div(pos_pass + neg_fail, len(decisions)),
        "positive_gate_recall": _safe_div(pos_pass, len(positive)),
        "negative_gate_tn_rate": _safe_div(neg_fail, len(negative)),
        "negative_candidate_rate": _safe_div(neg_pass, len(negative)),
        "avg_candidate_count_on_positive": _avg([d.rule_candidate_count for d in positive]),
        "avg_candidate_count_on_negative": _avg([d.rule_candidate_count for d in negative]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_final_e2e_metrics(decisions: list[RuleGateDecision]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for d in decisions:
        status = str(d.final_status or "")
        if status == "tp_exact_match":
            tp += 1
        elif status in {"fn_rule_miss", "fn_downstream_none"}:
            fn += 1
        elif status == "fp_fn_wrong_positive":
            fp += 1
            fn += 1
        elif status in {"tn_rule_blocked", "tn_recovered_none"}:
            tn += 1
        elif status == "fp_negative_to_positive":
            fp += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision or recall else 0.0
    return {
        "n_total": len(decisions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_downstream_conditional_metrics(decisions: list[RuleGateDecision]) -> dict[str, Any]:
    passed = [d for d in decisions if d.rule_gate_status in {"positive_rule_pass", "negative_rule_pass"}]
    out = compute_final_e2e_metrics(passed)
    out.update(
        {
            "conditional_eval_scope": "rule_passed_only",
            "n_rule_passed": len(passed),
            "n_rule_failed": len(decisions) - len(passed),
            "conditional_eval_population": len(passed),
        }
    )
    return out


def summarize_gate_by_field(decisions: list[RuleGateDecision], field: str) -> dict[str, Any]:
    buckets: dict[str, list[RuleGateDecision]] = {}
    for decision in decisions:
        key = str(getattr(decision, field, None) or "")
        buckets.setdefault(key, []).append(decision)
    return {key: compute_rule_gate_metrics(items) for key, items in buckets.items()}


def summarize_final_by_field(decisions: list[RuleGateDecision], field: str) -> dict[str, Any]:
    buckets: dict[str, list[RuleGateDecision]] = {}
    for decision in decisions:
        key = str(getattr(decision, field, None) or "")
        buckets.setdefault(key, []).append(decision)
    return {key: compute_final_e2e_metrics(items) for key, items in buckets.items()}


def summarize_downstream_conditional_by_field(decisions: list[RuleGateDecision], field: str) -> dict[str, Any]:
    buckets: dict[str, list[RuleGateDecision]] = {}
    for decision in decisions:
        key = str(getattr(decision, field, None) or "")
        buckets.setdefault(key, []).append(decision)
    return {key: compute_downstream_conditional_metrics(items) for key, items in buckets.items()}


def decision_to_row(decision: RuleGateDecision) -> dict[str, Any]:
    return asdict(decision)


def write_decisions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_predictions_csv(path, rows)


strict_metrics_from_predictions = compute_strict_metrics
legacy_rule_prediction_type = RuleEvalPrediction


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _avg(values: list[int]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0
