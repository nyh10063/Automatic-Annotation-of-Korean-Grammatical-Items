from __future__ import annotations

import ast
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from kmwe.data.factory import AGROUP_INPUT_CONSTRUCTION_VERSION_V2
from kmwe.data.factory_bgroup_encoder import build_bgroup_cross_encoder_input
from kmwe.data.rule_e2e_eval import (
    RuleE2EEvalConfig,
    apply_downstream_result,
    build_bgroup_llm_generation_config,
    build_bgroup_llm_prompt_payload,
    build_rule_gate_decision,
    compute_downstream_conditional_metrics,
    compute_final_e2e_metrics,
    compute_rule_gate_metrics,
    decision_to_row,
    filter_instances_for_mode,
    load_test_instances,
    parse_bgroup_llm_raw_output,
    render_bgroup_llm_prompt_text,
    summarize_final_by_field,
    summarize_gate_by_field,
    summarize_downstream_conditional_by_field,
    write_decisions_csv,
)
from kmwe.data.rule_eval import validate_input_paths
from kmwe.stages.eval_rule_gold import _detect_candidates_for_instance, _prepare_runtime
from kmwe.stages.infer_step1 import _apply_encoder_confidence, _build_encoder_scorer, _score_candidates_with_encoder
from kmwe.stages.train_bgroup_encoder_ce import (
    _label_space as _bgroup_label_space,
    _load_checkpoint as _load_bgroup_checkpoint,
    _resolve_device as _resolve_bgroup_device,
    _score_batch as _score_bgroup_batch,
)
from kmwe.utils.jsonio import write_json, write_jsonl_line


def _maybe_init_wandb(cfg: dict[str, Any], run_context: Any, summary_cfg: dict[str, Any], logger: logging.Logger):
    wandb_cfg = dict(cfg.get("wandb", {}) or {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        logger.warning("[eval_rule_end_to_end][wandb] import failed: %s", exc)
        return None
    run = wandb.init(
        project=str(wandb_cfg.get("project") or "kmwe-eval"),
        entity=str(wandb_cfg.get("entity") or "") or None,
        group=str(wandb_cfg.get("group") or "") or f"{getattr(run_context, 'exp_id', 'default')}:eval_rule_end_to_end",
        name=str(wandb_cfg.get("name") or "") or f"eval_rule_end_to_end/{getattr(run_context, 'exp_id', 'default')}/{getattr(run_context, 'run_id', 'manual')}",
        mode=str(wandb_cfg.get("mode") or "online"),
        tags=["eval_rule_end_to_end", str(getattr(run_context, 'exp_id', 'default'))],
        config=summary_cfg,
        reinit=True,
    )
    logger.info("[eval_rule_end_to_end][wandb] init ok")
    return run


def _wandb_log_report(wandb_run: Any, report: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    runtime_meta = dict(report.get("runtime_meta") or {})
    gate_metrics = dict(report.get("rule_gate_metrics") or {})
    conditional_metrics = dict(report.get("downstream_conditional_metrics") or {})
    final_metrics = dict(report.get("final_end_to_end_metrics") or {})
    payload: dict[str, Any] = {
        "meta/n_instances": int(report.get("n_instances") or 0),
        "meta/evaluated_mode": str(report.get("evaluated_mode") or ""),
    }
    meta_fields = {
        "gold_policy": runtime_meta.get("gold_policy"),
        "effective_gold_source": runtime_meta.get("effective_gold_source"),
        "positive_gate_policy": runtime_meta.get("positive_gate_policy"),
        "negative_gate_policy": runtime_meta.get("negative_gate_policy"),
        "candidate_scoring_batch_size": runtime_meta.get("candidate_scoring_batch_size"),
        "a_group_accept_threshold": runtime_meta.get("a_group_accept_threshold"),
        "a_input_construction_version": runtime_meta.get("a_input_construction_version"),
        "b_group_max_seq_len": runtime_meta.get("b_group_max_seq_len"),
        "a_scorer_loaded": runtime_meta.get("a_scorer_loaded"),
        "b_scorer_loaded": runtime_meta.get("b_scorer_loaded"),
        "b_llm_loaded": runtime_meta.get("b_llm_loaded"),
        "b_llm_backend": runtime_meta.get("b_llm_backend"),
        "b_llm_model_name_or_path": runtime_meta.get("b_llm_model_name_or_path_runtime") or runtime_meta.get("b_llm_model_name_or_path"),
        "b_llm_allow_multiple": runtime_meta.get("b_llm_allow_multiple"),
        "b_llm_max_input_len": runtime_meta.get("b_llm_max_input_len"),
        "b_llm_max_new_tokens": runtime_meta.get("b_llm_max_new_tokens"),
        "b_llm_do_sample": runtime_meta.get("b_llm_do_sample"),
        "b_llm_temperature": runtime_meta.get("b_llm_temperature"),
        "b_llm_top_p": runtime_meta.get("b_llm_top_p"),
        "b_llm_use_bf16": runtime_meta.get("b_llm_use_bf16"),
        "b_llm_attn_implementation": runtime_meta.get("b_llm_attn_implementation"),
        "n_examples": gate_metrics.get("n_examples"),
        "n_positive": gate_metrics.get("n_positive"),
        "n_negative": gate_metrics.get("n_negative"),
        "n_rule_passed": conditional_metrics.get("conditional_eval_population"),
        "n_rule_failed": (gate_metrics.get("n_examples") or 0) - (conditional_metrics.get("conditional_eval_population") or 0),
        "n_final_tp": final_metrics.get("tp"),
        "n_final_fp": final_metrics.get("fp"),
        "n_final_fn": final_metrics.get("fn"),
        "n_final_tn": final_metrics.get("tn"),
    }
    for k, v in meta_fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            payload[f"meta/{k}"] = int(v)
        elif isinstance(v, (int, float, str)):
            payload[f"meta/{k}"] = v
    for section_name in ("rule_gate_metrics", "downstream_conditional_metrics", "final_end_to_end_metrics"):
        section = dict(report.get(section_name) or {})
        prefix = section_name.replace("_metrics", "")
        for k, v in section.items():
            if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
                payload[f"{prefix}/{k}"] = v
            elif isinstance(v, bool):
                payload[f"{prefix}/{k}"] = int(v)
    wandb_run.log(payload)

def run_eval_rule_end_to_end(*, cfg: dict[str, Any], run_context: Any) -> dict[str, Any]:
    logger = logging.getLogger("kmwe")
    e2e_cfg = RuleE2EEvalConfig(
        gold_path=str(
            cfg.get("paths", {}).get("gold_xlsx")
            or cfg.get("paths", {}).get("gold_b_xlsx")
            or ""
        ),
        dict_path=str(cfg.get("paths", {}).get("dict_xlsx") or ""),
        gold_sheet_name=str(cfg.get("rule_e2e", {}).get("gold_sheet_name", "gold")),
        split_name=str(cfg.get("rule_e2e", {}).get("split_name", "test")),
        mode=str(cfg.get("rule_e2e", {}).get("mode", "gate_only")),
        max_examples=_parse_optional_int(cfg.get("rule_e2e", {}).get("max_examples")),
        a_checkpoint=_parse_optional_str(cfg.get("rule_e2e", {}).get("a_checkpoint")),
        b_checkpoint=_parse_optional_str(cfg.get("rule_e2e", {}).get("b_checkpoint")),
        b_llm_model_name_or_path=_parse_optional_str(cfg.get("rule_e2e", {}).get("b_llm_model_name_or_path")),
        b_llm_backend=str(cfg.get("rule_e2e", {}).get("b_llm_backend", "hf") or "hf"),
        b_llm_allow_multiple=bool(cfg.get("rule_e2e", {}).get("b_llm_allow_multiple", False)),
        b_llm_max_input_len=int(cfg.get("rule_e2e", {}).get("b_llm_max_input_len", 2048)),
        b_llm_max_new_tokens=int(cfg.get("rule_e2e", {}).get("b_llm_max_new_tokens", 8)),
        b_llm_do_sample=bool(cfg.get("rule_e2e", {}).get("b_llm_do_sample", False)),
        b_llm_temperature=float(cfg.get("rule_e2e", {}).get("b_llm_temperature", 1.0)),
        b_llm_top_p=float(cfg.get("rule_e2e", {}).get("b_llm_top_p", 1.0)),
        b_llm_api_key_env=str(cfg.get("rule_e2e", {}).get("b_llm_api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY"),
        a_group_accept_threshold=float(cfg.get("rule_e2e", {}).get("a_group_accept_threshold", 0.55)),
        candidate_scoring_batch_size=int(cfg.get("rule_e2e", {}).get("candidate_scoring_batch_size", 32)),
        b_group_max_seq_len=int(cfg.get("rule_e2e", {}).get("b_group_max_seq_len", 256)),
    )
    outputs_dir = Path(run_context.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(
        cfg,
        run_context,
        {
            "stage": "eval_rule_end_to_end",
            "evaluated_mode": e2e_cfg.mode,
            "gold_path": e2e_cfg.gold_path,
            "dict_path": e2e_cfg.dict_path,
            "split_name": e2e_cfg.split_name,
            "a_checkpoint": e2e_cfg.a_checkpoint,
            "b_checkpoint": e2e_cfg.b_checkpoint,
            "b_llm_model_name_or_path": e2e_cfg.b_llm_model_name_or_path,
            "b_llm_backend": e2e_cfg.b_llm_backend,
            "gold_policy": "single_if_forced",
            "effective_gold_source": "gold_e_ids_single_if_forced",
        },
        logger,
    )

    path_errors = validate_input_paths(e2e_cfg)
    if path_errors:
        raise FileNotFoundError(json.dumps(path_errors, ensure_ascii=False))

    runtime = _prepare_runtime(_augment_cfg_for_rule_runtime(cfg), e2e_cfg, run_context, logger)
    instances, gold_meta = load_test_instances(
        e2e_cfg.gold_path,
        sheet_name=e2e_cfg.gold_sheet_name,
        expredict_map=runtime["expredict_map"],
        split_name=e2e_cfg.split_name,
        max_examples=e2e_cfg.max_examples,
    )
    instances = filter_instances_for_mode(instances, e2e_cfg.mode)

    a_scorer = _maybe_build_a_scorer(cfg, e2e_cfg, run_context, logger)
    b_scorer = _maybe_build_b_scorer(e2e_cfg)
    b_llm_bundle = _maybe_build_b_llm_bundle(cfg, e2e_cfg, logger)

    decisions = []
    top_error_patterns: Counter[str] = Counter()
    for instance in instances:
        candidates = _detect_candidates_for_instance(instance, runtime)
        candidates = _filter_candidates_by_target_span(candidates, instance.gold_span_segments)
        decision = build_rule_gate_decision(
            instance,
            candidates=candidates,
            downstream_mode=None if e2e_cfg.mode == "gate_only" else e2e_cfg.mode,
        )
        if e2e_cfg.mode == "a_group" and decision.rule_gate_status in {"positive_rule_pass", "negative_rule_pass"}:
            pred_e_id, raw_output, err = _run_a_group_downstream(
                candidates=candidates,
                sentence=instance.sentence,
                runtime=runtime,
                scorer=a_scorer,
                cfg=e2e_cfg,
                input_construction_version=str((cfg.get("finetune", {}) or {}).get("input_construction_version") or AGROUP_INPUT_CONSTRUCTION_VERSION_V2).strip() or AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
                logger=logger,
            )
            decision = apply_downstream_result(decision, pred_e_id=pred_e_id, raw_output=raw_output, error_reason=err)
        elif e2e_cfg.mode == "b_group_encoder" and decision.rule_gate_status in {"positive_rule_pass", "negative_rule_pass"}:
            pred_e_id, raw_output, err = _run_b_group_encoder_downstream(
                decision=decision,
                runtime=runtime,
                scorer_bundle=b_scorer,
                cfg=e2e_cfg,
            )
            decision = apply_downstream_result(decision, pred_e_id=pred_e_id, raw_output=raw_output, error_reason=err)
        elif e2e_cfg.mode == "b_group_llm" and decision.rule_gate_status in {"positive_rule_pass", "negative_rule_pass"}:
            pred_e_id, raw_output, err = _run_b_group_llm_downstream(
                decision=decision,
                runtime=runtime,
                llm_bundle=b_llm_bundle,
                cfg=e2e_cfg,
            )
            decision = apply_downstream_result(decision, pred_e_id=pred_e_id, raw_output=raw_output, error_reason=err)
        decisions.append(decision)
        if decision.final_error_reason:
            top_error_patterns[decision.final_error_reason] += 1

    gate_metrics = compute_rule_gate_metrics(decisions)
    final_metrics = None if e2e_cfg.mode == "gate_only" else compute_final_e2e_metrics(decisions)
    conditional_metrics = None if e2e_cfg.mode == "gate_only" else compute_downstream_conditional_metrics(decisions)
    group_a = [d for d in decisions if str(d.group or "").lower() == "a"]
    group_b = [d for d in decisions if str(d.group or "").lower() == "b"]
    by_e_id_gate = summarize_gate_by_field([d for d in decisions if d.gold_e_id != "__NONE__"], "gold_e_id")
    by_polyset_gate = summarize_gate_by_field([d for d in decisions if d.polyset_id], "polyset_id")
    by_e_id_final = {} if final_metrics is None else summarize_final_by_field(decisions, "gold_e_id")
    by_polyset_final = {} if final_metrics is None else summarize_final_by_field([d for d in decisions if d.polyset_id], "polyset_id")

    report = {
        "stage": "eval_rule_end_to_end",
        "evaluated_mode": e2e_cfg.mode,
        "run": {
            "exp_id": getattr(run_context, "exp_id", None),
            "run_id": getattr(run_context, "run_id", None),
        },
        "inputs": {
            "gold_path": e2e_cfg.gold_path,
            "dict_path": e2e_cfg.dict_path,
            "gold_sheet_name": e2e_cfg.gold_sheet_name,
            "split_name": e2e_cfg.split_name,
            "max_examples": e2e_cfg.max_examples,
            "a_checkpoint": e2e_cfg.a_checkpoint,
            "b_checkpoint": e2e_cfg.b_checkpoint,
        },
        "gold_meta": gold_meta,
        "runtime_meta": {
            "dict_source": runtime["dict_source"],
            "dict_stats": runtime["dict_stats"],
            "detect_rule_count": len(runtime["detect_rules"]),
            "kiwi_model": runtime["kiwi_model"],
            "gold_policy": "single_if_forced",
            "effective_gold_source": "gold_e_ids_single_if_forced",
            "positive_gate_policy": "gold_e_id_and_target_span_v2",
            "negative_gate_policy": "any_target_span_candidate_passes_v2",
            "a_group_accept_threshold": e2e_cfg.a_group_accept_threshold,
            "a_input_construction_version": str((cfg.get("finetune", {}) or {}).get("input_construction_version") or AGROUP_INPUT_CONSTRUCTION_VERSION_V2).strip() or AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
            "candidate_scoring_batch_size": e2e_cfg.candidate_scoring_batch_size,
            "b_group_max_seq_len": e2e_cfg.b_group_max_seq_len,
            "b_llm_model_name_or_path": e2e_cfg.b_llm_model_name_or_path,
            "b_llm_backend": e2e_cfg.b_llm_backend,
            "b_llm_allow_multiple": e2e_cfg.b_llm_allow_multiple,
            "b_llm_max_input_len": e2e_cfg.b_llm_max_input_len,
            "b_llm_max_new_tokens": e2e_cfg.b_llm_max_new_tokens,
            "b_llm_do_sample": e2e_cfg.b_llm_do_sample,
            "b_llm_temperature": e2e_cfg.b_llm_temperature,
            "b_llm_top_p": e2e_cfg.b_llm_top_p,
            "a_scorer_loaded": bool(a_scorer is not None),
            "b_scorer_loaded": bool(b_scorer is not None),
            "b_llm_loaded": bool(b_llm_bundle is not None),
            "b_llm_model_backend_runtime": (b_llm_bundle or {}).get("backend"),
            "b_llm_model_name_or_path_runtime": (b_llm_bundle or {}).get("model_name_or_path"),
            "b_llm_use_bf16": (b_llm_bundle or {}).get("use_bf16"),
            "b_llm_attn_implementation": (b_llm_bundle or {}).get("attn_implementation"),
        },
        "n_instances": len(instances),
        "rule_gate_metrics": gate_metrics,
        "downstream_conditional_metrics": conditional_metrics,
        "final_end_to_end_metrics": final_metrics,
        "group_a_metrics": compute_final_e2e_metrics(group_a) if final_metrics is not None else compute_rule_gate_metrics(group_a),
        "group_b_metrics": compute_final_e2e_metrics(group_b) if final_metrics is not None else compute_rule_gate_metrics(group_b),
        "overall_metrics": final_metrics or gate_metrics,
        "by_e_id_metrics": by_e_id_final or by_e_id_gate,
        "by_e_id_rule_gate_metrics": summarize_gate_by_field(decisions, "gold_e_id"),
        "by_e_id_downstream_conditional_metrics": summarize_downstream_conditional_by_field(decisions, "gold_e_id"),
        "by_e_id_final_end_to_end_metrics": summarize_final_by_field(decisions, "gold_e_id"),
        "by_polyset_metrics": by_polyset_final or by_polyset_gate,
        "top_error_patterns": top_error_patterns.most_common(20),
        "implementation_status": _implementation_status(e2e_cfg.mode),
        "outputs": {
            "e2e_predictions_jsonl": str(outputs_dir / "e2e_predictions.jsonl"),
            "e2e_predictions_csv": str(outputs_dir / "e2e_predictions.csv"),
            "gate_summary_json": str(outputs_dir / "gate_summary.json"),
            "error_breakdown_json": str(outputs_dir / "error_breakdown.json"),
            "error_cases_jsonl": str(outputs_dir / "error_cases.jsonl"),
            "polyset_metrics_csv": str(outputs_dir / "polyset_metrics.csv"),
            "downstream_prompt_log_jsonl": str(outputs_dir / "downstream_prompt_log.jsonl"),
        },
    }

    rows = [decision_to_row(d) for d in decisions]
    with (outputs_dir / "e2e_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            write_jsonl_line(fh, row)
    write_decisions_csv(outputs_dir / "e2e_predictions.csv", rows)
    write_json(outputs_dir / "gate_summary.json", gate_metrics, indent=2)
    write_json(outputs_dir / "error_breakdown.json", {"top_error_patterns": report["top_error_patterns"]}, indent=2)
    with (outputs_dir / "error_cases.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            if row.get("final_error_reason"):
                write_jsonl_line(fh, row)
    _write_polyset_csv(outputs_dir / "polyset_metrics.csv", report["by_polyset_metrics"])
    _write_downstream_prompt_log(outputs_dir / "downstream_prompt_log.jsonl", decisions)
    write_json(outputs_dir / "eval_rule_end_to_end_report.json", report, indent=2)
    _wandb_log_report(wandb_run, report)
    if wandb_run is not None:
        wandb_run.finish()
    return report


def _augment_cfg_for_rule_runtime(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out.setdefault("paths", {})
    out.setdefault("silver", {})
    return out


def _filter_candidates_by_target_span(candidates: list[dict[str, Any]], gold_span_segments: Any) -> list[dict[str, Any]]:
    gold_spans = _normalize_span_segments_any(gold_span_segments)
    if not gold_spans:
        return list(candidates)
    filtered = [cand for cand in candidates if _candidate_overlaps_target_span(cand, gold_spans)]
    return filtered


def _candidate_overlaps_target_span(candidate: dict[str, Any], gold_spans: list[tuple[int, int]]) -> bool:
    cand_spans = _normalize_span_segments_any(candidate.get("span_segments"))
    if not cand_spans:
        return False
    for c0, c1 in cand_spans:
        for g0, g1 in gold_spans:
            if min(c1, g1) > max(c0, g0):
                return True
    return False


def _normalize_span_segments_any(value: Any) -> list[tuple[int, int]]:
    if value is None or value == "":
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
    out: list[tuple[int, int]] = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            if len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                start = int(item[0]); end = int(item[1])
                if end > start:
                    out.append((start, end))
                return
            for child in item:
                walk(child)

    walk(parsed)
    out.sort(key=lambda x: (x[0], x[1]))
    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for span in out:
        if span not in seen:
            seen.add(span)
            deduped.append(span)
    return deduped


def _maybe_build_a_scorer(cfg: dict[str, Any], e2e_cfg: RuleE2EEvalConfig, run_context: Any, logger: logging.Logger):
    if e2e_cfg.mode != "a_group":
        return None
    if not e2e_cfg.a_checkpoint:
        raise ValueError("rule_e2e.a_checkpoint is required for mode=a_group")
    scorer_cfg = dict(cfg)
    scorer_cfg.setdefault("infer", {})
    scorer_cfg["infer"] = dict(scorer_cfg["infer"])
    scorer_cfg["infer"]["encoder_scoring_checkpoint"] = e2e_cfg.a_checkpoint
    scorer_cfg["infer"]["scoring_method"] = "head_logits"
    scorer_cfg.setdefault("runtime", {})
    return _build_encoder_scorer(
        cfg=scorer_cfg,
        run_context=run_context,
        enabled=True,
        max_seq_len=256,
        logger=logger,
        require_head_logits=True,
        disallow_fallback_scoring=True,
    )


def _maybe_build_b_scorer(e2e_cfg: RuleE2EEvalConfig):
    if e2e_cfg.mode != "b_group_encoder":
        return None
    if not e2e_cfg.b_checkpoint:
        raise ValueError("rule_e2e.b_checkpoint is required for mode=b_group_encoder")
    ckpt_dir = Path(e2e_cfg.b_checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"b_group checkpoint directory not found: {ckpt_dir}")
    head_path = ckpt_dir / "head.pt"
    enc_dir = ckpt_dir / "encoder"
    tok_dir = ckpt_dir / "tokenizer"
    if not head_path.exists():
        raise FileNotFoundError(f"b_group head.pt not found: {head_path}")
    if not enc_dir.exists():
        raise FileNotFoundError(f"b_group encoder directory not found: {enc_dir}")
    if not tok_dir.exists():
        raise FileNotFoundError(f"b_group tokenizer directory not found: {tok_dir}")
    device = _resolve_bgroup_device("auto")
    model, tokenizer, scorer = _load_bgroup_checkpoint(
        encoder_name=str(enc_dir),
        tokenizer_name=str(tok_dir),
        head_path=head_path,
        device=device,
    )
    model.eval()
    scorer.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "scorer": scorer,
        "device": device,
        "mixed_precision": "fp16",
        "checkpoint_dir": str(ckpt_dir),
    }


def _render_openai_messages_as_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = str(msg.get("role") or "").strip() or "unknown"
        content = str(msg.get("content") or "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts).strip()

def _resolve_b_llm_constrained_decoding(cfg: dict[str, Any]) -> bool:
    e2e_cfg = dict(cfg.get("rule_e2e", {}) or {})
    if "b_llm_constrained_decoding" in e2e_cfg:
        return bool(e2e_cfg.get("b_llm_constrained_decoding"))
    return True


def _normalize_token_id_sequence(ids: Any) -> list[int]:
    if ids is None:
        return []
    if isinstance(ids, int):
        return [int(ids)]
    out: list[int] = []
    for v in list(ids):
        if v is None:
            continue
        out.append(int(v))
    return out


def _build_allowed_label_token_sequences(tokenizer: Any, metadata: dict[str, Any]) -> list[list[int]]:
    number_map = dict(metadata.get("candidate_number_to_eid") or {})
    candidate_numbers = sorted(number_map.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x))
    labels = [str(num) for num in candidate_numbers]
    labels.append("NONE")
    seqs: list[list[int]] = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if ids:
            seqs.append([int(x) for x in ids])
    dedup: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seq in seqs:
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(seq)
    return dedup


def _build_prefix_allowed_tokens_fn(*, prompt_len: int, allowed_sequences: Sequence[Sequence[int]], eos_token_id: Any, pad_token_id: Any) -> Callable[[int, Any], list[int]]:
    eos_ids = _normalize_token_id_sequence(eos_token_id)
    pad_ids = _normalize_token_id_sequence(pad_token_id)
    stop_ids = list(dict.fromkeys(eos_ids + pad_ids))
    def prefix_allowed_tokens_fn(batch_id: int, input_ids: Any) -> list[int]:
        if getattr(input_ids, "dim", lambda: 1)() == 1:
            full_ids = [int(x) for x in input_ids.tolist()]
        else:
            full_ids = [int(x) for x in input_ids[0].tolist()]
        cur_len = len(full_ids)
        gen_len = max(cur_len - prompt_len, 0)
        generated = full_ids[prompt_len:cur_len] if cur_len > prompt_len else []
        next_ids: list[int] = []
        any_complete = False
        for seq in allowed_sequences:
            if gen_len > len(seq):
                continue
            if generated == list(seq[:gen_len]):
                if gen_len == len(seq):
                    any_complete = True
                else:
                    next_ids.append(int(seq[gen_len]))
        if next_ids:
            return list(dict.fromkeys(next_ids))
        if any_complete and stop_ids:
            return stop_ids
        if stop_ids:
            return stop_ids
        return []
    return prefix_allowed_tokens_fn


def _maybe_build_b_llm_bundle(cfg: dict[str, Any], e2e_cfg: RuleE2EEvalConfig, logger: logging.Logger):
    if e2e_cfg.mode != "b_group_llm":
        return None
    backend = str(e2e_cfg.b_llm_backend or "hf").strip().lower()
    model_name = str(e2e_cfg.b_llm_model_name_or_path or "").strip()
    if not model_name:
        raise ValueError("rule_e2e.b_llm_model_name_or_path is required for mode=b_group_llm")

    llm_cfg = dict(cfg.get("llm_sft", {}) or {})
    use_bf16 = bool(llm_cfg.get("use_bf16", True))
    attn_implementation = str(llm_cfg.get("attn_implementation") or "").strip()
    if attn_implementation.lower() == "auto":
        attn_implementation = ""

    if backend == "openai":
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("b_group_llm backend=openai requires the openai package.") from exc
        import os
        api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for b_group_llm backend=openai")
        base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        logger.info('[eval_rule_end_to_end][b_group_llm] loaded OpenAI model=%s backend=%s base_url=%s', model_name, backend, base_url or 'default')
        return {
            "backend": backend,
            "model_name_or_path": model_name,
            "client": client,
            "device": None,
            "use_bf16": None,
            "attn_implementation": None,
            "checkpoint_path": model_name,
            "resolved_model_path": model_name,
            "resolved_tokenizer_path": None,
        }

    if backend != "hf":
        raise NotImplementedError(f"b_group_llm backend={backend} is not implemented yet")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("b_group_llm mode requires torch and transformers.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(model_name)
    tokenizer_path = ckpt_dir / "tokenizer" if ckpt_dir.exists() and (ckpt_dir / "tokenizer").exists() else ckpt_dir
    model_path = ckpt_dir / "model" if ckpt_dir.exists() and (ckpt_dir / "model").exists() else ckpt_dir

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_load_kwargs: dict[str, Any] = {}
    if device.type == "cuda" and use_bf16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    if attn_implementation:
        model_load_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, **model_load_kwargs)
    model.to(device)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        if getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    logger.info('[eval_rule_end_to_end][b_group_llm] loaded model=%s backend=%s device=%s use_bf16=%s attn_implementation=%s model_path=%s tokenizer_path=%s', model_name, backend, device, use_bf16, attn_implementation or 'auto', model_path, tokenizer_path)
    return {
        "backend": backend,
        "model_name_or_path": model_name,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "use_bf16": use_bf16,
        "attn_implementation": attn_implementation or "auto",
        "checkpoint_path": model_name,
        "resolved_model_path": str(model_path),
        "resolved_tokenizer_path": str(tokenizer_path),
    }


def _run_b_group_llm_downstream(*, decision, runtime: dict[str, Any], llm_bundle, cfg: RuleE2EEvalConfig):
    if not llm_bundle:
        raise RuntimeError("b_group_llm bundle is not initialized")
    payload = build_bgroup_llm_prompt_payload(
        decision,
        expredict_meta=runtime.get("expredict_map") or {},
        allow_multiple=bool(cfg.b_llm_allow_multiple),
    )
    gen_cfg = build_bgroup_llm_generation_config(cfg)
    backend = str(llm_bundle.get("backend") or "hf").strip().lower()

    if backend == "openai":
        client = llm_bundle["client"]
        messages = list(payload.get("messages") or [])
        prompt_text = _render_openai_messages_as_text(messages)
        req = {
            "model": str(llm_bundle.get("model_name_or_path") or ""),
            "messages": messages,
            "max_tokens": int(gen_cfg.get("max_new_tokens", 8)),
        }
        if bool(gen_cfg.get("do_sample", False)):
            req["temperature"] = float(gen_cfg.get("temperature", 1.0))
            req["top_p"] = float(gen_cfg.get("top_p", 1.0))
        else:
            req["temperature"] = 0.0
        resp = client.chat.completions.create(**req)
        choice = (getattr(resp, "choices", None) or [None])[0]
        msg = getattr(choice, "message", None) if choice is not None else None
        raw_text = str(getattr(msg, "content", "") or "")
    else:
        tokenizer = llm_bundle["tokenizer"]
        model = llm_bundle["model"]
        prompt_text = render_bgroup_llm_prompt_text(tokenizer, payload, add_generation_prompt=True)
        constrained_decoding = _resolve_b_llm_constrained_decoding({"rule_e2e": {"b_llm_constrained_decoding": getattr(cfg, "b_llm_constrained_decoding", True)}})

        import torch

        device = llm_bundle["device"]
        model.eval()
        with torch.no_grad():
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=int(gen_cfg.get("max_input_len", 2048)),
                add_special_tokens=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_len = int(enc["input_ids"].shape[1])
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(gen_cfg.get("max_new_tokens", 8)),
                "do_sample": bool(gen_cfg.get("do_sample", False)),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if bool(gen_kwargs["do_sample"]):
                gen_kwargs["temperature"] = float(gen_cfg.get("temperature", 1.0))
                gen_kwargs["top_p"] = float(gen_cfg.get("top_p", 1.0))
            if constrained_decoding:
                allowed_sequences = _build_allowed_label_token_sequences(tokenizer, dict(payload.get("metadata") or {}))
                if allowed_sequences:
                    gen_kwargs["prefix_allowed_tokens_fn"] = _build_prefix_allowed_tokens_fn(prompt_len=prompt_len, allowed_sequences=allowed_sequences, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                    longest = max(len(seq) for seq in allowed_sequences)
                    gen_kwargs["max_new_tokens"] = max(1, longest + 1)
            output_ids = model.generate(**enc, **gen_kwargs)
            new_ids = output_ids[0][prompt_len:]
            raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    candidate_e_ids = list(payload.get("metadata", {}).get("candidate_e_ids") or [])
    parsed = parse_bgroup_llm_raw_output(
        raw_text=raw_text,
        candidate_e_ids=candidate_e_ids,
        allow_multiple=bool(cfg.b_llm_allow_multiple),
    )
    pred_e_id = str(parsed.get("pred_e_id") or "__NONE__").strip()
    raw_output = {
        "downstream_messages": list(payload.get("messages") or []),
        "downstream_prompt_text": prompt_text,
        "downstream_candidate_e_ids": candidate_e_ids,
        "downstream_candidate_number_to_eid": dict(payload.get("candidate_number_to_eid") or {}),
        "downstream_generation_config": dict(gen_cfg),
        "downstream_model_name_or_path": llm_bundle.get("model_name_or_path"),
        "downstream_checkpoint_path": llm_bundle.get("checkpoint_path"),
        "downstream_backend": llm_bundle.get("backend"),
        "downstream_raw_model_output": raw_text,
        "downstream_parsed_output": dict(parsed),
        "downstream_gold_policy": "single_if_forced",
        "downstream_effective_gold_source": "gold_e_ids_single_if_forced",
        "downstream_allow_multiple": bool(cfg.b_llm_allow_multiple),
        "downstream_constrained_decoding": bool(constrained_decoding) if backend == "hf" else None,
    }
    if pred_e_id == "__NONE__":
        pred_e_id = None
    if parsed.get("ok"):
        return pred_e_id, raw_output, None
    err = f"bgroup_llm_{parsed.get('status') or 'parse_failure'}"
    if parsed.get("error_type"):
        err = f"{err}:{parsed.get('error_type')}"
    return None, raw_output, err


def _run_a_group_downstream(*, candidates: list[dict[str, Any]], sentence: str, runtime: dict[str, Any], scorer, cfg: RuleE2EEvalConfig, input_construction_version: str, logger: logging.Logger):
    scored = [dict(c) for c in candidates]
    _score_candidates_with_encoder(
        candidates=scored,
        raw_sentence=sentence,
        context_left="",
        context_right="",
        scorer=scorer,
        scoring_enabled=True,
        batch_size=cfg.candidate_scoring_batch_size,
        logger=logger,
        require_head_logits=True,
        expredict_map=dict(runtime.get("expredict_map") or {}),
        input_construction_version=str(input_construction_version or AGROUP_INPUT_CONSTRUCTION_VERSION_V2).strip() or AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
    )
    scored = _apply_encoder_confidence(
        scored,
        use_sigmoid_prob=True,
        temperature=1.0,
        write_encoder_prob=True,
        encoder_prob_field="encoder_prob",
        encoder_prob_only_when_head_logits=True,
        encoder_scoring_method="head_logits",
    )
    scored.sort(
        key=lambda c: (
            float(c.get("confidence", 0.0) or 0.0),
            float(c.get("encoder_score", 0.0) or 0.0),
            str(c.get("e_id") or ""),
        ),
        reverse=True,
    )
    if not scored:
        return None, {"top1_confidence": None, "n_candidates": 0}, "no_candidates_after_rule_gate"
    top1 = scored[0]
    top1_conf = float(top1.get("confidence", 0.0) or 0.0)
    pred_eid = str(top1.get("e_id") or "").strip() if top1_conf >= float(cfg.a_group_accept_threshold) else None
    raw_output = {
        "top1_e_id": str(top1.get("e_id") or "").strip() or None,
        "top1_confidence": top1_conf,
        "downstream_top1_e_id": str(top1.get("e_id") or "").strip() or None,
        "downstream_top1_confidence": top1_conf,
        "threshold": float(cfg.a_group_accept_threshold),
        "n_candidates": len(scored),
        "downstream_candidate_e_ids": [str(c.get("e_id") or "").strip() for c in scored],
        "downstream_input_text_a": top1.get("encoder_input_text_a"),
        "downstream_input_text_b": top1.get("encoder_input_text_b"),
        "downstream_input_version": top1.get("encoder_input_version") or input_construction_version,
        "downstream_input_text_b_format": top1.get("encoder_input_text_b_format"),
    }
    if pred_eid is None:
        return None, raw_output, f"group_a_below_threshold<{cfg.a_group_accept_threshold:.2f}"
    return pred_eid, raw_output, None


def _run_b_group_encoder_downstream(*, decision, runtime: dict[str, Any], scorer_bundle, cfg: RuleE2EEvalConfig):
    import torch

    example, missing_meta = _build_bgroup_scoring_example(decision, runtime)
    if not (example.get("candidate_inputs") or []):
        return None, {"n_candidates": 0, "missing_candidate_meta": missing_meta}, "bgroup_no_candidate_inputs_after_meta_filter"

    with torch.no_grad():
        scored_batch = _score_bgroup_batch(
            model=scorer_bundle["model"],
            tokenizer=scorer_bundle["tokenizer"],
            scorer=scorer_bundle["scorer"],
            examples=[example],
            max_seq_len=int(cfg.b_group_max_seq_len),
            device=str(scorer_bundle["device"]),
            mixed_precision=str(scorer_bundle["mixed_precision"]),
            freeze_encoder=True,
        )
    if not scored_batch:
        return None, {"n_candidates": len(example.get("candidate_e_ids") or []), "missing_candidate_meta": missing_meta}, "bgroup_empty_scored_batch"

    item = scored_batch[0]
    logits = item["logits_with_none"].detach().float()
    probs = torch.softmax(logits, dim=0).tolist()
    labels = _bgroup_label_space(item["example"])
    pred_index = int(max(range(len(probs)), key=lambda i: probs[i]))
    pred_label = labels[pred_index]
    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    raw_output = {
        "pred_label": pred_label,
        "pred_prob": probs[pred_index] if probs else None,
        "n_candidates": len(example.get("candidate_e_ids") or []),
        "missing_candidate_meta": missing_meta,
        "ranked_labels": ranked,
        "downstream_input_text_a": ((example.get("candidate_inputs") or [{}])[0].get("text_a") if (example.get("candidate_inputs") or []) else None),
        "downstream_input_text_bs": [c.get("text_b") for c in (example.get("candidate_inputs") or [])],
        "downstream_input_text_b_by_eid": {
            str(c.get("candidate_e_id") or ""): c.get("text_b")
            for c in (example.get("candidate_inputs") or [])
            if str(c.get("candidate_e_id") or "").strip()
        },
        "downstream_candidate_e_ids": list(example.get("candidate_e_ids") or []),
        "downstream_label_space": labels,
        "downstream_gold_label_index": int(example.get("label_index") or 0),
        "downstream_pred_label_index": pred_index,
        "downstream_checkpoint_path": str(Path(scorer_bundle.get("checkpoint_dir") or "")),
        "downstream_max_seq_len": int(cfg.b_group_max_seq_len),
        "downstream_input_mode": "bgroup_cross_encoder_pair_v1",
        "downstream_span_marker_style": "[SPAN]...[/SPAN]",
        "downstream_text_b_format": "canonical_form_plus_gloss_plain",
    }
    if pred_label == "__NONE__":
        return None, raw_output, None
    return pred_label, raw_output, None


def _build_bgroup_scoring_example(decision, runtime: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    expredict_map = runtime.get("expredict_map") or {}
    candidate_inputs = []
    candidate_e_ids = []
    missing_meta = []
    normalized_spans = _normalize_span_segments_any(decision.gold_span_segments)

    for eid in decision.rule_candidate_e_ids:
        meta = expredict_map.get(str(eid).strip()) or {}
        canonical_form = str(meta.get("canonical_form") or "").strip()
        gloss = str(meta.get("gloss") or "").strip()
        polyset_id = str(meta.get("polyset_id") or decision.polyset_id or "").strip()
        group = str(meta.get("group") or decision.group or "").strip()
        if not canonical_form:
            missing_meta.append(str(eid))
            continue
        built = build_bgroup_cross_encoder_input(
            {
                "target_sentence": decision.sentence,
                "span_segments": normalized_spans,
                "polyset_id": polyset_id,
                "group_key": decision.example_key,
            },
            {
                "e_id": str(eid),
                "canonical_form": canonical_form,
                "gloss": gloss,
            },
        )
        candidate_e_ids.append(str(eid))
        candidate_inputs.append(
            {
                "candidate_e_id": str(eid),
                "canonical_form": str(built["meta"].get("canonical_form") or ""),
                "gloss": str(built["meta"].get("gloss") or ""),
                "polyset_id": polyset_id,
                "group": group,
                "text_a": built["text_a"],
                "text_b": built["text_b"],
                "meta": built["meta"],
            }
        )

    gold_e_id = str(decision.gold_e_id or "__NONE__")
    label_index = candidate_e_ids.index(gold_e_id) if gold_e_id in candidate_e_ids else len(candidate_e_ids)
    example = {
        "group_key": decision.example_key,
        "polyset_id": str(decision.polyset_id or ""),
        "gold_example_role": str(decision.gold_example_role or ""),
        "target_sentence": decision.sentence,
        "span_segments": normalized_spans,
        "candidate_inputs": candidate_inputs,
        "candidate_e_ids": candidate_e_ids,
        "gold_e_id": gold_e_id,
        "label_index": int(label_index),
    }
    return example, missing_meta


def _write_downstream_prompt_log(path: Path, decisions: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for decision in decisions:
            raw = dict(getattr(decision, "raw_downstream_output", None) or {})
            if not raw:
                continue
            row = {
                "example_key": getattr(decision, "example_key", None),
                "split": getattr(decision, "split", None),
                "group": getattr(decision, "group", None),
                "polyset_id": getattr(decision, "polyset_id", None),
                "gold_e_id": getattr(decision, "gold_e_id", None),
                "gold_example_role": getattr(decision, "gold_example_role", None),
                "rule_gate_status": getattr(decision, "rule_gate_status", None),
                "rule_candidate_e_ids": list(getattr(decision, "rule_candidate_e_ids", []) or []),
                "downstream_mode": getattr(decision, "downstream_mode", None),
                "downstream_messages": raw.get("downstream_messages"),
                "downstream_prompt_text": raw.get("downstream_prompt_text"),
                "downstream_candidate_e_ids": raw.get("downstream_candidate_e_ids"),
                "downstream_candidate_number_to_eid": raw.get("downstream_candidate_number_to_eid"),
                "downstream_generation_config": raw.get("downstream_generation_config"),
                "downstream_model_name_or_path": raw.get("downstream_model_name_or_path"),
                "downstream_checkpoint_path": raw.get("downstream_checkpoint_path"),
                "downstream_backend": raw.get("downstream_backend"),
                "downstream_raw_model_output": raw.get("downstream_raw_model_output"),
                "downstream_parsed_output": raw.get("downstream_parsed_output"),
                "downstream_input_text_a": raw.get("downstream_input_text_a"),
                "downstream_input_text_b": raw.get("downstream_input_text_b"),
                "downstream_input_text_bs": raw.get("downstream_input_text_bs"),
                "downstream_input_text_b_by_eid": raw.get("downstream_input_text_b_by_eid"),
                "downstream_input_version": raw.get("downstream_input_version"),
                "downstream_input_text_b_format": raw.get("downstream_input_text_b_format"),
                "downstream_top1_e_id": raw.get("downstream_top1_e_id"),
                "downstream_top1_confidence": raw.get("downstream_top1_confidence"),
            }
            write_jsonl_line(fh, row)


def _write_polyset_csv(path: Path, metrics: dict[str, Any]) -> None:
    import csv

    rows = []
    for polyset_id, payload in metrics.items():
        row = {"polyset_id": polyset_id}
        row.update(payload)
        rows.append(row)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _implementation_status(mode: str) -> str:
    mode_l = str(mode or "gate_only").strip().lower()
    if mode_l == "a_group":
        return "a_group_v1"
    if mode_l == "b_group_encoder":
        return "b_group_encoder_v1"
    if mode_l == "b_group_llm":
        return "b_group_llm_v1"
    return "gate_only_v1"
