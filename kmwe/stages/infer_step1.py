from __future__ import annotations

import json
import logging
import math
import atexit
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kmwe.core.config_loader import ConfigError
from kmwe.core.run_context import RunContext
from kmwe.data.factory import (
    AGROUP_INPUT_CONSTRUCTION_VERSION,
    AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
    build_agroup_pair_encoder_input,
    format_encoder_input as _build_encoder_input_text,
)
from kmwe.stages import build_silver as silver_loader
from kmwe.stages import validate_dict as validate_dict_loader
from kmwe.utils.input_override import apply_forced_input_jsonl
from kmwe.utils.jsonio import write_json, write_jsonl_line
from kmwe.utils.morph import analyze_with_kiwi


AGROUP_ENCODER_EIDS = {"ece001", "edf003"}


def _new_agroup_layer_counts() -> dict[str, int]:
    return {
        "gold_total": 0,
        "rule_detected": 0,
        "encoder_passed": 0,
        "final_tp": 0,
    }


def _update_agroup_layer_counts(
    counts: dict[str, int],
    *,
    rule_detected: bool,
    encoder_passed: bool,
    final_tp: bool,
) -> None:
    counts["gold_total"] = int(counts.get("gold_total", 0)) + 1
    if rule_detected:
        counts["rule_detected"] = int(counts.get("rule_detected", 0)) + 1
    if encoder_passed:
        counts["encoder_passed"] = int(counts.get("encoder_passed", 0)) + 1
    if final_tp:
        counts["final_tp"] = int(counts.get("final_tp", 0)) + 1


def _agroup_layer_summary(counts: dict[str, int]) -> dict[str, Any]:
    gold_total = max(int(counts.get("gold_total", 0)), 0)
    rule_detected = max(int(counts.get("rule_detected", 0)), 0)
    encoder_passed = max(int(counts.get("encoder_passed", 0)), 0)
    final_tp = max(int(counts.get("final_tp", 0)), 0)
    return {
        "gold_total": gold_total,
        "rule_detected": rule_detected,
        "encoder_passed": encoder_passed,
        "final_tp": final_tp,
        "rule_detected_rate": float(rule_detected / gold_total) if gold_total > 0 else 0.0,
        "encoder_pass_rate_given_gold": float(encoder_passed / gold_total) if gold_total > 0 else 0.0,
        "encoder_pass_rate_given_rule_detected": float(encoder_passed / rule_detected) if rule_detected > 0 else 0.0,
        "final_tp_rate": float(final_tp / gold_total) if gold_total > 0 else 0.0,
        "final_tp_rate_given_encoder_passed": float(final_tp / encoder_passed) if encoder_passed > 0 else 0.0,
    }


def _masked_mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom


def _flatten_for_wandb(
    value: Any,
    *,
    prefix: str = "",
    out: dict[str, Any] | None = None,
    max_depth: int = 6,
) -> dict[str, Any]:
    out = out or {}
    if max_depth < 0:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k)
            new_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_for_wandb(v, prefix=new_prefix, out=out, max_depth=max_depth - 1)
        return out
    if isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix or "value"] = value
        return out
    if isinstance(value, (list, tuple)):
        if len(value) <= 20 and all(isinstance(x, (str, int, float, bool)) or x is None for x in value):
            out[prefix or "value"] = list(value)
        else:
            out[prefix or "value"] = str(value)
        return out
    out[prefix or "value"] = str(value)
    return out


def _maybe_init_wandb(
    cfg: dict[str, Any],
    *,
    stage: str,
    exp_id: str,
    run_id: str,
    logger: logging.Logger,
) -> tuple[Any | None, dict[str, Any]]:
    infer_cfg = cfg.get("infer", {}) or {}
    wandb_cfg_base = cfg.get("wandb", {}) or {}
    wandb_cfg_stage = infer_cfg.get("wandb", {}) or {}
    wandb_cfg = {**wandb_cfg_base, **wandb_cfg_stage}
    if not bool(wandb_cfg.get("enabled", False)):
        return None, {"enabled": False}
    try:
        import wandb  # type: ignore
    except Exception as exc:
        logger.warning("[infer_step1][wandb] import 실패로 비활성화: %s", exc)
        return None, {"enabled": False, "error": str(exc)}

    project = str(wandb_cfg.get("project") or "").strip()
    if not project:
        project = "kmwe-project"
    entity = str(wandb_cfg.get("entity") or wandb_cfg.get("team") or "").strip() or None
    mode = str(wandb_cfg.get("mode") or "online").strip() or "online"
    run_name = str(wandb_cfg.get("run_name") or f"{exp_id}/{stage}/{run_id}").strip()
    raw_tags = wandb_cfg.get("tags")
    tags = [str(x) for x in raw_tags] if isinstance(raw_tags, (list, tuple)) else []
    if stage not in tags:
        tags.append(stage)
    if exp_id not in tags:
        tags.append(exp_id)

    config_flat = _flatten_for_wandb(cfg)
    config_flat.update(
        {
            "stage": stage,
            "exp_id": exp_id,
            "run_id": run_id,
        }
    )
    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=run_name,
        tags=tags,
        config=config_flat,
    )
    finish_hook = None
    if run is not None:
        def _finish_at_exit() -> None:
            try:
                run.finish()
            except Exception:
                return
        finish_hook = _finish_at_exit
        atexit.register(_finish_at_exit)
    logger.info(
        "[infer_step1][wandb] enabled=true project=%s entity=%s mode=%s name=%s",
        project,
        entity or "",
        mode,
        run_name,
    )
    return run, {
        "enabled": True,
        "project": project,
        "entity": entity,
        "mode": mode,
        "run_name": run_name,
        "tags": tags,
        "finish_hook": finish_hook,
    }


def _wandb_log_safe(run: Any | None, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
            continue
        if isinstance(value, dict):
            small = True
            flat_local: dict[str, Any] = {}
            for sub_k, sub_v in value.items():
                if not isinstance(sub_v, (str, int, float, bool)) and sub_v is not None:
                    small = False
                    break
                flat_local[f"{key}.{sub_k}"] = sub_v
            if small:
                payload.update(flat_local)
                continue
        payload[key] = value
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception:
        return

def run_infer_step1(*, cfg: dict[str, Any], run_context: RunContext) -> None:
    logger = logging.getLogger("kmwe")
    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    exp_id = str(run_context.exp_id or cfg.get("exp", {}).get("exp_id") or "")
    run_id = str(run_context.run_id or "")

    input_path, input_path_source, input_path_forced = _resolve_input_path(cfg, run_context, logger)
    if not input_path.exists():
        raise ConfigError(f"infer_step1 입력 JSONL이 존재하지 않습니다: {input_path}")
    _validate_input_jsonl_schema(input_path)

    dict_source, dict_stats, dict_bundle = silver_loader._load_dict_stats(cfg)
    logger.info(
        "infer_step1 dict 로딩 성공: source=%s patterns=%s rules=%s",
        dict_source,
        dict_stats["n_patterns_total"],
        dict_stats["n_rules_total"],
    )

    rule_sets = silver_loader._prepare_stage_rules(
        dict_bundle.get("rules", []),
        allowed_scopes={"", "all", "infer"},
    )
    detect_rules = rule_sets["detect_rules"]
    verify_rules = rule_sets["verify_rules"]
    morph_verify_rules = rule_sets["morph_verify_rules"]
    context_rules = rule_sets["context_rules"]
    ignored_rules = rule_sets["ignored_rules"]
    ignored_verify = rule_sets["ignored_verify"]
    ignored_context = rule_sets["ignored_context"]
    n_verify_rules_skipped_morph_unsupported = rule_sets["n_verify_rules_skipped_morph_unsupported"]
    logger.info("infer_step1 detect 규칙 수: %s (무시됨: %s)", len(detect_rules), ignored_rules)
    logger.info("infer_step1 verify 규칙 수: %s (무시됨: %s)", len(verify_rules), ignored_verify)
    logger.info(
        "infer_step1 verify morph 규칙 수: %s (미지원: %s)",
        len(morph_verify_rules),
        n_verify_rules_skipped_morph_unsupported,
    )
    logger.info("infer_step1 context 규칙 수: %s (무시됨: %s)", len(context_rules), ignored_context)

    output_path = outputs_dir / "infer_candidates.jsonl"
    report_path = outputs_dir / "infer_step1_report.json"

    infer_cfg = cfg.get("infer", {})
    silver_cfg = cfg.get("silver", {})
    uncertainty_cfg = infer_cfg.get("uncertainty", {}) or {}
    postprocess_cfg = infer_cfg.get("postprocess", {}) or {}
    output_cfg = infer_cfg.get("output", {}) or {}
    infer_wandb_cfg = infer_cfg.get("wandb", {}) or {}
    wandb_log_every_n_examples = max(0, int(infer_wandb_cfg.get("log_every_n_examples", 0) or 0))
    wandb_run, wandb_meta = _maybe_init_wandb(
        cfg,
        stage="infer_step1",
        exp_id=exp_id,
        run_id=run_id,
        logger=logger,
    )

    triage_cfg = infer_cfg.get("triage_thresholds") or silver_cfg.get("triage_thresholds") or {}
    confirm_min_score = int(triage_cfg.get("confirm_min_score", 3))
    hold_min_score = int(triage_cfg.get("hold_min_score", 1))

    detect_window_chars = int(
        infer_cfg.get("detect_match_window_chars", silver_cfg.get("detect_match_window_chars", 60))
    )
    detect_max_matches_per_rule = int(
        infer_cfg.get("detect_max_matches_per_rule", silver_cfg.get("detect_max_matches_per_rule", 50))
    )
    morph_window_chars = int(
        infer_cfg.get("morph_window_chars", (silver_cfg.get("morph", {}) or {}).get("window_chars", 80))
    )
    verify_window_chars = int(cfg.get("verify", {}).get("window_chars", morph_window_chars))
    logger.info(
        "[infer][verify_window] window_chars=%s (fallback=morph_window_chars=%s)",
        verify_window_chars,
        morph_window_chars,
    )
    context_window_chars = int(
        infer_cfg.get("context_window_chars", silver_cfg.get("context_window_chars", 40))
    )

    morph_enabled_cfg = bool(silver_cfg.get("morph", {}).get("enabled", False))
    morph_enabled = morph_enabled_cfg or (len(morph_verify_rules) > 0)
    if morph_enabled and (not morph_enabled_cfg) and (len(morph_verify_rules) > 0):
        logger.info("infer_step1: silver.morph.enabled=false 이지만 morph_verify_rules가 있어 Kiwi 분석을 활성화합니다.")
    include_morph_tokens = bool(infer_cfg.get("include_morph_tokens", False))
    candidate_scoring_cfg = infer_cfg.get("candidate_scoring", {}) or {}
    candidate_scoring_enabled = bool(candidate_scoring_cfg.get("enabled", True))
    candidate_scoring_batch_size = int(candidate_scoring_cfg.get("batch_size", 32))
    candidate_scoring_max_seq_len = int(candidate_scoring_cfg.get("max_seq_len", 256))
    agroup_input_construction_version = str(
        (cfg.get("finetune", {}) or {}).get("input_construction_version")
        or AGROUP_INPUT_CONSTRUCTION_VERSION
    ).strip() or AGROUP_INPUT_CONSTRUCTION_VERSION
    group_a_require_head_logits = bool(infer_cfg.get("group_a_require_head_logits", True))
    group_a_disable_fallback_scoring = bool(infer_cfg.get("group_a_disable_fallback_scoring", True))
    if group_a_require_head_logits and not candidate_scoring_enabled:
        raise ConfigError(
            "infer_step1 A-group 연구 모드에서는 candidate_scoring.enabled=true가 필수입니다."
        )
    scorer = _build_encoder_scorer(
        cfg=cfg,
        run_context=run_context,
        enabled=candidate_scoring_enabled,
        max_seq_len=candidate_scoring_max_seq_len,
        logger=logger,
        require_head_logits=group_a_require_head_logits,
        disallow_fallback_scoring=group_a_disable_fallback_scoring,
    )

    uncertainty_enabled = bool(uncertainty_cfg.get("enabled", True))
    margin_policy = str(uncertainty_cfg.get("margin_policy", "span_polyset_top1_vs_top2"))
    margin_threshold = float(uncertainty_cfg.get("margin_threshold", 0.10))
    low_conf_threshold = float(uncertainty_cfg.get("low_conf_threshold", 0.55))
    group_a_accept_threshold = float(infer_cfg.get("group_a_accept_threshold", low_conf_threshold))
    use_sigmoid_prob = bool(uncertainty_cfg.get("use_sigmoid_prob", True))
    temperature = float(uncertainty_cfg.get("temperature", 1.0))
    write_encoder_prob = bool(output_cfg.get("write_encoder_prob", True))
    write_dropped_candidates = bool(output_cfg.get("write_dropped_candidates", True))
    encoder_prob_field = str(output_cfg.get("encoder_prob_field", "encoder_prob"))
    encoder_prob_only_when_head_logits = bool(
        output_cfg.get("encoder_prob_only_when_head_logits", True)
    )
    encoder_scoring_method = scorer.scoring_method if scorer is not None else "missing"
    head_loaded = bool(scorer is not None and getattr(scorer, "head", None) is not None)
    logger.info(
        "[infer_step1][agroup_scoring] require_head_logits=%s disable_fallback=%s scoring_method=%s head_loaded=%s group_a_accept_threshold=%.4f",
        group_a_require_head_logits,
        group_a_disable_fallback_scoring,
        encoder_scoring_method,
        head_loaded,
        group_a_accept_threshold,
    )
    encoder_score_source_hint = str(
        uncertainty_cfg.get("encoder_score_source") or infer_cfg.get("encoder_score_source") or ""
    ).strip().lower()

    postprocess_enabled = bool(postprocess_cfg.get("enabled", True))
    nms_cfg = postprocess_cfg.get("nms", {}) or {}
    nms_scope = str(nms_cfg.get("scope", "same_eid_or_polyset"))
    nms_metric = str(nms_cfg.get("metric", "char_iou"))
    nms_iou_threshold = float(nms_cfg.get("iou_threshold", 0.60))
    nms_short_span_len_le = int(nms_cfg.get("short_span_len_le", 4))
    nms_short_span_min_overlap_ratio = float(nms_cfg.get("short_span_min_overlap_ratio", 0.90))
    nms_tie_breaker = str(nms_cfg.get("tie_breaker", "score_then_longer_then_span_key"))
    polyset_competition = bool(postprocess_cfg.get("polyset_competition", True))
    ambiguous_only_polyset_topk = bool(postprocess_cfg.get("ambiguous_only_polyset_topk", True))
    polyset_topk_when_ambiguous = int(postprocess_cfg.get("polyset_topk_when_ambiguous", 2))

    logger.info(
        "infer_step1 routing: auto_confirm=LLM 미전송(최종 DecisionLine 확정 아님), to_llm=LLM 대상"
    )
    logger.info(
        "infer_step1 uncertainty 설정: enabled=%s policy=%s margin_threshold=%.4f low_conf_threshold=%.4f use_sigmoid_prob=%s temperature=%.4f",
        uncertainty_enabled,
        margin_policy,
        margin_threshold,
        low_conf_threshold,
        use_sigmoid_prob,
        temperature,
    )
    logger.info(
        "ambiguous_thresholds_used: margin=%.4f low_conf=%.4f use_sigmoid_prob=%s temperature=%.4f",
        margin_threshold,
        low_conf_threshold,
        use_sigmoid_prob,
        temperature,
    )
    if encoder_score_source_hint:
        logger.info(
            "infer_step1 encoder_score_source 힌트: %s (finetuned|stub|missing)",
            encoder_score_source_hint,
        )

    pos_mapper, _ = silver_loader._load_pos_mapping(cfg=cfg, run_context=run_context, logger=logger)
    expredict_map, expredict_rebuilt = _ensure_expredict_map(dict_bundle, logger)

    components_by_eid = silver_loader._index_components_by_eid(dict_bundle.get("components", []))

    n_sents = 0
    n_input_candidates_total = 0
    n_candidates_total = 0
    n_hard_fail = 0
    candidates_by_eid: Counter[str] = Counter()
    n_groups_span_polyset_total = 0
    n_ambiguous_groups_total = 0
    kept_top1_groups_total = 0
    kept_topk_groups_total = 0
    n_to_llm_candidates_total = 0
    n_auto_confirm_candidates_total = 0
    n_group_a_dropped_total = 0
    n_ambiguous_candidates_total = 0
    agroup_layers_total = _new_agroup_layer_counts()
    agroup_layers_by_eid: dict[str, dict[str, int]] = {
        eid: _new_agroup_layer_counts() for eid in sorted(AGROUP_ENCODER_EIDS)
    }
    dropped_reason_counts: Counter[str] = Counter()
    dropped_by_eid: Counter[str] = Counter()
    encoder_score_missing_total = 0
    encoder_score_source_counts: Counter[str] = Counter()
    no_candidate_samples: list[dict[str, Any]] = []
    no_candidate_counts: Counter[str] = Counter()
    no_candidate_top_samples: dict[str, list[list[Any]]] = defaultdict(list)
    locate_fail_hint_from_samples_used = 0
    locate_fail_hint_from_samples_total = 0
    nms_state = {
        "did_log_scope": False,
        "requested": nms_scope,
        "effective": nms_scope,
        "calls": 0,
        "dropped": 0,
    }

    def _norm_or_none(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def _extract_e_id(record: dict[str, Any]) -> Any:
        for key in ("e_id", "gold_eid", "target_eid"):
            value = record.get(key)
            if value is None and isinstance(record.get("meta"), dict):
                value = record["meta"].get(key)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
            value = _norm_or_none(value)
            if value is not None:
                return value
        return None

    def _classify_no_candidate(
        *,
        detect_result: dict[str, Any],
        verify_result: dict[str, Any],
        detect_candidates_len: int,
        final_candidates_len: int,
    ) -> tuple[str, str | None, str | None]:
        detect_rules_with_any = int(detect_result.get("n_detect_rules_with_any_match") or 0)
        detect_regex_spans = int(detect_result.get("n_detect_regex_match_spans") or 0)
        detect_candidates_total = int(detect_result.get("n_candidates_total") or 0)
        components_span_fail = int(detect_result.get("n_components_span_fail") or 0)
        hard_fail_all = int(verify_result.get("n_candidates_discarded_by_hard_fail") or 0)
        if detect_rules_with_any == 0 and detect_regex_spans == 0:
            return (
                "detect_no_hit",
                "detect_rules_with_any_match=0",
                "infer_candidates.jsonl#debug.detect.n_detect_rules_with_any_match",
            )
        if detect_rules_with_any > 0 and detect_candidates_total == 0 and components_span_fail > 0:
            return (
                "locate_components_fail",
                "detect matched but components span failed",
                "infer_candidates.jsonl#debug.detect.n_components_span_fail",
            )
        if detect_candidates_len > 0 and final_candidates_len == 0 and hard_fail_all >= detect_candidates_len:
            return (
                "verify_hard_fail_all",
                "verify_hard_fail consumed all candidates",
                "infer_candidates.jsonl#debug.verify.n_candidates_discarded_by_hard_fail",
            )
        return ("unknown", None, None)

    with input_path.open("r", encoding="utf-8") as fp, output_path.open("w", encoding="utf-8") as out_fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_sentence = record.get("raw_sentence") or record.get("target_sentence")
            if not raw_sentence:
                continue
            n_sents += 1

            morph_tokens_in = record.get("morph_tokens")
            morph_tokens_span = None
            if isinstance(morph_tokens_in, list):
                morph_tokens_span = morph_tokens_in
            elif morph_enabled:
                try:
                    kiwi_model = silver_cfg.get("morph", {}).get("kiwi_model", "cong-global")
                    morph_tokens_span = analyze_with_kiwi(raw_sentence, model=kiwi_model)
                except Exception as exc:
                    logger.warning("infer_step1 morph 분석 실패: %s", exc, exc_info=True)
                    morph_tokens_span = []
            if isinstance(morph_tokens_span, list):
                for token in morph_tokens_span:
                    token["pos_std"] = pos_mapper(str(token.get("pos", "")))

            morph_tokens_verify = morph_tokens_span if morph_enabled else None

            detect_result = silver_loader._detect_candidates(
                raw_sentence,
                detect_rules,
                expredict_map,
                confirm_min_score=confirm_min_score,
                hold_min_score=hold_min_score,
                **silver_loader._build_detect_kwargs(
                    record=record,
                    raw_sentence=raw_sentence,
                    components_by_eid=components_by_eid,
                    morph_tokens=morph_tokens_span,
                    detect_match_window_chars=detect_window_chars,
                    detect_max_matches_per_rule=detect_max_matches_per_rule,
                    include_debug_ctx=True,
                ),
            )
            candidates = detect_result["candidates"]
            detect_candidates_len = len(candidates)

            verify_result = silver_loader._apply_verify_rules(
                raw_sentence=raw_sentence,
                candidates=candidates,
                rules=verify_rules,
                morph_rules=morph_verify_rules,
                morph_tokens=morph_tokens_verify,
                confirm_min_score=confirm_min_score,
                hold_min_score=hold_min_score,
                morph_window_chars=morph_window_chars,
                verify_window_chars=verify_window_chars,
            )
            context_result = silver_loader._apply_context_rules(
                raw_sentence=raw_sentence,
                candidates=candidates,
                rules=context_rules,
                confirm_min_score=confirm_min_score,
                hold_min_score=hold_min_score,
                context_window_chars=context_window_chars,
            )
            # build_silver와 동일 정책: 동일 span_key 경쟁 시 confirm 금지(강등)
            silver_loader._apply_span_competition_guard(
                candidates,
                context_result["triage_transition_counts"],
            )

            _attach_pattern_metadata(candidates, expredict_map)
            scoring_candidates = [
                cand for cand in candidates if str(cand.get("group") or "").strip().lower() == "a"
            ]
            _score_candidates_with_encoder(
                candidates=scoring_candidates,
                raw_sentence=raw_sentence,
                context_left=str(record.get("context_left") or ""),
                context_right=str(record.get("context_right") or ""),
                scorer=scorer,
                scoring_enabled=candidate_scoring_enabled,
                batch_size=candidate_scoring_batch_size,
                logger=logger,
                require_head_logits=group_a_require_head_logits,
                expredict_map=expredict_map,
                input_construction_version=agroup_input_construction_version,
            )

            n_input_candidates_total += len(candidates)
            candidates, dropped_candidates, post_stats, post_meta = _postprocess_candidates(
                candidates=candidates,
                expredict_map=expredict_map,
                uncertainty_enabled=uncertainty_enabled,
                margin_policy=margin_policy,
                margin_threshold=margin_threshold,
                low_conf_threshold=low_conf_threshold,
                group_a_accept_threshold=group_a_accept_threshold,
                use_sigmoid_prob=use_sigmoid_prob,
                temperature=temperature,
                postprocess_enabled=postprocess_enabled,
                nms_scope=nms_scope,
                nms_metric=nms_metric,
                nms_iou_threshold=nms_iou_threshold,
                nms_short_span_len_le=nms_short_span_len_le,
                nms_short_span_min_overlap_ratio=nms_short_span_min_overlap_ratio,
                nms_tie_breaker=nms_tie_breaker,
                polyset_competition=polyset_competition,
                ambiguous_only_polyset_topk=ambiguous_only_polyset_topk,
                polyset_topk_when_ambiguous=polyset_topk_when_ambiguous,
                write_encoder_prob=write_encoder_prob,
                encoder_prob_field=encoder_prob_field,
                encoder_prob_only_when_head_logits=encoder_prob_only_when_head_logits,
                encoder_scoring_method=encoder_scoring_method,
                encoder_score_source_hint=encoder_score_source_hint,
                nms_state=nms_state,
                logger=logger,
            )

            n_groups_span_polyset_total += post_stats["n_groups_span_polyset"]
            n_ambiguous_groups_total += post_stats["n_ambiguous_groups"]
            kept_top1_groups_total += post_stats["kept_top1_groups"]
            kept_topk_groups_total += post_stats["kept_topk_groups"]
            n_to_llm_candidates_total += post_stats["n_to_llm_candidates"]
            n_auto_confirm_candidates_total += post_stats["n_auto_confirm_candidates"]
            n_group_a_dropped_total += len(dropped_candidates)
            for dropped in dropped_candidates:
                dropped_reason_counts[str(dropped.get("routing_reason") or "unknown")] += 1
                dropped_eid = str(dropped.get("e_id") or "")
                if dropped_eid:
                    dropped_by_eid[dropped_eid] += 1
            n_ambiguous_candidates_total += post_stats["n_ambiguous_candidates"]
            encoder_score_missing_total += post_stats["encoder_score_missing_count"]
            for source_key, count in post_stats["encoder_score_source_counts"].items():
                encoder_score_source_counts[source_key] += count

            agroup_target_eid = None
            agroup_rule_detected = False
            agroup_encoder_passed = False
            agroup_final_tp = False
            target_eid_value = _extract_e_id(record)
            if target_eid_value is not None:
                target_eid_text = str(target_eid_value).strip()
                if target_eid_text in AGROUP_ENCODER_EIDS:
                    agroup_target_eid = target_eid_text
                    detect_candidates_raw = detect_result.get("candidates") or []
                    agroup_rule_detected = any(
                        str(c.get("e_id") or "").strip() == agroup_target_eid
                        for c in detect_candidates_raw
                    )
                    agroup_final_tp = any(
                        str(c.get("e_id") or "").strip() == agroup_target_eid
                        and bool(c.get("auto_confirm"))
                        for c in candidates
                    )
                    agroup_encoder_passed = agroup_final_tp or any(
                        str(c.get("e_id") or "").strip() == agroup_target_eid
                        and str(c.get("routing_reason") or "") == "nms_drop_after_group_a_accept"
                        for c in dropped_candidates
                    )
                    _update_agroup_layer_counts(
                        agroup_layers_total,
                        rule_detected=agroup_rule_detected,
                        encoder_passed=agroup_encoder_passed,
                        final_tp=agroup_final_tp,
                    )
                    _update_agroup_layer_counts(
                        agroup_layers_by_eid[agroup_target_eid],
                        rule_detected=agroup_rule_detected,
                        encoder_passed=agroup_encoder_passed,
                        final_tp=agroup_final_tp,
                    )

            for candidate in candidates:
                n_candidates_total += 1
                eid = str(candidate.get("e_id", ""))
                if eid:
                    candidates_by_eid[eid] += 1
                if candidate.get("hard_fail_triggered"):
                    n_hard_fail += 1

            if len(candidates) == 0:
                example_id = _norm_or_none(record.get("example_id"))
                e_id = _extract_e_id(record)
                split_value = _norm_or_none(record.get("split"))
                if split_value is None and isinstance(record.get("meta"), dict):
                    split_value = _norm_or_none(record["meta"].get("split"))
                if example_id is not None or e_id is not None:
                    subbucket, why, debug_ptr = _classify_no_candidate(
                        detect_result=detect_result,
                        verify_result=verify_result,
                        detect_candidates_len=detect_candidates_len,
                        final_candidates_len=len(candidates),
                    )
                    locate_fail_hint = ""
                    if subbucket == "locate_components_fail":
                        locate_fail_hint_from_samples_total += 1
                        fail_samples = detect_result.get("detect_components_span_fail_samples") or []
                        if isinstance(fail_samples, list) and fail_samples:
                            debug_sample = None
                            for item in fail_samples:
                                if str(item.get("e_id") or "") == str(e_id or ""):
                                    debug_sample = item
                                    break
                            if debug_sample is None:
                                debug_sample = fail_samples[0]

                            def _summarize_fail_sample(sample: dict[str, Any]) -> list[str]:
                                parts: list[str] = []
                                if sample.get("e_id"):
                                    parts.append(f"e_id={sample.get('e_id')}")
                                if sample.get("rule_id"):
                                    parts.append(f"rule_id={sample.get('rule_id')}")
                                if sample.get("note"):
                                    parts.append(f"note={sample.get('note')}")
                                if sample.get("failed_required_comp_ids"):
                                    parts.append(
                                        f"required_comp_missing={sample.get('failed_required_comp_ids')}"
                                    )
                                    parts.append("required_comp_missing")
                                if sample.get("gap_violations"):
                                    parts.append(f"gap_violations={sample.get('gap_violations')}")
                                    parts.append("gap_out_of_bounds")
                                if "anchor_selected_kind" in sample and sample.get(
                                    "anchor_selected_kind"
                                ) is None:
                                    parts.append("anchor_not_found")
                                special_counts = sample.get("special_candidate_counts") or {}
                                if isinstance(special_counts, dict):
                                    for key in special_counts.keys():
                                        key_l = str(key).lower()
                                        if "bridge" in key_l:
                                            parts.append("special_generated_but_dropped")
                                            if "thing" in key_l:
                                                parts.append("thing_bridge")
                                            if "jong" in key_l:
                                                parts.append("jong_bridge")
                                        if "adnominal" in key_l:
                                            parts.append("special_generated_but_dropped")
                                            parts.append("adnominal")
                                        if "nde" in key_l:
                                            parts.append("special_generated_but_dropped")
                                            parts.append("nde")
                                return parts

                            parts = _summarize_fail_sample(debug_sample)
                            if debug_sample is fail_samples[0] and len(fail_samples) > 1:
                                parts.extend(["alt_sample"])
                                parts.extend(_summarize_fail_sample(fail_samples[1]))
                            locate_fail_hint = " ".join([str(p) for p in parts if p])
                        if locate_fail_hint:
                            locate_fail_hint_from_samples_used += 1
                        else:
                            combined_text = f"{why or ''} {debug_ptr or ''}".strip()
                            if combined_text:
                                locate_fail_hint = combined_text
                        if len(locate_fail_hint) > 200:
                            locate_fail_hint = locate_fail_hint[:200]
                    sample = {
                        "example_id": example_id,
                        "e_id": _norm_or_none(e_id),
                        "split": split_value,
                        "subbucket": subbucket,
                        "why": _norm_or_none(why),
                        "debug_ptr": _norm_or_none(debug_ptr),
                        "locate_fail_hint": locate_fail_hint,
                    }
                    no_candidate_samples.append(sample)
                    no_candidate_counts[subbucket] += 1
                    if len(no_candidate_top_samples[subbucket]) < 20:
                        no_candidate_top_samples[subbucket].append(
                            [
                                str(example_id) if example_id is not None else None,
                                str(e_id) if e_id is not None else None,
                            ]
                        )

            output_record = {
                "doc_id": record.get("doc_id"),
                "sent_index": record.get("sent_index"),
                "example_id": record.get("example_id"),
                "instance_id": record.get("instance_id"),
                "target_sentence": raw_sentence,
                "candidates": candidates,
                "debug": {
                    "detect": {k: v for k, v in detect_result.items() if k != "candidates"},
                    "verify": verify_result,
                    "context": context_result,
                    "span_competition_guard": {
                        "enabled": True,
                        "policy": "no_confirm_if_same_span_key",
                    },
                },
            }
            if include_morph_tokens and morph_tokens_span is not None:
                output_record["morph_tokens"] = morph_tokens_span
            if write_dropped_candidates and dropped_candidates:
                output_record["dropped_candidates"] = dropped_candidates

            write_jsonl_line(out_fp, output_record)
            if wandb_log_every_n_examples > 0 and n_sents % wandb_log_every_n_examples == 0:
                _wandb_log_safe(
                    wandb_run,
                    {
                        "progress/examples_seen": n_sents,
                        "progress/input_candidates_total": n_input_candidates_total,
                        "progress/output_candidates_total": n_candidates_total,
                        "progress/no_candidate_total": len(no_candidate_samples),
                        "progress/hard_fail_total": n_hard_fail,
                    },
                    step=n_sents,
                )

    report = {
        "input_path": str(input_path),
        "input_path_source": input_path_source,
        "input_path_forced": input_path_forced,
        "output_path": str(output_path),
        "n_sents": n_sents,
        "n_input_candidates_total": n_input_candidates_total,
        "n_candidates_total": n_candidates_total,
        "n_hard_fail": n_hard_fail,
        "encoder_score_missing_count": encoder_score_missing_total,
        "encoder_score_source_counts": dict(encoder_score_source_counts),
        "encoder_scoring_method": encoder_scoring_method,
        "agroup_scoring": {
            "require_head_logits": group_a_require_head_logits,
            "disable_fallback_scoring": group_a_disable_fallback_scoring,
            "head_loaded": head_loaded,
            "scoring_method": encoder_scoring_method,
            "group_a_accept_threshold": group_a_accept_threshold,
        },
        "agroup_analysis": {
            "overall": _agroup_layer_summary(agroup_layers_total),
            "by_eid": {
                eid: _agroup_layer_summary(counts)
                for eid, counts in sorted(agroup_layers_by_eid.items())
            },
        },
        "output": {
            "write_encoder_prob": write_encoder_prob,
            "write_dropped_candidates": write_dropped_candidates,
            "encoder_prob_field": encoder_prob_field,
            "encoder_prob_only_when_head_logits": encoder_prob_only_when_head_logits,
        },
        "routing": {
            "n_groups_span_polyset": n_groups_span_polyset_total,
            "n_ambiguous_groups": n_ambiguous_groups_total,
            "kept_top1_groups": kept_top1_groups_total,
            "kept_topk_groups": kept_topk_groups_total,
            "n_to_llm_candidates": n_to_llm_candidates_total,
            "n_auto_confirm_candidates": n_auto_confirm_candidates_total,
            "n_group_a_dropped": n_group_a_dropped_total,
            "n_ambiguous_candidates": n_ambiguous_candidates_total,
            "dropped_reason_counts": dict(dropped_reason_counts),
            "dropped_by_eid": dict(dropped_by_eid),
        },
        "uncertainty": {
            "enabled": uncertainty_enabled,
            "margin_policy": margin_policy,
            "margin_threshold": margin_threshold,
            "low_conf_threshold": low_conf_threshold,
            "use_sigmoid_prob": use_sigmoid_prob,
            "temperature": temperature,
        },
        "expredict_map_rebuilt": expredict_rebuilt,
        "n_candidates_by_eid_topN": candidates_by_eid.most_common(
            int(cfg.get("infer", {}).get("report_topn_eid", 20))
        ),
    }
    write_json(report_path, report, indent=2)
    no_candidate_breakdown_path = outputs_dir / "no_candidate_breakdown.json"
    no_candidate_samples_path = outputs_dir / "no_candidate_samples.jsonl"
    total_no_candidate = len(no_candidate_samples)
    subbucket_sum = sum(no_candidate_counts.values())
    if subbucket_sum != total_no_candidate:
        raise ConfigError(
            "no_candidate 리포트: subbucket 합이 total_no_candidate와 다릅니다. "
            f"sum={subbucket_sum} total={total_no_candidate}"
        )
    breakdown = {
        "total_no_candidate": total_no_candidate,
        "subbuckets": dict(no_candidate_counts),
        "top_samples": dict(no_candidate_top_samples),
    }
    try:
        write_json(no_candidate_breakdown_path, breakdown, indent=2)
    except Exception as exc:
        raise ConfigError(f"no_candidate_breakdown.json 쓰기 실패: {exc}") from exc
    try:
        with no_candidate_samples_path.open("w", encoding="utf-8") as fp:
            for row in no_candidate_samples:
                write_jsonl_line(fp, row)
    except Exception as exc:
        raise ConfigError(f"no_candidate_samples.jsonl 쓰기 실패: {exc}") from exc
    try:
        with no_candidate_samples_path.open("r", encoding="utf-8") as fp:
            lines_written = sum(1 for _ in fp)
    except Exception as exc:
        raise ConfigError(f"no_candidate_samples.jsonl 검증 실패: {exc}") from exc
    if lines_written != total_no_candidate:
        raise ConfigError(
            "no_candidate 리포트: total_no_candidate와 jsonl lines가 다릅니다. "
            f"total={total_no_candidate} lines={lines_written}"
        )
    breakdown_bytes = no_candidate_breakdown_path.stat().st_size
    logger.info(
        "[infer_step1][no_candidate_report] wrote breakdown=%s bytes=%s total_no_candidate=%s",
        no_candidate_breakdown_path,
        breakdown_bytes,
        total_no_candidate,
    )
    logger.info(
        "[infer_step1][no_candidate_report] wrote samples=%s lines=%s",
        no_candidate_samples_path,
        lines_written,
    )

    locate_breakdown_path = outputs_dir / "infer_step1_locate_components_fail_breakdown.json"
    locate_samples_path = outputs_dir / "infer_step1_locate_components_fail_samples.jsonl"
    locate_reason_counts: Counter[str] = Counter()
    locate_eid_counts: Counter[str] = Counter()
    locate_samples: list[dict[str, Any]] = []

    def _reason_code_from_sample(sample: dict[str, Any]) -> str:
        text = (
            f"{sample.get('why') or ''} "
            f"{sample.get('debug_ptr') or ''} "
            f"{sample.get('locate_fail_hint') or ''}"
        ).lower()
        if "gap_out_of_bounds" in text:
            return "gap_out_of_bounds"
        if "required_comp_missing" in text:
            return "required_comp_missing"
        if "anchor_not_found" in text:
            return "anchor_not_found"
        if "special_generated_but_dropped" in text:
            return "special_generated_but_dropped"
        if "gap" in text and (
            "out_of_bounds" in text
            or "min_gap" in text
            or "max_gap" in text
            or "gap constraint" in text
            or "gap failed" in text
        ):
            return "gap_out_of_bounds"
        if (
            "required" in text
            or "missing required" in text
            or "required_comp" in text
            or ("is_required" in text and "missing" in text)
        ):
            return "required_comp_missing"
        if "no candidates" in text or "candidate=0" in text or "empty candidates" in text:
            return "no_candidates"
        if "all_gap_failed" in text or "all gap failed" in text:
            return "all_gap_failed"
        if ("anchor" in text and ("not found" in text or "missing" in text)) or (
            "anchor_rank" in text and ("0" in text or "none" in text)
        ):
            return "anchor_not_found"
        if (
            ("thing_bridge" in text or "jong_bridge" in text or "adnominal" in text or "nde" in text)
            and ("dropped" in text or "filtered" in text or "not selected" in text)
        ):
            return "special_generated_but_dropped"
        if "no candidate" in text or "no_candidates" in text:
            return "no_candidates"
        return "unknown"

    try:
        with no_candidate_samples_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if sample.get("subbucket") != "locate_components_fail":
                    continue
                reason_code = _reason_code_from_sample(sample)
                reason_text = str(sample.get("why") or sample.get("debug_ptr") or "")
                if len(reason_text) > 200:
                    reason_text = reason_text[:200]
                locate_fail_hint = sample.get("locate_fail_hint")
                if locate_fail_hint is None:
                    locate_fail_hint = ""
                row = {
                    "example_id": sample.get("example_id"),
                    "instance_id": sample.get("instance_id"),
                    "e_id": sample.get("e_id"),
                    "span_key": sample.get("span_key"),
                    "target_sentence": sample.get("target_sentence"),
                    "subbucket": "locate_components_fail",
                    "reason_code": reason_code,
                    "reason_detail": reason_text,
                    "why": sample.get("why") or "",
                    "debug_ptr": sample.get("debug_ptr") or "",
                    "locate_fail_hint": locate_fail_hint,
                }
                locate_samples.append(row)
                locate_reason_counts[reason_code] += 1
                eid_value = sample.get("e_id")
                if eid_value is not None and str(eid_value).strip() != "":
                    locate_eid_counts[str(eid_value)] += 1
    except Exception as exc:
        raise ConfigError(f"locate_components_fail read failed: {exc}") from exc

    locate_total = len(locate_samples)
    top_eids = locate_eid_counts.most_common(20)
    breakdown_payload = {
        "total_locate_components_fail": locate_total,
        "reason_code_counts": dict(locate_reason_counts),
        "top_eids": [[eid, count] for eid, count in top_eids],
    }
    try:
        write_json(locate_breakdown_path, breakdown_payload, indent=2)
    except Exception as exc:
        raise ConfigError(
            f"infer_step1_locate_components_fail_breakdown write failed: {exc}"
        ) from exc
    try:
        with locate_samples_path.open("w", encoding="utf-8") as fp:
            if locate_samples:
                for row in locate_samples:
                    write_jsonl_line(fp, row)
            else:
                fp.write("\n")
    except Exception as exc:
        raise ConfigError(
            f"infer_step1_locate_components_fail_samples write failed: {exc}"
        ) from exc

    locate_breakdown_bytes = locate_breakdown_path.stat().st_size
    locate_samples_lines = len(locate_samples)
    unknown_after_norm = locate_reason_counts.get("unknown", 0)
    unknown_ratio = (unknown_after_norm / locate_total) if locate_total > 0 else 0.0
    hint_text_non_empty = sum(
        1 for row in locate_samples if str(row.get("locate_fail_hint") or "").strip()
    )
    hint_text_empty = locate_total - hint_text_non_empty
    logger.info(
        "[infer_step1][locate_components_fail] wrote breakdown=%s bytes=%s total=%s unique_reason_codes=%s",
        locate_breakdown_path,
        locate_breakdown_bytes,
        locate_total,
        len(locate_reason_counts),
    )
    logger.info(
        "[infer_step1][locate_components_fail] wrote samples=%s lines=%s",
        locate_samples_path,
        locate_samples_lines,
    )
    top_reason_codes = locate_reason_counts.most_common(3)
    logger.info(
        "[infer_step1][locate_components_fail] top_reason_codes=%s",
        top_reason_codes,
    )
    logger.info(
        "[infer_step1][locate_components_fail] hint_text_non_empty=%s total=%s empty=%s",
        hint_text_non_empty,
        locate_total,
        hint_text_empty,
    )
    logger.info(
        "[infer_step1][locate_components_fail] hint_source=span_fail_samples used=%s total=%s",
        locate_fail_hint_from_samples_used,
        locate_fail_hint_from_samples_total,
    )
    logger.info(
        "[infer_step1][locate_components_fail] reason_code_counts_after_norm=%s",
        dict(locate_reason_counts),
    )
    logger.info(
        "[infer_step1][locate_components_fail] unknown_after_norm=%s total=%s unknown_ratio=%.4f",
        unknown_after_norm,
        locate_total,
        unknown_ratio,
    )

    no_candidate_detect_no_hit_path = outputs_dir / "no_candidate_detect_no_hit_by_eid.json"
    if not no_candidate_samples_path.exists():
        raise ConfigError(f"no_candidate_samples.jsonl not found: {no_candidate_samples_path}")
    detect_no_hit_total = 0
    detect_no_hit_by_eid: Counter[str] = Counter()
    detect_no_hit_examples_by_eid: dict[str, list[str]] = defaultdict(list)
    try:
        with no_candidate_samples_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("subbucket") != "detect_no_hit":
                    continue
                detect_no_hit_total += 1
                e_id = row.get("e_id")
                if e_id is None or (isinstance(e_id, str) and e_id.strip() == ""):
                    continue
                e_id_key = str(e_id)
                detect_no_hit_by_eid[e_id_key] += 1
                example_id = row.get("example_id")
                if example_id is None or (isinstance(example_id, str) and example_id.strip() == ""):
                    continue
                examples = detect_no_hit_examples_by_eid[e_id_key]
                if len(examples) < 20:
                    examples.append(str(example_id))
    except Exception as exc:
        raise ConfigError(f"no_candidate_detect_no_hit_by_eid read failed: {exc}") from exc

    top_eids = detect_no_hit_by_eid.most_common(30)
    payload = {
        "total_detect_no_hit": detect_no_hit_total,
        "top_eids": [[eid, count] for eid, count in top_eids],
        "top_examples_by_eid": detect_no_hit_examples_by_eid,
    }
    try:
        write_json(no_candidate_detect_no_hit_path, payload, indent=2)
    except Exception as exc:
        raise ConfigError(f"no_candidate_detect_no_hit_by_eid write failed: {exc}") from exc
    detect_no_hit_bytes = no_candidate_detect_no_hit_path.stat().st_size
    logger.info(
        "[infer_step1][no_candidate_report] wrote no_candidate_detect_no_hit_by_eid.json bytes=%s total_detect_no_hit=%s top_eids=%s",
        detect_no_hit_bytes,
        detect_no_hit_total,
        len(top_eids),
    )
    logger.info(
        "infer_step1 ambiguity routing: n_groups_span_polyset=%s n_ambiguous_groups=%s kept_top1_groups=%s kept_topk_groups=%s",
        n_groups_span_polyset_total,
        n_ambiguous_groups_total,
        kept_top1_groups_total,
        kept_topk_groups_total,
    )
    logger.info(
        "infer_step1 routing candidates: n_to_llm_candidates=%s n_auto_confirm_candidates=%s n_ambiguous_candidates=%s",
        n_to_llm_candidates_total,
        n_auto_confirm_candidates_total,
        n_ambiguous_candidates_total,
    )
    logger.info(
        "infer_step1 encoder_score_source_counts=%s encoder_score_missing_count=%s n_input_candidates_total=%s",
        dict(encoder_score_source_counts),
        encoder_score_missing_total,
        n_input_candidates_total,
    )
    logger.info(
        "[infer_step1][agroup] overall=%s",
        json.dumps(_agroup_layer_summary(agroup_layers_total), ensure_ascii=False),
    )
    logger.info(
        "[infer_step1][agroup] by_eid=%s",
        json.dumps(
            {eid: _agroup_layer_summary(counts) for eid, counts in sorted(agroup_layers_by_eid.items())},
            ensure_ascii=False,
        ),
    )
    if sum(encoder_score_source_counts.values()) != n_input_candidates_total:
        logger.warning(
            "infer_step1 encoder_score_source_counts 합 불일치: sum=%s n_input_candidates_total=%s",
            sum(encoder_score_source_counts.values()),
            n_input_candidates_total,
        )
    logger.info(
        "infer_step1 NMS summary: calls=%s requested=%s effective=%s dropped=%s",
        nms_state["calls"],
        nms_state["requested"],
        nms_state["effective"],
        nms_state["dropped"],
    )
    ignored_detect_count = int(ignored_rules) if isinstance(ignored_rules, int) else len(ignored_rules or [])
    ignored_verify_count = int(ignored_verify) if isinstance(ignored_verify, int) else len(ignored_verify or [])
    ignored_context_count = (
        int(ignored_context) if isinstance(ignored_context, int) else len(ignored_context or [])
    )
    final_metrics: dict[str, Any] = {
        "rules/detect_count": len(detect_rules),
        "rules/verify_count": len(verify_rules),
        "rules/verify_morph_count": len(morph_verify_rules),
        "rules/context_count": len(context_rules),
        "rules/ignored_detect_count": ignored_detect_count,
        "rules/ignored_verify_count": ignored_verify_count,
        "rules/ignored_context_count": ignored_context_count,
        "rules/ignored_verify_morph_unsupported_count": n_verify_rules_skipped_morph_unsupported,
        "window/detect_window_chars": detect_window_chars,
        "window/morph_window_chars": morph_window_chars,
        "window/verify_window_chars": verify_window_chars,
        "window/context_window_chars": context_window_chars,
        "candidates/n_total": n_candidates_total,
        "candidates/n_hard_fail": n_hard_fail,
        "score/encoder_score_missing_total": encoder_score_missing_total,
        "polyset/n_groups_span_polyset_total": n_groups_span_polyset_total,
        "polyset/n_ambiguous_groups_total": n_ambiguous_groups_total,
        "polyset/kept_top1_groups_total": kept_top1_groups_total,
        "polyset/kept_topk_groups_total": kept_topk_groups_total,
        "routing/n_to_llm_candidates_total": n_to_llm_candidates_total,
        "routing/n_auto_confirm_candidates_total": n_auto_confirm_candidates_total,
        "routing/n_ambiguous_candidates_total": n_ambiguous_candidates_total,
        "agroup/gold_total": int(agroup_layers_total.get("gold_total", 0)),
        "agroup/rule_detected": int(agroup_layers_total.get("rule_detected", 0)),
        "agroup/encoder_passed": int(agroup_layers_total.get("encoder_passed", 0)),
        "agroup/final_tp": int(agroup_layers_total.get("final_tp", 0)),
        "agroup/rule_detected_rate": _agroup_layer_summary(agroup_layers_total)["rule_detected_rate"],
        "agroup/encoder_pass_rate_given_rule_detected": _agroup_layer_summary(agroup_layers_total)["encoder_pass_rate_given_rule_detected"],
        "agroup/final_tp_rate": _agroup_layer_summary(agroup_layers_total)["final_tp_rate"],
        "no_candidate/total": total_no_candidate,
        "paths/input_path": str(input_path),
        "paths/output_path": str(output_path),
        "paths/report_path": str(report_path),
        "wandb/enabled": bool(wandb_meta.get("enabled", False)),
    }
    for source_key, source_count in encoder_score_source_counts.items():
        final_metrics[f"score/encoder_score_source_counts.{source_key}"] = int(source_count)
    for subbucket, bucket_count in no_candidate_counts.items():
        final_metrics[f"no_candidate/by_subbucket.{subbucket}"] = int(bucket_count)
    _wandb_log_safe(wandb_run, final_metrics, step=n_sents if n_sents > 0 else None)
    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            top_candidates_rows = [[eid, int(count)] for eid, count in candidates_by_eid.most_common(50)]
            if top_candidates_rows:
                table_top = wandb.Table(columns=["e_id", "count"], data=top_candidates_rows)
                _wandb_log_safe(wandb_run, {"tables/top_candidates_by_eid": table_top}, step=n_sents)

            no_candidate_rows = [
                [
                    row.get("example_id"),
                    row.get("e_id"),
                    row.get("subbucket"),
                    row.get("why"),
                    row.get("debug_ptr"),
                ]
                for row in no_candidate_samples[:50]
            ]
            if no_candidate_rows:
                table_no_candidate = wandb.Table(
                    columns=["example_id", "e_id", "subbucket", "why", "debug_ptr"],
                    data=no_candidate_rows,
                )
                _wandb_log_safe(wandb_run, {"tables/no_candidate_samples": table_no_candidate}, step=n_sents)
        except Exception as exc:
            logger.warning("[infer_step1][wandb] table 업로드 실패(무시): %s", exc)
        try:
            wandb_run.summary["n_sents"] = int(n_sents)
            wandb_run.summary["n_candidates_total"] = int(n_candidates_total)
            wandb_run.summary["n_hard_fail"] = int(n_hard_fail)
            wandb_run.summary["total_no_candidate"] = int(total_no_candidate)
            wandb_run.summary["output_path"] = str(output_path)
            wandb_run.summary["report_path"] = str(report_path)
            wandb_run.summary["agroup_scoring_method"] = str(encoder_scoring_method)
            wandb_run.summary["agroup_head_loaded"] = bool(head_loaded)
            wandb_run.summary["agroup_gold_total"] = int(agroup_layers_total.get("gold_total", 0))
            wandb_run.summary["agroup_rule_detected"] = int(agroup_layers_total.get("rule_detected", 0))
            wandb_run.summary["agroup_encoder_passed"] = int(agroup_layers_total.get("encoder_passed", 0))
            wandb_run.summary["agroup_final_tp"] = int(agroup_layers_total.get("final_tp", 0))
        except Exception:
            pass
        finish_hook = wandb_meta.get("finish_hook")
        if callable(finish_hook):
            try:
                atexit.unregister(finish_hook)
            except Exception:
                pass
        try:
            wandb_run.finish()
        except Exception as exc:
            logger.warning("[infer_step1][wandb] finish 실패(무시): %s", exc)
    logger.info("infer_step1 완료: %s", output_path)


def _postprocess_candidates(
    *,
    candidates: list[dict[str, Any]],
    expredict_map: dict[str, dict[str, Any]],
    uncertainty_enabled: bool,
    margin_policy: str,
    margin_threshold: float,
    low_conf_threshold: float,
    group_a_accept_threshold: float,
    use_sigmoid_prob: bool,
    temperature: float,
    postprocess_enabled: bool,
    nms_scope: str,
    nms_metric: str,
    nms_iou_threshold: float,
    nms_short_span_len_le: int,
    nms_short_span_min_overlap_ratio: float,
    nms_tie_breaker: str,
    polyset_competition: bool,
    ambiguous_only_polyset_topk: bool,
    polyset_topk_when_ambiguous: int,
    encoder_score_source_hint: str,
    nms_state: dict[str, Any],
    logger: logging.Logger,
    write_encoder_prob: bool = True,
    encoder_prob_field: str = "encoder_prob",
    encoder_prob_only_when_head_logits: bool = True,
    encoder_scoring_method: str = "missing",
    debug_key: str | None = None,
    debug_span_key: str | None = None,
    debug_polyset_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, str]]:
    stats = {
        "n_input_candidates": len(candidates),
        "n_groups_span_polyset": 0,
        "n_ambiguous_groups": 0,
        "n_ambiguous_candidates": 0,
        "kept_top1_groups": 0,
        "kept_topk_groups": 0,
        "n_to_llm_candidates": 0,
        "n_auto_confirm_candidates": 0,
        "encoder_score_missing_count": 0,
        "encoder_score_source_counts": Counter(),
    }
    meta = {"encoder_score_source": "missing", "dropped_by_topk": [], "dropped_by_nms": []}
    dropped_candidates: list[dict[str, Any]] = []
    if not candidates:
        return candidates, dropped_candidates, stats, meta

    def _cand_id(cand: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(cand.get("e_id") or ""),
            str(cand.get("span_key") or ""),
            str(cand.get("polyset_id") or ""),
        )

    def _record_dropped_candidate(
        cand: dict[str, Any], *, analysis_stage: str = "encoder_reject"
    ) -> None:
        dropped_candidates.append(
            {
                "e_id": str(cand.get("e_id") or ""),
                "span_key": str(cand.get("span_key") or ""),
                "encoder_score": _to_float(cand.get("encoder_score")),
                "confidence": _to_float(cand.get("confidence")),
                "routing_reason": str(cand.get("routing_reason") or ""),
                "triage": "discard",
                "analysis_stage": analysis_stage,
            }
        )

    _attach_pattern_metadata(candidates, expredict_map)
    group_a_candidates: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []

    for cand in candidates:
        _ensure_encoder_fields(cand)
        group_value = str(cand.get("group") or "").strip().lower()
        cand["encoder_rank"] = None
        cand["encoder_score_rank_in_polyset"] = None
        cand["encoder_score_margin"] = None
        cand["margin"] = None
        if group_value == "a":
            source = str(cand.get("encoder_score_source") or "missing").strip().lower()
            if source not in {"finetuned", "stub", "missing"}:
                source = "missing"
                cand["encoder_score_source"] = "missing"
            stats["encoder_score_source_counts"][source] += 1
            if cand.get("encoder_score") is None:
                stats["encoder_score_missing_count"] += 1
            group_a_candidates.append(cand)
            continue

        cand["ambiguous"] = False
        cand["routing_reason"] = (
            "group_b_direct_to_llm" if group_value == "b" else "missing_group_direct_to_llm"
        )
        _set_route_fields(cand, route="to_llm")
        kept.append(cand)

    if group_a_candidates:
        encoder_score_source, _missing_count = _resolve_encoder_score_source(
            candidates=group_a_candidates,
            encoder_score_source_hint=encoder_score_source_hint,
            logger=logger,
        )
        meta["encoder_score_source"] = encoder_score_source

        can_score_group_a = (
            uncertainty_enabled
            and encoder_score_source != "missing"
            and stats["encoder_score_missing_count"] == 0
        )

        if not can_score_group_a:
            if encoder_score_source == "missing":
                stats["encoder_score_source_counts"] = Counter({"missing": len(group_a_candidates)})
                stats["encoder_score_missing_count"] = len(group_a_candidates)
            for cand in group_a_candidates:
                _ensure_encoder_fields(cand)
                cand["encoder_score"] = None
                cand["encoder_score_source"] = "missing"
                cand["ambiguous"] = False
                cand["margin"] = None
                cand["encoder_score_rank_in_polyset"] = None
                cand["encoder_score_margin"] = None
                cand["triage"] = "discard"
                cand["routing_reason"] = (
                    "group_a_missing_encoder_score" if uncertainty_enabled else "group_a_uncertainty_disabled"
                )
                _set_route_fields(cand, route="drop")
                _record_dropped_candidate(cand)
        else:
            scored_group_a = _apply_encoder_confidence(
                group_a_candidates,
                use_sigmoid_prob=use_sigmoid_prob,
                temperature=temperature,
                write_encoder_prob=write_encoder_prob,
                encoder_prob_field=encoder_prob_field,
                encoder_prob_only_when_head_logits=encoder_prob_only_when_head_logits,
                encoder_scoring_method=encoder_scoring_method,
            )
            span_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for idx, cand in enumerate(scored_group_a):
                span_key = str(cand.get("span_key") or "").strip()
                if not span_key:
                    span_key = f"__group_a_idx__:{idx}"
                span_groups[span_key].append(cand)

            for span_key, span_group in span_groups.items():
                span_group.sort(
                    key=lambda c: (
                        float(c.get("confidence", 0.0) or 0.0),
                        float(c.get("encoder_score", 0.0) or 0.0),
                        _span_length(c.get("span_segments") or []),
                        str(c.get("e_id") or ""),
                    ),
                    reverse=True,
                )
                top1 = span_group[0]
                for idx, cand in enumerate(span_group, start=1):
                    cand["encoder_rank"] = idx
                    cand["encoder_score_rank_in_polyset"] = idx
                    cand["encoder_score_margin"] = None
                    cand["margin"] = None
                    cand["ambiguous"] = False
                    cand["encoder_score_source"] = encoder_score_source
                    if idx > 1:
                        cand["triage"] = "discard"
                        cand["routing_reason"] = "group_a_lower_score_same_span"
                        _set_route_fields(cand, route="drop")
                        _record_dropped_candidate(cand)

                top1_conf = float(top1.get("confidence", 0.0) or 0.0)
                if top1_conf >= float(group_a_accept_threshold):
                    top1["triage"] = "confirm"
                    top1["routing_reason"] = "group_a_accept"
                    _set_route_fields(top1, route="auto_confirm")
                    kept.append(top1)
                    stats["n_auto_confirm_candidates"] += 1
                    stats["kept_top1_groups"] += 1
                else:
                    top1["triage"] = "discard"
                    top1["routing_reason"] = f"group_a_below_threshold<{group_a_accept_threshold:.2f}"
                    _set_route_fields(top1, route="drop")
                    _record_dropped_candidate(top1)

    if nms_scope.lower() != "same_eid_or_polyset":
        if not nms_state.get("did_log_scope"):
            logger.warning(
                "NMS scope forced: same_eid_or_polyset (requested=%s)",
                nms_scope,
            )
            nms_state["did_log_scope"] = True
        nms_scope = "same_eid_or_polyset"
    else:
        if not nms_state.get("did_log_scope"):
            logger.info(
                "NMS scope applied: same_eid_or_polyset (requested=%s)",
                nms_scope,
            )
            nms_state["did_log_scope"] = True
    nms_state["effective"] = nms_scope

    if postprocess_enabled and nms_scope.lower() != "off":
        nms_state["calls"] += 1
        kept_before_nms = kept
        kept = _apply_nms(
            candidates=kept,
            scope=nms_scope,
            metric=nms_metric,
            iou_threshold=nms_iou_threshold,
            short_span_len_le=nms_short_span_len_le,
            short_span_min_overlap_ratio=nms_short_span_min_overlap_ratio,
            tie_breaker=nms_tie_breaker,
        )
        kept_ids = {_cand_id(c) for c in kept}
        dropped = [c for c in kept_before_nms if _cand_id(c) not in kept_ids]
        if dropped:
            meta["dropped_by_nms"].extend([_cand_id(c) for c in dropped])
            for cand in dropped:
                if str(cand.get("group") or "").strip().lower() == "a" and str(cand.get("routing_reason") or "") == "group_a_accept":
                    cand["routing_reason"] = "nms_drop_after_group_a_accept"
                    _set_route_fields(cand, route="drop")
                    _record_dropped_candidate(cand, analysis_stage="final_reject_after_encoder")
                else:
                    _record_dropped_candidate(cand, analysis_stage="postprocess_nms")
        nms_state["dropped"] += max(0, len(kept_before_nms) - len(kept))

    stats["n_to_llm_candidates"] = sum(1 for c in kept if c.get("to_llm"))
    stats["n_auto_confirm_candidates"] = sum(1 for c in kept if c.get("auto_confirm"))
    stats["n_ambiguous_candidates"] = sum(1 for c in kept if c.get("ambiguous"))

    return kept, dropped_candidates, stats, meta


def _resolve_encoder_score_source(

    *,
    candidates: list[dict[str, Any]],
    encoder_score_source_hint: str,
    logger: logging.Logger,
) -> tuple[str, int]:
    numeric_count = 0
    missing_count = 0
    for cand in candidates:
        score = _to_float(cand.get("encoder_score"))
        if score is None:
            missing_count += 1
        else:
            numeric_count += 1
    hint = encoder_score_source_hint.strip().lower()
    if hint == "finetuned_model":
        hint = "finetuned"
    if hint == "placeholder":
        hint = "stub"
    if hint in {"finetuned", "stub", "missing"}:
        if hint == "finetuned" and missing_count > 0:
            logger.warning(
                "infer_step1 encoder_score_source=finetuned 이지만 결측 존재 -> missing 처리",
            )
            return "missing", missing_count
        return hint, missing_count

    source_votes: Counter[str] = Counter()
    for cand in candidates:
        source = str(cand.get("encoder_score_source", "")).strip().lower()
        if source in {"finetuned", "stub", "missing"}:
            source_votes[source] += 1
    if source_votes:
        chosen, _count = source_votes.most_common(1)[0]
        if chosen == "finetuned" and missing_count > 0:
            return "missing", missing_count
        return chosen, missing_count

    if numeric_count == 0:
        return "missing", missing_count
    if missing_count > 0:
        logger.warning(
            "infer_step1 encoder_score 일부 결측: missing_count=%s -> ambiguity pruning 스킵",
            missing_count,
        )
        return "missing", missing_count
    return "finetuned", missing_count


def _ensure_encoder_fields(candidate: dict[str, Any]) -> None:
    candidate.setdefault("encoder_score", None)
    candidate.setdefault("encoder_score_source", None)
    candidate.setdefault("encoder_rank", None)
    candidate.setdefault("confidence", None)
    candidate.setdefault("encoder_prob", None)
    candidate.setdefault("margin", None)
    candidate.setdefault("ambiguous", None)


def _set_route_fields(candidate: dict[str, Any], *, route: str) -> None:
    candidate["route"] = route
    candidate["auto_confirm"] = route == "auto_confirm"
    candidate["to_llm"] = route == "to_llm"
    candidate["dropped"] = route == "drop"


def _attach_pattern_metadata(
    candidates: list[dict[str, Any]], expredict_map: dict[str, dict[str, Any]]
) -> None:
    for cand in candidates:
        e_id = str(cand.get("e_id", ""))
        row = expredict_map.get(e_id, {}) if e_id else {}
        if isinstance(row, dict):
            if not cand.get("group"):
                group = row.get("group")
                if group:
                    cand["group"] = str(group).lower()
            if not cand.get("polyset_id"):
                polyset_id = row.get("polyset_id")
                if polyset_id:
                    cand["polyset_id"] = str(polyset_id)


def _apply_encoder_confidence(
    candidates: list[dict[str, Any]],
    *,
    use_sigmoid_prob: bool,
    temperature: float,
    write_encoder_prob: bool,
    encoder_prob_field: str,
    encoder_prob_only_when_head_logits: bool,
    encoder_scoring_method: str,
) -> list[dict[str, Any]]:
    safe_temp = temperature if temperature > 0 else 1.0
    should_write_encoder_prob = write_encoder_prob and (
        (not encoder_prob_only_when_head_logits) or encoder_scoring_method == "head_logits"
    )
    for cand in candidates:
        score = _to_float(cand.get("encoder_score")) or 0.0
        cand["encoder_score"] = score
        prob = _sigmoid(score / safe_temp)
        if should_write_encoder_prob:
            cand[encoder_prob_field] = prob
        else:
            cand[encoder_prob_field] = None
        if use_sigmoid_prob:
            cand["confidence"] = prob
        else:
            cand["confidence"] = float(score)
    return candidates


def _compute_margin_and_ambiguity(
    *,
    top1: dict[str, Any],
    top2: dict[str, Any] | None,
    margin_threshold: float,
    low_conf_threshold: float,
) -> tuple[float, bool]:
    conf_top1 = float(top1.get("confidence", 0.0) or 0.0)
    if top2 is None:
        margin = 1.0
        ambiguous = conf_top1 < low_conf_threshold
        return margin, ambiguous
    conf_top2 = float(top2.get("confidence", 0.0) or 0.0)
    margin = conf_top1 - conf_top2
    ambiguous = (margin < margin_threshold) or (conf_top1 < low_conf_threshold)
    return margin, ambiguous


def _routing_reason_for_group(
    *,
    top1: dict[str, Any],
    top2: dict[str, Any] | None,
    margin: float,
    margin_threshold: float,
    low_conf_threshold: float,
) -> str:
    conf_top1 = float(top1.get("confidence", 0.0) or 0.0)
    if top2 is None:
        return "single_candidate_polyset"
    if conf_top1 < low_conf_threshold:
        return f"low_confidence<{low_conf_threshold:.2f}"
    if margin < margin_threshold:
        return f"ambiguous_polyset_margin<{margin_threshold:.2f}"
    return "confident_polyset"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_nms(
    *,
    candidates: list[dict[str, Any]],
    scope: str,
    metric: str,
    iou_threshold: float,
    short_span_len_le: int,
    short_span_min_overlap_ratio: float,
    tie_breaker: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    scope = scope.lower()
    metric = metric.lower()
    tie_breaker = tie_breaker.lower()

    def _score(cand: dict[str, Any]) -> float:
        score = _to_float(cand.get("encoder_score"))
        if score is None:
            score = _to_float(cand.get("score")) or 0.0
        return score

    def _sort_key(cand: dict[str, Any]) -> tuple[float, int, str]:
        length = _span_length(cand.get("span_segments") or [])
        span_key = str(cand.get("span_key", ""))
        return (_score(cand), length, span_key)

    sorted_candidates = sorted(candidates, key=_sort_key, reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in sorted_candidates:
        cand_segments = cand.get("span_segments") or []
        cand_len = _span_length(cand_segments)
        if cand_len <= 0:
            kept.append(cand)
            continue
        should_keep = True
        for kept_cand in kept:
            if not _nms_scope_match(cand, kept_cand, scope):
                continue
            kept_segments = kept_cand.get("span_segments") or []
            overlap = _overlap_ratio(
                cand_segments,
                kept_segments,
                metric=metric,
                short_span_len_le=short_span_len_le,
                short_span_min_overlap_ratio=short_span_min_overlap_ratio,
                iou_threshold=iou_threshold,
            )
            if overlap:
                should_keep = False
                break
        if should_keep:
            kept.append(cand)
    return kept


def _nms_scope_match(
    cand: dict[str, Any], kept: dict[str, Any], scope: str
) -> bool:
    if scope == "global":
        return True
    eid_a = str(cand.get("e_id", ""))
    eid_b = str(kept.get("e_id", ""))
    poly_a = str(cand.get("polyset_id", "") or "")
    poly_b = str(kept.get("polyset_id", "") or "")
    same_eid = bool(eid_a) and eid_a == eid_b
    same_poly = bool(poly_a) and poly_a == poly_b
    if scope == "same_eid_only":
        return same_eid
    if scope == "polyset_only":
        return same_poly
    # SSOT: NMS is dedup-only; do not suppress across different e_id even if same polyset.
    return same_eid


def _overlap_ratio(
    seg_a: list[list[int]],
    seg_b: list[list[int]],
    *,
    metric: str,
    short_span_len_le: int,
    short_span_min_overlap_ratio: float,
    iou_threshold: float,
) -> bool:
    len_a = _span_length(seg_a)
    len_b = _span_length(seg_b)
    if len_a <= 0 or len_b <= 0:
        return False
    overlap = _intersection_length(seg_a, seg_b)
    if overlap <= 0:
        return False
    min_len = min(len_a, len_b)
    if min_len <= short_span_len_le:
        return (overlap / float(min_len)) >= short_span_min_overlap_ratio
    if metric == "char_iou":
        union = len_a + len_b - overlap
        if union <= 0:
            return False
        return (overlap / float(union)) >= iou_threshold
    return False


def _span_length(span_segments: list[list[int]]) -> int:
    total = 0
    for seg in span_segments:
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            continue
        total += max(0, int(seg[1]) - int(seg[0]))
    return total


def _normalize_segments(span_segments: list[list[int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for seg in span_segments:
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            continue
        try:
            s = int(seg[0])
            e = int(seg[1])
        except (TypeError, ValueError):
            continue
        out.append((s, e))
    return sorted(out, key=lambda x: x[0])


def _intersection_length(
    seg_a: list[list[int]], seg_b: list[list[int]]
) -> int:
    a = _normalize_segments(seg_a)
    b = _normalize_segments(seg_b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        if a_end <= b_start:
            i += 1
            continue
        if b_end <= a_start:
            j += 1
            continue
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return total


class _EncoderScorer:
    def __init__(
        self,
        *,
        tokenizer,
        model,
        head,
        scoring_method: str,
        device: str,
        max_seq_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.head = head
        self.scoring_method = scoring_method
        self.device = device
        self.max_seq_len = max_seq_len

    def score_texts(self, texts: list[str], *, batch_size: int) -> list[float]:
        import torch

        scores: list[float] = []
        self.model.eval()
        if self.head is not None:
            self.head.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                attn_mask = inputs.get("attention_mask")
                if attn_mask is None:
                    raise RuntimeError("infer_step1 scorer attention_mask missing")
                pooled = _masked_mean_pool(outputs.last_hidden_state, attn_mask)
                if self.head is not None:
                    logits = self.head(pooled).squeeze(-1)
                    batch_scores = logits.detach().cpu().tolist()
                else:
                    batch_scores = pooled.mean(dim=1).detach().cpu().tolist()
                scores.extend([float(s) for s in batch_scores])
        return scores

    def score_pairs(self, text_as: list[str], text_bs: list[str], *, batch_size: int) -> list[float]:
        import torch

        if len(text_as) != len(text_bs):
            raise RuntimeError("infer_step1 scorer pair input length mismatch")
        scores: list[float] = []
        self.model.eval()
        if self.head is not None:
            self.head.eval()
        with torch.no_grad():
            for i in range(0, len(text_as), batch_size):
                chunk_as = text_as[i : i + batch_size]
                chunk_bs = text_bs[i : i + batch_size]
                inputs = self.tokenizer(
                    chunk_as,
                    chunk_bs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                attn_mask = inputs.get("attention_mask")
                if attn_mask is None:
                    raise RuntimeError("infer_step1 scorer attention_mask missing")
                pooled = _masked_mean_pool(outputs.last_hidden_state, attn_mask)
                if self.head is not None:
                    logits = self.head(pooled).squeeze(-1)
                    batch_scores = logits.detach().cpu().tolist()
                else:
                    batch_scores = pooled.mean(dim=1).detach().cpu().tolist()
                scores.extend([float(s) for s in batch_scores])
        return scores


def _build_encoder_scorer(
    *,
    cfg: dict[str, Any],
    run_context: RunContext,
    enabled: bool,
    max_seq_len: int,
    logger: logging.Logger,
    require_head_logits: bool,
    disallow_fallback_scoring: bool,
) -> _EncoderScorer | None:
    if not enabled:
        if require_head_logits:
            raise ConfigError("infer_step1 A-group head_logits 강제 모드에서는 candidate_scoring.enabled=true가 필요합니다.")
        logger.info("infer_step1 candidate_scoring disabled -> encoder_score=missing")
        return None
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        logger.warning("infer_step1 encoder 로드 실패(의존성): %s", exc)
        return None

    runtime_cfg = cfg.get("runtime", {}) or {}
    device = _resolve_runtime_device(runtime_cfg)
    artifacts_root = _artifacts_root_from_outputs_dir(run_context.outputs_dir, logger)
    infer_cfg = cfg.get("infer", {}) or {}
    override = infer_cfg.get("encoder_scoring_checkpoint") or infer_cfg.get("finetune_checkpoint")
    requested = str(infer_cfg.get("scoring_method") or ("head_logits" if require_head_logits else "auto"))
    no_head_requested = requested.endswith("_no_head")
    if require_head_logits and no_head_requested:
        raise ConfigError("infer_step1 A-group 연구 모드에서는 no-head scoring 요청이 허용되지 않습니다.")
    requested_scoring_method = str(infer_cfg.get("scoring_method", "auto") or "auto").strip().lower()
    if override:
        checkpoint_dir = Path(str(override))
        if not checkpoint_dir.exists():
            logger.error("infer_step1 encoder_scoring override missing: %s", checkpoint_dir)
            raise ConfigError(f"infer_step1 encoder_scoring override missing: {checkpoint_dir}")
        source = "override"
    else:
        checkpoint_dir = _latest_finetune_checkpoint(artifacts_root, run_context.exp_id)
        if checkpoint_dir is None:
            if require_head_logits or disallow_fallback_scoring:
                raise ConfigError("infer_step1 A-group 연구 모드에서는 finetune checkpoint와 head.pt가 필수입니다.")
            logger.warning("infer_step1 finetune checkpoint 없음 -> encoder_score=stub")
            return None
        source = "auto_latest"

    encoder_dir = checkpoint_dir / "encoder"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    head_path = checkpoint_dir / "head.pt"
    if source == "override":
        if (
            not encoder_dir.exists()
            or not tokenizer_dir.exists()
            or ((not no_head_requested) and (not head_path.exists()))
        ):
            raise ConfigError(
                "infer_step1 override checkpoint incomplete: "
                f"encoder_dir={encoder_dir} tokenizer_dir={tokenizer_dir} head_path={head_path}"
            )
    elif not encoder_dir.exists() or not tokenizer_dir.exists():
        if require_head_logits or disallow_fallback_scoring:
            raise ConfigError(f"infer_step1 A-group 연구 모드에서는 완전한 finetune checkpoint가 필요합니다: {checkpoint_dir}")
        logger.warning("infer_step1 finetune checkpoint 불완전 -> encoder_score=stub")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception as exc:
        logger.info("infer_step1 tokenizer.json 직접 로드 사용")
        from transformers import PreTrainedTokenizerFast
        import json

        tokenizer_file = tokenizer_dir / "tokenizer.json"
        special_tokens_map_path = tokenizer_dir / "special_tokens_map.json"
        if not tokenizer_file.exists():
            raise

        def _special_token_value(value: Any) -> str | None:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                content = value.get("content")
                if isinstance(content, str):
                    return content
            return None

        special_kwargs: dict[str, str] = {}
        if special_tokens_map_path.exists():
            try:
                payload = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))
                for key in [
                    "unk_token",
                    "sep_token",
                    "pad_token",
                    "cls_token",
                    "mask_token",
                    "bos_token",
                    "eos_token",
                ]:
                    token_value = _special_token_value(payload.get(key))
                    if token_value is not None:
                        special_kwargs[key] = token_value
            except Exception as map_exc:
                logger.warning("infer_step1 special_tokens_map 로드 실패: %s", map_exc)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            **special_kwargs,
        )
    model = AutoModel.from_pretrained(encoder_dir)
    model.to(device)
    head = None
    scoring_method = "missing"
    if (not no_head_requested) and head_path.exists():
        try:
            head = nn.Linear(int(model.config.hidden_size), 1)
            payload = torch.load(head_path, map_location=device)
            if isinstance(payload, dict) and "state_dict" in payload:
                head.load_state_dict(payload["state_dict"])
            else:
                head.load_state_dict(payload)
            head.to(device)
            head.eval()
            scoring_method = "head_logits"
        except Exception as exc:
            if requested_scoring_method == "head_logits":
                raise ConfigError(
                    f"infer_step1 scoring_method=head_logits requested but head load failed: checkpoint={checkpoint_dir} err={exc}"
                ) from exc
            if require_head_logits or disallow_fallback_scoring:
                raise ConfigError(
                    f"infer_step1 A-group 연구 모드에서는 head load fallback이 금지됩니다: checkpoint={checkpoint_dir} err={exc}"
                ) from exc
            head = None
            scoring_method = "pooled_mean_no_head"
            logger.warning(
                "infer_step1 finetune head load failed -> fallback scoring_method=pooled_mean_no_head: %s",
                exc,
            )
    else:
        if requested_scoring_method == "head_logits" or require_head_logits or disallow_fallback_scoring:
            raise ConfigError(
                f"infer_step1 A-group 연구 모드에서는 head.pt가 필수입니다: checkpoint={checkpoint_dir}"
            )
        if not no_head_requested:
            scoring_method = "pooled_mean_no_head"
            logger.warning(
                "infer_step1 finetune head.pt missing -> fallback scoring_method=pooled_mean_no_head (checkpoint=%s)",
                checkpoint_dir,
            )
    logger.info(
        "[scoring] requested=%s no_head_requested=%s head_used=%s require_head_logits=%s disable_fallback=%s",
        requested,
        no_head_requested,
        head is not None,
        require_head_logits,
        disallow_fallback_scoring,
    )
    logger.info(
        "infer_step1 encoder_scoring_source=%s checkpoint=%s device=%s scoring_method=%s",
        source,
        checkpoint_dir,
        device,
        scoring_method,
    )
    return _EncoderScorer(
        tokenizer=tokenizer,
        model=model,
        head=head,
        scoring_method=scoring_method,
        device=device,
        max_seq_len=max_seq_len,
    )


def _resolve_runtime_device(runtime_cfg: dict[str, Any]) -> str:
    device = str(runtime_cfg.get("device", "auto"))
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def _latest_finetune_checkpoint(artifacts_root: Path, exp_id: str) -> Path | None:
    finetune_root = artifacts_root / exp_id / "train_finetune"
    if not finetune_root.exists():
        return None
    run_dirs = [p for p in finetune_root.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    for run_dir in reversed(run_dirs):
        candidate = run_dir / "outputs" / "checkpoints" / "last"
        if candidate.exists():
            return candidate
    return None


def _score_candidates_with_encoder(
    *,
    candidates: list[dict[str, Any]],
    raw_sentence: str,
    context_left: str,
    context_right: str,
    scorer: _EncoderScorer | None,
    scoring_enabled: bool,
    batch_size: int,
    logger: logging.Logger,
    require_head_logits: bool,
    expredict_map: dict[str, dict[str, Any]],
    input_construction_version: str,
) -> None:
    if not candidates:
        return
    if not scoring_enabled:
        for cand in candidates:
            cand["encoder_score"] = None
            cand["encoder_score_source"] = "missing"
        return
    if scorer is None:
        if require_head_logits:
            raise ConfigError("infer_step1 A-group 연구 모드에서는 scorer=None fallback이 허용되지 않습니다.")
        for cand in candidates:
            score = _to_float(cand.get("score")) or 0.0
            cand["encoder_score"] = float(score)
            cand["encoder_score_source"] = "stub"
        return

    pair_mode = str(input_construction_version or AGROUP_INPUT_CONSTRUCTION_VERSION).strip() == AGROUP_INPUT_CONSTRUCTION_VERSION_V2
    texts: list[str] = []
    text_as: list[str] = []
    text_bs: list[str] = []
    fallback_count = 0
    for cand in candidates:
        try:
            if pair_mode:
                eid = str(cand.get("e_id") or "").strip()
                meta_row = expredict_map.get(eid) or {}
                canonical_form = str(meta_row.get("canonical_form") or "").strip()
                gloss = str(meta_row.get("gloss") or "").strip()
                if not canonical_form:
                    raise ConfigError(
                        f"infer_step1 A-group pair mode candidate meta missing canonical_form: e_id={eid}"
                    )
                built = build_agroup_pair_encoder_input(
                    {
                        "target_sentence": raw_sentence,
                        "span_segments": [tuple(seg) for seg in (cand.get("span_segments") or [])],
                    },
                    {
                        "e_id": eid,
                        "canonical_form": canonical_form,
                        "gloss": gloss,
                    },
                )
                text_a = str(built.get("text_a") or "")
                text_b = str(built.get("text_b") or "")
                if not text_a or not text_b:
                    raise ConfigError(
                        f"infer_step1 A-group pair mode built empty text_a/text_b: e_id={eid}"
                    )
                text_as.append(text_a)
                text_bs.append(text_b)
                cand["encoder_input_text_a"] = text_a
                cand["encoder_input_text_b"] = text_b
                cand["encoder_input_version"] = AGROUP_INPUT_CONSTRUCTION_VERSION_V2
                cand["encoder_input_text_b_format"] = "canonical_form_plus_gloss_plain"
            else:
                built_text = _build_encoder_input_text(
                    e_id=str(cand["e_id"]),
                    target_sentence=raw_sentence,
                    span_segments=[tuple(seg) for seg in (cand.get("span_segments") or [])],
                    context_left=context_left,
                    context_right=context_right,
                )
                texts.append(built_text)
                cand["encoder_input_text"] = built_text
                cand["encoder_input_version"] = AGROUP_INPUT_CONSTRUCTION_VERSION
        except Exception as exc:
            if require_head_logits:
                raise ConfigError(f"infer_step1 A-group encoder input construction 실패: {exc}") from exc
            fallback_count += 1
            if pair_mode:
                text_as.append(raw_sentence)
                text_bs.append(raw_sentence)
            else:
                texts.append(raw_sentence)
    logger.info(
        "[infer_step1] encoder_input_source=%s fallback_count=%s",
        "factory.build_agroup_pair_encoder_input" if pair_mode else "factory._build_encoder_input_text",
        fallback_count,
    )
    try:
        if pair_mode:
            scores = scorer.score_pairs(text_as, text_bs, batch_size=batch_size)
        else:
            scores = scorer.score_texts(texts, batch_size=batch_size)
    except Exception as exc:
        if require_head_logits:
            raise ConfigError(f"infer_step1 A-group encoder scoring 실패: {exc}") from exc
        logger.warning("infer_step1 encoder scoring 실패 -> stub fallback: %s", exc)
        for cand in candidates:
            score = _to_float(cand.get("score")) or 0.0
            cand["encoder_score"] = float(score)
            cand["encoder_score_source"] = "stub"
        return
    for cand, score in zip(candidates, scores):
        cand["encoder_score"] = float(score)
        cand["encoder_score_source"] = "finetuned"


def _resolve_input_path(
    cfg: dict[str, Any],
    run_context: RunContext,
    logger: logging.Logger,
) -> tuple[Path, str, bool]:
    cfg, forced_path, forced_source = apply_forced_input_jsonl(cfg, stage="infer_step1")

    infer_cfg = cfg.get("infer", {})
    # NOTE: infer_step1은 JSONL만 받는다 (csv는 여기서 직접 처리하지 않음)
    input_path = infer_cfg.get("input_jsonl") or infer_cfg.get("input_path")

    input_path_source: str | None = None
    input_path_forced = False

    if forced_path:
        input_path = forced_path
        input_path_source = forced_source or "forced"
        input_path_forced = True
    elif input_path:
        input_path_source = "infer.input_jsonl"

    if input_path:
        p = Path(str(input_path))
        if not p.exists():
            raise ConfigError(f"infer_step1 입력 JSONL이 존재하지 않습니다: {p} (source={input_path_source})")
        logger.info(
            "[paths] stage=infer_step1 input_path=%s forced=%s source=%s",
            p,
            input_path_forced,
            input_path_source,
        )
        return p, input_path_source, input_path_forced

    # auto_latest ingest_corpus
    artifacts_root = _artifacts_root_from_outputs_dir(run_context.outputs_dir, logger)
    ingest_root = artifacts_root / run_context.exp_id / "ingest_corpus"

    candidate: Path | None = None
    if ingest_root.exists():
        run_dirs = sorted(
            [p for p in ingest_root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if run_dirs:
            cand = run_dirs[0] / "outputs" / "ingest_corpus.jsonl"
            if cand.exists() and cand.stat().st_size > 0:
                candidate = cand

    if candidate is None:
        raise ConfigError(
            "infer_step1 input_jsonl 경로를 찾지 못했습니다. "
            f"exp_id={run_context.exp_id} searched={ingest_root}/*/outputs/ingest_corpus.jsonl"
        )

    logger.info(
        "[paths] stage=infer_step1 input_path=%s forced=%s source=%s",
        candidate,
        False,
        "auto_latest_ingest_corpus",
    )
    return candidate, "auto_latest_ingest_corpus", False


def _validate_input_jsonl_schema(input_path: Path, max_lines: int = 50) -> None:
    valid = 0
    checked = 0
    with input_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if checked >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            checked += 1
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ConfigError(
                    f"infer_step1 input_jsonl schema invalid: JSON parse error in first {max_lines} lines: path={input_path} err={exc}"
                ) from exc
            if obj.get("raw_sentence") or obj.get("target_sentence"):
                valid += 1
    if valid < 1:
        raise ConfigError(
            f"infer_step1 input_jsonl schema invalid: missing raw_sentence/target_sentence in sampled records (path={input_path}, sampled={checked})"
        )

def _artifacts_root_from_outputs_dir(outputs_dir: Path, logger: logging.Logger) -> Path:
    outputs_dir = Path(outputs_dir)
    if len(outputs_dir.parents) < 4:
        raise ValueError(f"outputs_dir 경로 깊이가 부족합니다: {outputs_dir}")
    artifacts_root = outputs_dir.parents[3]
    logger.info("infer_step1 artifacts_root(from outputs_dir): %s", artifacts_root)
    return artifacts_root


def _ensure_pos_std(morph_tokens, pos_mapping):
    """Ensure each morph token dict has 'pos_std'. Falls back to agreed mapping rules."""
    if morph_tokens is None:
        return None
    if not isinstance(morph_tokens, list):
        return morph_tokens

    kiwi_map = {}
    sejong_map = {}
    if isinstance(pos_mapping, dict):
        kiwi_map = pos_mapping.get("kiwi", {}) if isinstance(pos_mapping.get("kiwi"), dict) else {}
        sejong_map = pos_mapping.get("sejong", {}) if isinstance(pos_mapping.get("sejong"), dict) else {}

    def _map_one(pos):
        if not pos:
            return "UNK"
        # explicit dict mapping if present
        if isinstance(pos_mapping, dict) and isinstance(pos_mapping.get(pos), str):
            return str(pos_mapping[pos])
        if pos in kiwi_map:
            return str(kiwi_map[pos])
        if pos in sejong_map:
            return str(sejong_map[pos])

        # fallback rules (project policy)
        if pos in {"VV-R", "VV-I"}: return "VV"
        if pos in {"VA-R", "VA-I"}: return "VA"
        if pos in {"VX-R", "VX-I"}: return "VX"
        if pos in {"XSA-R", "XSA-I"}: return "XSA"
        if pos in {"SSO", "SSC"}: return "SS"
        if pos == "XSM": return "XSA"
        if isinstance(pos, str) and pos.startswith("W_"): return "SW"
        if pos in {"Z_CODA", "Z_SIOT"}: return "UNK"
        if pos in {"USER0","USER1","USER2","USER3","USER4"}: return "NNP"
        if pos == "UN": return "UNK"
        if pos in {"MMA","MMD","MMN"}: return "MM"
        return pos

    out = []
    for tok in morph_tokens:
        if not isinstance(tok, dict):
            out.append(tok)
            continue
        if tok.get("pos_std"):
            out.append(tok)
            continue
        pos = tok.get("pos") or tok.get("tag") or tok.get("pos_kiwi")
        new_tok = dict(tok)
        new_tok["pos_std"] = _map_one(pos)
        out.append(new_tok)
    return out


def _ensure_expredict_map(
    dict_bundle: dict[str, Any], logger: logging.Logger
) -> tuple[dict[str, dict[str, Any]], bool]:
    expredict_map = dict_bundle.get("expredict_map")
    if not isinstance(expredict_map, dict):
        expredict_map = {}

    required_keys = set(validate_dict_loader.SHEET1_REQUIRED_KEYS)
    optional_keys = set(validate_dict_loader.SHEET1_OPTIONAL_KEYS)
    required_keys.add("disconti_allowed")

    def _missing_required(row: dict[str, Any]) -> bool:
        return any(key not in row for key in required_keys.union(optional_keys))

    def _needs_rebuild() -> bool:
        if not expredict_map:
            return True
        edf003 = expredict_map.get("edf003")
        if isinstance(edf003, dict) and "disconti_allowed" not in edf003:
            return True
        sample = list(expredict_map.values())[:5]
        for row in sample:
            if not isinstance(row, dict) or _missing_required(row):
                return True
        return False

    if not _needs_rebuild():
        return expredict_map, False

    records = dict_bundle.get("expredict") or dict_bundle.get("sheet1_patterns") or []
    rebuilt: dict[str, dict[str, Any]] = {}
    if isinstance(records, list):
        for row in records:
            if not isinstance(row, dict):
                continue
            row = validate_dict_loader._ensure_sheet1_keys(row)  # type: ignore[attr-defined]
            e_id = row.get("e_id")
            if e_id:
                rebuilt[str(e_id)] = row
    expredict_map = rebuilt
    logger.warning(
        "[WARN] expredict_map missing required keys -> rebuilt from sheet1 records."
    )
    return expredict_map, True
