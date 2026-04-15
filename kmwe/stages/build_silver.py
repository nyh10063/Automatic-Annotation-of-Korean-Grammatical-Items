from __future__ import annotations

import json
import logging
import os
from datetime import datetime
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Callable

from kmwe.core.config_loader import ConfigError
from kmwe.core.fs_guard import assert_under_dir
from kmwe.stages import validate_dict as validate_dict_loader
from kmwe.core.run_context import RunContext
from kmwe.utils.input_override import apply_forced_input_jsonl
from kmwe.utils.for_users_export import export_silver_user_csv
from kmwe.utils.jsonio import write_json, write_jsonl_line
from kmwe.utils.morph import analyze_with_kiwi

ADNORM_EQUIV = {
    "ㄴ": {"ㄴ", "ᆫ"},
    "ᆫ": {"ㄴ", "ᆫ"},
    "ㄹ": {"ㄹ", "ᆯ"},
    "ᆯ": {"ㄹ", "ᆯ"},
}

# validate_dict stage callable signature를 그대로 따름


def run_build_silver(*, cfg: dict[str, Any], run_context: RunContext) -> None:
    logger = logging.getLogger("kmwe")
    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    input_path, input_path_source, input_path_forced = _resolve_input_path(cfg, run_context, logger)
    if not input_path.exists():
        raise ConfigError(f"build_silver 입력 JSONL이 존재하지 않습니다: {input_path}")

    dict_source, dict_stats, dict_bundle = _load_dict_stats(cfg)
    logger.info(
        "build_silver dict 로딩 성공: source=%s patterns=%s rules=%s",
        dict_source,
        dict_stats["n_patterns_total"],
        dict_stats["n_rules_total"],
    )

    output_path = outputs_dir / "silver.jsonl"
    report_path = outputs_dir / "build_silver_report.json"

    logger.info("build_silver pipeline_order=detect->verify->context (MVP)")

    rule_sets = _prepare_stage_rules(dict_bundle.get("rules", []))
    detect_rules = rule_sets["detect_rules"]
    verify_rules = rule_sets["verify_rules"]
    morph_verify_rules = rule_sets["morph_verify_rules"]
    ignored_rules = rule_sets["ignored_rules"]
    ignored_verify = rule_sets["ignored_verify"]
    n_verify_rules_skipped_morph_unsupported = rule_sets["n_verify_rules_skipped_morph_unsupported"]
    n_verify_rules_supported_surface = len(verify_rules)
    n_verify_rules_supported_surface_raw_sentence = sum(
        1 for r in verify_rules if str(r.get("target", "")).lower() == "raw_sentence"
    )
    n_verify_rules_supported_surface_token_window = sum(
        1 for r in verify_rules if str(r.get("target", "")).lower() == "token_window"
    )
    context_rules = rule_sets["context_rules"]
    ignored_context = rule_sets["ignored_context"]
    logger.info("build_silver detect 규칙 수: %s (무시됨: %s)", len(detect_rules), ignored_rules)
    logger.info("build_silver verify 규칙 수: %s (무시됨: %s)", len(verify_rules), ignored_verify)
    logger.info(
        "build_silver verify morph 규칙 수: %s (미지원: %s)",
        len(morph_verify_rules),
        n_verify_rules_skipped_morph_unsupported,
    )
    logger.info("build_silver context 규칙 수: %s (무시됨: %s)", len(context_rules), ignored_context)

    n_input_sentences = 0
    n_output_records = 0
    triage_counts = {"discard": 0}
    n_candidates_total = 0
    triage_counts_candidates = {"confirm": 0, "hold": 0, "discard": 0}
    n_sentences_with_candidates = 0
    n_verify_rules_applied = 0
    n_verify_rules_applied_morph = 0
    n_candidates_discarded_by_hard_fail = 0
    score_deltas: list[int] = []
    n_context_rules_applied = 0
    n_context_pos_hits = 0
    n_context_neg_hits = 0
    score_deltas_context: list[int] = []
    triage_transition_counts = _init_triage_transition_counts()
    n_triage_changed_total = 0
    n_span_competition_groups = 0
    n_candidates_downgraded_by_competition = 0
    n_detect_rules_matched_regex = 0
    n_detect_rules_with_any_match = 0
    n_detect_regex_match_spans = 0
    n_detect_candidates_total = 0
    n_detect_candidates_components = 0
    n_detect_candidates_fallback_match_span = 0
    n_detect_components_span_fail = 0
    n_detect_fl_treated_as_fx = 0
    n_detect_optional_ignored = 0
    n_detect_disconti_generated = 0
    n_candidates_with_span_text_mismatch = 0
    n_component_match_special_adnominal = 0
    n_component_match_special_nde = 0
    n_component_match_thing_bridge = 0
    n_component_match_choose_nde_over_normal = 0
    n_component_match_choose_adnominal_over_normal = 0
    thing_bridge_fused = 0
    thing_bridge_form_counts = {"거": 0, "게": 0, "건": 0, "걸": 0}
    thing_bridge_fused = 0
    thing_bridge_form_counts = {"거": 0, "게": 0, "건": 0, "걸": 0}
    n_component_match_order_bounded = 0
    n_component_span_fail_required = 0
    detect_components_span_fail_samples: list[dict[str, Any]] = []
    morph_enabled = bool(cfg.get("silver", {}).get("morph", {}).get("enabled", True))
    morph_window_chars = int(cfg.get("silver", {}).get("morph", {}).get("window_chars", 20))
    verify_window_chars = int(cfg.get("verify", {}).get("window_chars", morph_window_chars))
    logger.info(
        "[silver][verify_window] window_chars=%s (fallback=morph_window_chars=%s)",
        verify_window_chars,
        morph_window_chars,
    )
    kiwi_model = str(cfg.get("silver", {}).get("morph", {}).get("kiwi_model", "cong-global"))
    morph_dump_tokens = bool(cfg.get("silver", {}).get("morph", {}).get("dump_tokens", False))
    morph_dump_max_tokens = int(
        cfg.get("silver", {}).get("morph", {}).get("dump_max_tokens_per_sentence", 200)
    )
    morph_dump_max_sentences = int(
        cfg.get("silver", {}).get("morph", {}).get("dump_max_sentences", 200)
    )
    pos_mapper, pos_mapping_source = _load_pos_mapping(cfg, run_context, logger)
    n_sentences_morph_analyzed = 0
    n_morph_tokens_total = 0

    expredict_map = {row.get("e_id"): row for row in dict_bundle.get("expredict", [])}
    components_by_eid = _index_components_by_eid(dict_bundle.get("components", []))
    thresholds = cfg.get("silver", {}).get("triage_thresholds", {})
    confirm_min_score = int(thresholds.get("confirm_min_score", 3))
    hold_min_score = int(thresholds.get("hold_min_score", 1))
    context_window_chars = int(cfg.get("silver", {}).get("context_window_chars", 40))

    morph_dump_count = 0
    morph_dump_truncated = False
    morph_dump_path = outputs_dir / "morph_tokens.jsonl"
    morph_dump_fp = None
    if morph_enabled and morph_dump_tokens:
        morph_dump_fp = morph_dump_path.open("w", encoding="utf-8")

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            n_input_sentences += 1
            raw_sentence = record.get("target_sentence", "")
            morph_tokens: list[dict[str, Any]] = []
            if morph_enabled:
                try:
                    morph_tokens = analyze_with_kiwi(raw_sentence, model=kiwi_model)
                except Exception as exc:
                    logger.warning("morph 분석 실패: %s", exc, exc_info=True)
                    morph_tokens = []
                for token in morph_tokens:
                    token["pos_std"] = pos_mapper(str(token.get("pos", "")))
                n_sentences_morph_analyzed += 1
                n_morph_tokens_total += len(morph_tokens)
                if morph_dump_fp and not morph_dump_truncated:
                    if morph_dump_count >= morph_dump_max_sentences:
                        morph_dump_truncated = True
                    else:
                        truncated = False
                        tokens_to_dump = morph_tokens
                        if len(tokens_to_dump) > morph_dump_max_tokens:
                            tokens_to_dump = tokens_to_dump[:morph_dump_max_tokens]
                            truncated = True
                        write_jsonl_line(
                            morph_dump_fp,
                            {
                                "doc_id": record.get("doc_id"),
                                "sent_index": record.get("sent_index"),
                                "target_sentence": raw_sentence,
                                "morph_tokens": tokens_to_dump,
                                "debug": {"truncated": truncated},
                            },
                        )
                        morph_dump_count += 1

            detect_match_window_chars = int(
                cfg.get("silver", {}).get("detect_match_window_chars", 12)
            )
            detect_max_matches_per_rule = int(
                cfg.get("silver", {}).get("detect_max_matches_per_rule", 50)
            )
            detect_result = _detect_candidates(
                raw_sentence,
                detect_rules,
                expredict_map,
                confirm_min_score,
                hold_min_score,
                **_build_detect_kwargs(
                    record=record,
                    raw_sentence=raw_sentence,
                    components_by_eid=components_by_eid,
                    morph_tokens=morph_tokens,
                    detect_match_window_chars=detect_match_window_chars,
                    detect_max_matches_per_rule=detect_max_matches_per_rule,
                ),
            )
            candidates = detect_result["candidates"]
            n_detect_rules_matched_regex += detect_result.get("n_rules_matched_regex", 0)
            n_detect_rules_with_any_match += detect_result.get(
                "n_detect_rules_with_any_match", 0
            )
            n_detect_regex_match_spans += detect_result.get(
                "n_detect_regex_match_spans", 0
            )
            n_detect_candidates_total += detect_result.get("n_candidates_total", 0)
            n_detect_candidates_components += detect_result.get("n_candidates_components", 0)
            n_detect_candidates_fallback_match_span += detect_result.get(
                "n_candidates_fallback_match_span", 0
            )
            n_detect_components_span_fail += detect_result.get("n_components_span_fail", 0)
            n_detect_fl_treated_as_fx += detect_result.get("n_fl_treated_as_fx", 0)
            n_detect_optional_ignored += detect_result.get("n_optional_ignored", 0)
            n_detect_disconti_generated += detect_result.get("n_disconti_generated", 0)
            n_candidates_with_span_text_mismatch += detect_result.get(
                "n_candidates_with_span_text_mismatch", 0
            )
            n_component_match_special_adnominal += detect_result.get(
                "n_component_match_special_adnominal", 0
            )
            n_component_match_special_nde += detect_result.get("n_component_match_special_nde", 0)
            n_component_match_thing_bridge += detect_result.get(
                "n_component_match_thing_bridge", 0
            )
            n_component_match_choose_nde_over_normal += detect_result.get(
                "n_component_match_choose_nde_over_normal", 0
            )
            n_component_match_choose_adnominal_over_normal += detect_result.get(
                "n_component_match_choose_adnominal_over_normal", 0
            )
            thing_bridge_fused += detect_result.get("thing_bridge_fused", 0)
            form_counts = detect_result.get("thing_bridge_form_counts", {})
            for form in thing_bridge_form_counts:
                thing_bridge_form_counts[form] += int(form_counts.get(form, 0) or 0)
            n_component_match_order_bounded += detect_result.get(
                "n_component_match_order_bounded", 0
            )
            n_component_span_fail_required += detect_result.get(
                "n_component_span_fail_required", 0
            )
            detect_components_span_fail_samples.extend(
                detect_result.get("detect_components_span_fail_samples", [])
            )

            verify_result = _apply_verify_rules(
                raw_sentence,
                candidates,
                verify_rules,
                morph_verify_rules,
                morph_tokens,
                confirm_min_score,
                hold_min_score,
                morph_window_chars,
                verify_window_chars,
            )
            n_verify_rules_applied += verify_result["n_verify_rules_applied"]
            n_verify_rules_applied_morph += verify_result["n_verify_rules_applied_morph"]
            n_candidates_discarded_by_hard_fail += verify_result["n_candidates_discarded_by_hard_fail"]
            score_deltas.extend(verify_result["score_deltas"])

            context_result = _apply_context_rules(
                raw_sentence,
                candidates,
                context_rules,
                confirm_min_score,
                hold_min_score,
                context_window_chars,
            )
            n_context_rules_applied += context_result["n_context_rules_applied"]
            n_context_pos_hits += context_result["n_context_pos_hits"]
            n_context_neg_hits += context_result["n_context_neg_hits"]
            score_deltas_context.extend(context_result["score_deltas"])
            n_triage_changed_total += context_result["n_triage_changed_total"]
            _merge_transition_counts(triage_transition_counts, context_result["triage_transition_counts"])

            competition_result = _apply_span_competition_guard(
                candidates,
                triage_transition_counts,
            )
            n_span_competition_groups += competition_result["n_span_competition_groups"]
            n_candidates_downgraded_by_competition += competition_result["n_candidates_downgraded_by_competition"]
            n_triage_changed_total += competition_result["n_triage_changed_total"]

            if candidates:
                n_sentences_with_candidates += 1
                n_candidates_total += len(candidates)
                for candidate in candidates:
                    triage_counts_candidates[candidate["triage"]] += 1

            # silver_labels = confirm만 저장 (train_weak에 바로 투입 가능한 최소 스키마)
            silver_labels = []
            for cand in sorted(candidates, key=lambda c: (str(c.get("e_id", "")), str(c.get("span_key", "")))):
                if cand.get("triage") != "confirm":
                    continue
                if cand.get("hard_fail_triggered", False):
                    continue
                silver_labels.append(
                    {
                        "e_id": cand["e_id"],
                        "span_segments": cand["span_segments"],
                        "span_key": cand["span_key"],
                        "span_text": cand.get("span_text", ""),
                        "score": cand.get("score", 0),
                        "stage_hits": cand.get("stage_hits", {}),
                    }
                )

            if silver_labels:
                record_triage = "confirm"
            elif any(c.get("triage") == "hold" for c in candidates):
                record_triage = "hold"
            else:
                record_triage = "discard"

            output_record = {
                "doc_id": record.get("doc_id"),
                "sent_id": record.get("sent_id"),
                "sent_index": record.get("sent_index"),
                "example_id": record.get("example_id"),
                "instance_id": record.get("instance_id"),
                "target_sentence": record.get("target_sentence"),
                "source": record.get("source", ""),
                "meta": record.get("meta", {}),
                "candidates": candidates,
                "silver_labels": silver_labels,
                "triage": record_triage,
                "debug": {
                    "note": "build_silver",
                    "n_candidates": len(candidates),
                    "n_confirm": sum(1 for c in candidates if c.get("triage") == "confirm"),
                    "n_hold": sum(1 for c in candidates if c.get("triage") == "hold"),
                    "n_discard": sum(1 for c in candidates if c.get("triage") == "discard"),
                },
            }
            write_jsonl_line(f_out, output_record)
            n_output_records += 1
            triage_counts.setdefault(record_triage, 0)
            triage_counts[record_triage] += 1

    avg_morph_tokens_per_sentence = 0.0
    if n_sentences_morph_analyzed > 0:
        avg_morph_tokens_per_sentence = float(n_morph_tokens_total) / n_sentences_morph_analyzed

    report = {
        "input_jsonl_path": str(input_path),
        "input_path_source": input_path_source,
        "input_path_forced": input_path_forced,
        "dict_xlsx": cfg.get("paths", {}).get("dict_xlsx"),
        "exp_id": run_context.exp_id,
        "run_id": run_context.run_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "n_input_sentences": n_input_sentences,
        "n_records_loaded": n_input_sentences,
        "n_output_records": n_output_records,
        "triage_counts": triage_counts,
        "n_candidates_total": n_candidates_total,
        "triage_counts_candidates": triage_counts_candidates,
        "n_sentences_with_candidates": n_sentences_with_candidates,
        "n_verify_rules_applied": n_verify_rules_applied,
        "morph_enabled": morph_enabled,
        "pos_mapping_source": pos_mapping_source,
        "n_sentences_morph_analyzed": n_sentences_morph_analyzed,
        "avg_morph_tokens_per_sentence": avg_morph_tokens_per_sentence,
        "n_verify_rules_supported_surface": n_verify_rules_supported_surface,
        "n_verify_rules_supported_surface_raw_sentence": n_verify_rules_supported_surface_raw_sentence,
        "n_verify_rules_supported_surface_token_window": n_verify_rules_supported_surface_token_window,
        "n_verify_rules_supported_morph": len(morph_verify_rules),
        "n_verify_rules_applied_morph": n_verify_rules_applied_morph,
        "n_verify_rules_skipped_morph_unsupported": n_verify_rules_skipped_morph_unsupported,
        "morph_dump_enabled": morph_enabled and morph_dump_tokens,
        "morph_dump_path": str(morph_dump_path) if morph_enabled and morph_dump_tokens else None,
        "morph_dump_count": morph_dump_count,
        "morph_dump_truncated": morph_dump_truncated,
        "n_candidates_discarded_by_hard_fail": n_candidates_discarded_by_hard_fail,
        "score_delta_summary": _summarize_deltas(score_deltas),
        "n_context_rules_filtered": len(context_rules),
        "n_context_rules_ignored": ignored_context,
        "n_context_rules_applied": n_context_rules_applied,
        "n_context_rules_applied_windowed": n_context_rules_applied,
        "n_context_pos_hits": n_context_pos_hits,
        "n_context_neg_hits": n_context_neg_hits,
        "triage_transition_counts": triage_transition_counts,
        "n_triage_changed_total": n_triage_changed_total,
        "score_delta_summary_context": _summarize_deltas(score_deltas_context),
        "context_window_chars": context_window_chars,
        "n_span_competition_groups": n_span_competition_groups,
        "n_candidates_downgraded_by_competition": n_candidates_downgraded_by_competition,
        "n_detect_rules_total": len(detect_rules),
        "n_detect_rules_matched_regex": n_detect_rules_matched_regex,
        "n_detect_rules_with_any_match": n_detect_rules_with_any_match,
        "n_detect_regex_match_spans": n_detect_regex_match_spans,
        "n_detect_candidates_total": n_detect_candidates_total,
        "n_detect_candidates_components": n_detect_candidates_components,
        "n_detect_candidates_fallback_match_span": n_detect_candidates_fallback_match_span,
        "n_detect_components_span_fail": n_detect_components_span_fail,
        "n_detect_fl_treated_as_fx": n_detect_fl_treated_as_fx,
        "n_detect_optional_ignored": n_detect_optional_ignored,
        "n_detect_disconti_generated": n_detect_disconti_generated,
        "n_candidates_with_span_text_mismatch": n_candidates_with_span_text_mismatch,
        "n_component_match_special_adnominal": n_component_match_special_adnominal,
        "n_component_match_special_nde": n_component_match_special_nde,
        "n_component_match_thing_bridge": n_component_match_thing_bridge,
        "n_component_match_choose_nde_over_normal": n_component_match_choose_nde_over_normal,
        "n_component_match_choose_adnominal_over_normal": n_component_match_choose_adnominal_over_normal,
        "thing_bridge_fused": thing_bridge_fused,
        "thing_bridge_form_counts": thing_bridge_form_counts,
        "n_component_match_order_bounded": n_component_match_order_bounded,
        "n_component_span_fail_required": n_component_span_fail_required,
        "detect_components_span_fail_samples": detect_components_span_fail_samples[:20],
        "dict_source": dict_source,
        **dict_stats,
    }
    if morph_dump_fp:
        morph_dump_fp.close()
    report["n_component_match_special_geot"] = report["n_component_match_thing_bridge"]
    report["special_geot_form_counts"] = report["thing_bridge_form_counts"]
    report["special_geot_fused"] = report["thing_bridge_fused"]
    report.setdefault("warnings", []).append(
        "DEPRECATED: 'special_geot*' keys will be removed; use 'thing_bridge*' keys."
    )
    write_json(report_path, report, indent=2)
    for_users_cfg = cfg.get("build_silver", {}).get("for_users", {}) or {}
    export_enabled = bool(for_users_cfg.get("enabled", True))
    out_path = str(for_users_cfg.get("out_path", "for_users/build_silver_latest.csv"))
    if export_enabled:
        out_csv = Path(out_path)
        if not out_csv.is_absolute():
            out_csv = outputs_dir / out_csv
        assert_under_dir(out_csv, outputs_dir, what="build_silver for_users")
        try:
            info = export_silver_user_csv(
                silver_jsonl_path=str(output_path),
                out_csv_path=str(out_csv),
                run_id=run_context.run_id,
                exp_id=run_context.exp_id,
                stage="build_silver",
                input_jsonl_path=report.get("input_jsonl_path"),
                report_json_path=str(report_path),
            )
            report["for_users_csv_path"] = info.get("out_csv_path", str(out_csv))
            report["for_users_csv_rows"] = info.get("n_sentences", 0)
            report["for_users_csv_schema_version"] = str(
                for_users_cfg.get("columns_version", "v1")
            )
        except Exception as exc:
            logger.warning("for_users csv export failed: %s", exc, exc_info=True)
    write_json(report_path, report, indent=2)

    logger.info("build_silver 입력 ingest jsonl 경로: %s", input_path)
    logger.info("silver.jsonl 생성 경로: %s", output_path)
    if n_candidates_total == 0:
        logger.warning("build_silver 후보가 0개입니다. 규칙/패턴/입력을 확인하세요.")


def _resolve_input_path(
    cfg: dict[str, Any], run_context: RunContext, logger: logging.Logger
) -> tuple[Path, str, bool]:
    cfg, forced_path, forced_source = apply_forced_input_jsonl(cfg, stage="build_silver")
    configured = (
        cfg.get("silver", {}).get("input_jsonl")
        or cfg.get("build_silver", {}).get("input_jsonl")
        or cfg.get("build_silver", {}).get("input_path")
    )
    input_path_source = None
    input_path_forced = False
    if forced_path:
        configured = forced_path
        input_path_source = forced_source
        input_path_forced = True
    elif configured:
        input_path_source = "silver.input_jsonl"
    if configured:
        logger.info(
            "[paths] stage=build_silver input_path=%s forced=%s source=%s",
            configured,
            input_path_forced,
            input_path_source or "silver.input_jsonl",
        )
        return Path(configured), input_path_source or "silver.input_jsonl", input_path_forced

    ingest_dir = run_context.run_dir.parent.parent / "ingest_corpus"
    if not ingest_dir.exists():
        raise ConfigError("build_silver 입력 경로를 찾을 수 없습니다: silver.input_jsonl 또는 build_silver.input_jsonl을 설정하세요.")

    run_dirs = sorted(
        [p for p in ingest_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise ConfigError("ingest_corpus run 디렉터리를 찾을 수 없습니다. 입력 JSONL 경로를 명시하세요.")

    selected_run = run_dirs[0]
    candidate = selected_run / "outputs" / "ingest_corpus.jsonl"
    logger.info("자동 선택된 ingest run_dir: %s", selected_run)
    logger.info("선택된 input_jsonl 경로: %s", candidate)
    input_path_source = "auto_latest"
    logger.info(
        "[paths] stage=build_silver input_path=%s forced=%s source=%s",
        candidate,
        input_path_forced,
        input_path_source,
    )
    return candidate, input_path_source, input_path_forced


def _load_pos_mapping(
    cfg: dict[str, Any], run_context: RunContext, logger: logging.Logger
) -> tuple[Callable[[str], str], str]:
    mode = cfg.get("silver", {}).get("morph", {}).get("pos_mapping", "auto")
    mapping_table: dict[str, Any] | None = None
    source = "fallback"
    if isinstance(mode, str) and mode not in {"fallback", "none"}:
        if mode == "auto":
            artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts"))
            mapping_dir = artifacts_dir / run_context.exp_id / "pos_mapping"
            mapping_path = _find_latest_mapping_path(mapping_dir)
            if mapping_path and mapping_path.exists():
                mapping_table = json.loads(mapping_path.read_text(encoding="utf-8"))
                source = "pos_mapping"
        else:
            mapping_path = Path(mode)
            if mapping_path.exists():
                mapping_table = json.loads(mapping_path.read_text(encoding="utf-8"))
                source = "pos_mapping"

    if mapping_table:
        return _build_pos_mapper(mapping_table), source

    try:
        from kmwe.stages.pos_mapping import map_pos as fallback_mapper

        return fallback_mapper, "fallback"
    except Exception as exc:
        logger.warning("pos_mapping fallback 로드 실패: %s", exc, exc_info=True)
        return lambda _tag: "UNK", "fallback"


def _find_latest_mapping_path(mapping_dir: Path) -> Path | None:
    if not mapping_dir.exists():
        return None
    run_dirs = sorted(
        [p for p in mapping_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        candidate = run_dir / "outputs" / "pos_mapping.json"
        if candidate.exists():
            return candidate
    return None


def _build_pos_mapper(mapping_table: dict[str, Any]) -> Callable[[str], str]:
    direct_map = mapping_table.get("direct_map", {}) or {}
    prefix_map = mapping_table.get("prefix_map", []) or []
    range_map = mapping_table.get("range_map", {}) or {}
    fallback = mapping_table.get("fallback", "UNK")

    def mapper(tag: str) -> str:
        if tag in direct_map:
            return direct_map[tag]
        for item in prefix_map:
            prefix = item.get("prefix")
            target = item.get("target")
            if prefix and str(tag).startswith(str(prefix)):
                return str(target)
        if isinstance(range_map, dict):
            for key, value in range_map.items():
                if _tag_in_range(str(tag), str(key)):
                    return str(value)
        return str(fallback)

    return mapper


def _tag_in_range(tag: str, range_key: str) -> bool:
    if "-" not in range_key:
        return False
    start, end = range_key.split("-", 1)
    prefix = ""
    for a, b in zip(start, end):
        if a == b and not a.isdigit():
            prefix += a
        else:
            break
    start_num = start[len(prefix) :]
    end_num = end[len(prefix) :]
    if not start_num.isdigit() or not end_num.isdigit():
        return False
    if not tag.startswith(prefix):
        return False
    tag_num = tag[len(prefix) :]
    if not tag_num.isdigit():
        return False
    return int(start_num) <= int(tag_num) <= int(end_num)


def _load_dict_stats(cfg: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dict_bundle_path = cfg.get("silver", {}).get("dict_bundle_path")
    if dict_bundle_path:
        path = Path(dict_bundle_path)
        if not path.exists():
            raise ConfigError(f"dict_bundle_path 파일이 존재하지 않습니다: {path}")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        return "dict_bundle_path", _stats_from_bundle(bundle), bundle

    dict_xlsx = cfg.get("paths", {}).get("dict_xlsx")
    if not dict_xlsx:
        raise ConfigError("dict_xlsx 경로가 필요합니다: paths.dict_xlsx를 설정하세요.")
    frames = validate_dict_loader._load_dict_xlsx(Path(dict_xlsx), _noop_issue)  # type: ignore[attr-defined]
    if frames is None:
        raise ConfigError(f"dict_xlsx 로딩에 실패했습니다: {dict_xlsx}")
    bundle = _bundle_from_frames(frames)
    return "dict_xlsx", _stats_from_frames(frames), bundle


def _stats_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    expredict = bundle.get("expredict", []) or []
    rules = bundle.get("rules", []) or []
    components = bundle.get("components", []) or []
    n_group_a = sum(1 for row in expredict if str(row.get("group", "")).lower() == "a")
    n_group_b = sum(1 for row in expredict if str(row.get("group", "")).lower() == "b")
    stage_counts = _count_rules_by_stage(rules)
    return {
        "n_patterns_total": len(expredict),
        "n_group_a": n_group_a,
        "n_group_b": n_group_b,
        "n_components_total": len(components),
        "n_rules_total": len(rules),
        "n_rules_by_stage": stage_counts,
    }


def _stats_from_frames(frames: dict[str, Any]) -> dict[str, Any]:
    expredict = frames["expredict"]
    rules = frames["rules"]
    components = frames.get("components")
    n_group_a = sum(1 for _, row in expredict.iterrows() if str(row.get("group", "")).lower() == "a")
    n_group_b = sum(1 for _, row in expredict.iterrows() if str(row.get("group", "")).lower() == "b")
    stage_counts = _count_rules_by_stage(
        [row.to_dict() for _, row in rules.iterrows()]
    )
    n_components_total = len(components) if components is not None else 0
    return {
        "n_patterns_total": len(expredict),
        "n_group_a": n_group_a,
        "n_group_b": n_group_b,
        "n_components_total": n_components_total,
        "n_rules_total": len(rules),
        "n_rules_by_stage": stage_counts,
    }


def _bundle_from_frames(frames: dict[str, Any]) -> dict[str, Any]:
    components = frames.get("components")
    component_keys = {
        "e_id",
        "comp_id",
        "comp_surf",
        "comp_order",
        "is_required",
        "anchor_rank",
        "order_policy",
        "min_gap_to_next",
        "max_gap_to_next",
    }
    components_rows = []
    if components is not None:
        for _, row in components.iterrows():
            item = row.to_dict()
            components_rows.append({key: item.get(key) for key in component_keys})
    expredict_rows = []
    for _, row in frames["expredict"].iterrows():
        item = row.to_dict()
        expredict_rows.append(validate_dict_loader._ensure_sheet1_keys(item))  # type: ignore[attr-defined]
    expredict_map = {
        row.get("e_id"): row for row in expredict_rows if row.get("e_id")
    }
    return {
        "expredict": expredict_rows,
        "expredict_map": expredict_map,
        "rules": [row.to_dict() for _, row in frames["rules"].iterrows()],
        "components": components_rows,
    }


def _filter_detect_rules(rules: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    filtered = []
    ignored = 0
    for rule in rules:
        if (
            str(rule.get("stage", "")).lower() == "detect"
            and str(rule.get("rule_type", "")).lower() == "surface_regex"
            and str(rule.get("engine", "")).lower() == "re"
            and str(rule.get("target", "")).lower() == "raw_sentence"
        ):
            filtered.append(rule)
        else:
            ignored += 1
    return filtered, ignored


def _split_verify_rules(
    rules: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    filtered = []
    morph_filtered = []
    ignored = 0
    morph_unsupported = 0
    for rule in rules:
        if str(rule.get("stage", "")).lower() != "verify":
            ignored += 1
            continue
        rule_type = str(rule.get("rule_type", "")).lower()
        engine = str(rule.get("engine", "")).lower()
        target = str(rule.get("target", "")).lower()
        if rule_type == "surface_regex" and engine == "re" and target in {
            "raw_sentence",
            "token_window",
        }:
            filtered.append(rule)
            continue
        if target == "morph_tokens":
            if engine == "json" and rule_type in {"pos_seq", "morph_check"}:
                morph_filtered.append(rule)
            else:
                morph_unsupported += 1
            continue
        ignored += 1
    return filtered, morph_filtered, ignored, morph_unsupported


def _filter_context_rules(rules: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    filtered = []
    ignored = 0
    for rule in rules:
        stage = str(rule.get("stage", "")).lower()
        rule_type = str(rule.get("rule_type", "")).lower()
        if (
            stage == "context"
            and rule_type in {"context_pos_regex", "context_neg_regex"}
            and str(rule.get("engine", "")).lower() == "re"
            and str(rule.get("target", "")).lower() == "raw_sentence"
        ):
            filtered.append(rule)
        else:
            ignored += 1
    return filtered, ignored


def _filter_rules_by_scope(
    rules: Iterable[dict[str, Any]],
    *,
    allowed_scopes: set[str] | None = None,
) -> list[dict[str, Any]]:
    if allowed_scopes is None:
        return list(rules)
    normalized = {str(scope).strip().lower() for scope in allowed_scopes}
    return [
        rule
        for rule in rules
        if str(rule.get("scope", "")).strip().lower() in normalized
    ]


def _prepare_stage_rules(
    rules: Iterable[dict[str, Any]],
    *,
    allowed_scopes: set[str] | None = None,
) -> dict[str, Any]:
    scoped_rules = _filter_rules_by_scope(rules, allowed_scopes=allowed_scopes)
    detect_rules, ignored_rules = _filter_detect_rules(scoped_rules)
    verify_rules, morph_verify_rules, ignored_verify, n_verify_rules_skipped_morph_unsupported = (
        _split_verify_rules(scoped_rules)
    )
    context_rules, ignored_context = _filter_context_rules(scoped_rules)
    return {
        "scoped_rules": scoped_rules,
        "detect_rules": detect_rules,
        "verify_rules": verify_rules,
        "morph_verify_rules": morph_verify_rules,
        "context_rules": context_rules,
        "ignored_rules": ignored_rules,
        "ignored_verify": ignored_verify,
        "ignored_context": ignored_context,
        "n_verify_rules_skipped_morph_unsupported": n_verify_rules_skipped_morph_unsupported,
    }


def _build_detect_kwargs(
    *,
    record: dict[str, Any],
    raw_sentence: str,
    components_by_eid: dict[str, list[dict[str, Any]]],
    morph_tokens: list[dict[str, Any]] | None,
    detect_match_window_chars: int,
    detect_max_matches_per_rule: int,
    include_debug_ctx: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "components_by_eid": components_by_eid,
        "morph_tokens": morph_tokens,
        "detect_match_window_chars": detect_match_window_chars,
        "detect_max_matches_per_rule": detect_max_matches_per_rule,
        "record_meta": {
            "doc_id": record.get("doc_id"),
            "sent_index": record.get("sent_index"),
            "example_id": record.get("example_id"),
            "instance_id": record.get("instance_id"),
            "target_sentence": raw_sentence,
        },
    }
    if include_debug_ctx:
        kwargs["debug_ctx"] = {
            "example_id": record.get("example_id"),
            "instance_id": record.get("instance_id"),
        }
    return kwargs


def _detect_candidates(
    raw_sentence: str,
    rules: list[dict[str, Any]],
    expredict_map: dict[str, dict[str, Any]],
    confirm_min_score: int,
    hold_min_score: int,
    components_by_eid: dict[str, list[dict[str, Any]]] | None = None,
    detect_match_window_chars: int = 12,
    detect_max_matches_per_rule: int = 50,
    morph_tokens: list[dict[str, Any]] | None = None,
    record_meta: dict[str, Any] | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logger = logging.getLogger("kmwe")
    debug_example = os.getenv("KMWE_DEBUG_EXAMPLE")
    debug_eid = os.getenv("KMWE_DEBUG_EID")

    def _debug_match(e_id_value: Any) -> bool:
        if not (debug_example and debug_eid and debug_ctx):
            return False
        example_key = f"{debug_ctx.get('example_id')}#{debug_ctx.get('instance_id')}"
        return example_key == debug_example and str(e_id_value) == str(debug_eid)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    n_rules_matched_regex = 0
    n_rules_with_any_match = 0
    n_regex_match_spans = 0
    n_components_span_fail = 0
    n_candidates_components = 0
    n_candidates_fallback_match_span = 0
    n_disconti_generated = 0
    n_fl_treated_as_fx = 0
    n_optional_ignored = 0
    n_candidates_with_span_text_mismatch = 0
    n_component_match_special_adnominal = 0
    n_component_match_special_nde = 0
    n_component_match_thing_bridge = 0
    n_component_match_choose_nde_over_normal = 0
    n_component_match_choose_adnominal_over_normal = 0
    thing_bridge_fused = 0
    thing_bridge_form_counts = {"거": 0, "게": 0, "건": 0, "걸": 0}
    n_component_match_order_bounded = 0
    n_component_span_fail_required = 0
    components_span_fail_samples: list[dict[str, Any]] = []
    for rule in rules:
        e_id = rule.get("e_id")
        pattern = rule.get("pattern")
        if not e_id or not pattern:
            continue
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        expredict_row = expredict_map.get(e_id, {}) or {}
        base = expredict_row.get("default_confidence", 0) or 0
        delta = rule.get("confidence_delta", 0) or 0
        score = int(base) + int(delta)
        rule_id = rule.get("rule_id")
        disconti_allowed = str(expredict_row.get("disconti_allowed", "")).lower() == "y"
        spacing_policy = str(expredict_row.get("spacing_policy", "nrm"))
        rule_match_found = False
        for match_idx, match in enumerate(regex.finditer(raw_sentence)):
            if match_idx >= detect_max_matches_per_rule:
                break
            rule_match_found = True
            n_regex_match_spans += 1
            detect_window = _build_match_window(
                match.span(), len(raw_sentence), detect_match_window_chars
            )
            record_meta = record_meta or {}
            base_fail_fields = {
                "doc_id": record_meta.get("doc_id"),
                "sent_index": record_meta.get("sent_index"),
                "example_id": record_meta.get("example_id"),
                "instance_id": record_meta.get("instance_id"),
                "target_sentence": record_meta.get("target_sentence"),
            }
            if _debug_match(e_id):
                logger.info(
                    "[DBG:C2] ex=%s e_id=%s match_span=%s detect_window=%s",
                    f"{debug_ctx.get('example_id')}#{debug_ctx.get('instance_id')}",
                    e_id,
                    [match.start(), match.end()],
                    [detect_window[0], detect_window[1]],
                )
            if not components_by_eid:
                n_components_span_fail += 1
                if len(components_span_fail_samples) < 20:
                    components_span_fail_samples.append(
                        {
                            "e_id": str(e_id),
                            "rule_id": rule_id,
                            "ruleset_id": rule.get("ruleset_id"),
                            "match_span": [match.start(), match.end()],
                            "match_text": match.group(0),
                            "detect_window": [detect_window[0], detect_window[1]],
                            "anchor_hint_span": [match.start(), match.end()],
                            "anchor_selected_span": None,
                            "anchor_selected_kind": None,
                            "note": "components span empty (no components index, anchor_hint_applied=True)",
                            **base_fail_fields,
                        }
                    )
                continue
            comps_for_eid = components_by_eid.get(str(e_id), [])
            if not comps_for_eid:
                n_components_span_fail += 1
                if len(components_span_fail_samples) < 20:
                    components_span_fail_samples.append(
                        {
                            "e_id": str(e_id),
                            "rule_id": rule_id,
                            "ruleset_id": rule.get("ruleset_id"),
                            "match_span": [match.start(), match.end()],
                            "match_text": match.group(0),
                            "detect_window": [detect_window[0], detect_window[1]],
                            "anchor_hint_span": [match.start(), match.end()],
                            "anchor_selected_span": None,
                            "anchor_selected_kind": None,
                            "note": "components span empty (no components for e_id, anchor_hint_applied=True)",
                            **base_fail_fields,
                        }
                    )
                continue
            span_segments_list, meta = _locate_components_spans(
                raw_sentence,
                str(e_id),
                comps_for_eid,
                anchor_strategy="gat_or_best",
                spacing_policy=spacing_policy,
                disconti_allowed=disconti_allowed,
                expredict_row=expredict_row,
                detect_window=detect_window,
                morph_tokens=morph_tokens,
                anchor_hint_span=(match.start(), match.end()),
                debug_ctx={**(debug_ctx or {}), "e_id": str(e_id)},
            )
            if _debug_match(e_id):
                logger.info(
                    "[DBG:C2] ex=%s e_id=%s span_segments_n=%s meta=%s",
                    f"{debug_ctx.get('example_id')}#{debug_ctx.get('instance_id')}",
                    e_id,
                    len(span_segments_list),
                    {
                        "anchor_hint_applied": meta.get("anchor_hint_applied"),
                        "anchor_selected_comp_id": meta.get("anchor_selected_comp_id"),
                        "per_comp_selected": meta.get("per_comp_selected"),
                        "gap_violations": meta.get("gap_violations"),
                        "span_fail_required": meta.get("span_fail_required"),
                    },
                )
            n_fl_treated_as_fx += meta.get("fl_treated_as_fx", 0)
            n_optional_ignored += meta.get("optional_ignored", 0)
            n_component_match_special_adnominal += meta.get("special_adnominal", 0)
            n_component_match_special_nde += meta.get("special_nde", 0)
            n_component_match_thing_bridge += meta.get("thing_bridge", 0)
            n_component_match_choose_nde_over_normal += meta.get("choose_nde_over_normal", 0)
            n_component_match_choose_adnominal_over_normal += meta.get(
                "choose_adnominal_over_normal", 0
            )
            thing_bridge_fused += meta.get("thing_bridge_fused", 0)
            form_counts = meta.get("thing_bridge_form_counts", {})
            for form in thing_bridge_form_counts:
                thing_bridge_form_counts[form] += int(form_counts.get(form, 0) or 0)
            n_component_match_order_bounded += meta.get("order_bounded", 0)
            n_component_span_fail_required += meta.get("span_fail_required", 0)
            if not span_segments_list:
                n_components_span_fail += 1
                if len(components_span_fail_samples) < 20:
                    per_comp_debug = meta.get("per_comp_debug", {})
                    trimmed_per_comp_debug = {}
                    for comp_id, debug in per_comp_debug.items():
                        if not isinstance(debug, dict):
                            continue
                        trimmed = dict(debug)
                        candidates_top = trimmed.get("candidates_top") or []
                        trimmed["candidates_top"] = list(candidates_top)[:5]
                        trimmed_per_comp_debug[comp_id] = trimmed
                    gap_violations = meta.get("gap_violations", [])
                    gap_violations = list(gap_violations)[:10]
                    morph_snippet = meta.get("morph_token_snippet", [])
                    morph_snippet = list(morph_snippet)[:40]
                    components_span_fail_samples.append(
                        {
                            "e_id": str(e_id),
                            "rule_id": rule_id,
                            "ruleset_id": rule.get("ruleset_id"),
                            "match_span": [match.start(), match.end()],
                            "match_text": match.group(0),
                            "detect_window": [detect_window[0], detect_window[1]],
                            "anchor_hint_span": [match.start(), match.end()],
                            "anchor_selected_span": meta.get("anchor_selected_span"),
                            "anchor_selected_kind": meta.get("anchor_selected_kind"),
                            "failed_required_comp_ids": meta.get("failed_required_comp_ids", []),
                            "failed_optional_comp_ids": meta.get("failed_optional_comp_ids", []),
                            "per_comp_debug": trimmed_per_comp_debug,
                            "gap_violations": gap_violations,
                            "search_ranges": meta.get("search_ranges", {}),
                            "special_candidate_counts": meta.get("special_candidate_counts", {}),
                            "morph_token_snippet": morph_snippet,
                            "note": "components span empty (anchor_hint_applied=True)",
                            **base_fail_fields,
                        }
                    )
                continue
            for span_segments in span_segments_list:
                span_key = _span_key_from_segments(span_segments)
                key = (str(e_id), span_key)
                if key not in merged:
                    candidate = _create_candidate_from_span_segments(
                        e_id=str(e_id),
                        span_segments=span_segments,
                        raw_sentence=raw_sentence,
                    )
                    if meta.get("component_match_notes"):
                        candidate["component_match_notes"] = list(
                            meta.get("component_match_notes", [])
                        )
                    morph_snippet, morph_window = _make_morph_snippet(
                        morph_tokens, span_segments=span_segments, window_chars=20
                    )
                    candidate["debug_meta"] = {
                        "detect": {
                            "rule_id": rule_id,
                            "ruleset_id": rule.get("ruleset_id"),
                            "match_span": [match.start(), match.end()],
                            "match_text": match.group(0),
                            "detect_window": [detect_window[0], detect_window[1]],
                            "anchor_hint_span": [match.start(), match.end()],
                            "anchor_selected_span": meta.get("anchor_selected_span"),
                            "anchor_selected_kind": meta.get("anchor_selected_kind"),
                        },
                        "components_debug": {
                            "failed_required_comp_ids": meta.get("failed_required_comp_ids", []),
                            "gap_violations": meta.get("gap_violations", []),
                            "per_comp": meta.get("per_comp_debug", {}),
                        },
                        "bridge": meta.get("bridge_detail"),
                        "thing_bridge": meta.get("thing_bridge_detail"),
                        "morph_snippet": morph_snippet,
                        "morph_snippet_window": morph_window,
                    }
                    merged[key] = candidate
                    n_candidates_components += 1
                    if len(span_segments) > 1:
                        n_disconti_generated += 1
                    expected_text = _span_text_from_segments(raw_sentence, span_segments)
                    if candidate["span_text"] != expected_text:
                        n_candidates_with_span_text_mismatch += 1
                merged[key]["score"] += score
                if rule_id:
                    merged[key]["stage_hits"]["detect"].append(rule_id)
        if rule_match_found:
            n_rules_with_any_match += 1
    n_rules_matched_regex = n_rules_with_any_match

    candidates = list(merged.values())
    for candidate in candidates:
        score = candidate["score"]
        if score >= confirm_min_score:
            candidate["triage"] = "confirm"
        elif score >= hold_min_score:
            candidate["triage"] = "hold"
        else:
            candidate["triage"] = "discard"
    return {
        "candidates": candidates,
        "n_rules_matched_regex": n_rules_matched_regex,
        "n_candidates_total": len(candidates),
        "n_candidates_components": n_candidates_components,
        "n_candidates_fallback_match_span": n_candidates_fallback_match_span,
        "n_components_span_fail": n_components_span_fail,
        "n_fl_treated_as_fx": n_fl_treated_as_fx,
        "n_optional_ignored": n_optional_ignored,
        "n_disconti_generated": n_disconti_generated,
        "n_candidates_with_span_text_mismatch": n_candidates_with_span_text_mismatch,
        "n_component_match_special_adnominal": n_component_match_special_adnominal,
        "n_component_match_special_nde": n_component_match_special_nde,
        "n_component_match_thing_bridge": n_component_match_thing_bridge,
        "n_component_match_choose_nde_over_normal": n_component_match_choose_nde_over_normal,
        "n_component_match_choose_adnominal_over_normal": n_component_match_choose_adnominal_over_normal,
        "thing_bridge_fused": thing_bridge_fused,
        "thing_bridge_form_counts": thing_bridge_form_counts,
        "n_component_match_order_bounded": n_component_match_order_bounded,
        "n_component_span_fail_required": n_component_span_fail_required,
        "n_detect_regex_match_spans": n_regex_match_spans,
        "n_detect_rules_with_any_match": n_rules_with_any_match,
        "detect_components_span_fail_samples": components_span_fail_samples,
    }


def _apply_verify_rules(
    raw_sentence: str,
    candidates: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    morph_rules: list[dict[str, Any]],
    morph_tokens: list[dict[str, Any]],
    confirm_min_score: int,
    hold_min_score: int,
    morph_window_chars: int,
    verify_window_chars: int | None = None,
) -> dict[str, Any]:
    vw = int(verify_window_chars) if verify_window_chars is not None else int(morph_window_chars)
    n_verify_rules_applied = 0
    n_verify_rules_applied_morph = 0
    n_candidates_discarded_by_hard_fail = 0
    score_deltas: list[int] = []

    for candidate in candidates:
        candidate_discarded = False
        for rule in rules:
            e_id = rule.get("e_id")
            if e_id and str(e_id) != candidate.get("e_id"):
                continue
            pattern = rule.get("pattern")
            if not pattern:
                continue
            try:
                regex = re.compile(pattern)
            except re.error:
                continue
            target = str(rule.get("target", "")).lower()
            if target == "token_window":
                window = _build_candidate_window(
                    raw_sentence, candidate, window_chars=vw
                )
                verify_text = window["text"]
            else:
                verify_text = raw_sentence
            if not regex.search(verify_text):
                continue
            n_verify_rules_applied += 1
            rule_id = rule.get("rule_id")
            if rule_id:
                candidate["stage_hits"]["verify"].append(rule_id)
            hard_fail = bool(rule.get("hard_fail", False))
            if hard_fail:
                candidate["hard_fail_triggered"] = True
                if rule_id:
                    candidate["hard_fail_reasons"].append(str(rule_id))
                else:
                    candidate["hard_fail_reasons"].append("verify_hard_fail")
                candidate["triage"] = "discard"
                if not candidate_discarded:
                    n_candidates_discarded_by_hard_fail += 1
                    candidate_discarded = True
                continue
            delta = int(rule.get("confidence_delta", 0) or 0)
            candidate["score"] += delta
            score_deltas.append(delta)

        if candidate.get("hard_fail_triggered"):
            candidate["triage"] = "discard"
            continue

        if morph_rules and morph_tokens is not None:
            window = _build_candidate_window(raw_sentence, candidate, morph_window_chars)
            tokens_in_window = _filter_tokens_by_window(morph_tokens, window["start"], window["end"])
            for rule in morph_rules:
                e_id = rule.get("e_id")
                if e_id and str(e_id) != candidate.get("e_id"):
                    continue
                if not _evaluate_morph_rule(rule, tokens_in_window):
                    continue
                n_verify_rules_applied_morph += 1
                rule_id = rule.get("rule_id")
                if rule_id:
                    candidate["stage_hits"]["verify"].append(rule_id)
                hard_fail = bool(rule.get("hard_fail", False))
                if hard_fail:
                    candidate["hard_fail_triggered"] = True
                    if rule_id:
                        candidate["hard_fail_reasons"].append(str(rule_id))
                    else:
                        candidate["hard_fail_reasons"].append("verify_hard_fail_morph")
                    candidate["triage"] = "discard"
                    if not candidate_discarded:
                        n_candidates_discarded_by_hard_fail += 1
                        candidate_discarded = True
                    continue
                delta = int(rule.get("confidence_delta", 0) or 0)
                candidate["score"] += delta
                score_deltas.append(delta)

        if candidate.get("hard_fail_triggered"):
            candidate["triage"] = "discard"
        else:
            score = candidate["score"]
            if score >= confirm_min_score:
                candidate["triage"] = "confirm"
            elif score >= hold_min_score:
                candidate["triage"] = "hold"
            else:
                candidate["triage"] = "discard"

    return {
        "n_verify_rules_applied": n_verify_rules_applied,
        "n_verify_rules_applied_morph": n_verify_rules_applied_morph,
        "n_candidates_discarded_by_hard_fail": n_candidates_discarded_by_hard_fail,
        "score_deltas": score_deltas,
    }


def _apply_context_rules(
    raw_sentence: str,
    candidates: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    confirm_min_score: int,
    hold_min_score: int,
    context_window_chars: int,
) -> dict[str, Any]:
    n_context_rules_applied = 0
    n_context_pos_hits = 0
    n_context_neg_hits = 0
    score_deltas: list[int] = []
    triage_transition_counts = _init_triage_transition_counts()
    n_triage_changed_total = 0

    for candidate in candidates:
        prev_triage = candidate.get("triage")
        if candidate.get("hard_fail_triggered"):
            continue
        window = _build_candidate_window(raw_sentence, candidate, context_window_chars)
        for rule in rules:
            e_id = rule.get("e_id")
            if e_id and str(e_id) != candidate.get("e_id"):
                continue
            pattern = rule.get("pattern")
            if not pattern:
                continue
            try:
                regex = re.compile(pattern)
            except re.error:
                continue
            if not regex.search(window["text"]):
                continue
            n_context_rules_applied += 1
            rule_id = rule.get("rule_id")
            if rule_id:
                candidate["stage_hits"]["context"].append(rule_id)
            rule_type = str(rule.get("rule_type", "")).lower()
            if rule_type == "context_pos_regex":
                n_context_pos_hits += 1
            elif rule_type == "context_neg_regex":
                n_context_neg_hits += 1
            delta = int(rule.get("confidence_delta", 0) or 0)
            candidate["score"] += delta
            score_deltas.append(delta)

        score = candidate["score"]
        if score >= confirm_min_score:
            candidate["triage"] = "confirm"
        elif score >= hold_min_score:
            candidate["triage"] = "hold"
        else:
            candidate["triage"] = "discard"

        new_triage = candidate.get("triage")
        if prev_triage != new_triage:
            n_triage_changed_total += 1
            _bump_transition(triage_transition_counts, prev_triage, new_triage)

    return {
        "n_context_rules_applied": n_context_rules_applied,
        "n_context_pos_hits": n_context_pos_hits,
        "n_context_neg_hits": n_context_neg_hits,
        "score_deltas": score_deltas,
        "triage_transition_counts": triage_transition_counts,
        "n_triage_changed_total": n_triage_changed_total,
    }


def _summarize_deltas(deltas: list[int]) -> dict[str, float | None]:
    if not deltas:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(min(deltas)),
        "mean": float(sum(deltas)) / len(deltas),
        "max": float(max(deltas)),
    }


def _count_rules_by_stage(rules: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"detect": 0, "verify": 0, "context": 0}
    for row in rules:
        stage = str(row.get("stage", "")).lower()
        if stage in counts:
            counts[stage] += 1
    return counts


def _build_candidate_window(
    raw_sentence: str,
    candidate: dict[str, Any],
    context_window_chars: int | None = None,
    *,
    window_chars: int | None = None,
) -> dict[str, Any]:
    w = window_chars if window_chars is not None else int(context_window_chars or 0)
    half = w // 2
    segments = candidate.get("span_segments") or []
    if not segments:
        return {"start": 0, "end": len(raw_sentence), "text": raw_sentence}
    starts = [int(seg[0]) for seg in segments]
    ends = [int(seg[1]) for seg in segments]
    span_start = min(starts)
    span_end = max(ends)
    w0 = max(0, span_start - half)
    w1 = min(len(raw_sentence), span_end + half)
    return {"start": w0, "end": w1, "text": raw_sentence[w0:w1]}


def _build_match_window(
    match_span: tuple[int, int], text_len: int, window_chars: int
) -> tuple[int, int]:
    start, end = match_span
    w0 = max(0, start - window_chars)
    w1 = min(text_len, end + window_chars)
    return (w0, w1)


def _get_token_offsets(token: dict[str, Any]) -> tuple[int | None, int | None, bool]:
    start = token.get("start")
    end = token.get("end")
    used_fallback = False
    if start is None or end is None:
        start = token.get("char_start")
        end = token.get("char_end")
        used_fallback = True
    try:
        return int(start), int(end), used_fallback
    except (TypeError, ValueError):
        return None, None, used_fallback


def _filter_tokens_by_window(
    morph_tokens: list[dict[str, Any]], window_start: int, window_end: int
) -> list[dict[str, Any]]:
    filtered = []
    for token in morph_tokens:
        start, end, _used_fallback = _get_token_offsets(token)
        if start is None or end is None:
            continue
        if end > window_start and start < window_end:
            filtered.append(token)
    return filtered


def _evaluate_morph_rule(rule: dict[str, Any], morph_tokens: list[dict[str, Any]]) -> bool:
    if not morph_tokens:
        return False
    pattern = rule.get("pattern")
    if not pattern:
        return False
    if isinstance(pattern, str):
        try:
            rule_spec = json.loads(pattern)
        except json.JSONDecodeError:
            return False
    elif isinstance(pattern, dict):
        rule_spec = pattern
    else:
        return False

    rule_type = str(rule.get("rule_type", "")).lower()
    if rule_type == "pos_seq":
        must_contain = rule_spec.get("must_contain", []) or []
        must_not_contain = rule_spec.get("must_not_contain", []) or []
        return _check_morph_conditions(morph_tokens, must_contain, must_not_contain)
    if rule_type == "morph_check":
        require = rule_spec.get("require", []) or []
        forbid = rule_spec.get("forbid", []) or []
        return _check_morph_conditions(morph_tokens, require, forbid)
    return False


def _check_morph_conditions(
    morph_tokens: list[dict[str, Any]],
    must_contain: list[dict[str, Any]],
    must_not_contain: list[dict[str, Any]],
) -> bool:
    for cond in must_contain:
        if not any(_token_matches_condition(token, cond) for token in morph_tokens):
            return False
    for cond in must_not_contain:
        if any(_token_matches_condition(token, cond) for token in morph_tokens):
            return False
    return True


def _token_matches_condition(token: dict[str, Any], cond: dict[str, Any]) -> bool:
    matched_any = False
    for key in ("lemma", "surface", "pos", "pos_std"):
        if key not in cond:
            continue
        matched_any = True
        expected = cond.get(key)
        actual = token.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return matched_any


def _apply_guard_window(
    text: str, search_offset: int, detect_window: tuple[int, int] | None
) -> tuple[str, int]:
    if detect_window is None:
        return text, 0
    w0 = max(0, detect_window[0] - search_offset)
    w1 = min(len(text), detect_window[1] - search_offset)
    if w1 <= w0:
        return "", 0
    return text[w0:w1], w0


def _find_jong_bridge_match(
    text: str,
    direction: str,
    *,
    target_jong: int,
    tail_text: str,
) -> tuple[int, int] | None:
    matches: list[tuple[int, int]] = []
    for idx, ch in enumerate(text):
        code = ord(ch) - 0xAC00
        if code < 0 or code > 11171:
            continue
        jong = code % 28
        if jong != target_jong:
            continue
        j = idx + 1
        if j < len(text) and text[j] == " ":
            j += 1
        if text.startswith(tail_text, j):
            matches.append((idx, j + len(tail_text)))
    if not matches:
        return None
    if direction == "left_last":
        return matches[-1]
    return matches[0]


def _find_nde_candidate(
    text: str,
    offset: int,
    direction: str,
    *,
    has_yo_option: bool,
) -> tuple[tuple[int, int], dict[str, Any]] | None:
    pattern = re.compile(
        r"(?:"
        r"(?:은|는|ㄴ|ᆫ)\s*데(?:요)?"
        r"|"
        r"(?:은데|는데|인데|운데)(?:요)?"
        r")"
    )
    matches: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        end = match.end()
        text_match = match.group(0)
        if not has_yo_option and text_match.endswith("요"):
            end -= 1
            text_match = text_match[:-1]
        kind = "spaced" if any(ch.isspace() for ch in text_match) else "glued"
        matches.append(
            {
                "start": match.start() + offset,
                "end": end + offset,
                "length": end - match.start(),
                "kind": kind,
            }
        )
    if not matches:
        bridge = _find_jong_bridge_match(text, direction, target_jong=4, tail_text="데")
        if bridge:
            start, end = bridge
            if has_yo_option and text[end : end + 1] == "요":
                end += 1
            return (
                (start + offset, end + offset),
                {"special": "nde", "nde_match_kind": "bridge"},
            )
        return None
    if direction == "left_last":
        chosen = sorted(matches, key=lambda x: (x["start"], x["length"]))[-1]
    else:
        chosen = sorted(matches, key=lambda x: (x["start"], x["length"]))[0]
    return (
        (chosen["start"], chosen["end"]),
        {"special": "nde", "nde_match_kind": chosen["kind"]},
    )


def _collect_match_candidates(
    *,
    search_text: str,
    search_offset: int,
    pattern: re.Pattern[str],
    direction: str,
    options: list[str],
    has_adnominal: bool,
    nde_hint: bool,
    thing_hint: bool,
    morph_tokens: list[dict[str, Any]] | None,
    max_candidates_per_comp: int,
    debug: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in pattern.finditer(search_text):
        if len(candidates) >= max_candidates_per_comp:
            break
        start = match.start() + search_offset
        end = match.end() + search_offset
        candidates.append(
            {
                "abs_span": (start, end),
                "kind": "normal",
                "length": end - start,
                "match_info": None,
            }
        )
    if nde_hint:
        nde_match = _find_nde_candidate(
            search_text,
            search_offset,
            direction,
            has_yo_option=any(opt.endswith("요") for opt in options),
        )
        if nde_match and len(candidates) < max_candidates_per_comp:
            span, info = nde_match
            candidates.append(
                {
                    "abs_span": span,
                    "kind": "nde",
                    "length": span[1] - span[0],
                    "match_info": info,
                }
            )
    if has_adnominal and morph_tokens:
        allowed_surfaces: set[str] = set()
        for opt in options:
            allowed_surfaces.update(ADNORM_EQUIV.get(opt, {opt}))
        window_start = search_offset
        window_end = search_offset + len(search_text)
        for token in morph_tokens:
            if len(candidates) >= max_candidates_per_comp:
                break
            if token.get("pos") != "ETM":
                continue
            surf = token.get("surface")
            if surf not in allowed_surfaces:
                continue
            start, end, used_fallback = _get_token_offsets(token)
            if start is None or end is None or start < 0 or end < 0:
                if debug is not None:
                    debug["adnominal_skipped_missing_char_offsets"] = (
                        debug.get("adnominal_skipped_missing_char_offsets", 0) + 1
                    )
                continue
            if used_fallback and debug is not None:
                debug["adnominal_offset_fallback_used"] = (
                    debug.get("adnominal_offset_fallback_used", 0) + 1
                )
            vis_start, vis_end = start, end
            if surf in {"ᆫ", "ᆯ"}:
                vis_start = max(end - 1, 0)
                vis_end = end
            if vis_start < window_start or vis_end > window_end:
                if debug is not None:
                    debug["adnominal_skipped_out_of_bounds"] = (
                        debug.get("adnominal_skipped_out_of_bounds", 0) + 1
                    )
                continue
            if vis_end <= vis_start:
                continue
            candidates.append(
                {
                    "abs_span": (vis_start, vis_end),
                    "kind": "adnominal",
                    "length": vis_end - vis_start,
                    "match_info": {
                        "special": "adnominal",
                        "morph_span": (start, end),
                        "visible_span": (vis_start, vis_end),
                        "surface": surf,
                    },
                }
            )
    if thing_hint:
        separator_pattern = r"(?=($|\s|[은는이가을를도만까지조차마저]|[\\.,!?…]))"
        split_pattern = re.compile(rf"(거|게){separator_pattern}")
        fused_pattern = re.compile(r"(건|걸)(?=($|\s|[\\.,!?…]))")
        for match in split_pattern.finditer(search_text):
            if len(candidates) >= max_candidates_per_comp:
                break
            start = match.start() + search_offset
            end = start + 1
            candidates.append(
                {
                    "abs_span": (start, end),
                    "kind": "thing_bridge",
                    "length": end - start,
                    "match_info": {"special": "geot", "form": match.group(1), "fused": False},
                }
            )
        for match in fused_pattern.finditer(search_text):
            if len(candidates) >= max_candidates_per_comp:
                break
            start = match.start() + search_offset
            end = start + 1
            candidates.append(
                {
                    "abs_span": (start, end),
                    "kind": "thing_bridge",
                    "length": end - start,
                    "match_info": {"special": "geot", "form": match.group(1), "fused": True},
                }
            )
    return candidates


def _make_morph_snippet(
    morph_tokens: list[dict[str, Any]] | None,
    *,
    span_segments: list[list[int, int]] | None,
    window_chars: int = 20,
) -> tuple[list[dict[str, Any]], list[int] | None]:
    if not morph_tokens or not span_segments:
        return [], None
    starts = [int(seg[0]) for seg in span_segments]
    ends = [int(seg[1]) for seg in span_segments]
    span_start = min(starts)
    span_end = max(ends)
    w0 = max(0, span_start - window_chars)
    w1 = span_end + window_chars
    snippet: list[dict[str, Any]] = []
    for token in morph_tokens:
        start, end, _used_fallback = _get_token_offsets(token)
        if start is None or end is None:
            continue
        if end <= w0 or start >= w1:
            continue
        snippet.append(
            {
                "surface": token.get("surface"),
                "pos": token.get("pos"),
                "lemma": token.get("lemma"),
                "start": start,
                "end": end,
            }
        )
    return snippet, [w0, w1]


def _select_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    neighbor_span: tuple[int, int] | None,
    anchor_hint_span: tuple[int, int] | None,
    direction: str,
    min_gap_to_next: int | None,
    max_gap_to_next: int | None,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    gap_ok = []
    gap_fail = []
    for cand in candidates:
        if neighbor_span is None or (min_gap_to_next is None and max_gap_to_next is None):
            gap_ok.append(cand)
            continue
        if direction == "left_last":
            gap = neighbor_span[0] - cand["abs_span"][1]
        else:
            gap = cand["abs_span"][0] - neighbor_span[1]
        ok = True
        if min_gap_to_next is not None and gap < min_gap_to_next:
            ok = False
        if max_gap_to_next is not None and gap > max_gap_to_next:
            ok = False
        (gap_ok if ok else gap_fail).append(cand)
    if gap_ok:
        candidates = gap_ok

    def sort_key(cand: dict[str, Any]) -> tuple[int, int, int, int]:
        length = int(cand.get("length", 0))
        if anchor_hint_span is not None:
            hint_start, hint_end = anchor_hint_span
            cand_start, cand_end = cand["abs_span"]
            if cand_end < hint_start:
                distance = hint_start - cand_end
            elif hint_end < cand_start:
                distance = cand_start - hint_end
            else:
                distance = 0
        elif neighbor_span is None:
            distance = 0
        else:
            if direction == "left_last":
                distance = abs(neighbor_span[0] - cand["abs_span"][1])
            else:
                distance = abs(cand["abs_span"][0] - neighbor_span[1])
        kind_order = {"nde": 3, "adnominal": 2, "thing_bridge": 1, "normal": 0}
        kind_priority = kind_order.get(cand.get("kind", "normal"), 0)
        if direction == "left_last":
            pos_priority = cand["abs_span"][0]
        else:
            pos_priority = -cand["abs_span"][0]
        return (length, -distance, pos_priority, kind_priority)

    selected = sorted(candidates, key=sort_key)[-1]
    normal_exists = any(c.get("kind") == "normal" for c in candidates)
    if normal_exists and selected.get("kind") == "nde":
        selected["match_info"] = {
            **(selected.get("match_info") or {}),
            "choose_over_normal": "nde",
        }
    if normal_exists and selected.get("kind") == "adnominal":
        selected["match_info"] = {
            **(selected.get("match_info") or {}),
            "choose_over_normal": "adnominal",
        }
    return selected


def _find_adnominal_candidate(
    text: str, offset: int, direction: str, options: list[str]
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for opt in options:
        pattern = re.compile(re.escape(opt))
        for match in pattern.finditer(text):
            candidates.append((match.start() + offset, match.end() + offset))
    for idx, ch in enumerate(text):
        code = ord(ch) - 0xAC00
        if code < 0 or code > 11171:
            continue
        jong = code % 28
        if jong in (4, 8):
            candidates.append((idx + offset, idx + offset + 1))
    if not candidates:
        return None
    if direction == "left_last":
        return max(candidates, key=lambda x: x[0])
    return min(candidates, key=lambda x: x[0])


def _has_jong_bridge_option(row: dict[str, Any], *, target: str) -> bool:
    options = _parse_comp_options(str(row.get("comp_surf", "")))
    allowed = ADNORM_EQUIV.get(target, {target})
    return any(opt in allowed for opt in options)


def _get_single_comp_surface(row: dict[str, Any]) -> str | None:
    options = _parse_comp_options(str(row.get("comp_surf", "")))
    if len(options) == 1:
        return options[0]
    return None


def _get_comp_surface_options(row: dict[str, Any], *, limit: int = 4) -> list[str]:
    raw = str(row.get("comp_surf", "") or "")
    opts = _parse_comp_options(raw)
    out: list[str] = []
    for s in opts:
        s = str(s).strip()
        if s and s not in out:
            out.append(s)
    if "것" in out:
        for form in ["거", "게", "건", "걸"]:
            if form not in out:
                out.append(form)
    return out[:limit]


def _parse_int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _match_jong_bridge_with_next(
    text: str,
    *,
    direction: str,
    target_jong: int,
    next_surface: str,
    detect_window: tuple[int, int] | None,
    search_offset: int,
    allow_space: bool = True,
    next_span: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    window_text, window_offset = _apply_guard_window(text, search_offset, detect_window)
    matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if not next_surface:
        return None
    if next_span is not None:
        next_start_local = next_span[0] - search_offset - window_offset
        if next_start_local < 0:
            return None
        if next_start_local + len(next_surface) > len(window_text):
            return None
        if window_text[next_start_local : next_start_local + len(next_surface)] != next_surface:
            return None
        candidate_starts = [next_start_local]
    else:
        candidate_starts = [m.start() for m in re.finditer(re.escape(next_surface), window_text)]
    for next_start_local in candidate_starts:
        prev_idx = next_start_local - 1
        if prev_idx < 0:
            continue
        if allow_space and window_text[prev_idx] == " ":
            prev_idx -= 1
        if prev_idx < 0:
            continue
        ch = window_text[prev_idx]
        code = ord(ch) - 0xAC00
        if code < 0 or code > 11171:
            continue
        jong = code % 28
        if jong != target_jong:
            continue
        cur_start = search_offset + window_offset + prev_idx
        cur_end = cur_start + 1
        next_start = search_offset + window_offset + next_start_local
        next_end = next_start + len(next_surface)
        matches.append(((cur_start, cur_end), (next_start, next_end)))
    if not matches:
        return None
    if direction == "left_last":
        return matches[-1]
    return matches[0]


def _update_special_match_meta(
    meta: dict[str, Any],
    row: dict[str, Any],
    match_info: dict[str, Any] | None,
    match_span: tuple[int, int] | None = None,
) -> None:
    if not match_info:
        return
    special = match_info.get("special")
    if special == "adnominal":
        meta["special_adnominal"] += 1
        if not meta.get("bridge_detail"):
            meta["bridge_detail"] = {
                "applied": True,
                "bridge_type": "surface_equiv",
                "comp_id": str(row.get("comp_id", "")),
                "from_surface": match_info.get("surface"),
                "to_surface": match_info.get("surface"),
                "token_surface": match_info.get("surface"),
                "token_pos": "ETM",
                "token_span": match_info.get("visible_span") or match_info.get("morph_span"),
                "next_comp_id": None,
                "next_span": None,
                "reason": "ETM_equivalence_or_jong_bridge",
            }
        if match_info.get("choose_over_normal") == "adnominal":
            meta["choose_adnominal_over_normal"] += 1
            comp_id = str(row.get("comp_id", ""))
            comp_surf = str(row.get("comp_surf", ""))
            meta["component_match_notes"].append(
                f"{comp_id}({comp_surf}): choose=adnominal_over_normal gap_ok"
            )
        return
    if special == "nde":
        meta["special_nde"] += 1
        match_kind = match_info.get("nde_match_kind")
        if match_info.get("choose_over_normal") == "nde":
            meta["choose_nde_over_normal"] += 1
        if match_kind:
            comp_id = str(row.get("comp_id", ""))
            comp_surf = str(row.get("comp_surf", ""))
            meta["component_match_notes"].append(
                f"{comp_id}({comp_surf}): nde_match={match_kind}"
            )
        if match_info.get("choose_over_normal") == "nde":
            comp_id = str(row.get("comp_id", ""))
            comp_surf = str(row.get("comp_surf", ""))
            meta["component_match_notes"].append(
                f"{comp_id}({comp_surf}): choose=nde_over_normal longer_match"
            )
        return
    if special == "geot":
        # thing_bridge: '것' required 컴포넌트가 '거/게/건/걸'로 축약/융합된 경우를 브릿지 매칭으로 복구
        meta["thing_bridge"] += 1
        if not meta.get("thing_bridge_detail"):
            meta["thing_bridge_detail"] = {
                "applied": True,
                "form": match_info.get("form"),
                "span": list(match_span) if match_span else None,
                "fused": bool(match_info.get("fused")),
                "reason": "것 component matched via thing_bridge",
            }
        form = match_info.get("form")
        if isinstance(form, str) and form in meta["thing_bridge_form_counts"]:
            meta["thing_bridge_form_counts"][form] += 1
        if match_info.get("fused"):
            meta["thing_bridge_fused"] += 1
        comp_id = str(row.get("comp_id", ""))
        comp_surf = str(row.get("comp_surf", ""))
        suffix = "_fused" if match_info.get("fused") else ""
        if form:
            meta["component_match_notes"].append(
                f"{comp_id}({comp_surf}): thing_bridge{suffix}->{form}"
            )


def _apply_span_competition_guard(
    candidates: list[dict[str, Any]],
    triage_transition_counts: dict[str, int],
) -> dict[str, Any]:
    groups: dict[tuple[tuple[int, int], ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        segments = candidate.get("span_segments") or []
        key = tuple(tuple(map(int, seg)) for seg in segments)
        groups.setdefault(key, []).append(candidate)

    n_span_competition_groups = 0
    n_candidates_downgraded_by_competition = 0
    n_triage_changed_total = 0
    for group_candidates in groups.values():
        if len(group_candidates) < 2:
            continue
        e_ids = {str(row.get("e_id")) for row in group_candidates if row.get("e_id") is not None}
        if len(e_ids) < 2:
            continue
        n_span_competition_groups += 1
        for candidate in group_candidates:
            if candidate.get("triage") != "confirm":
                continue
            candidate["triage"] = "hold"
            n_candidates_downgraded_by_competition += 1
            n_triage_changed_total += 1
            _bump_transition(triage_transition_counts, "confirm", "hold")

    return {
        "n_span_competition_groups": n_span_competition_groups,
        "n_candidates_downgraded_by_competition": n_candidates_downgraded_by_competition,
        "n_triage_changed_total": n_triage_changed_total,
    }


def _init_triage_transition_counts() -> dict[str, int]:
    return {
        "confirm->hold": 0,
        "confirm->discard": 0,
        "hold->confirm": 0,
        "hold->discard": 0,
        "discard->confirm": 0,
        "discard->hold": 0,
    }


def _bump_transition(counts: dict[str, int], prev: Any, new: Any) -> None:
    if prev is None or new is None:
        return
    key = f"{prev}->{new}"
    if key in counts:
        counts[key] += 1


def _merge_transition_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _noop_issue(*_args: Any, **_kwargs: Any) -> None:
    return None


def _index_components_by_eid(components_rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in components_rows:
        e_id = row.get("e_id")
        if not e_id:
            continue
        grouped.setdefault(str(e_id), []).append(row)
    return grouped


def _build_component_regex(comp_surf: str, spacing_policy: str, mode: str = "MVP") -> re.Pattern[str]:
    del spacing_policy, mode
    options = _parse_comp_options(comp_surf)
    if not options:
        options = [comp_surf]
    escaped = [re.escape(opt) for opt in options]
    if len(escaped) == 1:
        pattern = escaped[0]
    else:
        pattern = "(?:" + "|".join(escaped) + ")"
    return re.compile(pattern)


def _parse_comp_options(comp_surf: str) -> list[str]:
    s = (comp_surf or "").strip()
    if not s:
        return []

    def _clean(raw: str) -> str:
        cleaned = raw.strip().rstrip("?")
        if not cleaned:
            return ""
        m = re.search(r"\(([^)]{1,16})\)\s*$", cleaned)
        if m and re.fullmatch(r"[A-Za-z0-9_]+", m.group(1) or ""):
            cleaned = cleaned[: m.start()].rstrip()
        return cleaned

    if "/" in s:
        return [x for x in (_clean(part) for part in s.split("/")) if x]

    # Dict SSOT 무수정 보호: "것|거" 같은 단순 OR만 옵션 분리.
    if "|" in s:
        regexy_tokens = [
            "re:",
            "(?:",
            "[",
            "]",
            "+",
            "*",
            "?",
            "{",
            "}",
            "\\",
            "^",
            "$",
            ".",
        ]
        if not any(tok in s for tok in regexy_tokens):
            return [x for x in (_clean(part) for part in s.split("|")) if x]
        warnings.warn(
            f"[build_silver] regex-like comp_surf with '|': {s!r} (kept as literal; suggested_fix=use '/' for options)",
            RuntimeWarning,
            stacklevel=2,
        )

    single = _clean(s)
    return [single] if single else []


def _find_best_match(
    comp_surf: str,
    text: str,
    direction: str,
    spacing_policy: str,
    *,
    detect_window: tuple[int, int] | None = None,
    search_offset: int = 0,
    thing_hint: bool = False,
    morph_tokens: list[dict[str, Any]] | None = None,
    neighbor_span: tuple[int, int] | None = None,
    min_gap_to_next: int | None = None,
    max_gap_to_next: int | None = None,
    anchor_hint_span: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, dict[str, Any] | None, str | None, dict[str, Any]]:
    debug: dict[str, Any] = {
        "search_start": int(search_offset),
        "search_end": int(search_offset + len(text)),
        "neighbor_span": neighbor_span,
        "gap_min": min_gap_to_next,
        "gap_max": max_gap_to_next,
        "n_candidates_total": 0,
        "n_candidates_gap_ok": 0,
        "n_candidates_in_bounds": 0,
        "failure_reason": None,
        "candidates_top": [],
        "gap_violations": [],
        "special_candidate_counts": {"adnominal": 0, "nde": 0, "thing_bridge": 0},
        "adnominal_skipped_missing_char_offsets": 0,
        "adnominal_skipped_out_of_bounds": 0,
        "adnominal_offset_fallback_used": 0,
    }
    options = _parse_comp_options(comp_surf)
    opt_set = set(options)
    has_adnominal = any(opt in {"ㄴ", "ᆫ", "ㄹ", "ᆯ"} for opt in opt_set)
    nde_hint = any("는데" in opt for opt in opt_set) or (
        any(opt.endswith("데") for opt in opt_set)
        and any(opt in {"은", "는", "ㄴ"} for opt in opt_set)
    )
    pattern = _build_component_regex(comp_surf, spacing_policy)
    if detect_window is None:
        search_text = text
        search_offset_local = 0
    else:
        search_text, search_offset_local = _apply_guard_window(text, search_offset, detect_window)

    candidates = _collect_match_candidates(
        search_text=search_text,
        search_offset=search_offset_local,
        pattern=pattern,
        direction=direction,
        options=options,
        has_adnominal=has_adnominal,
        nde_hint=nde_hint,
        thing_hint=thing_hint,
        morph_tokens=morph_tokens,
        max_candidates_per_comp=50,
        debug=debug,
    )
    if not candidates:
        debug["failure_reason"] = "no_candidates"
        return None, None, None, debug
    neighbor_span_local = None
    if neighbor_span is not None:
        neighbor_span_local = (neighbor_span[0] - search_offset, neighbor_span[1] - search_offset)
    anchor_hint_local = None
    if anchor_hint_span is not None:
        anchor_hint_local = (anchor_hint_span[0] - search_offset, anchor_hint_span[1] - search_offset)
    neighbor_span_local = None
    if neighbor_span is not None:
        neighbor_span_local = (neighbor_span[0] - search_offset, neighbor_span[1] - search_offset)
    anchor_hint_local = None
    if anchor_hint_span is not None:
        anchor_hint_local = (anchor_hint_span[0] - search_offset, anchor_hint_span[1] - search_offset)

    gap_ok_candidates = []
    for cand in candidates:
        gap_ok = True
        gap = None
        if neighbor_span_local is not None and (min_gap_to_next is not None or max_gap_to_next is not None):
            if direction == "left_last":
                gap = neighbor_span_local[0] - cand["abs_span"][1]
            else:
                gap = cand["abs_span"][0] - neighbor_span_local[1]
            if min_gap_to_next is not None and gap < min_gap_to_next:
                gap_ok = False
            if max_gap_to_next is not None and gap > max_gap_to_next:
                gap_ok = False
        if gap_ok:
            gap_ok_candidates.append(cand)
        else:
            debug["gap_violations"].append(
                {
                    "cand_span": cand["abs_span"],
                    "neighbor_span": neighbor_span_local,
                    "gap": gap,
                    "min_gap": min_gap_to_next,
                    "max_gap": max_gap_to_next,
                    "reason": "gap_out_of_bounds",
                }
            )

    debug["n_candidates_total"] = len(candidates)
    debug["n_candidates_gap_ok"] = len(gap_ok_candidates)
    debug["n_candidates_in_bounds"] = len(candidates)
    for cand in candidates:
        kind = str(cand.get("kind", "normal"))
        if kind in debug["special_candidate_counts"]:
            debug["special_candidate_counts"][kind] += 1
    if not gap_ok_candidates and neighbor_span_local is not None:
        debug["failure_reason"] = "all_gap_failed"

    def _cand_debug(cand: dict[str, Any]) -> dict[str, Any]:
        gap = None
        if neighbor_span_local is not None:
            if direction == "left_last":
                gap = neighbor_span_local[0] - cand["abs_span"][1]
            else:
                gap = cand["abs_span"][0] - neighbor_span_local[1]
        return {
            "kind": cand.get("kind", "normal"),
            "span": cand.get("abs_span"),
            "span_abs": (cand["abs_span"][0] + search_offset, cand["abs_span"][1] + search_offset),
            "gap_to_neighbor": gap,
            "gap_ok": cand in gap_ok_candidates,
        }

    debug["candidates_top"] = [_cand_debug(c) for c in candidates[:5]]

    selected = _select_best_candidate(
        candidates,
        neighbor_span=neighbor_span_local,
        anchor_hint_span=anchor_hint_local,
        direction=direction,
        min_gap_to_next=min_gap_to_next,
        max_gap_to_next=max_gap_to_next,
    )
    if selected is None:
        debug["failure_reason"] = debug.get("failure_reason") or "no_selection"
        return None, None, None, debug
    selected_kind = selected.get("kind", "normal")
    return selected["abs_span"], selected.get("match_info"), selected_kind, debug


def _match_component(
    comp_row: dict[str, Any],
    search_text: str,
    direction: str,
    spacing_policy: str,
    *,
    detect_window: tuple[int, int] | None = None,
    search_offset: int = 0,
    morph_tokens: list[dict[str, Any]] | None = None,
    neighbor_span: tuple[int, int] | None = None,
    anchor_hint_span: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, dict[str, Any] | None, str | None, dict[str, Any]]:
    comp_surf = str(comp_row.get("comp_surf", ""))
    options = _parse_comp_options(comp_surf)
    opt_set = set(options)
    has_adnominal = any(opt in {"ㄴ", "ᆫ", "ㄹ", "ᆯ"} for opt in opt_set)
    nde_hint = any("는데" in opt for opt in opt_set) or (
        any(opt.endswith("데") for opt in opt_set)
        and any(opt in {"은", "는", "ㄴ"} for opt in opt_set)
    )
    is_required = str(comp_row.get("is_required", "y")).lower() == "y"
    thing_hint = "것" in opt_set and is_required
    min_gap_to_next = _parse_int_value(comp_row.get("min_gap_to_next")) if comp_row else None
    max_gap_to_next = _parse_int_value(comp_row.get("max_gap_to_next")) if comp_row else None
    match, match_info, match_kind, debug = _find_best_match(
        comp_surf,
        search_text,
        direction,
        spacing_policy,
        detect_window=detect_window,
        search_offset=search_offset,
        thing_hint=thing_hint,
        morph_tokens=morph_tokens,
        neighbor_span=neighbor_span,
        min_gap_to_next=min_gap_to_next,
        max_gap_to_next=max_gap_to_next,
        anchor_hint_span=anchor_hint_span,
    )
    debug["comp_id"] = str(comp_row.get("comp_id", ""))
    debug["is_required"] = is_required
    if match is None:
        return None, None, match_kind, debug
    if match_info is None:
        return match, None, match_kind, debug
    special = match_info.get("special")
    if special == "nde" and nde_hint:
        return match, match_info, match_kind, debug
    if special == "adnominal":
        return match, match_info, match_kind, debug
    if special == "geot" and thing_hint:
        return match, match_info, match_kind, debug
    return match, None, match_kind, debug


def _maybe_override_with_bridge_nearest_left_last(
    row: dict[str, Any],
    match_span: tuple[int, int] | None,
    left_neighbor: tuple[int, int],
    left_neighbor_row: dict[str, Any] | None,
    raw_sentence: str,
    detect_window: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    if match_span is None or left_neighbor_row is None:
        return None
    next_surf = _get_single_comp_surface(left_neighbor_row)
    if not next_surf:
        return None
    bridge_text = raw_sentence[: left_neighbor[1]]
    bridge = None
    if _has_jong_bridge_option(row, target="ㄹ"):
        bridge = _match_jong_bridge_with_next(
            bridge_text,
            direction="left_last",
            target_jong=8,
            next_surface=next_surf,
            detect_window=detect_window,
            search_offset=0,
            allow_space=True,
            next_span=left_neighbor,
        )
    if bridge is None and _has_jong_bridge_option(row, target="ㄴ"):
        bridge = _match_jong_bridge_with_next(
            bridge_text,
            direction="left_last",
            target_jong=4,
            next_surface=next_surf,
            detect_window=detect_window,
            search_offset=0,
            allow_space=True,
            next_span=left_neighbor,
        )
    if not bridge:
        return None
    cur_span, next_span = bridge
    dist_normal = abs(left_neighbor[0] - match_span[1])
    dist_bridge = abs(left_neighbor[0] - cur_span[1])
    return cur_span, next_span, dist_normal, dist_bridge


def _locate_components_spans(
    raw_sentence: str,
    e_id: str,
    components: list[dict[str, Any]],
    *,
    anchor_strategy: str = "gat_or_best",
    spacing_policy: str,
    disconti_allowed: bool,
    expredict_row: dict[str, Any] | None = None,
    detect_window: tuple[int, int] | None = None,
    morph_tokens: list[dict[str, Any]] | None = None,
    anchor_hint_span: tuple[int, int] | None = None,
    debug_ctx: dict[str, Any] | None = None,
) -> tuple[list[list[list[int, int]]], dict[str, Any]]:
    del anchor_strategy
    disconti_allowed_raw = expredict_row.get("disconti_allowed", None) if expredict_row else None
    if expredict_row and "disconti_allowed" in expredict_row:
        disconti_allowed_eval = str(disconti_allowed_raw or "").strip().lower() == "y"
        disconti_allowed = disconti_allowed_eval
    meta = {
        "fl_treated_as_fx": 0,
        "optional_ignored": 0,
        "special_adnominal": 0,
        "special_nde": 0,
        "thing_bridge": 0,
        "thing_bridge_fused": 0,
        "thing_bridge_form_counts": {"거": 0, "게": 0, "건": 0, "걸": 0},
        "choose_nde_over_normal": 0,
        "choose_adnominal_over_normal": 0,
        "order_bounded": 0,
        "span_fail_required": 0,
        "anchor_selected_span": None,
        "anchor_selected_kind": None,
        "failed_required_comp_ids": [],
        "failed_optional_comp_ids": [],
        "per_comp_debug": {},
        "gap_violations": [],
        "search_ranges": {},
        "special_candidate_counts": {"adnominal": 0, "nde": 0, "thing_bridge": 0},
        "morph_token_snippet": [],
        "component_match_notes": [],
        "bridge_detail": None,
        "thing_bridge_detail": None,
    }
    required = []
    optional_count = 0
    for row in components:
        is_required = str(row.get("is_required", "y")).lower() == "y"
        if is_required:
            required.append(row)
        else:
            optional_count += 1
    meta["optional_ignored"] = optional_count
    if expredict_row:
        e_comp_value = expredict_row.get("e_comp_id")
        e_comp_raw = str(e_comp_value).strip() if e_comp_value is not None else ""
        if e_comp_raw:
            expected = [part for part in e_comp_raw.split(";") if part.strip()]
            if len(expected) == 1 and len(components) > 1:
                return [], meta
    if not required:
        return [], meta
    for row in required:
        order_policy = str(row.get("order_policy", "fx")).lower()
        if order_policy == "fl":
            meta["fl_treated_as_fx"] += 1
    required_sorted = sorted(
        required,
        key=lambda r: (int(r.get("comp_order", 0) or 0), str(r.get("comp_id", ""))),
    )
    anchor_candidates = [row for row in required_sorted if int(row.get("anchor_rank", 0) or 0) > 0]
    if anchor_candidates:
        anchor = sorted(
            anchor_candidates,
            key=lambda r: (int(r.get("anchor_rank", 0) or 0), -len(str(r.get("comp_surf", "")))),
        )[0]
    else:
        anchor = sorted(
            required_sorted, key=lambda r: -len(str(r.get("comp_surf", "")))
        )[0]
    anchor_span, special, anchor_kind, debug = _match_component(
        anchor,
        raw_sentence,
        "right_first",
        spacing_policy,
        detect_window=detect_window,
        search_offset=0,
        morph_tokens=morph_tokens,
        anchor_hint_span=anchor_hint_span,
    )
    meta["per_comp_debug"][str(anchor.get("comp_id", ""))] = debug
    meta["gap_violations"].extend(debug.get("gap_violations", []))
    meta["search_ranges"][str(anchor.get("comp_id", ""))] = {
        "search_start": debug.get("search_start"),
        "search_end": debug.get("search_end"),
        "neighbor_span": debug.get("neighbor_span"),
    }
    for k in meta["special_candidate_counts"]:
        meta["special_candidate_counts"][k] += int(debug.get("special_candidate_counts", {}).get(k, 0) or 0)
    _update_special_match_meta(meta, anchor, special, match_span=anchor_span)
    if not anchor_span:
        meta["failed_required_comp_ids"].append(str(anchor.get("comp_id", "")))
        return [], meta
    meta["anchor_selected_span"] = [int(anchor_span[0]), int(anchor_span[1])]
    meta["anchor_selected_kind"] = anchor_kind or "normal"
    spans: dict[str, tuple[int, int]] = {}
    spans[str(anchor.get("comp_id", ""))] = anchor_span
    anchor_index = required_sorted.index(anchor)
    left = list(reversed(required_sorted[:anchor_index]))
    right = required_sorted[anchor_index + 1 :]
    right_bound = anchor_span[0]
    left_neighbor = anchor_span
    left_neighbor_row = anchor
    for row in left:
        meta["order_bounded"] += 1
        search_text = raw_sentence[:right_bound]
        match_span, special, _, debug = _match_component(
            row,
            search_text,
            "left_last",
            spacing_policy,
            detect_window=detect_window,
            search_offset=0,
            morph_tokens=morph_tokens,
            neighbor_span=left_neighbor,
        )
        meta["per_comp_debug"][str(row.get("comp_id", ""))] = debug
        meta["gap_violations"].extend(debug.get("gap_violations", []))
        meta["search_ranges"][str(row.get("comp_id", ""))] = {
            "search_start": debug.get("search_start"),
            "search_end": debug.get("search_end"),
            "neighbor_span": debug.get("neighbor_span"),
        }
        for k in meta["special_candidate_counts"]:
            meta["special_candidate_counts"][k] += int(
                debug.get("special_candidate_counts", {}).get(k, 0) or 0
            )
        if debug.get("failure_reason") == "all_gap_failed":
            match_span = None
            special = None
        _update_special_match_meta(meta, row, special, match_span=match_span)
        bridge_override = _maybe_override_with_bridge_nearest_left_last(
            row,
            match_span,
            left_neighbor,
            left_neighbor_row,
            raw_sentence,
            detect_window,
        )
        if bridge_override is not None and match_span is not None:
            cur_span, next_span, dist_normal, dist_bridge = bridge_override
            comp_id = str(row.get("comp_id", ""))
            selected = "normal"
            if dist_bridge < dist_normal:
                selected = "bridge"
                match_span = cur_span
                spans[comp_id] = cur_span
                spans[str(left_neighbor_row.get("comp_id", ""))] = next_span
            debug["tie_break"] = {
                "applied": True,
                "policy": "nearest_to_neighbor_with_bridge",
                "dist_normal": dist_normal,
                "dist_bridge": dist_bridge,
                "selected": selected,
            }
        if (
            left_neighbor_row is not None
            and debug.get("failure_reason") in {"no_candidates", "all_gap_failed"}
        ):
            observed_next = raw_sentence[left_neighbor[0] : left_neighbor[1]] if left_neighbor else ""
            observed_next = observed_next.strip()
            base_next_surfs = _get_comp_surface_options(left_neighbor_row, limit=4)
            next_surfs: list[str] = []
            if observed_next:
                next_surfs.append(observed_next)
            for s in base_next_surfs:
                if s and s not in next_surfs:
                    next_surfs.append(s)
            next_surfs = next_surfs[:4]
            if next_surfs:
                best = None  # (dist_bridge, cur_span, next_span, next_surf, target)
                bridge_text = raw_sentence[: left_neighbor[1]]
                for next_surf in next_surfs:
                    if _has_jong_bridge_option(row, target="ㄹ"):
                        bridge_r = _match_jong_bridge_with_next(
                            bridge_text,
                            direction="left_last",
                            target_jong=8,
                            next_surface=next_surf,
                            detect_window=detect_window,
                            search_offset=0,
                            allow_space=True,
                            next_span=left_neighbor,
                        )
                        if bridge_r:
                            cur_span, next_span = bridge_r
                            dist = abs(left_neighbor[0] - cur_span[1])
                            cand = (dist, cur_span, next_span, next_surf, "ㄹ")
                            if best is None or cand[0] < best[0]:
                                best = cand
                    if _has_jong_bridge_option(row, target="ㄴ"):
                        bridge_n = _match_jong_bridge_with_next(
                            bridge_text,
                            direction="left_last",
                            target_jong=4,
                            next_surface=next_surf,
                            detect_window=detect_window,
                            search_offset=0,
                            allow_space=True,
                            next_span=left_neighbor,
                        )
                        if bridge_n:
                            cur_span, next_span = bridge_n
                            dist = abs(left_neighbor[0] - cur_span[1])
                            cand = (dist, cur_span, next_span, next_surf, "ㄴ")
                            if best is None or cand[0] < best[0]:
                                best = cand
                if best:
                    _, cur_span, next_span, chosen_next_surf, chosen_target = best
                    spans[str(row.get("comp_id", ""))] = cur_span
                    spans[str(left_neighbor_row.get("comp_id", ""))] = next_span
                    if meta.get("bridge_detail") is None:
                        meta["bridge_detail"] = {
                            "applied": True,
                            "bridge_type": "left_bridge_multi_next",
                            "comp_id": str(row.get("comp_id", "")),
                            "from_surface": f"jong:{chosen_target}",
                            "to_surface": chosen_next_surf,
                            "token_surface": None,
                            "token_pos": None,
                            "token_span": list(cur_span),
                            "next_comp_id": str(left_neighbor_row.get("comp_id", "")),
                            "next_span": list(next_span),
                            "reason": "jong_bridge_multi_next_surface",
                        }
                    debug["bridge_multi_next"] = {
                        "applied": True,
                        "observed_next": observed_next,
                        "next_surfs": next_surfs,
                        "chosen_next_surf": chosen_next_surf,
                        "chosen_target": chosen_target,
                    }
                    right_bound = cur_span[0]
                    left_neighbor = cur_span
                    left_neighbor_row = row
                    continue
        if not match_span:
            meta["span_fail_required"] += 1
            meta["failed_required_comp_ids"].append(str(row.get("comp_id", "")))
            return [], meta
        spans[str(row.get("comp_id", ""))] = match_span
        right_bound = match_span[0]
        left_neighbor = match_span
        left_neighbor_row = row
    left_bound = anchor_span[1]
    right_neighbor = anchor_span
    idx = 0
    while idx < len(right):
        row = right[idx]
        meta["order_bounded"] += 1
        search_text = raw_sentence[left_bound:]
        next_row = right[idx + 1] if idx + 1 < len(right) else None
        match_span, special, _, debug = _match_component(
            row,
            search_text,
            "right_first",
            spacing_policy,
            detect_window=detect_window,
            search_offset=left_bound,
            morph_tokens=morph_tokens,
            neighbor_span=right_neighbor,
        )
        meta["per_comp_debug"][str(row.get("comp_id", ""))] = debug
        meta["gap_violations"].extend(debug.get("gap_violations", []))
        meta["search_ranges"][str(row.get("comp_id", ""))] = {
            "search_start": debug.get("search_start"),
            "search_end": debug.get("search_end"),
            "neighbor_span": debug.get("neighbor_span"),
        }
        for k in meta["special_candidate_counts"]:
            meta["special_candidate_counts"][k] += int(
                debug.get("special_candidate_counts", {}).get(k, 0) or 0
            )
        if debug.get("failure_reason") == "all_gap_failed":
            match_span = None
            special = None
        _update_special_match_meta(meta, row, special, match_span=match_span)
        if (
            next_row
            and debug.get("failure_reason") in {"no_candidates", "all_gap_failed"}
        ):
            next_surf = _get_single_comp_surface(next_row)
            if next_surf:
                bridge = None
                if _has_jong_bridge_option(row, target="ㄹ"):
                    bridge = _match_jong_bridge_with_next(
                        search_text,
                        direction="right_first",
                        target_jong=8,
                        next_surface=next_surf,
                        detect_window=detect_window,
                        search_offset=left_bound,
                        allow_space=True,
                    )
                if bridge is None and _has_jong_bridge_option(row, target="ㄴ"):
                    bridge = _match_jong_bridge_with_next(
                        search_text,
                        direction="right_first",
                        target_jong=4,
                        next_surface=next_surf,
                        detect_window=detect_window,
                        search_offset=left_bound,
                        allow_space=True,
                    )
                if bridge:
                    cur_span, next_span = bridge
                    spans[str(row.get("comp_id", ""))] = cur_span
                    spans[str(next_row.get("comp_id", ""))] = next_span
                    if meta.get("bridge_detail") is None:
                        meta["bridge_detail"] = {
                            "applied": True,
                            "bridge_type": "jong_bridge",
                            "comp_id": str(row.get("comp_id", "")),
                            "from_surface": "jong",
                            "to_surface": next_surf,
                            "token_surface": None,
                            "token_pos": None,
                            "token_span": list(cur_span),
                            "next_comp_id": str(next_row.get("comp_id", "")),
                            "next_span": list(next_span),
                            "reason": "jong_bridge",
                        }
                    left_bound = next_span[1]
                    right_neighbor = next_span
                    idx += 2
                    continue
        if not match_span:
            meta["span_fail_required"] += 1
            meta["failed_required_comp_ids"].append(str(row.get("comp_id", "")))
            return [], meta
        start = left_bound + match_span[0]
        end = left_bound + match_span[1]
        spans[str(row.get("comp_id", ""))] = (start, end)
        left_bound = end
        right_neighbor = (start, end)
        idx += 1
    ordered_spans = []
    for row in required_sorted:
        comp_id = str(row.get("comp_id", ""))
        if comp_id not in spans:
            meta["failed_required_comp_ids"].append(comp_id)
            return [], meta
        ordered_spans.append((row, spans[comp_id]))
    for idx, (row, span) in enumerate(ordered_spans[:-1]):
        max_gap = row.get("max_gap_to_next")
        if max_gap is None or max_gap == "":
            continue
        try:
            max_gap_val = int(max_gap)
        except (TypeError, ValueError):
            continue
        next_span = ordered_spans[idx + 1][1]
        if next_span[0] - span[1] > max_gap_val:
            return [], meta
    comp_spans = [span for _, span in ordered_spans]
    segments = _segments_from_component_spans(raw_sentence, comp_spans)
    if debug_ctx:
        debug_example = os.getenv("KMWE_DEBUG_EXAMPLE", "")
        debug_eid = os.getenv("KMWE_DEBUG_EID")
        if debug_example:
            parts = debug_example.split("#")
            want_example = parts[0]
            want_inst = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            if (
                debug_ctx.get("example_id") == want_example
                and debug_ctx.get("instance_id") == want_inst
                and debug_ctx.get("e_id") == debug_eid
            ):
                keys = list(expredict_row.keys()) if isinstance(expredict_row, dict) else []
                print(
                    "[DBG:C4] ex=%s#%s e_id=%s disconti_allowed_raw=%r disconti_allowed_eval=%s "
                    "segments=%s n_segments=%d has_key=%s"
                    % (
                        debug_ctx.get("example_id"),
                        debug_ctx.get("instance_id"),
                        debug_ctx.get("e_id"),
                        disconti_allowed_raw,
                        disconti_allowed_eval,
                        segments,
                        len(segments),
                        ("disconti_allowed" in keys),
                    )
                )
    if disconti_allowed:
        if not segments:
            return [], meta
        return [segments], meta
    if len(segments) > 1:
        return [], meta
    if morph_tokens and detect_window:
        w0, w1 = detect_window
        snippet = []
        for token in morph_tokens:
            if len(snippet) >= 40:
                break
            start, end, _used_fallback = _get_token_offsets(token)
            if start is not None and end is not None and w0 <= start <= w1:
                snippet.append(
                    {
                        "surface": token.get("surface"),
                        "pos": token.get("pos"),
                        "lemma": token.get("lemma"),
                        "char_start": start,
                        "char_end": end,
                    }
                )
        meta["morph_token_snippet"] = snippet
    return [segments], meta


def _create_candidate_from_span_segments(
    *, e_id: str, span_segments: list[list[int, int]], raw_sentence: str
) -> dict[str, Any]:
    return {
        "e_id": str(e_id),
        "span_segments": [[int(s), int(e)] for s, e in span_segments],
        "span_key": _span_key_from_segments(span_segments),
        "span_text": _span_text_from_segments(raw_sentence, span_segments),
        "score": 0,
        "stage_hits": {"detect": [], "verify": [], "context": []},
        "hard_fail_triggered": False,
        "hard_fail_reasons": [],
        "rule_engine": "re",
    }


def _segments_from_component_spans(
    raw_sentence: str, comp_spans: list[tuple[int, int]]
) -> list[list[int, int]]:
    if not comp_spans:
        return []
    spans = sorted(comp_spans, key=lambda x: x[0])
    segments: list[list[int, int]] = []
    cur_s, cur_e = spans[0]
    for start, end in spans[1:]:
        gap = raw_sentence[cur_e:start]
        if gap.strip() == "":
            cur_e = max(cur_e, end)
        else:
            segments.append([int(cur_s), int(cur_e)])
            cur_s, cur_e = start, end
    segments.append([int(cur_s), int(cur_e)])
    return segments


def _span_key_from_segments(span_segments: list[list[int, int]]) -> str:
    return "|".join(f"{int(s)}:{int(e)}" for s, e in span_segments)


def _span_text_from_segments(
    raw_sentence: str, span_segments: list[list[int, int]], gap_marker: str = " … "
) -> str:
    if not span_segments:
        return ""
    if len(span_segments) == 1:
        s, e = span_segments[0]
        return raw_sentence[int(s) : int(e)]
    return gap_marker.join(raw_sentence[int(s) : int(e)] for s, e in span_segments)
