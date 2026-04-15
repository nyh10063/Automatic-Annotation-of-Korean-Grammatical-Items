#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kmwe.data.rule_eval import RuleEvalConfig, RuleEvalInstance
from kmwe.stages import build_silver as silver_loader
from kmwe.stages.build_bgroup_sft import _build_prompt_core
from kmwe.stages.eval_rule_gold import _detect_candidates_for_instance, _prepare_runtime
from kmwe.utils.morph import analyze_with_kiwi


B_DEBUG_EIDS = {"ece002", "ece003", "edf004", "edf005", "ept001", "ept002", "ept003"}


REVIEWER_LABELS = {
    "ece001": "다면(가정/조건 제시)",
    "edf003": "ㄴ/은 적 있/없(경험 유무 서술)",
    "ece002": "ㄴ/은/는데1(상황/배경 제시)",
    "ece003": "ㄴ/은/는데2(대립/대조)",
    "edf004": "고 말1(안타까움)",
    "edf005": "고 말2(의지)",
    "ept001": "까지1(범위의 끝)",
    "ept002": "까지2(더함)",
    "ept003": "까지3(지나침)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public B-pipeline LLM runner")
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--tokenizer_dir", required=True, type=Path)
    parser.add_argument("--dict_xlsx", required=True, type=Path)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--verify_window_chars", type=int, default=20)
    return parser.parse_args()


def _read_input_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "sentence" not in fieldnames:
            raise ValueError(f"input csv must contain a 'sentence' column: {path}")
        for row in reader:
            sentence_parts: list[str] = []
            first_cell = str(row.get(fieldnames[0], "")).strip() if fieldnames else ""
            sentence_cell = str(row.get("sentence", "")).strip()

            # Reviewer input should be sentence-only. For older id,sentence files,
            # ignore numeric ids but recover non-numeric first cells if a row was
            # accidentally written without the id column.
            if fieldnames[0] == "sentence":
                sentence_parts.append(sentence_cell)
            elif first_cell and not first_cell.isdigit():
                sentence_parts.append(first_cell)
                if sentence_cell:
                    sentence_parts.append(sentence_cell)
            else:
                sentence_parts.append(sentence_cell)

            # Be forgiving for reviewer-authored CSVs: if a sentence contains
            # unquoted commas, csv.DictReader stores the extra cells under None.
            extras = row.get(None) or []
            sentence_parts.extend(
                str(part).strip()
                for part in extras
                if part is not None and str(part).strip() and str(part).strip().lower() != "none"
            )
            sentence = ", ".join(part for part in sentence_parts if part and part.lower() != "none")
            if sentence:
                rows.append({"id": str(len(rows) + 1), "sentence": sentence})
    if not rows:
        raise ValueError(f"no valid rows found in input csv: {path}")
    return rows


def _load_model_and_tokenizer(model_dir: Path, tokenizer_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def _render_chat_messages(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    rendered_messages = [dict(m) for m in messages]
    kwargs: dict[str, Any] = {}
    model_name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    if "qwen3" in model_name:
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        rendered_messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("chat template rendering failed")
    return rendered


def _parse_single_decision(raw_text: str, candidate_e_ids: list[str]) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {"status": "parse_failure", "pred_e_ids": [], "decision_line": "", "error_type": "empty_output"}
    line = text.splitlines()[0].strip()
    if line.upper().startswith("DECISION:"):
        line = line.split(":", 1)[1].strip()
    if line == "NONE":
        return {"status": "ok", "pred_e_ids": [], "decision_line": line, "error_type": None}
    if not line.isdigit():
        return {"status": "protocol_failure", "pred_e_ids": [], "decision_line": line, "error_type": "non_numeric_decision"}
    idx = int(line) - 1
    if idx < 0 or idx >= len(candidate_e_ids):
        return {"status": "protocol_failure", "pred_e_ids": [], "decision_line": line, "error_type": "candidate_index_out_of_range"}
    return {"status": "ok", "pred_e_ids": [candidate_e_ids[idx]], "decision_line": line, "error_type": None}


def _prepare_b_runtime(dict_xlsx: Path, output_dir: Path, logger: logging.Logger) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = {
        "paths": {"dict_xlsx": str(dict_xlsx)},
        "silver": {
            "morph": {"kiwi_model": "cong-global"},
            "detect_match_window_chars": 12,
            "detect_max_matches_per_rule": 50,
            "triage_thresholds": {"confirm_min_score": 3, "hold_min_score": 1},
        },
    }
    rule_cfg = RuleEvalConfig(gold_path=str(dict_xlsx), dict_path=str(dict_xlsx))
    run_context = SimpleNamespace(run_dir=output_dir, exp_id="public_release", run_id="b_public")
    runtime = _prepare_runtime(cfg, rule_cfg, run_context, logger)

    _, _, dict_bundle = silver_loader._load_dict_stats(cfg)
    rule_sets = silver_loader._prepare_stage_rules(
        dict_bundle.get("rules", []),
        allowed_scopes={"", "all", "infer"},
    )
    hard_fail_rules = [r for r in (rule_sets.get("verify_rules") or []) if bool(r.get("hard_fail", False))]
    morph_hard_fail_rules = [r for r in (rule_sets.get("morph_verify_rules") or []) if bool(r.get("hard_fail", False))]
    return runtime, hard_fail_rules, morph_hard_fail_rules


def _summarize_rule_inventory(
    target_eids: set[str],
    expredict_map: dict[str, dict[str, Any]],
    components_by_eid: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eid in sorted(target_eids):
        exp = dict(expredict_map.get(eid, {}) or {})
        components = []
        for comp in components_by_eid.get(eid, []) or []:
            components.append(
                {
                    "comp_id": comp.get("comp_id"),
                    "comp_order": comp.get("comp_order"),
                    "comp_surf": comp.get("comp_surf"),
                    "anchor_rank": comp.get("anchor_rank"),
                    "min_gap_to_next": comp.get("min_gap_to_next"),
                    "max_gap_to_next": comp.get("max_gap_to_next"),
                    "order_policy": comp.get("order_policy"),
                    "is_required": comp.get("is_required"),
                }
            )
        rows.append(
            {
                "e_id": eid,
                "group": exp.get("group"),
                "canonical_form": exp.get("canonical_form") or exp.get("대표형"),
                "gloss": exp.get("gloss") or exp.get("뜻풀이"),
                "disconti_allowed": exp.get("disconti_allowed"),
                "components": components,
            }
        )
    return rows


def _summarize_candidates(
    candidates: list[dict[str, Any]],
    expredict_map: dict[str, dict[str, Any]],
    components_by_eid: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in candidates:
        eid = str(cand.get("e_id") or "").strip()
        meta = expredict_map.get(eid, {}) if isinstance(expredict_map, dict) else {}
        debug_meta = dict(cand.get("debug_meta") or {})
        comp_rows = []
        for comp in components_by_eid.get(eid, []) or []:
            comp_rows.append(
                {
                    "comp_id": comp.get("comp_id"),
                    "comp_order": comp.get("comp_order"),
                    "comp_surf": comp.get("comp_surf"),
                    "anchor_rank": comp.get("anchor_rank"),
                    "min_gap_to_next": comp.get("min_gap_to_next"),
                    "max_gap_to_next": comp.get("max_gap_to_next"),
                    "order_policy": comp.get("order_policy"),
                    "is_required": comp.get("is_required"),
                }
            )
        out.append(
            {
                "e_id": eid,
                "score": cand.get("score"),
                "triage": cand.get("triage"),
                "hard_fail_triggered": bool(cand.get("hard_fail_triggered", False)),
                "span_segments": cand.get("span_segments") or [],
                "matched_text": cand.get("matched_text") or cand.get("surface_text") or cand.get("text") or "",
                "group": meta.get("group"),
                "canonical_form": meta.get("canonical_form") or meta.get("대표형"),
                "gloss": meta.get("gloss") or meta.get("뜻풀이"),
                "disconti_allowed": meta.get("disconti_allowed"),
                "components": comp_rows,
                "bridge_detail": debug_meta.get("bridge") or debug_meta.get("bridge_detail"),
                "thing_bridge_detail": debug_meta.get("thing_bridge") or debug_meta.get("thing_bridge_detail"),
                "failure_reason": debug_meta.get("failure_reason"),
                "search_ranges": debug_meta.get("search_ranges"),
                "gap_violations": debug_meta.get("gap_violations"),
            }
        )
    return out

def _detect_and_filter_candidates(
    *,
    row_id: str,
    sentence: str,
    runtime: dict[str, Any],
    hard_fail_rules: list[dict[str, Any]],
    morph_hard_fail_rules: list[dict[str, Any]],
    verify_window_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instance = RuleEvalInstance(
        example_key=row_id,
        sentence=sentence,
        gold_e_id="",
        meta={"example_id": row_id, "instance_id": row_id},
    )
    expredict_map = dict(runtime.get("expredict_map") or {})
    components_by_eid = dict(runtime.get("components_by_eid") or {})
    raw_candidates = [dict(c) for c in _detect_candidates_for_instance(instance, runtime)]
    debug = {
        "rule_inventory": _summarize_rule_inventory(B_DEBUG_EIDS, expredict_map, components_by_eid),
        "raw_detected_candidates": _summarize_candidates(raw_candidates, expredict_map, components_by_eid),
        "after_hard_drop_candidates": [],
        "after_group_filter_candidates": [],
    }
    if not raw_candidates:
        return [], debug

    morph_tokens = analyze_with_kiwi(sentence, model=runtime["kiwi_model"])
    for token in morph_tokens:
        token["pos_std"] = runtime["pos_mapper"](str(token.get("pos", "")))

    silver_loader._apply_verify_rules(
        raw_sentence=sentence,
        candidates=raw_candidates,
        rules=hard_fail_rules,
        morph_rules=morph_hard_fail_rules,
        morph_tokens=morph_tokens,
        confirm_min_score=runtime["confirm_min_score"],
        hold_min_score=runtime["hold_min_score"],
        morph_window_chars=verify_window_chars,
        verify_window_chars=verify_window_chars,
    )

    kept_after_hard_drop = [
        c for c in raw_candidates
        if str(c.get("triage", "")) != "discard" and not bool(c.get("hard_fail_triggered", False))
    ]
    debug["after_hard_drop_candidates"] = _summarize_candidates(kept_after_hard_drop, expredict_map, components_by_eid)

    kept = [
        c for c in kept_after_hard_drop
        if str(c.get("e_id") or "").strip() in B_DEBUG_EIDS
    ]
    kept.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    debug["after_group_filter_candidates"] = _summarize_candidates(kept, expredict_map, components_by_eid)
    return kept, debug


def _build_messages(sentence: str, candidates: list[dict[str, Any]], expredict_map: dict[str, dict[str, Any]]) -> tuple[list[dict[str, str]], list[str], list[list[int]], str, str]:
    if not candidates:
        raise ValueError("cannot build prompt without candidates")

    primary = candidates[0]
    primary_spans = primary.get("span_segments") or []
    filtered = [c for c in candidates if (c.get("span_segments") or []) == primary_spans]
    if not filtered:
        filtered = candidates

    candidate_e_ids: list[str] = []
    seen: set[str] = set()
    for cand in filtered:
        eid = str(cand.get("e_id") or "").strip()
        if eid and eid not in seen:
            seen.add(eid)
            candidate_e_ids.append(eid)

    prompt_row = {
        "target_sentence": sentence,
        "span_segments_parsed": primary_spans,
        "candidate_e_ids": candidate_e_ids,
    }
    system_prompt, user_prompt = _build_prompt_core(
        prompt_row,
        expredict_map,
        allow_multiple=False,
    )

    target_span_text = ""
    for line in user_prompt.splitlines():
        if line.startswith("표적 표현:"):
            target_span_text = line.split(":", 1)[1].strip()
            break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = system_prompt + "\n\n" + user_prompt
    return messages, candidate_e_ids, primary_spans, target_span_text, prompt_text


def _generate_one(*, model: Any, tokenizer: Any, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    prompt_text = _render_chat_messages(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _forms_for_eids(eids: list[str], expredict_map: dict[str, dict[str, Any]]) -> list[str]:
    forms: list[str] = []
    for eid in eids or []:
        eid_str = str(eid).strip()
        meta = expredict_map.get(eid_str, {}) if isinstance(expredict_map, dict) else {}
        form = str(REVIEWER_LABELS.get(eid_str) or meta.get("canonical_form") or meta.get("대표형") or eid_str).strip()
        if form and form not in forms:
            forms.append(form)
    return forms


def _glosses_for_eids(eids: list[str], expredict_map: dict[str, dict[str, Any]]) -> list[str]:
    glosses: list[str] = []
    for eid in eids or []:
        meta = expredict_map.get(str(eid), {}) if isinstance(expredict_map, dict) else {}
        gloss = str(meta.get("gloss") or meta.get("뜻풀이") or "").strip()
        if gloss and gloss not in glosses:
            glosses.append(gloss)
    return glosses


def _write_csv(path: Path, rows: list[dict[str, Any]], expredict_map: dict[str, dict[str, Any]]) -> None:
    # Reviewer-facing CSV: column names stay familiar, values use canonical forms.
    fieldnames = [
        "id",
        "sentence",
        "target_span_text",
        "candidate_e_ids",
        "pred_e_ids",
        "pred_glosses",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pred_eids = row.get("pred_e_ids", []) or []
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "sentence": row.get("sentence", ""),
                    "target_span_text": row.get("target_span_text", ""),
                    "candidate_e_ids": ";".join(_forms_for_eids(row.get("candidate_e_ids", []) or [], expredict_map)),
                    "pred_e_ids": ";".join(_forms_for_eids(pred_eids, expredict_map)),
                    "pred_glosses": ";".join(_glosses_for_eids(pred_eids, expredict_map)),
                }
            )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary = {
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_no_candidate": sum(1 for r in rows if r.get("status") == "no_candidate"),
        "n_parse_failure": sum(1 for r in rows if r.get("status") == "parse_failure"),
        "n_protocol_failure": sum(1 for r in rows if r.get("status") == "protocol_failure"),
        "input_csv": str(args.input_csv),
        "model_dir": str(args.model_dir),
        "tokenizer_dir": str(args.tokenizer_dir),
        "dict_xlsx": str(args.dict_xlsx),
        "verify_window_chars": args.verify_window_chars,
        "max_new_tokens": args.max_new_tokens,
        "policy": {
            "candidate_stage": "detect",
            "hard_drop_only": True,
            "group_filter": sorted(B_DEBUG_EIDS),
            "allow_multiple": False,
            "prompt_source": "build_bgroup_sft._build_prompt_core",
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_prompt_preview(path: Path, row: dict[str, Any]) -> None:
    payload = {
        "id": row.get("id", ""),
        "sentence": row.get("sentence", ""),
        "target_span_text": row.get("target_span_text", ""),
        "candidate_e_ids": row.get("candidate_e_ids", []),
        "prompt_text": row.get("prompt_text", ""),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_debug_detection(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = {
                "id": row.get("id", ""),
                "sentence": row.get("sentence", ""),
                "rule_inventory": row.get("rule_inventory", []),
                "raw_detected_candidates": row.get("raw_detected_candidates", []),
                "after_hard_drop_candidates": row.get("after_hard_drop_candidates", []),
                "after_group_filter_candidates": row.get("after_group_filter_candidates", []),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("run_b_llm_public")

    args.input_csv = args.input_csv.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir = args.model_dir.resolve()
    args.tokenizer_dir = args.tokenizer_dir.resolve()
    args.dict_xlsx = args.dict_xlsx.resolve()

    rows = _read_input_csv(args.input_csv)
    model, tokenizer = _load_model_and_tokenizer(args.model_dir, args.tokenizer_dir)
    runtime, hard_fail_rules, morph_hard_fail_rules = _prepare_b_runtime(args.dict_xlsx, args.output_dir, logger)
    expredict_map = dict(runtime.get("expredict_map") or {})

    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        rid = row["id"]
        sentence = row["sentence"]
        candidates, debug = _detect_and_filter_candidates(
            row_id=rid,
            sentence=sentence,
            runtime=runtime,
            hard_fail_rules=hard_fail_rules,
            morph_hard_fail_rules=morph_hard_fail_rules,
            verify_window_chars=args.verify_window_chars,
        )
        if not candidates:
            pred_rows.append(
                {
                    "id": rid,
                    "sentence": sentence,
                    "status": "no_candidate",
                    "target_span_text": "",
                    "candidate_e_ids": [],
                    "pred_e_ids": [],
                    "decision_line": "",
                    "raw_output": "",
                    "error_type": None,
                    "prompt_text": "",
                    "rule_inventory": debug["rule_inventory"],
                    "raw_detected_candidates": debug["raw_detected_candidates"],
                    "after_hard_drop_candidates": debug["after_hard_drop_candidates"],
                    "after_group_filter_candidates": debug["after_group_filter_candidates"],
                }
            )
            continue

        messages, candidate_e_ids, primary_spans, target_span_text, prompt_text = _build_messages(sentence, candidates, expredict_map)
        raw_output = _generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
        )
        parsed = _parse_single_decision(raw_output, candidate_e_ids)
        pred_rows.append(
            {
                "id": rid,
                "sentence": sentence,
                "status": parsed["status"],
                "target_span_text": target_span_text,
                "primary_span_segments": primary_spans,
                "candidate_e_ids": candidate_e_ids,
                "pred_e_ids": parsed["pred_e_ids"],
                "decision_line": parsed["decision_line"],
                "raw_output": raw_output,
                "error_type": parsed["error_type"],
                "messages": messages,
                "prompt_text": prompt_text,
                "rule_inventory": debug["rule_inventory"],
                "raw_detected_candidates": debug["raw_detected_candidates"],
                "after_hard_drop_candidates": debug["after_hard_drop_candidates"],
                "after_group_filter_candidates": debug["after_group_filter_candidates"],
            }
        )

    _write_csv(args.output_dir / "predictions.csv", pred_rows, expredict_map)
    _write_jsonl(args.output_dir / "predictions.jsonl", pred_rows)
    _write_summary(args.output_dir / "summary.json", pred_rows, args)
    _write_debug_detection(args.output_dir / "debug_detection.jsonl", pred_rows)
    if pred_rows:
        _write_prompt_preview(args.output_dir / "prompt_preview_first.json", pred_rows[0])
    logger.info("wrote outputs under %s", args.output_dir)


if __name__ == "__main__":
    main()
