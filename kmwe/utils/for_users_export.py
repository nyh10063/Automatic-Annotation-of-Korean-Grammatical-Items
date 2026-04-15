from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _safe_join(values: Iterable[Any], sep: str = ";") -> str:
    items = [str(value) for value in values if value is not None]
    return sep.join(items)


def _candidate_item(candidate: dict[str, Any]) -> str:
    e_id = candidate.get("e_id", "")
    score = candidate.get("score", "")
    span_text = candidate.get("span_text", "")
    span_segments = candidate.get("span_segments", candidate.get("span", ""))
    stage_hits = candidate.get("stage_hits", {}) or {}
    detect_hits = _safe_join(stage_hits.get("detect", []))
    verify_hits = _safe_join(stage_hits.get("verify", []))
    context_hits = _safe_join(_merge_context_hits(stage_hits))
    return (
        f"{e_id}|score={score}|span={span_text}|segments={span_segments}"
        f"|detect={detect_hits}|verify={verify_hits}|context={context_hits}"
    )


def _extract_eids(candidates: list[dict[str, Any]]) -> str:
    return _safe_join([candidate.get("e_id", "") for candidate in candidates], sep=";")


def _merge_context_hits(stage_hits: dict[str, Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for key in ("context", "context_pos", "context_neg"):
        values = stage_hits.get(key, []) or []
        for value in values:
            token = str(value)
            if token in seen:
                continue
            seen.add(token)
            merged.append(value)
    return merged


def _sort_key(candidate: dict[str, Any]) -> tuple[int, str, int]:
    score = candidate.get("score", 0)
    try:
        score_value = int(score)
    except (TypeError, ValueError):
        score_value = 0
    e_id = str(candidate.get("e_id", ""))
    segments = candidate.get("span_segments") or []
    span_start = int(segments[0][0]) if segments else 0
    return (-score_value, e_id, span_start)


def _pretty_block(sent_index: Any, target_sentence: str, buckets: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["=" * 80, f"sent_index={sent_index} | sentence={target_sentence}"]
    for triage in ("confirm", "hold", "drop"):
        candidates = buckets.get(triage, [])
        if not candidates:
            continue
        lines.append(f"[{triage}] {len(candidates)} candidates")
        for candidate in sorted(candidates, key=_sort_key):
            e_id = candidate.get("e_id", "")
            score = candidate.get("score", "")
            span_segments = candidate.get("span_segments", "")
            span_text = candidate.get("span_text", "")
            lines.append(
                f"- e_id={e_id} score={score} span={span_segments} span_text={span_text}"
            )
    return "\n".join(lines)


def _match_blocks(target_sentence: str, candidates: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for candidate in candidates:
        stage_hits = candidate.get("stage_hits", {}) or {}
        detect_hits = stage_hits.get("detect", []) or []
        verify_hits = stage_hits.get("verify", []) or []
        context_hits = _merge_context_hits(stage_hits)
        blocks.append(
            "\n".join(
                [
                    "=== MATCH ===",
                    f"SENT: {target_sentence}",
                    f"EID: {candidate.get('e_id', '')} TRIAGE: {candidate.get('triage', '')} "
                    f"SCORE: {candidate.get('score', '')}",
                    f"SPAN: {candidate.get('span_segments', '')} SPAN_TEXT: {candidate.get('span_text', '')}",
                    f"DETECT_HITS: {detect_hits}",
                    f"VERIFY_HITS: {verify_hits}",
                    f"CONTEXT_HITS: {context_hits}",
                ]
            )
        )
    return "\n\n".join(blocks)


def export_silver_user_csv(
    silver_jsonl_path: str,
    out_csv_path: str,
    *,
    run_id: str | None = None,
    exp_id: str | None = None,
    stage: str = "build_silver",
    input_jsonl_path: str | None = None,
    report_json_path: str | None = None,
) -> dict[str, Any]:
    input_path = Path(silver_jsonl_path)
    output_path = Path(out_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        return {"out_csv_path": str(output_path), "n_sentences": 0}

    if stage != "build_silver":
        rows: list[dict[str, Any]] = []
        n_sentences = 0
        with input_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                n_sentences += 1
                obj = json.loads(line)

                doc_id = obj.get("doc_id", "")
                sent_index = obj.get("sent_index", obj.get("sent_id", ""))
                target_sentence = obj.get("target_sentence", obj.get("raw_sentence", ""))
                source = obj.get("source", "")
                row_id = obj.get("row_id", "")

                candidates = obj.get("candidates", []) or []
                buckets = {"confirm": [], "hold": [], "discard": []}
                for candidate in candidates:
                    triage = (candidate.get("triage") or "discard").lower()
                    if triage not in buckets:
                        triage = "discard"
                    buckets[triage].append(candidate)

                buckets_pretty = {
                    "confirm": buckets["confirm"],
                    "hold": buckets["hold"],
                    "drop": buckets["discard"],
                }

                confirm_items = [_candidate_item(candidate) for candidate in buckets["confirm"]]
                hold_items = [_candidate_item(candidate) for candidate in buckets["hold"]]
                discard_items = [_candidate_item(candidate) for candidate in buckets["discard"]]

                match_candidates = buckets["confirm"] + buckets["hold"]

                rows.append(
                    {
                        "run_id": run_id or "",
                        "exp_id": exp_id or "",
                        "stage": stage,
                        "input_jsonl_path": input_jsonl_path or "",
                        "doc_id": doc_id,
                        "sent_index": sent_index,
                        "row_id": row_id,
                        "target_sentence": target_sentence,
                        "source": source,
                        "n_candidates": len(candidates),
                        "n_confirm": len(buckets["confirm"]),
                        "n_hold": len(buckets["hold"]),
                        "n_discard": len(buckets["discard"]),
                        "confirm_eids": _extract_eids(buckets["confirm"]),
                        "hold_eids": _extract_eids(buckets["hold"]),
                        "discard_eids": _extract_eids(buckets["discard"]),
                        "confirm_details": " || ".join(confirm_items),
                        "hold_details": " || ".join(hold_items),
                        "discard_details": " || ".join(discard_items),
                        "pretty_sentence_block": _pretty_block(
                            sent_index, target_sentence, buckets_pretty
                        ),
                        "match_blocks": _match_blocks(target_sentence, match_candidates),
                    }
                )

        fieldnames = list(rows[0].keys()) if rows else []
        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return {"out_csv_path": str(output_path), "n_sentences": n_sentences}

    report_samples: list[dict[str, Any]] = []
    if report_json_path:
        report_path = Path(report_json_path)
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_samples = report.get("detect_components_span_fail_samples", []) or []

    def _json_min(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)

    def _row_key(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        return (
            row.get("doc_id"),
            row.get("sent_index"),
            row.get("example_id"),
            row.get("instance_id"),
        )

    fail_map: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for sample in report_samples:
        key = _row_key(sample)
        fail_map.setdefault(key, []).append(sample)

    fieldnames = [
        "row_type",
        "run_id",
        "exp_id",
        "stage",
        "doc_id",
        "sent_index",
        "example_id",
        "instance_id",
        "target_sentence",
        "n_candidates_all",
        "n_confirm",
        "n_hold",
        "n_discard",
        "confirmed_span_keys_json",
        "hold_span_keys_json",
        "status",
        "status_detail",
        "e_id",
        "triage",
        "score",
        "span_key",
        "span_segments_json",
        "span_text",
        "hard_fail",
        "hard_fail_reasons_json",
        "bridge_applied",
        "bridge_json",
        "thing_bridge_applied",
        "thing_bridge_json",
        "morph_snippet_json",
        "detect_components_json",
        "rule_id",
        "ruleset_id",
        "match_span",
        "match_text",
        "detect_window",
        "anchor_selected_span",
        "anchor_selected_kind",
        "failure_reason",
        "gap_violations_json",
        "per_comp_debug_json",
    ]

    rows: list[dict[str, Any]] = []
    n_sentences = 0
    used_fail_samples: set[int] = set()
    with input_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_sentences += 1
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            sent_index = obj.get("sent_index")
            example_id = obj.get("example_id")
            instance_id = obj.get("instance_id")
            target_sentence = obj.get("target_sentence", obj.get("raw_sentence", ""))

            candidates = obj.get("candidates", []) or []
            n_confirm = sum(1 for c in candidates if c.get("triage") == "confirm")
            n_hold = sum(1 for c in candidates if c.get("triage") == "hold")
            n_discard = sum(1 for c in candidates if c.get("triage") == "discard")
            confirm_span_keys = [c.get("span_key") for c in candidates if c.get("triage") == "confirm"]
            hold_span_keys = [c.get("span_key") for c in candidates if c.get("triage") == "hold"]

            row_key = (doc_id, sent_index, example_id, instance_id)
            has_fail = bool(fail_map.get(row_key))
            status = "HAS_FAIL" if has_fail else ("NO_CANDIDATE" if not candidates else "OK")
            status_detail = f"n_fail={len(fail_map.get(row_key, []))}"

            rows.append(
                {
                    "row_type": "sentence",
                    "run_id": run_id or "",
                    "exp_id": exp_id or "",
                    "stage": stage,
                    "doc_id": doc_id,
                    "sent_index": sent_index,
                    "example_id": example_id,
                    "instance_id": instance_id,
                    "target_sentence": target_sentence,
                    "n_candidates_all": len(candidates),
                    "n_confirm": n_confirm,
                    "n_hold": n_hold,
                    "n_discard": n_discard,
                    "confirmed_span_keys_json": _json_min(confirm_span_keys),
                    "hold_span_keys_json": _json_min(hold_span_keys),
                    "status": status,
                    "status_detail": status_detail,
                    "e_id": "",
                    "triage": "",
                    "score": "",
                    "span_key": "",
                    "span_segments_json": "",
                    "span_text": "",
                    "hard_fail": "",
                    "hard_fail_reasons_json": "",
                    "bridge_applied": "",
                    "bridge_json": "",
                    "thing_bridge_applied": "",
                    "thing_bridge_json": "",
                    "morph_snippet_json": "",
                    "detect_components_json": "",
                    "rule_id": "",
                    "ruleset_id": "",
                    "match_span": "",
                    "match_text": "",
                    "detect_window": "",
                    "anchor_selected_span": "",
                    "anchor_selected_kind": "",
                    "failure_reason": "",
                    "gap_violations_json": "",
                    "per_comp_debug_json": "",
                }
            )

            for cand in candidates:
                debug_meta = cand.get("debug_meta") or {}
                bridge = debug_meta.get("bridge") or {}
                thing_bridge = debug_meta.get("thing_bridge") or {}
                morph_snippet = {
                    "window": debug_meta.get("morph_snippet_window"),
                    "tokens": debug_meta.get("morph_snippet") or [],
                }
                detect_components = {
                    "detect": debug_meta.get("detect") or {},
                    "components_debug": debug_meta.get("components_debug") or {},
                }
                rows.append(
                    {
                        "row_type": "candidate",
                        "run_id": run_id or "",
                        "exp_id": exp_id or "",
                        "stage": stage,
                        "doc_id": doc_id,
                        "sent_index": sent_index,
                        "example_id": example_id,
                        "instance_id": instance_id,
                        "target_sentence": target_sentence,
                        "n_candidates_all": "",
                        "n_confirm": "",
                        "n_hold": "",
                        "n_discard": "",
                        "confirmed_span_keys_json": "",
                        "hold_span_keys_json": "",
                        "status": "",
                        "status_detail": "",
                        "e_id": cand.get("e_id"),
                        "triage": cand.get("triage"),
                        "score": cand.get("score"),
                        "span_key": cand.get("span_key"),
                        "span_segments_json": _json_min(cand.get("span_segments") or []),
                        "span_text": cand.get("span_text", ""),
                        "hard_fail": bool(cand.get("hard_fail_triggered")),
                        "hard_fail_reasons_json": _json_min(cand.get("hard_fail_reasons") or []),
                        "bridge_applied": bool(bridge.get("applied")),
                        "bridge_json": _json_min(bridge or {}),
                        "thing_bridge_applied": bool(thing_bridge.get("applied")),
                        "thing_bridge_json": _json_min(thing_bridge or {}),
                        "morph_snippet_json": _json_min(morph_snippet),
                        "detect_components_json": _json_min(detect_components),
                        "rule_id": "",
                        "ruleset_id": "",
                        "match_span": "",
                        "match_text": "",
                        "detect_window": "",
                        "anchor_selected_span": "",
                        "anchor_selected_kind": "",
                        "failure_reason": "",
                        "gap_violations_json": "",
                        "per_comp_debug_json": "",
                    }
                )

            for sample in fail_map.get(row_key, []):
                used_fail_samples.add(id(sample))
                per_comp_debug = sample.get("per_comp_debug") or {}
                failure_reason = sample.get("note") or ""
                if not failure_reason:
                    for debug in per_comp_debug.values():
                        if isinstance(debug, dict) and debug.get("failure_reason"):
                            failure_reason = debug.get("failure_reason")
                            break
                rows.append(
                    {
                        "row_type": "detect_fail",
                        "run_id": run_id or "",
                        "exp_id": exp_id or "",
                        "stage": stage,
                        "doc_id": doc_id,
                        "sent_index": sent_index,
                        "example_id": example_id,
                        "instance_id": instance_id,
                        "target_sentence": target_sentence,
                        "n_candidates_all": "",
                        "n_confirm": "",
                        "n_hold": "",
                        "n_discard": "",
                        "confirmed_span_keys_json": "",
                        "hold_span_keys_json": "",
                        "status": "",
                        "status_detail": "",
                        "e_id": sample.get("e_id"),
                        "triage": "",
                        "score": "",
                        "span_key": "",
                        "span_segments_json": "",
                        "span_text": "",
                        "hard_fail": "",
                        "hard_fail_reasons_json": "",
                        "bridge_applied": "",
                        "bridge_json": "",
                        "thing_bridge_applied": "",
                        "thing_bridge_json": "",
                        "morph_snippet_json": _json_min(
                            {
                                "tokens": sample.get("morph_token_snippet", []),
                            }
                        ),
                        "detect_components_json": "",
                        "rule_id": sample.get("rule_id"),
                        "ruleset_id": sample.get("ruleset_id"),
                        "match_span": _json_min(sample.get("match_span")),
                        "match_text": sample.get("match_text"),
                        "detect_window": _json_min(sample.get("detect_window")),
                        "anchor_selected_span": _json_min(sample.get("anchor_selected_span")),
                        "anchor_selected_kind": sample.get("anchor_selected_kind"),
                        "failure_reason": failure_reason,
                        "gap_violations_json": _json_min(sample.get("gap_violations", [])),
                        "per_comp_debug_json": _json_min(per_comp_debug),
                    }
                )

    for sample in report_samples:
        if id(sample) in used_fail_samples:
            continue
        per_comp_debug = sample.get("per_comp_debug") or {}
        failure_reason = sample.get("note") or ""
        if not failure_reason:
            for debug in per_comp_debug.values():
                if isinstance(debug, dict) and debug.get("failure_reason"):
                    failure_reason = debug.get("failure_reason")
                    break
        rows.append(
            {
                "row_type": "detect_fail",
                "run_id": run_id or "",
                "exp_id": exp_id or "",
                "stage": stage,
                "doc_id": sample.get("doc_id"),
                "sent_index": sample.get("sent_index"),
                "example_id": sample.get("example_id"),
                "instance_id": sample.get("instance_id"),
                "target_sentence": sample.get("target_sentence"),
                "n_candidates_all": "",
                "n_confirm": "",
                "n_hold": "",
                "n_discard": "",
                "confirmed_span_keys_json": "",
                "hold_span_keys_json": "",
                "status": "",
                "status_detail": "",
                "e_id": sample.get("e_id"),
                "triage": "",
                "score": "",
                "span_key": "",
                "span_segments_json": "",
                "span_text": "",
                "hard_fail": "",
                "hard_fail_reasons_json": "",
                "bridge_applied": "",
                "bridge_json": "",
                "thing_bridge_applied": "",
                "thing_bridge_json": "",
                "morph_snippet_json": _json_min(
                    {
                        "tokens": sample.get("morph_token_snippet", []),
                    }
                ),
                "detect_components_json": "",
                "rule_id": sample.get("rule_id"),
                "ruleset_id": sample.get("ruleset_id"),
                "match_span": _json_min(sample.get("match_span")),
                "match_text": sample.get("match_text"),
                "detect_window": _json_min(sample.get("detect_window")),
                "anchor_selected_span": _json_min(sample.get("anchor_selected_span")),
                "anchor_selected_kind": sample.get("anchor_selected_kind"),
                "failure_reason": failure_reason,
                "gap_violations_json": _json_min(sample.get("gap_violations", [])),
                "per_comp_debug_json": _json_min(per_comp_debug),
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {"out_csv_path": str(output_path), "n_sentences": n_sentences}
