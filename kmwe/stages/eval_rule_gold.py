from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from kmwe.data.rule_eval import (
    RuleEvalConfig,
    build_prediction,
    compute_coverage_metrics,
    compute_strict_metrics,
    load_gold_instances,
    prediction_to_row,
    summarize_coverage_by_field,
    validate_input_paths,
    write_predictions_csv,
)
from kmwe.stages import build_silver as silver_loader
from kmwe.utils.jsonio import write_json, write_jsonl_line
from kmwe.utils.morph import analyze_with_kiwi


def run_eval_rule_gold(*, cfg: dict[str, Any], run_context: Any) -> dict[str, Any]:
    logger = logging.getLogger("kmwe")
    rule_cfg = RuleEvalConfig(
        gold_path=str(
            cfg.get("paths", {}).get("gold_xlsx")
            or cfg.get("paths", {}).get("gold_b_xlsx")
            or ""
        ),
        dict_path=str(cfg.get("paths", {}).get("dict_xlsx") or ""),
        gold_sheet_name=str(cfg.get("rule_eval", {}).get("gold_sheet_name", "gold")),
        with_downstream=cfg.get("rule_eval", {}).get("with_downstream"),
    )
    outputs_dir = Path(run_context.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    path_errors = validate_input_paths(rule_cfg)
    if path_errors:
        raise FileNotFoundError(json.dumps(path_errors, ensure_ascii=False))

    runtime = _prepare_runtime(cfg, rule_cfg, run_context, logger)
    instances, gold_meta = load_gold_instances(
        rule_cfg.gold_path,
        sheet_name=rule_cfg.gold_sheet_name,
        expredict_map=runtime["expredict_map"],
    )

    predictions = []
    top_error_patterns: Counter[str] = Counter()
    for instance in instances:
        candidates = _detect_candidates_for_instance(instance, runtime)
        prediction = build_prediction(instance, candidates=candidates)
        predictions.append(prediction)
        if prediction.error_reason:
            top_error_patterns[prediction.error_reason] += 1

    coverage_metrics = compute_coverage_metrics(predictions)
    strict_metrics = compute_strict_metrics(predictions)
    group_a_predictions = [p for p in predictions if str(p.gold_group or "").lower() == "a"]
    group_b_predictions = [p for p in predictions if str(p.gold_group or "").lower() == "b"]
    by_e_id_metrics = summarize_coverage_by_field(
        [p for p in predictions if p.gold_e_id != "__NONE__"],
        "gold_e_id",
    )
    by_polyset_metrics = summarize_coverage_by_field(
        [p for p in predictions if p.gold_polyset_id],
        "gold_polyset_id",
    )

    report = {
        "stage": "eval_rule_gold",
        "with_downstream": rule_cfg.with_downstream,
        "inputs": {
            "gold_path": rule_cfg.gold_path,
            "dict_path": rule_cfg.dict_path,
            "gold_sheet_name": rule_cfg.gold_sheet_name,
        },
        "gold_meta": gold_meta,
        "runtime_meta": {
            "dict_source": runtime["dict_source"],
            "dict_stats": runtime["dict_stats"],
            "detect_rule_count": len(runtime["detect_rules"]),
            "kiwi_model": runtime["kiwi_model"],
        },
        "n_instances": len(instances),
        "group_a_rule_metrics": compute_coverage_metrics(group_a_predictions),
        "group_b_rule_metrics": compute_coverage_metrics(group_b_predictions),
        "overall_rule_metrics": strict_metrics,
        "coverage_metrics": coverage_metrics,
        "by_e_id_metrics": by_e_id_metrics,
        "by_polyset_metrics": by_polyset_metrics,
        "top_error_patterns": top_error_patterns.most_common(20),
        "coverage_summary": coverage_metrics,
        "outputs": {
            "rule_predictions_jsonl": str(outputs_dir / "rule_predictions.jsonl"),
            "rule_predictions_csv": str(outputs_dir / "rule_predictions.csv"),
            "coverage_summary_json": str(outputs_dir / "coverage_summary.json"),
            "polyset_metrics_csv": str(outputs_dir / "polyset_metrics.csv"),
        },
        "implementation_status": "detect_only_v1",
    }

    rows = [prediction_to_row(p) for p in predictions]
    with (outputs_dir / "rule_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            write_jsonl_line(fh, row)
    write_predictions_csv(outputs_dir / "rule_predictions.csv", rows)
    write_json(outputs_dir / "coverage_summary.json", coverage_metrics, indent=2)
    write_json(outputs_dir / "error_breakdown.json", {"top_error_patterns": report["top_error_patterns"]}, indent=2)
    with (outputs_dir / "error_cases.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            if row.get("error_reason"):
                write_jsonl_line(fh, row)
    _write_polyset_csv(outputs_dir / "polyset_metrics.csv", by_polyset_metrics)
    write_json(outputs_dir / "eval_rule_gold_report.json", report, indent=2)
    return report


def _prepare_runtime(cfg: dict[str, Any], rule_cfg: RuleEvalConfig, run_context: Any, logger: logging.Logger) -> dict[str, Any]:
    runtime_cfg = dict(cfg)
    runtime_cfg.setdefault("paths", {})
    runtime_cfg["paths"] = dict(runtime_cfg["paths"])
    runtime_cfg["paths"]["dict_xlsx"] = rule_cfg.dict_path
    dict_source, dict_stats, dict_bundle = silver_loader._load_dict_stats(runtime_cfg)
    rule_sets = silver_loader._prepare_stage_rules(
        dict_bundle.get("rules", []),
        allowed_scopes={"", "all", "infer"},
    )
    expredict_map = dict(dict_bundle.get("expredict_map") or {})
    if not expredict_map:
        expredict_map = {row.get("e_id"): row for row in dict_bundle.get("expredict", []) if row.get("e_id")}
    components_by_eid = silver_loader._index_components_by_eid(dict_bundle.get("components", []))
    pos_mapper, _ = silver_loader._load_pos_mapping(runtime_cfg, run_context, logger)
    silver_cfg = cfg.get("silver", {}) or {}
    kiwi_model = str((silver_cfg.get("morph", {}) or {}).get("kiwi_model", "cong-global"))
    detect_match_window_chars = int(silver_cfg.get("detect_match_window_chars", 12))
    detect_max_matches_per_rule = int(silver_cfg.get("detect_max_matches_per_rule", 50))
    triage_cfg = silver_cfg.get("triage_thresholds", {}) or {}
    return {
        "dict_source": dict_source,
        "dict_stats": dict_stats,
        "expredict_map": expredict_map,
        "components_by_eid": components_by_eid,
        "detect_rules": rule_sets["detect_rules"],
        "confirm_min_score": int(triage_cfg.get("confirm_min_score", 3)),
        "hold_min_score": int(triage_cfg.get("hold_min_score", 1)),
        "kiwi_model": kiwi_model,
        "pos_mapper": pos_mapper,
        "detect_match_window_chars": detect_match_window_chars,
        "detect_max_matches_per_rule": detect_max_matches_per_rule,
    }


def _detect_candidates_for_instance(instance: Any, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    morph_tokens = analyze_with_kiwi(instance.sentence, model=runtime["kiwi_model"])
    for token in morph_tokens:
        token["pos_std"] = runtime["pos_mapper"](str(token.get("pos", "")))
    record = {
        "example_id": instance.meta.get("example_id"),
        "instance_id": instance.meta.get("instance_id"),
        "target_sentence": instance.sentence,
    }
    detect_result = silver_loader._detect_candidates(
        instance.sentence,
        runtime["detect_rules"],
        runtime["expredict_map"],
        runtime["confirm_min_score"],
        runtime["hold_min_score"],
        **silver_loader._build_detect_kwargs(
            record=record,
            raw_sentence=instance.sentence,
            components_by_eid=runtime["components_by_eid"],
            morph_tokens=morph_tokens,
            detect_match_window_chars=runtime["detect_match_window_chars"],
            detect_max_matches_per_rule=runtime["detect_max_matches_per_rule"],
            include_debug_ctx=False,
        ),
    )
    return list(detect_result.get("candidates") or [])


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
