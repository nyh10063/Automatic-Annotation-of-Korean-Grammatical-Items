#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kmwe.data.factory import AGROUP_INPUT_CONSTRUCTION_VERSION_V2
from kmwe.data.rule_eval import RuleEvalConfig, RuleEvalInstance
from kmwe.stages import build_silver as silver_loader
from kmwe.stages.eval_rule_gold import _detect_candidates_for_instance, _prepare_runtime
from kmwe.stages.infer_step1 import _apply_encoder_confidence, _build_encoder_scorer, _score_candidates_with_encoder
from kmwe.utils.morph import analyze_with_kiwi


DEFAULT_A_THRESHOLD = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public A-pipeline encoder runner")
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--best_dir", required=True, type=Path)
    parser.add_argument("--dict_xlsx", required=True, type=Path)
    parser.add_argument("--verify_window_chars", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=DEFAULT_A_THRESHOLD)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def _read_input_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "id" not in fieldnames or "sentence" not in fieldnames:
            raise ValueError(f"input csv must contain 'id' and 'sentence': {path}")
        for row in reader:
            rid = str(row.get("id", "")).strip()
            sentence = str(row.get("sentence", "")).strip()
            if rid and sentence:
                rows.append({"id": rid, "sentence": sentence})
    if not rows:
        raise ValueError(f"no valid rows found in input csv: {path}")
    return rows


def _prepare_a_runtime(dict_xlsx: Path, output_dir: Path, logger: logging.Logger) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    run_context = SimpleNamespace(run_dir=output_dir, exp_id="public_release", run_id="a_public")
    runtime = _prepare_runtime(cfg, rule_cfg, run_context, logger)

    _, _, dict_bundle = silver_loader._load_dict_stats(cfg)
    rule_sets = silver_loader._prepare_stage_rules(
        dict_bundle.get("rules", []),
        allowed_scopes={"", "all", "infer"},
    )
    hard_fail_rules = [r for r in (rule_sets.get("verify_rules") or []) if bool(r.get("hard_fail", False))]
    morph_hard_fail_rules = [r for r in (rule_sets.get("morph_verify_rules") or []) if bool(r.get("hard_fail", False))]
    return runtime, hard_fail_rules, morph_hard_fail_rules


def _detect_and_filter_candidates(
    *,
    row_id: str,
    sentence: str,
    runtime: dict[str, Any],
    hard_fail_rules: list[dict[str, Any]],
    morph_hard_fail_rules: list[dict[str, Any]],
    verify_window_chars: int,
) -> list[dict[str, Any]]:
    instance = RuleEvalInstance(
        example_key=row_id,
        sentence=sentence,
        gold_e_id="",
        meta={"example_id": row_id, "instance_id": row_id},
    )
    candidates = list(_detect_candidates_for_instance(instance, runtime))
    if not candidates:
        return []

    morph_tokens = analyze_with_kiwi(sentence, model=runtime["kiwi_model"])
    for token in morph_tokens:
        token["pos_std"] = runtime["pos_mapper"](str(token.get("pos", "")))

    silver_loader._apply_verify_rules(
        raw_sentence=sentence,
        candidates=candidates,
        rules=hard_fail_rules,
        morph_rules=morph_hard_fail_rules,
        morph_tokens=morph_tokens,
        confirm_min_score=runtime["confirm_min_score"],
        hold_min_score=runtime["hold_min_score"],
        morph_window_chars=verify_window_chars,
        verify_window_chars=verify_window_chars,
    )

    kept = [c for c in candidates if str(c.get("triage", "")) != "discard" and not bool(c.get("hard_fail_triggered", False))]
    kept = [c for c in kept if str((runtime.get("expredict_map") or {}).get(str(c.get("e_id") or ""), {}).get("group", "")).strip().lower() == "a"]
    kept.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    return kept


def _build_scorer(best_dir: Path, output_dir: Path, logger: logging.Logger):
    cfg = {
        "infer": {
            "encoder_scoring_checkpoint": str(best_dir),
            "scoring_method": "head_logits",
        },
        "runtime": {},
    }
    # infer_step1._build_encoder_scorer always calls _artifacts_root_from_outputs_dir()
    # even when encoder_scoring_checkpoint override is provided. For the public runner we
    # keep the original scorer path, but hand it a synthetic deep outputs_dir so that the
    # depth check passes without changing the original scoring logic.
    synthetic_outputs_dir = output_dir / '_public_release' / 'exp' / 'run' / 'outputs'
    synthetic_outputs_dir.mkdir(parents=True, exist_ok=True)
    run_context = SimpleNamespace(outputs_dir=synthetic_outputs_dir, exp_id="public_release")
    return _build_encoder_scorer(
        cfg=cfg,
        run_context=run_context,
        enabled=True,
        max_seq_len=256,
        logger=logger,
        require_head_logits=True,
        disallow_fallback_scoring=True,
    )


def _run_a_downstream(
    *,
    candidates: list[dict[str, Any]],
    sentence: str,
    runtime: dict[str, Any],
    scorer: Any,
    threshold: float,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[str | None, dict[str, Any], str | None, list[dict[str, Any]]]:
    scored = [dict(c) for c in candidates]
    _score_candidates_with_encoder(
        candidates=scored,
        raw_sentence=sentence,
        context_left="",
        context_right="",
        scorer=scorer,
        scoring_enabled=True,
        batch_size=batch_size,
        logger=logger,
        require_head_logits=True,
        expredict_map=dict(runtime.get("expredict_map") or {}),
        input_construction_version=AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
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
        return None, {"top1_confidence": None, "n_candidates": 0}, "no_candidates_after_rule_gate", scored
    top1 = scored[0]
    top1_conf = float(top1.get("confidence", 0.0) or 0.0)
    pred_eid = str(top1.get("e_id") or "").strip() if top1_conf >= float(threshold) else None
    raw_output = {
        "top1_e_id": str(top1.get("e_id") or "").strip() or None,
        "top1_confidence": top1_conf,
        "threshold": float(threshold),
        "n_candidates": len(scored),
        "downstream_candidate_e_ids": [str(c.get("e_id") or "").strip() for c in scored],
        "downstream_input_text_a": top1.get("encoder_input_text_a"),
        "downstream_input_text_b": top1.get("encoder_input_text_b"),
        "downstream_input_version": top1.get("encoder_input_version") or AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
        "downstream_input_text_b_format": top1.get("encoder_input_text_b_format"),
    }
    if pred_eid is None:
        return None, raw_output, f"group_a_below_threshold<{threshold:.2f}", scored
    return pred_eid, raw_output, None, scored


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "sentence",
        "status",
        "candidate_e_ids",
        "pred_e_id",
        "top1_e_id",
        "top1_confidence",
        "threshold",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "sentence": row.get("sentence", ""),
                    "status": row.get("status", ""),
                    "candidate_e_ids": "|".join(row.get("candidate_e_ids", [])),
                    "pred_e_id": row.get("pred_e_id", "") or "",
                    "top1_e_id": row.get("top1_e_id", "") or "",
                    "top1_confidence": row.get("top1_confidence", ""),
                    "threshold": row.get("threshold", ""),
                    "error": row.get("error", "") or "",
                }
            )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary = {
        "n_rows": len(rows),
        "n_predicted": sum(1 for r in rows if r.get("pred_e_id")),
        "n_no_candidate": sum(1 for r in rows if r.get("status") == "no_candidate"),
        "n_below_threshold": sum(1 for r in rows if r.get("status") == "below_threshold"),
        "input_csv": str(args.input_csv),
        "best_dir": str(args.best_dir),
        "dict_xlsx": str(args.dict_xlsx),
        "verify_window_chars": args.verify_window_chars,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "policy": {
            "candidate_stage": "detect",
            "hard_drop_only": True,
            "group_filter": "a",
            "encoder_input_source": "infer_step1._score_candidates_with_encoder",
            "encoder_input_version": AGROUP_INPUT_CONSTRUCTION_VERSION_V2,
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def _write_input_preview(path: Path, row: dict[str, Any]) -> None:
    payload = {
        "id": row.get("id", ""),
        "sentence": row.get("sentence", ""),
        "candidate_e_ids": row.get("candidate_e_ids", []),
        "top1_e_id": row.get("top1_e_id", None),
        "top1_confidence": row.get("top1_confidence", None),
        "downstream_input_text_a": row.get("downstream_input_text_a", None),
        "downstream_input_text_b": row.get("downstream_input_text_b", None),
        "downstream_input_version": row.get("downstream_input_version", None),
        "downstream_input_text_b_format": row.get("downstream_input_text_b_format", None),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger('run_a_public')

    args.input_csv = args.input_csv.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.best_dir = args.best_dir.resolve()
    args.dict_xlsx = args.dict_xlsx.resolve()

    rows = _read_input_csv(args.input_csv)
    runtime, hard_fail_rules, morph_hard_fail_rules = _prepare_a_runtime(args.dict_xlsx, args.output_dir, logger)
    scorer = _build_scorer(args.best_dir, args.output_dir, logger)

    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        rid = row['id']
        sentence = row['sentence']
        candidates = _detect_and_filter_candidates(
            row_id=rid,
            sentence=sentence,
            runtime=runtime,
            hard_fail_rules=hard_fail_rules,
            morph_hard_fail_rules=morph_hard_fail_rules,
            verify_window_chars=args.verify_window_chars,
        )
        if not candidates:
            pred_rows.append({
                'id': rid,
                'sentence': sentence,
                'status': 'no_candidate',
                'candidate_e_ids': [],
                'pred_e_id': None,
                'top1_e_id': None,
                'top1_confidence': None,
                'threshold': float(args.threshold),
                'error': None,
            })
            continue

        pred_eid, raw_output, err, scored = _run_a_downstream(
            candidates=candidates,
            sentence=sentence,
            runtime=runtime,
            scorer=scorer,
            threshold=float(args.threshold),
            batch_size=int(args.batch_size),
            logger=logger,
        )
        status = 'ok' if pred_eid else ('below_threshold' if err and err.startswith('group_a_below_threshold') else 'error')
        pred_rows.append({
            'id': rid,
            'sentence': sentence,
            'status': status,
            'candidate_e_ids': [str(c.get('e_id') or '').strip() for c in scored],
            'pred_e_id': pred_eid,
            'top1_e_id': raw_output.get('top1_e_id'),
            'top1_confidence': raw_output.get('top1_confidence'),
            'threshold': raw_output.get('threshold'),
            'error': err,
            'downstream_input_text_a': raw_output.get('downstream_input_text_a'),
            'downstream_input_text_b': raw_output.get('downstream_input_text_b'),
            'downstream_input_version': raw_output.get('downstream_input_version'),
            'downstream_input_text_b_format': raw_output.get('downstream_input_text_b_format'),
        })

    _write_csv(args.output_dir / 'predictions.csv', pred_rows)
    _write_jsonl(args.output_dir / 'predictions.jsonl', pred_rows)
    _write_summary(args.output_dir / 'summary.json', pred_rows, args)
    if pred_rows:
        _write_input_preview(args.output_dir / 'input_preview_first.json', pred_rows[0])
    logger.info('wrote outputs under %s', args.output_dir)


if __name__ == '__main__':
    main()
