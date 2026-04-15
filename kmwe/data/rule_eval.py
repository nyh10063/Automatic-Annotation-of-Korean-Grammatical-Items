from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


CORE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sentence": ("sentence", "target_sentence", "문장"),
    "gold_e_id": ("gold_e_id", "e_id", "gold_eid", "anchor_eid"),
    "example_id": ("example_id", "id"),
}
OPTIONAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "instance_id": ("instance_id",),
    "gold_example_role": ("gold_example_role", "example_role", "role"),
    "split": ("split",),
    "span_segments": ("span_segments",),
    "source": ("source",),
    "note": ("note",),
    "gold_e_ids_single_if_forced": ("gold_e_ids_single_if_forced",),
    "gold_e_ids": ("gold_e_ids",),
    "decision_type": ("decision_type",),
}


@dataclass
class RuleEvalConfig:
    gold_path: str
    dict_path: str
    gold_sheet_name: str = "gold"
    with_downstream: str | None = None


@dataclass
class RuleEvalInstance:
    example_key: str
    sentence: str
    gold_e_id: str
    anchor_eid: str | None = None
    gold_group: str | None = None
    gold_polyset_id: str | None = None
    gold_example_role: str | None = None
    gold_span_segments: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleEvalPrediction:
    example_key: str
    sentence: str
    gold_e_id: str
    gold_group: str | None
    gold_polyset_id: str | None
    gold_example_role: str | None
    candidate_e_ids: list[str]
    candidate_count: int
    gold_in_candidates: bool
    matched_rule_ids: list[str]
    pred_e_id: str | None
    status: str
    error_reason: str | None
    miss_stage: str
    meta: dict[str, Any] = field(default_factory=dict)


def validate_input_paths(cfg: RuleEvalConfig) -> dict[str, str]:
    gold_path = Path(cfg.gold_path)
    dict_path = Path(cfg.dict_path)
    errors: dict[str, str] = {}
    if not gold_path.exists():
        errors["gold_path"] = f"missing: {gold_path}"
    if not dict_path.exists():
        errors["dict_path"] = f"missing: {dict_path}"
    return errors


def load_gold_frame(gold_path: str, sheet_name: str = "gold") -> pd.DataFrame:
    return pd.read_excel(Path(gold_path), sheet_name=sheet_name, dtype=object, engine="openpyxl")


def resolve_columns(columns: list[str]) -> tuple[dict[str, str], dict[str, str | None]]:
    normalized = {str(c).strip(): str(c) for c in columns}
    core: dict[str, str] = {}
    missing: list[str] = []
    for key, aliases in CORE_COLUMN_ALIASES.items():
        match = next((normalized[a] for a in aliases if a in normalized), None)
        if match is None:
            missing.append(key)
        else:
            core[key] = match
    if missing:
        raise ValueError(f"missing required core columns: {', '.join(missing)}")
    optional: dict[str, str | None] = {}
    for key, aliases in OPTIONAL_COLUMN_ALIASES.items():
        optional[key] = next((normalized[a] for a in aliases if a in normalized), None)
    return core, optional


def load_gold_instances(
    gold_path: str,
    *,
    sheet_name: str,
    expredict_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[RuleEvalInstance], dict[str, Any]]:
    frame = load_gold_frame(gold_path, sheet_name=sheet_name)
    core_cols, optional_cols = resolve_columns(list(frame.columns))
    instances: list[RuleEvalInstance] = []
    missing_optional: dict[str, int] = {k: 0 for k, v in optional_cols.items() if v is None}
    for _, raw_row in frame.iterrows():
        row = {str(col): _normalize_value(raw_row.get(col)) for col in frame.columns}
        instance = make_instance_from_row(row, core_cols, optional_cols, expredict_map or {})
        instances.append(instance)
    return instances, {
        "n_rows": int(len(frame)),
        "core_columns": core_cols,
        "optional_columns": optional_cols,
        "missing_optional_columns": sorted(missing_optional.keys()),
    }


def build_example_key(example_id: str, instance_id: str | None = None) -> str:
    if instance_id and instance_id != example_id:
        return f"{example_id}#{instance_id}"
    return example_id


def make_instance_from_row(
    row: dict[str, Any],
    core_columns: dict[str, str],
    optional_columns: dict[str, str | None],
    expredict_map: dict[str, dict[str, Any]],
) -> RuleEvalInstance:
    example_id = _required_text(row, core_columns["example_id"], "example_id")
    sentence = _required_text(row, core_columns["sentence"], "sentence")
    anchor_eid = _required_text(row, core_columns["gold_e_id"], "gold_e_id")
    instance_id = _optional_text(row, optional_columns.get("instance_id"))
    role = (_optional_text(row, optional_columns.get("gold_example_role")) or "").strip().lower() or None
    split = _optional_text(row, optional_columns.get("split"))
    span_segments = row.get(optional_columns.get("span_segments")) if optional_columns.get("span_segments") else None
    source = _optional_text(row, optional_columns.get("source"))
    note = _optional_text(row, optional_columns.get("note"))
    forced_single_gold = _optional_text(row, optional_columns.get("gold_e_ids_single_if_forced"))
    gold_e_ids_raw = _optional_text(row, optional_columns.get("gold_e_ids"))
    decision_type = _optional_text(row, optional_columns.get("decision_type"))
    if (role or "").startswith("neg"):
        gold_e_id = "__NONE__"
        gold_source = "negative_role"
    elif forced_single_gold:
        gold_e_id = forced_single_gold
        gold_source = "gold_e_ids_single_if_forced"
    else:
        gold_e_id = anchor_eid
        gold_source = core_columns["gold_e_id"]
    meta_row = expredict_map.get(gold_e_id, {}) or expredict_map.get(anchor_eid, {})
    return RuleEvalInstance(
        example_key=build_example_key(example_id, instance_id),
        sentence=sentence,
        gold_e_id=gold_e_id,
        anchor_eid=anchor_eid,
        gold_group=_optional_meta(meta_row, "group"),
        gold_polyset_id=_optional_meta(meta_row, "polyset_id"),
        gold_example_role=role,
        gold_span_segments=span_segments,
        meta={
            "example_id": example_id,
            "instance_id": instance_id,
            "split": split,
            "source": source,
            "note": note,
            "gold_source": gold_source,
            "gold_e_ids_single_if_forced": forced_single_gold,
            "gold_e_ids": gold_e_ids_raw,
            "decision_type": decision_type,
        },
    )


def dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_prediction(
    instance: RuleEvalInstance,
    *,
    candidates: list[dict[str, Any]],
    miss_stage: str = "unknown",
) -> RuleEvalPrediction:
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
    pred_e_id = _pick_top_candidate_eid(candidates)
    if instance.gold_e_id == "__NONE__":
        status = "negative_candidate" if candidate_e_ids else "negative_clear"
        error_reason = None if not candidate_e_ids else "negative_generated_candidates"
    elif gold_in_candidates:
        status = "gold_detected"
        error_reason = None
    elif not candidate_e_ids:
        status = "gold_missed_no_candidate"
        error_reason = "no_candidate"
        miss_stage = "detect"
    else:
        status = "gold_missed_wrong_candidates"
        error_reason = "gold_not_in_candidates"
    meta = dict(instance.meta)
    if instance.anchor_eid is not None:
        meta["anchor_eid"] = instance.anchor_eid
    return RuleEvalPrediction(
        example_key=instance.example_key,
        sentence=instance.sentence,
        gold_e_id=instance.gold_e_id,
        gold_group=instance.gold_group,
        gold_polyset_id=instance.gold_polyset_id,
        gold_example_role=instance.gold_example_role,
        candidate_e_ids=candidate_e_ids,
        candidate_count=len(candidate_e_ids),
        gold_in_candidates=gold_in_candidates,
        matched_rule_ids=matched_rule_ids,
        pred_e_id=pred_e_id,
        status=status,
        error_reason=error_reason,
        miss_stage=miss_stage,
        meta=meta,
    )


def compute_coverage_metrics(predictions: list[RuleEvalPrediction]) -> dict[str, Any]:
    positive = [p for p in predictions if p.gold_e_id and p.gold_e_id != "__NONE__"]
    negative = [p for p in predictions if p.gold_e_id == "__NONE__"]
    pos_hits = sum(1 for p in positive if p.gold_in_candidates)
    pos_no_candidate = sum(1 for p in positive if p.candidate_count == 0)
    neg_with_candidate = sum(1 for p in negative if p.candidate_count > 0)
    return {
        "n_examples": len(predictions),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "positive_candidate_recall": _safe_div(pos_hits, len(positive)),
        "gold_in_candidate_rate": _safe_div(pos_hits, len(positive)),
        "positive_no_candidate_rate": _safe_div(pos_no_candidate, len(positive)),
        "negative_candidate_rate": _safe_div(neg_with_candidate, len(negative)),
        "avg_candidate_count_on_positive": _avg([p.candidate_count for p in positive]),
        "avg_candidate_count_on_negative": _avg([p.candidate_count for p in negative]),
    }


def compute_strict_metrics(predictions: list[RuleEvalPrediction]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for pred in predictions:
        gold = pred.gold_e_id or "__NONE__"
        guess = pred.pred_e_id or "__NONE__"
        gold_pos = gold != "__NONE__"
        pred_pos = guess != "__NONE__"
        if gold_pos and pred_pos and gold == guess:
            tp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        elif gold_pos and pred_pos and gold != guess:
            fp += 1
            fn += 1
        elif not gold_pos and pred_pos:
            fp += 1
        else:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision or recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_coverage_by_field(predictions: list[RuleEvalPrediction], field: str) -> dict[str, Any]:
    buckets: dict[str, list[RuleEvalPrediction]] = {}
    for pred in predictions:
        value = getattr(pred, field, None)
        key = str(value or "")
        buckets.setdefault(key, []).append(pred)
    out: dict[str, Any] = {}
    for key, items in buckets.items():
        out[key] = compute_coverage_metrics(items)
    return out


def prediction_to_row(prediction: RuleEvalPrediction) -> dict[str, Any]:
    return asdict(prediction)


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _required_text(row: dict[str, Any], key: str, logical_name: str) -> str:
    value = _optional_text(row, key)
    if not value:
        raise ValueError(f"{logical_name} is empty")
    return value


def _optional_text(row: dict[str, Any], key: str | None) -> str | None:
    if not key:
        return None
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_meta(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return value


def _pick_top_candidate_eid(candidates: list[dict[str, Any]]) -> str | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda c: (
            -float(c.get("score", 0) or 0),
            str(c.get("e_id") or ""),
        ),
    )
    eid = str(ordered[0].get("e_id") or "").strip()
    return eid or None


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _avg(values: list[int]) -> float:
    return _safe_div(sum(values), len(values))
