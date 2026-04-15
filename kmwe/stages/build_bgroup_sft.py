from __future__ import annotations

import ast
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from kmwe.core.config_loader import ConfigError
from kmwe.core.run_context import RunContext
from kmwe.core.utils import iso_now
from kmwe.stages.infer_step2_rerank import _build_marked_sentence, _extract_target_span_text
from kmwe.utils.jsonio import write_json, write_jsonl_line

_ALLOWED_ROLES = {"pos_conti", "pos_disconti", "neg_target_absent"}
_ALLOWED_SPLITS = {"train", "dev", "test"}
_RENDERER_VERSION = "bgroup_prompt_v2"
_SYSTEM_PROMPT_VERSION = "infer_step2_system_v3"

_SYSTEM_PROMPT_MULTI = '''당신은 한국어 표현문형 의미 선택기다.

표적 구간은 [SPAN] 와 [/SPAN] 사이에 표시된다.
주어진 문맥과 후보 의미를 비교하여, 표적 구간에 동시에 성립하는 후보가 여러 개이면 함께 선택하라.
어떤 후보도 문맥에 맞지 않으면 NONE을 선택하라.

출력:
- 한 줄만 출력한다.
- 문맥에 맞는 후보가 있으면 해당 번호들을 쉼표로만 연결해 쓴다.
- 어떤 후보도 맞지 않으면 NONE만 쓴다.
- 이유나 설명은 쓰지 않는다.
- 후보 목록에 없는 번호는 쓰지 않는다.'''

_SYSTEM_PROMPT_SINGLE = '''당신은 한국어 표현문형 의미 선택기다.

표적 구간은 [SPAN] 와 [/SPAN] 사이에 표시된다.
주어진 문맥과 후보 의미를 비교하여, 표적 구간에 가장 알맞은 후보 하나만 선택하라.
어떤 후보도 문맥에 맞지 않으면 NONE을 선택하라.

출력:
- 문맥에 가장 알맞은 후보가 있으면 해당 번호 하나만 쓴다.
- 어떤 후보도 맞지 않으면 NONE만 쓴다.
- 출력에는 반드시 번호 하나 혹은 NONE만 기록되어야 한다.
- 이유나 설명은 쓰지 않는다.
- 후보 목록에 없는 번호는 쓰지 않는다.'''


def _parse_multi_preserve_order(raw: Any) -> list[str]:
    if raw is None:
        return []
    try:
        if pd.isna(raw):
            return []
    except Exception:
        pass
    s = str(raw).strip()
    if not s:
        return []
    if s.lower() in {"nan", "none", "null"}:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for chunk in s.replace(",", ";").split(";"):
        item = chunk.strip()
        if not item:
            continue
        if item.lower() in {"nan", "none", "null"}:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_span_segments(raw: Any) -> list[tuple[int, int]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        text = str(raw).strip()
        if not text:
            return []
        try:
            items = ast.literal_eval(text)
        except Exception:
            return []
    out: list[tuple[int, int]] = []
    for item in items or []:
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
    return out


def _span_segments_to_jsonable(spans: list[tuple[int, int]]) -> list[list[int]]:
    return [[int(s), int(e)] for s, e in spans]


def _derive_decision_type(gold_e_ids: list[str]) -> str:
    if not gold_e_ids:
        return "none"
    if len(gold_e_ids) == 1:
        return "one"
    return "multi"


def _effective_gold_e_ids(row: dict[str, Any], allow_multiple: bool) -> list[str]:
    if allow_multiple:
        return list(row.get("gold_e_ids") or [])
    forced = list(row.get("gold_e_ids_single_if_forced") or [])
    return forced


def _load_expredict_meta(dict_xlsx: Path, sheet_name: str = "expredict") -> dict[str, dict[str, str]]:
    if not dict_xlsx.exists():
        return {}
    df = pd.read_excel(dict_xlsx, sheet_name=sheet_name, engine="openpyxl")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        eid = str(row.get("e_id") or "").strip()
        if not eid:
            continue
        out[eid] = {
            "canonical_form": str(row.get("canonical_form") or "").strip(),
            "gloss": str(row.get("gloss") or "").strip(),
            "pragmatics": str(row.get("pragmatics") or "").strip(),
            "disambiguation_hint": str(row.get("disambiguation_hint") or "").strip(),
        }
    return out


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = "" if pd.isna(row.get("example_id")) else str(row.get("example_id")).strip()
    out["instance_id"] = "" if pd.isna(row.get("instance_id")) else str(row.get("instance_id")).strip()
    out["target_sentence"] = "" if pd.isna(row.get("target_sentence")) else str(row.get("target_sentence"))
    out["context_left"] = "" if pd.isna(row.get("context_left")) else str(row.get("context_left"))
    out["context_right"] = "" if pd.isna(row.get("context_right")) else str(row.get("context_right"))
    out["gold_example_role"] = "" if pd.isna(row.get("gold_example_role")) else str(row.get("gold_example_role")).strip().lower()
    out["split"] = "" if pd.isna(row.get("split")) else str(row.get("split")).strip().lower()
    out["pattern_type"] = "" if pd.isna(row.get("pattern_type")) else str(row.get("pattern_type")).strip()
    out["source"] = "" if pd.isna(row.get("source")) else str(row.get("source")).strip()
    out["note"] = "" if pd.isna(row.get("note")) else str(row.get("note"))
    anchor_raw = row.get("anchor_eid") if "anchor_eid" in row else row.get("e_id")
    out["anchor_eid"] = "" if pd.isna(anchor_raw) else str(anchor_raw).strip()
    out["candidate_e_ids"] = _parse_multi_preserve_order(row.get("candidate_e_ids"))
    out["gold_e_ids"] = _parse_multi_preserve_order(row.get("gold_e_ids"))
    out["gold_e_ids_single_if_forced"] = _parse_multi_preserve_order(row.get("gold_e_ids_single_if_forced"))
    out["decision_type_raw"] = "" if pd.isna(row.get("decision_type")) else str(row.get("decision_type")).strip().lower()
    out["span_segments_raw"] = "" if pd.isna(row.get("span_segments")) else str(row.get("span_segments"))
    out["span_segments_parsed"] = _parse_span_segments(row.get("span_segments"))
    out["example_key_full"] = f"{out['example_id']}#{out['instance_id']}"
    return out


def _validate_row(row: dict[str, Any], allow_multiple: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in ["example_id", "instance_id", "target_sentence", "gold_example_role", "split"]:
        if not row.get(key):
            errors.append(f"missing_required:{key}")
    if row.get("gold_example_role") and row["gold_example_role"] not in _ALLOWED_ROLES:
        errors.append("invalid_gold_example_role")
    if row.get("split") and row["split"] not in _ALLOWED_SPLITS:
        errors.append("invalid_split")
    candidate_e_ids = row.get("candidate_e_ids") or []
    gold_e_ids = row.get("gold_e_ids") or []
    gold_e_ids_single_if_forced = row.get("gold_e_ids_single_if_forced") or []
    effective_gold_e_ids = _effective_gold_e_ids(row, allow_multiple)
    if not candidate_e_ids:
        errors.append("empty_candidate_e_ids")
    if not row.get("span_segments_parsed"):
        errors.append("invalid_span_segments")
    else:
        sent_len = len(row.get("target_sentence") or "")
        for start, end in row["span_segments_parsed"]:
            if start < 0 or end <= start or end > sent_len:
                errors.append("span_segments_oob")
                break
    cand_set = set(candidate_e_ids)
    if any(eid not in cand_set for eid in gold_e_ids):
        errors.append("gold_not_subset_of_candidates")
    if any(eid not in cand_set for eid in gold_e_ids_single_if_forced):
        errors.append("gold_single_if_forced_not_subset_of_candidates")
    role = row.get("gold_example_role")
    if role == "neg_target_absent" and effective_gold_e_ids:
        errors.append("negative_row_has_gold_e_ids")
    if role in {"pos_conti", "pos_disconti"} and not effective_gold_e_ids:
        errors.append("positive_row_missing_gold_e_ids")
    if (not allow_multiple) and len(effective_gold_e_ids) > 1:
        errors.append("multiple_gold_e_ids_not_allowed")
    original_decision_type = _derive_decision_type(gold_e_ids)
    effective_decision_type = _derive_decision_type(effective_gold_e_ids)
    row["effective_gold_e_ids"] = effective_gold_e_ids
    row["effective_decision_type"] = effective_decision_type
    row["decision_type"] = original_decision_type
    raw_decision_type = row.get("decision_type_raw") or ""
    if not raw_decision_type:
        warnings.append("decision_type_missing_backfilled")
    elif raw_decision_type != original_decision_type:
        errors.append("decision_type_mismatch")
    if len(candidate_e_ids) > 8:
        warnings.append("candidate_count_high")
    return errors, warnings


def _render_decision_line_from_gold_eids(gold_e_ids: list[str], candidate_e_ids: list[str]) -> str:
    if not gold_e_ids:
        return "NONE"
    cand_index = {eid: i for i, eid in enumerate(candidate_e_ids)}
    ordered = sorted(list(dict.fromkeys(gold_e_ids)), key=lambda x: (cand_index.get(x, 10**9), x))
    labels: list[str] = []
    for eid in ordered:
        idx = cand_index.get(eid)
        if idx is None:
            continue
        labels.append(str(idx + 1))
    if not labels:
        return "NONE"
    return ",".join(labels)


def _build_prompt_core(row: dict[str, Any], expredict_meta: dict[str, dict[str, str]], allow_multiple: bool) -> tuple[str, str]:
    target_sentence = str(row.get("target_sentence") or "")
    spans = row.get("span_segments_parsed") or []
    target_sentence_marked = _build_marked_sentence(target_sentence, spans)
    target_span_text = _extract_target_span_text(target_sentence, spans, "")
    lines: list[str] = []
    lines.append(f"문장: {target_sentence_marked}")
    lines.append(f"표적 표현: {target_span_text}")
    lines.append("")
    lines.append("후보 의미:")
    for idx, eid in enumerate(row.get("candidate_e_ids") or [], start=1):
        meta = expredict_meta.get(eid, {})
        lines.append(
            f"{idx}) 대표형={meta.get('canonical_form', '')} | 뜻풀이={meta.get('gloss', '')}"
        )
    lines.append("")
    if allow_multiple:
        lines.append("위 문장의 표적 표현에 동시에 성립하는 후보가 있으면 함께 선택하라.")
        system = _SYSTEM_PROMPT_MULTI
    else:
        lines.append("위 문장의 표적 표현에 가장 알맞은 후보를 선택하라.")
        system = _SYSTEM_PROMPT_SINGLE
    return system, "\n".join(lines)


def _log_sample_examples(logger: logging.Logger, examples_by_split: dict[str, list[dict[str, Any]]]) -> None:
    seen_roles: set[str] = set()
    seen_decisions: set[str] = set()
    logged_none = False
    max_candidate_example: dict[str, Any] | None = None
    for split, examples in examples_by_split.items():
        for ex in examples:
            meta = ex.get("metadata", {}) or {}
            role = str(meta.get("gold_example_role") or "")
            decision = str(meta.get("decision_type") or "")
            cand_count = int(meta.get("candidate_count") or 0)
            if max_candidate_example is None or cand_count > int((max_candidate_example.get("metadata", {}) or {}).get("candidate_count") or 0):
                max_candidate_example = ex
            if role and role not in seen_roles:
                seen_roles.add(role)
                logger.info("[build_bgroup_sft][sample][role=%s] user=%s", role, ex["messages"][1]["content"])
                logger.info("[build_bgroup_sft][sample][role=%s] assistant=%s", role, ex["messages"][2]["content"])
            if decision and decision not in seen_decisions:
                seen_decisions.add(decision)
                logger.info("[build_bgroup_sft][sample][decision_type=%s] assistant=%s", decision, ex["messages"][2]["content"])
            if (not logged_none) and ex["messages"][2]["content"].strip() == "DECISION: NONE":
                logged_none = True
                logger.info("[build_bgroup_sft][sample][none] user=%s", ex["messages"][1]["content"])
        logger.info("[build_bgroup_sft][split=%s] exported_rows=%s", split, len(examples))
    if max_candidate_example is not None:
        meta = max_candidate_example.get("metadata", {}) or {}
        logger.info(
            "[build_bgroup_sft][sample][max_candidate_count=%s] key=%s assistant=%s",
            meta.get("candidate_count"),
            meta.get("example_key_full"),
            max_candidate_example["messages"][2]["content"],
        )


def _maybe_log_wandb(cfg: dict[str, Any], run_context: RunContext, report: dict[str, Any], sample_rows: list[dict[str, Any]]) -> None:
    logger = logging.getLogger("kmwe")
    wandb_cfg = cfg.get("wandb", {}) or {}
    build_cfg = cfg.get("build_bgroup_sft", {}) or {}
    wandb_cfg = {**wandb_cfg, **(build_cfg.get("wandb", {}) or {})}
    if not bool(wandb_cfg.get("enabled", False)):
        logger.info("[build_bgroup_sft][wandb] disabled")
        return
    try:
        import wandb  # type: ignore
    except Exception as exc:
        logger.warning("[build_bgroup_sft][wandb] import failed: %s", exc)
        return
    try:
        run = wandb.init(
            project=str(wandb_cfg.get("project") or "kmwe-bgroup-sft"),
            entity=str(wandb_cfg.get("entity") or "") or None,
            group=str(wandb_cfg.get("group") or "") or f"{cfg.get('exp', {}).get('exp_id', 'default')}:build_bgroup_sft",
            name=str(wandb_cfg.get("name") or "") or f"build_bgroup_sft/{cfg.get('exp', {}).get('exp_id', 'default')}/{run_context.run_id}",
            mode=str(wandb_cfg.get("mode") or "online"),
            tags=["build_bgroup_sft", "bgroup_sft"],
            config={
                "gold_xlsx": report.get("gold_xlsx"),
                "n_input_rows": report.get("n_input_rows"),
                "n_valid_rows": report.get("n_valid_rows"),
                "n_error_rows": report.get("n_error_rows"),
                "n_warning_rows": report.get("n_warning_rows"),
            },
            reinit=True,
        )
        log_payload = {
            "build_bgroup_sft/n_input_rows": report.get("n_input_rows", 0),
            "build_bgroup_sft/n_valid_rows": report.get("n_valid_rows", 0),
            "build_bgroup_sft/n_error_rows": report.get("n_error_rows", 0),
            "build_bgroup_sft/n_warning_rows": report.get("n_warning_rows", 0),
        }
        for key, value in (report.get("split_counts") or {}).items():
            log_payload[f"build_bgroup_sft/split_count/{key}"] = value
        for key, value in (report.get("role_counts") or {}).items():
            log_payload[f"build_bgroup_sft/role_count/{key}"] = value
        for key, value in (report.get("decision_type_counts") or {}).items():
            log_payload[f"build_bgroup_sft/decision_type_count/{key}"] = value
        wandb.log(log_payload)
        if sample_rows:
            table = wandb.Table(columns=["example_key_full", "gold_example_role", "decision_type", "candidate_count", "assistant_target"])
            for row in sample_rows[:20]:
                table.add_data(row.get("example_key_full"), row.get("gold_example_role"), row.get("decision_type"), row.get("candidate_count"), row.get("assistant_target"))
            wandb.log({"build_bgroup_sft/sample_rows": table})
        run.finish()
        logger.info("[build_bgroup_sft][wandb] logged summary ok")
    except Exception as exc:
        logger.warning("[build_bgroup_sft][wandb] init/log failed: %s", exc)


def run_build_bgroup_sft(*, cfg: dict[str, Any], run_context: RunContext) -> None:
    logger = logging.getLogger("kmwe")
    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for_users_dir = outputs_dir / "for_users"
    for_users_dir.mkdir(parents=True, exist_ok=True)

    build_cfg = cfg.get("build_bgroup_sft", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    gold_xlsx = Path(str(paths_cfg.get("gold_b_xlsx") or paths_cfg.get("gold_xlsx") or "")).expanduser()
    if not gold_xlsx.exists():
        raise ConfigError(f"build_bgroup_sft gold.xlsx 경로가 유효하지 않습니다: {gold_xlsx}")
    dict_xlsx = Path(str(paths_cfg.get("dict_xlsx") or paths_cfg.get("expredict_xlsx") or "")).expanduser()
    if not dict_xlsx.exists():
        raise ConfigError(f"build_bgroup_sft dict_xlsx 경로가 유효하지 않습니다: {dict_xlsx}")

    gold_sheet_name = str(build_cfg.get("gold_sheet_name") or "gold")
    allow_multiple = bool(build_cfg.get("allow_multiple", cfg.get("llm_rerank", {}).get("transduction", {}).get("allow_multiple", True)))

    logger.info("[build_bgroup_sft] gold_xlsx=%s sheet=%s", gold_xlsx, gold_sheet_name)
    logger.info("[build_bgroup_sft] dict_xlsx=%s", dict_xlsx)
    logger.info("[build_bgroup_sft] allow_multiple=%s", allow_multiple)

    df = pd.read_excel(gold_xlsx, sheet_name=gold_sheet_name, engine="openpyxl")
    rows = [_normalize_row(r) for r in df.to_dict(orient="records")]
    logger.info("[build_bgroup_sft] n_input_rows=%s", len(rows))
    logger.info("[build_bgroup_sft] role_counts_raw=%s", dict(Counter(r.get("gold_example_role") or "" for r in rows)))
    logger.info("[build_bgroup_sft] split_counts_raw=%s", dict(Counter(r.get("split") or "" for r in rows)))

    expredict_meta = _load_expredict_meta(dict_xlsx)

    valid_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    role_counts = Counter()
    split_counts = Counter()
    decision_type_counts = Counter()
    candidate_count_distribution = Counter()

    for row in rows:
        errors, warnings = _validate_row(row, allow_multiple=allow_multiple)
        if warnings:
            warning_rows.append({"example_key_full": row["example_key_full"], "warnings": warnings})
        if errors:
            error_rows.append({"example_key_full": row["example_key_full"], "errors": errors})
            issue_rows.append({"example_key_full": row["example_key_full"], "level": "error", "messages": errors})
            continue
        valid_rows.append(row)
        role_counts[row["gold_example_role"]] += 1
        split_counts[row["split"]] += 1
        decision_type_counts[row["decision_type"]] += 1
        candidate_count_distribution[len(row["candidate_e_ids"])] += 1
        if warnings:
            issue_rows.append({"example_key_full": row["example_key_full"], "level": "warning", "messages": warnings})

    logger.info("[build_bgroup_sft] n_valid_rows=%s n_error_rows=%s n_warning_rows=%s", len(valid_rows), len(error_rows), len(warning_rows))
    for item in error_rows[:5]:
        logger.warning("[build_bgroup_sft][error_sample] key=%s errors=%s", item["example_key_full"], item["errors"])

    examples_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    sample_rows_for_wandb: list[dict[str, Any]] = []
    for row in valid_rows:
        system_prompt, user_prompt = _build_prompt_core(row, expredict_meta, allow_multiple)
        assistant_target = _render_decision_line_from_gold_eids(row["effective_gold_e_ids"], row["candidate_e_ids"])
        metadata = {
            "example_id": row["example_id"],
            "instance_id": row["instance_id"],
            "example_key_full": row["example_key_full"],
            "anchor_eid": row["anchor_eid"],
            "gold_example_role": row["gold_example_role"],
            "split": row["split"],
            "source": row["source"],
            "pattern_type": row["pattern_type"],
            "candidate_e_ids": row["candidate_e_ids"],
            "candidate_number_to_eid": {str(i + 1): eid for i, eid in enumerate(row["candidate_e_ids"])},
            "gold_e_ids": row["gold_e_ids"],
            "effective_gold_e_ids": row["effective_gold_e_ids"],
            "decision_type": row["decision_type"],
            "effective_decision_type": row.get("effective_decision_type") or row["decision_type"],
            "candidate_count": len(row["candidate_e_ids"]),
            "span_segments": _span_segments_to_jsonable(row["span_segments_parsed"]),
            "target_sentence": row["target_sentence"],
            "is_discontinuous": row["gold_example_role"] == "pos_disconti",
            "renderer_version": _RENDERER_VERSION,
            "system_prompt_version": _SYSTEM_PROMPT_VERSION,
            "span_bundle_key": row.get("span_bundle_key", ""),
            "source_run_id": row.get("source_run_id", ""),
            "match_key": row.get("match_key", ""),
            "gold_e_ids_single_if_forced": row.get("gold_e_ids_single_if_forced") or [],
        }
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_target},
            ],
            "metadata": metadata,
        }
        if row["split"] not in examples_by_split:
            raise ConfigError(f"build_bgroup_sft split 값이 올바르지 않습니다: {row['split']}")
        examples_by_split[row["split"]].append(example)
        sample_rows_for_wandb.append(
            {
                "example_key_full": row["example_key_full"],
                "gold_example_role": row["gold_example_role"],
                "decision_type": row["decision_type"],
                "candidate_count": len(row["candidate_e_ids"]),
                "assistant_target": assistant_target,
            }
        )

    _log_sample_examples(logger, examples_by_split)

    out_train = outputs_dir / "train.jsonl"
    out_dev = outputs_dir / "dev.jsonl"
    out_test = outputs_dir / "test.jsonl"
    for path, examples in [(out_train, examples_by_split["train"]), (out_dev, examples_by_split["dev"]), (out_test, examples_by_split["test"] )]:
        with path.open("w", encoding="utf-8") as f:
            for ex in examples:
                write_jsonl_line(f, ex)

    summary_path = for_users_dir / "build_bgroup_sft_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "split",
            "gold_example_role",
            "n_rows",
            "n_one",
            "n_multi",
            "n_none",
            "avg_candidate_count",
            "min_candidate_count",
            "max_candidate_count",
            "n_validation_errors",
            "n_validation_warnings",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ["train", "dev", "test"]:
            split_rows = [r for r in valid_rows if r["split"] == split]
            role_groups = Counter(r["gold_example_role"] for r in split_rows)
            for role, n_rows in sorted(role_groups.items()):
                role_rows = [r for r in split_rows if r["gold_example_role"] == role]
                cand_counts = [len(r["candidate_e_ids"]) for r in role_rows] or [0]
                dt_counts = Counter(r["decision_type"] for r in role_rows)
                writer.writerow(
                    {
                        "split": split,
                        "gold_example_role": role,
                        "n_rows": n_rows,
                        "n_one": dt_counts.get("one", 0),
                        "n_multi": dt_counts.get("multi", 0),
                        "n_none": dt_counts.get("none", 0),
                        "avg_candidate_count": round(sum(cand_counts) / len(cand_counts), 4) if cand_counts else 0,
                        "min_candidate_count": min(cand_counts) if cand_counts else 0,
                        "max_candidate_count": max(cand_counts) if cand_counts else 0,
                        "n_validation_errors": 0,
                        "n_validation_warnings": sum(1 for wr in warning_rows if any(vr["example_key_full"] == wr["example_key_full"] for vr in role_rows)),
                    }
                )

    issues_path = outputs_dir / "validation_issues.json"
    write_json(issues_path, {"errors": error_rows, "warnings": warning_rows, "issues": issue_rows}, indent=2)

    report = {
        "schema_version": "build_bgroup_sft_v1",
        "created_at": iso_now(),
        "gold_xlsx": str(gold_xlsx),
        "gold_sheet_name": gold_sheet_name,
        "dict_xlsx": str(dict_xlsx),
        "allow_multiple": allow_multiple,
        "n_input_rows": len(rows),
        "n_valid_rows": len(valid_rows),
        "n_error_rows": len(error_rows),
        "n_warning_rows": len(warning_rows),
        "split_counts": dict(split_counts),
        "role_counts": dict(role_counts),
        "decision_type_counts": dict(decision_type_counts),
        "candidate_count_distribution": {str(k): v for k, v in sorted(candidate_count_distribution.items())},
        "outputs": {
            "train_jsonl": str(out_train),
            "dev_jsonl": str(out_dev),
            "test_jsonl": str(out_test),
            "summary_csv": str(summary_path),
            "validation_issues_json": str(issues_path),
        },
    }
    report_path = outputs_dir / "build_bgroup_sft_report.json"
    write_json(report_path, report, indent=2)
    logger.info("[build_bgroup_sft] wrote train=%s dev=%s test=%s", out_train, out_dev, out_test)
    logger.info("[build_bgroup_sft] wrote summary=%s report=%s", summary_path, report_path)

    _maybe_log_wandb(cfg, run_context, report, sample_rows_for_wandb)
