from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import shutil
import torch.nn.functional as F
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kmwe.core.config_loader import ConfigError
from kmwe.core.run_context import RunContext
from kmwe.core.utils import iso_now
from kmwe.stages.infer_step2_rerank import _extract_decision_line, _parse_decision_line
from kmwe.utils.jsonio import write_json


class _TrainLlmSftOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "[train_llm_sft]" in record.getMessage()


@dataclass
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    metadata: dict[str, Any]
    prompt_text: str
    full_text: str
    assistant_target: str


def load_sft_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser()
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ConfigError(f"JSONL 파싱 실패: {p}:{line_no} - {exc}") from exc
            if not isinstance(obj, dict):
                raise ConfigError(f"JSONL row는 dict여야 합니다: {p}:{line_no}")
            rows.append(obj)
    return rows


def _resolve_sft_paths(cfg: dict[str, Any]) -> tuple[Path, Path | None, Path | None]:
    sft_cfg = cfg.get("llm_sft", {}) or {}
    input_dir_raw = str(sft_cfg.get("input_dir") or "").strip()
    train_raw = str(sft_cfg.get("train_jsonl") or "").strip()
    dev_raw = str(sft_cfg.get("dev_jsonl") or "").strip()
    test_raw = str(sft_cfg.get("test_jsonl") or "").strip()

    if input_dir_raw:
        input_dir = Path(input_dir_raw).expanduser()
        train_path = input_dir / "train.jsonl"
        dev_path = input_dir / "dev.jsonl"
        test_path = input_dir / "test.jsonl"
    else:
        if not train_raw:
            raise ConfigError("llm_sft.input_dir 또는 llm_sft.train_jsonl 중 하나는 필수입니다.")
        train_path = Path(train_raw).expanduser()
        dev_path = Path(dev_raw).expanduser() if dev_raw else None
        test_path = Path(test_raw).expanduser() if test_raw else None

    if not train_path.exists():
        raise ConfigError(f"llm_sft train.jsonl 경로가 유효하지 않습니다: {train_path}")
    if dev_path is not None and not dev_path.exists():
        raise ConfigError(f"llm_sft dev.jsonl 경로가 유효하지 않습니다: {dev_path}")
    if test_path is not None and not test_path.exists():
        raise ConfigError(f"llm_sft test.jsonl 경로가 유효하지 않습니다: {test_path}")
    return train_path, dev_path, test_path


def _resolve_model_name(cfg: dict[str, Any]) -> str:
    sft_cfg = cfg.get("llm_sft", {}) or {}
    model_name = str(sft_cfg.get("model_name_or_path") or "").strip()
    if not model_name:
        raise ConfigError("llm_sft.model_name_or_path는 필수입니다.")
    return model_name


def _resolve_backend(cfg: dict[str, Any]) -> str:
    sft_cfg = cfg.get("llm_sft", {}) or {}
    backend = str(sft_cfg.get("backend") or "hf").strip().lower()
    if backend not in {"hf", "openai"}:
        raise ConfigError(f"llm_sft.backend는 hf 또는 openai 여야 합니다. got={backend}")
    return backend


def _resolve_allow_multiple(cfg: dict[str, Any]) -> bool:
    sft_cfg = cfg.get("llm_sft", {}) or {}
    if "allow_multiple" in sft_cfg:
        return bool(sft_cfg.get("allow_multiple"))
    rerank_cfg = cfg.get("llm_rerank", {}) or {}
    transduction_cfg = rerank_cfg.get("transduction", {}) or {}
    return bool(transduction_cfg.get("allow_multiple", True))


def _render_chat_messages(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ConfigError("tokenizer.apply_chat_template()를 지원하지 않는 tokenizer입니다.")
    model_name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    template_kwargs: dict[str, Any] = {}
    rendered_messages = [dict(m) for m in messages]
    if "qwen3" in model_name:
        # Keep the official Qwen template, but request non-thinking mode when the stack supports it.
        template_kwargs["enable_thinking"] = False
    try:
        rendered = tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
    except Exception as exc:
        raise ConfigError(f"chat template 직렬화 실패: {exc}") from exc
    if not isinstance(rendered, str) or not rendered:
        raise ConfigError("chat template 직렬화 결과가 비어 있습니다.")
    if "qwen3" in model_name:
        # Some Qwen3 stacks still materialize a think block even with enable_thinking=False.
        rendered = re.sub(r"<think>\s*</think>\s*", "", rendered, count=1, flags=re.DOTALL)
        rendered = rendered.replace("<think>\n", "").replace("\n</think>", "")
    return rendered


def _compose_prompt_and_target(tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[str, str, str]:
    if len(messages) != 3:
        raise ConfigError(f"SFT example messages 길이는 3이어야 합니다. got={len(messages)}")
    roles = [str(m.get("role") or "") for m in messages]
    if roles != ["system", "user", "assistant"]:
        raise ConfigError(f"SFT example role 순서가 올바르지 않습니다. got={roles}")
    assistant = str(messages[2].get("content") or "")
    prompt_text = _render_chat_messages(tokenizer, messages[:2], add_generation_prompt=True)
    full_text = _render_chat_messages(tokenizer, messages, add_generation_prompt=False)
    return prompt_text, full_text, assistant


def _encode_examples(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    logger: logging.Logger,
    split_name: str,
) -> list[EncodedExample]:
    encoded: list[EncodedExample] = []
    truncated = 0
    role_counts = Counter()
    decision_counts = Counter()

    for row in rows:
        messages = row.get("messages") or []
        metadata = row.get("metadata") or {}
        prompt_text, full_text, assistant_target = _compose_prompt_and_target(tokenizer, messages)

        full_ids = tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )
        if len(full_ids) >= max_seq_len:
            truncated += 1

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        if len(labels) < len(full_ids):
            labels.extend([-100] * (len(full_ids) - len(labels)))
        labels = labels[: len(full_ids)]

        encoded.append(
            EncodedExample(
                input_ids=full_ids,
                attention_mask=[1] * len(full_ids),
                labels=labels,
                metadata=metadata,
                prompt_text=prompt_text,
                full_text=full_text,
                assistant_target=assistant_target,
            )
        )
        role_counts[str(metadata.get("gold_example_role") or "")] += 1
        decision_counts[str(metadata.get("decision_type") or "")] += 1

    logger.info(
        "[train_llm_sft][encode][%s] n_rows=%d truncated=%d role_counts=%s decision_counts=%s",
        split_name,
        len(encoded),
        truncated,
        dict(role_counts),
        dict(decision_counts),
    )
    return encoded


def _pad_batch(examples: list[EncodedExample], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(ex.input_ids) for ex in examples)
    input_ids = []
    attention_mask = []
    labels = []
    metadata = []
    for ex in examples:
        pad_len = max_len - len(ex.input_ids)
        input_ids.append(ex.input_ids + [pad_token_id] * pad_len)
        attention_mask.append(ex.attention_mask + [0] * pad_len)
        labels.append(ex.labels + [-100] * pad_len)
        metadata.append(ex.metadata)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": metadata,
    }


def _iter_batches(rows: list[EncodedExample], batch_size: int, shuffle: bool, seed: int):
    indices = list(range(len(rows)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield [rows[i] for i in batch_idx]


def _compute_per_example_loss(logits: Any, labels: Any) -> list[float]:
    import torch

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = loss_flat.view(shift_labels.size())
    valid_mask = shift_labels.ne(-100)
    per_example: list[float] = []
    for i in range(shift_labels.size(0)):
        valid = valid_mask[i]
        n_valid = int(valid.sum().item())
        if n_valid <= 0:
            per_example.append(0.0)
            continue
        ex_loss = float(token_loss[i][valid].mean().item())
        per_example.append(ex_loss)
    return per_example


def _build_role_probe_examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    probes: dict[str, dict[str, Any]] = {}
    for row in rows:
        md = row.get("metadata") or {}
        role = str(md.get("gold_example_role") or "").strip()
        if role and role not in probes:
            probes[role] = row
    return probes


def _log_role_probe_predictions(
    *,
    model: Any,
    tokenizer: Any,
    probe_examples_by_role: dict[str, dict[str, Any]],
    gen_cfg: dict[str, Any],
    logger: logging.Logger,
    allow_multiple: bool,
    prefix: str,
) -> None:
    if not probe_examples_by_role:
        return
    ordered = [probe_examples_by_role[k] for k in sorted(probe_examples_by_role)]
    infer_examples = build_dev_infer_examples(ordered)
    raw_pred_rows = generate_dev_predictions(
        model=model,
        tokenizer=tokenizer,
        infer_examples=infer_examples,
        gen_cfg=gen_cfg,
    )
    parsed_pred_rows = parse_dev_predictions(raw_pred_rows, allow_multiple=allow_multiple)
    for gold_row, raw_row, parsed_row in zip(ordered, raw_pred_rows, parsed_pred_rows):
        md = gold_row.get("metadata") or {}
        role = str(md.get("gold_example_role") or "")
        gold_answer = str((gold_row.get("messages") or [{}, {}, {"content": ""}])[2].get("content") or "")
        logger.info(
            "[train_llm_sft][%s][role=%s] gold=%s pred=%s status=%s error=%s raw=%s",
            prefix,
            role,
            gold_answer,
            parsed_row.get("decision_line") or "",
            parsed_row.get("status"),
            parsed_row.get("error_type"),
            str(raw_row.get("raw_text") or "")[:300],
        )


def _log_dataset_overview(logger: logging.Logger, split_name: str, rows: list[dict[str, Any]]) -> None:
    role_counts = Counter()
    decision_counts = Counter()
    candidate_counts = []
    sample_by_role: dict[str, dict[str, Any]] = {}
    sample_by_decision: dict[str, dict[str, Any]] = {}
    none_sample: dict[str, Any] | None = None
    max_candidate_sample: dict[str, Any] | None = None

    for row in rows:
        md = row.get("metadata") or {}
        role = str(md.get("gold_example_role") or "")
        decision = str(md.get("decision_type") or "")
        cand_count = int(md.get("candidate_count") or len(md.get("candidate_e_ids") or []))
        role_counts[role] += 1
        decision_counts[decision] += 1
        candidate_counts.append(cand_count)
        if role and role not in sample_by_role:
            sample_by_role[role] = row
        if decision and decision not in sample_by_decision:
            sample_by_decision[decision] = row
        if none_sample is None and str(row.get("messages", [{}, {}, {"content": ""}])[2].get("content") or "").strip() in {"DECISION: NONE", "NONE"}:
            none_sample = row
        if max_candidate_sample is None or cand_count > int((max_candidate_sample.get("metadata") or {}).get("candidate_count") or 0):
            max_candidate_sample = row

    logger.info(
        "[train_llm_sft][dataset][%s] rows=%d role_counts=%s decision_counts=%s candidate_count(min/avg/max)=%s/%.3f/%s",
        split_name,
        len(rows),
        dict(role_counts),
        dict(decision_counts),
        min(candidate_counts) if candidate_counts else 0,
        (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
        max(candidate_counts) if candidate_counts else 0,
    )

    for role, row in sample_by_role.items():
        logger.info(
            "[train_llm_sft][sample][%s][role=%s] target=%s",
            split_name,
            role,
            str((row.get("messages") or [{}, {"content": ""}])[1].get("content") or "")[:500],
        )
        logger.info(
            "[train_llm_sft][sample][%s][role=%s] assistant=%s",
            split_name,
            role,
            str((row.get("messages") or [{}, {}, {"content": ""}])[2].get("content") or ""),
        )
    for decision, row in sample_by_decision.items():
        logger.info(
            "[train_llm_sft][sample][%s][decision=%s] assistant=%s",
            split_name,
            decision,
            str((row.get("messages") or [{}, {}, {"content": ""}])[2].get("content") or ""),
        )
    if none_sample is not None:
        logger.info(
            "[train_llm_sft][sample][%s][none] target=%s",
            split_name,
            str((none_sample.get("messages") or [{}, {"content": ""}])[1].get("content") or "")[:500],
        )
    if max_candidate_sample is not None:
        md = max_candidate_sample.get("metadata") or {}
        logger.info(
            "[train_llm_sft][sample][%s][max_candidate_count=%s] key=%s assistant=%s",
            split_name,
            md.get("candidate_count"),
            md.get("example_key_full"),
            str((max_candidate_sample.get("messages") or [{}, {}, {"content": ""}])[2].get("content") or ""),
        )


def build_dev_infer_examples(dev_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    infer_examples: list[dict[str, Any]] = []
    for ex in dev_examples:
        messages = ex.get("messages") or []
        if len(messages) != 3:
            raise ConfigError(f"dev example messages 길이는 3이어야 합니다. got={len(messages)}")
        infer_examples.append(
            {
                "messages": [
                    {"role": str(messages[0].get("role") or "system"), "content": str(messages[0].get("content") or "")},
                    {"role": str(messages[1].get("role") or "user"), "content": str(messages[1].get("content") or "")},
                ],
                "metadata": dict(ex.get("metadata") or {}),
                "gold_assistant": str(messages[2].get("content") or ""),
            }
        )
    return infer_examples


def generate_dev_predictions(
    model: Any,
    tokenizer: Any,
    infer_examples: list[dict[str, Any]],
    gen_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch

    device = next(model.parameters()).device
    max_input_len = int(gen_cfg.get("max_input_len", gen_cfg.get("max_seq_len", 2048)))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 16))
    do_sample = bool(gen_cfg.get("do_sample", False))
    temperature = float(gen_cfg.get("temperature", 1.0))
    top_p = float(gen_cfg.get("top_p", 1.0))

    pred_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for ex in infer_examples:
            messages = ex.get("messages") or []
            if len(messages) != 2:
                raise ConfigError(f"dev infer example messages 길이는 2이어야 합니다. got={len(messages)}")
            prompt_text = _render_chat_messages(tokenizer, messages, add_generation_prompt=True)
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
                add_special_tokens=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            output_ids = model.generate(**enc, **gen_kwargs)
            prompt_len = int(enc["input_ids"].shape[1])
            new_ids = output_ids[0][prompt_len:]
            raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            pred_rows.append(
                {
                    "raw_text": raw_text,
                    "prompt_text": prompt_text,
                    "metadata": dict(ex.get("metadata") or {}),
                }
            )
    model.train()
    return pred_rows


def parse_decision_line(raw_text: str, candidate_e_ids: list[str], allow_multiple: bool) -> dict[str, Any]:
    raw_text = raw_text if isinstance(raw_text, str) else ""
    decision_line = _extract_decision_line(raw_text, allow_multiple=allow_multiple)
    if not decision_line:
        return {
            "status": "parse_failure",
            "pred_e_ids": [],
            "raw_text": raw_text,
            "decision_line": "",
            "error_type": "missing_decision_line",
        }
    parsed = _parse_decision_line(decision_line, allow_multiple=allow_multiple, candidate_eids=candidate_e_ids)
    if not parsed.get("protocol_ok"):
        return {
            "status": "protocol_failure",
            "pred_e_ids": [],
            "raw_text": raw_text,
            "decision_line": decision_line,
            "error_type": "protocol_violation",
        }
    pred_e_ids = list(parsed.get("e_ids") or [])
    if len(pred_e_ids) != len(list(dict.fromkeys(pred_e_ids))):
        return {
            "status": "protocol_failure",
            "pred_e_ids": [],
            "raw_text": raw_text,
            "decision_line": decision_line,
            "error_type": "duplicate_eids",
        }
    cand_set = set(candidate_e_ids or [])
    if any(eid not in cand_set for eid in pred_e_ids):
        return {
            "status": "protocol_failure",
            "pred_e_ids": [],
            "raw_text": raw_text,
            "decision_line": decision_line,
            "error_type": "eid_out_of_candidates",
        }
    if (not allow_multiple) and len(pred_e_ids) > 1:
        return {
            "status": "protocol_failure",
            "pred_e_ids": [],
            "raw_text": raw_text,
            "decision_line": decision_line,
            "error_type": "multiple_eids_not_allowed",
        }
    if parsed.get("decision") == "NONE":
        pred_e_ids = []
    return {
        "status": "ok",
        "pred_e_ids": pred_e_ids,
        "raw_text": raw_text,
        "decision_line": decision_line,
        "error_type": None,
    }


def parse_dev_predictions(pred_rows: list[dict[str, Any]], allow_multiple: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pred_rows:
        metadata = dict(row.get("metadata") or {})
        candidate_e_ids = list(metadata.get("candidate_e_ids") or [])
        parsed = parse_decision_line(
            raw_text=str(row.get("raw_text") or ""),
            candidate_e_ids=candidate_e_ids,
            allow_multiple=allow_multiple,
        )
        rows.append({**parsed, "metadata": metadata})
    return rows


def _resolve_eval_gold_e_ids(metadata: dict[str, Any]) -> list[str]:
    effective = [str(x).strip() for x in (metadata.get("effective_gold_e_ids") or []) if str(x).strip()]
    if effective:
        return effective
    forced = [str(x).strip() for x in (metadata.get("gold_e_ids_single_if_forced") or []) if str(x).strip()]
    if forced:
        return forced
    return [str(x).strip() for x in (metadata.get("gold_e_ids") or []) if str(x).strip()]


def evaluate_bgroup_strict_set(parsed_pred_rows: list[dict[str, Any]], gold_examples: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    valid_count = 0
    parse_failure_count = 0
    protocol_failure_count = 0
    none_count = 0
    positive_total = 0
    negative_total = 0
    positive_exact_match = 0
    negative_none_correct = 0

    for pred_row, gold_row in zip(parsed_pred_rows, gold_examples):
        metadata = gold_row.get("metadata") or {}
        gold_set = set(_resolve_eval_gold_e_ids(metadata))
        status = str(pred_row.get("status") or "")

        if status == "ok":
            valid_count += 1
            pred_set = set(pred_row.get("pred_e_ids") or [])
            if len(pred_set) == 0:
                none_count += 1
        else:
            pred_set = None
            if status == "parse_failure":
                parse_failure_count += 1
            elif status == "protocol_failure":
                protocol_failure_count += 1

        if gold_set:
            positive_total += 1
            # Policy: invalid output is counted as FN for positive gold.
            if status != "ok":
                fn += 1
            else:
                if pred_set == gold_set:
                    tp += 1
                    positive_exact_match += 1
                elif len(pred_set or set()) == 0:
                    fn += 1
                else:
                    # Wrong positive prediction should count as both an incorrect positive
                    # and a missed gold positive, matching the encoder-side evaluation.
                    fp += 1
                    fn += 1
        else:
            negative_total += 1
            # Policy: invalid output is counted as FP for negative gold.
            if status != "ok":
                fp += 1
            else:
                if len(pred_set or set()) == 0:
                    tn += 1
                    negative_none_correct += 1
                else:
                    fp += 1

    total = len(gold_examples)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    valid_rate = valid_count / total if total > 0 else 0.0
    none_rate = none_count / total if total > 0 else 0.0
    pos_exact = positive_exact_match / positive_total if positive_total > 0 else 0.0
    neg_none_acc = negative_none_correct / negative_total if negative_total > 0 else 0.0
    return {
        "dev_strict_set_precision": precision,
        "dev_strict_set_recall": recall,
        "dev_strict_set_f1": f1,
        "dev_strict_set_accuracy": accuracy,
        "dev_valid_decision_rate": valid_rate,
        "dev_tp": tp,
        "dev_fp": fp,
        "dev_fn": fn,
        "dev_tn": tn,
        "dev_parse_failure_count": parse_failure_count,
        "dev_protocol_failure_count": protocol_failure_count,
        "dev_none_rate": none_rate,
        "dev_positive_exact_match_rate": pos_exact,
        "dev_negative_none_accuracy": neg_none_acc,
    }


def log_dev_metrics(logger: logging.Logger, metrics: dict[str, Any], prefix: str = "dev") -> None:
    logger.info(
        "[train_llm_sft][%s] f1=%.6f acc=%.6f precision=%.6f recall=%.6f valid_rate=%.6f tp=%s fp=%s fn=%s tn=%s parse_fail=%s protocol_fail=%s none_rate=%.6f pos_exact=%.6f neg_none_acc=%.6f",
        prefix,
        float(metrics.get(f"{prefix}_strict_set_f1", 0.0)),
        float(metrics.get(f"{prefix}_strict_set_accuracy", 0.0)),
        float(metrics.get(f"{prefix}_strict_set_precision", 0.0)),
        float(metrics.get(f"{prefix}_strict_set_recall", 0.0)),
        float(metrics.get(f"{prefix}_valid_decision_rate", 0.0)),
        metrics.get(f"{prefix}_tp", 0),
        metrics.get(f"{prefix}_fp", 0),
        metrics.get(f"{prefix}_fn", 0),
        metrics.get(f"{prefix}_tn", 0),
        metrics.get(f"{prefix}_parse_failure_count", 0),
        metrics.get(f"{prefix}_protocol_failure_count", 0),
        float(metrics.get(f"{prefix}_none_rate", 0.0)),
        float(metrics.get(f"{prefix}_positive_exact_match_rate", 0.0)),
        float(metrics.get(f"{prefix}_negative_none_accuracy", 0.0)),
    )


def _log_parse_failure_samples(
    *,
    logger: logging.Logger,
    gold_examples: list[dict[str, Any]],
    raw_pred_rows: list[dict[str, Any]],
    parsed_pred_rows: list[dict[str, Any]],
    prefix: str,
    max_samples: int = 5,
) -> None:
    shown = 0
    for gold_row, raw_row, parsed_row in zip(gold_examples, raw_pred_rows, parsed_pred_rows):
        if str(parsed_row.get("status") or "") != "parse_failure":
            continue
        metadata = gold_row.get("metadata") or {}
        logger.info(
            "[train_llm_sft][%s][parse_failure_sample_%d] key=%s role=%s error=%s gold=%s candidate_e_ids=%s raw_text=%s",
            prefix,
            shown + 1,
            metadata.get("example_key_full") or "",
            metadata.get("gold_example_role") or "",
            parsed_row.get("error_type") or "",
            metadata.get("gold_e_ids") or [],
            metadata.get("candidate_e_ids") or [],
            str(raw_row.get("raw_text") or "")[:500],
        )
        shown += 1
        if shown >= max_samples:
            break
    if shown > 0:
        logger.info(
            "[train_llm_sft][%s] logged_parse_failure_samples=%d",
            prefix,
            shown,
        )


def _write_prediction_csv(
    path: Path,
    split_name: str,
    gold_examples: Sequence[Dict[str, Any]],
    raw_pred_rows: Sequence[Dict[str, Any]],
    parsed_pred_rows: Sequence[Dict[str, Any]],
) -> str:
    fieldnames = [
        "example_key_full",
        "split",
        "gold_example_role",
        "candidate_e_ids",
        "candidate_number_to_eid",
        "gold_assistant",
        "gold_e_ids",
        "gold_e_ids_single_if_forced",
        "effective_gold_e_ids",
        "effective_decision_type",
        "raw_text",
        "prompt_text",
        "decision_line",
        "status",
        "error_type",
        "pred_e_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex, raw_row, parsed_row in zip(gold_examples, raw_pred_rows, parsed_pred_rows):
            messages = list(ex.get("messages") or [])
            md = dict(ex.get("metadata") or {})
            payload = {
                "example_key_full": md.get("example_key_full"),
                "split": split_name,
                "gold_example_role": md.get("gold_example_role"),
                "candidate_e_ids": json.dumps(md.get("candidate_e_ids") or [], ensure_ascii=False),
                "candidate_number_to_eid": json.dumps(md.get("candidate_number_to_eid") or {}, ensure_ascii=False),
                "gold_assistant": str((messages[2].get("content") if len(messages) >= 3 else "") or ""),
                "gold_e_ids": json.dumps(md.get("gold_e_ids") or [], ensure_ascii=False),
                "gold_e_ids_single_if_forced": json.dumps(md.get("gold_e_ids_single_if_forced") or [], ensure_ascii=False),
                "effective_gold_e_ids": json.dumps(md.get("effective_gold_e_ids") or [], ensure_ascii=False),
                "effective_decision_type": str(md.get("effective_decision_type") or ""),
                "raw_text": str(raw_row.get("raw_text") or ""),
                "prompt_text": str(raw_row.get("prompt_text") or ""),
                "decision_line": parsed_row.get("decision_line") or "",
                "status": parsed_row.get("status") or "",
                "error_type": parsed_row.get("error_type") or "",
                "pred_e_ids": json.dumps(parsed_row.get("pred_e_ids") or [], ensure_ascii=False),
            }
            writer.writerow(payload)
    return str(path)


def run_dev_evaluation(
    model: Any,
    tokenizer: Any,
    dev_examples: list[dict[str, Any]],
    gen_cfg: dict[str, Any],
    logger: logging.Logger,
    allow_multiple: bool,
) -> dict[str, Any]:
    infer_examples = build_dev_infer_examples(dev_examples)
    raw_pred_rows = generate_dev_predictions(
        model=model,
        tokenizer=tokenizer,
        infer_examples=infer_examples,
        gen_cfg=gen_cfg,
    )
    parsed_pred_rows = parse_dev_predictions(raw_pred_rows, allow_multiple=allow_multiple)
    metrics = evaluate_bgroup_strict_set(parsed_pred_rows=parsed_pred_rows, gold_examples=dev_examples)
    log_dev_metrics(logger, metrics, prefix="dev")
    _log_parse_failure_samples(
        logger=logger,
        gold_examples=dev_examples,
        raw_pred_rows=raw_pred_rows,
        parsed_pred_rows=parsed_pred_rows,
        prefix="dev",
        max_samples=5,
    )
    return {**metrics, "raw_pred_rows": raw_pred_rows, "parsed_pred_rows": parsed_pred_rows}


def _save_checkpoint(model: Any, tokenizer: Any, ckpt_path: Path) -> str:
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_path / "model")
    tokenizer.save_pretrained(ckpt_path / "tokenizer")
    return str(ckpt_path)


def _copy_checkpoint(src: str | Path, dst: str | Path) -> str:
    src_p = Path(src)
    dst_p = Path(dst)
    if dst_p.exists():
        shutil.rmtree(dst_p)
    shutil.copytree(src_p, dst_p)
    return str(dst_p)


def maybe_update_best_checkpoint(metrics: dict[str, Any], ckpt_path: str, best_state: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    current_f1 = float(metrics.get("dev_strict_set_f1", 0.0))
    current_valid = float(metrics.get("dev_valid_decision_rate", 0.0))
    current_step = metrics.get("step", 0)

    is_better = False
    if current_f1 > float(best_state.get("best_metric", float("-inf"))):
        is_better = True
    elif current_f1 == float(best_state.get("best_metric", float("-inf"))):
        if current_valid > float(best_state.get("best_valid_rate", float("-inf"))):
            is_better = True
        elif current_valid == float(best_state.get("best_valid_rate", float("-inf"))):
            prev_step = best_state.get("best_step")
            if prev_step is None or current_step < prev_step:
                is_better = True

    if is_better:
        best_ckpt_dir = Path(output_dir) / "checkpoints" / "best"
        _copy_checkpoint(ckpt_path, best_ckpt_dir)
        return {
            "best_metric": current_f1,
            "best_valid_rate": current_valid,
            "best_ckpt_path": str(best_ckpt_dir),
            "best_step": current_step,
            "best_epoch": metrics.get("epoch"),
        }
    return best_state


def save_last_checkpoint(trainer_or_model: Any, tokenizer: Any, ckpt_path: str | Path) -> str:
    model = getattr(trainer_or_model, "model", trainer_or_model)
    return _save_checkpoint(model, tokenizer, Path(ckpt_path))


def _maybe_init_wandb(cfg: dict[str, Any], run_context: RunContext, summary_cfg: dict[str, Any]) -> Any:
    logger = logging.getLogger("kmwe")
    base = cfg.get("wandb", {}) or {}
    override = (cfg.get("llm_sft", {}) or {}).get("wandb", {}) or {}
    wandb_cfg = {**base, **override}
    if not bool(wandb_cfg.get("enabled", False)):
        logger.info("[train_llm_sft][wandb] disabled")
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        logger.warning("[train_llm_sft][wandb] import failed: %s", exc)
        return None
    run = wandb.init(
        project=str(wandb_cfg.get("project") or "kmwe-llm-sft"),
        entity=str(wandb_cfg.get("entity") or "") or None,
        group=str(wandb_cfg.get("group") or "") or f"{cfg.get('exp', {}).get('exp_id', 'default')}:train_llm_sft",
        name=str(wandb_cfg.get("name") or "") or f"train_llm_sft/{cfg.get('exp', {}).get('exp_id', 'default')}/{run_context.run_id}",
        mode=str(wandb_cfg.get("mode") or "online"),
        tags=["train_llm_sft", str(cfg.get('exp', {}).get('exp_id') or 'default')],
        config=summary_cfg,
        reinit=True,
    )
    logger.info("[train_llm_sft][wandb] init ok")
    return run


def _write_openai_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            messages = row.get("messages") or []
            payload = {"messages": messages}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_train_llm_sft_openai(*, cfg: dict[str, Any], run_context: RunContext) -> dict[str, Any]:
    logger = logging.getLogger("kmwe")
    sft_cfg = cfg.get("llm_sft", {}) or {}
    openai_cfg = dict(sft_cfg.get("openai") or {})

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai backend는 'openai' 패키지가 필요합니다.") from exc

    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = outputs_dir / "train_llm_sft_live.log"
    train_log_handler = logging.FileHandler(train_log_path, encoding="utf-8")
    train_log_handler.setLevel(logging.INFO)
    train_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    train_log_handler.addFilter(_TrainLlmSftOnlyFilter())
    logger.addHandler(train_log_handler)

    model_name = _resolve_model_name(cfg)
    train_path, dev_path, test_path = _resolve_sft_paths(cfg)
    allow_multiple = _resolve_allow_multiple(cfg)
    epochs = max(1, int(sft_cfg.get("epochs", 1)))

    train_examples = load_sft_jsonl(train_path)
    dev_examples = load_sft_jsonl(dev_path) if dev_path is not None else []
    test_examples = load_sft_jsonl(test_path) if test_path is not None else []

    _log_dataset_overview(logger, "train", train_examples)
    if dev_examples:
        _log_dataset_overview(logger, "dev", dev_examples)
    if test_examples:
        _log_dataset_overview(logger, "test", test_examples)

    for split_name, rows in (("train", train_examples), ("dev", dev_examples), ("test", test_examples)):
        for row in rows:
            messages = row.get("messages") or []
            if len(messages) != 3:
                raise ConfigError(f"openai backend용 {split_name} example messages 길이는 3이어야 합니다.")
            roles = [str(m.get("role") or "") for m in messages]
            if roles != ["system", "user", "assistant"]:
                raise ConfigError(f"openai backend용 {split_name} role 순서가 올바르지 않습니다. got={roles}")

    api_key = str(openai_cfg.get("api_key") or "").strip()
    if not api_key:
        api_key_env = str(openai_cfg.get("api_key_env") or "OPENAI_API_KEY").strip()
        api_key = str(os.environ.get(api_key_env) or "").strip()
    if not api_key:
        raise ConfigError("openai backend 사용 시 API key가 필요합니다. llm_sft.openai.api_key 또는 env를 설정하세요.")

    upload_train_path = outputs_dir / "openai_train.jsonl"
    upload_dev_path = outputs_dir / "openai_dev.jsonl"
    _write_openai_jsonl(upload_train_path, train_examples)
    if dev_examples:
        _write_openai_jsonl(upload_dev_path, dev_examples)

    client = OpenAI(api_key=api_key)
    logger.info(
        "[train_llm_sft][openai] model=%s train_rows=%d dev_rows=%d allow_multiple=%s epochs=%d",
        model_name,
        len(train_examples),
        len(dev_examples),
        allow_multiple,
        epochs,
    )

    with upload_train_path.open("rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    logger.info("[train_llm_sft][openai] uploaded training_file_id=%s", train_file.id)

    dev_file_id = None
    if dev_examples:
        with upload_dev_path.open("rb") as f:
            dev_file = client.files.create(file=f, purpose="fine-tune")
        dev_file_id = dev_file.id
        logger.info("[train_llm_sft][openai] uploaded validation_file_id=%s", dev_file_id)

    job_kwargs: dict[str, Any] = {
        "model": model_name,
        "training_file": train_file.id,
        "hyperparameters": {"n_epochs": epochs},
    }
    if dev_file_id:
        job_kwargs["validation_file"] = dev_file_id
    suffix = str(openai_cfg.get("suffix") or "").strip()
    if suffix:
        job_kwargs["suffix"] = suffix

    job = client.fine_tuning.jobs.create(**job_kwargs)
    logger.info("[train_llm_sft][openai] created fine_tuning_job_id=%s status=%s", job.id, getattr(job, "status", "unknown"))

    report = {
        "stage": "train_llm_sft",
        "backend": "openai",
        "created_at": iso_now(),
        "exp_id": cfg.get("exp", {}).get("exp_id", "default"),
        "run_id": run_context.run_id,
        "model_name_or_path": model_name,
        "train_jsonl": str(train_path),
        "dev_jsonl": str(dev_path) if dev_path else "",
        "test_jsonl": str(test_path) if test_path else "",
        "allow_multiple": allow_multiple,
        "epochs": epochs,
        "n_train": len(train_examples),
        "n_dev": len(dev_examples),
        "n_test": len(test_examples),
        "openai": {
            "training_file_id": train_file.id,
            "validation_file_id": dev_file_id,
            "fine_tuning_job_id": job.id,
            "job_status": getattr(job, "status", None),
        },
        "outputs": {
            "train_live_log": str(train_log_path),
            "openai_train_jsonl": str(upload_train_path),
            "openai_dev_jsonl": str(upload_dev_path) if dev_examples else "",
        },
    }
    report_path = outputs_dir / "train_llm_sft_report.json"
    write_json(report_path, report, indent=2)
    logger.info("[train_llm_sft] wrote report=%s", report_path)
    logger.info("[train_llm_sft] wrote live_log=%s", train_log_path)

    logger.removeHandler(train_log_handler)
    train_log_handler.close()

    return {
        "backend": "openai",
        "fine_tuning_job_id": job.id,
        "training_file_id": train_file.id,
        "validation_file_id": dev_file_id,
        "report_path": str(report_path),
    }


def run_train_llm_sft(*, cfg: dict[str, Any], run_context: RunContext) -> dict[str, Any]:
    backend = _resolve_backend(cfg)
    if backend == "openai":
        return _run_train_llm_sft_openai(cfg=cfg, run_context=run_context)

    logger = logging.getLogger("kmwe")
    sft_cfg = cfg.get("llm_sft", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("train_llm_sft stage requires 'torch'.") from exc

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("train_llm_sft stage requires 'transformers'. Please install transformers.") from exc

    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = outputs_dir / "train_llm_sft_live.log"
    train_log_handler = logging.FileHandler(train_log_path, encoding="utf-8")
    train_log_handler.setLevel(logging.INFO)
    train_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    train_log_handler.addFilter(_TrainLlmSftOnlyFilter())
    logger.addHandler(train_log_handler)

    model_name = _resolve_model_name(cfg)
    train_path, dev_path, test_path = _resolve_sft_paths(cfg)
    allow_multiple = _resolve_allow_multiple(cfg)
    gen_cfg = dict(sft_cfg.get("gen_cfg") or {})
    max_seq_len = int(sft_cfg.get("max_seq_len", 2048))
    batch_size = max(1, int(sft_cfg.get("batch_size", 1)))
    lr = float(sft_cfg.get("lr", 2e-5))
    epochs = max(1, int(sft_cfg.get("epochs", 1)))
    grad_accum_steps = max(1, int(sft_cfg.get("grad_accum_steps", runtime_cfg.get("grad_accum_steps", 1) or 1)))
    log_every_steps = max(1, int(sft_cfg.get("log_every_steps", 10)))
    shuffle = bool(sft_cfg.get("shuffle", True))
    seed = int(runtime_cfg.get("seed", 42) or 42)
    effective_batch = batch_size * grad_accum_steps
    log_probe_predictions = bool(sft_cfg.get("log_probe_predictions", True))
    probe_every_steps = max(1, int(sft_cfg.get("probe_every_steps", max(log_every_steps, 1))))
    use_bf16 = bool(sft_cfg.get("use_bf16", torch.cuda.is_available()))
    attn_implementation = str(sft_cfg.get("attn_implementation") or "").strip()

    train_examples = load_sft_jsonl(train_path)
    dev_examples = load_sft_jsonl(dev_path) if dev_path is not None else []
    test_examples = load_sft_jsonl(test_path) if test_path is not None else []

    _log_dataset_overview(logger, "train", train_examples)
    if dev_examples:
        _log_dataset_overview(logger, "dev", dev_examples)
    if test_examples:
        _log_dataset_overview(logger, "test", test_examples)

    probe_source_rows = dev_examples if dev_examples else train_examples
    probe_examples_by_role = _build_role_probe_examples(probe_source_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "[train_llm_sft][paths] model_name_or_path=%s train=%s dev=%s test=%s outputs_dir=%s",
        model_name,
        train_path,
        dev_path,
        test_path,
        outputs_dir,
    )
    logger.info(
        "[train_llm_sft][config] device=%s max_seq_len=%d batch_size=%d grad_accum_steps=%d effective_batch=%d epochs=%d lr=%s allow_multiple=%s use_bf16=%s attn_implementation=%s",
        device,
        max_seq_len,
        batch_size,
        grad_accum_steps,
        effective_batch,
        epochs,
        lr,
        allow_multiple,
        use_bf16,
        attn_implementation or "auto",
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_load_kwargs: dict[str, Any] = {}
    if device.type == "cuda" and use_bf16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    if attn_implementation:
        model_load_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_load_kwargs)
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
    if bool(sft_cfg.get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("[train_llm_sft] gradient_checkpointing enabled")
    logger.info(
        "[train_llm_sft][model] dtype=%s pad_token_id=%s eos_token_id=%s",
        str(next(model.parameters()).dtype),
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    )

    train_encoded = _encode_examples(train_examples, tokenizer, max_seq_len, logger, "train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    wandb_run = _maybe_init_wandb(
        cfg,
        run_context,
        {
            "stage": "train_llm_sft",
            "run_id": run_context.run_id,
            "exp_id": cfg.get("exp", {}).get("exp_id", "default"),
            "model_name_or_path": model_name,
            "train_jsonl": str(train_path),
            "dev_jsonl": str(dev_path) if dev_path else "",
            "test_jsonl": str(test_path) if test_path else "",
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch": effective_batch,
            "epochs": epochs,
            "lr": lr,
            "allow_multiple": allow_multiple,
            "use_bf16": use_bf16,
            "attn_implementation": attn_implementation or "auto",
            "log_probe_predictions": log_probe_predictions,
            "probe_every_steps": probe_every_steps,
            "gen_cfg": gen_cfg,
            "n_train": len(train_examples),
            "n_dev": len(dev_examples),
            "n_test": len(test_examples),
        },
    )

    progress_path = outputs_dir / "train_llm_sft_progress.jsonl"
    started_at = datetime.now().astimezone().isoformat()
    global_step = 0
    micro_step = 0
    examples_seen = 0
    ema_loss: float | None = None
    ema_decay = 0.98
    best_state = {
        "best_metric": float("-inf"),
        "best_valid_rate": float("-inf"),
        "best_ckpt_path": None,
        "best_step": None,
        "best_epoch": None,
    }
    dev_history: list[dict[str, Any]] = []

    model.train()
    optimizer.zero_grad()
    for epoch in range(epochs):
        logger.info("[train_llm_sft][epoch_start] epoch=%d/%d", epoch + 1, epochs)
        epoch_seed = seed + epoch
        preview_indices = list(range(len(train_encoded)))
        if shuffle:
            preview_rng = random.Random(epoch_seed)
            preview_rng.shuffle(preview_indices)
        preview_keys = [
            str((train_encoded[i].metadata or {}).get("example_key_full") or "")
            for i in preview_indices[:10]
        ]
        logger.info(
            "[train_llm_sft][epoch_order] epoch=%d seed=%d first10_example_key_full=%s",
            epoch + 1,
            epoch_seed,
            preview_keys,
        )
        for batch_examples in _iter_batches(train_encoded, batch_size=batch_size, shuffle=shuffle, seed=epoch_seed):
            batch = _pad_batch(batch_examples, pad_token_id=int(tokenizer.pad_token_id))
            batch_roles = Counter(str(md.get("gold_example_role") or "") for md in batch["metadata"])
            if global_step == 0 and micro_step == 0:
                logger.info("[train_llm_sft][first_batch] role_counts=%s", dict(batch_roles))
                logger.info("[train_llm_sft][first_batch] first_prompt=%s", batch_examples[0].prompt_text[:1200])
                logger.info("[train_llm_sft][first_batch] first_target=%s", batch_examples[0].assistant_target)

            input_ids = torch.tensor(batch["input_ids"], device=device, dtype=torch.long)
            attention_mask = torch.tensor(batch["attention_mask"], device=device, dtype=torch.long)
            labels = torch.tensor(batch["labels"], device=device, dtype=torch.long)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = out.loss
            loss = raw_loss / grad_accum_steps
            loss.backward()

            per_example_losses = _compute_per_example_loss(out.logits.detach(), labels.detach())
            role_loss_sums: dict[str, float] = {}
            role_loss_counts: dict[str, int] = {}
            for ex_obj, ex_loss in zip(batch_examples, per_example_losses):
                role = str((ex_obj.metadata or {}).get("gold_example_role") or "")
                role_loss_sums[role] = role_loss_sums.get(role, 0.0) + float(ex_loss)
                role_loss_counts[role] = role_loss_counts.get(role, 0) + 1
            role_avg_loss = {
                role: (role_loss_sums[role] / role_loss_counts[role])
                for role in sorted(role_loss_sums)
                if role_loss_counts.get(role, 0) > 0
            }

            loss_value = float(raw_loss.detach().item())
            if ema_loss is None:
                ema_loss = loss_value
            else:
                ema_loss = (ema_decay * ema_loss) + ((1.0 - ema_decay) * loss_value)

            micro_step += 1
            examples_seen += len(batch_examples)

            if micro_step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % log_every_steps == 0 or global_step == 1:
                    lr_now = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else lr
                    logger.info(
                        "[train_llm_sft][progress] step=%d epoch=%d loss=%.6f ema_loss=%.6f lr=%.8f examples_seen=%d batch_roles=%s role_avg_loss=%s",
                        global_step,
                        epoch + 1,
                        loss_value,
                        float(ema_loss if ema_loss is not None else loss_value),
                        lr_now,
                        examples_seen,
                        dict(batch_roles),
                        role_avg_loss,
                    )
                    with progress_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss": loss_value,
                            "ema_loss": float(ema_loss if ema_loss is not None else loss_value),
                            "lr": lr_now,
                            "examples_seen": examples_seen,
                            "batch_roles": dict(batch_roles),
                            "role_avg_loss": role_avg_loss,
                            "ts": iso_now(),
                        }, ensure_ascii=False) + "\n")
                    if wandb_run is not None:
                        wandb_payload = {
                            "train/step": global_step,
                            "train/loss": loss_value,
                            "train/ema_loss": float(ema_loss if ema_loss is not None else loss_value),
                            "train/lr": lr_now,
                            "train/examples_seen": examples_seen,
                        }
                        for role_name, role_loss in role_avg_loss.items():
                            wandb_payload[f"train/role_loss/{role_name}"] = role_loss
                        wandb_run.log(wandb_payload)
                    if log_probe_predictions and (global_step == 1 or global_step % probe_every_steps == 0):
                        _log_role_probe_predictions(
                            model=model,
                            tokenizer=tokenizer,
                            probe_examples_by_role=probe_examples_by_role,
                            gen_cfg=gen_cfg,
                            logger=logger,
                            allow_multiple=allow_multiple,
                            prefix=f"probe_step_{global_step}",
                        )

        epoch_ckpt_path = _save_checkpoint(model, tokenizer, outputs_dir / "checkpoints" / f"epoch_{epoch + 1}")
        logger.info("[train_llm_sft][checkpoint] saved epoch=%d path=%s", epoch + 1, epoch_ckpt_path)

        if dev_examples:
            dev_eval = run_dev_evaluation(
                model=model,
                tokenizer=tokenizer,
                dev_examples=dev_examples,
                gen_cfg=gen_cfg,
                logger=logger,
                allow_multiple=allow_multiple,
            )
            dev_predictions_path = _write_prediction_csv(
                path=outputs_dir / "dev_predictions.csv",
                split_name="dev",
                gold_examples=dev_examples,
                raw_pred_rows=list(dev_eval.get("raw_pred_rows") or []),
                parsed_pred_rows=list(dev_eval.get("parsed_pred_rows") or []),
            )
            _write_prediction_csv(
                path=outputs_dir / f"dev_predictions_epoch_{epoch + 1}.csv",
                split_name="dev",
                gold_examples=dev_examples,
                raw_pred_rows=list(dev_eval.get("raw_pred_rows") or []),
                parsed_pred_rows=list(dev_eval.get("parsed_pred_rows") or []),
            )
            dev_metrics = {k: v for k, v in dev_eval.items() if k not in {"raw_pred_rows", "parsed_pred_rows"}}
            dev_metrics.update({"epoch": epoch + 1, "step": global_step})
            dev_history.append(dev_metrics)
            best_state = maybe_update_best_checkpoint(
                metrics=dev_metrics,
                ckpt_path=epoch_ckpt_path,
                best_state=best_state,
                output_dir=outputs_dir,
            )
            logger.info(
                "[train_llm_sft][best] epoch=%d step=%d best_f1=%.6f best_valid_rate=%.6f best_ckpt=%s",
                epoch + 1,
                global_step,
                float(best_state.get("best_metric", 0.0)),
                float(best_state.get("best_valid_rate", 0.0)),
                best_state.get("best_ckpt_path"),
            )
            if log_probe_predictions:
                _log_role_probe_predictions(
                    model=model,
                    tokenizer=tokenizer,
                    probe_examples_by_role=probe_examples_by_role,
                    gen_cfg=gen_cfg,
                    logger=logger,
                    allow_multiple=allow_multiple,
                    prefix=f"probe_epoch_{epoch + 1}",
                )
            if wandb_run is not None:
                log_payload = {f"dev/{k[4:]}": v for k, v in dev_metrics.items() if k.startswith("dev_")}
                log_payload["dev/epoch"] = epoch + 1
                log_payload["dev/step"] = global_step
                wandb_run.log(log_payload)

    last_ckpt_path = save_last_checkpoint(model, tokenizer, outputs_dir / "checkpoints" / "last")
    final_test_metrics: dict[str, Any] = {}
    test_predictions_path = ""
    test_eval_ckpt_path = str(best_state.get("best_ckpt_path") or last_ckpt_path)
    if test_examples:
        import gc

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        test_ckpt_dir = Path(test_eval_ckpt_path)
        test_tokenizer = AutoTokenizer.from_pretrained(test_ckpt_dir / "tokenizer", trust_remote_code=True)
        test_tokenizer.padding_side = "right"
        if test_tokenizer.pad_token is None:
            test_tokenizer.pad_token = test_tokenizer.eos_token or test_tokenizer.unk_token

        test_model_load_kwargs: dict[str, Any] = {}
        if device.type == "cuda" and use_bf16:
            test_model_load_kwargs["torch_dtype"] = torch.bfloat16
        if attn_implementation:
            test_model_load_kwargs["attn_implementation"] = attn_implementation

        test_model = AutoModelForCausalLM.from_pretrained(
            test_ckpt_dir / "model",
            trust_remote_code=True,
            **test_model_load_kwargs,
        )
        test_model.to(device)
        if getattr(test_model.config, "pad_token_id", None) is None:
            test_model.config.pad_token_id = test_tokenizer.pad_token_id
        if getattr(test_model.config, "eos_token_id", None) is None:
            test_model.config.eos_token_id = test_tokenizer.eos_token_id
        if getattr(test_model, "generation_config", None) is not None:
            if getattr(test_model.generation_config, "pad_token_id", None) is None:
                test_model.generation_config.pad_token_id = test_tokenizer.pad_token_id
            if getattr(test_model.generation_config, "eos_token_id", None) is None:
                test_model.generation_config.eos_token_id = test_tokenizer.eos_token_id

        logger.info("[train_llm_sft][test_eval] checkpoint=best path=%s", test_eval_ckpt_path)
        test_eval = run_dev_evaluation(
            model=test_model,
            tokenizer=test_tokenizer,
            dev_examples=test_examples,
            gen_cfg=gen_cfg,
            logger=logger,
            allow_multiple=allow_multiple,
        )
        test_predictions_path = _write_prediction_csv(
            path=outputs_dir / "test_predictions.csv",
            split_name="test",
            gold_examples=test_examples,
            raw_pred_rows=list(test_eval.get("raw_pred_rows") or []),
            parsed_pred_rows=list(test_eval.get("parsed_pred_rows") or []),
        )
        _write_prediction_csv(
            path=outputs_dir / "test_predictions_best.csv",
            split_name="test",
            gold_examples=test_examples,
            raw_pred_rows=list(test_eval.get("raw_pred_rows") or []),
            parsed_pred_rows=list(test_eval.get("parsed_pred_rows") or []),
        )
        final_test_metrics = {k.replace("dev_", "test_", 1): v for k, v in test_eval.items() if k not in {"raw_pred_rows", "parsed_pred_rows"}}
        log_dev_metrics(logger, final_test_metrics, prefix="test")
        if wandb_run is not None:
            wandb_run.log({f"test/{k[5:]}": v for k, v in final_test_metrics.items() if k.startswith("test_")})

    report = {
        "stage": "train_llm_sft",
        "created_at": iso_now(),
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "exp_id": cfg.get("exp", {}).get("exp_id", "default"),
        "run_id": run_context.run_id,
        "model_name_or_path": model_name,
        "train_jsonl": str(train_path),
        "dev_jsonl": str(dev_path) if dev_path else "",
        "test_jsonl": str(test_path) if test_path else "",
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch": effective_batch,
        "epochs": epochs,
        "lr": lr,
        "allow_multiple": allow_multiple,
        "use_bf16": use_bf16,
        "attn_implementation": attn_implementation or "auto",
        "gen_cfg": gen_cfg,
        "n_train": len(train_examples),
        "n_dev": len(dev_examples),
        "n_test": len(test_examples),
        "global_steps": global_step,
        "micro_steps": micro_step,
        "examples_seen": examples_seen,
        "best_state": best_state,
        "dev_history": dev_history,
        "last_ckpt_path": last_ckpt_path,
        **final_test_metrics,
        "outputs": {
            "progress_jsonl": str(progress_path),
            "train_live_log": str(train_log_path),
            "dev_predictions": str(outputs_dir / "dev_predictions.csv") if dev_examples else "",
            "dev_predictions_per_epoch_glob": str(outputs_dir / "dev_predictions_epoch_*.csv") if dev_examples else "",
            "test_predictions": test_predictions_path,
            "test_predictions_best": str(outputs_dir / "test_predictions_best.csv") if test_examples else "",
            "last_checkpoint": str(last_ckpt_path),
            "best_checkpoint": str(best_state.get("best_ckpt_path") or ""),
            "test_eval_checkpoint": str(test_eval_ckpt_path),
        },
    }
    report_path = outputs_dir / "train_llm_sft_report.json"
    write_json(report_path, report, indent=2)
    logger.info("[train_llm_sft] wrote report=%s", report_path)
    logger.info("[train_llm_sft] wrote live_log=%s", train_log_path)
    logger.info("[train_llm_sft] saved last checkpoint=%s", last_ckpt_path)

    if wandb_run is not None:
        wandb_run.log({
            "final/best_dev_strict_set_f1": float(best_state.get("best_metric", 0.0)),
            "final/best_dev_valid_rate": float(best_state.get("best_valid_rate", 0.0)),
            "final/global_steps": global_step,
            "final/examples_seen": examples_seen,
        })
        wandb_run.finish()

    logger.removeHandler(train_log_handler)
    train_log_handler.close()

    return {
        "best_ckpt_path": best_state.get("best_ckpt_path"),
        "best_metric": best_state.get("best_metric"),
        "last_ckpt_path": last_ckpt_path,
    }
