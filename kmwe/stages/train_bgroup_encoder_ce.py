from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from kmwe.core.config_loader import ConfigError
from kmwe.core.run_context import RunContext
from kmwe.core.utils import iso_now
from kmwe.data.factory_bgroup_encoder import (
    BGROUP_CANDIDATE_SCORING,
    BGROUP_INPUT_MODE,
    BGROUP_SPAN_MARKER_STYLE,
    BGROUP_TEXT_B_FORMAT,
    load_bgroup_cross_encoder_examples,
)
from kmwe.utils.jsonio import write_json, write_jsonl_line


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    serialized = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple)):
            serialized[k] = json.dumps(v, ensure_ascii=False)
        else:
            serialized[k] = v
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(serialized.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(serialized)


def _mean_pool(last_hidden_state, attention_mask):
    weights = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return summed / denom


class LocalSoftmaxHead:  # small wrapper for typing friendliness in logs
    pass


class GroupedCandidateScorerModule(__import__('torch').nn.Module):
    def __init__(self, hidden_size: int):
        import torch.nn as nn

        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.none_bias = nn.Parameter(__import__('torch').zeros(1))

    def forward(self, pooled):
        return self.linear(pooled).squeeze(-1)


def _resolve_device(device_cfg: str) -> str:
    import torch

    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _autocast_ctx(device: str, mixed_precision: str):
    import torch

    if device.startswith("cuda") and mixed_precision not in {"off", "false", "none", "cpu"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _save_checkpoint(*, model, tokenizer, scorer, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    enc_dir = ckpt_dir / "encoder"
    tok_dir = ckpt_dir / "tokenizer"
    enc_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(enc_dir)
    tokenizer.save_pretrained(tok_dir)
    import torch

    torch.save(scorer.state_dict(), ckpt_dir / "head.pt")


def _load_checkpoint(*, encoder_name: str, tokenizer_name: str, head_path: Path, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    added = tokenizer.add_special_tokens({"additional_special_tokens": ["[SPAN]", "[/SPAN]"]})
    model = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        hidden_size = int(getattr(model.config, "d_model", 0) or 0)
    if hidden_size <= 0:
        raise ConfigError("encoder hidden_size를 확인할 수 없습니다.")
    scorer = GroupedCandidateScorerModule(hidden_size)
    state = torch.load(head_path, map_location="cpu")
    scorer.load_state_dict(state)
    model.to(device)
    scorer.to(device)
    return model, tokenizer, scorer


def _prepare_model_and_tokenizer(*, encoder_name: str, tokenizer_name: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    added = tokenizer.add_special_tokens({"additional_special_tokens": ["[SPAN]", "[/SPAN]"]})
    model = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        hidden_size = int(getattr(model.config, "d_model", 0) or 0)
    if hidden_size <= 0:
        raise ConfigError("encoder hidden_size를 확인할 수 없습니다.")
    scorer = GroupedCandidateScorerModule(hidden_size)
    model.to(device)
    scorer.to(device)
    return model, tokenizer, scorer, int(added)


def _count_trainable_params(module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def _iter_batches(examples: list[dict[str, Any]], batch_size: int, *, shuffle: bool, seed: int):
    indices = list(range(len(examples)))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        yield [examples[j] for j in indices[i : i + batch_size]]


def _score_batch(*, model, tokenizer, scorer, examples: list[dict[str, Any]], max_seq_len: int, device: str, mixed_precision: str, freeze_encoder: bool):
    import torch

    text_as: list[str] = []
    text_bs: list[str] = []
    spans: list[tuple[int, int]] = []
    for ex in examples:
        start = len(text_as)
        for cand in ex.get("candidate_inputs") or []:
            text_as.append(str(cand.get("text_a") or ""))
            text_bs.append(str(cand.get("text_b") or ""))
        spans.append((start, len(text_as)))
    if not text_as:
        return []
    enc = tokenizer(
        text_as,
        text_bs,
        truncation=True,
        padding=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad() if freeze_encoder else nullcontext():
        with _autocast_ctx(device, mixed_precision):
            outputs = model(**enc)
            pooled = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])
    cand_logits_flat = scorer(pooled)
    none_logit = scorer.none_bias.view(1)
    out = []
    for ex, (start, end) in zip(examples, spans):
        logits = cand_logits_flat[start:end]
        logits_with_none = torch.cat([logits, none_logit], dim=0)
        out.append(
            {
                "example": ex,
                "logits_with_none": logits_with_none,
                "candidate_count": int(end - start),
            }
        )
    return out


def _label_space(example: dict[str, Any]) -> list[str]:
    return list(example.get("candidate_e_ids") or []) + ["__NONE__"]


def _build_debug_rows(*, scored_batch: list[dict[str, Any]], split: str, epoch: int, step: int, max_examples: int) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    for item in scored_batch[: max(0, max_examples)]:
        ex = item["example"]
        logits = item["logits_with_none"].detach().float()
        probs = torch.softmax(logits, dim=0).tolist()
        logits_list = logits.tolist()
        labels = _label_space(ex)
        ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        pred_index = int(max(range(len(probs)), key=lambda i: probs[i]))
        pred_label = labels[pred_index]
        rows.append(
            {
                "timestamp": iso_now(),
                "split": split,
                "epoch": int(epoch),
                "step": int(step),
                "group_key": ex.get("group_key"),
                "polyset_id": ex.get("polyset_id"),
                "role": ex.get("gold_example_role"),
                "question_text_a": (ex.get("candidate_inputs") or [{}])[0].get("text_a", ""),
                "question_candidate_text_bs": [c.get("text_b") for c in (ex.get("candidate_inputs") or [])],
                "question_candidate_e_ids": ex.get("candidate_e_ids"),
                "answer_gold_e_id": ex.get("gold_e_id"),
                "answer_gold_index": ex.get("label_index"),
                "answer_pred_e_id": pred_label,
                "answer_pred_index": pred_index,
                "answer_candidate_logits": logits_list,
                "answer_candidate_probs": probs,
                "answer_ranked_labels": ranked,
                "answer_selected_prob": probs[pred_index] if probs else None,
                "input_mode": BGROUP_INPUT_MODE,
                "span_marker_style": BGROUP_SPAN_MARKER_STYLE,
                "text_b_format": BGROUP_TEXT_B_FORMAT,
                "candidate_scoring": BGROUP_CANDIDATE_SCORING,
            }
        )
    return rows


def _evaluate_split(*, model, tokenizer, scorer, examples: list[dict[str, Any]], batch_size: int, max_seq_len: int, device: str, mixed_precision: str, split_name: str, debug_path: Path | None = None, debug_csv_path: Path | None = None, debug_max_examples: int = 0, prediction_path: Path | None = None, prediction_csv_path: Path | None = None, record_all_predictions: bool = False, epoch: int = 0, step: int = 0, freeze_encoder: bool = False) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if not examples:
        return {
            f"{split_name}_loss_mean": 0.0,
            f"{split_name}_top1_acc": 0.0,
            f"{split_name}_sample_acc": 0.0,
            f"{split_name}_mrr": 0.0,
            f"{split_name}_pos_acc": 0.0,
            f"{split_name}_none_acc": 0.0,
            f"{split_name}_balanced_acc": 0.0,
            f"{split_name}_positive_exact_precision": 0.0,
            f"{split_name}_positive_exact_recall": 0.0,
            f"{split_name}_positive_exact_f1": 0.0,
            f"{split_name}_precision": 0.0,
            f"{split_name}_recall": 0.0,
            f"{split_name}_f1": 0.0,
            f"{split_name}_macro_f1": 0.0,
            f"{split_name}_acc_by_polyset": {},
            f"{split_name}_n_examples": 0,
        }

    losses: list[float] = []
    correct = 0
    rr_sum = 0.0
    pos_total = pos_correct = 0
    none_total = none_correct = 0
    tp = fp = fn = 0
    gold_counts = Counter()
    pred_counts = Counter()
    tp_by_label = Counter()
    acc_by_polyset_num = Counter()
    acc_by_polyset_den = Counter()

    model.eval()
    scorer.eval()
    for batch in _iter_batches(examples, batch_size, shuffle=False, seed=0):
        scored = _score_batch(
            model=model,
            tokenizer=tokenizer,
            scorer=scorer,
            examples=batch,
            max_seq_len=max_seq_len,
            device=device,
            mixed_precision=mixed_precision,
            freeze_encoder=freeze_encoder,
        )
        for item in scored:
            ex = item["example"]
            logits = item["logits_with_none"]
            label = int(ex.get("label_index") or 0)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], device=logits.device))
            losses.append(float(loss.detach().cpu().item()))
            probs = torch.softmax(logits.detach().float(), dim=0).tolist()
            pred_index = int(max(range(len(probs)), key=lambda i: probs[i]))
            labels = _label_space(ex)
            gold_label = labels[label]
            pred_label = labels[pred_index]
            gold_counts[gold_label] += 1
            pred_counts[pred_label] += 1
            if pred_index == label:
                correct += 1
                tp_by_label[gold_label] += 1
                acc_by_polyset_num[str(ex.get("polyset_id") or "")] += 1
            acc_by_polyset_den[str(ex.get("polyset_id") or "")] += 1
            ranking = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            rr_sum += 1.0 / float(ranking.index(label) + 1)
            gold_is_none = (gold_label == "__NONE__")
            pred_is_none = (pred_label == "__NONE__")
            if gold_is_none:
                none_total += 1
                if pred_is_none:
                    none_correct += 1
            else:
                pos_total += 1
                if pred_label == gold_label:
                    pos_correct += 1
            if not gold_is_none and not pred_is_none and pred_label == gold_label:
                tp += 1
            elif gold_is_none and not pred_is_none:
                fp += 1
            elif (not gold_is_none) and pred_is_none:
                fn += 1
            elif (not gold_is_none) and (not pred_is_none) and pred_label != gold_label:
                fp += 1
                fn += 1
        if record_all_predictions and prediction_path is not None:
            for row in _build_debug_rows(scored_batch=scored, split=split_name, epoch=epoch, step=step, max_examples=len(scored)):
                with prediction_path.open("a", encoding="utf-8") as out_fp:
                    write_jsonl_line(out_fp, row)
                if prediction_csv_path is not None:
                    _append_csv_row(prediction_csv_path, row)
        if debug_path is not None and debug_max_examples > 0:
            for row in _build_debug_rows(scored_batch=scored, split=split_name, epoch=epoch, step=step, max_examples=debug_max_examples):
                with debug_path.open("a", encoding="utf-8") as out_fp:
                    write_jsonl_line(out_fp, row)
                if debug_csv_path is not None:
                    _append_csv_row(debug_csv_path, row)

    top1_acc = float(correct / len(examples)) if examples else 0.0
    pos_acc = float(pos_correct / pos_total) if pos_total else 0.0
    none_acc = float(none_correct / none_total) if none_total else 0.0
    balanced_acc = (pos_acc + none_acc) / 2.0 if (pos_total and none_total) else top1_acc
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    labels_for_macro = sorted(set(gold_counts.keys()) | set(pred_counts.keys()))
    per_label_f1: list[float] = []
    for label in labels_for_macro:
        label_tp = int(tp_by_label.get(label, 0))
        label_fp = max(0, int(pred_counts.get(label, 0)) - label_tp)
        label_fn = max(0, int(gold_counts.get(label, 0)) - label_tp)
        lp = float(label_tp / (label_tp + label_fp)) if (label_tp + label_fp) else 0.0
        lr = float(label_tp / (label_tp + label_fn)) if (label_tp + label_fn) else 0.0
        lf = float(2 * lp * lr / (lp + lr)) if (lp + lr) else 0.0
        per_label_f1.append(lf)
    macro_f1 = float(sum(per_label_f1) / len(per_label_f1)) if per_label_f1 else 0.0
    return {
        f"{split_name}_loss_mean": float(sum(losses) / len(losses)) if losses else 0.0,
        f"{split_name}_top1_acc": top1_acc,
        f"{split_name}_sample_acc": top1_acc,
        f"{split_name}_mrr": float(rr_sum / len(examples)) if examples else 0.0,
        f"{split_name}_pos_acc": pos_acc,
        f"{split_name}_none_acc": none_acc,
        f"{split_name}_balanced_acc": balanced_acc,
        f"{split_name}_positive_exact_precision": precision,
        f"{split_name}_positive_exact_recall": recall,
        f"{split_name}_positive_exact_f1": f1,
        f"{split_name}_precision": precision,
        f"{split_name}_recall": recall,
        f"{split_name}_f1": f1,
        f"{split_name}_macro_f1": macro_f1,
        f"{split_name}_acc_by_polyset": {k: float(acc_by_polyset_num[k] / acc_by_polyset_den[k]) for k in sorted(acc_by_polyset_den)},
        f"{split_name}_n_examples": len(examples),
    }


def run_train_bgroup_encoder_ce(*, cfg: dict[str, Any], run_context: RunContext) -> None:
    logger = logging.getLogger("kmwe")
    stage_name = "train_bgroup_encoder_ce"
    stage_cfg = cfg.get("bgroup_encoder_ce", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("train_bgroup_encoder_ce stage requires torch and transformers.") from exc

    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = outputs_dir / "train_bgroup_encoder_ce_progress.jsonl"
    progress_csv_path = outputs_dir / "train_bgroup_encoder_ce_progress.csv"
    dev_eval_path = outputs_dir / "train_bgroup_encoder_ce_dev_eval.jsonl"
    dev_eval_csv_path = outputs_dir / "train_bgroup_encoder_ce_dev_eval.csv"
    test_eval_path = outputs_dir / "train_bgroup_encoder_ce_test_eval.jsonl"
    test_eval_csv_path = outputs_dir / "train_bgroup_encoder_ce_test_eval.csv"
    test_predictions_path = outputs_dir / "test_predictions.jsonl"
    test_predictions_csv_path = outputs_dir / "test_predictions.csv"
    debug_path = outputs_dir / "debug_predictions.jsonl"
    debug_csv_path = outputs_dir / "debug_predictions.csv"
    input_samples_path = outputs_dir / "input_samples.jsonl"
    input_samples_csv_path = outputs_dir / "input_samples.csv"
    report_path = outputs_dir / "train_bgroup_encoder_ce_report.json"

    encoder_name = str(stage_cfg.get("encoder") or "").strip()
    tokenizer_name = str(stage_cfg.get("tokenizer") or encoder_name).strip()
    if not encoder_name:
        raise ConfigError("bgroup_encoder_ce.encoder가 비어 있습니다.")
    if not tokenizer_name:
        raise ConfigError("bgroup_encoder_ce.tokenizer가 비어 있습니다.")

    batch_size = max(1, int(stage_cfg.get("batch_size", 4) or 4))
    max_seq_len = max(8, int(stage_cfg.get("max_seq_len", 256) or 256))
    lr = float(stage_cfg.get("lr", 2e-5) or 2e-5)
    max_epochs = max(1, int(stage_cfg.get("epochs", stage_cfg.get("max_epochs", 5)) or 5))
    max_steps = max(1, int(stage_cfg.get("max_steps", 20000) or 20000))
    shuffle_enabled = bool(stage_cfg.get("shuffle", True))
    freeze_encoder = bool(stage_cfg.get("freeze_encoder", False))
    device = _resolve_device(str(runtime_cfg.get("device") or "auto"))
    mixed_precision = str(runtime_cfg.get("mixed_precision") or "auto")
    seed = int(runtime_cfg.get("seed", 42) or 42)
    progress_log_every_steps = max(1, int(stage_cfg.get("progress_log_every_steps", 20) or 20))

    debug_cfg = stage_cfg.get("debug_predictions", {}) or {}
    debug_enabled = bool(debug_cfg.get("enabled", True))
    debug_every_steps = max(1, int(debug_cfg.get("every_steps", 50) or 50))
    debug_max_examples = max(1, int(debug_cfg.get("max_examples_per_step", 1) or 1))

    dev_cfg = stage_cfg.get("dev_eval", {}) or {}
    dev_enabled = bool(dev_cfg.get("enabled", True))
    dev_every_epochs = max(1, int(dev_cfg.get("every_epochs", 1) or 1))

    test_cfg = stage_cfg.get("test_eval", {}) or {}
    test_enabled = bool(test_cfg.get("enabled", True))
    test_source = str(test_cfg.get("source") or "best").strip().lower()
    if test_source not in {"best", "last"}:
        raise ConfigError(f"bgroup_encoder_ce.test_eval.source는 best 또는 last여야 합니다: {test_source}")

    early_stop_cfg = stage_cfg.get("early_stop", {}) or {}
    early_stop_enabled = bool(early_stop_cfg.get("enabled", True))
    early_stop_metric = str(early_stop_cfg.get("metric") or "dev_f1")
    early_stop_direction = str(early_stop_cfg.get("direction") or "max").strip().lower()
    early_stop_patience = max(1, int(early_stop_cfg.get("patience", 3) or 3))

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    by_split, data_summary, sample_rows = load_bgroup_cross_encoder_examples(cfg=cfg, logger=logger)
    train_examples = by_split.get("train") or []
    dev_examples = by_split.get("dev") or []
    test_examples = by_split.get("test") or []
    if not train_examples:
        raise ConfigError("B-group encoder CE 학습용 train split이 비어 있습니다.")
    if dev_enabled and not dev_examples:
        raise ConfigError("B-group encoder CE dev_eval.enabled=true 이지만 dev split이 비어 있습니다.")
    if test_enabled and not test_examples:
        logger.warning("[bgroup_encoder_ce] test_eval.enabled=true 이지만 test split이 비어 있습니다.")
        test_enabled = False

    for row in sample_rows:
        with input_samples_path.open("a", encoding="utf-8") as out_fp:
            write_jsonl_line(out_fp, row)
        _append_csv_row(input_samples_csv_path, row)

    logger.info("[bgroup_encoder_ce][paths] encoder=%s tokenizer=%s outputs_dir=%s", encoder_name, tokenizer_name, outputs_dir)
    logger.info("[bgroup_encoder_ce][splits] train=%s dev=%s test=%s", len(train_examples), len(dev_examples), len(test_examples))

    model, tokenizer, scorer, added_special_tokens = _prepare_model_and_tokenizer(
        encoder_name=encoder_name,
        tokenizer_name=tokenizer_name,
        device=device,
    )
    if freeze_encoder:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    encoder_trainable_params = _count_trainable_params(model)
    head_trainable_params = _count_trainable_params(scorer)
    total_trainable_params = encoder_trainable_params + head_trainable_params

    optim_params = list(scorer.parameters()) if freeze_encoder else list(model.parameters()) + list(scorer.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=lr)

    wandb_run = None
    wandb_cfg = cfg.get("wandb", {}) or {}
    if bool(wandb_cfg.get("enabled", False)):
        try:
            import wandb

            init_kwargs = {
                "project": str(wandb_cfg.get("project") or "kmwe"),
                "mode": str(wandb_cfg.get("mode") or "online"),
                "name": str(run_context.run_id or ""),
                "group": str(wandb_cfg.get("group") or "") or None,
                "config": {
                    "stage": stage_name,
                    "encoder": encoder_name,
                    "tokenizer": tokenizer_name,
                    "input_mode": BGROUP_INPUT_MODE,
                    "span_marker_style": BGROUP_SPAN_MARKER_STYLE,
                    "text_b_format": BGROUP_TEXT_B_FORMAT,
                    "candidate_scoring": BGROUP_CANDIDATE_SCORING,
                    "freeze_encoder": freeze_encoder,
                    "batch_size": batch_size,
                    "max_seq_len": max_seq_len,
                    "lr": lr,
                    "epochs": max_epochs,
                    "max_steps": max_steps,
                },
            }
            entity = str(wandb_cfg.get("entity") or "").strip()
            if entity:
                init_kwargs["entity"] = entity
            wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            logger.warning("[bgroup_encoder_ce] wandb init skipped: %s", exc)
            wandb_run = None

    report: dict[str, Any] = {
        "stage": stage_name,
        "run_id": str(run_context.run_id or ""),
        "exp_id": str(cfg.get("exp", {}).get("exp_id") or "default"),
        "encoder_name": encoder_name,
        "tokenizer_name": tokenizer_name,
        "input_mode": BGROUP_INPUT_MODE,
        "span_marker_style": BGROUP_SPAN_MARKER_STYLE,
        "text_b_format": BGROUP_TEXT_B_FORMAT,
        "candidate_scoring": BGROUP_CANDIDATE_SCORING,
        "freeze_encoder": freeze_encoder,
        "encoder_trainable_params": encoder_trainable_params,
        "head_trainable_params": head_trainable_params,
        "total_trainable_params": total_trainable_params,
        "added_span_special_tokens": added_special_tokens,
        "data_summary": data_summary,
        "input_samples_path": str(input_samples_path),
        "input_samples_csv_path": str(input_samples_csv_path),
        "progress_path": str(progress_path),
        "progress_csv_path": str(progress_csv_path),
        "dev_eval_path": str(dev_eval_path),
        "dev_eval_csv_path": str(dev_eval_csv_path),
        "test_eval_path": str(test_eval_path),
        "test_eval_csv_path": str(test_eval_csv_path),
        "debug_predictions_path": str(debug_path),
        "debug_predictions_csv_path": str(debug_csv_path),
        "best_metric_name": early_stop_metric,
        "best_metric_value": None,
        "best_epoch": None,
        "best_step": None,
        "stop_reason": None,
        "test_metrics": None,
    }

    checkpoints_last_dir = outputs_dir / "checkpoints" / "last"
    checkpoints_best_dir = outputs_dir / "checkpoints" / "best"
    best_metric = None
    best_epoch = None
    best_step = None
    best_wait = 0
    loss_ema = None
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        if global_step >= max_steps:
            report["stop_reason"] = "max_steps_reached"
            break
        model.train(not freeze_encoder)
        scorer.train()
        for batch in _iter_batches(train_examples, batch_size, shuffle=shuffle_enabled, seed=seed + epoch):
            if global_step >= max_steps:
                report["stop_reason"] = "max_steps_reached"
                break
            scored = _score_batch(
                model=model,
                tokenizer=tokenizer,
                scorer=scorer,
                examples=batch,
                max_seq_len=max_seq_len,
                device=device,
                mixed_precision=mixed_precision,
                freeze_encoder=freeze_encoder,
            )
            if not scored:
                continue
            losses = []
            batch_correct = 0
            role_counter = Counter()
            pred_counter = Counter()
            avg_prob_by_role = defaultdict(list)
            for item in scored:
                ex = item["example"]
                logits = item["logits_with_none"]
                label = int(ex.get("label_index") or 0)
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], device=logits.device))
                losses.append(loss)
                probs = torch.softmax(logits.detach().float(), dim=0).tolist()
                pred_index = int(max(range(len(probs)), key=lambda i: probs[i]))
                batch_correct += int(pred_index == label)
                role = str(ex.get("gold_example_role") or "")
                role_counter[role] += 1
                pred_counter[_label_space(ex)[pred_index]] += 1
                avg_prob_by_role[role].append(float(max(probs)))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_value = float(loss.detach().cpu().item())
            loss_ema = loss_value if loss_ema is None else (0.9 * loss_ema + 0.1 * loss_value)

            if global_step % progress_log_every_steps == 0 or global_step == 1:
                progress_row = {
                    "timestamp": iso_now(),
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss_value,
                    "loss_ema": loss_ema,
                    "train_batch_size_examples": len(batch),
                    "train_batch_top1_acc": float(batch_correct / len(batch)) if batch else 0.0,
                    "role_counts_batch": dict(role_counter),
                    "pred_counts_batch": dict(pred_counter),
                    "avg_prob_by_role": {k: float(sum(v) / len(v)) for k, v in avg_prob_by_role.items() if v},
                }
                with progress_path.open("a", encoding="utf-8") as out_fp:
                    write_jsonl_line(out_fp, progress_row)
                _append_csv_row(progress_csv_path, progress_row)
                if wandb_run is not None:
                    wandb_run.log({f"train/{k}": v for k, v in progress_row.items() if k not in {"timestamp", "role_counts_batch", "pred_counts_batch", "avg_prob_by_role"}}, step=global_step)

            if debug_enabled and (global_step % debug_every_steps == 0):
                for row in _build_debug_rows(scored_batch=scored, split="train", epoch=epoch, step=global_step, max_examples=debug_max_examples):
                    with debug_path.open("a", encoding="utf-8") as fp:
                        write_jsonl_line(fp, row)
                    _append_csv_row(debug_csv_path, row)

        _save_checkpoint(model=model, tokenizer=tokenizer, scorer=scorer, ckpt_dir=checkpoints_last_dir)
        logger.info("[bgroup_encoder_ce] wrote checkpoint_last: %s", checkpoints_last_dir)

        if dev_enabled and epoch % dev_every_epochs == 0:
            dev_metrics = _evaluate_split(
                model=model,
                tokenizer=tokenizer,
                scorer=scorer,
                examples=dev_examples,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                device=device,
                mixed_precision=mixed_precision,
                split_name="dev",
                debug_path=debug_path if debug_enabled else None,
                debug_csv_path=debug_csv_path if debug_enabled else None,
                debug_max_examples=debug_max_examples if debug_enabled else 0,
                epoch=epoch,
                step=global_step,
                freeze_encoder=freeze_encoder,
            )
            dev_row = {
                "timestamp": iso_now(),
                "epoch": epoch,
                "step": global_step,
                **dev_metrics,
            }
            with dev_eval_path.open("a", encoding="utf-8") as out_fp:
                write_jsonl_line(out_fp, dev_row)
            _append_csv_row(dev_eval_csv_path, dev_row)
            if wandb_run is not None:
                wandb_run.log({f"dev/{k.replace('dev_', '')}": v for k, v in dev_metrics.items() if not isinstance(v, dict)}, step=global_step)

            current_metric = dev_metrics.get(early_stop_metric)
            if current_metric is None:
                raise ConfigError(f"early_stop.metric이 dev metric에 없습니다: {early_stop_metric}")
            is_better = (
                best_metric is None
                or (early_stop_direction == "max" and float(current_metric) > float(best_metric))
                or (early_stop_direction == "min" and float(current_metric) < float(best_metric))
            )
            if is_better:
                best_metric = float(current_metric)
                best_epoch = epoch
                best_step = global_step
                best_wait = 0
                _save_checkpoint(model=model, tokenizer=tokenizer, scorer=scorer, ckpt_dir=checkpoints_best_dir)
                logger.info("[bgroup_encoder_ce] wrote checkpoint_best: %s metric=%s epoch=%s step=%s", checkpoints_best_dir, best_metric, best_epoch, best_step)
            else:
                best_wait += 1
                if early_stop_enabled and best_wait >= early_stop_patience:
                    report["stop_reason"] = "early_stop"
                    break

    if report["stop_reason"] is None:
        report["stop_reason"] = "max_epochs_reached"

    report["best_metric_value"] = best_metric
    report["best_epoch"] = best_epoch
    report["best_step"] = best_step

    test_checkpoint_dir = checkpoints_best_dir if test_source == "best" else checkpoints_last_dir
    test_epoch_ref = best_epoch if test_source == "best" else epoch
    test_step_ref = best_step if test_source == "best" else global_step

    if test_enabled and test_checkpoint_dir.exists() and (test_checkpoint_dir / "head.pt").exists():
        test_model, test_tokenizer, test_scorer = _load_checkpoint(
            encoder_name=str(test_checkpoint_dir / "encoder"),
            tokenizer_name=str(test_checkpoint_dir / "tokenizer"),
            head_path=test_checkpoint_dir / "head.pt",
            device=device,
        )
        test_metrics = _evaluate_split(
            model=test_model,
            tokenizer=test_tokenizer,
            scorer=test_scorer,
            examples=test_examples,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
            mixed_precision=mixed_precision,
            split_name="test",
            debug_path=debug_path if debug_enabled else None,
            debug_csv_path=debug_csv_path if debug_enabled else None,
            debug_max_examples=debug_max_examples if debug_enabled else 0,
            prediction_path=test_predictions_path,
            prediction_csv_path=test_predictions_csv_path,
            record_all_predictions=True,
            epoch=int(test_epoch_ref or 0),
            step=int(test_step_ref or 0),
            freeze_encoder=freeze_encoder,
        )
        test_row = {
            "timestamp": iso_now(),
            "checkpoint_source": test_source,
            "checkpoint_dir": str(test_checkpoint_dir),
            "best_epoch": best_epoch,
            "best_step": best_step,
            "test_epoch_ref": test_epoch_ref,
            "test_step_ref": test_step_ref,
            **test_metrics,
        }
        with test_eval_path.open("a", encoding="utf-8") as out_fp:
            write_jsonl_line(out_fp, test_row)
        _append_csv_row(test_eval_csv_path, test_row)
        logger.info(
            "[bgroup_encoder_ce][test_eval][checkpoint_used] source=%s best_epoch=%s best_step=%s checkpoint_dir=%s",
            test_source,
            best_epoch,
            best_step,
            test_checkpoint_dir,
        )
        report["test_metrics"] = test_metrics
        report["test_predictions_path"] = str(test_predictions_path)
        report["test_predictions_csv_path"] = str(test_predictions_csv_path)
        report["test_eval_best_epoch"] = best_epoch
        report["test_eval_best_step"] = best_step
        report["test_eval_source"] = test_source
        report["test_eval_epoch_ref"] = test_epoch_ref
        report["test_eval_step_ref"] = test_step_ref
        if wandb_run is not None:
            wandb_run.log({f"test/{k.replace('test_', '')}": v for k, v in test_metrics.items() if not isinstance(v, dict)}, step=int(test_step_ref or global_step))
    else:
        report["test_eval_best_epoch"] = None
        report["test_eval_best_step"] = None
        report["test_eval_source"] = test_source
        report["test_eval_epoch_ref"] = None
        report["test_eval_step_ref"] = None

    if wandb_run is not None:
        try:
            wandb_run.summary.update(
                {
                    "input_mode": BGROUP_INPUT_MODE,
                    "span_marker_style": BGROUP_SPAN_MARKER_STYLE,
                    "text_b_format": BGROUP_TEXT_B_FORMAT,
                    "candidate_scoring": BGROUP_CANDIDATE_SCORING,
                    "freeze_encoder": freeze_encoder,
                    "best_metric_name": early_stop_metric,
                    "best_metric_value": best_metric,
                    "best_epoch": best_epoch,
                    "best_step": best_step,
                    "progress_path": str(progress_path),
                    "dev_eval_path": str(dev_eval_path),
                    "test_eval_path": str(test_eval_path),
                    "test_predictions_path": str(test_predictions_path),
                    "debug_predictions_path": str(debug_path),
                    "input_samples_path": str(input_samples_path),
                }
            )
            report["wandb_run_id"] = getattr(wandb_run, "id", None)
            report["wandb_run_url"] = getattr(wandb_run, "url", None)
            wandb_run.finish()
        except Exception:
            pass

    write_json(report_path, report, indent=2)
