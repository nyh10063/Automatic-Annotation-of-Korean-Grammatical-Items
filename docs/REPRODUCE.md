# Reproduce Reviewer Runs

This document describes the minimal inference workflow.

## 1. Prepare Input

Create a CSV file with a single `sentence` column.

```csv
sentence
"나도 언니처럼 예쁘다면 참 좋을 텐데."
"저는 제주도에 간 적이 있습니다."
```

Place it under `reviewer_inputs/`, or pass its path directly to the scripts.

## 2. Prepare Checkpoints

A pipeline expects:

```text
checkpoints/a_best/encoder
checkpoints/a_best/tokenizer
checkpoints/a_best/head.pt
```

B pipeline expects a model directory and a tokenizer directory. These may be Hugging Face downloads or Google Drive paths.

## 3. Run A

```bash
bash scripts/run_a_infer.sh \
  reviewer_inputs/a_input.csv \
  outputs/a_run \
  checkpoints/a_best \
  data/dict/expredict_public.xlsx
```

## 4. Run B

```bash
bash scripts/run_b_infer.sh \
  reviewer_inputs/b_input.csv \
  outputs/b_run \
  /path/to/b_model \
  /path/to/b_tokenizer \
  data/dict/expredict_public.xlsx
```

## 5. Inspect Results

Reviewer-facing output:

```text
outputs/a_run/predictions.csv
outputs/b_run/predictions.csv
```

Debugging output:

```text
predictions.jsonl
debug_detection.jsonl
summary.json
```
