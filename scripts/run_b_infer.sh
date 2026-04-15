#!/usr/bin/env bash
set -euo pipefail

INPUT_CSV="${1:-reviewer_inputs/b_input.csv}"
OUTPUT_DIR="${2:-outputs/b_run}"
B_MODEL_DIR="${3:-checkpoints/b_model}"
B_TOKENIZER_DIR="${4:-checkpoints/b_tokenizer}"
DICT_XLSX="${5:-data/dict/expredict.xlsx}"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[ERROR] input csv not found: $INPUT_CSV" >&2
  exit 1
fi
if [[ ! -d "$B_MODEL_DIR" ]]; then
  echo "[ERROR] B model dir not found: $B_MODEL_DIR" >&2
  exit 1
fi
if [[ ! -d "$B_TOKENIZER_DIR" ]]; then
  echo "[ERROR] B tokenizer dir not found: $B_TOKENIZER_DIR" >&2
  exit 1
fi
if [[ ! -f "$DICT_XLSX" ]]; then
  echo "[ERROR] dict xlsx not found: $DICT_XLSX" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Running B pipeline"
echo "[INFO] input_csv=$INPUT_CSV"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] b_model_dir=$B_MODEL_DIR"
echo "[INFO] b_tokenizer_dir=$B_TOKENIZER_DIR"
echo "[INFO] dict_xlsx=$DICT_XLSX"

python3 scripts/run_b_llm_public.py   --input_csv "$INPUT_CSV"   --output_dir "$OUTPUT_DIR"   --model_dir "$B_MODEL_DIR"   --tokenizer_dir "$B_TOKENIZER_DIR"   --dict_xlsx "$DICT_XLSX"

echo "[OK] B pipeline finished"
