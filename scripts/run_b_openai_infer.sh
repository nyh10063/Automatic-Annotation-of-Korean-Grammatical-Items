#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

INPUT_CSV="${1:-reviewer_inputs/b_input.csv}"
OUTPUT_DIR="${2:-outputs/b_openai_run}"
DICT_XLSX="${3:-data/dict/expredict.xlsx}"
OPENAI_MODEL_NAME="${OPENAI_MODEL:-gpt-4.1}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set." >&2
  echo "Set it before running, for example:" >&2
  echo "  export OPENAI_API_KEY='your_api_key'" >&2
  echo "Do not commit API keys to GitHub." >&2
  exit 1
fi
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[ERROR] input csv not found: $INPUT_CSV" >&2
  exit 1
fi
if [[ ! -f "$DICT_XLSX" ]]; then
  echo "[ERROR] dict xlsx not found: $DICT_XLSX" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Running optional B OpenAI pipeline"
echo "[INFO] input_csv=$INPUT_CSV"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] openai_model=$OPENAI_MODEL_NAME"
echo "[INFO] dict_xlsx=$DICT_XLSX"

python3 scripts/run_b_openai_public.py \
  --input_csv "$INPUT_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --dict_xlsx "$DICT_XLSX" \
  --model "$OPENAI_MODEL_NAME"

echo "[OK] Optional B OpenAI pipeline finished"
