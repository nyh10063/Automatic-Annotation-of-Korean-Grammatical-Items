#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_a_infer.sh #     reviewer_inputs/a_input.csv #     outputs/a_run #     /path/to/a_best_dir #     /path/to/expredict.xlsx
#
# Expected A best dir layout:
#   <A_BEST_DIR>/
#     head.pt
#     encoder/
#     tokenizer/

INPUT_CSV="${1:-reviewer_inputs/a_input.csv}"
OUTPUT_DIR="${2:-outputs/a_run}"
A_BEST_DIR="${3:-checkpoints/a_best}"
DICT_XLSX="${4:-data/dict/expredict.xlsx}"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[ERROR] input csv not found: $INPUT_CSV" >&2
  exit 1
fi
if [[ ! -d "$A_BEST_DIR" ]]; then
  echo "[ERROR] A best checkpoint dir not found: $A_BEST_DIR" >&2
  exit 1
fi
if [[ ! -f "$DICT_XLSX" ]]; then
  echo "[ERROR] dict xlsx not found: $DICT_XLSX" >&2
  exit 1
fi

A_ENCODER_DIR="${A_BEST_DIR}/encoder"
A_TOKENIZER_DIR="${A_BEST_DIR}/tokenizer"
A_HEAD_PT="${A_BEST_DIR}/head.pt"

if [[ ! -d "$A_ENCODER_DIR" ]]; then
  echo "[ERROR] encoder dir not found: $A_ENCODER_DIR" >&2
  exit 1
fi
if [[ ! -d "$A_TOKENIZER_DIR" ]]; then
  echo "[ERROR] tokenizer dir not found: $A_TOKENIZER_DIR" >&2
  exit 1
fi
if [[ ! -f "$A_HEAD_PT" ]]; then
  echo "[ERROR] head.pt not found: $A_HEAD_PT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Running A pipeline"
echo "[INFO] input_csv=$INPUT_CSV"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] a_best_dir=$A_BEST_DIR"
echo "[INFO] dict_xlsx=$DICT_XLSX"

python3 scripts/run_a_public.py   --input_csv "$INPUT_CSV"   --output_dir "$OUTPUT_DIR"   --best_dir "$A_BEST_DIR"   --dict_xlsx "$DICT_XLSX"

echo "[OK] A pipeline finished"
