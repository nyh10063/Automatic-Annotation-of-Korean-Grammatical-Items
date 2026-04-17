#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ "$#" -lt 4 ]; then
  echo "Usage: bash scripts/run_a_infer.sh <input_csv> <output_dir> <a_best_dir|auto> <dict_xlsx>" >&2
  exit 1
fi

INPUT_CSV="$1"
OUTPUT_DIR="$2"
A_BEST_DIR="$3"
DICT_XLSX="$4"

HF_REPO_ID="nyh1006/kmwe-a-pipeline-encoder"
AUTO_ROOT="checkpoints/hf_a"

if [ "$A_BEST_DIR" = "auto" ]; then
  echo "[INFO] downloading A checkpoint from Hugging Face: $HF_REPO_ID"
  python3 - <<'INNER'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nyh1006/kmwe-a-pipeline-encoder",
    local_dir="checkpoints/hf_a",
)
INNER

  if [ -d "$AUTO_ROOT/a_best" ]; then
    A_BEST_DIR="$AUTO_ROOT/a_best"
  elif [ -d "$AUTO_ROOT/a-best" ]; then
    A_BEST_DIR="$AUTO_ROOT/a-best"
  else
    echo "[ERROR] A best checkpoint dir not found after download under: $AUTO_ROOT" >&2
    exit 1
  fi
fi

if [ ! -f "$INPUT_CSV" ]; then
  echo "[ERROR] input csv not found: $INPUT_CSV" >&2
  exit 1
fi

if [ ! -d "$A_BEST_DIR" ]; then
  echo "[ERROR] A best checkpoint dir not found: $A_BEST_DIR" >&2
  exit 1
fi

if [ ! -f "$DICT_XLSX" ]; then
  echo "[ERROR] dict xlsx not found: $DICT_XLSX" >&2
  exit 1
fi

python3 scripts/run_a_public.py   --input_csv "$INPUT_CSV"   --output_dir "$OUTPUT_DIR"   --best_dir "$A_BEST_DIR"   --dict_xlsx "$DICT_XLSX"
