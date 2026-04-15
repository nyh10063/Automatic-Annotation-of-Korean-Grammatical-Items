# Checkpoints

Model files are not committed to this repository.

## A Pipeline

A checkpoint is available on Hugging Face:

```text
nyh1006/kmwe-a-pipeline-encoder
```

Expected local layout:

```text
checkpoints/a_best/
├─ encoder/
├─ tokenizer/
└─ head.pt
```

Example:

```bash
hf download nyh1006/kmwe-a-pipeline-encoder \
  --local-dir checkpoints/a_best
```

## B Pipeline

B uses a fine-tuned Qwen-based LLM checkpoint. Provide the model and tokenizer directories explicitly when running `scripts/run_b_infer.sh`.

Expected arguments:

```text
/path/to/b_model
/path/to/b_tokenizer
```

If using Hugging Face, download the B checkpoint first and pass the downloaded `model` and `tokenizer` paths to the script.
