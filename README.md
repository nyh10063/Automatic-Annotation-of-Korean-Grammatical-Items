# KMWE Reviewer Release

This repository provides a minimal reviewer-facing release of the A/B pipeline inference and evaluation described in the paper.

## Scope

- A-pipeline inference
- A-pipeline evaluation
- B-pipeline inference
- B-pipeline evaluation
- reviewer-provided input support

## Quick Start

1. Install dependencies from `requirements.txt`.
2. Place reviewer input files under `reviewer_inputs/`.
3. Run `scripts/check_env.sh`.
4. Run `scripts/run_a_infer.sh` or `scripts/run_b_infer.sh`.
5. Run `scripts/run_a_eval.sh` or `scripts/run_b_eval.sh` if gold labels are available.

## Repository Layout

- `kmwe/`: core code
- `configs/`: minimal public configs
- `scripts/`: reviewer entrypoints
- `docs/`: input and reproduction guides
- `examples/`: sample inputs and expected outputs
- `reviewer_inputs/`: place reviewer data here
- `outputs/`: generated predictions and evaluation outputs
- `checkpoints/`: place or download model checkpoints here

## Notes

This release is intended for inference and evaluation on reviewer-provided data. Training assets and full research data are not included in this repository.
