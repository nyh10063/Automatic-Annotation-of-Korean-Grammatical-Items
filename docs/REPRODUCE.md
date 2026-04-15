# Reproduce Reviewer Runs

## Sample run

1. Run `scripts/check_env.sh`
2. Run `scripts/run_a_infer.sh`
3. Run `scripts/run_b_infer.sh`
4. If gold labels are available, run the corresponding eval scripts.

## Reviewer run

1. Put your input files under `reviewer_inputs/`
2. Run the corresponding pipeline script
3. Check outputs under `outputs/`
