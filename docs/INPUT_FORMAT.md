# Input Format

## A-pipeline input

Expected CSV columns:
- `id`
- `sentence`
- `target` or span-marked sentence depending on the released runner

## B-pipeline input

Expected CSV columns:
- `id`
- `sentence`
- task-specific target information as documented by the runner

Reviewer files should be placed under `reviewer_inputs/`.
