# Input Format

Both A and B pipelines use the same reviewer input format.

## Required CSV Column

Only one column is required:

```csv
sentence
"나도 언니처럼 예쁘다면 참 좋을 텐데."
"저는 제주도에 간 적이 있습니다."
```

The runner automatically assigns `id = 1, 2, 3, ...` in the output.

## Commas in Sentences

If a sentence contains a comma, the safest CSV format is to quote the whole sentence:

```csv
sentence
"올해 초에는 해내고 말겠다고 했지만, 쉽지 않았다."
```

The runner tries to recover unquoted comma-containing sentences, but standard CSV quoting is recommended.

## Not Required

Do not add these columns for reviewer inference:

- `id`
- `target`
- gold labels

Gold labels are only needed for separate evaluation scripts.
