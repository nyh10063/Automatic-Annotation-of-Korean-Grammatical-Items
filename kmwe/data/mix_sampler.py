from __future__ import annotations

import random
from collections import Counter
from typing import Any, Iterator


class WeightedMixtureSampler:
    def __init__(
        self,
        iterators: dict[str, Iterator[Any]],
        weights: dict[str, float],
        *,
        seed: int,
        deterministic: bool = True,
    ) -> None:
        self.iterators = dict(iterators)
        self.random = random.Random(seed) if deterministic else random.Random()
        self._set_weights(weights)
        self.n_samples = 0
        self.counts_by_corpus: Counter[str] = Counter()

    def _set_weights(self, weights: dict[str, float]) -> None:
        missing = [k for k, v in weights.items() if v > 0 and k not in self.iterators]
        if missing:
            raise ValueError(f"weights에 정의된 corpus가 index에 없습니다: {missing}")
        self.weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
        self.corpora = sorted(self.weights.keys())
        self.weight_values = [self.weights[k] for k in self.corpora]

    def set_weights(self, weights: dict[str, float]) -> None:
        self._set_weights(weights)

    def sample(self) -> tuple[str, Any] | None:
        while self.corpora:
            corpus = self.random.choices(self.corpora, weights=self.weight_values, k=1)[0]
            iterator = self.iterators.get(corpus)
            if iterator is None:
                self._drop_corpus(corpus)
                continue
            try:
                item = next(iterator)
            except StopIteration:
                self._drop_corpus(corpus)
                continue
            self.n_samples += 1
            self.counts_by_corpus[corpus] += 1
            return corpus, item
        return None

    def _drop_corpus(self, corpus: str) -> None:
        if corpus in self.iterators:
            self.iterators.pop(corpus, None)
        if corpus in self.weights:
            self.weights.pop(corpus, None)
        if corpus in self.corpora:
            idx = self.corpora.index(corpus)
            self.corpora.pop(idx)
            self.weight_values.pop(idx)

    def observed_ratio(self) -> dict[str, float]:
        total = float(self.n_samples) if self.n_samples > 0 else 1.0
        return {k: v / total for k, v in self.counts_by_corpus.items()}
