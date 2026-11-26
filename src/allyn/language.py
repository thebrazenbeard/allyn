"""Bounded, deterministic n-gram statistics for ALLYN's first language slice.

This is an online statistical predictor, not grammar induction, neural training,
or a claim of semantic understanding. Each ``observe`` call is one utterance;
the learner never infers transitions across calls. Predictions use interpolated
additive smoothing and suffix backoff over the observed token vocabulary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from math import fsum
from typing import Any


_SNAPSHOT_SCHEMA = "allyn.ngram.v1"
_MAX_ORDER = 5
_MAX_TOKEN_CHARS = 4096
_MAX_UTTERANCE_TOKENS = 10_000
_MAX_CONTEXT_TOKENS = 100_000
_MAX_CONTEXT_ROWS = 100_000
_MAX_CONTINUATIONS = 1_000_000
_MAX_COUNT = (1 << 63) - 1


def _validate_order(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_ORDER:
        raise ValueError(f"order must be an integer from 1 to {_MAX_ORDER}")
    return value


def _validate_token(value: object) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TOKEN_CHARS:
        raise ValueError("tokens must be non-empty strings of at most 4096 characters")
    return value


def _validate_tokens(value: object, *, limit: int, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of tokens, not a string")
    if len(value) > limit:
        raise ValueError(f"{label} exceeds the {limit}-token limit")
    return tuple(_validate_token(token) for token in value)


class NGramLearner:
    """Variable-order count model with normalized interpolation/backoff.

    ``order`` is the largest n-gram size, including the predicted token.
    Count updates are atomic with respect to input validation and capacity
    failures, but this object does not provide concurrent-call isolation.
    """

    def __init__(self, order: int = 3) -> None:
        self.order = _validate_order(order)
        self._counts: dict[tuple[str, ...], Counter[str]] = {}
        self._continuations = 0

    def observe(self, tokens: Sequence[str]) -> None:
        """Learn one utterance, resetting all left context at its boundary."""
        words = _validate_tokens(tokens, limit=_MAX_UTTERANCE_TOKENS, label="tokens")
        if not words:
            return

        # Build a complete batch before touching existing state.
        changes: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for index, word in enumerate(words):
            for context_size in range(min(index, self.order - 1) + 1):
                context = words[index - context_size : index]
                changes[context][word] += 1

        additional_rows = sum(context not in self._counts for context in changes)
        if len(self._counts) + additional_rows > _MAX_CONTEXT_ROWS:
            raise ValueError("ngram context capacity exceeded")

        additional_continuations = 0
        for context, updates in changes.items():
            existing = self._counts.get(context, Counter())
            additional_continuations += sum(word not in existing for word in updates)
            if sum(existing.values()) + sum(updates.values()) > _MAX_COUNT:
                raise ValueError("ngram count capacity exceeded")
            for word, increment in updates.items():
                if existing[word] + increment > _MAX_COUNT:
                    raise ValueError("ngram count capacity exceeded")
        if self._continuations + additional_continuations > _MAX_CONTINUATIONS:
            raise ValueError("ngram continuation capacity exceeded")

        for context, updates in changes.items():
            self._counts.setdefault(context, Counter()).update(updates)
        self._continuations += additional_continuations

    def next_distribution(self, context: Sequence[str]) -> dict[str, float]:
        """Return positive, normalized probabilities for observed token types.

        A missing high-order context backs off to its shorter suffix. An
        observed context interpolates its empirical counts with the shorter
        distribution using one unit of prior strength; unigrams use add-one
        smoothing. No prediction includes unobserved vocabulary items.
        """
        words = _validate_tokens(context, limit=_MAX_CONTEXT_TOKENS, label="context")
        unigram = self._counts.get(())
        if not unigram:
            return {}

        vocabulary = sorted(unigram)
        base_total = sum(unigram.values()) + len(vocabulary)
        distribution = {word: (unigram[word] + 1.0) / base_total for word in vocabulary}

        tail = words[-(self.order - 1) :] if self.order > 1 else ()
        for context_size in range(1, len(tail) + 1):
            continuation = self._counts.get(tail[-context_size:])
            if not continuation:
                continue
            total = sum(continuation.values()) + 1.0
            distribution = {
                word: (continuation.get(word, 0) + distribution[word]) / total
                for word in vocabulary
            }

        # Compensate for floating-point accumulation without changing ranking.
        normalizer = fsum(distribution.values())
        return {word: probability / normalizer for word, probability in distribution.items()}

    def predict_next(self, context: Sequence[str]) -> str | None:
        """Return the most likely observed token (lexical tie-break)."""
        distribution = self.next_distribution(context)
        if not distribution:
            return None
        # next_distribution inserts sorted tokens; max preserves first on ties.
        return max(distribution, key=distribution.__getitem__)

    def snapshot(self) -> dict[str, Any]:
        """Return detached, canonical, JSON-compatible count state."""
        rows = [
            {
                "context": list(context),
                "next": {word: counts[word] for word in sorted(counts)},
            }
            for context, counts in sorted(self._counts.items(), key=lambda item: (len(item[0]), item[0]))
        ]
        return {"schema": _SNAPSHOT_SCHEMA, "order": self.order, "rows": rows}

    @classmethod
    def from_snapshot(cls, data: object) -> NGramLearner:
        """Restore bounded state satisfying partial structural constraints.

        Snapshot bytes have no authentication; provenance and permission checks
        must be supplied separately by an integrating runtime. Passing these
        necessary suffix-count checks does not prove the counts arose from any
        sequence of ``observe`` calls: whole-corpus realizability is unchecked.
        """
        if type(data) is not dict or set(data) != {"schema", "order", "rows"}:
            raise ValueError("invalid ngram snapshot object")
        if type(data["schema"]) is not str or data["schema"] != _SNAPSHOT_SCHEMA:
            raise ValueError("unsupported ngram snapshot schema")
        learner = cls(order=data["order"])
        rows = data["rows"]
        if type(rows) is not list or len(rows) > _MAX_CONTEXT_ROWS:
            raise ValueError("invalid ngram snapshot rows")

        continuations = 0
        for row in rows:
            if type(row) is not dict or set(row) != {"context", "next"}:
                raise ValueError("invalid ngram snapshot row")
            context_data = row["context"]
            if type(context_data) is not list or len(context_data) >= learner.order:
                raise ValueError("invalid ngram context")
            context = tuple(_validate_token(token) for token in context_data)
            if context in learner._counts:
                raise ValueError("duplicate ngram context")
            next_data = row["next"]
            if type(next_data) is not dict or not next_data:
                raise ValueError("invalid ngram continuation counts")
            continuations += len(next_data)
            if continuations > _MAX_CONTINUATIONS:
                raise ValueError("ngram continuation capacity exceeded")

            count_total = 0
            counts: Counter[str] = Counter()
            for token, count in next_data.items():
                _validate_token(token)
                if type(count) is not int or not 1 <= count <= _MAX_COUNT:
                    raise ValueError("ngram counts must be positive bounded integers")
                count_total += count
                if count_total > _MAX_COUNT:
                    raise ValueError("ngram count capacity exceeded")
                counts[token] = count
            learner._counts[context] = counts

        if learner._counts:
            unigram = learner._counts.get(())
            if unigram is None:
                raise ValueError("missing unigram counts")
            vocabulary = set(unigram)
            for context in learner._counts:
                if not set(context).issubset(vocabulary):
                    raise ValueError("context contains an unobserved token")

            # Every observed higher-order transition contributes to its suffix
            # transition. This is a necessary, NOT sufficient, constraint:
            # a forged unigram {a: 1} plus bigram a -> a: 1 passes, although
            # that transition requires at least two observed a tokens.
            for size in range(1, learner.order):
                suffix_totals: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
                for context, counts in learner._counts.items():
                    if len(context) == size:
                        suffix_totals[context[1:]].update(counts)
                for suffix, totals in suffix_totals.items():
                    parent = learner._counts.get(suffix)
                    if parent is None:
                        raise ValueError("missing lower-order context")
                    if any(count > parent[token] for token, count in totals.items()):
                        raise ValueError("higher-order counts exceed lower-order evidence")

        learner._continuations = continuations
        return learner
