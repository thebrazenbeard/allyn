"""Advisory text-overlap audit for caller-declared TRAIN/DEV/HELDOUT splits.

This is *not* a semantic deduplicator or independent data custodian. The
caller supplies all records and labels; source IDs cannot be authenticated
here. The deterministic lexical heuristic can miss paraphrases and flag
legitimate boilerplate. Its findings are warnings to review, not judgments
about teaching efficacy or license rights.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import re
import unicodedata

from .evaluation import SourcedExample


SCHEMA = "allyn.split-overlap-audit.v0"
CLAIM_CEILING = "CALLER_DECLARED_LEXICAL_OVERLAP_ONLY"
_MAX_EXAMPLES_PER_PARTITION = 256
_MAX_WORDS_PER_EXAMPLE = 256
_MAX_TOTAL_WORDS = 40_000
_MAX_TOKEN_CHARACTERS = 4_096
_MAX_ID_CHARACTERS = 256
_MAX_PARTITION_UTF8_BYTES = 1_000_000
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class SplitCollision:
    left_partition: str
    left_source_id: str
    left_record_id: str
    right_partition: str
    right_source_id: str
    right_record_id: str
    reason: str
    token_dice: float
    shingle_jaccard: float


@dataclass(frozen=True, slots=True)
class SplitAudit:
    """Sorted deterministic warnings and the number of comparisons made."""

    train_count: int
    dev_count: int
    heldout_count: int
    comparisons: int
    collisions: tuple[SplitCollision, ...]
    schema: str = SCHEMA
    claim_ceiling: str = CLAIM_CEILING

    @property
    def has_indicated_overlap(self) -> bool:
        return bool(self.collisions)


@dataclass(frozen=True, slots=True)
class _Entry:
    source_id: str
    record_id: str
    words: tuple[str, ...]
    bag: Counter[str]
    shingles: frozenset[tuple[str, ...]]


def _normalize(words: tuple[str, ...]) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", " ".join(words)).casefold()
    return tuple(_WORD_RE.findall(text))


def _ingest(values: Iterable[SourcedExample], name: str, shingle_size: int) -> tuple[_Entry, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must contain SourcedExample values")
    try:
        it = iter(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable") from exc

    results: list[_Entry] = []
    partition_bytes = 0
    for index, item in enumerate(it):
        if index >= _MAX_EXAMPLES_PER_PARTITION:
            raise ValueError(f"{name} exceeds {_MAX_EXAMPLES_PER_PARTITION} examples")
        if type(item) is not SourcedExample:
            raise TypeError(f"{name}[{index}] must be SourcedExample")
        for label, identifier in (("source_id", item.source_id), ("record_id", item.record_id)):
            if (type(identifier) is not str or not identifier.strip()
                    or len(identifier) > _MAX_ID_CHARACTERS):
                raise ValueError(f"{name}[{index}].{label} must be nonblank and bounded")
            try:
                identifier.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{name}[{index}].{label} must be UTF-8 encodable") from exc
        if isinstance(item.tokens, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{index}].tokens must be a token iterable")
        try:
            token_iterator = iter(item.tokens)
        except TypeError as exc:
            raise TypeError(f"{name}[{index}].tokens must be iterable") from exc
        source_words: list[str] = []
        for token_index, token in enumerate(token_iterator):
            if token_index >= _MAX_WORDS_PER_EXAMPLE:
                raise ValueError(f"{name}[{index}] exceeds {_MAX_WORDS_PER_EXAMPLE} tokens")
            if type(token) is not str or not token or len(token) > _MAX_TOKEN_CHARACTERS:
                raise ValueError(f"{name}[{index}] contains invalid or oversized token")
            try:
                token_bytes = len(token.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ValueError(f"{name}[{index}] contains non-UTF-8 text") from exc
            partition_bytes += token_bytes
            if partition_bytes > _MAX_PARTITION_UTF8_BYTES:
                raise ValueError(f"{name} exceeds {_MAX_PARTITION_UTF8_BYTES} input bytes")
            source_words.append(token)
        words = _normalize(tuple(source_words))
        if not words:
            raise ValueError(f"{name}[{index}] has no normalized words")
        if len(words) > _MAX_WORDS_PER_EXAMPLE:
            raise ValueError(f"{name}[{index}] exceeds {_MAX_WORDS_PER_EXAMPLE} normalized words")
        shingles = frozenset(tuple(words[n:n + shingle_size])
                             for n in range(len(words) - shingle_size + 1))
        results.append(_Entry(item.source_id, item.record_id, words, Counter(words), shingles))
    return tuple(sorted(results, key=lambda row: (row.source_id, row.record_id, row.words)))


def _token_dice(left: _Entry, right: _Entry) -> float:
    shared = sum((left.bag & right.bag).values())
    return (2.0 * shared) / (len(left.words) + len(right.words))


def _shingle_jaccard(left: _Entry, right: _Entry) -> float:
    combined = left.shingles | right.shingles
    if not combined:
        return 0.0
    return len(left.shingles & right.shingles) / len(combined)


def audit_partition_overlap(
    train: Iterable[SourcedExample],
    dev: Iterable[SourcedExample],
    heldout: Iterable[SourcedExample],
    *,
    token_threshold: float = 0.85,
    shingle_size: int = 3,
    min_words_for_near: int = 6,
    shingle_threshold: float = 0.35,
    check_within_heldout: bool = True,
) -> SplitAudit:
    """Return potential cross-split leaks and heldout self-repeats.

    Exact comparisons use normalized text, including case and punctuation
    folding. A near match needs both a high token-multiset Dice coefficient
    and enough contiguous n-gram overlap. This heuristic is deliberately
    conservative about reordering; paraphrases can evade it. Short repeated
    prompts may be benign and require human inspection.
    """
    if type(token_threshold) not in (int, float) or not 0 < token_threshold <= 1:
        raise ValueError("token_threshold must be in (0, 1]")
    if type(shingle_threshold) not in (int, float) or not 0 < shingle_threshold <= 1:
        raise ValueError("shingle_threshold must be in (0, 1]")
    if type(shingle_size) is not int or not 2 <= shingle_size <= 8:
        raise ValueError("shingle_size must be an integer between 2 and 8")
    if type(min_words_for_near) is not int or not 2 <= min_words_for_near <= 100:
        raise ValueError("min_words_for_near must be an integer from 2 through 100")
    if type(check_within_heldout) is not bool:
        raise TypeError("check_within_heldout must be bool")

    partitions = {
        "train": _ingest(train, "train", shingle_size),
        "dev": _ingest(dev, "dev", shingle_size),
        "heldout": _ingest(heldout, "heldout", shingle_size),
    }
    if sum(len(row.words) for items in partitions.values() for row in items) > _MAX_TOTAL_WORDS:
        raise ValueError(f"audit exceeds {_MAX_TOTAL_WORDS} normalized words")

    pairs = (("train", "dev"), ("train", "heldout"), ("dev", "heldout"))
    collisions: list[SplitCollision] = []
    comparisons = 0

    def compare(left_name: str, left: _Entry, right_name: str, right: _Entry) -> None:
        nonlocal comparisons
        comparisons += 1
        if left.words == right.words:
            reason, dice, jaccard = "NORMALIZED_EXACT", 1.0, 1.0
        elif min(len(left.words), len(right.words)) >= min_words_for_near:
            dice = _token_dice(left, right)
            if dice < token_threshold:
                return
            jaccard = _shingle_jaccard(left, right)
            if jaccard < shingle_threshold:
                return
            reason = "HIGH_LEXICAL_SIMILARITY"
        else:
            return
        collisions.append(SplitCollision(left_name, left.source_id, left.record_id,
                                         right_name, right.source_id, right.record_id,
                                         reason, dice, jaccard))

    for left_name, right_name in pairs:
        for left in partitions[left_name]:
            for right in partitions[right_name]:
                compare(left_name, left, right_name, right)
    if check_within_heldout:
        heldout_rows = partitions["heldout"]
        for index, left in enumerate(heldout_rows):
            for right in heldout_rows[index + 1:]:
                compare("heldout", left, "heldout", right)

    return SplitAudit(len(partitions["train"]), len(partitions["dev"]),
                      len(partitions["heldout"]), comparisons, tuple(collisions))
