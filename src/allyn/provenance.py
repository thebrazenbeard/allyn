"""Bounded, caller-declared corpus provenance and partition manifest.

Opaque IDs and license references are *declarations*, not authenticated
identity, legal clearance, possession of a license, or access permission.
The module does not open files, access hosts, or authorize corpus use. A data
steward and applicable rights holder must independently approve actual use.
Identifiers are restricted to an opaque ASCII form; callers must still avoid
putting secrets or personal information into those identifiers because no
lexical validator can recognize every credential or sensitive label.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Literal


SCHEMA = "allyn.provenance-manifest.v0"
CLAIM_CEILING = "CALLER_DECLARED_RIGHTS_AND_SPLITS_NOT_VERIFIED"
_MAX_RECORDS = 10_000
_MAX_ID_CHARS = 256
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z", re.ASCII)
_USES = {"train": "training", "dev": "selection", "heldout": "evaluation"}

Split = Literal["train", "dev", "heldout"]
IntendedUse = Literal["training", "selection", "evaluation"]


def _opaque_id(value: object, field: str) -> str:
    """Accept short opaque labels, not paths, URLs, emails, or freeform text."""
    if type(value) is not str or not 1 <= len(value) <= _MAX_ID_CHARS:
        raise ValueError(f"{field} must be a nonempty opaque ID up to {_MAX_ID_CHARS} characters")
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} must be an ASCII opaque ID, not a path or private text")
    return value


def _split(value: object) -> Split:
    if type(value) is not str or value not in _USES:
        raise ValueError("split must be train, dev, or heldout")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """A declaration about a source, not evidence of legal authorization."""

    source_group_id: str
    record_id: str
    license_record_id: str
    split: Split
    intended_use: IntendedUse
    excluded: bool = False


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Immutable detached metadata and an unauthenticated consistency digest."""

    records: tuple[ProvenanceRecord, ...]
    manifest_sha256: str
    claim_ceiling: str = CLAIM_CEILING
    schema: str = SCHEMA

    def _checked(self) -> ProvenanceManifest:
        """Revalidate externally constructed descriptors before public use.

        This is only structural/digest consistency, never authentication.
        """
        if self.schema != SCHEMA or self.claim_ceiling != CLAIM_CEILING:
            raise ValueError("unsupported provenance manifest schema or claim ceiling")
        checked = build_manifest(self.records)
        if checked.manifest_sha256 != self.manifest_sha256:
            raise ValueError("provenance manifest digest mismatch")
        return checked

    def included(self, split: Split) -> tuple[ProvenanceRecord, ...]:
        """Return only declared non-excluded records for one split."""
        target = _split(split)
        checked = self._checked()
        return tuple(r for r in checked.records if r.split == target and not r.excluded)

    def snapshot(self) -> dict[str, object]:
        """Detached JSON-compatible export; the digest does not authenticate it."""
        checked = self._checked()
        return {
            "schema": checked.schema,
            "claim_ceiling": checked.claim_ceiling,
            "manifest_sha256": checked.manifest_sha256,
            "records": [
                {
                    "source_group_id": r.source_group_id,
                    "record_id": r.record_id,
                    "license_record_id": r.license_record_id,
                    "split": r.split,
                    "intended_use": r.intended_use,
                    "excluded": r.excluded,
                }
                for r in checked.records
            ],
        }


def build_manifest(
    records: Iterable[ProvenanceRecord], *, max_records: int = _MAX_RECORDS
) -> ProvenanceManifest:
    """Validate bounded declarations and disjoint source-family partitions.

    Every record requires a *claimed* license reference. No license is looked
    up or interpreted, and even a complete manifest cannot grant any rights.
    The source family is confined to one declared split, including excluded
    records, so one excluded member cannot conceal a cross-split collision.
    """
    if type(max_records) is not int or not 1 <= max_records <= _MAX_RECORDS:
        raise ValueError(f"max_records must be an integer from 1 to {_MAX_RECORDS}")
    if isinstance(records, (str, bytes, bytearray)):
        raise TypeError("records must be an iterable of ProvenanceRecord")
    try:
        source = iter(records)
    except TypeError as exc:
        raise TypeError("records must be an iterable of ProvenanceRecord") from exc

    normalized: list[ProvenanceRecord] = []
    seen_records: set[str] = set()
    split_by_family: dict[str, Split] = {}
    for index, record in enumerate(source):
        if index >= max_records:
            raise ValueError(f"provenance manifest exceeds {max_records} records")
        if type(record) is not ProvenanceRecord:
            raise TypeError(f"records[{index}] must be ProvenanceRecord")
        family = _opaque_id(record.source_group_id, "source_group_id")
        identifier = _opaque_id(record.record_id, "record_id")
        license_ref = _opaque_id(record.license_record_id, "license_record_id")
        split = _split(record.split)
        if type(record.intended_use) is not str or record.intended_use != _USES[split]:
            raise ValueError(f"intended_use must be {_USES[split]} for split={split}")
        if type(record.excluded) is not bool:
            raise ValueError("excluded must be bool")
        if identifier in seen_records:
            raise ValueError("duplicate record_id in provenance manifest")
        assigned = split_by_family.get(family)
        if assigned is not None and assigned != split:
            raise ValueError("source_group_id may not cross declared splits")
        seen_records.add(identifier)
        split_by_family[family] = split
        normalized.append(
            ProvenanceRecord(family, identifier, license_ref, split, record.intended_use, record.excluded)
        )
    if not normalized:
        raise ValueError("provenance manifest must contain at least one record")

    ordered = tuple(sorted(normalized, key=lambda r: (r.source_group_id, r.record_id)))
    payload = {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "records": [
            {
                "source_group_id": r.source_group_id,
                "record_id": r.record_id,
                "license_record_id": r.license_record_id,
                "split": r.split,
                "intended_use": r.intended_use,
                "excluded": r.excluded,
            }
            for r in ordered
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return ProvenanceManifest(ordered, sha256(canonical.encode("ascii")).hexdigest())
