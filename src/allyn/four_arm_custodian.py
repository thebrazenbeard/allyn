"""One-use, local *four-arm* scoring with advisory heldout replay control.

This is a cooperating evaluator PROCESS boundary, not independent dataset
custody, verified license/source provenance, OS isolation, or an intelligence
claim. A trusted operator can supply the five private partitions over stdin
to one disposable process. The teacher must not receive that input. The same
heldout cases are scored against four already-declared lesson arms in one
evaluation, with only aggregate and per-group counts emitted.

The registry lives in a caller-selected local directory. It prevents exact
replay only when actors cooperate with that directory; same-account tampering,
partial/near-duplicate tests, alternate directories and adaptive pre-peeking
are not prevented. Full-specimen SHA-256 is deliberately omitted from public
receipts: a teacher could brute-force a digest of low-entropy heldout answers.
Even *aggregate or per-group scores* can leak labels for very small groups
if the observer knows the arm predictions. Receipts are for the authorized
evaluator after candidate/control freeze, not a teacher-facing private key.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from .custodian import (
    AlreadyEvaluatedError, _atomic_claim, _replace_receipt,
    _safe_request_id, _state_directory,
)
from .evaluation import SourcedExample
from .experiment import (
    ArmScore, FourArmSpecimen, evaluate_four_arm, freeze_four_arm_specimen,
)


REQUEST_SCHEMA = "allyn.four-arm-custody-request.v0"
RECEIPT_SCHEMA = "allyn.four-arm-custody-receipt.v0"
CLAIM_CEILING = "LOCAL_FOUR_ARM_PROCESS_BOUNDARY_AND_ADVISORY_REPLAY_REGISTRY_ONLY"
_MAX_STDIN_BYTES = 2_000_000


@dataclass(frozen=True, slots=True)
class AnonymousGroupScore:
    """Ordinal group only; private source/record IDs and answers withheld."""

    group_index: int
    arms: tuple[ArmScore, ...]


@dataclass(frozen=True, slots=True)
class FourArmCustodyReceipt:
    """One evaluator-facing receipt for all arms on the same heldout set.

    Counts are not a privacy guarantee for small or low-entropy test groups.
    """

    request_id: str
    arms: tuple[ArmScore, ...]
    groups: tuple[AnonymousGroupScore, ...]
    evaluated_count: int
    heldout_group_count: int
    size_gate: str
    status: str = "evaluated"
    schema: str = RECEIPT_SCHEMA
    claim_ceiling: str = CLAIM_CEILING


def _checked(specimen: FourArmSpecimen) -> FourArmSpecimen:
    if type(specimen) is not FourArmSpecimen:
        raise TypeError("specimen must be FourArmSpecimen")
    checked = freeze_four_arm_specimen(
        specimen.student,
        specimen.genuine,
        specimen.irrelevant,
        specimen.misleading,
        specimen.heldout,
        order=specimen.order,
    )
    if (
        specimen.schema != checked.schema
        or specimen.claim_ceiling != checked.claim_ceiling
        or specimen.specimen_sha256 != checked.specimen_sha256
    ):
        raise ValueError("invalid or modified four-arm specimen")
    return checked


def evaluate_four_arm_once(
    specimen: FourArmSpecimen,
    *,
    state_dir: os.PathLike[str] | str,
    request_id: str,
) -> FourArmCustodyReceipt:
    """Score all four arms with one exact heldout corpus, claiming it first.

    The heldout-token claim follows ``allyn.custodian``'s exact algorithm,
    sharing its state namespace. A previous one-arm or four-arm evaluation
    against these exact tokens blocks another pass through either interface.
    """
    requested = _safe_request_id(request_id)
    checked = _checked(specimen)
    directory = _state_directory(state_dir)
    public_path = directory / f"request-{requested}.json"
    if public_path.exists():
        raise AlreadyEvaluatedError("request ID already used")

    heldout_payload = json.dumps(
        sorted(row.tokens for row in checked.heldout),
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    heldout_digest = hashlib.sha256(heldout_payload).hexdigest()
    _atomic_claim(
        directory / f"heldout-{heldout_digest}.claim",
        {"schema": RECEIPT_SCHEMA, "status": "heldout_consumed"},
    )
    _atomic_claim(
        directory / f"specimen-{checked.specimen_sha256}.claim",
        {"schema": RECEIPT_SCHEMA, "status": "consumed"},
    )
    _atomic_claim(public_path, {
        "schema": RECEIPT_SCHEMA, "request_id": requested,
        "status": "reserved", "claim_ceiling": CLAIM_CEILING,
    })
    try:
        result = evaluate_four_arm(checked)
        receipt = FourArmCustodyReceipt(
            request_id=requested,
            arms=result.arms,
            groups=tuple(
                AnonymousGroupScore(group_index=index, arms=group.arms)
                for index, group in enumerate(result.group_scores, start=1)
            ),
            evaluated_count=result.arms[0].evaluated_count,
            heldout_group_count=result.heldout_group_count,
            size_gate=result.size_gate,
        )
        _replace_receipt(public_path, asdict(receipt))
        return receipt
    except Exception:
        # The exact heldout set remains consumed after a failure. Do not emit
        # private samples, tracebacks or arbitrary exception strings.
        _replace_receipt(public_path, {
            "schema": RECEIPT_SCHEMA, "request_id": requested,
            "status": "failed", "claim_ceiling": CLAIM_CEILING,
        })
        raise


def _wire_specimen(value: Any) -> FourArmSpecimen:
    if type(value) is not dict or set(value) != {
        "schema", "student", "genuine", "irrelevant", "misleading", "heldout", "order",
    } or value["schema"] != REQUEST_SCHEMA:
        raise ValueError("invalid four-arm request schema")

    def records(data: Any, label: str) -> list[SourcedExample]:
        if type(data) is not list:
            raise TypeError(f"{label} must be list")
        result: list[SourcedExample] = []
        for row in data:
            if type(row) is not dict or set(row) != {"source_id", "record_id", "tokens"}:
                raise TypeError(f"{label} contains invalid record")
            if type(row["tokens"]) is not list:
                raise TypeError(f"{label} tokens must be list")
            result.append(SourcedExample(row["source_id"], row["record_id"], tuple(row["tokens"])))
        return result

    return freeze_four_arm_specimen(
        records(value["student"], "student"),
        records(value["genuine"], "genuine"),
        records(value["irrelevant"], "irrelevant"),
        records(value["misleading"], "misleading"),
        records(value["heldout"], "heldout"),
        order=value["order"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot local ALLYN four-arm evaluator")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args(argv)
    try:
        payload = sys.stdin.buffer.read(_MAX_STDIN_BYTES + 1)
        if len(payload) > _MAX_STDIN_BYTES:
            raise ValueError("request size exceeded")
        specimen = _wire_specimen(json.loads(payload.decode("utf-8")))
        receipt = evaluate_four_arm_once(specimen, state_dir=args.state_dir, request_id=args.request_id)
    except (AlreadyEvaluatedError, ValueError, TypeError, OSError, UnicodeError) as exc:
        category = "already consumed" if isinstance(exc, AlreadyEvaluatedError) else "rejected"
        print(f"four-arm custody evaluation {category}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
