"""Disposable local scoring process with an *advisory* one-use registry.

Run ``python -m allyn.custodian --state-dir DIR --request-id ID`` and supply
one JSON evaluation request on stdin. The child process consumes heldout data
only to score the submitted lesson; stdout and its durable receipt omit those
examples. A separate trusted evaluator *can* feed this process without giving
the heldout dataset to a teacher process, but this module cannot enforce who
has seen either the stdin or filesystem.

Aggregate scores can reveal answers on tiny or low-entropy test sets when
combined with known candidate predictions. Give score receipts only to the
authorized evaluator until the experiment is frozen and complete; do not
use them as a secret-safe teacher feedback channel.

This is NOT independent custody, tamper resistance, secure isolation, verified
provenance, or model-weight teaching. Claim and receipt files are advisory
against accidental repeat evaluation in a cooperating local state directory.
A caller who can alter datasets, directories, or account files can defeat it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

from .evaluation import FrozenSpecimen, SourcedExample, evaluate_frozen, freeze_specimen


REQUEST_SCHEMA = "allyn.custody-request.v0"
RECEIPT_SCHEMA = "allyn.custody-receipt.v0"
CLAIM_CEILING = "LOCAL_PROCESS_BOUNDARY_AND_ADVISORY_REPLAY_REGISTRY_ONLY"
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}\Z")
_MAX_STDIN_BYTES = 2_000_000


class AlreadyEvaluatedError(ValueError):
    """A request ID or exact frozen specimen was already claimed in this dir."""


@dataclass(frozen=True, slots=True)
class CustodyReceipt:
    """Evaluator-facing score receipt; no raw records or answer-key tokens.

    Aggregates can still disclose information for tiny or known candidate sets.
    """

    request_id: str
    baseline_accuracy: float
    taught_accuracy: float
    gain: float
    evaluated_count: int
    status: str = "evaluated"
    schema: str = RECEIPT_SCHEMA
    claim_ceiling: str = CLAIM_CEILING


def _safe_request_id(value: str) -> str:
    if type(value) is not str or not _REQUEST_ID.fullmatch(value):
        raise ValueError("request_id must be a safe 1-64 character ASCII identifier")
    return value


def _state_directory(directory: os.PathLike[str] | str) -> Path:
    path = Path(directory)
    if path.is_symlink():
        raise ValueError("state directory cannot be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("state directory must be an ordinary directory")
    return path


def _atomic_claim(path: Path, content: dict[str, object]) -> None:
    """Never truncate an existing claim, even on a failed earlier run."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise AlreadyEvaluatedError("request ID or frozen specimen already used") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(content, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())


def _replace_receipt(path: Path, content: dict[str, object]) -> None:
    """Publish a complete sanitized result without opening the claim for edit."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix="receipt-", suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            os.chmod(temporary, 0o600)
            json.dump(content, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def evaluate_once(
    specimen: FrozenSpecimen, *, state_dir: os.PathLike[str] | str, request_id: str
) -> CustodyReceipt:
    """Score once per exact heldout tokens in a cooperating dir and request ID.

    Both reservations are exclusive on cooperative local filesystems. Failure
    after reservation leaves the heldout claim consumed. A lesson-only change
    does *not* unlock the same test questions. Modified/partial heldout sets,
    new directories, same-user mutation or malicious local callers are outside
    the protection offered here.
    """
    requested = _safe_request_id(request_id)
    if type(specimen) is not FrozenSpecimen:
        raise TypeError("specimen must be a FrozenSpecimen")
    # Re-check its canonical content before reserving a digest-based claim.
    checked = freeze_specimen(
        specimen.student, specimen.lessons, specimen.heldout,
        order=specimen.order,
        allow_shared_source_groups=specimen.allow_shared_source_groups,
    )
    if (specimen.schema != checked.schema or
        specimen.claim_ceiling != checked.claim_ceiling or
        specimen.specimen_sha256 != checked.specimen_sha256):
        raise ValueError("invalid or modified frozen specimen")

    directory = _state_directory(state_dir)
    public_path = directory / f"request-{requested}.json"
    digest_path = directory / f"specimen-{checked.specimen_sha256}.claim"
    # The full specimen digest changes when a teacher tweaks lessons; it is
    # insufficient as a replay barrier. Hash only canonical HELDOUT tokens,
    # independent of declared source labels or lesson/student content.
    heldout_payload = json.dumps(
        sorted(row.tokens for row in checked.heldout),
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    heldout_digest = hashlib.sha256(heldout_payload).hexdigest()
    heldout_path = directory / f"heldout-{heldout_digest}.claim"
    if public_path.exists():
        raise AlreadyEvaluatedError("request ID already used")
    _atomic_claim(heldout_path, {"schema": RECEIPT_SCHEMA, "status": "heldout_consumed"})
    _atomic_claim(digest_path, {"schema": RECEIPT_SCHEMA, "status": "consumed"})
    try:
        _atomic_claim(public_path, {
            "schema": RECEIPT_SCHEMA, "request_id": requested,
            "status": "reserved", "claim_ceiling": CLAIM_CEILING,
        })
    except Exception:
        # Digest remains consumed on partial failure: fail closed.
        raise
    try:
        transfer = evaluate_frozen(checked).transfer
        receipt = CustodyReceipt(
            request_id=requested,
            baseline_accuracy=transfer.baseline_accuracy,
            taught_accuracy=transfer.taught_accuracy,
            gain=transfer.gain,
            evaluated_count=transfer.evaluated_count,
        )
        _replace_receipt(public_path, asdict(receipt))
        return receipt
    except Exception:
        _replace_receipt(public_path, {
            "schema": RECEIPT_SCHEMA, "request_id": requested,
            "status": "failed", "claim_ceiling": CLAIM_CEILING,
        })
        raise


def _from_wire(data: Any) -> FrozenSpecimen:
    if type(data) is not dict or set(data) != {
        "schema", "student", "lessons", "heldout", "order", "allow_shared_source_groups"
    } or data["schema"] != REQUEST_SCHEMA:
        raise ValueError("invalid custody request schema")

    def records(raw: Any, name: str) -> list[SourcedExample]:
        if type(raw) is not list:
            raise TypeError(f"{name} must be a list")
        out: list[SourcedExample] = []
        for row in raw:
            if type(row) is not dict or set(row) != {"source_id", "record_id", "tokens"}:
                raise TypeError(f"{name} contains invalid records")
            if type(row["tokens"]) is not list:
                raise TypeError(f"{name} token record must be a list")
            out.append(SourcedExample(row["source_id"], row["record_id"], tuple(row["tokens"])))
        return out

    return freeze_specimen(
        records(data["student"], "student"),
        records(data["lessons"], "lessons"),
        records(data["heldout"], "heldout"),
        order=data["order"],
        allow_shared_source_groups=data["allow_shared_source_groups"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot, local, advisory ALLYN evaluator")
    parser.add_argument("--state-dir", required=True, help="Private cooperating evaluator state directory")
    parser.add_argument("--request-id", required=True, help="Unique ASCII request ID")
    args = parser.parse_args(argv)
    try:
        payload = sys.stdin.buffer.read(_MAX_STDIN_BYTES + 1)
        if len(payload) > _MAX_STDIN_BYTES:
            raise ValueError("request size exceeded")
        specimen = _from_wire(json.loads(payload.decode("utf-8")))
        receipt = evaluate_once(specimen, state_dir=args.state_dir, request_id=args.request_id)
    except (AlreadyEvaluatedError, ValueError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        # Generic errors avoid echoing private heldout samples or traceback.
        category = "already consumed" if isinstance(exc, AlreadyEvaluatedError) else "rejected"
        print(f"custody evaluation {category}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
