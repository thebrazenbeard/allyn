"""Tests for source-group and rights declarations, not rights authorization."""

from __future__ import annotations

import json
from itertools import repeat
import unittest

from allyn.provenance import (
    CLAIM_CEILING,
    ProvenanceManifest,
    ProvenanceRecord,
    build_manifest,
)


def declaration(
    group: str,
    identifier: str,
    split: str = "train",
    *,
    license_record_id: str = "rights-reviewed-external",
    excluded: bool = False,
) -> ProvenanceRecord:
    use = {"train": "training", "dev": "selection", "heldout": "evaluation"}.get(split, "training")
    return ProvenanceRecord(group, identifier, license_record_id, split, use, excluded)


class ProvenanceManifestTests(unittest.TestCase):
    def test_complete_manifest_is_caller_declared_not_permission(self) -> None:
        manifest = build_manifest([
            declaration("bookA", "a1"),
            declaration("bookB", "b1", "dev"),
            declaration("bookC", "c1", "heldout"),
        ])
        self.assertEqual(len(manifest.records), 3)
        self.assertEqual(len(manifest.included("train")), 1)
        self.assertEqual(len(manifest.included("heldout")), 1)
        self.assertEqual(manifest.claim_ceiling, CLAIM_CEILING)
        self.assertNotIn("authorized", manifest.snapshot())
        self.assertEqual(len(manifest.manifest_sha256), 64)

    def test_canonical_digest_independent_of_input_order(self) -> None:
        records = [declaration("bookC", "c1", "heldout"), declaration("bookA", "a1"), declaration("bookB", "b1", "dev")]
        first = build_manifest(records)
        second = build_manifest(reversed(records))
        self.assertEqual(first, second)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)

    def test_missing_license_declaration_rejected(self) -> None:
        for bad in ("", " ", None, "private/license.txt", "C:\\secrets\\license", "https://example.com/claims"):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                build_manifest([declaration("family", "record", license_record_id=bad)])

    def test_source_family_cannot_cross_splits_even_when_excluded(self) -> None:
        records = [declaration("family", "r1", excluded=True), declaration("family", "r2", "heldout")]
        with self.assertRaisesRegex(ValueError, "source_group_id may not cross"):
            build_manifest(records)

    def test_record_ids_are_unique_globally(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            build_manifest([declaration("a", "record"), declaration("b", "record", "dev")])

    def test_use_must_match_split(self) -> None:
        bad = ProvenanceRecord("a", "record", "spdx-license-ref", "heldout", "training")
        with self.assertRaisesRegex(ValueError, "intended_use must be evaluation"):
            build_manifest([bad])
        with self.assertRaisesRegex(ValueError, "split must be"):
            build_manifest([declaration("a", "record", "train-eval")])

    def test_exclusion_marker_not_eligible_for_any_split(self) -> None:
        manifest = build_manifest([declaration("a", "r1"), declaration("a", "r2", excluded=True)])
        self.assertEqual(tuple(r.record_id for r in manifest.included("train")), ("r1",))
        self.assertEqual(len(manifest.records), 2)
        self.assertTrue(manifest.snapshot()["records"][1]["excluded"])
        with self.assertRaisesRegex(ValueError, "excluded must be bool"):
            build_manifest([ProvenanceRecord("a", "r1", "license-ref", "train", "training", 1)])

    def test_rejects_absolute_path_email_and_invalid_unicode_ids(self) -> None:
        for bad in ("/etc/passwd", "C:\\Users\\patri", "\\\\nas\\share", "email@example.com", "name with spaces", "bad\ud800", "a" * 257):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                build_manifest([declaration(bad, "record")])

    def test_stream_and_capacity_are_bounded_before_materialization(self) -> None:
        class IncreasingRecords:
            def __iter__(self):
                index = 0
                while True:
                    yield declaration("group", f"record{index}")
                    index += 1

        with self.assertRaisesRegex(ValueError, "exceeds 3 records"):
            build_manifest(IncreasingRecords(), max_records=3)
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            build_manifest(repeat(declaration("g", "r")))
        for bad_capacity in (0, -1, True, 10_001):
            with self.assertRaises(ValueError):
                build_manifest([declaration("g", "r")], max_records=bad_capacity)

    def test_snapshot_is_detached_and_json_serializable(self) -> None:
        manifest = build_manifest([declaration("family", "r1")])
        snapshot = manifest.snapshot()
        json.dumps(snapshot)
        snapshot["records"][0]["source_group_id"] = "tampered"
        self.assertEqual(manifest.records[0].source_group_id, "family")
        self.assertEqual(manifest.snapshot()["records"][0]["source_group_id"], "family")

    def test_forged_public_manifest_descriptor_is_rechecked(self) -> None:
        real = build_manifest([declaration("family", "r1")])
        altered = ProvenanceManifest((declaration("family", "other"),), real.manifest_sha256)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            altered.snapshot()
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            altered.included("train")
        wrong_ceiling = ProvenanceManifest(real.records, real.manifest_sha256, claim_ceiling="CERTIFIED")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            wrong_ceiling.included("train")

    def test_invalid_iterable_and_empty_manifest(self) -> None:
        for value in (None, "not records", ["bad"], [], {}, b"string"):
            with self.subTest(value=type(value).__name__), self.assertRaises((TypeError, ValueError)):
                build_manifest(value)


if __name__ == "__main__":
    unittest.main()
