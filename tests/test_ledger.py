"""Behavioral contract for ALLYN's source-attributed prequential ledger."""

from __future__ import annotations

import unittest

from allyn.ledger import PrequentialLedger


class FakeNextTokenModel:
    def __init__(self, predictions: list[str | None]) -> None:
        self.predictions = iter(predictions)
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.fail_prediction = False
        self.fail_observation = False

    def predict_next(self, context: tuple[str, ...]) -> str | None:
        self.calls.append(("predict", tuple(context)))
        if self.fail_prediction:
            raise RuntimeError("predict failed")
        return next(self.predictions)

    def observe(self, tokens: tuple[str, ...]) -> None:
        self.calls.append(("observe", tuple(tokens)))
        if self.fail_observation:
            raise RuntimeError("observe failed")


class PrequentialLedgerTests(unittest.TestCase):
    def test_prediction_precedes_observation_and_carries_scoped_evidence(self) -> None:
        model = FakeNextTokenModel(["world"])
        ledger = PrequentialLedger()

        event = ledger.predict_then_observe(model, ["hello", "world"], "corpus:line-1")

        self.assertEqual(model.calls, [
            ("predict", ("hello",)),
            ("observe", ("hello", "world")),
        ])
        self.assertEqual(event.context, ("hello",))
        self.assertEqual(event.target, "world")
        self.assertEqual(event.prediction, "world")
        self.assertTrue(event.correct)
        self.assertEqual(event.source_id, "corpus:line-1")
        self.assertEqual(event.sequence_number, 1)
        self.assertEqual(event.event_id, "allyn-pq-000000000001")
        self.assertEqual(event.claim_ceiling, "OBSERVED_NEXT_TOKEN_MATCH_ONLY")
        self.assertEqual(ledger.events, (event,))

    def test_unseen_prediction_counts_as_incorrect_and_summary_is_defined(self) -> None:
        model = FakeNextTokenModel([None, "wrong", "right"])
        ledger = PrequentialLedger()

        self.assertEqual(ledger.summary(), {"total": 0, "correct": 0, "accuracy": None})
        first = ledger.predict_then_observe(model, ("a",), "one")
        second = ledger.predict_then_observe(model, ("b", "c"), "two")
        third = ledger.predict_then_observe(model, ("d", "right"), "three")

        self.assertFalse(first.correct)
        self.assertFalse(second.correct)
        self.assertTrue(third.correct)
        self.assertEqual((first.sequence_number, second.sequence_number, third.sequence_number), (1, 2, 3))
        self.assertEqual([e.event_id for e in ledger.events], [
            "allyn-pq-000000000001", "allyn-pq-000000000002", "allyn-pq-000000000003",
        ])
        self.assertEqual(model.calls[:2], [("predict", ()), ("observe", ("a",))])
        self.assertEqual(ledger.summary(), {"total": 3, "correct": 1, "accuracy": 1 / 3})

    def test_event_and_ledger_view_are_immutable_snapshots(self) -> None:
        model = FakeNextTokenModel(["b"])
        ledger = PrequentialLedger()
        tokens = ["a", "b"]
        event = ledger.predict_then_observe(model, tokens, "source")
        tokens[0] = "mutated"

        self.assertEqual(event.context, ("a",))
        self.assertIsInstance(ledger.events, tuple)
        with self.assertRaises(AttributeError):
            event.target = "tamper"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            ledger.events.append(event)

    def test_reject_bad_input_without_touching_model(self) -> None:
        model = FakeNextTokenModel(["unused"])
        ledger = PrequentialLedger()

        bad_inputs = ([], "token stream", ["ok", ""], ["ok", 9], ["ok", None])
        for bad in bad_inputs:
            with self.subTest(tokens=bad), self.assertRaises((TypeError, ValueError)):
                ledger.predict_then_observe(model, bad, "source")  # type: ignore[arg-type]
        for source in ("", "   ", 9):
            with self.subTest(source=source), self.assertRaises((TypeError, ValueError)):
                ledger.predict_then_observe(model, ("ok",), source)  # type: ignore[arg-type]
        self.assertEqual(model.calls, [])
        self.assertEqual(ledger.events, ())

    def test_prediction_failure_does_not_observe_or_append(self) -> None:
        model = FakeNextTokenModel(["valid"])
        model.fail_prediction = True
        ledger = PrequentialLedger()

        with self.assertRaisesRegex(RuntimeError, "predict failed"):
            ledger.predict_then_observe(model, ("a", "b"), "source")
        self.assertEqual(model.calls, [("predict", ("a",))])
        self.assertEqual(ledger.events, ())

        model.fail_prediction = False
        event = ledger.predict_then_observe(model, ("a", "valid"), "source")
        self.assertEqual(event.sequence_number, 1)

    def test_observation_failure_does_not_commit_event(self) -> None:
        model = FakeNextTokenModel(["b", "c"])
        model.fail_observation = True
        ledger = PrequentialLedger()

        with self.assertRaisesRegex(RuntimeError, "observe failed"):
            ledger.predict_then_observe(model, ("a", "b"), "source")
        self.assertEqual(model.calls, [("predict", ("a",)), ("observe", ("a", "b"))])
        self.assertEqual(ledger.events, ())
        model.fail_observation = False
        event = ledger.predict_then_observe(model, ("a", "c"), "source")
        self.assertEqual(event.event_id, "allyn-pq-000000000001")

    def test_failed_model_mutation_is_not_mistaken_for_ledger_rollback(self) -> None:
        class PartiallyMutatingModel:
            def __init__(self) -> None:
                self.observed_examples: list[tuple[str, ...]] = []

            def predict_next(self, context: tuple[str, ...]) -> str:
                return "b"

            def observe(self, tokens: tuple[str, ...]) -> None:
                self.observed_examples.append(tuple(tokens))
                raise RuntimeError("failed after mutation")

        model = PartiallyMutatingModel()
        ledger = PrequentialLedger()
        with self.assertRaisesRegex(RuntimeError, "failed after mutation"):
            ledger.predict_then_observe(model, ("a", "b"), "source")
        self.assertEqual(ledger.events, ())
        self.assertEqual(model.observed_examples, [("a", "b")])

    def test_invalid_prediction_type_rejected_before_observation(self) -> None:
        class InvalidModel:
            def __init__(self) -> None:
                self.observed = False

            def predict_next(self, context: tuple[str, ...]) -> object:
                return {"unsupported": 0.5}

            def observe(self, tokens: tuple[str, ...]) -> None:
                self.observed = True

        model = InvalidModel()
        ledger = PrequentialLedger()
        with self.assertRaises(TypeError):
            ledger.predict_then_observe(model, ("a", "b"), "source")
        self.assertFalse(model.observed)
        self.assertEqual(ledger.events, ())


if __name__ == "__main__":
    unittest.main()
