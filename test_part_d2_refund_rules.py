"""Isolated Part D2 rule tests; all ledger and run-log writes use temp files."""

from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any

from part_d_refund_agent import RefundAgent, STATE_DIR
from part_d_refund_agent_corrected import (
    _parse_delivery_minutes,
    _parse_money,
    json_safe,
)


class RefundRuleTests(unittest.TestCase):
    def test_compatibility_shim_exports_checkpoint_directory(self) -> None:
        self.assertEqual(STATE_DIR, Path(__file__).resolve().parent / "state")

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="instamart-d2-")
        self.temp_path = Path(self.temp_dir.name)
        self.ledger_path = self.temp_path / "refund_ledger.jsonl"
        self.run_log_path = self.temp_path / "agent_runs.jsonl"
        self.agent = RefundAgent(
            ledger_path=self.ledger_path,
            run_log_path=self.run_log_path,
        )
        self.order: dict[str, Any] = {
            "order_id": "ORD-100048",
            "status": "completed",
            "order_amount": 128.25,
            "delivery_minutes": 65.0,
            "delivery_minutes_source_valid": True,
            "delivery_timestamp": None,
        }
        self.agent.current_order_id = self.order["order_id"]
        self.agent.get_order = lambda _order_id: dict(self.order)  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self.agent.close()
        self.temp_dir.cleanup()

    def test_calculation_uses_full_order_amount_and_requires_review(self) -> None:
        result = self.agent.calculate_refund(1, 90, "ORD-100048")
        self.assertEqual(result["refund_inr"], Decimal("128.25"))
        self.assertFalse(result["eligible"])
        self.assertTrue(result["requires_human_review"])
        self.assertEqual(
            result["reason"], "refund_deadline_unverifiable_from_order_data"
        )

    def test_cancelled_and_exactly_45_minutes_are_ineligible(self) -> None:
        self.order.update(status="cancelled")
        cancelled = self.agent.issue_refund("ORD-100048", 128.25)
        self.assertEqual(cancelled["status"], "DENIED")
        self.assertFalse(cancelled["real_money_moved"])

        self.order.update(status="completed", delivery_minutes=45)
        exact_boundary = self.agent.issue_refund("ORD-100048", 128.25)
        self.assertEqual(exact_boundary["status"], "DENIED")
        self.assertFalse(exact_boundary["real_money_moved"])

    def test_invalid_delivery_durations_fail_closed(self) -> None:
        for duration in (None, "bad", float("nan"), float("inf"), -1):
            with self.subTest(duration=duration):
                self.order["delivery_minutes"] = duration
                self.order["delivery_minutes_source_valid"] = False
                result = self.agent.issue_refund("ORD-100048", 128.25)
                self.assertEqual(result["status"], "ESCALATE")
                self.assertEqual(
                    result["reason"], "delivery_duration_unverified_from_order_data"
                )
                self.assertFalse(result["real_money_moved"])
                self.assertFalse(self.ledger_path.exists())
        self.order["delivery_minutes"] = 45.01
        self.order["delivery_minutes_source_valid"] = True

    def test_over_45_minutes_routes_for_review_without_ledger_write(self) -> None:
        result = self.agent.issue_refund("ORD-100048", 128.25)
        self.assertEqual(result["status"], "ESCALATE")
        self.assertEqual(
            result["reason"], "refund_deadline_unverifiable_from_order_data"
        )
        self.assertFalse(result["real_money_moved"])
        self.assertFalse(self.ledger_path.exists())

    def test_unverified_duration_routes_for_review_without_ledger_write(self) -> None:
        self.order["delivery_minutes_source_valid"] = False
        result = self.agent.issue_refund("ORD-100048", 128.25)
        self.assertEqual(result["status"], "ESCALATE")
        self.assertEqual(
            result["reason"], "delivery_duration_unverified_from_order_data"
        )
        self.assertFalse(result["real_money_moved"])
        self.assertFalse(self.ledger_path.exists())

        calculation = self.agent.calculate_refund(1, 90, "ORD-100048")
        self.assertFalse(calculation["eligible"])
        self.assertTrue(calculation["requires_human_review"])
        self.assertEqual(
            calculation["reason"], "delivery_duration_unverified_from_order_data"
        )

    def test_exact_1000_and_above_1000_approval_boundary(self) -> None:
        self.order["order_amount"] = 1000.00
        at_limit = self.agent.calculate_refund(1, 90, "ORD-100048")
        self.assertEqual(at_limit["refund_inr"], Decimal("1000.00"))
        self.assertFalse(at_limit["approval_required"])
        self.assertEqual(
            self.agent.issue_refund("ORD-100048", 1000.00)["status"],
            "ESCALATE",
        )

        self.order["order_amount"] = 1000.01
        above_limit = self.agent.calculate_refund(1, 90, "ORD-100048")
        self.assertEqual(above_limit["refund_inr"], Decimal("1000.01"))
        self.assertTrue(above_limit["approval_required"])
        escalated = self.agent.issue_refund("ORD-100048", 1000.01)
        self.assertEqual(escalated["status"], "ESCALATE")
        self.assertIn("human_approval_required_above_INR_1000", escalated["reason"])
        self.assertFalse(escalated["real_money_moved"])

    def test_invalid_requested_amounts_are_rejected(self) -> None:
        for amount in (
            float("nan"), float("inf"), float("-inf"), -1, 0,
            None, "not-a-number", "1.001", True,
        ):
            with self.subTest(amount=amount):
                result = self.agent.issue_refund("ORD-100048", amount)
                self.assertEqual(result["status"], "DENIED")
                self.assertEqual(result["reason"], "invalid_refund_amount")
                self.assertFalse(result["real_money_moved"])

    def test_invalid_or_missing_trusted_order_amounts_are_rejected(self) -> None:
        for amount in (
            None, float("nan"), float("inf"), float("-inf"), -1, 0,
            "bad", "12.345",
        ):
            with self.subTest(amount=amount):
                self.order["order_amount"] = amount
                result = self.agent.issue_refund("ORD-100048", 128.25)
                self.assertEqual(result["status"], "DENIED")
                self.assertFalse(result["real_money_moved"])

    def test_money_parser_is_exact_and_json_compatible(self) -> None:
        self.assertEqual(_parse_money("1000.00"), Decimal("1000.00"))
        self.assertEqual(_parse_money("1000.01"), Decimal("1000.01"))
        self.assertIsNone(_parse_money("0.001"))
        self.assertIsNone(_parse_money("NaN"))
        encoded = json.dumps({"amount": Decimal("128.25")}, default=json_safe)
        self.assertEqual(json.loads(encoded)["amount"], 128.25)

    def test_delivery_duration_boundaries_and_invalid_values(self) -> None:
        self.assertEqual(_parse_delivery_minutes("45"), 45.0)
        self.assertEqual(_parse_delivery_minutes("45.01"), 45.01)
        for duration in (None, "invalid", float("nan"), float("inf"), -0.01, True):
            with self.subTest(duration=duration):
                self.assertIsNone(_parse_delivery_minutes(duration))

    def test_malformed_ledger_is_preserved_and_blocks_refund(self) -> None:
        original = (
            b'{"order_id":"ORD-100048","refund_inr":12.83,'
            b'"status":"SIMULATED_REFUND_RECORDED"}\n'
            b'{malformed historical bytes\xff\n'
        )
        self.ledger_path.write_bytes(original)
        self.agent.close()
        self.agent = RefundAgent(
            ledger_path=self.ledger_path,
            run_log_path=self.run_log_path,
        )
        self.agent.get_order = lambda _order_id: dict(self.order)  # type: ignore[method-assign]
        result = self.agent.issue_refund("ORD-100048", "128.25")
        self.assertEqual(result["status"], "ESCALATE")
        self.assertEqual(
            result["reason"], "refund_ledger_integrity_requires_human_review"
        )
        self.assertFalse(result["real_money_moved"])
        self.assertEqual(self.ledger_path.read_bytes(), original)

    def test_duplicate_and_conflicting_ledger_entries_are_not_reissued(self) -> None:
        for prior_amount, expected_reason in (
            (128.25, "existing_simulated_refund_requires_human_review"),
            (12.83, "conflicting_existing_simulated_refund_requires_human_review"),
        ):
            with self.subTest(prior_amount=prior_amount):
                self.ledger_path.write_text(
                    json.dumps(
                        {
                            "order_id": "ORD-100048",
                            "refund_inr": prior_amount,
                            "status": "SIMULATED_REFUND_RECORDED",
                            "real_money_moved": False,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self.agent.ledger_records = [
                    json.loads(self.ledger_path.read_text(encoding="utf-8"))
                ]
                before = self.ledger_path.read_bytes()
                result = self.agent.issue_refund("ORD-100048", 128.25)
                self.assertEqual(result["status"], "ESCALATE")
                self.assertEqual(result["reason"], expected_reason)
                self.assertEqual(
                    result["ledger_match"],
                    "already_recorded" if prior_amount == 128.25 else "conflict",
                )
                self.assertFalse(result["real_money_moved"])
                self.assertEqual(self.ledger_path.read_bytes(), before)

    def test_human_escalation_uses_isolated_run_log(self) -> None:
        result = self.agent.escalate_to_human("ORD-100048", 128.25)
        self.assertEqual(result["status"], "ESCALATE")
        self.assertIn("deadline_unverifiable", result["reason"])
        self.assertTrue(self.run_log_path.is_file())
        self.assertIn("escalation_tool", self.run_log_path.read_text(encoding="utf-8"))
        self.assertFalse(self.ledger_path.exists())


if __name__ == "__main__":
    unittest.main()
