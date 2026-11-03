"""Temporary-only tests for delivery-duration provenance and D2 fail-closed use."""

from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import clean_data
import part_d_refund_agent_corrected as agent_module
import part_d_rag as rag_module
from part_d_refund_agent import RefundAgent


class DeliveryDurationProvenanceTests(unittest.TestCase):
    def test_cleaning_marks_only_source_observations_valid(self) -> None:
        with tempfile.TemporaryDirectory(prefix="instamart-duration-clean-") as temp:
            root = Path(temp)
            raw_dir = root / "raw"
            clean_dir = root / "clean"
            output_dir = root / "outputs"
            raw_dir.mkdir()
            clean_dir.mkdir()
            output_dir.mkdir()

            pd.DataFrame(
                [{"product_id": "P001", "price_inr": 100, "category": "Grocery"}]
            ).to_csv(clean_dir / "products.csv", index=False)
            rows = [
                ["ORD-000001", "C0001", "P001", 1, "2025-01-01 10:00:00", "completed", 0, 100, "60"],
                ["ORD-000002", "C0001", "P001", 1, "2025-01-01 10:01:00", "completed", 0, 100, ""],
                ["ORD-000003", "C0001", "P001", 1, "2025-01-01 10:02:00", "completed", 0, 100, "bad"],
                ["ORD-000004", "C0001", "P001", 1, "2025-01-01 10:03:00", "completed", 0, 100, "-5"],
            ]
            with (raw_dir / "orders.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]
                )
                writer.writerows(rows)

            with patch.multiple(
                clean_data,
                RAW_DIR=raw_dir,
                CLEAN_DIR=clean_dir,
                OUTPUT_DIR=output_dir,
            ):
                clean_data.clean_orders()

            cleaned = pd.read_csv(clean_dir / "orders.csv").set_index("order_id")
            self.assertEqual(int(cleaned.loc["ORD-000001", "delivery_minutes_source_valid"]), 1)
            for order_id in ("ORD-000002", "ORD-000003", "ORD-000004"):
                self.assertEqual(int(cleaned.loc[order_id, "delivery_minutes_source_valid"]), 0)
                self.assertEqual(float(cleaned.loc[order_id, "delivery_minutes"]), 60.0)

    def _agent_for_schema(self, root: Path, *, include_provenance: bool, provenance: int = 1) -> RefundAgent:
        database_path = root / "orders.sqlite"
        scenario_path = root / "scenarios.csv"
        policy_dir = root / "policies"
        output_dir = root / "outputs"
        policy_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        scenario_path.write_text("scenario_id\n", encoding="utf-8")
        (policy_dir / "refund_policy.txt").write_text(
            "Refund policy test fixture.", encoding="utf-8"
        )

        connection = sqlite3.connect(database_path)
        try:
            provenance_column = (
                ", delivery_minutes_source_valid INTEGER" if include_provenance else ""
            )
            connection.execute(
                "CREATE TABLE orders (order_id TEXT, status TEXT, order_amount REAL, "
                "delivery_minutes REAL, order_ts TEXT" + provenance_column + ")"
            )
            values = ["ORD-100048", "completed", 128.25, 65.0, "2025-01-01 10:00:00"]
            if include_provenance:
                values.append(provenance)
            placeholders = ",".join("?" for _ in values)
            connection.execute(
                f"INSERT INTO orders VALUES ({placeholders})", values
            )
            connection.commit()
        finally:
            connection.close()

        patched = patch.multiple(
            agent_module,
            DB_PATH=database_path,
            SCENARIO_FILE=scenario_path,
            POLICY_DIR=policy_dir,
            REFUND_LEDGER=output_dir / "refund_ledger.jsonl",
            RUN_LOG=output_dir / "agent_runs.jsonl",
        )
        patched.start()
        self.addCleanup(patched.stop)
        rag_policy_patch = patch.object(rag_module, "POLICY_DIR", policy_dir)
        rag_policy_patch.start()
        self.addCleanup(rag_policy_patch.stop)
        return RefundAgent()

    def test_imputed_duration_is_not_eligibility_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="instamart-duration-agent-") as temp:
            agent = self._agent_for_schema(Path(temp), include_provenance=True, provenance=0)
            try:
                order = agent.get_order("ORD-100048")
                self.assertFalse(order["delivery_minutes_source_valid"])
                result = agent.issue_refund("ORD-100048", "128.25")
                self.assertEqual(result["status"], "ESCALATE")
                self.assertEqual(result["reason"], "delivery_duration_unverified_from_order_data")
                self.assertFalse(result["real_money_moved"])
                self.assertFalse(agent.refund_ledger_path.exists())
            finally:
                agent.close()

    def test_missing_provenance_column_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="instamart-duration-legacy-") as temp:
            agent = self._agent_for_schema(Path(temp), include_provenance=False)
            try:
                order = agent.get_order("ORD-100048")
                self.assertFalse(order["delivery_minutes_source_valid"])
                result = agent.issue_refund("ORD-100048", "128.25")
                self.assertEqual(result["status"], "ESCALATE")
                self.assertEqual(result["reason"], "delivery_duration_unverified_from_order_data")
                self.assertFalse(result["real_money_moved"])
                self.assertFalse(agent.refund_ledger_path.exists())
            finally:
                agent.close()

    def test_observed_late_duration_still_fails_closed_on_seven_day_rule(self) -> None:
        with tempfile.TemporaryDirectory(prefix="instamart-duration-deadline-") as temp:
            agent = self._agent_for_schema(Path(temp), include_provenance=True, provenance=1)
            try:
                result = agent.issue_refund("ORD-100048", "128.25")
                self.assertEqual(result["status"], "ESCALATE")
                self.assertEqual(
                    result["reason"], "refund_deadline_unverifiable_from_order_data"
                )
                self.assertFalse(result["real_money_moved"])
                self.assertFalse(agent.refund_ledger_path.exists())
            finally:
                agent.close()


if __name__ == "__main__":
    unittest.main()
