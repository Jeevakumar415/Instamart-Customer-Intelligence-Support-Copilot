"""Isolated synthetic tests for explicit Part B source binding."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import part_b_lapse_model as lapse


def _write_fixture(root: Path, *, segments_path: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    frames = {
        "customers.csv": pd.DataFrame(
            [{
                "customer_id": "C1", "city": "Test City", "signup_date": "2025-01-01",
                "age": 35, "gender": "F", "age_source_valid": 1,
                "signup_date_source_missing": 0, "signup_date_source_malformed": 0,
            }]
        ),
        "orders.csv": pd.DataFrame(
            [{
                "order_id": "O1", "customer_id": "C1", "product_id": "P1",
                "quantity": 1, "order_ts": "2025-08-01 10:00:00", "status": "completed",
                "discount_pct": 0, "order_amount": 100, "delivery_minutes": 30,
                "delivery_minutes_source_valid": 1,
            }]
        ),
        "products.csv": pd.DataFrame(
            [{"product_id": "P1", "product_name": "Test Product", "category": "Test", "price_inr": 100}]
        ),
        "tickets.csv": pd.DataFrame(
            [{"ticket_id": "T1", "customer_id": "C1", "order_id": "O1", "text": "Test", "true_category": "Other"}]
        ),
        "policy_eval.csv": pd.DataFrame(
            [{"question": "Test?", "expected_doc": "policy", "expected_fact": "fact"}]
        ),
        "agent_scenarios.csv": pd.DataFrame(
            [{"scenario_id": "S1", "customer_id": "C1", "message": "Test", "expected_outcome": "ESCALATE", "expected_refund_inr": 0}]
        ),
    }
    for filename, frame in frames.items():
        frame.to_csv(root / filename, index=False)
    segment_file = segments_path or root.parent / "customer_segments.csv"
    segment_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "customer_id": "C1", "recency": 31, "frequency": 1,
            "monetary": 100, "segment_id": 0, "segment_name": "Test",
            "is_premium": False,
        }]
    ).to_csv(segment_file, index=False)
    return segment_file


class PartBInputBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="part-b-input-binding-")
        self.base = Path(self.temp.name)
        self.root = self.base / "f2-like" / "clean"
        self.segments_path = _write_fixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def load(self):
        return lapse.load_part_b_inputs(self.root, segments_path=self.segments_path)

    def test_explicit_absolute_input_root_is_required(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, lapse.INPUT_ROOT_ENV):
                lapse.load_part_b_inputs()
        with self.assertRaisesRegex(ValueError, "absolute path"):
            lapse.load_part_b_inputs("relative/clean", segments_path=self.segments_path)

    def test_environment_input_root_is_used_explicitly(self) -> None:
        with patch.dict(os.environ, {lapse.INPUT_ROOT_ENV: str(self.root)}):
            bundle = lapse.load_part_b_inputs(segments_path=self.segments_path)
        self.assertEqual(bundle["root"], self.root.resolve())

    def test_successful_loading_uses_all_files_from_one_root(self) -> None:
        bundle = self.load()
        self.assertEqual(set(bundle["datasets"]), set(lapse.INPUT_REQUIRED_COLUMNS))
        self.assertEqual(bundle["root"], self.root.resolve())
        for record in bundle["provenance"]["input_files"].values():
            self.assertEqual(Path(record["resolved_path"]).parent, self.root.resolve())
        self.assertEqual(
            bundle["provenance"]["segmentation_artifact"]["resolved_path"],
            str(self.segments_path.resolve()),
        )

    def set_ticket_field(self, column: str, value) -> None:
        tickets_path = self.root / "tickets.csv"
        tickets = pd.read_csv(tickets_path)
        if column not in tickets.columns:
            tickets[column] = value
        else:
            tickets[column] = tickets[column].astype("object")
            tickets.loc[0, column] = value
        tickets.to_csv(tickets_path, index=False)

    def test_ticket_order_id_null_nan_empty_and_whitespace_are_optional(self) -> None:
        for value in (None, float("nan"), "", "   "):
            with self.subTest(value=repr(value)):
                self.set_ticket_field("order_id", value)
                self.assertEqual(len(self.load()["datasets"]["tickets.csv"]), 1)

    def test_ticket_order_id_nan_token_is_optional(self) -> None:
        tickets_path = self.root / "tickets.csv"
        tickets_path.write_text(
            "ticket_id,customer_id,order_id,text,true_category\n"
            "T1,C1,NaN,Test,Other\n",
            encoding="utf-8",
        )
        self.assertTrue(pd.isna(self.load()["datasets"]["tickets.csv"].loc[0, "order_id"]))

    def test_ticket_order_id_valid_reference_is_accepted(self) -> None:
        self.set_ticket_field("order_id", " O1 ")
        self.assertEqual(
            self.load()["datasets"]["tickets.csv"].loc[0, "order_id"], " O1 "
        )

    def test_ticket_order_id_dangling_reference_is_rejected(self) -> None:
        self.set_ticket_field("order_id", "MISSING")
        with self.assertRaisesRegex(ValueError, "tickets reference orders"):
            self.load()

    def test_ticket_order_id_column_remains_required(self) -> None:
        tickets_path = self.root / "tickets.csv"
        pd.read_csv(tickets_path).drop(columns="order_id").to_csv(
            tickets_path, index=False
        )
        with self.assertRaisesRegex(ValueError, "order_id"):
            self.load()

    def test_ticket_customer_reference_is_still_required(self) -> None:
        self.set_ticket_field("customer_id", "MISSING")
        with self.assertRaisesRegex(ValueError, "tickets reference customers"):
            self.load()

    def test_missing_required_file_is_rejected_without_fallback(self) -> None:
        (self.root / "products.csv").unlink()
        with self.assertRaisesRegex(ValueError, "products.csv"):
            self.load()

    def test_legacy_schema_without_required_provenance_is_rejected(self) -> None:
        orders_path = self.root / "orders.csv"
        orders = pd.read_csv(orders_path).drop(columns="delivery_minutes_source_valid")
        orders.to_csv(orders_path, index=False)
        with self.assertRaisesRegex(ValueError, "delivery_minutes_source_valid"):
            self.load()

    def test_working_directory_does_not_select_or_mix_inputs(self) -> None:
        other = self.base / "other-cwd"
        (other / "data" / "clean").mkdir(parents=True)
        (other / "data" / "clean" / "products.csv").write_text(
            "product_id,product_name,category,price_inr\nWRONG,Wrong,Wrong,0\n",
            encoding="utf-8",
        )
        original = Path.cwd()
        try:
            os.chdir(other)
            bundle = self.load()
        finally:
            os.chdir(original)
        self.assertEqual(
            bundle["provenance"]["input_files"]["products.csv"]["resolved_path"],
            str((self.root / "products.csv").resolve()),
        )
        self.assertEqual(bundle["datasets"]["products.csv"].loc[0, "product_id"], "P1")

    def test_root_path_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "traversal"):
            lapse.load_part_b_inputs(
                str(self.root / ".."), segments_path=self.segments_path
            )

    def test_symlinked_input_is_rejected_when_supported(self) -> None:
        external = self.base / "external-products.csv"
        external.write_bytes((self.root / "products.csv").read_bytes())
        product_path = self.root / "products.csv"
        product_path.unlink()
        try:
            product_path.symlink_to(external)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlink creation is unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "Symlinked or redirected"):
            self.load()

    def test_file_hashes_and_input_fingerprint_are_deterministic(self) -> None:
        first = self.load()["provenance"]
        second = self.load()["provenance"]
        self.assertEqual(first, second)
        for filename, record in first["input_files"].items():
            expected = hashlib.sha256((self.root / filename).read_bytes()).hexdigest()
            self.assertEqual(record["sha256"], expected)
        self.assertEqual(
            first["segmentation_artifact"]["sha256"],
            hashlib.sha256(self.segments_path.read_bytes()).hexdigest(),
        )

    def test_changed_source_hash_is_rejected_by_finalizer_provenance_gate(self) -> None:
        training_inputs = self.load()
        feature_rows = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
        provenance = lapse.build_run_provenance(
            "a" * 32,
            feature_rows,
            training_inputs["segments"],
            training_inputs["provenance"],
        )
        state = {
            "run_id": "a" * 32,
            "input_fingerprint": training_inputs["provenance"]["fingerprint"],
        }
        lapse.validate_lifecycle_run_id(state, provenance)
        (self.root / "customers.csv").write_bytes(
            (self.root / "customers.csv").read_bytes() + b"\n"
        )
        current_inputs = self.load()
        with self.assertRaisesRegex(ValueError, "do not match training input provenance"):
            lapse.validate_provenance_bundle(
                provenance,
                [],
                feature_rows,
                current_inputs["segments"],
                current_inputs["provenance"],
            )

    def test_changed_input_fingerprint_is_rejected(self) -> None:
        first = self.load()["provenance"]
        changed = dict(first)
        changed["input_files"] = dict(first["input_files"])
        changed["input_files"]["orders.csv"] = dict(first["input_files"]["orders.csv"])
        changed["input_files"]["orders.csv"]["sha256"] = "c" * 64
        changed["fingerprint"] = lapse.input_provenance_fingerprint(changed)
        with self.assertRaisesRegex(ValueError, "do not match training input provenance"):
            lapse.validate_input_provenance(first, changed)

    def test_segmentation_artifact_is_bound_to_training_provenance(self) -> None:
        training_inputs = self.load()
        expected = training_inputs["provenance"]
        self.assertEqual(
            expected["segmentation_artifact"]["relative_path"], None
        )
        self.assertTrue(expected["segmentation_artifact"]["sha256"])
        self.assertEqual(
            expected["fingerprint"], lapse.input_provenance_fingerprint(expected)
        )
        self.segments_path.write_text(
            self.segments_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        current = self.load()["provenance"]
        with self.assertRaisesRegex(ValueError, "do not match training input provenance"):
            lapse.validate_input_provenance(expected, current)

    def test_finalizer_accepts_only_matching_training_provenance(self) -> None:
        training_inputs = self.load()
        feature_rows = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
        provenance = lapse.build_run_provenance(
            "b" * 32,
            feature_rows,
            training_inputs["segments"],
            training_inputs["provenance"],
        )
        state = {
            "run_id": "b" * 32,
            "input_fingerprint": training_inputs["provenance"]["fingerprint"],
        }
        lapse.validate_lifecycle_run_id(state, provenance)
        returned = lapse.validate_provenance_bundle(
            provenance,
            [("training selection", provenance)],
            feature_rows,
            training_inputs["segments"],
            training_inputs["provenance"],
        )
        self.assertEqual(returned, provenance)

    def test_lifecycle_fingerprint_must_match_manifest_provenance(self) -> None:
        inputs = self.load()
        provenance = lapse.build_run_provenance(
            "c" * 32,
            pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]}),
            inputs["segments"],
            inputs["provenance"],
        )
        state = {"run_id": "c" * 32, "input_fingerprint": "0" * 64}
        with self.assertRaisesRegex(ValueError, "input fingerprints do not match"):
            lapse.validate_lifecycle_run_id(state, provenance)


if __name__ == "__main__":
    unittest.main()
