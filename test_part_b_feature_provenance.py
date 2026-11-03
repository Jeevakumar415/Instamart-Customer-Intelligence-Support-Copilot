"""Isolated regression tests for Part B source provenance and time boundaries."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

import clean_data
import part_b_lapse_model as lapse


PROJECT_ROOT = Path(__file__).resolve().parent
PHASE_25J_RUN = PROJECT_ROOT / "outputs" / "part_b_runs" / (
    "run-6f1660f79237492f830ad8b063ffacca"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_paths() -> list[Path]:
    paths: list[Path] = []
    if PHASE_25J_RUN.exists():
        paths.extend(path for path in PHASE_25J_RUN.rglob("*") if path.is_file())
    historical = [
        "data/instamart.db",
        "outputs/refund_ledger.jsonl",
        "outputs/refund_ledger.backup.jsonl",
        "outputs/agent_runs.jsonl",
        "outputs/d8_resume_summary.json",
        "outputs/lapse_model.joblib",
        "outputs/retention_list.csv",
        "outputs/isolated_data_prep/phase-2.5NF1/clean/customers.csv",
        "outputs/isolated_data_prep/phase-2.5NF1/clean/products.csv",
        "outputs/isolated_data_prep/phase-2.5NF1/clean/orders.csv",
        "outputs/isolated_data_prep/phase-2.5NF1/clean/tickets.csv",
        "outputs/isolated_data_prep/phase-2.5NF1/clean/policy_eval.csv",
        "outputs/isolated_data_prep/phase-2.5NF1/clean/agent_scenarios.csv",
        "outputs/isolated_data_prep/phase-2.5NF1/reports/cleaning_report.json",
        "outputs/isolated_data_prep/phase-2.5NF1/reports/validation_report.json",
        "part_d_refund_agent.py",
        "part_d_refund_agent_corrected.py",
        "test_part_d7_fallbacks.py",
        "test_part_d8_resume.py",
    ]
    paths.extend(PROJECT_ROOT / value for value in historical)
    state_dir = PROJECT_ROOT / "state"
    if state_dir.exists():
        paths.extend(
            path
            for path in state_dir.iterdir()
            if path.is_file() and ("D7" in path.name or "D8" in path.name)
        )
    old_lapse_dir = PROJECT_ROOT / "outputs" / "lapse"
    if old_lapse_dir.exists():
        paths.extend(path for path in old_lapse_dir.rglob("*") if path.is_file())
    return sorted(set(path for path in paths if path.is_file()))


def _customer_rows(**overrides) -> pd.DataFrame:
    rows = []
    for customer_id, age, age_valid in (
        ("C1", 31, 1),
        ("C2", 58, 0),
        ("C3", np.nan, 0),
        ("C4", 44, 1),
    ):
        rows.append(
            {
                "customer_id": customer_id,
                "city": "Pune",
                "age": age,
                "age_source_valid": age_valid,
                "gender": "unknown",
                "signup_date": "2025-08-01",
                "signup_date_source_missing": 0,
                "signup_date_source_malformed": 0,
            }
        )
    customers = pd.DataFrame(rows)
    for customer_id, values in overrides.items():
        mask = customers["customer_id"].eq(customer_id)
        for column, value in values.items():
            customers.loc[mask, column] = value
    return customers


def _order_rows() -> pd.DataFrame:
    rows = [
        ("O1", "C1", "2025-08-10", "completed", 20.0, 1),
        ("O2", "C1", "2025-08-15", "completed", 88.0, 0),
        ("O3", "C1", "2025-09-10", "completed", 999.0, 1),
        ("O4", "C2", "2025-08-12", "completed", 777.0, 0),
        ("O5", "C2", "2025-10-02", "completed", 1500.0, 1),
        ("O6", "C3", "2025-08-13", "completed", np.inf, 1),
        ("O7", "C4", "2025-08-14", "completed", -5.0, 1),
    ]
    orders = pd.DataFrame(
        rows,
        columns=[
            "order_id",
            "customer_id",
            "order_ts",
            "status",
            "delivery_minutes",
            "delivery_minutes_source_valid",
        ],
    )
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    orders["quantity"] = 1
    orders["discount_pct"] = 0.0
    orders["order_amount"] = 100.0
    orders["product_id"] = "P1"
    return orders


def _products() -> pd.DataFrame:
    return pd.DataFrame({"product_id": ["P1"], "category": ["Grocery"]})


def _features(customers=None, orders=None):
    return lapse.build_features(
        customers if customers is not None else _customer_rows(),
        orders if orders is not None else _order_rows(),
        products=_products(),
    ).set_index("customer_id").sort_index()


class CleaningProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw_dir = self.root / "raw"
        self.clean_dir = self.root / "clean"
        self.raw_dir.mkdir()
        self.clean_dir.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_customer_cleaning_preserves_age_and_signup_source_provenance(self):
        pd.DataFrame(
            [
                ["C1", "Pune", "2025-08-01", "42", "female"],
                ["C2", "Pune", "", "", "male"],
                ["C3", "Delhi", "not-a-date", "bad-age", "unknown"],
                ["C4", "Delhi", "2025-09-02", "101", "female"],
            ],
            columns=["customer_id", "city", "signup_date", "age", "gender"],
        ).to_csv(self.raw_dir / "customers.csv", index=False)

        with patch.object(clean_data, "RAW_DIR", self.raw_dir), patch.object(
            clean_data, "CLEAN_DIR", self.clean_dir
        ):
            clean_data.clean_customers()

        cleaned = pd.read_csv(self.clean_dir / "customers.csv")
        self.assertIn("age_source_valid", cleaned.columns)
        self.assertIn("signup_date_source_missing", cleaned.columns)
        self.assertIn("signup_date_source_malformed", cleaned.columns)
        self.assertEqual(cleaned["age_source_valid"].tolist(), [1, 0, 0, 0])
        self.assertEqual(
            cleaned["signup_date_source_missing"].tolist(), [0, 1, 0, 0]
        )
        self.assertEqual(
            cleaned["signup_date_source_malformed"].tolist(), [0, 0, 1, 0]
        )
        self.assertEqual(cleaned.loc[0, "age"], 42)
        self.assertTrue(cleaned["age"].notna().all())  # Analytics imputation remains.

    def test_order_cleaning_preserves_duration_validity_before_imputation(self):
        pd.DataFrame(
            [
                ["O1", "C1", "P1", 1, "2025-08-01", "completed", 0, 100, "30"],
                ["O2", "C2", "P1", 1, "2025-08-02", "completed", 0, 100, ""],
                ["O3", "C3", "P1", 1, "2025-08-03", "completed", 0, 100, "bad"],
                ["O4", "C4", "P1", 1, "2025-08-04", "completed", 0, 100, "inf"],
                ["O5", "C5", "P1", 1, "2025-08-05", "completed", 0, 100, "-inf"],
                ["O6", "C6", "P1", 1, "2025-08-06", "completed", 0, 100, "-5"],
                ["O7", "C7", "P1", 1, "2025-09-02", "completed", 0, 100, "90"],
            ],
            columns=[
                "order_id", "customer_id", "product_id", "quantity", "order_ts",
                "status", "discount_pct", "order_amount", "delivery_minutes",
            ],
        ).to_csv(self.raw_dir / "orders.csv", index=False)
        pd.DataFrame(
            [["P1", "Item", "Grocery", 100]],
            columns=["product_id", "product_name", "category", "price_inr"],
        ).to_csv(self.clean_dir / "products.csv", index=False)

        with patch.object(clean_data, "RAW_DIR", self.raw_dir), patch.object(
            clean_data, "CLEAN_DIR", self.clean_dir
        ):
            clean_data.clean_orders()

        cleaned = pd.read_csv(self.clean_dir / "orders.csv").set_index("order_id")
        self.assertIn("delivery_minutes_source_valid", cleaned.columns)
        self.assertEqual(int(cleaned.loc["O1", "delivery_minutes_source_valid"]), 1)
        for order_id in ("O2", "O3", "O4", "O5", "O6"):
            self.assertEqual(int(cleaned.loc[order_id, "delivery_minutes_source_valid"]), 0)
            self.assertTrue(np.isfinite(cleaned.loc[order_id, "delivery_minutes"]))
        self.assertEqual(int(cleaned.loc["O7", "delivery_minutes_source_valid"]), 1)
        self.assertEqual(float(cleaned.loc["O1", "delivery_minutes"]), 30.0)


class DeliveryFeatureProvenanceTests(unittest.TestCase):
    def test_only_valid_observed_pre_cutoff_duration_enters_average(self):
        features = _features()
        c1 = features.loc["C1"]
        self.assertEqual(c1["average_delivery_minutes"], 20.0)
        self.assertEqual(c1["observed_delivery_duration_count"], 1)
        self.assertEqual(c1["unavailable_delivery_duration_count"], 1)
        self.assertEqual(c1["delivery_duration_missing"], 0)
        self.assertEqual(c1["delivery_duration_provenance_unverified"], 0)

    def test_no_valid_duration_stays_missing_with_counts_and_indicator(self):
        features = _features()
        c2 = features.loc["C2"]
        self.assertTrue(np.isnan(c2["average_delivery_minutes"]))
        self.assertEqual(c2["observed_delivery_duration_count"], 0)
        self.assertEqual(c2["unavailable_delivery_duration_count"], 1)
        self.assertEqual(c2["delivery_duration_missing"], 1)

    def test_infinite_and_negative_durations_are_excluded_even_if_flagged_true(self):
        features = _features()
        for customer_id in ("C3", "C4"):
            row = features.loc[customer_id]
            self.assertTrue(np.isnan(row["average_delivery_minutes"]))
            self.assertEqual(row["observed_delivery_duration_count"], 0)
            self.assertEqual(row["delivery_duration_missing"], 1)

    def test_legacy_duration_schema_fails_closed_as_unverified(self):
        orders = _order_rows().drop(columns="delivery_minutes_source_valid")
        features = _features(orders=orders)
        self.assertTrue(features["average_delivery_minutes"].isna().all())
        self.assertTrue(features["delivery_duration_missing"].eq(1).all())
        self.assertTrue(
            features["delivery_duration_provenance_unverified"].eq(1).all()
        )
        self.assertTrue(features["observed_delivery_duration_count"].eq(0).all())

    def test_post_cutoff_duration_changes_cannot_change_features(self):
        orders = _order_rows()
        changed = orders.copy()
        changed.loc[changed["order_ts"].ge(lapse.CUTOFF), "delivery_minutes"] = 1e9
        changed.loc[changed["order_ts"].ge(lapse.CUTOFF), "status"] = "cancelled"
        before = _features(orders=orders)
        after = _features(orders=changed)
        feature_columns = [column for column in before.columns if column != "lapsed"]
        pd.testing.assert_frame_equal(before[feature_columns], after[feature_columns])

    def test_label_window_orders_change_label_not_pre_cutoff_features(self):
        orders = _order_rows()
        without_label_event = orders.loc[orders["order_id"] != "O3"].copy()
        before = _features(orders=orders)
        after = _features(orders=without_label_event)
        feature_columns = [column for column in before.columns if column != "lapsed"]
        pd.testing.assert_frame_equal(before[feature_columns], after[feature_columns])
        self.assertEqual(before.loc["C1", "lapsed"], 0)
        self.assertEqual(after.loc["C1", "lapsed"], 1)

    def test_orders_after_label_window_do_not_change_lapse_label(self):
        orders = _order_rows()
        self.assertEqual(lapse.calculate_lapsed_label("C2", orders), 1)


class AgeAndSignupProvenanceTests(unittest.TestCase):
    def test_age_and_provenance_are_not_part_of_part_b_feature_snapshot(self):
        customers = _customer_rows()
        features = _features(customers=customers)
        self.assertIn("age", customers.columns)
        self.assertIn("age_source_valid", customers.columns)
        self.assertNotIn("age", features.columns)
        self.assertNotIn("age_missing", features.columns)
        self.assertNotIn("age_source_valid", features.columns)

    def test_age_values_cannot_change_part_b_features(self):
        customers = _customer_rows()
        changed = customers.copy()
        changed["age"] = [10000, -50, np.inf, np.nan]
        changed["age_source_valid"] = [0, 1, 1, 0]
        baseline = _features(customers=customers)
        altered = _features(customers=changed)
        pd.testing.assert_frame_equal(baseline, altered)

    def test_future_signup_date_is_rejected(self):
        customers = _customer_rows(
            C1={
                "signup_date": "2025-09-02",
                "signup_date_source_missing": 0,
                "signup_date_source_malformed": 0,
            }
        )
        with self.assertRaisesRegex(ValueError, "after the Part B prediction cutoff"):
            _features(customers=customers)

    def test_missing_and_malformed_signup_dates_remain_distinguishable(self):
        customers = _customer_rows(
            C1={
                "signup_date": pd.NaT,
                "signup_date_source_missing": 1,
                "signup_date_source_malformed": 0,
            },
            C2={
                "signup_date": pd.NaT,
                "signup_date_source_missing": 0,
                "signup_date_source_malformed": 1,
            },
        )
        features = _features(customers=customers)
        self.assertEqual(features.loc["C1", "signup_date_missing"], 1)
        self.assertEqual(features.loc["C1", "signup_date_malformed"], 0)
        self.assertTrue(np.isnan(features.loc["C1", "tenure_days"]))
        self.assertEqual(features.loc["C2", "signup_date_missing"], 0)
        self.assertEqual(features.loc["C2", "signup_date_malformed"], 1)
        self.assertTrue(np.isnan(features.loc["C2", "tenure_days"]))

    def test_legacy_signup_schema_raises_clear_error(self):
        customers = _customer_rows().drop(
            columns=["signup_date_source_missing", "signup_date_source_malformed"]
        )
        with self.assertRaisesRegex(ValueError, "signup-date provenance fields"):
            _features(customers=customers)

    def test_conflicting_signup_provenance_flags_are_rejected(self):
        customers = _customer_rows(
            C1={
                "signup_date_source_missing": 1,
                "signup_date_source_malformed": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "provenance flags are inconsistent"):
            _features(customers=customers)


class FoldPreprocessingTests(unittest.TestCase):
    @staticmethod
    def feature_frame(rows: int = 12) -> pd.DataFrame:
        index = np.arange(rows)
        data = {
            "recency": index.astype(float),
            "frequency": (index % 4 + 1).astype(float),
            "monetary": (index * 10 + 100).astype(float),
            "average_discount_pct": (index % 5).astype(float),
            "cancellation_rate": (index % 3 / 10).astype(float),
            "average_delivery_minutes": (index + 20).astype(float),
            "delivery_duration_missing": np.zeros(rows, dtype=float),
            "tenure_days": (index * 10 + 100).astype(float),
            "city": np.where(index % 2, "Pune", "Delhi"),
            "gender": np.where(index % 2, "female", "male"),
            "favourite_category": np.where(index % 2, "Grocery", "Dairy"),
        }
        data["average_delivery_minutes"][9:] = np.nan
        data["delivery_duration_missing"][9:] = 1.0
        return pd.DataFrame(data)[lapse.ALL_FEATURES]

    def test_holdout_delivery_values_do_not_influence_training_imputer(self):
        frame = self.feature_frame()
        training = frame.iloc[:8].copy()
        holdout = frame.iloc[8:].copy()
        holdout["average_delivery_minutes"] = 100000.0
        preprocessor = lapse.make_preprocessor().fit(training)
        imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        duration_index = lapse.NUMERIC_FEATURES.index("average_delivery_minutes")
        self.assertEqual(
            imputer.statistics_[duration_index],
            training["average_delivery_minutes"].median(),
        )
        self.assertNotEqual(
            imputer.statistics_[duration_index],
            holdout["average_delivery_minutes"].median(),
        )

    def test_each_cross_validation_fold_fits_its_own_duration_imputer(self):
        frame = self.feature_frame()
        labels = pd.Series(np.tile([0, 1], len(frame) // 2))
        splitter = StratifiedKFold(n_splits=3, shuffle=False)
        duration_index = lapse.NUMERIC_FEATURES.index("average_delivery_minutes")
        for train_indices, validation_indices in splitter.split(frame, labels):
            fold_training = frame.iloc[train_indices]
            fold_validation = frame.iloc[validation_indices].copy()
            fold_validation["average_delivery_minutes"] = 99999.0
            fitted = clone(lapse.make_preprocessor()).fit(fold_training)
            imputer = fitted.named_transformers_["numeric"].named_steps["imputer"]
            self.assertEqual(
                imputer.statistics_[duration_index],
                fold_training["average_delivery_minutes"].median(),
            )
            self.assertNotEqual(
                imputer.statistics_[duration_index],
                fold_validation["average_delivery_minutes"].median(),
            )

    def test_missingness_indicators_are_stable_when_training_has_no_missing_values(self):
        frame = self.feature_frame()
        training = frame.iloc[:8].copy()
        training["delivery_duration_missing"] = 0.0
        holdout = frame.iloc[8:9].copy()
        holdout["delivery_duration_missing"] = 1.0
        preprocessor = lapse.make_preprocessor().fit(training)
        transformed = preprocessor.transform(holdout)
        self.assertEqual(transformed.shape[1], preprocessor.transform(training).shape[1])
        self.assertIn("delivery_duration_missing", lapse.NUMERIC_FEATURES)
        self.assertNotIn("age_missing", lapse.ALL_FEATURES)


class HistoricalArtifactPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.before = {path: _sha256(path) for path in _protected_paths()}

    @classmethod
    def tearDownClass(cls) -> None:
        after = {path: _sha256(path) for path in cls.before}
        if after != cls.before:
            changed = [str(path) for path in cls.before if cls.before[path] != after[path]]
            raise AssertionError(f"Protected evidence changed during isolated tests: {changed}")

    def test_protected_evidence_inventory_is_present(self):
        self.assertTrue(any(path.is_relative_to(PHASE_25J_RUN) for path in self.before))
        self.assertIn(PROJECT_ROOT / "data" / "instamart.db", self.before)
        self.assertIn(PROJECT_ROOT / "outputs" / "refund_ledger.jsonl", self.before)
        self.assertIn(PROJECT_ROOT / "outputs" / "d8_resume_summary.json", self.before)
        f2_root = PROJECT_ROOT / "outputs" / "isolated_data_prep" / "phase-2.5NF1"
        self.assertEqual(
            sum(path.is_relative_to(f2_root) for path in self.before), 8
        )


if __name__ == "__main__":
    unittest.main()
