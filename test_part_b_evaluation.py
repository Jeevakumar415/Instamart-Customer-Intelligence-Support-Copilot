"""Isolated tests for Part B split and evaluation safeguards."""

from __future__ import annotations

import os
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier

import part_b_finalize
import part_b_lapse_model as lapse


def synthetic_dataset(rows: int = 200) -> pd.DataFrame:
    index = np.arange(rows)
    data = {
        "customer_id": [f"C{value:04d}" for value in index],
        "lapsed": (index % 2).astype(int),
        "recency": index.astype(float),
        "frequency": (index % 15 + 1).astype(float),
        "monetary": (index * 10 + 100).astype(float),
        "average_discount_pct": (index % 20).astype(float),
        "cancellation_rate": (index % 5 / 10).astype(float),
        "average_delivery_minutes": (index % 70 + 20).astype(float),
        "delivery_duration_missing": np.zeros(rows, dtype=float),
        "tenure_days": (index * 3 + 30).astype(float),
        "city": np.where(index % 2, "Delhi", "Mumbai"),
        "gender": np.where(index % 3, "F", "M"),
        "favourite_category": np.where(index % 4, "Grocery", "Dairy"),
    }
    data["recency"][3] = np.nan
    return pd.DataFrame(data)


def synthetic_input_provenance() -> dict:
    root = Path(__file__).resolve().parent / "synthetic-input-fixture"
    files = {
        name: {
            "resolved_path": str(root / name),
            "relative_path": name,
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
        for name in sorted(lapse.INPUT_REQUIRED_COLUMNS)
    }
    provenance = {
        "schema_version": lapse.INPUT_PROVENANCE_SCHEMA_VERSION,
        "feature_schema_version": lapse.FEATURE_SCHEMA_VERSION,
        "input_root": str(root),
        "input_files": files,
        "segmentation_artifact": {
            "resolved_path": str(root.parent / "customer_segments.csv"),
            "relative_path": None,
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
        "split_configuration": lapse.SPLIT_CONFIGURATION,
    }
    provenance["fingerprint"] = lapse.input_provenance_fingerprint(provenance)
    return provenance


def _write_test_state(*args, **kwargs):
    if kwargs.get("initial"):
        kwargs.setdefault(
            "input_fingerprint", synthetic_input_provenance()["fingerprint"]
        )
    return lapse.write_run_state(*args, **kwargs)


def _attempt_run_lock_in_child(run_root: str, run_id: str, result_queue) -> None:
    try:
        with lapse.acquire_run_lock(Path(run_root), run_id):
            result_queue.put("acquired")
    except lapse.RunLockBusyError:
        result_queue.put("busy")


def _write_finalizer_fixture(run_root: Path, run_id: str):
    lapse_dir = run_root / "lapse"
    lapse_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_id": run_id,
        "feature_dataset_fingerprint": "a" * 64,
        "schema_version": lapse.PROVENANCE_SCHEMA_VERSION,
        "input_provenance": synthetic_input_provenance(),
    }
    outputs = part_b_finalize.finalizer_output_paths(run_root)
    (lapse_dir / "test_metrics.json").write_text(
        json.dumps({
            "provenance": provenance,
            "evaluation_split": "test",
            "selected_model": "logistic_regression",
            "metrics": {
                "accuracy": 0.5,
                "precision_lapsed": 0.5,
                "recall_lapsed": 0.5,
                "f1_lapsed": 0.5,
                "confusion_matrix": [[1, 1], [1, 1]],
                "classification_report": {"lapsed": {"f1-score": 0.5}},
            },
        }),
        encoding="utf-8",
    )
    (run_root / "retention_list.csv").write_text(
        "customer_id,lapse_probability,actually_lapsed\nC1,0.8,1\n",
        encoding="utf-8",
    )
    (lapse_dir / "random_forest_feature_importances.csv").write_text(
        f"feature,importance,run_id,feature_dataset_fingerprint\nrecency,0.5,{run_id},{'a' * 64}\n",
        encoding="utf-8",
    )
    (lapse_dir / "top_five_feature_importances.csv").write_text(
        f"feature,importance,run_id,feature_dataset_fingerprint,source_feature,business_reading\nrecency,0.5,{run_id},{'a' * 64},recency,fixture\n",
        encoding="utf-8",
    )
    (lapse_dir / "logistic_regression_coefficients.csv").write_text(
        f"feature,coefficient,absolute_coefficient,run_id,feature_dataset_fingerprint\nrecency,0.4,0.4,{run_id},{'a' * 64}\n",
        encoding="utf-8",
    )
    (lapse_dir / "model_interpretability.json").write_text(
        json.dumps({
            "provenance": provenance,
            "top_five_random_forest_features": [],
            "three_logistic_regression_coefficients": [],
        }),
        encoding="utf-8",
    )
    (lapse_dir / "task8_validation.json").write_text(
        json.dumps({"provenance": provenance, "fixture_check": True}),
        encoding="utf-8",
    )
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
    joblib.dump(model, run_root / "lapse_model.joblib")
    metadata = {
        "provenance": provenance,
        "selected_model": "logistic_regression",
        "selected_params": {"classifier__C": 1.0},
        "model_file": "lapse_model.joblib",
        "retention_file": "retention_list.csv",
        "test_metrics_file": "test_metrics.json",
        "output_inventory": part_b_finalize.build_output_inventory(
            run_root, outputs, exclude_names={"finalizer_metadata.json"}
        ),
        "output_inventory_self_hash_note": (
            "finalizer_metadata.json is hashed in run_state.json to avoid a self-referential digest"
        ),
    }
    (run_root / "finalizer_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return outputs, metadata


class PartBEvaluationTests(unittest.TestCase):
    def test_age_is_absent_from_lapse_and_finalizer_feature_contracts(self) -> None:
        self.assertNotIn("age", lapse.NUMERIC_FEATURES)
        self.assertNotIn("age_missing", lapse.NUMERIC_FEATURES)
        self.assertNotIn("age", lapse.ALL_FEATURES)
        self.assertNotIn("age_missing", lapse.ALL_FEATURES)
        self.assertEqual(part_b_finalize.FEATURES, lapse.ALL_FEATURES)
        self.assertEqual(
            lapse.NUMERIC_FEATURES,
            [
                "recency",
                "frequency",
                "monetary",
                "average_discount_pct",
                "cancellation_rate",
                "average_delivery_minutes",
                "delivery_duration_missing",
                "tenure_days",
            ],
        )
        self.assertEqual(
            lapse.CATEGORICAL_FEATURES,
            ["city", "gender", "favourite_category"],
        )

    def test_split_input_selection_discards_age_columns_if_supplied(self) -> None:
        dataset = synthetic_dataset()
        dataset["age"] = np.arange(len(dataset), dtype=float) + 20
        dataset["age_missing"] = np.zeros(len(dataset), dtype=float)

        splits = lapse.stratified_splits(dataset)

        for split_name in ("train", "validation", "test"):
            features = splits[f"X_{split_name}"]
            self.assertEqual(features.columns.tolist(), lapse.ALL_FEATURES)
            self.assertNotIn("age", features.columns)
            self.assertNotIn("age_missing", features.columns)

    def test_duplicate_customer_ids_are_rejected_without_disclosing_ids(self) -> None:
        dataset = synthetic_dataset()
        dataset.loc[1, "customer_id"] = dataset.loc[0, "customer_id"]
        with self.assertRaisesRegex(ValueError, "duplicate_ids=1") as raised:
            lapse.validate_customer_ids(dataset)
        self.assertNotIn("C0000", str(raised.exception))

    def test_missing_customer_ids_are_rejected(self) -> None:
        dataset = synthetic_dataset()
        dataset.loc[0, "customer_id"] = "   "
        with self.assertRaisesRegex(ValueError, "missing_rows=1"):
            lapse.validate_customer_ids(dataset)

    def test_matching_run_provenance_is_accepted(self) -> None:
        features = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
        segments = pd.DataFrame({"customer_id": ["C1"], "is_premium": [False]})
        input_provenance = synthetic_input_provenance()
        provenance = lapse.build_run_provenance(
            "a" * 32, features, segments, input_provenance
        )
        returned = lapse.validate_provenance_bundle(
            provenance,
            [("feature metadata", dict(provenance))],
            features,
            segments,
            input_provenance,
        )
        self.assertEqual(returned, provenance)

    def test_lifecycle_run_id_must_match_artifact_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "run IDs or input fingerprints do not match"):
            lapse.validate_lifecycle_run_id(
                {"run_id": "a" * 32}, {"run_id": "b" * 32}
            )

    def test_feature_fingerprint_is_deterministic_and_config_sensitive(self) -> None:
        features = pd.DataFrame({"customer_id": ["C1", "C2"], "lapsed": [0, 1]})
        first = lapse.dataframe_fingerprint(
            features,
            purpose="lapse_features_and_labels",
            split_configuration=lapse.SPLIT_CONFIGURATION,
        )
        second = lapse.dataframe_fingerprint(
            features.copy(),
            purpose="lapse_features_and_labels",
            split_configuration=lapse.SPLIT_CONFIGURATION,
        )
        changed_config = lapse.dataframe_fingerprint(
            features,
            purpose="lapse_features_and_labels",
            split_configuration=dict(lapse.SPLIT_CONFIGURATION, random_state=99),
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed_config)

    def test_mismatched_run_id_is_rejected(self) -> None:
        features = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
        segments = pd.DataFrame({"customer_id": ["C1"], "is_premium": [False]})
        input_provenance = synthetic_input_provenance()
        provenance = lapse.build_run_provenance(
            "a" * 32, features, segments, input_provenance
        )
        stale = dict(provenance, run_id="b" * 32)
        with self.assertRaisesRegex(ValueError, "feature metadata provenance"):
            lapse.validate_provenance_bundle(
                provenance, [("feature metadata", stale)], features, segments,
                input_provenance,
            )

    def test_mismatched_feature_fingerprint_is_rejected(self) -> None:
        features = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
        segments = pd.DataFrame({"customer_id": ["C1"], "is_premium": [False]})
        input_provenance = synthetic_input_provenance()
        provenance = lapse.build_run_provenance(
            "a" * 32, features, segments, input_provenance
        )
        changed_features = features.copy()
        changed_features.loc[0, "lapsed"] = 1
        with self.assertRaisesRegex(ValueError, "Feature dataset fingerprint"):
            lapse.validate_provenance_bundle(
                provenance, [("feature metadata", provenance)], changed_features,
                segments, input_provenance,
            )

    def test_missing_or_malformed_provenance_is_rejected(self) -> None:
        features = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
        segments = pd.DataFrame({"customer_id": ["C1"], "is_premium": [False]})
        input_provenance = synthetic_input_provenance()
        with self.assertRaisesRegex(ValueError, "missing, malformed"):
            lapse.validate_provenance_bundle(
                None, [], features, segments, input_provenance
            )
        malformed = {
            "schema_version": lapse.PROVENANCE_SCHEMA_VERSION,
            "run_id": "not-a-uuid",
            "feature_dataset_fingerprint": "short",
            "segments_input_fingerprint": "short",
            "split_configuration": lapse.SPLIT_CONFIGURATION,
            "input_provenance": input_provenance,
        }
        with self.assertRaisesRegex(ValueError, "invalid run ID"):
            lapse.validate_provenance_bundle(
                malformed, [], features, segments, input_provenance
            )

    def test_previous_age_inclusive_feature_contract_is_rejected(self) -> None:
        features = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
        segments = pd.DataFrame({"customer_id": ["C1"], "is_premium": [False]})
        input_provenance = synthetic_input_provenance()
        provenance = lapse.build_run_provenance(
            "a" * 32, features, segments, input_provenance
        )
        provenance["schema_version"] = "instamart-part-b-run-v1"
        with self.assertRaisesRegex(ValueError, "schema version is incompatible"):
            lapse.validate_provenance_bundle(
                provenance,
                [],
                features,
                segments,
                input_provenance,
            )

    def test_output_directory_collision_preserves_existing_contents(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-collision-") as temp:
            run_dir = Path(temp) / "run-existing"
            run_dir.mkdir()
            marker = run_dir / "marker.txt"
            marker.write_text("preserve", encoding="utf-8")
            with patch.dict(os.environ, {lapse.RUN_DIRECTORY_ENV: str(run_dir)}):
                with self.assertRaises(FileExistsError):
                    lapse.create_part_b_run_directory()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_run_lock_is_released_on_context_exit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-lock-release-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "a" * 32
            with lapse.acquire_run_lock(root, run_id) as lock:
                lock.assert_owner()
            with lapse.acquire_run_lock(root, run_id) as reacquired:
                reacquired.assert_owner()

    def test_run_lock_rejects_a_different_run_id_owner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-lock-owner-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            with lapse.acquire_run_lock(root, "a" * 32):
                pass
            with self.assertRaisesRegex(ValueError, "different run ID"):
                with lapse.acquire_run_lock(root, "b" * 32):
                    self.fail("mismatched lock owner must not acquire")

    def test_concurrent_lock_acquisition_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-lock-concurrent-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "c" * 32
            context = multiprocessing.get_context("spawn")
            result_queue = context.Queue()
            with lapse.acquire_run_lock(root, run_id):
                child = context.Process(
                    target=_attempt_run_lock_in_child,
                    args=(str(root), run_id, result_queue),
                )
                child.start()
                child.join(15)
                if child.is_alive():
                    child.terminate()
                    child.join()
                    self.fail("child process timed out while checking the run lock")
                self.assertEqual(child.exitcode, 0)
                self.assertEqual(result_queue.get(timeout=2), "busy")

    def test_second_finalizer_is_rejected_while_run_lock_is_owned(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-second-finalizer-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "d" * 32
            input_provenance = synthetic_input_provenance()
            _write_test_state(
                root, run_id, "TRAINING_IN_PROGRESS", initial=True,
                input_fingerprint=input_provenance["fingerprint"],
            )
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            with patch.dict(os.environ, {lapse.RUN_DIRECTORY_ENV: str(root)}):
                with lapse.acquire_run_lock(root, run_id):
                    with self.assertRaises(lapse.RunLockBusyError):
                        part_b_finalize.main()

    def test_finalizer_output_collision_preserves_existing_contents(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-finalize-collision-") as temp:
            existing = Path(temp) / "test_metrics.json"
            existing.write_text("historical", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                part_b_finalize.ensure_finalizer_outputs_absent([existing])
            self.assertEqual(existing.read_text(encoding="utf-8"), "historical")

    def test_manifest_hashes_reject_modified_input_artifacts(self) -> None:
        required_files = {
            "lapse_cohort_features.csv",
            "feature_dataset_metadata.json",
            "split_summary.json",
            "tuning_results.json",
            "selected_model_summary.json",
            "validation_metrics.json",
        }
        with tempfile.TemporaryDirectory(prefix="part-b-manifest-") as temp:
            root = Path(temp)
            for name in required_files:
                (root / name).write_text(name, encoding="utf-8")
            manifest = {
                "artifact_sha256": {
                    name: lapse.file_sha256(root / name) for name in required_files
                }
            }
            lapse.validate_manifest_artifact_hashes(manifest, root)
            (root / "selected_model_summary.json").write_text(
                "changed", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                lapse.validate_manifest_artifact_hashes(manifest, root)

    def test_manifest_hashes_are_checked_before_dependent_artifact_parsing(self) -> None:
        required_files = {
            "lapse_cohort_features.csv",
            "feature_dataset_metadata.json",
            "split_summary.json",
            "tuning_results.json",
            "selected_model_summary.json",
            "validation_metrics.json",
        }
        with tempfile.TemporaryDirectory(prefix="part-b-verify-before-parse-") as temp:
            root = Path(temp)
            for name in required_files:
                (root / name).write_text("malformed", encoding="utf-8")
            manifest = {
                "artifact_sha256": {name: "0" * 64 for name in required_files}
            }
            with patch.object(part_b_finalize.pd, "read_csv") as read_csv:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    part_b_finalize.load_verified_training_artifacts(root, manifest)
            read_csv.assert_not_called()

    def test_scripts_use_same_explicit_output_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-run-root-") as temp:
            run_dir = Path(temp) / "run-unique"
            with patch.dict(os.environ, {lapse.RUN_DIRECTORY_ENV: str(run_dir)}):
                self.assertEqual(lapse.resolve_part_b_run_directory(), run_dir.resolve())
                self.assertEqual(
                    part_b_finalize.resolve_part_b_run_directory(), run_dir.resolve()
                )

    def test_finalizer_has_no_historical_output_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                part_b_finalize.resolve_part_b_run_directory()
        historical_outputs = Path(lapse.__file__).resolve().parent / "outputs"
        with patch.dict(
            os.environ,
            {lapse.RUN_DIRECTORY_ENV: str(historical_outputs)},
        ):
            with self.assertRaises(ValueError):
                part_b_finalize.resolve_part_b_run_directory()

    def test_relative_run_directory_is_rejected(self) -> None:
        with patch.dict(
            os.environ, {lapse.RUN_DIRECTORY_ENV: "relative-run-directory"}
        ):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                lapse.resolve_part_b_run_directory()

    def test_run_inputs_require_explicit_root_configuration(self) -> None:
        self.assertEqual(lapse.INPUT_ROOT_ENV, "INSTAMART_PART_B_INPUT_DIR")
        self.assertTrue(lapse.SEGMENTS_PATH.is_absolute())

    def test_output_path_traversal_and_external_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-path-guard-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "traversal"):
                lapse.ensure_safe_output_path(root / ".." / "outside.txt", root)
            with self.assertRaisesRegex(ValueError, "outside"):
                lapse.ensure_safe_output_path(Path(temp) / "outside.txt", root)

    def test_path_guard_rejects_detected_windows_reparse_point(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-reparse-guard-") as temp:
            root = Path(temp) / "run"
            (root / "lapse").mkdir(parents=True)
            original = lapse.is_symlink_or_reparse_point
            with patch.object(
                lapse,
                "is_symlink_or_reparse_point",
                side_effect=lambda path: Path(path).name == "lapse"
                or original(path),
            ):
                with self.assertRaisesRegex(ValueError, "redirected"):
                    lapse.ensure_safe_output_path(root / "lapse" / "artifact.json", root)

    def test_symlinked_lapse_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-symlink-inner-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            lapse_dir = root / "lapse"
            target = root / "target"
            target.mkdir()
            run_id = "a" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root,
                run_id,
                "TRAINING_COMPLETE",
                expected_state="TRAINING_IN_PROGRESS",
            )
            try:
                lapse_dir.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Windows symlink creation unavailable: {exc}")
            with patch.dict(os.environ, {lapse.RUN_DIRECTORY_ENV: str(root)}):
                with self.assertRaisesRegex(ValueError, "Symlinked lapse"):
                    part_b_finalize._finalize(root)

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-symlink-escape-") as temp:
            parent = Path(temp)
            root = parent / "run"
            outside = parent / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "lapse").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Windows symlink creation unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "Symlinked"):
                lapse.ensure_safe_output_path(root / "lapse" / "metrics.json", root)

    def test_dangling_symlink_counts_as_finalizer_collision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-dangling-link-") as temp:
            parent = Path(temp)
            dangling = parent / "dangling.json"
            try:
                dangling.symlink_to(parent / "missing-target.json")
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Windows symlink creation unavailable: {exc}")
            with self.assertRaises(FileExistsError):
                part_b_finalize.ensure_finalizer_outputs_absent([dangling])

    def test_all_finalizer_destinations_are_preflighted_before_writes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-all-output-paths-") as temp:
            root = Path(temp) / "run"
            (root / "lapse").mkdir(parents=True)
            paths = part_b_finalize.finalizer_output_paths(root)
            with patch.object(
                part_b_finalize, "ensure_safe_output_path", wraps=lapse.ensure_safe_output_path
            ) as validate_path:
                checked = part_b_finalize.preflight_finalizer_outputs(root)
            self.assertEqual(checked, paths)
            self.assertEqual(validate_path.call_count, len(paths))
            self.assertTrue(all(not path.exists() for path in paths))

    def test_incomplete_and_failed_training_runs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-training-state-") as temp:
            root = Path(temp) / "incomplete"
            root.mkdir()
            run_id = "b" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            with self.assertRaisesRegex(ValueError, "TRAINING_IN_PROGRESS"):
                lapse.require_run_state(root, "TRAINING_COMPLETE")
            with self.assertRaisesRegex(ValueError, "TRAINING_IN_PROGRESS"):
                part_b_finalize._finalize(root)

            failed_root = Path(temp) / "failed"
            failed_root.mkdir()
            failed_id = "c" * 32
            lapse.write_run_state(
                failed_root, failed_id, "TRAINING_IN_PROGRESS", initial=True
            )
            lapse.write_run_state(
                failed_root,
                failed_id,
                "FAILED",
                expected_state="TRAINING_IN_PROGRESS",
            )
            with self.assertRaisesRegex(ValueError, "FAILED"):
                lapse.require_run_state(failed_root, "TRAINING_COMPLETE")
            with self.assertRaisesRegex(ValueError, "FAILED"):
                part_b_finalize._finalize(failed_root)

    def test_training_complete_requires_verified_manifest_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-training-complete-") as temp:
            root = Path(temp) / "run"
            lapse_dir = root / "lapse"
            lapse_dir.mkdir(parents=True)
            run_id = "d" * 32
            input_provenance = synthetic_input_provenance()
            _write_test_state(
                root, run_id, "TRAINING_IN_PROGRESS", initial=True,
                input_fingerprint=input_provenance["fingerprint"],
            )
            features = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
            segments = pd.DataFrame(
                {"customer_id": ["C1"], "is_premium": [False]}
            )
            provenance = lapse.build_run_provenance(
                run_id, features, segments, input_provenance
            )
            names = {
                "lapse_cohort_features.csv",
                "feature_dataset_metadata.json",
                "split_summary.json",
                "tuning_results.json",
                "selected_model_summary.json",
                "validation_metrics.json",
            }
            for name in names:
                (lapse_dir / name).write_text("fixture", encoding="utf-8")
            manifest = {
                "provenance": provenance,
                "artifact_sha256": {
                    name: lapse.file_sha256(lapse_dir / name) for name in names
                },
            }
            artifact_provenances = [
                (name, provenance)
                for name in (
                    "feature metadata",
                    "split summary",
                    "tuning results",
                    "selection metadata",
                    "validation metrics",
                )
            ]
            lapse.complete_training_run(
                root,
                run_id,
                manifest,
                lapse_dir,
                artifact_provenances,
                features,
                segments,
                input_provenance,
            )
            self.assertEqual(
                lapse.require_run_state(root, "TRAINING_COMPLETE")["run_id"],
                run_id,
            )

    def test_manifest_failure_leaves_training_in_progress(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-training-incomplete-") as temp:
            root = Path(temp) / "run"
            lapse_dir = root / "lapse"
            lapse_dir.mkdir(parents=True)
            run_id = "e" * 32
            input_provenance = synthetic_input_provenance()
            _write_test_state(
                root, run_id, "TRAINING_IN_PROGRESS", initial=True,
                input_fingerprint=input_provenance["fingerprint"],
            )
            features = pd.DataFrame({"customer_id": ["C1"], "lapsed": [0]})
            segments = pd.DataFrame(
                {"customer_id": ["C1"], "is_premium": [False]}
            )
            provenance = lapse.build_run_provenance(
                run_id, features, segments, input_provenance
            )
            with self.assertRaises(ValueError):
                lapse.complete_training_run(
                    root,
                    run_id,
                    {"provenance": provenance, "artifact_sha256": {}},
                    lapse_dir,
                    [],
                    features,
                    segments,
                    input_provenance,
                )
            with self.assertRaisesRegex(ValueError, "TRAINING_IN_PROGRESS"):
                lapse.require_run_state(root, "TRAINING_COMPLETE")

    def test_finalization_complete_only_after_all_outputs_validate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-finalization-state-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "f" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root,
                run_id,
                "FINALIZATION_IN_PROGRESS",
                expected_state="TRAINING_COMPLETE",
            )
            outputs, expected_metadata = _write_finalizer_fixture(root, run_id)
            part_b_finalize.complete_finalization_run(
                root, run_id, outputs, expected_metadata
            )
            state = lapse.require_run_state(root, "FINALIZATION_COMPLETE")
            self.assertEqual(state["state"], "FINALIZATION_COMPLETE")
            self.assertEqual(set(state["finalization_integrity"]), lapse.FINALIZER_OUTPUT_NAMES)

    def test_missing_finalizer_output_prevents_completion_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-finalization-incomplete-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "2" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root,
                run_id,
                "FINALIZATION_IN_PROGRESS",
                expected_state="TRAINING_COMPLETE",
            )
            outputs, expected_metadata = _write_finalizer_fixture(root, run_id)
            outputs[-1].unlink()
            with self.assertRaisesRegex(ValueError, "missing or invalid"):
                part_b_finalize.complete_finalization_run(
                    root, run_id, outputs, expected_metadata
                )
            with self.assertRaisesRegex(ValueError, "FINALIZATION_IN_PROGRESS"):
                lapse.require_run_state(root, "FINALIZATION_COMPLETE")

    def test_output_hash_mismatch_blocks_finalization_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-output-hash-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "3" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root,
                run_id,
                "FINALIZATION_IN_PROGRESS",
                expected_state="TRAINING_COMPLETE",
            )
            outputs, metadata = _write_finalizer_fixture(root, run_id)
            (root / "lapse" / "top_five_feature_importances.csv").write_text(
                "corrupted", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "inventory or hash"):
                part_b_finalize.complete_finalization_run(root, run_id, outputs, metadata)
            with self.assertRaisesRegex(ValueError, "FINALIZATION_IN_PROGRESS"):
                lapse.require_run_state(root, "FINALIZATION_COMPLETE")

    def test_malformed_finalizer_csv_blocks_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-malformed-csv-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "4" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root, run_id, "FINALIZATION_IN_PROGRESS", expected_state="TRAINING_COMPLETE"
            )
            outputs, metadata = _write_finalizer_fixture(root, run_id)
            bad_path = root / "retention_list.csv"
            bad_path.write_text("wrong,columns\nonly,these\n", encoding="utf-8")
            metadata["output_inventory"] = part_b_finalize.build_output_inventory(
                root, outputs, exclude_names={"finalizer_metadata.json"}
            )
            (root / "finalizer_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Retention list"):
                part_b_finalize.complete_finalization_run(root, run_id, outputs, metadata)

    def test_malformed_finalizer_json_blocks_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-malformed-json-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "5" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root, run_id, "FINALIZATION_IN_PROGRESS", expected_state="TRAINING_COMPLETE"
            )
            outputs, metadata = _write_finalizer_fixture(root, run_id)
            (root / "lapse" / "model_interpretability.json").write_text(
                "{malformed", encoding="utf-8"
            )
            metadata["output_inventory"] = part_b_finalize.build_output_inventory(
                root, outputs, exclude_names={"finalizer_metadata.json"}
            )
            (root / "finalizer_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "malformed"):
                part_b_finalize.complete_finalization_run(root, run_id, outputs, metadata)

    def test_unloadable_model_blocks_finalization_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-invalid-model-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "6" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root, run_id, "FINALIZATION_IN_PROGRESS", expected_state="TRAINING_COMPLETE"
            )
            outputs, metadata = _write_finalizer_fixture(root, run_id)
            (root / "lapse_model.joblib").write_bytes(b"not a joblib model")
            metadata["output_inventory"] = part_b_finalize.build_output_inventory(
                root, outputs, exclude_names={"finalizer_metadata.json"}
            )
            (root / "finalizer_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "cannot be reloaded"):
                part_b_finalize.complete_finalization_run(root, run_id, outputs, metadata)

    def test_output_provenance_mismatch_blocks_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-output-provenance-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "7" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root, run_id, "FINALIZATION_IN_PROGRESS", expected_state="TRAINING_COMPLETE"
            )
            outputs, metadata = _write_finalizer_fixture(root, run_id)
            metrics_path = root / "lapse" / "test_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["provenance"]["run_id"] = "8" * 32
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            metadata["output_inventory"] = part_b_finalize.build_output_inventory(
                root, outputs, exclude_names={"finalizer_metadata.json"}
            )
            (root / "finalizer_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "inconsistent with run provenance"):
                part_b_finalize.complete_finalization_run(root, run_id, outputs, metadata)

    def test_interrupted_finalization_does_not_appear_complete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part-b-finalization-interrupt-") as temp:
            root = Path(temp) / "run"
            root.mkdir()
            run_id = "1" * 32
            _write_test_state(root, run_id, "TRAINING_IN_PROGRESS", initial=True)
            lapse.write_run_state(
                root, run_id, "TRAINING_COMPLETE", expected_state="TRAINING_IN_PROGRESS"
            )
            lapse.write_run_state(
                root,
                run_id,
                "FINALIZATION_IN_PROGRESS",
                expected_state="TRAINING_COMPLETE",
            )
            with self.assertRaisesRegex(ValueError, "FINALIZATION_IN_PROGRESS"):
                lapse.require_run_state(root, "FINALIZATION_COMPLETE")
            with self.assertRaisesRegex(ValueError, "FINALIZATION_IN_PROGRESS"):
                lapse.require_run_state(root, "TRAINING_COMPLETE")

    def test_stratified_partitions_are_disjoint_and_60_20_20(self) -> None:
        dataset = synthetic_dataset()
        splits = lapse.stratified_splits(dataset)
        id_sets = [
            set(splits[f"ids_{name}"].astype(str))
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(id_sets[0] & id_sets[1])
        self.assertFalse(id_sets[0] & id_sets[2])
        self.assertFalse(id_sets[1] & id_sets[2])
        self.assertEqual(set.union(*id_sets), set(dataset["customer_id"]))
        summary = lapse.summarize_splits(splits)
        self.assertAlmostEqual(summary["train_size"], 0.60, delta=0.02)
        self.assertAlmostEqual(summary["validation_size"], 0.20, delta=0.02)
        self.assertAlmostEqual(summary["test_size"], 0.20, delta=0.02)
        self.assertEqual(summary["split_roles"]["test"], "one final evaluation of the selected model only")

    def test_model_selection_uses_validation_f1_only(self) -> None:
        validation_metrics = {
            "logistic_regression": {"f1_lapsed": 0.71},
            "random_forest": {"f1_lapsed": 0.64},
        }
        # Test metrics are intentionally not accepted by the selection API.
        self.assertEqual(
            lapse.select_model_by_validation_f1(validation_metrics),
            "logistic_regression",
        )

    def test_numeric_imputer_learns_from_training_rows_only(self) -> None:
        dataset = synthetic_dataset()
        train = dataset.loc[:99, lapse.ALL_FEATURES].copy()
        validation_or_test = dataset.loc[100:, lapse.ALL_FEATURES].copy()
        validation_or_test.loc[:, "recency"] = 100000.0

        preprocessor = lapse.make_preprocessor().fit(train)
        numeric_imputer = (
            preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        )
        expected_median = float(train["recency"].median())
        self.assertEqual(float(numeric_imputer.statistics_[0]), expected_median)
        self.assertGreater(float(validation_or_test["recency"].median()), expected_median)

    def test_oversampling_uses_only_rows_supplied_as_training(self) -> None:
        X_train = pd.DataFrame(
            {
                "recency": [11.0, 12.0, 21.0, 22.0, 23.0, 24.0],
                "frequency": [1, 2, 3, 4, 5, 6],
                "monetary": [100, 110, 120, 130, 140, 150],
                "average_discount_pct": [0, 1, 2, 3, 4, 5],
                "cancellation_rate": [0, 0, 0, 0, 0, 0],
                "average_delivery_minutes": [30, 31, 32, 33, 34, 35],
                "delivery_duration_missing": [0, 0, 0, 0, 0, 0],
                "tenure_days": [100, 110, 120, 130, 140, 150],
                "city": ["A", "A", "B", "B", "C", "C"],
                "gender": ["F", "M", "F", "M", "F", "M"],
                "favourite_category": ["G", "G", "D", "D", "F", "F"],
            }
        )
        y_train = pd.Series([1, 1, 0, 0, 0, 0])
        balanced_X, balanced_y = lapse.oversample_training_data(X_train, y_train)
        self.assertEqual(balanced_y.value_counts().nunique(), 1)
        self.assertTrue(set(balanced_X["recency"]).issubset(set(X_train["recency"])))
        self.assertNotIn(999999.0, set(balanced_X["recency"]))

    def test_final_fit_uses_train_and_validation_not_test(self) -> None:
        splits = lapse.stratified_splits(synthetic_dataset())
        recorded = {}

        class FitRecorder:
            def fit(self, X, y):
                recorded["ids"] = set(X.index)
                recorded["rows"] = len(y)
                return self

        with patch.object(part_b_finalize, "build_model_pipeline", return_value=FitRecorder()):
            _, X_fit, y_fit = part_b_finalize.fit_selected_model_on_development(
                splits, "logistic_regression", {"classifier__C": 1}
            )
        expected = set(splits["X_train"].index) | set(splits["X_validation"].index)
        self.assertEqual(recorded["ids"], expected)
        self.assertEqual(len(X_fit), len(y_fit))
        self.assertEqual(len(X_fit), len(splits["X_train"]) + len(splits["X_validation"]))
        self.assertTrue(set(X_fit.index).isdisjoint(set(splits["X_test"].index)))

    def test_final_test_evaluation_calls_predict_once(self) -> None:
        class CountingModel:
            calls = 0

            def predict(self, X):
                self.calls += 1
                return np.zeros(len(X), dtype=int)

        model = CountingModel()
        result = part_b_finalize.final_test_evaluation(
            model,
            pd.DataFrame({"feature": [1, 2, 3, 4]}),
            pd.Series([0, 1, 0, 1]),
            "logistic_regression",
        )
        self.assertEqual(result["evaluation_split"], "test")
        self.assertEqual(model.calls, 1)

    def test_finalizer_rejects_nonvalidation_or_mismatched_configuration(self) -> None:
        tuning = {
            "tuning_split": "training_partition_cross_validation",
            "random_forest": {"best_params": {"classifier__max_depth": 3}},
        }
        validation_metrics = {
            "evaluation_split": "validation",
            "logistic_regression": {"f1_lapsed": 0.4},
            "random_forest": {"f1_lapsed": 0.7},
        }
        with self.assertRaises(ValueError):
            part_b_finalize.validate_selected_configuration(
                {
                    "selection_split": "test",
                    "selected_model": "random_forest",
                    "selected_params": {"classifier__max_depth": 3},
                    "candidate_models": ["logistic_regression", "random_forest"],
                    "selected_validation_f1": 0.7,
                },
                tuning,
                validation_metrics,
            )
        with self.assertRaises(ValueError):
            part_b_finalize.validate_selected_configuration(
                {
                    "selection_split": "validation",
                    "selected_model": "random_forest",
                    "selected_params": {"classifier__max_depth": 8},
                    "candidate_models": ["logistic_regression", "random_forest"],
                    "selected_validation_f1": 0.7,
                },
                tuning,
                validation_metrics,
            )
        with self.assertRaises(ValueError):
            part_b_finalize.validate_selected_configuration(
                {
                    "selection_split": "validation",
                    "selected_model": "logistic_regression",
                    "selected_params": {"classifier__C": 1},
                    "candidate_models": ["logistic_regression", "random_forest"],
                    "selected_validation_f1": 0.4,
                },
                {
                    "logistic_regression": {"best_params": {"classifier__C": 1}},
                    "tuning_split": "training_partition_cross_validation",
                    **tuning,
                },
                validation_metrics,
            )

    def test_finalizer_accepts_matching_validation_selected_configuration(self) -> None:
        selection = {
            "selection_split": "validation",
            "selected_model": "random_forest",
            "selected_params": {"classifier__max_depth": 3},
            "candidate_models": ["logistic_regression", "random_forest"],
            "selected_validation_f1": 0.7,
        }
        tuning = {
            "tuning_split": "training_partition_cross_validation",
            "random_forest": {"best_params": {"classifier__max_depth": 3}},
        }
        metrics = {
            "evaluation_split": "validation",
            "logistic_regression": {"f1_lapsed": 0.4},
            "random_forest": {"f1_lapsed": 0.7},
        }
        self.assertEqual(
            part_b_finalize.validate_selected_configuration(
                selection, tuning, metrics
            ),
            ("random_forest", {"classifier__max_depth": 3}),
        )


if __name__ == "__main__":
    unittest.main()
