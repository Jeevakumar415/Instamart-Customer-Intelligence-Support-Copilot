from pathlib import Path
import hashlib
import io
import json
import math
import os
import re
import stat
import uuid

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PART B — TASKS 4–7
# LAPSE LABELS, LEAKAGE-SAFE FEATURES, TUNING AND EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SEGMENTS_PATH = PROJECT_ROOT / "outputs" / "customer_segments.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
VALIDATION_FROM_DEVELOPMENT = VALIDATION_SIZE / (1 - TEST_SIZE)
POSITIVE_LABEL = 1
RUN_DIRECTORY_ENV = "INSTAMART_PART_B_RUN_DIR"
INPUT_ROOT_ENV = "INSTAMART_PART_B_INPUT_DIR"
PROVENANCE_SCHEMA_VERSION = "instamart-part-b-run-v3"
FEATURE_SCHEMA_VERSION = "instamart-part-b-features-v2"
INPUT_PROVENANCE_SCHEMA_VERSION = "instamart-part-b-input-v1"
RUN_STATE_SCHEMA_VERSION = "instamart-part-b-lifecycle-v3"
LEGACY_RUN_STATE_SCHEMA_VERSION = "instamart-part-b-lifecycle-v2"
RUN_STATE_FILENAME = "run_state.json"
RUN_LOCK_FILENAME = "run.lock"
RUN_STATES = {
    "TRAINING_IN_PROGRESS",
    "TRAINING_COMPLETE",
    "FINALIZATION_IN_PROGRESS",
    "FINALIZATION_COMPLETE",
    "FAILED",
}
FINALIZER_OUTPUT_NAMES = {
    "lapse/test_metrics.json",
    "lapse_model.joblib",
    "retention_list.csv",
    "finalizer_metadata.json",
    "lapse/random_forest_feature_importances.csv",
    "lapse/top_five_feature_importances.csv",
    "lapse/logistic_regression_coefficients.csv",
    "lapse/model_interpretability.json",
    "lapse/task8_validation.json",
}


class RunLockBusyError(RuntimeError):
    """Raised when another process currently owns the Part B run lock."""


class PartBRunLock:
    """Local OS-backed exclusive lock; it does not provide distributed locking."""

    def __init__(self, run_root, run_id):
        self.run_root = Path(run_root).resolve(strict=True)
        self.run_id = run_id
        self.owner_token = uuid.uuid4().hex
        self.path = self.run_root / RUN_LOCK_FILENAME
        self._handle = None
        self._locked = False

    def __enter__(self):
        self.path = ensure_safe_output_path(
            self.path, self.run_root, must_not_exist=False
        )
        try:
            self._handle = self.path.open("r+b")
        except FileNotFoundError:
            try:
                self._handle = self.path.open("x+b")
            except FileExistsError:
                self._handle = self.path.open("r+b")
        try:
            self._handle.seek(0, os.SEEK_END)
            if self._handle.tell() == 0:
                self._handle.write(b"\0")
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.seek(0)
            try:
                if os.name == "nt":
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as exc:
                raise RunLockBusyError(
                    "Another process already owns this Part B run lock."
                ) from exc
            self._locked = True

            self._handle.seek(1)
            previous_bytes = self._handle.read()
            if previous_bytes:
                try:
                    previous = json.loads(previous_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Part B run lock metadata is malformed.") from exc
                if not isinstance(previous, dict) or previous.get("run_id") != self.run_id:
                    raise ValueError("Part B run lock belongs to a different run ID.")

            metadata = {
                "run_id": self.run_id,
                "owner_token": self.owner_token,
                "process_id": os.getpid(),
            }
            self._handle.seek(1)
            self._handle.truncate(1)
            self._handle.write(json.dumps(metadata).encode("utf-8"))
            self._handle.flush()
            os.fsync(self._handle.fileno())
            return self
        except BaseException:
            self._release_os_lock()
            raise

    def assert_owner(self):
        if not self._locked or self._handle is None:
            raise RuntimeError("Part B run lock is not held by this process.")
        self._handle.seek(1)
        try:
            metadata = json.loads(self._handle.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Part B run lock ownership metadata is invalid.") from exc
        if (
            metadata.get("run_id") != self.run_id
            or metadata.get("owner_token") != self.owner_token
        ):
            raise RuntimeError("Part B run lock ownership mismatch.")

    def _release_os_lock(self):
        if self._handle is None:
            return
        if self._locked:
            try:
                self._handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._locked = False
        self._handle.close()
        self._handle = None

    def __exit__(self, exc_type, exc, traceback):
        ownership_error = None
        try:
            self.assert_owner()
        except RuntimeError as error:
            ownership_error = error
        finally:
            self._release_os_lock()
        if ownership_error is not None and exc is None:
            raise ownership_error
        return False


def acquire_run_lock(run_root, run_id):
    return PartBRunLock(run_root, run_id)
SPLIT_CONFIGURATION = {
    "train_fraction": 0.60,
    "validation_fraction": VALIDATION_SIZE,
    "test_fraction": TEST_SIZE,
    "validation_fraction_of_development": VALIDATION_FROM_DEVELOPMENT,
    "stratify_column": "lapsed",
    "random_state": RANDOM_STATE,
    "cv_folds": 5,
    "cv_shuffle": True,
    "cv_random_state": RANDOM_STATE,
    "cv_scoring": "f1",
    "selection_metric": "validation_f1_lapsed",
}

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "delivery_duration_missing",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

INPUT_REQUIRED_COLUMNS = {
    "customers.csv": {
        "customer_id", "city", "signup_date", "age", "gender",
        "age_source_valid", "signup_date_source_missing",
        "signup_date_source_malformed",
    },
    "orders.csv": {
        "order_id", "customer_id", "product_id", "quantity", "order_ts",
        "status", "discount_pct", "order_amount", "delivery_minutes",
        "delivery_minutes_source_valid",
    },
    "products.csv": {"product_id", "product_name", "category", "price_inr"},
    "tickets.csv": {"ticket_id", "customer_id", "order_id", "text", "true_category"},
    "policy_eval.csv": {"question", "expected_doc", "expected_fact"},
    "agent_scenarios.csv": {
        "scenario_id", "customer_id", "message", "expected_outcome",
        "expected_refund_inr",
    },}
INPUT_UNIQUE_IDENTIFIERS = {
    "customers.csv": "customer_id",
    "orders.csv": "order_id",
    "products.csv": "product_id",
    "tickets.csv": "ticket_id",
    "agent_scenarios.csv": "scenario_id",
}
SEGMENT_REQUIRED_COLUMNS = {
    "customer_id", "recency", "frequency", "monetary", "segment_id",
    "segment_name", "is_premium",
}


def resolve_part_b_run_directory() -> Path:
    """Return the explicitly configured isolated Part B run directory."""
    configured = os.environ.get(RUN_DIRECTORY_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            f"Set {RUN_DIRECTORY_ENV} to a new absolute run directory; "
            "there is no fallback to historical outputs."
        )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{RUN_DIRECTORY_ENV} must be an absolute path.")
    path = path.resolve()

    project_root = Path(__file__).resolve().parent
    dedicated_runs_root = (project_root / "outputs" / "part_b_runs").resolve()
    if path == project_root or path in project_root.parents:
        raise ValueError("The Part B run directory cannot be the project or its parent.")
    if project_root in path.parents and dedicated_runs_root not in path.parents:
        raise ValueError(
            "Project-local Part B outputs must be under outputs/part_b_runs/<run>."
        )
    if path == dedicated_runs_root:
        raise ValueError("Choose a unique child run directory under outputs/part_b_runs.")
    return path


def create_part_b_run_directory() -> Path:
    """Create a new run directory without reusing or overwriting existing state."""
    path = resolve_part_b_run_directory()
    path.mkdir(parents=True, exist_ok=False)
    return path


def ensure_safe_output_path(path, run_root, *, must_not_exist=True):
    """Validate a run output path across Windows path aliases and reject redirects."""
    root = Path(run_root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if ".." in candidate.parts:
        raise ValueError("Output path traversal is not allowed.")

    # Inspect the caller-supplied path before resolving it. Resolving first would
    # hide symlink/reparse components because Path.resolve() follows redirects.
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if is_symlink_or_reparse_point(current):
            raise ValueError("Symlinked or redirected Part B output paths are not allowed.")

    # On Windows, a path may use the 8.3 short name (for example JEEVAK~1)
    # while the configured root resolves to its long name (Jeevakumar).
    # Resolve both sides before doing containment/relative-path comparisons.
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        # Windows: 8.3 short-path vs long-path mismatch — re-resolve root
        # without strict=True so both sides share the same canonical form.
        root_reresolved = Path(str(root)).resolve()
        try:
            relative = resolved.relative_to(root_reresolved)
        except ValueError as exc:
            raise ValueError(
                "Output path is outside the configured Part B run directory."
            ) from exc
        root = root_reresolved
    if not relative.parts:
        raise ValueError("The run directory itself is not an output file path.")

    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if is_symlink_or_reparse_point(current):
            raise ValueError("Symlinked or redirected Part B output paths are not allowed.")
        if index < len(relative.parts) - 1 and os.path.lexists(current):
            if not current.is_dir():
                raise ValueError("An output parent path is not a directory.")

    if must_not_exist and os.path.lexists(resolved):
        raise FileExistsError("Part B output already exists; refusing to overwrite it.")
    return resolved


def is_symlink_or_reparse_point(path):
    path = Path(path)
    if path.is_symlink():
        return True
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(info, "st_file_attributes", 0)
    return bool(reparse_attribute and file_attributes & reparse_attribute)


def _read_run_state(run_root):
    state_path = ensure_safe_output_path(
        Path(run_root) / RUN_STATE_FILENAME, run_root, must_not_exist=False
    )
    if not state_path.is_file():
        raise ValueError("Part B run lifecycle state is missing or invalid.")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Part B run lifecycle state is malformed.") from exc
    legacy_fields = {"schema_version", "run_id", "state", "finalization_integrity"}
    current_fields = legacy_fields | {"input_fingerprint"}
    schema_version = state.get("schema_version") if isinstance(state, dict) else None
    expected_fields = (
        legacy_fields
        if schema_version == LEGACY_RUN_STATE_SCHEMA_VERSION
        else current_fields
    )
    if (
        not isinstance(state, dict)
        or set(state) != expected_fields
        or schema_version not in {RUN_STATE_SCHEMA_VERSION, LEGACY_RUN_STATE_SCHEMA_VERSION}
        or not isinstance(state.get("run_id"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", state["run_id"])
        or state.get("state") not in RUN_STATES
    ):
        raise ValueError("Part B run lifecycle state is malformed or unsupported.")
    if schema_version == RUN_STATE_SCHEMA_VERSION:
        fingerprint = state.get("input_fingerprint")
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        ):
            raise ValueError("Part B lifecycle input fingerprint is malformed.")
    integrity = state["finalization_integrity"]
    if state["state"] == "FINALIZATION_COMPLETE":
        if not isinstance(integrity, dict) or set(integrity) != FINALIZER_OUTPUT_NAMES:
            raise ValueError("Completed Part B run has invalid output integrity metadata.")
        for record in integrity.values():
            if (
                not isinstance(record, dict)
                or set(record) != {"sha256", "size_bytes"}
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] < 0
            ):
                raise ValueError("Completed Part B run has malformed output hashes.")
    elif integrity is not None:
        raise ValueError("Incomplete Part B run cannot contain completion integrity data.")
    return state


def read_run_state(run_root):
    """Read and validate the lifecycle marker without changing it."""
    return _read_run_state(run_root)


def write_run_state(
    run_root,
    run_id,
    state,
    *,
    expected_state=None,
    initial=False,
    finalization_integrity=None,
    input_fingerprint=None,
):
    """Atomically create or advance the lifecycle marker for this run."""
    if state not in RUN_STATES:
        raise ValueError("Unsupported Part B lifecycle state.")
    if state == "FINALIZATION_COMPLETE":
        if (
            not isinstance(finalization_integrity, dict)
            or set(finalization_integrity) != FINALIZER_OUTPUT_NAMES
        ):
            raise ValueError("Finalization completion requires the full output inventory.")
    elif finalization_integrity is not None:
        raise ValueError("Only completed runs may record finalization integrity.")
    if input_fingerprint is not None and (
        not isinstance(input_fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", input_fingerprint)
    ):
        raise ValueError("Part B lifecycle input fingerprint is malformed.")
    root = Path(run_root).resolve(strict=True)
    state_path = ensure_safe_output_path(
        root / RUN_STATE_FILENAME, root, must_not_exist=initial
    )
    if initial:
        if state != "TRAINING_IN_PROGRESS":
            raise ValueError("A new Part B run must begin in TRAINING_IN_PROGRESS.")
        if os.path.lexists(state_path):
            raise FileExistsError("Part B lifecycle state already exists.")
        current_input_fingerprint = input_fingerprint
    else:
        previous = _read_run_state(root)
        if previous["run_id"] != run_id or previous["state"] != expected_state:
            raise ValueError("Unsafe Part B lifecycle state transition.")
        previous_fingerprint = previous.get("input_fingerprint")
        if (
            input_fingerprint is not None
            and previous_fingerprint is not None
            and input_fingerprint != previous_fingerprint
        ):
            raise ValueError("Part B lifecycle input fingerprint cannot change during a run.")
        current_input_fingerprint = (
            input_fingerprint if input_fingerprint is not None else previous_fingerprint
        )
        allowed = {
            ("TRAINING_IN_PROGRESS", "TRAINING_COMPLETE"),
            ("TRAINING_IN_PROGRESS", "FAILED"),
            ("TRAINING_COMPLETE", "FINALIZATION_IN_PROGRESS"),
            ("FINALIZATION_IN_PROGRESS", "FINALIZATION_COMPLETE"),
            ("FINALIZATION_IN_PROGRESS", "FAILED"),
        }
        if (expected_state, state) not in allowed:
            raise ValueError("Unsupported Part B lifecycle state transition.")

    payload = {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "state": state,
        "finalization_integrity": finalization_integrity,
        "input_fingerprint": current_input_fingerprint,
    }
    temp_path = root / f".run_state-{uuid.uuid4().hex}.tmp"
    ensure_safe_output_path(temp_path, root)
    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ensure_safe_output_path(state_path, root, must_not_exist=initial)
        os.replace(temp_path, state_path)
    finally:
        if os.path.lexists(temp_path):
            temp_path.unlink()
    return payload


def require_run_state(run_root, expected_state):
    state = _read_run_state(run_root)
    if state["state"] != expected_state:
        raise ValueError(
            f"Part B run is {state['state']}; expected {expected_state}."
        )
    return state
def validate_lifecycle_run_id(run_state, provenance):
    input_provenance = (
        provenance.get("input_provenance") if isinstance(provenance, dict) else None
    )
    if (
        not isinstance(run_state, dict)
        or not isinstance(provenance, dict)
        or run_state.get("run_id") != provenance.get("run_id")
        or not isinstance(input_provenance, dict)
        or run_state.get("input_fingerprint") != input_provenance.get("fingerprint")
    ):
        raise ValueError(
            "Lifecycle and artifact provenance run IDs or input fingerprints do not match."
        )


def complete_training_run(
    run_root, run_id, manifest, lapse_dir, artifact_provenances, features,
    segments, input_provenance,
):
    """Validate the final training handoff before marking it ready to finalize."""
    validate_manifest_artifact_hashes(manifest, lapse_dir)
    provenance = manifest.get("provenance") if isinstance(manifest, dict) else None
    state = _read_run_state(run_root)
    if state["run_id"] != run_id:
        raise ValueError("Training run ID does not match lifecycle state.")
    validate_lifecycle_run_id(state, provenance)
    validate_provenance_bundle(
        provenance,
        artifact_provenances,
        features,
        segments,
        input_provenance,
    )
    write_run_state(
        run_root,
        run_id,
        "TRAINING_COMPLETE",
        expected_state="TRAINING_IN_PROGRESS",
    )


def validate_customer_ids(dataset):
    """Reject missing/duplicate IDs without exposing customer identifiers."""
    if "customer_id" not in dataset.columns:
        raise ValueError("Feature dataset is missing the customer_id column.")
    normalized = dataset["customer_id"].astype("string").str.strip()
    missing = normalized.isna() | normalized.eq("")
    duplicate_rows = normalized.loc[~missing]
    duplicate_mask = duplicate_rows.duplicated(keep=False)
    distinct_duplicate_ids = int(duplicate_rows.loc[duplicate_mask].nunique())
    duplicate_extra_rows = int(duplicate_rows.duplicated(keep="first").sum())
    missing_rows = int(missing.sum())
    if missing_rows or distinct_duplicate_ids:
        raise ValueError(
            "Invalid customer IDs: "
            f"missing_rows={missing_rows}; "
            f"duplicate_ids={distinct_duplicate_ids}; "
            f"duplicate_extra_rows={duplicate_extra_rows}."
        )
    return normalized


def dataframe_fingerprint(dataframe, *, purpose, split_configuration=None):
    """Hash canonical ordered values, columns/dtypes, and supplied context."""
    descriptor = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "purpose": purpose,
        "columns": [str(column) for column in dataframe.columns],
        "dtypes": [str(dtype) for dtype in dataframe.dtypes],
        "split_configuration": split_configuration,
        "row_count": int(len(dataframe)),
    }

    def canonical_cell(value):
        if pd.isna(value):
            return ["null"]
        if isinstance(value, (bool, np.bool_)):
            return ["bool", bool(value)]
        if isinstance(value, (int, np.integer)):
            return ["int", str(int(value))]
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if math.isnan(number):
                return ["null"]
            return ["float", number.hex()]
        if isinstance(value, pd.Timestamp):
            return ["timestamp", value.isoformat()]
        return ["str", str(value)]

    payload = {
        "descriptor": descriptor,
        "rows": [
            [canonical_cell(value) for value in row]
            for row in dataframe.itertuples(index=False, name=None)
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _reject_input_reparse_components(path):
    """Reject symlink/reparse components before resolving an input path."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("Part B input paths must be absolute.")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if is_symlink_or_reparse_point(current):
            raise ValueError("Symlinked or redirected Part B input paths are not allowed.")


def _read_input_file(path, root=None):
    """Read and hash one regular input file, enforcing its declared boundary."""
    path = Path(path)
    _reject_input_reparse_components(path)
    resolved = path.resolve(strict=True)
    if root is not None:
        root = Path(root).resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Part B input file is outside the configured input root.") from exc
        if not relative.parts or ".." in relative.parts:
            raise ValueError("Part B input path traversal is not allowed.")
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Part B inputs must be regular files.")
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise RuntimeError("Part B input changed while it was being read.")
    return payload, resolved, len(payload), _sha256_bytes(payload)


def _validate_input_frame(filename, frame):
    required = INPUT_REQUIRED_COLUMNS[filename]
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Part B input {filename} has an incompatible schema; "
            f"missing columns: {sorted(missing)}."
        )
    if frame.empty:
        raise ValueError(f"Part B input {filename} contains no data rows.")
    identifier = INPUT_UNIQUE_IDENTIFIERS.get(filename)
    if identifier:
        values = frame[identifier].astype("string").str.strip()
        if values.isna().any() or values.eq("").any() or values.duplicated().any():
            raise ValueError(
                f"Part B input {filename} has missing or duplicate {identifier} values."
            )
    if filename == "customers.csv":
        for column in (
            "age_source_valid",
            "signup_date_source_missing",
            "signup_date_source_malformed",
        ):
            flags = frame[column].astype("string").str.strip().str.lower()
            if not flags.isin({"0", "1", "true", "false"}).all():
                raise ValueError(f"Part B input customers.csv has invalid {column} flags.")
        missing_flag = _source_valid_flags(frame["signup_date_source_missing"])
        malformed_flag = _source_valid_flags(frame["signup_date_source_malformed"])
        parsed_signup = pd.to_datetime(frame["signup_date"], errors="coerce")
        if (missing_flag & malformed_flag).any():
            raise ValueError("Part B input customer signup flags are inconsistent.")
        if ((missing_flag | malformed_flag) & parsed_signup.notna()).any():
            raise ValueError("Part B input customer signup flags conflict with dates.")
        if (~(missing_flag | malformed_flag) & parsed_signup.isna()).any():
            raise ValueError("Part B input customer signup provenance is incomplete.")
        if (parsed_signup.dropna() > CUTOFF).any():
            raise ValueError("Part B input contains signup dates after the prediction cutoff.")
    elif filename == "orders.csv":
        flags = frame["delivery_minutes_source_valid"].astype("string").str.strip().str.lower()
        if not flags.isin({"0", "1", "true", "false"}).all():
            raise ValueError("Part B input orders.csv has invalid duration provenance flags.")
        valid = _source_valid_flags(frame["delivery_minutes_source_valid"])
        durations = pd.to_numeric(frame["delivery_minutes"], errors="coerce")
        if (valid & (~durations.notna() | ~np.isfinite(durations) | durations.lt(0))).any():
            raise ValueError("Part B input marks an invalid duration as source-valid.")
        parsed_order_ts = pd.to_datetime(frame["order_ts"], errors="coerce")
        if parsed_order_ts.isna().any():
            raise ValueError("Part B input orders.csv contains invalid order timestamps.")
    elif filename == "products.csv":
        values = frame["product_id"].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError("Part B input products.csv has missing product IDs.")


def _record_input_file(path, relative_path, root=None):
    payload, resolved, size_bytes, digest = _read_input_file(path, root=root)
    return {
        "resolved_path": str(resolved),
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "sha256": digest,
    }, payload


def input_provenance_fingerprint(input_provenance):
    payload = {
        key: value
        for key, value in input_provenance.items()
        if key != "fingerprint"
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def load_part_b_inputs(input_root=None, *, segments_path=None):
    """Load all six cleaned datasets from one explicit root and bind segments."""
    if input_root is None:
        input_root = os.environ.get(INPUT_ROOT_ENV)
    if not isinstance(input_root, (str, os.PathLike)) or not str(input_root).strip():
        raise ValueError(f"Set the explicit {INPUT_ROOT_ENV} input-root setting.")
    configured_root = Path(input_root)
    if not configured_root.is_absolute():
        raise ValueError("The Part B input root must be an absolute path.")
    if ".." in configured_root.parts:
        raise ValueError("Part B input-root traversal is not allowed.")
    _reject_input_reparse_components(configured_root)
    root = configured_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The configured Part B input root is not a directory.")

    dataframes = {}
    file_records = {}
    for filename in sorted(INPUT_REQUIRED_COLUMNS):
        candidate = root / filename
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Part B input path traversal is not allowed.") from exc
        try:
            record, payload = _record_input_file(candidate, relative, root=root)
        except FileNotFoundError as exc:
            raise ValueError(f"Required Part B input file is missing: {filename}.") from exc
        try:
            frame = pd.read_csv(io.BytesIO(payload))
        except Exception as exc:
            raise ValueError(f"Part B input {filename} is not a valid CSV.") from exc
        _validate_input_frame(filename, frame)
        file_records[filename] = record
        dataframes[filename] = frame

    customers = dataframes["customers.csv"]
    orders = dataframes["orders.csv"]
    products = dataframes["products.csv"]
    customer_ids = set(customers["customer_id"].astype("string").str.strip())
    product_ids = set(products["product_id"].astype("string").str.strip())
    if not set(orders["customer_id"].astype("string").str.strip()).issubset(customer_ids):
        raise ValueError("Part B orders reference customers outside the configured input root.")
    if not set(orders["product_id"].astype("string").str.strip()).issubset(product_ids):
        raise ValueError("Part B orders reference products outside the configured input root.")
    tickets = dataframes["tickets.csv"]
    if not set(tickets["customer_id"].astype("string").str.strip()).issubset(customer_ids):
        raise ValueError("Part B tickets reference customers outside the configured input root.")
    ticket_order_ids = tickets["order_id"].astype("string").str.strip()
    ticket_order_ids = ticket_order_ids[ticket_order_ids.notna() & ticket_order_ids.ne("")]
    if not set(ticket_order_ids).issubset(
        set(orders["order_id"].astype("string").str.strip())
    ):
        raise ValueError("Part B tickets reference orders outside the configured input root.")
    scenarios = dataframes["agent_scenarios.csv"]
    if not set(scenarios["customer_id"].astype("string").str.strip()).issubset(customer_ids):
        raise ValueError("Part B scenarios reference customers outside the configured input root.")

    segments_path = Path(segments_path) if segments_path is not None else SEGMENTS_PATH
    if not segments_path.is_absolute():
        raise ValueError("The Part B segmentation dependency path must be absolute.")
    segment_record, segment_payload = _record_input_file(
        segments_path, relative_path=None
    )
    try:
        segments = pd.read_csv(io.BytesIO(segment_payload))
    except Exception as exc:
        raise ValueError("The Part B segmentation dependency is not a valid CSV.") from exc
    missing_segment_columns = SEGMENT_REQUIRED_COLUMNS - set(segments.columns)
    if missing_segment_columns:
        raise ValueError(
            "The Part B segmentation dependency has an incompatible schema; "
            f"missing columns: {sorted(missing_segment_columns)}"
        )
    segment_ids = segments["customer_id"].astype("string").str.strip()
    if segment_ids.isna().any() or segment_ids.eq("").any() or segment_ids.duplicated().any():
        raise ValueError("The Part B segmentation dependency has invalid customer IDs.")
    if not set(segment_ids).issubset(customer_ids):
        raise ValueError("The segmentation dependency references customers outside the configured input root.")

    provenance = {
        "schema_version": INPUT_PROVENANCE_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_root": str(root),
        "input_files": file_records,
        "segmentation_artifact": segment_record,
        "split_configuration": SPLIT_CONFIGURATION,
    }
    provenance["fingerprint"] = input_provenance_fingerprint(provenance)
    return {
        "root": root,
        "datasets": dataframes,
        "segments": segments,
        "provenance": provenance,
    }


def validate_input_provenance(expected, actual):
    """Require exact source-file and configuration identity at finalization."""
    required = {
        "schema_version", "feature_schema_version", "input_root",
        "input_files", "segmentation_artifact", "split_configuration",
        "fingerprint",
    }
    if not isinstance(expected, dict) or set(expected) != required:
        raise ValueError("Input provenance is missing, malformed, or unsupported.")
    if expected.get("schema_version") != INPUT_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Input provenance schema version is incompatible.")
    if expected.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Input feature schema version is incompatible.")
    root = expected.get("input_root")
    root_path = Path(root) if isinstance(root, str) else None
    if (
        root_path is None
        or not root_path.is_absolute()
        or ".." in root_path.parts
        or set(expected.get("input_files", {})) != set(INPUT_REQUIRED_COLUMNS)
    ):
        raise ValueError("Input provenance root or dataset inventory is invalid.")
    for filename, record in expected["input_files"].items():
        if (
            not isinstance(record, dict)
            or set(record) != {"resolved_path", "relative_path", "size_bytes", "sha256"}
            or record.get("relative_path") != filename
            or not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            or not Path(record.get("resolved_path", "")).is_absolute()
        ):
            raise ValueError(f"Input provenance record is invalid for {filename}.")
        # Normalize both sides to resolve Windows 8.3 short-path vs long-path.
        if Path(record["resolved_path"]).resolve().parent != root_path.resolve():
            raise ValueError(f"Input provenance path escapes the configured root: {filename}.")
    segment_record = expected.get("segmentation_artifact")
    if (
        not isinstance(segment_record, dict)
        or set(segment_record) != {"resolved_path", "relative_path", "size_bytes", "sha256"}
        or segment_record.get("relative_path") is not None
        or not isinstance(segment_record.get("size_bytes"), int)
        or segment_record["size_bytes"] < 0
        or not isinstance(segment_record.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", segment_record["sha256"])
        or not isinstance(segment_record.get("resolved_path"), str)
        or not Path(segment_record["resolved_path"]).is_absolute()
    ):
        raise ValueError("Segmentation artifact provenance is invalid.")
    if expected.get("split_configuration") != SPLIT_CONFIGURATION:
        raise ValueError("Input provenance split configuration is incompatible.")
    if not isinstance(expected.get("fingerprint"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected["fingerprint"]
    ):
        raise ValueError("Input provenance fingerprint is malformed.")
    if input_provenance_fingerprint(expected) != expected["fingerprint"]:
        raise ValueError("Input provenance fingerprint failed its integrity check.")
    if not isinstance(actual, dict) or actual != expected:
        raise ValueError("Current input files do not match training input provenance.")
    return expected


def validate_manifest_artifact_hashes(manifest, lapse_dir):
    expected_files = {
        "lapse_cohort_features.csv",
        "feature_dataset_metadata.json",
        "split_summary.json",
        "tuning_results.json",
        "selected_model_summary.json",
        "validation_metrics.json",
    }
    artifact_hashes = manifest.get("artifact_sha256") if isinstance(manifest, dict) else None
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_files:
        raise ValueError("Run manifest artifact hashes are missing or incompatible.")
    for filename in expected_files:
        expected_hash = artifact_hashes[filename]
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"Run manifest has an invalid hash for {filename}.")
        if file_sha256(Path(lapse_dir) / filename) != expected_hash:
            raise ValueError(f"Run artifact hash mismatch: {filename}.")


def build_run_provenance(run_id, feature_dataset, segments, input_provenance):
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": run_id,
        "input_provenance": input_provenance,
        "feature_dataset_fingerprint": dataframe_fingerprint(
            feature_dataset,
            purpose="lapse_features_and_labels",
            split_configuration=SPLIT_CONFIGURATION,
        ),
        "segments_input_fingerprint": dataframe_fingerprint(
            segments, purpose="customer_segments_input"
        ),
        "split_configuration": SPLIT_CONFIGURATION,
    }


def validate_provenance_bundle(
    expected_provenance,
    artifact_provenances,
    feature_dataset,
    segments,
    input_provenance,
):
    """Fail closed unless all artifacts identify the same data and run."""
    expected_fields = {
        "schema_version",
        "run_id",
        "feature_dataset_fingerprint",
        "segments_input_fingerprint",
        "split_configuration",
        "input_provenance",
    }
    if not isinstance(expected_provenance, dict) or set(expected_provenance) != expected_fields:
        raise ValueError("Run provenance is missing, malformed, or unsupported.")
    if expected_provenance["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Run provenance schema version is incompatible.")
    if not isinstance(expected_provenance["run_id"], str) or not re.fullmatch(
        r"[0-9a-f]{32}", expected_provenance["run_id"]
    ):
        raise ValueError("Run provenance has an invalid run ID.")
    for field in ("feature_dataset_fingerprint", "segments_input_fingerprint"):
        if not isinstance(expected_provenance[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_provenance[field]
        ):
            raise ValueError(f"Run provenance has an invalid {field}.")
    if expected_provenance["split_configuration"] != SPLIT_CONFIGURATION:
        raise ValueError("Run split configuration is incompatible.")
    validate_input_provenance(
        expected_provenance["input_provenance"], input_provenance
    )

    actual_feature_fingerprint = dataframe_fingerprint(
        feature_dataset,
        purpose="lapse_features_and_labels",
        split_configuration=SPLIT_CONFIGURATION,
    )
    actual_segments_fingerprint = dataframe_fingerprint(
        segments, purpose="customer_segments_input"
    )
    if expected_provenance["feature_dataset_fingerprint"] != actual_feature_fingerprint:
        raise ValueError("Feature dataset fingerprint does not match run provenance.")
    if expected_provenance["segments_input_fingerprint"] != actual_segments_fingerprint:
        raise ValueError("Customer segments input fingerprint does not match provenance.")

    for artifact_name, artifact_provenance in artifact_provenances:
        if not isinstance(artifact_provenance, dict) or artifact_provenance != expected_provenance:
            raise ValueError(
                f"{artifact_name} provenance is missing or mismatched for this run."
            )
    return expected_provenance


def _source_valid_flags(series):
    """Convert a provenance flag column to a boolean Series."""
    normalised = series.astype("string").str.strip().str.lower()
    return normalised.isin({"1", "true"})


def calculate_lapsed_label(customer_id, completed_orders):
    """Return 1 when the customer placed no completed orders in the label window."""
    completed_orders = completed_orders.copy()
    completed_orders["order_ts"] = pd.to_datetime(
        completed_orders["order_ts"], errors="coerce"
    )
    window = completed_orders.loc[
        completed_orders["customer_id"].astype("string").str.strip().eq(str(customer_id))
        & completed_orders["order_ts"].ge(CUTOFF)
        & completed_orders["order_ts"].lt(LABEL_END)
    ]
    return int(len(window) == 0)


def stratified_splits(dataset):
    """Return disjoint 60/20/20 train, validation, and test partitions."""
    customer_ids = validate_customer_ids(dataset)
    X = dataset[ALL_FEATURES].copy()
    y = dataset["lapsed"].astype(int).copy()
    customer_ids = customer_ids.astype(str)

    X_development, X_test, y_development, y_test, ids_development, ids_test = (
        train_test_split(
            X,
            y,
            customer_ids,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )
    X_train, X_validation, y_train, y_validation, ids_train, ids_validation = (
        train_test_split(
            X_development,
            y_development,
            ids_development,
            test_size=VALIDATION_FROM_DEVELOPMENT,
            stratify=y_development,
            random_state=RANDOM_STATE,
        )
    )
    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_validation": ids_validation,
        "ids_test": ids_test,
    }


def summarize_splits(splits):
    """Return a JSON-serialisable summary of partition sizes and configuration."""
    total = sum(
        len(splits[f"y_{name}"])
        for name in ("train", "validation", "test")
    )
    return {
        "train_size": len(splits["y_train"]) / total,
        "validation_size": len(splits["y_validation"]) / total,
        "test_size": len(splits["y_test"]) / total,
        "train_rows": len(splits["y_train"]),
        "validation_rows": len(splits["y_validation"]),
        "test_rows": len(splits["y_test"]),
        "total_rows": total,
        "random_state": RANDOM_STATE,
        "split_roles": {
            "train": "hyperparameter search and model fitting",
            "validation": "model selection",
            "test": "one final evaluation of the selected model only",
        },
    }


def select_model_by_validation_f1(candidates):
    """Return the name of the model with the highest validation F1."""
    best_name = None
    best_f1 = -1.0
    for name, metrics in candidates.items():
        if not isinstance(metrics, dict):
            continue
        score = metrics.get("f1_lapsed", -1.0)
        if score > best_f1:
            best_f1 = score
            best_name = name
    return best_name


def make_preprocessor():
    """Return a ColumnTransformer that imputes and scales numeric features
    and one-hot encodes categoricals.  Fitting uses only supplied data."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipeline(model_name, params=None):
    """Build a preprocessing + classifier pipeline for the named model."""
    params = params or {}
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            **{k.replace("classifier__", ""): v for k, v in params.items()},
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            **{k.replace("classifier__", ""): v for k, v in params.items()},
        )
    else:
        raise ValueError(f"Unsupported lapse model: {model_name}")
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier),
    ])


def evaluate_model(model, X, y, label):
    """Return the binary metrics for the positive (lapsed) class."""
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_lapsed": float(
            precision_score(y, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall_lapsed": float(
            recall_score(y, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1_lapsed": float(
            f1_score(y, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["not_lapsed", "lapsed"],
            output_dict=True,
            zero_division=0,
        ),
    }


def oversample_training_data(X_train, y_train):
    """Balance the training partition by repeating minority-class rows.
    Only rows present in X_train are ever duplicated — no synthetic data."""
    majority_label = int(y_train.value_counts().idxmax())
    minority_label = 1 - majority_label
    minority_mask = y_train.eq(minority_label)
    majority_count = int(y_train.eq(majority_label).sum())
    minority_count = int(minority_mask.sum())
    if minority_count == 0 or minority_count >= majority_count:
        return X_train, y_train
    shortfall = majority_count - minority_count
    minority_X = X_train.loc[minority_mask]
    minority_y = y_train.loc[minority_mask]
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.choice(len(minority_X), size=shortfall, replace=True)
    extra_X = minority_X.iloc[sample_indices]
    extra_y = minority_y.iloc[sample_indices]
    balanced_X = pd.concat([X_train, extra_X], axis=0, ignore_index=True)
    balanced_y = pd.concat([y_train, extra_y], axis=0, ignore_index=True)
    return balanced_X, balanced_y


def build_features(customers, orders, products=None):
    """Build the leakage-safe pre-cutoff feature snapshot for each eligible customer."""
    customer_source = customers.copy()

    # Validate required provenance columns.
    for col in ("signup_date_source_missing", "signup_date_source_malformed"):
        if col not in customer_source.columns:
            raise ValueError(
                f"customers dataset is missing signup-date provenance fields "
                f"(expected {col})."
            )

    orders_all = orders.copy()
    orders_all["order_ts"] = pd.to_datetime(orders_all["order_ts"], errors="coerce")

    # Only orders strictly before the prediction cutoff enter features.
    pre_cutoff = orders_all.loc[orders_all["order_ts"].lt(CUTOFF)].copy()

    # Determine delivery duration provenance availability.
    provenance_unverified = "delivery_minutes_source_valid" not in pre_cutoff.columns
    if provenance_unverified:
        order_duration_validity = pd.Series(False, index=pre_cutoff.index)
    else:
        order_duration_validity = _source_valid_flags(
            pre_cutoff["delivery_minutes_source_valid"]
        )

    completed_pre_cutoff = pre_cutoff.loc[pre_cutoff["status"].eq("completed")]

    eligible_ids = (
        completed_pre_cutoff["customer_id"].astype("string").str.strip().unique()
    )

    feature_rows = []
    for customer_id in eligible_ids:
        customer_completed = completed_pre_cutoff.loc[
            completed_pre_cutoff["customer_id"].astype("string").str.strip().eq(
                str(customer_id)
            )
        ]
        customer_all = pre_cutoff.loc[
            pre_cutoff["customer_id"].astype("string").str.strip().eq(
                str(customer_id)
            )
        ]

        last_completed = customer_completed["order_ts"].max()
        recency = int((CUTOFF - last_completed).days)
        frequency = int(len(customer_completed))
        monetary = float(customer_completed["order_amount"].sum())

        discount_values = pd.to_numeric(
            customer_completed["discount_pct"], errors="coerce"
        )
        average_discount_pct = float(discount_values.mean())

        total_final_orders = int(len(customer_all))
        cancelled_orders = int(customer_all["status"].eq("cancelled").sum())
        cancellation_rate = (
            cancelled_orders / total_final_orders if total_final_orders else 0.0
        )

        source_valid_for_customer = order_duration_validity.loc[
            customer_completed.index
        ]
        observed_duration = pd.to_numeric(
            customer_completed.loc[
                source_valid_for_customer, "delivery_minutes"
            ],
            errors="coerce",
        )
        observed_duration = observed_duration.loc[
            observed_duration.notna()
            & np.isfinite(observed_duration)
            & observed_duration.ge(0)
        ]
        observed_duration_count = int(len(observed_duration))
        unavailable_duration_count = frequency - observed_duration_count
        average_delivery_minutes = (
            float(observed_duration.mean())
            if observed_duration_count
            else np.nan
        )
        delivery_duration_missing = int(observed_duration_count == 0)

        customer_record = customer_source.loc[
            customer_source["customer_id"].eq(customer_id)
        ]
        if customer_record.empty:
            continue
        customer_record = customer_record.iloc[0]

        signup_date = pd.to_datetime(
            customer_record["signup_date"], errors="coerce"
        )
        signup_missing = bool(
            _source_valid_flags(
                pd.Series([customer_record["signup_date_source_missing"]])
            ).iloc[0]
        )
        signup_malformed = bool(
            _source_valid_flags(
                pd.Series([customer_record["signup_date_source_malformed"]])
            ).iloc[0]
        )
        if signup_missing and signup_malformed:
            raise ValueError(
                "Customer signup provenance flags are inconsistent "
                "(both missing and malformed are set)."
            )
        if signup_missing or signup_malformed or pd.isna(signup_date):
            tenure_days = np.nan
        elif signup_date > CUTOFF:
            raise ValueError(
                "Customer signup date is after the Part B prediction cutoff."
            )
        else:
            tenure_days = max(0, int((CUTOFF - signup_date).days))

        category_counts = (
            customer_completed["product_id"]
            .map(
                products.set_index("product_id")["category"]
                if products is not None
                else {}
            )
            .dropna()
            .value_counts()
        )
        favourite_category = (
            str(category_counts.index[0])
            if not category_counts.empty
            else "Unknown"
        )

        feature_rows.append(
            {
                "customer_id": customer_id,
                "recency": recency,
                "frequency": frequency,
                "monetary": monetary,
                "average_discount_pct": average_discount_pct,
                "cancellation_rate": cancellation_rate,
                "average_delivery_minutes": average_delivery_minutes,
                "delivery_duration_missing": delivery_duration_missing,
                "delivery_duration_provenance_unverified": int(provenance_unverified),
                "observed_delivery_duration_count": observed_duration_count,
                "unavailable_delivery_duration_count": unavailable_duration_count,
                "tenure_days": tenure_days,
                "signup_date_missing": int(signup_missing),
                "signup_date_malformed": int(signup_malformed),
                "city": customer_record["city"],
                "gender": customer_record["gender"],
                "favourite_category": favourite_category,
            }
        )

    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise ValueError("Part B feature generation produced an empty cohort.")

    # The label window is deliberately separate from pre-cutoff features.
    # Adding it here preserves the established build_features() contract;
    # calculate_lapsed_label() reads only completed orders in [CUTOFF, LABEL_END).
    completed_for_labels = orders_all.loc[
        orders_all["status"].astype("string").str.strip().str.lower().eq("completed")
    ].copy()
    features["lapsed"] = [
        calculate_lapsed_label(customer_id, completed_for_labels)
        for customer_id in features["customer_id"]
    ]
    return features


def attach_lapse_labels(features, orders):
    """Attach the September inactivity label without changing features."""
    completed_orders = orders.loc[
        orders["status"].eq("completed")
    ].copy()
    completed_orders["order_ts"] = pd.to_datetime(
        completed_orders["order_ts"], errors="coerce"
    )

    labels = []
    for customer_id in features["customer_id"]:
        labels.append(
            calculate_lapsed_label(customer_id, completed_orders)
        )

    result = features.copy()
    result["lapsed"] = labels
    return result


def build_feature_dataset(customers, orders, products=None):
    """Build a provenance-safe feature dataset with the September label."""
    features = build_features(customers, orders, products)
    return attach_lapse_labels(features, orders)


def build_training_artifact_metadata(
    run_id,
    feature_dataset,
    input_provenance,
    segments,
):
    provenance = build_run_provenance(
        run_id,
        feature_dataset,
        segments,
        input_provenance,
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": run_id,
        "provenance": provenance,
        "feature_columns": ALL_FEATURES,
        "target_column": "lapsed",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def save_training_artifact(path, frame, metadata, run_root):
    """Write a run-scoped CSV and its provenance sidecar without overwriting."""
    path = ensure_safe_output_path(path, run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    metadata_path = path.with_suffix(".metadata.json")
    metadata_path = ensure_safe_output_path(metadata_path, run_root)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path, metadata_path


def load_training_artifact(path, run_root, expected_run_id):
    """Load a feature artifact only when its sidecar and run identity agree."""
    path = ensure_safe_output_path(path, run_root, must_not_exist=False)
    metadata_path = ensure_safe_output_path(
        path.with_suffix(".metadata.json"), run_root, must_not_exist=False
    )
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("Required Part B training artifact or provenance is missing.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(path)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("Part B training artifact or metadata is malformed.") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or metadata.get("run_id") != expected_run_id
    ):
        raise ValueError("Part B training artifact belongs to a different run.")
    return frame, metadata


def build_model_search(model_name, X_train, y_train):
    """Tune a classifier using training-only stratified cross-validation."""
    pipeline = build_model_pipeline(model_name)
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    if model_name == "logistic_regression":
        parameter_grid = {
            "classifier__C": [0.1, 1.0, 10.0],
        }
    elif model_name == "random_forest":
        parameter_grid = {
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    else:
        raise ValueError(f"Unsupported lapse model: {model_name}")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def evaluate_classifier(model, X, y):
    """Return the binary metrics for the positive (lapsed) class."""
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_lapsed": float(
            precision_score(y, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall_lapsed": float(
            recall_score(y, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1_lapsed": float(
            f1_score(y, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["not_lapsed", "lapsed"],
            output_dict=True,
            zero_division=0,
        ),
    }

def save_json(path, data):
    """Create a run-scoped JSON artifact without overwriting existing evidence."""
    path = Path(path)
    run_root = path.parent.parent if path.parent.name == "lapse" else path.parent
    safe_path = ensure_safe_output_path(path, run_root)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        data, indent=2, ensure_ascii=False,
        default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist() if isinstance(value, np.ndarray) else str(value) if isinstance(value, Path) else (_ for _ in ()).throw(TypeError(f"Unsupported JSON value: {type(value).__name__}")),
    )
    with safe_path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return safe_path


def _run_training(run_dir, run_id, input_bundle):
    lapse_dir = run_dir / "lapse"
    ensure_safe_output_path(lapse_dir, run_dir)
    lapse_dir.mkdir(exist_ok=False)
    ensure_safe_output_path(lapse_dir / "lapse_cohort_features.csv", run_dir)

    orders = input_bundle["datasets"]["orders.csv"].copy()
    customers = input_bundle["datasets"]["customers.csv"].copy()
    products = input_bundle["datasets"]["products.csv"].copy()

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    orders["status"] = (
        orders["status"].astype("string").str.strip().str.lower()
    )

    for column in [
        "order_amount",
        "discount_pct",
        "delivery_minutes",
    ]:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")

    # Only final statuses are considered for the cancellation rate.
    orders = orders.loc[
        orders["status"].isin(["completed", "cancelled"])
    ].copy()

    # --------------------------------------------------------
    # TASK 4: COHORT, LABELS, AND FEATURES
    # --------------------------------------------------------
    dataset = build_features(customers, orders, products=products)

    feature_columns = ALL_FEATURES
    missing_columns = set(feature_columns) - set(dataset.columns)

    if missing_columns:
        raise ValueError(
            f"Missing model feature columns: {sorted(missing_columns)}"
        )

    # Numeric imputation is fitted only inside the training-fold pipelines.
    # The explicit Unknown category is deterministic and label-independent.
    for column in NUMERIC_FEATURES:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        dataset[column] = (
            dataset[column]
            .astype("string")
            .fillna("Unknown")
            .replace("", "Unknown")
        )

    if not set(dataset["lapsed"].unique()).issubset({0, 1}):
        raise AssertionError("Lapse label must be binary.")

    # Explicitly prevent use of segmentation outputs as model features.
    forbidden_features = {
        "segment_id",
        "segment_name",
        "is_premium",
        "lapsed",
        "customer_id",
    }

    if forbidden_features.intersection(feature_columns):
        raise AssertionError(
            "A forbidden identifier or post-hoc label is used as a feature."
        )

    validate_customer_ids(dataset)
    dataset_path = lapse_dir / "lapse_cohort_features.csv"
    ensure_safe_output_path(dataset_path, run_dir)
    dataset.to_csv(dataset_path, index=False)
    # Fingerprint and train on the exact serialized feature table finalization reads.
    dataset = pd.read_csv(dataset_path)
    validate_customer_ids(dataset)
    segments = input_bundle["segments"].copy()
    input_provenance = input_bundle["provenance"]
    provenance = build_run_provenance(
        run_id, dataset, segments, input_provenance
    )
    save_json(
        lapse_dir / "feature_dataset_metadata.json",
        {"provenance": provenance, "feature_file": dataset_path.name},
    )

    cohort_count = int(len(dataset))
    lapsed_count = int(dataset["lapsed"].sum())
    lapse_rate = float(dataset["lapsed"].mean())

    cohort_summary = {
        "cutoff": str(CUTOFF),
        "label_window_start": str(CUTOFF),
        "label_window_end_exclusive": str(LABEL_END),
        "cohort_count": cohort_count,
        "lapsed_count": lapsed_count,
        "not_lapsed_count": cohort_count - lapsed_count,
        "lapse_rate": lapse_rate,
        "features": feature_columns,
        "missing_feature_values_before_pipeline_imputation": int(
            dataset[feature_columns].isna().sum().sum()
        ),
    }

    save_json(
        lapse_dir / "cohort_summary.json",
        {**cohort_summary, "provenance": provenance},
    )

    print("\nCOHORT AND LABEL SUMMARY")
    print(json.dumps(cohort_summary, indent=2))

    if cohort_count != 1156 or lapsed_count != 443:
        raise AssertionError(
            "Cohort reference mismatch. Expected 1,156 customers and "
            "443 lapsed customers. Inspect lapse_cohort_features.csv."
        )

    # --------------------------------------------------------
    # TASK 5: STRATIFIED TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------------
    splits = stratified_splits(dataset)
    X_train = splits["X_train"]
    X_validation = splits["X_validation"]
    y_train = splits["y_train"]
    y_validation = splits["y_validation"]
    split_summary = summarize_splits(splits)
    split_summary["provenance"] = provenance

    save_json(lapse_dir / "split_summary.json", split_summary)

    print("\nTRAIN / VALIDATION / TEST SPLIT")
    print(json.dumps(split_summary, indent=2))

    if sum(split_summary[f"{name}_rows"] for name in ("train", "validation", "test")) != cohort_count:
        raise AssertionError("Split rows do not account for the full cohort.")

    # --------------------------------------------------------
    # TASK 5: VALIDATION BASELINE — ALWAYS PREDICT NOT LAPSED
    # --------------------------------------------------------
    baseline_predictions = np.zeros(len(y_validation), dtype=int)

    baseline = {
        "model": "Always predict not lapsed",
        "evaluation_split": "validation",
        "provenance": provenance,
        "accuracy": float(accuracy_score(y_validation, baseline_predictions)),
        "precision_lapsed": float(
            precision_score(
                y_validation,
                baseline_predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_lapsed": float(
            recall_score(
                y_validation,
                baseline_predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_lapsed": float(
            f1_score(
                y_validation,
                baseline_predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_validation, baseline_predictions, labels=[0, 1]
        ).tolist(),
    }

    save_json(lapse_dir / "baseline_metrics.json", baseline)

    print("\nBASELINE")
    print(json.dumps(baseline, indent=2))

    # --------------------------------------------------------
    # TASK 6: GRID SEARCH FOR LOGISTIC REGRESSION
    # --------------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    }

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    print("\nTUNING LOGISTIC REGRESSION...")
    logistic_search.fit(X_train, y_train)

    # --------------------------------------------------------
    # TASK 6: GRID SEARCH FOR RANDOM FOREST
    # --------------------------------------------------------
    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    forest_grid = {
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    }

    forest_search = GridSearchCV(
        estimator=forest_pipeline,
        param_grid=forest_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    print("\nTUNING RANDOM FOREST...")
    forest_search.fit(X_train, y_train)

    tuning_results = {
        "provenance": provenance,
        "tuning_split": "training_partition_cross_validation",
        "model_selection_split": "validation",
        "final_evaluation_split": "test",
        "logistic_regression": {
            "best_params": logistic_search.best_params_,
            "best_cv_f1": float(logistic_search.best_score_),
            "all_results": pd.DataFrame(logistic_search.cv_results_)[
                [
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "rank_test_score",
                ]
            ].to_dict(orient="records"),
        },
        "random_forest": {
            "best_params": forest_search.best_params_,
            "best_cv_f1": float(forest_search.best_score_),
            "all_results": pd.DataFrame(forest_search.cv_results_)[
                [
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "rank_test_score",
                ]
            ].to_dict(orient="records"),
        },
    }

    save_json(lapse_dir / "tuning_results.json", tuning_results)

    logistic_cv_results = pd.DataFrame(logistic_search.cv_results_)
    logistic_cv_results["score_role"] = "training_partition_cross_validation"
    logistic_cv_results["run_id"] = run_id
    logistic_cv_results["feature_dataset_fingerprint"] = provenance[
        "feature_dataset_fingerprint"
    ]
    ensure_safe_output_path(lapse_dir / "logistic_grid_results.csv", run_dir)
    logistic_cv_results.to_csv(
        lapse_dir / "logistic_grid_results.csv",
        index=False,
    )
    forest_cv_results = pd.DataFrame(forest_search.cv_results_)
    forest_cv_results["score_role"] = "training_partition_cross_validation"
    forest_cv_results["run_id"] = run_id
    forest_cv_results["feature_dataset_fingerprint"] = provenance[
        "feature_dataset_fingerprint"
    ]
    ensure_safe_output_path(lapse_dir / "random_forest_grid_results.csv", run_dir)
    forest_cv_results.to_csv(
        lapse_dir / "random_forest_grid_results.csv",
        index=False,
    )

    print("\nTUNING SUMMARY")
    print(
        "Logistic regression best parameters:",
        logistic_search.best_params_,
    )
    print(
        "Logistic regression best CV F1:",
        round(logistic_search.best_score_, 4),
    )
    print(
        "Random forest best parameters:",
        forest_search.best_params_,
    )
    print(
        "Random forest best CV F1:",
        round(forest_search.best_score_, 4),
    )

    # --------------------------------------------------------
    # TASK 6: MODEL SELECTION ON VALIDATION ONLY
    # --------------------------------------------------------
    logistic_metrics = evaluate_model(
        logistic_search.best_estimator_,
        X_validation,
        y_validation,
        "Tuned Logistic Regression",
    )

    forest_metrics = evaluate_model(
        forest_search.best_estimator_,
        X_validation,
        y_validation,
        "Tuned Random Forest",
    )

    validation_metrics = {
        "evaluation_split": "validation",
        "provenance": provenance,
        "baseline": baseline,
        "logistic_regression": logistic_metrics,
        "random_forest": forest_metrics,
    }
    save_json(lapse_dir / "validation_metrics.json", validation_metrics)

    validation_scores = {
        "logistic_regression": logistic_metrics,
        "random_forest": forest_metrics,
    }
    selected_name = select_model_by_validation_f1(validation_scores)
    selected_params = (
        logistic_search.best_params_
        if selected_name == "logistic_regression"
        else forest_search.best_params_
    )
    selected_metrics = validation_scores[selected_name]

    selected_summary = {
        "provenance": provenance,
        "selected_model": selected_name,
        "selected_params": selected_params,
        "selection_split": "validation",
        "candidate_models": ["logistic_regression", "random_forest"],
        "selection_basis": (
            "Highest lapsed-class F1 on the validation split; model "
            "hyperparameters were selected by cross-validation on training only."
        ),
        "selected_validation_f1": selected_metrics["f1_lapsed"],
        "required_minimum_validation_f1": 0.55,
        "meets_validation_f1_requirement": bool(
            selected_metrics["f1_lapsed"] >= 0.55
        ),
        "reasoning": (
            "Missing a lapsed customer can mean losing future revenue, "
            "while incorrectly targeting a loyal customer wastes retention "
            "resources. Compare recall and precision alongside F1; no "
            "operational threshold has been validated."
        ),
    }

    save_json(lapse_dir / "selected_model_summary.json", selected_summary)

    print("\nSELECTED MODEL")
    print(json.dumps(selected_summary, indent=2))

    # --------------------------------------------------------
    # TASK 7: TRAINING-ONLY RANDOM OVERSAMPLING EXPERIMENT
    # --------------------------------------------------------
    balanced_X_train, balanced_y_train = oversample_training_data(
        X_train, y_train
    )

    if balanced_y_train.value_counts().nunique() != 1:
        raise AssertionError(
            "Oversampling did not produce equal class counts."
        )

    balanced_logistic = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    C=logistic_search.best_params_["classifier__C"],
                ),
            ),
        ]
    )

    balanced_logistic.fit(balanced_X_train, balanced_y_train)

    balanced_metrics = evaluate_model(
        balanced_logistic,
        X_validation,
        y_validation,
        "Oversampled Logistic Regression",
    )

    # Compare the balanced logistic model with the original tuned
    # logistic model on validation rows; test remains untouched here.
    oversampling_comparison = {
        "provenance": provenance,
        "evaluation_split": "validation",
        "original_training_class_counts": {
            str(key): int(value)
            for key, value in y_train.value_counts().sort_index().items()
        },
        "balanced_training_class_counts": {
            str(key): int(value)
            for key, value in balanced_y_train.value_counts().sort_index().items()
        },
        "original_logistic_validation_metrics": logistic_metrics,
        "oversampled_logistic_validation_metrics": balanced_metrics,
        "recall_change": float(
            balanced_metrics["recall_lapsed"]
            - logistic_metrics["recall_lapsed"]
        ),
        "precision_change": float(
            balanced_metrics["precision_lapsed"]
            - logistic_metrics["precision_lapsed"]
        ),
        "f1_change": float(
            balanced_metrics["f1_lapsed"]
            - logistic_metrics["f1_lapsed"]
        ),
        "interpretation": (
            "Oversampling can increase recall by making the minority class "
            "more visible during training, but may also increase false "
            "positives. Compare precision and recall together before using "
            "the model for retention outreach."
        ),
    }

    save_json(
        lapse_dir / "oversampling_comparison.json",
        oversampling_comparison,
    )

    print("\nOVERSAMPLING EXPERIMENT")
    print(json.dumps(oversampling_comparison, indent=2))

    # Save a complete, reproducible record of the experiment.
    run_summary = {
        "provenance": provenance,
        "cohort": cohort_summary,
        "split": split_summary,
        "baseline": baseline,
        "logistic_regression_best_params": logistic_search.best_params_,
        "logistic_regression_best_cv_f1": float(
            logistic_search.best_score_
        ),
        "random_forest_best_params": forest_search.best_params_,
        "random_forest_best_cv_f1": float(forest_search.best_score_),
        "validation_metrics": validation_metrics,
        "selected_model": selected_summary,
        "oversampling": {
            "evaluation_split": "validation",
            "original_recall": logistic_metrics["recall_lapsed"],
            "oversampled_recall": balanced_metrics["recall_lapsed"],
            "original_precision": logistic_metrics["precision_lapsed"],
            "oversampled_precision": balanced_metrics["precision_lapsed"],
            "original_f1": logistic_metrics["f1_lapsed"],
            "oversampled_f1": balanced_metrics["f1_lapsed"],
        },
        "features": feature_columns,
        "forbidden_features_excluded": sorted(forbidden_features),
        "test_rows_used_for_tuning_or_selection": False,
        "test_rows_used_for_preprocessing_fit": False,
        "test_evaluation_performed": False,
        "random_state": RANDOM_STATE,
    }

    save_json(lapse_dir / "lapse_model_run_summary.json", run_summary)

    manifested_files = [
        dataset_path,
        lapse_dir / "feature_dataset_metadata.json",
        lapse_dir / "split_summary.json",
        lapse_dir / "tuning_results.json",
        lapse_dir / "selected_model_summary.json",
        lapse_dir / "validation_metrics.json",
    ]
    save_json(
        lapse_dir / "run_manifest.json",
        {
            "provenance": provenance,
            "feature_file": dataset_path.name,
            "artifact_sha256": {
                path.name: file_sha256(path) for path in manifested_files
            },
        },
    )

    manifest = json.loads(
        (lapse_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    artifact_provenances = [
            ("feature dataset metadata", json.loads(
                (lapse_dir / "feature_dataset_metadata.json").read_text(encoding="utf-8")
            )["provenance"]),
            ("split summary", json.loads(
                (lapse_dir / "split_summary.json").read_text(encoding="utf-8")
            )["provenance"]),
            ("tuning results", json.loads(
                (lapse_dir / "tuning_results.json").read_text(encoding="utf-8")
            )["provenance"]),
            ("selected model summary", json.loads(
                (lapse_dir / "selected_model_summary.json").read_text(encoding="utf-8")
            )["provenance"]),
            ("validation metrics", json.loads(
                (lapse_dir / "validation_metrics.json").read_text(encoding="utf-8")
            )["provenance"]),
        ]
    complete_training_run(
        run_dir,
        run_id,
        manifest,
        lapse_dir,
        artifact_provenances,
        dataset,
        segments,
        input_provenance,
    )

    print("\n" + "=" * 65)
    print("PART B TASKS 4–7 — LAPSE MODELING FINISHED")
    print("=" * 65)
    print(f"Cohort: {cohort_count}")
    print(f"Lapsed: {lapsed_count} ({lapse_rate:.2%})")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_validation)}")
    print(f"Test rows reserved: {len(splits['X_test'])}")
    print(f"Selected model: {selected_name}")
    print(f"Selected validation F1: {selected_metrics['f1_lapsed']:.4f}")
    print(
        "Meets minimum validation F1 >= 0.55: "
        f"{selected_summary['meets_validation_f1_requirement']}"
    )

    if not selected_summary["meets_validation_f1_requirement"]:
        print(
            "\nWARNING: The current model does not meet the required validation F1. "
            "Inspect the metrics and feature engineering before continuing "
            "to final model persistence."
        )


def main():
    input_bundle = load_part_b_inputs()
    run_dir = create_part_b_run_directory()
    run_id = uuid.uuid4().hex
    with acquire_run_lock(run_dir, run_id):
        write_run_state(
            run_dir,
            run_id,
            "TRAINING_IN_PROGRESS",
            initial=True,
            input_fingerprint=input_bundle["provenance"]["fingerprint"],
        )
        try:
            _run_training(run_dir, run_id, input_bundle)
        except BaseException:
            try:
                current = _read_run_state(run_dir)
                if current["state"] == "TRAINING_IN_PROGRESS":
                    write_run_state(
                        run_dir,
                        run_id,
                        "FAILED",
                        expected_state="TRAINING_IN_PROGRESS",
                    )
            except Exception:
                pass
            raise


if __name__ == "__main__":
    main()
