
from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd

from part_b_lapse_model import (
    ALL_FEATURES,
    FINALIZER_OUTPUT_NAMES,
    PROJECT_ROOT,
    RUN_STATE_SCHEMA_VERSION,
    SPLIT_CONFIGURATION,
    build_model_pipeline,
    evaluate_model,
    ensure_safe_output_path,
    file_sha256,
    load_part_b_inputs,
    acquire_run_lock,
    is_symlink_or_reparse_point,
    resolve_part_b_run_directory,
    read_run_state,
    require_run_state,
    select_model_by_validation_f1,
    summarize_splits,
    stratified_splits,
    validate_customer_ids,
    validate_lifecycle_run_id,
    validate_manifest_artifact_hashes,
    validate_provenance_bundle,
    write_run_state,
)


# ============================================================
# PART B — TASK 8
# MODEL PERSISTENCE, INTERPRETABILITY AND RETENTION LIST
# ============================================================

PROJECT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEGMENTS_PATH = PROJECT_OUTPUT_DIR / "customer_segments.csv"

FEATURES = ALL_FEATURES


def validate_selected_configuration(selection, tuning, validation_metrics):
    if tuning.get("tuning_split") != "training_partition_cross_validation":
        raise ValueError("Tuning results must come from training-only CV.")
    if validation_metrics.get("evaluation_split") != "validation":
        raise ValueError("Model selection metrics must be validation metrics.")
    if selection.get("selection_split") != "validation":
        raise ValueError("Selected model must be selected on validation data.")
    model_name = selection.get("selected_model")
    if model_name not in {"logistic_regression", "random_forest"}:
        raise ValueError("Selected model configuration is unsupported.")
    params = selection.get("selected_params")
    if not isinstance(params, dict):
        raise ValueError("Selected model parameters are missing or invalid.")
    if params != tuning.get(model_name, {}).get("best_params"):
        raise ValueError("Selected parameters do not match training-only tuning.")
    candidate_names = selection.get("candidate_models")
    expected_candidates = ["logistic_regression", "random_forest"]
    if candidate_names != expected_candidates:
        raise ValueError("Selected model candidate set is unsupported.")
    candidates = {name: validation_metrics.get(name) for name in candidate_names}
    selected_by_validation = select_model_by_validation_f1(candidates)
    if selected_by_validation != model_name:
        raise ValueError("Selected model is not the validation-F1 winner.")
    selected_score = validation_metrics[model_name]["f1_lapsed"]
    if selection.get("selected_validation_f1") != selected_score:
        raise ValueError("Selected validation F1 does not match its report.")
    return model_name, params


def fit_selected_model_on_development(splits, model_name, params):
    X_fit = pd.concat([splits["X_train"], splits["X_validation"]], axis=0)
    y_fit = pd.concat([splits["y_train"], splits["y_validation"]], axis=0)
    model = build_model_pipeline(model_name, params).fit(X_fit, y_fit)
    return model, X_fit, y_fit


def final_test_evaluation(model, X_test, y_test, model_name):
    """Evaluate the selected final model once on the held-out test partition."""
    return {
        "evaluation_split": "test",
        "metrics": evaluate_model(model, X_test, y_test, f"Final {model_name}"),
    }


def ensure_finalizer_outputs_absent(paths):
    collisions = [
        Path(path)
        for path in paths
        if Path(path).exists() or Path(path).is_symlink() or os.path.lexists(path)
    ]
    if collisions:
        raise FileExistsError(
            "Finalizer output already exists in this run directory; "
            "choose a fresh run directory."
        )


def finalizer_output_paths(run_root):
    run_root = Path(run_root)
    lapse_dir = run_root / "lapse"
    return [
        lapse_dir / "test_metrics.json",
        run_root / "lapse_model.joblib",
        run_root / "retention_list.csv",
        run_root / "finalizer_metadata.json",
        lapse_dir / "random_forest_feature_importances.csv",
        lapse_dir / "top_five_feature_importances.csv",
        lapse_dir / "logistic_regression_coefficients.csv",
        lapse_dir / "model_interpretability.json",
        lapse_dir / "task8_validation.json",
    ]


def output_file_record(path):
    path = Path(path)
    return {
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_output_name(path, root):
    current = Path(path)
    root = Path(root)
    parts = [current.name]
    current = current.parent
    while not os.path.samefile(current, root):
        parent = current.parent
        if parent == current:
            raise ValueError(
                "Output path is outside the configured Part B run directory."
            )
        parts.append(current.name)
        current = parent
    return Path(*reversed(parts)).as_posix()


def build_output_inventory(run_root, output_paths, *, exclude_names=()):
    root = Path(run_root).resolve(strict=True)
    inventory = {}
    for path in output_paths:
        path = ensure_safe_output_path(path, root, must_not_exist=False)
        relative = _relative_output_name(path, root)
        if relative in exclude_names:
            continue
        if not path.is_file():
            raise ValueError(f"Finalizer output is not a regular file: {Path(path).name}")
        inventory[relative] = output_file_record(path)
    return inventory


def preflight_finalizer_outputs(run_root):
    paths = finalizer_output_paths(run_root)
    for path in paths:
        ensure_safe_output_path(path, run_root)
    ensure_finalizer_outputs_absent(paths)
    return paths


def complete_finalization_run(run_root, run_id, output_paths, expected_metadata):
    root = Path(run_root).resolve(strict=True)
    normalized_output_paths = [
        ensure_safe_output_path(path, root, must_not_exist=False)
        for path in output_paths
    ]
    for output_path in normalized_output_paths:
        if not output_path.is_file():
            raise ValueError("A required finalizer output is missing or invalid.")
    relative_names = {
        _relative_output_name(path, root) for path in normalized_output_paths
    }
    if (
        len(relative_names) != len(normalized_output_paths)
        or relative_names != FINALIZER_OUTPUT_NAMES
    ):
        raise ValueError("Finalizer output list is incomplete or unsupported.")

    output_inventory = build_output_inventory(
        root, normalized_output_paths, exclude_names={"finalizer_metadata.json"}
    )
    metadata_path = root / "finalizer_metadata.json"
    metadata = _read_json_object(metadata_path)
    state = require_run_state(root, "FINALIZATION_IN_PROGRESS")
    if state["run_id"] != run_id:
        raise ValueError("Finalizer run ID does not match lifecycle state.")
    validate_lifecycle_run_id(state, metadata.get("provenance"))
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError("Finalizer metadata failed lifecycle validation.")

    if metadata.get("output_inventory") != output_inventory:
        raise ValueError("Finalizer output inventory or hash validation failed.")
    if metadata.get("output_inventory_self_hash_note") != (
        "finalizer_metadata.json is hashed in run_state.json to avoid a self-referential digest"
    ):
        raise ValueError("Finalizer metadata does not explain its self-hash limitation.")

    metrics = _read_json_object(root / "lapse" / "test_metrics.json")
    interpretation = _read_json_object(root / "lapse" / "model_interpretability.json")
    validation = _read_json_object(root / "lapse" / "task8_validation.json")
    if (
        metrics.get("provenance") != metadata.get("provenance")
        or metrics.get("evaluation_split") != "test"
        or metrics.get("selected_model") != metadata.get("selected_model")
        or interpretation.get("provenance") != metadata.get("provenance")
        or validation.get("provenance") != metadata.get("provenance")
        or not all(value is True for key, value in validation.items() if key != "provenance")
    ):
        raise ValueError("Finalizer JSON outputs are inconsistent with run provenance.")
    if not isinstance(metrics.get("metrics"), dict) or not metrics["metrics"]:
        raise ValueError("Final test metrics are malformed.")
    required_metric_fields = {
        "accuracy",
        "precision_lapsed",
        "recall_lapsed",
        "f1_lapsed",
        "confusion_matrix",
        "classification_report",
    }
    if not required_metric_fields.issubset(metrics["metrics"]):
        raise ValueError("Final test metrics have an incomplete schema.")
    for field in (
        "accuracy",
        "precision_lapsed",
        "recall_lapsed",
        "f1_lapsed",
    ):
        value = metrics["metrics"][field]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Final test metrics contain an invalid score.")
    if not isinstance(metrics["metrics"]["confusion_matrix"], list) or not isinstance(
        metrics["metrics"]["classification_report"], dict
    ):
        raise ValueError("Final test metrics contain malformed report fields.")
    if not isinstance(interpretation.get("top_five_random_forest_features"), list) or not isinstance(
        interpretation.get("three_logistic_regression_coefficients"), list
    ):
        raise ValueError("Model interpretation JSON has an invalid schema.")

    retention = _read_csv(root / "retention_list.csv")
    if (
        retention.columns.tolist()
        != ["customer_id", "lapse_probability", "actually_lapsed"]
        or len(retention) > 10
        or retention["customer_id"].duplicated().any()
        or not retention["lapse_probability"].between(0, 1).all()
        or not retention["lapse_probability"].is_monotonic_decreasing
        or not retention["actually_lapsed"].isin([0, 1]).all()
    ):
        raise ValueError("Retention list failed structural validation.")

    rf_importances = _read_csv(root / "lapse" / "random_forest_feature_importances.csv")
    top_features = _read_csv(root / "lapse" / "top_five_feature_importances.csv")
    coefficients = _read_csv(root / "lapse" / "logistic_regression_coefficients.csv")
    run_id_in_provenance = metadata.get("provenance", {}).get("run_id")
    fingerprint = metadata.get("provenance", {}).get("feature_dataset_fingerprint")
    for frame, required_columns in (
        (rf_importances, ["feature", "importance", "run_id", "feature_dataset_fingerprint"]),
        (top_features, ["feature", "importance", "run_id", "feature_dataset_fingerprint", "source_feature", "business_reading"]),
        (coefficients, ["feature", "coefficient", "absolute_coefficient", "run_id", "feature_dataset_fingerprint"]),
    ):
        if not set(required_columns).issubset(frame.columns):
            raise ValueError("An interpretability CSV has an invalid schema.")
        if not frame["run_id"].eq(run_id_in_provenance).all() or not frame[
            "feature_dataset_fingerprint"
        ].eq(fingerprint).all():
            raise ValueError("An interpretability CSV has mismatched provenance.")
    if not np.isfinite(rf_importances["importance"].to_numpy(dtype=float)).all() or not np.isfinite(
        top_features["importance"].to_numpy(dtype=float)
    ).all() or not np.isfinite(coefficients["coefficient"].to_numpy(dtype=float)).all():
        raise ValueError("An interpretability CSV contains invalid numeric values.")
    if not np.isclose(
        coefficients["absolute_coefficient"].to_numpy(dtype=float),
        coefficients["coefficient"].abs().to_numpy(dtype=float),
    ).all():
        raise ValueError("Logistic coefficient output is inconsistent.")

    try:
        model = joblib.load(root / "lapse_model.joblib")
    except Exception as exc:
        raise ValueError("Saved final model cannot be reloaded.") from exc
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ValueError("Saved final model does not expose the expected prediction API.")

    complete_inventory = build_output_inventory(root, normalized_output_paths)
    write_run_state(
        root,
        run_id,
        "FINALIZATION_COMPLETE",
        expected_state="FINALIZATION_IN_PROGRESS",
        finalization_integrity=complete_inventory,
    )


def _read_json_object(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Finalizer JSON output is malformed: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Finalizer JSON output is not an object: {Path(path).name}")
    return value


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Finalizer CSV output is malformed: {Path(path).name}") from exc


def load_verified_training_artifacts(lapse_dir, manifest):
    """Check all manifest hashes before parsing dependent training artifacts."""
    validate_manifest_artifact_hashes(manifest, lapse_dir)
    lapse_dir = Path(lapse_dir)
    dataset = pd.read_csv(lapse_dir / "lapse_cohort_features.csv")
    tuning = json.loads((lapse_dir / "tuning_results.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (lapse_dir / "selected_model_summary.json").read_text(encoding="utf-8")
    )
    validation_metrics = json.loads(
        (lapse_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    feature_metadata = json.loads(
        (lapse_dir / "feature_dataset_metadata.json").read_text(encoding="utf-8")
    )
    split_summary = json.loads(
        (lapse_dir / "split_summary.json").read_text(encoding="utf-8")
    )
    return dataset, tuning, selection, validation_metrics, feature_metadata, split_summary


def save_json(path, data, run_root):
    """Save JSON with NumPy scalar conversion."""
    path = ensure_safe_output_path(path, run_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported JSON type: {type(value).__name__}")

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=convert),
        encoding="utf-8",
    )
    print(f"Saved: {path}")


def _finalize(run_dir):
    run_dir = Path(run_dir).resolve(strict=True)
    run_state = require_run_state(run_dir, "TRAINING_COMPLETE")
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Configured Part B run directory does not exist: {run_dir}"
        )
    lapse_dir = run_dir / "lapse"
    if is_symlink_or_reparse_point(lapse_dir):
        raise ValueError("Symlinked lapse artifact directories are not allowed.")
    if not lapse_dir.is_dir():
        raise FileNotFoundError(
            f"Configured Part B run has no lapse artifacts: {lapse_dir}"
        )
    FEATURES_PATH = lapse_dir / "lapse_cohort_features.csv"
    FEATURE_METADATA_PATH = lapse_dir / "feature_dataset_metadata.json"
    RUN_MANIFEST_PATH = lapse_dir / "run_manifest.json"
    TUNING_PATH = lapse_dir / "tuning_results.json"
    TEST_METRICS_PATH = lapse_dir / "test_metrics.json"
    SELECTED_MODEL_PATH = lapse_dir / "selected_model_summary.json"
    VALIDATION_METRICS_PATH = lapse_dir / "validation_metrics.json"
    SPLIT_SUMMARY_PATH = lapse_dir / "split_summary.json"
    MODEL_PATH = run_dir / "lapse_model.joblib"
    RETENTION_PATH = run_dir / "retention_list.csv"
    FINALIZER_METADATA_PATH = run_dir / "finalizer_metadata.json"
    finalizer_outputs = preflight_finalizer_outputs(run_dir)

    for path in [
        FEATURES_PATH,
        FEATURE_METADATA_PATH,
        RUN_MANIFEST_PATH,
        TUNING_PATH,
        SELECTED_MODEL_PATH,
        VALIDATION_METRICS_PATH,
        SPLIT_SUMMARY_PATH,
    ]:
        ensure_safe_output_path(path, run_dir, must_not_exist=False)
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    (
        dataset,
        tuning,
        selection,
        validation_metrics,
        feature_metadata,
        split_summary,
    ) = load_verified_training_artifacts(lapse_dir, run_manifest)
    # Load the same input root that was used during training to enable
    # provenance re-verification at finalization time.
    training_input_provenance = run_manifest.get("provenance", {}).get(
        "input_provenance", {}
    )
    training_input_root = training_input_provenance.get("input_root")
    input_bundle = load_part_b_inputs(
        training_input_root, segments_path=SEGMENTS_PATH
    )
    segments = input_bundle["segments"]

    required = set(FEATURES + ["customer_id", "lapsed"])
    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(f"Lapse dataset missing columns: {sorted(missing)}")
    validate_customer_ids(dataset)

    if run_manifest.get("feature_file") != FEATURES_PATH.name:
        raise ValueError("Run manifest points to an incompatible feature file.")
    if feature_metadata.get("feature_file") != FEATURES_PATH.name:
        raise ValueError("Feature metadata points to an incompatible feature file.")
    provenance = run_manifest.get("provenance")
    validate_lifecycle_run_id(run_state, provenance)
    validate_provenance_bundle(
        provenance,
        [
            ("feature dataset metadata", feature_metadata.get("provenance")),
            ("split summary", split_summary.get("provenance")),
            ("tuning results", tuning.get("provenance")),
            ("selected model summary", selection.get("provenance")),
            ("validation metrics", validation_metrics.get("provenance")),
        ],
        dataset,
        segments,
        input_bundle["provenance"],
    )
    if split_summary.get("random_state") != SPLIT_CONFIGURATION["random_state"]:
        raise ValueError("Split summary has an incompatible random state.")
    if tuning.get("tuning_split") != "training_partition_cross_validation":
        raise ValueError("Tuning results are not marked as training-only CV.")
    if tuning.get("model_selection_split") != "validation":
        raise ValueError("Tuning results have an incompatible selection split.")
    if tuning.get("final_evaluation_split") != "test":
        raise ValueError("Tuning results have an incompatible test split role.")

    forbidden = {
        "customer_id",
        "lapsed",
        "segment_id",
        "segment_name",
        "is_premium",
    }

    if forbidden.intersection(FEATURES):
        raise AssertionError("Forbidden fields detected in model features.")

    selected_name, selected_params = validate_selected_configuration(
        selection, tuning, validation_metrics
    )

    splits = stratified_splits(dataset)
    recomputed_split_summary = summarize_splits(splits)
    for key, value in recomputed_split_summary.items():
        if split_summary.get(key) != value:
            raise ValueError(f"Saved split summary does not match current split: {key}.")

    write_run_state(
        run_dir,
        run_state["run_id"],
        "FINALIZATION_IN_PROGRESS",
        expected_state="TRAINING_COMPLETE",
    )
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    ids_train = splits["ids_train"]
    ids_validation = splits["ids_validation"]

    # Parameters came from training-only CV; choose no configuration using test data.
    selected_model, X_fit, y_fit = fit_selected_model_on_development(
        splits, selected_name, selected_params
    )

    # Persist first, then perform one final test evaluation through the reloaded model.
    ensure_safe_output_path(MODEL_PATH, run_dir)
    joblib.dump(selected_model, MODEL_PATH)
    reloaded_model = joblib.load(MODEL_PATH)
    final_evaluation = final_test_evaluation(
        reloaded_model,
        X_test,
        y_test,
        selected_name,
    )
    save_json(
        TEST_METRICS_PATH,
        {
            "provenance": provenance,
            "evaluation_split": final_evaluation["evaluation_split"],
            "purpose": "one final evaluation of the validation-selected model",
            "selected_model": selected_name,
            "metrics": final_evaluation["metrics"],
        },
        run_dir,
    )

    print("\nMODEL PERSISTENCE")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Selected model: {selected_name}")
    print(f"Final test F1: {final_evaluation['metrics']['f1_lapsed']:.4f}")

    # --------------------------------------------------------
    # TOP FIVE RANDOM FOREST FEATURE IMPORTANCES
    # --------------------------------------------------------
    if selected_name == "random_forest":
        forest_model = reloaded_model
    else:
        forest_model = build_model_pipeline(
            "random_forest", tuning["random_forest"]["best_params"]
        ).fit(X_fit, y_fit)
    preprocessor = forest_model.named_steps["preprocessor"]
    classifier = forest_model.named_steps["classifier"]

    transformed_feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    importance_df["run_id"] = provenance["run_id"]
    importance_df["feature_dataset_fingerprint"] = provenance[
        "feature_dataset_fingerprint"
    ]

    ensure_safe_output_path(
        lapse_dir / "random_forest_feature_importances.csv", run_dir
    )
    importance_df.to_csv(
        lapse_dir / "random_forest_feature_importances.csv",
        index=False,
    )

    top_five = importance_df.head(5).copy()

    business_readings = {
        "recency": (
            "More days since the last completed order may indicate "
            "weaker recent engagement."
        ),
        "frequency": (
            "Higher historical order frequency may indicate a stronger "
            "repeat-purchase habit."
        ),
        "monetary": (
            "Historical spend indicates the customer's previous value."
        ),
        "average_discount_pct": (
            "Discount dependence may affect response to retention offers."
        ),
        "cancellation_rate": (
            "Frequent cancellations may signal ordering friction."
        ),
        "average_delivery_minutes": (
            "Longer delivery times may indicate service friction."
        ),
        "tenure_days": (
            "Longer customer tenure may reflect a more established relationship."
        ),
        "city": (
            "Location can capture geographic differences in service or demand."
        ),
        "gender": (
            "A demographic association, not evidence of a causal effect."
        ),
        "favourite_category": (
            "The preferred product category may reflect different buying habits."
        ),
    }

    def base_feature(feature_name):
        for feature in FEATURES:
            if feature_name == f"numeric__{feature}":
                return feature
            if feature_name.startswith(f"categorical__{feature}_"):
                return feature
        return feature_name

    top_five["source_feature"] = top_five["feature"].map(base_feature)
    top_five["business_reading"] = top_five["source_feature"].map(
        business_readings
    ).fillna("Interpret as an association, not a causal effect.")

    ensure_safe_output_path(lapse_dir / "top_five_feature_importances.csv", run_dir)
    top_five.to_csv(
        lapse_dir / "top_five_feature_importances.csv",
        index=False,
    )

    print("\nTOP FIVE RANDOM FOREST FEATURES")
    print(top_five.to_string(index=False))

    # --------------------------------------------------------
    # LOGISTIC REGRESSION COEFFICIENT SIGNS
    # --------------------------------------------------------
    if selected_name == "logistic_regression":
        logistic_model = reloaded_model
    else:
        logistic_model = build_model_pipeline(
            "logistic_regression", tuning["logistic_regression"]["best_params"]
        ).fit(X_fit, y_fit)

    logistic_preprocessor = logistic_model.named_steps["preprocessor"]
    logistic_classifier = logistic_model.named_steps["classifier"]

    logistic_names = logistic_preprocessor.get_feature_names_out()
    coefficients = logistic_classifier.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": logistic_names,
            "coefficient": coefficients,
        }
    )
    coefficient_df["absolute_coefficient"] = (
        coefficient_df["coefficient"].abs()
    )
    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient", ascending=False
    )
    coefficient_df["run_id"] = provenance["run_id"]
    coefficient_df["feature_dataset_fingerprint"] = provenance[
        "feature_dataset_fingerprint"
    ]

    ensure_safe_output_path(
        lapse_dir / "logistic_regression_coefficients.csv", run_dir
    )
    coefficient_df.to_csv(
        lapse_dir / "logistic_regression_coefficients.csv",
        index=False,
    )

    coefficient_df["source_feature"] = coefficient_df["feature"].map(
        base_feature
    )

    # Select three distinct original features for readable reporting.
    selected_coefficients = []
    used_features = set()

    for _, row in coefficient_df.iterrows():
        source = row["source_feature"]

        if source in used_features:
            continue

        used_features.add(source)

        coefficient = float(row["coefficient"])

        if coefficient > 0:
            interpretation = (
                "Higher encoded/scaled values are associated with higher "
                "predicted odds of lapse, holding other features constant."
            )
        elif coefficient < 0:
            interpretation = (
                "Higher encoded/scaled values are associated with lower "
                "predicted odds of lapse, holding other features constant."
            )
        else:
            interpretation = "The fitted coefficient is zero."

        selected_coefficients.append(
            {
                "feature": row["feature"],
                "source_feature": source,
                "coefficient": coefficient,
                "sign": "positive" if coefficient > 0 else (
                    "negative" if coefficient < 0 else "zero"
                ),
                "interpretation": interpretation,
            }
        )

        if len(selected_coefficients) == 3:
            break

    save_json(
        lapse_dir / "model_interpretability.json",
        {
            "provenance": provenance,
            "top_five_random_forest_features": top_five.to_dict(
                orient="records"
            ),
            "three_logistic_regression_coefficients": selected_coefficients,
            "note": (
                "Feature importance and coefficient signs describe model "
                "associations. They do not establish causation. Numeric "
                "coefficients apply to standardized numeric features; "
                "categorical coefficients apply to one-hot encoded contrasts."
            ),
        },
        run_dir,
    )

    print("\nTHREE LOGISTIC REGRESSION COEFFICIENTS")
    print(json.dumps(selected_coefficients, indent=2))

    # --------------------------------------------------------
    # RETENTION LIST: TRAINING + VALIDATION EXAMPLES ONLY
    # --------------------------------------------------------
    # Premium is a post-hoc whole-window label from the RFM segmentation.
    # It is used only to filter the retention list, never as a model feature.
    if "is_premium" not in segments.columns:
        raise ValueError(
            "customer_segments.csv must contain the is_premium column."
        )

    premium_ids = set(
        segments.loc[
            segments["is_premium"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            ),
            "customer_id",
        ].astype(str)
    )

    fit_customer_ids = pd.concat([ids_train, ids_validation]).astype(str)

    retention_frame = pd.DataFrame(
        {
            "customer_id": fit_customer_ids.to_numpy(),
            "actual_lapsed": y_fit.to_numpy(),
            "lapse_probability": reloaded_model.predict_proba(X_fit)[:, 1],
        }
    )

    retention = retention_frame.loc[
        retention_frame["customer_id"].isin(premium_ids)
    ].copy()

    retention = retention.sort_values(
        "lapse_probability", ascending=False
    ).reset_index(drop=True)

    # Required columns and ordering.
    retention["lapse_probability"] = retention[
        "lapse_probability"
    ].astype(float)
    retention["actual_lapsed"] = retention["actual_lapsed"].astype(int)

    retention = retention.rename(
        columns={"actual_lapsed": "actually_lapsed"}
    )

    retention = retention[
        [
            "customer_id",
            "lapse_probability",
            "actually_lapsed",
        ]
    ]

    # The institute requests top 10, or all if fewer than 10 exist.
    retention = retention.head(10)

    ensure_safe_output_path(RETENTION_PATH, run_dir)
    retention.to_csv(RETENTION_PATH, index=False)

    true_lapsed_count = int(retention["actually_lapsed"].sum())

    print("\nRETENTION LIST")
    print(f"Saved: {RETENTION_PATH}")
    print(f"Premium train/validation examples included: {len(retention)}")
    print(f"Lapse labels among listed examples: {true_lapsed_count}")
    print(retention.to_string(index=False))

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------
    validation = {
        "provenance": provenance,
        "model_file_exists": MODEL_PATH.exists(),
        "retention_file_exists": RETENTION_PATH.exists(),
        "selected_model_config_consumed": selected_name in tuning,
        "final_test_evaluation_saved": (
            json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8")).get(
                "evaluation_split"
            ) == "test"
        ),
        "retention_columns_correct": retention.columns.tolist() == [
            "customer_id",
            "lapse_probability",
            "actually_lapsed",
        ],
        "retention_rows_at_most_10": len(retention) <= 10,
        "retention_sorted_descending": retention[
            "lapse_probability"
        ].is_monotonic_decreasing,
        "retention_only_premium_train_validation_examples": set(
            retention["customer_id"]
        ).issubset(premium_ids & set(fit_customer_ids)),
    }

    save_json(
        lapse_dir / "task8_validation.json",
        validation,
        run_dir,
    )

    finalizer_metadata = {
        "provenance": provenance,
        "selected_model": selected_name,
        "selected_params": selected_params,
        "model_file": MODEL_PATH.name,
        "retention_file": RETENTION_PATH.name,
        "test_metrics_file": TEST_METRICS_PATH.name,
        "output_inventory": build_output_inventory(
            run_dir,
            finalizer_outputs,
            exclude_names={"finalizer_metadata.json"},
        ),
        "output_inventory_self_hash_note": (
            "finalizer_metadata.json is hashed in run_state.json to avoid a self-referential digest"
        ),
    }
    save_json(FINALIZER_METADATA_PATH, finalizer_metadata, run_dir)

    print("\nTASK 8 VALIDATION")
    for name, passed in validation.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(validation.values()):
        raise AssertionError(
            "Task 8 validation failed. Inspect "
            "outputs/lapse/task8_validation.json."
        )

    complete_finalization_run(
        run_dir,
        run_state["run_id"],
        finalizer_outputs,
        finalizer_metadata,
    )

    print("\n" + "=" * 65)
    print("PART B TASK 8 — MODEL PERSISTENCE COMPLETED")
    print("=" * 65)
    print(f"Model: {MODEL_PATH}")
    print(f"Retention list: {RETENTION_PATH}")
    print("All Task 8 validation checks passed.")


def main():
    run_dir = resolve_part_b_run_directory()
    state = read_run_state(run_dir)
    with acquire_run_lock(run_dir, state["run_id"]):
        state = require_run_state(run_dir, "TRAINING_COMPLETE")
        if (
            state.get("schema_version") != RUN_STATE_SCHEMA_VERSION
            or not state.get("input_fingerprint")
        ):
            raise ValueError(
                "This run lacks explicit input provenance and cannot be finalized."
            )
        try:
            _finalize(run_dir)
        except BaseException:
            try:
                write_run_state(
                    run_dir,
                    state["run_id"],
                    "FAILED",
                    expected_state="FINALIZATION_IN_PROGRESS",
                )
            except Exception:
                pass
            raise


if __name__ == "__main__":
    main()
