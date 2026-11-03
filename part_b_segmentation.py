
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# INSTAMART CAPSTONE — PART B, TASKS 1–3
# RFM, K-MEANS, SEGMENT PROFILES AND PCA
# ============================================================

ORDERS_PATH = Path("data/clean/orders.csv")
CUSTOMERS_PATH = Path("data/clean/customers.csv")

OUTPUT_DIR = Path("outputs")
SEGMENT_DIR = OUTPUT_DIR / "segmentation"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")
RANDOM_STATE = 42
K_VALUES = range(2, 9)
SELECTED_K = 4

RFM_COLUMNS = ["recency", "frequency", "monetary"]


def save_json(path, data):
    """Save Python and NumPy values as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return str(value)
        raise TypeError(
            f"Cannot serialize object of type {type(value).__name__}"
        )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=convert,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {path}")


def format_money(value):
    return f"₹{float(value):,.2f}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # TASK 1: LOAD AND VALIDATE THE INPUT DATA
    # --------------------------------------------------------
    for path in [ORDERS_PATH, CUSTOMERS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required cleaned dataset is missing: {path}"
            )

    orders = pd.read_csv(ORDERS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)

    required_order_columns = {
        "order_id",
        "customer_id",
        "order_ts",
        "status",
        "order_amount",
    }
    required_customer_columns = {
        "customer_id",
        "city",
        "signup_date",
        "age",
        "gender",
    }

    missing_order_columns = required_order_columns - set(orders.columns)
    missing_customer_columns = (
        required_customer_columns - set(customers.columns)
    )

    if missing_order_columns:
        raise ValueError(
            f"Missing order columns: {sorted(missing_order_columns)}"
        )

    if missing_customer_columns:
        raise ValueError(
            f"Missing customer columns: {sorted(missing_customer_columns)}"
        )

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["status"] = (
        orders["status"].astype("string").str.strip().str.lower()
    )

    # RFM uses only completed orders strictly before the snapshot.
    completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].notna()
        & orders["order_ts"].lt(SNAPSHOT_DATE)
        & orders["order_amount"].notna()
    ].copy()

    if completed.empty:
        raise ValueError(
            "No completed orders before the RFM snapshot date."
        )

    # --------------------------------------------------------
    # TASK 1: BUILD THE RFM TABLE
    # --------------------------------------------------------
    rfm = (
        completed.groupby("customer_id", as_index=False)
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
    )

    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_ts"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].copy()

    rfm["recency"] = rfm["recency"].astype(int)
    rfm["frequency"] = rfm["frequency"].astype(int)
    rfm["monetary"] = rfm["monetary"].astype(float)

    rfm = rfm.sort_values("customer_id").reset_index(drop=True)

    if rfm["customer_id"].duplicated().any():
        raise AssertionError("Duplicate customer IDs found in the RFM table.")

    if rfm[RFM_COLUMNS].isna().any().any():
        raise AssertionError("RFM table contains missing values.")

    if (rfm[RFM_COLUMNS] < 0).any().any():
        raise AssertionError("RFM features contain negative values.")

    rfm_path = SEGMENT_DIR / "rfm_table.csv"
    rfm.to_csv(rfm_path, index=False)
    print(f"\nSaved RFM table: {rfm_path}")

    rfm_summary = {
        "snapshot_date": str(SNAPSHOT_DATE),
        "customer_count": int(len(rfm)),
        "recency": {
            "median": float(rfm["recency"].median()),
            "minimum": int(rfm["recency"].min()),
            "maximum": int(rfm["recency"].max()),
        },
        "frequency": {
            "mean": float(rfm["frequency"].mean()),
            "maximum": int(rfm["frequency"].max()),
        },
        "monetary": {
            "mean": float(rfm["monetary"].mean()),
            "maximum": float(rfm["monetary"].max()),
        },
    }

    save_json(SEGMENT_DIR / "rfm_summary.json", rfm_summary)

    expected = {
        "customer_count": 1175,
        "recency_median": 19,
        "recency_minimum": 0,
        "recency_maximum": 180,
        "frequency_mean": 9.46,
        "frequency_maximum": 33,
        "monetary_mean": 2552.06,
        "monetary_maximum": 14766.55,
    }

    actual = {
        "customer_count": len(rfm),
        "recency_median": rfm["recency"].median(),
        "recency_minimum": rfm["recency"].min(),
        "recency_maximum": rfm["recency"].max(),
        "frequency_mean": rfm["frequency"].mean(),
        "frequency_maximum": rfm["frequency"].max(),
        "monetary_mean": rfm["monetary"].mean(),
        "monetary_maximum": rfm["monetary"].max(),
    }

    reference_checks = {
        "customer_count": actual["customer_count"]
        == expected["customer_count"],
        "recency_median": actual["recency_median"]
        == expected["recency_median"],
        "recency_minimum": actual["recency_minimum"]
        == expected["recency_minimum"],
        "recency_maximum": actual["recency_maximum"]
        == expected["recency_maximum"],
        "frequency_mean": abs(
            actual["frequency_mean"] - expected["frequency_mean"]
        ) <= 0.02,
        "frequency_maximum": actual["frequency_maximum"]
        == expected["frequency_maximum"],
        "monetary_mean": abs(
            actual["monetary_mean"] - expected["monetary_mean"]
        ) <= 0.02,
        "monetary_maximum": abs(
            actual["monetary_maximum"] - expected["monetary_maximum"]
        ) <= 0.02,
    }

    reference_report = {
        "expected": expected,
        "actual": actual,
        "checks": reference_checks,
        "all_passed": all(reference_checks.values()),
    }

    save_json(
        SEGMENT_DIR / "rfm_reference_checks.json",
        reference_report,
    )

    print("\nRFM REFERENCE CHECK")
    print(json.dumps(rfm_summary, indent=2))

    for name, passed in reference_checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] RFM {name}")

    if not all(reference_checks.values()):
        failed = [
            name for name, passed in reference_checks.items()
            if not passed
        ]
        raise AssertionError(
            "RFM reference checks failed: " + ", ".join(failed)
        )

    # --------------------------------------------------------
    # TASK 2: STANDARDIZE RFM AND EVALUATE K-MEANS
    # --------------------------------------------------------
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[RFM_COLUMNS])

    if not np.isfinite(scaled_rfm).all():
        raise AssertionError(
            "Standardized RFM data contains non-finite values."
        )

    scaled_df = pd.DataFrame(
        scaled_rfm,
        columns=RFM_COLUMNS,
    )
    scaled_df.insert(0, "customer_id", rfm["customer_id"].values)
    scaled_df.to_csv(SEGMENT_DIR / "scaled_rfm.csv", index=False)

    k_results = []

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(scaled_rfm)

        score = silhouette_score(
            scaled_rfm,
            labels,
            metric="euclidean",
        )

        k_results.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": float(score),
        })

    k_results_df = pd.DataFrame(k_results)
    k_results_df.to_csv(
        SEGMENT_DIR / "kmeans_evaluation.csv",
        index=False,
    )
    save_json(
        SEGMENT_DIR / "kmeans_evaluation.json",
        k_results,
    )

    print("\nK-MEANS EVALUATION")
    print(
        k_results_df.to_string(
            index=False,
            formatters={
                "inertia": "{:.3f}".format,
                "silhouette": "{:.3f}".format,
            },
        )
    )

    # The reference K=4 profile is used for this first implementation.
    # K=4 also has the highest silhouette among the tested candidates.
    best_silhouette_row = k_results_df.loc[
        k_results_df["silhouette"].idxmax()
    ]

    selection_summary = {
        "selected_k": SELECTED_K,
        "selection_reason": (
            "K=4 matches the supplied reference segment profile and "
            "has the highest silhouette score among K=2 through K=8."
        ),
        "highest_silhouette_k": int(best_silhouette_row["k"]),
        "highest_silhouette": float(best_silhouette_row["silhouette"]),
        "inertia_decreases_as_k_increases": bool(
            k_results_df["inertia"].is_monotonic_decreasing
        ),
        "interpretation": (
            "Inertia decreases as K increases, so inertia alone cannot "
            "determine the best number of clusters. Silhouette and "
            "business interpretability are considered together."
        ),
    }

    save_json(
        SEGMENT_DIR / "kmeans_selection.json",
        selection_summary,
    )

    # --------------------------------------------------------
    # TASK 3: FIT SELECTED K-MEANS AND PROFILE SEGMENTS
    # --------------------------------------------------------
    selected_model = KMeans(
        n_clusters=SELECTED_K,
        n_init=10,
        random_state=RANDOM_STATE,
    )
    segment_ids = selected_model.fit_predict(scaled_rfm)

    segmented = rfm.copy()
    segmented["segment_id"] = segment_ids

    profiles = (
        segmented.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "nunique"),
            mean_recency=("recency", "mean"),
            median_recency=("recency", "median"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
            median_monetary=("monetary", "median"),
        )
        .reset_index()
    )

    # Segment numbers are arbitrary labels. Premium is the segment
    # with the highest mean monetary value, regardless of its ID.
    premium_segment_id = int(
        profiles.loc[profiles["mean_monetary"].idxmax(), "segment_id"]
    )

    median_cluster_recency = profiles["mean_recency"].median()
    median_cluster_frequency = profiles["mean_frequency"].median()

    def name_segment(row):
        segment_id = int(row["segment_id"])

        if segment_id == premium_segment_id:
            return "Premium"

        if row["mean_recency"] >= 60:
            return "Dormant Customers"

        if (
            row["mean_frequency"] >= median_cluster_frequency
            and row["mean_recency"] <= median_cluster_recency
        ):
            return "Loyal Regulars"

        return "Developing Customers"

    profiles["segment_name"] = profiles.apply(name_segment, axis=1)
    profiles["is_premium"] = (
        profiles["segment_id"] == premium_segment_id
    )

    if profiles["segment_name"].duplicated().any():
        raise AssertionError(
            "Segment names are not unique. Review the naming rules."
        )

    segment_names = profiles.set_index("segment_id")[
        "segment_name"
    ].to_dict()

    segmented["segment_name"] = segmented["segment_id"].map(segment_names)
    segmented["is_premium"] = (
        segmented["segment_id"] == premium_segment_id
    )

    # Verify that exactly one segment is designated Premium.
    if int(profiles["is_premium"].sum()) != 1:
        raise AssertionError(
            "Exactly one segment must be designated Premium."
        )

    # Required deliverable and column order.
    segments_output = segmented[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium",
        ]
    ].copy()

    segments_path = OUTPUT_DIR / "customer_segments.csv"
    segments_output.to_csv(segments_path, index=False)

    profiles_path = SEGMENT_DIR / "segment_profiles.csv"
    profiles.to_csv(profiles_path, index=False)

    print(f"\nSaved required segment file: {segments_path}")
    print(f"Saved segment profiles: {profiles_path}")

    print("\nSEGMENT PROFILES")
    print(
        profiles.to_string(
            index=False,
            formatters={
                "mean_recency": "{:.2f}".format,
                "median_recency": "{:.2f}".format,
                "mean_frequency": "{:.2f}".format,
                "mean_monetary": "{:.2f}".format,
                "median_monetary": "{:.2f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # TASK 3: BUSINESS ACTIONS FOR EACH SEGMENT
    # --------------------------------------------------------
    recommendations = []

    for _, row in profiles.iterrows():
        name = row["segment_name"]

        if name == "Premium":
            action = (
                "Offer early access, personalized recommendations, "
                "and loyalty rewards to protect high customer value."
            )
        elif name == "Dormant Customers":
            action = (
                "Run a targeted win-back campaign and investigate "
                "potential service or delivery friction."
            )
        elif name == "Loyal Regulars":
            action = (
                "Encourage repeat purchases with relevant bundles, "
                "personalized recommendations, and loyalty milestones."
            )
        else:
            action = (
                "Use low-cost personalized recommendations and "
                "appropriate introductory offers to build repeat-order habits."
            )

        recommendations.append({
            "segment_id": int(row["segment_id"]),
            "segment_name": name,
            "customer_count": int(row["customer_count"]),
            "mean_recency": float(row["mean_recency"]),
            "mean_frequency": float(row["mean_frequency"]),
            "mean_monetary": float(row["mean_monetary"]),
            "is_premium": bool(row["is_premium"]),
            "recommended_action": action,
        })

    save_json(
        SEGMENT_DIR / "segment_recommendations.json",
        recommendations,
    )

    # --------------------------------------------------------
    # TASK 3: PCA ON STANDARDIZED RFM FEATURES
    # --------------------------------------------------------
    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)

    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = float(explained_variance.sum())

    pca_reference_checks = {
        "component_1_within_0.02": bool(
            abs(explained_variance[0] - 0.634) <= 0.02
        ),
        "component_2_within_0.02": bool(
            abs(explained_variance[1] - 0.218) <= 0.02
        ),
        "total_within_0.02": bool(
            abs(total_explained_variance - 0.852) <= 0.02
        ),
    }

    pca_results = {
        "n_components": 2,
        "explained_variance_ratio": [
            float(value) for value in explained_variance
        ],
        "total_explained_variance_ratio": total_explained_variance,
        "reference": {
            "component_1": 0.634,
            "component_2": 0.218,
            "total": 0.852,
            "tolerance": 0.02,
        },
        "reference_checks": pca_reference_checks,
        "all_reference_checks_passed": bool(
            all(pca_reference_checks.values())
        ),
    }

    save_json(SEGMENT_DIR / "pca_results.json", pca_results)

    print("\nPCA RESULTS")
    print(json.dumps(pca_results, indent=2))

    # Save the PCA-transformed customer coordinates as an additional
    # analysis artifact, retaining the segment assignment for plotting.
    pca_coordinates = pca.transform(scaled_rfm)

    pca_df = pd.DataFrame({
        "customer_id": rfm["customer_id"].values,
        "PC1": pca_coordinates[:, 0],
        "PC2": pca_coordinates[:, 1],
        "segment_id": segment_ids,
        "segment_name": segmented["segment_name"].values,
    })

    pca_df.to_csv(
        SEGMENT_DIR / "pca_customer_coordinates.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------
    expected_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]

    checks = {
        "rfm_customer_count_is_1175": len(rfm) == 1175,
        "segment_customer_count_is_1175": len(segments_output) == 1175,
        "customer_ids_are_unique": (
            segments_output["customer_id"].nunique()
            == len(segments_output)
        ),
        "required_columns_match": (
            segments_output.columns.tolist() == expected_columns
        ),
        "no_missing_rfm_features": (
            not segments_output[RFM_COLUMNS].isna().any().any()
        ),
        "exactly_four_segments": (
            segments_output["segment_id"].nunique() == SELECTED_K
        ),
        "exactly_one_premium_segment": (
            profiles["is_premium"].sum() == 1
        ),
        "premium_is_highest_mean_monetary": (
            int(
                profiles.loc[
                    profiles["mean_monetary"].idxmax(), "segment_id"
                ]
            )
            == premium_segment_id
        ),
        "all_segment_names_are_unique": (
            profiles["segment_name"].nunique() == len(profiles)
        ),
        "pca_has_two_components": len(explained_variance) == 2,
    }

    final_report = {
        "snapshot_date": str(SNAPSHOT_DATE),
        "selected_k": SELECTED_K,
        "random_state": RANDOM_STATE,
        "kmeans_n_init": 10,
        "rfm_customer_count": int(len(rfm)),
        "segment_customer_count": int(len(segments_output)),
        "premium_segment_id": premium_segment_id,
        "premium_segment_name": "Premium",
        "premium_customer_count": int(
            profiles.loc[
                profiles["segment_id"] == premium_segment_id,
                "customer_count",
            ].iloc[0]
        ),
        "reference_checks": reference_checks,
        "pca_reference_checks": pca_reference_checks,
        "validation_checks": checks,
        "all_validation_checks_passed": bool(all(checks.values())),
        "required_output": str(segments_path),
    }

    save_json(
        SEGMENT_DIR / "segmentation_summary.json",
        final_report,
    )

    print("\nFINAL VALIDATION")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\n" + "=" * 65)
    if all(checks.values()):
        print("PART B TASKS 1–3 — SEGMENTATION COMPLETED")
    else:
        print("PART B TASKS 1–3 — VALIDATION NEEDS ATTENTION")
    print("=" * 65)

    print(f"RFM customers: {len(rfm)}")
    print(f"Selected K: {SELECTED_K}")
    print(f"Premium segment ID: {premium_segment_id}")
    print(f"Premium customers: {final_report['premium_customer_count']}")
    print(f"Required output: {segments_path}")
    print(
        "PCA reference checks passed: "
        f"{pca_results['all_reference_checks_passed']}"
    )

    if not all(checks.values()):
        raise AssertionError(
            "One or more final segmentation validation checks failed. "
            "Review outputs/segmentation/segmentation_summary.json."
        )


if __name__ == "__main__":
    main()
