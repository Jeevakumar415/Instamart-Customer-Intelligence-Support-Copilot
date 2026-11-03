
from pathlib import Path
import json

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")

def normalize_text(series):
    return series.astype("string").str.strip().replace("", pd.NA)


def clean_customers():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        RAW_DIR / "customers.csv",
        dtype={"customer_id": "string"},
        keep_default_na=False,
    )
    input_rows = len(df)

    # Standardize city spellings.
    df["city"] = normalize_text(df["city"]).str.title()

    # Normalize all gender spellings to the three required values.
    gender = normalize_text(df["gender"]).str.lower()
    gender_map = {
        "male": "male",
        "m": "male",
        "man": "male",
        "female": "female",
        "f": "female",
        "woman": "female",
        "unknown": "unknown",
        "": "unknown",
    }
    df["gender"] = gender.map(gender_map).fillna("unknown")

    # Invalid ages become missing before city-level median imputation.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    age_source_valid = df["age"].notna() & df["age"].between(10, 100)
    invalid_age = df["age"].notna() & ~df["age"].between(10, 100)
    df["age_source_valid"] = age_source_valid.astype("int8")
    df.loc[invalid_age, "age"] = float("nan")

    city_medians = df.groupby("city", dropna=False)["age"].transform("median")
    overall_median = df["age"].median()
    df["age"] = df["age"].fillna(city_medians).fillna(overall_median)
    df["age"] = df["age"].round().astype("Int64")

    raw_signup_date = df["signup_date"].astype("string").str.strip()
    signup_date_missing = raw_signup_date.isna() | raw_signup_date.eq("")
    parsed_signup_date = pd.to_datetime(raw_signup_date, errors="coerce")
    signup_date_malformed = ~signup_date_missing & parsed_signup_date.isna()
    df["signup_date_source_missing"] = signup_date_missing.astype("int8")
    df["signup_date_source_malformed"] = signup_date_malformed.astype("int8")
    df["signup_date"] = parsed_signup_date

    df.to_csv(CLEAN_DIR / "customers.csv", index=False)

    return {
        "input_rows": input_rows,
        "output_rows": len(df),
        "city_counts": {
            str(k): int(v)
            for k, v in df["city"].value_counts().items()
        },
        "gender_counts": {
            str(k): int(v)
            for k, v in df["gender"].value_counts().items()
        },
        "missing_ages": int(df["age"].isna().sum()),
        "minimum_age": int(df["age"].min()),
        "maximum_age": int(df["age"].max()),
        "invalid_ages_replaced": int(invalid_age.sum()),
    }


def clean_products():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        RAW_DIR / "products.csv",
        dtype={"product_id": "string"},
    )

    for column in ["product_id", "product_name", "category"]:
        df[column] = normalize_text(df[column])

    df["category"] = df["category"].str.title()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["product_id"], keep="first")

    df.to_csv(CLEAN_DIR / "products.csv", index=False)
    return {"output_rows": len(df)}


def clean_orders():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        RAW_DIR / "orders.csv",
        dtype={
            "order_id": "string",
            "customer_id": "string",
            "product_id": "string",
        },
    )
    input_rows = len(df)

    # Rule 1: convert timestamps first.
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")

    # Rule 2: remove exact duplicate rows.
    exact_duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Normalize text and numeric columns.
    for column in ["order_id", "customer_id", "product_id", "status"]:
        df[column] = normalize_text(df[column])

    df["status"] = df["status"].str.lower()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(
        df["discount_pct"], errors="coerce"
    )
    df["order_amount"] = pd.to_numeric(
        df["order_amount"], errors="coerce"
    )
    df["delivery_minutes"] = pd.to_numeric(
        df["delivery_minutes"], errors="coerce"
    )

    # Rule 3: retain the latest record for each order_id.
    df = df.dropna(subset=["order_id", "order_ts"])
    df = df.sort_values(
        ["order_id", "order_ts"], kind="stable"
    )
    df = df.drop_duplicates(subset=["order_id"], keep="last").copy()

    # Preserve whether a valid non-negative duration existed before any
    # cleaning or imputation. D2 uses this marker as eligibility provenance.
    duration = df["delivery_minutes"]
    duration_source_valid = (
        duration.notna()
        & duration.ge(0)
        & ~duration.isin([float("inf"), float("-inf")])
    )
    df["delivery_minutes_source_valid"] = duration_source_valid.astype("int8")

    # Join product details for amount and delivery imputation.
    products = pd.read_csv(
        CLEAN_DIR / "products.csv",
        dtype={"product_id": "string"},
    )
    df = df.merge(
        products[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # Rule 4: replace missing/unparseable amounts using product price.
    calculated_amount = (
        df["price_inr"]
        * df["quantity"]
        * (1 - df["discount_pct"] / 100)
    ).round(2)

    missing_amount = df["order_amount"].isna()
    df.loc[missing_amount, "order_amount"] = calculated_amount[missing_amount]

    # Rule 5: negative delivery times become missing.
    invalid_delivery = ~np.isfinite(df["delivery_minutes"]) | df[
        "delivery_minutes"
    ].lt(0)
    df.loc[invalid_delivery, "delivery_minutes"] = float("nan")

    # Analytics-facing imputation uses the completed-order median for that
    # product category. The source-validity flag remains false for these
    # values so Part B can exclude them from observed-duration features.
    completed = df["status"].eq("completed")
    category_medians = df.loc[completed].groupby("category")[
        "delivery_minutes"
    ].median()

    completed_missing = completed & df["delivery_minutes"].isna()
    df.loc[completed_missing, "delivery_minutes"] = (
        df.loc[completed_missing, "category"].map(category_medians)
    )

    # Cancelled orders deliberately retain missing delivery times.
    completed_after = df["status"].eq("completed")
    cancelled_after = df["status"].eq("cancelled")

    # Persist timestamp in the requested format.
    df["order_ts"] = df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Internal join fields do not belong in the clean orders table.
    df = df.drop(columns=["price_inr", "category"])

    df.to_csv(CLEAN_DIR / "orders.csv", index=False)

    return {
        "input_rows": input_rows,
        "exact_duplicate_rows_removed": exact_duplicates_removed,
        "output_rows": len(df),
        "status_counts": {
            str(k): int(v)
            for k, v in df["status"].value_counts().items()
        },
        "missing_order_amounts": int(df["order_amount"].isna().sum()),
        "completed_missing_delivery": int(
            (completed_after & df["delivery_minutes"].isna()).sum()
        ),
        "cancelled_missing_delivery": int(
            (cancelled_after & df["delivery_minutes"].isna()).sum()
        ),
        "completed_revenue_inr": round(
            df.loc[completed_after, "order_amount"].sum(), 2
        ),
    }


def copy_evaluation_inputs():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}

    for filename in ["tickets.csv", "policy_eval.csv", "agent_scenarios.csv"]:
        source = RAW_DIR / filename
        if not source.exists():
            reports[filename] = {"status": "missing"}
            continue

        # Keep empty ticket strings as empty strings.
        if filename == "tickets.csv":
            df = pd.read_csv(source, keep_default_na=False)
        else:
            df = pd.read_csv(source)

        df.to_csv(CLEAN_DIR / filename, index=False)
        reports[filename] = {
            "status": "copied",
            "rows": len(df),
        }

    return reports


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = {
        "customers": clean_customers(),
        "products": clean_products(),
        "orders": clean_orders(),
        "other_inputs": copy_evaluation_inputs(),
    }

    report_path = OUTPUT_DIR / "cleaning_report.json"
    report_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    print(f"\nCleaning report saved to {report_path}")


if __name__ == "__main__":
    main()
