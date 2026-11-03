
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")
OUTPUT_DIR = Path("outputs/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(filename, data):
    path = OUTPUT_DIR / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Saved: {path}")


def main():
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    # Convert fields explicitly for reproducible analysis.
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    for column in [
        "order_amount",
        "quantity",
        "discount_pct",
        "delivery_minutes",
    ]:
        orders[column] = pd.to_numeric(
            orders[column], errors="coerce"
        )

    orders["status"] = orders["status"].astype("string").str.lower()
    completed = orders.loc[
        orders["status"].eq("completed")
    ].copy()

    print("=" * 65)
    print("INSTAMART PART A — EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    # 1. Dataset shape and data types.
    dataset_info = {}
    for name, df in [
        ("customers", customers),
        ("products", products),
        ("orders", orders),
    ]:
        dataset_info[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {
                col: str(dtype)
                for col, dtype in df.dtypes.items()
            },
        }

    save_json("dataset_info.json", dataset_info)

    # 2. Pandas versus SQLite aggregate cross-check.
    pandas_revenue = float(completed["order_amount"].sum())

    with sqlite3.connect(DB_PATH) as connection:
        sql_summary = pd.read_sql_query(
            """
            SELECT
                COUNT(*) AS completed_order_count,
                SUM(order_amount) AS completed_revenue,
                AVG(order_amount) AS average_order_amount
            FROM orders
            WHERE status = 'completed'
            """,
            connection,
        )

    sqlite_revenue = float(sql_summary.loc[0, "completed_revenue"])

    cross_check = {
        "pandas_completed_revenue": round(pandas_revenue, 2),
        "sqlite_completed_revenue": round(sqlite_revenue, 2),
        "difference": round(pandas_revenue - sqlite_revenue, 8),
        "match": bool(
            np.isclose(pandas_revenue, sqlite_revenue, atol=0.01)
        ),
        "sqlite_summary": sql_summary.to_dict(orient="records"),
    }

    save_json("pandas_sql_crosscheck.json", cross_check)

    print("\nPANDAS / SQLITE CROSS-CHECK")
    print(json.dumps(cross_check, indent=2))

    # 3. Pivot-table reconciliation: city x product category.
    orders_with_details = (
        completed.merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
            validate="many_to_one",
        )
    )

    pivot = pd.pivot_table(
        orders_with_details,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
    )

    pivot["Grand Total"] = pivot.sum(axis=1)
    pivot.loc["Grand Total"] = pivot.sum(axis=0)

    pivot_path = OUTPUT_DIR / "revenue_pivot_city_category.csv"
    pivot.to_csv(pivot_path)
    print(f"\nSaved pivot table: {pivot_path}")

    pivot_total = float(
        pivot.loc["Grand Total", "Grand Total"]
    )

    pivot_check = {
        "pivot_grand_total": round(pivot_total, 2),
        "expected_completed_revenue": round(pandas_revenue, 2),
        "match": bool(
            np.isclose(pivot_total, pandas_revenue, atol=0.01)
        ),
    }

    save_json("pivot_reconciliation.json", pivot_check)

    print("\nPIVOT RECONCILIATION")
    print(json.dumps(pivot_check, indent=2))

    # 4. Descriptive statistics for numeric fields.
    numeric_columns = [
        "quantity",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    ]

    descriptive = completed[numeric_columns].describe(
        percentiles=[0.25, 0.5, 0.75]
    )

    descriptive_path = OUTPUT_DIR / "descriptive_statistics.csv"
    descriptive.to_csv(descriptive_path)
    print(f"\nSaved descriptive statistics: {descriptive_path}")

    # 5. IQR outlier analysis.
    outlier_results = {}

    for column in ["order_amount", "delivery_minutes"]:
        values = completed[column].dropna()

        if values.empty:
            outlier_results[column] = {
                "count": 0,
                "outlier_count": 0,
                "lower_bound": None,
                "upper_bound": None,
            }
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (values < lower_bound) | (values > upper_bound)
        )

        outlier_results[column] = {
            "count": int(len(values)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_percent": round(
                float(outlier_mask.mean() * 100), 2
            ),
        }

    save_json("iqr_outlier_analysis.json", outlier_results)
    print("\nIQR OUTLIER ANALYSIS")
    print(json.dumps(outlier_results, indent=2))

    # 6. Correlation analysis.
    correlation_columns = [
        "quantity",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    ]

    correlation = completed[correlation_columns].corr(
        method="pearson"
    )

    correlation_path = OUTPUT_DIR / "correlation_matrix.csv"
    correlation.to_csv(correlation_path)
    print(f"\nSaved correlation matrix: {correlation_path}")

    # Record only finite correlations.
    correlation_summary = {}
    for col in correlation.columns:
        correlation_summary[col] = {
            row: (
                None if pd.isna(correlation.loc[row, col])
                else float(correlation.loc[row, col])
            )
            for row in correlation.index
        }

    save_json("correlation_matrix.json", correlation_summary)

    print("\nEDA COMPLETED")
    print(f"EDA outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
