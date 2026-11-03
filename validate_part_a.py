
from pathlib import Path
import json
import sqlite3
import pandas as pd

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"
SQL_RESULTS = ROOT / "outputs" / "sql_results"

checks = []


def check(name, condition, detail=""):
    passed = bool(condition)
    checks.append({
        "check": name,
        "passed": passed,
        "detail": str(detail),
    })
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE — PART A VALIDATION")
    print("=" * 65)

    # 1. Required source and cleaned files.
    required_files = [
        RAW / "customers.csv",
        RAW / "products.csv",
        RAW / "orders.csv",
        CLEAN / "customers.csv",
        CLEAN / "products.csv",
        CLEAN / "orders.csv",
        CLEAN / "tickets.csv",
        CLEAN / "policy_eval.csv",
        CLEAN / "agent_scenarios.csv",
        DB,
        ROOT / "sql" / "queries.sql",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    check("Required files exist", not missing, missing or "All present")

    if missing:
        print("\nCreate or restore the missing files before continuing.")
        return

    # 2. Raw dataset counts and audit checkpoints.
    raw_customers = pd.read_csv(
        RAW / "customers.csv", keep_default_na=False
    )
    raw_orders = pd.read_csv(
        RAW / "orders.csv", keep_default_na=False
    )

    raw_amount = pd.to_numeric(
        raw_orders["order_amount"].replace("", pd.NA),
        errors="coerce",
    )
    raw_delivery = pd.to_numeric(
        raw_orders["delivery_minutes"].replace("", pd.NA),
        errors="coerce",
    )
    raw_age = pd.to_numeric(
        raw_customers["age"].replace("", pd.NA),
        errors="coerce",
    )

    check("Raw customers = 1200", len(raw_customers) == 1200)
    check("Raw orders = 13084", len(raw_orders) == 13084)
    check(
        "Raw exact duplicate orders = 119",
        int(raw_orders.duplicated().sum()) == 119,
    )
    check(
        "Raw pending orders = 656",
        int(
            raw_orders["status"].str.strip().str.lower()
            .eq("pending").sum()
        ) == 656,
    )
    check(
        "Raw distinct order IDs = 12309",
        raw_orders["order_id"].nunique() == 12309,
    )
    check(
        "Raw nonnumeric or blank amounts = 120",
        int(raw_amount.isna().sum()) == 120,
    )
    check(
        "Raw negative delivery times = 106",
        int((raw_delivery < 0).sum()) == 106,
    )
    check(
        "Raw blank delivery times = 2541",
        int(raw_delivery.isna().sum()) == 2541,
    )
    check(
        "Raw blank ages = 61",
        int(raw_age.isna().sum()) == 61,
    )
    check(
        "Raw invalid ages = 6",
        int(
            (raw_age.notna() & ~raw_age.between(10, 100)).sum()
        ) == 6,
    )
    check(
        "Raw city spellings = 24",
        raw_customers["city"].nunique() == 24,
    )
    check(
        "Raw nonblank gender spellings = 10",
        raw_customers.loc[
            raw_customers["gender"].str.strip().ne(""), "gender"
        ].nunique() == 10,
    )
    check(
        "Raw blank genders = 30",
        int(raw_customers["gender"].str.strip().eq("").sum()) == 30,
    )

    # 3. Cleaned customer data.
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }

    actual_cities = {
        str(key): int(value)
        for key, value in customers["city"].value_counts().items()
    }

    check("Clean customers = 1200", len(customers) == 1200)
    check("Clean products = 40", len(products) == 40)
    check("Clean orders = 12309", len(orders) == 12309)
    check(
        "Clean city distribution matches reference",
        actual_cities == expected_cities,
        actual_cities,
    )
    check(
        "No missing cleaned customer ages",
        customers["age"].notna().all(),
    )
    check(
        "Gender counts match reference",
        customers["gender"].value_counts().to_dict()
        == {"female": 610, "male": 560, "unknown": 30},
        customers["gender"].value_counts().to_dict(),
    )
    check(
        "Completed orders = 11110",
        int(orders["status"].eq("completed").sum()) == 11110,
    )
    check(
        "Cancelled orders = 1199",
        int(orders["status"].eq("cancelled").sum()) == 1199,
    )
    check(
        "No missing order amounts",
        pd.to_numeric(
            orders["order_amount"], errors="coerce"
        ).notna().all(),
    )
    completed = orders["status"].eq("completed")
    check(
        "Completed orders have delivery times",
        pd.to_numeric(
            orders.loc[completed, "delivery_minutes"],
            errors="coerce",
        ).notna().all(),
    )
    cancelled = orders["status"].eq("cancelled")
    check(
        "Cancelled orders retain missing delivery times",
        pd.to_numeric(
            orders.loc[cancelled, "delivery_minutes"],
            errors="coerce",
        ).isna().all(),
    )

    revenue = pd.to_numeric(
        orders.loc[completed, "order_amount"],
        errors="coerce",
    ).sum()

    check(
        "Completed revenue = 2998668.80",
        round(float(revenue), 2) == 2998668.80,
        f"Actual: {revenue:.2f}",
    )

    # 4. Database contents and row counts.
    with sqlite3.connect(DB) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        check(
            "Database contains exactly three tables",
            tables == {"customers", "products", "orders"},
            sorted(tables),
        )

        for table, expected in {
            "customers": 1200,
            "products": 40,
            "orders": 12309,
        }.items():
            count = conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            check(
                f"Database {table} row count",
                count == expected,
                f"Actual: {count}; expected: {expected}",
            )

    # 5. SQL results and selected reference checkpoints.
    result_files = [
        SQL_RESULTS / f"q{i}_results.csv"
        for i in range(1, 8)
    ]
    missing_results = [
        str(path) for path in result_files if not path.exists()
    ]
    check(
        "All seven SQL result files exist",
        not missing_results,
        missing_results or "All present",
    )

    if not missing_results:
        expected_rows = {
            1: 30,
            2: 30,
            3: 11,
            4: 25,
            5: 550,
            6: 1175,
            7: 183,
        }

        for query_number, expected in expected_rows.items():
            result = pd.read_csv(
                SQL_RESULTS / f"q{query_number}_results.csv"
            )
            check(
                f"Q{query_number} result row count",
                len(result) == expected,
                f"Actual: {len(result)}; expected: {expected}",
            )

        q2 = pd.read_csv(SQL_RESULTS / "q2_results.csv")
        check(
            "Q2 total revenue",
            round(q2["revenue_inr"].sum(), 2) == 2998668.80,
        )

        q7 = pd.read_csv(SQL_RESULTS / "q7_results.csv")
        check(
            "Q7 final cumulative revenue",
            round(
                q7["cumulative_revenue_inr"].iloc[-1], 2
            ) == 2998668.80,
        )

    # 6. Save a machine-readable validation report.
    passed_count = sum(item["passed"] for item in checks)
    failed_count = len(checks) - passed_count

    report = {
        "total_checks": len(checks),
        "passed": passed_count,
        "failed": failed_count,
        "checks": checks,
    }

    report_path = ROOT / "outputs" / "part_a_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print(f"VALIDATION SUMMARY: {passed_count} passed, {failed_count} failed")
    print(f"Report saved to: {report_path}")

    if failed_count == 0:
        print("ALL CHECKPOINTS PASSED")
    else:
        print("VALIDATION NEEDS ATTENTION — review the FAIL entries above.")


if __name__ == "__main__":
    main()
